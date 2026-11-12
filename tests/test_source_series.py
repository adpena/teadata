from base64 import b64decode
import csv
from io import BytesIO, StringIO
import json

import pytest

from teadata.dataset_metadata import DataPeriod
from teadata.manifest import ContractError
from teadata.release import DataRelease

from test_catalog_projections import make_edition_fixture


def releases(tmp_path, *, normalized=True, unique_entity_key=False):
    children = {}
    for year, amount in ((2023, 6), (2024, 7), (2025, 8)):
        directory = tmp_path / str(year)
        directory.mkdir()
        fixture = make_edition_fixture(
            directory, families=("student",), normalized=normalized or year != 2025,
            amount_period=DataPeriod("school_year", year - 1, year).to_dict(),
            entity_grain=True, observed_amount=amount, unique_entity_key=unique_entity_key,
        )
        children[year] = DataRelease(fixture["root"], offline=True)
    return children


def use_children(parent, children, monkeypatch):
    original = DataRelease._observation_year_options
    calls = []

    def options(dataset, fields, scope, edition):
        calls.append(edition)
        for child in children.values():
            yield from original(child, dataset, fields, scope, edition)

    monkeypatch.setattr(parent, "_observation_year_options", options)
    return calls


def excel_export(series, target):
    from openpyxl import load_workbook

    series.to_excel(target)
    book = load_workbook(target, read_only=True, data_only=True)
    try:
        assert book.sheetnames == ["Data", "Provenance"]
        data, provenance = list(book["Data"].values), list(book["Provenance"].values)
        chunks = [row[3] for row in provenance
                  if len(row) >= 4 and row[0] == "Series selection record" and type(row[1]) is int]
        assert json.loads(b64decode("".join(chunks))) == series.provenance()
        cells = {str(cell) for row in provenance for cell in row if cell is not None}
        for source in series.provenance()["sources"]:
            assert source["provenance"]["release"]["manifest_sha256"] in cells
            assert source["provenance"]["citation"] in cells
        # Write-only XLSX omits trailing blank cells; its header fixes the width.
        width = len(data[0])
        return [row + (None,) * (width - len(row)) for row in data]
    finally:
        book.close()


def test_series_csv_and_excel_stream_raw_values_statuses_and_selection(tmp_path, monkeypatch):
    from teadata.source_series import ObservationSeries

    children = releases(tmp_path)
    parent = children[2023]
    use_children(parent, children, monkeypatch)
    series = parent.series("campus_student", "amount", scope="campus", years=[25, 23, 24], ids=[2, 4])
    expected = series.to_arrow().to_pylist()
    monkeypatch.setattr(ObservationSeries, "to_arrow", lambda *_: pytest.fail("Export collected the series"))
    monkeypatch.setattr(ObservationSeries, "to_df", lambda *_: pytest.fail("Export collected a DataFrame"))
    stream = BytesIO()
    series.to_csv(stream)
    assert not stream.closed
    rows = list(csv.DictReader(StringIO(stream.getvalue().decode())))
    assert rows == [{key: "" if value is None else str(value) for key, value in row.items()} for row in expected]
    data = excel_export(series, tmp_path / "series.xlsx")
    header, *rows = data
    assert list(header) == series.schema.names
    assert rows == [tuple(row[name] for name in header) for row in expected]


def test_series_empty_exports_and_failed_paths_are_atomic(tmp_path, monkeypatch):
    from teadata.source_series import ObservationSeries

    children = releases(tmp_path)
    parent = children[2023]
    use_children(parent, children, monkeypatch)
    empty = parent.series("campus_student", "amount", scope="campus", years=[23, 24], ids=[999])
    stream = BytesIO()
    empty.to_csv(stream)
    assert list(csv.reader(StringIO(stream.getvalue().decode()))) == [empty.schema.names]
    excel = BytesIO()
    assert excel_export(empty, excel) == [tuple(empty.schema.names)]
    assert not excel.closed

    series = parent.series("campus_student", "amount", scope="campus", years=[23], ids=[4])
    original = ObservationSeries.iter_batches

    def fail_after_first(self):
        yield next(original(self))
        raise RuntimeError("interrupted source")

    monkeypatch.setattr(ObservationSeries, "iter_batches", fail_after_first)
    for method, extension in (("to_csv", "csv"), ("to_excel", "xlsx")):
        path = tmp_path / f"failed.{extension}"
        with pytest.raises(RuntimeError, match="interrupted source"):
            getattr(series, method)(path)
        assert not path.exists()
        path.write_bytes(b"existing result")
        with pytest.raises(FileExistsError):
            getattr(series, method)(path)
        assert path.read_bytes() == b"existing result"
    assert not list(tmp_path.glob(".teadata-export-*"))


def test_series_plans_once_and_streams_selected_years_with_provenance(tmp_path, monkeypatch):
    children = releases(tmp_path)
    parent = children[2023]
    discoveries = use_children(parent, children, monkeypatch)
    real_artifact = DataRelease.artifact
    reads = []

    def artifact(release, role, **kwargs):
        reads.append(release.manifest_sha256)
        return real_artifact(release, role, **kwargs)

    monkeypatch.setattr(DataRelease, "artifact", artifact)
    ids = [4, "'000000002", "000000999"]
    years = ["2024–25", 23, 2024]
    series = parent.series("campus_student", "aMoUnT", scope="campus", years=years, ids=ids)
    ids.clear()
    years.clear()
    assert discoveries == [None]
    assert series.years == (2025, 2023, 2024)
    assert series.for_year("25").release is children[2025]
    with pytest.raises(ContractError, match="outside this series"):
        series.for_year(22)
    provenance = series.provenance()
    assert provenance["fields"] == ["amount"]
    assert provenance["ids"] == ["000000004", "000000002", "000000999"]
    assert [item["provenance"]["release"]["manifest_sha256"] for item in provenance["sources"]] == [
        children[year].manifest_sha256 for year in series.years]
    assert not reads
    batches = series.iter_batches()
    assert next(batches)["measurement_school_year"].to_pylist() == [2025, 2025]
    assert reads == [children[2025].manifest_sha256]
    batches.close()
    table = series.to_arrow()
    assert table["measurement_school_year"].to_pylist() == [2025, 2025, 2023, 2023, 2024, 2024]
    assert table["amount"].to_pylist() == [None, 8, None, 6, None, 7]
    assert table["VALUE"].to_pylist() == ["-999", "8", "-999", "6", "-999", "7"]
    assert table["amount_status"].to_pylist() == ["suppressed", "observed"] * 3
    assert json.loads(table.schema.metadata[b"teadata_provenance"]) == provenance
    assert series.to_df().attrs["teadata_provenance"] == provenance


def test_series_rejects_bad_years_pins_and_incompatible_projections_before_reads(tmp_path, monkeypatch):
    children = releases(tmp_path, normalized=False)
    parent = children[2023]
    discoveries = use_children(parent, children, monkeypatch)
    monkeypatch.setattr(DataRelease, "artifact", lambda *a, **k: pytest.fail("Invalid series opened a native leaf"))
    args = {"scope": "campus", "years": [23, 24]}
    for arguments, message in (({"years": [23, "2022-23"]}, "distinct"),
                               ({"years": [None]}, "explicit school_year"),
                               ({"ids": ["000000000"]}, "nonzero"),
                               ({"ids": [4, "'000000004"]}, "same ID"),
                               ({"ids": [True]}, "Invalid ID"),
                               ({"ids": [4], "names": ["Observed"]}, "either ids or names"),
                               ({"names": ["Observed"]}, "requires an entity release"),
                               ({"editions": {25: None}}, "EditionKey")):
        with pytest.raises((TypeError, ValueError, ContractError), match=message):
            parent.series("campus_student", "amount", **(args | arguments))
    assert not discoveries
    with pytest.raises(ContractError, match="Fields are unavailable together.*2026"):
        parent.series("campus_student", "amount", scope="campus", years=[23, 26])
    with pytest.raises(ContractError, match="fields, types or publisher keys"):
        parent.series("campus_student", "amount", scope="campus", years=[23, 25])
    first = children[2023].observations("campus_student", "amount", school_year=23, scope="campus").edition
    last = children[2024].observations("campus_student", "amount", school_year=24, scope="campus").edition
    discoveries.clear()
    planned = parent.series("campus_student", "amount", **args, editions={23: first, "2023-24": last})
    assert planned.years == (2023, 2024)
    assert discoveries == [first, last]



def test_series_requires_same_support_roles_even_when_physical_schema_matches(tmp_path, monkeypatch):
    from teadata.export_provenance import ExportProvenance
    from teadata.manifest import canonical_json
    from teadata.source_views import _ExportSelection

    children = releases(tmp_path)
    parent = children[2023]
    use_children(parent, children, monkeypatch)
    original = _ExportSelection.provenance

    def different_role(selection, release, **kwargs):
        source = original(selection, release, **kwargs)
        if release is children[2024]:
            document = source.to_dict()
            raw, = [column for column in document["columns"] if column["column"] == "VALUE"]
            raw["support_for"] = [{"field": "amount", "role": "status"}]
            source = ExportProvenance(canonical_json(document))
        return source

    monkeypatch.setattr(_ExportSelection, "provenance", different_role)
    monkeypatch.setattr(DataRelease, "artifact", lambda *a, **k: pytest.fail("Incompatible series opened a native leaf"))
    with pytest.raises(ContractError, match="fields, types or publisher keys.*2024|incompatible roles or types.*2024"):
        parent.series("campus_student", "amount", scope="campus", years=[23, 24])


def test_series_pins_ambiguous_edition_and_round_trips_literal_year(tmp_path, monkeypatch):
    children = releases(tmp_path)
    parent = children[2023]

    def candidate(name, year, amount):
        directory = tmp_path / name
        directory.mkdir()
        fixture = make_edition_fixture(directory, families=("student",), normalized=True, entity_grain=True,
                                       amount_period=DataPeriod("school_year", year - 1, year).to_dict(),
                                       observed_amount=amount)
        return DataRelease(fixture["root"], offline=True)

    children["revision"] = candidate("revision", 2024, 99)
    children["literal"] = candidate("literal", 25, 9)
    use_children(parent, children, monkeypatch)
    args = {"scope": "campus", "years": [23, 24]}
    with pytest.raises(ContractError, match="Multiple editions match school year 2024"):
        parent.series("campus_student", "amount", **args)
    key = children["revision"].observations("campus_student", "amount", school_year=24, scope="campus").edition
    assert parent.series("campus_student", "amount", **args, editions={24: key}).for_year(24).edition == key
    with pytest.raises(ContractError, match="Fields are unavailable together.*2023"):
        parent.series("campus_student", "amount", **args, editions={23: key})
    literal = parent.series("campus_student", "amount", scope="campus", years=["0025", 25])
    assert literal.years == (25, 2025)
    assert literal.for_year(f"{literal.years[0]:04d}").release is children["literal"]
    assert literal.for_year(25).release is children[2025]


def test_series_preserves_multiple_fields_and_nullable_context_from_real_catalog(tmp_path):
    from hashlib import sha256

    from teadata.manifest import canonical_json

    document = {}
    for year, extra_context in ((2023, "early_context"), (2024, None), (2025, "later_context")):
        directory = tmp_path / str(year)
        directory.mkdir()
        period = DataPeriod("school_year", year - 1, year).to_dict()
        fixture = make_edition_fixture(directory, families=("student",), normalized=True, entity_grain=True,
                                       amount_period=period, name_period=period, observed_amount=year,
                                       extra_context=extra_context)
        catalog = fixture["catalog"].to_dict()
        for artifact in catalog["artifacts"].values():
            destination = tmp_path / f"{year}.parquet"
            destination.hardlink_to(directory / artifact["location"])
            artifact["location"] = destination.name
        for digest in fixture["resources"]:
            destination = tmp_path / f"{digest}.json"
            if not destination.exists():
                destination.hardlink_to(directory / destination.name)
        if not document:
            document = catalog
        else:
            for name in ("artifacts", "column_schemas", "editions", "provenance"):
                document[name].update(catalog[name])
            document["field_bindings"].extend(catalog["field_bindings"])
    document.pop("build_id")
    document["build_id"] = sha256(canonical_json(document).encode()).hexdigest()
    manifest = tmp_path / "release.json"
    manifest.write_text(canonical_json({"schema_version": 3, "catalog": document}))
    release = DataRelease(manifest, offline=True)
    view = release.series("campus_student", "amount", "campus_name", scope="campus", years=[24, 23, 25],
                          ids=["000000004"])
    rows = view.to_arrow().to_pylist()
    assert [(row["measurement_school_year"], row["amount"], row["campus_name"]) for row in rows] == [
        (2024, 2024, "Observed"), (2023, 2023, "Observed"), (2025, 2025, "Observed")]
    assert [row["early_context"] for row in rows] == [None, "retained context", None]
    assert [row["later_context"] for row in rows] == [None, None, "retained context"]
    assert all(row["NAME"] == "Observed" and row["amount_status"] == "observed" for row in rows)
    output = BytesIO()
    view.to_csv(output)
    csv_rows = list(csv.DictReader(StringIO(output.getvalue().decode())))
    assert [int(row["measurement_school_year"]) for row in csv_rows] == [2024, 2023, 2025]
    assert [row["early_context"] for row in csv_rows] == ["", "retained context", ""]
    assert [row["later_context"] for row in csv_rows] == ["", "", "retained context"]
    header, *excel_rows = excel_export(view, tmp_path / "nullable-context.xlsx")
    assert excel_rows == [tuple(row[name] for name in header) for row in rows]


def test_series_distinguishes_unknown_fields_from_unavailable_years(tmp_path, monkeypatch):
    children = releases(tmp_path)
    parent = children[2023]
    use_children(parent, children, monkeypatch)
    monkeypatch.setattr(DataRelease, "artifact", lambda *a, **k: pytest.fail("Invalid selector opened a data leaf"))
    with pytest.raises(ContractError, match=r"Unknown source field.*Did you mean.*amount"):
        parent.series("campus_student", "amont", scope="campus", years=[23])
    with pytest.raises(ContractError, match=r"Available years.*2023, 2024, 2025"):
        parent.series("campus_student", "AMOUNT", scope="campus", years=[26])
    with pytest.raises(ContractError, match="distinct canonical fields"):
        parent.series("campus_student", "AMOUNT", "amount", scope="campus", years=[24])


def test_series_names_pin_the_registry_separately_from_observations(tmp_path, monkeypatch):
    from teadata import EntitySelectionError
    from teadata.entity_store import EntityStore

    children = releases(tmp_path)
    parent = children[2023]
    use_children(parent, {2023: parent}, monkeypatch)
    registry = children[2024]
    monkeypatch.setattr(registry, "_v3", False)
    monkeypatch.setattr(registry, "artifact", lambda role: tmp_path / role)
    monkeypatch.setattr(EntityStore, "__enter__", lambda store: store)
    monkeypatch.setattr(EntityStore, "iter_campuses", lambda store, **kwargs: iter([
        {"name": "Observed", "lookup_key": "000000004", "is_private": False},
        {"name": "Observed", "lookup_key": "tefa:9001", "is_private": True},
    ]))
    args = {"scope": "campus", "years": [23], "entity_release": registry}
    series = parent.series("campus_student", "amount", names=["oBseRved"], **args)
    assert series.to_arrow()["amount"].to_pylist() == [6]
    assert series.provenance()["name_selection"] == {
        "manifest_sha256": registry.manifest_sha256, "build_id": registry.build_id,
        "resolved": [{"name": "oBseRved", "id": "000000004"}],
    }
    excel_export(series, tmp_path / "named-series.xlsx")
    with pytest.raises(EntitySelectionError, match="Did you mean"):
        parent.series("campus_student", "amount", names=["Observd"], **args)


def test_pair_years_aligns_only_two_pinned_leaves_and_keeps_absences(tmp_path, monkeypatch):
    from teadata.source_series import _PROVENANCE_KEY

    children = releases(tmp_path, unique_entity_key=True)
    parent = children[2023]
    use_children(parent, children, monkeypatch)
    actual = DataRelease.artifact
    opened = []

    def track(release, role, **kwargs):
        opened.append(release.manifest_sha256)
        return actual(release, role, **kwargs)

    monkeypatch.setattr(DataRelease, "artifact", track)
    series = parent.series("campus_student", "amount", scope="campus", years=[25, 23, 24],
                           ids=[4, 999, 2])
    pair = series.pair_years(from_year="2022–23", to_year=25)
    receipt = pair.provenance()
    assert not opened
    assert receipt["cross_year_comparability"] == receipt["historical_entity_continuity"] == "unassessed"
    table = pair.to_arrow()
    assert opened == [children[year].manifest_sha256 for year in (2023, 2025)]
    assert table["campus_number"].to_pylist() == ["000000004", "000000999", "000000002"]
    assert table["from_campus_number"].to_pylist() == ["004", None, "002"]
    assert table["from_present"].to_pylist() == [True, False, True]
    assert table["to_present"].to_pylist() == [True, False, True]
    assert table["from_amount"].to_pylist() == [6, None, None]
    assert table["to_amount"].to_pylist() == [8, None, None]
    assert table["to_VALUE"].to_pylist() == ["8", None, "-999"]
    assert table["to_amount_status"].to_pylist() == ["observed", None, "suppressed"]
    assert json.loads(table.schema.metadata[_PROVENANCE_KEY]) == receipt
    assert pair.to_df().attrs["teadata_provenance"] == receipt

    full = parent.series("campus_student", "amount", scope="campus", years=[23, 25])
    assert full.pair_years(from_year=23, to_year=25).to_arrow()["campus_number"].to_pylist() == [
        "000000001", "000000002", "000000003", "000000004",
    ]


def test_pair_years_rejects_invalid_selection_and_publisher_grain_before_io(tmp_path, monkeypatch):
    children = releases(tmp_path)
    parent = children[2023]
    use_children(parent, children, monkeypatch)
    monkeypatch.setattr(DataRelease, "artifact", lambda *a, **k: pytest.fail("Invalid pair opened a leaf"))
    series = parent.series("campus_student", "amount", scope="campus", years=[23, 25])
    with pytest.raises(ValueError, match="earlier"):
        series.pair_years(from_year=25, to_year=23)
    with pytest.raises(ContractError, match="Selected years"):
        series.pair_years(from_year=23, to_year=24)
    with pytest.raises(ContractError, match="Publisher keys.*source_record"):
        series.pair_years(from_year=23, to_year=25)


def test_pair_years_validates_rows_outside_requested_cohort(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pyarrow as pa
    from teadata import source_series

    children = releases(tmp_path, unique_entity_key=True)
    parent = children[2023]
    use_children(parent, children, monkeypatch)
    series = parent.series("campus_student", "amount", scope="campus", years=[23, 25], ids=[4])
    pair = series.pair_years(from_year=23, to_year=25)
    bad_keys = pa.table({"campus_number": ["004", "002", "'000000002"]})
    monkeypatch.setattr(source_series, "_open_export", lambda *a, **k: SimpleNamespace(to_arrow=lambda: bad_keys))
    with pytest.raises(ContractError, match="duplicate canonical campus_number.*2023"):
        pair.to_arrow()
