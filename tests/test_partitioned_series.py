"""One routed cohort retains every physical source through years and exports."""

import csv
from io import BytesIO, StringIO

import pytest

from teadata.manifest import ContractError
from teadata.release import DataRelease
from teadata.source_views import PartitionedSourceTable
from test_catalog_projections import make_edition_fixture
from test_source_series import excel_export


def test_standalone_class_size_series_does_not_require_a_student_roster(tmp_path):
    from teadata.dataset_metadata import DataPeriod

    fixture = make_edition_fixture(tmp_path, families=("student",), dataset_name="tprs_class_size",
        normalized=True, unique_entity_key=True, amount_period=DataPeriod("school_year", 2024, 2025).to_dict())
    release = DataRelease(fixture["root"], offline=True)
    series = release.series("tprs_class_size", "amount", scope="campus", years=[25], ids=[4])
    assert series.to_arrow()["amount"].to_pylist() == [5]
    assert series.for_year(25).edition.dataset == "tprs_class_size"
    assert not release.catalog.native_sources


@pytest.fixture
def routed_series(tmp_path, monkeypatch):
    from shutil import copyfile
    from teadata.dataset_metadata import DataPeriod
    from teadata.manifest import DatasetCatalog
    from teadata.release import publish_release

    children = {}
    cohorts = {}
    for year in (2023, 2025):
        fixtures = []
        for partition in (0, 1):
            path = tmp_path / f"{year}-{partition}"
            path.mkdir()
            ids = tuple(f"{partition * 100 + n:03d}" for n in range(1, 5))
            fixture = make_edition_fixture(path, families=(f"partition{partition}",), dataset_name="tprs_class_size",
                normalized=True, unique_entity_key=True, identifiers=ids,
                observed_amount=year - 2017 + partition * 10,
                extra_context="partition_note" if year == 2025 else None,
                amount_period=DataPeriod("school_year", year - 1, year).to_dict())
            child = DataRelease(fixture["root"], offline=True)
            fixtures.append(child)
            key, _, _ = next(iter(fixture["leaves"].values()))
            cohorts[key.identity] = tuple(identifier.zfill(9) for identifier in ids)
        # A routed year consists of disjoint editions from ONE pinned child.
        target = tmp_path / f"composed-{year}"
        target.mkdir()
        sections = {name: {} for name in ("editions", "artifacts", "column_schemas", "provenance", "value_policies")}
        bindings = []
        for child in fixtures:
            document = child.catalog.to_dict()
            for name in sections:
                sections[name].update(document[name])
            bindings.extend(document["field_bindings"])
            for source in child.path.parent.iterdir():
                if source.is_file() and source != child.path:
                    copyfile(source, target / source.name)
        catalog = DatasetCatalog.create_editions(editions=sections["editions"], artifacts=sections["artifacts"],
            field_bindings=bindings, column_schemas=sections["column_schemas"], records=sections["provenance"],
            value_policies=sections["value_policies"], column_schema_loader=lambda ref: (target / ref.location).read_bytes())
        children[year] = DataRelease(publish_release(target, catalog, schema_version=3), offline=True)
    parent = children[2023]
    original = DataRelease._observation_year_options

    def options(dataset, fields, scope, edition):
        for child in children.values():
            yield from original(child, dataset, fields, scope, edition)

    def route(dataset, fields, *, school_year, scope, edition, identifiers, _options):
        assert dataset == "tprs_class_size" and scope == "campus" and edition is None
        assert _options is not None  # One coherent metadata discovery for the series.
        year, matches = parent._observation_matches(fields, scope, school_year, _options)
        parts = []
        for child, key in matches:
            ids = tuple(identifier for identifier in identifiers if identifier in cohorts[key.identity])
            if ids:
                parts.append(parent._observation_view(fields, year, child, key)._for_entity_ids(ids))
        return parts[0] if len(parts) == 1 else PartitionedSourceTable(tuple(parts))

    monkeypatch.setattr(parent, "_observation_year_options", options)
    # Real roster routing is exercised with materialized TPRS reports in
    # test_tprs_students. Here distinct tiny leaves isolate series composition.
    monkeypatch.setattr(parent, "_entity_observations", route)
    return parent, children


def test_routed_series_keeps_all_years_partitions_masks_and_export_lineage(routed_series, monkeypatch, tmp_path):
    from teadata.source_series import ObservationSeries

    parent, children = routed_series
    with monkeypatch.context() as planning:
        planning.setattr(DataRelease, "artifact", lambda *a, **k: pytest.fail("Planning opened a leaf"))
        series = parent.series("tprs_class_size", "AMOUNT", scope="campus", years=[25, "2022-23"],
                               ids=["'004", 104, "002", 102])
        assert series.years == (2025, 2023)
        assert len(series.for_year(25).partitions) == 2
        sources = series.provenance()["sources"]
        assert len(sources) == 4
        for year in (2023, 2025):
            assert {tuple(source["provenance"]["selector"]["entity_ids"]) for source in sources
                    if source["school_year"] == year} == {
                ("000000002", "000000004"), ("000000102", "000000104"),
            }
        assert {source["provenance"]["release"]["manifest_sha256"] for source in sources} == {
            child.manifest_sha256 for child in children.values()}
        assert series.schema.field("partition_note").nullable
        pair = series.pair_years(from_year=23, to_year=25)
        assert all(len(endpoint["provenance"]["partitions"]) == 2
                   for endpoint in pair.provenance()["endpoints"].values())
    actual = series.to_arrow().to_pylist()
    assert len(actual) == 8
    values = {(row["measurement_school_year"], int(row["campus_number"])): row["amount"] for row in actual}
    assert values == {(2025, 2): None, (2025, 4): 8, (2025, 102): None, (2025, 104): 18,
                      (2023, 2): None, (2023, 4): 6, (2023, 102): None, (2023, 104): 16}
    assert all(row["amount_status"] == "suppressed" and row["VALUE"] == "-999"
               for row in actual if int(row["campus_number"]) in (2, 102))
    assert series.for_year(25).to_arrow().num_rows == 4
    compared = pair.to_arrow().to_pylist()
    assert [row["campus_number"] for row in compared] == ["000000004", "000000104", "000000002", "000000102"]
    assert [row["from_amount"] for row in compared] == [6, 16, None, None]
    assert [row["to_amount"] for row in compared] == [8, 18, None, None]
    assert all(row["from_present"] and row["to_present"] for row in compared)
    frame = series.to_df()
    assert frame.attrs["teadata_provenance"] == series.provenance()
    monkeypatch.setattr(ObservationSeries, "to_arrow", lambda *_: pytest.fail("Export collected whole series"))
    stream = BytesIO()
    series.to_csv(stream)
    assert len(list(csv.DictReader(StringIO(stream.getvalue().decode())))) == 8
    assert len(excel_export(series, tmp_path / "parts.xlsx")) == 9


def test_partitioned_sql_export_records_every_source_and_year(routed_series):
    import pyarrow as pa
    from sqlalchemy import Column, Integer, JSON, MetaData, String, Table, Text, create_engine, select

    parent, _ = routed_series
    series = parent.series("tprs_class_size", "amount", scope="campus", years=[23, 25], ids=[4, 104])
    metadata = MetaData()
    receipts = Table("imports", metadata, Column("id", String(64), primary_key=True), Column("receipt", JSON))
    facts = Table("facts", metadata, Column("import_id", String(64), nullable=False),
                  *(Column(f.name, Integer if pa.types.is_integer(f.type) else Text) for f in series.schema))
    engine = create_engine("sqlite://")
    metadata.create_all(engine)
    with engine.begin() as connection:
        result = series.to_sql(connection, facts, receipts=receipts)
        assert result.rows_by_year == {2023: 2, 2025: 2} and not result.missing_entity_years
        document = connection.execute(select(receipts.c.receipt)).scalar_one()
        assert len(document["selection"]["sources"]) == 4
        stored = pa.Table.from_batches(series.iter_sql_batches(connection, facts, receipts=receipts,
                                                              import_id=result.import_id))
        assert sorted(stored["amount"].to_pylist()) == [6, 8, 16, 18]
    engine.dispose()


def test_pair_rejects_duplicate_normalized_ids_across_parts(routed_series, monkeypatch):
    from types import SimpleNamespace
    import pyarrow as pa

    parent, _ = routed_series
    pair = parent.series("tprs_class_size", "amount", scope="campus", years=[23, 25],
                         ids=[4, 104]).pair_years(from_year=23, to_year=25)
    # Each individual source remains unique; the collision exists across parts.
    monkeypatch.setattr("teadata.source_series._open_export", lambda *a, **k: SimpleNamespace(
        to_arrow=lambda: pa.table({"campus_number": ["'000000004"]})))
    with pytest.raises(ContractError, match="duplicate canonical campus_number.*2023"):
        pair.to_arrow()
