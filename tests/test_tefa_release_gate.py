"""Private release acceptance with inert storage, no SQLite or roster download."""

from copy import deepcopy
from dataclasses import replace
from importlib.resources import files
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from teadata import verification
from teadata.provenance import validate_record_graph
from teadata.teadata_config import load_config
from teadata.values import Measurement, ValueStatus


@pytest.fixture
def private_release(monkeypatch):
    reference = json.loads(files("teadata").joinpath("reference_cases/tefa_2026_27.json").read_text())
    expected = reference["source"]
    root = Path(__file__).resolve().parents[1]
    records = validate_record_graph(
        lambda key: json.loads((root / "provenance/records" / (key + ".json")).read_text()),
        [expected["acquisition_record_id"]],
    )
    metadata = {key: deepcopy(expected[key]) for key in ("publisher", "source_url", "version", "period")}
    metadata.update(download_url=expected["request_url"], identifiers=["tefa_listing_id"],
                    provenance_ids=[expected["acquisition_record_id"]],
                    references=[{"role": "schema", "status": "available",
                                 "reviewed_sha256": expected["column_schema_sha256"]}])
    source = {"status": "loaded", "section": "data_sources", "requested_year": None, "resolved_year": 2027,
              "metadata": metadata, "artifact": deepcopy(expected["artifact"]),
              "reader": {"columns": deepcopy(reference["reader_columns"])},
              "reads": [{"columns": list(reference["reader_columns"])}]}
    document = {"schema_version": 2, "datasets": {"tefa": source}, "provenance": records}
    build_id = "a" * 64
    stored = {}
    for number, case in enumerate(reference["cases"]):
        projected, raw = case["expected"], case["raw"]
        record = {key: deepcopy(projected[key]) for key in
                  ("lookup_key", "source_namespace", "source_id", "name", "grade_range", "lat", "lon")}
        record.update(id=f"inert-stored-identity-{number}", source_build_id=build_id, is_private=True, is_charter=False,
                      campus_number=None, campus_number_canon=None, district_id=None, district_name=None,
                      district_number=None, district_number_canon=None, enrollment=None, rating=None)
        meta = {field: deepcopy(raw[original]) for field, original in reference["reader_columns"].items()}
        meta.update({key: projected[key] for key in
                     ("private_listing_location_status", "private_listing_grade_status", "school_full_address")})
        meta.update(private_source_dataset="tefa", private_source_version="2026-27",
                    school_website=raw["website"], display_grade_range=raw["displayGradeRange"])
        record["meta"] = meta
        stored[projected["lookup_key"]] = record

    class Catalog:
        native_sources = ()

        def to_dict(self):
            return deepcopy(document)

        def measurement(self, entity, metric, *, scope):
            assert entity.is_private and scope == "campus"
            return Measurement(None, ValueStatus.NOT_APPLICABLE, reason="outside_metric_population",
                               _source_json=json.dumps({"build_id": build_id,
                                                       "dataset": reference["private_metrics"][metric]}))

    opened, looked_up = [], []

    class InertStore:
        def __init__(self, path, *, immutable):
            assert path == "inert-no-database" and immutable is True
            opened.append(path)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get_campus(self, key, *, include_meta, cache):
            assert include_meta is True and cache is False
            looked_up.append(key)
            return deepcopy(stored.get(key))

    monkeypatch.setattr("teadata.entity_store.EntityStore", InertStore)
    release = SimpleNamespace(catalog=Catalog(), build_id=build_id,
                              artifact=lambda role: "inert-no-database")
    return SimpleNamespace(release=release, reference=reference, document=document,
                           source=source, stored=stored, opened=opened, looked_up=looked_up)


def test_private_gate_checks_all_edge_cases_and_exact_raw_fields(private_release):
    fixture = private_release
    result = verification._validate_tefa_cases(fixture.release)
    assert (result["cases"], result["raw_fields"], result["private_measurements"]) == (12, 192, 48)
    assert result["status"] == "checked"
    assert len(fixture.opened) == 1 and len(set(fixture.looked_up)) == 12
    assert {"tefa:5EihFVKUqIj5W2", "tefa:vDRKUxdXPTnKEE"}.issubset(fixture.looked_up)
    reverse = [case for case in fixture.reference["cases"] if case["raw"]["serviceType"] == ["online", "local"]]
    assert len(reverse) == 3 and all(case["expected"]["lat"] is not None for case in reverse)
    assert sum(case["expected"]["private_listing_grade_status"] == "inconsistent"
               for case in fixture.reference["cases"]) == 2


@pytest.mark.parametrize("case_id", [
    "inconsistent-grade-sunrise", "inconsistent-grade-abc", "pre-k-local", "mixed-local-online",
])
def test_actual_store_row_conversion_matches_unknown_and_valid_grade_oracles(private_release, case_id):
    from teadata.entity_store import _campus_record

    fixture = private_release
    case = next(item for item in fixture.reference["cases"] if item["case_id"] == case_id)
    expected = fixture.stored[case["expected"]["lookup_key"]]
    # These are the ordinary _campus_select column positions. Inconsistent
    # domain grades persist as SQL NULL; the established store view returns ''.
    sql_grade = None if case["expected"]["private_listing_grade_status"] == "inconsistent" else expected["grade_range"]
    row = (None, None, expected["name"], False, True, None, None, None, None,
           sql_grade, expected["lon"], expected["lat"], json.dumps(expected["meta"]),
           None, None, None, expected["lookup_key"], "tefa", expected["source_id"], expected["id"], None)
    projected = _campus_record(row, include_meta=True, build_id=fixture.release.build_id)
    for key, value in expected.items():
        assert projected[key] == value and type(projected[key]) is type(value), key
    assert projected["meta"]["tefa_listing_display_grade_range"] == case["raw"]["displayGradeRange"]
    fixture.stored[expected["lookup_key"]] = projected
    assert verification._validate_tefa_cases(fixture.release)["cases"] == 12


def test_public_only_catalog_reports_no_private_coverage(private_release):
    private_release.document["datasets"].pop("tefa")
    result = verification._validate_tefa_cases(private_release.release)
    assert result["status"] == "not_present" and result["cases"] == 0 and result["uncovered"]
    assert private_release.opened == []


def test_current_official_config_and_actual_receipt_closure_pass(private_release):
    """Read only Config/receipt metadata; the entity boundary remains inert."""
    fixture = private_release
    config = load_config()
    selected = config.resolve_dataset("tefa")
    fixture.source.update(requested_year=selected.requested_year, resolved_year=selected.resolved_year,
                          metadata=selected.metadata.to_dict(), reader=dict(selected.reader),
                          reads=[{"columns": config.column_schema("tefa")}])
    root = Path(__file__).resolve().parents[1]
    fixture.document["provenance"] = validate_record_graph(
        lambda key: json.loads((root / "provenance/records" / (key + ".json")).read_text()),
        selected.metadata.provenance_ids,
    )
    assert verification._validate_tefa_cases(fixture.release)["cases"] == 12


def test_private_edition_does_not_masquerade_as_public_only_coverage(private_release):
    private_release.document["datasets"].pop("tefa")
    private_release.document["editions"] = {"inert-edition": {"key": {"dataset": "tefa"}}}
    with pytest.raises(verification.ReceiptError, match="source edition lacks"):
        verification._validate_tefa_cases(private_release.release)
    assert private_release.opened == []


@pytest.mark.parametrize("target,key,bad", [
    ("source", "requested_year", 2025),
    ("source", "resolved_year", 2025),
    ("metadata", "version", "2025"),
    ("metadata", "publisher", "Unknown legacy publisher"),
    ("metadata", "download_url", "https://example.invalid/legacy.xlsx"),
    ("artifact", "sha256", "0" * 64),
    ("artifact", "size_bytes", 1),
])
def test_private_source_fence_precedes_storage(private_release, target, key, bad):
    source = private_release.source
    selected = source if target == "source" else source[target]
    selected[key] = bad
    with pytest.raises(verification.ReceiptError, match="private source metadata or artifact"):
        verification._validate_tefa_cases(private_release.release)
    assert private_release.opened == []


def test_private_reader_fence_refuses_legacy_aliases_before_storage(private_release):
    private_release.source["reader"]["columns"]["tefa_listing_id"] = "ID"
    with pytest.raises(verification.ReceiptError, match="reader/schema"):
        verification._validate_tefa_cases(private_release.release)
    assert private_release.opened == []


@pytest.mark.parametrize("change", ["missing_anchor", "wrong_http", "wrong_record_identity"])
def test_private_source_requires_actual_bound_http_observation(private_release, change):
    fixture = private_release
    identifier = fixture.reference["source"]["acquisition_record_id"]
    if change == "missing_anchor":
        fixture.source["metadata"]["provenance_ids"] = []
    elif change == "wrong_http":
        fixture.document["provenance"][identifier]["parameters"]["http_method"] = "POST"
    else:
        fixture.document["provenance"][identifier]["artifact"]["sha256"] = "0" * 64
    with pytest.raises(verification.ReceiptError, match="verified official HTTP acquisition"):
        verification._validate_tefa_cases(fixture.release)
    assert fixture.opened == []


@pytest.mark.parametrize("key,bad", [
    ("source_id", "a-different-opaque-id"), ("is_private", False), ("is_charter", True),
    ("campus_number", "000000001"), ("district_number", "000001"), ("enrollment", 0),
    ("rating", "A"), ("source_build_id", "b" * 64),
])
def test_private_gate_refuses_identity_and_public_fact_transplants(private_release, key, bad):
    private_release.stored["tefa:qdgEKylWUBk0u6"][key] = bad
    with pytest.raises(verification.ReceiptError, match="core " + key + " differs"):
        verification._validate_tefa_cases(private_release.release)


@pytest.mark.parametrize("identifier,latitude,longitude", [
    ("7EH0FnFFCKT4bl", 33.1974054, -96.6528902),
    ("P9FB7ifiRDdeJk", 29.887227786869026, -95.64239514232791),
    ("m3dnV0ENXzzNKH", 0.0, 0.0),
    ("qdgEKylWUBk0u6", 33.3236208, -96.5663624),
])
def test_private_gate_refuses_office_pins_flips_missing_points_and_float_rounding(
    private_release, identifier, latitude, longitude,
):
    private_release.stored["tefa:" + identifier].update(lat=latitude, lon=longitude)
    with pytest.raises(verification.ReceiptError, match="core (lat|lon) differs"):
        verification._validate_tefa_cases(private_release.release)


def test_private_gate_retains_raw_array_order(private_release):
    private_release.stored["tefa:M0tA4gdiaMrEhd"]["meta"]["tefa_listing_service_type"] = ["local", "online"]
    with pytest.raises(verification.ReceiptError, match="raw field tefa_listing_service_type"):
        verification._validate_tefa_cases(private_release.release)


def test_private_gate_retains_two_duplicate_name_address_identities(private_release):
    private_release.stored["tefa:vDRKUxdXPTnKEE"]["id"] = private_release.stored["tefa:5EihFVKUqIj5W2"]["id"]
    with pytest.raises(verification.ReceiptError, match="identity is missing or collapsed"):
        verification._validate_tefa_cases(private_release.release)


@pytest.mark.parametrize("change", ["observed_zero", "foreign_build", "private_source"])
def test_private_gate_rejects_false_or_transplanted_na_measurements(private_release, monkeypatch, change):
    catalog = private_release.release.catalog
    original = catalog.measurement

    def altered(*args, **kwargs):
        value = original(*args, **kwargs)
        if change == "observed_zero":
            return replace(value, value=0, status=ValueStatus.OBSERVED)
        origin = value.source or {}
        origin["build_id" if change == "foreign_build" else "dataset"] = "foreign" if change == "foreign_build" else "tefa"
        return replace(value, _source_json=json.dumps(origin))

    monkeypatch.setattr(catalog, "measurement", altered)
    with pytest.raises(verification.ReceiptError, match="not release-bound N/A"):
        verification._validate_tefa_cases(private_release.release)


def test_canonical_release_validation_keeps_public_and_private_coverage(
    private_release, monkeypatch, capsys,
):
    release = private_release.release
    calls = []
    monkeypatch.setattr("teadata.release.DataRelease", lambda path, *, offline: release)
    monkeypatch.setattr(verification, "_validate_release_provenance", lambda value: 1)
    release.verify = lambda: calls.append("artifacts")

    def public(value):
        assert value is release
        calls.append("public")
        return {"cases": 5, "measurements": 73, "attributes": 9}

    original = verification._validate_tefa_cases

    def private(value):
        assert value is release
        calls.append("private")
        return original(value)

    monkeypatch.setattr(verification, "_validate_publisher_cases", public)
    monkeypatch.setattr(verification, "_validate_tefa_cases", private)
    verification._validate_release("inert-manifest.json", "app")
    assert calls == ["artifacts", "public", "private"]
    text = capsys.readouterr().out
    assert '"measurements": 73' in text and 'TEFA private example coverage:' in text
    assert '"private_measurements": 48' in text and '"raw_fields": 192' in text


def test_composed_release_holds_until_attached_sources_are_qualified(monkeypatch):
    pin = "a" * 64
    catalog = SimpleNamespace(to_dict=lambda: {"schema_version": 2},
                              native_sources=[SimpleNamespace(manifest_sha256=pin)])
    release = SimpleNamespace(catalog=catalog, verify=lambda: None)
    monkeypatch.setattr("teadata.release.DataRelease", lambda path, *, offline: release)
    for name in ("_validate_release_provenance", "_validate_publisher_cases", "_validate_tefa_cases"):
        monkeypatch.setattr(verification, name, lambda value: None)
    with pytest.raises(verification.ReceiptError, match="structural verification only.*" + pin):
        verification._validate_release("candidate.json", "app")


def test_composed_release_replays_exact_tprs_attachment(monkeypatch, tmp_path, capsys):
    from teadata.tprs_statewide_intake import REVIEWED_DISTRICT_STATE_MANIFEST

    ref = SimpleNamespace(manifest_sha256=REVIEWED_DISTRICT_STATE_MANIFEST)
    catalog = SimpleNamespace(to_dict=lambda: {"schema_version": 2}, native_sources=[ref])
    parent = SimpleNamespace(catalog=catalog, verify=lambda: None, path=tmp_path / "parent.json", build_id="parent")
    child = SimpleNamespace(path=tmp_path / "child" / "release.json", remote=False)
    monkeypatch.setattr("teadata.release.DataRelease", lambda path, *, offline: parent)
    monkeypatch.setattr("teadata.release._open_native_source", lambda *args, **kwargs: child)
    for name in ("_validate_release_provenance", "_validate_publisher_cases", "_validate_tefa_cases"):
        monkeypatch.setattr(verification, name, lambda value: None)
    seen = []
    monkeypatch.setattr(verification, "_retained_store", lambda path: path)

    def replay(release, store):
        seen.append((release, store))
        return {"status": "PASS_REVIEWED_NATIVE"}

    monkeypatch.setattr("teadata.tprs_statewide_intake.qualify_statewide_tprs", replay)
    verification._validate_release("candidate.json", "app")
    assert seen == [(child, child.path.parent / "provenance")]
    assert "PASS_REVIEWED_NATIVE" in capsys.readouterr().out
    child.remote = True
    with pytest.raises(verification.ReceiptError, match="Stage the exact reviewed TPRS attachment locally"):
        verification._validate_release("candidate.json", "app")
