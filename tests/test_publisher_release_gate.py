"""Release checks read persisted values, independently of excerpt construction."""

from dataclasses import replace
from importlib.resources import files
import json
import sqlite3
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import pytest

from teadata import Campus, DataEngine, District, verification
from teadata.manifest import DatasetCatalog
from teadata.persistence import create_engine, create_sessionmaker, ensure_schema, export_dataengine
from teadata.teadata_config import canonical_campus_number, load_config, normalize_district_number_value


@pytest.fixture
def stored_publisher_cases(tmp_path, catalog, monkeypatch):
    # Freeze the real selected contracts into this local unit catalog. The
    # global fixture intentionally omits schema resources; the release gate
    # must exercise DatasetCatalog's real header projection, not a mock.
    configured = load_config()
    document = catalog.to_dict()
    schemas = {}
    for scope in ("campus", "district"):
        dataset = f"{scope}_tapr_student_staff_profile"
        source = configured.resolve_dataset(dataset, 2025)
        contract = configured.column_contract(dataset, 2025)
        document["datasets"][dataset].update(
            resolved_year=2025, reader=source.reader, column_schema_sha256=contract["sha256"],
            reads=[{"profile": None, "columns": configured.column_schema(dataset, 2025)}],
        )
        schemas[contract["sha256"]] = contract["text"]
    catalog = DatasetCatalog.create(
        datasets=document["datasets"], value_policies=document["value_policies"],
        processor=document["processor_sha256"], records=document["provenance"],
        metric_bindings=document["metric_bindings"], environment=document["build_environment"],
        column_schemas=schemas,
    )
    reference = verification._publisher_reference_for_release(catalog.datasets)
    coverage = reference["release_validation"]
    cases = [case for case in reference["cases"] if case["case_id"] in coverage["case_ids"]]
    # This controlled unit database uses the ordinary catalog fixture's source
    # bytes. Publisher facts remain literal; the real-bundle witness uses its
    # original source hashes and is qualified separately.
    for source in reference["sources"].values():
        if "release_input" in source:
            selected_input = source["release_input"]
            selected_input["sha256"] = catalog.datasets[selected_input["dataset"]]["artifact"]["sha256"]
    repo = DataEngine()
    repo.catalog = catalog
    numbers = {case["entity"].get("governing_district", case["entity"]["number"])
               for case in cases}
    for number in numbers:
        repo.add_district(District(id=uuid5(NAMESPACE_URL, number), name=number, district_number=number))
    for case in cases:
        scope = case["entity"]["type"]
        number = case["entity"]["number"]
        if scope == "district":
            entity = repo._districts[uuid5(NAMESPACE_URL, number)]
        else:
            entity = Campus(id=uuid5(NAMESPACE_URL, number),
                            district_id=uuid5(NAMESPACE_URL, case["entity"]["governing_district"]),
                            name=case["entity"]["name"], campus_number=number,
                            charter_type="", is_charter=False)
            repo.add_campus(entity)
        entity.meta = {fact["field"]: fact["source"]["raw_token"] for fact in case["facts"]}
        expected = {fact["expected"]["metric"]: fact["expected"]["value"] for fact in case["facts"]}
        for attribute, metric in case.get("attribute_metrics", coverage["attribute_metrics"]).items():
            setattr(entity, attribute, expected[metric])
    path = tmp_path / "entities.sqlite"
    engine = create_engine(f"sqlite:///{path}")
    try:
        ensure_schema(engine)
        with create_sessionmaker(engine).begin() as session:
            export_dataengine(repo, session, include_meta_entries=False)
    finally:
        engine.dispose()
    monkeypatch.setattr(verification, "_publisher_reference_for_release", lambda _: reference)
    return SimpleNamespace(catalog=catalog, artifact=lambda role: path), reference, path


def test_publisher_gate_reads_real_stored_rows_and_reports_limited_coverage(stored_publisher_cases):
    release, _, _ = stored_publisher_cases
    result = verification._validate_publisher_cases(release)
    assert (result["cases"], result["measurements"], result["attributes"]) == (5, 73, 9)
    assert "Private-school governing affiliation" in result["uncovered"]


def test_publisher_gate_rejects_wrong_attribute_despite_correct_raw_metadata(stored_publisher_cases):
    release, _, path = stored_publisher_cases
    with sqlite3.connect(path) as connection:
        result = connection.execute("UPDATE campuses SET rating='C' WHERE campus_number_canon=?",
                                    (canonical_campus_number("227901180"),))
        assert result.rowcount == 1
    with pytest.raises(verification.ReceiptError, match="kiker.*attribute rating"):
        verification._validate_publisher_cases(release)


def test_publisher_gate_rejects_wrong_stored_raw_value_despite_correct_attribute(stored_publisher_cases):
    release, _, path = stored_publisher_cases
    with sqlite3.connect(path) as connection:
        key = canonical_campus_number("227901180")
        row = connection.execute("SELECT meta FROM campuses WHERE campus_number_canon=?", (key,)).fetchone()
        payload = json.loads(row[0])
        payload["overall_rating"] = "C"
        result = connection.execute("UPDATE campuses SET meta=? WHERE campus_number_canon=?", (json.dumps(payload), key))
        assert result.rowcount == 1
    with pytest.raises(verification.ReceiptError, match="kiker.*metric overall_rating value"):
        verification._validate_publisher_cases(release)


def test_publisher_gate_refuses_changed_source_before_opening_store(stored_publisher_cases, monkeypatch):
    release, reference, _ = stored_publisher_cases
    reference["sources"]["accountability_2025"]["release_input"]["sha256"] = "0" * 64
    monkeypatch.setattr("teadata.entity_store.EntityStore", lambda *args, **kwargs: pytest.fail("Read data before source fence"))
    with pytest.raises(verification.ReceiptError, match="source changed: accountability"):
        verification._validate_publisher_cases(release)


@pytest.fixture
def publisher_2026_inputs():
    edition = json.loads(files("teadata").joinpath("reference_cases/publisher_2026.json").read_text())
    datasets = {}
    for dataset, source, period in (
        ("accountability", edition["accountability"], {"kind": "calendar_year", "year": 2026}),
        ("campus_transfer_reports", edition["campus_transfers"],
         {"kind": "school_year", "start_year": 2025, "end_year": 2026}),
    ):
        datasets[dataset] = {"status": "loaded", "artifact": {"sha256": source["sha256"]},
            "requested_year": 2026, "resolved_year": 2026,
            "metadata": {"version": "2026", "period": period, "references": [
                {"reference_id": name, "provenance_ids": [receipt]} for name, receipt in source["references"].items()
            ]}}
    datasets["accountability"]["metadata"]["source_qualifiers"] = {
        "family": "accountability", "publisher_release_status": "preliminary", "reference_id": "publication",
    }
    return datasets


def test_2026_publisher_reference_replaces_only_edition_specific_witnesses(publisher_2026_inputs):
    original = verification._publisher_reference()
    current = verification._publisher_reference_for_release(publisher_2026_inputs)
    before = {case["case_id"]: case for case in original["cases"]}
    after = {case["case_id"]: case for case in current["cases"]}
    active = current["release_validation"]["case_ids"]
    assert set(active) == (set(original["release_validation"]["case_ids"]) - {"rocketship-reporting-only-notrated"}
                           | {"trinity-beeville-reporting-only-2026"})
    for key in set(before) & set(active):
        assert [fact for fact in before[key]["facts"] if fact["field"] != "overall_rating"] == [
            fact for fact in after[key]["facts"] if fact["field"] != "overall_rating"]
    austin, = [fact for fact in after["austin-isd-2025"]["facts"] if fact["field"] == "overall_rating"]
    assert (austin["source"]["cell"], austin["expected"]["value"], austin["expected"]["period"]["year"]) == ("G9343", "B", 2026)
    campus = after["trinity-beeville-reporting-only-2026"]
    assert campus["attribute_metrics"] == {"rating": "overall_rating"}
    assert campus["facts"][0]["expected"]["status"] == "observed"
    assert campus["facts"][0]["expected"]["value"] == "Not Rated: Minimum Size rules"
    assert verification._publisher_reference() == original


@pytest.mark.parametrize("failure", ["transfer_hash", "masking_receipt", "final_label", "workbook_year"])
def test_2026_publisher_reference_requires_exact_input_lineage(publisher_2026_inputs, failure):
    accountability = publisher_2026_inputs["accountability"]
    transfers = publisher_2026_inputs["campus_transfer_reports"]
    if failure == "transfer_hash":
        transfers["artifact"]["sha256"] = "0" * 64
    elif failure == "masking_receipt":
        transfers["metadata"]["references"] = []
    elif failure == "final_label":
        accountability["metadata"]["source_qualifiers"]["publisher_release_status"] = "final"
    else:
        accountability["resolved_year"] = 2025
    with pytest.raises(verification.ReceiptError, match="2026 publisher"):
        verification._publisher_reference_for_release(publisher_2026_inputs)


def test_publisher_gate_rejects_a_bad_asian_mapping_even_when_enrollment_is_correct(stored_publisher_cases):
    release, _, path = stored_publisher_cases
    with sqlite3.connect(path) as connection:
        key = normalize_district_number_value("227901")
        row = connection.execute("SELECT meta FROM districts WHERE district_number_canon=?", (key,)).fetchone()
        payload = json.loads(row[0])
        payload["district_2025_student_enrollment_asian_count"] = "125"
        result = connection.execute("UPDATE districts SET meta=? WHERE district_number_canon=?", (json.dumps(payload), key))
        assert result.rowcount == 1
    with pytest.raises(verification.ReceiptError, match="austin.*student_enrollment_asian_count value"):
        verification._validate_publisher_cases(release)


@pytest.mark.parametrize("component,incorrect,reported", [
    ("code", "DPNTINDC", "code"),
    ("header_row1", "District 2025 Student Enrollment: American Indian Count", "label"),
    ("column_1_based", 179, "position"),
    ("parent_sha256", "0" * 64, "input_sha256"),
])
def test_publisher_gate_refuses_wrong_source_lineage_with_unchanged_values(
    stored_publisher_cases, monkeypatch, component, incorrect, reported,
):
    release, reference, path = stored_publisher_cases
    case = next(c for c in reference["cases"] if c["case_id"] == "austin-isd-2025")
    fact = next(f for f in case["facts"] if f["field"] == "district_2025_student_enrollment_asian_count")
    expectations = json.dumps([[f["expected"] for f in c["facts"]]
                               for c in reference["cases"] if "facts" in c], sort_keys=True)
    with sqlite3.connect(path) as connection:
        before = connection.execute("SELECT enrollment,rating,meta FROM districts ORDER BY id").fetchall()
    if component == "parent_sha256":
        reference["sources"]["tapr_district_csv"]["sha256"] = incorrect
    else:
        fact["source"][component] = incorrect
    # Only a source locator changes; all literal numeric expectations, stored
    # values, attributes and valid frozen catalog resources remain untouched.
    assert json.dumps([[f["expected"] for f in c["facts"]]
                       for c in reference["cases"] if "facts" in c], sort_keys=True) == expectations
    monkeypatch.setattr("teadata.entity_store.EntityStore", lambda *a, **k: pytest.fail("Read values before lineage fence"))
    with pytest.raises(verification.ReceiptError, match="source lineage " + reported + " differs"):
        verification._validate_publisher_cases(release)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT enrollment,rating,meta FROM districts ORDER BY id").fetchall() == before


def test_publisher_gate_refuses_invalid_case_attribute_expectations(stored_publisher_cases):
    release, reference, _ = stored_publisher_cases
    case = next(c for c in reference["cases"] if c["case_id"] == "highland-park-race-populations-2025")
    case["attribute_metrics"] = {"rating": "overall_rating"}
    with pytest.raises(verification.ReceiptError, match="highland-park.*invalid attribute expectations"):
        verification._validate_publisher_cases(release)


def test_publisher_gate_requires_governed_units_even_when_missing(stored_publisher_cases, monkeypatch):
    release, _, _ = stored_publisher_cases
    original = release.catalog.measurement

    def without_unit(entity, metric, **kwargs):
        value = original(entity, metric, **kwargs)
        return replace(value, unit=None) if metric == "enrollment" else value

    monkeypatch.setattr(release.catalog, "measurement", without_unit)
    with pytest.raises(verification.ReceiptError, match="metric enrollment unit"):
        verification._validate_publisher_cases(release)


def test_publisher_gate_rejects_invented_disclosure_bounds(stored_publisher_cases, monkeypatch):
    release, _, _ = stored_publisher_cases
    original = release.catalog.measurement

    def with_invented_bounds(entity, metric, **kwargs):
        value = original(entity, metric, **kwargs)
        return replace(value, lower=1, upper=9) if value.is_suppressed else value

    monkeypatch.setattr(release.catalog, "measurement", with_invented_bounds)
    with pytest.raises(verification.ReceiptError, match="kiker.*lower"):
        verification._validate_publisher_cases(release)
