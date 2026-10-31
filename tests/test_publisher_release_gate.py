"""Release checks read persisted values, independently of excerpt construction."""

from dataclasses import replace
import json
import sqlite3
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import pytest

from teadata import Campus, DataEngine, District, verification
from teadata.persistence import create_engine, create_sessionmaker, ensure_schema, export_dataengine
from teadata.teadata_config import canonical_campus_number


@pytest.fixture
def stored_publisher_cases(tmp_path, catalog, monkeypatch):
    reference = verification._publisher_reference()
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
        entity.enrollment, entity.rating = expected["enrollment"], expected["overall_rating"]
    path = tmp_path / "entities.sqlite"
    engine = create_engine(f"sqlite:///{path}")
    try:
        ensure_schema(engine)
        with create_sessionmaker(engine).begin() as session:
            export_dataengine(repo, session, include_meta_entries=False)
    finally:
        engine.dispose()
    monkeypatch.setattr(verification, "_publisher_reference", lambda: reference)
    return SimpleNamespace(catalog=catalog, artifact=lambda role: path), reference, path


def test_publisher_gate_reads_real_stored_rows_and_reports_limited_coverage(stored_publisher_cases):
    release, _, _ = stored_publisher_cases
    result = verification._validate_publisher_cases(release)
    assert (result["cases"], result["measurements"], result["attributes"]) == (4, 15, 8)
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
