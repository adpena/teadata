"""Exercise offline PEIMS edition production and exact-source rejection."""

from datetime import UTC, datetime
from hashlib import sha256
import json
from uuid import uuid4

import pytest

from teadata.column_schema import source_schema_fingerprint
from teadata.dataset_metadata import DataPeriod, DatasetMetadata, EditionKey, ResolvedDataset
from teadata.entities import Campus, District
from teadata.manifest import ContractError, canonical_json
from teadata.peims_leaves import PRIMARY_DISABILITY, PROGRAMS, materialize_peims_students
from teadata.provenance import ProvenanceStore
from teadata.query import Query
from teadata.release import DataRelease
from teadata.scripts.refresh_peims import main
from teadata.statewide import _Projection, _reviewed_peims
from teadata.teadata_config import load_config


YEAR = 2026
FIELD = "total_enrollment_count"
SELECTOR = "https://example.test/peims-selector"


def fixture_config(tmp_path, *, damage=None, dataset=PROGRAMS):
    store = ProvenanceStore(tmp_path / "store")
    period = DataPeriod("school_year", YEAR - 1, YEAR)
    metadata = DatasetMetadata(title="Fixture PEIMS program counts", version=str(YEAR), period=period, source_url=SELECTOR)
    source = ResolvedDataset(dataset, "data_sources", YEAR, YEAR, "", metadata)
    tables = []
    acquisitions = {}
    scope_rows = (
        ("state", {}, ["0"]),
        ("district", {"DISTRICT NUMBER": "district_number"}, ["001901", "0"]),
        ("campus", {"DISTRICT NUMBER": "district_number", "CAMPUS NUMBER": "campus_number"},
            ["001901", "001901001", "-999"]),
    )
    disability = dataset == PRIMARY_DISABILITY
    if disability and damage != "extra-campus":
        scope_rows = scope_rows[:2]
    for scope, geography, cells in scope_rows:
        if scope == "state" and damage == "masked-state":
            cells[-1] = "-999"
        if scope == "campus" and damage == "campus-prefix":
            cells[1] = "002901001"
        if scope == "campus" and damage == "bad-mask":
            cells[2] = "-1"
        if disability and scope != "state":
            geography = {"DISTRICT" if label == "DISTRICT NUMBER" else label: field
                         for label, field in geography.items()}
        if disability and scope == "district":
            cells[-1] = "-999"
        columns = {"SCOPE": "source_scope", "AGGREGATION LEVEL": "aggregation_level", "YEAR": "school_year",
                   **geography, "LD" if disability else "ALL ENROLLMENT": FIELD}
        headers = list(columns)
        preamble = b"PEIMS report for 2025-2026\n"
        aggregation = "COUNTYREGIONDISTRICT" if disability and scope == "district" and damage != "wrong-aggregation" else scope.upper()
        row = (",".join(["STATE", aggregation, "2025-2026", *cells]) + "\n").encode()
        body = row * (2 if scope == "campus" and damage == "duplicate" else 1)
        last_data_record = 3 + body.count(b"\n") - 1
        footer = b"Publisher FERPA footnote\n"
        raw = preamble + (",".join(headers) + "\n").encode() + body + footer
        path = tmp_path / f"{scope}.csv"
        path.write_bytes(raw)
        url = f"https://example.test/{scope}.csv"
        acquisitions[scope] = store.record_acquisition(path, source,
            downloaded_at=datetime(YEAR, 11, 4, tzinfo=UTC), recorded_at=datetime(YEAR, 11, 4, tzinfo=UTC),
            requested_url=url, resolved_url=url,
            response_headers={"content-type": "application/csv"})
        tables.append({"table": scope, "header_identity": "position", "input_sha256": sha256(raw).hexdigest(),
            "source_schema_sha256": source_schema_fingerprint(headers, headers), "metric_period": period.to_dict(),
            "layout": {"code_record": 2, "label_record": None, "first_data_record": 3,
                       "last_data_record": last_data_record, "annotations": [
                           {"first_record": 1, "last_record": 1, "kind": "preamble", "sha256": sha256(preamble).hexdigest()},
                           {"first_record": last_data_record + 1, "last_record": last_data_record + 1, "kind": "footer",
                            "sha256": sha256(footer).hexdigest()}]},
            "fields": [{"position": index, "code": code, "label": code, "field": field,
                        "disposition": "fact" if field == FIELD else "identity",
                        "dictionary_status": "not_documented" if field == FIELD else "not_applicable",
                        "publisher_type": None, "publisher_description": None}
                       for index, (code, field) in enumerate(columns.items(), 1)]})
    if damage == "wrong-scope":
        acquisitions["state"], acquisitions["campus"] = acquisitions["campus"], acquisitions["state"]
    crosswalk = tmp_path / "crosswalk.json"
    crosswalk.write_text(canonical_json({"schema_version": 2, "mapping_version": 1,
        "datasets": [{"dataset": dataset, "edition": str(YEAR), "source_only": True, "tables": tables}]}) + "\n")
    masking = tmp_path / "masking.txt"
    masking.write_text("Synthetic publisher masking rules")
    references = []
    for reference_id, role, path in (("columns", "schema", crosswalk), ("masking", "masking_methodology", masking)):
        receipt = store.record_acquisition(path, ResolvedDataset("reference:" + reference_id, "reference", None, None,
            "", DatasetMetadata(title=reference_id)))
        references.append({"reference_id": reference_id, "role": role, "title": reference_id,
                           "status": "available", "path": path.name, "reviewed_sha256": sha256(path.read_bytes()).hexdigest(),
                           "provenance_ids": [receipt]})
    config = tmp_path / "config.json"
    config.write_text(canonical_json({"data_sources": {dataset: {str(YEAR): {
        "path": "unused.csv", "metadata": {**metadata.to_dict(), "references": references, "value_policy": "policy",
            "metrics": {FIELD: {"field": FIELD, "period": period.to_dict(), "unit": "count"}}},
        "reader": {"profiles": {"source": {"column_schema": crosswalk.name, "acquisition_ids": acquisitions}}}}}},
        "value_policies": {"policy": {"policy_id": "policy", "reference_id": "masking", "tokens": {
            "-999": {"status": "suppressed", "reason": "fixture_mask"}}}}}) + "\n")
    return store, config, sha256(config.read_bytes()).hexdigest(), acquisitions


def test_program_cli_materializes_exact_scope_leaves_and_retains_suppression(tmp_path):
    store, config, digest, acquisitions = fixture_config(tmp_path)
    output = tmp_path / "candidate"
    assert main(["--store", str(store.root), "--config", str(config), "--config-sha256", digest,
        "--source", PROGRAMS, str(YEAR), "--output", str(output)]) == 0
    release = DataRelease(output / "release.json", offline=True)
    release.verify()
    materialization = json.loads((output / "MATERIALIZATION.json").read_bytes())
    assert materialization["release_activation"] is False
    assert {item["key"]["scope"] for item in materialization["editions"]} == set(acquisitions)
    for item in materialization["editions"]:
        key = EditionKey.from_dict(item["key"])
        assert item["acquisition_id"] == acquisitions[key.scope]
        selection, = release.catalog.resolve_artifacts([(key, FIELD)])
        rows = release.analytical_dataset(selection["artifact"].identity).to_table(columns=list(selection["columns"]))
        row, = rows.to_pylist()
        assert row["peims_source_record"] == 3
        if key.scope == "campus":
            assert row["campus_number"] == "001901001"
            assert (row[FIELD], row[f"peims_raw_{FIELD}"], row[f"peims_status_{FIELD}"]) == (None, "-999", "suppressed")
        else:
            assert (row[FIELD], row[f"peims_raw_{FIELD}"], row[f"peims_status_{FIELD}"]) == (0, "0", "observed")


@pytest.mark.parametrize("damage", [None, "masked-state"])
def test_statewide_peims_comparison_preserves_counts_and_both_statuses(tmp_path, monkeypatch, damage):
    store, config, digest, _ = fixture_config(tmp_path, damage=damage)
    output = tmp_path / "candidate"
    materialize_peims_students(store, config=config, config_sha256=digest,
                               selections=[(PROGRAMS, YEAR)], output=output)
    release = DataRelease(output / "release.json", offline=True)
    editions = {row["scope"]: EditionKey.from_dict({name: row[name] for name in EditionKey.__dataclass_fields__})
                for row in release.catalog.editions(include_metadata=False)}
    cfg = load_config(config)
    monkeypatch.setattr("teadata.statewide.load_config", lambda: cfg)
    _reviewed_peims.cache_clear()
    try:
        district = Query([District(uuid4(), "Test ISD", district_number="001901")], repo=None)
        row, = district >> ("compare_statewide", release, editions["district"], editions["state"])
        assert (row.metric, row.unit, row.local_value, row.difference_pp) == (FIELD, "count", 0, None)
        expected = ("-999", "suppressed", None, None) if damage else ("0", "observed", 0, 0)
        assert (row.statewide_raw, row.statewide_status, row.statewide_value, row.difference) == expected
        campus = Campus(uuid4(), None, "School", "", False, campus_number="001901001")
        result, = Query([campus], repo=None).compare_statewide(
            release=release, local=editions["campus"], statewide=editions["state"])
        assert (result.local_raw, result.local_status, result.difference) == ("-999", "suppressed", None)
    finally:
        _reviewed_peims.cache_clear()


@pytest.mark.parametrize("invalid_value", [0, False, "0"])
def test_statewide_comparison_rejects_a_value_on_a_suppressed_cell(invalid_value):
    projection = _Projection(None, (), {FIELD: ("raw", "status")}, None)
    with pytest.raises(ContractError, match="raw/status"):
        projection.value({FIELD: invalid_value, "raw": "-999", "status": "suppressed"}, FIELD, "count")


@pytest.mark.parametrize(("damage", "message"), [
    ("wrong-scope", "acquisition differs"),
    ("duplicate", "duplicate table key"),
    ("campus-prefix", "identities disagree"),
    ("bad-mask", "invalid count token"),
])
def test_invalid_source_cannot_leave_a_candidate(tmp_path, damage, message):
    store, config, digest, _ = fixture_config(tmp_path, damage=damage)
    output = tmp_path / "candidate"
    with pytest.raises(ValueError, match=message):
        materialize_peims_students(store, config=config, config_sha256=digest,
            selections=[(PROGRAMS, YEAR)], output=output)
    assert not output.exists()
    assert not list(tmp_path.glob(".peims-student-*"))


@pytest.mark.parametrize(("damage", "message"), [
    (None, None), ("extra-campus", "one distinct acquisition"),
    ("wrong-aggregation", "differs from its selected scope"),
])
def test_primary_disability_accepts_only_state_and_district(tmp_path, damage, message):
    store, config, digest, _ = fixture_config(tmp_path, damage=damage, dataset=PRIMARY_DISABILITY)
    output = tmp_path / "candidate"
    options = {"config": config, "config_sha256": digest, "selections": [(PRIMARY_DISABILITY, YEAR)], "output": output}
    if message:
        with pytest.raises(ValueError, match=message):
            materialize_peims_students(store, **options)
        assert not output.exists()
        return
    report = materialize_peims_students(store, **options)
    assert not report["release_activation"]
    release = DataRelease(output / "release.json", offline=True)
    release.verify()
    for edition in report["editions"]:
        key = EditionKey.from_dict(edition["key"])
        binding, = release.catalog.resolve_artifacts([(key, FIELD)])
        row, = release.analytical_dataset(binding["artifact"].identity).to_table().to_pylist()
        observed = (row[FIELD], row[f"peims_raw_{FIELD}"], row[f"peims_status_{FIELD}"])
        assert observed == ((0, "0", "observed") if key.scope == "state" else (None, "-999", "suppressed"))
    assert {item["key"]["scope"] for item in report["editions"]} == {"state", "district"}


def test_candidate_does_not_replace_concurrently_created_empty_directory(tmp_path, monkeypatch):
    from teadata import release

    store, config, digest, _ = fixture_config(tmp_path)
    output = tmp_path / "candidate"
    publish = release.publish_release

    def raced(*args, **kwargs):
        result = publish(*args, **kwargs)
        output.mkdir()
        return result

    monkeypatch.setattr(release, "publish_release", raced)
    with pytest.raises(FileExistsError):
        materialize_peims_students(store, config=config, config_sha256=digest,
            selections=[(PROGRAMS, YEAR)], output=output)
    assert list(output.iterdir()) == []
