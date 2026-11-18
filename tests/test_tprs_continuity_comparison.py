"""Synthetic exact native contracts; real publisher goldens remain separate."""

from dataclasses import replace
from hashlib import sha256
from uuid import uuid4

import pytest

from teadata import DataEngine, native_witnesses, tprs_student_continuity as continuity, tprs_students
from teadata.analytical import write_leaf_artifact
from teadata.dataset_metadata import ArtifactRef, DatasetMetadata, EditionKey, ResolvedDataset, STATEWIDE_SINGLETON_GRAIN
from teadata.entities import Campus, District
from teadata.manifest import BuildRecorder, ContractError, canonical_json
from teadata.provenance import ProvenanceStore
from teadata.query import Query
from teadata.release import DataRelease, publish_release
from teadata.statewide import _reviewed_continuity
from teadata.teadata_config import Config, load_config

from test_tprs_student_inventory import full_report
from test_tprs_students import html, receipt


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("continuity-comparison")
    store = ProvenanceStore(root / "provenance")
    selected = load_config().resolve_dataset(continuity.DATASET, 2026, reader_profile="source", exact=True)
    metadata = selected.metadata.to_dict()
    for index, reference in enumerate(metadata["references"]):
        path = root / f"reference-{index}.txt"
        path.write_text("Synthetic fixture reference; not publisher evidence")
        reference_id = store.record_acquisition(path, ResolvedDataset("reference:synthetic", "reference", 2026, 2026, "", DatasetMetadata()))
        reference.update(path=path.name, url=None, reviewed_sha256=sha256(path.read_bytes()).hexdigest(), provenance_ids=[reference_id])
    reports = {}
    for scope in ("campus", "district", "state"):
        rows, identifier = full_report(tprs_students.definition(), scope)
        attrition = next(row for row in rows if row[0] == "Total Student Attrition")
        attrition[1:3] = ["100", "10.0%"] if scope == "state" else ["0", "*" if scope == "district" else "0.0%"]
        if scope == "campus":
            start = rows.index(["Retention Rates by Grade"])
            rows.pop(start + 2)  # A source-absent grade, never inferred zero.
        raw = html(rows)
        acquired = receipt(store, root, raw, scope=scope, identifier=identifier)
        reports[scope] = continuity.parse_continuity_report(raw, scope=scope, identifier=identifier, receipt=acquired)
    original = {scope: EditionKey(tprs_students.DATASET, "2026", EditionKey.source_revision((report.sha256,)),
        scope, scope, STATEWIDE_SINGLETON_GRAIN if scope == "state" else f"one row per {scope}_number")
        for scope, report in reports.items()}
    editions = {scope: continuity._edition(key) for scope, key in original.items()}
    schema = continuity._crosswalk({editions[scope].table: [report.sha256] for scope, report in reports.items()})
    path = root / "columns.json"
    path.write_text(canonical_json(schema) + "\n")
    acquired = store.record_acquisition(path, ResolvedDataset("reference:synthetic-columns", "reference", 2026, 2026, "", DatasetMetadata()))
    next(reference for reference in metadata["references"] if reference["reference_id"] == "columns").update(
        path=path.name, reviewed_sha256=sha256(path.read_bytes()).hexdigest(), provenance_ids=[acquired])
    policy = continuity._policy()
    config = Config.from_dict({"data_sources": {continuity.DATASET: {2026: {"path": "source-only", "metadata": metadata,
        "reader": {"profiles": {"source": {"column_schema": path.name}}}}}},
        "value_policies": {policy.policy_id: policy.to_dict()}})
    config.base_dir = root
    source = config.resolve_dataset(continuity.DATASET, 2026, reader_profile="source", exact=True)
    recorder = BuildRecorder(config, store)
    for scope, report in reports.items():
        table, support = continuity.projection([report])
        key = editions[scope]
        recipe = continuity._recipe(original[scope], continuity.definition())
        contract = recorder.leaf_contract(key, source, native_schema=table.schema.serialize().to_pybytes(),
            columns=table.column_names, key_columns=[] if scope == "state" else [scope + "_number"],
            rows=1, inputs=[report.sha256], recipe=recipe, toolchain={"fixture": "synthetic"}, column_support=support)
        identity = sha256(canonical_json(contract).encode()).hexdigest()
        leaf = root / (identity + ".parquet")
        physical = write_leaf_artifact(leaf, table.to_batches(), contract=contract)
        process = store.record_processing(dataset=continuity.DATASET, inputs=[report.receipt, acquired], outputs={"leaf": leaf},
            processor="synthetic fixture", code_version=recorder.processor_sha256,
            parameters={"recipe": recipe, "toolchain": {"fixture": "synthetic"}, "leaf_contract_sha256": identity})
        artifact = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name, canonical_json(contract), process, "leaf")
        recorder.record_edition(key, source, artifacts=[artifact], acquisition_ids=[report.receipt])
    publish_release(root, recorder.catalog(schema_version=4), schema_version=4)
    return DataRelease(root / "release.json", offline=True), editions


@pytest.fixture
def admitted(prepared, monkeypatch):
    release, keys = prepared
    original = native_witnesses.reviewed_source_additions
    # Admit this exact tiny fixture only; never extend the real registry.
    monkeypatch.setattr(native_witnesses, "reviewed_source_additions",
        lambda pin: ("tprs_continuity", {}) if pin == release.manifest_sha256 else original(pin))
    return release, keys


def test_scope_period_direct_texas_and_protected_values(admitted):
    release, keys = admitted
    engine = DataEngine()
    entities = {"district": District(uuid4(), "Fixture", district_number="001902"),
                "campus": Campus(uuid4(), None, "Fixture", "", False, campus_number="001902001")}
    for scope, entity in entities.items():
        rows = Query([entity], engine).compare_statewide(release=release, local=keys[scope], statewide=keys["state"], school_year=25)
        by_name = {row.metric: row for row in rows}
        assert len(by_name) == (22 if scope == "campus" else 24)
        assert all(row.period == continuity.PERIOD for row in by_name.values())
        count = by_name["attrition_2024_25_count"]
        assert (count.local_value, count.statewide_value, count.difference) == (0, 100, -100)
        percent = by_name["attrition_2024_25_percent"]
        assert percent.statewide_value == 10.0
        if scope == "district":
            assert (percent.local_raw, percent.local_status, percent.local_value, percent.difference_pp) == ("*", "suppressed", None, None)
        else:
            assert percent.difference_pp == -10.0
            missing = by_name["retention_2024_25_special_education_grade_1_percent"]
            assert (missing.local_status, missing.local_raw, missing.local_value, missing.difference) == ("missing", None, None, None)
            assert not any(name.startswith("underreported") for name in by_name)
        with pytest.raises(ContractError, match="Requested school year ends in 2026"):
            Query([entity], engine).compare_statewide(release=release, local=keys[scope], statewide=keys["state"],
                metric="attrition_2024_25_percent", school_year=26)


def test_unreviewed_pin_is_not_inferred_from_dataset_name(prepared):
    release, keys = prepared
    with pytest.raises(ContractError, match="exact reviewed native"):
        _reviewed_continuity(release, keys["district"])


@pytest.mark.parametrize("damage", ["receipt", "column", "status", "value", "reason", "row", "boolean"])
def test_row_contract_rejects_conflicting_source_or_typed_cells(admitted, damage):
    release, keys = admitted
    _, _, _, _, contract = _reviewed_continuity(release, keys["campus"])
    row, = release.table(keys["campus"]).to_arrow().to_pylist()
    field = "attrition_2024_25_count"
    changes = {"receipt": ("tprs_report_receipt", "f" * 64), "column": ("tprs_column_" + field, 99),
               "status": ("tprs_status_" + field, "missing"), "value": (field, 1),
               "reason": ("tprs_reason_" + field, "invented"), "row": ("tprs_row_" + field, None),
               "boolean": (field, False)}
    row.update([changes[damage]])
    with pytest.raises(ContractError):
        contract.value(row, field, "count")


def test_metadata_scope_and_no_leaf_planning(admitted, monkeypatch):
    release, keys = admitted
    with monkeypatch.context() as guard:
        guard.setattr(DataRelease, "artifact", lambda *a, **k: pytest.fail("Planning opened observations"))
        _, _, metadata, _, _ = _reviewed_continuity(release, keys["campus"])
        assert len(metadata.metrics) == 22
    with pytest.raises(ContractError):
        _reviewed_continuity(release, replace(keys["campus"], table="campus-wrong"))


@pytest.mark.parametrize("context", ["not JSON", "{}", '["invented"]', '["Retention Rates by Grade","Retention Rates by Grade"]'])
def test_missing_cell_requires_valid_unpublished_context(admitted, context):
    release, keys = admitted
    contract = _reviewed_continuity(release, keys["campus"])[-1]
    row, = release.table(keys["campus"]).to_arrow().to_pylist()
    row["tprs_unpublished_sections"] = context
    with pytest.raises(ContractError, match="context"):
        contract.value(row, "retention_2024_25_special_education_grade_1_percent", "percent")


@pytest.mark.parametrize("raw,status,reason", [
    ("*", "suppressed", "publisher_small_count_masking"),
    ("-", "not_applicable", "publisher_no_students_in_group"),
    ("?", "invalid", "publisher_statistically_improbable"),
    ("", "missing", "publisher_blank_no_measure_published"),
])
def test_documented_null_states_keep_raw_and_do_not_become_zero(admitted, raw, status, reason):
    release, keys = admitted
    contract = _reviewed_continuity(release, keys["district"])[-1]
    row, = release.table(keys["district"]).to_arrow().to_pylist()
    field = "attrition_2024_25_percent"
    row.update({field: None, "tprs_raw_" + field: raw, "tprs_status_" + field: status, "tprs_reason_" + field: reason})
    assert contract.value(row, field, "percent") == (raw, status, None)


def test_absent_report_and_private_scope_are_distinct(admitted):
    release, keys = admitted
    engine = DataEngine()
    absent = District(uuid4(), "No report", district_number="001903")
    row, = Query([absent], engine).compare_statewide(release=release, local=keys["district"], statewide=keys["state"],
        metric="attrition_2024_25_count")
    assert row.local_status == "unavailable" and row.local_value is None and row.difference is None
    private = Campus(uuid4(), None, "Private", "", False, is_private=True, source_namespace="tefa", source_id="synthetic")
    row, = Query([private], engine).compare_statewide(release=release, local=keys["campus"], statewide=keys["state"],
        metric="attrition_2024_25_count")
    assert row.local_status == "not_applicable" and row.local_value is None and row.difference is None


def test_policy_and_duplicate_acquisition_fail_closed(admitted, monkeypatch):
    release, keys = admitted
    with monkeypatch.context() as guard:
        guard.setattr(release.catalog, "policy", lambda *a, **k: None)
        with pytest.raises(ContractError, match="disclosure policy"):
            _reviewed_continuity(release, keys["district"])
    acquisitions = release.catalog.acquisitions(continuity.DATASET, edition=keys["district"])
    monkeypatch.setattr(release.catalog, "acquisitions", lambda *a, **k: [*acquisitions, *acquisitions])
    with pytest.raises(ContractError, match="repeats a publisher identifier"):
        _reviewed_continuity(release, keys["district"])
