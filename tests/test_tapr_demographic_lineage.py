"""Publisher CSV -> preparation -> enrichment -> persisted demographic measures.

Expected values come only from the independently reviewed publisher reference.
The small input files retain exact records from the corrected publisher revision.
All receipts created here describe local fixtures, never original HTTP events.
"""

from copy import deepcopy
from dataclasses import replace
import csv
from hashlib import sha256
from importlib.resources import files
import io
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import pandas as pd
import pyarrow.parquet as pq
import pytest

from teadata import Campus, DataEngine, District
from teadata.dataset_metadata import DatasetMetadata, ResolvedDataset
from teadata.enrichment import campuses, districts
from teadata.entity_store import EntityStore
from teadata.manifest import BuildRecorder, canonical_json
from teadata.persistence import create_engine, create_sessionmaker, ensure_schema, export_dataengine
from teadata.provenance import ProvenanceStore
from teadata.scripts.build_tapr_profiles import prepare_profiles
from teadata.teadata_config import Config, load_config
from teadata.verification import _publisher_reference_for_release


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = json.loads(files("teadata").joinpath("reference_cases/publisher_2025_canonical_stud.json").read_text())
CANONICAL_RELEASE_INPUTS = {
    entry["release_input"]["dataset"]: {"artifact": {"sha256": entry["release_input"]["sha256"]}}
    for entry in CANONICAL["sources"].values()
}
REFERENCE = _publisher_reference_for_release(CANONICAL_RELEASE_INPUTS)
CASES = {
    "district": next(c for c in REFERENCE["cases"] if c["case_id"] == "austin-isd-2025"),
    "campus": next(c for c in REFERENCE["cases"] if c["case_id"] == "highland-park-race-populations-2025"),
}


def test_canonical_goldens_require_both_exact_prepared_inputs():
    historical = json.loads(files("teadata").joinpath("reference_cases/publisher_2025.json").read_text())
    for missing in CANONICAL_RELEASE_INPUTS:
        mixed = {key: value for key, value in CANONICAL_RELEASE_INPUTS.items() if key != missing}
        assert _publisher_reference_for_release(mixed) == historical


def demographic_facts(scope):
    return [fact for fact in CASES[scope]["facts"]
            if fact["source"]["source_id"] == f"tapr_{scope}_csv"]


def fixture_rows(scope):
    source = REFERENCE["sources"][f"tapr_{scope}_csv"]
    selection = source["fixture_selection"]
    raw = (ROOT / selection["path"]).read_bytes()
    assert (sha256(raw).hexdigest(), len(raw)) == (selection["sha256"], selection["size_bytes"])
    assert selection["parent_sha256"] == source["sha256"]
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
    assert len(rows) == 3 and len({len(row) for row in rows}) == 1
    assert rows[2][0] == CASES[scope]["entity"]["number"]
    for fact in demographic_facts(scope):
        locator = fact["source"]
        column = locator["column_1_based"] - 1
        assert [row[column] for row in rows] == [locator["header_row1"], locator["code"], locator["raw_token"]]
        assert locator["record_row_1_based"] == selection["logical_records_1_based"][-1]
    return raw, rows


def local_schema(store, directory, document, filename):
    path = directory / filename
    path.write_text(canonical_json(document) + "\n")
    identifier = store.record_acquisition(path, ResolvedDataset(
        "reference:demographic_fixture_schema", "reference", 2025, 2025, str(path),
        DatasetMetadata(title="Local bounded demographic fixture contract")))
    return {"reference_id": "schema", "role": "schema", "path": path.name,
            "reviewed_sha256": sha256(path.read_bytes()).hexdigest(), "provenance_ids": [identifier]}


def source_config(tmp_path, *, old_destinations=False):
    original = load_config()
    resource = json.loads((ROOT / "teadata/schemas/tapr--2025--student-profile-crosswalk.json").read_text())
    store = ProvenanceStore(tmp_path / "source-history")
    entries, documents = {}, {}
    for scope in CASES:
        dataset = f"{scope}_tapr_student_profile"
        selected = deepcopy(next(d for d in resource["datasets"] if d["dataset"] == dataset))
        original_selected = deepcopy(selected)
        document = {**resource, "datasets": [selected]}
        raw, _ = fixture_rows(scope)
        path = tmp_path / f"{scope}.csv"
        path.write_bytes(raw)
        table, = selected["tables"]
        # Only the bounded input identity and extent change in the normal test.
        table["input_sha256"] = sha256(raw).hexdigest()
        table["layout"]["last_data_record"] = 3
        restored = deepcopy(selected)
        restored["tables"][0]["input_sha256"] = original_selected["tables"][0]["input_sha256"]
        restored["tables"][0]["layout"]["last_data_record"] = original_selected["tables"][0]["layout"]["last_data_record"]
        assert restored == original_selected
        if old_destinations:
            # Explicit counterfactual: restore the two original destinations.
            for field in table["fields"]:
                if field["code"] in {scope[0].upper() + "PNTINDC", scope[0].upper() + "PNTINDP"}:
                    field["field"] = field["field"].replace("american_indian", "asian")
                elif field["code"] in {scope[0].upper() + "PNTASIC", scope[0].upper() + "PNTASIP"}:
                    unit = field["field"].rsplit("_", 1)[-1]
                    field["field"] = f"{scope}_2024_25_asian_students_{unit}"
        observation = store.record_acquisition(path, ResolvedDataset(
            dataset, "data_sources", 2025, 2025, str(path),
            DatasetMetadata(title="Exact three-record public CSV fixture", version="2025")),
            parameters={"fixture_selection": REFERENCE["sources"][f"tapr_{scope}_csv"]["fixture_selection"]})
        schema = local_schema(store, tmp_path, document, f"{scope}-source-schema.json")
        metadata = replace(original.resolve_dataset(dataset, 2025).metadata,
                           provenance_ids=(), references=(), source_url=None).to_dict()
        metadata["references"] = [schema, {
            "reference_id": "csv_headers", "role": "source_notes", "path": path.name,
            "provenance_ids": [observation], "reviewed_sha256": table["input_sha256"]}]
        entries[dataset] = {2025: {"path": path.name, "metadata": metadata, "reader": {"profiles": {
            "source": {"column_schema": schema["path"], "tapr_source": {
                "scope": scope, "tables": {"student_profile": path.name}}}}}}}
        documents[scope] = document
    path = tmp_path / "source-config.json"
    path.write_text(canonical_json({"data_sources": entries}) + "\n")
    return load_config(path), store, documents


def prepared_config(tmp_path, documents, prepared, history):
    original = load_config()
    entries = {}
    for scope, document in documents.items():
        dataset = f"{scope}_tapr_student_profile"
        path = prepared / f"{scope}.parquet"
        document = deepcopy(document)
        selected, = document["datasets"]
        # This separate fixture contract now selects the real prepared bytes.
        del selected["source_only"]
        selected["prepared_input_sha256"] = sha256(path.read_bytes()).hexdigest()
        schema = local_schema(history, tmp_path, document, f"{scope}-prepared-schema.json")
        metadata = replace(original.resolve_dataset(dataset, 2025).metadata,
                           value_policy="numeric", provenance_ids=(), references=(), source_url=None).to_dict()
        metadata["references"] = [schema]
        observation = history.record_acquisition(path, ResolvedDataset(
            dataset, "data_sources", 2025, 2025, str(path), DatasetMetadata.from_dict(metadata)))
        metadata["provenance_ids"] = [observation]
        entries[dataset] = {2025: {"path": str(path), "metadata": metadata,
                                  "reader": {"column_schema": schema["path"]}}}
    bindings = {key: deepcopy(original.resolved_metric_bindings()[key]) for key in ("enrollment", "tapr.*")}
    for binding in bindings.values():
        binding["dataset"] = "{scope}_tapr_student_profile"
    config = {"data_sources": entries, "value_policies": {"numeric": original.value_policies["numeric"].to_dict()},
              "metric_bindings": bindings}
    path = tmp_path / "prepared-config.json"
    path.write_text(canonical_json(config) + "\n")
    return path, load_config(path)


def test_public_demographics_survive_preparation_enrichment_and_entity_store(tmp_path):
    cfg, history, documents = source_config(tmp_path)
    prepared = tmp_path / "prepared"
    result = prepare_profiles(cfg, edition="2025", family="student_profile", scopes=["district", "campus"],
                              input_dir=None, output_dir=prepared, reference_store=history)
    archive = ProvenanceStore(prepared / "provenance")
    for scope in CASES:
        info = result["scopes"][scope]
        assert info["rows"] == 1 and info["raw_string_readback_equal"]
        table = pq.read_table(prepared / f"{scope}.parquet")
        assert [column[0].as_py() for column in table.columns] == fixture_rows(scope)[1][2]
        closure = archive.verify_record_closure([info["processing_record_id"]])
        assert all(record["method"] == "local_observation" and record["downloaded_at"] is None
                   for record in closure.values() if record["kind"] == "acquisition")

    path, configured = prepared_config(tmp_path, documents, prepared, archive)
    repo = DataEngine()
    district = District(id=uuid5(NAMESPACE_URL, "227901"), name="AUSTIN ISD", district_number="227901")
    campus = Campus(id=uuid5(NAMESPACE_URL, "227901119"), district_id=district.id,
                    name="HIGHLAND PARK EL", campus_number="227901119", charter_type="", is_charter=False)
    repo.add_district(district)
    repo.add_campus(campus)
    assert not district.meta and not campus.meta
    with BuildRecorder(configured, archive) as recorder:
        for scope, module in (("district", districts), ("campus", campuses)):
            apply = getattr(module, f"enrich_{'districts' if scope == 'district' else 'campuses'}_from_config")
            assert apply(repo, str(path), f"{scope}_tapr_student_profile", 2025) == (2025, 1)
    repo.catalog = recorder.catalog()
    for scope in CASES:
        processing = archive.read_record(result["scopes"][scope]["processing_record_id"])
        assert repo.catalog.datasets[f"{scope}_tapr_student_profile"]["artifact"]["sha256"] == (
            processing["outputs"]["prepared_source"]["sha256"])
    database = tmp_path / "entities.sqlite"
    engine = create_engine(f"sqlite:///{database}")
    try:
        ensure_schema(engine)
        with create_sessionmaker(engine).begin() as session:
            export_dataengine(repo, session, include_meta_entries=False)
    finally:
        engine.dispose()
    checked = 0
    with EntityStore(database, immutable=True) as store:
        for scope, case in CASES.items():
            getter = store.get_district if scope == "district" else store.get_campus
            entity = SimpleNamespace(**getter(case["entity"]["number"], include_meta=True, cache=False))
            headers = {row["field"]: row for row in store.catalog.header_records(f"{scope}_tapr_student_profile")}
            for fact in demographic_facts(scope):
                expected = fact["expected"]
                value = store.catalog.measurement(entity, expected["metric"], scope=scope)
                assert value.value == expected["value"] and type(value.value) is type(expected["value"]), fact
                assert value.status.value == expected["status"] and value.unit == expected["unit"], fact
                assert value.raw == fact["source"]["raw_token"], fact
                assert value.source["period"] == expected["period"] and value.source["field"] == fact["field"], fact
                assert value.source["dataset"] == f"{scope}_tapr_student_profile" and value.source["input_binding"], fact
                assert headers[fact["field"]]["code"] == fact["source"]["code"]
                assert headers[fact["field"]]["label"] == fact["source"]["header_row1"]
                checked += 1
    assert checked == 60


@pytest.mark.parametrize("scope", CASES)
def test_canonical_prepared_reader_needs_no_renames_for_real_xlsx(tmp_path, scope):
    configured = load_config()
    dataset = f"{scope}_tapr_student_staff_profile"
    source = configured.resolve_dataset(dataset, 2025)
    assert "columns" not in source.reader
    raw, rows = fixture_rows(scope)
    resource = json.loads((ROOT / "teadata/schemas/tapr--2025--student-profile-crosswalk.json").read_text())
    selected = deepcopy(next(d for d in resource["datasets"] if d["dataset"] == f"{scope}_tapr_student_profile"))
    table, = selected["tables"]
    canonical = [field["field"] for field in table["fields"]]
    path = tmp_path / "canonical-prepared.xlsx"
    pd.DataFrame([rows[2]], columns=canonical).to_excel(path, index=False)
    selected = {"dataset": dataset, "edition": "2025", "tables": [table],
                "prepared_input_sha256": sha256(path.read_bytes()).hexdigest()}
    table["input_sha256"] = sha256(raw).hexdigest()
    table["layout"]["last_data_record"] = 3
    document = {**resource, "datasets": [selected]}
    schema = local_schema(ProvenanceStore(tmp_path / "canonical-history"), tmp_path, document, "canonical-schema.json")
    cfg = Config.from_dict({"data_sources": {dataset: {2025: {"path": str(path),
        "reader": {"dtype": "str", "keep_default_na": False, "column_schema": schema["path"]},
        "metadata": {"version": "2025", "references": [schema]}}}}})
    cfg.base_dir = tmp_path
    year, frame = cfg.load_df(dataset, 2025)
    assert year == 2025 and list(frame.columns) == canonical
    assert frame.iloc[0].tolist() == rows[2]
    for fact in demographic_facts(scope):
        assert frame.iloc[0][fact["field"]] == fact["source"]["raw_token"]
    assert all(f"{scope}_2024_25_asian_students_{unit}" not in frame for unit in ("count", "percent"))


@pytest.mark.parametrize("scope", CASES)
def test_original_crosswalk_destinations_fail_the_independent_asian_oracle(tmp_path, scope):
    cfg, _, _ = source_config(tmp_path, old_destinations=True)
    _, frame = cfg.load_df(f"{scope}_tapr_student_profile", 2025, reader_profile="source")
    fact = next(f for f in demographic_facts(scope) if f["field"] == f"{scope}_2025_student_enrollment_asian_count")
    with pytest.raises(AssertionError):
        assert int(frame.iloc[0][fact["field"]]) == fact["expected"]["value"]
