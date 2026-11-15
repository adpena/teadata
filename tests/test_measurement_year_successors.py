"""Synthetic successor releases exercise real catalog/planner APIs, not TEA availability."""
import csv
from decimal import Decimal
from hashlib import sha256
from shutil import copyfile
from uuid import uuid4

import pyarrow as pa
import pytest

from teadata import DataEngine, District
from teadata.column_schema import source_schema_fingerprint
from teadata.dataset_metadata import (
    ArtifactRef, DataPeriod, DatasetMetadata, EditionKey, NativeSourceRef,
    ResolvedDataset, STATEWIDE_SINGLETON_GRAIN,
)
from teadata.analytical import write_leaf_artifact
from teadata.manifest import BuildRecorder, ContractError, canonical_json, file_identity
from teadata.provenance import ProvenanceStore
from teadata.query import Query
from teadata.release import DataRelease, using_release
from teadata.teadata_config import Config

from test_catalog_projections import attach_observation_child
from test_catalog_projections import selected_catalog as _selected_catalog

# Preserve the shared pytest fixture while avoiding imported-name shadowing.
selected_catalog = _selected_catalog


ENROLLMENT = "tprs_student_membership_enrollment_reports"
COUNT = "enrollment_total_students_count"
CHRONIC = "tprs_attendance_chronic_absence"
RATE = "chronic_absence_all_students_percent"
ECD = "chronic_absence_economically_disadvantaged_percent"
SOF = "summary_of_finances_attendance"
REFINED = "inputs_lpe_ada_tot_refined"


def _child(root, label, dataset, scope, years, rows, *, edition="2026", table_name=None):
    """Write at most four synthetic rows with real immutable metadata and leaves."""
    assert len(rows) <= 4
    stage = root / label
    stage.mkdir()
    store = ProvenanceStore(stage / "provenance")
    identifier = "state_identifier" if scope == "state" else scope + "_number"
    names = [identifier, *years]
    raw = stage / "source.csv"
    with raw.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[name.upper() for name in names])
        writer.writeheader()
        writer.writerows({name.upper(): value for name, value in row.items()} for row in rows)
    raw_sha = file_identity(raw)["sha256"]
    unit = "count" if dataset == ENROLLMENT else "percent" if dataset == CHRONIC else "attendance_equivalent"
    fields = [{"position": index, "code": name.upper(), "label": "Synthetic " + name,
               "field": name, "disposition": "identity" if name == identifier else "fact",
               "publisher_type": None, "publisher_description": None,
               "dictionary_status": "not_applicable" if name == identifier else "not_documented"}
              for index, name in enumerate(names, 1)]
    table_name = table_name or "synthetic_" + label
    mapping = {"table": table_name, "input_sha256": raw_sha,
               "source_schema_sha256": source_schema_fingerprint(
                   [field["label"] for field in fields], [field["code"] for field in fields]),
               "fields": fields, "metric_period": {"kind": "unknown"}}
    schema = stage / "columns.json"
    schema.write_text(canonical_json({"schema_version": 2, "mapping_version": 1,
        "datasets": [{"dataset": dataset, "edition": edition, "source_only": True, "tables": [mapping]}]}))
    masking = stage / "masking.txt"
    masking.write_text("Synthetic fixture only: * suppressed; . missing; no inferred bounds.\n")
    references = []
    for name, path, role in (("columns", schema, "schema"), ("masking", masking, "masking_methodology")):
        reference = store.record_acquisition(path, ResolvedDataset(
            "reference:synthetic", "reference", None, None, str(path),
            DatasetMetadata(title="Synthetic successor " + name)))
        references.append({"reference_id": name, "role": role, "path": path.name,
                           "reviewed_sha256": file_identity(path)["sha256"],
                           "status": "available", "provenance_ids": [reference]})
    schema_sha = file_identity(schema)["sha256"]
    (stage / (schema_sha + ".json")).write_bytes(schema.read_bytes())
    metadata = {"title": "Synthetic successor; not an official future source", "version": edition,
                "references": references, "value_policy": "synthetic",
                "metrics": {field: {"field": field, "unit": unit,
                            "period": DataPeriod("school_year", year - 1, year).to_dict()}
                            for field, year in years.items()}}
    config = Config.from_dict({"data_sources": {dataset: {int(edition): {
        "path": str(raw), "metadata": metadata,
        "reader": {"profiles": {"source": {"column_schema": schema.name}}}}}},
        "value_policies": {"synthetic": {"reference_id": "masking", "tokens": {
            "*": {"status": "suppressed", "reason": "synthetic masked cell"},
            ".": {"status": "missing", "reason": "synthetic missing cell"}}}}})
    config.base_dir = stage
    source = config.resolve_dataset(dataset, int(edition), exact=True, reader_profile="source")
    recorder = BuildRecorder(config, store)
    acquisition = store.record_acquisition(raw, source)
    arrays = [pa.array([row[identifier] for row in rows], type=pa.string())]
    native_fields = [pa.field(identifier, pa.string(), nullable=False)]
    support = {"fields": {}, "context": []}
    for field in years:
        tokens = [row[field] for row in rows]
        statuses = [{"*": "suppressed", ".": "missing"}.get(token, "observed") for token in tokens]
        kind = pa.int64() if unit == "count" else pa.decimal128(12, 3)
        values = [None if state != "observed" else int(token) if unit == "count" else Decimal(token)
                  for token, state in zip(tokens, statuses, strict=True)]
        for name, array in ((field, pa.array(values, type=kind)),
                            ("raw_" + field, pa.array(tokens, type=pa.string())),
                            ("status_" + field, pa.array(statuses, type=pa.string()))):
            arrays.append(array)
            native_fields.append(pa.field(name, array.type))
        support["fields"][field] = {"raw": "raw_" + field, "status": "status_" + field}
    native = pa.Table.from_arrays(arrays, schema=pa.schema(native_fields))
    key = EditionKey(dataset, edition, EditionKey.source_revision((raw_sha,)), scope, table_name,
                     STATEWIDE_SINGLETON_GRAIN if scope == "state" else "one row per " + identifier)
    recipe = {"operation": "synthetic successor-release regression only"}
    tools = {"fixture": "synthetic"}
    contract = recorder.leaf_contract(key, source, native_schema=native.schema.serialize().to_pybytes(),
        columns=native.column_names, key_columns=[] if scope == "state" else [identifier], rows=len(rows),
        inputs=[raw_sha], recipe=recipe, toolchain=tools, column_support=support)
    identity = sha256(canonical_json(contract).encode()).hexdigest()
    leaf = stage / (identity + ".parquet")
    physical = write_leaf_artifact(leaf, native.to_batches(), contract=contract)
    process = store.record_processing(dataset=dataset, inputs=[acquisition], outputs={"leaf": leaf},
        processor="synthetic successor fixture", code_version=recorder.processor_sha256,
        parameters={"leaf_contract_sha256": identity, "recipe": recipe, "toolchain": tools})
    artifact = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name,
                           canonical_json(contract), process, "leaf")
    recorder.record_edition(key, source, artifacts=[artifact], acquisition_ids=[acquisition])
    catalog = recorder.catalog(schema_version=3)
    catalog.verify_metadata()
    manifest = stage / "native.json"
    manifest.write_text(canonical_json({"schema_version": 3, "catalog": catalog.to_dict()}))
    return DataRelease(manifest, offline=True)


def _parent(root, selected_catalog, children):
    """Attach exact child bytes, retaining adjacent metadata/leaf identities."""
    root.mkdir()
    references = []
    for index, child in enumerate(children):
        resources = [*child.catalog.column_schema_refs(),
                     *(child.catalog.artifact_ref(key) for key in child.catalog.to_dict()["artifacts"])]
        for ref in resources:
            source, target = child.path.parent / ref.location, root / ref.location
            if target.exists():
                assert target.read_bytes() == source.read_bytes()
            else:
                copyfile(source, target)
        name = f"native-{index}.json"
        copyfile(child.path, root / name)
        keys = tuple(EditionKey.from_dict({name: row[name] for name in EditionKey.__dataclass_fields__})
                     for row in child.catalog.editions(include_metadata=False))
        references.append(NativeSourceRef(name, child.manifest_sha256, child.build_id, keys))
    first = DataRelease(root / references[0].location, offline=True)
    return attach_observation_child(selected_catalog, first, additional_refs=references[1:])


def _engine(release, identifiers):
    engine = DataEngine(indexes_enabled=False)
    engine.catalog = release.catalog
    districts = {}
    for number in identifiers:
        district = District(uuid4(), "Synthetic district", district_number=number)
        engine.add_district(district)
        districts[number] = district
    engine.bind_release(release)
    return engine, districts


def test_enrollment_successor_uses_measurement_year_and_preserves_old_pin(tmp_path, selected_catalog, monkeypatch):
    old_child = _child(tmp_path, "enrollment_old", ENROLLMENT, "district", {COUNT: 2026}, [
        {"district_number": "000001", COUNT: "11"}, {"district_number": "000002", COUNT: "0"},
        {"district_number": "000003", COUNT: "*"}, {"district_number": "000005", COUNT: "29"}])
    new_child = _child(tmp_path, "enrollment_new", ENROLLMENT, "district", {COUNT: 2027}, [
        {"district_number": "000001", COUNT: "17"}, {"district_number": "000002", COUNT: "0"},
        {"district_number": "000003", COUNT: "*"}, {"district_number": "000004", COUNT: "."}])
    lagged = _child(tmp_path, "enrollment_lagged", ENROLLMENT, "district", {COUNT: 2025}, [
        {"district_number": "000001", COUNT: "999"}], edition="2099")
    old = _parent(tmp_path / "old", selected_catalog, [old_child])
    new = _parent(tmp_path / "new", selected_catalog, [old_child, new_child, lagged])
    _, old_entities = _engine(old, ["000001", "000005"])
    engine, entities = _engine(new, [f"{number:06d}" for number in range(1, 6)])
    old_namespace, namespace = old_entities["000001"].tprs, entities["000001"].tprs
    with monkeypatch.context() as guard:
        guard.setattr(DataRelease, "artifact", lambda *a, **k: pytest.fail("Metadata opened a data leaf"))
        assert old_namespace.years(COUNT) == (2026,)
        assert namespace.years(COUNT) == (2025, 2026, 2027)
        old_plan, plan = old_namespace.select(COUNT), namespace.select(COUNT)
        assert plan.measurement_period.end_year == 2027 and plan.edition.edition == "2026"
        assert plan.provenance().to_dict()["release"]["manifest_sha256"] == new_child.manifest_sha256
    selected_artifact = plan.release.catalog.resolve_artifacts(((plan.edition, COUNT),))[0]["artifact"]
    selected_path = (plan.release.path.parent / selected_artifact.location).resolve()
    opened = []
    original_artifact = DataRelease.artifact

    def track_artifact(release, role, **kwargs):
        result = original_artifact(release, role, **kwargs)
        if result.suffix == ".parquet":
            opened.append(result.resolve())
        return result

    with monkeypatch.context() as track:
        track.setattr(DataRelease, "artifact", track_artifact)
        assert plan.to_arrow()[COUNT].to_pylist() == [17]
    assert opened and set(opened) == {selected_path}
    with using_release(new):
        assert old_namespace.enrollment_total_students_count == 11
        assert namespace.enrollment_total_students_count == 17
        assert old_plan.to_arrow()[COUNT].to_pylist() == [11]
        query = Query([entities["000001"]], engine) >> ("observations", ENROLLMENT, None, (COUNT,))
        assert query.measurement_period.end_year == 2027 and query.to_arrow()[COUNT].to_pylist() == [17]
        assert entities["000002"].tprs.enrollment_total_students_count == 0
        for number, status in (("000003", "suppressed"), ("000004", "missing")):
            row, = entities[number].tprs.select(COUNT).to_arrow().to_pylist()
            assert row[COUNT] is None and row["status_" + COUNT] == status
        assert old_entities["000005"].tprs.enrollment_total_students_count == 29
        with pytest.raises(ContractError, match="No published rows for district 000005"):
            _ = entities["000005"].tprs.enrollment_total_students_count
        assert entities["000005"].tprs.select(COUNT).to_arrow().num_rows == 0
        with pytest.raises(ContractError, match="Available years"):
            namespace.at(28).select(COUNT)
    assert old_namespace.release is old and namespace.release is new
    assert old.manifest_sha256 != new.manifest_sha256


def test_chronic_successor_keeps_independent_latest_years_and_refuses_mixed_fields(tmp_path, selected_catalog, monkeypatch):
    old_state = _child(tmp_path, "chronic_old", CHRONIC, "state", {RATE: 2025, ECD: 2025}, [
        {"state_identifier": "S", RATE: "18.0", ECD: "22.2"}])
    new_state = _child(tmp_path, "chronic_new", CHRONIC, "state", {RATE: 2026, ECD: 2025}, [
        {"state_identifier": "S", RATE: "17.5", ECD: "21.0"}])
    old_region = _child(tmp_path, "region_old", CHRONIC, "region", {RATE: 2025}, [
        {"region_number": "01", RATE: "18.1"}, {"region_number": "02", RATE: "18.2"}])
    new_region = _child(tmp_path, "region_new", CHRONIC, "region", {RATE: 2026}, [
        {"region_number": "01", RATE: "17.1"}])
    enrollment = _child(tmp_path, "enrollment", ENROLLMENT, "district", {COUNT: 2027}, [
        {"district_number": "000001", COUNT: "17"}])
    sof_old = _child(tmp_path, "sof_old", SOF, "district", {REFINED: 2026}, [
        {"district_number": "000001", REFINED: "100.125"}], table_name="inputs_lpe")
    sof_new = _child(tmp_path, "sof_new", SOF, "district", {REFINED: 2027}, [
        {"district_number": "000001", REFINED: "200.375"}], table_name="inputs_lpe")
    companion_fields = {"inputs_dpe": "inputs_dpe_ada_tot_refined",
                        "outputs_lpe": "outputs_lpe_adj_avg_ada",
                        "outputs_dpe": "outputs_dpe_adj_avg_ada"}
    companions = [_child(tmp_path, "sof_" + table, SOF, "district", {field: 2026}, [
        {"district_number": "000001", field: "300.125"}], table_name=table)
        for table, field in companion_fields.items()]
    old = _parent(tmp_path / "old", selected_catalog, [old_state, old_region, sof_old])
    new = _parent(tmp_path / "new", selected_catalog,
                  [old_state, old_region, sof_old, new_state, new_region, enrollment, sof_new, *companions])
    engine, entities = _engine(new, ["000001"])
    with monkeypatch.context() as guard:
        guard.setattr(DataRelease, "artifact", lambda *a, **k: pytest.fail("Metadata opened a data leaf"))
        old_plan = old.observations(CHRONIC, RATE, scope="state")
        chronic = new.observations(CHRONIC, RATE, scope="state")
        counts = entities["000001"].tprs.select(COUNT)
        refined = Query([entities["000001"]], engine) >> ("observations", SOF, None, (REFINED,))
        assert [counts.measurement_period.end_year, chronic.measurement_period.end_year,
                refined.measurement_period.end_year] == [2027, 2026, 2027]
        assert new.observation_years(CHRONIC, RATE, scope="state") == {RATE: (2025, 2026)}
        assert new.observation_years(SOF, REFINED, *companion_fields.values(), scope="district") == {
            REFINED: (2026, 2027), **{field: (2026,) for field in companion_fields.values()}}
        companion_plans = {table: new.observations(SOF, field, scope="district")
                           for table, field in companion_fields.items()}
        assert refined.edition.table == "inputs_lpe"
        assert all(plan.measurement_period.end_year == 2026 and plan.edition.table == table
                   for table, plan in companion_plans.items())
        with pytest.raises(ContractError, match="share a latest school year"):
            new.observations(SOF, REFINED, *companion_fields.values(), scope="district")
        with pytest.raises(ContractError, match="share a latest school year"):
            new.observations(CHRONIC, RATE, ECD, scope="state")
        with pytest.raises(ContractError, match="Available years"):
            new.observations(CHRONIC, RATE, scope="state", school_year=2027)
        with pytest.raises(ContractError, match="unavailable for district.*Available scopes"):
            entities["000001"].dataset(CHRONIC)
    with using_release(new):
        assert old_plan.measurement_period.end_year == 2025
        assert old_plan.to_arrow()[RATE].to_pylist() == [Decimal("18.0")]
        chronic_rows = chronic.to_arrow()
        assert chronic_rows[RATE].to_pylist() == [Decimal("17.5")]
        assert ECD not in chronic_rows.column_names
        assert refined.to_arrow()[REFINED].to_pylist() == [Decimal("200.375")]
        for table, plan in companion_plans.items():
            assert plan.to_arrow()[companion_fields[table]].to_pylist() == [Decimal("300.125")]
        common = new.observations(CHRONIC, RATE, ECD, scope="state", school_year=2025)
        row, = common.to_arrow().to_pylist()
        assert (row[RATE], row[ECD]) == (Decimal("18.0"), Decimal("22.2"))
        missing_region = new.observations(CHRONIC, RATE, scope="region").query(
            "region_number == @number", params={"number": "02"})
        assert missing_region.measurement_period.end_year == 2026
        assert missing_region.to_arrow().num_rows == 0
        assert old.observations(CHRONIC, RATE, scope="region").query(
            "region_number == @number", params={"number": "02"}).to_arrow()[RATE].to_pylist() == [Decimal("18.2")]
    assert old_plan.release.manifest_sha256 == old_state.manifest_sha256
    assert chronic.release.manifest_sha256 == new_state.manifest_sha256
