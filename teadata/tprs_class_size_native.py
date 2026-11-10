"""Build offline native class-size candidates from retained TPRS reports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
from typing import Any

from pyarrow.feather import read_table, write_feather

from .analytical import write_leaf_artifact
from .column_schema import source_schema_fingerprint
from .dataset_metadata import ArtifactRef, DatasetMetadata, EditionKey, ResolvedDataset
from .manifest import BuildRecorder, ContractError, canonical_json
from .provenance import ProvenanceStore
from .release import DataRelease, _install_candidate, publish_release
from .teadata_config import load_config_bytes
from .tprs_class_size import CLASS_SIZE_POLICY, CLASS_SIZE_UNIT, _COLUMNS, _FIELDS, _TITLE, read_class_size_report
from .tprs_statewide_intake import (
    DATASET as STUDENT_DATASET,
    REVIEWED_CAMPUS_STUDENT_MANIFEST,
    ReportSelection,
    _project_class_size_reports,
    _reviewed_leaves,
    _verify_campus_census,
)
from .tprs_students import EDITION, PERIOD, _student_workbook_cells, definition


DATASET = "tprs_class_size"
REVIEWED_CLASS_SIZE_MANIFEST = "1512f0ef1a8b1df41f969c683c3ec1dab0d607499dddb589127c6f5a41b9af4e"
_STUDENT_MANIFEST_NAME = "tprs-student-2026-campus-complete.release.json"
_SCOPES = ("district", "state")


def _crosswalk(inputs: Mapping[str, list[str]]) -> dict[str, Any]:
    labels = [(label, field) for group in _FIELDS.values() for label, field in group.items()]
    fields = [
        {"position": index, "code": label, "label": label, "field": field, "dictionary_status": "not_documented",
         "publisher_type": None, "publisher_description": None, "disposition": "fact"}
        for index, (label, field) in enumerate(labels, 1)
    ]
    source_labels = [label for label, _ in labels]
    schema = source_schema_fingerprint(source_labels, source_labels)
    return {"schema_version": 2, "mapping_version": 1, "datasets": [
        {"dataset": DATASET, "edition": str(EDITION), "source_only": True, "tables": [
            {"table": scope, "header_identity": "position", "input_sha256s": hashes,
             "metric_period": PERIOD.to_dict(), "source_schema_sha256": schema, "fields": fields}
            for scope, hashes in inputs.items()
        ]}
    ]}


def class_size_campus_edition(student: EditionKey) -> EditionKey:
    """Translate an exact 2025–26 Student campus partition to its class-size edition."""
    if (student.dataset, student.edition, student.scope, student.table, student.grain) != (
        STUDENT_DATASET, str(EDITION), "campus", "campus", "one row per campus_number",
    ):
        raise ValueError("Expected a 2025–26 TPRS Student campus edition key")
    return replace(student, dataset=DATASET, table=f"campus-{student.identity}")


def _class_size_edition(student: EditionKey) -> EditionKey:
    return class_size_campus_edition(student) if student.scope == "campus" else replace(student, dataset=DATASET)


def _reviewed_class_size_inputs(release: DataRelease, store: ProvenanceStore):
    if not release.offline or release.remote or release.manifest_sha256 != REVIEWED_CLASS_SIZE_MANIFEST:
        raise ContractError("Class-size qualification requires the exact reviewed local release")
    index = release.catalog.to_dict()
    if index["schema_version"] != 4:
        raise ContractError("Class-size qualification requires the reviewed native edition catalog")
    student = DataRelease(release.path.with_name(_STUDENT_MANIFEST_NAME),
                          expected_sha256=REVIEWED_CAMPUS_STUDENT_MANIFEST, offline=True)
    leaves = _reviewed_leaves(student)
    _verify_campus_census(leaves, store)
    expected = {_class_size_edition(leaf.artifact.edition): leaf for leaf in leaves}
    artifacts = [release.catalog.artifact_ref(identifier) for identifier in index["artifacts"]]
    if len(artifacts) != len(expected) or {artifact.edition for artifact in artifacts} != expected.keys():
        raise ContractError("Class-size editions differ from the exact Student Information partitions")
    bindings = {}
    roots = {witness["xlsx_receipt"] for witness in definition()["witnesses"].values()}
    for artifact in artifacts:
        key = artifact.edition
        leaf = expected[key]
        reports = sorted(leaf.reports, key=lambda item: item.identifier)
        metadata = release.catalog.metadata(key.dataset, edition=key)
        schemas = [ref for ref in metadata.references if ref.reference_id == "columns"]
        if len(schemas) != 1 or len(schemas[0].provenance_ids) != 1 or not schemas[0].reviewed_sha256:
            raise ContractError("Class-size leaf lacks its retained field bindings")
        schema_receipt, = schemas[0].provenance_ids
        roots.update({artifact.processing_id, *(record for ref in metadata.references for record in ref.provenance_ids)})
        bindings[key] = artifact, leaf, reports, schema_receipt, schemas[0].reviewed_sha256
    graph = store.verify_record_closure(sorted(roots))
    for key, (artifact, leaf, reports, schema_receipt, schema_sha256) in bindings.items():
        record = graph[artifact.processing_id]
        schema = graph[schema_receipt]
        manifest_ids = schema["source_metadata"]["provenance_ids"]
        if (len(manifest_ids) != 1 or graph[manifest_ids[0]]["dataset"] != "reference:tprs_class_size_student_attachment"
                or graph[manifest_ids[0]]["artifact"]["sha256"] != student.manifest_sha256
                or schema["dataset"] != "reference:tprs_class_size_crosswalk"
                or schema["artifact"]["sha256"] != schema_sha256
                or record != release.catalog._processing_record(artifact)
                or record["processor"] != "teadata.tprs_class_size_native" or artifact.output != "leaf"
                or record["parameters"]["recipe"] != {"operation": "tprs_class_size",
                    "student_attachment": student.manifest_sha256, "student_edition": leaf.artifact.edition.identity}
                or record["inputs"] != [*(item.receipt for item in reports), schema_receipt,
                                         *(receipt for _, receipt in leaf.campus_rosters)]
                or artifact.contract["inputs"] != [item.sha256 for item in reports]):
            raise ContractError(f"Class-size processing lineage differs for {key.identity}")
    return {key: (artifact, reports) for key, (artifact, _, reports, _, _) in bindings.items()}, sorted(graph)


def retained_class_size_evidence(release: DataRelease, store: ProvenanceStore) -> list[str]:
    """Verify the exact census, processing and workbook evidence before receipt reuse."""
    return _reviewed_class_size_inputs(release, store)[1]


def _verify_class_size_workbooks(store: ProvenanceStore, reports: Mapping[tuple[str, str], ReportSelection]) -> dict[str, Any]:
    results = {}
    for scope, witness in definition()["witnesses"].items():
        identifier = witness["identifier"]
        selected = reports.get((scope, identifier))
        if selected is None:
            raise ContractError(f"Class-size census lacks the named {scope} workbook witness")
        report = read_class_size_report(store, selected.receipt, scope=scope, identifier=identifier,
                                        expected_sha256=selected.sha256)
        cells = _student_workbook_cells(store, scope, witness)
        headings = [row for (row, column), value in cells.items() if column == "A" and value == _TITLE]
        if len(headings) != 1 or [cells.get((headings[0], chr(ord("B") + index)))
                                  for index in range(len(_COLUMNS[scope]))] != _COLUMNS[scope]:
            raise ContractError(f"Class-size workbook headings differ for {scope} {identifier}")
        workbook = {}
        section = ""
        for row in sorted(row for row, column in cells if column == "A" and row > headings[0]):
            label = cells[row, "A"]
            if label in _FIELDS:
                section = label
                continue
            field = _FIELDS.get(section, {}).get(label)
            if field is None or field in workbook or (row, "B") not in cells:
                raise ContractError(f"Class-size workbook row differs for {scope} {identifier}")
            workbook[field] = cells[row, "B"]
        if workbook != {cell.field: cell.measurement.raw for cell in report.cells}:
            raise ContractError(f"Class-size workbook differs from the census report for {scope} {identifier}")
        results[scope] = {"identifier": identifier, "local_cells": len(workbook),
                          "report_receipt": report.receipt, "workbook_receipt": witness["xlsx_receipt"]}
    return results


def qualify_class_sizes(release: DataRelease, store: ProvenanceStore) -> dict[str, Any]:
    """Replay every reviewed class-size leaf and compare named publisher workbook cells."""
    bindings, evidence = _reviewed_class_size_inputs(release, store)
    release.verify()
    selected = {(report.scope, report.identifier): report for _, reports in bindings.values() for report in reports}
    witnesses = _verify_class_size_workbooks(store, selected)
    scopes: dict[str, dict[str, int]] = {}
    for key, (artifact, reports) in bindings.items():
        expected, support = _project_class_size_reports(store, reports)
        physical = release.analytical_dataset(artifact.identity).to_table().replace_schema_metadata(None)
        actual = release.table(key).to_arrow().replace_schema_metadata(None)
        if (artifact.contract["column_support"] != support or not physical.equals(expected, check_metadata=True)
                or not actual.select(expected.column_names).equals(expected, check_metadata=True)):
            raise ContractError(f"Class-size native rows, values, statuses or lineage differ for {key.identity}")
        summary = scopes.setdefault(key.scope, {"leaves": 0, "reports": 0, "tables": 0})
        summary["leaves"] += 1
        summary["reports"] += expected.num_rows
        summary["tables"] += sum(expected["class_size_table_present"].to_pylist())
    return {"status": "PASS_REVIEWED_NATIVE", "manifest_sha256": release.manifest_sha256,
            "build_id": release.build_id, "structural_integrity": True,
            "student_attachment": REVIEWED_CAMPUS_STUDENT_MANIFEST, "retained_records": len(evidence),
            "html_replay": scopes, "publisher_workbook_witnesses": witnesses}


def materialize_class_sizes(
    release: DataRelease, store: ProvenanceStore, output: Path, *, include_campus: bool = False,
) -> dict[str, Any]:
    """Write a verified local candidate without changing the active release."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("Class-size candidate requires a new output directory")
    if include_campus and release.manifest_sha256 != REVIEWED_CAMPUS_STUDENT_MANIFEST:
        raise ContractError("Campus class sizes require the complete reviewed Student campus attachment")
    leaves = _reviewed_leaves(release)
    if include_campus:
        _verify_campus_census(leaves, store)
    selections = {}
    inputs = {}
    for leaf in leaves:
        student = leaf.artifact.edition
        if student.scope not in _SCOPES and not include_campus:
            continue
        key = _class_size_edition(student)
        reports = sorted(leaf.reports, key=lambda report: report.identifier)
        hashes = [item.sha256 for item in reports]
        if key.revision_sha256 != EditionKey.source_revision(tuple(hashes)) or key.table in inputs:
            raise ContractError("Class-size partition differs from its exact Student report ledger")
        selections[key] = (leaf, reports)
        inputs[key.table] = hashes
    state, = (leaf.artifact.edition for leaf in leaves if leaf.artifact.edition.scope == "state")
    source_metadata = release.catalog.metadata(state.dataset, edition=state)
    references = [ref for ref in source_metadata.references if ref.reference_id in {"definitions", "legend"}]
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".tprs-class-size-", dir=output.parent) as directory:
        root = Path(directory)
        stage = root / "bundle"
        stage.mkdir(mode=0o700)
        target = ProvenanceStore(stage / "provenance")
        roots = {item.receipt for _, reports in selections.values() for item in reports}
        roots.update(receipt for leaf, _ in selections.values() for _, receipt in leaf.campus_rosters)
        roots.update(identifier for ref in references for identifier in ref.provenance_ids)
        for identifier in sorted(roots):
            target.copy_record_from(store, identifier)
        manifest_receipt = target.record_acquisition(release.path, ResolvedDataset(
            "reference:tprs_class_size_student_attachment", "reference", EDITION, EDITION, "",
            DatasetMetadata(title="Exact reviewed TPRS Student Information attachment", version=str(EDITION)),
        ))
        if target.read_record(manifest_receipt)["artifact"]["sha256"] != release.manifest_sha256:
            raise ContractError("Reviewed Student Information manifest bytes changed")
        schema = root / "class-size-crosswalk.json"
        schema_bytes = (canonical_json(_crosswalk(inputs)) + "\n").encode()
        schema.write_bytes(schema_bytes)
        schema_receipt = target.record_acquisition(schema, ResolvedDataset(
            "reference:tprs_class_size_crosswalk", "reference", EDITION, EDITION, "",
            DatasetMetadata(title="Class-size field bindings", version=str(EDITION), provenance_ids=(manifest_receipt,)),
        ))
        fields = [field for group in _FIELDS.values() for field in group.values()]
        metadata = {
            "title": "2025–26 TPRS Class Size Information", "publisher": source_metadata.publisher,
            "version": str(EDITION), "period": PERIOD.to_dict(), "source_url": source_metadata.source_url,
            "value_policy": CLASS_SIZE_POLICY.policy_id,
            "metrics": {field: {"field": field, "unit": CLASS_SIZE_UNIT} for field in fields},
            "references": [*(ref.to_dict() for ref in references), {
                "reference_id": "columns", "role": "schema", "title": "Class-size field bindings",
                "path": schema.name, "status": "available", "reviewed_sha256": sha256(schema_bytes).hexdigest(),
                "provenance_ids": [schema_receipt],
            }],
        }
        policy = replace(CLASS_SIZE_POLICY, reference_id="legend")
        config = load_config_bytes(canonical_json({
            "value_policies": {policy.policy_id: policy.to_dict()},
            "data_sources": {DATASET: {str(EDITION): {
                "path": "source-only-class-size", "metadata": metadata,
                "reader": {"profiles": {"source": {"column_schema": schema.name}}},
            }}},
        }).encode(), path=root / "config.yaml")
        source = config.resolve_dataset(DATASET, EDITION, reader_profile="source", exact=True)
        recorder = BuildRecorder(config, target)
        toolchain = {"python": sys.version, "pyarrow": version("pyarrow"), "platform": platform.platform()}
        scopes: dict[str, int] = {}
        for key, (student_leaf, reports) in selections.items():
            table, support = _project_class_size_reports(store, reports)
            acquisitions = [item.receipt for item in reports]
            hashes = inputs[key.table]
            if table["tprs_report_receipt"].to_pylist() != acquisitions or table["tprs_report_sha256"].to_pylist() != hashes:
                raise ContractError("Class-size projection does not match the exact report ledger")
            recipe = {"operation": "tprs_class_size", "student_attachment": release.manifest_sha256,
                      "student_edition": student_leaf.artifact.edition.identity}
            contract = recorder.leaf_contract(
                key, source, native_schema=table.schema.serialize().to_pybytes(), columns=table.column_names,
                key_columns=[key.scope + "_number"] if key.scope != "state" else [], rows=table.num_rows,
                inputs=hashes, recipe=recipe, toolchain=toolchain, column_support=support,
            )
            encoded = canonical_json(contract)
            leaf = stage / (sha256(encoded.encode()).hexdigest() + ".parquet")
            physical = write_leaf_artifact(leaf, table.to_batches(max_chunksize=512), contract=contract, batch_bytes=8 * 1024**2)
            processing = target.record_processing(
                dataset=DATASET, inputs=[*acquisitions, schema_receipt,
                                        *(receipt for _, receipt in student_leaf.campus_rosters)], outputs={"leaf": leaf},
                processor="teadata.tprs_class_size_native", code_version=recorder.processor_sha256,
                parameters={"recipe": recipe, "leaf_contract_sha256": leaf.stem, "toolchain": toolchain},
                validation={"rows": table.num_rows, "release_activation": False},
            )
            artifact = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name, encoded, processing, "leaf")
            recorder.record_edition(key, source, artifacts=[artifact], acquisition_ids=acquisitions)
            write_feather(table, root / f"{key.identity}.arrow")
            scopes[key.scope] = scopes.get(key.scope, 0) + table.num_rows
            del table
        manifest = publish_release(stage, recorder.catalog(schema_version=4), schema_version=4)
        candidate = DataRelease(manifest, offline=True)
        candidate.verify()
        for key in selections:
            table = read_table(root / f"{key.identity}.arrow")
            actual = candidate.table(key).to_arrow().select(table.column_names).replace_schema_metadata(None)
            if not table.equals(actual):
                raise ContractError("Class-size native readback differs from the retained report projection")
        result = {"status": "unpublished_native_class_sizes", "manifest_sha256": candidate.manifest_sha256,
                  "scopes": scopes,
                  "student_attachment": release.manifest_sha256, "release_activation": False}
        (stage / "MATERIALIZATION.json").write_text(canonical_json(result) + "\n", encoding="utf-8")
        _install_candidate(stage, output)
    return result
