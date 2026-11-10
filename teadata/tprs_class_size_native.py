"""Build an offline district and Texas class-size candidate from retained TPRS reports."""

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

from .analytical import write_leaf_artifact
from .column_schema import source_schema_fingerprint
from .dataset_metadata import STATEWIDE_SINGLETON_GRAIN, ArtifactRef, DatasetMetadata, EditionKey, ResolvedDataset
from .manifest import BuildRecorder, ContractError, canonical_json
from .provenance import ProvenanceStore
from .release import DataRelease, _install_candidate, publish_release
from .teadata_config import load_config_bytes
from .tprs_class_size import CLASS_SIZE_POLICY, CLASS_SIZE_UNIT, _FIELDS
from .tprs_statewide_intake import _reviewed_leaves, project_reviewed_class_sizes
from .tprs_students import EDITION, PERIOD


DATASET = "tprs_class_size"
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


def materialize_class_sizes(release: DataRelease, store: ProvenanceStore, output: Path) -> dict[str, Any]:
    """Write a verified local candidate without changing the active release."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("Class-size candidate requires a new output directory")
    leaves = {leaf.artifact.edition.scope: leaf for leaf in _reviewed_leaves(release)
              if leaf.artifact.edition.scope in _SCOPES}
    reports = {scope: sorted(leaves[scope].reports, key=lambda report: report.identifier) for scope in _SCOPES}
    inputs = {scope: [report.sha256 for report in rows] for scope, rows in reports.items()}
    source_metadata = release.catalog.metadata(leaves["state"].artifact.edition.dataset,
                                               edition=leaves["state"].artifact.edition)
    references = [ref for ref in source_metadata.references if ref.reference_id in {"definitions", "legend"}]
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".tprs-class-size-", dir=output.parent) as directory:
        root = Path(directory)
        stage = root / "bundle"
        stage.mkdir(mode=0o700)
        target = ProvenanceStore(stage / "provenance")
        roots = {item.receipt for rows in reports.values() for item in rows}
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
        projections = {}
        for scope in _SCOPES:
            table, support = project_reviewed_class_sizes(release, store, scope=scope)
            acquisitions = [item.receipt for item in reports[scope]]
            if table["tprs_report_receipt"].to_pylist() != acquisitions or table["tprs_report_sha256"].to_pylist() != inputs[scope]:
                raise ContractError("Class-size projection does not match the exact report ledger")
            key = EditionKey(DATASET, str(EDITION), EditionKey.source_revision(tuple(inputs[scope])),
                             scope, scope, STATEWIDE_SINGLETON_GRAIN if scope == "state" else "one row per district_number")
            recipe = {"operation": "tprs_class_size", "student_attachment": release.manifest_sha256,
                      "student_edition": leaves[scope].artifact.edition.identity}
            contract = recorder.leaf_contract(
                key, source, native_schema=table.schema.serialize().to_pybytes(), columns=table.column_names,
                key_columns=["district_number"] if scope == "district" else [], rows=table.num_rows,
                inputs=inputs[scope], recipe=recipe, toolchain=toolchain, column_support=support,
            )
            encoded = canonical_json(contract)
            leaf = stage / (sha256(encoded.encode()).hexdigest() + ".parquet")
            physical = write_leaf_artifact(leaf, table.to_batches(max_chunksize=512), contract=contract, batch_bytes=8 * 1024**2)
            processing = target.record_processing(
                dataset=DATASET, inputs=[*acquisitions, schema_receipt], outputs={"leaf": leaf},
                processor="teadata.tprs_class_size_native", code_version=recorder.processor_sha256,
                parameters={"recipe": recipe, "leaf_contract_sha256": leaf.stem, "toolchain": toolchain},
                validation={"rows": table.num_rows, "release_activation": False},
            )
            artifact = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name, encoded, processing, "leaf")
            recorder.record_edition(key, source, artifacts=[artifact], acquisition_ids=acquisitions)
            projections[key] = table
        manifest = publish_release(stage, recorder.catalog(schema_version=4), schema_version=4)
        candidate = DataRelease(manifest, offline=True)
        candidate.verify()
        for key, table in projections.items():
            actual = candidate.table(key).to_arrow().select(table.column_names).replace_schema_metadata(None)
            if not table.equals(actual):
                raise ContractError("Class-size native readback differs from the retained report projection")
        result = {"status": "unpublished_native_class_sizes", "manifest_sha256": candidate.manifest_sha256,
                  "scopes": {key.scope: table.num_rows for key, table in projections.items()},
                  "student_attachment": release.manifest_sha256, "release_activation": False}
        (stage / "MATERIALIZATION.json").write_text(canonical_json(result) + "\n", encoding="utf-8")
        _install_candidate(stage, output)
    return result
