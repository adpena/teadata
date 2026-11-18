"""Source-bound 2024–25 continuity cohorts in the 2026 TPRS Student report.

The shared inventory owns physical layout. This separate interpretation owns
only attrition, K–9 retention and district/state underreporting; it never
extends a historical Membership/Enrollment contract.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import errno
from functools import lru_cache
from hashlib import sha256
from importlib.resources import files
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .analytical import write_leaf_artifact
from .column_schema import _unique_object, source_schema_fingerprint
from .dataset_metadata import ArtifactRef, DataPeriod, DatasetMetadata, EditionKey, ReferenceDocument, ResolvedDataset
from .manifest import BuildRecorder, ContractError, canonical_json
from .provenance import ProvenanceStore
from .release import DataRelease, _install_candidate, publish_release
from .teadata_config import Config, _PACKAGED_CONFIG_PATH, _load_packaged_config, load_config_bytes
from .tprs_statewide_intake import ReportSelection, _reviewed_leaves, _verify_campus_census, campus_roster
from .tprs_student_inventory import inventory_student_report
from .tprs_students import _need, _read_student_input, _student_workbook_cells
from .tprs_students import definition as student_definition
from .values import Measurement, ValuePolicy, ValueStatus, measure

DATASET = "tprs_student_continuity"
EDITION = 2026
PERIOD = DataPeriod("school_year", 2024, 2025)
RESOURCE = "methodologies/tprs-student-continuity-2026.json"
CONTRACT_SHA256 = "5875db1689db35522b150ee82485b44cc3f8ad85f56e0ceef7f0c98dd72d71a9"
_POLICY_SHA256 = "d13b4c90563086887f77c4a211813d63977d62aa70ca866579307edfc1ca183e"
_COUNT = re.compile(r"(?:0|[1-9][0-9]*|[1-9][0-9]{0,2}(?:,[0-9]{3})+)")
_PERCENT = re.compile(r"(?:0|[1-9][0-9]*)\.[0-9]%")
_SCOPES = ("campus", "district", "state")


@lru_cache(maxsize=1)
def _configuration(raw: bytes, path: Path) -> Config:
    """Reuse only an identical captured configuration, never a path-only cache."""
    return _load_packaged_config(raw, path)


def _configured_source() -> ResolvedDataset:
    config = _configuration(_PACKAGED_CONFIG_PATH.read_bytes(), _PACKAGED_CONFIG_PATH)
    return config.resolve_dataset(DATASET, EDITION, reader_profile="source", exact=True)


def definition() -> dict[str, Any]:
    """Return detached frozen definitions and independent publisher witnesses."""
    raw = files("teadata").joinpath(RESOURCE).read_bytes()
    _need(sha256(raw).hexdigest() == CONTRACT_SHA256, "Continuity methodology changed")
    source = _configured_source()
    reference, = [ref for ref in source.metadata.references if ref.reference_id == "continuity_grid"]
    _need(reference.reviewed_sha256 == CONTRACT_SHA256, "Continuity configuration methodology differs")
    return json.loads(raw, object_pairs_hook=_unique_object)


def _resolve_contract(contract: Mapping[str, Any] | None) -> Mapping[str, Any]:
    grid = definition() if contract is None else contract
    _need(isinstance(grid, Mapping), "Continuity requires a captured reviewed methodology")
    try:
        identity = sha256((canonical_json(dict(grid)) + "\n").encode()).hexdigest()
    except (TypeError, ValueError) as error:
        raise ContractError("Continuity methodology is not canonical JSON") from error
    _need(identity == CONTRACT_SHA256, "Unsupported continuity interpretation identity")
    return grid


def _student_contract(grid: Mapping[str, Any]) -> Mapping[str, Any]:
    contract = student_definition()
    identity = sha256((canonical_json(contract) + "\n").encode()).hexdigest()
    _need(identity == grid["student_grid_sha256"], "Continuity requires the exact retained Student layout")
    reader = files("teadata").joinpath("tprs_student_inventory.py").read_bytes()
    _need(sha256(reader).hexdigest() == grid["inventory_reader_sha256"], "Continuity inventory reader changed")
    return contract


def semantic_fields(scope: str, contract: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    _need(scope in _SCOPES, "Unsupported continuity scope")
    grid = _resolve_contract(contract)
    return [dict(field) for field in grid["fields"] if scope in field["scopes"]]


def _policy() -> ValuePolicy:
    policy = _configuration(_PACKAGED_CONFIG_PATH.read_bytes(), _PACKAGED_CONFIG_PATH).value_policies["tprs_student_selected_2026"]
    _need(sha256(canonical_json(policy.to_dict()).encode()).hexdigest() == _POLICY_SHA256,
          "Continuity disclosure policy changed")
    return policy


def _measurement(raw: str | None, unit: str, policy: ValuePolicy, reason: str | None = None) -> Measurement:
    if raw is None or raw == "":
        return Measurement(None, ValueStatus.MISSING, raw, reason or "publisher_blank_no_measure_published",
                           policy.policy_id, unit=unit)
    _need(raw != "**", "Complementary disability masking is not documented for continuity groups")
    value = measure(raw, policy, unit=unit, minimum=0, maximum=100 if unit == "percent" else None)
    _need(value.status != ValueStatus.INVALID or raw == "?", "Unknown or out-of-domain continuity token")
    if value.status == ValueStatus.OBSERVED:
        _need((_COUNT if unit == "count" else _PERCENT).fullmatch(raw) is not None,
              "Continuity numeric token differs from the reviewed precision")
        formatted = (format(value.value, ",") if "," in raw else str(value.value)) if unit == "count" else f"{value.value:.1f}%"
        _need(formatted == raw, "Continuity numeric value does not round-trip to its original token")
    return value


@dataclass(frozen=True, slots=True)
class ContinuityCell:
    field: str
    measurement: Measurement
    row: int | None
    column: int | None


@dataclass(frozen=True, slots=True)
class ContinuityReport:
    scope: str
    identifier: str
    receipt: str
    sha256: str
    cells: tuple[ContinuityCell, ...]
    unpublished_sections: tuple[str, ...]
    contract_sha256: str


def parse_continuity_report(raw: bytes, *, scope: str, identifier: str, receipt: str = "",
                            contract: Mapping[str, Any] | None = None) -> ContinuityReport:
    """Interpret exact local inventory coordinates, never embedded benchmarks."""
    grid = _resolve_contract(contract)
    inventory = inventory_student_report(raw, scope=scope, identifier=identifier, contract=_student_contract(grid))
    cells = {(cell.section, cell.label, cell.column): cell for cell in inventory.cells}
    policy = _policy()
    selected = []
    for field in semantic_fields(scope, grid):
        cell = cells.get((field["section"], field["label"], field["columns"][scope]))
        if cell is None:
            _need(scope == "campus" and field["section"] == "Retention Rates by Grade",
                  "Continuity report omits a required published local cell")
            reason = ("retention_section_not_published" if "Retention Rates by Grade" in inventory.unpublished_sections
                      else "retention_grade_row_not_published")
            value = _measurement(None, field["unit"], policy, reason)
        else:
            value = _measurement(cell.raw, field["unit"], policy)
        selected.append(ContinuityCell(field["field"], value, cell.row if cell else None, cell.column if cell else None))
    return ContinuityReport(scope, identifier, receipt, inventory.sha256, tuple(selected),
                            inventory.unpublished_sections, CONTRACT_SHA256)


def read_continuity_report(store: ProvenanceStore, receipt: str, *, scope: str, identifier: str,
                           expected_sha256: str, contract: Mapping[str, Any] | None = None) -> ContinuityReport:
    grid = _resolve_contract(contract)
    raw, _ = _read_student_input(store, receipt, scope=scope, identifier=identifier, expected_sha256=expected_sha256)
    return parse_continuity_report(raw, scope=scope, identifier=identifier, receipt=receipt, contract=grid)


def projection(reports: Sequence[ContinuityReport], *, contract: Mapping[str, Any] | None = None
               ) -> tuple[pa.Table, dict[str, Any]]:
    """Project supported native values with exact original raw/status/coordinates."""
    grid = _resolve_contract(contract)
    _need(bool(reports) and len({report.scope for report in reports}) == 1, "Select one nonempty continuity scope")
    _need(len({report.identifier for report in reports}) == len(reports), "Duplicate continuity identifier")
    _need(all(report.contract_sha256 == CONTRACT_SHA256 for report in reports), "Continuity parse/projection contracts differ")
    scope = reports[0].scope
    _need(scope != "state" or len(reports) == 1, "Continuity requires one direct Texas report")
    fields = semantic_fields(scope, grid)
    names = [field["field"] for field in fields]
    _need(all([cell.field for cell in report.cells] == names for report in reports), "Continuity scope/field order differs")
    arrays = {}
    support: dict[str, Any] = {"fields": {}, "context": ["tprs_report_receipt", "tprs_report_sha256", "tprs_unpublished_sections"]}
    for index, field in enumerate(fields):
        name = field["field"]
        cells = [report.cells[index] for report in reports]
        arrays[name] = pa.array([cell.measurement.value for cell in cells], type=pa.int64() if field["unit"] == "count" else pa.float64())
        pointers = {}
        for role, values in (
            ("raw", [cell.measurement.raw for cell in cells]),
            ("status", [cell.measurement.status.value for cell in cells]),
            ("reason", [cell.measurement.reason for cell in cells]),
            ("row", [cell.row for cell in cells]),
            ("column", [cell.column for cell in cells]),
        ):
            column = f"tprs_{role}_{name}"
            arrays[column] = pa.array(values, type=pa.int64() if role in {"row", "column"} else pa.string())
            pointers[role] = column
        support["fields"][name] = {"raw": pointers["raw"], "status": pointers["status"],
                                  "auxiliary": [pointers[role] for role in ("row", "column", "reason")]}
    if scope != "state":
        key = scope + "_number"
        arrays[key] = pa.array([report.identifier for report in reports], type=pa.string())
        support["context"].append(key)
    arrays["tprs_report_receipt"] = pa.array([report.receipt for report in reports], type=pa.string())
    arrays["tprs_report_sha256"] = pa.array([report.sha256 for report in reports], type=pa.string())
    arrays["tprs_unpublished_sections"] = pa.array([canonical_json(list(report.unpublished_sections)) for report in reports], type=pa.string())
    return pa.table(arrays), support


def _crosswalk(inputs: Mapping[str, Sequence[str]], *, contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    grid = _resolve_contract(contract)
    tables = []
    for table, hashes in inputs.items():
        scope = table.split("-", 1)[0]
        selected = semantic_fields(scope, grid)
        labels = [field["label"] for field in selected]
        tables.append({"table": table, "header_identity": "position", "input_sha256s": list(hashes),
            "metric_period": PERIOD.to_dict(), "source_schema_sha256": source_schema_fingerprint(labels, labels),
            "fields": [{"position": index, "code": field["label"], "label": field["label"], "field": field["field"],
                        "dictionary_status": "not_documented", "publisher_type": None, "publisher_description": None,
                        "disposition": "fact"} for index, field in enumerate(selected, 1)]})
    return {"schema_version": 2, "mapping_version": 1, "datasets": [
        {"dataset": DATASET, "edition": str(EDITION), "source_only": True, "tables": tables}]}


def verify_publisher_witnesses(store: ProvenanceStore, *, contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Compare frozen original HTML and independent OOXML local tokens/locators."""
    grid = _resolve_contract(contract)
    student_grid = _student_contract(grid)
    results = {}
    for scope, witness in grid["witnesses"].items():
        report = read_continuity_report(store, witness["html_receipt"], scope=scope, identifier=witness["identifier"],
                                        expected_sha256=witness["html_sha256"], contract=grid)
        workbook = _student_workbook_cells(store, scope, witness, contract=student_grid)
        actual = {cell.field: cell for cell in report.cells}
        _need(set(actual) == {cell["field"] for cell in witness["cells"]}, "Continuity golden field coverage differs")
        missing = 0
        for expected in witness["cells"]:
            cell = actual[expected["field"]]
            _need((cell.measurement.raw, cell.row, cell.column) ==
                  (expected["raw"], expected["html_row"], expected["html_column"]), "Continuity original HTML golden differs")
            if expected["xlsx_row"] is None:
                missing += 1
                _need(cell.measurement.status == ValueStatus.MISSING and cell.measurement.reason == expected["reason"],
                      "Continuity absent-grade golden differs")
                field, = [field for field in semantic_fields(scope, grid) if field["field"] == cell.field]
                _need(not any(column == "A" and value == field["label"] for (_, column), value in workbook.items()),
                      "Continuity missing workbook grade is actually present")
            else:
                _need(workbook.get((expected["xlsx_row"], "A")) == next(
                    field["label"] for field in semantic_fields(scope, grid) if field["field"] == cell.field),
                    "Continuity workbook golden label differs")
                _need(workbook.get((expected["xlsx_row"], expected["xlsx_column"])) == expected["raw"],
                      "Continuity independent workbook golden differs")
        results[scope] = {"identifier": witness["identifier"], "bindings": len(actual),
                         "published_cells": len(actual) - missing, "absent_grade_bindings": missing,
                         "html_receipt": witness["html_receipt"], "xlsx_receipt": witness["xlsx_receipt"]}
    return results


def continuity_campus_edition(student: EditionKey) -> EditionKey:
    """Translate an exact retained Student campus partition without rebalancing."""
    from .tprs_partitions import derived_campus_edition

    return derived_campus_edition(student, DATASET)


def _edition(student: EditionKey) -> EditionKey:
    return continuity_campus_edition(student) if student.scope == "campus" else replace(student, dataset=DATASET)


def _selection_ledger(student: DataRelease, store: ProvenanceStore, grid: Mapping[str, Any]) -> dict[str, Any]:
    _need(student.offline and not student.remote and student.manifest_sha256 == grid["student_manifest_sha256"],
          "Continuity requires the exact retained offline Student attachment")
    leaves = _reviewed_leaves(student)
    _verify_campus_census(leaves, store)
    _need(all(leaf.grid_sha256 == grid["student_grid_sha256"] for leaf in leaves), "Continuity Student grid differs")
    ledger = {"schema_version": 1, "student_manifest_sha256": student.manifest_sha256,
              "student_grid_sha256": grid["student_grid_sha256"],
              "partitions": [{"student_edition": leaf.artifact.edition.to_dict(),
                  "reports": [asdict(report) for report in sorted(leaf.reports, key=lambda item: item.identifier)]}
                  for leaf in sorted(leaves, key=lambda item: item.artifact.edition.identity)],
              "campus_rosters": sorted({item for leaf in leaves for item in leaf.campus_rosters})}
    _validate_selection(ledger, store, grid)
    return ledger


def _validate_selection(ledger: Mapping[str, Any], store: ProvenanceStore, grid: Mapping[str, Any]
                        ) -> dict[EditionKey, tuple[ReportSelection, ...]]:
    _need(sha256((canonical_json(dict(ledger)) + "\n").encode()).hexdigest() == grid["selection_sha256"],
          "Continuity selection differs from the independently frozen retained source ledger")
    selections = {}
    observed: dict[str, set[str]] = {scope: set() for scope in _SCOPES}
    for partition in ledger["partitions"]:
        student = EditionKey.from_dict(partition["student_edition"])
        key = _edition(student)
        reports = tuple(ReportSelection(**item) for item in partition["reports"])
        _need(key not in selections and bool(reports) and all(report.scope == key.scope for report in reports),
              "Continuity source partition identity differs")
        _need([report.identifier for report in reports] == sorted(report.identifier for report in reports),
              "Continuity source partition order differs")
        _need(key.revision_sha256 == EditionKey.source_revision(tuple(report.sha256 for report in reports)),
              "Continuity partition source revision differs")
        for report in reports:
            _need(report.identifier not in observed[key.scope], "Continuity repeats a source entity")
            observed[key.scope].add(report.identifier)
        selections[key] = reports
    _need({scope: len(ids) for scope, ids in observed.items()} == grid["source_counts"] and observed["state"] == {"S"},
          "Continuity selected scope census differs")
    expected = set()
    for region, receipt in ledger["campus_rosters"]:
        identifiers = set(campus_roster(store, region, receipt))
        _need(expected.isdisjoint(identifiers), "Continuity campus rosters overlap")
        expected.update(identifiers)
    _need(expected == observed["campus"], "Continuity campus census differs from the original publisher rosters")
    return selections


def _retain_closure(target: ProvenanceStore, source: ProvenanceStore, roots: Sequence[str]) -> None:
    """Retain validated immutable bytes/receipts; hard links avoid duplicate storage."""
    graph = source.verify_record_closure(roots)
    for folder in ("objects", "records"):
        (target.root / folder).mkdir(parents=True, exist_ok=True)
    for identifier, record in graph.items():
        artifacts = [record["artifact"]] if record["kind"] == "acquisition" else record["outputs"].values()
        pairs = [(source.verify_artifact(item["sha256"]), target.root / "objects" / item["sha256"]) for item in artifacts]
        pairs.append((source.root / "records" / f"{identifier}.json", target.root / "records" / f"{identifier}.json"))
        for original, destination in pairs:
            if destination.exists():
                _need(sha256(destination.read_bytes()).hexdigest() == sha256(original.read_bytes()).hexdigest(),
                      "Continuity immutable provenance destination conflicts")
                continue
            try:
                os.link(original, destination)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copyfile(original, destination)


def _metadata(columns: ReferenceDocument, methodology: ReferenceDocument, selection: ReferenceDocument,
              student: ReferenceDocument) -> DatasetMetadata:
    baseline = _configured_source().metadata
    references = [columns if ref.reference_id == "columns" else methodology if ref.reference_id == "continuity_grid" else ref
                  for ref in baseline.references]
    return replace(baseline, references=tuple([*references, selection, student]))


def _batches(store: ProvenanceStore, reports: Sequence[ReportSelection], grid: Mapping[str, Any],
             *, rows: int = 16) -> Iterator[tuple[pa.Table, dict[str, Any], Sequence[ContinuityReport]]]:
    for offset in range(0, len(reports), rows):
        parsed = [read_continuity_report(store, report.receipt, scope=report.scope, identifier=report.identifier,
                    expected_sha256=report.sha256, contract=grid) for report in reports[offset:offset + rows]]
        table, support = projection(parsed, contract=grid)
        yield table, support, parsed


def _recipe(student: EditionKey, grid: Mapping[str, Any]) -> dict[str, str]:
    return {"operation": "tprs_student_continuity", "student_attachment": grid["student_manifest_sha256"],
            "student_edition": student.identity, "continuity_grid": CONTRACT_SHA256,
            "selection_sha256": grid["selection_sha256"]}


def materialize_continuity(student: DataRelease, store: ProvenanceStore, output: Path) -> dict[str, Any]:
    """Seal an offline unpublished native child from the retained Student census."""
    _need(not output.exists() and not output.is_symlink(), "Continuity requires a new exclusive output directory")
    grid = definition()
    _student_contract(grid)
    ledger = _selection_ledger(student, store, grid)
    selections = _validate_selection(ledger, store, grid)
    witnesses = verify_publisher_witnesses(store, contract=grid)
    inputs = {key.table: [report.sha256 for report in reports] for key, reports in selections.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".tprs-continuity-", dir=output.parent) as temporary:
        root = Path(temporary)
        stage = root / "bundle"
        stage.mkdir(mode=0o700)
        target = ProvenanceStore(stage / "provenance")
        baseline = _configured_source()
        roots = {report.receipt for reports in selections.values() for report in reports}
        roots.update(receipt for _, receipt in ledger["campus_rosters"])
        roots.update(receipt for ref in baseline.metadata.references for receipt in ref.provenance_ids)
        roots.update(witness["xlsx_receipt"] for witness in grid["witnesses"].values())
        _retain_closure(target, store, sorted(roots))

        def capture(path: Path, name: str, title: str, parents: tuple[str, ...] = ()) -> str:
            return target.record_acquisition(path, ResolvedDataset("reference:" + name, "reference", EDITION, EDITION, "",
                DatasetMetadata(title=title, version=str(EDITION), provenance_ids=parents)))

        parent_receipt = capture(student.path, "tprs_continuity_student_attachment", "Exact retained Student Information attachment")
        _need(target.read_record(parent_receipt)["artifact"]["sha256"] == grid["student_manifest_sha256"], "Student manifest changed")
        selection_path = root / "continuity-selection.json"
        selection_path.write_text(canonical_json(ledger) + "\n", encoding="utf-8")
        selection_receipt = capture(selection_path, "tprs_continuity_selection", "Exact retained Student partition/report selection", (parent_receipt,))
        grid_path = root / "continuity-methodology.json"
        grid_path.write_text(canonical_json(dict(grid)) + "\n", encoding="utf-8")
        golden_roots = tuple(sorted({receipt for witness in grid["witnesses"].values()
                                     for receipt in (witness["html_receipt"], witness["xlsx_receipt"])}))
        grid_receipt = capture(grid_path, "tprs_continuity_methodology", "Frozen continuity methodology and original publisher goldens", golden_roots)
        schema_path = root / "continuity-crosswalk.json"
        schema_path.write_text(canonical_json(_crosswalk(inputs, contract=grid)) + "\n", encoding="utf-8")
        schema_receipt = capture(schema_path, "tprs_continuity_crosswalk", "Exact scope/partition continuity field bindings",
                                 (grid_receipt, selection_receipt, parent_receipt))
        metadata = _metadata(
            ReferenceDocument("columns", "schema", path=schema_path.name, title="Continuity source-label field bindings",
                              reviewed_sha256=sha256(schema_path.read_bytes()).hexdigest(), provenance_ids=(schema_receipt,)),
            ReferenceDocument("continuity_grid", "methodology", path=grid_path.name,
                              title="2026 continuity source definitions, scope and independent publisher goldens",
                              reviewed_sha256=CONTRACT_SHA256, provenance_ids=(grid_receipt,)),
            ReferenceDocument("selection", "source_notes", path=selection_path.name, title="Exact retained Student partition/report selection",
                              reviewed_sha256=grid["selection_sha256"], provenance_ids=(selection_receipt,)),
            ReferenceDocument("student_attachment", "source_notes", title="Exact retained Student Information attachment",
                              path=f"provenance/objects/{student.manifest_sha256}",
                              reviewed_sha256=student.manifest_sha256, provenance_ids=(parent_receipt,)),
        )
        policy = _policy()
        config = load_config_bytes(canonical_json({"value_policies": {policy.policy_id: policy.to_dict()},
            "data_sources": {DATASET: {str(EDITION): {"path": "source-only-continuity", "metadata": metadata.to_dict(),
                "reader": {"profiles": {"source": {"column_schema": schema_path.name}}}}}}}).encode(), path=root / "config.yaml")
        source = config.resolve_dataset(DATASET, EDITION, reader_profile="source", exact=True)
        recorder = BuildRecorder(config, target)
        toolchain = {"python": sys.version, "pyarrow": version("pyarrow"), "platform": platform.platform()}
        common = [schema_receipt, grid_receipt, selection_receipt, parent_receipt]
        rosters = [receipt for _, receipt in ledger["campus_rosters"]]
        for partition in ledger["partitions"]:
            original = EditionKey.from_dict(partition["student_edition"])
            key = _edition(original)
            reports = selections[key]
            first, support, _ = next(_batches(store, reports[:1], grid))
            recipe = _recipe(original, grid)
            contract = recorder.leaf_contract(key, source, native_schema=first.schema.serialize().to_pybytes(),
                columns=first.column_names, key_columns=[] if key.scope == "state" else [key.scope + "_number"],
                rows=len(reports), inputs=inputs[key.table], recipe=recipe, toolchain=toolchain, column_support=support)
            encoded = canonical_json(contract)
            path = stage / (sha256(encoded.encode()).hexdigest() + ".parquet")

            def batches() -> Iterator[pa.RecordBatch]:
                for table, actual_support, _ in _batches(store, reports, grid):
                    _need(actual_support == support, "Continuity batch support differs")
                    yield from table.to_batches()

            physical = write_leaf_artifact(path, batches(), contract=contract, batch_bytes=8 * 1024**2)
            acquisitions = [report.receipt for report in reports]
            processing = target.record_processing(dataset=DATASET, inputs=[*acquisitions, *common, *rosters], outputs={"leaf": path},
                processor="teadata.tprs_student_continuity", code_version=recorder.processor_sha256,
                parameters={"recipe": recipe, "leaf_contract_sha256": path.stem, "toolchain": toolchain},
                validation={"rows": len(reports), "field_correctness_qualified": False, "release_activation": False})
            artifact = ArtifactRef(key, physical["sha256"], physical["size_bytes"], path.name, encoded, processing, "leaf")
            recorder.record_edition(key, source, artifacts=[artifact], acquisition_ids=acquisitions)
        manifest = publish_release(stage, recorder.catalog(schema_version=4), schema_version=4)
        candidate = DataRelease(manifest, offline=True)
        candidate.verify()
        result = {"status": "unpublished_native_continuity", "manifest_sha256": candidate.manifest_sha256,
                  "build_id": candidate.build_id, "student_attachment": student.manifest_sha256,
                  "continuity_grid_sha256": CONTRACT_SHA256, "selection_sha256": grid["selection_sha256"],
                  "scopes": grid["source_counts"], "editions": len(selections), "publisher_witnesses": witnesses,
                  "field_correctness_qualified": False, "release_activation": False}
        (stage / "MATERIALIZATION.json").write_text(canonical_json(result) + "\n", encoding="utf-8")
        _install_candidate(stage, output)
    return result


def _captured_reference(metadata: DatasetMetadata, name: str, store: ProvenanceStore,
                         *, maximum: int = 8 * 1024**2) -> tuple[ReferenceDocument, str, bytes]:
    reference, = [ref for ref in metadata.references if ref.reference_id == name]
    _need(len(reference.provenance_ids) == 1 and reference.reviewed_sha256 is not None,
          "Continuity reference lacks exact captured provenance: " + name)
    assert reference.reviewed_sha256 is not None
    receipt, = reference.provenance_ids
    record = store.verify_record_closure([receipt])[receipt]
    _need(record["kind"] == "acquisition" and record["artifact"]["sha256"] == reference.reviewed_sha256,
          "Continuity captured reference differs: " + name)
    _need(0 < record["artifact"]["size_bytes"] <= maximum, "Continuity captured reference exceeds its budget")
    _need((store.root / "objects" / reference.reviewed_sha256).stat().st_size == record["artifact"]["size_bytes"],
          "Continuity captured reference size differs")
    raw = store.verify_artifact(reference.reviewed_sha256).read_bytes()
    _need(len(raw) == record["artifact"]["size_bytes"], "Continuity captured reference size differs")
    return reference, receipt, raw


def _continuity_inputs(release: DataRelease, store: ProvenanceStore, expected_manifest_sha256: str
                       ) -> tuple[dict[EditionKey, tuple[ArtifactRef, tuple[ReportSelection, ...]]], list[str]]:
    _need(release.offline and not release.remote and release.manifest_sha256 == expected_manifest_sha256,
          "Continuity replay requires the exact selected offline child")
    grid = definition()
    index = release.catalog.to_dict()
    _need(index["schema_version"] == 4, "Continuity replay requires the native edition catalog")
    artifacts = [release.catalog.artifact_ref(identifier) for identifier in index["artifacts"]]
    _need(bool(artifacts), "Continuity replay has no native artifacts")
    metadata = release.catalog.metadata(DATASET, edition=artifacts[0].edition)
    columns, schema_receipt, schema_raw = _captured_reference(metadata, "columns", store)
    methodology, grid_receipt, grid_raw = _captured_reference(metadata, "continuity_grid", store)
    selection, selection_receipt, ledger_raw = _captured_reference(metadata, "selection", store)
    parent, parent_receipt, _ = _captured_reference(metadata, "student_attachment", store, maximum=4 * 1024**2)
    _need(sha256(grid_raw).hexdigest() == CONTRACT_SHA256 and grid_raw == (canonical_json(dict(grid)) + "\n").encode(),
          "Continuity captured interpretation changed")
    _need(parent.reviewed_sha256 == grid["student_manifest_sha256"], "Continuity parent pin changed")
    ledger = json.loads(ledger_raw, object_pairs_hook=_unique_object)
    _need(ledger_raw == (canonical_json(ledger) + "\n").encode(), "Continuity source ledger is not canonical")
    selections = _validate_selection(ledger, store, grid)
    expected_schema = _crosswalk({key.table: [report.sha256 for report in reports] for key, reports in selections.items()}, contract=grid)
    _need(schema_raw == (canonical_json(expected_schema) + "\n").encode(), "Continuity captured field/source bindings changed")
    expected_metadata = _metadata(columns, methodology, selection, parent)
    _need(metadata == expected_metadata, "Continuity metadata differs from its frozen scope/period contract")
    _need({artifact.edition for artifact in artifacts} == set(selections) and len(artifacts) == len(selections),
          "Continuity exact partition inventory differs")
    roots = {witness["xlsx_receipt"] for witness in grid["witnesses"].values()}
    result = {}
    common = [schema_receipt, grid_receipt, selection_receipt, parent_receipt]
    rosters = [receipt for _, receipt in ledger["campus_rosters"]]
    original_keys = {_edition(EditionKey.from_dict(partition["student_edition"])): EditionKey.from_dict(partition["student_edition"])
                     for partition in ledger["partitions"]}
    for artifact in artifacts:
        key = artifact.edition
        _need(release.catalog.metadata(DATASET, edition=key) == expected_metadata, "Continuity partition metadata differs")
        _need(release.catalog.policy(DATASET, edition=key) == _policy(), "Continuity partition policy differs")
        record = release.catalog._processing_record(artifact)
        recipe = _recipe(original_keys[key], grid)
        reports = selections[key]
        _need(record == store.read_record(artifact.processing_id)
              and record["dataset"] == DATASET and record["processor"] == "teadata.tprs_student_continuity"
              and record["code_version"] == artifact.contract["producer"]["code_sha256"]
              and record["parameters"] == {"recipe": recipe, "leaf_contract_sha256": sha256(canonical_json(artifact.contract).encode()).hexdigest(),
                                           "toolchain": artifact.contract["producer"]["toolchain"]}
              and record["inputs"] == [*(report.receipt for report in reports), *common, *rosters]
              and artifact.contract["inputs"] == [report.sha256 for report in reports]
              and artifact.contract["recipe_sha256"] == sha256(canonical_json(recipe).encode()).hexdigest()
              and artifact.output == "leaf", "Continuity processing/source lineage differs")
        roots.add(artifact.processing_id)
        result[key] = artifact, reports
    graph = store.verify_record_closure(sorted(roots))
    _need(store.read_record(schema_receipt)["source_metadata"]["provenance_ids"] == [grid_receipt, selection_receipt, parent_receipt],
          "Continuity schema provenance differs")
    _need(store.read_record(selection_receipt)["source_metadata"]["provenance_ids"] == [parent_receipt],
          "Continuity selection provenance differs")
    golden_roots = sorted({receipt for witness in grid["witnesses"].values()
                           for receipt in (witness["html_receipt"], witness["xlsx_receipt"])})
    _need(store.read_record(grid_receipt)["source_metadata"]["provenance_ids"] == golden_roots,
          "Continuity independent golden provenance differs")
    return result, sorted(graph)


def retained_continuity_evidence(release: DataRelease, store: ProvenanceStore, *, expected_manifest_sha256: str) -> list[str]:
    """Bind exact child, original report census and independent publisher evidence."""
    return _continuity_inputs(release, store, expected_manifest_sha256)[1]


def qualify_continuity(release: DataRelease, store: ProvenanceStore, *, expected_manifest_sha256: str) -> dict[str, Any]:
    """Replay source values/statuses; this result alone does not authorize release admission."""
    bindings, evidence = _continuity_inputs(release, store, expected_manifest_sha256)
    grid = definition()
    witnesses = verify_publisher_witnesses(store, contract=grid)
    release.verify()
    scopes: dict[str, dict[str, Any]] = {}
    for key, (artifact, reports) in bindings.items():
        actual = release.table(key).to_arrow().replace_schema_metadata(None)
        physical = pq.ParquetFile(release.artifact(artifact.identity))
        physical_batches = physical.iter_batches(batch_size=16)
        summary = scopes.setdefault(key.scope, {"reports": 0, "editions": 0, "fields": len(semantic_fields(key.scope, grid)),
                                               "statuses": Counter(), "missing_reasons": Counter(), "section_omissions": Counter()})
        offset = 0
        for expected, support, parsed in _batches(store, reports, grid):
            batch = next(physical_batches, None)
            _need(batch is not None and pa.Table.from_batches([batch]).replace_schema_metadata(None).equals(expected),
                  "Continuity physical source/raw/value/status/coordinate replay differs")
            _need(artifact.contract["column_support"] == support
                  and actual.slice(offset, expected.num_rows).select(expected.column_names).equals(expected),
                  "Continuity native public source replay differs")
            for report in parsed:
                summary["section_omissions"].update(report.unpublished_sections)
                for cell in report.cells:
                    summary["statuses"][cell.measurement.status.value] += 1
                    if cell.measurement.status == ValueStatus.MISSING:
                        summary["missing_reasons"][cell.measurement.reason] += 1
            offset += expected.num_rows
        _need(next(physical_batches, None) is None and offset == actual.num_rows == artifact.contract["rows"] == len(reports),
              "Continuity native row census differs")
        summary["reports"] += offset
        summary["editions"] += 1
    _need({scope: summary["reports"] for scope, summary in scopes.items()} == grid["source_counts"],
          "Continuity replay scope totals differ")
    return {"status": "PASS_CONTINUITY_SOURCE_REPLAY", "manifest_sha256": release.manifest_sha256,
            "build_id": release.build_id, "continuity_grid_sha256": CONTRACT_SHA256,
            "student_attachment": grid["student_manifest_sha256"], "selection_sha256": grid["selection_sha256"],
            "retained_records": len(evidence), "scopes": scopes, "publisher_witnesses": witnesses,
            "scope": "Exact selected retained child and publisher literal goldens; independent review, scientific dispatch admission, normal composition and publication remain separate."}
