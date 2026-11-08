"""Build an offline campus transfer edition with sourced historical endpoints."""

from __future__ import annotations

from csv import reader
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
from pathlib import Path
import platform
import shutil
import sys
from collections import defaultdict
from collections.abc import Collection, Iterator, Mapping
from tempfile import mkdtemp
from typing import Any, TypedDict

import pyarrow as pa
import pyarrow.compute as arrow_compute
import pyarrow.parquet as pq

from .analytical import write_leaf_artifact
from .askted_directory import _verified_archive
from .askted_lifecycle import LifecycleEventKind, LifecycleReportKind, _PORTAL, read_lifecycle_report
from .acquisition import _form_request
from .column_schema import crosswalk_dataset, parse_column_schema, table_input_sha256s
from .dataset_metadata import ArtifactRef, DataPeriod, DatasetMetadata, EditionKey, ReferenceDocument, ResolvedDataset
from .entity_store import EntityStore
from .identifiers import canonical_source_identifier
from .manifest import BuildRecorder, canonical_json, file_identity
from .provenance import ProvenanceStore, validate_reference_records
from .release import DataRelease, publish_release
from .scripts.download_askted import ARCHIVE_URL
from .teadata_config import Config, load_config_bytes
from .transfer_leaves import _leaf_projection
from .transfer_sources import _ACTION, _captured_input, _selected_encoding, _write_index, read_transfer_table, validate_campus_directions

DATASET = "campus_transfer_reports"
DIRECTORY_REFERENCE = "historical_endpoint_directory"
LIFECYCLE_REFERENCE = "historical_endpoint_lifecycle"
_DETAIL = "transfer_row_kind"
_LOCATORS = ("transfer_source_input", "transfer_source_record")
_HISTORICAL_FIELDS = ("reporting_campus_askted_record", "partner_campus_askted_record", "historical_askted_archive_as_of")
_OBSOLESCENCE_FIELDS = (
    "reporting_campus_askted_obsolete_record", "reporting_campus_askted_obsolete_date",
    "partner_campus_askted_obsolete_record", "partner_campus_askted_obsolete_date",
)
_ENDPOINTS = (("transfer_reporting_number", "transfer_reporting_district_number", "REPORT_CAMPUS_NAME"),
              ("transfer_partner_number", "transfer_partner_district_number", "CAMPNAME_RES_OR_ATTEND"))
pc: Any = arrow_compute


def _validate_campus_acquisition(acquisition: dict, source: ResolvedDataset, year: int) -> None:
    period = DataPeriod("school_year", year - 1, year)
    request = acquisition.get("parameters", {})
    controls = request.get("form_controls", [])
    if (acquisition["kind"] != "acquisition" or acquisition["dataset"] != DATASET
            or acquisition.get("section") != "data_sources" or acquisition.get("method") != "download"
            or (acquisition.get("requested_year"), acquisition.get("resolved_year")) != (year, year)
            or acquisition["source_metadata"].get("version") != str(year)
            or acquisition["source_metadata"].get("source_url") != source.metadata.source_url
            or DataPeriod.from_dict(acquisition["source_metadata"].get("period")) != period
            or source.metadata.period != period or source.metadata.version != str(year)
            or request.get("form_url") != _ACTION or request.get("http_method") != "GET"
            or request.get("http_status") != 200 or request.get("form_content_type") is not None
            or dict(controls) != {"_service": "marykay", "_program": "adhoc.download_transfer_report.sas",
                                 "scope": "Campus", "agg_level": "campus", "RptClass": "transfer", "_debug": "0",
                                 "school_year": str(year), "report_format": "csv",
                                 "Campus_Transfer_Report": "Get a Campus Transfer Report"}
            or len(controls) != 9):
        raise ValueError("Campus edition requires the exact statewide publisher acquisition")
    expected_url, _, _ = _form_request(_ACTION, "GET", [tuple(item) for item in controls], None)
    if (acquisition.get("requested_url") != expected_url or acquisition.get("resolved_url") != expected_url
            or acquisition.get("response_headers", {}).get("content-type", "").split(";", 1)[0].strip().lower() != "text/csv"):
        raise ValueError("Campus acquisition URL and media type differ from the statewide form request")


def _prepared_table(store: ProvenanceStore, config: Config, source: ResolvedDataset, year: int,
                    preparation: str) -> tuple[pa.Table, dict, str, str]:
    original, = source.metadata.provenance_ids
    graph = store.verify_record_closure([preparation, original])
    record, acquisition = graph[preparation], graph[original]
    validation, parameters = record.get("validation", {}), record.get("parameters", {})
    policy = config.value_policies.get(source.metadata.value_policy or "")
    if (record["kind"] != "processing" or record["dataset"] != DATASET
            or record.get("processor") != "teadata.scripts.refresh_transfers"
            or parameters.get("processor_sha256") != record.get("code_version")
            or set(record["outputs"]) != {"prepared_source", "annotations", "validation_report"}
            or validation.get("status") != "PASS"
            or any(validation.get(key) is not True for key in (
                "prepared_source_qualified", "canonical_edges_qualified", "raw_string_readback_equal"))
            or original not in record["inputs"] or policy is None or policy.to_dict() != parameters.get("value_policy")):
        raise ValueError("Campus edition requires an accepted preparation and its exact original/policy")
    _validate_campus_acquisition(acquisition, source, year)
    digest = acquisition["artifact"]["sha256"]
    outputs = record["outputs"]
    report_digest = outputs["validation_report"]["sha256"]
    report = json.loads(_captured_input(store.verify_artifact(report_digest), report_digest))
    entry, = report.get("inputs", ())
    if (report.get("status") != "prepared" or report.get("grain") != "campus" or report.get("year") != year
            or any(report.get(key) is not True for key in (
                "prepared_source_qualified", "canonical_edges_qualified", "raw_string_readback_equal"))
            or report.get("publisher_totals_computed") is not False
            or report.get("policy_sha256") != sha256(json.dumps(policy.to_dict(), sort_keys=True).encode()).hexdigest()
            or report.get("implementation") != {key: parameters.get(key) for key in ("processor_sha256", "cli_sha256")}
            or report.get("toolchain") != parameters.get("toolchain")
            or entry.get("record_id") != original or entry.get("artifact") != acquisition["artifact"]
            or entry.get("source_input") != 0 or entry.get("region") is not None or entry.get("status") != "parsed"):
        raise ValueError("Campus preparation report differs from the recorded statewide input")
    encoding, encoding_details = _selected_encoding(acquisition, entry["encoding"].get("selected_encoding"))
    if encoding_details != entry["encoding"]:
        raise ValueError("Campus preparation encoding differs from its retained acquisition")
    published = read_transfer_table(store.verify_artifact(digest), grain="campus", year=year, policy=policy,
                                    expected_sha256=digest, encoding=encoding)
    annotations = outputs["annotations"]["sha256"]
    expected_notes = {"sources": [{"source_input": 0, "record_id": original, "input_sha256": digest,
                                   "layout": published.layout, "annotations": list(published.annotations)}]}
    if json.loads(_captured_input(store.verify_artifact(annotations), annotations)) != expected_notes:
        raise ValueError("Campus preparation annotations differ from the exact original report")
    directions = validate_campus_directions(published, max_examples=report["source_validation"]["examples_limit"])
    if directions != report["source_validation"] or directions["raw_count_token_differences"]:
        raise ValueError("Campus directional witness differs from the exact original report")
    table = published.table.append_column(_LOCATORS[0], pa.repeat(pa.scalar(0, type=pa.int32()), published.table.num_rows))
    if not table.num_rows == report["rows"] == entry["rows"] == validation["rows"] or table.column_names != report["columns"]:
        raise ValueError("Campus preparation report row/schema coverage differs from the exact original")
    prepared = outputs["prepared_source"]
    if not 0 < prepared["size_bytes"] <= 16 * 1024 * 1024:
        raise ValueError("Campus prepared artifact exceeds byte admission")
    with pq.ParquetFile(store.verify_artifact(prepared["sha256"])) as native:
        if (native.metadata.num_rows != table.num_rows or native.schema_arrow != table.schema
                or sum(native.metadata.row_group(i).total_byte_size for i in range(native.num_row_groups)) > 256 * 1024 * 1024):
            raise ValueError("Campus preparation metadata differs from source coverage or byte admission")
        offset = 0
        for batch in native.iter_batches(batch_size=4096, use_threads=False):
            if not pa.Table.from_batches([batch]).equals(table.slice(offset, batch.num_rows), check_metadata=True):
                raise ValueError("Campus preparation differs from the exact original table")
            offset += batch.num_rows
        if offset != table.num_rows:
            raise ValueError("Campus preparation readback ended early")
    return table, directions, digest, prepared["sha256"]


def _fall_directory(store: ProvenanceStore, source: ResolvedDataset, year: int) -> tuple[str, str, tuple[str, str]]:
    reference, = [item for item in source.metadata.references if item.reference_id == DIRECTORY_REFERENCE]
    receipt, = reference.provenance_ids
    if (reference.role != "source_notes" or reference.status != "available" or reference.url != ARCHIVE_URL
            or reference.reviewed_sha256 is None):
        raise ValueError("AskTED transfer endpoint needs a reviewed fall directory reference")
    archived, directory = _verified_archive(store, receipt, year=year, season="fall",
                                            expected_sha256=reference.reviewed_sha256)
    return directory, archived.publisher_declared_as_of, (receipt, archived.index_receipt)


def _fall_identity_rows(directory: str, interested: Collection[str]) -> Iterator[tuple[int, str, str | None, str]]:
    rows = reader(StringIO(directory, newline=""), strict=True)
    header = next(rows, [])
    campus_at, district_at, name_at = (header.index(name) for name in ("School Number", "District Number", "School Name"))
    for record, row in enumerate(rows, 2):
        if not row:
            continue
        if campus_at >= len(row):
            raise ValueError("AskTED fall row lacks its school-number column")
        campus = canonical_source_identifier(row[campus_at], width=9)
        if campus is None or campus not in interested:
            continue
        district, name = None, ""
        if len(row) == len(header):
            try:
                district = canonical_source_identifier(row[district_at], width=6)
            except ValueError:
                pass
            name = row[name_at].strip()
        yield record, campus, district, name


_HOLD_REASONS = {
    "missing_endpoint_number": "Campus transfer detail has a missing endpoint number",
    "ambiguous_fall_identity": "AskTED evidence is ambiguous or invalid for transfer endpoint",
    "invalid_fall_identity": "AskTED evidence is ambiguous or invalid for transfer endpoint",
    "district_absent_from_registry": "Historical transfer endpoint district is absent from the companion registry",
    "missing_fall_evidence": "Missing exact fall AskTED evidence for transfer endpoint",
    "conflicting_transfer_identity": "Transfer observation disagrees with fall AskTED identity",
}


class HistoricalEndpointHold(ValueError):
    """Complete endpoint reasons and affected-row counts for a withheld candidate."""

    def __init__(self, observations: pa.Table, issues: dict[str, set[str | None]]):
        problems = []
        affected = pa.repeat(False, len(observations))
        for reason, identifiers in sorted(issues.items()):
            keys = sorted(key for key in identifiers if key is not None)
            matches = pa.array([*keys, None] if None in identifiers else keys, type=pa.string())
            reporting, partner = [pc.is_in(observations[name], value_set=matches) for name, _, _ in _ENDPOINTS]
            distinct = pc.or_(reporting, partner)
            affected = pc.or_(affected, distinct)
            problems.append({
                "reason": reason, "endpoint_ids": keys, "affected_detail_rows": pc.sum(distinct).as_py(),
                "reporting_rows": pc.sum(reporting).as_py(), "partner_rows": pc.sum(partner).as_py(),
            })
        self.report = {"status": "HOLD_HISTORICAL_TRANSFER_ENDPOINTS", "affected_detail_rows": pc.sum(affected).as_py(),
                       "problems": problems, "release_activation": False}
        super().__init__("; ".join(_HOLD_REASONS[reason] for reason in sorted(issues)))


def _endpoint_context(table: pa.Table, directory: str, as_of: str, entities: Path) -> tuple[pa.Table, dict]:
    details = pc.equal(table[_DETAIL], "detail")
    observations = table.filter(details)
    issues: dict[str, set[str | None]] = defaultdict(set)
    with EntityStore(entities, immutable=True) as registry:
        current = {"'" + row["lookup_key"] for row in registry.iter_campuses(include_meta=False) if not row["is_private"]}
        missing = {value for key, _, _ in _ENDPOINTS for value in pc.unique(observations[key]).to_pylist()} - current
        if None in missing:
            issues["missing_endpoint_number"].add(None)
            missing.remove(None)
        reviewed: dict[str, dict] = {}
        seen: set[str] = set()
        for record, campus, district, name in _fall_identity_rows(directory, missing):
            if campus in seen:
                issues["ambiguous_fall_identity"].add(campus)
                continue
            seen.add(campus)
            if district is None or campus[:7] != district or not name:
                issues["invalid_fall_identity"].add(campus)
            elif registry.get_district(district, include_meta=False, cache=False) is None:
                issues["district_absent_from_registry"].add(campus)
            else:
                reviewed[campus] = {"record": record, "district": district, "name": name}
    if unresolved := missing - seen:
        issues["missing_fall_evidence"].update(unresolved)
    keys = sorted(reviewed)
    positions = pa.array([reviewed[key]["record"] for key in keys], type=pa.int64())
    context = []
    for campus, district, name in _ENDPOINTS:
        values = pa.array(keys, type=table[campus].type)
        affected = observations.filter(pc.is_in(observations[campus], value_set=values))
        for row in affected.select([campus, district, name]).group_by([campus, district, name]).aggregate([]).to_pylist():
            evidence = reviewed[row[campus]]
            if row[district] != evidence["district"] or row[name].strip() != evidence["name"]:
                issues["conflicting_transfer_identity"].add(row[campus])
        indexes = pc.index_in(table[campus], value_set=values)
        context.append(pc.if_else(details, pc.take(positions, indexes), pa.scalar(None, type=pa.int64())))
    if issues:
        raise HistoricalEndpointHold(observations, issues)
    historical = pc.or_(pc.is_valid(context[0]), pc.is_valid(context[1]))
    context.append(pc.if_else(historical, pa.scalar(as_of), pa.scalar(None, type=pa.string())))
    for field, array in zip(_HISTORICAL_FIELDS, context, strict=True):
        table = table.append_column(field, array)
    return table, reviewed


class _ObsoleteEndpoint(TypedDict):
    record: int
    obsolete_date: str


@dataclass(frozen=True)
class _CampusObsolescence:
    receipt: str
    sha256: str
    printed_date: str
    endpoints: dict[str, _ObsoleteEndpoint]

    def recipe(self) -> dict[str, Any]:
        return {"receipt": self.receipt, "sha256": self.sha256, "printed_date": self.printed_date,
                "endpoints": self.endpoints}


def _lifecycle_reference(source: ResolvedDataset) -> ReferenceDocument | None:
    refs = [item for item in source.metadata.references if item.reference_id == LIFECYCLE_REFERENCE]
    if not refs:
        return None
    reference, = refs
    if (reference.role != "source_notes" or reference.status != "available" or reference.url != _PORTAL
            or len(reference.provenance_ids) != 1 or not reference.reviewed_sha256):
        raise ValueError("Campus lifecycle context requires the pinned public AskTED report")
    return reference


def _campus_obsolescence(store: ProvenanceStore, reference: ReferenceDocument, *, as_of: str,
                         historical: Mapping[str, Mapping[str, Any]]) -> _CampusObsolescence:
    assert reference.reviewed_sha256 is not None
    receipt, = reference.provenance_ids
    report = read_lifecycle_report(store, receipt, report=LifecycleReportKind.PUBLIC_CAMPUS_CLOSURES,
                                  expected_sha256=reference.reviewed_sha256)
    if reference.title != f"AskTED All Public Schools Closed, printed {report.printed_date.isoformat()}":
        raise ValueError("AskTED lifecycle citation differs from the pinned report print date")
    label_date = date.fromisoformat(as_of)
    matches: dict[str, _ObsoleteEndpoint] = {}
    for event in report.events:
        subject, = event.participants
        identifier = "'" + subject.identifier
        identity = historical.get(identifier)
        if (event.kind is not LifecycleEventKind.OBSOLETE or identity is None
                or subject.name.strip() != identity["name"]):
            continue
        if event.event_date is None or event.event_date > report.printed_date:
            raise ValueError(f"Unresolved AskTED obsolete date for historical campus {identifier}")
        if event.event_date <= label_date:
            continue
        if identifier in matches:
            raise ValueError(f"Ambiguous AskTED obsolete dates for historical campus {identifier}")
        matches[identifier] = {"record": event.source_record, "obsolete_date": event.event_date.isoformat()}
    return _CampusObsolescence(report.receipt, report.sha256, report.printed_date.isoformat(), matches)


def _append_campus_obsolescence(table: pa.Table, evidence: _CampusObsolescence) -> pa.Table:
    keys = tuple(evidence.endpoints)
    ids = pa.array(keys, type=pa.string())
    records = pa.array([evidence.endpoints[key]["record"] for key in keys], type=pa.int64())
    dates = pa.array([evidence.endpoints[key]["obsolete_date"] for key in keys], type=pa.string())
    for index, ((campus, _, _), fall_locator) in enumerate(zip(_ENDPOINTS, _HISTORICAL_FIELDS[:2], strict=True)):
        offsets = pc.index_in(table[campus], value_set=ids)
        admitted = pc.is_valid(table[fall_locator])
        for position, values in enumerate((records, dates), index * 2):
            values = pc.if_else(admitted, pc.take(values, offsets), pa.scalar(None, type=values.type))
            table = table.append_column(_OBSOLESCENCE_FIELDS[position], values)
    return table


def materialize_campus_transfers(store: ProvenanceStore, *, config: Path, config_sha256: str, year: int,
                                preparation: str, registry_manifest: Path, registry_sha256: str, output: Path) -> dict:
    """Retain publisher observations and admit missing endpoint evidence without changing current entities."""
    if type(year) is not int or not 2 <= year <= 9999 or output.exists() or output.is_symlink():
        raise ValueError("Campus transfer candidate needs an explicit edition and a new output directory")
    raw_config = _captured_input(config, config_sha256)
    cfg = load_config_bytes(raw_config, path=config)
    source = cfg.resolve_dataset(DATASET, year, reader_profile="source", exact=True)
    validate_reference_records(source, store.read_record, verify_all_claimed=True)
    table, directions, original_sha, prepared_sha = _prepared_table(store, cfg, source, year, preparation)
    directory, as_of, directory_receipts = _fall_directory(store, source, year)
    raw_manifest = _captured_input(registry_manifest, registry_sha256)
    registry = DataRelease(registry_manifest, expected_sha256=registry_sha256, offline=True)
    entities = registry.artifact("entities")
    entities_identity = json.loads(raw_manifest)["artifacts"]["entities"]
    if file_identity(entities) != entities_identity:
        raise ValueError("Companion entity registry differs from the pinned manifest before resolution")
    try:
        table, historical = _endpoint_context(table, directory, as_of, entities)
    finally:
        if file_identity(entities) != entities_identity:
            raise ValueError("Companion entity registry changed during historical endpoint resolution")

    lifecycle_ref = _lifecycle_reference(source)
    obsolescence = None
    context = _HISTORICAL_FIELDS
    if lifecycle_ref is not None:
        obsolescence = _campus_obsolescence(store, lifecycle_ref, as_of=as_of, historical=historical)
        table = _append_campus_obsolescence(table, obsolescence)
        context += _OBSOLESCENCE_FIELDS

    resource = cfg.column_contract(DATASET, year, reader_profile="source")
    if resource is None:
        raise ValueError("Campus edition needs a reviewed crosswalk")
    document = parse_column_schema(resource["text"].encode())
    crosswalk, = crosswalk_dataset(document, DATASET, str(year))["tables"]
    table_input_sha256s(crosswalk, acquisition_sha256s=[original_sha])
    names, support, operations = _leaf_projection(table.column_names, crosswalk["fields"], grain="campus", year=year,
                                                   extra_context=context)
    count_field, = [field["field"] for field in crosswalk["fields"] if field["code"] == "TRANSFERS_IN_OR_OUT"]
    if ({metric.field for metric in source.metadata.metrics.values() if metric.unit == "count"} != {count_field}
            or DataPeriod.from_dict(crosswalk.get("metric_period")) != source.metadata.period
            or next(field for field in crosswalk["fields"] if field["field"] == count_field).get("disposition") != "fact"):
        raise ValueError("Campus edition requires one exact publisher count field and unit")
    projected = table.rename_columns(names)

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=".campus-transfer-leaf-", dir=output.parent))
    try:
        target = ProvenanceStore(stage / "provenance")
        roots = list(dict.fromkeys([preparation, *directory_receipts,
                                   *(identifier for ref in source.metadata.references for identifier in ref.provenance_ids)]))
        for root in roots:
            target.copy_record_from(store, root)
        for role, raw in (("source-config", raw_config), ("companion-registry-manifest", raw_manifest)):
            path = stage / (role + ".input")
            path.write_bytes(raw)
            roots.append(target.record_acquisition(path, ResolvedDataset(
                "reference:transfer_" + role, "reference", None, None, "", DatasetMetadata(title=role))))
        recorder = BuildRecorder(cfg, target)
        toolchain = {"python": sys.version, "pyarrow": version("pyarrow"), "platform": platform.platform(),
                     "architecture": platform.machine()}
        lock = Path(__file__).parents[1] / "uv.lock"
        if lock.is_file():
            toolchain["lock_sha256"] = file_identity(lock)["sha256"]
        key = EditionKey(DATASET, str(year), EditionKey.source_revision((original_sha,)), "campus", crosswalk["table"],
                         "one publisher table row per original source input and logical CSV record")
        recipe = {"operation": "campus_transfer_historical_endpoint_admission", "preparation": preparation,
                  "config_sha256": config_sha256, "publisher_projection": operations, "batch_rows": 4096,
                  "registry_manifest_sha256": registry_sha256, "registry_entities": entities_identity,
                  "registry_build_id": registry.build_id, "directory_receipts": list(directory_receipts),
                  "directory_archive_as_of": as_of, "historical_endpoints": historical}
        physical_inputs = [original_sha, prepared_sha]
        if obsolescence is not None:
            recipe["campus_obsolescence"] = obsolescence.recipe()
            physical_inputs.append(obsolescence.sha256)
        contract = recorder.leaf_contract(key, source, native_schema=projected.schema.serialize().to_pybytes(),
            columns=names, key_columns=_LOCATORS, rows=projected.num_rows, inputs=physical_inputs,
            recipe=recipe, toolchain=toolchain, column_support=support)
        leaf_id = sha256(canonical_json(contract).encode()).hexdigest()
        leaf = stage / (leaf_id + ".parquet")
        physical = write_leaf_artifact(leaf, projected.to_batches(max_chunksize=4096), contract=contract, batch_bytes=8 * 1024 * 1024)
        with pq.ParquetFile(leaf) as readback:
            offset = 0
            for batch in readback.iter_batches(batch_size=4096, use_threads=False):
                actual = pa.Table.from_batches([batch.replace_schema_metadata(None)])
                if not actual.equals(projected.slice(offset, batch.num_rows), check_metadata=True):
                    raise ValueError("Campus transfer native readback differs from the exact source projection")
                offset += batch.num_rows
            if offset != projected.num_rows:
                raise ValueError("Campus transfer native readback ended early")
        validation = {"raw_string_readback_equal": True, "rows": projected.num_rows, "directional_readback": directions,
                      "historical_endpoints_qualified": len(historical), "publisher_totals_computed": False,
                      "field_correctness_qualified": False, "release_activation": False}
        processing = target.record_processing(dataset=DATASET, inputs=roots, outputs={"leaf": leaf},
            processor="teadata.scripts.refresh_transfers.materialize-campus", code_version=recorder.processor_sha256,
            parameters={"leaf_contract_sha256": leaf_id, "recipe": recipe, "toolchain": toolchain}, validation=validation)
        ref = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name, canonical_json(contract), processing, "leaf")
        recorder.record_edition(key, source, artifacts=[ref], acquisition_ids=source.metadata.provenance_ids)
        catalog = recorder.catalog(schema_version=3)
        manifest = publish_release(stage, catalog, schema_version=3)
        result = {"status": "unpublished_native_transfer_candidate", "grain": "campus", "edition": key.to_dict(),
                  "build_id": catalog.build_id, "manifest": file_identity(manifest), "processing_record_id": processing,
                  "historical_endpoints": historical, **validation}
        _write_index(stage, result, filename="MATERIALIZATION.json")
        if output.exists() or output.is_symlink():
            raise FileExistsError("Campus transfer candidate destination appeared before completion")
        stage.rename(output)
        return result
    finally:
        if stage.exists():
            shutil.rmtree(stage)
