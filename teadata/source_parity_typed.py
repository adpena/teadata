"""Bounded raw and typed replay for the admitted PEIMS and district-transfer recipes."""

from __future__ import annotations

from contextlib import ExitStack
import csv
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Literal, TypedDict

import pyarrow as pa
import pyarrow.compute as arrow_compute
import pyarrow.parquet as pq

from .acquisition import AcquisitionError
from .campus_transfer_leaves import (
    DATASET as CAMPUS_TRANSFERS,
    DIRECTORY_REFERENCE,
    _ENDPOINTS,
    _HISTORICAL_FIELDS,
    _OBSOLESCENCE_FIELDS,
    _CampusObsolescence,
    _append_campus_obsolescence,
    _campus_obsolescence,
    _fall_directory,
    _fall_identity_rows,
    _lifecycle_reference,
    _validate_campus_acquisition,
)
from .column_schema import crosswalk_dataset, parse_column_schema, table_input_sha256s
from .dataset_metadata import COLUMN_SCHEMA_MAX_BYTES, DataPeriod, ResolvedDataset
from .manifest import canonical_json
from .district_types import DISTRICT_TYPE_COLUMNS, DISTRICT_TYPE_REFERENCE, reviewed_district_types
from .peims_leaves import DATASETS as PEIMS_DATASETS, PRIMARY_DISABILITY
from .peims_leaves import ENCODING, SCOPES_BY_DATASET, _count_columns, _key_columns
from .peims_staff import (
    STAFF,
    STAFF_EMPLOYEE_BASIS,
    STAFF_EMPLOYEE_COUNT,
    STAFF_RECIPE,
    staff_allows_unavailable_average,
    staff_fact_units,
    validate_staff_request,
)
from .provenance import validate_record_graph
from .source_csv import SourceLines
from .source_parity import _FieldReport, _Uncovered
from .teadata_config import load_config_bytes
from .transfer_leaves import _leaf_projection
from .transfer_sources import _FOOTNOTES, _selected_encoding, _transfer_projection, _transfer_titles, district_batch_inputs, transfer_columns
from .values import ValueStatus

if TYPE_CHECKING:
    from .dataset_metadata import EditionKey
    from .provenance import ProvenanceStore
    from .release import DataRelease
    from .values import ValuePolicy


pc: Any = arrow_compute
_BATCH_ROWS = 512
_POSITIONS = ("transfer_source_record", "transfer_first_line", "transfer_last_line")


@dataclass(frozen=True)
class _Original:
    path: Path
    sha256: str
    encoding: str
    region: str | None
    rows: int
    annotations: tuple[dict[str, Any], ...] = ()


class _FallEndpoint(TypedDict):
    record: int
    district: str
    name: str


@dataclass(frozen=True)
class _CampusHistory:
    as_of: str
    endpoints: dict[str, _FallEndpoint]
    obsolescence: _CampusObsolescence | None = None

    @property
    def context(self) -> tuple[str, ...]:
        return _HISTORICAL_FIELDS + (_OBSOLESCENCE_FIELDS if self.obsolescence is not None else ())


@dataclass(frozen=True)
class _Typed:
    names: list[str]
    fields: list[dict[str, Any]]
    originals: list[_Original]
    artifact: Path
    prepared: Path | None
    layout: dict[str, Any] | None
    projection: dict[str, str]
    period: DataPeriod
    policy: ValuePolicy
    rows: int
    lineage: dict[str, Any]
    staff_scope: str | None = None
    district_types: dict[str, str] | None = None
    transfer_grain: Literal["campus", "district"] | None = None
    campus_history: _CampusHistory | None = None


def _digest(value: Any) -> str:
    return sha256(canonical_json(value).encode()).hexdigest()


def _resource(store: ProvenanceStore, artifact: dict, maximum: int, label: str) -> bytes:
    if not 0 < artifact["size_bytes"] <= maximum:
        raise _Uncovered(f"{label} exceeds its retained resource budget")
    raw = store.verify_artifact(artifact["sha256"]).read_bytes()
    if len(raw) != artifact["size_bytes"] or sha256(raw).hexdigest() != artifact["sha256"]:
        raise ValueError(f"{label} changed during replay")
    return raw


def _validate_fall_endpoints(directory: str, claimed: Any) -> dict[str, _FallEndpoint]:
    if not isinstance(claimed, dict):
        raise ValueError("Campus transfer recipe lacks historical endpoint records")
    observed: dict[str, _FallEndpoint] = {}
    for record, campus, district, name in _fall_identity_rows(directory, claimed):
        if campus in observed or district is None or campus[:7] != district or not name:
            raise ValueError("Campus transfer recipe names an ambiguous or malformed fall identity")
        observed[campus] = {"record": record, "district": district, "name": name}
    if any(not isinstance(entry, dict) or type(entry.get("record")) is not int for entry in claimed.values()):
        raise ValueError("Campus transfer recipe lacks exact historical row positions")
    if observed != claimed:
        raise ValueError("Campus transfer recipe differs from the exact fall archive rows")
    return observed


def _campus_history(store: ProvenanceStore, source: ResolvedDataset, year: int, recipe: dict,
                    graph: dict, direct_inputs: list[str], config_artifact: dict, resource_bytes: int) -> _CampusHistory:
    reference, = [item for item in source.metadata.references if item.reference_id == DIRECTORY_REFERENCE]
    directory_receipt, = reference.provenance_ids
    index_receipt = graph[directory_receipt]["parameters"]["archive_index_record"]
    directory_receipts = [directory_receipt, index_receipt]
    if (len(set(directory_receipts)) != 2 or recipe.get("directory_receipts") != directory_receipts
            or not set(directory_receipts) <= set(direct_inputs)):
        raise ValueError("Campus transfer recipe does not bind both AskTED archive receipts")
    if any(graph[identifier]["artifact"]["size_bytes"] > resource_bytes for identifier in directory_receipts):
        raise _Uncovered("AskTED archive evidence exceeds its retained resource budget")
    try:
        directory, as_of, observed = _fall_directory(store, source, year)
    except AcquisitionError as error:
        raise ValueError(str(error)) from error
    if directory_receipts != list(observed) or recipe.get("directory_archive_as_of") != as_of:
        raise ValueError("Campus transfer recipe differs from its exact fall archive")
    companion, = [graph[value]["artifact"] for value in direct_inputs if graph[value]["kind"] == "acquisition"
                  and graph[value]["dataset"] == "reference:transfer_companion-registry-manifest"]
    manifest = json.loads(_resource(store, companion, resource_bytes, "Companion registry manifest"))
    if (companion["sha256"] != recipe.get("registry_manifest_sha256")
            or manifest["artifacts"]["entities"] != recipe.get("registry_entities")
            or manifest["catalog"]["build_id"] != recipe.get("registry_build_id")):
        raise ValueError("Campus transfer recipe differs from its companion registry manifest")
    endpoints = _validate_fall_endpoints(directory, recipe.get("historical_endpoints"))
    reference = _lifecycle_reference(source)
    if config_artifact["filename"] != "source-config.input":
        raise ValueError("Campus transfer retained configuration has an unexpected filename")
    raw_config = _resource(store, config_artifact, resource_bytes, "Transfer configuration")
    try:
        config = load_config_bytes(raw_config, path=Path(config_artifact["filename"]))
    except (RuntimeError, OverflowError) as error:
        raise ValueError("Campus transfer retained configuration cannot be parsed") from error
    configured = config.resolve_dataset(source.dataset, year, reader_profile="source", exact=True)
    if _lifecycle_reference(configured) != reference:
        raise ValueError("Campus lifecycle reference differs from its retained source configuration")
    if reference is None:
        if "campus_obsolescence" in recipe:
            raise ValueError("Campus transfer recipe lacks its reviewed lifecycle reference")
        return _CampusHistory(as_of, endpoints)
    receipt, = reference.provenance_ids
    if receipt not in direct_inputs:
        raise ValueError("Campus transfer recipe omits its reviewed lifecycle input")
    if graph[receipt]["artifact"]["size_bytes"] > resource_bytes:
        raise _Uncovered("AskTED lifecycle report exceeds its retained resource budget")
    obsolescence = _campus_obsolescence(store, reference, as_of=as_of, historical=endpoints)
    if canonical_json(recipe.get("campus_obsolescence")) != canonical_json(obsolescence.recipe()):
        raise ValueError("Campus transfer recipe differs from the exact AskTED obsolete dates")
    return _CampusHistory(as_of, endpoints, obsolescence)


def _check_campus_directional_witness(report: dict, validation: object, policy: ValuePolicy, original_sha256: str) -> None:
    directions = report["source_validation"]
    if (not isinstance(validation, dict) or not isinstance(directions, dict) or directions.get("status") != "PASS"
            or directions.get("raw_count_token_differences") != 0
            or directions.get("input_sha256") != original_sha256 or directions != validation.get("directional_readback")
            or report.get("policy_sha256") != sha256(json.dumps(policy.to_dict(), sort_keys=True).encode()).hexdigest()):
        raise ValueError("Campus preparation differs from its accepted directional witness")


def _selection(release: DataRelease, store: ProvenanceStore, key: EditionKey, *, resource_bytes: int) -> _Typed:
    variables = list(release.catalog.variables(edition=key))
    plans = release.catalog.resolve_artifacts((key, row["field"]) for row in variables)
    if not variables or len(plans) != 1:
        raise _Uncovered("Only one complete typed native leaf per edition is supported")
    ref = plans[0]["artifact"]
    contract = ref.contract
    graph = store.verify_record_closure([ref.processing_id])
    record = graph[ref.processing_id]
    parameters = record.get("parameters", {})
    recipe = parameters.get("recipe", {})
    peims = key.dataset in PEIMS_DATASETS
    staff = key.dataset == STAFF
    campus = key.dataset == CAMPUS_TRANSFERS
    grain: Literal["campus", "district"] = "campus" if campus else "district"
    if peims:
        pair = ("teadata.scripts.refresh_peims", STAFF_RECIPE if staff else "reviewed_peims_student_csv_projection")
        config_name = "peims_leaf_config"
    elif campus:
        pair = ("teadata.scripts.refresh_transfers.materialize-campus", "campus_transfer_historical_endpoint_admission")
        config_name = "transfer_source-config"
    else:
        pair = ("teadata.scripts.refresh_transfers.materialize", "accepted_transfer_preparation_projection")
        config_name = "transfer_leaf_config"
    if (record.get("processor"), recipe.get("operation")) != pair:
        raise _Uncovered("No supported recorded typed replay recipe")
    output = record.get("outputs", {}).get(ref.output, {})
    if (record["kind"] != "processing" or record["dataset"] != key.dataset
            or parameters.get("leaf_contract_sha256") != ref.identity
            or record["code_version"] != contract["producer"]["code_sha256"]
            or parameters.get("toolchain") != contract["producer"]["toolchain"]
            or _digest(recipe) != contract["recipe_sha256"]
            or any(output.get(name) != getattr(ref, name) for name in ("sha256", "size_bytes"))):
        raise ValueError("Typed leaf processing does not bind its selected descriptor and recipe")
    configs = [graph[value]["artifact"] for value in record["inputs"]
               if graph[value]["kind"] == "acquisition" and graph[value]["dataset"] == f"reference:{config_name}"]
    if [item["sha256"] for item in configs] != [recipe.get("config_sha256")]:
        raise ValueError("Typed leaf recipe does not bind its exact retained source configuration")
    metadata = release.catalog.metadata(key.dataset, edition=key)
    period = metadata.period
    if (period is None or period.kind != "school_year" or period.end_year is None or str(period.end_year) != key.edition
            or period.start_year != period.end_year - 1):
        raise ValueError("Typed replay requires its exact recorded school-year period")
    policy = release.catalog.policy(key.dataset, edition=key)
    if _digest(policy.to_dict()) != contract["policy_sha256"]:
        raise ValueError("Typed leaf policy differs from its reviewed edition policy")
    schemas = [value for value in metadata.references if value.role == "schema" and value.status == "available"
               and value.reviewed_sha256 and value.provenance_ids]
    if len(schemas) != 1:
        raise _Uncovered("A uniquely reviewed source crosswalk with retained receipts is required")
    schema, = schemas
    schema_records = [graph[value] for value in schema.provenance_ids if value in graph]
    if len(schema_records) != len(schema.provenance_ids):
        raise ValueError("A reviewed source crosswalk receipt is outside the typed leaf closure")
    descriptors = [value["artifact"] for value in schema_records if value["kind"] == "acquisition"
                   and value["artifact"]["sha256"] == schema.reviewed_sha256]
    if not descriptors:
        raise ValueError("Typed leaf lacks its exact reviewed crosswalk bytes")
    document = parse_column_schema(_resource(store, descriptors[0], min(COLUMN_SCHEMA_MAX_BYTES, resource_bytes), "Crosswalk"))
    mapping, = [value for value in crosswalk_dataset(document, key.dataset, key.edition)["tables"]
                if value["table"] == key.table]
    fields = mapping["fields"]
    if (_digest(mapping) != contract["mapping_sha256"]
            or {value["field"] for value in fields} != {value["field"] for value in variables}):
        raise ValueError("Reviewed crosswalk and typed leaf field contract differ")
    acquisitions = release.catalog.acquisitions(key.dataset, edition=key)
    if not acquisitions:
        raise _Uncovered("Typed leaf original receipts are unavailable")
    hashes = [value["artifact"]["sha256"] for value in acquisitions]
    ids = [value["record_id"] for value in acquisitions]
    if list(table_input_sha256s(mapping, acquisition_sha256s=hashes)) != hashes:
        raise ValueError("Typed leaf originals differ from its ordered crosswalk inputs")
    prepared = None
    projection = {}
    district_types = None
    history = None
    attached_hashes = []
    if peims:
        if (key.scope not in SCOPES_BY_DATASET[key.dataset] or key.table != key.scope or len(ids) != 1
                or recipe.get("encoding") != ENCODING
                or mapping.get("header_identity") != "position" or not mapping.get("layout")
                or mapping["layout"]["label_record"] is not None):
            raise _Uncovered("No supported recorded PEIMS positional source layout")
        if ids[0] not in record["inputs"] or mapping.get("metric_period") != period.to_dict():
            raise ValueError("PEIMS original receipt or field period differs from the retained projection")
        if any(token.isascii() and token.isdecimal() for token in policy.tokens):
            raise _Uncovered("PEIMS count tokens conflict with the admitted integer projection")
        names = []
        support: dict[str, dict[str, str | list[str]]] = {}
        for field in fields:
            name = field["field"]
            names.append(name)
            if field["disposition"] == "fact":
                raw, status = f"peims_raw_{name}", f"peims_status_{name}"
                names.extend([status, raw])
                support[name] = {"raw": raw, "status": status}
            elif field["disposition"] != "identity":
                raise _Uncovered("PEIMS has an unsupported source field disposition")
        if staff:
            assert period.end_year is not None
            units = {row["field"]: row["unit"] for row in variables if row["field"] in support}
            if units != staff_fact_units(period.end_year):
                raise ValueError("PEIMS staffing facts differ from the exact edition's reviewed units")
            if STAFF_EMPLOYEE_COUNT in support:
                names.append(STAFF_EMPLOYEE_BASIS)
                support[STAFF_EMPLOYEE_COUNT]["auxiliary"] = [STAFF_EMPLOYEE_BASIS]
        names.append("peims_source_record")
        keys = _key_columns(key.dataset, key.scope)
        context = ["peims_source_record"]
        refs = [value for value in metadata.references if value.reference_id == DISTRICT_TYPE_REFERENCE]
        if refs and key.dataset == PRIMARY_DISABILITY and key.scope == "district" and "charter_status" not in names:
            roster_ref, = refs
            roster_receipt, = roster_ref.provenance_ids
            if roster_receipt not in record["inputs"] or recipe.get("district_type_roster") != roster_receipt:
                raise ValueError("PEIMS historical charter classification lacks its exact roster input")
            district_types = reviewed_district_types(metadata, store, graph)
            attached_hashes.append(roster_ref.reviewed_sha256)
            names.extend(DISTRICT_TYPE_COLUMNS)
            context.extend(DISTRICT_TYPE_COLUMNS)
        elif "district_type_roster" in recipe:
            raise ValueError("PEIMS historical charter classification has an unexpected roster")
        expected_support = {"fields": support, "context": context}
        if contract["columns"] != names or contract["key_columns"] != keys or contract.get("column_support") != expected_support:
            raise ValueError("PEIMS leaf does not retain all reviewed fields, raw tokens, statuses and row keys")
        inputs = [(ENCODING, None, contract["rows"], ())]
        upstream = graph
    else:
        if (key.scope, key.table) != (grain, "report_rows") or (campus and period.end_year in {2021, 2022}):
            raise _Uncovered("Transfer table or historical CSV layout is unsupported")
        preparation_id = recipe.get("preparation" if campus else "prepared_processing_id")
        if preparation_id not in record["inputs"]:
            raise ValueError("Transfer leaf does not bind its historical preparation")
        upstream = validate_record_graph(graph.__getitem__, [preparation_id])
        preparation = upstream[preparation_id]
        validation, prior = preparation.get("validation", {}), preparation.get("parameters", {})
        if (preparation["kind"] != "processing" or preparation["dataset"] != key.dataset
                or preparation["processor"] != "teadata.scripts.refresh_transfers"
                or (not campus and preparation["code_version"] != recipe.get("historical_preparation_producer"))
                or preparation["code_version"] != prior.get("processor_sha256")
                or not isinstance(validation, dict) or validation.get("status") != "PASS" or validation.get("raw_string_readback_equal") is not True
                or prior.get("value_policy") != policy.to_dict()
                or (not campus and recipe.get("publisher_totals_computed") is not False)
                or set(preparation["outputs"]) != {"prepared_source", "annotations", "validation_report"}):
            raise ValueError("Transfer preparation does not match its accepted historical recipe")
        report = json.loads(_resource(store, preparation["outputs"]["validation_report"], min(2 * 1024**2, resource_bytes), "Transfer report"))
        if not isinstance(report, dict):
            raise ValueError("Transfer preparation report must be a JSON object")
        annotations = json.loads(_resource(store, preparation["outputs"]["annotations"],
                                           min(2 * 1024**2, resource_bytes), "Transfer annotations"))["sources"]
        if campus:
            original, = ids
            source = ResolvedDataset(key.dataset, "data_sources", period.end_year, period.end_year, "", metadata)
            _validate_campus_acquisition(upstream[original], source, period.end_year)
            _check_campus_directional_witness(report, record["validation"], policy, hashes[0])
            if (record["validation"].get("publisher_totals_computed") is not False
                    or any(validation.get(name) is not True or report.get(name) is not True for name in (
                        "prepared_source_qualified", "canonical_edges_qualified"))
                    or report.get("implementation") != {name: prior.get(name) for name in ("processor_sha256", "cli_sha256")}
                    or report.get("toolchain") != prior.get("toolchain")):
                raise ValueError("Campus preparation differs from its accepted directional witness")
            history = _campus_history(store, source, period.end_year, recipe, graph, record["inputs"], configs[0],
                                      resource_bytes)
            if history.obsolescence is not None:
                attached_hashes.append(history.obsolescence.sha256)
            if record["validation"].get("historical_endpoints_qualified") != len(history.endpoints):
                raise ValueError("Campus preparation differs from its historical endpoint coverage")
            ordered, controls = list(zip(ids, hashes, strict=True)), {original: None}
        else:
            index, = [upstream[value]["artifact"] for value in preparation["inputs"]
                      if upstream[value]["kind"] == "acquisition"
                      and upstream[value]["dataset"] == "reference:transfer_batch_index"]
            ordered, controls = district_batch_inputs(store, _resource(store, index, min(2 * 1024**2, resource_bytes), "Transfer index"),
                                                     period.end_year, all_offered=True)
        artifact = preparation["outputs"]["prepared_source"]
        prepared = store.verify_artifact(artifact["sha256"])
        hashes.append(artifact["sha256"])
        extra = history.context if history is not None else ()
        names, support, operations = _leaf_projection([*report["columns"], *extra], fields, grain=grain,
                                                      year=period.end_year, extra_context=extra)
        if (ordered != list(zip(ids, hashes[:-1], strict=True)) or not set(ids) <= set(preparation["inputs"])
                or report["status"] != "prepared" or report["year"] != period.end_year or report["grain"] != key.scope
                or report.get("publisher_totals_computed") is not False or report.get("raw_string_readback_equal") is not True
                or report["rows"] != contract["rows"] or validation.get("rows") != contract["rows"]
                or names != contract["columns"] or operations != recipe.get("publisher_projection")
                or contract.get("column_support") != support or contract["key_columns"] != ["transfer_source_input", "transfer_source_record"]):
            raise ValueError("Transfer leaf differs from its exact annual ordered preparation and projection")
        projection = {item["prepared_column"]: item["field"] for item in operations}
        inputs = []
        for number, (entry, claimed, annotation) in enumerate(zip(report["inputs"], acquisitions, annotations, strict=True)):
            receipt = upstream[claimed["record_id"]]
            encoding, encoding_evidence = _selected_encoding(receipt, entry["encoding"]["selected_encoding"])
            if (entry["source_input"] != number or entry["status"] != "parsed" or entry["record_id"] != claimed["record_id"]
                    or entry["artifact"] != claimed["artifact"] or annotation["layout"] != "original"
                    or (campus and entry["region"] is not None)
                    or annotation["source_input"] != number or annotation["record_id"] != claimed["record_id"]
                    or annotation["input_sha256"] != claimed["artifact"]["sha256"]
                    or entry["encoding"] != encoding_evidence or type(entry["rows"]) is not int or entry["rows"] < 1):
                raise ValueError("Transfer input differs from its accepted ordered decoding contract")
            inputs.append((encoding, controls[claimed["record_id"]], entry["rows"], tuple(annotation["annotations"])))
        if sum(rows for _, _, rows, _ in inputs) != contract["rows"]:
            raise ValueError("Transfer originals do not account for every native row")
    if contract["inputs"] != [*hashes, *attached_hashes]:
        raise ValueError("Typed leaf inputs differ from its exact physical derivation")
    originals = []
    for claimed, (encoding, region, rows, annotations) in zip(acquisitions, inputs, strict=True):
        receipt = upstream.get(claimed["record_id"])
        if (receipt is None or receipt != {name: value for name, value in claimed.items() if name != "record_id"}
                or receipt["kind"] != "acquisition" or receipt["method"] != "download" or receipt["dataset"] != key.dataset
                or receipt["resolved_year"] != period.end_year or receipt["requested_year"] != period.end_year
                or receipt["source_metadata"].get("version") != key.edition
                or receipt["source_metadata"].get("period") != period.to_dict()):
            raise ValueError("Catalog acquisition differs from its exact upstream original receipt")
        digest = claimed["artifact"]["sha256"]
        if staff:
            assert period.end_year is not None
            validate_staff_request(receipt, period.end_year, key.scope)
        originals.append(_Original(store.verify_artifact(digest), digest, encoding, region, rows, annotations))
    return _Typed(names, fields, originals, release.artifact(ref.identity), prepared,
                  mapping.get("layout") if peims else None, projection, period, policy, contract["rows"],
                  {"leaf_processing_id": ref.processing_id, "leaf_sha256": ref.sha256,
                   "prepared_sha256": hashes[-1] if prepared else None, "crosswalk_sha256": schema.reviewed_sha256,
                   "acquisitions": acquisitions}, staff_scope=key.scope if staff else None,
                  district_types=district_types, transfer_grain=None if peims else grain, campus_history=history)


def _wire(original: _Original, selection: _Typed, *, limit: int) -> Iterator[tuple[list[str], int, int, int, int]]:
    """Decode publisher CSV independently of the producer's Arrow CSV reader."""
    peims = selection.layout is not None
    header = [field["code"] for field in selection.fields]
    layout = selection.layout
    annotation = None
    if layout is not None:
        intervals = sorted((row["first_record"], row["last_record"], row["sha256"]) for row in layout["annotations"])
        bounds = [(layout["code_record"], layout["code_record"], None),
                  (layout["first_data_record"], layout["last_data_record"], None), *intervals]
        end = 0
        for first, last, _ in sorted(bounds, key=lambda row: row[0]):
            if first != end + 1 or last < first:
                raise ValueError("PEIMS source has overlapping or unclaimed physical records")
            end = last
        annotations = iter(intervals)
        annotation = next(annotations, None)
        digest = sha256()
        if header != [field["label"] for field in selection.fields]:
            raise ValueError("PEIMS reviewed code and label headers differ")
    titles = (_transfer_titles(selection.transfer_grain, year=selection.period.end_year, region=original.region)
              if selection.transfer_grain else [])
    preamble: list[str] = []
    notes: list[str] = []
    found_header = footer = False
    count = record = 0
    transfer_annotations = iter(original.annotations)
    with original.path.open("rb") as stream:
        lines = SourceLines(stream, limit, "Publisher CSV record exceeds its working-byte limit", encoding=original.encoding)
        reader = csv.reader(lines, strict=True)
        try:
            while True:
                first_line = reader.line_num + 1
                lines.record.clear()
                row = next(reader, None)
                if row is None:
                    break
                record += 1
                last_line = reader.line_num
                data = False
                kind = "blank"
                if layout is not None:
                    if first_line != record or last_line != record:
                        raise ValueError("PEIMS source requires one physical line per CSV record")
                    if annotation is not None and annotation[0] <= record <= annotation[1]:
                        digest.update(lines.record)
                        if record == annotation[1]:
                            if digest.hexdigest() != annotation[2]:
                                raise ValueError("PEIMS retained annotation bytes differ from the reviewed crosswalk")
                            annotation = next(annotations, None)
                            digest = sha256()
                    elif record == layout["code_record"]:
                        if row != header:
                            raise ValueError("PEIMS literal headers differ from their reviewed positions")
                        found_header = True
                    elif layout["first_data_record"] <= record <= layout["last_data_record"]:
                        data = True
                    else:
                        raise ValueError("Unexpected PEIMS source record outside its reviewed layout")
                elif not found_header:
                    if row == header and preamble == titles:
                        found_header = True
                        kind = "header"
                    elif len(row) <= 1:
                        title = " ".join(row[0].split()) if row else ""
                        if title:
                            kind = "preamble"
                            preamble.append(title)
                            if preamble != titles[:len(preamble)]:
                                raise ValueError("Unexpected transfer publisher title")
                    else:
                        raise ValueError("Unexpected transfer preamble or header")
                elif len(row) == len(header) and not footer:
                    data = True
                else:
                    footer = True
                    note = " ".join(row[0].split()) if len(row) == 1 else ""
                    if note:
                        kind = "footer"
                        notes.append(note)
                    if len(row) > 1 or tuple(notes) != _FOOTNOTES[:len(notes)]:
                        raise ValueError("Unexpected transfer publisher footnote or data after the footer")
                if data:
                    if len(row) != len(header) or not found_header:
                        raise ValueError(f"Publisher data at CSV record {record} differs from its literal header")
                    count += 1
                    yield row, record, first_line, last_line, len(lines.record)
                elif not peims:
                    observed = {"kind": kind, "record": record, "first_line": first_line, "last_line": last_line,
                                "raw_sha256": sha256(lines.record).hexdigest(), "text": lines.record.decode(lines.encoding)}
                    if next(transfer_annotations, None) != observed:
                        raise ValueError("Transfer original annotations differ from retained preparation")
        except csv.Error as error:
            raise ValueError(f"Invalid publisher CSV near physical line {reader.line_num}: {error}") from error
        if lines.digest.hexdigest() != original.sha256:
            raise ValueError("Publisher original changed during typed replay")
    if (not found_header or count != original.rows
            or (peims and (record != end or annotation is not None))
            or (not peims and (tuple(notes) != _FOOTNOTES or next(transfer_annotations, None) is not None))):
        raise ValueError("Publisher records do not account for the full retained input")


def _staff_replay(literals: pa.Array, name: str, selection: _Typed) -> tuple[pa.Array, pa.Array]:
    year = selection.period.end_year
    assert year is not None and selection.staff_scope is not None
    unit = staff_fact_units(year)[name]
    allow_dot = staff_allows_unavailable_average(name, year, selection.staff_scope)
    if allow_dot and ("." not in selection.policy.tokens or selection.policy.tokens["."].status != ValueStatus.MISSING):
        raise ValueError("PEIMS staffing replay lacks its reviewed unavailable average pay status")
    values, statuses = [], []
    for token in literals.to_pylist():
        if token == "." and allow_dot:
            values.append(None)
            statuses.append(ValueStatus.MISSING.value)
            continue
        components = token.split(".")
        exact_digits = all(part.isascii() and part.isdecimal() for part in components)
        if not exact_digits or (unit == "fte" and (len(components) != 2 or len(components[1]) != 2)) or (
            unit != "fte" and len(components) != 1
        ):
            raise ValueError("PEIMS staffing replay contains an unsupported numeric token")
        values.append(Decimal(token) if unit == "fte" else int(token))
        statuses.append(ValueStatus.OBSERVED.value)
    return pa.array(values, type=pa.decimal128(16, 2) if unit == "fte" else pa.int64()), pa.array(statuses, type=pa.string())


def _batch(selection: _Typed, index: int, rows: list[list[str]], positions: list[tuple[int, int, int]]) -> pa.Table:
    fields = selection.fields
    columns = [pa.array(values, type=pa.string()) for values in zip(*rows, strict=True)]
    if selection.prepared is None:
        arrays: dict[str, pa.Array | pa.ChunkedArray] = {}
        for field, literals in zip(fields, columns, strict=True):
            name = field["field"]
            if field["disposition"] == "identity":
                arrays[name] = literals
            else:
                values, statuses = (_staff_replay(literals, name, selection) if selection.staff_scope is not None
                                    else _count_columns(pa.chunked_array([literals]), selection.policy))
                arrays[name], arrays[f"peims_status_{name}"] = values, statuses
                arrays[f"peims_raw_{name}"] = literals
        if selection.staff_scope is not None and STAFF_EMPLOYEE_COUNT in arrays:
            arrays[STAFF_EMPLOYEE_BASIS] = pa.array([
                "distinct_employees" if role.startswith("TOTAL ") else "positions"
                for role in arrays["staff_role"].to_pylist()], type=pa.string())
        arrays["peims_source_record"] = pa.array([position[0] for position in positions], type=pa.int64())
        if selection.district_types is not None:
            codes = [selection.district_types.get(identifier) for identifier in arrays["district_number"].to_pylist()]
            arrays[DISTRICT_TYPE_COLUMNS[0]] = pa.array(codes, type=pa.string())
            arrays[DISTRICT_TYPE_COLUMNS[1]] = pa.array([None if code is None else code == "I" for code in codes], type=pa.bool_())
        return pa.table(arrays)
    context = [pa.array(values, type=pa.int64()) for values in zip(*positions, strict=True)]
    grain = selection.transfer_grain
    assert grain is not None
    source = pa.Table.from_arrays([*columns, *context],
                                  names=[*transfer_columns(grain, year=selection.period.end_year), *_POSITIONS])
    source = _transfer_projection(source, grain=grain, period=selection.period, policy=selection.policy,
                                  region=selection.originals[index].region, layout="original")
    return source.append_column("transfer_source_input", pa.repeat(pa.scalar(index, type=pa.int32()), len(rows)))


def _append_historical_locators(table: pa.Table, history: _CampusHistory, seen: set[str]) -> pa.Table:
    details = pc.equal(table["transfer_row_kind"], "detail")
    keys = tuple(history.endpoints)
    ids = pa.array(keys, type=pa.string())
    records = pa.array([history.endpoints[key]["record"] for key in keys], type=pa.int64())
    locations = []
    for column, (campus, district, name) in zip(_HISTORICAL_FIELDS[:2], _ENDPOINTS, strict=True):
        offsets = pc.index_in(table[campus], value_set=ids)
        affected = table.select([campus, district, name]).filter(pc.and_(details, pc.is_valid(offsets)))
        for row in affected.group_by([campus, district, name]).aggregate([]).to_pylist():
            identifier = row[campus]
            evidence = history.endpoints[identifier]
            if row[district] != evidence["district"] or row[name].strip() != evidence["name"]:
                raise ValueError("Campus transfer source disagrees with the recorded fall identity")
            seen.add(identifier)
        locator = pc.if_else(details, pc.take(records, offsets), pa.scalar(None, type=pa.int64()))
        locations.append(locator)
        table = table.append_column(column, locator)
    attached = pc.or_(pc.is_valid(locations[0]), pc.is_valid(locations[1]))
    return table.append_column(_HISTORICAL_FIELDS[2], pc.if_else(attached, history.as_of, None))

class _Target:
    def __init__(self, file: pq.ParquetFile, stage: str, names: list[str], limit: int):
        if file.schema_arrow.names != names:
            raise ValueError(f"{stage} does not retain all projected native columns in order")
        self.stage = stage
        self.names = names
        self.limit = limit
        self.batches = file.iter_batches(batch_size=_BATCH_ROWS, use_threads=False)
        self.pending: pa.RecordBatch | None = None
        self.offset = 0
        self.fields: dict[str, _FieldReport] = {
            name: {"checked": 0, "mismatches": 0, "examples": []} for name in names}

    def compare(self, expected: pa.Table, source_input: int, records: pa.Array, examples_left: int) -> int:
        if set(expected.column_names) != set(self.names):
            raise ValueError(f"{self.stage} replay did not produce the complete projected schema")
        start = 0
        while start < len(expected):
            if self.pending is None:
                self.pending = next(self.batches, None)
                self.offset = 0
            if self.pending is None:
                raise ValueError(f"{self.stage} ends before the original publisher rows")
            if self.pending.nbytes > self.limit:
                raise _Uncovered(f"{self.stage} native batch exceeds its working-byte slot")
            count = min(len(expected) - start, len(self.pending) - self.offset)
            for name, stored in zip(self.names, self.pending.columns, strict=True):
                wanted = expected[name].slice(start, count)
                actual = stored.slice(self.offset, count)
                if actual.type != wanted.type:
                    raise ValueError(f"{self.stage} column {name!r} has a different native type")
                different = pc.or_(pc.fill_null(pc.not_equal(wanted, actual), False),
                                   pc.not_equal(pc.is_null(wanted), pc.is_null(actual)))
                mismatches = pc.sum(pc.cast(different, pa.int64())).as_py() or 0
                summary = self.fields[name]
                summary["checked"] += count
                summary["mismatches"] += mismatches
                if mismatches and examples_left:
                    for mismatch in pc.indices_nonzero(different).slice(0, examples_left).to_pylist():
                        original, value = wanted[mismatch].as_py(), actual[mismatch].as_py()
                        if isinstance(original, Decimal):
                            original = str(original)
                        if isinstance(value, Decimal):
                            value = str(value)
                        summary["examples"].append({"source_input": source_input, "source_record": records[start + mismatch].as_py(),
                            "raw": original[:1024] if isinstance(original, str) else original,
                            "actual": value[:1024] if isinstance(value, str) else value,
                            "raw_truncated": isinstance(original, str) and len(original) > 1024,
                            "actual_truncated": isinstance(value, str) and len(value) > 1024})
                        examples_left -= 1
            self.offset += count
            start += count
            if self.offset == len(self.pending):
                self.pending = None
        return examples_left

    def finish(self) -> dict[str, _FieldReport]:
        if self.pending is not None or next(self.batches, None) is not None:
            raise ValueError(f"{self.stage} includes rows outside the retained originals")
        for summary in self.fields.values():
            summary["examples_truncated"] = summary["mismatches"] > len(summary["examples"])
        return self.fields


def compare_typed(release: DataRelease, store: ProvenanceStore, key: EditionKey, *,
                  max_examples: int, working_bytes: int) -> dict[str, Any]:
    """Replay an admitted native leaf in bounded columnar batches."""
    if working_bytes < 512 * 1024:
        raise _Uncovered("Typed replay requires at least 512 KiB of working bytes")
    slot = working_bytes // 8
    selection = _selection(release, store, key, resource_bytes=slot)
    paths = {"artifact": selection.artifact}
    columns = {"artifact": selection.names}
    if selection.prepared is not None:
        paths["prepared"] = selection.prepared
        reverse = {target: source for source, target in selection.projection.items()}
        columns["prepared"] = [reverse.get(name, name) for name in selection.names
                               if selection.campus_history is None or name not in selection.campus_history.context]
    before = {stage: path.stat() for stage, path in paths.items()}
    with ExitStack() as stack:
        targets = {stage: _Target(stack.enter_context(pq.ParquetFile(path)), stage, columns[stage], slot)
                   for stage, path in paths.items()}
        examples_left = max_examples
        total = 0
        seen: set[str] = set()
        for index, original in enumerate(selection.originals):
            rows: list[list[str]] = []
            positions: list[tuple[int, int, int]] = []
            size = 0

            def compare_pending() -> None:
                nonlocal examples_left, total, size
                expected = _batch(selection, index, rows, positions)
                if expected.nbytes > slot:
                    raise _Uncovered("Projected typed batch exceeds its working-byte slot")
                records = expected["transfer_source_record" if selection.prepared else "peims_source_record"].chunk(0)
                if "prepared" in targets:
                    examples_left = targets["prepared"].compare(expected, index, records, examples_left)
                    if selection.campus_history is not None:
                        expected = _append_historical_locators(expected, selection.campus_history, seen)
                        if selection.campus_history.obsolescence is not None:
                            expected = _append_campus_obsolescence(expected, selection.campus_history.obsolescence)
                    expected = expected.rename_columns([selection.projection.get(name, name) for name in expected.column_names])
                examples_left = targets["artifact"].compare(expected, index, records, examples_left)
                total += len(rows)
                size = 0
                rows.clear()
                positions.clear()

            for row, record, first, last, byte_count in _wire(original, selection, limit=min(slot, 1024**2)):
                # Bound Python row staging before Arrow allocates the columnar projection.
                row_bytes = 4 * byte_count + 128 * (len(row) + 1)
                if row_bytes > slot:
                    raise _Uncovered("Publisher row exceeds its working-byte slot")
                if rows and (len(rows) == _BATCH_ROWS or size + row_bytes > slot):
                    compare_pending()
                rows.append(row)
                positions.append((record, first, last))
                size += row_bytes
            if rows:
                compare_pending()
        if selection.campus_history is not None and seen != selection.campus_history.endpoints.keys():
            raise ValueError("Recorded historical endpoint IDs do not all occur in publisher detail rows")
        diagnostics = {stage: target.finish() for stage, target in targets.items()}
    for stage, path in paths.items():
        initial, final = before[stage], path.stat()
        if (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns, initial.st_ctime_ns) != (
                final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns, final.st_ctime_ns):
            raise ValueError(f"{stage} changed during typed source comparison")
    if total != selection.rows:
        raise ValueError("Original publisher row count differs from the typed leaf contract")
    mismatches = sum(value["mismatches"] for fields in diagnostics.values() for value in fields.values())
    if selection.campus_history is not None:
        selection.lineage["historical_endpoint_evidence"] = {
            "fall_archive_as_of": selection.campus_history.as_of,
            "recorded_endpoint_count": len(selection.campus_history.endpoints),
            "companion_roster_gap_requalified": False,
        }
        if selection.campus_history.obsolescence is not None:
            selection.lineage["campus_obsolescence"] = selection.campus_history.obsolescence.recipe()
    return {"status": "FAIL_MISMATCH" if mismatches else "PASS_RAW_PARITY",
            "rows": total, "fields": len(selection.fields), "raw_cells": total * len(selection.fields),
            "compared_cells": sum(value["checked"] for fields in diagnostics.values() for value in fields.values()),
            "mismatches": mismatches, "stages": diagnostics, "lineage": selection.lineage,
            "working_bytes": working_bytes,
            "working_budget_scope": "Retained row staging, Arrow batches and comparison buffers, not a process RSS guarantee"}
