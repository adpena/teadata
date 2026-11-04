"""Materialize the fiscal 2025 Federal Report Card PPE source editions."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import platform
import re
import sys
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlencode

import pyarrow as pa
import pyarrow.compute as arrow_compute
import pyarrow.parquet as pq

from .column_schema import crosswalk_dataset, parse_column_schema, reader_options, table_input_sha256s
from .dataset_metadata import (
    STATEWIDE_SINGLETON_GRAIN, ArtifactRef, DataPeriod, DatasetMetadata, EditionKey, ResolvedDataset,
)
from .manifest import BuildRecorder, ContractError, canonical_json, file_identity
from .native_witnesses import _Rows
from .provenance import ProvenanceStore, validate_reference_records
from .teadata_config import Config, load_config, load_config_bytes
from .values import ValueStatus


DATASET = "frc_per_pupil_expenditure"
YEAR = 2025
PERIOD = DataPeriod("fiscal_year", YEAR - 1, YEAR)
UNIT = "currency_per_pupil"
BROKER = "https://rptsvr1.tea.texas.gov/cgi/sas/broker"
SELECTOR = "https://rptsvr1.tea.texas.gov/perfreport/frc/frc_srch.html?year=2025"
SOURCE_ROW = "frc_source_table_row"
_MAX_HTML_BYTES = 8 * 1024**2
_BASE_CONTROLS = [("_service", "marykay"), ("_program", "perfrept.perfmast.sas"), ("_debug", "0"),
                  ("frc", "yes"), ("ccyy", str(YEAR))]
_IDENTITY = {"state": (), "district": ("District", "District Name"),
             "campus": ("District", "District Name", "Campus", "Campus Name")}
_KEYS = {"state": [], "district": ["district_number"], "campus": ["campus_number"]}
_PPE_HEAD = ("Enrollment", "Total", "Site", "DistrictAllocation", "Total", "Site", "DistrictAllocation", "Total")
_FUND_HEAD = ["", "State & Localand Federal", "State & Local", "Federal"]
_BLANK_NOTE = ["", "Blank cell indicates there are no data available in the group."]
_FOOTER = [_BLANK_NOTE, ["", "TEA | Analytics, Assessment, and Reporting | Performance Reporting", ""]]
_TITLE = ["", "Texas Education Agency", ""]
_SELECTED_WITNESSES = {"district": "084908", "campus": "084908042"}
_NUMBER = r"(?:0|[1-9][0-9]{0,2})(?:,[0-9]{3})*"
pc: Any = arrow_compute


def _read_report(store: ProvenanceStore, receipt: str, scope: str, *, selected: str | None = None) -> str:
    record = store.read_record(receipt)
    artifact, parameters, metadata = record["artifact"], record["parameters"], record["source_metadata"]
    if selected is not None:
        level = {"state": "S", "district": "D", "campus": "C"}[scope]
        controls = [*_BASE_CONTROLS, ("lev", level), ("id", selected), ("prgopt", "reports/frc/frc.sas")]
        url = f"{BROKER}?{urlencode(controls)}"
        exact_request = (parameters.get("form_url") == BROKER and parameters.get("form_content_type") is None
                         and parameters.get("form_controls") == [list(pair) for pair in controls]
                         and metadata["source_url"] == SELECTOR)
    else:
        query = "&".join(f"{key}={value}" for key, value in _BASE_CONTROLS)
        flag = "&d_only=Y" if scope == "district" else ""
        url = f"{BROKER}?{query}&lev=S{flag}&prgopt=reports/frc/frc-link-report10.sas"
        exact_request = parameters.get("role") == "statewide_bulk" and parameters.get("scope") == scope
    if (not exact_request or record["kind"] != "acquisition" or record["method"] != "download"
            or record["dataset"] != DATASET or record["section"] != "data_sources"
            or (record["requested_year"], record["resolved_year"]) != (YEAR, YEAR)
            or (record["requested_url"], record["resolved_url"]) != (url, url)
            or parameters.get("http_method") != "GET" or parameters.get("http_status") != 200
            or metadata["version"] != str(YEAR) or metadata["period"] != PERIOD.to_dict()
            or metadata["publisher"] != "Texas Education Agency" or not record["downloaded_at"]
            or record["response_headers"].get("content-type", "").split(";")[0] != "text/html"
            or not 0 < artifact["size_bytes"] <= _MAX_HTML_BYTES):
        raise ContractError(f"FRC PPE receipt does not bind the exact {scope} report")
    path = store.verify_artifact(artifact["sha256"])
    if path.stat().st_size != artifact["size_bytes"]:
        raise ContractError("FRC PPE source size differs from its receipt")
    return path.read_text(encoding="utf-8")


def _selected_row(text: str, entity: str = "") -> tuple[int, list[str]]:
    before, marker, after = text.partition(">Part (x): Per-pupil Expenditure</h1>")
    body, end, _ = after.partition(">Part (xi):")
    if not marker or not end or "preceding fiscal year (2024-25)" not in body:
        raise ContractError("FRC PPE report section or fiscal period differs")
    title = _Rows(before).rows[:3]
    if (title[:2] != [_TITLE, ["2025 Federal Report Card"]] or len(title) != 3 or len(title[2]) != 1
            or (f"({entity})" not in title[2][0] if entity else title[2] != ["State"])):
        raise ContractError("FRC PPE report geography or edition differs")
    rows = _Rows(body).rows
    if len(rows) < 3 or rows[0] not in (_FUND_HEAD, ["", "State & Local and Federal", "State & Local", "Federal"]):
        raise ContractError("FRC PPE report fund groups differ")
    if rows[1] != ["", *_PPE_HEAD] or _BLANK_NOTE not in rows:
        raise ContractError("FRC PPE report column roles or blank-value note differs")
    totals = [(index, row[1:]) for index, row in enumerate(rows, 1) if row and row[0] == "Total"]
    if len(totals) != 1 or len(totals[0][1]) != len(_PPE_HEAD):
        raise ContractError("FRC PPE report must have one complete published total")
    return totals[0]


def _bulk_rows(text: str, scope: str) -> list[tuple[int, list[str]]]:
    rows = _Rows(text).rows
    identities = _IDENTITY[scope]
    prefix = [_TITLE, [f"2025 {scope.title()} Level Per-Pupil Expenditure Summary Report"],
              ["State"], _FUND_HEAD, [*identities, *_PPE_HEAD]]
    if rows[:5] != prefix or rows[-2:] != _FOOTER:
        raise ContractError("FRC PPE statewide bulk headings or footer differ")
    width = len(prefix[-1])
    districts: set[str] = set()
    seen: set[str] = set()
    district = name = ""
    result = []
    for index, row in enumerate(rows[5:-2], 6):
        if len(row) == width:
            district, name = row[:2]
            if re.fullmatch(r"[0-9]{6}", district) is None or district == "000000" or not name or district in districts:
                raise ContractError("FRC PPE district block is malformed or repeated")
            districts.add(district)
        elif scope == "campus" and len(row) == width - 2 and district:
            row = [district, name, *row]
        else:
            raise ContractError("FRC PPE statewide bulk row width differs")
        key = row[2] if scope == "campus" else district
        if scope == "campus" and (re.fullmatch(rf"{district}[0-9]{{3}}", key) is None or key.endswith("000") or not row[3]):
            raise ContractError("FRC PPE campus does not bind to the literal district group")
        if key in seen:
            raise ContractError("FRC PPE entity identifier is repeated")
        seen.add(key)
        result.append((index, row))
    return result


def _projection(rows: list[tuple[int, list[str]]], fields: list[dict[str, Any]]) -> tuple[pa.Table, dict[str, Any]]:
    arrays: dict[str, pa.Array] = {}
    support: dict[str, Any] = {"fields": {}, "context": [SOURCE_ROW]}
    physical, data = zip(*rows, strict=True)
    for field, values in zip(fields, zip(*data, strict=True), strict=True):
        name = field["field"]
        raw = pa.array(values, type=pa.string())
        if field["disposition"] == "identity":
            arrays[name] = raw
            continue
        enrollment = field["code"] == "Enrollment"
        missing = pc.equal(raw, "")
        valid = pc.match_substring_regex(raw, rf"^{_NUMBER}$" if enrollment else rf"^\$-?{_NUMBER}$")
        if not pc.all(valid if enrollment else pc.or_(valid, missing)).as_py():
            raise ContractError(f"FRC PPE contains an unreviewed token in {name}")
        number = pc.replace_substring(pc.replace_substring(raw, ",", ""), "$", "")
        arrays[name] = pc.cast(pc.if_else(missing, None, number), pa.int64(), safe=True)
        raw_name, status_name = f"frc_raw_{name}", f"frc_status_{name}"
        arrays[raw_name] = raw
        arrays[status_name] = pc.if_else(missing, ValueStatus.MISSING.value, ValueStatus.OBSERVED.value)
        support["fields"][name] = {"raw": raw_name, "status": status_name}
    arrays[SOURCE_ROW] = pa.array(physical, type=pa.int64())
    return pa.table(arrays), support


def _admit(config: Config, canonical: Config, store: ProvenanceStore) -> tuple[ResolvedDataset, list[dict[str, Any]], dict[str, Any]]:
    source = config.resolve_dataset(DATASET, YEAR, reader_profile="source", exact=True)
    reviewed = canonical.resolve_dataset(DATASET, YEAR, reader_profile="source", exact=True)
    schema = config.column_contract(DATASET, YEAR, reader_profile="source")
    reviewed_schema = canonical.column_contract(DATASET, YEAR, reader_profile="source")
    if (source.to_dict() != reviewed.to_dict() or not config.is_source_only(DATASET, YEAR)
            or source.metadata.period != PERIOD or schema is None or schema != reviewed_schema):
        raise ContractError("FRC PPE source differs from the packaged exact edition")
    policy = config.value_policies.get(source.metadata.value_policy)
    if (policy is None or policy != canonical.value_policies.get(source.metadata.value_policy)
            or set(policy.tokens) != {""} or policy.tokens[""].status != ValueStatus.MISSING):
        raise ContractError("FRC PPE requires the reviewed publisher blank-value policy")
    references = [rid for ref in source.metadata.references for rid in ref.provenance_ids]
    records = store.verify_record_closure(references)
    validate_reference_records(source, records.__getitem__, verify_all_claimed=True)
    tables = crosswalk_dataset(parse_column_schema(schema["text"].encode()), DATASET, str(YEAR))["tables"]
    if {table["table"] for table in tables} != set(_IDENTITY):
        raise ContractError("FRC PPE crosswalk must declare all three source scopes")
    return source, tables, reader_options(source.reader, "source")


def _build_leaf(recorder: BuildRecorder, target: ProvenanceStore, source: ResolvedDataset, mapping: dict[str, Any],
                options: dict[str, Any], scope: str, stage: Path, config_receipt: str, toolchain: dict[str, str]) -> dict[str, Any]:
    from .analytical import write_leaf_artifact

    receipt = options["acquisition_ids"][scope]
    record = target.read_record(receipt)
    (digest,) = table_input_sha256s(mapping)
    if record["artifact"]["sha256"] != digest:
        raise ContractError("FRC PPE source bytes differ from the reviewed crosswalk")
    references = [rid for ref in source.metadata.references for rid in ref.provenance_ids]
    text = _read_report(target, receipt, scope, selected="" if scope == "state" else None)
    rows = [_selected_row(text)] if scope == "state" else _bulk_rows(text, scope)
    fields = mapping["fields"]
    layout = mapping["layout"]
    units = {field["field"]: "count" if field["code"] == "Enrollment" else UNIT
             for field in fields if field["disposition"] == "fact"}
    if (tuple(field["code"] for field in fields) != (*_IDENTITY[scope], *_PPE_HEAD)
            or mapping["header_identity"] != "position" or mapping["metric_period"] != PERIOD.to_dict()
            or units != {name: metric.unit for name, metric in source.metadata.metrics.items()}
            or not rows or [index for index, _ in rows] != list(range(layout["first_data_record"], layout["last_data_record"] + 1))):
        raise ContractError("FRC PPE positional mapping, units or reviewed record range differs")
    witnesses = []
    if scope != "state":
        entity = _SELECTED_WITNESSES[scope]
        witness = options["witness_ids"][scope]
        if target.read_record(witness)["artifact"]["sha256"] == digest:
            raise ContractError("FRC PPE witness is the same artifact as the bulk source")
        if record["source_metadata"]["provenance_ids"] != [options["acquisition_ids"]["state"]]:
            raise ContractError("FRC PPE bulk report does not bind the published state source")
        _, values = _selected_row(_read_report(target, witness, scope, selected=entity), entity)
        matches = [row[-len(_PPE_HEAD):] for _, row in rows if row[0 if scope == "district" else 2] == entity]
        if matches != [values]:
            raise ContractError("FRC PPE selected report differs from the bulk observation")
        witnesses.append(witness)
    table, support = _projection(rows, fields)
    keys = _KEYS[scope]
    grain = STATEWIDE_SINGLETON_GRAIN if not keys else f"one row per {keys[0]}"
    edition = EditionKey(DATASET, str(YEAR), EditionKey.source_revision((digest,)), scope, scope, grain)
    recipe = {"operation": "frc_ppe_published_totals", "html_row_locator": "Part (x)" if not keys else "statewide bulk",
              "witness_receipts": witnesses, "format": "HTML"}
    contract = recorder.leaf_contract(edition, source, native_schema=table.schema.serialize().to_pybytes(),
        columns=table.column_names, key_columns=keys, rows=len(table), inputs=[digest], recipe=recipe,
        toolchain=toolchain, column_support=support)
    identity = sha256(canonical_json(contract).encode()).hexdigest()
    path = stage / f"{identity}.parquet"
    physical = write_leaf_artifact(path, table.to_batches(max_chunksize=2048), contract=contract, batch_bytes=8 * 1024**2)
    if not table.equals(pq.read_table(path).replace_schema_metadata(None), check_metadata=True):
        raise ContractError("FRC PPE native readback differs from the publisher projection")
    processing = target.record_processing(dataset=DATASET, inputs=list(dict.fromkeys([receipt, *references, *witnesses, config_receipt])),
        outputs={"leaf": path}, processor="teadata.frc_ppe", code_version=recorder.processor_sha256,
        parameters={"leaf_contract_sha256": identity, "recipe": recipe, "toolchain": toolchain},
        validation={"native_projection_readback_equal": True, "rows": len(table), "selected_report_cells": len(witnesses) * len(_PPE_HEAD),
                    "release_activation": False})
    ref = ArtifactRef(edition, physical["sha256"], physical["size_bytes"], path.name, canonical_json(contract), processing, "leaf")
    recorder.record_edition(edition, source, artifacts=[ref], acquisition_ids=[receipt])
    return {"key": edition.to_dict(), "artifact_id": identity, "rows": len(table), "source_receipt": receipt,
            "processing_record_id": processing, "selected_report_receipts": witnesses}


def materialize_frc_ppe(store: ProvenanceStore, *, config: Path, config_sha256: str, scopes: Sequence[str], output: Path) -> dict[str, Any]:
    """Build explicitly selected, unpublished FY2025 PPE leaves without deriving a state row."""
    from .release import _install_candidate, publish_release

    if not scopes or len(set(scopes)) != len(scopes) or set(scopes) - set(_IDENTITY):
        raise ContractError("Select distinct FRC PPE state, district or campus scopes")
    if output.exists() or output.is_symlink():
        raise FileExistsError("FRC PPE candidate requires a new exclusive output directory")
    with config.open("rb") as stream:
        raw = stream.read(2 * 1024**2 + 1)
    if not 0 < len(raw) <= 2 * 1024**2 or sha256(raw).hexdigest() != config_sha256:
        raise ContractError("FRC PPE source Config differs from its pin or exceeds 2 MiB")
    cfg = load_config_bytes(raw, path=config)
    source, tables, options = _admit(cfg, load_config(), store)
    toolchain = {"python": sys.version, "pyarrow": version("pyarrow"), "platform": platform.platform(),
                 "architecture": platform.machine()}
    lock = Path(__file__).parents[1] / "uv.lock"
    if lock.is_file():
        toolchain["lock_sha256"] = file_identity(lock)["sha256"]
    references = [rid for ref in source.metadata.references for rid in ref.provenance_ids]
    roots = [*references, *(options["acquisition_ids"][scope] for scope in scopes),
             *(options["witness_ids"][scope] for scope in scopes if scope != "state")]
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".frc-ppe-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir(mode=0o700)
        target = ProvenanceStore(stage / "provenance")
        recorder = BuildRecorder(cfg, target)
        for identifier in dict.fromkeys(roots):
            target.copy_record_from(store, identifier)
        captured = Path(temporary) / "source-config.input"
        captured.write_bytes(raw)
        config_receipt = target.record_acquisition(captured, ResolvedDataset("reference:frc_ppe_config", "reference", None, None,
            "", DatasetMetadata(title="Exact reviewed FY2025 FRC PPE source Config")))
        results = [_build_leaf(recorder, target, source, table, options, table["table"], stage, config_receipt, toolchain)
                   for table in tables if table["table"] in scopes]
        catalog = recorder.catalog(schema_version=4)
        publish_release(stage, catalog, schema_version=4)
        result = {"status": "unpublished_native_frc_ppe_candidate", "build_id": catalog.build_id,
                  "release_activation": False, "editions": results}
        (stage / "MATERIALIZATION.json").write_text(canonical_json(result) + "\n", encoding="utf-8")
        _install_candidate(stage, output)
        return result
