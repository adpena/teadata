"""Build district Indicator 11 leaves from retained public TEA regional workbooks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, localcontext
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
import platform
import re
import sys
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any
from zipfile import ZipFile

from .column_schema import crosswalk_dataset, parse_column_schema, reader_options, table_input_sha256s
from .dataset_metadata import ArtifactRef, DataPeriod, DatasetMetadata, EditionKey, ResolvedDataset
from .financial_leaves import NS, _package, _xml
from .manifest import BuildRecorder, ContractError, canonical_json, file_identity
from .provenance import ProvenanceStore, validate_reference_records
from .values import ValuePolicy, ValueStatus, measure

if TYPE_CHECKING:
    from xml.etree.ElementTree import Element

    import pyarrow as pa

    from .teadata_config import Config


DATASET = "spp_timely_initial_evaluation"
_RECEIPT_DATASET = "reference:tea_spp_district_detail"
SELECTOR = "https://rptsvr1.tea.texas.gov/idea/index.html"
BASE_URL = "https://rptsvr1.tea.texas.gov/idea/Region_SPP_PublicRptDownload"
MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
REGIONS = tuple(f"{region:02d}" for region in range(1, 21))
_SHEET = "Ind11"
_TABLE = "district"
_SOURCE_BYTES = 1024**2
_CONFIG_BYTES = 2 * 1024**2
_MAX_ROWS_PER_WORKBOOK = 2500
_IDENTITIES = ("esc_region", "district_number", "district_name")
_CONSENT = "initial_evaluation_consent_count"
_TIMELY = "timely_initial_evaluation_count"
_PERCENT = "timely_initial_evaluation_reported_percent"
_TARGET = "timely_initial_evaluation_met_target"
_METRICS = {_CONSENT: "count", _TIMELY: "count", _PERCENT: "percent", _TARGET: None}
_CONTEXT = ("spp_source_input", "spp_source_record", "spp_source_sheet", "spp_display_consistency")


@dataclass(frozen=True)
class _Cell:
    token: str | None
    kind: str


_ABSENT = _Cell(None, "absent")


def workbook_url(year: int, region: str) -> str:
    return f"{BASE_URL}/sy{year}/Region{region}_masked_{year - 1}-{year % 100:02d}.xlsx"


def _admit(config: Config, store: ProvenanceStore, year: int) -> tuple[
    ResolvedDataset, ValuePolicy, list[str], list[str], tuple[str, ...], list[str]
]:
    source = config.resolve_dataset(DATASET, year, exact=True, reader_profile="source")
    period = DataPeriod("school_year", year - 1, year)
    edition = str(year + 1)
    if (not config.is_source_only(DATASET, year) or source.metadata.version != edition
            or source.metadata.period != period or source.metadata.provenance_ids
            or {name: item.unit for name, item in source.metadata.metrics.items()} != _METRICS
            or any(name != item.field or item.period not in (None, period)
                   for name, item in source.metadata.metrics.items())):
        raise ContractError("SPP source requires its exact school year, report edition and metric definitions")
    options = reader_options(source.reader, "source")
    acquisitions = options.get("acquisition_ids")
    if (not isinstance(acquisitions, dict) or set(acquisitions) != set(REGIONS)
            or any(not isinstance(identifier, str) for identifier in acquisitions.values())
            or len(set(acquisitions.values())) != len(REGIONS)):
        raise ContractError("SPP source requires one distinct receipt per ESC region")
    schema = config.column_contract(DATASET, year, reader_profile="source")
    if schema is None:
        raise ContractError("SPP source requires a reviewed workbook crosswalk")
    document = parse_column_schema(schema["text"].encode())
    table, = crosswalk_dataset(document, DATASET, edition)["tables"]
    fields = table["fields"]
    if (table["table"] != _TABLE or table.get("header_identity") != "position"
            or "layout" in table or table.get("metric_period") != period.to_dict()
            or [field["field"] for field in fields] != [*_IDENTITIES, *_METRICS]
            or [field["disposition"] for field in fields] != ["identity"] * len(_IDENTITIES) + ["fact"] * len(_METRICS)):
        raise ContractError("SPP workbook crosswalk differs from the supported exact fields and period")
    headers = [field["label"] for field in fields]
    identifiers = [acquisitions[region] for region in REGIONS]
    references = list(dict.fromkeys(identifier for reference in source.metadata.references
                                    for identifier in reference.provenance_ids))
    records = store.verify_record_closure([*identifiers, *references])
    validate_reference_records(source, records.__getitem__, verify_all_claimed=True)
    policy = config.value_policies.get(source.metadata.value_policy)
    if (policy is None or {token: rule.status for token, rule in policy.tokens.items()} != {
            "-999": ValueStatus.SUPPRESSED, "nr": ValueStatus.MISSING}
            or any(rule.lower is not None or rule.upper is not None for rule in policy.tokens.values())
            or not any(ref.reference_id == policy.reference_id and ref.role == "masking_methodology"
                       and ref.url == SELECTOR and ref.status == "available" and ref.reviewed_sha256
                       and ref.provenance_ids for ref in source.metadata.references)):
        raise ContractError("SPP source requires its reviewed publisher masking policy and exact token rules")
    inputs = table_input_sha256s(table)
    if "input_sha256s" not in table or len(inputs) != len(REGIONS):
        raise ContractError("SPP crosswalk requires ordered source identities for every region")
    for region, identifier, digest in zip(REGIONS, identifiers, inputs, strict=True):
        record = records[identifier]
        url = workbook_url(year, region)
        parameters = record.get("parameters", {})
        if (record["kind"] != "acquisition" or record["method"] != "download"
                or record["dataset"] != _RECEIPT_DATASET or record["section"] != "reference"
                or record["requested_url"] != url or record["resolved_url"] != url
                or record["requested_year"] != year or record["resolved_year"] != year
                or record["source_metadata"].get("version") != edition
                or record["source_metadata"].get("source_url") != SELECTOR
                or record["source_metadata"].get("period") != period.to_dict()
                or parameters.get("esc_region") != region or parameters.get("report_edition") != year + 1
                or parameters.get("http_method") != "GET" or parameters.get("http_status") != 200
                or record["response_headers"].get("content-type", "").split(";", 1)[0].strip().lower() != MIME
                or record["artifact"]["sha256"] != digest or not 0 < record["artifact"]["size_bytes"] <= _SOURCE_BYTES):
            raise ContractError(f"SPP region {region} acquisition differs from its exact source contract")
    return source, policy, headers, identifiers, inputs, references


def _decode(cell: Element, strings: list[str]) -> _Cell:
    kind = cell.get("t")
    if kind not in (None, "n", "s") or cell.find(f"{NS}f") is not None:
        raise ContractError("SPP workbook has an unexpected cell type or formula")
    raw = cell.findtext(f"{NS}v")
    if kind == "s":
        if raw is None or re.fullmatch(r"[0-9]+", raw) is None or int(raw) >= len(strings):
            raise ContractError("SPP workbook has an invalid shared-string index")
        return _Cell(strings[int(raw)], "shared_string")
    return _Cell(raw, "numeric" if raw is not None else "blank")


def _counts(cell: _Cell, policy: ValuePolicy) -> tuple[int | None, ValueStatus]:
    result = measure(cell.token, policy, unit="count", minimum=0)
    if cell.kind in {"absent", "blank"} and result.status == ValueStatus.MISSING:
        return None, result.status
    if cell.kind != "numeric" or re.fullmatch(r"-?[0-9]+", cell.token or "") is None:
        raise ContractError("SPP workbook has an invalid count cell")
    if result.status == ValueStatus.SUPPRESSED:
        return None, result.status
    value = result.numeric
    if result.status != ValueStatus.OBSERVED or type(value) is not int or not 0 <= value < 1 << 63:
        raise ContractError("SPP workbook has an unsupported count token")
    return value, result.status


def _percent(cell: _Cell, policy: ValuePolicy) -> tuple[Decimal | None, ValueStatus]:
    result = measure(cell.token, policy, unit="percent", minimum=0, maximum=100)
    if cell.kind == "shared_string" and cell.token and result.status == ValueStatus.MISSING:
        return None, result.status
    if (cell.kind not in {"shared_string", "numeric"} or cell.token is None or result.status != ValueStatus.OBSERVED
            or re.fullmatch(r"[0-9]{1,3}\.[0-9]{1,2}", cell.token) is None):
        raise ContractError("SPP workbook has an unsupported published rate")
    return Decimal(cell.token), result.status


def _target(cell: _Cell, policy: ValuePolicy) -> tuple[bool | None, ValueStatus]:
    result = measure(cell.token, policy, unit="text")
    if cell.kind == "shared_string" and cell.token:
        if result.status == ValueStatus.MISSING:
            return None, result.status
        if result.status == ValueStatus.OBSERVED and cell.token in {"YES", "NO"}:
            return cell.token == "YES", result.status
    raise ContractError("SPP workbook has an unsupported target status")


def _consistency(consent: int | None, timely: int | None, percent: Decimal | None, target: bool | None,
                 count_statuses: tuple[ValueStatus, ValueStatus]) -> str:
    if percent is None:
        if target is not None or set(count_statuses) != {ValueStatus.MISSING}:
            raise ContractError("SPP no-report fields have inconsistent count or target states")
        return "not_reported"
    if target is None or ValueStatus.MISSING in count_statuses:
        raise ContractError("SPP published rate lacks expected counts or target state")
    if consent is None or timely is None:
        return "withheld_count_pair"
    if not 0 <= timely <= consent or consent == 0 or target != (timely == consent):
        raise ContractError("SPP count pair or independently published target disagrees")
    with localcontext() as context:
        context.prec = 50
        exact = Decimal(timely) * 100 / Decimal(consent)
        precision = Decimal(".01") if percent.as_tuple().exponent == -2 else Decimal(".1")
        direct = exact.quantize(precision, rounding=ROUND_HALF_UP)
        if direct == percent:
            return "direct_display_match"
        two_stage = exact.quantize(Decimal(".01"), rounding=ROUND_HALF_UP).quantize(precision, rounding=ROUND_HALF_UP)
        if two_stage == percent:
            return "consistent_with_two_stage_display_rounding"
    raise ContractError("SPP published rate differs from the observed count pair and established display patterns")


def _sheet(path: Path, headers: list[str], region: str, index: int, policy: ValuePolicy) -> list[dict[str, Any]]:
    with path.open("rb") as stream:
        raw = stream.read(_SOURCE_BYTES + 1)
    if not 0 < len(raw) <= _SOURCE_BYTES:
        raise ContractError("SPP workbook exceeds its bounded source size")
    with ZipFile(BytesIO(raw)) as archive:
        sheets, strings = _package(archive)
        root = _xml(archive, sheets[_SHEET])
    if root.find(f"{NS}mergeCells/{NS}mergeCell") is not None:
        raise ContractError("SPP Ind11 contains merged cells")
    rows = root.findall(f"{NS}sheetData/{NS}row")
    numbers = [int(row.attrib["r"]) for row in rows]
    if not numbers or numbers[0] != 1 or numbers[-1] > _MAX_ROWS_PER_WORKBOOK or numbers != sorted(set(numbers)):
        raise ContractError("SPP workbook has invalid physical row coordinates")
    records: list[dict[str, Any]] = []
    for number, row in zip(numbers, rows, strict=True):
        cells = {}
        for cell in row.findall(f"{NS}c"):
            position = cell.get("r", "")
            match = re.fullmatch(r"([A-Z]+)([1-9][0-9]*)", position)
            if match is None or int(match[2]) != number or match[1] in cells:
                raise ContractError("SPP workbook has an invalid physical cell coordinate")
            cells[match[1]] = _decode(cell, strings)
        if any(value.token is not None for column, value in cells.items() if column not in "ABCDEFG" or len(column) != 1):
            raise ContractError("SPP workbook has an unreviewed populated column")
        selected = [cells.get(column, _ABSENT) for column in "ABCDEFG"]
        tokens = [cell.token for cell in selected]
        if number == 1:
            if tokens != headers:
                raise ContractError("SPP workbook headings differ from the reviewed physical columns")
            continue
        if not any(token is not None for token in tokens):
            continue
        region_token, district, name = tokens[:3]
        if (region_token != region or district is None or district == "000000"
                or re.fullmatch(r"[0-9]{6}", district) is None or name is None or not name.strip()
                or any(cell.kind != "shared_string" for cell in selected[:3])):
            raise ContractError("SPP district identity does not match the workbook scope")
        consent, consent_status = _counts(selected[3], policy)
        timely, timely_status = _counts(selected[4], policy)
        percent, percent_status = _percent(selected[5], policy)
        target, target_status = _target(selected[6], policy)
        record: dict[str, Any] = dict(zip(_IDENTITIES, (region_token, district, name), strict=True))
        record.update(zip(_CONTEXT, (index, number, _SHEET,
                      _consistency(consent, timely, percent, target, (consent_status, timely_status))), strict=True))
        for field, (value, status), cell in zip(_METRICS, ((consent, consent_status), (timely, timely_status),
                                               (percent, percent_status), (target, target_status)), selected[3:], strict=True):
            record[field] = value
            record[f"spp_raw_{field}"] = cell.token
            record[f"spp_status_{field}"] = status.value
            record[f"spp_cell_type_{field}"] = cell.kind
        records.append(record)
    if not records:
        raise ContractError("SPP regional workbook has no district rows")
    return records


def _project(store: ProvenanceStore, inputs: tuple[str, ...], headers: list[str],
             policy: ValuePolicy) -> tuple[pa.Table, dict[str, Any], dict[str, int]]:
    import pyarrow as pa

    records = [row for index, (region, digest) in enumerate(zip(REGIONS, inputs, strict=True))
               for row in _sheet(store.verify_artifact(digest), headers, region, index, policy)]
    identifiers = [row["district_number"] for row in records]
    if len(set(identifiers)) != len(records):
        raise ContractError("SPP edition contains duplicate district IDs across workbook regions")
    types = {**dict.fromkeys(_IDENTITIES, pa.string()), _CONSENT: pa.int64(), _TIMELY: pa.int64(),
             _PERCENT: pa.decimal128(5, 2), _TARGET: pa.bool_(),
             **dict(zip(_CONTEXT, (pa.int16(), pa.int32(), pa.string(), pa.string()), strict=True))}
    support = {"fields": {}, "context": list(_CONTEXT)}
    for field in _METRICS:
        raw, status, kind = (f"spp_{prefix}_{field}" for prefix in ("raw", "status", "cell_type"))
        types.update(dict.fromkeys((raw, status, kind), pa.string()))
        support["fields"][field] = {"raw": raw, "status": status, "auxiliary": [kind]}
    schema = pa.schema([(name, kind) for name, kind in types.items()])
    table = pa.Table.from_pylist(sorted(records, key=lambda row: row["district_number"]), schema=schema)
    display = Counter(row["spp_display_consistency"] for row in records)
    return table, support, dict(display)


def materialize_spp_ind11(store: ProvenanceStore, *, config: Path, config_sha256: str,
                         years: Sequence[int], output: Path) -> dict[str, Any]:
    """Build an unpublished, checksum-pinned source-only district candidate."""
    import pyarrow.parquet as pq

    from .analytical import write_leaf_artifact
    from .release import DataRelease, _install_candidate, publish_release
    from .teadata_config import load_config_bytes

    if (not years or len(set(years)) != len(years)
            or any(type(year) is not int or not 2 <= year < 9999 for year in years)):
        raise ValueError("Select distinct exact SPP school-year endings")
    if output.exists() or output.is_symlink():
        raise FileExistsError("SPP candidate requires a new exclusive output directory")
    with config.open("rb") as stream:
        raw_config = stream.read(_CONFIG_BYTES + 1)
    if not 0 < len(raw_config) <= _CONFIG_BYTES or sha256(raw_config).hexdigest() != config_sha256:
        raise ContractError("SPP Config differs from its pin or exceeds its bounded size")
    cfg = load_config_bytes(raw_config, path=config)
    selections = [_admit(cfg, store, year) for year in sorted(years)]
    toolchain = {"python": sys.version, "pyarrow": version("pyarrow"),
                 "platform": platform.platform(), "architecture": platform.machine()}
    lock = Path(__file__).parents[1] / "uv.lock"
    if lock.is_file():
        toolchain["lock_sha256"] = file_identity(lock)["sha256"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".spp-ind11-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir(mode=0o700)
        retained = ProvenanceStore(stage / "provenance")
        recorder = BuildRecorder(cfg, retained)
        config_input = Path(temporary) / "source-config.input"
        config_input.write_bytes(raw_config)
        config_id = retained.record_acquisition(config_input, ResolvedDataset("reference:spp_ind11_config", "reference",
            None, None, "", DatasetMetadata(title="Exact reviewed SPP Indicator 11 source Config")))
        editions = []
        for source, policy, headers, identifiers, inputs, references in selections:
            for identifier in [*identifiers, *references]:
                retained.copy_record_from(store, identifier)
            table, support, display = _project(retained, inputs, headers, policy)
            edition = source.metadata.version
            if edition is None:
                raise ContractError("SPP report edition is required")
            key = EditionKey(DATASET, edition, EditionKey.source_revision(inputs), "district", _TABLE,
                             "one publisher row per district and school year")
            recipe = {"operation": "reviewed_spp_ind11_workbook_projection", "sheet": _SHEET,
                      "config_sha256": config_sha256, "regions": list(REGIONS)}
            contract = recorder.leaf_contract(key, source, native_schema=table.schema.serialize().to_pybytes(),
                columns=table.column_names, key_columns=["district_number"], rows=table.num_rows, inputs=inputs,
                recipe=recipe, toolchain=toolchain, column_support=support)
            identifier = sha256(canonical_json(contract).encode()).hexdigest()
            leaf = stage / f"{identifier}.parquet"
            physical = write_leaf_artifact(leaf, table.to_batches(), contract=contract, batch_bytes=4 * 1024**2)
            if not table.equals(pq.read_table(leaf).replace_schema_metadata(None), check_metadata=True):
                raise ContractError("SPP native leaf readback differs from its source projection")
            processing = retained.record_processing(dataset=DATASET, inputs=[*identifiers, *references, config_id],
                outputs={"leaf": leaf}, processor="teadata.spp_leaves.materialize_spp_ind11",
                code_version=recorder.processor_sha256, parameters={"leaf_contract_sha256": identifier,
                    "recipe": recipe, "toolchain": toolchain}, validation={"rows": table.num_rows,
                    "native_projection_readback_equal": True, "publisher_display_consistency": display,
                    "field_correctness_qualified": False, "release_activation": False})
            ref = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name, canonical_json(contract),
                              processing, "leaf")
            recorder.record_edition(key, source, artifacts=[ref], acquisition_ids=identifiers)
            editions.append({"key": key.to_dict(), "rows": table.num_rows, "artifact_id": identifier,
                             "processing_record_id": processing, "publisher_display_consistency": display})
        catalog = recorder.catalog(schema_version=4)
        manifest = publish_release(stage, catalog, schema_version=4)
        manifest_identity = file_identity(manifest)
        DataRelease(manifest, expected_sha256=manifest_identity["sha256"], offline=True).verify()
        result = {"status": "unpublished_native_spp_ind11_candidate", "build_id": catalog.build_id,
                  "manifest": manifest_identity, "field_correctness_qualified": False,
                  "release_activation": False, "editions": editions}
        (stage / "MATERIALIZATION.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        _install_candidate(stage, output)
    return result
