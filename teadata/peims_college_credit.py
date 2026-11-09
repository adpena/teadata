"""Build native PEIMS College Credit leaves from exact retained TEA reports."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import csv
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import platform
import re
import sys
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, urlsplit

from .column_schema import crosswalk_dataset, parse_column_schema, reader_options, table_input_sha256s
from .dataset_metadata import ArtifactRef, DataPeriod, DatasetMetadata, EditionKey, ResolvedDataset
from .manifest import BuildRecorder, ContractError, canonical_json, file_identity
from .provenance import ProvenanceStore, validate_reference_records
from .teadata_config import load_config, load_config_bytes
from .values import MeasurementUnit, ValueStatus

if TYPE_CHECKING:
    import pyarrow as pa

    from .teadata_config import Config


DATASET = "peims_college_credit"
YEARS = range(2022, 2027)
SERVICE_FIELDS = ("service_code", "service_description")
SERVICE_KEY = "college_credit_service_key"
SOURCE_RECORD = "peims_source_record"
BROKER = "https://rptsvr1.tea.texas.gov/cgi/sas/broker"
ENCODING = "cp1252"
_MAX_ARTIFACT_BYTES = 8 * 1024**2
_MAX_LINE_BYTES = 16 * 1024
_MAX_WINDOW_ROWS = 30_000
_BATCH_ROWS = 512
_RECIPE = "reviewed_peims_college_credit_csv_projection"
_REGION_NUMBERS = frozenset(f"{value:02d}" for value in range(1, 21))
_CHARTER_LABELS = frozenset({"TRADITIONAL ISD/CSD", "OPEN ENROLLMENT CHARTER"})
_SERVICE_CODE = re.compile(r"[A-Z0-9]{8}")
_DISTRICT_NUMBER = re.compile(r"[0-9]{6}")
_INTEGER = re.compile(r"[0-9]+")
_AVERAGE = re.compile(r"(?:0|[0-9]+\.[0-9])")
_GEO_FIELDS = {
    "state": (),
    "region": (("Region", "region_number"), ("Region Name", "region_name")),
    "district": (("Region", "region_number"), ("Region Name", "region_name"), ("District", "district_number"),
                 ("District Name", "district_name"), ("Charter Status", "charter_status")),
}
SCOPES = tuple(_GEO_FIELDS)
_SCOPE_TITLES = {"state": "Statewide Totals", "region": "Statewide Region Totals", "district": "Statewide District Totals"}
_FACTS = (
    ("Total CTE Students Earning College Credit", "students_earning_college_credit_count", MeasurementUnit.COUNT),
    ("Total CTE Students' College Credit Hours", "college_credit_hours", MeasurementUnit.CREDIT_HOURS),
    ("Average Hours per CTE Student", "average_credit_hours_per_student", MeasurementUnit.CREDIT_HOURS_PER_STUDENT),
    ("", "publisher_separator", None),
    ("Total Students Earning College Credit", "students_earning_college_credit_count", MeasurementUnit.COUNT),
    ("Total Students' College Credit Hours", "college_credit_hours", MeasurementUnit.CREDIT_HOURS),
    ("Average Hours per Student", "average_credit_hours_per_student", MeasurementUnit.CREDIT_HOURS_PER_STUDENT),
)


def college_credit_fields(year: int, scope: str) -> tuple[tuple[str, str, MeasurementUnit | None], ...]:
    """Describe one literal header and the three explicitly distinct CTE populations."""
    prefix = {2022: "cte_excluding_participants", 2026: "cte_excluding_explorers"}.get(year, "cte_all_program_groups")
    return (*((label, name, None) for label, name in _GEO_FIELDS[scope]),
            ("Service", SERVICE_FIELDS[0], None), ("Service Description", SERVICE_FIELDS[1], None),
            *((label, f"{prefix}_{name}" if index < 3 else name, unit) for index, (label, name, unit) in enumerate(_FACTS)))


def college_credit_keys(scope: str) -> list[str]:
    return [SERVICE_KEY] if scope == "state" else [f"{scope}_number", SERVICE_KEY]


def college_credit_support(year: int, scope: str) -> dict[str, Any]:
    return {"fields": {name: {"raw": f"peims_raw_{name}", "status": f"peims_status_{name}"}
                       for _, name, unit in college_credit_fields(year, scope) if unit is not None},
            "context": [SERVICE_KEY, SOURCE_RECORD]}


def matches_college_credit_header(table: dict[str, Any], year: int, scope: str) -> bool:
    layout = table.get("layout", {})
    header = 7 if scope == "state" else 9
    expected = tuple((label, label, name, "identity" if unit is None else "fact")
                     for label, name, unit in college_credit_fields(year, scope))
    return (table.get("header_identity") == "position" and layout.get("code_only_export") is True
            and (layout.get("code_record"), layout.get("label_record"), layout.get("first_data_record")) == (header, None, header + 1)
            and tuple((row["code"], row["label"], row["field"], row.get("disposition")) for row in table["fields"]) == expected)


@dataclass(frozen=True, slots=True)
class _Source:
    year: int
    scope: str
    config: ResolvedDataset
    receipt: str
    digest: str
    table: dict[str, Any]
    references: tuple[str, ...]
    rows: int

    @property
    def fields(self) -> tuple[tuple[str, str, MeasurementUnit | None], ...]:
        return college_credit_fields(self.year, self.scope)

    @property
    def support(self) -> dict[str, Any]:
        return college_credit_support(self.year, self.scope)

    @property
    def schema(self) -> pa.Schema:
        import pyarrow as pa

        types = {None: pa.string(), MeasurementUnit.COUNT: pa.int64(), MeasurementUnit.CREDIT_HOURS: pa.int64(),
                 MeasurementUnit.CREDIT_HOURS_PER_STUDENT: pa.decimal128(16, 1)}
        fields = [(name, types[unit]) for _, name, unit in self.fields]
        fields.extend((column, pa.string()) for support in self.support["fields"].values() for column in support.values())
        return pa.schema([*fields, (SERVICE_KEY, pa.string()), (SOURCE_RECORD, pa.int64())])


def _table(config: Config, year: int, scope: str) -> dict[str, Any]:
    contract = config.column_contract(DATASET, year, reader_profile="source")
    if contract is None:
        raise ContractError("College Credit requires a reviewed positional crosswalk")
    tables = crosswalk_dataset(parse_column_schema(contract["text"].encode()), DATASET, str(year))["tables"]
    return next((table for table in tables if table["table"] == scope), {})


def _admit(config: Config, canonical: Config, store: ProvenanceStore, year: int, scope: str) -> _Source:
    source = config.resolve_dataset(DATASET, year, reader_profile="source", exact=True)
    reviewed = canonical.resolve_dataset(DATASET, year, reader_profile="source", exact=True)
    table = _table(config, year, scope)
    metadata = source.metadata
    period = DataPeriod("school_year", year - 1, year)
    policy = config.value_policies.get(metadata.value_policy)
    facts = {name: unit.value for _, name, unit in college_credit_fields(year, scope) if unit is not None}
    if (not config.is_source_only(DATASET, year) or metadata != reviewed.metadata or source.reader != reviewed.reader
            or metadata.period != period or metadata.version != str(year) or metadata.provenance_ids
            or any(metric.field != name for name, metric in metadata.metrics.items())
            or {name: metric.unit for name, metric in metadata.metrics.items()} != facts
            or policy != canonical.value_policies.get(reviewed.metadata.value_policy) or policy is None
            or set(policy.tokens) != {"-999"}
            or any(rule.status != ValueStatus.SUPPRESSED for rule in policy.tokens.values())):
        raise ContractError("College Credit requires the exact reviewed periods, units and mask policy")
    if (not table or table != _table(canonical, year, scope) or not matches_college_credit_header(table, year, scope)
            or table["metric_period"] != period.to_dict()):
        raise ContractError("College Credit edition or scope differs from its packaged crosswalk")
    options = reader_options(source.reader, "source")
    receipt, rows = options["acquisition_ids"][scope], options["row_counts"][scope]
    layout = table["layout"]
    if type(rows) is not int or not 0 < rows <= layout["last_data_record"] - layout["first_data_record"] + 1 <= _MAX_WINDOW_ROWS:
        raise ContractError("College Credit row count or physical window exceeds its admission limits")
    references = tuple(dict.fromkeys(value for ref in metadata.references for value in ref.provenance_ids))
    records = store.verify_record_closure([receipt, *references])
    validate_reference_records(source, records.__getitem__, verify_all_claimed=True)
    record = records[receipt]
    controls = [["_service", "marykay"], ["_program", "adhoc.download_college_credit_report.sas"], ["RptClass", "CollegeCred"],
                ["_debug", "0"], ["school_year", str(year)], ["report", scope.title()], ["format", "csv"]]
    parameters = record["parameters"]
    url = urlsplit(record["requested_url"] or "")
    digest, = table_input_sha256s(table)
    identity = record["artifact"]
    if ((record["kind"], record["method"], record["section"], record["dataset"]) !=
            ("acquisition", "download", "reference", f"reference:{DATASET}:{scope}:csv")
            or (record["requested_year"], record["resolved_year"], record["source_metadata"]["version"]) != (year, year, str(year))
            or record["source_metadata"]["publisher"] != "Texas Education Agency" or not record["downloaded_at"]
            or record["resolved_url"] != record["requested_url"] or url._replace(query="").geturl() != BROKER
            or [list(pair) for pair in parse_qsl(url.query, keep_blank_values=True, strict_parsing=True)] != controls
            or (parameters.get("http_method"), parameters.get("http_status"), parameters.get("form_url")) != ("GET", 200, BROKER)
            or parameters.get("form_content_type") is not None or parameters.get("form_controls") != controls
            or record["response_headers"].get("content-type", "").split(";", 1)[0] not in {"application/csv", "text/csv"}
            or identity["sha256"] != digest or not 0 < identity["size_bytes"] <= _MAX_ARTIFACT_BYTES):
        raise ContractError("College Credit receipt differs from the reviewed statewide CSV request")
    if store.verify_artifact(digest).stat().st_size != identity["size_bytes"]:
        raise ContractError("College Credit receipt byte count differs")
    return _Source(year, scope, source, receipt, digest, table, references, rows)


def _records(path: Path, source: _Source) -> Iterator[tuple[int, list[str]]]:
    layout = source.table["layout"]
    header = [label for label, _, _ in source.fields]
    header_line, last = layout["code_record"], layout["last_data_record"]
    with path.open("rb") as stream:
        def physical_line() -> bytes:
            line = stream.readline(_MAX_LINE_BYTES + 1)
            if len(line) > _MAX_LINE_BYTES or not line.endswith(b"\n"):
                raise ContractError("College Credit record is absent, unterminated or too long")
            return line

        preamble = b"".join(physical_line() for _ in range(header_line - 1))
        titles = list(csv.reader(preamble.decode(ENCODING).splitlines(), strict=True))
        if [row for row in titles if row] != [["TEXAS EDUCATION AGENCY"],
            [f"College Credit Reports - {_SCOPE_TITLES[source.scope]}"], [f"PEIMS data for {source.year - 1}-{source.year}"]]:
            raise ContractError("College Credit CSV preamble belongs to another edition or scope")
        if list(csv.reader([physical_line().decode(ENCODING)], strict=True)) != [header]:
            raise ContractError("College Credit publisher heading differs")
        for number in range(header_line + 1, last + 1):
            row, = csv.reader([physical_line().decode(ENCODING)], strict=True)
            if len(row) != len(header):
                raise ContractError(f"College Credit CSV width differs at record {number}")
            if any(row):
                yield number, row
        footer = stream.read(_MAX_LINE_BYTES + 1)
    spans = [(1, header_line - 1, "preamble", preamble), (last + 1, last + len(footer.splitlines()), "footer", footer)]
    expected = [{"first_record": first, "last_record": stop, "kind": kind, "sha256": sha256(data).hexdigest()}
                for first, stop, kind, data in spans]
    if not footer or len(footer) > _MAX_LINE_BYTES or layout["annotations"] != expected:
        raise ContractError("College Credit preamble or footer differs from reviewed bytes")


def _batches(path: Path, source: _Source) -> Iterator[pa.RecordBatch]:
    import pyarrow as pa

    schema, fields, support = source.schema, source.fields, source.support["fields"]
    data: dict[str, list[Any]] = {name: [] for name in schema.names}
    identities: dict[str, tuple[str, ...]] = {}
    seen: set[tuple[str, str]] = set()
    totals: set[str] = set()
    offset = len(_GEO_FIELDS[source.scope])
    for number, row in _records(path, source):
        if row[offset + 5] != "":
            raise ContractError(f"College Credit separator is populated at record {number}")
        if source.scope != "state" and (row[0] not in _REGION_NUMBERS or not row[1].strip()):
            raise ContractError(f"Invalid College Credit region at record {number}")
        if source.scope == "district" and (not _DISTRICT_NUMBER.fullmatch(row[2]) or row[2] == "000000"
                                           or not row[3].strip() or row[4] not in _CHARTER_LABELS):
            raise ContractError(f"Invalid College Credit district at record {number}")
        entity = "Texas" if source.scope == "state" else row[2 if source.scope == "district" else 0]
        identity = tuple(row[:offset])
        if identities.setdefault(entity, identity) != identity:
            raise ContractError(f"Conflicting College Credit geography at record {number}")
        service, description = row[offset:offset + 2]
        total = (service, description) == (("Total", "") if source.scope == "state" else ("", "Total"))
        if not total and (not _SERVICE_CODE.fullmatch(service) or not description.strip()):
            raise ContractError(f"Invalid College Credit course or total at record {number}")
        key = "Total" if total else service
        if (entity, key) in seen:
            raise ContractError(f"Duplicate College Credit course or total at record {number}")
        seen.add((entity, key))
        if total:
            totals.add(entity)
        for (_, name, unit), token in zip(fields, row, strict=True):
            if unit is None:
                data[name].append(token)
                continue
            average = unit == MeasurementUnit.CREDIT_HOURS_PER_STUDENT
            suppressed = token == ("-999.0" if average else "-999")
            if not suppressed and not (_AVERAGE if average else _INTEGER).fullmatch(token):
                raise ContractError(f"Unreviewed College Credit token for {name} at record {number}")
            data[name].append(None if suppressed else Decimal(token) if average else int(token))
            data[support[name]["raw"]].append(token)
            data[support[name]["status"]].append((ValueStatus.SUPPRESSED if suppressed else ValueStatus.OBSERVED).value)
        data[SERVICE_KEY].append(key)
        data[SOURCE_RECORD].append(number)
        if len(data[SOURCE_RECORD]) == _BATCH_ROWS:
            yield pa.RecordBatch.from_pydict(data, schema=schema)
            for values in data.values():
                values.clear()
    if (len(seen) != source.rows or totals != set(identities)
            or (source.scope == "region" and totals != _REGION_NUMBERS)):
        raise ContractError("College Credit row count or complete publisher totals differ")
    if data[SOURCE_RECORD]:
        yield pa.RecordBatch.from_pydict(data, schema=schema)


def materialize_peims_college_credit(store: ProvenanceStore, *, config: Path, config_sha256: str,
                                     editions: Sequence[tuple[int, str]], output: Path) -> dict[str, Any]:
    """Write selected source-only editions and validate their exact native readback."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .analytical import write_leaf_artifact
    from .release import _install_candidate, publish_release

    if not editions or len(set(editions)) != len(editions) or any(
            type(year) is not int or year not in YEARS or scope not in _GEO_FIELDS for year, scope in editions):
        raise ValueError("Select distinct 2022–2026 College Credit state, region or district editions")
    if output.exists() or output.is_symlink():
        raise FileExistsError("College Credit candidate requires a new exclusive output directory")
    limit = 2 * 1024**2
    with config.open("rb") as stream:
        config_bytes = stream.read(limit + 1)
    if not 0 < len(config_bytes) <= limit or sha256(config_bytes).hexdigest() != config_sha256:
        raise ContractError("College Credit Config differs from its reviewed pin or exceeds 2 MiB")
    cfg, canonical = load_config_bytes(config_bytes, path=config), load_config()
    sources = [_admit(cfg, canonical, store, year, scope) for year, scope in sorted(editions)]
    toolchain = {"python": sys.version, "pyarrow": version("pyarrow"), "platform": platform.platform(),
                 "architecture": platform.machine()}
    lock = Path(__file__).parents[1] / "uv.lock"
    if lock.is_file():
        toolchain["lock_sha256"] = file_identity(lock)["sha256"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".peims-college-credit-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir(mode=0o700)
        target = ProvenanceStore(stage / "provenance")
        recorder = BuildRecorder(cfg, target)
        config_copy = Path(temporary) / "source-config.input"
        config_copy.write_bytes(config_bytes)
        config_id = target.record_acquisition(config_copy, ResolvedDataset("reference:peims_college_credit_config",
            "reference", None, None, "", DatasetMetadata(title="Exact reviewed PEIMS College Credit source Config")))
        for identifier in dict.fromkeys(value for source in sources for value in (source.receipt, *source.references)):
            target.copy_record_from(store, identifier)
        results = []
        for source in sources:
            raw = target.verify_artifact(source.digest)
            keys = college_credit_keys(source.scope)
            key = EditionKey(DATASET, str(source.year), EditionKey.source_revision((source.digest,)), source.scope,
                             source.scope, "one publisher row per " + " and ".join(keys))
            recipe = {"operation": _RECIPE, "encoding": ENCODING, "config_sha256": config_sha256}
            schema = source.schema
            contract = recorder.leaf_contract(key, source.config, native_schema=schema.serialize().to_pybytes(),
                columns=schema.names, key_columns=keys, rows=source.rows, inputs=[source.digest], recipe=recipe,
                toolchain=toolchain, column_support=source.support)
            leaf_id = sha256(canonical_json(contract).encode()).hexdigest()
            leaf = stage / f"{leaf_id}.parquet"
            physical = write_leaf_artifact(leaf, _batches(raw, source), contract=contract, batch_bytes=8 * 1024**2)
            with pq.ParquetFile(leaf) as readback:
                for expected, index in zip(_batches(raw, source), range(readback.num_row_groups), strict=True):
                    actual = readback.read_row_group(index, use_threads=False).replace_schema_metadata(None)
                    if not pa.Table.from_batches([expected]).equals(actual, check_metadata=True):
                        raise ContractError("College Credit native readback differs from the publisher projection")
            processing = target.record_processing(dataset=DATASET, inputs=[source.receipt, *source.references, config_id],
                outputs={"leaf": leaf}, processor="teadata.peims_college_credit", code_version=recorder.processor_sha256,
                parameters={"leaf_contract_sha256": leaf_id, "recipe": recipe, "toolchain": toolchain},
                validation={"native_projection_readback_equal": True, "rows": source.rows,
                            "field_correctness_qualified": False, "release_activation": False})
            artifact = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name,
                                   canonical_json(contract), processing, "leaf")
            recorder.record_edition(key, source.config, artifacts=[artifact], acquisition_ids=[source.receipt])
            results.append({"key": key.to_dict(), "artifact_id": leaf_id, "rows": source.rows,
                            "acquisition_id": source.receipt, "processing_record_id": processing})
        catalog = recorder.catalog(schema_version=4)
        publish_release(stage, catalog, schema_version=4)
        result = {"status": "unpublished_native_peims_college_credit_candidate", "build_id": catalog.build_id,
                  "field_correctness_qualified": False, "release_activation": False, "editions": results}
        (stage / "MATERIALIZATION.json").write_text(canonical_json(result) + "\n", encoding="utf-8")
        _install_candidate(stage, output)
        return result
