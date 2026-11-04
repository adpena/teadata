"""Materialize explicitly selected native PEIMS course report editions."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import csv
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from .column_schema import crosswalk_dataset, parse_column_schema, reader_options, table_input_sha256s
from .dataset_metadata import ArtifactRef, DataPeriod, DatasetMetadata, EditionKey, ResolvedDataset
from .manifest import BuildRecorder, ContractError, canonical_json, file_identity
from .peims_course_sources import CSV_HEADERS, DATASET, ENCODING, GRADE_LABELS, course_csv_preamble, open_course_report
from .provenance import ProvenanceStore, validate_reference_records
from .teadata_config import load_config, load_config_bytes
from .values import ValueStatus

if TYPE_CHECKING:
    import pyarrow as pa

    from .teadata_config import Config


COURSE_RECIPE = "reviewed_peims_course_csv_projection"
COURSE_FACT_UNITS = {"teacher_head_count": "count", "teacher_fte_count": "fte", "contract_teacher_head_count": "count",
                     "contract_teacher_fte_count": "fte", "course_enrollment_count": "count"}
SOURCE_RECORD = "peims_source_record"
_CSV_BLOCK_BYTES = 512 * 1024
_MAX_PHYSICAL_LINE = 16 * 1024
_MAX_RECORDS = 300_000
_MAX_BATCH_BYTES = 8 * 1024**2
_DISTRICT_CONTEXT = ("district_name", "charter_status", "region_number")


@dataclass(frozen=True)
class _CourseSource:
    year: int
    scope: str
    source: ResolvedDataset
    receipt: str
    digest: str
    table: dict[str, Any]
    references: tuple[str, ...]

    @property
    def names(self) -> list[str]:
        return [field["field"] for field in self.table["fields"]]

    @property
    def keys(self) -> list[str]:
        return ["school_year", *(["district_number"] if self.scope == "district" else []), "course_code"]

    @property
    def rows(self) -> int:
        return self.table["layout"]["last_data_record"] - 9

    @property
    def support(self) -> dict[str, Any]:
        return {"fields": {field: {"raw": f"peims_raw_{field}", "status": f"peims_status_{field}"}
                           for field in COURSE_FACT_UNITS}, "context": [SOURCE_RECORD]}

    @property
    def schema(self) -> pa.Schema:
        import pyarrow as pa

        native = {"count": pa.int64(), "fte": pa.decimal128(16, 2)}
        fields = [(name, native[COURSE_FACT_UNITS[name]] if name in COURSE_FACT_UNITS else pa.string())
                  for name in self.names]
        for support in self.support["fields"].values():
            fields.extend((name, pa.string()) for name in support.values())
        return pa.schema([*fields, (SOURCE_RECORD, pa.int64())])


def _table(config: Config, year: int, scope: str) -> dict[str, Any]:
    resource = config.column_contract(DATASET, year, reader_profile="source")
    if resource is None:
        raise ContractError("PEIMS course source requires its reviewed positional crosswalk")
    tables = crosswalk_dataset(parse_column_schema(resource["text"].encode()), DATASET, str(year))["tables"]
    return next((table for table in tables if table["table"] == scope), {})


def _admit(config: Config, canonical: Config, store: ProvenanceStore, year: int, scope: str) -> _CourseSource:
    source = config.resolve_dataset(DATASET, year, reader_profile="source", exact=True)
    canonical_source = canonical.resolve_dataset(DATASET, year, reader_profile="source", exact=True)
    table = _table(config, year, scope)
    reviewed = _table(canonical, year, scope)
    metadata = source.metadata
    policy = config.value_policies.get(metadata.value_policy)
    rule = None if policy is None else policy.tokens.get("-999")
    if (not config.is_source_only(DATASET, year) or metadata != canonical_source.metadata or metadata.version != str(year)
            or metadata.period != DataPeriod("school_year", year - 1, year) or metadata.provenance_ids
            or {key: metric.unit for key, metric in metadata.metrics.items()} != COURSE_FACT_UNITS
            or policy != canonical.value_policies.get(canonical_source.metadata.value_policy)
            or policy is None or set(policy.tokens) != {"-999"} or rule is None or rule.status != ValueStatus.SUPPRESSED):
        raise ContractError("PEIMS course source requires the exact period, units and masking policy")
    if (not table or table != reviewed or table["header_identity"] != "position"
            or tuple(field["code"] for field in table["fields"]) != CSV_HEADERS[scope]
            or {field["field"] for field in table["fields"] if field["disposition"] == "fact"} != set(COURSE_FACT_UNITS)
            or table["metric_period"] != metadata.period.to_dict()):
        raise ContractError("PEIMS course edition or scope differs from the packaged crosswalk")
    layout = table["layout"]
    if (layout["code_record"] != 9 or layout["label_record"] is not None or layout["first_data_record"] != 10
            or not 9 < layout["last_data_record"] <= _MAX_RECORDS):
        raise ContractError("PEIMS course physical framing differs")
    references = tuple(dict.fromkeys(identifier for ref in metadata.references for identifier in ref.provenance_ids))
    records = store.verify_record_closure(references)
    validate_reference_records(source, records.__getitem__, verify_all_claimed=True)
    receipt = reader_options(source.reader, "source")["acquisition_ids"][scope]
    if receipt != reader_options(canonical_source.reader, "source")["acquisition_ids"][scope]:
        raise ContractError("PEIMS course receipt differs from the packaged source selection")
    path, record = open_course_report(store, receipt, year, scope, "csv")
    digest, = table_input_sha256s(table)
    if record["artifact"]["sha256"] != digest or not path.is_file():
        raise ContractError("PEIMS course acquisition differs from the reviewed raw input")
    return _CourseSource(year, scope, source, receipt, digest, table, references)


def _body_window(path: Path, source: _CourseSource) -> tuple[int, int]:
    def physical_line() -> bytes:
        line = stream.readline(_MAX_PHYSICAL_LINE + 1)
        if not line.endswith(b"\n") or len(line) > _MAX_PHYSICAL_LINE:
            raise ContractError("PEIMS course physical record is absent, unterminated or too long")
        return line

    with path.open("rb") as stream:
        preamble = [physical_line() for _ in range(9)]
        if list(csv.reader([line.decode(ENCODING) for line in preamble], strict=True)) != course_csv_preamble(
                source.year, source.scope):
            raise ContractError("PEIMS course preamble or ninth physical header differs")
        start = stream.tell()
        for _ in range(source.rows):
            physical_line()
        stop = stream.tell()
        footer = stream.read(_MAX_PHYSICAL_LINE + 1)
    if len(footer) > _MAX_PHYSICAL_LINE:
        raise ContractError("PEIMS course footer exceeds its admitted size")
    bounds = [(1, 8, b"".join(preamble[:8])), (10 + source.rows, 9 + source.rows + len(footer.splitlines()), footer)]
    annotations = source.table["layout"]["annotations"]
    if len(annotations) != len(bounds) or any(
            (annotation["first_record"], annotation["last_record"], annotation["sha256"])
            != (first, last, sha256(content).hexdigest())
            for annotation, (first, last, content) in zip(annotations, bounds, strict=True)):
        raise ContractError("PEIMS course raw preamble or footer differs from its reviewed bytes")
    return start, stop - start


def _batches(path: Path, source: _CourseSource) -> Iterator[pa.RecordBatch]:
    import pyarrow as pa
    import pyarrow.compute as arrow_compute
    import pyarrow.csv as arrow_csv

    pc: Any = arrow_compute

    def require(valid, message):
        if pc.all(pc.fill_null(valid, False)).as_py() is not True:
            raise ContractError(message)

    offset, size = _body_window(path, source)
    names, schema = source.names, source.schema
    support = source.support["fields"]
    next_record = 10
    seen_districts: set[str] = set()
    active_district: str | None = None
    active_context: tuple[str, ...] = ()
    seen_courses: set[str] = set()
    with pa.memory_map(str(path)) as archive, archive.get_stream(offset, size) as body:
        with arrow_csv.open_csv(body,
            read_options=arrow_csv.ReadOptions(column_names=names, encoding=ENCODING, block_size=_CSV_BLOCK_BYTES,
                                               use_threads=False),
            parse_options=arrow_csv.ParseOptions(ignore_empty_lines=False, newlines_in_values=False),
            convert_options=arrow_csv.ConvertOptions(column_types=dict.fromkeys(names, pa.string()),
                                                     strings_can_be_null=False, quoted_strings_can_be_null=False,
                                                     null_values=[])) as rows:
            for batch in rows:
                values = dict(zip(names, batch.columns, strict=True))
                aggregation = "STATE" if source.scope == "state" else "COUNTYREGIONDISTRICT"
                for name, token in (("source_scope", "STATE"), ("aggregation_level", aggregation),
                                    ("school_year", source.source.metadata.period.label)):
                    require(pc.equal(values[name], token), "PEIMS course row differs from its selected scope or year")
                require(pc.match_substring_regex(values["course_code"], r"^[A-Z0-9]{8}$"), "Malformed course code")
                require(pc.is_in(values["grade_level"], value_set=pa.array(sorted(GRADE_LABELS))), "Unreviewed course grade label")
                for name in ("subject", "course_title"):
                    require(pc.match_substring_regex(values[name], r"\S"), "Blank PEIMS course subject or title")
                runs = [(batch.num_rows, None)]
                if source.scope == "district":
                    require(pc.match_substring_regex(values["district_number"], r"^[0-9]{6}$"), "Malformed district ID")
                    require(pc.not_equal(values["district_number"], "000000"), "Zero district ID")
                    require(pc.match_substring_regex(values["region_number"], r"^(?:0[1-9]|1[0-9]|20)$"), "Malformed region code")
                    require(pc.is_in(values["charter_status"], value_set=pa.array(["OPEN ENROLLMENT CHARTER", "TRADITIONAL ISD/CSD"])),
                            "Unreviewed district charter status")
                    require(pc.match_substring_regex(values["district_name"], r"\S"), "Blank district name")
                    encoded = pc.run_end_encode(values["district_number"])
                    runs = list(zip(encoded.run_ends.to_pylist(), encoded.values.to_pylist(), strict=True))
                start = 0
                for stop, district in runs:
                    width = stop - start
                    if district is not None:
                        if district != active_district:
                            if district in seen_districts:
                                raise ContractError("PEIMS course district group reappeared out of order")
                            seen_districts.add(district)
                            active_district = district
                            active_context = tuple(values[name][start].as_py() for name in _DISTRICT_CONTEXT)
                            seen_courses.clear()
                        for name, token in zip(_DISTRICT_CONTEXT, active_context, strict=True):
                            require(pc.equal(values[name].slice(start, width), token), "Conflicting PEIMS course district identity")
                    old_count = len(seen_courses)
                    seen_courses.update(values["course_code"].slice(start, width).to_pylist())
                    if len(seen_courses) != old_count + width:
                        raise ContractError("Duplicate PEIMS course code within the selected geography")
                    start = stop
                for name, unit in COURSE_FACT_UNITS.items():
                    raw = values[name]
                    numeric = pc.match_substring_regex(raw, r"^[0-9]+(?:\.[0-9]{2})?$" if unit == "fte" else r"^[0-9]+$")
                    allowed = pc.or_(numeric, pc.equal(raw, "-999")) if name == "course_enrollment_count" else numeric
                    require(allowed, f"Unreviewed PEIMS course numeric token in {name}")
                    values[name] = pc.cast(pc.if_else(numeric, raw, pa.scalar(None, pa.string())), schema.field(name).type)
                    values[support[name]["raw"]] = raw
                    values[support[name]["status"]] = pc.if_else(numeric, ValueStatus.OBSERVED.value, ValueStatus.SUPPRESSED.value)
                end = next_record + batch.num_rows
                values[SOURCE_RECORD] = pa.array(range(next_record, end), type=pa.int64())
                yield pa.RecordBatch.from_arrays([values[name] for name in schema.names], schema=schema)
                next_record = end
    if next_record != source.rows + 10:
        raise ContractError("PEIMS course row count differs from its reviewed physical range")


def materialize_peims_courses(store: ProvenanceStore, *, config: Path, config_sha256: str,
                             editions: Sequence[tuple[int, str]], output: Path) -> dict[str, Any]:
    """Build source-only state and district course leaves and verify native readback."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .analytical import write_leaf_artifact
    from .release import _install_candidate, publish_release

    if (not editions or len(set(editions)) != len(editions)
            or any(type(year) is not int or not 2 <= year <= 9999 or scope not in CSV_HEADERS for year, scope in editions)):
        raise ValueError("Select distinct exact PEIMS course school years and state/district scopes")
    if output.exists() or output.is_symlink():
        raise FileExistsError("PEIMS course candidate requires a new exclusive output directory")
    with config.open("rb") as stream:
        config_bytes = stream.read(2 * 1024**2 + 1)
    if not 0 < len(config_bytes) <= 2 * 1024**2 or sha256(config_bytes).hexdigest() != config_sha256:
        raise ContractError("PEIMS course Config differs from its pin or exceeds 2 MiB")
    cfg = load_config_bytes(config_bytes, path=config)
    canonical = load_config()
    sources = [_admit(cfg, canonical, store, year, scope) for year, scope in sorted(editions)]
    toolchain = {"python": sys.version, "pyarrow": version("pyarrow"), "platform": platform.platform(),
                 "architecture": platform.machine()}
    lock = Path(__file__).parents[1] / "uv.lock"
    if lock.is_file():
        toolchain["lock_sha256"] = file_identity(lock)["sha256"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".peims-courses-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir(mode=0o700)
        target = ProvenanceStore(stage / "provenance")
        recorder = BuildRecorder(cfg, target)
        captured = Path(temporary) / "source-config.input"
        captured.write_bytes(config_bytes)
        config_id = target.record_acquisition(captured, ResolvedDataset("reference:peims_course_config", "reference", None,
            None, "", DatasetMetadata(title="Exact reviewed PEIMS course source Config")))
        results = []
        for identifier in dict.fromkeys(identifier for source in sources for identifier in (source.receipt, *source.references)):
            target.copy_record_from(store, identifier)
        for source in sources:
            raw = target.verify_artifact(source.digest)
            key = EditionKey(DATASET, str(source.year), EditionKey.source_revision((source.digest,)), source.scope,
                             source.scope, "one publisher row per " + " and ".join(source.keys))
            recipe = {"operation": COURSE_RECIPE, "encoding": ENCODING, "config_sha256": config_sha256,
                      "input_block_bytes": _CSV_BLOCK_BYTES}
            schema = source.schema
            contract = recorder.leaf_contract(key, source.source, native_schema=schema.serialize().to_pybytes(),
                columns=schema.names, key_columns=source.keys, rows=source.rows, inputs=[source.digest],
                recipe=recipe, toolchain=toolchain, column_support=source.support)
            leaf_id = sha256(canonical_json(contract).encode()).hexdigest()
            leaf = stage / f"{leaf_id}.parquet"
            physical = write_leaf_artifact(leaf, _batches(raw, source), contract=contract, batch_bytes=_MAX_BATCH_BYTES)
            with pq.ParquetFile(leaf) as readback:
                for expected, index in zip(_batches(raw, source), range(readback.num_row_groups), strict=True):
                    actual = readback.read_row_group(index, use_threads=False).replace_schema_metadata(None)
                    if not pa.Table.from_batches([expected]).equals(actual, check_metadata=True):
                        raise ContractError("PEIMS course native leaf readback differs from the publisher projection")
            processing = target.record_processing(dataset=DATASET, inputs=[source.receipt, *source.references, config_id],
                outputs={"leaf": leaf}, processor="teadata.peims_courses", code_version=recorder.processor_sha256,
                parameters={"leaf_contract_sha256": leaf_id, "recipe": recipe, "toolchain": toolchain},
                validation={"native_projection_readback_equal": True, "rows": source.rows,
                            "field_correctness_qualified": False, "release_activation": False})
            artifact = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name,
                                   canonical_json(contract), processing, "leaf")
            recorder.record_edition(key, source.source, artifacts=[artifact], acquisition_ids=[source.receipt])
            results.append({"key": key.to_dict(), "artifact_id": leaf_id, "rows": source.rows,
                            "acquisition_id": source.receipt, "processing_record_id": processing})
        catalog = recorder.catalog(schema_version=4)
        publish_release(stage, catalog, schema_version=4)
        result = {"status": "unpublished_native_peims_course_candidate", "build_id": catalog.build_id,
                  "field_correctness_qualified": False, "release_activation": False, "editions": results}
        (stage / "MATERIALIZATION.json").write_text(canonical_json(result) + "\n", encoding="utf-8")
        _install_candidate(stage, output)
        return result
