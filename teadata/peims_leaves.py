"""Build unpublished native leaves for reviewed PEIMS student report tables."""

from __future__ import annotations

from collections.abc import Sequence
import csv
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from .column_schema import crosswalk_dataset, parse_column_schema, reader_options, table_input_sha256s
from .dataset_metadata import ArtifactRef, DataPeriod, DatasetMetadata, EditionKey, ResolvedDataset
from .manifest import BuildRecorder, canonical_json, file_identity
from .provenance import ProvenanceStore, validate_reference_records
from .values import ValuePolicy, ValueStatus, measure

if TYPE_CHECKING:
    import pyarrow as pa

    from .teadata_config import Config


ENROLLMENT = "peims_student_enrollment_by_ethnicity"
PROGRAMS = "peims_student_programs_and_special_populations"
PRIMARY_DISABILITY = "peims_special_education_by_primary_disability"
_STUDENT_SCOPES = frozenset({"state", "district", "campus"})
SCOPES_BY_DATASET = {
    ENROLLMENT: _STUDENT_SCOPES,
    PROGRAMS: _STUDENT_SCOPES,
    PRIMARY_DISABILITY: frozenset({"state", "district"}),
}
DATASETS = tuple(SCOPES_BY_DATASET)
SOURCE_BYTES = 8 * 1024**2
BATCH_BYTES = 8 * 1024**2
BATCH_ROWS = 4096
MAX_SOURCE_RECORDS = 60_000
ENCODING = "cp1252"


def _admit_source(config: Config, dataset: str, year: int, store: ProvenanceStore) -> tuple[
    ResolvedDataset, ValuePolicy, list[dict[str, Any]], dict[str, str], list[str]
]:
    source = config.resolve_dataset(dataset, year, reader_profile="source", exact=True)
    if (not config.is_source_only(dataset, year) or source.metadata.version != str(year)
            or source.metadata.period != DataPeriod("school_year", year - 1, year)
            or source.metadata.provenance_ids):
        raise ValueError("PEIMS requires a source-only exact annual edition with acquisitions assigned by scope")
    policy = config.value_policies.get(source.metadata.value_policy)
    if policy is None:
        raise ValueError("PEIMS requires its reviewed value policy")
    if any(token.isascii() and token.isdecimal() for token in policy.tokens):
        raise ValueError("This PEIMS reader requires nonnegative integer tokens to be observed counts")
    resource = config.column_contract(dataset, year, reader_profile="source")
    if resource is None:
        raise ValueError("PEIMS requires its reviewed positional crosswalk")
    tables = crosswalk_dataset(parse_column_schema(resource["text"].encode()), dataset, str(year))["tables"]
    identifiers = reader_options(source.reader, "source").get("acquisition_ids")
    scopes = SCOPES_BY_DATASET[dataset]
    if (len(tables) != len(scopes) or {table["table"] for table in tables} != scopes
            or not isinstance(identifiers, dict) or set(identifiers) != scopes
            or any(not isinstance(identifier, str) for identifier in identifiers.values())
            or len(set(identifiers.values())) != len(scopes)):
        raise ValueError("PEIMS requires one distinct acquisition per supported publisher table")
    references = [identifier for ref in source.metadata.references for identifier in ref.provenance_ids]
    records = store.verify_record_closure([*identifiers.values(), *references])
    validate_reference_records(source, records.__getitem__, verify_all_claimed=True)
    for table in tables:
        record = records[identifiers[table["table"]]]
        expected, = table_input_sha256s(table)
        if table.get("metric_period") != source.metadata.period.to_dict():
            raise ValueError("PEIMS crosswalk period differs from its exact source edition")
        observed = record.get("source_metadata", {})
        content_type = record.get("response_headers", {}).get("content-type", "").split(";", 1)[0].strip().lower()
        if (record["kind"] != "acquisition" or record["method"] != "download"
                or record["dataset"] != dataset or record["section"] != source.section
                or record["requested_year"] != year or record["resolved_year"] != year
                or observed.get("version") != str(year) or observed["period"] != source.metadata.period.to_dict()
                or observed.get("source_url") != source.metadata.source_url or content_type not in {"text/csv", "application/csv"}
                or record["artifact"]["sha256"] != expected or not 0 < record["artifact"]["size_bytes"] <= SOURCE_BYTES):
            raise ValueError(f"PEIMS {dataset} {table['table']} acquisition differs from its exact source contract")
    return source, policy, tables, identifiers, references


def _source_body(path: Path, table: dict[str, Any]) -> tuple[list[str], bytes, range]:
    layout = table.get("layout")
    if table.get("header_identity") != "position" or not layout or layout["label_record"] is not None:
        raise ValueError("PEIMS requires a reviewed single-header positional layout")
    with path.open("rb") as stream:
        raw = stream.read(SOURCE_BYTES + 1)
    expected, = table_input_sha256s(table)
    if not 0 < len(raw) <= SOURCE_BYTES or sha256(raw).hexdigest() != expected:
        raise ValueError("PEIMS source differs from its pin or exceeds the byte limit")
    if raw.count(b"\n") + raw.count(b"\r") > 2 * MAX_SOURCE_RECORDS:
        raise ValueError("PEIMS source exceeds the physical record limit")
    physical = raw.splitlines(keepends=True)
    start, end, header = layout["first_data_record"], layout["last_data_record"], layout["code_record"]
    if not 1 <= header < start <= end <= len(physical) <= MAX_SOURCE_RECORDS:
        raise ValueError("PEIMS source differs from its reviewed physical bounds")
    intervals = [(header, header), (start, end)]
    for annotation in layout["annotations"]:
        first, last = annotation["first_record"], annotation["last_record"]
        if not 1 <= first <= last <= len(physical) or sha256(b"".join(physical[first - 1:last])).hexdigest() != annotation["sha256"]:
            raise ValueError("PEIMS preamble or footer differs from its reviewed bytes")
        intervals.append((first, last))
    next_record = 1
    for first, last in sorted(intervals):
        if first != next_record:
            raise ValueError("PEIMS source contains overlapping or unclaimed physical records")
        next_record = last + 1
    if next_record != len(physical) + 1:
        raise ValueError("PEIMS source contains unclaimed tail records")
    observed, = csv.reader([physical[header - 1].decode(ENCODING)], strict=True)
    fields = table["fields"]
    if observed != [field["code"] for field in fields] or observed != [field["label"] for field in fields]:
        raise ValueError("PEIMS source differs from its reviewed ordered headers")
    return [field["field"] for field in fields], b"".join(physical[start - 1:end]), range(start, end + 1)


def _all_valid(values: pa.Array | pa.ChunkedArray) -> bool:
    import pyarrow.compute as arrow_compute

    pc: Any = arrow_compute
    return pc.all(pc.fill_null(values, False)).as_py() is True


def _count_columns(raw: pa.ChunkedArray, policy: ValuePolicy) -> tuple[pa.ChunkedArray, pa.ChunkedArray]:
    import pyarrow as pa
    import pyarrow.compute as arrow_compute

    pc: Any = arrow_compute
    observed = pc.match_substring_regex(raw, r"^[0-9]+$")
    special = pc.unique(pc.filter(raw, pc.invert(observed)))
    interpreted = [measure(token, policy, unit="count", minimum=0) for token in special.to_pylist()]
    if any(value.status not in {ValueStatus.MISSING, ValueStatus.SUPPRESSED} for value in interpreted):
        raise ValueError("PEIMS source contains an unknown or invalid count token")
    values = pc.cast(pc.if_else(observed, raw, pa.scalar(None, pa.string())), pa.int64())
    statuses = pc.take(pa.array([value.status.value for value in interpreted], type=pa.string()),
                       pc.index_in(raw, value_set=special))
    return values, pc.if_else(observed, ValueStatus.OBSERVED.value, statuses)


def _project(path: Path, table: dict[str, Any], source: ResolvedDataset, policy: ValuePolicy) -> tuple[
    pa.Table, list[str], dict[str, Any]
]:
    import pyarrow as pa
    import pyarrow.compute as arrow_compute
    import pyarrow.csv as pacsv

    pc: Any = arrow_compute
    names, body, records = _source_body(path, table)
    if any(name.startswith("peims_") for name in names):
        raise ValueError("PEIMS reviewed fields collide with reserved support columns")
    literals = pacsv.read_csv(pa.BufferReader(body),
        read_options=pacsv.ReadOptions(column_names=names, use_threads=False, encoding=ENCODING, block_size=len(body)),
        parse_options=pacsv.ParseOptions(ignore_empty_lines=False, newlines_in_values=False),
        convert_options=pacsv.ConvertOptions(column_types=dict.fromkeys(names, pa.string()), strings_can_be_null=False,
                                             quoted_strings_can_be_null=False, null_values=[]))
    scope = table["table"]
    key_columns = ["school_year"]
    if scope != "state":
        key_columns.append(f"{scope}_number")
    if source.dataset == ENROLLMENT:
        key_columns.append("ethnicity")
    if not set(key_columns) <= set(names) or literals.num_rows != len(records):
        raise ValueError("PEIMS native table differs from its reviewed layout or grain")
    if not _all_valid(pc.equal(literals["school_year"], source.metadata.period.label)):
        raise ValueError("PEIMS row period differs from its source edition")
    for name, width in (("district_number", 6), ("campus_number", 9)):
        if name in names and (not _all_valid(pc.match_substring_regex(literals[name], rf"^[0-9]{{{width}}}$"))
                              or not _all_valid(pc.not_equal(literals[name], "0" * width))):
            raise ValueError(f"PEIMS contains a malformed or unpadded {name}")
    if scope == "campus" and ("district_number" not in names or not _all_valid(pc.equal(
            pc.utf8_slice_codeunits(literals["campus_number"], 0, 6), literals["district_number"]))):
        raise ValueError("PEIMS campus and district identities disagree")
    if source.dataset in (PROGRAMS, PRIMARY_DISABILITY):
        aggregation = ("COUNTYREGIONDISTRICT" if source.dataset == PRIMARY_DISABILITY and scope == "district"
                       else scope.upper())
        if (not {"source_scope", "aggregation_level"} <= set(names)
                or not _all_valid(pc.equal(literals["source_scope"], "STATE"))
                or not _all_valid(pc.equal(literals["aggregation_level"], aggregation))):
            raise ValueError("PEIMS report differs from its selected scope")
    if (not all(_all_valid(pc.not_equal(literals[name], "")) for name in key_columns)
            or literals.select(key_columns).group_by(key_columns).aggregate([]).num_rows != literals.num_rows):
        raise ValueError("PEIMS contains a blank or duplicate table key")
    counts = {metric.field for metric in source.metadata.metrics.values() if metric.unit == "count"}
    columns: dict[str, pa.Array | pa.ChunkedArray] = {}
    fields = {}
    for field in table["fields"]:
        name = field["field"]
        if field["disposition"] == "identity":
            columns[name] = literals[name]
            continue
        if field["disposition"] != "fact" or name not in counts:
            raise ValueError(f"PEIMS field {name!r} lacks a reviewed count definition")
        raw, status = f"peims_raw_{name}", f"peims_status_{name}"
        columns[name], columns[status] = _count_columns(literals[name], policy)
        columns[raw] = literals[name]
        fields[name] = {"raw": raw, "status": status}
    columns["peims_source_record"] = pa.array(records, type=pa.int64())
    return pa.table(columns), key_columns, {"fields": fields, "context": ["peims_source_record"]}


def materialize_peims_students(store: ProvenanceStore, *, config: Path, config_sha256: str,
                             selections: Sequence[tuple[str, int]], output: Path) -> dict[str, Any]:
    """Build one opt-in leaf per exact publisher scope without activating a release."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .analytical import write_leaf_artifact
    from .release import _install_candidate, publish_release
    from .teadata_config import load_config_bytes

    if (not selections or len(set(selections)) != len(selections)
            or any(dataset not in DATASETS or type(year) is not int or not 2 <= year <= 9999 for dataset, year in selections)):
        raise ValueError("Select distinct exact PEIMS student datasets and years")
    if output.exists() or output.is_symlink():
        raise FileExistsError("PEIMS candidate requires a new exclusive output directory")
    with config.open("rb") as stream:
        config_bytes = stream.read(2 * 1024**2 + 1)
    if not 0 < len(config_bytes) <= 2 * 1024**2 or sha256(config_bytes).hexdigest() != config_sha256:
        raise ValueError("PEIMS Config differs from its pin or exceeds 2 MiB")
    cfg = load_config_bytes(config_bytes, path=config)
    sources = [_admit_source(cfg, dataset, year, store) for dataset, year in selections]
    toolchain = {"python": sys.version, "pyarrow": version("pyarrow"),
                 "platform": platform.platform(), "architecture": platform.machine()}
    lock = Path(__file__).parents[1] / "uv.lock"
    if lock.is_file():
        toolchain["lock_sha256"] = file_identity(lock)["sha256"]
    cli_sha = file_identity(Path(__file__).parent / "scripts/refresh_peims.py")["sha256"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".peims-student-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir(mode=0o700)
        target = ProvenanceStore(stage / "provenance")
        recorder = BuildRecorder(cfg, target)
        captured = Path(temporary) / "source-config.input"
        captured.write_bytes(config_bytes)
        config_id = target.record_acquisition(captured, ResolvedDataset("reference:peims_leaf_config", "reference", None,
            None, "", DatasetMetadata(title="Exact reviewed PEIMS student source Config")))
        results: list[dict[str, Any]] = []
        for source, policy, tables, identifiers, references in sources:
            for identifier in [*identifiers.values(), *references]:
                target.copy_record_from(store, identifier)
            for table in tables:
                scope = table["table"]
                original = identifiers[scope]
                source_sha, = table_input_sha256s(table)
                projected, keys, support = _project(target.verify_artifact(source_sha), table, source, policy)
                edition = str(source.resolved_year)
                grain = "one publisher row per " + " and ".join(keys)
                key = EditionKey(source.dataset, edition, EditionKey.source_revision((source_sha,)), scope, scope, grain)
                recipe = {"operation": "reviewed_peims_student_csv_projection", "encoding": ENCODING,
                          "config_sha256": config_sha256, "cli_sha256": cli_sha, "batch_rows": BATCH_ROWS}
                contract = recorder.leaf_contract(key, source, native_schema=projected.schema.serialize().to_pybytes(),
                    columns=projected.column_names, key_columns=keys, rows=projected.num_rows, inputs=[source_sha],
                    recipe=recipe, toolchain=toolchain, column_support=support)
                leaf_id = sha256(canonical_json(contract).encode()).hexdigest()
                leaf = stage / f"{leaf_id}.parquet"
                batches = projected.to_batches(max_chunksize=BATCH_ROWS)
                physical = write_leaf_artifact(leaf, batches, contract=contract, batch_bytes=BATCH_BYTES)
                with pq.ParquetFile(leaf) as readback:
                    if readback.num_row_groups != len(batches) or any(
                            not pa.Table.from_batches([expected]).equals(
                                readback.read_row_group(index, use_threads=False).replace_schema_metadata(None), check_metadata=True)
                            for index, expected in enumerate(batches)):
                        raise ValueError("PEIMS native leaf readback differs from its source projection")
                processing = target.record_processing(dataset=source.dataset, inputs=[original, *references, config_id],
                    outputs={"leaf": leaf}, processor="teadata.scripts.refresh_peims", code_version=recorder.processor_sha256,
                    parameters={"leaf_contract_sha256": leaf_id, "recipe": recipe, "toolchain": toolchain},
                    validation={"native_projection_readback_equal": True, "rows": projected.num_rows,
                                "field_correctness_qualified": False, "release_activation": False})
                ref = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name,
                    canonical_json(contract), processing, "leaf")
                recorder.record_edition(key, source, artifacts=[ref], acquisition_ids=[original])
                results.append({"key": key.to_dict(), "artifact_id": leaf_id, "rows": projected.num_rows,
                                "acquisition_id": original, "processing_record_id": processing})
        catalog = recorder.catalog(schema_version=3)
        publish_release(stage, catalog, schema_version=3)
        result = {"status": "unpublished_native_peims_candidate", "build_id": catalog.build_id,
                  "field_correctness_qualified": False, "release_activation": False, "editions": results}
        (stage / "MATERIALIZATION.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        _install_candidate(stage, output)
        return result
