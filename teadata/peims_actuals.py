"""Build explicit PEIMS Single File district and charter financial account leaves."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import platform
import re
import sys
from tempfile import TemporaryDirectory
from typing import Any, Literal
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.compute as arrow_compute
import pyarrow.csv as csv
import pyarrow.parquet as pq

from .analytical import write_leaf_artifact
from .column_schema import crosswalk_dataset, parse_column_schema, reader_options, table_input_sha256s
from .dataset_metadata import ArtifactRef, DataPeriod, DatasetMetadata, EditionKey, ResolvedDataset
from .manifest import BuildRecorder, canonical_json, file_identity
from .provenance import ProvenanceStore, validate_reference_records
from .release import _install_candidate, publish_release
from .teadata_config import Config, load_config, load_config_bytes
from .values import ValueStatus

DATASET = "peims_financial_single_file_actuals"
BUDGET_DATASET = "peims_financial_single_file_budgets"
_RECORD = "source_record"
_WIDTHS = (6, 3, 4, 2, 4, 3, 2)
_BLOCK_BYTES = 2 * 1024**2
_MAX_ARCHIVE_BYTES = 40 * 1024**2
_MAX_MEMBER_BYTES = 320 * 1024**2
pc: Any = arrow_compute


@dataclass(frozen=True)
class _FinancialBasis:
    name: Literal["actual", "budget"]
    dataset: str
    amount_code: str

    @property
    def amount(self) -> str:
        return f"{self.name}_amount"

    @property
    def raw(self) -> str:
        return f"raw_{self.amount}"

    @property
    def status(self) -> str:
        return f"{self.amount}_status"


_ACTUALS = _FinancialBasis("actual", DATASET, "ACTAMT")
_BUDGETS = _FinancialBasis("budget", BUDGET_DATASET, "BUDGAMT")


@dataclass(frozen=True)
class _Archive:
    basis: _FinancialBasis
    source: ResolvedDataset
    table: str
    member: str
    rows: int
    receipt_id: str
    digest: str
    codes: tuple[str, ...]
    names: tuple[str, ...]
    references: tuple[str, ...]

    @property
    def schema(self) -> pa.Schema:
        return pa.schema(
            [(name, pa.int64() if name == self.basis.amount else pa.string()) for name in self.names]
            + [(self.basis.raw, pa.string()), (self.basis.status, pa.string()), (_RECORD, pa.int64())]
        )

    @property
    def key_columns(self) -> tuple[str, ...]:
        return self.names[:len(_WIDTHS)]


def _tables(config: Config, year: int, basis: _FinancialBasis) -> dict[str, dict[str, Any]]:
    resource = config.column_contract(basis.dataset, year, reader_profile="source")
    if resource is None:
        raise ValueError("PEIMS financial accounts require the reviewed positional crosswalk")
    document = parse_column_schema(resource["text"].encode())
    return {table["table"]: table for table in crosswalk_dataset(document, basis.dataset, str(year))["tables"]}


def _admit(config: Config, canonical: Config, store: ProvenanceStore, year: int,
           basis: _FinancialBasis) -> Iterator[_Archive]:
    source = config.resolve_dataset(basis.dataset, year, reader_profile="source", exact=True)
    if (source.metadata.period != DataPeriod("fiscal_year", year - 1, year)
            or source.metadata.version != str(year)
            or not config.is_source_only(basis.dataset, year)):
        raise ValueError(f"PEIMS financial {basis.name}s require an exact source-only fiscal edition")
    qualifiers = source.metadata.source_qualifiers
    if qualifiers is None or qualifiers.family != "peims_financial" or qualifiers.financial_basis != basis.name:
        raise ValueError(f"PEIMS financial {basis.name}s require explicit {basis.name} basis")
    metrics = source.metadata.metrics
    amount = basis.amount
    if (source.metadata.value_policy is not None or set(metrics) != {amount}
            or metrics[amount].field != amount or metrics[amount].unit != "currency"
            or metrics[amount].period not in (None, source.metadata.period)):
        raise ValueError("PEIMS financial accounts require observed signed currency and the exact fiscal edition")
    tables = _tables(config, year, basis)
    reviewed = _tables(canonical, year, basis)
    archives = reader_options(source.reader, "source").get("archives")
    if not isinstance(archives, dict) or archives.keys() != tables.keys() or tables.keys() != reviewed.keys():
        raise ValueError("PEIMS financial accounts require separate district and charter sources")
    references = tuple(dict.fromkeys(
        identifier for ref in source.metadata.references for identifier in ref.provenance_ids
    ))
    records = store.verify_record_closure(references)
    validate_reference_records(source, records.__getitem__, verify_all_claimed=True)
    for population, table in tables.items():
        if table["fields"] != reviewed[population]["fields"]:
            raise ValueError(f"PEIMS financial {basis.name}s {population} columns differ from the packaged source crosswalk")
        selection = archives[population]
        if (not isinstance(selection, dict)
                or set(selection) != {"receipt_id", "member", "rows", "requested_url", "resolved_url"}
                or type(selection["rows"]) is not int or selection["rows"] < 1
                or not isinstance(selection["member"], str)
                or re.fullmatch(r"[a-z0-9]+\.csv", selection["member"]) is None):
            raise ValueError(f"Invalid {population} archive contract")
        identifier = selection["receipt_id"]
        record = store.read_record(identifier)
        digest, = table_input_sha256s(table)
        artifact = record["artifact"]
        if (record["kind"] != "acquisition" or record["method"] != "download"
                or record["dataset"] != basis.dataset or record["section"] != source.section
                or (record["requested_year"], record["resolved_year"]) != (year, year)
                or record["source_metadata"]["version"] != str(year)
                or record["source_metadata"]["period"] != source.metadata.period.to_dict()
                or record["source_metadata"]["source_url"] != source.metadata.source_url
                or record["requested_url"] != selection["requested_url"]
                or record["resolved_url"] != selection["resolved_url"]
                or record["parameters"].get("http_method") != "GET"
                or record["parameters"].get("http_status") != 200
                or record["parameters"].get("population") != population
                or artifact["sha256"] != digest or not 0 < artifact["size_bytes"] <= _MAX_ARCHIVE_BYTES):
            raise ValueError(f"PEIMS financial {basis.name}s {population} acquisition differs from its reviewed contract")
        fields = table["fields"]
        codes = tuple(field["code"] for field in fields)
        names = tuple(field["field"] for field in fields)
        if (table.get("header_identity") != "position" or table.get("metric_period") != source.metadata.period.to_dict()
                or len(names) != 9 or names[0] != "district_number" or names[7:] != (amount, "source_update_date")
                or codes[0] != "DISTRICT" or codes[7:] != (basis.amount_code, "DTUPDATE")
                or set(names) & {basis.raw, basis.status, _RECORD}
                or [field["field"] for field in fields if field["disposition"] == "fact"] != [amount]):
            raise ValueError(f"PEIMS financial {basis.name}s {population} crosswalk is not the reviewed account layout")
        yield _Archive(basis, source, population, selection["member"], selection["rows"], identifier, digest,
                       codes, names, references)


def _all(values: pa.Array) -> bool:
    return pc.all(pc.fill_null(values, False)).as_py() is True


def _batches(path: Path, archive: _Archive) -> Iterator[pa.RecordBatch]:
    schema = archive.schema
    offset, previous = 2, None
    with ZipFile(path) as package:
        members = package.infolist()
        if (len(members) != 1 or members[0].filename != archive.member
                or not 0 < members[0].file_size <= _MAX_MEMBER_BYTES):
            raise ValueError("PEIMS financial archive differs from the single reviewed CSV member")
        with package.open(members[0]) as member:
            header = ",".join(archive.codes).encode("ascii")
            if member.readline(len(header) + 3) not in {header + b"\r\n", header + b"\n"}:
                raise ValueError("PEIMS financial CSV header changed")
            reader = csv.open_csv(member,
                read_options=csv.ReadOptions(column_names=list(archive.names), use_threads=False, block_size=_BLOCK_BYTES),
                parse_options=csv.ParseOptions(ignore_empty_lines=False, newlines_in_values=False, quote_char=False),
                convert_options=csv.ConvertOptions(column_types=dict.fromkeys(archive.names, pa.string()),
                    strings_can_be_null=False, null_values=[]))
            for batch in reader:
                columns = batch.columns
                if any(not _all(pc.match_substring_regex(column, rf"^[0-9]{{{width}}}$"))
                       for width, column in zip(_WIDTHS, columns[:len(_WIDTHS)], strict=True)):
                    raise ValueError("PEIMS financial CSV has a malformed account code")
                if not _all(pc.not_equal(columns[0], "000000")):
                    raise ValueError("PEIMS financial CSV has a zero district identifier")
                raw = columns[7]
                if not _all(pc.match_substring_regex(raw, r"^-?[0-9]+$")):
                    raise ValueError("PEIMS financial CSV has an unknown amount token")
                if not _all(pc.match_substring_regex(columns[8], r"^[0-9]{2}[A-Z]{3}[0-9]{4}$")):
                    raise ValueError("PEIMS financial CSV has a malformed publisher update date")
                # These reviewed files are ordered by their full account key. Checking strict
                # ordering proves uniqueness without a statewide hash table or a sort.
                keys = pc.binary_join_element_wise(*columns[:len(_WIDTHS)], "")
                if ((previous is not None and previous >= keys[0].as_py()) or
                        (len(keys) > 1 and not _all(pc.less(keys.slice(0, len(keys) - 1), keys.slice(1))))):
                    raise ValueError("PEIMS financial CSV account keys are repeated or out of order")
                previous = keys[-1].as_py()
                stop = offset + batch.num_rows
                columns[7] = pc.cast(raw, pa.int64(), safe=True)
                yield pa.RecordBatch.from_arrays([
                    *columns, raw, pa.repeat(ValueStatus.OBSERVED.value, batch.num_rows),
                    pa.array(range(offset, stop), type=pa.int64()),
                ], schema=schema)
                offset = stop
    if offset - 2 != archive.rows:
        raise ValueError("PEIMS financial CSV differs from the reviewed record count")


def _verify_leaf(leaf: Path, source: Path, archive: _Archive) -> None:
    with pq.ParquetFile(leaf) as readback:
        for index, expected in enumerate(_batches(source, archive)):
            actual = readback.read_row_group(index, use_threads=False).replace_schema_metadata(None)
            if not pa.Table.from_batches([expected]).equals(actual, check_metadata=True):
                raise ValueError("PEIMS financial native readback differs from the exact publisher projection")
        if readback.metadata.num_rows != archive.rows:
            raise ValueError("PEIMS financial native readback has a different record count")


def materialize_peims_actuals(store: ProvenanceStore, *, config: Path, config_sha256: str,
                             years: Sequence[int], output: Path) -> dict[str, Any]:
    """Project pinned actuals archives to an unpublished analytical bundle."""
    return _materialize(store, config=config, config_sha256=config_sha256, years=years, output=output, basis=_ACTUALS)


def materialize_peims_budgets(store: ProvenanceStore, *, config: Path, config_sha256: str,
                             years: Sequence[int], output: Path) -> dict[str, Any]:
    """Project pinned budget archives to an unpublished analytical bundle."""
    return _materialize(store, config=config, config_sha256=config_sha256, years=years, output=output, basis=_BUDGETS)


def _materialize(store: ProvenanceStore, *, config: Path, config_sha256: str,
                 years: Sequence[int], output: Path, basis: _FinancialBasis) -> dict[str, Any]:
    if not years or len(set(years)) != len(years) or any(type(year) is not int or not 2 <= year <= 9999 for year in years):
        raise ValueError(f"Select distinct exact PEIMS financial {basis.name}s editions")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"PEIMS financial {basis.name}s require a new candidate directory")
    with config.open("rb") as stream:
        config_bytes = stream.read(2 * 1024**2 + 1)
    if not 0 < len(config_bytes) <= 2 * 1024**2 or sha256(config_bytes).hexdigest() != config_sha256:
        raise ValueError(f"PEIMS financial {basis.name}s Config differs from its pin or exceeds 2 MiB")
    cfg = load_config_bytes(config_bytes, path=config)
    canonical = load_config()
    selections = [archive for year in sorted(years) for archive in _admit(cfg, canonical, store, year, basis)]
    toolchain = {"python": sys.version, "pyarrow": version("pyarrow"),
                 "platform": platform.platform(), "architecture": platform.machine()}
    lock = Path(__file__).parents[1] / "uv.lock"
    if lock.is_file():
        toolchain["lock_sha256"] = file_identity(lock)["sha256"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".peims-{basis.name}s-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir(mode=0o700)
        target = ProvenanceStore(stage / "provenance")
        recorder = BuildRecorder(cfg, target)
        captured = Path(temporary) / "source-config.input"
        captured.write_bytes(config_bytes)
        config_id = target.record_acquisition(captured, ResolvedDataset(
            f"reference:peims_financial_{basis.name}s_config", "reference", None, None, "",
            DatasetMetadata(title=f"Exact reviewed PEIMS Single File Financial {basis.name.title()}s Config")))
        roots = dict.fromkeys(identifier for archive in selections for identifier in (archive.receipt_id, *archive.references))
        for identifier in roots:
            target.copy_record_from(store, identifier)
        results = []
        for archive in selections:
            source = archive.source
            raw = target.verify_artifact(archive.digest)
            key = EditionKey(basis.dataset, str(source.resolved_year), EditionKey.source_revision((archive.digest,)),
                             "district", archive.table, "one publisher row per complete account code tuple")
            recipe = {"operation": f"peims_single_file_{basis.name}s_projection", "archive_member": archive.member,
                      "config_sha256": config_sha256, "input_block_bytes": _BLOCK_BYTES}
            schema = archive.schema
            contract = recorder.leaf_contract(key, source, native_schema=schema.serialize().to_pybytes(),
                columns=schema.names, key_columns=archive.key_columns, rows=archive.rows, inputs=[archive.digest],
                recipe=recipe, toolchain=toolchain,
                column_support={"fields": {basis.amount: {"raw": basis.raw, "status": basis.status}}, "context": [_RECORD]})
            identity = sha256(canonical_json(contract).encode()).hexdigest()
            leaf = stage / f"{identity}.parquet"
            physical = write_leaf_artifact(leaf, _batches(raw, archive), contract=contract, batch_bytes=16 * 1024**2)
            _verify_leaf(leaf, raw, archive)
            processing = target.record_processing(dataset=basis.dataset, inputs=[archive.receipt_id, *archive.references, config_id],
                outputs={"leaf": leaf}, processor="teadata.peims_actuals", code_version=recorder.processor_sha256,
                parameters={"leaf_contract_sha256": identity, "recipe": recipe, "toolchain": toolchain},
                validation={"native_projection_readback_equal": True, "rows": archive.rows,
                            "cross_report_reconciliation_qualified": False, "release_activation": False})
            ref = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name,
                              canonical_json(contract), processing, "leaf")
            recorder.record_edition(key, source, artifacts=[ref], acquisition_ids=[archive.receipt_id])
            results.append({"key": key.to_dict(), "artifact_id": identity, "rows": archive.rows,
                            "acquisition_id": archive.receipt_id, "processing_record_id": processing})
        catalog = recorder.catalog(schema_version=4)
        publish_release(stage, catalog, schema_version=4)
        result = {"status": f"unpublished_native_peims_{basis.name}s_candidate", "build_id": catalog.build_id,
                  "cross_report_reconciliation_qualified": False, "release_activation": False, "editions": results}
        (stage / "MATERIALIZATION.json").write_text(canonical_json(result) + "\n", encoding="utf-8")
        _install_candidate(stage, output)
    return result
