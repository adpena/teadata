"""Reviewed SOF DPE quantities; publisher formulas are not recalculated.

The frozen field ledger owns units, periods, selection and original coordinates.
Shared financial framing and native catalogs own storage and researcher access.
"""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Iterator
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from html.parser import HTMLParser
from importlib.resources import files
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
import platform
import re
import sys
from tempfile import TemporaryDirectory
from typing import Any
import warnings
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.compute as arrow_compute
import pyarrow.parquet as pq
from openpyxl import load_workbook

from .analytical import write_leaf_artifact
from .column_schema import source_schema_fingerprint
from .dataset_metadata import ArtifactRef, DataPeriod, DatasetMetadata, EditionKey, ResolvedDataset
from .financial_leaves import MAX_SOURCE_BYTES, _NUMERIC_TOKEN, _literal_batch, _package, _sheet_frame
from .identifiers import canonical_source_identifiers
from .manifest import BuildRecorder, ContractError, canonical_json
from .provenance import ProvenanceStore
from .release import DataRelease, _install_candidate, publish_release
from .teadata_config import load_config_bytes

DATASET = "summary_of_finances_dpe"
RESOURCE = "methodologies/sof-dpe-47422-v1.json"
CONTRACT_SHA256 = "6dc13f9389ae7742ae2767312143aa7ebeeb988f07a650b913a084faae35b8b7"
pc: Any = arrow_compute


def definition() -> dict[str, Any]:
    """Detached, immutable-version source interpretation; never a latest fallback."""
    raw = files("teadata").joinpath(RESOURCE).read_bytes()
    if sha256(raw).hexdigest() != CONTRACT_SHA256:
        raise ContractError("SOF DPE interpretation differs from its reviewed identity")
    return json.loads(raw)


def _selected(table: dict[str, Any]) -> list[dict[str, Any]]:
    return [field for field in table["facts"] if field["admission"] == "selected"]


def _numbers(tokens: list[str | None], field: dict[str, Any]) -> pa.Array:
    """Decode exact source decimals, refusing fractional headcounts and overflow."""
    values = []
    precision, scale = field["precision"], field["scale"]
    if type(precision) is not int or type(scale) is not int or not 1 <= precision <= 76 or not 0 <= scale <= precision:
        raise ContractError("SOF DPE requires explicit supported decimal precision and scale")
    for token in tokens:
        if token is None:
            values.append(None)
            continue
        if not isinstance(token, str) or _NUMERIC_TOKEN.fullmatch(token) is None:
            raise ContractError(f"SOF DPE {field['field']} has an unreviewed numeric token {token!r}")
        try:
            value = Decimal(token)
        except InvalidOperation as error:
            raise ContractError("SOF DPE requires an exact finite decimal") from error
        if (not value.is_finite() or max(0, -int(value.as_tuple().exponent)) > scale
                or max(0, value.adjusted() + 1) > precision - scale):
            raise ContractError(f"SOF DPE {field['field']} exceeds its frozen decimal precision")
        if field["unit"] == "count":
            if value < 0 or value != value.to_integral_value() or value > 2**63 - 1:
                raise ContractError(f"SOF DPE {field['field']} is not a nonnegative integral headcount")
            values.append(int(value))
        else:
            values.append(value)
    native_type = (pa.int64() if field["unit"] == "count" else
                   pa.decimal128(precision, scale) if precision <= 38 else pa.decimal256(precision, scale))
    result = pa.array(values, type=native_type)
    if any(actual is not None and (raw is None or Decimal(raw) != actual)
           for raw, actual in zip(tokens, result.to_pylist(), strict=True)):
        raise ContractError("SOF DPE native representation changed an original numeric value")
    return result


def iter_sof_dpe_tables(path: Path) -> Iterator[tuple[str, pa.Table]]:
    """Open one pinned original workbook and project one physical table at a time."""
    grid = definition()
    with path.open("rb") as source:
        raw = source.read(MAX_SOURCE_BYTES + 1)
    if not 0 < len(raw) <= MAX_SOURCE_BYTES or sha256(raw).hexdigest() != grid["workbook_sha256"]:
        raise ContractError("SOF DPE requires the exact retained run47422 publisher workbook")
    identities: dict[str, str] | None = None
    with ZipFile(BytesIO(raw)) as archive:
        sheets, strings = _package(archive, literal=True)
        if set(sheets) != {name[:-4] + suffix for name in grid["tables"] for suffix in ("_dpe", "_lpe")}:
            raise ContractError("SOF DPE physical worksheet roster differs")
        for name, spec in grid["tables"].items():
            count, rows = _sheet_frame(archive, sheets[name], strings, header_row=1,
                                       codes=spec["headers"], literal=True)
            batch = _literal_batch(rows, strings, spec["headers"], 1, count)
            del rows
            if count != spec["rows"]:
                raise ContractError("SOF DPE publisher row count differs from its source roster")
            data = {item["field"]: batch.column(item["position"] - 1) for item in spec["identities"]}
            data["financial_raw_district_number"] = data["district_number"]
            # The shared source normalizer preserves the historical Excel
            # quote. Native entity keys use the same six digits without it.
            data["district_number"] = pc.utf8_slice_codeunits(
                canonical_source_identifiers(data["district_number"], width=6), start=1)
            numbers, runs = data["district_number"].to_pylist(), data["sof_run_id"].to_pylist()
            if (any(number is None or number == "000000" for number in numbers) or len(set(numbers)) != count
                    or set(data["publisher_schoolyear"].to_pylist()) != {"2026"}
                    or Counter(runs) != spec["row_runs"]):
                raise ContractError("SOF DPE identifier, year or row-run distribution differs")
            current = dict(zip(numbers, runs, strict=True))
            if name == "inputs_dpe":
                identities = current
            elif identities is not None and any(identities.get(number) != run for number, run in current.items()):
                raise ContractError("SOF DPE worksheets disagree on district or row-run identity")
            if "publisher_estimate_type_cd" in data and set(data["publisher_estimate_type_cd"].to_pylist()) != {"1"}:
                raise ContractError("SOF DPE estimate code differs from the reviewed publisher code")
            if "publisher_payment_cycle_cd" in data and set(data["publisher_payment_cycle_cd"].to_pylist()) != {"2"}:
                raise ContractError("SOF DPE payment-cycle code differs; no named meaning is inferred")
            for field in _selected(spec):
                key = field["field"]
                source_values = batch.column(field["position"] - 1).to_pylist()
                data[key] = (pa.array([None if token in (None, "") else token for token in source_values], type=pa.string())
                             if field["unit"] == "text" else _numbers(source_values, field))
                data["financial_raw_" + key] = pa.array(source_values, type=pa.string())
                if field["unit"] != "text":
                    data["financial_status_" + key] = pa.array(
                        ["missing" if token is None else "observed" for token in source_values], type=pa.string())
                data["financial_column_" + key] = pa.repeat(pa.scalar(field["position"], type=pa.int16()), count)
            data["financial_source_sheet"] = pa.repeat(name, count)
            data["financial_source_row"] = pa.array(range(2, count + 2), type=pa.int32())
            data["sof_estimate_basis"] = pa.repeat("DPE", count)
            table = pa.table(data)
            for golden in spec["literal_goldens"]:
                index = numbers.index(golden["district_number"])
                if index + 2 != golden["source_row"] or any(
                    table["financial_raw_" + key][index].as_py() != token for key, token in golden["cells"].items()
                ):
                    raise ContractError("SOF DPE differs from frozen original-cell goldens")
            yield name, table


def _support(spec: dict[str, Any]) -> dict[str, Any]:
    return {"fields": {"district_number": {"raw": "financial_raw_district_number"}, **{
        field["field"]: {"raw": "financial_raw_" + field["field"],
                         **({"status": "financial_status_" + field["field"]} if field["unit"] != "text" else {}),
                         "auxiliary": ["financial_column_" + field["field"]]} for field in _selected(spec)}},
        "context": ["financial_source_sheet", "financial_source_row", "sof_estimate_basis"]}


class _ReportValues(HTMLParser):
    """Collect only explicitly requested Crystal divs, preserving ambiguity."""

    def __init__(self, wanted: set[str]):
        super().__init__()
        self.wanted = wanted
        self.stack: list[tuple[str, list[str]]] = []
        self.values: dict[str, set[str]] = {}

    def handle_starttag(self, tag, attrs):
        if tag == "div":
            if len(self.stack) >= 64:
                raise ContractError("SOF report nesting exceeds the retained witness bound")
            self.stack.append((dict(attrs).get("id", ""), []))

    def handle_data(self, data):
        for name, parts in self.stack:
            if name in self.wanted:
                parts.append(data)

    def handle_endtag(self, tag):
        if tag == "div" and self.stack:
            name, parts = self.stack.pop()
            if name in self.wanted:
                self.values.setdefault(name, set()).add(" ".join(" ".join(parts).split()))


def _report_values(raw: bytes, wanted: set[str]) -> dict[str, set[str]]:
    if not 0 < len(raw) <= 2 * 1024**2:
        raise ContractError("SOF report witness exceeds its bounded source size")
    text = raw.decode("utf-8-sig")
    contents = re.findall(r"'content':'((?:\\.|[^'\\])*)'", text)
    if not contents or len(contents) > 8:
        raise ContractError("SOF report omits its retained Crystal page content")
    parser = _ReportValues(wanted)
    for content in contents:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            decoded = ast.literal_eval("'" + content + "'").replace(r"\/", "/")
        parser.feed(decoded)
    return parser.values


def _verify_report_goldens(store: ProvenanceStore, grid: dict[str, Any],
                           source_values: dict[tuple[str, str, str], str | None]) -> int:
    count = 0
    for case in grid["report_goldens"]:
        record = store.read_record(case["receipt"])
        if (record["artifact"]["sha256"] != case["sha256"]
                or record["parameters"].get("cdn") != case["district"]
                or record["parameters"].get("run") != case["row_run"]):
            raise ContractError("SOF independent report district, row run or source bytes differ")
        wanted = {*case["headers"], *(check["item"] for check in case["checks"])}
        values = _report_values(store.verify_artifact(case["sha256"]).read_bytes(), wanted)
        if any(values.get(name) != {text} for name, text in case["headers"].items()):
            raise ContractError("SOF independent report edition, population or headers differ")
        for check in case["checks"]:
            if (values.get(check["item"]) != {check["display"]}
                    or source_values[check["table"], case["district"], check["field"]] != check["raw"]):
                raise ContractError("SOF original report display or its original workbook cell differs from the frozen golden")
            count += 1
    return count


def _configuration(store: ProvenanceStore, stage: Path) -> tuple[bytes, str]:
    grid = definition()
    dictionary = store.verify_artifact(grid["dictionary_sha256"])
    with dictionary.open("rb") as handle:
        book = load_workbook(handle, read_only=True, data_only=False, keep_links=False)
        try:
            definitions = {row[0]: (index, row[2], row[3]) for index, row in enumerate(book["Variable Listing"].values, 1)
                           if index > 1 and row[0]}
        finally:
            book.close()
    for table in grid["tables"].values():
        for field in _selected(table):
            if definitions.get(field["dictionary_code"]) != (
                field["dictionary_row"], field["publisher_type"], field["publisher_description"]
            ):
                raise ContractError("SOF DPE dictionary binding differs from its reviewed definition")
    references = []
    for name, receipt in grid["source_receipts"].items():
        record = store.read_record(receipt)
        expected = grid["workbook_sha256"] if name == "sof_bulk_2026" else grid["dictionary_sha256"] if name == "sof_dictionary_2026" else None
        if expected is not None and record["artifact"]["sha256"] != expected:
            raise ContractError("SOF DPE original acquisition identifies different bytes")
        titles = {"sof_bulk_2026": "2025–26 Summary of Finances statewide download",
                  "sof_dictionary_2026": "2025–26 Summary of Finances data dictionary, run47422",
                  "state_funding_index": "TEA State Funding publication index",
                  "sof_timeline_2026": "TEA funding data sources and update timelines"}
        references.append({"reference_id": name, "role": "data_dictionary" if name == "sof_dictionary_2026" else "source_notes",
            "title": titles.get(name, "Original SOF district report at its matching row run"), "status": "available", "url": record["requested_url"],
            "reviewed_sha256": record["artifact"]["sha256"], "provenance_ids": [receipt]})
    audit = store.verify_artifact(grid["audit_sha256"])
    audit_id = store.record_acquisition(audit, ResolvedDataset("reference:sof_dpe_audit", "reference", 2026, 2026, "",
        DatasetMetadata(title="TEA Data preliminary field inventory and original report evidence", publisher="TEA Data",
            notes="Retained preliminary recommendations, superseded by the reviewed semantic grid; not a publisher dictionary.")))
    references.append({"reference_id": "field_audit", "role": "source_notes", "status": "available",
        "title": "Retained source inventory and original report evidence; interpretation governed by the reviewed methodology",
        "path": "provenance/objects/" + grid["audit_sha256"], "reviewed_sha256": grid["audit_sha256"], "provenance_ids": [audit_id]})
    grid_path = stage / "sof-dpe-methodology.json"
    grid_path.write_bytes(files("teadata").joinpath(RESOURCE).read_bytes())
    receipt = store.record_acquisition(grid_path, ResolvedDataset("reference:sof_dpe_methodology", "reference", 2026, 2026,
        "", DatasetMetadata(title="Reviewed SOF DPE field admission, periods, units and limitations")))
    references.append({"reference_id": "semantic_grid", "role": "methodology", "path": grid_path.name,
        "status": "available", "reviewed_sha256": CONTRACT_SHA256, "provenance_ids": [receipt]})
    mappings, metrics = [], {}
    for name, table in grid["tables"].items():
        fields: list[dict[str, Any]] = [{"code": item["publisher_header"], "label": item["publisher_header"], "field": item["field"],
                   "disposition": "identity", "dictionary_status": "not_applicable", "publisher_type": None,
                   "publisher_description": None} for item in table["identities"]]
        for field in _selected(table):
            fields.append({"code": field["publisher_header"], "label": field["publisher_header"], "field": field["field"],
                "disposition": "fact", "dictionary_status": "documented", "publisher_type": field["publisher_type"],
                "publisher_description": field["publisher_description"]})
            metrics[field["field"]] = {"field": field["field"], "unit": field["unit"], "period": field["period"]}
        labels: list[str] = [field["code"] for field in fields]
        mappings.append({"table": name, "header_identity": "position", "input_sha256": grid["workbook_sha256"],
            "source_schema_sha256": source_schema_fingerprint(labels, labels), "metric_period": {"kind": "unknown"},
            "fields": [{"position": index, **field} for index, field in enumerate(fields, 1)]})
    crosswalk = {"schema_version": 2, "mapping_version": 1, "datasets": [
        {"dataset": DATASET, "edition": grid["edition"], "source_only": True, "tables": mappings}]}
    schema = stage / "sof-dpe-crosswalk.json"
    schema.write_text(canonical_json(crosswalk) + "\n")
    receipt = store.record_acquisition(schema, ResolvedDataset("reference:sof_dpe_columns", "reference", 2026, 2026,
        "", DatasetMetadata(title="Selected SOF DPE native fields and original publisher headers")))
    references.append({"reference_id": "schema", "role": "schema", "path": schema.name, "status": "available",
        "reviewed_sha256": sha256(schema.read_bytes()).hexdigest(), "provenance_ids": [receipt]})
    metadata = {"title": "2025–26 SOF District Payment Estimate: reviewed quantities as published", "version": grid["edition"],
        "publisher": "Texas Education Agency", "period": {"kind": "unknown"}, "identifiers": ["district_number"],
        "metrics": metrics, "references": references,
        "notes": "Frozen Near Final funding observations, not expenditures, receipts, Enrollment or Membership. "
                 "Field periods can differ from funding year and are not overwritten by it. Codes remain text. "
                 "No formulas recalculated;104 held fields and three already canonical attendance fields are excluded. "
                 "The methodology gives the original definitions, periods, input-vintage limits and complete admission ledger.",
        "source_qualifiers": {"family": "sof", "publication_run_id": grid["edition"], "publisher_release_status": "Near Final",
                              "reference_id": "state_funding_index", "row_run_column": "sof_run_id"}}
    raw = store.verify_artifact(grid["workbook_sha256"])
    acquisition = store.record_acquisition(raw, ResolvedDataset(DATASET, "data_sources", 2026, 2026, "",
        DatasetMetadata.from_dict(metadata)), parameters={"original_publisher_receipt": grid["source_receipts"]["sof_bulk_2026"]})
    metadata["provenance_ids"] = [acquisition]
    return (canonical_json({"data_sources": {DATASET: {2026: {"path": "provenance/objects/" + grid["workbook_sha256"], "metadata": metadata,
        "reader": {"profiles": {"source": {"column_schema": schema.name}}}}}}}) + "\n").encode(), acquisition


def materialize_sof_dpe(store: ProvenanceStore, *, output: Path) -> dict[str, Any]:
    """Build an exclusive unpublished child with the shared native-source tools."""
    grid = definition()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".sof-dpe-", dir=output.parent) as directory:
        stage = Path(directory) / "bundle"
        stage.mkdir()
        target = ProvenanceStore(stage / "provenance")
        for receipt in grid["source_receipts"].values():
            target.copy_record_from(store, receipt)
        target.retain_artifact(store.verify_artifact(grid["audit_sha256"]))
        config_raw, acquisition = _configuration(target, stage)
        config_path = stage / "source-config.input"
        config_path.write_bytes(config_raw)
        config = load_config_bytes(config_raw, path=config_path)
        source = config.resolve_dataset(DATASET, 2026, exact=True, reader_profile="source")
        recorder = BuildRecorder(config, target)
        config_id = target.record_acquisition(config_path, ResolvedDataset("reference:sof_dpe_config", "reference", 2026, 2026,
            "", DatasetMetadata(title="Exact SOF DPE source configuration")))
        roots = {*grid["source_receipts"].values(), acquisition, config_id,
                 *(receipt for ref in source.metadata.references for receipt in ref.provenance_ids)}
        toolchain = {"python": sys.version, "pyarrow": version("pyarrow"), "openpyxl": version("openpyxl"),
                     "platform": platform.platform(), "architecture": platform.machine()}
        for name, table in iter_sof_dpe_tables(target.verify_artifact(grid["workbook_sha256"])):
            key = EditionKey(DATASET, grid["edition"], EditionKey.source_revision((grid["workbook_sha256"],)),
                             "district", name, f"one publisher LEA row in worksheet {name}")
            recipe = {"operation": "exact_sof_dpe", "worksheet": name, "methodology_sha256": CONTRACT_SHA256,
                      "numeric_representation": "exact original numeric lexeme; no rounding or scale conversion",
                      "formulas_recalculated": False}
            contract = recorder.leaf_contract(key, source, native_schema=table.schema.serialize().to_pybytes(),
                columns=table.column_names, key_columns=["district_number"], rows=table.num_rows,
                inputs=[grid["workbook_sha256"]], recipe=recipe, toolchain=toolchain, column_support=_support(grid["tables"][name]))
            leaf = stage / (sha256(canonical_json(contract).encode()).hexdigest() + ".parquet")
            physical = write_leaf_artifact(leaf, table.to_batches(max_chunksize=512), contract=contract, batch_bytes=8 * 1024**2)
            if not pq.read_table(leaf).replace_schema_metadata(None).equals(table):
                raise ContractError("SOF DPE Parquet roundtrip changed original numeric values")
            processing = target.record_processing(dataset=DATASET, inputs=sorted(roots), outputs={"leaf": leaf},
                processor="teadata.sof_dpe", code_version=recorder.processor_sha256,
                parameters={"leaf_contract_sha256": leaf.stem, "recipe": recipe, "toolchain": toolchain},
                validation={"exact_native_readback": True, "field_correctness_qualified": False, "release_activation": False})
            ref = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name, canonical_json(contract), processing, "leaf")
            recorder.record_edition(key, source, artifacts=[ref], acquisition_ids=[acquisition])
        manifest = publish_release(stage, recorder.catalog(schema_version=4), schema_version=4)
        child = DataRelease(manifest, offline=True)
        child.verify()
        result = {"status": "PREPARED_SOF_DPE_NOT_QUALIFIED", "manifest_sha256": child.manifest_sha256,
                  "build_id": child.build_id, "field_correctness_qualified": False, "release_activation": False}
        (stage / "MATERIALIZATION.json").write_text(canonical_json(result) + "\n")
        _install_candidate(stage, output)
    return result


def retained_sof_dpe_evidence(release: DataRelease, store: ProvenanceStore) -> list[str]:
    return sorted(store.verify_record_closure([
        release.catalog.artifact_ref(identifier).processing_id for identifier in release.catalog.to_dict()["artifacts"]]))


def qualify_sof_dpe(release: DataRelease, store: ProvenanceStore) -> dict[str, Any]:
    """Replay selected original cells and frozen semantic bindings; no formula proof."""
    grid = definition()
    artifacts = [release.catalog.artifact_ref(identifier) for identifier in release.catalog.to_dict()["artifacts"]]
    refs = {ref.edition.table: ref for ref in artifacts}
    if not release.offline or set(refs) != set(grid["tables"]) or len(artifacts) != 4 or len(refs) != 4:
        raise ContractError("SOF DPE requires exactly four original offline district tables")
    cells = 0
    source_values = {}
    for name, expected in iter_sof_dpe_tables(store.verify_artifact(grid["workbook_sha256"])):
        ref = refs[name]
        key = ref.edition
        metadata = release.catalog.metadata(DATASET, edition=key)
        qualifiers = metadata.source_qualifiers
        records = store.verify_record_closure([ref.processing_id])
        record = records[ref.processing_id]
        fields = _selected(grid["tables"][name])
        references = {reference.reference_id: reference for reference in metadata.references}
        if (key.dataset != DATASET or key.edition != grid["edition"] or key.scope != "district"
                or key.revision_sha256 != EditionKey.source_revision((grid["workbook_sha256"],))
                or qualifiers is None or qualifiers.publication_run_id != grid["edition"]
                or qualifiers.publisher_release_status != "Near Final" or qualifiers.row_run_column != "sof_run_id"
                or references["semantic_grid"].reviewed_sha256 != CONTRACT_SHA256
                or references["sof_dictionary_2026"].reviewed_sha256 != grid["dictionary_sha256"]
                or references["field_audit"].reviewed_sha256 != grid["audit_sha256"]
                or any(references[name].provenance_ids != (receipt,) for name, receipt in grid["source_receipts"].items())
                or any(metadata.metrics[field["field"]].unit != field["unit"]
                       or metadata.metrics[field["field"]].period != DataPeriod.from_dict(field["period"]) for field in fields)
                or record["processor"] != "teadata.sof_dpe"
                or record["code_version"] != ref.contract["producer"]["code_sha256"]
                or record["parameters"]["toolchain"] != ref.contract["producer"]["toolchain"]
                or record["parameters"]["recipe"]["methodology_sha256"] != CONTRACT_SHA256
                or ref.contract["inputs"] != [grid["workbook_sha256"]]
                or ref.contract["column_support"] != _support(grid["tables"][name])
                or records != json.loads(release.catalog.edition_metadata_bytes(key.identity))["provenance"]):
            raise ContractError("SOF DPE units, periods, original source, runs or processing lineage differ")
        physical = release.analytical_dataset(ref.identity).to_table().replace_schema_metadata(None)
        public = release.table(key).to_arrow().replace_schema_metadata(None)
        if not physical.equals(expected) or not public.select(expected.column_names).equals(expected):
            raise ContractError("SOF DPE native/public cells differ from exact original publisher replay")
        numbers = expected["district_number"].to_pylist()
        for case in grid["report_goldens"]:
            for check in case["checks"]:
                if check["table"] == name:
                    source_values[name, case["district"], check["field"]] = expected[
                        "financial_raw_" + check["field"]][numbers.index(case["district"])].as_py()
        cells += expected.num_rows * len(fields)
    report_cells = _verify_report_goldens(store, grid, source_values)
    return {"status": "PASS_REVIEWED_NATIVE", "manifest_sha256": release.manifest_sha256,
            "tables": 4, "selected_fields": grid["admission_counts"]["selected"], "original_cells": cells,
            "independent_report_literal_goldens": report_cells,
            "publisher_formula_qualification": False, "scope": "Original selected raw/value/status/coordinate fidelity and frozen field semantics."}
