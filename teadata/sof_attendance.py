"""Exact decimal attendance observations from the reviewed SOF run47422 workbook."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from importlib.resources import files
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import sys
from tempfile import TemporaryDirectory
from typing import Any
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq
from openpyxl import load_workbook

from .analytical import write_leaf_artifact
from .column_schema import source_schema_fingerprint
from .dataset_metadata import ArtifactRef, DataPeriod, DatasetMetadata, EditionKey, ResolvedDataset
from .financial_leaves import NS, _literal_batch, _literal_token, _package, _sheet_frame, _xml
from .manifest import BuildRecorder, ContractError, canonical_json
from .provenance import ProvenanceStore
from .release import DataRelease, _install_candidate, publish_release
from .teadata_config import load_config_bytes


DATASET = "summary_of_finances_attendance"
RESOURCE = "methodologies/sof-attendance-47422.json"
PERIOD = DataPeriod("school_year", 2025, 2026)
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_IDENTITIES = {"DISTRICT_ID": "district_number", "SOF_RUN_ID": "sof_run_id", "SCHOOLYEAR": "sof_school_year"}


def definition() -> dict[str, Any]:
    return json.loads(files("teadata").joinpath(RESOURCE).read_bytes())


def exact_decimal(raw: str | None, *, precision: int, scale: int) -> Decimal | None:
    """Interpret an original numeric lexeme without rounding or binary conversion."""
    if raw is None:
        return None
    if not isinstance(raw, str) or _NUMBER.fullmatch(raw) is None:
        raise ContractError("SOF attendance requires a literal numeric token or an absent cell")
    try:
        value = Decimal(raw)
    except InvalidOperation as error:
        raise ContractError("Invalid SOF attendance decimal") from error
    if not value.is_finite():
        raise ContractError("SOF attendance requires a finite decimal")
    if (max(0, -int(value.as_tuple().exponent)) > scale
            or max(0, value.adjusted() + 1) > precision - scale):
        raise ContractError("SOF attendance exceeds its reviewed decimal precision or scale")
    return value


def read_sof_attendance(path: Path) -> dict[str, pa.Table]:
    """Replay only reviewed attendance fields from original OOXML cells."""
    grid = definition()
    if sha256(path.read_bytes()).hexdigest() != grid["workbook_sha256"]:
        raise ContractError("SOF attendance requires the exact original run47422 workbook")
    result = {}
    identities = None
    with ZipFile(path) as archive:
        sheets, strings = _package(archive, literal=True)
        for name, selected in grid["tables"].items():
            header = _xml(archive, sheets[name], literal=True).find(NS + "sheetData/" + NS + "row")
            if header is None:
                raise ContractError("SOF attendance worksheet has no publisher header")
            codes = []
            for cell in header:
                code = _literal_token(cell, strings)
                if code is None:
                    raise ContractError("SOF attendance worksheet has an absent publisher header")
                codes.append(code)
            count, rows = _sheet_frame(archive, sheets[name], strings, header_row=1, codes=codes, literal=True)
            if count != grid["rows_per_table"]:
                raise ContractError("SOF attendance row count differs from the reviewed reporting roster")
            batch = _literal_batch(rows, strings, codes, 1, count)
            numbers = batch["DISTRICT_ID"].to_pylist()
            runs = batch["SOF_RUN_ID"].to_pylist()
            years = batch["SCHOOLYEAR"].to_pylist()
            if (any(not isinstance(number, str) or re.fullmatch(r"[0-9]{6}", number) is None for number in numbers)
                    or len(set(numbers)) != count or set(years) != {"2026"}
                    or Counter(runs) != grid["row_run_counts"]):
                raise ContractError("SOF attendance identifier, year or row-run population differs")
            observed = dict(zip(numbers, runs, strict=True))
            if identities is not None and observed != identities:
                raise ContractError("SOF attendance tables disagree on LEA identifiers or row runs")
            identities = observed
            data = {field: batch[code] for code, field in _IDENTITIES.items()}
            data["financial_raw_district_number"] = batch["DISTRICT_ID"]
            if "estimate_code" in selected:
                estimate = batch["ESTIMATE_TYPE_CD"]
                if set(estimate.to_pylist()) != {selected["estimate_code"]}:
                    raise ContractError("SOF attendance LPE/DPE estimate code differs")
                data["sof_estimate_type_cd"] = estimate
            for code in selected["fields"]:
                field = name + "_" + code.lower()
                raw = batch[code].to_pylist()
                values = [exact_decimal(value, precision=selected["precision"], scale=selected["scale"]) for value in raw]
                data[field] = pa.array(values, type=pa.decimal128(selected["precision"], selected["scale"]))
                data["financial_raw_" + field] = pa.array(raw, type=pa.string())
                data["financial_status_" + field] = pa.array(["missing" if value is None else "observed" for value in raw])
                data["financial_scale_" + field] = pa.array(
                    [None if value is None else -int(value.as_tuple().exponent) for value in values], type=pa.int16())
                data["financial_column_" + field] = pa.repeat(pa.scalar(codes.index(code) + 1, type=pa.int16()), count)
            data["financial_source_sheet"] = pa.repeat(name, count)
            data["financial_source_row"] = pa.array(range(2, count + 2), type=pa.int32())
            data["sof_estimate_basis"] = pa.repeat(name.rsplit("_", 1)[1].upper(), count)
            result[name] = pa.table(data)
    for case in grid["literal_goldens"]:
        table = result[case["table"]]
        index = table["district_number"].to_pylist().index(case["district"])
        if table["financial_raw_" + case["field"]][index].as_py() != case["raw"]:
            raise ContractError("SOF attendance differs from the independently frozen literal source-cell golden")
    return result


def _support(table: pa.Table, fields: list[str]) -> dict[str, Any]:
    return {"fields": {"district_number": {"raw": "financial_raw_district_number"}, **{
        field: {"raw": "financial_raw_" + field, "status": "financial_status_" + field,
                "auxiliary": ["financial_scale_" + field, "financial_column_" + field]} for field in fields}},
        "context": [name for name in ("financial_source_sheet", "financial_source_row", "sof_estimate_basis",
                                     "sof_estimate_type_cd") if name in table.column_names]}


def _configuration(store: ProvenanceStore, stage: Path, dictionary: Path, dictionary_origin: Path) -> tuple[bytes, str]:
    grid = definition()
    if sha256(dictionary.read_bytes()).hexdigest() != grid["dictionary_sha256"]:
        raise ContractError("SOF attendance dictionary differs from the reviewed exact-run dictionary")
    workbook = load_workbook(dictionary, read_only=True, data_only=False, keep_links=False)
    try:
        definitions = {row[0]: row[3] for row in workbook["Variable Listing"].values
                       if row[0] in grid["definitions"] and row[2] == "Num"}
        if definitions != grid["definitions"]:
            raise ContractError("SOF attendance selected definitions differ from the original dictionary")
    finally:
        workbook.close()
    origin_raw = dictionary_origin.read_bytes()
    if sha256(origin_raw).hexdigest() != grid["dictionary_origin_sha256"]:
        raise ContractError("SOF dictionary origin differs from its retained custody observation")
    origin = json.loads(origin_raw)
    if origin.get("sha256") != grid["dictionary_sha256"] or origin.get("size_bytes") != dictionary.stat().st_size:
        raise ContractError("SOF dictionary custody observation binds different bytes")
    origin_path = stage / "sof-dictionary-origin.json"
    origin_path.write_bytes(origin_raw)
    origin_id = store.record_acquisition(origin_path, ResolvedDataset("reference:sof_dictionary_origin", "reference",
        2026, 2026, "", DatasetMetadata(title="Retained SOF dictionary download-custody observation")))
    references = []
    for identifier, title in ((grid["workbook_receipt"], "Original run47422 publisher workbook"),
                              (grid["index_receipt"], "Publisher run47422 Near Final index")):
        record = store.read_record(identifier)
        references.append({"reference_id": "publisher_workbook" if identifier == grid["workbook_receipt"] else "publisher_index",
                           "role": "source_notes", "title": title, "status": "available", "url": record["requested_url"],
                           "reviewed_sha256": record["artifact"]["sha256"], "provenance_ids": [identifier]})
    dictionary_id = store.record_acquisition(dictionary, ResolvedDataset("reference:sof_attendance_dictionary", "reference",
        2026, 2026, "", DatasetMetadata(title="Retained original run47422 dictionary", version="47422",
            source_url=origin["url"], notes="Local re-observation of exact retained bytes. The original download time "
            "is unrecorded; the custody observation time is not substituted for it.")),
        parameters={"origin_observation_receipt": origin_id})
    references.append({"reference_id": "dictionary", "role": "data_dictionary", "status": "available",
                       "title": "Original run47422 data dictionary", "reviewed_sha256": grid["dictionary_sha256"],
                       "url": origin["url"],
                       "provenance_ids": [dictionary_id]})
    references.append({"reference_id": "dictionary_origin", "role": "source_notes", "status": "available",
                       "title": "Original dictionary download-custody observation; exact download time unrecorded",
                       "path": origin_path.name, "reviewed_sha256": grid["dictionary_origin_sha256"],
                       "provenance_ids": [origin_id]})
    grid_path = stage / "sof-attendance-semantic-grid.json"
    grid_path.write_bytes(files("teadata").joinpath(RESOURCE).read_bytes())
    grid_id = store.record_acquisition(grid_path, ResolvedDataset("reference:sof_attendance_grid", "reference", 2026, 2026,
        "", DatasetMetadata(title="Reviewed exact SOF attendance semantic grid")))
    references.append({"reference_id": "semantic_grid", "role": "methodology", "status": "available",
                       "title": "SOF attendance field, period, estimate and precision contract",
                       "path": grid_path.name,
                       "reviewed_sha256": sha256(grid_path.read_bytes()).hexdigest(), "provenance_ids": [grid_id]})
    mappings = []
    metrics = {}
    for name, selection in grid["tables"].items():
        fields: list[dict[str, Any]] = []
        for code, field in _IDENTITIES.items():
            fields.append({"code": code, "label": code, "field": field, "disposition": "identity",
                           "dictionary_status": "not_applicable", "publisher_type": None, "publisher_description": None})
        for code in selection["fields"]:
            field = name + "_" + code.lower()
            fields.append({"code": code, "label": code, "field": field, "disposition": "fact",
                           "dictionary_status": "documented", "publisher_type": "Num",
                           "publisher_description": grid["definitions"][code]})
            metrics[field] = {"field": field, "unit": grid["unit"], "period": PERIOD.to_dict()}
        fields = [{"position": index, **field} for index, field in enumerate(fields, 1)]
        labels = [field["code"] for field in fields]
        mappings.append({"table": name, "header_identity": "position", "input_sha256": grid["workbook_sha256"],
                         "source_schema_sha256": source_schema_fingerprint(labels, labels),
                         "metric_period": PERIOD.to_dict(), "fields": fields})
    crosswalk = {"schema_version": 2, "mapping_version": 1, "datasets": [
        {"dataset": DATASET, "edition": grid["edition"], "source_only": True, "tables": mappings}]}
    schema = stage / "sof-attendance-crosswalk.json"
    schema.write_text(canonical_json(crosswalk) + "\n")
    schema_id = store.record_acquisition(schema, ResolvedDataset("reference:sof_attendance_crosswalk", "reference", 2026,
        2026, "", DatasetMetadata(title="Exact selected SOF attendance crosswalk")))
    references.append({"reference_id": "schema", "role": "schema", "status": "available", "path": schema.name,
                       "reviewed_sha256": sha256(schema.read_bytes()).hexdigest(), "provenance_ids": [schema_id]})
    metadata = {"title": "2025–26 SOF attendance equivalents", "version": grid["edition"],
                "publisher": "Texas Education Agency", "period": PERIOD.to_dict(), "identifiers": ["district_number"],
                "metrics": metrics, "references": references,
                "notes": grid["unit_definition"] + " General adjusted refined ADA is entirely missing. "
                         "SSRA-specific adjusted ADA retains its distinct publisher definition. "
                         "The projection excludes every other finance field and worksheet.",
                "source_qualifiers": {"family": "sof", "publication_run_id": grid["edition"],
                    "publisher_release_status": grid["publisher_release_status"], "reference_id": "publisher_index",
                    "row_run_column": "sof_run_id"}}
    raw_source = store.verify_artifact(grid["workbook_sha256"])
    bound = store.record_acquisition(raw_source, ResolvedDataset(DATASET, "data_sources", 2026, 2026, "",
        DatasetMetadata.from_dict(metadata)), parameters={"original_publisher_receipt": grid["workbook_receipt"]})
    metadata["provenance_ids"] = [bound]
    config = {"data_sources": {DATASET: {2026: {"path": str(raw_source), "metadata": metadata,
              "reader": {"profiles": {"source": {"column_schema": schema.name}}}}}}}
    return (canonical_json(config) + "\n").encode(), bound


def materialize_sof_attendance(store: ProvenanceStore, *, dictionary: Path, dictionary_origin: Path, output: Path) -> dict[str, Any]:
    """Prepare a narrow exact-decimal child for reviewed normal-parent admission."""
    grid = definition()
    tables = read_sof_attendance(store.verify_artifact(grid["workbook_sha256"]))
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".sof-attendance-", dir=output.parent) as directory:
        stage = Path(directory) / "bundle"
        stage.mkdir()
        config_bytes, acquisition = _configuration(store, stage, dictionary, dictionary_origin)
        config_path = stage / "source-config.input"
        config_path.write_bytes(config_bytes)
        config = load_config_bytes(config_bytes, path=config_path)
        source = config.resolve_dataset(DATASET, 2026, exact=True, reader_profile="source")
        target = ProvenanceStore(stage / "provenance")
        roots = {acquisition, *(identifier for reference in source.metadata.references for identifier in reference.provenance_ids)}
        for identifier in sorted(roots):
            target.copy_record_from(store, identifier)
        config_id = target.record_acquisition(config_path, ResolvedDataset("reference:sof_attendance_config", "reference",
            2026, 2026, "", DatasetMetadata(title="Exact SOF attendance source configuration")))
        recorder = BuildRecorder(config, target)
        toolchain = {"python": sys.version, "pyarrow": version("pyarrow"), "openpyxl": version("openpyxl"),
                     "platform": platform.platform(), "architecture": platform.machine()}
        editions = []
        for name, table in tables.items():
            key = EditionKey(DATASET, grid["edition"], EditionKey.source_revision((grid["workbook_sha256"],)),
                             "district", name, f"one publisher LEA row in worksheet {name}")
            fields = [name + "_" + code.lower() for code in grid["tables"][name]["fields"]]
            recipe = {"operation": "exact_sof_attendance", "worksheet": name, "raw_representation": "original_ooxml_numeric_token",
                      "decimal_precision": grid["tables"][name]["precision"], "decimal_scale": grid["tables"][name]["scale"]}
            contract = recorder.leaf_contract(key, source, native_schema=table.schema.serialize().to_pybytes(),
                columns=table.column_names, key_columns=["district_number"], rows=table.num_rows,
                inputs=[grid["workbook_sha256"]], recipe=recipe, toolchain=toolchain, column_support=_support(table, fields))
            contract_bytes = canonical_json(contract)
            leaf = stage / (sha256(contract_bytes.encode()).hexdigest() + ".parquet")
            physical = write_leaf_artifact(leaf, table.to_batches(max_chunksize=512), contract=contract, batch_bytes=8 * 1024**2)
            if not pq.read_table(leaf).replace_schema_metadata(None).equals(table):
                raise ContractError("SOF exact-decimal Parquet readback differs")
            processing = target.record_processing(dataset=DATASET, inputs=[*sorted(roots), config_id], outputs={"leaf": leaf},
                processor="teadata.sof_attendance", code_version=recorder.processor_sha256,
                parameters={"leaf_contract_sha256": leaf.stem, "recipe": recipe, "toolchain": toolchain},
                validation={"exact_decimal_readback": True, "release_activation": False})
            ref = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name, contract_bytes, processing, "leaf")
            recorder.record_edition(key, source, artifacts=[ref], acquisition_ids=[acquisition])
            editions.append(key.to_dict())
        manifest = publish_release(stage, recorder.catalog(schema_version=4), schema_version=4)
        release = DataRelease(manifest, offline=True)
        release.verify()
        result = {"status": "PREPARED_SOF_ATTENDANCE", "manifest_sha256": release.manifest_sha256,
                  "build_id": release.build_id, "editions": editions}
        (stage / "MATERIALIZATION.json").write_text(canonical_json(result) + "\n")
        _install_candidate(stage, output)
    return result


def retained_sof_evidence(release: DataRelease, store: ProvenanceStore) -> list[str]:
    return sorted(store.verify_record_closure([
        release.catalog.artifact_ref(identifier).processing_id for identifier in release.catalog.to_dict()["artifacts"]]))


def qualify_sof_attendance(release: DataRelease, store: ProvenanceStore) -> dict[str, Any]:
    """Replay the complete selected values, identity, precision and retained closure."""
    grid = definition()
    release.verify()
    expected = read_sof_attendance(store.verify_artifact(grid["workbook_sha256"]))
    refs = [release.catalog.artifact_ref(identifier) for identifier in release.catalog.to_dict()["artifacts"]]
    if len(refs) != 4 or {ref.edition.table for ref in refs} != set(expected) or not release.offline:
        raise ContractError("SOF attendance requires exactly its four reviewed offline tables")
    for ref in refs:
        key = ref.edition
        metadata = release.catalog.metadata(DATASET, edition=key)
        records = store.verify_record_closure([ref.processing_id])
        record = records[ref.processing_id]
        fields = [key.table + "_" + code.lower() for code in grid["tables"][key.table]["fields"]]
        references = {reference.reference_id: reference for reference in metadata.references}
        qualifiers = metadata.source_qualifiers
        embedded = json.loads(release.catalog.edition_metadata_bytes(key.identity))["provenance"]
        if (key.dataset != DATASET or key.edition != "47422" or key.scope != "district" or metadata.period != PERIOD
                or key.revision_sha256 != EditionKey.source_revision((grid["workbook_sha256"],))
                or any(metadata.metrics[field].unit != grid["unit"] for field in fields)
                or qualifiers is None or qualifiers.publication_run_id != "47422"
                or qualifiers.publisher_release_status != "Near Final"
                or record["processor"] != "teadata.sof_attendance"
                or record["code_version"] != ref.contract["producer"]["code_sha256"]
                or record["parameters"]["toolchain"] != ref.contract["producer"]["toolchain"]
                or ref.contract["inputs"] != [grid["workbook_sha256"]]
                or references["dictionary"].reviewed_sha256 != grid["dictionary_sha256"]
                or references["semantic_grid"].reviewed_sha256 != sha256(files("teadata").joinpath(RESOURCE).read_bytes()).hexdigest()
                or records != embedded
                or ref.contract["column_support"] != _support(expected[key.table], fields)):
            raise ContractError("SOF attendance field, period, run, precision or lineage binding differs")
        physical = release.analytical_dataset(ref.identity).to_table().replace_schema_metadata(None)
        public = release.table(key).to_arrow().select(expected[key.table].column_names).replace_schema_metadata(None)
        if not physical.equals(expected[key.table]) or not public.equals(expected[key.table]):
            raise ContractError("SOF attendance public/native rows differ from literal publisher replay")
    return {"status": "PASS_REVIEWED_NATIVE", "manifest_sha256": release.manifest_sha256,
            "build_id": release.build_id, "tables": 4, "rows": 4 * grid["rows_per_table"]}
