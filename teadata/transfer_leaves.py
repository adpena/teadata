"""Explicit offline district-transfer leaves from accepted source preparations.

This creates an adjacent unpublished candidate, never the active release. The
reviewed Config owns publisher fields; transfer_sources owns normalization.
"""

from __future__ import annotations

from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Sequence

from .column_schema import crosswalk_dataset, parse_column_schema, table_input_sha256s
from .dataset_metadata import ArtifactRef, DataPeriod, DatasetMetadata, EditionKey, ResolvedDataset
from .district_transfer_identity import CORRECTED, REVIEW_YEAR, reviewed_repair_receipt
from .manifest import BuildRecorder, canonical_json, file_identity
from .provenance import ProvenanceStore, _identifier, validate_reference_records
from .transfer_sources import (
    _HEADER_FIRST_DISTRICT_YEARS,
    _captured_input,
    _historical_district_masking_receipt,
    _validate_transfer_policy,
    _write_index,
    district_batch_inputs,
    transfer_columns,
    transfer_value_columns,
)

if TYPE_CHECKING:
    from .teadata_config import Config


def _leaf_projection(columns: Sequence[str], fields: Sequence[dict], *, grain: str,
                     year: int | None = None, extra_context: Sequence[str] = ()) -> tuple[list[str], dict, list[dict]]:
    """Derive physical renames/support from the reviewed headers and one projector."""
    if [field["code"] for field in fields] != list(transfer_columns(grain, year=year)):
        raise ValueError("Transfer leaf requires every literal publisher header in its original order")
    values = transfer_value_columns(grain, year=year)
    if grain == "campus":
        values.update(REPORT_DISTRICT="transfer_reporting_district_number",
                      DISTRICT_RES_OR_ATTEND="transfer_partner_district_number")
    renames, supported, operations = {}, {}, []
    for field in fields:
        code, target = field["code"], field["field"]
        prepared_column = values.get(code, code)
        if prepared_column not in columns or field["label"] != code:
            raise ValueError("Transfer reviewed header differs from its literal source/prepared projection")
        renames[prepared_column] = target
        operations.append({"publisher_code": code, "prepared_column": prepared_column, "field": target})
        if code in values:
            entry: dict[str, Any] = {"raw": code}
            if prepared_column == "transfer_count_value":
                entry["status"] = "transfer_count_status"
            if grain == "district" and prepared_column in {"transfer_reporting_number", "transfer_partner_number"}:
                entry["auxiliary"] = ["transfer_reporting_district_number" if prepared_column == "transfer_reporting_number"
                                      else "transfer_partner_district_number"]
                if year == REVIEW_YEAR and prepared_column == "transfer_partner_number":
                    entry["auxiliary"].append(CORRECTED)
            supported[target] = entry
    names = [renames.get(name, name) for name in columns]
    if len(set(names)) != len(names):
        raise ValueError("Transfer canonical projection collides with retained raw/technical columns")
    support = {"fields": supported, "context": ["transfer_source_input", "transfer_source_record",
        "transfer_first_line", "transfer_last_line", "transfer_row_kind", *extra_context]}
    return names, support, operations


def _admit_preparation(store: ProvenanceStore, cfg: Config, year: int, identifier: str) -> dict:
    """Check the retained producer's own contract, not current global code equality."""
    graph = store.verify_record_closure([identifier])
    record = graph[identifier]
    parameters, validation = record.get("parameters", {}), record.get("validation", {})
    if (record["kind"] != "processing" or record["dataset"] != "district_transfer_reports"
            or record.get("processor") != "teadata.scripts.refresh_transfers"
            or parameters.get("processor_sha256") != record["code_version"]
            or validation.get("status") != "PASS" or validation.get("raw_string_readback_equal") is not True
            or set(record["outputs"]) != {"prepared_source", "annotations", "validation_report"}):
        raise ValueError("District input is not an accepted complete source preparation")
    _identifier(record["code_version"])
    _identifier(parameters["cli_sha256"])
    report_artifact = record["outputs"]["validation_report"]
    report = json.loads(_captured_input(store.verify_artifact(report_artifact["sha256"]), report_artifact["sha256"]))
    toolchain = parameters.get("toolchain")
    if (not isinstance(toolchain, dict)
            or any(not isinstance(toolchain.get(name), str) or not toolchain[name].strip()
                   for name in ("python", "pyarrow", "platform", "architecture"))
            or report.get("toolchain") != toolchain):
        raise ValueError("District preparation requires complete recorded toolchain strings matching its report")
    if (report.get("status") != "prepared" or report.get("grain") != "district" or report.get("year") != year
            or report.get("raw_string_readback_equal") is not True
            or report.get("implementation") != {"processor_sha256": record["code_version"], "cli_sha256": parameters["cli_sha256"]}
            or type(report.get("rows")) is not int or report["rows"] < 1 or validation.get("rows") != report["rows"]
            or report.get("source_validation", {}).get("status") != "PASS"
            or report.get("publisher_totals_computed") is not False):
        raise ValueError("District preparation report differs from its recorded edition/validation contract")
    source_validation = report["source_validation"]
    if (any(type(source_validation.get(name)) is not int or source_validation[name] != 0 for name in
            ("duplicate_detail_keys", "duplicate_total_keys", "districts_with_multiple_report_regions"))
            or source_validation.get("totals_computed_from_details") is not False):
        raise ValueError("District preparation has contradictory source-grain validation")
    indices = [graph[value] for value in record["inputs"]
               if graph[value].get("dataset") == "reference:transfer_batch_index" and graph[value]["kind"] == "acquisition"]
    if len(indices) != 1:
        raise ValueError("Statewide district leaf requires one retained regional batch-index input")
    index_artifact = indices[0]["artifact"]
    raw_index = _captured_input(store.verify_artifact(index_artifact["sha256"]), index_artifact["sha256"])
    acquisitions, regions = district_batch_inputs(store, raw_index, year, all_offered=True)
    actual = report.get("inputs", [])
    if (len(actual) != len(acquisitions) or any(item.get("status") != "parsed" or item.get("source_input") != number
            or item.get("record_id") != selected[0] or item.get("artifact") != graph[selected[0]]["artifact"]
            or type(item.get("rows")) is not int or item["rows"] < 1
            for number, (item, selected) in enumerate(zip(actual, acquisitions, strict=True)))
            or sum(item["rows"] for item in actual) != report["rows"]
            or any(value not in record["inputs"] for value, _ in acquisitions)):
        raise ValueError("District preparation does not retain the complete ordered regional inputs")
    source = cfg.resolve_dataset("district_transfer_reports", year, reader_profile="source", exact=True)
    if (source.metadata.version != str(year) or source.metadata.period != DataPeriod("school_year", year - 1, year)
            or list(source.metadata.provenance_ids) != [value for value, _ in acquisitions]):
        raise ValueError("District source contract differs from its exact edition/period/original acquisitions")
    policy_id = source.metadata.value_policy
    policy = cfg.value_policies.get(policy_id) if policy_id is not None else None
    if (policy is None or policy.to_dict() != parameters.get("value_policy")
            or report.get("policy_id") != policy.policy_id
            or report.get("policy_sha256") != sha256(json.dumps(policy.to_dict(), sort_keys=True).encode()).hexdigest()):
        raise ValueError("District leaf value policy differs from its retained preparation")
    validate_reference_records(source, store.read_record, verify_all_claimed=True)
    witness_sha = None
    if year in _HEADER_FIRST_DISTRICT_YEARS:
        _validate_transfer_policy(policy, grain="district", year=year)
        region_csv, = (identifier for identifier, region in regions.items() if region == "13")
        witness = _historical_district_masking_receipt(source, policy, region_csv, graph.__getitem__)
        if witness not in record["inputs"]:
            raise ValueError("Historical district preparation did not bind its masking reference")
        witness_sha = graph[witness]["artifact"]["sha256"]
    if year == REVIEW_YEAR:
        witness, _ = reviewed_repair_receipt(source, acquisitions, graph.__getitem__)
        if witness not in record["inputs"]:
            raise ValueError("District preparation did not bind its approved partner identity review")
        witness_sha = graph[witness]["artifact"]["sha256"]
    resource = cfg.column_contract(source.dataset, year, reader_profile="source")
    if resource is None:
        raise ValueError("District leaf requires a reviewed source-header resource")
    document = parse_column_schema(resource["text"].encode())
    branch = crosswalk_dataset(document, source.dataset, str(year))
    if len(branch["tables"]) != 1:
        raise ValueError("Statewide district transfers require one logical table per edition")
    table, = branch["tables"]
    count_code = transfer_columns("district", year=year)[-1]
    count_field, = [field["field"] for field in table["fields"] if field["code"] == count_code]
    if {value.field for value in source.metadata.metrics.values() if value.unit == "count"} != {count_field}:
        raise ValueError("District source must expose its one publisher count field with count units")
    count_header, = [row for row in cfg.header_records(source.dataset, edition=str(year), table=table["table"], reader_profile="source")
                     if row["field"] == count_field]
    if (count_header.get("disposition", "fact") != "fact"
            or DataPeriod.from_dict(count_header["period"]) != source.metadata.period):
        raise ValueError("District count field must be a fact with the exact school-year period")
    if "input_sha256s" not in table:
        raise ValueError("District source contract must enumerate every original input")
    table_input_sha256s(table, acquisition_sha256s=[digest for _, digest in acquisitions])
    artifact = record["outputs"]["prepared_source"]
    if not 0 < artifact["size_bytes"] <= 16 * 1024 * 1024 or report["rows"] * 12 > 8 * 1024 * 1024:
        raise ValueError("District prepared bytes or compact row-key index exceed admission")
    names, support, operations = _leaf_projection(report["columns"], table["fields"], grain="district", year=year)
    return {"year": year, "processing_id": identifier, "record": record, "report": report,
        "acquisitions": acquisitions, "source": source, "table": table, "artifact": artifact,
        "names": names, "support": support, "operations": operations, "witness_sha": witness_sha}


def materialize_district_transfers(store: ProvenanceStore, *, config: Path, config_sha256: str,
                                 preparations: Sequence[tuple[int, str]], output: Path) -> dict:
    """Emit one bounded native leaf per explicit annual preparation; no activation."""
    from .analytical import write_leaf_artifact
    from .release import publish_release
    from .teadata_config import load_config_bytes

    if (not preparations or any(type(year) is not int or not 2 <= year <= 9999 for year, _ in preparations)
            or len({year for year, _ in preparations}) != len(preparations)):
        raise ValueError("Select distinct explicit annual district preparations")
    raw_config = _captured_input(config, config_sha256)
    cfg = load_config_bytes(raw_config, path=config)
    admitted = [_admit_preparation(store, cfg, year, identifier) for year, identifier in preparations]
    if output.exists() or output.is_symlink():
        raise FileExistsError("District leaf candidate requires a new exclusive output directory")
    import pyarrow as pa
    import pyarrow.compute as arrow_compute
    import pyarrow.parquet as pq
    pc: Any = arrow_compute

    # Check actual native schemas/row keys before the first candidate writer.
    for item in admitted:
        artifact, report = item["artifact"], item["report"]
        publisher_columns = transfer_columns("district", year=item["year"])
        prepared = store.verify_artifact(artifact["sha256"])
        with pq.ParquetFile(prepared) as native:
            schema = native.schema_arrow.remove_metadata()
            if (schema.names != report.get("columns") or native.metadata.num_rows != report["rows"]
                    or schema.names[:len(publisher_columns)] != list(publisher_columns)
                    or any(not (pa.types.is_string(schema.field(name).type) or pa.types.is_large_string(schema.field(name).type))
                           for name in publisher_columns)
                    or schema.field("transfer_source_input").type != pa.int32()
                    or schema.field("transfer_source_record").type != pa.int64()):
                raise ValueError("Prepared district schema/rows differ from their accepted source report")
            keys = native.read(columns=["transfer_source_input", "transfer_source_record"])
            if any(column.null_count for column in keys.columns):
                raise ValueError("District source row keys cannot be null")
            grouped = keys.group_by(keys.column_names).aggregate([("transfer_source_record", "count")])
            counts = keys.group_by("transfer_source_input").aggregate([("transfer_source_record", "count")]).to_pylist()
            if (grouped.num_rows != report["rows"]
                    or {row["transfer_source_input"]: row["transfer_source_record_count"] for row in counts}
                    != {number: source["rows"] for number, source in enumerate(report["inputs"])}
                    or not pc.all(pc.greater_equal(keys["transfer_source_record"], 1)).as_py()):
                raise ValueError("District source row keys are duplicate, foreign or incomplete")
            item["schema"] = pa.schema([pa.field(name, field.type, nullable=field.nullable)
                                        for name, field in zip(item["names"], schema, strict=True)])
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".district-transfer-leaves-", dir=output.parent))
    stage.chmod(0o700)
    try:
        target_store = ProvenanceStore(stage / "provenance")
        captured = stage / "source-config.input"
        descriptor = os.open(captured, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw_config)
        config_id = target_store.record_acquisition(captured, ResolvedDataset("reference:transfer_leaf_config", "reference", None, None,
            "", DatasetMetadata(title="Exact explicitly pinned district-transfer native source contract")))
        recorder = BuildRecorder(cfg, target_store)
        toolchain = {"python": sys.version, "pyarrow": version("pyarrow"),
                     "platform": platform.platform(), "architecture": platform.machine()}
        lock = Path(__file__).parents[1] / "uv.lock"
        if lock.is_file():
            toolchain["lock_sha256"] = sha256(lock.read_bytes()).hexdigest()
        result: dict[str, Any] = {"status": "unpublished_native_transfer_candidate", "release_activation": False,
            "field_correctness_qualified": False, "grain": "district", "editions": []}
        for item in admitted:
            identifier, source, report = item["processing_id"], item["source"], item["report"]
            roots = [identifier, *(value for ref in source.metadata.references for value in ref.provenance_ids)]
            for value in roots:
                target_store.copy_record_from(store, value)
            raw_hashes = [digest for _, digest in item["acquisitions"]]
            key = EditionKey(source.dataset, str(item["year"]), EditionKey.source_revision(tuple(raw_hashes)),
                "district", item["table"]["table"], "one publisher table row per original source input and logical CSV record")
            recipe = {"operation": "accepted_transfer_preparation_projection", "prepared_processing_id": identifier,
                "publisher_projection": item["operations"], "config_sha256": config_sha256,
                "cli_sha256": sha256((Path(__file__).parent / "scripts/refresh_transfers.py").read_bytes()).hexdigest(),
                "historical_preparation_producer": item["record"]["code_version"],
                "batch_rows": 4096, "batch_bytes": 8 * 1024 * 1024, "publisher_totals_computed": False}
            contract = recorder.leaf_contract(key, source, native_schema=item["schema"].serialize().to_pybytes(),
                columns=item["names"], key_columns=["transfer_source_input", "transfer_source_record"], rows=report["rows"],
                inputs=[*raw_hashes, item["artifact"]["sha256"], *([item["witness_sha"]] if item["witness_sha"] else [])],
                recipe=recipe, toolchain=toolchain, column_support=item["support"])
            leaf_id = sha256(canonical_json(contract).encode()).hexdigest()
            leaf = stage / (leaf_id + ".parquet")
            prepared = target_store.verify_artifact(item["artifact"]["sha256"])
            with pq.ParquetFile(prepared) as native:
                batches = (pa.RecordBatch.from_arrays(batch.columns, schema=item["schema"])
                           for batch in native.iter_batches(batch_size=4096, use_threads=False))
                physical = write_leaf_artifact(leaf, batches, contract=contract, batch_bytes=8 * 1024 * 1024)
            with pq.ParquetFile(prepared) as before, pq.ParquetFile(leaf) as after:
                for old, new in zip(before.iter_batches(batch_size=4096, use_threads=False),
                                    after.iter_batches(batch_size=4096, use_threads=False), strict=True):
                    projected = pa.RecordBatch.from_arrays(old.columns, schema=item["schema"])
                    if not projected.equals(new.replace_schema_metadata(None), check_metadata=True):
                        raise ValueError("District native leaf readback differs from its accepted preparation")
            target_store.verify_artifact(item["artifact"]["sha256"])
            processing = target_store.record_processing(dataset=source.dataset, inputs=[*roots, config_id], outputs={"leaf": leaf},
                processor="teadata.scripts.refresh_transfers.materialize", code_version=recorder.processor_sha256,
                parameters={"leaf_contract_sha256": leaf_id, "recipe": recipe, "toolchain": toolchain},
                validation={"raw_string_readback_equal": True, "rows": report["rows"], "field_correctness_qualified": False,
                            "publisher_totals_computed": False, "release_activation": False})
            ref = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name, canonical_json(contract), processing, "leaf")
            recorder.record_edition(key, source, artifacts=[ref], acquisition_ids=[value for value, _ in item["acquisitions"]])
            result["editions"].append({"key": key.to_dict(), "artifact_id": leaf_id, "rows": report["rows"],
                "prepared_processing_id": identifier, "processing_record_id": processing})
        catalog = recorder.catalog(schema_version=3)
        manifest = publish_release(stage, catalog, schema_version=3)
        result.update(build_id=catalog.build_id, manifest=file_identity(manifest))
        _write_index(stage, result, filename="MATERIALIZATION.json")
        if output.exists() or output.is_symlink():
            raise FileExistsError("District leaf candidate destination appeared before completion")
        stage.rename(output)
        return result
    finally:
        if stage.exists():
            shutil.rmtree(stage)
