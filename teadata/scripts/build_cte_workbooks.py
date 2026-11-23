"""Materialize explicitly reviewed public CTE workbook editions."""

from __future__ import annotations

from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from typing import TYPE_CHECKING, Any
from zipfile import ZipFile

from ..analytical import write_leaf_artifact
from ..column_schema import source_schema_fingerprint
from ..dataset_metadata import ArtifactRef, DatasetMetadata, EditionKey, ResolvedDataset, STATEWIDE_SINGLETON_GRAIN
from ..financial_leaves import MAX_BATCH_BYTES, MAX_SOURCE_BYTES, _package
from ..manifest import BuildRecorder, ContractError, canonical_json
from ..native_cte_workbooks import SCOPES, cte_workbook_projection, read_cte_sheet
from ..release import publish_release
from ..teadata_config import load_config_bytes

if TYPE_CHECKING:
    from ..provenance import ProvenanceStore


def _crosswalk(plan: dict[str, Any]) -> dict[str, Any]:
    tables = []
    for scope in plan.get("published_scopes", SCOPES):
        sheet = plan["scopes"][scope]
        headers = sheet["headers"]
        fields = []
        for position, (header, field) in enumerate(zip(headers, sheet["fields"], strict=True), 1):
            identity = field not in plan["metrics"]
            fields.append({"position": position, "code": header, "label": header, "field": field,
                           "disposition": "identity" if identity else "fact",
                           "publisher_type": None, "publisher_description": None,
                           "dictionary_status": "not_applicable" if identity else "not_documented"})
        tables.append({"table": scope, "input_sha256": plan["sha256"], "header_identity": "position",
                       "source_schema_sha256": source_schema_fingerprint(headers, headers), "fields": fields,
                       "metric_period": {"kind": "unknown"} if scope in plan.get("period_holds", {}) else plan["period"]})
    return {"schema_version": 2, "mapping_version": 1, "datasets": [
        {"dataset": plan["dataset"], "edition": plan["edition"], "source_only": True, "tables": tables}]}


def build_cte_workbook(plan: dict[str, Any], source_store: ProvenanceStore, output: Path) -> Path:
    """Create one immutable child; never attach an unqualified workbook here."""
    from ..provenance import ProvenanceStore

    published_scopes = tuple(plan.get("published_scopes", SCOPES))
    if not published_scopes or not set(published_scopes) <= set(SCOPES) or len(published_scopes) != len(set(published_scopes)):
        raise ContractError("Select distinct existing workbook scopes")
    output.mkdir()
    store = ProvenanceStore(output / "provenance")
    original_id = plan["publisher_receipt"]
    store.copy_record_from(source_store, original_id)
    publisher = store.read_record(original_id)
    digest = publisher["artifact"]["sha256"]
    if (publisher["method"] != "download" or digest != plan["sha256"]
            or not 0 < publisher["artifact"]["size_bytes"] <= MAX_SOURCE_BYTES):
        raise ContractError("CTE workbook requires an exact reviewed publisher observation")
    workbook = output / publisher["artifact"]["filename"]
    workbook.hardlink_to(store.verify_artifact(digest))
    schema = output / "cte-crosswalk.json"
    schema.write_text(canonical_json(_crosswalk(plan)) + "\n")
    dataset, year, edition = plan["dataset"], plan["catalog_year"], plan["edition"]
    schema_id = store.record_acquisition(schema, ResolvedDataset(dataset + ":schema", "reference", year, year, "",
        DatasetMetadata(title="TEA Data physical CTE workbook crosswalk")))
    metadata: dict[str, Any] = {
        "title": plan["title"], "publisher": "Texas Education Agency", "version": edition,
        "source_url": publisher["resolved_url"], "period": {"kind": "unknown"},
        "notes": plan["notes"],
        "references": [
            {"reference_id": "public_workbook", "title": "Publisher workbook and About worksheet",
             "role": "source_notes", "url": publisher["resolved_url"], "status": "available",
             "reviewed_sha256": digest, "provenance_ids": [original_id]},
            {"reference_id": "column_crosswalk", "title": "Original worksheet headers and canonical field names",
             "role": "schema", "path": schema.name, "status": "available",
             "reviewed_sha256": sha256(schema.read_bytes()).hexdigest(), "provenance_ids": [schema_id]},
        ],
    }
    observed = store.record_acquisition(workbook, ResolvedDataset(dataset, "data_sources", year, year, "",
        DatasetMetadata.from_dict(metadata)), parameters={"publisher_receipt": original_id})
    metadata["provenance_ids"] = [observed]
    config = output / "sources.yaml"
    config.write_text(canonical_json({"data_sources": {dataset: {year: {
        "path": workbook.name, "metadata": metadata,
        "reader": {"profiles": {"source": {"column_schema": schema.name}}}}}}}) + "\n")
    cfg = load_config_bytes(config.read_bytes(), path=config)
    source = cfg.resolve_dataset(dataset, year, reader_profile="source", exact=True)
    recorder = BuildRecorder(cfg, store)
    recorder.before_read(source, reader_profile="source")
    toolchain = {"python": sys.version, "architecture": platform.machine(), "pyarrow": version("pyarrow")}

    with ZipFile(workbook) as archive:
        sheets, strings = _package(archive, literal=True)
        if set(sheets) != {plan["about"], *(item["sheet"] for item in plan["scopes"].values())}:
            raise ContractError("CTE workbook has an unexpected worksheet")
        for scope in published_scopes:
            table = plan["scopes"][scope]
            sheet = table["sheet"]
            raw = read_cte_sheet(archive, sheets[sheet], strings, table["headers"], table["fields"])
            if raw.num_rows != table["rows"]:
                raise ContractError(f"CTE workbook row count differs for {dataset}/{scope}")
            projected, support = cte_workbook_projection(raw)
            keys = table["keys"]
            grain = table["grain"] if keys else STATEWIDE_SINGLETON_GRAIN
            if keys and (any(projected[name].null_count for name in keys)
                         or projected.select(keys).group_by(keys).aggregate([]).num_rows != raw.num_rows):
                raise ContractError(f"Missing or duplicate CTE keys for {dataset}/{scope}")
            key = EditionKey(dataset, edition, EditionKey.source_revision((digest,)), scope, scope, grain)
            recipe = {"operation": "public_cte_workbook_source", "sheet": sheet, "header_row": 1,
                      "about_sheet": plan["about"], "value_representation": "original decoded text and numeric XML tokens"}
            contract = recorder.leaf_contract(key, source, native_schema=projected.schema.serialize().to_pybytes(),
                columns=projected.column_names, key_columns=keys, rows=raw.num_rows, inputs=[digest],
                recipe=recipe, toolchain=toolchain, column_support=support)
            leaf_id = sha256(canonical_json(contract).encode()).hexdigest()
            leaf = output / (leaf_id + ".parquet")
            physical = write_leaf_artifact(leaf, projected.to_batches(max_chunksize=1024), contract=contract,
                                           batch_bytes=MAX_BATCH_BYTES)
            processing = store.record_processing(dataset=dataset, inputs=[observed, original_id, schema_id],
                outputs={"leaf": leaf}, processor="teadata.scripts.build_cte_workbooks",
                code_version=recorder.processor_sha256,
                parameters={"leaf_contract_sha256": leaf_id, "recipe": recipe, "toolchain": toolchain},
                validation={"source_rows": raw.num_rows, "field_correctness_qualified": False})
            ref = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name,
                              canonical_json(contract), processing, "leaf")
            recorder.record_edition(key, source, artifacts=[ref], acquisition_ids=[observed])
    return publish_release(output, recorder.catalog(schema_version=3).externalize_editions(), schema_version=4)


if __name__ == "__main__":
    import argparse

    from ..provenance import ProvenanceStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="Reviewed public workbook input JSON")
    parser.add_argument("source_store", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    selected = json.loads(arguments.plan.read_bytes())
    result = build_cte_workbook(selected, ProvenanceStore(arguments.source_store), arguments.output)
    print(sha256(result.read_bytes()).hexdigest())
