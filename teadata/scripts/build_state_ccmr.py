"""Build a reviewable native release from the eight pinned Texas CCMR reports."""

from __future__ import annotations

from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from shutil import which
import sys
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from ..analytical import write_leaf_artifact
from ..column_schema import source_schema_fingerprint
from ..dataset_metadata import ArtifactRef, DataPeriod, DatasetMetadata, EditionKey, ResolvedDataset, STATEWIDE_SINGLETON_GRAIN
from ..manifest import BuildRecorder, ContractError, canonical_json, file_identity
from ..native_state_ccmr import (
    DATASET, EDITIONS, FIELDS, read_state_ccmr, state_ccmr_reference, state_ccmr_source_labels, state_ccmr_table,
)
from ..release import _install_candidate, publish_release
from ..teadata_config import load_config_bytes

if TYPE_CHECKING:
    from ..provenance import ProvenanceStore


def _reference(name: str, title: str, url: str, digest: str, receipt: str) -> dict:
    return {"reference_id": name, "title": title, "role": "source_notes", "url": url,
            "status": "available", "reviewed_sha256": digest, "provenance_ids": [receipt]}


def build_state_ccmr(source_store: ProvenanceStore, output: Path) -> Path:
    from ..provenance import ProvenanceStore

    reference = state_ccmr_reference()
    reports, manual = reference["reports"], reference["method_2018"]
    if tuple(row["rating_edition"] for row in reports) != EDITIONS:
        raise ContractError("State CCMR requires all eight reviewed report editions")
    pdf_extractor = which("pdftotext")
    if pdf_extractor is None:
        raise ContractError("State CCMR PDF preparation requires Poppler pdftotext")
    extractor_sha256 = file_identity(Path(pdf_extractor))["sha256"]
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".state-ccmr-", dir=output.parent) as directory:
        stage = Path(directory) / "bundle"
        stage.mkdir(mode=0o700)
        store = ProvenanceStore(stage / "provenance")
        store.copy_record_from(source_store, manual["receipt"])
        datasets, editions = [], {}
        for row in reports:
            year = row["rating_edition"]
            graduate_year = row["annual_graduates_year_end"]
            period = DataPeriod("school_year", graduate_year - 1, graduate_year).to_dict()
            digest, receipt = row["source_sha256"], row["source_receipt"]
            store.copy_record_from(source_store, receipt)
            labels = list(state_ccmr_source_labels(year))
            fields = [{"position": position, "code": label, "label": label, "field": field,
                       "disposition": "fact", "dictionary_status": "not_documented",
                       "publisher_type": None, "publisher_description": None}
                      for position, (label, field) in enumerate(zip(labels, FIELDS, strict=True), 1)]
            datasets.append({"dataset": DATASET, "edition": str(year), "source_only": True, "tables": [{
                "table": "state", "header_identity": "position", "input_sha256": digest,
                "source_schema_sha256": source_schema_fingerprint(labels, labels),
                "metric_period": period, "fields": fields,
            }]})
            refs = [_reference("state_report", "Original Texas CCMR report", row["direct_report"], digest, receipt)]
            if year == 2018:
                refs.append(_reference("adopted_manual", "2018 Accountability Manual, Chapter 2",
                                       manual["url"], manual["sha256"], manual["receipt"]))
            editions[year] = {"path": f"provenance/objects/{digest}", "metadata": {
                "title": "Texas annual graduate CCMR accountability totals",
                "publisher": "Texas Education Agency", "version": str(year), "period": period,
                "source_url": row["direct_report"], "references": refs,
                "notes": "Direct Texas totals. Credit retains the publisher's display precision and can include weighted credit. "
                         "Report editions can use different rules. The published CCMR percent is an independent observation.",
                "metrics": {field: {"field": field, "period": period, "unit": unit}
                            for field, unit in zip(FIELDS, ("count", None, "percent"), strict=True)},
            }, "reader": {"profiles": {"source": {"column_schema": "column-crosswalk.json"}}}}
        schema_path = stage / "column-crosswalk.json"
        schema_path.write_text(canonical_json({"schema_version": 2, "mapping_version": 1, "datasets": datasets}) + "\n")
        schema_pin = sha256(schema_path.read_bytes()).hexdigest()
        schema_receipt = store.record_acquisition(schema_path, ResolvedDataset(
            DATASET + ":crosswalk", "reference", None, None, "",
            DatasetMetadata(title="Reviewed Texas CCMR report labels and cell positions")))
        for source in editions.values():
            source["metadata"]["references"].append({"reference_id": "column_crosswalk", "role": "schema",
                "title": "Review of report labels and cell positions", "path": schema_path.name,
                "status": "available", "reviewed_sha256": schema_pin, "provenance_ids": [schema_receipt]})
        config = canonical_json({"data_sources": {DATASET: editions}}).encode()
        cfg = load_config_bytes(config, path=stage / "sources.yaml")
        recorder = BuildRecorder(cfg, store)
        toolchain = {"python": sys.version, "pyarrow": version("pyarrow")}
        for row in reports:
            year = row["rating_edition"]
            source = cfg.resolve_dataset(DATASET, year, exact=True, reader_profile="source")
            raw = read_state_ccmr(store, row, manual)
            table, support = state_ccmr_table(raw)
            digest, receipt = row["source_sha256"], row["source_receipt"]
            key = EditionKey(DATASET, str(year), EditionKey.source_revision((digest,)), "state", "state",
                             STATEWIDE_SINGLETON_GRAIN)
            recipe = {"operation": "direct_texas_accountability_ccmr_display", "report_edition": year,
                      "graduate_ending_year": row["annual_graduates_year_end"], "schema_sha256": schema_pin,
                      "percent": "as_published"}
            if year == 2018:
                recipe.update(report_pdf_page=3, manual_pdf_page=4, pdftotext_sha256=extractor_sha256)
            contract = recorder.leaf_contract(key, source, native_schema=table.schema.serialize().to_pybytes(),
                columns=table.column_names, key_columns=[], rows=1, inputs=[digest],
                recipe=recipe, toolchain=toolchain, column_support=support)
            contract_json = canonical_json(contract)
            leaf_id = sha256(contract_json.encode()).hexdigest()
            leaf = stage / (leaf_id + ".parquet")
            physical = write_leaf_artifact(leaf, table.to_batches(), contract=contract, batch_bytes=65536)
            roots = [receipt, schema_receipt]
            if year == 2018:
                roots.append(manual["receipt"])
            processing = store.record_processing(dataset=DATASET, inputs=roots, outputs={"leaf": leaf},
                processor="teadata.scripts.build_state_ccmr", code_version=recorder.processor_sha256,
                parameters={"leaf_contract_sha256": leaf_id, "recipe": recipe, "toolchain": toolchain},
                validation={"parsed_publisher_display": True, "scientific_qualification": False})
            ref = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name, contract_json, processing, "leaf")
            recorder.record_edition(key, source, artifacts=[ref], acquisition_ids=[receipt])
        publish_release(stage, recorder.catalog(schema_version=3).externalize_editions(), schema_version=4)
        _install_candidate(stage, output)
    return output / "release.json"


if __name__ == "__main__":
    import argparse

    from ..provenance import ProvenanceStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_store", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(build_state_ccmr(ProvenanceStore(args.source_store), args.output))
