"""Prepare complete declared TAPR editions into a new, unpublished directory.

This source-preparation target never runs the entity engine, rewrites retained
workbooks, changes release pointers, or assigns scientific units from labels.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from teadata.dataset_metadata import DatasetMetadata
from teadata.manifest import processor_identity, verify_reference_receipts
from teadata.provenance import ProvenanceStore
from teadata.scripts.paths import data_path
from teadata.tapr import merge_tapr_tables
from teadata.teadata_config import load_config


def prepare_profiles(cfg, *, edition: str, scopes: list[str], input_dir: Path | None,
                     output_dir: Path, reference_store: ProvenanceStore | None = None) -> dict:
    """Validate all selected inputs, then publish one complete new generation."""
    import pyarrow.parquet as pq

    if edition not in {"2024", "2025"} or not scopes or len(scopes) != len(set(scopes)) or set(scopes) - {"campus", "district"}:
        raise ValueError("Select a declared 2024/2025 edition and distinct campus/district scopes")
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Preparation output already exists: {output_dir}")
    prepared = []
    for scope in scopes:
        dataset = scope + "_tapr_student_staff_profile"
        sources = cfg.read_tapr_sources(dataset, int(edition), input_dir=input_dir)
        table = merge_tapr_tables(sources, key_field=scope + "_number")
        if table.column_names != cfg.column_schema(dataset, int(edition), reader_profile="source"):
            raise ValueError("Full TAPR preparation does not account for its complete field contract")
        prepared.append((scope, dataset, sources, table))
    # No writer or provenance archive runs until every selected input validates.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".tapr-preparation-", dir=output_dir.parent))
    stage.chmod(0o700)
    try:
        store = ProvenanceStore(stage / "provenance")
        receipt = {"schema_version": 1, "edition": edition, "status": "validated_source_preparation",
                   "release_activation": False,
                   "toolchain": {"python": sys.version, "packages": {name: version(name) for name in ("pyarrow", "duckdb", "pandas", "openpyxl")}},
                   "processor_sha256": processor_identity(),
                   "preparation_code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                   "scopes": {}}
        for scope, dataset, sources, table in prepared:
            source = cfg.resolve_dataset(dataset, int(edition))
            contract = cfg.column_contract(dataset, int(edition), reader_profile="source")
            assert contract is not None
            crosswalk_path = stage / (scope + "-crosswalk.json")
            crosswalk_path.write_text(contract["text"])
            contract_observation = store.record_acquisition(crosswalk_path,
                replace(source, path=str(crosswalk_path), dataset=dataset + ":schema", section="reference",
                        metadata=DatasetMetadata(title="TEA Data full TAPR source/header contract")))
            inputs = [contract_observation]
            references = []
            for reference in cfg.reader_references(dataset, int(edition), reader_profile="source"):
                for identifier in reference.provenance_ids:
                    if reference_store is None:
                        raise ValueError("Claimed source references require an explicit reference_store / --reference-store")
                    store.copy_record_from(reference_store, identifier)
                    inputs.append(identifier)
                if reference.role == "schema" and reference.path == contract["path"]:
                    reference = replace(reference, provenance_ids=tuple(dict.fromkeys([
                        *reference.provenance_ids, contract_observation])))
                references.append(reference)
            bound_metadata = replace(source.metadata, provenance_ids=(), references=tuple(references))
            verify_reference_receipts(replace(source, metadata=bound_metadata), store,
                                      reference_base=cfg.base_dir, verify_all_claimed=True)
            annotations = []
            for item in sources:
                acquisition = store.record_acquisition(item.path, replace(source, path=str(item.path),
                    metadata=bound_metadata),
                    parameters={"observation": "retained local raw source; original acquisition time remains unknown",
                                "table": item.mapping["table"]})
                if store.read_record(acquisition)["artifact"]["sha256"] != item.input_sha256:
                    raise ValueError("TAPR input changed before provenance binding")
                inputs.append(acquisition)
                if "dictionary" in item.mapping:
                    descriptor = item.mapping["dictionary"]
                    path = item.path.with_name(descriptor["filename"])
                    observation = store.record_acquisition(path, replace(source, path=str(path),
                        dataset=dataset + ":dictionary", section="reference",
                        metadata=DatasetMetadata(title="Retained publisher TAPR Name/Type/Length/Label dictionary")))
                    if store.read_record(observation)["artifact"]["sha256"] != descriptor["sha256"]:
                        raise ValueError("Publisher dictionary changed before provenance binding")
                    inputs.append(observation)
                annotations.append({"table": item.mapping["table"], "source_sha256": item.input_sha256,
                                    "annotations": list(item.annotations)})
            note_path = stage / (scope + "-annotations.json")
            note_path.write_text(json.dumps(annotations, ensure_ascii=True, sort_keys=True) + "\n")
            parquet_path = stage / (scope + ".parquet")
            pq.write_table(table, parquet_path, compression="zstd", row_group_size=4096)
            reopened = pq.read_table(parquet_path)
            if not reopened.equals(table, check_metadata=True):
                raise ValueError("TAPR prepared source artifact differs from validated raw strings")
            validation = {"rows": table.num_rows, "fields": table.num_columns,
                          "source_columns_accounted": sum(len(item.mapping["fields"]) for item in sources),
                          "raw_string_readback_equal": True, "all_historical_source_rows_retained": True}
            processing = store.record_processing(dataset=dataset, inputs=list(dict.fromkeys(inputs)),
                outputs={"prepared_source": parquet_path, "crosswalk": crosswalk_path, "annotations": note_path},
                processor="teadata.scripts.build_tapr_profiles", code_version=receipt["processor_sha256"],
                parameters={"edition": edition, "column_schema_sha256": contract["sha256"],
                            "preparation_code_sha256": receipt["preparation_code_sha256"],
                            "toolchain": receipt["toolchain"], "whitespace": "preserve", "geospatial_filter": False},
                validation=validation)
            store.verify_record_closure([processing])
            receipt["scopes"][scope] = {**validation, "dataset": dataset, "processing_record_id": processing,
                "column_schema_sha256": contract["sha256"],
                "local_reference_closure_verified": True,
                "reference_states": [{"reference_id": ref.reference_id, "availability": ref.status,
                    "acquisition_state": "retained" if ref.provenance_ids else "not_acquired",
                    "provenance_ids": list(ref.provenance_ids)} for ref in references],
                "source_tables": [{"table": item.mapping["table"], "source_revision_sha256": item.input_sha256,
                                   "publisher_columns": len(item.mapping["fields"]), "rows": item.table.num_rows}
                                  for item in sources],
                "coverage_scope": "Complete columns of the explicitly declared source tables; not all annual TAPR download families.",
                "qualified_metrics": {key: value.to_dict() for key, value in source.metadata.metrics.items()},
                "other_field_semantics": "Unit/reference period unknown unless explicitly bound by qualified_metrics; edition is not a field reference period.",
                "identity_fields": [name for name in table.column_names if name in {"campus_number", "campus_name", "district_number", "district_name"}],
                "identity_storage": "Lossless source/history attributes; not additional metric keys or entity.meta hydration."}
        (stage / "preparation.json").write_text(json.dumps(receipt, ensure_ascii=True, sort_keys=True, indent=2) + "\n")
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"Preparation output appeared during staging: {output_dir}")
        os.rename(stage, output_dir)
        return receipt
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edition", required=True, choices=("2024", "2025"))
    parser.add_argument("--scope", action="append", choices=("campus", "district"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--reference-store", type=Path,
                        help="Canonical store containing the exact claimed reference receipts and bytes")
    parser.add_argument("--input-dir", type=Path, default=data_path("tapr", "raw"))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = prepare_profiles(load_config(args.config), edition=args.edition,
        scopes=args.scope or ["campus", "district"], input_dir=args.input_dir, output_dir=args.output_dir,
        reference_store=ProvenanceStore(args.reference_store) if args.reference_store is not None else None)
    print(json.dumps({"edition": args.edition, "output_dir": str(args.output_dir),
                      "scopes": {key: {"rows": value["rows"], "fields": value["fields"]}
                                 for key, value in result["scopes"].items()}}))


if __name__ == "__main__":
    main()
