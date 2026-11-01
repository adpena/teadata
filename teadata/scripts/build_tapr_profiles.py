"""Prepare complete declared TAPR editions into a new, unpublished directory.

This source-preparation target never runs the entity engine, rewrites retained
workbooks, changes release pointers, or assigns scientific units from labels.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
from importlib.metadata import version
from itertools import zip_longest
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from teadata.column_schema import crosswalk_dataset, parse_column_schema, reader_options
from teadata.analytical import write_leaf_artifact
from teadata.dataset_metadata import ArtifactRef, DatasetMetadata, EditionKey
from teadata.identifiers import canonical_source_identifiers
from teadata.manifest import BuildRecorder, _validate_methodology, canonical_json, file_identity, processor_identity, verify_reference_receipts
from teadata.provenance import ProvenanceStore, validate_reference_records
from teadata.release import publish_release
from teadata.scripts.paths import data_path
from teadata.tapr import (
    TAPR_PROFILE_FAMILIES, merge_tapr_tables, qualified_tapr_family,
    reviewed_tapr_data_record_count, tapr_source_selection, validate_tapr_key,
)
from teadata.teadata_config import load_config


def prepare_profiles(cfg, *, edition: str, scopes: list[str], input_dir: Path | None,
                     output_dir: Path, reference_store: ProvenanceStore | None = None,
                     family: str = "student_staff_profile",
                     processing_inputs: Mapping[str, Sequence[str]] | None = None,
                     prepared_inputs: Mapping[str, tuple[str, tuple[str, ...]]] | None = None) -> dict:
    """Validate all selected inputs, then publish one complete new generation.

    ``scopes``, ``family`` and ``input_dir`` select CSV intake only. An explicit
    ``prepared_inputs`` mapping instead selects every output dataset itself.
    """
    try:
        year = int(edition)
    except (TypeError, ValueError) as error:
        raise ValueError("Select a canonical annual integer edition") from error
    if not isinstance(edition, str) or edition != str(year) or year < 1:
        raise ValueError("Select a canonical annual integer edition")
    if prepared_inputs is None and (not scopes or len(scopes) != len(set(scopes))
            or set(scopes) - {"campus", "district"} or family not in TAPR_PROFILE_FAMILIES):
        raise ValueError("Select a declared family and distinct campus/district scopes")
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Preparation output already exists: {output_dir}")
    if prepared_inputs is not None:
        return _prepare_edition_leaves(cfg, edition=edition, year=year, output_dir=output_dir,
            reference_store=reference_store, prepared_inputs=prepared_inputs, processing_inputs=processing_inputs)
    import pyarrow.parquet as pq
    selected_datasets = {scope + "_tapr_" + family for scope in scopes}
    dependencies = dict(processing_inputs or {})
    if set(dependencies) - selected_datasets:
        raise ValueError("Processing inputs must name an exact selected dataset/scope")
    for dataset, identifiers in dependencies.items():
        if isinstance(identifiers, (str, bytes)) or not isinstance(identifiers, Sequence):
            raise ValueError("Processing inputs require an ordered sequence of receipt IDs")
        dependencies[dataset] = list(dict.fromkeys(identifiers))
        for identifier in dependencies[dataset]:
            if reference_store is None:
                raise ValueError("Processing inputs require an explicit reference_store")
            if reference_store.read_record(identifier)["kind"] != "processing":
                raise ValueError("Selected processing input is not a processing receipt")
            reference_store.verify_record_closure([identifier])
    prepared = []
    for scope in scopes:
        dataset = scope + "_tapr_" + family
        sources = cfg.read_tapr_sources(dataset, year, input_dir=input_dir)
        table = merge_tapr_tables(sources, key_field=scope + "_number")
        if table.column_names != cfg.column_schema(dataset, year, reader_profile="source"):
            raise ValueError("Full TAPR preparation does not account for its complete field contract")
        prepared.append((scope, dataset, sources, table))
    # No writer or provenance archive runs until every selected input validates.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".tapr-preparation-", dir=output_dir.parent))
    stage.chmod(0o700)
    try:
        store = ProvenanceStore(stage / "provenance")
        receipt = {"schema_version": 1, "edition": edition, "family": family,
                   "status": "validated_source_preparation",
                   "release_activation": False,
                   "toolchain": {"python": sys.version, "packages": {name: version(name) for name in ("pyarrow", "duckdb", "pandas", "openpyxl")}},
                   "processor_sha256": processor_identity(),
                   "preparation_code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                   "scopes": {}}
        for scope, dataset, sources, table in prepared:
            source = cfg.resolve_dataset(dataset, year, reader_profile="source")
            contract = cfg.column_contract(dataset, year, reader_profile="source")
            assert contract is not None
            crosswalk_path = stage / (scope + "-crosswalk.json")
            crosswalk_path.write_text(contract["text"])
            contract_observation = store.record_acquisition(crosswalk_path,
                replace(source, path=str(crosswalk_path), dataset=dataset + ":schema", section="reference",
                        metadata=DatasetMetadata(title="TEA Data full TAPR source/header contract")))
            inputs = [contract_observation]
            selected_processing = dependencies.get(dataset, [])
            for identifier in selected_processing:
                assert reference_store is not None
                store.copy_record_from(reference_store, identifier)
                inputs.append(identifier)
            selected = crosswalk_dataset(parse_column_schema(contract["text"].encode()), dataset, edition)
            if selected.get("source_only"):
                # Original HTTP observations may predate a source-family split.
                # An explicit reviewed source-notes reference binds those exact
                # bytes without relabeling the immutable acquisition's dataset.
                raw_references = {identifier for ref in source.metadata.references
                                  if ref.reference_id == "csv_headers" and ref.role == "source_notes" and ref.reviewed_sha256 in
                                  {item.input_sha256 for item in sources}
                                  for identifier in ref.provenance_ids}
                if reference_store is None or not (source.metadata.provenance_ids or raw_references):
                    raise ValueError("Source-only preparation requires original acquisition receipts in an explicit reference_store")
                acquired = set()
                expected = {item.input_sha256 for item in sources}
                for identifier in dict.fromkeys([*source.metadata.provenance_ids, *sorted(raw_references)]):
                    store.copy_record_from(reference_store, identifier)
                    record = store.read_record(identifier)
                    if (record["kind"] != "acquisition"
                            or (identifier in source.metadata.provenance_ids and record["dataset"] != dataset)
                            or record["artifact"]["sha256"] not in expected):
                        raise ValueError("Source acquisition receipt does not identify the selected raw input")
                    acquired.add(record["artifact"]["sha256"])
                    inputs.append(identifier)
                if acquired != expected:
                    raise ValueError("Original acquisition receipts do not account for every source-only raw input")
            references = []
            for reference in cfg.reader_references(dataset, year, reader_profile="source"):
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
                    parameters={"observation": "local raw-source read; acquisition history is retained separately when available",
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
                            "toolchain": receipt["toolchain"], "family": family,
                            "selected_processing_inputs": selected_processing,
                            "whitespace": "preserve", "geospatial_filter": False},
                validation=validation)
            store.verify_record_closure([processing])
            receipt["scopes"][scope] = {**validation, "dataset": dataset, "processing_record_id": processing,
                "selected_processing_inputs": selected_processing,
                "column_schema_sha256": contract["sha256"],
                "local_reference_closure_verified": True,
                "reference_states": [{"reference_id": ref.reference_id, "availability": ref.status,
                    "acquisition_state": "retained" if ref.provenance_ids else "not_acquired",
                    "provenance_ids": list(ref.provenance_ids)} for ref in references],
                "source_tables": [{"table": item.mapping["table"], "source_revision_sha256": item.input_sha256,
                                   "publisher_columns": len(item.mapping["fields"]), "rows": item.table.num_rows,
                                   **({"metric_period": item.mapping["metric_period"]}
                                      if "metric_period" in item.mapping else {})}
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


def _prepare_edition_leaves(cfg, *, edition: str, year: int, output_dir: Path,
                            reference_store: ProvenanceStore | None,
                            prepared_inputs: Mapping[str, tuple[str, tuple[str, ...]]],
                            processing_inputs: Mapping[str, Sequence[str]] | None) -> dict:
    """One explicit raw-wide cohort, derived from accepted source preparations."""
    extra = dict(processing_inputs or {})
    if reference_store is None or not prepared_inputs or set(extra) - set(prepared_inputs):
        raise ValueError("Prepared mode requires one reference store and explicit selected datasets")
    validated = []
    # All metadata and typed closures validate before any Arrow import/open or staging write.
    for dataset, selected in prepared_inputs.items():
        if (not isinstance(selected, tuple) or len(selected) != 2 or not isinstance(selected[1], tuple)
                or len(selected[1]) != 1 or not isinstance(selected[1][0], str)):
            raise ValueError("Prepared input requires a processing ID and an explicit ordered singleton key")
        identifier, keys = selected
        source = cfg.resolve_dataset(dataset, year, reader_profile="source")
        if source.resolved_year != year or source.metadata.version != edition:
            raise ValueError("Prepared input requires an exact registered edition")
        selection = tapr_source_selection(reader_options(source.reader, "source"))
        scope = selection["scope"]
        family = qualified_tapr_family(source.dataset, scope + "_number")
        resource = cfg.column_contract(dataset, year, reader_profile="source")
        if resource is None:
            raise ValueError("Prepared input requires a reviewed source contract")
        document = parse_column_schema(resource["text"].encode())
        branch = crosswalk_dataset(document, dataset, edition)
        if (len(branch["tables"]) != 1
                or set(selection["tables"]) != {table["table"] for table in branch["tables"]}):
            raise ValueError("Prepared dataset requires one complete, unambiguous declared source table")
        table, = branch["tables"]
        source_rows = reviewed_tapr_data_record_count(table)
        cfg.header_records(dataset, edition=edition, reader_profile="source")
        validate_tapr_key(table, keys[0])
        if keys != (scope + "_number",):
            raise ValueError("Explicit key differs from the selected TAPR source scope")
        graph = reference_store.verify_record_closure([identifier])
        record = graph[identifier]
        parameters, validation = record.get("parameters", {}), record.get("validation", {})
        if (record["kind"] != "processing" or record["dataset"] != dataset
                or not isinstance(parameters, dict) or not isinstance(validation, dict)
                or record.get("processor") != "teadata.scripts.build_tapr_profiles"
                or parameters.get("edition") != edition or parameters.get("family") != family
                or parameters.get("column_schema_sha256") != resource["sha256"]
                or record.get("outputs", {}).get("crosswalk", {}).get("sha256") != resource["sha256"]
                or validation.get("raw_string_readback_equal") is not True
                or validation.get("all_historical_source_rows_retained") is not True
                or type(validation.get("rows")) is not int or validation["rows"] != source_rows
                or validation.get("fields") != len(table["fields"])
                or validation.get("source_columns_accounted") != len(table["fields"])):
            raise ValueError("Processing receipt is not the selected complete source preparation")
        artifact = record["outputs"]["prepared_source"]
        raw_sha = table["input_sha256"]
        raw_refs = {value for ref in source.metadata.references
                    if ref.reference_id == "csv_headers" and ref.role == "source_notes" and ref.reviewed_sha256 == raw_sha
                    for value in ref.provenance_ids}
        acquisitions = list(dict.fromkeys([*source.metadata.provenance_ids, *sorted(raw_refs)]))
        if not acquisitions or any(graph[value]["kind"] != "acquisition"
                or graph[value]["artifact"]["sha256"] != raw_sha
                or graph[value].get("source_metadata", {}).get("version") != edition for value in acquisitions):
            raise ValueError("Selected prepared source does not close its exact upstream raw revision")
        validate_reference_records(source, graph.__getitem__, verify_all_claimed=True)
        _validate_methodology(source.metadata, cfg.value_policies.get(source.metadata.value_policy))
        declared = parameters.get("selected_processing_inputs", [])
        if (not isinstance(declared, list) or any(not isinstance(value, str) for value in declared)
                or len(set(declared)) != len(declared)
                or any(value not in record["inputs"] or graph[value]["kind"] != "processing" for value in declared)):
            raise ValueError("Prepared annotation dependencies are not explicit typed input edges")
        dependencies = extra.get(dataset, [])
        if isinstance(dependencies, (str, bytes)) or not isinstance(dependencies, Sequence):
            raise ValueError("Additional processing inputs require an ordered receipt sequence")
        dependencies = list(dict.fromkeys(dependencies))
        for value in dependencies:
            if reference_store.read_record(value)["kind"] != "processing":
                raise ValueError("Additional dependency is not a processing receipt")
            reference_store.verify_record_closure([value])
        validated.append((dataset, scope, family, table["table"], source, identifier, keys, acquisitions,
                          artifact, raw_sha, validation["rows"], dependencies, declared))

    import pyarrow as pa
    import pyarrow.compute as arrow_compute
    import pyarrow.parquet as pq

    pc: Any = arrow_compute  # Arrow registers compute kernels dynamically.

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".tapr-edition-", dir=output_dir.parent))
    stage.chmod(0o700)
    try:
        store = ProvenanceStore(stage / "provenance")
        recorder = BuildRecorder(cfg, store)
        result = {"schema_version": 3, "edition": edition, "status": "unpublished_raw_wide_cohort",
                  "release_activation": False, "datasets": {}}
        toolchain = {"python": sys.version, "pyarrow": version("pyarrow")}
        lock = Path(__file__).parents[2] / "uv.lock"
        if lock.is_file():
            toolchain["lock_sha256"] = sha256(lock.read_bytes()).hexdigest()
        recipe = {"operation": "lossless_prepared_source_batches", "batch_size": 4096, "compression": "zstd",
                  "preparation_code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "values_interpreted": False}
        for dataset, scope, family, table_name, source, identifier, keys, acquisitions, artifact, raw_sha, rows, dependencies, declared in validated:
            for value in [identifier, *dependencies]:
                store.copy_record_from(reference_store, value)
            prepared = store.verify_artifact(artifact["sha256"])
            key = EditionKey(dataset, edition, EditionKey.source_revision((raw_sha,)), scope, table_name, "one row per " + keys[0])
            with pq.ParquetFile(prepared) as native:
                schema = native.schema_arrow.remove_metadata()
                if (schema.names != cfg.column_schema(dataset, year, reader_profile="source")
                        or any(not (pa.types.is_string(field.type) or pa.types.is_large_string(field.type)) for field in schema)
                        or native.metadata.num_rows != rows):
                    raise ValueError("Prepared native physical schema/rows differ from the complete raw contract")
                contract = recorder.leaf_contract(key, source, native_schema=schema.serialize().to_pybytes(),
                    columns=schema.names, key_columns=keys, rows=rows, inputs=[raw_sha, artifact["sha256"]], recipe=recipe, toolchain=toolchain)
                identity = sha256(canonical_json(contract).encode()).hexdigest()
                leaf = stage / (identity + ".parquet")

                def batches():
                    identities = []
                    for batch in native.iter_batches(batch_size=4096, use_threads=False):
                        canonical = canonical_source_identifiers(batch.column(schema.get_field_index(keys[0])),
                            width=9 if keys[0] == "campus_number" else 6)
                        if pc.any(pc.is_null(canonical)).as_py():
                            raise ValueError("Prepared source has missing physical keys")
                        identities.append(canonical)
                        yield batch
                    all_keys = pa.chunked_array(identities)
                    if pc.count_distinct(all_keys).as_py() != rows:
                        raise ValueError("Prepared source has duplicate physical keys")

                physical = write_leaf_artifact(leaf, batches(), contract=contract)
                with pq.ParquetFile(leaf) as reopened:
                    for before, after in zip_longest(native.iter_batches(batch_size=4096, use_threads=False),
                            reopened.iter_batches(batch_size=4096, use_threads=False)):
                        if before is None or after is None or not before.equals(after, check_metadata=False):
                            raise ValueError("Raw leaf native batch readback differs from selected preparation")
            if file_identity(prepared)["sha256"] != artifact["sha256"]:
                raise ValueError("Prepared source changed during materialization")
            processing = store.record_processing(dataset=dataset, inputs=[identifier, *dependencies], outputs={"leaf": leaf},
                processor="teadata.scripts.build_tapr_profiles", code_version=recorder.processor_sha256,
                parameters={"edition": edition, "family": family, "leaf_contract_sha256": identity,
                    "recipe": recipe, "toolchain": toolchain, "key_columns": list(keys), "selected_processing_inputs": dependencies},
                validation={"rows": rows, "fields": len(schema), "raw_string_readback_equal": True, "values_interpreted": False})
            ref = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name, canonical_json(contract), processing, "leaf")
            recorder.record_edition(key, source, artifacts=[ref], acquisition_ids=acquisitions)
            result["datasets"][dataset] = {"rows": rows, "fields": len(schema), "key": key.to_dict(), "key_columns": list(keys),
                "artifact_id": identity, "processing_record_id": processing, "prepared_processing_id": identifier,
                "inherited_processing_inputs": declared, "raw_string_readback_equal": True, "values_interpreted": False}
        catalog = recorder.catalog(schema_version=3)
        manifest = publish_release(stage, catalog, schema_version=3)
        result["manifest"] = file_identity(manifest)
        result["build_id"] = catalog.build_id
        (stage / "preparation.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError("Cohort output appeared during staging")
        os.rename(stage, output_dir)
        return result
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edition", required=True, help="Canonical annual integer edition, exactly declared by Config")
    parser.add_argument("--family", choices=TAPR_PROFILE_FAMILIES)
    parser.add_argument("--scope", action="append", choices=("campus", "district"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--reference-store", type=Path,
                        help="Canonical store containing the exact claimed reference receipts and bytes")
    parser.add_argument("--processing-input", action="append", nargs=2, default=[], metavar=("DATASET", "RECORD_ID"),
                        help="Explicit processing dependency for one exact selected dataset/scope; applicability is caller-reviewed")
    parser.add_argument("--prepared-input", action="append", nargs=3, metavar=("DATASET", "PROCESSING_ID", "KEY_COLUMN"),
                        help="Opt-in raw-wide schema3 selection: exact prepared_source processing output and singleton physical key for each explicitly selected Config dataset")
    parser.add_argument("--input-dir", type=Path, default=data_path("tapr", "raw"))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    processing_inputs: dict[str, list[str]] = {}
    for dataset, identifier in args.processing_input:
        processing_inputs.setdefault(dataset, []).append(identifier)
    prepared_inputs = None
    if args.prepared_input is not None:
        if args.family is not None or args.scope is not None:
            parser.error("Prepared dataset selection cannot be combined with CSV --family or --scope")
        prepared_inputs = {}
        for dataset, identifier, key in args.prepared_input:
            if dataset in prepared_inputs:
                parser.error("Duplicate prepared dataset selection")
            prepared_inputs[dataset] = (identifier, (key,))
    result = prepare_profiles(load_config(args.config), edition=args.edition,
        scopes=args.scope or ["campus", "district"], input_dir=args.input_dir, output_dir=args.output_dir,
        reference_store=ProvenanceStore(args.reference_store) if args.reference_store is not None else None,
        family=args.family or "student_staff_profile", processing_inputs=processing_inputs, prepared_inputs=prepared_inputs)
    if prepared_inputs is not None:
        print(json.dumps({"schema_version": 3, "edition": args.edition, "output_dir": str(args.output_dir),
                          "datasets": result["datasets"]}))
        return
    print(json.dumps({"edition": args.edition, "output_dir": str(args.output_dir),
                      "scopes": {key: {"rows": value["rows"], "fields": value["fields"]}
                                 for key, value in result["scopes"].items()}}))


if __name__ == "__main__":
    main()
