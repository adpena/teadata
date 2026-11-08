"""Build a native local candidate for one exact fall or spring AskTED archive."""

from __future__ import annotations

import argparse
from csv import reader
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from typing import Any

from .analytical import write_leaf_artifact
from .askted_directory import DirectoryField, DirectorySource, Season, _archive_rows, _projection, _verified_archive
from .column_schema import source_schema_fingerprint
from .dataset_metadata import ArtifactRef, DataPeriod, DatasetMetadata, EditionKey, ResolvedDataset
from .manifest import BuildRecorder, canonical_json, file_identity
from .provenance import ProvenanceStore
from .release import _install_candidate, publish_release
from .scripts import download_askted
from .teadata_config import load_config_bytes

DATASET = "public_school_directory"
_RECIPE = "askted_archive_raw_directory_projection"
_BOUNDS = 2048
_LOCATOR = "source_record"

def _scoped_fields(scope: str, publisher_prefix: str) -> dict[str, str]:
    local = {DirectoryField.NAME, DirectoryField.SITE_CITY, DirectoryField.MAILING_CITY}
    return {
        "district_number": "District Number",
        "campus_number" if scope == "campus" else "source_campus_number": "School Number",
        **{f"{scope}_{field}" if field in local else str(field): column
           for field, column in _projection(publisher_prefix).items()},
    }


_FIELDS = {"district": _scoped_fields("district", "District"), "campus": _scoped_fields("campus", "School")}


def _metadata(stage: Path, target: ProvenanceStore, archive: DirectorySource) -> tuple[bytes, dict[str, Any]]:
    year, edition = archive.school_year_end, f"{archive.school_year_end}-{archive.season}"
    period = DataPeriod("as_of", as_of=archive.publisher_declared_as_of).to_dict()
    tables = []
    for scope, fields in _FIELDS.items():
        headers = list(fields.values())
        tables.append({"table": scope, "input_sha256": archive.sha256, "header_identity": "code",
            "source_schema_sha256": source_schema_fingerprint(headers, headers), "metric_period": period,
            "fields": [{"position": i, "code": header, "label": header, "field": field,
                        "disposition": "identity", "dictionary_status": "not_applicable",
                        "publisher_type": None, "publisher_description": None}
                       for i, (field, header) in enumerate(fields.items(), 1)]})
    crosswalk = stage / "askted-crosswalk.json"
    crosswalk.write_text(canonical_json({"schema_version": 2, "mapping_version": 1, "datasets": [
        {"dataset": DATASET, "edition": edition, "source_only": True, "tables": tables},
    ]}), encoding="utf-8")
    crosswalk_pin = file_identity(crosswalk)["sha256"]
    crosswalk_receipt = target.record_acquisition(crosswalk, ResolvedDataset(
        DATASET + ":crosswalk", "reference", year, year, "",
        DatasetMetadata(title="TEA Data mapping for selected AskTED archive fields", version=edition)))
    metadata = {
        "title": f"AskTED {year - 1}–{year} {archive.season} school and district directory",
        "publisher": archive.publisher, "source_url": archive.source_url,
        "download_url": archive.source_url, "version": edition, "period": period,
        "notes": ("Selected raw publisher cells. The date is the publisher's archive label. "
                  "Repeated identifiers and empty cells are retained. Campus county and ESC fields describe "
                  "the district's administrative context as printed in that row."),
        "provenance_ids": [archive.receipt],
        "references": [
            {"reference_id": "archive_index", "role": "source_notes", "status": "available",
             "url": archive.source_url, "reviewed_sha256": target.read_record(archive.index_receipt)["artifact"]["sha256"],
             "provenance_ids": [archive.index_receipt]},
            {"reference_id": "definitions", "role": "glossary", "status": "unreviewed", "url": archive.definitions_url},
            {"reference_id": "selected_fields", "role": "schema", "status": "available",
             "path": crosswalk.name, "reviewed_sha256": crosswalk_pin, "provenance_ids": [crosswalk_receipt]},
        ],
    }
    config = {"data_sources": {DATASET: {year: {"path": f"provenance/objects/{archive.sha256}",
        "metadata": metadata, "reader": {"profiles": {"source": {"column_schema": crosswalk.name}}}}}}}
    return canonical_json(config).encode(), {"crosswalk_sha256": crosswalk_pin,
                                            "crosswalk_receipt": crosswalk_receipt, "edition": edition}


def build_askted_archive(store: ProvenanceStore, receipt: str, *, year: int, season: Season,
                        expected_sha256: str, output: Path) -> Path:
    """Write a new standalone directory release without touching an active release."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    archive, text = _verified_archive(store, receipt, year=year, season=season, expected_sha256=expected_sha256)
    header = next(reader(StringIO(text, newline=""), strict=True))
    source_ordinals = {label: number for number, label in enumerate(header, 1)}
    header_digest = sha256(canonical_json(header).encode()).hexdigest()
    scoped = {scope: {field: [] for field in (*fields, _LOCATOR)} for scope, fields in _FIELDS.items()}
    for record, _, campus_id, raw in _archive_rows(text):
        for scope, fields in _FIELDS.items():
            if scope == "campus" and campus_id is None:
                continue
            columns = scoped[scope]
            for name, original in fields.items():
                columns[name].append(raw[original])
            columns[_LOCATOR].append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".askted-native-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir(mode=0o700)
        target = ProvenanceStore(stage / "provenance")
        for identifier in (archive.receipt, archive.index_receipt):
            target.copy_record_from(store, identifier)
        config, evidence = _metadata(stage, target, archive)
        config_path = stage / "source-config.json"
        config_path.write_bytes(config)
        cfg = load_config_bytes(config, path=config_path)
        source = cfg.resolve_dataset(DATASET, year, exact=True, reader_profile="source")
        recorder = BuildRecorder(cfg, target)
        config_id = target.record_acquisition(config_path, ResolvedDataset(
            DATASET + ":config", "reference", year, year, "", DatasetMetadata(title="Exact AskTED candidate Config")))
        edition = evidence["edition"]
        config_digest = file_identity(config_path)["sha256"]
        archive_parser_digest = file_identity(Path(download_askted.__file__))["sha256"]
        toolchain = {"python": platform.python_version(), "pyarrow": version("pyarrow"),
                     "platform": platform.platform(), "architecture": platform.machine()}
        entries = []
        for scope, fields in _FIELDS.items():
            columns = scoped[scope]
            schema = pa.schema([*(pa.field(name, pa.string(), nullable=False) for name in fields),
                                pa.field(_LOCATOR, pa.int64(), nullable=False)])
            table = pa.Table.from_pydict(columns, schema=schema)
            if not table.num_rows:
                raise ValueError(f"AskTED archive has no {scope} rows")
            key = EditionKey(DATASET, edition, EditionKey.source_revision((archive.sha256,)),
                             scope, scope, "one logical source record per row")
            recipe = {"operation": _RECIPE, "scope": scope, "batch_rows": _BOUNDS,
                      "archive_index_receipt": archive.index_receipt,
                      "original_header_sha256": header_digest,
                      "publisher_column_ordinals": {field: source_ordinals[label] for field, label in fields.items()},
                      "data_origin": archive.data_origin,
                      "config_sha256": config_digest, "archive_parser_sha256": archive_parser_digest,
                      "crosswalk_sha256": evidence["crosswalk_sha256"]}
            contract = recorder.leaf_contract(key, source, native_schema=schema.serialize().to_pybytes(),
                columns=table.column_names, key_columns=[_LOCATOR], rows=table.num_rows, inputs=[archive.sha256],
                recipe=recipe, toolchain=toolchain, column_support={"fields": {}, "context": [_LOCATOR]})
            identity = sha256(canonical_json(contract).encode()).hexdigest()
            leaf = stage / f"{identity}.parquet"
            physical = write_leaf_artifact(leaf, table.to_batches(max_chunksize=_BOUNDS), contract=contract,
                                           batch_bytes=8 * 1024**2)
            if not table.equals(pq.read_table(leaf, use_threads=False).replace_schema_metadata(None), check_metadata=True):
                raise ValueError("AskTED Parquet readback differs from its exact projected records")
            processing = target.record_processing(dataset=DATASET,
                inputs=[archive.receipt, archive.index_receipt, evidence["crosswalk_receipt"], config_id],
                outputs={"leaf": leaf}, processor="teadata.askted_leaves", code_version=recorder.processor_sha256,
                parameters={"leaf_contract_sha256": identity, "recipe": recipe, "toolchain": toolchain},
                validation={"native_raw_readback_equal": True, "projected_cells": table.num_rows * len(fields),
                            "release_activation": False})
            ref = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name,
                              canonical_json(contract), processing, "leaf")
            recorder.record_edition(key, source, artifacts=[ref], acquisition_ids=[archive.receipt])
            entries.append({"edition": key.to_dict(), "rows": table.num_rows, "processing_receipt": processing})
        catalog = recorder.catalog(schema_version=4)
        publish_release(stage, catalog, schema_version=4)
        (stage / "MATERIALIZATION.json").write_text(canonical_json({
            "status": "unpublished_native_askted_candidate", "build_id": catalog.build_id,
            "generic_source_parity_qualified": False, "release_activation": False, "editions": entries,
        }) + chr(10), encoding="utf-8")
        _install_candidate(stage, output)
    return output / "release.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--season", choices=("fall", "spring"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build_askted_archive(ProvenanceStore(args.store), args.receipt, year=args.year,
                               season=args.season, expected_sha256=args.sha256, output=args.output))


if __name__ == "__main__":
    main()
