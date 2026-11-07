"""Build the exact 2025 special-education Indicator 1 native source candidate."""

from __future__ import annotations

import csv
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import platform
import re
from shutil import which
from subprocess import check_output
import sys
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .analytical import write_leaf_artifact
from .column_schema import crosswalk_dataset, parse_column_schema
from .dataset_metadata import ArtifactRef, DatasetMetadata, EditionKey, ResolvedDataset
from .manifest import BuildRecorder, ContractError, canonical_json, file_identity
from .provenance import ProvenanceStore, validate_reference_records
from .rda import rda_indicator1_metadata, read_rda_indicator1
from .release import _install_candidate, publish_release
from .teadata_config import load_config_bytes

NATIVE_DATASET = "rda_special_education_indicator1"
_YEAR = 2025
_AREA = "04"
_CONFIG = Path(__file__).with_name("teadata_sources.yaml")
_RAW_COLUMNS = {
    "single_year_rate": "single_year_rate_raw",
    "multi_year_rate": "multi_year_rate_raw",
    "numerator": "numerator_raw",
    "denominator": "denominator_raw",
}
_STATE_COLUMNS = ("subject", "state_rate", "state_numerator", "state_denominator")
_KEYS = {"district": "district_number", "state": "subject"}
_STATE_NUMBER = re.compile(r"(?:0|[1-9][0-9]{0,2})(?:,[0-9]{3})*")


def _checked_source(store: ProvenanceStore, reference: dict[str, Any]) -> Path:
    receipt = store.read_record(reference["receipt"])
    if (receipt.get("kind") != "acquisition" or receipt.get("requested_url") != reference["url"]
            or receipt["source_metadata"].get("version") != str(_YEAR)
            or receipt.get("requested_year") != _YEAR
            or receipt["artifact"]["sha256"] != reference["sha256"]
            or receipt["artifact"]["size_bytes"] != reference["size_bytes"]):
        raise ContractError("RDA receipt differs from the reviewed source")
    if "form_controls" in reference and (
        receipt["parameters"].get("http_method") != reference["method"]
        or receipt["parameters"].get("form_controls") != reference["form_controls"]
    ):
        raise ContractError("RDA receipt differs from the reviewed publisher selection")
    return store.verify_artifact(reference["sha256"])


def _district_projection(path: Path, mapping: dict[str, Any], metadata: dict[str, Any]) -> pa.Table:
    observed = read_rda_indicator1(path, edition=_YEAR, area=_AREA)
    count = metadata["district_rows"]
    columns = {"DISTRICT": observed["district_number"].slice(0, count),
               "REGION": observed["education_service_center"].slice(0, count)}
    names = {"DISTRICT": "district_number", "REGION": "education_service_center"}
    for index, field in enumerate(metadata["fields"]):
        subject = observed["subject"].slice(index * count, 1)[0].as_py()
        if subject != field["subject"]:
            raise ContractError("RDA district projection has a different subject order")
        for metric, raw in _RAW_COLUMNS.items():
            code = field[f"{metric}_code"]
            columns[code] = observed[raw].slice(index * count, count)
            names[code] = f"{subject.lower().replace(' ', '_')}_{metric}"
    with path.open(encoding="cp1252", newline="") as stream:
        publisher_order = [code for code in next(csv.reader(stream, strict=True)) if code in columns]
    if publisher_order != [field["code"] for field in mapping["fields"]] or len(columns) != len(publisher_order):
        raise ContractError("RDA district crosswalk differs from the reviewed CSV column order")
    if any(field["field"] != names[field["code"]] for field in mapping["fields"]):
        raise ContractError("RDA district crosswalk differs from the reviewed subject or statistic binding")
    return pa.table({field["field"]: columns[field["code"]] for field in mapping["fields"]})


def _state_projection(text: str, metadata: dict[str, Any]) -> pa.Table:
    heading = "1. SPED STAAR 3-8 Passing Rate"
    if not all(marker in text for marker in (
        "2025 Results Driven Accountability", "State Report", "Special Education", heading,
        "Indicator", "Rate", "Numerator", "Denominator",
    )) or text.count(heading) != 1:
        raise ContractError("RDA state page lacks its reviewed edition, area or table headings")
    section = text.split(heading, 1)[1].split("2. SPED Year-After-Exit", 1)[0]
    rows = [re.split(r" {2,}", line.strip()) for line in section.splitlines() if line.strip()]
    expected_fields = metadata["fields"]
    if len(rows) != len(expected_fields):
        raise ContractError("RDA state Indicator 1 row count changed")
    result = []
    for ordinal, row, field in zip(("i", "ii", "iii", "iv"), rows, expected_fields, strict=True):
        expected = field["direct_texas"]
        if len(row) != 10 or row[0] != f"({ordinal}) {field['subject']}":
            raise ContractError("RDA state Indicator 1 subject or report layout changed")
        rate, numerator, denominator = row[2:5]
        if (rate != expected["rate_percent"] or not _STATE_NUMBER.fullmatch(numerator)
                or not _STATE_NUMBER.fullmatch(denominator)
                or int(numerator.replace(",", "")) != expected["numerator"]
                or int(denominator.replace(",", "")) != expected["denominator"]):
            raise ContractError("RDA state Indicator 1 values differ from the reviewed direct Texas report")
        result.append(dict(zip(_STATE_COLUMNS, (field["subject"], rate, numerator, denominator), strict=True)))
    schema = pa.schema([(name, pa.string()) for name in _STATE_COLUMNS])
    return pa.Table.from_pylist(result, schema=schema)


def materialize_rda_special_education_indicator1(store: ProvenanceStore, output: Path) -> dict[str, Any]:
    """Build a local candidate in a new directory from retained publisher receipts."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("RDA candidate requires a new exclusive output directory")
    metadata = rda_indicator1_metadata(_YEAR, _AREA)
    config_bytes = _CONFIG.read_bytes()
    cfg = load_config_bytes(config_bytes, path=_CONFIG)
    source = cfg.resolve_dataset(NATIVE_DATASET, _YEAR, reader_profile="source", exact=True)
    crosswalk = cfg.column_contract(NATIVE_DATASET, _YEAR, reader_profile="source")
    if crosswalk is None or not cfg.is_source_only(NATIVE_DATASET, _YEAR):
        raise ContractError("RDA native build requires the packaged source-only crosswalk")
    document = parse_column_schema(crosswalk["text"].encode())
    mappings = crosswalk_dataset(document, NATIVE_DATASET, str(_YEAR))["tables"]
    table_mappings = {table["table"]: table for table in mappings}
    if set(table_mappings) != set(_KEYS):
        raise ContractError("RDA crosswalk must describe both reviewed source scopes")
    district_source, state_source = metadata["district_csv"], metadata["references"]["state_report"]
    refs = [*metadata["references"].values(), *metadata["district_report_witnesses"].values()]
    configured_refs = {ref.reference_id: ref for ref in source.metadata.references}
    for name, ref in metadata["references"].items():
        selected = configured_refs.get(name)
        if (selected is None or selected.url != ref["url"] or selected.reviewed_sha256 != ref["sha256"]
                or selected.provenance_ids != (ref["receipt"],)):
            raise ContractError("RDA configuration differs from the reviewed publisher references")
    roots = list(dict.fromkeys([
        *(reference["receipt"] for reference in [district_source, *refs]),
        *(identifier for ref in source.metadata.references for identifier in ref.provenance_ids),
    ]))
    store.verify_record_closure(roots)
    validate_reference_records(source, store.read_record)
    district_path, state_path = (_checked_source(store, item) for item in (district_source, state_source))
    pdftotext = which("pdftotext")
    if pdftotext is None:
        raise ContractError("RDA state extraction requires Poppler pdftotext")
    page, = {item["direct_texas"]["report_page"] for item in metadata["fields"]}
    text = check_output([pdftotext, "-layout", "-enc", "UTF-8", "-f", str(page), "-l", str(page),
                         str(state_path), "-"], encoding="utf-8", timeout=15)
    table_by_scope = {"district": _district_projection(district_path, table_mappings["district"], metadata),
                      "state": _state_projection(text, metadata)}
    source_by_scope = {"district": district_source, "state": state_source}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".rda-indicator1-", dir=output.parent) as directory:
        stage = Path(directory) / "bundle"
        stage.mkdir(mode=0o700)
        target = ProvenanceStore(stage / "provenance")
        for identifier in roots:
            target.copy_record_from(store, identifier)
        captured_config = Path(directory) / "teadata_sources.yaml"
        captured_config.write_bytes(config_bytes)
        config_id = target.record_acquisition(captured_config, ResolvedDataset(
            f"reference:{NATIVE_DATASET}:config", "reference", _YEAR, _YEAR, "",
            DatasetMetadata(title="Exact packaged RDA source configuration", version=str(_YEAR)),
        ))
        recorder = BuildRecorder(cfg, target)
        toolchain = {"python": sys.version, "pyarrow": version("pyarrow"), "platform": platform.platform(),
                     "architecture": platform.machine()}
        reviewed_selection = sha256(canonical_json(metadata).encode()).hexdigest()
        results = []
        for table_mapping in mappings:
            scope = table_mapping["table"]
            table, original, key_column = table_by_scope[scope], source_by_scope[scope], _KEYS[scope]
            revision = original["sha256"]
            if (revision != table_mapping["input_sha256"]
                    or table.column_names != [field["field"] for field in table_mapping["fields"]]):
                raise ContractError("RDA native leaf differs from the reviewed input or columns")
            key = EditionKey(NATIVE_DATASET, str(_YEAR), EditionKey.source_revision((revision,)),
                             scope, scope, f"one publisher row per {key_column}")
            recipe = {"operation": "rda_indicator1_raw_projection", "area": _AREA,
                      "publisher_test_administration": metadata["test_administration"], "scope": scope,
                      "reviewed_selection_sha256": reviewed_selection}
            if scope == "state":
                recipe.update(pdf_page=page, pdftotext_sha256=file_identity(Path(pdftotext))["sha256"])
            contract = recorder.leaf_contract(key, source, native_schema=table.schema.serialize().to_pybytes(),
                columns=table.column_names, key_columns=[key_column], rows=table.num_rows, inputs=[revision],
                recipe=recipe, toolchain=toolchain)
            contract_json = canonical_json(contract)
            identity = sha256(contract_json.encode()).hexdigest()
            leaf = stage / f"{identity}.parquet"
            physical = write_leaf_artifact(leaf, table.to_batches(max_chunksize=512), contract=contract,
                                           batch_bytes=8 * 1024**2)
            if not table.equals(pq.read_table(leaf).replace_schema_metadata(None), check_metadata=True):
                raise ContractError("RDA native readback differs from publisher projection")
            processing = target.record_processing(dataset=NATIVE_DATASET, inputs=[*roots, config_id],
                outputs={"leaf": leaf}, processor="teadata.rda_native", code_version=recorder.processor_sha256,
                parameters={"leaf_contract_sha256": identity, "recipe": recipe, "toolchain": toolchain},
                validation={"rows": table.num_rows,
                    "native_raw_readback_equal": True, "release_activation": False})
            artifact = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name,
                                   contract_json, processing, "leaf")
            recorder.record_edition(key, source, artifacts=[artifact], acquisition_ids=[original["receipt"]])
            results.append({"edition": key.to_dict(), "rows": table.num_rows,
                            "source_receipt": original["receipt"], "processing_record_id": processing})
        catalog = recorder.catalog(schema_version=4)
        manifest = publish_release(stage, catalog, schema_version=4)
        result = {"status": "unpublished_native_rda_indicator1_candidate", "build_id": catalog.build_id,
                  "manifest_sha256": file_identity(manifest)["sha256"], "editions": results}
        (stage / "MATERIALIZATION.json").write_text(canonical_json(result) + "\n", encoding="utf-8")
        _install_candidate(stage, output)
    return result
