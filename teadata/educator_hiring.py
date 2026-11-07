"""Materialize the reviewed 2025–26 district and Texas teacher-hiring products."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import platform
import re
import sys
from tempfile import TemporaryDirectory
from typing import Any

from openpyxl import load_workbook
import pyarrow as pa
import pyarrow.parquet as pq

from .analytical import write_leaf_artifact
from .column_schema import crosswalk_dataset, parse_column_schema, reader_options
from .dataset_metadata import (
    STATEWIDE_SINGLETON_GRAIN, ArtifactRef, DataPeriod, DatasetMetadata, EditionKey, MetricDefinition, ResolvedDataset,
)
from .manifest import BuildRecorder, ContractError, canonical_json, file_identity
from .provenance import ProvenanceStore, validate_reference_records
from .release import _install_candidate, publish_release
from .teadata_config import load_config_bytes
from .values import ValueStatus

DISTRICT_DATASET = "educator_district_new_hires"
STATE_DATASET = "educator_state_new_hires"
YEAR = 2026
PERIOD = DataPeriod("school_year", YEAR - 1, YEAR)
SOURCE_ROW = "publisher_worksheet_row"
_CONFIG = Path(__file__).with_name("teadata_sources.yaml")

# Reviewed against both publisher dictionaries and their actual worksheet headers.
# The district category is an entry pathway or certification at hire.
CATEGORIES = (
    ("Intern Certified", "Intern Certified", "intern_certified"),
    ("One Year OOS Certified", "One Year, Out-of-State Certified", "one_year_out_of_state_certified"),
    ("Standard OOS Certified", "Standard, Out-of-State Certified", "standard_out_of_state_certified"),
    ("Standard Direct Entry", "Standard Certified, Direct Entry", "standard_certified_direct_entry"),
    ("Standard Lagged Entry", "Standard Certified, Lagged Entry", "standard_certified_lagged_entry"),
    ("No TX Cert Permit", "No Texas Certification/Permit", "no_texas_certification_or_permit"),
    ("Emergency Permits", "Emergency Permit", "emergency_permit"),
    ("Reenterers", "Re-enterer", "reenterer"),
    ("Previously Part Time", "Previously Part-Time", "previously_part_time"),
)
_CATEGORY_LABELS = frozenset(row[1] for row in CATEGORIES)
_DISTRICT_HEADERS = ("district", "schyr", "region", "cert_cat", "count", "distname", "geotypex", "region_2", "Year")
_DISTRICT_FIELDS = ("district_number", "school_year_end", "education_service_center",
                    "certification_at_hire_category", "new_hires_count", "district_name", "district_geotype",
                    "education_service_center_label", "school_year_label")
_STATE_HEADERS = ("Academic Year", "Total New Hires", *(row[0] for row in CATEGORIES),
                  *("Pct " + row[0] for row in CATEGORIES))
_STATE_FIELDS = ("school_year_label", "total_new_hires_count", *(row[2] + "_count" for row in CATEGORIES),
                 *(row[2] + "_ratio" for row in CATEGORIES))
_SOURCES = {
    DISTRICT_DATASET: {
        "receipt": "bec3aac823d68bc6600033955fc4b27b94124d8724cfb4932e3c13ed93f92e34",
        "sha256": "87ef6bdd10d6045e663f3a7aaa6ba819e4bf67ab9067404ec540011f793ab1af",
        "edition": "5", "scope": "district", "worksheet": "District Counts",
        "url": "https://tea4avcastro.tea.state.tx.us/oess/edrs/regional-dashboards/downloads/new-hires-by-school-system-view-3.xlsx",
    },
    STATE_DATASET: {
        "receipt": "792c5aa57758dde8613cc7c4a4a460684b45c5826f6167ad002a069e1bec7a7a",
        "sha256": "fa02256f69a594e95a250b80d32bb3a43ecfe65f493857305ac4e54f0e1d77ee",
        "edition": "2026-04", "scope": "state", "worksheet": "New Hires",
        "url": "https://tea4avcastro.tea.state.tx.us/oess/edrs/new-hire-and-attrition-dashboard/downloads/Attrition_New%20Hires_Workbook.xlsx",
    },
}


def _read_rows(path: Path, dataset: str) -> tuple[list[tuple], list[int]]:
    selection = _SOURCES[dataset]
    district = dataset == DISTRICT_DATASET
    expected = _DISTRICT_HEADERS if district else _STATE_HEADERS
    rows, locators = [], []
    with path.open("rb") as stream:
        book = load_workbook(stream, read_only=True, data_only=False, keep_links=False)
        try:
            source = iter(book[selection["worksheet"]].values)
            if next(source) != expected:
                raise ContractError("Educator hiring worksheet headers differ from the reviewed source")
            for ordinal, row in enumerate(source, 2):
                if (row[1] == YEAR if district else row[0] == "2025-26"):
                    if len(row) != len(expected):
                        raise ContractError("Educator hiring source row width changed")
                    rows.append(row)
                    locators.append(ordinal)
        finally:
            book.close()
    if not rows or (not district and len(rows) != 1):
        raise ContractError("Educator hiring source lacks the exact 2025–26 population")
    return rows, locators


def _projection(rows: list[tuple], locators: list[int], dataset: str) -> tuple[pa.Table, dict[str, Any]]:
    district = dataset == DISTRICT_DATASET
    names = _DISTRICT_FIELDS if district else _STATE_FIELDS
    columns: dict[str, Any] = {}
    support: dict[str, Any] = {"fields": {}, "context": [SOURCE_ROW]}
    seen = set()
    for row in rows:
        if district:
            identifier, year, region, category, count, _, _, _, label = row
            key = identifier, year, category
            if (not isinstance(identifier, str) or re.fullmatch(r"[0-9]{6}", identifier) is None
                    or identifier == "000000" or type(year) is not int or year != YEAR
                    or not isinstance(region, str) or re.fullmatch(r"0[1-9]|1[0-9]|20", region) is None
                    or category not in _CATEGORY_LABELS or label != "2025-2026"
                    or key in seen):
                raise ContractError("Educator hiring district key, category or measurement year changed")
            seen.add(key)
        elif row[0] != "2025-26":
            raise ContractError("Educator hiring state measurement year changed")
    for index, name in enumerate(names):
        values = [row[index] for row in rows]
        numeric = name.endswith(("_count", "_ratio"))
        if numeric:
            ratio = name.endswith("_ratio")
            if any(type(value) not in ((int, float) if ratio else (int,))
                   or not (0 <= value <= 1 if ratio else value >= 0) for value in values):
                raise ContractError(f"Educator hiring has an unreviewed numeric or disclosure token in {name}")
            columns[name] = pa.array(values, type=pa.float64() if ratio else pa.int64())
            raw, status = f"raw_{name}", f"status_{name}"
            columns[raw] = pa.array([str(value) for value in values], type=pa.string())
            columns[status] = pa.repeat(pa.scalar(ValueStatus.OBSERVED.value), len(rows))
            support["fields"][name] = {"raw": raw, "status": status}
        else:
            data_type = pa.int16() if name == "school_year_end" else pa.string()
            if data_type == pa.string() and any(not isinstance(value, str) or not value for value in values):
                raise ContractError(f"Educator hiring has an unreviewed identity or label token in {name}")
            columns[name] = pa.array(values, type=data_type)
    columns[SOURCE_ROW] = pa.array(locators, type=pa.int32())
    return pa.table(columns), support


def _qualify(district: pa.Table, state: pa.Table) -> dict[str, Any]:
    counts = Counter()
    for category, count in zip(district["certification_at_hire_category"].to_pylist(),
                               district["new_hires_count"].to_pylist(), strict=True):
        counts[category] += count
    totals = state.to_pylist()[0]
    total = totals["total_new_hires_count"]
    if total <= 0 or sum(counts.values()) != total:
        raise ContractError("Educator hiring district total differs from the direct Texas report")
    for _, category, name in CATEGORIES:
        count = totals[name + "_count"]
        if counts[category] != count or totals[name + "_ratio"] != count / total:
            raise ContractError(f"Educator hiring category or published ratio differs for {category}")
    return {"district_rows": district.num_rows, "districts": len(set(district["district_number"].to_pylist())),
            "state_total_new_hires": total, "matched_categories": len(CATEGORIES),
            "state_ratios_exactly_match_counts": True}


def materialize_educator_hiring(store: ProvenanceStore, output: Path) -> dict[str, Any]:
    """Build the exact two current-year source editions offline in a new directory."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("Educator hiring candidate requires a new exclusive output directory")
    raw_config = _CONFIG.read_bytes()
    config = load_config_bytes(raw_config, path=_CONFIG)
    sources, projections = {}, {}
    roots = []
    for dataset, selection in _SOURCES.items():
        source = config.resolve_dataset(dataset, YEAR, reader_profile="source", exact=True)
        options = reader_options(source.reader, "source")
        names = _DISTRICT_FIELDS if dataset == DISTRICT_DATASET else _STATE_FIELDS
        metrics = {name: MetricDefinition(name, unit="ratio" if name.endswith("_ratio") else "count")
                   for name in names if name.endswith(("_count", "_ratio"))}
        if (source.metadata.version != selection["edition"] or source.metadata.period != PERIOD
                or source.metadata.source_url != selection["url"] or source.metadata.metrics != metrics
                or not config.is_source_only(dataset, YEAR) or options["acquisition_id"] != selection["receipt"]
                or options["worksheet"] != selection["worksheet"]):
            raise ContractError("Educator hiring configuration differs from the reviewed edition")
        receipt = store.read_record(selection["receipt"])
        if (receipt.get("kind") != "acquisition" or receipt["artifact"]["sha256"] != selection["sha256"]
                or receipt["source_metadata"].get("version") != selection["edition"]
                or receipt.get("requested_url") != selection["url"] or receipt.get("resolved_url") != selection["url"]):
            raise ContractError("Educator hiring receipt differs from the reviewed publisher bytes or product")
        reference_ids = [identifier for ref in source.metadata.references for identifier in ref.provenance_ids]
        store.verify_record_closure([selection["receipt"], *reference_ids])
        validate_reference_records(source, store.read_record)
        resource = config.column_contract(dataset, YEAR, reader_profile="source")
        if resource is None:
            raise ContractError("Educator hiring requires its reviewed source crosswalk")
        mapping, = crosswalk_dataset(parse_column_schema(resource["text"].encode()), dataset, selection["edition"])["tables"]
        headers = _DISTRICT_HEADERS if dataset == DISTRICT_DATASET else _STATE_HEADERS
        if (mapping["table"] != selection["scope"] or mapping["input_sha256"] != selection["sha256"]
                or [field["field"] for field in mapping["fields"]] != list(names)
                or [field["code"] for field in mapping["fields"]] != list(headers)):
            raise ContractError("Educator hiring crosswalk differs from the reviewed fields and headers")
        rows, locators = _read_rows(store.verify_artifact(selection["sha256"]), dataset)
        projections[dataset] = _projection(rows, locators, dataset)
        sources[dataset] = source
        roots.extend([selection["receipt"], *reference_ids])
    qualification = _qualify(projections[DISTRICT_DATASET][0], projections[STATE_DATASET][0])
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".educator-hiring-", dir=output.parent) as directory:
        stage = Path(directory) / "bundle"
        stage.mkdir(mode=0o700)
        target = ProvenanceStore(stage / "provenance")
        for identifier in dict.fromkeys(roots):
            target.copy_record_from(store, identifier)
        captured = Path(directory) / "teadata_sources.yaml"
        captured.write_bytes(raw_config)
        config_id = target.record_acquisition(captured, ResolvedDataset("reference:educator_hiring:config", "reference",
            YEAR, YEAR, "", DatasetMetadata(title="Exact educator hiring source configuration", version=str(YEAR))))
        recorder = BuildRecorder(config, target)
        toolchain = {"python": sys.version, "pyarrow": version("pyarrow"), "openpyxl": version("openpyxl"),
                     "platform": platform.platform(), "architecture": platform.machine()}
        results = []
        for dataset, selection in _SOURCES.items():
            source = sources[dataset]
            table, support = projections[dataset]
            keys = ["district_number", "school_year_end", "certification_at_hire_category"] if dataset == DISTRICT_DATASET else []
            grain = "one publisher row per " + " and ".join(keys) if keys else STATEWIDE_SINGLETON_GRAIN
            key = EditionKey(dataset, selection["edition"], EditionKey.source_revision((selection["sha256"],)),
                             selection["scope"], selection["scope"], grain)
            recipe = {"operation": "educator_hiring_2026", "worksheet": selection["worksheet"],
                      "school_year_end": YEAR, "category_mapping": CATEGORIES}
            contract = recorder.leaf_contract(key, source, native_schema=table.schema.serialize().to_pybytes(),
                columns=table.column_names, key_columns=keys, rows=table.num_rows, inputs=[selection["sha256"]],
                recipe=recipe, toolchain=toolchain, column_support=support)
            contract_json = canonical_json(contract)
            identity = sha256(contract_json.encode()).hexdigest()
            leaf = stage / f"{identity}.parquet"
            physical = write_leaf_artifact(leaf, table.to_batches(max_chunksize=512), contract=contract, batch_bytes=8 * 1024**2)
            if not table.equals(pq.read_table(leaf).replace_schema_metadata(None), check_metadata=True):
                raise ContractError("Educator hiring native readback differs from publisher projection")
            processing = target.record_processing(dataset=dataset, inputs=[*dict.fromkeys(roots), config_id],
                outputs={"leaf": leaf}, processor="teadata.educator_hiring", code_version=recorder.processor_sha256,
                parameters={"leaf_contract_sha256": identity, "recipe": recipe, "toolchain": toolchain},
                validation={**qualification, "rows": table.num_rows, "native_readback_equal": True,
                            "release_activation": False})
            artifact = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name,
                                   contract_json, processing, "leaf")
            recorder.record_edition(key, source, artifacts=[artifact], acquisition_ids=[selection["receipt"]])
            results.append({"edition": key.to_dict(), "rows": table.num_rows, "processing_record_id": processing})
        catalog = recorder.catalog(schema_version=4)
        manifest = publish_release(stage, catalog, schema_version=4)
        result = {"status": "unpublished_native_educator_hiring", "build_id": catalog.build_id,
                  "manifest_sha256": file_identity(manifest)["sha256"],
                  "qualification": qualification, "editions": results, "release_activation": False}
        (stage / "MATERIALIZATION.json").write_text(canonical_json(result) + "\n", encoding="utf-8")
        _install_candidate(stage, output)
    return result
