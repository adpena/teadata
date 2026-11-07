"""Materialize reviewed measurement years from district and Texas hiring products."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import platform
import re
import sys
from tempfile import TemporaryDirectory
from typing import Any, TypedDict

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
class HiringSourceSpec(TypedDict):
    receipt: str
    sha256: str
    edition: str
    scope: str
    worksheet: str
    url: str
    years: tuple[int, ...]


_SOURCES: dict[str, HiringSourceSpec] = {
    DISTRICT_DATASET: {
        "receipt": "bec3aac823d68bc6600033955fc4b27b94124d8724cfb4932e3c13ed93f92e34",
        "sha256": "87ef6bdd10d6045e663f3a7aaa6ba819e4bf67ab9067404ec540011f793ab1af",
        "edition": "5", "scope": "district", "worksheet": "District Counts", "years": tuple(range(2010, YEAR + 1)),
        "url": "https://tea4avcastro.tea.state.tx.us/oess/edrs/regional-dashboards/downloads/new-hires-by-school-system-view-3.xlsx",
    },
    STATE_DATASET: {
        "receipt": "792c5aa57758dde8613cc7c4a4a460684b45c5826f6167ad002a069e1bec7a7a",
        "sha256": "fa02256f69a594e95a250b80d32bb3a43ecfe65f493857305ac4e54f0e1d77ee",
        "edition": "2026-04", "scope": "state", "worksheet": "New Hires", "years": tuple(range(2016, YEAR + 1)),
        "url": "https://tea4avcastro.tea.state.tx.us/oess/edrs/new-hire-and-attrition-dashboard/downloads/Attrition_New%20Hires_Workbook.xlsx",
    },
}
SCHOOL_YEARS = tuple(sorted(set(_SOURCES[DISTRICT_DATASET]["years"]) & set(_SOURCES[STATE_DATASET]["years"])))
AVAILABLE_SCHOOL_YEARS = tuple(sorted({year for source in _SOURCES.values() for year in source["years"]}))
_EPP = {
    "reference_id": "hiring_epp_comparison",
    "receipt": "7556f3c4f2e144e7c5a3768b0789494f78fa3db75d29727f63bcf3972260fad8",
    "sha256": "2ed577abc10c60f32d093228fbc685c9fbc5584dda1a7635e95907256ca7db98",
    "url": "https://tea4avcastro.tea.state.tx.us/oess/edrs/regional-dashboards/downloads/new-hires-by-school-system-and-epp-views-4-and-5.xlsx",
}
_EPP_HEADERS = ("district", "schyr", "region", "distname", "geotype", "region_epp", "total_new_hires",
                "EPP_assigned", "cert_cat", "region_2", "Year", "region_epp2")


def _selected_years(school_years: Sequence[int], *, available: Sequence[int] = AVAILABLE_SCHOOL_YEARS) -> tuple[int, ...]:
    if (not school_years or any(type(year) is not int or year not in available for year in school_years)
            or len(set(school_years)) != len(school_years)):
        raise ContractError(f"Select distinct supported school-year ending years: {tuple(available)}")
    return tuple(sorted(school_years))


def _table_name(dataset: str, school_year: int) -> str:
    scope = _SOURCES[dataset]["scope"]
    return scope if school_year == YEAR else f"{scope}_{school_year}"


def _read_years(path: Path, dataset: str, school_years: Sequence[int]) -> dict[int, tuple[list[tuple], list[int]]]:
    years = _selected_years(school_years, available=_SOURCES[dataset]["years"])
    selection = _SOURCES[dataset]
    district = dataset == DISTRICT_DATASET
    expected = _DISTRICT_HEADERS if district else _STATE_HEADERS
    selected: dict[int, tuple[list[tuple], list[int]]] = {year: ([], []) for year in years}
    labels = {f"{year - 1}-{year % 100:02d}": year for year in years}
    with path.open("rb") as stream:
        book = load_workbook(stream, read_only=True, data_only=False, keep_links=False)
        try:
            source = iter(book[selection["worksheet"]].values)
            if next(source) != expected:
                raise ContractError("Educator hiring worksheet headers differ from the reviewed source")
            for ordinal, row in enumerate(source, 2):
                year = row[1] if district else labels.get(row[0])
                if year in selected:
                    if len(row) != len(expected):
                        raise ContractError("Educator hiring source row width changed")
                    rows, locators = selected[year]
                    rows.append(row)
                    locators.append(ordinal)
        finally:
            book.close()
    if any(not rows or (not district and len(rows) != 1) for rows, _ in selected.values()):
        raise ContractError("Educator hiring source lacks an exact selected school-year population")
    return selected


def _read_rows(path: Path, dataset: str, *, school_year: int = YEAR) -> tuple[list[tuple], list[int]]:
    return _read_years(path, dataset, (school_year,))[school_year]


def _projection(rows: list[tuple], locators: list[int], dataset: str, *,
                school_year: int = YEAR) -> tuple[pa.Table, dict[str, Any]]:
    _selected_years((school_year,), available=_SOURCES[dataset]["years"])
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
                    or identifier == "000000" or type(year) is not int or year != school_year
                    or not isinstance(region, str) or re.fullmatch(r"0[1-9]|1[0-9]|20", region) is None
                    or category not in _CATEGORY_LABELS or label != f"{school_year - 1}-{school_year}"
                    or key in seen):
                raise ContractError("Educator hiring district key, category or measurement year changed")
            seen.add(key)
        elif row[0] != f"{school_year - 1}-{school_year % 100:02d}":
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
            if data_type == pa.string() and any(not isinstance(value, str)
                                               or (not value and name != "district_name") for value in values):
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


def _epp_reference(source: ResolvedDataset, store: ProvenanceStore) -> Path:
    references = [ref for ref in source.metadata.references if ref.reference_id == _EPP["reference_id"]]
    if (len(references) != 1 or references[0].role != "source_notes" or references[0].url != _EPP["url"]
            or references[0].reviewed_sha256 != _EPP["sha256"] or references[0].provenance_ids != (_EPP["receipt"],)):
        raise ContractError("Educator hiring requires the exact reviewed EPP comparison reference")
    record = store.read_record(_EPP["receipt"])
    if (record.get("kind") != "acquisition" or record["artifact"]["sha256"] != _EPP["sha256"]
            or record["source_metadata"].get("version") != "5" or record.get("requested_url") != _EPP["url"]
            or record.get("resolved_url") != _EPP["url"]):
        raise ContractError("Educator hiring EPP acquisition differs from the reviewed publisher product")
    return store.verify_artifact(_EPP["sha256"])


def _epp_counts(path: Path, years: Sequence[int]) -> Counter[tuple[int, str, str]]:
    selected = _selected_years(years, available=_SOURCES[DISTRICT_DATASET]["years"])
    counts: Counter[tuple[int, str, str]] = Counter()
    seen = set()
    with path.open("rb") as stream:
        book = load_workbook(stream, read_only=True, data_only=False, keep_links=False)
        try:
            rows = iter(book["District Counts Long"].values)
            if next(rows) != _EPP_HEADERS:
                raise ContractError("Educator hiring EPP headers differ from the reviewed source")
            for row in rows:
                district, year, region, _, _, epp_region, count, epp, category, _, label, _ = row
                if year not in selected:
                    continue
                key = year, district, category, epp_region, epp
                if (type(year) is not int or not isinstance(district, str)
                        or re.fullmatch(r"[0-9]{6}", district) is None or district == "000000"
                        or not isinstance(region, str) or re.fullmatch(r"0[1-9]|1[0-9]|20", region) is None
                        or category not in _CATEGORY_LABELS or not isinstance(epp_region, str) or not epp_region
                        or not isinstance(epp, str) or not epp or key in seen or label != f"{year - 1}-{year}"
                        or type(count) is not int or count < 0):
                    raise ContractError("Educator hiring EPP key, count or population changed")
                seen.add(key)
                counts[year, district, category] += count
        finally:
            book.close()
    return counts


def _qualify_epp(district: pa.Table, expected: Counter[tuple[int, str, str]], year: int) -> dict[str, Any]:
    observed = {(year, row["district_number"], row["certification_at_hire_category"]): row["new_hires_count"]
                for row in district.select(["district_number", "certification_at_hire_category", "new_hires_count"]).to_pylist()}
    selected = {key: value for key, value in expected.items() if key[0] == year}
    if observed.keys() != selected.keys() or observed != selected:
        raise ContractError("District hiring keys or counts differ from the EPP-disaggregated publisher report")
    return {"district_rows": district.num_rows, "districts": len(set(district["district_number"].to_pylist())),
            "epp_comparison_keys": len(observed), "epp_comparison_exact": True,
            "comparison_reference_id": _EPP["reference_id"], "direct_state_comparison": "unavailable"}


def materialize_educator_hiring(store: ProvenanceStore, output: Path, *,
                               school_years: Sequence[int] = (YEAR,)) -> dict[str, Any]:
    """Build selected paired source years offline in a new exclusive directory."""
    years = _selected_years(school_years)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Educator hiring candidate requires a new exclusive output directory")
    raw_config = _CONFIG.read_bytes()
    config = load_config_bytes(raw_config, path=_CONFIG)
    sources, projections = {}, {}
    roots = []
    for dataset, selection in _SOURCES.items():
        source_years = tuple(year for year in years if year in selection["years"])
        if not source_years:
            continue
        source = config.resolve_dataset(dataset, YEAR, reader_profile="source", exact=True)
        options = reader_options(source.reader, "source")
        names = _DISTRICT_FIELDS if dataset == DISTRICT_DATASET else _STATE_FIELDS
        metrics = {name: MetricDefinition(name, unit="ratio" if name.endswith("_ratio") else "count")
                   for name in names if name.endswith(("_count", "_ratio"))}
        if (source.metadata.version != selection["edition"] or source.metadata.period != DataPeriod()
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
        mappings = crosswalk_dataset(parse_column_schema(resource["text"].encode()), dataset, selection["edition"])["tables"]
        headers = _DISTRICT_HEADERS if dataset == DISTRICT_DATASET else _STATE_HEADERS
        if ({mapping["table"] for mapping in mappings} != {_table_name(dataset, year) for year in selection["years"]}
                or len(mappings) != len(selection["years"])):
            raise ContractError("Educator hiring crosswalk differs from the reviewed measurement years")
        by_table = {mapping["table"]: mapping for mapping in mappings}
        if any(mapping["input_sha256"] != selection["sha256"]
               or mapping["metric_period"] != DataPeriod("school_year", year - 1, year).to_dict()
               or [field["field"] for field in mapping["fields"]] != list(names)
               or [field["code"] for field in mapping["fields"]] != list(headers)
               for year in selection["years"] for mapping in (by_table[_table_name(dataset, year)],)):
            raise ContractError("Educator hiring crosswalk differs from the reviewed fields and headers")
        for year, (rows, locators) in _read_years(store.verify_artifact(selection["sha256"]), dataset, source_years).items():
            projections[dataset, year] = _projection(rows, locators, dataset, school_year=year)
        sources[dataset] = source
        roots.extend([selection["receipt"], *reference_ids])
    earlier = tuple(year for year in years if year not in _SOURCES[STATE_DATASET]["years"])
    expected = _epp_counts(_epp_reference(sources[DISTRICT_DATASET], store), earlier) if earlier else Counter()
    qualifications = {
        year: (_qualify_epp(projections[DISTRICT_DATASET, year][0], expected, year) if year in earlier else
               _qualify(projections[DISTRICT_DATASET, year][0], projections[STATE_DATASET, year][0])) for year in years
    }
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
        for (dataset, year), (table, support) in projections.items():
            selection = _SOURCES[dataset]
            source = sources[dataset]
            keys = ["district_number", "school_year_end", "certification_at_hire_category"] if dataset == DISTRICT_DATASET else []
            grain = "one publisher row per " + " and ".join(keys) if keys else STATEWIDE_SINGLETON_GRAIN
            key = EditionKey(dataset, selection["edition"], EditionKey.source_revision((selection["sha256"],)),
                             selection["scope"], _table_name(dataset, year), grain)
            recipe = {"operation": "educator_hiring", "worksheet": selection["worksheet"],
                      "school_year_end": year, "category_mapping": CATEGORIES}
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
                validation={**qualifications[year], "school_year_end": year,
                            "rows": table.num_rows, "native_readback_equal": True,
                            "release_activation": False})
            artifact = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name,
                                   contract_json, processing, "leaf")
            recorder.record_edition(key, source, artifacts=[artifact], acquisition_ids=[selection["receipt"]])
            results.append({"edition": key.to_dict(), "school_year_end": year,
                            "rows": table.num_rows, "processing_record_id": processing})
        catalog = recorder.catalog(schema_version=4)
        manifest = publish_release(stage, catalog, schema_version=4)
        result = {"status": "unpublished_native_educator_hiring", "build_id": catalog.build_id,
                  "manifest_sha256": file_identity(manifest)["sha256"],
                  "qualification": {"school_years": list(years), "by_school_year": qualifications},
                  "school_years_by_scope": {selection["scope"]: [year for year in years if year in selection["years"]]
                                             for selection in _SOURCES.values()},
                  "unavailable_school_years_by_scope": {selection["scope"]: [year for year in years if year not in selection["years"]]
                                                         for selection in _SOURCES.values()},
                  "editions": results, "release_activation": False}
        (stage / "MATERIALIZATION.json").write_text(canonical_json(result) + "\n", encoding="utf-8")
        _install_candidate(stage, output)
    return result
