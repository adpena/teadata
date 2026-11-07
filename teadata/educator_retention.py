"""Build qualified cohort editions from TEA's statewide retention workbook."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from hashlib import sha256
from importlib.metadata import version
from math import isclose
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
from typing import Any, NamedTuple

from openpyxl import load_workbook
import pyarrow as pa
import pyarrow.parquet as pq

from .analytical import write_leaf_artifact
from .column_schema import crosswalk_dataset, parse_column_schema, reader_options
from .dataset_metadata import ArtifactRef, DataPeriod, DatasetMetadata, EditionKey, MetricDefinition, ResolvedDataset
from .manifest import BuildRecorder, ContractError, canonical_json, file_identity
from .provenance import ProvenanceStore, validate_reference_records
from .release import _install_candidate, publish_release
from .teadata_config import Config, load_config_bytes
from .values import ValueStatus

REPORT_YEAR = 2026
COHORT_YEARS = tuple(range(2014, 2026))
EDITION = "2026-04"
WORKBOOK_RECEIPT = "4b225a3078c31dab9a072742d390be2af2dbb13af3347f81248b96bf5748c928"
WORKBOOK_SHA = "893883bcf8960da35a06d8dcf01016b4bf7816fab093da60b6f29e1c83bd17ac"
WORKBOOK_URL = "https://tea4avcastro.tea.state.tx.us/oess/edrs/teacher-retention-dashboard/downloads/Teacher_Retention_Workbook.xlsx"
RULES_RECEIPT = "7c538cb7ecc13e94a2fa3c91443979d2e8a1ff62b0960c70d3ad0cce3d287ce4"
RULES_SHA = "26a4bebd511f01d15739dc2f616e13cc805a068554400a7d64f7b3fe6506e315"
RULES_URL = "https://tea4avcastro.tea.state.tx.us/oess/edrs/teacher-retention-dashboard/downloads/Business_Rules_Document_Retention.pdf"
VALUE_POLICY = "edrs_retention_2026"
ROUTE_DATASET = "educator_state_retention_by_preparation_route"
DEMOGRAPHIC_DATASET = "educator_state_retention_by_demographic_group"
GENDER_DATASET = "educator_state_retention_by_gender"
HELD_ROUTE_COHORT = 2018
SOURCE_ROW = "publisher_worksheet_row"
KEYS = ("first_teaching_school_year", "publisher_group")
FIELDS = (*KEYS, "first_year_teacher_count", *(f"retained_year_{n}_{kind}"
          for n in range(1, 6) for kind in ("count", "ratio")))
METRICS = {name: MetricDefinition(name, unit=name.rsplit("_", 1)[-1]) for name in FIELDS[2:]}
_CONFIG = Path(__file__).with_name("teadata_sources.yaml")


class _View(NamedTuple):
    sheet: str
    identity_headers: tuple[str, str]
    total: str
    groups: tuple[str, ...]

    @property
    def headers(self) -> tuple[str, ...]:
        return (*self.identity_headers, "Total Teaching in First Year",
                *(label for n in range(1, 6) for label in (f"Total Retained Year {n}", f"Pct Retained Year {n}")))


VIEWS = {
    ROUTE_DATASET: _View("Retention by Preparation Route", ("First Year of Teaching", "Preparation Route"), "All routes",
                         ("Alternative", "Out of State", "Post-baccalaureate", "Undergraduate")),
    DEMOGRAPHIC_DATASET: _View("Retention by Demographic Group", ("First Teaching Year", "Race/Ethnicity"), "All Groups",
                               ("Black / African American", "Hispanic / Latino", "Other", "White")),
    GENDER_DATASET: _View("Retention by Gender", ("First Teaching Year", "Gender"), "All Groups", ("Female", "Male")),
}
_COHORT_LABELS = {f"{year - 1}-{year % 100:02d}": year for year in COHORT_YEARS}
_Populations = dict[str, dict[int, dict[str, tuple[int, tuple]]]]


def _admitted_years(dataset: str) -> tuple[int, ...]:
    return tuple(year for year in COHORT_YEARS if (dataset, year) != (ROUTE_DATASET, HELD_ROUTE_COHORT))


def _validate_rows(rows: Iterable[tuple], view: _View) -> dict[int, dict[str, tuple[int, tuple]]]:
    iterator = iter(rows)
    if next(iterator, None) != view.headers:
        raise ContractError(f"Retention worksheet headers differ for {view.sheet}")
    by_year: dict[int, dict[str, tuple[int, tuple]]] = {year: {} for year in COHORT_YEARS}
    allowed = {view.total, *view.groups}
    for number, row in enumerate(iterator, 2):
        if len(row) != len(FIELDS):
            raise ContractError(f"Retention row width changed in {view.sheet} at row {number}")
        label, group, initial, *followup = row
        if not isinstance(label, str) or label not in _COHORT_LABELS or not isinstance(group, str) or group not in allowed:
            raise ContractError(f"Retention cohort or group changed in {view.sheet} at row {number}")
        year = _COHORT_LABELS[label]
        if group in by_year[year] or type(initial) is not int or initial <= 0:
            raise ContractError(f"Retention duplicate group or invalid first-year count at row {number}")
        previous = initial
        for horizon, (count, ratio) in enumerate(zip(followup[::2], followup[1::2], strict=True), 1):
            if year + horizon > REPORT_YEAR:
                if count is not None or ratio is not None:
                    raise ContractError(f"Retention has a value beyond the reporting horizon at row {number}")
            else:
                if (type(count) is not int or not 0 <= count <= previous or type(ratio) not in (int, float)
                        or not 0 <= ratio <= 1 or not isclose(ratio, count / initial, rel_tol=0, abs_tol=2e-15)):
                    raise ContractError(f"Retention count or ratio conflicts at row {number}, horizon {horizon}")
                previous = count
        by_year[year][group] = number, row
    for year, groups in by_year.items():
        if groups.keys() != allowed:
            raise ContractError(f"Retention group roster differs in {view.sheet} for cohort {year}")
        total = groups[view.total][1]
        for column in (2, *range(3, len(FIELDS), 2)):
            count = total[column]
            if count is not None and sum(groups[group][1][column] for group in view.groups) != count:
                raise ContractError(f"Retention group counts do not match the direct total in {view.sheet} for {year}")
    return by_year


def _read(path: Path) -> _Populations:
    with path.open("rb") as file:
        book = load_workbook(file, read_only=True, data_only=False, keep_links=False)
        try:
            populations = {dataset: _validate_rows(book[view.sheet].values, view) for dataset, view in VIEWS.items()}
        finally:
            book.close()
    for year in COHORT_YEARS:
        totals = {populations[dataset][year][view.total][1][2:] for dataset, view in VIEWS.items()}
        if len(totals) != 1:
            raise ContractError(f"Direct statewide retention totals differ across tabs for cohort {year}")
    return populations


def _project(groups: dict[str, tuple[int, tuple]]) -> tuple[pa.Table, dict[str, Any]]:
    locators, rows = zip(*groups.values(), strict=True)
    columns = {name: pa.array([row[index] for row in rows], type=pa.string()) for index, name in enumerate(KEYS)}
    columns[SOURCE_ROW] = pa.array(locators, type=pa.int32())
    support: dict[str, Any] = {"fields": {}, "context": [SOURCE_ROW]}
    for index, name in enumerate(FIELDS[2:], 2):
        values = [row[index] for row in rows]
        columns[name] = pa.array(values, type=pa.float64() if name.endswith("_ratio") else pa.int64())
        raw, status = f"raw_{name}", f"status_{name}"
        columns[raw] = pa.array(["" if value is None else str(value) for value in values], type=pa.string())
        columns[status] = pa.array([ValueStatus.MISSING.value if value is None else ValueStatus.OBSERVED.value
                                    for value in values], type=pa.string())
        support["fields"][name] = {"raw": raw, "status": status}
    return pa.table(columns), support


def _admit(store: ProvenanceStore, config: Config) -> tuple[dict[str, ResolvedDataset], set[str]]:
    record = store.read_record(WORKBOOK_RECEIPT)
    if (record.get("kind") != "acquisition" or record["artifact"]["sha256"] != WORKBOOK_SHA
            or record.get("source_metadata", {}).get("version") != EDITION
            or record.get("requested_url") != WORKBOOK_URL or record.get("resolved_url") != WORKBOOK_URL):
        raise ContractError("Retention receipt differs from the reviewed TEA workbook")
    policy = config.value_policies.get(VALUE_POLICY)
    if (policy is None or policy.reference_id != "retention_workbook" or set(policy.tokens) != {""}
            or policy.tokens[""].status != ValueStatus.MISSING):
        raise ContractError("Retention requires its reviewed blank-only value policy")
    sources, roots = {}, {WORKBOOK_RECEIPT, RULES_RECEIPT}
    for dataset, view in VIEWS.items():
        source = config.resolve_dataset(dataset, REPORT_YEAR, reader_profile="source", exact=True)
        options = reader_options(source.reader, "source")
        resource = config.column_contract(dataset, REPORT_YEAR, reader_profile="source")
        if (not config.is_source_only(dataset, REPORT_YEAR) or source.metadata.version != EDITION
                or source.metadata.period != DataPeriod() or source.metadata.source_url != WORKBOOK_URL
                or source.metadata.metrics != METRICS or source.metadata.value_policy != VALUE_POLICY
                or options.get("acquisition_id") != WORKBOOK_RECEIPT or options.get("worksheet") != view.sheet
                or resource is None):
            raise ContractError(f"Retention configuration differs for {dataset}")
        refs = {reference.reference_id: reference for reference in source.metadata.references}
        for role, reference_id, url, digest, receipt in (
            ("data_dictionary", "retention_workbook", WORKBOOK_URL, WORKBOOK_SHA, WORKBOOK_RECEIPT),
            ("methodology", "retention_business_rules", RULES_URL, RULES_SHA, RULES_RECEIPT),
        ):
            ref = refs.get(reference_id)
            if ref is None or (ref.role, ref.url, ref.reviewed_sha256, ref.provenance_ids) != (role, url, digest, (receipt,)):
                raise ContractError(f"Retention lacks exact publisher reference {reference_id}")
        roots.update(identifier for ref in refs.values() for identifier in ref.provenance_ids)
        validate_reference_records(source, store.read_record, verify_all_claimed=True)
        tables = crosswalk_dataset(parse_column_schema(resource["text"].encode()), dataset, EDITION)["tables"]
        years = _admitted_years(dataset)
        if (len(tables) != len(years) or {table["table"] for table in tables} != {f"state_{year}" for year in years}):
            raise ContractError(f"Retention crosswalk edition scope differs for {dataset}")
        by_name = {table["table"]: table for table in tables}
        for year in years:
            mapping = by_name[f"state_{year}"]
            if (mapping["input_sha256"] != WORKBOOK_SHA
                    or mapping["metric_period"] != DataPeriod("school_year", year - 1, year).to_dict()
                    or tuple((field["code"], field["label"], field["field"], field["disposition"])
                             for field in mapping["fields"]) != tuple(
                        (header, header, name, "identity" if name in KEYS else "fact")
                        for header, name in zip(view.headers, FIELDS, strict=True))):
                raise ContractError(f"Retention crosswalk fields or cohort differ for {dataset} {year}")
        sources[dataset] = source
    store.verify_record_closure(sorted(roots))
    return sources, roots


def materialize_educator_retention(store: ProvenanceStore, output: Path, *,
                                  cohort_years: Sequence[int] = COHORT_YEARS) -> dict[str, Any]:
    """Build exact or all eligible first-teaching cohorts from retained bytes."""
    if (not cohort_years or any(type(year) is not int or year not in COHORT_YEARS for year in cohort_years)
            or len(set(cohort_years)) != len(cohort_years)):
        raise ContractError(f"Select distinct supported first-teaching school-year ending years: {COHORT_YEARS}")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Retention candidate requires a new exclusive output directory")
    years = sorted(cohort_years)
    raw_config = _CONFIG.read_bytes()
    config = load_config_bytes(raw_config, path=_CONFIG)
    sources, roots = _admit(store, config)
    populations = _read(store.verify_artifact(WORKBOOK_SHA))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".educator-retention-", dir=output.parent) as directory:
        stage = Path(directory) / "bundle"
        stage.mkdir(mode=0o700)
        target = ProvenanceStore(stage / "provenance")
        for identifier in sorted(roots):
            target.copy_record_from(store, identifier)
        captured = Path(directory) / "teadata_sources.yaml"
        captured.write_bytes(raw_config)
        config_id = target.record_acquisition(captured, ResolvedDataset("reference:educator_retention:config", "reference",
            REPORT_YEAR, REPORT_YEAR, "", DatasetMetadata(title="Exact educator retention source configuration", version=EDITION)))
        recorder = BuildRecorder(config, target)
        toolchain = {"python": sys.version, "pyarrow": version("pyarrow"), "openpyxl": version("openpyxl"),
                     "platform": platform.platform(), "architecture": platform.machine()}
        results = []
        for dataset, source in sources.items():
            view = VIEWS[dataset]
            for year in years:
                if year not in _admitted_years(dataset):
                    continue
                table, support = _project(populations[dataset][year])
                key = EditionKey(dataset, EDITION, EditionKey.source_revision((WORKBOOK_SHA,)), "state", f"state_{year}",
                                 "one publisher row per first teaching school year and reported group")
                recipe = {"operation": "educator_retention", "worksheet": view.sheet, "cohort_end_year": year,
                          "followup": "continuous", "raw_cell_encoding": "decoded_python_scalar_text"}
                contract = recorder.leaf_contract(key, source, native_schema=table.schema.serialize().to_pybytes(),
                    columns=table.column_names, key_columns=list(KEYS), rows=table.num_rows, inputs=[WORKBOOK_SHA],
                    recipe=recipe, toolchain=toolchain, column_support=support)
                contract_json = canonical_json(contract)
                leaf = stage / f"{sha256(contract_json.encode()).hexdigest()}.parquet"
                physical = write_leaf_artifact(leaf, table.to_batches(max_chunksize=512), contract=contract, batch_bytes=8 * 1024**2)
                if not table.equals(pq.read_table(leaf).replace_schema_metadata(None), check_metadata=True):
                    raise ContractError("Retention native readback differs from publisher projection")
                processing = target.record_processing(dataset=dataset, inputs=[*sorted(roots), config_id],
                    outputs={"leaf": leaf}, processor="teadata.educator_retention", code_version=recorder.processor_sha256,
                    parameters={"leaf_contract_sha256": leaf.stem, "recipe": recipe, "toolchain": toolchain},
                    validation={"subgroups_reconcile": True, "direct_totals_match_across_tabs": True,
                                "exact_native_projection": True, "rows": table.num_rows, "release_activation": False})
                artifact = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name,
                                       contract_json, processing, "leaf")
                recorder.record_edition(key, source, artifacts=[artifact], acquisition_ids=[WORKBOOK_RECEIPT])
                results.append({"edition": key.to_dict(), "cohort_end_year": year,
                                "rows": table.num_rows, "processing_record_id": processing})
        catalog = recorder.catalog(schema_version=4)
        manifest = publish_release(stage, catalog, schema_version=4)
        result = {"status": "unpublished_native_educator_retention", "build_id": catalog.build_id,
                  "manifest_sha256": file_identity(manifest)["sha256"], "editions": results,
                  "requested_cohort_end_years": years, "held_route_cohort_end_year": HELD_ROUTE_COHORT,
                  "release_activation": False}
        (stage / "MATERIALIZATION.json").write_text(canonical_json(result) + "\n", encoding="utf-8")
        _install_candidate(stage, output)
    return result
