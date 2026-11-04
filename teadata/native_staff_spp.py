"""Offline qualification of the exact reviewed staffing and evaluation bundles."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
from importlib.resources import files
import json
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from . import spp_leaves
from .dataset_metadata import DataPeriod
from .manifest import ContractError
from .native_witnesses import _BROKER, _Rows
from .peims_staff import STAFF, STAFF_EMPLOYEE_BASIS, STAFF_EMPLOYEE_COUNT, staff_fact_units
from .source_parity import validate_source_parity

if TYPE_CHECKING:
    from .dataset_metadata import ArtifactRef
    from .provenance import ProvenanceStore
    from .release import DataRelease


_STAFF_SELECTOR = "https://rptsvr1.tea.texas.gov/adhocrpt/adpeb.html"
_SPP_HEADER = ["State Performance Plan / Annual Performance Report Indicators",
               "Target", "Rate", "MetTarget", "Rate", "MetState Target"]


def reviewed_staff_spp(manifest_sha256: str) -> tuple[str, int, dict[str, Any]] | None:
    reviews = json.loads(files("teadata").joinpath("reference_cases/native_staff_spp.json").read_text())
    for dataset, review in reviews.items():
        if manifest_sha256 in review["manifests"]:
            return dataset, review["manifests"][manifest_sha256], review["editions"]
    return None


def _read_report(store: ProvenanceStore, ref: ArtifactRef, receipt: str, *, dataset: str, url: str,
                 selector: str, method: str, year: int, controls: list[tuple[str, str]]) -> str:
    record = store.read_record(receipt)
    artifact = record["artifact"]
    parameters = record["parameters"]
    metadata = record["source_metadata"]
    if (record["kind"] != "acquisition" or record["method"] != "download"
            or record["section"] != "reference" or record["dataset"] != dataset
            or (record["requested_year"], record["resolved_year"]) != (year, year)
            or record["requested_url"] != url or record["resolved_url"] != url
            or metadata["source_url"] != selector or metadata["version"] != ref.edition.edition
            or metadata["period"] != DataPeriod("school_year", year - 1, year).to_dict()
            or parameters.get("form_url") != _BROKER or parameters.get("http_method") != method
            or parameters.get("http_status") != 200
            or parameters.get("form_controls") != [list(pair) for pair in controls]
            or record["response_headers"].get("content-type", "").split(";", 1)[0].lower() != "text/html"
            or not 0 < artifact["size_bytes"] <= 1024**2 or artifact["sha256"] in ref.contract["inputs"]):
        raise ContractError("Independent TEA report receipt differs from the reviewed selection")
    path = store.verify_artifact(artifact["sha256"])
    with path.open("rb") as stream:
        data = stream.read(1024**2 + 1)
    if len(data) != artifact["size_bytes"] or sha256(data).hexdigest() != artifact["sha256"]:
        raise ContractError("Independent TEA report bytes differ from its receipt")
    return data.decode("cp1252")


def _staff_report(store: ProvenanceStore, ref: ArtifactRef, table, entity: str, receipt: str) -> dict[str, int]:
    from pyarrow.dataset import field

    key = ref.edition
    year = int(key.edition)
    prefix = [("_service", "marykay"), ("_program", "adhoc.addispatch.sas")]
    sizing = [("linespg", "60"), ("charsln", "120")]
    annual = ("endyear", key.edition[-2:])
    family = [("major", "pe"), ("minor", "b")]
    if key.scope == "state" and entity == "state":
        controls = [*prefix, *family, ("loop", "1"), *sizing, annual, ("selsumm", "ss"), ("key", ""), ("format", "W")]
        method, url, geography, selected = "POST", _BROKER, "Statewide", None
    elif key.scope == "district" and re.fullmatch(r"[0-9]{6}", entity):
        controls = [*prefix, annual, *family, ("format", "w"), ("selsumm", "id"), *sizing,
                    ("grouping", "&grouping."), ("loop", "2"), ("_debug", "0"), ("key", entity)]
        method, url, geography = "GET", f"{_BROKER}?{urlencode(controls)}", f"District_{entity}"
        selected = field("district_number") == entity
    else:
        raise ContractError("Independent staffing witness has unsupported geography")
    content = _read_report(store, ref, receipt, dataset="reference:peims_staff_fte_salary_witness",
                           url=url, selector=_STAFF_SELECTOR, method=method, year=year, controls=controls)
    if (f"{year - 1}-{year} Staff Salaries and FTE Counts -- Excluding ESC Staff" not in content
            or f"Staff Salary FTE Report_{geography}_{year - 1}-{key.edition[-2:]}" not in content
            or (selected is not None and re.search(rf"\({entity}\s*\)", content) is None)):
        raise ContractError("Independent staffing report differs in geography, edition or population")

    # The HTML puts Employee Count before the three payroll fields in 2025–26.
    facts = staff_fact_units(year)
    columns = ([STAFF_EMPLOYEE_COUNT] if year == 2026 else []) + [name for name in facts if name != STAFF_EMPLOYEE_COUNT]
    raw_columns = [f"peims_raw_{name}" for name in columns]
    fields = ["staff_category", "staff_role", *raw_columns]
    if year == 2026:
        fields.append(STAFF_EMPLOYEE_BASIS)
    native = table.to_table(columns=fields, filter=selected).to_pylist()
    actual = []
    category = " "
    for row in _Rows(content).rows:
        if len(row) == len(columns) + 1:
            if not any(row[1:]):
                category = row[0] or " "
            else:
                actual.append((category, row[0], *(token.removeprefix("$").replace(",", "") for token in row[1:])))
    if not actual or actual != [tuple(row[name] for name in ("staff_category", "staff_role", *raw_columns)) for row in native]:
        raise ContractError("Independent staffing roles or raw publisher values differ")
    if year == 2026:
        note = "Total rows reflect distinct employees within each group. Detail rows reflect positions."
        if (note not in content or any(row[STAFF_EMPLOYEE_BASIS] != (
                "distinct_employees" if row["staff_role"].startswith("TOTAL ") else "positions") for row in native)):
            raise ContractError("Independent staffing report differs in employee-count basis")
    return {"rows": len(native), "values": len(native) * (2 + len(columns))}


def _compare_spp_display(workbook: str, published: str, precision: dict[str, int] | None) -> str:
    if workbook == published == "NR" and precision is None:
        return "exact"
    if re.fullmatch(r"(?:[0-9]{1,3})\.[0-9]{1,2}%", published) is None:
        raise ContractError("Independent SPP publisher rate token differs")
    html = published.removesuffix("%")
    if precision is None and workbook == html:
        return "exact"
    if precision == {"workbook": 1, "html": 2} and (
            re.fullmatch(r"[0-9]{1,3}\.[0-9]", workbook) is not None
            and re.fullmatch(r"[0-9]{1,3}\.[0-9]{2}", html) is not None
            and Decimal(html).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP) == Decimal(workbook)):
        return "consistent_reviewed_display_precision"
    raise ContractError("Independent SPP publisher rate differs from the reviewed workbook display")


def _spp_report(store: ProvenanceStore, ref: ArtifactRef, table, witness: dict[str, Any]) -> dict[str, Any]:
    from pyarrow.dataset import field

    entity, receipt = witness["entity"], witness["receipt"]
    key = ref.edition
    year = int(key.edition) - 1
    rate, target = spp_leaves._PERCENT, spp_leaves._TARGET
    native, = table.to_table(columns=["district_name", "esc_region", f"spp_raw_{rate}", f"spp_raw_{target}"],
                             filter=field("district_number") == entity).to_pylist()
    selection = f"{native['district_name']} ({entity})"
    controls = [("_service", "marykay"), ("DistrictSelection", f"['{selection}']"),
                ("_program", "idea.district_reports.sas"), ("_DEBUG", "0"),
                ("year", str(year)[-2:]), ("fmt", "html"), ("search", "district"), ("search_box", "")]
    content = _read_report(store, ref, receipt, dataset="reference:spp_ind11_district_witness",
                           url=_BROKER, selector=spp_leaves.SELECTOR, method="POST", year=year, controls=controls)
    rows = _Rows(content).rows
    index, = (i for i, row in enumerate(rows) if row[:1] == ["11: Child Find"])
    headings, values = rows[:index], rows[index]
    if (headings.count([f"{key.edition} District Profile"]) != 1
            or headings.count([f"(School Year {year - 1}-{str(year)[-2:]})"]) != 1
            or headings.count(["", "State", "LEA"]) != 1 or headings.count(_SPP_HEADER) != 1
            or sum(selection in row for row in headings) != 1
            or not any(f"REGION {native['esc_region']}" in row for row in headings)
            or len(values) != 6 or values[1] != "100.00%" or values[5] != native[f"spp_raw_{target}"]):
        raise ContractError("Independent SPP district report differs in identity or target")
    display = _compare_spp_display(native[f"spp_raw_{rate}"], values[4], witness.get("publisher_rate_precision"))
    return {"values": 2, "workbook_rate": native[f"spp_raw_{rate}"], "report_rate": values[4],
            "reported_target": values[5], "display_comparison": display}


def _spp_parity(release: DataRelease, store: ProvenanceStore, refs: dict[str, ArtifactRef]) -> dict[str, Any]:
    from .column_schema import crosswalk_dataset, parse_column_schema
    from .manifest import canonical_json

    results: list[dict[str, Any]] = []
    for identifier, ref in sorted(refs.items()):
        key = ref.edition
        metadata = release.catalog.metadata(key.dataset, edition=key)
        period = metadata.period
        year = int(key.edition) - 1
        if (period != DataPeriod("school_year", year - 1, year)
                or key.scope != "district" or key.table != "district"):
            raise ContractError("Reviewed SPP release differs in geography or measurement period")
        acquisitions = release.catalog.acquisitions(key.dataset, edition=key)
        ids = [entry["record_id"] for entry in acquisitions]
        hashes = tuple(entry["artifact"]["sha256"] for entry in acquisitions)
        if len(ids) != len(spp_leaves.REGIONS) or len(set(ids)) != len(ids) or list(hashes) != ref.contract["inputs"]:
            raise ContractError("Reviewed SPP release does not bind its ordered region inputs")
        graph = store.verify_record_closure([ref.processing_id])
        processing = graph[ref.processing_id]
        recipe = processing["parameters"]["recipe"]
        if (processing["processor"] != "teadata.spp_leaves.materialize_spp_ind11"
                or recipe["operation"] != "reviewed_spp_ind11_workbook_projection"
                or recipe["regions"] != list(spp_leaves.REGIONS)
                or processing["parameters"]["leaf_contract_sha256"] != ref.identity
                or [item for item in processing["inputs"] if item in ids] != ids):
            raise ContractError("Reviewed SPP release differs from its processing receipt")
        schema, = (reference.reviewed_sha256 for reference in metadata.references if reference.role == "schema")
        if schema is None:
            raise ContractError("Reviewed SPP release lacks a source crosswalk")
        document = parse_column_schema(release.catalog.column_schema_bytes(schema))
        mapping, = crosswalk_dataset(document, key.dataset, key.edition)["tables"]
        if sha256(canonical_json(mapping).encode()).hexdigest() != ref.contract["mapping_sha256"]:
            raise ContractError("Reviewed SPP crosswalk differs from the leaf binding")
        for region, acquisition in zip(spp_leaves.REGIONS, acquisitions, strict=True):
            original = {name: value for name, value in acquisition.items() if name != "record_id"}
            if (graph[acquisition["record_id"]] != original or original["requested_url"] != spp_leaves.workbook_url(year, region)
                    or not 0 < original["artifact"]["size_bytes"] <= spp_leaves._SOURCE_BYTES):
                raise ContractError("Reviewed SPP workbook receipt differs from its region source")
        policy = release.catalog.policy(key.dataset, edition=key)
        expected, _, _ = spp_leaves._project(store, hashes, [item["label"] for item in mapping["fields"]], policy)
        actual = release.analytical_dataset(ref.identity).to_table().replace_schema_metadata(None)
        if not actual.equals(expected, check_metadata=True):
            raise ContractError("SPP native facts, statuses or source locations differ from the retained workbooks")
        results.append({"edition_id": identifier, "edition": key.to_dict(), "status": "PASS_RAW_PARITY",
                        "workbooks": len(ids), "rows": len(actual),
                        "raw_cells": len(actual) * len(mapping["fields"]), "compared_cells": len(actual) * actual.num_columns})
    return {"status": "PASS_RAW_PARITY", "editions": results, "coverage": {
        "requested_editions": len(refs), "checked_editions": len(results), "uncovered_editions": 0, "failed_editions": 0,
        "raw_cells": sum(item["raw_cells"] for item in results), "compared_cells": sum(item["compared_cells"] for item in results)}}


def qualify_staff_spp(release: DataRelease, parity_store: ProvenanceStore, witness_store: ProvenanceStore,
                      plan: tuple[str, int, dict[str, Any]]) -> dict[str, Any]:
    dataset, schema, cases = plan
    catalog = release.catalog.to_dict()
    if catalog["schema_version"] != schema or catalog["editions"].keys() != cases.keys():
        raise ContractError("Candidate does not contain the exact reviewed schema and editions")
    artifacts = [release.catalog.artifact_ref(artifact) for artifact in catalog["artifacts"]]
    refs = {ref.edition.identity: ref for ref in artifacts}
    if (refs.keys() != cases.keys() or len(artifacts) != len(refs)
            or any(ref.edition.dataset != dataset for ref in refs.values())):
        raise ContractError("Candidate does not contain the exact reviewed family and leaves")
    release.verify()
    try:
        parity = (validate_source_parity(release, parity_store, edition_ids=sorted(refs)) if dataset == STAFF
                  else _spp_parity(release, parity_store, refs))
    except FileNotFoundError as error:
        raise ContractError("Retained raw evidence required by the reviewed candidate is unavailable") from error
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ContractError(f"Retained raw evidence failed the reviewed candidate replay: {error}") from error
    result = {"status": "HOLD", "build_id": release.build_id, "manifest_sha256": release.manifest_sha256,
              "structural_integrity": True, "parity": parity, "publisher_witnesses": None}
    if parity["status"] != "PASS_RAW_PARITY":
        return result
    reports = []
    for identifier, ref in sorted(refs.items()):
        witnesses = cases[identifier]
        if not witnesses or len({item["entity"] for item in witnesses}) != len(witnesses):
            raise ContractError("Each edition needs distinct independent publisher witnesses")
        table = release.analytical_dataset(ref.identity)
        for witness in witnesses:
            receipt, entity = witness["receipt"], witness["entity"]
            try:
                values = (_staff_report(witness_store, ref, table, entity, receipt) if dataset == STAFF
                          else _spp_report(witness_store, ref, table, witness))
            except (ContractError, OSError, ValueError, KeyError, TypeError) as error:
                raise ContractError(f"Publisher witness {receipt} failed for {dataset} {entity}: {error}") from error
            reports.append({"edition_id": identifier, "entity": entity, "receipt": receipt, **values})
    return {**result, "status": "PASS_REVIEWED_NATIVE", "publisher_witnesses": {"reports": reports}}
