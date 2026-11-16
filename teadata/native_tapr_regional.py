"""Offline publisher replay for the retained 2024–25 regional TAPR family."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import re
from typing import TYPE_CHECKING, Any

from .dataset_metadata import DataPeriod, EditionKey
from .manifest import ContractError
from .native_witnesses import _Rows
from .source_parity import validate_source_parity

if TYPE_CHECKING:
    from .provenance import ProvenanceStore
    from .release import DataRelease

REVIEWED_MANIFEST = "9c4e189a85b1a12130207e4a3546d437a0704f4833c45225c401b8f88ea1a50c"
_SCOPES = {"region_tapr_student_profile": ("region", "student_profile", 20),
           "region_tapr_staff_profile": ("region", "staff_profile", 20),
           "state_tapr_staff_profile": ("state", "staff_profile", 1)}
_RACES = {"African American": "african_american", "Hispanic": "hispanic", "White": "white",
          "American Indian": "american_indian", "Asian": "asian", "Pacific Islander": "pacific_islander",
          "Two or More Races": "two_or_more_races"}
_STAFF = {"teacher_total_full_time_equiv_count": "fte", "teacher_experience_average": "years",
          "teacher_total_base_salary_average": "currency", "teacher_student_ratio": "ratio",
          "teacher_turnover_ratio": "percent"}


def _number(token: object, unit: str, *, displayed: bool = False) -> Decimal:
    numeric = r"[0-9]+(?:\.[0-9]+)?"
    pattern = numeric
    if displayed:
        pattern = (r"\$[0-9]{1,3}(?:,[0-9]{3})*" if unit == "currency" else
                   r"[0-9]{1,3}(?:,[0-9]{3})*" if unit == "count" else numeric + "%" if unit == "percent" else
                   r"[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?")
    if not isinstance(token, str) or re.fullmatch(pattern, token) is None:
        raise ContractError(f"TAPR regional {unit} control is not an observed numeric token: {token!r}")
    value = Decimal(token.replace(",", "").removeprefix("$").removesuffix("%"))
    if (not value.is_finite() or value < 0 or (unit == "percent" and value > 100)
            or (unit == "count" and value != value.to_integral_value())):
        raise ContractError(f"TAPR regional {unit} control is outside its reviewed domain")
    return value


def _one(rows: list[list[str]], label: str) -> list[str]:
    selected = [row for row in rows if row and row[0] == label]
    if len(selected) != 1:
        raise ContractError(f"TAPR witness requires exactly one {label!r} row; found {len(selected)}")
    return selected[0]


def _between(rows: list[list[str]], start: str, end: str) -> list[list[str]]:
    first = rows.index(_one(rows, start))
    last = next((i for i in range(first + 1, len(rows)) if rows[i] and rows[i][0] == end), None)
    if last is None:
        raise ContractError(f"TAPR witness section {start!r} has no following {end!r}")
    return rows[first + 1:last]


def _cells(rows: list[list[str]], family: str, scope: str, entity: str) -> dict[str, tuple[str, str, int]]:
    heading = "Student" if family == "student_profile" else "Staff"
    identity = ["State"] if scope == "state" else None
    if (len(rows) < 8 or rows[:2] != [["", "Texas Education Agency", ""], [f"2024-25 {heading} Information"]]
            or (identity is not None and rows[2] != identity)
            or (scope == "region" and (len(rows[2]) != 1 or not rows[2][0].startswith(f"REGION {entity}: ")))):
        raise ContractError("TAPR witness report year, family or geography differs")
    result: dict[str, tuple[str, str, int]] = {}
    if family == "student_profile":
        if (rows[4:7] != [["", "Membership", "Enrollment"], ["", "Region", "State", "Region", "State"],
                          ["Student Information", "Count", "Percent", "Count", "Percent", "Count", "Percent", "Count", "Percent"]]):
            raise ContractError("TAPR Student witness population or columns differ")
        ethnic = _between(rows, "Ethnic Distribution", "Sex")
        for population, offset in (("membership", 1), ("enrollment", 5)):
            for label, name in {"Total Students": "all_students", **_RACES, "Special Education": "special_ed"}.items():
                selected = ethnic if label in _RACES else rows
                row = _one(selected, label)
                if len(row) != 9:
                    raise ContractError(f"TAPR Student witness {label!r} columns differ")
                result[f"student_{population}_{name}_count"] = ("count", row[offset], 2025)
                if name != "all_students":
                    result[f"student_{population}_{name}_percent"] = ("percent", row[offset + 1], 2025)
        for label, name in (("Total Mobile Students", "mobility"), ("Total Student Attrition", "attrition")):
            _one(rows, "Mobility (2023-24)" if name == "mobility" else "Student Attrition (2023-24)")
            row = _one(rows, label)
            if len(row) != 9:
                raise ContractError(f"TAPR Student witness {label!r} columns differ")
            for suffix, offset in (("count", 1), ("percent", 2)):
                result[f"student_membership_{name}_all_students_{suffix}"] = (suffix, row[offset], 2024)
        return result
    width = 3 if scope == "state" else 5
    if rows[5] != ["Staff Information", *(["Count", "Percent"] * (1 if scope == "state" else 2))]:
        raise ContractError("TAPR Staff witness columns differ")
    sections = {
        "teacher_total_full_time_equiv_count": ("Teachers", _between(rows, "Professional Staff", "Professional Support")),
        "teacher_experience_average": ("Teachers", _between(rows, "Average Years Experience of Campus Staff",
                                                              "Average Teacher Salary by Years of Experience (regular duties only)")),
        "teacher_total_base_salary_average": ("Teachers", _between(rows, "Average Actual Salaries (regular duties only)",
                                                                   "Turnover Rate for Teachers")),
        "teacher_student_ratio": ("Number of Students per Teacher", rows),
        "teacher_turnover_ratio": ("Turnover Rate for Teachers", rows),
    }
    for name, (label, section) in sections.items():
        row = _one(section, label)
        paired = name in {"teacher_total_full_time_equiv_count", "teacher_student_ratio"}
        if len(row) != (width if paired else (2 if scope == "state" else 3)):
            raise ContractError(f"TAPR Staff witness {name!r} columns differ")
        result[f"staff_{name}"] = (_STAFF[name], row[1], 2025)
    return result


def _witnesses(release: DataRelease, store: ProvenanceStore) -> list[dict[str, Any]]:
    witnesses = []
    for identifier in release.catalog.to_dict()["artifacts"]:
        artifact_ref = release.catalog.artifact_ref(identifier)
        edition = artifact_ref.edition
        metadata = release.catalog.metadata(edition.dataset, edition=edition)
        references = [ref for ref in metadata.references if ref.reference_id.startswith("printed_")]
        expected = {"printed_texas"} if edition.scope == "state" else {"printed_region_06", "printed_region_16"}
        if {ref.reference_id for ref in references} != expected:
            raise ContractError("TAPR regional witness selection differs")
        for reference in references:
            receipt, = reference.provenance_ids
            record = store.verify_record_closure([receipt])[receipt]
            artifact = record["artifact"]
            if (record["kind"] != "acquisition" or record["method"] != "download" or not record["downloaded_at"]
                    or record["source_metadata"]["publisher"] != "Texas Education Agency"
                    or record["requested_year"] != 2025 or record["resolved_year"] != 2025
                    or record["requested_url"] != reference.url or record["resolved_url"] != reference.url
                    or record["parameters"].get("http_status") != 200
                    or record["parameters"].get("http_method") != "GET"
                    or artifact["sha256"] in artifact_ref.contract["inputs"]
                    or reference.reviewed_sha256 != artifact["sha256"] or not 0 < artifact["size_bytes"] <= 1024**2):
                raise ContractError("TAPR regional publisher witness receipt differs")
            entity = "state" if edition.scope == "state" else reference.reference_id[-2:]
            rows = _Rows(store.verify_artifact(artifact["sha256"]).read_text(encoding="cp1252")).rows
            cells = _cells(rows, edition.table, edition.scope, entity)
            if set(cells) != set(metadata.metrics):
                raise ContractError("TAPR regional qualified metric set differs from publisher controls")
            from pyarrow.dataset import field
            table = release.analytical_dataset(identifier)
            selected = (table.to_table(filter=field("region_number") == entity) if edition.scope == "region"
                        else table.to_table()).to_pylist()
            if len(selected) != 1:
                raise ContractError("TAPR regional witness entity is missing or duplicated")
            for metric, (unit, token, year) in cells.items():
                binding = metadata.metrics[metric]
                suffix = metric.replace("membership_mobility_", "membership_2024_mobility_").replace(
                    "membership_attrition_", "membership_2023_24_attrition_")
                if (binding.unit != unit or binding.period != DataPeriod("school_year", year - 1, year)
                        or binding.field != f"{edition.scope}_2025_{suffix}"):
                    raise ContractError(f"TAPR regional unit, period or field differs for {metric}")
                value = selected[0][binding.field]
                if _number(value, unit) != _number(token, unit, displayed=True):
                    raise ContractError(f"TAPR regional publisher value differs for {entity} {metric}")
            witnesses.append({"dataset": edition.dataset, "entity": entity, "receipt": receipt,
                              "compared_cells": len(cells), "source_downloaded_at": record["downloaded_at"]})
    return witnesses


def retained_regional_evidence(release: DataRelease, store: ProvenanceStore) -> list[str]:
    """Retain both printed controls and the lagged annual-attrition source closure."""
    roots = [ref.processing_id for ref in (release.catalog.artifact_ref(i)
             for i in release.catalog.to_dict()["artifacts"])]
    for edition in release.catalog.editions(include_metadata=False):
        key = EditionKey.from_dict({k: v for k, v in edition.items() if k != "edition_id"})
        roots.extend(receipt for reference in release.catalog.metadata(key.dataset, edition=key).references
                     if reference.reference_id.startswith("printed_") or reference.reference_id == "annual_attrition_witness"
                     for receipt in reference.provenance_ids)
    return sorted(store.verify_record_closure(roots))


def _attrition(release: DataRelease, store: ProvenanceStore) -> dict[str, Any]:
    from openpyxl import load_workbook

    artifact, = (release.catalog.artifact_ref(i) for i in release.catalog.to_dict()["artifacts"]
                 if release.catalog.artifact_ref(i).edition.dataset == "region_tapr_student_profile")
    metadata = release.catalog.metadata(artifact.edition.dataset, edition=artifact.edition)
    reference, = (r for r in metadata.references if r.reference_id == "annual_attrition_witness")
    receipt, = reference.provenance_ids
    record = store.verify_record_closure([receipt])[receipt]
    if (record["method"] != "download" or not record["downloaded_at"]
            or record["source_metadata"]["period"] != DataPeriod("school_year", 2023, 2024).to_dict()
            or record["requested_url"] != reference.url or record["resolved_url"] != reference.url
            or record["artifact"]["sha256"] != reference.reviewed_sha256):
        raise ContractError("TAPR regional annual attrition receipt differs")
    with store.verify_artifact(record["artifact"]["sha256"]).open("rb") as source:
        workbook = load_workbook(source, read_only=True, data_only=True)
        try:
            overview = " ".join(str(cell) for row in workbook["Overview"].values for cell in row if cell)
            headers, *rows = workbook["region"].values
        finally:
            workbook.close()
    if (headers != ("REGION", "REGNNAME", "RA0ATTR24D", "RA0ATTR24N", "RA0ATTR24R")
            or len(rows) != 20 or {row[0] for row in rows} != {f"{n:02d}" for n in range(1, 21)}
            or not all(text in overview for text in ("fall of the 2023-24 school year",
                       "fall of the 2024-25 school year", "did not return to the same campus"))):
        raise ContractError("TAPR regional annual attrition definition, columns or roster differs")
    native = {r["region_number"]: r for r in release.analytical_dataset(artifact.identity).to_table().to_pylist()}
    for region, _, denominator, numerator, rate in rows:
        den, num, pct = (_number(str(v), unit, displayed="," in str(v) or "%" in str(v)) for v, unit in
                         ((denominator, "count"), (numerator, "count"), (rate, "percent")))
        if (not 0 <= num <= den or den <= 0
                or (num * 100 / den).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP) != pct
                or num != _number(native[region][metadata.metrics["student_membership_attrition_all_students_count"].field], "count")
                or pct != _number(native[region][metadata.metrics["student_membership_attrition_all_students_percent"].field], "percent")):
            raise ContractError(f"TAPR regional annual attrition differs for ESC {region}")
    return {"receipt": receipt, "regions": 20, "compared_cells": 40, "measurement_year": "2023–24"}


def qualify_tapr_regional(release: DataRelease, store: ProvenanceStore) -> dict[str, Any]:
    """Compare complete raw exports and explicit semantic controls, never infer missing values."""
    if release.manifest_sha256 != REVIEWED_MANIFEST or not release.offline:
        raise ContractError("Regional TAPR qualification requires its exact reviewed offline manifest")
    refs = [release.catalog.artifact_ref(i) for i in release.catalog.to_dict()["artifacts"]]
    if (len(refs) != 3 or {r.edition.dataset for r in refs} != set(_SCOPES)
            or any((r.edition.scope, r.edition.table, r.contract["rows"]) != _SCOPES[r.edition.dataset]
                   or r.edition.edition != "2025" for r in refs)):
        raise ContractError("Regional TAPR requires exactly the three reviewed editions and row counts")
    retained_regional_evidence(release, store)
    release.verify()
    parity = validate_source_parity(release, store, edition_ids=[ref.edition.identity for ref in refs], verify_export=True)
    if parity["status"] != "PASS_RAW_PARITY":
        return {"status": "HOLD", "parity": parity}
    witnesses = _witnesses(release, store)
    attrition = _attrition(release, store)
    return {"status": "PASS_REVIEWED_NATIVE", "manifest_sha256": release.manifest_sha256,
            "build_id": release.build_id, "raw_parity": parity, "publisher_witnesses": witnesses,
            "annual_attrition": attrition,
            "limits": "38 regional Student and five Staff bindings only; other raw fields remain unbound. "
                      "Retained source checks, not fresh official reacquisition or complete source-year coverage."}
