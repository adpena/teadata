"""Shared source identity for the PEIMS teacher FTE and course reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from .dataset_metadata import DataPeriod
from .manifest import ContractError
from .provenance import ProvenanceStore


DATASET = "peims_teacher_course_enrollment"
BROKER = "https://rptsvr1.tea.texas.gov/cgi/sas/broker"
SELECTOR = "https://rptsvr1.tea.texas.gov/adhocrpt/adfte.html"
ENCODING = "cp1252"
MEASURES = ("TEACHER HEAD COUNT", "TEACHER FTE COUNT", "CONTRACT TEACHER HEAD COUNT",
            "CONTRACT TEACHER FTE COUNT", "COURSE ENROLLMENT")
_GEOGRAPHY = ("SCOPE", "AGGREGATION LEVEL", "YEAR")
_COURSE = ("SUBJECT", "COURSE", "COURSE TITLE", "GRADE LEVEL", *MEASURES)
CSV_HEADERS = {"state": (*_GEOGRAPHY, *_COURSE), "district": (
    *_GEOGRAPHY, "DISTRICT", "DISTRICT NAME", "CHARTER STATUS", "REGION", *_COURSE,
), "campus": (
    *_GEOGRAPHY, "DISTRICT", "DISTRICT NAME", "CAMPUS", "CAMPUS NAME", "CHARTER STATUS", "REGION", *_COURSE,
)}
COURSE_FACT_UNITS = {"teacher_head_count": "count", "teacher_fte_count": "fte",
                     "contract_teacher_head_count": "count", "contract_teacher_fte_count": "fte",
                     "course_enrollment_count": "count"}
SOURCE_RECORD = "peims_source_record"
_GEO_FIELDS = {
    "state": (),
    "district": ("district_number", "district_name", "charter_status", "region_number"),
    "campus": ("source_district_number", "source_district_name", "campus_number", "campus_name",
               "charter_status", "region_number"),
}

GRADE_LABELS = frozenset({
    "ALL GRADE LEVELS", "EARLY EDUCATION", "ELEMENTARY (GRADES 1-6)",
    "GRADE 1", "GRADE 2", "GRADE 3", "GRADE 4", "GRADE 5", "GRADE 6", "GRADE 7", "GRADE 8", "GRADES 9-12",
    "KINDERGARTEN", "KINDERGARTEN/ELEMENTARY (K-6)", "MIDDLE SCHOOL (GRADES 6 - 8)",
    "NOT APPLICABLE", "PRE-KINDERGARTEN", "SECONDARY (GRADES 7-12)",
})


def course_field_names(scope: str) -> tuple[str, ...]:
    return ("source_scope", "aggregation_level", "school_year", *_GEO_FIELDS[scope],
            "subject", "course_code", "course_title", "grade_level", *COURSE_FACT_UNITS)


def course_column_support() -> dict[str, Any]:
    return {"fields": {field: {"raw": f"peims_raw_{field}", "status": f"peims_status_{field}"}
                       for field in COURSE_FACT_UNITS}, "context": [SOURCE_RECORD]}


def matches_course_header(table: dict[str, Any], scope: str) -> bool:
    layout = table.get("layout", {})
    expected = tuple((header, header, field, "fact" if field in COURSE_FACT_UNITS else "identity")
                     for header, field in zip(CSV_HEADERS[scope], course_field_names(scope), strict=True))
    return (table.get("header_identity") == "position"
            and (layout.get("code_record"), layout.get("label_record"), layout.get("first_data_record")) == (9, None, 10)
            and tuple((row["code"], row["label"], row["field"], row.get("disposition")) for row in table["fields"])
            == expected)


def course_csv_preamble(year: int, scope: str) -> list[list[str]]:
    geography = {"state": "STATEWIDE", "district": "STATEWIDE BY DISTRICT", "campus": "STATEWIDE BY CAMPUS"}[scope]
    return [["TEXAS EDUCATION AGENCY"], [], ["TOTAL ENROLLMENT AND TEACHER FTE COUNTS BY SUBJECT AND COURSE"],
            [], [f"PEIMS DATA {year - 1}-{year}"], [], [geography], [], list(CSV_HEADERS[scope])]


def open_course_report(store: ProvenanceStore, identifier: str, year: int, scope: str,
                       fmt: Literal["csv", "html"]) -> tuple[Path, dict[str, Any]]:
    record = store.read_record(identifier)
    parameters = record["parameters"]
    metadata = record["source_metadata"]
    artifact = record["artifact"]
    report = {"state": "StateState", "district": "StateDistrict", "campus": "StateCampus"}[scope]
    controls = [["_service", "marykay"], ["_program", "adhoc.std_driver1.sas"], ["RptClass", "FteEnroll"],
                ["_debug", "0"], ["SchoolYr", f"{year % 100:02}"], ["report", report], ["Campus", ""],
                ["format", fmt], ["submit", "Next"]]
    headers = record["response_headers"]
    content_type = headers.get("content-type", "").split(";", 1)[0]
    csv_disposition = f"attachment; filename=FteEnroll{report}{year % 100:02}state.csv"
    maximum = {"state": 8, "district": 64, "campus": 128}[scope] * 1024**2
    if (record["kind"] != "acquisition" or record["method"] != "download" or record["dataset"] != DATASET
            or record["section"] != "data_sources" or (record["requested_year"], record["resolved_year"]) != (year, year)
            or (record["requested_url"], record["resolved_url"], parameters.get("form_url")) != (BROKER,) * 3
            or parameters.get("http_method") != "POST" or parameters.get("http_status") != 200
            or parameters.get("form_content_type") != "application/x-www-form-urlencoded"
            or parameters.get("form_controls") != controls or metadata.get("version") != str(year)
            or metadata.get("period") != DataPeriod("school_year", year - 1, year).to_dict()
            or metadata.get("source_url") != SELECTOR or metadata.get("publisher") != "Texas Education Agency"
            or content_type != ("application/csv" if fmt == "csv" else "text/html")
            or (fmt == "csv" and headers.get("content-disposition") != csv_disposition)
            or not record["downloaded_at"] or not 0 < artifact["size_bytes"] <= maximum):
        raise ContractError(f"Receipt {identifier} is not the exact statewide {year} {fmt.upper()} request for {scope}")
    path = store.verify_artifact(artifact["sha256"])
    if path.stat().st_size != artifact["size_bytes"]:
        raise ContractError(f"Receipt {identifier} artifact size differs")
    return path, record
