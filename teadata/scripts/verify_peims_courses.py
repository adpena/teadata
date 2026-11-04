"""Compare retained statewide PEIMS course CSV and HTML source reports offline."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import re
from typing import Any, Literal, NamedTuple

from teadata.dataset_metadata import DataPeriod
from teadata.manifest import ContractError
from teadata.provenance import ProvenanceStore


_DATASET = "peims_teacher_course_enrollment"
_BROKER = "https://rptsvr1.tea.texas.gov/cgi/sas/broker"
_SELECTOR = "https://rptsvr1.tea.texas.gov/adhocrpt/adfte.html"
_ROWS = {2023: 1386, 2024: 1397, 2025: 1422, 2026: 1453}
_STANDARD_FOOTER_SHA256 = {
    "csv": "2b84fb6e114e983247ff277f83034d30ee2d4323dae633047e2020a9dafe5953",
    "html": "7b70e943548e2c8d03879c24b993336e03f49071bdc26c30a300d4049aa44ce7",
}
_ENGINEERING_REVISION_FOOTER_SHA256 = {
    "csv": "2d6c1186f89685ab9632e10af8d4f8a15396305c5c8037b159a78925f30dd22c",
    "html": "1ab5594c3c2161a7ac074516a4a15d8ab8e990457eca95ea055478452530bb57",
}
_MEASURES = ("TEACHER HEAD COUNT", "TEACHER FTE COUNT", "CONTRACT TEACHER HEAD COUNT",
             "CONTRACT TEACHER FTE COUNT", "COURSE ENROLLMENT")
_CSV_HEADER = ("SCOPE", "AGGREGATION LEVEL", "YEAR", "SUBJECT", "COURSE", "COURSE TITLE", "GRADE LEVEL", *_MEASURES)
_HTML_HEADER = ("Subject", "Course", "Course Title", "Grade Level", "TeacherHead Count", "TeacherFTE Count",
                "Contract TeacherHead Count", "Contract TeacherFTE Count", "CourseEnrollment")
_GRADE_LABELS = frozenset({
    "ALL GRADE LEVELS", "EARLY EDUCATION", "ELEMENTARY (GRADES 1-6)",
    "GRADE 1", "GRADE 2", "GRADE 3", "GRADE 4", "GRADE 5", "GRADE 6", "GRADE 7", "GRADE 8", "GRADES 9-12",
    "KINDERGARTEN", "KINDERGARTEN/ELEMENTARY (K-6)", "MIDDLE SCHOOL (GRADES 6 - 8)",
    "NOT APPLICABLE", "PRE-KINDERGARTEN", "SECONDARY (GRADES 7-12)",
})
_COUNT = r"(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)"
_FORMAT = Literal["csv", "html"]


class _Course(NamedTuple):
    subject: str
    title: str
    grade: str
    measures: tuple[Decimal | None, ...]


def _open_report(store: ProvenanceStore, identifier: str, year: int, fmt: _FORMAT) -> tuple[str, dict[str, Any]]:
    record = store.read_record(identifier)
    parameters = record["parameters"]
    metadata = record["source_metadata"]
    artifact = record["artifact"]
    controls = [["_service", "marykay"], ["_program", "adhoc.std_driver1.sas"], ["RptClass", "FteEnroll"],
                ["_debug", "0"], ["SchoolYr", f"{year % 100:02}"], ["report", "StateState"], ["Campus", ""],
                ["format", fmt], ["submit", "Next"]]
    headers = record["response_headers"]
    content_type = headers.get("content-type", "").split(";", 1)[0]
    csv_disposition = f"attachment; filename=FteEnrollStateState{year % 100:02}state.csv"
    if (record["kind"] != "acquisition" or record["method"] != "download" or record["dataset"] != _DATASET
            or record["section"] != "data_sources" or (record["requested_year"], record["resolved_year"]) != (year, year)
            or (record["requested_url"], record["resolved_url"], parameters.get("form_url")) != (_BROKER,) * 3
            or parameters.get("http_method") != "POST" or parameters.get("http_status") != 200
            or parameters.get("form_content_type") != "application/x-www-form-urlencoded"
            or parameters.get("form_controls") != controls or metadata.get("version") != str(year)
            or metadata.get("period") != DataPeriod("school_year", year - 1, year).to_dict()
            or metadata.get("source_url") != _SELECTOR or metadata.get("publisher") != "Texas Education Agency"
            or content_type != ("application/csv" if fmt == "csv" else "text/html")
            or (fmt == "csv" and headers.get("content-disposition") != csv_disposition)
            or not record["downloaded_at"] or not 0 < artifact["size_bytes"] <= 8 * 1024**2):
        raise ContractError(f"Receipt {identifier} is not the exact statewide {year} {fmt.upper()} request")
    path = store.verify_artifact(artifact["sha256"])
    if path.stat().st_size != artifact["size_bytes"]:
        raise ContractError(f"Receipt {identifier} artifact size differs")
    return path.read_text(encoding="cp1252"), {
        "receipt_id": identifier, "sha256": artifact["sha256"], "downloaded_at": record["downloaded_at"],
        "recorded_at": record["recorded_at"],
        "requested_url": record["requested_url"], "resolved_url": record["resolved_url"],
    }


def _csv_rows(text: str, year: int) -> list[list[str]]:
    period = f"{year - 1}-{year}"
    reader = csv.reader(StringIO(text, newline=""), strict=True)
    rows = []
    for physical_record, row in enumerate(reader, 1):
        if reader.line_num != physical_record:
            raise ContractError("CSV contains a record spanning multiple physical lines")
        rows.append(row)
    preamble = [["TEXAS EDUCATION AGENCY"], [], ["TOTAL ENROLLMENT AND TEACHER FTE COUNTS BY SUBJECT AND COURSE"],
                [], [f"PEIMS DATA {period}"], [], ["STATEWIDE"], [], list(_CSV_HEADER)]
    if rows[:9] != preamble:
        raise ContractError("CSV statewide preamble or ninth physical header differs")
    stop = 9 + _ROWS[year]
    body, footer = rows[9:stop], [row for row in rows[stop:] if row]
    _check_footer(footer, year, "csv")
    if any(len(row) != len(_CSV_HEADER) or row[:3] != ["STATE", "STATE", period] for row in body):
        raise ContractError("CSV course record width, geography or year differs")
    return [row[3:] for row in body]


def _html_rows(text: str, year: int) -> list[list[str]]:
    from teadata.native_witnesses import _Rows

    titles = re.findall(r"<h1\b[^>]*>\s*([^<]*?)\s*</h1>", text, re.IGNORECASE)
    if titles != [f"{year - 1}-{year} Teacher FTE Counts and Course Enrollment Reports"]:
        raise ContractError("HTML course report title or year differs")
    rows = _Rows(text).rows
    if rows[:2] != [["Statewide"], list(_HTML_HEADER)]:
        raise ContractError("HTML course report geography or second table header differs")
    stop = 2 + _ROWS[year]
    body, footer = rows[2:stop], rows[stop:]
    _check_footer(footer, year, "html")
    if any(len(row) != len(_HTML_HEADER) for row in body):
        raise ContractError("HTML course record width differs")
    return body


def _check_footer(rows: list[list[str]], year: int, fmt: _FORMAT) -> None:
    """Reject changes to the reviewed masking, grade and revision notes."""
    expected = _ENGINEERING_REVISION_FOOTER_SHA256 if year == 2026 else _STANDARD_FOOTER_SHA256
    if any(len(row) != 1 for row in rows):
        raise ContractError(f"{fmt.upper()} course footer differs")
    digest = sha256("\n".join(row[0] for row in rows).encode("utf-8")).hexdigest()
    if digest != expected[fmt]:
        raise ContractError(f"{fmt.upper()} course footer differs")


def _course_rows(rows: list[list[str]], fmt: _FORMAT) -> dict[str, _Course]:
    result = {}
    subject = ""
    masked = "-999" if fmt == "csv" else "N/A"
    for row in rows:
        subject_label, code, title, grade = row[:4]
        subject_label, title, grade = (" ".join(value.split()) for value in (subject_label, title, grade))
        subject = subject if fmt == "html" and not subject_label else subject_label
        if not subject or not title or re.fullmatch(r"[A-Z0-9]{8}", code) is None or code in result:
            raise ContractError(f"{fmt.upper()} invalid or duplicate course key: {code!r}")
        if grade not in _GRADE_LABELS:
            raise ContractError(f"{fmt.upper()} unreviewed intended grade label for {code!r}: {grade!r}")
        measures: list[Decimal | None] = []
        for index, token in enumerate(row[4:]):
            pattern = _COUNT + (r"(?:\.[0-9]+)?" if index in (1, 3) else "")
            if token == masked:
                measures.append(None)
            elif re.fullmatch(pattern, token):
                measures.append(Decimal(token.replace(",", "")))
            else:
                raise ContractError(f"{fmt.upper()} invalid {_MEASURES[index]} token for {code!r}: {token!r}")
        result[code] = _Course(subject, title, grade, tuple(measures))
    return result


def verify_pair(store: ProvenanceStore, year: int, csv_id: str, html_id: str) -> dict[str, Any]:
    csv_text, csv_source = _open_report(store, csv_id, year, "csv")
    html_text, html_source = _open_report(store, html_id, year, "html")
    raw = _course_rows(_csv_rows(csv_text, year), "csv")
    displayed = _course_rows(_html_rows(html_text, year), "html")
    period = f"{year - 1}-{year}"
    if len(raw) != _ROWS[year] or raw.keys() != displayed.keys():
        raise ContractError(f"Statewide {period} CSV and HTML course keys differ")
    for key, course in raw.items():
        if course != displayed[key]:
            raise ContractError(f"Statewide {period} CSV and HTML course label or measures differ for {key!r}")
    return {"school_year": period, "rows": len(raw), "compared_cells": len(raw) * len(_MEASURES),
            "masked_cells": sum(value is None for course in raw.values() for value in course.measures),
            "csv": csv_source, "html": html_source}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True, help="Existing retained provenance store")
    parser.add_argument("--pair", nargs=3, action="append", required=True, metavar=("YEAR", "CSV_RECEIPT", "HTML_RECEIPT"))
    args = parser.parse_args(argv)
    try:
        selections = [(int(year), csv_id, html_id) for year, csv_id, html_id in args.pair]
        if sorted(year for year, _, _ in selections) != sorted(_ROWS):
            raise ContractError("Select one exact CSV/HTML receipt pair per year from 2023 through 2026")
        if not (args.store / "records").is_dir() or not (args.store / "objects").is_dir():
            raise ContractError("An existing retained provenance store is required")
        store = ProvenanceStore(args.store)
        reports = [verify_pair(store, *selection) for selection in sorted(selections)]
    except (ContractError, OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        parser.exit(2, f"Course report verification failed: {error}\n")
    print(json.dumps({"status": "PASS_STATEWIDE_COURSE_REPORT_PARITY", "geography": "state",
                      "scope": "same-publisher source representation comparison", "measures": _MEASURES,
                      "rows": sum(item["rows"] for item in reports),
                      "compared_cells": sum(item["compared_cells"] for item in reports),
                      "masked_cells": sum(item["masked_cells"] for item in reports), "editions": reports}, sort_keys=True))


if __name__ == "__main__":
    main()
