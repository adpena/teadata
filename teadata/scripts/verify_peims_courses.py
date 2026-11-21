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

from teadata.manifest import ContractError
from teadata.peims_course_sources import (
    CSV_HEADERS, ENCODING, MEASURES as _MEASURES, course_csv_preamble, course_grade_labels, open_course_report,
)
from teadata.provenance import ProvenanceStore


_CURRENT_ROWS = {2023: 1386, 2024: 1397, 2025: 1422, 2026: 1453}
_HISTORICAL_ROWS = {2019: 1342, 2020: 1358, 2021: 1379, 2022: 1385}
_EARLY_ROWS = {2013: 1073, 2014: 1095, 2015: 1149, 2016: 1225, 2017: 1224, 2018: 1319}
_ROWS = _EARLY_ROWS | _HISTORICAL_ROWS | _CURRENT_ROWS
_2013_FOOTER_SHA256 = {
    "csv": "298d80b1d189eef54af8bc792ef59a78f7a621c836fa007ab78071bd838b862b",
    "html": "5ed885d938e702aaa5a6d4f50a1814f05560db5b97265231784ba6ce98e3b55d",
}
_2014_2018_CSV_FOOTER_SHA256 = "95073454b1a48e27ba4b5d18ac70b260c590eb6e223ea2985f150ce46335c9d7"
_STANDARD_FOOTER_SHA256 = {
    "csv": "2b84fb6e114e983247ff277f83034d30ee2d4323dae633047e2020a9dafe5953",
    "html": "7b70e943548e2c8d03879c24b993336e03f49071bdc26c30a300d4049aa44ce7",
}
_ENGINEERING_REVISION_FOOTER_SHA256 = {
    "csv": "2d6c1186f89685ab9632e10af8d4f8a15396305c5c8037b159a78925f30dd22c",
    "html": "1ab5594c3c2161a7ac074516a4a15d8ab8e990457eca95ea055478452530bb57",
}
_HTML_HEADER = ("Subject", "Course", "Course Title", "Grade Level", "TeacherHead Count", "TeacherFTE Count",
                "Contract TeacherHead Count", "Contract TeacherFTE Count", "CourseEnrollment")
_COUNT = r"(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)"
_FORMAT = Literal["csv", "html"]


class _Course(NamedTuple):
    subject: str
    title: str
    grade: str
    measures: tuple[Decimal | Literal["ERROR"] | None, ...]


def _open_report(store: ProvenanceStore, identifier: str, year: int, fmt: _FORMAT) -> tuple[str, dict[str, Any]]:
    path, record = open_course_report(store, identifier, year, "state", fmt)
    artifact = record["artifact"]
    return path.read_text(encoding=ENCODING), {
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
    if rows[:9] != course_csv_preamble(year, "state"):
        raise ContractError("CSV statewide preamble or ninth physical header differs")
    stop = 9 + _ROWS[year]
    body, footer = rows[9:stop], [row for row in rows[stop:] if row]
    _check_footer(footer, year, "csv")
    if any(len(row) != len(CSV_HEADERS["state"]) or row[:3] != ["STATE", "STATE", period] for row in body):
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
    expected_digest = expected[fmt]
    if year == 2013:
        expected_digest = _2013_FOOTER_SHA256[fmt]
    elif 2014 <= year <= 2018 and fmt == "csv":
        expected_digest = _2014_2018_CSV_FOOTER_SHA256
    if any(len(row) != 1 for row in rows):
        raise ContractError(f"{fmt.upper()} course footer differs")
    digest = sha256("\n".join(row[0] for row in rows).encode("utf-8")).hexdigest()
    if digest != expected_digest:
        raise ContractError(f"{fmt.upper()} course footer differs")


def _course_rows(rows: list[list[str]], fmt: _FORMAT, *, year: int | None = None,
                 teacher_fte_errors: frozenset[str] = frozenset()) -> dict[str, _Course]:
    result = {}
    subject = ""
    masked = "-999" if fmt == "csv" else "N/A"
    grade_labels = course_grade_labels(year)
    errors: set[str] = set()
    for row in rows:
        subject_label, code, title, grade = row[:4]
        subject_label, title, grade = (" ".join(value.split()) for value in (subject_label, title, grade))
        subject = subject if fmt == "html" and not subject_label else subject_label
        if not subject or not title or re.fullmatch(r"[A-Z0-9]{8}", code) is None or code in result:
            raise ContractError(f"{fmt.upper()} invalid or duplicate course key: {code!r}")
        if grade not in grade_labels:
            raise ContractError(f"{fmt.upper()} unreviewed intended grade label for {code!r}: {grade!r}")
        measures: list[Decimal | Literal["ERROR"] | None] = []
        for index, token in enumerate(row[4:]):
            pattern = _COUNT + (r"(?:\.[0-9]+)?" if index in (1, 3) else "")
            if index == 4 and token == masked:
                measures.append(None)
            elif index == 1 and token == "ERROR" and code in teacher_fte_errors:
                errors.add(code)
                measures.append("ERROR")
            elif re.fullmatch(pattern, token):
                measures.append(Decimal(token.replace(",", "")))
            else:
                raise ContractError(f"{fmt.upper()} invalid {_MEASURES[index]} token for {code!r}: {token!r}")
        result[code] = _Course(subject, title, grade, tuple(measures))
    if errors != teacher_fte_errors:
        raise ContractError(f"{fmt.upper()} publisher teacher FTE ERROR course identities differ")
    return result


def verify_pair(store: ProvenanceStore, year: int, csv_id: str, html_id: str) -> dict[str, Any]:
    csv_text, csv_source = _open_report(store, csv_id, year, "csv")
    html_text, html_source = _open_report(store, html_id, year, "html")
    raw = _course_rows(_csv_rows(csv_text, year), "csv", year=year)
    displayed = _course_rows(_html_rows(html_text, year), "html", year=year)
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
        if sorted(year for year, _, _ in selections) not in (sorted(_CURRENT_ROWS), sorted(_HISTORICAL_ROWS),
                                                            sorted(_EARLY_ROWS)):
            raise ContractError("Select one exact CSV/HTML receipt pair per year for 2013–2018, 2019–2022 or 2023–2026")
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
