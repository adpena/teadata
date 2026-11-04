import csv
from datetime import datetime, timezone
from hashlib import sha256
from html import escape

import pytest

from teadata.dataset_metadata import DataPeriod, DatasetMetadata, ResolvedDataset
from teadata.manifest import ContractError
from teadata.peims_course_sources import BROKER, SELECTOR
from teadata.provenance import ProvenanceStore
from teadata.scripts import verify_peims_courses as check


@pytest.mark.parametrize(("damage", "message"), [
    (None, ""),
    ("request-geography", "exact statewide"), ("request-url", "exact statewide"),
    ("request-year", "exact statewide"), ("request-duplicate-control", "exact statewide"),
    ("response-filename", "exact statewide"),
    ("csv-year", "geography or year"), ("csv-geography", "geography or year"),
    ("csv-header", "ninth physical header"), ("html-header", "second table header"),
    ("html-geography", "second table header"), ("html-title", "title or year"),
    ("csv-footer", "CSV course footer differs"), ("html-footer", "HTML course footer differs"),
    ("wrong-code", "invalid or duplicate course key"), ("padded-code", "invalid or duplicate course key"),
    ("duplicate", "invalid or duplicate course key"), ("duplicate-by-grade", "invalid or duplicate course key"),
    ("grade-label", "unreviewed intended grade label"),
    ("different-key", "course keys differ"), ("invalid-token", "invalid COURSE ENROLLMENT token"),
    ("mask-as-zero", "measures differ"), ("changed-fte", "measures differ"),
])
def test_statewide_course_source_replay(tmp_path, monkeypatch, damage, message):
    monkeypatch.setattr(check, "_ROWS", {2026: 2})
    store = ProvenanceStore(tmp_path / "provenance")
    source_csv = tmp_path / "courses.csv"
    source_html = tmp_path / "courses.html"
    csv_header = ["SCOPE", "AGGREGATION LEVEL", "YEAR", "SUBJECT", "COURSE", "COURSE TITLE", "GRADE LEVEL",
                  "TEACHER HEAD COUNT", "TEACHER FTE COUNT", "CONTRACT TEACHER HEAD COUNT", "CONTRACT TEACHER FTE COUNT",
                  "COURSE ENROLLMENT"]
    if damage == "csv-header":
        csv_header[-1] = "MEMBERSHIP"
    csv_footer = ["-999 indicates counts or percentages are not available under the reviewed policy",
                  "Grade and release notes"]
    html_footer = ["N/A indicates counts or percentages are not available under the reviewed policy",
                   "Grade and release notes"]
    monkeypatch.setattr(check, "_ENGINEERING_REVISION_FOOTER_SHA256", {
        fmt: sha256("\n".join(footer).encode("utf-8")).hexdigest()
        for fmt, footer in (("csv", csv_footer), ("html", html_footer))
    })
    if damage == "csv-footer":
        csv_footer[0] = "-999 indicates counts or percentages are not available for other reasons"
    if damage == "html-footer":
        html_footer[1] = "Other grade and release notes"
    csv_prefix = ["DISTRICT" if damage == "csv-geography" else "STATE", "STATE",
                  "2024-2025" if damage == "csv-year" else "2025-2026"]
    first_code = " 00123456" if damage == "padded-code" else "00123456"
    second_code = "00123456" if damage == "duplicate-by-grade" else "00123457"
    second_grade = {"duplicate-by-grade": "GRADE 8", "grade-label": "GRADE UNKNOWN"}.get(damage, "GRADES 9-12")
    with source_csv.open("w", encoding="cp1252", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerows([
            ["TEXAS EDUCATION AGENCY"], [], ["TOTAL ENROLLMENT AND TEACHER FTE COUNTS BY SUBJECT AND COURSE"],
            [], ["PEIMS DATA 2025-2026"], [], ["STATEWIDE"], [], csv_header,
            [*csv_prefix, "EXAMPLE   SUBJECT", first_code, "Course   One", "GRADES 9-12", "2000", "1.50", "0", "0", "-999"],
            [*csv_prefix, "EXAMPLE SUBJECT", second_code, "Course Two", second_grade, "1", "0.125", "1", "0.75", "0"],
            [], *[[line] for line in csv_footer],
        ])
    course_code = {"wrong-code": "123457", "duplicate": "00123456", "different-key": "00123458"}.get(damage, second_code)
    masked = {"invalid-token": "-999", "mask-as-zero": "0"}.get(damage, "N/A")
    year_label = "2024-2025" if damage == "html-title" else "2025-2026"
    html_rows = [
        ["District" if damage == "html-geography" else "Statewide"],
        ["Subject", "Course", "Course Title", "Grade Level", "TeacherHead Count", "TeacherFTE Count",
         "Contract TeacherHead Count", "Contract TeacherFTE Count", "Membership" if damage == "html-header" else "CourseEnrollment"],
        ["EXAMPLE SUBJECT", "00123456", "Course One", "GRADES 9-12", "2,000", "1.5", "0", "0.00", masked],
        ["", course_code, "Course Two", second_grade, "1", "0.126" if damage == "changed-fte" else "0.125", "1", "0.75", "0"],
        *[[line] for line in html_footer],
    ]
    table = "".join("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>" for row in html_rows)
    source_html.write_text(f"<h1>{year_label} Teacher FTE Counts and Course Enrollment Reports</h1><table>{table}</table>",
                           encoding="cp1252")

    receipt_ids = []
    for fmt, source, media in (("csv", source_csv, "application/csv"), ("html", source_html, "text/html")):
        request_year = 2025 if fmt == "csv" and damage == "request-year" else 2026
        report = "StateDistrict" if fmt == "csv" and damage == "request-geography" else "StateState"
        url = "https://example.invalid/other" if fmt == "csv" and damage == "request-url" else BROKER
        metadata = DatasetMetadata(title="PEIMS teacher counts and course enrollment", version=str(request_year),
                                   source_url=SELECTOR, publisher="Texas Education Agency",
                                   period=DataPeriod("school_year", request_year - 1, request_year))
        dataset = ResolvedDataset("peims_teacher_course_enrollment", "data_sources", request_year, request_year, "", metadata)
        controls = [["_service", "marykay"], ["_program", "adhoc.std_driver1.sas"], ["RptClass", "FteEnroll"],
                    ["_debug", "0"], ["SchoolYr", str(request_year)[-2:]], ["report", report], ["Campus", ""],
                    ["format", fmt], ["submit", "Next"]]
        if fmt == "csv" and damage == "request-duplicate-control":
            controls.append(["format", fmt])
        headers = {"content-type": media}
        if fmt == "csv":
            scope = "StateDistrict" if damage == "response-filename" else "StateState"
            headers["content-disposition"] = f"attachment; filename=FteEnroll{scope}{request_year % 100:02}state.csv"
        receipt_ids.append(store.record_acquisition(
            source, dataset, requested_url=url, resolved_url=url, downloaded_at=datetime(2026, 11, 4, tzinfo=timezone.utc),
            response_headers=headers, parameters={"http_status": 200, "http_method": "POST",
                "form_url": url, "form_content_type": "application/x-www-form-urlencoded", "form_controls": controls},
        ))

    if damage:
        with pytest.raises(ContractError, match=message):
            check.verify_pair(store, 2026, *receipt_ids)
    else:
        result = check.verify_pair(store, 2026, *receipt_ids)
        assert (result["rows"], result["compared_cells"], result["masked_cells"]) == (2, 10, 1)
        assert [result[fmt]["receipt_id"] for fmt in ("csv", "html")] == receipt_ids
        assert result["csv"]["downloaded_at"] == "2026-11-04T00:00:00+00:00"


def test_cli_requires_all_four_exact_years(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        check.main(["--store", str(tmp_path), "--pair", "2026", "csv", "html"])
    assert error.value.code == 2
    assert "one exact CSV/HTML receipt pair per year" in capsys.readouterr().err
