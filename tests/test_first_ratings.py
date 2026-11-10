from datetime import datetime

from openpyxl import Workbook
from openpyxl.utils.datetime import CALENDAR_WINDOWS_1900
import pytest

from teadata.first_ratings import (
    _HEADERS,
    _POPULATIONS,
    _check_roster,
    _literal_rows,
    _project,
    _rating,
    _receipt,
    materialize_first_ratings,
)
from teadata.manifest import ContractError
from teadata.provenance import ProvenanceStore

DISTRICT, UNIVERSITY = _POPULATIONS
UPDATED = datetime(2026, 8, 7)
RAW_DATE = "46241"


@pytest.mark.parametrize(("population", "rating", "score", "score_status"), [
    (DISTRICT, "F = Substandard Achievement", 0, "observed"),
    (DISTRICT, "F = Substandard Achievement", 98, "observed"),
    (UNIVERSITY, "Pass", None, "not_applicable"),
])
def test_first_rating_preserves_scale_score_and_raw_identifier(population, rating, score, score_status):
    status = "Pass" if population.university else "Fail"
    raw_score = None if score is None else str(score)
    entry = _rating(population, 3, (1902, "Example", status, rating, score, UPDATED, 7),
                    ("1902", "Example", status, rating, raw_score, RAW_DATE, "7"), CALENDAR_WINDOWS_1900)
    row, = _project({entry.district_number: entry}).to_pylist()
    assert row["district_number"] == "001902"
    assert row["first_raw_district_number"] == "1902"
    assert row["first_rating"] == rating
    assert row["first_score"] == score
    assert row["first_raw_score"] == raw_score
    assert row["first_score_status"] == score_status


@pytest.mark.parametrize("score", [None, -1, True, 101])
def test_first_district_rejects_invalid_score(score):
    with pytest.raises(ContractError, match="source row 3"):
        _rating(DISTRICT, 3, (1902, "Example", "Pass", "A = Superior Achievement", score, UPDATED, 7),
                (), CALENDAR_WINDOWS_1900)


def test_first_workbook_rejects_cells_outside_crosswalk(tmp_path):
    path = tmp_path / "sample.xlsx"
    book = Workbook()
    sheet = book.active
    sheet.title = DISTRICT.sheet
    sheet.append([DISTRICT.workbook_title])
    sheet.merge_cells("A1:G1")
    sheet.append(_HEADERS)
    sheet.append([1902, "Example", "Pass", "A = Superior Achievement", 90, UPDATED, 7, "Unmapped"])
    book.save(path)
    with pytest.raises(ContractError, match="workbook framing"):
        _literal_rows(path, DISTRICT)


def test_first_roster_checks_coverage_duplicates_and_status_icon():
    rating = "A = Superior Achievement"
    entry = _rating(DISTRICT, 3, (1902, "Example", "Pass", rating, 90, UPDATED, 7),
                    ("1902", "Example", "Pass", rating, "90", RAW_DATE, "7"), CALENDAR_WINDOWS_1900)
    records = {entry.district_number: entry}
    heading = f"<h1>Final Ratings</h1><h2>{DISTRICT.roster_heading}</h2>"
    row = f'<tr><td>001902</td><td>Example</td><td>{rating}</td><td><img alt="Passed"></td></tr>'
    _check_roster(DISTRICT, [heading + row], records)
    with pytest.raises(ContractError, match="incomplete"):
        _check_roster(DISTRICT, [heading], records)
    with pytest.raises(ContractError, match="repeated"):
        _check_roster(DISTRICT, [heading + row + row], records)
    with pytest.raises(ContractError, match="disagrees with workbook"):
        _check_roster(DISTRICT, [heading + row.replace("Passed", "Failed")], records)


def test_first_receipt_rejects_a_different_population():
    from teadata.first_ratings import EDITION, PERIOD

    url = "https://example.test/first"
    parameters = {"http_method": "POST", "scope": DISTRICT.scope}
    receipt = {"kind": "acquisition", "method": "download", "dataset": DISTRICT.dataset,
               "requested_url": url, "resolved_url": url, "requested_year": int(EDITION),
               "resolved_year": int(EDITION), "parameters": {"http_status": 200, **parameters},
               "source_metadata": {"title": DISTRICT.roster_heading, "version": EDITION, "period": PERIOD.to_dict()}}
    _receipt(receipt, DISTRICT, url, parameters)
    with pytest.raises(ContractError, match="Unexpected FIRST acquisition"):
        _receipt(receipt, UNIVERSITY, url, parameters)


def test_first_candidate_never_overwrites_existing_output(tmp_path):
    existing = tmp_path / "release"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="exclusive"):
        materialize_first_ratings(ProvenanceStore(tmp_path / "store"), existing)
