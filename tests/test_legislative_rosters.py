from dataclasses import replace
from datetime import datetime, timezone

import pytest

from teadata import legislative_rosters
from teadata.dataset_metadata import DataPeriod, DatasetMetadata, ResolvedDataset
from teadata.legislative_rosters import (
    LegislativePlan,
    LegislativeSchoolRoster,
    RosterSource,
    _parse_rows,
    congressional_representation_plan,
    read_legislative_school_rosters,
)
from teadata.provenance import ProvenanceStore


PLAN = LegislativePlan.STATE_BOARD_OF_EDUCATION
NAMES = {"chapel hill isd (smith)": "212909", "chapel hill isd (titus)": "225906"}
SMITH = "Chapel Hill ISD (Smith) (80%)"
TITUS = "Chapel Hill ISD (Titus) (100%)"


@pytest.mark.parametrize(("instant", "plan"), [
    ("2023-01-03T17:00:00Z", LegislativePlan.US_HOUSE_C2193),
    ("2027-01-03T16:59:59Z", LegislativePlan.US_HOUSE_C2193),
    ("2027-01-03T17:00:00Z", LegislativePlan.US_HOUSE_C2333),
    ("2027-01-03T11:00:00-06:00", LegislativePlan.US_HOUSE_C2333),
])
def test_congressional_plan_changes_at_the_start_of_the_new_term(instant, plan):
    assert congressional_representation_plan(datetime.fromisoformat(instant)) is plan


def test_congressional_plan_rejects_an_ambiguous_or_earlier_date():
    with pytest.raises(ValueError, match="timezone-aware"):
        congressional_representation_plan(datetime(2027, 1, 3))
    with pytest.raises(ValueError, match="No reviewed"):
        congressional_representation_plan(datetime(2023, 1, 3, 16, 59, 59, tzinfo=timezone.utc))


def _rows():
    rows = [
        (None, None, None, None),
        (None, f"Red-160\nData: 2020 Census, SY 25-26\n{PLAN}  10/04/2021 8:03:48 PM",
         "District Population Analysis with School District Subtotals", None),
        (None, None, None, None),
        (None, None, f"SBOE DISTRICTS - {PLAN}", None),
        (None, None, None, None),
        (None, None, None, None),
        (None, None, None, None, "Total"),
    ]
    for seat in range(1, 16):
        rows.extend([
            (f"DISTRICT {seat}", None, None, None),
            (None, None, None, "Total:"),
            (None, None, None, "<18:"),
            (TITUS if seat == 2 else SMITH, None, None, "Total:"),
            (None, None, None, "<18:"),
        ])
    return [(*row, None) if len(row) == 4 else row for row in rows]


def test_red160_keeps_county_qualifiers_split_districts_and_physical_rows():
    rows = _parse_rows(_rows(), PLAN, NAMES)
    assert len(rows) == 15
    assert (rows[0].source_label, rows[0].worksheet_row) == (SMITH, 11)
    assert (rows[1].source_label, rows[1].worksheet_row) == (TITUS, 16)
    workbook = RosterSource("a" * 64, "b" * 64)
    roster = LegislativeSchoolRoster(PLAN, DataPeriod("school_year", 2025, 2026), 2020,
                                     workbook, replace(workbook, receipt="c" * 64), rows)
    assert roster.district_ids(1) == ("212909",)
    assert roster.district_ids(2) == ("225906",)
    assert roster.district_ids(15) == ("212909",)
    for invalid in (True, 16):
        with pytest.raises(ValueError, match="1 to 15"):
            roster.district_ids(invalid)


@pytest.mark.parametrize(("index", "row", "error"), [
    (0, (None, "Different period", None, None), "worksheet header"),
    (4, (SMITH, None, None, "Total:"), "worksheet header"),
    (6, ("DISTRICT 2", None, None, None), "constituency block"),
    (6, ("DISTRICT 1", None, None, "Total:"), "constituency block"),
    (7, (SMITH, None, None, "Total:"), "constituency block"),
    (9, ("Chapel Hill ISD (80%)", None, None, "Total:"), "Unknown or repeated"),
    (9, (SMITH, None, None, "<18:"), "school subtotal"),
    (10, (None, None, None, "Total:"), "school subtotal"),
    (14, (SMITH, None, None, "Total:"), "Incomplete"),
])
def test_red160_rejects_missing_ambiguous_and_drifting_rows(index, row, error):
    rows = _rows()
    rows[index + 1] = (*row, None)
    with pytest.raises(ValueError, match=error):
        _parse_rows(rows, PLAN, NAMES)


def test_red160_rejects_same_school_twice_in_one_constituency():
    rows = _rows()
    rows[12:12] = rows[10:12]
    with pytest.raises(ValueError, match="Unknown or repeated"):
        _parse_rows(rows, PLAN, NAMES)


def test_red160_rejects_workbook_repacked_with_a_leading_blank_row(tmp_path, monkeypatch):
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "Red160"
    for index, row in enumerate(_rows(), start=2):
        for column, value in zip((1, 2, 8, 4, 5), row, strict=True):
            sheet.cell(index, column, value)
    sheet.cell(3, 24, "Publisher end of projection")
    path = tmp_path / "repacked.xlsx"
    workbook.save(path)
    monkeypatch.setattr(legislative_rosters, "_workbook", lambda *_: path)
    monkeypatch.setattr(legislative_rosters, "_district_names", lambda *_: NAMES)
    source = RosterSource("a" * 64, "b" * 64)
    with pytest.raises(ValueError, match="worksheet header"):
        read_legislative_school_rosters(ProvenanceStore(tmp_path), district_locator=source, workbooks={PLAN: source})


def test_locator_does_not_open_a_gdal_virtual_source(tmp_path):
    store = ProvenanceStore(tmp_path / "provenance")
    csv = tmp_path / "separate.csv"
    csv.write_text("NAME,DISTRICT_C\nExample ISD,212909\n")
    path = tmp_path / "locator.geojson"
    path.write_text(f"""<OGRVRTDataSource><OGRVRTLayer name="separate">
    <SrcDataSource>{csv.as_posix()}</SrcDataSource><SrcLayer>separate</SrcLayer>
    </OGRVRTLayer></OGRVRTDataSource>""")
    locator_url = legislative_rosters._LOCATOR
    receipt = store.record_acquisition(
        path,
        ResolvedDataset("districts", "spatial", 2026, 2026, "", DatasetMetadata(
            title="Test locator", publisher="Texas Education Agency", version="2026",
            period=DataPeriod("school_year", 2025, 2026),
        )),
        downloaded_at=datetime.now(timezone.utc),
        requested_url=locator_url + "?f=geojson", resolved_url=locator_url + "?f=geojson",
        parameters={"form_url": locator_url, "http_method": "GET", "http_status": 200},
    )
    source = RosterSource(receipt, store.read_record(receipt)["artifact"]["sha256"])
    with pytest.raises(ValueError, match="local GeoJSON file"):
        read_legislative_school_rosters(store, district_locator=source, workbooks={PLAN: source})
