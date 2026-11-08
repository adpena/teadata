from csv import writer
from datetime import datetime, timezone

import pytest

from teadata import EntitySelectionError
from teadata.askted_directory import DirectoryField as Field, read_askted_directory
from teadata.dataset_metadata import DataPeriod, DatasetMetadata, ResolvedDataset
from teadata.provenance import ProvenanceStore
from teadata.scripts.download_askted import ARCHIVE_URL


def archive(tmp_path, *, year=2026, season="fall", rows=(), date_override=None, attachment=None):
    store = ProvenanceStore(tmp_path / "provenance")
    downloaded = datetime(2026, 11, 7, tzinfo=timezone.utc)
    date = f"{year - 1}-10-31" if season == "fall" else f"{year}-03-31"
    index_date = date_override or ("10/31/" + str(year - 1) if season == "fall" else "03/31/" + str(year))
    html = tmp_path / "index.html"
    html.write_text(f"<table><tr><td>{year-1}-{year} School Year</td><td>"
                    f"<a href=\"javascript:__doPostBack('{season}','')\">{season.title()} "
                    f"(as of {index_date})</a></td></tr></table>")
    index = store.record_acquisition(html, ResolvedDataset("public_school_directory", "data_sources", year, year, "",
        DatasetMetadata(title="Fixture archive index", source_url=ARCHIVE_URL)), downloaded_at=downloaded, requested_url=ARCHIVE_URL,
        resolved_url=ARCHIVE_URL, response_headers={"content-type": "text/html"},
        parameters={"http_method": "GET", "http_status": 200})
    raw = tmp_path / "directory.csv"
    with raw.open("w", encoding="utf-8", newline="") as stream:
        csv = writer(stream)
        csv.writerow(("District Number", "District Name", "School Number", "School Name", "District Type", "School Status",
                      "Charter Type", "District City", "District Site City", "School City", "School Site City",
                      "County Number", "County Name", "ESC Region Served", "ESC Region PEIMS", "ESC Region Geographic",
                      f"District Enrollment as of Oct {year-1}", f"School Enrollment as of Oct {year-1}"))
        csv.writerows(rows)
    receipt = store.record_acquisition(raw, ResolvedDataset("public_school_directory", "data_sources", year, year, "",
        DatasetMetadata(title="Fixture archive", source_url=ARCHIVE_URL, version=f"{year}-{season}",
                        period=DataPeriod("as_of", as_of=date))), downloaded_at=downloaded, requested_url=ARCHIVE_URL, resolved_url=ARCHIVE_URL,
        response_headers={"content-type": "text/csv",
                          "content-disposition": f"attachment; filename={attachment or f'ArchivedSchoolAndDistrict{season.title()}{year}.csv'}"},
        parameters={"http_method": "POST", "http_status": 200, "archive_index_record": index,
                    "school_year_end": year, "season": season, "publisher_declared_as_of": date})
    return store, receipt, store.read_record(receipt)["artifact"]["sha256"]


def row(district, campus, name, *, served="'07", site="SCHOOL CITY", district_name="NORTHSIDE ISD"):
    return (district, district_name, campus, name, "INDEPENDENT", "Active", "", "OFFICE MAIL", "OFFICE CITY",
            "SCHOOL MAIL", site, "'015", "BEXAR COUNTY", served, "'05", "'08", "100", "25")


def test_dated_directory_uses_full_parent_context_for_each_campus(tmp_path):
    store, receipt, digest = archive(tmp_path, rows=[
        row("'015915", "015915001", "FIRST HIGH", site=" "),
        row("015915", "015915002", "SECOND HIGH", served="'08", site="ANOTHER CITY"),
        row("015915", "015915002", "SECOND HIGH", served="'08", site="THIRD CITY"),
    ])
    directory = read_askted_directory(store, receipt, year=2026, season="fall", expected_sha256=digest)
    first = directory.campuses["015915001"]
    assert first.value(Field.SITE_CITY) is None and first.value(Field.MAILING_CITY) == "SCHOOL MAIL"
    assert first.observations[0].fields[Field.SITE_CITY] == " "
    assert first.value(Field.ADMINISTRATIVE_COUNTY_NAME) == "BEXAR COUNTY"
    assert first.values(Field.ESC_REGION_SERVED) == ("07", "08")
    assert directory.campuses["015915002"].values(Field.SITE_CITY) == ("ANOTHER CITY", "THIRD CITY")
    assert directory.cohort("campus", field=Field.ESC_REGION_SERVED, value=7).conflicting_ids == (
        "015915001", "015915002",
    )
    cohort = directory.cohort("campus", field=Field.ESC_REGION_PEIMS, value="'5")
    assert cohort.ids == ("015915001", "015915002") and not cohort.conflicting_ids
    assert cohort.source.sha256 == digest and cohort.source.publisher_declared_as_of == "2025-10-31"


def test_aliases_retain_source_occurrences_and_report_distinct_ids(tmp_path):
    store, receipt, digest = archive(tmp_path, rows=[
        row("015915", "015915001", "NEW HIGH"),
        row("244905", "244905001", "SAME HIGH"),
    ])
    directory = read_askted_directory(store, receipt, year=2026, season="fall", expected_sha256=digest)
    old_path = tmp_path / "old"
    old_path.mkdir()
    old_store, old_receipt, old_digest = archive(old_path, year=2025, season="spring", rows=[
        row("015915", "015915001", "SAME HIGH"),
        row("244905", "244905001", "same high"),
        row("244905", "244905001", " Same High "),
    ])
    old = read_askted_directory(old_store, old_receipt, year=2025, season="spring", expected_sha256=old_digest)
    assert directory.resolve_name("campus", "same high").identifier == "244905001"
    with pytest.raises(EntitySelectionError) as error:
        directory.resolve_name("campus", "same high", aliases_from=[old])
    assert error.value.reason == "ambiguous_name"
    assert {candidate[1] for candidate in error.value.candidates} == {"015915001", "244905001"}
    with pytest.raises(EntitySelectionError) as districts:
        directory.resolve_name("district", "northside isd")
    candidates = districts.value.to_dict()["candidates"]
    assert isinstance(candidates, list)
    assert {field: candidates[0][field] for field in ("id", "name", "site_city", "administrative_county")} == {
        "id": "015915", "name": "NORTHSIDE ISD", "site_city": "OFFICE CITY", "administrative_county": "BEXAR COUNTY",
    }
    assert candidates[0]["place_source"]["receipt"] == receipt
    aliases = error.value.to_dict()["candidates"]
    assert isinstance(aliases, list)
    assert len(aliases[1]["name_occurrences"]) == 3
    assert {item["source"]["receipt"] for item in aliases[1]["name_occurrences"]} == {receipt, old_receipt}
    match = directory.resolve_name("campus", "new high", aliases_from=[old])
    assert match.identifier == "015915001" and match.occurrences[0].source.receipt == receipt
    with pytest.raises(ValueError, match="on or before"):
        old.resolve_name("campus", "same high", aliases_from=[directory])


@pytest.mark.parametrize(("options", "requested_season", "message"), [
    ({"date_override": "10/30/2025"}, "fall", "reviewed publisher fall edition"),
    ({"attachment": "ArchivedSchoolAndDistrictSpring2026.csv"}, "fall", "attachment"),
    ({}, "spring", "exactly one selected year and season"),
    ({"rows": [row("015915", "244905001", "WRONG PARENT")]}, "fall", "IDs disagree"),
    ({"rows": [row("", "", "")]}, "fall", "no district identifier"),
    ({"rows": [row("015915", "", "UNKEYED HIGH")]}, "fall", "without a campus identifier"),
    ({"rows": [row("015915", "015915001", "WRONG ESC", served="21")]}, "fall", "publisher code range"),
])
def test_bad_archive_context_fails(tmp_path, options, requested_season, message):
    store, receipt, digest = archive(tmp_path, **options)
    with pytest.raises((ValueError, RuntimeError), match=message):
        read_askted_directory(store, receipt, year=2026, season=requested_season, expected_sha256=digest)
