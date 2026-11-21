from shutil import copy2
from unittest.mock import Mock

import pytest

from teadata import geography_rosters as rosters
from teadata.askted_directory import AskTEDDirectory, DirectoryEntry, DirectoryField, DirectoryObservation, DirectorySource
from teadata.dataset_metadata import DataPeriod
from teadata.legislative_rosters import LegislativeDistrictSchool, LegislativePlan, LegislativeSchoolRoster, RosterSource


def test_geography_rosters_preserve_scope_source_rows_and_unassigned_districts(tmp_path, monkeypatch):
    served, peims, geographic = (DirectoryField.ESC_REGION_SERVED, DirectoryField.ESC_REGION_PEIMS,
                                DirectoryField.ESC_REGION_GEOGRAPHIC)

    def observation(record, identifier, name, region):
        return DirectoryObservation(record, identifier, identifier, {
            DirectoryField.NAME: name, served: region, peims: "02", geographic: "03",
        })

    directory = AskTEDDirectory(DirectorySource(2026, "spring", "2026-03-31", "archive", "index", "directory-sha"), {
        "001901": DirectoryEntry("001901", (observation(2, "001901", "A ISD", "1"),)),
        "001902": DirectoryEntry("001902", (observation(3, "001902", "B ISD", "01"),
                                            observation(4, "001902", "B ISD", "02"))),
        "001903": DirectoryEntry("001903", (observation(5, "001903", "C ISD", ""),)),
    }, {})
    house = LegislativePlan.TEXAS_HOUSE
    locator, workbook = RosterSource("locator", "locator-sha"), RosterSource("workbook", "workbook-sha")
    legislature = LegislativeSchoolRoster(house, DataPeriod("school_year", 2025, 2026), 2020, workbook, locator, (
        LegislativeDistrictSchool(1, "001901", "A ISD (100%)", 12),
        LegislativeDistrictSchool(2, "001901", "A ISD (0%)", 71),
    ))
    store = Mock()
    store.read_record.side_effect = lambda receipt: {
        "artifact": {"sha256": receipt + "-sha"}, "requested_url": "https://example.org/" + receipt,
        "downloaded_at": "2026-11-20T00:00:00+00:00",
        "source_metadata": {"publisher": "Test", "provenance_ids": ["catalog"],
                            **({"version": receipt} if receipt != "index" else {})},
    }
    monkeypatch.setattr(rosters, "read_askted_directory", lambda *args, **kwargs: directory)
    monkeypatch.setattr(rosters, "read_legislative_school_rosters", lambda *args, **kwargs: (legislature,))
    artifact = tmp_path / "rosters.sqlite3"
    kwargs = {"directory": RosterSource("archive", "directory-sha"), "year": 2026, "season": "spring",
              "district_locator": locator, "workbooks": {house: workbook}, "output": artifact}
    digest = rosters.build_geography_rosters(store, **kwargs)
    with rosters.open_geography_rosters(artifact, expected_sha256=digest) as reader:
        result = reader.get(served, 1)
        assert result.geography_number == "01"
        assert result.citation.period == DataPeriod("as_of", as_of="2026-03-31")
        assert [source["receipt"] for source in result.citation.acquisitions] == ["archive", "index"]
        assert result.citation.acquisitions[1]["edition"] is None
        assert result.members == (rosters.GeographyRosterMember("001901", "A ISD", (2,)),)
        assert reader.get(served, 2).members == ()
        assert reader.directory_issues(served) == (
            rosters.GeographyRosterIssue("001902", "conflict", ("01", "02"), (3, 4)),
            rosters.GeographyRosterIssue("001903", "missing", (), (5,)),
        )
        assert [member.district_number for member in reader.get(peims, 2).members] == ["001901", "001902", "001903"]
        legislative = reader.get(house, 2)
        assert legislative.citation.census_year == 2020
        assert legislative.members == (rosters.GeographyRosterMember("001901", "A ISD (0%)", (71,)),)
        with pytest.raises(KeyError, match="unavailable"):
            reader.get(LegislativePlan.US_HOUSE_C2193, 2)
        with pytest.raises(ValueError, match="positive geography number"):
            reader.get(served, True)
    with pytest.raises(FileExistsError):
        rosters.build_geography_rosters(store, **kwargs)
    with pytest.raises(ValueError, match="SHA-256"):
        with rosters.open_geography_rosters(artifact, expected_sha256="0" * 64):
            pytest.fail("An unpinned roster file was accepted")
    journal = artifact.with_suffix(".sqlite3-wal")
    journal.touch()
    with pytest.raises(ValueError, match="checkpointed"):
        with rosters.open_geography_rosters(artifact, expected_sha256=digest):
            pytest.fail("A database with sidecars was accepted")
    journal.unlink()
    replacement = tmp_path / "replacement.sqlite3"
    copy2(artifact, replacement)
    opener = rosters.readonly_connection

    def replace_then_open(*args, **kwargs):
        replacement.replace(artifact)
        return opener(*args, **kwargs)

    monkeypatch.setattr(rosters, "readonly_connection", replace_then_open)
    with pytest.raises(ValueError, match="changed while opening"):
        with rosters.open_geography_rosters(artifact, expected_sha256=digest):
            pytest.fail("A different SQLite file was accepted")
