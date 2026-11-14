"""The retained 2025 campus report has distinct benchmark roles and actual periods.

Publisher cells below were frozen from receipt430eeb…8bfd17, original HTML
9d008bfd…a1634121, campus227901002. The short fixture isolates its source grid.
"""

from decimal import Decimal
from datetime import datetime, timezone
from hashlib import sha256

import pytest

from teadata.attendance_chronic import read_attendance_report
from teadata.manifest import ContractError


_URL = ("https://rptsvr1.tea.texas.gov/cgi/sas/broker?_service=marykay&_program=perfrept.perfmast.sas"
        "&ccyy=2025&id=227901002&lev=C&prgopt=reports/tapr/attendgrad.sas")
_HEADERS = ["", "State", "District", "Campus", "AfricanAmerican", "Hispanic", "White", "AmericanIndian",
            "Asian", "PacificIslander", "Two orMoreRaces", "SpecialEd", "EconDisadv", "EB/EL"]
_ROWS = [
    _HEADERS, ["Attendance Rate"],
    ["2024-25", "93.9%", "92.3%", "94.0%", "93.4%", "92.1%", "95.4%", "*", "95.3%", "*", "93.7%", "91.3%", "91.0%", "89.4%"],
    ["2023-24", "93.6%", "91.3%", "91.9%", "92.7%", "89.8%", "93.4%", "*", "93.6%", "*", "92.4%", "87.6%", "89.8%", "87.7%"],
    ["Chronic Absenteeism"],
    ["2024-25", "18.0%", "23.3%", "16.9%", "20.0%", "24.1%", "11.2%", "*", "16.0%", "*", "16.8%", "28.5%", "29.7%", "35.9%"],
    ["2023-24", "19.0%", "26.4%", "24.3%", "20.3%", "32.8%", "18.5%", "*", "16.7%", "*", "23.4%", "36.9%", "34.6%", "38.6%"],
    ["Indicates there are no students in the group."],
    ["Indicates results are masked due to small numbers to protect student confidentiality."],
    ["Indicates that the data for this item were statistically improbable or were reported outside a reasonable range."],
    ["2024-25 reporting now reflects Attendance and Chronic Absenteeism data from the 2024-25 and 2023-24 school years."],
]


def _report(rows):
    return ("<table>" + "".join("<tr>" + "".join("<td>" + cell + "</td>" for cell in row)
                               + "</tr>" for row in rows) + "</table>").encode("latin-1")


def test_published_local_rates_keep_both_actual_periods_and_masks():
    result = read_attendance_report(_report(_ROWS), scope="campus", identifier="227901002", url=_URL)
    assert set(result) == {2024, 2025}
    assert result[2025]["chronic_absence_all_students_percent"] == Decimal("16.9")
    assert result[2024]["chronic_absence_all_students_percent"] == Decimal("24.3")
    assert result[2025]["attendance_all_students_percent"] == Decimal("94.0")
    assert result[2025]["chronic_absence_american_indian_percent"] is None
    assert result[2025]["attendance_raw_chronic_absence_american_indian_percent"] == "*"
    assert result[2025]["attendance_status_chronic_absence_american_indian_percent"] == "suppressed"
    assert result[2025]["attendance_column_chronic_absence_all_students_percent"] == 4
    assert len([field for field in result[2025] if field.startswith("chronic_absence_")]) == 11


@pytest.mark.parametrize("replacement", ["2025-26", "2023-24"])
def test_source_edition_cannot_replace_or_duplicate_the_actual_year(replacement):
    rows = [list(row) for row in _ROWS]
    rows[5][0] = replacement
    with pytest.raises(ContractError, match="measurement period"):
        read_attendance_report(_report(rows), scope="campus", identifier="227901002", url=_URL)


@pytest.mark.parametrize("token,status", [("0.0%", "observed"), ("-", "not_applicable"), ("?", "invalid")])
def test_named_legend_preserves_zero_absent_group_and_questionable_values(token, status):
    rows = [list(row) for row in _ROWS]
    rows[5][3] = token
    result = read_attendance_report(_report(rows), scope="campus", identifier="227901002", url=_URL)
    assert result[2025]["attendance_status_chronic_absence_all_students_percent"] == status
    assert result[2025]["attendance_raw_chronic_absence_all_students_percent"] == token
    assert result[2025]["chronic_absence_all_students_percent"] == (Decimal(0) if status == "observed" else None)


@pytest.mark.parametrize("token", ["", "**", "100.1%", "16.90%", "16.9"])
def test_unreviewed_token_or_representation_fails(token):
    rows = [list(row) for row in _ROWS]
    rows[5][3] = token
    with pytest.raises(ContractError):
        read_attendance_report(_report(rows), scope="campus", identifier="227901002", url=_URL)


def test_report_identity_and_population_column_change_fail():
    with pytest.raises(ContractError, match="scope, identifier or source edition"):
        read_attendance_report(_report(_ROWS), scope="campus", identifier="227901003", url=_URL)
    rows = [list(row) for row in _ROWS]
    rows[0][-1] = "Other"
    with pytest.raises(ContractError, match="population"):
        read_attendance_report(_report(rows), scope="campus", identifier="227901002", url=_URL)


def test_generated_attendance_crosswalk_covers_every_source_only_field(tmp_path, monkeypatch):
    from teadata import attendance_chronic as source_module
    from teadata.dataset_metadata import DatasetMetadata, ResolvedDataset
    from teadata.manifest import BuildRecorder
    from teadata.provenance import ProvenanceStore
    from teadata.teadata_config import load_config_bytes

    store = ProvenanceStore(tmp_path / "provenance")
    reports = []
    for index in range(21):
        path = tmp_path / f"input-{index}.html"
        path.write_text(f"synthetic source {index}")
        identifier = store.record_acquisition(path, ResolvedDataset("fixture", "reference", 2025, 2025,
            "", DatasetMetadata(version="2025")), downloaded_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            requested_url=f"https://example.org/input-{index}.html")
        reports.append({"scope": "state" if index == 0 else "region", "identifier": "S" if index == 0 else f"{index:02}",
                        "receipt": identifier})
    grid = source_module.definition()
    grid["glossary_receipt"] = reports[0]["receipt"]
    grid["glossary_sha256"] = sha256((tmp_path / "input-0.html").read_bytes()).hexdigest()
    monkeypatch.setattr(source_module, "definition", lambda: grid)
    stage = tmp_path / "stage"
    stage.mkdir()
    raw_config = source_module._configuration(store, stage, {"reports": reports})
    path = stage / "source-config.input"
    path.write_bytes(raw_config)
    config = load_config_bytes(raw_config, path=path)
    selected = config.resolve_dataset(source_module.DATASET, 2025, exact=True, reader_profile="source")
    contract = config.column_contract(source_module.DATASET, 2025, reader_profile="source")
    assert contract is not None and BuildRecorder(config, store).processor_sha256
    closure = store.verify_record_closure(selected.metadata.provenance_ids)
    assert all(closure[identifier]["dataset"] == source_module.DATASET for identifier in selected.metadata.provenance_ids)
    assert all(closure[identifier]["method"] == "local_observation" for identifier in selected.metadata.provenance_ids)
    assert all(closure[identifier]["downloaded_at"] is None for identifier in selected.metadata.provenance_ids)
    originals = {identifier for reference in selected.metadata.references for identifier in reference.provenance_ids
                 if reference.reference_id.startswith("publisher_")}
    assert originals == {item["receipt"] for item in reports}
