import csv
from datetime import UTC, date, datetime

import pytest

from teadata.askted_lifecycle import LifecycleEventKind, LifecycleReportKind, read_lifecycle_report
from teadata.dataset_metadata import DatasetMetadata, ResolvedDataset
from teadata.provenance import ProvenanceStore


PORTAL = "https://tealprod.tea.state.tx.us/Tea.AskTed.Web/Forms/ReportSelection.aspx"
VIEWER = "https://tealprod.tea.state.tx.us/Tea.AskTed.Web/Forms/ReportViewer.aspx"
PRINTED = date(2026, 11, 7)


def _record(tmp_path, *, kind, title, category, sort, details, reported_date="11/7/2026"):
    heading = f"{title} Sorted by {sort}" if sort else title
    path = tmp_path / "crViewer.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for detail in details:
            writer.writerow(["AskTED", heading, heading, *detail, f"Printed Date:  {reported_date}", "Page -1 of 1"])
    metadata = DatasetMetadata(title=f"AskTED {title}", publisher="Texas Education Agency", source_url=PORTAL,
                               download_url=VIEWER, version=f"printed-{PRINTED.isoformat()}")
    source = ResolvedDataset(f"reference:askted_{kind}", "reference", None, None, "", metadata)
    store = ProvenanceStore(tmp_path / "provenance")
    now = datetime(2026, 11, 7, tzinfo=UTC)
    receipt = store.record_acquisition(path, source, downloaded_at=now, recorded_at=now,
        requested_url=VIEWER, resolved_url=VIEWER, parameters={
            "category": category, "sort": sort, "publisher_printed_date": PRINTED.isoformat(),
            "publisher_report_title": title, "report_type": "rpt", "export_format": "CharacterSeparatedValues",
            "range": "all", "export_attachment_name": "crViewer.csv",
        })
    return store, receipt, store.read_record(receipt)["artifact"]["sha256"]


def _campuses(tmp_path, details, **kwargs):
    return _record(tmp_path, kind=LifecycleReportKind.PUBLIC_CAMPUS_CLOSURES, title="All Public Schools Closed",
                   category="school", sort="School Name", details=details, **kwargs)


def test_closure_reader_keeps_repeat_ids_publisher_labels_and_unresolved_dates(tmp_path):
    first = ["School", " (014-906-204)", "Type", "NA", "Dormant Date", "Jan 01, 1800"]
    second = ["School", "Renamed (legacy) (014-906-204)", "Type", "REGULAR", "Obsolete Date", "Jan 08, 2025"]
    store, receipt, digest = _campuses(tmp_path, [first, second, second])
    result = read_lifecycle_report(store, receipt, report=LifecycleReportKind.PUBLIC_CAMPUS_CLOSURES,
                                   expected_sha256=digest)
    assert result.receipt == receipt and result.printed_date == PRINTED
    assert [item.source_record for item in result.events] == [1, 2, 3]
    assert [item.participants[0].identifier for item in result.events] == ["014906204"] * 3
    assert [item.participants[0].name for item in result.events] == ["", "Renamed (legacy)", "Renamed (legacy)"]
    assert [item.kind for item in result.events] == [LifecycleEventKind.DORMANT, LifecycleEventKind.OBSOLETE,
                                                    LifecycleEventKind.OBSOLETE]
    assert result.events[0].event_date is None and result.events[0].raw_date == "Jan 01, 1800"
    assert result.events[1].event_date == date(2025, 1, 8)
    assert result.events[1].raw == result.events[2].raw


def test_reorganization_reader_preserves_roles_and_multiline_csv_records(tmp_path):
    detail = ["County", "NACOGDOCHES COUNTY (174)", "Official Name", "WODEN ISD (174-906)", "Region", "07",
              "Effective ", "Jul 01, 2022", "Annexing District \rAnnexed District ",
              "WODEN ISD (174-906)", "ETOILE ISD (174-910)"]
    second = [*detail[:8], "Consolidated Districts", "WODEN CONS ISD (174-906)", "ETOILE ISD (174-910)"]
    report = LifecycleReportKind.DISTRICT_CONSOLIDATIONS_ANNEXATIONS
    store, receipt, digest = _record(tmp_path, kind=report,
        title="All Public School District Consolidations and Annexations", category="district", sort=None,
        details=[detail, second])
    result = read_lifecycle_report(store, receipt, report=report, expected_sha256=digest)
    annexation, consolidation = result.events
    assert annexation.source_record == 1 and consolidation.source_record == 2
    assert annexation.kind == LifecycleEventKind.ANNEXATION
    assert annexation.date_label == "Effective "
    assert annexation.raw[11] == "Annexing District \rAnnexed District "
    assert [(p.identifier, p.role) for p in annexation.participants] == [
        ("174906", "official_result"), ("174906", "annexing"), ("174910", "annexed")]
    assert [p.role for p in consolidation.participants] == ["official_result", "participant", "participant"]
    assert consolidation.participants[0].identifier == consolidation.participants[1].identifier
    assert consolidation.participants[0].name != consolidation.participants[1].name


@pytest.mark.parametrize(("damage", "error"), [
    ("report", "selected public export"), ("hash", "content pin"),
    ("date", "layout or print date"), ("id", "record 1: Invalid campus"),
])
def test_lifecycle_reader_fails_closed_on_unbound_or_drifted_report(tmp_path, damage, error):
    identifier = "014906204" if damage == "id" else "014-906-204"
    detail = ["School", f"A ({identifier})", "Type", "REGULAR", "Obsolete Date", "Jan 08, 2025"]
    store, receipt, digest = _campuses(tmp_path, [detail],
                                     reported_date="11/8/2026" if damage == "date" else "11/7/2026")
    kind = LifecycleReportKind.PUBLIC_DISTRICT_CLOSURES if damage == "report" else LifecycleReportKind.PUBLIC_CAMPUS_CLOSURES
    with pytest.raises(ValueError, match=error):
        read_lifecycle_report(store, receipt, report=kind, expected_sha256="0" * 64 if damage == "hash" else digest)


def test_charter_reader_requires_its_own_category_and_bounded_object(tmp_path):
    report = LifecycleReportKind.CHARTER_CAMPUS_CLOSURES
    detail = ["School", "Example (014-906-204)", "Type", "REGULAR", "Obsolete Date", "Jan 08, 2025"]
    store, receipt, digest = _record(tmp_path, kind=report, title="All Charter Schools Closed",
                                     category="charter", sort="School Name", details=[detail])
    selection = dict(report=report, expected_sha256=digest)
    assert read_lifecycle_report(store, receipt, **selection).events[0].event_date == date(2025, 1, 8)
    path = store.root / "objects" / digest
    with path.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="object size differs"):
        read_lifecycle_report(store, receipt, **selection)
