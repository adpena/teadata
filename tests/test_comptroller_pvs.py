from datetime import datetime, timezone

import pytest

from teadata import comptroller_pvs
from teadata.comptroller_pvs import _county_rows, _index_counties, read_comptroller_county_roster
from teadata.dataset_metadata import DataPeriod, DatasetMetadata, ReferenceDocument, ResolvedDataset
from teadata.provenance import ProvenanceStore


INDEX = "https://comptroller.texas.gov/taxes/property-tax/pvs/2025f/index.php"
REPORTS = "/auto-data/PT2/PVS/2025F/"
TITLE = ("<h1>2025 Final Findings</h1>"
         "<h1>SCHOOL DISTRICT PROPERTY VALUE STUDY<br>(GOV’T. CODE §403.302)</h1>"
         "<h1>2025 Final Findings</h1>")
NAMES = {"188": "Potter", "191": "Randall"}
INDEX_TITLE = ("<h1>School District Property Value Study<br>Government Code Section 403.302<br>"
               "2025 Final Findings</h1><h2>School District Findings by County</h2>")


def _district(county, label, number="188901"):
    return (f"<h2>{number[:3]}-{number[3:]} {label}</h2>"
            f"<a href={REPORTS}{county}{number}1D.php>School District Summary Worksheet</a>")


POTTER = "<h2>188 Potter County</h2>" + _district("188", "Amarillo ISD (split district)")
RANDALL = "<h2>191 Randall County</h2>" + _district("191", "Amarillo ISD (Randall label)")


@pytest.fixture
def capture(tmp_path, monkeypatch):
    monkeypatch.setattr(comptroller_pvs, "_PAGE_CODES", frozenset({"151", "188"}))
    store = ProvenanceStore(tmp_path / "provenance")

    def record(url, html, parents=(), status=200, references=()):
        path = tmp_path / "source.html"
        path.write_text(f"<main>{html}</main>", encoding="utf-8")
        return store.record_acquisition(
            path, ResolvedDataset("reference:comptroller_pvs_county_rosters", "reference", 2025, 2025, "",
                                  DatasetMetadata(title="Study", publisher="Texas Comptroller of Public Accounts",
                                                  version="2025-final", period=DataPeriod("calendar_year", year=2025),
                                                  source_url=INDEX, provenance_ids=parents, references=references)),
            downloaded_at=datetime(2026, 11, 9, tzinfo=timezone.utc), requested_url=url, resolved_url=url,
            parameters={"http_method": "GET", "http_status": status},
        )

    index = record(INDEX, (INDEX_TITLE + '<ul class="three-columns">'
                           f"<li><a href={REPORTS}151index.php>Loving</a></li>"
                           f"<li><a href={REPORTS}188index.php>Potter-Randall</a></li></ul><h2>Resources</h2>"))
    loving_url = "https://comptroller.texas.gov" + REPORTS + "151index.php"
    loving = TITLE + "<h2>151 Loving County</h2>" + _district("151", "Wink-Loving ISD", "248902")
    pages = {"151": record(loving_url, loving, (index,)),
             "188": record("https://comptroller.texas.gov" + REPORTS + "188index.php",
                           TITLE + POTTER + RANDALL, (index,))}
    return store, index, pages, record, loving_url, loving


def test_2025_final_roster_keeps_two_counties_and_local_labels_for_one_district(capture):
    store, index, pages, *_ = capture
    roster = read_comptroller_county_roster(store, index_receipt=index, county_page_receipts=pages)
    assert roster.period == DataPeriod("calendar_year", year=2025)
    assert roster.index_receipt == index
    assert roster.for_county("151")[0].district_id == "248902"
    assert roster.for_district("'015913") == ()
    potter, randall = roster.for_district(188901)
    assert (potter.county_code, potter.county_name, potter.source_district_label) == (
        "188", "Potter", "Amarillo ISD (split district)")
    assert (randall.county_code, randall.county_name, randall.source_district_label) == (
        "191", "Randall", "Amarillo ISD (Randall label)")
    assert randall.worksheet_url == "https://comptroller.texas.gov" + REPORTS + "1911889011D.php"
    assert potter.county_page_receipt == randall.county_page_receipt == pages["188"]


@pytest.mark.parametrize("malformed", [
    POTTER.replace("Potter County", "Wrong County") + RANDALL,
    POTTER + POTTER.replace("188901", "188902").replace("188-901", "188-902") + RANDALL,
    POTTER + RANDALL.replace("1911889011D.php", "1881889011D.php"),
    POTTER + RANDALL.replace("1911889011D.php", "1911889021D.php"),
    POTTER + RANDALL.replace("2025F", "2025P"),
    POTTER + RANDALL.split("<a", 1)[0],
    POTTER + RANDALL + _district("191", "Repeated district"),
    POTTER + RANDALL.replace("<a href=", "<a href=https://example.com href="),
    POTTER,
])
def test_county_page_rejects_ambiguous_or_incomplete_heading_link_pairs(malformed):
    with pytest.raises(ValueError):
        _county_rows(f"<main>{TITLE}{malformed}</main>", NAMES, "r")


def test_county_and_index_ignore_unrelated_footer_and_reject_relocated_source_links():
    report = f"<main>{TITLE}{POTTER}{RANDALL}</main>"
    assert len(_county_rows(report + "<footer><a href=/other>Help</a></footer>", NAMES, "r")) == 2
    with pytest.raises(ValueError, match="does not match"):
        _county_rows(report + "<footer>" + _district("191", "Other ISD", "191901") + "</footer>", NAMES, "r")
    with pytest.raises(ValueError, match="outside the county list"):
        _index_counties(f"<main>{INDEX_TITLE}<ul class=three-columns></ul>"
                        f"<a href={REPORTS}188index.php>Potter-Randall</a></main>")
    with pytest.raises(ValueError, match="direct link entries"):
        _index_counties(f"<main>{INDEX_TITLE}<ul class=three-columns><li><aside><ul>"
                        f"<li><a href={REPORTS}188index.php>Potter-Randall</a></li></ul></aside></li></ul></main>")


def test_county_page_rejects_sidebar_rows_and_orphan_details():
    sidebar = "<aside>" + _district("191", "Other ISD", "191901") + "</aside>"
    with pytest.raises(ValueError, match="does not match"):
        _county_rows(f"<main>{TITLE}{POTTER}{RANDALL}{sidebar}</main>", NAMES, "r")
    detail = f"<a href={REPORTS}1911919011L.php>Deduction Detail</a>"
    with pytest.raises(ValueError, match="Orphan Comptroller"):
        _county_rows(f"<main>{TITLE}{POTTER}{RANDALL}{detail}</main>", NAMES, "r")


def test_county_report_elements_cannot_come_from_inert_or_ancillary_content():
    report = f"<main>{TITLE}{POTTER}{RANDALL}</main>"
    with pytest.raises(ValueError, match="main report boundary"):
        _county_rows(f"<template>{report}</template>", NAMES, "r")
    with pytest.raises(ValueError, match="crosses a sidebar"):
        _county_rows(report.replace("School District Summary Worksheet",
                                    "<aside>School District Summary Worksheet</aside>", 1), NAMES, "r")
    with pytest.raises(ValueError, match="does not match"):
        _county_rows(f"<main>{TITLE}<aside></footer>{POTTER}</aside>{RANDALL}</main>", NAMES, "r")


def test_county_roster_requires_complete_distinct_inputs(capture):
    store, index, pages, *_ = capture
    with pytest.raises(ValueError, match="all 253"):
        read_comptroller_county_roster(store, index_receipt=index, county_page_receipts={"188": pages["188"]})
    with pytest.raises(ValueError, match="distinct"):
        read_comptroller_county_roster(store, index_receipt=index, county_page_receipts=dict.fromkeys(pages, index))


@pytest.mark.parametrize("mutation", ["wrong-url", "no-parent", "wrong-status", "changed-bytes"])
def test_county_roster_rejects_wrong_lineage_and_changed_bytes(capture, mutation):
    store, index, pages, record, url, html = capture
    if mutation == "changed-bytes":
        digest = store.read_record(pages["151"])["artifact"]["sha256"]
        path = store.root / "objects" / digest
        path.write_text(f"<main>{html.replace('Wink-Loving', 'Wink-Wrong!')}</main>", encoding="utf-8")
    else:
        pages["151"] = record(url.replace("151index", "148index") if mutation == "wrong-url" else url,
                              html, () if mutation == "no-parent" else (index,),
                              status=206 if mutation == "wrong-status" else 200)
    with pytest.raises(ValueError, match="direct 2025 Final|checksum mismatch"):
        read_comptroller_county_roster(store, index_receipt=index, county_page_receipts=pages)


def test_county_roster_rejects_an_extra_edge_to_an_already_selected_receipt(capture):
    store, index, pages, record, url, html = capture
    reference = ReferenceDocument("notes", "source_notes", url="https://comptroller.texas.gov" + REPORTS
                                  + "188index.php", provenance_ids=(pages["188"],))
    pages["151"] = record(url, html, (index,), references=(reference,))
    with pytest.raises(ValueError, match="direct 2025 Final"):
        read_comptroller_county_roster(store, index_receipt=index, county_page_receipts=pages)


def test_county_roster_rejects_malformed_transport_metadata(capture):
    store, index, pages, record, url, html = capture
    pages["151"] = record(url, html, (index,), status=200.0)
    with pytest.raises(ValueError, match="direct 2025 Final"):
        read_comptroller_county_roster(store, index_receipt=index, county_page_receipts=pages)
    malformed = store.read_record(pages["151"])
    malformed["downloaded_at"] = "not a timestamp"
    pages["151"] = store._record(malformed)
    with pytest.raises(ValueError, match="acquisition timestamps"):
        read_comptroller_county_roster(store, index_receipt=index, county_page_receipts=pages)
    malformed["downloaded_at"] = "2026-11-09X00:00:00+00:00"
    malformed["parameters"]["http_status"] = 200
    pages["151"] = store._record(malformed)
    with pytest.raises(ValueError, match="direct 2025 Final"):
        read_comptroller_county_roster(store, index_receipt=index, county_page_receipts=pages)
