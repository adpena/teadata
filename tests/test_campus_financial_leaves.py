from hashlib import sha256
from io import BytesIO
import json
import os
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pyarrow.parquet as pq
import pytest

from teadata import campus_financial_leaves as finance
from teadata.acquisition import AcquisitionClient
from teadata.financial_leaves import NS, RID
from teadata.provenance import ProvenanceStore
from teadata.scripts import download_campus_peims_financials as downloader
from teadata.scripts.refresh_financials import main


class _Response(BytesIO):
    status = 200

    def __init__(self, url, body, headers):
        super().__init__(body)
        self.url = url
        self.headers = headers


def _workbook(campus, *, damage=None):
    sheet = ET.Element(NS + "worksheet")
    ET.SubElement(sheet, NS + "dimension", ref="A1:G40")
    sheet_data = ET.SubElement(sheet, NS + "sheetData")
    rows = {row: ET.SubElement(sheet_data, NS + "row", r=str(row)) for row in range(1, 42)}
    shared = ET.Element(NS + "sst")

    def cell(coordinate, token, *, literal=False, style=None):
        node = ET.SubElement(rows[int(coordinate[1:])], NS + "c", r=coordinate)
        if literal:
            index = str(len(shared))
            ET.SubElement(ET.SubElement(shared, NS + "si"), NS + "t").text = token
            node.set("t", "s")
            token = index
        if style is not None:
            node.set("s", str(style))
        ET.SubElement(node, NS + "v").text = token

    cell("A1", f"2024-2025 PEIMS ACTUAL FINANCIAL DATA BY CAMPUS\nTOTALS FOR CAMPUS ({campus})\nDISTRICT", literal=True)
    cell("A2", " ", literal=True)
    zero_membership = campus == "084908104"
    cell("A3", f"Total Enrolled Membership: {'0' if zero_membership else '1,023'}", literal=True)
    for column, label, _ in finance._COLUMNS:
        cell(f"{column}5", label, literal=True)
    for section_row, section, labels in finance._SECTIONS:
        cell(f"A{section_row}", section, literal=True)
        for row, label in enumerate(labels, section_row + 1):
            cell(f"A{row}", "Changed label" if damage == "line" and row == 29 else label, literal=True)
            for column, _, role in finance._COLUMNS:
                if damage == "missing" and (column, row) == ("E", 29):
                    continue
                share = role.endswith("_share")
                value = "0" if role.endswith("_per_student") and zero_membership else "1"
                if (column, row) == ("C", 7):
                    value = "0.1234"
                if (column, row) == ("E", 12):
                    value = "-19"
                if (column, row) == ("E", 29):
                    value = "23456"
                style = int(share) ^ (damage == "format" and (column, row) == ("E", 29))
                cell(f"{column}{row}", value, style=style)
    if damage == "tail":
        cell("A41", "Unexpected footer", literal=True)
    merges = ET.SubElement(sheet, NS + "mergeCells")
    for coordinate in sorted(finance._MERGES):
        ET.SubElement(merges, NS + "mergeCell", ref=coordinate)
    styles = ET.Element(NS + "styleSheet")
    formats = ET.SubElement(styles, NS + "numFmts")
    xfs = ET.SubElement(styles, NS + "cellXfs")
    for index, code in (("164", finance._AMOUNT_FORMAT), ("165", finance._SHARE_FORMAT)):
        ET.SubElement(formats, NS + "numFmt", numFmtId=index, formatCode=code)
        ET.SubElement(xfs, NS + "xf", numFmtId=index)
    workbook = ET.Element(NS + "workbook")
    ET.SubElement(ET.SubElement(workbook, NS + "sheets"), NS + "sheet",
                  {"name": finance._SHEET, RID: "rId1"})
    relationships = ET.Element("Relationships")
    ET.SubElement(relationships, "Relationship", Id="rId1", Target="worksheets/sheet1.xml")
    payload = BytesIO()
    with ZipFile(payload, "w") as archive:
        for part, document in {
            "xl/workbook.xml": workbook, "xl/_rels/workbook.xml.rels": relationships,
            "xl/worksheets/sheet1.xml": sheet, "xl/sharedStrings.xml": shared, "xl/styles.xml": styles,
        }.items():
            archive.writestr(part, ET.tostring(document))
    return payload.getvalue()


def _intake(tmp_path, monkeypatch, *, damage=None):
    monkeypatch.setattr(downloader.time, "sleep", lambda _: None)
    store = ProvenanceStore(tmp_path / "provenance")

    def response(request, **kwargs):
        url = request.full_url
        args = parse_qs(urlsplit(url).query)
        headers = {}
        if not args:
            body = ('<h3>Select District for Financial Actual Campus 2024-2025 Report:</h3>'
                    '<input name="_program" value="sfadhoc.new_pop_campusbox_actual25.sas">'
                    '<select name="who_list"><option value="084908">Hitchcock</option></select>').encode()
        elif "who_list" in args:
            body = ('<h2>Select a Campus:</h2><input name="_program" value="sfadhoc.new_Campus_actual25.sas">'
                    '<select name="which_camp"><option value="084908104">First</option>'
                    '<option value="084908106">Second</option></select>').encode()
        else:
            campus, = args["which_camp"]
            body = _workbook(campus, damage=damage if campus == "084908106" else None)
            headers = {"Content-Disposition": "attachment; filename=2024-2025 PEIMS_Campus_Data.xlsx"}
        return _Response(url, body, headers)

    client = AcquisitionClient(store, opener=SimpleNamespace(open=response))
    manifest = downloader.acquire_campus_financials(client, 2025, ["084908"])
    return store, manifest


def _write_intake(tmp_path, manifest):
    path = tmp_path / "intake.json"
    source = json.dumps(manifest, sort_keys=True).encode()
    path.write_bytes(source)
    return path, sha256(source).hexdigest()


def test_campus_producer_preserves_exact_source_lexemes_and_unqualified_ratios(tmp_path, monkeypatch, capsys):
    store, manifest = _intake(tmp_path, monkeypatch)
    intake, pin = _write_intake(tmp_path, manifest)
    output = tmp_path / "raw-candidate"
    assert main(["--store", str(store.root), "--campus-intake", str(intake), "--campus-intake-sha256", pin,
                 "--output", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert json.loads((output / "MATERIALIZATION.json").read_bytes()) == result
    assert result["rows"] == {"campus_reports": 2, "campus_financial_facts": 384}
    assert result["source_projection_readback_equal"] is True
    assert result["field_correctness_qualified"] is False and result["release_activation"] is False
    assert result["lexemes"]["zero_membership_per_student_holds"] == 64
    assert result["lexemes"]["negative_lexemes"] == 2
    assert result["lexemes"]["decimal_lexemes"] == 2
    assert result["fiscal_period"] == {"kind": "fiscal_year", "start_year": 2024, "end_year": 2025}
    assert result["financial_basis"] == "actual" and result["reference_urls"]["raw_account_dictionary"].startswith("https://tea.texas.gov/")
    reports = pq.read_table(output / "campus_reports.parquet").to_pylist()
    assert [(row["campus_number"], row["reported_membership"], row["membership_raw"]) for row in reports] == [
        ("084908104", 0, "Total Enrolled Membership: 0"), ("084908106", 1023, "Total Enrolled Membership: 1,023"),
    ]
    facts = {(row["campus_number"], row["source_cell"]): row
             for row in pq.read_table(output / "campus_financial_facts.parquet").to_pylist()}
    function, program = facts["084908104", "E12"], facts["084908104", "E29"]
    assert function["line_label"] == program["line_label"] == "Total Operating Expenditures"
    assert function["line_occurrence"] == program["line_occurrence"] == 1
    assert function["section_label"] != program["section_label"]
    assert (function["raw_value"], program["raw_value"]) == ("-19", "23456")
    assert function["source_row"] == 12 and function["column_role"] == "all_funds_amount"
    assert facts["084908104", "C7"]["raw_value"] == "0.1234"
    assert facts["084908104", "G29"]["raw_value"] == "0"
    assert facts["084908104", "G29"]["normalization_hold_reason"] == "zero_reported_membership"
    assert facts["084908106", "G29"]["normalization_hold_reason"] == "unqualified_membership_denominator"
    assert facts["084908104", "C7"]["normalization_hold_reason"] == "unqualified_field_semantics"
    receipt_id = result["processing_record_id"]
    graph = store.verify_record_closure([receipt_id])
    assert len(graph) == 6
    assert result["input_receipt_count"] == 4
    assert graph[receipt_id]["outputs"] == result["outputs"]
    assert reports[0]["source_receipt_id"] == function["source_receipt_id"] == manifest["districts"]["084908"]["workbook_receipts"]["084908104"]
    with pytest.raises(FileExistsError, match="exclusive"):
        finance.materialize_campus_financial_workbooks(store, intake=intake, intake_sha256=pin, output=output)


@pytest.mark.parametrize("damage,expected", [("line", "line label"), ("missing", "value or format"),
                                               ("tail", "unaccounted"), ("format", "value or format")])
def test_changed_workbook_fails_before_candidate_or_processing_receipt(tmp_path, monkeypatch, damage, expected):
    store, manifest = _intake(tmp_path, monkeypatch, damage=damage)
    intake, pin = _write_intake(tmp_path, manifest)
    output = tmp_path / "candidate"
    records = set((store.root / "records").iterdir())
    with pytest.raises(ValueError, match=f"Campus 084908106: .*{expected}"):
        finance.materialize_campus_financial_workbooks(store, intake=intake, intake_sha256=pin, output=output)
    assert not output.exists() and not list(tmp_path.glob(".campus-financials-*"))
    assert set((store.root / "records").iterdir()) == records


def test_pinned_receipt_archive_requires_complete_publisher_roster(tmp_path, monkeypatch):
    store, manifest = _intake(tmp_path, monkeypatch)
    scope = manifest["districts"]["084908"]
    scope["listed_campuses"].pop()
    scope["workbook_receipts"].pop("084908106")
    intake, pin = _write_intake(tmp_path, manifest)
    output = tmp_path / "candidate"
    with pytest.raises(ValueError, match="publisher's roster"):
        finance.materialize_campus_financial_workbooks(store, intake=intake, intake_sha256=pin, output=output)
    assert not output.exists()


def test_roster_parser_uses_verified_bytes_without_reopening_archive_path(tmp_path, monkeypatch):
    store, manifest = _intake(tmp_path, monkeypatch)
    receipt = manifest["districts"]["084908"]["roster_receipt"]
    archive = store.verify_artifact(store.read_record(receipt)["artifact"]["sha256"])
    original_read = finance._read_bounded

    def read_then_replace(path, digest, size=None):
        data = original_read(path, digest, size)
        if path == archive:
            archive.write_bytes(b"<html>Changed after the verified read</html>")
        return data

    monkeypatch.setattr(finance, "_read_bounded", read_then_replace)
    workbooks = finance._admit_intake(store, json.dumps(manifest).encode())
    assert [workbook.campus for workbook in workbooks] == ["084908104", "084908106"]
    assert b"Changed" in archive.read_bytes()


def test_pinned_intake_and_archived_payload_are_verified(tmp_path, monkeypatch):
    store, manifest = _intake(tmp_path, monkeypatch)
    intake, pin = _write_intake(tmp_path, manifest)
    output = tmp_path / "candidate"
    with pytest.raises(ValueError, match="pinned size or SHA-256"):
        finance.materialize_campus_financial_workbooks(store, intake=intake, intake_sha256="0" * 64, output=output)
    receipt = manifest["districts"]["084908"]["workbook_receipts"]["084908104"]
    archive = store.verify_artifact(store.read_record(receipt)["artifact"]["sha256"])
    original = archive.read_bytes()
    archive.write_bytes(b"!" + original[1:])
    with pytest.raises(ValueError, match="pinned size or SHA-256"):
        finance.materialize_campus_financial_workbooks(store, intake=intake, intake_sha256=pin, output=output)
    assert not output.exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO support requires POSIX")
def test_intake_requires_a_regular_file_without_waiting_for_fifo_writers(tmp_path):
    fifo = tmp_path / "intake.json"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular file"):
        finance.materialize_campus_financial_workbooks(ProvenanceStore(tmp_path / "archive"),
            intake=fifo, intake_sha256="0" * 64, output=tmp_path / "candidate")
