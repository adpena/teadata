from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import pyarrow as pa
import pytest

from teadata import finance_qualification as finance
from teadata.provenance import ProvenanceStore
from teadata.scripts.refresh_financials import main


@pytest.mark.parametrize("raw,value", [("0", 0), ("-999", -999), ("-9223372036854775808", -(1 << 63)),
                                        ("9223372036854775807", (1 << 63) - 1)])
def test_amount_preserves_source_zero_and_signed_int64(raw, value):
    assert finance._matched(raw, value, entity="001901", cell="E7") == value


@pytest.mark.parametrize("raw", ["", "-", "<5", "1.0", "1,000", "N/A", "9223372036854775808"])
def test_qualification_refuses_other_representations(raw):
    with pytest.raises(ValueError, match="integer lexeme|int64"):
        finance._matched(raw, 0, entity="001901", cell="E7")


def test_mismatch_holds_instead_of_replacing_original_amount():
    with pytest.raises(ValueError, match="Publisher finance witness differs for 001901001 at E7"):
        finance._matched("0", 19, entity="001901001", cell="E7")


def _page(district="001901", *, object_header=True):
    return "\n".join([finance._PDF_TITLE, f"Campus Number:{district}001 Total Membership: 0",
        "General Per All Per", "Fund % Student Funds % Student",
        finance._PDF_OBJECT if object_header else "Different section",
        "Total Expenditures 4 5.00 6 -9 8.00 0", finance._PDF_FUNCTION,
        "Total Expenditures 11 12.00 13 14 15.00 16"])


def test_campus_pdf_reads_only_all_funds_object_amount():
    assert finance._pdf_amounts(_page(), "001901") == {"001901001": (1, -9)}
    with pytest.raises(ValueError, match="missing a selected object total"):
        finance._pdf_amounts(_page(object_header=False), "001901")
    with pytest.raises(ValueError, match="fund columns or campus grain"):
        finance._pdf_amounts(_page() + "\f" + _page(), "001901")
    with pytest.raises(ValueError, match="selected FY2025 district"):
        finance._pdf_amounts(_page("001902"), "001901")


def _district_html(district_columns=6):
    headers = ("", "General Fund", "%", "Per Student", "All Funds", "%", "Per Student", "All Funds", "%", "Per Student")
    numbers = (finance._PIC38, "$1", "2.0", "$3", "$-19", "5.0", "$6", "$7", "8.0", "$9")
    return (f'<td class="c systemtitle">2024 - 2025 Actual Financial Data Totals for EXAMPLE (001901)</td>'
            f'<table class="table"><tr><th></th><th colspan="{district_columns}">District</th><th colspan="3">State</th></tr>'
            '<tr>' + ''.join(f'<th>{name}</th>' for name in headers) + '</tr>\n'
            '<tr>' + ''.join(f'<td>{item}</td>' for item in numbers) + '</tr></table>').encode()


def test_district_html_uses_only_district_all_funds_amount():
    assert finance._district_html_amount(_district_html(), "001901") == (2, -19)
    with pytest.raises(ValueError, match="District and State column groups"):
        finance._district_html_amount(_district_html(3), "001901")


def _pwr(heading, raw):
    ns = finance.NS.strip("{}")
    contents = BytesIO()
    with ZipFile(contents, "w") as book:
        book.writestr('xl/workbook.xml', f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="24_25 AF Act" r:id="rId1"/></sheets></workbook>')
        book.writestr('xl/_rels/workbook.xml.rels', '<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        book.writestr('xl/sharedStrings.xml', f'<sst xmlns="{ns}"><si><t>DISTRICT NUMBER</t></si><si><t>{heading}</t></si></sst>')
        book.writestr('xl/worksheets/sheet1.xml', f'<worksheet xmlns="{ns}"><sheetData><row r="1"><c r="C1" t="s"><v>0</v></c><c r="BN1" t="s"><v>1</v></c></row><row r="2"><c r="C2"><v>1901</v></c><c r="BN2"><v>{raw}</v></c></row></sheetData></worksheet>')
    return contents.getvalue()


def test_pwr_uses_bn_header_and_preserves_original_dollar_lexeme():
    assert finance._pwr_amounts(_pwr("AF " + finance._PIC38, "0"), {"001901"}) == {"001901": ("BN2", "0")}
    with pytest.raises(ValueError, match="PIC 38 header differs"):
        finance._pwr_amounts(_pwr("GF " + finance._PIC38, "0"), {"001901"})
    with pytest.raises(ValueError, match="exact selected district cohort"):
        finance._pwr_amounts(_pwr("AF " + finance._PIC38, "0"), {"001902"})


def test_district_receipt_requires_exact_media_and_year():
    receipt = {"source_metadata": {"version": "2025"}, "requested_year": None, "resolved_year": None,
               "response_headers": {"content-type": "text/html; charset=iso-8859-1"}}
    finance._district_edition(receipt, "text/html")
    with pytest.raises(ValueError, match="edition or media type"):
        finance._district_edition({**receipt, "resolved_year": 2024}, "text/html")
    with pytest.raises(ValueError, match="edition or media type"):
        finance._district_edition(receipt, "application/pdf")


def test_leaf_binds_one_exact_amount_with_raw_status_and_witness_support():
    selection = finance._Selection("district_pwr_actuals", "district", "pic38", "BN1", "label", "C1", "notes",
        "https://example.test", ("source",), ("witness",), {
            "001901": finance._Amount("-999", -999, "source", "24_25 AF Act", "BN2", "witness", "line 4"),
            "001902": finance._Amount("0", 0, "source", "24_25 AF Act", "BN3", "witness2", "line 4"),
        })
    table, support = finance._leaf_table(selection)
    assert table["pic38"].type == pa.int64() and table["pic38"].to_pylist() == [-999, 0]
    assert table["raw_value"].to_pylist() == ["-999", "0"] and table["value_status"].to_pylist() == ["observed"] * 2
    assert support["fields"] == {"pic38": {"raw": "raw_value", "status": "value_status"}}
    assert set(table.column_names) == {"district_number", "pic38", "raw_value", "value_status", *finance._CONTEXT}


def test_qualified_finance_cli_refuses_other_source_arguments(tmp_path):
    with pytest.raises(SystemExit) as failure:
        main(["--qualification-case", str(tmp_path / "case.json"), "--qualification-case-sha256", "1" * 64,
              "--store", str(tmp_path), "--output", str(tmp_path / "result"),
              "--source", "district_pwr_actuals", "2025"])
    assert failure.value.code == 2


def _case():
    return {"schema_version": 1, "fiscal_end_year": 2025,
            **{name: f"{number:064x}" for number, name in enumerate(sorted(finance._CASE_IDENTITIES), 1)},
            "districts": {"001901": {"campus_pdf": "a" * 64, "district_html": "b" * 64}}}


def test_qualification_accepts_an_explicit_single_district_selection():
    case = _case()
    assert finance._qualification_case(json.dumps(case).encode()) == case


@pytest.mark.parametrize("damage", ["duplicate", "wrong_year", "unpadded", "missing_witness", "invalid_pin"])
def test_qualification_case_rejects_ambiguous_or_unbound_input(damage):
    case = _case()
    if damage == "wrong_year":
        case["fiscal_end_year"] = 2024
    if damage == "unpadded":
        case["districts"]["1901"] = case["districts"].pop("001901")
    if damage == "missing_witness":
        del case["districts"]["001901"]["campus_pdf"]
    if damage == "invalid_pin":
        case["cohort_intake_sha256"] = "not-an-identity"
    raw = json.dumps(case)
    if damage == "duplicate":
        raw = '{"schema_version": 1, ' + raw[1:]
    with pytest.raises(ValueError):
        finance._qualification_case(raw.encode())


def test_qualification_rejects_wrong_digest_before_opening_witnesses(tmp_path):
    case = tmp_path / "case.json"
    case.write_text(json.dumps(_case()))
    output = tmp_path / "output"
    store = ProvenanceStore(tmp_path / "store")
    with pytest.raises(ValueError, match="pinned size or SHA-256"):
        finance.qualify_financial_actuals(store, case_path=case, case_sha256="0" * 64, output=output)
    assert not output.exists()
    assert not store.root.exists()


def test_qualification_cli_requires_and_forwards_explicit_case_and_digest(tmp_path, monkeypatch):
    case = tmp_path / "case.json"
    case.write_text(json.dumps(_case()))
    digest = sha256(case.read_bytes()).hexdigest()
    output = tmp_path / "output"
    store = tmp_path / "store"
    args = ["--store", str(store), "--output", str(output), "--qualification-case", str(case)]
    with pytest.raises(SystemExit) as failure:
        main(args)
    assert failure.value.code == 2

    def qualify(selected_store, *, case_path, case_sha256, output):
        assert selected_store.root == store and case_path == case and case_sha256 == digest
        assert output == tmp_path / "output"
        return {"release_activation": False}

    monkeypatch.setattr(finance, "qualify_financial_actuals", qualify)
    assert main([*args, "--qualification-case-sha256", digest]) == 0


def test_qualified_finance_install_reserves_name_and_exposes_manifest_last(tmp_path, monkeypatch):
    stage = tmp_path / "staged"
    stage.mkdir()
    (stage / "release.json").write_text("manifest")
    (stage / "leaf.parquet").write_text("leaf")
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError):
        finance._install_candidate(stage, output)
    assert list(output.iterdir()) == []
    assert len(list(stage.iterdir())) == 2
    output.rmdir()
    rename = Path.rename

    def observe(source, destination):
        assert not (output / "release.json").exists()
        if source.name == "release.json":
            assert (output / "leaf.parquet").is_file()
        return rename(source, destination)

    monkeypatch.setattr(Path, "rename", observe)
    finance._install_candidate(stage, output)
    assert (output / "release.json").read_text() == "manifest"
