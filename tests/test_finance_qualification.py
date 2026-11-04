from io import BytesIO
from zipfile import ZipFile

import pyarrow as pa
import pytest

from teadata import finance_qualification as finance
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
        main(["--qualify-e3-fy2025", "--store", str(tmp_path), "--output", str(tmp_path / "result"),
              "--source", "district_pwr_actuals", "2025"])
    assert failure.value.code == 2
