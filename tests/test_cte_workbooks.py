from io import BytesIO
import zipfile

from openpyxl import Workbook, load_workbook
import pytest

from teadata.financial_leaves import _package
from teadata.native_cte_workbooks import cte_workbook_projection, read_cte_sheet
from teadata.source_views import _write_excel_rows


def test_cte_workbook_keeps_compound_keys_raw_ids_zero_masks_and_blank(tmp_path):
    book = Workbook()
    sheet = book.active
    sheet.title = "District"
    headers = ["District", "IBC", "Earned"]
    fields = ["district_number", "ibc", "earned"]
    sheet.append(headers)
    sheet.append([1902, "001", 0])
    sheet.append(["001902", "002", -1])
    sheet.append(["001903", "001", None])
    path = tmp_path / "cte.xlsx"
    book.save(path)
    with zipfile.ZipFile(path) as archive:
        sheets, strings = _package(archive, literal=True)
        raw = read_cte_sheet(archive, sheets["District"], strings, headers, fields)
        with pytest.raises(ValueError, match="publisher headers differ"):
            read_cte_sheet(archive, sheets["District"], strings, headers[::-1], fields)
    table, support = cte_workbook_projection(raw)
    assert table["district_number"].to_pylist() == ["'001902", "'001902", "'001903"]
    assert table["raw_district_number"].to_pylist() == ["1902", "001902", "001903"]
    assert table["ibc"].to_pylist() == ["001", "002", "001"]
    assert table["earned"].to_pylist() == ["0", "-1", None]
    assert table["worksheet_row"].to_pylist() == [2, 3, 4]
    assert support["fields"] == {"district_number": {"raw": "raw_district_number"}}


def test_excel_retains_literal_empty_text_as_distinct_from_missing_cell():
    class Provenance:
        def _append_sheet(self, workbook, *, rows=None):
            workbook.create_sheet("Provenance")

        def to_dict(self):
            return {}

    payload = BytesIO()
    _write_excel_rows(Provenance(), (("Missing", "Empty", "Zero"),), 3, 1,
                      iter([(None, "", "0")]), payload)
    payload.seek(0)
    book = load_workbook(payload, read_only=True)
    try:
        assert list(book["Data"].values) == [("Missing", "Empty", "Zero"), (None, "", "0")]
    finally:
        book.close()
