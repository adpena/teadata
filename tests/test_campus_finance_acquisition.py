from io import BytesIO
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

from openpyxl import Workbook
import pytest

from teadata.acquisition import AcquisitionClient, AcquisitionError
from teadata.provenance import ProvenanceStore
from teadata.scripts import download_campus_peims_financials as downloader
from teadata.scripts.download_campus_peims_financials import acquire_campus_financials


@pytest.fixture(autouse=True)
def recorded_pauses(monkeypatch):
    pauses = []
    monkeypatch.setattr(downloader.time, "sleep", pauses.append)
    return pauses


def _selector(field, options, program, heading):
    options = ''.join(f'<option value="{value}">Label</option>' for value in options)
    return f'<h3>{heading}</h3><input name="_program" value="{program}"><select name="{field}">{options}</select>'.encode()


def _workbook(fiscal_year, campus):
    book = Workbook()
    sheet = book.active
    sheet["A1"] = f"{fiscal_year} PEIMS ACTUAL FINANCIAL DATA BY CAMPUS\nTOTALS FOR CAMPUS ({campus})\nDISTRICT"
    sheet["A3"] = "Total Enrolled Membership: 012"
    for column, label in zip("BCDEFG", ("General\nFund", "%", "Per\nStudent", "All\nFunds", "%", "Per\nStudent"), strict=True):
        sheet[f"{column}5"] = label
    sheet["A6"] = "Expenditures by Object   (Objects 6100-6600)"
    sheet["A11"] = "Expenditures by Function (Objects 6100-6400 Only)"
    sheet["A28"] = "Program expenditures by Program  (Objects 6100-6400 only)"
    sheet["B12"] = -102
    sheet["C12"] = 0.123456789
    sheet["D12"] = "<5"
    sheet["E12"] = "*"
    sheet["B29"] = 0
    result = BytesIO()
    book.save(result)
    return result.getvalue()


class _Response(BytesIO):
    status = 200

    def __init__(self, url, body, filename=None):
        super().__init__(body)
        self.url = url
        self.headers = {} if filename is None else {"Content-Disposition": f"attachment; filename={filename}"}


def _publisher(tmp_path, *, year=2025, roster=("084908104", "084908106"), body=None, attachment=None):
    fiscal_year = f"{year - 1}-{year}"
    suffix = str(year)[-2:]
    campus_program = f"sfadhoc.new_Campus_actual{suffix}.sas"
    roster_program = f"sfadhoc.new_pop_campusbox_actual{suffix}.sas"
    payloads = {campus: _workbook(fiscal_year, campus) for campus in roster}
    calls = []

    def open_request(request, **kwargs):
        url = request.full_url
        calls.append(url)
        selected = urlsplit(url)
        if selected.path.endswith("_new_camp_actual.html"):
            assert selected.path.endswith(f"{str(year - 1)[-2:]}{suffix}_new_camp_actual.html")
            index = _selector("who_list", ("084908", "227825"), roster_program,
                              f"Select District for Financial Actual Campus {fiscal_year} Report:")
            return _Response(url, index)
        args = parse_qs(selected.query)
        if args["_program"] == [roster_program]:
            assert args["who_list"] == ["084908"]
            return _Response(url, _selector("which_camp", roster, campus_program, "Select a Campus:"))
        assert args["_program"] == [campus_program]
        assert args["ptype"] == ["E"]
        campus, = args["which_camp"]
        return _Response(url, payloads[campus] if body is None else body,
                         f"{fiscal_year} PEIMS_Campus_Data.xlsx" if attachment is None else attachment)

    archive = ProvenanceStore(tmp_path / "receipts")
    return AcquisitionClient(archive, opener=SimpleNamespace(open=open_request)), calls, payloads


@pytest.mark.parametrize("year", [2024, 2025])
def test_acquisition_retains_complete_roster_exact_workbook_and_receipt_lineage(tmp_path, year, recorded_pauses):
    client, calls, payloads = _publisher(tmp_path, year=year)
    first = acquire_campus_financials(client, year, ["084908"], ["084908104"])
    district = first["districts"]["084908"]
    assert first["publisher_district_count"] == 2
    assert district["listed_campuses"] == ["084908104", "084908106"]
    assert set(district["workbook_receipts"]) == {"084908104"}
    record_id = district["workbook_receipts"]["084908104"]
    graph = client.store.verify_record_closure([record_id])
    assert set(graph) == {record_id, district["roster_receipt"], first["district_index_receipt"]}
    record = graph[record_id]
    assert record["source_metadata"]["source_qualifiers"]["financial_basis"] == "actual"
    assert record["source_metadata"]["period"] == {"kind": "fiscal_year", "start_year": year - 1, "end_year": year}
    assert client.store.verify_artifact(record["artifact"]["sha256"]).read_bytes() == payloads["084908104"]
    assert len(calls) == 3
    assert recorded_pauses == [0.5, 0.5]
    assert acquire_campus_financials(client, year, ["084908"], ["084908104"]) == first
    assert len(calls) == 3
    assert recorded_pauses == [0.5, 0.5]
    refreshed = acquire_campus_financials(client, year, ["084908"], ["084908104"], refresh=True)
    assert len(calls) == 6
    assert refreshed != first
    assert set(client.store.verify_record_closure([record_id])) == set(graph)
    assert len(list((client.store.root / "objects").iterdir())) == 3


@pytest.mark.parametrize("roster", [("227825004",), ("084908104", "084908104"), ("84908104",), ()])
def test_rejects_malformed_or_wrong_district_roster_before_any_workbook(tmp_path, roster):
    client, calls, _ = _publisher(tmp_path, roster=roster)
    with pytest.raises(AcquisitionError, match="changed TEA which_camp roster"):
        acquire_campus_financials(client, 2025, ["084908"])
    assert len(calls) == 2
    assert len(client.store.history("campus_peims_financials")) == 1


def test_selection_without_zero_padding_or_membership_fails_closed(tmp_path):
    client, calls, _ = _publisher(tmp_path)
    with pytest.raises(ValueError, match="6-digit district IDs"):
        acquire_campus_financials(client, 2025, ["84908"])
    assert not calls
    with pytest.raises(AcquisitionError, match="Campuses absent.*084908107"):
        acquire_campus_financials(client, 2025, ["084908"], ["084908107"])
    assert len(calls) == 2
    with pytest.raises(AcquisitionError, match="Districts absent.*000001"):
        acquire_campus_financials(client, 2025, ["000001"])
    assert len(calls) == 2


@pytest.mark.parametrize("reason", ["wrong_campus", "wrong_edition", "html", "wrong_attachment"])
def test_invalid_workbook_does_not_get_a_success_receipt(tmp_path, reason):
    options = {
        "wrong_campus": {"body": _workbook("2024-2025", "084908106")},
        "wrong_edition": {"body": _workbook("2023-2024", "084908104")},
        "html": {"body": b"<html>200 Error in HTML form</html>"},
        "wrong_attachment": {"attachment": "2025-2026 PEIMS_Campus_Data.xlsx"},
    }
    client, calls, _ = _publisher(tmp_path, **options[reason])
    with pytest.raises(AcquisitionError, match="campus actual workbook|PEIMS campus XLSX attachment"):
        acquire_campus_financials(client, 2025, ["084908"], ["084908104"])
    assert len(calls) == 3
    assert len(client.store.history("campus_peims_financials")) == 2


def test_cached_same_size_corruption_fails_without_a_download(tmp_path):
    client, calls, _ = _publisher(tmp_path)
    first = acquire_campus_financials(client, 2025, ["084908"])
    acquired = first["districts"]["084908"]["workbook_receipts"]
    assert len(acquired) == 2
    record = client.store.read_record(acquired["084908104"])
    archive = client.store.verify_artifact(record["artifact"]["sha256"])
    raw = archive.read_bytes()
    archive.write_bytes(b"!" + raw[1:])
    with pytest.raises(ValueError, match="Archived artifact checksum mismatch"):
        acquire_campus_financials(client, 2025, ["084908"])
    assert len(calls) == 4
