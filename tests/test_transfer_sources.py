"""Small original-form fixtures; no live TEA requests or current-data oracle."""

from io import BytesIO
from datetime import datetime, timezone
import json
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from teadata.acquisition import NoRedirect, _form_request
from teadata.dataset_metadata import DatasetMetadata, ResolvedDataset
from teadata.provenance import ProvenanceStore
from teadata.transfer_sources import (
    REGIONAL_FORM_URL, acquire_regional_transfers, read_regional_form,
)


FORM = '''<form name="Select_Region_Summary" method="GET" action="/cgi/sas/broker">
<input name="_service" type="hidden" value="marykay">
<input name="_program" type="hidden" value="adhoc.download_transfer_report.sas">
<input name="scope" type="hidden" value="Region">
<input name="agg_level" type="hidden" value="region">
<input name="_debug" type="hidden" value="0">
<select name="school_year"><option value="2025">2024-2025</option>
<option value="2024">2023-2024</option></select>
<select name="report_format"><option value="html">HTML Report</option>
<option value="csv">CSV Report</option></select>
<select name="selected_id"><option value="01">Region I Edinburg
<option value="13">Region XIII Austin</select>
<input name="Region_District_Xfr_Report" type="submit" value="Get a Region District Transfer Report">
</form>'''


def retained_form(tmp_path, text=FORM):
    store = ProvenanceStore(tmp_path / "archive")
    path = tmp_path / "form.html"
    path.write_bytes(text.encode("iso-8859-1"))
    identifier = store.record_acquisition(path, ResolvedDataset(
        "transfer_report_region_form", "reference", None, None, str(path),
        DatasetMetadata(title="Tiny retained selector fixture")),
        downloaded_at=datetime.now(timezone.utc),
        requested_url=REGIONAL_FORM_URL, resolved_url=REGIONAL_FORM_URL,
        response_headers={"Content-Type": "text/html; charset=iso-8859-1"})
    digest = store.read_record(identifier)["artifact"]["sha256"]
    return store, read_regional_form(store, identifier, digest)


class Response(BytesIO):
    status = 200

    def __init__(self, request, *, content_type="text/csv"):
        super().__init__(b"original CSV with unparsed publisher headings\n")
        self.url = request.full_url
        self.headers = {"Content-Type": content_type}


def install_transport(monkeypatch, opened):
    class Opener:
        def open(self, request, timeout):
            assert timeout == 20
            return opened(request)

    monkeypatch.setattr("teadata.transfer_sources.build_opener", lambda handler: Opener())


def test_retained_form_implicit_options_and_exact_selection(tmp_path):
    _, form = retained_form(tmp_path)
    assert list(form.regions.items()) == [("01", "Region I Edinburg"), ("13", "Region XIII Austin")]
    period, regions = form.selection(2025, ["13", "01"])
    assert period.to_dict() == {"kind": "school_year", "start_year": 2024, "end_year": 2025}
    assert regions == ("13", "01")
    assert form.controls(2025, "01")[-4:] == [
        ("school_year", "2025"), ("report_format", "csv"), ("selected_id", "01"),
        ("Region_District_Xfr_Report", "Get a Region District Transfer Report"),
    ]


@pytest.mark.parametrize("year,regions", [(True, ["01"]), (2026, ["01"]),
    (2025, ["1"]), (2025, []), (2025, ["13", "13"])])
def test_unoffered_selection_fails_before_output_or_transport(tmp_path, monkeypatch, year, regions):
    store, form = retained_form(tmp_path)
    calls = []
    install_transport(monkeypatch, lambda request: calls.append(request))
    output = tmp_path / "batch"
    with pytest.raises(ValueError):
        acquire_regional_transfers(store, form, year=year, regions=regions, output=output)
    assert not output.exists() and not calls


@pytest.mark.parametrize("before,after", [
    ('value="region"', 'value="district"'),
    ('method="GET"', 'method="POST"'),
    ('action="/cgi/sas/broker"', 'action="https://example.org/collect"'),
    ('value="2025">2024-2025', 'value="2025">unknown'),
    ('name="selected_id"', 'name="different_id"'),
    ('name="_debug" type="hidden"', 'name="_debug" type="hidden" disabled'),
])
def test_changed_form_contract_is_not_silently_replayed(tmp_path, before, after):
    with pytest.raises(ValueError):
        store, form = retained_form(tmp_path, FORM.replace(before, after))
        acquire_regional_transfers(store, form, year=2025, regions=["01"], output=tmp_path / "batch")


def test_sequential_batch_holds_prefix_and_explicit_resume_skips_only_pinned_region(tmp_path, monkeypatch):
    store, form = retained_form(tmp_path)
    requests = []

    def interrupted(request):
        requests.append(request.full_url)
        if len(requests) == 2:
            raise TimeoutError("second region failed")
        return Response(request)

    install_transport(monkeypatch, interrupted)
    output = tmp_path / "first"
    with pytest.raises(TimeoutError):
        acquire_regional_transfers(store, form, year=2025, regions=["01", "13"], output=output)
    index = json.loads((output / "BATCH_INDEX.json").read_text())
    assert index["status"] == "HOLD"
    assert [item["status"] for item in index["items"]] == ["acquired", "failed"]
    assert len(requests) == 2  # no automatic retry
    identifier = index["items"][0]["record_id"]
    raw = store.verify_artifact(index["items"][0]["artifact"]["sha256"]).read_bytes()
    assert raw.startswith(b"original CSV")
    requests.clear()
    install_transport(monkeypatch, lambda request: (requests.append(request.full_url), Response(request))[1])
    resumed = acquire_regional_transfers(store, form, year=2025, regions=["01", "13"],
        output=tmp_path / "second", resume={"01": identifier})
    assert resumed["status"] == "PASS" and not resumed["field_correctness_qualified"]
    assert [item["status"] for item in resumed["items"]] == ["resumed", "acquired"]
    assert len(requests) == 1 and "selected_id=13" in requests[0]
    with pytest.raises(ValueError, match="exact selected"):
        acquire_regional_transfers(store, form, year=2025, regions=["13"],
            output=tmp_path / "wrong-region", resume={"13": identifier})
    assert not (tmp_path / "wrong-region").exists()


def test_non_csv_original_is_retained_but_rejected_not_qualified(tmp_path, monkeypatch):
    store, form = retained_form(tmp_path)
    install_transport(monkeypatch, lambda request: Response(request, content_type="text/html"))
    output = tmp_path / "batch"
    with pytest.raises(ValueError, match="exact selected"):
        acquire_regional_transfers(store, form, year=2025, regions=["13", "01"], output=output)
    index = json.loads((output / "BATCH_INDEX.json").read_text())
    assert index["status"] == "HOLD"
    assert [item["status"] for item in index["items"]] == ["rejected", "pending"]
    assert store.read_record(index["items"][0]["record_id"])["response_headers"]["content-type"] == "text/html"


def test_keyboard_interrupt_persists_prefix_without_becoming_an_ordinary_failure(tmp_path, monkeypatch):
    store, form = retained_form(tmp_path)
    requests = []

    def interrupted(request):
        requests.append(request.full_url)
        if len(requests) == 2:
            raise KeyboardInterrupt
        return Response(request)

    install_transport(monkeypatch, interrupted)
    output = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        acquire_regional_transfers(store, form, year=2025, regions=["01", "13"], output=output)
    index = json.loads((output / "BATCH_INDEX.json").read_text())
    assert index["status"] == "HOLD" and len(requests) == 2
    assert [item["status"] for item in index["items"]] == ["acquired", "interrupted"]
    assert index["items"][1]["error"]["type"] == "KeyboardInterrupt"
    assert "record_id" not in index["items"][1]


def test_interrupt_during_completion_index_preserves_hold_and_cleans_only_owned_temp(tmp_path, monkeypatch):
    store, form = retained_form(tmp_path)
    install_transport(monkeypatch, Response)
    original_dump = json.dump
    interruption = KeyboardInterrupt("completion serialization interrupted")

    def interrupted_dump(report, stream, *args, **kwargs):
        if report.get("status") == "PASS":
            stream.write('{"partial":')
            raise interruption
        return original_dump(report, stream, *args, **kwargs)

    monkeypatch.setattr("teadata.transfer_sources.json.dump", interrupted_dump)
    output = tmp_path / "completion-interrupted"
    with pytest.raises(KeyboardInterrupt) as caught:
        acquire_regional_transfers(store, form, year=2025, regions=["01", "13"], output=output)
    assert caught.value is interruption
    index = json.loads((output / "BATCH_INDEX.json").read_text())
    assert index["status"] == "HOLD"
    assert [item["status"] for item in index["items"]] == ["acquired", "acquired"]
    assert not list(output.glob(".BATCH_INDEX.*.tmp"))
    assert (output / "BATCH_INDEX.json").stat().st_mode & 0o777 == 0o600


def test_canonical_url_encoding_and_redirect_rejection(tmp_path):
    _, form = retained_form(tmp_path)
    url, body, controls = _form_request("https://rptsvr1.tea.texas.gov/cgi/sas/broker", "GET", form.controls(2025, "01"), None)
    assert body is None and controls[-2] == ("selected_id", "01")
    assert url.endswith("selected_id=01&Region_District_Xfr_Report=Get+a+Region+District+Transfer+Report")
    with pytest.raises(HTTPError, match="redirect blocked"):
        NoRedirect().redirect_request(Request(url), None, 302, "redirect", {}, "https://example.org")
