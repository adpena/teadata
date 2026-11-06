import json
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.parse import urlencode

import pytest

from teadata.manifest import ContractError
from teadata.provenance import ProvenanceStore
from teadata.scripts import acquire_tprs_campuses as campus
from teadata.tprs_statewide_intake import validate_campus_selection


def test_rate_limit_persists_across_region_and_process_boundaries(tmp_path, monkeypatch):
    store = ProvenanceStore(tmp_path)
    monkeypatch.setattr(campus, "campus_roster", lambda *args: ("001901001",))
    monkeypatch.setattr(campus, "time", type("Clock", (), {"time": staticmethod(lambda: 1000.0)}))
    calls = []

    def rate_limit(*args, **kwargs):
        pace = json.loads((tmp_path / "tprs-campus-broker-pace.json").read_bytes())
        assert pace == {"not_before": 1008}
        calls.append(1)
        raise HTTPError(campus.BROKER, 429, "Too Many Requests", {"Retry-After": "900"}, None)

    monkeypatch.setattr(campus, "acquire_form", rate_limit)
    first = campus.acquire_region(store, "01", "a" * 64)
    later = campus.acquire_region(store, "02", "b" * 64)
    assert first["stop"] == "HTTP 429" and first["requested"] == 1
    assert json.loads((tmp_path / "tprs-campus-broker-pace.json").read_bytes()) == {"not_before": 1900}
    assert later["stop"] == "cooldown" and later["requested"] == 0 and calls == [1]


def test_retained_complete_region_seals_same_intake_without_requests(tmp_path, monkeypatch):
    store = ProvenanceStore(tmp_path)
    key = "001901001"
    pin = "a" * 64
    selected = campus.ReportSelection("campus", key, pin, "b" * 64)
    monkeypatch.setattr(campus, "campus_roster", lambda *args: (key,))
    monkeypatch.setattr(campus, "_retained", lambda *args: ({key: selected}, {}))
    monkeypatch.setattr(store, "read_record", lambda *args: {"artifact": {"size_bytes": 1}})
    intake = tmp_path / "intake.json"
    result = campus.acquire_region(store, "01", pin, limit=0, intake=intake)
    again = campus.acquire_region(store, "01", pin, limit=0, intake=intake)
    assert result == again and result["requested"] == 0 and result["admitted"] == 1
    selection = json.loads(intake.read_text())
    assert selection["region"] == "01" and selection["roster_receipt"] == pin
    assert selection["reports"][0]["identifier"] == key
    monkeypatch.setattr("teadata.tprs_statewide_intake.campus_roster", lambda *args: (key,))
    assert validate_campus_selection(store, selection) == pin
    selection["reports"][0]["identifier"] = "001901002"
    with pytest.raises(ContractError, match="differs from its exact ordered campus roster"):
        validate_campus_selection(store, selection)
    with campus._broker_lock(store):
        with pytest.raises(ContractError, match="Cannot lock"):
            campus.acquire_region(store, "01", pin, limit=0, intake=intake)


def test_malformed_retained_report_does_not_block_other_campuses(tmp_path, monkeypatch):
    store = ProvenanceStore(tmp_path)
    bad, other = "001901001", "001901002"
    receipt = "a" * 64
    artifact = {"sha256": "b" * 64}
    retained = {"kind": "acquisition", "method": "download", "downloaded_at": "2026-11-06T00:00:00Z",
                "requested_url": campus.BROKER + "?" + urlencode(campus._controls(bad)), "artifact": artifact}
    monkeypatch.setattr(campus, "campus_roster", lambda *args: (bad, other))
    monkeypatch.setattr(store, "read_record", lambda *args: retained)

    def parse(*args, identifier, **kwargs):
        if identifier == bad:
            raise ContractError("unexpected publisher page")

    def acquire(*args, **kwargs):
        assert dict(kwargs["fields"])["id"] == other
        return SimpleNamespace(record_id="c" * 64)

    monkeypatch.setattr(campus, "read_student_report", parse)
    monkeypatch.setattr(campus, "acquire_form", acquire)
    record_path = store.root / "records" / (receipt + ".json")
    record_path.parent.mkdir()
    record_path.touch()
    result = campus.acquire_region(store, "01", "d" * 64)
    assert result["requested"] == result["admitted"] == 1
    assert result["holds"] == {bad: f"{receipt}: unexpected publisher page"}


@pytest.mark.parametrize("transport", [True, False])
def test_fresh_failure_returns_structured_stop(tmp_path, monkeypatch, transport):
    store = ProvenanceStore(tmp_path)
    key = "001901001"
    receipt = "a" * 64
    monkeypatch.setattr(campus, "campus_roster", lambda *args: (key,))
    monkeypatch.setattr(store, "read_record", lambda *args: {"artifact": {"sha256": "b" * 64}})

    def acquire(*args, **kwargs):
        if transport:
            raise campus.AcquisitionError("incomplete body")
        return SimpleNamespace(record_id=receipt)

    def parse(*args, **kwargs):
        raise ContractError("unexpected publisher page")

    monkeypatch.setattr(campus, "acquire_form", acquire)
    monkeypatch.setattr(campus, "read_student_report", parse)
    result = campus.acquire_region(store, "01", "c" * 64)
    assert result["requested"] == 1 and result["admitted"] == 0
    assert result["stop"] == ("AcquisitionError" if transport else "report_hold")
    assert result["holds"] == ({} if transport else {key: f"{receipt}: unexpected publisher page"})
