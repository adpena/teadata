from datetime import datetime, timezone
from email.message import Message
import json
from threading import Event
import time
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.parse import urlencode

import pytest

from teadata.acquisition import IncompleteTransfer, TransportInterruption
from teadata.manifest import ContractError
from teadata.provenance import ProvenanceStore
from teadata.scripts import acquire_tprs_campuses as campus
from teadata.scripts import verify_tprs_campus_coverage as coverage
from teadata.tprs_statewide_intake import validate_campus_selection


def _mock_request(monkeypatch, request):
    def acquire(client, *args, **kwargs):
        client.before_open()
        return request(client, *args, **kwargs)

    monkeypatch.setattr(campus, "acquire_form", acquire)


def test_coverage_union_requires_all_ordinals_and_checks_overlap_before_leaf_reads(tmp_path, monkeypatch):
    store = ProvenanceStore(tmp_path)
    ids = {f"001901{number:03d}" for number in range(1, 514)}
    roster_pins = {"01": "f" * 64}
    verified = []
    monkeypatch.setattr(coverage, "statewide_rosters", lambda *_: {"01": ids})
    monkeypatch.setattr(coverage, "DataRelease", lambda path, **_: SimpleNamespace(
        path=path, verify=lambda: verified.append(path),
    ))
    def selection(release, _, rosters):
        ordinal = int(release.path.name == "second")
        return release.path.name, {"01": set(coverage.campus_partition_ids(rosters["01"], ordinal))}, ordinal

    monkeypatch.setattr(coverage, "_selection", selection)
    first = (tmp_path / "first", "a" * 64)
    second = (tmp_path / "second", "b" * 64)
    result = coverage.verify_campus_coverage(store, roster_pins, [second, first])
    assert result["complete"] and result["verified"] == 513
    assert result["regions"]["01"]["missing_ordinals"] == []
    assert verified == [second[0], first[0]]
    verified.clear()
    overlap = (tmp_path / "overlap", "c" * 64)
    with pytest.raises(ContractError, match="overlap in region 01"):
        coverage.verify_campus_coverage(store, roster_pins, [first, overlap])
    with pytest.raises(ContractError, match="requires an exact manifest SHA-256"):
        coverage.verify_campus_coverage(store, roster_pins, [(first[0], None)])
    assert verified == []


@pytest.mark.parametrize(("header", "wait_source", "expected"), [
    ("900", "publisher", 1900),
    ("15", "publisher", 1015),
    ("0", "publisher", 1008),
    ("Thu Jan  1 00:17:10 1970", "publisher", 1030),
    ("Thursday, 01-Jan-70 00:17:10 GMT", "publisher", 1030),
    ("Thu, 01 Jan 1970 00:00:00 GMT", "publisher", 1008),
    ("Thu, 01 Jan 1970 00:17:10 +0000", "invalid_publisher_header", 1040),
    (None, "local_fallback", 1040),
])
def test_rate_limit_persists_across_region_and_process_boundaries(tmp_path, monkeypatch, header, wait_source, expected):
    store = ProvenanceStore(tmp_path)
    monkeypatch.setattr(campus, "campus_roster", lambda *args: ("001901001",))
    monkeypatch.setattr(campus, "time", type("Clock", (), {"time": staticmethod(lambda: 1000.0)}))
    calls = []

    def rate_limit(*args, **kwargs):
        pace = json.loads((tmp_path / "tprs-campus-broker-pace.json").read_bytes())
        assert pace == {"not_before": 1008}
        calls.append(1)
        raise HTTPError(campus.BROKER, 429, "Too Many Requests", {} if header is None else {"Retry-After": header}, None)

    _mock_request(monkeypatch, rate_limit)
    first = campus.acquire_region(store, "01", "a" * 64, fallback_wait=40)
    later = campus.acquire_region(store, "02", "b" * 64)
    expected_status = "invalid_retry_after" if wait_source == "invalid_publisher_header" else "HTTP 429"
    assert first["stop"] == expected_status and first["requested"] == 1
    assert first["attempts"]["001901001"]["backpressure"] == {
        "source": wait_source, "retry_after": header if wait_source == "publisher" else None,
        **({"rejected_headers": [header]} if wait_source == "invalid_publisher_header" else {}),
    }
    assert json.loads((tmp_path / "tprs-campus-broker-pace.json").read_bytes()) == {"not_before": expected}
    assert later["stop"] == "cooldown" and later["requested"] == 0 and calls == [1]


def test_duplicate_retry_after_uses_local_fallback(monkeypatch):
    monkeypatch.setattr(campus.time, "time", lambda: 1000.0)
    headers = Message()
    headers["Retry-After"] = "30"
    headers["Retry-After"] = "900"
    error = HTTPError(campus.BROKER, 429, "Too Many Requests", headers, None)
    decision = campus._retry_policy(error, 1008, 40)
    assert decision.status == "invalid_retry_after"
    assert decision.deadline == 1040
    assert decision.wait_source == "invalid_publisher_header"
    assert decision.rejected_headers == ("30", "900")


def test_obsolete_http_date_uses_full_fifty_year_cutoff():
    now = datetime(2026, 11, 6, tzinfo=timezone.utc).timestamp()
    assert campus._cooldown("Monday, 06-Dec-76 08:49:37 GMT", now, minimum_wait=0) == now
    now = datetime(2050, 1, 1, tzinfo=timezone.utc).timestamp()
    assert campus._cooldown("Tuesday, 29-Feb-00 00:00:00 GMT", now, minimum_wait=0) == now


@pytest.mark.parametrize("failure_phase", ["admission", "response"])
def test_pace_write_failure_stops_transport_and_preserves_deadline(tmp_path, monkeypatch, failure_phase):
    store = ProvenanceStore(tmp_path)
    monkeypatch.setattr(campus, "campus_roster", lambda *args: ("001901001", "001901002"))
    monkeypatch.setattr(campus.time, "time", lambda: 1000.0)
    original = campus._save
    writes = []
    calls = []

    def fail_once(path, payload, **kwargs):
        writes.append(payload["not_before"])
        if len(writes) == (1 if failure_phase == "admission" else 2):
            raise OSError("transient disk error")
        return original(path, payload, **kwargs)

    def throttle(client, *args, **kwargs):
        calls.append(1)
        raise HTTPError(campus.BROKER, 429, "Limited", {"Retry-After": "900"}, None)

    monkeypatch.setattr(campus, "_save", fail_once)
    _mock_request(monkeypatch, throttle)
    with pytest.raises(ContractError, match="Could not persist"):
        campus.acquire_region(store, "01", "a" * 64, jobs=2)
    assert calls == ([] if failure_phase == "admission" else [1])
    assert writes == ([1300] if failure_phase == "admission" else [1300, 1900, 1900])
    if calls:
        assert json.loads((tmp_path / "tprs-campus-broker-pace.json").read_bytes()) == {"not_before": 1900}


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
    _mock_request(monkeypatch, acquire)
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

    _mock_request(monkeypatch, acquire)
    monkeypatch.setattr(campus, "read_student_report", parse)
    result = campus.acquire_region(store, "01", "c" * 64)
    assert result["requested"] == 1 and result["admitted"] == 0
    assert result["stop"] == ("AcquisitionError" if transport else "report_hold")
    assert result["holds"] == ({} if transport else {key: f"{receipt}: unexpected publisher page"})


@pytest.mark.parametrize(("limit", "succeeds", "status"), [
    (3, True, "admitted"),
    (3, False, "retry_exhausted"),
    (2, False, "retry_deferred"),
])
def test_transient_attempts_use_one_pace_and_request_budget(tmp_path, monkeypatch, limit, succeeds, status):
    store = ProvenanceStore(tmp_path)
    key, next_key = "001901001", "001901002"
    monkeypatch.setattr(campus, "campus_roster", lambda *args: (key, next_key))
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(campus, "time", SimpleNamespace(
        time=lambda: clock.now, sleep=lambda seconds: setattr(clock, "now", clock.now + seconds),
    ))
    monkeypatch.setattr(store, "read_record", lambda *args: {"artifact": {"sha256": "b" * 64}})
    monkeypatch.setattr(campus, "read_student_report", lambda *a, **k: None)
    calls = []

    def acquire(*args, **kwargs):
        assert dict(kwargs["fields"])["id"] == key
        pace = json.loads((tmp_path / "tprs-campus-broker-pace.json").read_bytes())
        assert pace["not_before"] == clock.now + 8
        calls.append(clock.now)
        if succeeds and len(calls) == 3:
            return SimpleNamespace(record_id="a" * 64)
        if len(calls) == 1:
            raise TransportInterruption("network timeout")
        raise IncompleteTransfer("short response")

    _mock_request(monkeypatch, acquire)
    result = campus.acquire_region(store, "01", "c" * 64, limit=limit)
    assert calls == [1000 + 8 * index for index in range(limit)]
    assert result["requested"] == limit and result["admitted"] == int(succeeds)
    assert result["attempts"][key]["attempts"] == limit
    assert result["attempts"][key]["status"] == status
    assert result["attempts"][key]["failures"][0] == "TransportInterruption"
    assert result["stop"] == (None if succeeds else status)


@pytest.mark.parametrize("corrupt_receipt", [False, True])
def test_two_in_flight_drain_throttles_and_keep_the_later_deadline(tmp_path, monkeypatch, corrupt_receipt):
    store = ProvenanceStore(tmp_path)
    first, second, third = ("001901001", "001901002", "001901003")
    monkeypatch.setattr(campus, "campus_roster", lambda *args: (first, second, third))
    monkeypatch.setattr(campus, "MIN_INTERVAL", 0.01)
    second_started = Event()
    first_handled = Event()
    original_policy = campus._retry_policy
    calls = []
    longer_deadline = []

    def acquire(client, *args, **kwargs):
        identifier = dict(kwargs["fields"])["id"]
        if identifier == first:
            time.sleep(0.04)  # Setup takes longer than the launch interval.
        client.before_open()
        calls.append((identifier, time.monotonic()))
        if identifier == first:
            assert second_started.wait(3)
            if corrupt_receipt:
                return SimpleNamespace(record_id="a" * 64)
            raise HTTPError(campus.BROKER, 429, "Limited", {"Retry-After": "invalid"}, None)
        assert identifier == second
        second_started.set()
        assert first_handled.wait(3)
        longer_deadline.append(time.time() + 900)
        raise HTTPError(campus.BROKER, 503, "Limited", {"Retry-After": "900"}, None)

    def apply_policy(*args):
        result = original_policy(*args)
        if result.status == "invalid_retry_after":
            first_handled.set()
        return result

    def read_record(_):
        first_handled.set()
        raise ValueError("invalid source receipt")

    monkeypatch.setattr(campus, "acquire_form", acquire)
    monkeypatch.setattr(campus, "_retry_policy", apply_policy)
    monkeypatch.setattr(store, "read_record", read_record)
    result = campus.acquire_region(store, "01", "c" * 64, jobs=2, limit=3, interval=0.01)
    assert [identifier for identifier, _ in calls] == [first, second]
    assert calls[1][1] - calls[0][1] >= 0.009
    assert result["requested"] == 2 and result["stop"] == ("report_hold" if corrupt_receipt else "invalid_retry_after")
    assert result["attempts"][second]["status"] == "HTTP 503"
    pace = json.loads((tmp_path / "tprs-campus-broker-pace.json").read_bytes())
    assert pace["not_before"] >= longer_deadline[0]
    later = campus.acquire_region(store, "01", "c" * 64, jobs=2)
    assert later["stop"] == "cooldown" and later["requested"] == 0


def test_short_retry_after_preserves_parallel_reservation_until_drain(tmp_path, monkeypatch):
    store = ProvenanceStore(tmp_path)
    first, second = "001901001", "001901002"
    monkeypatch.setattr(campus, "campus_roster", lambda *args: (first, second))
    monkeypatch.setattr(campus, "MIN_INTERVAL", 0.01)
    monkeypatch.setattr(campus, "PARALLEL_DRAIN_RESERVATION", 60)
    second_started, deadline_computed, deadline_persisted = Event(), Event(), Event()
    original_policy, original_persist = campus._retry_policy, campus._persist_pace

    def apply_policy(*args):
        decision = original_policy(*args)
        deadline_computed.set()
        return decision

    def persist(pace, deadline):
        original_persist(pace, deadline)
        if deadline_computed.is_set():
            deadline_persisted.set()

    def acquire(client, *args, **kwargs):
        identifier = dict(kwargs["fields"])["id"]
        client.before_open()
        if identifier == first:
            assert second_started.wait(3)
            raise HTTPError(campus.BROKER, 429, "Limited", {"Retry-After": "1"}, None)
        second_started.set()
        assert deadline_persisted.wait(3)
        pace = json.loads((tmp_path / "tprs-campus-broker-pace.json").read_bytes())
        assert pace["not_before"] > time.time() + 50
        return SimpleNamespace(record_id="c" * 64)

    monkeypatch.setattr(campus, "_retry_policy", apply_policy)
    monkeypatch.setattr(campus, "_persist_pace", persist)
    monkeypatch.setattr(campus, "acquire_form", acquire)
    monkeypatch.setattr(store, "read_record", lambda _: {"artifact": {"sha256": "d" * 64}})
    monkeypatch.setattr(campus, "read_student_report", lambda *args, **kwargs: None)
    result = campus.acquire_region(store, "01", "a" * 64, jobs=2, limit=2, interval=0.01)
    assert result["stop"] == "HTTP 429" and result["admitted"] == 1
    pace = json.loads((tmp_path / "tprs-campus-broker-pace.json").read_bytes())
    assert pace["not_before"] < time.time() + 5


@pytest.mark.parametrize("fail_first_response_write", [False, True])
def test_interrupt_drains_started_request_and_persists_its_throttle(tmp_path, monkeypatch, fail_first_response_write):
    store = ProvenanceStore(tmp_path)
    first, second = "001901001", "001901002"
    monkeypatch.setattr(campus, "campus_roster", lambda *args: (first, second))
    interrupt = Event()
    calls = []
    deadline = []

    def acquire(*args, **kwargs):
        calls.append(dict(kwargs["fields"])["id"])
        assert interrupt.wait(3)
        deadline.append(time.time() + 900)
        raise HTTPError(campus.BROKER, 429, "Limited", {"Retry-After": "900"}, None)

    def wait_for_result(*args, **kwargs):
        interrupt.set()
        raise KeyboardInterrupt

    save = campus._save
    response_writes = []

    def save_deadline(path, payload, **kwargs):
        if deadline:
            response_writes.append(payload["not_before"])
            if fail_first_response_write and len(response_writes) == 1:
                raise OSError("transient response write failure")
        return save(path, payload, **kwargs)

    _mock_request(monkeypatch, acquire)
    monkeypatch.setattr(campus, "_save", save_deadline)
    monkeypatch.setattr(campus, "wait", wait_for_result)
    with pytest.raises(KeyboardInterrupt):
        campus.acquire_region(store, "01", "c" * 64, jobs=2, limit=2)
    assert calls == [first]
    pace = json.loads((tmp_path / "tprs-campus-broker-pace.json").read_bytes())
    assert pace["not_before"] >= deadline[0]
    assert len(response_writes) == (2 if fail_first_response_write else 1)


def test_inventory_uses_one_retained_scan_and_exact_rosters(tmp_path, monkeypatch):
    store = ProvenanceStore(tmp_path)
    first, second = "001901001", "002901001"
    receipts = {"01": "a" * 64, "02": "b" * 64}
    preferred = {first: "c" * 64}
    selection = campus.ReportSelection("campus", first, preferred[first], "d" * 64)
    monkeypatch.setattr(campus, "REGIONS", tuple(receipts))
    rosters = {"01": (first,), "02": (second,)}
    monkeypatch.setattr(campus, "campus_roster", lambda store, region, receipt: rosters[region])
    scans = []

    def retained(actual_store, identifiers, choices):
        scans.append((actual_store, identifiers, choices))
        return {first: selection}, {second: "publisher body needs review"}

    monkeypatch.setattr(campus, "_retained", retained)
    result = campus.retained_inventory(store, receipts, preferred=preferred)
    assert scans == [(store, {first, second}, preferred)]
    assert result["eligible"] == 2 and result["valid_reports"] == 1
    assert result["remaining"] == [second]
    assert result["reports"][first]["receipt"] == preferred[first]
    assert result["retained_holds"] == {second: "publisher body needs review"}
    assert result["regions"]["01"] == {"roster": 1, "admitted": 1, "roster_receipt": receipts["01"]}
    assert result["regions"]["02"]["admitted"] == 0
    with campus._broker_lock(store), pytest.raises(ContractError, match="Cannot lock"):
        campus.retained_inventory(store, receipts)


def test_inventory_rejects_incomplete_overlapping_and_unrelated_preferences(tmp_path, monkeypatch):
    store = ProvenanceStore(tmp_path)
    receipts = {"01": "a" * 64, "02": "b" * 64}
    monkeypatch.setattr(campus, "REGIONS", tuple(receipts))
    with pytest.raises(ContractError, match="each TPRS region"):
        campus.retained_inventory(store, {"01": receipts["01"]})
    monkeypatch.setattr(campus, "campus_roster", lambda *args: ("001901001",))
    with pytest.raises(ContractError, match="multiple TPRS regions"):
        campus.retained_inventory(store, receipts)
    monkeypatch.setattr(campus, "campus_roster", lambda store, region, receipt: (region + "1901001",))
    with pytest.raises(ContractError, match="outside the retained statewide roster"):
        campus.retained_inventory(store, receipts, preferred={"003901001": "c" * 64})


def test_inventory_rejects_a_processing_receipt_as_roster(tmp_path, monkeypatch):
    store = ProvenanceStore(tmp_path)
    monkeypatch.setattr(campus, "REGIONS", ("01",))
    monkeypatch.setattr(store, "read_record", lambda receipt: {"kind": "processing", "outputs": {}})
    with pytest.raises(ContractError, match="Campus roster receipt must identify an acquisition"):
        campus.retained_inventory(store, {"01": "a" * 64})


def test_inventory_cli_is_offline_and_rejects_regional_controls(tmp_path, monkeypatch, capsys):
    (tmp_path / "records").mkdir()
    (tmp_path / "objects").mkdir()
    path = tmp_path / "rosters.json"
    receipts = {"01": "a" * 64}
    path.write_text(json.dumps({"edition": "2026", "regions": {"01": {"receipt": receipts["01"], "campuses": 2}}}))
    monkeypatch.setattr(campus, "REGIONS", tuple(receipts))
    monkeypatch.setattr(campus, "campus_roster", lambda *args: ("001901001",))
    monkeypatch.setattr(campus, "AcquisitionClient", lambda *args, **kwargs: pytest.fail("offline inventory made a client"))
    cli = ["tprs-campuses", "--store", str(tmp_path), "--inventory", str(path)]
    monkeypatch.setattr("sys.argv", cli)
    campus.main()
    result = json.loads(capsys.readouterr().out)
    assert result["eligible"] == 1 and result["remaining"] == ["001901001"]
    assert not (tmp_path / "tprs-campus-broker-pace.json").exists()
    for flags in (("--limit", "25"), ("--interval", "8"), ("--roster-receipt", receipts["01"]),
                  ("--intake", str(tmp_path / "intake.json")), ("--partition", "0"), ("--jobs", "2")):
        monkeypatch.setattr("sys.argv", [*cli, *flags])
        with pytest.raises(SystemExit, match="1"):
            campus.main()
        assert "does not take regional acquisition options" in capsys.readouterr().err
    path.write_text('{"edition":"2026","regions":{"01":{},"01":{}}}')
    with pytest.raises(ContractError, match="unique keys"):
        campus._inventory_receipts(path)


def test_regional_cli_preserves_defaults_and_explicit_offline_limit(tmp_path, monkeypatch, capsys):
    (tmp_path / "records").mkdir()
    (tmp_path / "objects").mkdir()
    observed = []

    def acquire(store, region, receipt, **options):
        observed.append((region, receipt, options["limit"], options["interval"]))
        return {"region": region}

    monkeypatch.setattr(campus, "acquire_region", acquire)
    cli = ["tprs-campuses", "--store", str(tmp_path), "--region", "01", "--roster-receipt", "a" * 64]
    monkeypatch.setattr("sys.argv", cli)
    campus.main()
    monkeypatch.setattr("sys.argv", [*cli, "--limit", "0"])
    campus.main()
    assert observed == [("01", "a" * 64, 25, campus.MIN_INTERVAL), ("01", "a" * 64, 0, campus.MIN_INTERVAL)]
    assert [json.loads(line) for line in capsys.readouterr().out.splitlines()] == [{"region": "01"}] * 2
