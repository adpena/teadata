import json
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.parse import urlencode

import pytest

from teadata.manifest import ContractError
from teadata.provenance import ProvenanceStore
from teadata.scripts import acquire_tprs_campuses as campus
from teadata.scripts import verify_tprs_campus_coverage as coverage
from teadata.tprs_statewide_intake import validate_campus_selection


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
                  ("--intake", str(tmp_path / "intake.json")), ("--partition", "0")):
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
