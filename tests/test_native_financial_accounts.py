"""Check retained financial account proof boundaries with synthetic witnesses."""

from collections import Counter
from hashlib import sha256
from pathlib import Path
from unittest.mock import Mock
from zipfile import ZipFile

import pytest
import pyarrow as pa

from teadata import native_financial_accounts as gate
from teadata import release, verification
from teadata.dataset_metadata import ReferenceDocument
from teadata.manifest import ContractError
from teadata.provenance import ProvenanceStore


def test_budget_registry_selects_only_reviewed_budget_manifests():
    assert gate.is_reviewed_budget("bcff73e52199f4d8e527db2d66a89f8ee3a825d3734e7902e16ab4f7824e7f38")
    assert gate.is_reviewed_budget("4fbb1f4e22096df39fb377ad27dac0704b7f2397edd02e8d4b9df0af673af061")
    assert not gate.is_reviewed_budget("863522f79c4026d960465e56a4576b316fc1d99e63bdbc6c6be104e7186a4c94")
    assert not gate.is_reviewed_budget("0" * 64)


def test_budget_raw_replay_normalizes_record_endings_without_remapping_values(tmp_path):
    source = tmp_path / "budget.zip"
    with ZipFile(source, "w") as archive:
        archive.writestr("budget.csv", b"DISTRICT,BUDGAMT\r\n001902,-19\r\n001903,0\r\n")
    assert gate._source_hash(source, "budget.csv", ["DISTRICT", "BUDGAMT"]) == sha256(
        b"001902,-19\n001903,0\n").hexdigest()
    with pytest.raises(ContractError, match="publisher header differs"):
        gate._source_hash(source, "budget.csv", ["DISTRICT", "ACTAMT"])
    with pytest.raises(ContractError, match="archive member differs"):
        gate._source_hash(source, "actual.csv", ["DISTRICT", "BUDGAMT"])


@pytest.mark.parametrize("damage", [None, "amount", "missing-entity", "scope", "receipt"])
def test_budget_pwr_controls_are_exact_and_require_both_populations(tmp_path, monkeypatch, damage):
    from openpyxl import Workbook

    url = "https://tea.texas.gov/media/423716"
    path = tmp_path / "witness.xlsx"
    workbook = Workbook()
    assert workbook.active is not None
    workbook.remove(workbook.active)
    for group in ("GF", "AF"):
        sheet = workbook.create_sheet(f"25_26 {group} Bud")
        sheet.append(["REPORT", "DISTRICT NAME", "DISTRICT NUMBER", f"{group} Instruction (Function 11,95)"])
        sheet.append(["2026 Budget PWR", "EXAMPLE ISD", "001902", 1001 if damage == "amount" else 1000])
        if damage != "missing-entity" or group == "AF":
            sheet.append(["2026 Budget PWR", "EXAMPLE CHARTER", "002801", -50])
    workbook.save(path)
    digest = sha256(path.read_bytes()).hexdigest()
    witness = ReferenceDocument(reference_id="budget_pwr", role="source_notes", status="available",
                                title="Fixture PWR", url=url, reviewed_sha256=digest, provenance_ids=("1" * 64,))
    store = Mock(spec=ProvenanceStore)
    store.verify_artifact.return_value = path
    store.read_record.return_value = {
        "kind": "acquisition", "method": "download", "section": "reference",
        "dataset": "reference:peims_financial_budgets:budget_pwr",
        "requested_url": "https://example.invalid" if damage == "receipt" else url,
        "source_metadata": {"period": gate._PERIOD.to_dict()},
        "parameters": {"http_method": "GET", "http_status": 200},
        "artifact": {"sha256": digest, "size_bytes": path.stat().st_size},
    }
    scopes = {"001902": "district", "002801": "district" if damage == "scope" else "charter"}
    totals = Counter({(cdn, group): amount for cdn, amount in (("001902", 1000), ("002801", -50))
                      for group in ("GF", "AF")})
    monkeypatch.setattr(gate, "_COVERAGE", {(scope, group): 1 for scope in ("district", "charter") for group in ("GF", "AF")})
    if damage is None:
        assert gate._workbook_control(store, witness, scopes, totals)["compared_cells"] == 4
    else:
        with pytest.raises(ContractError):
            gate._workbook_control(store, witness, scopes, totals)


def test_canonical_budget_gate_requires_a_reviewed_manifest_and_separate_witness_store(monkeypatch):
    manifest = Path("budget/release.json")
    witness_store = Path("independent")
    selected = Mock(path=manifest, manifest_sha256="manifest")
    qualifies = Mock(return_value={"status": "PASS_REVIEWED_NATIVE"})
    monkeypatch.setattr(release, "DataRelease", Mock(return_value=selected))
    monkeypatch.setattr(gate, "is_reviewed_budget", lambda digest: digest == "manifest")
    monkeypatch.setattr(gate, "qualify_budget", qualifies)
    monkeypatch.setattr(verification, "_retained_store", lambda path: path)
    assert not gate.is_reviewed_budget("unreviewed")
    assert verification._validate_native_release(manifest, "trusted", witness_store)["status"] == "PASS_REVIEWED_NATIVE"
    qualifies.assert_called_once_with(selected, manifest.parent / "provenance", witness_store)


def test_actual_operating_control_keeps_signed_values_and_fund_scope():
    table = pa.table({
        "district_number": ["001902"] * 8,
        "fund_code": ["199", "199", "224", "699", "199", "199", "199", "199"],
        "object_code": ["6100", "6499", "6100", "6100", "6500", "6100", "6100", "6099"],
        "function_code": ["11", "11", "11", "11", "11", "71", "11", "11"],
        "program_intent_code": ["23", "33", "43", "23", "23", "23", "24", "23"],
        "actual_amount": [100, -20, 30, 40, 50, 60, 70, 80],
    })
    totals: Counter[tuple[str, str]] = Counter()
    gate._accumulate_actual_control(table, totals)
    assert totals == {("001902", "GF"): 80, ("001902", "AF"): 110, ("001902", "excluded_fund_699"): 40}


@pytest.mark.parametrize("damage", [None, "amount", "missing-raw", "extra-raw", "retyped", "duplicate", "period"])
def test_actual_pwr_compares_exact_operating_amounts_and_validates_full_roster(tmp_path, monkeypatch, damage):
    from openpyxl import Workbook

    url = "https://tea.texas.gov/example.xlsx"
    path = tmp_path / "witness.xlsx"
    workbook = Workbook()
    assert workbook.active is not None
    workbook.remove(workbook.active)
    for group in ("GF", "AF"):
        sheet = workbook.create_sheet(f"24_25 {group} Act")
        sheet.append(["SOURCE", "TAB NAME", "DISTRICT NUMBER", "DISTRICT NAME", "District Type",
                      f"{group} Students with Disabilities (PICs 23,33,43)"])
        edition = "2024 Actual PWR" if damage == "period" else "2025 Actual PWR"
        for cdn, population, amount in (
            ("001902", "INDEPENDENT", -20 if damage == "amount" else -10),
            ("001903", "COMMON" if damage == "retyped" and group == "AF" else "INDEPENDENT", 0),
            ("001904", "INDEPENDENT" if damage == "retyped" and group == "AF" else "COMMON", 0),
            ("001902" if damage == "duplicate" else "002801", "CHARTER", 999),
        ):
            sheet.append([edition, sheet.title, cdn, "Fixture", population, amount])
    workbook.save(path)
    store = Mock(spec=ProvenanceStore)
    store.verify_artifact.return_value = path
    store.read_record.return_value = {
        "kind": "acquisition", "method": "download", "section": "reference", "requested_url": url,
        "resolved_url": url, "source_metadata": {"source_url": url},
        "downloaded_at": "2026-11-04T19:34:19+00:00",
        "parameters": {"http_method": "GET", "http_status": 200},
        "response_headers": {"content-type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
        "artifact": {"sha256": sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size},
    }
    entities = {"001902", "001903", "001904"}
    if damage == "missing-raw":
        entities.remove("001903")
    if damage == "extra-raw":
        entities.add("002801")
    totals = Counter({("001902", "GF"): -10, ("001902", "AF"): -10, ("001902", "excluded_fund_699"): 21})
    monkeypatch.setattr(gate, "_ACTUAL_PWR", {2025: ("1" * 64, 2)})
    if damage is None:
        report = gate._actual_workbook_control(store, 2025, entities, totals)
        assert report["compared_cells"] == 4
        assert report["fund_groups"]["AF"]["common_unchecked"] == 1
        assert report["fund_groups"]["AF"]["charter_unchecked"] == 1
        assert report["independent_districts_with_nonzero_excluded_fund_699"] == 1
    else:
        with pytest.raises(ContractError):
            gate._actual_workbook_control(store, 2025, entities, totals)


def test_canonical_actual_pwr_dispatch_uses_explicit_separate_status(monkeypatch):
    import sys

    manifest = Path("actuals/release.json")
    witness_store = Path("independent")
    selected = Mock(path=manifest)
    qualifies = Mock(return_value={"status": gate.PASS_ACTUAL_PWR_CONTROLS})
    monkeypatch.setattr(release, "DataRelease", Mock(return_value=selected))
    monkeypatch.setattr(gate, "qualify_actual_pwr", qualifies)
    monkeypatch.setattr(verification, "_retained_store", lambda path: path)
    args = ["check", "--kind", "library", "--native-release", str(manifest), "--native-pin", "trusted",
            "--source-store", str(witness_store), "--actual-pwr-controls"]
    monkeypatch.setattr(sys, "argv", args)
    verification.main()
    qualifies.assert_called_once_with(selected, manifest.parent / "provenance", witness_store)

    qualifies.return_value = {"status": "PASS_REVIEWED_NATIVE"}
    with pytest.raises(SystemExit) as error:
        verification.main()
    assert error.value.code == 2

    monkeypatch.setattr(sys, "argv", ["check", "--kind", "library", "--actual-pwr-controls"])
    with pytest.raises(SystemExit) as error:
        verification.main()
    assert error.value.code == 2
