"""Check retained budget proof boundaries with small synthetic witnesses."""

from collections import Counter
from hashlib import sha256
from pathlib import Path
from unittest.mock import Mock
from zipfile import ZipFile

import pytest

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
