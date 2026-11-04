"""Fail-closed tests for staffing and Indicator 11 candidate qualification."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest

from teadata import native_staff_spp as gate
from teadata import release, verification
from teadata.dataset_metadata import ArtifactRef, EditionKey
from teadata.manifest import ContractError
from teadata.peims_staff import STAFF
from teadata.provenance import ProvenanceStore
from teadata.release import DataRelease
from teadata.spp_leaves import DATASET as SPP


def test_known_manifest_registry_binds_disjoint_complete_witness_inventories():
    staff = gate.reviewed_staff_spp("8cd071e5cfaac027a43895aa2335206d87eb5bf19a9001b2cc42797c5b0c9325")
    spp = gate.reviewed_staff_spp("c7a02280b6228040552cb895da02a6fb9f9773887fc96e7121a257dc40adf95b")
    assert staff is not None and spp is not None
    assert staff[:2] == (STAFF, 3) and spp[:2] == (SPP, 4)
    assert gate.reviewed_staff_spp("d2f1f6c94eef232c197a464a847a21a1ed1b309cef3dbe63bf64bc1eddfa75aa") == spp
    assert gate.reviewed_staff_spp("0" * 64) is None
    assert gate.reviewed_staff_spp("465e130bcb4bc48a523e4901103d34a41ead89123141a2ca7bb0c6ef3171b4a0") is None
    assert len(staff[2]) == 8 and len(spp[2]) == 2
    witnesses = [w for plan in (staff, spp) for reports in plan[2].values() for w in reports]
    assert len(witnesses) == len({w["receipt"] for w in witnesses}) == 15
    assert [(w["entity"], w["publisher_rate_precision"]) for w in witnesses if "publisher_rate_precision" in w] == [
        ("227901", {"workbook": 1, "html": 2})]


def test_spp_report_precision_does_not_create_an_implicit_tolerance():
    assert gate._compare_spp_display("100.0", "100.0%", None) == "exact"
    assert gate._compare_spp_display("NR", "NR", None) == "exact"
    with pytest.raises(ContractError, match="reviewed workbook display"):
        gate._compare_spp_display("99.6", "99.62%", None)
    policy = {"workbook": 1, "html": 2}
    assert gate._compare_spp_display("99.6", "99.62%", policy) == "consistent_reviewed_display_precision"
    with pytest.raises(ContractError, match="reviewed workbook display"):
        gate._compare_spp_display("99.6", "99.65%", policy)
    with pytest.raises(ContractError, match="publisher rate token"):
        gate._compare_spp_display("0.0", "NR", None)


@pytest.mark.parametrize("damage", ["receipt", "controls", "raw-is-witness", "checksum"])
def test_report_open_requires_exact_source_receipt(tmp_path, damage):
    year = 2025
    key = EditionKey(SPP, "2026", "0" * 64, "district", "district", "grain")
    ref = cast(ArtifactRef, SimpleNamespace(edition=key, contract={"inputs": ["witness"] if damage == "raw-is-witness" else []}))
    path = tmp_path / "report.html"
    path.write_bytes(b"a")
    record = {"kind": "acquisition", "method": "download", "section": "reference",
              "dataset": "reference" if damage == "receipt" else "report",
              "requested_year": year, "resolved_year": year, "requested_url": gate._BROKER, "resolved_url": gate._BROKER,
              "source_metadata": {"source_url": "selector", "version": "2026", "period": {
                  "kind": "school_year", "start_year": 2024, "end_year": 2025}},
              "parameters": {"form_url": gate._BROKER, "http_method": "POST", "http_status": 200,
                             "form_controls": [] if damage == "controls" else [["fmt", "html"]]},
              "response_headers": {"content-type": "text/html"},
              "artifact": {"size_bytes": 1, "sha256": "witness"}}
    store = Mock(spec=ProvenanceStore)
    store.read_record.return_value = record
    store.verify_artifact.return_value = path
    with pytest.raises(ContractError, match="bytes differ" if damage == "checksum" else "receipt differs"):
        gate._read_report(store, ref, "receipt", dataset="report", url=gate._BROKER, selector="selector",
                          method="POST", year=year, controls=[("fmt", "html")])
    assert store.verify_artifact.called == (damage == "checksum")


@pytest.mark.parametrize("damage", [None, "scope", "extra-leaf", "parity", "no-reports", "witness"])
def test_reviewed_candidate_qualifies_only_after_raw_and_independent_checks(monkeypatch, damage):
    key = EditionKey(STAFF, "2026", "0" * 64, "state", "state", "grain")
    ref = SimpleNamespace(edition=key, identity="leaf")
    reports = [] if damage == "no-reports" else [{"entity": "state", "receipt": "source"}]
    plan = (SPP if damage == "scope" else STAFF, 3, {key.identity: reports})
    artifacts = {"leaf": {}} if damage != "extra-leaf" else {"leaf": {}, "other": {}}
    catalog = SimpleNamespace(to_dict=lambda: {"schema_version": 3, "editions": plan[2], "artifacts": artifacts},
                              artifact_ref=lambda _: ref)
    bundle = Mock(spec=DataRelease)
    bundle.catalog = catalog
    bundle.build_id = "build"
    bundle.manifest_sha256 = "manifest"
    parity = Mock(return_value={"status": "HOLD" if damage == "parity" else "PASS_RAW_PARITY"})
    witness = Mock(return_value={"values": 3})
    if damage == "witness":
        witness.side_effect = ContractError("Wrong report")
    monkeypatch.setattr(gate, "validate_source_parity", parity)
    monkeypatch.setattr(gate, "_staff_report", witness)
    if damage in {"scope", "extra-leaf", "no-reports", "witness"}:
        with pytest.raises(ContractError):
            gate.qualify_staff_spp(bundle, Mock(), Mock(), plan)
        if damage in {"scope", "extra-leaf"}:
            parity.assert_not_called()
        if damage != "witness":
            witness.assert_not_called()
    else:
        result = gate.qualify_staff_spp(bundle, Mock(), Mock(), plan)
        assert result["status"] == ("HOLD" if damage == "parity" else "PASS_REVIEWED_NATIVE")
        assert witness.called == (damage is None)


def test_canonical_gate_uses_bundled_raw_custody_and_explicit_witness_store(monkeypatch):
    manifest = Path("bundle/release.json")
    witness_store = Path("independent")
    selected = Mock(path=manifest, manifest_sha256="manifest")
    plan = (STAFF, 3, {"edition": []})
    qualifies = Mock(return_value={"status": "HOLD"})
    monkeypatch.setattr(release, "DataRelease", Mock(return_value=selected))
    monkeypatch.setattr(gate, "reviewed_staff_spp", Mock(return_value=plan))
    monkeypatch.setattr(gate, "qualify_staff_spp", qualifies)
    monkeypatch.setattr(verification, "_retained_store", lambda path: path)
    assert verification._validate_native_release(manifest, "trusted", witness_store)["status"] == "HOLD"
    qualifies.assert_called_once_with(selected, manifest.parent / "provenance", witness_store, plan)


def test_spp_report_uses_zero_padded_region_and_district_columns(monkeypatch):
    key = EditionKey(SPP, "2025", "0" * 64, "district", "district", "grain")
    ref = cast(ArtifactRef, SimpleNamespace(edition=key))
    witness = {"entity": "101912", "receipt": "receipt"}
    table = Mock()
    table.to_table.return_value.to_pylist.return_value = [{
        "district_name": "HOUSTON ISD", "esc_region": "04",
        "spp_raw_timely_initial_evaluation_reported_percent": "100.0",
        "spp_raw_timely_initial_evaluation_met_target": "NO",
    }]
    rows = [["2025 District Profile"], ["(School Year 2023-24)"], ["REGION 04"],
            ["HOUSTON ISD (101912)"], ["", "State", "LEA"], gate._SPP_HEADER,
            ["11: Child Find", "100.00%", "97.3%", "NO", "100.0%", "NO"]]

    def report():
        return "<table>" + "".join(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
        ) + "</table>"

    monkeypatch.setattr(gate, "_read_report", lambda *args, **kwargs: report())
    assert gate._spp_report(Mock(), ref, table, witness)["reported_target"] == "NO"
    rows[-1][-1] = "YES"
    with pytest.raises(ContractError, match="identity or target"):
        gate._spp_report(Mock(), ref, table, witness)
