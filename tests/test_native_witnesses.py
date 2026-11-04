import csv
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest

from teadata import native_witnesses, release, source_parity, verification
from teadata.dataset_metadata import ArtifactRef, EditionKey
from teadata.manifest import ContractError
from teadata.provenance import ProvenanceStore


def test_independent_count_distinguishes_suppression_bound_and_zero():
    field = "count"
    zero = {"count": 0, "peims_status_count": "observed"}
    suppressed = {"count": None, "peims_status_count": "suppressed", "peims_raw_count": "<2790"}
    native_witnesses._compare_count(zero, field, "0")
    native_witnesses._compare_count(suppressed, field, "<2,790")
    with pytest.raises(ContractError, match="status differs"):
        native_witnesses._compare_count(zero, field, "N/A")
    with pytest.raises(ContractError, match="bound differs"):
        native_witnesses._compare_count(suppressed, field, "<20")


def test_witness_requires_its_actual_tea_request_url():
    broker = native_witnesses._BROKER
    record = {
        "kind": "acquisition", "method": "download", "section": "reference",
        "requested_year": 2026, "resolved_year": 2026,
        "requested_url": "https://example.invalid/report", "resolved_url": "https://example.invalid/report",
        "response_headers": {"content-type": "text/html"},
        "artifact": {"sha256": "independent", "size_bytes": 1},
        "parameters": {"form_url": broker, "http_method": "POST", "http_status": 200,
                       "form_controls": [["_program", "adhoc.std_driver1.sas"], ["RptClass", "StudPgm"],
                                         ["SchoolYr", "26"], ["report", "StateState"]]},
        "dataset": "reference:peims_individual_witness",
    }
    ref = SimpleNamespace(edition=SimpleNamespace(dataset=native_witnesses._PROGRAM, scope="state", edition="2026"),
                          contract={"inputs": ["original"]})
    store = SimpleNamespace(read_record=lambda _: record, verify_artifact=Mock())
    with pytest.raises(ContractError, match="Invalid independent publisher acquisition"):
        native_witnesses._open_witness(cast(ProvenanceStore, store), "receipt", cast(ArtifactRef, ref), "state")
    store.verify_artifact.assert_not_called()


def test_primary_disability_witness_keeps_omitted_category_and_suppression_distinct():
    key = EditionKey(native_witnesses.PRIMARY_DISABILITY, "2023", "0" * 64, "district", "district", "district")
    rows = [
        ["Students Receiving Special Education Services By Primary Disability PEIMS Data 2022-2023"],
        ["Districtwide for AUSTIN ISD (227901 )"],
        ["District Name (District Number)/Charter Status/Region", "Primary Disability"],
        ["LD", "DB"],
        ["AUSTIN ISD (227901 )", "TRADITIONAL ISD/CSD", "13", "4,129", "N/A"],
    ]
    native = [{"learning": 4129, "peims_status_learning": "observed", "deafblind": None,
               "peims_status_deafblind": "suppressed", "delay": 0, "peims_status_delay": "observed"}]
    columns = {"LD": "learning", "DB": "deafblind", "DD": "delay"}
    assert native_witnesses._primary_disability(rows, native, key, "227901", columns, ["DD"]) == (2, ["DD"])
    with pytest.raises(ContractError, match="columns differ"):
        native_witnesses._primary_disability(rows, native, key, "227901", columns, [])
    rows[-1][-1] = "0"
    with pytest.raises(ContractError, match="count or status differs"):
        native_witnesses._primary_disability(rows, native, key, "227901", columns, ["DD"])


def test_transfer_witness_rejects_a_truncated_data_row_as_a_footer(tmp_path):
    report = tmp_path / "district.csv"
    with report.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerows([["District Level Transfer Report"], native_witnesses._TRANSFER_HEADER,
                          ["2025-2026", "13", "REGION", "DISTRICT  -  227901", "", "100", "Transfers In From",
                           "100", "PARTNER", "10"], ["2025-2026"],
                          *[[line] for line in native_witnesses._FOOTNOTES]])
    ref = SimpleNamespace(edition=SimpleNamespace(edition="2026"))
    with pytest.raises(ContractError, match="row width differs"):
        native_witnesses._transfer(report, cast(ArtifactRef, ref), "227901 DISTRICT", Mock())


@pytest.mark.parametrize("damage", [None, "metadata", "extra-edition", "parity", "witness"])
def test_native_gate_requires_reviewed_identity_parity_and_independent_witnesses(monkeypatch, damage):
    case = {"artifact": "leaf", "sha256": "bytes", "processing_id": "receipt", "metadata_sha256": "metadata"}
    edition = {"metadata": {"sha256": "changed" if damage == "metadata" else "metadata"}}
    document = {"schema_version": 4, "editions": {"edition": edition}, "artifacts": {"leaf": "edition"}}
    if damage == "extra-edition":
        document["editions"]["extra"] = edition
    checked = Mock()
    artifact = SimpleNamespace(sha256="bytes", processing_id="receipt")
    catalog = SimpleNamespace(to_dict=lambda: document, artifact_ref=lambda _: artifact)
    selected = SimpleNamespace(catalog=catalog, verify=checked, build_id="build", manifest_sha256="manifest")
    load = Mock(return_value=selected)
    parity = Mock(return_value={"status": "HOLD" if damage == "parity" else "PASS_RAW_PARITY"})
    witnesses = Mock(return_value={"peims_values": 1})
    if damage == "witness":
        witnesses.side_effect = ContractError("Publisher witness differs")
    monkeypatch.setattr(release, "DataRelease", load)
    monkeypatch.setattr(native_witnesses, "reviewed_native_references", lambda: {"edition": case})
    monkeypatch.setattr(native_witnesses, "validate_native_witnesses", witnesses)
    monkeypatch.setattr(source_parity, "validate_source_parity", parity)
    monkeypatch.setattr(verification, "_retained_store", lambda _: "store")

    def qualify():
        return verification._validate_native_release(Path("release.json"), "trusted", Path("store"))

    if damage in {"metadata", "extra-edition", "witness"}:
        with pytest.raises(verification.ReceiptError):
            qualify()
        if damage != "witness":
            checked.assert_not_called()
            parity.assert_not_called()
        return
    result = qualify()
    load.assert_called_once_with(Path("release.json"), expected_sha256="trusted", offline=True)
    checked.assert_called_once()
    parity.assert_called_once_with(selected, "store", edition_ids=["edition"])
    if damage == "parity":
        assert result["status"] == "HOLD"
        assert result["publisher_witnesses"] is None
        witnesses.assert_not_called()
    else:
        assert result["status"] == "PASS_REVIEWED_NATIVE"
        assert result["publisher_witnesses"] == {"peims_values": 1}
        witnesses.assert_called_once()
