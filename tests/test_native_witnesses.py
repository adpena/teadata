import csv
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest

from teadata import native_witnesses, release, source_parity, verification
from teadata.transfer_sources import transfer_columns
from teadata.dataset_metadata import ArtifactRef, EditionKey
from teadata.manifest import ContractError, canonical_json
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
    native = [{"region_number": "13", "charter_status": "TRADITIONAL ISD/CSD",
               "learning": 4129, "peims_status_learning": "observed", "deafblind": None,
               "peims_status_deafblind": "suppressed", "delay": 0, "peims_status_delay": "observed"}]
    columns = {"LD": "learning", "DB": "deafblind", "DD": "delay"}
    assert native_witnesses._primary_disability(rows, native, key, "227901", columns, ["DD"]) == (2, ["DD"])
    with pytest.raises(ContractError, match="columns differ"):
        native_witnesses._primary_disability(rows, native, key, "227901", columns, [])
    rows[-1][-1] = "0"
    with pytest.raises(ContractError, match="count or status differs"):
        native_witnesses._primary_disability(rows, native, key, "227901", columns, ["DD"])
    rows[-3][0] = "District Name (District Number)/Region"
    rows[-1].pop(1)
    rows[-1][-1] = "N/A"
    assert native_witnesses._primary_disability(rows, native, key, "227901", columns, ["DD"]) == (2, ["DD"])


def test_historical_district_classification_keeps_leading_zeroes_and_unknowns():
    import pyarrow as pa

    from teadata.district_types import project_district_types

    keys = pa.chunked_array([["001902", "108801"], ["015847", "227901"]])
    codes, charter = project_district_types(keys, {"001902": "H", "108801": "I", "227901": "A"})
    assert codes.to_pylist() == ["H", "I", None, "A"]
    assert charter.to_pylist() == [False, True, None, False]


def test_transfer_witness_rejects_a_truncated_data_row_as_a_footer(tmp_path):
    report = tmp_path / "district.csv"
    with report.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerows([["District Level Transfer Report"], transfer_columns("district", year=2026),
                          ["2025-2026", "13", "REGION", "DISTRICT  -  227901", "", "100", "Transfers In From",
                           "100", "PARTNER", "10"], ["2025-2026"],
                          *[[line] for line in native_witnesses._FOOTNOTES]])
    ref = SimpleNamespace(edition=SimpleNamespace(edition="2026"))
    with pytest.raises(ContractError, match="row width differs"):
        native_witnesses._transfer(report, cast(ArtifactRef, ref), "227901 DISTRICT", Mock())


@pytest.mark.parametrize("label", ["District (001902)", "District (01902)", "District (001903)"])
@pytest.mark.parametrize("year", [2020, 2021])
def test_historical_transfer_witness_requires_the_exact_parenthesized_district(tmp_path, label, year):
    report = tmp_path / "district.csv"
    headings = transfer_columns("district", year=year)
    row = [f"{year-1}-{year}", "13", "REGION", label, "", "100", "Transfers In From",
           "100", "Partner (002902)", "-999"]
    with report.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows([["District Level Transfer Report"], headings,
                                     row, [], *[[line] for line in native_witnesses._FOOTNOTES]])
    table = Mock()
    columns = ["school_year", "REPORT_REGION", "reporting_region_name", headings[3],
               "reporting_charter_status", "report_number", "REPORT_TYPE", "line_group_number",
               headings[8], "TRANSFERS_IN_OUT"]
    table.to_table.return_value.to_pylist.return_value = [dict(zip(columns, row, strict=True))]
    ref = cast(ArtifactRef, SimpleNamespace(edition=SimpleNamespace(edition=str(year))))
    if label == "District (001902)":
        assert native_witnesses._transfer(report, ref, "001902 DISTRICT", table) == 1
    else:
        with pytest.raises(ContractError, match="row width differs"):
            native_witnesses._transfer(report, ref, "001902 DISTRICT", table)
        table.to_table.assert_not_called()


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


def test_primary_disability_review_has_exact_independent_witness_inventory():
    pin = "6f2cc32a7b9cc1b43d8b0ed84f2870d3dd75ddcfef58e9e8508e37f34c41cd37"
    editions = native_witnesses.reviewed_primary_disability_references(pin)
    assert editions is not None and len(editions) == 8
    assert native_witnesses.reviewed_primary_disability_references("0" * 64) is None
    witnesses = [witness for case in editions.values() for witness in case["reports"]]
    assert len(witnesses) == len({witness["receipt"] for witness in witnesses}) == 8
    for entity in ("state", "227901"):
        assert sum(witness["entity"] == entity for witness in witnesses) == 4
    assert [(witness["entity"], witness["unreported_codes"]) for witness in witnesses if "unreported_codes" in witness] == [
        ("227901", ["DD"]), ("227901", ["DD"])]


def test_composed_profile_reuses_the_exact_disability_review_inventory():
    inline_pin = "6f2cc32a7b9cc1b43d8b0ed84f2870d3dd75ddcfef58e9e8508e37f34c41cd37"
    composed_pin = "465e130bcb4bc48a523e4901103d34a41ead89123141a2ca7bb0c6ef3171b4a0"
    composed = native_witnesses.reviewed_composed_disability_references(composed_pin)
    assert composed == native_witnesses.reviewed_primary_disability_references(inline_pin)
    assert native_witnesses.reviewed_composed_disability_references(inline_pin) == {}
    assert native_witnesses.reviewed_primary_disability_references(composed_pin) is None
    legacy = native_witnesses.reviewed_native_references()
    assert legacy.keys().isdisjoint(composed) and len(legacy | composed) == 17


def test_historical_peims_review_pins_each_year_scope_and_family():
    import json
    from importlib.resources import files

    review = json.loads(files("teadata").joinpath("reference_cases/native_peims_historical.json").read_text())
    editions = native_witnesses.reviewed_historical_peims_references(review["manifest_sha256"])
    assert editions is not None and len(editions) == 18
    assert native_witnesses.reviewed_historical_peims_references("0" * 64) is None
    witnesses = [witness for case in editions.values() for witness in case["reports"]]
    assert len(witnesses) == len({witness["receipt"] for witness in witnesses}) == 18
    assert {witness["entity"] for witness in witnesses} == {"state", "227901", "057829001"}


def test_historical_campus_ethnicity_witness_does_not_infer_an_unpublished_total():
    rows = [["Campus", "Ethnicity", "Count"], ["057829001", "Asian", "<10"]]
    native = [{
        "ethnicity": "Asian",
        "enrollment_by_ethnicity_count": None,
        "peims_status_enrollment_by_ethnicity_count": "suppressed",
        "peims_raw_enrollment_by_ethnicity_count": "<10",
    }]
    key = EditionKey(native_witnesses._ETHNICITY, "2023", "1" * 64, "campus", "campus", "school_year")
    assert native_witnesses._ethnicity(rows, native, key, "057829001", 200) == 1
    current = EditionKey(native_witnesses._ETHNICITY, "2026", "1" * 64, "campus", "campus", "school_year")
    with pytest.raises(ContractError, match="publisher total is missing"):
        native_witnesses._ethnicity(rows, native, current, "057829001", 200)


@pytest.mark.parametrize("damage", [
    None, "pin", "missing-edition", "overlap", "legacy-metadata", "disability-metadata", "receipt",
    "missing-store", "parity",
])
def test_composed_gate_requires_both_reviewed_constituents(monkeypatch, damage):
    key = EditionKey(native_witnesses.PRIMARY_DISABILITY, "2026", "1" * 64, "district", "district", "district")
    metadata = {"title": "Reviewed primary disability"}
    primary = {"artifact": "disability-leaf", "sha256": "disability-bytes", "processing_id": "disability-receipt",
               "metadata_sha256": sha256(canonical_json(metadata).encode()).hexdigest()}
    legacy = {"artifact": "legacy-leaf", "sha256": "legacy-bytes", "processing_id": "legacy-receipt",
              "metadata_sha256": "legacy-metadata"}
    editions = {"legacy": {"metadata": {"sha256": "changed" if damage == "legacy-metadata" else "legacy-metadata"}},
                key.identity: {"metadata": {"sha256": "externalized-disability-child"}}}
    routes = {"legacy-leaf": "legacy", "disability-leaf": key.identity}
    if damage == "missing-edition":
        del editions["legacy"]
        del routes["legacy-leaf"]
    artifacts = {
        "legacy-leaf": SimpleNamespace(sha256="legacy-bytes", processing_id="legacy-receipt"),
        "disability-leaf": SimpleNamespace(
            edition=key, sha256="disability-bytes",
            processing_id="changed" if damage == "receipt" else "disability-receipt",
        ),
    }
    decode = Mock(return_value=SimpleNamespace(
        to_dict=lambda: {"title": "Changed"} if damage == "disability-metadata" else metadata))
    catalog = SimpleNamespace(to_dict=lambda: {"schema_version": 4, "editions": editions, "artifacts": routes},
                              artifact_ref=artifacts.__getitem__, metadata=decode)
    selected = SimpleNamespace(catalog=catalog, verify=Mock(), build_id="build",
                               manifest_sha256="unreviewed" if damage == "pin" else "composition")
    load = Mock(return_value=selected)
    parity = Mock(return_value={"status": "HOLD" if damage == "parity" else "PASS_RAW_PARITY"})
    witnesses = Mock(return_value={"peims_values": 13, "transfer_rows": 1})
    primary_id = "legacy" if damage == "overlap" else key.identity
    monkeypatch.setattr(release, "DataRelease", load)
    monkeypatch.setattr(native_witnesses, "reviewed_native_references", lambda: {"legacy": legacy})
    monkeypatch.setattr(native_witnesses, "reviewed_composed_disability_references",
                        lambda digest: {primary_id: primary} if digest == "composition" else {})
    monkeypatch.setattr(source_parity, "validate_source_parity", parity)
    monkeypatch.setattr(native_witnesses, "validate_native_witnesses", witnesses)
    store = Mock(side_effect=verification.ReceiptError("Missing original receipts")
                 if damage == "missing-store" else None, return_value="retained-union")
    monkeypatch.setattr(verification, "_retained_store", store)

    if damage not in {None, "parity"}:
        with pytest.raises(verification.ReceiptError):
            verification._validate_native_release(Path("release.json"), "trusted", Path("sources"))
        parity.assert_not_called()
        witnesses.assert_not_called()
        return
    result = verification._validate_native_release(Path("release.json"), "trusted", Path("sources"))
    load.assert_called_once_with(Path("release.json"), expected_sha256="trusted", offline=True)
    decode.assert_called_once_with(key.dataset, edition=key)
    selected.verify.assert_called_once()
    store.assert_called_once_with(Path("sources"))
    parity.assert_called_once_with(selected, "retained-union", edition_ids=sorted(editions))
    if damage == "parity":
        assert result["status"] == "HOLD" and result["publisher_witnesses"] is None
        witnesses.assert_not_called()
    else:
        assert result["status"] == "PASS_REVIEWED_NATIVE"
        witnesses.assert_called_once_with(selected, "retained-union", {"legacy": legacy, key.identity: primary})


@pytest.mark.parametrize("damage", [None, "unknown-pin", "wrong-schema", "route", "metadata", "leaf", "receipt", "missing-custody"])
def test_primary_disability_gate_requires_exact_inline_edition_and_bundled_parity(monkeypatch, damage):
    key = EditionKey(native_witnesses.PRIMARY_DISABILITY, "2026", "1" * 64, "state", "state", "school_year")
    other = EditionKey(native_witnesses.PRIMARY_DISABILITY, "2025", "2" * 64, "state", "state", "school_year")
    pin = "known" if damage != "unknown-pin" else "unknown"
    path = Path("frozen/release.json")
    witnesses_path = Path("retained-witnesses")
    metadata = {"title": "Reviewed primary-disability edition"}
    case = {"artifact": "leaf", "sha256": "bytes", "processing_id": "receipt",
            "metadata_sha256": sha256(canonical_json(metadata).encode()).hexdigest()}
    references = {key.identity: case}
    document = {"schema_version": 4 if damage == "wrong-schema" else 3,
                "editions": {key.identity: {"metadata": {"title": "tampered"} if damage == "metadata" else metadata}},
                "artifacts": {"leaf": {"edition": (other if damage == "route" else key).to_dict()}}}
    ref = SimpleNamespace(sha256="changed" if damage == "leaf" else "bytes",
                          processing_id="changed" if damage == "receipt" else "receipt")
    catalog = SimpleNamespace(to_dict=lambda: document, artifact_ref=lambda _: ref)
    selected = SimpleNamespace(catalog=catalog, verify=Mock(), build_id="build", manifest_sha256=pin, path=path)
    load = Mock(return_value=selected)
    parity = Mock(return_value={"status": "PASS_RAW_PARITY"})
    witnesses = Mock(return_value={"peims_values": 13})
    monkeypatch.setattr(release, "DataRelease", load)
    monkeypatch.setattr(native_witnesses, "reviewed_primary_disability_references",
                        lambda digest: references if digest == "known" else None)
    monkeypatch.setattr(native_witnesses, "reviewed_native_references", lambda: {"legacy": case})
    monkeypatch.setattr(native_witnesses, "validate_native_witnesses", witnesses)
    monkeypatch.setattr(source_parity, "validate_source_parity", parity)

    def store(location):
        if damage == "missing-custody" and location == path.parent / "provenance":
            raise verification.ReceiptError("Missing original receipts")
        return location

    monkeypatch.setattr(verification, "_retained_store", store)
    if damage is not None:
        with pytest.raises(verification.ReceiptError):
            verification._validate_native_release(path, pin, witnesses_path)
        parity.assert_not_called()
        witnesses.assert_not_called()
    else:
        assert verification._validate_native_release(path, pin, witnesses_path)["status"] == "PASS_REVIEWED_NATIVE"
        selected.verify.assert_called_once()
        parity.assert_called_once_with(selected, path.parent / "provenance", edition_ids=[key.identity])
        witnesses.assert_called_once_with(selected, witnesses_path, references)
    load.assert_called_once_with(path, expected_sha256=pin, offline=True)


def test_native_witness_replay_rejects_an_edition_with_no_reports():
    artifact = SimpleNamespace(edition=SimpleNamespace(dataset=native_witnesses.PRIMARY_DISABILITY))
    selected = SimpleNamespace(catalog=SimpleNamespace(artifact_ref=Mock(return_value=artifact)), analytical_dataset=Mock())
    with pytest.raises(ContractError, match="requires independent publisher witnesses"):
        native_witnesses.validate_native_witnesses(cast(release.DataRelease, selected), Mock(),
                                                   {"edition": {"artifact": "id", "reports": []}})
    selected.analytical_dataset.assert_not_called()
