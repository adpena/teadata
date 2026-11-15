"""Fail-closed retained educator hiring review, using small synthetic sources."""

from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import openpyxl
import pyarrow as pa
import pyarrow.dataset as ds
import pytest

from teadata import native_educator_hiring as gate, verification
from teadata.dataset_metadata import DataPeriod, EditionKey
from teadata.manifest import ContractError
from teadata.provenance import ProvenanceStore
from teadata.release import DataRelease


_PIN = "8989e9860aadd5d43bcb30bcc40aab2af59f230429638fb0723a1496ca821626"
_DISTRICT = "educator_district_new_hires"
_EPP = "educator_district_new_hires_by_epp"
_STATE = "educator_state_new_hires"


def plan():
    review = gate.reviewed_educator_hiring(_PIN)
    assert review is not None
    return review


def identity_release(review):
    catalog = {"schema_version": 4, "editions": {}, "artifacts": {}}
    refs, acquisitions, periods = {}, {}, {}
    for identifier, case in review["editions"].items():
        key = EditionKey.from_dict(case["key"])
        source = review["sources"][key.dataset]
        catalog["editions"][identifier] = deepcopy({"key": case["key"], "metadata": case["metadata"]})
        catalog["artifacts"][case["artifact_id"]] = identifier
        refs[case["artifact_id"]] = SimpleNamespace(
            edition=key, identity=case["artifact_id"], sha256=case["leaf_sha256"],
            size_bytes=case["leaf_size_bytes"], processing_id=case["processing_id"],
            contract={"rows": case["rows"]})
        acquisitions[key.identity] = [{"record_id": source["receipt"], "artifact": deepcopy(source["artifact"])}]
        periods[key.identity] = DataPeriod("school_year", case["school_year_end"] - 1, case["school_year_end"]).to_dict()
    selected = SimpleNamespace(
        manifest_sha256=_PIN, build_id=review["build_id"], path=Path("relocated/child/release.json"),
        verify=Mock(), analytical_dataset=Mock(side_effect=AssertionError("Observation I/O before identity validation")),
        catalog=SimpleNamespace(
            to_dict=lambda: catalog, artifact_ref=refs.__getitem__,
            acquisitions=lambda dataset, edition: acquisitions[edition.identity],
            field_metadata=lambda field, scope, edition: {"period": periods[edition.identity]}))
    return selected, catalog, refs, acquisitions, periods


def native_row(source, raw, ordinal=2):
    row = dict(zip(source["fields"], raw, strict=True))
    row["publisher_worksheet_row"] = ordinal
    for name, value in tuple(row.items()):
        if name.endswith(("_count", "_ratio")):
            if name.endswith("_ratio"):
                row[name] = float(value)
            row[f"raw_{name}"] = str(value)
            row[f"status_{name}"] = "observed"
    return row


def test_frozen_plan_covers_all_45_tables_and_nine_publisher_examples():
    review = plan()
    assert review["manifest_sha256"] == _PIN
    assert review["build_id"] == "a6bc2dd366d57884facf80c3d2ace9a96c194e53c665b14b9f8bcbee51c1c328"
    assert review["catalog_schema_version"] == 4
    assert len(review["editions"]) == 45 and len(review["publisher_roots"]) == 7
    assert {(case["key"]["dataset"], case["school_year_end"], case["key"]["table"])
            for case in review["editions"].values()} == {
        (dataset, year, ("state" if dataset == _STATE else "district") + (f"_{year}" if year < 2026 else ""))
        for dataset in (_DISTRICT, _EPP, _STATE)
        for year in range(2016 if dataset == _STATE else 2010, 2027)}
    assert len(review["goldens"]) == 9
    assert {(case["dataset"], case["worksheet_row"]) for case in review["goldens"]} == {
        (_DISTRICT, row) for row in (78756, 78757, 78758, 150, 14332)} | {
        (_EPP, row) for row in (39, 40, 43)} | {(_STATE, 12)}
    facts = [case["facts"] for case in review["goldens"]]
    assert any(row.get("district_number") == "014802" for row in facts)
    assert any(row.get("district_name") == "" for row in facts)
    assert any(row.get("reported_epp_assignment") == "Na-Uncertified" for row in facts)
    assert any(row.get("intern_certified_ratio") == 0.10025823404158928 for row in facts)


def test_unknown_manifest_does_not_open_packaged_resources(monkeypatch):
    resource = Mock(side_effect=AssertionError("Unrelated pin opened review resources"))
    monkeypatch.setattr(gate, "files", resource)
    assert gate.reviewed_educator_hiring("0" * 64) is None
    resource.assert_not_called()


@pytest.mark.parametrize("damage", [
    "manifest", "build", "schema", "edition", "routing", "metadata", "edition-key",
    "leaf", "size", "processing", "rows", "acquisition", "source-artifact", "field-period",
])
def test_changed_identity_is_rejected_before_observation_io(damage):
    review = plan()
    selected, catalog, refs, acquisitions, periods = identity_release(review)
    gate.validate_educator_hiring_identity(selected, review)
    identifier, case = next(iter(review["editions"].items()))
    ref = refs[case["artifact_id"]]
    if damage == "manifest":
        selected.manifest_sha256 = "0" * 64
    elif damage == "build":
        selected.build_id = "0" * 64
    elif damage == "schema":
        catalog["schema_version"] = 3
    elif damage == "edition":
        catalog["editions"].pop(identifier)
    elif damage == "routing":
        catalog["artifacts"][case["artifact_id"]] = "wrong-edition"
    elif damage == "metadata":
        catalog["editions"][identifier]["metadata"]["sha256"] = "0" * 64
    elif damage == "edition-key":
        ref.edition = EditionKey.from_dict({**case["key"], "table": "state_1900"})
        selected.catalog.acquisitions = lambda *a, **k: []
    elif damage == "leaf":
        ref.sha256 = "0" * 64
    elif damage == "size":
        ref.size_bytes += 1
    elif damage == "processing":
        ref.processing_id = "0" * 64
    elif damage == "rows":
        ref.contract["rows"] += 1
    elif damage == "acquisition":
        acquisitions[ref.edition.identity][0]["record_id"] = "0" * 64
    elif damage == "source-artifact":
        acquisitions[ref.edition.identity][0]["artifact"]["filename"] = "another.xlsx"
    else:
        periods[ref.edition.identity] = {"kind": "unknown"}
    with pytest.raises(ContractError, match="Educator hiring.*differs"):
        gate.qualify_educator_hiring(selected, Mock(), Mock(), review)
    selected.verify.assert_not_called()
    selected.analytical_dataset.assert_not_called()


@pytest.mark.parametrize("dataset,raw", [
    (_DISTRICT, ("014802", 2010, "01", "Intern Certified", 0, "", "Rural", "Region 1", "2009-10")),
    (_EPP, ("015950", 2013, "01", "", "Rural", "NA", 0, "Na-Uncertified", "Intern Certified", "Region 1", "2012-13", "NA")),
])
def test_row_preserves_zero_blank_name_historical_id_and_literal_placeholder(dataset, raw):
    source = plan()["sources"][dataset]
    row = native_row(source, raw)
    gate._check_row(row, raw, 2, source)
    assert row["district_number"] == raw[0]
    assert row["district_name"] == ""
    assert row["raw_new_hires_count"] == "0" and row["status_new_hires_count"] == "observed"


@pytest.mark.parametrize("ratio", [0.10025823404158928, -0.0])
def test_ratio_requires_full_binary64_precision_and_signed_zero(ratio):
    source = {"fields": ["intern_certified_ratio"]}
    raw = (ratio,)
    row = native_row(source, raw)
    gate._check_row(row, raw, 2, source)
    row["intern_certified_ratio"] = round(ratio, 8) if ratio else 0.0
    with pytest.raises(ContractError, match="value or type differs"):
        gate._check_row(row, raw, 2, source)


@pytest.mark.parametrize("damage", ["raw", "status", "coordinate", "coordinate-type", "extra", "missing", "width", "count-type", "publisher-token"])
def test_row_raw_status_coordinate_and_exact_fields_fail_closed(damage):
    source = {"fields": ["district_number", "new_hires_count"]}
    raw = ("014802", 0)
    row = native_row(source, raw)
    if damage == "raw":
        row["raw_new_hires_count"] = "0.0"
    elif damage == "status":
        row["status_new_hires_count"] = "missing"
    elif damage == "coordinate":
        row["publisher_worksheet_row"] = 3
    elif damage == "coordinate-type":
        row["publisher_worksheet_row"] = 2.0
    elif damage == "extra":
        row["unreviewed_field"] = 0
    elif damage == "missing":
        row.pop("new_hires_count")
    elif damage == "width":
        raw += ("extra",)
    elif damage == "count-type":
        row["new_hires_count"] = 0.0
    else:
        raw = ("014802", "*")
    with pytest.raises(ContractError, match="Educator hiring publisher row 2"):
        gate._check_row(row, raw, 2, source)


@pytest.mark.parametrize("damage", [None, "missing-zero", "extra-zero", "epp-count", "state-count", "total", "ratio"])
def test_cross_products_require_zero_keys_and_exact_counts_and_ratios(damage):
    categories = [["Intern", "Intern Certified", "intern_certified"], ["Emergency", "Emergency Permit", "emergency_permit"]]
    district = {(2010, "014802", "Intern Certified"): 0, (2016, "015950", "Intern Certified"): 1,
                (2016, "015950", "Emergency Permit"): 0}
    epp = district.copy()
    state = {2016: {"total_new_hires_count": 1, "intern_certified_count": 1, "intern_certified_ratio": 1.0,
                    "emergency_permit_count": 0, "emergency_permit_ratio": 0.0}}
    if damage == "missing-zero":
        epp.pop((2016, "015950", "Emergency Permit"))
    elif damage == "extra-zero":
        epp[2010, "000000", "Emergency Permit"] = 0
    elif damage == "epp-count":
        epp[2016, "015950", "Intern Certified"] = 2
    elif damage == "state-count":
        state[2016]["intern_certified_count"] = 0
    elif damage == "total":
        state[2016]["total_new_hires_count"] = 2
    elif damage == "ratio":
        state[2016]["intern_certified_ratio"] = 0.9999999999999999
    if damage is not None:
        with pytest.raises(ContractError, match="Educator hiring"):
            gate._cross_checks(district, epp, state, categories)
    else:
        result = gate._cross_checks(district, epp, state, categories)
        assert result["district_epp_years"] == [2010, 2016]
        assert result["direct_texas_years"] == [2016]
        assert result["district_epp_keys"] == 3 and result["zero_keys_preserved"] is True


def synthetic_family(tmp_path, monkeypatch, damage=None):
    review = plan()
    review["editions"] = {identifier: case for identifier, case in review["editions"].items() if case["school_year_end"] == 2016}
    review["goldens"] = []
    raw_rows, native_rows, paths, records = {}, {}, {}, {}
    for dataset, source in review["sources"].items():
        if dataset == _STATE:
            values = {"school_year_label": "2015-16", "total_new_hires_count": 2}
            for _, _, name in review["category_mapping"]:
                values[name + "_count"] = 2 if name == "intern_certified" else 0
                values[name + "_ratio"] = 1 if name == "intern_certified" else 0
            rows = [tuple(values[field] for field in source["fields"])]
        else:
            rows = []
            for index, (_, category, _) in enumerate(review["category_mapping"]):
                values = {"district_number": "014802", "school_year_end": 2016, "education_service_center": "01",
                          "certification_at_hire_category": category, "new_hires_count": 2 if index == 0 else 0,
                          "district_name": "Historical district", "district_geotype": "Rural",
                          "education_service_center_label": "Region 1", "school_year_label": "2015-16",
                          "publisher_epp_region": "NA", "reported_epp_assignment": "Na-Uncertified", "publisher_epp_region_label": "NA"}
                rows.append(tuple(values[field] for field in source["fields"]))
        raw_rows[dataset] = rows
        native_rows[dataset] = [native_row(source, raw, ordinal) for ordinal, raw in enumerate(rows, 2)]
        case = next(case for case in review["editions"].values() if case["key"]["dataset"] == dataset)
        case["rows"] = len(rows)
        review["goldens"].append({"dataset": dataset, "worksheet_row": 2, "facts": {source["fields"][0]: rows[0][0]}})
    if damage == "early-eof":
        native_rows[_DISTRICT].pop()
    elif damage == "surplus":
        native_rows[_DISTRICT].append(deepcopy(native_rows[_DISTRICT][-1]))
    elif damage == "missing-source-row":
        raw_rows[_DISTRICT].pop()
    elif damage == "changed-source-row":
        raw_rows[_DISTRICT][0] = ("999999", *raw_rows[_DISTRICT][0][1:])
    elif damage == "duplicate":
        raw_rows[_DISTRICT][1] = raw_rows[_DISTRICT][0]
        native_rows[_DISTRICT][1] = native_row(review["sources"][_DISTRICT], raw_rows[_DISTRICT][0], 3)
    elif damage == "golden":
        review["goldens"][0]["facts"][review["sources"][_DISTRICT]["fields"][0]] = "999999"
    for dataset, source in review["sources"].items():
        path = tmp_path / f"{dataset}.xlsx"
        book = openpyxl.Workbook()
        book.active.title = source["worksheet"]
        book.active.append(source["headers"])
        for row in raw_rows[dataset]:
            book.active.append(row)
        book.save(path)
        book.close()
        data = path.read_bytes()
        source["sha256"] = sha256(data).hexdigest()
        source["artifact"] = {"filename": path.name, "sha256": source["sha256"], "size_bytes": len(data)}
        paths[source["sha256"]] = path
        records[source["receipt"]] = {"artifact": source["artifact"], "requested_url": source["url"],
                                     "resolved_url": source["url"], "method": source["method"], "downloaded_at": source["downloaded_at"]}
    selected, _, _, _, _ = identity_release(review)
    datasets = {case["artifact_id"]: case["key"]["dataset"] for case in review["editions"].values()}
    selected.analytical_dataset.side_effect = lambda artifact: ds.dataset(pa.table({
        name: [row[name] for row in native_rows[datasets[artifact]]]
        for name in native_rows[datasets[artifact]][0]}))
    processing, publisher = Mock(spec=ProvenanceStore), Mock(spec=ProvenanceStore)
    processing.verify_record_closure.return_value = {"processing-proof": {}}
    publisher.verify_record_closure.return_value = {"publisher-proof": {}}
    publisher.read_record.side_effect = records.__getitem__
    publisher.verify_artifact.side_effect = paths.__getitem__
    opened, closed_scans = [], []
    load_workbook, scan = openpyxl.load_workbook, gate._native_rows
    def tracked_workbook(stream, **kwargs):
        assert kwargs == {"read_only": True, "data_only": False, "keep_links": False}
        book = load_workbook(stream, **kwargs)
        book.close = Mock(wraps=book.close)
        opened.append((stream, book))
        return book
    def tracked_scan(release, case):
        try:
            yield from scan(release, case)
        finally:
            closed_scans.append(case["artifact_id"])
    monkeypatch.setattr(openpyxl, "load_workbook", tracked_workbook)
    monkeypatch.setattr(gate, "_native_rows", tracked_scan)
    return selected, processing, publisher, review, opened, closed_scans


@pytest.mark.parametrize("damage", [None, "early-eof", "surplus", "missing-source-row", "changed-source-row", "duplicate", "golden"])
def test_small_workbooks_cover_rows_examples_and_close_on_failure(tmp_path, monkeypatch, damage):
    selected, processing, publisher, review, opened, closed_scans = synthetic_family(tmp_path, monkeypatch, damage)
    if damage is None:
        result = gate.qualify_educator_hiring(selected, processing, publisher, review)
        assert result["status"] == "PASS_REVIEWED_NATIVE" and result["publisher_source_examples"] == 3
        assert result["cross_product_comparisons"]["district_epp_keys"] == 9
        assert result["cross_product_comparisons"]["direct_texas_years"] == [2016]
        assert result["qualification_scope"] == "pinned_native_family"
        assert {dataset: report["selected_source_rows"] for dataset, report in result["source_parity"].items()} == {
            _DISTRICT: 9, _EPP: 9, _STATE: 1}
        assert len(opened) == len(closed_scans) == 3
    else:
        with pytest.raises(ContractError, match="Educator hiring"):
            gate.qualify_educator_hiring(selected, processing, publisher, review)
        assert len(opened) == len(closed_scans) == 1
    selected.verify.assert_called_once_with()
    for stream, book in opened:
        assert stream.closed
        book.close.assert_called_once_with()


def test_native_dispatch_uses_both_relocated_stores_and_propagates_missing_evidence(monkeypatch):
    review = plan()
    child, _, _, _, _ = identity_release(review)
    processing, publisher = Mock(spec=ProvenanceStore), Mock(spec=ProvenanceStore)
    processing.verify_record_closure.return_value = {"processing-proof": {}}
    publisher.verify_record_closure.return_value = {"publisher-proof": {}}
    retained = Mock(return_value=processing)
    qualify = Mock(return_value={"status": "PASS_REVIEWED_NATIVE"})
    monkeypatch.setattr(verification, "_retained_store", retained)
    monkeypatch.setattr(gate, "qualify_educator_hiring", qualify)
    selected = verification._native_review(child, publisher)
    assert selected.label == "educator hiring"
    retained.assert_called_once_with(child.path.parent / "provenance")
    assert selected.qualify() == {"status": "PASS_REVIEWED_NATIVE"}
    qualify.assert_called_once_with(child, processing, publisher, review)
    assert selected.evidence() == {"processing": ["processing-proof"], "publisher": ["publisher-proof"]}
    processing.verify_record_closure.assert_called_once_with([case["processing_id"] for case in review["editions"].values()])
    publisher.verify_record_closure.assert_called_once_with(review["publisher_roots"])
    child.analytical_dataset.assert_not_called()
    for store in (processing, publisher):
        store.verify_record_closure.side_effect = FileNotFoundError("retained evidence missing")
        with pytest.raises(FileNotFoundError, match="retained evidence missing"):
            selected.evidence()
        store.verify_record_closure.side_effect = None
    qualify.side_effect = ContractError("source parity failed")
    with pytest.raises(ContractError, match="source parity failed"):
        selected.qualify()


def test_full_retained_educator_hiring_qualification(monkeypatch):
    manifest = os.environ.get("TEADATA_EDUCATOR_HIRING_MANIFEST")
    publisher_root = os.environ.get("TEADATA_EDUCATOR_HIRING_STORE")
    if manifest is None and publisher_root is None:
        pytest.skip("Provide retained educator hiring manifest and publisher store for the full source replay")
    assert manifest and publisher_root, "Both retained educator hiring evidence paths are required"
    monkeypatch.setattr("teadata.release._download", lambda *a, **k: pytest.fail("Implicit source download"))
    monkeypatch.setattr("socket.socket.connect", lambda *a, **k: pytest.fail("Network access during retained source review"))
    monkeypatch.setattr("socket.socket.connect_ex", lambda *a, **k: pytest.fail("Network access during retained source review"))
    release = DataRelease(manifest, expected_sha256=_PIN, offline=True)
    result = verification._native_review(release, ProvenanceStore(publisher_root)).qualify()
    assert result["status"] == "PASS_REVIEWED_NATIVE" and result["publisher_source_examples"] == 9
    assert len(result["source_parity"]) == 3
    assert result["cross_product_comparisons"]["district_epp_years"] == list(range(2010, 2027))
    assert result["cross_product_comparisons"]["direct_texas_years"] == list(range(2016, 2027))
