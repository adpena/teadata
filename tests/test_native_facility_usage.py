"""Independent facility checks; retained-artifact cases opt in to private custody."""
from copy import deepcopy
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import shutil
from unittest.mock import Mock

import pytest

from teadata import native_facility_usage as gate, verification
from teadata.manifest import ContractError
from teadata.provenance import ProvenanceStore
from teadata.release import DataRelease, _schema_publication_destinations


@pytest.fixture
def retained_facility():
    location = os.environ.get("TEADATA_FACILITY_TEST_ROOT")
    if not location:
        pytest.skip("Actual facility artifact tests require explicit retained private custody")
    root = Path(location)
    return root, DataRelease(root / "release.json", offline=True), json.loads((root / "REVIEW_PLAN.json").read_text())


def test_frozen_facility_semantics_keep_exact_source_examples_and_decimal_contract():
    plan = gate.review_template()
    assert len(plan["physical_contract"]) == 31
    assert {f["coordinate"]: f["type"] for f in plan["physical_contract"] if f["type"].startswith("decimal")} == {
        "Q": "decimal128(19,12)", "R": "decimal128(24,18)", "AA": "decimal128(18,13)",
        "AB": "decimal128(18,13)", "AC": "decimal128(9,1)"}
    assert plan["goldens"]["AC16356"] == "13721607"
    assert plan["goldens"]["AB11029"] is None
    assert plan["expected"]["category_cells"] == {"U": 116, "V": 27, "W": 18, "X": 8}
    assert "C" not in {f["coordinate"] for f in plan["physical_contract"]}


def test_independent_values_reject_float_rounding_and_string_status_support():
    fields = gate.review_template()["physical_contract"]
    acreage = next(f for f in fields if f["coordinate"] == "R")
    row = {"acreage": Decimal("0.008500000000000001"), "raw_acreage": "0.008500000000000001", "status_acreage": "observed"}
    gate._check_fields(row, {"R": row["raw_acreage"]}, [acreage], 11813)
    for damaged in ({**row, "acreage": float(row["acreage"])}, {**row, "status_acreage": "missing"}):
        with pytest.raises(ContractError):
            gate._check_fields(damaged, {"R": row["raw_acreage"]}, [acreage], 11813)
    text = next(f for f in fields if f["coordinate"] == "AF")
    with pytest.raises(ContractError, match="raw-only"):
        gate._check_fields({"submission_status": "INCOMPLETE", "raw_submission_status": "INCOMPLETE",
                            "status_submission_status": "observed"}, {"AF": "INCOMPLETE"}, [text], 2)


def test_missing_retained_store_is_not_created(tmp_path):
    missing = tmp_path / "absent"
    with pytest.raises(verification.ReceiptError, match="existing retained"):
        verification._retained_store(missing)
    assert not missing.exists()


def test_unknown_facility_pin_does_not_admit_a_source_family():
    assert gate.reviewed_facility_usage("0" * 64) is None


@pytest.mark.parametrize("damage", ["manifest", "build", "edition"])
def test_actual_identity_damage_fails_before_observation_reads(retained_facility, monkeypatch, damage):
    _, release, plan = retained_facility
    damaged = deepcopy(plan)
    if damage == "manifest":
        damaged["manifest_sha256"] = "0" * 64
    elif damage == "build":
        damaged["build_id"] = "0" * 64
    else:
        damaged["editions"].pop(next(iter(damaged["editions"])))
    monkeypatch.setattr(release, "analytical_dataset", Mock(side_effect=AssertionError("Premature data scan")))
    with pytest.raises(ContractError, match="identity"):
        gate.validate_facility_identity(release, damaged)


@pytest.mark.parametrize("damage", ["missing", "wrong"])
def test_missing_or_wrong_private_object_blocks_gate_and_reuse(retained_facility, tmp_path, monkeypatch, damage):
    root, release, plan = retained_facility
    original = ProvenanceStore(root / "provenance")
    store = ProvenanceStore(tmp_path / "private")
    for rid in plan["publisher_roots"]:
        store.copy_record_from(original, rid)
    target = store.root / "objects" / plan["source_sha256"]
    # Copies belong to this fixture; never mutate retained or hardlinked bytes.
    target.unlink()
    if damage == "wrong":
        target.write_bytes(b"different publisher bytes")
    monkeypatch.setattr(gate, "reviewed_facility_usage", lambda pin: plan if pin == release.manifest_sha256 else None)
    review = verification._native_review(release, store)
    for action in (review.qualify, review.evidence):
        with pytest.raises((ValueError, OSError)):
            action()


def test_public_only_assets_read_without_private_store(retained_facility, tmp_path, monkeypatch):
    root, release, plan = retained_facility
    public = tmp_path / "public"
    public.mkdir()
    (public / "release.json").write_bytes(release.path.read_bytes())
    _, names = _schema_publication_destinations(release.catalog)
    for name in names:
        assert Path(name).name == name and name.endswith((".json", ".parquet"))
        shutil.copyfile(root / name, public / name)
        if name.endswith(".json"):
            document = (public / name).read_text()
            assert "/Users/" not in document and ".local/" not in document
            assert not re.search(r"[^\s@]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+", document)
    assert not (public / "provenance").exists()
    selected = DataRelease(public / "release.json", offline=True)
    selected.verify()
    def no_private(*args, **kwargs):
        pytest.fail("Public reader accessed private provenance")
    monkeypatch.setattr(ProvenanceStore, "verify_record_closure", no_private)
    table = selected.observations("facility_usage_submission", "submission_status", scope="district", school_year=2026)
    assert table.to_arrow().num_rows == 1017
    assert table.to_df().shape[0] == 1017
    table.to_csv(public / "submission.csv")
    table.to_excel(public / "submission.xlsx")
    assert "Contact Email Address" not in (public / "submission.csv").read_text()
    assert plan["source_sha256"] not in {p.name for p in public.iterdir()}


@pytest.mark.parametrize("damage", ["extraction_parent", "processing_input", "grouping_recipe"])
def test_changed_private_lineage_blocks_qualification_and_evidence(retained_facility, tmp_path, monkeypatch, damage):
    root, release, plan = retained_facility
    original = ProvenanceStore(root / "provenance")
    store = ProvenanceStore(tmp_path / "provenance")
    artifacts = [release.catalog.artifact_ref(aid) for aid in release.catalog.to_dict()["artifacts"]]
    for rid in [*plan["publisher_roots"], *(ref.processing_id for ref in artifacts)]:
        store.copy_record_from(original, rid)
    if damage == "extraction_parent":
        rid = plan["publisher_roots"][0]
        record = store.read_record(rid)
        record["derived_from"] = "0" * 64
    else:
        rid = next(ref.processing_id for ref in artifacts if ref.edition.dataset == "facility_usage_submission")
        record = store.read_record(rid)
        if damage == "processing_input":
            record["inputs"] = []
        else:
            record["parameters"]["recipe"]["require_equal"] = ["B"]
    # Replace a fixture copy, never alter the retained record or its hardlinks.
    path = store.root / "records" / (rid + ".json")
    path.unlink()
    path.write_text(json.dumps(record))
    monkeypatch.setattr(gate, "reviewed_facility_usage", lambda pin: plan if pin == release.manifest_sha256 else None)
    monkeypatch.setattr(verification, "_retained_store", lambda path: store)
    review = verification._native_review(release, store)
    for action in (review.qualify, review.evidence):
        with pytest.raises((ValueError, OSError)):
            action()


def test_missing_adjacent_processing_store_blocks_native_dispatch(retained_facility, tmp_path, monkeypatch):
    root, release, plan = retained_facility
    publisher = ProvenanceStore(root / "provenance")
    _, names = _schema_publication_destinations(release.catalog)
    for name in names:
        shutil.copyfile(root / name, tmp_path / name)
    shutil.copyfile(release.path, tmp_path / "release.json")
    release = DataRelease(tmp_path / "release.json", offline=True)
    monkeypatch.setattr(gate, "reviewed_facility_usage", lambda pin: plan if pin == release.manifest_sha256 else None)
    with pytest.raises(verification.ReceiptError, match="existing retained"):
        verification._native_review(release, publisher)
    assert not (tmp_path / "provenance").exists()
