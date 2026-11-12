"""One fail-closed qualification plan for native and composed releases."""

import importlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest

from teadata import native_financial_accounts, native_staff_spp, release, verification
from teadata.dataset_metadata import DatasetMetadata, ResolvedDataset
from teadata.provenance import ProvenanceStore
from teadata.release import DataRelease


@pytest.mark.parametrize("module,pin_name,qualifier,collector,label", [
    ("first_ratings", "REVIEWED_MANIFEST", "qualify_first_ratings", "retained_first_evidence", "FIRST"),
    ("native_finance_staff", "REVIEWED_MANIFEST", "qualify_finance_staff", "retained_finance_staff_evidence", "finance and staff"),
    ("tprs_statewide_intake", "REVIEWED_STATEWIDE_TPRS_MANIFESTS", "qualify_statewide_tprs", "retained_tprs_evidence", "TPRS"),
    ("tprs_class_size_native", "REVIEWED_CLASS_SIZE_MANIFEST", "qualify_class_sizes", "retained_class_size_evidence", "class size"),
])
def test_existing_family_plan_keeps_evidence_separate_from_qualification(monkeypatch, module, pin_name, qualifier, collector, label):
    family = importlib.import_module("teadata." + module)
    pins = getattr(family, pin_name)
    pin = pins if isinstance(pins, str) else next(iter(pins))
    selected = cast(DataRelease, SimpleNamespace(manifest_sha256=pin))
    store = Mock(spec=ProvenanceStore)
    qualify = Mock(return_value={"status": "PASS_REVIEWED_NATIVE"})
    evidence = Mock(return_value=["receipt"])
    monkeypatch.setattr(family, qualifier, qualify)
    monkeypatch.setattr(family, collector, evidence)
    plan = verification._native_review(selected, store)
    assert plan.label == label
    assert plan.evidence() == ["receipt"]
    qualify.assert_not_called()
    assert plan.qualify() == {"status": "PASS_REVIEWED_NATIVE"}
    qualify.assert_called_once_with(selected, store)
    evidence.assert_called_once_with(selected, store)


@pytest.mark.parametrize("family", ["staff", "spp", "budget"])
@pytest.mark.parametrize("status", ["PASS_REVIEWED_NATIVE", "HOLD"])
def test_parent_and_standalone_use_the_same_additional_family_gate(tmp_path, monkeypatch, family, status):
    pin = "1" * 64
    child = cast(DataRelease, SimpleNamespace(manifest_sha256=pin, path=tmp_path / "child/release.json"))
    ref = SimpleNamespace(manifest_sha256=pin)
    parent = SimpleNamespace(build_id="parent", verify=lambda: None,
                             catalog=SimpleNamespace(native_sources=[ref], to_dict=lambda: {"schema_version": 2}))
    witness, processing = Mock(spec=ProvenanceStore), Mock(spec=ProvenanceStore)
    selected_plan = (family, 3, {"edition": [{"receipt": "witness"}]})
    qualify = Mock(return_value={"status": status})
    monkeypatch.setattr(native_financial_accounts, "is_reviewed_budget", lambda selected: selected == pin and family == "budget")
    monkeypatch.setattr(native_financial_accounts, "qualify_budget", qualify)
    monkeypatch.setattr(native_staff_spp, "reviewed_staff_spp", lambda selected: selected_plan if selected == pin and family != "budget" else None)
    monkeypatch.setattr(native_staff_spp, "qualify_staff_spp", qualify)
    parent_path = tmp_path / "parent.json"
    monkeypatch.setattr(release, "DataRelease", lambda path, **options: parent if Path(path) == parent_path else child)
    monkeypatch.setattr(verification, "_retained_store", lambda path: processing if path == child.path.parent / "provenance" else witness)
    monkeypatch.setattr(verification, "_open_reviewed_native_evidence", lambda *args: (child, witness))
    monkeypatch.setattr(verification, "_validate_release_provenance", lambda *args: 0)
    monkeypatch.setattr(verification, "_validate_publisher_cases", lambda *args: {})
    monkeypatch.setattr(verification, "_validate_tefa_cases", lambda *args: {})
    assert verification._validate_native_release(child.path, pin, tmp_path / "witnesses") == {"status": status}
    expected = (child, processing, witness) + (() if family == "budget" else (selected_plan,))
    qualify.assert_called_once_with(*expected)
    if status == "HOLD":
        with pytest.raises(verification.ReceiptError, match="did not pass qualification"):
            verification._validate_release(str(parent_path), "app")
    else:
        verification._validate_release(str(parent_path), "app")
    assert qualify.call_count == 2
    assert qualify.call_args.args == expected


def test_multiple_same_family_attachments_keep_manifest_bound_evidence(tmp_path, monkeypatch):
    pins = ["1" * 64, "2" * 64]
    asset = tmp_path / "entities.sqlite"
    asset.write_bytes(b"base")
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps({"artifacts": {"entities": {"filename": asset.name}}, "catalog": {"native_sources": [{}]}}))
    parent = SimpleNamespace(catalog=SimpleNamespace(native_sources=[SimpleNamespace(manifest_sha256=pin) for pin in pins]),
                             verify_native_sources=lambda: {pin: {"asset": asset} for pin in pins})
    monkeypatch.setattr(verification, "source_identity", lambda root: {})
    monkeypatch.setattr(release, "DataRelease", lambda *args, **options: parent)
    monkeypatch.setattr(verification, "_open_reviewed_native_evidence", lambda selected, ref, label: (ref, None))
    qualify = Mock(side_effect=AssertionError("Evidence capture performed semantic replay"))
    monkeypatch.setattr(verification, "_native_review", lambda child, store: verification._NativeReview(
        "same-family", qualify, lambda: [child.manifest_sha256]))
    captured = verification._input_identity(tmp_path, kind="library", paired_library=None, release=str(manifest))
    assert captured["native_evidence"] == {
        pin: {"family": "same-family", "records": [pin]} for pin in pins}
    qualify.assert_not_called()


def _retained_fixture(tmp_path):
    (tmp_path / ".local").mkdir(exist_ok=True)
    processing_store = ProvenanceStore(tmp_path / ".local/child/provenance")
    publisher_store = ProvenanceStore(tmp_path / ".local/publisher")

    def acquire(store, name, content):
        path = tmp_path / ".local" / name
        path.write_text(content)
        definition = ResolvedDataset("example", "data_sources", 2026, 2026, str(path), DatasetMetadata.from_dict({}))
        return store.record_acquisition(path, definition)

    raw = acquire(processing_store, "raw.csv", "value\n42\n")
    publisher = acquire(publisher_store, "independent.html", "<p>42</p>")
    leaf = tmp_path / ".local/leaf.parquet"
    leaf.write_bytes(b"fixture leaf, not an actual Parquet artifact")
    processing = processing_store.record_processing(dataset="example", inputs=[raw], outputs={"leaf": leaf},
                                                   processor="fixture", code_version="fixture")
    child = cast(DataRelease, SimpleNamespace(manifest_sha256="1" * 64, path=processing_store.root.parent / "release.json",
        catalog=SimpleNamespace(to_dict=lambda: {"artifacts": {"leaf": {}}},
                                artifact_ref=lambda identifier: SimpleNamespace(processing_id=processing))))
    return child, processing_store, publisher_store, raw, processing, publisher


@pytest.mark.parametrize("target,action", [("raw", "corrupt"), ("publisher", "corrupt"), ("publisher", "delete")])
def test_receipt_reuse_rehashes_raw_and_independent_bytes_without_semantic_replay(tmp_path, monkeypatch, target, action):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    for name, value in (("user.name", "Fixture"), ("user.email", "fixture@example.invalid")):
        subprocess.run(["git", "-C", str(tmp_path), "config", name, value], check=True)
    (tmp_path / ".gitignore").write_text(".local/\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "--quiet", "-m", "Fixture"], check=True)
    child, processing_store, publisher_store, raw, processing, publisher = _retained_fixture(tmp_path)
    pin = child.manifest_sha256
    asset = tmp_path / "entities.sqlite"
    asset.write_bytes(b"base")
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps({"artifacts": {"entities": {"filename": asset.name}}, "catalog": {"native_sources": [{}]}}))
    parent = SimpleNamespace(catalog=SimpleNamespace(native_sources=[SimpleNamespace(manifest_sha256=pin)]),
                             verify_native_sources=lambda: {pin: {"asset": asset}})
    monkeypatch.setattr(release, "DataRelease", lambda *args, **options: parent)
    monkeypatch.setattr(verification, "_open_reviewed_native_evidence", lambda *args: (child, publisher_store))
    monkeypatch.setattr(native_staff_spp, "reviewed_staff_spp", lambda selected: ("staff", 3, {"edition": [{"receipt": publisher}]}))
    qualifier = Mock(side_effect=AssertionError("Receipt identity performed semantic qualification"))
    monkeypatch.setattr(native_staff_spp, "qualify_staff_spp", qualifier)
    monkeypatch.setattr(verification, "_environment_identity", lambda *args: {"python": verification._controller_python_version()})
    checks = [verification.Check("fixture", ("@python", "-c", "pass"))]
    receipt = verification.run_checks(tmp_path, kind="library", checks=checks, release=str(manifest))
    document = verification.verify_receipt(receipt, tmp_path, kind="library", checks=checks, release=str(manifest))
    records = document["inputs"]["native_evidence"][pin]["records"]
    assert records == {"processing": sorted([raw, processing]), "publisher": [publisher]}
    store, identifier = (processing_store, raw) if target == "raw" else (publisher_store, publisher)
    archived = store.verify_artifact(store.read_record(identifier)["artifact"]["sha256"])
    if action == "delete":
        archived.unlink()
    else:
        archived.write_bytes(b"changed")
    with pytest.raises(verification.ReceiptError, match="retained native sources"):
        verification.verify_receipt(receipt, tmp_path, kind="library", checks=checks, release=str(manifest))
    qualifier.assert_not_called()


def test_budget_evidence_captures_independent_pwr_roots(tmp_path, monkeypatch):
    child, processing_store, publisher_store, raw, processing, publisher = _retained_fixture(tmp_path)
    key = SimpleNamespace(dataset="budget")
    monkeypatch.setattr(child, "catalog", SimpleNamespace(
        to_dict=lambda: {"artifacts": {"leaf": {}}},
        artifact_ref=lambda identifier: SimpleNamespace(processing_id=processing, edition=key),
        metadata=lambda *args, **kwargs: SimpleNamespace(references=[
        SimpleNamespace(reference_id="budget_pwr", provenance_ids=[publisher]),
        SimpleNamespace(reference_id="unrelated", provenance_ids=["not-required"]),
    ])))
    monkeypatch.setattr(native_financial_accounts, "is_reviewed_budget", lambda pin: True)
    qualifier = Mock(side_effect=AssertionError("Evidence capture performed budget replay"))
    monkeypatch.setattr(native_financial_accounts, "qualify_budget", qualifier)
    plan = verification._native_review(child, publisher_store)
    assert plan.evidence() == {"processing": sorted([raw, processing]), "publisher": [publisher]}
    qualifier.assert_not_called()
