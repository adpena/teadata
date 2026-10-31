from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256

import pytest
import yaml

from teadata.dataset_metadata import DatasetMetadata, ReferenceDocument
from teadata.manifest import ContractError, verify_source_receipts
from teadata.provenance import ProvenanceStore
from teadata.sources import observe_sources
from teadata.teadata_config import load_config


def configuration(tmp_path, references=()):
    source = tmp_path / "sample.csv"
    source.write_text("value\n42\n")
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump({"data_sources": {"sample": {2025: {
        "path": str(source), "metadata": DatasetMetadata(title="Original", references=tuple(references)).to_dict(),
    }}}}))
    return path


def test_improved_metadata_retains_actual_download_history(tmp_path):
    path = configuration(tmp_path)
    store = ProvenanceStore(tmp_path / "provenance")
    source = load_config(path).resolve_dataset("sample")
    receipt = store.record_acquisition(source.path, source, downloaded_at=datetime(2025, 1, 1, tzinfo=timezone.utc), requested_url="https://example.org/source.csv")
    raw = yaml.safe_load(path.read_text())
    metadata = raw["data_sources"]["sample"][2025]["metadata"]
    metadata.update(title="Reviewed title", provenance_ids=[receipt])
    path.write_text(yaml.safe_dump(raw))
    observe_sources(path, store.root)
    updated = load_config(path).resolve_dataset("sample")
    assert len(updated.metadata.provenance_ids) == 2
    assert updated.metadata.provenance_ids[0] == receipt
    original, observation = [store.read_record(i) for i in updated.metadata.provenance_ids]
    assert original["method"] == "download"
    assert original["downloaded_at"] == "2025-01-01T00:00:00+00:00"
    assert observation["method"] == "local_observation"
    assert observation["downloaded_at"] is None
    verify_source_receipts(updated, store, reference_base=tmp_path)
    observe_sources(path, store.root)
    assert load_config(path).resolve_dataset("sample").metadata.provenance_ids == updated.metadata.provenance_ids


def test_source_observation_does_not_approve_changed_reviewed_document(tmp_path):
    document = tmp_path / "methodology.md"
    document.write_text("Reviewed masking rules")
    reference = ReferenceDocument.from_dict({"reference_id": "masking", "role": "masking_methodology", "status": "available", "url": "https://example.org/masking", "path": document.name, "reviewed_sha256": sha256(document.read_bytes()).hexdigest()})
    path = configuration(tmp_path, [reference])
    store = ProvenanceStore(tmp_path / "provenance")
    observe_sources(path, store.root)
    before = path.read_bytes()
    document.write_text("Different masking rules requiring review")
    with pytest.raises(ContractError, match="reviewed methodology"):
        observe_sources(path, store.root)
    assert path.read_bytes() == before


def test_selected_receipts_cannot_include_other_artifact_bytes(tmp_path):
    path = configuration(tmp_path)
    store = ProvenanceStore(tmp_path / "provenance")
    observe_sources(path, store.root)
    source = load_config(path).resolve_dataset("sample")
    other = tmp_path / "other.csv"
    other.write_text("value\n99\n")
    wrong = store.record_acquisition(other, source)
    source = replace(source, metadata=replace(source.metadata, provenance_ids=(*source.metadata.provenance_ids, wrong)))
    with pytest.raises(ContractError, match="Input bytes"):
        verify_source_receipts(source, store, reference_base=tmp_path)


def test_unresolved_transfers_block_release_and_source_review_is_not_invented():
    from teadata import DataEngine
    from teadata.load_data import _release_validation
    repo = DataEngine()
    result = _release_validation(repo)
    assert result["referential_integrity"] == "passed"
    assert result["source_review_status"] == "pending"
    assert result["quality_status"] == "needs_review"
    repo._xfers_missing = {"either": 1}
    with pytest.raises(ContractError, match="unresolved transfer endpoints"):
        _release_validation(repo)
