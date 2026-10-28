"""Public catalog validation must reject well-hashed but contradictory evidence."""

from copy import deepcopy
from hashlib import sha256

import pytest

from teadata.dataset_metadata import DatasetMetadata
from teadata.manifest import ContractError, DatasetCatalog, canonical_json


@pytest.fixture
def catalog_inputs():
    metadata = DatasetMetadata(title="Reviewed current description").to_dict()
    artifact = {"filename": "sample.csv", "sha256": "a" * 64, "size_bytes": 12}
    record = {
        "kind": "acquisition", "dataset": "sample", "section": "data_sources",
        "artifact": dict(artifact), "source_metadata": deepcopy(metadata),
        "method": "local_observation", "downloaded_at": None,
    }
    return metadata, artifact, record


def make_catalog(metadata, artifact, records, *, section="data_sources"):
    """Rehash every modified receipt so failures test semantics, not checksums."""
    receipts = {
        sha256((canonical_json(record) + "\n").encode()).hexdigest(): record
        for record in records
    }
    metadata = {**metadata, "provenance_ids": list(receipts)}
    return DatasetCatalog.create(
        datasets={"sample": {"status": "loaded", "section": section,
                             "metadata": metadata, "artifact": artifact,
                             "reads": [{"columns": ["count"], "rows": 1}]}},
        records=receipts, value_policies={}, metric_bindings={}, processor="0" * 64,
    )


@pytest.mark.parametrize("field,value", [
    ("kind", "processing"), ("kind", None),
    ("dataset", "different_source"), ("dataset", None),
    ("section", "spatial"), ("section", None),
])
def test_public_catalog_rejects_other_events_sources_and_sections(catalog_inputs, field, value):
    metadata, artifact, record = catalog_inputs
    record[field] = value
    with pytest.raises(ContractError, match="section, or event kind"):
        make_catalog(metadata, artifact, [record])


@pytest.mark.parametrize("field,value", [
    ("sha256", "b" * 64), ("size_bytes", 11), ("size_bytes", -1),
    ("size_bytes", 12.0), ("size_bytes", "12"), ("size_bytes", True),
    ("size_bytes", None),
])
def test_public_catalog_rejects_mismatched_receipt_artifacts(catalog_inputs, field, value):
    metadata, artifact, record = catalog_inputs
    record["artifact"][field] = value
    with pytest.raises(ContractError, match="Input bytes"):
        make_catalog(metadata, artifact, [record])


@pytest.mark.parametrize("size", [-1, True, 12.0, "12", None])
def test_public_catalog_requires_nonnegative_integer_source_size(catalog_inputs, size):
    metadata, artifact, record = catalog_inputs
    artifact["size_bytes"] = size
    record["artifact"]["size_bytes"] = size
    with pytest.raises(ContractError, match="input checksum"):
        make_catalog(metadata, artifact, [record])


@pytest.mark.parametrize("section", [None, "reference", "", 1])
def test_public_catalog_requires_a_declared_source_section(catalog_inputs, section):
    metadata, artifact, record = catalog_inputs
    record["section"] = section
    with pytest.raises(ContractError, match="source section"):
        make_catalog(metadata, artifact, [record], section=section)


def test_public_catalog_cannot_relabel_bytes_without_a_current_metadata_observation(catalog_inputs):
    metadata, artifact, record = catalog_inputs
    record["source_metadata"]["title"] = "Earlier unreviewed description"
    with pytest.raises(ContractError, match="metadata differs"):
        make_catalog(metadata, artifact, [record])


@pytest.mark.parametrize("historical_first", [True, False])
def test_same_bytes_historical_receipts_remain_usable_with_a_current_observation(catalog_inputs, historical_first):
    metadata, artifact, current = catalog_inputs
    historical = deepcopy(current)
    historical["source_metadata"]["title"] = "Earlier description"
    historical.update(method="download", downloaded_at="2025-01-01T00:00:00+00:00")
    # Receipt selection changes are excluded from metadata equivalence; they
    # cannot be known before a receipt computes its own content identity.
    current["source_metadata"]["provenance_ids"] = ["f" * 64]
    records = [historical, current] if historical_first else [current, historical]
    selected = make_catalog(metadata, artifact, records)
    observations = selected.acquisitions("sample")
    assert len(observations) == 2
    assert {item["method"] for item in observations} == {"download", "local_observation"}
    assert selected.metadata("sample").title == metadata["title"]


def test_one_correct_receipt_cannot_hide_other_selected_input_bytes(catalog_inputs):
    metadata, artifact, current = catalog_inputs
    contradictory = deepcopy(current)
    contradictory["artifact"]["sha256"] = "b" * 64
    with pytest.raises(ContractError, match="Input bytes"):
        make_catalog(metadata, artifact, [current, contradictory])
