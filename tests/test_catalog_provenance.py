"""Public catalog validation must reject well-hashed but contradictory evidence."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256

import pytest

from teadata.dataset_metadata import DataPeriod, DatasetMetadata, ReferenceDocument, ResolvedDataset
from teadata.manifest import ContractError, DatasetCatalog, canonical_json
from teadata.provenance import ProvenanceStore


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


@pytest.mark.parametrize("typed_producer", [False, True])
def test_source_downloads_follow_only_exact_data_lineage(tmp_path, typed_producer):
    store = ProvenanceStore(tmp_path / "provenance")
    date = datetime(2026, 11, 4, tzinfo=timezone.utc)
    current = DatasetMetadata(title="Campus transfers", version="2026", period=DataPeriod("school_year", 2025, 2026))

    def capture(name, metadata, *, dataset="sample", year=2026):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        url = "https://publisher.example/" + name
        return store.record_acquisition(path, ResolvedDataset(dataset, "data_sources", year, year, str(path), metadata),
                                        downloaded_at=date, recorded_at=date, requested_url=url, resolved_url=url)

    original = capture("statewide.csv", current)
    other_year = capture("prior.csv", replace(current, version="2025", period=DataPeriod("school_year", 2024, 2025)),
                         year=2025)
    other_period = capture("different-period.csv", replace(current, period=DataPeriod("school_year", 2024, 2025)))
    other_dataset = capture("other.csv", current, dataset="other")
    reference = capture("methodology.csv", current)
    methodology = ReferenceDocument("methodology", "methodology", url="https://publisher.example/methodology.csv",
                                    provenance_ids=(reference,))
    path = tmp_path / "prepared.parquet"
    path.write_bytes(b"prepared transfer bytes")
    parents = (original, other_year, other_period, other_dataset)
    producer = None
    references = (methodology,)
    if typed_producer:
        producer = store.record_processing(dataset="sample", inputs=(*parents, reference), outputs={"prepared": path},
                                           processor="test", code_version="0" * 64, recorded_at=date)
    else:
        references += (ReferenceDocument("masking", "masking_methodology", url="https://publisher.example/statewide.csv",
                                         provenance_ids=(original,)),)
    metadata = replace(current, provenance_ids=() if typed_producer else parents, references=references)
    prepared = store.record_acquisition(path, ResolvedDataset("sample", "data_sources", 2026, 2026, str(path), metadata),
                                        derived_from=producer, recorded_at=date)
    records = store.verify_record_closure([prepared])
    catalog = DatasetCatalog.create(
        datasets={"sample": {"status": "loaded", "section": "data_sources",
                             "metadata": replace(metadata, provenance_ids=(prepared,)).to_dict(),
                             "artifact": records[prepared]["artifact"]}},
        records=records, value_policies={}, metric_bindings={}, processor="0" * 64,
    )
    assert [row["record_id"] for row in catalog.acquisitions("sample")] == [prepared]
    download, = catalog.source_downloads("sample")
    assert download["record_id"] == original
    assert download["downloaded_at"] == date.isoformat()
    download["source_metadata"]["title"] = "Changed by caller"
    assert catalog.source_downloads("sample")[0]["source_metadata"]["title"] == current.title


def test_source_downloads_preserve_direct_unknown_period(catalog_inputs):
    metadata, artifact, record = catalog_inputs
    record.update(method="download", downloaded_at="2025-01-01T00:00:00+00:00")
    catalog = make_catalog(metadata, artifact, [record])
    assert catalog.source_downloads("sample") == catalog.acquisitions("sample")
