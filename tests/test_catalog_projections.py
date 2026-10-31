"""Catalog projections detach selected metadata without decoding the whole release."""

from copy import deepcopy
import gzip
from hashlib import sha256
import pickle
from pathlib import Path

import pytest

from teadata.engine import DataEngine
from teadata.manifest import ContractError, DatasetCatalog, canonical_json
from teadata.release import SNAPSHOT_MAGIC


@pytest.fixture
def selected_catalog():
    metadata = {"title": "Current description", "period": {"kind": "unknown"}}
    artifact = {"filename": "sample.csv", "sha256": "a" * 64, "size_bytes": 12}
    records = []
    for title, method, date in [("Earlier description", "download", "2024-01-01T00:00:00+00:00"),
                                ("Current description", "local_observation", None)]:
        records.append({"kind": "acquisition", "dataset": "sample", "section": "data_sources",
            "artifact": dict(artifact), "source_metadata": {**metadata, "title": title},
            "method": method, "downloaded_at": date,
            "parameters": {"selection": {"keys": ["A", "B"]}}})
    receipts = {sha256((canonical_json(r) + "\n").encode()).hexdigest(): r for r in records}
    first, second = receipts
    return DatasetCatalog.create(
        datasets={"sample": {"status": "loaded", "section": "data_sources",
            "metadata": {**metadata, "provenance_ids": [first, second, first]},
            "artifact": artifact, "reads": [{"columns": ["value"], "rows": 1}]}},
        records=receipts, value_policies={}, metric_bindings={}, processor="0" * 64,
    )


def test_selected_projections_preserve_order_unknown_dates_and_detach_every_nested_return(selected_catalog):
    catalog = selected_catalog
    original = catalog.to_dict()
    build_id = catalog.build_id
    sources = catalog.datasets
    observations = catalog.acquisitions("sample")
    assert [o["method"] for o in observations] == ["download", "local_observation", "download"]
    assert [o["downloaded_at"] for o in observations] == ["2024-01-01T00:00:00+00:00", None, "2024-01-01T00:00:00+00:00"]
    assert observations[0]["record_id"] == observations[2]["record_id"]
    sources["sample"]["reads"][0]["columns"].append("caller-only")
    sources["sample"]["metadata"]["period"]["kind"] = "changed by caller"
    observations[0]["artifact"]["sha256"] = "b" * 64
    observations[0]["parameters"]["selection"]["keys"].append("caller-only")
    observations[0]["source_metadata"]["title"] = "changed by caller"
    assert observations[2]["parameters"]["selection"]["keys"] == ["A", "B"]
    assert observations[2]["artifact"]["sha256"] == "a" * 64
    assert catalog.datasets == original["datasets"]
    assert catalog.acquisitions("sample")[0]["source_metadata"]["title"] == "Earlier description"
    assert catalog.to_dict() == original and catalog.build_id == build_id


def test_selected_projections_need_no_full_document_parse_or_file_io(selected_catalog, monkeypatch):
    expected_sources = selected_catalog.datasets
    expected_observations = selected_catalog.acquisitions("sample")
    with monkeypatch.context() as forbidden:
        forbidden.setattr("teadata.manifest.json.loads", lambda *a, **k: pytest.fail("whole document decoded"))
        forbidden.setattr(Path, "open", lambda *a, **k: pytest.fail("artifact/file I/O"))
        forbidden.setattr("builtins.open", lambda *a, **k: pytest.fail("artifact/file I/O"))
        forbidden.setattr(DataEngine, "from_snapshot", lambda *a, **k: pytest.fail("engine load"))
        assert selected_catalog.datasets == expected_sources
        assert selected_catalog.acquisitions("sample") == expected_observations
        assert selected_catalog.period_label("sample") == "Period not specified"
        with pytest.raises(ContractError, match="Dataset was not loaded in this release"):
            selected_catalog.acquisitions("not-loaded")


def test_derived_record_lookup_is_not_pickled_and_old_default_state_restores_it(selected_catalog):
    catalog = selected_catalog
    expected = catalog.to_dict()
    observations = catalog.acquisitions("sample")
    # A non-pickleable value in the derived lookup must never enter snapshot state.
    catalog._records["derived-only"] = lambda: None
    restored = pickle.loads(pickle.dumps(catalog))
    assert restored.to_dict() == expected and restored.acquisitions("sample") == observations
    assert "derived-only" not in restored._records
    legacy_state = catalog.__getstate__()
    assert "_records" not in legacy_state
    old_default_state = DatasetCatalog.__new__(DatasetCatalog)
    old_default_state.__setstate__(legacy_state)
    assert old_default_state.to_dict() == expected
    assert old_default_state.acquisitions("sample") == observations


@pytest.mark.parametrize("invalid", [False, True])
def test_snapshot_load_still_revalidates_canonical_document(tmp_path, selected_catalog, invalid):
    catalog = selected_catalog
    if invalid:
        document = catalog.to_dict()
        document["datasets"]["sample"]["metadata"]["title"] = "Unvalidated metadata"
        catalog._serialized = canonical_json(document)
    engine = DataEngine()
    engine.catalog = catalog
    path = tmp_path / "empty-engine.gz"
    with gzip.open(path, "wb") as stream:
        stream.write(SNAPSHOT_MAGIC + catalog.build_id.encode() + b"\n")
        pickle.dump(engine, stream, protocol=pickle.HIGHEST_PROTOCOL)
    if invalid:
        with pytest.raises(ContractError, match="catalog identity mismatch"):
            DataEngine.from_snapshot(path)
    else:
        loaded = DataEngine.from_snapshot(path)
        assert loaded.catalog is not catalog
        assert loaded.catalog.to_dict() == catalog.to_dict()
        assert loaded.catalog.acquisitions("sample") == catalog.acquisitions("sample")


def test_projection_optimization_does_not_weaken_receipt_validation(selected_catalog):
    document = selected_catalog.to_dict()
    identifier = next(iter(document["provenance"]))
    record = deepcopy(document["provenance"].pop(identifier))
    record["dataset"] = "unrelated"
    replacement = sha256((canonical_json(record) + "\n").encode()).hexdigest()
    document["provenance"][replacement] = record
    metadata = document["datasets"]["sample"]["metadata"]
    metadata["provenance_ids"] = [replacement if i == identifier else i for i in metadata["provenance_ids"]]
    document["build_id"] = sha256(canonical_json({k: v for k, v in document.items() if k != "build_id"}).encode()).hexdigest()
    with pytest.raises(ContractError, match="section, or event kind"):
        DatasetCatalog(document)
