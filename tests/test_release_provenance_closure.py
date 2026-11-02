"""Complete selected record graphs; fixture observations never claim HTTP downloads."""

from dataclasses import replace
from copy import deepcopy
from types import SimpleNamespace

import pytest

from teadata.dataset_metadata import DatasetMetadata, ReferenceDocument, ResolvedDataset
from teadata.manifest import BuildRecorder
from teadata.provenance import ProvenanceStore
from teadata.teadata_config import Config
from teadata.verification import ReceiptError, _validate_release, _validate_release_provenance


@pytest.fixture
def nested_release(tmp_path):
    store = ProvenanceStore(tmp_path / "history")

    def reference(name, text, references=()):
        path = tmp_path / (name + ".txt")
        path.write_text(text)
        url = "https://example.invalid/" + name
        source = ResolvedDataset("reference:" + name, "reference", None, None, str(path),
                                 DatasetMetadata(source_url=url, references=references))
        identifier = store.record_acquisition(path, source)
        return identifier, ReferenceDocument.from_dict({
            "reference_id": name, "role": "data_dictionary", "status": "unreviewed",
            "url": url, "provenance_ids": [identifier],
        })

    ancestor, older = reference("older", "Original publisher definitions")
    intermediate, current = reference("current", "Current definitions", (older,))
    unrelated, _ = reference("unrelated", "Outside the selected source graph")
    path = tmp_path / "source.csv"
    path.write_text("value\n0\n-999\n")
    cfg = Config.from_dict({"data_sources": {"sample": {2025: {
        "path": str(path), "metadata": DatasetMetadata(references=(current,)).to_dict(),
    }}}})
    source = cfg.resolve_dataset("sample", 2025)
    prior = store.record_acquisition(path, source)
    changed = replace(source, metadata=replace(source.metadata, notes="Current reviewed source facts",
                                               provenance_ids=(prior,)))
    selected = store.record_acquisition(path, changed)
    cfg.data_sources["sample"].entries[2025] = replace(
        cfg.data_sources["sample"].entries[2025], metadata=replace(changed.metadata, provenance_ids=(selected,)),
    )
    with BuildRecorder(cfg, store) as build:
        _, frame = cfg.load_df("sample", 2025)
        catalog = build.catalog()
    processing = store.record_processing(
        dataset="test_release", inputs=[selected], outputs={"synthetic": path},
        processor="test", code_version="test", parameters={"build_id": catalog.build_id},
    )
    release = SimpleNamespace(catalog=catalog, processing={"record_id": processing, "record": store.read_record(processing)})
    return release, store, {ancestor, intermediate, prior, selected}, ancestor, unrelated, frame


def test_builder_embeds_selected_nested_and_prior_observation_ancestors(nested_release):
    release, store, identifiers, _, unrelated, frame = nested_release
    records = release.catalog.to_dict()["provenance"]
    assert set(records) == identifiers and unrelated not in records
    assert frame["value"].tolist() == [0, -999]
    for identifier in identifiers:
        assert records[identifier] == store.read_record(identifier)
        assert records[identifier]["downloaded_at"] is None
    assert release.catalog.metadata("sample").notes == "Current reviewed source facts"
    assert _validate_release_provenance(release) == len(identifiers) + 1


@pytest.mark.parametrize("damage", ["removed", "transplanted"])
def test_qualification_rejects_missing_or_foreign_ancestor_before_artifact_reads(nested_release, monkeypatch, damage):
    release, store, _, ancestor, unrelated, _ = nested_release
    document = release.catalog.to_dict()
    current_facts = deepcopy(document["datasets"]["sample"])
    if damage == "removed":
        del document["provenance"][ancestor]
    else:
        document["provenance"][ancestor] = store.read_record(unrelated)
    release.catalog = SimpleNamespace(to_dict=lambda: document)
    release.verify = lambda: pytest.fail("Invalid record graph reached release artifact bytes")
    monkeypatch.setattr("teadata.release.DataRelease", lambda _: release)
    with pytest.raises(ReceiptError, match="provenance record graph is incomplete or invalid"):
        _validate_release("unused-metadata-fixture", "library")
    assert document["datasets"]["sample"] == current_facts
