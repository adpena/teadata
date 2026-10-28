from dataclasses import replace
import gzip
import json
import pickle
import sqlite3

import pytest

from teadata import DataEngine, ProvenanceStore
from teadata.manifest import (
    BuildRecorder,
    ContractError,
    DatasetCatalog,
    write_sqlite_catalog,
)
from teadata.release import DataRelease, publish_release, SNAPSHOT_MAGIC
from teadata.teadata_config import Config


@pytest.fixture
def catalog(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("count\n42\n")
    cfg = Config.from_dict(
        {
            "value_policies": {"numeric": {}},
            "data_sources": {
                "sample": {
                    2025: {
                        "path": str(source),
                        "metadata": {
                            "value_policy": "numeric",
                            "period": {
                                "kind": "school_year",
                                "start_year": 2024,
                                "end_year": 2025,
                            },
                        },
                    }
                }
            },
        }
    )
    store = ProvenanceStore(tmp_path / "provenance")
    resolved = cfg.resolve_dataset("sample")
    receipt = store.record_acquisition(source, resolved)
    cfg.data_sources["sample"].entries[2025] = replace(
        cfg.data_sources["sample"].entries[2025],
        metadata=replace(resolved.metadata, provenance_ids=(receipt,)),
    )
    with BuildRecorder(cfg, store) as build:
        cfg.load_df("sample")
        return build.catalog()


@pytest.fixture
def release(tmp_path, catalog):
    staging = tmp_path / "staging"
    staging.mkdir()
    paths = {}
    for role in ("entities", "maps", "boundaries"):
        path = staging / (role + ".sqlite")
        with sqlite3.connect(path) as connection:
            write_sqlite_catalog(connection, catalog)
        paths[role] = path
    engine = DataEngine()
    engine.catalog = catalog
    paths["snapshot"] = staging / "snapshot.pkl.gz"
    with gzip.open(paths["snapshot"], "wb") as stream:
        stream.write(SNAPSHOT_MAGIC + catalog.build_id.encode() + b"\n")
        pickle.dump(engine, stream)
    store = ProvenanceStore(tmp_path / "provenance")
    record_id = store.record_processing(
        dataset="test_release",
        inputs=catalog.metadata("sample").provenance_ids,
        outputs=paths,
        processor="fixture",
        code_version="test",
        parameters={"build_id": catalog.build_id},
    )
    return publish_release(
        tmp_path / "published",
        catalog,
        paths,
        processing={"record_id": record_id, "record": store.read_record(record_id)},
    )


def test_release_binds_all_artifacts(release, catalog):
    selected = DataRelease(release)
    assert selected.build_id == catalog.build_id
    assert set(selected.verify()) == {"entities", "maps", "boundaries", "snapshot"}
    engine = DataEngine.from_snapshot(selected.artifact("snapshot"))
    assert engine.catalog.build_id == selected.build_id


def test_tampered_artifact_is_rejected_even_after_a_successful_read(release):
    selected = DataRelease(release)
    path = selected.artifact("entities")
    with path.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ContractError, match="checksum or size"):
        selected.artifact("entities")


def test_catalog_tampering_is_rejected(catalog):
    document = catalog.to_dict()
    document["datasets"]["sample"]["metadata"]["period"]["end_year"] = 2026
    with pytest.raises(ContractError, match="identity mismatch"):
        DatasetCatalog(document)


def test_catalog_is_not_mutable_through_public_accessors(catalog):
    document = catalog.datasets
    document["sample"]["metadata"]["period"]["end_year"] = 2030
    assert catalog.metadata("sample").period.end_year == 2025


def test_path_traversal_is_rejected(release):
    document = json.loads(release.read_text())
    document["artifacts"]["entities"]["filename"] = "../secret.sqlite"
    release.write_text(json.dumps(document))
    with pytest.raises(ContractError, match="portable filenames"):
        DataRelease(release)


def test_failed_publication_keeps_previous_release(release, catalog, tmp_path):
    before = release.read_bytes()
    with pytest.raises(ContractError, match="incomplete"):
        publish_release(
            release.parent, catalog, {"entities": tmp_path / "missing"}, processing={}
        )
    assert release.read_bytes() == before


def test_observing_migrated_source_does_not_invent_download_date(catalog, tmp_path):
    record = next((tmp_path / "provenance" / "records").glob("*.json"))
    assert json.loads(record.read_text())["downloaded_at"] is None
    read = catalog.datasets["sample"]["reads"][0]
    assert read["rows"] == 1 and read["columns"] == ["count"]


def test_legacy_snapshots_require_rebuild(tmp_path):
    path = tmp_path / "legacy.pkl"
    path.write_bytes(pickle.dumps(({}, DataEngine())))
    with pytest.raises(ContractError, match="rebuild"):
        DataEngine.from_snapshot(path)


def test_failed_checksum_cannot_be_hidden_by_restoring_mtime(release):
    import os

    selected = DataRelease(release)
    path = selected.artifact("snapshot")
    stat = path.stat()
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ContractError, match="checksum"):
        selected.artifact("snapshot")


def test_policy_mutation_does_not_change_catalog(catalog):
    policy = catalog.policy("sample")
    policy.tokens["unexpected"] = None
    assert "unexpected" not in catalog.policy("sample").tokens


def test_schema_drift_is_rejected_before_publication(tmp_path):
    from teadata.manifest import schema_fingerprint

    source = tmp_path / "changed.csv"
    source.write_text("renamed_column\n1\n")
    cfg = Config.from_dict(
        {
            "data_sources": {
                "example": {
                    2025: {
                        "path": str(source),
                        "reader": {
                            "expected_columns_sha256": schema_fingerprint(
                                ["expected_column"]
                            )
                        },
                    }
                }
            }
        }
    )
    with pytest.raises(ContractError, match="schema changed"):
        cfg.load_df("example")


def test_unreviewed_masking_methodology_blocks_a_build(tmp_path):
    cfg = Config.from_dict(
        {
            "value_policies": {
                "masked": {
                    "reference_id": "masking",
                    "tokens": {"-1": {"status": "suppressed", "reason": "protected"}},
                }
            },
            "data_sources": {
                "example": {
                    2025: {
                        "path": str(tmp_path / "source.csv"),
                        "metadata": {
                            "value_policy": "masked",
                            "references": [
                                {
                                    "reference_id": "masking",
                                    "role": "masking_methodology",
                                    "status": "unreviewed",
                                }
                            ],
                        },
                    }
                }
            },
        }
    )
    with BuildRecorder(cfg, ProvenanceStore(tmp_path / "provenance")):
        with pytest.raises(ContractError, match="reviewed methodology"):
            cfg.load_df("example")
