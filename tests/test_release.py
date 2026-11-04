from dataclasses import replace
from contextlib import closing
import errno
import gzip
from hashlib import sha256
from io import BytesIO
import json
import os
import pickle
import sqlite3
from unittest.mock import Mock
from uuid import UUID

import pytest
from shapely.geometry import Point

from teadata import DataEngine, ProvenanceStore, citation
from teadata import boundary_store, entity_store, map_store
from teadata.entities import Campus, District
import teadata.release as release_module
from teadata.manifest import (
    BuildRecorder,
    ContractError,
    DatasetCatalog,
    canonical_json,
    file_identity,
    read_sqlite_catalog,
    write_sqlite_catalog,
)
from teadata.release import DataRelease, publish_release, SNAPSHOT_MAGIC
from teadata.persistence.sqlalchemy_store import create_engine, create_sessionmaker, ensure_schema, export_dataengine
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
    engine = DataEngine()
    engine.catalog = catalog
    district = District(id=UUID(int=1), name="MAIN DISTRICT", district_number="123456")
    district.meta = {"district_main": 1}
    engine.add_district(district)
    campus = Campus(id=UUID(int=2), district_id=district.id, name="MAIN CAMPUS", charter_type="", is_charter=False)
    campus.campus_number, campus.district_number = "123456789", "123456"
    campus.meta = {"campus_main": 1}
    engine.add_campus(campus)
    paths = {}
    for role in ("entities", "maps", "boundaries"):
        path = staging / (role + ".sqlite")
        with closing(sqlite3.connect(path)) as connection, connection:
            write_sqlite_catalog(connection, catalog)
            if role == "maps":
                connection.execute("CREATE TABLE campus_profile(lookup_key TEXT PRIMARY KEY,payload BLOB)")
                connection.execute("CREATE TABLE closures_map(district_number TEXT PRIMARY KEY,payload_base BLOB,payload_transfers BLOB)")
                connection.execute("INSERT INTO campus_profile VALUES('123456789','{\"campus\":1}')")
                connection.execute("INSERT INTO closures_map VALUES('123456','{\"district\":1}','{\"transfers\":1}')")
            elif role == "boundaries":
                connection.execute("CREATE TABLE boundaries(district_number TEXT PRIMARY KEY,wkb BLOB)")
                connection.execute("INSERT INTO boundaries VALUES (?,?)", ("'123456", Point(1, 1).wkb))
        if role == "entities":
            database = create_engine(f"sqlite:///{path}")
            try:
                ensure_schema(database)
                Session = create_sessionmaker(database)
                with Session.begin() as session:
                    export_dataengine(engine, session, replace=True)
            finally:
                database.dispose()
        paths[role] = path
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
    with pytest.raises(ContractError, match="no analytical read model"):
        selected.analytical_dataset("campus")


@pytest.mark.parametrize("unsupported", [False, True])
def test_file_identity_cache_advice_preserves_complete_hash(tmp_path, monkeypatch, unsupported):
    window = 16 * 1024 * 1024
    path = tmp_path / "artifact"
    with path.open("wb") as stream:
        stream.seek(window + 19)
        stream.write(b"x")
    calls = []

    def advise(fd, offset, length, policy):
        assert os.fstat(fd).st_size == window + 20
        calls.append((offset, length, policy))
        if unsupported:
            raise OSError(errno.EINVAL, "cache advice unsupported")

    monkeypatch.setattr(os, "posix_fadvise", advise, raising=False)
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
    expected = file_identity(path)
    assert calls == []
    assert file_identity(path, discard_cached_reads=True) == expected
    assert calls == ([(0, window, 4)] if unsupported else [(0, window, 4), (window, 20, 4)])


def test_streaming_release_verification_still_rejects_replaced_bytes(release, monkeypatch):
    identify = Mock(wraps=file_identity)
    monkeypatch.setattr(release_module, "file_identity", identify)
    selected = DataRelease(release)
    verified = selected.verify(discard_cached_reads=True)
    assert {call.args[0] for call in identify.call_args_list} == set(verified.values())
    assert all(call.kwargs == {"discard_cached_reads": True} for call in identify.call_args_list)

    # Same file size, so the checksum and the verified file identity both matter.
    path = verified["maps"]
    with path.open("r+b") as stream:
        original = stream.read(1)
        stream.seek(0)
        stream.write(bytes([original[0] ^ 1]))
    with pytest.raises(ContractError, match="checksum or size"):
        selected.verify(discard_cached_reads=True)


def test_current_release_reuses_a_pinned_build_cache_offline(release, tmp_path, monkeypatch):
    monkeypatch.delenv("TEADATA_RELEASE", raising=False)
    monkeypatch.setenv("TEADATA_RELEASE_URL", "https://example.invalid/v1/release.json")
    monkeypatch.setenv("TEADATA_RELEASE_SHA256", sha256(release.read_bytes()).hexdigest())
    monkeypatch.setenv("TEADATA_RELEASE_OFFLINE", "0")
    monkeypatch.setattr(release_module, "user_cache_path", lambda _: tmp_path / "cache")

    def response(url, *, timeout):
        filename = url.removeprefix("https://example.invalid/v1/")
        return BytesIO((release.parent / filename).read_bytes())

    monkeypatch.setattr(release_module, "urlopen", response)
    try:
        build = release_module.current_release()
        assert build.manifest_sha256 == sha256(release.read_bytes()).hexdigest()
        staged = build.verify()
        release_module._load_release.cache_clear()
        monkeypatch.setenv("TEADATA_RELEASE_OFFLINE", "1")
        network = Mock(side_effect=AssertionError("Offline startup must not use the network"))
        monkeypatch.setattr(release_module, "urlopen", network)
        runtime = release_module.current_release()
        assert runtime.offline and runtime.build_id == build.build_id
        assert runtime.manifest_sha256 == build.manifest_sha256
        assert runtime.verify() == staged

        monkeypatch.setenv("TEADATA_RELEASE_SHA256", "f" * 64)
        with pytest.raises(ContractError, match="trusted pin"):
            release_module.current_release()
        monkeypatch.delenv("TEADATA_RELEASE_SHA256")
        with pytest.raises(ContractError, match="expected manifest SHA-256"):
            release_module.current_release()
        monkeypatch.setenv("TEADATA_RELEASE_OFFLINE", "true")
        with pytest.raises(ContractError, match="must be 0 or 1"):
            release_module.current_release()
        with release_module.using_release(runtime):
            assert release_module.current_release() is runtime
        network.assert_not_called()
    finally:
        release_module._load_release.cache_clear()


def test_current_release_applies_pin_to_the_selected_local_manifest(release, monkeypatch):
    monkeypatch.setenv("TEADATA_RELEASE", str(release))
    monkeypatch.setenv("TEADATA_RELEASE_URL", "https://example.invalid/release.json")
    original = release.read_bytes()
    pin = sha256(original).hexdigest()
    monkeypatch.setenv("TEADATA_RELEASE_SHA256", pin)
    monkeypatch.setenv("TEADATA_RELEASE_OFFLINE", "1")
    selected = release_module.current_release()
    assert selected.path == release
    release.write_bytes(original + b" ")
    assert selected.manifest_sha256 == pin
    with pytest.raises(ContractError, match="trusted pin"):
        DataRelease(release, expected_sha256=selected.manifest_sha256, offline=True)
    monkeypatch.setenv("TEADATA_RELEASE_SHA256", "")
    with pytest.raises(ContractError, match="Manifest pin must be a SHA-256 identity"):
        release_module.current_release()


def test_release_offline_reads_policy_without_selecting_a_manifest(release, tmp_path, monkeypatch):
    monkeypatch.setenv("TEADATA_RELEASE", str(tmp_path / "missing.json"))
    monkeypatch.delenv("TEADATA_RELEASE_OFFLINE", raising=False)
    assert release_module.release_offline() is False
    monkeypatch.setenv("TEADATA_RELEASE_OFFLINE", "1")
    assert release_module.release_offline() is True
    with release_module.using_release(DataRelease(release)):
        assert release_module.release_offline() is False
    monkeypatch.setenv("TEADATA_RELEASE_OFFLINE", "true")
    with pytest.raises(ContractError, match="must be 0 or 1"):
        release_module.release_offline()


def test_citation_pins_release_without_emitting_private_locations(release, tmp_path, monkeypatch):
    monkeypatch.setattr(release_module, "user_cache_path", lambda _: tmp_path / "cache")

    def copy_manifest(_url, target, **_options):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(release.read_bytes())

    monkeypatch.setattr(release_module, "_download", copy_manifest)
    local = DataRelease(release)
    official_url = f"{citation().repository_url}/releases/download/v0.0.127/release.json"
    official = DataRelease(official_url)
    signed = DataRelease(f"{official_url}?token=private")
    monkeypatch.setattr(DataRelease, "artifact", Mock(side_effect=AssertionError("implicit artifact access")))

    for selected in (local, signed, official):
        cite = citation(selected)
        assert cite.build_id == local.build_id
        assert f"Build ID: {local.build_id}" in cite.text and local.build_id in cite.bibtex
        assert str(release) not in cite.text + cite.bibtex
        assert "token=private" not in cite.text + cite.bibtex
        assert cite.manifest_url == (official_url if selected is official else None)
    assert official_url in citation(official).text and official_url in citation(official).bibtex


def test_tampered_artifact_is_rejected_even_after_a_successful_read(release):
    selected = DataRelease(release)
    path = selected.artifact("entities")
    with path.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ContractError, match="checksum or size"):
        selected.artifact("entities")


@pytest.mark.parametrize("reader", ["discover", "campus", "district"])
def test_default_maps_reuse_release_catalog_validation_on_cold_and_warm_reads(release, monkeypatch, reader):
    selected = DataRelease(release)
    validate = Mock(wraps=release_module.read_sqlite_catalog)
    monkeypatch.setattr(release_module, "read_sqlite_catalog", validate)
    monkeypatch.setattr(map_store, "read_sqlite_catalog", Mock(side_effect=AssertionError("Duplicate catalog construction")))
    with release_module.using_release(selected):
        for _ in range(2):
            if reader == "discover":
                assert map_store.discover_map_store().name == selected._artifacts["maps"]["filename"]
            elif reader == "campus":
                assert map_store.load_campus_profile_payload("123456789") == {"campus": 1}
            else:
                assert map_store.load_map_payload_parts("123456") == ({"district": 1}, {"transfers": 1})
    validate.assert_called_once_with(selected.artifact("maps"), immutable=True)


@pytest.mark.parametrize("fault", ["missing_catalog", "malformed_catalog", "build_projection", "legacy_identity"])
def test_explicit_maps_still_validate_catalog_and_identity_schema(release, monkeypatch, fault):
    path = DataRelease(release).artifact("maps")
    monkeypatch.setattr(map_store, "current_release", Mock(side_effect=AssertionError("Explicit path must not select a release")))
    assert map_store.discover_map_store(path) == path
    with closing(sqlite3.connect(path)) as connection:
        if fault == "missing_catalog":
            connection.execute("DROP TABLE teadata_catalog")
        elif fault == "malformed_catalog":
            connection.execute("UPDATE teadata_catalog SET document='{}'")
        elif fault == "build_projection":
            connection.execute("UPDATE teadata_catalog SET build_id=?", ("0" * 64,))
        else:
            connection.execute("ALTER TABLE campus_profile RENAME COLUMN lookup_key TO campus_number")
        connection.commit()
    with pytest.raises(ContractError):
        map_store.discover_map_store(path)


def test_empty_explicit_map_path_does_not_fall_back_to_release(monkeypatch):
    monkeypatch.setattr(map_store, "current_release", Mock(side_effect=AssertionError("Empty path is explicit")))
    with pytest.raises(ContractError, match="required dataset catalog"):
        map_store.discover_map_store("")


@pytest.mark.parametrize("change", ["write", "replacement"])
def test_default_map_discovery_rechecks_changed_artifact_after_warm_read(release, change, tmp_path):
    selected = DataRelease(release)
    with release_module.using_release(selected):
        path = map_store.discover_map_store()
        if change == "write":
            with path.open("ab") as stream:
                stream.write(b"tampered")
        else:
            replacement = tmp_path / "replacement.sqlite"
            replacement.write_bytes(path.read_bytes())
            with closing(sqlite3.connect(replacement)) as connection:
                connection.execute("UPDATE campus_profile SET payload='{\"campus\":2}'")
                connection.commit()
            os.replace(replacement, path)
        with pytest.raises(ContractError, match="checksum or size"):
            map_store.discover_map_store()


def _bind_artifact_bytes(release, path, role="maps"):
    """Keep the tiny fixture's declared hashes/receipt consistent for build tests."""
    document = json.loads(release.read_text())
    identity = file_identity(path)
    for key in ("sha256", "size_bytes"):
        document["artifacts"][role][key] = identity[key]
        document["processing"]["record"]["outputs"][role][key] = identity[key]
    document["processing"]["record_id"] = sha256(
        (canonical_json(document["processing"]["record"]) + "\n").encode()
    ).hexdigest()
    release.write_text(canonical_json(document))


def test_default_maps_still_reject_legacy_identity_after_release_verification(release):
    path = DataRelease(release).artifact("maps")
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("ALTER TABLE campus_profile RENAME COLUMN lookup_key TO campus_number")
        connection.commit()
    _bind_artifact_bytes(release, path)
    with release_module.using_release(DataRelease(release)):
        with pytest.raises(ContractError, match="Legacy map campus identity"):
            map_store.discover_map_store()


def _foreign_catalog(catalog):
    document = catalog.to_dict()
    return DatasetCatalog.create(datasets=document["datasets"], value_policies=document["value_policies"],
        processor="f" * 64, records=document["provenance"], environment=document["build_environment"],
        metric_bindings=document["metric_bindings"])


def test_explicit_foreign_catalog_is_independent_but_default_rejects_manifest_mismatch(release, catalog):
    foreign = _foreign_catalog(catalog)
    path = DataRelease(release).artifact("maps")
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("UPDATE teadata_catalog SET build_id=?,document=?",
            (foreign.build_id, canonical_json(foreign.to_dict())))
        connection.commit()
    assert map_store.discover_map_store(path) == path
    _bind_artifact_bytes(release, path)
    with release_module.using_release(DataRelease(release)):
        with pytest.raises(ContractError, match="different data build"):
            map_store.discover_map_store()


@pytest.mark.parametrize("change", ["payload", "catalog", "identity_schema"])
def test_default_map_readers_use_verified_main_file_while_explicit_paths_follow_wal(release, change):
    path = DataRelease(release).artifact("maps")
    with closing(sqlite3.connect(path)) as writer:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        _bind_artifact_bytes(release, path)
        before = file_identity(path)
        if change == "payload":
            writer.execute("UPDATE campus_profile SET payload='{\"campus\":2}'")
            writer.execute("UPDATE closures_map SET payload_base='{\"district\":2}',payload_transfers='{\"transfers\":2}'")
        elif change == "catalog":
            writer.execute("UPDATE teadata_catalog SET build_id=?", ("0" * 64,))
        else:
            writer.execute("ALTER TABLE campus_profile RENAME COLUMN lookup_key TO campus_number")
        writer.commit()
        assert file_identity(path) == before and path.with_name(path.name + "-wal").stat().st_size > 0
        with release_module.using_release(DataRelease(release)):
            assert map_store.load_campus_profile_payload("123456789") == {"campus": 1}
            assert map_store.load_map_payload_parts("123456") == ({"district": 1}, {"transfers": 1})
            assert read_sqlite_catalog(path, immutable=True).build_id == DataRelease(release).build_id
        if change == "payload":
            assert map_store.load_campus_profile_payload("123456789", store_path=path) == {"campus": 2}
            assert map_store.load_map_payload_parts("123456", store_path=path) == ({"district": 2}, {"transfers": 2})
        else:
            with pytest.raises(ContractError):
                map_store.discover_map_store(path)


@pytest.mark.parametrize("reader", ["campus", "district", "meta_keys"])
@pytest.mark.parametrize("change", ["payload", "catalog", "schema"])
def test_default_entity_readers_ignore_wal_while_explicit_and_required_paths_remain_mutable(release, catalog, reader, change):
    path = DataRelease(release).artifact("entities")
    foreign = _foreign_catalog(catalog)

    def read(explicit=None):
        if reader == "campus":
            return entity_store.load_campus("123456789", store_path=explicit)
        if reader == "district":
            return entity_store.load_district("123456", store_path=explicit)
        return entity_store.list_meta_keys("campus", store_path=explicit)

    with closing(sqlite3.connect(path)) as writer:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        _bind_artifact_bytes(release, path, "entities")
        before = file_identity(path)
        if change == "catalog":
            writer.execute("UPDATE teadata_catalog SET build_id=?,document=?",
                (foreign.build_id, canonical_json(foreign.to_dict())))
        if change == "schema":
            if reader == "campus":
                writer.execute("ALTER TABLE campuses RENAME COLUMN lookup_key TO wal_key")
            elif reader == "district":
                writer.execute("ALTER TABLE districts RENAME COLUMN name TO wal_name")
            else:
                writer.execute("DROP TABLE campus_meta")
        elif reader == "meta_keys":
            writer.execute("UPDATE campus_meta SET key='wal_only' WHERE key='campus_main'")
        else:
            table = "campuses" if reader == "campus" else "districts"
            writer.execute(f"UPDATE {table} SET name='WAL value'")
        writer.commit()
        assert file_identity(path) == before and path.with_name(path.name + "-wal").stat().st_size > 0
        with release_module.using_release(DataRelease(release)):
            result = read()
            if reader == "meta_keys":
                assert "campus_main" in result and "wal_only" not in result
            else:
                assert result["name"] == ("MAIN CAMPUS" if reader == "campus" else "MAIN DISTRICT")
                if reader == "campus":
                    assert result["source_build_id"] == catalog.build_id
        if change == "schema":
            if reader == "meta_keys":
                assert read(path) == []  # Existing explicit missing-table policy.
            else:
                with pytest.raises(ContractError if reader == "campus" else RuntimeError):
                    read(path)
        else:
            result = read(path)
            if reader == "meta_keys":
                assert "wal_only" in result and "campus_main" not in result
            else:
                assert result["name"] == "WAL value"
                if reader == "campus" and change == "catalog":
                    assert result["source_build_id"] == foreign.build_id
            with entity_store.EntityStore(path) as required_path_store:
                assert required_path_store.catalog.build_id == (foreign.build_id if change == "catalog" else catalog.build_id)


@pytest.mark.parametrize("change", ["payload", "catalog", "schema"])
def test_default_boundaries_ignore_wal_while_explicit_and_required_paths_remain_mutable(release, catalog, change):
    path = DataRelease(release).artifact("boundaries")
    foreign = _foreign_catalog(catalog)
    with closing(sqlite3.connect(path)) as writer:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        _bind_artifact_bytes(release, path, "boundaries")
        before = file_identity(path)
        if change == "catalog":
            writer.execute("UPDATE teadata_catalog SET build_id=?,document=?",
                (foreign.build_id, canonical_json(foreign.to_dict())))
        if change == "schema":
            writer.execute("DROP TABLE boundaries")
        else:
            writer.execute("UPDATE boundaries SET wkb=?", (Point(2, 2).wkb,))
        writer.commit()
        assert file_identity(path) == before and path.with_name(path.name + "-wal").stat().st_size > 0
        with release_module.using_release(DataRelease(release)):
            assert boundary_store.load_boundary("123456").equals(Point(1, 1))
        if change == "schema":
            assert boundary_store.load_boundary("123456", store_path=path) is None
        else:
            assert boundary_store.load_boundary("123456", store_path=path).equals(Point(2, 2))
            required_path_store = boundary_store.BoundaryStore(path)
            try:
                assert required_path_store.get("123456").equals(Point(2, 2))
            finally:
                required_path_store.close()
            assert read_sqlite_catalog(path).build_id == (foreign.build_id if change == "catalog" else catalog.build_id)


def test_publication_rejects_catalog_matching_only_in_wal_before_linking_main_file(release, catalog, tmp_path):
    selected = DataRelease(release)
    foreign = _foreign_catalog(catalog)
    source = tmp_path / "wal-only-catalog.sqlite"
    with closing(sqlite3.connect(source)) as writer:
        write_sqlite_catalog(writer, foreign)
        writer.commit()
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        writer.execute("UPDATE teadata_catalog SET build_id=?,document=?",
            (catalog.build_id, canonical_json(catalog.to_dict())))
        writer.commit()
        assert read_sqlite_catalog(source).build_id == catalog.build_id
        assert read_sqlite_catalog(source, immutable=True).build_id == foreign.build_id
        artifacts = {"maps": source, **{role: selected.artifact(role) for role in ("entities", "boundaries", "snapshot")}}
        destination = tmp_path / "must-not-publish"
        with pytest.raises(ContractError, match="Cannot publish mixed data builds: maps"):
            publish_release(destination, catalog, artifacts, processing=selected.processing)
        assert list(destination.iterdir()) == []


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
    from hashlib import sha256

    source = tmp_path / "changed.csv"
    source.write_text("renamed_column\n1\n")
    schema = tmp_path / "columns.json"
    schema.write_text(json.dumps({"schema_version": 1, "columns": ["expected_column"]}))
    cfg = Config.from_dict(
        {
            "data_sources": {
                "example": {
                    2025: {
                        "path": str(source),
                        "metadata": {
                            "references": [
                                {
                                    "reference_id": "columns",
                                    "role": "schema",
                                    "path": "columns.json",
                                    "status": "available",
                                    "reviewed_sha256": sha256(schema.read_bytes()).hexdigest(),
                                }
                            ],
                        },
                        "reader": {
                            "column_schema": "columns.json",
                        },
                    }
                }
            }
        }
    )
    cfg.base_dir = tmp_path
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
