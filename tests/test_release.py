from dataclasses import replace
from contextlib import closing
from copy import deepcopy
import errno
import gzip
from hashlib import sha256
from io import BytesIO
import json
import os
import pickle
import signal
import sqlite3
import sys
from threading import Barrier, Event, Thread
from types import SimpleNamespace
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


def _threaded_artifacts(selected, roles):
    start = Barrier(len(roles) + 1)
    results, errors = [], []

    def read(role):
        try:
            start.wait(timeout=5)
            results.append(selected.artifact(role))
        except BaseException as error:
            errors.append(error)

    threads = [Thread(target=read, args=(role,), daemon=True) for role in roles]
    for thread in threads:
        thread.start()
    start.wait(timeout=5)
    return threads, results, errors


def _finish_artifact_threads(threads):
    for thread in threads:
        thread.join(timeout=5)
    assert not any(thread.is_alive() for thread in threads), "Artifact verification stranded a caller"


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
        explicit = DataRelease(runtime.source, expected_sha256=runtime.manifest_sha256, offline=True,
                               cache_dir=runtime.cache)
        files = explicit.verified_cache_inventory()
        assert files[explicit.path.relative_to(explicit.cache).as_posix()]["sha256"] == runtime.manifest_sha256
        assert len(files) == len({path for path in staged.values()}) + 1

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


def test_offline_manifest_rejects_fifo_and_symlink_ancestors(tmp_path):
    url = "https://example.invalid/release.json"
    name = sha256(url.encode()).hexdigest() + ".json"
    cache = tmp_path / "cache"
    slot = cache / "manifests"
    slot.mkdir(parents=True)
    os.mkfifo(slot / name)
    with pytest.raises(ContractError, match="regular files"):
        DataRelease(url, expected_sha256="0" * 64, offline=True, cache_dir=cache)
    (slot / name).unlink()
    slot.rmdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / name).write_text("{}")
    slot.symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ContractError, match="without symlink paths"):
        DataRelease(url, expected_sha256="0" * 64, offline=True, cache_dir=cache)


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


@pytest.mark.parametrize("atomic", [False, True])
def test_current_release_detects_local_replacement_with_preserved_mtime(release, monkeypatch, atomic):
    original = release.read_bytes().rstrip() + b"\n"
    revised = original[:-1] + b" "
    release.write_bytes(original)
    monkeypatch.setenv("TEADATA_RELEASE", str(release))
    monkeypatch.setenv("TEADATA_RELEASE_SHA256", sha256(original).hexdigest())
    selected = release_module.current_release()
    cache_uses_change_time = os.name != "nt" or sys.version_info >= (3, 12)
    if cache_uses_change_time:
        assert release_module.current_release() is selected

    before = release.stat()
    before_stamp = release_module._local_manifest_stamp(release)
    target = release.with_suffix(".replacement") if atomic else release
    target.write_bytes(revised)
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    if atomic:
        target.replace(release)
    after = release.stat()
    assert after.st_mtime_ns == before.st_mtime_ns
    if not atomic and cache_uses_change_time and release_module._local_manifest_stamp(release) == before_stamp:
        pytest.skip("Filesystem did not expose the in-place change in stat metadata")

    with release_module.using_release(selected):
        assert release_module.current_release() is selected
    with pytest.raises(ContractError, match="trusted pin"):
        release_module.current_release()
    monkeypatch.setenv("TEADATA_RELEASE_SHA256", sha256(revised).hexdigest())
    assert release_module.current_release().manifest_sha256 == sha256(revised).hexdigest()


def test_current_release_tracks_alias_target_even_with_a_shared_inode(release, tmp_path, monkeypatch):
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other = other_dir / release.name
    other.hardlink_to(release)
    alias = tmp_path / "active.json"
    try:
        alias.symlink_to(release)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows account cannot create symbolic links")
        raise
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEADATA_RELEASE", alias.name)
    monkeypatch.delenv("TEADATA_RELEASE_SHA256", raising=False)

    selected = release_module.current_release()
    assert selected.source == alias.name
    assert selected.path == release
    alias.unlink()
    alias.symlink_to(other)
    assert release_module.current_release().path == other


@pytest.mark.skipif(os.name == "nt" and sys.version_info < (3, 12), reason="Local manifests bypass the cache")
def test_current_release_rejects_replacement_between_stat_and_open(release, monkeypatch):
    original = release.read_bytes()
    replacement = release.with_suffix(".replacement")
    replacement.write_bytes(original + b" ")
    monkeypatch.setenv("TEADATA_RELEASE", str(release))
    monkeypatch.delenv("TEADATA_RELEASE_SHA256", raising=False)
    load_release = release_module._load_release

    def swap_before_load(*args, **kwargs):
        if replacement.exists():
            replacement.replace(release)
        return load_release(*args, **kwargs)

    monkeypatch.setattr(release_module, "_load_release", swap_before_load)
    with pytest.raises(ContractError, match="Local release manifest changed"):
        release_module.current_release()
    assert release_module.current_release().manifest_sha256 == sha256(original + b" ").hexdigest()


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


def test_concurrent_same_release_verifies_one_artifact_once(release, monkeypatch):
    selected = DataRelease(release)
    entered, finish, duplicate = Event(), Event(), Event()
    calls = []

    def identify(path, **options):
        calls.append(path)
        if len(calls) > 1:
            duplicate.set()
        entered.set()
        assert finish.wait(5)
        return file_identity(path, **options)

    semantic = Mock(wraps=release_module.read_sqlite_catalog)
    monkeypatch.setattr(release_module, "file_identity", identify)
    monkeypatch.setattr(release_module, "read_sqlite_catalog", semantic)
    threads, results, errors = _threaded_artifacts(selected, ("maps", "maps"))
    try:
        assert entered.wait(5)
        assert not duplicate.wait(.05)
    finally:
        finish.set()
        _finish_artifact_threads(threads)
    assert not errors and results[0] == results[1]
    assert len(calls) == semantic.call_count == 1
    assert selected.artifact("maps") == results[0]
    assert len(calls) == semantic.call_count == 1


def test_different_artifacts_do_not_share_a_verification_gate(release, monkeypatch):
    selected = DataRelease(release)
    both_hashing = Barrier(2)

    def identify(path, **options):
        both_hashing.wait(timeout=5)
        return file_identity(path, **options)

    monkeypatch.setattr(release_module, "file_identity", identify)
    threads, results, errors = _threaded_artifacts(selected, ("maps", "entities"))
    _finish_artifact_threads(threads)
    assert not errors and len(set(results)) == 2


def test_waiting_caller_restats_changed_file_after_acquiring_gate(release, monkeypatch):
    selected = DataRelease(release)
    path = selected.artifact("maps")
    original_lock = selected._artifact_lock
    gate = original_lock(selected._artifacts["maps"]["sha256"])
    requested = Event()

    def request(digest):
        requested.set()
        return original_lock(digest)

    monkeypatch.setattr(selected, "_artifact_lock", request)
    with gate:
        threads, results, errors = _threaded_artifacts(selected, ("maps",))
        assert requested.wait(5)
        with path.open("ab") as stream:
            stream.write(b"changed while waiting")
    _finish_artifact_threads(threads)
    assert not results and len(errors) == 1
    assert isinstance(errors[0], ContractError) and "checksum or size" in str(errors[0])


@pytest.mark.parametrize("phase", ["hash", "semantic"])
def test_artifact_changed_during_verification_is_not_admitted(release, monkeypatch, phase):
    selected = DataRelease(release)

    def changed_hash(path, **options):
        result = file_identity(path, **options)
        path.touch()
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000_000))
        return result

    def changed_catalog(path, **options):
        result = read_sqlite_catalog(path, **options)
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000_000))
        return result

    monkeypatch.setattr(release_module, "file_identity" if phase == "hash" else "read_sqlite_catalog",
                        changed_hash if phase == "hash" else changed_catalog)
    with pytest.raises(ContractError, match="changed during verification"):
        selected.artifact("maps")
    assert "maps" not in selected._verified


def test_failed_verification_releases_gate_and_retries(release, monkeypatch):
    selected = DataRelease(release)
    # Use the manifest's actual portable name rather than depending on its spelling.
    path = selected.path.parent / selected._artifacts["maps"]["filename"]
    identify = Mock(side_effect=[OSError("temporary read failure"), file_identity(path)])
    monkeypatch.setattr(release_module, "file_identity", identify)
    with pytest.raises(OSError, match="temporary read failure"):
        selected.artifact("maps")
    threads, results, errors = _threaded_artifacts(selected, ("maps",))
    _finish_artifact_threads(threads)
    assert not errors and results == [path] and identify.call_count == 2


@pytest.mark.parametrize("schema_version", [3, 4])
def test_native_leaf_verification_coalesces_and_survives_pickle(tmp_path, monkeypatch, schema_version):
    from test_source_parity import fixture_release

    selected, _, edition, _ = fixture_release(tmp_path, schema_version=schema_version)
    descriptor = selected.catalog.resolve_artifacts([(edition, "membership")])[0]["artifact"]
    identify = Mock(wraps=file_identity)
    semantic = Mock(wraps=release_module.validate_leaf_artifact)
    monkeypatch.setattr(release_module, "file_identity", identify)
    monkeypatch.setattr(release_module, "validate_leaf_artifact", semantic)
    threads, results, errors = _threaded_artifacts(selected, (descriptor.identity, descriptor.identity))
    _finish_artifact_threads(threads)
    assert not errors and results[0] == results[1]
    assert identify.call_count == semantic.call_count == 1
    for restored in (pickle.loads(pickle.dumps(selected)), deepcopy(selected)):
        assert restored.artifact(descriptor.identity) == results[0]
        assert not {"_artifact_locks", "_artifact_lock_pid"}.intersection(restored.__getstate__())


@pytest.mark.parametrize("separate_path", [False, True])
def test_native_aliases_share_gate_but_keep_path_and_descriptor_checks(tmp_path, monkeypatch, separate_path):
    from test_source_parity import fixture_release

    selected, _, edition, _ = fixture_release(tmp_path)
    ref = selected.catalog.resolve_artifacts([(edition, "membership")])[0]["artifact"]
    alias = replace(ref, output="alias", location="alias.parquet" if separate_path else ref.location)
    if separate_path:
        (tmp_path / alias.location).write_bytes((tmp_path / ref.location).read_bytes())
    monkeypatch.setattr(selected.catalog, "artifact_ref", lambda role: alias if role == "alias" else ref)
    identify = Mock(wraps=file_identity)
    semantic = Mock(wraps=release_module.validate_leaf_artifact)
    monkeypatch.setattr(release_module, "file_identity", identify)
    monkeypatch.setattr(release_module, "validate_leaf_artifact", semantic)
    first, second = selected.artifact(ref.identity), selected.artifact("alias")
    assert (first != second) is separate_path
    assert identify.call_count == (2 if separate_path else 1)
    assert [call.args[1] for call in semantic.call_args_list] == [ref, alias]


def test_legacy_release_state_excludes_locks_and_loads_old_state(release):
    selected = DataRelease(release)
    selected.artifact("maps")
    state = selected.__getstate__()
    assert not {"_artifact_locks", "_artifact_lock_pid"}.intersection(state)
    for restored in (pickle.loads(pickle.dumps(selected)), deepcopy(selected)):
        assert restored.artifact("maps") == selected.artifact("maps")
    old = DataRelease.__new__(DataRelease)
    old.__setstate__(state)
    assert old.artifact("maps") == selected.artifact("maps")


def test_windows_311_does_not_reuse_artifact_stat_cache(release, monkeypatch):
    selected = DataRelease(release)
    identify = Mock(wraps=file_identity)
    semantic = Mock(wraps=release_module.read_sqlite_catalog)
    monkeypatch.setattr(release_module, "_artifact_cache_reusable", lambda: False)
    monkeypatch.setattr(release_module, "file_identity", identify)
    monkeypatch.setattr(release_module, "read_sqlite_catalog", semantic)
    assert selected.artifact("maps") == selected.artifact("maps")
    assert identify.call_count == semantic.call_count == 2


@pytest.mark.parametrize("platform_name,python,expected", [
    ("nt", (3, 11), False), ("nt", (3, 12), True), ("posix", (3, 11), True),
])
def test_artifact_stat_reuse_follows_supported_platform_policy(monkeypatch, platform_name, python, expected):
    monkeypatch.setattr(release_module, "os", SimpleNamespace(name=platform_name))
    monkeypatch.setattr(release_module, "sys", SimpleNamespace(version_info=python))
    assert release_module._artifact_cache_reusable() is expected


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork is unavailable")
@pytest.mark.filterwarnings("ignore:This process .*multi-threaded:DeprecationWarning")
def test_forked_release_replaces_locks_owned_by_vanished_parent_thread(release):
    selected = DataRelease(release)
    gate = selected._artifact_lock(selected._artifacts["maps"]["sha256"])
    held, release_parent = Event(), Event()

    def hold():
        with gate, release_module._ARTIFACT_LOCK_REGISTRY:
            held.set()
            release_parent.wait(10)

    thread = Thread(target=hold, daemon=True)
    thread.start()
    assert held.wait(5)
    reader, writer = os.pipe()
    pid = None
    try:
        pid = os.fork()
        if pid == 0:
            os.close(reader)
            signal.alarm(5)
            try:
                selected.artifact("maps")
                os.write(writer, b"verified")
            finally:
                os._exit(0)
        os.close(writer)
        assert os.read(reader, 64) == b"verified"
    finally:
        if pid != 0:
            os.close(reader)
            release_parent.set()
            thread.join(timeout=5)
            if pid is not None:
                os.waitpid(pid, 0)


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
