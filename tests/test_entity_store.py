from contextlib import closing
import sqlite3
import uuid

import pytest

from teadata.classes import Campus, DataEngine, District
from teadata.persistence.sqlalchemy_store import (
    create_engine,
    create_sessionmaker,
    ensure_schema,
    export_dataengine,
)
from teadata import entity_store


def _build_entity_store(tmp_path, catalog):
    repo = DataEngine()
    district = District(
        id=uuid.uuid4(),
        name="Test ISD",
        district_number="123456",
        enrollment=1200,
        rating="A",
    )
    district.meta = {"district_2025_demo": "ok", "district_extra": "extra"}
    repo.add_district(district)

    campus = Campus(
        id=uuid.uuid4(),
        district_id=district.id,
        name="Test Campus",
        charter_type="Charter",
        is_charter=True,
    )
    campus.campus_number = "123456789"
    campus.district_number = "123456"
    campus.enrollment = 200
    campus.meta = {"campus_2025_demo": 5}
    repo.add_campus(campus)

    path = tmp_path / "entities.sqlite"
    engine = create_engine(f"sqlite:///{path}")
    ensure_schema(engine)
    Session = create_sessionmaker(engine, expire_on_commit=False)
    with Session.begin() as session:
        export_dataengine(repo, session, replace=True)
    engine.dispose()
    from teadata.manifest import write_sqlite_catalog
    with closing(sqlite3.connect(path)) as conn:
        write_sqlite_catalog(conn, catalog)
        conn.commit()
    return path


def test_entity_store_round_trip(tmp_path, catalog):
    path = _build_entity_store(tmp_path, catalog)

    campus_row = entity_store.load_campus("123456789", store_path=path)
    assert campus_row
    assert campus_row["campus_number"] == "123456789"
    assert campus_row["meta"]["campus_2025_demo"] == 5

    district_row = entity_store.load_district("123456", store_path=path)
    assert district_row
    assert district_row["district_number"] == "123456"
    assert district_row["campus_count"] == 1
    assert district_row["is_charter"] is True
    assert district_row["district_type"] == "Charter"

    keys = entity_store.list_meta_keys("campus", store_path=path)
    assert "campus_2025_demo" in keys


def test_entity_store_get_meta_after_no_meta_lookup(tmp_path, catalog):
    path = _build_entity_store(tmp_path, catalog)
    store = entity_store.EntityStore(path)

    campus_no_meta = store.get_campus("123456789", include_meta=False)
    assert campus_no_meta
    assert campus_no_meta["meta"] == {}
    campus_with_meta = store.get_campus("123456789", include_meta=True)
    assert campus_with_meta
    assert campus_with_meta["meta"]["campus_2025_demo"] == 5

    district_no_meta = store.get_district("123456", include_meta=False)
    assert district_no_meta
    assert district_no_meta["meta"] == {}
    district_with_meta = store.get_district("123456", include_meta=True)
    assert district_with_meta
    assert district_with_meta["meta"]["district_2025_demo"] == "ok"


def test_entity_store_meta_keys_limit_does_not_truncate_cache(tmp_path, catalog):
    path = _build_entity_store(tmp_path, catalog)
    store = entity_store.EntityStore(path)

    limited = store.list_meta_keys("district", limit=1)
    full = store.list_meta_keys("district")

    assert len(limited) == 1
    assert "district_2025_demo" in full
    assert "district_extra" in full


def test_entity_store_load_raises_for_invalid_database(tmp_path, catalog):
    invalid = tmp_path / "invalid.sqlite"
    invalid.write_text("not a sqlite database", encoding="utf-8")

    with pytest.raises(RuntimeError, match="required dataset catalog"):
        entity_store.load_campus("123456789", store_path=invalid)
    with pytest.raises(RuntimeError, match="required dataset catalog"):
        entity_store.load_district("123456", store_path=invalid)


@pytest.fixture
def connections_opened(monkeypatch):
    connect = sqlite3.connect
    opened = []

    def tracked_connect(*args, **kwargs):
        conn = connect(*args, **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    return opened


def assert_connections_closed(connections):
    assert connections
    for conn in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            conn.execute("SELECT 1")


@pytest.mark.parametrize("flag", ["0", "1"])
@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("include_meta", [False, True])
def test_scoped_store_reuses_connection_and_catalog_without_implicit_memoization(
    tmp_path, catalog, monkeypatch, connections_opened, flag, cache, include_meta
):
    path = _build_entity_store(tmp_path, catalog)
    # Fixture construction is outside the resource lifetime under review.
    connections_opened.clear()
    monkeypatch.setenv("TEADATA_ENTITY_STORE_IN_MEMORY_CACHE", flag)
    store = entity_store.EntityStore(path)
    with store as reader:
        assert reader is store and reader.catalog.build_id == catalog.build_id
        for _ in range(20):
            campus = reader.get_campus("'123456789", include_meta=include_meta, cache=cache)
            district = reader.get_district("'123456", include_meta=include_meta, cache=cache)
            assert campus is not None and district is not None
            assert campus["source_build_id"] == district["source_build_id"] == catalog.build_id
            assert campus["is_charter"] is True and campus["is_private"] is False
            assert district["district_type"] == "Charter" and district["campus_count"] == 1
            assert campus["meta"] == ({"campus_2025_demo": 5} if include_meta else {})
            assert district["meta"] == ({"district_2025_demo": "ok", "district_extra": "extra"} if include_meta else {})
        # One transient catalog reader and one persistent query connection.
        assert len(connections_opened) == 2
        assert_connections_closed(connections_opened[:1])
        assert connections_opened[1].execute("SELECT 1").fetchone() == (1,)
        assert bool(reader._campus_cache) is cache
        assert bool(reader._district_cache) is cache
        assert bool(reader._campus_meta_loaded) is (cache and include_meta)
        assert bool(reader._district_meta_loaded) is (cache and include_meta)
    assert_connections_closed(connections_opened)


@pytest.mark.parametrize("kind,number", [("campus", "123456789"), ("district", "123456")])
def test_uncached_lookup_ignores_existing_records_without_poisoning_their_meta_state(
    tmp_path, catalog, kind, number
):
    path = _build_entity_store(tmp_path, catalog)
    with entity_store.EntityStore(path) as store:
        lookup = getattr(store, f"get_{kind}")
        cached = lookup(number, include_meta=False)
        cached["name"] = "Caller-mutated cached record"
        fresh = lookup(number, include_meta=True, cache=False)
        assert fresh is not cached and fresh["name"] != cached["name"]
        assert fresh["meta"] and cached["meta"] == {}
        # An uncached full read must not mark the cached partial row as full.
        upgraded = lookup(number, include_meta=True)
        assert upgraded["meta"] == fresh["meta"]
        assert lookup(number) is upgraded  # Default caching stays compatible.
        minimal = lookup(number, include_meta=False, cache=False)
        assert minimal is not upgraded and minimal["meta"] == {}
        assert lookup(number) is upgraded


def test_close_is_idempotent_clears_owned_caches_and_prevents_reopening(tmp_path, catalog):
    path = _build_entity_store(tmp_path, catalog)
    store = entity_store.EntityStore(path)
    with store:
        store.get_campus("123456789")
        store.get_district("123456")
        store.list_meta_keys("campus")
        store.list_meta_keys("district")
        connection = store._conn
    store.close()
    assert store._conn is None
    assert not store._campus_cache and not store._district_cache
    assert not store._campus_meta_loaded and not store._district_meta_loaded
    assert not store._meta_keys_cache and "catalog" not in store.__dict__
    assert_connections_closed([connection])
    for use in [
        lambda: store.get_campus("123456789", cache=False),
        lambda: store.get_district("123456", cache=False),
        lambda: store.list_meta_keys("campus"),
        lambda: next(iter(store.iter_campuses())),
        lambda: next(iter(store.iter_districts())),
        lambda: store.cache_key,
        lambda: store.__enter__(),
    ]:
        with pytest.raises(RuntimeError, match="EntityStore is closed"):
            use()


def test_scoped_store_closes_connection_on_consumer_error(tmp_path, catalog, connections_opened):
    path = _build_entity_store(tmp_path, catalog)
    connections_opened.clear()
    store = entity_store.EntityStore(path)
    with pytest.raises(ValueError, match="consumer failed"):
        with store:
            store.get_campus("123456789")
            raise ValueError("consumer failed")
    assert_connections_closed(connections_opened)
    assert store._conn is None and not store._campus_cache


def test_context_entry_closes_catalog_connection_when_validation_fails(tmp_path, connections_opened):
    invalid = tmp_path / "empty.sqlite"
    invalid.touch()
    store = entity_store.EntityStore(invalid)
    with pytest.raises(RuntimeError, match="required dataset catalog"):
        with store:
            pytest.fail("Invalid catalogs must not enter the reader scope")
    assert_connections_closed(connections_opened)
    assert store._conn is None and "catalog" not in store.__dict__
    with pytest.raises(RuntimeError, match="EntityStore is closed"):
        store.get_campus("123456789", cache=False)


@pytest.mark.parametrize("method,argument", [("load_campus", "123456789"), ("load_district", "123456"), ("list_meta_keys", "campus")])
def test_one_shot_helpers_close_their_catalog_and_query_connections(
    tmp_path, catalog, connections_opened, method, argument
):
    path = _build_entity_store(tmp_path, catalog)
    connections_opened.clear()
    assert getattr(entity_store, method)(argument, store_path=path)
    assert len(connections_opened) == 2
    assert_connections_closed(connections_opened)
