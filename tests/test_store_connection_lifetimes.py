"""Owned SQLite readers close even when callers retain their connections."""

from contextlib import closing
import sqlite3

import pytest
from shapely.geometry import Point

from teadata import boundary_store, map_store
from teadata.manifest import ContractError, write_sqlite_catalog


@pytest.fixture
def stores(tmp_path, catalog):
    maps = tmp_path / "maps.sqlite"
    boundaries = tmp_path / "boundaries.sqlite"
    for path in (maps, boundaries):
        with closing(sqlite3.connect(path)) as connection:
            write_sqlite_catalog(connection, catalog)
            if path == maps:
                connection.execute("CREATE TABLE campus_profile(lookup_key TEXT,payload BLOB)")
                connection.execute("CREATE TABLE closures_map(district_number TEXT,payload_base BLOB,payload_transfers BLOB)")
                connection.execute("INSERT INTO campus_profile VALUES (?,?)", ("123456789", '{"campus":true}'))
                connection.execute("INSERT INTO closures_map VALUES (?,?,?)", ("123456", '{"district":true}', '{"transfers":true}'))
            else:
                connection.execute("CREATE TABLE boundaries(district_number TEXT,wkb BLOB)")
                connection.execute("INSERT INTO boundaries VALUES (?,?)", ("123456", Point(0, 0).wkb))
            connection.commit()
    return maps, boundaries


@pytest.fixture
def connections(monkeypatch):
    connect = sqlite3.connect
    retained = []

    def tracked(*args, **kwargs):
        connection = connect(*args, **kwargs)
        retained.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", tracked)
    try:
        yield retained
    finally:
        # Keep regressions hygienic even when an assertion discovers an open reader.
        for connection in retained:
            connection.close()


def assert_closed(connections):
    assert connections
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")


@pytest.mark.parametrize("reader", ["campus", "district", "boundary"])
@pytest.mark.parametrize("found", [True, False])
def test_payload_readers_close_owned_connections_on_hit_and_miss(stores, connections, reader, found):
    maps, boundaries = stores
    if reader == "campus":
        result = map_store.load_campus_profile_payload("123456789" if found else "987654321", store_path=maps)
        assert result == ({"campus": True} if found else None)
    elif reader == "district":
        result = map_store.load_map_payload_parts("123456" if found else "987654", store_path=maps)
        assert result == (({"district": True}, {"transfers": True}) if found else (None, None))
    else:
        result = boundary_store.load_boundary("123456" if found else "987654", store_path=boundaries)
        assert result.equals(Point(0, 0)) if found else result is None
    assert_closed(connections)


@pytest.mark.parametrize("reader", ["campus", "district", "boundary"])
def test_payload_decode_errors_preserve_policy_and_close_owned_connections(stores, connections, reader):
    maps, boundaries = stores
    with closing(sqlite3.connect(boundaries if reader == "boundary" else maps)) as connection:
        if reader == "campus":
            connection.execute("UPDATE campus_profile SET payload='invalid json'")
        elif reader == "district":
            connection.execute("UPDATE closures_map SET payload_base='invalid json'")
        else:
            connection.execute("UPDATE boundaries SET wkb=x'00'")
        connection.commit()
    if reader == "campus":
        with pytest.raises(ValueError):
            map_store.load_campus_profile_payload("123456789", store_path=maps)
    elif reader == "district":
        assert map_store.load_map_payload_parts("123456", store_path=maps) == (None, None)
    else:
        assert boundary_store.load_boundary("123456", store_path=boundaries) is None
    assert_closed(connections)


@pytest.mark.parametrize("failure", ["schema", "sqlite"])
def test_map_discovery_failure_closes_owned_connections(stores, connections, failure, monkeypatch):
    maps, _ = stores
    with closing(sqlite3.connect(maps)) as connection:
        connection.execute("DROP TABLE campus_profile")
        if failure == "schema":
            connection.execute("CREATE TABLE campus_profile(campus_number TEXT,payload BLOB)")
        connection.commit()
    class BrokenProbe(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql == "PRAGMA table_info(campus_profile)":
                raise sqlite3.OperationalError("fixture schema query failed")
            return super().execute(sql, *args, **kwargs)

    if failure == "schema":
        with pytest.raises(ContractError, match="rebuild"):
            map_store.discover_map_store(maps)
    else:
        tracked = sqlite3.connect
        monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: tracked(*args, **kwargs, factory=BrokenProbe))
        with pytest.raises(sqlite3.OperationalError, match="fixture schema query"):
            map_store.discover_map_store(maps)
    assert_closed(connections)


@pytest.mark.parametrize("reader", ["campus", "district", "boundary"])
def test_payload_sqlite_errors_preserve_policy_and_close_owned_connections(stores, connections, monkeypatch, reader):
    maps, boundaries = stores

    class BrokenRead(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql.startswith(("SELECT payload", "SELECT wkb")):
                raise sqlite3.OperationalError("fixture payload query failed")
            return super().execute(sql, *args, **kwargs)

    tracked = sqlite3.connect
    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: tracked(*args, **kwargs, factory=BrokenRead))
    if reader == "campus":
        with pytest.raises(ContractError, match="rebuild"):
            map_store.load_campus_profile_payload("123456789", store_path=maps)
    elif reader == "district":
        assert map_store.load_map_payload_parts("123456", store_path=maps) == (None, None)
    else:
        assert boundary_store.load_boundary("123456", store_path=boundaries) is None
    assert_closed(connections)
