import pytest
import sqlite3
from teadata import boundary_store, entity_store, map_store
from teadata.manifest import write_sqlite_catalog


@pytest.mark.parametrize(
    ("module", "method", "role"),
    [
        (boundary_store, "discover_boundary_store", "boundaries"),
        (entity_store, "discover_entity_store", "entities"),
        (map_store, "discover_map_store", "maps"),
    ],
)
def test_sidecars_select_one_release(monkeypatch, tmp_path, catalog, module, method, role):
    with sqlite3.connect(tmp_path / (role + ".sqlite")) as connection:
        write_sqlite_catalog(connection, catalog)
        if role == "maps":
            connection.execute("CREATE TABLE campus_profile (lookup_key TEXT PRIMARY KEY, payload BLOB)")

    class Release:
        def artifact(self, selected):
            assert selected == role
            return tmp_path / (selected + ".sqlite")

    monkeypatch.setattr(module, "current_release", lambda: Release())
    monkeypatch.setenv("TEADATA_" + role.upper() + "_STORE", "unrelated.sqlite")
    assert getattr(module, method)() == tmp_path / (role + ".sqlite")
