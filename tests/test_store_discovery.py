import pytest
from teadata import boundary_store, entity_store, map_store


@pytest.mark.parametrize(
    ("module", "method", "role"),
    [
        (boundary_store, "discover_boundary_store", "boundaries"),
        (entity_store, "discover_entity_store", "entities"),
        (map_store, "discover_map_store", "maps"),
    ],
)
def test_sidecars_select_one_release(monkeypatch, tmp_path, module, method, role):
    class Release:
        def artifact(self, selected):
            assert selected == role
            return tmp_path / (selected + ".sqlite")

    monkeypatch.setattr(module, "current_release", lambda: Release())
    monkeypatch.setenv("TEADATA_" + role.upper() + "_STORE", "unrelated.sqlite")
    assert getattr(module, method)() == tmp_path / (role + ".sqlite")
