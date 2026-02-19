from pathlib import Path

import pytest

from teadata import boundary_store, entity_store, map_store


@pytest.mark.parametrize(
    ("module", "discover_name", "resolve_name", "env_name"),
    [
        (map_store, "discover_map_store", "_resolve_map_store", "TEADATA_MAP_STORE"),
        (
            boundary_store,
            "discover_boundary_store",
            "_resolve_boundary_store",
            "TEADATA_BOUNDARY_STORE",
        ),
        (
            entity_store,
            "discover_entity_store",
            "_resolve_entity_store",
            "TEADATA_ENTITY_STORE",
        ),
    ],
)
def test_discover_store_checks_cwd_cache_before_parents(
    monkeypatch, tmp_path: Path, module, discover_name: str, resolve_name: str, env_name: str
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setattr(module, "_discover_snapshot", None)

    candidate = tmp_path / ".cache" / "candidate.sqlite"
    seen_sources: list[str] = []

    def fake_newest_sqlite(folder: Path):
        if folder == tmp_path / ".cache":
            return candidate
        return None

    def fake_resolve(path: Path, source: str):
        seen_sources.append(source)
        return path

    monkeypatch.setattr(module, "_newest_sqlite", fake_newest_sqlite)
    monkeypatch.setattr(module, resolve_name, fake_resolve)

    discover = getattr(module, discover_name)
    found = discover()

    assert found == candidate
    assert seen_sources == ["cwd-cache"]
