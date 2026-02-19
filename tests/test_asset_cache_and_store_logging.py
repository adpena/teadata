import io
import logging
from pathlib import Path

from teadata import assets, boundary_store, map_store


class _DummyResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def test_ensure_local_asset_namespaces_cache_by_url(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("TEADATA_ASSET_CACHE_DIR", str(tmp_path / "asset-cache"))
    monkeypatch.setenv("URL_ONE", "https://example.com/a/latest.sqlite")
    monkeypatch.setenv("URL_TWO", "https://example.com/b/latest.sqlite")

    payload_by_url = {
        "https://example.com/a/latest.sqlite": b"asset-one",
        "https://example.com/b/latest.sqlite": b"asset-two",
    }

    def fake_urlopen(req, timeout=60):  # noqa: ARG001
        return _DummyResponse(payload_by_url[req.full_url])

    monkeypatch.setattr(assets.urllib.request, "urlopen", fake_urlopen)

    local_hint = tmp_path / "missing.sqlite"
    first = assets.ensure_local_asset(local_hint, url_env="URL_ONE", label="map store")
    second = assets.ensure_local_asset(
        local_hint, url_env="URL_TWO", label="map store"
    )

    assert first != second
    assert first.exists()
    assert second.exists()
    assert first.read_bytes() == b"asset-one"
    assert second.read_bytes() == b"asset-two"


def test_map_store_logs_errors_when_sqlite_read_fails(
    monkeypatch, tmp_path: Path, caplog
):
    invalid = tmp_path / "invalid.sqlite"
    invalid.write_text("not a sqlite database", encoding="utf-8")
    monkeypatch.setattr(map_store, "discover_map_store", lambda _store_path=None: invalid)

    with caplog.at_level(logging.ERROR, logger="teadata.map_store"):
        payload = map_store.load_map_payload_parts("123456")

    assert payload == (None, None)
    assert any(
        "map_store.load_map_payload_parts_failed" in rec.getMessage()
        for rec in caplog.records
    )


def test_boundary_store_logs_errors_when_sqlite_read_fails(
    monkeypatch, tmp_path: Path, caplog
):
    invalid = tmp_path / "invalid.sqlite"
    invalid.write_text("not a sqlite database", encoding="utf-8")
    monkeypatch.setattr(
        boundary_store, "discover_boundary_store", lambda _store_path=None: invalid
    )

    class _FakeWkb:
        @staticmethod
        def loads(_blob):
            return object()

    monkeypatch.setattr(boundary_store, "shapely_wkb", _FakeWkb())

    with caplog.at_level(logging.ERROR, logger="teadata.boundary_store"):
        payload = boundary_store.load_boundary("123456")

    assert payload is None
    assert any(
        "boundary_store.load_boundary_failed" in rec.getMessage()
        for rec in caplog.records
    )
