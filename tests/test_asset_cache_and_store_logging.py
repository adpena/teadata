import logging
from pathlib import Path

from teadata import boundary_store, map_store


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
