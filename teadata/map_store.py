import gzip
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

from .teadata_config import canonical_campus_number, canonical_district_number

try:
    from .engine import _discover_snapshot  # type: ignore
except Exception:  # pragma: no cover - defensive import
    _discover_snapshot = None


_GZIP_MAGIC = b"\x1f\x8b"
_MAP_PREFIX = "map_payloads_"
_LAST_LOGGED_MAP_STORE: Optional[str] = None

logger = logging.getLogger(__name__)


def map_store_path_for_snapshot(snapshot_path: Path) -> Optional[Path]:
    name = snapshot_path.name
    if name.startswith("repo_"):
        base = name[len("repo_") :]
        if base.endswith(".pkl.gz"):
            base = base[: -len(".pkl.gz")]
        elif base.endswith(".pkl"):
            base = base[: -len(".pkl")]
        return snapshot_path.with_name(f"{_MAP_PREFIX}{base}.sqlite")
    return None


def _newest_sqlite(folder: Path) -> Optional[Path]:
    try:
        if not folder.exists() or not folder.is_dir():
            return None
        picks = sorted(folder.glob(f"{_MAP_PREFIX}*.sqlite"), key=lambda p: p.stat().st_mtime, reverse=True)
        return picks[0] if picks else None
    except Exception:
        return None


def _log_map_store(path: Path, source: str) -> None:
    global _LAST_LOGGED_MAP_STORE
    try:
        path_str = str(path)
    except Exception:
        return
    if _LAST_LOGGED_MAP_STORE == path_str:
        return
    _LAST_LOGGED_MAP_STORE = path_str
    logger.info("map_store.discovered source=%s path=%s", source, path_str)


def discover_map_store(explicit: str | Path | None = None) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        if p.exists() and p.is_file():
            _log_map_store(p, "explicit")
            return p

    env = os.environ.get("TEADATA_MAP_STORE")
    if env:
        p = Path(env)
        if p.exists() and p.is_file():
            _log_map_store(p, "env")
            return p

    if _discover_snapshot:
        try:
            snap = _discover_snapshot()
            if snap:
                candidate = map_store_path_for_snapshot(Path(snap))
                if candidate and candidate.exists():
                    _log_map_store(candidate, "snapshot")
                    return candidate
        except Exception:
            pass

    try:
        package_dir = Path(__file__).resolve().parent
        pkg_cache = package_dir / ".cache"
        candidate = _newest_sqlite(pkg_cache)
        if candidate:
            _log_map_store(candidate, "package-cache")
            return candidate
    except Exception:
        pass

    for base in Path.cwd().parents:
        candidate = _newest_sqlite(base / ".cache")
        if candidate:
            _log_map_store(candidate, "parent-cache")
            return candidate

    return None


def _district_lookup_keys(district_number: Optional[str]) -> Iterable[str]:
    if not district_number:
        return ()
    keys = []
    canonical = canonical_district_number(district_number)
    if canonical:
        keys.append(canonical)
    raw = str(district_number).strip()
    if raw and raw not in keys:
        keys.append(raw)
    if raw.startswith("'") and len(raw) > 1:
        digits = raw[1:]
        if digits not in keys:
            keys.append(digits)
        if digits.isdigit():
            normalized = str(int(digits))
            if normalized not in keys:
                keys.append(normalized)
    return keys


def _campus_lookup_keys(campus_number: Optional[str]) -> Iterable[str]:
    if not campus_number:
        return ()
    keys = []
    canonical = canonical_campus_number(campus_number)
    if canonical:
        keys.append(canonical)
    raw = str(campus_number).strip()
    if raw and raw not in keys:
        keys.append(raw)
    if raw.startswith("'") and len(raw) > 1:
        digits = raw[1:]
        if digits not in keys:
            keys.append(digits)
        if digits.isdigit():
            normalized = str(int(digits)).rjust(9, "0")
            if normalized not in keys:
                keys.append(normalized)
    return keys


def _decode_payload(payload: Any) -> Optional[dict]:
    if payload is None:
        return None
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    if isinstance(payload, bytes):
        data = payload
        if data.startswith(_GZIP_MAGIC):
            try:
                data = gzip.decompress(data)
            except Exception:
                return None
        try:
            text = data.decode("utf-8")
        except Exception:
            return None
        return json.loads(text) if text else None
    if isinstance(payload, str):
        return json.loads(payload) if payload else None
    return None


def load_map_payload_parts(
    district_number: str, *, store_path: str | Path | None = None
) -> Tuple[Optional[dict], Optional[dict]]:
    path = discover_map_store(store_path)
    if not path:
        return None, None

    try:
        with sqlite3.connect(path) as conn:
            for key in _district_lookup_keys(district_number):
                row = conn.execute(
                    "SELECT payload_base, payload_transfers FROM closures_map WHERE district_number = ?",
                    (key,),
                ).fetchone()
                if row:
                    base = _decode_payload(row[0])
                    transfers = _decode_payload(row[1]) or {}
                    return base, transfers
    except Exception:
        return None, None

    return None, None


def load_campus_profile_payload(
    campus_number: str, *, store_path: str | Path | None = None
) -> Optional[dict]:
    path = discover_map_store(store_path)
    if not path:
        return None

    try:
        with sqlite3.connect(path) as conn:
            for key in _campus_lookup_keys(campus_number):
                row = conn.execute(
                    "SELECT payload FROM campus_profile WHERE campus_number = ?",
                    (key,),
                ).fetchone()
                if row:
                    return _decode_payload(row[0])
    except Exception:
        return None

    return None
