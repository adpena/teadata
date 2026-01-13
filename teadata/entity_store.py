from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .assets import ensure_local_asset, is_lfs_pointer
from .teadata_config import canonical_campus_number, canonical_district_number

try:
    from .engine import _discover_snapshot  # type: ignore
except Exception:  # pragma: no cover - defensive import
    _discover_snapshot = None


_ENTITY_PREFIX = "entities_"
_LAST_LOGGED_ENTITY_STORE: Optional[str] = None

logger = logging.getLogger(__name__)


def entity_store_path_for_snapshot(snapshot_path: Path) -> Optional[Path]:
    name = snapshot_path.name
    if name.startswith("repo_"):
        base = name[len("repo_") :]
        if base.endswith(".pkl.gz"):
            base = base[: -len(".pkl.gz")]
        elif base.endswith(".pkl"):
            base = base[: -len(".pkl")]
        return snapshot_path.with_name(f"{_ENTITY_PREFIX}{base}.sqlite")
    return None


def _newest_sqlite(folder: Path) -> Optional[Path]:
    try:
        if not folder.exists() or not folder.is_dir():
            return None
        picks = sorted(
            folder.glob(f"{_ENTITY_PREFIX}*.sqlite"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return picks[0] if picks else None
    except Exception:
        return None


def _log_entity_store(path: Path, source: str) -> None:
    global _LAST_LOGGED_ENTITY_STORE
    try:
        path_str = str(path)
    except Exception:
        return
    if _LAST_LOGGED_ENTITY_STORE == path_str:
        return
    _LAST_LOGGED_ENTITY_STORE = path_str
    logger.info("entity_store.discovered source=%s path=%s", source, path_str)


def _resolve_entity_store(path: Path, source: str) -> Optional[Path]:
    resolved = ensure_local_asset(
        path, url_env="TEADATA_ENTITY_STORE_URL", label="entity store"
    )
    if not resolved.exists() or is_lfs_pointer(resolved):
        logger.warning("entity_store.unavailable source=%s path=%s", source, path)
        return None
    _log_entity_store(resolved, source)
    return resolved


def discover_entity_store(explicit: str | Path | None = None) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        if p.exists() and p.is_file():
            resolved = _resolve_entity_store(p, "explicit")
            if resolved:
                return resolved

    env = os.environ.get("TEADATA_ENTITY_STORE")
    if env:
        p = Path(env)
        if p.exists() and p.is_file():
            resolved = _resolve_entity_store(p, "env")
            if resolved:
                return resolved

    if _discover_snapshot:
        try:
            snap = _discover_snapshot()
            if snap:
                candidate = entity_store_path_for_snapshot(Path(snap))
                if candidate and candidate.exists():
                    resolved = _resolve_entity_store(candidate, "snapshot")
                    if resolved:
                        return resolved
        except Exception:
            pass

    try:
        package_dir = Path(__file__).resolve().parent
        pkg_cache = package_dir / ".cache"
        candidate = _newest_sqlite(pkg_cache)
        if candidate:
            resolved = _resolve_entity_store(candidate, "package-cache")
            if resolved:
                return resolved
    except Exception:
        pass

    for base in Path.cwd().parents:
        candidate = _newest_sqlite(base / ".cache")
        if candidate:
            resolved = _resolve_entity_store(candidate, "parent-cache")
            if resolved:
                return resolved

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


def _strip_leading_apostrophe(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.startswith("'"):
        return text[1:]
    return text


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


def _decode_json(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    if isinstance(payload, (bytes, bytearray)):
        try:
            text = payload.decode("utf-8")
        except Exception:
            return {}
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            return {}
    if isinstance(payload, str):
        if not payload:
            return {}
        try:
            return json.loads(payload)
        except Exception:
            return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _district_type_from_flag(flag: bool | None) -> str:
    return "Charter" if flag else "ISD"


def load_campus(
    campus_number: str,
    *,
    store_path: str | Path | None = None,
    include_meta: bool = True,
) -> Optional[dict[str, Any]]:
    path = discover_entity_store(store_path)
    if not path:
        return None

    query = """
        SELECT c.campus_number, c.campus_number_canon, c.name, c.is_charter, c.is_private,
               c.enrollment, c.rating, c.grade_range, c.lon, c.lat, c.meta,
               d.name, d.district_number, d.district_number_canon
        FROM campuses c
        LEFT JOIN districts d ON d.id = c.district_id
        WHERE c.campus_number_canon = ? OR c.campus_number = ?
    """

    try:
        with sqlite3.connect(path) as conn:
            for key in _campus_lookup_keys(campus_number):
                row = conn.execute(query, (key, key)).fetchone()
                if not row:
                    continue
                meta = _decode_json(row[10]) if include_meta else {}
                return {
                    "campus_number": row[0] or "",
                    "campus_number_canon": row[1] or "",
                    "name": row[2] or "",
                    "is_charter": bool(row[3]),
                    "is_private": bool(row[4]),
                    "enrollment": row[5],
                    "rating": row[6],
                    "grade_range": row[7] or "",
                    "lon": row[8],
                    "lat": row[9],
                    "meta": meta,
                    "district_name": row[11] or "",
                    "district_number": _strip_leading_apostrophe(row[12]),
                    "district_number_canon": row[13] or "",
                }
    except Exception:
        return None

    return None


def load_district(
    district_number: str,
    *,
    store_path: str | Path | None = None,
    include_meta: bool = True,
) -> Optional[dict[str, Any]]:
    path = discover_entity_store(store_path)
    if not path:
        return None

    query = """
        SELECT d.name, d.district_number, d.district_number_canon, d.enrollment, d.rating, d.meta,
               COALESCE(counts.campus_count, 0), COALESCE(counts.has_charter, 0)
        FROM districts d
        LEFT JOIN (
            SELECT district_id,
                   COUNT(*) AS campus_count,
                   MAX(CASE WHEN is_charter THEN 1 ELSE 0 END) AS has_charter
            FROM campuses
            GROUP BY district_id
        ) counts ON counts.district_id = d.id
        WHERE d.district_number_canon = ? OR d.district_number = ?
    """

    try:
        with sqlite3.connect(path) as conn:
            for key in _district_lookup_keys(district_number):
                row = conn.execute(query, (key, key)).fetchone()
                if not row:
                    continue
                has_charter = bool(row[7])
                meta = _decode_json(row[5]) if include_meta else {}
                return {
                    "name": row[0] or "",
                    "district_number": _strip_leading_apostrophe(row[1] or row[2]),
                    "district_number_canon": row[2] or "",
                    "enrollment": row[3],
                    "rating": row[4],
                    "meta": meta,
                    "campus_count": row[6] or 0,
                    "is_charter": has_charter,
                    "district_type": _district_type_from_flag(has_charter),
                }
    except Exception:
        return None

    return None


def list_meta_keys(
    entity_type: str, *, store_path: str | Path | None = None, limit: int | None = None
) -> list[str]:
    path = discover_entity_store(store_path)
    if not path:
        return []
    table = "district_meta" if entity_type == "district" else "campus_meta"
    query = f"SELECT DISTINCT key FROM {table} ORDER BY key"
    if limit:
        query = f"{query} LIMIT {int(limit)}"
    try:
        with sqlite3.connect(path) as conn:
            rows = conn.execute(query).fetchall()
    except Exception:
        return []
    keys = [row[0] for row in rows if row and row[0]]
    return keys


@dataclass
class EntityStore:
    path: Path
    max_cache: int = 64
    _conn: Optional[sqlite3.Connection] = field(default=None, init=False, repr=False)
    _campus_cache: dict[str, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )
    _district_cache: dict[str, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )

    def _connect(self) -> Optional[sqlite3.Connection]:
        if self._conn is not None:
            return self._conn
        if not self.path.exists():
            return None
        uri = f"file:{self.path}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        return self._conn

    @property
    def cache_key(self) -> str:
        try:
            stat = self.path.stat()
            return f"{self.path}:{stat.st_mtime_ns}"
        except Exception:
            return str(self.path)

    def get_campus(self, campus_number: str, *, include_meta: bool = True) -> Optional[dict[str, Any]]:
        if not campus_number:
            return None
        key = canonical_campus_number(campus_number) or str(campus_number).strip()
        if not key:
            return None
        cached = self._campus_cache.get(key)
        if cached is not None:
            if include_meta or not cached.get("meta"):
                return cached
        record = self._load_campus_from_conn(key, include_meta=include_meta)
        if record:
            self._campus_cache[key] = record
        return record

    def get_district(
        self, district_number: str, *, include_meta: bool = True
    ) -> Optional[dict[str, Any]]:
        if not district_number:
            return None
        key = canonical_district_number(district_number) or str(district_number).strip()
        if not key:
            return None
        cached = self._district_cache.get(key)
        if cached is not None:
            if include_meta or not cached.get("meta"):
                return cached
        record = self._load_district_from_conn(key, include_meta=include_meta)
        if record:
            self._district_cache[key] = record
        return record

    def iter_campuses(self, *, include_meta: bool = True) -> Iterable[dict[str, Any]]:
        conn = self._connect()
        if conn is None:
            return []
        query = """
            SELECT c.campus_number, c.campus_number_canon, c.name, c.is_charter, c.is_private,
                   c.enrollment, c.rating, c.grade_range, c.lon, c.lat, c.meta,
                   d.name, d.district_number, d.district_number_canon
            FROM campuses c
            LEFT JOIN districts d ON d.id = c.district_id
        """
        try:
            rows = conn.execute(query)
        except Exception:
            return []
        for row in rows:
            meta = _decode_json(row[10]) if include_meta else {}
            yield {
                "campus_number": row[0] or "",
                "campus_number_canon": row[1] or "",
                "name": row[2] or "",
                "is_charter": bool(row[3]),
                "is_private": bool(row[4]),
                "enrollment": row[5],
                "rating": row[6],
                "grade_range": row[7] or "",
                "lon": row[8],
                "lat": row[9],
                "meta": meta,
                "district_name": row[11] or "",
                "district_number": _strip_leading_apostrophe(row[12]),
                "district_number_canon": row[13] or "",
            }

    def iter_districts(self, *, include_meta: bool = True) -> Iterable[dict[str, Any]]:
        conn = self._connect()
        if conn is None:
            return []
        query = """
            SELECT d.name, d.district_number, d.district_number_canon, d.enrollment, d.rating, d.meta,
                   COALESCE(counts.campus_count, 0), COALESCE(counts.has_charter, 0)
            FROM districts d
            LEFT JOIN (
                SELECT district_id,
                       COUNT(*) AS campus_count,
                       MAX(CASE WHEN is_charter THEN 1 ELSE 0 END) AS has_charter
                FROM campuses
                GROUP BY district_id
            ) counts ON counts.district_id = d.id
        """
        try:
            rows = conn.execute(query)
        except Exception:
            return []
        for row in rows:
            has_charter = bool(row[7])
            meta = _decode_json(row[5]) if include_meta else {}
            yield {
                "name": row[0] or "",
                "district_number": _strip_leading_apostrophe(row[1] or row[2]),
                "district_number_canon": row[2] or "",
                "enrollment": row[3],
                "rating": row[4],
                "meta": meta,
                "campus_count": row[6] or 0,
                "is_charter": has_charter,
                "district_type": _district_type_from_flag(has_charter),
            }

    def list_meta_keys(self, entity_type: str, *, limit: int | None = None) -> list[str]:
        table = "district_meta" if entity_type == "district" else "campus_meta"
        query = f"SELECT DISTINCT key FROM {table} ORDER BY key"
        if limit:
            query = f"{query} LIMIT {int(limit)}"
        conn = self._connect()
        if conn is None:
            return []
        try:
            rows = conn.execute(query).fetchall()
        except Exception:
            return []
        return [row[0] for row in rows if row and row[0]]

    def _load_campus_from_conn(
        self, key: str, *, include_meta: bool = True
    ) -> Optional[dict[str, Any]]:
        conn = self._connect()
        if conn is None:
            return None
        query = """
            SELECT c.campus_number, c.campus_number_canon, c.name, c.is_charter, c.is_private,
                   c.enrollment, c.rating, c.grade_range, c.lon, c.lat, c.meta,
                   d.name, d.district_number, d.district_number_canon
            FROM campuses c
            LEFT JOIN districts d ON d.id = c.district_id
            WHERE c.campus_number_canon = ? OR c.campus_number = ?
        """
        try:
            row = conn.execute(query, (key, key)).fetchone()
        except Exception:
            row = None
        if not row:
            return None
        meta = _decode_json(row[10]) if include_meta else {}
        return {
            "campus_number": row[0] or "",
            "campus_number_canon": row[1] or "",
            "name": row[2] or "",
            "is_charter": bool(row[3]),
            "is_private": bool(row[4]),
            "enrollment": row[5],
            "rating": row[6],
            "grade_range": row[7] or "",
            "lon": row[8],
            "lat": row[9],
            "meta": meta,
            "district_name": row[11] or "",
            "district_number": _strip_leading_apostrophe(row[12]),
            "district_number_canon": row[13] or "",
        }

    def _load_district_from_conn(
        self, key: str, *, include_meta: bool = True
    ) -> Optional[dict[str, Any]]:
        conn = self._connect()
        if conn is None:
            return None
        query = """
            SELECT d.name, d.district_number, d.district_number_canon, d.enrollment, d.rating, d.meta,
                   COALESCE(counts.campus_count, 0), COALESCE(counts.has_charter, 0)
            FROM districts d
            LEFT JOIN (
                SELECT district_id,
                       COUNT(*) AS campus_count,
                       MAX(CASE WHEN is_charter THEN 1 ELSE 0 END) AS has_charter
                FROM campuses
                GROUP BY district_id
            ) counts ON counts.district_id = d.id
            WHERE d.district_number_canon = ? OR d.district_number = ?
        """
        try:
            row = conn.execute(query, (key, key)).fetchone()
        except Exception:
            row = None
        if not row:
            return None
        has_charter = bool(row[7])
        meta = _decode_json(row[5]) if include_meta else {}
        return {
            "name": row[0] or "",
            "district_number": _strip_leading_apostrophe(row[1] or row[2]),
            "district_number_canon": row[2] or "",
            "enrollment": row[3],
            "rating": row[4],
            "meta": meta,
            "campus_count": row[6] or 0,
            "is_charter": has_charter,
            "district_type": _district_type_from_flag(has_charter),
        }
