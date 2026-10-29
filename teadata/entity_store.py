from __future__ import annotations

from contextlib import closing
import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable, Optional

from .manifest import ContractError, read_sqlite_catalog, read_sqlite_catalog_connection
from .release import current_release
from .teadata_config import canonical_campus_number, canonical_district_number

logger = logging.getLogger(__name__)


def entity_store_path_for_snapshot(snapshot_path: Path) -> Path:
    base = snapshot_path.name.removeprefix("repo_").removesuffix(".gz").removesuffix(".pkl")
    return snapshot_path.with_name(f"entities_{base}.sqlite")


def discover_entity_store(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        read_sqlite_catalog(path)
        return path
    return current_release().artifact("entities")


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
    """Decode SQLite metadata without replacing malformed/unsupported data.

    SQL NULL and JSON null represent empty nullable metadata. All other values
    must be UTF-8 JSON objects; parser security limits remain in force.
    """
    if payload is None:
        return {}
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = payload.decode("utf-8")
        except UnicodeError as error:
            raise ContractError("Entity metadata must be UTF-8 JSON") from error
    if not isinstance(payload, str):
        raise ContractError("Entity metadata storage must be JSON text or SQL NULL")
    try:
        value = json.loads(payload, parse_constant=_reject_json_constant, parse_float=_finite_json_float)
    except (ValueError, RecursionError) as error:
        raise ContractError(f"Entity metadata contains malformed or unsupported JSON: {error}") from error
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ContractError("Entity metadata must be a JSON object or null")
    return value


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON constant {value}")


def _finite_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON number exceeds the finite Float64 range")
    return number


def _collect_meta_keys_from_json(
    conn: sqlite3.Connection, table: str, *, limit: int | None = None
) -> list[str]:
    keys: set[str] = set()
    try:
        rows = conn.execute(f"SELECT meta FROM {table}")
    except Exception:
        return []
    for row in rows:
        if not row:
            continue
        meta = _decode_json(row[0])
        keys.update(meta.keys())
        if limit and len(keys) >= limit:
            break
    return sorted(keys)


def _district_type_from_flag(flag: bool | None) -> str:
    return "Charter" if flag else "ISD"


def _nullable_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


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
               c.enrollment, c.rating, c.aea, c.aea_raw, c.grade_range, c.lon, c.lat, c.meta,
               d.name, d.district_number, d.district_number_canon
        FROM campuses c
        LEFT JOIN districts d ON d.id = c.district_id
        WHERE c.campus_number_canon = ? OR c.campus_number = ?
    """

    try:
        with closing(sqlite3.connect(path)) as conn:
            for key in _campus_lookup_keys(campus_number):
                row = conn.execute(query, (key, key)).fetchone()
                if not row:
                    continue
                meta = _decode_json(row[12]) if include_meta else {}
                return {
                    "campus_number": row[0] or "",
                    "campus_number_canon": row[1] or "",
                    "name": row[2] or "",
                    "is_charter": bool(row[3]),
                    "is_private": bool(row[4]),
                    "enrollment": row[5],
                    "rating": row[6],
                    "aea": _nullable_bool(row[7]),
                    "aea_raw": row[8],
                    "grade_range": row[9] or "",
                    "lon": row[10],
                    "lat": row[11],
                    "meta": meta,
                    "district_name": row[13] or "",
                    "district_number": _strip_leading_apostrophe(row[14]),
                    "district_number_canon": row[15] or "",
                }
    except ContractError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed querying campus from entity store '{path}': {exc}") from exc

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
        SELECT d.name, d.district_number, d.district_number_canon, d.enrollment, d.rating, d.aea, d.aea_raw, d.meta,
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
        with closing(sqlite3.connect(path)) as conn:
            for key in _district_lookup_keys(district_number):
                row = conn.execute(query, (key, key)).fetchone()
                if not row:
                    continue
                has_charter = bool(row[9])
                meta = _decode_json(row[7]) if include_meta else {}
                return {
                    "name": row[0] or "",
                    "district_number": _strip_leading_apostrophe(row[1] or row[2]),
                    "district_number_canon": row[2] or "",
                    "enrollment": row[3],
                    "rating": row[4],
                    "aea": _nullable_bool(row[5]),
                    "aea_raw": row[6],
                    "meta": meta,
                    "campus_count": row[8] or 0,
                    "is_charter": has_charter,
                    "district_type": _district_type_from_flag(has_charter),
                }
    except ContractError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Failed querying district from entity store '{path}': {exc}"
        ) from exc

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
        with closing(sqlite3.connect(path)) as conn:
            rows = conn.execute(query).fetchall()
            keys = [row[0] for row in rows if row and row[0]]
            if keys:
                return keys
            return _collect_meta_keys_from_json(
                conn,
                "districts" if entity_type == "district" else "campuses",
                limit=limit,
            )
    except ContractError:
        raise
    except Exception:
        return []


@dataclass
class EntityStore:
    """A reusable catalog-bound reader; scoped use closes all owned resources.

    Lookups memoize by default. Streaming consumers can use ``cache=False``
    inside ``with EntityStore(path) as store`` to retain only caller-owned rows.
    A closed store cannot be reopened; create another reader for another scope.
    """

    path: Path
    max_cache: int = 64
    immutable: bool = False
    _conn: Optional[sqlite3.Connection] = field(default=None, init=False, repr=False)
    _campus_cache: dict[str, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )
    _district_cache: dict[str, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )
    _campus_meta_loaded: set[str] = field(default_factory=set, init=False, repr=False)
    _district_meta_loaded: set[str] = field(
        default_factory=set, init=False, repr=False
    )
    _meta_keys_cache: dict[str, list[str]] = field(
        default_factory=dict, init=False, repr=False
    )
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        self.path = Path(self.path)
        if type(self.immutable) is not bool:
            raise ValueError("EntityStore immutable mode must be boolean")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("EntityStore is closed")

    def close(self) -> None:
        """Close the SQLite connection and release every store-owned cache."""
        conn, self._conn = self._conn, None
        self._closed = True
        try:
            if conn is not None:
                conn.close()
        finally:
            self._campus_cache.clear()
            self._district_cache.clear()
            self._campus_meta_loaded.clear()
            self._district_meta_loaded.clear()
            self._meta_keys_cache.clear()
            self.__dict__.pop("catalog", None)

    def __enter__(self) -> EntityStore:
        self._ensure_open()
        try:
            self.catalog
            self._connect()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @cached_property
    def catalog(self):
        self._ensure_open()
        return read_sqlite_catalog_connection(self.connection)

    @property
    def connection(self) -> sqlite3.Connection:
        """The store-owned read-only SQLite connection; closed with this store.

        Immutable mode reads only the exact checkpointed main-file artifact,
        including its catalog. Callers may execute projections on this same
        connection rather than opening a reader with different WAL semantics.
        """
        connection = self._connect()
        if connection is None:
            raise ContractError(f"Entity artifact does not exist: {self.path.name}")
        return connection

    def _connect(self) -> Optional[sqlite3.Connection]:
        self._ensure_open()
        if self._conn is not None:
            return self._conn
        if not self.path.exists():
            return None
        uri = self.path.resolve().as_uri() + "?mode=ro" + ("&immutable=1" if self.immutable else "")
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        return self._conn

    @property
    def cache_key(self) -> str:
        return self.catalog.build_id

    def get_campus(self, campus_number: str, *, include_meta: bool = True, cache: bool = True) -> Optional[dict[str, Any]]:
        self._ensure_open()
        if not campus_number:
            return None
        key = canonical_campus_number(campus_number) or str(campus_number).strip()
        if not key:
            return None
        cached = self._campus_cache.get(key) if cache else None
        if cached is not None:
            if not include_meta or key in self._campus_meta_loaded:
                return cached
        record = self._load_campus_from_conn(key, include_meta=include_meta)
        if record:
            record["source_build_id"] = self.catalog.build_id
            if cache:
                self._campus_cache[key] = record
                if include_meta:
                    self._campus_meta_loaded.add(key)
                else:
                    self._campus_meta_loaded.discard(key)
        return record

    def get_district(
        self, district_number: str, *, include_meta: bool = True, cache: bool = True
    ) -> Optional[dict[str, Any]]:
        self._ensure_open()
        if not district_number:
            return None
        key = canonical_district_number(district_number) or str(district_number).strip()
        if not key:
            return None
        cached = self._district_cache.get(key) if cache else None
        if cached is not None:
            if not include_meta or key in self._district_meta_loaded:
                return cached
        record = self._load_district_from_conn(key, include_meta=include_meta)
        if record:
            record["source_build_id"] = self.catalog.build_id
            if cache:
                self._district_cache[key] = record
                if include_meta:
                    self._district_meta_loaded.add(key)
                else:
                    self._district_meta_loaded.discard(key)
        return record

    def iter_campuses(self, *, include_meta: bool = True) -> Iterable[dict[str, Any]]:
        conn = self._connect()
        if conn is None:
            return []
        query = """
            SELECT c.campus_number, c.campus_number_canon, c.name, c.is_charter, c.is_private,
                   c.enrollment, c.rating, c.aea, c.aea_raw, c.grade_range, c.lon, c.lat, c.meta,
                   d.name, d.district_number, d.district_number_canon
            FROM campuses c
            LEFT JOIN districts d ON d.id = c.district_id
        """
        try:
            rows = conn.execute(query)
        except Exception:
            return []
        for row in rows:
            meta = _decode_json(row[12]) if include_meta else {}
            yield {
                "source_build_id": self.catalog.build_id,
                "campus_number": row[0] or "",
                "campus_number_canon": row[1] or "",
                "name": row[2] or "",
                "is_charter": bool(row[3]),
                "is_private": bool(row[4]),
                "enrollment": row[5],
                "rating": row[6],
                "aea": _nullable_bool(row[7]),
                "aea_raw": row[8],
                "grade_range": row[9] or "",
                "lon": row[10],
                "lat": row[11],
                "meta": meta,
                "district_name": row[13] or "",
                "district_number": _strip_leading_apostrophe(row[14]),
                "district_number_canon": row[15] or "",
            }

    def iter_districts(self, *, include_meta: bool = True) -> Iterable[dict[str, Any]]:
        conn = self._connect()
        if conn is None:
            return []
        query = """
            SELECT d.name, d.district_number, d.district_number_canon, d.enrollment, d.rating, d.aea, d.aea_raw, d.meta,
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
            has_charter = bool(row[9])
            meta = _decode_json(row[7]) if include_meta else {}
            yield {
                "source_build_id": self.catalog.build_id,
                "name": row[0] or "",
                "district_number": _strip_leading_apostrophe(row[1] or row[2]),
                "district_number_canon": row[2] or "",
                "enrollment": row[3],
                "rating": row[4],
                "aea": _nullable_bool(row[5]),
                "aea_raw": row[6],
                "meta": meta,
                "campus_count": row[8] or 0,
                "is_charter": has_charter,
                "district_type": _district_type_from_flag(has_charter),
            }

    def list_meta_keys(self, entity_type: str, *, limit: int | None = None) -> list[str]:
        self._ensure_open()
        cached = self._meta_keys_cache.get(entity_type)
        if cached is not None:
            return cached[:limit] if limit else list(cached)
        table = "district_meta" if entity_type == "district" else "campus_meta"
        query = f"SELECT DISTINCT key FROM {table} ORDER BY key"
        conn = self._connect()
        if conn is None:
            return []
        keys: list[str] = []
        try:
            rows = conn.execute(query).fetchall()
            keys = [row[0] for row in rows if row and row[0]]
        except Exception:
            keys = []
        if not keys:
            keys = _collect_meta_keys_from_json(
                conn,
                "districts" if entity_type == "district" else "campuses",
            )
        self._meta_keys_cache[entity_type] = keys
        return keys[:limit] if limit else keys

    def _load_campus_from_conn(
        self, key: str, *, include_meta: bool = True
    ) -> Optional[dict[str, Any]]:
        conn = self._connect()
        if conn is None:
            return None
        query = """
            SELECT c.campus_number, c.campus_number_canon, c.name, c.is_charter, c.is_private,
                   c.enrollment, c.rating, c.aea, c.aea_raw, c.grade_range, c.lon, c.lat, c.meta,
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
        meta = _decode_json(row[12]) if include_meta else {}
        return {
            "campus_number": row[0] or "",
            "campus_number_canon": row[1] or "",
            "name": row[2] or "",
            "is_charter": bool(row[3]),
            "is_private": bool(row[4]),
            "enrollment": row[5],
            "rating": row[6],
            "aea": _nullable_bool(row[7]),
            "aea_raw": row[8],
            "grade_range": row[9] or "",
            "lon": row[10],
            "lat": row[11],
            "meta": meta,
            "district_name": row[13] or "",
            "district_number": _strip_leading_apostrophe(row[14]),
            "district_number_canon": row[15] or "",
        }

    def _load_district_from_conn(
        self, key: str, *, include_meta: bool = True
    ) -> Optional[dict[str, Any]]:
        conn = self._connect()
        if conn is None:
            return None
        query = """
            SELECT d.name, d.district_number, d.district_number_canon, d.enrollment, d.rating, d.aea, d.aea_raw, d.meta,
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
        has_charter = bool(row[9])
        meta = _decode_json(row[7]) if include_meta else {}
        return {
            "name": row[0] or "",
            "district_number": _strip_leading_apostrophe(row[1] or row[2]),
            "district_number_canon": row[2] or "",
            "enrollment": row[3],
            "rating": row[4],
            "aea": _nullable_bool(row[5]),
            "aea_raw": row[6],
            "meta": meta,
            "campus_count": row[8] or 0,
            "is_charter": has_charter,
            "district_type": _district_type_from_flag(has_charter),
        }


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
