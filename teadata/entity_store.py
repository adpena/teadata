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

from ._sqlite import readonly_connection
from .attributes import district_type_from_subtype, district_type_label
from .manifest import ContractError, read_sqlite_catalog, read_sqlite_catalog_connection
from .release import current_release
from .identifiers import (
    EntityIdentityError,
    campus_lookup_key,
    canonical_source_identifier,
    normalize_campus_lookup_key,
    validate_stored_district_numbers,
)

logger = logging.getLogger(__name__)


def _campus_select(*, include_meta: bool) -> str:
    meta = "c.meta" if include_meta else "NULL AS meta"
    return f"""
    SELECT c.campus_number, c.campus_number_canon, c.name, c.is_charter, c.is_private,
           c.enrollment, c.rating, c.aea, c.aea_raw, c.grade_range, c.lon, c.lat, {meta},
           d.name, d.district_number, d.district_number_canon,
           c.lookup_key,c.source_namespace,c.source_id,c.id,c.district_id
    FROM campuses c LEFT JOIN districts d ON d.id = c.district_id
"""


def _district_select(*, include_meta: bool, has_projection: bool, lookup: bool = False) -> str:
    meta = "d.meta" if include_meta else "NULL AS meta"
    fields = ("a.organization_sub_type, a.district_boundary_status, a.district_boundary_reason, a.district_id"
              if has_projection else "NULL, NULL, NULL, NULL")
    join = "LEFT JOIN district_source_projection a ON a.district_id = d.id" if has_projection else ""
    query = f"""
        SELECT d.name, d.district_number, d.district_number_canon, d.enrollment, d.rating, d.aea, d.aea_raw, {meta},
               COALESCE(counts.campus_count, 0), d.id, {fields}
        FROM districts d
        LEFT JOIN (
            SELECT district_id,
                   COUNT(*) AS campus_count
            FROM campuses
            GROUP BY district_id
        ) counts ON counts.district_id = d.id
        {join}
    """
    if lookup:
        query += " WHERE d.district_number_canon IN (?, ?) OR d.district_number IN (?, ?)"
    return query


def _unique_row(cursor: sqlite3.Cursor, kind: str, key: str) -> tuple | None:
    rows = cursor.fetchmany(2)
    if len(rows) > 1:
        raise ContractError(f"Entity store contains duplicate {kind} {key!r}")
    return rows[0] if rows else None


def _campus_record(row, *, include_meta: bool, build_id: str) -> dict[str, Any]:
    try:
        key = campus_lookup_key(campus_number=row[0], source_namespace=row[17], source_id=row[18], is_private=bool(row[4]))
    except (ValueError, TypeError) as error:
        raise ContractError("Stored campus has an invalid source identity") from error
    if key is None or key != row[16]:
        raise ContractError("Stored campus key differs from its source identity")
    if bool(row[4]) and (row[20] is not None or row[14] is not None):
        raise ContractError("Private school cannot have public governing affiliation")
    return {
        "id": row[19], "lookup_key": key, "source_namespace": row[17], "source_id": row[18],
        "district_id": row[20], "source_build_id": build_id,
        "campus_number": row[0], "campus_number_canon": row[1],
        "name": row[2] or "", "is_charter": bool(row[3]), "is_private": bool(row[4]),
        "enrollment": row[5], "rating": row[6], "aea": _nullable_bool(row[7]), "aea_raw": row[8],
        "grade_range": row[9] or "", "lon": row[10], "lat": row[11],
        "meta": _decode_json(row[12]) if include_meta else {}, "district_name": row[13],
        "district_number": _strip_leading_apostrophe(row[14]) if row[14] is not None else None,
        "district_number_canon": row[15],
    }


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


def _district_record(row, *, include_meta: bool, build_id: str) -> dict[str, Any]:
    meta = _decode_json(row[7]) if include_meta else {}
    if include_meta or row[13] is None:
        subtype = meta.get("organization_sub_type")
        boundary_status = meta.get("district_boundary_status")
        boundary_reason = meta.get("district_boundary_reason")
    else:
        subtype, boundary_status, boundary_reason = row[10:13]
    kind = district_type_from_subtype(subtype)
    return {
        "id": row[9], "source_build_id": build_id, "name": row[0] or "",
        "district_number": _strip_leading_apostrophe(row[1] or row[2]),
        "district_number_canon": row[2] or "", "enrollment": row[3], "rating": row[4],
        "aea": _nullable_bool(row[5]), "aea_raw": row[6],
        "meta": meta, "campus_count": row[8] or 0,
        "organization_sub_type": subtype, "district_boundary_status": boundary_status,
        "district_boundary_reason": boundary_reason,
        "is_charter": None if kind == "unknown" else kind == "charter",
        "district_type": district_type_label(kind),
    }


def _nullable_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def load_campus(
    campus_number: str, *, store_path: str | Path | None = None,
    include_meta: bool = True,
) -> Optional[dict[str, Any]]:
    """Read one public or source-qualified campus through the canonical store."""
    path = discover_entity_store(store_path)
    with EntityStore(path, immutable=store_path is None) as store:
        return store.get_campus(campus_number, include_meta=include_meta)


def load_district(
    district_number: str,
    *,
    store_path: str | Path | None = None,
    include_meta: bool = True,
) -> Optional[dict[str, Any]]:
    path = discover_entity_store(store_path)
    if not path.is_file():
        raise RuntimeError(f"Failed querying district from entity store '{path}': file is unavailable")
    try:
        with EntityStore(path, immutable=store_path is None) as store:
            return store.get_district(district_number, include_meta=include_meta)
    except sqlite3.Error as error:
        raise RuntimeError(f"Failed querying district from entity store '{path}': {error}") from error


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
        with closing(readonly_connection(path, immutable=store_path is None)) as conn:
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
        catalog = read_sqlite_catalog_connection(self.connection)
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(campuses)")}
        if not {"lookup_key", "source_namespace", "source_id"}.issubset(columns):
            raise ContractError("Legacy entity store campus identity schema requires a source-bound rebuild")
        return catalog

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
        key = normalize_campus_lookup_key(campus_number)
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
        try:
            key = canonical_source_identifier(district_number, width=6)
        except ValueError:
            return None
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
            return
        try:
            keys: set[str] = set()
            for row in conn.execute(_campus_select(include_meta=include_meta)):
                record = _campus_record(row, include_meta=include_meta, build_id=self.catalog.build_id)
                key = record["lookup_key"]
                if key in keys:
                    raise ContractError(f"Entity store contains duplicate campus lookup key {key!r}")
                keys.add(key)
                yield record
        except sqlite3.OperationalError as error:
            raise ContractError("Entity store requires the source-qualified campus identity schema; rebuild it") from error

    def iter_districts(self, *, include_meta: bool = True) -> Iterable[dict[str, Any]]:
        conn = self._connect()
        if conn is None:
            return []
        query = _district_select(include_meta=include_meta, has_projection=self._has_district_projection)
        rows = conn.execute(query)
        _ = self._validated_district_numbers
        for row in rows:
            yield _district_record(row, include_meta=include_meta, build_id=self.catalog.build_id)

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

    def _load_campus_from_conn(self, key: str, *, include_meta: bool = True) -> Optional[dict[str, Any]]:
        conn = self._connect()
        if conn is None:
            return None
        try:
            cursor = conn.execute(_campus_select(include_meta=include_meta) + " WHERE c.lookup_key = ?", (key,))
            row = _unique_row(cursor, "campus lookup key", key)
        except sqlite3.OperationalError as error:
            raise ContractError("Entity store requires the source-qualified campus identity schema; rebuild it") from error
        return _campus_record(row, include_meta=include_meta, build_id=self.catalog.build_id) if row else None

    def _load_district_from_conn(
        self, key: str, *, include_meta: bool = True
    ) -> Optional[dict[str, Any]]:
        conn = self._connect()
        if conn is None:
            return None
        _ = self._validated_district_numbers
        query = _district_select(include_meta=include_meta, has_projection=self._has_district_projection, lookup=True)
        keys = (key, key[1:])
        row = _unique_row(conn.execute(query, (*keys, *keys)), "district number", key)
        if not row:
            return None
        return _district_record(row, include_meta=include_meta, build_id=self.catalog.build_id)

    @cached_property
    def _has_district_projection(self) -> bool:
        return self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'district_source_projection'"
        ).fetchone() is not None

    @cached_property
    def _validated_district_numbers(self) -> None:
        """Check stored spelling and uniqueness before the first district read."""
        try:
            validate_stored_district_numbers(self.connection.execute(
                "SELECT district_number, district_number_canon FROM districts"
            ))
        except EntityIdentityError as error:
            raise ContractError(f"Invalid entity store: {error}") from error
