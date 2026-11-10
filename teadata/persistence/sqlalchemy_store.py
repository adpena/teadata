"""SQLAlchemy persistence helpers for :mod:`teadata` entities.

The goal of this module is to bridge the in-memory :class:`~teadata.classes.DataEngine`
container with a relational database so that large snapshots can be stored in a
PostgreSQL or SQLite database and queried without loading a complete snapshot.

The design favors round-tripping the canonical :class:`~teadata.classes.District`
and :class:`~teadata.classes.Campus` objects without changing their public API.
A few guiding principles:

* Preserve the canonical identifiers (UUID primary keys and TEA campus/district
  numbers) as indexed columns for fast lookup.
* Keep enrichment payloads in a JSON column so downstream systems retain access
  to dynamic attributes exposed via ``obj.meta``.
* Store campus-to-campus transfer edges in a dedicated association table so the
  :class:`~teadata.classes.DataEngine` graph APIs keep functioning after
  hydration from SQL.
* Capture geometry as WKB and GeoJSON and hydrate it with the library's existing
  Shapely dependency; callers can omit geometry when it is unnecessary.

The module is intentionally lightweight and does **not** require a Django
settings module.  It can therefore be reused inside a Django project (via
``django.db.backends.postgresql`` + SQLAlchemy session management) or any other
Python application that can provide a SQLAlchemy engine/Session.
"""

from __future__ import annotations

from collections import OrderedDict, UserDict, defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal
from functools import partial
from itertools import chain, islice
import json
import math
from struct import pack
import uuid
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    MutableMapping,
    TypeVar,
    cast,
)

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Text,
    Table,
    Uuid,
    and_,
    case,
    cast as sql_cast,
    create_engine as _sa_create_engine,
    column as sql_column,
    delete,
    func,
    inspect,
    insert,
    literal,
    or_,
    select,
    text,
    table as sql_table,
)
from sqlalchemy import Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import OperationalError
from sqlalchemy.schema import CreateTable
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    MappedColumn,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from sqlalchemy import JSON as _BaseJSON

from ..classes import (
    Campus,
    DataEngine,
    District,
    _coerce_grade_spans,
    _spans_to_bounds,
)
from ..identifiers import (
    canonical_source_identifier,
    stored_district_number,
    validate_stored_district_numbers,
)
from ..manifest import (
    CATALOG_TABLE,
    CATALOG_TABLE_DDL,
    ContractError,
    DatasetCatalog,
    _SQLMetadataOrigin,
    _entity_sql_metadata_origin,
    canonical_json,
)

from shapely import wkb as _shapely_wkb
from shapely.geometry import shape as _shapely_shape
from shapely.geometry.base import BaseGeometry as _BaseGeometry
from shapely.geometry import mapping as _shapely_mapping

# PostgreSQL JSONB normalizes signed zero. JSON retains the source document
# just as SQLite's JSON text does; denormalized lookup columns remain separate.
_JSONType = _BaseJSON()
_CATALOG_RELATION = sql_table(
    CATALOG_TABLE,
    sql_column("singleton", Integer),
    sql_column("build_id", Text),
    sql_column("document", Text),
)

# ---------------------------------------------------------------------------
# SQLAlchemy ORM models
# ---------------------------------------------------------------------------

_naming_convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=_naming_convention)


def _uuid_column(*args, **kwargs) -> MappedColumn[uuid.UUID]:
    """Use SQLAlchemy's native UUID/portable CHAR(32) implementation."""
    return mapped_column(Uuid[uuid.UUID](), *args, **kwargs)


class DistrictRecord(Base):
    __tablename__ = "districts"

    id: Mapped[uuid.UUID] = _uuid_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    enrollment: Mapped[int | None] = mapped_column(Integer)
    district_number: Mapped[str | None] = mapped_column(String(16), index=True)
    district_number_canon: Mapped[str | None] = mapped_column(String(16), unique=True, index=True)
    aea: Mapped[bool | None] = mapped_column(Boolean)
    aea_raw: Mapped[str | None] = mapped_column(String(8))
    rating: Mapped[str | None] = mapped_column(Text)
    polygon_wkb: Mapped[bytes | None] = mapped_column(LargeBinary)
    polygon_geojson: Mapped[dict[str, Any] | None] = mapped_column(_JSONType)
    meta: Mapped[dict[str, Any] | None] = mapped_column(_JSONType)

    campuses: Mapped[list["CampusRecord"]] = relationship(
        back_populates="district", cascade="all, delete-orphan", passive_deletes=True
    )
    meta_entries: Mapped[list["DistrictMetaRecord"]] = relationship(
        back_populates="district", cascade="all, delete-orphan", passive_deletes=True
    )


class DistrictSourceProjection(Base):
    """Sparse district fields for readers that exclude the full metadata."""

    __tablename__ = "district_source_projection"

    district_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("districts.id", ondelete="CASCADE"), primary_key=True
    )
    organization_sub_type: Mapped[str | None] = mapped_column(Text)
    district_boundary_status: Mapped[str | None] = mapped_column(String(16))
    district_boundary_reason: Mapped[str | None] = mapped_column(Text)


class CampusRecord(Base):
    __tablename__ = "campuses"

    id: Mapped[uuid.UUID] = _uuid_column(primary_key=True)
    district_id: Mapped[uuid.UUID | None] = _uuid_column(
        ForeignKey("districts.id", ondelete="SET NULL"), index=True, nullable=True
    )
    lookup_key: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    source_namespace: Mapped[str | None] = mapped_column(Text)
    source_id: Mapped[str | None] = mapped_column(Text)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    charter_type: Mapped[str | None] = mapped_column(String(64))
    is_charter: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_private: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_magnet: Mapped[str | None] = mapped_column(String(64))
    enrollment: Mapped[int | None] = mapped_column(Integer)
    rating: Mapped[str | None] = mapped_column(Text)
    aea: Mapped[bool | None] = mapped_column(Boolean)
    aea_raw: Mapped[str | None] = mapped_column(String(8))
    grade_range: Mapped[str | None] = mapped_column(String(64))
    grade_range_low_code: Mapped[int | None] = mapped_column(Integer, index=True)
    grade_range_high_code: Mapped[int | None] = mapped_column(Integer, index=True)
    grade_range_code_spans: Mapped[list[list[int | None]] | None] = mapped_column(
        _JSONType
    )
    school_type: Mapped[str | None] = mapped_column(String(64))
    school_status_date: Mapped[date | None] = mapped_column(Date)
    update_date: Mapped[date | None] = mapped_column(Date)
    district_number: Mapped[str | None] = mapped_column(String(16), index=True)
    district_number_canon: Mapped[str | None] = mapped_column(String(16), index=True)
    campus_number: Mapped[str | None] = mapped_column(String(16), index=True)
    campus_number_canon: Mapped[str | None] = mapped_column(String(16), index=True)
    point_wkb: Mapped[bytes | None] = mapped_column(LargeBinary)
    point_geojson: Mapped[dict[str, Any] | None] = mapped_column(_JSONType)
    lon: Mapped[float | None] = mapped_column(Float)
    lat: Mapped[float | None] = mapped_column(Float)
    meta: Mapped[dict[str, Any] | None] = mapped_column(_JSONType)

    district: Mapped[DistrictRecord | None] = relationship(back_populates="campuses")
    meta_entries: Mapped[list["CampusMetaRecord"]] = relationship(
        back_populates="campus", cascade="all, delete-orphan", passive_deletes=True
    )

    transfers_out: Mapped[list["CampusTransferRecord"]] = relationship(
        back_populates="source",
        cascade="all, delete-orphan",
        foreign_keys=lambda: [CampusTransferRecord.source_id],
    )
    transfers_in: Mapped[list["CampusTransferRecord"]] = relationship(
        back_populates="destination",
        cascade="all, delete-orphan",
        foreign_keys=lambda: [CampusTransferRecord.destination_id],
    )


class CampusTransferRecord(Base):
    __tablename__ = "campus_transfers"

    source_id: Mapped[uuid.UUID] = _uuid_column(
        ForeignKey("campuses.id", ondelete="CASCADE"), primary_key=True, index=True
    )
    destination_id: Mapped[uuid.UUID] = _uuid_column(
        ForeignKey("campuses.id", ondelete="CASCADE"), primary_key=True, index=True
    )
    student_count: Mapped[int | None] = mapped_column(Integer)
    masked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    source: Mapped[CampusRecord] = relationship(
        back_populates="transfers_out", foreign_keys=[source_id]
    )
    destination: Mapped[CampusRecord] = relationship(
        back_populates="transfers_in", foreign_keys=[destination_id]
    )


class DistrictMetaRecord(Base):
    __tablename__ = "district_meta"

    district_id: Mapped[uuid.UUID] = _uuid_column(
        ForeignKey("districts.id", ondelete="CASCADE"), primary_key=True, index=True
    )
    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, primary_key=True, default=0)
    value_type: Mapped[str | None] = mapped_column(String(32))
    value_text: Mapped[str | None] = mapped_column(Text)
    value_numeric: Mapped[float | None] = mapped_column(Float)
    value_bool: Mapped[bool | None] = mapped_column(Boolean)
    value_datetime: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    value_json: Mapped[Any | None] = mapped_column(_JSONType)

    district: Mapped[DistrictRecord] = relationship(back_populates="meta_entries")

    __table_args__ = (
        Index("ix_district_meta_key", "key"),
        Index("ix_district_meta_text", "key", "value_text"),
        Index("ix_district_meta_numeric", "key", "value_numeric"),
        Index("ix_district_meta_datetime", "key", "value_datetime"),
    )


class CampusMetaRecord(Base):
    __tablename__ = "campus_meta"

    campus_id: Mapped[uuid.UUID] = _uuid_column(
        ForeignKey("campuses.id", ondelete="CASCADE"), primary_key=True, index=True
    )
    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, primary_key=True, default=0)
    value_type: Mapped[str | None] = mapped_column(String(32))
    value_text: Mapped[str | None] = mapped_column(Text)
    value_numeric: Mapped[float | None] = mapped_column(Float)
    value_bool: Mapped[bool | None] = mapped_column(Boolean)
    value_datetime: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    value_json: Mapped[Any | None] = mapped_column(_JSONType)

    campus: Mapped[CampusRecord] = relationship(back_populates="meta_entries")

    __table_args__ = (
        Index("ix_campus_meta_key", "key"),
        Index("ix_campus_meta_text", "key", "value_text"),
        Index("ix_campus_meta_numeric", "key", "value_numeric"),
        Index("ix_campus_meta_datetime", "key", "value_datetime"),
    )


# ---------------------------------------------------------------------------
# Engine / session helpers
# ---------------------------------------------------------------------------


def create_engine(url: str, *, echo: bool = False, **kwargs):
    """Wrapper around :func:`sqlalchemy.create_engine` for convenience."""

    return _sa_create_engine(url, echo=echo, **kwargs)


def create_sqlite_memory_engine(echo: bool = False):
    """Quick helper for unit tests or experimenting without PostgreSQL."""

    return create_engine("sqlite+pysqlite:///:memory:", echo=echo)


def create_sessionmaker(engine, *, expire_on_commit: bool = False, **kwargs):
    """Return a configured ``sessionmaker`` factory for the given engine."""

    return sessionmaker(
        bind=engine, expire_on_commit=expire_on_commit, class_=Session, **kwargs
    )


def ensure_schema(engine) -> None:
    """Create the database schema if it does not already exist."""

    if engine.dialect.name != "sqlite":
        with engine.connect() as connection:
            _validate_storage(connection)
        Base.metadata.create_all(engine)
        with engine.begin() as connection:
            connection.exec_driver_sql(CATALOG_TABLE_DDL)
        return
    # SQLAlchemy stores indexes in a set of identity-hashed objects. Its default
    # creation order changes between processes, changing SQLite root pages and
    # release checksums even when every row is identical.
    with engine.begin() as connection:
        _validate_storage(connection)
        for table in Base.metadata.sorted_tables:
            if not inspect(connection).has_table(table.name, schema=table.schema):
                connection.execute(CreateTable(table))
                for index in sorted(table.indexes, key=lambda item: str(item.name)):
                    index.create(connection)
        connection.exec_driver_sql(CATALOG_TABLE_DDL)


def _validate_storage(connection) -> None:
    """Refuse legacy lossy physical types before reading or writing data."""
    inspector = inspect(connection)
    if inspector.has_table(CATALOG_TABLE):
        columns = {
            column["name"]: column for column in inspector.get_columns(CATALOG_TABLE)
        }
        build_column = columns.get("build_id")
        if (
            build_column is None
            or not isinstance(build_column["type"], String)
            or build_column["nullable"]
        ):
            raise ContractError(
                "Legacy database catalog lacks its bounded build ID fence; recreate and refill from source"
            )
    for table in Base.metadata.sorted_tables:
        if not inspector.has_table(table.name):
            continue
        uuid_names = {
            column.name for column in table.columns if isinstance(column.type, Uuid)
        }
        for column in inspector.get_columns(table.name):
            if connection.dialect.name == "sqlite" and column["name"] in uuid_names:
                storage = column["type"]
                if not isinstance(storage, String) or storage.length != 32:
                    raise ContractError(
                        "Legacy SQLite UUID storage is unsafe; rebuild this database "
                        "from its catalog-bound source. Numeric coercion cannot be repaired reliably."
                    )
            if (
                column["name"] in table.c
                and isinstance(table.c[column["name"]].type, _BaseJSON)
                and isinstance(column["type"], JSONB)
            ):
                raise ContractError(
                    "Legacy JSONB storage normalizes source values; rebuild this database "
                    "from its catalog-bound source using JSON columns."
                )
        if table.name == "campuses":
            names = {column["name"] for column in inspector.get_columns(table.name)}
            if not {"lookup_key", "source_namespace", "source_id"}.issubset(names):
                raise ContractError("Legacy campus identity schema requires a source-bound rebuild")


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _clean_meta(meta: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(meta, Mapping):
        return None
    if isinstance(meta, LazyMetaDict):
        meta = meta._materialize()

    def _coerce(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, Mapping):
            return {str(k): _coerce(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_coerce(v) for v in value]
        try:
            json.dumps(value)
            return value
        except TypeError:
            return repr(value)

    return {str(k): _coerce(v) for k, v in meta.items()}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"y", "yes", "true", "t", "1"}:
        return True
    if text in {"n", "no", "false", "f", "0"}:
        return False
    return bool(text)


def _normalize_aea_raw(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "Y" if value else "N"
    if isinstance(value, (int, float)):
        if value != value:  # NaN
            return None
        return "Y" if bool(value) else "N"
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"y", "yes", "true", "t", "1"}:
        return "Y"
    if lowered in {"n", "no", "false", "f", "0"}:
        return "N"
    return None


def _coerce_nullable_bool(value: Any) -> bool | None:
    raw = _normalize_aea_raw(value)
    if raw == "Y":
        return True
    if raw == "N":
        return False
    return None


def _flatten_meta(meta: Mapping[str, Any] | None) -> list[tuple[str, int, Any]]:
    if not isinstance(meta, Mapping):
        return []

    collected: dict[str, list[Any]] = defaultdict(list)

    def _walk(path: Iterable[str], value: Any) -> None:
        if isinstance(value, Mapping):
            for k, v in value.items():
                _walk((*path, str(k)), v)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                _walk(path, item)
            return
        key = ".".join(path)
        collected[key].append(value)

    for k, v in meta.items():
        _walk((str(k),), v)

    flattened: list[tuple[str, int, Any]] = []
    for key, values in collected.items():
        for idx, value in enumerate(values):
            flattened.append((key, idx, value))
    return flattened


_UNVERIFIABLE_META = object()


def _meta_value_token(value: Any) -> Any:
    kind = type(value)
    if kind in (str, int, bool) or value is None:
        return kind, value
    if kind is float:
        return pack("!d", value)
    if kind is Decimal:
        return Decimal, value.as_tuple()
    if kind is list:
        return list, tuple(_meta_value_token(item) for item in value)
    if kind in (dict, OrderedDict):
        return kind, tuple((_meta_value_token(key), _meta_value_token(item)) for key, item in value.items())
    raise TypeError("Value is outside the raw JSON contract")


def _watched_value_token(value: Any) -> Any:
    try:
        return _meta_value_token(value)
    except (TypeError, RecursionError):
        return _UNVERIFIABLE_META


class LazyMetaDict(UserDict[str, Any]):
    """A standard mutable mapping with one source-bound lazy data boundary.

    Supported SQL point reads and writes can stay deferred. Whole-mapping
    operations load the complete current payload and apply local edits.
    """

    def __init__(
        self,
        loader: Callable[[], Mapping[str, Any] | None]
        | Mapping[str, Any]
        | None = None,
        *,
        seed: Mapping[str, Any] | None = None,
        read_key: Callable[[str], Mapping[str, Any] | None] | None = None,
        read_keys: Callable[[], Iterable[str]] | None = None,
        _sql_origin: _SQLMetadataOrigin | None = None,
        _validate_write: Callable[[], Any] | None = None,
    ):
        self._sql_origin = _sql_origin
        self._replaced = False
        self._edited: set[str] | None = None
        self._watched: dict[str, Any] | None = None
        self._exposed_keys: frozenset[str] | None = None
        self._read_key = read_key
        self._read_keys = read_keys
        self._validate_write = _validate_write
        self._point_keys: set[str] | None = None
        self._overrides: dict[str, Any] | None = None
        self._removed: set[str] | None = None
        if callable(loader):
            self._loader: Callable[[], Mapping[str, Any] | None] | None = loader
            self._loaded = False
            self._payload = dict(seed or {})
        else:
            self._loader = None
            self._loaded = True
            self._payload = dict(loader or {})

    def _materialize(self) -> dict[str, Any]:
        if not self._loaded:
            assert self._loader is not None
            changed = self._edited_keys() if self._watched is not None else ()
            # Replace, rather than merge: a prefetched key may have been deleted.
            payload = dict(self._loader() or {})
            observed = self._point_keys or ()
            for key in observed:
                if key in payload:
                    # Preserve the object already returned by an explicit point read.
                    payload[key] = self._payload[key]
            if changed:
                payload.update((key, value) for key, value in self._payload.items() if key in changed)
            if self._watched is not None:
                for key in tuple(self._watched):
                    if key not in payload:
                        del self._watched[key]
                    elif key not in observed and key not in changed:
                        # Refresh stale seeds without detaching an identical live alias.
                        token = _watched_value_token(payload[key])
                        if token == self._watched[key] and key in self._payload:
                            payload[key] = self._payload[key]
                        else:
                            self._watched[key] = token
            for key in self._removed or ():
                payload.pop(key, None)
            if self._overrides is not None:
                payload.update(self._overrides)
            self._set_materialized(payload)
        return self._payload

    def _set_materialized(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self._loaded = True
        self._loader = None
        self._read_key = None
        self._read_keys = None
        self._validate_write = None
        self._point_keys = None
        self._overrides = None
        self._removed = None

    @property
    def data(self) -> dict[str, Any]:
        payload = self._materialize()
        if self._sql_origin is not None and self._exposed_keys is None:
            self._exposed_keys = frozenset(payload)
            for key, value in payload.items():
                self._watch(key, value)
        return payload

    @data.setter
    def data(self, payload: dict[str, Any]) -> None:
        if self._loaded and payload is self._payload:
            return
        if self._sql_origin is not None:
            self._replaced = True
        self._set_materialized(payload)

    def __getitem__(self, key: str) -> Any:
        if self._overrides is not None and key in self._overrides:
            return self._overrides[key]
        if self._removed is not None and key in self._removed:
            raise KeyError(key)
        if key in self._payload:
            value = self._payload[key]
        elif self._read_key is not None and isinstance(key, str):
            values = self._read_key(key)
            if values is None or key not in values:
                raise KeyError(key)
            value = self._payload[key] = values[key]
            if self._point_keys is None:
                self._point_keys = set()
            self._point_keys.add(key)
        else:
            value = self._materialize()[key]
        if self._sql_origin is not None:
            self._watch_mutable(key, value)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        if self._validate_write is not None and type(key) is str and _point_meta_key(key):
            if self._overrides is None:
                self._validate_write()
                self._overrides = {}
            self._overrides[key] = value
            self._discard_source_key(key)
        else:
            self._materialize()[key] = value
        self._record_edits((key,))

    def __delitem__(self, key: str) -> None:
        if self._validate_write is not None and self._read_key is not None and type(key) is str and _point_meta_key(key):
            if self._overrides is not None and key in self._overrides:
                del self._overrides[key]
            else:
                if self._removed is not None and key in self._removed:
                    raise KeyError(key)
                values = self._read_key(key)
                if (values is None or key not in values) and not (key in self._payload and self._was_edited(key)):
                    raise KeyError(key)
                if self._overrides is None:
                    self._overrides = {}
            if self._removed is None:
                self._removed = set()
            self._removed.add(key)
            self._discard_source_key(key)
        else:
            del self._materialize()[key]
        self._record_edits((key,))

    def _discard_source_key(self, key: str) -> None:
        self._payload.pop(key, None)
        if self._point_keys is not None:
            self._point_keys.discard(key)
        if self._watched is not None:
            self._watched.pop(key, None)

    def __iter__(self) -> Iterator[str]:
        return iter(self._materialize())

    def __len__(self) -> int:
        return len(self._materialize())

    def __repr__(self) -> str:
        return repr(self._materialize())

    def __contains__(self, key: object) -> bool:
        if self._overrides is not None and key in self._overrides:
            return True
        if self._removed is not None and key in self._removed:
            return False
        if key in self._payload:
            return True
        if self._read_key is None or not isinstance(key, str):
            return key in self._materialize()
        try:
            self[key]
        except KeyError:
            return False
        return True

    def iter_keys(self) -> Iterator[str]:
        """List exact top-level string names in lexical order, without value hydration when supported."""
        names = tuple(self._read_keys() if not self._loaded and self._read_keys is not None else self._materialize())
        if not all(isinstance(name, str) for name in names):
            raise TypeError("iter_keys() requires string metadata keys. Use ordinary mapping iteration for mixed keys.")
        if self._loaded or self._overrides is None and self._watched is None:
            return iter(sorted(names))
        current = set(names)
        if self._watched is not None:
            # Full hydration also retains edited aliases removed from the live row.
            current.update(key for key in self._watched
                           if key not in current and key in self._payload and self._was_edited(key))
        current.difference_update(self._removed or ())
        current.update(self._overrides or ())
        return iter(sorted(current))

    def pop(self, key: str, *default) -> Any:
        payload = self._materialize()
        present = key in payload
        result = payload.pop(key, *default)
        if present:
            self._record_edits((key,))
        return result

    def popitem(self) -> tuple[str, Any]:
        key, value = self._materialize().popitem()
        self._record_edits((key,))
        return key, value

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in self._materialize():
            self[key] = default
        return self[key]

    def clear(self) -> None:
        payload = self._materialize()
        self._record_edits(payload)
        payload.clear()

    def __or__(self, other):
        if not isinstance(other, (dict, UserDict)):
            return NotImplemented
        result = self.copy()
        result.update(other)
        return result

    def __ror__(self, other):
        if not isinstance(other, (dict, UserDict)):
            return NotImplemented
        result = self.copy()
        supplied = dict(other)
        result._record_edits(supplied.keys() - result._payload.keys())
        supplied.update(result._payload)
        result._payload = supplied
        return result

    def __ior__(self, other):
        if other is not self:
            self.update(other)
        return self

    def _record_edits(self, keys: Iterable[str]) -> None:
        if self._sql_origin is not None:
            if self._edited is None:
                self._edited = set(keys)
            else:
                self._edited.update(keys)

    def _watch(self, key: str, value: Any) -> None:
        if self._edited is not None and key in self._edited:
            return
        if self._watched is None:
            self._watched = {}
        if key not in self._watched:
            self._watched[key] = _watched_value_token(value)

    def _watch_mutable(self, key: str, value: Any) -> None:
        if type(value) not in (str, int, bool, float, type(None)):
            self._watch(key, value)

    def _was_edited(self, key: str) -> bool:
        if self._replaced or self._edited is not None and key in self._edited:
            return True
        if self._watched is not None and key in self._watched:
            original = self._watched[key]
            changed = (original is _UNVERIFIABLE_META or key not in self._payload
                       or _watched_value_token(self._payload[key]) != original)
        else:
            changed = (self._exposed_keys is not None and key not in self._exposed_keys
                       and key in self._payload)
        if changed:
            self._record_edits((key,))
        return changed

    def _edited_keys(self) -> set[str]:
        changed = set(self._edited or ())
        if self._watched is not None:
            changed.update(key for key in self._watched if self._was_edited(key))
        if self._exposed_keys is not None:
            changed.update(self._payload.keys() - self._exposed_keys)
        if changed:
            self._record_edits(changed)
        return changed

    def _copy_origin(self, copied: LazyMetaDict) -> LazyMetaDict:
        if self._sql_origin is not None:
            copied._sql_origin = self._sql_origin
            for key, value in self._payload.items():
                self._watch_mutable(key, value)
            if self._point_keys is not None:
                copied._point_keys = self._point_keys.copy()
            copied._replaced = self._replaced
            copied._edited = self._edited_keys()
            copied._watched = self._watched.copy() if self._watched is not None else None
        return copied

    def copy(self) -> LazyMetaDict:
        payload = self._materialize()
        return self._copy_origin(type(self)(payload))

    def __copy__(self):
        # A public whole-mapping copy has the same authoritative view as dict().
        return self.copy()

    def _copy_deferred(self):
        """Give an engine union independent metadata without preloading rows."""
        if self._loader is None:
            return self.copy()
        copied = type(self)(self._loader, seed=self._payload, read_key=self._read_key)
        copied._read_keys = self._read_keys
        copied._validate_write = self._validate_write
        if self._overrides is not None:
            copied._overrides = self._overrides.copy()
        if self._removed is not None:
            copied._removed = self._removed.copy()
        if self._sql_origin is None and self._point_keys is not None:
            copied._point_keys = self._point_keys.copy()
        return self._copy_origin(copied)

    def __reduce__(self):
        # A persisted graph must contain values, never Session/loader closures.
        payload = self._materialize()
        if self._sql_origin is None:
            return type(self), (payload,)
        return type(self), (payload,), (self._sql_origin, self._edited_keys(), self._replaced)

    def __setstate__(self, state: tuple[_SQLMetadataOrigin, set[str], bool]) -> None:
        self._sql_origin, self._edited, self._replaced = state
        for key, value in self._payload.items():
            self._watch_mutable(key, value)


def _default_postgresql_json(connection) -> bool:
    if connection.dialect.driver != "psycopg":
        return False
    from psycopg import Cursor, ServerCursor, _cmodule
    from psycopg.pq import Format
    from psycopg.types.json import JsonBinaryLoader, JsonLoader
    from psycopg.types.string import TextBinaryLoader, TextLoader

    driver = connection.connection.driver_connection
    adapters = driver.adapters
    json_oid = adapters.types["json"].oid
    native = _cmodule._psycopg
    # The CASE result can be varchar even when the JSON token is cast to text.
    text_loaders = ((Format.TEXT, TextLoader, getattr(native, "TextLoader", None)),
                    (Format.BINARY, TextBinaryLoader, getattr(native, "TextBinaryLoader", None)))
    return (driver.info.parameter_status("server_encoding") == "UTF8"
            and driver.info.parameter_status("client_encoding") == "UTF8"
            and driver.cursor_factory is Cursor
            and driver.server_cursor_factory is ServerCursor
            and adapters.get_loader(json_oid, Format.TEXT) is JsonLoader
            and adapters.get_loader(json_oid, Format.BINARY) is JsonBinaryLoader
            and getattr(JsonLoader, "_loads", None) is json.loads
            and getattr(JsonBinaryLoader, "_loads", None) is json.loads
            and all(adapters.get_loader(adapters.types[name].oid, mode) in (python_loader, native_loader or python_loader)
                    for name in ("text", "varchar") for mode, python_loader, native_loader in text_loaders))


def _default_sqlite_text(connection) -> bool:
    return getattr(connection.connection.driver_connection, "text_factory", None) is str


def _point_meta_dialect(connection) -> str | None:
    dialect = connection.dialect
    if getattr(dialect, "_json_deserializer", None) is not None:
        return None
    if dialect.name == "postgresql" and _default_postgresql_json(connection):
        return "postgresql"
    if (dialect.name == "sqlite" and (dialect.server_version_info or ()) >= (3, 38, 0)
            and _default_sqlite_text(connection)):
        try:
            connection.exec_driver_sql("SELECT json_valid('{}'), '{}' -> '$.x' FROM json_each('{}')").close()
        except OperationalError as error:
            if str(error.orig) not in {"no such function: json_valid", "no such function: ->", "no such table: json_each"}:
                raise
        else:
            return "sqlite"
    return None


def _point_meta_key(key: str) -> bool:
    # SQLite's path grammar is narrower than Python's JSON object key grammar.
    return isinstance(key, str) and key.isascii() and key.isidentifier()


def _unsafe_postgresql_meta(raw):
    # PostgreSQL's JSON type accepts escapes its extractors cannot decode.
    return raw.op("~")(r"\\u(?:0000|[dD][89a-fA-F][0-9a-fA-F]{2})")


def _unsafe_sqlite_meta(meta, raw):
    return or_(func.typeof(meta) != "text", func.instr(raw, func.char(0)) > 0,
               func.instr(raw, r"\u0000") > 0, func.json_valid(meta) != 1)


def _projected_meta_token(meta, key: str, dialect: str):
    """Project JSON text. SQL NULL means absent; empty text requests a whole read."""
    raw = sql_cast(meta, Text)
    if dialect == "postgresql":
        # JSON can store escapes that native extraction cannot decode even in
        # unrelated keys. Keep such documents on the typed JSON reader.
        return case(
            (_unsafe_postgresql_meta(raw), ""),
            (func.json_typeof(meta) == "object", sql_cast(meta.op("->")(literal(key, Text)), Text)),
            else_=None,
        )

    value = sql_cast(meta.op("->")(f"$.{key}"), Text)
    entries = func.json_each(meta).table_valued("key")
    count = select(func.count()).select_from(entries).where(entries.c.key.collate("BINARY") == key).scalar_subquery()
    # Some SQLite versions compare embedded NUL keys as if they ended early.
    # Duplicate keys and mismatches with exact enumeration also need Python's
    # original decoder. JSON -> text preserves big integers and signed zero.
    return case(
        (meta.is_(None), None),
        (_unsafe_sqlite_meta(meta, raw), ""),
        (func.json_type(meta) != "object", None),
        (count == 1, func.coalesce(value, "")),
        (and_(count == 0, value.is_(None)), None),
        else_="",
    )


def _meta_projection(meta, keys: tuple[str, ...], dialect: str):
    tokens = tuple(_projected_meta_token(meta, key, dialect) for key in keys)
    return (*tokens, case((or_(*(token == "" for token in tokens)), meta), else_=None))


def _meta_names_projection(meta, dialect: str):
    raw = sql_cast(meta, Text)
    if dialect == "postgresql":
        unsafe = _unsafe_postgresql_meta(raw)
        keys = func.json_object_keys(meta).table_valued("name").render_derived()
        names = select(func.json_agg(keys.c.name)).scalar_subquery()
        token = case((unsafe, ""), (func.json_typeof(meta) == "object", sql_cast(names, Text)),
                     else_=None)
    else:
        keys = func.json_each(meta).table_valued("key")
        names = select(func.json_group_array(keys.c.key)).scalar_subquery()
        # A lone surrogate can be valid JSON text but invalid UTF-8 after
        # SQLite decodes a key. The original JSON codec preserves that key.
        unsafe = or_(_unsafe_sqlite_meta(meta, raw), raw.op("GLOB")(r"*\u[dD][89a-fA-F][0-9a-fA-F][0-9a-fA-F]*"))
        token = case((meta.is_(None), None), (unsafe, ""),
                     (func.json_type(meta) == "object", sql_cast(names, Text)),
                     else_=None)
    return token, case((unsafe, meta), else_=None)


def _decode_meta_projection(keys: tuple[str, ...], values) -> Mapping[str, Any]:
    if "" in values[:-1]:
        raw = values[-1]
        return {key: raw[key] for key in keys if key in raw} if isinstance(raw, Mapping) else {}
    return {key: json.loads(token) for key, token in zip(keys, values[:-1], strict=True) if token is not None}


def _lazy_meta(
    default_session: Session,
    session_factory: Callable[[], Session] | None,
    record_cls: type[DistrictRecord] | type[CampusRecord],
    pk_value: uuid.UUID,
    build_id: str,
    dialect: str | None,
    seed: Mapping[str, Any] | None,
    origin: _SQLMetadataOrigin,
) -> LazyMetaDict:
    projection_bind = default_session.get_bind() if dialect is not None else None

    def _load(key: str | None = None, *, names_only: bool = False, check_only: bool = False) -> Mapping[str, Any]:
        # Reuse an active caller transaction, including its uncommitted writes.
        # Closing a second Session on SQLite's shared in-memory connection would
        # roll that transaction back. Only factory-created Sessions are owned.
        if not default_session.in_transaction() and session_factory is not None:
            active = session_factory()
            close_after = True
        else:
            active = default_session
            close_after = False
        try:
            projected = (dialect is not None and (names_only or key is not None and _point_meta_key(key))
                         and active.get_bind(mapper=record_cls) is projection_bind)
            if projected:
                connection = active.connection(bind_arguments={"mapper": record_cls})
                projected = (_default_postgresql_json(connection) if dialect == "postgresql"
                             else _default_sqlite_text(connection))
            columns = (literal(1),) if check_only else (record_cls.meta,)
            if projected and dialect is not None:
                if names_only:
                    columns = _meta_names_projection(record_cls.meta, dialect)
                elif key is not None:
                    columns = _meta_projection(record_cls.meta, (key,), dialect)
            row = active.execute(
                select(*columns)
                .select_from(record_cls)
                .join(_CATALOG_RELATION, _CATALOG_RELATION.c.singleton == 1)
                .where(record_cls.id == pk_value, _CATALOG_RELATION.c.build_id == build_id)
            ).one_or_none()
            if row is None:
                raise ContractError(
                    "Lazy metadata source was deleted or its dataset catalog changed"
                )
            if check_only:
                return {}
            if projected and key is not None:
                return _decode_meta_projection((key,), row)
            if projected and names_only:
                if row[0] != "":
                    return dict.fromkeys(json.loads(row[0])) if row[0] is not None else {}
                raw = row[-1]
            else:
                raw = row[0]
            if not isinstance(raw, Mapping):
                return {}
            if names_only:
                return dict.fromkeys(raw)
            return raw if key is None else {key: raw[key]} if key in raw else {}
        finally:
            if close_after:
                active.close()

    return LazyMetaDict(_load, seed=seed, read_key=_load if dialect is not None else None,
                        read_keys=partial(_load, names_only=True) if dialect is not None else None,
                        _sql_origin=origin, _validate_write=partial(_load, check_only=True) if dialect is not None else None)


def _meta_value_payload(value: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"value_json": value}

    if value is None:
        payload["value_type"] = "null"
        return payload

    if isinstance(value, bool):
        payload["value_type"] = "bool"
        payload["value_bool"] = value
        return payload

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        payload["value_type"] = "number"
        try:
            numeric = float(value)
        except OverflowError:
            numeric = None
        if (
            numeric is not None
            and math.isfinite(numeric)
            and (not isinstance(value, int) or int(numeric) == value)
        ):
            payload["value_numeric"] = numeric
        return payload

    parsed_dt: datetime | None = None
    if isinstance(value, datetime):
        parsed_dt = value
        payload["value_text"] = value.isoformat()
    elif isinstance(value, date):
        parsed_dt = datetime.combine(value, datetime.min.time())
        payload["value_text"] = value.isoformat()
    elif isinstance(value, str):
        # PostgreSQL TEXT cannot index NUL; its JSON column retains the exact
        # escaped raw string, including NUL, without inventing a replacement.
        if "\x00" not in value:
            payload["value_text"] = value
        try:
            parsed_dt = datetime.fromisoformat(value)
        except ValueError:
            parsed_dt = None
        if parsed_dt is None:
            payload["value_type"] = "string"
            return payload
    else:
        payload["value_text"] = str(value)
        payload["value_type"] = "string"
        return payload

    payload["value_type"] = "datetime"
    if parsed_dt.tzinfo is not None:
        parsed_dt = parsed_dt.astimezone(timezone.utc).replace(tzinfo=None)
    payload["value_datetime"] = parsed_dt
    return payload


def _meta_record_for(entity: Literal["district", "campus"]):
    if entity == "district":
        return DistrictMetaRecord, DistrictMetaRecord.district_id
    if entity == "campus":
        return CampusMetaRecord, CampusMetaRecord.campus_id
    raise ValueError(f"Unsupported entity type: {entity}")


def _accumulate_meta_value(
    bucket: MutableMapping[str, Any], key: str, ordinal: int, value: Any
) -> None:
    existing = bucket.get(key)
    if existing is None and ordinal == 0:
        bucket[key] = value
        return
    if not isinstance(existing, list):
        sequence: list[Any] = [] if existing is None else [existing]
        bucket[key] = sequence
    else:
        sequence = existing
    while len(sequence) <= ordinal:
        sequence.append(None)
    sequence[ordinal] = value


def fetch_meta_values(
    session: Session,
    *,
    entity: Literal["district", "campus"],
    keys: Iterable[str],
    ids: Iterable[uuid.UUID] | None = None,
) -> dict[uuid.UUID, dict[str, Any]]:
    """Return a mapping of entity UUID to selected meta key/value pairs."""

    key_set = {k for k in keys if k}
    if not key_set:
        return {}

    record_cls, id_column = _meta_record_for(entity)
    stmt = select(record_cls).where(record_cls.key.in_(key_set))
    if ids is not None:
        ids = list(ids)
        if not ids:
            return {}
        stmt = stmt.where(id_column.in_(ids))

    payload: dict[uuid.UUID, dict[str, Any]] = {}
    for record in session.execute(stmt).scalars():
        parent_id = getattr(record, id_column.key)
        bucket = payload.setdefault(parent_id, {})
        _accumulate_meta_value(bucket, record.key, record.ordinal, record.value_json)

    # Compress one-element lists while preserving explicit ``None`` values
    for values in payload.values():
        for key, value in list(values.items()):
            if isinstance(value, list) and len(value) == 1:
                values[key] = value[0]

    return payload


def available_meta_keys(
    session: Session,
    *,
    entity: Literal["district", "campus"],
    with_counts: bool = False,
) -> dict[str, int] | list[str]:
    """Return distinct enrichment keys (optionally with occurrence counts)."""

    record_cls, _ = _meta_record_for(entity)
    stmt = (
        select(record_cls.key, func.count())
        .group_by(record_cls.key)
        .order_by(record_cls.key)
    )
    rows = session.execute(stmt).all()
    if with_counts:
        return {row[0]: int(row[1]) for row in rows}
    return [row[0] for row in rows]


def _dump_polygon(polygon: Any) -> tuple[bytes | None, dict[str, Any] | None]:
    if polygon is None:
        return None, None

    if isinstance(polygon, _BaseGeometry):
        try:
            return polygon.wkb, _shapely_mapping(polygon)
        except Exception:  # pragma: no cover - shapely serialization failure
            pass

    # Fallback: attempt to coerce to GeoJSON-like mapping from coordinates
    coords: list[list[float]] = []
    try:
        for pt in polygon:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                x, y = float(pt[0]), float(pt[1])
                coords.append([x, y])
    except Exception:
        return None, None

    if not coords:
        return None, None

    if coords[0] != coords[-1]:
        coords.append(coords[0])

    return None, {"type": "Polygon", "coordinates": [coords]}


def _dump_point(
    point: Any,
) -> tuple[bytes | None, dict[str, Any] | None, float | None, float | None]:
    if point is None:
        return None, None, None, None

    if isinstance(point, _BaseGeometry):
        try:
            mapping = _shapely_mapping(point)
            lon = float(mapping["coordinates"][0])
            lat = float(mapping["coordinates"][1])
            return point.wkb, mapping, lon, lat
        except Exception:  # pragma: no cover
            pass

    try:
        x, y = point
        lon = float(x)
        lat = float(y)
        return None, {"type": "Point", "coordinates": [lon, lat]}, lon, lat
    except Exception:
        return None, None, None, None


def _load_polygon(wkb_bytes: bytes | None, geojson: Mapping[str, Any] | None) -> Any:
    if wkb_bytes:
        try:
            return _shapely_wkb.loads(wkb_bytes)
        except Exception:  # pragma: no cover
            pass

    if geojson:
        try:
            return _shapely_shape(geojson)
        except Exception:  # pragma: no cover
            pass
        coords = geojson.get("coordinates")
        if isinstance(coords, list) and coords:
            try:
                ring = coords[0]
                return [(float(x), float(y)) for x, y in ring]
            except Exception:
                return None
    return None


def _load_point(
    wkb_bytes: bytes | None,
    geojson: Mapping[str, Any] | None,
    lon: float | None,
    lat: float | None,
) -> Any:
    if wkb_bytes:
        try:
            return _shapely_wkb.loads(wkb_bytes)
        except Exception:  # pragma: no cover
            pass

    if geojson:
        try:
            return _shapely_shape(geojson)
        except Exception:  # pragma: no cover
            pass
        coords = geojson.get("coordinates")
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            try:
                return (float(coords[0]), float(coords[1]))
            except Exception:
                return None

    if lon is not None and lat is not None:
        return (lon, lat)
    return None


# ---------------------------------------------------------------------------
# DataEngine <-> ORM bridge
# ---------------------------------------------------------------------------


_T = TypeVar("_T")


def _batches(values: Iterable[_T], size: int) -> Iterator[list[_T]]:
    iterator = iter(values)
    while batch := list(islice(iterator, size)):
        yield batch


def _checked_batch_size(size: int) -> int:
    if type(size) is not int or size <= 0:
        raise ValueError("batch_size must be a positive integer")
    return size


def _catalog_row(session: Session) -> tuple[str, str] | None:
    rows = session.execute(
        select(
            _CATALOG_RELATION.c.singleton,
            _CATALOG_RELATION.c.document,
            _CATALOG_RELATION.c.build_id,
        ).limit(2)
    ).all()
    if not rows:
        return None
    if len(rows) != 1 or rows[0][0] != 1:
        raise ContractError(
            "Database must contain exactly one dataset catalog at singleton 1"
        )
    return rows[0][1], rows[0][2]


def _index_key(key: str) -> str:
    try:
        key.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(
            "Metadata index keys must be valid UTF-8; use include_meta_entries=False to retain raw JSON only"
        ) from error
    if "\x00" in key or len(key) > 255:
        raise ValueError(
            "Metadata index keys must contain at most 255 characters and no NUL; "
            "use include_meta_entries=False to retain raw JSON only"
        )
    return key


def _bind_catalog(repo: DataEngine, session: Session, *, replace: bool) -> None:
    catalog = getattr(repo, "catalog", None)
    if not isinstance(catalog, DatasetCatalog):
        raise ContractError("Database exports require a catalog-bound DataEngine")
    document = canonical_json(DatasetCatalog(catalog.to_dict()).to_dict())
    previous = _catalog_row(session)
    if previous is None:
        occupied = select(
            *(
                select(1).select_from(table).limit(1).exists()
                for table in Base.metadata.sorted_tables
            )
        )
        if not replace and any(session.execute(occupied).one()):
            raise ContractError(
                "An unbound populated database requires explicit replace=True"
            )
        session.execute(
            text(
                f"INSERT INTO {CATALOG_TABLE} (singleton,build_id,document) VALUES (1,:build_id,:document)"
            ),
            {"build_id": catalog.build_id, "document": document},
        )
        return
    existing = DatasetCatalog(json.loads(previous[0]), expected_build_id=previous[1])
    if existing.build_id != catalog.build_id and not replace:
        raise ContractError(
            "A partial export cannot mix dataset catalogs; use explicit replace=True"
        )
    # Acquire the destination's write lock before entity DML. The predicate also
    # refuses a catalog changed by a concurrent writer while this call waited.
    result = session.connection().execute(
        text(
            f"UPDATE {CATALOG_TABLE} SET document=:document,build_id=:build_id WHERE singleton=1 AND build_id=:previous"
        ),
        {
            "document": document,
            "build_id": catalog.build_id,
            "previous": existing.build_id,
        },
    )
    if result.rowcount != 1:
        raise ContractError(
            "The destination catalog changed concurrently; roll back and retry"
        )


def _write_rows(
    session: Session,
    table,
    rows: Iterable[dict[str, Any]],
    *,
    replace: bool,
    batch_size: int,
) -> None:
    if replace:
        statement = insert(table)
    else:
        dialect = session.get_bind().dialect.name
        factory = {"sqlite": sqlite_insert, "postgresql": pg_insert}.get(dialect)
        if factory is None:
            raise NotImplementedError(
                "Partial database exports support SQLite and PostgreSQL"
            )
        statement = factory(table)
        statement = statement.on_conflict_do_update(
            index_elements=list(table.primary_key.columns),
            set_={
                column.name: statement.excluded[column.name]
                for column in table.columns
                if not column.primary_key
            },
        )
    connection = session.connection()
    for batch in _batches(rows, batch_size):
        connection.execute(statement, batch)


def _district_payload(district: District, *, include_geometry: bool) -> dict[str, Any]:
    aea_raw = _normalize_aea_raw(getattr(district, "aea_raw", district.aea))
    wkb, geojson = (
        _dump_polygon(getattr(district, "polygon", None))
        if include_geometry
        else (None, None)
    )
    return {
        "id": district.id,
        "name": district.name,
        "enrollment": district.enrollment,
        "district_number": district.district_number or None,
        "district_number_canon": district.district_number_canon or None,
        "aea_raw": aea_raw,
        "aea": _coerce_nullable_bool(aea_raw if aea_raw is not None else district.aea),
        "rating": district.rating,
        "polygon_wkb": wkb,
        "polygon_geojson": geojson,
        "meta": _clean_meta(district.meta),
    }


def _campus_payload(campus: Campus, *, include_geometry: bool) -> dict[str, Any]:
    lookup_key = campus.lookup_key
    if not lookup_key:
        raise ContractError("Released campuses require a canonical lookup identity")
    aea_raw = _normalize_aea_raw(getattr(campus, "aea_raw", campus.aea))
    wkb, geojson, lon, lat = _dump_point(getattr(campus, "point", None))
    spans = [[low, high] for low, high in campus.grade_range_code_spans]
    return {
        "id": campus.id,
        "lookup_key": lookup_key,
        "source_namespace": campus.source_namespace,
        "source_id": campus.source_id,
        "district_id": campus.district_id,
        "name": campus.name,
        "charter_type": campus.charter_type,
        "is_charter": _coerce_bool(campus.is_charter),
        "is_private": _coerce_bool(campus.is_private),
        "is_magnet": campus.is_magnet,
        "enrollment": campus.enrollment,
        "rating": campus.rating,
        "aea_raw": aea_raw,
        "aea": _coerce_nullable_bool(aea_raw if aea_raw is not None else campus.aea),
        "grade_range": campus.grade_range,
        "grade_range_low_code": campus.grade_range_low_code,
        "grade_range_high_code": campus.grade_range_high_code,
        "grade_range_code_spans": spans or None,
        "school_type": campus.school_type,
        "school_status_date": campus.school_status_date,
        "update_date": campus.update_date,
        "district_number": campus.district_number,
        "district_number_canon": canonical_source_identifier(campus.district_number, width=6),
        "campus_number": campus.campus_number,
        "campus_number_canon": None if campus.is_private else "'" + lookup_key,
        "point_wkb": wkb if include_geometry else None,
        "point_geojson": geojson if include_geometry else None,
        "lon": lon,
        "lat": lat,
        "meta": _clean_meta(campus.meta),
    }


def _write_entities(
    session: Session,
    model,
    meta_model,
    rows: Iterable[dict[str, Any]],
    *,
    replace: bool,
    include_meta_entries: bool,
    batch_size: int,
) -> None:
    parent_column = (
        meta_model.__table__.c.district_id
        if model is DistrictRecord
        else meta_model.__table__.c.campus_id
    )
    for batch in _batches(rows, batch_size):
        _write_rows(
            session, model.__table__, batch, replace=replace, batch_size=batch_size
        )
        if not replace:
            session.connection().execute(
                delete(meta_model.__table__).where(
                    parent_column.in_([row["id"] for row in batch])
                )
            )
        if include_meta_entries:
            entries = (
                {
                    parent_column.name: row["id"],
                    "key": _index_key(key),
                    "ordinal": ordinal,
                    **_meta_value_payload(value),
                }
                for row in batch
                for key, ordinal, value in _flatten_meta(row["meta"])
            )
            # Fill every physical column: Core executemany uses one stable
            # parameter shape, preserving explicit nulls across mixed values.
            complete = (
                {
                    column.name: entry.get(column.name)
                    for column in meta_model.__table__.columns
                }
                for entry in entries
            )
            _write_rows(
                session,
                meta_model.__table__,
                complete,
                replace=True,
                batch_size=batch_size,
            )


def export_dataengine(
    repo: DataEngine,
    session: Session,
    *,
    replace: bool = False,
    include_meta_entries: bool = True,
    include_geometry: bool = True,
    batch_size: int = 256,
) -> None:
    """Persist a catalog-bound graph in the caller's transaction.

    The default upserts only supplied entities and transfer edges; unrelated
    rows remain. ``replace=True`` explicitly replaces the entire destination.
    One destination contains one dataset catalog; partial writes from another
    release are refused. The caller must roll back on any failure and owns commit.
    SQLAlchemy Core batches avoid ORM hydration, per-entity queries, and loading
    existing transfer history. ``batch_size`` bounds additional row buffers.
    """
    batch_size = _checked_batch_size(batch_size)
    if type(replace) is not bool:
        raise TypeError("replace must be an explicit boolean")
    for entity in chain(repo._districts.values(), repo._campuses.values()):
        meta = entity.meta
        origin = _entity_sql_metadata_origin(entity)
        meta_origin = meta._sql_origin if isinstance(meta, LazyMetaDict) else None
        if origin is None and meta_origin is None:
            continue
        if (origin is None or not isinstance(meta, LazyMetaDict) or meta_origin != origin
                or origin.entity_id != entity.id or meta._replaced
                or meta._edited_keys() or not origin.matches_fallback("enrollment", entity.enrollment)
                or not origin.matches_fallback("rating", entity.rating)
                or repo.catalog is None or repo.catalog.build_id != origin.build_id):
            raise ContractError(
                f"Cannot export entity {entity.id} as release data. "
                "Its SQL-bound metadata, fallback values, or catalog were changed or could not be verified. "
                "Reload the original release for SQL export or export caller data separately."
            )
    if not replace and session.get_bind().dialect.name not in {"sqlite", "postgresql"}:
        raise NotImplementedError(
            "Partial database exports support SQLite and PostgreSQL"
        )
    _validate_storage(session.connection())
    # SQLAlchemy 2.1 autoflushes Core statements too. Refuse an incompatible
    # catalog before flushing any pending work owned by the caller.
    with session.no_autoflush:
        _bind_catalog(repo, session, replace=replace)
    session.flush()
    projection = DistrictSourceProjection.__table__
    has_projection = inspect(session.connection()).has_table(projection.name)
    if replace:
        if has_projection:
            session.connection().execute(delete(projection))
        for model in (
            CampusMetaRecord,
            DistrictMetaRecord,
            CampusTransferRecord,
            CampusRecord,
            DistrictRecord,
        ):
            session.connection().execute(delete(cast(Table, model.__table__)))
    _write_entities(
        session,
        DistrictRecord,
        DistrictMetaRecord,
        (
            _district_payload(entity, include_geometry=include_geometry)
            for entity in repo._districts.values()
        ),
        replace=replace,
        include_meta_entries=include_meta_entries,
        batch_size=batch_size,
    )
    if has_projection:
        fields = ("organization_sub_type", "district_boundary_status", "district_boundary_reason")
        _write_rows(
            session, projection,
            ({"district_id": district.id, **{field: district.meta.get(field) for field in fields}}
             for district in repo._districts.values()),
            replace=replace, batch_size=batch_size,
        )
    if not replace:
        # The catalog lock serializes supported writers, including when an
        # older destination lacks the unique index or stores quoted numbers.
        with session.connection().execute(
            select(DistrictRecord.district_number, DistrictRecord.district_number_canon)
        ) as rows:
            validate_stored_district_numbers((raw, canonical) for raw, canonical in rows)
    _write_entities(
        session,
        CampusRecord,
        CampusMetaRecord,
        (
            _campus_payload(entity, include_geometry=include_geometry)
            for entity in repo._campuses.values()
        ),
        replace=replace,
        include_meta_entries=include_meta_entries,
        batch_size=batch_size,
    )
    edges = (
        {
            "source_id": source_id,
            "destination_id": destination_id,
            "student_count": count,
            "masked": bool(masked),
        }
        for source_id, destinations in repo._xfers_out.items()
        for destination_id, count, masked in destinations
        if source_id is not None and destination_id is not None
    )
    _write_rows(
        session,
        CampusTransferRecord.__table__,
        edges,
        replace=replace,
        batch_size=batch_size,
    )
    session.expire_all()


def _read_catalog(session: Session) -> tuple[DatasetCatalog, str]:
    if not inspect(session.connection()).has_table(CATALOG_TABLE):
        raise ContractError(
            "Database lacks its required dataset catalog; rebuild from source"
        )
    with session.no_autoflush:
        row = _catalog_row(session)
    if row is None:
        raise ContractError(
            "Database lacks its required dataset catalog; rebuild from source"
        )
    return DatasetCatalog(json.loads(row[0]), expected_build_id=row[1]), row[1]


def _prefetch_raw_meta(
    session: Session, model, keys: Iterable[str] | None, batch_size: int,
    *, dialect: str | None, build_id: str,
) -> dict[uuid.UUID, dict[str, Any]]:
    selected = tuple(dict.fromkeys(keys or ()))
    if not selected:
        return {}
    seeds: dict[uuid.UUID, dict[str, Any]] = {}
    # Keep statement size and parameter count bounded even for unusual callers.
    projected = (dialect is not None and len(selected) <= 16 and all(map(_point_meta_key, selected))
                 and session.get_bind(mapper=model) is session.get_bind())
    columns = _meta_projection(model.meta, selected, dialect) if projected and dialect is not None else (model.meta,)
    statement = (
        select(model.id, *columns)
        .join(_CATALOG_RELATION, _CATALOG_RELATION.c.singleton == 1)
        .where(_CATALOG_RELATION.c.build_id == build_id)
        .execution_options(yield_per=batch_size)
    )
    with session.execute(statement) as rows:
        for row in rows:
            if projected:
                seed = dict(_decode_meta_projection(selected, row[1:]))
            else:
                raw = row[1]
                seed = {key: raw[key] for key in selected if key in raw} if isinstance(raw, Mapping) else {}
            if seed:
                seeds[row[0]] = seed
    return seeds


def import_dataengine(
    session: Session,
    *,
    lazy_meta: bool = True,
    prefetch_district_meta_keys: Iterable[str] | None = None,
    prefetch_campus_meta_keys: Iterable[str] | None = None,
    meta_session_factory: Callable[[], Session] | None = None,
    batch_size: int = 256,
) -> DataEngine:
    """Hydrate a catalog-bound graph without an additional full ORM graph.

    The returned DataEngine itself contains the complete entity/transfer graph;
    database row buffers are bounded by ``batch_size``. Eager metadata is selected
    with its entity, and lazy metadata refuses a different source release.
    Prefetch keys refer to exact top-level raw keys, preserving nested values.
    The caller controls transaction isolation for a consistent read while other
    clients mutate the database. Lazy reads observe that release's current rows.
    """
    batch_size = _checked_batch_size(batch_size)
    _validate_storage(session.connection())
    catalog, build_id = _read_catalog(session)
    meta_dialect = _point_meta_dialect(session.connection()) if lazy_meta else None
    repo = DataEngine()
    repo.catalog = catalog
    import_id = uuid.uuid4() if lazy_meta else None
    if import_id is not None:
        repo._lazy_sql_import = True
    if lazy_meta and meta_session_factory is None:
        meta_session_factory = sessionmaker(
            bind=session.get_bind(), expire_on_commit=False
        )
    district_seed = (
        _prefetch_raw_meta(
            session, DistrictRecord, prefetch_district_meta_keys, batch_size,
            dialect=meta_dialect, build_id=build_id,
        )
        if lazy_meta
        else {}
    )
    campus_seed = (
        _prefetch_raw_meta(session, CampusRecord, prefetch_campus_meta_keys, batch_size,
                           dialect=meta_dialect, build_id=build_id)
        if lazy_meta
        else {}
    )

    def statement(model):
        columns = [
            column
            for column in model.__table__.columns
            if not lazy_meta or column.name != "meta"
        ]
        return select(*columns).execution_options(yield_per=batch_size)

    with repo.bulk():
        with session.execute(statement(DistrictRecord)) as rows:
            for d in rows.mappings():
                origin = (_SQLMetadataOrigin(build_id, "district", d["id"], d["enrollment"], d["rating"], import_id)
                          if import_id is not None else None)
                meta = (
                    _lazy_meta(session, meta_session_factory, DistrictRecord, d["id"],
                               build_id, meta_dialect, district_seed.pop(d["id"], None), origin)
                    if origin is not None
                    else dict(d["meta"] or {})
                )
                district = District(
                    id=d["id"],
                    name=d["name"],
                    enrollment=d["enrollment"],
                    district_number=stored_district_number(d["district_number"], d["district_number_canon"]) or "",
                    aea=d["aea"],
                    aea_raw=d["aea_raw"],
                    rating=d["rating"],
                    boundary=_load_polygon(d["polygon_wkb"], d["polygon_geojson"]),
                    meta=meta,
                )
                if origin is not None:
                    district._sql_metadata_origin = origin
                repo.add_district(district)
        with session.execute(statement(CampusRecord)) as rows:
            for c in rows.mappings():
                origin = (_SQLMetadataOrigin(build_id, "campus", c["id"], c["enrollment"], c["rating"], import_id)
                          if import_id is not None else None)
                meta = (
                    _lazy_meta(session, meta_session_factory, CampusRecord, c["id"],
                               build_id, meta_dialect, campus_seed.pop(c["id"], None), origin)
                    if origin is not None
                    else dict(c["meta"] or {})
                )
                campus = Campus(
                    id=c["id"],
                    district_id=c["district_id"],
                    name=c["name"],
                    charter_type=c["charter_type"] or "",
                    is_charter=bool(c["is_charter"]),
                    is_private=bool(c["is_private"]),
                    is_magnet=c["is_magnet"],
                    enrollment=c["enrollment"],
                    rating=c["rating"],
                    aea=c["aea"],
                    aea_raw=c["aea_raw"],
                    grade_range=c["grade_range"],
                    school_type=c["school_type"],
                    school_status_date=c["school_status_date"],
                    update_date=c["update_date"],
                    district_number=c["district_number"],
                    campus_number=c["campus_number"],
                    source_namespace=c["source_namespace"],
                    source_id=c["source_id"],
                    location=_load_point(
                        c["point_wkb"], c["point_geojson"], c["lon"], c["lat"]
                    ),
                    meta=meta,
                )
                if campus.lookup_key != c["lookup_key"]:
                    raise ContractError("Stored campus lookup key differs from its source identity")
                if c["grade_range_code_spans"]:
                    spans = tuple(_coerce_grade_spans(c["grade_range_code_spans"]))
                    if spans:
                        object.__setattr__(campus, "grade_range_code_spans", spans)
                        low, high = _spans_to_bounds(spans)
                        object.__setattr__(campus, "grade_range_low_code", low)
                        object.__setattr__(campus, "grade_range_high_code", high)
                if origin is not None:
                    campus._sql_metadata_origin = origin
                repo.add_campus(campus)
    repo._xfers_out = defaultdict(list)
    repo._xfers_in = defaultdict(list)
    repo._xfers_missing = {"src": 0, "dst": 0, "either": 0}
    with session.execute(statement(CampusTransferRecord)) as rows:
        for edge in rows.mappings():
            repo._xfers_out[edge["source_id"]].append(
                (edge["destination_id"], edge["student_count"], edge["masked"])
            )
            repo._xfers_in[edge["destination_id"]].append(
                (edge["source_id"], edge["student_count"], edge["masked"])
            )
    if _read_catalog(session)[1] != build_id:
        raise ContractError(
            "The dataset catalog changed during hydration; retry in a consistent read transaction"
        )
    return repo
