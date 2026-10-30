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

from collections import UserDict, defaultdict
from datetime import date, datetime, timezone
from itertools import islice
import json
import math
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
    create_engine as _sa_create_engine,
    column as sql_column,
    delete,
    func,
    inspect,
    insert,
    select,
    text,
    table as sql_table,
)
from sqlalchemy import Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
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
from ..teadata_config import canonical_campus_number, canonical_district_number
from ..manifest import (
    CATALOG_TABLE,
    CATALOG_TABLE_DDL,
    ContractError,
    DatasetCatalog,
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
    district_number_canon: Mapped[str | None] = mapped_column(String(16), index=True)
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


class LazyMetaDict(UserDict[str, Any]):
    """A standard mutable mapping with one source-bound lazy data boundary.

    Sparse prefetched reads may use the seed. Every whole-mapping operation or
    mutation first loads the complete current payload. UserDict supplies mapping
    equality, unions, updates, deletion, and copying without dict C fast paths.
    """

    def __init__(
        self,
        loader: Callable[[], Mapping[str, Any] | None]
        | Mapping[str, Any]
        | None = None,
        *,
        seed: Mapping[str, Any] | None = None,
    ):
        if callable(loader):
            self._loader: Callable[[], Mapping[str, Any] | None] | None = loader
            self._loaded = False
            self._payload = dict(seed or {})
        else:
            self._loader = None
            self._loaded = True
            self._payload = dict(loader or {})

    @property
    def data(self) -> dict[str, Any]:
        if not self._loaded:
            assert self._loader is not None
            # Replace, rather than merge: a prefetched key may have been deleted.
            payload = dict(self._loader() or {})
            self._payload = payload
            self._loaded = True
            self._loader = None
        return self._payload

    @data.setter
    def data(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self._loaded = True
        self._loader = None

    def __getitem__(self, key: str) -> Any:
        if key in self._payload:
            return self._payload[key]
        return self.data[key]

    def __contains__(self, key: object) -> bool:
        return key in self._payload or key in self.data

    def pop(self, key: str, *default) -> Any:
        return self.data.pop(key, *default)

    def popitem(self) -> tuple[str, Any]:
        return self.data.popitem()

    def setdefault(self, key: str, default: Any = None) -> Any:
        return self.data.setdefault(key, default)

    def __copy__(self):
        # UserDict's default implementation assumes data is an instance slot.
        return type(self)(self.data.copy())

    def __reduce__(self):
        # A persisted graph must contain values, never Session/loader closures.
        return type(self), (dict(self.data),)


def _meta_loader(
    default_session: Session,
    session_factory: Callable[[], Session] | None,
    record_cls: type[DistrictRecord] | type[CampusRecord],
    pk_value: uuid.UUID,
    build_id: str,
) -> Callable[[], Mapping[str, Any] | None]:
    column = record_cls.id

    def _load() -> Mapping[str, Any] | None:
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
            row = active.execute(
                select(record_cls.meta)
                .join(_CATALOG_RELATION, _CATALOG_RELATION.c.singleton == 1)
                .where(column == pk_value, _CATALOG_RELATION.c.build_id == build_id)
            ).one_or_none()
            if row is None:
                raise ContractError(
                    "Lazy metadata source was deleted or its dataset catalog changed"
                )
            result = row[0]
            return result if isinstance(result, Mapping) else {}
        finally:
            if close_after:
                active.close()

    return _load


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
        "district_number_canon": getattr(district, "district_number_canon", None)
        or canonical_district_number(district.district_number),
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
        "district_number_canon": getattr(campus, "district_number_canon", None)
        or canonical_district_number(campus.district_number),
        "campus_number": campus.campus_number,
        "campus_number_canon": getattr(campus, "campus_number_canon", None)
        or canonical_campus_number(campus.campus_number),
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
    if replace:
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
    session: Session, model, keys: Iterable[str] | None, batch_size: int
) -> dict[uuid.UUID, dict[str, Any]]:
    selected = tuple(dict.fromkeys(keys or ()))
    if not selected:
        return {}
    seeds: dict[uuid.UUID, dict[str, Any]] = {}
    statement = select(model.id, model.meta).execution_options(yield_per=batch_size)
    with session.execute(statement) as rows:
        for identifier, raw in rows:
            if isinstance(raw, Mapping):
                seed = {key: raw[key] for key in selected if key in raw}
                if seed:
                    seeds[identifier] = seed
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
    repo = DataEngine()
    repo.catalog = catalog
    if lazy_meta and meta_session_factory is None:
        meta_session_factory = sessionmaker(
            bind=session.get_bind(), expire_on_commit=False
        )
    district_seed = (
        _prefetch_raw_meta(
            session, DistrictRecord, prefetch_district_meta_keys, batch_size
        )
        if lazy_meta
        else {}
    )
    campus_seed = (
        _prefetch_raw_meta(session, CampusRecord, prefetch_campus_meta_keys, batch_size)
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
                meta = (
                    LazyMetaDict(
                        _meta_loader(
                            session,
                            meta_session_factory,
                            DistrictRecord,
                            d["id"],
                            build_id,
                        ),
                        seed=district_seed.pop(d["id"], None),
                    )
                    if lazy_meta
                    else dict(d["meta"] or {})
                )
                repo.add_district(
                    District(
                        id=d["id"],
                        name=d["name"],
                        enrollment=d["enrollment"],
                        district_number=d["district_number"] or "",
                        aea=d["aea"],
                        aea_raw=d["aea_raw"],
                        rating=d["rating"],
                        boundary=_load_polygon(d["polygon_wkb"], d["polygon_geojson"]),
                        meta=meta,
                    )
                )
        with session.execute(statement(CampusRecord)) as rows:
            for c in rows.mappings():
                meta = (
                    LazyMetaDict(
                        _meta_loader(
                            session,
                            meta_session_factory,
                            CampusRecord,
                            c["id"],
                            build_id,
                        ),
                        seed=campus_seed.pop(c["id"], None),
                    )
                    if lazy_meta
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
