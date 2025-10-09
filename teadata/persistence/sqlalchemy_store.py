"""SQLAlchemy persistence helpers for :mod:`teadata` entities.

The goal of this module is to bridge the in-memory :class:`~teadata.classes.DataEngine`
container with a relational database so that large snapshots can be stored in a
PostgreSQL (or PostgreSQL-compatible) database and queried efficiently without
loading a 75MB pickle into every web request.

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
* Capture geometry as both WKB (for geospatial tooling) and GeoJSON-ish
  fallbacks so we can recreate shapely objects when available while remaining
  resilient when Shapely is not installed in the consumer environment.

The module is intentionally lightweight and does **not** require a Django
settings module.  It can therefore be reused inside a Django project (via
``django.db.backends.postgresql`` + SQLAlchemy session management) or any other
Python application that can provide a SQLAlchemy engine/Session.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone
import json
import uuid
from typing import Any, Callable, Iterable, Literal, Mapping, MutableMapping

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
    create_engine as _sa_create_engine,
    delete,
    func,
    select,
)
from sqlalchemy import Index
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    defer,
    mapped_column,
    relationship,
    sessionmaker,
)

try:  # JSON type that degrades gracefully outside PostgreSQL
    from sqlalchemy.dialects.postgresql import JSONB as _JSONType
except Exception:  # pragma: no cover - fallback for non-Postgres backends
    from sqlalchemy import JSON as _JSONType  # type: ignore

try:  # PostgreSQL-optimized UUID type; otherwise fall back to generic UUID/CHAR
    from sqlalchemy.dialects.postgresql import UUID as _PGUUID
except Exception:  # pragma: no cover
    _PGUUID = None

try:  # SQLAlchemy 2.0 native UUID type (works across backends)
    from sqlalchemy import Uuid as _GenericUUID  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    _GenericUUID = None

from ..classes import Campus, DataEngine, District

try:  # Optional Shapely helpers for geometry round-tripping
    from shapely import wkb as _shapely_wkb
    from shapely.geometry import shape as _shapely_shape
    from shapely.geometry.base import BaseGeometry as _BaseGeometry
    from shapely.geometry import mapping as _shapely_mapping

    _SHAPELY_AVAILABLE = True
except Exception:  # pragma: no cover - Shapely intentionally optional
    _shapely_wkb = None
    _shapely_shape = None
    _BaseGeometry = None
    _shapely_mapping = None
    _SHAPELY_AVAILABLE = False

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


def _uuid_column(**kwargs) -> mapped_column[Any]:
    """Return a mapped column configured for UUID primary/foreign keys."""

    if _PGUUID is not None:  # Prefer PostgreSQL UUID implementation
        return mapped_column(_PGUUID(as_uuid=True), **kwargs)
    if _GenericUUID is not None:  # SQLAlchemy 2.0 portable UUID
        return mapped_column(_GenericUUID(as_uuid=True), **kwargs)  # type: ignore[misc]

    # Fallback: store UUIDs as CHAR(36) while still returning uuid.UUID objects
    class _UUIDString(String):
        def __init__(self):
            super().__init__(length=36)

        def bind_processor(self, dialect):  # pragma: no cover - simple conversion
            def process(value: Any):
                if value is None:
                    return None
                if isinstance(value, uuid.UUID):
                    return str(value)
                return str(uuid.UUID(str(value)))

            return process

        def result_processor(self, dialect, coltype):  # pragma: no cover
            def process(value: Any):
                if value is None:
                    return None
                return uuid.UUID(str(value))

            return process

    return mapped_column(_UUIDString(), **kwargs)


class DistrictRecord(Base):
    __tablename__ = "districts"

    id: Mapped[uuid.UUID] = _uuid_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    enrollment: Mapped[int | None] = mapped_column(Integer)
    district_number: Mapped[str | None] = mapped_column(String(16), index=True)
    district_number_canon: Mapped[str | None] = mapped_column(String(16), index=True)
    aea: Mapped[bool | None] = mapped_column(Boolean)
    rating: Mapped[str | None] = mapped_column(String(16))
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
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    charter_type: Mapped[str | None] = mapped_column(String(64))
    is_charter: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    enrollment: Mapped[int | None] = mapped_column(Integer)
    rating: Mapped[str | None] = mapped_column(String(16))
    aea: Mapped[bool | None] = mapped_column(Boolean)
    grade_range: Mapped[str | None] = mapped_column(String(64))
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

    return sessionmaker(bind=engine, expire_on_commit=expire_on_commit, class_=Session, **kwargs)


def ensure_schema(engine) -> None:
    """Create the database schema if it does not already exist."""

    Base.metadata.create_all(engine)


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


class LazyMetaDict(dict):
    """Dictionary-like wrapper that loads values on demand via a callback."""

    def __init__(self, loader: Callable[[], Mapping[str, Any] | None], *, seed: Mapping[str, Any] | None = None):
        super().__init__()
        self._loader = loader
        self._loaded = False
        if seed:
            super().update(seed)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._loader is None:  # pragma: no cover - safety net
            self._loaded = True
            return
        payload = self._loader() or {}
        super().update(payload)
        self._loaded = True
        self._loader = None  # release references for GC

    def __getitem__(self, key):  # type: ignore[override]
        if not dict.__contains__(self, key):
            self._ensure_loaded()
        return dict.__getitem__(self, key)

    def get(self, key, default=None):  # type: ignore[override]
        if not dict.__contains__(self, key):
            self._ensure_loaded()
        return dict.get(self, key, default)

    def __contains__(self, key):  # type: ignore[override]
        if dict.__contains__(self, key):
            return True
        self._ensure_loaded()
        return dict.__contains__(self, key)

    def keys(self):  # type: ignore[override]
        self._ensure_loaded()
        return super().keys()

    def values(self):  # type: ignore[override]
        self._ensure_loaded()
        return super().values()

    def items(self):  # type: ignore[override]
        self._ensure_loaded()
        return super().items()

    def update(self, *args, **kwargs):  # type: ignore[override]
        self._ensure_loaded()
        return super().update(*args, **kwargs)

    def setdefault(self, key, default=None):  # type: ignore[override]
        self._ensure_loaded()
        return super().setdefault(key, default)

    def pop(self, key, default=None):  # type: ignore[override]
        self._ensure_loaded()
        return super().pop(key, default)


def _meta_loader(
    default_session: Session,
    session_factory: Callable[[], Session] | None,
    record_cls: type[DistrictRecord] | type[CampusRecord],
    pk_value: uuid.UUID,
) -> Callable[[], Mapping[str, Any] | None]:
    column = record_cls.id

    def _load() -> Mapping[str, Any] | None:
        active = session_factory() if session_factory is not None else default_session
        close_after = session_factory is not None
        try:
            result = active.execute(select(record_cls.meta).where(column == pk_value)).scalar_one_or_none()
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
        payload["value_numeric"] = float(value)
        return payload

    parsed_dt: datetime | None = None
    if isinstance(value, datetime):
        parsed_dt = value
        payload["value_text"] = value.isoformat()
    elif isinstance(value, date):
        parsed_dt = datetime.combine(value, datetime.min.time())
        payload["value_text"] = value.isoformat()
    elif isinstance(value, str):
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
    if "value_text" not in payload and isinstance(value, str):
        payload["value_text"] = value
    return payload


def _build_meta_records(
    meta: Mapping[str, Any] | None,
    record_cls: type[DistrictMetaRecord] | type[CampusMetaRecord],
    *,
    assume_clean: bool = False,
) -> list[DistrictMetaRecord | CampusMetaRecord]:
    if assume_clean and isinstance(meta, Mapping):
        clean = dict(meta)
    else:
        clean = _clean_meta(meta) or {}
    flattened = _flatten_meta(clean)
    records: list[DistrictMetaRecord | CampusMetaRecord] = []
    for key, ordinal, value in flattened:
        payload = _meta_value_payload(value)
        records.append(record_cls(key=key, ordinal=ordinal, **payload))
    return records


def _meta_record_for(entity: Literal["district", "campus"]):
    if entity == "district":
        return DistrictMetaRecord, DistrictMetaRecord.district_id
    if entity == "campus":
        return CampusMetaRecord, CampusMetaRecord.campus_id
    raise ValueError(f"Unsupported entity type: {entity}")


def _accumulate_meta_value(bucket: MutableMapping[str, Any], key: str, ordinal: int, value: Any) -> None:
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
    stmt = select(record_cls.key, func.count()).group_by(record_cls.key).order_by(record_cls.key)
    rows = session.execute(stmt).all()
    if with_counts:
        return {row[0]: int(row[1]) for row in rows}
    return [row[0] for row in rows]


def _dump_polygon(polygon: Any) -> tuple[bytes | None, dict[str, Any] | None]:
    if polygon is None:
        return None, None

    if _SHAPELY_AVAILABLE and isinstance(polygon, _BaseGeometry):
        try:
            return polygon.wkb, _shapely_mapping(polygon)
        except Exception:  # pragma: no cover - shapely serialization failure
            pass

    # Fallback: attempt to coerce to GeoJSON-like mapping from coordinates
    coords: list[list[float]] = []
    try:
        for pt in polygon:  # type: ignore[assignment]
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


def _dump_point(point: Any) -> tuple[bytes | None, dict[str, Any] | None, float | None, float | None]:
    if point is None:
        return None, None, None, None

    if _SHAPELY_AVAILABLE and isinstance(point, _BaseGeometry):
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
    if wkb_bytes and _SHAPELY_AVAILABLE:
        try:
            return _shapely_wkb.loads(wkb_bytes)
        except Exception:  # pragma: no cover
            pass

    if geojson:
        if _SHAPELY_AVAILABLE:
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
    if wkb_bytes and _SHAPELY_AVAILABLE:
        try:
            return _shapely_wkb.loads(wkb_bytes)
        except Exception:  # pragma: no cover
            pass

    if geojson:
        if _SHAPELY_AVAILABLE:
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


def export_dataengine(
    repo: DataEngine,
    session: Session,
    *,
    replace: bool = True,
) -> None:
    """Persist the contents of ``repo`` into the database bound to ``session``.

    Parameters
    ----------
    repo:
        The populated :class:`~teadata.classes.DataEngine` instance.
    session:
        A live SQLAlchemy :class:`~sqlalchemy.orm.Session`.
    replace:
        When ``True`` (default) the destination tables are truncated prior to
        inserting fresh rows.  Set to ``False`` to perform in-place upserts.
    """

    if replace:
        session.execute(delete(CampusTransferRecord))
        session.execute(delete(CampusRecord))
        session.execute(delete(DistrictRecord))
        session.flush()

    # Districts -------------------------------------------------------------
    for district in repo._districts.values():
        record = session.get(DistrictRecord, district.id)
        if record is None:
            record = DistrictRecord(id=district.id)
            session.add(record)

        record.name = district.name
        record.enrollment = district.enrollment
        record.district_number = district.district_number or None
        record.district_number_canon = district.district_number_canon or None
        record.aea = district.aea
        record.rating = district.rating
        poly_wkb, poly_geojson = _dump_polygon(getattr(district, "polygon", None))
        record.polygon_wkb = poly_wkb
        record.polygon_geojson = poly_geojson
        clean_meta = _clean_meta(district.meta)
        record.meta = clean_meta
        record.meta_entries = _build_meta_records(
            clean_meta, DistrictMetaRecord, assume_clean=True
        )

    session.flush()

    # Campuses --------------------------------------------------------------
    for campus in repo._campuses.values():
        record = session.get(CampusRecord, campus.id)
        if record is None:
            record = CampusRecord(id=campus.id)
            session.add(record)

        record.district_id = campus.district_id
        record.name = campus.name
        record.charter_type = campus.charter_type
        record.is_charter = bool(campus.is_charter)
        record.enrollment = campus.enrollment
        record.rating = campus.rating
        record.aea = campus.aea
        record.grade_range = campus.grade_range
        record.school_type = campus.school_type
        record.school_status_date = campus.school_status_date
        record.update_date = campus.update_date
        record.district_number = campus.district_number
        record.district_number_canon = campus.district_number_canon
        record.campus_number = campus.campus_number
        record.campus_number_canon = campus.campus_number_canon
        pt_wkb, pt_geojson, lon, lat = _dump_point(getattr(campus, "point", None))
        record.point_wkb = pt_wkb
        record.point_geojson = pt_geojson
        record.lon = lon
        record.lat = lat
        clean_meta = _clean_meta(campus.meta)
        record.meta = clean_meta
        record.meta_entries = _build_meta_records(
            clean_meta, CampusMetaRecord, assume_clean=True
        )

    session.flush()

    # Transfers -------------------------------------------------------------
    existing: dict[tuple[uuid.UUID, uuid.UUID], CampusTransferRecord] = {}
    if not replace:
        result = session.execute(select(CampusTransferRecord))
        for row in result.scalars():
            existing[(row.source_id, row.destination_id)] = row

    for source_id, edges in repo._xfers_out.items():
        for dest_id, count, masked in edges:
            if source_id is None or dest_id is None:
                continue
            key = (source_id, dest_id)
            record = existing.get(key)
            if record is None:
                record = CampusTransferRecord(
                    source_id=source_id,
                    destination_id=dest_id,
                    student_count=count,
                    masked=bool(masked),
                )
                session.add(record)
                existing[key] = record
            else:
                record.student_count = count
                record.masked = bool(masked)

    session.flush()


def import_dataengine(
    session: Session,
    *,
    lazy_meta: bool = True,
    prefetch_district_meta_keys: Iterable[str] | None = None,
    prefetch_campus_meta_keys: Iterable[str] | None = None,
    meta_session_factory: Callable[[], Session] | None = None,
) -> DataEngine:
    """Hydrate a :class:`~teadata.classes.DataEngine` from the SQL database."""

    repo = DataEngine()

    district_stmt = select(DistrictRecord).options(defer(DistrictRecord.meta))
    campus_stmt = select(CampusRecord).options(defer(CampusRecord.meta))

    districts = session.execute(district_stmt).scalars().all()
    campuses = session.execute(campus_stmt).scalars().all()
    transfers = session.execute(select(CampusTransferRecord)).scalars().all()

    district_ids = [d.id for d in districts]
    campus_ids = [c.id for c in campuses]

    district_seed: dict[uuid.UUID, dict[str, Any]] = {}
    campus_seed: dict[uuid.UUID, dict[str, Any]] = {}
    if prefetch_district_meta_keys:
        district_seed = fetch_meta_values(
            session,
            entity="district",
            keys=prefetch_district_meta_keys,
            ids=district_ids,
        )
    if prefetch_campus_meta_keys:
        campus_seed = fetch_meta_values(
            session,
            entity="campus",
            keys=prefetch_campus_meta_keys,
            ids=campus_ids,
        )

    with repo.bulk():
        for d in districts:
            if lazy_meta:
                loader = _meta_loader(session, meta_session_factory, DistrictRecord, d.id)
                seed = district_seed.get(d.id) if district_seed else None
                meta_payload = LazyMetaDict(loader, seed=seed)
            else:
                meta_payload = dict(d.meta or {})

            district = District(
                id=d.id,
                name=d.name,
                enrollment=d.enrollment,
                district_number=d.district_number or "",
                aea=d.aea,
                rating=d.rating,
                boundary=_load_polygon(d.polygon_wkb, d.polygon_geojson),
                meta=meta_payload,
            )
            repo.add_district(district)

        for c in campuses:
            if lazy_meta:
                loader = _meta_loader(session, meta_session_factory, CampusRecord, c.id)
                seed = campus_seed.get(c.id) if campus_seed else None
                meta_payload = LazyMetaDict(loader, seed=seed)
            else:
                meta_payload = dict(c.meta or {})

            campus = Campus(
                id=c.id,
                district_id=c.district_id,
                name=c.name,
                charter_type=c.charter_type or "",
                is_charter=bool(c.is_charter),
                enrollment=c.enrollment,
                rating=c.rating,
                aea=c.aea,
                grade_range=c.grade_range,
                school_type=c.school_type,
                school_status_date=c.school_status_date,
                update_date=c.update_date,
                district_number=c.district_number,
                campus_number=c.campus_number,
                location=_load_point(c.point_wkb, c.point_geojson, c.lon, c.lat),
                meta=meta_payload,
            )
            repo.add_campus(campus)

    repo._rebuild_indexes()

    repo._xfers_out = defaultdict(list)
    repo._xfers_in = defaultdict(list)
    repo._xfers_missing = {"src": 0, "dst": 0, "either": 0}

    for edge in transfers:
        repo._xfers_out[edge.source_id].append((edge.destination_id, edge.student_count, edge.masked))
        repo._xfers_in[edge.destination_id].append((edge.source_id, edge.student_count, edge.masked))

    return repo
