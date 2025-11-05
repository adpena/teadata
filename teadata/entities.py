"""Domain entities (District, Campus) and collection helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date
from functools import cached_property, singledispatchmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple
import uuid

from teadata.teadata_config import (
    canonical_campus_number,
    canonical_district_number,
    normalize_campus_number_value,
    normalize_district_number_value,
)

from .geometry import (
    GeoField,
    SHAPELY,
    ShapelyPoint,
    district_centroid_xy,
    euclidean,
    haversine_miles,
    point_in_polygon,
    polygon_area,
)
from .grades import coerce_grade_bounds, coerce_grade_spans, spans_to_bounds

__all__ = [
    "District",
    "Campus",
    "EntityMap",
    "EntityList",
    "ReadOnlyEntityView",
    "is_charter",
    "is_private",
]


def validate_non_empty_str(name: str):
    """Decorator factory: enforce non-empty string attribute on __post_init__."""

    def deco(cls):
        orig_post = getattr(cls, "__post_init__", None)

        def post(self):
            if not getattr(self, name) or not isinstance(getattr(self, name), str):
                raise ValueError(f"{cls.__name__}.{name} must be a non-empty string")
            if orig_post:
                orig_post(self)

        cls.__post_init__ = post
        return cls

    return deco


def is_charter(obj: Any) -> bool:
    flag = getattr(obj, "is_charter", None)
    if flag is None and hasattr(obj, "meta"):
        try:
            return bool(obj.meta.get("is_charter"))
        except Exception:  # pragma: no cover - defensive
            return False
    return bool(flag)


def is_private(obj: Any) -> bool:
    flag = getattr(obj, "is_private", None)
    if flag is None and hasattr(obj, "meta"):
        try:
            return bool(obj.meta.get("is_private"))
        except Exception:  # pragma: no cover - defensive
            return False
    return bool(flag)


_is_charter = is_charter
_is_private = is_private


@validate_non_empty_str("name")
@dataclass(slots=True)
class District:
    id: uuid.UUID
    name: str
    enrollment: Optional[int] = None
    district_number: str = ""
    aea: Optional[bool] = None
    rating: Optional[str] = None

    _district_number_canon: str = field(init=False, repr=False, compare=False)

    boundary: Any = field(default=None, repr=False)
    _polygon: Any = field(init=False, repr=False, default=None)
    _prepared: Any = field(default=None, init=False, repr=False, compare=False)

    polygon = GeoField("polygon")

    meta: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    _repo: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.name = self.name.upper()
        if self.boundary is not None:
            self.polygon = self.boundary

        raw_dn = self.district_number
        cdn = normalize_district_number_value(raw_dn)
        if not cdn:
            cdn = ""
        self._district_number_canon = cdn
        if cdn:
            can = canonical_district_number(raw_dn)
            self.district_number = can or ""
        else:
            self.district_number = ""

    def __contains__(self, campus: "Campus") -> bool:
        if self.polygon is None or campus.point is None:
            return False
        if SHAPELY:
            return self.polygon.contains(campus.point)
        return point_in_polygon(campus.point, list(self.polygon))

    def __and__(self, other: "District") -> float:
        if self.polygon is None or other.polygon is None:
            return 0.0
        if SHAPELY:
            return self.polygon.intersection(other.polygon).area
        return (
            polygon_area(list(self.polygon)) if self.polygon == other.polygon else 0.0
        )

    def __or__(self, other: "District") -> float:
        if self.polygon is None or other.polygon is None:
            return 0.0
        if SHAPELY:
            return self.polygon.union(other.polygon).area
        if self.polygon == other.polygon:
            return polygon_area(list(self.polygon))
        return polygon_area(list(self.polygon)) + polygon_area(list(other.polygon))

    def __matmul__(self, campus: "Campus") -> float:
        if self.polygon is None or campus.point is None:
            return float("inf")
        if SHAPELY:
            return self.polygon.distance(campus.point)
        if campus in self:
            return 0.0
        verts = list(self.polygon)
        return min(euclidean(campus.point, v) for v in verts)

    def __format__(self, spec: str) -> str:
        if spec == "brief":
            return f"{self.name} (enr={self.enrollment:,}, rating={self.rating or '-'})"
        return f"District<{self.name}>"

    __match_args__ = ("name", "enrollment")

    @property
    def district_number_canon(self) -> str:
        return self._district_number_canon

    @cached_property
    def area(self) -> float:
        if self.polygon is None:
            return 0.0
        if SHAPELY:
            return self.polygon.area
        return polygon_area(list(self.polygon))

    @property
    def campuses(self) -> "EntityList":
        repo = getattr(self, "_repo", None)
        if repo is None:
            return EntityList()
        ids = repo._campuses_by_district.get(self.id, [])
        return EntityList([repo._campuses[cid] for cid in ids])

    if SHAPELY:
        from shapely.prepared import prep as shapely_prep  # type: ignore

    @property
    def prepared(self):
        if not SHAPELY or self.polygon is None:
            return None
        if self._prepared is not None:
            return self._prepared
        try:
            self._prepared = self.shapely_prep(self.polygon)
        except Exception:  # pragma: no cover - shapely failure path
            self._prepared = None
        return self._prepared

    def __getattr__(self, name: str):
        meta = object.__getattribute__(self, "meta")
        if isinstance(meta, dict) and name in meta:
            return meta[name]
        raise AttributeError(f"{self.__class__.__name__} has no attribute {name!r}")

    def __dir__(self):
        base = list(super().__dir__())
        m = object.__getattribute__(self, "meta")
        if isinstance(m, dict):
            base.extend(k for k in m.keys() if isinstance(k, str))
        return sorted(set(base))

    def to_dict(
        self, *, include_meta: bool = True, include_geometry: bool = False
    ) -> dict:
        out = {
            "id": str(self.id),
            "name": self.name,
            "enrollment": self.enrollment,
            "district_number": self.district_number,
            "aea": self.aea,
            "rating": self.rating,
        }
        if include_meta and isinstance(self.meta, dict):
            for k, v in self.meta.items():
                if k not in out:
                    out[k] = v

        if include_geometry:
            poly = getattr(self, "polygon", None)
            try:
                if poly is not None:
                    b = poly.bounds if hasattr(poly, "bounds") else None
                    out["geometry_bounds"] = tuple(b) if b else None
                    if hasattr(poly, "wkt"):
                        out["geometry_wkt"] = poly.wkt
            except Exception:  # pragma: no cover - defensive
                out["geometry_bounds"] = None
        return out


@validate_non_empty_str("name")
@dataclass(slots=True)
class Campus:
    id: uuid.UUID
    district_id: uuid.UUID
    name: str
    charter_type: str
    is_charter: bool
    is_private: bool = False
    is_magnet: Optional[str] = None
    enrollment: Optional[int] = None
    rating: Optional[str] = None

    aea: Optional[bool] = None
    grade_range: Optional[str] = None
    school_type: Optional[str] = None
    school_status_date: Optional[date] = None
    update_date: Optional[date] = None

    grade_range_code_spans: tuple[tuple[Optional[int], Optional[int]], ...] = field(
        init=False, repr=False, default=(), compare=False
    )
    grade_range_low_code: Optional[int] = field(
        init=False, repr=False, default=None, compare=False
    )
    grade_range_high_code: Optional[int] = field(
        init=False, repr=False, default=None, compare=False
    )

    district_number: Optional[str] = None
    campus_number: Optional[str] = None

    _district_number_canon: Optional[str] = field(init=False, repr=False, compare=False)
    _campus_number_canon: Optional[str] = field(init=False, repr=False, compare=False)

    _repo: Any = field(default=None, repr=False, compare=False)

    location: Any = field(default=None, repr=False)
    _point: Any = field(init=False, repr=False, default=None)

    point = GeoField("point")

    meta: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "grade_range":
            object.__setattr__(self, name, value)
            spans = tuple(coerce_grade_spans(value))
            object.__setattr__(self, "grade_range_code_spans", spans)
            low, high = spans_to_bounds(spans)
            object.__setattr__(self, "grade_range_low_code", low)
            object.__setattr__(self, "grade_range_high_code", high)
            return
        object.__setattr__(self, name, value)

    def __post_init__(self):
        self.name = self.name.upper()
        self.is_charter = bool(self.is_charter)
        self.is_private = bool(self.is_private)
        if self.location is not None:
            self.point = self.location

        spans = tuple(coerce_grade_spans(self.grade_range))
        object.__setattr__(self, "grade_range_code_spans", spans)
        low, high = spans_to_bounds(spans)
        object.__setattr__(self, "grade_range_low_code", low)
        object.__setattr__(self, "grade_range_high_code", high)

        raw_dn = self.district_number
        if raw_dn:
            canon_dn = normalize_district_number_value(raw_dn)
            self._district_number_canon = canon_dn
            if canon_dn:
                self.district_number = canonical_district_number(raw_dn) or raw_dn
        else:
            self._district_number_canon = None

        raw_cn = self.campus_number
        if raw_cn:
            canon_cn = normalize_campus_number_value(raw_cn)
            self._campus_number_canon = canon_cn
            if canon_cn:
                self.campus_number = canonical_campus_number(raw_cn) or raw_cn
        else:
            self._campus_number_canon = None

    def __getattr__(self, name: str):
        meta = object.__getattribute__(self, "meta")
        if isinstance(meta, dict) and name in meta:
            return meta[name]
        raise AttributeError(f"{self.__class__.__name__} has no attribute {name!r}")

    def __dir__(self):
        base = list(super().__dir__())
        m = object.__getattribute__(self, "meta")
        if isinstance(m, dict):
            base.extend(k for k in m.keys() if isinstance(k, str))
        return sorted(set(base))

    def to_dict(
        self, *, include_meta: bool = True, include_geometry: bool = False
    ) -> dict:
        out = {
            "id": str(self.id),
            "district_id": str(self.district_id),
            "name": self.name,
            "charter_type": self.charter_type,
            "is_charter": bool(self.is_charter),
            "is_private": bool(self.is_private),
            "is_magnet": self.is_magnet,
            "enrollment": self.enrollment,
            "rating": self.rating,
            "aea": self.aea,
            "grade_range": self.grade_range,
            "school_type": self.school_type,
            "school_status_date": self.school_status_date,
            "update_date": self.update_date,
            "district_number": self.district_number,
            "campus_number": self.campus_number,
        }
        if include_meta and isinstance(self.meta, dict):
            for k, v in self.meta.items():
                if k not in out:
                    out[k] = v

        percent_change: Optional[float]
        try:
            percent_change = self.enrollment_percent_change_from_2015()
        except ValueError:
            percent_change = None
        except Exception:  # pragma: no cover - defensive
            percent_change = None

        derived_attrs = {
            "percent_enrollment_change": percent_change,
            "num_charter_transfer_destinations": self.num_charter_transfer_destinations,
            "num_charter_transfer_destinations_masked": self.num_charter_transfer_destinations_masked,
            "total_unmasked_charter_transfers_out": self.total_unmasked_charter_transfers_out,
        }
        for key, value in derived_attrs.items():
            if key not in out:
                out[key] = value

        if include_geometry:
            pt = getattr(self, "point", None)
            try:
                if pt is not None:
                    out["geometry_point"] = (float(pt.x), float(pt.y))
            except Exception:  # pragma: no cover - defensive
                out["geometry_point"] = None
        return out

    def to_flat_dict(self) -> dict:
        return self.to_dict(include_meta=True, include_geometry=True)

    def __lt__(self, other: "Campus") -> bool:
        return self.enrollment > other.enrollment

    def __sub__(self, other: "Campus | Tuple[float, float]") -> float:
        return self.distance_to(other)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        attrs = [f"name={self.name!r}", f"district_number={self.district_number!r}"]
        return f"Campus({', '.join(attrs)})"

    def __hash__(self):
        return hash(self.id)

    @property
    def coords(self) -> tuple[float, float] | None:
        if self.point is None:
            return None
        try:
            return (self.point.x, self.point.y)
        except Exception:  # pragma: no cover - shapely fallback
            return None

    @property
    def district(self) -> Optional[District]:
        if self._repo is None:
            return None
        return self._repo._districts.get(self.district_id)

    def distance_to(self, other: Any) -> float:
        return self._distance_to(other)

    @singledispatchmethod
    def _distance_to(self, other: Any) -> float:
        raise TypeError(f"Cannot compute distance to {type(other)!r}")

    def _distance_to_campus(self, other: "Campus") -> float:
        if self.point is None or other.point is None:
            return float("inf")
        if (
            SHAPELY
            and isinstance(self.point, ShapelyPoint)
            and isinstance(other.point, ShapelyPoint)
        ):
            return self.point.distance(other.point)
        return euclidean((self.point.x, self.point.y), (other.point.x, other.point.y))

    @_distance_to.register
    def _(self, other: tuple) -> float:
        if self.point is None:
            return float("inf")
        if SHAPELY and isinstance(self.point, ShapelyPoint):
            return self.point.distance(ShapelyPoint(*other))
        return euclidean((self.point.x, self.point.y), other)

    def enrollment_percent_change_from_2015(self) -> float:
        enrollment = self.enrollment
        baseline = None
        if (
            isinstance(self.meta, dict)
            and "campus_2015_student_enrollment_all_students_count" in self.meta
        ):
            baseline = self.meta["campus_2015_student_enrollment_all_students_count"]
        elif hasattr(self, "campus_2015_student_enrollment_all_students_count"):
            baseline = getattr(
                self, "campus_2015_student_enrollment_all_students_count"
            )
        if enrollment is None:
            raise ValueError("Current enrollment is missing or None.")
        if baseline is None:
            raise ValueError("Baseline 2015 enrollment is missing or None.")
        try:
            enrollment = float(enrollment)
            baseline = float(baseline)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(
                "Enrollment or baseline could not be cast to float."
            ) from exc
        if baseline == 0:
            raise ValueError(
                "Baseline 2015 enrollment is zero, cannot compute percent change."
            )
        return (enrollment - baseline) / baseline

    @property
    def num_charter_transfer_destinations(self) -> int:
        if not hasattr(self, "_repo") or self._repo is None:
            return 0
        try:
            edges = self._repo.transfers_out(self)
        except Exception:
            return 0
        ids = set()
        for to_campus, _count, _masked in ((e[0], e[1], e[2]) for e in edges):
            if to_campus is not None and is_charter(to_campus):
                ids.add(getattr(to_campus, "id", None))
        return len(ids)

    @property
    def num_charter_transfer_destinations_masked(self) -> int:
        if not hasattr(self, "_repo") or self._repo is None:
            return 0
        try:
            edges = self._repo.transfers_out(self)
        except Exception:
            return 0
        masked_ids = set()
        for to_campus, _count, masked in ((e[0], e[1], e[2]) for e in edges):
            if to_campus is not None and is_charter(to_campus) and bool(masked):
                masked_ids.add(getattr(to_campus, "id", None))
        return len(masked_ids)

    @property
    def total_unmasked_charter_transfers_out(self) -> int:
        if not hasattr(self, "_repo") or self._repo is None:
            return 0
        try:
            edges = self._repo.transfers_out(self)
        except Exception:
            return 0
        total = 0
        for to_campus, count, masked in ((e[0], e[1], e[2]) for e in edges):
            if to_campus is None or not is_charter(to_campus):
                continue
            if count is None:
                continue
            try:
                total += int(count)
            except Exception:
                continue
        return total


class EntityMap(dict):
    def to_dicts(
        self, *, include_meta: bool = True, include_geometry: bool = False
    ) -> list[dict]:
        rows = []
        for obj in self.values():
            if hasattr(obj, "to_dict"):
                rows.append(
                    obj.to_dict(
                        include_meta=include_meta, include_geometry=include_geometry
                    )
                )
            else:
                try:
                    rows.append(dict(vars(obj)))
                except Exception:
                    rows.append({"value": obj})
        return rows

    def to_df(
        self,
        columns: list[str] | None = None,
        *,
        include_meta: bool = True,
        include_geometry: bool = False,
    ):
        try:
            import pandas as pd  # type: ignore
        except Exception as exc:
            raise ImportError(
                "pandas is required for .to_df(); install pandas to use this feature"
            ) from exc
        data = self.to_dicts(
            include_meta=include_meta, include_geometry=include_geometry
        )
        df = pd.DataFrame(data)
        if columns is not None:
            cols = [c for c in columns if c in df.columns]
            df = df[cols]
        return df

    def unique(self, attr, *, dropna: bool = True, sort: bool = True):
        if callable(attr):
            getter = attr
        else:
            getter = lambda o: getattr(o, attr, None)

        vals = set()
        for obj in self.values():
            try:
                v = getter(obj)
            except Exception:
                v = None
            if v is None and dropna:
                continue
            vals.add(v)
        out = list(vals)
        if sort:
            try:
                out.sort()
            except Exception:
                pass
        return out

    def value_counts(
        self, attr, *, dropna: bool = True, sort: bool = True, descending: bool = True
    ):
        if callable(attr):
            getter = attr
        else:
            getter = lambda o: getattr(o, attr, None)

        counts = {}
        for obj in self.values():
            try:
                v = getter(obj)
            except Exception:
                v = None
            if v is None and dropna:
                continue
            counts[v] = counts.get(v, 0) + 1

        if sort:
            items = list(counts.items())
            items.sort(key=lambda kv: (-(kv[1]) if descending else kv[1], kv[0]))
            return items
        return counts


class EntityList(list):
    def to_dicts(
        self, *, include_meta: bool = True, include_geometry: bool = False
    ) -> list[dict]:
        rows = []
        for obj in self:
            if hasattr(obj, "to_dict"):
                rows.append(
                    obj.to_dict(
                        include_meta=include_meta, include_geometry=include_geometry
                    )
                )
            else:
                try:
                    d = dict(vars(obj))
                except Exception:
                    d = {"value": obj}
                rows.append(d)
        return rows

    def to_df(
        self,
        columns: list[str] | None = None,
        *,
        include_meta: bool = True,
        include_geometry: bool = False,
    ):
        try:
            import pandas as pd  # type: ignore
        except Exception as exc:
            raise ImportError(
                "pandas is required for .to_df(); install pandas to use this feature"
            ) from exc
        data = self.to_dicts(
            include_meta=include_meta, include_geometry=include_geometry
        )
        df = pd.DataFrame(data)
        if columns is not None:
            cols = [c for c in columns if c in df.columns]
            df = df[cols]
        return df

    def unique(self, attr):
        if callable(attr):
            getter = attr
        else:
            getter = lambda o: getattr(o, attr, None)
        values = [getter(obj) for obj in self]
        return sorted(set(v for v in values if v is not None))

    def value_counts(
        self, attr, *, dropna: bool = True, sort: bool = True, descending: bool = True
    ):
        if callable(attr):
            getter = attr
        else:
            getter = lambda o: getattr(o, attr, None)

        counts = {}
        for obj in self:
            try:
                v = getter(obj)
            except Exception:
                v = None
            if v is None and dropna:
                continue
            counts[v] = counts.get(v, 0) + 1

        if sort:
            items = list(counts.items())
            items.sort(key=lambda kv: (-(kv[1]) if descending else kv[1], kv[0]))
            return items
        return counts

    def head(self, n=5):
        return self[:n]

    def sample(self, n=5, seed=None):
        import random

        rng = random.Random(seed)
        return rng.sample(self, min(n, len(self)))


class ReadOnlyEntityView:
    __slots__ = ("_m",)

    def __init__(self, backing: EntityMap):
        self._m = backing

    def __len__(self):
        return len(self._m)

    def __iter__(self):
        return iter(self._m)

    def __getitem__(self, k):
        return self._m[k]

    def keys(self):
        return self._m.keys()

    def values(self):
        return self._m.values()

    def items(self):
        return self._m.items()

    def unique(self, *args, **kwargs):
        return self._m.unique(*args, **kwargs)

    def value_counts(self, *args, **kwargs):
        return self._m.value_counts(*args, **kwargs)

    def __dir__(self):
        base = set(type(self).__dict__.keys())
        base.update(
            ["unique", "value_counts", "keys", "values", "items", "to_df", "to_dicts"]
        )
        return sorted(base)

    def to_dicts(self, *args, **kwargs):
        return self._m.to_dicts(*args, **kwargs)

    def to_df(self, *args, **kwargs):
        return self._m.to_df(*args, **kwargs)


# Register the Campus-specific overload after class creation to avoid forward references
Campus._distance_to.register(Campus)(Campus._distance_to_campus)
