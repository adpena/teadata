from __future__ import annotations
from dataclasses import dataclass, field, fields, is_dataclass
from functools import lru_cache, cached_property, singledispatchmethod, wraps
from typing import Iterable, List, Dict, Optional, Any, Callable, Tuple
from contextlib import contextmanager
from collections import defaultdict
from datetime import date
import math
import time
import uuid
from pathlib import Path
import os
import pickle
import io

from teadata.teadata_config import (
    canonical_campus_number,
    canonical_district_number,
    normalize_campus_number_value,
    normalize_district_number_value,
)

try:
    from platformdirs import user_cache_dir  # type: ignore
except Exception:
    user_cache_dir = None  # optional; we can still work without it


# --- Backward-compatible unpickling (remap old module paths like "classes" -> current module) ---
class _ModuleMappingUnpickler(pickle.Unpickler):
    """
    An Unpickler that remaps legacy module names to the current module path
    so that older snapshots created when this file lived at top-level 'classes.py'
    can still be loaded after packaging as 'teadata.classes'.
    """

    def __init__(self, file, module_map: dict[str, str] | None = None):
        super().__init__(file)
        self._module_map = module_map or {}

    def find_class(self, module, name):
        remapped = self._module_map.get(module, module)
        return super().find_class(remapped, name)


def _compat_pickle_load(fobj) -> object:
    """
    Try a normal pickle.load first. If it fails with ModuleNotFoundError due to moved modules,
    retry using _ModuleMappingUnpickler remapping legacy names (e.g., 'classes' -> current module).
    """
    try:
        return pickle.load(fobj)
    except ModuleNotFoundError as e:
        # Rewind and retry with mapping
        try:
            # If fobj is a real file, seek; else, copy into BytesIO
            try:
                fobj.seek(0)
                bio = fobj  # type: ignore
            except Exception:
                data = fobj.read()
                bio = io.BytesIO(data)
            current_mod = __name__  # e.g., 'teadata.classes'
            module_map = {
                "classes": current_mod,  # old snapshots pickled with top-level classes.py
                "__main__": current_mod,  # some scripts pickle while running as __main__
            }
            return _ModuleMappingUnpickler(bio, module_map).load()
        except Exception as ee:
            raise ee from e


# Global profiling flag for timeit decorator
ENABLE_PROFILING = False

# --------- Optional Shapely support (falls back to pure-Python) ---------
try:
    from shapely.geometry import (
        Point as ShapelyPoint,
        Polygon as ShapelyPolygon,
        MultiPolygon,
    )

    SHAPELY = True
except Exception:
    SHAPELY = False

# --------- Utilities ---------


def timeit(func: Callable):
    """Decorator: log how long the function took (example of AOP-style concern)."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        dt = (time.perf_counter() - t0) * 1000
        if ENABLE_PROFILING:
            print(f"[timeit] {func.__name__} took {dt:.2f} ms")
        return out

    return wrapper


def validate_non_empty_str(name: str):
    """Decorator factory: enforce non-empty string attribute on __post_init__."""

    def deco(cls):
        orig_post = getattr(cls, "__post_init__", None)

        @wraps(orig_post)
        def post(self):
            if not getattr(self, name) or not isinstance(getattr(self, name), str):
                raise ValueError(f"{cls.__name__}.{name} must be a non-empty string")
            if orig_post:
                orig_post(self)

        cls.__post_init__ = post
        return cls

    return deco


# --------- Minimal geometry helpers (when Shapely is absent) ---------


def point_in_polygon(
    point: Tuple[float, float], polygon: List[Tuple[float, float]]
) -> bool:
    """Ray casting algorithm. polygon is list of (x,y), point is (x,y)."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # check edges that straddle the horizontal ray
        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1
        ):
            inside = not inside
    return inside


def polygon_area(polygon: List[Tuple[float, float]]) -> float:
    """Shoelace formula."""
    area = 0.0
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    (x1, y1), (x2, y2) = a, b
    return math.hypot(x2 - x1, y2 - y1)


# --------- Geodesic helper (miles) ---------
def haversine_miles(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Approximate great-circle distance between two lon/lat points in miles.
    x = longitude, y = latitude (degrees).
    """
    from math import radians, sin, cos, sqrt, atan2

    R = 3958.7613  # Earth radius in miles
    lon1, lat1, lon2, lat2 = map(radians, (x1, y1, x2, y2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def _point_xy(pt: Any) -> Tuple[float, float] | None:
    """
    Extract (x, y) from a Shapely point or a tuple; return None if unavailable.
    """
    if pt is None:
        return None
    if SHAPELY:
        try:
            # Works for shapely Point
            return (pt.x, pt.y)
        except Exception:
            pass
    if (
        isinstance(pt, tuple)
        and len(pt) == 2
        and all(isinstance(v, (int, float)) for v in pt)
    ):
        return (float(pt[0]), float(pt[1]))
    return None


def _probably_lonlat(x: float, y: float) -> bool:
    # crude heuristic: lon in [-180, 180], lat in [-90, 90]
    return -180.0 <= x <= 180.0 and -90.0 <= y <= 90.0


# --------- Descriptor to enforce "geometry-like" fields ---------


class GeoField:
    """Descriptor that accepts a Shapely geometry or pure-Python fallback types."""

    def __init__(self, geom_type: str):
        self.geom_type = geom_type
        self.private_name = None

    def __set_name__(self, owner, name):
        self.private_name = f"_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private_name, None)

    def __set__(self, obj, value):
        # Accept Shapely or tuples/lists
        if self.geom_type == "point":
            ok = (SHAPELY and isinstance(value, ShapelyPoint)) or (
                isinstance(value, tuple)
                and len(value) == 2
                and all(isinstance(v, (int, float)) for v in value)
            )
        elif self.geom_type == "polygon":
            ok = (SHAPELY and isinstance(value, (ShapelyPolygon, MultiPolygon))) or (
                isinstance(value, list)
                and all(isinstance(p, tuple) and len(p) == 2 for p in value)
            )
        else:
            ok = False
        if not ok:
            raise TypeError(
                f"Invalid {self.geom_type} geometry for {obj.__class__.__name__}"
            )
        setattr(obj, self.private_name, value)


# --------- Domain Models ---------


@validate_non_empty_str("name")
@dataclass(slots=True)
class District:
    id: uuid.UUID
    name: str
    enrollment: Optional[int] = None
    district_number: str = ""  # public, with apostrophe for display/export
    aea: Optional[bool] = None
    rating: Optional[str] = None

    # hidden canonical
    _district_number_canon: str = field(init=False, repr=False, compare=False)

    boundary: Any = field(
        default=None, repr=False
    )  # handled by GeoField descriptor below
    _polygon: Any = field(init=False, repr=False, default=None)
    _prepared: Any = field(default=None, init=False, repr=False, compare=False)

    # plug in descriptor (must be defined after dataclass attributes exist)
    polygon = GeoField("polygon")

    # Arbitrary enrichment payload (columns we don’t model explicitly)
    meta: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    _repo: "DataEngine" = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        # route boundary to descriptor-backed 'polygon' for validation
        if self.boundary is not None:
            self.polygon = self.boundary

        # 1) derive canonical key (no apostrophe)
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

    # --- Operator overloading ideas ---

    def __contains__(self, campus: "Campus") -> bool:
        """Enable: campus in district  (point-in-polygon)."""
        if self.polygon is None or campus.point is None:
            return False
        if SHAPELY:
            return self.polygon.contains(campus.point)
        return point_in_polygon(campus.point, list(self.polygon))

    def __and__(self, other: "District") -> float:
        """
        Intersection area between two districts.
        Returns area (arbitrary units) so you can sort/filter by overlap magnitude.
        """
        if self.polygon is None or other.polygon is None:
            return 0.0
        if SHAPELY:
            return self.polygon.intersection(other.polygon).area
        # crude: approximate overlap via polygon area if identical; else 0 (no robust poly clipping here)
        return (
            polygon_area(list(self.polygon)) if self.polygon == other.polygon else 0.0
        )

    def __or__(self, other: "District") -> float:
        """Union area (handy for Jaccard-like indices)."""
        if self.polygon is None or other.polygon is None:
            return 0.0
        if SHAPELY:
            return self.polygon.union(other.polygon).area
        # crude union fallback
        if self.polygon == other.polygon:
            return polygon_area(list(self.polygon))
        return polygon_area(list(self.polygon)) + polygon_area(list(other.polygon))

    def __matmul__(self, campus: "Campus") -> float:
        """
        Overload @ for 'distance to district' (0 if inside).
        Enables: district @ campus
        """
        if self.polygon is None or campus.point is None:
            return float("inf")
        if SHAPELY:
            return self.polygon.distance(campus.point)
        # fallback: min vertex distance if outside; 0 if inside
        if campus in self:
            return 0.0
        verts = list(self.polygon)
        return min(euclidean(campus.point, v) for v in verts)

    def __format__(self, spec: str) -> str:
        """Custom formatting, e.g., f'{d:brief}'."""
        if spec == "brief":
            return f"{self.name} (enr={self.enrollment:,}, rating={self.rating or '-'})"
        return f"District<{self.name}>"

    # pattern matching convenience (match by name/enrollment)
    __match_args__ = ("name", "enrollment")

    # Convenience accessor for internal logic
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
        """
        All campuses whose district_id == this district's id, using the engine's index.
        Returns an empty EntityList if the district is not attached to a DataEngine.
        """
        repo = getattr(self, "_repo", None)
        if repo is None:
            return EntityList()
        ids = repo._campuses_by_district.get(self.id, [])
        return EntityList([repo._campuses[cid] for cid in ids])

    # --------- Prepared polygon for fast point-in-polygon checks (Shapely only) ---------
    if SHAPELY:
        from shapely.prepared import prep as shapely_prep  # type: ignore

    @property
    def prepared(self):
        """Prepared polygon for fast point-in-polygon checks (Shapely only). Works with slots."""
        if not SHAPELY or self.polygon is None:
            return None
        if self._prepared is not None:
            return self._prepared
        try:
            self._prepared = self.shapely_prep(self.polygon)
        except Exception:
            self._prepared = None
        return self._prepared

    def __getattr__(self, name: str):
        """
        Allow dot-access for enrichment fields stored in meta:
        e.g., dist.overall_rating_2025 -> dist.meta['overall_rating_2025']
        """
        meta = object.__getattribute__(self, "meta")
        if isinstance(meta, dict) and name in meta:
            return meta[name]
        raise AttributeError(f"{self.__class__.__name__} has no attribute {name!r}")

    def __dir__(self):
        # makes tab-completion show meta-backed keys too
        base = list(super().__dir__())
        m = object.__getattribute__(self, "meta")
        if isinstance(m, dict):
            base.extend(k for k in m.keys() if isinstance(k, str))
        return sorted(set(base))

    def to_dict(
        self, *, include_meta: bool = True, include_geometry: bool = False
    ) -> dict:
        """
        Serialize this District into a plain dict suitable for DataFrame/JSON.

        Args:
            include_meta: include enrichment fields stored in self.meta
            include_geometry: include simple geometry representation (bounds; and WKT if available)

        Notes:
            Geometry is summarized to avoid heavy objects in JSON/DFs.
        """
        out = {
            "id": str(self.id),
            "name": self.name,
            "enrollment": self.enrollment,
            "district_number": self.district_number,
            "aea": self.aea,
            "rating": self.rating,
        }
        if include_meta and isinstance(self.meta, dict):
            # Do not overwrite canonical keys
            for k, v in self.meta.items():
                if k not in out:
                    out[k] = v

        if include_geometry:
            poly = getattr(self, "polygon", None)
            try:
                if poly is not None:
                    # Prefer shapely attributes if present
                    b = poly.bounds if hasattr(poly, "bounds") else None
                    out["geometry_bounds"] = tuple(b) if b else None
                    if hasattr(poly, "wkt"):
                        out["geometry_wkt"] = poly.wkt
            except Exception:
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
    enrollment: Optional[int] = None
    rating: Optional[str] = None

    aea: Optional[bool] = None
    grade_range: Optional[str] = None
    school_type: Optional[str] = None
    school_status_date: Optional[date] = None
    update_date: Optional[date] = None

    district_number: Optional[str] = None
    campus_number: Optional[str] = None

    _district_number_canon: Optional[str] = field(init=False, repr=False, compare=False)
    _campus_number_canon: Optional[str] = field(init=False, repr=False, compare=False)

    _repo: "DataEngine" = field(default=None, repr=False, compare=False)

    location: Any = field(
        default=None, repr=False
    )  # handled by GeoField descriptor below
    _point: Any = field(init=False, repr=False, default=None)

    point = GeoField("point")

    # Arbitrary enrichment payload (columns not explicitly modeled)
    meta: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if self.location is not None:
            self.point = self.location

        # District
        raw_dn = self.district_number
        cdn = normalize_district_number_value(raw_dn)
        self._district_number_canon = cdn
        if cdn:
            can = canonical_district_number(raw_dn)
            self.district_number = can or None
        else:
            self.district_number = None

        # Campus
        raw_cn = self.campus_number
        cdc = normalize_campus_number_value(raw_cn)
        self._campus_number_canon = cdc
        if cdc:
            can = canonical_campus_number(raw_cn)
            self.campus_number = can or None
        else:
            self.campus_number = None

    def __getattr__(self, name: str):
        # Allow dot-access for enrichment fields stored in meta
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

    def transfers_out(self) -> "EntityList":
        """List of destination campuses this campus sends students to (ignores suppressed counts)."""
        if not hasattr(self, "_repo") or self._repo is None:
            return EntityList([])
        rows = [c for (c, cnt, masked) in self._repo.transfers_out(self)]
        return EntityList(rows)

    def transfers_out_edges(self) -> list[tuple["Campus", Optional[int], bool]]:
        """Edges as (to_campus, count, masked)."""
        if not hasattr(self, "_repo") or self._repo is None:
            return []
        return self._repo.transfers_out(self)

    def to_dict(
        self, *, include_meta: bool = True, include_geometry: bool = False
    ) -> dict:
        """
        Serialize this Campus into a plain dict suitable for DataFrame/JSON.

        Args:
            include_meta: include enrichment fields stored in self.meta
            include_geometry: include simple geometry representation (lon/lat; WKT if available)
        """
        out = {
            "id": str(self.id),
            "district_id": str(self.district_id) if self.district_id else None,
            "name": self.name,
            "charter_type": self.charter_type,
            "is_charter": self.is_charter,
            "enrollment": self.enrollment,
            "rating": self.rating,
            "aea": self.aea,
            "grade_range": self.grade_range,
            "school_type": self.school_type,
            "school_status_date": (
                self.school_status_date.isoformat()
                if hasattr(self.school_status_date, "isoformat")
                else self.school_status_date
            ),
            "update_date": (
                self.update_date.isoformat()
                if hasattr(self.update_date, "isoformat")
                else self.update_date
            ),
            "district_number": self.district_number,
            "campus_number": self.campus_number,
        }
        if include_meta and isinstance(self.meta, dict):
            for k, v in self.meta.items():
                if k not in out:
                    out[k] = v

        if include_geometry:
            pt = getattr(self, "point", None) or getattr(self, "location", None)
            try:
                if pt is not None:
                    out["lon"] = float(pt.x) if hasattr(pt, "x") else float(pt[0])
                    out["lat"] = float(pt.y) if hasattr(pt, "y") else float(pt[1])
                    if hasattr(pt, "wkt"):
                        out["geometry_wkt"] = pt.wkt
            except Exception:
                out["lon"] = out["lat"] = None
        return out

    @property
    def district_number_canon(self) -> Optional[str]:
        return self._district_number_canon

    @property
    def campus_number_canon(self) -> Optional[str]:
        return self._campus_number_canon

    @property
    def percent_enrollment_change(self) -> float:
        """
        Compute the absolute percent change in enrollment from the 2015 baseline.
        Uses the 'campus_2015_student_enrollment_all_students_count' field from meta if present,
        or from the object directly if available.
        Returns:
            float: abs((self.enrollment - baseline) / baseline)
        Raises:
            ValueError: if enrollment or baseline is missing or None.
        """
        enrollment = self.enrollment
        # Try meta first, then direct attribute as fallback
        baseline = None
        if isinstance(self.meta, dict) and "campus_2015_student_enrollment_all_students_count" in self.meta:
            baseline = self.meta["campus_2015_student_enrollment_all_students_count"]
        elif hasattr(self, "campus_2015_student_enrollment_all_students_count"):
            baseline = getattr(self, "campus_2015_student_enrollment_all_students_count")
        if enrollment is None:
            raise ValueError("Current enrollment is missing or None.")
        if baseline is None:
            raise ValueError("Baseline 2015 enrollment is missing or None.")
        try:
            enrollment = float(enrollment)
            baseline = float(baseline)
        except Exception:
            raise ValueError("Enrollment or baseline could not be cast to float.")
        if baseline == 0:
            raise ValueError("Baseline 2015 enrollment is zero, cannot compute percent change.")
        return abs((enrollment - baseline) / baseline)

    @property
    def district(self) -> Optional["District"]:
        if self._repo is None:
            return None
        return self._repo._districts.get(self.district_id)

    @property
    def coords(self) -> tuple[float, float]:
        return (self.point.x, self.point.y) if self.point else None

    # Rich comparisons: sort campuses by enrollment descending
    def __lt__(self, other: "Campus") -> bool:
        return self.enrollment > other.enrollment

    def __sub__(self, other: "Campus | Tuple[float, float]") -> float:
        """
        Overload '-' for distance. Enables:
          campus1 - campus2   OR   campus - (x, y)
        """
        return self.distance_to(other)

    @singledispatchmethod
    def distance_to(self, other) -> float:
        raise TypeError("unsupported type for distance")

    def _distance_to_campus(self, other: "Campus") -> float:
        if self.point is None or other.point is None:
            return float("inf")
        if SHAPELY:
            return self.point.distance(other.point)
        return euclidean(self.point, other.point)

    @distance_to.register
    def _(self, other: tuple) -> float:
        if self.point is None:
            return float("inf")
        if SHAPELY and isinstance(self.point, ShapelyPoint):
            return self.point.distance(ShapelyPoint(*other))
        return euclidean(self.point, other)


# Register the Campus-specific overload *after* the class is created to avoid
# forward-reference issues with singledispatchmethod on Python 3.12.
Campus.distance_to.register(Campus)(Campus._distance_to_campus)


# --------- Helper: unwrap Query to its first item ---------
# --------- Helper: unwrap Query to its first item ---------
def _unwrap_query(obj: Any) -> Any:
    """
    If obj is a Query, return its first() item; otherwise return obj unchanged.
    Intended to make repo methods tolerant of Query inputs.
    """
    try:
        if isinstance(obj, Query):
            return obj.first()
    except NameError:
        # Query not defined yet (during type checking), just return as-is
        pass
    return obj


# --------- Snapshot discovery helpers ---------
def _iter_parents(start: Path):
    """Yield start and its parents up to filesystem root."""
    cur = start.resolve()
    yielded = set()
    while True:
        if cur in yielded:
            break
        yielded.add(cur)
        yield cur
        if cur.parent == cur:
            break
        cur = cur.parent


def _newest_pickle(folder: Path) -> Optional[Path]:
    try:
        if not folder.exists() or not folder.is_dir():
            return None
        picks = list(folder.glob("*.pkl"))
        if not picks:
            return None
        picks.sort(key=lambda pp: pp.stat().st_mtime, reverse=True)
        return picks[0]
    except Exception:
        return None


def _discover_snapshot(explicit: str | Path | None = None) -> Optional[Path]:
    """
    Find a .pkl snapshot to bootstrap the repo, searching in this priority order:
      1) explicit path argument
      2) TEADATA_SNAPSHOT environment variable
      3) the package's own '.cache' directory (inner library dir)
      4) .cache/*.pkl in CWD and its parents (repo-root style)
      5) platform-specific user cache dir (platformdirs), if available

    The “inner library dir” is resolved from this module file:
        Path(__file__).resolve().parent / ".cache"
    This is prioritized so that when the package includes its own snapshot,
    it is selected over any older copies in the outer repo directory.
    """
    # 1) explicit argument
    if explicit:
        p = Path(explicit)
        if p.exists() and p.is_file():
            return p

    # 2) environment variable
    env = os.environ.get("TEADATA_SNAPSHOT")
    if env:
        p = Path(env)
        if p.exists() and p.is_file():
            return p

    # 3) package (inner library) .cache first
    try:
        package_dir = Path(__file__).resolve().parent
        pkg_cache = package_dir / ".cache"
        p = _newest_pickle(pkg_cache)
        if p is not None:
            return p
    except Exception:
        pass

    # 4) walk up from CWD for .cache/*.pkl
    for base in _iter_parents(Path.cwd()):
        p = _newest_pickle(base / ".cache")
        if p is not None:
            return p

    # 5) user cache dir (platformdirs)
    if user_cache_dir:
        try:
            ucache = Path(user_cache_dir("teadata", "adpena"))
            p = _newest_pickle(ucache)
            if p is not None:
                return p
        except Exception:
            pass

    return None


# --------- Rich mapping containers (pandas-like helpers) ---------


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
        except Exception as e:
            raise ImportError(
                "pandas is required for .to_df(); install pandas to use this feature"
            ) from e
        data = self.to_dicts(
            include_meta=include_meta, include_geometry=include_geometry
        )
        df = pd.DataFrame(data)
        if columns is not None:
            cols = [c for c in columns if c in df.columns]
            df = df[cols]
        return df

    """
    Dict keyed by UUID with District/Campus values, with pandas-like helpers.
    You can call .unique() directly on these maps.
    """

    def unique(self, attr, *, dropna: bool = True, sort: bool = True):
        """
        Return unique values of an attribute across all entities in this map.

        - attr can be a string attribute name (canonical or enriched via .meta),
          or a callable taking the entity and returning a value.
        - dropna: skip None values
        - sort: return a sorted list
        """
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
                # heterogeneous types may not sort; leave as-is
                pass
        return out

    def value_counts(
        self, attr, *, dropna: bool = True, sort: bool = True, descending: bool = True
    ):
        """
        Return counts of values for an attribute across all entities in this map.

        - attr can be a string attribute name (canonical or enriched via .meta),
          or a callable taking the entity and returning a value.
        - dropna: skip None values
        - sort: return a sorted list (by count desc/asc depending on `descending`)

        Returns:
            List[Tuple[value, count]] if sort=True, else a dict[value] = count
        """
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
        """
        Convert this collection of entities into a list of dicts by calling .to_dict() on each.
        Falls back to vars(obj) if an entity lacks .to_dict().
        """
        rows = []
        for obj in self:
            if hasattr(obj, "to_dict"):
                rows.append(
                    obj.to_dict(
                        include_meta=include_meta, include_geometry=include_geometry
                    )
                )
            else:
                # basic fallback
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
        """
        Return a pandas DataFrame for this collection. Requires pandas.
        """
        try:
            import pandas as pd  # type: ignore
        except Exception as e:
            raise ImportError(
                "pandas is required for .to_df(); install pandas to use this feature"
            ) from e
        data = self.to_dicts(
            include_meta=include_meta, include_geometry=include_geometry
        )
        df = pd.DataFrame(data)
        if columns is not None:
            # Only keep requested columns that actually exist
            cols = [c for c in columns if c in df.columns]
            df = df[cols]
        return df

    """
    A thin wrapper around list to add Pandas-like helpers for District and Campus collections.
    """

    def unique(self, attr):
        """
        Return sorted unique values of an attribute across all objects in the list.
        Example:
            districts.unique("rating")
        """
        if callable(attr):
            getter = attr
        else:
            getter = lambda o: getattr(o, attr, None)
        values = [getter(obj) for obj in self]
        return sorted(set(v for v in values if v is not None))

    def value_counts(
        self, attr, *, dropna: bool = True, sort: bool = True, descending: bool = True
    ):
        """
        Return counts of values for an attribute across all objects in the list.
        Returns a list of (value, count) pairs by default (sorted by count).
        """
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
        """Return the first n elements, like pandas.DataFrame.head()."""
        return self[:n]

    def sample(self, n=5, seed=None):
        """Random sample of n elements."""
        import random

        rng = random.Random(seed)
        return rng.sample(self, min(n, len(self)))


class ReadOnlyEntityView:
    """
    Read-only view that exposes mapping operations and EntityMap helpers (e.g., .unique()).
    Prevents accidental mutation while keeping a nice, pandas-like API at the repo surface:

        repo.districts.unique("rating")
        repo.campuses.unique(lambda c: c.charter_type)
    """

    __slots__ = ("_m",)

    def __init__(self, backing: EntityMap):
        self._m = backing

    # --- read-only mapping protocol ---
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

    # --- helpers mirrored from EntityMap ---
    def unique(self, *args, **kwargs):
        return self._m.unique(*args, **kwargs)

    def value_counts(self, *args, **kwargs):
        return self._m.value_counts(*args, **kwargs)

    # Let hasattr()/dir() discover helper methods cleanly
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


# --------- Chainable query wrapper for repo results ---------
class Query:
    def to_dicts(
        self, *, include_meta: bool = True, include_geometry: bool = False
    ) -> list[dict]:
        """
        Materialize the current query items as a list of dicts suitable for DataFrame/JSON.

        This method is tuple-aware:
        - (Campus, Campus|None, miles)   from ("nearest_charter_same_type",) → "campus_*", "match_*", "distance_miles"
        - (Campus, Campus|None, count, masked) from ("transfers_out",)       → "campus_*", "to_*", "count", "masked"
        - Any other tuple: each element is flattened with prefixes "p0_", "p1_", ...
        """

        def _obj_to_basic_dict(obj, *, prefix: str = "") -> dict:
            # Prefer each object's own serializer if available
            if hasattr(obj, "to_dict"):
                d = obj.to_dict(
                    include_meta=include_meta, include_geometry=include_geometry
                )
            else:
                try:
                    d = dict(vars(obj))
                except Exception:
                    # Fallback to a minimal representation
                    d = {"value": obj}
            if prefix:
                return {f"{prefix}{k}": v for k, v in d.items()}
            return d

        rows = []
        for item in self._items:
            # Special-case: results from >> ("nearest_charter_same_type",)

            if isinstance(item, tuple):
                # Detect the nearest-charter tuple signature robustly
                is_ncst = (
                    len(item) == 3
                    and (Campus is not None)
                    and (getattr(item[0].__class__, "__name__", "") == "Campus")
                    and (
                        item[1] is None
                        or getattr(item[1].__class__, "__name__", "") == "Campus"
                    )
                )
                if is_ncst:
                    campus, match, miles = item
                    row = {}
                    row.update(_obj_to_basic_dict(campus, prefix="campus_"))
                    if match is not None:
                        row.update(_obj_to_basic_dict(match, prefix="match_"))
                    else:
                        # Provide consistent keys for the "match_" side when missing
                        row["match_id"] = None
                        row["match_name"] = None
                        row["match_enrollment"] = None
                        row["match_rating"] = None
                        row["match_school_type"] = None
                        row["match_district_number"] = None
                        row["match_campus_number"] = None
                    row["distance_miles"] = miles
                    rows.append(row)
                    continue

                # Special-case: results from >> ("transfers_out",)
                # Shape: (Campus, Campus|None, count, masked)
                is_transfers = (
                    len(item) == 4
                    and (getattr(item[0].__class__, "__name__", "") == "Campus")
                    and (
                        item[1] is None
                        or getattr(item[1].__class__, "__name__", "") == "Campus"
                    )
                )
                if is_transfers:
                    campus, to_campus, count, masked = item
                    row = {}
                    row.update(_obj_to_basic_dict(campus, prefix="campus_"))
                    if to_campus is not None:
                        row.update(_obj_to_basic_dict(to_campus, prefix="to_"))
                    else:
                        # Provide consistent keys when the destination campus is missing
                        row["to_id"] = None
                        row["to_name"] = None
                        row["to_enrollment"] = None
                        row["to_rating"] = None
                        row["to_school_type"] = None
                        row["to_district_number"] = None
                        row["to_campus_number"] = None
                    row["count"] = count
                    row["masked"] = masked
                    rows.append(row)
                    continue

                # Generic tuple flattening: prefix p0_, p1_, ...
                row = {}
                for i, elem in enumerate(item):
                    row.update(_obj_to_basic_dict(elem, prefix=f"p{i}_"))
                rows.append(row)
                continue

            # Non-tuple items use the normal single-object path
            rows.append(_obj_to_basic_dict(item))

        return rows

    def to_df(
        self,
        columns: list[str] | None = None,
        *,
        include_meta: bool = True,
        include_geometry: bool = False,
    ):
        """
        Materialize the current query items as a pandas DataFrame (requires pandas).

        Tuple rows are auto-flattened with prefixes (e.g., "campus_*", "match_*", or "p0_*")
        to avoid column collisions.
        """
        try:
            import pandas as pd  # type: ignore
        except Exception as e:
            raise ImportError(
                "pandas is required for .to_df(); install pandas to use this feature"
            ) from e
        data = self.to_dicts(
            include_meta=include_meta, include_geometry=include_geometry
        )
        df = pd.DataFrame(data)
        if columns is not None:
            cols = [c for c in columns if c in df.columns]
            df = df[cols]
        return df

    """
    Lightweight, chainable view over a list of repo objects (Districts/Campuses).
    Enables: repo >> ("nearest_charter", (x,y), 200, 25) >> ("filter", pred) >> ("take", 5)
    """

    def __init__(self, items: List[Any], repo: "DataEngine"):
        self._items = list(items)
        self._repo = repo

    # iteration so list(...) and comprehensions work
    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def to_list(self) -> List[Any]:
        return list(self._items)

    def all(self) -> List[Any]:
        return list(self._items)

    def first(self) -> Optional[Any]:
        return self._items[0] if self._items else None

    def __getattr__(self, name: str):
        """
        Delegate missing attributes to the first item in the query (if any).
        Enables: dist_q.name, dist_q.polygon, dist_q.some_method(...)
        """
        if not self._items:
            raise AttributeError(
                f"'Query' object has no attribute '{name}' (empty result set)"
            )
        return getattr(self._items[0], name)

    def __getitem__(self, idx: int) -> Any:
        """Index into the underlying list (e.g., dist_q[0])."""
        return self._items[idx]

    def __bool__(self) -> bool:
        """Truthiness reflects whether the query has any items."""
        return bool(self._items)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        n = len(self._items)
        sample = self._items[0].__class__.__name__ if n else "None"
        return f"<{cls} len={n} first={sample}>"

    def __rshift__(self, op):
        """
        Support a small pipeline DSL on top of an existing item list.

        - ("filter", predicate) -> keep items where predicate(item) is True
        - ("take", n)           -> first n items
        - ("sort", keyfunc, reverse=False) -> sort in place by key
        - ("map", func)         -> map items (returns list of mapped values)
        - ("where", predicate)    -> alias of ("filter", predicate)
        - ("select", func)        -> alias of ("map", func)
        - callable              -> predicate filter (same as ("filter", callable))
        - ("nearest", (x,y|None), n?, max_miles?)           -> restart pipeline with nearest campuses; if (x,y) is None, infers from current items
        - ("nearest_charter", (x,y|None), n?, max_miles?)   -> restart with nearest charter campuses; if (x,y) is None, infers from current items
        - ("campuses_in",)      -> expand current District items into their campuses (chainable)
        - ("within", target_or_None, charter_only=False, covers=False) -> campuses within a polygon/district (infers target from chain if None)
        - ("radius", (lon, lat)|None, miles, limit=None, charter_only=False) -> campuses within miles (fast with KDTree; infers coords if None)
        - ("knn", (lon, lat)|None, k, charter_only=False) -> k nearest campuses (fast with KDTree; infers coords if None)
        - Attribute access on Query delegates to its first item (e.g., `dist_q.name`).
        """
        # Predicate-only filter
        if callable(op):
            self._items = [o for o in self._items if op(o)]
            return self

        if not (isinstance(op, tuple) and len(op) >= 1):
            raise ValueError(f"Unsupported >> operation on Query: {op!r}")

        key = op[0]

        if key == "filter":
            pred = op[1]
            self._items = [o for o in self._items if pred(o)]
            return self

        if key == "where":  # alias of filter
            pred = op[1]
            self._items = [o for o in self._items if pred(o)]
            return self

        if key == "take":
            n = int(op[1])
            self._items = self._items[:n]
            return self

        if key == "sort":
            keyfunc = op[1]
            reverse = bool(op[2]) if len(op) >= 3 else False
            self._items.sort(key=keyfunc, reverse=reverse)
            return self

        if key == "map":
            func = op[1]
            # mapping returns a *plain list*, not a Query, to allow terminal projections
            return [func(o) for o in self._items]

        if key == "select":  # alias of map
            func = op[1]
            return [func(o) for o in self._items]

        # New: campuses_in op expands Districts into their campuses
        if key == "campuses_in":
            # Expand current District items into their campuses
            campuses: List[Any] = []
            for item in self._items:
                # If item is a District, pull campuses; if already Campus, keep it
                if hasattr(item, "id") and item.__class__.__name__ == "District":
                    campuses.extend(self._repo.campuses_in(item))
                elif item.__class__.__name__ == "Campus":
                    campuses.append(item)
            self._items = campuses
            return self

        # New: distinct operator
        if key == "distinct":
            keyfunc = op[1]
            seen = set()
            out = []
            for o in self._items:
                k = keyfunc(o)
                if k in seen:
                    continue
                seen.add(k)
                out.append(o)
            self._items = out
            return self

        # Restart-style ops that source new items from the repo
        if key in {"nearest", "nearest_charter"}:
            coords = op[1] if len(op) >= 2 else None
            n = int(op[2]) if len(op) >= 3 else 1
            max_m = float(op[3]) if len(op) >= 4 and op[3] is not None else None
            charter = key == "nearest_charter"

            # Infer coordinates if None:
            if coords is None and self._items:
                seed = self._items[0]
                # Try Campus first
                if hasattr(seed, "point") and seed.point is not None:
                    xy = _point_xy(seed.point)
                    if xy:
                        coords = xy
                # Try District centroid (Shapely)
                elif (
                    hasattr(seed, "polygon")
                    and seed.polygon is not None
                    and SHAPELY
                    and hasattr(seed.polygon, "centroid")
                ):
                    cent = seed.polygon.centroid
                    coords = (cent.x, cent.y)

            if coords is None:
                raise ValueError(
                    "nearest/nearest_charter requires coordinates or a seed item with geometry in the chain"
                )

            items = self._repo.nearest_campuses(
                coords[0],
                coords[1],
                limit=n,
                charter_only=charter,
                max_miles=max_m,
                geodesic=True,
            )
            self._items = items
            return self

        # Spatial containment: campuses within polygon/district
        if key == "within":
            target = op[1] if len(op) >= 2 else None
            charter_only = bool(op[2]) if len(op) >= 3 else False
            covers = bool(op[3]) if len(op) >= 4 else False

            # Infer target from current items if None (e.g., District in chain)
            if target is None and self._items:
                seed = self._items[0]
                if hasattr(seed, "polygon"):
                    target = seed

            # Resolve polygon geometry
            poly = None
            if target is not None:
                poly = getattr(target, "polygon", None) or getattr(
                    target, "boundary", None
                )

            if poly is None:
                raise ValueError(
                    "within requires a polygon/district target or a District earlier in the chain"
                )

            # If current items are empty, source candidates smartly using STRtree (Shapely 2.x) or bbox
            items: List[Any] = []
            candidates = None

            if SHAPELY and not self._items:
                # Try STRtree first
                self._repo._ensure_point_strtree()
                if self._repo._point_tree is not None:
                    try:
                        idxs = self._repo._point_tree.query(
                            poly, predicate="covers", return_indices=True
                        )
                        if isinstance(idxs, tuple):
                            _, idxs = idxs
                        idxs = list(map(int, idxs)) if idxs is not None else []
                    except TypeError:
                        idxs = []
                    except Exception:
                        idxs = []
                    if idxs:
                        cand_ids = [self._repo._point_ids[i] for i in idxs]
                        candidates = [self._repo._campuses[cid] for cid in cand_ids]

                # If STRtree had no hits, try cheap AABB prefilter
                if candidates is None:
                    bbox_cands = self._repo._bbox_candidates(poly)
                    if bbox_cands:
                        candidates = bbox_cands

            if candidates is None:
                candidates = list(self._items) if self._items else []

            for c in candidates:
                if charter_only and not getattr(c, "is_charter", False):
                    continue
                loc = getattr(c, "point", None) or getattr(c, "location", None)
                if loc is None:
                    continue
                try:
                    ok = (
                        (poly.covers(loc) if SHAPELY else False)
                        if covers or True
                        else (loc.within(poly) if SHAPELY else False)
                    )
                except Exception:
                    ok = False
                if (
                    not SHAPELY
                    and isinstance(poly, list)
                    and isinstance(loc, tuple)
                    and len(loc) == 2
                ):
                    ok = point_in_polygon(loc, poly)
                if ok:
                    items.append(c)

            # If fast path produced zero results and we didn't have a seed list, double-check with a direct scan
            if not items and SHAPELY and not self._items:
                slow: List[Any] = []
                for c in self._repo._campuses.values():
                    if charter_only and not getattr(c, "is_charter", False):
                        continue
                    loc = getattr(c, "point", None) or getattr(c, "location", None)
                    if loc is None:
                        continue
                    try:
                        if poly.covers(loc):
                            slow.append(c)
                    except Exception:
                        pass
                if ENABLE_PROFILING:
                    print(
                        f"[sanity] Query>>within fast=0 slow={len(slow)} — using {'slow' if slow else 'fast'} path"
                    )
                if slow:
                    items = slow

            self._items = items
            return self

        # Radius and KNN queries (fast when KDTree is available; robust fallback otherwise)
        if key == "radius":
            coords = op[1] if len(op) >= 2 else None
            miles = float(op[2]) if len(op) >= 3 else 1.0
            limit = int(op[3]) if len(op) >= 4 and op[3] is not None else None
            charter_only = bool(op[4]) if len(op) >= 5 else False

            # Infer coordinates if None from current items (Campus or District centroid)
            if coords is None and self._items:
                seed = self._items[0]
                if hasattr(seed, "point") and seed.point is not None:
                    xy = _point_xy(seed.point)
                    if xy:
                        coords = xy
                elif (
                    hasattr(seed, "polygon")
                    and seed.polygon is not None
                    and SHAPELY
                    and hasattr(seed.polygon, "centroid")
                ):
                    cent = seed.polygon.centroid
                    coords = (cent.x, cent.y)

            if coords is None:
                raise ValueError(
                    "radius requires coordinates or a seed item with geometry in the chain"
                )

            items = self._repo.radius_campuses(
                coords[0], coords[1], miles, charter_only=charter_only, limit=limit
            )
            self._items = items
            return self

        if key == "knn":
            coords = op[1] if len(op) >= 2 else None
            k = int(op[2]) if len(op) >= 3 else 1
            charter_only = bool(op[3]) if len(op) >= 4 else False

            # Infer coordinates if None
            if coords is None and self._items:
                seed = self._items[0]
                if hasattr(seed, "point") and seed.point is not None:
                    xy = _point_xy(seed.point)
                    if xy:
                        coords = xy
                elif (
                    hasattr(seed, "polygon")
                    and seed.polygon is not None
                    and SHAPELY
                    and hasattr(seed.polygon, "centroid")
                ):
                    cent = seed.polygon.centroid
                    coords = (cent.x, cent.y)

            if coords is None:
                raise ValueError(
                    "knn requires coordinates or a seed item with geometry in the chain"
                )

            items = self._repo.knn_campuses(
                coords[0], coords[1], k, charter_only=charter_only
            )
            self._items = items
            return self

        if key == "nearest_charter_same_type":
            # Usage:
            #   (repo >> ("campuses_in", district)) >> ("nearest_charter_same_type", k)
            # or, starting from a District chainable:
            #   (repo >> ("district", "ALDINE ISD")) >> ("campuses_in",) >> ("nearest_charter_same_type",)
            k = int(op[1]) if len(op) >= 2 else 1

            # Determine campuses from current items
            campuses: List[Any] = []
            for item in self._items:
                if item.__class__.__name__ == "Campus":
                    campuses.append(item)
                elif item.__class__.__name__ == "District":
                    campuses.extend(self._repo.campuses_in(item))
            if not campuses:
                self._items = []
                return self

            res = self._repo.nearest_charter_same_type(campuses, k=k)

            # Represent each result as a tuple (campus, match, miles)
            out = []
            for c in campuses:
                r = res.get(str(c.id), {"match": None, "miles": None})
                out.append((c, r["match"], r["miles"]))
            self._items = out
            return self

        if key == "transfers_out":
            # Expand a list of Campus items into tuples: (campus, to_campus, count, masked)
            campuses = [
                it
                for it in self._items
                if getattr(it.__class__, "__name__", "") == "Campus"
            ]
            rows = []
            for c in campuses:
                for to_c, cnt, masked in self._repo.transfers_out(c):
                    rows.append((c, to_c, cnt, masked))
            self._items = rows
            return self

        if key == "transfers_in":
            # Expand a list of Campus items into tuples: (from_campus, campus, count, masked)
            campuses = [
                it
                for it in self._items
                if getattr(it.__class__, "__name__", "") == "Campus"
            ]
            rows = []
            for c in campuses:
                for from_c, cnt, masked in self._repo.transfers_in(c):
                    rows.append((from_c, c, cnt, masked))
            self._items = rows
            return self

        if key == "where_to":
            # Predicate applied to the 'to' campus when items are tuples (campus, to_campus, ...)
            pred = op[1]
            new_items = []
            for it in self._items:
                if isinstance(it, tuple) and len(it) >= 2:
                    to_c = it[1]
                    try:
                        if pred(to_c):
                            new_items.append(it)
                    except Exception:
                        # ignore predicate errors per-item
                        pass
            self._items = new_items
            return self

        raise ValueError(f"Unsupported Query op: {op!r}")


# --------- Repository with expressive operators ---------


class DataEngine:
    @staticmethod
    def _match_district_number(d: "District", key) -> bool:
        canon = normalize_district_number_value(key)
        if canon is None:
            return False
        stored = getattr(d, "district_number", "") or ""
        # Compare also unsuffixed variants for safety
        stored_uns = stored[1:] if stored.startswith("'") else stored
        canon_uns = canon[1:] if canon.startswith("'") else canon
        return stored == canon or stored_uns == canon_uns

    def _as_district(self, v):
        """
        Coerce v to a District instance if possible.
        Handles District, dict, or foreign class with attributes.
        Guarantees a non-empty name by deriving from common fields or using a fallback.
        """

        def _coerce_uuid_safe(x):
            try:
                return (
                    uuid.UUID(x)
                    if (x is not None and not isinstance(x, uuid.UUID))
                    else x
                )
            except Exception:
                return uuid.uuid4()

        def _derive_name_from_dict(m: dict) -> Optional[str]:
            for key in (
                "name",
                "district_name",
                "DISTNAME",
                "District",
                "DISTRICT",
                "DISTRICT_NAME",
            ):
                val = m.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            meta = m.get("meta")
            if isinstance(meta, dict):
                for key in ("name", "district_name", "DISTNAME"):
                    val = meta.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
            return None

        def _derive_name_from_obj(obj) -> Optional[str]:
            candidates = [
                getattr(obj, "name", None),
                getattr(obj, "district_name", None),
                getattr(obj, "DISTNAME", None),
            ]
            meta = getattr(obj, "meta", None)
            if isinstance(meta, dict):
                candidates.extend(
                    [meta.get("name"), meta.get("district_name"), meta.get("DISTNAME")]
                )
            for val in candidates:
                if isinstance(val, str) and val.strip():
                    return val.strip()
            return None

        # 1) Already a District
        if isinstance(v, District):
            # Ensure the name isn't empty; fix-up in place if needed
            if not (isinstance(v.name, str) and v.name.strip()):
                dn = getattr(v, "district_number", None)
                v.name = f"Unknown District {dn or v.id}"
            return v

        # 2) Mapping input
        if isinstance(v, dict):
            id_ = _coerce_uuid_safe(v.get("id"))
            dn = (
                v.get("district_number")
                or v.get("cdn")
                or v.get("CDN")
                or v.get("District Number")
            )
            name = _derive_name_from_dict(v) or f"Unknown District {dn or id_}"
            enrollment = v.get("enrollment") or 0
            aea = v.get("aea")
            rating = v.get("rating")
            # Accept either 'polygon' or 'boundary' for geometry; pass None if absent
            boundary = v.get("polygon", v.get("boundary"))
            meta = v.get("meta", {}) if isinstance(v.get("meta", {}), dict) else {}
            return District(
                id=id_,
                name=name,
                enrollment=(
                    int(enrollment) if isinstance(enrollment, (int, float)) else 0
                ),
                district_number=dn or "",
                aea=aea,
                rating=rating,
                boundary=boundary,
                meta=meta,
            )

        # 3) Foreign object with attributes
        id_ = _coerce_uuid_safe(getattr(v, "id", None))
        dn = (
            getattr(v, "district_number", None)
            or getattr(v, "cdn", None)
            or getattr(v, "CDN", None)
        )
        name = _derive_name_from_obj(v) or f"Unknown District {dn or id_}"
        enrollment = getattr(v, "enrollment", 0) or 0
        aea = getattr(v, "aea", None)
        rating = getattr(v, "rating", None)
        boundary = (
            getattr(v, "polygon", None)
            or getattr(v, "boundary", None)
            or getattr(v, "geometry", None)
        )
        meta = (
            getattr(v, "meta", {}) if isinstance(getattr(v, "meta", {}), dict) else {}
        )

        return District(
            id=id_,
            name=name,
            enrollment=int(enrollment) if isinstance(enrollment, (int, float)) else 0,
            district_number=dn or "",
            aea=aea,
            rating=rating,
            boundary=boundary,
            meta=meta,
        )

    def _as_campus(self, v):
        """
        Coerce v to a Campus instance if possible.
        Handles Campus, dict, or foreign class with attributes.
        """
        if isinstance(v, Campus):
            return v
        # Dict input
        if isinstance(v, dict):
            id = v.get("id")
            try:
                id = (
                    uuid.UUID(id)
                    if not isinstance(id, uuid.UUID) and id is not None
                    else id
                )
            except Exception:
                pass
            district_id = v.get("district_id")
            try:
                district_id = (
                    uuid.UUID(district_id)
                    if not isinstance(district_id, uuid.UUID)
                    and district_id is not None
                    else district_id
                )
            except Exception:
                pass
            name = v.get("name")
            enrollment = v.get("enrollment")
            charter_type = v.get("charter_type")
            is_charter = v.get("is_charter")
            rating = v.get("rating")
            aea = v.get("aea")
            grade_range = v.get("grade_range")
            school_type = v.get("school_type")
            school_status_date = v.get("school_status_date")
            update_date = v.get("update_date")
            district_number = v.get("district_number", "")
            campus_number = v.get("campus_number", "")
            # Accept either 'point' or 'location' for geometry
            location = v.get("point", v.get("location"))
            meta = v.get("meta", {})
            return Campus(
                id=id,
                district_id=district_id,
                name=name,
                enrollment=enrollment,
                charter_type=charter_type,
                is_charter=is_charter,
                rating=rating,
                aea=aea,
                grade_range=grade_range,
                school_type=school_type,
                school_status_date=school_status_date,
                update_date=update_date,
                district_number=district_number,
                campus_number=campus_number,
                location=location,
                meta=meta,
            )
        # Foreign class/object with attributes
        id = getattr(v, "id", None)
        try:
            id = (
                uuid.UUID(id)
                if not isinstance(id, uuid.UUID) and id is not None
                else id
            )
        except Exception:
            pass
        district_id = getattr(v, "district_id", None)
        try:
            district_id = (
                uuid.UUID(district_id)
                if not isinstance(district_id, uuid.UUID) and district_id is not None
                else district_id
            )
        except Exception:
            pass
        name = getattr(v, "name", None)
        enrollment = getattr(v, "enrollment", None)
        charter_type = getattr(v, "charter_type", None)
        is_charter = getattr(v, "is_charter", None)
        rating = getattr(v, "rating", None)
        aea = getattr(v, "aea", None)
        grade_range = getattr(v, "grade_range", None)
        school_type = getattr(v, "school_type", None)
        school_status_date = getattr(v, "school_status_date", None)
        update_date = getattr(v, "update_date", None)
        district_number = getattr(v, "district_number", "")
        campus_number = getattr(v, "campus_number", "")
        # Prefer 'point', else 'location'
        location = getattr(v, "point", None)
        if location is None:
            location = getattr(v, "location", None)
        meta = getattr(v, "meta", {}) if hasattr(v, "meta") else {}
        return Campus(
            id=id,
            district_id=district_id,
            name=name,
            enrollment=enrollment,
            charter_type=charter_type,
            is_charter=is_charter,
            rating=rating,
            aea=aea,
            grade_range=grade_range,
            school_type=school_type,
            school_status_date=school_status_date,
            update_date=update_date,
            district_number=district_number,
            campus_number=campus_number,
            location=location,
            meta=meta,
        )

    @classmethod
    def from_snapshot(
        cls, snapshot: str | Path | None = None, *, search: bool = True
    ) -> "DataEngine":
        """
        Load a DataEngine from a pickled snapshot (.pkl). This method **never** returns None.
        It either returns a valid DataEngine instance or raises a clear RuntimeError/TypeError.

        Snapshot discovery prioritizes the package's own `.cache` folder over outer repo folders.
        Accepts snapshot payloads saved by different builder scripts, including:
          - a full DataEngine instance
          - a dict with keys like "_districts"/"districts" and "_campuses"/"campuses"
          - a tuple produced by load_data2.py such as:
                (districts_dict, campuses_dict)
            or  (districts_dict, campuses_dict, meta_dict)
            or  lists of District/Campus instead of dicts
        """
        # Resolve the path to open
        path: Optional[Path]
        if snapshot is not None:
            path = Path(snapshot)
        else:
            path = _discover_snapshot(None) if search else None

        if path is None:
            return cls()  # empty engine (no snapshot found)

        if not path.exists() or not path.is_file():
            raise RuntimeError(f"Snapshot not found or not a file: {path}")

        try:
            with open(path, "rb") as f:
                obj = _compat_pickle_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to unpickle snapshot {path}: {e}") from e

        # Case 1: the snapshot already contains a DataEngine
        if isinstance(obj, cls):
            return obj

        # Create an instance early so we can use instance coercers
        eng = cls()

        # Helper to coerce various container shapes into {uuid -> District/Campus}
        def _coerce_entity_map(x, kind: str):
            """
            Accept a dict or list/tuple and return a dict keyed by entity.id, coercing to District or Campus.
            kind: "district" or "campus"
            """
            out = {}
            if isinstance(x, dict):
                values = list(x.values())
            elif isinstance(x, (list, tuple)):
                values = list(x)
            else:
                return out
            for v in values:
                try:
                    obj2 = (
                        eng._as_district(v) if kind == "district" else eng._as_campus(v)
                    )
                except Exception:
                    obj2 = None
                if getattr(obj2, "id", None) is not None:
                    out[obj2.id] = obj2
            return out

        # Case 2: dict payload with districts/campuses inside
        if isinstance(obj, dict):
            dmap = None
            cmap = None
            for k in ("_districts", "districts"):
                if k in obj:
                    dmap = _coerce_entity_map(obj.get(k), kind="district")
                    break
            if dmap is None:
                dmap = {}
            for k in ("_campuses", "campuses"):
                if k in obj:
                    cmap = _coerce_entity_map(obj.get(k), kind="campus")
                    break
            if cmap is None:
                cmap = {}

            with eng.bulk():
                for d in dmap.values():
                    eng.add_district(d)
                for c in cmap.values():
                    if c.district_id in eng._districts:
                        eng.add_campus(c)
            eng._rebuild_indexes()
            if os.environ.get("TEADATA_DEBUG"):
                print(
                    f"[snapshot] loaded districts={len(eng._districts)} campuses={len(eng._campuses)} from {path}"
                )
            return eng

        # Case 3: tuple/list payloads emitted by loader scripts (e.g., load_data2.py)
        # Case: tuple/list payloads (various shapes)
        if isinstance(obj, (tuple, list)):
            parts = list(obj)

            # (A) If any element is already a DataEngine instance, just use it.
            for p in parts:
                if isinstance(p, cls):
                    if os.environ.get("TEADATA_DEBUG"):
                        print(
                            f"[snapshot] tuple contains DataEngine; returning embedded engine "
                            f"with {len(p._districts)} districts / {len(p._campuses)} campuses"
                        )
                    return p

            # (B) Otherwise try to coerce “(district_map, campus_map)” style payloads
            if len(parts) >= 2:
                dmap = _coerce_entity_map(parts[0], kind="district")
                cmap = _coerce_entity_map(parts[1], kind="campus")
                with eng.bulk():
                    for d in dmap.values():
                        eng.add_district(d)
                    for c in cmap.values():
                        if c.district_id in eng._districts:
                            eng.add_campus(c)
                eng._rebuild_indexes()
                if os.environ.get("TEADATA_DEBUG"):
                    print(
                        f"[snapshot] tuple->coerced maps: districts={len(eng._districts)} "
                        f"campuses={len(eng._campuses)} from {path}"
                    )
                return eng

        # Unsupported payload type
        raise TypeError(f"Unsupported snapshot payload type: {type(obj)!r} in {path}")

    @classmethod
    def load_default(cls) -> "DataEngine":
        """
        Convenience: try to load a snapshot discovered from common locations; otherwise return a fresh engine.
        Never returns None.
        """
        return cls.from_snapshot(None, search=True)

    def __init__(self, snapshot: str | Path | None = None, *, autoload: bool = False):
        # Optional early autoload from snapshot
        if snapshot is not None or autoload:
            eng = self.from_snapshot(snapshot, search=autoload and snapshot is None)
            # If we got a different instance, copy its state into self
            if eng is not self:
                # Initialize empty containers first (will be re-assigned below)
                self._districts = EntityMap()
                self._campuses = EntityMap()
                self._campuses_by_district = defaultdict(list)

                # Transfer enrichment diagnostics: counts of rows skipped due to missing campuses
                self._xfers_missing = {"src": 0, "dst": 0, "either": 0}

                # Copy over the dictionaries
                try:
                    self._districts.update(eng._districts)
                    self._campuses.update(eng._campuses)
                    self._campuses_by_district.update(eng._campuses_by_district)

                    # Copy enrichment maps if present
                    self._xfers_out = getattr(eng, "_xfers_out", defaultdict(list))  # type: ignore
                    self._xfers_in = getattr(eng, "_xfers_in", defaultdict(list))  # type: ignore
                    self._campus_by_number = getattr(eng, "_campus_by_number", {})

                    self._xfers_missing = getattr(
                        eng, "_xfers_missing", {"src": 0, "dst": 0, "either": 0}
                    )

                except Exception:
                    pass
                # Initialize indexes as empty; they will rebuild on demand
                self._in_bulk = False
                self._kdtree = None
                self._xy_deg = None
                self._xy_rad = None
                self._campus_list = None
                self._point_tree = None
                self._point_geoms = None
                self._point_ids = None
                self._geom_id_to_index = None
                self._xy_deg_np = None
                self._campus_list_np = None
                self._all_xy_np = None
                self._all_campuses_np = None
                self._xy_to_index = None

                # Student transfer edges (optional enrichment).
                # _xfers_out: campus_id -> list[(to_campus_id, count:int|None, masked:bool)]
                # _xfers_in:  campus_id -> list[(from_campus_id, count:int|None, masked:bool)]
                self._xfers_out: Dict[
                    uuid.UUID, List[Tuple[uuid.UUID, Optional[int], bool]]
                ] = defaultdict(
                    list
                )  # type: ignore
                self._xfers_in: Dict[
                    uuid.UUID, List[Tuple[uuid.UUID, Optional[int], bool]]
                ] = defaultdict(
                    list
                )  # type: ignore

                # Fast lookup by normalized campus number (e.g., "'101902001")
                self._campus_by_number: Dict[str, uuid.UUID] = {}

                self._rebuild_indexes()
                return
        # Normal cold init (no snapshot)
        self._districts: Dict[uuid.UUID, District] = EntityMap()
        self._campuses: Dict[uuid.UUID, Campus] = EntityMap()
        self._campuses_by_district: Dict[uuid.UUID, List[uuid.UUID]] = defaultdict(list)
        self._in_bulk = False
        # Spatial/attribute indexes (optional, built on demand)
        self._kdtree = None
        self._xy_deg = None
        self._xy_rad = None
        self._campus_list = None

        # Shapely STRtree for campus points (built lazily)
        self._point_tree = None
        self._point_geoms = None
        self._point_ids = None
        self._geom_id_to_index = None

        # Vectorized fallback arrays (NumPy) for haversine when KDTree unavailable
        self._xy_deg_np = None
        self._campus_list_np = None

        # All-campus XY caches (NumPy) for bbox candidate generation
        self._all_xy_np = None
        self._all_campuses_np = None

        # Coordinate → index lists (robust legacy mapping when STRtree returns copies)
        self._xy_to_index = None  # dict[(x,y)] -> [index]

    # --- Public, read-only views of entities ---
    @property
    def districts(self):
        """
        Read-only mapping of {uuid.UUID: District} with pandas-like helpers (.unique).
        """
        return ReadOnlyEntityView(self._districts)

    @property
    def campuses(self):
        """
        Read-only mapping of {uuid.UUID: Campus} with pandas-like helpers (.unique).
        """
        return ReadOnlyEntityView(self._campuses)

    # --- Convenience name lookups ---
    def district_by_name(self, name: str) -> Optional["District"]:
        """
        Case-insensitive exact match for district name.
        Returns the first match or None.
        """
        target = (name or "").strip().lower()
        for d in self._districts.values():
            if d.name.lower() == target:
                return d
        return None

    def campus_by_name(self, name: str) -> Optional["Campus"]:
        """
        Case-insensitive exact match for campus name.
        Returns the first match or None.
        """
        target = (name or "").strip().lower()
        for c in self._campuses.values():
            if c.name.lower() == target:
                return c
        return None

    def __len__(self) -> int:
        return len(self._districts) + len(self._campuses)

    def __iter__(self):
        # iterate districts first, then campuses
        yield from self._districts.values()
        yield from self._campuses.values()

    def __getitem__(self, key: uuid.UUID) -> District | Campus:
        return self._districts.get(key) or self._campuses[key]

    def __or__(self, other: "DataEngine") -> "DataEngine":
        """Repo union: r3 = r1 | r2 (non-destructive merge)."""
        r = DataEngine()
        for d in (*self._districts.values(), *other._districts.values()):
            r.add_district(d)
        for c in (*self._campuses.values(), *other._campuses.values()):
            r.add_campus(c)
        return r

    def __rshift__(self, query):
        """
        Overloaded >> operator supports:
          1) Predicate callables (existing behavior):
             repo >> (lambda o: isinstance(o, Campus) and o.enrollment >= 2000)
             -> Query[District|Campus]

          2) Tuple-based mini-DSL:
            repo >> ("district", "'011901")        # TEA district_number
            repo >> ("district", "ALDINE ISD")      # exact name (case-insensitive)
            repo >> ("district", "ALDINE%")         # SQL-like wildcard (% and _) supported
            repo >> ("district", "ALDINE*ISD")      # glob wildcards (* and ?) supported
            repo >> ("district", 101902)            # district number as int
            repo >> ("district", "101902")          # number string (no apostrophe)
            repo >> ("district", "'101902")         # canonical apostrophe-prefixed string

             -> Query[District] (chainable; use >> "campuses_in" etc)
             Note: "district" returns a Query[District]. Most repo methods accept Query inputs by unwrapping the first item.

             repo >> ("charters_within", district)
             -> Query[Campus]
        """
        # 1) Existing behavior: predicate filter
        if callable(query):
            return Query([o for o in self if query(o)], self)

        # 2) Tuple-based queries
        if isinstance(query, tuple) and len(query) >= 2:
            key = query[0]

            if key == "district":
                raw = query[1]

                # 1) District-number lookup if the input looks number-ish (int or contains any digit)
                if isinstance(raw, int) or (
                    isinstance(raw, str) and any(ch.isdigit() for ch in raw)
                ):
                    hits = [
                        d
                        for d in self._districts.values()
                        if self._match_district_number(d, raw)
                    ]
                    return Query(hits, self)

                # 2) Name lookup (supports wildcards); case-insensitive
                target = (str(raw) if raw is not None else "").strip()
                if not target:
                    return Query([], self)

                import fnmatch

                # Allow SQL-like '%' and '_' or glob '*' and '?'
                pattern = target.replace("%", "*").replace("_", "?")
                pattern_up = pattern.upper()
                has_glob = any(ch in pattern_up for ch in ("*", "?"))

                matches = []
                for d in self._districts.values():
                    name_up = (d.name or "").upper()
                    if has_glob:
                        if fnmatch.fnmatchcase(name_up, pattern_up):
                            matches.append(d)
                    else:
                        if name_up == pattern_up:
                            matches.append(d)

                return Query(matches, self)

            if key == "campuses_in":
                # Accept a District instance or a Query[District]; unwrap if needed
                district = _unwrap_query(query[1])
                if district is None:
                    return Query([], self)
                return Query(self.campuses_in(district), self)

            if key == "charters_within":
                district = _unwrap_query(query[1])
                return Query(self.charter_campuses_within(district), self)

            if key == "nearest":
                seed = query[1]
                n = int(query[2]) if len(query) >= 3 else 1
                max_m = (
                    float(query[3])
                    if len(query) >= 4 and query[3] is not None
                    else None
                )
                # If seed is a Query or Campus, try to infer coords
                seed = _unwrap_query(seed)
                if hasattr(seed, "point"):
                    xy = _point_xy(getattr(seed, "point"))
                    if xy is not None:
                        items = self.nearest_campuses(
                            xy[0],
                            xy[1],
                            limit=n,
                            charter_only=False,
                            max_miles=max_m,
                            geodesic=True,
                        )
                        return Query(items, self)
                # Otherwise assume it's a (lon, lat) tuple
                coords = seed
                items = self.nearest_campuses(
                    coords[0],
                    coords[1],
                    limit=n,
                    charter_only=False,
                    max_miles=max_m,
                    geodesic=True,
                )
                return Query(items, self)

            if key == "nearest_charter":
                seed = query[1]
                n = int(query[2]) if len(query) >= 3 else 1
                max_m = (
                    float(query[3])
                    if len(query) >= 4 and query[3] is not None
                    else None
                )
                seed = _unwrap_query(seed)
                if hasattr(seed, "point"):
                    xy = _point_xy(getattr(seed, "point"))
                    if xy is not None:
                        items = self.nearest_campuses(
                            xy[0],
                            xy[1],
                            limit=n,
                            charter_only=True,
                            max_miles=max_m,
                            geodesic=True,
                        )
                        return Query(items, self)
                coords = seed
                items = self.nearest_campuses(
                    coords[0],
                    coords[1],
                    limit=n,
                    charter_only=True,
                    max_miles=max_m,
                    geodesic=True,
                )
                return Query(items, self)

        raise ValueError(f"Unsupported >> query: {query!r}")

    @contextmanager
    def bulk(self):
        """
        Context manager to defer index maintenance while bulk-loading.
        """
        prev = self._in_bulk
        self._in_bulk = True
        try:
            yield
        finally:
            self._in_bulk = prev
            self._rebuild_indexes()

    def _rebuild_indexes(self):
        self._campuses_by_district.clear()
        for c in self._campuses.values():
            self._campuses_by_district[c.district_id].append(c.id)
        # Invalidate spatial index; rebuilt lazily
        self._kdtree = None
        self._xy_deg = None
        self._xy_rad = None
        self._campus_list = None
        # Invalidate Shapely STRtree and vectorized caches
        self._point_tree = None
        self._point_geoms = None
        self._point_ids = None
        self._geom_id_to_index = None
        self._xy_deg_np = None
        self._campus_list_np = None

        # Rebuild campus-number index (supports enrichment joins by campus_number)
        try:
            self._campus_by_number = {}
            for cid, c in self._campuses.items():
                num = getattr(c, "campus_number", None)
                if not num:
                    continue

                key = canonical_campus_number(num)
                if not key:
                    continue
                self._campus_by_number[key] = cid
                digits = key[1:]
                self._campus_by_number[digits] = cid
                if digits.isdigit():
                    self._campus_by_number[str(int(digits))] = cid
        except Exception:
            self._campus_by_number = {}

        if not hasattr(self, "_xfers_missing"):
            self._xfers_missing = {"src": 0, "dst": 0, "either": 0}

    def profile(self, on: bool = True):
        global ENABLE_PROFILING
        ENABLE_PROFILING = bool(on)

    def _ensure_point_strtree(self):
        """Build a Shapely STRtree over campus points for fast spatial containment queries (Shapely 2.x–only)."""
        if not SHAPELY:
            self._point_tree = None
            self._point_geoms = None
            self._point_ids = None
            return

        # Already built?
        if getattr(self, "_point_tree", None) is not None:
            return

        try:
            from shapely.strtree import STRtree  # Shapely 2.x API
        except Exception:
            # Shapely not available or too old
            self._point_tree = None
            self._point_geoms = None
            self._point_ids = None
            return

        geoms: list = []
        ids: list = []
        for cid, c in self._campuses.items():
            p = getattr(c, "point", None) or getattr(c, "location", None)
            if p is None:
                continue
            try:
                # Ensure numeric coordinates to avoid lazy/proxy objects
                float(p.x)
                float(p.y)
            except Exception:
                continue
            geoms.append(p)
            ids.append(cid)

        if not geoms:
            self._point_tree = None
            self._point_geoms = None
            self._point_ids = None
            return

        self._point_tree = STRtree(geoms)
        self._point_geoms = geoms
        self._point_ids = ids
        # Clear legacy maps; we rely on indices-only in Shapely 2.x
        self._geom_id_to_index = None
        self._geom_wkb_to_index = None
        self._xy_to_index = None

        if ENABLE_PROFILING:
            try:
                print(f"[strtree] built with {len(geoms)} points")
            except Exception:
                pass

    def _ensure_kdtree(self):
        """
        Build a KD-tree over campus points (in radians) for fast radius/KNN queries.
        Uses scipy.spatial.cKDTree if available; otherwise leaves tree as None.
        """
        if getattr(self, "_kdtree", None) is not None:
            return
        try:
            import numpy as np
            from scipy.spatial import cKDTree  # type: ignore
        except Exception:
            # No SciPy/NumPy available; skip building the tree
            self._kdtree = None
            self._xy_deg = None
            self._xy_rad = None
            self._campus_list = None
            return

        xy = []
        campus_list = []
        for c in self._campuses.values():
            p = getattr(c, "point", None) or getattr(c, "location", None)
            if p is None:
                continue
            try:
                xy.append((float(p.x), float(p.y)))
                campus_list.append(c)
            except Exception:
                continue

        if not xy:
            self._kdtree = None
            self._xy_deg = None
            self._xy_rad = None
            self._campus_list = None
            return

        import numpy as np

        self._xy_deg = np.array(xy, dtype=float)
        self._xy_rad = np.radians(self._xy_deg)
        from scipy.spatial import cKDTree  # type: ignore

        self._kdtree = cKDTree(self._xy_rad)
        self._campus_list = campus_list

    def _ensure_all_xy_arrays(self):
        """Build NumPy arrays of all campus coordinates for fast bbox candidate queries."""
        try:
            import numpy as np  # type: ignore
        except Exception:
            self._all_xy_np = None
            self._all_campuses_np = None
            return
        if self._all_xy_np is not None and self._all_campuses_np is not None:
            return
        xy = []
        clist = []
        for c in self._campuses.values():
            p = getattr(c, "point", None) or getattr(c, "location", None)
            if p is None:
                continue
            try:
                xy.append((float(p.x), float(p.y)))
                clist.append(c)
            except Exception:
                continue
        if not xy:
            self._all_xy_np = None
            self._all_campuses_np = None
            return
        import numpy as np  # type: ignore

        self._all_xy_np = np.array(xy, dtype=float)
        self._all_campuses_np = clist

    def _bbox_candidates(self, poly) -> List[Campus]:
        """Return campuses whose points fall within the polygon's axis-aligned bounding box."""
        if not SHAPELY:
            # Fallback: simple Python loop without NumPy
            minx, miny, maxx, maxy = (
                poly.bounds if hasattr(poly, "bounds") else (None, None, None, None)
            )
            if minx is None:
                return []
            out = []
            for c in self._campuses.values():
                p = getattr(c, "point", None) or getattr(c, "location", None)
                if p is None:
                    continue
                try:
                    x, y = float(p.x), float(p.y)
                except Exception:
                    continue
                if (minx <= x <= maxx) and (miny <= y <= maxy):
                    out.append(c)
            return out

        # NumPy fast path
        self._ensure_all_xy_arrays()
        if self._all_xy_np is None:
            return []
        import numpy as np  # type: ignore

        minx, miny, maxx, maxy = poly.bounds
        xs = self._all_xy_np[:, 0]
        ys = self._all_xy_np[:, 1]
        mask = (xs >= minx) & (xs <= maxx) & (ys >= miny) & (ys <= maxy)
        idxs = np.nonzero(mask)[0]
        return [self._all_campuses_np[int(i)] for i in idxs]

    def _haversine_miles_vec(self, lon: float, lat: float):
        import numpy as np

        R = 3958.7613
        lon1, lat1 = np.radians([lon, lat])
        lon2 = np.radians(self._xy_deg_np[:, 0])
        lat2 = np.radians(self._xy_deg_np[:, 1])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    @timeit
    def radius_campuses(
        self,
        lon: float,
        lat: float,
        miles: float,
        *,
        charter_only: bool = False,
        limit: Optional[int] = None,
    ) -> List[Campus]:
        """
        Fast radius query using KDTree when available; robust haversine fallback otherwise.
        Returns campuses within 'miles' of (lon, lat), optionally filtered to charters, optionally limited.
        """
        # Try KDTree path
        self._ensure_kdtree()
        try:
            import numpy as np  # noqa: F401
        except Exception:
            pass

        results: List[Tuple[float, Campus]] = []

        if (
            self._kdtree is not None
            and self._xy_deg is not None
            and self._campus_list is not None
        ):
            # Query in radians
            R = 3958.7613
            r = miles / R
            import numpy as np

            idxs = self._kdtree.query_ball_point(np.radians([lon, lat]), r=r)
            idxs = idxs if isinstance(idxs, list) else idxs[0]
            for i in idxs:
                c = self._campus_list[i]
                if charter_only and not getattr(c, "is_charter", False):
                    continue
                x, y = self._xy_deg[i]
                d = haversine_miles(lon, lat, x, y)
                results.append((d, c))
        else:
            # Fallback: vectorized haversine if NumPy available; else brute-force
            try:
                import numpy as np  # noqa: F401
            except Exception:
                for c in self._campuses.values():
                    if c.point is None:
                        continue
                    if charter_only and not getattr(c, "is_charter", False):
                        continue
                    x, y = c.point.x, c.point.y
                    d = haversine_miles(lon, lat, x, y)
                    if d <= miles:
                        results.append((d, c))
            else:
                if self._xy_deg_np is None or self._campus_list_np is None:
                    import numpy as np

                    xy = []
                    clist = []
                    for c in self._campuses.values():
                        if c.point is None:
                            continue
                        if charter_only and not getattr(c, "is_charter", False):
                            continue
                        xy.append((c.point.x, c.point.y))
                        clist.append(c)
                    if not xy:
                        return []
                    self._xy_deg_np = np.array(xy, dtype=float)
                    self._campus_list_np = clist
                dists = self._haversine_miles_vec(lon, lat)
                import numpy as np

                idxs = np.where(dists <= miles)[0]
                order = np.argsort(dists[idxs])
                idxs = [int(idxs[i]) for i in order]
                if limit is not None:
                    idxs = idxs[:limit]
                return [self._campus_list_np[i] for i in idxs]

        results.sort(key=lambda t: t[0])
        if limit is not None:
            results = results[:limit]
        return [c for _, c in results]

    @timeit
    def knn_campuses(
        self, lon: float, lat: float, k: int, *, charter_only: bool = False
    ) -> List[Campus]:
        """
        k-nearest neighbor query. KDTree if available; else brute-force sort by haversine.
        """
        self._ensure_kdtree()
        results: List[Tuple[float, Campus]] = []

        if (
            self._kdtree is not None
            and self._xy_deg is not None
            and self._campus_list is not None
        ):
            from math import isfinite
            import numpy as np

            dists, idxs = self._kdtree.query(
                np.radians([lon, lat]), k=k if k > 1 else 1
            )
            # Normalize outputs
            if k == 1:
                dists = [float(dists)]
                idxs = [int(idxs)]
            # Re-compute accurate miles and filter charter if needed
            for d0, i in zip(dists, idxs):
                c = self._campus_list[int(i)]
                if charter_only and not getattr(c, "is_charter", False):
                    continue
                x, y = self._xy_deg[int(i)]
                dm = haversine_miles(lon, lat, x, y)
                results.append((dm, c))
        else:
            # Fallback: brute-force
            for c in self._campuses.values():
                if c.point is None:
                    continue
                if charter_only and not getattr(c, "is_charter", False):
                    continue
                dm = haversine_miles(lon, lat, c.point.x, c.point.y)
                results.append((dm, c))

        results.sort(key=lambda t: t[0])
        return [c for _, c in results[:k]]

    def nearest_charter_same_type(
        self,
        campuses: Iterable["Campus"],
        *,
        k: int = 1,
    ) -> Dict[str, Dict[str, Any]]:
        """
        For each campus in `campuses`, find the nearest charter campus with the *same* school_type.
        Returns a dict keyed by str(campus.id) -> {"match": Campus|None, "miles": float|None}.

        Implementation:
        - Group target campuses by school_type.
        - Build a KDTree per school_type over charter campuses only (if SciPy available).
        - Fallback to vectorized / brute-force when SciPy is unavailable.
        """
        from collections import defaultdict

        try:
            import numpy as np  # type: ignore
        except Exception:
            np = None  # type: ignore

        by_type_targets: Dict[str, list[Campus]] = defaultdict(list)
        for c in campuses:
            st = (getattr(c, "school_type", None) or "").strip()
            p = getattr(c, "point", None) or getattr(c, "location", None)
            if not st or p is None:
                continue
            by_type_targets[st].append(c)

        if not by_type_targets:
            return {}

        by_type_charters_xy: Dict[str, list[tuple[float, float]]] = defaultdict(list)
        by_type_charters_obj: Dict[str, list[Campus]] = defaultdict(list)
        for cand in self._campuses.values():
            if not getattr(cand, "is_charter", False):
                continue
            st = (getattr(cand, "school_type", None) or "").strip()
            if not st:
                continue
            p = getattr(cand, "point", None) or getattr(cand, "location", None)
            if p is None:
                continue
            try:
                x, y = float(p.x), float(p.y)
            except Exception:
                continue
            by_type_charters_xy[st].append((x, y))
            by_type_charters_obj[st].append(cand)

        trees: Dict[str, Any] = {}
        degcoords: Dict[str, Any] = {}
        for st, xy in by_type_charters_xy.items():
            if not xy:
                continue
            if np is None:
                trees[st] = None
                degcoords[st] = xy
                continue
            arr_deg = np.array(xy, dtype=float)
            try:
                from scipy.spatial import cKDTree  # type: ignore
            except Exception:
                trees[st] = None
                degcoords[st] = arr_deg
                continue
            arr_rad = np.radians(arr_deg)
            trees[st] = cKDTree(arr_rad)
            degcoords[st] = arr_deg  # keep degrees for accurate miles

        results: Dict[str, Dict[str, Any]] = {}
        R = 3958.7613  # miles

        for st, targets in by_type_targets.items():
            tree = trees.get(st)
            cand_deg = degcoords.get(st)
            cand_objs = by_type_charters_obj.get(st, [])
            if not cand_objs:
                for c in targets:
                    results[str(c.id)] = {"match": None, "miles": None}
                continue

            tgt_pairs = []
            tgt_list = []
            for c in targets:
                p = getattr(c, "point", None) or getattr(c, "location", None)
                if p is None:
                    continue
                try:
                    lon, lat = float(p.x), float(p.y)
                except Exception:
                    continue
                tgt_pairs.append((lon, lat))
                tgt_list.append(c)

            if not tgt_list:
                continue

            if tree is not None and np is not None:
                tgt_deg = np.array(tgt_pairs, dtype=float)
                tgt_rad = np.radians(tgt_deg)
                from .classes import haversine_miles

                d_rad, idx = tree.query(tgt_rad, k=1)
                idx = np.atleast_1d(idx)
                for (lon1, lat1), j, camp in zip(tgt_deg, idx, tgt_list):
                    lon2, lat2 = cand_deg[int(j)]
                    dm = haversine_miles(lon1, lat1, float(lon2), float(lat2))
                    results[str(camp.id)] = {
                        "match": cand_objs[int(j)],
                        "miles": float(dm),
                    }
            else:
                # fallback
                try:
                    import numpy as np  # type: ignore
                except Exception:
                    np = None  # type: ignore

                from .classes import haversine_miles

                if np is None:
                    for (lon1, lat1), camp in zip(tgt_pairs, tgt_list):
                        best_dm, best_idx = None, None
                        for j, (lon2, lat2) in enumerate(cand_deg):
                            dm = haversine_miles(lon1, lat1, float(lon2), float(lat2))
                            if best_dm is None or dm < best_dm:
                                best_dm, best_idx = dm, j
                        results[str(camp.id)] = {
                            "match": (
                                cand_objs[best_idx] if best_idx is not None else None
                            ),
                            "miles": float(best_dm) if best_dm is not None else None,
                        }
                else:
                    cand_arr = np.array(cand_deg, dtype=float)
                    for (lon1, lat1), camp in zip(tgt_pairs, tgt_list):
                        lon2r = np.radians(cand_arr[:, 0])
                        lat2r = np.radians(cand_arr[:, 1])
                        lon1r, lat1r = np.radians([lon1, lat1])
                        dlon = lon2r - lon1r
                        dlat = lat2r - lat1r
                        a = (
                            np.sin(dlat / 2) ** 2
                            + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
                        )
                        miles = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                        j = int(np.argmin(miles))
                        results[str(camp.id)] = {
                            "match": cand_objs[j],
                            "miles": float(miles[j]),
                        }

        return results

    # --- CRUD ---
    def add_district(self, d: District):
        d._repo = self
        self._districts[d.id] = d

    def add_campus(self, c: Campus):
        c._repo = self  # attach back-reference
        self._campuses[c.id] = c
        # Only append to _campuses_by_district if the district exists
        if not self._in_bulk:
            if c.district_id in self._districts:
                self._campuses_by_district[c.district_id].append(c.id)

    def campuses_in(self, d: Any) -> "EntityList":
        """
        Return campuses for a district. Accepts:
          - District
          - Query containing a District (first() is used)
        """
        d = _unwrap_query(d)
        if d is None or not hasattr(d, "id"):
            raise ValueError("campuses_in expects a District or a Query[District]")
        return EntityList(
            [self._campuses[cid] for cid in self._campuses_by_district.get(d.id, [])]
        )

    def charter_campuses_within(self, district: Any):
        """
        Return a list of Campus objects that are physically located *within*
        the given district's boundary and are charter campuses (is_charter=True).
        Uses pure spatial containment; district_id membership is ignored.
        Accepts a District or a Query[District].
        """
        district = _unwrap_query(district)
        if district is None:
            return []

        poly = getattr(district, "polygon", None) or getattr(district, "boundary", None)
        if poly is None:
            return []

        # Quick AABB prefilter (NumPy) to cut the candidate set; then exact covers
        if SHAPELY and hasattr(poly, "bounds"):
            bbox_cands = self._bbox_candidates(poly)
            if bbox_cands:
                prep = getattr(district, "prepared", None)
                out_bb: List[Campus] = []
                for c in bbox_cands:
                    if not getattr(c, "is_charter", False):
                        continue
                    p = getattr(c, "point", None) or getattr(c, "location", None)
                    if p is None:
                        continue
                    try:
                        inside = prep.covers(p) if prep is not None else poly.covers(p)
                    except Exception:
                        inside = False
                    if inside:
                        out_bb.append(c)
                if out_bb:
                    return out_bb

        # STRtree (Shapely 2.x) fast path using indices
        if SHAPELY:
            self._ensure_point_strtree()
            if self._point_tree is not None:
                try:
                    idxs = self._point_tree.query(
                        poly, predicate="covers", return_indices=True
                    )
                    # Some builds return (geoms, idxs) if return_geometry=True sneaks in; normalize:
                    if isinstance(idxs, tuple):
                        _, idxs = idxs
                    idxs = list(map(int, idxs)) if idxs is not None else []
                except TypeError:
                    # Old API; fall back to slow scan
                    idxs = []
                except Exception:
                    idxs = []

                if idxs:
                    prep = getattr(district, "prepared", None)
                    out: List[Campus] = []
                    for i in idxs:
                        cid = self._point_ids[i]
                        c = self._campuses.get(cid)
                        if c is None or not getattr(c, "is_charter", False):
                            continue
                        p = getattr(c, "point", None) or getattr(c, "location", None)
                        if p is None:
                            continue
                        try:
                            inside = (
                                prep.covers(p) if prep is not None else poly.covers(p)
                            )
                        except Exception:
                            inside = False
                        if inside:
                            out.append(c)
                    if out:
                        return out

        # Final robust fallback: exact scan
        slow: List[Campus] = []
        for c in self._campuses.values():
            if not getattr(c, "is_charter", False):
                continue
            p = getattr(c, "point", None) or getattr(c, "location", None)
            if p is None:
                continue
            try:
                if SHAPELY and poly.covers(p):
                    slow.append(c)
            except Exception:
                # Non-shapely fallback (rare in this code path)
                pass

        if ENABLE_PROFILING:
            try:
                print(
                    f"[sanity] charter_campuses_within fast=0 slow={len(slow)} — using {'slow' if slow else 'fast'} path"
                )
            except Exception:
                pass
        return slow

    # ---------------- Transfers enrichment (campus->campus edges) ----------------
    def apply_transfers_from_dataframe(
        self,
        df,
        *,
        src_col: str = "REPORT_CAMPUS",
        dst_col: str = "CAMPUS_RES_OR_ATTEND",
        count_col: str = "TRANSFERS_IN_OR_OUT",
        type_col: str = "REPORT_TYPE",
        want_type: str = "Transfers Out To",
    ) -> int:
        """
        Ingest a campus-to-campus transfer table and build bidirectional edge maps.
        Returns the number of *source campuses* for which at least one outgoing edge was recorded.

        The dataframe is expected to contain:
          - `src_col`: source campus number (int/str), e.g. 101902001
          - `dst_col`: destination campus number (int/str)
          - `count_col`: number of transfers (may be masked like -999)
          - `type_col`: when provided, only rows with `type_col == want_type` are used.

        Campus numbers are normalized to canonical 9-digit apostrophe-prefixed strings
        and mapped to repo campus UUIDs via `_campus_by_number`.
        """
        # Filter to the requested report type if present
        if type_col in df.columns:
            df = df[df[type_col] == want_type]

        def norm(x):
            if x is None:
                return None
            # keep ints, int-like floats, and strings robustly
            s = str(int(x)) if isinstance(x, float) and x == int(x) else str(x)
            s = s.strip()
            key = canonical_campus_number(s)
            if not key:
                return None
            return key

        # Reset maps
        self._xfers_out = defaultdict(list)  # type: ignore
        self._xfers_in = defaultdict(list)  # type: ignore

        # Reset diagnostics
        self._xfers_missing = {"src": 0, "dst": 0, "either": 0}

        updated_sources = set()

        # Iterate rows and build edges
        for _, row in df.iterrows():
            src_key = norm(row.get(src_col))
            dst_key = norm(row.get(dst_col))
            if not src_key or not dst_key:
                continue

            src_digits = src_key[1:] if src_key and src_key.startswith("'") else src_key
            dst_digits = dst_key[1:] if dst_key and dst_key.startswith("'") else dst_key

            src_id = self._campus_by_number.get(src_key) or self._campus_by_number.get(
                src_digits
            )
            dst_id = self._campus_by_number.get(dst_key) or self._campus_by_number.get(
                dst_digits
            )
            if src_id is None or dst_id is None:
                if src_id is None and dst_id is None:
                    self._xfers_missing["either"] = (
                        self._xfers_missing.get("either", 0) + 1
                    )
                elif src_id is None:
                    self._xfers_missing["src"] = self._xfers_missing.get("src", 0) + 1
                else:
                    self._xfers_missing["dst"] = self._xfers_missing.get("dst", 0) + 1
                # Skip edges where either side is not in the repo
                continue

            raw = row.get(count_col)
            try:
                cnt = int(raw)
            except Exception:
                cnt = None

            # Many TEA files use -999 for masked small counts
            masked = cnt is not None and cnt < 0
            cnt = None if masked else cnt

            self._xfers_out[src_id].append((dst_id, cnt, bool(masked)))
            self._xfers_in[dst_id].append((src_id, cnt, bool(masked)))
            updated_sources.add(src_id)

        # Sort edges deterministically: largest counts first, then by campus name
        def _sort_key(edge):
            to_id, cnt, _ = edge
            cobj = self._campuses.get(to_id)
            return (-(cnt or -1), (cobj.name if cobj else ""))

        for k in list(self._xfers_out.keys()):
            self._xfers_out[k].sort(key=_sort_key)

        for k in list(self._xfers_in.keys()):
            # for transfers_in sort by the incoming count, same logic
            from_id, cnt, _ = (
                self._xfers_in[k][0] if self._xfers_in[k] else (None, None, None)
            )
            self._xfers_in[k].sort(
                key=lambda ed: (
                    -(ed[1] or -1),
                    (
                        self._campuses.get(ed[0]).name
                        if self._campuses.get(ed[0])
                        else ""
                    ),
                )
            )

        return len(updated_sources)

    def transfers_out(
        self, campus: "Campus"
    ) -> List[Tuple["Campus", Optional[int], bool]]:
        """Return list of (to_campus, count, masked) for a given source campus."""
        cid = getattr(campus, "id", None)
        out = []
        for to_id, cnt, masked in self._xfers_out.get(cid, []):
            c2 = self._campuses.get(to_id)
            if c2 is not None:
                out.append((c2, cnt, masked))
        return out

    def transfers_in(
        self, campus: "Campus"
    ) -> List[Tuple["Campus", Optional[int], bool]]:
        """Return list of (from_campus, count, masked) for a given destination campus."""
        cid = getattr(campus, "id", None)
        out = []
        for from_id, cnt, masked in self._xfers_in.get(cid, []):
            c1 = self._campuses.get(from_id)
            if c1 is not None:
                out.append((c1, cnt, masked))
        return out

    @timeit
    def nearest_campuses(
        self,
        x: float,
        y: float,
        *,
        limit: int = 1,
        charter_only: bool = False,
        max_miles: Optional[float] = None,
        geodesic: bool = True,
    ) -> List[Campus]:
        """
        Find the nearest N campuses to (x, y).

        Parameters
        ----------
        x, y : float
            Coordinates. If geodesic=True and values look like lon/lat, distances are in miles (haversine).
            Otherwise, falls back to planar distance (units of your CRS).
        limit : int
            Number of campuses to return (use 1 for the single nearest).
        charter_only : bool
            If True, only consider campuses with is_charter=True.
        max_miles : Optional[float]
            If provided (and geodesic distance is used), filter out campuses farther than this many miles.
            If geodesic=False, this is treated in *planar* units.
        geodesic : bool
            If True (default) and x/y look like lon/lat, compute great-circle miles.
        """
        # Collect candidates with distances
        results: List[Tuple[float, Campus]] = []
        for c in self._campuses.values():
            if c.point is None:
                continue
            if charter_only and not getattr(c, "is_charter", False):
                continue
            cxy = _point_xy(c.point)
            if cxy is None:
                continue
            if geodesic and _probably_lonlat(x, y) and _probably_lonlat(*cxy):
                d = haversine_miles(x, y, cxy[0], cxy[1])
            else:
                # planar fallback
                if SHAPELY and hasattr(c.point, "distance"):
                    # distance in CRS units (assumed miles if your data is projected to a miles-based CRS)
                    d = (
                        c.point.distance(ShapelyPoint(x, y))
                        if SHAPELY
                        else euclidean((x, y), cxy)
                    )
                else:
                    d = euclidean((x, y), cxy)
            if max_miles is not None and d > max_miles:
                continue
            results.append((d, c))

        results.sort(key=lambda t: t[0])
        if limit <= 1:
            return [results[0][1]] if results else []
        return [c for _, c in results[:limit]]

    @timeit
    @lru_cache(maxsize=2048)
    def nearest_campus(
        self, x: float, y: float, charter_only: bool = False
    ) -> Optional[Campus]:
        """
        Backward-compatible convenience for the single nearest campus.
        Uses geodesic miles when (x,y) look like lon/lat.
        """
        res = self.nearest_campuses(
            x, y, limit=1, charter_only=charter_only, max_miles=None, geodesic=True
        )
        return res[0] if res else None


# Public helper for quick-start scripts/tests
def load_default_repo() -> DataEngine:
    """Return a DataEngine loaded from the newest available snapshot, or an empty one."""
    return DataEngine.from_snapshot(None, search=True)


def to_df(self, which: str = "districts", **kwargs):
    """
    Convenience: DataFrame of 'districts' or 'campuses'.
    Example: repo.to_df("campuses", columns=["name","enrollment"])
    """
    if which == "districts":
        return self.districts.to_df(**kwargs)
    elif which == "campuses":
        return self.campuses.to_df(**kwargs)
    else:
        raise ValueError("which must be 'districts' or 'campuses'")


def to_dicts(self, which: str = "districts", **kwargs) -> list[dict]:
    if which == "districts":
        return self.districts.to_dicts(**kwargs)
    elif which == "campuses":
        return self.campuses.to_dicts(**kwargs)
    else:
        raise ValueError("which must be 'districts' or 'campuses'")


def inspect_object(o):
    # 1) Canonical dataclass fields
    base = [f.name for f in fields(o)] if is_dataclass(o) else []
    # 2) Enriched fields (live in .meta but are dot-accessible)
    meta = sorted(getattr(o, "meta", {}).keys())
    # 3) Public methods/attrs (handy to discover helpers/operators)
    public = [n for n in dir(o) if not n.startswith("_")]

    print("dataclass fields:", base)
    print("enriched keys   :", meta)
    print("public members  :", public)

    # If your class implements to_dict()
    try:
        d = o.to_dict()
        print("to_dict keys    :", list(d.keys()))
    except Exception:
        pass


# --------- Demo data ---------


def demo_repo() -> DataEngine:
    repo = DataEngine()
    d1 = District(
        id=uuid.uuid4(),
        name="Austin ISD",
        enrollment=73000,
        district_number="",
        rating="B",
        boundary=(
            (ShapelyPolygon([(0, 0), (5, 0), (5, 5), (0, 5)]))
            if SHAPELY
            else [(0, 0), (5, 0), (5, 5), (0, 5)]
        ),
    )
    d2 = District(
        id=uuid.uuid4(),
        name="Round Rock ISD",
        enrollment=51000,
        district_number="",
        rating="A",
        boundary=(
            (ShapelyPolygon([(4, 2), (9, 2), (9, 7), (4, 7)]))
            if SHAPELY
            else [(4, 2), (9, 2), (9, 7), (4, 7)]
        ),
    )

    with repo.bulk():
        repo.add_district(d1)
        repo.add_district(d2)

        c1 = Campus(
            id=uuid.uuid4(),
            district_id=d1.id,
            name="Akins HS",
            campus_number="",
            charter_type="",
            is_charter=False,
            enrollment=2600,
            district_number="",
            location=ShapelyPoint(2, 1) if SHAPELY else (2, 1),
        )
        c2 = Campus(
            id=uuid.uuid4(),
            district_id=d1.id,
            name="Kealing MS",
            campus_number="",
            charter_type="",
            is_charter=False,
            enrollment=1200,
            district_number="",
            location=ShapelyPoint(3, 3) if SHAPELY else (3, 3),
        )
        c3 = Campus(
            id=uuid.uuid4(),
            district_id=d2.id,
            name="Westwood HS",
            campus_number="",
            charter_type="",
            is_charter=False,
            enrollment=2800,
            district_number="",
            location=ShapelyPoint(8, 6) if SHAPELY else (8, 6),
        )
        repo.add_campus(c1)
        repo.add_campus(c2)
        repo.add_campus(c3)

    return repo


# --------- Example usage (uncomment to run as script) ---------
if __name__ == "__main__":
    repo = demo_repo()
    aisd, rr = list(repo._districts.values())

    # 1) Membership via "in"
    for campus in repo.campuses_in(aisd):
        print(f"{campus.name} in Austin ISD? {'yes' if campus in aisd else 'no'}")

    # 2) Distances via '@' and '-' overloads
    target = repo.nearest_campus(4.5, 4.5)
    print("Nearest to (4.5,4.5):", target.name)

    # Campus-to-campus distance
    c_list = repo.campuses_in(aisd)
    d_cc = c_list[0] - c_list[1]
    print("Distance campus0 - campus1:", d_cc)

    # 3) District overlaps via '&' and union via '|'
    inter_area = aisd & rr
    union_area = aisd | rr
    print("Intersection area:", inter_area, "Union area:", union_area)

    # 4) Repo pipeline operator '>>'
    big = repo >> (lambda o: isinstance(o, Campus) and o.enrollment >= 2000)
    print("Big campuses:", [c.name for c in big])

    # 5) Pattern matching + custom formatting
    match aisd:
        case District(name, enr) if enr > 70000:
            print(f"Match says: {aisd:brief}")
