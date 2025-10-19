"""Core data engine responsible for snapshot loading and spatial queries."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, is_dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import io
import math
import os
import pickle
import tempfile
import time
import urllib.request
import uuid

from teadata.teadata_config import (
    canonical_campus_number,
    canonical_district_number,
    normalize_campus_number_value,
    normalize_district_number_value,
)

from .entities import (
    Campus,
    District,
    EntityList,
    EntityMap,
    ReadOnlyEntityView,
    is_charter,
    is_private,
)
from .geometry import (
    SHAPELY,
    MultiPolygon,
    ShapelyPoint,
    ShapelyPolygon,
    district_centroid_xy,
    euclidean,
    haversine_miles,
    point_in_polygon,
    point_xy,
    probably_lonlat,
)
from .query import Query, unwrap_query

try:
    from platformdirs import user_cache_dir  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    user_cache_dir = None

DEFAULT_REPO_PKL_RELATIVE = (
    ".cache/repo_Current_Districts_2025_Schools_2024_to_2025.pkl"
)
DEFAULT_REPO_PKL_URL = (
    "https://raw.githubusercontent.com/adpena/teadata/"
    "bddeb222dd579542453ae47163ebe39cc3a07081/teadata/.cache/"
    "repo_Current_Districts_2025_Schools_2024_to_2025.pkl"
)


class _ModuleMappingUnpickler(pickle.Unpickler):
    """Remap legacy module names when loading snapshots."""

    def __init__(self, file, module_map: dict[str, str] | None = None):
        super().__init__(file)
        self._module_map = module_map or {}

    def find_class(self, module, name):  # pragma: no cover - exercised indirectly
        remapped = self._module_map.get(module, module)
        return super().find_class(remapped, name)


def _compat_pickle_load(fobj) -> object:
    try:
        return pickle.load(fobj)
    except ModuleNotFoundError as e:  # pragma: no cover - compatibility path
        try:
            try:
                fobj.seek(0)
                bio = fobj  # type: ignore
            except Exception:
                data = fobj.read()
                bio = io.BytesIO(data)
            current_mod = __name__
            module_map = {
                "classes": current_mod,
                "__main__": current_mod,
            }
            return _ModuleMappingUnpickler(bio, module_map).load()
        except Exception as ee:
            raise ee from e


ENABLE_PROFILING = False


def timeit(func):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        dt = (time.perf_counter() - t0) * 1000
        if ENABLE_PROFILING:
            print(f"[timeit] {func.__name__} took {dt:.2f} ms")
        return out

    return wrapper


def _iter_parents(start: Path):
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

    if explicit:
        p = Path(explicit)
        if p.exists() and p.is_file():
            return p

    env = os.environ.get("TEADATA_SNAPSHOT")
    if env:
        p = Path(env)
        if p.exists() and p.is_file():
            return p

    try:
        package_dir = Path(__file__).resolve().parent
        pkg_cache = package_dir / ".cache"
        p = _newest_pickle(pkg_cache)
        if p is not None:
            return p
    except Exception:
        pass

    for base in _iter_parents(Path.cwd()):
        p = _newest_pickle(base / ".cache")
        if p is not None:
            return p

    if user_cache_dir:
        try:
            ucache = Path(user_cache_dir("teadata", "adpena"))
            p = _newest_pickle(ucache)
            if p is not None:
                return p
        except Exception:
            pass

    return None


@dataclass
class _CharterCacheEntry:
    objects: list[Campus]
    coords_deg: Any
    tree: Any


@dataclass
class _CharterCache:
    has_numpy: bool
    entries: Dict[str, _CharterCacheEntry]

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
            is_private = v.get("is_private", False)
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
                is_private=is_private,
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
        is_private = getattr(v, "is_private", False)
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
            is_private=is_private,
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
            try:
                obj._rehydrate_geometries()
            except Exception:
                pass
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
            try:
                eng._rehydrate_geometries()
            except Exception:
                pass
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
                    try:
                        p._rehydrate_geometries()
                    except Exception:
                        pass
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
                try:
                    eng._rehydrate_geometries()
                except Exception:
                    pass
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
                self._charter_cache = None

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
                try:
                    self._rehydrate_geometries()
                except Exception:
                    pass
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
        self._charter_cache = None

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

    def campus_by_number(self, campus_number: Any) -> Optional["Campus"]:
        """Return a campus by its TEA campus number in any common format."""

        if campus_number is None:
            return None

        index = getattr(self, "_campus_by_number", None)
        if not index:
            # Ensure the lookup table is populated (e.g., after bulk loads)
            try:
                self._rebuild_indexes()
            except Exception:
                index = {}
            else:
                index = getattr(self, "_campus_by_number", {})

        candidates: list[str] = []

        canonical = canonical_campus_number(campus_number)
        if canonical:
            candidates.append(canonical)
            digits = canonical[1:]
            if digits:
                candidates.append(digits)
                if digits.isdigit():
                    candidates.append(str(int(digits)))

        raw = str(campus_number).strip()
        if raw:
            candidates.append(raw)
            if raw.startswith(("'", "’", "`")):
                trimmed = raw[1:].lstrip()
                if trimmed:
                    candidates.append(trimmed)
            try:
                candidates.append(str(int(raw)))
            except Exception:
                pass

        seen: set[str] = set()
        for key in candidates:
            if not key or key in seen:
                continue
            seen.add(key)
            cid = index.get(key) if index is not None else None
            if cid is None:
                continue
            campus = self._campuses.get(cid)
            if campus is not None:
                return campus

        return None

    def __len__(self) -> int:
        return len(self._districts) + len(self._campuses)

    def __iter__(self):
        # iterate districts first, then campuses
        yield from self._districts.values()
        yield from self._campuses.values()

    def __getitem__(self, key: uuid.UUID) -> District | Campus:
        return self._districts.get(key) or self._campuses[key]

    def _rehydrate_geometries(self) -> None:
        """
        Ensure polygon/point attributes are Shapely geometries when Shapely is available.
        Older snapshots may deserialize tuples/lists instead of actual geometry objects;
        we reassign through the GeoField descriptor to normalize them.
        """

        if not SHAPELY:
            return

        for district in self._districts.values():
            poly = getattr(district, "polygon", None)
            if poly is None or isinstance(poly, (ShapelyPolygon, MultiPolygon)):
                continue
            try:
                district.polygon = poly
            except Exception:
                continue

        for campus in self._campuses.values():
            pt = getattr(campus, "point", None)
            if pt is None:
                pt = getattr(campus, "location", None)
            if pt is None or isinstance(pt, ShapelyPoint):
                continue
            try:
                campus.point = pt
            except Exception:
                continue

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

             repo >> ("privates_within", district)
             -> Query[Campus]

             repo >> ("private_campuses_in", district, max_miles?)
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

            if key == "campus":
                raw = query[1]

                if isinstance(raw, int) or (
                    isinstance(raw, str) and any(ch.isdigit() for ch in raw)
                ):
                    campus = self.campus_by_number(raw)
                    return Query([campus] if campus is not None else [], self)

                target = (str(raw) if raw is not None else "").strip()
                if not target:
                    return Query([], self)

                import fnmatch

                pattern = target.replace("%", "*").replace("_", "?")
                pattern_up = pattern.upper()
                has_glob = any(ch in pattern_up for ch in ("*", "?"))

                matches = []
                for c in self._campuses.values():
                    name_up = (c.name or "").upper()
                    if has_glob:
                        if fnmatch.fnmatchcase(name_up, pattern_up):
                            matches.append(c)
                    else:
                        if name_up == pattern_up:
                            matches.append(c)

                return Query(matches, self)

            if key == "campuses_in":
                # Accept a District instance or a Query[District]; unwrap if needed
                district = unwrap_query(query[1])
                if district is None:
                    return Query([], self)
                return Query(self.campuses_in(district), self)

            if key == "private_campuses_in":
                district = unwrap_query(query[1])
                if district is None:
                    return Query([], self)
                max_m = (
                    float(query[2])
                    if len(query) >= 3 and query[2] is not None
                    else None
                )
                return Query(
                    self.private_campuses_in(district, max_miles=max_m), self
                )

            if key == "charters_within":
                district = unwrap_query(query[1])
                return Query(self.charter_campuses_within(district), self)

            if key == "privates_within":
                district = unwrap_query(query[1])
                return Query(self.private_campuses_within(district), self)

            if key == "nearest":
                seed = query[1]
                n = int(query[2]) if len(query) >= 3 else 1
                max_m = (
                    float(query[3])
                    if len(query) >= 4 and query[3] is not None
                    else None
                )
                # If seed is a Query or Campus, try to infer coords
                seed = unwrap_query(seed)
                if hasattr(seed, "point"):
                    xy = point_xy(getattr(seed, "point"))
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
                seed = unwrap_query(seed)
                if hasattr(seed, "point"):
                    xy = point_xy(getattr(seed, "point"))
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

    def _clear_charter_cache(self):
        self._charter_cache = None

    def _ensure_charter_cache(self) -> _CharterCache:
        cache = getattr(self, "_charter_cache", None)
        if cache is not None:
            return cache
        try:
            import numpy as np  # type: ignore
        except Exception:
            np = None  # type: ignore
        coords_by_type: Dict[str, list[tuple[float, float]]] = defaultdict(list)
        objs_by_type: Dict[str, list[Campus]] = defaultdict(list)
        for cand in self._campuses.values():
            if not is_charter(cand):
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
            coords_by_type[st].append((x, y))
            objs_by_type[st].append(cand)
        entries: Dict[str, _CharterCacheEntry] = {}
        for st, coords in coords_by_type.items():
            if not coords:
                continue
            tree = None
            coords_payload: Any = coords
            if np is not None:
                arr_deg = np.array(coords, dtype=float)
                try:
                    from scipy.spatial import cKDTree  # type: ignore
                except Exception:
                    coords_payload = arr_deg
                else:
                    coords_payload = arr_deg
                    tree = cKDTree(np.radians(arr_deg))
            entries[st] = _CharterCacheEntry(objs_by_type[st], coords_payload, tree)
        cache = _CharterCache(has_numpy=np is not None, entries=entries)
        self._charter_cache = cache
        return cache

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

        self._charter_cache = None

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

    @staticmethod
    def _extract_polygon_coords(poly) -> List[Tuple[float, float]]:
        """Return a cleaned list of (x, y) vertices for non-Shapely polygons."""

        try:
            coords = list(poly)
        except TypeError:
            return []

        cleaned: List[Tuple[float, float]] = []
        for pt in coords:
            if (
                isinstance(pt, (tuple, list))
                and len(pt) == 2
                and all(isinstance(v, (int, float)) for v in pt)
            ):
                cleaned.append((float(pt[0]), float(pt[1])))
        return cleaned

    def _bbox_candidates(self, poly) -> List[Campus]:
        """Return campuses whose points fall within the polygon's axis-aligned bounding box."""
        if not SHAPELY:
            # Fallback: simple Python loop without NumPy
            if hasattr(poly, "bounds"):
                minx, miny, maxx, maxy = poly.bounds
            else:
                coords = self._extract_polygon_coords(poly)
                if not coords:
                    return []
                xs = [pt[0] for pt in coords]
                ys = [pt[1] for pt in coords]
                minx = min(xs)
                maxx = max(xs)
                miny = min(ys)
                maxy = max(ys)
            if minx is None:
                return []
            out = []
            for c in self._campuses.values():
                p = getattr(c, "point", None) or getattr(c, "location", None)
                if p is None:
                    continue
                xy = point_xy(p)
                if xy is None:
                    continue
                x, y = xy
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
                if charter_only and not is_charter(c):
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
                    if charter_only and not is_charter(c):
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
                        if charter_only and not is_charter(c):
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
                if charter_only and not is_charter(c):
                    continue
                x, y = self._xy_deg[int(i)]
                dm = haversine_miles(lon, lat, x, y)
                results.append((dm, c))
        else:
            # Fallback: brute-force
            for c in self._campuses.values():
                if c.point is None:
                    continue
                if charter_only and not is_charter(c):
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
        """Return the nearest charter campus of the same school type for each campus."""
        from collections import defaultdict

        targets_by_type: Dict[str, list[Campus]] = defaultdict(list)
        for campus in campuses:
            st = (getattr(campus, "school_type", None) or "").strip()
            pt = getattr(campus, "point", None) or getattr(campus, "location", None)
            if not st or pt is None:
                continue
            targets_by_type[st].append(campus)

        if not targets_by_type:
            return {}

        cache = self._ensure_charter_cache()
        entries = cache.entries

        np_mod = None
        if cache.has_numpy:
            try:
                import numpy as np  # type: ignore
            except Exception:
                np_mod = None
            else:
                np_mod = np

        results: Dict[str, Dict[str, Any]] = {}
        R = 3958.7613

        for st, targets in targets_by_type.items():
            entry = entries.get(st)
            if entry is None or not entry.objects:
                for campus in targets:
                    results[str(campus.id)] = {"match": None, "miles": None}
                continue

            cand_objs = entry.objects
            cand_deg = entry.coords_deg

            tgt_pairs: list[tuple[float, float]] = []
            tgt_list: list[Campus] = []
            for campus in targets:
                pt = getattr(campus, "point", None) or getattr(campus, "location", None)
                if pt is None:
                    continue
                try:
                    lon, lat = float(pt.x), float(pt.y)
                except Exception:
                    continue
                tgt_pairs.append((lon, lat))
                tgt_list.append(campus)

            if not tgt_list:
                continue

            if entry.tree is not None and np_mod is not None:
                tgt_deg = np_mod.array(tgt_pairs, dtype=float)
                dists, idx = entry.tree.query(np_mod.radians(tgt_deg), k=1)
                idx = np_mod.atleast_1d(idx)
                for (lon1, lat1), j, campus in zip(tgt_deg, idx, tgt_list):
                    lon2, lat2 = entry.coords_deg[int(j)]
                    dm = haversine_miles(float(lon1), float(lat1), float(lon2), float(lat2))
                    results[str(campus.id)] = {"match": cand_objs[int(j)], "miles": float(dm)}
                continue

            if np_mod is not None:
                cand_arr = (
                    cand_deg
                    if isinstance(cand_deg, np_mod.ndarray)
                    else np_mod.array(cand_deg, dtype=float)
                )
                for (lon1, lat1), campus in zip(tgt_pairs, tgt_list):
                    lon1r, lat1r = np_mod.radians([lon1, lat1])
                    lon2r = np_mod.radians(cand_arr[:, 0])
                    lat2r = np_mod.radians(cand_arr[:, 1])
                    dlon = lon2r - lon1r
                    dlat = lat2r - lat1r
                    a = np_mod.sin(dlat / 2) ** 2 + np_mod.cos(lat1r) * np_mod.cos(lat2r) * np_mod.sin(dlon / 2) ** 2
                    miles = R * 2 * np_mod.arctan2(np_mod.sqrt(a), np_mod.sqrt(1 - a))
                    j = int(np_mod.argmin(miles))
                    results[str(campus.id)] = {"match": cand_objs[j], "miles": float(miles[j])}
                continue

            cand_list = list(cand_deg)
            for (lon1, lat1), campus in zip(tgt_pairs, tgt_list):
                best_dm: Optional[float] = None
                best_obj: Optional[Campus] = None
                for (lon2, lat2), cand in zip(cand_list, cand_objs):
                    dm = haversine_miles(float(lon1), float(lat1), float(lon2), float(lat2))
                    if best_dm is None or dm < best_dm:
                        best_dm = float(dm)
                        best_obj = cand
                results[str(campus.id)] = {"match": best_obj, "miles": best_dm}

        return results

    def nearest_charter_transfer_destination(
        self,
        campuses: Iterable["Campus"],
    ) -> Dict[str, Dict[str, Any]]:
        """
        For each campus, among the *charter* campuses (excluding private schools) that
        receive student transfers from that campus (edges in _xfers_out), return the
        spatially nearest one.

        Returns a dict keyed by str(campus.id) -> {"match": Campus|None, "miles": float|None}.
        If a campus has no charter transfer destinations or lacks geometry, returns None values.
        """
        results: Dict[str, Dict[str, Any]] = {}

        for c in campuses:
            cid = getattr(c, "id", None)
            if cid is None:
                continue
            # Source point
            p = getattr(c, "point", None) or getattr(c, "location", None)
            try:
                lon1, lat1 = (float(p.x), float(p.y)) if p is not None else (None, None)
            except Exception:
                lon1, lat1 = (None, None)

            best_dm: Optional[float] = None
            best_obj: Optional[Campus] = None

            # Gather charter destinations from transfers_out
            edges = self._xfers_out.get(cid, [])
            if edges:
                for to_id, _cnt, _masked in edges:
                    dest = self._campuses.get(to_id)
                    if dest is None or not is_charter(dest):
                        continue
                    q = getattr(dest, "point", None) or getattr(dest, "location", None)
                    try:
                        lon2, lat2 = (
                            (float(q.x), float(q.y)) if q is not None else (None, None)
                        )
                    except Exception:
                        lon2, lat2 = (None, None)
                    if None in (lon1, lat1, lon2, lat2):
                        continue
                    dm = haversine_miles(lon1, lat1, lon2, lat2)
                    if best_dm is None or dm < best_dm:
                        best_dm, best_obj = float(dm), dest

            results[str(cid)] = {
                "match": best_obj,
                "miles": best_dm if best_dm is not None else None,
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
            self._clear_charter_cache()

    def campuses_in(self, d: Any) -> "EntityList":
        """
        Return campuses for a district. Accepts:
          - District
          - Query containing a District (first() is used)
        """
        d = unwrap_query(d)
        if d is None or not hasattr(d, "id"):
            raise ValueError("campuses_in expects a District or a Query[District]")
        campuses = [
            self._campuses[cid]
            for cid in self._campuses_by_district.get(d.id, [])
            if cid in self._campuses
        ]

        return EntityList(
            [c for c in campuses if not is_charter(c) and not is_private(c)]
        )

    def private_campuses_in(
        self, d: Any, *, max_miles: Optional[float] = None
    ) -> "EntityList":
        """Return private-school campuses attached to a district by membership.

        When ``max_miles`` is provided the results are limited to campuses whose
        coordinates fall within that many miles of the district's centroid. Campuses
        without point geometry (or districts without usable geometry) are omitted
        from the radius-filtered result.
        """

        d = unwrap_query(d)
        if d is None or not hasattr(d, "id"):
            raise ValueError("private_campuses_in expects a District or a Query[District]")

        campuses = [
            self._campuses[cid]
            for cid in self._campuses_by_district.get(d.id, [])
            if cid in self._campuses and is_private(self._campuses[cid])
        ]

        if max_miles is None:
            return EntityList(campuses)

        centroid = district_centroid_xy(d)
        if centroid is None:
            return EntityList([])

        cx, cy = centroid
        filtered: List[Campus] = []
        for campus in campuses:
            pt = getattr(campus, "point", None) or getattr(campus, "location", None)
            xy = point_xy(pt)
            if xy is None:
                continue
            if probably_lonlat(cx, cy) and probably_lonlat(*xy):
                dist = haversine_miles(cx, cy, xy[0], xy[1])
            elif SHAPELY and pt is not None and hasattr(pt, "distance"):
                try:
                    dist = pt.distance(ShapelyPoint(cx, cy))
                except Exception:
                    dist = euclidean((cx, cy), xy)
            else:
                dist = euclidean((cx, cy), xy)

            if dist <= max_miles:
                filtered.append(campus)

        return EntityList(filtered)

    def _campuses_within_filtered(
        self,
        district: Any,
        *,
        predicate: Callable[["Campus"], bool],
        label: str,
    ) -> List[Campus]:
        district = unwrap_query(district)
        if district is None:
            return []

        poly = getattr(district, "polygon", None) or getattr(district, "boundary", None)
        if poly is None:
            return []

        poly_xy: List[Tuple[float, float]] = []
        bbox_min_x = bbox_min_y = bbox_max_x = bbox_max_y = None  # type: Optional[float]
        if not SHAPELY or not hasattr(poly, "covers"):
            try:
                coords = list(poly)
            except TypeError:
                coords = []
            cleaned: List[Tuple[float, float]] = []
            for pt in coords:
                if (
                    isinstance(pt, (tuple, list))
                    and len(pt) == 2
                    and all(isinstance(v, (int, float)) for v in pt)
                ):
                    cleaned.append((float(pt[0]), float(pt[1])))
            if cleaned:
                poly_xy = cleaned
                xs = [p[0] for p in cleaned]
                ys = [p[1] for p in cleaned]
                bbox_min_x = min(xs)
                bbox_max_x = max(xs)
                bbox_min_y = min(ys)
                bbox_max_y = max(ys)

        # Quick AABB prefilter (NumPy) to cut the candidate set; then exact covers
        if SHAPELY and hasattr(poly, "bounds"):
            bbox_cands = self._bbox_candidates(poly)
            if bbox_cands:
                prep = getattr(district, "prepared", None)
                out_bb: List[Campus] = []
                for c in bbox_cands:
                    try:
                        if not predicate(c):
                            continue
                    except Exception:
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
                        if c is None:
                            continue
                        try:
                            if not predicate(c):
                                continue
                        except Exception:
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
        poly_coords: Optional[List[Tuple[float, float]]] = None
        for c in self._campuses.values():
            try:
                if not predicate(c):
                    continue
            except Exception:
                continue
            p = getattr(c, "point", None) or getattr(c, "location", None)
            if p is None:
                continue
            inside = False
            if SHAPELY and hasattr(poly, "covers"):
                try:
                    inside = poly.covers(p)
                except Exception:
                    inside = False
            if not inside:
                if poly_coords is None:
                    poly_coords = self._extract_polygon_coords(poly)
                if poly_coords:
                    xy = point_xy(p)
                    if xy is not None:
                        try:
                            inside = point_in_polygon(xy, poly_coords)
                        except Exception:
                            inside = False
            if inside:
                slow.append(c)

        if ENABLE_PROFILING:
            try:
                print(
                    f"[sanity] {label} fast=0 slow={len(slow)} — using {'slow' if slow else 'fast'} path"
                )
            except Exception:
                pass
        return slow

    def charter_campuses_within(self, district: Any):
        """
        Return a list of Campus objects that are physically located *within*
        the given district's boundary and are charter campuses (excludes private schools).
        Uses pure spatial containment; district_id membership is ignored.
        Accepts a District or a Query[District].
        """

        return self._campuses_within_filtered(
            district,
            predicate=is_charter,
            label="charter_campuses_within",
        )

    def private_campuses_within(self, district: Any):
        """
        Return a list of Campus objects that are physically located *within*
        the given district's boundary and are private schools (Campus.is_private is True).
        Uses pure spatial containment; district_id membership is ignored.
        Accepts a District or a Query[District].
        """

        return self._campuses_within_filtered(
            district,
            predicate=is_private,
            label="private_campuses_within",
        )

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
            If True, only consider campuses with is_charter=True and is_private=False.
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
            if charter_only and not is_charter(c):
                continue
            cxy = point_xy(c.point)
            if cxy is None:
                continue
            if geodesic and probably_lonlat(x, y) and probably_lonlat(*cxy):
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
