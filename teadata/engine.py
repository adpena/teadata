"""Core data engine responsible for snapshot loading and spatial queries."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from heapq import nsmallest
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple
import fnmatch
import gzip
import logging
import math
import os
import pickle
import sys
import time
import uuid

from .manifest import ContractError, DatasetCatalog, read_sqlite_catalog
from .identifiers import EntityIdentityError, canonical_source_identifier
from .release import current_release, snapshot_build_id
from .boundary_store import BoundaryStore
from teadata.identifiers import normalize_campus_lookup_key
from teadata.teadata_config import (
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
    EARTH_RADIUS_MILES,
    SHAPELY,
    MultiPolygon,
    ShapelyPoint,
    ShapelyPolygon,
    district_boundary_geometry,
    district_boundary_issue,
    euclidean,
    haversine_miles,
    point_in_polygon,
    point_xy,
    probably_lonlat,
)
from .query import Query, unwrap_query

if TYPE_CHECKING:
    from .observations import _ObservationModel
    from .release import DataRelease

GZIP_MAGIC = b"\x1f\x8b"


logger = logging.getLogger(__name__)

ENABLE_PROFILING = False


def _normalize_aea_raw(value: Any) -> Optional[str]:
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


def _coerce_aea_bool(value: Any) -> Optional[bool]:
    raw = _normalize_aea_raw(value)
    if raw == "Y":
        return True
    if raw == "N":
        return False
    return None


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in ("0", "false", "no", "off", "")


DEFAULT_INDEXES_ENABLED = not _env_flag("TEADATA_DISABLE_INDEXES", default=False)
LOG_MEMORY = _env_flag("TEADATA_LOG_MEMORY", default=False)


def _read_proc_status_value(key: str) -> Optional[int]:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(f"{key}:"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        return int(parts[1]) * 1024
    except Exception:
        return None
    return None


def _read_proc_meminfo() -> dict[str, int]:
    info: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(("MemTotal:", "MemAvailable:")):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        info[parts[0].rstrip(":")] = int(parts[1]) * 1024
    except Exception:
        return info
    return info


def _ru_maxrss_bytes() -> Optional[int]:
    try:
        import resource
    except Exception:
        return None
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None
    if sys.platform == "darwin":
        return int(rss)
    return int(rss) * 1024


def _format_bytes(value: Optional[int]) -> Optional[str]:
    if value is None:
        return None
    return f"{value / (1024 * 1024):.1f}MB"


def _read_cgroup_value(path: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read().strip()
    except Exception:
        return None
    if not raw or raw == "max":
        return None
    if raw.isdigit():
        return int(raw)
    return None


def _read_cgroup_memory() -> dict[str, int]:
    current = _read_cgroup_value("/sys/fs/cgroup/memory.current")
    maximum = _read_cgroup_value("/sys/fs/cgroup/memory.max")
    if current is None or maximum is None:
        current = _read_cgroup_value("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        maximum = _read_cgroup_value("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    info: dict[str, int] = {}
    if current is not None:
        info["current"] = current
    if maximum is not None and maximum < 1 << 60:
        info["max"] = maximum
    return info


def _memory_snapshot(label: str, extra: Optional[dict[str, object]] = None) -> str:
    parts: list[str] = [f"label={label}", f"pid={os.getpid()}"]
    rss = _read_proc_status_value("VmRSS")
    hwm = _read_proc_status_value("VmHWM")
    maxrss = _ru_maxrss_bytes()
    meminfo = _read_proc_meminfo()
    cgroup = _read_cgroup_memory()
    rss_fmt = _format_bytes(rss)
    hwm_fmt = _format_bytes(hwm)
    maxrss_fmt = _format_bytes(maxrss)
    if rss_fmt:
        parts.append(f"rss={rss_fmt}")
    if hwm_fmt:
        parts.append(f"hwm={hwm_fmt}")
    if maxrss_fmt:
        parts.append(f"ru_maxrss={maxrss_fmt}")
    if "MemAvailable" in meminfo:
        parts.append(f"mem_available={_format_bytes(meminfo['MemAvailable'])}")
    if "MemTotal" in meminfo:
        parts.append(f"mem_total={_format_bytes(meminfo['MemTotal'])}")
    if "current" in cgroup:
        parts.append(f"cgroup_current={_format_bytes(cgroup['current'])}")
    if "max" in cgroup:
        parts.append(f"cgroup_max={_format_bytes(cgroup['max'])}")
    if extra:
        for key, value in extra.items():
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _log_memory(label: str, extra: Optional[dict[str, object]] = None) -> None:
    if not LOG_MEMORY:
        return
    logger.info("memory %s", _memory_snapshot(label, extra=extra))


def timeit(func):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        dt = (time.perf_counter() - t0) * 1000
        if ENABLE_PROFILING:
            print(f"[timeit] {func.__name__} took {dt:.2f} ms")
        return out

    return wrapper


def _is_gzip_file(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(2) == GZIP_MAGIC
    except Exception:
        return False


def _load_snapshot_payload(path: Path):
    opener = gzip.open if _is_gzip_file(path) else open
    expected = snapshot_build_id(path)
    with opener(path, "rb") as stream:
        stream.readline(100)
        result = pickle.load(stream)
    if getattr(getattr(result, "catalog", None), "build_id", None) != expected:
        raise ContractError(
            "Snapshot payload and header have different build identities"
        )
    return result


def _discover_snapshot(explicit: str | Path | None = None) -> Path:
    return (
        Path(explicit)
        if explicit is not None
        else current_release().artifact("snapshot")
    )


# Legacy warmed snapshots can refer to these classes during unpickling.
@dataclass
class _CharterCacheEntry:
    objects: list[Campus]
    coords_deg: Any
    tree: Any


@dataclass
class _CharterCache:
    has_numpy: bool
    entries: Dict[str, _CharterCacheEntry]


def _unit_sphere_xyz(points):
    import numpy as np

    lon, lat = np.radians(points).T
    cos_lat = np.cos(lat)
    return np.stack((cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)), axis=-1)


@dataclass
class _SphericalCampusIndex:
    campuses: list[Campus]
    points: list[tuple[float, float]]
    xyz: Any
    tree: Any

    @classmethod
    def build(cls, entries: list[tuple[Campus, tuple[float, float]]]) -> _SphericalCampusIndex:
        campuses = [campus for campus, _ in entries]
        points = [point for _, point in entries]
        xyz = _unit_sphere_xyz(points)
        try:
            from scipy.spatial import cKDTree  # type: ignore[unresolved-import]
        except ImportError:
            tree = None
        else:
            tree = cKDTree(xyz)
        return cls(campuses, points, xyz, tree)

    def candidates(
        self, origin: tuple[float, float], limit: int | None, miles: float | None
    ) -> Iterable[int]:
        import numpy as np

        if (limit is not None and limit <= 0) or (miles is not None and not miles >= 0):
            return ()
        size = len(self.campuses)
        if miles is None and (limit is None or limit >= size):
            return range(size)
        seed = _unit_sphere_xyz(origin)
        chord = (
            2 * math.sin(min(math.pi, miles / EARTH_RADIUS_MILES) / 2)
            if miles is not None else math.inf
        )
        # Expand only the candidate boundary. Final ranking and cutoff use haversine.
        slack = 8 * math.ulp(1.0)
        if self.tree is not None:
            if limit is not None and limit < size:
                kth = self.tree.query(seed, k=[limit])[0][0]
                chord = min(chord, float(kth))
            return sorted(self.tree.query_ball_point(seed, chord + slack))

        delta = self.xyz - seed
        squared = np.einsum("ij,ij->i", delta, delta)
        if limit is not None and limit < size:
            kth = np.partition(squared, limit - 1)[limit - 1]
            chord = min(chord, math.sqrt(kth))
        return np.flatnonzero(squared <= (chord + slack) ** 2).tolist()

    def rank(
        self,
        origin: tuple[float, float],
        limit: int | None,
        miles: float | None = None,
        *,
        exclude: uuid.UUID | None = None,
    ) -> list[tuple[float, Campus]]:
        candidate_limit = limit
        if exclude is not None and limit is not None:
            candidate_limit = limit + 1
        results = []
        for i in self.candidates(origin, candidate_limit, miles):
            campus = self.campuses[i]
            if campus.id != exclude:
                distance = haversine_miles(*origin, *self.points[i])
                if miles is None or distance <= miles:
                    results.append((distance, campus))
        results.sort(key=lambda row: row[0])
        return results[:limit]


@dataclass
class _CampusSpatialCaches:
    geodesic: dict[bool, _SphericalCampusIndex] | None = None
    by_school_type: dict[str, _SphericalCampusIndex] | None = None
    bounding_box: tuple[Any, list[Campus]] | None = None
    containment: tuple[Any, list[Campus]] | None = None


def _campus_point(campus: Campus) -> tuple[float, float] | None:
    point = point_xy(campus.point) or point_xy(campus.location)
    return point if point is not None and all(map(math.isfinite, point)) else None


class DataEngine:
    catalog: DatasetCatalog

    def _observation_model(self, sources: Iterable[DataRelease]) -> _ObservationModel:
        from .observations import _ObservationModel

        catalog = getattr(self, "catalog", None)
        if catalog is None:
            raise ContractError("Observation planning requires an entity build catalog")
        return _ObservationModel(catalog, sources)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: str | Path | None = None,
        *,
        search: bool = True,
        eager_indexes: bool = False,
        indexes_enabled: bool | None = None,
    ) -> "DataEngine":
        """Load a catalog-bound engine; default loading verifies the complete release.

        Explicit paths are for trusted locally built snapshots. Legacy tuple,
        dictionary, and module-remapping formats must be rebuilt from source.
        """
        if snapshot is None and not search:
            raise ContractError("Select a release or supply an explicit snapshot")
        if snapshot is None:
            release = current_release()
            path = release.artifact("snapshot")
        else:
            release = None
            path = Path(snapshot)
        obj = _load_snapshot_payload(path)
        if not isinstance(obj, cls) or not isinstance(
            getattr(obj, "catalog", None), DatasetCatalog
        ):
            raise ContractError(
                "Snapshot lacks a dataset catalog; rebuild it from source"
            )
        # Revalidate serialized catalogs; pickle construction bypasses __init__.
        obj.catalog = DatasetCatalog(obj.catalog.to_dict())
        try:
            for campus in obj._campuses.values():
                if not campus.lookup_key:
                    raise ValueError("Missing campus identity")
        except (ValueError, TypeError, AttributeError) as error:
            raise ContractError("Snapshot uses a legacy or invalid campus identity; rebuild it from source") from error
        if release and obj.catalog.build_id != release.build_id:
            raise ContractError("Snapshot belongs to a different data build")
        obj._set_indexes_enabled(
            DEFAULT_INDEXES_ENABLED if indexes_enabled is None else indexes_enabled
        )
        try:
            obj._rebuild_indexes()
        except EntityIdentityError as error:
            raise ContractError(f"Snapshot contains an invalid entity registry: {error}") from error
        obj._rehydrate_geometries()
        if eager_indexes:
            obj.warm_indexes()
        if release:
            obj.attach_boundary_store(release.artifact("boundaries"))
        return obj

    @classmethod
    def load_default(cls, *, indexes_enabled: bool | None = None) -> "DataEngine":
        """
        Load the configured verified release; missing artifacts raise an error.
        """
        return cls.from_snapshot(None, search=True, indexes_enabled=indexes_enabled)

    def __init__(
        self,
        snapshot: str | Path | None = None,
        *,
        autoload: bool = False,
        eager_indexes: bool = False,
        indexes_enabled: bool | None = None,
    ):
        if indexes_enabled is None:
            indexes_enabled = DEFAULT_INDEXES_ENABLED
        self._indexes_enabled = bool(indexes_enabled)
        self._boundary_store = None
        self._boundary_store_path = None
        # Optional early autoload from snapshot
        if snapshot is not None or autoload:
            eng = self.from_snapshot(
                snapshot,
                search=autoload and snapshot is None,
                eager_indexes=eager_indexes,
                indexes_enabled=indexes_enabled,
            )
            # Adopt the complete validated state, including catalog, measurements,
            # unresolved facts, and attached boundary store. There is one loader.
            self.__dict__.update(eng.__dict__)
            for district in self._districts.values():
                district._repo = self
            for campus in self._campuses.values():
                campus._repo = self
            return
        # Normal cold init (no snapshot)
        self._districts: EntityMap = EntityMap()
        self._campuses: EntityMap = EntityMap()
        self._campuses_by_district: Dict[uuid.UUID, List[uuid.UUID]] = defaultdict(list)
        self._xfers_out: Dict[uuid.UUID, List[Tuple[uuid.UUID, Optional[int], bool]]] = defaultdict(list)
        self._xfers_in: Dict[uuid.UUID, List[Tuple[uuid.UUID, Optional[int], bool]]] = defaultdict(list)
        self._xfers_missing = {"src": 0, "dst": 0, "either": 0}
        self._xfers_unresolved = []
        self._transfer_measurements = {}
        self._in_bulk = False
        self._clear_spatial_indexes()
        self._boundary_store = None
        self._boundary_store_path = None

        # Fast name lookups (lowercase exact match)
        self._district_by_name_lower: Dict[str, uuid.UUID] = {}
        self._campus_by_name_lower: Dict[str, uuid.UUID] = {}
        self._districts_by_number_norm: dict[str, uuid.UUID] = {}
        self._campus_by_key: dict[str, uuid.UUID] = {}
        self._districts_by_name_lower_multi: Dict[str, List[uuid.UUID]] = {}
        self._campuses_by_name_lower_multi: Dict[str, List[uuid.UUID]] = {}
        self._district_name_upper_index: List[Tuple[str, uuid.UUID]] = []
        self._campus_name_upper_index: List[Tuple[str, uuid.UUID]] = []
        self._district_wildcard_cache: Dict[str, Tuple[int, List[uuid.UUID]]] = {}
        self._campus_wildcard_cache: Dict[str, Tuple[int, List[uuid.UUID]]] = {}
        self._name_lookup_version = 0

        if eager_indexes and self._indexes_enabled:
            try:
                self.warm_indexes()
            except Exception:
                pass

    def warm_indexes(self) -> None:
        """
        Eagerly (re)build spatial and vectorized caches. Useful when the host app
        front-loads caching (e.g., teadata-app under Django) to avoid first-request
        latency. Safe to call multiple times.
        """
        if not getattr(self, "_indexes_enabled", True):
            return
        try:
            self._ensure_kdtree()
        except Exception:
            pass
        try:
            self._ensure_point_strtree()
        except Exception:
            pass
        try:
            self._ensure_all_xy_arrays()
        except Exception:
            pass
        try:
            self._ensure_charter_cache()
        except Exception:
            pass
        import gc

        gc.collect()

    def clear_indexes(self) -> None:
        """
        Force-clear all spatial, vectorized, and name-lookup caches to free up memory.
        The engine remains fully functional; caches will be rebuilt lazily on demand.
        """
        self._clear_spatial_indexes()

        # Keep identity indexes available to validate inserts even with caches cleared.
        self._campus_by_name_lower = {}
        self._district_by_name_lower = {}
        self._districts_by_name_lower_multi = {}
        self._campuses_by_name_lower_multi = {}
        self._district_name_upper_index = []
        self._campus_name_upper_index = []
        self._district_wildcard_cache = {}
        self._campus_wildcard_cache = {}
        self._name_lookup_version = getattr(self, "_name_lookup_version", 0) + 1

        if self._boundary_store:
            self._boundary_store.close()

        import gc

        gc.collect()

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
        if not target:
            return None
        try:
            did = self._district_by_name_lower.get(target)
            if did is not None:
                hit = self._districts.get(did)
                if hit is not None:
                    return hit
        except Exception:
            pass
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
        if not target:
            return None
        try:
            cid = self._campus_by_name_lower.get(target)
            if cid is not None:
                hit = self._campuses.get(cid)
                if hit is not None:
                    return hit
        except Exception:
            pass
        for c in self._campuses.values():
            if c.name.lower() == target:
                return c
        return None

    def campus_by_key(self, lookup_key: str | None) -> Optional["Campus"]:
        """Find a public TEA campus or a source-qualified private school."""
        key = normalize_campus_lookup_key(lookup_key)
        if key is None:
            return None
        return self._campuses.get(self._campus_by_key.get(key))

    def campus_by_number(self, campus_number: Any) -> Optional["Campus"]:
        """Find only a real TEA campus number; private keys are separate."""
        if campus_number is None or ":" in str(campus_number):
            return None
        return self.campus_by_key(str(campus_number))

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
        """Build a separate engine from disjoint entity identities."""
        r = DataEngine()
        with r.bulk():
            for repo in (self, other):
                for entity in repo:
                    detached = copy(entity)
                    detached._repo = None
                    copy_deferred = getattr(entity.meta, "_copy_deferred", None)
                    detached.meta = copy_deferred() if callable(copy_deferred) else copy(entity.meta)
                    if isinstance(detached, District):
                        r.add_district(detached)
                    else:
                        r.add_campus(detached)
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
                    hits: List[District] = []
                    norm = normalize_district_number_value(raw)
                    if norm and (did := self._districts_by_number_norm.get(norm)) is not None:
                        hits = [self._districts[did]]
                    return Query(hits, self)

                # 2) Name lookup (supports wildcards); case-insensitive
                target = (str(raw) if raw is not None else "").strip()
                if not target:
                    return Query([], self)

                # Allow SQL-like '%' and '_' or glob '*' and '?'
                pattern = target.replace("%", "*").replace("_", "?")
                pattern_up = pattern.upper()
                has_glob = any(ch in pattern_up for ch in ("*", "?"))

                matches: List[District] = []
                if has_glob:
                    index = getattr(self, "_district_name_upper_index", None)
                    cache = getattr(self, "_district_wildcard_cache", {})
                    version = getattr(self, "_name_lookup_version", 0)
                    ids: List[uuid.UUID] | None = None
                    cached = cache.get(pattern_up)
                    if cached and cached[0] == version:
                        ids = list(cached[1])
                    if ids is None:
                        if index:
                            ids = [
                                did
                                for name_up, did in index
                                if fnmatch.fnmatchcase(name_up, pattern_up)
                            ]
                        else:
                            ids = [
                                d.id
                                for d in self._districts.values()
                                if fnmatch.fnmatchcase(
                                    (d.name or "").upper(), pattern_up
                                )
                            ]
                        cache[pattern_up] = (version, list(ids))
                    matches = [
                        self._districts[did] for did in ids if did in self._districts
                    ]
                else:
                    key_lower = target.lower()
                    multi = getattr(self, "_districts_by_name_lower_multi", None)
                    if multi:
                        ids = multi.get(key_lower, [])
                        matches = [
                            self._districts[did]
                            for did in ids
                            if did in self._districts
                        ]
                    else:
                        for d in self._districts.values():
                            if (d.name or "").upper() == pattern_up:
                                matches.append(d)

                return Query(matches, self)

            if key == "campus":
                raw = query[1]

                if isinstance(raw, str) and raw.startswith("tefa:"):
                    campus = self.campus_by_key(raw)
                    return Query([campus] if campus is not None else [], self)

                if isinstance(raw, int) or (
                    isinstance(raw, str) and any(ch.isdigit() for ch in raw)
                ):
                    campus = self.campus_by_number(raw)
                    return Query([campus] if campus is not None else [], self)

                target = (str(raw) if raw is not None else "").strip()
                if not target:
                    return Query([], self)

                pattern = target.replace("%", "*").replace("_", "?")
                pattern_up = pattern.upper()
                has_glob = any(ch in pattern_up for ch in ("*", "?"))

                matches: List[Campus] = []
                if has_glob:
                    index = getattr(self, "_campus_name_upper_index", None)
                    cache = getattr(self, "_campus_wildcard_cache", {})
                    version = getattr(self, "_name_lookup_version", 0)
                    ids: List[uuid.UUID] | None = None
                    cached = cache.get(pattern_up)
                    if cached and cached[0] == version:
                        ids = list(cached[1])
                    if ids is None:
                        if index:
                            ids = [
                                cid
                                for name_up, cid in index
                                if fnmatch.fnmatchcase(name_up, pattern_up)
                            ]
                        else:
                            ids = [
                                c.id
                                for c in self._campuses.values()
                                if fnmatch.fnmatchcase(
                                    (c.name or "").upper(), pattern_up
                                )
                            ]
                        cache[pattern_up] = (version, list(ids))
                    matches = [
                        self._campuses[cid] for cid in ids if cid in self._campuses
                    ]
                else:
                    key_lower = target.lower()
                    multi = getattr(self, "_campuses_by_name_lower_multi", None)
                    if multi:
                        ids = multi.get(key_lower, [])
                        matches = [
                            self._campuses[cid] for cid in ids if cid in self._campuses
                        ]
                    else:
                        for c in self._campuses.values():
                            if (c.name or "").upper() == pattern_up:
                                matches.append(c)

                return Query(matches, self)

            if key == "campuses_in":
                # Accept a District instance or a Query[District]; unwrap if needed
                district = unwrap_query(query[1])
                if district is None:
                    return Query([], self)
                return Query(self.campuses_in(district), self)

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

    def attach_boundary_store(self, path: str | Path, *, max_cache: int = 16) -> None:
        catalog = read_sqlite_catalog(path)
        expected = getattr(self, "catalog", None)
        if expected is not None and catalog.build_id != expected.build_id:
            raise ContractError("Boundary store belongs to a different data release")
        store = BoundaryStore(Path(path), max_cache=max_cache)
        self._boundary_store = store
        self._boundary_store_path = str(path)
        self._district_cover_index = None

    def _missing_boundary_message(
        self,
        district: Any,
        *,
        context: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> str:
        name = getattr(district, "name", None)
        dn = getattr(district, "district_number", None)
        if not dn:
            dn = getattr(district, "_district_number_canon", None)
        ident = name or dn or getattr(district, "id", None) or "district"
        parts = [f"Boundary geometry unavailable for {ident}."]
        if context:
            parts.append(f"Spatial query '{context}' requires district boundaries.")
        if reason:
            parts.append(f"{reason}.")
        parts.append(
            "Select a complete verified release with TEADATA_RELEASE or "
            "TEADATA_RELEASE_URL, or attach a boundary store with the same catalog."
        )
        parts.append(
            "If you want boundaries embedded in the snapshot, rebuild with "
            "TEADATA_SPLIT_BOUNDARIES=0."
        )
        return " ".join(parts)

    def ensure_boundary(
        self,
        district: Any,
        *,
        required: bool = False,
        context: Optional[str] = None,
    ) -> Any:
        district = unwrap_query(district)
        if district is None:
            if required:
                raise ValueError(
                    self._missing_boundary_message(
                        "district", context=context, reason="District is missing"
                    )
                )
            return None
        boundary = district_boundary_geometry(district)
        if boundary is not None:
            if required and SHAPELY and hasattr(boundary, "geom_type") and (issue := district_boundary_issue(boundary)):
                raise ValueError(f"Cannot use boundary for {district.district_number}: {issue}")
            if getattr(district, "boundary", None) is None:
                try:
                    district.boundary = boundary
                except Exception:
                    pass
            return boundary
        store = getattr(self, "_boundary_store", None)
        if store is None:
            if required:
                raise ValueError(
                    self._missing_boundary_message(
                        district, context=context, reason="No boundary store attached"
                    )
                )
            return None
        key = getattr(district, "district_number", None)
        if not key and hasattr(district, "_district_number_canon"):
            key = getattr(district, "_district_number_canon", None)
        if not key:
            if required:
                raise ValueError(
                    self._missing_boundary_message(
                        district,
                        context=context,
                        reason="District number is missing",
                    )
                )
            return None
        geom = store.get(key)
        if geom is None:
            if required:
                raise ValueError(
                    self._missing_boundary_message(
                        district,
                        context=context,
                        reason=f"No boundary found for district_number={key}",
                    )
                )
            return None
        if required and SHAPELY and (issue := district_boundary_issue(geom)):
            raise ValueError(f"Cannot use boundary for {key}: {issue}")
        try:
            district.boundary = geom
        except Exception:
            pass
        try:
            district.polygon = geom
        except Exception:
            pass
        try:
            district._prepared = None
        except Exception:
            pass
        return geom

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_campus_spatial_caches", None)
        state["_district_cover_index"] = None
        return state

    def _clear_spatial_indexes(self) -> None:
        self._district_cover_index = None
        self._campus_spatial_caches: _CampusSpatialCaches | None = None

    def _spatial_caches(self) -> _CampusSpatialCaches:
        caches = getattr(self, "_campus_spatial_caches", None)
        if caches is None:
            caches = self._campus_spatial_caches = _CampusSpatialCaches()
        return caches

    def _drop_legacy_spatial_indexes(self) -> None:
        # Old pickles can still contain these derived attributes.
        for attr in (
            "_charter_cache",
            "_all_xy_np",
            "_kdtree",
            "_kdtree_charter",
            "_xy_deg",
            "_xy_rad",
            "_campus_list",
            "_xy_deg_charter",
            "_xy_rad_charter",
            "_campus_list_charter",
            "_point_tree",
            "_point_geoms",
            "_point_ids",
            "_geom_id_to_index",
            "_geom_wkb_to_index",
            "_xy_deg_np",
            "_campus_list_np",
            "_xy_deg_np_charter",
            "_campus_list_np_charter",
            "_all_campuses_np",
            "_xy_to_index",
        ):
            self.__dict__.pop(attr, None)

    def _set_indexes_enabled(self, enabled: bool) -> None:
        self._indexes_enabled = bool(enabled)
        if not self._indexes_enabled:
            self._clear_spatial_indexes()
        _log_memory(
            "indexes.toggle",
            {
                "enabled": self._indexes_enabled,
                "districts": len(getattr(self, "_districts", {})),
                "campuses": len(getattr(self, "_campuses", {})),
            },
        )

    def _charter_points_by_type(self) -> dict[str, list[tuple[Campus, tuple[float, float]]]]:
        entries = defaultdict(list)
        for campus in tuple(self._campuses.values()):
            if not is_charter(campus) or is_private(campus):
                continue
            school_type = (campus.school_type or "").strip()
            point = _campus_point(campus)
            if school_type and point is not None and probably_lonlat(*point):
                entries[school_type].append((campus, point))
        return entries

    def _ensure_charter_cache(self) -> dict[str, _SphericalCampusIndex]:
        caches = self._spatial_caches()
        if caches.by_school_type is None:
            caches.by_school_type = {
                school_type: _SphericalCampusIndex.build(group)
                for school_type, group in self._charter_points_by_type().items()
            }
        return caches.by_school_type

    def _rebuild_indexes(self) -> None:
        district_numbers: dict[str, uuid.UUID] = {}
        campus_keys: dict[str, uuid.UUID] = {}
        overlap = self._districts.keys() & self._campuses.keys()
        if overlap:
            raise EntityIdentityError(f"Campus and district share entity ID {min(overlap, key=str)}")
        for entities, identity_index, kind in ((self._districts, district_numbers, District), (self._campuses, campus_keys, Campus)):
            for uid, entity in entities.items():
                if not isinstance(entity, kind) or uid != entity.id:
                    raise EntityIdentityError(f"Invalid {kind.__name__} registry entry for ID {uid}")
                if entity._repo is not self:
                    raise EntityIdentityError(f"{kind.__name__} {uid} has a different registry owner")
                self._register_identity(identity_index, entity, self._entity_key(entity))
        self._districts_by_number_norm = district_numbers
        self._campus_by_key = campus_keys
        self._district_cover_index = None
        self._campuses_by_district.clear()
        for c in self._campuses.values():
            if c.district_id in self._districts:
                self._campuses_by_district[c.district_id].append(c.id)
        self._clear_spatial_indexes()
        self._drop_legacy_spatial_indexes()

        # Rebuild name indexes separately from the entity identity constraint.
        try:
            self._campus_by_name_lower = {}
            self._district_by_name_lower = {}
            self._districts_by_name_lower_multi = {}
            self._campuses_by_name_lower_multi = {}
            self._district_name_upper_index = []
            self._campus_name_upper_index = []
            self._district_wildcard_cache = {}
            self._campus_wildcard_cache = {}
            self._name_lookup_version = getattr(self, "_name_lookup_version", 0) + 1
            for did, d in self._districts.items():
                dname = d.name or ""
                key = dname.lower()
                if key:
                    ikey = sys.intern(key)
                    if ikey not in self._district_by_name_lower:
                        self._district_by_name_lower[ikey] = did
                    self._districts_by_name_lower_multi.setdefault(ikey, []).append(did)
                    self._district_name_upper_index.append((dname.upper(), did))
            for cid, c in self._campuses.items():
                cname = c.name or ""
                name_key = cname.lower()
                if name_key:
                    iname_key = sys.intern(name_key)
                    if iname_key not in self._campus_by_name_lower:
                        self._campus_by_name_lower[iname_key] = cid
                    self._campuses_by_name_lower_multi.setdefault(iname_key, []).append(
                        cid
                    )
                    self._campus_name_upper_index.append((cname.upper(), cid))
        except (ValueError, TypeError, AttributeError):
            raise
        except Exception:
            logger.exception("Unable to rebuild entity name indexes")
            self._campus_by_name_lower = {}
            self._district_by_name_lower = {}
            self._districts_by_name_lower_multi = {}
            self._campuses_by_name_lower_multi = {}
            self._district_name_upper_index = []
            self._campus_name_upper_index = []
            self._district_wildcard_cache = {}
            self._campus_wildcard_cache = {}
            self._name_lookup_version = getattr(self, "_name_lookup_version", 0) + 1

        if not hasattr(self, "_xfers_missing"):
            self._xfers_missing = {"src": 0, "dst": 0, "either": 0}

    def profile(self, on: bool = True):
        global ENABLE_PROFILING
        ENABLE_PROFILING = bool(on)

    def _ensure_point_strtree(self) -> tuple[Any, list[Campus]] | None:
        if not self._indexes_enabled or not SHAPELY:
            return None
        caches = self._spatial_caches()
        if caches.containment is None:
            from shapely.strtree import STRtree

            campuses, points = [], []
            for campus in tuple(self._campuses.values()):
                point = _campus_point(campus)
                if point is not None:
                    campuses.append(campus)
                    geometry = campus.point
                    points.append(geometry if isinstance(geometry, ShapelyPoint) else ShapelyPoint(*point))
            if campuses:
                caches.containment = STRtree(points), campuses
        return caches.containment

    def _ensure_kdtree(self) -> dict[bool, _SphericalCampusIndex]:
        if not self._indexes_enabled:
            return {}
        # Builders publish only into the generation they started in. A mutation
        # detaches it, so a slow build cannot refill a cache after invalidation.
        caches = self._spatial_caches()
        if caches.geodesic is not None:
            return caches.geodesic

        all_entries: list[tuple[Campus, tuple[float, float]]] = []
        charters: list[tuple[Campus, tuple[float, float]]] = []
        for campus in tuple(self._campuses.values()):
            point = _campus_point(campus)
            if point is None:
                continue
            entry = (campus, point)
            all_entries.append(entry)
            if is_charter(campus) and not is_private(campus):
                charters.append(entry)
        caches.geodesic = {
            charter_only: _SphericalCampusIndex.build(entries)
            for charter_only, entries in ((False, all_entries), (True, charters))
            if entries and all(probably_lonlat(*point) for _, point in entries)
        }
        return caches.geodesic

    def _ensure_all_xy_arrays(self) -> tuple[Any, list[Campus]] | None:
        if not self._indexes_enabled:
            return None
        caches = self._spatial_caches()
        if caches.bounding_box is None:
            import numpy as np

            campuses, points = [], []
            for campus in tuple(self._campuses.values()):
                point = _campus_point(campus)
                if point is not None:
                    campuses.append(campus)
                    points.append(point)
            if campuses:
                caches.bounding_box = np.array(points, dtype=float), campuses
        return caches.bounding_box

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
        if not SHAPELY or not getattr(self, "_indexes_enabled", True):
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
        index = self._ensure_all_xy_arrays()
        if index is None:
            return []

        points, campuses = index
        minx, miny, maxx, maxy = poly.bounds
        xs, ys = points.T
        mask = (xs >= minx) & (xs <= maxx) & (ys >= miny) & (ys <= maxy)
        return [campuses[i] for i in mask.nonzero()[0]]

    def _nearby_campuses(
        self,
        origin: tuple[float, float],
        *,
        limit: int | None,
        miles: float | None,
        charter_only: bool,
        geodesic: bool,
        allow_planar: bool,
    ) -> list[Campus]:
        if (limit is not None and limit <= 0) or (miles is not None and not miles >= 0):
            return []
        if not all(map(math.isfinite, origin)):
            return []
        geographic_origin = geodesic and probably_lonlat(*origin)
        if geographic_origin:
            index = self._ensure_kdtree().get(charter_only)
            if index is not None:
                return [campus for _, campus in index.rank(origin, limit, miles)]
        elif not allow_planar:
            return []

        results = []
        for campus in tuple(self._campuses.values()):
            if charter_only and (not is_charter(campus) or is_private(campus)):
                continue
            point = _campus_point(campus)
            if point is None:
                continue
            if geographic_origin and probably_lonlat(*point):
                distance = haversine_miles(*origin, *point)
            elif allow_planar:
                distance = euclidean(origin, point)
            else:
                continue
            if miles is None or distance <= miles:
                results.append((distance, campus))
        ordered = (
            sorted(results, key=lambda row: row[0])
            if limit is None else nsmallest(limit, results, key=lambda row: row[0])
        )
        return [campus for _, campus in ordered]

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
        """Return campuses within an inclusive great-circle distance in miles."""
        return self._nearby_campuses(
            (lon, lat), limit=limit, miles=miles, charter_only=charter_only,
            geodesic=True, allow_planar=False,
        )

    @timeit
    def knn_campuses(
        self, lon: float, lat: float, k: int, *, charter_only: bool = False
    ) -> List[Campus]:
        """Return up to k campuses in increasing great-circle distance."""
        return self._nearby_campuses(
            (lon, lat), limit=k, miles=None, charter_only=charter_only,
            geodesic=True, allow_planar=False,
        )

    def nearest_charter_same_type(
        self,
        campuses: Iterable[Campus],
        *,
        k: int = 1,
    ) -> Dict[str, Dict[str, Any]]:
        """Return one nearest peer charter of the same school type per campus."""
        results: Dict[str, Dict[str, Any]] = {}
        indexes = None
        groups = None
        for campus in campuses:
            results[str(campus.id)] = {"match": None, "miles": None}
            school_type = (campus.school_type or "").strip()
            point = _campus_point(campus)
            if is_private(campus) or not school_type or point is None or not probably_lonlat(*point):
                continue
            if self._indexes_enabled:
                if indexes is None:
                    indexes = self._ensure_charter_cache()
                index = indexes.get(school_type)
                ranked = index.rank(point, 1, exclude=campus.id) if index else []
                best = ranked[0] if ranked else None
            else:
                if groups is None:
                    groups = self._charter_points_by_type()
                candidates = (
                    (haversine_miles(*point, *other_point), other)
                    for other, other_point in groups.get(school_type, ())
                    if other.id != campus.id
                )
                best = min(candidates, key=lambda row: row[0], default=None)
            if best is not None:
                results[str(campus.id)] = {"match": best[1], "miles": best[0]}
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
    @staticmethod
    def _entity_key(entity: District | Campus) -> str | None:
        if isinstance(entity, Campus):
            return entity.lookup_key
        return entity.district_number_canon or None

    @staticmethod
    def _register_identity(index: dict[str, uuid.UUID], entity: District | Campus, key: str | None) -> None:
        if key is None:
            return
        existing_id = index.get(key)
        if existing_id is not None and existing_id != entity.id:
            kind = "campus lookup key" if isinstance(entity, Campus) else "district number"
            raise EntityIdentityError(f"Duplicate canonical {kind} {key!r}: {existing_id} and {entity.id}")
        index[sys.intern(key)] = entity.id

    def _add_entity(self, entity: District | Campus, entities: EntityMap, index: dict[str, uuid.UUID]) -> bool:
        other_entities = self._campuses if entities is self._districts else self._districts
        if entity.id in other_entities:
            raise EntityIdentityError(f"Campus and district share entity ID {entity.id}")
        existing = entities.get(entity.id)
        if existing is entity:
            return False
        if existing is not None:
            raise EntityIdentityError(f"Duplicate {type(entity).__name__} ID {entity.id}")
        if entity._repo is not None and entity._repo is not self:
            raise EntityIdentityError(f"{type(entity).__name__} {entity.id} already belongs to another engine")
        self._register_identity(index, entity, self._entity_key(entity))
        entities[entity.id] = entity
        entity._repo = self
        return True

    def add_district(self, d: District) -> None:
        if not isinstance(d, District):
            raise TypeError("add_district requires a District")
        if not self._add_entity(d, self._districts, self._districts_by_number_norm):
            return
        self._district_cover_index = None
        if not self._in_bulk:
            if not hasattr(self, "_district_by_name_lower"):
                self._district_by_name_lower = {}
            if not hasattr(self, "_districts_by_name_lower_multi"):
                self._districts_by_name_lower_multi = {}
            if not hasattr(self, "_district_name_upper_index"):
                self._district_name_upper_index = []
            if not hasattr(self, "_district_wildcard_cache"):
                self._district_wildcard_cache = {}
            if not hasattr(self, "_name_lookup_version"):
                self._name_lookup_version = 0
            key = (d.name or "").lower()
            if key and key not in self._district_by_name_lower:
                self._district_by_name_lower[key] = d.id
            if key:
                self._districts_by_name_lower_multi.setdefault(key, []).append(d.id)
                self._district_name_upper_index.append(((d.name or "").upper(), d.id))
            self._district_wildcard_cache = {}
            self._name_lookup_version += 1

    def add_campus(self, c: Campus) -> None:
        if not isinstance(c, Campus):
            raise TypeError("add_campus requires a Campus")
        if not self._add_entity(c, self._campuses, self._campus_by_key):
            return
        self._clear_spatial_indexes()
        # Only append to _campuses_by_district if the district exists
        if not self._in_bulk:
            if c.district_id in self._districts:
                self._campuses_by_district[c.district_id].append(c.id)
            if not hasattr(self, "_campus_by_name_lower"):
                self._campus_by_name_lower = {}
            if not hasattr(self, "_campuses_by_name_lower_multi"):
                self._campuses_by_name_lower_multi = {}
            if not hasattr(self, "_campus_name_upper_index"):
                self._campus_name_upper_index = []
            if not hasattr(self, "_campus_wildcard_cache"):
                self._campus_wildcard_cache = {}
            if not hasattr(self, "_name_lookup_version"):
                self._name_lookup_version = 0
            key = (c.name or "").lower()
            if key and key not in self._campus_by_name_lower:
                self._campus_by_name_lower[key] = c.id
            if key:
                self._campuses_by_name_lower_multi.setdefault(key, []).append(c.id)
                self._campus_name_upper_index.append(((c.name or "").upper(), c.id))
            self._campus_wildcard_cache = {}
            self._name_lookup_version += 1

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

        return EntityList([c for c in campuses if not is_private(c)])


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

        poly = self.ensure_boundary(district, required=True, context=label)
        if poly is None:
            return []

        # Quick AABB prefilter (NumPy) to cut the candidate set; then exact covers
        if SHAPELY and hasattr(poly, "bounds"):
            bbox_cands = self._bbox_candidates(poly)
            if bbox_cands:
                prep = getattr(district, "prepared", None) if poly is getattr(district, "polygon", None) else None
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
        if SHAPELY and getattr(self, "_indexes_enabled", True):
            index = self._ensure_point_strtree()
            if index is not None:
                tree, campuses = index
                try:
                    idxs = tree.query(poly, predicate="covers")
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
                        c = campuses[i]
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

    def _districts_covering(self, campus: "Campus") -> "EntityList":
        """All reported district geometries covering a point, never governance.

        A cached native polygon index avoids scanning all polygons per school.
        Boundary points can have multiple results; callers must not choose one.
        """
        point = getattr(campus, "point", None)
        if point is None:
            return EntityList([])
        if not SHAPELY or ShapelyPoint is None:
            raise RuntimeError("Indexed geographic containment requires Shapely")
        if not isinstance(point, ShapelyPoint):
            point = ShapelyPoint(*point)
        cached = getattr(self, "_district_cover_index", None)
        if cached is None:
            from shapely.strtree import STRtree
            districts, geometries = [], []
            for district in self._districts.values():
                geometry = self.ensure_boundary(district)
                if geometry is not None and district_boundary_issue(geometry) is None:
                    districts.append(district)
                    geometries.append(geometry)
            cached = (STRtree(geometries), districts)
            self._district_cover_index = cached
        tree, districts = cached
        return EntityList(sorted((districts[int(index)] for index in tree.query(point, predicate="covered_by")),
                                 key=lambda district: district.district_number))

    def private_campuses_within(self, district: Any):
        """
        Return private schools inside or on the given district's boundary.
        The boundary-inclusive predicate is polygon.covers(point).
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
        value_policy=None,
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
        and mapped to repo campus UUIDs via `_campus_by_key`.
        """
        # Filter to the requested report type if present
        if type_col in df.columns:
            df = df[df[type_col] == want_type]

        # A required source schema cannot degrade into a successful empty load.
        if (
            src_col not in df.columns
            or dst_col not in df.columns
            or count_col not in df.columns
        ):
            raise ContractError("Transfer source lacks required identifier/count columns")

        def norm(x):
            try:
                value = canonical_source_identifier(x, width=9)
            except ValueError as error:
                raise ContractError("Invalid campus identifier in transfer source") from error
            if value is None:
                raise ContractError("Missing campus identifier in transfer source")
            return value

        # Reset maps
        outgoing = defaultdict(list)  # type: ignore
        incoming = defaultdict(list)  # type: ignore

        # Reset diagnostics
        missing = {"src": 0, "dst": 0, "either": 0}

        unresolved = []
        measurements = {}
        pairs = set()
        updated_sources = set()

        # Vectorized normalization of campus numbers
        src_series = df[src_col].map(norm)
        dst_series = df[dst_col].map(norm)
        counts = df[count_col].to_numpy()

        # Precompute campus-id lookups for all observed normalized keys
        id_cache: Dict[str, Optional[uuid.UUID]] = {}
        observed_keys = set(src_series.dropna())
        observed_keys.update(dst_series.dropna())
        for normed in observed_keys:
            digits = normed[1:] if normed.startswith("'") else normed
            id_cache[normed] = self._campus_by_key.get(digits)

        src_ids = src_series.map(id_cache).to_numpy()
        dst_ids = dst_series.map(id_cache).to_numpy()

        for src_key, dst_key, src_id, dst_id, raw in zip(
            src_series.to_numpy(), dst_series.to_numpy(), src_ids, dst_ids, counts
        ):
            from .values import ValueStatus, measure, PLAIN_NUMERIC

            pair = (src_key, dst_key)
            if pair in pairs:
                raise ContractError(f"Duplicate transfer pair: {src_key} -> {dst_key}")
            pairs.add(pair)

            value = measure(raw, value_policy or PLAIN_NUMERIC, minimum=0, unit="count")
            if value.status == ValueStatus.INVALID or (
                value.numeric is not None and int(value.numeric) != value.numeric
            ):
                raise ContractError(
                    f"Invalid transfer count for {src_key} -> {dst_key}: {raw!r}"
                )
            masked = value.is_suppressed
            cnt = int(value.numeric) if value.numeric is not None else None

            if src_id is None or dst_id is None:
                unresolved.append(
                    {
                        "source_campus": src_key,
                        "destination_campus": dst_key,
                        "measurement": value.to_dict(),
                        "reason": "unmatched_entity",
                    }
                )
                if src_id is None and dst_id is None:
                    missing["either"] = missing.get("either", 0) + 1
                elif src_id is None:
                    missing["src"] = missing.get("src", 0) + 1
                else:
                    missing["dst"] = missing.get("dst", 0) + 1
                continue

            outgoing[src_id].append((dst_id, cnt, bool(masked)))
            incoming[dst_id].append((src_id, cnt, bool(masked)))
            measurements[(src_id, dst_id)] = value
            updated_sources.add(src_id)

        # Sort edges deterministically: largest counts first, then by campus name
        def _sort_key(edge):
            to_id, cnt, _ = edge
            cobj = self._campuses.get(to_id)
            return (-(cnt or -1), (cobj.name if cobj else ""))

        for k in list(outgoing.keys()):
            outgoing[k].sort(key=_sort_key)

        for k in list(incoming.keys()):
            # for transfers_in sort by the incoming count, same logic
            incoming[k].sort(
                key=lambda ed: (
                    -(ed[1] or -1),
                    (
                        self._campuses.get(ed[0]).name
                        if self._campuses.get(ed[0])
                        else ""
                    ),
                )
            )

        self._xfers_out = outgoing
        self._xfers_in = incoming
        self._xfers_missing = missing
        self._xfers_unresolved = unresolved
        self._transfer_measurements = measurements
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
        return self._nearby_campuses(
            (x, y), limit=limit, miles=max_miles, charter_only=charter_only,
            geodesic=geodesic, allow_planar=True,
        )

    @timeit
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
