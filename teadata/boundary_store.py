from __future__ import annotations

import logging
import os
import sqlite3
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .assets import ensure_local_asset, is_lfs_pointer
from .teadata_config import canonical_district_number

try:
    from .engine import _discover_snapshot  # type: ignore
except Exception:  # pragma: no cover - defensive import
    _discover_snapshot = None

try:
    from shapely import wkb as shapely_wkb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    shapely_wkb = None


_BOUNDARY_PREFIX = "boundaries_"
_LAST_LOGGED_BOUNDARY_STORE: Optional[str] = None

logger = logging.getLogger(__name__)


def boundary_store_path_for_snapshot(snapshot_path: Path) -> Optional[Path]:
    name = snapshot_path.name
    if name.startswith("repo_"):
        base = name[len("repo_") :]
        if base.endswith(".pkl.gz"):
            base = base[: -len(".pkl.gz")]
        elif base.endswith(".pkl"):
            base = base[: -len(".pkl")]
        return snapshot_path.with_name(f"{_BOUNDARY_PREFIX}{base}.sqlite")
    return None


def _newest_sqlite(folder: Path) -> Optional[Path]:
    try:
        if not folder.exists() or not folder.is_dir():
            return None
        picks = sorted(
            folder.glob(f"{_BOUNDARY_PREFIX}*.sqlite"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return picks[0] if picks else None
    except Exception:
        return None


def _log_boundary_store(path: Path, source: str) -> None:
    global _LAST_LOGGED_BOUNDARY_STORE
    try:
        path_str = str(path)
    except Exception:
        return
    if _LAST_LOGGED_BOUNDARY_STORE == path_str:
        return
    _LAST_LOGGED_BOUNDARY_STORE = path_str
    logger.info("boundary_store.discovered source=%s path=%s", source, path_str)


def _resolve_boundary_store(path: Path, source: str) -> Optional[Path]:
    resolved = ensure_local_asset(
        path, url_env="TEADATA_BOUNDARY_STORE_URL", label="boundary store"
    )
    if not resolved.exists() or is_lfs_pointer(resolved):
        logger.warning("boundary_store.unavailable source=%s path=%s", source, path)
        return None
    _log_boundary_store(resolved, source)
    return resolved


def discover_boundary_store(explicit: str | Path | None = None) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        if p.exists() and p.is_file():
            resolved = _resolve_boundary_store(p, "explicit")
            if resolved:
                return resolved

    env = os.environ.get("TEADATA_BOUNDARY_STORE")
    if env:
        p = Path(env)
        if p.exists() and p.is_file():
            resolved = _resolve_boundary_store(p, "env")
            if resolved:
                return resolved

    if _discover_snapshot:
        try:
            snap = _discover_snapshot()
            if snap:
                candidate = boundary_store_path_for_snapshot(Path(snap))
                if candidate and candidate.exists():
                    resolved = _resolve_boundary_store(candidate, "snapshot")
                    if resolved:
                        return resolved
        except Exception:
            pass

    try:
        package_dir = Path(__file__).resolve().parent
        pkg_cache = package_dir / ".cache"
        candidate = _newest_sqlite(pkg_cache)
        if candidate:
            resolved = _resolve_boundary_store(candidate, "package-cache")
            if resolved:
                return resolved
    except Exception:
        pass

    for base in Path.cwd().parents:
        candidate = _newest_sqlite(base / ".cache")
        if candidate:
            resolved = _resolve_boundary_store(candidate, "parent-cache")
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


def load_boundary(
    district_number: Optional[str],
    *,
    store_path: str | Path | None = None,
) -> Any:
    if district_number is None or shapely_wkb is None:
        return None
    path = discover_boundary_store(store_path)
    if not path:
        return None
    try:
        with sqlite3.connect(path) as conn:
            for key in _district_lookup_keys(district_number):
                row = conn.execute(
                    "SELECT wkb FROM boundaries WHERE district_number = ?",
                    (key,),
                ).fetchone()
                if row:
                    return shapely_wkb.loads(row[0])
    except Exception:
        return None
    return None


@dataclass
class BoundaryStore:
    path: Path
    max_cache: int = 16
    _conn: Optional[sqlite3.Connection] = field(default=None, init=False, repr=False)
    _cache: OrderedDict[str, Any] = field(
        default_factory=OrderedDict, init=False, repr=False
    )

    def _connect(self) -> Optional[sqlite3.Connection]:
        if self._conn is not None:
            return self._conn
        if not self.path.exists():
            return None
        uri = f"file:{self.path}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        return self._conn

    def get(self, district_number: Optional[str]) -> Any:
        if district_number is None:
            return None
        if shapely_wkb is None:
            return None
        key = canonical_district_number(district_number) or str(district_number).strip()
        if not key:
            return None
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        conn = self._connect()
        if conn is None:
            return None
        row = conn.execute(
            "SELECT wkb FROM boundaries WHERE district_number = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        geom = shapely_wkb.loads(row[0])
        self._cache[key] = geom
        if len(self._cache) > self.max_cache:
            self._cache.popitem(last=False)
        return geom

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
