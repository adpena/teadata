from __future__ import annotations

import logging
import sqlite3
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .manifest import read_sqlite_catalog
from .release import current_release
from .teadata_config import canonical_district_number

from shapely import wkb as shapely_wkb

logger = logging.getLogger(__name__)


def boundary_store_path_for_snapshot(snapshot_path: Path) -> Path:
    base = snapshot_path.name.removeprefix("repo_").removesuffix(".gz").removesuffix(".pkl")
    return snapshot_path.with_name(f"boundaries_{base}.sqlite")


def discover_boundary_store(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        read_sqlite_catalog(path)
        return path
    return current_release().artifact("boundaries")


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
        logger.exception(
            "boundary_store.load_boundary_failed path=%s district_number=%s",
            path,
            district_number,
        )
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
