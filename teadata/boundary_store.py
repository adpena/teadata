from __future__ import annotations

import sqlite3
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .teadata_config import canonical_district_number

try:
    from shapely import wkb as shapely_wkb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    shapely_wkb = None


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
