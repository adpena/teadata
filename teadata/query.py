"""Query pipeline utilities and operator dispatch."""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import uuid

from .entities import (
    Campus,
    EntityList,
    EntityMap,
    ReadOnlyEntityView,
    is_charter,
    is_private,
)
from .geometry import point_xy
from .grades import coerce_grade_bounds, coerce_grade_spans

__all__ = [
    "Query",
    "EntityList",
    "EntityMap",
    "ReadOnlyEntityView",
    "unwrap_query",
]


def unwrap_query(obj: Any) -> Any:
    """If ``obj`` is a Query return its first item; otherwise return ``obj``."""

    if isinstance(obj, Query):
        return obj.first()
    return obj


class Query:
    """Lightweight, chainable view over a list of repository objects."""

    _PRESETS: Dict[str, Dict[str, Sequence[str] | Dict[str, str]]] = {
        "nearest_charter": {
            "column_order": [
                "campus_name",
                "campus_district_number",
                "campus_campus_number",
                "campus_school_type",
                "match_name",
                "match_district_number",
                "match_campus_number",
                "match_school_type",
                "distance_miles",
            ],
            "rename": {"distance_miles": "miles_to_match"},
        },
        "nearest_charter_same_type": {
            "column_order": [
                "campus_name",
                "campus_district_number",
                "campus_campus_number",
                "campus_school_type",
                "match_name",
                "match_district_number",
                "match_campus_number",
                "match_school_type",
                "distance_miles",
            ],
            "rename": {"distance_miles": "miles_to_match"},
        },
        "nearest_charter_transfer_destination": {
            "column_order": [
                "campus_name",
                "campus_district_number",
                "campus_campus_number",
                "to_name",
                "to_district_number",
                "to_campus_number",
                "to_school_type",
                "count",
                "masked",
            ]
        },
        "transfers_out": {
            "column_order": [
                "campus_name",
                "campus_district_number",
                "campus_campus_number",
                "to_name",
                "to_district_number",
                "to_campus_number",
                "to_school_type",
                "count",
                "masked",
            ]
        },
    }

    _HANDLERS: Dict[str, Callable[["Query", tuple], Any]] = {}
    _ALIASES: Dict[str, str] = {
        "where": "filter",
        "select": "map",
        "grade_overlaps": "grade_overlap",
        "grade_range_overlap": "grade_overlap",
    }

    def __init__(self, items: List[Any], repo: "DataEngine"):
        self._items = list(items)
        self._repo = repo

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
        if not self._items:
            raise AttributeError(f"'Query' object has no attribute '{name}' (empty result set)")
        return getattr(self._items[0], name)

    def __getitem__(self, idx: int) -> Any:
        return self._items[idx]

    def __bool__(self) -> bool:
        return bool(self._items)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        cls = self.__class__.__name__
        n = len(self._items)
        sample = self._items[0].__class__.__name__ if n else "None"
        return f"<{cls} len={n} first={sample}>"

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def to_dicts(self, *, include_meta: bool = True, include_geometry: bool = False) -> list[dict]:
        """Materialize the current query items as a list of dictionaries."""

        def _obj_to_basic_dict(obj, *, prefix: str = "") -> dict:
            if hasattr(obj, "to_dict"):
                d = obj.to_dict(include_meta=include_meta, include_geometry=include_geometry)
            else:
                try:
                    d = dict(vars(obj))
                except Exception:
                    d = {"value": obj}
            if prefix:
                return {f"{prefix}{k}": v for k, v in d.items()}
            return d

        rows = []
        for item in self._items:
            if isinstance(item, tuple):
                is_ncst = (
                    len(item) == 3
                    and getattr(item[0].__class__, "__name__", "") == "Campus"
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

                is_transfers = (
                    len(item) == 4
                    and getattr(item[0].__class__, "__name__", "") == "Campus"
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

                row = {}
                for i, elem in enumerate(item):
                    row.update(_obj_to_basic_dict(elem, prefix=f"p{i}_"))
                rows.append(row)
                continue

            rows.append(_obj_to_basic_dict(item))

        return rows

    def to_df(
        self,
        columns: list[str] | None = None,
        *,
        include_meta: bool = True,
        include_geometry: bool = False,
        column_order: Sequence[str] | None = None,
        rename: Dict[str, str] | None = None,
        preset: str | None = None,
    ):
        """Materialize the current query items as a pandas DataFrame.

        Parameters
        ----------
        columns : list[str] | None
            Optional subset of columns to retain.
        column_order : Sequence[str] | None
            Preferred ordering applied after materialisation. Missing columns are ignored.
        rename : Dict[str, str] | None
            Column rename mapping applied via DataFrame.rename.
        preset : str | None
            Named preset (e.g., 'nearest_charter_same_type', 'transfers_out') providing
            a default ordering/rename configuration for common tuple outputs.
        """

        try:
            import pandas as pd  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "pandas is required for .to_df(); install pandas to use this feature"
            ) from exc

        data = self.to_dicts(include_meta=include_meta, include_geometry=include_geometry)
        df = pd.DataFrame(data)

        config_order: Sequence[str] | None = None
        config_rename: Dict[str, str] = {}
        if preset is not None:
            key = preset.lower()
            if key not in self._PRESETS:
                raise ValueError(f"Unknown preset '{preset}' for Query.to_df()")
            preset_cfg = self._PRESETS[key]
            config_order = preset_cfg.get("column_order")  # type: ignore[assignment]
            config_rename = dict(preset_cfg.get("rename", {}))  # type: ignore[arg-type]

        if rename:
            config_rename.update(rename)
        if config_rename:
            df = df.rename(columns=config_rename)

        order = list(column_order or [])
        if config_order:
            order = list(config_order) if not order else list(order) + [
                c for c in config_order if c not in order
            ]
        if columns:
            order = list(columns) if not order else list(order) + [
                c for c in columns if c not in order
            ]
        if order:
            cols = [c for c in order if c in df.columns]
            cols += [c for c in df.columns if c not in cols]
            df = df[cols]
        return df

    # ------------------------------------------------------------------
    # Dispatch implementation
    # ------------------------------------------------------------------
    def __rshift__(self, op):
        if callable(op):
            return self._op_filter_callable(op)
        if not (isinstance(op, tuple) and op):
            raise ValueError(f"Unsupported >> operation on Query: {op!r}")

        key = op[0]
        if key in self._ALIASES:
            key = self._ALIASES[key]
        handler = self._HANDLERS.get(key)
        if handler is None:
            raise ValueError(f"Unsupported Query op: {op!r}")
        return handler(self, op)

    # ---------------------- individual handlers ----------------------
    def _op_filter_callable(self, predicate: Callable[[Any], bool]):
        self._items = [o for o in self._items if predicate(o)]
        return self

    def _op_filter(self, op: tuple):
        pred = op[1]
        return self._op_filter_callable(pred)

    def _op_take(self, op: tuple):
        n = int(op[1])
        self._items = self._items[:n]
        return self

    def _op_sort(self, op: tuple):
        keyfunc = op[1]
        reverse = bool(op[2]) if len(op) >= 3 else False
        self._items.sort(key=keyfunc, reverse=reverse)
        return self

    def _op_map(self, op: tuple):
        func = op[1]
        return [func(o) for o in self._items]

    def _op_distinct(self, op: tuple):
        keyfunc = op[1]
        seen = set()
        out = []
        for obj in self._items:
            key = keyfunc(obj)
            if key in seen:
                continue
            seen.add(key)
            out.append(obj)
        self._items = out
        return self

    def _op_grade_overlap(self, op: tuple):
        if len(op) == 2:
            low, high = coerce_grade_bounds(op[1])
        elif len(op) >= 3:
            low, high = coerce_grade_bounds(op[1], op[2])
        else:
            raise ValueError("grade_overlap requires a grade specification (low[, high])")
        if low is None and high is None:
            raise ValueError("grade_overlap received an empty/unknown grade span")

        want_low, want_high = low, high

        def _item_grade_spans(obj: Any) -> list[tuple[Optional[int], Optional[int]]]:
            return coerce_grade_spans(obj)

        def _overlaps(obj: Any) -> bool:
            spans = _item_grade_spans(obj)
            for low_val, high_val in spans:
                if low_val is None or high_val is None:
                    continue
                if not (want_high < low_val or high_val < want_low):
                    return True
            return False

        self._items = [o for o in self._items if _overlaps(o)]
        return self

    def _op_campuses_in(self, op: tuple):
        campuses: List[Any] = []
        for item in self._items:
            if getattr(item.__class__, "__name__", "") == "District":
                campuses.extend(self._repo.campuses_in(item))
            elif getattr(item.__class__, "__name__", "") == "Campus":
                campuses.append(item)
        self._items = campuses
        return self

    def _op_private_campuses_in(self, op: tuple):
        max_m = float(op[1]) if len(op) >= 2 and op[1] is not None else None

        campuses: List[Any] = []
        for item in self._items:
            if getattr(item.__class__, "__name__", "") == "District":
                campuses.extend(self._repo.private_campuses_in(item, max_miles=max_m))
            elif getattr(item.__class__, "__name__", "") == "Campus" and is_private(item):
                campuses.append(item)

        if max_m is not None and campuses:
            allowed_cache: Dict[uuid.UUID, set[uuid.UUID]] = {}
            filtered: List[Any] = []
            for campus in campuses:
                district = campus.district
                if district is None:
                    continue
                did = district.id
                allowed = allowed_cache.get(did)
                if allowed is None:
                    allowed = {
                        c.id for c in self._repo.private_campuses_in(district, max_miles=max_m)
                    }
                    allowed_cache[did] = allowed
                if campus.id in allowed:
                    filtered.append(campus)
            campuses = filtered

        self._items = campuses
        return self

    def _resolve_seed_coords(self, op: tuple, *, offset: int = 1) -> Tuple[float, float]:
        if len(op) <= offset:
            raise ValueError("operation requires coordinates or a seed item in the chain")
        seed = op[offset]
        seed = unwrap_query(seed)
        if hasattr(seed, "point"):
            xy = point_xy(getattr(seed, "point"))
            if xy is not None:
                return xy
        if not isinstance(seed, (list, tuple)):
            if not self._items:
                raise ValueError(
                    "operation requires coordinates or a seed item with geometry in the chain"
                )
            seed = self._items[0]
        if hasattr(seed, "point"):
            xy = point_xy(getattr(seed, "point"))
            if xy is None:
                raise ValueError(
                    "operation requires coordinates or a seed item with geometry in the chain"
                )
            return xy
        coords = seed
        if not (isinstance(coords, (list, tuple)) and len(coords) == 2):
            raise ValueError(
                "operation requires coordinates or a seed item with geometry in the chain"
            )
        return (float(coords[0]), float(coords[1]))

    def _op_nearest(self, op: tuple):
        charter = op[0] == "nearest_charter"
        if len(op) >= 2 and op[1] is not None:
            coords = self._resolve_seed_coords(op)
        else:
            if not self._items:
                raise ValueError(
                    "nearest/nearest_charter requires coordinates or a seed item with geometry in the chain"
                )
            seed = self._items[0]
            coords = point_xy(getattr(seed, "point", None))
            if coords is None:
                raise ValueError(
                    "nearest/nearest_charter requires coordinates or a seed item with geometry in the chain"
                )
        n = int(op[2]) if len(op) >= 3 else 1
        max_m = float(op[3]) if len(op) >= 4 and op[3] is not None else None
        items = self._repo.nearest_campuses(
            coords[0],
            coords[1],
            limit=n,
            charter_only=charter,
            max_miles=max_m,
            geodesic=True,
        )
        return Query(items, self._repo)

    def _op_radius(self, op: tuple):
        if len(op) < 3:
            raise ValueError("radius requires coordinates and a distance in miles")
        coords = self._resolve_seed_coords(op)
        miles = float(op[2])
        limit = int(op[3]) if len(op) >= 4 and op[3] is not None else None
        charter_only = bool(op[4]) if len(op) >= 5 else False
        items = self._repo.campuses_within_radius(
            coords[0], coords[1], miles, charter_only=charter_only, limit=limit
        )
        return Query(items, self._repo)

    def _op_knn(self, op: tuple):
        if len(op) < 3:
            raise ValueError("knn requires coordinates and k")
        coords = self._resolve_seed_coords(op)
        k = int(op[2])
        charter_only = bool(op[3]) if len(op) >= 4 else False
        items = self._repo.knn_campuses(
            coords[0], coords[1], k, charter_only=charter_only
        )
        return Query(items, self._repo)

    def _op_within(self, op: tuple):
        if len(op) < 3:
            raise ValueError("within requires a seed and target")
        seed = unwrap_query(op[1]) if len(op) >= 2 else None
        target = unwrap_query(op[2]) if len(op) >= 3 else None
        charter_only = bool(op[3]) if len(op) >= 4 else False
        covers = bool(op[4]) if len(op) >= 5 else False
        if target is None:
            if seed is None and self._items:
                seed = self._items[0]
            if seed is None:
                raise ValueError(
                    "within requires a target polygon/district or a seed item in the chain"
                )
            target = seed
        return Query(
            self._repo.campuses_within(target, charter_only=charter_only, covers=covers),
            self._repo,
        )

    def _op_nearest_charter_same_type(self, op: tuple):
        k = int(op[1]) if len(op) >= 2 else 1
        campuses: List[Any] = []
        for item in self._items:
            if getattr(item.__class__, "__name__", "") == "Campus":
                campuses.append(item)
            elif getattr(item.__class__, "__name__", "") == "District":
                campuses.extend(self._repo.campuses_in(item))
        if not campuses:
            self._items = []
            return self
        res = self._repo.nearest_charter_same_type(campuses, k=k)
        out = []
        for c in campuses:
            r = res.get(str(c.id), {"match": None, "miles": None})
            out.append((c, r["match"], r["miles"]))
        self._items = out
        return self

    def _op_nearest_charter_transfer_destination(self, op: tuple):
        campuses: List[Any] = []
        for item in self._items:
            if getattr(item.__class__, "__name__", "") == "Campus":
                campuses.append(item)
            elif getattr(item.__class__, "__name__", "") == "District":
                campuses.extend(self._repo.campuses_in(item))
        if not campuses:
            self._items = []
            return self
        res = self._repo.nearest_charter_transfer_destination(campuses)
        out = []
        for c in campuses:
            r = res.get(str(c.id), {"match": None, "miles": None})
            out.append((c, r["match"], r["miles"]))
        self._items = out
        return self

    def _op_transfers_out(self, op: tuple):
        charter_only = False
        pred_to = None
        if len(op) >= 2:
            arg = op[1]
            if isinstance(arg, bool):
                charter_only = arg
            elif callable(arg):
                pred_to = arg

        campuses = [
            it
            for it in self._items
            if getattr(it.__class__, "__name__", "") == "Campus"
        ]
        rows = []
        for c in campuses:
            for to_c, cnt, masked in self._repo.transfers_out(c):
                if charter_only and not (to_c is not None and is_charter(to_c)):
                    continue
                if pred_to is not None:
                    try:
                        if not pred_to(to_c):
                            continue
                    except Exception:
                        continue
                rows.append((c, to_c, cnt, masked))
        self._items = rows
        return self

    def _op_transfers_in(self, op: tuple):
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

    def _op_where_to(self, op: tuple):
        pred = op[1]
        new_items = []
        for it in self._items:
            if isinstance(it, tuple) and len(it) >= 2:
                to_c = it[1]
                try:
                    if pred(to_c):
                        new_items.append(it)
                except Exception:
                    pass
        self._items = new_items
        return self


def _register(name: str, func: Callable[[Query, tuple], Any]):
    Query._HANDLERS[name] = func


_register("filter", Query._op_filter)
_register("take", Query._op_take)
_register("sort", Query._op_sort)
_register("map", Query._op_map)
_register("distinct", Query._op_distinct)
_register("grade_overlap", Query._op_grade_overlap)
_register("campuses_in", Query._op_campuses_in)
_register("private_campuses_in", Query._op_private_campuses_in)
_register("nearest", Query._op_nearest)
_register("nearest_charter", Query._op_nearest)
_register("radius", Query._op_radius)
_register("knn", Query._op_knn)
_register("within", Query._op_within)
_register("nearest_charter_same_type", Query._op_nearest_charter_same_type)
_register(
    "nearest_charter_transfer_destination", Query._op_nearest_charter_transfer_destination
)
_register("transfers_out", Query._op_transfers_out)
_register("transfers_in", Query._op_transfers_in)
_register("where_to", Query._op_where_to)
