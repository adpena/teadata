"""Compatibility shims for the refactored teadata package structure."""

from __future__ import annotations

from .engine import (
    DEFAULT_REPO_PKL_RELATIVE,
    DEFAULT_REPO_PKL_URL,
    ENABLE_PROFILING,
    DataEngine as _DataEngine,
    _compat_pickle_load,
    _discover_snapshot,
    timeit,
)
from collections import OrderedDict
from dataclasses import fields, is_dataclass
from typing import Any

from .entities import (
    Campus as _Campus,
    District as _District,
    EntityList,
    EntityMap,
    ReadOnlyEntityView,
    is_charter,
    is_private,
)
from .geometry import (
    GeoField,
    SHAPELY,
    ShapelyPoint,
    haversine_miles,
    point_in_polygon,
    polygon_area,
    point_xy,
    district_centroid_xy,
    probably_lonlat,
    euclidean,
)
from .grades import (
    GRADE_ALIAS_MAP,
    GRADE_IGNORE_TOKENS,
    GRADE_NAME_TO_CODE,
    GRADE_NUMBER_TO_CODE,
    GRADE_PHRASE_SUBS,
    GRADE_SEQUENCE,
    coerce_grade_bounds,
    coerce_grade_spans,
    grade_segment_to_span,
    grade_spec_to_segments,
    grade_spec_to_tokens,
    grade_token_to_code,
    grade_value_to_code,
    normalize_grade_bounds,
    spans_to_bounds,
)
from .query import Query, unwrap_query

# Maintain backwards-compatible module paths for pickled objects.
DataEngine = _DataEngine
DataEngine.__module__ = __name__
District = _District
District.__module__ = __name__
Campus = _Campus
Campus.__module__ = __name__
_point_xy = point_xy
_coerce_grade_spans = coerce_grade_spans
_spans_to_bounds = spans_to_bounds

__all__ = [
    "DataEngine",
    "District",
    "Campus",
    "Query",
    "unwrap_query",
    "ENABLE_PROFILING",
    "timeit",
    "DEFAULT_REPO_PKL_RELATIVE",
    "DEFAULT_REPO_PKL_URL",
    "_compat_pickle_load",
    "_discover_snapshot",
    "GeoField",
    "SHAPELY",
    "ShapelyPoint",
    "point_in_polygon",
    "polygon_area",
    "euclidean",
    "haversine_miles",
    "point_xy",
    "_point_xy",
    "district_centroid_xy",
    "probably_lonlat",
    "GRADE_SEQUENCE",
    "GRADE_NAME_TO_CODE",
    "GRADE_NUMBER_TO_CODE",
    "GRADE_ALIAS_MAP",
    "GRADE_IGNORE_TOKENS",
    "GRADE_PHRASE_SUBS",
    "grade_spec_to_tokens",
    "grade_spec_to_segments",
    "grade_token_to_code",
    "grade_segment_to_span",
    "grade_value_to_code",
    "normalize_grade_bounds",
    "spans_to_bounds",
    "_spans_to_bounds",
    "coerce_grade_spans",
    "coerce_grade_bounds",
    "_coerce_grade_spans",
    "is_charter",
    "is_private",
    "EntityList",
    "EntityMap",
    "ReadOnlyEntityView",
    "inspect_object",
]


def inspect_object(obj: Any) -> OrderedDict[str, Any]:
    """Return an ordered mapping of the public data stored on *obj*.

    The original ``teadata.classes.inspect_object`` helper was frequently
    used in notebooks to explore ``District`` and ``Campus`` entities. This
    compatibility shim focuses on those dataclass-based objects while
    remaining generic enough for other simple containers:

    * dataclass fields (including ``slots=True`` dataclasses) are added in the
      declared order;
    * attributes stored in ``__dict__`` come next;
    * items from a ``meta`` dictionary, when present, are exposed using the
      ``"meta.<key>"`` naming convention.

    Only public attribute names (those not starting with ``"_"``) are
    included. Values are the live attribute values, making the result handy for
    interactive inspection or quick conversion to a ``dict``/``DataFrame``.
    """

    items: "OrderedDict[str, Any]" = OrderedDict()

    def _maybe_add(name: str, value: Any) -> None:
        if not name or name.startswith("_") or name in items:
            return
        items[name] = value

    if is_dataclass(obj):
        for field in fields(obj):
            _maybe_add(field.name, getattr(obj, field.name))

    # ``slots=True`` dataclasses (e.g., District, Campus) expose ``__slots__``.
    for slot_name in getattr(type(obj), "__slots__", ()):  # type: ignore[arg-type]
        if isinstance(slot_name, str) and hasattr(obj, slot_name):
            _maybe_add(slot_name, getattr(obj, slot_name))

    if hasattr(obj, "__dict__"):
        for name, value in vars(obj).items():
            _maybe_add(name, value)

    meta = getattr(obj, "meta", None)
    if isinstance(meta, dict):
        for key, value in meta.items():
            _maybe_add(f"meta.{key}", value)

    return items
