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
    "coerce_grade_spans",
    "coerce_grade_bounds",
    "is_charter",
    "is_private",
    "EntityList",
    "EntityMap",
    "ReadOnlyEntityView",
]
