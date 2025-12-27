"""Geometry helpers and descriptors used throughout the teadata package."""

from __future__ import annotations

from typing import Any, List, Tuple
import math

try:
    from shapely.geometry import (
        MultiPolygon,
        Point as ShapelyPoint,
        Polygon as ShapelyPolygon,
    )

    SHAPELY = True
except Exception:  # pragma: no cover - optional dependency
    MultiPolygon = ShapelyPoint = ShapelyPolygon = None  # type: ignore
    SHAPELY = False


__all__ = [
    "SHAPELY",
    "ShapelyPoint",
    "ShapelyPolygon",
    "MultiPolygon",
    "point_in_polygon",
    "polygon_area",
    "euclidean",
    "haversine_miles",
    "point_xy",
    "district_centroid_xy",
    "probably_lonlat",
    "GeoField",
]


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


def haversine_miles(x1: float, y1: float, x2: float, y2: float) -> float:
    """Approximate great-circle distance between two lon/lat points in miles."""
    from math import atan2, cos, radians, sin, sqrt

    R = 3958.7613  # Earth radius in miles
    lon1, lat1, lon2, lat2 = map(radians, (x1, y1, x2, y2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def point_xy(pt: Any) -> Tuple[float, float] | None:
    """Extract (x, y) from a Shapely point or a tuple; return None if unavailable."""
    if pt is None:
        return None
    if SHAPELY:
        try:
            return (pt.x, pt.y)
        except Exception:  # pragma: no cover - shapely failure path
            pass
    if (
        isinstance(pt, tuple)
        and len(pt) == 2
        and all(isinstance(v, (int, float)) for v in pt)
    ):
        return (float(pt[0]), float(pt[1]))
    return None


def district_centroid_xy(district: Any) -> Tuple[float, float] | None:
    """Return (x, y) for the district centroid when geometry is available."""

    if district is None:
        return None

    poly = getattr(district, "polygon", None) or getattr(district, "boundary", None)
    if poly is None:
        return None

    if SHAPELY and hasattr(poly, "centroid"):
        try:
            cent = poly.centroid
            return (float(cent.x), float(cent.y))
        except Exception:  # pragma: no cover - shapely failure path
            pass

    try:
        coords = list(poly)
    except TypeError:
        return None

    cleaned: List[Tuple[float, float]] = []
    for pt in coords:
        if (
            isinstance(pt, (tuple, list))
            and len(pt) == 2
            and all(isinstance(v, (int, float)) for v in pt)
        ):
            cleaned.append((float(pt[0]), float(pt[1])))
    if not cleaned:
        return None

    area = 0.0
    cx = 0.0
    cy = 0.0
    for (x1, y1), (x2, y2) in zip(cleaned, cleaned[1:] + cleaned[:1]):
        cross = x1 * y2 - x2 * y1
        area += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    area *= 0.5
    if area == 0:
        xs = [p[0] for p in cleaned]
        ys = [p[1] for p in cleaned]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    cx = cx / (6 * area)
    cy = cy / (6 * area)
    return (cx, cy)


def probably_lonlat(x: float, y: float) -> bool:
    """Return True when (x, y) look like lon/lat coordinates."""

    return -180.0 <= x <= 180.0 and -90.0 <= y <= 90.0


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
        target = None

        if self.geom_type == "point":
            if SHAPELY:
                if isinstance(value, ShapelyPoint):
                    target = value
                elif (
                    isinstance(value, tuple)
                    and len(value) == 2
                    and all(isinstance(v, (int, float)) for v in value)
                ):
                    try:
                        target = ShapelyPoint(float(value[0]), float(value[1]))
                    except Exception:  # pragma: no cover - shapely failure path
                        target = None
            else:
                if (
                    isinstance(value, tuple)
                    and len(value) == 2
                    and all(isinstance(v, (int, float)) for v in value)
                ):
                    target = (float(value[0]), float(value[1]))

        elif self.geom_type == "polygon":
            if SHAPELY:
                if isinstance(value, (ShapelyPolygon, MultiPolygon)):
                    target = value
                elif isinstance(value, list) and all(
                    isinstance(p, tuple) and len(p) == 2 for p in value
                ):
                    try:
                        target = ShapelyPolygon(value)
                    except Exception:  # pragma: no cover - shapely failure path
                        target = None
            else:
                if isinstance(value, list) and all(
                    isinstance(p, tuple) and len(p) == 2 for p in value
                ):
                    target = value

        if target is None:
            raise TypeError(
                f"Invalid {self.geom_type} geometry for {obj.__class__.__name__}"
            )

        setattr(obj, self.private_name, target)
