"""Canonical school identity/coordinate presentation shared by batch and apps.

These pure helpers do not interpret masking or infer dataset periods. They are
moved from the app so its scalar and materialized attribute paths share rules.
"""
from __future__ import annotations

from .teadata_config import normalize_campus_number_value, normalize_district_number_value

def clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned
        return ""
    return str(value)

def campus_district_name(campus) -> str:
    district = getattr(campus, "district", None)
    if district is not None:
        name = getattr(district, "name", None)
        if name:
            return str(name)
    meta_name = getattr(campus, "meta", {}).get("district_name")
    if meta_name:
        return str(meta_name)
    return ""

def valid_latlon(lat: float, lon: float) -> bool:
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return False
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0

def looks_like_texas(lat: float, lon: float) -> bool:
    return 25.0 <= lat <= 37.0 and -107.0 <= lon <= -93.0

def campus_coordinates(campus) -> tuple[float | None, float | None]:
    coords = getattr(campus, "coords", None)
    if not coords:
        lat = getattr(campus, "lat", None)
        lon = getattr(campus, "lon", None)
        if lat is None or lon is None:
            return (None, None)
        coords = (lon, lat)
    try:
        if hasattr(coords, "x") and hasattr(coords, "y"):
            lon = float(getattr(coords, "x"))
            lat = float(getattr(coords, "y"))
        else:
            first, second = coords
            lon = float(first)
            lat = float(second)
    except Exception:
        return (None, None)

    candidates = [(lat, lon), (lon, lat)]
    for lat_candidate, lon_candidate in candidates:
        if looks_like_texas(lat_candidate, lon_candidate):
            return (lat_candidate, lon_candidate)
    for lat_candidate, lon_candidate in candidates:
        if valid_latlon(lat_candidate, lon_candidate):
            return (lat_candidate, lon_candidate)
    return (None, None)

def district_name(entity) -> str:
    return getattr(entity, "name", "")

def district_number(entity) -> str:
    slug = normalize_district_number_value(getattr(entity, "district_number", None))
    if slug:
        return slug
    meta = getattr(entity, "meta", {}) or {}
    fallback = meta.get("district_number")
    if fallback:
        return normalize_district_number_value(fallback) or ""
    return ""

def district_type(entity) -> str:
    raw_type = getattr(entity, "district_type", None) or getattr(entity, "districtType", None)
    if raw_type:
        normalized = str(raw_type).strip().lower()
        if normalized in {"charter", "isd"}:
            return normalized
    if is_charter(entity):
        return "charter"
    return "isd"

def is_charter(entity) -> bool:
    flag = getattr(entity, "is_charter", None)
    if flag is None:
        flag = getattr(entity, "isCharter", None)
    if flag is None:
        meta = getattr(entity, "meta", {}) or {}
        flag = meta.get("is_charter")
        if flag is None:
            flag = meta.get("isCharter")
        if flag is None:
            flag = meta.get("charter")
    if flag is None:
        campuses = getattr(entity, "campuses", None) or []
        for campus in campuses:
            if getattr(campus, "is_charter", False):
                return True
    return bool(flag)


def attribute_value(entity, name: str, *, scope: str) -> object:
    """Evaluate only the requested attribute, preserving sparse getter laziness."""
    if scope not in {"campus", "district"}:
        raise ValueError("Unknown attribute scope")
    if name == "district_name":
        return campus_district_name(entity) if scope == "campus" else district_name(entity)
    if name == "is_charter":
        return is_charter(entity)
    if scope == "campus":
        if name in {"campus_name", "campus_grade_range"}:
            return clean_text(getattr(entity, "name" if name == "campus_name" else "grade_range", ""))
        if name == "campus_number":
            return normalize_campus_number_value(getattr(entity, "campus_number", None)) or ""
        if name == "is_private":
            return bool(getattr(entity, "is_private", False))
        if name in {"campus_latitude", "campus_longitude"}:
            coordinates = campus_coordinates(entity)
            return coordinates[0 if name == "campus_latitude" else 1]
    else:
        if name == "district_number":
            return district_number(entity)
        if name == "district_type":
            return district_type(entity)
    raise ValueError(f"Unknown {scope} attribute: {name}")


def entity_attributes(entity, *, scope: str) -> dict[str, object]:
    """Named attribute projection for a normalized EntityStore context."""
    values: dict[str, object] = {
        name: attribute_value(entity, name, scope=scope) for name in ("district_name", "is_charter")
    }
    if scope == "campus":
        latitude, longitude = campus_coordinates(entity)
        values.update({name: attribute_value(entity, name, scope=scope)
                       for name in ("campus_name", "campus_number", "campus_grade_range")})
        values.update(
            campus_latitude=latitude, campus_longitude=longitude,
            is_private=attribute_value(entity, "is_private", scope=scope),
        )
    else:
        values.update({name: attribute_value(entity, name, scope=scope)
                       for name in ("district_number", "district_type")})
    return values
