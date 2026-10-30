import uuid
import math
import pickle
from pathlib import Path
import gzip
import json
import pandas as pd
import os
import sqlite3
import sys
import tempfile
from contextvars import ContextVar
from urllib.parse import quote
from teadata.attributes import campus_district_name, is_traditional_district

from teadata.manifest import BuildRecorder, ContractError, write_sqlite_catalog
from teadata.provenance import ProvenanceStore
from teadata.identifiers import canonical_source_identifier, campus_lookup_key, normalize_campus_lookup_key
from teadata.release import publish_release, SNAPSHOT_MAGIC
from teadata.transfer_summary import transfer_measurement, normalized_transfer_measurement

from datetime import datetime, date
from typing import Any, Iterable, Mapping, MutableMapping, Optional, List

from teadata import teadata_config as _cfg_mod
from teadata.classes import (
    Campus,
    DataEngine,
    District,
    coerce_grade_spans,
    haversine_miles,
)
from teadata.map_store import map_store_path_for_snapshot
from teadata.entity_store import entity_store_path_for_snapshot
from teadata.persistence.sqlalchemy_store import (
    create_engine as _sql_create_engine,
    create_sessionmaker as _sql_create_sessionmaker,
    ensure_schema as _sql_ensure_schema,
    export_dataengine as _sql_export_dataengine,
)
from teadata.teadata_config import (
    canonical_campus_number,
    canonical_district_number,
    load_config,
)
from teadata.enrichment.districts import enrich_districts_from_config
from teadata.enrichment.campuses import (
    enrich_campuses_from_config,
    DEFAULT_PEIMS_FINANCIAL_COLUMNS,
)
from teadata.enrichment.charter_networks import add_charter_networks_from_config

try:  # optional performance dependency
    import polars as pl
except Exception:  # pragma: no cover - optional dependency
    pl = None  # type: ignore[assignment]

CFG = str(
    Path(
        os.environ.get(
            "TEADATA_CONFIG",
            str(Path(__file__).resolve().with_name("teadata_sources.yaml")),
        )
    )
    .expanduser()
    .resolve()
)
YEAR = (
    int(os.environ["TEADATA_BUILD_YEAR"])
    if os.environ.get("TEADATA_BUILD_YEAR")
    else None
)

SPLIT_BOUNDARIES = os.getenv("TEADATA_SPLIT_BOUNDARIES", "1") not in (
    "0",
    "false",
    "False",
    None,
)

BUILD_MAP_STORE = os.getenv("TEADATA_BUILD_MAP_STORE", "1") not in (
    "0",
    "false",
    "False",
    None,
)

BUILD_ENTITY_STORE = os.getenv("TEADATA_BUILD_ENTITY_STORE", "1") not in (
    "0",
    "false",
    "False",
    None,
)


_STATEWIDE_DISTRICT_NUMBER_ALIASES: list[str] = [
    "Organization  Number",
    "Organization Number",
    "OrganizationNumber",
    "Organization #",
    "Organization ID",
    "Organization Id",
    "OrganizationID",
    "Organization Code",
    "OrganizationCode",
]

for _alias in _cfg_mod._DEFAULT_DISTRICT_ALIASES:
    if _alias not in _STATEWIDE_DISTRICT_NUMBER_ALIASES:
        _STATEWIDE_DISTRICT_NUMBER_ALIASES.append(_alias)


def _record_value_is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    return False


def _get_record_value(record: Mapping[str, Any], aliases: Iterable[str]) -> Any:
    for alias in aliases:
        value = record.get(alias)
        if not _record_value_is_missing(value):
            return value

    lower_map: dict[str, str] = {}
    normalized_map: dict[str, str] = {}
    for key in record.keys():
        if not isinstance(key, str):
            continue
        lower_map.setdefault(key.lower(), key)
        normalized_map.setdefault(" ".join(key.split()).lower(), key)

    for alias in aliases:
        key = lower_map.get(alias.lower())
        if key is None:
            continue
        value = record.get(key)
        if not _record_value_is_missing(value):
            return value

    for alias in aliases:
        normalized_alias = " ".join(alias.split()).lower()
        key = normalized_map.get(normalized_alias)
        if key is None:
            continue
        value = record.get(key)
        if not _record_value_is_missing(value):
            return value

    return None


def parse_date(val: Optional[str]) -> Optional[date]:
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    s = str(val).strip()
    if not s:
        return None
    for fmt in ("%m/%d/%Y", "%m/%d/%Y %I:%M:%S %p"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Unrecognized date format: {val}")


def _normalize_aea_raw(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, bool):
        return "Y" if value else "N"
    if isinstance(value, (int, float)):
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


def _district_lookup_keys(district_number: Optional[str]) -> list[str]:
    """Return canonical lookup aliases for a district number."""

    if not district_number:
        return []

    keys: list[str] = []
    primary = str(district_number)
    if primary:
        keys.append(primary)

    digits = (
        primary[1:]
        if isinstance(primary, str) and primary.startswith("'") and len(primary) > 1
        else primary
    )

    if digits and digits not in keys:
        keys.append(digits)

    if isinstance(digits, str) and digits.isdigit():
        normalized = str(int(digits))
        if normalized not in keys:
            keys.append(normalized)

    return keys




def _private_coerce_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        if not math.isfinite(val):
            return None
        return int(val)
    text = str(val).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def _private_coerce_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    if isinstance(val, (int, float)):
        out = float(val)
        if math.isfinite(out):
            return out
        return None
    text = str(val).strip()
    if not text:
        return None
    try:
        out = float(text)
    except Exception:
        return None
    if math.isfinite(out):
        return out
    return None


def _private_coerce_bool_flag(val: Any) -> bool:
    if val is None:
        return False
    try:
        if pd.isna(val):
            return False
    except Exception:
        pass
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    text = str(val).strip().lower()
    if not text:
        return False
    if text in {"1", "y", "yes", "true", "t"}:
        return True
    if text in {"0", "n", "no", "false", "f"}:
        return False
    return False




def _tefa_grade_label(code: Optional[int]) -> Optional[str]:
    if code is None:
        return None
    if code == -1:
        return "Pre-K"
    if code == 0:
        return "K"
    if 1 <= code <= 12:
        return str(code)
    return None


def _normalize_tefa_display_grade(display_grade_range: Any) -> Optional[str]:
    if display_grade_range is None:
        return None
    try:
        if pd.isna(display_grade_range):
            return None
    except Exception:
        pass
    text = str(display_grade_range).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered.startswith("grades "):
        text = text[7:].strip()
    text = text.replace("\N{EN DASH}", "-").replace("\N{EM DASH}", "-")
    text = text.replace("PreK", "Pre-K").replace("PREK", "Pre-K")
    text = text.replace("Pre K", "Pre-K")
    return text or None


def _tefa_grade_range(
    min_grade: Any,
    max_grade: Any,
    display_grade_range: Any,
) -> Optional[str]:
    low = _private_coerce_int(min_grade)
    high = _private_coerce_int(max_grade)

    if low is not None and high is not None and low > high:
        low, high = high, low

    low_label = _tefa_grade_label(low)
    high_label = _tefa_grade_label(high)

    if low_label and high_label:
        if low_label == high_label:
            return low_label
        return f"{low_label}-{high_label}"
    if low_label:
        return low_label
    if high_label:
        return high_label

    return _normalize_tefa_display_grade(display_grade_range)


def _tefa_school_type(
    min_grade: Any,
    max_grade: Any,
    is_pre_k: Any,
    is_elementary: Any,
    is_middle: Any,
    is_high: Any,
) -> str:
    low = _private_coerce_int(min_grade)
    high = _private_coerce_int(max_grade)
    if low is not None and high is not None and low > high:
        low, high = high, low

    pre_k = _private_coerce_bool_flag(is_pre_k)
    elementary = _private_coerce_bool_flag(is_elementary)
    middle = _private_coerce_bool_flag(is_middle)
    high_flag = _private_coerce_bool_flag(is_high)
    core_flags = sum(int(flag) for flag in (elementary, middle, high_flag))

    if core_flags >= 2:
        return "Elementary/Secondary"
    if high_flag:
        return "High School"
    if middle:
        if low is not None and low >= 7:
            return "Junior High School"
        return "Middle School"
    if elementary or pre_k:
        return "Elementary School"

    if high is not None and high <= 5:
        return "Elementary School"
    if low is not None and low >= 9:
        return "High School"
    if low is not None and high is not None:
        if low <= 5 and high >= 6:
            return "Elementary/Secondary"
        if low >= 7 and high <= 9:
            return "Junior High School"
        if low >= 6 and high <= 8:
            return "Middle School"
    return "Other"








def _private_coords(lon_raw: Any, lat_raw: Any) -> Optional[tuple[float, float]]:
    lon_val = _private_coerce_float(lon_raw)
    lat_val = _private_coerce_float(lat_raw)
    if lon_val is None or lat_val is None:
        return None
    return (lon_val, lat_val)


def _coerce_numeric(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_percent_enrollment_change(campus: Optional[Campus]) -> Optional[float]:
    if campus is None:
        return None
    try:
        return round(campus.enrollment_percent_change() * 100.0, 1)
    except ValueError:
        return None


def _materialize_percent_enrollment_change(repo: DataEngine) -> int:
    materialized = 0
    for campus in getattr(repo, "_campuses", {}).values():
        if campus is None:
            continue
        meta = getattr(campus, "meta", None)
        if not isinstance(meta, MutableMapping):
            meta = dict(meta) if meta is not None else {}
            campus.meta = meta

        percent_change = _compute_percent_enrollment_change(campus)
        if percent_change is None:
            meta["percent_enrollment_change"] = "N/A"
            continue
        meta["percent_enrollment_change"] = percent_change
        materialized += 1
    return materialized


def run_enrichments(repo: DataEngine) -> None:
    """Run required enrichments using each source's selected contract."""
    cfg = load_config(CFG)
    for scope, enrich in (
        ("district", enrich_districts_from_config),
        ("campus", enrich_campuses_from_config),
    ):
        _, count = enrich(
            repo,
            CFG,
            "accountability",
            YEAR,
            select=["overall_rating"],
            aliases={"overall_rating": "rating"},
            reader_kwargs={"reader_profile": "ratings"},
        )
        if count == 0:
            raise ContractError(f"No {scope} accountability records matched")
        _, count = enrich(repo, CFG, f"{scope}_tapr_student_staff_profile", YEAR)
        if count == 0:
            raise ContractError(f"No {scope} TAPR records matched")
    _, frame = cfg.load_df("campus_transfer_reports", YEAR)
    count = repo.apply_transfers_from_dataframe(
        frame, value_policy=cfg.value_policies["campus_transfers"]
    )
    if count == 0:
        raise ContractError("No campus transfer records matched")
    for dataset, select in (
        ("campus_peims_financials", DEFAULT_PEIMS_FINANCIAL_COLUMNS),
        ("campus_tapr_historical_enrollment", None),
        ("campus_planned_closures", None),
    ):
        _, count = enrich_campuses_from_config(repo, CFG, dataset, YEAR, select=select)
        if count == 0:
            raise ContractError(f"No campus records matched: {dataset}")


_BUILD_DIRECTORY: ContextVar[Path | None] = ContextVar(
    "teadata_build_directory", default=None
)


def _repo_cache_dir() -> Path:
    d = _BUILD_DIRECTORY.get() or Path(".cache")
    d.mkdir(exist_ok=True)
    return d


def _snapshot_path(districts_fp: str, campuses_fp: str) -> Path:
    d = _repo_cache_dir()
    tag = f"repo_{Path(districts_fp).stem}_{Path(campuses_fp).stem}.pkl"
    return d / tag


def _boundary_store_path(districts_fp: str, campuses_fp: str) -> Path:
    d = _repo_cache_dir()
    tag = f"boundaries_{Path(districts_fp).stem}_{Path(campuses_fp).stem}.sqlite"
    return d / tag


def _map_store_path(districts_fp: str, campuses_fp: str) -> Path:
    snap = _snapshot_path(districts_fp, campuses_fp)
    candidate = map_store_path_for_snapshot(snap)
    if candidate is not None:
        return candidate
    d = _repo_cache_dir()
    tag = f"map_payloads_{Path(districts_fp).stem}_{Path(campuses_fp).stem}.sqlite"
    return d / tag


def _entity_store_path(districts_fp: str, campuses_fp: str) -> Path:
    snap = _snapshot_path(districts_fp, campuses_fp)
    candidate = entity_store_path_for_snapshot(snap)
    if candidate is not None:
        return candidate
    d = _repo_cache_dir()
    tag = f"entities_{Path(districts_fp).stem}_{Path(campuses_fp).stem}.sqlite"
    return d / tag


def _prepare_repo_for_pickle(repo: DataEngine) -> None:
    """Drop transient spatial indexes so the pickle is small and portable."""
    for attr in (
        "_district_cover_index",
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
        "_xy_deg_np",
        "_campus_list_np",
        "_xy_deg_np_charter",
        "_campus_list_np_charter",
        "_all_xy_np",
        "_all_campuses_np",
        "_xy_to_index",
    ):
        if hasattr(repo, attr):
            setattr(repo, attr, None)


def _strip_repo_boundaries(repo: DataEngine) -> None:
    for district in repo._districts.values():
        district.boundary = None
        district._polygon = None
        district._prepared = None


def _save_boundary_store(
    repo: DataEngine, districts_fp: str, campuses_fp: str
) -> Optional[Path]:
    if not SPLIT_BOUNDARIES:
        return None
    path = _boundary_store_path(districts_fp, campuses_fp)
    tmp_path = path.with_suffix(".sqlite.tmp")
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        pass

    try:
        conn = sqlite3.connect(tmp_path)
        write_sqlite_catalog(conn, repo.catalog)
        conn.execute(
            "CREATE TABLE boundaries (district_number TEXT PRIMARY KEY, wkb BLOB)"
        )
        rows = []
        for district in repo._districts.values():
            district_number = canonical_district_number(
                getattr(district, "district_number", None)
            )
            if not district_number:
                continue
            boundary = getattr(district, "boundary", None) or getattr(
                district, "polygon", None
            )
            if boundary is None:
                continue
            try:
                wkb = boundary.wkb
            except Exception as exc:
                raise ContractError("Required map payload failed") from exc
            rows.append((district_number, sqlite3.Binary(wkb)))
        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO boundaries (district_number, wkb) VALUES (?, ?)",
                rows,
            )
        conn.commit()
        conn.close()
        tmp_path.replace(path)
        return path
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise


TEXAS_BOUNDS = (
    (25.5, -106.65),
    (36.5, -93.5),
)




def _display_campus_number(value: Any) -> str:
    number = normalize_campus_lookup_key(value)
    return number or ""


def _campus_profile_url(campus_number: Any) -> str:
    slug = normalize_campus_lookup_key(campus_number)
    if not slug:
        return ""
    return f"/campuses/profiles/{quote(slug, safe='')}/"


def _map_campus_name(record_name: Any, campus: Optional[Campus], campus_number: Any = None) -> str:
    """Preserve human labels; strict key parsing only detects ID-shaped labels."""
    name = record_name.strip() if isinstance(record_name, str) else ""
    if name:
        if campus_number:
            try:
                record_key = normalize_campus_lookup_key(name)
            except ValueError:
                record_key = None
            if record_key and record_key == normalize_campus_lookup_key(campus_number):
                name = ""
        if name and name.isdigit() and len(name) >= 6:
            name = ""
    if name:
        return name
    if campus is None:
        return ""
    value = (getattr(campus, "name", None) or getattr(campus, "campus_name", None)
             or getattr(campus, "campus_name_long", None))
    return value.strip() if isinstance(value, str) else ""


def _is_charter_district(district: District) -> bool:
    return district.meta.get("organization_sub_type") == "CHARTER"


def _resolve_district_type(
    district: District, district_type: Optional[str] = None
) -> str:
    if district_type:
        normalized = str(district_type).strip().lower()
        if normalized in {"isd", "charter"}:
            return normalized
    return "charter" if _is_charter_district(district) else "isd"


def _build_campus_lookup(
    repo: DataEngine,
    district: District,
    campus_numbers: Optional[set[str]] = None,
    extra_campuses: Optional[list[Campus]] = None,
) -> dict:
    campus_index: dict = {}
    campus_numbers = campus_numbers or set()
    campus_numbers = {
        normalize_campus_lookup_key(number)
        for number in campus_numbers
        if normalize_campus_lookup_key(number)
    }

    def _store_campus(campus: Campus) -> None:
        canonical = campus.lookup_key
        if not canonical:
            return
        if canonical in campus_index:
            return
        campus_index[canonical] = campus

    campus_iter = list(getattr(district, "campuses", None) or [])
    if extra_campuses:
        campus_iter.extend(c for c in extra_campuses if c is not None)
    for campus in campus_iter:
        _store_campus(campus)

    missing_numbers = campus_numbers.difference(campus_index.keys())
    for campus_number in missing_numbers:
        candidates = (campus_number,)
        campus_obj = None
        for candidate in candidates:
            try:
                campus_obj = repo.campus_by_key(candidate)
            except Exception:
                campus_obj = None
            if campus_obj is not None:
                break
        if campus_obj is not None:
            _store_campus(campus_obj)

    return campus_index


def _valid_latlon(lat: Any, lon: Any) -> bool:
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return False
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def _looks_like_texas(lat: float, lon: float) -> bool:
    return 25.0 <= lat <= 37.0 and -107.0 <= lon <= -93.0


def _haversine_distance(
    origin_latlon: Optional[tuple], dest_latlon: Optional[tuple]
) -> Optional[float]:
    if not origin_latlon or not dest_latlon:
        return None
    try:
        origin_lat, origin_lon = origin_latlon
        dest_lat, dest_lon = dest_latlon
        if not (
            _valid_latlon(origin_lat, origin_lon) and _valid_latlon(dest_lat, dest_lon)
        ):
            return None
        return haversine_miles(origin_lon, origin_lat, dest_lon, dest_lat)
    except Exception:
        return None


def _get_campus_stat_value(campus: Optional[Campus], key: str) -> Any:
    if campus is None:
        return None

    candidates = [key]
    if key.startswith("campus_"):
        candidates.append(key[len("campus_") :])

    for candidate in candidates:
        value = getattr(campus, candidate, None)
        if value not in (None, ""):
            return value

    meta = getattr(campus, "meta", {}) or {}
    for candidate in candidates:
        if candidate in meta and meta[candidate] not in (None, ""):
            return meta[candidate]
        prefixed = f"campus_{candidate}"
        if prefixed in meta and meta[prefixed] not in (None, ""):
            return meta[prefixed]
    return None


def _select_preferred_origin(origins: list[dict]) -> Optional[dict]:
    best = None
    for entry in origins:
        if not isinstance(entry, dict):
            continue
        distance = entry.get("distance")
        count_value = entry.get("count") or 0
        if best is None:
            best = entry
            continue
        best_distance = best.get("distance")
        best_count = best.get("count") or 0
        if distance is not None and best_distance is not None:
            if distance < best_distance:
                best = entry
            elif distance == best_distance and count_value > best_count:
                best = entry
        elif distance is not None and best_distance is None:
            best = entry
        elif distance is None and best_distance is None and count_value > best_count:
            best = entry
    return best


def _extract_latlon(
    campus_index: dict, campus_number: Any
) -> Optional[tuple[float, float]]:
    if campus_number is None:
        return None
    campus = campus_index.get(normalize_campus_lookup_key(campus_number))
    if not campus:
        campus = campus_index.get(str(campus_number))
    if not campus:
        return None

    coords = getattr(campus, "coords", None)
    if coords is None:
        return None

    try:
        if hasattr(coords, "x") and hasattr(coords, "y"):
            lon = float(coords.x)
            lat = float(coords.y)
        else:
            first, second = coords
            lon = float(first)
            lat = float(second)
    except (TypeError, ValueError):
        return None

    candidates = [
        (lat, lon),
        (lon, lat),
    ]

    for candidate in candidates:
        if _looks_like_texas(*candidate):
            return candidate

    for candidate in candidates:
        if _valid_latlon(*candidate):
            return candidate

    return None


def _format_stat_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            if pd.isna(value):
                return None
        if float(value).is_integer():
            return f"{int(value):,}"
        return f"{float(value):,.2f}"
    if hasattr(value, "item"):
        return _format_stat_value(value.item())
    text = str(value).strip()
    return text or None


def _group_transfer_rows(
    working_df: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the declared pair grain without summing or inventing counts."""
    from teadata.identifiers import canonical_source_identifier

    group_columns = ["campus_campus_number", "to_campus_number"]
    columns = group_columns + ["count", "masked"]
    if not set(columns).issubset(working_df.columns):
        raise ContractError("Transfer projection lacks required pair/count columns")
    result = working_df[columns].copy()
    for column in group_columns:
        try:
            result[column] = result[column].map(lambda value: normalize_campus_lookup_key(canonical_source_identifier(value, width=9)))
        except ValueError as error:
            raise ContractError("Transfer projection has an invalid campus identifier") from error
        if result[column].isna().any():
            raise ContractError("Transfer projection has a missing campus identifier")
    if result.duplicated(group_columns).any():
        raise ContractError("Duplicate transfer pair in map projection")
    values = [normalized_transfer_measurement(row["count"], row["masked"])
              for row in result.to_dict(orient="records")]
    result["count"] = pd.Series([value.numeric for value in values], index=result.index, dtype=object)
    result["status"] = [value.status.value for value in values]
    result["measurement"] = [value.to_dict() for value in values]
    return result


def _collect_charter_transfer_entries(
    repo: DataEngine, district: District, *, only_closures: bool = False
) -> list[dict]:
    charter_entries: list[dict] = []
    origin_candidates: list[Campus] = []
    campus_numbers: set[str] = set()

    district_campuses = list(repo.campuses_in(district))

    for campus in district_campuses:
        if campus is None or not getattr(campus, "is_charter", False):
            continue
        if only_closures and not getattr(campus, "facing_closure", False):
            continue

        canonical_number = normalize_campus_lookup_key(
            getattr(campus, "campus_number", None)
        )
        inbound_edges = []
        edge_rows = [edge for edge in repo.transfers_in(campus) if edge and edge[0] is not None]
        for origin_campus, count, masked in edge_rows:
            if origin_campus is None:
                continue
            if getattr(origin_campus, "is_charter", False) or getattr(
                origin_campus, "is_private", False
            ):
                continue
            inbound_edges.append(
                {
                    "campus": origin_campus,
                    "count": count,
                    "masked": bool(masked),
                    "measurement": transfer_measurement(repo, origin_campus, campus, count, masked),
                }
            )
            origin_candidates.append(origin_campus)
        charter_entries.append(
            {
                "campus": campus,
                "canonical_number": canonical_number,
                "edges": inbound_edges,
            }
        )
        if canonical_number:
            campus_numbers.add(canonical_number)

    campus_lookup = _build_campus_lookup(
        repo,
        district,
        campus_numbers=campus_numbers,
        extra_campuses=origin_candidates,
    )

    for entry in charter_entries:
        campus_number = entry.get("canonical_number")
        latlon = _extract_latlon(campus_lookup, campus_number)
        entry["latlon"] = latlon
        processed_edges = []
        for edge in entry.get("edges", []):
            origin_campus = edge.get("campus")
            origin_number = normalize_campus_lookup_key(
                getattr(origin_campus, "campus_number", None)
            )
            origin_latlon = _extract_latlon(campus_lookup, origin_number)
            distance = _haversine_distance(latlon, origin_latlon)
            value = edge["measurement"]
            count_value = value.numeric
            processed_edges.append(
                {
                    "campus": origin_campus,
                    "canonical_number": origin_number,
                    "latlon": origin_latlon,
                    "distance": distance,
                    "count": count_value,
                    "masked": bool(edge.get("masked")),
                    "status": value.status.value,
                    "measurement": value.to_dict(),
                }
            )
        entry["edges"] = processed_edges
        entry["nearest_origin"] = _select_preferred_origin(processed_edges)

    return charter_entries


def _build_charter_map_payload(repo: DataEngine, district: District) -> dict:
    catalog = repo.catalog
    payload = {
        "aisdCampuses": [],
        "charterCampuses": [],
        "privateCampuses": [],
        "maxTransferCount": 0,
        "bounds": None,
        "districtBoundary": None,
    }

    if district is None:
        payload["bounds"] = TEXAS_BOUNDS
        return payload

    entries = _collect_charter_transfer_entries(
        repo,
        district,
        only_closures=False,
    )
    if not entries:
        payload["bounds"] = TEXAS_BOUNDS
        return payload

    try:
        private_candidates = list(repo.private_campuses_within(district)) if is_traditional_district(district) else []
    except Exception:
        private_candidates = []

    campus_lookup = _build_campus_lookup(
        repo,
        district,
        extra_campuses=private_candidates,
    )

    def _clean_text(value: Any) -> str:
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                return cleaned
        return ""

    def _get_school_type(campus_obj: Optional[Campus]) -> str:
        if campus_obj is None:
            return "Other"
        value = getattr(campus_obj, "school_type", None) or getattr(
            campus_obj, "schoolType", None
        )
        text = _clean_text(value)
        return text or "Other"

    charter_field_map = {
        "grade_range": "Grade Range",
        "enrollment": catalog.metric_label("enrollment", "Enrollment"),
        "overall_rating": catalog.metric_label("rating", "Overall Rating"),
    }
    origin_field_map = {
        "grade_range": "Grade Range",
        "enrollment": catalog.metric_label("enrollment", "Enrollment"),
        "overall_rating": catalog.metric_label("rating", "Overall Rating"),
        "total_estimated_charter_transfers_out": catalog.period_label(
            "campus_transfer_reports", short=True
        )
        + " Total Estimated Charter Transfers Out",
    }

    origin_lookup: dict[str, dict] = {}
    charter_markers: list[dict] = []
    all_latlons: list[tuple[float, float]] = []
    max_transfer = 0

    def _collect_stats(
        campus_obj: Optional[Campus], field_map: dict[str, str]
    ) -> list[dict]:
        stats = []
        for column, label in field_map.items():
            value = _format_stat_value(_get_campus_stat_value(campus_obj, column))
            if value not in (None, "", "nan"):
                stats.append(
                    {
                        "id": column.removeprefix("campus_").removeprefix("to_"),
                        "label": label,
                        "value": value,
                    }
                )
        return stats

    for entry in entries:
        campus = entry.get("campus")
        canonical_number = entry.get("canonical_number")
        latlon = entry.get("latlon")
        if latlon and _valid_latlon(*latlon):
            all_latlons.append(latlon)
            lat, lon = latlon
        else:
            lat = None
            lon = None

        transfers_in = []
        origin_numbers = set()
        for edge in entry.get("edges", []):
            origin = edge.get("campus")
            origin_number = edge.get("canonical_number")
            origin_numbers.add(origin_number)
            origin_latlon = edge.get("latlon")
            if origin_latlon and _valid_latlon(*origin_latlon):
                all_latlons.append(origin_latlon)
            count_value = edge.get("count")
            if count_value and count_value > max_transfer:
                max_transfer = count_value

            transfers_in.append(
                {
                    "from": origin_number,
                    "from_name": _map_campus_name(None, origin, origin_number),
                    "count": None if edge.get("masked") else count_value,
                    "masked": bool(edge.get("masked")),
                    "status": edge["status"],
                    "measurement": edge["measurement"],
                }
            )

            if origin_number:
                origin_lat = origin_lon = None
                if origin_latlon and _valid_latlon(*origin_latlon):
                    origin_lat, origin_lon = origin_latlon
                facing_closure = bool(getattr(origin, "facing_closure", False))
                origin_entry = origin_lookup.setdefault(
                    origin_number,
                    {
                        "campusNumber": origin_number,
                        "campusNumberDisplay": _display_campus_number(origin_number),
                        "name": _map_campus_name(None, origin, origin_number),
                        "profileUrl": _campus_profile_url(origin_number),
                        "districtName": campus_district_name(origin),
                        "lat": origin_lat,
                        "lon": origin_lon,
                        "schoolType": _get_school_type(origin),
                        "gradeRange": _format_stat_value(
                            _get_campus_stat_value(origin, "grade_range")
                        ),
                        "isCharter": False,
                        "is_charter": False,
                        "facingClosure": facing_closure,
                        "stats": _collect_stats(origin, origin_field_map),
                        "transfersOut": [],
                    },
                )
                origin_entry["transfersOut"].append(
                    {
                        "to": canonical_number,
                        "to_name": _map_campus_name(None, campus, canonical_number),
                        "count": None if edge.get("masked") else count_value,
                        "masked": bool(edge.get("masked")),
                        "status": edge["status"],
                        "measurement": edge["measurement"],
                    }
                )

        charter_markers.append(
            {
                "campusNumber": canonical_number,
                "campusNumberDisplay": _display_campus_number(canonical_number),
                "name": _map_campus_name(None, campus, canonical_number),
                "profileUrl": _campus_profile_url(canonical_number),
                "districtName": campus_district_name(campus),
                "lat": lat,
                "lon": lon,
                "schoolType": _get_school_type(campus),
                "gradeRange": _format_stat_value(
                    _get_campus_stat_value(campus, "grade_range")
                ),
                "isCharter": True,
                "is_charter": True,
                "stats": _collect_stats(campus, charter_field_map),
                "transfersIn": sorted(
                    transfers_in,
                    key=lambda item: (
                        item["count"] is None,
                        -(item["count"] or 0),
                        item.get("from") or "",
                    ),
                ),
                "num_district_campus_transfer_origins": len(
                    [number for number in origin_numbers if number]
                ),
            }
        )

    payload["charterCampuses"] = charter_markers
    payload["aisdCampuses"] = list(origin_lookup.values())
    payload["maxTransferCount"] = max_transfer

    all_private_entries = []
    for campus in private_candidates:
        campus_number = getattr(campus, "campus_number", None)
        canonical_number = normalize_campus_lookup_key(campus_number)
        latlon = _extract_latlon(campus_lookup, campus.lookup_key)
        if not latlon or not _valid_latlon(*latlon):
            continue
        all_latlons.append(latlon)
        lat, lon = latlon
        raw_grade_range = getattr(campus, "grade_range", None) or getattr(
            campus, "gradeRange", None
        )
        grade_range = _format_stat_value(raw_grade_range)
        stats = _map_stats_for_campus(campus)
        all_private_entries.append(
            {
                "campusNumber": canonical_number,
                "campusNumberDisplay": None,
                "name": _map_campus_name(
                    getattr(campus, "name", None), campus, campus_number
                ),
                "profileUrl": _campus_profile_url(campus.lookup_key),
                "districtName": campus_district_name(campus),
                "lat": lat,
                "lon": lon,
                "gradeRange": grade_range,
                "lookupKey": campus.lookup_key, "sourceNamespace": campus.source_namespace, "sourceId": campus.source_id,
                "isPrivate": True,
                "stats": stats,
            }
        )
    payload["privateCampuses"] = all_private_entries

    valid_points = [(lat, lon) for lat, lon in all_latlons if _valid_latlon(lat, lon)]
    if valid_points:
        lats = [lat for lat, _ in valid_points]
        lons = [lon for _, lon in valid_points]
        padding = 0.02
        payload["bounds"] = [
            (min(lats) - padding, min(lons) - padding),
            (max(lats) + padding, max(lons) + padding),
        ]

    if payload["bounds"] is None:
        payload["bounds"] = TEXAS_BOUNDS

    return payload


def build_closures_map_payload(
    repo: DataEngine, district: District, *, district_type: Optional[str] = None
) -> dict:
    catalog = repo.catalog
    try:
        from shapely.geometry import mapping as shapely_mapping  # type: ignore
    except Exception:
        shapely_mapping = None

    district_mode = _resolve_district_type(district, district_type)
    if district_mode == "charter":
        return _build_charter_map_payload(repo, district)

    payload = {
        "aisdCampuses": [],
        "charterCampuses": [],
        "privateCampuses": [],
        "maxTransferCount": 0,
        "bounds": None,
        "districtBoundary": None,
    }

    if district is None:
        return payload

    boundary = getattr(district, "boundary", None) or getattr(district, "polygon", None)
    if boundary is None:
        try:
            repo.ensure_boundary(district)
            boundary = getattr(district, "boundary", None) or getattr(
                district, "polygon", None
            )
        except Exception:
            boundary = None
    if boundary is not None:
        try:
            if shapely_mapping:
                payload["districtBoundary"] = shapely_mapping(boundary)
        except Exception:
            payload["districtBoundary"] = None
        try:
            if hasattr(boundary, "bounds"):
                minx, miny, maxx, maxy = boundary.bounds
                sw = (float(miny), float(minx))
                ne = (float(maxy), float(maxx))
                if _valid_latlon(*sw) and _valid_latlon(*ne):
                    payload["bounds"] = [sw, ne]
        except Exception:
            payload["bounds"] = None

    transfer_rows = repo >> ("campuses_in", district) >> ("transfers_out", True)
    required_columns = {
        "campus_campus_number",
        "campus_name",
        "to_campus_number",
        "to_name",
        "count",
        "masked",
        "campus_num_charter_transfer_destinations",
        "campus_num_charter_transfer_destinations_masked",
        "campus_total_unmasked_charter_transfers_out",
        "campus_total_estimated_masked_charter_transfers_out",
        "campus_total_estimated_charter_transfers_out",
        "to_is_charter",
        "campus_is_charter",
    }
    required_columns.update(
        {
            "campus_grade_range",
            "campus_enrollment",
            "campus_overall_rating",
            "to_grade_range",
            "to_enrollment",
            "to_overall_rating",
        }
    )
    transfer_df = transfer_rows.to_df(columns=sorted(required_columns))

    if transfer_df.empty:
        return payload

    origin_field_map = {
        "campus_grade_range": "Grade Range",
        "campus_enrollment": catalog.metric_label("enrollment", "Enrollment"),
        "campus_overall_rating": catalog.metric_label("rating", "Overall Rating"),
        "campus_num_charter_transfer_destinations": catalog.period_label(
            "campus_transfer_reports", short=True
        )
        + " Number of Charter Campuses Receiving Student Transfers - TOTAL",
        "campus_total_estimated_charter_transfers_out": catalog.period_label(
            "campus_transfer_reports", short=True
        )
        + " Total Estimated Charter Transfers Out",
    }
    charter_field_map = {
        "to_grade_range": "Grade Range",
        "to_enrollment": catalog.metric_label("enrollment", "Enrollment"),
        "to_overall_rating": catalog.metric_label("rating", "Overall Rating"),
    }

    available_columns = [col for col in required_columns if col in transfer_df.columns]
    if (
        "campus_campus_number" not in available_columns
        or "to_campus_number" not in available_columns
    ):
        raise ContractError("Transfer projection lacks required campus identifiers")

    working_df = transfer_df[available_columns].copy()

    for column in (
        "campus_num_charter_transfer_destinations",
        "campus_num_charter_transfer_destinations_masked",
        "campus_total_unmasked_charter_transfers_out",
    ):
        if column in working_df.columns:
            working_df[column] = pd.to_numeric(working_df[column], errors="coerce")

    grouped = _group_transfer_rows(working_df)
    working_df["campus_campus_number"] = grouped["campus_campus_number"]
    working_df["to_campus_number"] = grouped["to_campus_number"]

    campus_numbers = set(working_df["campus_campus_number"].dropna())
    campus_numbers.update(working_df["to_campus_number"].dropna())

    try:
        private_candidates = list(repo.private_campuses_within(district)) if is_traditional_district(district) else []
    except Exception:
        private_candidates = []

    private_campuses = [
        campus for campus in private_candidates if getattr(campus, "is_private", False)
    ]

    campus_lookup = _build_campus_lookup(
        repo, district, campus_numbers, extra_campuses=private_campuses
    )

    def _clean_text(value: Any) -> str:
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                return cleaned
        return ""

    def _collect_stats(row: Any, field_map: dict[str, str]) -> list[dict]:
        stats = []
        for column, label in field_map.items():
            value = None
            if isinstance(row, dict):
                value = _format_stat_value(row.get(column))
            else:
                value = _format_stat_value(getattr(row, column, None))
            if value is not None and value != "nan":
                stats.append(
                    {
                        "id": column.removeprefix("campus_").removeprefix("to_"),
                        "label": label,
                        "value": value,
                    }
                )
        return stats

    transfers_out: dict[str, list[dict]] = {}
    transfers_in: dict[str, list[dict]] = {}

    max_transfer = 0
    for row in grouped.itertuples(index=False):
        origin = row.campus_campus_number
        dest = row.to_campus_number
        masked = bool(row.masked)
        destination_obj = campus_lookup.get(dest)
        origin_obj = campus_lookup.get(origin)
        if destination_obj is None or origin_obj is None:
            raise ContractError("Transfer projection refers to a missing campus entity")
        value = transfer_measurement(repo, origin_obj, destination_obj, row.count, masked)
        count_value = value.numeric
        if count_value is not None and count_value > max_transfer:
            max_transfer = count_value

        transfers_out.setdefault(origin, []).append(
            {
                "to": dest,
                "to_name": _map_campus_name(None, destination_obj, dest),
                "count": count_value,
                "masked": masked,
                "status": value.status.value,
                "measurement": value.to_dict(),
            }
        )
        transfers_in.setdefault(dest, []).append(
            {
                "from": origin,
                "from_name": _map_campus_name(None, origin_obj, origin),
                "count": count_value,
                "masked": masked,
                "status": value.status.value,
                "measurement": value.to_dict(),
            }
        )

    origin_columns = [
        "campus_campus_number",
        "campus_name",
        "campus_is_charter",
    ] + list(origin_field_map.keys())
    origin_columns = [col for col in origin_columns if col in working_df.columns]
    origin_details = (
        working_df[origin_columns]
        .drop_duplicates(subset=["campus_campus_number"])
        .where(pd.notnull, None)
    )

    charter_columns = ["to_campus_number", "to_name"] + list(charter_field_map.keys())
    charter_columns = [col for col in charter_columns if col in working_df.columns]
    charter_details = (
        working_df[charter_columns]
        .drop_duplicates(subset=["to_campus_number"])
        .where(pd.notnull, None)
    )

    all_latlons: list[tuple[float, float]] = []

    for record in origin_details.itertuples(index=False):
        campus_number = getattr(record, "campus_campus_number", None)
        campus_obj = campus_lookup.get(campus_number)
        latlon = _extract_latlon(campus_lookup, campus_number)
        if latlon:
            all_latlons.append(latlon)
            lat, lon = latlon
        else:
            lat = None
            lon = None
        facing_closure = False
        if campus_obj is not None:
            facing_closure = bool(getattr(campus_obj, "facing_closure", False))
        payload["aisdCampuses"].append(
            {
                "campusNumber": campus_number,
                "campusNumberDisplay": _display_campus_number(campus_number),
                "name": _map_campus_name(
                    getattr(record, "campus_name", None), campus_obj, campus_number
                ),
                "profileUrl": _campus_profile_url(campus_number),
                "districtName": campus_district_name(campus_obj),
                "lat": lat,
                "lon": lon,
                "facingClosure": facing_closure,
                "gradeRange": _format_stat_value(
                    getattr(record, "campus_grade_range", None)
                ),
                "isCharter": bool(getattr(record, "campus_is_charter", False)),
                "is_charter": bool(getattr(record, "campus_is_charter", False)),
                "stats": _collect_stats(record, origin_field_map),
                "transfersOut": sorted(
                    transfers_out.get(campus_number, []),
                    key=lambda item: (
                        item["count"] is None,
                        -(item["count"] or 0),
                        item["to"],
                    ),
                ),
            }
        )

    for record in charter_details.itertuples(index=False):
        campus_number = getattr(record, "to_campus_number", None)
        campus_obj = campus_lookup.get(campus_number)
        latlon = _extract_latlon(campus_lookup, campus_number)
        if latlon:
            all_latlons.append(latlon)
            lat, lon = latlon
        else:
            lat = None
            lon = None
        campus_transfers_in = transfers_in.get(campus_number, [])

        payload["charterCampuses"].append(
            {
                "campusNumber": campus_number,
                "campusNumberDisplay": _display_campus_number(campus_number),
                "name": _map_campus_name(
                    getattr(record, "to_name", None), campus_obj, campus_number
                ),
                "profileUrl": _campus_profile_url(campus_number),
                "districtName": campus_district_name(campus_obj),
                "lat": lat,
                "lon": lon,
                "gradeRange": _format_stat_value(
                    getattr(record, "to_grade_range", None)
                ),
                "isCharter": bool(getattr(record, "to_is_charter", True)),
                "is_charter": bool(getattr(record, "to_is_charter", True)),
                "stats": _collect_stats(record, charter_field_map),
                "transfersIn": sorted(
                    campus_transfers_in,
                    key=lambda item: (
                        item["count"] is None,
                        -(item["count"] or 0),
                        item["from"],
                    ),
                ),
                "num_district_campus_transfer_origins": len(
                    {
                        transfer.get("from")
                        for transfer in campus_transfers_in
                        if transfer and transfer.get("from")
                    }
                ),
            }
        )

    for campus in private_campuses:
        campus_number = getattr(campus, "campus_number", None)
        canonical_number = normalize_campus_lookup_key(campus_number)
        latlon = _extract_latlon(campus_lookup, campus.lookup_key)
        if not latlon:
            continue

        all_latlons.append(latlon)
        lat, lon = latlon

        raw_grade_range = getattr(campus, "grade_range", None)
        if raw_grade_range is None:
            raw_grade_range = getattr(campus, "gradeRange", None)
        grade_range = _format_stat_value(raw_grade_range)

        stats = _map_stats_for_campus(campus)

        payload["privateCampuses"].append(
            {
                "campusNumber": canonical_number,
                "campusNumberDisplay": None,
                "name": _map_campus_name(
                    getattr(campus, "name", None), campus, campus_number
                ),
                "profileUrl": _campus_profile_url(campus.lookup_key),
                "districtName": campus_district_name(campus),
                "lat": lat,
                "lon": lon,
                "gradeRange": grade_range,
                "lookupKey": campus.lookup_key, "sourceNamespace": campus.source_namespace, "sourceId": campus.source_id,
                "isPrivate": True,
                "stats": stats,
            }
        )

    payload["maxTransferCount"] = max_transfer

    if payload["bounds"] is None and all_latlons:
        valid_points = [
            (lat, lon) for lat, lon in all_latlons if _valid_latlon(lat, lon)
        ]
        if valid_points:
            lats = [lat for lat, _ in valid_points]
            lons = [lon for _, lon in valid_points]
            padding = 0.02
            payload["bounds"] = [
                (min(lats) - padding, min(lons) - padding),
                (max(lats) + padding, max(lons) + padding),
            ]

    return payload


def _clean_text(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        return text
    if value is None:
        return ""
    return str(value).strip()


def _campus_rating_value(campus: Optional[Campus]) -> Optional[str]:
    return campus.metric("rating").value if campus is not None else None


def _campus_enrollment_value(campus: Optional[Campus]) -> Optional[int]:
    return campus.metric("enrollment").numeric if campus is not None else None


def _campus_grade_range(campus: Optional[Campus]) -> str:
    if campus is None:
        return ""
    grade_range = getattr(campus, "grade_range", None)
    if grade_range:
        return _clean_text(grade_range)
    meta = getattr(campus, "meta", {}) or {}
    for key in ("grade_range", "campus_grade_range"):
        value = meta.get(key)
        if value:
            return _clean_text(value)
    return ""


def _format_number(value: Optional[int]) -> str:
    if value is None:
        return ""
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


def _map_stats_for_campus(campus: Campus) -> list[dict]:
    catalog = campus.dataset_catalog
    stats = [
        {
            "id": "grade_range",
            "label": "Grade range",
            "value": _campus_grade_range(campus),
        }
    ]
    for metric, title in (("enrollment", "Enrollment"), ("rating", "Overall Rating")):
        value = campus.metric(metric)
        stats.append(
            {
                "id": "overall_rating" if metric == "rating" else metric,
                "label": catalog.metric_label(metric, title),
                "value": value.display(),
                "measurement": value.to_dict(),
            }
        )
    return stats


def _extract_campus_latlon(
    campus: Optional[Campus],
) -> tuple[Optional[float], Optional[float]]:
    if campus is None:
        return None, None
    coords = getattr(campus, "coords", None)
    if coords is None:
        return None, None
    try:
        if hasattr(coords, "x") and hasattr(coords, "y"):
            lon = float(coords.x)
            lat = float(coords.y)
        else:
            first, second = coords
            lon = float(first)
            lat = float(second)
    except (TypeError, ValueError):
        return None, None

    candidates = [
        (lat, lon),
        (lon, lat),
    ]
    for candidate in candidates:
        if _looks_like_texas(*candidate):
            return candidate
    for candidate in candidates:
        if _valid_latlon(*candidate):
            return candidate
    return None, None


def _serialize_map_campus(campus: Optional[Campus]) -> dict[str, object]:
    if campus is None:
        return {}
    campus_number = getattr(campus, "campus_number", None)
    canonical_number = normalize_campus_lookup_key(campus_number)
    lat, lon = _extract_campus_latlon(campus)
    return {
        "lookupKey": campus.lookup_key,
        "sourceNamespace": campus.source_namespace, "sourceId": campus.source_id,
        "campusNumber": _display_campus_number(canonical_number) if canonical_number is not None else None,
        "canonicalCampusNumber": canonical_number,
        "name": _clean_text(getattr(campus, "name", "")),
        "districtName": campus_district_name(campus),
        "lat": lat,
        "lon": lon,
        "profileUrl": _campus_profile_url(campus.lookup_key),
        "charter": bool(getattr(campus, "is_charter", False)),
        "isPrivate": bool(getattr(campus, "is_private", False)),
        "stats": _map_stats_for_campus(campus),
    }


def _serialize_transfer_destination(
    campus: Optional[Campus],
    count: Any,
    masked: Any,
    *, measurement=None,
) -> dict[str, object]:
    payload = _serialize_map_campus(campus)
    value = measurement or normalized_transfer_measurement(count, masked)
    payload.update(
        {
            "count": value.numeric,
            "masked": value.is_suppressed,
            "status": value.status.value,
            "measurement": value.to_dict(),
        }
    )
    return payload


def _bounds_from_points(points: list[tuple[Optional[float], Optional[float]]]):
    valid_points = [
        (lat, lon)
        for lat, lon in points
        if lat is not None and lon is not None and _valid_latlon(lat, lon)
    ]
    if not valid_points:
        return None
    lats = [lat for lat, _ in valid_points]
    lons = [lon for _, lon in valid_points]
    padding = 0.02
    return [
        (min(lats) - padding, min(lons) - padding),
        (max(lats) + padding, max(lons) + padding),
    ]


def _bounds_from_radius(
    lat: Optional[float],
    lon: Optional[float],
    radius_miles: Optional[float],
):
    if lat is None or lon is None or radius_miles in (None, ""):
        return None
    if not _valid_latlon(lat, lon):
        return None
    try:
        radius = float(radius_miles)
    except (TypeError, ValueError):
        return None
    if radius <= 0:
        return None

    miles_per_degree_lat = 69.0
    try:
        lat_delta = radius / miles_per_degree_lat
    except ZeroDivisionError:
        lat_delta = 0.0

    lon_scale = math.cos(math.radians(float(lat)))
    miles_per_degree_lon = 69.172 * lon_scale if lon_scale else 0.0
    lon_delta = radius / miles_per_degree_lon if miles_per_degree_lon else 0.0

    sw_lat = lat - lat_delta
    sw_lon = lon - lon_delta
    ne_lat = lat + lat_delta
    ne_lon = lon + lon_delta

    return [(sw_lat, sw_lon), (ne_lat, ne_lon)]


def _grade_spans(campus: Optional[Campus]) -> list[tuple[Optional[int], Optional[int]]]:
    if campus is None:
        return []
    spans = getattr(campus, "grade_range_code_spans", None)
    if spans:
        return [tuple(span) for span in spans if isinstance(span, (list, tuple))]
    grade_range = getattr(campus, "grade_range", None)
    if grade_range:
        try:
            return [tuple(span) for span in coerce_grade_spans(grade_range)]
        except Exception:
            return []
    meta = getattr(campus, "meta", {}) or {}
    meta_grade = meta.get("grade_range") or meta.get("campus_grade_range")
    if meta_grade:
        try:
            return [tuple(span) for span in coerce_grade_spans(meta_grade)]
        except Exception:
            return []
    return []


def _spans_overlap(
    span_a: tuple[Optional[int], Optional[int]],
    span_b: tuple[Optional[int], Optional[int]],
) -> bool:
    low_a, high_a = span_a
    low_b, high_b = span_b
    if low_a is None and high_a is None:
        return True
    if low_b is None and high_b is None:
        return True
    low_a = -float("inf") if low_a is None else low_a
    high_a = float("inf") if high_a is None else high_a
    low_b = -float("inf") if low_b is None else low_b
    high_b = float("inf") if high_b is None else high_b
    return high_a >= low_b and high_b >= low_a


def _grades_overlap(campus_a: Campus, campus_b: Campus) -> bool:
    spans_a = _grade_spans(campus_a)
    spans_b = _grade_spans(campus_b)
    if not spans_a or not spans_b:
        return True
    for span_a in spans_a:
        for span_b in spans_b:
            if _spans_overlap(span_a, span_b):
                return True
    return False


def _district_boundary_payload(
    repo: Optional[DataEngine], district: Optional[District]
):
    if district is None:
        return (None, None)
    try:
        from shapely.geometry import mapping as shapely_mapping  # type: ignore
    except Exception:
        shapely_mapping = None

    boundary = getattr(district, "boundary", None) or getattr(district, "polygon", None)
    if boundary is None and repo is not None:
        try:
            repo.ensure_boundary(district)
        except Exception:
            boundary = None
        else:
            boundary = getattr(district, "boundary", None) or getattr(
                district, "polygon", None
            )
    if boundary is None:
        return (None, None)

    geojson = None
    bounds = None
    if shapely_mapping:
        try:
            geojson = shapely_mapping(boundary)
        except Exception:
            geojson = None
    try:
        if hasattr(boundary, "bounds"):
            minx, miny, maxx, maxy = boundary.bounds
            sw = (float(miny), float(minx))
            ne = (float(maxy), float(maxx))
            if _valid_latlon(*sw) and _valid_latlon(*ne):
                bounds = [sw, ne]
    except Exception:
        bounds = None
    return (geojson, bounds)


def _collect_private_campuses(
    repo: DataEngine,
    base_campuses: list[Campus],
) -> list[dict[str, object]]:
    if not base_campuses:
        return []

    district_ids = set()
    districts = []
    for campus in base_campuses:
        district = getattr(campus, "district", None)
        district_id = getattr(district, "id", None)
        if district is None or district_id in district_ids or not is_traditional_district(district):
            continue
        district_ids.add(district_id)
        districts.append(district)

    results: dict[object, dict[str, object]] = {}

    for district in districts:
        try:
            candidates = list(repo.private_campuses_within(district))
        except Exception:
            candidates = []
        for private_campus in candidates:
            if not getattr(private_campus, "is_private", False):
                continue
            overlaps: list[dict[str, str]] = []
            for base in base_campuses:
                base_district = getattr(base, "district", None)
                if getattr(base_district, "id", None) != getattr(district, "id", None):
                    continue
                if not _grades_overlap(private_campus, base):
                    continue
                overlaps.append(
                    {
                        "campusNumber": _display_campus_number(
                            getattr(base, "campus_number", "")
                        ),
                        "name": _clean_text(getattr(base, "name", "")),
                        "profileUrl": _campus_profile_url(
                            getattr(base, "campus_number", "")
                        ),
                    }
                )
            if not overlaps:
                continue

            key = getattr(private_campus, "id", None) or normalize_campus_lookup_key(
                getattr(private_campus, "campus_number", None)
            )
            if key in results:
                existing = results[key].get("overlapsWith") or []
                results[key]["overlapsWith"] = existing + overlaps
                continue

            entry = _serialize_map_campus(private_campus)
            entry["overlapsWith"] = overlaps
            results[key] = entry

    return sorted(
        [
            {
                **entry,
                "overlapsWith": sorted(
                    entry.get("overlapsWith", []),
                    key=lambda item: (
                        item.get("name") or "",
                        item.get("campusNumber") or "",
                    ),
                ),
            }
            for entry in results.values()
        ],
        key=lambda item: (item.get("name", ""), item.get("campusNumber", "")),
    )


def _build_isd_profile_map_payload(repo: DataEngine, campus: Campus) -> dict:
    origin_entry = _serialize_map_campus(campus)
    origin_entry["isOrigin"] = True

    lat, lon = origin_entry.get("lat"), origin_entry.get("lon")
    points = [(lat, lon)] if lat is not None and lon is not None else []

    charter_destinations = []
    district_destinations = []
    isd_origins = []
    max_transfer = 0

    for to_campus, count, masked in repo.transfers_out(campus):
        if to_campus is None:
            continue
        entry = _serialize_transfer_destination(to_campus, count, masked,
            measurement=transfer_measurement(repo, campus, to_campus, count, masked))
        dest_lat, dest_lon = entry.get("lat"), entry.get("lon")
        if dest_lat is not None and dest_lon is not None:
            points.append((dest_lat, dest_lon))
        count_value = entry.get("count")
        if count_value and count_value > max_transfer:
            max_transfer = count_value
        if entry.get("charter"):
            charter_destinations.append(entry)
        else:
            district_destinations.append(entry)

    for from_campus, count, masked in repo.transfers_in(campus):
        if from_campus is None or getattr(from_campus, "is_charter", False):
            continue
        entry = _serialize_transfer_destination(from_campus, count, masked,
            measurement=transfer_measurement(repo, from_campus, campus, count, masked))
        entry["charter"] = False
        entry["isDistrict"] = True
        origin_lat, origin_lon = entry.get("lat"), entry.get("lon")
        if origin_lat is not None and origin_lon is not None:
            points.append((origin_lat, origin_lon))
        isd_origins.append(entry)

    charter_destinations.sort(key=lambda item: (item.get("name", "")))
    district_destinations.sort(key=lambda item: (item.get("name", "")))
    isd_origins.sort(
        key=lambda item: (item.get("districtName") or "", item.get("name", ""))
    )

    district = getattr(campus, "district", None)
    boundary_geojson, boundary_bounds = _district_boundary_payload(repo, district)

    private_overlays = _collect_private_campuses(repo, [campus])
    for item in private_overlays:
        priv_lat, priv_lon = item.get("lat"), item.get("lon")
        if priv_lat is not None and priv_lon is not None:
            points.append((priv_lat, priv_lon))

    bounds = boundary_bounds or _bounds_from_points(points)

    return {
        "origin": origin_entry,
        "charterDestinations": charter_destinations,
        "districtDestinations": district_destinations,
        "isdOrigins": isd_origins,
        "privateCampuses": private_overlays,
        "maxTransferCount": max_transfer,
        "districtBoundary": boundary_geojson,
        "bounds": bounds,
    }


def _build_charter_profile_map_payload(repo: DataEngine, campus: Campus) -> dict:
    charter_entry = _serialize_map_campus(campus)
    charter_entry["isCharter"] = True

    lat, lon = charter_entry.get("lat"), charter_entry.get("lon")
    points = [(lat, lon)] if lat is not None and lon is not None else []

    isd_origins = []
    max_transfer = 0
    origin_campuses = []

    for from_campus, count, masked in repo.transfers_in(campus):
        if from_campus is None:
            continue
        origin_campuses.append(from_campus)
        entry = _serialize_transfer_destination(from_campus, count, masked,
            measurement=transfer_measurement(repo, from_campus, campus, count, masked))
        entry["charter"] = False
        entry["isDistrict"] = True
        origin_lat, origin_lon = entry.get("lat"), entry.get("lon")
        if origin_lat is not None and origin_lon is not None:
            points.append((origin_lat, origin_lon))
        count_value = entry.get("count")
        if count_value and count_value > max_transfer:
            max_transfer = count_value
        isd_origins.append(entry)

    isd_origins.sort(
        key=lambda item: (
            item.get("districtName") or "",
            item.get("name", ""),
        )
    )

    private_overlays = _collect_private_campuses(repo, origin_campuses)
    for item in private_overlays:
        priv_lat, priv_lon = item.get("lat"), item.get("lon")
        if priv_lat is not None and priv_lon is not None:
            points.append((priv_lat, priv_lon))

    bounds = _bounds_from_points(points)

    return {
        "charter": charter_entry,
        "isdOrigins": isd_origins,
        "privateCampuses": private_overlays,
        "maxTransferCount": max_transfer,
        "districtBoundary": None,
        "bounds": bounds,
    }


def _build_private_profile_map_payload(repo: DataEngine, campus: Campus) -> dict:
    private_entry = _serialize_map_campus(campus)
    private_entry["isPrivateCampus"] = True

    containing = [district for district in repo._districts_covering(campus) if is_traditional_district(district)]
    metadata = repo.catalog.metadata("districts")
    relations = [{"district_number": district.district_number, "name": district.name,
                  "boundary_dataset": "districts", "boundary_version": metadata.version,
                  "relation": "point_covered_by", "boundary_predicate": "covers"}
                 for district in containing]
    boundary_geojson, boundary_bounds = None, None
    features, boundary_points = [], []
    for district, relation in zip(containing, relations, strict=True):
        geometry, bounds = _district_boundary_payload(repo, district)
        if geometry:
            if geometry.get("type") == "FeatureCollection":
                features.extend({**feature, "properties": {**feature.get("properties", {}), **relation}}
                                for feature in geometry.get("features", []))
            elif geometry.get("type") == "Feature":
                features.append({**geometry, "properties": {**geometry.get("properties", {}), **relation}})
            else:
                features.append({"type": "Feature", "properties": relation, "geometry": geometry})
        if bounds:
            boundary_points.extend(bounds)
    if features:
        boundary_geojson = {"type": "FeatureCollection", "features": features}
    boundary_bounds = _bounds_from_points(boundary_points)

    nearby_public = []
    points = []

    priv_lat, priv_lon = private_entry.get("lat"), private_entry.get("lon")
    if priv_lat is not None and priv_lon is not None:
        points.append((priv_lat, priv_lon))

    district_campuses = {candidate.id: candidate for district in containing
                         for candidate in repo.campuses_in(district)}.values()

    for candidate in district_campuses:
        if getattr(candidate, "is_charter", False) or getattr(
            candidate, "is_private", False
        ):
            continue
        if not _grades_overlap(candidate, campus):
            continue
        entry = _serialize_map_campus(candidate)
        entry["isDistrict"] = True
        entry_lat, entry_lon = entry.get("lat"), entry.get("lon")
        if entry_lat is not None and entry_lon is not None:
            points.append((entry_lat, entry_lon))
        nearby_public.append(entry)

    nearby_public.sort(
        key=lambda item: (
            item.get("districtName") or "",
            item.get("name", ""),
        )
    )

    bounds = boundary_bounds or _bounds_from_points(points)
    focus_bounds = _bounds_from_radius(priv_lat, priv_lon, 3.0)

    return {
        "containingDistricts": relations,
        "comparisonAvailable": bool(containing),
        "private": private_entry,
        "publicCampuses": nearby_public,
        "privateCampuses": [],
        "districtBoundary": boundary_geojson,
        "bounds": bounds,
        "focusBounds": focus_bounds,
    }


def build_campus_profile_map_payload(repo: DataEngine, campus: Campus) -> dict:
    if campus is None:
        return {}
    if getattr(campus, "is_private", False):
        return _build_private_profile_map_payload(repo, campus)
    if getattr(campus, "is_charter", False):
        return _build_charter_profile_map_payload(repo, campus)
    return _build_isd_profile_map_payload(repo, campus)


def _split_map_payload(payload: dict) -> tuple[dict, dict]:
    base = dict(payload)
    transfers = {"transfersOut": {}, "transfersIn": {}}

    base["aisdCampuses"] = []
    for campus in payload.get("aisdCampuses") or []:
        campus_copy = dict(campus)
        transfers_out = campus_copy.pop("transfersOut", None)
        if transfers_out is not None:
            key = normalize_campus_lookup_key(campus_copy.get("campusNumber"))
            if key:
                transfers["transfersOut"][key] = transfers_out
        base["aisdCampuses"].append(campus_copy)

    base["charterCampuses"] = []
    for campus in payload.get("charterCampuses") or []:
        campus_copy = dict(campus)
        transfers_in = campus_copy.pop("transfersIn", None)
        if transfers_in is not None:
            key = normalize_campus_lookup_key(campus_copy.get("campusNumber"))
            if key:
                transfers["transfersIn"][key] = transfers_in
        base["charterCampuses"].append(campus_copy)

    return base, transfers


def _encode_payload(payload: dict) -> bytes:
    raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return gzip.compress(raw, mtime=0)


def _save_map_store(
    repo: DataEngine, districts_fp: str, campuses_fp: str
) -> Optional[Path]:
    if not BUILD_MAP_STORE:
        return None
    print(f"[maps] district classification {_district_classification_counts(repo)}")
    path = _map_store_path(districts_fp, campuses_fp)
    tmp_path = path.with_suffix(".sqlite.tmp")
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        pass

    try:
        conn = sqlite3.connect(tmp_path)
        write_sqlite_catalog(conn, repo.catalog)
        conn.execute(
            "CREATE TABLE closures_map (district_number TEXT PRIMARY KEY, payload_base BLOB, payload_transfers BLOB)"
        )
        conn.execute(
            "CREATE TABLE campus_profile (lookup_key TEXT PRIMARY KEY, payload BLOB)"
        )
        rows = []
        for district in repo._districts.values():
            district_number = canonical_district_number(
                getattr(district, "district_number", None)
            )
            if not district_number:
                continue
            try:
                payload = build_closures_map_payload(repo, district)
            except Exception as exc:
                raise ContractError("Required map payload failed") from exc
            base_payload, transfers_payload = _split_map_payload(payload)
            rows.append(
                (
                    district_number,
                    sqlite3.Binary(_encode_payload(base_payload)),
                    sqlite3.Binary(_encode_payload(transfers_payload)),
                )
            )
            if len(rows) >= 50:
                conn.executemany(
                    "INSERT OR REPLACE INTO closures_map (district_number, payload_base, payload_transfers) VALUES (?, ?, ?)",
                    rows,
                )
                conn.commit()
                rows = []
        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO closures_map (district_number, payload_base, payload_transfers) VALUES (?, ?, ?)",
                rows,
            )
        conn.commit()

        rows = []
        for campus in repo._campuses.values():
            campus_number = campus.lookup_key
            if not campus_number:
                raise ContractError("Map source campus lacks its canonical identity")
            try:
                payload = build_campus_profile_map_payload(repo, campus)
            except Exception as exc:
                raise ContractError("Required map payload failed") from exc
            rows.append(
                (
                    campus_number,
                    sqlite3.Binary(_encode_payload(payload)),
                )
            )
            if len(rows) >= 100:
                conn.executemany(
                    "INSERT OR REPLACE INTO campus_profile (lookup_key, payload) VALUES (?, ?)",
                    rows,
                )
                conn.commit()
                rows = []
        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO campus_profile (lookup_key, payload) VALUES (?, ?)",
                rows,
            )
        conn.commit()
        conn.close()
        tmp_path.replace(path)
        return path
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise


def _save_entity_store(
    repo: DataEngine, districts_fp: str, campuses_fp: str
) -> Optional[Path]:
    if not BUILD_ENTITY_STORE:
        return None
    path = _entity_store_path(districts_fp, campuses_fp)
    tmp_path = path.with_suffix(".sqlite.tmp")
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        pass
    try:
        engine = _sql_create_engine(f"sqlite:///{tmp_path}")
        _sql_ensure_schema(engine)
        Session = _sql_create_sessionmaker(engine, expire_on_commit=False)
        with Session.begin() as session:
            _sql_export_dataengine(
                repo,
                session,
                replace=True,
                include_meta_entries=False,
                include_geometry=False,
            )
        engine.dispose()
        tmp_path.replace(path)
        return path
    except Exception:
        try:
            engine.dispose()
        except Exception:
            pass
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise


def _save_repo_snapshot(repo: DataEngine, districts_fp: str, campuses_fp: str) -> Path:
    path = _snapshot_path(districts_fp, campuses_fp).with_suffix(".pkl.gz")
    _prepare_repo_for_pickle(repo)
    with (
        path.open("wb") as stream,
        gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as compressed,
    ):
        compressed.write(SNAPSHOT_MAGIC + repo.catalog.build_id.encode("ascii") + b"\n")
        pickle.dump(repo, compressed, protocol=pickle.HIGHEST_PROTOCOL)
    return path


# ------------------ Sanity probe: fast vs slow charter-within ------------------
def sanity_probe_charters(
    repo: DataEngine, district_name: str, *, n_print: int = 5
) -> None:
    """Compare fast path (repo.charter_campuses_within) vs a direct slow scan for a district.
    Prints counts and a small diff if there is a mismatch.
    """
    # Resolve district
    try:
        dist_q = repo >> ("district", district_name)
        dist = dist_q.first() if hasattr(dist_q, "first") else None
    except Exception:
        dist = None
    if dist is None:
        # fallback simple search
        up = district_name.upper()
        dist = next(
            (d for d in repo._districts.values() if (d.name or "").upper() == up), None
        )
    if dist is None:
        print(f"[sanity] district not found: {district_name}")
        return

    poly = getattr(dist, "polygon", None) or getattr(dist, "boundary", None)
    if poly is None:
        print(f"[sanity] district has no polygon: {district_name}")
        return

    # Fast path
    fast: List[Campus] = repo.charter_campuses_within(dist)

    # Slow exact scan
    slow: List[Campus] = []
    for c in repo._campuses.values():
        if not getattr(c, "is_charter", False):
            continue
        p = getattr(c, "point", None) or getattr(c, "location", None)
        if p is None:
            continue
        try:
            inside = poly.covers(p)
        except Exception:
            inside = False
        if inside:
            slow.append(c)

    status = "OK" if len(fast) == len(slow) else "MISMATCH"
    print(f"[sanity] {district_name}: fast={len(fast)} slow={len(slow)} {status}")

    if status == "MISMATCH":
        fast_ids = {c.id for c in fast}
        slow_ids = {c.id for c in slow}
        only_fast = [c for c in fast if c.id not in slow_ids][:n_print]
        only_slow = [c for c in slow if c.id not in fast_ids][:n_print]
        if only_fast:
            print("  only_fast:", [c.name for c in only_fast])
        if only_slow:
            print("  only_slow:", [c.name for c in only_slow])


def _add_reporting_entities(repo: DataEngine, frame: pd.DataFrame) -> dict[str, int]:
    """Preserve reporting entities even when a later spatial snapshot omits them.

    Identity, grades, classification, and accountability population come from the
    declared report. No location or active/closed status is inferred. Enrollment
    is populated separately from the TAPR enrollment field after enrichment.
    """
    districts = {canonical_district_number(d.district_number): d for d in repo._districts.values()}
    campuses = {canonical_campus_number(c.campus_number): c for c in repo._campuses.values()}
    seen = set()
    added = {"districts": 0, "campuses": 0}
    for row in frame.to_dict(orient="records"):
        try:
            district_number = canonical_source_identifier(row.get("district_number"), width=6)
            campus_number = canonical_source_identifier(row.get("campus_number"), width=9)
        except ValueError as error:
            raise ContractError("Reporting roster has an invalid source identifier") from error
        if district_number is None:
            raise ContractError("Reporting roster has an invalid district identifier")
        if campus_number is None and (
            any(not _record_value_is_missing(row.get(field)) for field in ("campus_name", "grade_range", "charter_type"))
            or (not _record_value_is_missing(row.get("school_type")) and str(row["school_type"]).strip().lower() != "district")
        ):
            raise ContractError("Reporting campus row is missing its campus identifier")
        identity = (district_number, campus_number)
        if identity in seen:
            raise ContractError("Reporting roster has duplicate entity identifiers")
        seen.add(identity)
        if campus_number and campus_number.lstrip("'")[:6] != district_number.lstrip("'"):
            raise ContractError("Reporting campus identifier does not belong to its district")
        district = districts.get(district_number)
        if district is None:
            name = row.get("district_name")
            if not isinstance(name, str) or not name.strip():
                raise ContractError("Reporting district has no name")
            district = District(id=uuid.uuid5(uuid.NAMESPACE_URL, "teadata:district:" + district_number),
                                district_number=district_number, name=name, enrollment=None,
                                meta={"identity_source": "accountability"})
            repo.add_district(district)
            districts[district_number] = district
            added["districts"] += 1
        if campus_number is None:
            district.meta["accountability_student_count"] = row.get("accountability_student_count")
            continue
        campus = campuses.get(campus_number)
        if campus is None:
            name = row.get("campus_name")
            charter = str(row.get("charter_status", "")).strip().lower()
            if not isinstance(name, str) or not name.strip() or charter not in {"yes", "no"}:
                raise ContractError("Reporting campus lacks a name or explicit charter classification")
            campus = Campus(id=uuid.uuid5(uuid.NAMESPACE_URL, "teadata:campus:" + campus_number),
                district_id=district.id, district_number=district_number, campus_number=campus_number,
                name=name, enrollment=None, is_charter=charter == "yes",
                charter_type=str(row.get("charter_type") or ""),
                grade_range=row.get("grade_range"), school_type=row.get("school_type"),
                aea=_coerce_aea_bool(row.get("aea")), aea_raw=_normalize_aea_raw(row.get("aea")),
                location=None, meta={"identity_source": "accountability", "location_status": "unavailable"})
            repo.add_campus(campus)
            campuses[campus_number] = campus
            added["campuses"] += 1
        campus.meta["accountability_student_count"] = row.get("accountability_student_count")
    if not seen:
        raise ContractError("Reporting roster is empty")
    return added


def _add_tefa_entities(repo: DataEngine, frame: pd.DataFrame, *, version: int | None) -> int:
    """Retain the selected legacy source's exact private listing identities.

    Directory district enrichment is a raw claim, never public governance.
    This does not reconcile old IDs with a different roster or infer locations.
    """
    required = {"ID", "Name", "Vendor Type"}
    if not required.issubset(frame.columns):
        raise ContractError("Private TEFA source lacks required identity/type columns")
    selected = frame[frame["Vendor Type"].fillna("").astype(str).str.strip().str.lower() == "schools"]
    records = selected.to_dict(orient="records")
    seen: set[str] = set()
    for record in records:
        source_id = record["ID"]
        try:
            key = campus_lookup_key(source_namespace="tefa", source_id=source_id, is_private=True)
        except (ValueError, TypeError) as error:
            raise ContractError("Private TEFA listing has an invalid opaque source ID") from error
        if key in seen:
            raise ContractError("Private TEFA source has duplicate listing IDs")
        seen.add(key)
        if not isinstance(record["Name"], str) or not record["Name"].strip():
            raise ContractError("Private TEFA listing has no source name")
    for record in records:
        source_id = record["ID"]
        key = campus_lookup_key(source_namespace="tefa", source_id=source_id, is_private=True)
        meta = {
            "private_source_dataset": "tefa", "private_source_version": str(version) if version is not None else None,
            "reported_directory_district_number": _get_record_value(record, ["School District Number"]),
            "reported_directory_district_name": _get_record_value(record, ["School District Name20", "School District Name"]),
            "directory_affiliation_origin": "unverified_legacy_enrichment",
        }
        for output, candidates in (
            ("school_full_address", ["Address Formatted", "Address City State Zip"]),
            ("school_website", ["Contact Website"]),
            ("display_grade_range", ["Display Grade Range"]),
        ):
            value = _get_record_value(record, candidates)
            if value is not None:
                meta[output] = value
        campus = Campus(
            id=uuid.uuid5(uuid.NAMESPACE_URL, "teadata:private:" + key),
            district_id=None, district_number=None, campus_number=None,
            source_namespace="tefa", source_id=source_id,
            name=record["Name"], charter_type="", is_charter=False, is_private=True,
            enrollment=None,
            grade_range=_tefa_grade_range(record.get("Min Grade"), record.get("Max Grade"), record.get("Display Grade Range")),
            school_type=_tefa_school_type(record.get("Min Grade"), record.get("Max Grade"),
                record.get("Is Pre K", record.get("Is Pre-K")), record.get("Is Elementary"),
                record.get("Is Middle"), record.get("Is High")),
            location=_private_coords(record.get("Location Lng"), record.get("Location Lat")),
            meta=meta,
        )
        repo.add_campus(campus)
    return len(records)


def _release_validation(repo: DataEngine) -> dict[str, Any]:
    """Gate structural completeness while keeping source review explicitly pending."""
    if any(repo._xfers_missing.values()) or getattr(repo, "_xfers_unresolved", ()):
        raise ContractError("Release has unresolved transfer endpoints; review the reporting roster before publication")
    return {
        "source_coverage": "passed",
        "referential_integrity": "passed",
        "artifact_binding": "passed",
        "quality_status": "needs_review",
        "source_review_status": "pending",
        "district_count": len(repo._districts),
        "campus_count": len(repo._campuses),
        "reporting_only_campus_count": sum(c.meta.get("identity_source") == "accountability" for c in repo._campuses.values()),
        "public_campuses_without_location": sum(not c.is_private and c.location is None for c in repo._campuses.values()),
        "unmatched_transfer_endpoints": dict(repo._xfers_missing),
        **_district_classification_counts(repo),
    }


def _district_classification_counts(repo: DataEngine) -> dict[str, int]:
    return {
        "traditional_boundary_count": sum(is_traditional_district(d) and d.polygon is not None for d in repo._districts.values()),
        "district_missing_subtype_count": sum(not d.meta.get("organization_sub_type") for d in repo._districts.values()),
    }


def _build_repo(districts_fp: str, campuses_fp: str) -> DataEngine:
    repo = DataEngine()

    cfg_obj = load_config(CFG)
    _, gdf_districts = cfg_obj.load_df("districts", YEAR, section="spatial")
    _, gdf_campuses = cfg_obj.load_df("campuses", YEAR, section="spatial")

    # (Optional) vectorized normalize for districts (small but tidy)
    gdf_districts["district_number_norm"] = gdf_districts["DISTRICT_C"].apply(
        canonical_district_number
    )

    dn_to_id: dict[str, uuid.UUID] = {}

    with repo.bulk():
        # Districts
        for row in gdf_districts.itertuples(index=False):
            district_number = getattr(row, "district_number_norm", None)
            if not district_number:
                district_number = canonical_district_number(getattr(row, "DISTRICT_C"))
            if not district_number:
                district_number = ""
            rating_val = getattr(row, "RATING", "")
            rating_str = str(rating_val) if rating_val is not None else ""
            district_aea_raw = _normalize_aea_raw(getattr(row, "AEA", None))
            d = District(
                id=uuid.uuid5(
                    uuid.NAMESPACE_URL, "teadata:district:" + district_number
                ),
                name=getattr(row, "NAME", None),
                enrollment=None,
                rating=rating_str,
                aea=_coerce_aea_bool(district_aea_raw),
                aea_raw=district_aea_raw,
                boundary=getattr(row, "geometry"),
            )
            # Attach normalized ID as extra attribute
            d.district_number = district_number
            repo.add_district(d)
            for key in _district_lookup_keys(district_number):
                if key:
                    dn_to_id[key] = d.id

        # Inject statewide charter networks (and others with no geometry) from config, if provided
        try:
            charter_networks = add_charter_networks_from_config(repo, CFG, YEAR)
            if charter_networks:
                # refresh the mapping used for campus linking
                refreshed: dict[str, uuid.UUID] = {}
                for d in repo._districts.values():
                    for key in _district_lookup_keys(
                        getattr(d, "district_number", None)
                    ):
                        if key:
                            refreshed[key] = d.id
                dn_to_id = refreshed
                print(f"[charters] added {charter_networks} statewide districts")
        except Exception as e:
            raise ContractError(f"[charters] add failed: {e}") from e

        # Inject statewide districts from reference dataset (no geometry)
        try:
            cfg_obj = load_config(CFG)
            ref_year, ref_fp = cfg_obj.resolve(
                "all_districts_reference", YEAR, section="data_sources"
            )
            _, ref_df = cfg_obj.load_df("all_districts_reference", YEAR)
            added_ref = 0
            for record in ref_df.to_dict(orient="records"):
                raw_dn = _get_record_value(record, _STATEWIDE_DISTRICT_NUMBER_ALIASES)
                district_number = canonical_district_number(raw_dn)
                if not district_number:
                    continue

                lookup_keys = _district_lookup_keys(district_number)

                classification = {
                    "organization_sub_type": record.get("Organization Sub Type"),
                    "district_type_source": "all_districts_reference",
                    "district_type_source_version": str(ref_year),
                }
                existing_id = next((dn_to_id[key] for key in lookup_keys if key in dn_to_id), None)
                if existing_id is not None:
                    repo._districts[existing_id].meta.update(classification)
                    continue

                name = (
                    record.get("Organization Name")
                    or record.get("Organization  Name")
                    or record.get("Organization")
                    or (str(raw_dn).strip() if raw_dn else None)
                    or "Unnamed District"
                )

                district = District(
                    id=uuid.uuid5(
                        uuid.NAMESPACE_URL, "teadata:district:" + district_number
                    ),
                    name=name,
                    enrollment=None,
                    boundary=None,
                    meta=classification,
                )
                district.district_number = district_number
                repo.add_district(district)
                for key in lookup_keys:
                    if key:
                        dn_to_id[key] = district.id
                added_ref += 1

            if added_ref:
                print(
                    f"[districts_ref] added {added_ref} statewide districts from reference {ref_year}"
                )
        except Exception as e:
            raise ContractError(f"[districts_ref] add failed: {e}") from e

        # Inject statewide districts from second reference dataset (no geometry)
        try:
            cfg_obj = load_config(CFG)
            ref2_year, ref2_fp = cfg_obj.resolve(
                "all_districts_reference2", YEAR, section="data_sources"
            )
            _, ref2_df = cfg_obj.load_df("all_districts_reference2", YEAR)
            added_ref2 = 0
            for record in ref2_df.to_dict(orient="records"):
                raw_dn = record.get("DISTRICT")
                district_number = canonical_district_number(raw_dn)
                if not district_number:
                    continue

                lookup_keys = _district_lookup_keys(district_number)
                if any(key in dn_to_id for key in lookup_keys if key):
                    continue

                name = (
                    record.get("DISTNAME")
                    or (str(raw_dn).strip() if raw_dn else None)
                    or "Unnamed District"
                )

                district = District(
                    id=uuid.uuid5(
                        uuid.NAMESPACE_URL, "teadata:district:" + district_number
                    ),
                    name=name,
                    enrollment=None,
                    boundary=None,
                )
                district.district_number = district_number
                repo.add_district(district)
                for key in lookup_keys:
                    if key:
                        dn_to_id[key] = district.id
                added_ref2 += 1

            if added_ref2:
                print(
                    f"[districts_ref2] added {added_ref2} statewide districts from reference {ref2_year}"
                )
        except Exception as e:
            raise ContractError(f"[districts_ref2] add failed: {e}") from e

        # Campuses
        for row in gdf_campuses.itertuples(index=False):
            raw_district = getattr(row, "USER_District_Number")
            district_key = canonical_district_number(raw_district)
            lookup_keys = _district_lookup_keys(district_key)
            if raw_district:
                raw_str = str(raw_district).strip()
                if raw_str:
                    raw_aliases = _district_lookup_keys(raw_str) or [raw_str]
                    for key in raw_aliases:
                        if key and key not in lookup_keys:
                            lookup_keys.append(key)

            district_id = next(
                (dn_to_id.get(k) for k in lookup_keys if k in dn_to_id), None
            )

            campus_aea_raw = _normalize_aea_raw(getattr(row, "USER_AEA", None))
            c = Campus(
                id=uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    "teadata:campus:" + str(getattr(row, "USER_School_Number", None)),
                ),
                district_id=district_id,
                name=getattr(row, "USER_School_Name", "Unnamed Campus"),
                enrollment=None,
                district_number=district_key if district_id is not None else None,
                campus_number=canonical_campus_number(
                    getattr(row, "USER_School_Number", None)
                ),
                aea=_coerce_aea_bool(campus_aea_raw),
                aea_raw=campus_aea_raw,
                grade_range=getattr(row, "USER_Grade_Range", None),
                school_type=getattr(row, "School_Type", None),
                school_status_date=parse_date(
                    getattr(row, "USER_School_Status_Date", None)
                ),
                update_date=parse_date(getattr(row, "USER_Update_Date", None)),
                charter_type=getattr(row, "USER_Charter_Type", ""),
                is_charter=(
                    getattr(row, "USER_Charter_Type", "")
                    in ["OPEN ENROLLMENT CHARTER", "COLLEGE/UNIVERSITY CHARTER"]
                ),
                is_magnet=getattr(row, "USER_Magnet_Status", None),
                location=getattr(row, "geometry"),
                meta={"reported_district_number": raw_district, "identity_source": "campus_geometry"},
            )
            repo.add_campus(c)

        _, reporting = cfg_obj.load_df("accountability", YEAR, reader_profile="population")
        _add_reporting_entities(repo, reporting)
        dn_to_id = {
            key: district.id for district in repo._districts.values()
            for key in _district_lookup_keys(district.district_number) if key
        }

        # Optional private school campuses
        private_loaded = 0
        private_year = None
        private_dataset_name = None
        private_fp = None
        try:
            cfg_obj = load_config(CFG)
            private_dataset_name = cfg_obj.options["private_dataset"]
            if private_dataset_name and private_dataset_name != "tefa":
                raise ContractError("Selected private source lacks a supported opaque identity contract")
            if private_dataset_name:
                private_year, private_fp = cfg_obj.resolve(
                    private_dataset_name, YEAR, section="data_sources"
                )
        except ContractError:
            raise
        except Exception as exc:
            raise ContractError("Private school source is not configured") from exc

        if private_fp:
            try:
                _, df_private = cfg_obj.load_df(private_dataset_name, YEAR)
            except Exception as exc:
                raise ContractError("Private school source failed to load") from exc
            else:
                private_loaded = _add_tefa_entities(repo, df_private, version=private_year)

                print(
                    f"[private] loaded {private_loaded} private campuses from {private_year} (dataset={private_dataset_name})"
                )

    # Run all enrichments once data is loaded
    run_enrichments(repo)

    return repo


def load_repo(districts_fp: str, campuses_fp: str) -> DataEngine:
    if os.environ.get("PYTHONHASHSEED") != "0" or sys.flags.hash_randomization != 0:
        raise ContractError(
            "Reproducible builds require PYTHONHASHSEED=0 and actual hash_randomization=0; the module CLI sets both automatically"
        )
    cfg = load_config(CFG)
    for dataset, path in (("districts", districts_fp), ("campuses", campuses_fp)):
        if (
            Path(cfg.resolve_dataset(dataset, YEAR, section="spatial").path).resolve()
            != Path(path).resolve()
        ):
            raise ContractError("Build paths must match the selected source contracts")
    output = (
        Path(
            os.environ.get(
                "TEADATA_OUTPUT_DIR", str(Path(__file__).resolve().parent / ".cache")
            )
        )
        .expanduser()
        .resolve()
    )
    output.mkdir(exist_ok=True)
    store = ProvenanceStore(
        os.environ.get("TEADATA_PROVENANCE_DIR", str(Path.cwd() / "provenance"))
    )
    with (
        tempfile.TemporaryDirectory(prefix=".build-", dir=output) as temporary,
        BuildRecorder(cfg, store) as build,
    ):
        token = _BUILD_DIRECTORY.set(Path(temporary))
        try:
            repo = _build_repo(districts_fp, campuses_fp)
            expected = (
                (set(cfg.data_sources) - {"tepsac", "tefa"})
                | {cfg.options["private_dataset"]}
                | set(cfg.spatial)
            )
            if set(build.datasets) != expected:
                raise ContractError(
                    f"Build source coverage mismatch: {sorted(expected - set(build.datasets))}"
                )
            repo.catalog = build.catalog()
            for scope, entities in (("district", repo._districts.values()), ("campus", repo._campuses.values())):
                for entity in entities:
                    if scope == "campus" and entity.is_private:
                        continue
                    entity.enrollment = repo.catalog.measurement(entity, "enrollment", scope=scope).numeric
            _materialize_percent_enrollment_change(repo)
            for campus in repo._campuses.values():
                campus.meta.update(repo.catalog.charter_transfer_summary(campus))
            validation = _release_validation(repo)
            boundaries = _save_boundary_store(repo, districts_fp, campuses_fp)
            maps = _save_map_store(repo, districts_fp, campuses_fp)
            entities = _save_entity_store(repo, districts_fp, campuses_fp)
            if any(path is None for path in (boundaries, maps, entities)):
                raise ContractError(
                    "All sidecars are required for a complete data release"
                )
            _strip_repo_boundaries(repo)
            snapshot = _save_repo_snapshot(repo, districts_fp, campuses_fp)
            artifacts = {
                "snapshot": snapshot,
                "boundaries": boundaries,
                "maps": maps,
                "entities": entities,
            }
            # Derive from the exact normalized entities bytes that serve source
            # reads, rather than the engine's richer in-memory Python metadata.
            from teadata.analytical_build import build_analytical_artifacts

            analytical, analytical_report = build_analytical_artifacts(
                entities, repo.catalog, Path(temporary)
            )
            artifacts.update(analytical)
            validation["analytical"] = {key: analytical_report[key] for key in ("source", "scopes")}
            inputs = [
                receipt
                for value in build.datasets.values()
                for receipt in value["metadata"]["provenance_ids"]
            ]
            processing_id = store.record_processing(
                dataset="teadata_release",
                inputs=inputs,
                outputs=artifacts,
                processor="teadata.load_data",
                parameters={"build_id": repo.catalog.build_id},
                code_version=repo.catalog.to_dict()["processor_sha256"],
                validation=validation,
                diagnostics={"analytical": analytical_report["runtime"]},
            )
            publish_release(
                output,
                repo.catalog,
                artifacts,
                processing={
                    "record_id": processing_id,
                    "record": store.read_record(processing_id),
                },
            )
            repo.attach_boundary_store(
                output / f"boundaries_{repo.catalog.build_id}.sqlite"
            )
            return repo
        finally:
            _BUILD_DIRECTORY.reset(token)


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0" or sys.flags.hash_randomization != 0:
        environment = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
        os.execve(
            sys.executable,
            [sys.executable, "-s", "-P", "-m", "teadata.load_data"],
            {**environment, "PYTHONHASHSEED": "0"},
        )
    cfg = load_config(CFG)
    repo = load_repo(
        cfg.resolve_dataset("districts", YEAR, section="spatial").path,
        cfg.resolve_dataset("campuses", YEAR, section="spatial").path,
    )
    print(
        f"Loaded {len(repo._districts)} districts and {len(repo._campuses)} campuses; build {repo.catalog.build_id}"
    )
