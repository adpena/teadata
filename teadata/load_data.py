import uuid
import math
import pickle
from pathlib import Path
import gzip
import json
import geopandas as gpd
import pandas as pd
import hashlib
import inspect
import os
import shutil
import sqlite3

from datetime import datetime, date
from typing import Any, Iterable, Mapping, Optional, List

from teadata import classes as _classes_mod
from teadata import teadata_config as _cfg_mod
from teadata.classes import (
    Campus,
    DataEngine,
    District,
    coerce_grade_spans,
    haversine_miles,
)
from teadata.engine import _load_snapshot_payload
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

CFG = str(Path(__file__).resolve().with_name("teadata_sources.yaml"))
YEAR = 2025

# Optional env toggles
DISABLE_CACHE = os.getenv("TEADATA_DISABLE_CACHE", "0") not in (
    "0",
    "false",
    "False",
    None,
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


def _first_existing_dataset(cfg, candidates: list[str]) -> str | None:
    for name in candidates:
        try:
            cfg.resolve(name, YEAR, section="data_sources")
            return name
        except Exception:
            continue
    return None


def _latest_year_for_dataset(cfg, dataset: str) -> int:
    ymap = cfg.data_sources.get(dataset)
    if not ymap:
        raise KeyError(f"Dataset not found: {dataset}")
    years = ymap.available_years()
    if not years:
        resolved = ymap.resolve(YEAR, strict=False)
        if resolved:
            resolved_year, _ = resolved
            return YEAR if resolved_year == 9999 else resolved_year
        raise KeyError(f"No years available for dataset: {dataset}")
    return max(years)


def _private_clean_number(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    text = str(val).strip()
    if not text:
        return None
    try:
        return int(float(text.replace(",", "")))
    except Exception:
        return None


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


def _private_tefa_id(val: Any) -> Optional[str]:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    text = str(val).strip()
    return text or None


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


def _existing_campus_number_digits(repo: DataEngine) -> set[str]:
    used: set[str] = set()
    for campus in repo._campuses.values():
        canonical = canonical_campus_number(getattr(campus, "campus_number", None))
        if not canonical or len(canonical) <= 1:
            continue
        digits = canonical[1:]
        if not digits.isdigit():
            continue
        used.add(digits.zfill(9))
    return used


def _build_tefa_private_campus_number_map(
    ids: Iterable[Any],
    used_digits: set[str],
) -> dict[str, str]:
    """Build deterministic, collision-free private campus numbers from TEFA IDs."""

    unique_ids: set[str] = set()
    for raw_id in ids:
        key = _private_tefa_id(raw_id)
        if key:
            unique_ids.add(key)

    mapping: dict[str, str] = {}
    modulus = 1_000_000_000
    for key in sorted(unique_ids):
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        seed = int(digest[:16], 16) % modulus
        candidate = seed

        while True:
            digits = f"{candidate:09d}"
            if digits not in used_digits:
                used_digits.add(digits)
                mapping[key] = f"'{digits}"
                break
            candidate = (candidate + 1) % modulus
            if candidate == seed:
                raise RuntimeError("Unable to allocate unique private campus number")

    return mapping


def _tepsac_grade_span(low: Any, high: Any) -> Optional[str]:
    try:
        low_missing = pd.isna(low)
    except Exception:
        low_missing = False
    try:
        high_missing = pd.isna(high)
    except Exception:
        high_missing = False

    low_s = "" if low is None or low_missing else str(low).strip()
    high_s = "" if high is None or high_missing else str(high).strip()
    if not low_s and not high_s:
        return None
    if not low_s:
        return high_s or None
    if not high_s or low_s == high_s:
        return low_s or None
    return f"{low_s}-{high_s}"


def _tepsac_coords(lon_raw: Any, lat_raw: Any) -> Optional[tuple[float, float]]:
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

    ratio_change: Optional[float] = None

    compute_change = getattr(campus, "enrollment_percent_change_from_2015", None)
    if callable(compute_change):
        try:
            derived = compute_change()
        except Exception:
            derived = None
        if derived not in (None, ""):
            ratio_change = _coerce_numeric(derived)

    if ratio_change is None:
        meta = getattr(campus, "meta", {}) or {}

        enrollment_value = _coerce_numeric(getattr(campus, "enrollment", None))
        if enrollment_value is None:
            for key in (
                "campus_2025_student_enrollment_all_students_count",
                "enrollment",
                "student_enrollment",
            ):
                enrollment_value = _coerce_numeric(meta.get(key))
                if enrollment_value is not None:
                    break

        baseline_value = None
        for key in (
            "campus_2015_student_enrollment_all_students_count",
            "campus_2014_student_enrollment_all_students_count",
        ):
            baseline_value = _coerce_numeric(meta.get(key))
            if baseline_value is not None:
                break
        if baseline_value is None:
            baseline_value = _coerce_numeric(
                getattr(campus, "campus_2015_student_enrollment_all_students_count", None)
            )

        if enrollment_value is None or baseline_value in (None, 0):
            return None
        ratio_change = (enrollment_value - baseline_value) / baseline_value

    if ratio_change is None:
        return None

    percent_change = ratio_change * 100.0
    if not math.isfinite(percent_change):
        return None
    return round(percent_change, 1)


def _materialize_percent_enrollment_change(repo: DataEngine) -> int:
    materialized = 0
    for campus in getattr(repo, "_campuses", {}).values():
        if campus is None:
            continue
        meta = getattr(campus, "meta", None)
        if not isinstance(meta, dict):
            try:
                meta = dict(meta or {})
            except Exception:
                meta = {}
            campus.meta = meta

        percent_change = _compute_percent_enrollment_change(campus)
        if percent_change is None:
            meta["percent_enrollment_change"] = "N/A"
            continue
        meta["percent_enrollment_change"] = percent_change
        materialized += 1
    return materialized


def run_enrichments(repo: DataEngine) -> None:
    """Run both district and campus enrichments with sensible fallbacks.
    - Districts from 'accountability' (sheet: 2011-2025 Summary)
    - Campuses from 'campus_accountability' if present else 'accountability'
    """
    cfg_obj = load_config(CFG)

    # District enrichment
    try:
        acc_select = ["overall_rating_2025"]
        acc_year, updated = enrich_districts_from_config(
            repo,
            CFG,
            "accountability",
            YEAR,
            select=acc_select,
            rename={"2025 Overall Rating": "overall_rating_2025"},
            aliases={"overall_rating_2025": "rating"},
            reader_kwargs={"sheet_name": "2011-2025 Summary"},
        )
        print(f"Enriched {updated} districts from accountability {acc_year}")
    except Exception as e:
        print(f"[enrich] accountability (districts) failed: {e}")

    # District enrichment
    try:
        acc_select = ["number_of_students"]
        acc_year, updated = enrich_districts_from_config(
            repo,
            CFG,
            "accountability",
            YEAR,
            select=acc_select,
            rename={"Number of Students": "number_of_students"},
            aliases={
                     "number_of_students": "enrollment"
                     },
            reader_kwargs={"sheet_name": "2025 State Summary"},
        )
        print(f"Enriched {updated} districts enrollment from accountability {acc_year}")
    except Exception as e:
        print(f"[enrich] accountability - enrollment (districts) failed: {e}")

    try:
        tapr_year = _latest_year_for_dataset(cfg_obj, "district_tapr_student_staff_profile")
        yr_tapr, n_tapr = enrich_districts_from_config(
            repo,
            CFG,
            "district_tapr_student_staff_profile",
            tapr_year,
            select=None,
            rename=None,
            aliases=None,
            reader_kwargs=None,
        )
        print(f"Enriched {n_tapr} districts from TAPR student/staff profile {yr_tapr}")
    except Exception as e:
        print(f"[enrich] district_tapr_student_staff_profile failed: {e}")

    # Campus enrichment
    try:
        ds_name = (
            _first_existing_dataset(cfg_obj, ["campus_accountability", "accountability"])
            or "accountability"
        )
        cam_select = ["overall_rating_2025"]
        cam_year, cam_updated = enrich_campuses_from_config(
            repo,
            CFG,
            ds_name,
            YEAR,
            select=cam_select,
            rename={"2025 Overall Rating": "overall_rating_2025"},
            aliases={"overall_rating_2025": "rating"},
            reader_kwargs={"sheet_name": "2011-2025 Summary"},
        )
        print(f"Enriched {cam_updated} campuses from {cam_year} (dataset={ds_name})")
    except Exception as e:
        print(f"[enrich] campus failed: {e}")

    try:
        dist_year, xfers_fp = cfg_obj.resolve(
            "campus_transfer_reports", YEAR, section="data_sources"
        )

        df = pd.read_csv(
            xfers_fp,
            dtype={
                "REPORT_CAMPUS": "string",
                "CAMPUS_RES_OR_ATTEND": "string",
                "TRANSFERS_IN_OR_OUT": "string",
                "REPORT_TYPE": "string",
            },
        )
        # optional: trim and coerce the count to numeric while keeping masks
        # (apply_transfers_from_dataframe handles -999 masking too)
        updated = repo.apply_transfers_from_dataframe(
            df,
            src_col="REPORT_CAMPUS",
            dst_col="CAMPUS_RES_OR_ATTEND",
            count_col="TRANSFERS_IN_OR_OUT",
            type_col="REPORT_TYPE",
            want_type="Transfers Out To",
        )
        print(f"[enrich:transfers] built outgoing edges for {updated} campuses")
    except Exception as e:
        print(f"[enrich:transfers] failed: {e}")

    try:
        yr_peims, n_peims = enrich_campuses_from_config(
            repo,
            CFG,
            "campus_peims_financials",
            YEAR,
            select=DEFAULT_PEIMS_FINANCIAL_COLUMNS,
            rename=None,
            reader_kwargs=None,
        )
        print(f"Enriched {n_peims} campuses from PEIMS financials {yr_peims}")
    except Exception as e:
        print(f"[enrich] campus_peims_financials failed: {e}")

    try:
        tapr_year = _latest_year_for_dataset(cfg_obj, "campus_tapr_student_staff_profile")
        yr_tapr, n_tapr = enrich_campuses_from_config(
            repo,
            CFG,
            "campus_tapr_student_staff_profile",
            tapr_year,
            select=None,
            rename=None,
            reader_kwargs=None,
        )
        print(f"Enriched {n_tapr} campuses from TAPR student/staff profile {yr_tapr}")
    except Exception as e:
        print(f"[enrich] campus_tapr_student_staff_profile failed: {e}")

    try:
        hist_year = _latest_year_for_dataset(cfg_obj, "campus_tapr_historical_enrollment")
        yr_hist, n_hist = enrich_campuses_from_config(
            repo,
            CFG,
            "campus_tapr_historical_enrollment",
            hist_year,
            select=None,
            rename=None,
            reader_kwargs=None,
        )
        print(f"Enriched {n_hist} campuses from TAPR historical enrollment {yr_hist}")
    except Exception as e:
        print(f"[enrich] campus_tapr_historical_enrollment failed: {e}")

    try:
        yr_closure, n_closure = enrich_campuses_from_config(
            repo,
            CFG,
            "campus_planned_closures",
            YEAR,
            select=None,
            rename=None,
            reader_kwargs=None,
        )
        print(f"Enriched {n_closure} campuses from planned closures {yr_closure}")
    except Exception as e:
        print(f"[enrich] campus_planned_closures failed: {e}")

    try:
        materialized = _materialize_percent_enrollment_change(repo)
        print(
            "[enrich:derived] materialized percent_enrollment_change for "
            f"{materialized} campuses"
        )
    except Exception as e:
        print(f"[enrich:derived] percent_enrollment_change materialization failed: {e}")


# ------------------ Repo snapshot cache (warm start) ------------------
def _file_mtime(p: str | Path) -> float:
    try:
        return Path(p).stat().st_mtime
    except FileNotFoundError:
        return 0.0


# --- robust cache signature helpers ---
def _file_size(p: str | Path) -> int:
    try:
        return Path(p).stat().st_size
    except FileNotFoundError:
        return 0


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _path_signature(p: str | Path) -> str:
    """Return a stable signature for a *local* file path (mtime+size).
    For missing files, returns 'missing'.
    """
    try:
        path = Path(p)
        st = path.stat()
        return f"{int(st.st_mtime)}:{st.st_size}"
    except FileNotFoundError:
        return "missing"


def _source_signature(module) -> str:
    """Return a signature for a loaded module's source file (mtime+size path)."""
    try:
        f = inspect.getsourcefile(module) or inspect.getfile(module)
        if not f:
            return "nosrc"
        return f"{f}|{_path_signature(f)}"
    except Exception:
        return "nosrc"


def _safe_path_or_url_signature(path_or_url: str) -> str:
    """If it's a local file, use mtime+size. If it's a URL or non-local,
    include the literal string so signature changes when config path changes."""
    s = str(path_or_url)
    if "://" in s:
        return f"url:{s}"
    return f"file:{Path(s)}|{_path_signature(s)}"


def _repo_cache_dir() -> Path:
    d = Path(".cache")
    d.mkdir(exist_ok=True)
    return d


def _package_cache_dir() -> Path | None:
    try:
        d = Path(__file__).resolve().parent / ".cache"
        d.mkdir(exist_ok=True)
        return d
    except Exception:
        return None


def _copy_cache_artifact(src: Path) -> None:
    pkg_dir = _package_cache_dir()
    if pkg_dir is None:
        return
    try:
        shutil.copy2(src, pkg_dir / src.name)
    except Exception:
        pass


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


# Helper to compute robust extra signature for cache invalidation (config content hash, code, data source signatures)
def _compute_extra_signature() -> dict:
    sig: dict[str, str] = {}
    # Config file signature (content hash + mtime/size)
    try:
        cfg_txt = Path(CFG).read_text(encoding="utf-8")
        sig["cfg_text_sha256"] = _sha256_text(cfg_txt)
        sig["cfg_path_sig"] = _path_signature(CFG)
    except Exception:
        sig["cfg_text_sha256"] = "err"
        sig["cfg_path_sig"] = _path_signature(CFG)

    # Code signatures: this file, classes.py, teadata_config.py
    try:
        sig["code_load_data"] = _source_signature(__import__(__name__))
    except Exception:
        sig["code_load_data"] = "nosrc"
    try:
        sig["code_classes"] = _source_signature(_classes_mod)
    except Exception:
        sig["code_classes"] = "nosrc"
    try:
        sig["code_teadata_config"] = _source_signature(_cfg_mod)
    except Exception:
        sig["code_teadata_config"] = "nosrc"

    # Resolved data sources that affect enrichment
    try:
        cfg = load_config(CFG)
        for ds in (
            "accountability",
            "campus_accountability",
            "charter_reference",
            "campus_peims_financials",
            "campus_tapr_student_staff_profile",
            "campus_tapr_historical_enrollment",
            "district_tapr_student_staff_profile",
            "campus_planned_closures",
            "campus_transfer_reports",
            "tefa",
            "tepsac",
        ):
            try:
                _, p = cfg.resolve(ds, YEAR, section="data_sources")
                sig[f"ds:{ds}"] = _safe_path_or_url_signature(p)
            except Exception:
                sig[f"ds:{ds}"] = "unresolved"
    except Exception:
        sig["cfg_resolve_error"] = "1"

    return sig


def _prepare_repo_for_pickle(repo: DataEngine) -> None:
    """Drop transient spatial indexes so the pickle is small and portable."""
    for attr in (
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
        try:
            district.boundary = None
        except Exception:
            pass
        try:
            district.polygon = None
        except Exception:
            pass
        if hasattr(district, "_prepared"):
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
        if path.exists():
            path.unlink()
    except Exception:
        pass

    try:
        conn = sqlite3.connect(tmp_path)
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
            except Exception:
                continue
            rows.append((district_number, sqlite3.Binary(wkb)))
        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO boundaries (district_number, wkb) VALUES (?, ?)",
                rows,
            )
        conn.commit()
        conn.close()
        tmp_path.replace(path)
        _copy_cache_artifact(path)
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
        return None


TEXAS_BOUNDS = (
    (25.5, -106.65),
    (36.5, -93.5),
)


def _canonical_campus_number(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.lstrip("'")


def _display_campus_number(value: Any) -> str:
    number = _canonical_campus_number(value)
    return number or ""


def _campus_profile_url(campus_number: Any) -> str:
    slug = _canonical_campus_number(campus_number)
    if not slug:
        return ""
    return f"/campuses/profiles/{slug}/"


def _is_charter_district(district: District) -> bool:
    campuses = getattr(district, "campuses", None) or []
    first = campuses[0] if campuses else None
    return bool(first and getattr(first, "is_charter", False))


def _resolve_district_type(district: District, district_type: Optional[str] = None) -> str:
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
        _canonical_campus_number(number)
        for number in campus_numbers
        if _canonical_campus_number(number)
    }

    def _store_campus(campus: Campus) -> None:
        campus_number = getattr(campus, "campus_number", None)
        canonical = _canonical_campus_number(campus_number)
        if not canonical:
            return
        if canonical in campus_index:
            return
        campus_index[canonical] = campus
        campus_index[f"'{canonical}"] = campus

    campus_iter = list(getattr(district, "campuses", None) or [])
    if extra_campuses:
        campus_iter.extend(c for c in extra_campuses if c is not None)
    for campus in campus_iter:
        _store_campus(campus)

    missing_numbers = campus_numbers.difference(campus_index.keys())
    for campus_number in missing_numbers:
        candidates = (campus_number, f"'{campus_number}")
        campus_obj = None
        for candidate in candidates:
            try:
                campus_obj = (repo >> ("campus", candidate)).first()
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


def _haversine_distance(origin_latlon: Optional[tuple], dest_latlon: Optional[tuple]) -> Optional[float]:
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


def _extract_latlon(campus_index: dict, campus_number: Any) -> Optional[tuple[float, float]]:
    if campus_number is None:
        return None
    campus = campus_index.get(_canonical_campus_number(campus_number))
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


def _coerce_masked(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    text = str(value).strip().lower()
    if text in {"true", "yes", "y"}:
        return True
    if text in {"false", "no", "n"}:
        return False
    return False


def _group_transfer_rows(
    working_df: pd.DataFrame,
) -> pd.DataFrame:
    group_columns = ["campus_campus_number", "to_campus_number"]
    aggregate_columns = group_columns + ["count", "masked"]
    aggregate_df = working_df[aggregate_columns]

    if pl is not None:
        try:
            grouped_polars = (
                pl.from_pandas(aggregate_df, include_index=False)
                .group_by(group_columns)
                .agg(
                    pl.col("count").sum().alias("count"),
                    pl.col("masked").max().alias("masked"),
                )
            )
            grouped_pdf = grouped_polars.to_pandas(use_pyarrow_extension_array=True)
            return grouped_pdf[group_columns + ["count", "masked"]]
        except Exception:
            pass

    return (
        aggregate_df.groupby(group_columns, dropna=False)
        .agg({"count": "sum", "masked": "max"})
        .reset_index()
    )


def _collect_charter_transfer_entries(
    repo: DataEngine, district: District, *, only_closures: bool = False
) -> list[dict]:
    charter_entries: list[dict] = []
    origin_candidates: list[Campus] = []
    campus_numbers: set[str] = set()

    try:
        district_campuses = list(repo.campuses_in(district))
    except Exception:
        district_campuses = list(getattr(district, "campuses", None) or [])

    for campus in district_campuses:
        if campus is None or not getattr(campus, "is_charter", False):
            continue
        if only_closures and not getattr(campus, "facing_closure", False):
            continue

        canonical_number = _canonical_campus_number(
            getattr(campus, "campus_number", None)
        )
        inbound_edges = []
        try:
            edge_rows = [
                edge
                for edge in repo.transfers_in(campus)
                if edge and edge[0] is not None
            ]
        except Exception:
            edge_rows = []
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
            origin_number = _canonical_campus_number(
                getattr(origin_campus, "campus_number", None)
            )
            origin_latlon = _extract_latlon(campus_lookup, origin_number)
            distance = _haversine_distance(latlon, origin_latlon)
            count_value = None
            if not edge.get("masked"):
                try:
                    if edge.get("count") is not None:
                        count_value = int(round(float(edge.get("count"))))
                except (TypeError, ValueError):
                    count_value = None
            processed_edges.append(
                {
                    "campus": origin_campus,
                    "canonical_number": origin_number,
                    "latlon": origin_latlon,
                    "distance": distance,
                    "count": count_value,
                    "masked": bool(edge.get("masked")),
                }
            )
        entry["edges"] = processed_edges
        entry["nearest_origin"] = _select_preferred_origin(processed_edges)

    return charter_entries


def _build_charter_map_payload(repo: DataEngine, district: District) -> dict:
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
        private_candidates = list(repo.private_campuses_in(district))
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

    def _get_district_name(campus_obj: Optional[Campus]) -> str:
        if campus_obj is None:
            return ""
        district_obj = getattr(campus_obj, "district", None)
        name = getattr(district_obj, "name", None)
        if not name:
            name = getattr(campus_obj, "district_name", None)
        return _clean_text(name)

    def _get_campus_name(record_name: Any, campus_obj: Optional[Campus], campus_number: Any = None) -> str:
        name = _clean_text(record_name)
        if name:
            if campus_number:
                canonical_record = _canonical_campus_number(name)
                canonical_number = _canonical_campus_number(campus_number)
                if (
                    canonical_record
                    and canonical_number
                    and canonical_record == canonical_number
                ):
                    name = ""
            if name and name.isdigit() and len(name) >= 6:
                name = ""
        if name:
            return name
        if campus_obj is None:
            return ""
        campus_name = getattr(campus_obj, "name", None)
        if not campus_name:
            campus_name = getattr(campus_obj, "campus_name", None)
        if not campus_name:
            campus_name = getattr(campus_obj, "campus_name_long", None)
        return _clean_text(campus_name)

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
        "enrollment": "2024-25 Enrollment",
        "overall_rating_2025": "2025 Overall Rating",
    }
    origin_field_map = {
        "grade_range": "Grade Range",
        "enrollment": "2024-25 Enrollment",
        "overall_rating_2025": "2025 Overall Rating",
        "total_estimated_charter_transfers_out": "2024-25 Total Estimated Charter Transfers Out",
    }

    origin_lookup: dict[str, dict] = {}
    charter_markers: list[dict] = []
    all_latlons: list[tuple[float, float]] = []
    max_transfer = 0

    def _collect_stats(campus_obj: Optional[Campus], field_map: dict[str, str]) -> list[dict]:
        stats = []
        for column, label in field_map.items():
            value = _format_stat_value(_get_campus_stat_value(campus_obj, column))
            if value not in (None, "", "nan"):
                stats.append({"label": label, "value": value})
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
                    "from_name": _get_campus_name(None, origin, origin_number),
                    "count": None if edge.get("masked") else count_value,
                    "masked": bool(edge.get("masked")),
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
                        "name": _get_campus_name(None, origin, origin_number),
                        "profileUrl": _campus_profile_url(origin_number),
                        "districtName": _get_district_name(origin),
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
                        "to_name": _get_campus_name(None, campus, canonical_number),
                        "count": None if edge.get("masked") else count_value,
                        "masked": bool(edge.get("masked")),
                    }
                )

        charter_markers.append(
            {
                "campusNumber": canonical_number,
                "campusNumberDisplay": _display_campus_number(canonical_number),
                "name": _get_campus_name(None, campus, canonical_number),
                "profileUrl": _campus_profile_url(canonical_number),
                "districtName": _get_district_name(campus),
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
        canonical_number = _canonical_campus_number(campus_number)
        latlon = _extract_latlon(campus_lookup, campus_number)
        if not latlon or not _valid_latlon(*latlon):
            continue
        all_latlons.append(latlon)
        lat, lon = latlon
        raw_grade_range = getattr(campus, "grade_range", None) or getattr(
            campus, "gradeRange", None
        )
        enrollment_raw = getattr(campus, "enrollment", None)
        stats = []
        grade_range = _format_stat_value(raw_grade_range)
        if grade_range:
            stats.append({"label": "Grade Range", "value": grade_range})
        enrollment_value = _format_stat_value(enrollment_raw)
        if enrollment_value:
            stats.append({"label": "2024-25 Enrollment", "value": enrollment_value})
        all_private_entries.append(
            {
                "campusNumber": canonical_number,
                "campusNumberDisplay": _display_campus_number(canonical_number),
                "name": _get_campus_name(
                    getattr(campus, "name", None), campus, campus_number
                ),
                "profileUrl": _campus_profile_url(canonical_number),
                "districtName": _get_district_name(campus),
                "lat": lat,
                "lon": lon,
                "gradeRange": grade_range,
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

    try:
        transfer_rows = (
            repo
            >> ("campuses_in", district)
            >> ("where", lambda x: (x.enrollment or 0) > 0)
            >> ("transfers_out", True)
        )
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
                "campus_overall_rating_2025",
                "to_grade_range",
                "to_enrollment",
                "to_overall_rating_2025",
            }
        )
        try:
            transfer_df = transfer_rows.to_df(columns=sorted(required_columns))
        except TypeError:
            transfer_df = transfer_rows.to_df()
    except Exception:
        return payload

    if transfer_df.empty:
        return payload

    origin_field_map = {
        "campus_grade_range": "Grade Range",
        "campus_enrollment": "2024-25 Enrollment",
        "campus_overall_rating_2025": "2025 Overall Rating",
        "campus_num_charter_transfer_destinations": "2024-25 Number of Charter Campuses Receiving Student Transfers - TOTAL",
        "campus_total_estimated_charter_transfers_out": "2024-25 Total Estimated Charter Transfers Out",
    }
    charter_field_map = {
        "to_grade_range": "Grade Range",
        "to_enrollment": "2024-25 Enrollment",
        "to_overall_rating_2025": "2025 Overall Rating",
    }

    available_columns = [col for col in required_columns if col in transfer_df.columns]
    if (
        "campus_campus_number" not in available_columns
        or "to_campus_number" not in available_columns
    ):
        return payload

    working_df = transfer_df[available_columns].copy()

    working_df["campus_campus_number"] = working_df["campus_campus_number"].map(
        _canonical_campus_number
    )
    working_df["to_campus_number"] = working_df["to_campus_number"].map(
        _canonical_campus_number
    )

    working_df = working_df.dropna(subset=["campus_campus_number", "to_campus_number"])

    if working_df.empty:
        return payload

    for column in (
        "campus_num_charter_transfer_destinations",
        "campus_num_charter_transfer_destinations_masked",
        "campus_total_unmasked_charter_transfers_out",
    ):
        if column in working_df.columns:
            working_df[column] = pd.to_numeric(working_df[column], errors="coerce")

    if "campus_total_estimated_charter_transfers_out" not in working_df.columns:
        if "campus_total_unmasked_charter_transfers_out" in working_df.columns:
            unmasked = working_df["campus_total_unmasked_charter_transfers_out"].fillna(
                0
            )
            if (
                "campus_total_estimated_masked_charter_transfers_out"
                in working_df.columns
            ):
                masked_estimates = pd.to_numeric(
                    working_df["campus_total_estimated_masked_charter_transfers_out"],
                    errors="coerce",
                ).fillna(0)
            elif (
                "campus_num_charter_transfer_destinations_masked" in working_df.columns
            ):
                masked_estimates = (
                    working_df["campus_num_charter_transfer_destinations_masked"]
                    .fillna(0)
                    .mul(5)
                )
            else:
                masked_estimates = 0
            working_df["campus_total_estimated_charter_transfers_out"] = (
                unmasked + masked_estimates
            )

    if "count" in working_df.columns:
        working_df["count"] = pd.to_numeric(
            working_df["count"], errors="coerce"
        ).fillna(0)
    else:
        working_df["count"] = 0

    if "masked" in working_df.columns:
        working_df["masked"] = working_df["masked"].map(_coerce_masked)
    else:
        working_df["masked"] = False

    grouped = _group_transfer_rows(working_df)

    campus_numbers = set(working_df["campus_campus_number"].dropna())
    campus_numbers.update(working_df["to_campus_number"].dropna())

    try:
        private_candidates = list(repo.private_campuses_in(district))
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

    def _get_district_name(campus_obj: Optional[Campus]) -> str:
        if campus_obj is None:
            return ""

        district_obj = getattr(campus_obj, "district", None)
        name = getattr(district_obj, "name", None)
        if not name:
            name = getattr(campus_obj, "district_name", None)
        return _clean_text(name)

    def _get_campus_name(record_name: Any, campus_obj: Optional[Campus], campus_number: Any = None) -> str:
        name = _clean_text(record_name)
        if name:
            if campus_number:
                canonical_record = _canonical_campus_number(name)
                canonical_number = _canonical_campus_number(campus_number)
                if (
                    canonical_record
                    and canonical_number
                    and canonical_record == canonical_number
                ):
                    name = ""
            if name and name.isdigit() and len(name) >= 6:
                name = ""
        if name:
            return name

        if campus_obj is None:
            return ""

        campus_name = getattr(campus_obj, "name", None)
        if not campus_name:
            campus_name = getattr(campus_obj, "campus_name", None)
        if not campus_name:
            campus_name = getattr(campus_obj, "campus_name_long", None)
        return _clean_text(campus_name)

    def _collect_stats(row: Any, field_map: dict[str, str]) -> list[dict]:
        stats = []
        for column, label in field_map.items():
            value = None
            if isinstance(row, dict):
                value = _format_stat_value(row.get(column))
            else:
                value = _format_stat_value(getattr(row, column, None))
            if value is not None and value != "nan":
                stats.append({"label": label, "value": value})
        return stats

    transfers_out: dict[str, list[dict]] = {}
    transfers_in: dict[str, list[dict]] = {}

    max_transfer = 0
    for row in grouped.itertuples(index=False):
        origin = row.campus_campus_number
        dest = row.to_campus_number
        masked = bool(row.masked)
        count_value = None if masked else row.count
        if count_value is not None:
            try:
                count_value = int(round(float(count_value)))
            except (TypeError, ValueError):
                count_value = None
        if count_value and count_value > max_transfer:
            max_transfer = count_value

        destination_obj = campus_lookup.get(dest)
        origin_obj = campus_lookup.get(origin)

        transfers_out.setdefault(origin, []).append(
            {
                "to": dest,
                "to_name": _get_campus_name(None, destination_obj, dest),
                "count": count_value,
                "masked": masked,
            }
        )
        transfers_in.setdefault(dest, []).append(
            {
                "from": origin,
                "from_name": _get_campus_name(None, origin_obj, origin),
                "count": count_value,
                "masked": masked,
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
        if campus_obj is None:
            campus_obj = campus_lookup.get(f"'{campus_number}")
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
                "name": _get_campus_name(
                    getattr(record, "campus_name", None), campus_obj, campus_number
                ),
                "profileUrl": _campus_profile_url(campus_number),
                "districtName": _get_district_name(campus_obj),
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
        if campus_obj is None:
            campus_obj = campus_lookup.get(f"'{campus_number}")
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
                "name": _get_campus_name(
                    getattr(record, "to_name", None), campus_obj, campus_number
                ),
                "profileUrl": _campus_profile_url(campus_number),
                "districtName": _get_district_name(campus_obj),
                "lat": lat,
                "lon": lon,
                "gradeRange": _format_stat_value(getattr(record, "to_grade_range", None)),
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
        canonical_number = _canonical_campus_number(campus_number)
        latlon = _extract_latlon(campus_lookup, campus_number)
        if not latlon:
            continue

        all_latlons.append(latlon)
        lat, lon = latlon

        raw_grade_range = getattr(campus, "grade_range", None)
        if raw_grade_range is None:
            raw_grade_range = getattr(campus, "gradeRange", None)
        grade_range = _format_stat_value(raw_grade_range)

        enrollment_raw = getattr(campus, "enrollment", None)
        enrollment_value = _format_stat_value(enrollment_raw)

        stats = []
        if enrollment_value:
            stats.append({"label": "2024-25 Enrollment", "value": enrollment_value})

        payload["privateCampuses"].append(
            {
                "campusNumber": canonical_number,
                "campusNumberDisplay": _display_campus_number(canonical_number),
                "name": _get_campus_name(
                    getattr(campus, "name", None), campus, campus_number
                ),
                "profileUrl": _campus_profile_url(canonical_number),
                "districtName": _get_district_name(campus),
                "lat": lat,
                "lon": lon,
                "gradeRange": grade_range,
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


def _campus_district_name(campus: Optional[Campus]) -> str:
    if campus is None:
        return ""
    district = getattr(campus, "district", None)
    name = getattr(district, "name", None)
    if not name:
        name = getattr(campus, "district_name", None)
    return _clean_text(name)


def _campus_rating_value(campus: Optional[Campus]) -> Optional[str]:
    if campus is None:
        return None
    rating = getattr(campus, "rating", None)
    if rating:
        return str(rating)
    meta = getattr(campus, "meta", {}) or {}
    return meta.get("overall_rating_2025") or meta.get("overall_rating")


def _campus_enrollment_value(campus: Optional[Campus]) -> Optional[int]:
    if campus is None:
        return None
    enrollment = getattr(campus, "enrollment", None)
    if enrollment not in (None, ""):
        try:
            return int(enrollment)
        except Exception:
            try:
                return int(float(enrollment))
            except Exception:
                return None
    meta = getattr(campus, "meta", {}) or {}
    for key in (
        "campus_2025_student_enrollment_all_students_count",
        "enrollment",
        "student_enrollment",
    ):
        value = meta.get(key)
        if value not in (None, ""):
            try:
                return int(value)
            except Exception:
                try:
                    return int(float(value))
                except Exception:
                    continue
    return None


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


def _map_stats_for_campus(campus: Optional[Campus]) -> list[dict[str, str]]:
    stats: list[dict[str, str]] = []
    grade_range = _campus_grade_range(campus)
    if grade_range:
        stats.append({"label": "Grade range", "value": grade_range})
    enrollment_value = _campus_enrollment_value(campus)
    enrollment_text = _format_number(enrollment_value) if enrollment_value else ""
    if enrollment_text:
        stats.append({"label": "2024-25 Enrollment", "value": enrollment_text})
    rating_value = _campus_rating_value(campus)
    if rating_value:
        stats.append({"label": "2025 Overall Rating", "value": _clean_text(rating_value)})
    return stats


def _extract_campus_latlon(campus: Optional[Campus]) -> tuple[Optional[float], Optional[float]]:
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
    canonical_number = _canonical_campus_number(campus_number)
    lat, lon = _extract_campus_latlon(campus)
    return {
        "campusNumber": _display_campus_number(canonical_number),
        "canonicalCampusNumber": canonical_number,
        "name": _clean_text(getattr(campus, "name", "")),
        "districtName": _campus_district_name(campus),
        "lat": lat,
        "lon": lon,
        "profileUrl": _campus_profile_url(canonical_number),
        "charter": bool(getattr(campus, "is_charter", False)),
        "isPrivate": bool(getattr(campus, "is_private", False)),
        "stats": _map_stats_for_campus(campus),
    }


def _serialize_transfer_destination(
    campus: Optional[Campus],
    count: Any,
    masked: Any,
) -> dict[str, object]:
    payload = _serialize_map_campus(campus)
    numeric_count: Optional[int] = None
    if not masked and count is not None:
        try:
            numeric_count = int(count)
        except Exception:
            try:
                numeric_count = int(float(count))
            except Exception:
                numeric_count = None
    payload.update(
        {
            "count": numeric_count,
            "masked": bool(masked),
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


def _district_boundary_payload(repo: Optional[DataEngine], district: Optional[District]):
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
            boundary = getattr(district, "boundary", None) or getattr(district, "polygon", None)
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
        if district is None or district_id in district_ids:
            continue
        district_ids.add(district_id)
        districts.append(district)

    results: dict[object, dict[str, object]] = {}

    for district in districts:
        try:
            candidates = list(repo.private_campuses_in(district))
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

            key = getattr(private_campus, "id", None) or _canonical_campus_number(
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
        entry = _serialize_transfer_destination(to_campus, count, masked)
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
        entry = _serialize_transfer_destination(from_campus, count, masked)
        entry["charter"] = False
        entry["isDistrict"] = True
        origin_lat, origin_lon = entry.get("lat"), entry.get("lon")
        if origin_lat is not None and origin_lon is not None:
            points.append((origin_lat, origin_lon))
        isd_origins.append(entry)

    charter_destinations.sort(key=lambda item: (item.get("name", "")))
    district_destinations.sort(key=lambda item: (item.get("name", "")))
    isd_origins.sort(key=lambda item: (item.get("districtName", ""), item.get("name", "")))

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
        entry = _serialize_transfer_destination(from_campus, count, masked)
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
            item.get("districtName", ""),
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

    district = getattr(campus, "district", None)
    boundary_geojson, boundary_bounds = _district_boundary_payload(repo, district)

    nearby_public = []
    points = []

    priv_lat, priv_lon = private_entry.get("lat"), private_entry.get("lon")
    if priv_lat is not None and priv_lon is not None:
        points.append((priv_lat, priv_lon))

    try:
        district_campuses = list(repo.campuses_in(district)) if district else []
    except Exception:
        district_campuses = []

    for candidate in district_campuses:
        if getattr(candidate, "is_charter", False) or getattr(candidate, "is_private", False):
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
            item.get("districtName", ""),
            item.get("name", ""),
        )
    )

    bounds = boundary_bounds or _bounds_from_points(points)
    focus_bounds = _bounds_from_radius(priv_lat, priv_lon, 3.0)

    return {
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
            key = _canonical_campus_number(campus_copy.get("campusNumber"))
            if key:
                transfers["transfersOut"][key] = transfers_out
        base["aisdCampuses"].append(campus_copy)

    base["charterCampuses"] = []
    for campus in payload.get("charterCampuses") or []:
        campus_copy = dict(campus)
        transfers_in = campus_copy.pop("transfersIn", None)
        if transfers_in is not None:
            key = _canonical_campus_number(campus_copy.get("campusNumber"))
            if key:
                transfers["transfersIn"][key] = transfers_in
        base["charterCampuses"].append(campus_copy)

    return base, transfers


def _encode_payload(payload: dict) -> bytes:
    raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return gzip.compress(raw)


def _save_map_store(
    repo: DataEngine, districts_fp: str, campuses_fp: str
) -> Optional[Path]:
    if not BUILD_MAP_STORE:
        return None
    path = _map_store_path(districts_fp, campuses_fp)
    tmp_path = path.with_suffix(".sqlite.tmp")
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        pass
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass

    try:
        conn = sqlite3.connect(tmp_path)
        conn.execute(
            "CREATE TABLE closures_map (district_number TEXT PRIMARY KEY, payload_base BLOB, payload_transfers BLOB)"
        )
        conn.execute(
            "CREATE TABLE campus_profile (campus_number TEXT PRIMARY KEY, payload BLOB)"
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
            except Exception:
                continue
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
            campus_number = canonical_campus_number(
                getattr(campus, "campus_number", None)
            )
            if not campus_number:
                continue
            try:
                payload = build_campus_profile_map_payload(repo, campus)
            except Exception:
                continue
            rows.append(
                (
                    campus_number,
                    sqlite3.Binary(_encode_payload(payload)),
                )
            )
            if len(rows) >= 100:
                conn.executemany(
                    "INSERT OR REPLACE INTO campus_profile (campus_number, payload) VALUES (?, ?)",
                    rows,
                )
                conn.commit()
                rows = []
        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO campus_profile (campus_number, payload) VALUES (?, ?)",
                rows,
            )
        conn.commit()
        conn.close()
        tmp_path.replace(path)
        _copy_cache_artifact(path)
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
        return None


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
        if path.exists():
            path.unlink()
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
        _copy_cache_artifact(path)
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
        return None


def _load_repo_snapshot(
    districts_fp: str, campuses_fp: str, extra_sig: dict
) -> DataEngine | None:
    snap = _snapshot_path(districts_fp, campuses_fp)
    candidates = [snap, snap.with_suffix(snap.suffix + ".gz")]
    src_mtimes = {
        "districts": _file_mtime(districts_fp),
        "campuses": _file_mtime(campuses_fp),
    }

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            loaded = _load_snapshot_payload(candidate)
            if not (isinstance(loaded, tuple) and len(loaded) >= 2):
                continue
            meta, repo = loaded[0], loaded[1]
            if (
                meta.get("version") == 4
                and meta.get("src_mtimes") == src_mtimes
                and meta.get("extra_sig", {}) == (extra_sig or {})
            ):
                return repo
        except Exception:
            continue
    return None


def _save_repo_snapshot(
    repo: DataEngine, districts_fp: str, campuses_fp: str, extra_sig: dict
) -> None:
    snap = _snapshot_path(districts_fp, campuses_fp)
    snap_gz = snap.with_suffix(snap.suffix + ".gz")
    meta = {
        "version": 4,
        "src_mtimes": {
            "districts": _file_mtime(districts_fp),
            "campuses": _file_mtime(campuses_fp),
        },
        "extra_sig": extra_sig or {},
    }
    try:
        _prepare_repo_for_pickle(repo)
        with snap.open("wb") as f:
            pickle.dump((meta, repo), f, protocol=pickle.HIGHEST_PROTOCOL)
        with gzip.open(snap_gz, "wb") as f:
            pickle.dump((meta, repo), f, protocol=pickle.HIGHEST_PROTOCOL)
        _copy_cache_artifact(snap)
        _copy_cache_artifact(snap_gz)
        # Attempt to also copy snapshots to absolute project cache path; fail quietly if missing
        try:
            abs_cache_dir = Path("/Users/adpena/PycharmProjects/teadata/.cache")
            if abs_cache_dir.is_dir():
                shutil.copy2(snap, abs_cache_dir / snap.name)
                shutil.copy2(snap_gz, abs_cache_dir / snap_gz.name)
        except Exception:
            pass
    except Exception:
        # Cache is best-effort; ignore failures
        pass


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


def load_repo(districts_fp: str, campuses_fp: str) -> DataEngine:
    # Compute extra signature (config + code + resolved accountability) to keep cache honest
    extra_sig = _compute_extra_signature()
    try:
        print("[cache] extra_sig keys:", sorted(extra_sig.keys()))
    except Exception:
        pass

    # Try warm-start snapshot first (including config/accountability freshness)
    snap = (
        None
        if DISABLE_CACHE
        else _load_repo_snapshot(districts_fp, campuses_fp, extra_sig)
    )
    if snap is not None:
        # Re-run enrichments to ensure latest aliases/logic land on the cached repo
        run_enrichments(snap)
        if SPLIT_BOUNDARIES:
            boundary_path = _boundary_store_path(districts_fp, campuses_fp)
            if boundary_path.exists():
                try:
                    snap.attach_boundary_store(boundary_path)
                except Exception:
                    pass
        _save_map_store(snap, districts_fp, campuses_fp)
        _save_entity_store(snap, districts_fp, campuses_fp)
        return snap

    repo = DataEngine()

    gdf_districts = gpd.read_file(districts_fp, engine="pyogrio")
    gdf_campuses = gpd.read_file(campuses_fp, engine="pyogrio")

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
                id=uuid.uuid4(),
                name=getattr(row, "NAME", None),
                enrollment=getattr(row, "ENROLLMENT", 0),
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
            print(f"[charters] add failed: {e}")

        # Inject statewide districts from reference dataset (no geometry)
        try:
            cfg_obj = load_config(CFG)
            ref_year, ref_fp = cfg_obj.resolve(
                "all_districts_reference", YEAR, section="data_sources"
            )
            ref_df = pd.read_excel(ref_fp)
            added_ref = 0
            for record in ref_df.to_dict(orient="records"):
                raw_dn = _get_record_value(record, _STATEWIDE_DISTRICT_NUMBER_ALIASES)
                district_number = canonical_district_number(raw_dn)
                if not district_number:
                    continue

                lookup_keys = _district_lookup_keys(district_number)

                if any(key in dn_to_id for key in lookup_keys if key):
                    continue

                name = (
                    record.get("Organization Name")
                    or record.get("Organization  Name")
                    or record.get("Organization")
                    or (str(raw_dn).strip() if raw_dn else None)
                    or "Unnamed District"
                )

                enrollment_val = record.get("Enrollment as of Oct 2024")
                enrollment = 0
                if isinstance(enrollment_val, (int, float)) and not pd.isna(
                    enrollment_val
                ):
                    try:
                        enrollment = int(float(enrollment_val))
                    except (TypeError, ValueError):
                        enrollment = 0
                elif isinstance(enrollment_val, str):
                    cleaned = enrollment_val.replace(",", "").strip()
                    if cleaned:
                        try:
                            enrollment = int(float(cleaned))
                        except ValueError:
                            enrollment = 0

                district = District(
                    id=uuid.uuid4(),
                    name=name,
                    enrollment=enrollment,
                    boundary=None,
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
            print(f"[districts_ref] add failed: {e}")

        # Inject statewide districts from second reference dataset (no geometry)
        try:
            cfg_obj = load_config(CFG)
            ref2_year, ref2_fp = cfg_obj.resolve(
                "all_districts_reference2", YEAR, section="data_sources"
            )
            ref2_df = pd.read_excel(ref2_fp)
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
                    id=uuid.uuid4(),
                    name=name,
                    enrollment=0,
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
            print(f"[districts_ref2] add failed: {e}")

        # Add default fallback District object
        fallback_district = District(
            id=uuid.uuid4(),
            name="Unknown District",
            enrollment=0,
            rating="",
            boundary=None,
        )
        fallback_district.district_number = (
            canonical_district_number("000000") or "'000000"
        )
        repo.add_district(fallback_district)
        fallback_id = fallback_district.id

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
            if district_id is None:
                district_id = fallback_id

            campus_aea_raw = _normalize_aea_raw(getattr(row, "USER_AEA", None))
            c = Campus(
                id=uuid.uuid4(),
                district_id=district_id,
                name=getattr(row, "USER_School_Name", "Unnamed Campus"),
                enrollment=getattr(row, "USER_School_Enrollment_as_of_Oc", -999999),
                district_number=district_key,
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
            )
            repo.add_campus(c)

        # Optional private school campuses
        private_loaded = 0
        private_year = None
        private_dataset_name = None
        private_fp = None
        try:
            cfg_obj = load_config(CFG)
            private_dataset_name = _first_existing_dataset(cfg_obj, ["tefa", "tepsac"])
            if private_dataset_name:
                private_year, private_fp = cfg_obj.resolve(
                    private_dataset_name, YEAR, section="data_sources"
                )
        except Exception:
            private_fp = None

        if private_fp:
            try:
                if private_dataset_name == "tefa":
                    df_private = pd.read_excel(private_fp)
                else:
                    df_private = pd.read_csv(
                        private_fp,
                        dtype={
                            "district_number": "string",
                            "school_number": "string",
                        },
                    )
            except Exception as exc:
                print(f"[private] failed to load {private_fp}: {exc}")
            else:
                if private_dataset_name == "tefa":
                    if "Vendor Type" in df_private.columns:
                        vendor_series = (
                            df_private["Vendor Type"]
                            .fillna("")
                            .astype(str)
                            .str.strip()
                            .str.lower()
                        )
                        df_private = df_private[vendor_series == "schools"].copy()

                    used_campus_digits = _existing_campus_number_digits(repo)
                    tefa_id_map = _build_tefa_private_campus_number_map(
                        df_private.get("ID", pd.Series(dtype="object")),
                        used_campus_digits,
                    )

                    for record in df_private.to_dict(orient="records"):
                        raw_district = _get_record_value(
                            record,
                            ["School District Number"],
                        )
                        district_key = canonical_district_number(raw_district)
                        lookup_keys = _district_lookup_keys(district_key)
                        if not lookup_keys and raw_district is not None:
                            lookup_keys = _district_lookup_keys(str(raw_district).strip())

                        district_id = next(
                            (dn_to_id.get(k) for k in lookup_keys if k in dn_to_id),
                            None,
                        )
                        if district_id is None:
                            district_id = fallback_id

                        school_name = _get_record_value(record, ["Name"])
                        tefa_id = _private_tefa_id(_get_record_value(record, ["ID"]))
                        campus_number = tefa_id_map.get(tefa_id) if tefa_id else None

                        location = _tepsac_coords(
                            _get_record_value(record, ["Location Lng"]),
                            _get_record_value(record, ["Location Lat"]),
                        )

                        district_name = _get_record_value(
                            record,
                            ["School District Name20", "School District Name"],
                        )

                        meta_payload: dict[str, Any] = {}
                        school_full_address = _get_record_value(
                            record,
                            ["Address Formatted", "Address City State Zip"],
                        )
                        if school_full_address is not None:
                            meta_payload["school_full_address"] = school_full_address

                        school_website = _get_record_value(record, ["Contact Website"])
                        if school_website is not None:
                            meta_payload["school_website"] = school_website

                        if district_name is not None:
                            meta_payload["district_name"] = district_name

                        display_grade_range = _get_record_value(
                            record, ["Display Grade Range"]
                        )
                        if display_grade_range is not None:
                            meta_payload["display_grade_range"] = display_grade_range

                        if tefa_id is not None:
                            meta_payload["tefa_id"] = tefa_id

                        if private_year is not None:
                            meta_payload["private_school_dataset_year"] = private_year
                        if private_dataset_name:
                            meta_payload["private_school_dataset"] = private_dataset_name

                        campus = Campus(
                            id=uuid.uuid4(),
                            district_id=district_id,
                            name=school_name or "Unnamed Private School",
                            charter_type="",
                            is_charter=False,
                            is_private=True,
                            enrollment=None,
                            grade_range=_tefa_grade_range(
                                _get_record_value(record, ["Min Grade"]),
                                _get_record_value(record, ["Max Grade"]),
                                display_grade_range,
                            ),
                            school_type=_tefa_school_type(
                                _get_record_value(record, ["Min Grade"]),
                                _get_record_value(record, ["Max Grade"]),
                                _get_record_value(record, ["Is Pre K", "Is Pre-K"]),
                                _get_record_value(record, ["Is Elementary"]),
                                _get_record_value(record, ["Is Middle"]),
                                _get_record_value(record, ["Is High"]),
                            ),
                            district_number=district_key
                            or (str(raw_district).strip() if raw_district else None),
                            campus_number=campus_number,
                            location=location,
                            meta=meta_payload,
                        )
                        repo.add_campus(campus)
                        private_loaded += 1
                else:
                    for row in df_private.itertuples(index=False):
                        raw_district = getattr(row, "district_number", None)
                        district_key = canonical_district_number(raw_district)
                        lookup_keys: list[str] = []
                        if district_key:
                            lookup_keys.append(district_key)
                            digits = (
                                district_key[1:]
                                if isinstance(district_key, str)
                                and district_key.startswith("'")
                                else district_key
                            )
                            if digits:
                                lookup_keys.append(digits)
                                if isinstance(digits, str) and digits.isdigit():
                                    lookup_keys.append(str(int(digits)))
                        elif raw_district:
                            lookup_keys.append(str(raw_district).strip())

                        district_id = next(
                            (dn_to_id.get(k) for k in lookup_keys if k in dn_to_id),
                            None,
                        )
                        if district_id is None:
                            district_id = fallback_id

                        meta_fields = [
                            "school_full_address",
                            "school_website",
                            "school_accreditations",
                        ]
                        meta_payload = {}
                        for field_name in meta_fields:
                            val = getattr(row, field_name, None)
                            try:
                                if hasattr(pd, "isna") and pd.isna(val):
                                    continue
                            except Exception:
                                pass
                            if val is not None:
                                meta_payload[field_name] = val
                        if private_year is not None:
                            meta_payload["private_school_dataset_year"] = private_year
                        if private_dataset_name:
                            meta_payload["private_school_dataset"] = private_dataset_name

                        campus = Campus(
                            id=uuid.uuid4(),
                            district_id=district_id,
                            name=getattr(row, "school_name", "Unnamed Private School"),
                            charter_type="",
                            is_charter=False,
                            is_private=True,
                            enrollment=_private_clean_number(
                                getattr(row, "enrollment", None)
                            ),
                            grade_range=_tepsac_grade_span(
                                getattr(row, "grade_low", None),
                                getattr(row, "grade_high", None),
                            ),
                            district_number=district_key
                            or (str(raw_district).strip() if raw_district else None),
                            campus_number=canonical_campus_number(
                                getattr(row, "school_number", None)
                            ),
                            location=_tepsac_coords(
                                getattr(row, "best_lng", None),
                                getattr(row, "best_lat", None),
                            ),
                            meta=meta_payload,
                        )
                        repo.add_campus(campus)
                        private_loaded += 1

                print(
                    f"[private] loaded {private_loaded} private campuses from {private_year} (dataset={private_dataset_name})"
                )

    # Run all enrichments once data is loaded
    run_enrichments(repo)

    boundary_path = _save_boundary_store(repo, districts_fp, campuses_fp)
    _save_map_store(repo, districts_fp, campuses_fp)
    _save_entity_store(repo, districts_fp, campuses_fp)
    if boundary_path is not None:
        _strip_repo_boundaries(repo)

    # Save snapshot for next warm start
    _save_repo_snapshot(repo, districts_fp, campuses_fp, extra_sig)
    if boundary_path is not None:
        try:
            repo.attach_boundary_store(boundary_path)
        except Exception:
            pass
    return repo


if __name__ == "__main__":
    # Resolve spatial file paths from the TEA config (prefers exact YEAR, else nearest prior)
    try:
        cfg_obj = load_config(CFG)
        dist_year, districts_fp = cfg_obj.resolve("districts", YEAR, section="spatial")
        camp_year, campuses_fp = cfg_obj.resolve("campuses", YEAR, section="spatial")
        print(f"[cfg] spatial districts -> year {dist_year}: {districts_fp}")
        print(f"[cfg] spatial campuses  -> year {camp_year}: {campuses_fp}")
    except Exception as e:
        # Fallback to explicit paths if config is missing or incomplete
        print(f"[cfg] spatial resolution failed ({e}); falling back to defaults")
        districts_fp = "shapes/Current_Districts_2025.geojson"
        campuses_fp = "shapes/Schools_2024_to_2025.geojson"

    repo = load_repo(districts_fp, campuses_fp)

    repo.profile(True)

    print(f"Loaded {len(repo._districts)} districts and {len(repo._campuses)} campuses")

    # Example usage
    some_district = next(iter(repo._districts.values()))
    print(
        "Example district:",
        some_district.name,
        getattr(some_district, "district_number"),
    )
    campuses = repo.campuses_in(some_district)
    print("Campuses in district:", [c.name for c in campuses])

    # 2) Pick a district (by name or district_number)
    #    (simple examples; replace with your own search logic)
    # dist = next(d for d in repo._districts.values() if d.name.upper() == "ALDINE ISD")
    dist_q = repo >> ("district", "ALDINE ISD")  # returns a chainable Query

    print(dist_q.name)

    dist = dist_q.first()  # unwrap the District object
    _or = getattr(dist, "overall_rating_2025", None)
    if _or is None:
        _or = getattr(getattr(dist, "meta", {}), "get", lambda k, d=None: d)(
            "overall_rating_2025"
        )
    print(_or)
    charters = repo >> ("charters_within", dist)  # returns a Query of campuses

    campuses = repo.campuses_in(dist)
    print("Campuses in district:", [c.name for c in campuses])

    # or, by TEA code:
    # dist = next(d for d in repo._districts.values() if d.district_number == "'011901")

    # 3) Get charter campuses physically inside that district’s boundary
    charters = repo.charter_campuses_within(dist)

    # 4) Work with the results
    print(f"{dist.name}: {len(charters)} charter campuses inside boundary")
    for c in sorted(charters, key=lambda x: x.name):
        print(
            f" - {c.name}, {c.campus_number} (type: {c.charter_type}, enrollment: {c.enrollment}, rating: {c.rating})"
        )

    # Show enriched attributes (if present)
    _enr = getattr(dist, "overall_rating_2025", None)
    if _enr is None:
        _enr = getattr(dist, "meta", {}).get("overall_rating_2025")
    if _enr is not None:
        print(
            "Enriched (accountability):",
            _enr,
            "| canonical rating:",
            getattr(dist, "rating", None),
        )

    # Sanity probe: ensure fast equals slow for Aldine
    sanity_probe_charters(repo, "ALDINE ISD")

    # 1) Single nearest (same as before, now geodesic-aware)
    c = repo.nearest_campus(-95.3698, 29.7604)  # Houston
    print(c.name)

    # 2) Top 5 nearest campuses within 10 miles (any campus)
    five = repo.nearest_campuses(-95.3698, 29.7604, limit=5, max_miles=10)
    [(c.name, c.enrollment) for c in five]

    # 3) Nearest charter campus only
    charter = repo.nearest_campus(-95.3698, 29.7604, charter_only=True)
    print(charter.name)

    # 4) Top 3 nearest charter campuses within 7 miles
    charters_3 = repo.nearest_campuses(
        -95.3698, 29.7604, limit=3, charter_only=True, max_miles=7
    )

    # 5) Using the >> operator
    d1 = repo >> ("nearest", (-95.3698, 29.7604))  # one campus
    d3 = repo >> ("nearest", (-95.3698, 29.7604), 3, 10)  # top 3 within 10 miles
    dc = repo >> (
        "nearest_charter",
        (-95.3698, 29.7604),
        5,
        15,
    )  # top 5 charters within 15 miles

    # 5 nearest charter campuses within 200 miles of arbitrary Aldine ISD campus,
    # rated "" and enrollment > 200, sorted by enrollment desc,
    # then return just names and enrollment.
    dist = repo >> ("district", "ALDINE ISD")
    result = (
        repo
        >> ("nearest_charter", repo.campuses_in(dist)[0].coords, 200, 200)
        >> (
            "filter",
            lambda c: (c.rating or "").startswith("") and (c.enrollment or 0) > 200,
        )
        >> ("sort", lambda c: c.enrollment, True)
        >> ("take", 5)
        >> ("map", lambda c: (c.name, c.enrollment))
    )

    print(result)

    result = (
        repo
        >> ("district", "ALDINE ISD")
        >> ("campuses_in",)
        >> ("nearest_charter", None, 200, 25)  # infer centroid or seed coords
        >> (
            "where",
            lambda c: (c.rating or "").startswith("") and (c.enrollment or 0) > 200,
        )
        >> ("sort", lambda c: c.enrollment, True)
        >> ("take", 5)
        >> ("select", lambda c: (c.name, c.enrollment))
    )
    print(result)

    dist_q = repo >> ("district", "ALDINE ISD")  # Query[District]
    print(dist_q.name)  # ✅ works (delegates to first District)
    print(dist_q.district_number)  # ✅

    val_unsuffixed = getattr(dist_q, "overall_rating_2025", None)
    if val_unsuffixed is None:
        val_unsuffixed = getattr(dist_q, "meta", {}).get("overall_rating_2025")
    val_canonical = getattr(dist_q, "rating", None)
    print(
        "(info) enriched rating:", val_unsuffixed, "| canonical rating:", val_canonical
    )

    dist_obj = dist_q.first() if hasattr(dist_q, "first") else None
    if dist_obj is not None:
        try:
            repo.ensure_boundary(dist_obj)
        except Exception:
            pass
    centroid = (
        dist_obj.polygon.centroid
        if dist_obj is not None and getattr(dist_obj, "polygon", None) is not None
        else None
    )
    first_campus = (dist_q >> ("campuses_in",))[0]  # __getitem__ for indexing
    if dist_q:
        ...  # __bool__ reflects non-empty
    print(first_campus.rating)  # readable __repr__
