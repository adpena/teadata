import uuid
import pickle
from pathlib import Path
import geopandas as gpd
import pandas as pd
import json
import hashlib
import inspect
import os
import classes as _classes_mod
import teadata_config as _cfg_mod
from classes import District, Campus, DataEngine, _point_xy

from datetime import datetime, date
from typing import Optional
from teadata_config import (
    _DEFAULT_DISTRICT_ALIASES,
    _DEFAULT_CAMPUS_ALIASES,
    load_config,
    normalize_district_number_column,
    normalize_campus_number_column,
)

CFG = "teadata_sources.yaml"
YEAR = 2025

def parse_date(val: Optional[str]) -> Optional[date]:
    if not val:
        return None

    # Try date-only first
    try:
        return datetime.strptime(val, "%m/%d/%Y").date()
    except ValueError:
        pass

    # Try datetime with AM/PM
    try:
        return datetime.strptime(val, "%m/%d/%Y %I:%M:%S %p").date()
    except ValueError:
        pass

    # If nothing worked
    raise ValueError(f"Unrecognized date format: {val}")


def normalize_district_code(value: str | int | float) -> str:
    """
    Normalize district code into a 6-digit, zero-padded string
    with a leading apostrophe (Excel style).
    """
    s = str(int(float(value))).zfill(6)
    return f"'{s}"


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
    d = Path('.cache')
    d.mkdir(exist_ok=True)
    return d

def _snapshot_path(districts_fp: str, campuses_fp: str) -> Path:
    d = _repo_cache_dir()
    tag = f"repo_{Path(districts_fp).stem}_{Path(campuses_fp).stem}.pkl"
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
        for ds in ("accountability", "campus_accountability"):
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
    for attr in ("_kdtree", "_xy_deg", "_xy_rad", "_campus_list",
                 "_point_tree", "_point_geoms", "_point_ids", "_geom_id_to_index",
                 "_xy_deg_np", "_campus_list_np", "_all_xy_np", "_all_campuses_np"):
        if hasattr(repo, attr):
            setattr(repo, attr, None)

def _load_repo_snapshot(districts_fp: str, campuses_fp: str, extra_sig: dict) -> DataEngine | None:
    snap = _snapshot_path(districts_fp, campuses_fp)
    if not snap.exists():
        return None
    try:
        with snap.open('rb') as f:
            meta, repo = pickle.load(f)
        src_mtimes = {
            'districts': _file_mtime(districts_fp),
            'campuses': _file_mtime(campuses_fp),
        }
        if (
            meta.get('version') == 4 and
            meta.get('src_mtimes') == src_mtimes and
            meta.get('extra_sig', {}) == (extra_sig or {})
        ):
            return repo
    except Exception:
        return None
    return None

def _save_repo_snapshot(repo: DataEngine, districts_fp: str, campuses_fp: str, extra_sig: dict) -> None:
    snap = _snapshot_path(districts_fp, campuses_fp)
    meta = {
        'version': 4,
        'src_mtimes': {
            'districts': _file_mtime(districts_fp),
            'campuses': _file_mtime(campuses_fp),
        },
        'extra_sig': extra_sig or {},
    }
    try:
        _prepare_repo_for_pickle(repo)
        with snap.open('wb') as f:
            pickle.dump((meta, repo), f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        # Cache is best-effort; ignore failures
        pass


# ------------------ Sanity probe: fast vs slow charter-within ------------------
from typing import List

def sanity_probe_charters(repo: DataEngine, district_name: str, *, n_print: int = 5) -> None:
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
        dist = next((d for d in repo._districts.values() if (d.name or "").upper() == up), None)
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


# ------------------ Enrichment (from config) ------------------
from typing import Iterable, Dict, Any, Tuple

def enrich_districts_from_config(
    repo: DataEngine,
    cfg_path: str,
    dataset: str,
    year: int,
    *,
    select: Iterable[str] | None = None,
    suffix: str | None = None,
    transforms: Dict[str, Any] | None = None,
    rename: Dict[str, str] | None = None,
    aliases: Dict[str, str] | None = None,
    reader_kwargs: Dict[str, Any] | None = None,
) -> Tuple[int, int]:
    """Load a dataset via teadata_config, normalize district_number, and copy selected
    columns onto District objects. Returns (resolved_year, updated_count)."""
    cfg = load_config(cfg_path)
    resolved_year, df = cfg.load_df(dataset, year, section="data_sources", **(reader_kwargs or {}))

    # Diagnostics: show first few columns to help spot mismatches
    try:
        cols_preview = list(df.columns)[:12] if hasattr(df, 'columns') else list(df.keys())
        print(f"[enrich:{dataset}] loaded columns (preview): {cols_preview}")
    except Exception:
        pass

    # If the first attempt didn't contain expected rename keys, and it's likely Excel, try all sheets
    expected_sources = set((rename or {}).keys())
    if isinstance(df, dict):
        # Already loaded all sheets (sheet_name=None). Pick the first sheet that contains any expected source column.
        for name, dfi in df.items():
            if not hasattr(dfi, 'columns'):
                continue
            if (expected_sources and expected_sources.intersection(set(dfi.columns))) or (not expected_sources):
                print(f"[enrich:{dataset}] using sheet: {name}")
                df = dfi
                break
        else:
            # fallback to first sheet
            name, df = next(iter(df.items()))
            print(f"[enrich:{dataset}] fallback to first sheet: {name}")
    else:
        # If rename is provided but none of the source columns are present, try reloading all sheets
        if expected_sources and not expected_sources.intersection(set(getattr(df, 'columns', []))):
            try:
                # re-resolve path so we can use pandas directly
                _, path = cfg.resolve(dataset, year, section="data_sources")
                all_sheets = pd.read_excel(path, sheet_name=None)
                print(f"[enrich:{dataset}] reloaded all sheets to search for columns: {sorted(list(all_sheets.keys()))}")
                chosen = None
                for name, dfi in all_sheets.items():
                    if expected_sources.intersection(set(dfi.columns)):
                        print(f"[enrich:{dataset}] using sheet: {name}")
                        chosen = dfi
                        break
                if chosen is None:
                    # fallback: first sheet
                    name, chosen = next(iter(all_sheets.items()))
                    print(f"[enrich:{dataset}] fallback to first sheet: {name}")
                df = chosen
            except Exception as e:
                print(f"[enrich:{dataset}] failed to reload all sheets: {e}")

    # Optional column renaming to make Python-friendly identifiers
    if rename:
        df = df.rename(columns=rename)
        missing = [k for k in rename.keys() if k not in df.columns and rename[k] not in df.columns]
        if missing:
            print(f"[enrich:{dataset}] warning: rename sources not found: {missing}")

    # Normalize to `district_number`
    df_norm, found = normalize_district_number_column(df, aliases=_DEFAULT_DISTRICT_ALIASES, new_col="district_number")
    if found is None:
        raise ValueError(f"No district-number column found in dataset '{dataset}'.")

    # Column selection + optional transforms
    if select is None:
        cols = [c for c in df_norm.columns if c != "district_number"]
    else:
        cols = [c for c in select if c in df_norm.columns]
        missing_sel = [c for c in (select or []) if c not in df_norm.columns]
        if missing_sel:
            print(f"[enrich:{dataset}] warning: selected columns not found after rename: {missing_sel}")
    pre_suffix_cols = list(cols)
    if not cols:
        print(f"[enrich:{dataset}] nothing to enrich: selection empty after checks")
        return resolved_year, 0

    # Normalize selected column values: strip strings, turn NaN/empty to None
    for c in cols:
        if c in df_norm.columns:
            df_norm[c] = df_norm[c].apply(lambda x: (None if (pd.isna(x) or (isinstance(x, str) and x.strip() == "")) else (str(x).strip() if isinstance(x, str) else x)))

    if transforms:
        for c, fn in transforms.items():
            if c in df_norm.columns:
                df_norm[c] = df_norm[c].map(fn)

    # Optional suffix to avoid collisions and mark provenance
    if suffix:
        rename_suffix = {c: f"{c}{suffix}" for c in cols}
        df_norm = df_norm.rename(columns=rename_suffix)
        cols = [rename_suffix[c] for c in cols]

    # Build alias mapping that accounts for suffix
    alias_effective: Dict[str, str] = {}
    if aliases:
        for src, tgt in aliases.items():
            # If the caller referred to the pre-suffix name, map it to the effective column name
            if suffix and src in pre_suffix_cols:
                alias_effective[f"{src}{suffix}"] = tgt
            else:
                alias_effective[src] = tgt

    # Build mapping {district_number -> {col: value}}; support both bare and Excel-style ('XXXXXX) keys
    sub = df_norm[["district_number"] + cols].drop_duplicates("district_number")
    mapping: Dict[str, Dict[str, Any]] = {}
    for rec in sub.itertuples(index=False):
        d = rec._asdict()
        dn = str(d.pop("district_number"))  # normalized: zero-padded 6-digit, no apostrophe
        mapping[dn] = d
        mapping[f"'{dn}"] = d  # also allow leading-apostrophe keys to match repo.district_number

    # Targeted probe for Aldine ISD (CDN 101902) to help debug mismatches
    try:
        probe_keys = ["101902", "'101902"]
        for pk in probe_keys:
            if pk in mapping:
                sample = {k: mapping[pk].get(k) for k in cols}
                print(f"[enrich:{dataset}] probe CDN {pk}: {sample}")
                break
    except Exception:
        pass

    # Apply to repo districts
    updated = 0
    for d in repo._districts.values():
        dn = getattr(d, "district_number", None)
        if not dn:
            continue
        # Try exact, then apostrophe-stripped variant
        attrs = mapping.get(dn)
        if not attrs:
            dn2 = str(dn).lstrip("'`’")
            attrs = mapping.get(dn2)
        if not attrs:
            continue
        for k, v in attrs.items():
            # Determine alias/target attribute even if the source attribute doesn't exist
            target = alias_effective.get(k)
            if not target and aliases:
                base = k
                if suffix and base.endswith(suffix):
                    base = base[: -len(suffix)]
                if base in aliases:
                    target = aliases[base]

            # Best-effort: set the source attribute only if District exposes it
            if hasattr(d, k):
                try:
                    setattr(d, k, v)
                except Exception:
                    pass
            else:
                # Persist unknown/source-only attributes into the District metadata
                try:
                    if not hasattr(d, 'meta') or d.meta is None:
                        d.meta = {}
                    d.meta[k] = v
                except Exception as ex:
                    print(f"[enrich:{dataset}] meta write failed for {dn} {k}: {ex}")

            # Always try to set the alias target (e.g., map overall_rating_2025 -> rating)
            if target:
                vv = v
                if isinstance(vv, str):
                    vv = vv.strip()
                if pd.isna(vv) if hasattr(pd, "isna") else False:
                    vv = None
                if target == "rating" and vv is not None:
                    vv = str(vv)
                try:
                    setattr(d, target, vv)
                except Exception as ex:
                    print(f"[enrich:{dataset}] alias setattr failed for {dn} {k}->{target}: {ex}")
        updated += 1
        # debug a single well-known district
        try:
            if str(dn).lstrip("'`’") == "101902":
                print(
                    f"[enrich:{dataset}] applied to Aldine: dn={dn} rating={getattr(d, 'rating', None)}"
                )
        except Exception:
            pass

    return resolved_year, updated

from typing import Iterable, Dict, Any, Tuple

def enrich_campuses_from_config(
    repo: DataEngine,
    cfg_path: str,
    dataset: str,
    year: int,
    *,
    select: Iterable[str] | None = None,
    suffix: str | None = None,
    transforms: Dict[str, Any] | None = None,
    rename: Dict[str, str] | None = None,
    aliases: Dict[str, str] | None = None,
    reader_kwargs: Dict[str, Any] | None = None,
) -> Tuple[int, int]:
    """
    Load a dataset via teadata_config, normalize campus_number, and copy selected
    columns onto Campus objects. Returns (resolved_year, updated_count).
    """
    cfg = load_config(cfg_path)
    resolved_year, df = cfg.load_df(dataset, year, section="data_sources", **(reader_kwargs or {}))

    # Optional column renaming
    if rename:
        df = df.rename(columns=rename)

    # Normalize to campus_number (9-digit zero-padded, handle apostrophes)
    df_norm, found = normalize_campus_number_column(df, aliases=_DEFAULT_CAMPUS_ALIASES, new_col="campus_number")
    if found is None:
        raise ValueError(f"No campus-number column found in dataset '{dataset}'.")

    # Column selection + optional transforms
    if select is None:
        cols = [c for c in df_norm.columns if c != "campus_number"]
    else:
        cols = [c for c in select if c in df_norm.columns]
        missing_sel = [c for c in (select or []) if c not in df_norm.columns]
        if missing_sel:
            print(f"[enrich:{dataset}] warning: selected columns not found after rename: {missing_sel}")
    pre_suffix_cols = list(cols)
    if not cols:
        return resolved_year, 0

    # Normalize selected values: strip strings, turn NaN/empty to None
    for c in cols:
        if c in df_norm.columns:
            df_norm[c] = df_norm[c].apply(
                lambda x: (None if (pd.isna(x) or (isinstance(x, str) and x.strip() == "")) else (str(x).strip() if isinstance(x, str) else x))
            )

    if transforms:
        for c, fn in transforms.items():
            if c in df_norm.columns:
                df_norm[c] = df_norm[c].map(fn)

    # Optional suffix for provenance or name collisions
    if suffix:
        rename_suffix = {c: f"{c}{suffix}" for c in cols}
        df_norm = df_norm.rename(columns=rename_suffix)
        cols = [rename_suffix[c] for c in cols]

    # Build alias mapping accounting for suffix
    alias_effective: Dict[str, str] = {}
    if aliases:
        for src, tgt in aliases.items():
            if suffix and src in pre_suffix_cols:
                alias_effective[f"{src}{suffix}"] = tgt
            else:
                alias_effective[src] = tgt

    # Build mapping {campus_number -> {col: value}}; support bare and Excel-style keys
    sub = df_norm[["campus_number"] + cols].drop_duplicates("campus_number")
    mapping: Dict[str, Dict[str, Any]] = {}
    for rec in sub.itertuples(index=False):
        d = rec._asdict()
        cn = str(d.pop("campus_number"))  # normalized 9-digit, no apostrophe
        mapping[cn] = d
        mapping[f"'{cn}"] = d  # Excel-style

    # Apply to repo campuses
    updated = 0
    for cobj in repo._campuses.values():
        cn = getattr(cobj, "campus_number", None)
        if not cn:
            continue
        attrs = mapping.get(cn) or mapping.get(str(cn).lstrip("'`’"))
        if not attrs:
            continue
        for k, v in attrs.items():
            # Determine alias/target attribute even if Campus doesn’t expose source k
            target = alias_effective.get(k)

            # Best-effort: set the source attribute only if Campus exposes it
            if hasattr(cobj, k):
                try:
                    setattr(cobj, k, v)
                except Exception:
                    pass
            else:
                # Persist unknown/source-only attributes into the Campus.metadata
                try:
                    if not hasattr(cobj, "meta") or cobj.meta is None:
                        cobj.meta = {}
                    cobj.meta[k] = v
                except Exception as ex:
                    print(f"[enrich:{dataset}] campus meta write failed for {cn} {k}: {ex}")

            # Always try to set alias target, e.g. some_field -> rating
            if target:
                vv = v
                if isinstance(vv, str):
                    vv = vv.strip()
                if pd.isna(vv) if hasattr(pd, "isna") else False:
                    vv = None
                try:
                    setattr(cobj, target, vv)
                except Exception as ex:
                    print(f"[enrich:{dataset}] campus alias setattr failed for {cn} {k}->{target}: {ex}")
        updated += 1

    return resolved_year, updated

def load_repo(districts_fp: str, campuses_fp: str) -> DataEngine:
    # Compute extra signature (config + code + resolved accountability) to keep cache honest
    extra_sig = _compute_extra_signature()
    try:
        print("[cache] extra_sig keys:", sorted(extra_sig.keys()))
    except Exception:
        pass

    # Try warm-start snapshot first (including config/accountability freshness)
    snap = _load_repo_snapshot(districts_fp, campuses_fp, extra_sig)
    if snap is not None:
        # Overlay district enrichment (already present)
        try:
            acc_select = ["overall_rating_2025"]
            enrich_districts_from_config(
                snap, CFG, "accountability", YEAR,
                select=acc_select,
                rename={"2025 Overall Rating": "overall_rating_2025"},
                aliases={"overall_rating_2025": "rating"},
                reader_kwargs={"sheet_name": "2011-2025 Summary"},
            )
        except Exception as e:
            print(f"[enrich] overlay on snapshot (district) failed: {e}")

        # ✅ Overlay campus enrichment (add this block)
        try:
            cam_select = ["overall_rating_2025"]
            cam_year, cam_updated = enrich_campuses_from_config(
                snap, CFG, "campus_accountability", YEAR,
                select=cam_select,
                rename={"2025 Overall Rating": "overall_rating_2025"},
                aliases={"overall_rating_2025": "rating"},
                # reader_kwargs={"sheet_name": "SomeSheet"},  # if Excel sheet needed
            )
            print(f"Enriched {cam_updated} campuses from {cam_year}")
        except Exception as e:
            print(f"[enrich] campus (overlay) failed: {e}")

        return snap

    repo = DataEngine()

    gdf_districts = gpd.read_file(districts_fp, engine="pyogrio")
    gdf_campuses = gpd.read_file(campuses_fp, engine="pyogrio")

    # (Optional) vectorized normalize for districts (small but tidy)
    gdf_districts["district_number_norm"] = gdf_districts["DISTRICT_C"].apply(normalize_district_code)

    dn_to_id: dict[str, uuid.UUID] = {}

    with repo.bulk():
        # Districts
        for row in gdf_districts.itertuples(index=False):
            district_number = getattr(row, "district_number_norm", None)
            if district_number is None:
                district_number = normalize_district_code(getattr(row, "DISTRICT_C"))
            rating_val = getattr(row, "RATING", "")
            rating_str = str(rating_val) if rating_val is not None else ""
            d = District(
                id=uuid.uuid4(),
                name=getattr(row, "NAME", None),
                enrollment=getattr(row, "ENROLLMENT", 0),
                rating=rating_str,
                aea=getattr(row, "AEA", None),
                boundary=getattr(row, "geometry"),
            )
            # Attach normalized ID as extra attribute
            d.district_number = district_number
            repo.add_district(d)
            dn_to_id[district_number] = d.id

        # Add default fallback District object
        fallback_district = District(
            id=uuid.uuid4(),
            name="Unknown District",
            enrollment=0,
            rating="",
            boundary=None,
        )
        fallback_district.district_number = '000000'
        repo.add_district(fallback_district)
        fallback_id = fallback_district.id

        # Campuses
        for row in gdf_campuses.itertuples(index=False):
            district_number = getattr(row, "USER_District_Number")

            district_id = dn_to_id.get(district_number, fallback_id)

            c = Campus(
                id=uuid.uuid4(),
                district_id=district_id,
                name=getattr(row, "USER_School_Name", "Unnamed Campus"),
                enrollment=getattr(row, "USER_School_Enrollment_as_of_Oc", -999999),
                district_number=district_number,
                campus_number=getattr(row, "USER_School_Number", None),

                aea=getattr(row, "USER_AEA", None),
                grade_range=getattr(row, "USER_Grade_Range", None),
                school_type=getattr(row, "School_Type", None),
                school_status_date=parse_date(getattr(row, "USER_School_Status_Date", None)),
                update_date=parse_date(getattr(row, "USER_Update_Date", None)),

                charter_type=getattr(row, "USER_Charter_Type", ""),
                is_charter=(getattr(row, "USER_Charter_Type", "") == "OPEN ENROLLMENT CHARTER"),
                location=getattr(row, "geometry"),
            )
            repo.add_campus(c)

    # --- Enrich districts from config: ACCOUNTABILITY ---
    try:
        acc_select = [
            "overall_rating_2025",  # renamed from "2025 Overall Rating"
        ]
        acc_year, updated = enrich_districts_from_config(
            repo,
            CFG,
            "accountability",
            YEAR,
            select=acc_select,
            # suffix argument removed
            rename={"2025 Overall Rating": "overall_rating_2025"},
            aliases={"overall_rating_2025": "rating"},  # also write to canonical District.rating
            reader_kwargs={"sheet_name": '2011-2025 Summary'},  # first worksheet
            transforms={"overall_rating_2025": lambda s: (None if (pd.isna(s) or (isinstance(s, str) and s.strip()=="")) else str(s).strip())},
        )
        print(f"Enriched {updated} districts from accountability {acc_year}")
        # One-off confirmation for Aldine
        try:
            aldine = next(d for d in repo._districts.values() if str(getattr(d, 'district_number', '')).lstrip("'`’") == '101902')
            enr = getattr(aldine, "overall_rating_2025", None)
            if enr is None:
                enr = getattr(aldine, "meta", {}).get("overall_rating_2025")
            print("[enrich:accountability] final Aldine:", enr, getattr(aldine, "rating", None))
        except Exception:
            pass
    except Exception as e:
        print(f"[enrich] accountability failed: {e}")

    # --- Enrich campuses from config: ACCOUNTABILITY ---
    try:
        cam_select = ["overall_rating_2025"]
        cam_year, cam_updated = enrich_campuses_from_config(
            repo, CFG, "accountability", YEAR,
            select=cam_select,
            rename={"2025 Overall Rating": "overall_rating_2025"},
            aliases={"overall_rating_2025": "rating"},
            reader_kwargs={"sheet_name": "2011-2025 Summary"},
        )
        print(f"Enriched {cam_updated} campuses from {cam_year}")
    except Exception as e:
        print(f"[enrich] campus failed: {e}")
    # Save snapshot for next warm start
    _save_repo_snapshot(repo, districts_fp, campuses_fp, extra_sig)
    return repo


if __name__ == "__main__":
    districts_fp = "shapes/Current_Districts_2025.geojson"
    campuses_fp = "shapes/Schools_2024_to_2025.geojson"

    repo = load_repo(districts_fp, campuses_fp)

    repo.profile(True)

    print(f"Loaded {len(repo._districts)} districts and {len(repo._campuses)} campuses")

    # Example usage
    some_district = next(iter(repo._districts.values()))
    print("Example district:", some_district.name, getattr(some_district, "district_number"))
    campuses = repo.campuses_in(some_district)
    print("Campuses in district:", [c.name for c in campuses])

    # 2) Pick a district (by name or district_number)
    #    (simple examples; replace with your own search logic)
    # dist = next(d for d in repo._districts.values() if d.name.upper() == "ALDINE ISD")
    dist_q = repo >> ("district", "ALDINE ISD")   # returns a chainable Query

    print(dist_q.name)

    dist = dist_q.first()                         # unwrap the District object
    print(dist.overall_rating_2025)
    charters = repo >> ("charters_within", dist)  # returns a Query of campuses

    # or, by TEA code:
    # dist = next(d for d in repo._districts.values() if d.district_number == "'011901")

    # 3) Get charter campuses physically inside that district’s boundary
    charters = repo.charter_campuses_within(dist)

    # 4) Work with the results
    print(f"{dist.name}: {len(charters)} charter campuses inside boundary")
    for c in sorted(charters, key=lambda x: x.name):
        print(f" - {c.name}, {c.campus_number} (type: {c.charter_type}, enrollment: {c.enrollment}, rating: {c.rating})")

    # Show enriched attributes (if present)
    _enr = getattr(dist, "overall_rating_2025", None)
    if _enr is None:
        _enr = getattr(dist, "meta", {}).get("overall_rating_2025")
    if _enr is not None:
        print("Enriched (accountability):", _enr, "| canonical rating:", getattr(dist, "rating", None))

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
    charters_3 = repo.nearest_campuses(-95.3698, 29.7604, limit=3, charter_only=True, max_miles=7)

    # 5) Using the >> operator
    d1 = repo >> ("nearest", (-95.3698, 29.7604))                  # one campus
    d3 = repo >> ("nearest", (-95.3698, 29.7604), 3, 10)           # top 3 within 10 miles
    dc = repo >> ("nearest_charter", (-95.3698, 29.7604), 5, 15)   # top 5 charters within 15 miles

    # 5 nearest charter campuses within 200 miles of arbitrary Aldine ISD campus,
    # rated "" and enrollment > 200, sorted by enrollment desc,
    # then return just names and enrollment.
    dist = repo >> ("district", "ALDINE ISD")
    result = (repo
              >> ("nearest_charter", repo.campuses_in(dist)[0].coords, 200, 200)
              >> ("filter", lambda c: (c.rating or "").startswith("") and (c.enrollment or 0) > 200)
              >> ("sort", lambda c: c.enrollment, True)
              >> ("take", 5)
              >> ("map", lambda c: (c.name, c.enrollment)))

    print(result)

    result = (repo
              >> ("district", "ALDINE ISD")
              >> ("campuses_in", )
              >> ("nearest_charter", None, 200, 25)       # infer centroid or seed coords
              >> ("where", lambda c: (c.rating or "").startswith("") and (c.enrollment or 0) > 200)
              >> ("sort", lambda c: c.enrollment, True)
              >> ("take", 5)
              >> ("select", lambda c: (c.name, c.enrollment)))
    print(result)

    dist_q = repo >> ("district", "ALDINE ISD")  # Query[District]
    print(dist_q.name)  # ✅ works (delegates to first District)
    print(dist_q.district_number)  # ✅

    val_unsuffixed = getattr(dist_q, "overall_rating_2025", None)
    if val_unsuffixed is None:
        val_unsuffixed = getattr(dist_q, "meta", {}).get("overall_rating_2025")
    val_canonical = getattr(dist_q, "rating", None)
    print("(info) enriched rating:", val_unsuffixed, "| canonical rating:", val_canonical)

    centroid = dist_q.polygon.centroid  # ✅ methods/attributes chain through
    first_campus = (dist_q >> ("campuses_in",))[0]  # __getitem__ for indexing
    if dist_q: ...  # __bool__ reflects non-empty
    print(first_campus.rating)  # readable __repr__
