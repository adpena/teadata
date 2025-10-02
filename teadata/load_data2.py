import uuid
import pickle
from pathlib import Path
import geopandas as gpd
import pandas as pd
import json
import hashlib
import inspect
import os

from datetime import datetime, date
from typing import Optional

from teadata import classes as _classes_mod
from teadata import teadata_config as _cfg_mod
from teadata.classes import District, Campus, DataEngine, _point_xy
from teadata.teadata_config import load_config, normalize_campus_number_value
from teadata.enrichment.districts import enrich_districts_from_config
from teadata.enrichment.campuses import enrich_campuses_from_config
from teadata.enrichment.charter_networks import add_charter_networks_from_config

CFG = "teadata_sources.yaml"
YEAR = 2025

# Optional env toggles
DISABLE_CACHE = os.getenv("TEADATA_DISABLE_CACHE", "0") not in (
    "0",
    "false",
    "False",
    None,
)


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


def _first_existing_dataset(cfg, candidates: list[str]) -> str | None:
    for name in candidates:
        try:
            cfg.resolve(name, YEAR, section="data_sources")
            return name
        except Exception:
            continue
    return None


def run_enrichments(repo: DataEngine) -> None:
    """Run both district and campus enrichments with sensible fallbacks.
    - Districts from 'accountability' (sheet: 2011-2025 Summary)
    - Campuses from 'campus_accountability' if present else 'accountability'
    """
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

    # Campus enrichment
    try:
        cfg = load_config(CFG)
        ds_name = (
            _first_existing_dataset(cfg, ["campus_accountability", "accountability"])
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
        cfg_obj = load_config(CFG)
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
            select=['instruction_af_perc', 'transportation_af_per_student', 'extracurricular_af_per_student', 'security_monitoring_af_per_student', 'students_w_disabilities_af_per_student', 'bilingual_ed_af_per_student', 'dyslexia_or_related_disorder_serv_af_per_student', 'ccmr_af_per_student', 'guidance_counseling_af_per_student', 'school_leadership_af_per_student'],  # let auto-detection pick the three canonical fields
            rename=None,
            reader_kwargs=None,
        )
        print(f"Enriched {n_peims} campuses from PEIMS financials {yr_peims}")
    except Exception as e:
        print(f"[enrich] campus_peims_financials failed: {e}")


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
    d = Path(".cache")
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
        for ds in (
                "accountability",
                "campus_accountability",
                "charter_reference",
                "campus_peims_financials",
                "campus_transfer_reports",
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
        "_xy_deg",
        "_xy_rad",
        "_campus_list",
        "_point_tree",
        "_point_geoms",
        "_point_ids",
        "_geom_id_to_index",
        "_xy_deg_np",
        "_campus_list_np",
        "_all_xy_np",
        "_all_campuses_np",
    ):
        if hasattr(repo, attr):
            setattr(repo, attr, None)


def _load_repo_snapshot(
    districts_fp: str, campuses_fp: str, extra_sig: dict
) -> DataEngine | None:
    snap = _snapshot_path(districts_fp, campuses_fp)
    if not snap.exists():
        return None
    try:
        with snap.open("rb") as f:
            meta, repo = pickle.load(f)
        src_mtimes = {
            "districts": _file_mtime(districts_fp),
            "campuses": _file_mtime(campuses_fp),
        }
        if (
            meta.get("version") == 4
            and meta.get("src_mtimes") == src_mtimes
            and meta.get("extra_sig", {}) == (extra_sig or {})
        ):
            return repo
    except Exception:
        return None
    return None


def _save_repo_snapshot(
    repo: DataEngine, districts_fp: str, campuses_fp: str, extra_sig: dict
) -> None:
    snap = _snapshot_path(districts_fp, campuses_fp)
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
    except Exception:
        # Cache is best-effort; ignore failures
        pass


# ------------------ Sanity probe: fast vs slow charter-within ------------------
from typing import List


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
        return snap

    repo = DataEngine()

    gdf_districts = gpd.read_file(districts_fp, engine="pyogrio")
    gdf_campuses = gpd.read_file(campuses_fp, engine="pyogrio")

    # (Optional) vectorized normalize for districts (small but tidy)
    gdf_districts["district_number_norm"] = gdf_districts["DISTRICT_C"].apply(
        normalize_district_code
    )

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

        # Inject statewide charter networks (no geometry) from config, if provided
        try:
            charter_networks = add_charter_networks_from_config(repo, CFG, YEAR)
            if charter_networks:
                # refresh the mapping used for campus linking
                dn_to_id = {
                    getattr(d, "district_number", None): d.id
                    for d in repo._districts.values()
                    if getattr(d, "district_number", None)
                }
                print(f"[charters] added {charter_networks} statewide districts")
        except Exception as e:
            print(f"[charters] add failed: {e}")

        # Add default fallback District object
        fallback_district = District(
            id=uuid.uuid4(),
            name="Unknown District",
            enrollment=0,
            rating="",
            boundary=None,
        )
        fallback_district.district_number = "000000"
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
                school_status_date=parse_date(
                    getattr(row, "USER_School_Status_Date", None)
                ),
                update_date=parse_date(getattr(row, "USER_Update_Date", None)),
                charter_type=getattr(row, "USER_Charter_Type", ""),
                is_charter=(
                    getattr(row, "USER_Charter_Type", "") == "OPEN ENROLLMENT CHARTER"
                ),
                location=getattr(row, "geometry"),
            )
            repo.add_campus(c)

    # Run all enrichments once data is loaded
    run_enrichments(repo)

    # Save snapshot for next warm start
    _save_repo_snapshot(repo, districts_fp, campuses_fp, extra_sig)
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

    centroid = dist_q.polygon.centroid  # ✅ methods/attributes chain through
    first_campus = (dist_q >> ("campuses_in",))[0]  # __getitem__ for indexing
    if dist_q:
        ...  # __bool__ reflects non-empty
    print(first_campus.rating)  # readable __repr__
