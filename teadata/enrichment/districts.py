import pandas as pd
from typing import Dict, Any, Optional

import teadata.classes as classes_mod

from . import enricher
from .base import Enricher

from teadata.teadata_config import load_config
from teadata.teadata_config import canonical_district_number
from teadata.teadata_config import normalize_district_number_column


def _profile_enabled() -> bool:
    return bool(getattr(classes_mod, "ENABLE_PROFILING", False))


def _debug(msg: str) -> None:
    if _profile_enabled():
        print(msg)


# -----------------------------
# Canonical district-number helper
# -----------------------------


def _canon_district_number(x: Any) -> Optional[str]:
    """
    Canonicalize *any* district-number-like value into the library's canonical
    representation: a six-digit, zero-padded string **with a leading apostrophe**.

    Examples -> "'011901"
    - 11901, "11901", "011901", "'011901", "011901-001" → "'011901"
    - None/empty → None
    """
    try:
        return canonical_district_number(x)
    except Exception:
        return None


# -----------------------------
# Core enrichment routine used by both the @enricher class and legacy wrapper
# -----------------------------


def _apply_district_enrichment_table(
    repo,
    cfg_path: str,
    dataset: str,
    year: int,
    *,
    select=None,
    rename=None,
    aliases=None,
    reader_kwargs=None,
):
    """
    Load <dataset> for <year> using teadata_config, normalize district_number to canonical
    (with leading apostrophe), then enrich District objects.

    Returns: (resolved_year, updated_count)
    """
    cfg = load_config(cfg_path)
    resolved_year, df = cfg.load_df(
        dataset, year, section="data_sources", **(reader_kwargs or {})
    )

    if _profile_enabled():
        _debug(
            f"[enrich:{dataset}] resolved_year={resolved_year} (requested={year}) rows={len(df)}"
        )

    # Optional column rename from spreadsheet-style headers to pythonic names
    if rename:
        df = df.rename(columns=rename)

    # Ensure we have a clean, machine-joinable district_number column
    df, found = normalize_district_number_column(df, new_col="district_number")

    if "district_number" not in df.columns:
        if _profile_enabled():
            _debug(
                f"[enrich:{dataset}] abort: district_number column missing after normalization (found={found})"
            )
        return resolved_year, 0

    # Canonicalize to leading-apostrophe format used across the repo
    df["district_number"] = df["district_number"].map(_canon_district_number)
    df = df[df["district_number"].notna()]

    if _profile_enabled():
        valid_rows = len(df)
        unique_keys = df["district_number"].nunique(dropna=True)
        _debug(
            f"[enrich:{dataset}] canonical district numbers -> valid_rows={valid_rows} unique_keys={unique_keys}"
        )

    if select is None:
        use_cols = [c for c in df.columns if c != "district_number"]
        if not use_cols:
            if _profile_enabled():
                _debug(
                    f"[enrich:{dataset}] abort: no columns available after excluding district_number"
                )
            return resolved_year, 0
    else:
        use_cols = [c for c in select if c != "district_number"]
        missing = [c for c in use_cols if c not in df.columns]
        if missing:
            if _profile_enabled():
                _debug(
                    f"[enrich:{dataset}] missing columns after rename: {', '.join(sorted(missing))}"
                )
            raise KeyError(
                "district enrichment missing expected columns after rename: "
                + ", ".join(sorted(missing))
            )
        if not use_cols:
            raise ValueError(
                "district enrichment requires a non-empty `select` collection of column names"
            )

    # Basic whitespace/NA cleanup on the value columns
    for c in use_cols:
        if c in df.columns:
            df[c] = df[c].apply(
                lambda x: (
                    None
                    if (pd.isna(x) or (isinstance(x, str) and x.strip() == ""))
                    else (x.strip() if isinstance(x, str) else x)
                )
            )

    # Build mapping keyed by canonical district_number → {attr: value}
    sub = df[["district_number"] + use_cols].drop_duplicates("district_number")
    mapping: Dict[str, Dict[str, Any]] = {}
    for r in sub.itertuples(index=False):
        key = getattr(r, "district_number")
        if key:
            mapping[key] = {k: getattr(r, k) for k in use_cols}

    if _profile_enabled():
        _debug(
            f"[enrich:{dataset}] prepared mapping for {len(mapping)} district numbers (select={use_cols})"
        )

    # Apply to repo districts
    updated = 0
    missing_no_number = 0
    missing_no_match = 0
    sample_missing: list[str] = []
    # repo may expose either ._districts (dict) or .districts (list/iterable)
    districts_iter = (
        repo._districts.values()
        if hasattr(repo, "_districts")
        else getattr(repo, "districts", [])
    )

    for d in districts_iter:
        dn = _canon_district_number(getattr(d, "district_number", None))
        if not dn:
            missing_no_number += 1
            continue
        attrs = mapping.get(dn)
        if not attrs:
            missing_no_match += 1
            if _profile_enabled() and len(sample_missing) < 10:
                sample_missing.append(str(getattr(d, "district_number", None)))
            continue
        # ensure meta exists for dynamic attribute overlay
        if getattr(d, "meta", None) is None:
            try:
                d.meta = {}
            except Exception:
                pass
        for k, v in attrs.items():
            # Source attribute lives in meta → readable via d.__getattr__ fallback
            if d.meta is not None:
                d.meta[k] = v
            # Optional alias to a canonical field (e.g., overall_rating_2025 → rating)
            if aliases and k in aliases:
                try:
                    setattr(d, aliases[k], v)
                except Exception:
                    # some District implementations may freeze fields; meta still holds value
                    pass
        updated += 1

    if _profile_enabled():
        _debug(
            f"[enrich:{dataset}] updated={updated} missing_no_number={missing_no_number} missing_no_match={missing_no_match}"
        )
        if sample_missing:
            _debug(
                f"[enrich:{dataset}] sample unmatched district_numbers: {sample_missing}"
            )

    total_repo_districts = updated + missing_no_number + missing_no_match
    summary_year = resolved_year if resolved_year is not None else year
    summary_msg = (
        f"District enrichment '{dataset}' ({summary_year}) updated {updated}"
        f" of {total_repo_districts} district object(s)."
    )
    if missing_no_number or missing_no_match:
        summary_msg += (
            " Missing district_number: "
            f"{missing_no_number}; no matching TAPR row: {missing_no_match}."
        )
    print(summary_msg)

    return resolved_year, updated


# -----------------------------
# Decorator-driven enricher (used by load_data2 pipeline)
# -----------------------------


@enricher("accountability")
class DistrictEnrichmentTable(Enricher):
    def apply(self, repo, cfg_path: str, year: int) -> Dict[str, Any]:
        year_resolved, updated = _apply_district_enrichment_table(
            repo,
            cfg_path,
            "accountability",
            year,
            select=["overall_rating_2025"],
            rename={"2025 Overall Rating": "overall_rating_2025"},
            aliases={"overall_rating_2025": "rating"},
            reader_kwargs={"sheet_name": "2011-2025 Summary"},
        )
        return {"updated": updated, "year": year_resolved}


@enricher("district_tapr_student_staff_profile")
class DistrictTaprStudentStaffProfile(Enricher):
    def apply(self, repo, cfg_path: str, year: int) -> Dict[str, Any]:
        year_resolved, updated = _apply_district_enrichment_table(
            repo,
            cfg_path,
            "district_tapr_student_staff_profile",
            year,
            select=None,
        )
        return {"updated": updated, "year": year_resolved}


# -----------------------------
# Back-compat helper for legacy callers
# -----------------------------


def enrich_districts_from_config(
    repo,
    cfg_path: str,
    dataset: str,
    year: int,
    *,
    select=None,
    rename=None,
    aliases=None,
    reader_kwargs=None,
):
    """Compatibility wrapper. Returns (resolved_year, updated_count)."""
    return _apply_district_enrichment_table(
        repo,
        cfg_path,
        dataset,
        year,
        select=select,
        rename=rename,
        aliases=aliases,
        reader_kwargs=reader_kwargs,
    )
