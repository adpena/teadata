from typing import Dict, Any

import teadata.classes as classes_mod
import pandas as pd
from . import enricher
from .base import Enricher

from teadata.teadata_config import load_config

from teadata.teadata_config import (
    canonical_campus_number,
    normalize_campus_number_column,
)

DEFAULT_PEIMS_FINANCIAL_COLUMNS: list[str] = [
    "instruction_af_perc",
    "transportation_af_per_student",
    "extracurricular_af_per_student",
    "security_monitoring_af_per_student",
    "students_w_disabilities_af_per_student",
    "bilingual_ed_af_per_student",
    "dyslexia_or_related_disorder_serv_af_per_student",
    "ccmr_af_per_student",
    "guidance_counseling_af_per_student",
    "school_leadership_af_per_student",
]

DEFAULT_PEIMS_FINANCIAL_COLUMNS: list[str] = [
    "instruction_af_perc",
    "transportation_af_per_student",
    "extracurricular_af_per_student",
    "security_monitoring_af_per_student",
    "students_w_disabilities_af_per_student",
    "bilingual_ed_af_per_student",
    "dyslexia_or_related_disorder_serv_af_per_student",
    "ccmr_af_per_student",
    "guidance_counseling_af_per_student",
    "school_leadership_af_per_student",
]

def _profile_enabled() -> bool:
    return bool(getattr(classes_mod, "ENABLE_PROFILING", False))

def _debug(msg: str) -> None:
    if _profile_enabled():
        print(msg)


def _canon_campus_number(x) -> str | None:
    return canonical_campus_number(x)

def _canon_series(series: pd.Series) -> pd.Series:
    """Vectorized canonicalization for a pandas Series of campus numbers."""
    return series.map(canonical_campus_number)

def _build_campus_multi_index(repo) -> dict[str, Any]:
    """Return a dict mapping multiple key shapes to Campus.id for robust matching.
    Keys created per campus_number:
      - canonical with apostrophe: '#########
      - without apostrophe: #########
      - int form (no leading zeros): e.g., 123456789 as string
    """
    idx: dict[str, Any] = {}
    for c in repo._campuses.values():
        cn = getattr(c, "campus_number", None)
        if not cn:
            continue
        key_can = _canon_campus_number(cn)
        if not key_can:
            continue
        idx[key_can] = c.id
        # Without apostrophe
        no_tick = key_can[1:]
        idx[no_tick] = c.id
        # Int-like
        if no_tick.isdigit():
            idx[str(int(no_tick))] = c.id
    return idx


# Shared helper for campus accountability enrichment
def _apply_campus_accountability(
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
    cfg = load_config(cfg_path)
    resolved_year, df = cfg.load_df(
        dataset, year, section="data_sources", **(reader_kwargs or {})
    )

    if _profile_enabled():
        _debug(
            f"[enrich:{dataset}] resolved_year={resolved_year} (requested={year}) rows={len(df)}"
        )

    # Rename spreadsheet headers to pythonic names
    if rename:
        df = df.rename(columns=rename)

    # Ensure we have a campus_number column; try standard helper, then fallbacks
    df, found = normalize_campus_number_column(df, new_col="campus_number")
    if found is None and "campus_number" not in df.columns:
        # Try common variations
        for guess in ("Campus Number", "CAMPUS", "campus", "Campus"):
            if guess in df.columns:
                df = df.rename(columns={guess: "campus_number"})
                break
    if "campus_number" not in df.columns:
        if _profile_enabled():
            _debug(
                f"[enrich:{dataset}] abort: campus_number column missing after normalization (found={found})"
            )
        return resolved_year, 0

    # Canonicalize to leading-apostrophe format used across the repo
    df["campus_number"] = _canon_series(df["campus_number"])
    df = df[df["campus_number"].notna()]

    if _profile_enabled():
        valid_rows = len(df)
        unique_keys = df["campus_number"].nunique(dropna=True)
        _debug(
            f"[enrich:{dataset}] canonical campus numbers -> valid_rows={valid_rows} unique_keys={unique_keys}"
        )

    if select is None:
        use_cols = [c for c in df.columns if c != "campus_number"]
    else:
        missing: list[str] = []
        use_cols = []
        for col in select:
            if col == "campus_number":
                continue
            if col not in df.columns:
                missing.append(col)
                continue
            if col not in use_cols:
                use_cols.append(col)
        if missing:
            if _profile_enabled():
                _debug(
                    f"[enrich:{dataset}] missing columns after rename: {', '.join(sorted(missing))}"
                )
            raise KeyError(
                "campus enrichment missing expected columns after rename: "
                + ", ".join(sorted(missing))
            )
        if not use_cols:
            raise ValueError(
                "campus enrichment requires a non-empty `select` collection of column names"
            )

    for c in use_cols:
        if c in df.columns:
            df[c] = df[c].apply(
                lambda x: (
                    None
                    if (pd.isna(x) or (isinstance(x, str) and x.strip() == ""))
                    else (x.strip() if isinstance(x, str) else x)
                )
            )

    # Build mapping from canonical campus_number -> selected attrs
    mapping: dict[str, dict[str, Any]] = {}
    sub = df[["campus_number"] + use_cols].drop_duplicates("campus_number")
    for r in sub.itertuples(index=False):
        key = getattr(r, "campus_number")
        if not key:
            continue
        record = {k: getattr(r, k) for k in use_cols}
        mapping[key] = record
        digits = key[1:]
        mapping.setdefault(digits, record)
        if digits.isdigit():
            mapping.setdefault(str(int(digits)), record)

    if _profile_enabled():
        _debug(
            f"[enrich:{dataset}] prepared mapping for {len(mapping)} campus numbers (select={use_cols})"
        )

    # Multi-key index for robust matching
    updated = 0
    missing_no_number = 0
    missing_no_match = 0
    sample_missing: list[str] = []
    for cobj in repo._campuses.values():
        cn = getattr(cobj, "campus_number", None)
        if not cn:
            missing_no_number += 1
            continue
        can = _canon_campus_number(cn)
        if not can:
            missing_no_number += 1

            continue
        digits = can[1:]
        attrs = mapping.get(can) or mapping.get(digits)
        if attrs is None and digits.isdigit():
            attrs = mapping.get(str(int(digits)))
        if not attrs:
            missing_no_match += 1
            if _profile_enabled() and len(sample_missing) < 10:
                sample_missing.append(str(cn))
            continue
        if getattr(cobj, "meta", None) is None:
            cobj.meta = {}
        for k, v in attrs.items():
            cobj.meta[k] = v
            if aliases and k in aliases:
                try:
                    setattr(cobj, aliases[k], v)
                except Exception:
                    pass
        updated += 1

    if _profile_enabled():
        _debug(
            f"[enrich:{dataset}] updated={updated} missing_no_number={missing_no_number} missing_no_match={missing_no_match}"
        )
        if sample_missing:
            _debug(
                f"[enrich:{dataset}] sample unmatched campus_numbers: {sample_missing}"
            )

    return resolved_year, updated


# Shared helper for campus PEIMS financials enrichment
def _apply_campus_peims_financials(
    repo,
    cfg_path: str,
    dataset: str,
    year: int,
    *,
    select=None,
    rename=None,
    reader_kwargs=None,
):
    """Load campus-level PEIMS financials and attach selected attributes.

    Only the explicitly requested columns in ``select`` are considered (after any
    optional ``rename`` mapping is applied). This keeps the enrichment predictable
    and avoids accidentally pulling in new headers when TEA updates the export
    format. Campus identifiers continue to be normalized with the shared helper
    so joins stay resilient across spreadsheets.
    """

    cfg = load_config(cfg_path)
    resolved_year, df = cfg.load_df(
        dataset, year, section="data_sources", **(reader_kwargs or {})
    )

    if _profile_enabled():
        _debug(
            f"[enrich:{dataset}] resolved_year={resolved_year} (requested={year}) rows={len(df)}"
        )

    if rename:
        df = df.rename(columns=rename)

    df, found = normalize_campus_number_column(df, new_col="campus_number")
    if found is None and "campus_number" not in df.columns:
        for guess in ("Campus Number", "CAMPUS", "campus", "Campus"):
            if guess in df.columns:
                df = df.rename(columns={guess: "campus_number"})
                break
    if "campus_number" not in df.columns:
        if _profile_enabled():
            _debug(
                f"[enrich:{dataset}] abort: campus_number column missing after normalization (found={found})"
            )
        return resolved_year, 0

    df["campus_number"] = _canon_series(df["campus_number"])
    df = df[df["campus_number"].notna()]

    if _profile_enabled():
        valid_rows = len(df)
        unique_keys = df["campus_number"].nunique(dropna=True)
        _debug(
            f"[enrich:{dataset}] canonical campus numbers -> valid_rows={valid_rows} unique_keys={unique_keys}"
        )

    if not select:
        raise ValueError(
            "campus PEIMS financial enrichment requires an explicit `select` collection of column names"
        )

    use_cols = list(select)
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        if _profile_enabled():
            _debug(
                f"[enrich:{dataset}] missing columns after rename: {', '.join(sorted(missing))}"
            )
        raise KeyError(
            "campus PEIMS financial enrichment missing expected columns after rename: "
            + ", ".join(sorted(missing))
        )

    sub = df[["campus_number"] + use_cols].drop_duplicates("campus_number")

    mapping: dict[str, dict[str, Any]] = {}
    for r in sub.itertuples(index=False):
        key = getattr(r, "campus_number")
        if not key:
            continue

        record = {k: getattr(r, k) for k in use_cols}

        mapping[key] = record
        digits = key[1:]
        mapping.setdefault(digits, record)
        if digits.isdigit():
            mapping.setdefault(str(int(digits)), record)

    if _profile_enabled():
        _debug(
            f"[enrich:{dataset}] prepared mapping for {len(mapping)} campus numbers (select={use_cols})"
        )

    updated = 0
    missing_no_number = 0
    missing_no_match = 0
    sample_missing: list[str] = []
    for campus in repo._campuses.values():
        cn = getattr(campus, "campus_number", None)
        if not cn:
            missing_no_number += 1
            continue
        can = _canon_campus_number(cn)
        if not can:
            missing_no_number += 1
            continue
        digits = can[1:]
        attrs = mapping.get(can) or mapping.get(digits)
        if attrs is None and digits.isdigit():
            attrs = mapping.get(str(int(digits)))
        if not attrs:
            missing_no_match += 1
            if _profile_enabled() and len(sample_missing) < 10:
                sample_missing.append(str(cn))
            continue
        if getattr(campus, "meta", None) is None:
            campus.meta = {}
        wrote = False
        for k, v in attrs.items():
            campus.meta[k] = v
            wrote = True
        if wrote:
            updated += 1

    if _profile_enabled():
        _debug(
            f"[enrich:{dataset}] updated={updated} missing_no_number={missing_no_number} missing_no_match={missing_no_match}"
        )
        if sample_missing:
            _debug(
                f"[enrich:{dataset}] sample unmatched campus_numbers: {sample_missing}"
            )

    return resolved_year, updated


def enrich_campuses_from_config(
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
    """Dispatch to the appropriate enrichment routine based on dataset name."""

    ds_lower = dataset.lower()
    if ds_lower == "campus_peims_financials":
        return _apply_campus_peims_financials(
            repo,
            cfg_path,
            dataset,
            year,
            select=select,
            rename=rename,
            reader_kwargs=reader_kwargs,
        )

    return _apply_campus_accountability(
        repo,
        cfg_path,
        dataset,
        year,
        select=select,
        rename=rename,
        aliases=aliases,
        reader_kwargs=reader_kwargs,
    )


@enricher("campus_accountability")
class CampusAccountability(Enricher):
    def apply(self, repo, cfg_path: str, year: int) -> Dict[str, Any]:
        year_resolved, updated = _apply_campus_accountability(
            repo,
            cfg_path,
            "campus_accountability",
            year,
            select=["overall_rating_2025"],
            rename={"2025 Overall Rating": "overall_rating_2025"},
            aliases={"overall_rating_2025": "rating"},
            reader_kwargs={"sheet_name": "2011-2025 Summary"},
        )
        return {"updated": updated, "year": year_resolved}


@enricher("campus_peims_financials")
class CampusPEIMSFinancials(Enricher):
    def apply(self, repo, cfg_path: str, year: int) -> Dict[str, Any]:
        yr, updated = _apply_campus_peims_financials(
            repo,
            cfg_path,
            "campus_peims_financials",
            year,
            select=DEFAULT_PEIMS_FINANCIAL_COLUMNS,
            rename=None,
            reader_kwargs=None,
        )
        return {"updated": updated, "year": yr}


@enricher("campus_tapr_student_staff_profile")
class CampusTaprStudentStaffProfile(Enricher):
    def apply(self, repo, cfg_path: str, year: int) -> Dict[str, Any]:
        yr, updated = _apply_campus_accountability(
            repo,
            cfg_path,
            "campus_tapr_student_staff_profile",
            year,
            select=None,
            rename=None,
            aliases=None,
            reader_kwargs=None,
        )
        return {"updated": updated, "year": yr}
