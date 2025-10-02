import pandas as pd
from typing import Dict, Any, Optional

from .common import pick_sheet_with_columns, prepare_columns  # optional utilities if you use them elsewhere
from . import enricher
from .base import Enricher

from teadata.teadata_config import load_config
from teadata.teadata_config import normalize_district_number_column
from teadata.teadata_config import normalize_district_number_value


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
    if x is None:
        return None
    try:
        s = str(x).strip()
        if not s:
            return None
        # strip common Excel leading quote/backtick variants before normalizing
        if s.startswith(("'", "`", "’")):
            s = s[1:].strip()
        norm = normalize_district_number_value(s)  # returns 6-digit zero-padded (no apostrophe)
        if norm is None:
            return None
        return "'" + norm
    except Exception:
        return None


# -----------------------------
# Core enrichment routine used by both the @enricher class and legacy wrapper
# -----------------------------

def _apply_district_accountability(
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
    resolved_year, df = cfg.load_df(dataset, year, section="data_sources", **(reader_kwargs or {}))

    # Optional column rename from spreadsheet-style headers to pythonic names
    if rename:
        df = df.rename(columns=rename)

    # Ensure we have a clean, machine-joinable district_number column
    df, found = normalize_district_number_column(df, new_col="district_number")

    # Canonicalize to leading-apostrophe format used across the repo
    if "district_number" in df.columns:
        df["district_number"] = df["district_number"].map(_canon_district_number)
        df = df[df["district_number"].notna()]

    # Decide which value columns to carry through
    if select is None:
        use_cols = [c for c in df.columns if c != "district_number"]
    else:
        use_cols = [c for c in select if c in df.columns]

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

    # Apply to repo districts
    updated = 0
    # repo may expose either ._districts (dict) or .districts (list/iterable)
    districts_iter = (
        repo._districts.values() if hasattr(repo, "_districts") else getattr(repo, "districts", [])
    )

    for d in districts_iter:
        dn = _canon_district_number(getattr(d, "district_number", None))
        if not dn:
            continue
        attrs = mapping.get(dn)
        if not attrs:
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

    return resolved_year, updated


# -----------------------------
# Decorator-driven enricher (used by load_data2 pipeline)
# -----------------------------

@enricher("accountability")
class DistrictAccountability(Enricher):
    def apply(self, repo, cfg_path: str, year: int) -> Dict[str, Any]:
        year_resolved, updated = _apply_district_accountability(
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
    return _apply_district_accountability(
        repo,
        cfg_path,
        dataset,
        year,
        select=select,
        rename=rename,
        aliases=aliases,
        reader_kwargs=reader_kwargs,
    )
