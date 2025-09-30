from typing import Dict, Any
import pandas as pd
from .common import prepare_columns
from . import enricher
from .base import Enricher

from teadata.teadata_config import load_config

from teadata.teadata_config import normalize_campus_number_column


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

    # Rename spreadsheet headers to pythonic names
    if rename:
        df = df.rename(columns=rename)

    # Normalize campus number to 9-digit string (Excel-style variants handled later)
    df, found = normalize_campus_number_column(df, new_col="campus_number")
    if found is None:
        return resolved_year, 0

    # Column selection/cleanup
    if select is None:
        use_cols = [c for c in df.columns if c != "campus_number"]
    else:
        use_cols = [c for c in select if c in df.columns]
    for c in use_cols:
        if c in df.columns:
            df[c] = df[c].apply(
                lambda x: (
                    None
                    if (pd.isna(x) or (isinstance(x, str) and x.strip() == ""))
                    else (x.strip() if isinstance(x, str) else x)
                )
            )

    sub = df[["campus_number"] + use_cols].drop_duplicates("campus_number")
    mapping = {
        str(r.campus_number): {k: getattr(r, k) for k in use_cols}
        for r in sub.itertuples(index=False)
    }
    # allow Excel-style keys too (leading apostrophe)
    mapping.update({f"'{k}": v for k, v in mapping.items()})

    updated = 0
    for cobj in repo._campuses.values():
        cn = getattr(cobj, "campus_number", None)
        if not cn:
            continue
        attrs = mapping.get(str(cn)) or mapping.get(str(cn).lstrip("'`’"))
        if not attrs:
            continue
        for k, v in attrs.items():
            # Put source attr in meta so dot-access via __getattr__ works
            if getattr(cobj, "meta", None) is None:
                cobj.meta = {}
            cobj.meta[k] = v
            # Alias target (e.g., overall_rating_2025 -> rating)
            if aliases and k in aliases:
                try:
                    setattr(cobj, aliases[k], v)
                except Exception:
                    pass
        updated += 1

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
    """Compatibility wrapper for legacy callers. Returns (resolved_year, updated_count)."""
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
