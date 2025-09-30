import pandas as pd
from typing import Dict, Any
from .common import pick_sheet_with_columns, prepare_columns
from . import enricher
from .base import Enricher

try:
    from teadata_config import load_config
except ImportError:  # running inside a package (python -m ...)
    from ..teadata_config import load_config  # type: ignore

try:
    from teadata_config import normalize_district_number_column
except ImportError:
    from ..teadata_config import normalize_district_number_column  # type: ignore


# Shared helper for district accountability enrichment
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
    cfg = load_config(cfg_path)
    resolved_year, df = cfg.load_df(
        dataset, year, section="data_sources", **(reader_kwargs or {})
    )

    # Rename spreadsheet headers to pythonic names
    if rename:
        df = df.rename(columns=rename)

    # Normalize district number
    df, found = normalize_district_number_column(df, new_col="district_number")
    if found is None:
        return resolved_year, 0

    # Column selection/cleanup
    if select is None:
        use_cols = [c for c in df.columns if c != "district_number"]
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

    sub = df[["district_number"] + use_cols].drop_duplicates("district_number")
    mapping = {
        str(r.district_number).lstrip("'`’"): {k: getattr(r, k) for k in use_cols}
        for r in sub.itertuples(index=False)
    }

    updated = 0
    for d in repo._districts.values():
        dn = str(getattr(d, "district_number", "")).lstrip("'`’")
        attrs = mapping.get(dn)
        if not attrs:
            continue
        for k, v in attrs.items():
            # Write source attr to meta so dot-access works via __getattr__
            if getattr(d, "meta", None) is None:
                d.meta = {}
            d.meta[k] = v
            # Alias target (e.g., overall_rating_2025 -> rating)
            if aliases and k in aliases:
                try:
                    setattr(d, aliases[k], v)
                except Exception:
                    pass
        updated += 1

    return resolved_year, updated


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


# Exported function for load_data2.py and legacy callers
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
    """Compatibility wrapper for legacy callers. Returns (resolved_year, updated_count)."""
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
