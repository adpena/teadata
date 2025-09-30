from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from teadata.teadata_config import load_config
from teadata.teadata_config import (
    normalize_district_number_column,
    normalize_district_number_value,
)

from teadata.classes import DataEngine, District


def _read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()
    if suf in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if suf in {".csv"}:
        return pd.read_csv(p)
    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    # Fallback: try CSV then Excel
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.read_excel(p)


def add_charter_networks_from_config(
    repo: DataEngine,
    cfg_path: str | Path,
    year: int,
    *,
    dataset_name: str = "charter_reference",
    section: str = "data_sources",
    code_col: str = "District Number",
    name_col: str = "District Name",
    rating_col: Optional[str] = None,
    aea_col: Optional[str] = None,
    enrollment_col: Optional[str] = None,
) -> int:
    """
    Load a CSV/Parquet of statewide charter networks and inject them as Districts
    with boundary=None if they do not already exist.

    Returns the number of Districts created.
    """
    try:
        cfg = load_config(cfg_path)
        _, src = cfg.resolve(dataset_name, year, section=section)
    except Exception:
        # not configured = nothing to add
        return 0

    df = _read_table(src)

    # column access (case-insensitive convenience)
    cols = {str(c).lower(): c for c in df.columns if c is not None}

    def col(name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        key = str(name).lower()
        return cols.get(key, name)

    code_col = col(code_col) or "District Number"
    name_col = col(name_col) or "District Name"
    rating_col = col(rating_col) if rating_col else None
    aea_col = col(aea_col) if aea_col else None
    enrollment_col = col(enrollment_col) if enrollment_col else None

    created = 0
    for _, row in df.iterrows():
        try:
            dn = normalize_district_number_value(row.get(code_col))
            if not dn:
                continue
            # ensure canonical apostrophe-prefix to match repo keys
            if not str(dn).startswith("'"):
                dn = f"'{dn}"

            name = str(row.get(name_col, "") or "").strip() or f"Network {dn}"
            rating = str(row.get(rating_col, "") or "").strip() if rating_col else ""
            aea = row.get(aea_col) if aea_col else None

            enrollment = None
            if enrollment_col:
                try:
                    val = row.get(enrollment_col)
                    if pd.notna(val):
                        enrollment = int(val)
                except Exception:
                    enrollment = None
        except Exception:
            continue

        # already present?
        exists = next(
            (
                d
                for d in repo._districts.values()
                if getattr(d, "district_number", None) == dn
            ),
            None,
        )
        if exists:
            # optionally fill in missing basics without overriding real data
            if name and (not exists.name or exists.name == "Unknown District"):
                exists.name = name
            if rating and not getattr(exists, "rating", None):
                exists.rating = rating
            if aea is not None and not getattr(exists, "aea", None):
                exists.aea = aea
            if enrollment is not None and getattr(exists, "enrollment", None) is None:
                exists.enrollment = enrollment
            continue

        # create a virtual (statewide) district (no geometry)
        import uuid

        d = District(
            id=uuid.uuid4(),
            name=name,
            enrollment=enrollment if enrollment is not None else 0,
            rating=rating,
            aea=aea,
            boundary=None,  # important: no geometry
        )
        d.district_number = dn  # keep canonical code on the object
        repo.add_district(d)
        created += 1

    return created
