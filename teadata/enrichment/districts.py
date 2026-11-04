from collections.abc import Mapping
import logging
from typing import Dict, Any, Optional

from teadata.entities import District

from . import enricher, _record_matches
from .base import Enricher, apply_entity_attributes, clean_value_columns, require_unique_entity_rows

from teadata.teadata_config import load_config
from teadata.teadata_config import canonical_district_number
from teadata.teadata_config import normalize_enrichment_identifiers

logger = logging.getLogger(__name__)


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
    source_rows = len(df)
    ratings = dataset == "accountability" and (reader_kwargs or {}).get("reader_profile") == "ratings"
    if ratings:
        # TEA's historical summary interleaves district and campus rows. A
        # district rating is a district row, regardless of publisher row order.
        df = df.loc[df["Campus Number"].isna()].copy()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"[enrich:{dataset}] resolved_year={resolved_year} (requested={year}) rows={len(df)}"
        )

    # Optional column rename from spreadsheet-style headers to pythonic names
    if rename:
        df = df.rename(columns=rename)

    df = normalize_enrichment_identifiers(
        df, key="district_number", dataset=dataset, year=resolved_year,
        column="District Number" if ratings else None,
    )
    require_unique_entity_rows(df, "district_number", dataset, resolved_year)

    if logger.isEnabledFor(logging.DEBUG):
        valid_rows = len(df)
        unique_keys = df["district_number"].nunique(dropna=True)
        logger.debug(
            f"[enrich:{dataset}] canonical district numbers -> valid_rows={valid_rows} unique_keys={unique_keys}"
        )

    if select is None:
        use_cols = [c for c in df.columns if c != "district_number"]
        if not use_cols:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"[enrich:{dataset}] abort: no columns available after excluding district_number"
                )
            return resolved_year, 0
    else:
        use_cols = [c for c in select if c != "district_number"]
        missing = [c for c in use_cols if c not in df.columns]
        if missing:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
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
    clean_value_columns(df, use_cols)

    # Build mapping keyed by canonical district_number → {attr: value}
    mapping = df.set_index("district_number")[use_cols].to_dict(orient="index")

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"[enrich:{dataset}] prepared mapping for {len(mapping)} district numbers (select={use_cols})"
        )

    # Apply to repo districts
    matches: list[tuple[District, Mapping[str, Any]]] = []
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
            if logger.isEnabledFor(logging.DEBUG) and len(sample_missing) < 10:
                sample_missing.append(str(getattr(d, "district_number", None)))
            continue
        matches.append((d, attrs))

    apply_entity_attributes(matches, dataset=dataset, year=resolved_year, aliases=aliases)
    updated = len(matches)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"[enrich:{dataset}] updated={updated} missing_no_number={missing_no_number} missing_no_match={missing_no_match}"
        )
        if sample_missing:
            logger.debug(
                f"[enrich:{dataset}] sample unmatched district_numbers: {sample_missing}"
            )

    _record_matches(repo, dataset, year, resolved_year, "district",
                    matched=updated, unmatched=missing_no_match,
                    missing_identifier=missing_no_number, source_rows=source_rows)
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
            select=["overall_rating"],
                        aliases={"overall_rating": "rating"},
            reader_kwargs={"reader_profile": "ratings"},
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
