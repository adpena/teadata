from collections.abc import Mapping, MutableMapping
import logging
from typing import Dict, Any, Callable

from teadata.entities import Campus
import pandas as pd
from . import enricher, _record_matches
from .base import Enricher, apply_entity_attributes, clean_value_columns, require_unique_entity_rows

from teadata.teadata_config import load_config

from teadata.teadata_config import (
    canonical_campus_number,
    normalize_enrichment_identifiers,
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


logger = logging.getLogger(__name__)


# Shared helper for campus enrichment tables
def _apply_campus_enrichment_table(
    repo,
    cfg_path: str,
    dataset: str,
    year: int,
    *,
    select=None,
    rename=None,
    aliases=None,
    reader_kwargs=None,
    transform_df: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    record_hook: (
        Callable[[Any, Dict[str, Any], int | None], Dict[str, Any] | None] | None
    ) = None,
):
    cfg = load_config(cfg_path)
    resolved_year, df = cfg.load_df(
        dataset, year, section="data_sources", **(reader_kwargs or {})
    )
    source_rows = len(df)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"[enrich:{dataset}] resolved_year={resolved_year} (requested={year}) rows={len(df)}"
        )

    # Rename spreadsheet headers to pythonic names
    if rename:
        df = df.rename(columns=rename)

    source_column = "Campus Number" if dataset == "accountability" and (reader_kwargs or {}).get("reader_profile") == "ratings" else None
    df = normalize_enrichment_identifiers(
        df, key="campus_number", dataset=dataset, year=resolved_year, column=source_column,
    )
    require_unique_entity_rows(df, "campus_number", dataset, resolved_year)

    if transform_df:
        df = transform_df(df)
        if "campus_number" not in df.columns:
            raise KeyError(
                "campus enrichment transform_df removed the required 'campus_number' column"
            )
        df = normalize_enrichment_identifiers(
            df, key="campus_number", dataset=dataset, year=resolved_year, column="campus_number",
        )
        require_unique_entity_rows(df, "campus_number", dataset, resolved_year)

    if logger.isEnabledFor(logging.DEBUG):
        valid_rows = len(df)
        unique_keys = df["campus_number"].nunique(dropna=True)
        logger.debug(
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
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
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

    clean_value_columns(df, use_cols)

    # Build mapping from canonical campus_number -> selected attrs
    mapping = df.set_index("campus_number")[use_cols].to_dict(orient="index")

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"[enrich:{dataset}] prepared mapping for {len(mapping)} campus numbers (select={use_cols})"
        )

    matches: list[tuple[Campus, Mapping[str, Any]]] = []
    missing_no_number = 0
    missing_no_match = 0
    not_applicable = 0
    sample_missing: list[str] = []
    for cobj in repo._campuses.values():
        if getattr(cobj, "is_private", False):
            not_applicable += 1
            continue
        cn = getattr(cobj, "campus_number", None)
        if not cn:
            missing_no_number += 1
            continue
        can = canonical_campus_number(cn)
        if not can:
            missing_no_number += 1

            continue
        attrs = mapping.get(can)
        if not attrs:
            missing_no_match += 1
            if logger.isEnabledFor(logging.DEBUG) and len(sample_missing) < 10:
                sample_missing.append(str(cn))
            continue
        attrs_for_meta = attrs
        if record_hook:
            maybe_new_attrs = record_hook(cobj, dict(attrs), resolved_year)
            if isinstance(maybe_new_attrs, dict):
                attrs_for_meta = maybe_new_attrs
        matches.append((cobj, attrs_for_meta))

    apply_entity_attributes(matches, dataset=dataset, year=resolved_year, aliases=aliases)
    updated = len(matches)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"[enrich:{dataset}] updated={updated} missing_no_number={missing_no_number} missing_no_match={missing_no_match}"
        )
        if sample_missing:
            logger.debug(
                f"[enrich:{dataset}] sample unmatched campus_numbers: {sample_missing}"
            )

    _record_matches(repo, dataset, year, resolved_year, "campus",
                    matched=updated, unmatched=missing_no_match,
                    missing_identifier=missing_no_number, source_rows=source_rows,
                    not_applicable=not_applicable)
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

    if not select:
        raise ValueError(
            "campus PEIMS financial enrichment requires an explicit `select` collection of column names"
        )
    return _apply_campus_enrichment_table(
        repo, cfg_path, dataset, year, select=select, rename=rename,
        reader_kwargs=reader_kwargs,
    )


def _apply_campus_planned_closures(
    repo,
    cfg_path: str,
    dataset: str,
    year: int,
    *,
    reader_kwargs=None,
):
    result = _apply_campus_enrichment_table(
        repo, cfg_path, dataset, year,
        select=["campus_number", "facing_closure", "closure_date"],
        reader_kwargs=reader_kwargs,
    )
    # Expose nullable closure fields even for campuses outside this sparse source.
    for campus in repo._campuses.values():
        if not isinstance(campus.meta, MutableMapping):
            campus.meta = dict(campus.meta) if campus.meta is not None else {}
        for col in ("facing_closure", "closure_date"):
            campus.meta.setdefault(col, None)
    return result


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
    if ds_lower == "campus_planned_closures":
        return _apply_campus_planned_closures(
            repo,
            cfg_path,
            dataset,
            year,
            reader_kwargs=reader_kwargs,
        )

    return _apply_campus_enrichment_table(
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
class CampusEnrichmentTable(Enricher):
    def apply(self, repo, cfg_path: str, year: int) -> Dict[str, Any]:
        year_resolved, updated = _apply_campus_enrichment_table(
            repo,
            cfg_path,
            "accountability",
            year,
            select=["overall_rating"],
                        aliases={"overall_rating": "rating"},
            reader_kwargs={"reader_profile": "ratings"},
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
        yr, updated = _apply_campus_enrichment_table(
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


@enricher("campus_tapr_historical_enrollment")
class CampusTaprHistoricalEnrollment(Enricher):
    def apply(self, repo, cfg_path: str, year: int) -> Dict[str, Any]:
        yr, updated = _apply_campus_enrichment_table(
            repo,
            cfg_path,
            "campus_tapr_historical_enrollment",
            year,
            select=None,
            rename=None,
            aliases=None,
            reader_kwargs=None,
        )
        return {"updated": updated, "year": yr}


@enricher("campus_planned_closures")
class CampusPlannedClosures(Enricher):
    def apply(self, repo, cfg_path: str, year: int) -> Dict[str, Any]:
        yr, updated = _apply_campus_planned_closures(
            repo,
            cfg_path,
            "campus_planned_closures",
            year,
            reader_kwargs=None,
        )
        return {"updated": updated, "year": yr}
