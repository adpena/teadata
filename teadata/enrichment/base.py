from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping
import logging
from types import MemberDescriptorType
from typing import Any, Dict

import pandas as pd

from teadata.entities import Campus, District
from teadata.identifiers import EntityIdentityError

logger = logging.getLogger(__name__)


class EnrichmentConflictError(ValueError):
    """A source would overwrite a different fact on an existing entity."""

try:  # optional performance dependency
    import polars as pl
except Exception:  # pragma: no cover - optional dependency
    pl = None  # type: ignore[assignment]


def _is_text_series(series: pd.Series, *, sample_size: int = 64) -> bool:
    if pd.api.types.is_string_dtype(series):
        return True
    if series.dtype != object:
        return False
    non_null = series[series.notna()]
    if non_null.empty:
        return False
    sample = non_null.head(sample_size)
    return bool(len(sample)) and all(isinstance(v, str) for v in sample)


def clean_value_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Trim text and normalize blanks without changing row identity or numbers."""
    text_cols = [col for col in columns if col in df.columns and _is_text_series(df[col])]
    cleaned_pdf = None
    if text_cols and pl is not None:
        try:
            text_frame = pl.from_pandas(df[text_cols], include_index=False)
            cleaned = text_frame.with_columns(
                [pl.col(col).str.strip_chars().alias(col) for col in text_cols]
            ).with_columns(
                [
                    pl.when(pl.col(col) == "").then(None).otherwise(pl.col(col)).alias(col)
                    for col in text_cols
                ]
            )
            cleaned_pdf = cleaned.to_pandas(use_pyarrow_extension_array=True)
        except Exception:
            # Retain the pandas path when the optional accelerator cannot
            # represent a source column. It preserves the original index too.
            pass
    for col in text_cols:
        if cleaned_pdf is not None:
            # Polars has no pandas index. Attach its same-order array by
            # position; a Series would align labels and corrupt filtered rows.
            df[col] = cleaned_pdf[col].array
        else:
            cleaned = df[col].astype("string").str.strip().replace("", pd.NA)
            df[col] = cleaned.where(cleaned.notna(), None)
    for col in columns:
        if col in df.columns:
            df[col] = df[col].where(pd.notna(df[col]), None)


def require_unique_entity_rows(df: pd.DataFrame, key: str, dataset: str, year: int | None) -> None:
    """Reject ambiguous joins before an enricher writes to any entity."""
    duplicates = df.loc[df[key].duplicated(keep=False), key]
    if not duplicates.empty:
        sample = sorted(map(str, duplicates.unique()))[:8]
        raise EntityIdentityError(
            f"{dataset} ({year}): {len(duplicates)} rows share duplicate {key} values, including {', '.join(sample)}"
        )


def apply_entity_attributes(
    matches: list[tuple[District | Campus, Mapping[str, Any]]], *,
    dataset: str, year: int | None, aliases: Mapping[str, str] | None = None,
) -> None:
    """Add facts to existing entities, checking the full batch for collisions first."""
    aliases = aliases or {}
    if len(set(aliases.values())) != len(aliases):
        raise EnrichmentConflictError(f"{dataset} ({year}): multiple fields target the same enrichment alias")
    for entity, attrs in matches:
        meta = entity.meta
        if meta is not None and not isinstance(meta, dict):
            # A lazy store can answer individual reads from a prefetched seed.
            # Validate the same full payload that the mutation will use.
            len(meta)
        for field, value in attrs.items():
            if meta is None or field not in meta or meta[field] is value:
                continue
            old = meta[field]
            if pd.api.types.is_scalar(old) and pd.api.types.is_scalar(value):
                old_missing, new_missing = bool(pd.isna(old)), bool(pd.isna(value))
                if (old_missing and new_missing) or (not old_missing and not new_missing and old == value):
                    continue
            raise EnrichmentConflictError(f"{dataset} ({year}): conflicting existing field {field!r} for {entity.id}")
        for field in attrs.keys() & aliases.keys():
            alias = aliases[field]
            if alias.startswith("_") or alias in {"id", "district_id", "district_number", "campus_number", "is_private", "source_namespace", "source_id", "meta"}:
                raise EntityIdentityError(f"{dataset} ({year}): enrichment cannot write identity field {alias!r}")
            if isinstance(entity, (District, Campus)):
                writable = isinstance(getattr(type(entity), alias, None), MemberDescriptorType)
            else:
                writable = alias in vars(entity)
            if not writable:
                raise AttributeError(f"{dataset} ({year}): enrichment alias {alias!r} is not a writable {type(entity).__name__} field")
    for entity, attrs in matches:
        if not isinstance(entity.meta, MutableMapping):
            entity.meta = dict(entity.meta) if entity.meta is not None else {}
        for field, value in attrs.items():
            entity.meta.setdefault(field, value)
            if field in aliases:
                alias = aliases[field]
                setattr(entity, alias, value)
                logger.debug("Applied declared enrichment alias %s for %s (%s), entity %s", alias, dataset, year, getattr(entity, "id", None))


class Enricher(ABC):
    """Each Enricher loads a dataset via config and writes attributes onto District/Campus."""

    @abstractmethod
    def apply(self, repo, cfg_path: str, year: int) -> Dict[str, Any]: ...
