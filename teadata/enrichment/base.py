from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd

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


class Enricher(ABC):
    """Each Enricher loads a dataset via config and writes attributes onto District/Campus."""

    @abstractmethod
    def apply(self, repo, cfg_path: str, year: int) -> Dict[str, Any]: ...
