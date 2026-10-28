"""Strict source identifiers; interactive lookup normalization is separate."""

from __future__ import annotations

import math
import numbers
import re
from typing import Any


def canonical_source_identifier(value: Any, *, width: int) -> str | None:
    """Normalize a scalar source code without extracting digits from bad data.

    Blank cells and genuine scalar NA values are missing. Nonblank malformed,
    fractional, Boolean, overlong, negative, and all-zero codes are invalid.
    Excel text apostrophes, omitted leading zeros, and integral float cells are
    accepted without converting an integer/string through floating point.
    """
    if type(width) is not int or width < 1:
        raise ValueError("Identifier width must be a positive integer")
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean is not a source identifier")
    import pandas as pd

    if not pd.api.types.is_scalar(value):
        raise ValueError("Source identifier must be a scalar")
    try:
        absent = pd.isna(value)
        if not isinstance(absent, bool):
            absent = bool(absent)
        if absent:
            return None
    except (TypeError, ValueError):
        raise ValueError("Source identifier must be a scalar") from None
    if isinstance(value, numbers.Integral):
        text = str(int(value))
    elif isinstance(value, numbers.Real):
        if not math.isfinite(value) or value != int(value):
            raise ValueError("Source identifier must be a finite integer")
        text = str(int(value))
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("'"):
            text = text[1:].strip()
        if not re.fullmatch(r"\d+(?:\.0+)?", text):
            raise ValueError("Malformed source identifier")
        text = text.partition(".")[0]
    else:
        raise ValueError("Unsupported source identifier type")
    if len(text) > width or not text.isascii() or not text.isdigit() or int(text) == 0:
        raise ValueError("Source identifier has invalid width or value")
    return "'" + text.zfill(width)
