"""Strict source identifiers; interactive lookup normalization is separate."""

from __future__ import annotations

import math
import numbers
import re
from typing import Any
from urllib.parse import quote, unquote_to_bytes


MAX_CAMPUS_LOOKUP_KEY_BYTES = 1024


def normalize_campus_lookup_key(value: str | None) -> str | None:
    """Canonicalize a public TEA lookup or an explicitly qualified private key.

    Private IDs retain case and prefixes. Percent encoding is URI-component
    encoding, never an entity-resolution rule. The encoded ASCII key is bounded
    for transport and indexed storage independently of today's ID lengths.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("Campus lookup keys must be strings")
    if ":" in value:
        namespace, encoded = value.split(":", 1)
        if namespace != "tefa":
            raise ValueError("Unsupported private campus identity namespace")
        if re.search(r"%(?![0-9a-fA-F]{2})", encoded):
            raise ValueError("Malformed percent encoding in campus lookup key")
        try:
            identifier = unquote_to_bytes(encoded).decode("utf-8", errors="strict")
        except UnicodeError as error:
            raise ValueError("Private campus identity must be valid UTF-8") from error
        return campus_lookup_key(source_namespace=namespace, source_id=identifier, is_private=True)
    try:
        number = canonical_source_identifier(value, width=9)
    except ValueError:
        return None
    return number[1:] if number is not None else None


def campus_lookup_key(
    *, campus_number: Any = None, source_namespace: str | None = None,
    source_id: str | None = None, is_private: bool = False,
) -> str | None:
    """Build the sole lookup identity from explicit entity fields.

    Public in-memory entities may lack a TEA number; released stores require a
    nonempty key. Private entities require a supported, exact source identity.
    """
    if not is_private:
        if source_namespace is not None or source_id is not None:
            raise ValueError("Public campuses use their real TEA campus number")
        try:
            number = canonical_source_identifier(campus_number, width=9)
        except ValueError as error:
            raise ValueError("Public campus identity requires a real TEA campus number") from error
        return number[1:] if number is not None else None
    if campus_number is not None:
        raise ValueError("Private schools cannot have a fabricated TEA campus number")
    if source_namespace != "tefa" or not isinstance(source_id, str) or not source_id or "\x00" in source_id:
        raise ValueError("Private schools require a supported namespace and exact source ID")
    try:
        key = source_namespace + ":" + quote(source_id, safe="", encoding="utf-8", errors="strict")
    except UnicodeError as error:
        raise ValueError("Private campus identity must be valid UTF-8") from error
    if len(key.encode("ascii")) > MAX_CAMPUS_LOOKUP_KEY_BYTES:
        raise ValueError("Campus lookup key exceeds the shared 1024-byte bound")
    return key


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


def canonical_source_identifiers(values: Any, *, width: int):
    """Vectorize the scalar source-string contract without changing raw tokens.

    This explicit native boundary accepts Arrow string arrays only. Missing
    strings remain null; malformed nonblank identifiers fail rather than being
    extracted from notes, rounded, or silently discarded.
    """
    import pyarrow as pa
    import pyarrow.compute as arrow_compute

    # Arrow registers most compute kernels dynamically at import time.
    pc: Any = arrow_compute

    if type(width) is not int or width < 1:
        raise ValueError("Identifier width must be a positive integer")
    if not isinstance(values, (pa.Array, pa.ChunkedArray)) or not (
        pa.types.is_string(values.type) or pa.types.is_large_string(values.type)
    ):
        raise ValueError("Source identifier batches require Arrow strings")
    # Exact characters accepted by Python's str.strip, shared with the scalar
    # contract (including the four ASCII information separators).
    whitespace = " \t\n\r\v\f\x1c\x1d\x1e\x1f\x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"
    text = pc.utf8_trim(values, characters=whitespace)
    missing = pc.or_(pc.is_null(text), pc.fill_null(pc.equal(text, ""), False))
    text = pc.replace_substring_regex(text, pattern="^'", replacement="")
    text = pc.utf8_trim(text, characters=whitespace)
    valid = pc.match_substring_regex(text, pattern=rf"^[0-9]{{1,{width}}}(?:\.0+)?$")
    digits = pc.replace_substring_regex(text, pattern=r"\.0+$", replacement="")
    valid = pc.and_(valid, pc.invert(pc.match_substring_regex(digits, pattern="^0+$")))
    bad = pc.and_(pc.invert(missing), pc.invert(pc.fill_null(valid, False)))
    if pc.any(bad).as_py():
        position = pc.index(bad, True).as_py()
        raise ValueError(f"Invalid source identifier at batch row {position + 1}: {values[position].as_py()!r}")
    normalized = pc.binary_join_element_wise(pa.scalar("'", type=values.type),
        pc.utf8_lpad(digits, width=width, padding="0"), pa.scalar("", type=values.type))
    return pc.if_else(missing, pa.scalar(None, type=values.type), normalized)
