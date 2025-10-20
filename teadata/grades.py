"""Grade range parsing and normalization helpers."""

from __future__ import annotations

from typing import Any, Optional, Tuple
import re

__all__ = [
    "GRADE_SEQUENCE",
    "GRADE_NAME_TO_CODE",
    "GRADE_NUMBER_TO_CODE",
    "GRADE_ALIAS_MAP",
    "GRADE_IGNORE_TOKENS",
    "GRADE_PHRASE_SUBS",
    "grade_spec_to_tokens",
    "grade_spec_to_segments",
    "grade_token_to_code",
    "grade_segment_to_span",
    "normalize_grade_bounds",
    "spans_to_bounds",
    "grade_value_to_code",
    "coerce_grade_spans",
    "coerce_grade_bounds",
]

GRADE_SEQUENCE: list[tuple[str, int]] = [
    ("EARLY_EDUCATION", 0),
    ("PREK", 1),
    ("KINDERGARTEN", 2),
    ("FIRST", 3),
    ("SECOND", 4),
    ("THIRD", 5),
    ("FOURTH", 6),
    ("FIFTH", 7),
    ("SIXTH", 8),
    ("SEVENTH", 9),
    ("EIGHTH", 10),
    ("NINTH", 11),
    ("TENTH", 12),
    ("ELEVENTH", 13),
    ("TWELFTH", 14),
    ("ADULT_EDUCATION", 15),
]

GRADE_NAME_TO_CODE: dict[str, int] = {name: code for name, code in GRADE_SEQUENCE}
GRADE_NUMBER_TO_CODE: dict[int, int] = {
    idx: GRADE_NAME_TO_CODE[name]
    for idx, name in enumerate(
        [
            "FIRST",
            "SECOND",
            "THIRD",
            "FOURTH",
            "FIFTH",
            "SIXTH",
            "SEVENTH",
            "EIGHTH",
            "NINTH",
            "TENTH",
            "ELEVENTH",
            "TWELFTH",
        ],
        start=1,
    )
}

GRADE_ALIAS_MAP: dict[str, int] = {
    "EE": GRADE_NAME_TO_CODE["EARLY_EDUCATION"],
    "EARLYEDUCATION": GRADE_NAME_TO_CODE["EARLY_EDUCATION"],
    "PREK": GRADE_NAME_TO_CODE["PREK"],
    "PRE": GRADE_NAME_TO_CODE["PREK"],
    "PK": GRADE_NAME_TO_CODE["PREK"],
    "PREPRIMARY": GRADE_NAME_TO_CODE["PREK"],
    "PRESCHOOL": GRADE_NAME_TO_CODE["PREK"],
    "PRE SCHOOL": GRADE_NAME_TO_CODE["PREK"],
    "K": GRADE_NAME_TO_CODE["KINDERGARTEN"],
    "KG": GRADE_NAME_TO_CODE["KINDERGARTEN"],
    "KDG": GRADE_NAME_TO_CODE["KINDERGARTEN"],
    "KINDER": GRADE_NAME_TO_CODE["KINDERGARTEN"],
    "KINDERGARTEN": GRADE_NAME_TO_CODE["KINDERGARTEN"],
    "FIRST": GRADE_NAME_TO_CODE["FIRST"],
    "SECOND": GRADE_NAME_TO_CODE["SECOND"],
    "THIRD": GRADE_NAME_TO_CODE["THIRD"],
    "FOURTH": GRADE_NAME_TO_CODE["FOURTH"],
    "FIFTH": GRADE_NAME_TO_CODE["FIFTH"],
    "SIXTH": GRADE_NAME_TO_CODE["SIXTH"],
    "SEVENTH": GRADE_NAME_TO_CODE["SEVENTH"],
    "EIGHTH": GRADE_NAME_TO_CODE["EIGHTH"],
    "NINTH": GRADE_NAME_TO_CODE["NINTH"],
    "TENTH": GRADE_NAME_TO_CODE["TENTH"],
    "ELEVENTH": GRADE_NAME_TO_CODE["ELEVENTH"],
    "TWELFTH": GRADE_NAME_TO_CODE["TWELFTH"],
    "FRESHMAN": GRADE_NAME_TO_CODE["NINTH"],
    "SOPHOMORE": GRADE_NAME_TO_CODE["TENTH"],
    "JUNIOR": GRADE_NAME_TO_CODE["ELEVENTH"],
    "SENIOR": GRADE_NAME_TO_CODE["TWELFTH"],
    "AE": GRADE_NAME_TO_CODE["ADULT_EDUCATION"],
    "ADULT": GRADE_NAME_TO_CODE["ADULT_EDUCATION"],
    "ADULTED": GRADE_NAME_TO_CODE["ADULT_EDUCATION"],
    "ADULTEDUCATION": GRADE_NAME_TO_CODE["ADULT_EDUCATION"],
}

GRADE_IGNORE_TOKENS = {"UG", "NONE", "NA"}

GRADE_PHRASE_SUBS: tuple[tuple[str, str], ...] = (
    ("EARLY CHILDHOOD", "EARLY EDUCATION"),
    ("EARLY CHILDHOOD EDUCATION", "EARLY EDUCATION"),
    ("EARLY EDUCATION", "EE"),
    ("PRE-KINDERGARTEN", "PREK"),
    ("PRE KINDERGARTEN", "PREK"),
    ("PRE-KG", "PREK"),
    ("PRE-K", "PREK"),
    ("PRE K", "PREK"),
    ("PREKINDERGARTEN", "PREK"),
    ("PREKINDER", "PREK"),
    ("PRE KINDER", "PREK"),
    ("PRE-SCHOOL", "PRESCHOOL"),
    ("PRE SCHOOL", "PRESCHOOL"),
    ("PRESCHOOL", "PRESCHOOL"),
    ("KINDER GARTEN", "KINDERGARTEN"),
    ("KINDERGARTEN", "KG"),
    ("ADULT EDUCATION", "AE"),
)

_ORDINAL_SUFFIX_RE = re.compile(r"(ST|ND|RD|TH)$")
_GRADE_SEGMENT_RE = re.compile(r"[A-Z0-9]+(?:\s*-\s*[A-Z0-9]+)?")


def grade_spec_to_tokens(spec: str) -> list[str]:
    text = spec.upper()
    for src, dst in GRADE_PHRASE_SUBS:
        text = text.replace(src, dst)
    text = text.replace("\N{EN DASH}", "-").replace("\N{EM DASH}", "-")
    text = text.replace("-", " ")
    text = text.replace("/", " ")
    text = text.replace("\\", " ")
    text = text.replace("&", " ")
    text = text.replace("+", " ")
    text = text.replace(" TO ", " ")
    text = text.replace(" THRU ", " ")
    text = text.replace(" THROUGH ", " ")
    text = text.replace("’", "")
    text = text.replace("'", "")
    text = text.replace(".", "")
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    tokens = [tok for tok in text.split() if tok]
    return tokens


def grade_spec_to_segments(spec: str) -> list[str]:
    text = spec.upper()
    for src, dst in GRADE_PHRASE_SUBS:
        text = text.replace(src, dst)
    text = text.replace("\N{EN DASH}", "-").replace("\N{EM DASH}", "-")
    text = text.replace("/", " ")
    text = text.replace("\\", " ")
    text = text.replace("&", " ")
    text = text.replace("+", " ")
    text = text.replace(" TO ", " ")
    text = text.replace(" THRU ", " ")
    text = text.replace(" THROUGH ", " ")
    text = text.replace("’", "")
    text = text.replace("'", "")
    text = text.replace(".", " ")
    text = re.sub(r"[^A-Z0-9\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return _GRADE_SEGMENT_RE.findall(text)


def grade_token_to_code(token: Any) -> Optional[int]:
    if token is None:
        return None
    if isinstance(token, (int, float)):
        try:
            idx = int(token)
        except Exception:
            return None
        return GRADE_NUMBER_TO_CODE.get(idx)
    if not isinstance(token, str):
        return None
    text = token.upper().strip()
    if not text:
        return None
    if text in GRADE_ALIAS_MAP:
        return GRADE_ALIAS_MAP[text]
    text = _ORDINAL_SUFFIX_RE.sub("", text)
    if text.isdigit():
        return GRADE_NUMBER_TO_CODE.get(int(text))
    return GRADE_NAME_TO_CODE.get(text)


def grade_segment_to_span(segment: str) -> tuple[Optional[int], Optional[int]] | None:
    if not segment:
        return None
    if "-" not in segment:
        code = grade_token_to_code(segment)
        return (code, code)
    left, right = (part.strip() for part in segment.split("-", 1))
    low = grade_token_to_code(left)
    high = grade_token_to_code(right)
    return normalize_grade_bounds(low, high)


def normalize_grade_bounds(
    low: Optional[int], high: Optional[int]
) -> tuple[Optional[int], Optional[int]]:
    if low is None and high is None:
        return (None, None)
    if low is None:
        return (high, high)
    if high is None:
        return (low, low)
    return (min(low, high), max(low, high))


def spans_to_bounds(
    spans: Iterable[tuple[Optional[int], Optional[int]]],
) -> tuple[Optional[int], Optional[int]]:
    low = None
    high = None
    for lo, hi in spans:
        if lo is not None:
            low = lo if low is None else min(low, lo)
        if hi is not None:
            high = hi if high is None else max(high, hi)
    return (low, high)


def grade_value_to_code(spec: Any) -> Optional[int]:
    if spec is None:
        return None
    if isinstance(spec, (int, float)):
        return grade_token_to_code(spec)
    if isinstance(spec, str):
        tokens = grade_spec_to_tokens(spec)
        codes = [grade_token_to_code(tok) for tok in tokens]
        codes = [c for c in codes if c is not None]
        if not codes:
            return None
        return min(codes)
    return grade_token_to_code(spec)


def coerce_grade_spans(
    spec: Any, high: Any | None = None
) -> list[tuple[Optional[int], Optional[int]]]:
    if isinstance(spec, str) and spec.upper() in GRADE_IGNORE_TOKENS:
        return []
    if high is not None:
        low_code = grade_value_to_code(spec)
        high_code = grade_value_to_code(high)
        if low_code is None and high_code is None:
            return []
        return [normalize_grade_bounds(low_code, high_code)]
    if hasattr(spec, "grade_range_code_spans"):
        spans = getattr(spec, "grade_range_code_spans")
        if spans:
            return list(spans)
    if hasattr(spec, "grade_range"):
        spans = coerce_grade_spans(getattr(spec, "grade_range"))
        if spans:
            return spans
    if isinstance(spec, dict):
        spans_payload = spec.get("grade_range_code_spans")
        if spans_payload is not None:
            spans = coerce_grade_spans(spans_payload)
            if spans:
                return spans
        low_val = None
        high_val = None
        for key in ("low", "min", "start", "from", "lower"):
            if key in spec:
                low_val = grade_value_to_code(spec[key])
                break
        for key in ("high", "max", "end", "to", "upper"):
            if key in spec:
                high_val = grade_value_to_code(spec[key])
                break
        if low_val is not None or high_val is not None:
            normalized = normalize_grade_bounds(low_val, high_val)
            if normalized != (None, None):
                return [normalized]
        if "grade_range" in spec:
            spans = coerce_grade_spans(spec["grade_range"])
            if spans:
                return spans
        return []
    if isinstance(spec, (list, tuple)):
        if not spec:
            return []
        if len(spec) == 2 and not any(
            isinstance(elem, (list, tuple, dict)) for elem in spec
        ):
            normalized = normalize_grade_bounds(
                grade_value_to_code(spec[0]), grade_value_to_code(spec[1])
            )
            return [normalized] if normalized != (None, None) else []
        spans: list[tuple[Optional[int], Optional[int]]] = []
        for elem in spec:
            spans.extend(coerce_grade_spans(elem))
        return spans
    if isinstance(spec, str):
        segments = grade_spec_to_segments(spec)
        spans: list[tuple[Optional[int], Optional[int]]] = []
        for segment in segments:
            span = grade_segment_to_span(segment)
            if span and span != (None, None):
                spans.append(span)
        if spans:
            return spans
        tokens = grade_spec_to_tokens(spec)
        codes = [grade_token_to_code(tok) for tok in tokens]
        codes = [c for c in codes if c is not None]
        if not codes:
            return []
        normalized = normalize_grade_bounds(min(codes), max(codes))
        return [normalized] if normalized != (None, None) else []
    code = grade_token_to_code(spec)
    if code is None:
        return []
    return [(code, code)]


def coerce_grade_bounds(
    spec: Any, high: Any | None = None
) -> tuple[Optional[int], Optional[int]]:
    spans = coerce_grade_spans(spec, high)
    if not spans:
        return (None, None)
    return spans_to_bounds(spans)
