from __future__ import annotations

import random
import string

from teadata.teadata_config import (
    canonical_campus_number,
    canonical_district_number,
    normalize_campus_number_value,
    normalize_district_number_value,
)
from teadata.grades import coerce_grade_spans


def _random_digits(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(length))


def _random_slug(rng: random.Random, max_digits: int) -> str:
    digits = _random_digits(rng, rng.randint(1, max_digits))
    if rng.random() < 0.3:
        digits = f"'{digits}"
    if rng.random() < 0.3:
        digits = f"{digits}-{_random_digits(rng, 2)}"
    return digits


def _random_grade_spec(rng: random.Random) -> str:
    tokens = [rng.choice(["PK", "K", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"])]
    if rng.random() < 0.7:
        tokens.append(rng.choice(["K", "02", "05", "08", "12"]))
    return "-".join(tokens)


def main() -> None:
    rng = random.Random(2187)
    for _ in range(500):
        campus_raw = _random_slug(rng, 9)
        district_raw = _random_slug(rng, 6)
        campus_norm = normalize_campus_number_value(campus_raw)
        district_norm = normalize_district_number_value(district_raw)
        if campus_norm is not None:
            assert campus_norm.isdigit()
            assert len(campus_norm) == 9
            campus_canon = canonical_campus_number(campus_raw)
            assert campus_canon == f"'{campus_norm}"
        if district_norm is not None:
            assert district_norm.isdigit()
            assert len(district_norm) == 6
            district_canon = canonical_district_number(district_raw)
            assert district_canon == f"'{district_norm}"

        grade_spec = _random_grade_spec(rng)
        spans = coerce_grade_spans(grade_spec)
        for low, high in spans:
            if low is not None and high is not None:
                assert low <= high


if __name__ == "__main__":
    main()
