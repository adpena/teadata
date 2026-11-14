"""Statewide comparisons retain exact native decimal observations."""
from decimal import Decimal, localcontext
from typing import cast

import pytest

from teadata.dataset_metadata import ArtifactRef
from teadata.statewide import _Projection, _difference


def test_native_projection_keeps_decimal_precision_and_missingness():
    projection = _Projection(cast(ArtifactRef, None), (), {"value": ("raw", "status")}, None)
    number = Decimal("543.68900000000008001")
    assert projection.value({"raw": str(number), "status": "observed", "value": number}, "value", "ratio") == (
        str(number), "observed", number)
    for status in ("suppressed", "missing", "not_applicable"):
        assert projection.value({"raw": "*", "status": status, "value": None}, "value", "ratio") == (
            "*", status, None)


@pytest.mark.parametrize("left,right,expected", [
    (Decimal("543.68900000000008001"), Decimal("543.68900000000008000"), Decimal("0.00000000000000001")),
    (Decimal("0.10000000000000001"), 0.1, Decimal("0.00000000000000001")),
    (Decimal("-0.01"), 0, Decimal("-0.01")),
    (3, 1, 2),
    (18.4, 13.8, 4.6),
])
def test_difference_preserves_decimals_and_existing_binary_numeric_types(left, right, expected):
    actual = _difference(left, right)
    assert actual == expected
    assert type(actual) is type(expected)


def test_decimal_difference_does_not_inherit_a_lossy_caller_context():
    left = Decimal("123456789012345678901.12345678901234567")
    right = Decimal("0.00000000000000001")
    with localcontext() as context:
        context.prec = 4
        assert _difference(left, right) == Decimal("123456789012345678901.12345678901234566")


def test_finite_decimal_measurement_does_not_overflow_through_float():
    from teadata.values import Measurement, ValueStatus

    number = Decimal("1e1000")
    measured = Measurement(raw="1e1000", value=number, status=ValueStatus.OBSERVED)
    assert measured.numeric is number
    projection = _Projection(cast(ArtifactRef, None), (), {"value": ("raw", "status")}, None)
    assert projection.value({"raw": "1e1000", "status": "observed", "value": number}, "value", "ratio")[2] is number
