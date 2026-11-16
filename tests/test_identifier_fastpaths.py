"""Golden scalar IDs retain precision without importing dataframe machinery."""

import builtins
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from teadata.identifiers import canonical_source_identifier


def test_standard_scalar_ids_do_not_import_pandas(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "pandas" or name.startswith("pandas."):
            raise AssertionError("Normalizing a standard scalar must not import pandas")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    for value in (12345, 12345.0, "12345", "'012345", "12345.000"):
        assert canonical_source_identifier(value, width=6) == "'012345"
    assert canonical_source_identifier(1902001, width=9) == "'001902001"
    for value in (None, "", "  ", float("nan")):
        assert canonical_source_identifier(value, width=6) is None
    for value in (True, False, 0, -1, 1234567, 123.5, float("inf"), "12 345", "１２３"):
        with pytest.raises(ValueError):
            canonical_source_identifier(value, width=6)


@pytest.mark.parametrize("value", [np.int32(12345), np.int64(12345), np.uint64(12345), np.float32(12345),
                                  np.float64(12345), np.timedelta64(12345, "ns"), Fraction(12345, 1)])
def test_extended_numeric_ids_keep_the_same_canonical_identity(value):
    assert canonical_source_identifier(value, width=6) == "'012345"


@pytest.mark.parametrize("value", [pd.NA, pd.NaT, np.datetime64("NaT", "ns"), np.timedelta64("NaT", "ns"),
                                  Decimal("NaN"), np.float32("nan")])
def test_nullable_scalars_keep_missing_separate_from_zero(value):
    assert canonical_source_identifier(value, width=6) is None


@pytest.mark.parametrize("value", [np.bool_(True), Decimal("12345"), [12345], {"id": 12345}, complex(12345, 0)])
def test_nonmissing_unsupported_scalars_still_fail(value):
    with pytest.raises(ValueError):
        canonical_source_identifier(value, width=6)
