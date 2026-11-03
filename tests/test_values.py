import json

import pandas as pd
import pytest

from teadata.teadata_config import load_config
from teadata.values import (
    EstimationPolicy,
    Measurement,
    MeasurementUnit,
    ValuePolicy,
    ValueStatus,
    aggregate,
    measure,
)


@pytest.fixture
def policies():
    return load_config("teadata/teadata_sources.yaml").value_policies


@pytest.mark.parametrize(
    "token,status",
    [
        (-1, "suppressed"),
        ("-1.0", "suppressed"),
        (-3, "suppressed"),
        (-2, "invalid"),
        (".", "missing"),
        ("•", "missing"),
        ("n/a", "not_applicable"),
        (0, "observed"),
    ],
)
def test_tapr_documented_codes(policies, token, status):
    assert measure(token, policies["tapr_profile"]).status.value == status


def test_negative_financial_adjustment_is_observed(policies):
    value = measure("-1", policies["numeric"], unit="currency")
    assert value.numeric == -1 and value.status == ValueStatus.OBSERVED


def test_fractional_fte_is_observed_without_relaxing_integral_count():
    fte = measure("12.5", unit=MeasurementUnit.FTE, minimum=0)
    count = measure("12.5", unit=MeasurementUnit.COUNT, minimum=0)
    assert fte.status == ValueStatus.OBSERVED and fte.numeric == 12.5
    assert fte.raw == "12.5" and fte.unit == "fte"
    assert json.loads(json.dumps(fte.to_dict()))["unit"] == "fte"
    assert count.status == ValueStatus.INVALID and count.numeric is None
    assert count.reason == "outside_metric_domain" and count.raw == "12.5"
    assert measure("12", unit=MeasurementUnit.COUNT).numeric == 12


def test_declared_unit_enum_does_not_restrict_scalar_custom_units():
    scalar = measure("1.25", unit="caller_defined_unit")
    assert scalar.status == ValueStatus.OBSERVED and scalar.numeric == 1.25
    assert scalar.unit == "caller_defined_unit"


def test_period_is_not_universally_suppression(policies):
    assert measure(".").status == ValueStatus.INVALID
    assert measure(".", policies["tapr_profile"]).status == ValueStatus.MISSING


def test_unknown_negative_transfer_token_is_invalid(policies):
    assert (
        measure(-2, policies["campus_transfers"], minimum=0).status
        == ValueStatus.INVALID
    )
    assert measure(-999, policies["campus_transfers"]).is_suppressed


def test_transfer_masks_have_no_fabricated_bounds(policies):
    result = aggregate([measure(12), measure(-999, policies["campus_transfers"])])
    assert result.observed_sum == 12 and result.exact_total is None
    assert result.lower is None and result.upper is None


def test_estimate_preserves_observed_and_assumed_contributions(policies):
    cfg = load_config("teadata/teadata_sources.yaml")
    model = cfg.resolve_dataset("campus_transfer_reports").metadata.estimation
    summary = aggregate(
        [
            measure(12),
            measure(-999, policies["campus_transfers"]),
            measure(-999, policies["campus_transfers"]),
        ]
    )
    result = model.estimate(summary)
    assert result["status"] == "estimated"
    assert result["observed_sum"] == 12
    assert result["estimated_contribution"] == 10
    assert result["value"] == 22
    assert result["suppressed_count"] == 2
    assert summary.exact_total is None
    assert "not a documented disclosure bound" in result["assumption"]


def test_missing_and_invalid_are_not_imputed(policies):
    model = (
        load_config("teadata/teadata_sources.yaml")
        .resolve_dataset("campus_transfer_reports")
        .metadata.estimation
    )
    for raw in (None, "garbage"):
        result = model.estimate(aggregate([measure(12), measure(raw)]))
        assert result["value"] is None and result["status"] == "unavailable"


@pytest.mark.parametrize("raw", [pd.NA, float("nan"), float("inf"), True, 2**53 + 1])
def test_measurements_export_strict_json(raw):
    value = measure(raw)
    json.dumps(value.to_dict(), allow_nan=False)
    if raw is not pd.NA and raw == 2**53 + 1:
        assert value.numeric == raw


def test_text_cannot_produce_an_exact_numeric_aggregate():
    result = aggregate([Measurement("A", ValueStatus.OBSERVED)])
    assert result.invalid_count == 1 and result.exact_total is None


def test_equivalent_sentinel_rules_are_rejected():
    with pytest.raises(ValueError, match="Equivalent"):
        ValuePolicy.from_dict(
            {
                "policy_id": "bad",
                "tokens": {
                    "-1": {"status": "suppressed", "reason": "one"},
                    "-1.0": {"status": "missing", "reason": "two"},
                },
            }
        )


def test_imputation_assumptions_require_methodology():
    with pytest.raises(ValueError):
        EstimationPolicy.from_dict({"model_id": "arbitrary", "value_per_suppressed": 5})


def test_negative_financial_percentage_is_not_suppression_or_invalid(catalog):
    from types import SimpleNamespace

    entity = SimpleNamespace(
        campus_number="123456789", meta={"instruction_af_perc": -0.5}
    )
    value = catalog.measurement(entity, "financials.instruction_af_perc")
    assert value.status == ValueStatus.OBSERVED
    assert value.numeric == -0.5


def test_unresolved_transfer_rows_are_retained_and_do_not_get_imputed(catalog):
    from types import SimpleNamespace

    origin = SimpleNamespace(campus_number="'123456789")
    observed = SimpleNamespace(id=1, is_charter=True, is_private=False)
    suppressed = SimpleNamespace(id=2, is_charter=True, is_private=False)
    unresolved = {
        "source_campus": origin.campus_number,
        "destination_campus": "'220820102",
        "measurement": {"status": "suppressed"},
    }
    origin._repo = SimpleNamespace(
        transfers_out=lambda campus: [(observed, 12, False), (suppressed, None, True)],
        _xfers_unresolved=[unresolved],
    )
    result = catalog.charter_transfer_summary(origin)
    assert result["total_unmasked_charter_transfers_out"] == 12
    assert result["charter_transfer_estimate"]["status"] == "unavailable"
    assert result["total_estimated_charter_transfers_out"] is None
    assert result["unresolved_transfers"] == [unresolved]
    origin._repo._xfers_unresolved = []
    assert (
        catalog.charter_transfer_summary(origin)[
            "total_estimated_charter_transfers_out"
        ]
        == 17
    )


@pytest.mark.parametrize("raw,expected", [
    ("9007199254740993.0", 9007199254740993),
    ("9007199254740993e0", 9007199254740993),
    ("9.007199254740993e15", 9007199254740993),
    ("-9007199254740993.000", -9007199254740993),
    ("9,007,199,254,740,993.0", 9007199254740993),
])
@pytest.mark.parametrize("unit", [None, "count"])
def test_integral_decimal_and_exponent_tokens_remain_exact_without_float_rounding(raw, expected, unit):
    value = measure(raw, unit=unit)
    assert value.status == ValueStatus.OBSERVED and type(value.value) is int
    assert value.value == value.numeric == expected and value.raw == raw
    assert json.loads(json.dumps(value.to_dict()))["value"] == expected


@pytest.mark.parametrize("raw", ["9007199254740992.5", "1.0000000000000000000001", "1e-400"])
def test_exact_fractional_count_cannot_become_observed_by_float_rounding(raw):
    value = measure(raw, unit="count")
    assert value.status == ValueStatus.INVALID and value.numeric is None
    assert value.reason == "outside_metric_domain" and value.raw == raw


@pytest.mark.parametrize("raw", ["1e1000000", "-1e1000000", "1e9999999999999999999999999999"])
def test_huge_exponent_is_rejected_before_expanding_an_integer(raw):
    value = measure(raw, unit="count")
    assert value.status == ValueStatus.INVALID and value.value is None
    assert value.reason == "outside_metric_domain" and value.raw == raw


@pytest.mark.parametrize("raw,expected", [(0, 0), (0.0, 0), (12.5, 12.5), (-12.5, -12.5), (2**53 + 1, 2**53 + 1)])
def test_integral_token_fix_preserves_existing_real_and_integer_values(raw, expected):
    value = measure(raw)
    assert value.status == ValueStatus.OBSERVED and value.numeric == expected and value.raw == raw


def test_exact_integer_parsing_preserves_policy_precedence_and_literal_thresholds():
    policy = ValuePolicy.from_dict({"policy_id": "synthetic-precision-policy", "tokens": {
        "9007199254740993.0": {"status": "suppressed", "reason": "synthetic_declared_token"}}})
    protected = measure("9007199254740993.0", policy, unit="count")
    assert protected.status == ValueStatus.SUPPRESSED and protected.numeric is None
    assert protected.raw == "9007199254740993.0"
    for token in ("<10", "<2790"):
        threshold = measure(token, unit="count")
        assert threshold.status == ValueStatus.INVALID and threshold.raw == token and threshold.numeric is None
    threshold_policy = ValuePolicy.from_dict({"policy_id": "synthetic-threshold-policy", "tokens": {
        "<2790": {"status": "suppressed", "reason": "synthetic_declared_complementary_mask"}}})
    masked = measure("<2790", threshold_policy, unit="count")
    assert masked.status == ValueStatus.SUPPRESSED and masked.raw == "<2790" and masked.numeric is None
    assert masked.lower is None and masked.upper is None
    assert measure("-12.5", unit="currency").numeric == -12.5
    assert measure("12.5", unit="fte").numeric == 12.5
    assert measure("0e1000000", unit="count").numeric == 0
