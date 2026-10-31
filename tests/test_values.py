import json

import pandas as pd
import pytest

from teadata.teadata_config import load_config
from teadata.values import (
    EstimationPolicy,
    Measurement,
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
