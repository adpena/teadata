"""Canonical interpretation of source values and documented disclosure controls.

Rules belong to a versioned dataset contract. Negative values are never treated
as suppression merely because of their sign, and missing values are not zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import numbers
import re
from typing import Any, Iterable, Mapping


class ValueStatus(str, Enum):
    OBSERVED = "observed"
    SUPPRESSED = "suppressed"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"
    INVALID = "invalid"


@dataclass(frozen=True)
class TokenRule:
    status: ValueStatus
    reason: str
    lower: float | None = None
    upper: float | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TokenRule:
        if set(raw) - {"status", "reason", "lower", "upper"}:
            raise ValueError("Unknown value-token rule fields")
        rule = cls(
            ValueStatus(raw["status"]),
            str(raw["reason"]),
            raw.get("lower"),
            raw.get("upper"),
        )
        if rule.status == ValueStatus.OBSERVED or not rule.reason.strip():
            raise ValueError(
                "Special-token rules require a non-observed status and reason"
            )
        if any(
            value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not math.isfinite(value)
            )
            for value in (rule.lower, rule.upper)
        ):
            raise ValueError("Disclosure bounds must be finite numbers")
        if (
            rule.lower is not None
            and rule.upper is not None
            and rule.lower > rule.upper
        ):
            raise ValueError("Disclosure lower bound exceeds upper bound")
        if rule.status != ValueStatus.SUPPRESSED and any(
            v is not None for v in (rule.lower, rule.upper)
        ):
            raise ValueError("Only suppression rules can declare disclosure bounds")
        return rule

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "lower": self.lower,
            "upper": self.upper,
        }


@dataclass(frozen=True)
class ValuePolicy:
    policy_id: str
    tokens: Mapping[str, TokenRule] = field(default_factory=dict)
    reference_id: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ValuePolicy:
        if set(raw) - {"policy_id", "tokens", "reference_id"}:
            raise ValueError("Unknown value-policy fields")
        if not isinstance(raw.get("policy_id"), str) or not raw["policy_id"].strip():
            raise ValueError("A value policy requires policy_id")
        rules = raw.get("tokens", {})
        if not isinstance(rules, Mapping):
            raise ValueError("Value-policy tokens must be a mapping")
        normalized: dict[str, TokenRule] = {}
        for token, value in rules.items():
            key = _token(token)
            if key in normalized:
                raise ValueError(f"Equivalent source tokens declared twice: {token}")
            normalized[key] = TokenRule.from_dict(value)
        return cls(raw["policy_id"], normalized, raw.get("reference_id"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "reference_id": self.reference_id,
            "tokens": {k: v.to_dict() for k, v in self.tokens.items()},
        }


def _token(raw: Any) -> str:
    text = str(raw).strip()
    if re.fullmatch(r"[+-]?\d+(?:\.0+)?", text):
        return str(int(text.split(".")[0]))
    return text.casefold()


@dataclass(frozen=True)
class Measurement:
    value: int | float | str | None
    status: ValueStatus
    raw: Any = None
    reason: str | None = None
    policy_id: str | None = None
    lower: float | None = None
    upper: float | None = None
    unit: str | None = None

    @property
    def numeric(self) -> int | float | None:
        if (
            self.status == ValueStatus.OBSERVED
            and isinstance(self.value, numbers.Real)
            and not isinstance(self.value, bool)
            and math.isfinite(self.value)
        ):
            return (
                int(self.value)
                if isinstance(self.value, numbers.Integral)
                else float(self.value)
            )
        return None

    @property
    def is_suppressed(self) -> bool:
        return self.status == ValueStatus.SUPPRESSED

    def display(self, *, decimals: int | None = None) -> str:
        if self.status != ValueStatus.OBSERVED:
            return {
                ValueStatus.SUPPRESSED: "Suppressed",
                ValueStatus.MISSING: "Not available",
                ValueStatus.NOT_APPLICABLE: "Not applicable",
                ValueStatus.INVALID: "Invalid data",
            }[self.status]
        if not isinstance(self.value, numbers.Real) or isinstance(self.value, bool):
            return str(self.value)
        value = float(self.value)
        if decimals is None:
            decimals = 0 if value.is_integer() else 1
        text = f"{value:,.{decimals}f}"
        if self.unit == "percent":
            return text + "%"
        if self.unit == "currency":
            return "$" + text
        return text

    def to_dict(self) -> dict[str, Any]:
        raw = _json_scalar(self.raw)
        return {
            "value": self.value,
            "status": self.status.value,
            "raw": raw,
            "reason": self.reason,
            "policy_id": self.policy_id,
            "lower": self.lower,
            "upper": self.upper,
            "unit": self.unit,
        }


PLAIN_NUMERIC = ValuePolicy("numeric")


def _json_scalar(raw: Any) -> str | int | float | bool | None:
    if raw is None or isinstance(raw, (str, bool)):
        return raw
    if isinstance(raw, numbers.Integral):
        return int(raw)
    if isinstance(raw, numbers.Real):
        return float(raw) if math.isfinite(raw) else None
    return None if str(raw) == "<NA>" else str(raw)


def measure(
    raw: Any,
    policy: ValuePolicy = PLAIN_NUMERIC,
    *,
    unit: str | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> Measurement:
    """Interpret one value using its source's rules, preserving zero and negatives.

    Numeric source codes are interpreted before domain checks. A period has no
    universal meaning: it must appear in the source policy to be missing or
    suppressed. No point estimate is produced for a protected value.
    """
    if isinstance(raw, Measurement):
        return raw
    if (
        raw is None
        or (isinstance(raw, numbers.Real) and not math.isfinite(raw))
        or str(raw).strip() in {"", "<NA>"}
    ):
        return Measurement(
            None, ValueStatus.MISSING, raw, "absent", policy.policy_id, unit=unit
        )
    rule = policy.tokens.get(_token(raw))
    if rule is not None:
        return Measurement(
            None,
            rule.status,
            raw,
            rule.reason,
            policy.policy_id,
            rule.lower,
            rule.upper,
            unit,
        )
    if unit == "text":
        return Measurement(
            str(raw).strip(),
            ValueStatus.OBSERVED,
            raw,
            policy_id=policy.policy_id,
            unit=unit,
        )
    if isinstance(raw, bool):
        return Measurement(
            None,
            ValueStatus.INVALID,
            raw,
            "boolean_is_not_a_measure",
            policy.policy_id,
            unit=unit,
        )
    text = str(raw).strip()
    if unit == "currency":
        text = text.removeprefix("$")
        if text.startswith("(") and text.endswith(")"):
            text = "-" + text[1:-1]
    if unit == "percent":
        text = text.removesuffix("%")
    if not re.fullmatch(
        r"[+-]?(?:(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?", text
    ):
        return Measurement(
            None,
            ValueStatus.INVALID,
            raw,
            "unrecognized_numeric_token",
            policy.policy_id,
            unit=unit,
        )
    normalized = text.replace(",", "")
    number = (
        int(normalized) if re.fullmatch(r"[+-]?\d+", normalized) else float(normalized)
    )
    if (
        not math.isfinite(number)
        or (minimum is not None and number < minimum)
        or (maximum is not None and number > maximum)
    ):
        return Measurement(
            None,
            ValueStatus.INVALID,
            raw,
            "outside_metric_domain",
            policy.policy_id,
            unit=unit,
        )
    return Measurement(
        int(number) if isinstance(number, float) and number.is_integer() else number,
        ValueStatus.OBSERVED,
        raw,
        policy_id=policy.policy_id,
        unit=unit,
    )


@dataclass(frozen=True)
class Aggregate:
    observed_sum: float
    observed_count: int
    suppressed_count: int
    missing_count: int
    invalid_count: int
    lower: float | None
    upper: float | None

    @property
    def complete(self) -> bool:
        return self.suppressed_count + self.missing_count + self.invalid_count == 0

    @property
    def exact_total(self) -> float | None:
        return self.observed_sum if self.complete else None

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return {
            **asdict(self),
            "complete": self.complete,
            "exact_total": self.exact_total,
        }


def aggregate(values: Iterable[Measurement]) -> Aggregate:
    """Aggregate observed values while carrying suppression and missingness.

    Bounds are returned only if every contributing non-observed value has a
    documented bound. A partial observed sum is never labeled an exact total.
    """
    items = list(values)
    observed = [float(value.numeric) for value in items if value.numeric is not None]
    protected = [value for value in items if value.is_suppressed]
    missing = sum(
        value.status in {ValueStatus.MISSING, ValueStatus.NOT_APPLICABLE}
        for value in items
    )
    invalid = sum(
        value.status == ValueStatus.INVALID
        or (value.status == ValueStatus.OBSERVED and value.numeric is None)
        for value in items
    )
    total = math.fsum(observed)
    lower = (
        total + math.fsum(value.lower for value in protected if value.lower is not None)
        if not missing
        and not invalid
        and all(value.lower is not None for value in protected)
        else None
    )
    upper = (
        total + math.fsum(value.upper for value in protected if value.upper is not None)
        if not missing
        and not invalid
        and all(value.upper is not None for value in protected)
        else None
    )
    return Aggregate(
        total, len(observed), len(protected), missing, invalid, lower, upper
    )


@dataclass(frozen=True)
class EstimationPolicy:
    """An explicit analytical assumption, separate from disclosure controls."""

    model_id: str
    value_per_suppressed: float
    grain: str
    description: str
    reference_id: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> EstimationPolicy:
        if set(raw) != set(cls.__dataclass_fields__):
            raise ValueError(
                "Estimation policies require model_id, value_per_suppressed, grain, description, and reference_id"
            )
        if any(
            not isinstance(raw[key], str) or not raw[key].strip()
            for key in ("model_id", "grain", "description", "reference_id")
        ):
            raise ValueError(
                "Estimation policy identifiers and methodology must be nonempty"
            )
        value = raw["value_per_suppressed"]
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(
                "Imputation assumptions must be finite nonnegative numbers"
            )
        return cls(**raw)

    def estimate(self, summary: Aggregate) -> dict[str, Any]:
        """Preserve the observed sum and identify every modeled contribution.

        Missing and invalid counts cannot be silently imputed as suppressed.
        This produces a scenario estimate, not a disclosure bound or interval.
        """
        available = not summary.missing_count and not summary.invalid_count
        contribution = (
            summary.suppressed_count * self.value_per_suppressed if available else None
        )
        return {
            "status": "estimated" if available else "unavailable",
            "model_id": self.model_id,
            "value": summary.observed_sum + contribution
            if contribution is not None
            else None,
            "observed_sum": summary.observed_sum,
            "suppressed_count": summary.suppressed_count,
            "missing_count": summary.missing_count,
            "invalid_count": summary.invalid_count,
            "estimated_contribution": contribution,
            "value_per_suppressed": self.value_per_suppressed,
            "grain": self.grain,
            "assumption": self.description,
            "reference_id": self.reference_id,
        }
