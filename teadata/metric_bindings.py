"""Validated metric interpretation stored with each data release."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class MetricBinding:
    dataset: str
    field: str
    field_mode: str = "literal"
    attribute: str | None = None
    unit: str | None = None
    scopes: tuple[str, ...] = ("campus", "district")
    population: str = "all"
    minimum: float | None = None
    maximum: float | None = None
    unit_suffixes: Mapping[str, str] = field(default_factory=dict)
    unit_bounds: Mapping[str, Mapping[str, float | None]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> MetricBinding:
        if not isinstance(raw, Mapping) or set(raw) - set(cls.__dataclass_fields__):
            raise ValueError("Unknown or invalid metric binding fields")
        values = dict(raw)
        for name in ("dataset", "field"):
            if not isinstance(values.get(name), str) or not values[name].strip():
                raise ValueError("Metric bindings require dataset and field")
        if values.get("field_mode", "literal") not in {"literal", "metric"}:
            raise ValueError("Unknown metric binding field mode")
        if values.get("population", "all") not in {"all", "public", "private"}:
            raise ValueError("Unknown metric population")
        scopes = values.get("scopes", ("campus", "district"))
        if not isinstance(scopes, (list, tuple)) or not scopes or set(scopes) - {"campus", "district"}:
            raise ValueError("Unknown metric binding scopes")
        values["scopes"] = tuple(scopes)
        units = {"count", "percent", "currency", "text", "ratio", "years"}
        if values.get("unit") is not None and values["unit"] not in units:
            raise ValueError("Unknown metric binding unit")
        suffixes = values.get("unit_suffixes", {})
        if not isinstance(suffixes, Mapping) or any(
            not isinstance(key, str) or not key or value not in units for key, value in suffixes.items()
        ):
            raise ValueError("Invalid metric unit suffix rules")
        for name in ("minimum", "maximum"):
            value = values.get(name)
            if value is not None and (type(value) not in {int, float} or not math.isfinite(value)):
                raise ValueError("Metric bounds must be finite numbers")
        bounds = values.get("unit_bounds", {})
        if not isinstance(bounds, Mapping) or any(key not in units for key in bounds):
            raise ValueError("Invalid metric unit bounds")
        for domain in bounds.values():
            if not isinstance(domain, Mapping) or set(domain) - {"minimum", "maximum"}:
                raise ValueError("Invalid metric unit bounds")
            for limit in domain.values():
                if limit is not None and (type(limit) not in {int, float} or not math.isfinite(limit)):
                    raise ValueError("Metric bounds must be finite numbers")
            if domain.get("minimum") is not None and domain.get("maximum") is not None and domain["minimum"] > domain["maximum"]:
                raise ValueError("Metric unit bounds are reversed")
        if values.get("minimum") is not None and values.get("maximum") is not None and values["minimum"] > values["maximum"]:
            raise ValueError("Metric bounds are reversed")
        for name, allowed in (("dataset", {"{scope}", "{private_dataset}"}), ("field", {"{metric}"})):
            text = values[name]
            for token in allowed:
                text = text.replace(token, "")
            if "{" in text or "}" in text:
                raise ValueError("Unknown metric binding placeholder")
        return cls(**values)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["scopes"] = list(self.scopes)
        return result

    def bounds(self, unit: str | None) -> tuple[float | None, float | None]:
        domain = self.unit_bounds.get(unit, {}) if unit is not None else {}
        return domain.get("minimum", self.minimum), domain.get("maximum", self.maximum)


def parse_bindings(raw: Mapping[str, Any] | None) -> dict[str, MetricBinding | str]:
    if not isinstance(raw or {}, Mapping):
        raise ValueError("metric_bindings must be a mapping")
    result = {}
    for key, value in (raw or {}).items():
        if not isinstance(key, str) or not key or ("*" in key and not key.endswith(".*")):
            raise ValueError("Invalid canonical metric identifier")
        result[key] = value if isinstance(value, str) else MetricBinding.from_dict(value)
    for key in result:
        seen = set()
        target = key
        while isinstance(result.get(target), str):
            if target in seen:
                raise ValueError("Cyclic canonical metric alias")
            seen.add(target)
            target = result[target]
        if target not in result:
            raise ValueError("Canonical metric alias has no binding")
    return result


def bindings_document(bindings: Mapping[str, MetricBinding | str]) -> dict:
    return {key: value if isinstance(value, str) else value.to_dict() for key, value in bindings.items()}
