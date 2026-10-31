"""One observation summary for normalized campus-pair transfer facts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .values import Aggregate, Measurement, ValuePolicy, ValueStatus, aggregate, measure


def normalized_transfer_measurement(
    count: Any, masked: bool, *, policy: ValuePolicy | None = None
) -> Measurement:
    """Interpret already-normalized edge state without re-decoding a sentinel."""
    from .manifest import ContractError

    if type(masked) is not bool:
        raise ContractError("Normalized transfer masking flag must be Boolean")
    if masked:
        if measure(count, unit="count", minimum=0).status != ValueStatus.MISSING:
            raise ContractError(
                "Suppressed transfer edge must have a missing numeric count"
            )
        return Measurement(
            None,
            ValueStatus.SUPPRESSED,
            reason="normalized_source_suppression",
            policy_id=policy.policy_id if policy else None,
            unit="count",
        )
    # Raw source token decoding belongs to ingestion. An unmasked normalized
    # edge cannot acquire suppression merely by matching a policy token again.
    value = measure(count, unit="count", minimum=0)
    if value.status == ValueStatus.INVALID:
        raise ContractError("Normalized transfer count is invalid")
    return replace(value, policy_id=policy.policy_id if policy else None)


@dataclass(frozen=True)
class CharterTransferObservations:
    aggregate: Aggregate
    destination_count: int
    unresolved: tuple[dict[str, Any], ...]


def transfer_measurement(
    repo: Any,
    source: Any,
    destination: Any,
    count: Any,
    masked: bool,
    *,
    policy: ValuePolicy | None = None,
) -> Measurement:
    """Recover the ingested measurement, checking its normalized edge projection."""
    from .manifest import ContractError

    measurements = getattr(repo, "_transfer_measurements", {})
    value = measurements.get((source.id, destination.id)) if measurements else None
    if value is None:
        return normalized_transfer_measurement(count, masked, policy=policy)
    if (
        type(masked) is not bool
        or value.is_suppressed != masked
        or value.numeric != count
    ):
        raise ContractError("Transfer measurement differs from its normalized edge")
    return value


def charter_transfer_observations(
    campus: Any, *, policy: ValuePolicy | None = None
) -> CharterTransferObservations:
    """Summarize observations only; estimates belong to the configured model."""
    from .manifest import ContractError

    repo = getattr(campus, "_repo", None)
    catalog = getattr(repo, "catalog", None)
    if policy is None and catalog is not None:
        policy = catalog.policy("campus_transfer_reports")
    destinations = {}
    for destination, count, masked in (
        repo.transfers_out(campus) if repo is not None else ()
    ):
        if destination is None or not destination.is_charter or destination.is_private:
            continue
        if destination.id in destinations:
            raise ContractError("Duplicate normalized campus transfer pair")
        value = transfer_measurement(
            repo, campus, destination, count, masked, policy=policy
        )
        destinations[destination.id] = value
    unresolved = tuple(
        dict(row)
        for row in getattr(repo, "_xfers_unresolved", ())
        if row["source_campus"] == campus.campus_number
    )
    summary = aggregate(destinations.values())
    summary = replace(summary, missing_count=summary.missing_count + len(unresolved))
    return CharterTransferObservations(summary, len(destinations), unresolved)
