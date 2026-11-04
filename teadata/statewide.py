"""Compare reviewed local TAPR rates against their publisher statewide observation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from .dataset_metadata import STATEWIDE_SINGLETON_GRAIN, DataPeriod, DatasetMetadata, EditionKey
from .entities import Campus, District
from .identifiers import canonical_source_identifiers
from .manifest import ContractError, canonical_json
from .teadata_config import load_config
from .values import TokenRule, ValuePolicy, ValueStatus, measure

if TYPE_CHECKING:
    from .release import DataRelease

FAMILY = "college_readiness_components"
EDITION = 2025
_GRAD2_RATE_POLICY = ValuePolicy(
    "tapr_2025_grad2_reviewed_rates",
    {"-1": TokenRule(ValueStatus.SUPPRESSED, "TEA masked the postsecondary readiness rate")},
    reference_id="masking",
)


@lru_cache(maxsize=3)
def _reviewed_grad2(scope: str) -> tuple[str, str, DatasetMetadata]:
    config = load_config()
    dataset = f"{scope}_tapr_{FAMILY}"
    request, = config.tapr_source_requests(dataset, EDITION)
    metadata = config.resolve_dataset(dataset, EDITION, exact=True, reader_profile="source").metadata
    revision = EditionKey.source_revision((request.mapping["input_sha256"],))
    mapping = sha256(canonical_json(request.mapping).encode()).hexdigest()
    return revision, mapping, metadata


@dataclass(frozen=True)
class StatewideComparison:
    identifier: str | None
    scope: str
    metric: str
    period: DataPeriod
    local_edition: EditionKey
    statewide_edition: EditionKey
    local_raw: str | None
    local_status: str
    local_percent: float | None
    statewide_raw: str
    statewide_percent: float
    difference_pp: float | None


def compare_statewide(items: Sequence[object], *, release: DataRelease,
                      local: EditionKey, statewide: EditionKey, metric: str) -> list[StatewideComparison]:
    """Read just the matched GRAD2 rate columns from an explicit native release."""
    if (local.scope not in {"district", "campus"}
            or local.dataset != f"{local.scope}_tapr_{FAMILY}"
            or statewide.dataset != f"state_tapr_{FAMILY}"
            or statewide.scope != "state" or statewide.grain != STATEWIDE_SINGLETON_GRAIN
            or local.table != FAMILY or statewide.table != FAMILY
            or local.edition != str(EDITION) or statewide.edition != local.edition):
        raise ContractError("Statewide comparison requires matched 2025 TAPR GRAD2 editions")
    entity_type = Campus if local.scope == "campus" else District
    if any(not isinstance(item, entity_type) for item in items):
        raise ContractError("Query entities must match the selected local edition scope")

    key_column = f"{local.scope}_number"
    if local.grain != f"one row per {key_column}":
        raise ContractError("Local edition must contain one row per public entity")
    catalog = release.catalog
    local_metadata = catalog.metadata(local.dataset, edition=local)
    state_metadata = catalog.metadata(statewide.dataset, edition=statewide)
    local_revision, local_mapping, local_review = _reviewed_grad2(local.scope)
    state_revision, state_mapping, state_review = _reviewed_grad2(statewide.scope)
    if (local.revision_sha256, statewide.revision_sha256) != (local_revision, state_revision):
        raise ContractError("Statewide comparison requires the reviewed publisher revisions")
    local_rate = local_metadata.metrics.get(metric)
    state_rate = state_metadata.metrics.get(metric)
    if (local_rate is None or state_rate is None or local_rate.unit != "percent" or state_rate.unit != "percent"
            or local_rate != local_review.metrics.get(metric) or state_rate != state_review.metrics.get(metric)):
        raise ContractError("Statewide comparison requires the same reviewed percentage metric in both editions")
    period = local_rate.period or local_metadata.period
    if (period.kind == "unknown" or period != (state_rate.period or state_metadata.period)
            or period != (local_rate.period or local_review.period)
            or period != (state_rate.period or state_review.period)):
        raise ContractError("Statewide and local rates must cover the same known reference period")
    plans = catalog.resolve_artifacts(
        [(local, local_rate.field), (statewide, state_rate.field)], period=period
    )
    if len(plans) != 2:
        raise ContractError("Statewide comparison requires one local and one statewide leaf")
    local_plan, state_plan = plans
    if (local_metadata.value_policy is not None or state_metadata.value_policy is not None
            or local_plan["artifact"].contract["mapping_sha256"] != local_mapping
            or state_plan["artifact"].contract["mapping_sha256"] != state_mapping
            or local_plan["columns"] != (key_column, local_rate.field)
            or state_plan["columns"] != (state_rate.field,)):
        raise ContractError("Statewide comparison requires the reviewed raw GRAD2 leaf grain")
    masking = next((ref for ref in local_review.references if ref.reference_id == _GRAD2_RATE_POLICY.reference_id), None)
    if masking is None or not masking.provenance_ids or masking not in local_metadata.references:
        raise ContractError("Statewide comparison requires the retained TAPR rate masking methodology")
    if not items:
        return []

    import pyarrow as pa
    import pyarrow.compute as arrow_compute

    pc: Any = arrow_compute
    identifiers = [
        item.lookup_key if isinstance(item, Campus)
        else item.district_number_canon if isinstance(item, District) else None
        for item in items
    ]
    applicable = [not (isinstance(item, Campus) and item.is_private) for item in items]
    state_rows = release.analytical_dataset(state_plan["artifact"].identity).head(2, columns=[state_rate.field])
    if state_rows.num_rows != 1:
        raise ContractError("Statewide comparison requires exactly one publisher observation")
    state_raw = state_rows[state_rate.field][0].as_py()
    if not isinstance(state_raw, str):
        raise ContractError("Statewide rate must preserve its publisher token")
    state_value = measure(state_raw, unit="percent", minimum=0, maximum=100)
    if state_value.status != ValueStatus.OBSERVED or not isinstance(state_value.value, (int, float)):
        raise ContractError("The reviewed statewide percentage is unavailable")

    source_rows = release.analytical_dataset(local_plan["artifact"].identity).to_table(
        columns=[key_column, local_rate.field]
    )
    rate_type = source_rows[local_rate.field].type
    if not (pa.types.is_string(rate_type) or pa.types.is_large_string(rate_type)):
        raise ContractError("Local rate must preserve its publisher tokens")
    width = 9 if local.scope == "campus" else 6
    keys = canonical_source_identifiers(source_rows[key_column], width=width)
    wanted = pa.array(["'" + key for key in {
        identifier for identifier, applies in zip(identifiers, applicable, strict=True) if applies and identifier
    }], type=pa.string())
    selected = pc.is_in(keys, value_set=wanted)
    selected_keys = pc.filter(keys, selected).to_pylist()
    if len(selected_keys) != len(set(selected_keys)):
        raise ContractError("Local source contains duplicate selected identifiers")
    raw_by_key = dict(zip(selected_keys, pc.filter(source_rows[local_rate.field], selected).to_pylist(), strict=True))

    rows = []
    for identifier, applies in zip(identifiers, applicable, strict=True):
        raw = raw_by_key.get("'" + identifier) if applies and identifier else None
        value = measure(raw, _GRAD2_RATE_POLICY, unit="percent", minimum=0, maximum=100)
        local_percent = value.value if isinstance(value.value, (int, float)) else None
        status = "not_applicable" if not applies else "unavailable" if raw is None else value.status.value
        if value.status == ValueStatus.INVALID:
            status = "uninterpreted"
        difference = (float(Decimal(str(local_percent)) - Decimal(str(state_value.value)))
                      if local_percent is not None else None)
        rows.append(StatewideComparison(identifier or None, local.scope, metric, period, local, statewide,
                    raw, status, local_percent, state_raw, state_value.value, difference))
    return rows
