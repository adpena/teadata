"""Compare compatible publisher statistics using exact local and statewide editions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from hashlib import sha256
from math import isfinite
from typing import TYPE_CHECKING, Any

from .column_schema import crosswalk_dataset, parse_column_schema, table_input_sha256s
from .dataset_metadata import STATEWIDE_SINGLETON_GRAIN, ArtifactRef, DataPeriod, DatasetMetadata, EditionKey
from .entities import Campus, District
from .identifiers import canonical_source_identifiers
from .manifest import ContractError, canonical_json
from .peims_courses import DATASET as COURSES, course_key_columns
from .peims_leaves import PRIMARY_DISABILITY, SCOPES_BY_DATASET, _key_columns
from .teadata_config import load_config
from .values import MeasurementUnit, TokenRule, ValuePolicy, ValueStatus, measure

if TYPE_CHECKING:
    from .release import DataRelease

_TAPR_RATE_FAMILIES = {"college_career_military_readiness", "college_readiness_components",
                       "four_year_longitudinal_rate"}
EDITION = 2025
_TAPR_RATE_POLICY = ValuePolicy(
    "tapr_2025_reviewed_rates",
    {"-1": TokenRule(ValueStatus.SUPPRESSED, "TEA masked the rate")},
    reference_id="masking",
)
_NUMERIC_UNITS = frozenset(MeasurementUnit) - {MeasurementUnit.TEXT}
_VALUE_STATUSES = frozenset(item.value for item in ValueStatus)
Number = int | float


@lru_cache(maxsize=9)
def _reviewed_tapr(scope: str, family: str) -> tuple[str, str, DatasetMetadata]:
    config = load_config()
    dataset = f"{scope}_tapr_{family}"
    request, = config.tapr_source_requests(dataset, EDITION)
    metadata = config.resolve_dataset(dataset, EDITION, exact=True, reader_profile="source").metadata
    revision = EditionKey.source_revision((request.mapping["input_sha256"],))
    mapping = sha256(canonical_json(request.mapping).encode()).hexdigest()
    return revision, mapping, metadata


@lru_cache(maxsize=32)
def _reviewed_peims(dataset: str, year: int, scope: str) -> tuple[str, str, DatasetMetadata, str]:
    config = load_config()
    metadata = config.resolve_dataset(dataset, year, exact=True, reader_profile="source").metadata
    resource = config.column_contract(dataset, year, reader_profile="source")
    policy = config.value_policies.get(metadata.value_policy)
    if resource is None or policy is None:
        raise ContractError("PEIMS comparison requires reviewed headers and value policy")
    branch = crosswalk_dataset(parse_column_schema(resource["text"].encode()), dataset, str(year))
    mapping, = [table for table in branch["tables"] if table["table"] == scope]
    revision = EditionKey.source_revision(table_input_sha256s(mapping))
    return (revision, sha256(canonical_json(mapping).encode()).hexdigest(), metadata,
            sha256(canonical_json(policy.to_dict()).encode()).hexdigest())


@dataclass(frozen=True)
class StatewideComparison:
    identifier: str | None
    scope: str
    metric: str
    period: DataPeriod
    local_edition: EditionKey
    statewide_edition: EditionKey
    unit: str
    dimensions: tuple[tuple[str, str], ...]
    local_raw: str | None
    local_status: str
    local_value: Number | None
    statewide_raw: str | None
    statewide_status: str
    statewide_value: Number | None
    difference: Number | None

    @property
    def local_percent(self) -> Number | None:
        return self.local_value if self.unit == "percent" else None

    @property
    def statewide_percent(self) -> Number | None:
        return self.statewide_value if self.unit == "percent" else None

    @property
    def difference_pp(self) -> Number | None:
        return self.difference if self.unit == "percent" else None

    def to_dict(self, *, include_meta: bool = True, include_geometry: bool = False) -> dict[str, Any]:
        return {**vars(self), "local_percent": self.local_percent, "statewide_percent": self.statewide_percent,
                "difference_pp": self.difference_pp}


@dataclass(frozen=True)
class _Statistic:
    name: str
    unit: str
    period: DataPeriod
    local_field: str
    statewide_field: str


@dataclass(frozen=True)
class _Projection:
    artifact: ArtifactRef
    columns: tuple[str, ...]
    support: Mapping[str, tuple[str, str]]

    def value(self, row: dict[str, Any], field: str) -> tuple[str, str, Number | None]:
        if not self.support:
            raw = row[field]
            if not isinstance(raw, str):
                raise ContractError("TAPR rate must preserve its publisher token")
            value = measure(raw, _TAPR_RATE_POLICY, unit="percent", minimum=0, maximum=100)
            status = "uninterpreted" if value.status == ValueStatus.INVALID else value.status.value
            return raw, status, value.numeric

        raw_field, status_field = self.support[field]
        raw, status, value = row[raw_field], row[status_field], row[field]
        if status == ValueStatus.OBSERVED.value:
            valid = type(value) in {int, float, Decimal} and isfinite(value)
        else:
            valid = value is None
        if not isinstance(raw, str) or status not in _VALUE_STATUSES or not valid:
            raise ContractError("Typed PEIMS statistic does not match its retained raw/status contract")
        return raw, status, float(value) if isinstance(value, Decimal) else value


def _projection(release: DataRelease, key: EditionKey, fields: list[str], *,
                keys: list[str], mapping: str, policy: str | None, period: DataPeriod) -> _Projection:
    plans = release.catalog.resolve_artifacts(((key, field) for field in fields), period=period)
    if len(plans) != 1:
        raise ContractError("Comparison requires one reviewed leaf per edition")
    plan, = plans
    artifact = plan["artifact"]
    contract = artifact.contract
    if (contract["mapping_sha256"] != mapping or contract["policy_sha256"] != policy
            or contract["key_columns"] != keys):
        raise ContractError("Comparison leaf does not match the reviewed mapping, grain, or policy")
    support = artifact.column_support
    field_support = {}
    if policy is None:
        if support is not None or set(plan["columns"]) != set(keys + fields):
            raise ContractError("TAPR comparison requires exact raw columns")
    else:
        if support is None:
            raise ContractError("PEIMS comparison requires normalized values with raw and status support")
        for field in fields:
            selected = support.projection(field)
            if not {"raw", "status"} <= selected.keys():
                raise ContractError("PEIMS comparison requires normalized values with raw and status support")
            field_support[field] = selected["raw"], selected["status"]
    return _Projection(artifact, plan["columns"], field_support)


def _statistics(local: EditionKey, local_metadata: DatasetMetadata,
                statewide_metadata: DatasetMetadata, local_review: DatasetMetadata,
                statewide_review: DatasetMetadata, metric: str) -> list[_Statistic]:
    common = local_review.metrics.keys() & statewide_review.metrics.keys()
    if local.dataset == PRIMARY_DISABILITY and int(local.edition) < 2025:
        common.discard("primary_developmental_delay_count")
    names = sorted(common) if metric == "*" else [metric]
    result = []
    for name in names:
        left, right = local_metadata.metrics.get(name), statewide_metadata.metrics.get(name)
        if (name not in common or left is None or right is None
                or left != local_review.metrics[name] or right != statewide_review.metrics[name]
                or left.unit is None or left.unit != right.unit or left.unit not in _NUMERIC_UNITS):
            raise ContractError(f"Statewide comparison requires the same reviewed numeric metric in both editions: {name}")
        period = left.period or local_metadata.period
        if (period.kind == "unknown" or period != (right.period or statewide_metadata.period)
                or period != (left.period or local_review.period)
                or period != (right.period or statewide_review.period)):
            raise ContractError("Statewide and local values must cover the same known reference period")
        result.append(_Statistic(name, left.unit, period, left.field, right.field))
    if not result:
        raise ContractError("These editions have no reviewed common numeric metrics")
    return result


def _compare_source(release: DataRelease, local: EditionKey, statewide: EditionKey, metric: str,
                    dimensions: Mapping[str, str]) -> tuple[list[_Statistic], _Projection, _Projection]:
    key_column = f"{local.scope}_number"
    course_pair = local.dataset == statewide.dataset == COURSES
    if course_pair and local.scope != "district":
        raise ContractError("Campus course comparisons are on hold pending publisher geography qualification")
    catalog = release.catalog
    local_metadata = catalog.metadata(local.dataset, edition=local)
    state_metadata = catalog.metadata(statewide.dataset, edition=statewide)
    if course_pair or (local.dataset == statewide.dataset and local.dataset in SCOPES_BY_DATASET):
        scopes = {"state", "district"} if course_pair else SCOPES_BY_DATASET[local.dataset]
        if local.scope not in scopes or local.table != local.scope or statewide.table != "state":
            raise ContractError("Publisher PEIMS statistics are unavailable for the selected scope")
        local_keys = course_key_columns(local.scope) if course_pair else _key_columns(local.dataset, local.scope)
        state_keys = course_key_columns("state") if course_pair else _key_columns(local.dataset, "state")
        for edition, keys in ((local, local_keys), (statewide, state_keys)):
            if edition.grain != "one publisher row per " + " and ".join(keys):
                raise ContractError("PEIMS comparison requires the reviewed publisher grain")
        left_revision, left_mapping, left_review, left_policy = _reviewed_peims(local.dataset, int(local.edition), local.scope)
        right_revision, right_mapping, right_review, right_policy = _reviewed_peims(statewide.dataset, int(statewide.edition), "state")
        expected_dimensions = set(state_keys) - {"school_year"}
        if set(dimensions) != expected_dimensions:
            raise ContractError(f"Expected exact publisher dimensions {sorted(expected_dimensions)}")
    elif (local.table in _TAPR_RATE_FAMILIES and local.table == statewide.table
          and local.dataset == f"{local.scope}_tapr_{local.table}"
          and statewide.dataset == f"state_tapr_{local.table}" and local.edition == str(EDITION)):
        if local.grain != f"one row per {key_column}" or statewide.grain != STATEWIDE_SINGLETON_GRAIN or dimensions:
            raise ContractError("TAPR comparison requires the reviewed local grain and statewide singleton")
        local_keys, state_keys = [key_column], []
        left_revision, left_mapping, left_review = _reviewed_tapr(local.scope, local.table)
        right_revision, right_mapping, right_review = _reviewed_tapr("state", local.table)
        left_policy = right_policy = None
        masking = next((ref for ref in left_review.references if ref.reference_id == _TAPR_RATE_POLICY.reference_id), None)
        if masking is None or not masking.provenance_ids or masking not in local_metadata.references:
            raise ContractError("Statewide comparison requires the retained TAPR rate masking methodology")
    else:
        raise ContractError("These editions have no reviewed statewide comparison contract")
    if (local.revision_sha256, statewide.revision_sha256) != (left_revision, right_revision):
        raise ContractError("Statewide comparison requires the reviewed publisher revisions")
    if any((selected.value_policy, selected.references) != (review.value_policy, review.references)
           for selected, review in ((local_metadata, left_review), (state_metadata, right_review))):
        raise ContractError("Comparison requires the reviewed methodology and source references")
    statistics = _statistics(local, local_metadata, state_metadata, left_review, right_review, metric)
    period = statistics[0].period
    if any(stat.period != period for stat in statistics):
        raise ContractError("Select metrics with one common reference period")
    return (statistics, _projection(release, local, [stat.local_field for stat in statistics],
                                    keys=local_keys, mapping=left_mapping, policy=left_policy, period=period),
            _projection(release, statewide, [stat.statewide_field for stat in statistics],
                        keys=state_keys, mapping=right_mapping, policy=right_policy, period=period))


def compare_statewide(items: Sequence[object], *, release: DataRelease, local: EditionKey,
                      statewide: EditionKey, metric: str = "*",
                      dimensions: Mapping[str, str] | None = None) -> list[StatewideComparison]:
    """Compare reviewed metrics against the publisher's exact statewide values."""
    if local.scope not in {"district", "campus"} or statewide.scope != "state" or local.edition != statewide.edition:
        raise ContractError("Statewide comparison requires matched local and statewide editions")
    entity_type = Campus if local.scope == "campus" else District
    if any(not isinstance(item, entity_type) for item in items):
        raise ContractError("Query entities must match the selected local edition scope")
    dimensions = {} if dimensions is None else dimensions
    if not isinstance(dimensions, Mapping) or any(not isinstance(key, str) or not isinstance(value, str)
                                                  for key, value in dimensions.items()):
        raise ContractError("Comparison dimensions must use literal publisher strings")
    statistics, local_plan, state_plan = _compare_source(release, local, statewide, metric, dimensions)
    if not items:
        return []

    import pyarrow as pa
    import pyarrow.compute as arrow_compute
    import pyarrow.dataset as ds

    pc: Any = arrow_compute
    predicate = None
    for field, literal in dimensions.items():
        selected = ds.field(field) == literal
        predicate = selected if predicate is None else predicate & selected
    state_rows = release.analytical_dataset(state_plan.artifact.identity).head(
        2, columns=list(state_plan.columns), filter=predicate)
    if state_rows.num_rows != 1:
        raise ContractError("Select exactly one published statewide observation")
    state_row, = state_rows.to_pylist()
    state_values = [state_plan.value(state_row, stat.statewide_field) for stat in statistics]

    identifiers = [item.lookup_key if isinstance(item, Campus) else item.district_number_canon
                   if isinstance(item, District) else None for item in items]
    applicable = [not (isinstance(item, Campus) and item.is_private) for item in items]
    wanted = {"'" + identifier for identifier, applies in zip(identifiers, applicable, strict=True) if applies and identifier}
    raw_by_key = {}
    if wanted:
        source_rows = release.analytical_dataset(local_plan.artifact.identity).to_table(
            columns=list(local_plan.columns), filter=predicate)
        key_column = f"{local.scope}_number"
        keys = canonical_source_identifiers(source_rows[key_column], width=9 if local.scope == "campus" else 6)
        selected = pc.is_in(keys, value_set=pa.array(sorted(wanted), type=pa.string()))
        selected_keys = pc.filter(keys, selected).to_pylist()
        if len(selected_keys) != len(set(selected_keys)):
            raise ContractError("Local source contains duplicate selected identifiers")
        raw_by_key = dict(zip(selected_keys, source_rows.filter(selected).to_pylist(), strict=True))

    rows = []
    dimension_values = tuple(sorted(dimensions.items()))
    for identifier, applies in zip(identifiers, applicable, strict=True):
        local_row = raw_by_key.get("'" + identifier) if applies and identifier else None
        for stat, (state_raw, state_status, state_value) in zip(statistics, state_values, strict=True):
            if local_row is None:
                raw, status, value = None, "unavailable" if applies else "not_applicable", None
            else:
                raw, status, value = local_plan.value(local_row, stat.local_field)
            difference = None
            if value is not None and state_value is not None:
                difference = (value - state_value if isinstance(value, int) and isinstance(state_value, int)
                              else float(Decimal(str(value)) - Decimal(str(state_value))))
            rows.append(StatewideComparison(identifier or None, local.scope, stat.name, stat.period, local, statewide,
                        stat.unit, dimension_values, raw, status, value, state_raw, state_status, state_value, difference))
    return rows
