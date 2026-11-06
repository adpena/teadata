"""Compare compatible publisher statistics using exact local and statewide editions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from functools import lru_cache
from hashlib import sha256
import json
from math import isfinite
from typing import TYPE_CHECKING, Any

from .column_schema import crosswalk_dataset, parse_column_schema, table_input_sha256s
from .dataset_metadata import STATEWIDE_SINGLETON_GRAIN, ArtifactRef, DataPeriod, DatasetMetadata, EditionKey
from .entities import Campus, District
from .identifiers import canonical_source_identifiers
from .manifest import ContractError, canonical_json
from .peims_courses import DATASET as COURSES, course_key_columns
from .peims_leaves import PRIMARY_DISABILITY, SCOPES_BY_DATASET, _key_columns
from .release import _student_attachments
from .staar_comparison import FAMILIES as STAAR_FAMILIES, staar_rate_contract
from .tapr import tapr_source_identifiers
from .teadata_config import load_config
from . import tprs_students as students
from .values import MeasurementUnit, TokenRule, ValuePolicy, ValueStatus, measure

if TYPE_CHECKING:
    from .release import DataRelease

_TAPR_RAW_FAMILIES = {"college_career_military_readiness", "college_readiness_components",
                      "four_year_longitudinal_rate", "student_profile"}
EDITION = 2025
_TAPR_RATE_POLICY = ValuePolicy(
    "tapr_2025_reviewed_rates",
    {"-1": TokenRule(ValueStatus.SUPPRESSED, "TEA masked the rate")},
    reference_id="masking",
)
_TAPR_STUDENT_POLICY = ValuePolicy("tapr_2025_reviewed_student_profile", {}, reference_id="masking")
_NUMERIC_UNITS = frozenset(MeasurementUnit) - {MeasurementUnit.TEXT}
_VALUE_STATUSES = frozenset(item.value for item in ValueStatus)
_REGION_CODES = frozenset(f"{code:02d}" for code in range(1, 21))
Number = int | float


@lru_cache(maxsize=12)
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
class _StudentContract:
    scope: str
    fields: Mapping[str, Mapping[str, Any]]
    policy: ValuePolicy
    columns: Mapping[str, tuple[str, ...]]
    sources: Mapping[str, tuple[str, str]]
    local_columns: Mapping[tuple[str, str], int]
    cohort: tuple[str, str, str, str] | None = None

    def value(self, row: dict[str, Any], field: str, unit: str) -> tuple[str | None, str, Number | None]:
        spec = self.fields[field]
        raw_column, status_column, row_column, column_column, reason_column = self.columns[field]
        raw, status, value = row[raw_column], row[status_column], row[field]
        locator, column, reason = row[row_column], row[column_column], row[reason_column]
        identifier = "S" if self.scope == "state" else row[f"{self.scope}_number"]
        if ((row["tprs_report_receipt"], row["tprs_report_sha256"]) != self.sources.get(identifier)
                or unit != spec["unit"]):
            raise ContractError("Student comparison row differs from its exact source binding")
        if raw is None:
            if (self.scope != "campus" or spec["section"] != "Students by Grade"
                    or status != "missing" or value is not None or locator is not None or column is not None
                    or reason != "row_not_published"):
                raise ContractError("Student comparison requires an explicit missing campus grade row")
            return None, status, None
        if (not isinstance(raw, str) or type(locator) is not int or locator <= 0
                or type(column) is not int or column != self.local_columns[spec["population"], spec["quantity"]]):
            raise ContractError("Student comparison requires exact raw tokens and local cell locators")
        cell = students.StudentCell(spec["section"], spec["label"], spec["population"], spec["quantity"],
                                    "local", raw, locator, column, field, None, students.PERIOD)
        measured = cell.measurement(self.policy)
        valid_type = type(value) is (int if unit == "count" else float) if status == "observed" else value is None
        if (measured.status == ValueStatus.INVALID or measured.status.value != status
                or not valid_type or measured.value != value or measured.reason != reason):
            raise ContractError("Student comparison typed value differs from its retained token and status")
        return raw, status, value


def _reviewed_students(release: DataRelease, key: EditionKey) -> tuple[str, str, DatasetMetadata, str, _StudentContract]:
    keys = [] if key.scope == "state" else [f"{key.scope}_number"]
    grain = STATEWIDE_SINGLETON_GRAIN if not keys else f"one row per {keys[0]}"
    if (key.dataset != students.DATASET or key.edition != str(students.EDITION)
            or key.scope not in students.SCOPES or key.table != key.scope or key.grain != grain):
        raise ContractError("Student comparison requires the reviewed edition, scope and grain")
    config = load_config()
    review = config.resolve_dataset(students.DATASET, students.EDITION, exact=True, reader_profile="source").metadata
    metadata = release.catalog.metadata(key.dataset, edition=key)
    # Selected coverage changes the genuine derived schema observation, not the
    # field definitions, policy, or other publisher interpretation references.
    expected = {ref.reference_id: ref for ref in review.references}
    actual = {ref.reference_id: ref for ref in metadata.references}
    if set(actual) != set(expected) or len(actual) != len(metadata.references):
        raise ContractError("Student comparison requires the reviewed source references")
    columns = actual["columns"]
    if (replace(columns, path=expected["columns"].path, reviewed_sha256=expected["columns"].reviewed_sha256,
                provenance_ids=expected["columns"].provenance_ids) != expected["columns"]
            or any(actual[name] != ref for name, ref in expected.items() if name != "columns")
            or replace(metadata, references=review.references) != review):
        raise ContractError("Student comparison metadata differs from the packaged field contract")
    if review.value_policy is None:
        raise ContractError("Student comparison requires its reviewed disclosure policy")
    policy = config.value_policies[review.value_policy]
    if release.catalog.policy(key.dataset, edition=key) != policy:
        raise ContractError("Student comparison requires the reviewed disclosure policy")
    fields = {field["field"]: field for field in students.semantic_fields()}
    plans = release.catalog.resolve_artifacts([(key, field) for field in fields], period=students.PERIOD)
    if len(plans) != 1:
        raise ContractError("Student comparison requires one complete reviewed native leaf")
    artifact = plans[0]["artifact"]
    cohort = None
    attachments = _student_attachments(release.catalog, artifact)
    if attachments:
        reports, cohort = release._student_report_ledger(artifact, attachments)
        sources = {item.identifier: (item.receipt, item.sha256) for item in reports}
    else:
        sources = _selected_student_sources(release, key)
    hashes = [digest for _, digest in sources.values()]
    if key.revision_sha256 != EditionKey.source_revision(tuple(hashes)):
        raise ContractError("Student comparison source revision differs from its ordered inputs")
    contract = artifact.contract
    if (contract["inputs"] != hashes or contract["rows"] != len(sources)
            or (key.scope == "state" and len(sources) != 1)):
        raise ContractError("Student comparison leaf differs from its exact selected source coverage")
    support = artifact.column_support
    pointers = {}
    for field in fields:
        names = tuple(f"tprs_{kind}_{field}" for kind in ("raw", "status", "row", "column", "reason"))
        expected_support = {"representation": "normalized", "raw": names[0], "status": names[1],
                            "auxiliary": list(names[2:]), "context": ["tprs_report_receipt", "tprs_report_sha256", *keys]}
        if support is None or support.projection(field) != expected_support:
            raise ContractError("Student comparison requires the retained raw/status/locator support")
        pointers[field] = names
    table, = students._crosswalk({key.scope: hashes})["datasets"][0]["tables"]
    mapping = sha256(canonical_json(table).encode()).hexdigest()
    policy_hash = sha256(canonical_json(policy.to_dict()).encode()).hexdigest()
    local_columns = {(population, quantity): index + 2
                     for index, (population, quantity, role) in enumerate(students.definition()["column_roles"][key.scope])
                     if role == "local"}
    typed = _StudentContract(key.scope, fields, policy, pointers, sources, local_columns, cohort)
    return key.revision_sha256, mapping, replace(review, references=metadata.references), policy_hash, typed


def _selected_student_sources(release: DataRelease, key: EditionKey) -> dict[str, tuple[str, str]]:
    acquisitions = release.catalog.acquisitions(key.dataset, edition=key)
    document = release.catalog.to_dict()
    if document["schema_version"] == 4:
        document = json.loads(release.catalog.edition_metadata_bytes(key.identity))
    records = document["provenance"]  # Already closed and verified by DatasetCatalog.
    sources = {}
    for acquisition in acquisitions:
        parameters = acquisition.get("parameters", {})
        identifier = parameters.get("identifier")
        students._identifier(key.scope, identifier)
        original = parameters.get("original_download_receipt")
        if (parameters.get("scope") != key.scope or identifier in sources
                or original not in acquisition["source_metadata"].get("provenance_ids", ())):
            raise ContractError("Student comparison requires exact original-report lineage and unique identifiers")
        parent = records.get(original)
        if (acquisition.get("method") != "local_observation" or parent is None
                or any(acquisition["artifact"][name] != parent["artifact"][name] for name in ("sha256", "size_bytes"))):
            raise ContractError("Student comparison requires a local observation of the exact original bytes")
        digest = acquisition["artifact"]["sha256"]
        students._validate_report_acquisition(parent, scope=key.scope, identifier=identifier, expected_sha256=digest)
        sources[identifier] = original, digest
    return sources


@dataclass(frozen=True)
class _Projection:
    artifact: ArtifactRef
    columns: tuple[str, ...]
    support: Mapping[str, tuple[str, str]]
    raw_policy: ValuePolicy | None
    integer_rates: bool = False
    student: _StudentContract | None = None

    def value(self, row: dict[str, Any], field: str, unit: str) -> tuple[str | None, str, Number | None]:
        if self.student is not None:
            return self.student.value(row, field, unit)
        if self.raw_policy is not None:
            raw = row[field]
            if not isinstance(raw, str) or unit not in {"count", "percent"}:
                raise ContractError("TAPR comparison requires a publisher token with a reviewed count or percent unit")
            if (self.integer_rates and raw not in {"", "-1"}
                    and not (raw.isascii() and raw.isdecimal() and len(raw) <= 3 and raw == str(int(raw)))):
                return raw, "uninterpreted", None
            value = measure(raw, self.raw_policy, unit=unit, minimum=0, maximum=100 if unit == "percent" else None)
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


def _projection(release: DataRelease, key: EditionKey, fields: list[tuple[str, DataPeriod]], *,
                keys: list[str], mapping: str, policy: str | None, raw_policy: ValuePolicy | None,
                integer_rates: bool = False, student: _StudentContract | None = None) -> _Projection:
    by_period: dict[DataPeriod, list[tuple[EditionKey, str]]] = {}
    for field, period in fields:
        by_period.setdefault(period, []).append((key, field))
    plans = [plan for period, selected in by_period.items()
             for plan in release.catalog.resolve_artifacts(selected, period=period)]
    if len({plan["artifact"].identity for plan in plans}) != 1:
        raise ContractError("Comparison requires one reviewed leaf per edition")
    artifact = plans[0]["artifact"]
    columns = tuple(dict.fromkeys(column for plan in plans for column in plan["columns"]))
    contract = artifact.contract
    if (contract["mapping_sha256"] != mapping or contract["policy_sha256"] != policy
            or contract["key_columns"] != keys):
        raise ContractError("Comparison leaf does not match the reviewed mapping, grain, or policy")
    support = artifact.column_support
    field_support = {}
    if policy is None:
        if raw_policy is None or support is not None or set(columns) != {*keys, *(field for field, _ in fields)}:
            raise ContractError("TAPR comparison requires exact raw columns")
    else:
        if support is None:
            raise ContractError("PEIMS comparison requires normalized values with raw and status support")
        for field, _ in fields:
            selected = support.projection(field)
            if not {"raw", "status"} <= selected.keys():
                raise ContractError("PEIMS comparison requires normalized values with raw and status support")
            field_support[field] = selected["raw"], selected["status"]
    return _Projection(artifact, columns, field_support, raw_policy, integer_rates, student)


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
    raw_policy = None
    left_student = right_student = None
    staar = local.table in STAAR_FAMILIES and local.table == statewide.table
    if staar:
        if local.grain != f"one row per {key_column}" or statewide.grain != STATEWIDE_SINGLETON_GRAIN or dimensions:
            raise ContractError("STAAR comparison requires the reviewed local grain and statewide singleton")
        local_keys, state_keys = [key_column], []
        left = staar_rate_contract(release, local)
        right = staar_rate_contract(release, statewide)
        left_revision, left_mapping, local_metadata, raw_policy = (
            left.revision_sha256, left.mapping_sha256, left.metadata, left.value_policy)
        right_revision, right_mapping, state_metadata = right.revision_sha256, right.mapping_sha256, right.metadata
        if raw_policy != right.value_policy:
            raise ContractError("STAAR comparison requires one reviewed same-edition rate policy")
        left_review, right_review = local_metadata, state_metadata
        left_policy = right_policy = None
    elif local.dataset == statewide.dataset == students.DATASET:
        if dimensions:
            raise ContractError("Student comparison has no additional publisher dimensions")
        local_keys, state_keys = [key_column], []
        left_revision, left_mapping, left_review, left_policy, left_student = _reviewed_students(release, local)
        right_revision, right_mapping, right_review, right_policy, right_student = _reviewed_students(release, statewide)
        if left_student.cohort != right_student.cohort:
            raise ContractError("Student comparison requires the same pinned statewide cohort")
    elif course_pair or (local.dataset == statewide.dataset and local.dataset in SCOPES_BY_DATASET):
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
    elif (local.table in _TAPR_RAW_FAMILIES and local.table == statewide.table
          and local.dataset == f"{local.scope}_tapr_{local.table}"
          and statewide.dataset == f"state_tapr_{local.table}" and local.edition == str(EDITION)):
        if local.grain != f"one row per {key_column}" or statewide.grain != STATEWIDE_SINGLETON_GRAIN or dimensions:
            raise ContractError("TAPR comparison requires the reviewed local grain and statewide singleton")
        local_keys, state_keys = [key_column], []
        left_revision, left_mapping, left_review = _reviewed_tapr(local.scope, local.table)
        right_revision, right_mapping, right_review = _reviewed_tapr("state", local.table)
        left_policy = right_policy = None
        raw_policy = _TAPR_STUDENT_POLICY if local.table == "student_profile" else _TAPR_RATE_POLICY
        masking = next((ref for ref in left_review.references if ref.reference_id == raw_policy.reference_id), None)
        if masking is None or not masking.provenance_ids or masking not in local_metadata.references:
            raise ContractError("Statewide comparison requires the retained TAPR masking methodology")
    else:
        raise ContractError("These editions have no reviewed statewide comparison contract")
    if (local.revision_sha256, statewide.revision_sha256) != (left_revision, right_revision):
        raise ContractError("Statewide comparison requires the reviewed publisher revisions")
    if any((selected.value_policy, selected.references) != (review.value_policy, review.references)
           for selected, review in ((local_metadata, left_review), (state_metadata, right_review))):
        raise ContractError("Comparison requires the reviewed methodology and source references")
    statistics = _statistics(local, local_metadata, state_metadata, left_review, right_review, metric)
    return (statistics, _projection(release, local, [(stat.local_field, stat.period) for stat in statistics],
                                    keys=local_keys, mapping=left_mapping, policy=left_policy, raw_policy=raw_policy,
                                    integer_rates=staar, student=left_student),
            _projection(release, statewide, [(stat.statewide_field, stat.period) for stat in statistics],
                        keys=state_keys, mapping=right_mapping, policy=right_policy, raw_policy=raw_policy,
                        integer_rates=staar, student=right_student))


def compare_statewide(items: Sequence[object], *, release: DataRelease, local: EditionKey,
                      statewide: EditionKey, metric: str = "*",
                      dimensions: Mapping[str, str] | None = None) -> list[StatewideComparison]:
    """Compare reviewed metrics against the publisher's exact statewide values."""
    if local.scope not in {"district", "campus", "region"} or statewide.scope != "state" or local.edition != statewide.edition:
        raise ContractError("Statewide comparison requires matched local and statewide editions")
    regional = local.scope == "region"
    entity_type = Campus if local.scope == "campus" else District
    if not regional and any(not isinstance(item, entity_type) for item in items):
        raise ContractError("Query entities must match the selected local edition scope")
    if regional and (local.table not in STAAR_FAMILIES
                     or any(type(item) is not str or item not in _REGION_CODES for item in items)):
        raise ContractError("Region comparisons require literal education service center codes 01 through 20")
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
    state_values = [state_plan.value(state_row, stat.statewide_field, stat.unit) for stat in statistics]

    identifiers = [item if regional and isinstance(item, str) else item.lookup_key if isinstance(item, Campus) else item.district_number_canon
                   if isinstance(item, District) else None for item in items]
    applicable = [not (isinstance(item, Campus) and item.is_private) for item in items]
    key_prefix = "" if regional else "'"
    wanted = {key_prefix + identifier for identifier, applies in zip(identifiers, applicable, strict=True) if applies and identifier}
    full_student = local_plan.student if local_plan.student is not None and local_plan.student.cohort is not None else None
    raw_by_key = {}
    if wanted:
        key_column = f"{local.scope}_number"
        local_filter = predicate
        if full_student is not None:
            # This leaf has already committed to exact unquoted publisher IDs.
            local_filter = ds.field(key_column).isin([key[1:] for key in sorted(wanted)])
        source_rows = release.analytical_dataset(local_plan.artifact.identity).to_table(
            columns=list(local_plan.columns), filter=local_filter)
        keys = (tapr_source_identifiers(source_rows[key_column], key_field=key_column) if regional else
                canonical_source_identifiers(source_rows[key_column], width=9 if local.scope == "campus" else 6))
        selected = pc.is_in(keys, value_set=pa.array(sorted(wanted), type=pa.string()))
        selected_keys = pc.filter(keys, selected).to_pylist()
        if len(selected_keys) != len(set(selected_keys)):
            raise ContractError("Local source contains duplicate selected identifiers")
        raw_by_key = dict(zip(selected_keys, source_rows.filter(selected).to_pylist(), strict=True))

    rows = []
    dimension_values = tuple(sorted(dimensions.items()))
    for identifier, applies in zip(identifiers, applicable, strict=True):
        local_row = raw_by_key.get(key_prefix + identifier) if applies and identifier else None
        if local_row is None and full_student is not None and applies and identifier in full_student.sources:
            raise ContractError("Statewide student leaf omits a requested district from its report ledger")
        for stat, (state_raw, state_status, state_value) in zip(statistics, state_values, strict=True):
            if local_row is None:
                raw, status, value = None, "unavailable" if applies else "not_applicable", None
            else:
                raw, status, value = local_plan.value(local_row, stat.local_field, stat.unit)
            difference = None
            if value is not None and state_value is not None:
                difference = (value - state_value if isinstance(value, int) and isinstance(state_value, int)
                              else float(Decimal(str(value)) - Decimal(str(state_value))))
            rows.append(StatewideComparison(identifier or None, local.scope, stat.name, stat.period, local, statewide,
                        stat.unit, dimension_values, raw, status, value, state_raw, state_status, state_value, difference))
    return rows
