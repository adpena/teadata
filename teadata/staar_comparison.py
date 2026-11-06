"""Reviewed same-edition STAAR rates over the retained raw source contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from typing import TYPE_CHECKING

from .dataset_metadata import DataPeriod, DatasetMetadata, EditionKey, MetricDefinition, ReferenceDocument
from .manifest import ContractError, canonical_json
from .teadata_config import load_config
from .values import TokenRule, ValuePolicy, ValueStatus

if TYPE_CHECKING:
    from .release import DataRelease


_TESTED_GRADE_COUNTS = dict(zip(
    (f"staar_tested_grade_{grade}" for grade in range(3, 9)),
    (84, 84, 126, 84, 84, 168),
    strict=True,
))
FAMILIES = frozenset(_TESTED_GRADE_COUNTS)
_AGGREGATE_COUNTS = {
    "2025": {"staar_all_grades": 210},
    "2026": {"staar_all_grades": 210, "staar_end_of_course": 210},
}
_AGGREGATE_FAMILIES = frozenset(family for counts in _AGGREGATE_COUNTS.values() for family in counts)


def supports_staar_rates(family: str, edition: str) -> bool:
    """Identify the comparison routes with a reviewed rate-only contract."""
    return family in FAMILIES or family in _AGGREGATE_COUNTS.get(edition, {})


@dataclass(frozen=True)
class StaarRateContract:
    revision_sha256: str
    mapping_sha256: str
    metadata: DatasetMetadata
    value_policy: ValuePolicy
    observed_token_format: str = "canonical integer from 0 through 100"


def _digest(value: Mapping) -> str:
    return sha256(canonical_json(value).encode()).hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"Duplicate STAAR comparison binding key: {key}")
        result[key] = value
    return result


@lru_cache(maxsize=1)
def _bindings() -> dict:
    return _read_bindings("staar-tested-grade-comparison-bindings.json")


@lru_cache(maxsize=1)
def _aggregate_bindings() -> dict:
    return _read_bindings("staar-aggregate-comparison-bindings.json", aggregate=True)


def _read_bindings(resource: str, *, aggregate: bool = False) -> dict:
    path = Path(__file__).with_name("methodologies") / resource
    with path.open("rb") as stream:
        body = stream.read(1024 * 1024 + 1)
    if len(body) > 1024 * 1024:
        raise ContractError("STAAR comparison bindings exceed the metadata budget")
    return _parse_bindings(body, aggregate=aggregate)


def _parse_bindings(body: bytes, *, aggregate: bool = False) -> dict:
    document = json.loads(body, object_pairs_hook=_unique_object)
    expected_editions = set(_AGGREGATE_COUNTS) if aggregate else {"2024", "2025", "2026"}
    if document["schema_version"] != 1 or set(document["editions"]) != expected_editions:
        raise ContractError("Unsupported STAAR comparison bindings")
    for year, edition in document["editions"].items():
        expected_counts = _AGGREGATE_COUNTS[year] if aggregate else _TESTED_GRADE_COUNTS
        if (set(edition["families"]) != expected_counts.keys()
                or DataPeriod.from_dict(edition["period"]) != DataPeriod("school_year", int(year) - 1, int(year))):
            raise ContractError("STAAR comparison bindings require exact families and periods")
        names, codes, fields = [], [], []
        for family_name, family in edition["families"].items():
            if set(family["scopes"]) != {"state", "district", "campus", "region"}:
                raise ContractError("STAAR comparison bindings require all four publisher scopes")
            if len(family["rates"]) != expected_counts[family_name]:
                raise ContractError("STAAR comparison bindings have an incomplete rate family")
            for name, rate in family["rates"].items():
                if (set(rate) != {"code", "label", "field"}
                        or any(not isinstance(value, str) or not value for value in rate.values())
                        or not isinstance(name, str) or not name or not rate["code"].endswith("R")):
                    raise ContractError("Invalid STAAR published-rate binding")
                names.append(name)
                codes.append(rate["code"])
                fields.append(rate["field"])
        if not len(names) == len(set(names)) == len(set(codes)) == len(set(fields)):
            raise ContractError("STAAR comparison requires unique reviewed rate fields and codes per edition")
    return document

@lru_cache(maxsize=80)
def _source(scope: str, family: str, edition: str) -> tuple[str, str, dict, dict]:
    try:
        document = _aggregate_bindings() if family in _AGGREGATE_FAMILIES else _bindings()
        reviewed = document["editions"][edition]
        family_binding = reviewed["families"][family]
        pinned = family_binding["scopes"][scope]
    except KeyError as error:
        raise ContractError("No reviewed STAAR comparison for the selected scope, family and edition") from error
    config = load_config()
    dataset = f"{scope}_tapr_{family}"
    request, = config.tapr_source_requests(dataset, int(edition))
    metadata = config.resolve_dataset(dataset, int(edition), exact=True, reader_profile="source").metadata
    mapping = _digest(request.mapping)
    expected_key = None if scope == "state" else f"{scope}_number"
    if (request.key_field != expected_key or request.mapping["input_sha256"] != pinned["raw_sha256"]
            or mapping != pinned["mapping_sha256"]
            or _digest(metadata.to_dict()) not in pinned["metadata_sha256"]
            or request.mapping["metric_period"] != reviewed["period"]
            or metadata.period.to_dict() != reviewed["period"]):
        raise ContractError("STAAR comparison requires the pinned publisher source, mapping and period")
    fields = {field["code"]: field for field in request.mapping["fields"]}
    for name, rate in family_binding["rates"].items():
        field = fields.get(scope[0].upper() + rate["code"])
        if (field is None or field["disposition"] != "fact" or field["dictionary_status"] != "documented"
                or field["label"].strip() != pinned.get("label_prefix", "") + rate["label"]
                or field["field"] != f"{scope}_{edition}_{rate['field']}"):
            raise ContractError("STAAR comparison requires the reviewed publisher rate label and code")
    return EditionKey.source_revision((pinned["raw_sha256"],)), mapping, reviewed, family_binding


def staar_rate_contract(release: DataRelease, edition: EditionKey) -> StaarRateContract:
    """Discover reviewed rate definitions and interpretation evidence without reading observations.

    These definitions describe the publisher's rounded percentages for the
    accountability subset. Raw numerator and denominator columns retain their
    original tokens. Use a same-edition statewide leaf in the same release for
    comparison; this contract does not establish a longitudinal comparison.
    Observed rates require canonical integer tokens from 0 through 100. The
    value policy describes the exact suppression token; blanks remain missing
    and other spellings remain uninterpreted.
    """
    key = edition
    if key.dataset != f"{key.scope}_tapr_{key.table}":
        raise ContractError("STAAR comparison requires the qualified publisher dataset")
    revision, mapping, reviewed, family = _source(key.scope, key.table, key.edition)
    metadata = release.catalog.metadata(key.dataset, edition=key)
    if key.revision_sha256 != revision:
        raise ContractError("STAAR comparison requires the reviewed publisher revision")
    if _digest(metadata.to_dict()) not in family["scopes"][key.scope]["metadata_sha256"]:
        raise ContractError("STAAR comparison requires the reviewed source references and metadata")
    period = DataPeriod.from_dict(reviewed["period"])
    metrics = {name: MetricDefinition(f"{key.scope}_{key.edition}_{rate['field']}", period, "percent")
               for name, rate in family["rates"].items()}
    aggregate = key.table in _AGGREGATE_FAMILIES
    kind = "aggregate" if aggregate else "tested-grade"
    references = tuple(ReferenceDocument.from_dict({
        "reference_id": "staar_rate_legend" if index == 0 else f"staar_rate_reference_{index}",
        "role": "masking_methodology" if index == 0 else "source_notes",
        "title": f"TEA {key.edition} {kind} rate interpretation evidence",
        "url": witness["url"], "status": "available", "reviewed_sha256": witness["sha256"],
        "provenance_ids": [witness["provenance_id"]],
    }) for index, witness in enumerate(reviewed["methodology"]))
    policy = ValuePolicy(f"staar_{kind.replace('-', '_')}_{key.edition}_published_rates",
                         {token: TokenRule(ValueStatus(rule["status"]), rule["reason"])
                          for token, rule in reviewed["rate_tokens"].items()}, reference_id="staar_rate_legend")
    document = _aggregate_bindings() if aggregate else _bindings()
    metadata = replace(metadata, metrics=metrics, references=(*metadata.references, *references),
                       notes=" ".join((metadata.notes or "", *document["notes"], *reviewed.get("notes", []))).strip())
    return StaarRateContract(revision, mapping, metadata, policy)
