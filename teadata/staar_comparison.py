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


FAMILIES = frozenset(f"staar_tested_grade_{grade}" for grade in range(3, 9))


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
    path = Path(__file__).with_name("schemas") / "staar-tested-grade-comparison-bindings.json"
    with path.open("rb") as stream:
        body = stream.read(1024 * 1024 + 1)
    if len(body) > 1024 * 1024:
        raise ContractError("STAAR comparison bindings exceed the metadata budget")
    return _parse_bindings(body)


def _parse_bindings(body: bytes) -> dict:
    document = json.loads(body, object_pairs_hook=_unique_object)
    if document["schema_version"] != 1 or set(document["editions"]) != {"2025", "2026"}:
        raise ContractError("Unsupported STAAR comparison bindings")
    for year, edition in document["editions"].items():
        if (set(edition["families"]) != FAMILIES
                or DataPeriod.from_dict(edition["period"]) != DataPeriod("school_year", int(year) - 1, int(year))):
            raise ContractError("STAAR comparison bindings require exact tested-grade families and periods")
        names, codes = [], []
        for family in edition["families"].values():
            if set(family["scopes"]) != {"state", "district", "campus", "region"}:
                raise ContractError("STAAR comparison bindings require all four publisher scopes")
            for name, rate in family["rates"].items():
                if (set(rate) != {"code", "label", "field"}
                        or any(not isinstance(value, str) or not value for value in rate.values())
                        or name != rate["field"] or not rate["code"].endswith("R")):
                    raise ContractError("Invalid STAAR published-rate binding")
                names.append(name)
                codes.append(rate["code"])
        if len(names) != 630 or len(set(names)) != 630 or len(set(codes)) != 630:
            raise ContractError("STAAR comparison requires 630 unique reviewed rate fields and codes per edition")
    return document

@lru_cache(maxsize=48)
def _source(scope: str, family: str, edition: str) -> tuple[str, str, dict, dict]:
    try:
        reviewed = _bindings()["editions"][edition]
        family_binding = reviewed["families"][family]
        pinned = family_binding["scopes"][scope]
    except KeyError as error:
        raise ContractError("No reviewed STAAR comparison for the selected scope, family and edition") from error
    config = load_config()
    dataset = f"{scope}_tapr_{family}"
    request, = config.tapr_source_requests(dataset, int(edition))
    metadata = config.resolve_dataset(dataset, int(edition), exact=True, reader_profile="source").metadata
    mapping = _digest(request.mapping)
    if (request.mapping["input_sha256"] != pinned["raw_sha256"]
            or mapping != pinned["mapping_sha256"]
            or _digest(metadata.to_dict()) not in pinned["metadata_sha256"]
            or request.mapping["metric_period"] != reviewed["period"]
            or metadata.period.to_dict() != reviewed["period"]):
        raise ContractError("STAAR comparison requires the pinned publisher source, mapping and period")
    fields = {field["code"]: field for field in request.mapping["fields"]}
    for name, rate in family_binding["rates"].items():
        field = fields.get(scope[0].upper() + rate["code"])
        if (field is None or field["disposition"] != "fact" or field["dictionary_status"] != "documented"
                or field["label"].strip() != rate["label"]
                or field["field"] != f"{scope}_{edition}_{rate['field']}" or name != rate["field"]):
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
    references = tuple(ReferenceDocument.from_dict({
        "reference_id": "staar_rate_legend" if index == 0 else f"staar_rate_reference_{index}",
        "role": "masking_methodology" if index == 0 else "source_notes",
        "title": f"TEA {key.edition} tested-grade rate interpretation evidence",
        "url": witness["url"], "status": "available", "reviewed_sha256": witness["sha256"],
        "provenance_ids": [witness["provenance_id"]],
    }) for index, witness in enumerate(reviewed["methodology"]))
    policy = ValuePolicy(f"staar_tested_grade_{key.edition}_published_rates",
                         {token: TokenRule(ValueStatus(rule["status"]), rule["reason"])
                          for token, rule in reviewed["rate_tokens"].items()}, reference_id="staar_rate_legend")
    metadata = replace(metadata, metrics=metrics, references=(*metadata.references, *references),
                       notes=" ".join((metadata.notes or "", *_bindings()["notes"], *reviewed.get("notes", []))).strip())
    return StaarRateContract(revision, mapping, metadata, policy)
