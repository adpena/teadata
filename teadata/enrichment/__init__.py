from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, Literal
from .base import Enricher

_registry: Dict[str, Enricher] = {}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnrichmentOutcome:
    """Observed join coverage, not a claim about every source field/population."""

    dataset: str
    requested_year: int | None
    status: Literal["succeeded", "partial", "unmatched", "not_applicable", "unsupported", "failed"]
    scope: str | None = None
    required: bool = False
    resolved_year: int | None = None
    expected: int | None = None
    matched: int | None = None
    unmatched: int | None = None
    missing_identifier: int | None = None
    not_applicable: int | None = None
    source_rows: int | None = None
    source_status: Literal["available", "empty", "missing", "unknown"] = "unknown"
    updated: int | None = None
    error_type: str | None = None
    error: str | None = None
    error_truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _record_outcome(repo, outcome: EnrichmentOutcome) -> dict[str, Any]:
    if not hasattr(repo, "_enrichment_outcomes"):
        repo._enrichment_outcomes = []
    document = outcome.to_dict()
    repo._enrichment_outcomes.append(document)
    level = logging.WARNING if outcome.status in {"partial", "unmatched", "unsupported", "failed"} else logging.INFO
    logger.log(
        level, "Enrichment %s for %s scope=%s requested_year=%s resolved_year=%s matched=%s unmatched=%s missing_identifier=%s source_rows=%s error=%s",
        outcome.status, outcome.dataset, outcome.scope, outcome.requested_year, outcome.resolved_year,
        outcome.matched, outcome.unmatched, outcome.missing_identifier, outcome.source_rows, outcome.error,
        extra={"enrichment_outcome": document},
    )
    return document


def _record_matches(repo, dataset, year, resolved_year, scope, *, matched,
                    unmatched, missing_identifier, source_rows, not_applicable=None):
    """Retain the exact counters observed by an entity-join loop."""
    return _record_outcome(repo, EnrichmentOutcome(
        dataset=dataset, requested_year=year, resolved_year=resolved_year, scope=scope,
        status=(("partial" if unmatched or missing_identifier else "succeeded") if matched else
                "not_applicable" if not_applicable and not (unmatched or missing_identifier) else "unmatched"),
        expected=matched + unmatched + missing_identifier,
        matched=matched, unmatched=unmatched, missing_identifier=missing_identifier,
        not_applicable=not_applicable,
        source_rows=source_rows, source_status="available" if source_rows else "empty",
        updated=matched,
    ))


def enricher(name: str):
    def wrap(cls):
        if not issubclass(cls, Enricher):
            raise TypeError("enricher must extend Enricher")
        _registry[name] = cls()
        return cls

    return wrap


def run_enrichment(
    repo, cfg_path: str, year: int, *, datasets: list[str] | None = None
):
    """Run optional enrichers and retain failures/unsupported selections in stats.

    Existing enricher result keys are preserved. ``outcome`` contains observed
    coverage; null counts mean the enricher did not report that coverage.
    """
    selected = list(_registry) if datasets is None else datasets
    stats = {}
    for name in selected:
        enr = _registry.get(name)
        if not enr:
            stats[name] = {"outcome": _record_outcome(repo, EnrichmentOutcome(
                name, year, "unsupported", error="Enricher is not registered",
            ))}
            continue
        start = len(getattr(repo, "_enrichment_outcomes", ()))
        try:
            result = enr.apply(repo, cfg_path, year)
            observed = getattr(repo, "_enrichment_outcomes", ())[start:]
            outcome = observed[-1] if observed else _record_outcome(repo, EnrichmentOutcome(
                name, year, "succeeded", resolved_year=result.get("year"), updated=result.get("updated"),
            ))
            stats[name] = {**result, "outcome": dict(outcome)}
        except Exception as e:
            stats[name] = {"outcome": _record_outcome(repo, EnrichmentOutcome(
                name, year, "failed", error_type=type(e).__name__, error=str(e)[:4096],
                error_truncated=len(str(e)) > 4096,
                source_status="missing" if isinstance(e, FileNotFoundError) else "unknown",
            ))}
    return stats
