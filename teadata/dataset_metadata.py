"""Declared source metadata, distinct from observations made during acquisition."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Mapping
from urllib.parse import urlparse

from .values import EstimationPolicy


def _known_fields(raw: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown {label} fields: {', '.join(sorted(unknown))}")


def _date_text(value: Any) -> str | None:
    if value is None:
        return None
    text = value.isoformat() if isinstance(value, (date, datetime)) else str(value)
    try:
        if "T" in text:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("timestamps must include a timezone")
        else:
            date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"Expected an ISO date or timezone-aware timestamp: {text}"
        ) from exc
    return text


def validate_http_url(value: str | None) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ValueError("Source URLs must be strings")
    parsed = urlparse(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Source URLs must use HTTP(S) without credentials")


@dataclass(frozen=True)
class DataPeriod:
    """A source's reference period; a catalog year alone does not establish one."""

    kind: str = "unknown"
    start_year: int | None = None
    end_year: int | None = None
    year: int | None = None
    as_of: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str):
            raise ValueError("Reference period kind must be a string")
        fields = (self.start_year, self.end_year, self.year, self.as_of)
        if self.kind in {"school_year", "fiscal_year"}:
            if (
                type(self.start_year) is not int
                or type(self.end_year) is not int
                or self.end_year != self.start_year + 1
                or not 1 <= self.start_year < self.end_year <= 9999
                or self.year is not None
                or self.as_of is not None
            ):
                raise ValueError(
                    f"{self.kind} requires consecutive start_year/end_year"
                )
        elif self.kind == "calendar_year":
            if (
                type(self.year) is not int
                or not 1 <= self.year <= 9999
                or any(
                    v is not None for v in (self.start_year, self.end_year, self.as_of)
                )
            ):
                raise ValueError("calendar_year requires only year")
        elif self.kind == "as_of":
            if self.as_of is None or any(v is not None for v in fields[:3]):
                raise ValueError("as_of requires only an ISO date")
            object.__setattr__(self, "as_of", _date_text(self.as_of))
        elif self.kind != "unknown" or any(v is not None for v in fields):
            raise ValueError("Unknown or inconsistent reference period")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> DataPeriod:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("period must be a mapping")
        _known_fields(raw, set(cls.__dataclass_fields__), "period")
        return cls(**raw)

    @property
    def label(self) -> str | None:
        if self.kind in {"school_year", "fiscal_year"}:
            return f"{self.start_year}-{self.end_year}"
        if self.kind == "calendar_year":
            return str(self.year)
        return self.as_of

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class ReferenceDocument:
    reference_id: str
    role: str
    url: str | None = None
    path: str | None = None
    title: str | None = None
    status: str = "available"
    published_at: str | None = None
    provenance_ids: tuple[str, ...] = ()
    reviewed_sha256: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ReferenceDocument:
        _known_fields(raw, set(cls.__dataclass_fields__), "reference document")
        if (
            not isinstance(raw.get("reference_id"), str)
            or not raw["reference_id"].strip()
        ):
            raise ValueError("Reference documents require reference_id")
        if raw.get("role") not in {
            "data_dictionary",
            "glossary",
            "schema",
            "masking_methodology",
            "methodology",
            "source_notes",
        }:
            raise ValueError("Unknown reference-document role")
        if raw.get("status", "available") not in {
            "available",
            "not_found",
            "not_applicable",
            "unreviewed",
        }:
            raise ValueError("Unknown reference-document availability")
        if raw.get("status", "available") == "available" and not (
            raw.get("url") or raw.get("path")
        ):
            raise ValueError(
                "Available reference documents require a URL or portable path"
            )
        if raw.get("path") is not None:
            path = raw["path"]
            if (
                not isinstance(path, str)
                or not path
                or path.startswith(("/", "\\"))
                or ".." in path.split("/")
            ):
                raise ValueError(
                    "Reference paths must be portable paths within the source contract directory"
                )
        validate_http_url(raw.get("url"))
        values = dict(raw)
        reviewed = values.get("reviewed_sha256")
        if reviewed is not None and (not isinstance(reviewed, str) or len(reviewed) != 64 or any(c not in "0123456789abcdef" for c in reviewed)):
            raise ValueError("Reference review must identify exact SHA-256 document bytes")
        values["published_at"] = _date_text(values.get("published_at"))
        ids = values.get("provenance_ids", ())
        if not isinstance(ids, (list, tuple)) or any(
            not isinstance(i, str)
            or len(i) != 64
            or any(c not in "0123456789abcdef" for c in i)
            for i in ids
        ):
            raise ValueError("Reference provenance_ids must be SHA-256 identifiers")
        values["provenance_ids"] = tuple(ids)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            k: list(v) if k == "provenance_ids" else v
            for k, v in asdict(self).items()
            if v is not None
        }


@dataclass(frozen=True)
class MetricDefinition:
    """A semantic field whose reference period can differ from the report."""

    field: str
    period: DataPeriod | None = None
    unit: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> MetricDefinition:
        if not isinstance(raw, Mapping):
            raise ValueError("Metric definitions must be mappings")
        _known_fields(raw, {"field", "period", "unit"}, "metric definition")
        if not isinstance(raw.get("field"), str) or not raw["field"].strip():
            raise ValueError("Metric definitions require a field")
        if raw.get("unit") not in {
            None,
            "count",
            "percent",
            "currency",
            "text",
            "ratio",
            "years",
        }:
            raise ValueError("Unknown metric unit")
        return cls(
            raw["field"], DataPeriod.from_dict(raw["period"]) if raw.get("period") is not None else None, raw.get("unit")
        )

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "unit": self.unit,
                **({"period": self.period.to_dict()} if self.period is not None else {})}


@dataclass(frozen=True)
class DatasetMetadata:
    title: str | None = None
    publisher: str | None = None
    version: str | None = None
    period: DataPeriod = field(default_factory=DataPeriod)
    source_url: str | None = None
    download_url: str | None = None
    published_at: str | None = None
    revised_at: str | None = None
    license: str | None = None
    citation: str | None = None
    identifiers: tuple[str, ...] = ()
    notes: str | None = None
    provenance_ids: tuple[str, ...] = ()
    references: tuple[ReferenceDocument, ...] = ()
    field_year: int | None = None
    value_policy: str | None = None
    metrics: Mapping[str, MetricDefinition] = field(default_factory=dict)
    estimation: EstimationPolicy | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> DatasetMetadata:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("dataset metadata must be a mapping")
        _known_fields(raw, set(cls.__dataclass_fields__), "dataset metadata")
        values = dict(raw)
        values["period"] = DataPeriod.from_dict(values.get("period"))
        if values.get("estimation") is not None:
            values["estimation"] = EstimationPolicy.from_dict(values["estimation"])
        metrics = values.get("metrics", {})
        if not isinstance(metrics, Mapping) or any(
            not isinstance(k, str) or not k.strip() for k in metrics
        ):
            raise ValueError(
                "metrics must map semantic identifiers to field definitions"
            )
        values["metrics"] = {
            key: MetricDefinition.from_dict(value) for key, value in metrics.items()
        }
        references = values.get("references", ())
        if not isinstance(references, (list, tuple)):
            raise ValueError("references must be a list")
        values["references"] = tuple(
            ReferenceDocument.from_dict(item) for item in references
        )
        if len({item.reference_id for item in values["references"]}) != len(
            values["references"]
        ):
            raise ValueError(
                "Reference identifiers must be unique within a dataset version"
            )
        if values.get("estimation") and values["estimation"].reference_id not in {
            item.reference_id for item in values["references"]
        }:
            raise ValueError(
                "Estimation policy requires a source methodology reference"
            )
        if values.get("field_year") is not None and (
            type(values["field_year"]) is not int
            or not 1 <= values["field_year"] <= 9999
        ):
            raise ValueError("field_year must be a valid year")
        if values.get("value_policy") is not None and not isinstance(
            values["value_policy"], str
        ):
            raise ValueError("value_policy must be an identifier")
        for key in ("published_at", "revised_at"):
            values[key] = _date_text(values.get(key))
        ids = values.get("provenance_ids", ())
        if not isinstance(ids, (list, tuple)) or any(
            not isinstance(item, str)
            or len(item) != 64
            or any(c not in "0123456789abcdef" for c in item)
            for item in ids
        ):
            raise ValueError("provenance_ids must be SHA-256 record identifiers")
        values["provenance_ids"] = tuple(ids)
        identifiers = values.get("identifiers", ())
        if not isinstance(identifiers, (list, tuple)) or any(
            not isinstance(item, str) or not item.strip() for item in identifiers
        ):
            raise ValueError(
                "identifiers must be nonempty strings, such as DOIs or catalog identifiers"
            )
        values["identifiers"] = tuple(identifiers)
        for key in ("source_url", "download_url"):
            validate_http_url(values.get(key))
        for key in (
            "title",
            "publisher",
            "version",
            "source_url",
            "download_url",
            "license",
            "citation",
            "notes",
        ):
            if (
                key in values
                and values[key] is not None
                and not isinstance(values[key], str)
            ):
                raise ValueError(f"{key} must be a string")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        values = {k: v for k, v in asdict(self).items() if v is not None}
        values["period"] = self.period.to_dict()
        values["provenance_ids"] = list(self.provenance_ids)
        values["identifiers"] = list(self.identifiers)
        values["references"] = [reference.to_dict() for reference in self.references]
        values["metrics"] = {
            key: value.to_dict() for key, value in self.metrics.items()
        }
        return values


@dataclass(frozen=True)
class ResolvedDataset:
    dataset: str
    section: str
    requested_year: int | None
    resolved_year: int | None
    path: str
    metadata: DatasetMetadata = field(default_factory=DatasetMetadata)
    reader: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_fallback(self) -> bool | None:
        if self.resolved_year is None or self.requested_year is None:
            return None
        return self.resolved_year != self.requested_year

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "section": self.section,
            "requested_year": self.requested_year,
            "resolved_year": self.resolved_year,
            "is_fallback": self.is_fallback,
            "path": self.path,
            "metadata": self.metadata.to_dict(),
            "reader": dict(self.reader),
        }


@dataclass(frozen=True)
class DatasetDefinition:
    """The single configured authority for one dataset version."""

    path: str
    metadata: DatasetMetadata = field(default_factory=DatasetMetadata)
    reader: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "metadata": self.metadata.to_dict(),
            "reader": dict(self.reader),
        }
