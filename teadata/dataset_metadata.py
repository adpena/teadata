"""Declared source metadata, distinct from observations made during acquisition."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import base64
from datetime import date, datetime
from hashlib import sha256
from functools import cached_property
import json
import re
from typing import Any, Literal, Mapping, get_args
from urllib.parse import urlparse

from .values import EstimationPolicy, MeasurementUnit


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


def _digest_text(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Expected a SHA-256 identity")
    return value


def _record_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


COLUMN_SCHEMA_MAX_BYTES = 8 * 1024 * 1024
EDITION_METADATA_MAX_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class ColumnSchemaRef:
    """Exact external crosswalk bytes; distinct from a native Parquet leaf."""

    sha256: str
    size_bytes: int
    location: str

    def __post_init__(self) -> None:
        _digest_text(self.sha256)
        if type(self.size_bytes) is not int or not 0 < self.size_bytes <= COLUMN_SCHEMA_MAX_BYTES:
            raise ValueError("Column-schema resource exceeds its finite metadata byte budget")
        if not isinstance(self.location, str) or not self.location:
            raise ValueError("Column-schema resource location is required")
        if "://" in self.location:
            validate_http_url(self.location)
        elif (self.location in {".", ".."} or any(c in self.location for c in "/\\:")):
            raise ValueError("Column-schema resource requires an adjacent filename or HTTP(S) URL")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ColumnSchemaRef:
        if not isinstance(raw, Mapping) or set(raw) != {"sha256", "size_bytes", "location"}:
            raise ValueError("Unsupported column-schema resource descriptor")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EditionMetadataRef:
    """One bounded edition's catalog metadata, never a native data artifact."""

    sha256: str
    size_bytes: int
    location: str

    def __post_init__(self) -> None:
        # Use the same identity/location contract as the existing metadata resource.
        ColumnSchemaRef(self.sha256, self.size_bytes, self.location)
        if self.size_bytes > EDITION_METADATA_MAX_BYTES:
            raise ValueError("Edition metadata exceeds its finite metadata byte budget")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> EditionMetadataRef:
        if not isinstance(raw, Mapping) or set(raw) != {"sha256", "size_bytes", "location"}:
            raise ValueError("Unsupported edition-metadata resource descriptor")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EditionKey:
    """Exact source revision, separate from publisher edition and leaf identity."""

    dataset: str
    edition: str
    revision_sha256: str
    scope: str
    table: str
    grain: str

    def __post_init__(self) -> None:
        _digest_text(self.revision_sha256)
        if any(not isinstance(value, str) or not value.strip()
               for value in (self.dataset, self.edition, self.scope, self.table, self.grain)):
            raise ValueError("Edition keys require explicit dataset/edition/scope/table/grain")

    @staticmethod
    def source_revision(input_sha256: tuple[str, ...]) -> str:
        """Ordered upstream bytes, independent of receipts, time and this key."""
        if not input_sha256:
            raise ValueError("A source revision requires input bytes")
        return sha256(_record_json(list(dict.fromkeys(_digest_text(v) for v in input_sha256))).encode()).hexdigest()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> EditionKey:
        _known_fields(raw, set(cls.__dataclass_fields__), "edition key")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def identity(self) -> str:
        return sha256(_record_json(self.to_dict()).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class NativeSourceRef:
    """A native release and its admitted editions in an entity build."""

    location: str
    manifest_sha256: str
    build_id: str
    editions: tuple[EditionKey, ...]

    def __post_init__(self) -> None:
        _digest_text(self.manifest_sha256)
        _digest_text(self.build_id)
        location = self.location
        if not isinstance(location, str) or not location or any(c.isspace() or ord(c) < 32 for c in location):
            raise ValueError("Native manifest location is required")
        if any(c in location for c in "\\?#"):
            raise ValueError("Native manifest location cannot contain a query, fragment or backslash")
        if "://" in location:
            validate_http_url(location)
        elif not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.json", location) or location.lower() == "release.json":
            raise ValueError("Native manifest requires a distinct adjacent JSON filename or an immutable HTTP(S) URL")
        if (not isinstance(self.editions, tuple) or not self.editions
                or any(not isinstance(edition, EditionKey) for edition in self.editions)
                or len(set(self.editions)) != len(self.editions)):
            raise ValueError("Native source requires distinct exact editions")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> NativeSourceRef:
        if (not isinstance(raw, Mapping) or set(raw) != {"location", "manifest_sha256", "build_id", "editions"}
                or not isinstance(raw["editions"], list)
                or any(not isinstance(edition, dict) or set(edition) != set(EditionKey.__dataclass_fields__)
                       for edition in raw["editions"])):
            raise ValueError("Invalid native source descriptor")
        return cls(raw["location"], raw["manifest_sha256"], raw["build_id"],
                   tuple(EditionKey.from_dict(edition) for edition in raw["editions"]))

    def to_dict(self) -> dict[str, Any]:
        return {"location": self.location, "manifest_sha256": self.manifest_sha256, "build_id": self.build_id,
                "editions": [edition.to_dict() for edition in self.editions]}


def parse_native_sources(raw: object) -> tuple[NativeSourceRef, ...]:
    if not isinstance(raw, list):
        raise ValueError("Native sources must be a list of exact descriptors")
    sources = tuple(NativeSourceRef.from_dict(item) for item in raw)
    editions = [edition for source in sources for edition in source.editions]
    locations = {source.location if "://" in source.location else source.location.casefold() for source in sources}
    if (len(locations) != len(sources) or len({source.manifest_sha256 for source in sources}) != len(sources)
            or len(set(editions)) != len(editions)):
        raise ValueError("Native sources cannot repeat a location, manifest or exact edition")
    return sources


@dataclass(frozen=True)
class FieldColumnSupport:
    field: str
    raw: str
    status: str | None = None
    auxiliary: tuple[str, ...] = ()


@dataclass(frozen=True)
class LeafColumnSupport:
    """One optional physical partition; technical columns are not source headers."""

    columns: tuple[str, ...]
    fields: tuple[FieldColumnSupport, ...]
    context: tuple[str, ...]

    @classmethod
    def from_contract(cls, contract: Mapping[str, Any]) -> LeafColumnSupport | None:
        if "column_support" not in contract:
            return None
        raw = contract["column_support"]
        columns = contract["columns"]
        if (not isinstance(raw, dict) or set(raw) != {"fields", "context"}
                or not isinstance(raw["fields"], dict) or not isinstance(raw["context"], list)
                or not isinstance(columns, list) or not columns
                or any(not isinstance(name, str) or not name for name in columns) or len(set(columns)) != len(columns)):
            raise ValueError("Invalid leaf column-support map")
        fields = []
        for name, entry in raw["fields"].items():
            if (not isinstance(name, str) or not name or not isinstance(entry, dict)
                    or "raw" not in entry or set(entry) - {"raw", "status", "auxiliary"}
                    or any(not isinstance(entry[key], str) or not entry[key] for key in ("raw", "status") if key in entry)
                    or ("auxiliary" in entry and (not isinstance(entry["auxiliary"], list) or not entry["auxiliary"]
                        or any(not isinstance(value, str) or not value for value in entry["auxiliary"])))):
                raise ValueError("Normalized fields require explicit raw and optional status support")
            fields.append(FieldColumnSupport(name, entry["raw"], entry.get("status"), tuple(entry.get("auxiliary", ()))))
        if any(not isinstance(name, str) or not name for name in raw["context"]):
            raise ValueError("Leaf context requires explicit physical column names")
        support = cls(tuple(columns), tuple(fields), tuple(raw["context"]))
        technical = support.technical_columns
        if (not technical or len(set(technical)) != len(technical)
                or not set(technical) <= set(columns)
                or not {field.field for field in fields} <= set(support.public_columns)):
            raise ValueError("Leaf support columns overlap or reference foreign/public columns")
        return support

    @property
    def technical_columns(self) -> tuple[str, ...]:
        return tuple(name for field in self.fields for name in (field.raw, field.status, *field.auxiliary) if name is not None) + self.context

    @property
    def public_columns(self) -> tuple[str, ...]:
        technical = set(self.technical_columns)
        return tuple(name for name in self.columns if name not in technical)

    def validate_fields(self, fields: set[str], *, count_fields: set[str]) -> None:
        if set(self.public_columns) != fields:
            raise ValueError("Publisher-bound and technical columns must account for the complete physical schema")
        normalized = {field.field: field for field in self.fields}
        if any(name not in normalized or normalized[name].status is None for name in count_fields):
            raise ValueError("Normalized count fields require explicit raw and status support")

    def projection(self, field: str) -> dict[str, Any]:
        result: dict[str, Any] = {"context": list(self.context)}
        for entry in self.fields:
            if entry.field == field:
                result.update(representation="normalized", raw=entry.raw)
                if entry.status is not None:
                    result["status"] = entry.status
                if entry.auxiliary:
                    result["auxiliary"] = list(entry.auxiliary)
                break
        return result


STATEWIDE_SINGLETON_GRAIN = "one statewide observation"


def validate_leaf_keys(contract: Mapping[str, Any]) -> None:
    """The statewide singleton is identified by its edition alone."""
    keys = contract["key_columns"]
    edition = contract["edition"]
    singleton = edition["scope"] == "state" and edition["grain"] == STATEWIDE_SINGLETON_GRAIN
    if singleton:
        if keys != [] or type(contract["rows"]) is not int or contract["rows"] != 1:
            raise ValueError("A statewide singleton requires one row and no physical keys")
    elif (not isinstance(keys, list) or not keys or any(not isinstance(key, str) for key in keys)
          or len(set(keys)) != len(keys) or not set(keys) <= set(contract["columns"])):
        raise ValueError("Leaf grain requires explicit ordered native key columns")


@dataclass(frozen=True)
class ArtifactRef:
    """Transport bytes plus an independent, immutable native leaf contract.

    Contract JSON contains no file hash, final receipt or root identity. The
    native Arrow schema is opaque serialized IPC until selected opening.
    """

    edition: EditionKey
    sha256: str
    size_bytes: int
    location: str
    contract_json: str
    processing_id: str
    output: str

    def __post_init__(self) -> None:
        _digest_text(self.sha256)
        _digest_text(self.processing_id)
        if type(self.size_bytes) is not int or self.size_bytes <= 0 or not isinstance(self.output, str) or not self.output:
            raise ValueError("Artifact references require bytes and a processing output")
        if not isinstance(self.location, str) or not self.location:
            raise ValueError("Artifact location is required")
        if "://" in self.location:
            validate_http_url(self.location)
        elif (self.location in {".", ".."} or "/" in self.location or "\\" in self.location
              or ":" in self.location):
            raise ValueError("Artifact location must be an adjacent filename or explicit HTTP(S) URL")
        contract = json.loads(self.contract_json)
        required = {"edition", "schema", "schema_sha256", "columns", "key_columns", "rows", "inputs", "mapping_sha256",
                    "recipe_sha256", "policy_sha256", "semantics_sha256", "producer"}
        if (not isinstance(contract, dict) or not required <= set(contract)
                or set(contract) - required - {"column_support"} or contract["edition"] != self.edition.to_dict()):
            raise ValueError("Leaf contract does not bind its exact edition")
        schema = base64.b64decode(contract["schema"], validate=True)
        if not schema or sha256(schema).hexdigest() != _digest_text(contract["schema_sha256"]):
            raise ValueError("Leaf native schema identity mismatch")
        columns = contract["columns"]
        if (not isinstance(columns, list) or not columns
                or any(not isinstance(v, str) or not v for v in columns) or len(set(columns)) != len(columns)):
            raise ValueError("Leaf contract requires ordered native columns")
        LeafColumnSupport.from_contract(contract)
        validate_leaf_keys(contract)
        if type(contract["rows"]) is not int or contract["rows"] < 0:
            raise ValueError("Leaf row count must be a nonnegative integer")
        inputs = contract["inputs"]
        if not isinstance(inputs, list) or not inputs:
            raise ValueError("Leaf contract requires upstream input hashes")
        for digest in [*inputs, contract["mapping_sha256"], contract["recipe_sha256"], contract["semantics_sha256"]]:
            _digest_text(digest)
        if contract["policy_sha256"] is not None:
            _digest_text(contract["policy_sha256"])
        producer = contract["producer"]
        if (not isinstance(producer, dict) or set(producer) != {"code_sha256", "toolchain"}
                or not isinstance(producer["toolchain"], dict) or not producer["toolchain"]
                or any(not isinstance(k, str) or not isinstance(v, str) or not v
                       for k, v in producer["toolchain"].items())):
            raise ValueError("Leaf producer requires code and actual toolchain identities")
        _digest_text(producer["code_sha256"])
        object.__setattr__(self, "contract_json", _record_json(contract))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ArtifactRef:
        if set(raw) != {"edition", "sha256", "size_bytes", "location", "contract", "processing_id", "output", "format"} or raw["format"] != "parquet":
            raise ValueError("Unsupported artifact descriptor")
        return cls(EditionKey.from_dict(raw["edition"]), raw["sha256"], raw["size_bytes"], raw["location"],
                   _record_json(raw["contract"]), raw["processing_id"], raw["output"])

    @property
    def contract(self) -> dict[str, Any]:
        return json.loads(self.contract_json)

    @cached_property
    def column_support(self) -> LeafColumnSupport | None:
        return LeafColumnSupport.from_contract(self.contract)

    @cached_property
    def public_columns(self) -> tuple[str, ...]:
        support = self.column_support
        return tuple(self.contract["columns"]) if support is None else support.public_columns

    @property
    def identity(self) -> str:
        return sha256(self.contract_json.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {"edition": self.edition.to_dict(), "sha256": self.sha256, "size_bytes": self.size_bytes,
                "location": self.location, "contract": self.contract, "format": "parquet",
                "processing_id": self.processing_id, "output": self.output}


@dataclass(frozen=True)
class FieldBinding:
    """Physical projection for one exact qualified field; semantics stay canonical."""

    edition_id: str
    field: str
    artifact_id: str
    column: str
    dependencies: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _digest_text(self.edition_id)
        _digest_text(self.artifact_id)
        if any(not isinstance(v, str) or not v for v in (self.field, self.column)):
            raise ValueError("Field bindings require explicit public and physical columns")
        if (not isinstance(self.dependencies, tuple) or len(set(self.dependencies)) != len(self.dependencies)
                or any(not isinstance(pair, tuple) or len(pair) != 2 or not isinstance(pair[1], str)
                       or not pair[1] for pair in self.dependencies)):
            raise ValueError("Field dependencies require distinct qualified field pairs")
        for edition, _ in self.dependencies:
            _digest_text(edition)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FieldBinding:
        if set(raw) != {"edition_id", "field", "artifact_id", "column", "dependencies"}:
            raise ValueError("Invalid field binding")
        return cls(raw["edition_id"], raw["field"], raw["artifact_id"], raw["column"],
                   tuple(tuple(pair) for pair in raw["dependencies"]))

    def to_dict(self) -> dict[str, Any]:
        return {"edition_id": self.edition_id, "field": self.field, "artifact_id": self.artifact_id,
                "column": self.column, "dependencies": [list(pair) for pair in self.dependencies]}


def _parse_school_year(school_year: int | str | None) -> int | None:
    if school_year is None:
        return None
    if type(school_year) is int:
        if 0 <= school_year <= 99:
            return 2000 + school_year
        if 100 <= school_year <= 9999:
            return school_year
    elif isinstance(school_year, str):
        value = school_year.strip()
        if re.fullmatch(r"[0-9]{2}|[0-9]{4}", value):
            year = int(value)
            if len(value) == 2:
                return 2000 + year
            if 2 <= year <= 9999:
                return year
        elif match := re.fullmatch(r"([0-9]{4})[ \t]*[-–][ \t]*([0-9]{2}|[0-9]{4})", value):
            start, suffix = match.groups()
            end = int(start) + 1
            if 2 <= end <= 9999 and int(suffix) == (end % 100 if len(suffix) == 2 else end):
                return end
    if isinstance(school_year, str):
        received = repr(school_year[:80])
    elif type(school_year) is int and -99_999 <= school_year <= 99_999:
        received = str(school_year)
    else:
        received = type(school_year).__name__
    raise ValueError(f"Invalid school_year ({received}). Use an integer ending year such as 2025 or 25, "
                     "or text such as '2025', '25', '2024-25', or '2024–2025'. Range years must be consecutive.")


_EDITION_KEY_GUIDANCE = (
    "Build a key with EditionKey.from_dict() using a row from the native release's catalog.editions(include_metadata=False) "
    "with edition_id removed."
)


def _validate_observation_selection(dataset: str, fields: tuple[str, ...], scope: str,
                                    edition: EditionKey | None) -> None:
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("dataset must be a nonblank source name, for example 'peims_student_enrollment_by_ethnicity'.")
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError("scope must be a nonblank publisher scope, such as 'district', 'campus', or 'state'.")
    if not fields or any(not isinstance(field, str) or not field.strip() for field in fields) or len(set(fields)) != len(fields):
        raise ValueError("Select one or more distinct source fields by their exact nonblank string names.")
    if edition is not None and not isinstance(edition, EditionKey):
        raise TypeError(f"edition must be an EditionKey or None. {_EDITION_KEY_GUIDANCE}")


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
        unit = raw.get("unit")
        if unit is not None:
            try:
                unit = MeasurementUnit(unit).value
            except (TypeError, ValueError) as error:
                raise ValueError("Unknown metric unit") from error
        return cls(
            raw["field"], DataPeriod.from_dict(raw["period"]) if raw.get("period") is not None else None, unit
        )

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "unit": self.unit,
                **({"period": self.period.to_dict()} if self.period is not None else {})}


_SourceQualifierFamily = Literal["sof", "peims_financial", "accountability", "ccmr_outcomes_bonus"]
_SOURCE_QUALIFIER_FAMILIES = frozenset(get_args(_SourceQualifierFamily))


@dataclass(frozen=True)
class SourceQualifiers:
    """Publisher source state, separate from our processing qualification.

    Publication-run identity never asserts one run across every source row.
    Unknown values stay absent; publisher release wording is retained verbatim.
    """

    family: _SourceQualifierFamily
    reference_id: str | None = None
    publisher_release_status: str | None = None
    publication_run_id: str | None = None
    row_run_column: str | None = None
    financial_basis: Literal["actual", "budget"] | None = None
    graduate_class_year: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or self.family not in _SOURCE_QUALIFIER_FAMILIES:
            raise ValueError("Unknown source-qualifier family")
        for name in ("reference_id", "publisher_release_status", "publication_run_id", "row_run_column", "financial_basis"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"Source qualifier {name} must be a nonempty string or unknown")
        if self.family != "sof" and (self.publication_run_id is not None or self.row_run_column is not None):
            raise ValueError("Publication/row run qualifiers require the SOF family")
        if self.financial_basis is not None and (
            self.family != "peims_financial" or self.financial_basis not in {"actual", "budget"}
        ):
            raise ValueError("PEIMS financial basis must be actual or budget")
        if self.graduate_class_year is not None and (
            self.family != "ccmr_outcomes_bonus"
            or type(self.graduate_class_year) is not int
            or not 1 <= self.graduate_class_year <= 9999
        ):
            raise ValueError("Outcomes Bonus graduate_class_year must be a valid year")
        if any(value is not None for value in (self.publisher_release_status, self.publication_run_id, self.financial_basis, self.graduate_class_year)) and self.reference_id is None:
            raise ValueError("Publisher source qualifiers require an existing evidence reference")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SourceQualifiers:
        if not isinstance(raw, Mapping):
            raise ValueError("Source qualifiers must be a mapping")
        family = raw.get("family")
        if not isinstance(family, str) or family not in _SOURCE_QUALIFIER_FAMILIES:
            raise ValueError("Unknown source-qualifier family")
        allowed = {"family", "reference_id", "publisher_release_status"}
        if family == "sof":
            allowed.update(("publication_run_id", "row_run_column"))
        elif family == "peims_financial":
            allowed.add("financial_basis")
        elif family == "ccmr_outcomes_bonus":
            allowed.add("graduate_class_year")
        _known_fields(raw, allowed, "source qualifier")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return {name: value for name, value in asdict(self).items() if value is not None}


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
    source_qualifiers: SourceQualifiers | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> DatasetMetadata:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("dataset metadata must be a mapping")
        _known_fields(raw, set(cls.__dataclass_fields__), "dataset metadata")
        values = dict(raw)
        values["period"] = DataPeriod.from_dict(values.get("period"))
        if values.get("source_qualifiers") is not None:
            values["source_qualifiers"] = SourceQualifiers.from_dict(values["source_qualifiers"])
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
        qualifiers = values.get("source_qualifiers")
        if qualifiers is not None and qualifiers.reference_id is not None and qualifiers.reference_id not in {
            item.reference_id for item in values["references"]
        }:
            raise ValueError("Source qualifiers require an existing evidence reference")
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
        if self.source_qualifiers is not None:
            values["source_qualifiers"] = self.source_qualifiers.to_dict()
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
