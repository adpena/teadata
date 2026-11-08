"""Read and query exact, dated AskTED directory archives offline."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from csv import reader
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
from io import StringIO
import re
from types import MappingProxyType
from typing import Any, Literal

from .dataset_metadata import DataPeriod
from .identifiers import canonical_source_identifier
from .provenance import ProvenanceStore, validate_record_graph
from .scripts.download_askted import ARCHIVE_URL, ArchivePage, validate_archive_header, validate_archive_response
from .selectors import EntitySelectionError, _EntityNames

Scope = Literal["district", "campus"]
Season = Literal["fall", "spring"]
_MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
_MAX_INDEX_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class DirectorySource:
    school_year_end: int
    season: Season
    publisher_declared_as_of: str
    receipt: str
    index_receipt: str
    sha256: str
    publisher: str = "Texas Education Agency"
    source_url: str = ARCHIVE_URL
    definitions_url: str = "https://tea.texas.gov/reports-and-data/school-performance/accountability-research/exploring-askted.pdf"
    data_origin: Literal["publisher_reported_directory"] = "publisher_reported_directory"


def _archive_bytes(store: ProvenanceStore, artifact: Mapping[str, Any], limit: int) -> bytes:
    size, digest = artifact.get("size_bytes"), artifact.get("sha256")
    if (type(size) is not int or not 0 < size <= limit
            or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
        raise ValueError("AskTED archive object differs from its content or size contract")
    with (store.root / "objects" / digest).open("rb") as stream:
        body = stream.read(size + 1)
    if len(body) != size or sha256(body).hexdigest() != digest:
        raise ValueError("AskTED archive object differs from its receipt")
    return body


def _verified_archive(store: ProvenanceStore, receipt: str, *, year: int, season: Season,
                      expected_sha256: str) -> tuple[DirectorySource, str]:
    if type(year) is not int or not 1000 < year < 10_000 or season not in ("fall", "spring"):
        raise ValueError("Select an ending school year and an AskTED fall or spring edition")
    record = store.read_record(receipt)
    params = record.get("parameters", {})
    index_receipt = params.get("archive_index_record")
    if not isinstance(index_receipt, str) or index_receipt == receipt:
        raise ValueError("AskTED requires a distinct archive index receipt")
    graph = validate_record_graph(store.read_record, [receipt, index_receipt])
    if set(graph) != {receipt, index_receipt}:
        raise ValueError("AskTED archive receipts must be direct publisher downloads")
    index = graph[index_receipt]
    for item, method in ((record, "POST"), (index, "GET")):
        if (item["kind"] != "acquisition" or item["dataset"] != "public_school_directory"
                or item.get("section") != "data_sources" or item.get("method") != "download"
                or item.get("requested_url") != ARCHIVE_URL or item.get("resolved_url") != ARCHIVE_URL
                or (item.get("requested_year"), item.get("resolved_year")) != (year, year)
                or item.get("parameters", {}).get("http_method") != method
                or item.get("parameters", {}).get("http_status") != 200):
            raise ValueError("AskTED needs exact downloaded archive index and directory receipts")
    if index.get("response_headers", {}).get("content-type", "").split(";", 1)[0].strip().lower() != "text/html":
        raise ValueError("AskTED archive index must be HTML")
    artifact = record["artifact"]
    if artifact.get("sha256") != expected_sha256:
        raise ValueError("AskTED directory differs from its pinned source bytes")
    parser = ArchivePage()
    parser.feed(_archive_bytes(store, index["artifact"], _MAX_INDEX_BYTES).decode("utf-8-sig"))
    parser.close()
    matches = [item for item in parser.archives if (item.end_year, item.season) == (year, season)]
    if len(matches) != 1:
        raise ValueError("AskTED archive index must contain exactly one selected year and season")
    selected, = matches
    metadata = record["source_metadata"]
    if ((params.get("school_year_end"), params.get("season"), params.get("publisher_declared_as_of"))
            != (year, season, selected.declared_as_of) or metadata.get("version") != f"{year}-{season}"
            or DataPeriod.from_dict(metadata.get("period")) != DataPeriod("as_of", as_of=selected.declared_as_of)):
        raise ValueError(f"AskTED directory receipt differs from the reviewed publisher {season} edition")
    validate_archive_response(record.get("response_headers", {}), selected)
    directory = _archive_bytes(store, artifact, _MAX_ARCHIVE_BYTES).decode("utf-8-sig")
    validate_archive_header(next(reader(StringIO(directory, newline=""), strict=True), []), selected)
    return DirectorySource(year, season, selected.declared_as_of, receipt, index_receipt, expected_sha256), directory


class DirectoryField(StrEnum):
    NAME = "name"
    SITE_CITY = "site_city"
    MAILING_CITY = "mailing_city"
    ADMINISTRATIVE_COUNTY_NUMBER = "administrative_county_number"
    ADMINISTRATIVE_COUNTY_NAME = "administrative_county_name"
    ESC_REGION_SERVED = "esc_region_served"
    ESC_REGION_PEIMS = "esc_region_peims"
    ESC_REGION_GEOGRAPHIC = "esc_region_geographic"


_PLACE_FIELDS = {
    DirectoryField.ADMINISTRATIVE_COUNTY_NUMBER: "County Number",
    DirectoryField.ADMINISTRATIVE_COUNTY_NAME: "County Name",
    DirectoryField.ESC_REGION_SERVED: "ESC Region Served",
    DirectoryField.ESC_REGION_PEIMS: "ESC Region PEIMS",
    DirectoryField.ESC_REGION_GEOGRAPHIC: "ESC Region Geographic",
}
_CODE_WIDTHS = {DirectoryField.ADMINISTRATIVE_COUNTY_NUMBER: 3, DirectoryField.ESC_REGION_SERVED: 2,
                DirectoryField.ESC_REGION_PEIMS: 2, DirectoryField.ESC_REGION_GEOGRAPHIC: 2}


def _code(value: str | int, width: int) -> str | None:
    identifier = canonical_source_identifier(value, width=width)
    return identifier[1:] if identifier else None


def _place_code(field: DirectoryField, value: str | int) -> str | None:
    width = _CODE_WIDTHS[field]
    code = _code(value, width)
    if code and int(code) > (254 if width == 3 else 20):
        raise ValueError(f"{field} is outside the Texas publisher code range: {value!r}")
    return code


@dataclass(frozen=True, slots=True)
class DirectoryObservation:
    """Exact logical CSV record and the tokens used by one scoped entity."""

    record: int
    raw_identifier: str
    raw_district_number: str
    fields: Mapping[DirectoryField, str]


@dataclass(frozen=True, slots=True)
class DirectoryEntry:
    identifier: str
    observations: tuple[DirectoryObservation, ...]
    parent_observations: tuple[DirectoryObservation, ...] | None = None

    def values(self, field: DirectoryField) -> tuple[str, ...]:
        """Distinct nonblank values. A multi-valued field has a source conflict."""
        if not isinstance(field, DirectoryField):
            raise TypeError("Select a DirectoryField")
        rows = self.parent_observations if field in _PLACE_FIELDS and self.parent_observations else self.observations
        raw = {row.fields[field].strip() for row in rows}
        raw.discard("")
        if field in _CODE_WIDTHS:
            values = {_place_code(field, token) for token in raw}
            return tuple(sorted(value for value in values if value is not None))
        return tuple(sorted(raw))

    def value(self, field: DirectoryField) -> str | None:
        """Return the sole nonblank value, or None for a blank or conflict."""
        values = self.values(field)
        return values[0] if len(values) == 1 else None


@dataclass(frozen=True, slots=True)
class DirectoryCohort:
    scope: Scope
    field: DirectoryField
    value: str
    source: DirectorySource
    ids: tuple[str, ...]
    missing_ids: tuple[str, ...]
    conflicting_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DirectoryName:
    name: str
    source: DirectorySource
    record: int


@dataclass(frozen=True, slots=True)
class DirectoryNameMatch:
    identifier: str
    site_city: str | None
    administrative_county: str | None
    anchor_source: DirectorySource
    occurrences: tuple[DirectoryName, ...]


@dataclass(frozen=True, slots=True)
class AskTEDDirectory:
    source: DirectorySource
    districts: Mapping[str, DirectoryEntry]
    campuses: Mapping[str, DirectoryEntry]

    def entries(self, scope: Scope) -> Mapping[str, DirectoryEntry]:
        if scope == "district":
            return self.districts
        if scope == "campus":
            return self.campuses
        raise ValueError("Select district or campus scope")

    def cohort(self, scope: Scope, *, field: DirectoryField, value: str | int) -> DirectoryCohort:
        """Select an administrative county or one explicit district ESC assignment."""
        entries = self.entries(scope)
        if not isinstance(field, DirectoryField) or field not in _CODE_WIDTHS:
            raise ValueError("Select administrative county number or one of the three ESC region fields")
        target = _place_code(field, value)
        if target is None:
            raise ValueError("A nonblank county or ESC code is required")
        matched, missing, conflicts = [], [], []
        for identifier, entry in entries.items():
            values = entry.values(field)
            if len(values) > 1:
                conflicts.append(identifier)
            elif not values:
                missing.append(identifier)
            elif values[0] == target:
                matched.append(identifier)
        return DirectoryCohort(scope, field, target, self.source, tuple(matched), tuple(missing), tuple(conflicts))

    def resolve_name(self, scope: Scope, name: str, *,
                     aliases_from: Sequence[AskTEDDirectory] = ()) -> DirectoryNameMatch:
        """Resolve an exact name using this archive and explicitly supplied name observations."""
        anchor = self.entries(scope)
        directories = (self, *aliases_from)
        if any(not isinstance(item, AskTEDDirectory) for item in directories):
            raise TypeError("aliases_from must contain pinned AskTEDDirectory objects")
        if len({item.source.receipt for item in directories}) != len(directories):
            raise ValueError("Use each AskTED archive once in name selection")
        if any(item.source.publisher_declared_as_of > self.source.publisher_declared_as_of for item in directories[1:]):
            raise ValueError("Historical name archives must be dated on or before the selected edition")
        rosters = tuple((directory.source, directory.entries(scope)) for directory in directories)
        labels: dict[tuple[str, str], str] = {}
        for _, entries in rosters:
            for identifier, entry in entries.items():
                for label in entry.values(DirectoryField.NAME):
                    labels.setdefault((label.casefold(), identifier), label)

        def evidence(label: str, identifier: str) -> tuple[DirectoryName, ...]:
            normalized = label.strip().casefold()
            return tuple(
                DirectoryName(row.fields[DirectoryField.NAME], source, row.record)
                for source, entries in rosters if identifier in entries
                for row in entries[identifier].observations
                if row.fields[DirectoryField.NAME].strip().casefold() == normalized
            )

        candidates = _EntityNames(((label, key[1]) for key, label in labels.items()), str)
        try:
            identifier = candidates.resolve(name, scope=scope)
        except EntitySelectionError as error:
            contexts: dict[tuple[str, str], dict[str, object]] = {}
            for label, candidate in error.candidates:
                city, county = _place(anchor.get(candidate))
                fields: dict[str, object] = {
                    "name_occurrences": [asdict(item) for item in evidence(label, candidate)],
                }
                if city is not None:
                    fields["site_city"] = city
                if county is not None:
                    fields["administrative_county"] = county
                if city is not None or county is not None:
                    fields["place_source"] = asdict(self.source)
                contexts[label, candidate] = fields
            raise EntitySelectionError(scope, name, error.reason, error.candidates, contexts=contexts) from None
        city, county = _place(anchor.get(identifier))
        return DirectoryNameMatch(identifier, city, county, self.source, evidence(name, identifier))


def _place(entry: DirectoryEntry | None) -> tuple[str | None, str | None]:
    if entry is None:
        return None, None
    return entry.value(DirectoryField.SITE_CITY), entry.value(DirectoryField.ADMINISTRATIVE_COUNTY_NAME)


def _projection(prefix: str) -> dict[DirectoryField, str]:
    return {DirectoryField.NAME: f"{prefix} Name", DirectoryField.SITE_CITY: f"{prefix} Site City",
            DirectoryField.MAILING_CITY: f"{prefix} City", **_PLACE_FIELDS}


def _archive_rows(text: str) -> Iterator[tuple[int, str, str | None, dict[str, str]]]:
    """Validate and project source-order rows before grouping them by ID."""
    rows = reader(StringIO(text, newline=""), strict=True)
    header = next(rows, [])
    district, campus = _projection("District"), _projection("School")
    required = {"District Number", "School Number", *district.values(), *campus.values()}
    if missing := required - set(header):
        raise ValueError(f"AskTED directory place columns are missing: {sorted(missing)}")
    positions = {field: header.index(field) for field in sorted(required)}
    for record, row in enumerate(rows, 2):
        if not row:
            continue
        if len(row) != len(header):
            raise ValueError(f"AskTED directory record {record} has the wrong column count")
        raw = {field: row[index] for field, index in positions.items()}
        try:
            district_id, campus_id = _code(raw["District Number"], 6), _code(raw["School Number"], 9)
            for field in _CODE_WIDTHS:
                _place_code(field, raw[_PLACE_FIELDS[field]])
        except ValueError as error:
            raise ValueError(f"AskTED directory record {record}: {error}") from error
        if not district_id:
            raise ValueError(f"AskTED directory record {record} has no district identifier")
        if campus_id and campus_id[:6] != district_id:
            raise ValueError(f"AskTED campus and district IDs disagree at record {record}")
        if not campus_id and any(raw[campus[field]].strip() for field in
                                 (DirectoryField.NAME, DirectoryField.SITE_CITY, DirectoryField.MAILING_CITY)):
            raise ValueError(f"AskTED directory record {record} has school data without a campus identifier")
        yield record, district_id, campus_id, raw


def read_askted_directory(store: ProvenanceStore, receipt: str, *, year: int, season: Season,
                          expected_sha256: str) -> AskTEDDirectory:
    """Bind directory context to a publisher-declared season and verified receipt offline."""
    source, text = _verified_archive(store, receipt, year=year, season=season, expected_sha256=expected_sha256)
    district, campus = _projection("District"), _projection("School")
    district_rows: dict[str, list[DirectoryObservation]] = defaultdict(list)
    campus_rows: dict[str, list[DirectoryObservation]] = defaultdict(list)
    for record, district_id, campus_id, raw in _archive_rows(text):
        raw_district, raw_campus = raw["District Number"], raw["School Number"]
        district_rows[district_id].append(DirectoryObservation(record, raw_district, raw_district,
            MappingProxyType({field: raw[column] for field, column in district.items()})))
        if campus_id:
            campus_rows[campus_id].append(DirectoryObservation(record, raw_campus, raw_district,
                MappingProxyType({field: raw[column] for field, column in campus.items()})))
    districts = {key: DirectoryEntry(key, tuple(district_rows[key])) for key in sorted(district_rows)}
    campuses = {key: DirectoryEntry(key, tuple(campus_rows[key]), districts[key[:6]].observations)
                for key in sorted(campus_rows)}
    return AskTEDDirectory(source, MappingProxyType(districts), MappingProxyType(campuses))
