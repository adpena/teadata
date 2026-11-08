"""Read retained statewide AskTED lifecycle observations with exact source identity."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from hashlib import sha256
from io import StringIO
import re
from typing import Literal

from .identifiers import canonical_source_identifier
from .provenance import ProvenanceStore, validate_record_graph


_PORTAL = "https://tealprod.tea.state.tx.us/Tea.AskTed.Web/Forms/ReportSelection.aspx"
_VIEWER = "https://tealprod.tea.state.tx.us/Tea.AskTed.Web/Forms/ReportViewer.aspx"
_MAX_BYTES = 8 * 1024**2
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_UNRESOLVED_DATE = date(1800, 1, 1)


class LifecycleReportKind(StrEnum):
    PUBLIC_CAMPUS_CLOSURES = "public_campus_closures"
    PUBLIC_DISTRICT_CLOSURES = "public_district_closures"
    CHARTER_CAMPUS_CLOSURES = "charter_campus_closures"
    DISTRICT_CONSOLIDATIONS_ANNEXATIONS = "public_district_consolidations_annexations"


class LifecycleEventKind(StrEnum):
    OBSOLETE = "obsolete"
    DORMANT = "dormant"
    CONSOLIDATION = "consolidation"
    ANNEXATION = "annexation"


Scope = Literal["campus", "district"]
ParticipantRole = Literal["subject", "official_result", "participant", "annexing", "annexed"]


@dataclass(frozen=True, slots=True)
class LifecycleParticipant:
    identifier: str
    name: str
    role: ParticipantRole


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    """One source row. Its record number is scoped to the enclosing LifecycleReport."""

    source_record: int
    scope: Scope
    kind: LifecycleEventKind
    date_label: str
    raw_date: str
    event_date: date | None
    participants: tuple[LifecycleParticipant, ...]
    raw: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LifecycleReport:
    kind: LifecycleReportKind
    printed_date: date
    receipt: str
    sha256: str
    events: tuple[LifecycleEvent, ...]


@dataclass(frozen=True, slots=True)
class _Layout:
    title: str
    category: Literal["school", "district", "charter"]
    sort: str | None
    scope: Scope

    @property
    def heading(self) -> str:
        return f"{self.title} Sorted by {self.sort}" if self.sort else self.title


_LAYOUTS = {
    LifecycleReportKind.PUBLIC_CAMPUS_CLOSURES: _Layout("All Public Schools Closed", "school", "School Name", "campus"),
    LifecycleReportKind.PUBLIC_DISTRICT_CLOSURES: _Layout(
        "All Public School Districts Closed", "district", "District Name", "district"),
    LifecycleReportKind.CHARTER_CAMPUS_CLOSURES: _Layout("All Charter Schools Closed", "charter", "School Name", "campus"),
    LifecycleReportKind.DISTRICT_CONSOLIDATIONS_ANNEXATIONS: _Layout(
        "All Public School District Consolidations and Annexations", "district", None, "district"),
}


def _participant(text: str, scope: Scope, role: ParticipantRole) -> LifecycleParticipant:
    code = r"[0-9]{3}-[0-9]{3}-[0-9]{3}" if scope == "campus" else r"[0-9]{3}-[0-9]{3}"
    match = re.fullmatch(rf"(.*) \(({code})\)", text)
    if match is None:
        raise ValueError(f"Invalid {scope} label and identifier")
    identifier = canonical_source_identifier(match[2].replace("-", ""), width=9 if scope == "campus" else 6)
    if identifier is None:
        raise ValueError(f"Missing {scope} identifier")
    return LifecycleParticipant(identifier[1:], match[1], role)


def _event_date(raw: str) -> date | None:
    match = re.fullmatch(r"([A-Za-z]{3}) ([0-9]{2}), ([0-9]{4})", raw)
    if match is None or match[1] not in _MONTHS:
        raise ValueError("Malformed publisher event date")
    parsed = date(int(match[3]), _MONTHS.index(match[1]) + 1, int(match[2]))
    return None if parsed == _UNRESOLVED_DATE else parsed


def _event(row: tuple[str, ...], layout: _Layout, record: int) -> LifecycleEvent:
    if layout.sort:
        if row[3] != ("School" if layout.scope == "campus" else "District") or row[5] != "Type":
            raise ValueError("Unexpected closure columns")
        kinds = {"Obsolete Date": LifecycleEventKind.OBSOLETE, "Dormant Date": LifecycleEventKind.DORMANT}
        label, raw_date = row[7:9]
        if label not in kinds:
            raise ValueError("Unexpected closure date label")
        kind = kinds[label]
        participants = (_participant(row[4], layout.scope, "subject"),)
    else:
        if (row[3], row[5], row[7], row[9]) != ("County", "Official Name", "Region", "Effective "):
            raise ValueError("Unexpected district reorganization columns")
        label, raw_date = row[9:11]
        roles: tuple[ParticipantRole, ParticipantRole]
        if row[11] == "Consolidated Districts":
            kind, roles = LifecycleEventKind.CONSOLIDATION, ("participant", "participant")
        elif row[11] == "Annexing District \rAnnexed District ":
            kind, roles = LifecycleEventKind.ANNEXATION, ("annexing", "annexed")
        else:
            raise ValueError("Unexpected district reorganization roles")
        participants = (_participant(row[6], "district", "official_result"),
                        _participant(row[12], "district", roles[0]),
                        _participant(row[13], "district", roles[1]))
    return LifecycleEvent(record, layout.scope, kind, label, raw_date, _event_date(raw_date), participants, row)


def _parse(raw: bytes, layout: _Layout, printed_date: date) -> tuple[LifecycleEvent, ...]:
    events = []
    stream = csv.reader(StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
    width = 11 if layout.sort else 16
    printed = f"Printed Date:  {printed_date.month}/{printed_date.day}/{printed_date.year}"
    for record, cells in enumerate(stream, 1):
        row = tuple(cells)
        if (len(row) != width or row[:3] != ("AskTED", layout.heading, layout.heading)
                or row[-2] != printed or re.fullmatch(r"Page -?[0-9]+ of [1-9][0-9]*", row[-1]) is None):
            raise ValueError(f"AskTED lifecycle report layout or print date differs at record {record}")
        try:
            events.append(_event(row, layout, record))
        except ValueError as error:
            raise ValueError(f"AskTED lifecycle record {record}: {error}") from error
    if not events:
        raise ValueError("AskTED lifecycle report has no records")
    return tuple(events)


def read_lifecycle_report(store: ProvenanceStore, receipt: str, *, report: LifecycleReportKind,
                          expected_sha256: str) -> LifecycleReport:
    """Return one exact publisher export in source order, with logical record numbers.

    The report print date and individual lifecycle dates are distinct. The
    unresolved 1800 date token retains its raw value and has no usable date.
    """
    if not isinstance(report, LifecycleReportKind):
        raise TypeError("Select a LifecycleReportKind")
    layout = _LAYOUTS[report]
    record = store.read_record(receipt)
    artifact = record.get("artifact", {})
    size = artifact.get("size_bytes")
    if artifact.get("sha256") != expected_sha256 or type(size) is not int or not 0 < size <= _MAX_BYTES:
        raise ValueError("AskTED lifecycle report differs from its content pin or size limit")
    metadata = record.get("source_metadata", {})
    parameters = record.get("parameters", {})
    printed = parameters.get("publisher_printed_date")
    if (record["kind"] != "acquisition" or record["method"] != "download"
            or record["dataset"] != f"reference:askted_{report}" or record["section"] != "reference"
            or (record["requested_year"], record["resolved_year"]) != (None, None)
            or record["requested_url"] != _VIEWER or record["resolved_url"] != _VIEWER
            or metadata.get("source_url") != _PORTAL or metadata.get("publisher") != "Texas Education Agency"
            or metadata.get("period") != {"kind": "unknown"} or metadata.get("references") != []
            or metadata.get("provenance_ids") != [] or "derived_from" in record
            or parameters.get("category") != layout.category or parameters.get("sort") != layout.sort
            or parameters.get("publisher_report_title") != layout.title
            or parameters.get("report_type") != "rpt" or parameters.get("export_format") != "CharacterSeparatedValues"
            or parameters.get("range") != "all" or parameters.get("export_attachment_name") != "crViewer.csv"
            or not isinstance(printed, str) or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", printed) is None
            or metadata.get("version") != f"printed-{printed}"):
        raise ValueError("AskTED lifecycle receipt differs from the selected public export")
    printed_date = date.fromisoformat(printed)
    validate_record_graph(store.read_record, [receipt])
    path = store.root / "objects" / expected_sha256
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError("AskTED lifecycle object size differs from the selected receipt")
    with path.open("rb") as stream:
        raw = stream.read(size + 1)
    if len(raw) != size or sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("AskTED lifecycle bytes differ from the selected receipt")
    return LifecycleReport(report, printed_date, receipt, expected_sha256, _parse(raw, layout, printed_date))
