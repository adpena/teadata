"""Read the named 2025–26 TPRS Graduation Profile for intake qualification."""

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import re
from typing import Literal, TypeAlias

from .dataset_metadata import DataPeriod
from .manifest import ContractError
from .native_witnesses import _Rows
from .values import ValueStatus

GraduationScope: TypeAlias = Literal["state", "district", "campus"]
Quantity: TypeAlias = Literal["count", "percent"]
ColumnRole: TypeAlias = tuple[Literal["local", "district_benchmark", "state_benchmark"], Quantity]

EDITION = 2026
PERIOD = DataPeriod("school_year", 2024, 2025)
_TITLE = "2025-26 Graduation Profile"
_POPULATION = "Graduates (2024-25 Annual Graduates)"
_HEADERS = {
    "state": ("StateCount", "StatePercent"),
    "district": ("DistrictCount", "DistrictPercent", "StateCount", "StatePercent"),
    "campus": ("CampusCount", "CampusPercent", "DistrictCount", "StateCount"),
}
_LOCAL: tuple[ColumnRole, ...] = (("local", "count"), ("local", "percent"))
_ROLES: dict[GraduationScope, tuple[ColumnRole, ...]] = {
    "state": _LOCAL,
    "district": (*_LOCAL, ("state_benchmark", "count"), ("state_benchmark", "percent")),
    "campus": (*_LOCAL, ("district_benchmark", "count"), ("state_benchmark", "count")),
}
_ROWS = (
    "Total Graduates",
    "By Ethnicity:",
    "African American", "Hispanic", "White", "American Indian", "Asian", "Pacific Islander", "Two or More Races",
    "By Graduation Type:",
    "Minimum H.S. Program", "Recommended H.S. Program/Distinguished Achievement Program",
    "Foundation H.S. Program (No Endorsement)", "Foundation H.S. Program (Endorsement)",
    "Foundation H.S. Program (DLA)", "Texas First Early H.S. Completion Program (Texas First)",
    "",
    "Special Education Graduates", "Economically Disadvantaged Graduates",
    "Emergent Bilingual (EB)/English Learner (EL) Graduates", "At-Risk Graduates", "CTE Completers",
)
_LEGEND = [
    ["-", "Indicates there are no students in the group."],
    ["?", "Indicates that the data for this item were statistically improbable or were reported outside a reasonable range."],
]
_COUNT = re.compile(r"(?:0|[1-9][0-9]*|[1-9][0-9]{0,2}(?:,[0-9]{3})+)")
_PERCENT = re.compile(r"(?:0|[1-9][0-9]{0,2})\.[0-9]%")
_REPORT_TABLE = "Procedure Report: Detailed and/or summarized report"


class _GraduationRows(_Rows):
    """Keep the table and display geometry that the flat row reader discards."""

    def __init__(self, html: str):
        self.table: tuple[int, str | None] | None = None
        self.table_number = 0
        self.in_facts = False
        self.tables: list[tuple[int, str | None] | None] = []
        self.spans: list[list[tuple[str | None, str | None]]] = []
        super().__init__(html)
        if self.table is not None or len(self.rows) != len(self.spans):
            raise ContractError("TPRS Graduation Profile has an incomplete table")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            if self.table is not None:
                raise ContractError("TPRS Graduation Profile has a nested table")
            self.table_number += 1
            self.table = (self.table_number, dict(attrs).get("summary"))
            self.in_facts = len(self.rows) == 4
        if self.in_facts:
            allowed = (tag in {"table", "tbody", "thead", "colgroup", "col", "tr", "th", "td"}
                       if self.cell is None else tag == "br" and len(self.rows) == 4 and not attrs)
            if not allowed:
                raise ContractError("TPRS Graduation Profile has unreviewed markup in its facts table")
            if any(key in {"hidden", "aria-hidden"} or (
                key == "style" and (value or "").strip() != "background-color: #CDE0F0;"
            ) for key, value in attrs):
                raise ContractError("TPRS Graduation Profile facts table has unreviewed visibility")
        if tag == "tr":
            if self.table is None or self.row is not None:
                raise ContractError("TPRS Graduation Profile has a row outside its table")
            self.tables.append(self.table)
            self.spans.append([])
        elif tag in {"td", "th"}:
            if self.row is None or self.cell is not None:
                raise ContractError("TPRS Graduation Profile has an invalid cell boundary")
            dimensions = [(key, value) for key, value in attrs if key in {"colspan", "rowspan"}]
            spans = dict(dimensions)
            if len(spans) != len(dimensions):
                raise ContractError("TPRS Graduation Profile has a duplicate span")
            self.spans[-1].append((spans.get("colspan", "1"), spans.get("rowspan", "1")))
        super().handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag == "tr" and (self.row is None or self.cell is not None):
            raise ContractError("TPRS Graduation Profile has an invalid row boundary")
        if tag == "table":
            if self.table is None or self.row is not None:
                raise ContractError("TPRS Graduation Profile has an incomplete table")
            self.table = None
            self.in_facts = False
        super().handle_endtag(tag)

    def verify_grid(self, width: int) -> None:
        facts = self.tables[4]
        facts_rows = self.tables[4:-3]
        if (facts is None or facts[1] != _REPORT_TABLE
                or any(table != facts for table in facts_rows)
                or facts in (*self.tables[:4], *self.tables[-3:])):
            raise ContractError("TPRS Graduation Profile facts table differs")
        for row, spans in zip(self.rows[4:-3], self.spans[4:-3], strict=True):
            expected = [(str(width), "1")] if len(row) == 1 else [("1", "1")] * width
            if spans != expected:
                raise ContractError("TPRS Graduation Profile facts grid has unexpected spans")


@dataclass(frozen=True, slots=True)
class GraduationCell:
    label: str
    role: ColumnRole
    raw: str
    value: int | Decimal | None
    status: ValueStatus
    reason: str | None
    row: int
    column: int


@dataclass(frozen=True, slots=True)
class GraduationProfile:
    scope: GraduationScope
    identifier: str
    sha256: str
    cells: tuple[GraduationCell, ...]


def _value(raw: str, quantity: Quantity) -> tuple[int | Decimal | None, ValueStatus, str | None]:
    if raw == "-":
        return None, ValueStatus.NOT_APPLICABLE, "no_students_in_group"
    if raw == "?":
        return None, ValueStatus.INVALID, "publisher_outside_reasonable_range"
    if quantity == "count" and _COUNT.fullmatch(raw) and len(raw) <= 16:
        return int(raw.replace(",", "")), ValueStatus.OBSERVED, None
    if quantity == "percent" and _PERCENT.fullmatch(raw) and (percent := Decimal(raw[:-1])) <= 100:
        return percent, ValueStatus.OBSERVED, None
    raise ContractError(f"Unreviewed TPRS Graduation Profile {quantity} token: {raw!r}")


def parse_graduation_profile(raw: bytes, *, scope: GraduationScope, identifier: str) -> GraduationProfile:
    """Bind the retained named HTML. Coordinates are one-based HTML table-row ordinals."""
    if not isinstance(scope, str) or scope not in _ROLES or not isinstance(identifier, str):
        raise ContractError("Unsupported TPRS Graduation Profile geography")
    if scope == "state":
        valid_identifier = identifier == "S"
    else:
        width = 6 if scope == "district" else 9
        valid_identifier = re.fullmatch(rf"[0-9]{{{width}}}", identifier) is not None and identifier != "0" * width
    if not valid_identifier:
        raise ContractError("Invalid literal TPRS Graduation Profile identifier")
    if not isinstance(raw, bytes) or not 0 < len(raw) <= 2 * 1024**2:
        raise ContractError("TPRS Graduation Profile exceeds the bounded HTML size")
    try:
        parsed = _GraduationRows(raw.decode("cp1252"))
    except UnicodeError as error:
        raise ContractError("TPRS Graduation Profile encoding differs") from error
    rows = parsed.rows
    if len(rows) != len(_ROWS) + 9 or rows[:2] != [["", "Texas Education Agency", ""], [_TITLE]]:
        raise ContractError("TPRS Graduation Profile title or layout differs")
    geography, = rows[2] if len(rows[2]) == 1 else ("",)
    identity_matches = geography == "State" if scope == "state" else re.findall(r"\(([0-9]+)\)", geography) == [identifier]
    if not identity_matches or rows[4:6] != [["", *_HEADERS[scope]], [_POPULATION]]:
        raise ContractError("TPRS Graduation Profile identity, column roles, or population differs")
    if rows[-3:-1] != _LEGEND:
        raise ContractError("TPRS Graduation Profile disclosure legend differs")
    roles = _ROLES[scope]
    parsed.verify_grid(len(roles) + 1)
    cells = []
    for number, (expected, row) in enumerate(zip(_ROWS, rows[6:-3], strict=True), 7):
        if expected in {"By Ethnicity:", "By Graduation Type:", ""}:
            if row == [expected]:
                continue
            raise ContractError("TPRS Graduation Profile section boundary differs")
        if len(row) != len(roles) + 1 or row[0] != expected:
            raise ContractError("TPRS Graduation Profile data row differs")
        for column, (role, token) in enumerate(zip(roles, row[1:], strict=True), 2):
            value, status, reason = _value(token, role[1])
            cells.append(GraduationCell(expected, role, token, value, status, reason, number, column))
    return GraduationProfile(scope, identifier, sha256(raw).hexdigest(), tuple(cells))
