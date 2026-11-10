"""Read class-size observations from retained 2025–26 TPRS student reports."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import re
from typing import TYPE_CHECKING

from .native_witnesses import _Rows
from .tprs_students import PERIOD, _check_student_heading, _identifier, _need, _read_student_input, _student_html, definition
from .values import Measurement, TokenRule, ValuePolicy, ValueStatus, measure

if TYPE_CHECKING:
    from .dataset_metadata import DataPeriod
    from .provenance import ProvenanceStore

_TITLE = "Class Size Information"
_CAPTION = re.compile(r"class[ -]size\b", re.I)
_COLUMNS = {"campus": ["Campus", "District", "State"], "district": ["District", "State"], "state": ["State"]}
_FIELDS = {
    "Elementary": {
        "Kindergarten": "class_size_kindergarten_avg_size",
        **{f"Grade {grade}": f"class_size_grade_{grade}_avg_size" for grade in range(1, 7)},
    },
    "Secondary": {
        "English/Language Arts": "class_size_sec_english_avg_size",
        "Foreign Languages": "class_size_sec_for_lng_avg_size",
        "Mathematics": "class_size_sec_math_avg_size",
        "Science": "class_size_sec_sci_avg_size",
        "Social Studies": "class_size_sec_soc_stud_avg_size",
    },
}
_NUMERIC = re.compile(r"(?:0|[1-9][0-9]*)\.[0-9]")
CLASS_SIZE_UNIT = "students_per_class"
CLASS_SIZE_POLICY = ValuePolicy("tprs_class_size_2026", {
    "-": TokenRule(ValueStatus.NOT_APPLICABLE, "no_students"),
    "?": TokenRule(ValueStatus.INVALID, "improbable_or_outside_reporting_range"),
})


@lru_cache(maxsize=1)
def _class_size_legend() -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(row) for row in definition()["legend"] if row[0] in CLASS_SIZE_POLICY.tokens)


class _ReportRows(_Rows):
    """Keep table boundaries while retaining the shared document row coordinates."""

    def __init__(self, content: str) -> None:
        self.open_tables: list[tuple[int, bool]] = []
        self.tables: list[tuple[int, int, bool]] = []
        self.column_spans: set[int] = set()
        self.row_spans: set[int] = set()
        self.malformed_class_heading = False
        super().__init__(content)

    def _invalidate_table(self) -> None:
        if self.open_tables:
            start, _ = self.open_tables[-1]
            self.open_tables[-1] = (start, True)
            first = self.row[0] if self.row else " ".join("".join(self.cell or ()).split())
            if _CAPTION.match(first):
                self.malformed_class_heading = True

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._invalidate_table()
            self.open_tables.append((len(self.rows), False))
        elif tag == "tr" and self.row is not None:
            self._invalidate_table()
        elif tag in ("td", "th") and self.row is not None:
            if self.cell is not None:
                self._invalidate_table()
            for name, value in attrs:
                if name == "colspan" and value != "1":
                    self.column_spans.add(len(self.rows))
                if name == "rowspan" and value != "1":
                    self.row_spans.add(len(self.rows))
        super().handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if (tag == "tr" and self.cell is not None) or (tag == "table" and self.row is not None):
            self._invalidate_table()
        super().handle_endtag(tag)
        if tag == "table" and self.open_tables:
            start, invalid = self.open_tables.pop()
            self.tables.append((start, len(self.rows), invalid))


@dataclass(frozen=True, slots=True)
class ClassSizeCell:
    field: str
    measurement: Measurement
    row: int
    column: int


@dataclass(frozen=True, slots=True)
class ClassSizeReport:
    scope: str
    identifier: str
    receipt: str
    sha256: str
    cells: tuple[ClassSizeCell, ...]
    acquisition_notes: tuple[str, ...] = ()

    @property
    def measurement_period(self) -> DataPeriod:
        return PERIOD


def _class_size_cells(raw: bytes, scope: str, identifier: str) -> tuple[ClassSizeCell, ...]:
    _identifier(scope, identifier)
    report = _ReportRows(_student_html(raw))
    rows = report.rows
    _check_student_heading(rows, scope, identifier)
    _need(not report.malformed_class_heading, "Class-size table heading is malformed")
    headings = [
        i for i, row in enumerate(rows)
        if row and (
            row[0] == _TITLE
            or (len(row) > 1 and _CAPTION.match(row[0]) is not None)
        )
    ]
    if not headings:
        _need(scope == "campus", "State/district Student Information lacks a class-size table")
        return ()
    _need(len(headings) == 1, "Duplicate class-size table")
    start = headings[0]
    bounds = [(end, invalid) for first, end, invalid in report.tables if first == start]
    _need(len(bounds) == 1 and not bounds[0][1] and rows[start] == [_TITLE, *_COLUMNS[scope]]
          and start not in report.column_spans and start not in report.row_spans,
          "Class-size table boundary or geography headers differ")
    for legend in _class_size_legend():
        _need(rows.count(list(legend)) == 1, "Class-size table disclosure legend differs")
    cells = []
    groups: set[str] = set()
    fields: set[str] = set()
    section = ""
    group_start = 0
    for index in range(start + 1, bounds[0][0]):
        row = rows[index]
        _need(bool(row) and index not in report.row_spans,
              "Class-size table contains an empty or vertically spanned row")
        label, *values = row
        if label in _FIELDS:
            _need(not values and label not in groups and (not section or len(cells) > group_start),
                  "Class-size table section differs")
            groups.add(label)
            section = label
            group_start = len(cells)
            continue
        field = _FIELDS.get(section, {}).get(label)
        _need(field is not None and field not in fields and len(values) == len(_COLUMNS[scope])
              and index not in report.column_spans,
              "Class-size table has an unknown, duplicate, or malformed row")
        assert field is not None
        token = values[0]
        _need(token in CLASS_SIZE_POLICY.tokens or _NUMERIC.fullmatch(token) is not None,
              "Class-size table has an unrecognized value token")
        measurement = measure(token, CLASS_SIZE_POLICY, unit=CLASS_SIZE_UNIT)
        _need(token in CLASS_SIZE_POLICY.tokens or (measurement.numeric is not None and measurement.numeric > 0),
              "Class-size table requires a positive finite observed value")
        fields.add(field)
        cells.append(ClassSizeCell(field, measurement, index + 1, 2))
    _need(len(cells) > group_start and (scope == "campus" or len(fields) == sum(map(len, _FIELDS.values()))),
          "Class-size table has incomplete sections or measures")
    return tuple(cells)


def parse_class_size_report(raw: bytes, *, scope: str, identifier: str) -> ClassSizeReport:
    """Parse local cells only. Row and column are one-based HTML grid coordinates."""
    cells = _class_size_cells(raw, scope, identifier)
    return ClassSizeReport(scope, identifier, "", sha256(raw).hexdigest(), cells)


def read_class_size_report(store: ProvenanceStore, receipt: str, *, scope: str, identifier: str,
                           expected_sha256: str) -> ClassSizeReport:
    """Read the class-size table from an exact Student Information acquisition."""
    raw, notes = _read_student_input(store, receipt, scope=scope, identifier=identifier,
                                     expected_sha256=expected_sha256)
    cells = _class_size_cells(raw, scope, identifier)
    return ClassSizeReport(scope, identifier, receipt, expected_sha256, cells, notes)
