"""Account for retained 2026 Student Information cells without admitting measures.

This audit supplements the captured projection contract. It does not extend an
old grid, interpret blank/dot cells, or turn the report tail into observations.
Coordinates count actual parsed HTML cells, not merged spreadsheet positions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .dataset_metadata import DataPeriod
from .native_witnesses import _Rows
from .provenance import ProvenanceStore
from .tprs_students import (
    _need, _read_student_input, _resolve_contract, _student_html, _student_workbook_cells,
    parse_student_report, semantic_fields,
)


@dataclass(frozen=True, slots=True)
class StudentInventoryCell:
    section: str
    label: str
    raw: str
    row: int
    column: int
    disposition: Literal["qualified", "held"]
    reason: str | None
    field: str | None = None
    population: str | None = None
    quantity: str | None = None
    geography_role: str | None = None
    period: DataPeriod | None = None
    denominator: str | None = None


@dataclass(frozen=True, slots=True)
class StudentReportInventory:
    scope: str
    identifier: str
    sha256: str
    grid_sha256: str
    cells: tuple[StudentInventoryCell, ...]
    class_size_boundary_row: int | None
    unpublished_sections: tuple[str, ...] = ()


_MOBILITY = (
    "Total Mobile Students", "African American", "Hispanic", "White", "American Indian",
    "Asian", "Pacific Islander", "Two or More Races", "Special Ed Students who are Mobile",
    "Count and Percent of EB Students/EL who are Mobile",
    "Count and Percent of Econ Dis Students who are Mobile",
)
_GRADES = ("Kindergarten", *(f"Grade {grade}" for grade in range(1, 10)))
_GEOGRAPHIES = {"state": ["State"], "district": ["District", "State"],
                "campus": ["Campus", "District", "State"]}
_FOOTER = ["", "TEA | Analytics, Assessment, and Reporting | Performance Reporting", ""]


def inventory_student_report(raw: bytes, *, scope: str, identifier: str,
                             contract: Mapping[str, Any] | None = None) -> StudentReportInventory:
    """Give every published Student Information data cell one disposition.

    The tail has a separate, closed 2026 presentation layout. Its semantic
    fields remain null: headings and glossary alone do not qualify its periods
    or denominators. Some campus reports end after attrition or retention;
    absent sections are reported explicitly, never manufactured as zero cells.
    Original HTML retains whitespace and merged-cell evidence.
    """
    grid, _ = _resolve_contract(contract)
    report = parse_student_report(raw, scope=scope, identifier=identifier, contract=grid)
    rows = _Rows(_student_html(raw)).rows
    fields = {field["field"]: field for field in semantic_fields(grid)}
    cells = []
    for cell in report.cells:
        qualified = cell.field is not None
        reason = cell.hold
        if not qualified and not cell.raw:
            reason = "publisher_blank_no_measure_published"
        elif not qualified and cell.raw == ".":
            reason = "undocumented_dot_token"
        cells.append(StudentInventoryCell(
            cell.section, cell.label, cell.raw, cell.row, cell.column,
            "qualified" if qualified else "held", reason, cell.field,
            cell.population, cell.measure, cell.geography_role,
            cell.period if qualified else None,
            fields[cell.field]["denominator"] if qualified else None,
        ))
    position = next(i for i, row in enumerate(rows) if row == ["Mobility (2024-25)"])
    boundary = ["Class Size Information", *_GEOGRAPHIES[scope]]

    def take(expected: list[str]) -> None:
        nonlocal position
        _need(position < len(rows) and rows[position] == expected,
              f"Student inventory {scope} layout differs at parsed row {position + 1}; expected {expected!r}")
        position += 1

    def data(section: str, label: str, width: int) -> None:
        nonlocal position
        _need(position < len(rows) and len(rows[position]) == width + 1
              and rows[position][0] == label,
              f"Student inventory {scope} row/width differs at parsed row {position + 1}: {section} / {label}")
        for column, token in enumerate(rows[position][1:], 2):
            cells.append(StudentInventoryCell(section, label, token, position + 1, column, "held",
                "publisher_blank_no_measure_published" if not token else
                "unreviewed_tail_period_population_denominator"))
        position += 1

    def complete(boundary_row: int | None, unpublished: tuple[str, ...] = ()) -> StudentReportInventory:
        keys = {(cell.row, cell.column) for cell in cells}
        _need(len(keys) == len(cells), "Student inventory assigned a data coordinate more than once")
        return StudentReportInventory(scope, identifier, report.sha256, report.grid_sha256,
                                      tuple(cells), boundary_row, unpublished)

    def campus_end(unpublished: tuple[str, ...]) -> StudentReportInventory | None:
        if scope != "campus" or position >= len(rows) or rows[position] != grid["legend"][0]:
            return None
        remainder = rows[position:]
        _need(remainder in (grid["legend"], [*grid["legend"], _FOOTER]),
              "Student inventory campus has unrecognized data after the report tail")
        return complete(None, unpublished)

    take(["Mobility (2024-25)"])
    for label in _MOBILITY:
        data("Mobility (2024-25)", label, len(grid["headers"][scope]) - 1)
    take(["Student Attrition (2024-25)"])
    data("Student Attrition (2024-25)", "Total Student Attrition", len(grid["headers"][scope]) - 1)
    ended = campus_end(("Retention Rates by Grade", "Class Size Information"))
    if ended is not None:
        return ended
    if scope == "campus" and position < len(rows) and rows[position] == boundary:
        _need(rows.count(boundary) == 1, "Student inventory requires one exact Class Size boundary")
        take(boundary)
        return complete(position, ("Retention Rates by Grade",))
    if scope == "state":
        take(["Student Information", "Non-SpecialEducationRates", "SpecialEducationRates"])
    else:
        take(["", "Non-Special Education Rates", "Special Education Rates"])
        take(["Student Information", *_GEOGRAPHIES[scope], *_GEOGRAPHIES[scope]])
    take(["Retention Rates by Grade"])
    grades = []
    while position < len(rows) and rows[position] and rows[position][0] in _GRADES:
        label = rows[position][0]
        grades.append(label)
        data("Retention Rates by Grade", label, len(_GEOGRAPHIES[scope]) * 2)
    _need(bool(grades) and len(grades) == len(set(grades))
          and grades == [grade for grade in _GRADES if grade in grades]
          and (scope == "campus" or tuple(grades) == _GRADES),
          "Student inventory retention grade rows are absent, duplicate, unordered or incomplete")
    if scope != "campus":
        take(["", *_GEOGRAPHIES[scope]])
        take(["", *[item for _ in _GEOGRAPHIES[scope] for item in ("Count", "Percent")]])
        take(["Data Quality"])
        data("Data Quality", "Underreported Students", len(_GEOGRAPHIES[scope]) * 2)
    ended = campus_end(("Class Size Information",))
    if ended is not None:
        return ended
    _need(rows.count(boundary) == 1, "Student inventory requires one exact Class Size boundary")
    take(boundary)
    return complete(position)


def verify_inventory_workbook_witnesses(store: ProvenanceStore, *,
                                      contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Compare all inventoried cells in three independently acquired workbooks.

    Exact section/label matching locates rows; HTML and XLSX row numbers differ.
    An absent XML cell is reported separately from a stored empty XML cell.
    Neither is a numeric zero or an admitted Enrollment observation.
    """
    grid, _ = _resolve_contract(contract)
    comparisons = []
    for scope, witness in grid["witnesses"].items():
        raw, _ = _read_student_input(store, witness["html_receipt"], scope=scope,
                                    identifier=witness["identifier"], expected_sha256=witness["html_sha256"])
        report = inventory_student_report(raw, scope=scope, identifier=witness["identifier"], contract=grid)
        workbook = _student_workbook_cells(store, scope, witness, contract=grid)
        expected = {(cell.section, cell.label) for cell in report.cells}
        sections = {cell.section for cell in report.cells} - {""}
        labels = {}
        section = ""
        for number in sorted({number for number, _ in workbook}):
            label = workbook.get((number, "A"), "")
            if label == "Class Size Information":
                break
            if label in sections:
                section = label
            elif (section, label) in expected:
                _need((section, label) not in labels, "Student inventory workbook row repeats")
                labels[section, label] = number
        _need(set(labels) == expected, "Student inventory workbook data-row coverage differs")
        absent = stored_blank = 0
        for cell in report.cells:
            coordinate = (labels[cell.section, cell.label], chr(ord("A") + cell.column - 1))
            _need(workbook.get(coordinate, "") == cell.raw, "Student inventory HTML/workbook cell differs")
            absent += coordinate not in workbook
            stored_blank += coordinate in workbook and workbook[coordinate] == ""
        comparisons.append({"scope": scope, "cells": len(report.cells), "absent_xml_cells": absent,
                            "stored_blank_xml_cells": stored_blank})
    return {"status": "PASS_FULL_STUDENT_INFORMATION_HTML_XLSX_INVENTORY", "comparisons": comparisons,
            "scope": "Three named retained reports only; no additional numeric qualification"}
