"""Full retained-report accounting is distinct from semantic admission."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from teadata.dataset_metadata import MetricDefinition
from teadata.manifest import ContractError
from teadata.tprs_student_inventory import _GEOGRAPHIES, _GRADES, _MOBILITY, inventory_student_report
from teadata import tprs_students as students

import test_tprs_students as student_fixtures
from test_tprs_students import html, report_rows

grid = student_fixtures.grid
policy = student_fixtures.policy
reviewed174_grid = student_fixtures.reviewed174_grid
_ENDPOINTS = json.loads((Path(__file__).parent / "fixtures/tprs-student-tail-endpoints.json").read_bytes())["cases"]


def full_report(grid, scope):
    identifier = {"state": "S", "district": "001902", "campus": "001902001"}[scope]
    rows = report_rows(grid, scope, identifier)[:-len(grid["legend"])]
    width = len(grid["headers"][scope]) - 1
    rows.extend([[label, *["0"] * width] for label in _MOBILITY])
    rows.extend([["Student Attrition (2024-25)"], ["Total Student Attrition", *["*"] * width]])
    geographies = _GEOGRAPHIES[scope]
    if scope == "state":
        rows.append(["Student Information", "Non-SpecialEducationRates", "SpecialEducationRates"])
    else:
        rows.extend([["", "Non-Special Education Rates", "Special Education Rates"],
                     ["Student Information", *geographies, *geographies]])
    rows.append(["Retention Rates by Grade"])
    rows.extend([[grade, *["0.0%"] * (len(geographies) * 2)] for grade in _GRADES])
    if scope != "campus":
        rows.extend([["", *geographies], ["", *[x for _ in geographies for x in ("Count", "Percent")]],
                     ["Data Quality"], ["Underreported Students", *["-"] * (len(geographies) * 2)]])
    rows.extend([["Class Size Information", *geographies], *grid["legend"]])
    return rows, identifier


@pytest.mark.parametrize("scope", ["state", "district", "campus"])
def test_inventory_accounts_every_current_and_tail_data_coordinate(grid, scope):
    rows, identifier = full_report(grid, scope)
    result = inventory_student_report(html(rows), scope=scope, identifier=identifier)
    current = students.parse_student_report(html(rows), scope=scope, identifier=identifier)
    tail_count = 12 * (len(grid["headers"][scope]) - 1) + 10 * 2 * len(_GEOGRAPHIES[scope])
    if scope != "campus":
        tail_count += 2 * len(_GEOGRAPHIES[scope])
    assert len(result.cells) == len(current.cells) + tail_count
    assert len({(cell.row, cell.column) for cell in result.cells}) == len(result.cells)
    assert sum(cell.disposition == "qualified" for cell in result.cells) == 179
    tail = result.cells[len(current.cells):]
    assert all(cell.disposition == "held" and cell.field is None and cell.period is None
               and cell.population is None and cell.denominator is None for cell in tail)
    assert any(cell.raw == "0" for cell in tail) and any(cell.raw == "*" for cell in tail)
    assert result.class_size_boundary_row is not None
    assert rows[result.class_size_boundary_row - 1][0] == "Class Size Information"


@pytest.mark.parametrize("damage", ["unknown", "width", "duplicate", "reorder", "boundary", "missing"])
def test_inventory_rejects_unrecognized_or_incomplete_tail(grid, damage):
    rows, identifier = full_report(grid, "district")
    start = rows.index(["Mobility (2024-25)"])
    if damage == "unknown":
        rows[start + 1][0] = "Unknown mobile group"
    elif damage == "width":
        rows[start + 1].pop()
    elif damage == "duplicate":
        rows.insert(start + 2, rows[start + 1])
    elif damage == "reorder":
        rows[start + 1], rows[start + 2] = rows[start + 2], rows[start + 1]
    elif damage == "missing":
        rows.pop(start + 1)
    else:
        rows.insert(-len(grid["legend"]), ["Class Size Information", "District", "State"])
    with pytest.raises(ContractError, match="Student inventory"):
        inventory_student_report(html(rows), scope="district", identifier=identifier)


def test_inventory_preserves_explicit_blank_and_dot_as_held(grid):
    rows, identifier = full_report(grid, "district")
    next(row for row in rows if row[0] == "Career and Technical Education")[1] = "."
    next(row for row in rows if row[0] == "Students with Autism")[5] = ""
    report = inventory_student_report(html(rows), scope="district", identifier=identifier)
    assert any(cell.raw == "." and cell.reason == "undocumented_dot_token" for cell in report.cells)
    assert any(cell.raw == "" and cell.reason == "publisher_blank_no_measure_published" for cell in report.cells)


@pytest.mark.parametrize("retention,footer", [(False, False), (False, True), (True, False), (True, True)])
def test_campus_report_can_end_at_attrition_or_retention_without_class_size(grid, retention, footer):
    # Source layout witnesses: 015915668 /91c3f051... ends after attrition;
    # 015916005 /8dc6bea0... ends after its retention rows. No zero rows are added.
    rows, identifier = full_report(grid, "campus")
    boundary = rows.index(["Class Size Information", "Campus", "District", "State"])
    start = boundary if retention else rows.index(["", "Non-Special Education Rates", "Special Education Rates"])
    rows = [*rows[:start], *grid["legend"]]
    if footer:
        rows.append(["", "TEA | Analytics, Assessment, and Reporting | Performance Reporting", ""])
    report = inventory_student_report(html(rows), scope="campus", identifier=identifier)
    assert report.class_size_boundary_row is None
    assert report.unpublished_sections == (("Class Size Information",) if retention else
                                          ("Retention Rates by Grade", "Class Size Information"))
    assert any(cell.section == "Retention Rates by Grade" for cell in report.cells) is retention
    assert not any(cell.section == "Class Size Information" for cell in report.cells)


def test_absent_campus_sections_do_not_allow_unknown_or_duplicate_tail_data(grid):
    rows, identifier = full_report(grid, "campus")
    boundary = rows.index(["", "Non-Special Education Rates", "Special Education Rates"])
    for extra in (["Unknown data", "1"], ["Student Attrition (2024-25)"]):
        changed = [*rows[:boundary], *grid["legend"], extra]
        with pytest.raises(ContractError, match="unrecognized data"):
            inventory_student_report(html(changed), scope="campus", identifier=identifier)


def test_campus_can_publish_class_size_without_retention_rows(grid):
    # Source 015916046 /584bb16a... proceeds directly from attrition to Class Size.
    rows, identifier = full_report(grid, "campus")
    retention = rows.index(["", "Non-Special Education Rates", "Special Education Rates"])
    boundary = rows.index(["Class Size Information", "Campus", "District", "State"])
    changed = [*rows[:retention], *rows[boundary:]]
    report = inventory_student_report(html(changed), scope="campus", identifier=identifier)
    assert report.unpublished_sections == ("Retention Rates by Grade",)
    assert report.class_size_boundary_row is not None
    assert changed[report.class_size_boundary_row - 1][0] == "Class Size Information"
    assert not any(cell.section == "Retention Rates by Grade" for cell in report.cells)
    changed.insert(-len(grid["legend"]), ["Class Size Information", "Campus", "District", "State"])
    with pytest.raises(ContractError, match="one exact Class Size boundary"):
        inventory_student_report(html(changed), scope="campus", identifier=identifier)


@pytest.mark.parametrize("case", _ENDPOINTS, ids=lambda case: case["identifier"])
def test_actual_publisher_tail_layouts_preserve_original_cells_and_missing_sections(grid, case):
    rows, _ = full_report(grid, "campus")
    start = rows.index(["Student Attrition (2024-25)"])
    rows[2] = [f"Synthetic prefix ({case['identifier']})"]
    combined = [*rows[:start], *case["tail_rows"]]
    report = inventory_student_report(html(combined), scope="campus", identifier=case["identifier"])
    assert report.unpublished_sections == tuple(case["unpublished_sections"])
    attrition = [cell for cell in report.cells if cell.section == "Student Attrition (2024-25)"]
    assert [cell.raw for cell in attrition] == case["tail_rows"][1][1:]
    assert all(cell.disposition == "held" and cell.field is None for cell in attrition)
    assert (report.class_size_boundary_row is None) == ("Class Size Information" in report.unpublished_sections)


def test_new_composition_contract_preserves174_and_has_exact_denominators(grid, reviewed174_grid):
    old = students.semantic_fields(reviewed174_grid)
    new = students.semantic_fields(grid)
    assert len(old) == 174 and len(new) == 179 and new[:174] == old
    assert [field["field"] for field in new[174:]] == grid["denominator_fields"]
    assert all(field["population"] == "membership" and field["quantity"] == "percent"
               and field["denominator"] == "membership_total_students_with_disabilities_count" for field in new[174:])
    assert "membership_total_students_with_disabilities_percent" not in {field["field"] for field in new}


@pytest.mark.parametrize("token,status,value", [("*", "suppressed", None), ("**", "suppressed", None),
                                              ("-", "not_applicable", None), ("0.0%", "observed", 0.0),
                                              ("56.3%", "observed", 56.3)])
def test_composition_preserves_source_status_instead_of_calculating_counts(grid, policy, token, status, value):
    rows = report_rows(grid)
    next(row for row in rows if row[0] == "Students with Autism")[2] = token
    report = students.parse_student_report(html(rows), scope="district", identifier="001902")
    table, _, _ = students.projection([report], policy)
    field = "membership_students_with_autism_percent"
    assert table[field].to_pylist() == [value]
    assert table["tprs_raw_" + field].to_pylist() == [token]
    assert table["tprs_status_" + field].to_pylist() == [status]


def test_metric_denominator_is_optional_and_does_not_change_historical_serialization():
    assert MetricDefinition.from_dict({"field": "old", "unit": "count"}).to_dict() == {"field": "old", "unit": "count"}
    raw = {"field": "share", "unit": "percent", "denominator": "membership_total_students_with_disabilities_count"}
    assert MetricDefinition.from_dict(raw).to_dict() == raw
    for invalid in ("", "  ", True, 0, []):
        with pytest.raises(ValueError, match="denominator"):
            MetricDefinition.from_dict({**raw, "denominator": invalid})


def test_unknown_order_and_denominator_contracts_do_not_bypass_admission(grid):
    changed = deepcopy(grid)
    changed["field_order"][-1] = changed["field_order"][0]
    with pytest.raises(ContractError, match="Unsupported Student report grid"):
        students.semantic_fields(changed)


def test_workbook_explicit_blank_remains_distinct_from_omitted_xml_cell():
    from xml.etree.ElementTree import fromstring

    sheet = fromstring('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                       '<sheetData><row r="2"><c r="A2"/><c r="C2" t="s"><v>0</v></c>'
                       '<c r="D2"><v>0</v></c></row></sheetData></worksheet>')
    cells = students._student_sheet_cells(sheet, [""])
    assert cells == {(2, "A"): "", (2, "C"): "", (2, "D"): "0"}
    assert (2, "B") not in cells


@pytest.mark.parametrize("aliases", [False, True])
def test_header_denominator_survives_unknown_period_without_reading_observations(grid, aliases):
    from teadata.column_schema import header_records
    from teadata.dataset_metadata import DatasetMetadata

    field = "membership_students_with_autism_percent"
    metric = {"field": field, "unit": "percent", "denominator": "Disability Membership population"}
    metrics = {"one": metric, **({"two": metric} if aliases else {})}
    metadata = DatasetMetadata.from_dict({"metrics": metrics})
    document = students._crosswalk({"state": ["a" * 64]}, contract=grid)
    rows = header_records(document, dataset=students.DATASET, edition="2026", metadata=metadata)
    selected = next(row for row in rows if row["field"] == field)
    assert selected["denominator"] == metric["denominator"]
    assert all("denominator" not in row for row in rows if row["field"] != field)
    changed = DatasetMetadata.from_dict({"metrics": {**metrics, "conflict": {**metric, "denominator": "Another population"}}})
    with pytest.raises(ValueError, match="Conflicting reviewed denominators"):
        header_records(document, dataset=students.DATASET, edition="2026", metadata=changed)
