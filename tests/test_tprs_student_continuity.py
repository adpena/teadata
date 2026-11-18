"""The continuity overlay never changes the frozen current Student projection."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from teadata import tprs_student_continuity as continuity
from teadata.dataset_metadata import DataPeriod
from teadata.manifest import ContractError
from teadata.source_views import TPRSFields
from teadata.teadata_config import load_config
from teadata.tprs_students import definition as student_definition
from teadata.tprs_students import semantic_fields as student_fields
from teadata.values import ValueStatus

from test_tprs_student_inventory import full_report
from test_tprs_students import html


@pytest.fixture(scope="module")
def grid():
    return student_definition()


@pytest.mark.parametrize("scope,size", [("campus", 22), ("district", 24), ("state", 24)])
def test_exact_scope_period_and_membership_immutability(scope, size):
    fields = continuity.semantic_fields(scope)
    assert len(fields) == size
    assert all(field["period"] == DataPeriod("school_year", 2024, 2025).to_dict() for field in fields)
    assert all("2024_25" in field["field"] for field in fields)
    assert all(field["unit"] in {"count", "percent"} for field in fields)
    assert not {field["field"] for field in fields} & {field["field"] for field in student_fields()}
    assert len(student_fields()) == 179
    assert not any(field["field"].startswith("underreported") for field in continuity.semantic_fields("campus"))
    source = load_config().resolve_dataset(continuity.DATASET, 2026, reader_profile="source", exact=True)
    assert source.metadata.period == continuity.PERIOD
    assert source.metadata.version == "2026"
    assert load_config().is_source_only(continuity.DATASET, 2026)
    assert continuity.DATASET in TPRSFields(None, scope, "unused")._datasets


@pytest.mark.parametrize("scope,special_column", [("campus", 5), ("district", 4), ("state", 3)])
def test_actual_local_columns_and_percentage_point_roundtrip(grid, scope, special_column):
    rows, identifier = full_report(grid, scope)
    attrition = next(row for row in rows if row[0] == "Total Student Attrition")
    attrition[1:3] = ["1,234", "12.3%"]
    retention = next(row for row in rows if row[0] == "Kindergarten" and len(row) == ({"campus": 7, "district": 5, "state": 3}[scope]))
    retention[1] = "1.2%"
    retention[special_column - 1] = "3.4%"
    if scope != "campus":
        dq = next(row for row in rows if row[0] == "Underreported Students")
        dq[1:3] = ["0", "0.0%"]
    report = continuity.parse_continuity_report(html(rows), scope=scope, identifier=identifier)
    values = {cell.field: cell for cell in report.cells}
    assert values["attrition_2024_25_count"].measurement.value == 1234
    assert values["attrition_2024_25_percent"].measurement.value == 12.3
    assert values["retention_2024_25_non_special_education_kindergarten_percent"].column == 2
    special = values["retention_2024_25_special_education_kindergarten_percent"]
    assert (special.column, special.measurement.value, special.measurement.raw) == (special_column, 3.4, "3.4%")
    table, support = continuity.projection([report])
    assert table.num_rows == 1
    assert len(support["fields"]) == (22 if scope == "campus" else 24)
    assert table["tprs_raw_attrition_2024_25_count"].to_pylist() == ["1,234"]


@pytest.mark.parametrize("raw,status,reason", [
    ("0.0%", ValueStatus.OBSERVED, None),
    ("*", ValueStatus.SUPPRESSED, "publisher_small_count_masking"),
    ("-", ValueStatus.NOT_APPLICABLE, "publisher_no_students_in_group"),
    ("?", ValueStatus.INVALID, "publisher_statistically_improbable"),
    ("", ValueStatus.MISSING, "publisher_blank_no_measure_published"),
])
def test_documented_statuses_preserve_raw_and_coordinates(grid, raw, status, reason):
    rows, identifier = full_report(grid, "district")
    attrition = next(row for row in rows if row[0] == "Total Student Attrition")
    attrition[1:3] = ["0", raw]
    report = continuity.parse_continuity_report(html(rows), scope="district", identifier=identifier)
    cell = report.cells[1]
    assert (cell.measurement.status, cell.measurement.reason, cell.measurement.raw) == (status, reason, raw)
    assert cell.row is not None and cell.column == 3
    table, _ = continuity.projection([report])
    assert table[cell.field][0].as_py() == (0.0 if status == ValueStatus.OBSERVED else None)


@pytest.mark.parametrize("token", ["**", ".", "1%", "1.23%", "101.0%", "-1.0%", "unknown"])
def test_unknown_or_context_inappropriate_tokens_fail_closed(grid, token):
    rows, identifier = full_report(grid, "state")
    next(row for row in rows if row[0] == "Total Student Attrition")[1:3] = ["0", token]
    with pytest.raises(ContractError):
        continuity.parse_continuity_report(html(rows), scope="state", identifier=identifier)


def test_absent_grade_preserves_missing_without_eligibility_inference(grid):
    rows, identifier = full_report(grid, "campus")
    start = rows.index(["Retention Rates by Grade"])
    rows.pop(start + 2)
    report = continuity.parse_continuity_report(html(rows), scope="campus", identifier=identifier)
    for cell in report.cells:
        if "grade_1" in cell.field:
            assert (cell.measurement.raw, cell.row, cell.column) == (None, None, None)
            assert cell.measurement.status == ValueStatus.MISSING
            assert cell.measurement.reason == "retention_grade_row_not_published"


@pytest.mark.parametrize("case", json.loads((Path(__file__).parent / "fixtures/tprs-student-tail-endpoints.json").read_bytes())["cases"])
def test_original_retained_endpoint_layouts_remain_explicit(grid, case):
    rows, _ = full_report(grid, "campus")
    rows[2] = [f"SOURCE ENDPOINT ({case['identifier']})"]
    start = rows.index(["Student Attrition (2024-25)"])
    rows = [*rows[:start], *case["tail_rows"]]
    report = continuity.parse_continuity_report(html(rows), scope="campus", identifier=case["identifier"])
    assert report.unpublished_sections == tuple(case["unpublished_sections"])
    if "Retention Rates by Grade" in report.unpublished_sections:
        assert all(cell.measurement.reason == "retention_section_not_published" for cell in report.cells[2:])
    assert len(report.cells) == 22


def test_bad_or_unbound_contract_is_never_replaced_with_latest(grid):
    rows, identifier = full_report(grid, "state")
    changed = deepcopy(continuity.definition())
    changed["fields"][0]["columns"]["state"] = 4
    with pytest.raises(ContractError, match="interpretation identity"):
        continuity.parse_continuity_report(html(rows), scope="state", identifier=identifier, contract=changed)
    report = continuity.parse_continuity_report(html(rows), scope="state", identifier=identifier)
    with pytest.raises(ContractError, match="contracts differ"):
        continuity.projection([replace(report, contract_sha256="0" * 64)])
    with pytest.raises(ContractError, match="Duplicate"):
        continuity.projection([report, report])
