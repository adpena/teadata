from types import SimpleNamespace

import pytest

from teadata.manifest import ContractError
from teadata.native_tapr_regional import REVIEWED_MANIFEST, _cells, _number, qualify_tapr_regional


def _staff(scope="region"):
    def pair(value):
        return [value, "n/a", value, "n/a"] if scope == "region" else [value, "n/a"]

    def single(value):
        return [value, value] if scope == "region" else [value]
    return [["", "Texas Education Agency", ""], ["2024-25 Staff Information"],
            ["REGION 06: HUNTSVILLE"] if scope == "region" else ["State"], [], [],
            ["Staff Information", *(["Count", "Percent"] * (2 if scope == "region" else 1))],
            ["Professional Staff", *pair("20.0")], ["Teachers", *pair("19.1")],
            ["Professional Support", *pair("0.9")], ["Number of Students per Teacher", *pair("21.2")],
            ["Average Years Experience of Campus Staff"], ["Teachers", *single("7.1")],
            ["Average Teacher Salary by Years of Experience (regular duties only)"],
            ["Average Actual Salaries (regular duties only)"], ["Teachers", *single("$67,496")],
            ["Professional Support", *single("$50,000")], ["Turnover Rate for Teachers", *single("30.3%")]]


@pytest.mark.parametrize("scope", ["region", "state"])
def test_staff_sections_keep_teacher_meanings_and_variable_width(scope):
    cells = _cells(_staff(scope), "staff_profile", scope, "06" if scope == "region" else "state")
    assert cells == {
        "staff_teacher_total_full_time_equiv_count": ("fte", "19.1", 2025),
        "staff_teacher_experience_average": ("years", "7.1", 2025),
        "staff_teacher_total_base_salary_average": ("currency", "$67,496", 2025),
        "staff_teacher_student_ratio": ("ratio", "21.2", 2025),
        "staff_teacher_turnover_ratio": ("percent", "30.3%", 2025),
    }


@pytest.mark.parametrize("edit", ["duplicate", "wrong_year", "wrong_region", "columns"])
def test_staff_malformed_controls_raise_actionable_errors(edit):
    rows = _staff()
    if edit == "duplicate":
        rows.insert(8, rows[7])
    elif edit == "wrong_year":
        rows[1] = ["2023-24 Staff Information"]
    elif edit == "wrong_region":
        rows[2] = ["REGION 16: AMARILLO"]
    else:
        rows[5] = ["Staff Information", "State"]
    with pytest.raises(ContractError):
        _cells(rows, "staff_profile", "region", "06")


def _student():
    def row(label, count):
        return [label, count, "10.0%", "100", "10.0%", "200", "20.0%", "300", "30.0%"]
    labels = ["African American", "Hispanic", "White", "American Indian", "Asian", "Pacific Islander", "Two or More Races"]
    return [["", "Texas Education Agency", ""], ["2024-25 Student Information"], ["REGION 06: HUNTSVILLE"], [],
            ["", "Membership", "Enrollment"], ["", "Region", "State", "Region", "State"],
            ["Student Information", "Count", "Percent", "Count", "Percent", "Count", "Percent", "Count", "Percent"],
            row("Total Students", "1,000"), ["Ethnic Distribution"], *[row(label, "100") for label in labels],
            ["Sex"], row("Special Education", "100"), ["Mobility (2023-24)"], row("Total Mobile Students", "10"),
            row("White", "1"), ["Student Attrition (2023-24)"], row("Total Student Attrition", "20")]


def test_student_populations_race_sections_and_lagged_years_are_distinct():
    cells = _cells(_student(), "student_profile", "region", "06")
    assert len(cells) == 38
    assert cells["student_membership_all_students_count"] == ("count", "1,000", 2025)
    assert cells["student_enrollment_all_students_count"] == ("count", "200", 2025)
    assert cells["student_membership_white_count"] == ("count", "100", 2025)
    assert cells["student_membership_attrition_all_students_percent"] == ("percent", "10.0%", 2024)


@pytest.mark.parametrize("value,unit", [("*", "count"), ("<5", "count"), ("-", "count"),
    ("NaN", "count"), ("-1", "count"), ("1.2", "count"), ("100.1", "percent"), ("", "count")])
def test_numeric_controls_never_coerce_masking_or_invalid_values(value, unit):
    with pytest.raises(ContractError):
        _number(value, unit)


def test_numeric_control_keeps_zero_and_publisher_precision():
    assert _number("0", "count") == 0
    assert _number("$67,496", "currency", displayed=True) == 67496
    assert str(_number("123.4567", "fte")) == "123.4567"


@pytest.mark.parametrize("manifest,offline", [("0" * 64, True), (REVIEWED_MANIFEST, False)])
def test_gate_requires_exact_offline_release(manifest, offline):
    with pytest.raises(ContractError, match="exact reviewed offline"):
        qualify_tapr_regional(SimpleNamespace(manifest_sha256=manifest, offline=offline), None)
