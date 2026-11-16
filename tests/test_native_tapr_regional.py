from types import SimpleNamespace
from decimal import Decimal
import os
from uuid import uuid4

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


@pytest.fixture(scope="module")
def normal_regional():
    from teadata.release import DataRelease

    path, pin = os.getenv("TEADATA_NORMAL_REGIONAL_RELEASE"), os.getenv("TEADATA_NORMAL_REGIONAL_PIN")
    if not path or not pin:
        pytest.skip("Actual normal TAPR parent and exact pin are explicit retained-data opt-ins")
    return DataRelease(path, expected_sha256=pin, offline=True)


def test_normal_regional_index_has_exact_attached_source(normal_regional):
    ref, = (ref for ref in normal_regional.catalog.native_sources if ref.manifest_sha256 == REVIEWED_MANIFEST)
    assert {key.dataset for key in ref.editions} == {
        "region_tapr_student_profile", "region_tapr_staff_profile", "state_tapr_staff_profile",
    }
    assert len(normal_regional.catalog.native_sources) == 18
    assert sum(len(ref.editions) for ref in normal_regional.catalog.native_sources) == 209


def test_normal_regional_selection_never_substitutes_lagged_year(normal_regional):
    field = "region_2025_student_membership_all_students_count"
    source = normal_regional.observations("region_tapr_student_profile", field, scope="region", school_year=25)
    table = source.to_arrow()
    assert table.num_rows == 20
    assert set(table["region_number"].to_pylist()) == {f"{n:02d}" for n in range(1, 21)}
    assert {row["region_number"]: row[field] for row in table.to_pylist()}["06"] == "223549"
    origin = source.provenance().to_dict()
    assert origin["selector"]["entity_release"]["manifest_sha256"] == normal_regional.manifest_sha256
    with pytest.raises(ContractError):
        normal_regional.observations("region_tapr_student_profile", field, scope="region", school_year=24)


def test_normal_staff_statewide_comparison_uses_same_period_and_population(normal_regional):
    from teadata.entities import District
    from teadata.query import Query

    keys = {key.dataset: key for ref in normal_regional.catalog.native_sources for key in ref.editions
            if key.dataset in {"district_tapr_staff_profile", "state_tapr_staff_profile"}}
    query = Query([District(uuid4(), "Report witness", district_number="084908")], repo=None)
    rows = query.compare_statewide(release=normal_regional, local=keys["district_tapr_staff_profile"],
                                   statewide=keys["state_tapr_staff_profile"])
    actual = {row.metric: row for row in rows}
    assert len(actual) == 5
    assert actual["staff_teacher_total_full_time_equiv_count"].statewide_value == Decimal("369689.2")
    assert actual["staff_teacher_turnover_ratio"].statewide_value == Decimal("18.8")
    assert actual["staff_teacher_total_base_salary_average"].statewide_value == 63751
    assert all(row.period.end_year == 2025 for row in actual.values())
    assert actual["staff_teacher_total_full_time_equiv_count"].difference_pp is None
    assert all(row.entity_manifest_sha256 == normal_regional.manifest_sha256 for row in actual.values())
    assert all(row.local_manifest_sha256 != row.statewide_manifest_sha256 for row in actual.values())


def test_all_regional_staff_comparisons_share_one_direct_texas_baseline(normal_regional):
    from teadata.query import Query

    ref, = (r for r in normal_regional.catalog.native_sources if r.manifest_sha256 == REVIEWED_MANIFEST)
    keys = {key.dataset: key for key in ref.editions}
    rows = list(Query([f"{i:02d}" for i in range(1, 21)], repo=None).compare_statewide(
        release=normal_regional, local=keys["region_tapr_staff_profile"], statewide=keys["state_tapr_staff_profile"],
    ))
    assert len(rows) == 100
    assert {row.identifier for row in rows} == {f"{i:02d}" for i in range(1, 21)}
    assert {row.statewide_value for row in rows if row.metric == "staff_teacher_turnover_ratio"} == {Decimal("18.8")}
    assert all(row.local_manifest_sha256 == row.statewide_manifest_sha256 == REVIEWED_MANIFEST for row in rows)
