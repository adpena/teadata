import pytest

from teadata.load_data import (
    _build_tefa_private_campus_number_map,
    _tefa_grade_range,
    _tefa_school_type,
)


def test_tefa_grade_range_prefers_numeric_bounds():
    assert _tefa_grade_range(-1, 8, "Grades PreK-12") == "Pre-K-8"
    assert _tefa_grade_range(0, 0, "Grades K-K") == "K"
    assert _tefa_grade_range(9, 12, "Grades PreK-12") == "9-12"


def test_tefa_grade_range_falls_back_to_display_value():
    assert _tefa_grade_range(None, None, "Grades PreK–12") == "Pre-K-12"
    assert _tefa_grade_range(None, None, "") is None


@pytest.mark.parametrize(
    (
        "min_grade",
        "max_grade",
        "is_pre_k",
        "is_elementary",
        "is_middle",
        "is_high",
        "expected",
    ),
    [
        (-1, -1, True, False, False, False, "Elementary School"),
        (9, 12, False, False, False, True, "High School"),
        (6, 8, False, False, True, False, "Middle School"),
        (7, 9, False, False, True, False, "Junior High School"),
        (-1, 12, True, True, True, True, "Elementary/Secondary"),
    ],
)
def test_tefa_school_type_mapping(
    min_grade,
    max_grade,
    is_pre_k,
    is_elementary,
    is_middle,
    is_high,
    expected,
):
    assert (
        _tefa_school_type(
            min_grade,
            max_grade,
            is_pre_k,
            is_elementary,
            is_middle,
            is_high,
        )
        == expected
    )


def test_build_tefa_private_campus_number_map_is_collision_free():
    ids = ["recA", "recB", "recC", "recA", None, ""]
    reserved = {"000000001", "123456789", "999999999"}

    first = _build_tefa_private_campus_number_map(ids, set(reserved))
    second = _build_tefa_private_campus_number_map(ids, set(reserved))

    assert first == second
    assert set(first.keys()) == {"recA", "recB", "recC"}

    assigned = {value[1:] for value in first.values()}
    assert len(assigned) == 3
    assert assigned.isdisjoint(reserved)
    assert all(len(number) == 9 and number.isdigit() for number in assigned)
