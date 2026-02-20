import uuid
from types import SimpleNamespace

import pytest

from teadata.classes import Campus
from teadata.load_data import _materialize_percent_enrollment_change


def _make_campus(*, enrollment=100, meta=None):
    campus = Campus(
        id=uuid.uuid4(),
        district_id=uuid.uuid4(),
        name="Test Campus",
        charter_type="None",
        is_charter=False,
        enrollment=enrollment,
    )
    campus.meta.update(meta or {})
    return campus


def _make_repo(*campuses):
    return SimpleNamespace(_campuses={campus.id: campus for campus in campuses})


def test_materialize_percent_enrollment_change_persists_in_meta():
    campus = _make_campus(
        enrollment=110,
        meta={"campus_2015_student_enrollment_all_students_count": 100},
    )
    repo = _make_repo(campus)

    updated = _materialize_percent_enrollment_change(repo)

    assert updated == 1
    assert campus.meta["percent_enrollment_change"] == pytest.approx(10.0)


def test_materialize_percent_enrollment_change_rounds_to_tenths():
    campus = _make_campus(
        enrollment=107,
        meta={"campus_2015_student_enrollment_all_students_count": 93},
    )
    repo = _make_repo(campus)

    updated = _materialize_percent_enrollment_change(repo)

    assert updated == 1
    assert campus.meta["percent_enrollment_change"] == pytest.approx(15.1)


def test_materialize_percent_enrollment_change_writes_na_when_missing_inputs():
    campus = _make_campus(
        enrollment=110,
        meta={"percent_enrollment_change": 123.0},
    )
    repo = _make_repo(campus)

    updated = _materialize_percent_enrollment_change(repo)

    assert updated == 0
    assert campus.meta["percent_enrollment_change"] == "N/A"
