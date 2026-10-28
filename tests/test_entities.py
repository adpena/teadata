import uuid

import pytest

from teadata.entities import Campus
from teadata.engine import DataEngine


def _make_campus(*, enrollment=100, meta=None):
    campus = Campus(
        id=uuid.uuid4(),
        district_id=uuid.uuid4(),
        name="Test Campus",
        charter_type="None",
        is_charter=False,
        enrollment=enrollment,
    )
    if meta:
        campus.meta.update(meta)
    return campus


def test_campus_to_dict_includes_percent_enrollment_change(catalog):
    campus = _make_campus(
        enrollment=110,
        meta={"campus_2015_student_enrollment_all_students_count": 100},
    )

    repo = DataEngine()
    repo.catalog = catalog
    repo.add_campus(campus)
    _, enrollment_field = catalog.source_for_metric("enrollment", "campus")
    campus.meta[enrollment_field] = campus.enrollment

    result = campus.to_dict()

    assert "percent_enrollment_change" in result
    assert result["percent_enrollment_change"] == pytest.approx(10.0)


def test_campus_to_dict_handles_missing_percent_enrollment_change():
    campus = _make_campus()

    result = campus.to_dict()

    assert "percent_enrollment_change" in result
    assert result["percent_enrollment_change"] == "N/A"
