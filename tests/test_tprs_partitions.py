from dataclasses import replace

import pytest

from teadata.dataset_metadata import EditionKey
from teadata.manifest import ContractError
from teadata.release import DataRelease
from teadata.source_views import EntitySources, SourceTable
from teadata.tprs_partitions import STUDENT_DATASET, STUDENT_DERIVED_ROUTES, derived_campus_edition
from teadata.tprs_statewide_intake import TPRSStudentCampusIndex


@pytest.fixture
def student():
    return EditionKey(dataset=STUDENT_DATASET, edition="2026", scope="campus", table="campus",
                      grain="one row per campus_number", revision_sha256="a" * 64)


@pytest.mark.parametrize("dataset", STUDENT_DERIVED_ROUTES)
def test_derived_routes_preserve_exact_partition_identity(student, dataset):
    translated = derived_campus_edition(student, dataset)
    assert translated == replace(student, dataset=dataset, table=f"campus-{student.identity}")
    assert derived_campus_edition(replace(student, revision_sha256="b" * 64), dataset) != translated
    for invalid in (replace(student, edition="2025"), replace(student, scope="district"), translated):
        with pytest.raises(ValueError, match="Student campus edition"):
            derived_campus_edition(invalid, dataset)
    with pytest.raises(ValueError, match="registered"):
        derived_campus_edition(student, "invented")


@pytest.mark.parametrize("dataset", (STUDENT_DATASET, *STUDENT_DERIVED_ROUTES))
def test_publisher_namespace_routes_one_exact_key_without_opening_data(student, dataset, monkeypatch):
    students = (student, replace(student, revision_sha256="b" * 64))
    keys = tuple(key if dataset == STUDENT_DATASET else derived_campus_edition(key, dataset) for key in students)
    release = DataRelease.__new__(DataRelease)
    index = TPRSStudentCampusIndex("c" * 64, "2026", {"001902001": students[1]})
    monkeypatch.setattr(release, "tprs_student_campus_index", lambda *, edition: index)
    monkeypatch.setattr(EntitySources, "datasets", lambda self: keys)
    monkeypatch.setattr(release, "table", lambda key: SourceTable(release, key))
    monkeypatch.setattr(release, "artifact", lambda *a, **k: pytest.fail("Routing opened data"))
    namespace = EntitySources(release, "campus", "001902001", "2026")
    selected = namespace.dataset(dataset.upper())
    assert selected.edition == keys[1]
    assert namespace.dataset(keys[0]).edition == keys[0]  # Explicit selection never reroutes.
    with pytest.raises(ContractError, match="No admitted.*publisher edition 2026"):
        EntitySources(release, "campus", "001902002", "2026").dataset(dataset)
    monkeypatch.setattr(EntitySources, "datasets", lambda self: (keys[0], replace(keys[1], revision_sha256="d" * 64)))
    with pytest.raises(ContractError, match="route is outside"):
        namespace.dataset(dataset)


def test_all_derived_memos_are_runtime_only_and_independent():
    release = DataRelease.__new__(DataRelease)
    release._v3 = False
    names = [route.memo_attribute for route in STUDENT_DERIVED_ROUTES.values()]
    assert len(names) == len(set(names))
    for name in names:
        setattr(release, name, object())
    assert not set(names) & release.__getstate__().keys()
