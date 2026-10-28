import uuid

from teadata import Campus, DataEngine, District
from teadata.load_data import _map_stats_for_campus


def test_private_map_stats_use_public_metric_applicability_not_private_raw_count(catalog):
    repo = DataEngine()
    repo.catalog = catalog
    district = District(id=uuid.uuid4(), name="Home district")
    repo.add_district(district)
    field = catalog.source_for_metric("enrollment", "campus")[1]
    campus = Campus(id=uuid.uuid4(), district_id=district.id, name="Private school",
                    charter_type="", is_charter=False, is_private=True, enrollment=37, meta={field: -1})
    repo.add_campus(campus)
    stats = {item["id"]: item for item in _map_stats_for_campus(campus)}
    for name in ("enrollment", "overall_rating"):
        assert stats[name]["value"] == "Not applicable"
        assert stats[name]["measurement"]["status"] == "not_applicable"
        assert stats[name]["measurement"]["reason"] == "outside_metric_population"
    assert campus.enrollment == 37
    assert campus.meta[field] == -1


def test_public_missing_and_suppressed_map_enrollment_are_distinct(catalog):
    repo = DataEngine()
    repo.catalog = catalog
    district = District(id=uuid.uuid4(), name="Home district")
    repo.add_district(district)
    field = catalog.source_for_metric("enrollment", "campus")[1]
    campus = Campus(id=uuid.uuid4(), district_id=district.id, name="Public school",
                    charter_type="", is_charter=False, meta={field: -1})
    repo.add_campus(campus)
    stats = {item["id"]: item for item in _map_stats_for_campus(campus)}
    assert stats["enrollment"]["measurement"]["status"] == "suppressed"
    campus.meta.clear()
    stats = {item["id"]: item for item in _map_stats_for_campus(campus)}
    assert stats["enrollment"]["measurement"]["status"] == "missing"
