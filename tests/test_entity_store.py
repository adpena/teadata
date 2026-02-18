import uuid

import pytest

from teadata.classes import Campus, DataEngine, District
from teadata.persistence.sqlalchemy_store import (
    create_engine,
    create_sessionmaker,
    ensure_schema,
    export_dataengine,
)
from teadata import entity_store


def _build_entity_store(tmp_path):
    repo = DataEngine()
    district = District(
        id=uuid.uuid4(),
        name="Test ISD",
        district_number="123456",
        enrollment=1200,
        rating="A",
    )
    district.meta = {"district_2025_demo": "ok", "district_extra": "extra"}
    repo.add_district(district)

    campus = Campus(
        id=uuid.uuid4(),
        district_id=district.id,
        name="Test Campus",
        charter_type="Charter",
        is_charter=True,
    )
    campus.campus_number = "123456789"
    campus.district_number = "123456"
    campus.enrollment = 200
    campus.meta = {"campus_2025_demo": 5}
    repo.add_campus(campus)

    path = tmp_path / "entities.sqlite"
    engine = create_engine(f"sqlite:///{path}")
    ensure_schema(engine)
    Session = create_sessionmaker(engine, expire_on_commit=False)
    with Session.begin() as session:
        export_dataengine(repo, session, replace=True)
    engine.dispose()
    return path


def test_entity_store_round_trip(tmp_path):
    path = _build_entity_store(tmp_path)

    campus_row = entity_store.load_campus("123456789", store_path=path)
    assert campus_row
    assert campus_row["campus_number"] == "123456789"
    assert campus_row["meta"]["campus_2025_demo"] == 5

    district_row = entity_store.load_district("123456", store_path=path)
    assert district_row
    assert district_row["district_number"] == "123456"
    assert district_row["campus_count"] == 1
    assert district_row["is_charter"] is True
    assert district_row["district_type"] == "Charter"

    keys = entity_store.list_meta_keys("campus", store_path=path)
    assert "campus_2025_demo" in keys


def test_entity_store_get_meta_after_no_meta_lookup(tmp_path):
    path = _build_entity_store(tmp_path)
    store = entity_store.EntityStore(path)

    campus_no_meta = store.get_campus("123456789", include_meta=False)
    assert campus_no_meta
    assert campus_no_meta["meta"] == {}
    campus_with_meta = store.get_campus("123456789", include_meta=True)
    assert campus_with_meta
    assert campus_with_meta["meta"]["campus_2025_demo"] == 5

    district_no_meta = store.get_district("123456", include_meta=False)
    assert district_no_meta
    assert district_no_meta["meta"] == {}
    district_with_meta = store.get_district("123456", include_meta=True)
    assert district_with_meta
    assert district_with_meta["meta"]["district_2025_demo"] == "ok"


def test_entity_store_meta_keys_limit_does_not_truncate_cache(tmp_path):
    path = _build_entity_store(tmp_path)
    store = entity_store.EntityStore(path)

    limited = store.list_meta_keys("district", limit=1)
    full = store.list_meta_keys("district")

    assert len(limited) == 1
    assert "district_2025_demo" in full
    assert "district_extra" in full


def test_entity_store_load_raises_for_invalid_database(tmp_path):
    invalid = tmp_path / "invalid.sqlite"
    invalid.write_text("not a sqlite database", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Failed querying campus"):
        entity_store.load_campus("123456789", store_path=invalid)
    with pytest.raises(RuntimeError, match="Failed querying district"):
        entity_store.load_district("123456", store_path=invalid)
