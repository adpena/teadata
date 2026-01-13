import uuid

from teadata.classes import Campus, DataEngine, District
from teadata.persistence.sqlalchemy_store import (
    create_engine,
    create_sessionmaker,
    ensure_schema,
    export_dataengine,
)
from teadata import entity_store


def test_entity_store_round_trip(tmp_path):
    repo = DataEngine()
    district = District(
        id=uuid.uuid4(),
        name="Test ISD",
        district_number="123456",
        enrollment=1200,
        rating="A",
    )
    district.meta = {"district_2025_demo": "ok"}
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
