import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import pandas as pd
import pytest
from shapely.geometry import Polygon

from teadata import Campus, DataEngine, District
from teadata.attributes import attribute_value
from teadata.identifiers import MAX_CAMPUS_LOOKUP_KEY_BYTES, campus_lookup_key, normalize_campus_lookup_key
from teadata.load_data import _add_tefa_entities, build_campus_profile_map_payload
from teadata.manifest import ContractError


def ace_record():
    document = json.loads((Path(__file__).parents[1] / "teadata/reference_cases/publisher_2025.json").read_text())
    case = next(c for c in document["cases"] if c["entity"].get("source_id") == "rec1kXXvIucG7UkoA")
    return {name: cell["raw_token"] for name, cell in case["source_cells"].items()}


@pytest.mark.parametrize("value", ["227901180", "'227901180", "227901180.0"])
def test_public_lookup_key_stays_the_existing_nine_digits(value):
    assert normalize_campus_lookup_key(value) == "227901180"


@pytest.mark.parametrize("identifier", ["rec1kXXvIucG7UkoA", "Case/Sensitive % café", "rec", "001"])
def test_private_source_identity_roundtrips_exactly(identifier):
    key = campus_lookup_key(source_namespace="tefa", source_id=identifier, is_private=True)
    assert normalize_campus_lookup_key(key) == key
    assert key.startswith("tefa:")
    if identifier.startswith("rec"):
        assert key.startswith("tefa:rec")


@pytest.mark.parametrize("fields", [
    {}, {"source_namespace": "unknown", "source_id": "ID"},
    {"source_namespace": "tefa", "source_id": ""},
    {"source_namespace": "tefa", "source_id": "\x00"},
    {"source_namespace": "tefa", "source_id": "ID", "campus_number": "184873751"},
])
def test_private_identity_never_synthesizes_or_uses_a_tea_number(fields):
    with pytest.raises(ValueError):
        campus_lookup_key(is_private=True, **fields)


def test_lookup_key_bound_is_encoded_bytes_not_current_identifier_length():
    key = campus_lookup_key(source_namespace="tefa", source_id="x" * (MAX_CAMPUS_LOOKUP_KEY_BYTES - 5), is_private=True)
    assert len(key.encode("ascii")) == MAX_CAMPUS_LOOKUP_KEY_BYTES
    with pytest.raises(ValueError, match="bound"):
        campus_lookup_key(source_namespace="tefa", source_id="é" * 200, is_private=True)
    for malformed in ("tefa:%xx", "tefa:%ff", "unsupported:rec1"):
        with pytest.raises(ValueError):
            normalize_campus_lookup_key(malformed)
    assert normalize_campus_lookup_key("rec1kXXvIucG7UkoA") is None
    assert normalize_campus_lookup_key("campus227901180") is None


def test_actual_ace_listing_has_source_identity_and_no_public_governance(catalog):
    repo = DataEngine()
    repo.catalog = catalog
    assert _add_tefa_entities(repo, pd.DataFrame([ace_record()]), version=2025) == 1
    campus = next(iter(repo._campuses.values()))
    assert campus.lookup_key == "tefa:rec1kXXvIucG7UkoA"
    assert campus.id == uuid.uuid5(uuid.NAMESPACE_URL, "teadata:private:tefa:rec1kXXvIucG7UkoA")
    assert campus.campus_number is None and campus.district_id is None and campus.district is None
    assert campus.district_number is None
    assert campus.to_dict(include_meta=False)["district_id"] is None
    assert campus.meta["reported_directory_district_number"] == "227901"
    assert "district_name" not in campus.meta and "tefa_id" not in campus.meta
    assert repo.campus_by_key(campus.lookup_key) is campus
    assert (repo >> ("campus", campus.lookup_key)).first() is campus
    assert repo.campus_by_number("184873751") is None
    assert attribute_value(campus, "campus_number", scope="campus") is None
    assert attribute_value(campus, "campus_source_id", scope="campus") == "rec1kXXvIucG7UkoA"
    assert catalog.measurement(campus, "enrollment").status.value == "not_applicable"


def test_duplicate_source_ids_fail_before_any_private_listing_is_added():
    repo = DataEngine()
    with pytest.raises(ContractError, match="duplicate"):
        _add_tefa_entities(repo, pd.DataFrame([ace_record(), ace_record()]), version=2025)
    assert not repo._campuses


@pytest.mark.parametrize("point,expected", [((1, 1), 2), ((0, 0), 2), ((2, 1), 2), ((4, 4), 0), (None, 0)])
def test_private_map_containment_is_all_match_boundary_inclusive_and_not_governance(catalog, point, expected):
    repo = DataEngine()
    repo.catalog = catalog
    boundary = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    for number, subtype in [("123456", "INDEPENDENT"), ("123457", "COMMON"), ("123458", "CHARTER"), ("123459", "TJJD")]:
        repo.add_district(District(id=uuid.uuid5(uuid.NAMESPACE_URL, number), name=subtype,
            district_number=number, boundary=boundary, meta={"organization_sub_type": subtype}))
    private = Campus(id=uuid.uuid4(), district_id=None, name="Private geographic control",
        charter_type="", is_charter=False, is_private=True, source_namespace="tefa", source_id="recGeo", location=point)
    repo.add_campus(private)
    payload = build_campus_profile_map_payload(repo, private)
    assert len(payload["containingDistricts"]) == expected
    assert payload["comparisonAvailable"] is bool(expected)
    assert private.district is None and private.district_number is None
    assert payload["private"]["campusNumber"] is None
    assert payload["private"]["lookupKey"] == private.lookup_key
    assert all(r["relation"] == "point_covered_by" and r["boundary_predicate"] == "covers" for r in payload["containingDistricts"])
    assert all(r["boundary_dataset"] == "districts" and r["boundary_version"] == "2025" for r in payload["containingDistricts"])


def test_raw_directory_name_never_becomes_governing_name(catalog):
    entity = SimpleNamespace(is_private=True, district=None, meta={"district_name": "Austin ISD"})
    assert attribute_value(entity, "district_name", scope="campus") is None
    repo = DataEngine()
    repo.catalog = catalog
    _add_tefa_entities(repo, pd.DataFrame([ace_record()]), version=2025)
    private = next(iter(repo._campuses.values()))
    private.meta["district_name"] = "Raw directory claim"
    assert private.district_name is None
    assert private.to_dict()["district_name"] is None
    assert private.meta["district_name"] == "Raw directory claim"


@pytest.mark.parametrize("value", ["c1", "-123", "011901001-XX", "1.5", "000000000", "True"])
def test_bad_public_lookup_is_unresolved_not_a_digit_run(value):
    assert normalize_campus_lookup_key(value) is None


def test_missing_identity_slots_cannot_fall_back_to_metadata():
    campus = Campus(id=uuid.uuid4(), district_id=None, name="Explicit identity",
        charter_type="", is_charter=False, is_private=True, source_namespace="tefa", source_id="recTrue")
    del campus.source_id
    campus.meta.update(source_id="recForged", lookup_key="tefa:recForged")
    with pytest.raises(ValueError, match="rebuild"):
        campus.lookup_key
    with pytest.raises(AttributeError, match="explicit"):
        campus.source_id


def test_actual_private_identity_survives_store_lazy_json_and_analytical_roundtrip(tmp_path, catalog):
    from teadata.analytical_build import build_analytical_artifacts
    from teadata.entity_store import EntityStore
    from teadata.persistence import create_engine, create_sessionmaker, ensure_schema, export_dataengine, import_dataengine
    import pyarrow.parquet as pq

    repo = DataEngine()
    repo.catalog = catalog
    _add_tefa_entities(repo, pd.DataFrame([ace_record()]), version=2025)
    source = tmp_path / "entities.sqlite"
    engine = create_engine("sqlite:///" + str(source))
    ensure_schema(engine)
    factory = create_sessionmaker(engine)
    try:
        with factory.begin() as session:
            export_dataengine(repo, session)
        with factory() as session:
            loaded = import_dataengine(session, lazy_meta=True)
            private = loaded.campus_by_key("tefa:rec1kXXvIucG7UkoA")
            assert private.lookup_key == "tefa:rec1kXXvIucG7UkoA"
            assert not private.meta._loaded
            assert attribute_value(private, "campus_source_id", scope="campus") == "rec1kXXvIucG7UkoA"
            assert not private.meta._loaded
            assert private.to_dict()["district_id"] is None
            assert private.campus_number is None
    finally:
        engine.dispose()
    with EntityStore(source) as store:
        row = store.get_campus("tefa:rec1kXXvIucG7UkoA")
        assert row["district_id"] is None and row["district_number"] is None
        assert row["campus_number"] is None and row["source_id"] == "rec1kXXvIucG7UkoA"
        assert store.get_campus("184873751") is None
    artifacts, _ = build_analytical_artifacts(source, catalog, tmp_path / "read_models")
    attributes = pq.read_table(artifacts["campus_attributes"]).to_pylist()
    assert len(attributes) == 1
    assert attributes[0]["entity_id"] == attributes[0]["campus_lookup_key"] == "tefa:rec1kXXvIucG7UkoA"
    assert attributes[0]["campus_number"] is None
    assert attributes[0]["district_name"] is None
    assert attributes[0]["campus_source_id"] == "rec1kXXvIucG7UkoA"


def test_retired_private_membership_operator_has_no_alias():
    repo = DataEngine()
    assert not hasattr(repo, "private_campuses_in")
    with pytest.raises(ValueError, match="Unsupported"):
        repo >> ("private_campuses_in", None)


def test_legacy_entity_and_map_identity_layouts_refuse_without_repairs(tmp_path, catalog):
    import hashlib
    import sqlite3
    from teadata.entity_store import EntityStore
    from teadata.manifest import write_sqlite_catalog
    from teadata.map_store import load_campus_profile_payload

    for kind, ddl in [("entities", "CREATE TABLE campuses(id TEXT,campus_number TEXT)"),
                      ("maps", "CREATE TABLE campus_profile(campus_number TEXT,payload BLOB)")]:
        path = tmp_path / f"legacy_{kind}.sqlite"
        with sqlite3.connect(path) as connection:
            connection.execute(ddl)
            write_sqlite_catalog(connection, catalog)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        with pytest.raises(ContractError, match="rebuild"):
            if kind == "entities":
                with EntityStore(path):
                    pass
            else:
                load_campus_profile_payload("184873751", store_path=path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_nonreal_tea_number_cannot_be_persisted_as_public_identity():
    for value in (True, "c1", "011901001-XX", -123):
        with pytest.raises(ValueError, match="real TEA"):
            campus_lookup_key(campus_number=value)


def test_private_without_point_keeps_profile_and_snapshot_identity(tmp_path, catalog, monkeypatch):
    from teadata import load_data
    from teadata.map_store import load_campus_profile_payload

    repo = DataEngine()
    repo.catalog = catalog
    row = ace_record()
    row["Location Lat"] = None
    row["Location Lng"] = None
    _add_tefa_entities(repo, pd.DataFrame([row]), version=2025)
    campus = next(iter(repo._campuses.values()))
    assert campus.point is None and campus.location is None
    monkeypatch.setattr(load_data, "BUILD_MAP_STORE", True)
    monkeypatch.setattr(load_data, "_map_store_path", lambda *args: tmp_path / "maps.sqlite")
    path = load_data._save_map_store(repo, "districts", "campuses")
    payload = load_campus_profile_payload(campus.lookup_key, store_path=path)
    assert payload["private"]["lookupKey"] == campus.lookup_key
    assert payload["containingDistricts"] == [] and not payload["comparisonAvailable"]
    monkeypatch.setattr(load_data, "_snapshot_path", lambda *args: tmp_path / "repo.pkl.gz")
    snapshot = load_data._save_repo_snapshot(repo, "districts", "campuses")
    loaded = DataEngine.from_snapshot(snapshot)
    private = loaded.campus_by_key(campus.lookup_key)
    assert private is not None and private.district is None and private.location is None
    assert private.to_dict()["source_id"] == "rec1kXXvIucG7UkoA"


@pytest.mark.parametrize("subtype,expected", [("INDEPENDENT", 1), ("COMMON", 1), ("CHARTER", 0), ("TJJD", 0), (None, 0)])
def test_actual_private_listing_only_overlays_traditional_public_comparisons(catalog, subtype, expected):
    repo = DataEngine()
    repo.catalog = catalog
    _add_tefa_entities(repo, pd.DataFrame([ace_record()]), version=2025)
    private = next(iter(repo._campuses.values()))
    x, y = private.point.x, private.point.y
    district = District(id=uuid.uuid4(), name="Comparison boundary control", district_number="227901",
        boundary=Polygon([(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1), (x - 1, y + 1)]),
        meta={"organization_sub_type": subtype})
    repo.add_district(district)
    public = Campus(id=uuid.uuid4(), district_id=district.id, district_number="227901", campus_number="227901999",
        name="Comparison campus control", charter_type="", is_charter=False, grade_range="PK-12", location=(x, y))
    repo.add_campus(public)
    assert len(build_campus_profile_map_payload(repo, public)["privateCampuses"]) == expected


def test_split_snapshot_relocates_and_hydrates_boundary_index_once(tmp_path, catalog, monkeypatch):
    import shutil
    from teadata import load_data
    from teadata.engine import _load_snapshot_payload

    repo = DataEngine()
    repo.catalog = catalog
    _add_tefa_entities(repo, pd.DataFrame([ace_record()]), version=2025)
    private = next(iter(repo._campuses.values()))
    x, y = private.point.x, private.point.y
    polygon = Polygon([(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1), (x - 1, y + 1)])
    repo.add_district(District(id=uuid.uuid4(), district_number="227901", name="Owned boundary control",
        boundary=polygon, meta={"organization_sub_type": "INDEPENDENT"}))
    assert len(build_campus_profile_map_payload(repo, private)["containingDistricts"]) == 1
    assert repo._district_cover_index is not None
    repo.clear_indexes()
    assert repo._district_cover_index is None
    assert len(build_campus_profile_map_payload(repo, private)["containingDistricts"]) == 1
    monkeypatch.setattr(load_data, "SPLIT_BOUNDARIES", True)
    monkeypatch.setattr(load_data, "_boundary_store_path", lambda *args: tmp_path / "boundaries.sqlite")
    monkeypatch.setattr(load_data, "_snapshot_path", lambda *args: tmp_path / "repo.pkl.gz")
    boundary = load_data._save_boundary_store(repo, "districts", "campuses")
    load_data._strip_repo_boundaries(repo)
    snapshot = load_data._save_repo_snapshot(repo, "districts", "campuses")
    raw = _load_snapshot_payload(snapshot)
    assert raw._district_cover_index is None
    assert all(district.boundary is None and district.polygon is None and district._prepared is None
               for district in raw._districts.values())
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    moved_snapshot, moved_boundary = relocated / snapshot.name, relocated / boundary.name
    shutil.copyfile(snapshot, moved_snapshot)
    shutil.copyfile(boundary, moved_boundary)
    snapshot.unlink()
    boundary.unlink()
    loaded = DataEngine.from_snapshot(moved_snapshot)
    actual = loaded.campus_by_key(private.lookup_key)
    assert build_campus_profile_map_payload(loaded, actual)["containingDistricts"] == []
    loaded.attach_boundary_store(moved_boundary)
    assert loaded._district_cover_index is None
    statements = []
    loaded._boundary_store._connect().set_trace_callback(statements.append)
    try:
        for _ in range(2):
            assert len(build_campus_profile_map_payload(loaded, actual)["containingDistricts"]) == 1
        assert sum("SELECT wkb" in statement for statement in statements) == 1
        loaded.add_district(District(id=uuid.uuid4(), district_number="227902", name="Second boundary control",
            boundary=polygon, meta={"organization_sub_type": "COMMON"}))
        assert loaded._district_cover_index is None
        assert len(build_campus_profile_map_payload(loaded, actual)["containingDistricts"]) == 2
        assert actual.district is None
        loaded.clear_indexes()
        assert loaded._district_cover_index is None
    finally:
        loaded._boundary_store.close()


@pytest.mark.parametrize("private", [False, True])
def test_campus_name_query_preserves_colons_without_relaxing_key_lookup(private):
    repo = DataEngine()
    fields = {"is_private": True, "source_namespace": "tefa", "source_id": "recColonControl"} if private else {"campus_number": "123456001"}
    campus = Campus(id=uuid.uuid4(), district_id=None, name="Example: East Campus", charter_type="", is_charter=False, **fields)
    repo.add_campus(campus)
    assert (repo >> ("campus", "Example: East Campus")).first() is campus
    assert (repo >> ("campus", campus.lookup_key)).first() is campus
    assert repo.campus_by_key(campus.lookup_key) is campus
    with pytest.raises(ValueError, match="namespace"):
        repo.campus_by_key("unsupported:recColonControl")


@pytest.mark.parametrize("key", ["tefa:%xx", "tefa:%ff", "tefa:"])
def test_explicit_private_key_query_keeps_malformed_keys_strict(key):
    with pytest.raises(ValueError):
        DataEngine() >> ("campus", key)
