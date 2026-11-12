from copy import copy
from hashlib import sha256
from pathlib import Path
import json
import pickle
import shutil
from uuid import uuid4

import pytest

from teadata import Campus, DataEngine, District
from teadata.manifest import ContractError, canonical_json
from teadata.query import Query
from teadata.release import DataRelease
from teadata.source_views import SourceTable, TPRSFields
from teadata.tprs_students import DATASET

from test_catalog_projections import attach_observation_child, selected_catalog  # noqa: F401
from test_tprs_student_comparison import prepared_students  # noqa: F401


FIELD = "membership_total_students_count"


@pytest.fixture
def bound(tmp_path, request):
    source, _, keys = request.getfixturevalue("prepared_students")
    for path in source.iterdir():
        if path.is_file():
            shutil.copyfile(path, tmp_path / ("native.json" if path.name == "release.json" else path.name))
    child = DataRelease(tmp_path / "native.json", offline=True)
    release = attach_observation_child(request.getfixturevalue("selected_catalog"), child)
    engine = DataEngine()
    engine.catalog = release.catalog
    district = District(uuid4(), "Synthetic district", district_number="001902")
    campus = Campus(uuid4(), district.id, "Synthetic campus", "", False, campus_number="'001902001")
    engine.add_district(district)
    engine.add_campus(campus)
    engine.bind_release(release)
    return engine, release, keys, district, campus


def test_entity_namespace_discovery_planning_and_provenance_do_not_open_leaves(bound, monkeypatch):
    engine, release, keys, district, campus = bound
    with monkeypatch.context() as guard:
        guard.setattr(DataRelease, "artifact", lambda *a, **k: pytest.fail("Discovery opened a data leaf"))
        for entity, scope in ((district, "district"), (campus, "campus")):
            namespace = entity.tprs
            assert namespace.release is release
            assert namespace.variables(r"^membership_total_students_(count|percent)$") == (
                "membership_total_students_count", "membership_total_students_percent")
            assert FIELD in dir(namespace)
            assert namespace.years(FIELD) == (2026,)
            planned = namespace.at("2025–26").select(FIELD)
            assert isinstance(planned, SourceTable) and planned.edition == keys[scope]
            assert planned.measurement_period.end_year == 2026
            assert namespace.provenance(FIELD).to_dict() == planned.provenance().to_dict()
            assert namespace.at(26).school_year == 2026 and namespace.at(None).school_year is None
        assert engine._source_release is release
    for entity in (district, campus):
        assert entity.tprs.membership_total_students_count == 0
        assert entity.tprs[FIELD.upper()] == 0
        assert entity.tprs.at(26).membership_special_education_count is None
        row, = entity.tprs.select("membership_special_education_count").to_arrow().to_pylist()
        assert row["tprs_raw_membership_special_education_count"] == "*"
        assert row["tprs_status_membership_special_education_count"] == "suppressed"
    assert district.meta == campus.meta == {}


def test_entity_and_query_use_the_engine_pin_instead_of_a_later_global_release(bound, monkeypatch):
    engine, release, _, district, _ = bound
    monkeypatch.setattr("teadata.release.current_release", lambda: pytest.fail("Followed a global release"))
    namespace = district.tprs
    assert namespace.membership_total_students_count == 0
    query = Query([district], engine).observations(DATASET, FIELD)
    assert query.to_arrow()[FIELD].to_pylist() == [0]
    assert query.provenance().to_dict() == namespace.provenance(FIELD).to_dict()
    assert namespace.release is release


def test_entity_namespace_gives_actionable_field_year_and_row_errors(bound):
    engine, _, _, district, _ = bound
    with pytest.raises(AttributeError, match="Did you mean.*membership_total_students_count"):
        _ = district.tprs.membership_total_students_cout
    with pytest.raises(KeyError, match=r"tprs.variables\(\)"):
        _ = district.tprs["membership_total_students_cout"]
    with pytest.raises(ContractError, match="2025.*Available years.*2026"):
        district.tprs.at(25).select(FIELD)
    for invalid in (False, "2025-27", 25.0, []):
        with pytest.raises(ValueError, match="Invalid school_year.*2024-25"):
            district.tprs.at(invalid)
    for fields in ((), (FIELD, FIELD), (FIELD, False)):
        with pytest.raises(ValueError, match="distinct"):
            district.tprs.select(*fields)
    absent = District(uuid4(), "No published report", district_number="001903")
    engine.add_district(absent)
    with pytest.raises(ContractError, match="No published rows for district 001903"):
        _ = absent.tprs.membership_total_students_count


def test_entity_namespace_does_not_hide_binding_integrity_or_duplicate_rows(bound, monkeypatch):
    import pyarrow as pa

    _, release, keys, district, campus = bound
    child = release._native_edition_release(keys["campus"])
    metadata = next(ref for ref in child.catalog.edition_metadata_refs()
                    if ref == child.catalog._edition_metadata_refs[keys["campus"].identity])
    path = child.path.with_name(metadata.location)
    original = path.read_bytes()
    assert FIELD in campus.tprs.variables()
    try:
        path.write_bytes(b" " + original[1:])
        with pytest.raises(ContractError, match="checksum"):
            campus.tprs.variables()
    finally:
        path.write_bytes(original)
    with monkeypatch.context() as duplicate:
        duplicate.setattr(SourceTable, "to_arrow", lambda self: pa.table({FIELD: [0, 1]}))
        with pytest.raises(ContractError, match="Multiple published rows for district 001902"):
            _ = district.tprs.membership_total_students_count

    def bad_binding(*args, **kwargs):
        raise KeyError("missing bound column")

    monkeypatch.setattr(TPRSFields, "select", bad_binding)
    with pytest.raises(KeyError, match="missing bound column"):
        hasattr(district.tprs, FIELD)


def test_entity_namespace_rejects_name_and_family_collisions(bound, monkeypatch):
    namespace = bound[3].tprs
    monkeypatch.setattr(TPRSFields, "_indexes", lambda self: {DATASET: {FIELD, FIELD.upper()}})
    with pytest.raises(ContractError, match="Ambiguous TPRS field.*exact spelling"):
        namespace.select(FIELD.capitalize())
    companion = "membership_total_students_percent"
    monkeypatch.setattr(TPRSFields, "_indexes", lambda self: {DATASET: {FIELD, companion}, "tprs_class_size": {FIELD}})
    with pytest.raises(ContractError, match="unambiguous native family.*tprs_class_size"):
        namespace.select(FIELD)
    with pytest.raises(ContractError, match="unambiguous native family.*tprs_class_size"):
        namespace.select(FIELD, companion)
    assert namespace.select(companion).to_arrow()[companion].to_pylist() == [100.0]
    monkeypatch.setattr(TPRSFields, "_indexes", lambda self: {})
    assert namespace.variables() == ()
    with pytest.raises(ContractError, match="No qualified TPRS native families.*district"):
        namespace.select(FIELD)


def test_entity_namespace_requires_registered_public_identity_and_preserves_meta(bound):
    engine, _, _, district, campus = bound
    district.meta[FIELD] = 999
    assert district.membership_total_students_count == 999
    assert district.tprs.membership_total_students_count == 0
    district.meta["tprs"] = "Existing custom metadata"
    with pytest.raises(ContractError, match=r"entity.meta\['tprs'\]"):
        _ = district.tprs
    assert district.meta["tprs"] == "Existing custom metadata"
    del district.meta["tprs"]
    engine._districts.pop(district.id)
    with pytest.raises(ContractError, match="exact registered entity"):
        _ = district.tprs
    standalone = District(uuid4(), "Standalone", district_number="001902")
    with pytest.raises(ContractError, match="release.for_entity"):
        _ = standalone.tprs
    private = Campus(uuid4(), None, "Private school", "", False, is_private=True,
                     source_namespace="tefa", source_id="provider-1")
    engine.add_campus(private)
    with pytest.raises(ContractError, match="private school"):
        _ = private.tprs


def test_release_binding_validates_build_and_preserves_the_first_context(bound, request):
    engine, release, keys, district, _ = bound
    for invalid in (None, "release.json", True):
        with pytest.raises(TypeError, match="DataRelease"):
            engine.bind_release(invalid)
    with pytest.raises(ContractError, match="parent entity release"):
        engine.bind_release(release._native_edition_release(keys["district"]))
    equivalent = copy(release)
    equivalent.offline = False
    assert engine.bind_release(equivalent) is engine
    assert district.tprs.release is release and release.offline is True
    document = json.loads(release.path.read_text())
    record = document["processing"]["record"]
    record["note"] = "A distinct manifest with the same entity catalog"
    document["processing"]["record_id"] = sha256((canonical_json(record) + "\n").encode()).hexdigest()
    path = release.path.with_name("alternate.json")
    path.write_text(canonical_json(document))
    with pytest.raises(ContractError, match="different release manifest"):
        engine.bind_release(DataRelease(path, offline=True))
    engine.catalog = request.getfixturevalue("selected_catalog")
    with pytest.raises(ContractError, match="entity build"):
        engine.bind_release(release)


def test_runtime_source_binding_does_not_enter_pickle_and_default_loading_rebinds(bound, monkeypatch):
    engine, release, _, district, _ = bound
    del engine._source_release
    before = pickle.dumps(engine)
    with pytest.raises(ContractError, match="engine.bind_release"):
        _ = district.tprs
    engine.bind_release(release)
    assert pickle.dumps(engine) == before
    copied = pickle.loads(pickle.dumps(engine))
    with pytest.raises(ContractError, match="engine.bind_release"):
        _ = (copied >> ("district", 1902)).first().tprs
    copied.bind_release(release)
    assert (copied >> ("district", 1902)).first().tprs.membership_total_students_count == 0
    monkeypatch.setattr("teadata.engine.current_release", lambda: release)
    monkeypatch.setattr("teadata.engine._load_snapshot_payload", lambda path: pickle.loads(before))
    monkeypatch.setattr(DataRelease, "artifact", lambda self, role: Path(role))
    monkeypatch.setattr(DataEngine, "attach_boundary_store", lambda self, path: None)
    for loaded in (DataEngine.from_snapshot(), DataEngine(autoload=True)):
        assert loaded._source_release is release
        assert (loaded >> ("district", 1902)).first().tprs.release is release
    explicit = DataEngine.from_snapshot("trusted-local.pkl.gz")
    with pytest.raises(ContractError, match="engine.bind_release"):
        _ = (explicit >> ("district", 1902)).first().tprs
