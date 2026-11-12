"""Attached datasets share the entity TPRS planner, storage and provenance."""
from dataclasses import replace
import pickle

import pytest

from teadata.dataset_metadata import DataPeriod
from teadata.manifest import ContractError
from teadata.persistence import LazyMetaDict
from teadata.query import Query, describe_operators
from teadata.release import DataRelease
from teadata.source_views import DatasetFields, SourceTable
from teadata.tprs_students import DATASET

from test_entity_tprs import FIELD, bound  # noqa: F401
from test_catalog_projections import attach_observation_child, selected_catalog  # noqa: F401
from test_tprs_student_comparison import prepared_students  # noqa: F401


def test_generic_discovery_selection_and_pin_share_the_existing_planner(request, monkeypatch):
    fixture_values = request.getfixturevalue("bound")
    engine, release, _, district, campus = fixture_values
    with monkeypatch.context() as guard:
        guard.setattr(DataRelease, "artifact", lambda *a, **k: pytest.fail("Discovery opened a leaf"))
        for entity in (district, campus):
            assert entity.available_datasets() == (DATASET,)
            view = entity.dataset(DATASET.upper())
            assert view == getattr(entity, DATASET.upper())
            assert view.release is release and isinstance(view, DatasetFields)
            assert FIELD in view.variables("membership")
            assert view.years(FIELD) == (2026,)
            assert view.at(26).provenance(FIELD).to_dict() == entity.tprs.provenance(FIELD).to_dict()
            with pytest.raises(KeyError, match="Did you mean"):
                entity.dataset(DATASET[:-1])
    view = district.dataset(DATASET)
    assert view.at("2025-26")[FIELD.upper()] == 0
    assert view.school_year is None
    assert view.provenance(FIELD).to_dict() == Query([district], engine).observations(DATASET, FIELD).provenance().to_dict()
    assert view.select("membership_special_education_count").to_arrow().to_pylist()[0][
        "tprs_raw_membership_special_education_count"] == "*"
    with pytest.raises(ContractError, match="Available years"):
        view.at(25).select(FIELD)
    with pytest.raises(AttributeError, match="Did you mean"):
        _ = view.membership_total_students_cout


def test_dataset_collision_guard_never_hydrates_values_and_explicit_access_escapes(request):
    fixture_values = request.getfixturevalue("bound")
    _, _, _, district, _ = fixture_values
    district.meta = LazyMetaDict(lambda: pytest.fail("Loaded whole metadata"),
                                 read_key=lambda name: {}, read_keys=lambda: (), _validate_write=lambda: None)
    assert getattr(district, DATASET)[FIELD] == 0
    assert not district.meta._loaded
    district.meta[DATASET.upper()] = "User's value"
    with pytest.raises(ContractError, match="conflicts.*entity.meta.*entity.dataset"):
        getattr(district, DATASET)
    assert district.dataset(DATASET)[FIELD] == 0 and not district.meta._loaded
    district.meta = LazyMetaDict(lambda: pytest.fail("Loaded whole metadata"))
    for accessor in (lambda: getattr(district, DATASET), lambda: district.tprs):
        with pytest.raises(ContractError, match=r"entity.dataset\(name\)"):
            accessor()
    assert district.dataset(DATASET)[FIELD] == 0


def test_custom_mapping_keys_and_unbound_dataset_paths_preserve_contracts(request):
    engine, _, _, district, _ = request.getfixturevalue("bound")

    class CustomMapping(dict):
        def iter_keys(self):
            return iter(self)

    district.meta = CustomMapping({"local": "retained"})
    assert district.tprs[FIELD] == getattr(district, DATASET)[FIELD] == 0
    assert district.local == "retained"
    district.meta = LazyMetaDict({"local": "retained", 2: "mixed key"})
    assert district.tprs[FIELD] == getattr(district, DATASET)[FIELD] == 0
    del engine._source_release
    for access in (lambda: getattr(district, DATASET), lambda: district.dataset(DATASET)):
        with pytest.raises(ContractError, match="engine.bind_release"):
            access()


def test_known_dataset_wrong_scope_is_explained_without_leaf_reads(request, monkeypatch):
    fixture_values = request.getfixturevalue("bound")
    _, release, _, _, campus = fixture_values
    sources = tuple(replace(source, editions=tuple(key for key in source.editions if key.scope == "district"))
                    for source in release.catalog.native_sources)
    monkeypatch.setattr(type(release.catalog), "native_sources", property(lambda self: sources))
    monkeypatch.setattr(DataRelease, "artifact", lambda *a, **k: pytest.fail("Wrong scope opened a leaf"))
    assert campus.available_datasets() == ()
    with pytest.raises(ContractError, match="unavailable for campus.*Available scopes.*district"):
        campus.dataset(DATASET)


def test_non_school_periods_have_an_actionable_capability_error_before_leaves(request, monkeypatch):
    fixture_values = request.getfixturevalue("bound")
    _, release, _, district, _ = fixture_values
    monkeypatch.setattr(type(release.catalog), "_field_periods",
                        lambda self, key, fields: {name: DataPeriod("fiscal_year", 2025, 2026) for name in fields})
    monkeypatch.setattr(DataRelease, "artifact", lambda *a, **k: pytest.fail("Unsupported period opened a leaf"))
    view = district.dataset(DATASET)
    for action in (lambda: view.select(FIELD), lambda: view[FIELD], lambda: view.years(FIELD),
                   lambda: view.at(26).select(FIELD)):
        with pytest.raises(ContractError, match="fiscal_year.*publisher edition.*exact EditionKey"):
            action()


def test_scalar_multiple_rows_requires_explicit_filter_and_discovery_is_detached(request, monkeypatch):
    fixture_values = request.getfixturevalue("bound")
    import pyarrow as pa

    _, _, _, district, _ = fixture_values
    monkeypatch.setattr(SourceTable, "to_arrow", lambda self: pa.table({FIELD: [0, 1]}))
    with pytest.raises(ContractError, match=r"Multiple published rows.*select\(.*query\("):
        _ = district.dataset(DATASET)[FIELD]
    metadata = describe_operators()
    assert metadata["entity_access"]["selection"] == "entity.dataset(name)"
    metadata["entity_access"]["methods"].clear()
    assert "provenance" in describe_operators()["entity_access"]["methods"]


def test_native_access_adds_no_entity_or_snapshot_payload(request):
    fixture_values = request.getfixturevalue("bound")
    engine, release, _, district, _ = fixture_values
    before = pickle.dumps(engine)
    assert district.dataset(DATASET)[FIELD] == 0
    assert FIELD in getattr(district, DATASET).variables()
    assert pickle.dumps(engine) == before
    assert not any("name_indexes" in key for key in release.__getstate__())
    assert district.meta == {}
