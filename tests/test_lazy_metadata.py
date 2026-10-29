"""Lazy metadata obeys mapping semantics and preserves transaction ownership."""

from collections.abc import MutableMapping
import copy
import json
import pickle
import uuid

import pytest
from sqlalchemy import func, select

from teadata.classes import Campus, DataEngine, District
from teadata.manifest import ContractError
from teadata.persistence import (
    DistrictRecord,
    LazyMetaDict,
    create_engine,
    create_sessionmaker,
    ensure_schema,
    export_dataengine,
    import_dataengine,
)


@pytest.mark.parametrize(
    "operation",
    [
        lambda value: dict(value),
        lambda value: value | {"extra": 2},
        lambda value: {"extra": 2} | value,
        lambda value: value.copy(),
        copy.copy,
        copy.deepcopy,
        lambda value: pickle.loads(pickle.dumps(value)),
    ],
)
def test_complete_mapping_operations_load_once_and_preserve_source(operation):
    calls = []

    def load():
        calls.append(True)
        return {"raw": [1, {"nested": True}], "zero": -0.0}

    value = LazyMetaDict(load)
    assert isinstance(value, MutableMapping)
    result = operation(value)
    assert result["raw"] == [1, {"nested": True}]
    assert result["zero"] == -0.0
    assert len(calls) == 1
    assert value == {"raw": [1, {"nested": True}], "zero": -0.0}
    assert len(calls) == 1


def test_seed_is_sparse_and_whole_load_replaces_stale_keys():
    calls = []

    def load():
        calls.append(True)
        return {"fresh": 9}

    value = LazyMetaDict(load, seed={"deleted": 7})
    assert value["deleted"] == 7 and "deleted" in value
    assert not calls
    assert value == {"fresh": 9}
    assert "deleted" not in value and len(calls) == 1


def test_mutation_materializes_before_edit_and_never_resurrects_deleted_seed():
    value = LazyMetaDict(lambda: {"same": 1, "other": 2}, seed={"same": 0})
    value["same"] = 99
    assert dict(value) == {"same": 99, "other": 2}
    value = LazyMetaDict(lambda: {"same": 1, "other": 2}, seed={"same": 0})
    del value["same"]
    assert dict(value) == {"other": 2}
    value.update({"new": 3})
    value |= {"last": 4}
    assert dict(value) == {"other": 2, "new": 3, "last": 4}


def test_mutating_return_values_use_complete_current_payload():
    value = LazyMetaDict(lambda: {"same": 1, "other": 2}, seed={"same": 0})
    assert value.pop("same") == 1
    with pytest.raises(KeyError):
        value.pop("missing")
    assert value.pop("missing", 3) == 3
    value = LazyMetaDict(lambda: {"same": 1}, seed={"same": 0})
    assert value.setdefault("same", 99) == 1
    assert value.popitem() == ("same", 1)
    assert value == {}


def test_failed_lazy_source_does_not_become_empty_or_serialize_silently():
    def load():
        raise ContractError("source release changed")

    value = LazyMetaDict(load)
    with pytest.raises(ContractError, match="source release"):
        dict(value)
    assert not value._loaded
    with pytest.raises(TypeError):
        json.dumps(value)


@pytest.mark.parametrize(
    "entity",
    [
        District(id=uuid.UUID(int=1), name="District"),
        Campus(
            id=uuid.UUID(int=2),
            district_id=uuid.UUID(int=1),
            name="Campus",
            charter_type="",
            is_charter=False,
        ),
    ],
)
def test_entity_metadata_readers_accept_standard_mappings(entity):
    entity.meta = LazyMetaDict(lambda: {"custom": "source", "nested": [1]})
    assert entity.custom == "source"
    assert "custom" in dir(entity)
    assert entity.to_dict()["custom"] == "source"
    assert entity.to_dict()["nested"] == [1]


def test_lazy_read_reuses_active_caller_transaction_and_preserves_commit(catalog):
    engine = create_engine("sqlite:///:memory:")
    ensure_schema(engine)
    factory = create_sessionmaker(engine)
    repo = DataEngine()
    repo.catalog = catalog
    district = District(id=uuid.UUID(int=1), name="District")
    district.meta = {"custom": "uncommitted source"}
    repo.add_district(district)

    def unexpected_factory():
        raise AssertionError("Caller transaction must be reused")

    try:
        with factory.begin() as session:
            export_dataengine(repo, session)
            restored = import_dataengine(
                session, meta_session_factory=unexpected_factory
            )
            assert dict(restored._districts[district.id].meta) == district.meta
            assert session.scalar(select(func.count()).select_from(DistrictRecord)) == 1
        with factory() as session:
            assert session.scalar(select(func.count()).select_from(DistrictRecord)) == 1
    finally:
        engine.dispose()
