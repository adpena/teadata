"""Completion and helpful metadata access retain the lazy value boundary."""
from collections import defaultdict
from uuid import UUID

import pytest
from sqlalchemy import event

from teadata import Campus, DataEngine, District
from teadata.manifest import ContractError
from teadata.persistence import (
    LazyMetaDict, create_engine, create_sessionmaker, ensure_schema,
    export_dataengine, import_dataengine,
)
from teadata.release import DataRelease
from teadata.tprs_students import DATASET

from test_catalog_projections import attach_observation_child, selected_catalog  # noqa: F401
from test_entity_tprs import bound  # noqa: F401
from test_tprs_student_comparison import prepared_students  # noqa: F401


@pytest.fixture(params=['district', 'campus'])
def entity(request):
    if request.param == 'district':
        return District(UUID(int=1), 'District', district_number='000001')
    return Campus(UUID(int=2), UUID(int=1), 'Campus', '', False, campus_number='000001001')


def test_completion_and_metadata_suggestions_never_fetch_candidate_values(entity):
    calls = []
    raw = {'Membership': 0, 'Masked': '<5', 'odd.key (1)': 'literal', '雪': 'unicode'}

    def read_key(name):
        calls.append(name)
        return {name: raw[name]} if name in raw else {}

    entity.meta = LazyMetaDict(lambda: pytest.fail('Whole metadata hydration'),
                               read_key=read_key, read_keys=lambda: raw.keys())
    assert set(raw) <= set(dir(entity))
    assert calls == [] and not entity.meta._loaded
    assert entity.membership == 0 and calls == ['membership', 'Membership']
    assert entity.Masked == '<5'
    before = list(calls)
    with pytest.raises(AttributeError, match='Did you mean.*Membership'):
        _ = entity.Membershp
    assert calls == [*before, 'Membershp']  # Suggestions did not fetch a candidate.
    assert not entity.meta._loaded


def test_exact_metadata_keys_win_and_folded_ambiguity_is_explicit(entity):
    entity.meta = {'Count': 0, 'count': '<5', 7: 'not an attribute', '雪': 'value'}
    assert entity.Count == 0 and entity.count == '<5'
    with pytest.raises(AttributeError, match="Ambiguous.*COUNT.*Count.*count"):
        _ = entity.COUNT
    assert '雪' in dir(entity) and '7' not in dir(entity)


def test_default_mapping_never_invents_a_zero_or_mutates_a_missing_attribute(entity):
    entity.meta = defaultdict(int, {'Membership': 42})
    assert entity.membership == 42
    assert entity.Membership == 42
    with pytest.raises(AttributeError, match='Unknown.*attribute'):
        _ = entity.typo
    assert not hasattr(entity, 'typo')
    assert dict(entity.meta) == {'Membership': 42}


def test_loader_only_completion_lists_known_names_without_hydration(entity):
    entity.meta = LazyMetaDict(lambda: pytest.fail('Completion loaded unindexed metadata'),
                               seed={'known': 0}, _validate_write=lambda: None)
    entity.meta['added'] = '<5'
    assert {'known', 'added'} <= set(dir(entity))
    assert not entity.meta._loaded


@pytest.mark.parametrize('name', ['__dict__', '__deepcopy__', '__setstate__'])
def test_protocol_probes_never_request_metadata(entity, name):
    entity.meta = LazyMetaDict(lambda: pytest.fail('Protocol probe hydrated metadata'),
                               read_key=lambda key: pytest.fail('Protocol probe requested a value'))
    with pytest.raises(AttributeError):
        getattr(entity, name)
    assert not entity.meta._loaded


def test_changed_key_reader_is_not_silently_hidden(entity):
    def changed():
        raise ContractError('source catalog changed')

    entity.meta = LazyMetaDict(lambda: pytest.fail('Whole metadata hydration'),
                               read_key=lambda key: {}, read_keys=changed)
    for access in (lambda: dir(entity), lambda: entity.Membershp):
        with pytest.raises(ContractError, match='source catalog changed'):
            access()


def test_dataset_completion_is_index_only_and_does_not_require_source_binding(request, monkeypatch):
    engine, _, _, district, campus = request.getfixturevalue('bound')
    monkeypatch.setattr(DataRelease, 'artifact', lambda *a, **k: pytest.fail('Discovery opened data'))
    for entity in (district, campus):
        entity.meta = LazyMetaDict(lambda: pytest.fail('Discovery hydrated metadata'))
        assert DATASET in dir(entity)
        # Sparse completion is never accepted as a complete collision check.
        with pytest.raises(ContractError, match='cannot list keys'):
            getattr(entity, DATASET)
        assert entity.dataset(DATASET).dataset == DATASET
    del engine._source_release
    assert DATASET in dir(district) and DATASET in dir(campus)
    with pytest.raises(ContractError, match='engine.bind_release'):
        campus.dataset(DATASET)
    private = Campus(UUID(int=3), None, 'Private', '', False, is_private=True,
                     source_namespace='tefa', source_id='example')
    private._repo = engine
    assert DATASET not in dir(private)


def test_sql_completion_uses_only_keys_and_single_value_access_stays_projected(catalog):
    sql = create_engine('sqlite:///:memory:')
    ensure_schema(sql)
    factory = create_sessionmaker(sql)
    repo = DataEngine()
    repo.catalog = catalog
    district = District(UUID(int=1), 'District', district_number='000001')
    campus = Campus(UUID(int=2), district.id, 'Campus', '', False, campus_number='000001001')
    district.meta = campus.meta = {'Membership': 0, 'Masked': '<5', 'large': [1] * 10_000}
    repo.add_district(district)
    repo.add_campus(campus)
    statements = []

    def capture(connection, cursor, statement, parameters, context, executemany):
        if 'JOIN teadata_catalog' in statement:
            statements.append(statement)

    try:
        with factory.begin() as session:
            export_dataengine(repo, session, include_meta_entries=False)
        with factory() as session:
            restored = import_dataengine(session)
        event.listen(sql, 'before_cursor_execute', capture)
        for entity in (restored._districts[district.id], restored._campuses[campus.id]):
            assert {'Membership', 'Masked', 'large'} <= set(dir(entity))
            assert 'json_each(' in statements[-1]
            assert entity.Membership == 0
            assert ' -> ' in statements[-1]
            assert not entity.meta._loaded
        assert len(statements) == 4
    finally:
        event.remove(sql, 'before_cursor_execute', capture)
        sql.dispose()
