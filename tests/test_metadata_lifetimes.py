"""Transient metadata graphs must not need cyclic GC to release their owners."""
from contextlib import contextmanager
import gc
from hashlib import sha256
import weakref

import pytest

from teadata.manifest import ContractError, DatasetCatalog, canonical_json
from teadata.provenance import validate_record_graph

from test_catalog_projections import edition_fixture, make_edition_fixture, rehash_catalog  # noqa: F401


@contextmanager
def no_cyclic_gc():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()
        gc.collect()  # Clean up the deliberately failing pre-repair reproduction too.


def _catalog_reference(document, resources, key, operation):
    # The loader closes over bytes only, never this newly constructed catalog.
    catalog = DatasetCatalog(document, column_schema_loader=lambda ref: resources[ref.sha256],
                             edition_metadata_loader=lambda ref: resources[ref.sha256])
    reference = weakref.ref(catalog)
    if operation == 'metadata':
        catalog.metadata(key.dataset, edition=key)
    elif operation == 'variables':
        tuple(catalog.variables(edition=key))
    elif operation == 'projection':
        catalog.resolve_artifacts(((key, 'amount'),))
    elif operation == 'error':
        try:
            catalog.resolve_artifacts(((key, 'not_a_reported_field'),))
        except ContractError as error:
            assert 'unavailable' in str(error)
        else:
            raise AssertionError('Missing source field unexpectedly resolved')
    return reference  # No catalog/view/result/generator is returned as a strong owner.


@pytest.mark.parametrize('schema', [3, 4])
@pytest.mark.parametrize('operation', ['construct', 'metadata', 'variables', 'projection', 'error'])
def test_transient_catalog_graphs_release_without_a_gc_cycle(request, schema, operation):
    fixture = request.getfixturevalue('edition_fixture')
    original = fixture['catalog']
    resources = dict(fixture['resources'])
    if schema == 4:
        original = original.externalize_editions()
        for row in original.editions(include_metadata=False):
            raw = original.edition_metadata_bytes(row['edition_id'])
            resources[sha256(raw).hexdigest()] = raw
    document = original.to_dict()
    key = fixture['leaves']['student'][0]
    with no_cyclic_gc():
        reference = _catalog_reference(document, resources, key, operation)
        assert reference() is None


class Reader:
    def __init__(self, records):
        self.records = records

    def __call__(self, identifier):
        return self.records[identifier]


def _receipt_identifier(record):
    return sha256((canonical_json(record) + '\n').encode()).hexdigest()


def _acquisition():
    return {'schema_version': 1, 'kind': 'acquisition', 'dataset': 'example',
            'section': 'data_sources', 'requested_year': 2026, 'resolved_year': 2026,
            'source_metadata': {}, 'artifact': {'sha256': 'a' * 64, 'size_bytes': 0}}


def _reader_reference(records, identifier):
    reader = Reader(records)
    reference = weakref.ref(reader)
    try:
        validate_record_graph(reader, (identifier,))
    except ValueError as error:
        return reference, str(error)  # Retain no exception or traceback owner.
    return reference, None


@pytest.mark.parametrize('bad', [False, True])
def test_provenance_reader_releases_on_success_and_error_without_gc(bad):
    record = _acquisition()
    identifier = _receipt_identifier(record)
    if bad:
        record['artifact']['size_bytes'] = 1
    with no_cyclic_gc():
        reference, error = _reader_reference({identifier: record}, identifier)
        assert reference() is None
        assert error == ('Provenance receipt identity mismatch' if bad else None)


def test_deep_receipt_chain_keeps_postorder_without_the_recursion_limit():
    record = _acquisition()
    previous = _receipt_identifier(record)
    records = {previous: record}
    for _ in range(1_100):
        record = {'schema_version': 1, 'kind': 'processing', 'dataset': 'example',
                  'inputs': [previous], 'outputs': {'table': {'sha256': 'a' * 64, 'size_bytes': 0}},
                  'processor': 'fixture', 'code_version': 'fixture'}
        previous = _receipt_identifier(record)
        records[previous] = record
    actual = validate_record_graph(records.__getitem__, (previous,))
    assert list(actual) == list(records)


def test_deep_field_chain_keeps_projection_order_without_the_recursion_limit(tmp_path):
    source = make_edition_fixture(tmp_path, families=('student',), extra_fields=1_100)
    document = source['catalog'].to_dict()
    rows = [row for row in document['field_bindings'] if row['field'].startswith('extra_')]
    for first, second in zip(rows, rows[1:]):
        first['dependencies'] = [[second['edition_id'], second['field']]]
    catalog = rehash_catalog(document, source['resources'])
    key = source['leaves']['student'][0]
    plan, = catalog.resolve_artifacts(((key, rows[0]['field']),))
    assert plan['columns'] == ('campus_number', *(row['field'] for row in rows))
