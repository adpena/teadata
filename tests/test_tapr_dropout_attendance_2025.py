"""The 2025 edition binds complete raw DROP_ATT exports and dictionary evidence."""

from hashlib import sha256
from pathlib import Path

import pytest

from teadata.column_schema import parse_column_schema
from teadata.provenance import ProvenanceStore, validate_record_graph, validate_reference_records
from teadata.teadata_config import load_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(('scope', 'level', 'row_count', 'width', 'receipt'), [
    ('campus', 'C', 9084, 166, '816805b8ef38d0d25b66d3f01d16cc9bd48c82c9e81bbf182fe28084a6773f10'),
    ('district', 'D', 1208, 164, 'e4f57bf9d4f5e9be9673e27442c9d3bfe8cc0a73a81a861ebdd67f978f33ef7e'),
])
def test_2025_drop_att_source_only_exact_publisher_contract(scope, level, row_count, width, receipt, monkeypatch):
    dataset = f'{scope}_tapr_dropout_attendance'
    config = load_config()
    store = ProvenanceStore(ROOT / 'provenance')
    monkeypatch.setattr(store, 'verify_artifact', lambda *_: pytest.fail('Metadata discovery opened source bytes'))
    source = config.resolve_dataset(dataset, 2025, exact=True, reader_profile='source')
    assert source.metadata.provenance_ids == (receipt,)
    assert source.metadata.version == '2025' and source.metadata.period.kind == 'unknown'
    assert source.metadata.value_policy is None and not source.metadata.metrics
    assert config.is_source_only(dataset, 2025) and config.column_schema(dataset, 2025) is None
    assert config.availability_report()['data_sources'][dataset]['years'] == [2024, 2025]

    references = {ref.reference_id: ref for ref in source.metadata.references}
    roots = [receipt, *(rid for ref in references.values() for rid in ref.provenance_ids)]
    graph = validate_record_graph(store.read_record, roots)
    validate_reference_records(source, graph.__getitem__, verify_all_claimed=True)
    observation = graph[receipt]
    assert observation['dataset'] == dataset and observation['method'] == 'download'
    assert observation['requested_year'] == observation['resolved_year'] == 2025
    assert observation['downloaded_at'] and observation['parameters']['http_method'] == 'POST'
    fields = observation['parameters']['form_controls']
    assert [value for name, value in fields if name == 'key'] == ['AT24', '0708DR24', 'CA24', '0912DR24']
    assert [value for name, value in fields if name == 'var_type'] == ['N', 'D', 'R']
    controls = dict(fields)
    assert (controls['ccyy'], controls['sumlev'], controls['dsname'], controls['datafmt']) == ('2025', level, 'DROP_ATT', 'csv')
    assert observation['source_metadata']['provenance_ids'] == [
        references['selection'].provenance_ids[0], references['dictionary'].provenance_ids[0],
    ]

    schema = references['columns_source']
    assert schema.path is not None
    raw = (ROOT / 'teadata' / schema.path).read_bytes()
    assert sha256(raw).hexdigest() == schema.reviewed_sha256
    branch, = [entry for entry in parse_column_schema(raw)['datasets'] if entry['dataset'] == dataset]
    table, = branch['tables']
    assert table['input_sha256'] == observation['artifact']['sha256']
    assert table['layout'] == {'label_record': 1, 'code_record': 2, 'first_data_record': 3,
                               'last_data_record': row_count + 2, 'annotations': []}
    records = config.header_records(dataset, edition='2025', reader_profile='source')
    assert len(records) == len({record['code'] for record in records}) == len({record['field'] for record in records}) == width
    assert len(table['dictionary_records']) == 108
    assert all(record['source_sha256'] == references['dictionary'].reviewed_sha256
               and record['publisher_context']['heading'] for record in table['dictionary_records'])
    missing = [record for record in records if record['dictionary_status'] == 'not_documented']
    assert len(missing) == 54
    assert all(record['disposition'] == 'fact' and record['publisher_type'] is None
               and record['publisher_description'] is None and record['dictionary_occurrences'] == []
               and record['period'] == {'kind': 'unknown'} for record in missing)
    dropout = next(record for record in records if record['code'] == f'{level}A0708DR24N')
    assert dropout['label'] == 'Annual Dropout Rate, Grades 07-08, SY 2023-24, All Students, Numerator'
