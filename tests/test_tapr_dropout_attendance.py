"""The 2024 and 2025 editions bind complete DROP_ATT exports and evidence."""

from hashlib import sha256
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from teadata.column_schema import parse_column_schema
from teadata.provenance import ProvenanceStore, validate_record_graph, validate_reference_records
from teadata.teadata_config import load_config


ROOT = Path(__file__).resolve().parents[1]
SAS_FOOTER_SHA256 = '76cc1ea0db298b4c2230783ea487a94c8979e1f3bdfeeb0ae53b536f1e087f91'


@pytest.mark.parametrize(('year', 'scope', 'level', 'row_count', 'width', 'receipt', 'footer'), [
    (2024, 'region', 'R', 20, 163, 'cd76be122e065f8f688bb346689e06f8b5c438416fcf096302f189042ce4a3de', 23),
    (2024, 'state', 'S', 1, 162, '578c57d1ebd7eb99d2a147174b19e88c91e1a32e365035dce28f320a0ba13398', None),
    (2025, 'campus', 'C', 9084, 166, '816805b8ef38d0d25b66d3f01d16cc9bd48c82c9e81bbf182fe28084a6773f10', None),
    (2025, 'district', 'D', 1208, 164, 'e4f57bf9d4f5e9be9673e27442c9d3bfe8cc0a73a81a861ebdd67f978f33ef7e', None),
    (2025, 'region', 'R', 20, 163, 'c82bf5f0f24934ad811a10973eff262aef31a8fb60c7efe3864b8274b4cdf5ba', None),
    (2025, 'state', 'S', 1, 162, '31f693774c18fb53bfd383a75c89495d0ae10fc25d91bd1e5ae90ff545f3a4fb', 4),
])
def test_drop_att_source_only_exact_publisher_contract(year, scope, level, row_count, width, receipt, footer, monkeypatch):
    dataset = f'{scope}_tapr_dropout_attendance'
    config = load_config()
    store = ProvenanceStore(ROOT / 'provenance')
    monkeypatch.setattr(store, 'verify_artifact', lambda *_: pytest.fail('Metadata discovery opened source bytes'))
    source = config.resolve_dataset(dataset, year, exact=True, reader_profile='source')
    assert source.metadata.provenance_ids == (receipt,)
    assert source.metadata.version == str(year) and source.metadata.period.kind == 'unknown'
    assert source.metadata.value_policy is None and not source.metadata.metrics
    assert config.is_source_only(dataset, year) and config.column_schema(dataset, year) is None
    assert config.availability_report()['data_sources'][dataset]['years'] == [2024, 2025]

    references = {ref.reference_id: ref for ref in source.metadata.references}
    roots = [receipt, *(rid for ref in references.values() for rid in ref.provenance_ids)]
    graph = validate_record_graph(store.read_record, roots)
    validate_reference_records(source, graph.__getitem__, verify_all_claimed=True)
    observation = graph[receipt]
    assert observation['dataset'] == dataset and observation['method'] == 'download'
    assert observation['requested_year'] == observation['resolved_year'] == year
    assert observation['downloaded_at'] and observation['parameters']['http_method'] == 'POST'
    fields = observation['parameters']['form_controls']
    suffix = f'{(year - 1) % 100:02}'
    groups = [f'{group}{suffix}' for group in ('AT', '0708DR', 'CA', '0912DR')]
    assert [value for name, value in fields if name == 'key'] == groups
    assert [value for name, value in fields if name == 'var_type'] == ['N', 'D', 'R']
    controls = dict(fields)
    assert (controls['ccyy'], controls['sumlev'], controls['dsname'], controls['datafmt']) == (str(year), level, 'DROP_ATT', 'csv')
    assert observation['source_metadata']['provenance_ids'] == [
        references['selection'].provenance_ids[0], references['dictionary'].provenance_ids[0],
    ]

    schema = references['columns_source']
    if scope in {'state', 'region'}:
        schema_record = graph[schema.provenance_ids[0]]
        assert schema_record['method'] == 'local_observation'
        parents = [graph[rid] for rid in schema_record['source_metadata']['provenance_ids']]
        witnesses = {record['artifact']['filename']: record for record in parents}
        for group in groups:
            record = witnesses[f'{year}-{level}-DROP_ATT-{group}.csv']
            controls = record['parameters']['form_controls']
            assert record['method'] == 'download' and record['parameters']['http_method'] == 'POST'
            assert [value for name, value in controls if name == 'key'] == [group]
            assert ['sumlev', level] in controls and ['ccyy', str(year)] in controls
        named = graph[references['named_report'].provenance_ids[0]]
        assert named in parents and named['method'] == 'download'
        query = parse_qs(urlsplit(named['requested_url']).query)
        assert (query['ccyy'], query['lev'], query['id']) == ([str(year)], [level], ['S' if scope == 'state' else '13'])
    assert schema.path is not None
    raw = (ROOT / 'teadata' / schema.path).read_bytes()
    assert sha256(raw).hexdigest() == schema.reviewed_sha256
    branch, = [entry for entry in parse_column_schema(raw)['datasets'] if entry['dataset'] == dataset]
    table, = branch['tables']
    assert table['input_sha256'] == observation['artifact']['sha256']
    annotations = [] if footer is None else [
        {'first_record': footer, 'last_record': footer, 'kind': 'footer', 'sha256': SAS_FOOTER_SHA256},
    ]
    assert table['layout'] == {'label_record': 1, 'code_record': 2, 'first_data_record': 3,
                               'last_data_record': row_count + 2, 'annotations': annotations}
    records = config.header_records(dataset, edition=str(year), reader_profile='source')
    assert len(records) == len({record['code'] for record in records}) == len({record['field'] for record in records}) == width
    assert len(table['dictionary_records']) == 108
    assert all(record['source_sha256'] == references['dictionary'].reviewed_sha256
               and record['publisher_context']['heading'] for record in table['dictionary_records'])
    missing = [record for record in records if record['dictionary_status'] == 'not_documented']
    assert len(missing) == 54
    assert all(record['disposition'] == 'fact' and record['publisher_type'] is None
               and record['publisher_description'] is None and record['dictionary_occurrences'] == []
               and record['period'] == {'kind': 'unknown'} for record in missing)
    dropout = next(record for record in records if record['code'] == f'{level}A0708DR{suffix}N')
    if year == 2024:
        assert dropout['label'] == f'{scope.title()} 2023 Annual Dropout for Grades 07-08: All Students Numerator'
    else:
        assert dropout['label'] == 'Annual Dropout Rate, Grades 07-08, SY 2023-24, All Students, Numerator'
