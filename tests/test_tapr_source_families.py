"""Current source families retain labels, history and unbound strings."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import csv
from pathlib import Path
import socket
from urllib.parse import parse_qs, urlsplit

import pyarrow.parquet as pq
import pytest
import yaml

from teadata.column_schema import parse_column_schema, source_schema_fingerprint, validate_header_table
from teadata.dataset_metadata import DatasetMetadata, ResolvedDataset
from teadata.manifest import ContractError, DatasetCatalog, _validate_methodology, canonical_json
from teadata.provenance import ProvenanceStore, validate_record_graph, validate_reference_records
from teadata.scripts.build_tapr_profiles import prepare_profiles
from teadata.scripts.rename_tapr_campus_headers import sanitize_header
from teadata.tapr import TaprSourceRequest, qualified_tapr_family, read_tapr_table, reviewed_tapr_data_record_count
from teadata.teadata_config import Config, load_config


ROOT = Path(__file__).resolve().parents[1]
RESOURCE = ROOT / 'teadata/schemas/tapr--2024--reference-dropout-attendance-crosswalk.json'
UNKNOWN = {'kind': 'unknown'}
COMP4_GRADUATION_CUTOFF = {'kind': 'as_of', 'as_of': '2023-08-31'}
PROGRAM_SOURCES = [
    ('campus', 'annual_rhsp_dap_fhsp_e_fhsp_dla_texas_first_dla', 200, 9082, '4b7487ef04e74c8048317a75c784305793519c79c9aaf6df25742ad153a5d51d'),
    ('campus', 'longitudinal_rhsp_dap_fhsp_e_fhsp_dla_texas_first_dla', 158, 9082, 'ab64e994ba04f5930606f8f765e6ab15ae69b45f0e3409c9aad7ae2d7ce9d048'),
    ('district', 'annual_rhsp_dap_fhsp_e_fhsp_dla_texas_first_dla', 198, 1207, '59ed41d55bd88a7c24f0e111f4641c7a7a0033256ba69426cd2bf103bc3da5b2'),
    ('district', 'longitudinal_rhsp_dap_fhsp_e_fhsp_dla_texas_first_dla', 156, 1207, '0bd27394f8e7f1dde19062887bcdd6f4a3b14491805dd337bd8a0ac96e68f53e'),
]
EXTENDED_SOURCES = [
    ('campus', '5_year_extended_longitudinal_rate', 200, 9082, 'ddf82ad2dcbf50acaeb654379ad29f9aed665ad0ee96dd3347fd47d7dce18bd2'),
    ('campus', '6_year_extended_longitudinal_rate', 200, 9082, '4830b9a011ec4e6a073ee11c731b4e4a3ffe4880f709b02bef988aba013581a2'),
    ('district', '5_year_extended_longitudinal_rate', 198, 1207, '3b589f7efda3c3f4b3eacb394c46c648837287bca74c015eaded5b84ea3f9cc6'),
    ('district', '6_year_extended_longitudinal_rate', 198, 1207, 'cde1fa2f993995f92ec3925c626cce0438cc58148152f4a3f06bfbfcebed8505'),
]
SOURCES = [
    ('campus', 'reference', 22, 9082, '46bd6f225bfb3bc1aaf2a3fd733bb212d2e6ebd19e9e8c81486be15323e37f1b'),
    ('district', 'reference', 12, 1207, 'f0841681cbf8e28a91dfe31c9653b04cf4a6d933ad6bf30d334a603362b9d0b7'),
    ('campus', 'dropout_attendance', 166, 9082, '35a1fd68ec12257cf38a835a3cd63a8676affd81d1d4381c2c513b13ae2b0a0c'),
    ('district', 'dropout_attendance', 164, 1207, 'e05111f5db6e9b6e6477a4ac38b7a2e38c694f94312e5fa20c7082125fe863af'),
    ('campus', 'four_year_longitudinal_rate', 228, 9082, '1ecb3c44285694d03d6b054674ea235e1462a699a2fa8db64fc25e5286caab90'),
    ('district', 'four_year_longitudinal_rate', 226, 1207, '632d7f6c24087c3f557882724229ebc68318e510ac1f206181e39b5833aef16a'),
    ('campus', 'staar_school_progress', 2212, 9082, '751c3626c097b51f2d4467950aa7f77da976083fc63169d1c19268198bd5b96c'),
    ('district', 'staar_school_progress', 2210, 1207, '72df19ef8f42a1b404e80ef528a1f9cc9cc93cc27aa466891140ed2fc374fdfb'),
    *PROGRAM_SOURCES,
    *EXTENDED_SOURCES,
]
NEW_DATASETS = {scope + '_tapr_' + family for scope, family, *_ in SOURCES}
HISTORICAL_FAMILY_YEARS = {
    'dropout_attendance': [2024, 2025],
    'four_year_longitudinal_rate': [2024, 2025],
    'staar_school_progress': [2024, 2025, 2026],
}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def reject(*args, **kwargs):
        pytest.fail('Source-contract tests must not connect to a service')
    monkeypatch.setattr(socket.socket, 'connect', reject)
    monkeypatch.setattr(socket, 'create_connection', reject)


@pytest.mark.parametrize('scope,family,width,rows,raw_id', SOURCES)
def test_complete_source_only_discovery_and_typed_receipts_need_no_artifact_io(scope, family, width, rows, raw_id, monkeypatch):
    cfg = load_config()
    dataset = scope + '_tapr_' + family
    store = ProvenanceStore(ROOT / 'provenance')
    monkeypatch.setattr(store, 'verify_artifact', lambda *a, **k: pytest.fail('Metadata lookup opened an artifact'))
    monkeypatch.setattr('teadata.tapr.read_tapr_table', lambda *a, **k: pytest.fail('Discovery opened source rows'))
    source = cfg.resolve_dataset(dataset, 2024, reader_profile='source')
    metadata = source.metadata
    assert metadata.version == '2024' and metadata.period.to_dict() == UNKNOWN
    assert metadata.value_policy is None and metadata.estimation is None and metadata.field_year is None
    expected_metrics = 2 if family == 'four_year_longitudinal_rate' else 0
    assert len(metadata.metrics) == expected_metrics and metadata.provenance_ids == (raw_id,)
    _validate_methodology(metadata, None)
    assert cfg.is_source_only(dataset, 2024) and cfg.column_schema(dataset, 2024) is None
    assert cfg.availability_report()['data_sources'][dataset]['years'] == HISTORICAL_FAMILY_YEARS.get(family, [2024])
    assert cfg.reader_references(dataset, 2024, reader_profile='source') == list(metadata.references)
    roots = [raw_id, *(identifier for ref in metadata.references for identifier in ref.provenance_ids)]
    graph = validate_record_graph(store.read_record, roots)
    validate_reference_records(source, graph.__getitem__, verify_all_claimed=True)
    acquired = graph[raw_id]
    assert acquired['dataset'] == dataset and acquired['method'] == 'download'
    date_prefix = {'four_year_longitudinal_rate': '2026-11-01T09:06:',
                   'staar_school_progress': '2026-11-01T10:58:',
                   'annual_rhsp_dap_fhsp_e_fhsp_dla_texas_first_dla': '2026-11-02T02:41:',
                   'longitudinal_rhsp_dap_fhsp_e_fhsp_dla_texas_first_dla': '2026-11-02T02:41:',
                   '5_year_extended_longitudinal_rate': '2026-11-02T07:24:',
                   '6_year_extended_longitudinal_rate': '2026-11-02T07:24:'}.get(family, '2026-11-01T06:11:')
    assert acquired['downloaded_at'].startswith(date_prefix)
    assert acquired['response_headers']['content-type'] == 'text/&content_type.-separated-values'
    assert acquired['source_metadata']['period'] == UNKNOWN and acquired['source_metadata']['metrics'] == {}
    schema_ref = next(ref for ref in metadata.references if ref.role == 'schema')
    local = graph[schema_ref.provenance_ids[0]]
    assert local['method'] == 'local_observation' and local['downloaded_at'] is None
    assert local['requested_url'] is None and local['resolved_url'] is None
    assert schema_ref.path is not None
    assert local['artifact']['sha256'] == schema_ref.reviewed_sha256 == sha256((ROOT / 'teadata' / schema_ref.path).read_bytes()).hexdigest()
    records = cfg.header_records(dataset, edition='2024', reader_profile='source')
    assert len(records) == width == len({record['code'] for record in records}) == len({record['field'] for record in records})
    assert all(record['table'] == family for record in records)
    assert sum(record['period'] != UNKNOWN for record in records) == expected_metrics
    assert {record['field'] for record in records if record['disposition'] == 'identity'} == (
        {'campus_number', 'campus_name', 'district_number', 'district_name'} if scope == 'campus' else {'district_number', 'district_name'})
    contract = cfg.column_contract(dataset, 2024, reader_profile='source')
    assert contract is not None
    selected = next(item for item in parse_column_schema(contract['text'].encode())['datasets'] if item['dataset'] == dataset)
    assert selected['tables'][0]['layout'] == {'label_record': 1, 'code_record': 2,
        'first_data_record': 3, 'last_data_record': rows + 2, 'annotations': []}
    assert selected['tables'][0]['input_sha256'] == acquired['artifact']['sha256']
    # Test interpretation of the genuine observation without pretending that
    # new authoring metadata was the original acquisition metadata.
    observation_metadata = replace(DatasetMetadata.from_dict(acquired['source_metadata']), provenance_ids=(raw_id,))
    catalog = DatasetCatalog.create(datasets={dataset: {'status': 'loaded', 'section': 'data_sources',
        'metadata': observation_metadata.to_dict(), 'artifact': acquired['artifact'], 'reads': []}},
        records=graph, value_policies={}, metric_bindings={}, processor='0' * 64)
    with pytest.raises(ContractError, match='no value policy'):
        catalog.policy(dataset)
    with pytest.raises(ContractError, match='Reader profile is required'):
        cfg.load_df(dataset, 2024)
    if family in {'dropout_attendance', 'four_year_longitudinal_rate', 'staar_school_progress'}:
        identity_width = 4 if scope == 'campus' else 2
        expected_width = {'dropout_attendance': 162 + identity_width,
                          'staar_school_progress': 2304 + identity_width}.get(family, width)
        assert len(cfg.header_records(dataset, edition='2025', reader_profile='source')) == expected_width
        assert cfg.is_source_only(dataset, 2025)
    else:
        with pytest.raises(ContractError, match='edition unavailable'):
            cfg.header_records(dataset, edition='2025', reader_profile='source')
        with pytest.raises(ContractError, match='exact declared edition'):
            cfg.read_tapr_sources(dataset, 2025)


def test_human_readable_names_preserve_exact_code_label_maps_and_literal_lengths():
    cfg = load_config()
    expected = [('campus_tapr_reference', 'reference', 'C_RATING', '2024 Campus Overall Grade', 'campus_2024_overall_grade'),
        ('district_tapr_reference', 'reference', 'D_RATING', '2024 District Overall Grade', 'district_2024_overall_grade'),
        ('campus_tapr_dropout_attendance', 'dropout_attendance', 'CA0CA23N', '2023 campus Chronic Absenteeism All Students Group: Numerator', 'campus_2023_chronic_absenteeism_all_students_group_numerator'),
        ('district_tapr_dropout_attendance', 'dropout_attendance', 'DA0CA23N', '2023 district Chronic Absenteeism All Students Group: Numerator', 'district_2023_chronic_absenteeism_all_students_group_numerator')]
    for dataset, table, code, label, field in expected:
        assert cfg.header_mapping(dataset, edition='2024', table=table, reader_profile='source')[code] == field
        assert cfg.header_mapping(dataset, edition='2024', table=table, from_='field', to='label', reader_profile='source')[field] == label
    for scope in ['campus', 'district']:
        records = cfg.header_records(scope + '_tapr_reference', edition='2024', reader_profile='source')
        by_code = {record['code']: record for record in records}
        assert by_code['COUNTY']['field'] == 'county_number'
        assert by_code['REGION']['field'] == 'esc_region'
        assert by_code['REGION']['publisher_type'] == 'Character' and by_code['REGION']['publisher_length'] == '2'
        assert all(record['dictionary_status'] == 'documented' for record in records)
        assert all(record['dictionary_occurrences'][0]['source_sha256'] for record in records)
        records[0]['dictionary_occurrences'][0]['publisher_type'] = 'Caller mutation'
        assert cfg.header_records(scope + '_tapr_reference', edition='2024', reader_profile='source')[0]['dictionary_occurrences'][0]['publisher_type'] != 'Caller mutation'


@pytest.mark.parametrize('scope,missing_sha', [('campus', 'df20c2f8ff4841fc064d6e38085ee97ca44bbb722f37636b223806f5443b6793'),
    ('district', 'cc36466eed9e9d7c26977459909d15aa04aab7a8a6b47ffff0050e7c2107e401')])
def test_every_undocumented_fact_is_retained_without_guessed_type_or_semantics(scope, missing_sha):
    records = load_config().header_records(scope + '_tapr_dropout_attendance', edition='2024', reader_profile='source')
    missing = [record for record in records if record['dictionary_status'] == 'not_documented']
    assert len(missing) == 54
    assert sha256(canonical_json([record['code'] for record in missing]).encode()).hexdigest() == missing_sha
    assert all(record['disposition'] == 'fact' and record['publisher_type'] is None
               and record['publisher_description'] is None and record['dictionary_occurrences'] == []
               and record['period'] == UNKNOWN for record in missing)
    assert sum(record['dictionary_status'] == 'documented' for record in records) == 108


@pytest.mark.parametrize('damage', ['false_not_applicable', 'guessed_type', 'guessed_definition', 'code_alias', 'name_collision'])
def test_unknowns_and_reviewed_names_cannot_be_repaired_by_alias_or_inference(damage):
    table = deepcopy(parse_column_schema(RESOURCE.read_bytes())['datasets'][2]['tables'][0])
    field = next(field for field in table['fields'] if field['dictionary_status'] == 'not_documented')
    if damage == 'false_not_applicable':
        field['dictionary_status'] = 'not_applicable'
    elif damage == 'guessed_type':
        field['publisher_type'] = 'Number'
    elif damage == 'guessed_definition':
        field['publisher_description'] = 'Assumed from code suffix'
    elif damage == 'code_alias':
        field['code'] = table['fields'][4]['code']
    else:
        field['field'] = table['fields'][4]['field']
    with pytest.raises(ValueError, match='dictionary|Duplicate source code|checksum|collision'):
        validate_header_table(table)


def test_current_source_contracts_keep_explicit_edition_csv_opt_ins():
    current = yaml.safe_load((ROOT / 'teadata/teadata_sources.yaml').read_text())
    for dataset in NEW_DATASETS:
        family = dataset.split('_tapr_', 1)[1]
        assert list(current['data_sources'].pop(dataset)) == HISTORICAL_FAMILY_YEARS.get(family, [2024])
        assert current['schema']['data_sources'].pop(dataset) == ['csv']
    # Full pre-change Config preservation has one current baseline owner below.


STAAR_TESTED_GRADES = [
    (f'staar_tested_grade_{grade}', f'STAAR_GR{grade}', facts, occurrences, 0)
    for grade, facts, occurrences in [
        (3, 678, 704), (4, 678, 704), (5, 867, 900),
        (6, 678, 704), (7, 678, 704), (8, 1056, 1096),
    ]
]
STAAR_TESTED_GRADES_2024 = [
    (f'staar_tested_grade_{grade}', f'STAAR_GR{grade}', facts, occurrences, 0)
    for grade, facts, occurrences in [
        (3, 652, 678), (4, 652, 678), (5, 834, 867),
        (6, 652, 678), (7, 652, 678), (8, 1016, 1056),
    ]
]
STAAR_2024_WRAPPED = {('state', 'STAAR_GR3'), ('state', 'STAAR_GR6')}
STAAR_2024_WARNING_PARTS = {
    ('state', 'STAAR_GR3'): 3, ('state', 'STAAR_GR4'): 3,
    ('region', 'STAAR_GR5'): 4, ('region', 'STAAR_GR6'): 4,
    ('region', 'STAAR_GR7'): 3, ('district', 'STAAR_GR7'): 3,
    ('region', 'STAAR_GR8'): 4,
}
STAAR_2026 = STAAR_TESTED_GRADES + [
    ('staar_grades_3_to_8', 'STAAR_GR38', 300, 312, 0),
    ('staar_all_grades', 'STAAR_ALL', 945, 980, 0),
    ('staar_end_of_course', 'STAAR_EOC', 1134, 1176, 0),
    ('staar_school_progress', 'STAAR_SP', 2304, 2400, 0),
    ('staar_participation', 'PART', 2106, 2106, 0),
    ('staar_bilingual_esl_performance', 'BIL1', 527, 455, 72),
    ('staar_bilingual_esl_growth', 'BIL2', 270, 234, 36),
]


@pytest.mark.parametrize('year,family,publisher_file,facts,occurrences,undocumented', [
    *((2024, *row) for row in STAAR_TESTED_GRADES_2024),
    *((2025, *row) for row in STAAR_TESTED_GRADES),
    *((2026, *row) for row in STAAR_2026),
    (2024, 'staar_end_of_course', 'STAAR_EOC', 1092, 1134, 0),
    (2025, 'staar_end_of_course', 'STAAR_EOC', 1134, 1176, 0),
])
def test_staar_sources_keep_raw_scope_and_exact_receipts(
        year, family, publisher_file, facts, occurrences, undocumented, monkeypatch):
    cfg = load_config()
    store = ProvenanceStore(ROOT / 'provenance')
    monkeypatch.setattr(store, 'verify_artifact', lambda *a, **k: pytest.fail('Metadata lookup opened an artifact'))
    monkeypatch.setattr('teadata.tapr.read_tapr_table', lambda *a, **k: pytest.fail('Discovery opened source rows'))
    period = {'kind': 'school_year', 'start_year': year - 1, 'end_year': year}
    district_rows, campus_rows = {2024: (1207, 9082), 2025: (1208, 9084), 2026: (1202, 9105)}[year]
    scopes = [
        ('state', 1, ()),
        ('district', district_rows, ('district_number', 'district_name')),
        ('campus', campus_rows, ('campus_number', 'campus_name', 'district_number', 'district_name')),
        ('region', 20, ('region_number',)),
    ]
    for scope, rows, identity_fields in scopes:
        dataset = f'{scope}_tapr_{family}'
        source = cfg.resolve_dataset(dataset, year, exact=True, reader_profile='source')
        metadata = source.metadata
        assert metadata.version == str(year) and metadata.period.to_dict() == period
        assert not metadata.metrics and metadata.value_policy is None and metadata.estimation is None
        assert cfg.is_source_only(dataset, 2025) and cfg.is_source_only(dataset, 2026)
        if family.startswith('staar_tested_grade_') or publisher_file in {'STAAR_EOC', 'STAAR_SP', 'PART'}:
            assert cfg.data_sources[dataset].available_years() == [2024, 2025, 2026]
        elif scope == 'region':
            assert cfg.data_sources[dataset].available_years() == [2026]
        assert cfg.column_schema(dataset, year) is None
        with pytest.raises(ContractError, match='Reader profile is required'):
            cfg.load_df(dataset, year)
        request, = cfg.tapr_source_requests(dataset, year)
        assert qualified_tapr_family(dataset, request.key_field) == family
        assert request.key_field == (identity_fields[0] if identity_fields else None)
        letter = scope[0].upper()
        unprefixed_state = scope == 'state' and (year == 2026 or (year == 2025 and publisher_file == 'STAAR_GR3'))
        file_scope = '' if unprefixed_state else letter + '-'
        assert request.path.name == f'{year}-{file_scope}{publisher_file}-offered.csv'
        assert request.mapping['table'] == family and request.mapping['metric_period'] == period
        assert reviewed_tapr_data_record_count(request.mapping) == rows
        fields = request.mapping['fields']
        assert len(fields) == facts + len(identity_fields)
        assert tuple(field['field'] for field in fields if field['disposition'] == 'identity') == identity_fields
        if scope == 'region':
            assert fields[0]['code'] == 'REGION'
            assert fields[0]['label'] == ('2 Digit ESC Region' if year == 2024 else 'ESC Region #')
        fact_fields = fields[len(identity_fields):]
        documented = {field['code'] for field in fact_fields if field['dictionary_status'] == 'documented'}
        assert len(documented) == facts - undocumented

        references = {reference.reference_id: reference for reference in metadata.references}
        graph = validate_record_graph(store.read_record, [identifier for reference in references.values()
                                                         for identifier in reference.provenance_ids])
        validate_reference_records(source, graph.__getitem__, verify_all_claimed=True)
        raw_id, = references['csv_headers'].provenance_ids
        raw = graph[raw_id]
        assert raw['method'] == 'download' and raw['downloaded_at'] is not None
        assert raw['artifact']['sha256'] == request.mapping['input_sha256']
        controls = dict(raw['parameters']['form_controls'])
        assert (controls['ccyy'], controls['sumlev'], controls['dsname'], controls['datafmt']) == (
            str(year), letter, publisher_file, 'csv')
        masking = references['masking']
        if year in (2024, 2025):
            assert masking.status == 'available' and masking.role == 'masking_methodology'
            mask_id, = masking.provenance_ids
            assert graph[mask_id]['requested_url'] == f'https://rptsvr1.tea.texas.gov/perfreport/tapr/{year}/masking.html'
            assert [value for name, value in raw['parameters']['form_controls'] if name == 'var_type'] == ['N', 'D', 'R']
            dictionary_id, = references['dictionary'].provenance_ids
            params = parse_qs(urlsplit(graph[dictionary_id]['requested_url']).query)
            assert (params['ccyy'], params['sumlev'], params['dsname']) == ([str(year)], [letter], [publisher_file])
        if year == 2024 and publisher_file != 'STAAR_EOC':
            witness_id, = references['state_report_witness'].provenance_ids
            witness_query = parse_qs(urlsplit(graph[witness_id]['requested_url']).query)
            assert (witness_query['ccyy'], witness_query['lev'], witness_query['id'], witness_query['prgopt']) == (
                ['2024'], ['S'], ['S'], ['reports/tapr/performance.sas'])
            prefixes = [f'{scope.title()} 2024 {label}' for label in ('Domain 1A: ', '(STAAR Only) - ', '(STAAR & EOC) - ')]
            labels = [field['label'].lstrip('\n') for field in fact_fields]
            assert [sum(label.startswith(prefix) for label in labels) for prefix in prefixes] == [facts - 288, 72, 216]
            layout = request.mapping['layout']
            annotations = layout['annotations']
            if (scope, publisher_file) in STAAR_2024_WRAPPED:
                assert (layout['label_record'], layout['code_record'], layout['first_data_record']) == (None, 3, 4)
                preamble = annotations[0]
                assert (preamble['kind'], preamble['first_record'], preamble['last_record']) == ('preamble', 1, 2)
                annotations = annotations[1:]
            else:
                assert (layout['label_record'], layout['code_record'], layout['first_data_record']) == (1, 2, 3)
            partitions = STAAR_2024_WARNING_PARTS.get((scope, publisher_file), 0)
            assert len(annotations) == bool(partitions)
            if partitions:
                footer, = annotations
                footer_record = layout['last_data_record'] + 1
                assert footer == {'kind': 'footer', 'first_record': footer_record, 'last_record': footer_record,
                                  'sha256': '76cc1ea0db298b4c2230783ea487a94c8979e1f3bdfeeb0ae53b536f1e087f91'}
                schema_id, = references['columns_source'].provenance_ids
                witnesses = [graph[record] for record in graph[schema_id]['source_metadata']['provenance_ids']
                             if graph[record]['artifact']['filename'].startswith(f'2024-{letter}-{publisher_file}-partition-')]
                assert len(witnesses) == partitions
        elif year in (2024, 2025):
            witness_id, = references['named_report'].provenance_ids
            witness_query = parse_qs(urlsplit(graph[witness_id]['requested_url']).query)
            assert (witness_query['ccyy'], witness_query['lev'], witness_query['prgopt']) == (
                [str(year)], [letter], ['reports/tapr/performance.sas'])
            expected_id = {'state': 'S', 'region': '13', 'district': '227901', 'campus': '227901002'}
            if publisher_file == 'STAAR_EOC' or scope == 'region':
                assert witness_query['id'] == [expected_id[scope]]
            if year == 2025:
                assert all('SY 2024-25,' in field['label'] for field in fact_fields)
            if (year, scope, publisher_file) == (2025, 'region', 'STAAR_EOC'):
                layout = request.mapping['layout']
                assert (layout['label_record'], layout['code_record'], layout['first_data_record']) == (None, 3, 4)
                preamble, = layout['annotations']
                assert (preamble['kind'], preamble['first_record'], preamble['last_record']) == ('preamble', 1, 2)
        else:
            assert masking.status == 'unreviewed' and not masking.provenance_ids
            if scope == 'region':
                assert [value for name, value in raw['parameters']['form_controls'] if name == 'var_type'] == ['N', 'D', 'R']
                dictionary_id, = references['dictionary'].provenance_ids
                params = parse_qs(urlsplit(graph[dictionary_id]['requested_url']).query)
                assert (params['ccyy'], params['sumlev'], params['dsname']) == (['2026'], ['R'], [publisher_file])
                if family.startswith('staar_tested_grade_'):
                    witness_id, = references['regional_report'].provenance_ids
                    params = parse_qs(urlsplit(graph[witness_id]['requested_url']).query)
                    assert (params['ccyy'], params['lev'], params['id'], params['prgopt']) == (
                        ['2026'], ['R'], ['13'], ['reports/tapr/performance.sas'])
        if publisher_file == 'STAAR_SP':
            layout = request.mapping['layout']
            assert (layout['label_record'], layout['code_record'], layout['first_data_record']) == (None, 3, 4)
            preamble, = layout['annotations']
            assert (preamble['kind'], preamble['first_record'], preamble['last_record']) == ('preamble', 1, 2)
            assert sum(field['label'].startswith('\n') for field in fact_fields) == 2
            schema_id, = references['columns_source'].provenance_ids
            witnesses = [graph[record] for record in graph[schema_id]['source_metadata']['provenance_ids']
                         if '-batch-' in graph[record]['artifact']['filename']]
            assert len(witnesses) == 16
            assert {scope: sum(dict(record['parameters']['form_controls'])['sumlev'] == scope
                               for record in witnesses) for scope in 'SDCR'} == dict.fromkeys('SDCR', 4)
        dictionary = request.mapping['dictionary_records']
        assert len(dictionary) == occurrences
        assert {record['source_sha256'] for record in dictionary} == {references['dictionary'].reviewed_sha256}
        complete_definitions = {record['code'] for record in dictionary
                                if record['publisher_type'] is not None and record['publisher_description'] is not None}
        assert complete_definitions & {field['code'] for field in fact_fields} == documented


def test_regional_staar_addition_preserves_existing_2026_scope_crosswalks():
    fingerprint = sha256()
    for family, *_ in sorted(STAAR_2026):
        schema = ROOT / 'teadata/schemas' / f'tapr--2026--{family.replace("_", "-")}-crosswalk.json'
        branches = parse_column_schema(schema.read_bytes())['datasets']
        regional, = [branch for branch in branches if branch['dataset'] == f'region_tapr_{family}']
        unchanged = [branch for branch in branches if branch is not regional]
        fingerprint.update(family.encode() + b'\0')
        fingerprint.update(canonical_json(unchanged).encode() + b'\0')
    assert fingerprint.hexdigest() == '4dd3bebb3a26d11e6e5ef6fc9929b564e3dfa51f0998c85b9e50a6687b84ec69'


@pytest.mark.parametrize('year,family,publisher_file,program,facts,dictionary_only,scopes', [
    (2024, 'staar_school_progress', 'STAAR_SP', 'progress', 2208, 192, ('state', 'region')),
    (2024, 'staar_participation', 'PART', 'participation', 2028, 0, ('state', 'region', 'district', 'campus')),
    (2025, 'staar_school_progress', 'STAAR_SP', 'progress', 2304, 96, ('state', 'region', 'district', 'campus')),
    (2025, 'staar_participation', 'PART', 'participation', 2106, 0, ('state', 'region', 'district', 'campus')),
])
def test_historical_progress_participation_bind_logical_records_and_source_witnesses(
        year, family, publisher_file, program, facts, dictionary_only, scopes):
    cfg = load_config()
    store = ProvenanceStore(ROOT / 'provenance')
    for scope in scopes:
        dataset = f'{scope}_tapr_{family}'
        source = cfg.resolve_dataset(dataset, year, exact=True, reader_profile='source')
        request, = cfg.tapr_source_requests(dataset, year)
        layout = request.mapping['layout']
        assert (layout['label_record'], layout['code_record'], layout['first_data_record']) == (1, 2, 3)
        assert not layout['annotations']
        assert cfg.is_source_only(dataset, year) and cfg.column_schema(dataset, year) is None
        assert source.metadata.value_policy is None and not source.metadata.metrics
        assert source.metadata.period.to_dict() == {'kind': 'school_year', 'start_year': year - 1, 'end_year': year}
        assert cfg.data_sources[dataset].available_years() == [2024, 2025, 2026]
        fields = {field['code'] for field in request.mapping['fields'] if field['disposition'] == 'fact'}
        dictionary = {record['code'] for record in request.mapping['dictionary_records']}
        assert len(fields) == facts and fields <= dictionary and len(dictionary - fields) == dictionary_only
        refs = {ref.reference_id: ref for ref in source.metadata.references}
        graph = validate_record_graph(store.read_record, [rid for ref in refs.values() for rid in ref.provenance_ids])
        validate_reference_records(source, graph.__getitem__, verify_all_claimed=True)
        raw, = [graph[rid] for rid in refs['csv_headers'].provenance_ids]
        controls = dict(raw['parameters']['form_controls'])
        letter = scope[0].upper()
        assert (controls['ccyy'], controls['sumlev'], controls['dsname']) == (str(year), letter, publisher_file)
        assert raw['artifact']['sha256'] == request.mapping['input_sha256']
        named, = [graph[rid] for rid in refs['named_report'].provenance_ids]
        query = parse_qs(urlsplit(named['requested_url']).query)
        assert (query['ccyy'], query['lev'], query['prgopt']) == ([str(year)], [letter], [f'reports/tapr/{program}.sas'])


def test_latest_source_opt_ins_preserve_unaffected_preexisting_config():
    current = yaml.safe_load((ROOT / 'teadata/teadata_sources.yaml').read_text())
    selected_students = 'tprs_student_membership_enrollment_reports'
    selected_entry = current['data_sources'].pop(selected_students)
    assert list(selected_entry) == [2026, 'latest'] and selected_entry['latest'] == 2026
    assert current['schema']['data_sources'].pop(selected_students) == ['html']
    assert current['value_policies'].pop('tprs_student_selected_2026')['reference_id'] == 'legend'
    previous_tested_grades = {}
    for scope in ('state', 'district', 'campus', 'region'):
        for family, *_ in STAAR_TESTED_GRADES:
            dataset = f'{scope}_tapr_{family}'
            current['data_sources'][dataset].pop(2024)
            previous_tested_grades[dataset] = current['data_sources'][dataset]
        historical_eoc = current['data_sources'][f'{scope}_tapr_staar_end_of_course']
        for year in (2024, 2025):
            assert historical_eoc.pop(year)['metadata']['version'] == str(year)
        participation = current['data_sources'][f'{scope}_tapr_staar_participation']
        progress = current['data_sources'][f'{scope}_tapr_staar_school_progress']
        for year in (2024, 2025):
            assert participation.pop(year)['metadata']['version'] == str(year)
            if scope in {'state', 'region'} or year == 2025:
                assert progress.pop(year)['metadata']['version'] == str(year)
    assert sha256(canonical_json(previous_tested_grades).encode()).hexdigest() == '64178e073e15880351cfd020b6bb0c9d78b22c1d346fb5c1c81ce7f05975f0cb'
    for family, *_ in STAAR_2026:
        regional = 'region_tapr_' + family
        years = [2025, 2026] if family.startswith('staar_tested_grade_') else [2026]
        assert list(current['data_sources'].pop(regional)) == years
        assert current['schema']['data_sources'].pop(regional) == ['csv']
    for scope in ('state', 'district', 'campus'):
        for family, *_ in STAAR_2026:
            dataset = f'{scope}_tapr_{family}'
            if family == 'staar_school_progress' and scope != 'state':
                assert list(current['data_sources'][dataset]) == [2024, 2026]
                current['data_sources'][dataset].pop(2026)
                continue
            years = [2025, 2026] if family.startswith('staar_tested_grade_') else [2026]
            assert list(current['data_sources'].pop(dataset)) == years
            assert current['schema']['data_sources'].pop(dataset) == ['csv']
    # District transfers have an independent source contract and native writer.
    assert list(current['data_sources'].pop('district_transfer_reports')) == [2024, 2025, 2026]
    assert current['schema']['data_sources'].pop('district_transfer_reports') == ['parquet']
    # Student enrollment and programs have their own exact source contracts.
    for dataset in ('peims_student_enrollment_by_ethnicity', 'peims_student_programs_and_special_populations'):
        assert list(current['data_sources'].pop(dataset)) == [2023, 2024, 2025, 2026]
        assert current['schema']['data_sources'].pop(dataset) == ['csv']
        assert current['value_policies'].pop(dataset + '_2026')['reference_id'] == 'peims_masking'
        for year in (2023, 2024, 2025):
            reference = 'about' if dataset.endswith('_ethnicity') else 'peims_masking'
            assert current['value_policies'].pop(f'{dataset}_{year}')['reference_id'] == reference
    primary_disability = 'peims_special_education_by_primary_disability'
    assert list(current['data_sources'].pop(primary_disability)) == [2023, 2024, 2025, 2026]
    assert current['schema']['data_sources'].pop(primary_disability) == ['csv']
    assert current['value_policies'].pop('peims_primary_disability_csv')['reference_id'] == 'peims_disability_masking'
    staff = 'peims_staff_fte_and_salary'
    assert list(current['data_sources'].pop(staff)) == [2023, 2024, 2025, 2026]
    assert current['schema']['data_sources'].pop(staff) == ['csv']
    assert current['value_policies'].pop('peims_staff_csv')['reference_id'] == 'staff_definition'
    courses = 'peims_teacher_course_enrollment'
    assert list(current['data_sources'].pop(courses)) == [2023, 2024, 2025, 2026]
    assert current['schema']['data_sources'].pop(courses) == ['csv']
    assert current['value_policies'].pop('peims_course_csv')['reference_id'] == 'course_criteria'
    indicator = 'spp_timely_initial_evaluation'
    assert list(current['data_sources'].pop(indicator)) == [2024, 2025]
    assert current['schema']['data_sources'].pop(indicator) == ['xlsx']
    assert current['value_policies'].pop('spp_ind11_workbook')['reference_id'] == 'spp_masking'
    actuals = 'peims_financial_single_file_actuals'
    assert list(current['data_sources'].pop(actuals)) == [2023, 2024, 2025]
    assert current['schema']['data_sources'].pop(actuals) == ['zip']
    budgets = 'peims_financial_single_file_budgets'
    assert list(current['data_sources'].pop(budgets)) == [2026]
    assert current['schema']['data_sources'].pop(budgets) == ['zip']
    per_pupil = 'frc_per_pupil_expenditure'
    assert list(current['data_sources'].pop(per_pupil)) == [2025]
    assert current['schema']['data_sources'].pop(per_pupil) == ['html']
    assert current['value_policies'].pop('frc_ppe_published')['reference_id'] == 'frc_ppe_blank_note'
    research = 'accountability_research_four_year_graduation'
    assert list(current['data_sources'].pop(research)) == [2025]
    assert current['schema']['data_sources'].pop(research) == ['xlsx']
    assert current['value_policies'].pop('research_graduation_class_2025')['reference_id'] == 'publisher_disclosure'
    annual_dropout = 'accountability_research_annual_dropout'
    assert list(current['data_sources'].pop(annual_dropout)) == [2025]
    assert current['schema']['data_sources'].pop(annual_dropout) == ['xlsx']
    # Remove only the explicitly added 2025 outcome opt-ins before reconstructing history.
    for scope in ('campus', 'district'):
        for family in ('college_career_military_readiness', 'college_readiness_components'):
            dataset = scope + '_tapr_' + family
            assert list(current['data_sources'].pop(dataset)) == [2025]
            assert current['schema']['data_sources'].pop(dataset) == ['csv']
    for family in ('college_career_military_readiness', 'college_readiness_components',
                   'four_year_longitudinal_rate', 'student_profile'):
        statewide = 'state_tapr_' + family
        assert list(current['data_sources'].pop(statewide)) == [2025]
        assert current['schema']['data_sources'].pop(statewide) == ['csv']
    for scope in ('state', 'region'):
        dataset = scope + '_tapr_dropout_attendance'
        assert list(current['data_sources'].pop(dataset)) == [2024, 2025]
        assert current['schema']['data_sources'].pop(dataset) == ['csv']
    for scope in ('campus', 'district'):
        current['data_sources'][scope + '_tapr_dropout_attendance'].pop(2025)
        versions = current['data_sources'][scope + '_tapr_four_year_longitudinal_rate']
        versions.pop(2025)
        # The reviewed 2024 COMP4 metric bindings and report evidence have dedicated checks.
        metadata = versions[2024]['metadata']
        metadata.pop('notes')
        metadata['metrics'] = {}
        metadata['references'] = [
            ref for ref in metadata['references'] if not ref['reference_id'].startswith('comp4_')
        ]
    for scope, family, *_ in EXTENDED_SOURCES:
        dataset = scope + '_tapr_' + family
        current['data_sources'].pop(dataset)
        current['schema']['data_sources'].pop(dataset)
    # Exclude the deliberately revised TAPR entries and official private roster.
    # Their mappings, periods and provenance have dedicated source-grounded checks.
    # Reconstruct the retired TEPSAC raw-reader declaration only for the historical
    # fingerprint. Its immutable observation preserves an unknown period; the old
    # 2025 catalog key is not evidence of a measurement school year.
    assert 'tepsac' not in current['data_sources'] and 'tepsac' not in current['schema']['data_sources']
    retired_id = 'f5d62240acddb207b6cab8a8868777a6a5b6e38e7ae0e8351c28cb4c51b9c812'
    retired = ProvenanceStore(ROOT / 'provenance').read_record(retired_id)
    assert retired['dataset'] == 'tepsac' and retired['method'] == 'local_observation'
    assert retired['artifact']['sha256'] == 'f378d8e2fb2994223cad5c652c80f55b7319632f2d2627622d3afccaa9c18820'
    retired_metadata = deepcopy(retired['source_metadata'])
    assert retired_metadata['period'] == {'kind': 'unknown'}
    retired_metadata['provenance_ids'] = [
        'f28b6ae19f550a8243d84adb50f8bc63a974d6ecdfac72b8bf897d8b568b5611', retired_id,
    ]
    retired_schema, = [ref for ref in retired_metadata['references'] if ref['reference_id'] == 'columns_default']
    current['data_sources']['tepsac'] = {2025: {
        'path': 'data/tepsac/' + retired['artifact']['filename'], 'metadata': retired_metadata,
        'reader': {'dtype': {'district_number': 'str', 'school_number': 'str'},
                   'column_schema': retired_schema['path']},
    }}
    current['schema']['data_sources']['tepsac'] = ['csv']
    # Freeze all remaining cd837181 content, including 2024 and legacy bindings.
    for scope in ('campus', 'district'):
        for family in ('student_staff_profile', 'student_profile'):
            current['data_sources'][scope + '_tapr_' + family].pop(2025)
    private = current['data_sources']['tefa']
    assert set(private) == {2025, 2027}
    private.pop(2027)
    legacy_reader = private[2025]['reader']
    assert set(legacy_reader) == {'profiles'}
    assert set(legacy_reader['profiles']) == {'source'}
    # Reconstruct only the old reader projection for this historical comparison;
    # the retained workbook is now an explicit source-profile opt-in.
    private[2025]['reader'] = legacy_reader['profiles']['source']
    assert current['schema']['data_sources']['tefa'] == ['xlsx', 'json']
    current['schema']['data_sources']['tefa'] = ['xlsx']
    # Reconstruct only the intentionally revised spatial source contracts. Their
    # current editions and reader schema are checked by test_spatial_sources.
    spatial_years = current['options'].pop('spatial_years')
    assert spatial_years == {'districts': 2026, 'campuses': 2025}
    assert all(type(year) is int for year in spatial_years.values())
    assert set(current['spatial']['districts']) == {2025, 2026}
    current['spatial']['districts'].pop(2026)
    assert set(current['spatial']['campuses']) == {2025}
    historical_id = '0921b22eeca7d6b5c9211af116d2fcf5b8554103228b52b27a39118ed37196c1'
    # read_record verifies the immutable record's content ID, not the raw GIS body.
    historical = ProvenanceStore(ROOT / 'provenance').read_record(historical_id)
    assert historical['kind'] == 'acquisition' and historical['dataset'] == 'campuses'
    assert historical['section'] == 'spatial' and historical['resolved_year'] == 2025
    assert historical['artifact']['sha256'] == '96ab1476d745eafee42a0fa8c2f8d934996a3f45d886816ea8de8753c46a466b'
    metadata = deepcopy(historical['source_metadata'])
    metadata['provenance_ids'] = ['b781fe9fd52b060b9336814df0932a22ec07ce47801998b7a24f7160adbc571a', historical_id]
    schema, = [ref for ref in metadata['references'] if ref['reference_id'] == 'columns_default']
    campus = current['spatial']['campuses'][2025]
    campus['path'] = 'data/shapes/' + historical['artifact']['filename']
    campus['metadata'] = metadata
    campus['reader']['column_schema'] = schema['path']
    assert sha256(canonical_json(current).encode()).hexdigest() == 'bb96a9eab62808caecf37764a6e7096c6b9537420a76fef0ff4bba5d285f6cf8'


@pytest.mark.parametrize('scope,dictionary_sha,fields_sha', [
    ('campus', '21d9553b4285dc500dff9233e4877bee02bc649dd5d5f56bae2a49269a6b1a5a',
     '3e082219afcfbe1cd5ab182ead9dbfee656904f84b5addac2a4bf5efd0e7b68b'),
    ('district', '78c327c3a450b1be934872c9afa0146c21eb10a1b8223469eec99ad69dd74e36',
     'cfe0fae64d4a3c9b8be12b45bd98e64ae33af701ed7a8e2894a474a3f70e604d'),
])
def test_staar_school_progress_preserves_all_dictionary_occurrences_without_fabricated_columns(scope, dictionary_sha, fields_sha):
    cfg = load_config()
    dataset = scope + '_tapr_staar_school_progress'
    contract = cfg.column_contract(dataset, 2024, reader_profile='source')
    assert contract is not None
    selected = next(item for item in parse_column_schema(contract['text'].encode())['datasets'] if item['dataset'] == dataset)
    table, = selected['tables']
    occurrences = table['dictionary_records']
    fields = table['fields']
    assert sha256(canonical_json(occurrences).encode()).hexdigest() == dictionary_sha
    assert sha256(canonical_json([[field[key] for key in ('position', 'code', 'label', 'field', 'disposition')]
                                  for field in fields]).encode()).hexdigest() == fields_sha
    assert len(occurrences) == 2400 and [row['ordinal'] for row in occurrences] == list(range(1, 2401))
    assert sum(row['publisher_type'] is not None and row['publisher_description'] is not None for row in occurrences) == 2304
    actual_codes = {field['code'] for field in fields}
    unreturned = [row for row in occurrences if row['code'] not in actual_codes]
    assert len(unreturned) == 192
    assert sum(row['publisher_type'] is not None and row['publisher_description'] is not None for row in unreturned) == 96
    assert sum(row['publisher_type'] is None and row['publisher_description'] is None for row in unreturned) == 96
    records = cfg.header_records(dataset, edition='2024', reader_profile='source')
    assert len(records) == (2212 if scope == 'campus' else 2210)
    facts = [row for row in records if row['disposition'] == 'fact']
    identities = [row for row in records if row['disposition'] == 'identity']
    assert len(facts) == 2208 and all(row['dictionary_status'] == 'documented' for row in facts)
    assert all(row['field'] == sanitize_header(row['label']) and row['period'] == UNKNOWN for row in facts)
    assert all(len(row['dictionary_occurrences']) == 1 for row in facts)
    assert all(row['dictionary_status'] == 'not_applicable' and row['dictionary_occurrences'] == [] for row in identities)
    assert cfg.column_schema(dataset, 2024, reader_profile='source') == [row['field'] for row in fields]
    mapping = cfg.header_mapping(dataset, edition='2024', table='staar_school_progress', reader_profile='source')
    assert not {row['code'] for row in unreturned} & set(mapping)
    assert all('publisher_context' not in row for row in occurrences)  # Original extraction has no section-context claim.
    metadata = cfg.resolve_dataset(dataset, 2024, reader_profile='source').metadata
    assert metadata.value_policy is None and not metadata.metrics and metadata.estimation is None and metadata.field_year is None
    assert metadata.period.to_dict() == UNKNOWN
    for role in ['glossary', 'masking_methodology']:
        reference, = [ref for ref in metadata.references if ref.role == role]
        original, = [ref for ref in cfg.resolve_dataset(scope + '_tapr_four_year_longitudinal_rate', 2024, reader_profile='source').metadata.references if ref.role == role]
        assert reference.to_dict() == original.to_dict() and reference.status == 'available'
    assert 'field-specific STAAR School Progress masking applicability remains unreviewed' in metadata.notes


@pytest.mark.parametrize('scope,prefix', [('campus', 'C'), ('district', 'D')])
def test_comp4_binds_only_the_two_publisher_verified_graduation_rates(scope, prefix):
    cfg = load_config()
    dataset = scope + '_tapr_four_year_longitudinal_rate'
    records = cfg.header_records(dataset, edition='2024', reader_profile='source')
    facts = [record for record in records if record['disposition'] == 'fact']
    identities = [record for record in records if record['disposition'] == 'identity']
    assert len(facts) == 224 and all(record['dictionary_status'] == 'documented' for record in facts)
    assert all(record['dictionary_status'] == 'not_applicable' and record['dictionary_occurrences'] == [] for record in identities)
    ordinary = '4-Year Longitudinal Rate (Gr 9-12)'
    federal = '4-Year Federal Graduation Rate Without Exclusions (Gr 9-12)'
    headings = [record['dictionary_occurrences'][0]['publisher_context']['heading'] for record in facts]
    assert headings.count(ordinary) == 182 and headings.count(federal) == 42
    grouped = {}
    for record in facts:
        occurrence, = record['dictionary_occurrences']
        context = occurrence['publisher_context']
        assert context['source_column'] == 0
        assert context['source_line'] == (2492 if context['heading'] == federal else 2272)
        assert (context['source_line'], context['source_column']) < (occurrence['source_line'], occurrence['source_column'])
        assert ('federal_graduation_rate_without_exclusions_gr_9_12' in record['field']) == (context['heading'] == federal)
        grouped.setdefault(record['label'], []).append(record)
    pairs = [group for group in grouped.values() if len(group) == 2]
    assert len(pairs) == 42
    assert all({record['dictionary_occurrences'][0]['publisher_context']['heading'] for record in pair} == {ordinary, federal} for pair in pairs)
    by_code = {record['code']: record for record in facts}
    assert by_code[prefix + 'AGC4X23N']['field'] == scope + '_2023_4_year_longitudinal_graduates_for_all_students_numerator'
    assert by_code[prefix + 'AGC423N']['field'] == scope + '_2023_4_year_federal_graduation_rate_without_exclusions_gr_9_12_graduates_for_all_students_numerator'
    with pytest.raises(ContractError, match='Ambiguous header mapping'):
        cfg.header_mapping(dataset, edition='2024', table='four_year_longitudinal_rate', from_='label', reader_profile='source')
    by_code[prefix + 'AGC423N']['dictionary_occurrences'][0]['publisher_context']['heading'] = 'Caller change'
    fresh = cfg.header_records(dataset, edition='2024', reader_profile='source')
    assert next(record for record in fresh if record['code'] == prefix + 'AGC423N')['dictionary_occurrences'][0]['publisher_context']['heading'] == federal
    source = cfg.resolve_dataset(dataset, 2024, reader_profile='source')
    for role in ['glossary', 'masking_methodology']:
        reference, = [reference for reference in source.metadata.references if reference.role == role]
        original, = [reference for reference in cfg.resolve_dataset('campus_tapr_reference', 2024, reader_profile='source').metadata.references if reference.role == role]
        assert reference.to_dict() == original.to_dict() and reference.status == 'available'
        identifier, = reference.provenance_ids
        acquired = ProvenanceStore(ROOT / 'provenance').read_record(identifier)
        assert acquired['method'] == 'download' and acquired['artifact']['sha256'] == reference.reviewed_sha256
        assert acquired['requested_url'] == reference.url
        assert acquired['downloaded_at'].startswith('2026-10-31T17:11:09.')
    store = ProvenanceStore(ROOT / 'provenance')
    roots = [*source.metadata.provenance_ids, *(identifier for ref in source.metadata.references for identifier in ref.provenance_ids)]
    graph = validate_record_graph(store.read_record, roots)
    validate_reference_records(source, graph.__getitem__, verify_all_claimed=True)
    expected_metrics = {
        'four_year_longitudinal_graduation_rate_all_students': (
            prefix + 'AGC4X23R', scope + '_2023_4_year_longitudinal_graduates_for_all_students_rate', ordinary),
        'four_year_federal_graduation_rate_without_exclusions_all_students': (
            prefix + 'AGC423R',
            scope + '_2023_4_year_federal_graduation_rate_without_exclusions_gr_9_12_graduates_for_all_students_rate',
            federal),
    }
    assert source.metadata.metrics.keys() == expected_metrics.keys()
    for metric, (code, field, heading) in expected_metrics.items():
        assert source.metadata.metrics[metric].to_dict() == {
            'field': field, 'period': COMP4_GRADUATION_CUTOFF, 'unit': 'percent'}
        assert by_code[code]['field'] == field and by_code[code]['period'] == COMP4_GRADUATION_CUTOFF
        assert by_code[code]['dictionary_occurrences'][0]['publisher_context']['heading'] == heading
    reviewed_codes = {code for code, _, _ in expected_metrics.values()}
    assert all(record['period'] == UNKNOWN for record in records if record['code'] not in reviewed_codes)
    assert source.metadata.period.to_dict() == UNKNOWN
    assert source.metadata.value_policy is None and source.metadata.estimation is None and source.metadata.field_year is None
    report_id, entity, level = ('comp4_garland_high_report', '057909002', 'C') if scope == 'campus' else (
        'comp4_austin_isd_report', '227901', 'D')
    expected_reports = {report_id: (entity, level), 'comp4_marathon_report': ('022902001', 'C')}
    witnesses = {ref.reference_id: ref for ref in source.metadata.references if ref.reference_id.startswith('comp4_')}
    assert witnesses.keys() == expected_reports.keys()
    for name, (entity, level) in expected_reports.items():
        reference = witnesses[name]
        identifier, = reference.provenance_ids
        record = graph[identifier]
        assert reference.role == 'source_notes' and reference.status == 'available'
        assert reference.url and f'&ccyy=2024&id={entity}&lev={level}&' in reference.url
        assert record['method'] == 'download' and record['requested_url'] == reference.url
        assert record['artifact']['sha256'] == reference.reviewed_sha256
    _validate_methodology(source.metadata, None)


@pytest.mark.parametrize('scope,family,width,rows,raw_id', PROGRAM_SOURCES)
def test_program_headers_and_dictionary_only_gaps_preserve_publisher_evidence(scope, family, width, rows, raw_id):
    cfg = load_config()
    dataset = scope + '_tapr_' + family
    annual = family.startswith('annual_')
    expected = {
        ('campus', True): ('4f495ae87a27aee4a10d8e5af053efc735d5aca960fea6cd45ac1e337aa3aaa9',
                           'b9cdc9ef03834ee7363acf3a6fad2ccfeadd970f6d7de4bbe0d3454ed267e46e'),
        ('campus', False): ('7842d89c4ec6cde8d75fe1595bb3585f487b3dd92883a9c0fd4e2a2195024097',
                            '93799ab9821e332948628ac64669b9fe3c02bc4457cf688b14dadf3f947d3905'),
        ('district', True): ('62d08ce1499261722c17efe95f958165bce4b8454ccbaac03a3c1205f752cf66',
                             'f8cf5b928c2cb4d6f8364439b54c9eee28bb01a10f4c78e61fcc5e20d8e5536b'),
        ('district', False): ('9c0c90d4e03445d7181f3feba3c73b716d204e639158a4fcc4a107c47754a610',
                              'e6476804ca4b385b5af0109481c60d40205a5d7fcfc0533065275de84f4166c9'),
    }
    contract = cfg.column_contract(dataset, 2024, reader_profile='source')
    assert contract is not None
    branch, = [branch for branch in parse_column_schema(contract['text'].encode())['datasets'] if branch['dataset'] == dataset]
    table, = branch['tables']
    fields, occurrences = table['fields'], table['dictionary_records']
    field_sha, occurrence_sha = expected[scope, annual]
    assert sha256(canonical_json([[field[key] for key in ('position', 'code', 'label', 'field', 'disposition')]
                                  for field in fields]).encode()).hexdigest() == field_sha
    assert sha256(canonical_json(occurrences).encode()).hexdigest() == occurrence_sha
    assert len(fields) == width and reviewed_tapr_data_record_count(table) == rows
    assert len(occurrences) == (210 if annual else 204)
    assert all('publisher_length' not in field for field in fields)
    facts = [field for field in fields if field['disposition'] == 'fact']
    assert all(field['field'] == sanitize_header(field['label']) for field in facts)
    unknown = [field for field in facts if field['dictionary_status'] == 'not_documented']
    assert len(unknown) == (14 if annual else 0)
    assert all(field['publisher_type'] is None and field['publisher_description'] is None for field in unknown)
    actual_codes = {field['code'] for field in fields}
    absent = [occurrence for occurrence in occurrences if occurrence['code'] not in actual_codes]
    assert len(absent) == (0 if annual else 22)
    assert all(occurrence['publisher_type'] is None and occurrence['publisher_description'] is None for occurrence in absent)
    mapping = cfg.header_mapping(dataset, edition='2024', table=family, reader_profile='source')
    assert not {occurrence['code'] for occurrence in absent} & set(mapping)
    projected = cfg.header_records(dataset, edition='2024', reader_profile='source')
    assert all(record['dictionary_occurrences'] == [row for row in occurrences if row['code'] == record['code']]
               for record in projected)
    assert all(not record['dictionary_occurrences'] for record in projected if record['code'] in {field['code'] for field in unknown})
    known = next(record for record in projected if record['dictionary_occurrences'])
    known['dictionary_occurrences'][0]['publisher_description'] = 'Caller mutation'
    fresh = cfg.header_records(dataset, edition='2024', reader_profile='source')
    assert next(record for record in fresh if record['code'] == known['code'])['dictionary_occurrences'][0]['publisher_description'] != 'Caller mutation'
    assert cfg.resolve_dataset(dataset, 2024, reader_profile='source').metadata.provenance_ids == (raw_id,)


@pytest.mark.parametrize('scope,family,width,rows,raw_id', EXTENDED_SOURCES)
def test_extended_graduation_context_and_dictionary_only_codes_preserve_publisher_evidence(scope, family, width, rows, raw_id):
    cfg = load_config()
    dataset = scope + '_tapr_' + family
    five_year = family.startswith('5_')
    years, source_year = (5, 2022) if five_year else (6, 2021)
    expected = {
        ('campus', True): ('c8a79c457329940aa17bf1c48ce5aed78dff093ed71d659a47cf3861dbdd5729',
                           '8fb60bce50fa8a4f2023755c9d9dfef8183863174f546f16d755cc712d4041fc'),
        ('campus', False): ('a086b92aa590681d7c5eb5f051fbc470d1cde76be8e2e5464721874910519dd8',
                            '25f8d5955c4dc1eab0b1b700a4c2dcfff28e51ff73439939e1017bfa2a6e5669'),
        ('district', True): ('62e87eac5039a207f509343253559bc95c7b3a3d5ef6f6484b16f7caf02fd2ed',
                             '5aa73a1a78e55783bfb4eaeaa58ed2b3507545341036a8ef0896d1761391de1d'),
        ('district', False): ('21798d95f8fcde43e105530d709289bd35b4e27eb5342ab176483007ae2ff87c',
                              '25235a6513ba66ab3a67fd5c14b2cdfef0b7e62dbe4d4b646437c14ec73fa97c'),
    }
    contract = cfg.column_contract(dataset, 2024, reader_profile='source')
    assert contract is not None
    branch, = [branch for branch in parse_column_schema(contract['text'].encode())['datasets'] if branch['dataset'] == dataset]
    table, = branch['tables']
    fields, occurrences = table['fields'], table['dictionary_records']
    field_sha, occurrence_sha = expected[scope, five_year]
    assert sha256(canonical_json([[field[key] for key in ('position', 'code', 'label', 'field', 'disposition')]
                                  for field in fields]).encode()).hexdigest() == field_sha
    assert sha256(canonical_json(occurrences).encode()).hexdigest() == occurrence_sha
    assert len(fields) == width and reviewed_tapr_data_record_count(table) == rows
    assert len(occurrences) == 224 and [record['ordinal'] for record in occurrences] == list(range(1, 225))
    ordinary = f'{years}-Year Extended Longitudinal Rate (Gr 9-12)'
    federal = f'{years}-Year Extended Federal Graduation Rate Without Exclusions (Gr 9-12)'
    assert sum(record['publisher_context']['heading'] == ordinary for record in occurrences) == 182
    assert sum(record['publisher_context']['heading'] == federal for record in occurrences) == 42
    assert all(record.get('publisher_length') is None and record['publisher_type'] == 'Number'
               and str(source_year) in record['publisher_description'] for record in occurrences)
    facts = [field for field in fields if field['disposition'] == 'fact']
    assert len(facts) == 196 and all(field['dictionary_status'] == 'documented' for field in facts)
    assert all('publisher_length' not in field for field in fields)
    actual_codes = {field['code'] for field in fields}
    absent = [record for record in occurrences if record['code'] not in actual_codes]
    assert len(absent) == 28 and all(record['publisher_description'] is not None for record in absent)
    mapping = cfg.header_mapping(dataset, edition='2024', table=family, reader_profile='source')
    assert not {record['code'] for record in absent} & set(mapping)
    by_code = {record['code']: record for record in occurrences}
    renamed = [field for field in facts if field['field'] != sanitize_header(field['label'])]
    assert len(renamed) == (42 if five_year else 14)
    for field in renamed:
        assert by_code[field['code']]['publisher_context'] == {'heading': federal, 'source_line': 2492, 'source_column': 0}
        assert field['field'] == sanitize_header(field['label'].replace(f'{years}-Year Longitudinal:', federal))
    projected = cfg.header_records(dataset, edition='2024', reader_profile='source')
    assert all(record['dictionary_occurrences'] == [row for row in occurrences if row['code'] == record['code']]
               and record['period'] == UNKNOWN and record.get('publisher_length') is None for record in projected)
    metadata = cfg.resolve_dataset(dataset, 2024, reader_profile='source').metadata
    assert metadata.provenance_ids == (raw_id,) and metadata.value_policy is None and not metadata.metrics
    assert 'selection-form federal group labels state 2023' in metadata.notes
    for role in ['glossary', 'masking_methodology']:
        reference, = [ref for ref in metadata.references if ref.role == role]
        original, = [ref for ref in cfg.resolve_dataset(scope + '_tapr_reference', 2024, reader_profile='source').metadata.references if ref.role == role]
        assert reference.to_dict() == original.to_dict()


@pytest.mark.parametrize('scope,family,width,rows,raw_id', SOURCES)
def test_tiny_ordinary_source_preparation_preserves_raw_tokens_and_physical_table_independence(tmp_path, scope, family, width, rows, raw_id):
    dataset = scope + '_tapr_' + family
    contract = load_config().column_contract(dataset, 2024, reader_profile='source')
    assert contract is not None
    original = next(item for item in parse_column_schema(contract['text'].encode())['datasets'] if item['dataset'] == dataset)
    mapping = deepcopy(original['tables'][0])
    mapping['table'] = 'publisher_rows'  # A physical table is not a logical-family alias.
    row = ['-3'] * width
    positions = {field['code']: field['position'] - 1 for field in mapping['fields']}
    for code, value in {'CAMPUS': '001902001', 'CAMPNAME': 'Historical campus', 'DISTRICT': '001902', 'DISTNAME': 'Historical district'}.items():
        if code in positions:
            row[positions[code]] = value
    facts = [field for field in mapping['fields'] if field['disposition'] == 'fact']
    for field, value in zip(facts, ['', '0', '-1', '-3', '0012', 'N/A'], strict=False):
        row[field['position'] - 1] = value
    if family == 'reference':
        row[positions['REGION']] = '000777'  # Literal Length='2' is not a value-width constraint.
    path = tmp_path / 'input.csv'
    with path.open('w', newline='') as stream:
        csv.writer(stream).writerows([[field['label'] for field in mapping['fields']],
            [field['code'] for field in mapping['fields']], row])
    mapping['input_sha256'] = sha256(path.read_bytes()).hexdigest()
    mapping['layout']['last_data_record'] = 3
    assert qualified_tapr_family(dataset, scope + '_number') == family
    loaded = read_tapr_table(path, mapping=mapping, key_field=scope + '_number', dataset=dataset, edition='2024')
    assert loaded.table.to_pylist() == [dict(zip([field['field'] for field in mapping['fields']], row, strict=True))]
    schema = tmp_path / 'schema.json'
    schema.write_text(canonical_json({'schema_version': 2, 'mapping_version': 1, 'datasets': [
        {'dataset': dataset, 'edition': '2024', 'source_only': True, 'tables': [mapping]}]}) + '\n')
    store = ProvenanceStore(tmp_path / 'original')
    local_id = store.record_acquisition(path, ResolvedDataset(dataset, 'data_sources', 2024, 2024,
        str(path), DatasetMetadata(version='2024')))
    cfg = Config.from_dict({'data_sources': {dataset: {2024: {'path': str(path),
        'metadata': {'version': '2024', 'period': UNKNOWN, 'provenance_ids': [local_id],
            'references': [{'reference_id': 'columns', 'role': 'schema', 'path': schema.name,
                'status': 'available', 'reviewed_sha256': sha256(schema.read_bytes()).hexdigest()}]},
        'reader': {'profiles': {'source': {'column_schema': schema.name,
            'tapr_source': {'scope': scope, 'tables': {'publisher_rows': str(path)}}}}}}}}})
    cfg.base_dir = tmp_path
    output = tmp_path / 'prepared'
    result = prepare_profiles(cfg, edition='2024', family=family, scopes=[scope], input_dir=tmp_path,
        output_dir=output, reference_store=store)
    assert result['scopes'][scope]['rows'] == 1 and result['scopes'][scope]['fields'] == width
    assert result['scopes'][scope]['qualified_metrics'] == {}
    assert pq.read_table(output / (scope + '.parquet')).equals(loaded.table)
    closure = ProvenanceStore(output / 'provenance').verify_record_closure([result['scopes'][scope]['processing_record_id']])
    assert closure[local_id] == store.read_record(local_id)


@pytest.mark.parametrize('dataset', ['campus_tapr_ref', 'campus_tapr_DROP_ATT', 'district_tapr_reference_alias', 'campus_tapr_comp4',
                                     'campus_tapr_staar_sp', 'district_tapr_staar_school_progress_alias'])
def test_foreign_family_names_fail_before_source_io(tmp_path, dataset, monkeypatch):
    monkeypatch.setattr(Path, 'read_bytes', lambda *a, **k: pytest.fail('Foreign family opened source bytes'))
    with pytest.raises(ValueError, match='explicit qualified dataset'):
        read_tapr_table(tmp_path / 'absent.csv', mapping={}, key_field='campus_number', dataset=dataset, edition='2024')


@pytest.mark.parametrize('scope', ['campus', 'district'])
@pytest.mark.parametrize('family', ['annual_rhsp_dap_fhsp_e_fhsp_dla_texas_first_dla',
                                   'longitudinal_rhsp_dap_fhsp_e_fhsp_dla_texas_first_dla',
                                   '5_year_extended_longitudinal_rate', '6_year_extended_longitudinal_rate'])
def test_program_source_request_admission_preserves_scope_and_key_fences(tmp_path, monkeypatch, scope, family):
    dataset, key = scope + '_tapr_' + family, scope + '_number'
    code = 'CAMPUS' if scope == 'campus' else 'DISTRICT'
    mapping = {'table': 'publisher_program_rows', 'input_sha256': 'a' * 64,
        'source_schema_sha256': source_schema_fingerprint(['Publisher identifier'], [code]),
        'fields': [{'position': 1, 'code': code, 'label': 'Publisher identifier', 'field': key,
                    'disposition': 'identity'}]}
    path = tmp_path / 'not-opened.csv'
    monkeypatch.setattr(Path, 'open', lambda *a, **k: pytest.fail('Admission opened source bytes'))
    monkeypatch.setattr(Path, 'stat', lambda *a, **k: pytest.fail('Admission inspected source file'))
    with TaprSourceRequest(path, mapping, key, dataset, '2024').open() as reader:
        assert reader.request.dataset == dataset and reader.request.key_field == key
        with pytest.raises(ValueError, match='not completed validated EOF'):
            _ = reader.summary
    other = 'district' if scope == 'campus' else 'campus'
    for refused in [other + '_tapr_' + family, dataset + '_alias']:
        with pytest.raises(ValueError, match='explicit qualified dataset'):
            TaprSourceRequest(path, mapping, key, refused, '2024').open()
    wrong = deepcopy(mapping)
    wrong['fields'][0]['code'] = 'WRONG_SOURCE_KEY'
    wrong['source_schema_sha256'] = source_schema_fingerprint(['Publisher identifier'], ['WRONG_SOURCE_KEY'])
    with pytest.raises(ValueError, match='source key must bind publisher code'):
        TaprSourceRequest(path, wrong, key, dataset, '2024').open()
