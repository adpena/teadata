"""Current source families retain labels, history and unbound strings."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import csv
from pathlib import Path
import socket

import pyarrow.parquet as pq
import pytest
import yaml

from teadata.column_schema import parse_column_schema, validate_header_table
from teadata.dataset_metadata import DatasetMetadata, ResolvedDataset
from teadata.manifest import ContractError, DatasetCatalog, _validate_methodology, canonical_json
from teadata.provenance import ProvenanceStore, validate_record_graph, validate_reference_records
from teadata.scripts.build_tapr_profiles import prepare_profiles
from teadata.scripts.rename_tapr_campus_headers import sanitize_header
from teadata.tapr import qualified_tapr_family, read_tapr_table
from teadata.teadata_config import Config, load_config


ROOT = Path(__file__).resolve().parents[1]
RESOURCE = ROOT / 'teadata/schemas/tapr--2024--reference-dropout-attendance-crosswalk.json'
UNKNOWN = {'kind': 'unknown'}
SOURCES = [
    ('campus', 'reference', 22, 9082, '46bd6f225bfb3bc1aaf2a3fd733bb212d2e6ebd19e9e8c81486be15323e37f1b'),
    ('district', 'reference', 12, 1207, 'f0841681cbf8e28a91dfe31c9653b04cf4a6d933ad6bf30d334a603362b9d0b7'),
    ('campus', 'dropout_attendance', 166, 9082, '35a1fd68ec12257cf38a835a3cd63a8676affd81d1d4381c2c513b13ae2b0a0c'),
    ('district', 'dropout_attendance', 164, 1207, 'e05111f5db6e9b6e6477a4ac38b7a2e38c694f94312e5fa20c7082125fe863af'),
    ('campus', 'four_year_longitudinal_rate', 228, 9082, '1ecb3c44285694d03d6b054674ea235e1462a699a2fa8db64fc25e5286caab90'),
    ('district', 'four_year_longitudinal_rate', 226, 1207, '632d7f6c24087c3f557882724229ebc68318e510ac1f206181e39b5833aef16a'),
    ('campus', 'staar_school_progress', 2212, 9082, '751c3626c097b51f2d4467950aa7f77da976083fc63169d1c19268198bd5b96c'),
    ('district', 'staar_school_progress', 2210, 1207, '72df19ef8f42a1b404e80ef528a1f9cc9cc93cc27aa466891140ed2fc374fdfb'),
]
NEW_DATASETS = {scope + '_tapr_' + family for scope, family, *_ in SOURCES}


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
    assert not metadata.metrics and metadata.provenance_ids == (raw_id,)
    _validate_methodology(metadata, None)
    assert cfg.is_source_only(dataset, 2024) and cfg.column_schema(dataset, 2024) is None
    assert cfg.availability_report()['data_sources'][dataset]['years'] == [2024]
    assert cfg.reader_references(dataset, 2024, reader_profile='source') == list(metadata.references)
    roots = [raw_id, *(identifier for ref in metadata.references for identifier in ref.provenance_ids)]
    graph = validate_record_graph(store.read_record, roots)
    validate_reference_records(source, graph.__getitem__, verify_all_claimed=True)
    acquired = graph[raw_id]
    assert acquired['dataset'] == dataset and acquired['method'] == 'download'
    date_prefix = {'four_year_longitudinal_rate': '2026-11-01T09:06:',
                   'staar_school_progress': '2026-11-01T10:58:'}.get(family, '2026-11-01T06:11:')
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
    assert all(record['period'] == UNKNOWN and record['table'] == family for record in records)
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


def test_current_source_contracts_keep_explicit_one_year_csv_opt_ins():
    current = yaml.safe_load((ROOT / 'teadata/teadata_sources.yaml').read_text())
    for dataset in NEW_DATASETS:
        assert list(current['data_sources'].pop(dataset)) == [2024]
        assert current['schema']['data_sources'].pop(dataset) == ['csv']
    # Full pre-change Config preservation has one current baseline owner below.


def test_staar_school_progress_opt_ins_preserve_the_entire_preexisting_Config():
    current = yaml.safe_load((ROOT / 'teadata/teadata_sources.yaml').read_text())
    for scope in ['campus', 'district']:
        dataset = scope + '_tapr_staar_school_progress'
        current['data_sources'].pop(dataset)
        current['schema']['data_sources'].pop(dataset)
    # Exact pre-change d08ff9f Config, including existing COMP4/REF/DROP opt-ins
    # and every legacy default/profile/metric binding. One baseline owns this.
    assert sha256(canonical_json(current).encode()).hexdigest() == '440057779ba84f04070e3175623e789eafa75ab841ae888271361218b016e1a7'


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
def test_comp4_context_names_and_annual_references_do_not_assign_methodology(scope, prefix):
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
    assert source.metadata.metrics == {} and source.metadata.period.to_dict() == UNKNOWN
    assert source.metadata.value_policy is None and source.metadata.estimation is None and source.metadata.field_year is None
    assert 'field-specific COMP4 masking applicability remains unreviewed' in source.metadata.notes
    _validate_methodology(source.metadata, None)


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
