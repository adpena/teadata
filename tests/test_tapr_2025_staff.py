from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import csv
import json
from pathlib import Path

import pytest

from teadata.column_schema import header_records, header_rename_map, parse_column_schema, source_schema_fingerprint
from teadata.dataset_metadata import DatasetMetadata, EditionKey, ResolvedDataset
from teadata.manifest import ContractError, canonical_json
from teadata.provenance import ProvenanceStore
from teadata.scripts.build_tapr_profiles import main as prepare_main, prepare_profiles
from teadata.teadata_config import Config, load_config


ROOT = Path(__file__).resolve().parents[1]
RESOURCE = ROOT / 'teadata/schemas/tapr--2025--staff-profile-crosswalk.json'


@pytest.mark.parametrize('scope,columns,rows,raw_sha,dictionary_sha,new_count,raw_id', [
    ('campus', 106, 9084, 'ed29852dd7d7925963413b41c81a36508ccad83a4109a0aeebfa720bf90ff6ea',
     'd5a1f29d97d34946eae7705d5f8a1ef7f376f7dd143cf0ce7b2fe0f1ba1f719e', 15,
     '7ffcfafba5a510e01d1d613bc01598c6567ed11f1eee342136ca3aa129b49752'),
    ('district', 118, 1208, '29921c748875f91cc123c22dc5a295f1e61d1754842209bd86e56ad9f7c7abb1',
     'fead1a0ad8f3a35949dfefc8161781be4161ff7fd028e926f99c3b9c41d5a8d6', 17,
     '4a72e91345b11dca068b1ae0744bef9b8ca4c269463e83dbaffe148d0e13ceff'),
])
def test_staff_crosswalk_retains_every_actual_column_without_default_activation(scope, columns, rows, raw_sha, dictionary_sha, new_count, raw_id):
    document = parse_column_schema(RESOURCE.read_bytes())
    assert not (ROOT / 'docs/source-contracts' / RESOURCE.name).exists()
    assert sha256(RESOURCE.read_bytes()).hexdigest() == '579619e1d337a6c34078ad45f8138ac9277c4d3e6751995a56ad084c9599300e'
    dataset = next(d for d in document['datasets'] if d['dataset'] == scope + '_tapr_staff_profile')
    assert dataset['edition'] == '2025' and dataset['source_only'] is True
    table, = dataset['tables']
    assert table['input_sha256'] == raw_sha and table['table'] == 'staff_profile'
    assert table['layout'] == {'label_record': 1, 'code_record': 2, 'first_data_record': 3,
                               'last_data_record': rows + 2, 'annotations': []}
    records = header_records(document, dataset=dataset['dataset'], edition='2025')
    assert len(records) == columns == len(header_rename_map(records, from_='code', to='label'))
    assert all(record['period'] == {'kind': 'unknown'} for record in records)
    assert table['metric_period'] == {'kind': 'unknown'}
    expected_identity = {'district_number', 'district_name'}
    if scope == 'campus':
        expected_identity |= {'campus_number', 'campus_name'}
    assert {r['field'] for r in records if r['disposition'] == 'identity'} == expected_identity
    assert len([r for r in records if r['disposition'] == 'identity']) == len(expected_identity)
    assert {r['source_sha256'] for r in table['dictionary_records']} == {dictionary_sha}
    assert all(type(r['source_line']) is int and r['source_line'] > 0
               and type(r['source_column']) is int and r['source_column'] >= 0 for r in table['dictionary_records'])
    historical_path = ROOT / 'teadata/schemas/tapr--2025--header-crosswalk.json'
    assert sha256(historical_path.read_bytes()).hexdigest() == 'fdbdbb24ab218063d46cb5d4781ef9cca55b9e448020f01b8b7264e8c292e339'
    old_dataset = next(d for d in parse_column_schema(historical_path.read_bytes())['datasets'] if d['dataset'].startswith(scope + '_'))
    old_codes = {f['code'] for t in old_dataset['tables'] if t['table'] == 'staff' for f in t['fields']}
    assert old_codes < {r['code'] for r in records}
    assert len({r['code'] for r in records} - old_codes) == new_count
    cfg = load_config()
    assert cfg.availability_report()['data_sources'][dataset['dataset']]['years'] == [2024, 2025]
    assert cfg.header_records(dataset['dataset'], edition='2025', reader_profile='source') == records
    assert cfg.is_source_only(dataset['dataset'], 2025)
    assert cfg.column_schema(dataset['dataset'], 2025) is None
    source = cfg.resolve_dataset(dataset['dataset'], 2025)
    assert source.metadata.period.kind == 'unknown' and source.metadata.value_policy is None
    assert not source.metadata.metrics and not source.metadata.provenance_ids and source.metadata.field_year is None
    raw_reference = next(r for r in source.metadata.references if r.reference_id == 'csv_headers')
    assert raw_reference.provenance_ids == (raw_id,) and raw_reference.reviewed_sha256 == raw_sha
    assert source.reader['profiles']['source']['tapr_source']['tables']['staff_profile'] == f'data/tapr/raw/2025-{scope}-STAF-all-checked.csv'
    with pytest.raises(ContractError, match='Reader profile is required'):
        cfg.load_df(dataset['dataset'], 2025)


def test_code_only_and_dictionary_only_codes_do_not_become_input_fields_or_aliases():
    document = parse_column_schema(RESOURCE.read_bytes())
    campus = next(d for d in document['datasets'] if d['dataset'] == 'campus_tapr_staff_profile')
    table, = campus['tables']
    codes = {field['code'] for field in table['fields']}
    assert {'CPSXTOFC', 'CPSXTOFP', 'CPSCTOST'}.isdisjoint(codes)
    partial = [r for r in table['dictionary_records'] if r['publisher_type'] is None or r['publisher_description'] is None]
    assert [r['code'] for r in partial] == ['CPSXTOFC', 'CPSXTOFP']
    field = next(f for f in table['fields'] if f['code'] == 'CPSCTOFC')
    assert field['dictionary_status'] == 'not_documented'
    assert field['publisher_type'] is None and field['publisher_description'] is None
    assert field['label'] == 'Campus 2025 Staff: Central Admin Total Full Time Equiv Count'
    district = next(d for d in document['datasets'] if d['dataset'] == 'district_tapr_staff_profile')
    records = district['tables'][0]['dictionary_records']
    for code in ('DPSETOFC', 'DPSXTOFC'):
        occurrences = [r for r in records if r['code'] == code]
        assert len(occurrences) == 2 and occurrences[0]['ordinal'] != occurrences[1]['ordinal']
        assert len({(r['publisher_type'], r['publisher_description']) for r in occurrences}) == 1


@pytest.mark.parametrize('scope,width,rows,raw_sha,dictionary_sha,raw_id', [
    ('campus', 243, 9084, '7e0e762a8f2ba50687fd8c6a59e8052d5740ba57abbedecddfca064bbd77548e',
     '01b60e8eec23e5c158206d796c0ce82c713417e5f822f1ab81354199814ef209',
     '3dec9c6e933d9f88c6efacc6f00f8669f8dda0e9d3c51ee9c176f910011e7b35'),
    ('district', 242, 1208, '8ccd14fb2eaff8c73d94960950cc359d44bf4b81edb721975edf6795b8995811',
     '693d84e0fa2285918c0f0d437a8fbfdefa72b66c1a7acd8ddac94165b536568d',
     'cdef4db2551805ed9aa901243df73d91843d158d14426592607e842ce7036231'),
])
def test_student_source_contract_preserves_observed_headers_and_historical_names_without_period_inference(
        scope, width, rows, raw_sha, dictionary_sha, raw_id):
    resource = ROOT / 'teadata/schemas/tapr--2025--student-profile-crosswalk.json'
    document = parse_column_schema(resource.read_bytes())
    selected = next(d for d in document['datasets'] if d['dataset'] == scope + '_tapr_student_profile')
    table, = selected['tables']
    assert selected['source_only'] and 'prepared_input_sha256' not in selected
    assert table['input_sha256'] == raw_sha and table['layout'] == {
        'label_record': 1, 'code_record': 2, 'first_data_record': 3, 'last_data_record': rows + 2, 'annotations': []}
    historical = parse_column_schema((ROOT / 'teadata/schemas/tapr--2025--header-crosswalk.json').read_bytes())
    previous = next(d for d in historical['datasets'] if d['dataset'] == scope + '_tapr_student_staff_profile')
    original = next(t for t in previous['tables'] if t['table'] == 'student')
    assert [{key: f[key] for key in ['position', 'code', 'label', 'field']} for f in table['fields']] == original['fields']
    assert table['source_schema_sha256'] == original['source_schema_sha256']
    assert {r['source_sha256'] for r in table['dictionary_records']} == {dictionary_sha}
    cfg = load_config()
    source = cfg.resolve_dataset(selected['dataset'], 2025)
    records = cfg.header_records(selected['dataset'], edition='2025', reader_profile='source')
    assert len(records) == width and all(r['period'] == {'kind': 'unknown'} for r in records)
    assert not source.metadata.metrics and source.metadata.value_policy is None and source.metadata.field_year is None
    assert not source.metadata.provenance_ids
    raw_reference = next(r for r in source.metadata.references if r.reference_id == 'csv_headers')
    assert raw_reference.provenance_ids == (raw_id,) and raw_reference.reviewed_sha256 == raw_sha
    assert source.metadata.notes and 'independent processing receipt' in source.metadata.notes
    assert source.path.endswith(f'/data/tapr/raw/2025-{scope}-STUD-all-checked.csv')
    for role in ['data_dictionary', 'glossary', 'masking_methodology']:
        ref, = [r for r in source.metadata.references if r.role == role]
        assert ref.status == 'available' and ref.provenance_ids and ref.reviewed_sha256
    if scope == 'district':
        duplicate = [r for r in table['dictionary_records'] if r['code'] == 'DAUND24C']
        assert len(duplicate) == 2 and duplicate[0]['ordinal'] != duplicate[1]['ordinal']


@pytest.mark.parametrize('scope,family,width', [
    ('campus', 'student_profile', 243), ('district', 'student_profile', 242),
    ('campus', 'staff_profile', 106), ('district', 'staff_profile', 118),
])
def test_2025_natural_source_consumer_keeps_all_synthetic_columns_and_original_typed_history(
        tmp_path, scope, family, width):
    import pyarrow.parquet as pq

    cfg = load_config()
    dataset = scope + '_tapr_' + family
    contract = cfg.column_contract(dataset, 2025, reader_profile='source')
    assert contract is not None
    document = parse_column_schema(contract['text'].encode())
    selected = deepcopy(next(d for d in document['datasets'] if d['dataset'] == dataset))
    document['datasets'] = [selected]
    table, = selected['tables']
    path = tmp_path / 'original.csv'
    identities = {'CAMPUS': '001902001', 'CAMPNAME': 'Historical campus',
                  'DISTRICT': '001902', 'DISTNAME': 'Historical district'}
    tokens = iter(['.', '', '0', '-1', 'n/a'])
    values = [identities[field['code']] if field['code'] in identities else next(tokens, 'unchanged source text')
              for field in table['fields']]
    with path.open('w', newline='') as stream:
        csv.writer(stream).writerows([[f['label'] for f in table['fields']], [f['code'] for f in table['fields']], values])
    table['input_sha256'] = sha256(path.read_bytes()).hexdigest()
    table['layout']['last_data_record'] = 3
    schema = tmp_path / 'schema.json'
    schema.write_text(canonical_json(document) + '\n')
    original_store = ProvenanceStore(tmp_path / 'original-store')
    original_label = scope + '_tapr_student_studf_profile' if family == 'student_profile' else scope + '_tapr_student_staff_profile'
    raw_id = original_store.record_acquisition(path, ResolvedDataset(original_label, 'data_sources', 2025, 2025, '',
        DatasetMetadata(version='2025')), requested_url='https://example.test/original.csv',
        downloaded_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    original_record = original_store.read_record(raw_id)
    annotation = tmp_path / 'annotation.json'
    annotation.write_text('{"operation":"explicit label annotation; no alias"}')
    correction = original_store.record_processing(dataset='explicit_annotation', inputs=[raw_id],
        outputs={'annotation': annotation}, processor='synthetic fixture', code_version='fixture')
    configured = Config.from_dict({'data_sources': {dataset: {2025: {'path': str(path),
        'metadata': {'version': '2025', 'period': {'kind': 'unknown'}, 'references': [{
            'reference_id': 'csv_headers', 'role': 'source_notes', 'url': 'https://example.test/original.csv',
            'provenance_ids': [raw_id], 'reviewed_sha256': table['input_sha256']}]},
        'reader': {'profiles': {'source': {'column_schema': schema.name,
            'tapr_source': {'scope': scope, 'tables': {family: str(path)}}, 'references': [{
                'reference_id': 'columns_source', 'role': 'schema', 'path': schema.name,
                'reviewed_sha256': sha256(schema.read_bytes()).hexdigest()}]}}}}}}})
    configured.base_dir = tmp_path
    result = prepare_profiles(configured, edition='2025', family=family, scopes=[scope], input_dir=None,
        output_dir=tmp_path / 'prepared', reference_store=original_store, processing_inputs={dataset: [correction]})
    assert result['scopes'][scope]['fields'] == width and result['scopes'][scope]['rows'] == 1
    prepared = pq.read_table(tmp_path / 'prepared' / (scope + '.parquet'))
    assert [column[0].as_py() for column in prepared.columns] == values
    archive = ProvenanceStore(tmp_path / 'prepared/provenance')
    closure = archive.verify_record_closure([result['scopes'][scope]['processing_record_id']])
    assert raw_id in closure and closure[raw_id] == original_record
    assert original_store.read_record(raw_id) == original_record
    assert closure[correction] == original_store.read_record(correction)
    assert closure[correction]['kind'] == 'processing' and closure[correction]['inputs'] == [raw_id]
    assert correction in closure[result['scopes'][scope]['processing_record_id']]['inputs']
    assert result['scopes'][scope]['selected_processing_inputs'] == [correction]
    assert configured.resolve_dataset(dataset, 2025).metadata.provenance_ids == ()


@pytest.mark.parametrize('problem', ['foreign_dataset', 'acquisition_root', 'missing_closure', 'missing_store'])
def test_processing_dependency_failures_precede_source_reads_and_publication(tmp_path, monkeypatch, problem):
    store = ProvenanceStore(tmp_path / 'history')
    source = tmp_path / 'original.csv'
    source.write_bytes(b'original')
    raw = store.record_acquisition(source, ResolvedDataset('original', 'data_sources', None, None, ''))
    annotation = tmp_path / 'annotation.json'
    annotation.write_bytes(b'{"annotation":true}')
    processing = store.record_processing(dataset='annotation', inputs=[raw], outputs={'annotation': annotation},
        processor='fixture', code_version='fixture')
    dataset = 'campus_tapr_student_profile'
    selected = {dataset: [processing]}
    if problem == 'foreign_dataset':
        selected = {'district_tapr_student_profile': [processing]}
    elif problem == 'acquisition_root':
        selected = {dataset: [raw]}
    elif problem == 'missing_closure':
        (store.root / 'objects' / sha256(b'original').hexdigest()).unlink()
    cfg = Config()
    monkeypatch.setattr(cfg, 'read_tapr_sources', lambda *a, **k: pytest.fail('Invalid dependency reached raw source read'))
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare_profiles(cfg, edition='2025', scopes=['campus'], input_dir=None, family='student_profile',
            output_dir=tmp_path / 'output', reference_store=None if problem == 'missing_store' else store,
            processing_inputs=selected)
    assert not (tmp_path / 'output').exists()
    assert not list(tmp_path.glob('.tapr-preparation-*'))


def tiny_annual_source(tmp_path, year, *, scope='campus', family='student_profile', stem='', values=('.',), annotated=False,
                       physical_table=None):
    """A two-column local observation; no real future-year data is asserted."""
    path = tmp_path / (stem + 'source.csv')
    labels, codes = [scope.title(), 'Opaque source observation'], ['CAMPUS' if scope == 'campus' else 'DISTRICT', 'VALUE']
    keys = [f'001902{i:03}' if scope == 'campus' else f'001{901+i:03}' for i in range(1, len(values) + 1)]
    records = [labels, codes, *zip(keys, values)]
    if annotated:
        records.insert(3, [])
    with path.open('w', newline='') as stream:
        csv.writer(stream).writerows(records)
    fields = [{'position': 1, 'code': codes[0], 'label': labels[0], 'field': scope + '_number',
               'disposition': 'identity', 'dictionary_status': 'not_applicable', 'publisher_type': None, 'publisher_description': None},
              {'position': 2, 'code': codes[1], 'label': labels[1], 'field': 'source_observation',
               'disposition': 'fact', 'dictionary_status': 'not_documented', 'publisher_type': None, 'publisher_description': None}]
    table_name = physical_table or family
    table = {'table': table_name, 'metric_period': {'kind': 'unknown'},
        'input_sha256': sha256(path.read_bytes()).hexdigest(),
        'source_schema_sha256': source_schema_fingerprint(labels, codes),
        'fields': fields, 'layout': {'label_record': 1, 'code_record': 2, 'first_data_record': 3,
                                    'last_data_record': len(records), 'annotations': [{
                                        'first_record': 4, 'last_record': 4, 'kind': 'blank',
                                        'sha256': sha256(b'\r\n').hexdigest()}] if annotated else []}}
    schema = tmp_path / (stem + 'schema.json')
    dataset = scope + '_tapr_' + family
    schema.write_text(canonical_json({'schema_version': 2, 'mapping_version': 1, 'datasets': [{
        'dataset': dataset, 'edition': str(year), 'source_only': True, 'tables': [table]}]}) + '\n')
    store = ProvenanceStore(tmp_path / 'history')
    raw = store.record_acquisition(path, ResolvedDataset('original local observation', 'data_sources', year, year, '',
        DatasetMetadata(version=str(year))))
    config = tmp_path / (stem + 'config.json')
    config.write_text(canonical_json({'data_sources': {dataset: {year: {
        'path': path.name, 'metadata': {'version': str(year), 'period': {'kind': 'unknown'}, 'references': [{
            'reference_id': 'csv_headers', 'role': 'source_notes', 'path': path.name,
            'provenance_ids': [raw], 'reviewed_sha256': table['input_sha256']}]},
        'reader': {'profiles': {'source': {'column_schema': schema.name,
            'tapr_source': {'scope': scope, 'tables': {table_name: path.name}}, 'references': [{
                'reference_id': 'columns_source', 'role': 'schema', 'path': schema.name,
                'reviewed_sha256': sha256(schema.read_bytes()).hexdigest()}]}}}}}}}))
    return config, store


def test_registered_2026_annual_cli_uses_existing_exact_config_contract(tmp_path, capsys):
    config, store = tiny_annual_source(tmp_path, 2026)
    annotation = tmp_path / 'review.json'
    annotation.write_text('{"review":"synthetic source context"}')
    processing = store.record_processing(dataset='synthetic_review',
        inputs=[store.history('original local observation')[0]['record_id']], outputs={'review': annotation},
        processor='fixture', code_version='fixture')
    prepare_main(['--edition', '2026', '--family', 'student_profile', '--scope', 'campus',
        '--config', str(config), '--reference-store', str(store.root), '--input-dir', str(tmp_path),
        '--processing-input', 'campus_tapr_student_profile', processing,
        '--processing-input', 'campus_tapr_student_profile', processing,
        '--output-dir', str(tmp_path / 'prepared')])
    assert '"edition": "2026"' in capsys.readouterr().out
    import pyarrow.parquet as pq
    table = pq.read_table(tmp_path / 'prepared/campus.parquet')
    assert table.to_pydict() == {'campus_number': ['001902001'], 'source_observation': ['.']}
    import json
    receipt = json.loads((tmp_path / 'prepared/preparation.json').read_text())
    assert receipt['scopes']['campus']['selected_processing_inputs'] == [processing]
    assert ProvenanceStore(tmp_path / 'prepared/provenance').read_record(processing) == store.read_record(processing)


@pytest.mark.parametrize('configured_year', [2025, 2027])
def test_absent_or_prior_fallback_2026_fails_before_source_open(tmp_path, monkeypatch, configured_year):
    config, store = tiny_annual_source(tmp_path, configured_year)
    monkeypatch.setattr('teadata.tapr.read_tapr_table', lambda *a, **k: pytest.fail('Unregistered annual source opened'))
    with pytest.raises((ContractError, KeyError)):
        prepare_profiles(load_config(config), edition='2026', family='student_profile', scopes=['campus'],
            input_dir=tmp_path, output_dir=tmp_path / 'prepared', reference_store=store)
    assert not (tmp_path / 'prepared').exists()


@pytest.mark.parametrize('edition', ['02026', '2026 ', '2026.0', 'not-an-edition'])
def test_noncanonical_annual_editions_fail_before_source_read(tmp_path, monkeypatch, edition):
    cfg = Config()
    monkeypatch.setattr(cfg, 'read_tapr_sources', lambda *a, **k: pytest.fail('Noncanonical edition reached source read'))
    with pytest.raises(ValueError, match='canonical annual integer'):
        prepare_profiles(cfg, edition=edition, family='student_profile', scopes=['campus'],
            input_dir=None, output_dir=tmp_path / 'output')


@pytest.fixture
def prepared_cohort(tmp_path, request):
    """Four genuine tiny source preparations, including a typed review edge."""
    selected = getattr(request, 'param', 2025)
    if isinstance(selected, dict):
        year, annotated = selected['year'], False
        families = selected['families']
        physical_table = selected.get('physical_table')
    else:
        year, annotated = selected if isinstance(selected, tuple) else (selected, False)
        families, physical_table = ('student_profile', 'staff_profile'), None
    configs, inputs, annotations = {}, {}, {}
    pooled = ProvenanceStore(tmp_path / 'selected-references')
    for scope in ('campus', 'district'):
        for family in families:
            dataset = scope + '_tapr_' + family
            config, history = tiny_annual_source(tmp_path, year, scope=scope, family=family,
                stem=dataset + '-', values=('.', '***'), annotated=annotated and family == 'student_profile',
                physical_table=physical_table)
            configs.update(json.loads(config.read_text())['data_sources'])
            cfg = load_config(config)
            dependencies = {}
            if family == 'student_profile':
                review = tmp_path / (dataset + '-review.json')
                review.write_text('{"annotation":"synthetic original observation label correction"}')
                acquisition, = cfg.resolve_dataset(dataset, year).metadata.references[0].provenance_ids
                annotation = history.record_processing(dataset=dataset, inputs=[acquisition], outputs={'review': review},
                    processor='explicit-reviewed-annotation', code_version='fixture')
                dependencies[dataset] = [annotation]
                annotations[dataset] = annotation
            receipt = prepare_profiles(cfg, edition=str(year), family=family, scopes=[scope], input_dir=tmp_path,
                output_dir=tmp_path / (dataset + '-prepared'), reference_store=history, processing_inputs=dependencies)
            identifier = receipt['scopes'][scope]['processing_record_id']
            pooled.copy_record_from(ProvenanceStore(tmp_path / (dataset + '-prepared/provenance')), identifier)
            inputs[dataset] = (identifier, (scope + '_number',))
    cfg = Config.from_dict({'data_sources': configs})
    cfg.base_dir = tmp_path
    config = tmp_path / 'cohort-config.json'
    config.write_text(canonical_json({'data_sources': configs}))
    return cfg, pooled, inputs, annotations, config


@pytest.mark.parametrize('prepared_cohort', [2024, 2025, (2025, True)], indirect=True)
def test_raw_cohort_cli_keeps_all_fields_typed_closure_and_native_selection(prepared_cohort, tmp_path, capsys, monkeypatch):
    import pyarrow.dataset as ds
    from teadata.manifest import file_identity
    from teadata.release import DataRelease

    cfg, store, inputs, annotations, config = prepared_cohort
    output = tmp_path / 'new-cohort'
    args = ['--edition', str(cfg.resolve_dataset('campus_tapr_student_profile').resolved_year),
            '--config', str(config), '--reference-store', str(store.root), '--output-dir', str(output)]
    for dataset, (identifier, keys) in inputs.items():
        args += ['--prepared-input', dataset, identifier, keys[0]]
    # This mode consumes the exact selected prepared outputs, never rereads raw tables.
    monkeypatch.setattr('teadata.tapr.read_tapr_table', lambda *a, **k: pytest.fail('Raw source was reread'))
    prepare_main(args)
    assert json.loads(capsys.readouterr().out)['schema_version'] == 3
    release = DataRelease(output / 'release.json', offline=True)
    result = json.loads((output / 'preparation.json').read_text())
    assert result['status'] == 'unpublished_raw_wide_cohort' and result['release_activation'] is False
    assert len(release.verify()) == 4 and len(list(release.catalog.variables())) == 8
    assert set(result['datasets']) == set(inputs)
    roots = ProvenanceStore(output / 'provenance')
    for row in result['datasets'].values():
        key = EditionKey.from_dict(row['key'])
        selected, = release.catalog.resolve_artifacts([(key, 'source_observation')])
        assert selected['columns'] == (key.scope + '_number', 'source_observation')
        with monkeypatch.context() as discovery:
            discovery.setattr(Path, 'open', lambda *a, **k: pytest.fail('Discovery opened artifact'))
            assert len(list(release.catalog.variables(edition=key))) == 2
        scanner = release.analytical_dataset(selected['artifact'].identity).scanner(
            columns=list(selected['columns']), filter=ds.field('source_observation') == '***', batch_size=1)
        batches = list(scanner.to_batches())
        assert sum(batch.num_rows for batch in batches) == 1
        assert scanner.to_table().to_pydict()['source_observation'] == ['***']
        assert row['rows'] == 2 and row['fields'] == 2 and row['values_interpreted'] is False
        assert release.catalog.metadata(key.dataset, edition=key).period.kind == 'unknown'
        graph = roots.verify_record_closure([row['processing_record_id']])
        assert inputs[key.dataset][0] in graph
        if key.dataset in annotations:
            assert graph[annotations[key.dataset]]['kind'] == 'processing'
            assert annotations[key.dataset] in row['inherited_processing_inputs']
        with pytest.raises(ContractError, match='scalar interpretation'):
            release.catalog.measurement(object(), 'source_observation')
    assert file_identity(output / 'release.json') == result['manifest']
    assert not list(tmp_path.glob('.tapr-edition-*'))
    assert cfg.data_sources  # Config was not activated or replaced by preparation.


@pytest.mark.parametrize('damage', ['missing', 'foreign', 'key_name', 'reordered_keys', 'acquisition', 'edition', 'schema', 'annotation', 'missing_closure'])
def test_prepared_cohort_contract_failures_precede_native_open_and_output(prepared_cohort, tmp_path, monkeypatch, damage):
    import pyarrow.parquet as pq
    cfg, store, inputs, _, _ = prepared_cohort
    inputs = dict(inputs)
    dataset = 'campus_tapr_student_profile'
    identifier, keys = inputs[dataset]
    if damage == 'missing':
        inputs.clear()
    elif damage == 'foreign':
        inputs['not_selected'] = (identifier, keys)
    elif damage in {'key_name', 'reordered_keys'}:
        inputs[dataset] = (identifier, ('source_observation',) if damage == 'key_name' else (*keys, 'source_observation'))
    elif damage == 'acquisition':
        raw, = cfg.resolve_dataset(dataset, 2025).metadata.references[0].provenance_ids
        inputs[dataset] = (raw, keys)
    elif damage == 'missing_closure':
        record = store.read_record(identifier)
        (store.root / 'records' / (record['inputs'][0] + '.json')).unlink()
    else:
        record = store.read_record(identifier)
        parameters = deepcopy(record['parameters'])
        parameters[{'edition': 'edition', 'schema': 'column_schema_sha256', 'annotation': 'selected_processing_inputs'}[damage]] = {
            'edition': '2024', 'schema': '0' * 64, 'annotation': [record['inputs'][0]]}[damage]
        new = store.record_processing(dataset=dataset, inputs=record['inputs'],
            outputs={name: store.verify_artifact(value['sha256']) for name, value in record['outputs'].items()},
            processor=record['processor'], code_version=record['code_version'], parameters=parameters, validation=record['validation'])
        inputs[dataset] = (new, keys)
    monkeypatch.setattr(pq, 'ParquetFile', lambda *a, **k: pytest.fail('Unqualified native input opened'))
    with pytest.raises((ValueError, KeyError, FileNotFoundError, ContractError)):
        prepare_profiles(cfg, edition='2025', scopes=['campus', 'district'], input_dir=None,
            output_dir=tmp_path / 'blocked', reference_store=store, prepared_inputs=inputs)
    assert not (tmp_path / 'blocked').exists() and not list(tmp_path.glob('.tapr-edition-*'))


@pytest.mark.parametrize('damage', ['duplicate_key', 'wrong_type', 'writer_failure'])
def test_raw_cohort_native_failure_discards_only_its_stage(prepared_cohort, tmp_path, monkeypatch, damage):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from teadata.manifest import file_identity
    cfg, store, inputs, _, _ = prepared_cohort
    dataset = 'campus_tapr_student_profile'
    identifier, keys = inputs[dataset]
    record = store.read_record(identifier)
    original = store.verify_artifact(record['outputs']['prepared_source']['sha256'])
    before = file_identity(original)
    if damage == 'writer_failure':
        def fail(*a, **k):
            raise RuntimeError('injected native writer failure')
        monkeypatch.setattr('teadata.scripts.build_tapr_profiles.write_leaf_artifact', fail)
    else:
        table = pq.read_table(original)
        replacement = pa.array(['001902001', '001902001']) if damage == 'duplicate_key' else pa.array([1, 2])
        table = table.set_column(0, keys[0], replacement)
        changed = tmp_path / 'changed.parquet'
        pq.write_table(table, changed)
        new = store.record_processing(dataset=dataset, inputs=record['inputs'],
            outputs={name: changed if name == 'prepared_source' else store.verify_artifact(value['sha256'])
                     for name, value in record['outputs'].items()}, processor=record['processor'],
            code_version=record['code_version'], parameters=record['parameters'], validation=record['validation'])
        inputs = {**inputs, dataset: (new, keys)}
    with pytest.raises((ValueError, ContractError, RuntimeError)):
        prepare_profiles(cfg, edition='2025', scopes=['campus', 'district'], input_dir=None,
            output_dir=tmp_path / 'failed-native', reference_store=store, prepared_inputs=inputs)
    assert not (tmp_path / 'failed-native').exists() and not list(tmp_path.glob('.tapr-edition-*'))
    assert file_identity(original) == before


def test_prepared_cli_rejects_duplicate_selection_and_legacy_family_before_output(tmp_path):
    selected = ['--edition', '2025', '--prepared-input', 'campus_tapr_student_profile', '0' * 64, 'campus_number',
                '--output-dir', str(tmp_path / 'out')]
    with pytest.raises(SystemExit):
        prepare_main(selected + ['--prepared-input', 'campus_tapr_student_profile', '1' * 64, 'campus_number'])
    with pytest.raises(SystemExit):
        prepare_main(selected + ['--family', 'student_profile'])
    with pytest.raises(SystemExit):
        prepare_main(selected + ['--scope', 'campus'])
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('claimed_rows', [1, 3, 0, True, 2.0])
def test_prepared_row_claim_must_match_reviewed_layout_before_native_io(prepared_cohort, tmp_path, monkeypatch, claimed_rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    cfg, store, inputs, _, _ = prepared_cohort
    dataset = 'campus_tapr_student_profile'
    identifier, keys = inputs[dataset]
    record = store.read_record(identifier)
    original = pq.read_table(store.verify_artifact(record['outputs']['prepared_source']['sha256']))
    count = int(claimed_rows)
    changed = original.slice(0, count) if count <= 2 else pa.concat_tables([original, original.slice(0, 1)])
    path = tmp_path / 'self-consistent-but-wrong-count.parquet'
    pq.write_table(changed, path)
    replacement = store.record_processing(dataset=dataset, inputs=record['inputs'],
        outputs={name: path if name == 'prepared_source' else store.verify_artifact(value['sha256'])
                 for name, value in record['outputs'].items()}, processor=record['processor'],
        code_version=record['code_version'], parameters=record['parameters'],
        validation={**record['validation'], 'rows': claimed_rows})
    monkeypatch.setattr(pq, 'ParquetFile', lambda *a, **k: pytest.fail('Contradictory native preparation opened'))
    monkeypatch.setattr('teadata.scripts.build_tapr_profiles.tempfile.mkdtemp',
        lambda *a, **k: pytest.fail('Contradictory preparation created staging'))
    with pytest.raises(ValueError, match='selected complete source preparation'):
        prepare_profiles(cfg, edition='2025', scopes=['campus', 'district'], input_dir=None,
            output_dir=tmp_path / 'wrong-count', reference_store=store,
            prepared_inputs={**inputs, dataset: (replacement, keys)})
    assert not (tmp_path / 'wrong-count').exists()


@pytest.mark.parametrize('damage', ['missing', 'zero_data', 'overlap', 'bool', 'float', 'inverted'])
def test_reviewed_row_count_requires_unambiguous_nonempty_integer_layout(damage):
    from teadata.tapr import reviewed_tapr_data_record_count
    layout = {'code_record': 2, 'label_record': 1, 'first_data_record': 3, 'last_data_record': 4, 'annotations': []}
    if damage == 'missing':
        mapping = {}
    else:
        mapping = {'layout': layout}
        if damage in {'zero_data', 'overlap'}:
            note = {'first_record': 3, 'last_record': 4, 'kind': 'blank', 'sha256': '0' * 64}
            layout['annotations'] = [note] if damage == 'zero_data' else [note, note]
        elif damage == 'inverted':
            layout['last_data_record'] = 2
        else:
            layout['first_data_record'] = True if damage == 'bool' else 3.0
    with pytest.raises(ValueError):
        reviewed_tapr_data_record_count(mapping)


def test_reviewed_row_count_subtracts_only_annotation_intersection():
    from teadata.tapr import reviewed_tapr_data_record_count
    layout = {'code_record': 2, 'label_record': 1, 'first_data_record': 3, 'last_data_record': 6,
        'annotations': [{'first_record': 4, 'last_record': 4, 'kind': 'note', 'sha256': '0' * 64},
                        {'first_record': 6, 'last_record': 8, 'kind': 'footer', 'sha256': '1' * 64}]}
    assert reviewed_tapr_data_record_count({'layout': layout}) == 2


@pytest.mark.parametrize('prepared_cohort', [
    {'year': 2024, 'families': ('graduation_profile',)},
    {'year': 2024, 'families': ('student_staff_profile',), 'physical_table': 'profile'},
], indirect=True)
def test_explicit_prepared_selection_accepts_two_scopes_or_one_source_without_physical_family_guess(prepared_cohort, tmp_path):
    import pyarrow.dataset as ds
    from teadata.release import DataRelease

    cfg, store, inputs, _, _ = prepared_cohort
    for count in (2, 1):
        selected = dict(list(inputs.items())[:count])
        output = tmp_path / f'selected-{count}'
        result = prepare_profiles(cfg, edition='2024', scopes=[], input_dir=None, output_dir=output,
            reference_store=store, prepared_inputs=selected)
        assert set(result['datasets']) == set(selected)
        release = DataRelease(output / 'release.json', offline=True)
        assert len(list(release.catalog.editions())) == count
        for row in result['datasets'].values():
            key = EditionKey.from_dict(row['key'])
            declared = cfg.resolve_dataset(key.dataset, 2024).reader['profiles']['source']['tapr_source']
            assert key.table == next(iter(declared['tables']))
            projection, = release.catalog.resolve_artifacts([(key, 'source_observation')])
            selected_rows = release.analytical_dataset(projection['artifact'].identity).scanner(
                columns=list(projection['columns']), filter=ds.field('source_observation') == '***', batch_size=1).to_table()
            assert selected_rows.num_rows == 1 and selected_rows.column_names == [key.scope + '_number', 'source_observation']
            assert row['rows'] == 2 and row['fields'] == 2 and row['raw_string_readback_equal'] is True


@pytest.mark.parametrize('prepared_cohort', [{'year': 2024, 'families': ('graduation_profile',)}], indirect=True)
@pytest.mark.parametrize('damage', ['missing_profile', 'scope', 'table', 'ambiguous', 'family', 'unknown_family', 'no_store', 'unselected_dependency'])
def test_prepared_selection_rejects_inconsistent_declarations_before_native_io(prepared_cohort, tmp_path, monkeypatch, damage):
    import pyarrow.parquet as pq
    cfg, store, inputs, _, config = prepared_cohort
    dataset = 'campus_tapr_graduation_profile'
    identifier, keys = inputs[dataset]
    selected = {dataset: (identifier, keys)}
    document = json.loads(config.read_text())
    declaration = document['data_sources'][dataset]['2024']
    profile = declaration['reader']['profiles']['source']
    if damage in {'no_store', 'unselected_dependency'}:
        pass
    elif damage == 'missing_profile':
        declaration['reader']['profiles'].clear()
    elif damage == 'scope':
        profile['tapr_source']['scope'] = 'district'
    elif damage == 'table':
        profile['tapr_source']['tables'] = {'not_the_reviewed_table': 'source.csv'}
    elif damage == 'unknown_family':
        renamed = 'campus_tapr_not_registered'
        document['data_sources'][renamed] = document['data_sources'].pop(dataset)
        selected = {renamed: (identifier, keys)}
    elif damage == 'family':
        record = store.read_record(identifier)
        changed = store.record_processing(dataset=dataset, inputs=record['inputs'],
            outputs={name: store.verify_artifact(value['sha256']) for name, value in record['outputs'].items()},
            processor=record['processor'], code_version=record['code_version'],
            parameters={**record['parameters'], 'family': 'student_profile'}, validation=record['validation'])
        selected = {dataset: (changed, keys)}
    else:
        path = tmp_path / profile['column_schema']
        crosswalk = json.loads(path.read_text())
        tables = crosswalk['datasets'][0]['tables']
        tables.append({**deepcopy(tables[0]), 'table': 'second_table'})
        path.write_text(canonical_json(crosswalk))
        profile['references'][0]['reviewed_sha256'] = sha256(path.read_bytes()).hexdigest()
        profile['tapr_source']['tables']['second_table'] = 'second.csv'
    cfg = Config.from_dict(document)
    cfg.base_dir = tmp_path
    monkeypatch.setattr(pq, 'ParquetFile', lambda *a, **k: pytest.fail('Ambiguous native source opened'))
    with pytest.raises((ValueError, ContractError, KeyError)):
        prepare_profiles(cfg, edition='2024', scopes=['campus', 'district'], input_dir=None,
            output_dir=tmp_path / 'blocked-selection', reference_store=None if damage == 'no_store' else store,
            prepared_inputs=selected, processing_inputs={'district_tapr_graduation_profile':
                [inputs['district_tapr_graduation_profile'][0]]} if damage == 'unselected_dependency' else None)
    assert not (tmp_path / 'blocked-selection').exists() and not list(tmp_path.glob('.tapr-edition-*'))


@pytest.mark.parametrize('prepared_cohort', [
    {'year': 2024, 'families': ('student_staff_profile',), 'physical_table': 'profile'},
], indirect=True)
def test_prepared_prof_keeps_shared_unreviewed_methodology_rejection(prepared_cohort, tmp_path, monkeypatch):
    import pyarrow.parquet as pq
    _, store, inputs, _, config = prepared_cohort
    document = json.loads(config.read_text())
    document['value_policies'] = {'masked': {'policy_id': 'masked', 'reference_id': 'masking',
        'tokens': {'-1': {'status': 'suppressed', 'reason': 'synthetic policy'}}}}
    dataset = 'campus_tapr_student_staff_profile'
    metadata = document['data_sources'][dataset]['2024']['metadata']
    metadata['value_policy'] = 'masked'
    metadata['references'].append({'reference_id': 'masking', 'role': 'masking_methodology', 'status': 'unreviewed'})
    cfg = Config.from_dict(document)
    cfg.base_dir = tmp_path
    monkeypatch.setattr(pq, 'ParquetFile', lambda *a, **k: pytest.fail('Unreviewed methodology reached native source'))
    with pytest.raises(ContractError, match='retained reviewed methodology'):
        prepare_profiles(cfg, edition='2024', scopes=['campus', 'district'], input_dir=None,
            output_dir=tmp_path / 'unreviewed-prof', reference_store=store, prepared_inputs={dataset: inputs[dataset]})
    assert not (tmp_path / 'unreviewed-prof').exists()
