"""GRAD stays a lossless, source-qualified family with honest dictionary gaps."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import csv
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from teadata.column_schema import (
    header_records, parse_column_schema, source_schema_fingerprint, validate_header_table,
    validate_selected_crosswalk,
)
from teadata.dataset_metadata import DatasetMetadata, ResolvedDataset
from teadata.manifest import ContractError, canonical_json
from teadata.provenance import ProvenanceStore
from teadata.scripts.build_tapr_profiles import prepare_profiles
from teadata.tapr import merge_tapr_tables, read_tapr_table
from teadata.teadata_config import Config, load_config


ROOT = Path(__file__).resolve().parents[1]
RESOURCE = ROOT / 'teadata/schemas/tapr--2025--graduation-profile-crosswalk.json'
PERIOD = {'kind': 'school_year', 'start_year': 2023, 'end_year': 2024}


@pytest.mark.parametrize('scope,identities', [('campus', 4), ('district', 2)])
@pytest.mark.parametrize('edition,ending,period', [(2024, '23', {'kind': 'school_year', 'start_year': 2022, 'end_year': 2023}),
    (2025, '24', PERIOD)])
def test_full_publisher_coverage_is_discoverable_with_explicit_dictionary_gaps(scope, identities, edition, ending, period):
    cfg = load_config()
    dataset = scope + '_tapr_graduation_profile'
    records = cfg.header_records(dataset, edition=str(edition), reader_profile='source')
    assert len(records) == 42 + identities
    facts = [f for f in records if f['disposition'] == 'fact']
    missing = [f for f in facts if f['dictionary_status'] == 'not_documented']
    prefix = scope[0].upper()
    assert {f['code'] for f in missing} == {prefix + c + ending + kind for c, kind in [('A0GS', 'N'), ('A0GM', 'R'), ('A0GS', 'R')]}
    assert all(f['publisher_type'] is None and f['publisher_description'] is None for f in missing)
    assert sum(f['dictionary_status'] == 'documented' for f in facts) == 39
    assert all(f['period'] == period for f in facts)
    assert all(f['period'] == {'kind': 'unknown'} for f in records if f['disposition'] == 'identity')
    source = cfg.resolve_dataset(dataset, edition)
    assert source.metadata.version == str(edition) and source.metadata.period.to_dict() == period
    assert len(source.metadata.metrics) == 42
    assert all(m.unit is None and m.period is not None and m.period.to_dict() == period
               for m in source.metadata.metrics.values())
    assert cfg.availability_report()['data_sources'][dataset]['years'] == [2024, 2025]
    assert cfg.is_source_only(dataset, edition) is True
    assert cfg.column_schema(dataset, edition) is None
    with pytest.raises(ContractError, match='Reader profile is required'):
        cfg.load_df(dataset, edition)
    with pytest.raises(ContractError, match='edition unavailable'):
        cfg.header_records(dataset, edition='2023', reader_profile='source')
    with pytest.raises(ContractError, match='exact declared edition'):
        cfg.read_tapr_sources(dataset, 2026)
    # A source-only contract cannot certify an unrelated prepared workbook.
    with pytest.raises(ValueError, match='different prepared input'):
        validate_selected_crosswalk(parse_column_schema(RESOURCE.with_name(f'tapr--{edition}--graduation-profile-crosswalk.json').read_bytes()),
            dataset=dataset, edition=str(edition), input_sha256='0' * 64, columns=[f['field'] for f in records])


@pytest.mark.parametrize('change', ['missing_status', 'invented_type', 'invented_definition', 'false_documented'])
def test_undocumented_fields_cannot_lose_unknown_status_or_gain_inferred_semantics(change):
    document = parse_column_schema(RESOURCE.read_bytes())
    for dataset in document['datasets']:
        for original in dataset['tables'][0]['fields']:
            if original['dictionary_status'] != 'not_documented':
                continue
            table = deepcopy(dataset['tables'][0])
            field = table['fields'][original['position'] - 1]
            if change == 'missing_status':
                field.pop('dictionary_status')
            elif change == 'invented_type':
                field['publisher_type'] = 'Number'
            elif change == 'invented_definition':
                field['publisher_description'] = 'Guessed from a neighboring field'
            else:
                field['dictionary_status'] = 'documented'
            with pytest.raises(ValueError, match='dictionary status'):
                validate_header_table(table)


def test_undeclared_43rd_metric_fails_even_when_all_original_columns_remain(tmp_path):
    table = deepcopy(parse_column_schema(RESOURCE.read_bytes())['datasets'][0]['tables'][0])
    fields = table['fields']
    path = tmp_path / 'unreviewed-extra.csv'
    with path.open('w', newline='') as stream:
        csv.writer(stream).writerows([[f['label'] for f in fields] + ['Unreviewed metric'],
            [f['code'] for f in fields] + ['NEW43'], ['001902001', 'School', '001902', 'District'] + ['0'] * 43])
    table['input_sha256'] = sha256(path.read_bytes()).hexdigest()
    table.pop('layout')
    with pytest.raises(ValueError, match='record 3 has 47 fields; expected 46'):
        read_tapr_table(path, mapping=table, key_field='campus_number',
            dataset='campus_tapr_graduation_profile', edition='2025')


def test_source_only_contract_requires_explicit_unknown_metadata():
    document = parse_column_schema(RESOURCE.read_bytes())
    for dataset in document['datasets']:
        for field in dataset['tables'][0]['fields']:
            if field['dictionary_status'] == 'not_documented':
                for key in ('dictionary_status', 'publisher_type', 'publisher_description'):
                    field.pop(key)
    with pytest.raises(ValueError, match='publisher dictionary status'):
        parse_column_schema(canonical_json(document).encode())


def test_returned_metric_periods_are_detached_from_siblings_and_source_document():
    document = parse_column_schema(RESOURCE.read_bytes())
    original = deepcopy(document)
    records = header_records(document, dataset='campus_tapr_graduation_profile', edition='2025')
    facts = [field for field in records if field['disposition'] == 'fact']
    facts[0]['period']['start_year'] = 1900
    assert all(field['period'] == PERIOD for field in facts[1:])
    assert document == original
    assert header_records(document, dataset='campus_tapr_graduation_profile', edition='2025')[4]['period'] == PERIOD


@pytest.mark.parametrize('problem', ['missing_parent', 'conflicting_parent', 'duplicate_identity'])
def test_graduation_identity_history_cannot_bypass_shared_identity_validation(tmp_path, problem):
    mapping = deepcopy(parse_column_schema(RESOURCE.read_bytes())['datasets'][0]['tables'][0])
    fields = mapping['fields']
    row = ['001902001', 'Historical campus', '001902', 'Historical district'] + [''] * 42
    if problem == 'missing_parent':
        row[2] = ''
    elif problem == 'conflicting_parent':
        row[2] = '227901'
    rows = [row, row] if problem == 'duplicate_identity' else [row]
    path = tmp_path / 'invalid-identity.csv'
    with path.open('w', newline='') as stream:
        csv.writer(stream).writerows([[f['label'] for f in fields], [f['code'] for f in fields], *rows])
    mapping['input_sha256'] = sha256(path.read_bytes()).hexdigest()
    mapping.pop('layout')
    with pytest.raises(ValueError, match='duplicate campus_number|campus and district identifiers disagree'):
        read_tapr_table(path, mapping=mapping, key_field='campus_number',
            dataset='campus_tapr_graduation_profile', edition='2025')


def grad_fixture(tmp_path, *, known_download=False):
    dataset = 'campus_tapr_graduation_profile'
    path = tmp_path / 'grad.csv'
    labels = ['9 Digit Campus Number', 'Undocumented A', 'Undocumented B', 'Undocumented C']
    codes = ['CAMPUS', 'CA0GS24N', 'CA0GM24R', 'CA0GS24R']
    fields = ['campus_number', 'undocumented_a', 'undocumented_b', 'undocumented_c']
    with path.open('w', newline='') as stream:
        csv.writer(stream).writerows([labels, codes, ['001902001', '', '0', '-3']])
    mapping = {'table': 'graduation_profile', 'metric_period': PERIOD,
        'input_sha256': sha256(path.read_bytes()).hexdigest(),
        'source_schema_sha256': source_schema_fingerprint(labels, codes),
        'fields': [{'position': i, 'code': code, 'label': label, 'field': name,
            'disposition': 'identity' if i == 1 else 'fact',
            'dictionary_status': 'not_applicable' if i == 1 else 'not_documented',
            'publisher_type': None, 'publisher_description': None}
            for i, (code, label, name) in enumerate(zip(codes, labels, fields, strict=True), 1)]}
    document = {'schema_version': 2, 'mapping_version': 1, 'datasets': [
        {'dataset': dataset, 'edition': '2025', 'source_only': True, 'tables': [mapping]}]}
    schema = tmp_path / 'crosswalk.json'
    schema.write_text(canonical_json(document) + '\n')
    source = ResolvedDataset(dataset, 'data_sources', 2025, 2025, str(path),
        DatasetMetadata(version='2025'))
    store = ProvenanceStore(tmp_path / 'original-store')
    raw_id = store.record_acquisition(path, source,
        downloaded_at=datetime(2026, 10, 30, tzinfo=timezone.utc) if known_download else None,
        requested_url='https://example.test/grad.csv' if known_download else None)
    dictionary = tmp_path / 'dictionary.html'
    dictionary.write_text('Publisher response omits these three definitions.')
    dictionary_id = store.record_acquisition(dictionary, replace(source, dataset=dataset + ':dictionary'))
    entry = {'path': str(path), 'metadata': {'version': '2025', 'period': PERIOD,
        'provenance_ids': [raw_id], 'references': [
            {'reference_id': 'dictionary', 'role': 'data_dictionary', 'status': 'available',
             'path': 'dictionary.html',
             'provenance_ids': [dictionary_id], 'reviewed_sha256': sha256(dictionary.read_bytes()).hexdigest()}]},
        'reader': {'profiles': {'source': {'column_schema': 'crosswalk.json',
            'references': [{'reference_id': 'columns', 'role': 'schema', 'status': 'available',
                'path': 'crosswalk.json', 'reviewed_sha256': sha256(schema.read_bytes()).hexdigest()}],
            'tapr_source': {'scope': 'campus', 'tables': {'graduation_profile': str(path)}}}}}}
    cfg = Config.from_dict({'data_sources': {dataset: {2025: entry}}})
    cfg.base_dir = tmp_path
    return cfg, store, raw_id, dictionary_id, path, mapping


@pytest.mark.parametrize('problem', [None, 'missing_original', 'wrong_original', 'missing_dictionary'])
def test_source_only_producer_preserves_original_receipts_and_closed_reference_graph(tmp_path, problem):
    cfg, store, raw_id, dictionary_id, path, mapping = grad_fixture(tmp_path)
    if problem == 'wrong_original':
        dataset = cfg.data_sources['campus_tapr_graduation_profile'].entries[2025]
        cfg.data_sources['campus_tapr_graduation_profile'].entries[2025] = replace(
            dataset, metadata=replace(dataset.metadata, provenance_ids=(dictionary_id,)))
    elif problem == 'missing_dictionary':
        store.verify_artifact(store.read_record(dictionary_id)['artifact']['sha256']).unlink()
    output = tmp_path / 'output'
    if problem is not None:
        with pytest.raises((ValueError, FileNotFoundError)):
            prepare_profiles(cfg, edition='2025', family='graduation_profile', scopes=['campus'],
                input_dir=tmp_path, output_dir=output,
                reference_store=None if problem == 'missing_original' else store)
        assert not output.exists()
        assert not list(tmp_path.glob('.tapr-preparation-*'))
        return
    original = {i: store.read_record(i) for i in (raw_id, dictionary_id)}
    receipt = prepare_profiles(cfg, edition='2025', family='graduation_profile', scopes=['campus'],
        input_dir=tmp_path, output_dir=output, reference_store=store)
    table = pq.read_table(output / 'campus.parquet')
    assert table.to_pylist() == [{'campus_number': '001902001', 'undocumented_a': '',
        'undocumented_b': '0', 'undocumented_c': '-3'}]
    archive = ProvenanceStore(output / 'provenance')
    closure = archive.verify_record_closure([receipt['scopes']['campus']['processing_record_id']])
    assert all(closure[i] == record for i, record in original.items())
    assert receipt['family'] == 'graduation_profile'
    assert receipt['scopes']['campus']['source_tables'][0]['metric_period'] == PERIOD
    assert all(f.type == pa.string() for f in table.schema)


@pytest.mark.parametrize('known_download', [False, True])
def test_local_observation_keeps_known_and_unknown_acquisition_history_factual(tmp_path, known_download):
    cfg, store, raw_id, _, _, _ = grad_fixture(tmp_path, known_download=known_download)
    original = store.read_record(raw_id)
    output = tmp_path / 'output'
    receipt = prepare_profiles(cfg, edition='2025', family='graduation_profile', scopes=['campus'],
        input_dir=tmp_path, output_dir=output, reference_store=store)
    archive = ProvenanceStore(output / 'provenance')
    closure = archive.verify_record_closure([receipt['scopes']['campus']['processing_record_id']])
    local = [record for identifier, record in closure.items() if identifier != raw_id
        and record.get('dataset') == 'campus_tapr_graduation_profile'
        and record.get('method') == 'local_observation']
    assert len(local) == 1
    assert local[0]['downloaded_at'] is None
    assert local[0]['parameters']['observation'] == (
        'local raw-source read; acquisition history is retained separately when available')
    assert closure[raw_id] == original
    assert (original['downloaded_at'] is not None) == known_download


@pytest.mark.parametrize('period', [None, {}, [], 'unknown', {'kind': 'unknown', 'year': None}])
def test_invalid_or_implicit_metric_period_contracts_fail_validation(period):
    document = parse_column_schema(RESOURCE.read_bytes())
    document['datasets'][0]['tables'][0]['metric_period'] = period
    with pytest.raises(ValueError, match='explicit canonical period mapping'):
        parse_column_schema(canonical_json(document).encode())


def test_explicit_unknown_metric_period_remains_discoverable():
    document = parse_column_schema(RESOURCE.read_bytes())
    document['datasets'][0]['tables'][0]['metric_period'] = {'kind': 'unknown'}
    parsed = parse_column_schema(canonical_json(document).encode())
    records = header_records(parsed, dataset='campus_tapr_graduation_profile', edition='2025')
    assert all(record['period'] == {'kind': 'unknown'} for record in records)


def test_null_metric_period_is_rejected_before_discovery_or_source_io(tmp_path, monkeypatch):
    cfg, _, _, _, _, _ = grad_fixture(tmp_path)
    schema = tmp_path / 'crosswalk.json'
    document = parse_column_schema(schema.read_bytes())
    document['datasets'][0]['tables'][0]['metric_period'] = None
    schema.write_text(canonical_json(document) + '\n')
    profile = cfg.data_sources['campus_tapr_graduation_profile'].entries[2025].reader['profiles']['source']
    profile['references'][0]['reviewed_sha256'] = sha256(schema.read_bytes()).hexdigest()
    monkeypatch.setattr('teadata.tapr.read_tapr_table', lambda *a, **k: pytest.fail('Invalid period read source data'))
    with pytest.raises(ContractError, match='explicit canonical period mapping'):
        cfg.header_records('campus_tapr_graduation_profile', edition='2025', reader_profile='source')
    with pytest.raises(ContractError, match='explicit canonical period mapping'):
        cfg.read_tapr_sources('campus_tapr_graduation_profile', 2025)


@pytest.mark.parametrize('scope', ['campus', 'district'])
def test_grad_annual_references_reuse_exact_receipts_without_applying_token_or_unit_rules(scope):
    source = load_config().resolve_dataset(scope + '_tapr_graduation_profile', 2025)
    references = {reference.reference_id: reference for reference in source.metadata.references}
    expected = {
        'glossary': ('glossary', 'https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/2024-25-comprehensive-tapr-glossary-0.pdf',
            'e9cb78a172cd247ca3101727a81e31d6b59e9c7372e8c54863f4f0574e1412db',
            '846810954d479a13467d9e4add3b0ed626247bd48ca9b677ff8a5ec11aedd416'),
        'masking': ('masking_methodology', 'https://rptsvr1.tea.texas.gov/perfreport/tapr/2025/masking.html',
            '6e4908dc2727a5de0610f9a00a3ec8983810f3aa7577d604c80ba53d93992554',
            '7b6e5840645656bf3c0d977431e7f9fcd6ea984bb1734b6387471f8ad4851de8'),
    }
    store = ProvenanceStore(ROOT / 'provenance')
    for name, (role, url, identifier, digest) in expected.items():
        reference = references[name]
        assert reference.role == role and reference.url == url and reference.status == 'available'
        assert reference.provenance_ids == (identifier,) and reference.reviewed_sha256 == digest
        record = store.read_record(identifier)
        assert record['artifact']['sha256'] == digest
        assert record['requested_url'] == url and record['method'] == 'download'
    assert source.metadata.value_policy is None
    assert all(metric.unit is None for metric in source.metadata.metrics.values())


def test_graduation_is_not_merged_by_equal_values_with_another_family_or_period(tmp_path):
    _, _, _, _, path, mapping = grad_fixture(tmp_path)
    source = read_tapr_table(path, mapping=mapping, key_field='campus_number',
        dataset='campus_tapr_graduation_profile', edition='2025')
    foreign = replace(source, dataset='campus_tapr_student_staff_profile')
    with pytest.raises(ValueError, match='different qualified datasets'):
        merge_tapr_tables([source, foreign], key_field='campus_number')
    other_period = {**mapping, 'table': 'other_observation',
        'metric_period': {'kind': 'school_year', 'start_year': 2024, 'end_year': 2025}}
    with pytest.raises(ValueError, match='different metric reference periods'):
        merge_tapr_tables([source, replace(source, mapping=other_period)], key_field='campus_number')


def test_source_only_metadata_period_mismatch_fails_before_reading_rows(tmp_path, monkeypatch):
    cfg, _, _, _, _, _ = grad_fixture(tmp_path)
    dataset = cfg.data_sources['campus_tapr_graduation_profile'].entries[2025]
    cfg.data_sources['campus_tapr_graduation_profile'].entries[2025] = replace(dataset,
        metadata=replace(dataset.metadata, period=replace(dataset.metadata.period, start_year=2024, end_year=2025)))
    monkeypatch.setattr('teadata.tapr.read_tapr_table', lambda *a, **k: pytest.fail('Read contradictory period'))
    with pytest.raises(ContractError, match='metric period differs'):
        cfg.read_tapr_sources('campus_tapr_graduation_profile', 2025)


def test_existing_default_contracts_and_public_metric_bindings_remain_pinned(catalog):
    cfg = yaml.safe_load((ROOT / 'teadata/teadata_sources.yaml').read_bytes())
    expected = {('campus', 2024): 'bedc5ea74bcba865ebf65942878a233fc695bd5cf3b83a2dbfdbc3023415e04d',
        ('campus', 2025): '36fc570425e654fbc50bb257042d7078986626741f68745446316e8ea9860df4',
        ('district', 2024): '63bc2ae7dacef0db0a0491d2f02e9d5c2c80bc3f7474419c4ed5b75af1e4979d',
        ('district', 2025): 'fbdd676a4a2b3be9a2df36dfcd84337772e0247e64fe9f7c1c927c519f23dcbc'}
    for (scope, year), digest in expected.items():
        entry = cfg['data_sources'][scope + '_tapr_student_staff_profile'][year]
        assert sha256(canonical_json(entry).encode()).hexdigest() == digest
    for scope, digest in [('campus', 'da06b3e24f7acc2b553c25f09e35d4f21fbfb8ef1c270b6a7cba80317ceda2fa'),
                         ('district', '172240f0850fa64d294307874fcef3e69c5d94ee14e39da83d2682c75ae97c61')]:
        assert sha256(canonical_json(cfg['data_sources'][scope + '_tapr_graduation_profile'][2025]).encode()).hexdigest() == digest
    assert all('graduation' not in canonical_json(binding) for binding in cfg['metric_bindings'].values())
    assert not any('graduation' in dataset for dataset in catalog.datasets)
    loaded = load_config()
    assert not loaded.is_source_only('accountability', 2025)
    assert all(not loaded.is_source_only(scope + '_tapr_student_staff_profile', year)
               for scope in ('campus', 'district') for year in (2024, 2025))


@pytest.mark.parametrize('scope', ['campus', 'district'])
def test_2024_grad_period_and_dictionary_contradiction_are_discoverable_without_coalescing(scope):
    source = load_config().resolve_dataset(scope + '_tapr_graduation_profile', 2024)
    assert source.metadata.version == '2024'
    assert source.metadata.period.to_dict() == {'kind': 'school_year', 'start_year': 2022, 'end_year': 2023}
    notes = source.metadata.notes
    assert notes and '2023-24 Annual Graduates' in notes and 'pages 15-16' in notes
    assert 'not inferred from source-code suffixes' in notes and 'independent of retained 2024 PROF' in notes
    refs = {ref.reference_id: ref for ref in source.metadata.references}
    assert {refs[name].role for name in ('dictionary', 'glossary', 'masking', 'selection')} == {
        'data_dictionary', 'glossary', 'masking_methodology', 'source_notes'}
    assert refs['glossary'].reviewed_sha256 == 'e92631bfd01e9e2e5f4197bfe996069822c852dd5db423fe3faeea5ab7aa62e8'
    assert all(ref.provenance_ids for ref in source.metadata.references)
    assert source.metadata.value_policy is None


def test_2024_current_source_preserves_dot_blank_and_historical_members_independently(tmp_path):
    resource = RESOURCE.with_name('tapr--2024--graduation-profile-crosswalk.json')
    mapping = deepcopy(parse_column_schema(resource.read_bytes())['datasets'][0]['tables'][0])
    fields = mapping['fields']
    rows = [['001902001', 'Historical campus', '001902', 'Historical district', '.'] + [''] * 41,
            ['001902041', 'Another historical campus', '001902', 'Historical district', ''] + ['0'] * 41]
    path = tmp_path / 'source.csv'
    with path.open('w', newline='') as stream:
        csv.writer(stream).writerows([[f['label'] for f in fields], [f['code'] for f in fields], *rows])
    mapping['input_sha256'] = sha256(path.read_bytes()).hexdigest()
    mapping['layout']['last_data_record'] = 4
    result = read_tapr_table(path, mapping=mapping, key_field='campus_number',
        dataset='campus_tapr_graduation_profile', edition='2024')
    assert result.table.num_rows == 2
    assert result.table[fields[4]['field']].to_pylist() == ['.', '']
    assert result.table['campus_name'].to_pylist() == ['Historical campus', 'Another historical campus']
    assert result.table['campus_number'].to_pylist() == ['001902001', '001902041']
    assert result.edition == '2024' and result.dataset == 'campus_tapr_graduation_profile'


@pytest.mark.parametrize('edition', [2024, 2025])
def test_grad_discovery_opens_only_metadata_in_a_fresh_process(edition):
    code = '''
import importlib.abc, pathlib, socket, sqlite3, sys, urllib.request
heavy = ('teadata.classes','teadata.engine','teadata.entities','teadata.geometry','pyarrow','duckdb','sqlalchemy','pandas','numpy','geopandas','shapely')
class Trap(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if any(fullname == p or fullname.startswith(p + '.') for p in heavy): raise AssertionError('Heavy import: ' + fullname)
sys.meta_path.insert(0, Trap())
from teadata import load_config
import teadata.teadata_config as config_module
root = pathlib.Path(config_module.__file__).parent
edition = __EDITION__
allowed = {(root/'teadata_sources.yaml').resolve(),(root/f'schemas/tapr--{edition}--graduation-profile-crosswalk.json').resolve()}
original = pathlib.Path.open
def guarded(self,*args,**kwargs):
 if self.resolve() not in allowed: raise AssertionError('Non-metadata open: ' + str(self))
 return original(self,*args,**kwargs)
pathlib.Path.open = guarded
def forbidden(*args,**kwargs): raise AssertionError('Data/network/database I/O')
socket.create_connection = forbidden
sqlite3.connect = forbidden
urllib.request.urlopen = forbidden
cfg = load_config()
for scope, width in [('campus',46),('district',44)]:
 ds = scope + '_tapr_graduation_profile'
 records = cfg.header_records(ds,edition=str(edition),reader_profile='source')
 assert len(records) == width
 assert cfg.availability_report()['data_sources'][ds]['years'] == [2024,2025]
 assert cfg.is_source_only(ds,edition)
 assert len(cfg.resolve_dataset(ds,edition).metadata.metrics) == 42
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in heavy)
'''
    result = subprocess.run([sys.executable, '-B', '-c', code.replace('__EDITION__', str(edition))], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
