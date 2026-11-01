"""Independent publisher families preserve revisions, periods and raw tokens."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import csv
from pathlib import Path
import subprocess
import sys

import pytest

from teadata.column_schema import header_records, parse_column_schema, source_schema_fingerprint
from teadata.dataset_metadata import DataPeriod, DatasetMetadata, MetricDefinition, ResolvedDataset
from teadata.manifest import BuildRecorder, ContractError, DatasetCatalog, canonical_json
from teadata.provenance import ProvenanceStore
from teadata.scripts.build_tapr_profiles import prepare_profiles
from teadata.tapr import merge_tapr_tables
from teadata.teadata_config import Config, load_config


ROOT = Path(__file__).resolve().parents[1]
UNKNOWN = {'kind': 'unknown'}
LAGGED = {'kind': 'school_year', 'start_year': 2022, 'end_year': 2023}
REPORT = {'kind': 'school_year', 'start_year': 2023, 'end_year': 2024}
NUMERIC_GOLDEN_ORIGIN = {
    'raw_sha256': '8dc181ee75c5da8451a45c8f405f8cbc8fe9987efb198d9ded90d7d512f161a0',
    'acquisition_id': '9b115abb5b7c0f5ce4ebdd82f36fc14e06ab55658b496f31064776277b16658c',
    'six_change_audit_sha256': '451de1dd1119eed1066e51ea157c06b123da505d65ad19fd9155877ec696929f',
}


@pytest.mark.parametrize('scope,widths', [('campus', [243, 106]), ('district', [242, 118])])
def test_four_natural_source_leaves_are_complete_independent_and_have_no_default(scope, widths):
    cfg = load_config()
    for family, width in zip(['student_profile', 'staff_profile'], widths, strict=True):
        dataset = scope + '_tapr_' + family
        source = cfg.resolve_dataset(dataset, 2024)
        records = cfg.header_records(dataset, edition='2024', reader_profile='source')
        assert len(records) == width and cfg.column_schema(dataset, 2024) is None
        assert cfg.availability_report()['data_sources'][dataset]['years'] == [2024, 2025]
        assert cfg.is_source_only(dataset, 2024)
        assert source.metadata.version == '2024' and source.metadata.period.to_dict() == UNKNOWN
        assert source.metadata.value_policy is None and not source.metadata.provenance_ids
        assert len({r['code'] for r in records}) == width == len({r['field'] for r in records})
        expected = {scope[0].upper() + suffix for suffix in ['PETATTP', 'PEMALLP', 'PNTALLC']} if family == 'student_profile' else {scope[0].upper() + 'PSATOFC'}
        assert {r['code'] for r in records if r['period'] != UNKNOWN} == expected
        for record in records:
            if record['code'] in expected:
                assert record['period'] == (LAGGED if record['code'].endswith(('PETATTP', 'PEMALLP')) else REPORT)
        units = {name: metric.unit for name, metric in source.metadata.metrics.items()}
        assert units == ({'student_membership_attrition_all_students_percent': 'percent',
                          'student_membership_mobility_all_students_percent': 'percent',
                          'student_enrollment_all_students_count': 'count'} if family == 'student_profile'
                         else {'staff_all_staff_total_full_time_equiv_count': None})
        assert source.metadata.notes and 'snapshot' in source.metadata.notes and 'FTE is not headcount' in source.metadata.notes
        assert all(r['period'] == UNKNOWN for r in records if r['disposition'] == 'identity')
        assert all(r['table'] == family for r in records)
        with pytest.raises(ContractError, match='Reader profile is required'):
            cfg.load_df(dataset, 2024)
        with pytest.raises(ContractError, match='edition unavailable'):
            cfg.header_records(dataset, edition='2026', reader_profile='source')
        with pytest.raises(ContractError, match='exact declared edition'):
            cfg.read_tapr_sources(dataset, 2026)


def test_dictionary_gaps_unused_codes_and_identical_occurrences_are_preserved():
    cfg = load_config()
    campus = cfg.header_records('campus_tapr_staff_profile', edition='2024', reader_profile='source')
    field = next(r for r in campus if r['code'] == 'CPSCTOFC')
    assert field['dictionary_status'] == 'not_documented'
    assert field['publisher_type'] is None and field['publisher_description'] is None
    assert field['dictionary_occurrences'] == [] and field['period'] == UNKNOWN
    assert 'CPSCTOST' not in {r['code'] for r in campus}
    for family, codes in [('student_profile', ['DAUND23C']), ('staff_profile', ['DPSETOFC', 'DPSXTOFC'])]:
        records = cfg.header_records('district_tapr_' + family, edition='2024', reader_profile='source')
        for code in codes:
            records = cfg.header_records('district_tapr_' + family, edition='2024', reader_profile='source')
            record = next(r for r in records if r['code'] == code)
            occurrences = record['dictionary_occurrences']
            original = deepcopy(records)
            assert len(occurrences) == 2 and occurrences[0]['ordinal'] != occurrences[1]['ordinal']
            assert {r['publisher_description'] for r in occurrences} == {record['publisher_description']}
            occurrences[0]['publisher_description'] = 'Caller mutation'
            assert cfg.header_records('district_tapr_' + family, edition='2024', reader_profile='source') == original
    document = parse_column_schema((ROOT / 'teadata/schemas/tapr--2024--staff-profile-crosswalk.json').read_bytes())
    for dataset in document['datasets']:
        table = dataset['tables'][0]
        assert dataset['dataset'][0].upper() + 'PSCTOST' in {r['code'] for r in table['dictionary_records']}
        assert dataset['dataset'][0].upper() + 'PSCTOST' not in {r['code'] for r in table['fields']}


def test_real_dictionary_projection_uses_linear_work_and_detaches_ordered_duplicates():
    class CountingRows(list):
        visits = 0

        def __iter__(self):
            for row in super().__iter__():
                self.visits += 1
                yield row

    cfg = load_config()
    resource = ROOT / 'teadata/schemas/tapr--2024--staff-profile-crosswalk.json'
    for scope in ['campus', 'district']:
        dataset = scope + '_tapr_staff_profile'
        document = parse_column_schema(resource.read_bytes())
        table = next(d['tables'][0] for d in document['datasets'] if d['dataset'] == dataset)
        original_rows = deepcopy(table['dictionary_records'])
        counted = CountingRows(original_rows)
        table['dictionary_records'] = counted
        metadata = cfg.resolve_dataset(dataset, 2024).metadata
        expected = cfg.header_records(dataset, edition='2024', reader_profile='source')
        records = header_records(document, dataset=dataset, edition='2024', metadata=metadata)
        # Count actual dictionary-row visits rather than timing small fixtures.
        # The bound allows linear passes over rows/fields and rejects C*D work.
        assert counted.visits <= 2 * (len(original_rows) + len(table['fields']))
        assert records == expected
        assert scope[0].upper() + 'PSCTOST' not in {r['code'] for r in records}
        assert any(r['code'] == scope[0].upper() + 'PSCTOST' for r in original_rows)
        if scope == 'district':
            occurrences = next(r['dictionary_occurrences'] for r in records if r['code'] == 'DPSETOFC')
            assert [r['ordinal'] for r in occurrences] == [r['ordinal'] for r in original_rows if r['code'] == 'DPSETOFC']
            assert len(occurrences) == 2 and occurrences[0] is not occurrences[1]
            occurrences[0]['publisher_description'] = 'Caller mutation'
            assert occurrences[1]['publisher_description'] != 'Caller mutation'
            assert list(counted) == original_rows
            assert header_records(document, dataset=dataset, edition='2024', metadata=metadata) == expected


@pytest.mark.parametrize('problem', ['conflicting_duplicate', 'invented_code', 'false_documented'])
def test_dictionary_conflicts_and_silent_gap_aliases_fail(problem):
    document = parse_column_schema((ROOT / 'teadata/schemas/tapr--2024--staff-profile-crosswalk.json').read_bytes())
    table = document['datasets'][0]['tables'][0]
    if problem == 'conflicting_duplicate':
        duplicate = {**table['dictionary_records'][0], 'ordinal': len(table['dictionary_records']) + 1,
                     'publisher_description': 'Different definition'}
        table['dictionary_records'].append(duplicate)
    else:
        field = next(f for f in table['fields'] if f['code'] == 'CPSCTOFC')
        unused = next(r for r in table['dictionary_records'] if r['code'] == 'CPSCTOST')
        field.update(dictionary_status='documented', publisher_type=unused['publisher_type'],
                     publisher_description=unused['publisher_description'])
        if problem == 'invented_code':
            field['code'] = 'CPSCTOST'
    with pytest.raises(ValueError, match='dictionary|checksum'):
        parse_column_schema(canonical_json(document).encode())


def test_exact_registered_field_projection_refines_unknowns_and_preserves_explicit_unknowns():
    cfg = load_config()
    dataset = 'campus_tapr_student_staff_profile'
    records = cfg.header_records(dataset, edition='2024', reader_profile='source')
    assert all(record['period'] == UNKNOWN for record in records)
    contract = cfg.column_contract(dataset, 2024, reader_profile='source')
    assert contract is not None
    document = parse_column_schema(contract['text'].encode())
    # The legacy selection still owns its qualified metric periods; the raw
    # profile does not inherit that interpretation.
    records = header_records(document, dataset=dataset, edition='2024',
                             metadata=cfg.resolve_dataset(dataset, 2024).metadata)
    period_by_code = {r['code']: r['period'] for r in records}
    assert period_by_code['CPETATTP'] == period_by_code['CPEMALLP'] == LAGGED
    assert period_by_code['CPNTALLC'] == REPORT  # registered enrollment inherits selected metadata
    assert period_by_code['CPETBLAC'] == UNKNOWN  # report/year prefix is insufficient
    assert period_by_code['CAMPUS'] == UNKNOWN
    name = next(r['field'] for r in records if r['code'] == 'CPNTALLC')
    metadata = DatasetMetadata(period=DataPeriod.from_dict(REPORT), field_year=2024, metrics={
        'unrelated_selector': MetricDefinition(name, DataPeriod()),
        'student_membership_attrition_all_students_percent': MetricDefinition('some_other_field', DataPeriod.from_dict(LAGGED)),
    })
    projected = header_records(document, dataset=dataset, edition='2024', metadata=metadata)
    assert all(r['period'] == UNKNOWN for r in projected)


@pytest.mark.parametrize('scope,width,raw_id', [
    ('campus', 407, '5c564e395e98296e0bfe6388f6b7a51743e68832e04db17ec458130f4bfcfbc1'),
    ('district', 420, 'e206f265a486d928a815ed9ebf24c99655813c29b0bc37bc31fa74ba9604021e'),
])
def test_explicit_prof_metadata_binds_context_without_interpretation_or_source_io(scope, width, raw_id, monkeypatch):
    from teadata.manifest import _validate_methodology
    from teadata.provenance import validate_record_graph, validate_reference_records

    cfg = load_config()
    dataset = scope + '_tapr_student_staff_profile'
    base = cfg.resolve_dataset(dataset, 2024)
    raw = cfg.resolve_dataset(dataset, 2024, reader_profile='source')
    assert base.metadata.value_policy == 'tapr_profile' and base.metadata.metrics
    with pytest.raises(ContractError, match='retained reviewed methodology'):
        _validate_methodology(base.metadata, cfg.value_policies[base.metadata.value_policy])
    assert raw.metadata.period.to_dict() == UNKNOWN and raw.metadata.field_year is None
    assert raw.metadata.value_policy is None and raw.metadata.estimation is None and not raw.metadata.metrics
    _validate_methodology(raw.metadata, None)
    assert raw.metadata.provenance_ids == (raw_id,)
    assert cfg.reader_references(dataset, 2024, reader_profile='source') == list(raw.metadata.references)
    assert {ref.reference_id for ref in raw.metadata.references} == {
        'csv_headers', 'advanced_form', 'source_dictionary', 'glossary', 'masking', 'columns_source'}
    store = ProvenanceStore(ROOT / 'provenance')
    monkeypatch.setattr(store, 'verify_artifact', lambda *a, **k: pytest.fail('Metadata lookup hashed a source body'))
    graph = validate_record_graph(store.read_record, [raw_id, *(
        identifier for ref in raw.metadata.references for identifier in ref.provenance_ids)])
    validate_reference_records(raw, graph.__getitem__, verify_all_claimed=True)
    assert graph[raw_id]['method'] == 'download' and graph[raw_id]['downloaded_at'] is not None
    records = cfg.header_records(dataset, edition='2024', reader_profile='source')
    assert len(records) == width and all(record['period'] == UNKNOWN for record in records)
    assert cfg.resolve_dataset(dataset, 2024).to_dict() == base.to_dict()


def period_fixture(tmp_path):
    """One prepared tiny input binds a source crosswalk into a real frozen catalog."""
    dataset = 'campus_tapr_student_profile'
    names = ['campus_number', 'campus_2024_attrition', 'campus_2024_enrollment', 'campus_2024_undeclared']
    labels = ['ID', 'Attrition', 'Enrollment', 'Undeclared']
    codes = ['CAMPUS', 'ATTR', 'ENROLL', 'OPAQUE']
    prepared = tmp_path / 'prepared.csv'
    with prepared.open('w', newline='') as stream:
        csv.writer(stream).writerows([names, ['001902001', '1', '12', '0']])
    table = {'table': 'student_profile', 'metric_period': UNKNOWN,
        'input_sha256': 'a' * 64, 'source_schema_sha256': source_schema_fingerprint(labels, codes),
        'fields': [{'position': i, 'code': code, 'label': label, 'field': name,
                    'disposition': 'identity' if i == 1 else 'fact'}
                   for i, (code, label, name) in enumerate(zip(codes, labels, names, strict=True), 1)]}
    document = {'schema_version': 2, 'mapping_version': 1, 'datasets': [
        {'dataset': dataset, 'edition': '2024', 'prepared_input_sha256': sha256(prepared.read_bytes()).hexdigest(),
         'tables': [table]}]}
    schema = tmp_path / 'schema.json'
    schema.write_text(canonical_json(document) + '\n')
    store = ProvenanceStore(tmp_path / 'provenance')
    schema_id = store.record_acquisition(schema, ResolvedDataset('reference:schema', 'reference', None, None,
        str(schema), DatasetMetadata(title='Tiny reviewed schema')))
    entry = {'path': str(prepared), 'metadata': {'version': '2024', 'period': REPORT, 'field_year': 2024,
        'metrics': {'attrition': {'field': names[1], 'period': LAGGED},
                    'enrollment': {'field': names[2]}},
        'references': [{'reference_id': 'schema', 'role': 'schema', 'path': schema.name,
            'reviewed_sha256': sha256(schema.read_bytes()).hexdigest(), 'provenance_ids': [schema_id]}]},
        'reader': {'dtype': 'str', 'column_schema': schema.name}}
    cfg = Config.from_dict({'data_sources': {dataset: {2024: entry}}})
    cfg.base_dir = tmp_path
    source = cfg.resolve_dataset(dataset, 2024)
    raw_id = store.record_acquisition(prepared, source)
    definition = cfg.data_sources[dataset].entries[2024]
    cfg.data_sources[dataset].entries[2024] = replace(definition,
        metadata=replace(definition.metadata, provenance_ids=(raw_id,)))
    return cfg, store, dataset, schema, document


def test_config_and_frozen_catalog_share_period_projection_without_current_config_io(tmp_path, monkeypatch):
    cfg, store, dataset, _, _ = period_fixture(tmp_path)
    records = cfg.header_records(dataset, edition='2024')
    with BuildRecorder(cfg, store) as recorder:
        cfg.load_df(dataset, 2024)
    catalog = recorder.catalog()
    assert catalog.header_records(dataset) == records
    definition = cfg.data_sources[dataset].entries[2024]
    cfg.data_sources[dataset].entries[2024] = replace(definition, metadata=replace(definition.metadata,
        metrics={'attrition': MetricDefinition('campus_2024_attrition', DataPeriod.from_dict(REPORT))}))
    assert cfg.header_records(dataset, edition='2024') != records
    monkeypatch.setattr(Path, 'open', lambda *a, **k: pytest.fail('Frozen catalog opened a file'))
    monkeypatch.setattr(Config, 'resolve_dataset', lambda *a, **k: pytest.fail('Frozen catalog consulted current config'))
    assert catalog.header_records(dataset) == records
    records[1]['period']['start_year'] = 1900
    assert catalog.header_records(dataset)[1]['period'] == LAGGED


@pytest.mark.parametrize('problem', ['table_conflict', 'duplicate_field_conflict'])
def test_conflicting_reviewed_periods_fail_before_source_reads(tmp_path, monkeypatch, problem):
    cfg = load_config()
    dataset = 'campus_tapr_student_profile'
    definition = cfg.data_sources[dataset].entries[2024]
    if problem == 'table_conflict':
        contract = cfg.column_contract(dataset, 2024, reader_profile='source')
        assert contract is not None
        document = parse_column_schema(contract['text'].encode())
        document['datasets'][0]['tables'][0]['metric_period'] = REPORT
        schema = tmp_path / 'schema.json'
        schema.write_text(canonical_json(document) + '\n')
        reader = deepcopy(definition.reader)
        reader['profiles']['source']['column_schema'] = schema.name
        reference = reader['profiles']['source']['references'][0]
        reference.update(path=schema.name, reviewed_sha256=sha256(schema.read_bytes()).hexdigest())
        cfg.base_dir = tmp_path
        definition = replace(definition, reader=reader)
    else:
        metrics = dict(definition.metadata.metrics)
        first = next(iter(metrics.values()))
        metrics['contradictory'] = replace(first, period=DataPeriod.from_dict(REPORT))
        definition = replace(definition, metadata=replace(definition.metadata, metrics=metrics))
    cfg.data_sources[dataset].entries[2024] = definition
    monkeypatch.setattr('teadata.tapr.read_tapr_table', lambda *a, **k: pytest.fail('Contradictory metadata read rows'))
    for operation in [lambda: cfg.header_records(dataset, edition='2024', reader_profile='source'),
                      lambda: cfg.read_tapr_sources(dataset, 2024)]:
        with pytest.raises(ContractError, match='Conflicting.*periods'):
            operation()


def test_frozen_catalog_rejects_period_conflict_on_construction(tmp_path):
    cfg, store, dataset, _, _ = period_fixture(tmp_path)
    with BuildRecorder(cfg, store) as recorder:
        cfg.load_df(dataset, 2024)
    document = recorder.catalog().to_dict()
    source = document['datasets'][dataset]
    source['metadata']['metrics']['contradictory'] = {'field': 'campus_2024_attrition', 'period': REPORT}
    identifier = source['metadata']['provenance_ids'][0]
    record = document['provenance'].pop(identifier)
    record['source_metadata']['metrics'] = deepcopy(source['metadata']['metrics'])
    replacement = sha256((canonical_json(record) + '\n').encode()).hexdigest()
    document['provenance'][replacement] = record
    source['metadata']['provenance_ids'] = [replacement]
    with pytest.raises(ContractError, match='Conflicting reviewed field periods'):
        DatasetCatalog.create(datasets=document['datasets'], records=document['provenance'],
            value_policies=document['value_policies'], metric_bindings=document['metric_bindings'],
            column_schemas=document['column_schemas'], processor=document['processor_sha256'])


def source_fixture(tmp_path):
    cfg = load_config()
    dataset = 'campus_tapr_student_profile'
    contract = cfg.column_contract(dataset, 2024, reader_profile='source')
    assert contract is not None
    document = parse_column_schema(contract['text'].encode())
    selected = document['datasets'][0]
    document['datasets'] = [selected]
    table = selected['tables'][0]
    assert table['input_sha256'] == NUMERIC_GOLDEN_ORIGIN['raw_sha256']
    reference = next(r for r in cfg.resolve_dataset(dataset, 2024).metadata.references if r.reference_id == 'csv_headers')
    assert reference.provenance_ids == (NUMERIC_GOLDEN_ORIGIN['acquisition_id'],)
    path = tmp_path / 'source.csv'
    fields = table['fields']
    # Six observed nonzero lexical goldens, plus dot, blank and zero. These are
    # synthetic historical identities; the real acquisition retains exact IDs.
    # Exact public-input custody is pinned in NUMERIC_GOLDEN_ORIGIN above.
    examples = [('CPCTENGA', '7.5'), ('CPCTENGA', '2'), ('CPCTENGA', '1'),
                ('CPCTFLAA', '1'), ('CPCTG01A', '7.5'), ('CPCTGKGA', '7')]
    rows = []
    for index, (code, value) in enumerate(examples, 1):
        row = [''] * len(fields)
        for f in fields:
            row[f['position'] - 1] = {'CAMPUS': f'001902{index:03d}', 'CAMPNAME': 'Historical campus',
                'DISTRICT': '001902', 'DISTNAME': 'Historical district'}.get(f['code'], '')
        row[next(f['position'] - 1 for f in fields if f['code'] == code)] = value
        row[4:7] = ['.', '', '0']
        rows.append(row)
    with path.open('w', newline='') as stream:
        csv.writer(stream).writerows([[f['label'] for f in fields], [f['code'] for f in fields], *rows])
    table['input_sha256'] = sha256(path.read_bytes()).hexdigest()
    table['layout']['last_data_record'] = len(rows) + 2
    schema = tmp_path / 'schema.json'
    schema.write_text(canonical_json(document) + '\n')
    original = ProvenanceStore(tmp_path / 'original')
    raw_id = original.record_acquisition(path, ResolvedDataset('campus_tapr_student_staff_profile',
        'data_sources', 2024, 2024, str(path), DatasetMetadata(version='2024')),
        downloaded_at=datetime(2026, 10, 30, tzinfo=timezone.utc),
        requested_url='https://example.test/original.csv')
    entry = {'path': str(path), 'metadata': {'version': '2024', 'period': UNKNOWN, 'references': [{
        'reference_id': 'csv_headers', 'role': 'source_notes', 'url': 'https://example.test/original.csv',
        'reviewed_sha256': table['input_sha256'], 'provenance_ids': [raw_id]}]},
        'reader': {'profiles': {'source': {'column_schema': schema.name,
            'references': [{'reference_id': 'schema', 'role': 'schema', 'path': schema.name,
                'reviewed_sha256': sha256(schema.read_bytes()).hexdigest()}],
            'tapr_source': {'scope': 'campus', 'tables': {'student_profile': str(path)}}}}}}
    cfg = Config.from_dict({'data_sources': {dataset: {2024: entry}}})
    cfg.base_dir = tmp_path
    return cfg, original, raw_id, path, table, examples


def test_raw_revision_goldens_preserve_dot_blank_zero_six_values_and_historical_keys(tmp_path):
    cfg, _, _, _, _, examples = source_fixture(tmp_path)
    source = cfg.read_tapr_sources('campus_tapr_student_profile', 2024)[0]
    result = source.table
    assert result.num_rows == 6 and result['campus_number'].to_pylist() == [f'001902{i:03d}' for i in range(1, 7)]
    assert result.column(4).to_pylist() == ['.'] * 6
    assert result.column(5).to_pylist() == [''] * 6
    assert result.column(6).to_pylist() == ['0'] * 6
    by_code = {f['code']: f['field'] for f in source.mapping['fields']}
    assert [result[by_code[code]][i].as_py() for i, (code, _) in enumerate(examples)] == [v for _, v in examples]
    with pytest.raises(ValueError, match='different qualified datasets'):
        merge_tapr_tables([source, replace(source, dataset='campus_tapr_staff_profile')], key_field='campus_number')


@pytest.mark.parametrize('problem', [None, 'missing_binding', 'wrong_sha', 'wrong_url', 'wrong_kind',
                                    'unrelated_reference', 'wrong_role', 'foreign_direct_observation'])
def test_explicit_raw_derivation_reference_preserves_original_receipt_and_rejects_unbound_events(tmp_path, problem):
    cfg, original, raw_id, path, _, _ = source_fixture(tmp_path)
    dataset = 'campus_tapr_student_profile'
    definition = cfg.data_sources[dataset].entries[2024]
    reference = definition.metadata.references[0]
    if problem == 'missing_binding':
        refs = ()
    else:
        if problem == 'wrong_sha':
            reference = replace(reference, reviewed_sha256='0' * 64)
        elif problem == 'wrong_url':
            reference = replace(reference, url='https://example.test/wrong.csv')
        elif problem == 'unrelated_reference':
            reference = replace(reference, reference_id='unrelated')
        elif problem == 'wrong_role':
            reference = replace(reference, role='data_dictionary')
        elif problem == 'wrong_kind':
            processing = original.record_processing(dataset='some_source', inputs=[raw_id], outputs={'raw': path},
                processor='tiny test', code_version='1', parameters={})
            reference = replace(reference, provenance_ids=(processing,))
        refs = (reference,)
    cfg.data_sources[dataset].entries[2024] = replace(definition, metadata=replace(definition.metadata,
        references=refs, provenance_ids=(raw_id,) if problem == 'foreign_direct_observation' else ()))
    output = tmp_path / 'output'
    before = original.read_record(raw_id)
    if problem is not None:
        with pytest.raises((ValueError, ContractError), match='acquisition|receipt|reference|different document'):
            prepare_profiles(cfg, edition='2024', family='student_profile', scopes=['campus'],
                input_dir=tmp_path, output_dir=output, reference_store=original)
        assert not output.exists() and not list(tmp_path.glob('.tapr-preparation-*'))
        return
    receipt = prepare_profiles(cfg, edition='2024', family='student_profile', scopes=['campus'],
        input_dir=tmp_path, output_dir=output, reference_store=original)
    store = ProvenanceStore(output / 'provenance')
    closure = store.verify_record_closure([receipt['scopes']['campus']['processing_record_id']])
    assert closure[raw_id] == before and closure[raw_id]['dataset'] == 'campus_tapr_student_staff_profile'
    qualified = [r for r in closure.values() if r.get('dataset') == dataset and r.get('kind') == 'acquisition']
    assert len(qualified) == 1 and qualified[0]['downloaded_at'] is None
    assert receipt['scopes']['campus']['rows'] == 6


@pytest.mark.parametrize('edition', [2024, 2025])
def test_current_family_discovery_is_metadata_only_in_a_fresh_process(edition):
    code = '''
import importlib.abc,pathlib,socket,sys,urllib.request
heavy=('teadata.engine','teadata.entities','teadata.geometry','pyarrow','duckdb','pandas','numpy','sqlalchemy','geopandas','shapely')
class Trap(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if any(fullname==n or fullname.startswith(n+'.') for n in heavy): raise AssertionError(fullname)
sys.meta_path.insert(0,Trap())
from teadata.teadata_config import load_config
import teadata.teadata_config as module
edition=int(sys.argv[1])
root=pathlib.Path(module.__file__).parent
allowed={(root/'teadata_sources.yaml').resolve()}|{(root/f'schemas/tapr--{edition}--{family}-profile-crosswalk.json').resolve() for family in ['student','staff']}
original=pathlib.Path.open
def guarded(self,*a,**k):
 if self.resolve() not in allowed: raise AssertionError('Non-metadata open: '+str(self))
 return original(self,*a,**k)
pathlib.Path.open=guarded
def forbidden(*a,**k): raise AssertionError('Data/network I/O')
socket.create_connection=forbidden
urllib.request.urlopen=forbidden
cfg=load_config()
for scope in ['campus','district']:
 for family in ['student_profile','staff_profile']:
  dataset=scope+'_tapr_'+family
  assert cfg.header_records(dataset,edition=str(edition),reader_profile='source')
  assert cfg.is_source_only(dataset,edition)
assert not any(n==p or n.startswith(p+'.') for n in sys.modules for p in heavy)
'''
    result = subprocess.run([sys.executable, '-B', '-c', code, str(edition)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
