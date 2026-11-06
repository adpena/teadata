"""Genuine local source/preparation/leaf receipts; no publisher science claims."""
import csv
from dataclasses import replace
from decimal import Decimal, Inexact, localcontext
from hashlib import sha256
from io import BytesIO, StringIO
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from teadata.analytical import write_leaf_artifact
from teadata.column_schema import source_schema_fingerprint
from teadata.dataset_metadata import ArtifactRef, DatasetMetadata, EditionKey, ResolvedDataset
from teadata.manifest import BuildRecorder, ContractError, canonical_json, file_identity
from teadata.provenance import ProvenanceStore
from teadata.release import DataRelease, publish_release
from teadata.source_parity import validate_source_parity
from teadata.tapr import TaprReadBudget
from teadata.teadata_config import Config


def fixture_release(tmp_path, *, damage=None, schema_version=3, batch_bytes=1024, scope='district',
                    annotation=False, alias=False, last_enrollment=''):
    store = ProvenanceStore(tmp_path / 'history')
    dataset = f'{scope}_tapr_student_profile'
    names = [f'{scope}_number', 'membership', 'enrollment']
    prefix = scope[0].upper()
    labels, codes = [scope.title(), 'Membership', 'Enrollment'], [scope.upper(), prefix + 'PETALLC', prefix + 'PNTALLC']
    keys = ['01', '13', '20'] if scope == 'region' else ['001001', '001002', '001003']
    if alias:
        keys[1] = "'1002.0"
    membership = ['9007199254740993' if damage == 'normalized' else '0', '-999', '17']
    enrollment = ['11', '.', last_enrollment]
    raw = tmp_path / 'source.csv'
    data = [f'{key},{count},{other}' for key, count, other in zip(keys, membership, enrollment, strict=True)]
    notes = []
    if annotation:
        note = 'Source note'
        data.insert(1, note)
        notes.append({'kind': 'note', 'first_record': 4, 'last_record': 4, 'sha256': sha256((note + '\n').encode()).hexdigest()})
    raw.write_text('\n'.join([','.join(labels), ','.join(codes), *data, '']))
    source = ResolvedDataset(dataset, 'data_sources', 2025, 2025, str(raw), DatasetMetadata(version='2025'))
    acquisition = store.record_acquisition(raw, source)
    raw_sha = file_identity(raw)['sha256']
    table = {'table': 'STUD', 'input_sha256': raw_sha,
        'source_schema_sha256': source_schema_fingerprint(labels, codes), 'metric_period': {'kind': 'unknown'},
        'layout': {'label_record': 1, 'code_record': 2, 'first_data_record': 3, 'last_data_record': 2 + len(data), 'annotations': notes},
        'fields': [{'position': i, 'code': code, 'label': label, 'field': name,
            'disposition': 'identity' if i == 1 else 'fact', 'publisher_type': None, 'publisher_description': None,
            'dictionary_status': 'not_applicable' if i == 1 else 'not_documented'}
            for i, (code, label, name) in enumerate(zip(codes, labels, names, strict=True), 1)]}
    schema = tmp_path / 'crosswalk.json'
    schema.write_text(canonical_json({'schema_version': 2, 'mapping_version': 1,
        'datasets': [{'dataset': dataset, 'edition': '2025', 'source_only': True, 'tables': [table]}]}) + '\n')
    schema_sha = file_identity(schema)['sha256']
    schema_observation = store.record_acquisition(schema, replace(source, dataset='reference:local_crosswalk', path=str(schema)))
    cfg = Config.from_dict({'data_sources': {dataset: {2025: {'path': str(raw),
        'reader': {'profiles': {'source': {'column_schema': schema.name}}},
        'metadata': {'version': '2025', 'provenance_ids': [acquisition], 'references': [
            {'reference_id': 'schema', 'role': 'schema', 'path': schema.name, 'status': 'available',
             'reviewed_sha256': schema_sha, 'provenance_ids': [schema_observation]}]}}}}})
    cfg.base_dir = tmp_path
    source = cfg.resolve_dataset(dataset, 2025, reader_profile='source')
    actual = pa.table(dict(zip(names, [keys, membership, enrollment], strict=True)))
    if damage == 'prepared_wrong':
        actual = actual.set_column(1, 'membership', pa.array(['0', '-999', '18']))
    prepared = tmp_path / 'prepared.parquet'
    pq.write_table(actual, prepared, row_group_size=1)
    budget = TaprReadBudget(batch_rows=2, batch_bytes=batch_bytes)
    historical_code = '1' * 64  # Recorded historical identity; intentionally differs from current code.
    preparation = store.record_processing(dataset=dataset, inputs=[acquisition, schema_observation],
        outputs={'prepared_source': prepared, 'crosswalk': schema}, processor='teadata.scripts.build_tapr_profiles',
        code_version=historical_code, parameters={'edition': '2025', 'column_schema_sha256': schema_sha,
            'preparation_code_sha256': historical_code, 'source_read_budget': vars(budget)},
        validation={'rows': 3, 'fields': 3, 'source_columns_accounted': 3,
                    'raw_string_readback_equal': True, 'all_historical_source_rows_retained': True})
    if damage == 'reordered':
        actual = actual.take(pa.array([2, 0, 1]))
    elif damage == 'wrong_cell':
        actual = actual.set_column(1, 'membership', pa.array(['0', '-999', '18']))
    elif damage == 'mask_changed':
        actual = actual.set_column(1, 'membership', pa.array(['0', '.', '17']))
    elif damage == 'swapped':
        actual = actual.set_column(1, 'membership', actual['enrollment']).set_column(2, 'enrollment', actual['membership'])
    elif damage == 'missing_key':
        actual = actual.set_column(0, 'district_number', pa.array(['001001', '001002', '001004']))
    elif damage == 'duplicate_key':
        actual = actual.set_column(0, 'district_number', pa.array(['001001', '001001', '001003']))
    elif damage == 'key_lexeme':
        actual = actual.set_column(0, 'district_number', pa.array(['1001', '001002', '001003']))
    elif damage == 'null_fact':
        actual = actual.set_column(2, 'enrollment', pa.array(['11', '.', None], type=pa.string()))
    support = None
    if damage == 'normalized':
        actual = actual.set_column(1, 'membership', pa.array([int(membership[0]), None, 17], type=pa.int64()))
        actual = actual.append_column('membership_raw', pa.array(membership))
        actual = actual.append_column('membership_status', pa.array(['observed', 'suppressed', 'observed']))
        support = {'fields': {'membership': {'raw': 'membership_raw', 'status': 'membership_status'}}, 'context': []}
    recorder = BuildRecorder(cfg, store)
    key = EditionKey(dataset, '2025', EditionKey.source_revision((raw_sha,)), scope, 'STUD', f'one row per {names[0]}')
    recipe = {'operation': 'unsupported_transform' if damage == 'unknown_recipe' else 'lossless_admitted_source_row_groups',
        'values_interpreted': False, 'source_read_budget': vars(budget), 'preparation_code_sha256': historical_code}
    toolchain = {'python': sys.version, 'pyarrow': pa.__version__}
    contract = recorder.leaf_contract(key, source, native_schema=actual.schema.serialize().to_pybytes(), columns=actual.schema.names,
        key_columns=names[:1], rows=3, inputs=[raw_sha, file_identity(prepared)['sha256']],
        recipe=recipe, toolchain=toolchain, column_support=support)
    leaf_id = sha256(canonical_json(contract).encode()).hexdigest()
    leaf = tmp_path / 'leaf.parquet'
    write_leaf_artifact(leaf, actual.to_batches(max_chunksize=1), contract=contract)
    catalog_acquisition = acquisition
    processing_inputs = [preparation]
    if damage in {'detached_acquisition', 'sibling_acquisition'}:
        catalog_acquisition = store.record_acquisition(raw, source, parameters={'observation': 'another source observation'})
        if damage == 'sibling_acquisition':
            processing_inputs.append(catalog_acquisition)
    processing = store.record_processing(dataset=dataset, inputs=processing_inputs, outputs={'leaf': leaf},
        processor='teadata.scripts.build_tapr_profiles', code_version=recorder.processor_sha256,
        parameters={'edition': '2025', 'leaf_contract_sha256': leaf_id, 'recipe': recipe, 'toolchain': toolchain})
    physical = file_identity(leaf)
    ref = ArtifactRef(key, physical['sha256'], physical['size_bytes'], leaf.name, canonical_json(contract), processing, 'leaf')
    recorder.record_edition(key, source, artifacts=[ref], acquisition_ids=[catalog_acquisition])
    catalog = recorder.catalog(schema_version=schema_version)
    manifest = publish_release(tmp_path, catalog, schema_version=schema_version)
    return DataRelease(manifest), store, key, raw


@pytest.mark.parametrize('restore', [False, True])
def test_native_csv_export_matches_every_original_cell(tmp_path, restore):
    release, _, key, raw = fixture_release(tmp_path, schema_version=4)
    destination = BytesIO()
    release.to_csv(key, destination, restore_original_headers=restore)
    with raw.open(newline='') as source:
        expected = list(csv.reader(source))
    if not restore:
        expected[:2] = [['district_number', 'membership', 'enrollment']]
    assert list(csv.reader(StringIO(destination.getvalue().decode(), newline=''))) == expected


@pytest.mark.parametrize('restore', [False, True])
def test_excel_and_dataframe_export_share_reviewed_headers(tmp_path, restore):
    from openpyxl import load_workbook

    release, _, key, raw = fixture_release(tmp_path, schema_version=4, alias=True, last_enrollment='=1+1')
    workbook_path = tmp_path / 'table.xlsx'
    release.to_excel(key, workbook_path, restore_original_headers=restore)
    frame = release.to_df(key, restore_original_headers=restore)
    with raw.open(newline='') as source:
        expected = list(csv.reader(source))
    labels, codes, *data = expected
    assert frame.columns.to_list() == (list(zip(labels, codes, strict=True)) if restore else
                                       ['district_number', 'membership', 'enrollment'])
    assert frame.to_numpy().tolist() == data
    book = load_workbook(workbook_path, read_only=True)
    try:
        rows = list(book['Data'].iter_rows())
        headers = [labels, codes] if restore else [frame.columns.to_list()]
        assert [[cell.value for cell in row] for row in rows] == headers + data
        assert rows[-1][-1].data_type == 's'
    finally:
        book.close()


def test_typed_exports_preserve_nullable_integers(tmp_path):
    import pandas as pd
    from openpyxl import load_workbook

    release, _, key, _ = fixture_release(tmp_path, damage='normalized')
    frame = release.to_df(key)
    assert frame['membership'].iloc[0] == 9007199254740993
    assert pd.isna(frame['membership'].iloc[1])
    stream = BytesIO()
    release.to_excel(key, stream)
    book = load_workbook(stream, read_only=True)
    try:
        assert book['Data']['B2'].value == '9007199254740993'
        assert book['Data']['B3'].value is None
    finally:
        book.close()


def test_lazy_selection_reads_one_leaf_with_filtered_columns(tmp_path, monkeypatch):
    release, _, key, _ = fixture_release(tmp_path, damage='normalized', schema_version=4)
    native = release.analytical_dataset
    scans = []

    class Traced:
        def __init__(self, dataset):
            self.dataset = dataset

        def scanner(self, **kwargs):
            scans.append(kwargs)
            return self.dataset.scanner(**kwargs)

    monkeypatch.setattr(release, 'analytical_dataset', lambda name: Traced(native(name)))
    table = release.table(key)
    count = table['membership']
    selected = table[(count > 1) | count.is_null()][['membership']].where(table['enrollment'] != '11')
    assert scans == []
    result = selected.to_arrow()
    assert len(scans) == 1
    assert scans[0]['columns'] == ['district_number', 'membership', 'membership_raw', 'membership_status']
    assert result.to_pylist() == [
        {'district_number': '001002', 'membership': None, 'membership_raw': '-999', 'membership_status': 'suppressed'},
        {'district_number': '001003', 'membership': 17, 'membership_raw': '17', 'membership_status': 'observed'},
    ]
    with pytest.raises(ContractError, match='field unavailable'):
        table.select('unknown').to_arrow()
    with pytest.raises(pa.ArrowInvalid, match='unknown'):
        table[table['unknown'] == 1].to_arrow()


def test_source_query_string_uses_the_same_native_filter(tmp_path):
    release, _, key, _ = fixture_release(tmp_path, damage='normalized', schema_version=4)
    table = release.table(key)
    count = table['membership']
    inline = table[((count >= 10) & (count < 20)) | count.is_null()].select('membership')
    query = table.query('@low <= membership < @high or membership is None',
                        params={'low': 10, 'high': 20}).select('membership')
    assert query.to_arrow().equals(inline.to_arrow())
    assert table.query("district_number in @ids and not enrollment == '@literal'",
                       params={'ids': ['001001', '001002']}).select('membership').to_arrow().num_rows == 2


def test_query_decimals_bind_to_the_physical_column_type():
    from pyarrow import dataset

    from teadata.source_predicates import prepare_predicate

    values = [Decimal('0.34'), Decimal('0.35'), Decimal('0.36')]
    native = dataset.dataset(pa.table({'fte': pa.array(values, type=pa.decimal128(16, 2)),
                                       'rate': [float(value) for value in values], 'count': [1, 2, 3]}))

    def select(expression, params=None):
        return native.to_table(filter=prepare_predicate(expression, params)(native.schema))['fte'].to_pylist()

    assert select('fte == 0.35') == select('rate == 0.35') == [values[1]]
    assert select('fte > 0.35') == [values[2]]
    assert select('fte in [0.34, 0.36]') == [values[0], values[2]]
    with localcontext() as context:
        context.traps[Inexact] = True
        assert select('fte in [1, .35, 0.351]') == [values[1]]
    assert select('fte > -(.35)') == values
    assert select('count in [1.0, 2.1]') == [values[0]]
    assert select('count > 1e-100') == values
    assert select('fte == @value', {'value': values[1]}) == [values[1]]
    with pytest.raises(ValueError, match='Decimal parameter'):
        select('fte == @value', {'value': 0.35})
    with pytest.raises(ValueError, match='explicit cast'):
        select('fte == rate')


def test_query_exact_thresholds_preserve_nulls_and_native_numeric_ranges():
    from operator import eq, ge, gt, le, lt, ne

    from pyarrow import dataset

    from teadata.source_predicates import prepare_predicate

    precise_int = 2**53 + 1
    native = dataset.dataset(pa.table({
        'fte': pa.array([Decimal('-.35'), Decimal('0'), Decimal('.35'), None], type=pa.decimal128(16, 2)),
        'signed': pa.array([-(2**63), -1, precise_int, None], type=pa.int64()),
        'unsigned': pa.array([0, 2**64 - 1, precise_int, None], type=pa.uint64()),
    }))

    def select(expression, column, params=None):
        predicate = prepare_predicate(expression, params)(native.schema)
        return native.to_table(filter=predicate)[column].to_pylist()

    for spelling, operation in [('==', eq), ('!=', ne), ('<', lt), ('<=', le), ('>', gt), ('>=', ge)]:
        threshold = Decimal('-.349')
        values = [Decimal('-.35'), Decimal('0'), Decimal('.35')]
        with localcontext() as context:
            context.traps[Inexact] = True
            assert select(f'fte {spelling} -.349', 'fte') == [v for v in values if operation(v, threshold)]
            assert select(f'-.349 {spelling} fte', 'fte') == [v for v in values if operation(threshold, v)]
    assert select('1e-75 < fte < 1e37', 'fte') == [Decimal('.35')]
    assert select('fte == .35' + '0' * 61, 'fte') == [Decimal('.35')]
    assert select('fte < 10000000000000000000', 'fte') == values
    assert select('not (fte == 1e37)', 'fte') == values
    assert select('fte not in [1e37, .351]', 'fte') == [*values, None]
    assert select('signed == 9007199254740993.0', 'signed') == [precise_int]
    assert select('signed in [9007199254740993.0]', 'signed') == [precise_int]
    assert select('signed == @value', 'signed', {'value': float(precise_int)}) == []
    assert select('1e-1000 < signed < 1e309', 'signed') == [precise_int]
    assert select('signed < -9223372036854775807.5', 'signed') == [-(2**63)]
    assert select('18446744073709551615.0 <= unsigned', 'unsigned') == [2**64 - 1]
    assert select('unsigned in [-1, 18446744073709551615.0]', 'unsigned') == [2**64 - 1]
    with pytest.raises(ValueError, match='numeric values'):
        select('signed in [1, True]', 'signed')


@pytest.mark.parametrize('expression', [
    "__import__('os').system('echo unsafe')", 'membership.__class__ == 1',
    'membership + 1 == 2', 'membership > @unknown', 'membership > None',
    'membership in [x for x in range(10)]', 'membership == @ value',
])
def test_source_query_rejects_unsupported_syntax_before_leaf_io(tmp_path, monkeypatch, expression):
    release, _, key, _ = fixture_release(tmp_path)
    monkeypatch.setattr(release, 'artifact', lambda *args, **kwargs: pytest.fail('Unexpected data read'))
    with pytest.raises(ValueError):
        release.table(key).query(expression)


def test_source_query_parameters_cannot_inject_expressions_or_bypass_budget(tmp_path):
    release, _, key, _ = fixture_release(tmp_path)
    table = release.table(key)
    with pytest.raises(ValueError, match='bounded scalar'):
        table.query('membership == @value', params={'value': table['membership']})
    with pytest.raises(ValueError, match='parameter uses exceed'):
        table.query('enrollment == @text or membership == @text', params={'text': 'a' * 5000})


def test_filtered_formats_share_original_headers_and_values(tmp_path):
    from openpyxl import load_workbook

    release, _, key, _ = fixture_release(tmp_path, alias=True, schema_version=4)
    table = release.table(key)
    selected = table[table['membership'].isin(['-999', '17'])].select('membership')
    frame = selected.to_df(restore_original_headers=True)
    expected = [['District', 'Membership'], ['DISTRICT', 'DPETALLC'], ["'1002.0", '-999'], ['001003', '17']]
    assert frame.columns.tolist() == list(zip(*expected[:2], strict=True))
    assert frame.to_numpy().tolist() == expected[2:]
    assert frame.attrs['teadata']['manifest_sha256'] == release.manifest_sha256
    assert frame.attrs['teadata']['edition'] == key.to_dict()
    csv_buffer, xlsx_buffer = BytesIO(), BytesIO()
    selected.to_csv(csv_buffer, restore_original_headers=True)
    selected.to_excel(xlsx_buffer, restore_original_headers=True)
    assert list(csv.reader(StringIO(csv_buffer.getvalue().decode(), newline=''))) == expected
    book = load_workbook(xlsx_buffer, read_only=True)
    try:
        assert [list(row) for row in book['Data'].values] == expected
    finally:
        book.close()


def test_export_preserves_existing_paths_and_cleans_up_after_failure(tmp_path):
    from openpyxl.worksheet._writer import ALL_TEMP_FILES

    release, _, key, raw = fixture_release(tmp_path, last_enrollment='x' * 32_768)
    original = raw.read_bytes()
    for export in (release.to_csv, release.to_excel):
        with pytest.raises(FileExistsError):
            export(key, raw)
        assert raw.read_bytes() == original
    existing_scratch = set(ALL_TEMP_FILES)
    failed = tmp_path / 'failed.xlsx'
    with pytest.raises(ValueError, match='Excel cell limits'):
        release.to_excel(key, failed)
    assert not failed.exists()
    assert set(ALL_TEMP_FILES) == existing_scratch
    assert not list(tmp_path.glob('.teadata-export-*'))


@pytest.mark.parametrize('value', ['left\r\nright', 'left\uffffright', 'left_x005F_x000D_right'])
def test_xlsx_refuses_text_it_cannot_preserve_but_csv_keeps_it(tmp_path, value):
    release, _, key, _ = fixture_release(tmp_path, last_enrollment=value)
    csv_buffer = BytesIO()
    release.to_csv(key, csv_buffer, restore_original_headers=True)
    assert list(csv.reader(StringIO(csv_buffer.getvalue().decode(), newline='')))[-1][-1] == value
    path = tmp_path / 'unsafe.xlsx'
    with pytest.raises(ValueError, match='cannot preserve. Use CSV'):
        release.to_excel(key, path, restore_original_headers=True)
    assert not path.exists()


def test_source_projection_keeps_raw_alias_and_resolves_exact_edition(tmp_path):
    from uuid import uuid4

    from teadata.entities import District

    release, _, key, _ = fixture_release(tmp_path, schema_version=4, alias=True)
    sources = release.for_entity(District(uuid4(), 'Fixture', district_number='001002'), edition='2025')
    assert sources.datasets() == (key,)
    result = sources.dataset(key).select('membership')
    assert result.to_pylist() == [{'district_number': "'1002.0", 'membership': '-999'}]


@pytest.mark.parametrize(('damage', 'annotation'), [(None, False), (None, True), ('reordered', False)])
def test_optional_export_golden_checks_original_order(tmp_path, damage, annotation):
    release, store, key, _ = fixture_release(tmp_path, damage=damage, schema_version=4, annotation=annotation)
    checked = validate_source_parity(release, store, edition_ids=[key.identity], verify_export=True)['editions'][0]
    if damage is None:
        assert checked['csv_export'] == {'status': 'PASS_EXACT_CSV_EXPORT', 'rows': 3, 'cells': 9, 'header_records': 2}
    else:
        assert checked['status'] == 'FAIL_MISMATCH'
        assert checked['raw_parity_status'] == 'PASS_RAW_PARITY'
        assert checked['raw_cells'] == 9
        assert checked['csv_export'] == {'status': 'FAIL_MISMATCH', 'diagnostic': 'CSV export differs at original logical record 3'}


@pytest.mark.parametrize('schema_version', [3, 4])
@pytest.mark.parametrize('damage', [None, 'reordered'])
@pytest.mark.parametrize('scope', ['district', 'region'])
def test_complete_raw_parity_preserves_zero_masks_leading_zeros_and_historical_code(tmp_path, schema_version, damage, scope):
    release, store, key, _ = fixture_release(tmp_path, damage=damage, schema_version=schema_version, scope=scope)
    report = validate_source_parity(release, store, edition_ids=[key.identity])
    assert report['status'] == 'PASS_RAW_PARITY'
    assert report['coverage'] == {'requested_editions': 1, 'checked_editions': 1, 'uncovered_editions': 0,
                                  'failed_editions': 0, 'raw_cells': 9, 'compared_cells': 18}
    assert report['scientific_qualification'] is False
    assert report['publisher_authenticity_verified'] is False
    assert report['editions'][0]['mismatches'] == 0


@pytest.mark.parametrize('damage', ['wrong_cell', 'prepared_wrong', 'swapped', 'mask_changed', 'missing_key', 'duplicate_key', 'key_lexeme', 'null_fact'])
def test_internally_valid_but_source_wrong_leaf_blocks_without_exception(tmp_path, damage):
    release, store, key, _ = fixture_release(tmp_path, damage=damage)
    report = validate_source_parity(release, store, edition_ids=[key.identity])
    assert report['status'] == 'HOLD'
    assert report['editions'][0]['status'] == 'FAIL_MISMATCH'
    assert report['coverage']['failed_editions'] == 1
    assert report['exceptions_supported'] is False
    if damage == 'prepared_wrong':
        assert all(report['editions'][0]['stages'][stage]['membership']['mismatches'] == 1
                   for stage in ('prepared', 'artifact'))
    if damage == 'key_lexeme':
        assert report['editions'][0]['stages']['artifact']['district_number']['mismatches'] == 1
        assert report['editions'][0]['keysets']['artifact'] == {'raw_only': 0, 'target_only': 0}
    if damage == 'missing_key':
        assert report['editions'][0]['keysets']['artifact'] == {'raw_only': 1, 'target_only': 1}


@pytest.mark.parametrize('problem', ['unknown_recipe', 'normalized', 'absent_edition', 'absent_raw'])
def test_unsupported_or_unavailable_lineage_is_uncovered_never_pass(tmp_path, problem):
    release, store, key, raw = fixture_release(tmp_path, damage=problem if problem in {'unknown_recipe', 'normalized'} else None)
    if problem == 'absent_raw':
        store.verify_artifact(file_identity(raw)['sha256']).unlink()
    report = validate_source_parity(release, store, edition_ids=['f' * 64 if problem == 'absent_edition' else key.identity])
    assert report['status'] == 'HOLD'
    assert report['editions'][0]['status'] == 'HOLD_UNCOVERED'
    assert report['coverage']['checked_editions'] == 0


@pytest.mark.parametrize('damage', ['detached_acquisition', 'sibling_acquisition'])
def test_same_bytes_from_another_acquisition_do_not_establish_preparation_lineage(tmp_path, damage):
    release, store, key, _ = fixture_release(tmp_path, damage=damage)
    report = validate_source_parity(release, store, edition_ids=[key.identity])
    assert report['status'] == 'HOLD'
    assert report['editions'][0]['status'] == 'FAIL_MISMATCH'
    assert 'acquisition is outside the preparation closure' in report['editions'][0]['diagnostic']


def test_explicit_unique_scope_required(tmp_path):
    release, store, key, _ = fixture_release(tmp_path)
    for identifiers in [[], [key.identity, key.identity]]:
        with pytest.raises(ValueError, match='Explicit unique'):
            validate_source_parity(release, store, edition_ids=identifiers)


def test_small_working_budget_is_explicit_uncovered_without_large_allocations(tmp_path):
    release, store, key, _ = fixture_release(tmp_path)
    report = validate_source_parity(release, store, edition_ids=[key.identity], working_bytes=64)
    assert report['status'] == 'HOLD'
    assert report['working_bytes'] == 64
    assert report['editions'][0]['status'] == 'HOLD_UNCOVERED'
    assert 'working-byte' in report['editions'][0]['diagnostic']


def test_reordered_row_groups_are_each_read_once_within_the_byte_budget(tmp_path, monkeypatch):
    from collections import defaultdict
    import pyarrow.dataset as ds

    # Each one-row group fits, but two groups combined into a two-row batch do not.
    release, store, key, _ = fixture_release(tmp_path, damage='reordered', batch_bytes=40)
    scans: dict[str, list[list[int]]] = defaultdict(list)
    original_parquet, original_dataset = pq.ParquetFile, ds.dataset

    class CountedParquet:
        def __init__(self, path, *args, **kwargs):
            self.path, self.native = str(path), original_parquet(path, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.native, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.native.close()

        def iter_batches(self, *args, **kwargs):
            scans[self.path].append(kwargs['row_groups'])
            return self.native.iter_batches(*args, **kwargs)

    class CountedDataset:
        def __init__(self, path, *args, **kwargs):
            self.path, self.native = str(path), original_dataset(path, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.native, name)

        def scanner(self, *args, **kwargs):
            pytest.fail('Raw TAPR parity must not rescan with a dataset predicate')

    monkeypatch.setattr(pq, 'ParquetFile', CountedParquet)
    monkeypatch.setattr(ds, 'dataset', CountedDataset)
    report = validate_source_parity(release, store, edition_ids=[key.identity])
    assert report['status'] == 'PASS_RAW_PARITY', report['editions'][0].get('diagnostic')
    assert len(scans) == 2 and all(groups == [[0], [1], [2]] for groups in scans.values())


def test_legacy_schema2_is_explicit_uncovered_with_requested_scope(catalog, tmp_path):
    from types import SimpleNamespace

    report = validate_source_parity(SimpleNamespace(catalog=catalog), ProvenanceStore(tmp_path / 'history'),
                                    edition_ids=['a' * 64])
    assert report['status'] == 'HOLD'
    assert report['coverage']['uncovered_editions'] == 1
    assert report['editions'][0]['edition_id'] == 'a' * 64


def test_source_parity_cli_is_explicit_offline_and_reports_its_limited_scope(tmp_path, monkeypatch, capsys):
    import json
    from teadata import verification

    _, store, key, _ = fixture_release(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['verification', '--kind', 'library', '--source-parity',
        str(tmp_path / 'release.json'), '--source-store', str(store.root), '--source-edition', key.identity])
    verification.main()
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'PASS_RAW_PARITY'
    assert report['scope'] == 'admitted lossless TAPR, typed PEIMS and typed district transfer leaves'
    assert report['scientific_qualification'] is False
