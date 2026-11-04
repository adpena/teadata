"""Genuine local source/preparation/leaf receipts; no publisher science claims."""
from dataclasses import replace
from hashlib import sha256
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from teadata.analytical import write_leaf_artifact
from teadata.column_schema import source_schema_fingerprint
from teadata.dataset_metadata import ArtifactRef, DatasetMetadata, EditionKey, ResolvedDataset
from teadata.manifest import BuildRecorder, canonical_json, file_identity
from teadata.provenance import ProvenanceStore
from teadata.release import DataRelease, publish_release
from teadata.source_parity import validate_source_parity
from teadata.tapr import TaprReadBudget
from teadata.teadata_config import Config


def fixture_release(tmp_path, *, damage=None, schema_version=3):
    store = ProvenanceStore(tmp_path / 'history')
    dataset = 'district_tapr_student_profile'
    names = ['district_number', 'membership', 'enrollment']
    labels, codes = ['District', 'Membership', 'Enrollment'], ['DISTRICT', 'DPETALLC', 'DPNTALLC']
    raw = tmp_path / 'source.csv'
    raw.write_text('District,Membership,Enrollment\nDISTRICT,DPETALLC,DPNTALLC\n001001,0,11\n001002,-999,.\n001003,17,\n')
    source = ResolvedDataset(dataset, 'data_sources', 2025, 2025, str(raw), DatasetMetadata(version='2025'))
    acquisition = store.record_acquisition(raw, source)
    raw_sha = file_identity(raw)['sha256']
    table = {'table': 'STUD', 'input_sha256': raw_sha,
        'source_schema_sha256': source_schema_fingerprint(labels, codes), 'metric_period': {'kind': 'unknown'},
        'layout': {'label_record': 1, 'code_record': 2, 'first_data_record': 3, 'last_data_record': 5, 'annotations': []},
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
    actual = pa.table(dict(zip(names, [['001001', '001002', '001003'], ['0', '-999', '17'], ['11', '.', '']], strict=True)))
    if damage == 'prepared_wrong':
        actual = actual.set_column(1, 'membership', pa.array(['0', '-999', '18']))
    prepared = tmp_path / 'prepared.parquet'
    pq.write_table(actual, prepared, row_group_size=1)
    budget = TaprReadBudget(batch_rows=2, batch_bytes=1024)
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
        actual = actual.set_column(1, 'membership', pa.array([0, None, 17], type=pa.int64()))
        actual = actual.append_column('membership_raw', pa.array(['0', '-999', '17']))
        actual = actual.append_column('membership_status', pa.array(['observed', 'suppressed', 'observed']))
        support = {'fields': {'membership': {'raw': 'membership_raw', 'status': 'membership_status'}}, 'context': []}
    recorder = BuildRecorder(cfg, store)
    key = EditionKey(dataset, '2025', EditionKey.source_revision((raw_sha,)), 'district', 'STUD', 'one row per district_number')
    recipe = {'operation': 'unsupported_transform' if damage == 'unknown_recipe' else 'lossless_admitted_source_row_groups',
        'values_interpreted': False, 'source_read_budget': vars(budget), 'preparation_code_sha256': historical_code}
    toolchain = {'python': sys.version, 'pyarrow': pa.__version__}
    contract = recorder.leaf_contract(key, source, native_schema=actual.schema.serialize().to_pybytes(), columns=actual.schema.names,
        key_columns=['district_number'], rows=3, inputs=[raw_sha, file_identity(prepared)['sha256']],
        recipe=recipe, toolchain=toolchain, column_support=support)
    leaf_id = sha256(canonical_json(contract).encode()).hexdigest()
    leaf = tmp_path / 'leaf.parquet'
    write_leaf_artifact(leaf, actual.to_batches(max_chunksize=1), contract=contract)
    processing = store.record_processing(dataset=dataset, inputs=[preparation], outputs={'leaf': leaf},
        processor='teadata.scripts.build_tapr_profiles', code_version=recorder.processor_sha256,
        parameters={'edition': '2025', 'leaf_contract_sha256': leaf_id, 'recipe': recipe, 'toolchain': toolchain})
    physical = file_identity(leaf)
    ref = ArtifactRef(key, physical['sha256'], physical['size_bytes'], leaf.name, canonical_json(contract), processing, 'leaf')
    recorder.record_edition(key, source, artifacts=[ref], acquisition_ids=[acquisition])
    catalog = recorder.catalog(schema_version=schema_version)
    manifest = publish_release(tmp_path, catalog, schema_version=schema_version)
    return DataRelease(manifest), store, key, raw


@pytest.mark.parametrize('schema_version', [3, 4])
@pytest.mark.parametrize('damage', [None, 'reordered'])
def test_complete_raw_parity_preserves_zero_masks_leading_zeros_and_historical_code(tmp_path, schema_version, damage):
    release, store, key, _ = fixture_release(tmp_path, damage=damage, schema_version=schema_version)
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


def test_reordered_multiple_row_groups_use_one_native_scan_per_stage(tmp_path, monkeypatch):
    from collections import Counter
    import pyarrow.dataset as ds

    release, store, key, _ = fixture_release(tmp_path, damage='reordered')
    scans: Counter[str] = Counter()
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
            scans[self.path] += 1
            return self.native.iter_batches(*args, **kwargs)

    class CountedDataset:
        def __init__(self, path, *args, **kwargs):
            self.path, self.native = str(path), original_dataset(path, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.native, name)

        def scanner(self, *args, **kwargs):
            scans[self.path] += 1
            return self.native.scanner(*args, **kwargs)

    monkeypatch.setattr(pq, 'ParquetFile', CountedParquet)
    monkeypatch.setattr(ds, 'dataset', CountedDataset)
    report = validate_source_parity(release, store, edition_ids=[key.identity])
    assert report['status'] == 'PASS_RAW_PARITY'
    assert len(scans) == 2 and list(scans.values()) == [1, 1]


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
    assert report['scope'] == 'complete lossless TAPR raw-string leaves only'
    assert report['scientific_qualification'] is False
