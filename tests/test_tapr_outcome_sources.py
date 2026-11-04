"""Complete source-only outcome originals do not imply scientific activation."""

from copy import deepcopy
import csv
from hashlib import sha256
from pathlib import Path
import socket

import pyarrow.parquet as pq
import pytest

from teadata.column_schema import parse_column_schema
from teadata.dataset_metadata import DatasetMetadata, ResolvedDataset
from teadata.enrichment import _registry
from teadata.manifest import ContractError, canonical_json
from teadata.provenance import ProvenanceStore, validate_record_graph
from teadata.scripts.build_tapr_profiles import prepare_profiles
from teadata.scripts.rename_tapr_campus_headers import sanitize_header
from teadata.teadata_config import Config, load_config


ROOT = Path(__file__).resolve().parents[1]
UNKNOWN = {'kind': 'unknown'}
CCMR_RATE_PERIOD = {'kind': 'school_year', 'start_year': 2023, 'end_year': 2024}
COMP4_RATE_PERIOD = {'kind': 'as_of', 'as_of': '2024-08-31'}
QUALIFIED_RATES = {
    'four_year_longitudinal_rate': {
        'four_year_longitudinal_graduation_rate_all_students': 'DAGC4X24R',
        'four_year_federal_graduation_rate_without_exclusions_all_students': 'DAGC424R',
    },
    'college_career_military_readiness': {
        'ccmr_annual_graduates_all_students_rate': 'DA0GP24R',
    },
    'college_readiness_components': {
        'ccmr_annual_graduates_ap_ib_all_students_rate': 'DA0GY24R',
        'ccmr_annual_graduates_ap_ib_economically_disadvantaged_rate': 'DE0GY24R',
        'ccmr_annual_graduates_dual_credit_all_students_rate': 'DA0GUA24R',
        'ccmr_annual_graduates_dual_credit_economically_disadvantaged_rate': 'DE0GUA24R',
    },
}
SOURCES = [
    ('four_year_longitudinal_rate', 226, 224,
     '08ef046dbc4b2e3b6b0cb7705882851ec42035891ba98fe3a5ebc441e0eb13da',
     '00dea0bd885ab09780eb3a4df987bdc2252a837527639726883f9bfa3a60142e'),
    ('college_career_military_readiness', 277, 396,
     '7ae8e10c3c64ccba8723411334bf0ab069959ae40856a0388b27d123eb87cf49',
     '37e230f48bca38eb35763ce13201203ecd14a9bfc1173b45ce78d6e188147423'),
    ('college_readiness_components', 123, 165,
     '6a2df8bf0781ca7e8ed365f4d64cd9a925a73d05197b0bef7e6cf518af898735',
     '17bf30bf3b30b1b8e7d5bb165eb972f79c26bb54dfd83bfc418df9a6a7ca0c47'),
]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def reject(*args, **kwargs):
        pytest.fail('Source contract/preparation tests must not connect to a service')
    monkeypatch.setattr(socket.socket, 'connect', reject)
    monkeypatch.setattr(socket, 'create_connection', reject)


def source_table(cfg, family):
    dataset = 'district_tapr_' + family
    contract = cfg.column_contract(dataset, 2025, reader_profile='source')
    assert contract is not None
    branch, = parse_column_schema(contract['text'].encode())['datasets']
    assert branch['dataset'] == dataset and branch['edition'] == '2025'
    assert branch['source_only'] is True
    table, = branch['tables']
    return table


@pytest.mark.parametrize('family,width,occurrences,raw_sha,receipt', SOURCES)
def test_complete_outcome_discovery_stays_source_only(family, width, occurrences, raw_sha, receipt, monkeypatch):
    cfg = load_config()
    dataset = 'district_tapr_' + family
    table = source_table(cfg, family)
    metadata = cfg.resolve_dataset(dataset, 2025, reader_profile='source').metadata
    assert metadata.version == '2025' and metadata.period.to_dict() == UNKNOWN
    assert metadata.value_policy is None and metadata.estimation is None
    assert metadata.field_year is None and metadata.provenance_ids == ()
    qualified = QUALIFIED_RATES[family]
    is_comp4 = family == 'four_year_longitudinal_rate'
    period = COMP4_RATE_PERIOD if is_comp4 else CCMR_RATE_PERIOD
    reference_id = 'austin_isd_attendance_graduation_report' if is_comp4 else 'austin_isd_ccmr_report'
    fields_by_code = {field['code']: field['field'] for field in table['fields']}
    assert {name: metric.to_dict() for name, metric in metadata.metrics.items()} == {
        name: {'field': fields_by_code[code], 'period': period, 'unit': 'percent'}
        for name, code in qualified.items()
    }
    witness, = [ref for ref in metadata.references if ref.reference_id == reference_id]
    record_id, = witness.provenance_ids
    store = ProvenanceStore(ROOT / 'provenance')
    graph = validate_record_graph(store.read_record, [record_id])
    record = graph[record_id]
    assert witness.role == 'source_notes' and witness.status == 'available'
    assert record['method'] == 'download' and record['source_metadata']['version'] == '2025'
    assert witness.url and witness.url == record['requested_url']
    assert 'ccyy=2025&id=227901&lev=D' in witness.url
    assert witness.url.endswith('/attendgrad.sas' if is_comp4 else '%2Fccmr.sas')
    assert witness.reviewed_sha256 == record['artifact']['sha256']
    original, = [ref for ref in metadata.references if ref.reference_id == 'csv_headers']
    assert original.role == 'source_notes' and original.provenance_ids == (receipt,)
    assert original.reviewed_sha256 == table['input_sha256'] == raw_sha
    assert len(table['fields']) == width and len(table['dictionary_records']) == occurrences
    assert table['layout'] == {'label_record': 1, 'code_record': 2, 'first_data_record': 3,
                               'last_data_record': 1210, 'annotations': []}
    assert [row['ordinal'] for row in table['dictionary_records']] == list(range(1, occurrences + 1))
    assert len({row['field'] for row in table['fields']}) == width
    assert len({row['code'] for row in table['fields']}) == width
    records = cfg.header_records(dataset, edition='2025', reader_profile='source')
    assert len(records) == width
    assert {row['code']: row['period'] for row in records if row['period'] != UNKNOWN} == {
        code: period for code in qualified.values()
    }
    for record in records:
        assert record['dictionary_occurrences'] == [row for row in table['dictionary_records']
                                                   if row['code'] == record['code']]
    assert cfg.is_source_only(dataset, 2025) and cfg.column_schema(dataset, 2025) is None
    assert dataset not in _registry
    # Default loading must refuse the opt-in before opening any publisher CSV.
    original_open = Path.open
    def no_source_open(path, *args, **kwargs):
        if path.suffix == '.csv':
            pytest.fail('Default reader attempted to open a source-only CSV')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', no_source_open)
    with pytest.raises(ContractError, match='Reader profile is required'):
        cfg.load_df(dataset, 2025)


def test_literal_class_and_annual_graduate_contexts_do_not_become_other_populations():
    cfg = load_config()
    comp = cfg.header_records('district_tapr_four_year_longitudinal_rate', edition='2025', reader_profile='source')
    by_code = {row['code']: row for row in comp}
    ordinary, federal = by_code['DAGC4X24R'], by_code['DAGC424R']
    assert ordinary['label'] == federal['label'] == 'Longitudinal, 4-Year, Class of 2024, Graduates, All Students, Rate'
    assert ordinary['field'] != federal['field']
    assert ordinary['dictionary_occurrences'][0]['publisher_context']['heading'] == '4-Year Longitudinal Rate (Gr 9-12)'
    federal_heading = '4-Year Federal Graduation Rate Without Exclusions (Gr 9-12)'
    assert federal['dictionary_occurrences'][0]['publisher_context']['heading'] == federal_heading
    for row in comp:
        if row['disposition'] == 'identity':
            continue
        expected = sanitize_header(row['label']).replace('longitudinal_4_year_class_of_2024_',
                                                         'district_2024_4_year_longitudinal_', 1)
        if row['dictionary_occurrences'][0]['publisher_context']['heading'] == federal_heading:
            expected = expected.replace('district_2024_4_year_longitudinal_',
                                        'district_2024_' + sanitize_header(federal_heading) + '_', 1)
        assert row['field'] == expected
    ccmr = {row['code']: row for row in cfg.header_records('district_tapr_college_career_military_readiness',
                                                        edition='2025', reader_profile='source')}
    denominator = ccmr['DA0GP24D']
    assert len(denominator['dictionary_occurrences']) == 12
    assert denominator['label'] == 'District 2024 College, Career, & Military Ready Graduates: All Students Denominator'
    assert ccmr['DA0GP24R']['dictionary_occurrences'][0]['publisher_context']['heading'] == 'College, Career, and Military Ready Graduates (Student Achievement)'
    components = {row['code']: row for row in cfg.header_records('district_tapr_college_readiness_components',
                                                              edition='2025', reader_profile='source')}
    assert components['DA0GY24R']['label'] == 'District 2024 AP/IB Met Criteria in Any Subject Graduates: All Students Rate'
    assert components['DA0GUA24R']['label'] == 'District 2024 Dual Credit Graduates: 3 Hrs ELA or Math, or 9 Hrs Any Subject: All Students Rate'
    assert len(components['DA0GP24D']['dictionary_occurrences']) == 5
    for family in ('college_career_military_readiness', 'college_readiness_components'):
        assert all(row['field'] == sanitize_header(row['label']) for row in source_table(cfg, family)['fields']
                   if row['disposition'] == 'fact')
        metadata = cfg.resolve_dataset('district_tapr_' + family, 2025, reader_profile='source').metadata
        assert metadata.notes is not None
        assert 'Annual-graduate' in metadata.notes and 'Bonus' in metadata.notes and 'ERC' in metadata.notes


@pytest.mark.parametrize('family,width,occurrences,raw_sha,receipt', SOURCES)
def test_ordinary_native_preparation_preserves_every_raw_field_and_original_receipt(tmp_path, family, width, occurrences, raw_sha, receipt):
    dataset = 'district_tapr_' + family
    mapping = deepcopy(source_table(load_config(), family))
    row = ['-3'] * width
    by_code = {field['code']: field for field in mapping['fields']}
    row[by_code['DISTRICT']['position'] - 1] = '001902'
    row[by_code['DISTNAME']['position'] - 1] = 'Literal district'
    tokens = ['', '0', '-1', '-2', '<2,790', '9007199254740993.0', '0012', 'N/A', '.']
    facts = [field for field in mapping['fields'] if field['disposition'] == 'fact']
    for field, token in zip(facts, tokens, strict=False):
        row[field['position'] - 1] = token
    source = tmp_path / 'input.csv'
    with source.open('w', newline='') as stream:
        csv.writer(stream).writerows([[field['label'] for field in mapping['fields']],
                                     [field['code'] for field in mapping['fields']], row])
    mapping['input_sha256'] = sha256(source.read_bytes()).hexdigest()
    mapping['layout']['last_data_record'] = 3
    schema = tmp_path / 'schema.json'
    schema.write_text(canonical_json({'schema_version': 2, 'mapping_version': 1, 'datasets': [
        {'dataset': dataset, 'edition': '2025', 'source_only': True, 'tables': [mapping]}]}) + '\n')
    original_store = ProvenanceStore(tmp_path / 'original')
    # The original observation is not relabelled as the later canonical family.
    original_id = original_store.record_acquisition(source, ResolvedDataset('reference:original_selector',
        'reference', 2025, 2025, str(source), DatasetMetadata(version='2025')))
    cfg = Config.from_dict({'data_sources': {dataset: {2025: {'path': str(source),
        'metadata': {'version': '2025', 'period': UNKNOWN, 'references': [
            {'reference_id': 'csv_headers', 'role': 'source_notes', 'path': source.name,
             'status': 'available', 'reviewed_sha256': mapping['input_sha256'], 'provenance_ids': [original_id]},
            {'reference_id': 'columns', 'role': 'schema', 'path': schema.name,
             'status': 'available', 'reviewed_sha256': sha256(schema.read_bytes()).hexdigest()}]},
        'reader': {'profiles': {'source': {'column_schema': schema.name,
            'tapr_source': {'scope': 'district', 'tables': {family: str(source)}}}}}}}}})
    cfg.base_dir = tmp_path
    output = tmp_path / 'prepared'
    result = prepare_profiles(cfg, edition='2025', family=family, scopes=['district'], input_dir=tmp_path,
                              output_dir=output, reference_store=original_store)
    prepared = result['scopes']['district']
    assert prepared['rows'] == 1 and prepared['fields'] == width and prepared['qualified_metrics'] == {}
    assert pq.read_table(output / 'district.parquet').to_pylist() == [
        dict(zip([field['field'] for field in mapping['fields']], row, strict=True))]
    store = ProvenanceStore(output / 'provenance')
    closure = store.verify_record_closure([prepared['processing_record_id']])
    assert closure[original_id] == original_store.read_record(original_id)
    assert closure[original_id]['dataset'] == 'reference:original_selector'
