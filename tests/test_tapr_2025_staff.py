from hashlib import sha256
from pathlib import Path

import pytest

from teadata.column_schema import header_records, header_rename_map, parse_column_schema
from teadata.manifest import ContractError
from teadata.teadata_config import load_config


ROOT = Path(__file__).resolve().parents[1]
RESOURCE = ROOT / 'docs/source-contracts/tapr--2025--staff-profile-crosswalk.json'


@pytest.mark.parametrize('scope,columns,rows,raw_sha,dictionary_sha,new_count', [
    ('campus', 106, 9084, 'ed29852dd7d7925963413b41c81a36508ccad83a4109a0aeebfa720bf90ff6ea',
     'd5a1f29d97d34946eae7705d5f8a1ef7f376f7dd143cf0ce7b2fe0f1ba1f719e', 15),
    ('district', 118, 1208, '29921c748875f91cc123c22dc5a295f1e61d1754842209bd86e56ad9f7c7abb1',
     'fead1a0ad8f3a35949dfefc8161781be4161ff7fd028e926f99c3b9c41d5a8d6', 17),
])
def test_new_crosswalk_retains_every_actual_column_without_activating_config(scope, columns, rows, raw_sha, dictionary_sha, new_count):
    document = parse_column_schema(RESOURCE.read_bytes())
    assert not (ROOT / 'teadata/schemas' / RESOURCE.name).exists()
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
    assert cfg.availability_report()['data_sources'][dataset['dataset']]['years'] == [2024]
    with pytest.raises(ContractError, match='edition unavailable'):
        cfg.header_records(dataset['dataset'], edition='2025', reader_profile='source')


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
