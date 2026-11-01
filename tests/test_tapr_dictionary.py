from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from teadata.column_schema import header_records, source_schema_fingerprint, validate_header_table
from teadata.tapr_dictionary import extract_tapr_dictionary


# Tiny public markup from the retained2025 STAF source; source rendering includes
# the publisher's literal </br> spelling even inside its quoted style attribute.
COMPLETE = '''<span class="tooltip">CPSATOFC<span class="tooltiptext" style="dispaly</br>none"><div class="dd_hdr">Name:</div><div class="dd_desc">CPSATOFC</div><div class="dd_hdr">Type:</div><div class="dd_desc">Number   </div><div class="dd_hdr">Description:</div><div class="dd_desc">Campus 2025 Staff</br> All Staff Total Full Time Equiv Count</div></span></span>'''
CODE_ONLY = '''<span class="tooltip">CPSXTOFC<span class="tooltiptext" style="dispaly</br>none">CPSXTOFC   </span></span>'''


@pytest.mark.parametrize('scope,size,digest,code,length,description', [
    ('campus', 669, '18fc077546cd208c8777b737fa19d8f517048e0ed894bf87b5cd80b37b6b9bda',
     'CAMPUS', '9', '9 Digit Campus Number'),
    ('district', 671, 'd6440307ca082e06ee48f0f33642558538ed5a4ddaae4296c279415bda8336b6',
     'DISTRICT', '6', '6 Digit County District Number'),
])
def test_exact_public_ref_spans_keep_type_and_literal_length_separate(scope, size, digest, code, length, description):
    # Exact 2024 REF slices: campus body c0dbed06... bytes51069:51738;
    # district body9430d613... bytes51082:51753. Both start at2244/46 in
    # their original bodies. The isolated fixtures have their own SHA/1,0 anchor.
    path = Path(__file__).parent / 'fixtures/tapr_dictionary' / f'2024-{scope}-ref-first-tooltip.html'
    raw = path.read_bytes()
    assert len(raw) == size and sha256(raw).hexdigest() == digest
    row, = extract_tapr_dictionary(raw, encoding='iso-8859-1')
    assert (row.code, row.publisher_type, row.publisher_length, row.publisher_description) == (
        code, 'Character', length, description)
    assert row.source_html == raw.decode('iso-8859-1')
    assert row.source_sha256 == digest and (row.source_line, row.source_column) == (1, 0)
    record = row.column_record()
    assert record['publisher_type'] == 'Character' and record['publisher_length'] == length


@pytest.mark.parametrize('length', ['2', '002', 'not stated'])
def test_publisher_length_is_literal_supplemental_metadata_without_numeric_cast(length):
    markup = COMPLETE.replace('Number   ', 'Character').replace('<div class="dd_hdr">Description:',
        f'<div class="dd_hdr">Length:</div><div class="dd_desc">{length}</div><div class="dd_hdr">Description:')
    row, = extract_tapr_dictionary(markup.encode(), encoding='utf-8')
    assert row.publisher_type == 'Character' and row.publisher_length == length
    assert row.column_record()['publisher_length'] == length
    assert row.source_html == markup


def test_duplicate_length_heading_fails_instead_of_overwriting_the_first_value():
    markup = COMPLETE.replace('<div class="dd_hdr">Description:',
        '<div class="dd_hdr">Length:</div><div class="dd_desc">2</div>'
        '<div class="dd_hdr">Length:</div><div class="dd_desc">3</div><div class="dd_hdr">Description:')
    with pytest.raises(ValueError, match='Repeated dictionary field'):
        extract_tapr_dictionary(markup.encode(), encoding='utf-8')


def test_public_markup_keeps_order_duplicates_unknowns_and_original_source_spans():
    text = '<table>\n  ' + COMPLETE + '\n' + CODE_ONLY + '\n' + COMPLETE + '</table>'
    raw = text.encode('iso-8859-1')
    rows = extract_tapr_dictionary(raw, encoding='iso-8859-1')
    assert [(r.ordinal, r.code) for r in rows] == [(1, 'CPSATOFC'), (2, 'CPSXTOFC'), (3, 'CPSATOFC')]
    assert rows[0].publisher_type == 'Number'
    assert rows[0].publisher_description == 'Campus 2025 Staff All Staff Total Full Time Equiv Count'
    assert rows[1].publisher_type is None and rows[1].publisher_description is None
    assert all(row.publisher_length is None and 'publisher_length' not in row.column_record() for row in rows)
    assert (rows[0].source_line, rows[0].source_column) == (2, 2)
    assert (rows[1].source_line, rows[1].source_column) == (3, 0)
    assert [r.source_html for r in rows] == [COMPLETE, CODE_ONLY, COMPLETE]
    assert {r.source_sha256 for r in rows} == {sha256(raw).hexdigest()}
    assert rows[1].column_record() == {'ordinal': 2, 'code': 'CPSXTOFC',
        'publisher_type': None, 'publisher_description': None,
        'source_sha256': sha256(raw).hexdigest(), 'source_line': 3, 'source_column': 0}


def test_partial_components_are_independent_and_never_filled_from_another_occurrence():
    type_only = COMPLETE.replace('<div class="dd_hdr">Description:</div><div class="dd_desc">Campus 2025 Staff</br> All Staff Total Full Time Equiv Count</div>', '')
    description_only = COMPLETE.replace('<div class="dd_hdr">Type:</div><div class="dd_desc">Number   </div>', '')
    rows = extract_tapr_dictionary((type_only + description_only).encode(), encoding='utf-8')
    assert rows[0].publisher_type == 'Number' and rows[0].publisher_description is None
    assert rows[1].publisher_type is None and rows[1].publisher_description is not None


def test_conflicting_occurrences_remain_distinct_observations_for_schema_review():
    changed = COMPLETE.replace('All Staff Total Full Time Equiv Count', 'Different publisher description')
    rows = extract_tapr_dictionary((COMPLETE + changed).encode(), encoding='utf-8')
    assert len(rows) == 2 and rows[0].code == rows[1].code
    assert rows[0].publisher_description != rows[1].publisher_description


def test_encoding_entity_text_and_crlf_anchors_do_not_replace_raw_byte_identity():
    source = ('é\r\n ' + COMPLETE.replace('All Staff', 'Staff &amp; pédagogie')).encode('iso-8859-1')
    row, = extract_tapr_dictionary(source, encoding='iso-8859-1')
    assert (row.source_line, row.source_column) == (2, 1)
    assert row.publisher_description is not None
    assert 'Staff & pédagogie' in row.publisher_description
    assert '&amp;' in row.source_html and 'pédagogie' in row.source_html
    assert row.source_sha256 == sha256(source).hexdigest()
    with pytest.raises(UnicodeDecodeError):
        extract_tapr_dictionary(source, encoding='utf-8')


@pytest.mark.parametrize('description', ['A<p>B</p>C', 'A<div>B</div>C'])
def test_supported_block_boundaries_separate_rendered_text_without_changing_source(description):
    text = COMPLETE.replace('Campus 2025 Staff</br> All Staff Total Full Time Equiv Count', description)
    row, = extract_tapr_dictionary(text.encode(), encoding='utf-8')
    assert row.publisher_description == 'A B C'
    assert row.source_html == text


@pytest.mark.parametrize('old,new', [
    ('class="tooltip"', 'class="tooltip" class="other"'),
    ('class="tooltip"', 'class="other" class="tooltip"'),
    ('class="tooltiptext"', 'class="tooltiptext" class="other"'),
    ('class="dd_hdr"', 'class="dd_hdr" class="other"'),
    ('class="dd_desc"', 'class="dd_desc" class="other"'),
])
def test_duplicate_dictionary_class_attributes_cannot_hide_an_occurrence(old, new):
    with pytest.raises(ValueError, match='Ambiguous dictionary class attributes'):
        extract_tapr_dictionary((COMPLETE + COMPLETE.replace(old, new, 1)).encode(), encoding='utf-8')


@pytest.mark.parametrize('markup,reason', [
    (COMPLETE.replace('>CPSATOFC</div>', '>OTHER</div>', 1), 'code and Name differ'),
    (COMPLETE.replace('<div class="dd_hdr">Type:</div>', '<div class="dd_hdr">Name:</div>'), 'Repeated dictionary field'),
    (COMPLETE.replace('Type:</div>', 'Units:</div>'), 'Unsupported dictionary heading'),
    (COMPLETE.replace('<div class="dd_desc">Number   </div>', ''), 'heading lacks its value'),
    (COMPLETE.replace('<div class="dd_hdr">Type:</div>', ''), 'value lacks its heading'),
    (COMPLETE[:-7], 'Truncated dictionary tooltip'),
    (CODE_ONLY.replace('CPSXTOFC   ', 'OTHER'), 'Code-only dictionary body differs'),
    (COMPLETE.replace('<div class="dd_hdr">Name:', CODE_ONLY + '<div class="dd_hdr">Name:'), 'Nested dictionary tooltip'),
    ('<html>Not a publisher dictionary</html>', 'No publisher dictionary occurrences'),
])
def test_ambiguous_or_incomplete_markup_fails_instead_of_falling_back(markup, reason):
    with pytest.raises(ValueError, match=reason):
        extract_tapr_dictionary(markup.encode(), encoding='utf-8')


def table_with(records, *, documented=False):
    field = {'position': 1, 'code': 'CPSXTOFC', 'field': 'source_fact', 'label': 'Publisher source label',
             'disposition': 'fact', 'dictionary_status': 'documented' if documented else 'not_documented',
             'publisher_type': 'Number' if documented else None,
             'publisher_description': 'Description' if documented else None}
    return {'table': 'staff_profile', 'input_sha256': 'a' * 64,
            'source_schema_sha256': source_schema_fingerprint([field['label']], [field['code']]),
            'fields': [field], 'dictionary_records': records}


def occurrence(ordinal, kind, description):
    return {'ordinal': ordinal, 'code': 'CPSXTOFC', 'publisher_type': kind,
            'publisher_description': description}


def test_same_occurrence_list_keeps_partials_without_manufacturing_a_definition():
    records = [occurrence(1, 'Number', None), occurrence(2, None, 'Description')]
    table = table_with(records)
    validate_header_table(table)
    document = {'schema_version': 2, 'mapping_version': 1,
                'datasets': [{'dataset': 'campus_tapr_staff_profile', 'edition': '2025',
                              'source_only': True, 'tables': [table]}]}
    result, = header_records(document, dataset='campus_tapr_staff_profile', edition='2025')
    assert result['dictionary_status'] == 'not_documented' and result['publisher_type'] is None
    assert result['dictionary_occurrences'] == records
    result['dictionary_occurrences'][0]['publisher_type'] = 'Caller mutation'
    assert records[0]['publisher_type'] == 'Number'
    with pytest.raises(ValueError, match='differs from its publisher dictionary'):
        validate_header_table(table_with(records, documented=True))
    records.append(occurrence(3, 'Number', 'Description'))
    validate_header_table(table_with(records, documented=True))
    with pytest.raises(ValueError, match='differs from its publisher dictionary'):
        validate_header_table(table_with(records))


@pytest.mark.parametrize('records', [
    [occurrence(1, 'Number', 'Description'), occurrence(2, 'Text', 'Description')],
    [occurrence(1, 'Number', 'Description'), occurrence(2, None, 'Conflicting')],
    [occurrence(1, 'Number', None), occurrence(2, 'Text', None)],
])
def test_complete_and_partial_known_component_conflicts_still_fail(records):
    with pytest.raises(ValueError, match='Conflicting publisher dictionary definitions'):
        validate_header_table(table_with(records, documented=True))


def test_optional_length_survives_occurrence_projection_without_filling_field_metadata():
    records = [{**occurrence(1, 'Number', 'Description'), 'publisher_length': '002'},
               {**occurrence(2, 'Number', 'Description'), 'publisher_length': '002'}]
    table = table_with(records, documented=True)
    validate_header_table(table)
    document = {'schema_version': 2, 'mapping_version': 1, 'datasets': [{
        'dataset': 'campus_tapr_staff_profile', 'edition': '2025', 'tables': [table]}]}
    result, = header_records(document, dataset='campus_tapr_staff_profile', edition='2025')
    assert 'publisher_length' not in result  # No metadata synthesis from another occurrence.
    assert [row['publisher_length'] for row in result['dictionary_occurrences']] == ['002', '002']
    result['dictionary_occurrences'][0]['publisher_length'] = 'Caller change'
    assert records[0]['publisher_length'] == '002'
    table['fields'][0]['publisher_length'] = '002'
    validate_header_table(table)
    table['fields'][0]['publisher_length'] = '2'
    with pytest.raises(ValueError, match='differs from its publisher dictionary length'):
        validate_header_table(table)


def test_partial_length_never_manufactures_a_definition_and_conflicts_remain_strict():
    records = [{**occurrence(1, 'Number', None), 'publisher_length': '2'}, occurrence(2, None, 'Description')]
    validate_header_table(table_with(records))
    with pytest.raises(ValueError, match='differs from its publisher dictionary'):
        validate_header_table(table_with(records, documented=True))
    records.append({**occurrence(3, None, None), 'publisher_length': '3'})
    with pytest.raises(ValueError, match='Conflicting publisher dictionary definitions'):
        validate_header_table(table_with(records))


@pytest.mark.parametrize('length', ['', ' ', 2, True, [], {}])
def test_length_occurrence_rejects_nonliteral_or_empty_components(length):
    record = {**occurrence(1, None, None), 'publisher_length': length}
    with pytest.raises(ValueError, match='Invalid publisher dictionary occurrence'):
        validate_header_table(table_with([record]))


def test_unknown_length_is_optional_and_does_not_change_legacy_definition_requirements():
    table = table_with([{**occurrence(1, 'Number', 'Description'), 'publisher_length': None}], documented=True)
    validate_header_table(table)


@pytest.mark.parametrize('change', [
    {'source_sha256': 'a' * 64},
    {'source_sha256': 'invalid', 'source_line': 1, 'source_column': 0},
    {'source_sha256': 'a' * 64, 'source_line': 0, 'source_column': 0},
    {'source_sha256': 'a' * 64, 'source_line': True, 'source_column': 0},
    {'source_sha256': 'a' * 64, 'source_line': 1, 'source_column': -1},
    {'source_sha256': 'a' * 64, 'source_line': 1, 'source_column': False},
    {'source_sha256': 'a' * 64, 'source_line': 1.0, 'source_column': 0},
])
def test_occurrence_anchor_is_all_or_none_with_exact_scalar_coordinate_types(change):
    record = {**occurrence(1, None, None), **change}
    with pytest.raises(ValueError, match='dictionary source anchor'):
        validate_header_table(table_with([record]))


def test_anchored_unknowns_and_legacy_complete_records_share_one_validation_owner():
    unknown, = extract_tapr_dictionary(CODE_ONLY.encode(), encoding='utf-8')
    table = table_with([unknown.column_record()])
    validate_header_table(table)
    legacy = table_with([occurrence(1, 'Number', 'Description')], documented=True)
    validate_header_table(legacy)
    invalid = deepcopy(table)
    invalid['dictionary_records'][0]['publisher_type'] = ''
    with pytest.raises(ValueError, match='Invalid publisher dictionary occurrence'):
        validate_header_table(invalid)
