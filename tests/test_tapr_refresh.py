from argparse import Namespace
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest

from teadata.acquisition import AcquisitionClient, NoRedirect, acquire_form, replay_acquisition
from teadata.column_schema import source_schema_fingerprint
from teadata.dataset_metadata import DatasetMetadata, ReferenceDocument, ResolvedDataset
from teadata.manifest import canonical_json
from teadata.provenance import ProvenanceStore
from teadata.scripts.refresh_tapr import compare_inputs, inspect_dictionary, main
from teadata.tapr import compare_tapr_tables, read_tapr_table
from teadata.values import ValuePolicy, measure, source_token_statuses


def acquisition_inputs(tmp_path, *, edition='2025'):
    """A configured identity and real-shaped selected documentation closure."""
    references = ProvenanceStore(tmp_path / 'references')
    index = tmp_path / 'index.html'
    index.write_bytes(b'<a href="selection">Retained publisher index</a>')
    index_id = references.record_acquisition(index, ResolvedDataset('reference:index', 'reference', None, None, '',
        DatasetMetadata(source_url='https://example.test/index')))
    selected = []
    for role in ['selection', 'dictionary']:
        path = tmp_path / (role + '.html')
        path.write_text('<html>Retained ' + role + '</html>')
        url = 'https://example.test/' + role
        source = ResolvedDataset('reference:' + role, 'reference', None, None, '', DatasetMetadata(
            source_url=url, provenance_ids=(index_id,), version=edition))
        identifier = references.record_acquisition(path, source, requested_url=url,
            downloaded_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
        selected.append({'reference_id': role, 'role': 'data_dictionary' if role == 'dictionary' else 'source_notes',
            'url': url, 'provenance_ids': [identifier], 'reviewed_sha256': sha256(path.read_bytes()).hexdigest()})
    metadata = tmp_path / 'observation-metadata.json'
    metadata.write_text(json.dumps({'title': '2025 Campus STUD all-checked observation', 'version': edition,
        'period': {'kind': 'unknown'}, 'references': selected}))
    config = tmp_path / 'catalog.yaml'
    config.write_text('''value_policies: {config_policy: {tokens: {}}}
data_sources:
  campus_tapr_student_staff_profile:
    2025:
      path: nonexistent-source.xlsx
      metadata:
        version: '2025'
        period: {kind: school_year, start_year: 2024, end_year: 2025}
        field_year: 2025
        value_policy: config_policy
        metrics: {enrollment: {field: enrollment, unit: count}}
        references: [{reference_id: config_only, role: source_notes, status: unreviewed}]
''')
    args = ['acquire', '--url', 'https://rptsvr1.tea.texas.gov/cgi/sas/broker/',
        '--dataset', 'campus_tapr_student_staff_profile', '--edition', edition,
        '--method', 'POST', '--filename', '2025-campus-STUD.csv',
        '--content-type', 'application/x-www-form-urlencoded', '--no-redirects',
        '--metadata', str(metadata), '--metadata-sha256', sha256(metadata.read_bytes()).hexdigest(),
        '--config', str(config), '--config-sha256', sha256(config.read_bytes()).hexdigest(), '--year', '2025',
        '--reference-store', str(references.root), '--store', str(tmp_path / 'observations')]
    return args, metadata, config, references


@pytest.mark.parametrize('failure', ['dataset_typo', 'prior_year', 'edition', 'metadata_version',
    'metadata_sha', 'config_sha', 'missing_config', 'missing_year', 'missing_closure',
    'wrong_reference_sha', 'wrong_reference_url', 'configured_edition', 'oversized_metadata', 'oversized_config'])
def test_acquire_invalid_bound_inputs_fail_before_http(tmp_path, monkeypatch, failure):
    args, metadata, config, references = acquisition_inputs(tmp_path)
    monkeypatch.setattr('teadata.scripts.refresh_tapr.AcquisitionClient',
        lambda *a, **k: pytest.fail('Invalid preflight constructed HTTP client'))
    def option(name, value):
        args[args.index(name) + 1] = value
    if failure == 'dataset_typo':
        option('--dataset', 'campus_tapr_student_studf_profile')
    elif failure == 'prior_year':
        option('--year', '2026')
    elif failure == 'edition':
        option('--edition', '2024')
    elif failure == 'metadata_version':
        raw = json.loads(metadata.read_text())
        raw['version'] = '2024'
        metadata.write_text(json.dumps(raw))
        option('--metadata-sha256', sha256(metadata.read_bytes()).hexdigest())
    elif failure in {'metadata_sha', 'config_sha'}:
        option('--' + failure.replace('_', '-') + '256', '0' * 64)
    elif failure in {'missing_config', 'missing_year'}:
        name = '--config' if failure == 'missing_config' else '--year'
        index = args.index(name)
        del args[index:index + 2]
    elif failure == 'missing_closure':
        next((references.root / 'records').glob('*.json')).unlink()
    elif failure in {'wrong_reference_sha', 'wrong_reference_url'}:
        raw = json.loads(metadata.read_text())
        if failure == 'wrong_reference_sha':
            raw['references'][0]['reviewed_sha256'] = '0' * 64
        else:
            raw['references'][0]['url'] = 'https://example.test/not-the-selected-source'
        metadata.write_text(json.dumps(raw))
        option('--metadata-sha256', sha256(metadata.read_bytes()).hexdigest())
    elif failure == 'configured_edition':
        config.write_text(config.read_text().replace("version: '2025'", "version: '2024'"))
        option('--config-sha256', sha256(config.read_bytes()).hexdigest())
    else:
        selected = metadata if failure == 'oversized_metadata' else config
        selected.write_bytes(b' ' * (2 * 1024 * 1024 + 1))
        option('--metadata-sha256' if selected == metadata else '--config-sha256', sha256(selected.read_bytes()).hexdigest())
    with pytest.raises((ValueError, KeyError, FileNotFoundError, RuntimeError)):
        main(args)
    assert not (tmp_path / 'observations' / 'records').exists()


def test_acquire_cli_captures_actual_shape_metadata_config_controls_and_closed_history(tmp_path, monkeypatch, capsys):
    from urllib.parse import urlencode
    import teadata.scripts.refresh_tapr as refresh

    args, metadata_path, config_path, references = acquisition_inputs(tmp_path)
    original_metadata, original_config = metadata_path.read_bytes(), config_path.read_bytes()
    original_records = {path.name: path.read_bytes() for path in (references.root / 'records').glob('*.json')}
    controls = [('_service', 'marykay'), ('_program', 'perfrept.perfmast.sas'), ('_debug', '0'),
        ('tapr', 'all_c'), ('ccyy', '2025'), ('dsname', 'STUD'), ('sumlev', 'C'), ('level', 'Campus'),
        ('id', ''), ('prgopt', 'reports/tapr/dd/dd_tapr_step_7.sas'), ('key', 'ETALL|NTALL'),
        ('key', 'ETSIM|ETSIN|ETSIN|ETSPH|ETSPH|ETSAU|ETSAU|ETSBH|ETSBH|ETSNO|ETSNO'), ('datafmt', 'csv')]
    for name, value in controls:
        args.extend(['--field', name, value])
    read = refresh._read_pinned_input
    def captured(path, expected):
        raw = read(path, expected)
        path.write_bytes(b'changed after capture')
        return raw
    monkeypatch.setattr(refresh, '_read_pinned_input', captured)
    observed = []
    def opened(request, timeout):
        observed.append(request)
        assert timeout == 45
        assert request.full_url == 'https://rptsvr1.tea.texas.gov/cgi/sas/broker/'
        assert request.get_method() == 'POST' and request.data == urlencode(controls).encode()
        if len(observed) == 1:
            raise TimeoutError("Publisher response timed out")
        response = Response(b'complete original public CSV bytes')
        response.url = request.full_url
        return response
    def client(store, **kwargs):
        assert kwargs['attempts'] == 3 and kwargs['max_retry_wait'] == 30
        assert any(isinstance(handler, NoRedirect) for handler in kwargs['opener'].handlers)
        kwargs['opener'] = SimpleNamespace(open=opened)
        return AcquisitionClient(store, **kwargs)
    monkeypatch.setattr(refresh, 'AcquisitionClient', client)
    monkeypatch.setattr('teadata.acquisition.time.sleep', lambda delay: None)
    monkeypatch.setattr('socket.socket', lambda *a, **k: pytest.fail('Real socket attempted'))
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    archive = ProvenanceStore(tmp_path / 'observations')
    record = archive.read_record(result['record_id'])
    assert len(observed) == 2 and record['dataset'] == 'campus_tapr_student_staff_profile'
    assert record['parameters']['transport_attempts'] == 2
    assert record['section'] == 'data_sources' and record['requested_year'] == record['resolved_year'] == 2025
    actual = DatasetMetadata.from_dict(record['source_metadata'])
    expected = DatasetMetadata.from_dict(json.loads(original_metadata))
    assert replace(actual, provenance_ids=()).to_dict() == expected.to_dict()
    assert actual.period.kind == 'unknown' and actual.value_policy is None and not actual.metrics
    retained_inputs = {archive.read_record(identifier)['dataset']: archive.read_record(identifier)
                      for identifier in actual.provenance_ids}
    for role, raw in [('metadata', original_metadata), ('config', original_config)]:
        source = retained_inputs['reference:acquire_' + role]
        assert source['method'] == 'local_observation' and source['downloaded_at'] is None
        assert archive.verify_artifact(source['artifact']['sha256']).read_bytes() == raw
    assert record['parameters']['form_controls'] == [list(pair) for pair in controls]
    assert len(archive.verify_record_closure([result['record_id']])) == 6
    assert {path.name: path.read_bytes() for path in (references.root / 'records').glob('*.json')} == original_records


def test_acquire_reference_section_is_explicit_and_does_not_require_config(tmp_path, monkeypatch, capsys):
    args, metadata, _config, _references = acquisition_inputs(tmp_path, edition='2024')
    for name in ['--config', '--config-sha256', '--year']:
        index = args.index(name)
        del args[index:index + 2]
    args.extend(['--section', 'reference', '--timeout', '90', '--attempts', '1', '--max-retry-wait', '0'])
    args[args.index('--dataset') + 1] = 'tapr_2024_masking_reference'
    args[args.index('--method') + 1] = 'GET'
    def client(store, **kwargs):
        kwargs['opener'] = SimpleNamespace(open=lambda *a, **k: Response(b'masking document'))
        return AcquisitionClient(store, **kwargs)
    monkeypatch.setattr('teadata.scripts.refresh_tapr.AcquisitionClient', client)
    assert main(args) == 0
    archive = ProvenanceStore(tmp_path / 'observations')
    record = archive.read_record(json.loads(capsys.readouterr().out)['record_id'])
    assert record['section'] == 'reference' and record['dataset'] == 'tapr_2024_masking_reference'
    assert record['parameters']['transport_limits'] == {
        'timeout': 90, 'attempts': 1, 'max_retry_wait': 0, 'max_bytes': 32 * 1024 * 1024,
    }
    assert record['requested_year'] is record['resolved_year'] is None
    assert replace(DatasetMetadata.from_dict(record['source_metadata']), provenance_ids=()).to_dict() == DatasetMetadata.from_dict(json.loads(metadata.read_bytes())).to_dict()


def test_acquire_reference_section_rejects_config_flags_and_old_unbound_cli(tmp_path, monkeypatch):
    args, _metadata, _config, _references = acquisition_inputs(tmp_path)
    monkeypatch.setattr('teadata.scripts.refresh_tapr.AcquisitionClient', lambda *a, **k: pytest.fail('HTTP client'))
    with pytest.raises(ValueError, match='does not select'):
        main([*args, '--section', 'reference'])
    with pytest.raises(SystemExit):
        main(['acquire', '--url', 'https://example.test/reference', '--dataset', 'unregistered',
            '--edition', '2025', '--filename', 'reference.html', '--method', 'GET', '--store', str(tmp_path / 'archive')])


def test_acquire_no_redirect_handler_rejects_followup():
    from urllib.error import HTTPError
    from urllib.request import Request

    with pytest.raises(HTTPError, match='redirect blocked'):
        NoRedirect().redirect_request(Request('https://example.test/original'), None,
            302, 'Found', {}, 'https://example.test/not-authorized')


@pytest.mark.parametrize('length', [None, '002'])
@pytest.mark.parametrize('with_context', [False, True])
def test_dictionary_cli_keeps_partial_occurrences_and_exact_retained_history_offline(tmp_path, monkeypatch, capsys, length, with_context):
    def forbidden(*args, **kwargs):
        pytest.fail('Dictionary inspection attempted HTTP')
    monkeypatch.setattr('teadata.scripts.refresh_tapr.AcquisitionClient', forbidden)
    source_store = ProvenanceStore(tmp_path / 'original')
    selection = tmp_path / 'selection.html'
    selection.write_text('<form>Original source controls</form>')
    parent = source_store.record_acquisition(selection, ResolvedDataset('reference:selection', 'reference',
        None, None, '', DatasetMetadata(title='Original selection', source_url='https://example.test/selection')))
    original = tmp_path / 'dictionary.html'
    raw = (b'<span class="tooltip">X<span class="tooltiptext"><div class="dd_hdr">Name:</div>'
        b'<div class="dd_desc">X</div><div class="dd_hdr">Type:</div><div class="dd_desc">Number</div>'
        b'<div class="dd_hdr">Description:</div><div class="dd_desc">Original</br> description</div></span></span>\n'
        b'<span class="tooltip">Y<span class="tooltiptext">Y</span></span>')
    if length is not None:
        raw = raw.replace(b'<div class="dd_hdr">Description:',
            f'<div class="dd_hdr">Length:</div><div class="dd_desc">{length}</div><div class="dd_hdr">Description:'.encode())
    if with_context:
        raw = (b'<table><tr><td class="linecontent" colspan="2">Literal section</td></tr><tr><td>'
               + raw.replace(b'\n', b'</td></tr>\n<tr><td>') + b'</td></tr></table>')
    original.write_bytes(raw)
    metadata = DatasetMetadata(version='2025', references=(ReferenceDocument(reference_id='selection', url='https://example.test/selection',
        role='source_notes', provenance_ids=(parent,), reviewed_sha256=sha256(selection.read_bytes()).hexdigest()),))
    identifier = source_store.record_acquisition(original,
        ResolvedDataset('reference:dictionary', 'reference', 2025, 2025, '', metadata),
        downloaded_at=datetime(2025, 12, 1, tzinfo=timezone.utc),
        requested_url='https://example.test/dictionary', parameters={'http_method': 'GET'})
    before = source_store.read_record(identifier)
    archive, output = tmp_path / 'observations', tmp_path / 'dictionary-report.json'
    args = ['dictionary', '--source-store', str(source_store.root), '--record', identifier,
            '--encoding', 'utf-8', '--store', str(archive), '--output', str(output)]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    report = json.loads(output.read_text())
    assert result['status'] == 'inspected_dictionary'
    assert report['max_bytes'] == 2 * 1024 * 1024
    assert [row['code'] for row in report['dictionary_records']] == ['X', 'Y']
    assert report['dictionary_records'][0]['publisher_description'] == 'Original description'
    if length is not None:
        assert report['dictionary_records'][0]['publisher_length'] == length
    else:
        assert 'publisher_length' not in report['dictionary_records'][0]
    assert 'publisher_length' not in report['dictionary_records'][1]
    assert report['dictionary_records'][1]['publisher_type'] is None
    assert report['dictionary_records'][1]['publisher_description'] is None
    assert report['dictionary_records'][1]['source_line'] == 2
    assert report['dictionary_records'][1]['source_column'] == (8 if with_context else 0)
    if with_context:
        assert all(row['publisher_context'] == {'heading': 'Literal section', 'source_line': 1, 'source_column': 11}
                   for row in report['dictionary_records'])
    else:
        assert all('publisher_context' not in row for row in report['dictionary_records'])
    assert {r['source_sha256'] for r in report['dictionary_records']} == {sha256(raw).hexdigest()}
    closure = ProvenanceStore(archive).verify_record_closure([result['processing_record_id']])
    assert set(closure) == {parent, identifier, result['processing_record_id']}
    assert source_store.read_record(identifier) == before == closure[identifier]
    assert original.read_bytes() == raw and not report['source_activation']
    published = output.read_bytes()
    with pytest.raises(FileExistsError, match='already exists'):
        main(args)
    assert output.read_bytes() == published


@pytest.mark.parametrize('raw,encoding,error', [
    (b'<span class="tooltip">X<span class="tooltiptext">X', 'utf-8', ValueError),
    ('é'.encode('iso-8859-1'), 'utf-8', UnicodeDecodeError),
])
def test_dictionary_cli_failure_does_not_publish_a_successful_report(tmp_path, raw, encoding, error):
    source_store = ProvenanceStore(tmp_path / 'original')
    original = tmp_path / 'dictionary.html'
    original.write_bytes(raw)
    identifier = source_store.record_acquisition(original,
        ResolvedDataset('reference:dictionary', 'reference', None, None, '', DatasetMetadata()))
    output = tmp_path / 'report.json'
    archive = tmp_path / 'archive'
    with pytest.raises(error):
        main(['dictionary', '--source-store', str(source_store.root), '--record', identifier,
              '--encoding', encoding, '--store', str(archive), '--output', str(output)])
    assert not output.exists() and original.read_bytes() == raw
    assert len(list((archive / 'records').glob('*.json'))) == 1


def dictionary_input(tmp_path):
    raw = b'<span class="tooltip">X<span class="tooltiptext">X</span></span>'
    original = tmp_path / 'dictionary.html'
    original.write_bytes(raw)
    store = ProvenanceStore(tmp_path / 'source')
    identifier = store.record_acquisition(original,
        ResolvedDataset('reference:dictionary', 'reference', None, None, '', DatasetMetadata()))
    return raw, store, identifier, Namespace(source_store=store.root, record=identifier,
        encoding='utf-8', store=tmp_path / 'archive', output=tmp_path / 'report.json', max_bytes=len(raw))


@pytest.mark.parametrize('actual_growth', [False, True])
def test_dictionary_size_guard_precedes_bulk_read_and_closure_copy(tmp_path, monkeypatch, actual_growth):
    raw, store, identifier, args = dictionary_input(tmp_path)
    path = store.verify_artifact(sha256(raw).hexdigest())
    if actual_growth:
        path.write_bytes(raw + b' ')
    else:
        args.max_bytes = len(raw) - 1
    original_fdopen = os.fdopen
    identity = path.stat()
    opened = []
    class NoBodyRead:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *unused):
            self.stream.close()
        def fileno(self):
            return self.stream.fileno()
        def read(self, *unused):
            pytest.fail('Oversized dictionary body was read')
    def guarded_open(fd, *positional, **keywords):
        stream = original_fdopen(fd, *positional, **keywords)
        current = os.fstat(fd)
        if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino) and positional == ('rb',):
            assert actual_growth, 'Advertised oversize should fail before opening its body'
            opened.append(True)
            return NoBodyRead(stream)
        return stream
    monkeypatch.setattr(os, 'fdopen', guarded_open)
    def forbidden_copy(*unused):
        pytest.fail('Oversized dictionary reached bulk closure copy')
    monkeypatch.setattr(ProvenanceStore, 'copy_record_from', forbidden_copy)
    with pytest.raises(ValueError, match='exceeds max-bytes'):
        inspect_dictionary(args)
    assert opened == ([True] if actual_growth else [])
    assert not args.output.exists() and not (args.store / 'records').exists()


def test_dictionary_capped_read_binds_actual_bytes_before_any_processing_receipt(tmp_path, monkeypatch):
    raw, store, identifier, args = dictionary_input(tmp_path)
    selected = store.verify_artifact(sha256(raw).hexdigest())
    original_fdopen = os.fdopen
    identity = selected.stat()
    reads = []
    class ChangedRead:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *unused):
            self.stream.close()
        def fileno(self):
            return self.stream.fileno()
        def read(self, maximum):
            # Equivalent to same-length source mutation after stat/verification.
            assert maximum == args.max_bytes + 1
            reads.append(maximum)
            return self.stream.read(maximum).replace(b'>X<', b'>Y<')
    def changed_open(fd, *positional, **keywords):
        stream = original_fdopen(fd, *positional, **keywords)
        current = os.fstat(fd)
        return (ChangedRead(stream)
                if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino) and positional == ('rb',)
                else stream)
    monkeypatch.setattr(os, 'fdopen', changed_open)
    with pytest.raises(ValueError, match='bytes differ from the selected acquisition'):
        inspect_dictionary(args)
    assert reads == [args.max_bytes + 1]
    assert not args.output.exists() and not (args.store / 'records').exists()


def test_dictionary_does_not_parse_a_later_unbound_read_after_verification(tmp_path, monkeypatch):
    raw, store, identifier, args = dictionary_input(tmp_path)
    original_read = Path.read_bytes
    def changed_read(path):
        if path == args.store / 'objects' / sha256(raw).hexdigest():
            return raw.replace(b'>X<', b'>Y<')
        return original_read(path)
    monkeypatch.setattr(Path, 'read_bytes', changed_read)
    result = inspect_dictionary(args)
    report = json.loads(args.output.read_text())
    assert [row['code'] for row in report['dictionary_records']] == ['X']
    assert {row['source_sha256'] for row in report['dictionary_records']} == {sha256(raw).hexdigest()}
    assert ProvenanceStore(args.store).verify_record_closure([result['processing_record_id']])


@pytest.mark.parametrize('maximum', [0, -1])
def test_dictionary_limit_must_be_positive_before_loading_any_input(tmp_path, maximum):
    args = ['dictionary', '--source-store', str(tmp_path / 'missing'), '--record', 'a' * 64,
            '--encoding', 'utf-8', '--store', str(tmp_path / 'archive'),
            '--output', str(tmp_path / 'report.json'), '--max-bytes', str(maximum)]
    with pytest.raises(ValueError, match='positive integer'):
        main(args)
    assert not (tmp_path / 'archive').exists()


def test_dictionary_cli_accepts_an_explicit_reviewed_larger_limit(tmp_path, capsys):
    raw, store, identifier, args = dictionary_input(tmp_path)
    assert main(['dictionary', '--source-store', str(store.root), '--record', identifier,
                 '--encoding', 'utf-8', '--store', str(args.store), '--output', str(args.output),
                 '--max-bytes', '4194304']) == 0
    result = json.loads(capsys.readouterr().out)
    report = json.loads(args.output.read_text())
    assert report['max_bytes'] == 4194304
    assert report['source_artifact']['sha256'] == sha256(raw).hexdigest()
    assert ProvenanceStore(args.store).read_record(result['processing_record_id'])['parameters']['max_bytes'] == 4194304


class Response(BytesIO):
    status = 200
    url = 'https://example.test/source.csv'
    headers = {'ETag': 'publisher-tag', 'Last-Modified': 'Thu, 01 Jan 2026 00:00:00 GMT'}


def test_explicit_post_replay_keeps_repeated_controls_closed_history_and_old_bytes(tmp_path):
    requests = []
    payloads = iter([b'old original', b'new original', b'third original'])
    def opened(request, timeout):
        requests.append((request, timeout))
        return Response(next(payloads))
    store = ProvenanceStore(tmp_path / 'store')
    client = AcquisitionClient(store, opener=SimpleNamespace(open=opened), attempts=1, timeout=20, max_bytes=1024)
    source = ResolvedDataset('campus_tapr_student_profile', 'data_sources', None, None, '', DatasetMetadata(version='2024'))
    controls = [('key', 'A|B'), ('key', 'C'), ('empty', ''), ('datafmt', 'csv')]
    first = acquire_form(client, 'https://example.test/form', source, filename='public.csv', method='POST',
        fields=controls, content_type='application/x-www-form-urlencoded; charset=UTF-8')
    original = store.read_record(first.record_id)
    second = replay_acquisition(client, first.record_id)
    third = replay_acquisition(client, second.record_id)
    assert len(requests) == 3 and all(timeout == 20 for _, timeout in requests)
    assert all(request.data == b'key=A%7CB&key=C&empty=&datafmt=csv' for request, _ in requests)
    assert all(request.get_method() == 'POST' for request, _ in requests)
    assert all(request.get_header('Content-type') == 'application/x-www-form-urlencoded; charset=UTF-8' for request, _ in requests)
    assert all(not request.has_header('Authorization') and not request.has_header('Cookie') for request, _ in requests)
    assert store.read_record(first.record_id) == original and first.path.read_bytes() == b'old original'
    assert second.path.read_bytes() == b'new original' and third.path.read_bytes() == b'third original'
    closure = store.verify_record_closure([third.record_id])
    assert set(closure) == {first.record_id, second.record_id, third.record_id}
    assert original['parameters']['form_controls'] == [list(p) for p in controls]
    assert original['response_headers']['etag'] == 'publisher-tag'
    assert original['parameters']['transport_limits'] == {'timeout': 20, 'attempts': 1, 'max_bytes': 1024, 'max_retry_wait': 30}


def test_get_replay_preserves_exact_query_and_legacy_post_requires_explicit_controls_and_type(tmp_path):
    requests = []
    def opened(request, timeout):
        requests.append(request)
        return Response(b'original')
    store = ProvenanceStore(tmp_path / 'store')
    client = AcquisitionClient(store, opener=SimpleNamespace(open=opened), attempts=1)
    source = ResolvedDataset('sample', 'data_sources', None, None, '')
    first = acquire_form(client, 'https://example.test/form?edition=2024', source, filename='sample.csv',
        method='GET', fields=[('key', 'a'), ('key', 'b')])
    replay_acquisition(client, first.record_id)
    assert [r.full_url for r in requests] == ['https://example.test/form?edition=2024&key=a&key=b'] * 2
    with pytest.raises(ValueError, match='cannot change'):
        replay_acquisition(client, first.record_id, content_type='text/csv')
    path = tmp_path / 'legacy.csv'
    path.write_bytes(b'legacy')
    identifier = store.record_acquisition(path, source, requested_url='https://example.test/legacy',
        downloaded_at=datetime(2026, 1, 1, tzinfo=timezone.utc), parameters={'http_method': 'POST', 'form_controls': [['key', 'A']]})
    with pytest.raises(ValueError, match='explicit request content type'):
        replay_acquisition(client, identifier)
    repeated = replay_acquisition(client, identifier, content_type='application/x-www-form-urlencoded')
    assert requests[-1].data == b'key=A'
    local = store.record_acquisition(path, source)
    with pytest.raises(ValueError, match='original HTTP'):
        replay_acquisition(client, local)
    no_controls = store.record_acquisition(path, source, requested_url='https://example.test/legacy',
        downloaded_at=datetime(2026, 1, 1, tzinfo=timezone.utc), parameters={'http_method': 'POST'})
    with pytest.raises(ValueError, match='explicit public form controls'):
        replay_acquisition(client, no_controls, content_type='application/x-www-form-urlencoded')
    with pytest.raises(ValueError, match='cannot change'):
        replay_acquisition(client, repeated.record_id, content_type='text/csv')


@pytest.mark.parametrize('method,fields,content_type', [('PUT', [], None), ('POST', [], None),
    ('POST', [], 'text/csv'), ('GET', {'key': 'value'}, None), ('GET', [['', 'v']], None)])
def test_invalid_or_implicit_form_requests_fail_before_network(tmp_path, method, fields, content_type):
    client = AcquisitionClient(ProvenanceStore(tmp_path), opener=SimpleNamespace(open=lambda *a, **k: pytest.fail('network')))
    with pytest.raises((ValueError, TypeError)):
        acquire_form(client, 'https://example.test/form', ResolvedDataset('x', 'data_sources', None, None, ''),
            filename='x.csv', method=method, fields=fields, content_type=content_type)


@pytest.mark.parametrize('method,form_url,controls,observed', [
    ('GET', 'https://example.test/different', [['key', 'B']], 'https://example.test/observed?key=A'),
    ('POST', 'https://example.test/different', [['key', 'B']], 'https://example.test/observed?key=A'),
    ('GET', 'https://example.test/form', [['key', 'B'], ['key', 'A']], 'https://example.test/form?key=A&key=B'),
    ('GET', None, [['key', 'A']], 'https://example.test/form?key=A'),
])
def test_replay_rejects_auxiliary_request_drift_before_http(tmp_path, method, form_url, controls, observed):
    store = ProvenanceStore(tmp_path / 'store')
    path = tmp_path / 'original.csv'
    path.write_bytes(b'retained original')
    parameters = {'http_method': method, 'form_controls': controls,
        'form_content_type': 'application/x-www-form-urlencoded' if method == 'POST' else None}
    if form_url is not None:
        parameters['form_url'] = form_url
    identifier = store.record_acquisition(path, ResolvedDataset('sample', 'data_sources', None, None, ''),
        requested_url=observed, downloaded_at=datetime(2026, 1, 1, tzinfo=timezone.utc), parameters=parameters)
    original = (store.root / 'records' / (identifier + '.json')).read_bytes()
    client = AcquisitionClient(store, opener=SimpleNamespace(open=lambda *a, **k: pytest.fail('HTTP')))
    with pytest.raises(ValueError, match='original|observed'):
        replay_acquisition(client, identifier)
    assert (store.root / 'records' / (identifier + '.json')).read_bytes() == original


def test_legacy_get_replays_observed_url_without_reconstructing_controls(tmp_path):
    store = ProvenanceStore(tmp_path / 'store')
    path = tmp_path / 'original.csv'
    path.write_bytes(b'retained original')
    url = 'https://example.test/form?key=A&key=B&empty=&spelling=%2f'
    identifier = store.record_acquisition(path, ResolvedDataset('sample', 'data_sources', None, None, ''),
        requested_url=url, downloaded_at=datetime(2026, 1, 1, tzinfo=timezone.utc), parameters={'http_method': 'GET'})
    requests = []
    def opened(request, timeout):
        requests.append(request)
        return Response(b'replayed bytes')
    acquired = replay_acquisition(AcquisitionClient(store, opener=SimpleNamespace(open=opened)), identifier)
    assert len(requests) == 1 and requests[0].full_url == url and requests[0].data is None
    assert set(store.verify_record_closure([acquired.record_id])) == {identifier, acquired.record_id}


@pytest.mark.parametrize('parameters', ['charset=ISO-8859-1', 'charset=utf8',
    'charset=UTF-8; charset=ISO-8859-1', 'charset=UTF-8; charset=UTF-8', 'charset="UTF-8', 'boundary=anything'])
def test_post_rejects_unsupported_or_ambiguous_encoding_before_http(tmp_path, parameters):
    client = AcquisitionClient(ProvenanceStore(tmp_path), opener=SimpleNamespace(open=lambda *a, **k: pytest.fail('HTTP')))
    with pytest.raises(ValueError, match='only UTF-8'):
        acquire_form(client, 'https://example.test/form', ResolvedDataset('sample', 'data_sources', None, None, ''),
            filename='sample.csv', method='POST', fields=[('name', 'é')],
            content_type='application/x-www-form-urlencoded; ' + parameters)


@pytest.mark.parametrize('content_type', ['application/x-www-form-urlencoded',
    'application/x-www-form-urlencoded; charset=UTF-8', 'application/x-www-form-urlencoded; Charset = "utf-8"'])
def test_unicode_post_and_replay_use_explicit_utf8_control_bytes(tmp_path, content_type):
    requests = []
    def opened(request, timeout):
        requests.append(request)
        return Response(b'original')
    client = AcquisitionClient(ProvenanceStore(tmp_path), opener=SimpleNamespace(open=opened))
    first = acquire_form(client, 'https://example.test/form', ResolvedDataset('sample', 'data_sources', None, None, ''),
        filename='sample.csv', method='POST', fields=[('name', 'é'), ('name', '東京')], content_type=content_type)
    replay_acquisition(client, first.record_id)
    assert len(requests) == 2
    assert all(r.data == b'name=%C3%A9&name=%E6%9D%B1%E4%BA%AC' for r in requests)
    assert all(r.get_header('Content-type') == content_type for r in requests)


def source(tmp_path, name, rows, *, code='VALUE', label='Value', field='value', period=None):
    dataset = 'campus_tapr_student_profile'
    path = tmp_path / (name + '.csv')
    labels, codes = ['Campus Number', label], ['CAMPUS', code]
    with path.open('w', newline='') as stream:
        csv.writer(stream).writerows([labels, codes, *rows])
    mapping = {'table': 'student_profile', 'metric_period': period or {'kind': 'unknown'},
        'input_sha256': sha256(path.read_bytes()).hexdigest(), 'source_schema_sha256': source_schema_fingerprint(labels, codes),
        'fields': [{'position': i, 'code': c, 'label': human, 'field': canonical,
            'disposition': 'identity' if i == 1 else 'fact', 'dictionary_status': 'not_applicable' if i == 1 else 'not_documented',
            'publisher_type': None, 'publisher_description': None}
            for i, (c, human, canonical) in enumerate(zip(codes, labels, ['campus_number', field], strict=True), 1)]}
    schema = tmp_path / (name + '.json')
    schema.write_text(canonical_json({'schema_version': 2, 'mapping_version': 1,
        'datasets': [{'dataset': dataset, 'edition': '2024', 'source_only': True, 'tables': [mapping]}]}) + '\n')
    store = ProvenanceStore(tmp_path / (name + '-store'))
    identifier = store.record_acquisition(path, ResolvedDataset(dataset, 'data_sources', None, None, str(path), DatasetMetadata(version='2024')))
    native = read_tapr_table(path, mapping=mapping, key_field='campus_number', dataset=dataset, edition='2024')
    return native, store, identifier, schema


def test_keyed_native_diff_keeps_historical_members_schema_absence_and_independent_definitions(tmp_path):
    old, *_ = source(tmp_path, 'old', [['001902041', '7'], ['001902001', '.']], label='Old definition')
    new, *_ = source(tmp_path, 'new', [['001902099', '9'], ['001902001', '']], label='Changed definition', field='changed_value',
        period={'kind': 'school_year', 'start_year': 2023, 'end_year': 2024})
    result = compare_tapr_tables(old, new, key_field='campus_number')
    assert result['keysets']['overlap'] == 1
    assert result['keysets']['old_only']['examples'] == ["'001902041"]
    assert result['keysets']['new_only']['examples'] == ["'001902099"]
    value = result['columns']['common'][1]
    assert not value['header_equal'] and value['lexical_changes'] == 1
    assert value['raw_transition_examples'][0]['old_raw'] == '.'
    assert value['raw_transition_examples'][0]['new_raw'] == ''
    assert value['raw_transition_examples'][0]['new_status'] == 'unbound'
    assert result['source_structure']['old']['metric_period'] != result['source_structure']['new']['metric_period']
    other, *_ = source(tmp_path, 'other', [['001902001', '0']], code='NEW_CODE', label='Old definition')
    result = compare_tapr_tables(old, other, key_field='campus_number')
    assert [f['code'] for f in result['columns']['old_only']] == ['VALUE']
    assert [f['code'] for f in result['columns']['new_only']] == ['NEW_CODE']
    assert [f['code'] for f in result['columns']['common']] == ['CAMPUS']


def test_decimal_spelling_equivalence_is_exact_beyond_float_precision_and_unsupported_is_unknown(tmp_path):
    old_tokens = ['9007199254740993', '001', '1e3', '', '-1', '0']
    new_tokens = ['9007199254740992', '1.0', '1000', '0', '-1.0', '0.0']
    old, *_ = source(tmp_path, 'old', [[f'001902{i:03d}', v] for i, v in enumerate(old_tokens, 1)])
    new, *_ = source(tmp_path, 'new', [[f'001902{i:03d}', v] for i, v in enumerate(new_tokens, 1)])
    value = compare_tapr_tables(old, new, key_field='campus_number')['columns']['common'][1]
    assert value['lexical_changes'] == 6 and value['decimal_equal_changes'] == 3
    assert value['decimal_unequal_changes'] == 1 and value['unclassified_changes'] == 2
    assert all(p['old_status'] == p['new_status'] == 'unbound' for p in value['raw_transition_examples'])


def test_dictionary_only_changes_remain_dictionary_observations_not_fabricated_fields(tmp_path):
    old, *_ = source(tmp_path, 'old', [['001902001', '1']])
    new, *_ = source(tmp_path, 'new', [['001902001', '1']])
    dictionary = [{'ordinal': 1, 'code': 'UNUSED', 'publisher_type': 'Number', 'publisher_description': 'Available dictionary field',
        'source_sha256': 'a' * 64, 'source_line': 2, 'source_column': 0,
        'publisher_context': {'heading': 'Literal section', 'source_line': 1, 'source_column': 0}}]
    new = replace(new, mapping={**new.mapping, 'dictionary_records': dictionary})
    result = compare_tapr_tables(old, new, key_field='campus_number')
    assert result['source_structure']['new']['dictionary_records'] == dictionary
    assert not result['columns']['new_only'] and not result['columns']['old_only']
    assert [f['code'] for f in result['columns']['common']] == ['CAMPUS', 'VALUE']
    assert all(f['dictionary_definition_equal'] for f in result['columns']['common'])


def test_negative_observed_and_suppressed_policies_have_one_canonical_authority(tmp_path):
    old, *_ = source(tmp_path, 'old', [['001902001', '-1']])
    new, *_ = source(tmp_path, 'new', [['001902001', '-1.0']])
    observed = ValuePolicy.from_dict({'policy_id': 'numeric'})
    suppressed = ValuePolicy.from_dict({'policy_id': 'reviewed', 'reference_id': 'masking',
        'tokens': {'-1': {'status': 'suppressed', 'reason': 'explicit reviewed source rule'}}})
    for policy, expected in [(observed, 1), (suppressed, 0)]:
        result = compare_tapr_tables(old, new, key_field='campus_number', old_policies={'VALUE': policy}, new_policies={'VALUE': policy})
        value = result['columns']['common'][1]
        assert value['decimal_equal_changes'] == expected
        assert value['raw_transition_examples'][0]['old_status'] == measure('-1', policy).status.value
        assert result['selected_field_policies']['old']['VALUE'] == policy.to_dict()
    with pytest.raises(ValueError, match='actual non-identity'):
        compare_tapr_tables(old, new, key_field='campus_number', old_policies={'*': suppressed})


@pytest.mark.parametrize('bound', [0, 2, 10])
def test_changed_observation_examples_are_keyed_deterministic_bounded_and_policy_qualified(tmp_path, bound):
    old, *_ = source(tmp_path, 'old', [['001902003', '1'], ['001902001', '1'], ['001902004', '7'], ['001902002', '-1']])
    new, *_ = source(tmp_path, 'new', [['001902002', '0'], ['001902004', '7'], ['001902001', '2'], ['001902003', '1.0']])
    policy = ValuePolicy.from_dict({'policy_id': 'explicit', 'tokens': {'-1': {'status': 'suppressed', 'reason': 'source-specific rule'}}})
    result = compare_tapr_tables(old, new, key_field='campus_number', old_policies={'VALUE': policy},
        new_policies={'VALUE': policy}, max_examples=bound)
    field = result['columns']['common'][1]
    observations = field['changed_observations']
    assert result['key_field'] == 'campus_number'
    assert observations['count'] == field['lexical_changes'] == 3
    assert observations['truncated'] == (bound < 3)
    examples = observations['examples']
    assert [e['entity_key'] for e in examples] == ["'001902001", "'001902002", "'001902003"][:bound]
    assert len(examples) == min(bound, 3)
    if examples:
        assert (examples[0]['old_raw'], examples[0]['new_raw']) == ('1', '2')
        assert examples[0]['old_status'] == examples[0]['new_status'] == 'observed'
        assert examples[0]['decimal_lexeme_equivalence'] == 'unequal'
        assert examples[1]['old_status'] == 'suppressed' and examples[1]['new_status'] == 'observed'
        assert examples[1]['decimal_lexeme_equivalence'] == 'unclassified'
    if bound > 3:
        assert examples[2]['decimal_lexeme_equivalence'] == 'equal'


def test_changed_observation_examples_do_not_invent_cells_without_overlapping_keys(tmp_path):
    old, *_ = source(tmp_path, 'old', [['001902001', '1']])
    new, *_ = source(tmp_path, 'new', [['001902002', '2']])
    result = compare_tapr_tables(old, new, key_field='campus_number', max_examples=1)
    assert result['keysets']['overlap'] == 0
    assert result['columns']['common'][1]['changed_observations'] == {'count': 0, 'examples': [], 'truncated': False}


def test_status_bridge_matches_scalar_and_has_an_explicit_distinct_work_bound():
    policy = ValuePolicy.from_dict({'policy_id': 'explicit', 'tokens': {'-1': {'status': 'suppressed', 'reason': 'documented'}}})
    tokens = ['-1', '-1.00', ' -1 ', '0', '.', '', None, '9007199254740993', 'n/a']
    assert source_token_statuses(pa.array(tokens), policy).to_pylist() == [measure(t, policy).status.value for t in tokens]
    with pytest.raises(ValueError, match='distinct-token'):
        source_token_statuses(pa.array(['1', '2', '3']), policy, max_distinct=2)


@pytest.mark.parametrize('outside', ['1' * 39, '0.' + '1' * 39])
def test_outside_exact_decimal_envelope_is_unclassified_without_rounding(tmp_path, outside):
    old, *_ = source(tmp_path, 'old', [['001902001', outside]])
    new, *_ = source(tmp_path, 'new', [['001902001', '1']])
    field = compare_tapr_tables(old, new, key_field='campus_number', max_examples=0)['columns']['common'][1]
    assert field['unclassified_changes'] == 1
    assert field['decimal_equal_changes'] == field['decimal_unequal_changes'] == 0
    assert field['raw_transition_examples'] == [] and not field['all_raw_pairs_emitted']


def compare_args(tmp_path, old, new):
    args = Namespace(store=tmp_path / 'result-store', output=tmp_path / 'result.json', scope='campus', max_examples=2,
        old_policy=[], new_policy=[], policy_config=None, policy_config_sha256=None)
    for side, (_, store, identifier, schema) in [('old', old), ('new', new)]:
        for key, value in {'store': store.root, 'record': identifier, 'schema': schema,
            'schema_sha256': sha256(schema.read_bytes()).hexdigest(), 'dictionary': None,
            'dataset': 'campus_tapr_student_profile', 'edition': '2024', 'table': 'student_profile'}.items():
            setattr(args, side + '_' + key, value)
    return args


def test_offline_cli_reports_and_closed_history_are_append_only_without_config_activation(tmp_path, monkeypatch):
    old = source(tmp_path, 'old', [['001902001', '.']])
    new = source(tmp_path, 'new', [['001902001', '0']])
    args = compare_args(tmp_path, old, new)
    monkeypatch.setattr('teadata.scripts.refresh_tapr.AcquisitionClient', lambda *a, **k: pytest.fail('HTTP client'))
    result = compare_inputs(args)
    assert result['status'] == 'compared'
    store = ProvenanceStore(args.store)
    closure = store.verify_record_closure([result['processing_record_id']])
    assert old[2] in closure and new[2] in closure
    assert closure[old[2]] == old[1].read_record(old[2])
    report = json.loads(args.output.read_bytes())
    reader = Path(__file__).resolve().parents[1] / 'teadata/source_csv.py'
    assert report['implementation_sha256']['source_csv.py'] == sha256(reader.read_bytes()).hexdigest()
    assert closure[result['processing_record_id']]['code_version'] == sha256(
        json.dumps(report['implementation_sha256'], sort_keys=True).encode()).hexdigest()
    before = args.output.read_bytes()
    with pytest.raises(FileExistsError):
        compare_inputs(args)
    assert args.output.read_bytes() == before


def test_explicit_policy_capacity_limit_is_reported_as_comparison_block_not_source_drift(tmp_path):
    rows = [[str(100000000 + i), str(i)] for i in range(4097)]
    old, new = source(tmp_path, 'old', rows), source(tmp_path, 'new', rows)
    args = compare_args(tmp_path, old, new)
    config = tmp_path / 'policy.yaml'
    config.write_text('value_policies:\n  numeric:\n    tokens: {}\n')
    args.policy_config, args.policy_config_sha256 = config, sha256(config.read_bytes()).hexdigest()
    args.old_policy = [['VALUE', 'numeric']]
    result = compare_inputs(args)
    report = json.loads(args.output.read_bytes())
    assert result['status'] == 'blocked_comparison'
    assert report['keyed_comparison_attempted'] and not report['keyed_comparison_completed']
    assert 'distinct-token' in report['diagnostic']


@pytest.mark.parametrize('drift', ['source_sha', 'header', 'layout', 'schema_sha', 'edition', 'unexpected_dictionary', 'dictionary_definitions', 'multiple_inputs'])
def test_structural_or_pin_drift_preserves_bytes_and_blocked_report_without_keyed_comparison(tmp_path, monkeypatch, drift):
    old = source(tmp_path, 'old', [['001902001', '.']])
    new = source(tmp_path, 'new', [['001902001', '0']])
    args = compare_args(tmp_path, old, new)
    schema = args.new_schema
    document = json.loads(schema.read_bytes())
    table = document['datasets'][0]['tables'][0]
    if drift == 'source_sha':
        table['input_sha256'] = '0' * 64
    elif drift == 'multiple_inputs':
        table['input_sha256s'] = [table.pop('input_sha256'), 'b' * 64]
        table.pop('layout', None)
    elif drift == 'header':
        table['fields'][1]['label'] = 'Unreviewed actual header'
        table['source_schema_sha256'] = source_schema_fingerprint([f['label'] for f in table['fields']], [f['code'] for f in table['fields']])
    elif drift == 'layout':
        table['layout'] = {'label_record': 1, 'code_record': 2, 'first_data_record': 3, 'last_data_record': 4, 'annotations': []}
    elif drift == 'edition':
        args.new_edition = '2025'
    elif drift == 'unexpected_dictionary':
        args.new_dictionary = tmp_path / 'unbound-dictionary.html'
        args.new_dictionary.write_text('Not declared by this source contract')
    elif drift == 'dictionary_definitions':
        table['dictionary_records'] = [
            {'ordinal': 1, 'code': 'UNUSED', 'publisher_type': 'Number', 'publisher_description': 'Definition A'},
            {'ordinal': 2, 'code': 'UNUSED', 'publisher_type': 'Number', 'publisher_description': 'Definition B'}]
    schema.write_text(canonical_json(document))
    args.new_schema_sha256 = '0' * 64 if drift == 'schema_sha' else sha256(schema.read_bytes()).hexdigest()
    monkeypatch.setattr('teadata.scripts.refresh_tapr.compare_tapr_tables', lambda *a, **k: pytest.fail('Unreviewed keyed comparison'))
    result = compare_inputs(args)
    report = json.loads(args.output.read_bytes())
    assert result['status'] == 'blocked_contract_drift' and report['keyed_comparison_attempted'] is False
    assert report['keyed_comparison_completed'] is False
    assert not report.get('comparison') and '/Users/' not in report['diagnostic']
    if drift == 'multiple_inputs':
        assert 'single raw input' in report['diagnostic']
    archive = ProvenanceStore(args.store)
    closure = archive.verify_record_closure([result['processing_record_id']])
    assert old[2] in closure and new[2] in closure
    assert archive.verify_artifact(sha256(schema.read_bytes()).hexdigest()).read_bytes() == schema.read_bytes()


@pytest.mark.parametrize('target', ['crosswalk', 'policy'])
def test_comparison_parses_retained_pinned_bytes_after_operator_metadata_changes(tmp_path, monkeypatch, target):
    old, new = source(tmp_path, 'old', [['001902001', '-1']]), source(tmp_path, 'new', [['001902001', '-1.0']])
    args = compare_args(tmp_path, old, new)
    if target == 'policy':
        selected = tmp_path / 'policy.yaml'
        selected.write_text('value_policies:\n  reviewed:\n    tokens:\n      "-1":\n        status: suppressed\n        reason: explicit source rule\n')
        args.policy_config, args.policy_config_sha256 = selected, sha256(selected.read_bytes()).hexdigest()
        args.old_policy, args.new_policy = [['VALUE', 'reviewed']], [['VALUE', 'reviewed']]
    else:
        selected = args.new_schema
    original_bytes = selected.read_bytes()
    original = ProvenanceStore.record_acquisition
    def observe_then_change(self, path, *a, **k):
        identifier = original(self, path, *a, **k)
        if Path(path) == selected:
            selected.write_text('operator file changed after its immutable snapshot')
        return identifier
    monkeypatch.setattr(ProvenanceStore, 'record_acquisition', observe_then_change)
    result = compare_inputs(args)
    assert result['status'] == 'compared' and selected.read_bytes() != original_bytes
    archive = ProvenanceStore(args.store)
    assert archive.verify_artifact(sha256(original_bytes).hexdigest()).read_bytes() == original_bytes
    if target == 'policy':
        report = json.loads(args.output.read_bytes())
        field = report['comparison']['columns']['common'][1]
        assert field['decimal_equal_changes'] == 0
        assert field['raw_transition_examples'][0]['old_status'] == 'suppressed'


def test_comparison_decodes_the_retained_dictionary_snapshot(tmp_path, monkeypatch):
    from openpyxl import Workbook

    old, new = source(tmp_path, 'old', [['001902001', '1']]), source(tmp_path, 'new', [['001902001', '2']])
    args = compare_args(tmp_path, old, new)
    dictionary = tmp_path / 'dictionary.xlsx'
    book = Workbook()
    sheet = book.active
    assert sheet is not None
    sheet.append(['Name', 'Type', 'Length', 'Label'])
    sheet.append(['CAMPUS', 'Character', '9', 'Campus Number'])
    sheet.append(['VALUE', 'Number', '8', 'Value'])
    book.save(dictionary)
    original_bytes = dictionary.read_bytes()
    document = json.loads(args.old_schema.read_bytes())
    dataset = document['datasets'][0]
    dataset.pop('source_only')
    dataset['prepared_input_sha256'] = old[0].input_sha256
    table = dataset['tables'][0]
    table['dictionary'] = {'filename': dictionary.name, 'sha256': sha256(original_bytes).hexdigest(), 'sheet_name': sheet.title}
    for field, kind, length in zip(table['fields'], ['Character', 'Number'], ['9', '8'], strict=True):
        field.pop('dictionary_status')
        field.pop('publisher_description')
        field.update(publisher_type=kind, publisher_length=length)
    args.old_schema.write_text(canonical_json(document))
    args.old_schema_sha256, args.old_dictionary = sha256(args.old_schema.read_bytes()).hexdigest(), dictionary
    original = ProvenanceStore.record_acquisition
    def observe_then_change(self, path, *a, **k):
        identifier = original(self, path, *a, **k)
        if Path(path) == dictionary:
            dictionary.write_bytes(b'changed after immutable observation')
        return identifier
    monkeypatch.setattr(ProvenanceStore, 'record_acquisition', observe_then_change)
    result = compare_inputs(args)
    assert result['status'] == 'compared'
    assert dictionary.read_bytes() != original_bytes
    assert ProvenanceStore(args.store).verify_artifact(sha256(original_bytes).hexdigest()).read_bytes() == original_bytes


def test_cli_help_is_offline_and_acquisition_requires_explicit_subcommand(capsys):
    with pytest.raises(SystemExit) as caught:
        main(['--help'])
    assert caught.value.code == 0
    assert 'acquire' in capsys.readouterr().out
