"""Explicit append-only public acquisition/replay and pinned TAPR comparison.

No command changes source configuration or activates acquired/prepared data.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import tempfile
from urllib.request import build_opener

from teadata.acquisition import AcquisitionClient, NoRedirect, acquire_form, replay_acquisition
from teadata.column_schema import crosswalk_dataset, parse_column_schema, table_input_sha256s
from teadata.dataset_metadata import DatasetMetadata, ResolvedDataset
from teadata.provenance import ProvenanceStore, _identifier
from teadata.tapr import compare_tapr_tables, read_tapr_table


def _read_pinned_input(path: Path, expected_sha256: str) -> bytes:
    """Capture one small explicit input; the same bytes are parsed and retained."""
    _identifier(expected_sha256)
    limit = 2 * 1024 * 1024
    with path.open('rb') as stream:
        size = os.fstat(stream.fileno()).st_size
        if size > limit:
            raise ValueError('Acquisition metadata/config input exceeds 2 MiB')
        raw = stream.read(limit + 1)
    if len(raw) > limit or len(raw) != size or sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Captured acquisition input differs from its pinned SHA/size')
    return raw


def acquisition_source(args, archive: ProvenanceStore) -> ResolvedDataset:
    """Validate explicit observation metadata and selected documentation before HTTP."""
    from teadata.manifest import verify_reference_receipts
    from teadata.teadata_config import load_config_bytes

    metadata_raw = _read_pinned_input(args.metadata, args.metadata_sha256)
    metadata = DatasetMetadata.from_dict(json.loads(metadata_raw))
    if not args.edition.strip() or metadata.version != args.edition:
        raise ValueError('Explicit acquisition metadata version must equal edition')
    inputs = [('metadata', args.metadata.name, metadata_raw)]
    if args.section == 'data_sources':
        if args.config is None or args.config_sha256 is None or args.year is None:
            raise ValueError('Data-source acquisition requires pinned config and explicit year')
        config_raw = _read_pinned_input(args.config, args.config_sha256)
        cfg = load_config_bytes(config_raw, path=args.config)
        selected = cfg.resolve_dataset(args.dataset, args.year, strict=True)
        if selected.resolved_year != args.year or selected.metadata.version != args.edition:
            raise ValueError('Acquisition requires an exact configured year and edition; no fallback')
        inputs.append(('config', args.config.name, config_raw))
        source = replace(selected, metadata=metadata)
    else:
        if args.config is not None or args.config_sha256 is not None or args.year is not None:
            raise ValueError('Reference acquisition does not select a Config/year')
        if not args.dataset.strip():
            raise ValueError('Reference acquisition requires an explicit dataset label')
        source = ResolvedDataset(args.dataset, 'reference', None, None, '', metadata)
    reference_store = ProvenanceStore(args.reference_store) if args.reference_store is not None else archive
    identifiers = list(dict.fromkeys([*metadata.provenance_ids,
        *(identifier for ref in metadata.references for identifier in ref.provenance_ids)]))
    reference_store.verify_record_closure(identifiers)
    verify_reference_receipts(source, reference_store, check_local_paths=False, verify_all_claimed=True)
    for identifier in identifiers:
        archive.copy_record_from(reference_store, identifier)
    # Retain the actual captured contract inputs, not later rereads of mutable paths.
    with tempfile.TemporaryDirectory(prefix='teadata-acquire-inputs-') as directory:
        input_ids = []
        for role, filename, raw in inputs:
            staged = Path(directory) / role / filename
            staged.parent.mkdir()
            staged.write_bytes(raw)
            input_ids.append(archive.record_acquisition(staged, ResolvedDataset(
                'reference:acquire_' + role, 'reference', None, None, '',
                DatasetMetadata(title='Explicit SHA-pinned acquisition ' + role + ' input'))))
    source = replace(source, metadata=replace(metadata,
        provenance_ids=tuple([*metadata.provenance_ids, *input_ids])))
    archive.verify_record_closure(source.metadata.provenance_ids)
    verify_reference_receipts(source, archive, check_local_paths=False, verify_all_claimed=True)
    return source


def compare_inputs(args) -> dict:
    """Retain selected history and a new report, including blocked drift reports."""
    output = args.output
    if output.exists() or output.is_symlink():
        raise FileExistsError("Comparison output already exists; select a new observation path")
    archive = ProvenanceStore(args.store)
    inputs, selected, descriptions = [], [], {}
    secrets = [str(getattr(args, side + '_' + name)) for side in ['old', 'new']
               for name in ['store', 'schema', 'dictionary'] if getattr(args, side + '_' + name) is not None]
    if args.policy_config is not None:
        secrets.append(str(args.policy_config))
    policies = {'old': {}, 'new': {}}
    report = {'schema_version': 1, 'status': 'blocked_contract_drift', 'source_activation': False,
              'inputs': descriptions, 'toolchain': {'python': sys.version, 'pyarrow': version('pyarrow')}}
    phase = 'input_contracts'
    try:
        if args.old_policy or args.new_policy:
            if args.policy_config is None or args.policy_config_sha256 is None:
                raise ValueError("Field policies require an explicit SHA-pinned existing policy configuration")
            from teadata.teadata_config import load_config

            observation = archive.record_acquisition(args.policy_config, ResolvedDataset(
                'reference:caller_selected_field_policies', 'reference', None, None, '',
                DatasetMetadata(title='Explicit caller-selected field policy configuration; applicability unassessed')))
            inputs.append(observation)
            policy_artifact = archive.read_record(observation)['artifact']
            if policy_artifact['sha256'] != args.policy_config_sha256:
                raise ValueError("Policy configuration differs from the pinned bytes")
            cfg = load_config(archive.verify_artifact(policy_artifact['sha256']))
            for side in policies:
                for code, name in getattr(args, side + '_policy'):
                    if code in policies[side]:
                        raise ValueError("Repeated field policy selection is ambiguous")
                    policies[side][code] = cfg.value_policies[name]
            report['policy_configuration_sha256'] = args.policy_config_sha256
        for side in ['old', 'new']:
            store = ProvenanceStore(getattr(args, side + '_store'))
            identifier = getattr(args, side + '_record')
            record = store.read_record(identifier)
            if record['kind'] != 'acquisition':
                raise ValueError("Pinned raw source must be an acquisition observation")
            archive.copy_record_from(store, identifier)
            inputs.append(identifier)
            schema = getattr(args, side + '_schema')
            observation = archive.record_acquisition(schema, ResolvedDataset(
                getattr(args, side + '_dataset') + ':reviewed_crosswalk', 'reference', None, None, '',
                DatasetMetadata(title='Explicit SHA-pinned comparison crosswalk; validation recorded in report',
                                version=getattr(args, side + '_edition'))))
            inputs.append(observation)
            digest = archive.read_record(observation)['artifact']['sha256']
            descriptions[side] = {'record_id': identifier, 'artifact': record['artifact'],
                'original_dataset': record['dataset'], 'original_metadata': record['source_metadata'],
                'schema': {'filename': schema.name, 'sha256': digest,
                           'reviewed_sha256': getattr(args, side + '_schema_sha256')},
                'dataset': getattr(args, side + '_dataset'), 'edition': getattr(args, side + '_edition'),
                'table': getattr(args, side + '_table')}
            if digest != getattr(args, side + '_schema_sha256'):
                raise ValueError(side + ' crosswalk differs from the pinned resource SHA')
            raw = archive.verify_artifact(digest).read_bytes()
            document = parse_column_schema(raw)
            dataset, edition = descriptions[side]['dataset'], descriptions[side]['edition']
            tables = crosswalk_dataset(document, dataset, edition)['tables']
            choices = [t for t in tables if t['table'] == descriptions[side]['table']]
            if len(choices) != 1:
                raise ValueError(side + ' selected source table is unavailable or ambiguous')
            mapping = choices[0]
            table_inputs = table_input_sha256s(mapping)
            if len(table_inputs) != 1:
                raise ValueError(side + ' TAPR comparison requires a single raw input')
            if table_inputs[0] != record['artifact']['sha256']:
                raise ValueError(side + ' acquired revision differs from the reviewed crosswalk input')
            original_version = record['source_metadata'].get('version')
            if original_version != edition:
                raise ValueError(side + ' original source edition differs from the selected crosswalk')
            path = store.verify_artifact(record['artifact']['sha256'])
            secrets.extend([str(path), str(store.root)])
            dictionary = getattr(args, side + '_dictionary')
            if (dictionary is None) != (mapping.get('dictionary') is None):
                raise ValueError(side + ' explicit dictionary file must match its reviewed dictionary descriptor')
            if dictionary is not None:
                secrets.append(str(dictionary))
                observation = archive.record_acquisition(dictionary, ResolvedDataset(
                    dataset + ':dictionary', 'reference', None, None, '',
                    DatasetMetadata(title='Explicit comparison dictionary local observation')))
                inputs.append(observation)
                dictionary_sha = archive.read_record(observation)['artifact']['sha256']
                if dictionary_sha != mapping['dictionary']['sha256']:
                    raise ValueError(side + ' dictionary differs from the reviewed descriptor')
                dictionary = archive.verify_artifact(dictionary_sha)
                secrets.append(str(dictionary))
            scope = dataset.split('_', 1)[0]
            selected.append(read_tapr_table(path, mapping=mapping, key_field=scope + '_number',
                dataset=dataset, edition=edition, dictionary_path=dictionary))
        phase = 'comparison'
        report['comparison'] = compare_tapr_tables(*selected, key_field=args.scope + '_number',
            old_policies=policies['old'], new_policies=policies['new'], max_examples=args.max_examples)
        report['status'] = 'compared'
        report['keyed_comparison_completed'] = True
    except (OSError, ValueError, KeyError) as error:
        diagnostic = str(error)
        for value in sorted(secrets, key=len, reverse=True):
            diagnostic = diagnostic.replace(value, '[selected input]')
        report['diagnostic'] = diagnostic
        report['keyed_comparison_attempted'] = phase == 'comparison'
        report['keyed_comparison_completed'] = False
        if phase == 'comparison':
            report['status'] = 'blocked_comparison'
    if not inputs:
        raise ValueError(report.get('diagnostic', 'No valid pinned comparison inputs'))
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    report['implementation_sha256'] = {name: sha256((root / name).read_bytes()).hexdigest()
        for name in ['scripts/refresh_tapr.py', 'tapr.py', 'source_csv.py', 'values.py', 'column_schema.py',
                     'identifiers.py', 'provenance.py']}
    code = sha256(json.dumps(report['implementation_sha256'], sort_keys=True).encode()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='.tapr-comparison-', dir=output.parent) as directory:
        staged = Path(directory) / output.name
        staged.write_text(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + '\n')
        staged.chmod(0o600)
        processing = archive.record_processing(dataset='tapr_explicit_source_comparison', inputs=list(dict.fromkeys(inputs)),
            outputs={'comparison_report': staged}, processor='teadata.scripts.refresh_tapr', code_version=code,
            parameters={'scope': args.scope, 'no_config_activation': True, 'max_examples': args.max_examples},
            validation={'status': report['status'], 'scientific_equivalence_claimed': False})
        archive.verify_record_closure([processing])
        os.link(staged, output)  # Complete, verified publication; never replace an old report.
    return {'status': report['status'], 'processing_record_id': processing,
            'report_sha256': sha256(output.read_bytes()).hexdigest()}


def inspect_dictionary(args) -> dict:
    """Inspect one retained HTML artifact offline, preserving its exact history."""
    from teadata.tapr_dictionary import extract_tapr_dictionary

    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Dictionary output already exists; select a new observation path")
    if type(args.max_bytes) is not int or args.max_bytes < 1:
        raise ValueError('Dictionary max-bytes must be a positive integer')
    source, archive = ProvenanceStore(args.source_store), ProvenanceStore(args.store)
    record = source.read_record(args.record)
    if record['kind'] != 'acquisition':
        raise ValueError('Dictionary input must be a retained acquisition observation')
    artifact = record['artifact']
    expected_size = artifact['size_bytes']
    if type(expected_size) is not int or expected_size < 0:
        raise ValueError('Invalid retained dictionary size')
    if expected_size > args.max_bytes:
        raise ValueError('Retained dictionary exceeds max-bytes')
    selected = source.root / 'objects' / _identifier(artifact['sha256'])
    with selected.open('rb') as stream:
        if os.fstat(stream.fileno()).st_size > args.max_bytes:
            raise ValueError('Retained dictionary exceeds max-bytes')
        raw = stream.read(args.max_bytes + 1)
    if len(raw) > args.max_bytes:
        raise ValueError('Retained dictionary exceeds max-bytes')
    if len(raw) != expected_size or sha256(raw).hexdigest() != artifact['sha256']:
        raise ValueError('Parsed dictionary bytes differ from the selected acquisition')
    archive.copy_record_from(source, args.record)
    occurrences = extract_tapr_dictionary(raw, encoding=args.encoding)
    root = Path(__file__).resolve().parents[1]
    implementation = {name: sha256((root / name).read_bytes()).hexdigest()
                      for name in ['tapr_dictionary.py', 'scripts/refresh_tapr.py']}
    report = {'schema_version': 1, 'source_record_id': args.record,
              'source_artifact': artifact, 'encoding': args.encoding, 'max_bytes': args.max_bytes,
              'coordinate_semantics': 'Outer tooltip start tag; one-based line, zero-based decoded-character column',
              'text_projection': 'Decoded rendered text; presentation whitespace only is normalized. Raw artifact owns original bytes.',
              'dictionary_records': [row.column_record() for row in occurrences],
              'implementation_sha256': implementation, 'source_activation': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.tapr-dictionary-', dir=args.output.parent) as directory:
        staged = Path(directory) / args.output.name
        staged.write_text(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + '\n')
        staged.chmod(0o600)
        processing = archive.record_processing(dataset='tapr_explicit_dictionary_inspection', inputs=[args.record],
            outputs={'dictionary_occurrences': staged}, processor='teadata.scripts.refresh_tapr',
            code_version=sha256(json.dumps(implementation, sort_keys=True).encode()).hexdigest(),
            parameters={'encoding': args.encoding, 'max_bytes': args.max_bytes, 'no_config_activation': True},
            validation={'occurrences': len(occurrences), 'scientific_equivalence_claimed': False})
        archive.verify_record_closure([processing])
        os.link(staged, args.output)
    return {'status': 'inspected_dictionary', 'processing_record_id': processing,
            'report_sha256': sha256(args.output.read_bytes()).hexdigest()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    acquire = commands.add_parser('acquire', help='Explicit bounded public GET/POST acquisition; never change Config')
    acquire.add_argument('--url', required=True)
    acquire.add_argument('--dataset', required=True)
    acquire.add_argument('--edition', required=True)
    acquire.add_argument('--section', choices=['data_sources', 'reference'], default='data_sources')
    acquire.add_argument('--metadata', type=Path, required=True, help='Explicit DatasetMetadata JSON; never inherited from Config')
    acquire.add_argument('--metadata-sha256', required=True)
    acquire.add_argument('--config', type=Path, help='Data-source identity/year/version authority; required for data_sources')
    acquire.add_argument('--config-sha256')
    acquire.add_argument('--year', type=int)
    acquire.add_argument('--reference-store', type=Path, help='Existing selected reference closures; defaults to --store')
    acquire.add_argument('--no-redirects', action='store_true', help='Reject redirects without another HTTP request')
    acquire.add_argument('--filename', required=True)
    acquire.add_argument('--method', required=True, choices=['GET', 'POST'])
    acquire.add_argument('--field', action='append', nargs=2, default=[], metavar=('NAME', 'VALUE'))
    acquire.add_argument('--content-type')
    replay = commands.add_parser('replay', help='Repeat one retained public HTTP observation; do not infer missing controls')
    replay.add_argument('--record', required=True)
    replay.add_argument('--content-type', help='Explicit type for a legacy receipt that did not retain it')
    for command in [acquire, replay]:
        command.add_argument('--store', type=Path, required=True)
        command.add_argument('--timeout', type=float, default=20)
        command.add_argument('--max-bytes', type=int, default=32 * 1024 * 1024)
    dictionary = commands.add_parser('dictionary', help='Inspect one retained HTML dictionary offline; keep partial ordered occurrences')
    dictionary.add_argument('--source-store', type=Path, required=True)
    dictionary.add_argument('--record', required=True)
    dictionary.add_argument('--encoding', required=True, help='Explicit decoding of the retained publisher bytes')
    dictionary.add_argument('--max-bytes', type=int, default=2 * 1024 * 1024,
                            help='Positive retained HTML byte limit; default 2 MiB')
    dictionary.add_argument('--store', type=Path, required=True, help='Append-only output provenance archive')
    dictionary.add_argument('--output', type=Path, required=True, help='New occurrence report; never overwritten')
    compare = commands.add_parser('compare', help='Compare two pinned reviewed original inputs offline')
    compare.add_argument('--store', type=Path, required=True, help='Append-only report/provenance archive')
    compare.add_argument('--output', type=Path, required=True, help='New report path; never overwritten')
    compare.add_argument('--scope', required=True, choices=['campus', 'district'])
    compare.add_argument('--max-examples', type=int, default=20)
    for side in ['old', 'new']:
        for name in ['store', 'schema', 'dictionary']:
            compare.add_argument('--' + side + '-' + name, type=Path, required=name != 'dictionary')
        for name in ['record', 'schema-sha256', 'dataset', 'edition', 'table']:
            compare.add_argument('--' + side + '-' + name, required=True)
        compare.add_argument('--' + side + '-policy', action='append', nargs=2, default=[], metavar=('CODE', 'POLICY_ID'))
    compare.add_argument('--policy-config', type=Path)
    compare.add_argument('--policy-config-sha256')
    args = parser.parse_args(argv)
    if args.command == 'compare':
        result = compare_inputs(args)
    elif args.command == 'dictionary':
        result = inspect_dictionary(args)
    else:
        archive = ProvenanceStore(args.store)
        if args.command == 'replay':
            client = AcquisitionClient(archive, timeout=args.timeout, attempts=1, max_bytes=args.max_bytes, max_retry_wait=0)
            acquired = replay_acquisition(client, args.record, content_type=args.content_type)
        else:
            source = acquisition_source(args, archive)
            client = AcquisitionClient(archive, timeout=args.timeout, attempts=1,
                max_bytes=args.max_bytes, max_retry_wait=0,
                opener=build_opener(NoRedirect()) if args.no_redirects else None)
            acquired = acquire_form(client, args.url, source, filename=args.filename, method=args.method,
                                    fields=args.field, content_type=args.content_type)
            archive.verify_record_closure([acquired.record_id])
        record = archive.read_record(acquired.record_id)
        result = {'status': 'acquired', 'record_id': acquired.record_id, 'artifact': record['artifact']}
    print(json.dumps(result, sort_keys=True))
    return 0 if result['status'] in {'compared', 'acquired', 'inspected_dictionary'} else 2


if __name__ == '__main__':
    raise SystemExit(main())
