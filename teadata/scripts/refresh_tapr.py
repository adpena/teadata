"""Explicit append-only public acquisition/replay and pinned TAPR comparison.

No command changes source configuration or activates acquired/prepared data.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import tempfile

from teadata.acquisition import AcquisitionClient, acquire_form, replay_acquisition
from teadata.column_schema import crosswalk_dataset, parse_column_schema
from teadata.dataset_metadata import DatasetMetadata, ResolvedDataset
from teadata.provenance import ProvenanceStore
from teadata.tapr import compare_tapr_tables, read_tapr_table


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
            if mapping['input_sha256'] != record['artifact']['sha256']:
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
        for name in ['scripts/refresh_tapr.py', 'tapr.py', 'values.py', 'column_schema.py',
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    acquire = commands.add_parser('acquire', help='Explicit bounded public GET/POST acquisition; never change Config')
    acquire.add_argument('--url', required=True)
    acquire.add_argument('--dataset', required=True)
    acquire.add_argument('--edition', required=True)
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
    else:
        archive = ProvenanceStore(args.store)
        client = AcquisitionClient(archive, timeout=args.timeout, attempts=1, max_bytes=args.max_bytes, max_retry_wait=0)
        if args.command == 'replay':
            acquired = replay_acquisition(client, args.record, content_type=args.content_type)
        else:
            source = ResolvedDataset(args.dataset, 'data_sources', None, None, '',
                                     DatasetMetadata(version=args.edition))
            acquired = acquire_form(client, args.url, source, filename=args.filename, method=args.method,
                                    fields=args.field, content_type=args.content_type)
        record = archive.read_record(acquired.record_id)
        result = {'status': 'acquired', 'record_id': acquired.record_id, 'artifact': record['artifact']}
    print(json.dumps(result, sort_keys=True))
    return 0 if result['status'] in {'compared', 'acquired'} else 2


if __name__ == '__main__':
    raise SystemExit(main())
