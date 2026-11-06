"""Offline exact raw parity for explicitly selected admitted native leaves.

This checks transport/preparation fidelity, not scientific meanings or publisher
authenticity. Unsupported transformations and unavailable lineage remain HOLD.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
from io import TextIOWrapper
from pathlib import Path
from tempfile import TemporaryFile
from typing import TYPE_CHECKING, Any, NotRequired, Sequence, TypedDict

from .column_schema import crosswalk_dataset, parse_column_schema, table_input_sha256s
from .dataset_metadata import COLUMN_SCHEMA_MAX_BYTES, STATEWIDE_SINGLETON_GRAIN, EditionKey
from .manifest import ContractError, canonical_json
from .peims_leaves import DATASETS as PEIMS_DATASETS
from .provenance import validate_record_graph
from .source_csv import SourceLines
from .tapr import TAPR_KEY_WIDTHS, TAPR_SOURCE_KEYS, TaprReadBudget, TaprSourceRequest, qualified_tapr_family, reviewed_tapr_data_record_count, tapr_source_identifiers

if TYPE_CHECKING:
    from .provenance import ProvenanceStore
    from .release import DataRelease


class _Uncovered(ValueError):
    """The finite checker has no admitted replay contract for this selection."""


class _FieldReport(TypedDict):
    checked: int
    mismatches: int
    examples: list[dict[str, Any]]
    examples_truncated: NotRequired[bool]


@dataclass(frozen=True)
class _Selection:
    key: EditionKey
    mapping: dict[str, Any]
    raw: Path
    prepared: Path
    leaf: Path
    key_field: str | None
    budget: TaprReadBudget
    dictionary: Path | None
    lineage: dict[str, Any]


def _selection(release: DataRelease, store: ProvenanceStore, key: EditionKey) -> _Selection:
    variables = list(release.catalog.variables(edition=key))
    plans = release.catalog.resolve_artifacts((key, row['field']) for row in variables)
    if len(plans) != 1 or not variables:
        raise _Uncovered('Only one complete lossless native leaf per selected edition is supported')
    ref = plans[0]['artifact']
    if ref.column_support is not None:
        raise _Uncovered('Typed/normalized or technical-support leaves are not covered by raw-string parity')
    graph = store.verify_record_closure([ref.processing_id])
    leaf_record = graph[ref.processing_id]
    parameters = leaf_record.get('parameters', {})
    if not isinstance(parameters, dict) or not isinstance(parameters.get('recipe'), dict):
        raise _Uncovered('Recorded native replay parameters are unavailable')
    recipe = parameters['recipe']
    if (leaf_record.get('processor') != 'teadata.scripts.build_tapr_profiles'
            or recipe.get('operation') != 'lossless_admitted_source_row_groups'
            or recipe.get('values_interpreted') is not False):
        raise _Uncovered('No supported recorded lossless TAPR replay recipe')
    if (parameters.get('leaf_contract_sha256') != ref.identity
            or not any(item['sha256'] == ref.sha256 and item['size_bytes'] == ref.size_bytes
                       for item in leaf_record['outputs'].values())):
        raise ValueError('Leaf processing does not bind the selected descriptor')
    candidates = [identifier for identifier in leaf_record['inputs']
                  if graph[identifier]['kind'] == 'processing' and graph[identifier].get('dataset') == key.dataset
                  and 'prepared_source' in graph[identifier].get('outputs', {})]
    if len(candidates) != 1:
        raise _Uncovered('A unique complete upstream preparation is unavailable')
    preparation_id, = candidates
    prepared, contract = graph[preparation_id], ref.contract
    validation, prior = prepared.get('validation', {}), prepared.get('parameters', {})
    if not isinstance(validation, dict) or not isinstance(prior, dict):
        raise _Uncovered('Recorded preparation validation/parameters are unavailable')
    if (prepared.get('processor') != 'teadata.scripts.build_tapr_profiles'
            or validation.get('raw_string_readback_equal') is not True
            or validation.get('all_historical_source_rows_retained') is not True):
        raise _Uncovered('Complete historical raw preparation is not recorded')
    if (prior.get('edition') != key.edition or parameters.get('edition') != key.edition
            or recipe.get('source_read_budget') != prior.get('source_read_budget')
            or recipe.get('preparation_code_sha256') != prior.get('preparation_code_sha256')):
        raise ValueError('Historical preparation edition, reader budget or recorded code binding differs')
    if (not isinstance(recipe.get('source_read_budget'), dict)
            or not isinstance(recipe.get('preparation_code_sha256'), str)
            or len(recipe['preparation_code_sha256']) != 64
            or not isinstance(prepared['outputs'].get('crosswalk'), dict)):
        raise _Uncovered('Complete historical reader budget/code/crosswalk binding is unavailable')
    budget = TaprReadBudget(**recipe['source_read_budget'])
    schema = prepared['outputs']['crosswalk']
    if prior.get('column_schema_sha256') != schema['sha256'] or schema['size_bytes'] > COLUMN_SCHEMA_MAX_BYTES:
        raise ValueError('Preparation crosswalk descriptor differs from its recorded contract')
    document = parse_column_schema(store.verify_artifact(schema['sha256']).read_bytes())
    branch = crosswalk_dataset(document, key.dataset, key.edition)
    choices = [item for item in branch['tables'] if item['table'] == key.table]
    if len(choices) != 1:
        raise _Uncovered('Selected source table is unavailable or ambiguous')
    mapping = choices[0]
    if sha256(canonical_json(mapping).encode()).hexdigest() != contract['mapping_sha256']:
        raise ValueError('Prepared crosswalk and final leaf mapping differ')
    hashes = table_input_sha256s(mapping)
    if len(hashes) != 1 or key.scope not in TAPR_SOURCE_KEYS:
        raise _Uncovered('Only single-original TAPR tables are covered')
    key_field = TAPR_SOURCE_KEYS[key.scope]
    if key_field is None and key.grain != STATEWIDE_SINGLETON_GRAIN:
        raise _Uncovered('Statewide TAPR parity requires the reviewed singleton grain')
    qualified_tapr_family(key.dataset, key_field)
    names = [item['field'] for item in mapping['fields']]
    if (list(contract['columns']) != names or list(contract['key_columns']) != ([key_field] if key_field else [])
            or {row['field'] for row in variables} != set(names)
            or validation.get('fields') != len(names) or validation.get('source_columns_accounted') != len(names)
            or validation.get('rows') != contract['rows']):
        raise ValueError('Complete field, row or physical-key coverage differs')
    prepared_artifact = prepared['outputs']['prepared_source']
    if list(contract['inputs']) != [hashes[0], prepared_artifact['sha256']]:
        raise ValueError('Leaf does not retain its exact raw and prepared input hashes')
    acquisitions = release.catalog.acquisitions(key.dataset, edition=key)
    if not acquisitions:
        raise _Uncovered('Selected original acquisition lineage is unavailable')
    # Equal bytes can have several acquisition receipts. Only the selected
    # preparation's upstream graph identifies the receipt actually bound to it.
    upstream = validate_record_graph(graph.__getitem__, [preparation_id])
    for claimed in acquisitions:
        identifier = claimed['record_id']
        observed = upstream.get(identifier)
        if observed is None:
            raise ValueError('Catalog acquisition is outside the preparation closure')
        if (observed['kind'] != 'acquisition'
                or observed != {name: value for name, value in claimed.items() if name != 'record_id'}
                or observed['artifact']['sha256'] != hashes[0]
                or observed['source_metadata'].get('version') != key.edition):
            raise ValueError('Catalog acquisition and exact source receipt differ')
    dictionary = mapping.get('dictionary')
    dictionary_path = store.verify_artifact(dictionary['sha256']) if dictionary else None
    return _Selection(key, mapping, store.verify_artifact(hashes[0]),
        store.verify_artifact(prepared_artifact['sha256']), release.artifact(ref.identity), key_field, budget,
        dictionary_path, {'leaf_processing_id': ref.processing_id, 'raw_sha256': hashes[0],
            'prepared_sha256': prepared_artifact['sha256'], 'leaf_sha256': ref.sha256,
            'crosswalk_sha256': schema['sha256'], 'acquisitions': acquisitions})


def _compare(selection: _Selection, *, max_examples: int, working_bytes: int) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.compute as arrow_compute
    import pyarrow.parquet as pq

    pc: Any = arrow_compute
    names = [field['field'] for field in selection.mapping['fields']]
    # Three explicit retained-buffer slots: original, one target, alignment/scratch.
    # Canonical source decoding has its separate recorded batch/key budgets.
    slot = working_bytes // 3
    batches, size = [], 0
    request = TaprSourceRequest(selection.raw, selection.mapping, selection.key_field,
                               selection.key.dataset, selection.key.edition, selection.dictionary)
    with request.open(budget=selection.budget) as reader:
        for batch in reader:
            size += batch.nbytes
            if size > slot:
                raise _Uncovered('Original table exceeds its explicit working-byte slot')
            batches.append(batch)
        summary = reader.summary
    if not batches:
        raise _Uncovered('No original raw rows are available for parity coverage')
    raw = pa.Table.from_batches(batches)
    del batches, batch
    key_field = selection.key_field
    width = TAPR_KEY_WIDTHS.get(key_field or '', 0)
    raw_index_bound = 0
    raw_keys = raw_set = None
    if key_field is not None:
        raw_index_bound = (width + 10) * (raw.num_rows + 1)
        if 2 * raw_index_bound > slot:
            raise _Uncovered('Raw alignment buffers exceed their explicit working-byte slot')
        raw_keys = tapr_source_identifiers(raw[key_field], key_field=key_field)
        if raw_keys.nbytes > selection.budget.key_bytes:
            raise _Uncovered('Raw identity index exceeds its recorded byte budget')
        raw_set = raw_keys.combine_chunks()
    examples_used = 0

    def compare_stage(stage, path):
        nonlocal examples_used
        state = path.stat()
        signature = (state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns)
        batches, size = [], 0
        with pq.ParquetFile(path) as native:
            schema = native.schema_arrow.remove_metadata()
            if schema.names != names or any(not (pa.types.is_string(field.type) or pa.types.is_large_string(field.type)) for field in schema):
                raise ValueError(stage + ' does not have the complete raw-string schema')
            # The producer admits row groups by bytes as well as rows. A single
            # iterator can combine several groups within the row limit.
            for index in range(native.metadata.num_row_groups):
                for batch in native.iter_batches(batch_size=selection.budget.batch_rows,
                                                 row_groups=[index], use_threads=False):
                    size += batch.nbytes
                    if batch.nbytes > selection.budget.batch_bytes or size > slot:
                        raise _Uncovered(stage + ' table exceeds its explicit native working-byte slot')
                    batches.append(batch)
        actual = pa.Table.from_batches(batches, schema=schema)
        del batches
        scratch_base = 0
        alignment = None
        keyset = {'raw_only': 0, 'target_only': 0}
        if key_field is None:
            if raw.num_rows != 1 or actual.num_rows != 1:
                raise ValueError(stage + ' differs from the one-row statewide grain')
        else:
            scratch_base = 2 * raw_index_bound + 2 * (width + 10) * (actual.num_rows + 1) + 8 * raw.num_rows
            if scratch_base > slot:
                raise _Uncovered(stage + ' alignment exceeds its explicit working-byte slot')
            keys = tapr_source_identifiers(actual[key_field], key_field=key_field)
            if pc.any(pc.is_null(keys)).as_py():
                raise ValueError(stage + ' has malformed source identities')
            if keys.nbytes > selection.budget.key_bytes:
                raise _Uncovered(stage + ' identity index exceeds its recorded byte budget')
            if pc.count_distinct(keys).as_py() != len(keys):
                raise ValueError(stage + ' has duplicate source identities')
            # One native index for the stage; no per-batch predicate rescans.
            alignment = pc.index_in(raw_keys, value_set=keys.combine_chunks())
            keyset = {'raw_only': alignment.null_count,
                      'target_only': pc.sum(pc.cast(pc.invert(pc.is_in(keys, value_set=raw_set)), pa.int64())).as_py() or 0}
        fields: dict[str, _FieldReport] = {}
        for name in names:
            # Conservative retained gather/mask/example buffer admission. This is
            # not an allocator/RSS promise; the tables themselves occupy two slots.
            if scratch_base + 2 * raw[name].nbytes + 3 * actual[name].nbytes + 32 * (raw.num_rows + 1) > slot:
                raise _Uncovered(stage + ' column comparison exceeds its explicit working-byte slot')
            values = actual[name] if alignment is None else pc.take(actual[name], alignment)
            mask = pc.fill_null(pc.not_equal(raw[name], values), True)
            count = pc.sum(pc.cast(mask, pa.int64())).as_py() or 0
            item: _FieldReport = {'checked': raw.num_rows, 'mismatches': count, 'examples': []}
            remaining = max_examples - examples_used
            if count and remaining > 0:
                indices = pc.indices_nonzero(mask).slice(0, remaining)
                identity = {'source_key': raw_keys} if raw_keys is not None else {'row_number': [1]}
                examples = pa.table({**identity, 'raw': raw[name], 'actual': values}).take(indices)
                for example in examples.to_pylist():
                    for column in ('raw', 'actual'):
                        value = example[column]
                        example[column + '_characters'] = len(value) if value is not None else None
                        example[column + '_truncated'] = value is not None and len(value) > 1024
                        if example[column + '_truncated']:
                            example[column] = value[:1024]
                    item['examples'].append(example)
                    examples_used += 1
            item['examples_truncated'] = count > len(item['examples'])
            fields[name] = item
            del values, mask
        state = path.stat()
        if signature != (state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns):
            raise ValueError(stage + ' changed during source comparison')
        return fields, keyset

    diagnostics: dict[str, dict[str, _FieldReport]] = {}
    keysets: dict[str, dict[str, int]] = {}
    for stage, path in [('prepared', selection.prepared), ('artifact', selection.leaf)]:
        diagnostics[stage], keysets[stage] = compare_stage(stage, path)
    mismatches = sum(field['mismatches'] for fields in diagnostics.values() for field in fields.values())
    passed = not mismatches and not any(count for stage in keysets.values() for count in stage.values())
    return {'status': 'PASS_RAW_PARITY' if passed else 'FAIL_MISMATCH', 'rows': summary.rows,
        'fields': len(names), 'raw_cells': summary.rows * len(names), 'compared_cells': sum(
            field['checked'] for fields in diagnostics.values() for field in fields.values()),
        'mismatches': mismatches, 'keysets': keysets, 'stages': diagnostics, 'lineage': selection.lineage,
        'working_bytes': working_bytes, 'working_budget_scope': 'Retained Arrow tables and alignment/comparison buffers; canonical source decoding separately uses recorded batch/key budgets, not a process RSS guarantee'}


def _compare_csv_export(release: DataRelease, selection: _Selection) -> dict[str, Any]:
    layout = selection.mapping["layout"]
    fields = selection.mapping["fields"]
    rows = 0
    with TemporaryFile(mode="w+b") as output, selection.raw.open("rb") as source:
        release.to_csv(selection.key, output, restore_original_headers=True)
        output.seek(0)
        with TextIOWrapper(output, encoding="utf-8", newline="") as text:
            exported = csv.reader(text, strict=True)
            for kind in ("label", "code"):
                if next(exported, None) != [field[kind] for field in fields]:
                    raise ValueError(f"CSV export differs from the reviewed {kind} header")
            lines = SourceLines(source, selection.budget.record_bytes, "CSV parity source record exceeds its reviewed budget")
            for number, row in enumerate(csv.reader(lines, strict=True), 1):
                lines.record.clear()
                if not layout["first_data_record"] <= number <= layout["last_data_record"]:
                    continue
                if any(note["first_record"] <= number <= note["last_record"] for note in layout["annotations"]):
                    continue
                if next(exported, None) != row:
                    raise ValueError(f"CSV export differs at original logical record {number}")
                rows += 1
            if next(exported, None) is not None or rows != reviewed_tapr_data_record_count(selection.mapping):
                raise ValueError("CSV export and source row counts differ")
            if lines.digest.hexdigest() != selection.mapping["input_sha256"]:
                raise ValueError("Original CSV bytes changed during export comparison")
    return {"status": "PASS_EXACT_CSV_EXPORT", "rows": rows, "cells": rows * len(fields), "header_records": 2}


def validate_source_parity(release: DataRelease, store: ProvenanceStore, *, edition_ids: Sequence[str],
                           max_examples: int = 5, working_bytes: int = 128 * 1024**2,
                           verify_export: bool = False) -> dict[str, Any]:
    """Check every raw cell of each explicit supported edition; no waivers.

    Metadata/integrity-only checks and raw fidelity do not establish mapping,
    period, unit, value-policy applicability, publisher correctness or finality.
    """
    if (not edition_ids or len(set(edition_ids)) != len(edition_ids)
            or type(max_examples) is not int or not 0 <= max_examples <= 20
            or type(working_bytes) is not int or not 0 < working_bytes <= 1024**3
            or type(verify_export) is not bool):
        raise ValueError('Explicit unique edition IDs and bounded example/working-byte budgets are required')
    output: dict[str, Any] = {'scope': 'admitted lossless TAPR, typed PEIMS and typed district transfer leaves', 'status': 'HOLD', 'editions': [],
        'working_bytes': working_bytes,
        'scientific_qualification': False, 'publisher_authenticity_verified': False,
        'exceptions_supported': False, 'semantic_goldens_replaced': False}
    unsupported = None
    try:
        choices = {row['edition_id']: EditionKey.from_dict({key: row[key] for key in EditionKey.__dataclass_fields__})
                   for row in release.catalog.editions(include_metadata=False)}
    except ContractError as error:
        choices, unsupported = {}, str(error)
    for identifier in edition_ids:
        item: dict[str, Any] = {'edition_id': identifier}
        try:
            if identifier not in choices:
                raise _Uncovered(unsupported or 'Exact edition is not present in this release')
            key = choices[identifier]
            item['edition'] = key.to_dict()
            if key.dataset in (*PEIMS_DATASETS, "district_transfer_reports"):
                if verify_export:
                    raise _Uncovered("Original-header export comparison requires a complete raw TAPR edition")
                from .source_parity_typed import compare_typed

                result = compare_typed(release, store, key, max_examples=max_examples, working_bytes=working_bytes)
            else:
                selection = _selection(release, store, key)
                result = _compare(selection, max_examples=max_examples, working_bytes=working_bytes)
            item.update(result)
            if verify_export and item['status'] == 'PASS_RAW_PARITY':
                try:
                    item['csv_export'] = _compare_csv_export(release, selection)
                except (ContractError, OSError, ValueError, KeyError, TypeError, csv.Error) as error:
                    item.update(status='FAIL_MISMATCH', raw_parity_status='PASS_RAW_PARITY',
                                csv_export={'status': 'FAIL_MISMATCH', 'diagnostic': str(error)})
        except (_Uncovered, FileNotFoundError) as error:
            item.update(status='HOLD_UNCOVERED', diagnostic=str(error))
        except (ContractError, OSError, ValueError, KeyError, TypeError) as error:
            item.update(status='FAIL_MISMATCH', diagnostic=str(error))
        output['editions'].append(item)
    output['status'] = 'PASS_RAW_PARITY' if all(item['status'] == 'PASS_RAW_PARITY' for item in output['editions']) else 'HOLD'
    output['coverage'] = {'requested_editions': len(edition_ids),
        'checked_editions': sum(item['status'] == 'PASS_RAW_PARITY' for item in output['editions']),
        'uncovered_editions': sum(item['status'] == 'HOLD_UNCOVERED' for item in output['editions']),
        'failed_editions': sum(item['status'] == 'FAIL_MISMATCH' for item in output['editions']),
        'raw_cells': sum(item.get('raw_cells', 0) for item in output['editions']),
        'compared_cells': sum(item.get('compared_cells', 0) for item in output['editions'])}
    return output
