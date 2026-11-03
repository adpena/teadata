"""Lossless TAPR source preparation from reviewed edition/header resources.

CSV record classification is structural. Arrow decodes fact columns and
merges whole native tables; scientific token interpretation stays in values.py.
"""

from __future__ import annotations

import csv
from collections.abc import Generator, Iterator
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import io
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Mapping

from .column_schema import table_input_sha256s, validate_header_table, validate_source_layout
from .identifiers import canonical_source_identifiers
from .source_csv import SourceLines

if TYPE_CHECKING:
    from .values import ValuePolicy


TAPR_PROFILE_FAMILIES = ("student_staff_profile", "graduation_profile", "student_profile", "staff_profile",
                         "reference", "dropout_attendance", "four_year_longitudinal_rate", "staar_school_progress",
                         "annual_rhsp_dap_fhsp_e_fhsp_dla_texas_first_dla",
                         "longitudinal_rhsp_dap_fhsp_e_fhsp_dla_texas_first_dla",
                         "5_year_extended_longitudinal_rate", "6_year_extended_longitudinal_rate")


def qualified_tapr_family(dataset: str, key_field: str) -> str:
    """Resolve only the reader's existing exact scope-qualified family names."""
    if key_field in {"campus_number", "district_number"}:
        for family in TAPR_PROFILE_FAMILIES:
            if dataset == key_field.removesuffix("_number") + "_tapr_" + family:
                return family
    raise ValueError("TAPR preparation requires an explicit qualified dataset and source edition")


def tapr_source_selection(options: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the canonical reader_options result before source or leaf I/O."""
    selection = options.get("tapr_source")
    if (not isinstance(selection, dict) or set(selection) != {"scope", "tables"}
            or selection["scope"] not in {"campus", "district"}
            or not isinstance(selection["tables"], dict) or not selection["tables"]
            or any(not isinstance(value, str) or not value or "://" in value
                   for value in selection["tables"].values())):
        raise ValueError("Full TAPR source profile requires an explicit scope and local table paths")
    return selection


def compare_tapr_tables(old: TaprSourceTable, new: TaprSourceTable, *, key_field: str,
                        old_policies: Mapping[str, ValuePolicy] | None = None,
                        new_policies: Mapping[str, ValuePolicy] | None = None,
                        max_examples: int = 20) -> dict[str, Any]:
    """Compare pinned source observations by exact publisher code and strict key.

    Code alignment is descriptive, never a rename, equivalence or default-source
    binding. Native kernels own joins, lexical differences, decimal equality and
    aggregation; only fields and bounded summaries are visited in Python.
    """
    import pyarrow as pa
    import pyarrow.compute as arrow_compute
    from .values import source_token_statuses

    pc: Any = arrow_compute
    if key_field not in {"campus_number", "district_number"} or type(max_examples) is not int or max_examples < 0:
        raise ValueError("Comparison needs a strict source key and nonnegative example bound")
    scope = key_field.removesuffix("_number")
    policies = [dict(old_policies or {}), dict(new_policies or {})]
    fields, identities = [], []
    for source, selected_policy in zip([old, new], policies, strict=True):
        validate_header_table(source.mapping)
        if len(table_input_sha256s(source.mapping)) != 1:
            raise ValueError("TAPR source comparison requires a single raw input per physical table")
        if (not source.dataset.startswith(scope + "_tapr_")
                or source.table.column_names != [f["field"] for f in source.mapping["fields"]]
                or key_field not in source.table.column_names
                or any(not pa.types.is_string(c.type) or c.null_count for c in source.table.columns)):
            raise ValueError("Comparison requires complete exact-string source tables in one scope")
        definitions = {f["code"]: f for f in source.mapping["fields"]}
        if set(selected_policy) - {code for code, f in definitions.items() if f.get("disposition", "fact") != "identity"}:
            raise ValueError("Policies must explicitly select actual non-identity publisher codes")
        fields.append(definitions)
        identity = canonical_source_identifiers(source.table[key_field], width=9 if scope == "campus" else 6)
        if pc.any(pc.is_null(identity)).as_py() or pc.count_distinct(identity).as_py() != source.table.num_rows:
            raise ValueError("Comparison cannot skip missing or duplicate source keys")
        identities.append(identity)
    common = [code for code in fields[0] if code in fields[1]]
    left = old.table.select([fields[0][c]["field"] for c in common]).rename_columns(['a' + str(i) for i in range(len(common))]).append_column('__key', identities[0])
    right = new.table.select([fields[1][c]["field"] for c in common]).rename_columns(['b' + str(i) for i in range(len(common))]).append_column('__key', identities[1])
    joined = left.join(right, keys='__key', join_type='inner')
    def count(mask):
        return pc.sum(pc.cast(pc.fill_null(mask, False), pa.int64())).as_py() or 0
    def key_difference(a, b):
        difference = pa.table({'key': a}).join(pa.table({'key': b}), keys='key', join_type='left anti').sort_by([('key', 'ascending')])
        return {'count': difference.num_rows, 'examples': difference.slice(0, max_examples)['key'].to_pylist(),
                'all_keys_emitted': difference.num_rows <= max_examples}
    def classes(values):
        result = pc.if_else(pc.match_substring_regex(values, r'^[+-]?[0-9]+(\.[0-9]+)?$'), 'decimal_lexeme', 'other_text')
        for token, label in [('', 'blank'), ('0', 'zero'), ('.', 'literal_dot'), ('-1', 'literal_-1'),
                             ('-2', 'literal_-2'), ('-3', 'literal_-3'), ('•', 'literal_bullet'), ('n/a', 'literal_n/a')]:
            result = pc.if_else(pc.equal(values, token), label, result)
        return result
    results = []
    for i, code in enumerate(common):
        a, b = joined['a' + str(i)], joined['b' + str(i)]
        changed = pc.not_equal(a, b)
        statuses = [source_token_statuses(values, selected[code]) if code in selected
                    else pa.repeat('unbound', len(values)) for values, selected in zip([a, b], policies, strict=True)]
        eligible = pc.and_(pc.match_substring_regex(a, r'^[+-]?[0-9]{1,38}(\.[0-9]{1,38})?$'),
                           pc.match_substring_regex(b, r'^[+-]?[0-9]{1,38}(\.[0-9]{1,38})?$'))
        # Unbound equality describes decimal spelling only, not measurements.
        # Only the selected canonical policy can exclude scientific statuses.
        for status, selected in zip(statuses, policies, strict=True):
            if code in selected:
                eligible = pc.and_(eligible, pc.equal(status, 'observed'))
        decimal_a = pc.cast(pc.if_else(eligible, a, None), pa.decimal256(76, 38), safe=True)
        decimal_b = pc.cast(pc.if_else(eligible, b, None), pa.decimal256(76, 38), safe=True)
        equal = pc.fill_null(pc.equal(decimal_a, decimal_b), False)
        decimal_state = pc.if_else(eligible, pc.if_else(equal, 'equal', 'unequal'), 'unclassified')
        changes = pc.filter(pa.table({'entity_key': joined['__key'], 'old_raw': a, 'new_raw': b, 'old_class': classes(a), 'new_class': classes(b),
            'old_status': statuses[0], 'new_status': statuses[1], 'decimal_lexeme_equivalence': decimal_state}), changed)
        transition_keys = ['old_class', 'new_class', 'old_status', 'new_status', 'decimal_lexeme_equivalence']
        transitions = changes.group_by(transition_keys).aggregate([('old_raw', 'count')]).sort_by([(k, 'ascending') for k in transition_keys])
        pairs = changes.group_by(['old_raw', 'new_raw', 'old_status', 'new_status', 'decimal_lexeme_equivalence']).aggregate([('old_class', 'count')]).sort_by([('old_raw', 'ascending'), ('new_raw', 'ascending')])
        results.append({'code': code, 'old_header': dict(fields[0][code]), 'new_header': dict(fields[1][code]),
            'header_equal': all(fields[0][code].get(k) == fields[1][code].get(k) for k in ['position', 'code', 'label']),
            'canonical_field_equal': fields[0][code]['field'] == fields[1][code]['field'],
            'dictionary_definition_equal': all(fields[0][code].get(k) == fields[1][code].get(k)
                for k in ['publisher_type', 'publisher_length', 'publisher_description', 'dictionary_status']),
            'overlap_cells': joined.num_rows,
            'lexical_changes': changes.num_rows, 'decimal_equal_changes': count(pc.and_(changed, pc.and_(eligible, equal))),
            'decimal_unequal_changes': count(pc.and_(changed, pc.and_(eligible, pc.invert(equal)))),
            'unclassified_changes': count(pc.and_(changed, pc.invert(eligible))),
            'changed_observations': {'count': changes.num_rows,
                'examples': changes.sort_by([('entity_key', 'ascending')]).slice(0, max_examples).to_pylist(),
                'truncated': changes.num_rows > max_examples},
            'raw_class_status_transitions': transitions.to_pylist(), 'distinct_raw_transition_pairs': pairs.num_rows,
            'raw_transition_examples': pairs.slice(0, max_examples).to_pylist(), 'all_raw_pairs_emitted': pairs.num_rows <= max_examples})
    return {'status': 'exact_reviewed_sources_compared', 'scope': scope, 'key_field': key_field,
        'old': {'dataset': old.dataset, 'edition': old.edition, 'input_sha256': old.input_sha256,
                'table': old.mapping['table'], 'rows': old.table.num_rows, 'columns': old.table.num_columns},
        'new': {'dataset': new.dataset, 'edition': new.edition, 'input_sha256': new.input_sha256,
                'table': new.mapping['table'], 'rows': new.table.num_rows, 'columns': new.table.num_columns},
        'keysets': {'overlap': joined.num_rows, 'old_only': key_difference(identities[0], identities[1]),
                    'new_only': key_difference(identities[1], identities[0]), 'raw_order_equal': identities[0].equals(identities[1])},
        'source_structure': {side: {'layout': source.mapping.get('layout'), 'annotations': list(source.annotations),
            'metric_period': source.mapping.get('metric_period', {'kind': 'unknown'}),
            'dictionary': source.mapping.get('dictionary'), 'dictionary_records': source.mapping.get('dictionary_records')}
            for side, source in [('old', old), ('new', new)]},
        'columns': {'old_only': [dict(f) for c, f in fields[0].items() if c not in fields[1]],
                    'new_only': [dict(f) for c, f in fields[1].items() if c not in fields[0]], 'common': results},
        'selected_field_policies': {side: {code: policy.to_dict() for code, policy in selected.items()}
                                   for side, selected in zip(['old', 'new'], policies, strict=True)},
        'limits': ['Exact publisher-code alignment is not scientific equivalence, a rename or methodology/comparability review.',
            'Absent source columns are schema absence, not null/missing/masked/NA observations or retirement. All historical members remain independent.',
            'Default statuses are unbound; an explicit caller-selected field policy does not establish applicability. No units/causes are inferred.',
            'Unbound decimal lexeme equality is a spelling relation, not measured-value equality or arithmetic. Selected non-observed statuses are excluded; no sign/token is a universal masking policy.',
            'Decimal equality uses no floats or rounding, within 38 integer and 38 fractional digits. Other spellings are unclassified.',
            'Changed-observation examples are ordered by canonical entity key within each publisher code; examples are bounded, with exact totals and truncation flags.',
            'Transition/key examples are bounded; exact totals and original inputs are retained. No currentGIS filtering, coalescing or Config activation.']}


@dataclass(frozen=True)
class TaprSourceTable:
    table: Any
    path: Path
    input_sha256: str
    dataset: str
    edition: str
    annotations: tuple[dict[str, Any], ...]
    mapping: Mapping[str, Any]


@dataclass(frozen=True)
class TaprReadBudget:
    """Admission limits for source buffers, not a native allocator/RSS promise.

    Batch admission includes raw UTF8 bytes and all string offsets. Metadata
    charges retained text plus64 bytes per structural cell. The narrow global
    identity/record index is separately bounded before native conversion.
    """

    batch_bytes: int = 8 * 1024 * 1024
    record_bytes: int = 1024 * 1024
    metadata_bytes: int = 8 * 1024 * 1024
    key_bytes: int = 16 * 1024 * 1024
    batch_rows: int = 4096

    def __post_init__(self):
        if any(type(value) is not int or value < 1 or value > 1024**3
               for value in vars(self).values()):
            raise ValueError("TAPR byte/row budgets require positive bounded integers")


@dataclass(frozen=True)
class TaprSourceRequest:
    """One explicit source selection, with no I/O at construction."""

    path: Path
    mapping: Mapping[str, Any]
    key_field: str
    dataset: str
    edition: str
    dictionary_path: Path | None = None

    def open(self, *, budget: TaprReadBudget | None = None) -> TaprSourceReader:
        if budget is not None and not isinstance(budget, TaprReadBudget):
            raise ValueError("TAPR source requires a typed read budget")
        return TaprSourceReader(self, budget=budget if budget is not None else TaprReadBudget())


@dataclass(frozen=True)
class TaprSourceSummary:
    path: Path
    input_sha256: str
    dataset: str
    edition: str
    annotations: tuple[dict[str, Any], ...]
    mapping: Mapping[str, Any]
    rows: int
    source_stat: tuple[int, ...]


def _source_stat(path: Path) -> tuple[int, ...]:
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


class TaprSourceReader(Iterator[Any]):
    """One canonical source scanner; only a fully exhausted scan has a summary.

    Batches are provisional until EOF validates source identity, global keys,
    headers and annotations. Close early/exceptionally and no summary exists.
    """

    def __init__(self, request: TaprSourceRequest, *, budget: TaprReadBudget):
        qualified_tapr_family(request.dataset, request.key_field)
        if not isinstance(request.edition, str) or not request.edition.strip():
            raise ValueError("TAPR preparation requires an explicit qualified dataset and source edition")
        self.request = request
        self.mapping = deepcopy(request.mapping)
        validate_header_table(self.mapping)
        inputs = table_input_sha256s(self.mapping)
        if len(inputs) != 1:
            raise ValueError("TAPR physical source reader requires a single raw input")
        self.input_sha256 = inputs[0]
        try:
            validate_tapr_key(self.mapping, request.key_field)
        except ValueError as error:
            raise ValueError(f"{request.path}: {error}") from error
        self.budget = budget
        self._summary: TaprSourceSummary | None = None
        self._iterator = self._read()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._iterator.close()

    def __next__(self):
        return next(self._iterator)

    @property
    def summary(self) -> TaprSourceSummary:
        if self._summary is None:
            raise ValueError("TAPR source has not completed validated EOF")
        return self._summary

    def _read(self) -> Generator[Any, None, None]:
        import pyarrow as pa
        import pyarrow.compute as arrow_compute
        import pyarrow.csv as pacsv

        pc: Any = arrow_compute
        request, mapping, budget = self.request, self.mapping, self.budget
        path = request.path
        before = _source_stat(path)
        if "dictionary" in mapping:
            _verify_dictionary(request.dictionary_path or path.with_name(mapping["dictionary"]["filename"]),
                               mapping, max_bytes=budget.metadata_bytes)
        fields = mapping["fields"]
        width = len(fields)
        codes, names = [f["code"] for f in fields], [f["field"] for f in fields]
        if 8 * width > budget.batch_bytes:
            raise ValueError(f"{path}: string offsets exceed native batch byte budget")
        layout = mapping.get("layout")
        code_record = layout["code_record"] if layout else 2
        label_record = layout["label_record"] if layout else 1
        first, last = (layout["first_data_record"], layout["last_data_record"]) if layout else (3, None)
        notes = layout["annotations"] if layout else []
        headers: dict[int, list[str]] = {}
        note_records: dict[int, list[list[str]]] = {}
        note_digests = [sha256() for _ in notes]
        keys, positions = [], []
        key_width = 9 if request.key_field == "campus_number" else 6
        count = rows = metadata_bytes = 0
        body = bytearray()
        numbers: list[int] = []

        def decode():
            # A controlled single LF prevents Arrow from treating a literal
            # BOM at a later chunk's first data cell as a file signature.
            table = pacsv.read_csv(pa.BufferReader(b"\n" + body),
                read_options=pacsv.ReadOptions(column_names=codes, use_threads=False,
                                               skip_rows=1, block_size=len(body) + 1),
                parse_options=pacsv.ParseOptions(newlines_in_values=True, ignore_empty_lines=False),
                convert_options=pacsv.ConvertOptions(column_types=dict.fromkeys(codes, pa.string()),
                    null_values=[], strings_can_be_null=False, quoted_strings_can_be_null=False)).rename_columns(names)
            if table.nbytes > budget.batch_bytes or table.num_rows != len(numbers):
                raise ValueError(f"{path}: native CSV batch exceeds admitted byte/row budget")
            try:
                identity = canonical_source_identifiers(table[request.key_field], width=key_width)
                if pc.any(pc.is_null(identity)).as_py():
                    i = pc.index(pc.is_null(identity), True).as_py()
                    raise ValueError(f"record {numbers[i]} has a missing source identifier")
                if request.key_field == "campus_number" and "district_number" in names:
                    district = canonical_source_identifiers(table["district_number"], width=6)
                    parent = pc.utf8_slice_codeunits(identity, start=1, stop=7)
                    if pc.any(pc.or_(pc.is_null(district), pc.fill_null(pc.not_equal(
                            parent, pc.utf8_slice_codeunits(district, start=1)), True))).as_py():
                        raise ValueError("campus and district identifiers disagree")
            except ValueError as error:
                message = str(error)
                if match := re.search(r"batch row ([0-9]+)", message):
                    message += f" (source record {numbers[int(match[1]) - 1]})"
                raise ValueError(f"{path}: invalid {request.key_field}: {message}") from error
            keys.extend(identity.chunks)
            positions.append(pa.array(numbers, type=pa.int64()))
            # A batch has at most the byte-admitted record count. Arrow may
            # internally produce chunks; expose one bounded batch consistently.
            return table.combine_chunks().to_batches()[0]

        with path.open("rb") as stream:
            record_error = f"{path}: logical CSV record exceeds record byte budget"
            lines = SourceLines(stream, budget.record_bytes, record_error)
            reader = csv.reader(lines, strict=True)
            try:
                while True:
                    lines.record.clear()
                    metadata_record = count + 1 in {code_record, label_record} or any(
                        note["first_record"] <= count + 1 <= note["last_record"] for note in notes)
                    remaining = max(0, budget.metadata_bytes - metadata_bytes)
                    lines.limit = min(budget.record_bytes, remaining) if metadata_record else budget.record_bytes
                    lines.error = (f"{path}: header/annotation metadata byte budget exceeded"
                                   if metadata_record and remaining < budget.record_bytes else record_error)
                    try:
                        row = next(reader)
                    except StopIteration:
                        break
                    count += 1
                    segment = lines.record
                    note = next((i for i, n in enumerate(notes)
                                 if n["first_record"] <= count <= n["last_record"]), None)
                    if note is not None or count in {code_record, label_record}:
                        metadata_bytes += len(segment) + 64 * len(row)
                        if metadata_bytes > budget.metadata_bytes:
                            raise ValueError(f"{path}: header/annotation metadata byte budget exceeded")
                        if note is not None:
                            note_records.setdefault(note, []).append(row)
                            note_digests[note].update(segment)
                            if notes[note]["kind"] == "blank" and any(row):
                                raise ValueError(f"{path}: declared blank annotation contains nonblank source text")
                        else:
                            headers[count] = row
                    elif count >= first and (last is None or count <= last):
                        if len(row) != width:
                            raise ValueError(f"{path}: record {count} has {len(row)} fields; expected {width}")
                        # Fixed-width normalized keys, offsets and record numbers;
                        # reserve before conversion/retention, never per fact cell.
                        if (rows + 1) * (key_width + 18) > budget.key_bytes:
                            raise ValueError(f"{path}: global identity index byte budget exceeded")
                        data = bytes(segment)
                        if count == 1:
                            data = data.removeprefix(b"\xef\xbb\xbf")
                        if len(data) + 1 + 8 * width > budget.batch_bytes:
                            raise ValueError(f"{path}: record cannot fit native batch byte budget")
                        if numbers and (len(numbers) == budget.batch_rows
                                or len(body) + len(data) + 1 + 4 * width * (len(numbers) + 2) > budget.batch_bytes):
                            yield decode()
                            body.clear()
                            numbers.clear()
                        body.extend(data)
                        numbers.append(count)
                        rows += 1
                    else:
                        raise ValueError(f"{path}: unaccounted record {count}; review its layout/annotation provenance")
            except csv.Error as error:
                raise ValueError(f"{path}: invalid CSV near physical line {reader.line_num}: {error}") from error
            if numbers:
                yield decode()
            if lines.digest.hexdigest() != self.input_sha256:
                raise ValueError(f"{path}: input differs from the reviewed header crosswalk")
        if _source_stat(path) != before:
            raise ValueError(f"{path}: input changed during source preparation")
        if last is not None and count < last:
            raise ValueError(f"{path}: missing reviewed data records through {last}")
        if headers.get(code_record) != codes:
            raise ValueError(f"{path}: source-code header differs from reviewed record {code_record}")
        if label_record is not None and headers.get(label_record) != [f["label"] for f in fields]:
            raise ValueError(f"{path}: human-label header differs from reviewed record {label_record}")
        observed = []
        for i, note in enumerate(notes):
            records = note_records.get(i, [])
            if len(records) != note["last_record"] - note["first_record"] + 1 or note_digests[i].hexdigest() != note["sha256"]:
                raise ValueError(f"{path}: annotation range {note['first_record']}–{note['last_record']} differs from reviewed notes")
            observed.append({**note, "records": records})
        if not rows:
            raise ValueError(f"{path}: no TAPR data records")
        if layout is not None and rows != reviewed_tapr_data_record_count(mapping):
            raise ValueError(f"{path}: extracted data count differs from reviewed record layout")
        identity = pa.chunked_array(keys)
        if pc.count_distinct(identity).as_py() != rows:
            counts = pc.value_counts(identity)
            duplicate = pc.filter(counts.field("values"), pc.greater(counts.field("counts"), 1))[0]
            records = pc.filter(pa.chunked_array(positions), pc.equal(identity, duplicate))
            example = records.slice(0, 20).to_pylist()
            suffix = " (first20)" if len(records) > 20 else ""
            raise ValueError(f"{path}: invalid {request.key_field}: duplicate {request.key_field} at records {example}{suffix}")
        self._summary = TaprSourceSummary(path, self.input_sha256, request.dataset,
            request.edition, tuple(observed), mapping, rows, before)


def _verify_dictionary(path: Path, mapping: Mapping[str, Any], *, max_bytes: int) -> None:
    import pandas as pd

    descriptor = mapping["dictionary"]
    if path.stat().st_size > max_bytes:
        raise ValueError(f"{path}: publisher dictionary exceeds metadata byte budget")
    with path.open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(f"{path}: publisher dictionary exceeds metadata byte budget")
    if sha256(raw).hexdigest() != descriptor["sha256"]:
        raise ValueError(f"{path}: publisher dictionary bytes differ from reviewed edition")
    dictionary = pd.read_excel(io.BytesIO(raw), sheet_name=descriptor["sheet_name"],
                               dtype=str, keep_default_na=False, na_filter=False, engine="openpyxl")
    if list(dictionary.columns) != ["Name", "Type", "Length", "Label"]:
        raise ValueError(f"{path}: unexpected publisher dictionary layout")
    expected = pd.DataFrame([{"Name": field["code"], "Type": field["publisher_type"],
                              "Length": field["publisher_length"], "Label": field["label"]}
                             for field in mapping["fields"]])
    if not dictionary.equals(expected):
        raise ValueError(f"{path}: publisher dictionary does not match the complete reviewed crosswalk")


def validate_tapr_key(mapping: Mapping[str, Any], key_field: str) -> None:
    """Share the existing reviewed publisher-code/physical-key binding."""
    headers = [field["field"] for field in mapping["fields"]]
    if key_field not in {"campus_number", "district_number"} or key_field not in headers:
        raise ValueError(f"missing or unsupported source key {key_field!r}")
    expected_code = "CAMPUS" if key_field == "campus_number" else "DISTRICT"
    if mapping["fields"][headers.index(key_field)]["code"] != expected_code:
        raise ValueError(f"source key must bind publisher code {expected_code!r}")


def reviewed_tapr_data_record_count(mapping: Mapping[str, Any]) -> int:
    """Count a finite reviewed interval with the reader's annotation precedence."""
    if ("input_sha256" in mapping or "input_sha256s" in mapping) and len(table_input_sha256s(mapping)) != 1:
        raise ValueError("TAPR physical record layout requires a single raw input")
    layout = mapping.get("layout")
    if layout is None:
        raise ValueError("Prepared source requires a finite reviewed record layout")
    validate_source_layout(layout)
    first, last = layout["first_data_record"], layout["last_data_record"]
    excluded = sum(len(range(max(first, note["first_record"]), min(last, note["last_record"]) + 1))
                   for note in layout["annotations"])
    rows = last - first + 1 - excluded
    if rows < 1:
        raise ValueError("Reviewed source layout has no TAPR data records")
    return rows


def read_tapr_table(path: str | Path, *, mapping: Mapping[str, Any], key_field: str,
                    dataset: str, edition: str,
                    dictionary_path: str | Path | None = None,
                    budget: TaprReadBudget | None = None) -> TaprSourceTable:
    """Explicit eager API, composed from the canonical bounded source scanner."""
    import pyarrow as pa

    request = TaprSourceRequest(Path(path), mapping, key_field, dataset, edition,
        Path(dictionary_path) if dictionary_path is not None else None)
    with request.open(budget=budget) as reader:
        table = pa.Table.from_batches(reader)
        summary = reader.summary
    return TaprSourceTable(table, summary.path, summary.input_sha256, dataset, edition,
                           summary.annotations, summary.mapping)


def merge_tapr_tables(sources: list[TaprSourceTable], *, key_field: str):
    """Outer-merge this edition's native tables, retaining historical entities.

    Shared identity/header definitions must agree. This is deliberately not a
    cross-dataset metric arbitration rule or a comparison between periods.
    """
    import pyarrow as pa
    import pyarrow.compute as arrow_compute

    pc: Any = arrow_compute

    if not sources:
        raise ValueError("At least one source table is required")
    if key_field not in {"campus_number", "district_number"}:
        raise ValueError("Unsupported TAPR merge identity")
    width = 9 if key_field == "campus_number" else 6
    if len({(source.dataset, source.edition) for source in sources}) != 1:
        raise ValueError("TAPR tables from different qualified datasets or editions must remain independent")
    if len({tuple(sorted(source.mapping.get("metric_period", {}).items())) for source in sources}) != 1:
        raise ValueError("TAPR tables with different metric reference periods must remain independent")
    if len({source.mapping["table"] for source in sources}) != len(sources):
        raise ValueError("Different revisions or repeated observations of one TAPR table must remain independent")
    for source in sources:
        validate_header_table(source.mapping)
        if len(table_input_sha256s(source.mapping)) != 1:
            raise ValueError("TAPR source merge requires a single raw input per physical table")
        if source.table.column_names != [field["field"] for field in source.mapping["fields"]] or key_field not in source.table.column_names:
            raise ValueError("TAPR native table differs from its reviewed field mapping")
        identity = canonical_source_identifiers(source.table[key_field], width=width)
        if pc.any(pc.is_null(identity)).as_py() or pc.count_distinct(identity).as_py() != source.table.num_rows:
            raise ValueError("TAPR source table has missing or duplicate identities")
    result = sources[0].table
    definitions = {f["field"]: (f["code"], f["label"]) for f in sources[0].mapping["fields"]}
    for source in sources[1:]:
        for field in source.mapping["fields"]:
            if definitions.setdefault(field["field"], (field["code"], field["label"])) != (field["code"], field["label"]):
                raise ValueError(f"Conflicting shared header definition: {field['field']!r}")
        left_names = {name: f"a{index}" for index, name in enumerate(result.column_names)}
        right_names = {name: f"b{index}" for index, name in enumerate(source.table.column_names)}
        left = result.rename_columns(list(left_names.values())).append_column(
            "identity", canonical_source_identifiers(result[key_field], width=width))
        right = source.table.rename_columns(list(right_names.values())).append_column(
            "identity", canonical_source_identifiers(source.table[key_field], width=width))
        joined = left.join(right, keys="identity", join_type="full outer")
        columns = list(dict.fromkeys([*result.column_names, *source.table.column_names]))
        arrays = []
        for name in columns:
            if name in left_names and name in right_names:
                a, b = joined[left_names[name]], joined[right_names[name]]
                conflict = pc.fill_null(pc.and_(pc.and_(pc.not_equal(a, ""), pc.not_equal(b, "")), pc.not_equal(a, b)), False)
                if pc.any(conflict).as_py():
                    if name == key_field:
                        raise ValueError(f"Inconsistent TAPR {key_field} representations across source tables")
                    raise ValueError(f"Conflicting TAPR values for shared field {name!r}")
                arrays.append(pc.coalesce(pc.if_else(pc.fill_null(pc.equal(a, ""), False),
                                                     pa.scalar(None, type=a.type), a), b, a))
            else:
                arrays.append(joined[left_names[name]] if name in left_names else joined[right_names[name]])
        result = pa.Table.from_arrays(arrays, names=columns)
        result = result.take(pc.sort_indices(result, sort_keys=[(key_field, "ascending")]))
    return result
