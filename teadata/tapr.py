"""Lossless TAPR source preparation from reviewed edition/header resources.

CSV record classification is structural. Arrow decodes fact columns and
merges whole native tables; scientific token interpretation stays in values.py.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import io
from pathlib import Path
import re
from typing import Any, Mapping

from .column_schema import validate_header_table
from .identifiers import canonical_source_identifiers


@dataclass(frozen=True)
class TaprSourceTable:
    table: Any
    path: Path
    input_sha256: str
    dataset: str
    edition: str
    annotations: tuple[dict[str, Any], ...]
    mapping: Mapping[str, Any]


def _extract_records(raw: bytes, mapping: Mapping[str, Any], path: Path):
    """Extract only explicitly reviewed data/annotation record ranges.

    The standard CSV decoder handles quoted multiline records. This scan never
    converts fact cells, pads widths, strips tokens or infers notes from IDs.
    """
    fields = mapping["fields"]
    width = len(fields)
    layout = mapping.get("layout")
    code_record = layout["code_record"] if layout else 2
    label_record = layout["label_record"] if layout else 1
    first = layout["first_data_record"] if layout else 3
    last = layout["last_data_record"] if layout else None
    annotations = layout["annotations"] if layout else []
    text = raw.decode("utf-8-sig")
    stream = io.StringIO(text, newline="")
    reader = csv.reader(stream, strict=True)
    headers: dict[int, list[str]] = {}
    spans: list[tuple[int, int]] = []
    observed: list[dict[str, Any]] = []
    annotation_records: dict[int, list[list[str]]] = {}
    annotation_text: dict[int, list[str]] = {}
    data_records: list[int] = []
    count = 0
    try:
        while True:
            start = stream.tell()
            try:
                row = next(reader)
            except StopIteration:
                break
            end = stream.tell()
            count += 1
            note = next((i for i, item in enumerate(annotations)
                         if item["first_record"] <= count <= item["last_record"]), None)
            if note is not None:
                annotation_records.setdefault(note, []).append(row)
                segment = text[start:end]
                if count == 1 and raw.startswith(b"\xef\xbb\xbf"):
                    segment = "\ufeff" + segment
                annotation_text.setdefault(note, []).append(segment)
            elif count in {code_record, label_record}:
                headers[count] = row
            elif count >= first and (last is None or count <= last):
                if len(row) != width:
                    raise ValueError(f"{path}: record {count} has {len(row)} fields; expected {width}")
                data_records.append(count)
                if spans and spans[-1][1] == start:
                    spans[-1] = (spans[-1][0], end)
                else:
                    spans.append((start, end))
            else:
                raise ValueError(f"{path}: unaccounted record {count}; review its layout/annotation provenance")
    except csv.Error as error:
        raise ValueError(f"{path}: invalid CSV near physical line {reader.line_num}: {error}") from error
    if last is not None and count < last:
        raise ValueError(f"{path}: missing reviewed data records through {last}")
    if headers.get(code_record) != [field["code"] for field in fields]:
        raise ValueError(f"{path}: source-code header differs from reviewed record {code_record}")
    if label_record is not None and headers.get(label_record) != [field["label"] for field in fields]:
        raise ValueError(f"{path}: human-label header differs from reviewed record {label_record}")
    for index, note in enumerate(annotations):
        records = annotation_records.get(index, [])
        content = "".join(annotation_text.get(index, [])).encode("utf-8")
        if (len(records) != note["last_record"] - note["first_record"] + 1
                or sha256(content).hexdigest() != note["sha256"]):
            raise ValueError(f"{path}: annotation range {note['first_record']}–{note['last_record']} differs from reviewed notes")
        if note["kind"] == "blank" and any(any(cell for cell in row) for row in records):
            raise ValueError(f"{path}: declared blank annotation contains nonblank source text")
        observed.append({**note, "records": records})
    if not spans:
        raise ValueError(f"{path}: no TAPR data records")
    return "".join(text[start:end] for start, end in spans).encode("utf-8"), observed, data_records


def _verify_dictionary(path: Path, mapping: Mapping[str, Any]) -> None:
    import pandas as pd

    descriptor = mapping["dictionary"]
    raw = path.read_bytes()
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


def read_tapr_table(path: str | Path, *, mapping: Mapping[str, Any], key_field: str,
                    dataset: str, edition: str,
                    dictionary_path: str | Path | None = None) -> TaprSourceTable:
    """Read all publisher fields as exact strings, validating strict identities."""
    import pyarrow as pa
    import pyarrow.compute as arrow_compute
    import pyarrow.csv as pacsv

    pc: Any = arrow_compute  # Arrow registers compute kernels dynamically.
    path = Path(path)
    if (key_field not in {"campus_number", "district_number"}
            or dataset != key_field.removesuffix("_number") + "_tapr_student_staff_profile"
            or not isinstance(edition, str) or not edition.strip()):
        raise ValueError("TAPR preparation requires an explicit qualified dataset and source edition")
    validate_header_table(mapping)
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != mapping["input_sha256"]:
        raise ValueError(f"{path}: input differs from the reviewed header crosswalk")
    if "dictionary" in mapping:
        _verify_dictionary(Path(dictionary_path) if dictionary_path is not None
                           else path.with_name(mapping["dictionary"]["filename"]), mapping)
    fields = mapping["fields"]
    headers = [field["field"] for field in fields]
    if key_field not in {"campus_number", "district_number"} or key_field not in headers:
        raise ValueError(f"{path}: missing or unsupported source key {key_field!r}")
    expected_code = "CAMPUS" if key_field == "campus_number" else "DISTRICT"
    if fields[headers.index(key_field)]["code"] != expected_code:
        raise ValueError(f"{path}: source key must bind publisher code {expected_code!r}")
    body, annotations, data_records = _extract_records(raw, mapping, path)
    codes = [field["code"] for field in fields]
    table = pacsv.read_csv(pa.BufferReader(body),
        read_options=pacsv.ReadOptions(column_names=codes, use_threads=False),
        parse_options=pacsv.ParseOptions(newlines_in_values=True, ignore_empty_lines=False),
        convert_options=pacsv.ConvertOptions(column_types=dict.fromkeys(codes, pa.string()),
            null_values=[], strings_can_be_null=False, quoted_strings_can_be_null=False)).rename_columns(headers)
    try:
        identity = canonical_source_identifiers(table[key_field], width=9 if key_field == "campus_number" else 6)
        if pc.any(pc.is_null(identity)).as_py():
            position = pc.index(pc.is_null(identity), True).as_py()
            raise ValueError(f"record {data_records[position]} has a missing source identifier")
        if pc.count_distinct(identity).as_py() != table.num_rows:
            counts = pc.value_counts(identity)
            duplicate = pc.filter(counts.field("values"), pc.greater(counts.field("counts"), 1))[0]
            positions = pc.indices_nonzero(pc.equal(identity, duplicate)).to_pylist()
            raise ValueError(f"duplicate {key_field} at records {[data_records[index] for index in positions]}")
        if key_field == "campus_number" and "district_number" in headers:
            district = canonical_source_identifiers(table["district_number"], width=6)
            parent = pc.utf8_slice_codeunits(identity, start=1, stop=7)
            if pc.any(pc.or_(pc.is_null(district), pc.fill_null(pc.not_equal(parent, pc.utf8_slice_codeunits(district, start=1)), True))).as_py():
                raise ValueError("campus and district identifiers disagree")
    except ValueError as error:
        message = str(error)
        if match := re.search(r"batch row ([0-9]+)", message):
            message += f" (source record {data_records[int(match[1]) - 1]})"
        raise ValueError(f"{path}: invalid {key_field}: {message}") from error
    if sha256(path.read_bytes()).hexdigest() != mapping["input_sha256"]:
        raise ValueError(f"{path}: input changed during source preparation")
    return TaprSourceTable(table, path, mapping["input_sha256"], dataset, edition, tuple(annotations), mapping)


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
    if len({source.mapping["table"] for source in sources}) != len(sources):
        raise ValueError("Different revisions or repeated observations of one TAPR table must remain independent")
    for source in sources:
        validate_header_table(source.mapping)
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
