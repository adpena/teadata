from __future__ import annotations

import argparse
import csv
import math
import os
import re
from itertools import chain
from pathlib import Path
from typing import Any, Iterable

from openpyxl import Workbook  # type: ignore[import-untyped]
from openpyxl.styles import Alignment, Font, PatternFill  # type: ignore[import-untyped]
from openpyxl.utils import get_column_letter  # type: ignore[import-untyped]
from openpyxl.worksheet.table import Table, TableStyleInfo  # type: ignore[import-untyped]

from teadata.scripts.paths import data_path
from teadata.identifiers import canonical_source_identifier
from teadata.column_schema import parse_column_schema, crosswalk_dataset, validate_header_table
from teadata.manifest import file_identity
from teadata.teadata_config import load_config


def _read_tapr_csv(
    path: Path, *, mapping: dict[str, Any], key_field: str
) -> tuple[list[str], dict[str, dict[str, str]], list[dict[str, str]]]:
    """Read this two-header layout without repairing ambiguous source records.

    Human labels and publisher codes are returned verbatim as metadata. Body
    tokens retain the existing whitespace normalization; masking is interpreted
    elsewhere by the dataset's value policy.
    """
    widths = {"campus_number": 9, "district_number": 6}
    if key_field not in widths:
        raise ValueError(f"Unsupported TAPR key field: {key_field}")
    try:
        validate_header_table(mapping)
    except ValueError as error:
        raise ValueError(f"{path}: invalid header crosswalk: {error}") from error
    if file_identity(path)["sha256"] != mapping["input_sha256"]:
        raise ValueError(f"{path}: input differs from the reviewed header crosswalk")
    fields = mapping["fields"]
    headers = [field["field"] for field in fields]
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh, strict=True)
        try:
            labels = next(reader)
            codes = next(reader)
        except (StopIteration, csv.Error) as error:
            raise ValueError(f"{path}: expected human-label and source-code header records") from error
        if not labels or len(labels) != len(codes):
            raise ValueError(f"{path}: header width mismatch: {len(labels)} labels, {len(codes)} codes")

        if key_field not in headers:
            raise ValueError(f"{path}: missing key header {key_field!r}")
        if len(codes) != len(fields):
            raise ValueError(f"{path}: human-label/code headers differ from the reviewed crosswalk")
        key_code = "CAMPUS" if key_field == "campus_number" else "DISTRICT"
        key_column = headers.index(key_field)
        if codes[key_column].strip() != key_code:
            raise ValueError(f"{path}: expected source-code header {key_code!r} in record 2, column {key_column + 1}")
        if labels != [field["label"] for field in fields] or codes != [field["code"] for field in fields]:
            raise ValueError(f"{path}: human-label/code headers differ from the reviewed crosswalk")

        elements = []
        for code, label, header in zip(codes, labels, headers, strict=True):
            elements.append(
                {
                    "code": code,
                    "label": label,
                    "sanitized_label": header,
                }
            )

        rows: dict[str, dict[str, str]] = {}
        seen_keys: dict[str, int] = {}
        header_len = len(headers)
        try:
            for number, row in enumerate(reader, 3):
                if len(row) != header_len:
                    raise ValueError(f"{path}: record {number} has {len(row)} fields; expected {header_len}")
                record = {header: value.strip() for header, value in zip(headers, row, strict=True)}
                key = record[key_field]
                try:
                    identity = canonical_source_identifier(key, width=widths[key_field])
                    if identity is None:
                        raise ValueError("missing source identifier")
                    if key_field == "campus_number" and "district_number" in record:
                        district = canonical_source_identifier(record["district_number"], width=6)
                        if district is None or identity[1:7] != district[1:]:
                            raise ValueError("campus and district identifiers disagree")
                except ValueError as error:
                    raise ValueError(f"{path}: record {number} has invalid {key_field}: {key!r} ({error})") from error
                if identity in seen_keys:
                    raise ValueError(f"{path}: duplicate {key_field} at records {seen_keys[identity]} and {number}: {key!r}")
                seen_keys[identity] = number
                rows[key] = record
        except csv.Error as error:
            raise ValueError(f"{path}: invalid CSV near physical line {reader.line_num}: {error}") from error
        if not rows:
            raise ValueError(f"{path}: no TAPR data records")
    if file_identity(path)["sha256"] != mapping["input_sha256"]:
        raise ValueError(f"{path}: input changed while reading the reviewed header crosswalk")

    return headers, rows, elements


def _header_tables(cfg, dataset: str) -> dict[str, dict[str, Any]]:
    source = cfg.resolve_dataset(dataset, 2025)
    if source.resolved_year != 2025 or source.metadata.version != "2025":
        raise ValueError(f"No selected 2025 header crosswalk for {dataset}")
    contract = cfg.column_contract(dataset, 2025)
    if contract is None:
        raise ValueError(f"Header crosswalk unavailable for {dataset}")
    selected = crosswalk_dataset(parse_column_schema(contract["text"].encode()), dataset, "2025")
    tables = {table["table"]: table for table in selected["tables"]}
    if set(tables) != {"student", "staff"}:
        raise ValueError(f"Student/staff header crosswalk required for {dataset}")
    return tables


def _merge_tables(
    primary_headers: list[str],
    primary_rows: dict[str, dict[str, str]],
    secondary_headers: list[str],
    secondary_rows: dict[str, dict[str, str]],
    *,
    key_field: str,
    preferred_cols: Iterable[str],
):
    """Outer-merge available sources, retaining the existing blank-fill policy.

    One-sided entities are retained, and a blank primary token may be filled
    from the secondary source. Two nonblank tokens for a shared field must
    agree; this does not assert complete or identical source coverage.
    """
    width = {"campus_number": 9, "district_number": 6}[key_field]
    representations: dict[str, str] = {}
    for key in chain(primary_rows, secondary_rows):
        identity = canonical_source_identifier(key, width=width)
        if identity is None:
            raise ValueError(f"Missing TAPR {key_field}")
        previous = representations.setdefault(identity, key)
        if previous != key:
            raise ValueError(f"Inconsistent TAPR {key_field} representations: {previous!r}, {key!r}")
    keys = sorted(set(primary_rows) | set(secondary_rows))
    merged_rows: list[dict[str, str]] = []
    for key in keys:
        record: dict[str, str] = {}
        record.update(primary_rows.get(key, {}))
        for k, v in secondary_rows.get(key, {}).items():
            if k in record and record[k] not in (None, ""):
                if v not in (None, "") and record[k] != v:
                    raise ValueError(f"Conflicting TAPR values for {key_field} {key!r}, field {k!r}: {record[k]!r} and {v!r}")
                continue
            record[k] = v
        merged_rows.append(record)

    present = set(primary_headers) | set(secondary_headers)
    ordered_headers = list(dict.fromkeys([
        *(col for col in preferred_cols if col in present),
        *primary_headers,
        *secondary_headers,
    ]))

    if key_field not in ordered_headers:
        ordered_headers.insert(0, key_field)

    return ordered_headers, merged_rows


def _rows_for_headers(headers: list[str], records: list[dict[str, str]]) -> list[list[str]]:
    return [[record.get(h, "") for h in headers] for record in records]


_TABLE_STYLE_NAME = "TableStyleMedium16"
_TABLE_NAME_SANITIZER = re.compile(r"[^A-Za-z0-9_]")
_COLUMN_PADDING = 1
_DEFAULT_COLUMN_WIDTH = 12
_MAX_COLUMN_WIDTH = 255
_MIN_CHAR_WIDTH = 4
_MAX_HEADER_LINES = 4
_HEADER_FONT_COLOR = "00FFFFFF"
_HEADER_FILL_COLOR = "FF1F3864"
_HEADER_ALIGNMENT = Alignment(horizontal="center", vertical="center", wrapText=True)


def format_worksheet_as_table(worksheet, *, table_name: str | None = None) -> None:
    if worksheet.max_row < 1 or worksheet.max_column < 1:
        return

    worksheet.freeze_panes = "A2"
    _wrap_header_row(worksheet)
    _autofit_columns(worksheet)
    _apply_table_style(worksheet, requested_name=table_name)


def _wrap_header_row(worksheet) -> None:
    header_rows = worksheet.iter_rows(min_row=1, max_row=1)
    try:
        header = next(header_rows)
    except StopIteration:
        return

    for cell in header:
        cell.alignment = (cell.alignment or Alignment()).copy(
            horizontal=_HEADER_ALIGNMENT.horizontal,
            vertical=_HEADER_ALIGNMENT.vertical,
            wrapText=_HEADER_ALIGNMENT.wrapText,
        )
        cell.font = (cell.font or Font()).copy(color=_HEADER_FONT_COLOR)
        cell.fill = PatternFill(
            fill_type="solid",
            fgColor=_HEADER_FILL_COLOR,
            bgColor=_HEADER_FILL_COLOR,
        )


def _autofit_columns(worksheet) -> None:
    max_row = worksheet.max_row
    max_col = worksheet.max_column
    if max_col == 0:
        return

    for column_cells in worksheet.iter_cols(
        min_row=1,
        max_row=max_row,
        min_col=1,
        max_col=max_col,
    ):
        header_cell = column_cells[0]
        data_cells = column_cells[1:]

        data_width = max(
            (_cell_display_length(cell.value) for cell in data_cells),
            default=0,
        )
        header_text = _cell_text(header_cell.value)
        header_total_length = _cell_display_length(header_text)
        header_longest_word = _longest_word_length(header_text)

        adjusted_width = data_width + _COLUMN_PADDING if data_width else 0
        adjusted_width = max(adjusted_width, header_longest_word)

        if header_total_length and adjusted_width:
            estimated_lines = math.ceil(header_total_length / adjusted_width)
            if estimated_lines > _MAX_HEADER_LINES:
                widened_for_header = math.ceil(
                    header_total_length / _MAX_HEADER_LINES
                )
                adjusted_width = max(adjusted_width, widened_for_header)

        if adjusted_width == 0:
            adjusted_width = _DEFAULT_COLUMN_WIDTH

        adjusted_width = max(adjusted_width, _MIN_CHAR_WIDTH)
        adjusted_width = min(adjusted_width, _MAX_COLUMN_WIDTH)
        column_letter = get_column_letter(header_cell.column)
        worksheet.column_dimensions[column_letter].width = adjusted_width


def _apply_table_style(worksheet, requested_name: str | None) -> None:
    min_row = worksheet.min_row or 1
    max_row = worksheet.max_row or 1
    min_col = worksheet.min_column or 1
    max_col = worksheet.max_column or 1
    if max_row < min_row or max_col < min_col:
        return

    start = f"{get_column_letter(min_col)}{min_row}"
    end = f"{get_column_letter(max_col)}{max_row}"
    table_range = f"{start}:{end}"
    table_name = _generate_table_name(worksheet, requested_name)

    table = Table(displayName=table_name, ref=table_range)
    table.tableStyleInfo = TableStyleInfo(
        name=_TABLE_STYLE_NAME,
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )
    worksheet.add_table(table)


def _cell_display_length(value: object) -> int:
    if value in (None, ""):
        return 0
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
    if not text:
        return 0
    return max(len(part) for part in text.splitlines()) or 0


def _cell_text(value: object) -> str:
    if value in (None, ""):
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _longest_word_length(text: str) -> int:
    return max((len(word) for word in re.split(r"\s+", text) if word), default=0)


def _generate_table_name(worksheet, requested_name: str | None) -> str:
    base_name = requested_name or f"{worksheet.title}_table"
    base_name = _TABLE_NAME_SANITIZER.sub("_", base_name).strip("_") or "Table"
    if not base_name[0].isalpha() and base_name[0] != "_":
        base_name = f"_{base_name}"
    base_name = base_name[:250]

    existing_names = _collect_table_names(worksheet)
    candidate = base_name
    suffix = 1
    while candidate in existing_names:
        candidate = f"{base_name}_{suffix}"
        suffix += 1
    return candidate


def _collect_table_names(worksheet) -> set[str]:
    workbook = getattr(worksheet, "parent", None)
    tables: set[str] = set()
    worksheets = getattr(workbook, "worksheets", []) if workbook is not None else [worksheet]

    for ws in worksheets:
        ws_tables = getattr(ws, "tables", None)
        if not ws_tables:
            continue
        if isinstance(ws_tables, dict):
            tables.update(ws_tables.keys())
        else:
            tables.update(getattr(tbl, "displayName", "") for tbl in ws_tables)

    return {name for name in tables if name}


def _write_workbook(
    output_path: Path,
    *,
    sheet_name: str,
    headers: list[str],
    rows: list[list[str]],
    table_name: str,
) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name
    ws.append(headers)
    for row in rows:
        ws.append(row)
    format_worksheet_as_table(ws, table_name=table_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)


def _write_elements_workbook(
    output_path: Path,
    *,
    sheet_name: str,
    elements: list[dict[str, str]],
    table_name: str,
) -> None:
    headers = ["code", "label", "sanitized_label", "source"]
    rows = [[el.get(h, "") for h in headers] for el in elements]
    _write_workbook(
        output_path,
        sheet_name=sheet_name,
        headers=headers,
        rows=rows,
        table_name=table_name,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build 2025 TAPR campus and district profiles."
    )
    parser.add_argument("--input-dir", type=Path, default=data_path("tapr", "raw"))
    parser.add_argument("--output-dir", type=Path, default=data_path("tapr"))
    args = parser.parse_args(argv)
    campus_student = args.input_dir / "2025 Campus Student Information.csv"
    campus_staff = args.input_dir / "2025 Campus Staff Information.csv"
    district_student = args.input_dir / "2025 District Student Information.csv"
    district_staff = args.input_dir / "2025 District Staff Information.csv"
    campus_out = args.output_dir / "CAMPPROF_2024-2025_FINAL.xlsx"
    district_out = args.output_dir / "DISTPROF_2024-2025_FINAL.xlsx"
    campus_elements_out = args.output_dir / "CAMPPROF_2024-2025_data_elements.xlsx"
    district_elements_out = args.output_dir / "DISTPROF_2024-2025_data_elements.xlsx"
    cfg = load_config(os.environ.get("TEADATA_CONFIG") or Path(__file__).resolve().parents[1] / "teadata_sources.yaml")
    campus_mappings = _header_tables(cfg, "campus_tapr_student_staff_profile")
    district_mappings = _header_tables(cfg, "district_tapr_student_staff_profile")

    campus_student_headers, campus_student_rows, campus_student_elements = (
        _read_tapr_csv(campus_student, mapping=campus_mappings["student"], key_field="campus_number")
    )
    campus_staff_headers, campus_staff_rows, campus_staff_elements = _read_tapr_csv(
        campus_staff, mapping=campus_mappings["staff"], key_field="campus_number"
    )
    district_student_headers, district_student_rows, district_student_elements = (
        _read_tapr_csv(
            district_student, mapping=district_mappings["student"], key_field="district_number"
        )
    )
    district_staff_headers, district_staff_rows, district_staff_elements = (
        _read_tapr_csv(
            district_staff, mapping=district_mappings["staff"], key_field="district_number"
        )
    )

    campus_headers, campus_records = _merge_tables(
        campus_student_headers,
        campus_student_rows,
        campus_staff_headers,
        campus_staff_rows,
        key_field="campus_number",
        preferred_cols=(
            "campus_number",
            "campus_name",
            "district_number",
            "district_name",
        ),
    )
    district_headers, district_records = _merge_tables(
        district_student_headers,
        district_student_rows,
        district_staff_headers,
        district_staff_rows,
        key_field="district_number",
        preferred_cols=("district_number", "district_name"),
    )

    campus_rows = _rows_for_headers(campus_headers, campus_records)
    district_rows = _rows_for_headers(district_headers, district_records)

    _write_workbook(
        campus_out,
        sheet_name="campus_tapr_2025",
        headers=campus_headers,
        rows=campus_rows,
        table_name="campus_tapr_2025",
    )
    _write_workbook(
        district_out,
        sheet_name="district_tapr_2025",
        headers=district_headers,
        rows=district_rows,
        table_name="district_tapr_2025",
    )

    for el in campus_student_elements:
        el["source"] = "campus_student"
    for el in campus_staff_elements:
        el["source"] = "campus_staff"
    for el in district_student_elements:
        el["source"] = "district_student"
    for el in district_staff_elements:
        el["source"] = "district_staff"

    _write_elements_workbook(
        campus_elements_out,
        sheet_name="campus_tapr_elements_2025",
        elements=campus_student_elements + campus_staff_elements,
        table_name="campus_tapr_elements_2025",
    )
    _write_elements_workbook(
        district_elements_out,
        sheet_name="district_tapr_elements_2025",
        elements=district_student_elements + district_staff_elements,
        table_name="district_tapr_elements_2025",
    )


if __name__ == "__main__":
    main()
