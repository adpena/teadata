"""Read selected 2026 TPRS student reports without treating them as bulk CSV.

The original HTML and receipt remain authoritative.  The raw reader retains
every cell in the current profile table; the native projection includes only
the explicitly reviewed Membership/Enrollment measures in the packaged grid.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
import gzip
from hashlib import sha256
from importlib.resources import files
from io import BytesIO
import json
from itertools import chain, zip_longest
import platform
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, urlsplit

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .column_schema import _unique_object, parse_column_schema, source_schema_fingerprint
from .dataset_metadata import STATEWIDE_SINGLETON_GRAIN, ArtifactRef, DataPeriod, DatasetMetadata, EditionKey, ResolvedDataset
from .manifest import BuildRecorder, ContractError, canonical_json
from .native_witnesses import _Rows
from .provenance import ProvenanceStore
from .teadata_config import _PACKAGED_CONFIG_PATH, _load_packaged_config, load_config, load_config_bytes
from .values import Measurement, ValuePolicy, ValueStatus, measure

if TYPE_CHECKING:
    from .tprs_statewide_intake import ReportSelection

DATASET = "tprs_student_membership_enrollment_reports"
EDITION = 2026
PERIOD = DataPeriod("school_year", 2025, 2026)
RESOURCE = "methodologies/tprs-students-2026.json"
SCOPES = {"state": "S", "district": "D", "campus": "C"}
BROKER = "https://rptsvr1.tea.texas.gov/cgi/sas/broker"
MAX_REPORT_BYTES = 2 * 1024**2
MAX_SELECTION_BYTES = 64 * 1024**2
_DIGEST = re.compile(r"[0-9a-f]{64}")
_COUNT = re.compile(r"(?:0|[1-9][0-9]*|[1-9][0-9]{0,2}(?:,[0-9]{3})+)")
_PERCENT = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9])?%")


@lru_cache(maxsize=1)
def _reviewed_grid_sha256(config_bytes: bytes, config_path: Path) -> str | None:
    source = _load_packaged_config(config_bytes, config_path).resolve_dataset(
        DATASET, EDITION, reader_profile="source", exact=True
    )
    reference, = [ref for ref in source.metadata.references if ref.reference_id == "report_grid"]
    return reference.reviewed_sha256


def definition() -> dict[str, Any]:
    """Return a detached copy of the sole reviewed field/layout authority."""
    raw = files("teadata").joinpath(RESOURCE).read_bytes()
    config_path = _PACKAGED_CONFIG_PATH
    reviewed = _reviewed_grid_sha256(config_path.read_bytes(), config_path)
    _need(sha256(raw).hexdigest() == reviewed, "Reviewed student-report grid changed")
    return json.loads(raw, object_pairs_hook=_unique_object)


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _identifier(scope: str, identifier: str) -> None:
    _need(isinstance(scope, str) and scope in SCOPES, "Unsupported student-report scope")
    pattern = {"state": r"S", "district": r"[0-9]{6}", "campus": r"[0-9]{9}"}[scope]
    _need(isinstance(identifier, str) and re.fullmatch(pattern, identifier) is not None
          and identifier not in {"000000", "000000000"}, "Invalid literal student-report identifier")


def _source_bytes(path: Path, digest: str, size: int) -> bytes:
    """Bind the bounded immutable parse buffer, including a post-stat change."""
    with path.open("rb") as stream:
        raw = stream.read(MAX_REPORT_BYTES + 1)
    _need(0 < len(raw) == size <= MAX_REPORT_BYTES and sha256(raw).hexdigest() == digest,
          "Student source bytes changed during the read")
    return raw


@dataclass(frozen=True)
class StudentCell:
    section: str
    label: str
    population: str
    measure: str
    geography_role: str
    raw: str
    row: int
    column: int
    field: str | None
    hold: str | None
    period: DataPeriod

    def measurement(self, policy: ValuePolicy) -> Measurement:
        unit = "count" if self.measure == "count" else "percent"
        if self.raw == "**" and self.section != "Students with Disabilities by Type of Primary Disability":
            return Measurement(None, ValueStatus.INVALID, self.raw, "token_outside_documented_disability_group", policy.policy_id, unit=unit)
        if not self.raw:
            return Measurement(None, ValueStatus.INVALID, self.raw, "unreviewed_blank_token", policy.policy_id, unit=unit)
        value = measure(self.raw, policy, unit=unit, minimum=0, maximum=100 if unit == "percent" else None)
        numeric = _COUNT if unit == "count" else _PERCENT
        if value.status == ValueStatus.OBSERVED and numeric.fullmatch(self.raw) is None:
            return Measurement(None, ValueStatus.INVALID, self.raw, "unrecognized_numeric_token", policy.policy_id, unit=unit)
        return value


@dataclass(frozen=True)
class StudentReport:
    scope: str
    identifier: str
    receipt: str
    sha256: str
    cells: tuple[StudentCell, ...]
    acquisition_notes: tuple[str, ...] = ()


def semantic_fields(contract: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Compile the projection from the report-grid authority, never CSV codes."""
    result = []
    for row in (contract or definition())["rows"]:
        if not row["qualified"]:
            continue
        for population in ("membership", "enrollment"):
            for quantity in ("count", "percent"):
                result.append({"field": f"{population}_{row['name']}_{quantity}", "label": row["label"],
                    "section": row["section"], "population": population, "quantity": quantity,
                    "denominator": None if quantity == "count" else f"Total Students / {population}",
                    "unit": "count" if quantity == "count" else "percent"})
    _need(len({field["field"] for field in result}) == len(result), "Duplicate reviewed student field binding")
    return result


def _student_html(raw: bytes) -> str:
    _need(isinstance(raw, bytes) and 0 < len(raw) <= MAX_REPORT_BYTES, "Student report exceeds the bounded HTML size")
    try:
        return raw.decode("cp1252")
    except UnicodeError as error:
        raise ContractError("Student report encoding differs") from error


def _check_student_heading(rows: Sequence[list[str]], scope: str, identifier: str) -> None:
    _need(len(rows) > 10 and rows[0] == ["", "Texas Education Agency", ""]
          and rows[1] == ["2025-26 Student Information"] and len(rows[2]) == 1,
          "Student report title/edition differs")
    _need(rows[2] == ["State"] if scope == "state" else re.findall(r"\(([0-9]+)\)", rows[2][0]) == [identifier],
          "Student report geography differs")


def parse_student_report(raw: bytes, *, scope: str, identifier: str, receipt: str = "") -> StudentReport:
    """Parse the complete current table and preserve unqualified/lagged cells.

    Row/column locators are one-based parsed HTML table coordinates, not lines.
    Whitespace is normalized by the shared HTML reader; original bytes remain
    in the acquisition store.  Missing grade rows are never manufactured.
    """
    _identifier(scope, identifier)
    grid = definition()
    rows = _Rows(_student_html(raw)).rows
    _check_student_heading(rows, scope, identifier)
    _need(rows[4] == ["", "Membership", "Enrollment"], "Student report population headers differ")
    header = 5 if scope == "state" else 6
    _need(rows[header] == grid["headers"][scope], "Student report quantity/benchmark headers differ")
    if scope != "state":
        _need(rows[5] == grid["geography_headers"][scope], "Student report geography roles differ")
    for legend in grid["legend"]:
        _need(rows.count(legend) == 1, "Student report disclosure legend differs")
    expected = {(row["section"], row["label"]): row for row in grid["rows"]}
    fields = {(f["section"], f["label"], f["population"], f["quantity"]): f["field"]
              for f in semantic_fields(grid)}
    section = ""
    seen: set[tuple[str, str]] = set()
    sections: list[str] = []
    cells = []
    boundary = False
    for index in range(header + 1, len(rows)):
        row = rows[index]
        if len(row) == 1 and row[0] == "Mobility (2024-25)":
            boundary = True
            break
        if len(row) == 1:
            section = row[0]
            sections.append(section)
            continue
        _need(len(row) == len(rows[header]), "Student report data row width differs")
        key = section, row[0]
        _need(key in expected and key not in seen, "Student report has an unknown or duplicate row")
        seen.add(key)
        spec = expected[key]
        period = PERIOD if spec["period"] == "2025-26" else DataPeriod("school_year", 2024, 2025)
        for column, role in enumerate(grid["column_roles"][scope], 1):
            population, quantity, geography_role = role
            quantity = spec.get("quantity_overrides", {}).get(scope, {}).get(geography_role, quantity)
            field = fields.get((*key, population, quantity)) if geography_role == "local" else None
            hold = None if field is not None else (spec["hold"] or "embedded_benchmark_not_a_direct_state_observation")
            cells.append(StudentCell(section, row[0], population, quantity, geography_role, row[column],
                                     index + 1, column + 1, field, hold, period))
    _need(boundary and sections == grid["sections"], "Student report section boundaries differ")
    required = {key for key in expected if key[0] != "Students by Grade"}
    _need(required <= seen and any(key[0] == "Students by Grade" for key in seen),
          "Student report omits a required current row")
    if scope != "campus":
        _need(seen == set(expected), "State/district student report grade coverage differs")
    return StudentReport(scope, identifier, receipt, sha256(raw).hexdigest(), tuple(cells))


def _validate_report_acquisition(record: Mapping[str, Any], *, scope: str, identifier: str,
                                 expected_sha256: str) -> None:
    """Check retained request metadata without opening or interpreting HTML."""
    _identifier(scope, identifier)
    size = record.get("artifact", {}).get("size_bytes")
    _need(type(size) is int and 0 < size <= MAX_REPORT_BYTES, "Student report exceeds the bounded HTML size")
    parameters = record["parameters"]
    url = urlsplit(record["requested_url"])
    query = parse_qsl(url.query, keep_blank_values=True)
    controls = dict(query)
    expected = {"_debug": "0", "_program": "perfrept.perfmast.sas", "_service": "marykay",
                "ccyy": "2026", "id": identifier, "lev": SCOPES[scope], "prgopt": "reports/tapr/student.sas"}
    _need(record["kind"] == "acquisition" and record["method"] == "download"
          and record["section"] == "reference"
          and (record["requested_year"], record["resolved_year"]) in {(None, None), (EDITION, EDITION)}
          and record["source_metadata"].get("version") in {None, "2026"}
          and url.scheme == "https" and url.netloc == "rptsvr1.tea.texas.gov" and url.path == "/cgi/sas/broker"
          and not url.fragment and record["resolved_url"] == record["requested_url"]
          and len(query) == len(controls) and controls == expected
          and (parameters.get("http_method") == "GET" or
               (parameters.get("http_method") is None and parameters.get("controls") == expected))
          and parameters.get("http_status", parameters.get("status_code")) == 200
          and record["response_headers"].get("content-type", "").split(";")[0] == "text/html"
          and bool(record["downloaded_at"]) and record["artifact"]["sha256"] == expected_sha256,
          "Student report acquisition does not bind the exact public request")


def _read_student_input(store: ProvenanceStore, receipt: str, *, scope: str, identifier: str,
                        expected_sha256: str) -> tuple[bytes, tuple[str, ...]]:
    """Bind exact public request controls, receipt closure, bytes and geography."""
    _identifier(scope, identifier)
    _need(isinstance(receipt, str) and isinstance(expected_sha256, str)
          and _DIGEST.fullmatch(receipt) is not None and _DIGEST.fullmatch(expected_sha256) is not None,
          "Student report requires exact receipt and content pins")
    preliminary = store.read_record(receipt)
    _need(preliminary.get("artifact", {}).get("sha256") == expected_sha256,
          "Student report acquisition does not bind the exact public request")
    size = preliminary.get("artifact", {}).get("size_bytes")
    _need(type(size) is int and 0 < size <= MAX_REPORT_BYTES, "Student report exceeds the bounded HTML size")
    _need((store.root / "objects" / expected_sha256).stat().st_size == size, "Student report size differs")
    record = store.verify_record_closure([receipt])[receipt]
    _validate_report_acquisition(record, scope=scope, identifier=identifier, expected_sha256=expected_sha256)
    parameters = record["parameters"]
    path = store.verify_artifact(expected_sha256)
    _need(path.stat().st_size == size, "Student report size differs")
    raw = _source_bytes(path, expected_sha256, size)
    notes = []
    if record["source_metadata"].get("version") is None:
        notes.append("original_acquisition_edition_unspecified_verified_from_request_and_report_heading")
    if parameters.get("http_method") is None:
        notes.append("original_http_method_not_explicitly_recorded")
    return raw, tuple(notes)


def read_student_report(store: ProvenanceStore, receipt: str, *, scope: str, identifier: str,
                        expected_sha256: str) -> StudentReport:
    """Read the current population table from the verified Student Information bytes."""
    raw, notes = _read_student_input(store, receipt, scope=scope, identifier=identifier,
                                     expected_sha256=expected_sha256)
    report = parse_student_report(raw, scope=scope, identifier=identifier, receipt=receipt)
    return replace(report, acquisition_notes=notes)


def projection(reports: Sequence[StudentReport], policy: ValuePolicy, *, audit_raw: bool = False) -> tuple[pa.Table, dict[str, Any], dict[str, Any]]:
    """Project only reviewed fields; retain exact raw/status and cell locators."""
    _need(bool(reports) and len({r.scope for r in reports}) == 1, "Select one nonempty source scope")
    scope = reports[0].scope
    _need(len({r.identifier for r in reports}) == len(reports), "Duplicate student-report entity identifier")
    _need(scope != "state" or len(reports) == 1, "Texas requires one direct publisher report")
    fields = semantic_fields()
    values: dict[str, list[Any]] = {}
    support: dict[str, Any] = {"fields": {}, "context": ["tprs_report_receipt", "tprs_report_sha256"]}
    if scope != "state":
        key = scope + "_number"
        values[key] = [r.identifier for r in reports]
        support["context"].append(key)
    values["tprs_report_receipt"] = [r.receipt for r in reports]
    values["tprs_report_sha256"] = [r.sha256 for r in reports]
    index = [{cell.field: cell for cell in r.cells if cell.field is not None} for r in reports]
    holds = []
    for r in reports:
        for cell in r.cells:
            if cell.geography_role == "local" and cell.field is None:
                interpreted = cell.measurement(policy)
                holds.append({"receipt": r.receipt, "row": cell.row, "column": cell.column,
                    "section": cell.section, "label": cell.label, "population": cell.population,
                    "measure": cell.measure, "reason": cell.hold,
                    "token_status": interpreted.status.value, "token_reason": interpreted.reason})
                if audit_raw:
                    holds[-1]["raw"] = cell.raw
    arrays = {}
    for f in fields:
        name = f["field"]
        columns: dict[str, list[Any]] = {name: [], "raw": [], "status": [], "row": [], "column": [], "reason": []}
        for current in index:
            cell = current.get(name)
            value = (cell.measurement(policy) if cell else
                     Measurement(None, ValueStatus.MISSING, None, "row_not_published", policy.policy_id, unit=f["unit"]))
            _need(value.status != ValueStatus.INVALID, "Unreviewed token in a qualified student field: " + name)
            columns[name].append(value.value)
            columns["raw"].append(value.raw)
            columns["status"].append(value.status.value)
            columns["reason"].append(value.reason)
            columns["row"].append(cell.row if cell else None)
            columns["column"].append(cell.column if cell else None)
        arrays[name] = pa.array(columns[name], type=pa.int64() if f["unit"] == "count" else pa.float64())
        pointers = {}
        for kind in ("raw", "status", "row", "column", "reason"):
            column = f"tprs_{kind}_{name}"
            arrays[column] = pa.array(columns[kind], type=pa.int64() if kind in {"row", "column"} else pa.string())
            pointers[kind] = column
        support["fields"][name] = {"raw": pointers["raw"], "status": pointers["status"],
            "auxiliary": [pointers[kind] for kind in ("row", "column", "reason")]}
    arrays.update({name: pa.array(value, type=pa.string()) for name, value in values.items()})
    return pa.table(arrays), support, {"qualified_fields": len(fields), "selected_reports": len(reports),
        "raw_observations": sum(len(r.cells) for r in reports), "raw_field_holds": holds,
        "whole_report_semantics_qualified": False, "projection_qualified": True}


def _crosswalk(inputs: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    fields: list[dict[str, Any]] = [{"position": i, "code": f["label"], "label": f["label"], "field": f["field"],
               "dictionary_status": "not_documented", "publisher_type": None, "publisher_description": None,
               "disposition": "fact"} for i, f in enumerate(semantic_fields(), 1)]
    tables = [{"table": scope, "header_identity": "position", "input_sha256s": list(hashes),
               "metric_period": PERIOD.to_dict(), "source_schema_sha256": source_schema_fingerprint(
                   [f["label"] for f in fields], [f["code"] for f in fields]), "fields": fields}
              for scope, hashes in inputs.items()]
    result = {"schema_version": 2, "mapping_version": 1, "datasets": [
        {"dataset": DATASET, "edition": "2026", "source_only": True, "tables": tables}]}
    parse_column_schema(canonical_json(result).encode())
    return result


def _workbook_headers(cells: Mapping[tuple[int, str], str], grid: Mapping[str, Any], scope: str,
                      identifier: str) -> None:
    def row(number: int) -> list[str]:
        return [cells.get((number, chr(ord("A") + index)), "") for index in range(len(grid["headers"][scope]))]

    _need(cells.get((1, "A")) == "Texas Education Agency"
          and cells.get((2, "A")) == "2025-26 Student Information", "Student workbook edition differs")
    heading = cells.get((3, "A"), "")
    _need(heading == "State" if scope == "state" else re.findall(r"\(([0-9]+)\)", heading) == [identifier],
          "Student workbook geography differs")
    populations = [""] * len(grid["headers"][scope])
    for index, (population, quantity, role) in enumerate(grid["column_roles"][scope], 1):
        if quantity == "count" and role == "local":
            populations[index] = population.title()
    _need(row(5) == populations, "Student workbook population headers differ")
    header = 6 if scope == "state" else 7
    _need(row(header) == grid["headers"][scope], "Student workbook quantity/benchmark headers differ")
    if scope != "state":
        geography = [""] * len(grid["headers"][scope])
        for index, label in enumerate(grid["geography_headers"][scope][1:]):
            geography[index * 2 + 1] = label
        _need(row(6) == geography, "Student workbook geography role headers differ")


def verify_workbook_witnesses(store: ProvenanceStore) -> dict[str, Any]:
    """Independently decode the publisher's OOXML for the complete current grid.

    This is presentation agreement for three named exports, not a census of
    every selected report. Malformed numeric-typed dot cells remain text.
    """
    from .financial_leaves import NS, _package, _xml
    from zipfile import ZipFile

    grid = definition()
    comparisons = []
    for scope, witness in grid["witnesses"].items():
        report = read_student_report(store, witness["html_receipt"], scope=scope,
            identifier=witness["identifier"], expected_sha256=witness["html_sha256"])
        record = store.verify_record_closure([witness["xlsx_receipt"]])[witness["xlsx_receipt"]]
        url = urlsplit(record["requested_url"])
        pairs = parse_qsl(url.query, keep_blank_values=True)
        query = dict(pairs)
        expected = {"_debug": "0", "_program": "perfrept.perfmast.sas", "_service": "marykay",
                    "ccyy": "2026", "id": witness["identifier"], "lev": SCOPES[scope],
                    "prgopt": "reports/tapr/student.sas", "dest": "E"}
        _need(record["kind"] == "acquisition" and record["method"] == "download"
              and record["section"] == "reference"
              and (record["requested_year"], record["resolved_year"]) == (EDITION, EDITION)
              and record["source_metadata"].get("version") == "2026"
              and record["artifact"]["sha256"] == witness["xlsx_sha256"]
              and record["source_metadata"].get("provenance_ids") == [witness["html_receipt"]]
              and url.scheme == "https" and url.netloc == "rptsvr1.tea.texas.gov"
              and url.path == "/cgi/sas/broker" and not url.fragment
              and record["resolved_url"] == record["requested_url"]
              and len(pairs) == len(query) and query == expected
              and record["parameters"].get("http_method") == "GET"
              and record["parameters"].get("http_status") == 200 and bool(record["downloaded_at"])
              and record["response_headers"].get("content-type") == "application/ms-excel",
              "Student workbook witness is not the exact separately requested export")
        path = store.verify_artifact(witness["xlsx_sha256"])
        _need(path.stat().st_size == record["artifact"]["size_bytes"] and path.stat().st_size <= MAX_REPORT_BYTES,
              "Student workbook witness exceeds its bounded exact size")
        with ZipFile(BytesIO(_source_bytes(path, witness["xlsx_sha256"], record["artifact"]["size_bytes"]))) as archive:
            sheets, strings = _package(archive)
            _need(set(sheets) == {"Student Information"}, "Student workbook sheets differ")
            sheet = _xml(archive, sheets["Student Information"])
        cells: dict[tuple[int, str], str] = {}
        addresses = set()
        for row in sheet.findall(NS + "sheetData/" + NS + "row"):
            for cell in row.findall(NS + "c"):
                address = re.fullmatch(r"([A-Z]+)([0-9]+)", cell.attrib["r"])
                _need(address is not None and cell.find(NS + "f") is None, "Student workbook cell differs")
                assert address is not None
                key = int(address[2]), address[1]
                _need(key not in addresses and cell.get("t", "n") in {"n", "s"}, "Student workbook cell identity/type differs")
                addresses.add(key)
                raw = cell.findtext(NS + "v")
                if raw is not None:
                    text = strings[int(raw)] if cell.get("t", "n") == "s" else raw
                    cells[key] = " ".join(text.split())
        _workbook_headers(cells, grid, scope, witness["identifier"])
        labels: dict[tuple[str, str], int] = {}
        expected_rows = {(cell.section, cell.label) for cell in report.cells}
        section = ""
        for number in sorted({number for number, _ in cells}):
            label = cells.get((number, "A"), "")
            if label == "Mobility (2024-25)":
                break
            if label in grid["sections"]:
                section = label
            elif (section, label) in expected_rows:
                _need((section, label) not in labels, "Student workbook row label repeats")
                labels[section, label] = number
        grade_cells = 0
        for cell in report.cells:
            number = labels[cell.section, cell.label]
            column = chr(ord("A") + cell.column - 1)
            _need(cells.get((number, column), "") == cell.raw, "Student HTML/workbook current cell differs")
            grade_cells += cell.section == "Students by Grade"
        comparisons.append({"scope": scope, "html_receipt": report.receipt,
            "xlsx_receipt": witness["xlsx_receipt"], "current_cells": len(report.cells), "grade_cells": grade_cells})
    return {"status": "PASS_COMPLETE_CURRENT_GRID_HTML_XLSX_WITNESSES", "comparisons": comparisons,
            "scope": "Three named publisher exports only; literal dots/blank cells remain uninterpreted raw cells"}


_STATEWIDE_BATCH_ROWS = 16
_STATEWIDE_ROW_GROUP_ROWS = 128
_LEAF_BATCH_BYTES = 8 * 1024**2


def _statewide_row_groups(batches: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
    """Pack small parser batches into bounded Parquet row groups in source order."""
    def combine(parts: list[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        if len(parts) == 1:
            yield parts[0]
            return
        combined = pa.Table.from_batches(parts).combine_chunks()
        yield from combined.to_batches() if combined.nbytes <= _LEAF_BATCH_BYTES else parts

    pending: list[pa.RecordBatch] = []
    rows = size = 0
    for batch in batches:
        if pending and (rows + batch.num_rows > _STATEWIDE_ROW_GROUP_ROWS or size + batch.nbytes > _LEAF_BATCH_BYTES):
            yield from combine(pending)
            pending.clear()
            rows = size = 0
        pending.append(batch)
        rows += batch.num_rows
        size += batch.nbytes
    if pending:
        yield from combine(pending)


def _statewide_batches(store: ProvenanceStore, entries: Sequence[ReportSelection], policy: ValuePolicy
                       ) -> Iterator[tuple[pa.Table, dict[str, Any], dict[str, Any]]]:
    for offset in range(0, len(entries), _STATEWIDE_BATCH_ROWS):
        reports = [read_student_report(store, item.receipt, scope=item.scope, identifier=item.identifier,
                                       expected_sha256=item.sha256)
                   for item in entries[offset:offset + _STATEWIDE_BATCH_ROWS]]
        yield projection(reports, policy, audit_raw=True)


def _verify_statewide_projection(store: ProvenanceStore, entries: tuple[ReportSelection, ...],
                                 policy: ValuePolicy, path: Path, audit_path: Path) -> None:
    from .dataset_metadata import COLUMN_SCHEMA_MAX_BYTES
    from .tprs_statewide_intake import audit_rows

    with audit_path.open("rb") as stream:
        raw = stream.read(COLUMN_SCHEMA_MAX_BYTES + 1)
    _need(len(raw) <= COLUMN_SCHEMA_MAX_BYTES, "Compressed student audit exceeds its bounds")
    audit = audit_rows(raw, entries)
    batches = pq.ParquetFile(path).iter_batches(batch_size=_STATEWIDE_BATCH_ROWS)
    for projection_result, observed in zip_longest(_statewide_batches(store, entries, policy), batches):
        _need(projection_result is not None and observed is not None, "Statewide student readback row count differs")
        assert projection_result is not None and observed is not None
        table, _, audit_data = projection_result
        _need(table.equals(pa.Table.from_batches([observed]).replace_schema_metadata(None)),
              "Statewide student native readback differs from the original reports")
        for expected in audit_data["raw_field_holds"]:
            _need(next(audit, None) == expected, "Statewide student raw audit differs from the original report")
    _need(next(audit, None) is None, "Statewide student raw audit has extra rows")


def materialize_students(store: ProvenanceStore, *, config: Path, config_sha256: str,
                         intake: Path, intake_sha256: str, output: Path) -> dict[str, Any]:
    """Build a new, explicit receipt-selected native projection entirely offline."""
    from .analytical import write_leaf_artifact
    from .release import _install_candidate, _schema_publication_destinations, publish_release
    from .tprs_statewide_intake import (
        CAMPUS_COHORT_LEDGER_VERSION, CAMPUS_COHORT_OPERATION, CAMPUS_COHORT_VERSION, CAMPUS_INTAKE_VERSION,
        CAMPUS_PARTITION_LEDGER_VERSION, CAMPUS_PARTITION_OPERATION, CAMPUS_PARTITION_VERSION,
        STATEWIDE_OPERATION, ReportSelection, admit_campus_cohort, admit_statewide_selection, validate_campus_selection,
    )

    _need(not output.exists() and not output.is_symlink(), "Student candidate requires a new output directory")
    with config.open("rb") as stream:
        raw = stream.read(2 * 1024**2 + 1)
    _need(len(raw) <= 2 * 1024**2 and sha256(raw).hexdigest() == config_sha256, "Student Config pin differs")
    cfg = load_config_bytes(raw, path=config)
    source = cfg.resolve_dataset(DATASET, EDITION, reader_profile="source", exact=True)
    reviewed = load_config().resolve_dataset(DATASET, EDITION, reader_profile="source", exact=True)
    # This reader opens pinned acquisitions, not the resolved path placeholder.
    _need(replace(source, path=reviewed.path) == reviewed and cfg.is_source_only(DATASET, EDITION),
          "Student source differs from packaged definition")
    policy_id = source.metadata.value_policy
    if policy_id is None:
        raise ContractError("Student source requires its reviewed value policy")
    policy = cfg.value_policies[policy_id]
    _need(policy == load_config().value_policies[policy_id], "Student value policy changed")
    resource = cfg.column_contract(DATASET, EDITION, reader_profile="source")
    _need(resource == load_config().column_contract(DATASET, EDITION, reader_profile="source"), "Student field binding changed")
    expected_metrics = {field["field"]: (field["unit"], PERIOD) for field in semantic_fields()}
    _need({metric.field: (metric.unit, metric.period) for metric in source.metadata.metrics.values()} == expected_metrics,
          "Student semantic bindings differ from the reviewed report grid")
    with intake.open("rb") as stream:
        body = stream.read(4 * 1024**2 + 1)
    _need(0 < len(body) <= 4 * 1024**2 and sha256(body).hexdigest() == intake_sha256, "Student intake pin differs")
    selection = json.loads(body, object_pairs_hook=_unique_object)
    version = selection.get("schema_version") if isinstance(selection, dict) else None
    campus_cohort = version in (CAMPUS_COHORT_VERSION, CAMPUS_PARTITION_VERSION)
    campus_partition = None
    roster_selection = None
    if campus_cohort:
        _need(body == (canonical_json(selection) + "\n").encode(), "Campus cohort requires a canonical JSON intake")
        roster_selection = admit_campus_cohort(store, selection)
        if roster_selection.partition is not None:
            region, = roster_selection.roster
            campus_partition = {"region": region, "ordinal": roster_selection.partition}
    elif version == 2:
        roster_selection = admit_statewide_selection(store, selection)
    campus_receipt = None
    if roster_selection is None:
        _need(isinstance(selection, dict) and type(selection.get("schema_version")) is int
              and selection["schema_version"] in {1, CAMPUS_INTAKE_VERSION}
              and set(selection) == ({"schema_version", "dataset", "edition", "reports", "region", "roster_receipt"}
                                     if selection["schema_version"] == CAMPUS_INTAKE_VERSION
                                     else {"schema_version", "dataset", "edition", "reports"})
              and selection["dataset"] == DATASET and selection["edition"] == "2026"
              and isinstance(selection["reports"], list) and 0 < len(selection["reports"]) <= 20000,
              "Invalid selected-report intake")
        if selection["schema_version"] == CAMPUS_INTAKE_VERSION:
            campus_receipt = validate_campus_selection(store, selection)
    groups: dict[str, list[StudentReport]] = {}
    selected_bytes = 0
    for item in (() if roster_selection else selection["reports"]):
        _need(isinstance(item, dict) and set(item) == {"receipt", "sha256", "scope", "identifier"}, "Invalid report selection")
        _need(all(isinstance(value, str) for value in item.values())
              and _DIGEST.fullmatch(item["receipt"]) is not None and _DIGEST.fullmatch(item["sha256"]) is not None,
              "Report selections require literal identities and exact pins")
        _identifier(item["scope"], item["identifier"])
        size = store.read_record(item["receipt"])["artifact"]["size_bytes"]
        _need(type(size) is int and 0 < size <= MAX_REPORT_BYTES, "Student report exceeds the bounded HTML size")
        selected_bytes += size
        _need(selected_bytes <= MAX_SELECTION_BYTES, "Selected student reports exceed the bounded aggregate size")
        report = read_student_report(store, item["receipt"], scope=item["scope"], identifier=item["identifier"],
                                     expected_sha256=item["sha256"])
        groups.setdefault(report.scope, []).append(report)
    # Legacy selections fit in memory. Roster-bound projection validates in a
    # private stage and installs nothing until the complete readback passes.
    projections = {scope: projection(reports, policy) for scope, reports in groups.items()}
    if roster_selection:
        scopes = ("campus",) if campus_cohort else ("district", "state")
        source_groups = {scope: tuple(item for item in roster_selection.reports if item.scope == scope)
                         for scope in scopes}
    else:
        source_groups = {scope: tuple(ReportSelection(r.scope, r.identifier, r.receipt, r.sha256) for r in reports)
                         for scope, reports in groups.items()}
    witnesses = verify_workbook_witnesses(store)
    crosswalk = _crosswalk({scope: [r.sha256 for r in reports] for scope, reports in source_groups.items()})
    roots = [r.receipt for reports in source_groups.values() for r in reports]
    roster_receipts = [entry["receipt"] for entry in roster_selection.roster.values()] if roster_selection else []
    if campus_receipt is not None:
        roster_receipts.append(campus_receipt)
    roots.extend(roster_receipts)
    roots.extend(rid for ref in source.metadata.references for rid in ref.provenance_ids)
    roots.extend(w[role] for w in definition()["witnesses"].values() for role in ("html_receipt", "xlsx_receipt"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".tprs-students-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir(mode=0o700)
        target = ProvenanceStore(stage / "provenance")
        roster_path = grid_path = None
        grid_bytes = files("teadata").joinpath(RESOURCE).read_bytes()
        grid_digest = sha256(grid_bytes).hexdigest()
        if roster_selection:
            roster_bytes = (canonical_json(roster_selection.roster) + "\n").encode()
            roster_path = stage / f"tprs-roster-{sha256(roster_bytes).hexdigest()}.json"
            roster_path.write_bytes(roster_bytes)
            grid_path = stage / f"tprs-grid-{grid_digest}.json"
            grid_path.write_bytes(grid_bytes)
        for rid in dict.fromkeys(roots):
            target.copy_record_from(store, rid)
        captured = Path(temporary) / "intake.json"
        captured.write_bytes(body)
        intake_receipt = target.record_acquisition(captured, ResolvedDataset("reference:tprs_student_intake", "reference",
            EDITION, EDITION, "", DatasetMetadata(title="Exact selected public student reports", version="2026")))
        captured_config = Path(temporary) / "source-config.input"
        captured_config.write_bytes(raw)
        config_receipt = target.record_acquisition(captured_config, ResolvedDataset("reference:tprs_student_config", "reference",
            EDITION, EDITION, "", DatasetMetadata(title="Exact source configuration for selected student reports", version="2026")))
        # Only actual ordered source hashes and frozen receipt coverage differ.
        schema_path = Path(temporary) / "schemas/tprs-students-selected.json"
        schema_path.parent.mkdir()
        schema_bytes = (canonical_json(crosswalk) + "\n").encode()
        schema_path.write_bytes(schema_bytes)
        schema_parents = [rid for ref in source.metadata.references if ref.reference_id == "columns" for rid in ref.provenance_ids]
        schema_receipt = target.record_acquisition(schema_path, ResolvedDataset(
            "reference:tprs_student_selected_projection_schema", "reference", EDITION, EDITION, "",
            DatasetMetadata(title="Locally derived ordered-input projection of the packaged student field contract", version="2026",
                provenance_ids=tuple([*schema_parents, intake_receipt, config_receipt]))))
        document = deepcopy(yaml.safe_load(raw))
        entry = document["data_sources"][DATASET][EDITION]
        schema_ref = next(ref for ref in entry["metadata"]["references"] if ref["reference_id"] == "columns")
        schema_ref.update(path="schemas/tprs-students-selected.json", reviewed_sha256=sha256(schema_bytes).hexdigest(),
                          provenance_ids=[schema_receipt])
        entry["reader"]["profiles"]["source"]["column_schema"] = "schemas/tprs-students-selected.json"
        dynamic_raw = yaml.safe_dump(document, sort_keys=False).encode()
        dynamic_cfg = load_config_bytes(dynamic_raw, path=Path(temporary) / "selected-config.yaml")
        selected = dynamic_cfg.resolve_dataset(DATASET, EDITION, reader_profile="source", exact=True)
        recorder = BuildRecorder(dynamic_cfg, target)
        results = []
        edition_keys = {
            scope: EditionKey(DATASET, "2026", EditionKey.source_revision(tuple(r.sha256 for r in entries)), scope, scope,
                              STATEWIDE_SINGLETON_GRAIN if scope == "state" else f"one row per {scope}_number")
            for scope, entries in source_groups.items()
        }
        for scope, reports in source_groups.items():
            if roster_selection:
                chunks = _statewide_batches(target, reports, policy)
                first = next(chunks)
                table, support, _ = first
            else:
                table, support, audit = projections[scope]
            hashes = [r.sha256 for r in reports]
            keys = [] if scope == "state" else [scope + "_number"]
            key = edition_keys[scope]
            recipe: dict[str, Any] = {"operation": "selected_tprs_student_membership_enrollment",
                "definition_sha256": grid_digest, "intake_sha256": intake_sha256,
                "source_locators": "one-based parsed HTML row and column", "projection_qualified": True,
                "whole_report_semantics_qualified": False}
            if campus_partition is not None:
                recipe.update(operation=CAMPUS_PARTITION_OPERATION, report_ledger_version=CAMPUS_PARTITION_LEDGER_VERSION,
                              parquet_row_group_max_rows=_STATEWIDE_ROW_GROUP_ROWS, campus_partition=campus_partition)
            elif campus_cohort:
                assert roster_selection is not None
                recipe.update(operation=CAMPUS_COHORT_OPERATION, report_ledger_version=CAMPUS_COHORT_LEDGER_VERSION,
                              parquet_row_group_max_rows=_STATEWIDE_ROW_GROUP_ROWS,
                              campus_regions=sorted(roster_selection.roster))
            elif roster_selection:
                recipe.update(operation=STATEWIDE_OPERATION, report_ledger_version=2,
                              parquet_row_group_max_rows=_STATEWIDE_ROW_GROUP_ROWS,
                              cohort={name: edition.identity for name, edition in edition_keys.items()})
            toolchain = {"python": sys.version, "pyarrow": str(pa.__version__), "platform": platform.platform()}
            contract = recorder.leaf_contract(key, selected, native_schema=table.schema.serialize().to_pybytes(),
                columns=table.column_names, key_columns=keys, rows=len(reports), inputs=hashes, recipe=recipe,
                toolchain=toolchain, column_support=support)
            identity = sha256(canonical_json(contract).encode()).hexdigest()
            path = stage / (identity + ".parquet")
            outputs = {"leaf": path}
            observations = [r.receipt for r in reports] if roster_selection else []
            if roster_selection:
                assert roster_path is not None and grid_path is not None
                audit_tmp = stage / f"{scope}-raw-audit.tmp"
                audit = {"qualified_fields": len(support["fields"]), "selected_reports": len(reports),
                         "raw_observations": 0, "raw_field_hold_count": 0,
                         "whole_report_semantics_qualified": False, "projection_qualified": True}
                with audit_tmp.open("xb") as compressed, gzip.GzipFile(filename="", fileobj=compressed, mode="wb", mtime=0) as stream:
                    def batches():
                        for chunk, _, part in chain([first], chunks):
                            audit["raw_observations"] += part["raw_observations"]
                            audit["raw_field_hold_count"] += len(part["raw_field_holds"])
                            for hold in part["raw_field_holds"]:
                                stream.write((canonical_json(hold) + "\n").encode())
                            yield from chunk.to_batches()

                    physical = write_leaf_artifact(path, _statewide_row_groups(batches()), contract=contract,
                                                   batch_bytes=_LEAF_BATCH_BYTES)
                _verify_statewide_projection(target, reports, policy, path, audit_tmp)
                audit_path = stage / f"tprs-raw_audit-{sha256(audit_tmp.read_bytes()).hexdigest()}.jsonl.gz"
                audit_tmp.rename(audit_path)
                outputs.update(roster=roster_path, raw_audit=audit_path, grid=grid_path)
            else:
                physical = write_leaf_artifact(path, table.to_batches(max_chunksize=512), contract=contract,
                                               batch_bytes=_LEAF_BATCH_BYTES)
                _need(table.equals(pq.read_table(path).replace_schema_metadata(None)), "Student native readback differs")
            for report in groups.get(scope, ()):
                observation = target.record_acquisition(target.verify_artifact(report.sha256), ResolvedDataset(
                    DATASET, "data_sources", EDITION, EDITION, "", DatasetMetadata(
                        title="Locally verified retained2026 TPRS student report", version="2026", period=PERIOD,
                        publisher="Texas Education Agency", provenance_ids=(report.receipt,),
                        notes="Local observation after request, heading and grid validation. Original download receipt and time remain unchanged.")),
                    parameters={"original_download_receipt": report.receipt, "scope": scope,
                                "identifier": report.identifier, "original_acquisition_notes": list(report.acquisition_notes)})
                observations.append(observation)
            processing = target.record_processing(dataset=DATASET, inputs=list(dict.fromkeys(
                [*[r.receipt for r in reports], *observations, *roster_receipts, intake_receipt, config_receipt])),
                outputs=outputs, processor="teadata.tprs_students", code_version=recorder.processor_sha256,
                parameters={"recipe": recipe, "leaf_contract_sha256": identity, "toolchain": toolchain},
                validation={**audit, "release_activation": False})
            ref = ArtifactRef(key, physical["sha256"], physical["size_bytes"], path.name, canonical_json(contract), processing, "leaf")
            recorder.record_edition(key, selected, artifacts=[ref], acquisition_ids=observations)
            results.append({"edition": key.to_dict(), "rows": len(reports), **audit,
                "original_acquisition_notes": [{"receipt": report.receipt, "notes": list(report.acquisition_notes)}
                                               for report in groups.get(scope, ()) if report.acquisition_notes]})
        catalog = recorder.catalog(schema_version=4)
        publish_release(stage, catalog, schema_version=4)
        result = {"status": "PASS_REVIEWED_PROJECTION_WITH_EXPLICIT_RAW_HOLDS", "release_activation": False,
                  "whole_report_semantics_qualified": False, "publisher_witnesses": witnesses, "editions": results}
        if roster_selection:
            _, locations = _schema_publication_destinations(catalog)
            result["release_assets"] = ["release.json", *locations]
        if campus_partition is not None:
            result["campus_partition"] = campus_partition
        elif campus_cohort:
            assert roster_selection is not None
            result["campus_regions"] = sorted(roster_selection.roster)
        (stage / "MATERIALIZATION.json").write_text(canonical_json(result) + "\n", encoding="utf-8")
        _install_candidate(stage, output)
        return result
