"""Materialize the reviewed FY2025 campus actual report layout as raw observations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import sys
import tempfile
from typing import Any
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile, ZipFile

import pyarrow as pa
from pyarrow import compute
import pyarrow.parquet as pq

from .column_schema import _unique_object
from .dataset_metadata import DataPeriod, DatasetMetadata, ResolvedDataset
from .financial_leaves import NS, _package, _xml
from .manifest import file_identity
from .provenance import ProvenanceStore, validate_record_graph
from .scripts.download_campus_peims_financials import (
    DATASET, REPORTS, SECTION_LABELS, _broker_url, _parse_selector, _validate_attachment,
)


_YEAR = 2025
_FISCAL_YEAR = "2024-2025"
_PERIOD = DataPeriod(kind="fiscal_year", start_year=2024, end_year=_YEAR)
_INDEX = f"{REPORTS}/2425_new_camp_actual.html"
_ROSTER_PROGRAM = "sfadhoc.new_pop_campusbox_actual25.sas"
_WORKBOOK_PROGRAM = "sfadhoc.new_Campus_actual25.sas"
_REFERENCE_URLS = {
    "campus_actual_selector": _INDEX,
    "about_actual": f"{REPORTS}/AbtAct25.docx",
    "raw_account_dictionary": "https://tea.texas.gov/WorkArea/linkit.aspx?ItemID=2147494868&LinkIdentifier=id",
    "campus_accounting": "https://tea.texas.gov/data-reports/financial-accountability/fasrg/fasrg19-module1-far-appendices-0.pdf#page=209",
    "related_school_report_card_definitions": "https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/2025-src-definitions-english.pdf#page=2",
}
_SHEET = "2024 - 2025 PEIMS Campus Dat"
_MAX_INPUT_BYTES = 2 * 1024**2
_BATCH_CAMPUSES = 32
_COLUMNS = (
    ("B", "General\nFund", "general_fund_amount"),
    ("C", "%", "general_fund_share"),
    ("D", "Per\nStudent", "general_fund_per_student"),
    ("E", "All\nFunds", "all_funds_amount"),
    ("F", "%", "all_funds_share"),
    ("G", "Per\nStudent", "all_funds_per_student"),
)
_SECTIONS = (
    (6, "Expenditures\u00a0by\u00a0Object\u00a0\u00a0\u00a0(Objects\u00a06100-6600)", (
        "Total Expenditures", "Operating-Payroll", "Other Operating", "Non-Operating(Equipt/Supplies)",
    )),
    (11, "Expenditures\u00a0by\u00a0Function\u00a0(Objects\u00a06100-6400\u00a0Only)", (
        "Total Operating Expenditures", "Instruction (11,95)             *",
        "Instructional Res/Media (12)    *", "Curriculum/Staff Develop (13)   *",
        "Instructional Leadership (21)   *", "School Leadership (23)          *",
        "Guidance/Counseling Svcs (31)   *", "Social Work Services (32)       *",
        "Health Services (33)            *", "Transportation(34)              *",
        "Food (35)                      **", "Extracurricular (36)           **",
        "Plant Maint/Operation (51)     **", "Security/Monitoring (52)       **",
        "Data Processing Svcs  (53)    **", "Fund Raising CHARTER ONLY (81) **",
    )),
    (28, "Program\u00a0expenditures\u00a0by\u00a0Program\u00a0\u00a0(Objects\u00a06100-6400\u00a0only)", (
        "Total Operating Expenditures", "Regular", "Gifted & Talented", "Career & Technical",
        "Students with Disabilities", "State Compensatory ED", "Bilingual", "Early Education Allotment",
        "Dyslexia or Related Disorder Serv", "CCMR", "Athletic Programming", "Un-Allocated",
    )),
)
_MERGES = {"A1:G1", "A2:G2", "A3:G3", *(f"A{row}:G{row}" for row, _, _ in _SECTIONS)}
_AMOUNT = re.compile(r"-?[0-9]+")
_SHARE = re.compile(r"-?[0-9]+(?:\.[0-9]{1,4})?")
_MEMBERSHIP = re.compile(r"Total Enrolled Membership: ([0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)")
_AMOUNT_FORMAT = '"$"##,###,###,###,###,###,##0'
_SHARE_FORMAT = "#0.00%;(#0.00%)"
_FACTS_PER_CAMPUS = len(_COLUMNS) * sum(len(labels) for _, _, labels in _SECTIONS)
_pc: Any = compute  # Arrow registers its compute kernels dynamically.
_REPORT_SCHEMA = pa.schema([
    ("fiscal_end_year", pa.int16()), ("district_number", pa.string()), ("campus_number", pa.string()),
    ("source_receipt_id", pa.string()), ("worksheet", pa.string()), ("report_title", pa.string()),
    ("membership_cell", pa.string()), ("membership_raw", pa.string()), ("reported_membership", pa.int64()),
])
_FACT_SCHEMA = pa.schema([
    ("fiscal_end_year", pa.int16()), ("campus_number", pa.string()), ("source_receipt_id", pa.string()),
    ("worksheet", pa.string()), ("section_row", pa.int16()), ("section_label", pa.string()),
    ("line_label", pa.string()), ("line_occurrence", pa.int16()), ("source_row", pa.int16()),
    ("source_cell", pa.string()), ("column_label", pa.string()), ("column_role", pa.string()),
    ("raw_value", pa.string()), ("normalization_hold_reason", pa.string()),
])


@dataclass(frozen=True)
class _Workbook:
    district: str
    campus: str
    receipt_id: str
    artifact: Mapping[str, Any]


def _read_bounded(path: Path, expected_sha256: str, size: int | None = None) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Financial input must be a regular file")
        data = stream.read(_MAX_INPUT_BYTES + 1)
    if (len(data) > _MAX_INPUT_BYTES or (size is not None and size != len(data))
            or sha256(data).hexdigest() != expected_sha256):
        raise ValueError("Financial input differs from its pinned size or SHA-256")
    return data


def _check_receipt(record: Mapping[str, Any], url: str, role: str, *, parent: str | None = None,
                   district: str | None = None, campus: str | None = None) -> None:
    parameters = record.get("parameters") or {}
    metadata = record.get("source_metadata") or {}
    if (record.get("kind") != "acquisition" or record.get("method") != "download"
            or record.get("dataset") != DATASET or record.get("section") != "data_sources"
            or record.get("requested_year") != _YEAR or record.get("resolved_year") != _YEAR
            or record.get("requested_url") != url or record.get("resolved_url") != url
            or parameters.get("adapter") != DATASET or parameters.get("adapter_revision") != 1
            or parameters.get("role") != role or parameters.get("district_id") != district
            or parameters.get("campus_id") != campus or parameters.get("http_method") != "GET"
            or parameters.get("http_status") != 200 or metadata.get("source_url") != _INDEX
            or metadata.get("download_url") != url or metadata.get("period") != _PERIOD.to_dict()
            or metadata.get("provenance_ids") != ([parent] if parent else [])):
        raise ValueError(f"Financial {role} receipt does not match the publisher selection")
    if parent is not None and (metadata.get("source_qualifiers") or {}).get("financial_basis") != "actual":
        raise ValueError("Campus financial intake requires explicit publisher actuals")


def _admit_intake(store: ProvenanceStore, data: bytes) -> list[_Workbook]:
    manifest = json.loads(data, object_pairs_hook=_unique_object)
    if (not isinstance(manifest, dict) or set(manifest) != {"fiscal_end_year", "district_index_receipt", "publisher_district_count", "districts"}
            or type(manifest["fiscal_end_year"]) is not int or manifest["fiscal_end_year"] != _YEAR
            or type(manifest["publisher_district_count"]) is not int or not isinstance(manifest["districts"], dict)
            or not manifest["districts"]):
        raise ValueError("Select an exact FY2025 campus financial actuals intake manifest")
    roots = [manifest["district_index_receipt"]]
    for district, scope in manifest["districts"].items():
        if (not re.fullmatch(r"[0-9]{6}", district) or not isinstance(scope, dict)
                or set(scope) != {"roster_receipt", "listed_campuses", "workbook_receipts"}
                or not isinstance(scope["workbook_receipts"], dict) or not scope["workbook_receipts"]
                or not isinstance(scope["listed_campuses"], list)
                or any(not isinstance(campus, str) or not re.fullmatch(rf"{district}[0-9]{{3}}", campus)
                       for campus in scope["listed_campuses"])
                or len(scope["listed_campuses"]) != len(scope["workbook_receipts"])
                or set(scope["listed_campuses"]) != scope["workbook_receipts"].keys()):
            raise ValueError("Financial intake must cover every listed campus in each selected district")
        roots.extend([scope["roster_receipt"], *scope["workbook_receipts"].values()])
    if any(not isinstance(root, str) for root in roots) or len(set(roots)) != len(roots):
        raise ValueError("Financial intake requires unique receipt identities")
    graph = validate_record_graph(
        lambda receipt: json.loads(_read_bounded(store.root / "records" / f"{receipt}.json", receipt)), roots,
    )
    if graph.keys() != set(roots):
        raise ValueError("Financial intake receipt closure includes unselected records")
    index_id = manifest["district_index_receipt"]
    index = graph[index_id]
    _check_receipt(index, _INDEX, "district_index")

    def selector_bytes(record: Mapping[str, Any]) -> bytes:
        artifact = record["artifact"]
        return _read_bounded(store.root / "objects" / artifact["sha256"], artifact["sha256"], artifact["size_bytes"])

    all_districts = _parse_selector(selector_bytes(index), field="who_list",
        program=_ROSTER_PROGRAM, heading=f"Select District for Financial Actual Campus {_FISCAL_YEAR} Report:")
    if manifest["publisher_district_count"] != len(all_districts) or not manifest["districts"].keys() <= set(all_districts):
        raise ValueError("Financial intake district coverage differs from the publisher's index")
    result = []
    for district, scope in sorted(manifest["districts"].items()):
        roster_id = scope["roster_receipt"]
        roster = graph[roster_id]
        _check_receipt(roster, _broker_url(_ROSTER_PROGRAM, who_list=district), "campus_index", parent=index_id, district=district)
        listed = _parse_selector(selector_bytes(roster), field="which_camp",
            program=_WORKBOOK_PROGRAM, heading="Select a Campus:", district=district)
        if tuple(scope["listed_campuses"]) != listed:
            raise ValueError("Financial intake campus coverage differs from the publisher's roster")
        for campus, receipt in sorted(scope["workbook_receipts"].items()):
            record = graph[receipt]
            url = _broker_url(_WORKBOOK_PROGRAM, single="N", batch="N", app="PUBLIC", ptype="E", which_camp=campus, search="dist")
            _check_receipt(record, url, "campus_workbook", parent=roster_id, district=district, campus=campus)
            _validate_attachment(record["response_headers"], _FISCAL_YEAR)
            result.append(_Workbook(district, campus, receipt, record["artifact"]))
    return result


def _read_report(store: ProvenanceStore, workbook: _Workbook) -> tuple[dict[str, Any], Iterator[dict[str, Any]]]:
    source_hash = workbook.artifact["sha256"]
    source = _read_bounded(store.root / "objects" / source_hash, source_hash, workbook.artifact["size_bytes"])
    with ZipFile(BytesIO(source)) as archive:
        sheets, strings = _package(archive)
        if list(sheets) != [_SHEET]:
            raise ValueError("Financial actuals worksheet title or count changed")
        root = _xml(archive, sheets[_SHEET])
        styles = _xml(archive, "xl/styles.xml")
    formats = {node.attrib["numFmtId"]: node.attrib["formatCode"] for node in styles.findall(NS + "numFmts/" + NS + "numFmt")}
    cell_styles = [formats.get(node.attrib["numFmtId"]) for node in styles.findall(NS + "cellXfs/" + NS + "xf")]
    merges = [node.attrib["ref"] for node in root.findall(NS + "mergeCells/" + NS + "mergeCell")]
    physical_rows = root.findall(NS + "sheetData/" + NS + "row")
    dimension = root.find(NS + "dimension")
    if (dimension is None or dimension.get("ref") != "A1:G40"
            or len(merges) != len(_MERGES) or set(merges) != _MERGES
            or [row.attrib.get("r") for row in physical_rows] != [str(row) for row in range(1, 42)]):
        raise ValueError("Financial actuals physical framing changed")
    cells = {}
    for row in physical_rows:
        for cell in row.findall(NS + "c"):
            coordinate = cell.attrib["r"]
            if (coordinate in cells or not re.fullmatch(rf"[A-G]{row.attrib['r']}", coordinate)
                    or cell.find(NS + "f") is not None or cell.get("t", "n") not in {"s", "n"}
                    or len(cell.findall(NS + "v")) != len(cell) or len(cell) > 1):
                raise ValueError("Financial actuals has an invalid coordinate, duplicate, type or formula")
            if (raw := cell.findtext(NS + "v")) not in (None, ""):
                cells[coordinate] = (cell.get("t", "n"), raw, cell.get("s"))
            else:
                cells[coordinate] = (None, "", None)
    values = {coordinate: value for coordinate, value in cells.items() if value[0] is not None}

    def text(coordinate: str) -> str:
        kind, raw, _ = values.pop(coordinate, (None, "", None))
        if kind != "s" or not raw.isascii() or not raw.isdecimal() or int(raw) >= len(strings):
            raise ValueError(f"Financial actuals string differs at {coordinate}")
        return strings[int(raw)]

    title = text("A1")
    lines = title.splitlines()
    member = text("A3")
    membership = _MEMBERSHIP.fullmatch(member)
    if (len(lines) != 3 or lines[0] != f"{_FISCAL_YEAR} PEIMS ACTUAL FINANCIAL DATA BY CAMPUS"
            or not re.fullmatch(rf"TOTALS FOR .+ \({workbook.campus}\)", lines[1]) or not lines[2].strip()
            or text("A2") != " " or membership is None):
        raise ValueError("Financial actuals report identity or membership framing changed")
    membership_count = int(membership[1].replace(",", ""))
    for column, header, _ in _COLUMNS:
        if text(f"{column}5") != header:
            raise ValueError(f"Financial actuals column heading differs at {column}5")
    identity = {"fiscal_end_year": _YEAR, "campus_number": workbook.campus,
                "source_receipt_id": workbook.receipt_id, "worksheet": _SHEET}
    report = {**identity, "district_number": workbook.district, "report_title": title,
              "membership_cell": "A3", "membership_raw": member, "reported_membership": membership_count}

    def facts() -> Iterator[dict[str, Any]]:
        for section_row, section_label, labels in _SECTIONS:
            if text(f"A{section_row}") != section_label or " ".join(section_label.split()) != SECTION_LABELS[f"A{section_row}"]:
                raise ValueError(f"Financial actuals section differs at A{section_row}")
            occurrences: Counter[str] = Counter()
            for row, label in enumerate(labels, section_row + 1):
                if text(f"A{row}") != label:
                    raise ValueError(f"Financial actuals line label differs at A{row}")
                occurrences[label] += 1
                for column, header, role in _COLUMNS:
                    coordinate = f"{column}{row}"
                    kind, token, style = values.pop(coordinate, (None, "", None))
                    is_share = role.endswith("_share")
                    expected_format = _SHARE_FORMAT if is_share else _AMOUNT_FORMAT
                    if (kind != "n" or not (_SHARE if is_share else _AMOUNT).fullmatch(token)
                            or style is None or not style.isascii() or not style.isdecimal()
                            or int(style) >= len(cell_styles) or cell_styles[int(style)] != expected_format):
                        raise ValueError(f"Financial actuals value or format differs at {coordinate}")
                    is_per_student = role.endswith("_per_student")
                    reason = ("zero_reported_membership" if membership_count == 0 else "unqualified_membership_denominator") if is_per_student else "unqualified_field_semantics"
                    yield {**identity, "section_row": section_row, "section_label": section_label, "line_label": label,
                           "line_occurrence": occurrences[label], "source_row": row, "source_cell": coordinate,
                           "column_label": header, "column_role": role, "raw_value": token, "normalization_hold_reason": reason}
        if values:
            raise ValueError("Financial actuals has unaccounted worksheet payload")

    return report, facts()


def _batch_hash(table: pa.Table) -> str:
    batch, = table.combine_chunks().to_batches()
    return sha256(batch.serialize()).hexdigest()


def _write_candidate(store: ProvenanceStore, stage: Path, workbooks: list[_Workbook]) -> dict[str, Any]:
    report_path, fact_path = stage / "campus_reports.parquet", stage / "campus_financial_facts.parquet"
    report_hashes, fact_hashes = [], []
    summary: Counter[str] = Counter()
    with (pq.ParquetWriter(report_path, _REPORT_SCHEMA, compression="zstd") as report_writer,
          pq.ParquetWriter(fact_path, _FACT_SCHEMA, compression="zstd") as fact_writer):
        for offset in range(0, len(workbooks), _BATCH_CAMPUSES):
            reports, facts = [], []
            for workbook in workbooks[offset:offset + _BATCH_CAMPUSES]:
                try:
                    report, observations = _read_report(store, workbook)
                    reports.append(report)
                    facts.extend(observations)
                except (ValueError, KeyError, BadZipFile, ParseError) as error:
                    raise ValueError(f"Campus {workbook.campus}: {error}") from error
            for rows, schema, writer, hashes in ((reports, _REPORT_SCHEMA, report_writer, report_hashes),
                                                   (facts, _FACT_SCHEMA, fact_writer, fact_hashes)):
                table = pa.Table.from_pylist(rows, schema=schema)
                writer.write_table(table, row_group_size=table.num_rows)
                hashes.append(_batch_hash(table))
            raw = pa.array([row["raw_value"] for row in facts], type=pa.string())
            summary["integer_lexemes"] += _pc.sum(_pc.match_substring_regex(raw, r"^-?[0-9]+$")).as_py()
            summary["decimal_lexemes"] += _pc.sum(_pc.match_substring(raw, ".")).as_py()
            summary["negative_lexemes"] += _pc.sum(_pc.match_substring_regex(raw, r"^-[0-9.]*[1-9]")).as_py()
            summary["zero_lexemes"] += _pc.sum(_pc.match_substring_regex(raw, r"^-?0+(\.0+)?$")).as_py()
            summary["zero_membership_reports"] += sum(row["reported_membership"] == 0 for row in reports)
            summary["zero_membership_per_student_holds"] += sum(row["normalization_hold_reason"] == "zero_reported_membership" for row in facts)
    for artifact, schema, hashes, count in ((report_path, _REPORT_SCHEMA, report_hashes, len(workbooks)),
                                            (fact_path, _FACT_SCHEMA, fact_hashes, len(workbooks) * _FACTS_PER_CAMPUS)):
        with pq.ParquetFile(artifact) as reader:
            if (not reader.schema_arrow.equals(schema) or reader.metadata.num_rows != count
                    or reader.num_row_groups != len(hashes)
                    or any(_batch_hash(reader.read_row_group(index)) != digest for index, digest in enumerate(hashes))):
                raise ValueError("Financial actuals Parquet readback differs from exact source projection")
    return {"rows": {"campus_reports": len(workbooks), "campus_financial_facts": len(workbooks) * _FACTS_PER_CAMPUS},
            "lexemes": dict(summary), "source_projection_readback_equal": True}


def materialize_campus_financial_workbooks(store: ProvenanceStore, *, intake: Path, intake_sha256: str,
                                           output: Path) -> dict[str, Any]:
    """Build unpublished raw observations and a processing receipt from one pinned intake."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("Campus financial candidate requires a new exclusive output directory")
    data = _read_bounded(intake, intake_sha256)
    workbooks = _admit_intake(store, data)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".campus-financials-", dir=output.parent))
    try:
        validation = _write_candidate(store, stage, workbooks)
        snapshot = stage / "campus-financial-intake.json"
        snapshot.write_bytes(data)
        intake_receipt = store.record_acquisition(snapshot, ResolvedDataset("reference:campus_financial_intake",
            "reference", _YEAR, _YEAR, "", DatasetMetadata(title="Exact locally retained campus financial intake manifest")))
        snapshot.unlink()
        outputs = {role: stage / f"{role}.parquet" for role in validation["rows"]}
        code = file_identity(Path(__file__))["sha256"]
        toolchain = {"python": sys.version, "pyarrow": pa.__version__, "platform": platform.platform(),
                     "ooxml_guard_sha256": file_identity(Path(__file__).with_name("financial_leaves.py"))["sha256"],
                     "acquisition_adapter_sha256": file_identity(Path(__file__).parent / "scripts/download_campus_peims_financials.py")["sha256"]}
        source_details = {"fiscal_period": _PERIOD.to_dict(), "financial_basis": "actual",
                          "reference_urls": _REFERENCE_URLS, "reference_review_status": "linked_not_used_to_normalize"}
        receipt = store.record_processing(dataset=DATASET, inputs=[intake_receipt, *(item.receipt_id for item in workbooks)],
            outputs=outputs, processor="teadata.campus_financial_leaves.materialize_campus_financial_workbooks",
            code_version=code, parameters={"intake_sha256": intake_sha256, "toolchain": toolchain, **source_details},
            validation={**validation, "field_correctness_qualified": False, "release_activation": False})
        graph = store.verify_record_closure([receipt])
        result = {"status": "unpublished_raw_campus_financial_candidate", "field_correctness_qualified": False,
                  "release_activation": False, "intake_sha256": intake_sha256, "processing_record_id": receipt,
                  "input_receipt_count": len(graph) - 2, "outputs": graph[receipt]["outputs"], **source_details, **validation}
        (stage / "MATERIALIZATION.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if output.exists() or output.is_symlink():
            raise FileExistsError("Campus financial candidate output appeared before completion")
        stage.rename(output)
        return result
    finally:
        if stage.exists():
            shutil.rmtree(stage)
