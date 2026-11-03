"""Explicit TEA transfer-source intake; never activates source configuration.

Regional district reports are independent publisher observations, not sums of
campus transfers. Acquisition preserves their original headings and footnotes.
"""

from __future__ import annotations

from dataclasses import dataclass
import codecs
import csv
from email.message import Message
from hashlib import sha256
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, NotRequired, Sequence, TypedDict
from urllib.parse import urljoin
from urllib.request import build_opener

from .acquisition import AcquisitionClient, NoRedirect, _form_request, acquire_form
from .dataset_metadata import DataPeriod, DatasetMetadata, ResolvedDataset
from .identifiers import canonical_source_identifiers
from .provenance import ProvenanceStore, _identifier
from .source_csv import SourceLines
from .values import ValuePolicy, ValueStatus, measure


REGIONAL_FORM_URL = (
    "https://rptsvr1.tea.texas.gov/adhocrpt/Standard_Reports/Transfer_Reports/"
    "region_district_transfer_report.html"
)
_ACTION = "https://rptsvr1.tea.texas.gov/cgi/sas/broker"
_HIDDEN = (
    ("_service", "marykay"),
    ("_program", "adhoc.download_transfer_report.sas"),
    ("scope", "Region"),
    ("agg_level", "region"),
    ("_debug", "0"),
)
_SUBMIT = ("Region_District_Xfr_Report", "Get a Region District Transfer Report")
_SELECTS = ("school_year", "report_format", "selected_id")
_FORM_LIMIT = 2 * 1024 * 1024


class _RegionalFormParser(HTMLParser):
    """Only the observed regional form, including implicitly closed options."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.forms: list[dict] = []
        self.current: dict | None = None
        self.select: str | None = None
        self.option: tuple[str, list[str]] | None = None

    def _end_option(self) -> None:
        if self.option is not None:
            value, parts = self.option
            assert self.current is not None and self.select is not None
            options = self.current["selects"][self.select]
            if value in options:
                raise ValueError("Duplicate regional form option")
            options[value] = " ".join("".join(parts).split())
            self.option = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "form":
            if self.current is not None:
                raise ValueError("Nested regional form")
            if values.get("name") == "Select_Region_Summary":
                self.current = {"method": values.get("method"), "action": values.get("action"),
                                "inputs": [], "selects": {}, "order": []}
                self.forms.append(self.current)
        elif self.current is not None:
            if "disabled" in values or any(name in values for name in ("formaction", "formmethod", "form", "multiple")):
                raise ValueError("Unsupported regional form control override")
            if tag == "input":
                name, value = values.get("name"), values.get("value")
                if not name or value is None or values.get("type") not in {"hidden", "submit"}:
                    raise ValueError("Unexpected regional form input")
                self.current["inputs"].append((name, value, values["type"]))
                self.current["order"].append(name)
            elif tag == "select":
                name = values.get("name")
                if self.select is not None or name not in _SELECTS or name in self.current["selects"]:
                    raise ValueError("Unexpected regional form select")
                self.select = name
                self.current["selects"][name] = {}
                self.current["order"].append(name)
            elif tag == "option":
                option_value = values.get("value")
                if self.select is None or option_value is None:
                    raise ValueError("Regional option requires an explicit value")
                self._end_option()
                self.option = (option_value, [])
            elif tag in {"button", "textarea"}:
                raise ValueError("Unsupported regional form control")

    def handle_data(self, data: str) -> None:
        if self.option is not None:
            self.option[1].append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.current is not None:
            if tag in {"option", "select", "form"}:
                self._end_option()
            if tag == "select":
                self.select = None
            elif tag == "form":
                if self.select is not None:
                    raise ValueError("Unclosed regional select")
                self.current = None


@dataclass(frozen=True)
class RegionalTransferForm:
    record_id: str
    sha256: str
    years: Mapping[str, str]
    regions: Mapping[str, str]

    def selection(self, year: int, regions: Sequence[str]) -> tuple[DataPeriod, tuple[str, ...]]:
        if type(year) is not int or str(year) not in self.years:
            raise ValueError("Requested year is not offered by the pinned regional form")
        label = self.years[str(year)]
        match = re.fullmatch(r"([0-9]{4})-([0-9]{4})", label)
        if match is None:
            raise ValueError("Selected school-year label has an unsupported period")
        period = DataPeriod("school_year", int(match[1]), int(match[2]))
        if period.end_year != year:
            raise ValueError("Selected year control and published school-year label disagree")
        if isinstance(regions, (str, bytes)):
            raise ValueError("Select an explicit finite sequence of offered region IDs")
        selected = tuple(regions)
        if (not selected or any(not isinstance(v, str) or v not in self.regions for v in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError("Region IDs must be distinct exact values offered by the pinned form")
        return period, selected

    def controls(self, year: int, region: str) -> list[tuple[str, str]]:
        self.selection(year, [region])
        return [*_HIDDEN, ("school_year", str(year)), ("report_format", "csv"),
                ("selected_id", region), _SUBMIT]


def read_regional_form(store: ProvenanceStore, record_id: str, expected_sha256: str) -> RegionalTransferForm:
    """Read one bounded, SHA-pinned original regional selector acquisition."""
    _identifier(expected_sha256)
    record = store.read_record(record_id)
    if (record["kind"] != "acquisition" or record.get("method") != "download"
            or record.get("requested_url") != REGIONAL_FORM_URL
            or record.get("resolved_url") != REGIONAL_FORM_URL
            or record["artifact"]["sha256"] != expected_sha256
            or not 0 < record["artifact"]["size_bytes"] <= _FORM_LIMIT):
        raise ValueError("Pinned regional form does not match its original HTTP receipt")
    store.verify_record_closure([record_id])
    with store.verify_artifact(expected_sha256).open("rb") as stream:
        raw = stream.read(_FORM_LIMIT + 1)
    if len(raw) != record["artifact"]["size_bytes"] or sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Regional form bytes changed or exceed their limit")
    message = Message()
    message["content-type"] = record.get("response_headers", {}).get("content-type", "")
    encoding = message.get_content_charset()
    if message.get_content_type() != "text/html" or not encoding:
        raise ValueError("Regional HTML form requires its recorded explicit charset")
    parser = _RegionalFormParser()
    parser.feed(raw.decode(encoding, errors="strict"))
    parser.close()
    if len(parser.forms) != 1 or parser.current is not None:
        raise ValueError("Expected one complete named regional form")
    form = parser.forms[0]
    expected_inputs = [*((*pair, "hidden") for pair in _HIDDEN), (*_SUBMIT, "submit")]
    if (str(form["method"]).upper() != "GET" or urljoin(REGIONAL_FORM_URL, form["action"] or "") != _ACTION
            or form["inputs"] != expected_inputs
            or form["order"] != [*(pair[0] for pair in _HIDDEN), *_SELECTS, _SUBMIT[0]]
            or tuple(form["selects"]) != _SELECTS
            or "csv" not in form["selects"]["report_format"]
            or not form["selects"]["school_year"] or not form["selects"]["selected_id"]):
        raise ValueError("Pinned regional form controls differ from the observed contract")
    return RegionalTransferForm(record_id, expected_sha256,
                                form["selects"]["school_year"], form["selects"]["selected_id"])


def _limits(timeout: float, max_bytes: int) -> dict:
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or not 0 < timeout <= 20
            or type(max_bytes) is not int or not 0 < max_bytes <= 8 * 1024 * 1024):
        raise ValueError("Regional acquisition requires timeout <=20s and a byte cap <=8 MiB")
    return {"timeout": timeout, "attempts": 1, "max_bytes": max_bytes, "max_retry_wait": 0}


def _regional_receipt(store: ProvenanceStore, form: RegionalTransferForm, record_id: str,
                      year: int, region: str, period: DataPeriod, limits: dict) -> dict:
    record = store.read_record(record_id)
    controls = form.controls(year, region)
    expected_url, _, _ = _form_request(_ACTION, "GET", controls, None)
    params = record.get("parameters", {})
    metadata = record.get("source_metadata", {})
    headers = record.get("response_headers", {})
    message = Message()
    message["content-type"] = headers.get("content-type", "")
    if (record["kind"] != "acquisition" or record.get("method") != "download"
            or record.get("dataset") != "district_transfer_reports" or record.get("section") != "data_sources"
            or record.get("requested_year") != year or record.get("resolved_year") != year
            or record.get("requested_url") != expected_url or record.get("resolved_url") != expected_url
            or params.get("http_method") != "GET" or params.get("http_status") != 200
            or params.get("form_url") != _ACTION or params.get("form_content_type") is not None
            or params.get("form_controls") != [list(pair) for pair in controls]
            or params.get("transport_limits") != limits
            or metadata.get("version") != str(year) or metadata.get("period") != period.to_dict()
            or metadata.get("source_url") != REGIONAL_FORM_URL
            or form.record_id not in metadata.get("provenance_ids", [])
            or message.get_content_type() != "text/csv"
            or not 0 < record["artifact"]["size_bytes"] <= limits["max_bytes"]):
        raise ValueError("Regional receipt does not match the exact selected form/year/region/transport")
    return record


def _write_index(directory: Path, report: dict, *, filename: str = "BATCH_INDEX.json") -> None:
    descriptor, name = tempfile.mkstemp(dir=directory, prefix="." + filename + ".", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / filename)
    finally:
        temporary.unlink(missing_ok=True)


class RegionalAcquisitionItem(TypedDict):
    region: str
    label: str
    status: str
    record_id: NotRequired[str]
    artifact: NotRequired[dict[str, object]]
    error: NotRequired[dict[str, str]]


def acquire_regional_transfers(
    store: ProvenanceStore, form: RegionalTransferForm, *, year: int,
    regions: Sequence[str], output: Path, resume: Mapping[str, str] | None = None,
    timeout: float = 20, max_bytes: int = 8 * 1024 * 1024,
) -> dict:
    """Acquire a finite sequential batch; retain a truthful HOLD prefix on failure.

    Resume is opt-in by exact region/receipt pairs. This index projects original
    receipts and is not a prepared dataset or a value-correctness qualification.
    """
    # Recapture the form before any output or network action, rather than trusting
    # a caller-constructed descriptor or a changed mutable selection mapping.
    form = read_regional_form(store, form.record_id, form.sha256)
    period, selected = form.selection(year, regions)
    limits = _limits(timeout, max_bytes)
    resume = dict(resume or {})
    if set(resume) - set(selected):
        raise ValueError("Resume receipts must belong to explicitly selected regions")
    resumed = {}
    for region, identifier in resume.items():
        resumed[region] = _regional_receipt(store, form, identifier, year, region, period, limits)
        store.verify_record_closure([identifier])
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    items: list[RegionalAcquisitionItem] = [
        {"region": region, "label": form.regions[region], "status": "pending"}
        for region in selected
    ]
    report: dict = {
        "schema_version": 1, "status": "HOLD", "activation": False,
        "field_correctness_qualified": False,
        "recipe_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "form": {"record_id": form.record_id, "sha256": form.sha256, "url": REGIONAL_FORM_URL},
        "year": year, "period": period.to_dict(), "regions": list(selected),
        "transport_limits": limits,
        "items": items,
    }
    _write_index(output, report)
    client = AcquisitionClient(store, opener=build_opener(NoRedirect()), **limits)
    for item in items:
        region = item["region"]
        try:
            if region in resumed:
                identifier, record = resume[region], resumed[region]
                item.update(record_id=identifier, artifact=record["artifact"], status="resumed")
            else:
                source = ResolvedDataset("district_transfer_reports", "data_sources", year, year, "",
                    DatasetMetadata(
                        title=f"TEA {period.label} district transfer report, {form.regions[region]}",
                        publisher="Texas Education Agency", version=str(year), period=period,
                        source_url=REGIONAL_FORM_URL, provenance_ids=(form.record_id,),
                        notes="Original regional CSV with publisher headings/footnotes. Acquisition only; no activation or value qualification.",
                    ))
                acquired = acquire_form(client, _ACTION, source,
                    filename=f"district-transfers-{year}-region{region}-original.csv",
                    method="GET", fields=form.controls(year, region))
                item.update(record_id=acquired.record_id, status="rejected")
                record = store.read_record(acquired.record_id)
                item["artifact"] = record["artifact"]
                _regional_receipt(store, form, acquired.record_id, year, region, period, limits)
                item["status"] = "acquired"
            _write_index(output, report)
        except BaseException as error:
            # Interruptions are not ordinary source failures: persist the prefix
            # and immediately propagate, without retrying or accepting the batch.
            if item["status"] == "pending":
                item["status"] = "failed" if isinstance(error, Exception) else "interrupted"
            item["error"] = {"type": type(error).__name__, "message": str(error)[:1000]}
            _write_index(output, report)
            raise
    report["status"] = "PASS"
    _write_index(output, report)
    return report


_DISTRICT_COLUMNS = (
    "YEAR", "REPORT_REGION", "REPORT_REGION_NAME", "REPORT_DISTNAME_NUMBER",
    "REPORT_CHARTER_STATUS", "REPORT_NUMBER", "REPORT_TYPE", "LINE_GROUP_NUMBER",
    "DISTNAME_NUMBER_RES_ATTEND", "TRANSFERS_IN_OUT",
)
_FOOTNOTES = (
    "-999 indicates counts or percentages are not available (i.e. masked) to comply with the Family Educational Rights and Privacy Act (FERPA).",
    "Masked numbers are typically small although larger numbers may be masked to prevent imputation.",
    "The REPORT_NUMBER column and the LINE_GROUP_NUMBER column are added to assist sorting and filtering of rows in the file.",
)


def transfer_columns(grain: str) -> tuple[str, ...]:
    if grain == "district":
        return _DISTRICT_COLUMNS
    if grain == "campus":
        # Reuse the existing campus wire-schema authority, rather than another
        # copy of its fourteen names/order in Python.
        resource = Path(__file__).with_name("schemas") / "data_sources--campus_transfer_reports--2025--default.json"
        return tuple(json.loads(resource.read_bytes())["columns"])
    raise ValueError("Transfer grain must be campus or district")


@dataclass(frozen=True)
class TransferSourceTable:
    table: Any
    grain: str
    period: DataPeriod
    input_sha256: str
    encoding: str
    layout: str
    annotations: tuple[dict[str, Any], ...]


def _transfer_projection(table, *, grain: str, period: DataPeriod, policy: ValuePolicy, region: str | None):
    """Interpret whole Arrow columns; scalar values.py runs only per token."""
    import pyarrow as pa
    import pyarrow.compute as arrow_compute

    pc: Any = arrow_compute

    def require(condition, message: str) -> None:
        if not pc.all(pc.fill_null(condition, False)).as_py():
            raise ValueError(message)

    require(pc.equal(table["YEAR"], period.label), "Transfer source YEAR differs from the selected school-year label")
    regions = canonical_source_identifiers(table["REPORT_REGION"], width=2)
    require(pc.invert(pc.is_null(regions)), "Transfer row requires its report-time region")
    if region is not None:
        if not isinstance(region, str) or not re.fullmatch(r"[0-9]{2}", region):
            raise ValueError("Expected region must be an exact two-digit source control")
        require(pc.equal(regions, "'" + region), "Transfer row belongs to a different selected region")
    report_type, group = table["REPORT_TYPE"], table["LINE_GROUP_NUMBER"]
    incoming = pc.equal(report_type, "Transfers In From")
    outgoing = pc.equal(report_type, "Transfers Out To")
    has_direction = pc.or_(incoming, outgoing)
    require(pc.or_(pc.or_(incoming, outgoing), pc.equal(report_type, "")), "Unexpected transfer report direction")
    require(pc.or_(pc.and_(incoming, pc.equal(table["REPORT_NUMBER"], "100")),
                  pc.or_(pc.and_(outgoing, pc.equal(table["REPORT_NUMBER"], "200")),
                         pc.and_(pc.equal(report_type, ""), pc.is_in(table["REPORT_NUMBER"], value_set=pa.array(["100", "200"]))))),
            "Transfer report number and direction disagree")
    detail = pc.and_(pc.equal(group, "100"), has_direction)
    target_label = table["CAMPNAME_RES_OR_ATTEND" if grain == "campus" else "DISTNAME_NUMBER_RES_ATTEND"]
    expected_total = pc.if_else(incoming, "Total Transfers In", "Total Transfers Out")
    total = pc.and_(pc.and_(pc.equal(group, "200"), has_direction), pc.equal(target_label, expected_total))
    separator = pc.and_(pc.is_in(group, value_set=pa.array(["200", "300"])), pc.equal(target_label, ""))
    require(pc.or_(detail, pc.or_(total, separator)), "Unexpected transfer row kind/group/total label")

    def identifiers(values, width: int, *, required):
        result = canonical_source_identifiers(values, width=width)
        # Only an explicit optional row bypasses the check; unknown requirements
        # and required-but-missing values still reach require() as false/null.
        require(pc.if_else(required, pc.invert(pc.is_null(result)), True), "Transfer detail has a missing historical source identifier")
        return result

    if grain == "campus":
        reporting = identifiers(table["REPORT_CAMPUS"], 9, required=pa.array([True] * table.num_rows))
        partner = identifiers(table["CAMPUS_RES_OR_ATTEND"], 9, required=detail)
        reporting_district = identifiers(table["REPORT_DISTRICT"], 6, required=pa.array([True] * table.num_rows))
        partner_district = identifiers(table["DISTRICT_RES_OR_ATTEND"], 6, required=detail)
        for campus, district, required in [(reporting, reporting_district, pc.invert(pc.is_null(reporting))),
                                           (partner, partner_district, detail)]:
            require(pc.if_else(required, pc.equal(pc.utf8_slice_codeunits(campus, start=1, stop=7),
                                                pc.utf8_slice_codeunits(district, start=1)), True),
                    "Campus and report-time district identifiers disagree")
        for name in ("CAMPUS_RES_OR_ATTEND", "DISTRICT_RES_OR_ATTEND", "DISTNAME_RES_OR_ATTEND"):
            require(pc.or_(pc.invert(pc.or_(separator, total)), pc.equal(table[name], "")),
                    "Transfer separator/total unexpectedly carries a partner identity")
    else:
        def label_ids(values, required):
            matches = pc.extract_regex(values, pattern=r"^.*\S\s+-\s+(?P<number>[0-9]{6})$")
            digits = pc.struct_field(matches, "number")
            digits = pc.if_else(required, digits, pa.scalar(None, type=pa.string()))
            return identifiers(digits, 6, required=required)

        reporting = label_ids(table["REPORT_DISTNAME_NUMBER"], pa.array([True] * table.num_rows))
        partner = label_ids(table["DISTNAME_NUMBER_RES_ATTEND"], detail)
        reporting_district, partner_district = reporting, partner

    count_column = "TRANSFERS_IN_OR_OUT" if grain == "campus" else "TRANSFERS_IN_OUT"
    unique = pc.unique(table[count_column])
    if len(unique) > 4096:
        raise ValueError("Transfer source exceeds bounded distinct-count-token interpretation")
    interpreted = [measure(token, policy, unit="count", minimum=0) for token in unique.to_pylist()]
    if any(value.status == ValueStatus.INVALID for value in interpreted):
        raise ValueError("Transfer source has an invalid count token; raw values are retained unchanged")
    indices = pc.index_in(table[count_column], value_set=unique)
    statuses = pc.take(pa.array([value.status.value for value in interpreted]), indices)
    values = pc.take(pa.array([value.numeric for value in interpreted], type=pa.int64()), indices)
    require(pc.or_(pc.invert(separator), pc.equal(statuses, ValueStatus.MISSING.value)),
            "Transfer separator carries a nonmissing count")
    columns = {
        "transfer_reporting_region": regions,
        "transfer_row_kind": pc.if_else(detail, "detail", pc.if_else(total, "publisher_total", "separator")),
        "transfer_direction": pc.if_else(incoming, "in", pc.if_else(outgoing, "out", pa.scalar(None, type=pa.string()))),
        "transfer_reporting_number": reporting, "transfer_partner_number": partner,
        "transfer_reporting_district_number": reporting_district,
        "transfer_partner_district_number": partner_district,
        "transfer_count_status": statuses, "transfer_count_value": values,
    }
    for name, values in columns.items():
        table = table.append_column(name, values)
    return table


def read_transfer_table(path: Path, *, grain: str, year: int, policy: ValuePolicy,
                        expected_sha256: str, encoding: str, region: str | None = None,
                        layout: str = "original", max_bytes: int = 64 * 1024 * 1024) -> TransferSourceTable:
    """Read exact original columns, notes and row locations before interpretation.

    ``legacy_prepared`` explicitly admits the old header-first local CSV; it
    does not assert that its removed headings/notes or HTTP history are known.
    Input/batch/record limits are admissions, not a process RSS guarantee.
    """
    import pyarrow as pa
    import pyarrow.csv as pacsv

    _identifier(expected_sha256)
    if type(year) is not int or not 2 <= year <= 9999:
        raise ValueError("Transfer selection requires an explicit school-year end year")
    if layout not in {"original", "legacy_prepared"}:
        raise ValueError("Select original or explicitly legacy_prepared transfer layout")
    if type(max_bytes) is not int or not 0 < max_bytes <= 64 * 1024 * 1024:
        raise ValueError("Transfer source requires an input byte admission <=64 MiB")
    before = path.stat()
    if not 0 < before.st_size <= max_bytes:
        raise ValueError("Transfer source exceeds its admitted input byte limit")
    period = DataPeriod("school_year", year - 1, year)
    columns = transfer_columns(grain)
    if region is not None and grain != "district":
        raise ValueError("Regional report selection is supported only for district-grain originals")
    report_scope = "Region" if region is not None else grain.title()
    titles = ["T E X A S E D U C A T I O N A G E N C Y", f"{report_scope} Level Transfer Report", "PEIMS Student Transfer Report"]
    annotations: list[dict[str, Any]] = []
    nonblank_titles, nonblank_notes = [], []
    batches, positions = [], []
    body = bytearray()
    header = footer = False
    record = metadata_bytes = 0

    def decode():
        table = pacsv.read_csv(pa.BufferReader(b"\n" + body),
            read_options=pacsv.ReadOptions(column_names=list(columns), use_threads=False, skip_rows=1, block_size=len(body) + 1),
            parse_options=pacsv.ParseOptions(newlines_in_values=True, ignore_empty_lines=False),
            convert_options=pacsv.ConvertOptions(column_types=dict.fromkeys(columns, pa.string()),
                null_values=[], strings_can_be_null=False, quoted_strings_can_be_null=False))
        if table.num_rows != len(positions) or table.nbytes > 8 * 1024 * 1024:
            raise ValueError("Transfer native batch exceeds admitted rows/bytes")
        for i, name in enumerate(("transfer_source_record", "transfer_first_line", "transfer_last_line")):
            table = table.append_column(name, pa.array([position[i] for position in positions], type=pa.int64()))
        batches.append(table)
        body.clear()
        positions.clear()

    with path.open("rb") as stream:
        lines = SourceLines(stream, 1024 * 1024, "Transfer logical CSV record exceeds 1 MiB", encoding=encoding)
        reader = csv.reader(lines, strict=True)
        try:
            while True:
                first_line = reader.line_num + 1
                lines.record.clear()
                try:
                    row = next(reader)
                except StopIteration:
                    break
                record += 1
                segment = bytes(lines.record)
                if not header and tuple(row) == columns:
                    if (layout == "original" and nonblank_titles != titles) or (layout == "legacy_prepared" and record != 1):
                        raise ValueError("Transfer header is not preceded by the selected layout's exact publisher titles")
                    header, kind = True, "header"
                elif not header:
                    if layout != "original" or len(row) > 1:
                        raise ValueError("Unexpected transfer preamble/header schema")
                    text = " ".join((row[0] if row else "").split())
                    if text:
                        nonblank_titles.append(text)
                        if nonblank_titles != titles[:len(nonblank_titles)]:
                            raise ValueError(f"Unexpected transfer publisher title at CSV record {record}, physical line {first_line}")
                    kind = "preamble" if text else "blank"
                elif len(row) == len(columns) and not footer:
                    data = segment.decode(lines.encoding, errors="strict").encode("utf-8")
                    if len(data) + 1 + 8 * len(columns) > 8 * 1024 * 1024:
                        raise ValueError("Transfer record cannot fit native batch admission")
                    if positions and (len(positions) == 4096 or len(body) + len(data) + 1 + 8 * len(columns) * (len(positions) + 1) > 8 * 1024 * 1024):
                        decode()
                    body.extend(data)
                    positions.append((record, first_line, reader.line_num))
                    continue
                else:
                    footer = True
                    if len(row) > 1:
                        raise ValueError("Unexpected full-width/data row after transfer footer")
                    text = " ".join((row[0] if row else "").split())
                    if text:
                        nonblank_notes.append(text)
                        if nonblank_notes != list(_FOOTNOTES[:len(nonblank_notes)]):
                            raise ValueError("Unexpected transfer publisher footnote")
                    kind = "footer" if text else "blank"
                metadata_bytes += len(segment) + 64 * len(row)
                if metadata_bytes > 2 * 1024 * 1024:
                    raise ValueError("Transfer annotations exceed 2 MiB metadata admission")
                annotations.append({"kind": kind, "record": record, "first_line": first_line,
                    "last_line": reader.line_num, "raw_sha256": sha256(segment).hexdigest(),
                    "text": segment.decode(lines.encoding, errors="strict")})
        except csv.Error as error:
            raise ValueError(f"Invalid transfer CSV near physical line {reader.line_num}: {error}") from error
        if lines.digest.hexdigest() != expected_sha256:
            raise ValueError("Transfer input differs from its pinned original SHA")
        actual_encoding = lines.encoding
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("Transfer source changed during structural preparation")
    if not header or (layout == "original" and nonblank_notes != list(_FOOTNOTES)):
        raise ValueError("Transfer source lacks its complete exact header/footnotes")
    if positions:
        decode()
    if not batches:
        raise ValueError("Transfer source contains no report rows")
    table = pa.concat_tables(batches)
    table = _transfer_projection(table, grain=grain, period=period, policy=policy, region=region)
    return TransferSourceTable(table, grain, period, expected_sha256, actual_encoding, layout, tuple(annotations))


class TransferValidationError(ValueError):
    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


def _campus_pairs(source: TransferSourceTable):
    import pyarrow as pa
    import pyarrow.compute as arrow_compute

    pc: Any = arrow_compute
    if source.grain != "campus":
        raise ValueError("Campus pair interpretation does not establish district report totals")
    table = source.table.filter(pc.equal(source.table["transfer_row_kind"], "detail"))
    outgoing = pc.equal(table["transfer_direction"], "out")
    columns = {
        "YEAR": table["YEAR"], "direction": table["transfer_direction"],
        "source_number": pc.if_else(outgoing, table["transfer_reporting_number"], table["transfer_partner_number"]),
        "destination_number": pc.if_else(outgoing, table["transfer_partner_number"], table["transfer_reporting_number"]),
        "source_district": pc.if_else(outgoing, table["transfer_reporting_district_number"], table["transfer_partner_district_number"]),
        "destination_district": pc.if_else(outgoing, table["transfer_partner_district_number"], table["transfer_reporting_district_number"]),
        "count_status": table["transfer_count_status"], "count_value": table["transfer_count_value"],
        "raw_count": table["TRANSFERS_IN_OR_OUT"],
        "source_record": table["transfer_source_record"],
        "first_line": table["transfer_first_line"], "last_line": table["transfer_last_line"],
        "normalized_REPORT_REGION": table["transfer_reporting_region"],
        "normalized_REPORT_CAMPUS": table["transfer_reporting_number"],
        "normalized_CAMPUS_RES_OR_ATTEND": table["transfer_partner_number"],
        "normalized_REPORT_DISTRICT": table["transfer_reporting_district_number"],
        "normalized_DISTRICT_RES_OR_ATTEND": table["transfer_partner_district_number"],
    }
    # Original columns stay available to the old/new comparison. Native joins
    # select direction first and never collapse opposite-direction observations.
    columns.update({name: table[name] for name in transfer_columns("campus") if name != "YEAR"})
    return pa.table(columns)


def _number_true(values) -> int:
    import pyarrow.compute as arrow_compute
    pc: Any = arrow_compute
    return pc.sum(pc.fill_null(values, False)).as_py() or 0


def _pair_duplicates(table, keys: list[str]):
    import pyarrow.compute as arrow_compute
    pc: Any = arrow_compute
    counts = table.group_by(keys).aggregate([("source_record", "count")])
    return counts.filter(pc.greater(counts["source_record_count"], 1))


def validate_campus_directions(source: TransferSourceTable, *, max_examples: int = 10) -> dict:
    """Gate one canonical edge on independently reported In/inverse-Out equality.

    Unknown/invalid counts cannot qualify by matching each other. Examples are
    bounded; exact counts and original raw values/row locators remain retained.
    """
    import pyarrow as pa
    import pyarrow.compute as arrow_compute

    pc: Any = arrow_compute
    if type(max_examples) is not int or not 0 <= max_examples <= 100:
        raise ValueError("Transfer examples require a finite bound <=100")
    pairs = _campus_pairs(source)
    keys = ["YEAR", "source_number", "destination_number"]
    outgoing = pairs.filter(pc.equal(pairs["direction"], "out"))
    incoming = pairs.filter(pc.equal(pairs["direction"], "in"))
    report: dict = {"status": "HOLD", "input_sha256": source.input_sha256,
        "out_rows": outgoing.num_rows, "in_rows": incoming.num_rows,
        "examples_limit": max_examples, "scientific_equivalence_claimed": False}
    if not outgoing.num_rows or not incoming.num_rows:
        raise TransferValidationError("Campus direction gate requires independently reported pairs in both directions", report)
    allowed = pa.array([ValueStatus.OBSERVED.value, ValueStatus.SUPPRESSED.value])
    unknown = pairs.filter(pc.invert(pc.is_in(pairs["count_status"], value_set=allowed)))
    duplicates_out, duplicates_in = _pair_duplicates(outgoing, keys), _pair_duplicates(incoming, keys)
    report.update(unknown_count_rows=unknown.num_rows,
                  duplicate_out_keys=duplicates_out.num_rows, duplicate_in_keys=duplicates_in.num_rows)
    if unknown.num_rows or duplicates_out.num_rows or duplicates_in.num_rows:
        report["examples"] = {"unknown": unknown.slice(0, max_examples).to_pylist(),
            "duplicate_out": duplicates_out.slice(0, max_examples).to_pylist(),
            "duplicate_in": duplicates_in.slice(0, max_examples).to_pylist()}
        raise TransferValidationError("Campus directions have unknown counts or duplicate pairs", report)
    selected = [*keys, "source_district", "destination_district", "count_status", "count_value", "raw_count",
                "source_record", "first_line", "last_line"]
    joined = outgoing.select(selected).join(incoming.select(selected), keys=keys, join_type="full outer",
        left_suffix="_out", right_suffix="_in", coalesce_keys=True, use_threads=False)
    both = pc.and_(pc.is_valid(joined["source_record_out"]), pc.is_valid(joined["source_record_in"]))
    same_status = pc.equal(joined["count_status_out"], joined["count_status_in"])
    same_value = pc.or_(pc.equal(joined["count_status_out"], "suppressed"),
                        pc.fill_null(pc.equal(joined["count_value_out"], joined["count_value_in"]), False))
    same_district = pc.and_(pc.equal(joined["source_district_out"], joined["source_district_in"]),
                            pc.equal(joined["destination_district_out"], joined["destination_district_in"]))
    mismatch = pc.and_(both, pc.invert(pc.and_(same_status, pc.and_(same_value, same_district))))
    report.update(one_sided_pairs=_number_true(pc.invert(both)),
        status_mismatches=_number_true(pc.and_(both, pc.invert(same_status))),
        value_mismatches=_number_true(pc.and_(both, pc.invert(same_value))),
        historical_district_mismatches=_number_true(pc.and_(both, pc.invert(same_district))),
        raw_count_token_differences=_number_true(pc.and_(both, pc.not_equal(joined["raw_count_out"], joined["raw_count_in"]))),
        matching_observed=_number_true(pc.and_(both, pc.and_(pc.invert(mismatch), pc.equal(joined["count_status_out"], "observed")))),
        matching_suppressed=_number_true(pc.and_(both, pc.and_(pc.invert(mismatch), pc.equal(joined["count_status_out"], "suppressed")))))
    if report["one_sided_pairs"] or _number_true(mismatch):
        report["examples"] = joined.filter(pc.or_(pc.invert(both), pc.fill_null(mismatch, False))).slice(0, max_examples).to_pylist()
        raise TransferValidationError("Independent campus directions do not qualify one canonical edge", report)
    report["status"] = "PASS"
    report["comparison_scope"] = "Parsed pair/count/status/report-time district equality; not all-field byte identity"
    return report


def compare_campus_sources(old: TransferSourceTable, new: TransferSourceTable, *, max_examples: int = 10) -> dict:
    """Compare source observations without conflating padding with changed data."""
    import pyarrow.compute as arrow_compute

    pc: Any = arrow_compute
    if old.period != new.period or type(max_examples) is not int or not 0 <= max_examples <= 100:
        raise ValueError("Campus comparison requires one exact period and bounded examples")
    left, right = _campus_pairs(old), _campus_pairs(new)
    keys = ["YEAR", "direction", "source_number", "destination_number"]
    if _pair_duplicates(left, keys).num_rows or _pair_duplicates(right, keys).num_rows:
        raise ValueError("Campus comparison requires unique reported direction/pair keys")
    joined = left.join(right, keys=keys, join_type="full outer", left_suffix="_old", right_suffix="_new",
                       coalesce_keys=True, use_threads=False)
    both = pc.and_(pc.is_valid(joined["source_record_old"]), pc.is_valid(joined["source_record_new"]))
    changed_status = pc.and_(both, pc.not_equal(joined["count_status_old"], joined["count_status_new"]))
    same_value = pc.or_(pc.and_(pc.is_null(joined["count_value_old"]), pc.is_null(joined["count_value_new"])),
                        pc.fill_null(pc.equal(joined["count_value_old"], joined["count_value_new"]), False))
    changed_value = pc.and_(both, pc.invert(same_value))
    changed_district = pc.and_(both, pc.or_(pc.not_equal(joined["source_district_old"], joined["source_district_new"]),
                                           pc.not_equal(joined["destination_district_old"], joined["destination_district_new"])))
    identity_columns = {"REPORT_REGION", "REPORT_CAMPUS", "CAMPUS_RES_OR_ATTEND", "REPORT_DISTRICT", "DISTRICT_RES_OR_ATTEND"}
    lexical, identity_changes, other = {}, {}, {}
    changed_any = pc.or_(changed_status, pc.or_(changed_value, changed_district))
    for name in transfer_columns("campus"):
        if name == "YEAR":
            continue
        changed = pc.and_(both, pc.not_equal(joined[name + "_old"], joined[name + "_new"]))
        if name in identity_columns:
            normalized_changed = pc.and_(both, pc.not_equal(joined["normalized_" + name + "_old"], joined["normalized_" + name + "_new"]))
            lexical[name] = _number_true(pc.and_(changed, pc.invert(normalized_changed)))
            identity_changes[name] = _number_true(normalized_changed)
            changed_any = pc.or_(changed_any, normalized_changed)
        elif name != "TRANSFERS_IN_OR_OUT":
            other[name] = _number_true(changed)
            changed_any = pc.or_(changed_any, changed)
    return {"status": "compared", "old_sha256": old.input_sha256, "new_sha256": new.input_sha256,
        "period": old.period.to_dict(), "old_detail_rows": left.num_rows, "new_detail_rows": right.num_rows,
        "only_old": _number_true(pc.is_null(joined["source_record_new"])),
        "only_new": _number_true(pc.is_null(joined["source_record_old"])),
        "count_status_changes": _number_true(changed_status), "count_value_changes": _number_true(changed_value),
        "historical_district_changes": _number_true(changed_district),
        "raw_count_token_differences": _number_true(pc.and_(both, pc.not_equal(joined["raw_count_old"], joined["raw_count_new"]))),
        "identifier_lexeme_changes": lexical, "identifier_value_changes": identity_changes,
        "other_raw_field_changes": other,
        "changed_data_examples": joined.filter(changed_any).slice(0, max_examples).to_pylist(),
        "one_sided_examples": joined.filter(pc.invert(both)).slice(0, max_examples).to_pylist(),
        "old_table_rows": old.table.num_rows, "new_table_rows": new.table.num_rows,
        "examples_limit": max_examples, "activation": False, "scientific_equivalence_claimed": False,
        "scope_note": "Reported directional pair details compared; separators/totals and annotations remain in each prepared source. CSV quoting/layout byte differences are not field-value changes."}


def validate_district_rows(table, *, max_examples: int = 10) -> dict:
    """Check district source grain without imposing campus inverse equality."""
    import pyarrow.compute as arrow_compute
    pc: Any = arrow_compute

    details = table.filter(pc.equal(table["transfer_row_kind"], "detail"))
    totals = table.filter(pc.equal(table["transfer_row_kind"], "publisher_total"))
    detail_keys = ["YEAR", "transfer_reporting_number", "transfer_direction", "transfer_partner_number"]
    total_keys = ["YEAR", "transfer_reporting_number", "transfer_direction"]
    duplicate_details = details.group_by(detail_keys).aggregate([("transfer_source_record", "count")])
    duplicate_details = duplicate_details.filter(pc.greater(duplicate_details["transfer_source_record_count"], 1))
    duplicate_totals = totals.group_by(total_keys).aggregate([("transfer_source_record", "count")])
    duplicate_totals = duplicate_totals.filter(pc.greater(duplicate_totals["transfer_source_record_count"], 1))
    district_regions = table.group_by(["transfer_reporting_number"]).aggregate([("transfer_reporting_region", "count_distinct")])
    ambiguous_regions = district_regions.filter(pc.greater(district_regions["transfer_reporting_region_count_distinct"], 1))
    report: dict = {"status": "HOLD", "detail_rows": details.num_rows, "publisher_total_rows": totals.num_rows,
        "reporting_districts": district_regions.num_rows, "duplicate_detail_keys": duplicate_details.num_rows,
        "duplicate_total_keys": duplicate_totals.num_rows, "districts_with_multiple_report_regions": ambiguous_regions.num_rows,
        "totals_computed_from_details": False,
        "scope": "Observed district report rows only; no campus-sum, inverse-direction or exhaustive roster-coverage claim"}
    if duplicate_details.num_rows or duplicate_totals.num_rows or ambiguous_regions.num_rows:
        report["examples"] = {"duplicate_details": duplicate_details.slice(0, max_examples).to_pylist(),
            "duplicate_totals": duplicate_totals.slice(0, max_examples).to_pylist(),
            "multiple_regions": ambiguous_regions.slice(0, max_examples).to_pylist()}
        raise TransferValidationError("District source has conflicting report grain/region observations", report)
    report["status"] = "PASS"
    return report


def _captured_input(path: Path, expected_sha256: str) -> bytes:
    _identifier(expected_sha256)
    with path.open("rb") as stream:
        raw = stream.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024 or sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Transfer control input differs from its SHA or exceeds 2 MiB")
    return raw


def _selected_encoding(record: dict, explicit: str | None) -> tuple[str, dict]:
    raw_header = record.get("response_headers", {}).get("content-type")
    message = Message()
    message["content-type"] = raw_header or ""
    declared = message.get_content_charset()
    candidate = declared or explicit
    if candidate is None:
        raise ValueError("CSV has no declared MIME charset; select an explicit encoding")
    chosen = codecs.lookup(candidate).name
    if explicit is not None and codecs.lookup(explicit).name != chosen:
        raise ValueError("Explicit CSV encoding conflicts with its retained MIME charset")
    return chosen, {"raw_content_type": raw_header, "declared_charset": declared,
        "selected_encoding": chosen, "selection": "MIME" if declared else "explicit_argument"}


def prepare_transfer_sources(
    store: ProvenanceStore, *, grain: str, year: int, output: Path,
    policy_config: Path, policy_config_sha256: str, value_policy: str,
    inputs: Sequence[tuple[str, str]] = (), batch_index: tuple[Path, str] | None = None,
    encoding: str | None = None, compare_input: tuple[str, str] | None = None,
    compare_encoding: str | None = None, max_examples: int = 10,
) -> dict:
    """Retain raw→prepared derivation using the existing processing DAG only.

    No catalog or source metadata is activated. In particular, processing IDs
    never enter acquisition-only DatasetMetadata.provenance_ids.
    """
    from importlib.metadata import version
    import platform
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .manifest import processor_identity
    from .teadata_config import load_config_bytes

    transfer_columns(grain)
    if type(year) is not int or not 2 <= year <= 9999:
        raise ValueError("Transfer preparation requires an explicit annual edition")
    if type(max_examples) is not int or not 0 <= max_examples <= 100:
        raise ValueError("Transfer examples require a finite bound <=100")
    if bool(inputs) == (batch_index is not None):
        raise ValueError("Select explicit source receipt/SHA pairs or one pinned batch index")
    if compare_input is not None and grain != "campus":
        raise ValueError("Old-prepared comparison is explicitly campus-only")
    if compare_input is None and compare_encoding is not None:
        raise ValueError("Comparison encoding requires an explicit legacy comparison input")
    config_raw = _captured_input(policy_config, policy_config_sha256)
    wire_schema = (Path(__file__).parent / "schemas/data_sources--campus_transfer_reports--2025--default.json").read_bytes() if grain == "campus" else None
    cfg = load_config_bytes(config_raw, path=policy_config)
    if value_policy not in cfg.value_policies:
        raise ValueError("Selected value policy is not declared by the pinned Config")
    policy = cfg.value_policies[value_policy]
    if (set(policy.tokens) != {"-999", "."}
            or policy.tokens["-999"].status != ValueStatus.SUPPRESSED
            or policy.tokens["."].status != ValueStatus.MISSING
            or policy.tokens["-999"].lower is not None or policy.tokens["-999"].upper is not None):
        raise ValueError("Selected transfer policy does not match the retained publisher masking footnotes")
    selected = list(inputs)
    regions: dict[str, str] = {}
    index_raw = None
    if batch_index is not None:
        if grain != "district":
            raise ValueError("Regional district indices cannot select campus reports")
        index_raw = _captured_input(*batch_index)
        index = json.loads(index_raw)
        if index.get("schema_version") != 1 or index.get("status") != "PASS" or index.get("year") != year:
            raise ValueError("District preparation requires a complete exact-year acquisition index")
        form = read_regional_form(store, index["form"]["record_id"], index["form"]["sha256"])
        period, offered = form.selection(year, index["regions"])
        if index.get("period") != period.to_dict() or len(index.get("items", [])) != len(offered):
            raise ValueError("District index period/region coverage differs from its pinned form")
        for region, item in zip(offered, index["items"], strict=True):
            if item.get("region") != region or item.get("status") not in {"acquired", "resumed"}:
                raise ValueError("District index has an unaccepted or reordered region")
            record = _regional_receipt(store, form, item["record_id"], year, region, period,
                                        _limits(**{name: index["transport_limits"][name] for name in ("timeout", "max_bytes")}))
            if record["artifact"] != item["artifact"]:
                raise ValueError("District index artifact differs from its original acquisition")
            selected.append((item["record_id"], record["artifact"]["sha256"]))
            regions[item["record_id"]] = region
    if not selected or len({identifier for identifier, _ in selected}) != len(selected):
        raise ValueError("Transfer preparation requires distinct explicit source receipts")
    expected_dataset = grain + "_transfer_reports"
    roots = [identifier for identifier, _ in selected]
    if compare_input is not None:
        roots.append(compare_input[0])
    records = store.verify_record_closure(roots)
    encodings = {}
    for identifier, expected in [*selected, *([compare_input] if compare_input is not None else [])]:
        record = records[identifier]
        legacy = compare_input is not None and identifier == compare_input[0]
        requested_matches = record.get("requested_year") == year or (
            legacy and record.get("requested_year") is None and record.get("method") == "local_observation")
        if (record["kind"] != "acquisition" or record["dataset"] != expected_dataset
                or record["artifact"]["sha256"] != expected
                or not requested_matches or record.get("resolved_year") != year):
            raise ValueError("Transfer input differs from its exact dataset/year/acquisition/SHA")
        declared_period = DataPeriod.from_dict(record.get("source_metadata", {}).get("period"))
        if (legacy or declared_period.kind != "unknown") and declared_period != DataPeriod("school_year", year - 1, year):
            raise ValueError("Transfer receipt period conflicts with the selected school year")
        encodings[identifier] = _selected_encoding(record, compare_encoding if compare_input is not None and identifier == compare_input[0] else encoding)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Transfer preparation requires a new exclusive output directory")
    code = processor_identity()
    implementation = {"processor_sha256": code,
        "cli_sha256": sha256((Path(__file__).parent / "scripts/refresh_transfers.py").read_bytes()).hexdigest()}
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    report: dict = {"schema_version": 1, "status": "HOLD", "activation": False,
        "field_correctness_qualified": False, "grain": grain, "year": year,
        "inputs": [], "implementation": implementation,
        "policy_id": policy.policy_id, "policy_sha256": sha256(json.dumps(policy.to_dict(), sort_keys=True).encode()).hexdigest(),
        "toolchain": {"python": platform.python_version(), "pyarrow": version("pyarrow"),
                      "platform": platform.platform(), "architecture": platform.machine()},
        "coordinate_semantics": "One-based original logical CSV record and first/last physical lines; raw original artifact owns every column value"}
    attempts: list[dict[str, Any]] = [
        {"source_input": number, "record_id": identifier, "artifact": records[identifier]["artifact"],
         "encoding": encodings[identifier][1], "region": regions.get(identifier), "status": "pending"}
        for number, (identifier, _) in enumerate(selected)
    ]
    report["inputs"] = attempts
    if wire_schema is not None:
        report["wire_schema_sha256"] = sha256(wire_schema).hexdigest()
    _write_index(output, report, filename="PREPARATION.json")
    local_inputs = []
    controls: list[tuple[str, bytes]] = [("policy_config", config_raw)]
    if wire_schema is not None:
        controls.append(("wire_schema", wire_schema))
    if index_raw is not None:
        controls.append(("batch_index", index_raw))
    for role, raw in controls:
        captured = output / (role + ".input")
        descriptor = os.open(captured, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
        local_inputs.append(store.record_acquisition(captured, ResolvedDataset(
            "reference:transfer_" + role, "reference", None, None, "",
            DatasetMetadata(title="Exact explicitly pinned transfer preparation " + role + " input"))))
    tables = []
    annotations = []

    def retain_failure(error: Exception, *, phase: str, failed_input: dict | None = None) -> dict:
        report["error"] = {"type": type(error).__name__, "message": str(error)[:2000]}
        report["failed_phase"] = phase
        if failed_input is not None:
            failed_input["status"] = "failed"
            failed_input["error"] = report["error"]
            report["failed_input"] = failed_input
        if isinstance(error, TransferValidationError):
            report["source_validation" if phase == "source_validation" else "failed_validation"] = error.report
        _write_index(output, report, filename="VALIDATION.json")
        processing = store.record_processing(dataset=expected_dataset, inputs=[*roots, *local_inputs],
            outputs={"validation_report": output / "VALIDATION.json"}, processor="teadata.scripts.refresh_transfers",
            code_version=code, parameters={"no_config_activation": True, **implementation},
            validation={"status": "HOLD", "prepared_source_qualified": False, "failed_phase": phase})
        store.verify_record_closure([processing])
        report["processing_record_id"] = processing
        _write_index(output, report, filename="PREPARATION.json")
        return report

    for number, (identifier, expected) in enumerate(selected):
        selected_encoding, _ = encodings[identifier]
        attempted = attempts[number]
        attempted["status"] = "reading"
        _write_index(output, report, filename="PREPARATION.json")
        try:
            source = read_transfer_table(store.verify_artifact(expected), grain=grain, year=year, policy=policy,
                expected_sha256=expected, encoding=selected_encoding, region=regions.get(identifier))
        except Exception as error:
            return retain_failure(error, phase="structural_read", failed_input=attempted)
        table = source.table.append_column("transfer_source_input", pa.array([number] * source.table.num_rows, type=pa.int32()))
        tables.append(table)
        attempted.update(status="parsed", rows=table.num_rows)
        _write_index(output, report, filename="PREPARATION.json")
        annotations.append({"source_input": number, "record_id": identifier, "input_sha256": expected,
                            "layout": source.layout, "annotations": source.annotations})
    merged = pa.concat_tables(tables)
    combined = None
    gate = None
    try:
        if grain == "campus":
            if len(selected) != 1:
                raise ValueError("Campus direction gate requires one original statewide source, not implicit partitions")
            combined = TransferSourceTable(merged, grain, DataPeriod("school_year", year - 1, year),
                                           selected[0][1], report["inputs"][0]["encoding"]["selected_encoding"], "original", ())
            gate = validate_campus_directions(combined, max_examples=max_examples)
        report["source_validation"] = gate or validate_district_rows(merged, max_examples=max_examples)
    except TransferValidationError as error:
        return retain_failure(error, phase="source_validation")
    if compare_input is not None:
        assert combined is not None
        old_id, old_sha = compare_input
        old_encoding, old_evidence = encodings[old_id]
        legacy_attempt: dict[str, Any] = {"record_id": old_id, "artifact": records[old_id]["artifact"],
            "layout": "legacy_prepared", "role": "legacy_comparison", "encoding": old_evidence, "status": "reading"}
        report["old_source"] = legacy_attempt
        _write_index(output, report, filename="PREPARATION.json")
        try:
            old = read_transfer_table(store.verify_artifact(old_sha), grain="campus", year=year, policy=policy,
                expected_sha256=old_sha, encoding=old_encoding, layout="legacy_prepared")
            report["old_source_direction_validation"] = validate_campus_directions(old, max_examples=max_examples)
            report["old_prepared_comparison"] = compare_campus_sources(old, combined, max_examples=max_examples)
        except Exception as error:
            return retain_failure(error, phase="legacy_comparison", failed_input=legacy_attempt)
        legacy_attempt.update(status="parsed", rows=old.table.num_rows)
        report["old_source_encoding"] = old_evidence
    artifact = output / "prepared-source.parquet"
    descriptor = os.open(artifact, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        pq.write_table(merged, stream, compression="zstd", row_group_size=4096)
    offset = 0
    with pq.ParquetFile(artifact) as reopened:
        for batch in reopened.iter_batches(batch_size=4096, use_threads=False):
            actual = pa.Table.from_batches([batch])
            if not actual.equals(merged.slice(offset, batch.num_rows), check_metadata=True):
                raise ValueError("Transfer prepared native readback differs from retained raw strings/projections")
            offset += batch.num_rows
    if (offset != merged.num_rows or processor_identity() != code
            or (wire_schema is not None and (Path(__file__).parent / "schemas/data_sources--campus_transfer_reports--2025--default.json").read_bytes() != wire_schema)):
        raise ValueError("Transfer preparation source/row coverage changed before retention")
    report.update(status="prepared", rows=merged.num_rows, columns=merged.column_names,
                  raw_string_readback_equal=True, publisher_totals_computed=False)
    _write_index(output, {"sources": annotations}, filename="ANNOTATIONS.json")
    _write_index(output, report, filename="VALIDATION.json")
    processing = store.record_processing(dataset=expected_dataset, inputs=[*roots, *local_inputs],
        outputs={"prepared_source": artifact, "annotations": output / "ANNOTATIONS.json",
                 "validation_report": output / "VALIDATION.json"},
        processor="teadata.scripts.refresh_transfers", code_version=code,
        parameters={"no_config_activation": True, "value_policy": policy.to_dict(), **implementation,
                    "toolchain": report["toolchain"]},
        validation={"status": "PASS", "raw_string_readback_equal": True,
                    "rows": merged.num_rows, "field_correctness_qualified": False})
    store.verify_record_closure([processing])
    report["processing_record_id"] = processing
    _write_index(output, report, filename="PREPARATION.json")
    return report
