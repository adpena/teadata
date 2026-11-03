"""Explicit TEA transfer-source intake; never activates source configuration.

Regional district reports are independent publisher observations, not sums of
campus transfers. Acquisition preserves their original headings and footnotes.
"""

from __future__ import annotations

from dataclasses import dataclass
from email.message import Message
from hashlib import sha256
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping, NotRequired, Sequence, TypedDict
from urllib.parse import urljoin
from urllib.request import build_opener

from .acquisition import AcquisitionClient, NoRedirect, _form_request, acquire_form
from .dataset_metadata import DataPeriod, DatasetMetadata, ResolvedDataset
from .provenance import ProvenanceStore, _identifier


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


def _write_index(directory: Path, report: dict) -> None:
    descriptor, name = tempfile.mkstemp(dir=directory, prefix=".BATCH_INDEX.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / "BATCH_INDEX.json")
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
