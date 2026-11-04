"""Offline field qualification for explicitly selected FY2025 financial actuals."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html import unescape
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any
from urllib.parse import parse_qsl
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq

from .analytical import write_leaf_artifact
from .campus_financial_leaves import _FACTS_PER_CAMPUS, _admit_intake, _read_bounded, _read_report
from .column_schema import _unique_object, source_schema_fingerprint
from .dataset_metadata import ArtifactRef, DatasetMetadata, EditionKey, ResolvedDataset
from .financial_leaves import NS, _package, _xml
from .manifest import BuildRecorder, canonical_json, file_identity
from .provenance import ProvenanceStore
from .release import DataRelease, _install_candidate, publish_release
from .teadata_config import load_config_bytes
from .values import ValueStatus


_YEAR = 2025
_PERIOD = {"kind": "fiscal_year", "start_year": 2024, "end_year": _YEAR}
_CAMPUS_DATASET = "campus_peims_financials"
_CAMPUS_FIELD = "all_funds_total_expenditures"
_DISTRICT_DATASET = "district_pwr_actuals"
_DISTRICT_FIELD = "af_college_career_and_military_readiness_ccmr_pic_38"
_TABLE = "qualified_actual_expenditures"
_PIC38 = "College, Career, and Military Readiness (CCMR) (PIC 38)"
_PDF_TITLE = "2024-2025 PEIMS Actual Financial Data, Organized by Campus"
_PDF_OBJECT = "Expenditures by Object (Objects 6100-6600)"
_PDF_FUNCTION = "Expenditures by Function (Objects 6100-6400 Only)"
_BROKER = "https://rptsvr1.tea.texas.gov/cgi/sas/broker"
_REPORTS = "https://rptsvr1.tea.texas.gov/school.finance/forecasting/financial_reports"
_STANDARD_REPORTS = "https://tea.texas.gov/about-tea/state-funding/state-funding-reports-and-data/peims-financial-standard-reports"
_PWR_URL = "https://tea.texas.gov/about-tea/state-funding/state-funding-reports-and-data/actual2025-info-gen-all.xlsx"
_FASRG_URL = "https://tea.texas.gov/data-reports/financial-accountability/fasrg/fasrg19-module1-far-appendices-0.pdf"
_RAW_DICTIONARY_URL = "https://tea.texas.gov/data-reports/financial-reports/school-finance-reports-and-data/reports-and-downloadable-data/datadictionary.pdf"
_AMOUNT = re.compile(r"-?[0-9]+")
_PRINTED_AMOUNT = re.compile(r"\$?-?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)")
_HTML_CELL = re.compile(r"<(td|th)\b([^>]*)>(.*?)</\1>", re.S)
_CONTEXT = ("source_receipt_id", "source_sheet", "source_cell", "witness_receipt_id", "witness_locator")
_CASE_IDENTITIES = frozenset({"about_actual_receipt", "accounting_guide_receipt", "campus_preparation_receipt",
                             "cohort_intake_sha256", "district_pwr_receipt", "district_selector_receipt",
                             "raw_dictionary_receipt"})
_SHA256 = re.compile(r"[a-f0-9]{64}")


@dataclass(frozen=True)
class _Amount:
    raw: str
    value: int
    source_id: str
    sheet: str
    cell: str
    witness_id: str
    locator: str


@dataclass(frozen=True)
class _Selection:
    dataset: str
    scope: str
    field: str
    measure_locator: str
    label: str
    identity_locator: str
    notes: str
    source_url: str
    sources: tuple[str, ...]
    inputs: tuple[str, ...]
    values: dict[str, _Amount]


def _amount(raw: str) -> int:
    if not isinstance(raw, str) or not _AMOUNT.fullmatch(raw):
        raise ValueError("Reviewed finance field requires an original signed integer lexeme")
    result = int(raw)
    if not -(1 << 63) <= result < 1 << 63:
        raise ValueError("Reviewed finance field exceeds the signed int64 range")
    return result


def _printed_amount(raw: str) -> int:
    if not _PRINTED_AMOUNT.fullmatch(raw):
        raise ValueError("Publisher finance amount is not an explicit integer")
    return _amount(raw.removeprefix("$").replace(",", ""))


def _receipt(store: ProvenanceStore, identifier: str, url: str) -> dict[str, Any]:
    record = store.read_record(identifier)
    if (record["kind"] != "acquisition" or record["method"] != "download"
            or record["requested_url"] != url or record["resolved_url"] != url
            or record["parameters"].get("http_method") != "GET" or record["parameters"].get("http_status") != 200):
        raise ValueError("Finance witness requires the exact publisher acquisition")
    return record


def _bytes(store: ProvenanceStore, record: dict[str, Any]) -> bytes:
    artifact = record["artifact"]
    return _read_bounded(store.root / "objects" / artifact["sha256"], artifact["sha256"], artifact["size_bytes"])


def _district_edition(record: dict[str, Any], media: str) -> None:
    # The earlier PWR and Austin observations did not assign receipt years.
    if (record["source_metadata"].get("version") != str(_YEAR)
            or (record["requested_year"], record["resolved_year"]) not in {(None, None), (_YEAR, _YEAR)}
            or record["response_headers"].get("content-type", "").split(";")[0].strip().lower() != media):
        raise ValueError("District financial acquisition edition or media type differs")


def _pdf_amounts(text: str, district: str) -> dict[str, tuple[int, int]]:
    results = {}
    seen = set()
    for page, body in enumerate(text.split("\f"), 1):
        if not body.strip():
            continue
        lines = [" ".join(line.split()) for line in body.splitlines()]
        identifiers = [m[1] for line in lines if (m := re.fullmatch(r"Campus Number:\s*([0-9]{9}) Total Membership:\s*[0-9,]+", line))]
        if _PDF_TITLE not in lines or len(identifiers) != 1 or not identifiers[0].startswith(district):
            raise ValueError("Campus PDF differs from its selected FY2025 district or report")
        campus = identifiers[0]
        seen.add(campus)
        if _PDF_OBJECT not in lines:
            continue
        if (lines.count(_PDF_OBJECT) != 1 or lines.count(_PDF_FUNCTION) != 1
                or lines.index(_PDF_OBJECT) >= lines.index(_PDF_FUNCTION) or campus in results
                or "General Per All Per" not in lines or "Fund % Student Funds % Student" not in lines):
            raise ValueError("Campus PDF object section, fund columns or campus grain differs")
        totals = [line.split()[2:] for line in lines[lines.index(_PDF_OBJECT) + 1:lines.index(_PDF_FUNCTION)]
                  if line.startswith("Total Expenditures ")]
        if len(totals) != 1 or len(totals[0]) != 6:
            raise ValueError("Campus PDF requires exactly one six-role object Total Expenditures line")
        results[campus] = (page, _printed_amount(totals[0][3]))
    if not results or results.keys() != seen:
        raise ValueError("Campus PDF witness is missing a selected object total")
    return results


def _district_html_amount(data: bytes, district: str) -> tuple[int, int]:
    text = data.decode("iso-8859-1")

    def label(cell: re.Match[str]) -> str:
        return " ".join(unescape(re.sub(r"<br\s*/?>", " ", cell[3])).split())

    titles = [label(cell) for cell in _HTML_CELL.finditer(text) if 'class="c systemtitle"' in cell[2]]
    if len([title for title in titles if title.startswith("2024 - 2025 Actual Financial Data Totals for ")
            and title.endswith(f" ({district})")]) != 1:
        raise ValueError("District HTML witness differs from its selected FY2025 district")
    table, = re.finditer(r'<table\s+class="table"[^>]*>(.*?)</table>', text, re.S)
    rows = [(row, list(_HTML_CELL.finditer(row[1]))) for row in re.finditer(r"<tr>(.*?)</tr>", table[1], re.S)]
    headings = ("", "General Fund", "%", "Per Student", "All Funds", "%", "Per Student", "All Funds", "%", "Per Student")
    groups, = [cells for _, cells in rows if tuple(map(label, cells)) == ("", "District", "State")]
    if ('colspan="6"' not in groups[1][2] or 'colspan="3"' not in groups[2][2]
            or sum(tuple(map(label, cells)) == headings for _, cells in rows) != 1):
        raise ValueError("District witness must retain separate District and State column groups")
    matches = [(row, cells) for row, cells in rows if len(cells) == len(headings) and label(cells[0]) == _PIC38]
    if len(matches) != 1:
        raise ValueError("District HTML witness has ambiguous PIC 38 lines")
    row, cells = matches[0]
    printed = label(cells[4])
    if not printed.startswith("$"):
        raise ValueError("District witness does not label its amount as dollars")
    line = text.count("\n", 0, table.start(1) + row.start(1) + cells[4].start()) + 1
    return line, _printed_amount(printed)


def _pwr_amounts(data: bytes, districts: set[str]) -> dict[str, tuple[str, str]]:
    with ZipFile(BytesIO(data)) as archive:
        sheets, strings = _package(archive)
        sheet = _xml(archive, sheets["24_25 AF Act"])
    result = {}

    def raw(cell):
        if cell.find(NS + "f") is not None:
            raise ValueError("PWR reviewed field cannot be a formula")
        value = cell.findtext(NS + "v")
        return strings[int(value)] if cell.get("t") == "s" else value

    header, = sheet.findall(f".//{NS}c[@r='BN1']")
    district_header, = sheet.findall(f".//{NS}c[@r='C1']")
    if raw(header) != "AF " + _PIC38 or raw(district_header) != "DISTRICT NUMBER":
        raise ValueError("PWR literal district identity or BN1 All Funds PIC 38 header differs")
    for row in sheet.findall(NS + "sheetData/" + NS + "row"):
        number = row.attrib["r"]
        identities = row.findall(f"{NS}c[@r='C{number}']")
        if len(identities) > 1:
            raise ValueError("PWR has a duplicate physical district identity cell")
        if not identities or (code := raw(identities[0])) is None or not re.fullmatch(r"[0-9]{1,6}", code):
            continue
        district = code.zfill(6)
        if district not in districts:
            continue
        coordinate = f"BN{number}"
        cells = row.findall(f"{NS}c[@r='{coordinate}']")
        if district in result or len(cells) != 1 or cells[0].get("t", "n") != "n":
            raise ValueError("PWR selected district identity or numeric cell is duplicate or unavailable")
        result[district] = (coordinate, raw(cells[0]))
    if result.keys() != districts:
        raise ValueError("PWR original does not cover the exact selected district cohort")
    return result


def _matched(raw: str, witnessed: int, *, entity: str, cell: str) -> int:
    amount = _amount(raw)
    if amount != witnessed:
        raise ValueError(f"Publisher finance witness differs for {entity} at {cell}")
    return amount


def _campus_selection(store: ProvenanceStore, case: dict, pdf_tool: str) -> _Selection:
    prep_id = case["campus_preparation_receipt"]
    graph = store.verify_record_closure([prep_id])
    prep = graph[prep_id]
    intake_id, = [identifier for identifier in prep["inputs"]
                  if graph[identifier]["dataset"] == "reference:campus_financial_intake"]
    validation = prep["validation"]
    if (prep["processor"] != "teadata.campus_financial_leaves.materialize_campus_financial_workbooks"
            or prep["parameters"]["intake_sha256"] != case["cohort_intake_sha256"]
            or graph[intake_id]["artifact"]["sha256"] != case["cohort_intake_sha256"]
            or validation["source_projection_readback_equal"] is not True):
        raise ValueError("Campus raw preparation is not the reviewed FY2025 intake")
    intake = _bytes(store, graph[intake_id])
    selected = _admit_intake(store, intake)
    if ({workbook.district for workbook in selected} != case["districts"].keys()
            or validation["rows"] != {"campus_reports": len(selected), "campus_financial_facts": len(selected) * _FACTS_PER_CAMPUS}):
        raise ValueError("Campus raw preparation differs from the reviewed district selection or row coverage")
    artifact = prep["outputs"]["campus_financial_facts"]
    table = pq.read_table(pa.BufferReader(_bytes(store, {"artifact": artifact})),
        columns=["campus_number", "raw_value", "source_receipt_id"], filters=[("source_cell", "=", "E7")])
    prepared = {row["campus_number"]: (row["raw_value"], row["source_receipt_id"]) for row in table.to_pylist()}
    if table.num_rows != len(prepared) or prepared.keys() != {workbook.campus for workbook in selected}:
        raise ValueError("Campus prepared E7 cells are missing, duplicated or outside the reviewed cohort")
    values, pdf_ids = {}, []
    for district, witnesses in sorted(case["districts"].items()):
        pdf_id = witnesses["campus_pdf"]
        url = f"{_BROKER}?_service=marykay&_debug=0&_program=sfadhoc.allcamp_actual_report_2025.sas&who_list={district}"
        record = _receipt(store, pdf_id, url)
        if (record["resolved_year"] != _YEAR or record["parameters"].get("district_id") != district
                or record["response_headers"]["content-type"].split(";")[0].strip().lower() != "application/pdf"):
            raise ValueError("Campus PDF acquisition edition, district or media type differs")
        pdf_text = subprocess.run([pdf_tool, "-layout", "-", "-"], input=_bytes(store, record), capture_output=True,
                                  check=True, timeout=25).stdout.decode("utf-8")
        printed = _pdf_amounts(pdf_text, district)
        if printed.keys() != {workbook.campus for workbook in selected if workbook.district == district}:
            raise ValueError("Campus PDF and original workbook roster disagree")
        for workbook in (workbook for workbook in selected if workbook.district == district):
            _, observations = _read_report(store, workbook)
            fact, = [row for row in observations if row["source_cell"] == "E7"]
            raw = fact["raw_value"]
            if (raw, workbook.receipt_id) != prepared[workbook.campus] or fact["column_role"] != "all_funds_amount":
                raise ValueError("Campus raw preparation differs from its original E7 cell")
            page, amount = printed[workbook.campus]
            value = _matched(raw, amount, entity=workbook.campus, cell="E7")
            values[workbook.campus] = _Amount(raw, value, workbook.receipt_id, fact["worksheet"], "E7", pdf_id, f"page {page}")
        pdf_ids.append(pdf_id)
    return _Selection(_CAMPUS_DATASET, "campus", _CAMPUS_FIELD, "E7", "Total Expenditures", "A1",
        "Object section, All Funds amount, nominal dollars as published for the publisher-witnessed campus roster. "
        "Shares, per-student values and membership interpretation remain outside this qualification.",
        f"{_REPORTS}/2425_new_camp_actual.html", tuple(workbook.receipt_id for workbook in selected),
        (prep_id, *pdf_ids), values)


def _district_selection(store: ProvenanceStore, case: dict) -> _Selection:
    raw_id = case["district_pwr_receipt"]
    districts = set(case["districts"])
    source = _receipt(store, raw_id, _PWR_URL)
    _district_edition(source, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    values, html_ids = {}, []
    actual = _pwr_amounts(_bytes(store, source), districts)
    for district, witnesses in sorted(case["districts"].items()):
        html_id = witnesses["district_html"]
        record = store.read_record(html_id)
        url = record["requested_url"]
        expected = [("_service", "marykay"), ("_program", "sfadhoc.actual_report_2025.sas"), ("_service", "appserv"),
                    ("_debug", "0"), ("who_box", ""), ("who_list", district)]
        broker, _, query = url.partition("?")
        if (broker != _BROKER or parse_qsl(query, keep_blank_values=True) != expected
                or record["parameters"].get("form_url") != _BROKER
                or record["parameters"].get("form_controls") != [list(pair) for pair in expected]):
            raise ValueError("District HTML receipt does not bind the selected district controls")
        _receipt(store, html_id, url)
        _district_edition(record, "text/html")
        line, witnessed = _district_html_amount(_bytes(store, record), district)
        cell, raw = actual[district]
        value = _matched(raw, witnessed, entity=district, cell=cell)
        values[district] = _Amount(raw, value, raw_id, "24_25 AF Act", cell, html_id, f"line {line}")
        html_ids.append(html_id)
    return _Selection(_DISTRICT_DATASET, "district", _DISTRICT_FIELD, "BN1", "AF " + _PIC38, "C1",
        "District All Funds PIC 38 expenditures in nominal dollars for the publisher-witnessed district roster. "
        "No denominator, CCMR readiness rate, Outcomes Bonus award or SOF run is implied.",
        _STANDARD_REPORTS, (raw_id,), (raw_id, *html_ids), values)


def _local_observation(store: ProvenanceStore, path: Path, title: str) -> str:
    return store.record_acquisition(path, ResolvedDataset("reference:finance_qualification", "reference", None, None, "",
        DatasetMetadata(title=title)))


def _reference(store: ProvenanceStore, identifier: str, name: str, role: str, title: str, url: str) -> dict:
    record = _receipt(store, identifier, url)
    return {"reference_id": name, "role": role, "status": "available", "title": title, "url": url,
            "provenance_ids": [identifier], "reviewed_sha256": record["artifact"]["sha256"]}


def _source_config(store: ProvenanceStore, stage: Path, case: dict, selections: tuple[_Selection, ...]) -> tuple[Path, bytes]:
    references = [
        _reference(store, case["about_actual_receipt"], "about_actual", "source_notes", "About the FY2025 PEIMS Actual Financial Data Reports", f"{_REPORTS}/AbtAct25.docx"),
        _reference(store, case["accounting_guide_receipt"], "accounting_guide", "methodology", "FASRG version 19 FAR appendices, including campus coding and PIC 38", _FASRG_URL),
        _reference(store, case["raw_dictionary_receipt"], "raw_account_dictionary", "data_dictionary", "Raw PEIMS account-record dictionary, separate from the derived standard reports", _RAW_DICTIONARY_URL),
    ]
    sources = {}
    for selected in selections:
        identity = f"{selected.scope}_number"
        labels = ["DISTRICT NUMBER" if selected.scope == "district" else "Campus Number", selected.label]
        fields = [{"position": position, "field": field, "code": label, "label": label,
                   "disposition": "identity" if position == 1 else "fact",
                   "dictionary_status": "not_applicable" if position == 1 else "not_documented",
                   "publisher_type": None, "publisher_description": None}
                  for position, (field, label) in enumerate(zip((identity, selected.field), labels, strict=True), 1)]
        hashes = [store.read_record(identifier)["artifact"]["sha256"] for identifier in selected.sources]
        crosswalk = {"schema_version": 2, "mapping_version": 1, "datasets": [{"dataset": selected.dataset,
            "edition": str(_YEAR), "source_only": True, "tables": [{"table": _TABLE,
            "header_identity": "position", "input_sha256s": hashes, "source_schema_sha256": source_schema_fingerprint(labels, labels),
            "metric_period": _PERIOD, "fields": fields}]}]}
        raw_crosswalk = (canonical_json(crosswalk) + "\n").encode()
        resource = stage / f"{sha256(raw_crosswalk).hexdigest()}.json"
        resource.write_bytes(raw_crosswalk)
        schema_receipt = _local_observation(store, resource, "Exact reviewed FY2025 finance crosswalk")
        source_refs = [*references, {"reference_id": "columns", "role": "schema", "status": "available",
            "path": resource.name, "reviewed_sha256": file_identity(resource)["sha256"], "provenance_ids": [schema_receipt]}]
        if selected.scope == "district":
            source_refs.append(_reference(store, case["district_selector_receipt"], "district_selector", "source_notes",
                                          "FY2025 district financial actual selector", f"{_REPORTS}/2425_FinActRep.html"))
            source_refs.append(_reference(store, selected.sources[0], "original_workbook", "source_notes",
                                          "Original FY2025 statewide PWR actual workbook", _PWR_URL))
        source_metadata = {"title": f"FY2025 {selected.scope} financial actuals", "version": str(_YEAR), "publisher": "Texas Education Agency",
            "source_url": selected.source_url, "period": _PERIOD, "identifiers": [identity], "notes": selected.notes,
            "references": source_refs, "metrics": {selected.field: {"field": selected.field, "unit": "currency"}},
            "source_qualifiers": {"family": "peims_financial", "reference_id": "about_actual", "financial_basis": "actual"}}
        sources[selected.dataset] = {_YEAR: {"path": hashes[0], "metadata": source_metadata,
            "reader": {"profiles": {"source": {"column_schema": resource.name}}}}}
    config = stage / "source-config.json"
    raw = (canonical_json({"data_sources": sources}) + "\n").encode()
    config.write_bytes(raw)
    return config, raw


def _leaf_table(selected: _Selection) -> tuple[pa.Table, dict[str, Any]]:
    observed = ValueStatus.OBSERVED.value
    data = list(sorted(selected.values.items()))
    columns = {f"{selected.scope}_number": pa.array([identifier for identifier, _ in data], type=pa.string()),
        selected.field: pa.array([item.value for _, item in data], type=pa.int64()),
        "raw_value": pa.array([item.raw for _, item in data], type=pa.string()), "value_status": pa.repeat(observed, len(data))}
    for column, field in zip(_CONTEXT, ("source_id", "sheet", "cell", "witness_id", "locator"), strict=True):
        columns[column] = pa.array([getattr(item, field) for _, item in data], type=pa.string())
    return pa.table(columns), {"fields": {selected.field: {"raw": "raw_value", "status": "value_status"}}, "context": list(_CONTEXT)}


def _verify_release(path: Path, pin: str, selections: tuple[_Selection, ...]) -> None:
    release = DataRelease(path, expected_sha256=pin, offline=True)
    artifacts = release.verify()
    editions = list(release.catalog.editions(include_metadata=False))
    if len(artifacts) != len(editions) or len(editions) != len(selections):
        raise ValueError("Qualified finance release has unexpected artifact or edition coverage")
    for selected in selections:
        edition, = [row for row in editions if row["dataset"] == selected.dataset and row["scope"] == selected.scope]
        key = EditionKey.from_dict({name: edition[name] for name in EditionKey.__dataclass_fields__})
        identity = f"{selected.scope}_number"
        variables = {row["field"]: row for row in release.catalog.variables(edition=key)}
        if (set(variables) != {identity, selected.field} or variables[selected.field]["unit"] != "currency"
                or variables[selected.field]["period"] != _PERIOD):
            raise ValueError("Qualified finance catalog admits an unreviewed field, unit or period")
        plan, = release.catalog.resolve_artifacts([(key, identity), (key, selected.field)])
        expected, _ = _leaf_table(selected)
        actual = release.analytical_dataset(plan["artifact"].identity).to_table(columns=expected.column_names)
        if not expected.equals(actual.replace_schema_metadata(None), check_metadata=True):
            raise ValueError("Qualified finance release readback differs from witnessed signed amounts")


def _qualification_case(raw: bytes) -> dict[str, Any]:
    case = json.loads(raw, object_pairs_hook=_unique_object)
    if (not isinstance(case, dict) or set(case) != _CASE_IDENTITIES | {"schema_version", "fiscal_end_year", "districts"}
            or type(case["schema_version"]) is not int or case["schema_version"] != 1
            or type(case["fiscal_end_year"]) is not int or case["fiscal_end_year"] != _YEAR
            or not isinstance(case["districts"], dict) or not case["districts"]):
        raise ValueError("Select a reviewed FY2025 financial qualification case")
    identities = [case[name] for name in _CASE_IDENTITIES]
    for district, witnesses in case["districts"].items():
        if (district == "000000" or not re.fullmatch(r"[0-9]{6}", district) or not isinstance(witnesses, dict)
                or set(witnesses) != {"campus_pdf", "district_html"}):
            raise ValueError("Finance qualification requires exact district identifiers and both publisher witnesses")
        identities.extend(witnesses.values())
    if any(not isinstance(identity, str) or not _SHA256.fullmatch(identity) for identity in identities):
        raise ValueError("Finance qualification requires SHA-256 receipt identities and intake pin")
    return case


def qualify_financial_actuals(store: ProvenanceStore, *, case_path: Path, case_sha256: str, output: Path) -> dict[str, Any]:
    """Create a separate, receipt-bound release for two witnessed nominal dollar fields."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("Finance qualification requires a new exclusive output directory")
    raw_case = _read_bounded(case_path, case_sha256)
    case = _qualification_case(raw_case)
    pdf_tool = shutil.which("pdftotext")
    if pdf_tool is None:
        raise ValueError("Finance qualification requires Poppler pdftotext")
    selections = (_campus_selection(store, case, pdf_tool), _district_selection(store, case))
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".qualified-financials-", dir=output.parent))
    try:
        exact_case = stage / "witness-selection.json"
        exact_case.write_bytes(raw_case)
        case_id = _local_observation(store, exact_case, "Reviewed FY2025 financial source selection")
        config_path, raw_config = _source_config(store, stage, case, selections)
        config_id = _local_observation(store, config_path, "Exact FY2025 financial field source configuration")
        cfg = load_config_bytes(raw_config, path=config_path)
        recorder = BuildRecorder(cfg, store)
        toolchain = {"python": sys.version, "pyarrow": version("pyarrow"), "platform": platform.platform(),
                     "pdftotext_sha256": file_identity(Path(pdf_tool))["sha256"]}
        result: dict[str, Any] = {"status": "unpublished_qualified_financial_fields", "release_activation": False,
                                  "field_correctness_qualified": True, "fiscal_period": _PERIOD, "editions": []}
        for selected in selections:
            source = cfg.resolve_dataset(selected.dataset, _YEAR, reader_profile="source", exact=True)
            table, support = _leaf_table(selected)
            hashes = [store.read_record(identifier)["artifact"]["sha256"] for identifier in selected.sources]
            key = EditionKey(selected.dataset, str(_YEAR), EditionKey.source_revision(tuple(hashes)), selected.scope,
                _TABLE, f"one qualified FY2025 dollar amount per selected {selected.scope}")
            recipe = {"operation": "qualify_fy2025_actual_expenditures", "case_sha256": case_sha256,
                      "config_sha256": sha256(raw_config).hexdigest(), "field": selected.field, "unit": "currency",
                      "source_locators": {"identity": selected.identity_locator, "measure": selected.measure_locator},
                      "denominators_used": False}
            contract = recorder.leaf_contract(key, source, native_schema=table.schema.serialize().to_pybytes(),
                columns=table.column_names, key_columns=[f"{selected.scope}_number"], rows=table.num_rows,
                inputs=hashes, recipe=recipe, toolchain=toolchain, column_support=support)
            leaf_id = sha256(canonical_json(contract).encode()).hexdigest()
            leaf = stage / f"{leaf_id}.parquet"
            physical = write_leaf_artifact(leaf, table.to_batches(), contract=contract, batch_bytes=2 * 1024**2)
            if not table.equals(pq.read_table(leaf).replace_schema_metadata(None), check_metadata=True):
                raise ValueError("Qualified finance Parquet readback differs from matched values")
            refs = [identifier for item in source.metadata.references for identifier in item.provenance_ids]
            roots = list(dict.fromkeys([*selected.inputs, *refs, case_id, config_id]))
            processing = store.record_processing(dataset=selected.dataset, inputs=roots, outputs={"leaf": leaf},
                processor="teadata.finance_qualification.qualify_financial_actuals", code_version=recorder.processor_sha256,
                parameters={"leaf_contract_sha256": leaf_id, "recipe": recipe, "toolchain": toolchain},
                validation={"field_correctness_qualified": True, "publisher_matches": table.num_rows,
                            "raw_readback_equal": True, "release_activation": False})
            ref = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name, canonical_json(contract), processing, "leaf")
            recorder.record_edition(key, source, artifacts=[ref], acquisition_ids=selected.sources)
            result["editions"].append({"key": key.to_dict(), "field": selected.field, "artifact_id": leaf_id,
                "processing_record_id": processing, "publisher_matches": table.num_rows})
        catalog = recorder.catalog(schema_version=4)
        manifest = publish_release(stage, catalog, schema_version=4)
        manifest_identity = file_identity(manifest)
        _verify_release(manifest, manifest_identity["sha256"], selections)
        result.update(build_id=catalog.build_id, manifest=manifest_identity, release_verified=True,
                      catalog_readback_equal=True, qualification_scope="FY2025 selected fields and publisher-witnessed rosters only")
        (stage / "MATERIALIZATION.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _install_candidate(stage, output)
        return result
    finally:
        if stage.exists():
            shutil.rmtree(stage)
