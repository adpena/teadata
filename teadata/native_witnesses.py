"""Replay the separately selected TEA reports for the reviewed profile supplement."""

from __future__ import annotations

import csv
from html.parser import HTMLParser
from importlib.resources import files
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from .manifest import ContractError
from .peims_leaves import PRIMARY_DISABILITY
from .transfer_sources import _FOOTNOTES

if TYPE_CHECKING:
    from .dataset_metadata import ArtifactRef, EditionKey
    from .provenance import ProvenanceStore
    from .release import DataRelease


_PROGRAM = "peims_student_programs_and_special_populations"
_ETHNICITY = "peims_student_enrollment_by_ethnicity"
_TRANSFER = "district_transfer_reports"
_BROKER = "https://rptsvr1.tea.texas.gov/cgi/sas/broker"
_PROGRAM_FIELDS = {
    "allenrollment": "total_enrollment_count",
    "atrisk": "at_risk_count",
    "bilingual": "bilingual_program_count",
    "dyslexic": "dyslexia_identification_count",
    "economicallydisadvantaged": "economically_disadvantaged_count",
    "emergentbilingual": "emergent_bilingual_count",
    "esl": "english_as_a_second_language_program_count",
    "fostercare": "foster_care_count",
    "gifted&talented": "gifted_and_talented_program_count",
    "homeless": "homeless_count",
    "militaryconnected": "military_connected_count",
    "section504": "section_504_count",
    "specialeducation": "special_education_program_count",
    "titlei": "title_i_program_count",
}
_TRANSFER_HEADER = ("YEAR", "REPORT_REGION", "REPORT_REGION_NAME", "REPORT_DISTNAME_NUMBER",
                    "REPORT_CHARTER_STATUS", "REPORT_NUMBER", "REPORT_TYPE", "LINE_GROUP_NUMBER",
                    "DISTNAME_NUMBER_RES_ATTEND", "TRANSFERS_IN_OUT")
_TRANSFER_COLUMNS = ("school_year", "REPORT_REGION", "reporting_region_name", "REPORT_DISTNAME_NUMBER",
                     "reporting_charter_status", "report_number", "REPORT_TYPE", "line_group_number",
                     "DISTNAME_NUMBER_RES_ATTEND", "TRANSFERS_IN_OUT")


class _Rows(HTMLParser):
    def __init__(self, content: str):
        super().__init__()
        self.rows: list[list[str]] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None
        self.feed(content)

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        if tag in ("th", "td") and self.row is not None:
            self.cell = []

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self.cell is not None and self.row is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        if tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


def reviewed_native_references() -> dict[str, Any]:
    return json.loads(files("teadata").joinpath("reference_cases/native_peims_transfers.json").read_text())


def _label(value: str) -> str:
    return "".join(value.casefold().split())


def _count(token: str) -> tuple[int | None, str]:
    if token == "N/A" or re.fullmatch(r"<(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)", token):
        return None, "suppressed"
    if re.fullmatch(r"[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+", token):
        return int(token.replace(",", "")), "observed"
    raise ContractError(f"Unexpected TEA count token: {token!r}")


def _compare_count(native: dict[str, Any], field: str, token: str) -> None:
    if (native[field], native[f"peims_status_{field}"]) != _count(token):
        raise ContractError(f"Independent PEIMS count or status differs: {field}")
    if token.startswith("<") and native[f"peims_raw_{field}"].replace(",", "") != token.replace(",", ""):
        raise ContractError(f"Independent PEIMS suppression bound differs: {field}")


def _open_witness(store: ProvenanceStore, receipt: str, ref: ArtifactRef, entity: str) -> Path:
    record = store.read_record(receipt)
    key = ref.edition
    parameters = record.get("parameters", {})
    controls = parameters.get("form_controls", [])
    fields = dict(controls)
    artifact = record["artifact"]
    expected_method = "POST" if key.dataset in (_PROGRAM, PRIMARY_DISABILITY) and key.scope != "campus" else "GET"
    expected_url = _BROKER if expected_method == "POST" else f"{_BROKER}?{urlencode(fields)}"
    expected_media = "text/csv" if key.dataset == _TRANSFER else "text/html"
    if (record["kind"] != "acquisition" or record["method"] != "download"
            or record["section"] != "reference"
            or record["requested_year"] != record["resolved_year"] or record["resolved_year"] != int(key.edition)
            or record["requested_url"] != expected_url or record["resolved_url"] != expected_url
            or record["response_headers"].get("content-type", "").split(";")[0].strip().lower() != expected_media
            or parameters.get("form_url") != _BROKER or parameters.get("http_method") != expected_method
            or parameters.get("http_status") != 200 or len(controls) != len(fields)
            or artifact["sha256"] in ref.contract["inputs"]
            or not 0 < artifact["size_bytes"] <= 1024 * 1024):
        raise ContractError(f"Invalid independent publisher acquisition: {receipt}")
    if key.dataset == _TRANSFER:
        valid = (record["dataset"] == "reference:individual_district_transfer_report"
                 and fields.get("_program") == "adhoc.download_transfer_report.sas"
                 and fields.get("school_year") == key.edition and fields.get("scope") == "District"
                 and fields.get("agg_level") == "district" and fields.get("district") == entity)
    elif key.dataset in (_PROGRAM, PRIMARY_DISABILITY):
        disability = key.dataset == PRIMARY_DISABILITY
        valid = (record["dataset"] == ("reference:peims_primary_disability_witness" if disability
                                        else "reference:peims_individual_witness")
                 and fields.get("_program") == f"adhoc.std_driver{1 if key.scope == 'state' else 2}.sas"
                 and fields.get("RptClass") == ("SpecEd" if disability else "StudPgm")
                 and fields.get("SchoolYr") == key.edition[-2:]
                 and fields.get("report") == key.scope.title() * 2
                 and (not disability or (key.scope in ("state", "district") and fields.get("format") == "html"))
                 and (key.scope == "state" or fields.get(key.scope, "").startswith(f"{entity} ")))
    else:
        valid = (record["dataset"] == "reference:peims_individual_witness"
                 and fields.get("_program") == "adhoc.addispatch.sas" and fields.get("grouping") == "e"
                 and fields.get("endyear") == key.edition[-2:]
                 and fields.get("selsumm") == {"state": "ss", "district": "id", "campus": "nc"}[key.scope]
                 and fields.get("key", "") == ("" if key.scope == "state" else entity))
    if not valid:
        raise ContractError(f"Publisher witness selection differs: {receipt}")
    path = store.verify_artifact(artifact["sha256"])
    if path.stat().st_size != artifact["size_bytes"]:
        raise ContractError(f"Publisher witness size differs: {receipt}")
    return path


def _program(rows: list[list[str]], native: list[dict[str, Any]], entity: str) -> tuple[int, int]:
    index, = (index for index, row in enumerate(rows) if "All Enrollment" in row)
    if entity != "state" and not any(f"({entity})" in cell for row in rows[:index] for cell in row):
        raise ContractError("Program publisher report entity differs")
    record, = native
    facts = [(_PROGRAM_FIELDS[_label(label)], token)
             for label, token in zip(rows[index], rows[index + 1], strict=True) if _label(label) in _PROGRAM_FIELDS]
    if len(facts) != len(_PROGRAM_FIELDS) or {field for field, _ in facts} != set(_PROGRAM_FIELDS.values()):
        raise ContractError("Program publisher report lacks its reviewed measures")
    for field, token in facts:
        _compare_count(record, field, token)
    total = record["total_enrollment_count"]
    if type(total) is not int:
        raise ContractError("Program publisher witness requires an observed enrollment total")
    return len(facts), total


def _primary_disability(rows: list[list[str]], native: list[dict[str, Any]], key: EditionKey, entity: str,
                        columns: dict[str, str], expected_unreported: list[str]) -> tuple[int, list[str]]:
    group, = (index for index, row in enumerate(rows) if "Primary Disability" in row)
    title = "Students Receiving Special Education Services By Primary Disability"
    period = f"PEIMS Data {int(key.edition) - 1}-{key.edition}"
    if not any(title in cell and period in cell for row in rows[:group] for cell in row):
        raise ContractError("Primary disability publisher title or period differs")
    codes, published = rows[group + 1:group + 3]
    omitted = sorted(set(columns) - set(codes))
    if (len(codes) != len(set(codes)) or not set(codes) <= set(columns)
            or not isinstance(expected_unreported, list) or omitted != expected_unreported):
        raise ContractError("Primary disability publisher report columns differ")
    if key.scope == "district":
        if (not published or not re.search(rf"\({re.escape(entity)}\s*\)$", published[0])
                or not any("Districtwide for " in cell and published[0] in cell
                           for row in rows[:group] for cell in row)):
            raise ContractError("Primary disability publisher district differs")
        published = published[3:]
    elif not any(row == ["Statewide"] for row in rows[:group]):
        raise ContractError("Primary disability publisher state scope differs")
    record, = native
    for code, token in zip(codes, published, strict=True):
        _compare_count(record, columns[code], token)
    return len(codes), omitted


def _ethnicity(rows: list[list[str]], native: list[dict[str, Any]], key: EditionKey, entity: str, total: int) -> int:
    header, = (index for index, row in enumerate(rows) if "Ethnicity" in row)
    data = {_label(row["ethnicity"]): row for row in native}
    if len(data) != len(native):
        raise ContractError("Duplicate ethnicity witness categories")
    report_rows = rows[header + 1:]
    if key.scope != "state":
        column = rows[header].index(key.scope.title())
        if report_rows[0][column] != entity or any(any(row[:-2]) for row in report_rows[1:]):
            raise ContractError("Ethnicity publisher report entity differs")
    for row in report_rows[:-1]:
        name, token = row[-2:]
        _compare_count(data.pop(_label(name)), "enrollment_by_ethnicity_count", token)
    if (data or report_rows[-1][-2] != f"{key.scope.title()} Total"
            or _count(report_rows[-1][-1]) != (total, "observed")):
        raise ContractError("Ethnicity publisher categories or total differ")
    return len(report_rows)


def _transfer(path: Path, ref: ArtifactRef, district: str, table: Any) -> int:
    from pyarrow.dataset import field

    with path.open(encoding="utf-8", newline="") as stream:
        rows = [tuple(row) for row in csv.reader(stream, strict=True)]
    header, = (index for index, row in enumerate(rows) if row == _TRANSFER_HEADER)
    number = district.split(" ", 1)[0]
    period = f"{int(ref.edition.edition) - 1}-{ref.edition.edition}"
    body = rows[header + 1:]
    end = next((index for index, row in enumerate(body) if len(row) != len(_TRANSFER_HEADER)), len(body))
    source, footer = body[:end], body[end:]
    if (("District Level Transfer Report",) not in rows[:header] or not source
            or any(row[0] != period or not row[3].endswith(f"  -  {number}") for row in source)
            or any(len(row) > 1 for row in footer)
            or tuple(row[0].strip() for row in footer if row) != _FOOTNOTES):
        raise ContractError("Independent transfer report edition, district or row width differs")
    selected = table.to_table(columns=list(_TRANSFER_COLUMNS), filter=field("reporting_district_number") == "'" + number)
    if source != [tuple(row[name] for name in _TRANSFER_COLUMNS) for row in selected.to_pylist()]:
        raise ContractError("Independent transfer report rows differ")
    return len(source)


def validate_native_witnesses(release: DataRelease, store: ProvenanceStore, references: dict[str, Any]) -> dict[str, Any]:
    """Check the exact reviewed artifacts against independent report bytes."""
    from pyarrow.dataset import field

    cases = []
    peims_values = transfer_rows = 0
    totals: dict[tuple[str, str], int] = {}
    refs = [(release.catalog.artifact_ref(case["artifact"]), case) for case in references.values()]
    for ref, case in sorted(refs, key=lambda item: item[0].edition.dataset != _PROGRAM):
        key = ref.edition
        table = release.analytical_dataset(ref.identity)
        columns = ({entry["code"]: entry["field"] for entry in release.catalog.variables(edition=key)
                    if entry["unit"] == "count"} if key.dataset == PRIMARY_DISABILITY else {})
        for witness in case["reports"]:
            entity, receipt = witness["entity"], witness["receipt"]
            omitted: list[str] = []
            try:
                if "unreported_codes" in witness and key.dataset != PRIMARY_DISABILITY:
                    raise ContractError("Unreported codes require a primary disability witness")
                path = _open_witness(store, receipt, ref, entity)
                if key.dataset == _TRANSFER:
                    count = _transfer(path, ref, entity, table)
                    transfer_rows += count
                else:
                    content = path.read_text(encoding="cp1252")
                    if f"{int(key.edition) - 1}-{key.edition}" not in content:
                        raise ContractError("PEIMS publisher report school year differs")
                    rows = _Rows(content).rows
                    selection = None if key.scope == "state" else field(f"{key.scope}_number") == entity
                    native = table.to_table(filter=selection).to_pylist()
                    if key.dataset == _PROGRAM:
                        count, totals[(key.scope, entity)] = _program(rows, native, entity)
                    elif key.dataset == PRIMARY_DISABILITY:
                        count, omitted = _primary_disability(rows, native, key, entity, columns,
                                                            witness.get("unreported_codes", []))
                    else:
                        count = _ethnicity(rows, native, key, entity, totals[(key.scope, entity)])
                    peims_values += count
            except (ValueError, KeyError, TypeError, IndexError, OSError, csv.Error, ContractError) as error:
                raise ContractError(f"Publisher witness {receipt} failed for {key.dataset} {entity}: {error}") from error
            cases.append({"edition_id": key.identity, "entity": entity, "receipt": receipt,
                          "rows" if key.dataset == _TRANSFER else "values": count,
                          **({"unreported_codes": omitted} if omitted else {})})
    return {"reports": cases, "peims_values": peims_values, "transfer_rows": transfer_rows}
