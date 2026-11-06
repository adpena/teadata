"""Admission of the exact district roster and direct Texas TPRS report."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import gzip

from hashlib import sha256
from io import BytesIO

import json
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from .column_schema import _unique_object
from .dataset_metadata import ColumnSchemaRef, EditionKey

from .provenance import ProvenanceStore
from .tprs_students import BROKER, DATASET, EDITION, _DIGEST, _identifier, _need, _validate_report_acquisition, semantic_fields
from .values import ValueStatus


REGIONS = tuple(f"{number:02}" for number in range(1, 21))
DISTRICT_COUNT = 1202
MAX_ROSTER_BYTES = 256 * 1024
MAX_STATEWIDE_HTML_BYTES = 160 * 1024**2


@dataclass(frozen=True, slots=True)
class ReportSelection:
    scope: str
    identifier: str
    receipt: str
    sha256: str


@dataclass(frozen=True, slots=True)
class StatewideSelection:
    reports: tuple[ReportSelection, ...]
    roster: dict[str, dict[str, str]]


def roster_membership(roster: Mapping[str, Any], records: Mapping[str, Any]) -> dict[str, str]:
    """Return publisher district → roster receipt, verified from exact JSON bytes."""
    _need(set(roster) == set(REGIONS), "Statewide selection requires all 20 regional rosters")
    members: dict[str, str] = {}
    for region in REGIONS:
        entry = roster[region]
        _need(isinstance(entry, dict) and set(entry) == {"receipt", "response"}
              and isinstance(entry["receipt"], str) and isinstance(entry["response"], str),
              "Invalid roster entry")
        receipt = entry["receipt"]
        record = records[receipt]
        response = entry["response"].encode("utf-8")
        size = record["artifact"]["size_bytes"]
        _need(type(size) is int and 0 < len(response) == size <= MAX_ROSTER_BYTES
              and sha256(response).hexdigest() == record["artifact"]["sha256"], "Roster response differs from its receipt")
        url = urlsplit(record["requested_url"])
        controls = parse_qsl(url.query, keep_blank_values=True)
        expected = {"_service": "marykay", "_debug": "0", "_program": "perfrept.acct_json_response.sas",
                    "prmv_level": "D", "ccyy": str(EDITION), "prmv_var": "region", "prmv_text": region}
        parameters = record["parameters"]
        _need(record["kind"] == "acquisition" and record["method"] == "download" and record["section"] == "reference"
              and record["requested_year"] == record["resolved_year"] == EDITION
              and record["source_metadata"].get("version") == str(EDITION)
              and record["resolved_url"] == record["requested_url"] and record["downloaded_at"]
              and url._replace(query="").geturl() == BROKER and len(controls) == len(expected)
              and dict(controls) == expected and parameters.get("form_controls") == [list(pair) for pair in controls]
              and parameters.get("form_url") == BROKER and parameters.get("http_method") == "GET"
              and parameters.get("http_status") == 200
              and record["response_headers"].get("content-type", "").split(";")[0] == "application/json",
              "Roster receipt does not bind the exact public regional request")
        rows = json.loads(response, object_pairs_hook=_unique_object)
        _need(isinstance(rows, list) and bool(rows), "Roster response requires district records")
        for row in rows:
            _need(isinstance(row, dict) and row.get("REGION") == region, "Roster row does not belong to its region")
            identifier = row.get("DISTRICT")
            _identifier("district", identifier)
            _need(identifier not in members, "Duplicate district across regional rosters")
            members[identifier] = receipt
    _need(len(members) == DISTRICT_COUNT, "Statewide roster size differs from the reviewed edition")
    return members


def validate_report_ledger(reports: tuple[ReportSelection, ...], members: Mapping[str, str],
                           records: Mapping[str, Any], *, scope: str | None = None) -> None:
    districts = [entry.identifier for entry in reports if entry.scope == "district"]
    state = [entry for entry in reports if entry.scope == "state"]
    _need(districts == ([] if scope == "state" else sorted(members))
          and [entry.identifier for entry in state] == ([] if scope == "district" else ["S"])
          and len(reports) == len(districts) + len(state), "Report ledger does not cover the exact roster and Texas singleton")
    _need(len({entry.receipt for entry in reports}) == len({entry.sha256 for entry in reports}) == len(reports),
          "Report ledger repeats a receipt or source body")
    total = 0
    for entry in reports:
        record = records[entry.receipt]
        _validate_report_acquisition(record, scope=entry.scope, identifier=entry.identifier, expected_sha256=entry.sha256)
        metadata = record["source_metadata"]
        _need(record["requested_year"] == record["resolved_year"] == EDITION
              and metadata.get("version") == str(EDITION), "Statewide ledger requires original explicit-edition receipts")
        if entry.scope == "district":
            _need(members[entry.identifier] in metadata.get("provenance_ids", ()),
                  "District report receipt lacks its exact regional roster parent")
        total += record["artifact"]["size_bytes"]
    _need(total <= MAX_STATEWIDE_HTML_BYTES, "Statewide report ledger exceeds the aggregate source byte budget")


def admit_statewide_selection(store: ProvenanceStore, document: dict[str, Any]) -> StatewideSelection:
    """Read pinned roster bodies and validate selected report identities before parsing reports."""
    _need(set(document) == {"schema_version", "dataset", "edition", "rosters", "reports"}
          and type(document["schema_version"]) is int and document["schema_version"] == 2
          and document["dataset"] == DATASET and document["edition"] == str(EDITION), "Invalid statewide student intake")
    rosters = document["rosters"]
    _need(isinstance(rosters, dict) and set(rosters) == set(REGIONS)
          and all(isinstance(value, str) and _DIGEST.fullmatch(value) for value in rosters.values()), "Invalid roster receipt pins")
    roster = {}
    for region, receipt in sorted(rosters.items()):
        record = store.read_record(receipt)
        digest = record["artifact"]["sha256"]
        size = record["artifact"]["size_bytes"]
        _need(type(size) is int and 0 < size <= MAX_ROSTER_BYTES, "Roster response exceeds its byte budget")
        with store.verify_artifact(digest).open("rb") as stream:
            response = stream.read(size + 1)
        roster[region] = {"receipt": receipt, "response": response.decode("utf-8")}
    records = {receipt: store.read_record(receipt) for receipt in rosters.values()}
    members = roster_membership(roster, records)
    rows = document["reports"]
    _need(isinstance(rows, list) and len(rows) == len(members) + 1, "Incomplete statewide report selection")
    reports = []
    for row in rows:
        _need(isinstance(row, dict) and set(row) == {"scope", "identifier", "receipt", "sha256"}
              and all(isinstance(value, str) for value in row.values())
              and _DIGEST.fullmatch(row["receipt"]) is not None and _DIGEST.fullmatch(row["sha256"]) is not None,
              "Invalid report receipt pins")
        reports.append(ReportSelection(**row))
    selected = tuple(sorted(reports, key=lambda item: (item.scope, item.identifier)))
    validate_report_ledger(selected, members, {entry.receipt: store.read_record(entry.receipt) for entry in selected})
    return StatewideSelection(selected, roster)


STATEWIDE_OPERATION = "statewide_tprs_student_membership_enrollment"
MAX_AUDIT_LINE_BYTES = 4096
MAX_AUDIT_BYTES = 128 * 1024**2
MAX_AUDIT_ROWS = 1_000_000


def statewide_resources(record: Mapping[str, Any]) -> dict[str, ColumnSchemaRef]:
    """Route reviewed attachments by their processing output, never by a path in the intake."""
    recipe = record.get("parameters", {}).get("recipe", {})
    if recipe.get("operation") != STATEWIDE_OPERATION:
        _need("report_ledger_version" not in recipe, "Unsupported statewide student ledger operation")
        return {}
    _need(type(recipe.get("report_ledger_version")) is int and recipe["report_ledger_version"] == 2
          and record.get("processor") == "teadata.tprs_students"
          and set(record["outputs"]) == {"leaf", "roster", "raw_audit", "grid"}, "Invalid statewide student output contract")
    pair = recipe.get("cohort")
    _need(isinstance(pair, dict) and set(pair) == {"district", "state"}
          and all(isinstance(item, str) and _DIGEST.fullmatch(item) for item in pair.values())
          and pair["district"] != pair["state"], "Statewide student recipe requires exact district and Texas editions")
    refs = {}
    for role, suffix in (("roster", "json"), ("raw_audit", "jsonl.gz"), ("grid", "json")):
        artifact = record["outputs"][role]
        digest = artifact["sha256"]
        _need(isinstance(digest, str) and _DIGEST.fullmatch(digest) is not None, "Invalid student attachment identity")
        refs[role] = ColumnSchemaRef(digest, artifact["size_bytes"], f"tprs-{role}-{digest}.{suffix}")
    _need(refs["grid"].sha256 == recipe.get("definition_sha256"), "Student audit grid differs from its recipe pin")
    return refs


def statewide_ledger(document: Mapping[str, Any], key: EditionKey, roster: bytes,
                     processing: Mapping[str, Any]) -> tuple[ReportSelection, ...]:
    records = document["provenance"]
    inputs = processing["inputs"]
    intakes = [records[identifier] for identifier in inputs
               if records[identifier]["dataset"] == "reference:tprs_student_intake"]
    pin = processing["parameters"]["recipe"].get("intake_sha256")
    _need(isinstance(pin, str) and _DIGEST.fullmatch(pin) is not None and len(intakes) == 1
          and intakes[0]["kind"] == "acquisition" and intakes[0]["method"] == "local_observation"
          and intakes[0]["requested_year"] == intakes[0]["resolved_year"] == EDITION
          and intakes[0]["source_metadata"].get("version") == str(EDITION)
          and intakes[0]["artifact"]["sha256"] == pin, "Student ledger differs from its direct exact intake receipt")
    body = json.loads(roster, object_pairs_hook=_unique_object)
    _need(isinstance(body, dict), "Invalid statewide roster attachment")
    members = roster_membership(body, records)
    _need({entry["receipt"] for entry in body.values()} <= set(inputs),
          "Roster receipts are not direct processing inputs")
    receipts = document["edition"]["acquisition_ids"]
    _need(set(receipts) <= set(inputs), "Report receipts are not direct processing inputs")
    reports = tuple(ReportSelection(key.scope,
        dict(parse_qsl(urlsplit(records[receipt]["requested_url"]).query))["id"],
        receipt, records[receipt]["artifact"]["sha256"]) for receipt in receipts)
    validate_report_ledger(reports, members, records, scope=key.scope)
    return reports


def audit_rows(raw: bytes, reports: tuple[ReportSelection, ...]) -> Iterator[dict[str, Any]]:
    """Stream the exact scope's audited source coordinates with finite bounds."""
    receipts = {entry.receipt: number for number, entry in enumerate(reports)}
    last = (-1, 0, 0)
    total = 0
    with gzip.GzipFile(fileobj=BytesIO(raw)) as stream:
        for number in range(MAX_AUDIT_ROWS + 1):
            line = stream.readline(MAX_AUDIT_LINE_BYTES + 1)
            if not line:
                return
            total += len(line)
            _need(number < MAX_AUDIT_ROWS and total <= MAX_AUDIT_BYTES and len(line) <= MAX_AUDIT_LINE_BYTES
                  and line.endswith(b"\n"), "Student raw audit exceeds its bounds")
            row = json.loads(line, object_pairs_hook=_unique_object)
            fields = {"receipt", "row", "column", "section", "label", "population", "measure", "reason",
                      "token_status", "token_reason", "raw"}
            _need(isinstance(row, dict) and set(row) == fields and isinstance(row["receipt"], str)
                  and row["receipt"] in receipts and all(type(row[key]) is int and row[key] > 0 for key in ("row", "column"))
                  and all(isinstance(row[key], str) for key in ("section", "label", "population", "measure", "raw", "token_status"))
                  and all(row[key] is None or isinstance(row[key], str) for key in ("reason", "token_reason"))
                  and row["token_status"] in {status.value for status in ValueStatus}, "Invalid student raw audit record")
            current = (receipts[row["receipt"]], row["row"], row["column"])
            _need(current > last, "Duplicate or unordered student raw audit coordinates")
            last = current
            yield row


def verify_audit_coverage(raw: bytes, reports: tuple[ReportSelection, ...], grid_bytes: bytes) -> int:
    """Every report must carry each unbound local slot in its reviewed grid."""
    grid = json.loads(grid_bytes, object_pairs_hook=_unique_object)
    scope = reports[0].scope
    fields = {(f["section"], f["label"], f["population"], f["quantity"]) for f in semantic_fields(grid)}
    expected = {}
    for spec in grid["rows"]:
        section, label = spec["section"], spec["label"]
        for index, (population, quantity, role) in enumerate(grid["column_roles"][scope], 2):
            quantity = spec.get("quantity_overrides", {}).get(scope, {}).get(role, quantity)
            if role == "local" and (section, label, population, quantity) not in fields:
                expected[section, label, index] = population, quantity, spec["hold"]
    rows = audit_rows(raw, reports)
    for report in reports:
        seen = set()
        for _ in expected:
            row = next(rows, None)
            _need(row is not None and row["receipt"] == report.receipt, "Student audit omits a required report slot")
            assert row is not None
            slot = row["section"], row["label"], row["column"]
            _need(slot not in seen and expected.get(slot) == (row["population"], row["measure"], row["reason"]),
                  "Student audit has an invalid or repeated report slot")
            seen.add(slot)
    _need(next(rows, None) is None, "Student audit contains extra report slots")
    return len(reports) * len(expected)
