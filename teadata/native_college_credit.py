"""Offline source qualification for the reviewed PEIMS College Credit family.

This is a verification plan, not another materializer or data access API. All
expectations come from retained publisher CSVs and separately requested HTML.
"""

from __future__ import annotations

from collections import Counter
import csv
from decimal import Decimal
from hashlib import sha256
from importlib.resources import files
from io import BytesIO, StringIO
import json
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from .dataset_metadata import EditionKey
from .manifest import ContractError
from .native_witnesses import _BROKER, _Rows

if TYPE_CHECKING:
    from .provenance import ProvenanceStore
    from .release import DataRelease


_DATASET = "peims_college_credit"
_FACTS = ("students_earning_college_credit_count", "college_credit_hours", "average_credit_hours_per_student")
_TITLES = {"state": "Statewide Totals", "region": "Statewide Region Totals", "district": "Statewide District Totals"}
_MAX_SOURCE = 8 * 1024**2


class _PublisherRows(_Rows):
    """Keep literal internal label spacing; normalize header layout separately."""

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self.cell is not None and self.row is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        else:
            super().handle_endtag(tag)


def reviewed_college_credit(manifest_sha256: str) -> dict[str, Any] | None:
    """Select only the exact retained family, without opening any source leaves."""
    plan = json.loads(files("teadata").joinpath("reference_cases/native_college_credit.json").read_text())
    return plan if manifest_sha256 == plan["manifest_sha256"] else None


def validate_college_credit_identity(release: DataRelease, plan: dict[str, Any]) -> None:
    """Reject changed editions, metadata and leaves before qualification/reuse."""
    catalog = release.catalog.to_dict()
    if (release.manifest_sha256 != plan["manifest_sha256"] or release.build_id != plan["build_id"]
            or catalog["schema_version"] != plan["catalog_schema_version"]
            or catalog["editions"].keys() != plan["editions"].keys()
            or catalog["artifacts"] != {case["artifact_id"]: identifier for identifier, case in plan["editions"].items()}):
        raise ContractError("College Credit manifest, build, schema or edition set differs from its reviewed identity")
    for identifier, case in plan["editions"].items():
        entry = catalog["editions"][identifier]
        ref = release.catalog.artifact_ref(case["artifact_id"])
        acquisitions = release.catalog.acquisitions(_DATASET, edition=ref.edition)
        expected_source = case["source"]
        if (entry != {"key": case["key"], "metadata": case["metadata"]}
                or ref.edition.to_dict() != case["key"] or ref.identity != case["artifact_id"]
                or ref.sha256 != case["leaf_sha256"] or ref.size_bytes != case["leaf_size_bytes"]
                or ref.processing_id != case["processing_id"] or ref.contract["rows"] != case["rows"]
                or len(acquisitions) != 1
                or acquisitions[0]["record_id"] != expected_source["original_acquisition_id"]
                or acquisitions[0]["artifact"] != expected_source["artifact"]):
            raise ContractError(f"College Credit {ref.edition.edition}/{ref.edition.scope} leaf, metadata or processing identity differs")
        period = release.catalog.metadata(_DATASET, edition=ref.edition).period
        if period is None or period.to_dict() != expected_source["measurement_period"]:
            raise ContractError(f"College Credit {ref.edition.edition}/{ref.edition.scope} qualified measurement period differs")


def _report_bytes(store: ProvenanceStore, source: dict[str, Any], *, year: int, scope: str,
                  format: str, receipt: str) -> bytes:
    """Validate the actual request/receipt, not a filename containing a year."""
    record = store.read_record(receipt)
    controls = [["_service", "marykay"], ["_program", "adhoc.download_college_credit_report.sas"],
                ["RptClass", "CollegeCred"], ["_debug", "0"], ["school_year", str(year)],
                ["report", scope.title()], ["format", format]]
    url = f"{_BROKER}?{urlencode([(pair[0], pair[1]) for pair in controls])}"
    parameters, metadata = record["parameters"], record["source_metadata"]
    content_type = record["response_headers"].get("content-type", "").split(";", 1)[0].lower()
    allowed = {"application/csv", "text/csv"} if format == "csv" else {"text/html"}
    downloaded_at = source["downloaded_at"] if format == "csv" else source["retrieved"]
    if (record["kind"] != "acquisition" or record["method"] != "download" or record["section"] != "reference"
            or record["dataset"] != f"reference:{_DATASET}:{scope}:{format}"
            or (record["requested_year"], record["resolved_year"]) != (year, year)
            or record["requested_url"] != url or record["resolved_url"] != url
            or record["artifact"] != source["artifact"] or not 0 < source["artifact"]["size_bytes"] <= _MAX_SOURCE
            or record["downloaded_at"] != downloaded_at or not downloaded_at
            or metadata["publisher"] != "Texas Education Agency" or metadata["version"] != str(year)
            or parameters.get("http_method") != "GET" or parameters.get("http_status") != 200
            or parameters.get("form_url") != _BROKER or parameters.get("form_content_type") is not None
            or parameters.get("form_controls") != controls or content_type not in allowed):
        raise ContractError(f"College Credit {year}/{scope}/{format} acquisition request differs from its reviewed identity")
    # Both raw formats were acquired before the source period was qualified.
    # Preserve that original unknown; the exact qualified edition period and
    # the request/report year are checked independently, never rewritten here.
    if metadata["period"] != {"kind": "unknown"}:
        raise ContractError(f"College Credit {year}/{scope}/{format} original acquisition period differs")
    path = store.verify_artifact(source["artifact"]["sha256"])
    with path.open("rb") as stream:
        data = stream.read(_MAX_SOURCE + 1)
    if len(data) != source["artifact"]["size_bytes"] or sha256(data).hexdigest() != source["artifact"]["sha256"]:
        raise ContractError(f"College Credit {year}/{scope}/{format} retained bytes differ from the receipt")
    return data


def _fields(year: int) -> tuple[str, ...]:
    prefix = {2022: "cte_excluding_participants", 2026: "cte_excluding_explorers"}.get(year, "cte_all_program_groups")
    return (*(f"{prefix}_{field}" for field in _FACTS), *_FACTS)


def _csv_rows(data: bytes, source: dict[str, Any], year: int, scope: str) -> list[tuple[int, list[str]]]:
    lines = data.splitlines(keepends=True)
    first, last, heading = source["first_data_record"], source["last_data_record"], source["header_record"]
    publisher = list(csv.reader(StringIO(data.decode("cp1252")), strict=True))
    if (len(lines) != len(publisher) or first != heading + 1 or last >= len(lines)
            or sha256(b"".join(lines[:heading - 1])).hexdigest() != source["preamble_sha256"]
            or sha256(b"".join(lines[last:])).hexdigest() != source["footer_sha256"]
            or publisher[heading - 1] != source["header"]
            or [row for row in publisher[:heading - 1] if row] != [
                ["TEXAS EDUCATION AGENCY"], [f"College Credit Reports - {_TITLES[scope]}"],
                [f"PEIMS data for {year - 1}-{year}"]]):
        raise ContractError(f"College Credit {year}/{scope} publisher header, framing or measurement year differs")
    window = [(number, publisher[number - 1]) for number in range(first, last + 1)]
    if any(len(row) != len(source["header"]) for _, row in window):
        raise ContractError(f"College Credit {year}/{scope} publisher record width differs")
    rows = [(number, row) for number, row in window if any(row)]
    if len(rows) != source["source_rows"]:
        raise ContractError(f"College Credit {year}/{scope} publisher row count differs")
    return rows


def _raw_parity(release: DataRelease, store: ProvenanceStore, case: dict[str, Any]) -> dict[str, Any]:
    source, key = case["source"], EditionKey.from_dict(case["key"])
    year, scope = int(key.edition), key.scope
    receipt = source["original_acquisition_id"]
    graph = store.verify_record_closure([case["processing_id"]])
    processing = graph[case["processing_id"]]
    ref = release.catalog.artifact_ref(case["artifact_id"])
    acquisition, = release.catalog.acquisitions(_DATASET, edition=key)
    if (graph[receipt] != {name: value for name, value in acquisition.items() if name != "record_id"}
            or processing["processor"] != "teadata.peims_college_credit"
            or processing["dataset"] != _DATASET or processing["inputs"][0] != receipt
            or processing["parameters"]["leaf_contract_sha256"] != ref.identity
            or processing["parameters"]["recipe"]["operation"] != "reviewed_peims_college_credit_csv_projection"
            or processing["parameters"]["recipe"]["encoding"] != "cp1252"):
        raise ContractError(f"College Credit {year}/{scope} processing or acquisition closure differs")
    data = _report_bytes(store, source, year=year, scope=scope, format="csv", receipt=receipt)
    numbered = _csv_rows(data, source, year, scope)
    rows = [row for _, row in numbered]
    table = release.table(key)
    output = BytesIO()
    table.to_csv(output, restore_original_headers=True)
    actual = list(csv.reader(StringIO(output.getvalue().decode("utf-8")), strict=True))
    projected = table.to_arrow()
    if actual != [source["header"], *rows] or projected["peims_source_record"].to_pylist() != [n for n, _ in numbered]:
        raise ContractError(f"College Credit {year}/{scope} literal exports or source row locations differ")
    offset = {"state": 0, "region": 2, "district": 5}[scope]
    service_keys = ["Total" if (row[offset] == "Total" and not row[offset + 1])
                    or (not row[offset] and row[offset + 1] == "Total") else row[offset] for row in rows]
    if projected["college_credit_service_key"].to_pylist() != service_keys:
        raise ContractError(f"College Credit {year}/{scope} service row keys differ from the publisher")
    if scope == "region":
        # Keep the publisher's two-digit ESC IDs; do not infer them from names.
        if set(projected["region_number"].to_pylist()) != {f"{n:02d}" for n in range(1, 21)}:
            raise ContractError(f"College Credit {year}/region does not preserve all20 ESC IDs")
    positions = [i for i, label in enumerate(source["header"]) if label.startswith(("Total ", "Average "))]
    masked = 0
    for position, name in zip(positions, _fields(year), strict=True):
        tokens = [row[position] for row in rows]
        convert = Decimal if name.endswith("average_credit_hours_per_student") else int
        masks = [token in {"-999", "-999.0"} for token in tokens]
        values = [None if mask else convert(token) for token, mask in zip(tokens, masks, strict=True)]
        if (projected[name].to_pylist() != values or projected[f"peims_raw_{name}"].to_pylist() != tokens
                or projected[f"peims_status_{name}"].to_pylist() != ["suppressed" if mask else "observed" for mask in masks]):
            raise ContractError(f"College Credit {year}/{scope} typed/raw/status values differ for {name}")
        masked += sum(masks)
    return {"edition_id": key.identity, "year": year, "scope": scope, "rows": len(rows),
            "source_cells": len(rows) * len(source["header"]), "typed_facts": len(rows) * 6,
            "masked_cells": masked, "skipped_separator_records": source["last_data_record"] - source["first_data_record"] + 1 - len(rows),
            "source_receipt": receipt, "source_sha256": source["artifact"]["sha256"]}


def _html_parity(store: ProvenanceStore, witness: dict[str, Any], source: dict[str, Any], csv_data: bytes) -> dict[str, Any]:
    year = witness["edition"]
    content = _report_bytes(store, witness, year=year, scope="state", format="html", receipt=witness["receipt"]).decode("cp1252")
    if (f"{year - 1}-{year}" not in content or "College Credit Reports - Statewide Totals" not in content
            or "FERPA" not in content):
        raise ContractError(f"College Credit {year} independent HTML title, year or masking legend differs")
    # TEA wraps column labels with <br/>. A rendered line break is whitespace,
    # not concatenation ("CTEStudents"); leave all words/numbers untouched.
    parsed = _PublisherRows(re.sub(r"<br\s*/?>", " ", content, flags=re.IGNORECASE)).rows
    heading = [i for i, row in enumerate(parsed) if [" ".join(cell.split()) for cell in row] == source["header"]]
    if len(heading) != 1:
        raise ContractError(f"College Credit {year} independent HTML publisher header differs")
    published = [row for row in parsed[heading[0] + 1:] if len(row) == len(source["header"]) and any(row)]
    expected = [row for _, row in _csv_rows(csv_data, source, year, "state")]
    if len(published) != len(expected) or len({row[0] for row in published}) != len(published):
        raise ContractError(f"College Credit {year} independent HTML service set differs")
    for left, right in zip(expected, published, strict=True):
        if left[:2] != right[:2] or left[5] or right[5]:
            raise ContractError(f"College Credit {year} independent HTML service/label/separator differs")
        for i in (2, 3, 4, 6, 7, 8):
            if ((left[i] in {"-999", "-999.0"} and right[i] != "N/A")
                    or (left[i] not in {"-999", "-999.0"} and Decimal(left[i]) != Decimal(right[i].replace(",", "")))):
                raise ContractError(f"College Credit {year} independent HTML fact differs for {left[0]} column{i}")
    return {"year": year, "receipt": witness["receipt"], "sha256": witness["artifact"]["sha256"],
            "rows": len(expected), "matched_fact_cells": len(expected) * 6}


def qualify_college_credit(release: DataRelease, parity_store: ProvenanceStore,
                           witness_store: ProvenanceStore, plan: dict[str, Any]) -> dict[str, Any]:
    """Replay all15 CSV editions and all five independent Texas HTML witnesses."""
    try:
        validate_college_credit_identity(release, plan)
        release.verify()
        results = [_raw_parity(release, parity_store, case) for case in plan["editions"].values()]
        witnesses = plan["publisher_witnesses"]
        if len(witnesses) != 5 or {item["edition"] for item in witnesses} != set(range(2022, 2027)):
            raise ContractError("College Credit review requires all five distinct year-qualified HTML witnesses")
        witness_store.verify_record_closure([item["receipt"] for item in witnesses])
        reports = []
        for witness in witnesses:
            case, = (case for case in plan["editions"].values()
                     if case["key"]["scope"] == "state" and case["key"]["edition"] == str(witness["edition"]))
            source = case["source"]
            data = _report_bytes(parity_store, source, year=witness["edition"], scope="state", format="csv",
                                 receipt=source["original_acquisition_id"])
            reports.append(_html_parity(witness_store, witness, source, data))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        raise ContractError(f"College Credit retained source qualification failed: {error}") from error
    coverage: Counter[str] = Counter()
    for result in results:
        coverage.update({name: result[name] for name in ("rows", "source_cells", "typed_facts", "masked_cells", "skipped_separator_records")})
    return {"status": "PASS_REVIEWED_NATIVE", "dataset": _DATASET, "manifest_sha256": release.manifest_sha256,
            "build_id": release.build_id, "structural_integrity": True,
            "parity": {"status": "PASS_RAW_PARITY", "editions": results, "coverage": {
                **coverage, "requested_editions": len(plan["editions"]), "checked_editions": len(results),
                "uncovered_editions": 0, "failed_editions": 0}},
            "publisher_witnesses": {"status": "PASS_PUBLISHER_WITNESSES", "reports": reports},
            "limits": ["Course-level counts may count a student repeatedly; not unique students or a readiness rate.",
                       "CTE populations differ by year; all-student longitudinal comparability remains unassessed.",
                       "No campus report; earlier years and their unresolved display discrepancies are outside this review.",
                       "Direct Texas totals are independent publisher aggregates; suppressed values are not reconstructed."]}
