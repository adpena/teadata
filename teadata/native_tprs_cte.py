"""Replay the exact retained 2025–26 TPRS district and direct Texas CTE release."""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timezone
from hashlib import sha256
from importlib.resources import files
import json
from typing import TYPE_CHECKING, Any

from .dataset_metadata import EditionKey
from .manifest import ContractError, canonical_json
from .tprs_partitions import CTE_DATASET
from .tprs_statewide_intake import roster_membership
from .tprs_students import (
    CteMembership,
    _CTE_MEMBERSHIP_GRID,
    _CTE_MEMBERSHIP_RESOURCE,
    _compare_student_workbook,
    read_student_report,
    reconcile_cte_membership,
)

if TYPE_CHECKING:
    from .provenance import ProvenanceStore
    from .release import DataRelease

REVIEWED_MANIFEST = "791199c0b64f36de8f29bd022afcedf80f97f097b34803e62c8121bd9f2bb21a"
_POPULATIONS = ("membership_cte", "membership_cte_grades_9_12")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _plan() -> dict[str, Any]:
    result = json.loads(files("teadata").joinpath("reference_cases/native_tprs_cte.json").read_bytes())
    _require(result["manifest_sha256"] == REVIEWED_MANIFEST, "CTE review points to a different manifest")
    return result


def reviewed_cte_roots() -> list[str]:
    plan = _plan()
    return [
        *(receipt for entry in plan["rosters"].values() for receipt in entry.values()),
        plan["state_parent"],
        *(entry["xlsx_receipt"] for entry in plan["xlsx_witnesses"]),
    ]


def _read_roster(plan: dict[str, Any], store: ProvenanceStore) -> dict[str, str]:
    records = store.verify_record_closure(
        [receipt for entry in plan["rosters"].values() for receipt in entry.values()]
    )
    waves: dict[str, dict[str, str]] = {}
    for wave in ("previous", "final"):
        responses = {}
        for region, entry in plan["rosters"].items():
            receipt = entry[wave]
            digest = records[receipt]["artifact"]["sha256"]
            _require(records[entry["final"]]["artifact"]["sha256"] ==
                     records[entry["previous"]]["artifact"]["sha256"],
                     f"CTE regional roster changed between acquisition waves for ESC {region}")
            responses[region] = {
                "receipt": receipt,
                "response": store.verify_artifact(digest).read_text(encoding="utf-8"),
            }
        waves[wave] = roster_membership(responses, records)
    _require(waves["previous"].keys() == waves["final"].keys(), "CTE roster waves cover different districts")
    parents = waves["final"]
    for identifier in plan["pilot_districts"]:
        parents[identifier] = waves["previous"][identifier]
    return {**parents, "S": plan["state_parent"]}


def _project_observation(prefix: str, observation: CteMembership) -> dict[str, Any]:
    count, rate = observation.count_cell, observation.rate_cell
    cells = [{"row": cell.row, "column": cell.column, "token": cell.raw} for cell in observation.denominator_cells]
    tokens = [cell["token"] for cell in cells]
    quantities = {
        "count": (observation.count, count.raw, "observed", {"row": count.row, "column": count.column}),
        "denominator": (observation.denominator, tokens[0] if len(tokens) == 1 else canonical_json(tokens),
                        "observed", cells),
        "percent": (observation.rate.numeric, rate.raw, observation.rate.status.value,
                    {"row": rate.row, "column": rate.column, "reason": observation.rate.reason}),
    }
    row: dict[str, Any] = {}
    for quantity, (value, raw, status, evidence) in quantities.items():
        field = f"{prefix}_{quantity}"
        row.update({field: value, f"tprs_raw_{field}": raw, f"tprs_status_{field}": status,
                    f"tprs_evidence_{field}": canonical_json(evidence)})
    return row


def _leaf_rows(release: DataRelease, store: ProvenanceStore, key: EditionKey) -> list[dict[str, Any]]:
    ref, = (ref for identifier in release.catalog.to_dict()["artifacts"]
            if (ref := release.catalog.artifact_ref(identifier)).edition == key)
    record = store.verify_record_closure([ref.processing_id])[ref.processing_id]
    metadata = release.catalog.metadata(key.dataset, edition=key)
    schema, = [entry for entry in metadata.references if entry.reference_id == "columns"]
    rows = release.table(key).to_arrow().select(json.loads(ref.contract_json)["columns"]).to_pylist()
    receipts = [row["tprs_report_receipt"] for row in rows]
    _require(receipts == [entry["record_id"] for entry in release.catalog.acquisitions(key.dataset, edition=key)]
             and record["inputs"] == receipts + list(schema.provenance_ids)
             and record["outputs"]["leaf"]["sha256"] == ref.sha256
             and record["outputs"]["leaf"]["size_bytes"] == ref.size_bytes,
             f"CTE {key.scope} processing inputs or output differ from the retained edition")
    return rows


def qualify_tprs_cte(release: DataRelease, processing_store: ProvenanceStore,
                    publisher_store: ProvenanceStore) -> dict[str, Any]:
    """Verify frozen roster waves, every raw report, semantic fields and independent XLSX witnesses."""
    _require(release.manifest_sha256 == REVIEWED_MANIFEST, "CTE release differs from the reviewed native manifest")
    plan = _plan()
    grid_bytes = files("teadata").joinpath(_CTE_MEMBERSHIP_RESOURCE).read_bytes()
    _require(sha256(grid_bytes).hexdigest() == _CTE_MEMBERSHIP_GRID, "CTE report grid differs")
    grid = json.loads(grid_bytes)
    parents = _read_roster(plan, publisher_store)
    leaves = tuple(EditionKey.from_dict({name: row[name] for name in EditionKey.__dataclass_fields__})
                   for row in release.catalog.editions(dataset=CTE_DATASET, include_metadata=False))
    _require(len(leaves) == 2 and {(key.edition, key.scope) for key in leaves} == {("2026", "state"), ("2026", "district")},
             "CTE must have exactly the 2026 district and direct Texas editions")
    selection: dict[str, tuple[str, dict[str, Any]]] = {}
    for key in leaves:
        for row in _leaf_rows(release, processing_store, key):
            identifier = row["district_number"] if key.scope == "district" else "S"
            _require(identifier not in selection, f"CTE source identity repeats: {identifier}")
            selection[identifier] = key.scope, row
    ledger = sorted((identifier, row["tprs_report_receipt"]) for identifier, (_, row) in selection.items())
    _require(selection.keys() == parents.keys() and sha256(canonical_json(ledger).encode()).hexdigest() ==
             plan["report_ledger_sha256"], "CTE report ledger differs from the frozen statewide census")

    totals: Counter[tuple[str, str]] = Counter()
    state: Counter[tuple[str, str]] = Counter()
    statuses: Counter[str] = Counter()
    witnesses = {entry["identifier"]: entry for entry in plan["xlsx_witnesses"]}
    xlsx_cells = 0
    for identifier, (scope, row) in selection.items():
        receipt, digest = row["tprs_report_receipt"], row["tprs_report_sha256"]
        record = processing_store.read_record(receipt)
        retrieved_at = datetime.fromisoformat(record["downloaded_at"]).astimezone(timezone.utc)
        _require(record["source_metadata"]["provenance_ids"] == [parents[identifier]]
                 and retrieved_at.date() == date(2026, 11, 21), f"CTE download lineage changed for {identifier}")
        report = read_student_report(processing_store, receipt, scope=scope, identifier=identifier,
                                     expected_sha256=digest, contract=grid)
        expected = {"tprs_report_receipt": receipt, "tprs_report_sha256": report.sha256,
                    "tprs_downloaded_at": retrieved_at}
        if scope == "district":
            expected["district_number"] = identifier
        for prefix, observation in zip(_POPULATIONS, reconcile_cte_membership(report), strict=True):
            scope_totals = state if scope == "state" else totals
            scope_totals[prefix, "count"] += observation.count
            scope_totals[prefix, "denominator"] += observation.denominator
            statuses[observation.rate.status.value] += 1
            expected.update(_project_observation(prefix, observation))
        _require(row == expected, f"CTE native values, masks or source coordinates differ for {identifier}")
        witness = witnesses.get(identifier)
        if witness is not None:
            _compare_student_workbook(publisher_store, report, witness, contract=grid)
            _require(len(report.cells) == witness["cells"], f"CTE workbook witness coverage differs for {identifier}")
            xlsx_cells += len(report.cells)
    _require(totals == state, "CTE district numerators or denominators do not reconcile with direct Texas")
    return {"status": "PASS_REVIEWED_NATIVE", "dataset": CTE_DATASET, "districts": len(selection) - 1, "state_reports": 1,
            "workbook_witness_cells": xlsx_cells, "rate_statuses": dict(statuses)}
