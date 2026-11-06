"""Resume a 2025–26 TPRS campus region or inventory retained statewide reports."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import sqlite3
from tempfile import NamedTemporaryFile
import time
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qsl, urlsplit
from urllib.request import build_opener

from teadata.acquisition import AcquisitionClient, AcquisitionError, NoRedirect, acquire_form
from teadata.column_schema import _unique_object
from teadata.dataset_metadata import DatasetMetadata, ResolvedDataset
from teadata.manifest import ContractError, canonical_json
from teadata.provenance import ProvenanceStore, _publish_once
from teadata.tprs_statewide_intake import (
    CAMPUS_INTAKE_VERSION, CAMPUS_PARTITION_VERSION, MAX_STATEWIDE_HTML_BYTES, REGIONS, ReportSelection,
    admit_campus_cohort, campus_partition_ids, campus_report_has_roster, campus_roster,
)
from teadata.tprs_students import BROKER, DATASET, EDITION, MAX_REPORT_BYTES, MAX_SELECTION_BYTES, read_student_report

MIN_INTERVAL = 8.0
RATE_LIMIT_COOLDOWN = 300.0
RECIPE = "tprs-student-campus-2026"


@contextmanager
def _broker_lock(store: ProvenanceStore) -> Generator[Path]:
    """Hold a process-owned, portable lock independent of the durable pace record."""
    lock = sqlite3.connect(store.root / "tprs-campus-broker.lock.sqlite", timeout=0)
    try:
        try:
            lock.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as error:
            raise ContractError("Cannot lock this store's TPRS campus broker") from error
        yield store.root / "tprs-campus-broker-pace.json"
    finally:
        lock.close()


def _save(path: Path, document: dict, *, replace: bool = False) -> str:
    raw = (canonical_json(document) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            if replace:
                temporary.replace(path)
            else:
                _publish_once(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return sha256(raw).hexdigest()


def _cooldown(retry_after: str | None, now: float) -> float:
    if not retry_after:
        return now + RATE_LIMIT_COOLDOWN
    try:
        if retry_after.strip().isascii() and retry_after.strip().isdigit():
            advertised = now + float(retry_after)
        else:
            deadline = parsedate_to_datetime(retry_after)
            if deadline.tzinfo is None:
                raise ValueError("Retry-After date must have a timezone")
            advertised = deadline.timestamp()
        if not isfinite(advertised):
            raise ValueError("Invalid Retry-After value")
        datetime.fromtimestamp(advertised, timezone.utc)
    except (OSError, ValueError, TypeError, OverflowError) as error:
        raise ContractError("Invalid Retry-After; campus acquisition requires review") from error
    return max(now + RATE_LIMIT_COOLDOWN, advertised)


def _controls(identifier: str) -> dict[str, str]:
    return {"_debug": "0", "_program": "perfrept.perfmast.sas", "_service": "marykay",
            "ccyy": str(EDITION), "id": identifier, "lev": "C", "prgopt": "reports/tapr/student.sas"}


def _retained(store: ProvenanceStore, ids: set[str], preferred: Mapping[str, str], *,
              roster_receipt: str | None = None) -> tuple[
    dict[str, ReportSelection], dict[str, str]
]:
    candidates: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for path in (store.root / "records").glob("*.json"):
        record = store.read_record(path.stem)
        if record["kind"] != "acquisition" or record["method"] != "download":
            continue
        if roster_receipt is not None and not campus_report_has_roster(record, roster_receipt):
            continue
        parsed = urlsplit(record["requested_url"])
        pairs = parse_qsl(parsed.query, keep_blank_values=True)
        controls = dict(pairs)
        identifier = controls.get("id")
        if (identifier in ids and parsed._replace(query="").geturl() == BROKER
                and len(pairs) == len(controls) and controls == _controls(identifier)):
            candidates[identifier].append((record["downloaded_at"], path.stem, record["artifact"]["sha256"]))
    selected, holds = {}, {}
    for identifier in sorted(candidates.keys() | preferred.keys()):
        valid: dict[str, ReportSelection] = {}
        for _, receipt, digest in sorted(candidates[identifier], reverse=True):
            if (identifier in preferred and preferred[identifier] != receipt) or digest in valid:
                continue
            try:
                read_student_report(store, receipt, scope="campus", identifier=identifier, expected_sha256=digest)
            except (ContractError, OSError, ValueError) as error:
                holds[identifier] = f"{receipt}: {error}"
            else:
                valid[digest] = ReportSelection("campus", identifier, receipt, digest)
        if identifier in preferred and not valid:
            raise ContractError(f"Preferred campus receipt is not admissible: {identifier}")
        if len(valid) == 1:
            selected[identifier], = valid.values()
            holds.pop(identifier, None)
        elif len(valid) > 1:
            holds[identifier] = "Multiple validated publisher bodies; explicitly select a receipt"
    return selected, holds


def _inventory_receipts(path: Path) -> dict[str, str]:
    with path.open("rb") as stream:
        limit = 2 * 1024**2
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ContractError("Campus roster inventory exceeds its byte budget")
    try:
        inventory = json.loads(raw, object_pairs_hook=_unique_object)
    except ValueError as error:
        raise ContractError("Campus roster inventory must be valid JSON with unique keys") from error
    if not isinstance(inventory, dict) or inventory.get("edition") != str(EDITION):
        raise ContractError("Campus roster inventory must identify the 2026 edition")
    regions = inventory.get("regions")
    if not isinstance(regions, dict) or set(regions) != set(REGIONS):
        raise ContractError("Campus roster inventory must include each TPRS region exactly once")
    receipts = {}
    for region, entry in regions.items():
        receipt = entry.get("receipt") if isinstance(entry, dict) else None
        if not isinstance(receipt, str):
            raise ContractError(f"Missing roster receipt for region {region}")
        receipts[region] = receipt
    return receipts


def retained_inventory(store: ProvenanceStore, roster_receipts: Mapping[str, str], *,
                       preferred: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Inventory exact regional cohorts against one snapshot of retained reports."""
    if set(roster_receipts) != set(REGIONS):
        raise ContractError("Campus roster inventory must include each TPRS region exactly once")
    rosters: dict[str, set[str]] = {}
    eligible: set[str] = set()
    for region in REGIONS:
        ids = set(campus_roster(store, region, roster_receipts[region]))
        if eligible & ids:
            raise ContractError(f"Campus assigned to multiple TPRS regions through region {region}")
        rosters[region] = ids
        eligible.update(ids)
    choices = preferred or {}
    if not choices.keys() <= eligible:
        raise ContractError("Preferred receipts include campuses outside the retained statewide roster")
    with _broker_lock(store):
        reports, holds = _retained(store, eligible, choices)
        recorded_at = datetime.now(timezone.utc).isoformat()
    return {
        "status": "RETAINED_PUBLISHER_REPORT_INVENTORY",
        "edition": EDITION,
        "recorded_at": recorded_at,
        "eligible": len(eligible),
        "valid_reports": len(reports),
        "remaining": sorted(eligible - reports.keys()),
        "reports": {key: asdict(report) for key, report in reports.items()},
        "retained_holds": holds,
        "regions": {
            region: {"roster": len(ids), "admitted": len(ids & reports.keys()), "roster_receipt": roster_receipts[region]}
            for region, ids in rosters.items()
        },
    }


def acquire_region(store: ProvenanceStore, region: str, roster_receipt: str, *, limit: int = 25,
                   interval: float = MIN_INTERVAL, intake: Path | None = None,
                   preferred: Mapping[str, str] | None = None, partition: int | None = None) -> dict[str, Any]:
    """Use verified retained reports first and persist pacing before each request."""
    if type(limit) is not int or not 0 <= limit <= 50 or not MIN_INTERVAL <= interval <= 3600:
        raise ValueError("Use zero to 50 requests per pass and an interval between eight seconds and one hour")
    roster = campus_roster(store, region, roster_receipt)
    ids = set(roster if partition is None else campus_partition_ids(roster, partition))
    choices = preferred or {}
    if not choices.keys() <= ids:
        scope = "region" if partition is None else "campus scope"
        raise ContractError(f"Preferred receipts include campuses outside this exact {scope}")
    client = AcquisitionClient(store, opener=build_opener(NoRedirect()), timeout=60, attempts=1, max_bytes=MAX_REPORT_BYTES)
    with _broker_lock(store) as pace:
        if partition is None:
            reports, holds = _retained(store, ids, choices)
        else:
            reports, holds = _retained(store, ids, choices, roster_receipt=roster_receipt)
        reused = len(reports)
        state = json.loads(pace.read_bytes()) if pace.exists() else {"not_before": 0}
        not_before = state["not_before"]
        if type(not_before) not in {int, float} or not isfinite(not_before) or not_before < 0:
            raise ContractError("Invalid retained TPRS broker pace")
        requests = 0
        stop = None
        for identifier in sorted(ids - reports.keys() - holds.keys())[:limit]:
            now = time.time()
            if not_before > now and requests == 0:
                stop = "cooldown"
                break
            if not_before > now:
                time.sleep(not_before - now)
            not_before = time.time() + interval
            _save(pace, {"not_before": not_before}, replace=True)
            source = ResolvedDataset(f"reference:{RECIPE}:{identifier}", "reference", EDITION, EDITION, "",
                DatasetMetadata(title=f"TPRS campus {identifier} Student Information", version=str(EDITION),
                                publisher="Texas Education Agency", source_url=BROKER, provenance_ids=(roster_receipt,)))
            requests += 1
            try:
                acquired = acquire_form(client, BROKER, source, filename=f"2026-C-{identifier}-STUD.html",
                                        method="GET", fields=tuple(_controls(identifier).items()))
            except HTTPError as error:
                with error:
                    stop = f"HTTP {error.code}"
                    if error.code in {429, 503}:
                        now = time.time()
                        # Persist a conservative stop before inspecting an untrusted header.
                        _save(pace, {"not_before": now + RATE_LIMIT_COOLDOWN}, replace=True)
                        not_before = _cooldown(error.headers.get("Retry-After") if error.headers else None, now)
                        _save(pace, {"not_before": not_before}, replace=True)
                break
            except (AcquisitionError, OSError) as error:
                stop = type(error).__name__
                break
            digest = store.read_record(acquired.record_id)["artifact"]["sha256"]
            try:
                read_student_report(store, acquired.record_id, scope="campus", identifier=identifier, expected_sha256=digest)
            except (ContractError, OSError, ValueError) as error:
                holds[identifier] = f"{acquired.record_id}: {error}"
                stop = "report_hold"
                break
            reports[identifier] = ReportSelection("campus", identifier, acquired.record_id, digest)
        result = {"region": region, "roster_receipt": roster_receipt, "eligible": len(ids), "reused": reused,
                  "admitted": len(reports), "requested": requests, "holds": holds, "stop": stop,
                  "remaining": sorted(ids - reports.keys()),
                  "next_request_at": datetime.fromtimestamp(not_before, timezone.utc).isoformat()}
        if partition is not None:
            result.update(partition=partition, region_roster_size=len(roster))
        if intake is not None and ids == reports.keys():
            size = sum(store.read_record(report.receipt)["artifact"]["size_bytes"] for report in reports.values())
            budget = MAX_SELECTION_BYTES if partition is None else MAX_STATEWIDE_HTML_BYTES
            if size > budget:
                scope = "region" if partition is None else "partition"
                result["intake_hold"] = f"Complete {scope} exceeds the current bounded native materializer"
                return result
            document = {"dataset": DATASET, "edition": str(EDITION),
                        "reports": [asdict(reports[key]) for key in sorted(reports)]}
            if partition is None:
                document.update(schema_version=CAMPUS_INTAKE_VERSION, region=region, roster_receipt=roster_receipt)
            else:
                document.update(schema_version=CAMPUS_PARTITION_VERSION, partition=partition, rosters={region: roster_receipt})
                try:
                    admit_campus_cohort(store, document)
                except (ContractError, OSError, ValueError) as error:
                    result["intake_hold"] = str(error)
                    return result
            result["intake_sha256"] = _save(intake, document)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--region", choices=REGIONS)
    mode.add_argument("--inventory", type=Path, help="Read the 20-region roster receipt inventory and report offline coverage")
    parser.add_argument("--roster-receipt")
    parser.add_argument("--partition", type=int, help="Select a zero-based 512-campus slice of the pinned region")
    parser.add_argument("--limit", type=int, help="Maximum regional requests, including zero for offline coverage")
    parser.add_argument("--interval", type=float)
    parser.add_argument("--intake", type=Path, help="Seal a compatible intake only when the exact roster is complete")
    parser.add_argument("--prefer", action="append", default=[], metavar="CAMPUS=RECEIPT")
    args = parser.parse_args()
    try:
        preferred = dict(value.split("=", 1) for value in args.prefer)
        if len(preferred) != len(args.prefer):
            raise ValueError("Choose at most one receipt per campus")
        if not (args.store / "records").is_dir() or not (args.store / "objects").is_dir():
            raise ValueError("Select the existing retained publisher store")
        store = ProvenanceStore(args.store)
        if args.inventory is not None:
            if any(value is not None for value in (args.roster_receipt, args.limit, args.interval, args.intake, args.partition)):
                raise ValueError("--inventory does not take regional acquisition options")
            result = retained_inventory(store, _inventory_receipts(args.inventory), preferred=preferred)
        else:
            if args.roster_receipt is None:
                parser.error("--region requires --roster-receipt")
            result = acquire_region(store, args.region, args.roster_receipt, intake=args.intake, preferred=preferred,
                                    limit=25 if args.limit is None else args.limit,
                                    interval=MIN_INTERVAL if args.interval is None else args.interval, partition=args.partition)
        print(json.dumps(result, sort_keys=True))
    except (ContractError, OSError, ValueError) as error:
        parser.exit(1, f"TPRS campus acquisition failed: {error}\n")


if __name__ == "__main__":
    main()
