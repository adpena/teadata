"""Resume one exact 2025–26 TPRS campus region at a bounded broker pace."""

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
from teadata.dataset_metadata import DatasetMetadata, ResolvedDataset
from teadata.manifest import ContractError, canonical_json
from teadata.provenance import ProvenanceStore, _publish_once
from teadata.tprs_statewide_intake import CAMPUS_INTAKE_VERSION, REGIONS, ReportSelection, campus_roster
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


def _retained(store: ProvenanceStore, ids: set[str], preferred: Mapping[str, str]) -> tuple[
    dict[str, ReportSelection], dict[str, str]
]:
    candidates: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for path in (store.root / "records").glob("*.json"):
        record = store.read_record(path.stem)
        if record["kind"] != "acquisition" or record["method"] != "download":
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


def acquire_region(store: ProvenanceStore, region: str, roster_receipt: str, *, limit: int = 25,
                   interval: float = MIN_INTERVAL, intake: Path | None = None,
                   preferred: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Use verified retained reports first and persist pacing before each request."""
    if type(limit) is not int or not 0 <= limit <= 50 or not MIN_INTERVAL <= interval <= 3600:
        raise ValueError("Use zero to 50 requests per pass and an interval between eight seconds and one hour")
    ids = set(campus_roster(store, region, roster_receipt))
    choices = preferred or {}
    if not choices.keys() <= ids:
        raise ContractError("Preferred receipts include campuses outside this exact region")
    client = AcquisitionClient(store, opener=build_opener(NoRedirect()), timeout=60, attempts=1, max_bytes=MAX_REPORT_BYTES)
    with _broker_lock(store) as pace:
        reports, holds = _retained(store, ids, choices)
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
        if intake is not None and ids == reports.keys():
            size = sum(store.read_record(report.receipt)["artifact"]["size_bytes"] for report in reports.values())
            if size > MAX_SELECTION_BYTES:
                result["intake_hold"] = "Complete region exceeds the current bounded native materializer"
            else:
                result["intake_sha256"] = _save(intake, {"schema_version": CAMPUS_INTAKE_VERSION,
                    "dataset": DATASET, "edition": str(EDITION), "region": region, "roster_receipt": roster_receipt,
                    "reports": [asdict(reports[key]) for key in sorted(reports)]})
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--region", required=True, choices=REGIONS)
    parser.add_argument("--roster-receipt", required=True)
    parser.add_argument("--limit", type=int, default=25, help="Maximum requests, including zero for offline coverage")
    parser.add_argument("--interval", type=float, default=MIN_INTERVAL)
    parser.add_argument("--intake", type=Path, help="Seal a compatible intake only when the exact roster is complete")
    parser.add_argument("--prefer", action="append", default=[], metavar="CAMPUS=RECEIPT")
    args = parser.parse_args()
    try:
        preferred = dict(value.split("=", 1) for value in args.prefer)
        if len(preferred) != len(args.prefer):
            raise ValueError("Choose at most one receipt per campus")
        if not (args.store / "records").is_dir() or not (args.store / "objects").is_dir():
            raise ValueError("Select the existing retained publisher store")
        result = acquire_region(ProvenanceStore(args.store), args.region, args.roster_receipt, limit=args.limit,
                                interval=args.interval, intake=args.intake, preferred=preferred)
        print(json.dumps(result, sort_keys=True))
    except (ContractError, OSError, ValueError) as error:
        parser.exit(1, f"TPRS campus acquisition failed: {error}\n")


if __name__ == "__main__":
    main()
