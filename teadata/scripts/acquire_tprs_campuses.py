"""Resume a 2025–26 TPRS campus region or inventory retained statewide reports."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from collections.abc import Generator, Mapping
from concurrent.futures import FIRST_COMPLETED, CancelledError, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import re
import sqlite3
from tempfile import NamedTemporaryFile
from threading import Event, Lock
import time
from time import monotonic
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qsl, urlsplit
from urllib.request import build_opener

from teadata.acquisition import (
    AcquiredSource, AcquisitionClient, AcquisitionError, NoRedirect, acquire_form, is_transient_transport_failure,
)
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
PARALLEL_DRAIN_RESERVATION = 300.0
MAX_ATTEMPTS = 3
MAX_JOBS = 2
_THROTTLED = frozenset({429, 503})
RECIPE = "tprs-student-campus-2026"
_WEEKDAY = r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)"
_LONG_WEEKDAY = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
_CLOCK = r"[0-9]{2}:[0-9]{2}:[0-9]{2}"
_HTTP_DATE = re.compile(
    rf"(?:{_WEEKDAY}, [0-9]{{2}} {_MONTH} [0-9]{{4}} {_CLOCK} GMT|"
    rf"{_LONG_WEEKDAY}, [0-9]{{2}}-{_MONTH}-[0-9]{{2}} {_CLOCK} GMT|"
    rf"{_WEEKDAY} {_MONTH} (?: [0-9]|[0-9]{{2}}) {_CLOCK} [0-9]{{4}})"
)


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
                if os.name != "nt":
                    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
            else:
                _publish_once(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return sha256(raw).hexdigest()


def _cooldown(retry_after: str | None, now: float, *, minimum_wait: float = RATE_LIMIT_COOLDOWN) -> float:
    if retry_after is None:
        return now + minimum_wait
    try:
        if retry_after.strip().isascii() and retry_after.strip().isdigit():
            advertised = now + float(retry_after)
        else:
            value = retry_after.strip()
            if _HTTP_DATE.fullmatch(value) is None:
                raise ValueError("Retry-After requires an HTTP date or delay in seconds")
            deadline = parsedate_to_datetime(value)
            if deadline.tzinfo is None:  # The HTTP asctime form is UTC.
                deadline = deadline.replace(tzinfo=timezone.utc)
            if "-" in value:
                today = datetime.fromtimestamp(now, timezone.utc)
                try:
                    cutoff = today.replace(year=today.year + 50)
                except ValueError:  # February 29 in a non-leap target year.
                    cutoff = today.replace(year=today.year + 50, day=28)
                year = (cutoff.year // 100) * 100 + deadline.year % 100
                if (year, *deadline.timetuple()[1:6]) > cutoff.timetuple()[:6]:
                    year -= 100
                deadline = deadline.replace(year=year)
            advertised = deadline.timestamp()
        if not isfinite(advertised):
            raise ValueError("Invalid Retry-After value")
        datetime.fromtimestamp(advertised, timezone.utc)
    except (OSError, ValueError, TypeError, OverflowError) as error:
        raise ContractError("Invalid Retry-After; campus acquisition requires review") from error
    return max(now + minimum_wait, advertised)


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


def statewide_rosters(store: ProvenanceStore, roster_receipts: Mapping[str, str]) -> dict[str, set[str]]:
    """Read and reconcile the exact publisher campus roster for each region."""
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
    return rosters


def retained_inventory(store: ProvenanceStore, roster_receipts: Mapping[str, str], *,
                       preferred: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Inventory exact regional cohorts against one snapshot of retained reports."""
    rosters = statewide_rosters(store, roster_receipts)
    eligible = set().union(*rosters.values())
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


@dataclass(frozen=True)
class _RetryDecision:
    status: str
    retryable: bool
    deadline: float
    wait_source: str | None = None
    retry_after: str | None = None
    rejected_headers: tuple[str, ...] = ()


def _retry_policy(error: Exception, not_before: float, fallback_wait: float) -> _RetryDecision:
    retryable = is_transient_transport_failure(error)
    if not isinstance(error, HTTPError):
        return _RetryDecision(type(error).__name__, retryable, not_before)
    with error:
        status = f"HTTP {error.code}"
        headers = error.headers
        get_all = getattr(headers, "get_all", None)
        if get_all is not None:
            values = tuple(get_all("Retry-After", ()))
        elif headers is not None and "Retry-After" in headers:
            values = (headers["Retry-After"],)
        else:
            values = ()
    throttled = error.code in _THROTTLED
    if not throttled and not (retryable and values):
        return _RetryDecision(status, retryable, not_before)
    now = time.time()
    fallback = max(not_before, now + fallback_wait)
    if not values:
        return _RetryDecision(status, False, fallback, "local_fallback")
    if len(values) == 1:
        try:
            deadline = max(not_before, _cooldown(values[0], now, minimum_wait=0.0))
        except ContractError:
            pass
        else:
            return _RetryDecision(status if throttled else "retry_after", False,
                                  deadline, "publisher", values[0].strip())
    return _RetryDecision("invalid_retry_after", False, fallback, "invalid_publisher_header",
                          rejected_headers=tuple(value[:128] for value in values[:4]))


def _persist_pace(pace: Path, deadline: float) -> None:
    try:
        _save(pace, {"not_before": deadline}, replace=True)
    except OSError as error:
        raise ContractError("Could not persist the TPRS request deadline") from error


@dataclass
class _Launch:
    ready: Event = field(default_factory=Event)
    at: float | None = None
    backpressure: _RetryDecision | None = None


def _download_reports(store: ProvenanceStore, pace: Path, roster_receipt: str, identifiers: list[str],
                      reports: dict[str, ReportSelection], holds: dict[str, str], *, limit: int,
                      interval: float, jobs: int, not_before: float, fallback_wait: float) -> dict[str, Any]:
    stopped, start_gate = Event(), Lock()
    latest_floor = publisher_deadline = reserved_until = not_before

    def stop_launches() -> None:
        with start_gate:
            stopped.set()

    def download(identifier: str, launch: _Launch) -> AcquiredSource | None:
        nonlocal publisher_deadline

        def admit() -> None:
            nonlocal latest_floor, reserved_until
            with start_gate:
                if stopped.is_set():
                    raise CancelledError()
                now = time.time()
                # Keep a conservative deadline if the process exits with parallel work unresolved.
                reservation = max(interval, PARALLEL_DRAIN_RESERVATION) if jobs > 1 else interval
                reserved_until = max(publisher_deadline, now + reservation)
                try:
                    _persist_pace(pace, reserved_until)
                except ContractError:
                    stopped.set()
                    raise
                launch.at = time.time()
                latest_floor = launch.at + interval
                launch.ready.set()

        try:
            source = ResolvedDataset(f"reference:{RECIPE}:{identifier}", "reference", EDITION, EDITION, "",
                DatasetMetadata(title=f"TPRS campus {identifier} Student Information", version=str(EDITION),
                                publisher="Texas Education Agency", source_url=BROKER, provenance_ids=(roster_receipt,)))
            client = AcquisitionClient(store, opener=build_opener(NoRedirect()), timeout=60, attempts=1,
                                       max_bytes=MAX_REPORT_BYTES, before_open=admit)
            return acquire_form(client, BROKER, source, filename=f"2026-C-{identifier}-STUD.html",
                                method="GET", fields=tuple(_controls(identifier).items()))
        except CancelledError:
            return None
        except HTTPError as error:
            if error.code in _THROTTLED or (
                is_transient_transport_failure(error) and error.headers and "Retry-After" in error.headers
            ):
                with start_gate:
                    stopped.set()
                    launch.backpressure = _retry_policy(error, max(latest_floor, publisher_deadline), fallback_wait)
                    publisher_deadline = launch.backpressure.deadline
                    safe_deadline = max(reserved_until, publisher_deadline)
                    try:
                        _persist_pace(pace, safe_deadline)
                    except ContractError:
                        # A draining worker must retain the deadline even if its first write failed.
                        _persist_pace(pace, safe_deadline)
                        raise
            raise
        finally:
            launch.ready.set()

    pending = deque(identifiers)
    active: dict[Future[AcquiredSource | None], tuple[str, _Launch]] = {}
    outcomes: dict[str, dict[str, Any]] = {}
    submitted = requests = 0
    stop: str | None = None
    unexpected: Exception | None = None
    started = monotonic()
    with ThreadPoolExecutor(max_workers=jobs) as pool, ExitStack() as on_exit:
        on_exit.callback(stop_launches)
        while active or (pending and submitted < limit and stop is None):
            done = [future for future in active if future.done()]
            if not done:
                if stopped.is_set() or stop is not None or not pending or submitted == limit or len(active) == jobs:
                    if not active:
                        break
                    wait(active, return_when=FIRST_COMPLETED)
                    continue
                now = time.time()
                if not_before > now:
                    if submitted == 0:
                        stop = "cooldown"
                        break
                    if active:
                        wait(active, timeout=not_before - now, return_when=FIRST_COMPLETED)
                    else:
                        time.sleep(not_before - now)
                    continue
                if any(future.done() for future in active) or stopped.is_set():
                    continue
                identifier = pending.popleft()
                launch = _Launch()
                active[pool.submit(download, identifier, launch)] = identifier, launch
                submitted += 1
                launch.ready.wait()
                if launch.at is not None:
                    not_before = launch.at + interval
                continue
            for future in done:
                identifier, launch = active.pop(future)
                count = int(launch.at is not None)
                requests += count
                try:
                    acquired = future.result()
                except (AcquisitionError, OSError) as error:
                    outcome = outcomes.setdefault(identifier, {"attempts": 0, "failures": []})
                    outcome["attempts"] += count
                    decision = launch.backpressure or _retry_policy(error, not_before, fallback_wait)
                    status, retryable = decision.status, decision.retryable
                    not_before = max(not_before, decision.deadline)
                    outcome["failures"].append(status)
                    if decision.wait_source is not None:
                        outcome["backpressure"] = {"source": decision.wait_source,
                                                   "retry_after": decision.retry_after}
                        if decision.rejected_headers:
                            outcome["backpressure"]["rejected_headers"] = list(decision.rejected_headers)
                    if (retryable and count and outcome["attempts"] < MAX_ATTEMPTS and submitted < limit
                            and not stopped.is_set()):
                        pending.appendleft(identifier)
                        outcome["status"] = "retry_pending"
                        continue
                    if retryable:
                        status = "retry_exhausted" if outcome["attempts"] == MAX_ATTEMPTS else "retry_deferred"
                    outcome["status"] = status
                    stop = stop or status
                    stop_launches()
                    continue
                except Exception as error:
                    unexpected = unexpected or error
                    stop = stop or "acquisition_failure"
                    stop_launches()
                    continue
                if acquired is None:
                    continue
                outcome = outcomes.setdefault(identifier, {"attempts": 0, "failures": []})
                outcome["attempts"] += count
                outcome["receipt"] = acquired.record_id
                try:
                    digest = store.read_record(acquired.record_id)["artifact"]["sha256"]
                    read_student_report(store, acquired.record_id, scope="campus", identifier=identifier,
                                        expected_sha256=digest)
                except (ContractError, OSError, ValueError, KeyError) as error:
                    holds[identifier] = f"{acquired.record_id}: {error}"
                    outcome["status"] = "report_hold"
                    stop = stop or "report_hold"
                    stop_launches()
                except Exception as error:
                    unexpected = unexpected or error
                    stop = stop or "report_validation_failure"
                    stop_launches()
                else:
                    reports[identifier] = ReportSelection("campus", identifier, acquired.record_id, digest)
                    outcome["status"] = "admitted"
    if unexpected is not None:
        raise unexpected
    if requests:
        not_before = max(not_before, publisher_deadline)
        _persist_pace(pace, not_before)
    for outcome in outcomes.values():
        if outcome["status"] == "retry_pending":
            outcome["status"] = "retry_deferred"
    return {"requested": requests, "stop": stop, "attempts": outcomes, "jobs": jobs,
            "elapsed_seconds": round(monotonic() - started, 3) if requests else 0.0,
            "next_request_at": datetime.fromtimestamp(not_before, timezone.utc).isoformat()}


def acquire_region(store: ProvenanceStore, region: str, roster_receipt: str, *, limit: int = 25,
                   interval: float = MIN_INTERVAL, intake: Path | None = None,
                   preferred: Mapping[str, str] | None = None, partition: int | None = None,
                   jobs: int = 1, fallback_wait: float = RATE_LIMIT_COOLDOWN) -> dict[str, Any]:
    """Use verified retained reports first and persist pacing before each request."""
    if type(limit) is not int or not 0 <= limit <= 50 or not MIN_INTERVAL <= interval <= 3600:
        raise ValueError("Use zero to 50 requests per pass and an interval between eight seconds and one hour")
    if type(jobs) is not int or not 1 <= jobs <= MAX_JOBS:
        raise ValueError("Use one or two in-flight requests")
    if not isfinite(fallback_wait) or not 0 < fallback_wait <= 3600:
        raise ValueError("Throttle fallback must be positive and at most one hour")
    roster = campus_roster(store, region, roster_receipt)
    ids = set(roster if partition is None else campus_partition_ids(roster, partition))
    choices = preferred or {}
    if not choices.keys() <= ids:
        scope = "region" if partition is None else "campus scope"
        raise ContractError(f"Preferred receipts include campuses outside this exact {scope}")
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
        download = _download_reports(store, pace, roster_receipt, sorted(ids - reports.keys() - holds.keys()),
                                     reports, holds, limit=limit, interval=interval, jobs=jobs, not_before=not_before,
                                     fallback_wait=fallback_wait)
        result = {"region": region, "roster_receipt": roster_receipt, "eligible": len(ids), "reused": reused,
                  "admitted": len(reports), "holds": holds, "remaining": sorted(ids - reports.keys()), **download}
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
    parser.add_argument("--jobs", choices=(1, 2), type=int, help="Maximum requests in flight, with shared pacing")
    parser.add_argument("--throttle-fallback", type=float, help="Wait after throttling without a valid Retry-After")
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
            if any(value is not None for value in (
                args.roster_receipt, args.limit, args.interval, args.intake, args.partition, args.jobs, args.throttle_fallback,
            )):
                raise ValueError("--inventory does not take regional acquisition options")
            result = retained_inventory(store, _inventory_receipts(args.inventory), preferred=preferred)
        else:
            if args.roster_receipt is None:
                parser.error("--region requires --roster-receipt")
            result = acquire_region(store, args.region, args.roster_receipt, intake=args.intake, preferred=preferred,
                                    limit=25 if args.limit is None else args.limit, partition=args.partition,
                                    interval=MIN_INTERVAL if args.interval is None else args.interval,
                                    jobs=1 if args.jobs is None else args.jobs,
                                    fallback_wait=RATE_LIMIT_COOLDOWN if args.throttle_fallback is None else args.throttle_fallback)
        print(json.dumps(result, sort_keys=True))
    except (ContractError, OSError, ValueError) as error:
        parser.exit(1, f"TPRS campus acquisition failed: {error}\n")


if __name__ == "__main__":
    main()
