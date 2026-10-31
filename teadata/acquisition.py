"""Bounded HTTP acquisition with immutable receipts and reusable sessions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
from pathlib import Path
import tempfile
import time
from typing import Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

from .dataset_metadata import ResolvedDataset, validate_http_url
from .provenance import ProvenanceStore


class AcquisitionError(RuntimeError):
    """A source could not be acquired without violating its transport contract."""


@dataclass(frozen=True)
class AcquiredSource:
    record_id: str
    path: Path


class AcquisitionClient:
    def __init__(
        self,
        store: ProvenanceStore,
        *,
        opener=None,
        timeout: float = 45,
        attempts: int = 3,
        max_bytes: int = 512 * 1024 * 1024,
        max_retry_wait: float = 30,
    ):
        if timeout <= 0 or attempts < 1 or max_bytes < 1 or max_retry_wait < 0:
            raise ValueError("Acquisition limits must be positive")
        self.store = store
        self.opener = opener or build_opener()
        self.timeout = timeout
        self.attempts = attempts
        self.max_bytes = max_bytes
        self.max_retry_wait = max_retry_wait

    def fetch(
        self,
        url: str,
        dataset: ResolvedDataset,
        *,
        filename: str,
        body: bytes | None = None,
        headers: Mapping[str, str] | None = None,
        parameters: Mapping | None = None,
        validate: Callable[[Path], None] | None = None,
        validate_response: Callable[[Mapping[str, str]], None] | None = None,
    ) -> AcquiredSource:
        """Record only a complete, validated transfer; never activate source config.

        Request bodies, session cookies, and authorization headers are not stored.
        Callers record non-secret selection parameters explicitly. Only transient
        transport errors are retried; content and schema failures require review.
        """
        validate_http_url(url)
        if (
            not filename
            or Path(filename).name != filename
            or "\\" in filename
            or filename in {".", ".."}
        ):
            raise ValueError("Acquisition filenames must be portable filenames")
        request = Request(url, data=body, headers=dict(headers or {}))
        client_identity = sha256(Path(__file__).read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(prefix="teadata-acquisition-") as directory:
            target = Path(directory) / filename
            for attempt in range(self.attempts):
                try:
                    with self.opener.open(request, timeout=self.timeout) as response:
                        validate_http_url(response.url)
                        status = getattr(response, "status", 200)
                        if status != 200:
                            raise AcquisitionError(f"Expected HTTP 200; received {status}")
                        if validate_response is not None:
                            validate_response({key.lower(): value for key, value in response.headers.items()})
                        declared = response.headers.get("Content-Length")
                        if declared is not None:
                            try:
                                declared = int(declared)
                            except ValueError as error:
                                raise AcquisitionError("Invalid Content-Length") from error
                            if declared < 0 or declared > self.max_bytes:
                                raise AcquisitionError("Source exceeds the acquisition size limit")
                        size = 0
                        with target.open("wb") as stream:
                            while block := response.read(1024 * 1024):
                                size += len(block)
                                if size > self.max_bytes:
                                    raise AcquisitionError("Source exceeds the acquisition size limit")
                                stream.write(block)
                        if size == 0 or (declared is not None and size != declared):
                            raise AcquisitionError("Empty or incomplete source transfer")
                        if validate is not None:
                            validate(target)
                        record_id = self.store.record_acquisition(
                            target,
                            replace(dataset, path=str(target)),
                            downloaded_at=datetime.now(timezone.utc),
                            requested_url=url,
                            resolved_url=response.url,
                            response_headers=dict(response.headers),
                            parameters={
                                **dict(parameters or {}),
                                "http_method": request.get_method(),
                                "http_status": status,
                                "acquisition_client_sha256": client_identity,
                            },
                        )
                    record = self.store.read_record(record_id)
                    return AcquiredSource(
                        record_id, self.store.verify_artifact(record["artifact"]["sha256"])
                    )
                except HTTPError as error:
                    if error.code not in {408, 429, 500, 502, 503, 504} or attempt + 1 == self.attempts:
                        raise
                    try:
                        delay = self._retry_delay(error.headers.get("Retry-After"), attempt)
                    finally:
                        error.close()
                except (URLError, TimeoutError, ConnectionError):
                    if attempt + 1 == self.attempts:
                        raise
                    delay = self._retry_delay(None, attempt)
                time.sleep(delay)
        raise AcquisitionError("Acquisition did not complete")

    def _retry_delay(self, retry_after: str | None, attempt: int) -> float:
        delay = float(min(2**attempt, 8))
        if retry_after is not None:
            try:
                if retry_after.strip().isdigit():
                    delay = float(retry_after.strip())
                else:
                    deadline = parsedate_to_datetime(retry_after)
                    if deadline.tzinfo is None:
                        raise ValueError("Retry-After date must include a timezone")
                    delay = max(0.0, (deadline - datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError, OverflowError) as error:
                raise AcquisitionError("Invalid Retry-After; acquisition deferred") from error
        if delay > self.max_retry_wait:
            raise AcquisitionError("Retry-After exceeds the local wait limit; acquisition deferred")
        return delay
