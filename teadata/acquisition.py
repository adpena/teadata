"""Bounded HTTP acquisition with immutable receipts and reusable sessions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
from pathlib import Path
import tempfile
import time
from typing import Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.parse import urlencode

from .dataset_metadata import ResolvedDataset, validate_http_url
from .provenance import ProvenanceStore


class AcquisitionError(RuntimeError):
    """A source could not be acquired without violating its transport contract."""


class NoRedirect(HTTPRedirectHandler):
    """Refuse a follow-up request when a source requires its exact selected URL."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, "Acquisition redirect blocked", headers, fp)


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


def _form_request(url: str, method: str, fields: Sequence[Sequence[str]],
                  content_type: str | None) -> tuple[str, bytes | None, list[tuple[str, str]]]:
    """Validate and encode the one supported UTF-8 public form contract."""
    if method not in {"GET", "POST"}:
        raise ValueError("Public form acquisition requires explicit GET or POST")
    if (isinstance(fields, (str, bytes)) or any(isinstance(pair, (str, bytes))
            or len(pair) != 2 or any(not isinstance(v, str) for v in pair) or not pair[0]
            for pair in fields)):
        raise ValueError("Form controls must be ordered nonempty-name/string-value pairs")
    if content_type is not None and (not isinstance(content_type, str) or not content_type
                                    or '\r' in content_type or '\n' in content_type):
        raise ValueError("Invalid public form content type")
    if method == "POST" and (content_type is None or content_type.split(';', 1)[0].strip().lower() != "application/x-www-form-urlencoded"):
        raise ValueError("POST form replay requires explicit application/x-www-form-urlencoded")
    if method == "POST" and content_type is not None:
        parameters = content_type.split(';')[1:]
        if parameters:
            name, separator, value = parameters[0].partition('=')
            if (len(parameters) != 1 or not separator or name.strip().lower() != 'charset'
                    or value.strip().lower() not in {'utf-8', '"utf-8"'}):
                raise ValueError("POST form controls support only UTF-8, with no other content-type parameters")
    validate_http_url(url)
    if '#' in url:
        raise ValueError("Public acquisition URLs cannot contain a fragment")
    controls = [(pair[0], pair[1]) for pair in fields]
    encoded = urlencode(controls).encode("utf-8")
    requested = url
    if method == "GET" and controls:
        requested += ('&' if '?' in url else '?') + encoded.decode("ascii")
    return requested, encoded if method == "POST" else None, controls


def acquire_form(client: AcquisitionClient, url: str, dataset: ResolvedDataset, *,
                 filename: str, method: str, fields: Sequence[Sequence[str]] = (),
                 content_type: str | None = None, replayed_from: str | None = None) -> AcquiredSource:
    """Acquire explicit public controls, preserving order and repeated names.

    POST controls use UTF-8; only an absent or explicit UTF-8 charset is supported.
    Only a supplied content type becomes a request header. Authentication,
    cookies, arbitrary headers and configuration activation are not replayed.
    Transport bounds and immutable publication remain client/store concerns.
    """
    requested, body, controls = _form_request(url, method, fields, content_type)
    return client.fetch(requested, dataset, filename=filename,
        body=body,
        headers={"Content-Type": content_type} if content_type is not None else {},
        parameters={"form_url": url, "form_controls": controls, "form_content_type": content_type,
                    "transport_limits": {"timeout": client.timeout, "attempts": client.attempts,
                                         "max_bytes": client.max_bytes, "max_retry_wait": client.max_retry_wait},
                    **({"replayed_from": replayed_from} if replayed_from is not None else {})})


def replay_acquisition(client: AcquisitionClient, record_id: str, *, content_type: str | None = None) -> AcquiredSource:
    """Repeat one retained public request as a new immutable observation.

    Older GET receipts need no form reconstruction. Older POST receipts must
    already retain explicit controls; labels never supply missing request data.
    """
    record = client.store.read_record(record_id)
    if record["kind"] != "acquisition" or record.get("method") != "download":
        raise ValueError("Replay requires an original HTTP acquisition receipt")
    parameters = record.get("parameters", {})
    method = parameters.get("http_method")
    if method not in {"GET", "POST"} or not record.get("requested_url"):
        raise ValueError("Original receipt lacks an explicit public request method/URL")
    if method == "POST" and "form_controls" not in parameters:
        raise ValueError("Original POST receipt lacks explicit public form controls")
    retained_type = parameters.get("form_content_type")
    if "form_content_type" in parameters and content_type is not None and retained_type != content_type:
        raise ValueError("Replay cannot change the retained request content type")
    content_type = retained_type if "form_content_type" in parameters else content_type
    if method == "POST" and content_type is None:
        raise ValueError("Legacy POST receipt needs an explicit request content type")
    if method == "GET" and "form_controls" in parameters and "form_url" not in parameters:
        raise ValueError("GET form controls need their original form URL for exact replay")
    url = parameters.get("form_url", record["requested_url"])
    fields = parameters.get("form_controls", ())
    requested, _, _ = _form_request(url, method, fields, content_type)
    if requested != record["requested_url"]:
        raise ValueError("Reconstructed form request differs from the original observed request URL")
    from .dataset_metadata import DatasetMetadata, ReferenceDocument

    client.store.verify_record_closure([record_id])
    metadata = DatasetMetadata.from_dict(record["source_metadata"])
    retained_previous = next((r for r in metadata.references if r.reference_id == "previous_acquisition"), None)
    if retained_previous is not None and (retained_previous.role != "source_notes"
            or retained_previous.provenance_ids != (parameters.get("replayed_from"),)):
        raise ValueError("Existing previous_acquisition reference is not this request's replay history")
    previous = ReferenceDocument(reference_id="previous_acquisition", role="source_notes",
        url=record["requested_url"], title="Previous exact public acquisition; no revision equivalence asserted",
        provenance_ids=(record_id,), reviewed_sha256=record["artifact"]["sha256"])
    metadata = replace(metadata, references=tuple(r for r in metadata.references
                                                if r.reference_id != previous.reference_id) + (previous,))
    source = ResolvedDataset(record["dataset"], record["section"], record["requested_year"],
        record["resolved_year"], "", metadata)
    return acquire_form(client, url, source,
        filename=record["artifact"]["filename"], method=method,
        fields=fields,
        content_type=content_type,
        replayed_from=record_id)
