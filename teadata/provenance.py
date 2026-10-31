"""Immutable acquisition and processing receipts with content-addressed raw files.

No network activity occurs here. Downloaders supply observed URLs and timestamps;
observing an existing local file never invents its original download date.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from .dataset_metadata import DatasetMetadata, ResolvedDataset, validate_http_url


def _timestamp(value: datetime | None = None) -> str:
    value = value if value is not None else datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Observation timestamps must include a timezone")
    return value.astimezone(timezone.utc).isoformat()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _identifier(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("Expected a SHA-256 identifier")
    return value


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_once(temporary: Path, destination: Path) -> None:
    """Publish a complete file without replacing an existing record or object."""
    try:
        os.link(temporary, destination)
    except FileExistsError:
        if _file_digest(temporary) != _file_digest(destination):
            raise ValueError(
                f"Conflicting immutable provenance file: {destination.name}"
            )


class ProvenanceStore:
    """A portable directory of JSON receipts and exact source/output file bytes."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def retain_artifact(self, source: str | Path) -> dict[str, Any]:
        """Restore retained bytes without inventing another acquisition event."""
        return self._archive(source)

    def _archive(self, source: str | Path) -> dict[str, Any]:
        source = Path(source)
        objects = self.root / "objects"
        objects.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            digest = sha256()
            size = 0
            with (
                source.open("rb") as stream,
                tempfile.NamedTemporaryFile(dir=objects, delete=False) as target,
            ):
                temporary = Path(target.name)
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
                    target.write(block)
            content_hash = digest.hexdigest()
            _publish_once(temporary, objects / content_hash)
            return {"filename": source.name, "sha256": content_hash, "size_bytes": size}
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _record(self, payload: dict[str, Any]) -> str:
        raw = _canonical({"schema_version": 1, **payload})
        record_id = sha256(raw).hexdigest()
        records = self.root / "records"
        records.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=records, delete=False) as target:
                temporary = Path(target.name)
                target.write(raw)
            _publish_once(temporary, records / f"{record_id}.json")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return record_id

    def record_acquisition(
        self,
        source: str | Path,
        dataset: ResolvedDataset,
        *,
        downloaded_at: datetime | None = None,
        recorded_at: datetime | None = None,
        requested_url: str | None = None,
        resolved_url: str | None = None,
        response_headers: Mapping[str, str] | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> str:
        """Archive exact bytes and record one successful download or local observation.

        downloaded_at is supplied only when known from the acquisition itself.
        Repeated downloads remain separate receipts, even for identical bytes.
        Header capture excludes cookies, authorization, and unrelated headers.
        """
        observed = _timestamp(recorded_at)
        validate_http_url(requested_url)
        validate_http_url(resolved_url)
        downloaded = _timestamp(downloaded_at) if downloaded_at is not None else None
        if downloaded is not None and downloaded > observed:
            raise ValueError("downloaded_at cannot be later than recorded_at")
        if downloaded is None and any(
            value is not None
            for value in (requested_url, resolved_url, response_headers)
        ):
            raise ValueError("Observed HTTP metadata requires a known downloaded_at")
        headers = {
            key.lower(): value
            for key, value in (response_headers or {}).items()
            if key.lower()
            in {
                "etag",
                "last-modified",
                "content-type",
                "content-encoding",
                "content-length",
                "content-disposition",
            }
        }
        return self._record(
            {
                "kind": "acquisition",
                "dataset": dataset.dataset,
                "section": dataset.section,
                "requested_year": dataset.requested_year,
                "resolved_year": dataset.resolved_year,
                "source_metadata": dataset.metadata.to_dict(),
                "method": "download" if downloaded is not None else "local_observation",
                "recorded_at": observed,
                "downloaded_at": downloaded,
                "requested_url": requested_url,
                "resolved_url": resolved_url,
                "response_headers": headers,
                "parameters": dict(parameters or {}),
                "artifact": self._archive(source),
            }
        )

    def record_processing(
        self,
        *,
        dataset: str,
        inputs: Sequence[str],
        outputs: Mapping[str, str | Path],
        processor: str,
        code_version: str,
        parameters: Mapping[str, Any] | None = None,
        validation: Mapping[str, Any] | None = None,
        diagnostics: Mapping[str, Any] | None = None,
        recorded_at: datetime | None = None,
    ) -> str:
        """Link one transformation to its input receipts, code, and output artifacts."""
        if (
            not inputs
            or not outputs
            or not processor.strip()
            or not code_version.strip()
        ):
            raise ValueError(
                "Processing requires inputs, outputs, processor, and code_version"
            )
        for record_id in inputs:
            self.read_record(record_id)
        return self._record(
            {
                "kind": "processing",
                "dataset": dataset,
                "recorded_at": _timestamp(recorded_at),
                "inputs": list(inputs),
                "processor": processor,
                "code_version": code_version,
                "parameters": dict(parameters or {}),
                "validation": dict(validation) if validation is not None else None,
                **({"diagnostics": dict(diagnostics)} if diagnostics is not None else {}),
                "outputs": {
                    role: self._archive(path) for role, path in outputs.items()
                },
            }
        )

    def read_record(self, record_id: str) -> dict[str, Any]:
        path = self.root / "records" / f"{_identifier(record_id)}.json"
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != record_id:
            raise ValueError(f"Provenance receipt checksum mismatch: {record_id}")
        value = json.loads(raw)
        if value.get("schema_version") != 1 or value.get("kind") not in {
            "acquisition",
            "processing",
        }:
            raise ValueError("Unsupported provenance record schema")
        return value

    def history(self, dataset: str) -> list[dict[str, Any]]:
        """Return independent observations in timestamp order without rewriting history."""
        records = []
        for path in (self.root / "records").glob("*.json"):
            value = self.read_record(path.stem)
            if value.get("dataset") == dataset:
                records.append({"record_id": path.stem, **value})
        return sorted(
            records, key=lambda value: (value["recorded_at"], value["record_id"])
        )

    def verify_artifact(self, content_hash: str) -> Path:
        path = self.root / "objects" / _identifier(content_hash)
        if _file_digest(path) != content_hash:
            raise ValueError(f"Archived artifact checksum mismatch: {content_hash}")
        return path

    def verify_record_closure(self, record_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        """Verify only the requested immutable graph, including reference links.

        Archived hashes own content identity; portable reference paths need not
        exist outside the original configuration directory. Ordinary source
        builds still validate their selected local paths separately.
        """
        from .manifest import verify_reference_receipts

        records: dict[str, dict[str, Any]] = {}
        pending: set[str] = set()

        def visit(identifier: str) -> None:
            if identifier in records:
                return
            if identifier in pending:
                raise ValueError("Provenance records contain a cycle")
            pending.add(identifier)
            record = self.read_record(identifier)
            if record["kind"] == "acquisition":
                artifacts = [record["artifact"]]
                metadata = DatasetMetadata.from_dict(record["source_metadata"])
                dependencies = [*metadata.provenance_ids,
                                *(item for ref in metadata.references for item in ref.provenance_ids)]
            else:
                artifacts = list(record["outputs"].values())
                metadata = None
                dependencies = record["inputs"]
            for dependency in dependencies:
                visit(dependency)
            for artifact in artifacts:
                path = self.verify_artifact(artifact["sha256"])
                if path.stat().st_size != artifact["size_bytes"]:
                    raise ValueError("Provenance artifact size differs from its receipt")
            if metadata is not None:
                verify_reference_receipts(ResolvedDataset(record["dataset"], record["section"],
                    record["requested_year"], record["resolved_year"], "", metadata),
                    self, check_local_paths=False, verify_all_claimed=True)
            pending.remove(identifier)
            records[identifier] = record

        for identifier in record_ids:
            visit(identifier)
        return records

    def copy_record_from(self, source: ProvenanceStore, record_id: str) -> None:
        """Retain an exact selected receipt/object closure, preserving history."""
        records = source.verify_record_closure([record_id])
        for identifier, record in records.items():
            artifacts = [record["artifact"]] if record["kind"] == "acquisition" else record["outputs"].values()
            for artifact in artifacts:
                copied = self.retain_artifact(source.verify_artifact(artifact["sha256"]))
                if copied["sha256"] != artifact["sha256"] or copied["size_bytes"] != artifact["size_bytes"]:
                    raise ValueError("Copied provenance artifact differs from its receipt")
            if self._record(record) != identifier:
                raise ValueError("Copied provenance receipt changed its identity")
