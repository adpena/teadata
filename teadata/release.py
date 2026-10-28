"""Resolve one verified data release rather than discover independent assets."""

from __future__ import annotations

from functools import lru_cache
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
import json
import gzip
import os
from pathlib import Path
import tempfile
from urllib.parse import urljoin
from urllib.request import urlopen

from platformdirs import user_cache_path

from .dataset_metadata import validate_http_url
from .manifest import (
    ContractError,
    DatasetCatalog,
    canonical_json,
    file_identity,
    read_sqlite_catalog,
)


ROLES = frozenset({"snapshot", "entities", "maps", "boundaries"})
SNAPSHOT_MAGIC = b"TEADATA_SNAPSHOT_V1 "
_ACTIVE_RELEASE: ContextVar[DataRelease | None] = ContextVar(
    "teadata_active_release", default=None
)


def _safe_filename(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or value in {".", ".."}
        or "\\" in value
    ):
        raise ContractError("Release artifact names must be portable filenames")
    return value


def _download(url: str, path: Path, *, expected: dict | None = None) -> None:
    validate_http_url(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with (
            urlopen(url, timeout=60) as response,
            tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream,
        ):
            temporary = Path(stream.name)
            size = 0
            limit = expected["size_bytes"] if expected else 4 * 1024 * 1024
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > limit:
                    raise ContractError("Download exceeds the declared artifact size")
                stream.write(block)
        if expected and any(
            file_identity(temporary)[key] != expected[key]
            for key in ("sha256", "size_bytes")
        ):
            raise ContractError("Downloaded artifact checksum or size mismatch")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class DataRelease:
    """A complete bundle with one catalog and content hashes for every artifact."""

    def __init__(self, manifest: str | Path):
        self.source = str(manifest)
        self.remote = "://" in self.source
        self.cache = Path(user_cache_path("teadata")) / "releases"
        if self.remote:
            validate_http_url(self.source)
            identity = sha256(self.source.encode()).hexdigest()
            path = self.cache / "manifests" / f"{identity}.json"
            # The configured URL selects the release for this process lifetime.
            _download(self.source, path)
        else:
            path = Path(manifest).expanduser().resolve()
        self.path = path
        raw = path.read_bytes()
        if len(raw) > 4 * 1024 * 1024:
            raise ContractError("Release manifest is too large")
        document = json.loads(raw)
        if (
            document.get("schema_version") != 1
            or set(document.get("artifacts", {})) != ROLES
        ):
            raise ContractError(
                "A release requires exactly snapshot, entities, maps, and boundaries"
            )
        self.catalog = DatasetCatalog(document["catalog"])
        processing = document.get("processing", {})
        record = processing.get("record", {})
        if (
            processing.get("record_id")
            != sha256((canonical_json(record) + "\n").encode()).hexdigest()
        ):
            raise ContractError("Release requires a verified processing receipt")
        if (
            record.get("kind") != "processing"
            or record.get("parameters", {}).get("build_id") != self.catalog.build_id
        ):
            raise ContractError("Processing receipt belongs to a different data build")
        self._processing = canonical_json(processing)
        self._artifacts = document["artifacts"]
        self._verified: dict[str, tuple[Path, tuple[int, int, int, int]]] = {}
        for artifact in self._artifacts.values():
            _safe_filename(artifact.get("filename"))
            digest = artifact.get("sha256")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ContractError("Release artifacts require SHA-256 hashes")
            if (
                type(artifact.get("size_bytes")) is not int
                or artifact["size_bytes"] <= 0
            ):
                raise ContractError("Release artifacts require positive byte sizes")
        for role, artifact in self._artifacts.items():
            output = record.get("outputs", {}).get(role, {})
            if any(
                output.get(key) != artifact[key] for key in ("sha256", "size_bytes")
            ):
                raise ContractError("Artifact does not match its processing receipt")

    @property
    def processing(self) -> dict:
        return json.loads(self._processing)

    @property
    def build_id(self) -> str:
        return self.catalog.build_id

    def artifact(self, role: str) -> Path:
        if role not in ROLES:
            raise ContractError(f"Unknown artifact role: {role}")
        expected = self._artifacts[role]
        if self.remote:
            path = self.cache / "objects" / expected["sha256"]
            if not path.exists():
                _download(
                    urljoin(self.source, expected["filename"]), path, expected=expected
                )
        else:
            path = self.path.parent / expected["filename"]
        stat = path.stat()
        signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if self._verified.get(role) == (path, signature):
            return path
        identity = file_identity(path)
        if any(identity[key] != expected[key] for key in ("sha256", "size_bytes")):
            raise ContractError(f"Release artifact checksum or size mismatch: {role}")
        if role != "snapshot" and read_sqlite_catalog(path).build_id != self.build_id:
            raise ContractError(f"Artifact belongs to a different data build: {role}")
        if role == "snapshot" and snapshot_build_id(path) != self.build_id:
            raise ContractError("Snapshot belongs to a different data build")
        self._verified[role] = (path, signature)
        return path

    def verify(self) -> dict[str, Path]:
        return {role: self.artifact(role) for role in sorted(ROLES)}


@lru_cache(maxsize=4)
def _load_release(source: str, stamp: int) -> DataRelease:
    return DataRelease(source)


def current_release() -> DataRelease:
    active = _ACTIVE_RELEASE.get()
    if active is not None:
        return active
    source = os.environ.get("TEADATA_RELEASE") or os.environ.get("TEADATA_RELEASE_URL")
    if not source:
        source = str(Path(__file__).with_name(".cache") / "release.json")
    stamp = 0 if "://" in source else Path(source).expanduser().stat().st_mtime_ns
    return _load_release(source, stamp)


@contextmanager
def using_release(release: DataRelease | None = None):
    """Pin a request or cache refresh to one release, even during activation."""
    token = _ACTIVE_RELEASE.set(release or current_release())
    try:
        yield _ACTIVE_RELEASE.get()
    finally:
        _ACTIVE_RELEASE.reset(token)


def snapshot_build_id(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        compressed = stream.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(path, "rb") as stream:
        header = stream.readline(100)
    if not header.startswith(SNAPSHOT_MAGIC) or len(header) != len(SNAPSHOT_MAGIC) + 65:
        raise ContractError("Legacy snapshot requires a rebuild from source")
    return header[len(SNAPSHOT_MAGIC) : -1].decode("ascii")


def publish_release(
    directory: Path,
    catalog: DatasetCatalog,
    artifacts: dict[str, Path],
    *,
    processing: dict,
) -> Path:
    """Activate only a complete validated bundle; keep the previous pointer on failure."""
    if set(artifacts) != ROLES:
        raise ContractError("Cannot publish an incomplete release")
    directory.mkdir(parents=True, exist_ok=True)
    names = {}
    for role, source in artifacts.items():
        suffix = ".pkl.gz" if role == "snapshot" else ".sqlite"
        name = f"{role}_{catalog.build_id}{suffix}"
        destination = directory / name
        if (
            role != "snapshot"
            and read_sqlite_catalog(source).build_id != catalog.build_id
        ):
            raise ContractError(f"Cannot publish mixed data builds: {role}")
        identity = file_identity(source)
        if destination.exists():
            if file_identity(destination)["sha256"] != identity["sha256"]:
                raise ContractError(
                    "A build identity already has different artifact bytes"
                )
        else:
            # Same-filesystem staging makes activation atomic and avoids partial downloads.
            os.link(source, destination)
        names[role] = {**identity, "filename": name}
    document = {
        "schema_version": 1,
        "catalog": catalog.to_dict(),
        "artifacts": names,
        "processing": processing,
    }
    with tempfile.NamedTemporaryFile(dir=directory, mode="w", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(canonical_json(document) + "\n")
    try:
        DataRelease(temporary).verify()
        os.replace(temporary, directory / "release.json")
    finally:
        temporary.unlink(missing_ok=True)
    return directory / "release.json"
