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
import stat
import tempfile
import unicodedata
from urllib.parse import urljoin
from urllib.request import urlopen

from platformdirs import user_cache_path

from .analytical import ANALYTICAL_DATASETS, ANALYTICAL_ROLES, analytical_dataset, validate_artifact, validate_leaf_artifact
from .dataset_metadata import COLUMN_SCHEMA_MAX_BYTES, ColumnSchemaRef, EditionMetadataRef, validate_http_url
from .manifest import (
    ContractError,
    DatasetCatalog,
    canonical_json,
    file_identity,
    read_sqlite_catalog,
)


ROLES = frozenset({"snapshot", "entities", "maps", "boundaries"})
ANALYTICAL_RELEASE_ROLES = ROLES | ANALYTICAL_ROLES
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
            digest = sha256()
            limit = expected.get("size_bytes", 4 * 1024 * 1024) if expected else 4 * 1024 * 1024
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > limit:
                    raise ContractError("Download exceeds the declared artifact size")
                stream.write(block)
                digest.update(block)
        observed = {"sha256": digest.hexdigest(), "size_bytes": size}
        if expected and any(observed[key] != expected[key] for key in ("sha256", "size_bytes") if key in expected):
            raise ContractError("Downloaded artifact checksum or size mismatch")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _capture_column_schema(path: Path, descriptor: ColumnSchemaRef | EditionMetadataRef, *,
                           cached: tuple[Path, tuple[int, ...], bytes] | None = None) -> tuple[bytes, tuple[int, ...]]:
    """Admit one regular metadata file before reading its exact verified buffer."""
    def signature(state):
        return tuple(getattr(state, key) for key in
                     ("st_dev", "st_ino", "st_mode", "st_uid", "st_size", "st_mtime_ns", "st_ctime_ns"))
    try:
        state = path.stat(follow_symlinks=False)
        if (not stat.S_ISREG(state.st_mode) or not 0 < descriptor.size_bytes <= COLUMN_SCHEMA_MAX_BYTES
                or state.st_size != descriptor.size_bytes):
            raise ContractError("Column-schema resource must be regular and match its declared bounded size")
        before = signature(state)
        if cached is not None and cached[:2] == (path, before):
            return cached[2], before
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if (not stat.S_ISREG(opened.st_mode) or opened.st_size != descriptor.size_bytes
                    or signature(opened) != before):
                raise ContractError("Column-schema resource changed before reading")
            raw = stream.read(descriptor.size_bytes + 1)
            if signature(os.fstat(stream.fileno())) != before:
                raise ContractError("Column-schema resource changed during reading")
        if (signature(path.stat(follow_symlinks=False)) != before or len(raw) != descriptor.size_bytes
                or sha256(raw).hexdigest() != descriptor.sha256):
            raise ContractError("Column-schema resource checksum, size or identity mismatch")
        return raw, before
    except OSError as error:
        raise ContractError("Invalid column-schema resource filesystem entry") from error


def _schema_publication_destinations(catalog: DatasetCatalog) -> tuple[ColumnSchemaRef | EditionMetadataRef, ...]:
    """Reject every colliding local destination before metadata resolution/writes."""
    names = {"release.json"}
    def reserve(location):
        name = _safe_filename(location)
        key = unicodedata.normalize("NFC", name).casefold().rstrip(" .")
        if not key or "\0" in name or key in names:
            raise ContractError("Schema3 publication destinations collide with a reserved root or another resource")
        names.add(key)
    if catalog.to_dict()["schema_version"] == 4:
        # Child metadata must be read to discover destinations; no schema/leaf
        # is opened until every destination has passed this same preflight.
        resources, descriptors = catalog._publication_metadata()
        artifacts = [descriptor.to_dict() for descriptor in descriptors]
    else:
        resources = catalog.column_schema_refs()
        artifacts = catalog.to_dict()["artifacts"].values()
    for artifact in artifacts:
        if "://" not in artifact["location"]:
            reserve(artifact["location"])
    for resource in resources:
        reserve(resource.location)
    return resources


class DataRelease:
    """A complete bundle with one catalog and content hashes for every artifact."""

    def __init__(self, manifest: str | Path, *, expected_sha256: str | None = None, offline: bool = False):
        self.source = str(manifest)
        self.remote = "://" in self.source
        self.cache = Path(user_cache_path("teadata")) / "releases"
        if expected_sha256 is not None and (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
                or any(c not in "0123456789abcdef" for c in expected_sha256)):
            raise ContractError("Manifest pin must be a SHA-256 identity")
        if type(offline) is not bool:
            raise ContractError("Offline selection must be explicit boolean")
        self.offline = offline
        if self.remote:
            validate_http_url(self.source)
            identity = sha256(self.source.encode()).hexdigest()
            path = self.cache / "manifests" / f"{identity}.json"
            if offline and expected_sha256 is None:
                raise ContractError("Offline remote roots require an expected manifest SHA-256")
            # Existing URI slot and content check; never infer a pin from cached leaves.
            if not offline and (expected_sha256 is None or not path.exists()):
                _download(self.source, path, expected={"sha256": expected_sha256} if expected_sha256 else None)
        else:
            path = Path(manifest).expanduser().resolve()
        self.path = path
        with path.open("rb") as stream:
            raw = stream.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise ContractError("Release manifest is too large")
        if expected_sha256 is not None and sha256(raw).hexdigest() != expected_sha256:
            raise ContractError("Release manifest checksum differs from its trusted pin")
        document = json.loads(raw)
        version = document.get("schema_version")
        self._v3 = type(version) is int and version in (3, 4)
        if self._v3:
            if (set(document) != {"schema_version", "catalog"}
                    or type(document["catalog"].get("schema_version")) is not int
                    or document["catalog"]["schema_version"] != version):
                raise ContractError(f"Schema{version} roots require one authoritative edition catalog")
            self._verified = {}
            self._verified_bytes: dict[str, tuple[Path, tuple[int, int, int, int]]] = {}
            self._column_schema_data: dict[str, tuple[Path, tuple[int, ...], bytes]] = {}
            self._manifest_sha256 = sha256(raw).hexdigest()
            self.catalog = DatasetCatalog(document["catalog"], column_schema_loader=self._column_schema_bytes,
                                          edition_metadata_loader=self._column_schema_bytes)
            return
        expected_roles = {1: ROLES, 2: ANALYTICAL_RELEASE_ROLES}.get(version) if type(version) is int else None
        if expected_roles is None or set(document.get("artifacts", {})) != expected_roles:
            raise ContractError("A release requires the complete artifact set for its schema version")
        catalog_document = document["catalog"]
        if catalog_document.get("schema_version") != 2:
            raise ContractError("Legacy release roots require a schema2 catalog")
        self.catalog = DatasetCatalog(catalog_document)
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
        if self._v3:
            raise ContractError("Schema3 processing receipts are bound per selected leaf")
        return json.loads(self._processing)

    @property
    def build_id(self) -> str:
        return self.catalog.build_id

    def _resource_path(self, location: str, digest: str, size: int) -> Path:
        """One location/cache/download path for selected v3 bytes."""
        if "://" in location or self.remote:
            path = self.cache / "objects" / digest
            if not path.exists():
                if self.offline:
                    raise ContractError("Selected resource is not cached for offline use")
                _download(location if "://" in location else urljoin(self.source, location), path,
                          expected={"sha256": digest, "size_bytes": size})
            return path
        return self.path.parent / location

    def _column_schema_bytes(self, descriptor: ColumnSchemaRef | EditionMetadataRef) -> bytes:
        path = self._resource_path(descriptor.location, descriptor.sha256, descriptor.size_bytes)
        raw, before = _capture_column_schema(path, descriptor, cached=self._column_schema_data.get(descriptor.sha256))
        self._column_schema_data[descriptor.sha256] = path, before, raw
        return raw

    def __getstate__(self) -> dict:
        if self._v3:
            return {"_v3": True, "source": self.source, "pin": self._manifest_sha256, "offline": self.offline}
        return self.__dict__.copy()

    def __setstate__(self, state: dict) -> None:
        if state.get("_v3") is True:
            self.__init__(state["source"], expected_sha256=state["pin"], offline=state["offline"])
        else:
            self.__dict__.update(state)

    def artifact(self, role: str) -> Path:
        if self._v3:
            descriptor = self.catalog.artifact_ref(role)
            path = self._resource_path(descriptor.location, descriptor.sha256, descriptor.size_bytes)
            state = path.stat()
            signature = (state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns)
            if self._verified_bytes.get(descriptor.sha256) != (path, signature):
                observed = file_identity(path)
                if observed["sha256"] != descriptor.sha256 or observed["size_bytes"] != descriptor.size_bytes:
                    raise ContractError("Selected leaf checksum or size mismatch")
                self._verified_bytes[descriptor.sha256] = path, signature
            # Acceptance includes the full immutable descriptor, not merely SHA or a role.
            semantic = sha256(canonical_json(descriptor.to_dict()).encode()).hexdigest()
            if self._verified.get(semantic) != (path, signature):
                validate_leaf_artifact(path, descriptor)
                self._verified[semantic] = path, signature
            return path
        if role not in ANALYTICAL_RELEASE_ROLES:
            raise ContractError(f"Unknown artifact role: {role}")
        if role not in self._artifacts:
            raise ContractError("This release has no analytical read model; select a rebuilt release")
        expected = self._artifacts[role]
        if self.remote:
            path = self.cache / "objects" / expected["sha256"]
            if not path.exists():
                if self.offline:
                    raise ContractError("Release artifact is not cached for offline use")
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
        if role in ROLES - {"snapshot"} and read_sqlite_catalog(path).build_id != self.build_id:
            raise ContractError(f"Artifact belongs to a different data build: {role}")
        if role in ANALYTICAL_ROLES:
            validate_artifact(path, self.catalog, role, entities_sha256=self._artifacts["entities"]["sha256"])
        if role == "snapshot" and snapshot_build_id(path) != self.build_id:
            raise ContractError("Snapshot belongs to a different data build")
        self._verified[role] = (path, signature)
        return path

    def verify(self) -> dict[str, Path]:
        if self._v3:
            self.catalog.verify_metadata()
            return {identifier: self.artifact(identifier) for identifier in self.catalog.to_dict()["artifacts"]}
        return {role: self.artifact(role) for role in sorted(self._artifacts)}

    def analytical_dataset(self, name: str):
        """Return a standard Arrow Dataset for campus, district, attributes or sources.

        Use Arrow's native columns/filter/to_batches APIs or pass the dataset to
        another Arrow consumer. An older release refuses this explicit request;
        it never silently computes a different read model at request time.
        """
        if self._v3:
            import pyarrow.dataset as ds
            return ds.dataset(self.artifact(name), format="parquet")
        try:
            role = ANALYTICAL_DATASETS[name]
        except KeyError as error:
            raise ContractError(f"Unknown analytical dataset: {name}") from error
        return analytical_dataset(self.artifact(role), self.catalog, role,
                                  entities_sha256=self._artifacts["entities"]["sha256"])


@lru_cache(maxsize=4)
def _load_release(source: str, stamp: int, expected_sha256: str | None = None, offline: bool = False) -> DataRelease:
    return DataRelease(source, expected_sha256=expected_sha256, offline=offline)


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
    artifacts: dict[str, Path] | None = None,
    *,
    processing: dict | None = None,
    schema_version: int | None = None,
) -> Path:
    """Activate only a complete validated bundle; keep the previous pointer on failure."""
    if type(schema_version) is int and schema_version in (3, 4):
        if artifacts is not None or processing is not None or catalog.to_dict().get("schema_version") != schema_version:
            raise ContractError("Edition publication uses only its matching catalog artifact registry")
        document = {"schema_version": schema_version, "catalog": catalog.to_dict()}
        if len(canonical_json(document).encode()) + 1 > 4 * 1024 * 1024:
            raise ContractError("Release manifest is too large")
        if (directory / "release.json").exists() or (directory / "release.json").is_symlink():
            raise FileExistsError("Schema3 candidate root already exists")
        resources = _schema_publication_destinations(catalog)
        catalog.verify_metadata()
        admitted = []
        for descriptor in resources:
            destination = directory / descriptor.location
            if isinstance(descriptor, EditionMetadataRef):
                identifier = next(row["edition_id"] for row in catalog.editions(include_metadata=False)
                                  if catalog.to_dict()["editions"][row["edition_id"]]["metadata"] == descriptor.to_dict())
                raw = catalog.edition_metadata_bytes(identifier)
            else:
                raw = catalog.column_schema_bytes(descriptor.sha256)
            try:
                destination.stat(follow_symlinks=False)
            except FileNotFoundError:
                present = False
            else:
                _capture_column_schema(destination, descriptor)
                present = True
            admitted.append((destination, raw, present))
        directory.mkdir(parents=True, exist_ok=True)
        for destination, raw, present in admitted:
            if not present:
                resource_temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
                        resource_temporary = Path(stream.name)
                        stream.write(raw)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.link(resource_temporary, destination)
                finally:
                    if resource_temporary is not None:
                        resource_temporary.unlink(missing_ok=True)
    elif schema_version is not None:
        raise ContractError("Select explicit schema3/schema4 or the unchanged legacy publisher")
    elif catalog.to_dict().get("schema_version") != 2:
        raise ContractError("Legacy release publication requires a schema2 catalog")
    elif artifacts is None or processing is None or set(artifacts) not in (ROLES, ANALYTICAL_RELEASE_ROLES):
        raise ContractError("Cannot publish an incomplete release")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        names = {}
        for role, source in artifacts.items():
            suffix = ".parquet" if role in ANALYTICAL_ROLES else ".pkl.gz" if role == "snapshot" else ".sqlite"
            name = f"{role}_{catalog.build_id}{suffix}"
            destination = directory / name
            if (
                role in ROLES - {"snapshot"}
                and read_sqlite_catalog(source).build_id != catalog.build_id
            ):
                raise ContractError(f"Cannot publish mixed data builds: {role}")
            if role in ANALYTICAL_ROLES:
                validate_artifact(source, catalog, role, entities_sha256=file_identity(artifacts["entities"])["sha256"])
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
            "schema_version": 2 if set(artifacts) == ANALYTICAL_RELEASE_ROLES else 1,
            "catalog": catalog.to_dict(),
            "artifacts": names,
            "processing": processing,
        }
    with tempfile.NamedTemporaryFile(dir=directory, mode="w", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(canonical_json(document) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        DataRelease(temporary, offline=schema_version in (3, 4)).verify()
        if schema_version in (3, 4):
            os.link(temporary, directory / "release.json")
        else:
            os.replace(temporary, directory / "release.json")
    finally:
        temporary.unlink(missing_ok=True)
    return directory / "release.json"
