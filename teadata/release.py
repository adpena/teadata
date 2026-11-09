"""Resolve one verified data release rather than discover independent assets."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from functools import lru_cache, partial
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
import json
import gzip
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import TYPE_CHECKING, BinaryIO
import unicodedata
from urllib.parse import urljoin
from urllib.request import urlopen

from platformdirs import user_cache_path

from .analytical import ANALYTICAL_DATASETS, ANALYTICAL_ROLES, analytical_dataset, validate_artifact, validate_leaf_artifact
from .dataset_metadata import ArtifactRef, COLUMN_SCHEMA_MAX_BYTES, ColumnSchemaRef, EditionKey, EditionMetadataRef, validate_http_url
from .dataset_metadata import DataPeriod, FieldBinding, NativeSourceRef, _parse_school_year, _validate_observation_selection
from .manifest import (
    ContractError,
    DatasetCatalog,
    canonical_json,
    file_identity,
    read_sqlite_catalog,
)

if TYPE_CHECKING:
    from .source_series import ObservationSeries
    from .export_provenance import ExportProvenance
    from .entities import Campus, District
    from .source_views import EntitySources, PartitionedSourceTable, SourceTable
    from .tprs_statewide_intake import ReportSelection, TPRSStudentCampusIndex

_StudentCohort = tuple[str, str, str, str]
_ManifestStamp = tuple[int, int, int, int, int]
_LocalManifest = tuple[Path, _ManifestStamp]


ROLES = frozenset({"snapshot", "entities", "maps", "boundaries"})
ANALYTICAL_RELEASE_ROLES = ROLES | ANALYTICAL_ROLES
SNAPSHOT_MAGIC = b"TEADATA_SNAPSHOT_V1 "
_ACTIVE_RELEASE: ContextVar[DataRelease | None] = ContextVar(
    "teadata_active_release", default=None
)


@dataclass(frozen=True, slots=True)
class _CampusObservationPlan:
    request: tuple[tuple[str, ...], int | None]
    year: int
    source: NativeSourceRef
    location: tuple[Path, Path]
    keys: frozenset[EditionKey]
    index: TPRSStudentCampusIndex
    files: tuple[tuple[Path, tuple[int, ...]], ...]

    def matches_files(self, child: DataRelease) -> bool:
        if (child.path, child.cache) != self.location:
            return False
        try:
            for path, stamp in self.files:
                _require_cache_file(path)
                if _metadata_stamp(path.stat(follow_symlinks=False)) != stamp:
                    return False
        except (ContractError, OSError, ValueError):
            return False
        return True


def _open_native_source(ref: NativeSourceRef, parent: str | Path, *, offline: bool,
                        cache_dir: Path | None = None) -> DataRelease:
    location = ref.location
    if "://" not in location:
        location = urljoin(str(parent), location) if "://" in str(parent) else str(Path(parent).parent / location)
    child = DataRelease(location, expected_sha256=ref.manifest_sha256, offline=offline, cache_dir=cache_dir)
    if not child._v3 or child.build_id != ref.build_id:
        raise ContractError("Native source manifest has a different catalog type or build ID")
    editions = {EditionKey.from_dict({name: value for name, value in row.items() if name != "edition_id"})
                for row in child.catalog.editions(include_metadata=False)}
    if editions != set(ref.editions):
        raise ContractError("Native source editions differ from the entity catalog")
    return child


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


def _require_cache_file(path: Path) -> None:
    if not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode) or path.resolve(strict=True) != path:
        raise ContractError("Release cache entries must be regular files without symlink paths")


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


def _metadata_stamp(state: os.stat_result) -> tuple[int, ...]:
    return (state.st_dev, state.st_ino, state.st_mode, state.st_uid, state.st_size, state.st_mtime_ns, state.st_ctime_ns)


def _capture_column_schema(path: Path, descriptor: ColumnSchemaRef | EditionMetadataRef, *,
                           cached: tuple[Path, tuple[int, ...], bytes] | None = None) -> tuple[bytes, tuple[int, ...]]:
    """Admit one regular metadata file before reading its exact verified buffer."""
    def regular(state):
        return stat.S_ISREG(state.st_mode) and not (
            getattr(state, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        state = path.stat(follow_symlinks=False)
        if (not regular(state) or not 0 < descriptor.size_bytes <= COLUMN_SCHEMA_MAX_BYTES
                or state.st_size != descriptor.size_bytes):
            raise ContractError("Column-schema resource must be regular and match its declared bounded size")
        before = _metadata_stamp(state)
        if nofollow and cached is not None and cached[:2] == (path, before):
            return cached[2], before
        flags = os.O_RDONLY | nofollow | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as stream:
            opened = os.fstat(stream.fileno())
            opened_stamp = _metadata_stamp(opened)
            # Windows pathname and descriptor stat expose different identity
            # fields. Compare their size and mtime, then recheck each separately.
            matches = (opened_stamp == before if nofollow else
                       (opened.st_size, opened.st_mtime_ns) == (state.st_size, state.st_mtime_ns))
            if not regular(opened) or not matches:
                raise ContractError("Column-schema resource changed before reading")
            raw = stream.read(descriptor.size_bytes + 1)
            current = os.fstat(stream.fileno())
            if not regular(current) or _metadata_stamp(current) != opened_stamp:
                raise ContractError("Column-schema resource changed during reading")
            current = path.stat(follow_symlinks=False)
            if not regular(current) or _metadata_stamp(current) != before:
                raise ContractError("Column-schema resource changed during reading")
        if len(raw) != descriptor.size_bytes or sha256(raw).hexdigest() != descriptor.sha256:
            raise ContractError("Column-schema resource checksum, size or identity mismatch")
        return raw, before
    except OSError as error:
        raise ContractError("Invalid column-schema resource filesystem entry") from error


def _schema_publication_destinations(
    catalog: DatasetCatalog,
) -> tuple[tuple[ColumnSchemaRef | EditionMetadataRef, ...], tuple[str, ...]]:
    """Reject every colliding local destination before metadata resolution/writes."""
    names = {"release.json"}
    def reserve(location):
        name = _safe_filename(location)
        key = unicodedata.normalize("NFC", name).casefold().rstrip(" .")
        if not key or "\0" in name or key in names:
            raise ContractError("Schema3 publication destinations collide with a reserved root or another resource")
        names.add(key)
    document = catalog.to_dict()
    if document["schema_version"] == 4:
        # Child metadata must be read to discover destinations; no schema/leaf
        # is opened until every destination has passed this same preflight.
        resources, descriptors = catalog._publication_metadata()
    else:
        resources = catalog.column_schema_refs()
        descriptors = tuple(ArtifactRef.from_dict(row) for row in document["artifacts"].values())
    attachments = {item for descriptor in descriptors for item in _student_attachments(catalog, descriptor).values()}
    for attachment in attachments:
        reserve(attachment.location)
    for artifact in descriptors:
        if "://" not in artifact.location:
            reserve(artifact.location)
    for resource in resources:
        reserve(resource.location)
    locations = sorted([*(artifact.location for artifact in descriptors),
                        *(resource.location for resource in resources),
                        *(attachment.location for attachment in attachments)])
    return resources, tuple(locations)


def _student_attachments(catalog: DatasetCatalog, artifact: ArtifactRef) -> dict[str, ColumnSchemaRef]:
    if artifact.edition.dataset != "tprs_student_membership_enrollment_reports":
        return {}
    from .tprs_statewide_intake import student_report_resources

    refs = student_report_resources(catalog._processing_record(artifact))
    if refs and not getattr(catalog, "_v4", False):
        raise ContractError("Statewide student attachments require a schema4 release")
    return refs


def _manifest_stamp(state: os.stat_result) -> _ManifestStamp:
    return state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns


def _local_manifest_stamp(path: Path) -> _ManifestStamp:
    if os.name == "nt":
        # Windows pathname stat reports creation time as ctime. Use the
        # descriptor's change time to match the checks around the read.
        with path.open("rb") as stream:
            return _manifest_stamp(os.fstat(stream.fileno()))
    return _manifest_stamp(path.stat())


class DataRelease:
    """A complete bundle with one catalog and content hashes for every artifact."""

    _metadata_trace: set[tuple[Path, tuple[int, ...]]] | None = None

    def __init__(self, manifest: str | Path, *, expected_sha256: str | None = None, offline: bool = False,
                 cache_dir: Path | None = None, _local_manifest: _LocalManifest | None = None):
        self.source = str(manifest)
        self.remote = "://" in self.source
        self._cache_dir = cache_dir.expanduser().resolve() if cache_dir is not None else None
        self.cache = self._cache_dir or Path(user_cache_path("teadata")) / "releases"
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
            if offline:
                _require_cache_file(path)
            # Existing URI slot and content check; never infer a pin from cached leaves.
            if not offline and (expected_sha256 is None or not path.exists()):
                _download(self.source, path, expected={"sha256": expected_sha256} if expected_sha256 else None)
        else:
            path = Path(manifest).expanduser().resolve()
        if _local_manifest is not None and path != _local_manifest[0]:
            raise ContractError("Local release manifest changed during selection")
        self.path = path
        with path.open("rb") as stream:
            if _local_manifest is not None and _manifest_stamp(os.fstat(stream.fileno())) != _local_manifest[1]:
                raise ContractError("Local release manifest changed before reading")
            raw = stream.read(4 * 1024 * 1024 + 1)
            if _local_manifest is not None and _manifest_stamp(os.fstat(stream.fileno())) != _local_manifest[1]:
                raise ContractError("Local release manifest changed during reading")
        if len(raw) > 4 * 1024 * 1024:
            raise ContractError("Release manifest is too large")
        self._manifest_sha256 = sha256(raw).hexdigest()
        if expected_sha256 is not None and self._manifest_sha256 != expected_sha256:
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
        if record["parameters"].get("native_sources", []) != [ref.to_dict() for ref in self.catalog.native_sources]:
            raise ContractError("Processing receipt differs from the entity catalog's native sources")
        names = {Path(self.source).name.casefold()}
        for artifact in self._artifacts.values():
            names.add(_safe_filename(artifact.get("filename")).casefold())
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
        if any(ref.location.casefold() in names for ref in self.catalog.native_sources):
            raise ContractError("Native manifest location collides with an entity release file")
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

    @property
    def manifest_sha256(self) -> str:
        """SHA-256 of the manifest bytes parsed by this release."""
        return self._manifest_sha256

    def _resource_path(self, location: str, digest: str, size: int) -> Path:
        """One location/cache/download path for selected v3 bytes."""
        if "://" in location or self.remote:
            path = self.cache / "objects" / digest
            if self.offline and path.exists():
                _require_cache_file(path)
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
        if getattr(os, "O_NOFOLLOW", 0):
            self._column_schema_data[descriptor.sha256] = path, before, raw
        if (trace := getattr(self, "_metadata_trace", None)) is not None and len(trace) <= 256:
            trace.add((path, before))
        return raw

    def __getstate__(self) -> dict:
        if self._v3:
            return {"_v3": True, "source": self.source, "pin": self.manifest_sha256, "offline": self.offline,
                    "cache_dir": self._cache_dir}
        return {key: value for key, value in self.__dict__.items()
                if key not in {"_campus_observation_routes", "_campus_observation_plan"}}

    def __setstate__(self, state: dict) -> None:
        if state.get("_v3") is True:
            self.__init__(state["source"], expected_sha256=state["pin"], offline=state["offline"],
                          cache_dir=state.get("cache_dir"))
        else:
            self.__dict__.update(state)

    def artifact(self, role: str, *, discard_cached_reads: bool = False) -> Path:
        identify = partial(file_identity, discard_cached_reads=True) if discard_cached_reads else file_identity
        if self._v3:
            descriptor = self.catalog.artifact_ref(role)
            path = self._resource_path(descriptor.location, descriptor.sha256, descriptor.size_bytes)
            state = path.stat()
            signature = (state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns)
            if self._verified_bytes.get(descriptor.sha256) != (path, signature):
                observed = identify(path)
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
            if self.offline and path.exists():
                _require_cache_file(path)
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
        identity = identify(path)
        if any(identity[key] != expected[key] for key in ("sha256", "size_bytes")):
            raise ContractError(f"Release artifact checksum or size mismatch: {role}")
        if role in ROLES - {"snapshot"} and read_sqlite_catalog(path, immutable=True).build_id != self.build_id:
            raise ContractError(f"Artifact belongs to a different data build: {role}")
        if role in ANALYTICAL_ROLES:
            validate_artifact(path, self.catalog, role, entities_sha256=self._artifacts["entities"]["sha256"])
        if role == "snapshot" and snapshot_build_id(path) != self.build_id:
            raise ContractError("Snapshot belongs to a different data build")
        self._verified[role] = (path, signature)
        return path

    def verify(self, *, discard_cached_reads: bool = False) -> dict[str, Path]:
        """Verify the complete bundle, optionally releasing already hashed file pages."""
        paths = self._verify_artifacts(discard_cached_reads=discard_cached_reads)
        self.verify_native_sources(discard_cached_reads=discard_cached_reads)
        return paths

    def _verify_artifacts(self, *, discard_cached_reads: bool) -> dict[str, Path]:
        artifact = partial(self.artifact, discard_cached_reads=True) if discard_cached_reads else self.artifact
        if self._v3:
            self.catalog.verify_metadata()
            document = self.catalog.to_dict()
            paths = {identifier: artifact(identifier) for identifier in document["artifacts"]}
            if document["schema_version"] == 4:
                self._verify_student_cohorts(document, paths)
            else:
                for descriptor in document["artifacts"].values():
                    if descriptor["edition"]["dataset"] == "tprs_student_membership_enrollment_reports":
                        _student_attachments(self.catalog, ArtifactRef.from_dict(descriptor))
            return paths
        return {role: artifact(role) for role in sorted(self._artifacts)}

    def verify_native_sources(self, *, discard_cached_reads: bool = False) -> dict[str, dict[str, Path]]:
        """Verify attached native roots and return their manifests and content-addressed files."""
        verified = {}
        parent = self.source if self.remote else self.path
        for ref in self.catalog.native_sources:
            child = _open_native_source(ref, parent, offline=self.offline, cache_dir=getattr(self, "_cache_dir", None))
            child.verify(discard_cached_reads=discard_cached_reads)
            verified[ref.manifest_sha256] = {
                "manifest": child.path,
                **{digest: path for digest, (path, _) in child._verified_bytes.items()},
                **{digest: path for digest, (path, _, _) in child._column_schema_data.items()},
            }
        return verified

    def verified_cache_inventory(self) -> dict[str, dict[str, str | int]]:
        """Verify an offline URL release and list the exact cache files it needs."""
        if not self.remote or not self.offline or self._v3:
            raise ContractError("A cache inventory requires an offline entity release URL")
        _require_cache_file(self.path)
        if file_identity(self.path)["sha256"] != self.manifest_sha256:
            raise ContractError("Cached root manifest changed after selection")
        paths = self._verify_artifacts(discard_cached_reads=True)
        selected = [(self.path, self.manifest_sha256)]
        selected.extend((path, self._artifacts[role]["sha256"]) for role, path in paths.items())
        for pin, files in self.verify_native_sources(discard_cached_reads=True).items():
            selected.extend((path, pin if name == "manifest" else name) for name, path in files.items())
        inventory: dict[str, dict[str, str | int]] = {}
        for path, digest in selected:
            if not path.is_file() or path.resolve(strict=True) != path or not path.is_relative_to(self.cache):
                raise ContractError("Release cache files must be regular files within the selected cache")
            relative = path.relative_to(self.cache).as_posix()
            identity = {"sha256": digest, "size_bytes": path.stat().st_size}
            if relative in inventory and inventory[relative] != identity:
                raise ContractError("Release cache paths have contradictory identities")
            inventory[relative] = identity
        return dict(sorted(inventory.items()))

    def _verify_student_cohorts(self, document: dict, paths: dict[str, Path]) -> None:
        cohorts: dict[_StudentCohort, set[str]] = {}
        for identifier, edition_id in document["artifacts"].items():
            if document["editions"][edition_id]["key"]["dataset"] != "tprs_student_membership_enrollment_reports":
                continue
            artifact = self.catalog.artifact_ref(identifier)
            refs = _student_attachments(self.catalog, artifact)
            if not refs:
                continue
            cohort = self._verify_student_report_leaf(artifact, paths[identifier], refs)
            if cohort is None:
                continue
            scope = artifact.edition.scope
            seen = cohorts.setdefault(cohort, set())
            if scope in seen:
                raise ContractError("Duplicate student leaf in the statewide cohort")
            seen.add(scope)
        if any(scopes != {"district", "state"} for scopes in cohorts.values()):
            raise ContractError("A statewide student cohort requires matching district and Texas leaves")

    def _student_report_ledger(
        self, artifact: ArtifactRef, refs: dict[str, ColumnSchemaRef],
    ) -> tuple[tuple[ReportSelection, ...], _StudentCohort | None]:
        from .tprs_statewide_intake import CAMPUS_COHORT_OPERATION, CAMPUS_PARTITION_OPERATION, student_report_ledger
        record = self.catalog._processing_record(artifact)
        key = artifact.edition
        try:
            recipe = record["parameters"]["recipe"]
            campus = recipe["operation"] in (CAMPUS_COHORT_OPERATION, CAMPUS_PARTITION_OPERATION)
            if campus:
                if (key.edition != "2026" or key.scope != "campus" or key.table != "campus"
                        or key.grain != "one row per campus_number" or artifact.contract["key_columns"] != ["campus_number"]):
                    raise ContractError("Campus cohort requires the exact campus edition key and grain")
            elif recipe["cohort"].get(key.scope) != key.identity:
                raise ContractError("Statewide student leaf does not belong to its cohort")
            document = json.loads(self.catalog.edition_metadata_bytes(key.identity))
            reports = student_report_ledger(document, key, self._column_schema_bytes(refs["roster"]), record)
            hashes = [item.sha256 for item in reports]
            if (artifact.contract["inputs"] != hashes or artifact.contract["rows"] != len(reports)
                    or key.revision_sha256 != EditionKey.source_revision(tuple(hashes))):
                raise ContractError("Student leaf differs from its ordered publisher receipts")
            if campus:
                return reports, None
            return reports, (recipe["cohort"]["district"], recipe["cohort"]["state"],
                             recipe["intake_sha256"], refs["roster"].sha256)
        except (KeyError, OSError, EOFError, TypeError, ValueError) as error:
            raise ContractError("Invalid roster-bound student ledger") from error

    def _verify_student_report_leaf(
        self, artifact: ArtifactRef, path: Path, refs: dict[str, ColumnSchemaRef],
    ) -> _StudentCohort | None:
        import pyarrow.parquet as pq

        from .tprs_statewide_intake import verify_audit_coverage
        from .tprs_students import _STATEWIDE_ROW_GROUP_ROWS

        reports, cohort = self._student_report_ledger(artifact, refs)
        try:
            scope = artifact.edition.scope
            identifier_column = f"{scope}_number"
            columns = ([] if scope == "state" else [identifier_column]) + ["tprs_report_receipt", "tprs_report_sha256"]
            parquet = pq.ParquetFile(path)
            if scope == "campus" and any(parquet.metadata.row_group(index).num_rows > _STATEWIDE_ROW_GROUP_ROWS
                                         for index in range(parquet.num_row_groups)):
                raise ContractError("Campus student leaf exceeds its row-group bound")
            expected = iter(reports)
            for batch in parquet.iter_batches(columns=columns, batch_size=256):
                for row in batch.to_pylist():
                    item = next(expected, None)
                    if (item is None or (row.get(identifier_column, "S"), row["tprs_report_receipt"], row["tprs_report_sha256"])
                            != (item.identifier, item.receipt, item.sha256)):
                        raise ContractError("Student row identity differs from its report ledger")
            if next(expected, None) is not None:
                raise ContractError("Statewide student leaf omits report rows")
            metadata = self.catalog.metadata(artifact.edition.dataset, edition=artifact.edition)
            grid, = [ref for ref in metadata.references if ref.reference_id == "report_grid"]
            if refs["grid"].sha256 != grid.reviewed_sha256:
                raise ContractError("Statewide student audit grid differs from the reviewed source reference")
            count = self.catalog._processing_record(artifact)["validation"]["raw_field_hold_count"]
            if type(count) is not int or count != verify_audit_coverage(
                self._column_schema_bytes(refs["raw_audit"]), reports, self._column_schema_bytes(refs["grid"]),
            ):
                raise ContractError("Statewide student raw audit count differs")
            return cohort
        except (KeyError, OSError, EOFError, TypeError, ValueError) as error:
            raise ContractError("Invalid statewide student leaf or attachment") from error

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

    def for_entity(self, entity: District | Campus, *, edition: str) -> EntitySources:
        """Bind a public district or campus to exact native dataset namespaces."""
        from .entities import Campus, District
        from .source_views import EntitySources

        if isinstance(entity, District):
            scope, identifier = "district", entity.district_number_canon
        elif isinstance(entity, Campus) and not entity.is_private:
            scope, identifier = "campus", entity.lookup_key
        else:
            raise ValueError("Native source access requires a public district or campus")
        if not identifier or not isinstance(edition, str) or not edition:
            raise ValueError("Native source access requires an entity number and explicit edition")
        return EntitySources(self, scope, identifier, edition)

    def table(self, edition: EditionKey) -> SourceTable:
        """Start a lazy scan bound to a native edition without opening its table."""
        from .source_views import SourceTable

        if not isinstance(edition, EditionKey):
            raise TypeError("Native table selection requires an exact EditionKey")
        return SourceTable(self, edition)

    def tprs_student_campus_index(self, *, edition: str) -> TPRSStudentCampusIndex:
        """Prepare reusable point routes for one literal TPRS Student report edition."""
        from .tprs_statewide_intake import DATASET, TPRSStudentCampusIndex

        if not isinstance(edition, str) or len(edition) != 4 or not edition.isascii() or not edition.isdigit():
            raise ValueError("Select a four-digit publisher edition, such as '2026'")

        def matches(key: EditionKey) -> bool:
            return (key.dataset, key.scope, key.edition) == (DATASET, "campus", edition)

        def sources() -> Iterator[DataRelease]:
            if self._v3:
                yield self
            else:
                parent = self.source if self.remote else self.path
                for ref in self.catalog.native_sources:
                    if any(matches(key) for key in ref.editions):
                        yield _open_native_source(ref, parent, offline=self.offline,
                                                  cache_dir=getattr(self, "_cache_dir", None))

        routes: dict[str, EditionKey] = {}
        for child in sources():
            keys = [key for key in child.catalog._edition_keys.values() if matches(key)]
            if not keys:
                continue
            if not getattr(child.catalog, "_v4", False):
                raise ContractError("Campus routing requires schema4 Student report ledgers")
            artifacts: dict[str, list[str]] = {}
            for artifact_id, edition_id in child.catalog._artifact_routes.items():
                artifacts.setdefault(edition_id, []).append(artifact_id)
            for key in keys:
                ids = artifacts.get(key.identity, [])
                if len(ids) != 1:
                    raise ContractError("Campus routing requires one artifact per Student edition")
                source = child.source if child.remote else child.path
                selected = DataRelease(source, expected_sha256=child.manifest_sha256,
                                       offline=child.offline, cache_dir=getattr(child, "_cache_dir", None))
                metadata = selected.catalog._edition_child(key.identity)
                try:
                    bindings = (FieldBinding.from_dict(value) for value in metadata["field_bindings"])
                    foreign = any(dependency != key.identity for binding in bindings
                                  for dependency, _ in binding.dependencies)
                except (KeyError, TypeError, ValueError) as error:
                    raise ContractError(f"Invalid Student edition bindings: {error}") from error
                if foreign:
                    raise ContractError("Campus routing requires a self-contained Student edition")
                artifact = selected.catalog.artifact_ref(ids[0])
                if artifact.output != "leaf":
                    raise ContractError("Campus Student routing requires the analytical leaf processing output")
                refs = _student_attachments(selected.catalog, artifact)
                if not refs:
                    raise ContractError("Campus routing requires an admitted Student report ledger")
                reports, cohort = selected._student_report_ledger(artifact, refs)
                if cohort is not None:
                    raise ContractError("Campus routing requires a campus Student report ledger")
                for report in reports:
                    if report.identifier in routes:
                        raise ContractError(f"Multiple Student editions contain campus {report.identifier}")
                    routes[report.identifier] = key
        return TPRSStudentCampusIndex(self.manifest_sha256, edition, routes)

    def _observation_year_options(
        self, dataset: str, fields: tuple[str, ...], scope: str, edition: EditionKey | None,
        trace_metadata: bool = False,
    ) -> Iterator[tuple[DataRelease, EditionKey, dict[str, int]]]:
        _validate_observation_selection(dataset, fields, scope, edition)

        def sources():
            if self._v3:
                yield self, (
                    EditionKey.from_dict({name: value for name, value in row.items() if name != "edition_id"})
                    for row in self.catalog.editions(dataset=dataset, scope=scope, include_metadata=False)
                )
                return
            parent = self.source if self.remote else self.path
            for ref in self.catalog.native_sources:
                keys = [key for key in ref.editions if key.dataset == dataset and key.scope == scope
                        and (edition is None or key == edition)]
                if keys:
                    child = _open_native_source(ref, parent, offline=self.offline,
                                                cache_dir=getattr(self, "_cache_dir", None))
                    if trace_metadata and getattr(child.catalog, "_v4", False):
                        child._metadata_trace = set()
                    yield child, keys

        for child, keys in sources():
            for key in keys:
                if edition is not None and key != edition:
                    continue
                resolved = child.catalog._observation_fields(key, fields)
                periods = child.catalog._field_periods(key, tuple(resolved.values()))
                yield child, key, {
                    field: period.end_year
                    for field, canonical in resolved.items()
                    if (period := periods[canonical]).kind == "school_year" and period.end_year is not None
                }

    @staticmethod
    def _years_by_field(
        fields: tuple[str, ...], options: Iterable[tuple[DataRelease, EditionKey, dict[str, int]]],
    ) -> dict[str, tuple[int, ...]]:
        years: dict[str, set[int]] = {}
        for _, _, periods in options:
            for field, year in periods.items():
                years.setdefault(field, set()).add(year)
        return {field: tuple(sorted(years.get(field, ()))) for field in fields}

    def observation_years(self, dataset: str, *fields: str, scope: str,
                          edition: EditionKey | None = None) -> dict[str, tuple[int, ...]]:
        """List exact measurement years per field from release metadata without opening data leaves."""
        return self._years_by_field(fields, self._observation_year_options(dataset, fields, scope, edition))

    def observations(self, dataset: str, *fields: str, school_year: int | str | None = None,
                     scope: str, edition: EditionKey | None = None) -> SourceTable:
        """Plan one native table by exact or latest shared measurement school year."""
        school_year = _parse_school_year(school_year)
        options = list(self._observation_year_options(dataset, fields, scope, edition))
        return self._select_observation_view(fields, scope, school_year, options)

    def _select_observation_view(
        self, fields: tuple[str, ...], scope: str, school_year: int | None,
        options: Sequence[tuple[DataRelease, EditionKey, dict[str, int]]],
    ) -> SourceTable:
        school_year, matches = self._observation_matches(fields, scope, school_year, options)
        if len(matches) > 1:
            choices = [key.to_dict() for _, key in sorted(matches, key=lambda match: match[1].identity)]
            raise ContractError(f"Multiple editions match school year {school_year} and {scope}. Select edition: {choices}")
        child, key = matches[0]
        return self._observation_view(fields, school_year, child, key)

    def _observation_matches(
        self, fields: tuple[str, ...], scope: str, school_year: int | None,
        options: Sequence[tuple[DataRelease, EditionKey, dict[str, int]]],
    ) -> tuple[int, list[tuple[DataRelease, EditionKey]]]:
        available = self._years_by_field(fields, options)
        if any(not years for years in available.values()) and options:
            from .selectors import _field_suggestions

            names = {name for child, key, _ in options for name in child.catalog._edition_field_names(key)}
            folded = {name.casefold() for name in names}
            unknown = {field: _field_suggestions(field, names) for field, years in available.items()
                       if not years and field.casefold() not in folded}
            if unknown:
                raise ContractError(
                    f"Unknown source field for {scope}. Did you mean: {unknown}? "
                    "See catalog.variables() for exact names."
                )
        if school_year is None:
            latest = {years[-1] for years in available.values() if years}
            if len(latest) != 1 or any(not years for years in available.values()):
                raise ContractError(f"Fields do not share a latest school year for {scope}. Available years: {available}")
            school_year = latest.pop()

        matches = [(child, key) for child, key, years in options
                   if all(years.get(field) == school_year for field in fields)]
        if not matches:
            raise ContractError(f"Fields are unavailable together for school year {school_year} and {scope}. "
                                f"Available years: {available}")
        return school_year, matches

    def _observation_view(
        self, fields: tuple[str, ...], school_year: int, child: DataRelease, key: EditionKey,
    ) -> SourceTable:
        from .observations import _ObservationModel, _ObservationRequest
        from .source_views import SourceTable

        period = DataPeriod("school_year", school_year - 1, school_year)
        canonical_fields = tuple(child.catalog._observation_fields(key, fields).values())
        plan = _ObservationModel(self.catalog, [child]).plan(
            _ObservationRequest(key, field, period) for field in canonical_fields
        )
        if len(plan.projections) != 1:
            raise ContractError("Observation view requires fields from one native table")
        return SourceTable(child, key, _fields=canonical_fields, _period=period,
                           _entity_build=None if self._v3 else (self.build_id, self.manifest_sha256))

    def _entity_observations(
        self, dataset: str, fields: tuple[str, ...], *, school_year: int | None, scope: str,
        edition: EditionKey | None, identifiers: tuple[str, ...],
    ) -> SourceTable | PartitionedSourceTable:
        from .tprs_statewide_intake import DATASET

        route_campuses = dataset == DATASET and scope == "campus" and edition is None and bool(identifiers)
        memoize = (route_campuses and not self._v3 and bool(getattr(os, "O_NOFOLLOW", 0))
                   and 0 < len(fields) <= 32 and all(isinstance(name, str) and len(name) <= 256 for name in fields))
        request = (fields, school_year)
        parent = self.source if self.remote else self.path
        cache_dir = getattr(self, "_cache_dir", None)
        if memoize:
            memo: _CampusObservationPlan | None = getattr(self, "_campus_observation_plan", None)
            if memo is not None and memo.request == request:
                child = _open_native_source(memo.source, parent, offline=self.offline, cache_dir=cache_dir)
                if memo.matches_files(child):
                    return self._routed_student_campuses(fields, memo.year, child, memo.keys, memo.index, identifiers)
                self.__dict__.pop("_campus_observation_plan", None)
        selected_options = (self._observation_year_options(dataset, fields, scope, edition, True) if memoize else
                            self._observation_year_options(dataset, fields, scope, edition))
        options = list(selected_options)
        if route_campuses:
            year, matches = self._observation_matches(fields, scope, school_year, options)
            child, first = matches[0]
            identity = (child.manifest_sha256, first.edition)
            if getattr(child.catalog, "_v4", False) and all(
                (source.manifest_sha256, key.edition) == identity for source, key in matches
            ):
                index = self._automatic_student_campus_index(child, first.edition)
                if index is None:
                    return self._select_observation_view(fields, scope, year, options)._for_entity_ids(identifiers)
                matching_keys = frozenset(key for _, key in matches)
                trace = getattr(child, "_metadata_trace", None)
                if (memoize and trace and len(trace) <= 256 and len(options) <= 64
                        and all(len(str(path)) <= 4096 for path, _ in trace)
                        and all(source is child for source, _, _ in options)):
                    ref, = (ref for ref in self.catalog.native_sources if ref.manifest_sha256 == child.manifest_sha256)
                    memo = _CampusObservationPlan(request, year, ref, (child.path, child.cache), matching_keys,
                                                  index, tuple(trace))
                    selected = _open_native_source(ref, parent, offline=self.offline, cache_dir=cache_dir)
                    if memo.matches_files(selected):
                        # Keep returned views independent of catalogs used only for discovery.
                        result = self._routed_student_campuses(fields, year, selected, matching_keys, index, identifiers)
                        self._campus_observation_plan = memo
                        return result
                return self._routed_student_campuses(fields, year, child, matching_keys, index, identifiers)
        return self._select_observation_view(fields, scope, school_year, options)._for_entity_ids(identifiers)

    def _routed_student_campuses(
        self, fields: tuple[str, ...], year: int, child: DataRelease, matching_keys: frozenset[EditionKey],
        index: TPRSStudentCampusIndex, identifiers: tuple[str, ...],
    ) -> SourceTable | PartitionedSourceTable:
        from .source_views import PartitionedSourceTable

        groups: dict[EditionKey, list[str]] = {}
        missing, unavailable = [], []
        for identifier in identifiers:
            key = index.edition_for(identifier)
            if key is None:
                missing.append(identifier)
            elif key not in matching_keys:
                unavailable.append(identifier)
            else:
                groups.setdefault(key, []).append(identifier)
        if missing or unavailable:
            raise ContractError(
                f"Campus Student Information is incomplete for school year {year} "
                f"in report edition {index.edition}. "
                f"No report for {len(missing)} campus IDs: {missing[:5]}. "
                f"Selected fields unavailable for {len(unavailable)} campus IDs: {unavailable[:5]}."
            )
        parts = tuple(self._observation_view(fields, year, child, key)._for_entity_ids(tuple(ids))
                      for key, ids in groups.items())
        return parts[0] if len(parts) == 1 else PartitionedSourceTable(parts)

    def _automatic_student_campus_index(self, child: DataRelease, edition: str) -> TPRSStudentCampusIndex | None:
        from .tprs_statewide_intake import DATASET

        identity = (child.manifest_sha256, edition)
        indexes: dict[tuple[str, str], TPRSStudentCampusIndex] = self.__dict__.setdefault("_campus_observation_routes", {})
        if identity in indexes:
            return indexes[identity]
        catalog = child.catalog
        keys = {key.identity for key in catalog._edition_keys.values()
                if (key.dataset, key.scope, key.edition) == (DATASET, "campus", edition)}
        artifacts = [catalog.artifact_ref(identifier) for identifier, key in catalog._artifact_routes.items() if key in keys]
        roster_backed = [_student_attachments(catalog, artifact) for artifact in artifacts]
        if not all(roster_backed) or len(artifacts) != len(keys):
            return None
        index = child.tprs_student_campus_index(edition=edition)
        indexes[identity] = index
        return index

    def series(
        self, dataset: str, *fields: str, scope: str, years: Sequence[int | str],
        ids: Sequence[int | str] | None = None, editions: Mapping[int | str, EditionKey] | None = None,
        names: Sequence[str] | None = None, entity_release: DataRelease | None = None,
    ) -> ObservationSeries:
        """Plan the selected fields across explicit measurement school years."""
        from .source_series import ObservationSeries

        return ObservationSeries._plan(self, dataset, fields, scope=scope, years=years, ids=ids, editions=editions,
                                       names=names, entity_release=entity_release)

    def to_csv(self, edition: EditionKey, path_or_buf: str | Path | BinaryIO, *,
               restore_original_headers: bool = False) -> None:
        """Stream one native table to a CSV path or writable binary stream."""
        from .source_views import prepare_export, write_csv

        table = prepare_export(self, edition, restore_original_headers=restore_original_headers)
        write_csv(table, path_or_buf)

    def provenance(self, edition: EditionKey, *, restore_original_headers: bool = False) -> ExportProvenance:
        """Describe the direct table export without opening the data leaf."""
        from .source_views import _plan_export

        selection = _plan_export(self, edition, restore_original_headers=restore_original_headers,
                                 selected_fields=None, include_support=False, period=None)
        return selection.provenance(self, period=None, entity_build=None, entity_ids=None, opaque_filter=False)

    def to_excel(self, edition: EditionKey, path_or_buf: str | Path | BinaryIO, *,
                 restore_original_headers: bool = False) -> None:
        """Stream one native table to an XLSX path or writable binary stream."""
        from .source_views import prepare_export, write_excel

        table = prepare_export(self, edition, restore_original_headers=restore_original_headers)
        write_excel(table, path_or_buf)

    def to_df(self, edition: EditionKey, *, restore_original_headers: bool = False):
        """Materialize one native table as a pandas DataFrame."""
        from .source_views import prepare_export

        return prepare_export(self, edition, restore_original_headers=restore_original_headers).to_df()


@lru_cache(maxsize=4)
def _load_release(source: str, local_manifest: _LocalManifest | None, expected_sha256: str | None = None,
                  offline: bool = False) -> DataRelease:
    return DataRelease(source, expected_sha256=expected_sha256, offline=offline, _local_manifest=local_manifest)


def release_offline() -> bool:
    """Return the scoped or configured offline policy without opening a manifest."""
    active = _ACTIVE_RELEASE.get()
    if active is not None:
        return active.offline
    value = os.environ.get("TEADATA_RELEASE_OFFLINE", "0")
    if value not in {"0", "1"}:
        raise ContractError("TEADATA_RELEASE_OFFLINE must be 0 or 1")
    return value == "1"


def current_release() -> DataRelease:
    active = _ACTIVE_RELEASE.get()
    if active is not None:
        return active
    source = os.environ.get("TEADATA_RELEASE") or os.environ.get("TEADATA_RELEASE_URL")
    if not source:
        source = str(Path(__file__).with_name(".cache") / "release.json")
    expected_sha256 = os.environ.get("TEADATA_RELEASE_SHA256")
    offline = release_offline()
    local_manifest = None
    if "://" not in source:
        configured_path = Path(source).expanduser()
        if os.name == "nt" and sys.version_info < (3, 12):
            # Python 3.11 on Windows also reports creation time from fstat.
            configured_path.stat()
            return DataRelease(source, expected_sha256=expected_sha256, offline=offline)
        path = configured_path.resolve(strict=True)
        local_manifest = path, _local_manifest_stamp(configured_path)
    return _load_release(source, local_manifest, expected_sha256, offline)


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


def _install_candidate(stage: Path, output: Path) -> None:
    """Reserve the candidate name and expose its verified manifest last."""
    output.mkdir(mode=0o700)
    moved = []
    try:
        for entry in sorted(stage.iterdir(), key=lambda item: item.name == "release.json"):
            destination = output / entry.name
            entry.rename(destination)
            moved.append((destination, entry))
    except BaseException:
        for destination, entry in reversed(moved):
            destination.rename(entry)
        output.rmdir()
        raise


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
            raise ContractError("Edition publication uses only its catalog artifact registry with a matching schema version")
        document = {"schema_version": schema_version, "catalog": catalog.to_dict()}
        if len(canonical_json(document).encode()) + 1 > 4 * 1024 * 1024:
            raise ContractError("Release manifest is too large")
        if (directory / "release.json").exists() or (directory / "release.json").is_symlink():
            raise FileExistsError("Schema3 candidate root already exists")
        resources, _ = _schema_publication_destinations(catalog)
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
                and read_sqlite_catalog(source, immutable=True).build_id != catalog.build_id
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
        DataRelease(temporary, offline=True).verify()
        if schema_version in (3, 4):
            os.link(temporary, directory / "release.json")
        else:
            os.replace(temporary, directory / "release.json")
    finally:
        temporary.unlink(missing_ok=True)
    return directory / "release.json"
