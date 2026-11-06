"""One immutable dataset catalog shared by a snapshot and all its sidecars."""

from __future__ import annotations

from contextvars import ContextVar
from contextlib import closing
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import os
import sqlite3
import unicodedata
from typing import Any, Callable, Iterable, Mapping

from ._sqlite import readonly_connection
from .dataset_metadata import ArtifactRef, ColumnSchemaRef, DataPeriod, DatasetMetadata, EditionKey, EditionMetadataRef, FieldBinding, LeafColumnSupport, MetricDefinition, ResolvedDataset
from .metric_bindings import MetricBinding, parse_bindings
from .provenance import ProvenanceStore, validate_record_graph, validate_reference_records
from .values import Measurement, ValuePolicy, ValueStatus, measure
from .transfer_summary import charter_transfer_observations
from .column_schema import (
    parse_column_schema, reader_options, validate_selected_crosswalk,
    header_records as crosswalk_records, header_rename_map, crosswalk_dataset, table_input_sha256s,
)


CATALOG_SCHEMA = 2
CATALOG_TABLE = "teadata_catalog"
CATALOG_TABLE_DDL = (
    f"CREATE TABLE IF NOT EXISTS {CATALOG_TABLE} "
    "(singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
    "build_id TEXT NOT NULL CHECK(length(build_id)=64), document TEXT NOT NULL)"
)


class ContractError(RuntimeError):
    """An artifact or source does not satisfy the dataset contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def schema_fingerprint(columns: Any) -> str:
    """Identify an ordered source schema independently of row values."""
    return sha256(
        canonical_json([str(column) for column in columns]).encode()
    ).hexdigest()


def file_identity(path: str | Path, *, discard_cached_reads: bool = False) -> dict[str, Any]:
    """Hash every byte; optionally release cache pages behind a one-pass scan."""
    path = Path(path)
    digest = sha256()
    advise = getattr(os, "posix_fadvise", None) if discard_cached_reads else None
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None) if advise is not None else None
    discard_window = 16 * 1024 * 1024
    offset = position = 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if block:
                digest.update(block)
            if advise is not None and dontneed is not None:
                position += len(block)
                if position - offset >= discard_window or (not block and position > offset):
                    try:
                        advise(stream.fileno(), offset, position - offset, dontneed)
                    except OSError:
                        # Cache advice is best effort. Filesystem support does
                        # not change the validity of a complete checksum.
                        advise = None
                    offset = position
            if not block:
                break
    return {
        "filename": path.name,
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def processor_identity() -> str:
    """Hash processing source content without machine paths or modification times."""
    root = Path(__file__).resolve().parent
    digest = sha256()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if any(
            part in {"data", ".cache", "__pycache__"}
            for part in path.relative_to(root).parts
        ):
            continue
        if relative.as_posix() in {"verification.py", "acquisition.py", "sources.py"} or (
            relative.parts[0] == "scripts" and relative.name != "paths.py"
        ):
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    lockfile = root.parent / "uv.lock"
    if lockfile.exists():
        digest.update(b"uv.lock\0" + lockfile.read_bytes())
    return digest.hexdigest()


def _metadata_content(metadata: DatasetMetadata) -> dict[str, Any]:
    document = metadata.to_dict()
    document.pop("provenance_ids", None)
    return document


def _leaf_semantics(metadata: DatasetMetadata, rows: Iterable[Mapping[str, Any]]) -> str:
    """Canonical interpretation inputs; editorial source descriptions are excluded."""
    periods = {row["field"]: row["period"] for row in rows}
    semantic = {"periods": periods,
        "metrics": {name: value.to_dict() for name, value in metadata.metrics.items() if value.field in periods},
        "estimation": metadata.to_dict().get("estimation")}
    if metadata.source_qualifiers is not None:
        semantic["source_qualifiers"] = metadata.source_qualifiers.to_dict()
    return sha256(canonical_json(semantic).encode()).hexdigest()


def _validate_methodology(metadata: DatasetMetadata, policy: ValuePolicy | None) -> None:
    """One reviewed-methodology gate for source builds and both catalog formats."""
    required = set()
    if policy is not None:
        if any(rule.status == ValueStatus.SUPPRESSED for rule in policy.tokens.values()) and not policy.reference_id:
            raise ContractError("Suppression policies require an exact reviewed methodology reference")
        if policy.reference_id:
            required.add(policy.reference_id)
    if metadata.estimation is not None:
        required.add(metadata.estimation.reference_id)
    if metadata.source_qualifiers is not None and metadata.source_qualifiers.reference_id is not None:
        required.add(metadata.source_qualifiers.reference_id)
    references = {reference.reference_id: reference for reference in metadata.references}
    for identifier in required:
        reference = references.get(identifier)
        if (reference is None or reference.status != "available" or reference.reviewed_sha256 is None
                or not reference.provenance_ids):
            raise ContractError("Value policy, estimation or source qualifiers require retained reviewed methodology bytes")


def _validate_leaf_support(contract: Mapping[str, Any], rows: Iterable[Mapping[str, Any]], metadata: DatasetMetadata) -> None:
    support = LeafColumnSupport.from_contract(contract)
    row_run = metadata.source_qualifiers.row_run_column if metadata.source_qualifiers is not None else None
    if support is not None or row_run is not None:
        columns = set(contract["columns"])
        fields = {row["field"] for row in rows if row["field"] in columns}
        if support is not None:
            support.validate_fields(fields,
                count_fields={value.field for value in metadata.metrics.values() if value.unit == "count" and value.field in columns})
        if row_run is not None and (
            row_run not in columns or (row_run not in fields and (support is None or row_run not in support.context))
        ):
            raise ValueError("SOF row-run column must be a publisher-bound field or existing leaf context")


def _verify_input_receipt(record: Mapping[str, Any], *, dataset: str,
                          section: str, artifact: Mapping[str, Any]) -> None:
    if (record.get("kind") != "acquisition" or record.get("dataset") != dataset
            or record.get("section") != section):
        raise ContractError("Source receipt belongs to a different dataset, section, or event kind")
    recorded = record.get("artifact", {})
    if (not isinstance(recorded, dict) or recorded.get("sha256") != artifact["sha256"]
            or type(recorded.get("size_bytes")) is not int
            or recorded["size_bytes"] != artifact["size_bytes"]):
        raise ContractError(f"Input bytes do not match declared provenance for {dataset}")


class DatasetCatalog:
    """Validated, read-only metadata describing sources actually used in a build."""

    def __init__(self, document: Mapping[str, Any], *, expected_build_id: str | None = None,
                 column_schema_loader: Callable[[ColumnSchemaRef], bytes] | None = None,
                 edition_metadata_loader: Callable[[EditionMetadataRef], bytes] | None = None):
        document = json.loads(canonical_json(document))
        if type(document.get("schema_version")) is int and document["schema_version"] == 4:
            self._column_schema_loader = column_schema_loader
            self._edition_metadata_loader = edition_metadata_loader
            self._init_edition_index(document, expected_build_id)
            return
        if type(document.get("schema_version")) is int and document["schema_version"] == 3:
            self._column_schema_loader = column_schema_loader
            self._init_editions(document, expected_build_id)
            return
        if document.get("schema_version") != CATALOG_SCHEMA:
            raise ContractError(
                "Unsupported dataset catalog schema; rebuild the artifact"
            )
        identity = document.get("build_id")
        content = {k: v for k, v in document.items() if k != "build_id"}
        if identity != sha256(canonical_json(content).encode()).hexdigest():
            raise ContractError("Dataset catalog identity mismatch")
        if expected_build_id is not None and identity != expected_build_id:
            raise ContractError("Dataset catalog build ID projection mismatch")
        sources = document.get("datasets")
        if not isinstance(sources, dict) or not sources:
            raise ContractError("A data release requires loaded dataset metadata")
        policies = document.get("value_policies", {})
        records = document.get("provenance", {})
        column_schemas = document.get("column_schemas", {})
        if not isinstance(column_schemas, dict):
            raise ContractError("Invalid catalog column-schema resources")
        self._column_schemas = {}
        for digest, text in column_schemas.items():
            if not isinstance(text, str) or sha256(text.encode()).hexdigest() != digest:
                raise ContractError("Catalog column-schema resource checksum mismatch")
            try:
                self._column_schemas[digest] = parse_column_schema(text.encode())
            except ValueError as error:
                raise ContractError(f"Invalid catalog column-schema resource: {error}") from error
        for identifier, record in records.items():
            if (
                sha256((canonical_json(record) + "\n").encode()).hexdigest()
                != identifier
            ):
                raise ContractError("Provenance receipt identity mismatch")
        self._metadata = {}
        self._origin_cache: dict[tuple[str, str, str, str], str] = {}
        for key, value in sources.items():
            if not isinstance(value, dict) or value.get("status") != "loaded":
                raise ContractError(f"Dataset {key} was not successfully loaded")
            metadata = DatasetMetadata.from_dict(value.get("metadata"))
            self._metadata[key] = metadata
            policy = metadata.value_policy
            if policy and policy not in policies:
                raise ContractError(
                    f"Dataset {key} refers to missing value policy {policy}"
                )
            _validate_methodology(metadata, ValuePolicy.from_dict(policies[policy]) if policy else None)
            artifact = value.get("artifact", {})
            if (
                not isinstance(artifact, dict)
                or not isinstance(artifact.get("sha256"), str)
                or len(artifact["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in artifact["sha256"])
                or type(artifact.get("size_bytes")) is not int
                or artifact["size_bytes"] < 0
            ):
                raise ContractError(f"Dataset {key} has no input checksum")
            section = value.get("section")
            if section not in {"data_sources", "spatial"}:
                raise ContractError(f"Dataset {key} has no valid source section")
            if not metadata.provenance_ids:
                raise ContractError(f"Dataset {key} has no acquisition observation")
            metadata_matched = False
            for identifier in metadata.provenance_ids:
                record = records.get(identifier, {})
                _verify_input_receipt(record, dataset=key, section=section, artifact=artifact)
                if (value.get("column_schema_sha256") is not None
                        and record.get("resolved_year") != value.get("resolved_year")):
                    raise ContractError("Header crosswalk source year differs from its acquisition receipt")
                metadata_matched |= _metadata_content(metadata) == _metadata_content(DatasetMetadata.from_dict(record.get("source_metadata")))
            if not metadata_matched:
                raise ContractError(f"Source metadata differs from its receipt: {key}")
            references = {reference.reference_id: reference for reference in metadata.references}
            for reference in references.values():
                if reference.status != "available":
                    continue
                if not reference.provenance_ids:
                    raise ContractError("Available reference has no acquisition receipt")
                for identifier in reference.provenance_ids:
                    record = records.get(identifier, {})
                    if record.get("kind") != "acquisition" or reference.url not in {
                        record.get("requested_url"), record.get("resolved_url"), record.get("source_metadata", {}).get("source_url")
                    }:
                        raise ContractError("Reference receipt belongs to a different document")
                    if reference.reviewed_sha256 is not None and record.get("artifact", {}).get("sha256") != reference.reviewed_sha256:
                        raise ContractError("Reference bytes differ from reviewed methodology")
            crosswalk_sha = value.get("column_schema_sha256")
            if crosswalk_sha is not None:
                if crosswalk_sha not in self._column_schemas or metadata.version is None:
                    raise ContractError("Header crosswalk is not bound to the selected source edition and schema reference")
                reads = value.get("reads", [])
                if not reads:
                    raise ContractError("Header crosswalk requires a recorded prepared-schema read")
                try:
                    for read in reads:
                        options = reader_options(value.get("reader", {}), read["profile"])
                        schema_references = [item for item in metadata.references
                                             if item.role == "schema" and item.path == options.get("column_schema")]
                        if (len(schema_references) != 1 or schema_references[0].status != "available"
                                or schema_references[0].reviewed_sha256 != crosswalk_sha):
                            raise ValueError("Header crosswalk differs from the selected reader profile's reviewed schema reference")
                        validate_selected_crosswalk(self._column_schemas[crosswalk_sha], dataset=key,
                                                    edition=metadata.version, input_sha256=artifact["sha256"],
                                                    columns=read["columns"])
                    crosswalk_records(self._column_schemas[crosswalk_sha], dataset=key,
                                      edition=metadata.version, metadata=metadata)
                except (KeyError, ValueError) as error:
                    raise ContractError(f"Invalid selected header crosswalk: {error}") from error
        self._build_id = identity
        self._serialized = canonical_json(document)
        self._policies = {
            key: ValuePolicy.from_dict(value) for key, value in policies.items()
        }
        self._sources = sources
        self._records = records
        self._read_columns = {
            dataset: frozenset(column for read in source.get("reads", [])
                               for column in read.get("columns", []))
            for dataset, source in sources.items()
        }
        self._field_resolutions: dict[
            tuple[str, str],
            tuple[MetricBinding, str, str, MetricDefinition | None] | None,
        ] = {}
        if "metric_bindings" not in document:
            raise ContractError("Catalog lacks release-bound metric interpretation")
        try:
            self._bindings = parse_bindings(document["metric_bindings"])
        except ValueError as error:
            raise ContractError(str(error)) from error
        for binding in self._bindings.values():
            if not isinstance(binding, MetricBinding):
                continue
            for scope in binding.scopes:
                if binding.dataset.replace("{scope}", scope) not in sources:
                    raise ContractError("Metric binding refers to a source not loaded in this release")
                if self._metadata[binding.dataset.replace("{scope}", scope)].value_policy not in self._policies:
                    raise ContractError("Metric binding requires a loaded value policy")

    def _init_editions(self, document: dict[str, Any], expected_build_id: str | None, *,
                       edition_policies: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        """Load the opt-in complete index using small, already captured metadata."""
        try:
            identity = document["build_id"]
            content = {k: v for k, v in document.items() if k != "build_id"}
            if identity != sha256(canonical_json(content).encode()).hexdigest() or (expected_build_id is not None and identity != expected_build_id):
                raise ValueError("Dataset catalog identity mismatch")
            self._records = validate_record_graph(document["provenance"].__getitem__, list(document["provenance"]))
            self._policies = {name: ValuePolicy.from_dict(value) for name, value in document["value_policies"].items()}
            self._edition_policies = ({identifier: {name: ValuePolicy.from_dict(value) for name, value in policies.items()}
                                      for identifier, policies in edition_policies.items()}
                                     if edition_policies is not None else None)
            self._column_schema_refs = {digest: ColumnSchemaRef.from_dict(value)
                                        for digest, value in document["column_schemas"].items()}
            if any(digest != ref.sha256 for digest, ref in self._column_schema_refs.items()):
                raise ValueError("Column-schema resource identity differs from its registry key")
            self._column_schemas = {}
            self._column_schema_buffers: dict[str, bytes] = {}
            self._validated_editions: set[str] = set()
            self._edition_entries = document["editions"]
            if not isinstance(self._edition_entries, dict) or not self._edition_entries:
                raise ValueError("An edition catalog requires registered editions")
            self._edition_keys: dict[str, EditionKey] = {}
            self._edition_metadata: dict[str, DatasetMetadata] = {}
            inputs: dict[str, list[str]] = {}
            for identifier, entry in self._edition_entries.items():
                key = EditionKey.from_dict(entry["key"])
                if key.identity != identifier:
                    raise ValueError("Edition key identity mismatch")
                metadata = DatasetMetadata.from_dict(entry["metadata"])
                if metadata.version != key.edition:
                    raise ValueError("Edition differs from source metadata")
                acquisition_ids = entry["acquisition_ids"]
                if not acquisition_ids or len(set(acquisition_ids)) != len(acquisition_ids):
                    raise ValueError("Edition requires distinct upstream acquisitions")
                acquired = [self._records[value] for value in acquisition_ids]
                if any(record["kind"] != "acquisition" or record.get("source_metadata", {}).get("version") != key.edition for record in acquired):
                    raise ValueError("Source revision requires acquisition receipts")
                hashes = list(dict.fromkeys(record["artifact"]["sha256"] for record in acquired))
                validate_reference_records(ResolvedDataset(key.dataset, "data_sources", None, None, "", metadata),
                                           self._records.__getitem__, verify_all_claimed=True)
                for value in metadata.provenance_ids:
                    record = self._records[value]
                    if record.get("dataset") != key.dataset or record["artifact"]["sha256"] not in hashes:
                        raise ValueError("Source provenance IDs require exact acquisition observations")
                if EditionKey.source_revision(tuple(hashes)) != key.revision_sha256:
                    raise ValueError("Source revision differs from upstream bytes")
                descriptor = self._column_schema_refs[entry["column_schema_sha256"]]
                schema_ids = [value for reference in metadata.references
                              if reference.role == "schema" and reference.reviewed_sha256 == descriptor.sha256
                              for value in reference.provenance_ids]
                if not schema_ids or any(self._records[value]["kind"] != "acquisition"
                        or self._records[value]["artifact"]["sha256"] != descriptor.sha256
                        or self._records[value]["artifact"]["size_bytes"] != descriptor.size_bytes for value in schema_ids):
                    raise ValueError("Column-schema resource lacks its genuine schema observation")
                units: dict[str, set[str]] = {}
                for metric in metadata.metrics.values():
                    if metric.unit is not None:
                        units.setdefault(metric.field, set()).add(metric.unit)
                if any(len(values) > 1 for values in units.values()):
                    raise ValueError("Conflicting canonical metric units")
                self._edition_keys[identifier], self._edition_metadata[identifier] = key, metadata
                _validate_methodology(metadata, self._policy_for_edition(identifier) if metadata.value_policy else None)
                inputs[identifier] = hashes
            if set(self._column_schema_refs) != {entry["column_schema_sha256"] for entry in self._edition_entries.values()}:
                raise ValueError("Column-schema registry must account for exactly the registered editions")
            self._edition_inputs = inputs
            self._artifact_refs: dict[str, ArtifactRef] = {}
            self._artifact_keys: dict[str, tuple[str, ...]] = {}
            artifact_columns: dict[str, set[str]] = {}
            for identifier, value in document["artifacts"].items():
                ref = ArtifactRef.from_dict(value)
                edition = ref.edition.identity
                contract = ref.contract
                if identifier != ref.identity or self._edition_keys[edition] != ref.edition:
                    raise ValueError("Artifact contract identity mismatch")
                metadata = self._edition_metadata[edition]
                policy_sha = (sha256(canonical_json(self._policy_for_edition(edition).to_dict()).encode()).hexdigest()
                              if metadata.value_policy else None)
                if (contract["policy_sha256"] != policy_sha or not set(inputs[edition]) <= set(contract["inputs"])):
                    raise ValueError("Leaf policy or inputs differ from selected edition")
                record = self._records[ref.processing_id]
                output = record.get("outputs", {}).get(ref.output, {})
                if (record["kind"] != "processing" or record["code_version"] != contract["producer"]["code_sha256"]
                        or record.get("parameters", {}).get("leaf_contract_sha256") != identifier
                        or sha256(canonical_json(record.get("parameters", {}).get("recipe")).encode()).hexdigest() != contract["recipe_sha256"]
                        or record.get("parameters", {}).get("toolchain") != contract["producer"]["toolchain"]
                        or any(output.get(name) != getattr(ref, name) for name in ("sha256", "size_bytes"))):
                    raise ValueError("Leaf does not match its processing output")
                graph = validate_record_graph(self._records.__getitem__, record["inputs"])
                acquired_hashes = {artifact["sha256"] for item in graph.values()
                    for artifact in ([item["artifact"]] if item["kind"] == "acquisition" else item["outputs"].values())}
                if not set(contract["inputs"]) <= acquired_hashes:
                    raise ValueError("Leaf derivation does not close its physical inputs")
                self._artifact_refs[identifier] = ref
                self._artifact_keys[identifier] = tuple(contract["key_columns"])
                artifact_columns[identifier] = set(ref.public_columns)
            expected_fields = {(ref.edition.identity, column)
                               for ref in self._artifact_refs.values() for column in ref.public_columns}
            self._field_bindings: dict[tuple[str, str], FieldBinding] = {}
            for value in document["field_bindings"]:
                binding = FieldBinding.from_dict(value)
                qualified = binding.edition_id, binding.field
                ref = self._artifact_refs[binding.artifact_id]
                if (qualified in self._field_bindings or qualified not in expected_fields
                        or ref.edition.identity != binding.edition_id or binding.column != binding.field
                        or binding.column not in artifact_columns[binding.artifact_id]):
                    raise ValueError("Duplicate, foreign or invalid qualified field binding")
                self._field_bindings[qualified] = binding
            if set(self._field_bindings) != expected_fields:
                raise ValueError("Edition index does not account for every registered field")
            visiting: set[tuple[str, str]] = set()
            visited: set[tuple[str, str]] = set()

            def visit(qualified: tuple[str, str]) -> None:
                if qualified in visiting:
                    raise ValueError("Field dependencies contain a cycle")
                if qualified not in visited:
                    visiting.add(qualified)
                    for dependency in self._field_bindings[qualified].dependencies:
                        visit(dependency)
                    visiting.remove(qualified)
                    visited.add(qualified)
            for qualified in self._field_bindings:
                visit(qualified)
            self._field_headers: dict[tuple[str, str], dict[str, Any]] = {}
            self._build_id, self._serialized = identity, canonical_json(document)
            self._v3 = True
            self._field_resolutions = {}
            self._qualified_fields: dict[tuple[str, str], dict[str, Any]] = {}
            self._origin_cache = {}
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError(f"Invalid edition catalog: {error}") from error

    def _policy_for_edition(self, identifier: str) -> ValuePolicy:
        name = self._edition_metadata[identifier].value_policy
        if name is None:
            raise ValueError("Edition has no value policy")
        policies = self._edition_policies[identifier] if self._edition_policies is not None else self._policies
        return policies[name]

    def _init_edition_index(self, document: dict[str, Any], expected_build_id: str | None) -> None:
        """Validate the small v4 index without opening any child or data file."""
        try:
            if set(document) != {"schema_version", "build_id", "editions", "artifacts", "description"}:
                raise ValueError("Unsupported edition index fields")
            content = {key: value for key, value in document.items() if key != "build_id"}
            identity = sha256(canonical_json(content).encode()).hexdigest()
            if document["build_id"] != identity or (expected_build_id is not None and expected_build_id != identity):
                raise ValueError("Dataset catalog identity mismatch")
            if not isinstance(document["editions"], dict) or not document["editions"]:
                raise ValueError("An edition index requires registered editions")
            self._edition_keys, self._edition_metadata_refs = {}, {}
            for identifier, entry in document["editions"].items():
                if set(entry) != {"key", "metadata"}:
                    raise ValueError("Unsupported edition index entry")
                key = EditionKey.from_dict(entry["key"])
                if key.identity != identifier:
                    raise ValueError("Edition key identity mismatch")
                self._edition_keys[identifier] = key
                self._edition_metadata_refs[identifier] = EditionMetadataRef.from_dict(entry["metadata"])
            self._artifact_routes = document["artifacts"]
            if not isinstance(self._artifact_routes, dict) or not self._artifact_routes:
                raise ValueError("An edition index requires artifact routes")
            for identifier, edition in self._artifact_routes.items():
                if (not isinstance(identifier, str) or len(identifier) != 64
                        or any(c not in "0123456789abcdef" for c in identifier)
                        or edition not in self._edition_keys):
                    raise ValueError("Invalid artifact route")
            if set(self._artifact_routes.values()) != set(self._edition_keys):
                raise ValueError("Every edition requires an artifact route")
            self._edition_buffers: dict[str, bytes] = {}
            self._edition_children: dict[str, dict[str, Any]] = {}
            self._edition_views: dict[str, DatasetCatalog] = {}
            self._build_id, self._serialized = identity, canonical_json(document)
            self._v3, self._v4 = True, True
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError(f"Invalid edition index: {error}") from error

    def edition_metadata_refs(self) -> tuple[EditionMetadataRef, ...]:
        """Index-only child descriptors; no child or native leaf I/O."""
        if not getattr(self, "_v4", False):
            raise ContractError("Edition metadata resources require a schema4 catalog")
        return tuple(self._edition_metadata_refs.values())

    def edition_metadata_bytes(self, identifier: str) -> bytes:
        """Reverify the selected child's exact bounded bytes, including on warm use."""
        try:
            ref = self._edition_metadata_refs[identifier]
            if self._edition_metadata_loader is None:
                raise ValueError("Selected edition requires an explicit metadata resolver")
            raw = self._edition_metadata_loader(ref)
            if type(raw) is not bytes or len(raw) != ref.size_bytes:
                raise ValueError("Edition metadata resource byte size mismatch")
            if self._edition_buffers.get(identifier) is not raw:
                if sha256(raw).hexdigest() != ref.sha256:
                    raise ValueError("Edition metadata resource checksum mismatch")
                self._edition_buffers[identifier] = raw
            return raw
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise ContractError(f"Invalid selected edition metadata: {error}") from error

    def _edition_child(self, identifier: str) -> dict[str, Any]:
        raw = self.edition_metadata_bytes(identifier)
        if identifier not in self._edition_children:
            try:
                child = json.loads(raw)
                if (set(child) != {"schema_version", "edition", "artifacts", "field_bindings", "column_schemas",
                                   "provenance", "value_policies"} or type(child["schema_version"]) is not int
                        or child["schema_version"] != 4 or child["edition"]["key"] != self._edition_keys[identifier].to_dict()
                        or set(child["edition"]) != {"key", "metadata", "acquisition_ids", "column_schema_sha256"}
                        or set(child["artifacts"]) != {key for key, edition in self._artifact_routes.items() if edition == identifier}
                        or any(value["edition_id"] != identifier for value in child["field_bindings"])
                        or any(value["edition"] != child["edition"]["key"] for value in child["artifacts"].values())):
                    raise ValueError("Edition child differs from its exact index entry or artifact routes")
                self._edition_children[identifier] = child
            except (KeyError, TypeError, ValueError) as error:
                raise ContractError(f"Invalid edition metadata child: {error}") from error
        return self._edition_children[identifier]

    def _edition_view(self, identifier: str) -> DatasetCatalog:
        """Close selected foreign dependencies, then run the same catalog validator."""
        children: dict[str, dict[str, Any]] = {}
        def visit(selected):
            if selected not in children:
                child = children[selected] = self._edition_child(selected)
                for value in child["field_bindings"]:
                    for edition, _field in FieldBinding.from_dict(value).dependencies:
                        visit(edition)
        try:
            visit(identifier)
            if identifier not in self._edition_views:
                combined: dict[str, Any] = {"schema_version": 3, "editions": {}, "artifacts": {}, "field_bindings": [],
                                            "column_schemas": {}, "provenance": {}, "value_policies": {}, "description": ""}
                policies = {}
                for selected, child in children.items():
                    combined["editions"][selected] = child["edition"]
                    policies[selected] = child["value_policies"]
                    combined["field_bindings"].extend(child["field_bindings"])
                    for section in ("artifacts", "column_schemas", "provenance"):
                        for key, value in child[section].items():
                            if key in combined[section] and combined[section][key] != value:
                                raise ValueError("Conflicting metadata identity across edition children")
                            combined[section][key] = value
                combined["build_id"] = sha256(canonical_json(combined).encode()).hexdigest()
                view = object.__new__(DatasetCatalog)
                view._column_schema_loader = self._column_schema_loader
                view._init_editions(combined, None, edition_policies=policies)
                self._edition_views[identifier] = view
            view = self._edition_views[identifier]
            self._check_metadata_locations((*self.edition_metadata_refs(), *view.column_schema_refs(), *view._artifact_refs.values()))
            return view
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError(f"Invalid selected edition dependency closure: {error}") from error

    @staticmethod
    def _check_metadata_locations(refs) -> None:
        """Selected metadata may not route different bytes to one portable name."""
        locations = {("local", "release.json"): None}
        for ref in refs:
            location = ref.location
            key = (("url", location) if "://" in location else
                   ("local", unicodedata.normalize("NFC", location).casefold().rstrip(" .")))
            identity = ref.sha256, ref.size_bytes
            if not key[1] or "\0" in location or (key in locations and locations[key] != identity):
                raise ContractError("Edition metadata and artifact destinations collide")
            locations[key] = identity

    def externalize_editions(self) -> DatasetCatalog:
        """Explicitly migrate validated v3 metadata to v4; preserve every native identity.

        This constructs metadata in memory. It never publishes resources or opens
        native leaves, and never invents missing schema observations/references.
        """
        self._require_index()
        if getattr(self, "_v4", False):
            raise ContractError("Externalization requires an explicit schema3 catalog")
        self.verify_metadata()
        children, entries, accounted = {}, {}, set()
        for identifier, entry in self._edition_entries.items():
            artifacts = {key: ref.to_dict() for key, ref in self._artifact_refs.items() if ref.edition.identity == identifier}
            roots = list(entry["acquisition_ids"]) + list(self._edition_metadata[identifier].provenance_ids)
            roots.extend(value for reference in self._edition_metadata[identifier].references for value in reference.provenance_ids)
            roots.extend(ref["processing_id"] for ref in artifacts.values())
            records = validate_record_graph(self._records.__getitem__, list(dict.fromkeys(roots)))
            accounted.update(records)
            name = self._edition_metadata[identifier].value_policy
            child = {"schema_version": 4, "edition": deepcopy(entry), "artifacts": artifacts,
                     "field_bindings": [binding.to_dict() for (edition, _), binding in self._field_bindings.items() if edition == identifier],
                     "provenance": records, "value_policies": {name: self._policies[name].to_dict()} if name else {},
                     "column_schemas": {entry["column_schema_sha256"]: self._column_schema_refs[entry["column_schema_sha256"]].to_dict()}}
            raw = (canonical_json(child) + "\n").encode()
            digest = sha256(raw).hexdigest()
            descriptor = EditionMetadataRef(digest, len(raw), digest + ".json")
            children[digest] = raw
            entries[identifier] = {"key": entry["key"], "metadata": descriptor.to_dict()}
        if accounted != set(self._records):
            raise ContractError("Explicit metadata migration must account for every original provenance record")
        content = {"schema_version": 4, "editions": entries,
                   "artifacts": {key: ref.edition.identity for key, ref in self._artifact_refs.items()},
                   "description": self.to_dict()["description"]}
        return DatasetCatalog({**content, "build_id": sha256(canonical_json(content).encode()).hexdigest()},
            column_schema_loader=self._column_schema_loader, edition_metadata_loader=lambda ref: children[ref.sha256])

    def _publication_metadata(self) -> tuple[tuple[EditionMetadataRef | ColumnSchemaRef, ...], tuple[ArtifactRef, ...]]:
        """Complete descriptor preflight, before crosswalk or native data resolution."""
        refs: dict[str, EditionMetadataRef | ColumnSchemaRef] = {ref.sha256: ref for ref in self.edition_metadata_refs()}
        artifacts = {}
        for identifier in self._edition_keys:
            view = self._edition_view(identifier)
            for ref in view.column_schema_refs():
                if ref.sha256 in refs and refs[ref.sha256].to_dict() != ref.to_dict():
                    raise ContractError("Conflicting metadata resource identity")
                refs[ref.sha256] = ref
            for key, ref in view._artifact_refs.items():
                artifacts[key] = ref
        self._check_metadata_locations((*refs.values(), *artifacts.values()))
        return tuple(refs.values()), tuple(artifacts.values())

    @classmethod
    def create_editions(cls, *, editions: Mapping[str, Any], artifacts: Mapping[str, Any],
                        field_bindings: Iterable[Mapping[str, Any]], column_schemas: Mapping[str, Any],
                        records: Mapping[str, Any], value_policies: Mapping[str, Any], description: str = "",
                        column_schema_loader: Callable[[ColumnSchemaRef], bytes] | None = None,
                        schema_version: int = 3) -> DatasetCatalog:
        """Explicit unpublished v3 construction; the default writer stays v2."""
        content = {"schema_version": 3, "editions": dict(editions), "artifacts": dict(artifacts),
                   "field_bindings": list(field_bindings), "column_schemas": dict(column_schemas),
                   "provenance": dict(records), "value_policies": dict(value_policies), "description": description}
        catalog = cls({**content, "build_id": sha256(canonical_json(content).encode()).hexdigest()},
                      column_schema_loader=column_schema_loader)
        if type(schema_version) is not int or schema_version not in (3, 4):
            raise ContractError("Select explicit schema3 or schema4 edition construction")
        return catalog.externalize_editions() if schema_version == 4 else catalog

    def column_schema_refs(self) -> tuple[ColumnSchemaRef, ...]:
        """All schema descriptors; v4 explicitly reads all child metadata, never leaves."""
        self._require_index()
        if getattr(self, "_v4", False):
            refs = {}
            for identifier in self._edition_keys:
                for digest, value in self._edition_child(identifier)["column_schemas"].items():
                    ref = ColumnSchemaRef.from_dict(value)
                    if digest != ref.sha256 or (digest in refs and refs[digest] != ref):
                        raise ContractError("Conflicting column-schema resource identity")
                    refs[digest] = ref
            return tuple(refs.values())
        return tuple(self._column_schema_refs.values())

    def column_schema_bytes(self, digest: str) -> bytes:
        """Resolve and verify the exact bounded bytes that the parser consumes."""
        self._require_index()
        if self._column_schema_loader is None:
            raise ContractError("Selected metadata requires an explicit column-schema resolver")
        if getattr(self, "_v4", False):
            try:
                ref = next(ref for ref in self.column_schema_refs() if ref.sha256 == digest)
                raw = self._column_schema_loader(ref)
                if type(raw) is not bytes or len(raw) != ref.size_bytes or sha256(raw).hexdigest() != digest:
                    raise ContractError("Column-schema resource checksum or size mismatch")
                return raw
            except StopIteration as error:
                raise ContractError("Unknown column-schema resource") from error
        try:
            descriptor = self._column_schema_refs[digest]
            raw = self._column_schema_loader(descriptor)
            if type(raw) is not bytes or len(raw) != descriptor.size_bytes:
                raise ValueError("Column-schema resource byte size mismatch")
            if self._column_schema_buffers.get(digest) is not raw:
                if sha256(raw).hexdigest() != digest:
                    raise ValueError("Column-schema resource checksum mismatch")
                self._column_schema_buffers[digest] = raw
            return raw
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise ContractError(f"Invalid selected column-schema resource: {error}") from error

    def _ensure_edition(self, identifier: str) -> None:
        """Validate crosswalk semantics only for explicitly selected metadata."""
        entry = self._edition_entries[identifier]
        digest = entry["column_schema_sha256"]
        raw = self.column_schema_bytes(digest)  # Resolver rechecks a cached file's stat identity.
        if identifier in self._validated_editions:
            return
        try:
            if digest not in self._column_schemas:
                self._column_schemas[digest] = parse_column_schema(raw)
            document = self._column_schemas[digest]
            from .column_schema import crosswalk_dataset
            key, metadata = self._edition_keys[identifier], self._edition_metadata[identifier]
            table, = [value for value in crosswalk_dataset(document, key.dataset, key.edition)["tables"]
                      if value["table"] == key.table]
            table_input_sha256s(table, acquisition_sha256s=self._edition_inputs[identifier])
            rows = crosswalk_records(document, dataset=key.dataset, edition=key.edition,
                                     table=key.table, metadata=metadata)
            if {row["field"] for row in rows} != {field for edition, field in self._field_bindings if edition == identifier}:
                raise ValueError("Edition index differs from the complete selected crosswalk")
            semantics = _leaf_semantics(metadata, rows)
            mapping = sha256(canonical_json(table).encode()).hexdigest()
            for ref in self._artifact_refs.values():
                if ref.edition.identity == identifier:
                    if (ref.contract["mapping_sha256"] != mapping or ref.contract["semantics_sha256"] != semantics):
                        raise ValueError("Leaf mapping or semantics differ from selected edition")
                    _validate_leaf_support(ref.contract, rows, metadata)
            units = {metric.field: metric.unit for metric in metadata.metrics.values() if metric.unit is not None}
            headers = {(identifier, row["field"]): {
                **{name: row[name] for name in ("code", "label", "period")}, "unit": units.get(row["field"])} for row in rows}
            self._field_headers.update(headers)
            self._validated_editions.add(identifier)
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError(f"Invalid selected edition crosswalk: {error}") from error

    def _ensure_field_dependencies(self, fields: Iterable[tuple[str, str]], *,
                                   catalog_for: Callable[[str], DatasetCatalog] | None = None,
                                   visited: set[tuple[str, str]] | None = None,
                                   editions: set[str] | None = None) -> None:
        visited = set() if visited is None else visited
        editions = set() if editions is None else editions
        def visit(qualified: tuple[str, str]) -> None:
            if qualified not in visited:
                visited.add(qualified)
                catalog = self if catalog_for is None else catalog_for(qualified[0])
                if qualified[0] not in editions:
                    catalog._ensure_edition(qualified[0])
                    editions.add(qualified[0])
                for dependency in catalog._field_bindings[qualified].dependencies:
                    visit(dependency)
        for qualified in fields:
            visit(qualified)

    def verify_metadata(self) -> None:
        """Explicitly validate every registered edition's external metadata."""
        self._require_index()
        if getattr(self, "_v4", False):
            self._publication_metadata()
        for identifier in self._edition_keys:
            if getattr(self, "_v4", False):
                self._edition_view(identifier).verify_metadata()
            else:
                self._ensure_edition(identifier)

    def _require_index(self) -> None:
        if not getattr(self, "_v3", False):
            raise ContractError("Complete edition index unavailable in this schema2 catalog")

    def editions(self, *, dataset: str | None = None, scope: str | None = None, include_metadata: bool = True):
        """Yield detached editions. False is index-only; True may read child metadata.

        No child is prefetched before iteration, and neither mode opens native leaves.
        """
        self._require_index()
        if type(include_metadata) is not bool:
            raise ContractError("include_metadata must be an explicit boolean")
        for identifier, key in self._edition_keys.items():
            if (dataset is None or key.dataset == dataset) and (scope is None or key.scope == scope):
                row = {"edition_id": identifier, **key.to_dict()}
                if include_metadata:
                    row["metadata"] = self.metadata(key.dataset, edition=key).to_dict()
                yield row

    def _selected_edition(self, dataset: str, edition: EditionKey | None) -> str:
        self._require_index()
        candidates = [identifier for identifier, key in self._edition_keys.items()
                      if key.dataset == dataset and (edition is None or key == edition)]
        if len(candidates) != 1:
            raise ContractError("Exact edition selection is missing or ambiguous")
        return candidates[0]

    def variables(self, *, edition: EditionKey | None = None, dataset: str | None = None, scope: str | None = None):
        """Yield the complete registered field projection, not a new dictionary."""
        self._require_index()
        if getattr(self, "_v4", False):
            for row in self.editions(dataset=dataset, scope=scope, include_metadata=False):
                key = self._edition_keys[row["edition_id"]]
                if edition is None or edition == key:
                    yield from self._edition_view(key.identity).variables(edition=key)
            return
        validated: set[str] = set()
        for qualified, binding in self._field_bindings.items():
            key = self._edition_keys[qualified[0]]
            if ((edition is None or edition == key) and (dataset is None or key.dataset == dataset)
                    and (scope is None or key.scope == scope)):
                if qualified[0] not in validated:
                    self._ensure_edition(qualified[0])
                    validated.add(qualified[0])
                yield self._variable(qualified)

    def _variable(self, qualified: tuple[str, str]) -> dict[str, Any]:
        binding, row = self._field_bindings[qualified], self._field_headers[qualified]
        result: dict[str, Any] = {**binding.to_dict(), "edition": self._edition_keys[qualified[0]].to_dict(),
                "code": row["code"], "label": row["label"], "period": deepcopy(row["period"]),
                "unit": row["unit"], "policy_id": self._edition_metadata[qualified[0]].value_policy}
        support = self._artifact_refs[binding.artifact_id].column_support
        if support is not None:
            result["support"] = support.projection(binding.field)
        qualifiers = self._edition_metadata[qualified[0]].source_qualifiers
        if qualifiers is not None and qualifiers.row_run_column is not None:
            result.setdefault("support", {})["row_run_column"] = qualifiers.row_run_column
        return result

    def artifact_ref(self, identifier: str) -> ArtifactRef:
        self._require_index()
        if getattr(self, "_v4", False):
            try:
                return self._edition_view(self._artifact_routes[identifier]).artifact_ref(identifier)
            except KeyError as error:
                raise ContractError("Unknown selected artifact ID") from error
        try:
            descriptor = self._artifact_refs[identifier]
            self._ensure_edition(descriptor.edition.identity)
            self._ensure_field_dependencies(qualified for qualified, binding in self._field_bindings.items()
                                            if binding.artifact_id == identifier)
            return descriptor
        except KeyError as error:
            raise ContractError("Unknown selected artifact ID") from error

    def resolve_artifacts(self, fields: Iterable[tuple[EditionKey, str]], *, period: DataPeriod | None = None) -> tuple[dict[str, Any], ...]:
        """Resolve exact fields/dependencies; only selected metadata may need I/O."""
        self._require_index()
        columns: dict[str, list[str]] = {}
        column_sets: dict[str, set[str]] = {}
        artifacts: dict[str, ArtifactRef] = {}
        visited: set[tuple[str, str]] = set()
        validated_fields: set[tuple[str, str]] = set()
        validated_editions: set[str] = set()
        catalogs: dict[str, DatasetCatalog] = {}

        def catalog_for(identifier: str) -> DatasetCatalog:
            if not getattr(self, "_v4", False):
                return self
            if identifier not in catalogs:
                view = self._edition_view(identifier)
                for selected in view._edition_keys:
                    catalogs.setdefault(selected, view)
            return catalogs[identifier]

        def visit(qualified: tuple[str, str]) -> None:
            if qualified in visited:
                return
            visited.add(qualified)
            try:
                catalog = catalog_for(qualified[0])
                binding = catalog._field_bindings[qualified]
            except KeyError as error:
                raise ContractError("Qualified field unavailable") from error
            identifier = binding.artifact_id
            if identifier not in columns:
                artifacts[identifier] = catalog._artifact_refs[identifier]
                columns[identifier] = list(catalog._artifact_keys[identifier])
                column_sets[identifier] = set(columns[identifier])
            if binding.column not in column_sets[identifier]:
                columns[identifier].append(binding.column)
                column_sets[identifier].add(binding.column)
            support = artifacts[identifier].column_support
            if support is not None:
                projection = support.projection(binding.field)
                for column in [*(projection[name] for name in ("raw", "status") if name in projection),
                               *projection.get("auxiliary", ()), *projection["context"]]:
                    if column not in column_sets[identifier]:
                        columns[identifier].append(column)
                        column_sets[identifier].add(column)
            qualifiers = catalog._edition_metadata[qualified[0]].source_qualifiers
            if qualifiers is not None and qualifiers.row_run_column is not None:
                column = qualifiers.row_run_column
                if column not in column_sets[identifier]:
                    columns[identifier].append(column)
                    column_sets[identifier].add(column)
            for dependency in binding.dependencies:
                visit(dependency)
        for key, field in fields:
            qualified = key.identity, field
            if self._edition_keys.get(key.identity) != key:
                raise ContractError("Exact qualified field unavailable")
            catalog = catalog_for(qualified[0])
            if qualified not in catalog._field_bindings:
                raise ContractError("Exact qualified field unavailable")
            self._ensure_field_dependencies([qualified], catalog_for=catalog_for,
                                            visited=validated_fields, editions=validated_editions)
            actual_period = DataPeriod.from_dict(catalog._field_headers[qualified]["period"])
            if period is not None and (period.kind == "unknown" or actual_period != period):
                raise ContractError("Requested reference period is unsupported")
            visit(qualified)
        # Keep revalidation per operation: even a lazy input iterable may change
        # a resource between requested fields. Recheck bytes/stat, not the graph.
        for identifier in catalogs:
            self.edition_metadata_bytes(identifier)
        checked_schemas: set[ColumnSchemaRef] = set()
        for identifier in validated_editions:
            catalog = catalog_for(identifier)
            digest = catalog._edition_entries[identifier]["column_schema_sha256"]
            descriptor = catalog._column_schema_refs[digest]
            if descriptor not in checked_schemas:
                catalog.column_schema_bytes(digest)
                checked_schemas.add(descriptor)
        return tuple({"artifact": artifacts[identifier], "columns": tuple(selected)}
                     for identifier, selected in columns.items())

    @classmethod
    def create(
        cls,
        *,
        datasets: Mapping[str, Any],
        value_policies: Mapping[str, Any],
        processor: str,
        records: Mapping[str, Any] | None = None,
        environment: Mapping[str, Any] | None = None,
        metric_bindings: Mapping[str, Any] | None = None,
        column_schemas: Mapping[str, str] | None = None,
    ) -> DatasetCatalog:
        content = {
            "schema_version": CATALOG_SCHEMA,
            "processor_sha256": processor,
            "datasets": dict(datasets),
            "value_policies": dict(value_policies),
            "provenance": dict(records or {}),
            "build_environment": dict(environment or {}),
            "metric_bindings": dict(metric_bindings or {}),
        }
        if column_schemas:
            content["column_schemas"] = dict(column_schemas)
        return cls(
            {
                **content,
                "build_id": sha256(canonical_json(content).encode()).hexdigest(),
            }
        )

    @property
    def build_id(self) -> str:
        return self._build_id

    @property
    def datasets(self) -> dict[str, Any]:
        if getattr(self, "_v4", False):
            return {identifier: deepcopy(self._edition_view(identifier)._edition_entries[identifier])
                    for identifier in self._edition_keys}
        if getattr(self, "_v3", False):
            return deepcopy(self._edition_entries)
        return deepcopy(self._sources)

    def __getstate__(self) -> dict[str, Any]:
        if getattr(self, "_v3", False):
            return {"_v3": True, "_serialized": self._serialized}
        # Derived record lookup must not change the established snapshot state.
        return {key: value for key, value in self.__dict__.items() if key != "_records"}

    def __setstate__(self, state: dict[str, Any]) -> None:
        if state.get("_v3") is True:
            self.__init__(json.loads(state["_serialized"]))
            return
        self.__dict__.update(state)
        self._records = json.loads(self._serialized)["provenance"]

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._serialized)

    def _scalar_metadata(self, dataset: str) -> DatasetMetadata:
        try:
            return self._metadata[dataset]
        except KeyError as exc:
            raise ContractError(f"Dataset was not loaded in this release: {dataset}") from exc

    def metadata(self, dataset: str, *, edition: EditionKey | None = None) -> DatasetMetadata:
        if getattr(self, "_v4", False):
            identifier = self._selected_edition(dataset, edition)
            return self._edition_view(identifier).metadata(dataset, edition=self._edition_keys[identifier])
        if getattr(self, "_v3", False):
            return DatasetMetadata.from_dict(self._edition_entries[self._selected_edition(dataset, edition)]["metadata"])
        if edition is not None:
            raise ContractError("Qualified edition selection requires a schema3 catalog")
        try:
            return DatasetMetadata.from_dict(self._sources[dataset]["metadata"])
        except KeyError as exc:
            raise ContractError(
                f"Dataset was not loaded in this release: {dataset}"
            ) from exc

    def header_records(self, dataset: str, *, edition: str | None = None,
                       table: str | None = None, edition_key: EditionKey | None = None) -> list[dict[str, Any]]:
        """Return detached code/label/field records from this release's bound resource.

        Schema3 resolves only the selected external metadata resource. Legacy
        catalogs remain entirely inline; neither path reads source rows or YAML.
        """
        if getattr(self, "_v3", False):
            candidates = [identifier for identifier, key in self._edition_keys.items()
                          if key.dataset == dataset and (edition_key is None or key == edition_key)
                          and (edition is None or key.edition == edition) and (table is None or key.table == table)]
            if len(candidates) != 1:
                raise ContractError("Header crosswalk edition is missing or ambiguous")
            identifier = candidates[0]
            key = self._edition_keys[identifier]
            if getattr(self, "_v4", False):
                return self._edition_view(identifier).header_records(dataset, edition_key=key)
            self._ensure_edition(identifier)
            return crosswalk_records(self._column_schemas[self._edition_entries[identifier]["column_schema_sha256"]],
                dataset=key.dataset, edition=key.edition, table=key.table, metadata=self._edition_metadata[identifier])
        if edition_key is not None:
            raise ContractError("Qualified edition selection requires a schema3 catalog")
        metadata = self.metadata(dataset)
        selected = metadata.version
        if edition is not None and edition != selected:
            raise ContractError(f"Header crosswalk edition unavailable: {edition!r}; selected {selected!r}")
        digest = self._sources[dataset].get("column_schema_sha256")
        if digest is None or selected is None:
            raise ContractError(f"Header crosswalk unavailable for selected dataset {dataset!r}")
        try:
            return crosswalk_records(self._column_schemas[digest], dataset=dataset, edition=selected,
                                     table=table, metadata=metadata)
        except ValueError as error:
            raise ContractError(str(error)) from error

    def header_mapping(self, dataset: str, *, table: str, edition: str | None = None,
                       from_: str = "code", to: str = "field",
                       columns: Iterable[str] | None = None, edition_key: EditionKey | None = None) -> dict[str, str]:
        """Return a plain rename map; optional external headers detect collisions."""
        try:
            return header_rename_map(self.header_records(dataset, edition=edition, table=table, edition_key=edition_key),
                                     from_=from_, to=to, columns=columns)
        except ValueError as error:
            raise ContractError(str(error)) from error

    def bound_field_headers(self, fields: Iterable[str], *, scope: str) -> dict[str, dict[str, str]]:
        """Return reviewed publisher headers for exact snapshot field bindings."""
        self._require_scalar_interpretation()
        tables: dict[str, dict[str, dict[str, str]]] = {}
        headers = {}
        for field in fields:
            binding = self._resolve_field(field, scope)
            if binding is None:
                continue
            _, dataset, source_field, _ = binding
            if not self._sources[dataset].get("column_schema_sha256"):
                continue
            if dataset not in tables:
                by_field: dict[str, dict[str, str]] = {}
                for record in self.header_records(dataset):
                    name = record["field"]
                    if name in by_field:
                        raise ContractError(f"Publisher header is ambiguous for {name!r} in {dataset!r}")
                    by_field[name] = {key: record[key] for key in ("dataset", "edition", "table", "label", "code")}
                tables[dataset] = by_field
            if source_field in tables[dataset]:
                headers[field] = tables[dataset][source_field]
        return headers

    def acquisitions(self, dataset: str, *, edition: EditionKey | None = None) -> list[dict[str, Any]]:
        """Return immutable acquisition observations, including unknown download dates."""
        if getattr(self, "_v4", False):
            identifier = self._selected_edition(dataset, edition)
            return self._edition_view(identifier).acquisitions(dataset, edition=self._edition_keys[identifier])
        if getattr(self, "_v3", False):
            return [deepcopy({"record_id": identifier, **self._records[identifier]})
                    for identifier in self._edition_entries[self._selected_edition(dataset, edition)]["acquisition_ids"]]
        if edition is not None:
            raise ContractError("Qualified edition selection requires a schema3 catalog")
        return [
            deepcopy({"record_id": identifier, **self._records[identifier]})
            for identifier in self.metadata(dataset).provenance_ids
        ]

    def period_label(self, dataset: str, *, short: bool = False) -> str:
        period = self.metadata(dataset).period
        if period.label is None:
            return "Period not specified"
        if short and period.start_year is not None:
            return f"{period.start_year}-{str(period.end_year)[-2:]}"
        return period.label

    def field_year(self, dataset: str) -> int:
        metadata = self.metadata(dataset) if getattr(self, "_v3", False) else self._scalar_metadata(dataset)
        year = metadata.field_year
        if year is None:
            raise ContractError(f"Dataset {dataset} has no declared field year")
        return year

    def policy(self, dataset: str, *, edition: EditionKey | None = None) -> ValuePolicy:
        if getattr(self, "_v4", False):
            identifier = self._selected_edition(dataset, edition)
            return self._edition_view(identifier).policy(dataset, edition=self._edition_keys[identifier])
        metadata = (self.metadata(dataset, edition=edition) if getattr(self, "_v3", False) or edition is not None
                    else self._scalar_metadata(dataset))
        policy_id = metadata.value_policy
        if policy_id is None:
            raise ContractError(f"Dataset {dataset} has no value policy")
        policy = self._policy_for_edition(self._selected_edition(dataset, edition)) if getattr(self, "_v3", False) else self._policies[policy_id]
        return ValuePolicy.from_dict(policy.to_dict())

    def _resolve_metric(self, metric: str, scope: str) -> tuple[MetricBinding, str, str, MetricDefinition | None]:
        self._require_scalar_interpretation()
        if scope not in {"campus", "district"}:
            raise ContractError(f"Unknown entity scope: {scope}")
        selected = metric if metric in self._bindings else next((
            key for key in sorted(self._bindings, key=lambda item: len(item), reverse=True)
            if key.endswith(".*") and metric.startswith(key[:-1])
        ), None)
        if selected is None:
            raise ContractError(f"Unknown canonical metric: {metric}")
        suffix = metric[len(selected) - 1:] if selected.endswith(".*") else ""
        binding = self._bindings[selected]
        while isinstance(binding, str):
            binding = self._bindings[binding]
        if scope not in binding.scopes:
            raise ContractError(f"Metric {metric} is not defined for {scope}")
        dataset = binding.dataset.replace("{scope}", scope)
        selector = binding.field.replace("{metric}", suffix)
        metadata = self._metadata[dataset]
        definition = metadata.metrics.get(selector)
        key = self.field(dataset, selector, scope=scope) if binding.field_mode == "metric" else selector
        if definition is None:
            definition = next((value for value in metadata.metrics.values() if value.field == key), None)
        if key not in self._read_columns[dataset]:
            raise ContractError(f"Metric {metric} is not a declared field in this release")
        return binding, dataset, key, definition

    def source_for_metric(self, metric: str, scope: str) -> tuple[str, str]:
        _, dataset, key, _ = self._resolve_metric(metric, scope)
        return dataset, key

    def _field_metadata(self, binding: MetricBinding, dataset: str, key: str,
                        definition: MetricDefinition | None) -> dict[str, Any]:
        metadata = self._metadata[dataset]
        period = definition.period if definition and definition.period is not None else metadata.period
        return {
            "dataset": dataset, "field": key, "period": period.to_dict(),
            "field_year": metadata.field_year, "version": metadata.version,
            "unit": self._metric_unit(binding, key, definition),
            "population": binding.population, "policy_id": metadata.value_policy,
            "metadata": metadata.to_dict(), "acquisitions": self.acquisitions(dataset),
        }

    def field_metadata(self, field: str, *, scope: str = "campus", edition: EditionKey | None = None) -> dict[str, Any] | None:
        """Describe a known source field without interpreting its values.

        Opaque text can still have known provenance. A null result means this
        catalog cannot establish the field's source; names are never guessed.
        Returned metadata is detached from the catalog's immutable authority.
        """
        if getattr(self, "_v4", False):
            found = []
            for row in self.editions(scope=scope, include_metadata=False):
                key = self._edition_keys[row["edition_id"]]
                if edition is None or edition == key:
                    value = self._edition_view(key.identity).field_metadata(field, scope=scope, edition=key)
                    if value is not None:
                        found.append(value)
            if len(found) > 1:
                raise ContractError("Qualified field is ambiguous; select an exact EditionKey")
            return found[0] if found else None
        if getattr(self, "_v3", False):
            candidates = [(identifier, field) for identifier, key in self._edition_keys.items()
                          if key.scope == scope and (edition is None or key == edition)
                          and (identifier, field) in self._field_bindings]
            if len(candidates) > 1:
                raise ContractError("Qualified field is ambiguous; select an exact EditionKey")
            if not candidates:
                return None
            qualified = candidates[0]
            self._ensure_edition(qualified[0])
            if qualified not in self._qualified_fields:
                selected = self._variable(qualified)
                selected["metadata"] = self._edition_metadata[qualified[0]].to_dict()
                self._qualified_fields[qualified] = selected
            return deepcopy(self._qualified_fields[qualified])
        if edition is not None:
            raise ContractError("Qualified edition selection requires a schema3 catalog")
        resolved = self._resolve_field(field, scope)
        return self._field_metadata(*resolved) if resolved is not None else None

    def metric_metadata(self, metric: str, *, scope: str = "campus") -> dict[str, Any]:
        """Describe the exact canonical binding, including metric period overrides."""
        return self._field_metadata(*self._resolve_metric(metric, scope))

    def field(self, dataset: str, metric: str, *, scope: str = "campus") -> str:
        if getattr(self, "_v3", False):
            raise ContractError("Select a qualified source field from the schema3 index; names are not derived from years")
        metadata = self._metadata[dataset]
        definition = metadata.metrics.get(metric)
        return (
            definition.field
            if definition
            else f"{scope}_{self.field_year(dataset)}_{metric}"
        )

    def metric_period(self, metric: str, *, scope: str = "campus") -> DataPeriod:
        _, dataset, _, definition = self._resolve_metric(metric, scope)
        return definition.period if definition and definition.period is not None else self._metadata[dataset].period

    def _measure_bound(self, raw: Any, binding: MetricBinding, dataset: str, key: str,
                       definition: MetricDefinition | None, *, input_binding="caller_supplied") -> Measurement:
        if isinstance(raw, Measurement) and raw.source is not None:
            origin = raw.source
            if origin.get("build_id") != self.build_id or origin.get("dataset") != dataset or origin.get("field") != key:
                raise ContractError("Interpreted value belongs to another release or field")
        unit = self._metric_unit(binding, key, definition)
        minimum, maximum = binding.bounds(unit)
        value = measure(raw, self._policies[self._metadata[dataset].value_policy], unit=unit,
                        minimum=minimum, maximum=maximum)
        return self._with_origin(value, dataset, key, definition, input_binding=input_binding)

    @staticmethod
    def _metric_unit(binding: MetricBinding, key: str, definition: MetricDefinition | None) -> str | None:
        unit = definition.unit if definition and definition.unit is not None else binding.unit
        if unit is None:
            unit = next((binding.unit_suffixes[suffix] for suffix in sorted(binding.unit_suffixes, key=lambda item: len(item), reverse=True)
                         if key.endswith(suffix)), None)
        return unit

    def _entity_build_id(self, entity: Any) -> str | None:
        repo_catalog = getattr(getattr(entity, "_repo", None), "catalog", None)
        origins = [getattr(entity, "source_build_id", None), repo_catalog.build_id if repo_catalog is not None else None]
        if any(origin is not None and origin != self.build_id for origin in origins):
            raise ContractError("Entity belongs to another data release")
        return next((origin for origin in origins if origin is not None), None)

    def _contextual_measurement(self, raw: Any, binding: MetricBinding, dataset: str, key: str,
                               definition: MetricDefinition | None, *, entity=None,
                               input_binding="caller_supplied") -> Measurement:
        private = bool(getattr(entity, "is_private", False))
        if entity is not None and ((binding.population == "public" and private) or (binding.population == "private" and not private)):
            return self._with_origin(Measurement(None, ValueStatus.NOT_APPLICABLE,
                reason="outside_metric_population", unit=self._metric_unit(binding, key, definition)),
                dataset, key, definition, input_binding=input_binding)
        return self._measure_bound(raw, binding, dataset, key, definition, input_binding=input_binding)

    def _with_origin(self, value: Measurement, dataset: str, key: str,
                     definition: MetricDefinition | None, *, input_binding="caller_supplied") -> Measurement:
        metadata = self._metadata[dataset]
        period = definition.period if definition and definition.period is not None else metadata.period
        identity = (dataset, key, input_binding, canonical_json(period.to_dict()))
        if identity not in self._origin_cache:
            self._origin_cache[identity] = canonical_json({
                "dataset": dataset, "field": key, "period": period.to_dict(),
                "catalog_year": self._sources[dataset].get("resolved_year"),
                "build_id": self.build_id, "provenance_ids": list(metadata.provenance_ids),
                "input_binding": input_binding,
            })
        return replace(value, _source_json=self._origin_cache[identity])

    def measurement(
        self, entity: Any, metric: str, *, scope: str | None = None
    ) -> Measurement:
        self._require_scalar_interpretation()
        scope = scope or ("campus" if hasattr(entity, "campus_number") else "district")
        known_build = self._entity_build_id(entity)
        binding, dataset, key, definition = self._resolve_metric(metric, scope)
        meta = getattr(entity, "meta", {}) or {}
        raw = meta.get(key)
        if key not in meta and binding.attribute is not None:
            raw = getattr(entity, binding.attribute, None)
        return self._contextual_measurement(raw, binding, dataset, key, definition, entity=entity,
                                   input_binding="release_entity" if known_build is not None else "caller_supplied")

    def _resolve_field(self, field: str, scope: str) -> tuple[MetricBinding, str, str, MetricDefinition | None] | None:
        """Compile only release-invariant interpretation, including opaque fields."""
        self._require_scalar_interpretation()
        if scope not in {"campus", "district"}:
            raise ContractError("Unknown field scope")
        identity = (scope, field)
        if identity in self._field_resolutions:
            return self._field_resolutions[identity]
        if field in {"campus_number", "district_number", "campus_name", "district_name"}:
            self._field_resolutions[identity] = None
            return None
        candidates = []
        for dataset in self._sources:
            metadata = self._metadata[dataset]
            definition_items = [(name, definition) for name, definition in metadata.metrics.items() if definition.field == field]
            if field not in self._read_columns[dataset] and not definition_items:
                continue
            for name, binding in self._bindings.items():
                if not isinstance(binding, MetricBinding) or scope not in binding.scopes or binding.dataset.replace("{scope}", scope) != dataset:
                    continue
                if name.endswith(".*"):
                    selector = definition_items[0][0] if definition_items else field
                    prefix = f"{scope}_{metadata.field_year}_"
                    if binding.field_mode == "metric" and not definition_items:
                        if not field.startswith(prefix):
                            continue
                        selector = field[len(prefix):]
                    metric = name[:-1] + selector
                else:
                    metric = name
                resolved_binding, _, resolved_field, definition = self._resolve_metric(metric, scope)
                if resolved_field == field:
                    candidates.append((dataset, resolved_binding, definition, not name.endswith(".*")))
        if len({item[0] for item in candidates}) != 1:
            self._field_resolutions[identity] = None
            return None
        dataset, binding, definition, _ = max(candidates, key=lambda item: item[3])
        resolved = (binding, dataset, field, definition)
        self._field_resolutions[identity] = resolved
        return resolved

    def measure_field(self, raw: Any, field: str, *, scope: str = "campus", entity=None) -> Measurement | None:
        """Decode a bound raw value, optionally checking entity population and origin.

        The supplied raw value remains caller-supplied; entity context alone does
        not prove that arbitrary input bytes came from the release's entity store.
        Field contracts are reused; raw values and entity context are never cached.
        """
        self._require_scalar_interpretation()
        if entity is not None:
            self._entity_build_id(entity)
        resolved = self._resolve_field(field, scope)
        if resolved is None:
            return None
        binding, dataset, _, definition = resolved
        value = self._contextual_measurement(raw, binding, dataset, field, definition, entity=entity)
        if value.unit == "text" or (value.unit is None and isinstance(raw, bool)):
            return None
        if value.unit is None and value.status == ValueStatus.INVALID and value.reason == "unrecognized_numeric_token":
            return None
        return value

    def measurements_frame(self, entities, metrics: list[str], *, scope: str | None = None):
        """Long-form, status-preserving analytical values with field provenance."""
        self._require_scalar_interpretation()
        import pandas as pd
        rows = []
        for entity in entities:
            identity = getattr(entity, "campus_number", None) or getattr(entity, "district_number", None)
            for metric in metrics:
                value = self.measurement(entity, metric, scope=scope)
                origin = value.source or {}
                rows.append({"entity_id": identity, "metric": metric, "value": value.value,
                             "status": value.status.value, "raw": value.to_dict()["raw"],
                             "reason": value.reason, "unit": value.unit, "dataset": origin.get("dataset"),
                             "field": origin.get("field"), "period": origin.get("period"), "build_id": self.build_id,
                             "input_binding": origin.get("input_binding")})
        frame = pd.DataFrame(rows, columns=pd.Index(["entity_id", "metric", "value", "status", "raw", "reason", "unit", "dataset", "field", "period", "build_id", "input_binding"]))
        frame.attrs["teadata"] = self.to_dict()
        return frame

    def metric_label(
        self, metric: str, label: str, *, scope: str = "campus", short: bool = True
    ) -> str:
        value = self.metric_period(metric, scope=scope)
        period = value.label or "Period not specified"
        if short and value.start_year is not None:
            period = f"{value.start_year}-{str(value.end_year)[-2:]}"
        return f"{period} {label}"

    def charter_transfer_summary(self, campus: Any) -> dict[str, Any]:
        """Compute one campus's observed and estimated flows at the declared grain."""
        self._require_scalar_interpretation()
        policy = self.policy("campus_transfer_reports")
        model = self.metadata("campus_transfer_reports").estimation
        if model is None or model.grain != "origin_campus_destination_campus":
            raise ContractError(
                "Campus transfer estimates require a model at campus-pair grain"
            )
        observations = charter_transfer_observations(campus, policy=policy)
        summary = observations.aggregate
        unresolved = list(observations.unresolved)
        estimate = model.estimate(summary)
        return {
            "unresolved_transfers": unresolved,
            "unresolved_transfer_count": len(unresolved),
            "num_charter_transfer_destinations": observations.destination_count,
            "num_charter_transfer_destinations_masked": summary.suppressed_count,
            "total_unmasked_charter_transfers_out": summary.observed_sum,
            "total_estimated_masked_charter_transfers_out": estimate[
                "estimated_contribution"
            ],
            "total_estimated_charter_transfers_out": estimate["value"],
            "charter_transfer_aggregate": summary.to_dict(),
            "charter_transfer_estimate": estimate,
        }

    def _require_scalar_interpretation(self) -> None:
        if getattr(self, "_v3", False):
            schema = 4 if getattr(self, "_v4", False) else 3
            raise ContractError(f"Schema{schema} scalar interpretation is unavailable; this reader only opens qualified native leaves")


def write_sqlite_catalog(
    connection: sqlite3.Connection, catalog: DatasetCatalog
) -> None:
    connection.execute(CATALOG_TABLE_DDL)
    connection.execute(
        f"INSERT INTO {CATALOG_TABLE} (singleton,build_id,document) VALUES (1, ?, ?)",
        (catalog.build_id, canonical_json(catalog.to_dict())),
    )


def read_sqlite_catalog(path: str | Path, *, immutable: bool = False) -> DatasetCatalog:
    path = Path(path).resolve()
    try:
        with closing(readonly_connection(path, immutable=immutable)) as connection:
            return read_sqlite_catalog_connection(connection)
    except sqlite3.Error as exc:
        raise ContractError(
            f"Artifact lacks the required dataset catalog: {path.name}"
        ) from exc


def read_sqlite_catalog_connection(connection: sqlite3.Connection) -> DatasetCatalog:
    """Read the canonical catalog using an existing caller-owned SQLite reader."""
    try:
        columns = {row[1] for row in connection.execute(f"PRAGMA table_info({CATALOG_TABLE})")}
        projection = ",build_id" if "build_id" in columns else ""
        rows = connection.execute(
            f"SELECT singleton,document{projection} FROM {CATALOG_TABLE} LIMIT 2"
        ).fetchall()
    except sqlite3.Error as error:
        raise ContractError("Artifact lacks the required dataset catalog") from error
    if len(rows) != 1 or rows[0][0] != 1:
        raise ContractError("Artifact must contain exactly one dataset catalog")
    if projection and rows[0][2] is None:
        raise ContractError("Dataset catalog build ID projection is missing")
    return DatasetCatalog(json.loads(rows[0][1]), expected_build_id=rows[0][2] if projection else None)


ACTIVE_BUILD: ContextVar[BuildRecorder | None] = ContextVar(
    "teadata_build", default=None
)


class BuildRecorder:
    """Bind successful source reads to verified acquisition receipts."""

    def __init__(self, cfg: Any, store: ProvenanceStore):
        self.cfg = deepcopy(cfg)
        self.processor_sha256 = processor_identity()
        self.store = store
        self.datasets: dict[str, Any] = {}
        self.column_schemas: dict[str, str] = {}
        self.edition_entries: dict[str, Any] = {}
        self.edition_artifacts: dict[str, Any] = {}
        self.edition_fields: list[dict[str, Any]] = []
        self.edition_roots: set[str] = set()
        self._token = None

    def __enter__(self) -> BuildRecorder:
        if ACTIVE_BUILD.get() is not None:
            raise ContractError("Nested data builds are not supported")
        self._token = ACTIVE_BUILD.set(self)
        return self

    def __exit__(self, *_args: Any) -> None:
        if self._token is not None:
            ACTIVE_BUILD.reset(self._token)

    def before_read(self, source: ResolvedDataset, *, reader_profile: str | None = None) -> dict[str, Any]:
        if processor_identity() != self.processor_sha256:
            raise ContractError("Processing source changed during the build")
        expected = self.cfg.resolve_dataset(
            source.dataset, source.requested_year, section=source.section, reader_profile=reader_profile
        )
        if expected.to_dict() != source.to_dict():
            raise ContractError("Source configuration changed during the build")
        policy = self.cfg.value_policies.get(source.metadata.value_policy)
        _validate_methodology(source.metadata, policy)
        return verify_source_receipts(
            source, self.store, reference_base=self.cfg.base_dir
        )

    def after_read(
        self,
        source: ResolvedDataset,
        identity: Mapping[str, Any],
        rows: int,
        profile: str | None,
        effective_reader: Mapping[str, Any],
        columns: list[str],
    ) -> None:
        if file_identity(source.path) != identity:
            raise ContractError(f"Source changed while being read: {source.dataset}")
        if rows == 0:
            raise ContractError(f"Required dataset is empty: {source.dataset}")
        document: dict[str, Any] = {
            "dataset": source.dataset,
            "section": source.section,
            "requested_year": source.requested_year,
            "resolved_year": source.resolved_year,
            "metadata": source.metadata.to_dict(),
            "reader": dict(source.reader),
            "artifact": dict(identity),
            "status": "loaded",
            "reads": [],
        }
        contract = self.cfg.column_contract(source.dataset, source.requested_year,
                                            section=source.section, reader_profile=profile)
        if contract is not None:
            schema = parse_column_schema(contract["text"].encode())
            if schema["schema_version"] == 2:
                if source.metadata.version is None:
                    raise ContractError("Header crosswalk requires an explicit source edition")
                try:
                    validate_selected_crosswalk(schema, dataset=source.dataset, edition=source.metadata.version,
                                                input_sha256=identity["sha256"], columns=columns)
                except ValueError as error:
                    raise ContractError(f"Invalid build header crosswalk: {error}") from error
                document["column_schema_sha256"] = contract["sha256"]
                self.column_schemas[contract["sha256"]] = contract["text"]
        old = self.datasets.get(source.dataset)
        if old is not None:
            if {k: v for k, v in old.items() if k != "reads"} != {
                k: v for k, v in document.items() if k != "reads"
            }:
                raise ContractError(
                    f"Mixed source versions in one build: {source.dataset}"
                )
            document = old
        read = {
            "profile": profile,
            "rows": rows,
            "reader": dict(effective_reader),
            "columns": columns,
        }
        if read not in document["reads"]:
            document["reads"].append(read)
        self.datasets[source.dataset] = document

    def record_edition(self, key: EditionKey, source: ResolvedDataset, *, artifacts: Iterable[ArtifactRef],
                       acquisition_ids: Iterable[str], dependencies: Mapping[str, tuple[tuple[str, str], ...]] | None = None) -> None:
        """Opt-in producer registration; derive the index from one reviewed table.

        Native writers supply actual descriptors after materialization. No
        manual code/label/period/unit registry is accepted here.
        """
        if processor_identity() != self.processor_sha256:
            raise ContractError("Processing source changed during the build")
        expected = self.cfg.resolve_dataset(source.dataset, source.requested_year, section=source.section, reader_profile="source")
        if expected.to_dict() != source.to_dict() or key.dataset != source.dataset or key.edition != source.metadata.version:
            raise ContractError("Edition registration differs from selected Config source")
        if key.identity in self.edition_entries:
            raise ContractError("Edition already registered")
        contract = self.cfg.column_contract(source.dataset, source.requested_year, reader_profile="source")
        if contract is None:
            raise ContractError("Edition registration requires a reviewed source crosswalk")
        document = parse_column_schema(contract["text"].encode())
        selected, = [table for table in crosswalk_dataset(document, key.dataset, key.edition)["tables"] if table["table"] == key.table]
        rows = crosswalk_records(document, dataset=key.dataset, edition=key.edition, table=key.table, metadata=source.metadata)
        refs = list(artifacts)
        if not refs or any(ref.edition != key for ref in refs):
            raise ContractError("Edition registration requires exact native leaf descriptors")
        selected_dependencies = dict(dependencies or {})
        if set(selected_dependencies) - {row["field"] for row in rows}:
            raise ContractError("Dependencies name an unregistered field")
        additions = []
        native_columns = {ref.identity: set(ref.public_columns) for ref in refs}
        for ref in refs:
            _validate_leaf_support(ref.contract, rows, source.metadata)
        for row in rows:
            owning = [ref for ref in refs if row["field"] in native_columns[ref.identity]]
            if len(owning) != 1:
                raise ContractError("Each registered field requires exactly one authoritative leaf")
            additions.append(FieldBinding(key.identity, row["field"], owning[0].identity, row["field"],
                                          selected_dependencies.get(row["field"], ())).to_dict())
        acquisitions = list(acquisition_ids)
        roots = [*acquisitions, *(ref.processing_id for ref in refs),
                 *(identifier for reference in source.metadata.references for identifier in reference.provenance_ids)]
        records = self.store.verify_record_closure(roots)
        raw_hashes = []
        for identifier in acquisitions:
            record = records[identifier]
            if record["kind"] != "acquisition" or record.get("source_metadata", {}).get("version") != key.edition:
                raise ContractError("Edition source inputs require exact-edition acquisition receipts")
            raw_hashes.append(record["artifact"]["sha256"])
        try:
            table_input_sha256s(selected, acquisition_sha256s=raw_hashes)
        except ValueError as error:
            raise ContractError(str(error)) from error
        self.edition_entries[key.identity] = {"key": key.to_dict(), "metadata": source.metadata.to_dict(),
            "acquisition_ids": acquisitions,
            "column_schema_sha256": contract["sha256"]}
        self.column_schemas[contract["sha256"]] = contract["text"]
        self.edition_artifacts.update({ref.identity: ref.to_dict() for ref in refs})
        self.edition_fields.extend(additions)
        self.edition_roots.update(roots)

    def leaf_contract(self, key: EditionKey, source: ResolvedDataset, *, native_schema: bytes,
                      columns: Iterable[str], key_columns: Iterable[str], rows: int,
                      inputs: Iterable[str], recipe: Mapping[str, Any], toolchain: Mapping[str, str],
                      column_support: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Derive leaf meaning from the same Config/crosswalk and native writer.

        Writers supply metadata-free Arrow IPC bytes, not another type system.
        Neither file SHA nor the eventual processing receipt enters this hash.
        """
        import base64
        from .column_schema import crosswalk_dataset

        if processor_identity() != self.processor_sha256 or key.dataset != source.dataset or key.edition != source.metadata.version:
            raise ContractError("Leaf contract differs from selected build source")
        if self.cfg.resolve_dataset(source.dataset, source.requested_year, section=source.section, reader_profile="source").to_dict() != source.to_dict():
            raise ContractError("Source configuration changed during the build")
        resource = self.cfg.column_contract(source.dataset, source.requested_year, reader_profile="source")
        if resource is None:
            raise ContractError("Leaf contract requires a reviewed source crosswalk")
        document = parse_column_schema(resource["text"].encode())
        selected, = [table for table in crosswalk_dataset(document, key.dataset, key.edition)["tables"] if table["table"] == key.table]
        field_rows = crosswalk_records(document, dataset=key.dataset, edition=key.edition, table=key.table, metadata=source.metadata)
        policy = self.cfg.value_policies.get(source.metadata.value_policy)
        result: dict[str, Any] = {"edition": key.to_dict(), "schema": base64.b64encode(native_schema).decode("ascii"),
            "schema_sha256": sha256(native_schema).hexdigest(), "columns": list(columns), "key_columns": list(key_columns),
            "rows": rows, "inputs": list(inputs), "mapping_sha256": sha256(canonical_json(selected).encode()).hexdigest(),
            "semantics_sha256": _leaf_semantics(source.metadata, field_rows),
            "policy_sha256": sha256(canonical_json(policy.to_dict()).encode()).hexdigest() if policy is not None else None,
            "recipe_sha256": sha256(canonical_json(recipe).encode()).hexdigest(),
            "producer": {"code_sha256": self.processor_sha256, "toolchain": dict(toolchain)}}
        if column_support is not None:
            result["column_support"] = deepcopy(dict(column_support))
            _validate_leaf_support(result, field_rows, source.metadata)
        return result

    def catalog(self, *, schema_version: int = CATALOG_SCHEMA) -> DatasetCatalog:
        if processor_identity() != self.processor_sha256:
            raise ContractError("Processing source changed during the build")
        if type(schema_version) is int and schema_version in (3, 4):
            records = self.store.verify_record_closure(sorted(self.edition_roots))
            resources = {digest: text.encode() for digest, text in self.column_schemas.items()}
            descriptors = {digest: ColumnSchemaRef(digest, len(raw), digest + ".json").to_dict()
                           for digest, raw in resources.items()}
            return DatasetCatalog.create_editions(editions=self.edition_entries, artifacts=self.edition_artifacts,
                field_bindings=self.edition_fields, column_schemas=descriptors, records=records,
                value_policies={key: value.to_dict() for key, value in self.cfg.value_policies.items()},
                column_schema_loader=lambda ref: resources[ref.sha256], schema_version=schema_version)
        if schema_version != CATALOG_SCHEMA:
            raise ContractError("Unsupported catalog writer schema")
        identifiers = set()
        for source in self.datasets.values():
            identifiers.update(source["metadata"]["provenance_ids"])
            for reference in source["metadata"]["references"]:
                identifiers.update(reference["provenance_ids"])
        return DatasetCatalog.create(
            datasets=self.datasets,
            value_policies={
                key: value.to_dict() for key, value in self.cfg.value_policies.items()
            },
            processor=self.processor_sha256,
            metric_bindings=self.cfg.resolved_metric_bindings(),
            records=validate_record_graph(self.store.read_record, sorted(identifiers)),
            environment={
                "python": platform.python_version(),
                "hash_seed": os.environ.get("PYTHONHASHSEED", "random"),
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "architecture": platform.machine(),
                },
                "packages": {
                    name: version(name)
                    for name in (
                        "teadata",
                        "pandas",
                        "numpy",
                        "geopandas",
                        "shapely",
                        "pyogrio",
                        "openpyxl",
                        "sqlalchemy",
                    )
                },
            },
            column_schemas=self.column_schemas,
        )


def verify_source_receipts(
    source: ResolvedDataset,
    store: ProvenanceStore,
    *,
    reference_base: Path | None = None,
) -> dict[str, Any]:
    identity = file_identity(source.path)
    if not source.metadata.provenance_ids:
        raise ContractError(
            f"Source {source.dataset} needs an acquisition receipt; run the source-observation stage"
        )
    metadata_matched = False
    for record_id in source.metadata.provenance_ids:
        record = store.read_record(record_id)
        _verify_input_receipt(record, dataset=source.dataset, section=source.section, artifact=identity)
        metadata_matched |= _metadata_content(source.metadata) == _metadata_content(DatasetMetadata.from_dict(record["source_metadata"]))
    store.verify_artifact(identity["sha256"])
    if not metadata_matched:
        raise ContractError(f"Source metadata differs from its receipt: {source.dataset}")
    verify_reference_receipts(source, store, reference_base=reference_base)
    return identity


def verify_reference_receipts(
    source: ResolvedDataset, store: ProvenanceStore, *, reference_base: Path | None = None,
    check_local_paths: bool = True, verify_all_claimed: bool = False,
) -> None:
    """Verify archived documentation without confounding it with source observations."""
    try:
        validate_reference_records(source, store.read_record, verify_all_claimed=verify_all_claimed)
    except ValueError as error:
        raise ContractError(str(error)) from error
    for reference in source.metadata.references:
        if reference.status != "available" and not (verify_all_claimed and reference.provenance_ids):
            continue
        for record_id in reference.provenance_ids:
            record = store.read_record(record_id)
            store.verify_artifact(record["artifact"]["sha256"])
            if reference.path is not None and check_local_paths:
                if reference_base is None:
                    raise ContractError(
                        "Local reference needs a source contract directory"
                    )
                path = (reference_base / reference.path).resolve()
                if (
                    not path.is_relative_to(reference_base.resolve())
                    or file_identity(path)["sha256"] != record["artifact"]["sha256"]
                ):
                    raise ContractError("Local methodology differs from its receipt")
