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
from typing import Any, Iterable, Mapping

from .dataset_metadata import DataPeriod, DatasetMetadata, MetricDefinition, ResolvedDataset
from .metric_bindings import MetricBinding, parse_bindings
from .provenance import ProvenanceStore
from .values import Measurement, ValuePolicy, ValueStatus, measure
from .transfer_summary import charter_transfer_observations
from .column_schema import (
    parse_column_schema, reader_options, validate_selected_crosswalk,
    header_records as crosswalk_records, header_rename_map,
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


def file_identity(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
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

    def __init__(self, document: Mapping[str, Any], *, expected_build_id: str | None = None):
        document = json.loads(canonical_json(document))
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
            if policy:
                selected_policy = ValuePolicy.from_dict(policies[policy])
                if any(rule.status == ValueStatus.SUPPRESSED for rule in selected_policy.tokens.values()) and not selected_policy.reference_id:
                    raise ContractError("Suppression policies require an exact reviewed methodology reference")
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
            required_reviews = set()
            if policy and ValuePolicy.from_dict(policies[policy]).reference_id:
                required_reviews.add(ValuePolicy.from_dict(policies[policy]).reference_id)
            if metadata.estimation is not None:
                required_reviews.add(metadata.estimation.reference_id)
            for identifier in required_reviews:
                reference = references.get(identifier)
                if reference is None or reference.status != "available" or reference.reviewed_sha256 is None:
                    raise ContractError(f"Dataset {key} requires a review of exact methodology bytes")
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
                except (KeyError, ValueError) as error:
                    raise ContractError(f"Invalid selected header crosswalk: {error}") from error
        self._build_id = identity
        self._serialized = canonical_json(document)
        self._policies = {
            key: ValuePolicy.from_dict(value) for key, value in policies.items()
        }
        self._sources = sources
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
        return self.to_dict()["datasets"]

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._serialized)

    def metadata(self, dataset: str) -> DatasetMetadata:
        try:
            return DatasetMetadata.from_dict(self._sources[dataset]["metadata"])
        except KeyError as exc:
            raise ContractError(
                f"Dataset was not loaded in this release: {dataset}"
            ) from exc

    def header_records(self, dataset: str, *, edition: str | None = None,
                       table: str | None = None) -> list[dict[str, Any]]:
        """Return detached code/label/field records from this release's bound resource.

        Only the selected source edition is available. No package YAML, files,
        source rows or network are consulted, including for older releases.
        """
        metadata = self.metadata(dataset)
        selected = metadata.version
        if edition is not None and edition != selected:
            raise ContractError(f"Header crosswalk edition unavailable: {edition!r}; selected {selected!r}")
        digest = self._sources[dataset].get("column_schema_sha256")
        if digest is None or selected is None:
            raise ContractError(f"Header crosswalk unavailable for selected dataset {dataset!r}")
        try:
            return crosswalk_records(self._column_schemas[digest], dataset=dataset, edition=selected, table=table)
        except ValueError as error:
            raise ContractError(str(error)) from error

    def header_mapping(self, dataset: str, *, table: str, edition: str | None = None,
                       from_: str = "code", to: str = "field",
                       columns: Iterable[str] | None = None) -> dict[str, str]:
        """Return a plain rename map; optional external headers detect collisions."""
        try:
            return header_rename_map(self.header_records(dataset, edition=edition, table=table),
                                     from_=from_, to=to, columns=columns)
        except ValueError as error:
            raise ContractError(str(error)) from error

    def acquisitions(self, dataset: str) -> list[dict[str, Any]]:
        """Return immutable acquisition observations, including unknown download dates."""
        records = self.to_dict()["provenance"]
        return [
            {"record_id": identifier, **records[identifier]}
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
        year = self.metadata(dataset).field_year
        if year is None:
            raise ContractError(f"Dataset {dataset} has no declared field year")
        return year

    def policy(self, dataset: str) -> ValuePolicy:
        policy_id = self.metadata(dataset).value_policy
        if policy_id is None:
            raise ContractError(f"Dataset {dataset} has no value policy")
        return ValuePolicy.from_dict(self._policies[policy_id].to_dict())

    def _resolve_metric(self, metric: str, scope: str) -> tuple[MetricBinding, str, str, MetricDefinition | None]:
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

    def field_metadata(self, field: str, *, scope: str = "campus") -> dict[str, Any] | None:
        """Describe a known source field without interpreting its values.

        Opaque text can still have known provenance. A null result means this
        catalog cannot establish the field's source; names are never guessed.
        Returned metadata is detached from the catalog's immutable authority.
        """
        resolved = self._resolve_field(field, scope)
        return self._field_metadata(*resolved) if resolved is not None else None

    def metric_metadata(self, metric: str, *, scope: str = "campus") -> dict[str, Any]:
        """Describe the exact canonical binding, including metric period overrides."""
        return self._field_metadata(*self._resolve_metric(metric, scope))

    def field(self, dataset: str, metric: str, *, scope: str = "campus") -> str:
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
        dataset, _ = self.source_for_metric(metric, scope)
        value = self.metric_period(metric, scope=scope)
        period = value.label or "Period not specified"
        if short and value.start_year is not None:
            period = f"{value.start_year}-{str(value.end_year)[-2:]}"
        return f"{period} {label}"

    def charter_transfer_summary(self, campus: Any) -> dict[str, Any]:
        """Compute one campus's observed and estimated flows at the declared grain."""
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


def write_sqlite_catalog(
    connection: sqlite3.Connection, catalog: DatasetCatalog
) -> None:
    connection.execute(CATALOG_TABLE_DDL)
    connection.execute(
        f"INSERT INTO {CATALOG_TABLE} (singleton,build_id,document) VALUES (1, ?, ?)",
        (catalog.build_id, canonical_json(catalog.to_dict())),
    )


def read_sqlite_catalog(path: str | Path) -> DatasetCatalog:
    path = Path(path).resolve()
    try:
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
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
        self._token = None

    def __enter__(self) -> BuildRecorder:
        if ACTIVE_BUILD.get() is not None:
            raise ContractError("Nested data builds are not supported")
        self._token = ACTIVE_BUILD.set(self)
        return self

    def __exit__(self, *_args: Any) -> None:
        if self._token is not None:
            ACTIVE_BUILD.reset(self._token)

    def before_read(self, source: ResolvedDataset) -> dict[str, Any]:
        if processor_identity() != self.processor_sha256:
            raise ContractError("Processing source changed during the build")
        expected = self.cfg.resolve_dataset(
            source.dataset, source.requested_year, section=source.section
        )
        if expected.to_dict() != source.to_dict():
            raise ContractError("Source configuration changed during the build")
        policy = self.cfg.value_policies.get(source.metadata.value_policy)
        if policy is not None and policy.reference_id:
            references = {
                reference.reference_id: reference
                for reference in source.metadata.references
            }
            reference = references.get(policy.reference_id)
            if reference is None or reference.status != "available" or reference.reviewed_sha256 is None:
                raise ContractError(
                    f"Value policy requires reviewed methodology: {source.dataset}"
                )
        if policy is not None and any(rule.status == ValueStatus.SUPPRESSED for rule in policy.tokens.values()) and not policy.reference_id:
            raise ContractError("Suppression policies require a reviewed methodology reference")
        model = source.metadata.estimation
        if model is not None:
            reference = next((item for item in source.metadata.references if item.reference_id == model.reference_id), None)
            if reference is None or reference.status != "available" or reference.reviewed_sha256 is None:
                raise ContractError("Estimation requires reviewed methodology bytes")
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

    def catalog(self) -> DatasetCatalog:
        if processor_identity() != self.processor_sha256:
            raise ContractError("Processing source changed during the build")
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
            records={
                identifier: self.store.read_record(identifier)
                for identifier in sorted(identifiers)
            },
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
    for reference in source.metadata.references:
        if reference.status != "available" and not (verify_all_claimed and reference.provenance_ids):
            continue
        if not reference.provenance_ids:
            raise ContractError(
                f"Reference document {reference.reference_id} needs an acquisition receipt"
            )
        for record_id in reference.provenance_ids:
            record = store.read_record(record_id)
            if record.get("kind") != "acquisition":
                raise ContractError("Reference receipt is not an acquisition")
            if reference.reviewed_sha256 is not None and record["artifact"]["sha256"] != reference.reviewed_sha256:
                raise ContractError("Reference bytes differ from reviewed methodology")
            if reference.url not in {
                record.get("requested_url"),
                record.get("resolved_url"),
                record.get("source_metadata", {}).get("source_url"),
            }:
                raise ContractError("Reference receipt belongs to a different URL")
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
