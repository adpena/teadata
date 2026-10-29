"""Release-time Arrow materialization from one immutable entities artifact."""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import os
from pathlib import Path
import platform
import tempfile
import time
from types import SimpleNamespace
from typing import Any

from .analytical import (
    ANALYTICAL_METADATA, ANALYTICAL_ROLES, ANALYTICAL_SCHEMA_VERSION,
    ANALYTICAL_TABLES, _cell_values, analytical_schema, validate_artifact,
)
from .attributes import entity_attributes
from .entity_store import EntityStore
from .manifest import ContractError, DatasetCatalog, canonical_json, file_identity, processor_identity


def _source_signature(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _require_checkpointed(path: Path) -> None:
    if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
        raise ContractError("Analytical production requires an immutable checkpointed entities file")


def _source_columns(scope: str, name: str, kind: str, index: int,
                    info: dict[str, Any] | None) -> dict[str, Any]:
    metadata = info["metadata"] if info is not None else {}
    acquisitions = []
    common: dict[str, Any] = {
        "scope": scope, "name": name, "kind": kind, "selector_index": index,
        "field": info["field"] if info is not None else name if kind == "field" else None,
        "metric": name if kind == "metric" else None,
        "dataset": info["dataset"] if info is not None else None,
        "version": info["version"] if info is not None else None,
        "period": info["period"] if info is not None else {"kind": "unknown"},
        "dataset_period": metadata.get("period", {"kind": "unknown"}),
        "field_year": info["field_year"] if info is not None else None,
        "source_status": "catalog_bound" if info is not None else "unbound",
        "unit": info["unit"] if info is not None else None,
        "policy_id": info["policy_id"] if info is not None else None,
        "population": info["population"] if info is not None else None,
        **{key: metadata.get(key) for key in (
            "title", "publisher", "source_url", "download_url", "published_at", "revised_at",
            "license", "citation", "notes", "provenance_ids", "references",
        )},
        "acquisitions": acquisitions, "metadata_json": canonical_json(metadata) if metadata else None,
    }
    for item in info["acquisitions"] if info is not None else []:
        acquisitions.append({
            **{key: item.get(key) for key in (
                "record_id", "kind", "method", "recorded_at", "downloaded_at", "requested_url", "resolved_url",
            )},
            **{key: item.get("artifact", {}).get(key) for key in ("filename", "sha256", "size_bytes")},
            "record_json": canonical_json(item),
        })
    return common


def _context(record: dict[str, Any], scope: str) -> SimpleNamespace:
    entity = SimpleNamespace(**record)
    if scope == "campus":
        entity.district = SimpleNamespace(
            name=record.get("district_name", ""), district_number=record.get("district_number"),
        )
    return entity


def build_analytical_artifacts(
    entities: str | Path, catalog: DatasetCatalog, directory: str | Path, *,
    batch_size: int = 128, metrics: Sequence[str] = ("enrollment", "rating"),
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Materialize canonical results once; never consume fresh rich engine metadata.

    The final manifest owns child checksums. Children bind catalog/build/source
    inputs only, avoiding a manifest/child hash cycle. The returned diagnostics
    include source hydration separately from scalar and Arrow work.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if type(batch_size) is not int or not 1 <= batch_size <= 4096:
        raise ValueError("Analytical batches require 1–4096 entities")
    if (not isinstance(metrics, Sequence) or isinstance(metrics, (str, bytes))
            or any(not isinstance(name, str) or not name for name in metrics)
            or len(set(metrics)) != len(metrics)):
        raise ValueError("Canonical metrics must be unique nonempty names")
    # Validate the entire public selector contract before opening/hydrating data.
    metric_info = {scope: {name: catalog.metric_metadata(name, scope=scope) for name in metrics}
                   for scope in ("campus", "district")}
    source = Path(entities).resolve(strict=True)
    _require_checkpointed(source)
    signature = _source_signature(source)
    identity = file_identity(source)
    actual_processor = processor_identity()
    lockfile = Path(__file__).resolve().parent.parent / "uv.lock"
    producer = {"processor_sha256": actual_processor,
                "source_processor_sha256": catalog.to_dict()["processor_sha256"],
                "lock_sha256": file_identity(lockfile)["sha256"] if lockfile.is_file() else None,
                "python": platform.python_version(), "pyarrow": pa.__version__}
    output = Path(directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    destinations = {role: output / f"{role}_{catalog.build_id}.parquet" for role in ANALYTICAL_ROLES}
    if any(path.exists() or path.is_symlink() for path in destinations.values()):
        raise ContractError("Analytical output paths must be fresh")
    catalog_hash = hashlib.sha256(canonical_json(catalog.to_dict()).encode()).hexdigest()
    diagnostics: dict[str, Any] = {"source": identity, "scopes": {}, "runtime": {}}
    sources: list[dict[str, Any]] = []
    built: dict[str, Path] = {}

    def footer(role: str, rows: int) -> dict[bytes, bytes]:
        document = {
            "schema_version": ANALYTICAL_SCHEMA_VERSION, "role": role,
            "scope": ANALYTICAL_TABLES[role][1], "build_id": catalog.build_id,
            "catalog_sha256": catalog_hash, "entities_sha256": identity["sha256"],
            "producer_inputs": producer,
            "rows": rows,
        }
        return {ANALYTICAL_METADATA: canonical_json(document).encode(),
                b"teadata.build_id": catalog.build_id.encode(),
                b"teadata.entities_sha256": identity["sha256"].encode()}

    with tempfile.TemporaryDirectory(prefix=".analytical-", dir=output) as temporary:
        staging = Path(temporary)
        with EntityStore(source, immutable=True) as store:
            if store.catalog.to_dict() != catalog.to_dict():
                raise ContractError("Analytical source belongs to a different catalog")
            connection = store.connection
            for scope, table in (("campus", "campuses"), ("district", "districts")):
                bad_json = connection.execute(
                    f"SELECT count(*) FROM {table} WHERE meta IS NOT NULL AND CASE WHEN json_valid(meta) THEN json_type(meta) NOT IN ('object','null') ELSE 1 END"
                ).fetchone()[0]
                if bad_json:
                    raise ContractError("Entities metadata must contain JSON objects or null")
                fields = tuple(row[0] for row in connection.execute(
                    f"SELECT DISTINCT j.key FROM {table} c,json_each(c.meta) j WHERE json_type(c.meta)='object' ORDER BY j.key"
                ))
                identifiers = [row[0] for row in connection.execute(
                    f"SELECT COALESCE(NULLIF({scope}_number_canon,''),{scope}_number) AS public_id FROM {table} ORDER BY public_id"
                )]
                if (any(not isinstance(value, str) or not value for value in identifiers)
                        or len(set(identifiers)) != len(identifiers)):
                    raise ContractError("Analytical entities require unique nonempty public identities")
                if len(identifiers) >= 1 << 32 or len(fields) + len(metrics) >= 1 << 32:
                    raise ContractError("Analytical release-local ordinals exceed UInt32")
                selectors = [
                    _source_columns(scope, name, "field", index, catalog.field_metadata(name, scope=scope))
                    for index, name in enumerate(fields)
                ] + [
                    _source_columns(scope, name, "metric", len(fields) + index, metric_info[scope][name])
                    for index, name in enumerate(metrics)
                ]
                sources.extend(selectors)
                cells_role, attributes_role = f"{scope}_analytics", f"{scope}_attributes"
                cell_schema = analytical_schema(cells_role).with_metadata(footer(cells_role, len(identifiers) * len(selectors)))
                attr_schema = analytical_schema(attributes_role).with_metadata(footer(attributes_role, len(identifiers)))
                cell_path, attr_path = staging / (cells_role + ".parquet"), staging / (attributes_role + ".parquet")
                getter = store.get_campus if scope == "campus" else store.get_district
                times = {"source_hydration_seconds": 0.0, "canonical_scalar_seconds": 0.0, "arrow_write_seconds": 0.0}
                cell_count = 0
                with pq.ParquetWriter(cell_path, cell_schema, compression="zstd", use_dictionary=True) as cells_writer, pq.ParquetWriter(attr_path, attr_schema, compression="zstd", use_dictionary=True) as attr_writer:
                    for start in range(0, len(identifiers), batch_size):
                        batch = identifiers[start:start + batch_size]
                        started = time.monotonic()
                        records = [getter(identifier, include_meta=True, cache=False) for identifier in batch]
                        if any(record is None for record in records):
                            raise ContractError("An analytical source entity could not be read")
                        times["source_hydration_seconds"] += time.monotonic() - started
                        started = time.monotonic()
                        cells, attributes = [], []
                        for offset, (identifier, record) in enumerate(zip(batch, records, strict=True)):
                            if record is None:
                                raise ContractError("An analytical source entity is missing")
                            entity = _context(record, scope)
                            entity_index = start + offset
                            if (record[f"{scope}_number_canon"] or record[f"{scope}_number"]) != identifier:
                                raise ContractError("Analytical source identity changed during hydration")
                            attributes.append({"scope": scope, "entity_id": identifier,
                                               "entity_index": entity_index, **entity_attributes(entity, scope=scope)})
                            meta = record["meta"]
                            for selector in selectors:
                                name = selector["name"]
                                if selector["kind"] == "field":
                                    raw = meta.get(name)
                                    value = catalog.measure_field(raw, name, scope=scope, entity=entity)
                                    present = name in meta
                                else:
                                    value = catalog.measurement(entity, name, scope=scope)
                                    raw, present = value.raw, None
                                row = {key: selector[key] for key in (
                                    "scope", "name", "kind", "selector_index", "field", "metric", "dataset", "version",
                                    "period", "dataset_period", "field_year", "source_status",
                                )}
                                row.update(entity_id=identifier, entity_index=entity_index, raw_present=present)
                                row.update(_cell_values(raw, value))
                                row["input_binding"] = (value.source or {}).get("input_binding") if value is not None else None
                                cells.append(row)
                        times["canonical_scalar_seconds"] += time.monotonic() - started
                        started = time.monotonic()
                        cells_writer.write_table(pa.Table.from_pylist(cells, schema=cell_schema))
                        attr_writer.write_table(pa.Table.from_pylist(attributes, schema=attr_schema))
                        times["arrow_write_seconds"] += time.monotonic() - started
                        cell_count += len(cells)
                if cell_count != len(identifiers) * len(selectors):
                    raise ContractError("Analytical source projection is incomplete")
                diagnostics["scopes"][scope] = {"entities": len(identifiers), "fields": len(fields), "metrics": len(metrics), "cells": cell_count}
                diagnostics["runtime"][scope] = times
                built.update({cells_role: cell_path, attributes_role: attr_path})
            role = "analytical_sources"
            path = staging / (role + ".parquet")
            schema = analytical_schema(role).with_metadata(footer(role, len(sources)))
            pq.write_table(pa.Table.from_pylist(sources, schema=schema), path, compression="zstd", use_dictionary=True)
            built[role] = path
        _require_checkpointed(source)
        if _source_signature(source) != signature or file_identity(source) != identity:
            raise ContractError("The analytical source changed during production")
        if processor_identity() != actual_processor:
            raise ContractError("The analytical producer changed during production")
        for role, path in built.items():
            validate_artifact(path, catalog, role, entities_sha256=identity["sha256"])
        for role, path in built.items():
            # Exclusive links cannot replace a foreign or concurrently created file.
            os.link(path, destinations[role])
    return destinations, diagnostics
