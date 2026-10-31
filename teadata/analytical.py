"""Versioned, self-describing Arrow read models of canonical scalar results.

Arrow and Parquet own columns, scans and transport. This module never parses
masking tokens: the release-time producer calls the canonical scalar catalog.
Imports are lazy so ordinary source/metadata users do not initialize Arrow.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .manifest import ContractError, DatasetCatalog, canonical_json
from .values import Measurement, ValueStatus, presentation_value

if TYPE_CHECKING:
    import pyarrow as pa
    import pyarrow.dataset as ds


ANALYTICAL_SCHEMA_VERSION = 2
ANALYTICAL_METADATA = b"teadata.analytical"
ANALYTICAL_TABLES = {
    "campus_analytics": ("cells", "campus"),
    "district_analytics": ("cells", "district"),
    "campus_attributes": ("attributes", "campus"),
    "district_attributes": ("attributes", "district"),
    "analytical_sources": ("sources", "all"),
}
ANALYTICAL_DATASETS = {
    "campus": "campus_analytics", "district": "district_analytics",
    "campus_attributes": "campus_attributes", "district_attributes": "district_attributes",
    "sources": "analytical_sources",
}
ANALYTICAL_ROLES = frozenset(ANALYTICAL_TABLES)
PRESENTATION_COLUMNS = (
    "presentation_kind", "presentation_text", "presentation_integer",
    "presentation_integer_decimal", "presentation_float", "presentation_boolean",
)
_INT64_MIN, _INT64_MAX = -(1 << 63), (1 << 63) - 1


def _period_type():
    import pyarrow as pa

    return pa.struct([
        ("kind", pa.string()), ("start_year", pa.int16()), ("end_year", pa.int16()),
        ("year", pa.int16()), ("as_of", pa.string()),
    ])


def _bound_type():
    import pyarrow as pa

    return pa.struct([
        ("kind", pa.string()), ("integer", pa.int64()),
        ("integer_decimal", pa.large_string()), ("float", pa.float64()),
    ])


def _common_fields():
    import pyarrow as pa

    return [
        ("scope", pa.string()), ("name", pa.string()), ("kind", pa.string()),
        ("field", pa.string()), ("metric", pa.string()),
        ("dataset", pa.string()), ("version", pa.string()),
        ("period", _period_type()), ("dataset_period", _period_type()),
        ("field_year", pa.int16()), ("source_status", pa.string()),
        ("input_binding", pa.string()),
    ]


def _scalar_fields(prefix: str, *, boolean: bool, structured: bool):
    import pyarrow as pa

    fields = [
        (prefix + "_kind", pa.string()), (prefix + "_text", pa.large_string()),
        (prefix + "_integer", pa.int64()), (prefix + "_integer_decimal", pa.large_string()),
        (prefix + "_float", pa.float64()),
    ]
    if boolean:
        fields.append((prefix + "_boolean", pa.bool_()))
    if structured:
        fields.append((prefix + "_json", pa.large_string()))
    return fields


def analytical_schema(role: str) -> pa.Schema:
    """Return the ordinary Arrow schema used by one analytical artifact role."""
    import pyarrow as pa

    if role not in ANALYTICAL_TABLES:
        raise ContractError(f"Unknown analytical artifact role: {role}")
    table, _ = ANALYTICAL_TABLES[role]
    if table == "cells":
        fields = [
            ("entity_id", pa.string()), ("entity_index", pa.uint32()),
            ("selector_index", pa.uint32()), *_common_fields(),
            ("raw_present", pa.bool_()),
            *_scalar_fields("raw", boolean=True, structured=True),
            ("status", pa.string()), ("reason", pa.string()),
            ("policy_id", pa.string()), ("unit", pa.string()),
            ("lower", _bound_type()), ("upper", _bound_type()),
            ("measurement_raw_null", pa.bool_()), ("source_json", pa.large_string()),
            *_scalar_fields("value", boolean=False, structured=False),
            *_scalar_fields("presentation", boolean=True, structured=False),
        ]
    elif table == "attributes":
        fields = [
            ("scope", pa.string()), ("entity_id", pa.string()), ("entity_index", pa.uint32()),
            ("campus_name", pa.string()), ("campus_number", pa.string()),
            ("campus_lookup_key", pa.string()), ("campus_source_namespace", pa.string()),
            ("campus_source_id", pa.string()),
            ("district_name", pa.string()), ("district_number", pa.string()),
            ("campus_grade_range", pa.string()), ("campus_latitude", pa.float64()),
            ("campus_longitude", pa.float64()), ("district_type", pa.string()),
            ("is_charter", pa.bool_()), ("is_private", pa.bool_()),
        ]
    else:
        reference = pa.struct([
            ("reference_id", pa.string()), ("role", pa.string()), ("status", pa.string()),
            ("title", pa.string()), ("url", pa.string()), ("path", pa.string()),
            ("published_at", pa.string()), ("reviewed_sha256", pa.string()),
            ("provenance_ids", pa.list_(pa.string())),
        ])
        acquisition = pa.struct([
            ("record_id", pa.string()), ("kind", pa.string()), ("method", pa.string()),
            ("recorded_at", pa.string()), ("downloaded_at", pa.string()),
            ("requested_url", pa.string()), ("resolved_url", pa.string()),
            ("filename", pa.string()), ("sha256", pa.string()), ("size_bytes", pa.int64()),
            ("record_json", pa.large_string()),
        ])
        fields = [
            ("selector_index", pa.uint32()), *_common_fields(),
            ("unit", pa.string()), ("policy_id", pa.string()), ("population", pa.string()),
            ("title", pa.string()), ("publisher", pa.string()),
            ("source_url", pa.string()), ("download_url", pa.string()),
            ("published_at", pa.string()), ("revised_at", pa.string()),
            ("license", pa.string()), ("citation", pa.string()), ("notes", pa.string()),
            ("provenance_ids", pa.list_(pa.string())), ("references", pa.list_(reference)),
            ("acquisitions", pa.list_(acquisition)), ("metadata_json", pa.large_string()),
        ]
    return pa.schema(fields)


def _encode_scalar(value: Any, prefix: str, *, structured: bool = False) -> dict[str, Any]:
    result = {
        prefix + "_kind": "null", prefix + "_text": None,
        prefix + "_integer": None, prefix + "_integer_decimal": None,
        prefix + "_float": None, prefix + "_boolean": None, prefix + "_json": None,
    }
    if value is None:
        return result
    if type(value) is bool:
        kind, suffix, encoded = "boolean", "boolean", value
    elif type(value) is str:
        kind, suffix, encoded = "string", "text", value
    elif type(value) is int:
        kind = "integer"
        suffix, encoded = (("integer", value) if _INT64_MIN <= value <= _INT64_MAX
                           else ("integer_decimal", str(value)))
    elif type(value) is float:
        kind, suffix, encoded = "float", "float", value
    elif structured and type(value) in (dict, list):
        kind, suffix = type(value).__name__, "json"
        # Insertion order is part of canonical Python-repr/raw behavior.
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    else:
        raise TypeError(f"Unsupported analytical source value: {type(value).__name__}")
    result[prefix + "_kind"] = kind
    result[prefix + "_" + suffix] = encoded
    return result


def _decode_scalar(row: Mapping[str, Any], prefix: str) -> Any:
    kind = row[prefix + "_kind"]
    if kind == "null":
        return None
    if kind == "string":
        return row[prefix + "_text"]
    if kind == "integer":
        integer, decimal = row[prefix + "_integer"], row[prefix + "_integer_decimal"]
        if (integer is None) == (decimal is None):
            raise ContractError("An analytical integer requires exactly one integer lane")
        if decimal is None:
            return integer
        value = int(decimal)
        if str(value) != decimal:
            raise ContractError("An analytical integer must use canonical decimal text")
        return value
    if kind == "float":
        return row[prefix + "_float"]
    if kind == "boolean":
        return row[prefix + "_boolean"]
    if kind in {"dict", "list"}:
        value = json.loads(row[prefix + "_json"])
        if type(value).__name__ != kind:
            raise ContractError("Analytical JSON does not match its declared raw kind")
        return value
    raise ContractError(f"Unknown analytical value kind: {kind}")


def _encode_bound(value: int | float | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if type(value) not in (int, float):
        raise TypeError("Disclosure bounds must be integer or float values")
    encoded = _encode_scalar(value, "bound")
    return {key.removeprefix("bound_"): item for key, item in encoded.items()
            if key in {"bound_kind", "bound_integer", "bound_integer_decimal", "bound_float"}}


def _decode_bound(value: Mapping[str, Any] | None) -> int | float | None:
    if value is None:
        return None
    result = _decode_scalar({"bound_" + key: item for key, item in value.items()}, "bound")
    if type(result) not in (int, float):
        raise ContractError("An analytical disclosure bound is not numeric")
    return result


def measurement_from_row(row: Mapping[str, Any]) -> Measurement | None:
    """Materialize a sparse scalar result; this transports, never reinterprets."""
    if row["status"] is None:
        return None
    return Measurement(
        _decode_scalar(row, "value"), ValueStatus(row["status"]),
        None if row["measurement_raw_null"] else _decode_scalar(row, "raw"),
        row["reason"], row["policy_id"], _decode_bound(row["lower"]),
        _decode_bound(row["upper"]), row["unit"], row["source_json"],
    )


def raw_value_from_row(row: Mapping[str, Any]) -> Any:
    """Decode explicit raw lanes, including opaque structured source values."""
    return _decode_scalar(row, "raw")


def presentation_from_row(row: Mapping[str, Any]) -> Any:
    """Return the materialized format-boundary primitive, without Measurement."""
    return _decode_scalar(row, "presentation")


def presentation_record(value: Any) -> dict[str, Any]:
    """Encode a final format primitive using the same exact Arrow lanes.

    Structured opaque values become canonical JSON at the format boundary.
    This also transports selected custom application getters through IPC; it
    does not interpret any source value, suppression code, or dataset policy.
    """
    if isinstance(value, (dict, list)):
        value = canonical_json(value)
    # Match stdlib JSON's base scalar payload, including Enum/numpy subclasses,
    # without invoking user-defined conversion overrides or rounding integers.
    if isinstance(value, str):
        value = str.__str__(value)
    elif isinstance(value, bool):
        pass
    elif isinstance(value, int):
        value = int.__int__(value)
    elif isinstance(value, float):
        value = float.__float__(value)
        if not math.isfinite(value):
            raise ValueError("Non-finite analytical presentation value")
    encoded = _encode_scalar(value, "presentation")
    if encoded.pop("presentation_json") is not None:
        raise TypeError("An analytical presentation must be a primitive value")
    return encoded


def presentation_values(batch):
    """Decode a bounded Arrow batch at a Python format writer boundary.

    Column arrays become primitive lists once per batch; no per-cell mapping,
    Measurement, source context or JSON round trip is allocated.
    """
    indexes = [batch.schema.get_field_index(name) for name in PRESENTATION_COLUMNS]
    if -1 in indexes:
        raise ContractError("An analytical presentation batch lacks required lanes")
    arrays = [batch.column(index).to_pylist() for index in indexes]
    for kind, text, integer, decimal, real, boolean in zip(*arrays, strict=True):
        if kind == "null":
            yield None
        elif kind == "string":
            yield text
        elif kind == "integer":
            if (integer is None) == (decimal is None):
                raise ContractError("An analytical integer requires exactly one integer lane")
            if decimal is None:
                yield integer
            else:
                value = int(decimal)
                if str(value) != decimal:
                    raise ContractError("An analytical integer must use canonical decimal text")
                yield value
        elif kind == "float":
            yield real
        elif kind == "boolean":
            yield boolean
        else:
            raise ContractError(f"Unknown analytical presentation kind: {kind}")


def _cell_values(raw: Any, value: Measurement | None) -> dict[str, Any]:
    if value is not None and value.value is not None and type(value.value) not in (str, int, float):
        raise TypeError("A canonical measured value must be string, integer, float or null")
    display = raw if value is None else presentation_value(value)
    record = {
        **_encode_scalar(raw, "raw", structured=True),
        **_encode_scalar(value.value if value is not None else None, "value"),
        **presentation_record(display),
        "status": value.status.value if value is not None else None,
        "reason": value.reason if value is not None else None,
        "policy_id": value.policy_id if value is not None else None,
        "unit": value.unit if value is not None else None,
        "lower": _encode_bound(value.lower) if value is not None else None,
        "upper": _encode_bound(value.upper) if value is not None else None,
        "measurement_raw_null": value is not None and value.raw is None,
        "source_json": value._source_json if value is not None else None,
    }
    for name in ("value_boolean", "value_json"):
        if record.pop(name) is not None:
            raise TypeError("An analytical presentation must be a primitive value")
    return record


def artifact_metadata(path: str | Path) -> dict[str, Any]:
    """Read the versioned standard Parquet footer, without decoding cell rows."""
    import pyarrow.parquet as pq

    try:
        raw = pq.read_metadata(path).metadata[ANALYTICAL_METADATA]
        document = json.loads(raw)
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise ContractError("Analytical artifact metadata is missing or invalid") from error
    if (not isinstance(document, dict) or type(document.get("schema_version")) is not int
            or document["schema_version"] != ANALYTICAL_SCHEMA_VERSION):
        raise ContractError("Unsupported analytical artifact schema")
    return document


def validate_artifact(path: str | Path, catalog: DatasetCatalog, role: str,
                      *, entities_sha256: str | None = None) -> dict[str, Any]:
    """Validate format/schema/catalog/source binding; the manifest owns file hashes."""
    import hashlib
    import pyarrow.parquet as pq

    if role not in ANALYTICAL_ROLES:
        raise ContractError(f"Unknown analytical artifact role: {role}")
    document = artifact_metadata(path)
    source_hash = document.get("entities_sha256")
    if (not isinstance(source_hash, str) or len(source_hash) != 64
            or any(value not in "0123456789abcdef" for value in source_hash)):
        raise ContractError("Analytical artifacts require an exact entities checksum")
    expected_catalog = hashlib.sha256(canonical_json(catalog.to_dict()).encode()).hexdigest()
    if (document.get("role") != role or document.get("scope") != ANALYTICAL_TABLES[role][1]
            or document.get("build_id") != catalog.build_id
            or document.get("catalog_sha256") != expected_catalog
            or (entities_sha256 is not None and document.get("entities_sha256") != entities_sha256)):
        raise ContractError("Analytical artifact belongs to a different role, source, or catalog")
    if not pq.read_schema(path).remove_metadata().equals(analytical_schema(role)):
        raise ContractError("Analytical artifact columns do not match the declared schema")
    if (type(document.get("rows")) is not int or document["rows"] < 0
            or pq.read_metadata(path).num_rows != document["rows"]):
        raise ContractError("Analytical artifact row count does not match its metadata")
    producer = document.get("producer_inputs")
    if (not isinstance(producer, dict)
            or producer.get("source_processor_sha256") != catalog.to_dict()["processor_sha256"]
            or not isinstance(producer.get("processor_sha256"), str)
            or len(producer["processor_sha256"]) != 64
            or any(value not in "0123456789abcdef" for value in producer["processor_sha256"])
            or (producer.get("lock_sha256") is not None and (
                not isinstance(producer["lock_sha256"], str) or len(producer["lock_sha256"]) != 64
                or any(value not in "0123456789abcdef" for value in producer["lock_sha256"])))
            or any(not isinstance(producer.get(key), str) or not producer[key]
                   for key in ("python", "pyarrow"))):
        raise ContractError("Analytical artifact producer inputs are missing or inconsistent")
    return document


def analytical_dataset(path: str | Path, catalog: DatasetCatalog, role: str,
                       *, entities_sha256: str | None = None) -> ds.Dataset:
    """Return a standard Arrow Dataset, with native projection/filter/batch APIs."""
    import pyarrow.dataset as ds

    validate_artifact(path, catalog, role, entities_sha256=entities_sha256)
    return ds.dataset(path, format="parquet")
