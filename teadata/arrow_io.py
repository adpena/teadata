"""Typed primitive Arrow interchange without an implicit DataFrame conversion.

The optional ``speedups`` encoder uses the public Arrow C Data interface. The
ordinary PyArrow encoder remains available when nanoarrow is not installed.
Both implementations receive the same validated values; neither interprets
source periods, masking, measurement units or policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from math import isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa


def _primitive_kind(data_type: pa.DataType) -> str:
    import pyarrow as pa

    if not isinstance(data_type, pa.DataType):
        raise TypeError("Provide an explicit Arrow DataType, not a type name.")
    if pa.types.is_boolean(data_type):
        return "boolean"
    if pa.types.is_integer(data_type) and data_type.bit_width in {8, 16, 32, 64}:
        return "integer"
    if pa.types.is_float64(data_type):
        return "float"
    if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        return "string"
    raise TypeError(
        "Primitive Arrow interchange supports booleans, 8–64-bit integers, "
        "float64 and strings. Use PyArrow explicitly for other types."
    )


def primitive_array(values: Sequence[object], data_type: pa.DataType) -> pa.Array:
    """Encode explicitly typed builtin primitives, preserving nulls and precision.

    Booleans are not integers, integers are not implicitly narrowed to floats,
    and non-finite floats and invalid UTF-8 fail. A broken installed accelerator
    propagates its error; only a genuinely absent optional package selects the
    ordinary encoder.
    """
    import pyarrow as pa

    kind = _primitive_kind(data_type)
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError("Primitive Arrow values must be a sequence of scalar values.")
    if kind == "integer":
        signed = pa.types.is_signed_integer(data_type)
        lower = -(1 << (data_type.bit_width - 1)) if signed else 0
        upper = (1 << (data_type.bit_width - int(signed))) - 1
    for position, value in enumerate(values):
        if value is None:
            continue
        if kind == "integer":
            if type(value) is not int:
                raise TypeError(f"Arrow integer value at position {position} must be a builtin int.")
            if not lower <= value <= upper:
                raise OverflowError(f"Arrow integer value at position {position} exceeds its declared type.")
        elif kind == "boolean" and type(value) is not bool:
            raise TypeError(f"Arrow boolean value at position {position} must be a builtin bool.")
        elif kind == "float":
            if type(value) is not float:
                raise TypeError(f"Arrow float64 value at position {position} must be a builtin float.")
            if not isfinite(value):
                raise ValueError(f"Arrow float64 value at position {position} must be finite.")
        elif kind == "string":
            if type(value) is not str:
                raise TypeError(f"Arrow string value at position {position} must be a builtin str.")
            value.encode("utf-8", errors="strict")
    try:
        encoder = import_module("nanoarrow")
    except ModuleNotFoundError as error:
        if error.name != "nanoarrow":
            raise
        return pa.array(values, type=data_type)
    return pa.array(encoder.c_array(values, data_type))


def primitive_scalar(value: object, data_type: pa.DataType) -> pa.Scalar:
    """Build one explicitly typed scalar through the same primitive contract."""
    return primitive_array([value], data_type)[0]


def primitive_record_batch(
    rows: Sequence[Mapping[str, object]], schema: pa.Schema,
) -> pa.RecordBatch:
    """Encode flat primitive records without losing unknown keys or nullability.

    Missing nullable fields become null. Schema and field metadata, column order
    and native buffer ownership are preserved, including for an empty batch.
    """
    import pyarrow as pa

    if not isinstance(schema, pa.Schema):
        raise TypeError("Provide an explicit Arrow Schema.")
    names = schema.names
    if len(names) != len(set(names)):
        raise ValueError("Primitive Arrow schema contains duplicate field names.")
    for field in schema:
        _primitive_kind(field.type)
    if isinstance(rows, (str, bytes, bytearray)) or not isinstance(rows, Sequence):
        raise TypeError("Primitive Arrow records must be a sequence of mappings.")
    if rows and not names:
        raise ValueError("Nonempty primitive records require at least one declared field.")
    allowed = set(names)
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"Arrow record at position {position} must be a mapping.")
        if set(row) - allowed:
            raise ValueError(f"Arrow record at position {position} contains undeclared fields.")
    columns = []
    for field in schema:
        values = [row.get(field.name) for row in rows]
        if not field.nullable and any(value is None for value in values):
            raise ValueError(f"Non-nullable Arrow field {field.name!r} contains a missing value.")
        columns.append(primitive_array(values, field.type))
    return pa.RecordBatch.from_arrays(columns, schema=schema)
