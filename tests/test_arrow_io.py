"""Primitive interchange has identical validation with and without its encoder."""

from importlib.util import find_spec
import json
import subprocess
import sys

import pyarrow as pa
import pytest

from teadata import arrow_io


@pytest.mark.parametrize("fallback", [False, True])
def test_primitive_batches_preserve_values_metadata_and_nulls(monkeypatch, fallback):
    if fallback:
        def absent(name):
            raise ModuleNotFoundError(name=name)
        monkeypatch.setattr(arrow_io, "import_module", absent)
    schema = pa.schema([
        pa.field("count", pa.int64(), nullable=False, metadata={b"unit": b"count"}),
        pa.field("raw", pa.large_string()), pa.field("amount", pa.float64()),
        pa.field("observed", pa.bool_()),
    ], metadata={b"source": b"frozen"})
    rows = [{"count": 2**53 + 1, "raw": "<5", "amount": 6160.001, "observed": True},
            {"count": -7, "raw": ".", "amount": None, "observed": False}]
    batch = arrow_io.primitive_record_batch(rows, schema)
    assert batch.to_pylist() == rows
    assert batch.schema.equals(schema, check_metadata=True)
    assert arrow_io.primitive_record_batch([], schema).schema.equals(schema, check_metadata=True)
    assert arrow_io.primitive_scalar(None, pa.string()).as_py() is None


@pytest.mark.parametrize("value,data_type,error", [
    (True, pa.int64(), TypeError), (2**63, pa.int64(), OverflowError),
    (-1, pa.uint64(), OverflowError), (1, pa.float64(), TypeError),
    (float("inf"), pa.float64(), ValueError), ("\ud800", pa.string(), UnicodeEncodeError),
    (1, pa.bool_(), TypeError), (1, pa.date32(), TypeError),
])
def test_primitive_validation_precedes_optional_encoder(value, data_type, error):
    with pytest.raises(error):
        arrow_io.primitive_array([value], data_type)


def test_records_reject_unknown_fields_nullability_and_broken_encoder(monkeypatch):
    schema = pa.schema([pa.field("count", pa.int64(), nullable=False)])
    with pytest.raises(ValueError, match="undeclared"):
        arrow_io.primitive_record_batch([{"count": 1, "unknown": 2}], schema)
    with pytest.raises(ValueError, match="Non-nullable"):
        arrow_io.primitive_record_batch([{}], schema)
    with pytest.raises(ValueError, match="duplicate"):
        arrow_io.primitive_record_batch([], pa.schema([("n", pa.int64()), ("n", pa.int64())]))
    def broken(name):
        raise ModuleNotFoundError(name="missing_internal_dependency")
    monkeypatch.setattr(arrow_io, "import_module", broken)
    with pytest.raises(ModuleNotFoundError) as caught:
        arrow_io.primitive_array([1], pa.int64())
    assert caught.value.name == "missing_internal_dependency"


@pytest.mark.skipif(find_spec("nanoarrow") is None, reason="Optional encoder is not installed on this target")
def test_encoder_does_not_initialize_pandas_or_dataset():
    code = """
import json, sys
sys.modules['pandas'] = None
sys.modules['pyarrow.dataset'] = None
import pyarrow as pa
from teadata.arrow_io import primitive_record_batch
from teadata.identifiers import canonical_source_identifiers
batch = primitive_record_batch([{'id': '1902001', 'n': 9007199254740993}],
    pa.schema([('id', pa.string()), ('n', pa.int64())]))
assert canonical_source_identifiers(batch.column(0), width=9).to_pylist() == ["'001902001"]
assert batch.column(1).to_pylist() == [9007199254740993]
print(json.dumps({'pandas': sys.modules['pandas'], 'dataset': sys.modules['pyarrow.dataset']}))
"""
    actual = subprocess.run([sys.executable, "-c", code], check=True, text=True, capture_output=True)
    assert json.loads(actual.stdout) == {"pandas": None, "dataset": None}
