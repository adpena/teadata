from contextlib import closing
import json

import pyarrow as pa
from pyarrow.dataset import field
import pytest

from teadata.release import DataRelease
from teadata.source_views import _provenance_batches

from test_catalog_projections import make_edition_fixture
from test_partitioned_series import routed_series as routed_series


def receipt(view):
    value = view.provenance()
    return value if isinstance(value, dict) else value.to_dict()


def test_single_view_stream_is_lazy_filtered_exact_and_carries_receipt(tmp_path, monkeypatch):
    large = 2**53 + 1
    fixture = make_edition_fixture(tmp_path, families=("student",), normalized=True, observed_amount=large)
    release = DataRelease(fixture["root"], offline=True)
    key, _, _ = fixture["leaves"]["student"]
    view = release.table(key).select("amount")
    expected = view.to_arrow()
    source = DataRelease.analytical_dataset
    opened = []

    def open_leaf(self, identity):
        opened.append(identity)
        return source(self, identity)

    monkeypatch.setattr(DataRelease, "analytical_dataset", open_leaf)
    monkeypatch.setattr(type(view), "to_arrow", lambda *_: pytest.fail("Stream collected a table"))
    stream = view.iter_batches()
    assert opened == []
    with closing(stream):
        batches = list(stream)
    assert len(opened) == 1
    assert pa.Table.from_batches(batches).equals(expected)
    assert [value for batch in batches for value in batch["amount"].to_pylist()] == [0, None, None, large]
    assert all(json.loads(batch.schema.metadata[b"teadata_provenance"]) == receipt(view) for batch in batches)
    assert set(batches[0].schema.names) >= {"amount", "amount_status", "VALUE", "source_record", "row_kind"}
    filtered = view.where(field("amount") == large)
    assert [value for batch in filtered.iter_batches() for value in batch["amount"].to_pylist()] == [large]
    empty = view.where(field("amount") > large)
    assert sum(batch.num_rows for batch in empty.iter_batches()) == 0
    assert receipt(empty)["selector"]["opaque_filter"] is True


@pytest.mark.parametrize("kind", ["series", "partitioned"])
def test_multi_source_batches_retain_rows_receipts_and_close_before_later_leaves(routed_series, monkeypatch, kind):
    release, _ = routed_series
    series = release.series("tprs_class_size", "amount", scope="campus", years=[25, 23], ids=[2, 4, 102, 104])
    view = series if kind == "series" else series.for_year(25)
    expected = view.to_arrow()
    batches = list(view.iter_batches())
    assert pa.Table.from_batches(batches).equals(expected)
    assert all(json.loads(batch.schema.metadata[b"teadata_provenance"]) == receipt(view) for batch in batches)
    source = DataRelease.analytical_dataset
    opened = []

    def open_leaf(self, identity):
        opened.append(identity)
        if len(opened) > 1:
            raise RuntimeError("later leaf unavailable")
        return source(self, identity)

    monkeypatch.setattr(DataRelease, "analytical_dataset", open_leaf)
    stream = view.iter_batches()
    assert opened == []
    with closing(stream):
        assert next(stream).num_rows == 2
    assert len(opened) == 1
    assert list(stream) == []
    opened.clear()
    with closing(view.iter_batches()) as failing:
        next(failing)
        with pytest.raises(RuntimeError, match="later leaf unavailable"):
            next(failing)
    assert len(opened) == 2


@pytest.mark.parametrize("finish", ["exhaust", "early", "source_error", "encode_error"])
def test_batch_receipts_share_buffers_preserve_metadata_and_close_input(finish):
    original = pa.record_batch([pa.array([0, None, 2**53 + 1], type=pa.int64())], names=["value"])
    original = original.replace_schema_metadata({b"publisher": b"retained"})
    closed = []

    def source():
        try:
            yield original
            if finish == "source_error":
                raise RuntimeError("interrupted")
            yield original
        finally:
            closed.append(True)

    source_stream = source()
    # Retain and start the input so forwarding close is observable even when
    # receipt encoding fails before the helper starts consuming it.
    next(source_stream)
    provenance = {"invalid": object()} if finish == "encode_error" else {"pinned": "release"}
    stream = _provenance_batches(source_stream, provenance)
    if finish == "encode_error":
        with pytest.raises(TypeError):
            next(stream)
    elif finish == "source_error":
        with pytest.raises(RuntimeError, match="interrupted"):
            next(stream)
    else:
        batch = next(stream)
        assert batch.column(0).buffers()[1].address == original.column(0).buffers()[1].address
        assert batch.column(0).equals(original.column(0))
        assert batch.schema.metadata[b"publisher"] == b"retained"
        assert json.loads(batch.schema.metadata[b"teadata_provenance"]) == provenance
        if finish == "early":
            stream.close()
        else:
            assert list(stream) == []
    assert closed == [True]
