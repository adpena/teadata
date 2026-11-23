"""Plain projections keep integrity, complete ID validation and owned handles."""

from contextlib import contextmanager, closing
from datetime import datetime, time, timedelta, timezone
from importlib.util import find_spec
from io import BytesIO
import json
import subprocess
import sys
from types import SimpleNamespace

import pyarrow as pa
import pytest

from teadata.manifest import ContractError
from teadata.release import DataRelease
from teadata.source_views import SourceTable, _TableExport, write_csv, write_excel
from test_catalog_projections import make_edition_fixture


def test_plain_projection_matches_dataset_with_full_precision_and_status(tmp_path):
    fixture = make_edition_fixture(tmp_path, families=("student",), normalized=True, observed_amount=2**53 + 1)
    release = DataRelease(fixture["root"], offline=True)
    key, _, _ = fixture["leaves"]["student"]
    view = release.table(key).select("amount")
    actual = view.to_arrow()
    from pyarrow.dataset import scalar
    prior_executor = view.where(scalar(True)).to_arrow()
    assert actual.schema.remove_metadata().equals(prior_executor.schema.remove_metadata(), check_metadata=True)
    assert actual.replace_schema_metadata(None).equals(prior_executor.replace_schema_metadata(None))
    assert [0, None, None, 2**53 + 1] == actual["amount"].to_pylist()
    assert actual["amount_status"].to_pylist() == ["observed", "suppressed", "missing", "observed"]
    receipt = json.loads(actual.schema.metadata[b"teadata_provenance"])
    assert receipt == view.provenance().to_dict()
    assert receipt["selector"]["entity_release"] is None
    footer = view._prepare().schema.metadata
    assert footer and all(actual.schema.metadata[key] == value for key, value in footer.items())


@pytest.mark.parametrize("ids,expected", [(('000000001',), [0]), (('000000004',), [2**53 + 1]), ((), [])])
def test_plain_cohort_normalizes_raw_aliases_without_rewriting_source_ids(tmp_path, ids, expected):
    fixture = make_edition_fixture(tmp_path, families=("student",), normalized=True,
                                   identifiers=("1", "'000000002", "003", "000000004"), observed_amount=2**53 + 1,
                                   entity_grain=True)
    release = DataRelease(fixture["root"], offline=True)
    key, _, _ = fixture["leaves"]["student"]
    actual = SourceTable(release, key, _entity_ids=ids).select("amount").to_arrow()
    assert actual["amount"].to_pylist() == expected
    if ids == ('000000001',):
        assert actual["campus_number"].to_pylist() == ["1"]


def test_cohort_validates_unselected_ids_before_yielding_any_rows(tmp_path):
    fixture = make_edition_fixture(tmp_path, families=("student",), normalized=True,
                                   identifiers=("1", "2", "3", "not-an-id"), entity_grain=True)
    release = DataRelease(fixture["root"], offline=True)
    key, _, _ = fixture["leaves"]["student"]
    with closing(SourceTable(release, key, _entity_ids=("000000001",)).select("amount").iter_batches()) as batches:
        with pytest.raises(ValueError, match="identifier"):
            next(batches)


def test_preparation_closes_handles_and_consumption_owns_one_until_early_close(tmp_path, monkeypatch):
    fixture = make_edition_fixture(tmp_path, families=("student",), normalized=True)
    release = DataRelease(fixture["root"], offline=True)
    key, _, _ = fixture["leaves"]["student"]
    opened, active = [], []
    original = DataRelease._open_artifact
    @contextmanager
    def tracked(self, identity):
        with original(self, identity) as pair:
            opened.append(pair[0])
            active.append(pair[0])
            try:
                yield pair
            finally:
                active.remove(pair[0])
    monkeypatch.setattr(DataRelease, "_open_artifact", tracked)
    export = release.table(key).select("amount")._prepare()
    assert len(opened) == 1 and opened[0].closed and active == []
    batches = export.batches()
    assert len(opened) == 1
    next(batches)
    assert len(active) == 1 and not active[0].closed
    batches.close()
    assert active == [] and all(stream.closed for stream in opened)


@pytest.mark.parametrize("consumer", ["csv", "excel", "arrow"])
def test_consumer_failure_closes_retained_batch_iterator_immediately(consumer):
    closed = []
    # Arrow receives the wrong type; CSV rejects list values; Excel rejects a
    # source string that cannot roundtrip. Keep the iterator alive after error.
    schema = pa.schema([("value", pa.string())])
    if consumer == "csv":
        batch = pa.record_batch([pa.array([[1]])], names=["value"])
    elif consumer == "arrow":
        batch = pa.record_batch([pa.array([1])], names=["value"])
    else:
        batch = pa.record_batch([pa.array(["bad\x01text"])], names=["value"])
    def source():
        try:
            yield batch
        finally:
            closed.append(True)
    retained = source()
    export = SimpleNamespace(schema=schema, restore_original_headers=False,
        headers=(("value",),), rows_hint=None, batches=lambda: retained,
        provenance=lambda: SimpleNamespace(_append_sheet=lambda *a, **k: None, to_dict=lambda: {}))
    with pytest.raises((pa.ArrowException, ValueError, TypeError)):
        if consumer == "arrow":
            _TableExport.to_arrow(export)
        else:
            (write_csv if consumer == "csv" else write_excel)(export, BytesIO())
    assert closed == [True]
    assert list(retained) == []


def test_excel_preserves_timezone_aware_arrow_timestamps_as_text():
    from openpyxl import load_workbook

    retrieved = datetime(2026, 11, 21, 18, 25, 13, 912345, tzinfo=timezone.utc)
    offset = datetime(2026, 11, 22, 1, 2, 3, 123456, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    local = datetime(2026, 11, 21, 18, 25, 13)
    clock = time(8, 15, 23)
    batch = pa.record_batch({
        "retrieved": pa.array([retrieved, None], type=pa.timestamp("us", tz="UTC")),
        "offset": pa.array([offset, None], type=pa.timestamp("us", tz="+05:30")),
        "local": pa.array([local, None], type=pa.timestamp("us")),
        "clock": pa.array([clock, clock], type=pa.time64("us")),
    })
    export = SimpleNamespace(schema=batch.schema, headers=(tuple(batch.schema.names),), rows_hint=2,
                             batches=lambda: iter([batch]),
                             provenance=lambda: SimpleNamespace(_append_sheet=lambda *a, **k: None))
    output = BytesIO()
    write_excel(export, output)
    output.seek(0)
    with closing(load_workbook(output, read_only=True)) as workbook:
        rows = workbook["Data"].iter_rows(max_col=4)
        assert [cell.value for cell in next(rows)] == batch.schema.names
        first = next(rows)
        assert [cell.value for cell in first] == [retrieved.isoformat(), offset.isoformat(), local, clock]
        assert [cell.data_type for cell in first] == ["s", "s", "d", "d"]
        assert [cell.value for cell in next(rows)] == [None, None, None, clock]


def test_changed_artifact_after_preparation_fails_instead_of_reusing_stale_reader(tmp_path):
    fixture = make_edition_fixture(tmp_path, families=("student",), normalized=True)
    release = DataRelease(fixture["root"], offline=True)
    key, _, _ = fixture["leaves"]["student"]
    export = release.table(key).select("amount")._prepare()
    path = release.artifact(export.scanner.artifact.identity)
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ContractError):
        list(export.batches())


@pytest.mark.skipif(find_spec("nanoarrow") is None, reason="Optional encoder is not installed on this target")
def test_plain_source_selection_and_series_do_not_initialize_dataset_or_pandas(tmp_path):
    fixture = make_edition_fixture(tmp_path, families=("student",), normalized=True,
                                   amount_period={"kind": "school_year", "start_year": 2024, "end_year": 2025},
                                   entity_grain=True)
    code = """
import sys
sys.modules['pandas'] = None
sys.modules['pyarrow.dataset'] = None
from teadata.release import DataRelease
from teadata.source_views import SourceTable
release = DataRelease(sys.argv[1], offline=True)
key = next(iter(release.catalog.editions(scope='campus', include_metadata=False)))
from teadata.dataset_metadata import EditionKey
edition = EditionKey.from_dict({name: key[name] for name in EditionKey.__dataclass_fields__})
table = SourceTable(release, edition, _entity_ids=('000000001',)).select('amount')
assert table[['amount']].to_arrow()['amount'].to_pylist() == [0]
series = release.series('campus_student', 'amount', scope='campus', years=[25], ids=[1])
assert series.to_arrow()['amount'].to_pylist() == [0]
"""
    result = subprocess.run([sys.executable, "-c", code, str(fixture["root"])], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
