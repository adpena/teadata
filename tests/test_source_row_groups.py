from contextlib import closing

import pyarrow.parquet as pq
import pytest

from teadata.release import DataRelease

from test_catalog_projections import make_edition_fixture


def _record_scans(monkeypatch):
    original = pq.ParquetFile
    scans = []

    class ObservedFile:
        def __init__(self, *args, **kwargs):
            self.file = original(*args, **kwargs)

        def __getattr__(self, name):
            if name in {"read", "read_row_group", "read_row_groups"}:
                pytest.fail("Entity selection must keep bounded batch reads")
            return getattr(self.file, name)

        def __enter__(self):
            self.file.__enter__()
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def iter_batches(self, **kwargs):
            record = {**kwargs, "rows_decoded": 0, "closed": False}
            scans.append(record)
            batches = self.file.iter_batches(**kwargs)
            try:
                for batch in batches:
                    record["rows_decoded"] += batch.num_rows
                    yield batch
            finally:
                batches.close()
                record["closed"] = True

    monkeypatch.setattr(pq, "ParquetFile", ObservedFile)
    return scans


@pytest.mark.parametrize(
    ("identifiers", "selected", "groups", "values"),
    [
        (["001", "002", "003", "004"], ("000000001",), [0], [0]),
        (["001", "002", "003", "004"], ("000000004", "000000001"), [0, 3], [0, 2**53 + 1]),
        ([" '000000001 ", "1.0", "002", "1"], ("000000001",), [0, 1, 3], [0, None, 2**53 + 1]),
        (["001", "002", "003", "004"], ("000000009",), [], []),
        (["001", "002", "003", "004"], (), [], []),
        (["001", "002", "003", "004"], None, None, [0, None, None, 2**53 + 1]),
    ],
)
def test_entity_batches_decode_only_matching_groups_after_complete_key_validation(
    tmp_path, monkeypatch, identifiers, selected, groups, values,
):
    fixture = make_edition_fixture(
        tmp_path, families=("student",), normalized=True, entity_grain=True,
        identifiers=identifiers, row_group_rows=1, observed_amount=2**53 + 1,
    )
    release = DataRelease(fixture["root"], offline=True)
    edition, _, _ = fixture["leaves"]["student"]
    view = release.table(edition).select("amount")
    if selected is not None:
        view = view._for_entity_ids(selected)
    expected = view.to_arrow()
    scans = _record_scans(monkeypatch)
    batches = list(view.iter_batches())
    assert [value for batch in batches for value in batch["amount"].to_pylist()] == values
    assert [row["amount"] for row in expected.to_pylist()] == values
    key_scans = [call for call in scans if call["columns"] == ["campus_number"]]
    data_scans = [call for call in scans if call["columns"] != ["campus_number"]]
    if selected:
        assert [call["row_groups"] for call in key_scans] == [[0], [1], [2], [3]]
        assert sum(call["rows_decoded"] for call in key_scans) == 4
        assert all(call["batch_size"] == 65_536 for call in key_scans)
    else:
        assert key_scans == []
    if values:
        assert len(data_scans) == 1 and data_scans[0]["row_groups"] == groups
        assert data_scans[0]["batch_size"] == 256
        assert data_scans[0]["rows_decoded"] == len(values)
    else:
        assert data_scans == []
    assert all(call["closed"] and call["use_threads"] is False for call in scans)


def test_malformed_unselected_group_fails_before_any_data_is_emitted(tmp_path, monkeypatch):
    fixture = make_edition_fixture(
        tmp_path, families=("student",), normalized=True, entity_grain=True,
        identifiers=["001", "002", "003", "not-an-id"], row_group_rows=1,
    )
    release = DataRelease(fixture["root"], offline=True)
    edition, _, _ = fixture["leaves"]["student"]
    view = release.table(edition).select("amount")._for_entity_ids(("000000001",))
    scans = _record_scans(monkeypatch)
    with closing(view.iter_batches()) as batches, pytest.raises(ValueError, match="Invalid source identifier"):
        next(batches)
    assert len(scans) == 4
    assert all(call["columns"] == ["campus_number"] and call["closed"] for call in scans)


def test_selected_group_reader_closes_when_consumer_stops_early(tmp_path, monkeypatch):
    fixture = make_edition_fixture(
        tmp_path, families=("student",), normalized=True, entity_grain=True, row_group_rows=1,
    )
    release = DataRelease(fixture["root"], offline=True)
    edition, _, _ = fixture["leaves"]["student"]
    view = release.table(edition).select("amount")._for_entity_ids(("000000001", "000000004"))
    scans = _record_scans(monkeypatch)
    with closing(view.iter_batches()) as batches:
        assert next(batches).num_rows == 2
    assert all(call["closed"] for call in scans)
