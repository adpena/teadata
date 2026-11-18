"""One-file, streaming research exports keep values and existing lineage together."""
import csv
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from io import BytesIO, StringIO
import json
from zipfile import ZipFile

import pyarrow as pa
import pytest

from teadata.manifest import canonical_json
from teadata.release import DataRelease
from teadata.source_views import write_csv_bundle

from test_catalog_projections import make_edition_fixture
from test_partitioned_series import routed_series as routed_series
from test_source_parity import fixture_release


def unpack(target, expected_csv, expected_provenance):
    with ZipFile(target) as archive:
        assert archive.namelist() == ["data.csv", "provenance.json", "manifest.json"]
        assert archive.testzip() is None
        assert archive.read("data.csv") == expected_csv
        assert archive.read("provenance.json") == canonical_json(expected_provenance).encode("utf-8")
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest == {"schema_version": 1, "format": "teadata-csv-bundle", "members": {
            name: {"sha256": sha256(archive.read(name)).hexdigest(), "size_bytes": len(archive.read(name))}
            for name in ("data.csv", "provenance.json")}}


@pytest.mark.parametrize("publisher_headers", [False, True])
def test_native_bundle_uses_exact_csv_and_canonical_provenance(tmp_path, monkeypatch, publisher_headers):
    if publisher_headers:
        release, _, key, _ = fixture_release(tmp_path, schema_version=4, first_membership=str(2**53 + 1))
        view = release.table(key).select("enrollment", "membership").query(
            "membership != @token", params={"token": "not-a-publisher-token"}, record=True)
    else:
        fixture = make_edition_fixture(tmp_path, families=("student",), normalized=True, observed_amount=2**53 + 1)
        release = DataRelease(fixture["root"], offline=True)
        key, _, _ = fixture["leaves"]["student"]
        view = release.table(key).select("amount").query("amount >= 0 or amount is None", record=True)
    expected_csv = BytesIO()
    view.to_csv(expected_csv, restore_original_headers=publisher_headers)
    expected_provenance = view.provenance(restore_original_headers=publisher_headers).to_dict()
    monkeypatch.setattr(type(view), "to_arrow", lambda *_: pytest.fail("Bundle collected observations"))
    monkeypatch.setattr(type(view), "to_df", lambda *_: pytest.fail("Bundle collected a DataFrame"))
    stream = BytesIO()
    view.to_csv_bundle(stream, restore_original_headers=publisher_headers)
    assert not stream.closed
    unpack(stream, expected_csv.getvalue(), expected_provenance)
    assert b"9007199254740993" in expected_csv.getvalue()
    assert expected_provenance["selector"]["query_replay"]["restore_original_headers"] is publisher_headers


@pytest.mark.parametrize("kind", ["partitioned", "series"])
def test_routed_bundle_retains_all_partitions_and_years(routed_series, tmp_path, monkeypatch, kind):
    parent, _ = routed_series
    view = parent.series("tprs_class_size", "amount", scope="campus", years=[25, 23], ids=[2, 4, 102, 104])
    if kind == "partitioned":
        view = view.for_year(25)
    raw = BytesIO()
    view.to_csv(raw)
    provenance = view.provenance()
    expected = provenance if isinstance(provenance, dict) else provenance.to_dict()
    monkeypatch.setattr(type(view), "to_arrow", lambda *_: pytest.fail("Bundle collected observations"))
    path = tmp_path / "selection.zip"
    view.to_csv_bundle(path)
    unpack(path, raw.getvalue(), expected)
    rows = list(csv.DictReader(StringIO(raw.getvalue().decode())))
    assert len(rows) == (8 if kind == "series" else 4)
    assert {row["amount_status"] for row in rows} == {"observed", "suppressed"}
    if kind == "series":
        assert {row["measurement_school_year"] for row in rows} == {"2023", "2025"}
        assert expected["cross_year_comparability"] == "unassessed"


@dataclass
class ExactExport:
    """Writer boundary golden: literal tokens, precision and one owned iterator."""
    schema = pa.schema([("id", pa.string()), ("amount", pa.decimal128(25, 6)), ("raw", pa.string())])
    headers = (tuple(schema.names),)
    restore_original_headers = False
    rows_hint = None
    fail: bool = False
    during_write: Callable[[], None] | None = None
    started: int = 0
    closed: int = 0

    def provenance(self):
        return self

    def to_dict(self):
        return {"selection": "existing canonical provenance", "source": "https://example.org/official.csv"}

    def batches(self):
        self.started += 1
        try:
            yield pa.record_batch([
                pa.array(["000000001", "000000002", "000000003", "000000004"]),
                pa.array([Decimal("0"), Decimal("-1234567890123.456789"), None, None], type=self.schema.field("amount").type),
                pa.array(["0", "-1234567890123.456789", "<5", "."]),
            ], schema=self.schema)
            if self.during_write is not None:
                self.during_write()
            if self.fail:
                raise RuntimeError("interrupted source")
        finally:
            self.closed += 1


def test_bundle_streams_once_and_preserves_literals_precision_and_ids(tmp_path):
    source = ExactExport()
    path = tmp_path / "exact.zip"
    write_csv_bundle(source, path)
    with ZipFile(path) as archive:
        rows = list(csv.DictReader(StringIO(archive.read("data.csv").decode())))
    assert rows == [
        {"id": "000000001", "amount": "0.000000", "raw": "0"},
        {"id": "000000002", "amount": "-1234567890123.456789", "raw": "-1234567890123.456789"},
        {"id": "000000003", "amount": "", "raw": "<5"},
        {"id": "000000004", "amount": "", "raw": "."},
    ]
    assert source.started == source.closed == 1


def test_failed_bundle_closes_source_and_never_publishes_partial_file(tmp_path):
    source = ExactExport(fail=True)
    path = tmp_path / "failed.zip"
    with pytest.raises(RuntimeError, match="interrupted source"):
        write_csv_bundle(source, path)
    assert source.started == source.closed == 1
    assert not path.exists()
    assert not list(tmp_path.glob(".teadata-export-*"))
    stream = BytesIO()
    with pytest.raises(RuntimeError, match="interrupted source"):
        write_csv_bundle(source, stream)
    assert not stream.closed


@pytest.mark.parametrize("destination", ["existing", "symlink", "racing_writer"])
def test_bundle_never_overwrites_another_destination(tmp_path, destination):
    path = tmp_path / "owned.zip"
    if destination == "symlink":
        other = tmp_path / "target"
        other.write_bytes(b"preserved")
        path.symlink_to(other)
    elif destination == "existing":
        path.write_bytes(b"preserved")
    callback = (lambda: path.write_bytes(b"preserved")) if destination == "racing_writer" else None
    source = ExactExport(during_write=callback)
    with pytest.raises(FileExistsError):
        write_csv_bundle(source, path)
    assert path.read_bytes() == b"preserved"
    assert not list(tmp_path.glob(".teadata-export-*"))


def test_empty_bundle_has_csv_header_and_lineage(tmp_path):
    source = ExactExport()
    source.batches = lambda: iter(())
    output = BytesIO()
    write_csv_bundle(source, output)
    with ZipFile(output) as archive:
        assert archive.read("data.csv") == b'"id","amount","raw"\n'
        assert json.loads(archive.read("provenance.json")) == source.to_dict()
