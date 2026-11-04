"""Typed parity admits the exact original lineage and checks all retained cells."""

from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from teadata.peims_leaves import PROGRAMS, materialize_peims_students
from teadata.provenance import ProvenanceStore
from teadata.release import DataRelease
from teadata.source_parity import validate_source_parity
from teadata.source_parity_typed import _selection
from teadata.transfer_leaves import materialize_district_transfers
from test_peims_leaves import FIELD, YEAR, fixture_config
from test_transfer_leaves import prepared_fixture


def _peims(tmp_path):
    store, config, digest, _ = fixture_config(tmp_path)
    output = tmp_path / "native"
    materialize_peims_students(store, config=config, config_sha256=digest, selections=[(PROGRAMS, YEAR)], output=output)
    return DataRelease(output / "release.json", offline=True), ProvenanceStore(output / "provenance")


def _ids(release):
    return [row["edition_id"] for row in release.catalog.editions(include_metadata=False)]


def test_complete_peims_parity_distinguishes_zero_and_suppression(tmp_path):
    release, store = _peims(tmp_path)
    report = validate_source_parity(release, store, edition_ids=_ids(release))
    assert report["status"] == "PASS_RAW_PARITY"
    assert report["coverage"]["checked_editions"] == 3
    assert not report["scientific_qualification"] and not report["publisher_authenticity_verified"]
    for edition in report["editions"]:
        assert edition["rows"] == 1
        assert edition["compared_cells"] == edition["raw_cells"] + 3
        assert edition["stages"]["artifact"][f"peims_status_{FIELD}"]["mismatches"] == 0


def test_transfer_parity_checks_prepared_and_final_fields_and_original_record_locations(tmp_path, monkeypatch):
    store, config, digest, preparations, _ = prepared_fixture(tmp_path, monkeypatch, years=(2024,))
    output = tmp_path / "native"
    materialize_district_transfers(store, config=config, config_sha256=digest, preparations=preparations, output=output)
    release = DataRelease(output / "release.json", offline=True)
    report = validate_source_parity(release, ProvenanceStore(output / "provenance"), edition_ids=_ids(release))
    assert report["status"] == "PASS_RAW_PARITY"
    result, = report["editions"]
    assert result["rows"] == 9 and result["raw_cells"] == 90
    assert len(result["lineage"]["acquisitions"]) == 2
    assert set(result["stages"]) == {"prepared", "artifact"}
    for stage in result["stages"].values():
        assert stage["transfer_source_record"]["checked"] == 9
        assert stage["transfer_count_status"]["checked"] == 9
    assert result["stages"]["artifact"]["student_transfer_count"]["checked"] == 9


@pytest.mark.parametrize("budget,empty", [(64 * 1024, False), (128 * 1024**2, True)])
def test_typed_parity_holds_on_missing_lineage_or_too_small_budget(tmp_path, budget, empty):
    release, store = _peims(tmp_path)
    if empty:
        store = ProvenanceStore(tmp_path / "empty")
    report = validate_source_parity(release, store, edition_ids=_ids(release), working_bytes=budget)
    assert report["status"] == "HOLD" and report["coverage"]["uncovered_editions"] == 3


@pytest.mark.parametrize("field, replacement", [(f"peims_raw_{FIELD}", "0"), (f"peims_status_{FIELD}", "observed"),
                                               (FIELD, 0), ("peims_source_record", 5)])
def test_typed_parity_rejects_wrong_native_cells_after_descriptor_validation(tmp_path, monkeypatch, field, replacement):
    release, store = _peims(tmp_path)
    changed = tmp_path / "wrong.parquet"

    def fixture_selection(*args, **kwargs):
        selected = _selection(*args, **kwargs)
        # Inject an incorrect but well-typed target after the receipt checks
        # to exercise full cell/NULL/locator checking independently of checksums.
        native = pq.read_table(selected.artifact)
        index = native.schema.get_field_index(field)
        schema_field = native.schema.field(index)
        pq.write_table(native.set_column(index, schema_field, pa.array([replacement], type=schema_field.type)), changed)
        return replace(selected, artifact=changed)

    monkeypatch.setattr("teadata.source_parity_typed._selection", fixture_selection)
    campus, = [row["edition_id"] for row in release.catalog.editions(include_metadata=False) if row["scope"] == "campus"]
    result, = validate_source_parity(release, store, edition_ids=[campus], max_examples=1)["editions"]
    assert result["status"] == "FAIL_MISMATCH"
    assert result["mismatches"] == 1
    example, = result["stages"]["artifact"][field]["examples"]
    assert example["source_record"] == 3 and example["actual"] == replacement


def test_a_valid_leaf_from_an_unsupported_producer_remains_uncovered(tmp_path, monkeypatch):
    record_processing = ProvenanceStore.record_processing

    def alternate_producer(store, **kwargs):
        if kwargs["processor"] == "teadata.scripts.refresh_peims":
            kwargs["processor"] = "external.project.peims"
        return record_processing(store, **kwargs)

    monkeypatch.setattr(ProvenanceStore, "record_processing", alternate_producer)
    release, store = _peims(tmp_path)
    release.verify()
    report = validate_source_parity(release, store, edition_ids=_ids(release))
    assert report["coverage"]["uncovered_editions"] == 3
    assert all("No supported recorded typed replay recipe" == item["diagnostic"] for item in report["editions"])


def test_equal_bytes_from_an_unbound_peims_receipt_do_not_count_as_lineage(tmp_path, monkeypatch):
    release, store = _peims(tmp_path)
    acquisitions = release.catalog.acquisitions

    def alternate_receipt(*args, **kwargs):
        original, = acquisitions(*args, **kwargs)
        return [{**original, "record_id": "1" * 64}]

    monkeypatch.setattr(release.catalog, "acquisitions", alternate_receipt)
    report = validate_source_parity(release, store, edition_ids=_ids(release))
    assert report["coverage"]["failed_editions"] == 3
    assert all("original receipt or field period differs" in item["diagnostic"] for item in report["editions"])
