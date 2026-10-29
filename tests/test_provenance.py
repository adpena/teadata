from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path

import pytest

from teadata.dataset_metadata import DatasetMetadata, ResolvedDataset
from teadata.provenance import ProvenanceStore


NOW = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


def source_definition(path: Path) -> ResolvedDataset:
    return ResolvedDataset(
        dataset="example",
        section="data_sources",
        requested_year=2025,
        resolved_year=2024,
        path=str(path),
        metadata=DatasetMetadata.from_dict({"title": "Example source"}),
    )


def test_processing_diagnostics_are_explicit_per_run_receipt_evidence(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("value\n42\n")
    output = tmp_path / "model.parquet"
    output.write_bytes(b"owned synthetic output")
    store = ProvenanceStore(tmp_path / "provenance")
    acquired = store.record_acquisition(source, source_definition(source), recorded_at=NOW)
    common = dict(dataset="derived", inputs=[acquired], outputs={"model": output},
                  processor="canonical", code_version="reviewed", recorded_at=NOW,
                  validation={"cells": 1})
    without = store.record_processing(**common)
    first = store.record_processing(**common, diagnostics={"analytical": {"seconds": 1.0}})
    second = store.record_processing(**common, diagnostics={"analytical": {"seconds": 2.0}})
    a, b, c = [store.read_record(identifier) for identifier in (without, first, second)]
    assert "diagnostics" not in a
    assert b["validation"] == c["validation"] == {"cells": 1}
    assert b["outputs"] == c["outputs"]
    assert b["diagnostics"] != c["diagnostics"] and first != second


def test_local_observation_preserves_bytes_without_inventing_download_history(tmp_path):
    source = tmp_path / "raw.csv"
    source.write_bytes(b"value\n42\n")
    store = ProvenanceStore(tmp_path / "provenance")
    record_id = store.record_acquisition(
        source, source_definition(source), recorded_at=NOW
    )
    record = store.read_record(record_id)
    assert record["method"] == "local_observation"
    assert record["downloaded_at"] is None
    assert record["requested_url"] is None
    assert record["resolved_year"] == 2024
    assert record["artifact"]["sha256"] == sha256(source.read_bytes()).hexdigest()
    assert str(tmp_path) not in json.dumps(record)
    archived = store.verify_artifact(record["artifact"]["sha256"])
    source.unlink()
    assert archived.read_bytes() == b"value\n42\n"


def test_repeated_downloads_and_same_url_revisions_keep_independent_history(tmp_path):
    source = tmp_path / "raw.csv"
    store = ProvenanceStore(tmp_path / "history")
    source.write_bytes(b"first version")
    ids = []
    for index, contents in enumerate(
        (b"first version", b"first version", b"revised version")
    ):
        source.write_bytes(contents)
        ids.append(
            store.record_acquisition(
                source,
                source_definition(source),
                downloaded_at=NOW + timedelta(days=index),
                recorded_at=NOW + timedelta(days=index, seconds=1),
                requested_url="https://example.org/current.csv",
                resolved_url="https://cdn.example.org/current.csv",
                response_headers={
                    "ETag": f"version-{index}",
                    "Set-Cookie": "private-cookie",
                    "Authorization": "private-token",
                },
            )
        )
    records = store.history("example")
    assert [record["record_id"] for record in records] == ids
    assert len(set(ids)) == 3
    assert len(list((store.root / "objects").iterdir())) == 2
    assert records[0]["artifact"]["sha256"] == records[1]["artifact"]["sha256"]
    assert records[0]["artifact"]["sha256"] != records[2]["artifact"]["sha256"]
    assert (
        store.verify_artifact(records[0]["artifact"]["sha256"]).read_bytes()
        == b"first version"
    )
    assert records[0]["response_headers"] == {"etag": "version-0"}


def test_processing_links_multiple_inputs_code_parameters_validation_and_outputs(
    tmp_path,
):
    store = ProvenanceStore(tmp_path / "history")
    source = tmp_path / "raw.csv"
    source.write_text("raw")
    input_id = store.record_acquisition(
        source, source_definition(source), recorded_at=NOW
    )
    output = tmp_path / "normalized.csv"
    output.write_text("normalized")
    receipt_id = store.record_processing(
        dataset="example",
        inputs=[input_id],
        outputs={"normalized": output},
        processor="example.normalize",
        code_version="git:abcdef",
        parameters={"year": 2024},
        validation={"schema": "passed", "rows": 1},
        recorded_at=NOW + timedelta(seconds=1),
    )
    receipt = store.read_record(receipt_id)
    assert receipt["inputs"] == [input_id]
    assert receipt["code_version"] == "git:abcdef"
    assert receipt["parameters"] == {"year": 2024}
    assert receipt["validation"]["schema"] == "passed"
    assert (
        store.verify_artifact(receipt["outputs"]["normalized"]["sha256"]).read_text()
        == "normalized"
    )


def test_missing_validation_is_unknown_not_implicitly_passed(tmp_path):
    source = tmp_path / "raw.csv"
    source.write_text("data")
    store = ProvenanceStore(tmp_path / "history")
    input_id = store.record_acquisition(
        source, source_definition(source), recorded_at=NOW
    )
    output_id = store.record_processing(
        dataset="example",
        inputs=[input_id],
        outputs={"same": source},
        processor="copy",
        code_version="v1",
        recorded_at=NOW,
    )
    assert store.read_record(output_id)["validation"] is None


def test_repeating_identical_record_is_idempotent_and_does_not_overwrite(tmp_path):
    source = tmp_path / "raw.csv"
    source.write_text("data")
    store = ProvenanceStore(tmp_path / "history")
    first = store.record_acquisition(source, source_definition(source), recorded_at=NOW)
    path = store.root / "records" / f"{first}.json"
    before = path.stat().st_mtime_ns
    second = store.record_acquisition(
        source, source_definition(source), recorded_at=NOW
    )
    assert first == second
    assert path.stat().st_mtime_ns == before


def test_tampered_receipts_and_objects_are_rejected(tmp_path):
    source = tmp_path / "raw.csv"
    source.write_text("data")
    store = ProvenanceStore(tmp_path / "history")
    record_id = store.record_acquisition(
        source, source_definition(source), recorded_at=NOW
    )
    record = store.read_record(record_id)
    (store.root / "objects" / record["artifact"]["sha256"]).write_text("tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        store.verify_artifact(record["artifact"]["sha256"])
    with pytest.raises(ValueError, match="Conflicting immutable"):
        store.record_acquisition(source, source_definition(source), recorded_at=NOW)
    (store.root / "records" / f"{record_id}.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        store.read_record(record_id)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"recorded_at": NOW.replace(tzinfo=None)},
        {"recorded_at": NOW, "downloaded_at": NOW + timedelta(seconds=1)},
        {"recorded_at": NOW, "requested_url": "https://example.org/current.csv"},
        {"recorded_at": NOW, "downloaded_at": NOW, "requested_url": "file:raw.csv"},
    ],
)
def test_invalid_acquisition_facts_are_rejected_before_archiving(tmp_path, kwargs):
    source = tmp_path / "raw.csv"
    source.write_text("data")
    store = ProvenanceStore(tmp_path / "history")
    with pytest.raises(ValueError):
        store.record_acquisition(source, source_definition(source), **kwargs)
    assert not store.root.exists()


def test_lineage_rejects_unknown_input_records(tmp_path):
    source = tmp_path / "raw.csv"
    source.write_text("data")
    store = ProvenanceStore(tmp_path / "history")
    with pytest.raises(FileNotFoundError):
        store.record_processing(
            dataset="example",
            inputs=["0" * 64],
            outputs={"normalized": source},
            processor="normalize",
            code_version="v1",
        )


def test_record_identifier_cannot_escape_store(tmp_path):
    with pytest.raises(ValueError, match="SHA-256"):
        ProvenanceStore(tmp_path).read_record("../outside")
