"""Tests of edition-scoped endpoint admission, with a separate publisher readback."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from teadata import campus_transfer_leaves as leaves
from teadata.dataset_metadata import DataPeriod, DatasetMetadata, ResolvedDataset
from teadata.manifest import ContractError
from teadata.provenance import ProvenanceStore, validate_record_graph, validate_reference_records
from teadata.teadata_config import load_config


def test_packaged_2025_campus_transfer_source_profile_keeps_the_default_contract(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "teadata/teadata_sources.yaml")
    default = config.resolve_dataset(leaves.DATASET, 2025, exact=True)
    source = config.resolve_dataset(leaves.DATASET, 2025, reader_profile="source", exact=True)
    assert source.path == default.path and default.path.endswith("xfers_campus_2024_25.csv")
    assert default.metadata.metrics["count"].field == "TRANSFERS_IN_OR_OUT"
    assert source.metadata.metrics["count"].field == "student_transfer_count"
    assert source.metadata.period == DataPeriod("school_year", 2024, 2025)
    assert not config.is_source_only(leaves.DATASET, 2025)
    assert config.is_source_only(leaves.DATASET, 2025, reader_profile="source")
    fields = config.column_schema(leaves.DATASET, 2025, reader_profile="source")
    assert fields is not None and len(fields) == 14
    read = ProvenanceStore(root / "provenance").read_record
    original, = source.metadata.provenance_ids
    directory, = [ref for ref in source.metadata.references if ref.reference_id == leaves.DIRECTORY_REFERENCE]
    receipt, = directory.provenance_ids
    archive_index = read(receipt)["parameters"]["archive_index_record"]
    roots = [original, archive_index, *(item for ref in source.metadata.references for item in ref.provenance_ids)]
    graph = validate_record_graph(read, roots)
    validate_reference_records(source, graph.__getitem__, verify_all_claimed=True)
    assert graph[original]["dataset"] == leaves.DATASET and graph[original]["method"] == "download"
    assert graph[receipt]["source_metadata"]["version"] == "2025-fall"
    assert graph[archive_index]["parameters"]["http_method"] == "GET"

    opened = []

    def read_default(path, **kwargs):
        opened.append(path)
        return pd.DataFrame(columns=config.column_schema(leaves.DATASET, 2025))

    monkeypatch.setattr(config, "_load_table", read_default)
    with pytest.raises(ContractError, match="requires a dedicated preparation workflow"):
        config.load_df(leaves.DATASET, 2025, exact=True, reader_profile="source")
    assert not opened
    year, _ = config.load_df(leaves.DATASET, 2025, exact=True)
    assert year == 2025 and opened == [default.path]


@pytest.fixture
def endpoint_inputs(monkeypatch, tmp_path):
    class Registry:
        def __init__(self, path, *, immutable):
            assert path == tmp_path / "entities.sqlite" and immutable

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def iter_campuses(self, *, include_meta):
            assert not include_meta
            return iter([{"lookup_key": "000101001", "is_private": False}])

        def get_district(self, district, *, include_meta, cache):
            assert not include_meta and not cache
            return {"district_number": district} if district == "'000202" else None

    monkeypatch.setattr(leaves, "EntityStore", Registry)
    table = pa.table({
        "transfer_row_kind": ["detail", "detail", "separator"],
        "transfer_reporting_number": ["'000101001", "'000202001", "'000202001"],
        "transfer_partner_number": ["'000202001", "'000101001", None],
        "transfer_reporting_district_number": ["'000101", "'000202", "'000202"],
        "transfer_partner_district_number": ["'000202", "'000101", None],
        "REPORT_CAMPUS_NAME": ["Current", "Historical", "Ignored separator"],
        "CAMPNAME_RES_OR_ATTEND": ["Historical", "Current", ""],
    })
    directory = "School Number,District Number,School Name\n000202001,000202,Historical\n"
    return table, directory, tmp_path / "entities.sqlite"


def test_historical_endpoint_context_uses_detail_rows_only(endpoint_inputs):
    table, directory, entities = endpoint_inputs
    result, evidence = leaves._endpoint_context(table, directory, "2024-10-31", entities)
    assert result.select(table.column_names).equals(table)
    assert evidence == {"'000202001": {"record": 2, "district": "'000202", "name": "Historical"}}
    assert result["reporting_campus_askted_record"].to_pylist() == [None, 2, None]
    assert result["partner_campus_askted_record"].to_pylist() == [2, None, None]
    assert result["historical_askted_archive_as_of"].to_pylist() == ["2024-10-31", "2024-10-31", None]


@pytest.mark.parametrize(("rows", "message"), [
    ("000202002,000202,Other\n", "Missing exact fall AskTED evidence"),
    ("000202001,000202,Historical\n000202001,000202,Historical\n", "ambiguous or invalid"),
    ("000202001,000101,Historical\n", "ambiguous or invalid"),
    ("000202001,000202,Other\n", "disagrees with fall AskTED identity"),
])
def test_endpoint_context_rejects_missing_or_conflicting_directory_evidence(endpoint_inputs, rows, message):
    table, _, entities = endpoint_inputs
    directory = "School Number,District Number,School Name\n" + rows
    with pytest.raises(ValueError, match=message):
        leaves._endpoint_context(table, directory, "2024-10-31", entities)


def test_fall_directory_requires_archive_html_to_agree_with_the_csv_receipt(tmp_path):
    store = ProvenanceStore(tmp_path / "provenance")
    now = datetime.now(timezone.utc)
    url = leaves.ARCHIVE_URL
    index_path = tmp_path / "archive.html"
    index_path.write_text('<table><tr><td>2024-2025 School Year</td><td><a href="javascript:__doPostBack('
                          "'fall','')\">Fall (as of 10/30/2024)</a></td></tr></table>")
    index = store.record_acquisition(index_path, ResolvedDataset("public_school_directory", "data_sources", 2025, 2025, "",
        DatasetMetadata(title="Fixture archive index", source_url=url)), downloaded_at=now, requested_url=url,
        resolved_url=url, response_headers={"content-type": "text/html"}, parameters={"http_method": "GET", "http_status": 200})
    csv_path = tmp_path / "directory.csv"
    csv_path.write_text("School Number,District Number,School Name\n000202001,000202,Historical\n")
    receipt = store.record_acquisition(csv_path, ResolvedDataset("public_school_directory", "data_sources", 2025, 2025, "",
        DatasetMetadata(title="Fixture fall directory", source_url=url, version="2025-fall",
                        period=DataPeriod("as_of", as_of="2024-10-31"))), downloaded_at=now, requested_url=url,
        resolved_url=url, parameters={"http_method": "POST", "http_status": 200, "archive_index_record": index,
                                     "school_year_end": 2025, "season": "fall", "publisher_declared_as_of": "2024-10-31"})
    source = ResolvedDataset(leaves.DATASET, "data_sources", 2025, 2025, "", DatasetMetadata.from_dict({
        "title": "Fixture transfer report", "references": [{"reference_id": leaves.DIRECTORY_REFERENCE,
            "role": "source_notes", "url": url, "provenance_ids": [receipt],
            "reviewed_sha256": store.read_record(receipt)["artifact"]["sha256"]}],
    }))
    with pytest.raises(ValueError, match="reviewed publisher fall edition"):
        leaves._fall_directory(store, source, 2025)


def test_historical_hold_aggregates_both_roles_and_preserves_all_reasons(endpoint_inputs, monkeypatch, tmp_path, capsys):
    from teadata.scripts.refresh_transfers import main

    table, _, entities = endpoint_inputs
    changed = table.set_column(
        table.column_names.index("transfer_reporting_number"), "transfer_reporting_number",
        pa.array(["'000203001", "'000202001", "'000202001"]),
    )
    directory = "School Number,District Number,School Name\n000202001,000202,Other Name\n"
    output = tmp_path / "candidate"

    def fail(*args, **kwargs):
        return leaves._endpoint_context(changed, directory, "2024-10-31", entities)

    monkeypatch.setattr(leaves, "materialize_campus_transfers", fail)
    assert main(["materialize-campus", "--store", str(tmp_path), "--config", str(tmp_path / "config"),
                 "--config-sha256", "a" * 64, "--year", "2025", "--preparation", "b" * 64,
                 "--registry-manifest", str(tmp_path / "release.json"), "--registry-sha256", "c" * 64,
                 "--output", str(output)]) == 2
    import json

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "HOLD_HISTORICAL_TRANSFER_ENDPOINTS"
    assert result["affected_detail_rows"] == 2
    assert result["school_year_end"] == 2025
    assert result["problems"] == [
        {"reason": "conflicting_transfer_identity", "endpoint_ids": ["'000202001"],
         "affected_detail_rows": 2, "reporting_rows": 1, "partner_rows": 1},
        {"reason": "missing_fall_evidence", "endpoint_ids": ["'000203001"],
         "affected_detail_rows": 1, "reporting_rows": 1, "partner_rows": 0},
    ]
    assert not output.exists()


def test_source_replay_validates_fall_rows_and_both_native_locator_roles(endpoint_inputs):
    from teadata.source_parity_typed import _CampusHistory, _append_historical_locators, _validate_fall_endpoints

    table, directory, _ = endpoint_inputs
    claimed = {"'000202001": {"record": 2, "district": "'000202", "name": "Historical"}}
    historical = _validate_fall_endpoints(directory, claimed)
    seen = set()
    projected = _append_historical_locators(table, _CampusHistory("2024-10-31", historical), seen)
    assert seen == set(claimed)
    assert projected["reporting_campus_askted_record"].to_pylist() == [None, 2, None]
    assert projected["partner_campus_askted_record"].to_pylist() == [2, None, None]
    assert projected["historical_askted_archive_as_of"].to_pylist() == ["2024-10-31", "2024-10-31", None]
    with pytest.raises(ValueError, match="exact fall archive rows"):
        _validate_fall_endpoints(directory, {"'000202001": {**claimed["'000202001"], "record": 3}})
    with pytest.raises(ValueError, match="ambiguous or malformed fall identity"):
        _validate_fall_endpoints(directory + "000202001,000202,Historical\n", claimed)
    malformed = "School Number,District Number,School Name\n000202001,000202\n"
    with pytest.raises(ValueError, match="ambiguous or malformed fall identity"):
        _validate_fall_endpoints(malformed, claimed)


def test_fall_endpoint_locator_preserves_blank_logical_record(endpoint_inputs):
    from teadata.source_parity_typed import _validate_fall_endpoints

    table, directory, entities = endpoint_inputs
    directory = directory.replace("\n000202001", "\n\n000202001")
    _, evidence = leaves._endpoint_context(table, directory, "2024-10-31", entities)
    assert evidence["'000202001"]["record"] == 3
    assert _validate_fall_endpoints(directory, evidence) == evidence


def test_campus_witness_is_pinned_to_policy_and_input_even_when_reports_agree():
    from hashlib import sha256
    import json

    from teadata.source_parity_typed import _check_campus_directional_witness

    cfg = load_config()
    source = cfg.resolve_dataset(leaves.DATASET, 2025, reader_profile="source", exact=True)
    policy = cfg.value_policies[source.metadata.value_policy]
    witness = {"status": "PASS", "raw_count_token_differences": 0, "input_sha256": "a" * 64}
    report = {"source_validation": witness,
              "policy_sha256": sha256(json.dumps(policy.to_dict(), sort_keys=True).encode()).hexdigest()}
    validation = {"directional_readback": witness}
    _check_campus_directional_witness(report, validation, policy, "a" * 64)
    with pytest.raises(ValueError, match="directional witness"):
        _check_campus_directional_witness(report, validation, policy, "b" * 64)
    with pytest.raises(ValueError, match="directional witness"):
        _check_campus_directional_witness({**report, "policy_sha256": "c" * 64}, validation, policy, "a" * 64)
    with pytest.raises(ValueError, match="directional witness"):
        _check_campus_directional_witness(report, None, policy, "a" * 64)


def test_obsolete_context_is_separate_from_fall_identity(endpoint_inputs, monkeypatch):
    from dataclasses import replace
    from datetime import date

    from teadata.askted_lifecycle import LifecycleEvent, LifecycleEventKind, LifecycleParticipant, LifecycleReport
    from teadata.dataset_metadata import ReferenceDocument

    table, directory, entities = endpoint_inputs
    table, historical = leaves._endpoint_context(table, directory, "2024-10-31", entities)
    receipt, digest = "a" * 64, "b" * 64
    reference = ReferenceDocument.from_dict({
        "reference_id": leaves.LIFECYCLE_REFERENCE, "title": "AskTED All Public Schools Closed, printed 2026-11-07", "role": "source_notes",
        "status": "available", "url": leaves._PORTAL, "provenance_ids": [receipt], "reviewed_sha256": digest,
    })
    assert reference is not None

    def event(record, day, *, name="Historical", kind=LifecycleEventKind.OBSOLETE):
        return LifecycleEvent(record, "campus", kind, "Obsolete Date", str(day), day,
                              (LifecycleParticipant("000202001", name, "subject"),), ())

    def read(_store, actual_receipt, *, report, expected_sha256):
        assert (actual_receipt, expected_sha256) == (receipt, digest)
        return LifecycleReport(report, date(2026, 11, 7), receipt, digest, tuple(events))

    monkeypatch.setattr(leaves, "read_lifecycle_report", read)
    store = object()
    events = [event(4, date(2024, 10, 31)), event(5, date(2025, 1, 8), name="Different identity"),
              event(6, date(2025, 1, 8), kind=LifecycleEventKind.DORMANT)]
    with pytest.raises(ValueError, match="citation differs"):
        leaves._campus_obsolescence(store, replace(reference, title="AskTED All Public Schools Closed, printed 2026-11-06"),
                                   as_of="2024-10-31", historical=historical)
    empty = leaves._campus_obsolescence(store, reference, as_of="2024-10-31", historical=historical)
    assert all(leaves._append_campus_obsolescence(table, empty)[name].null_count == 3
               for name in leaves._OBSOLESCENCE_FIELDS)
    events.append(event(7, date(2025, 1, 8)))
    evidence = leaves._campus_obsolescence(store, reference, as_of="2024-10-31", historical=historical)
    result = leaves._append_campus_obsolescence(table, evidence)
    assert result.select(table.column_names).equals(table)
    assert [result[name].to_pylist() for name in leaves._OBSOLESCENCE_FIELDS] == [
        [None, 7, None], [None, "2025-01-08", None], [7, None, None], ["2025-01-08", None, None],
    ]
    for extra, message in ((event(8, date(2025, 1, 8)), "Ambiguous"),
                           (event(8, None), "Unresolved"), (event(8, date(2026, 11, 8)), "Unresolved")):
        events.append(extra)
        with pytest.raises(ValueError, match=message):
            leaves._campus_obsolescence(store, reference, as_of="2024-10-31", historical=historical)
        events.pop()


@pytest.mark.parametrize(("filename", "contents", "message"), [
    ("source-config.yaml", b"data_sources: [", "unexpected filename"),
    ("source-config.input", b"data_sources: [", "cannot be parsed"),
    ("source-config.input", b"year_min: .inf", "cannot be parsed"),
])
def test_campus_replay_rejects_malformed_retained_config(filename, contents, message, monkeypatch):
    import json

    from teadata import source_parity_typed as replay

    source = load_config().resolve_dataset(leaves.DATASET, 2025, reader_profile="source", exact=True)
    directory_ref, = [item for item in source.metadata.references if item.reference_id == leaves.DIRECTORY_REFERENCE]
    directory_id, = directory_ref.provenance_ids
    receipts = [directory_id, "a" * 64]
    directory = "School Number,District Number,School Name\n000202001,000202,Historical\n"
    as_of = "2024-10-31"
    manifest = {"artifacts": {"entities": {"sha256": "b" * 64}}, "catalog": {"build_id": "c" * 64}}
    companion = {"filename": "companion-registry-manifest.input", "sha256": "d" * 64, "size_bytes": 1}
    companion_id = "e" * 64
    graph = {identifier: {"kind": "acquisition", "dataset": "reference:askted", "artifact": {"size_bytes": 1}}
             for identifier in receipts}
    graph[directory_id]["parameters"] = {"archive_index_record": receipts[1]}
    graph[companion_id] = {"kind": "acquisition", "dataset": "reference:transfer_companion-registry-manifest",
                           "artifact": companion}
    recipe = {"directory_receipts": receipts, "directory_archive_as_of": as_of,
              "registry_manifest_sha256": companion["sha256"], "registry_entities": manifest["artifacts"]["entities"],
              "registry_build_id": manifest["catalog"]["build_id"],
              "historical_endpoints": {"'000202001": {"record": 2, "district": "'000202", "name": "Historical"}}}
    monkeypatch.setattr(replay, "_fall_directory", lambda *_: (directory, as_of, tuple(receipts)))
    monkeypatch.setattr(replay, "_resource", lambda _store, identity, *_args: json.dumps(manifest).encode()
                        if identity == companion else contents)
    with pytest.raises(ValueError, match=message):
        replay._campus_history(object(), source, 2025, recipe, graph, [*receipts, companion_id],
                               {"filename": filename}, 4096)
