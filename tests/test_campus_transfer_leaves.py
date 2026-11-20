"""Tests of edition-scoped endpoint admission, with a separate publisher readback."""

from datetime import datetime, timezone
from collections import Counter
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
    result, evidence, _ = leaves._endpoint_context(table, directory, "2024-10-31", entities)
    assert result.select(table.column_names).equals(table)
    assert evidence == {"'000202001": {"record": 2, "district": "'000202", "name": "Historical"}}
    assert result["reporting_campus_askted_record"].to_pylist() == [None, 2, None]
    assert result["partner_campus_askted_record"].to_pylist() == [2, None, None]
    assert result["historical_askted_archive_as_of"].to_pylist() == ["2024-10-31", "2024-10-31", None]


def test_reviewed_archive_preserves_source_labels_and_role_counts(endpoint_inputs):
    from teadata.source_parity_typed import _CampusHistory, _append_historical_locators

    table, _, entities = endpoint_inputs
    wire = "A" * 34
    long = wire + " LONG"
    directory = f"School Number,District Number,School Name\n000202001,000202,{long}\n"
    supplemental = "School Number,District Number,School Name\n000202001,000202,Earlier Name\n"
    archive = leaves._NameArchive(supplemental, ("a" * 64, "b" * 64), "c" * 64, 34)
    table = table.set_column(table.schema.get_field_index("REPORT_CAMPUS_NAME"), "REPORT_CAMPUS_NAME",
                             pa.array(["Current", wire, "Ignored separator"]))
    table = table.set_column(table.schema.get_field_index("CAMPNAME_RES_OR_ATTEND"), "CAMPNAME_RES_OR_ATTEND",
                             pa.array(["Earlier Name", "Current", ""]))
    counts = Counter()
    projected, fall, names = leaves._endpoint_context(table, directory, "2020-10-31", entities,
                                                     archives={"supplemental": archive}, proofs=counts)
    assert projected.select(table.column_names).equals(table)
    assert projected["reporting_campus_askted_record"].to_pylist() == [None, 2, None]
    assert {(p.role, p.archive, p.clipped, p.label, n) for p, n in counts.items()} == {
        ("transfer_reporting_number", "fall", True, wire, 1),
        ("transfer_partner_number", "supplemental", False, "Earlier Name", 1),
    }
    verified = Counter()
    history = _CampusHistory("2020-10-31", fall, names=names)
    assert _append_historical_locators(table, history, set(), verified).equals(projected)
    assert archive.recipe(verified) == archive.recipe(counts)
    witness = {"name": "Earlier Name", "district": "'999999", "record": 2}
    for label, alternate, width in ((wire[:-1], None, 34), (wire, None, None), ("Earlier Name", witness, 34)):
        assert leaves._match_name("reporting", "'000202001", "'000202", label, fall["'000202001"],
                                  {"supplemental": alternate} if alternate else {}, width) is None


def test_reviewed_parent_proof_binds_exact_campuses_and_spring_record(monkeypatch):
    from hashlib import sha256

    campus = "'220819003"
    district = "'220819"
    label = "HIGH POINT ACADEMY FW EL"
    fall = {campus: {"record": 8343, "district": district, "name": label}}
    signature = sha256(leaves.canonical_json([campus]).encode()).hexdigest()
    monkeypatch.setitem(leaves._REVIEWED_ENDPOINTS, 2023, leaves._ReviewedEndpoints("fall", signature, (2021, 2023)))
    monkeypatch.setattr(leaves, "_HISTORICAL_PARENT_CAMPUSES", frozenset([campus]))
    names = leaves._NameEvidence({}, {campus: {
        "spring_2021": {"record": 6200, "district": district, "name": label},
        "spring_2023": {"record": 8393, "district": district, "name": label},
    }})
    assert leaves._reviewed_parent_proof(2023, leaves._REVIEWED_REGISTRY, fall, names) == {
        campus: {"district": district, "fall_record": 8343, "spring_record": 8393}}
    with pytest.raises(ValueError, match="companion registry"):
        leaves._reviewed_parent_proof(2023, leaves._REVIEWED_REGISTRY, {"'220819004": fall[campus]}, names)
    wrong_archive = leaves._NameEvidence({}, {campus: {"spring_2021": names.aliases[campus]["spring_2021"]}})
    with pytest.raises(ValueError, match="matching fall and spring"):
        leaves._reviewed_parent_proof(2023, leaves._REVIEWED_REGISTRY, fall, wrong_archive)


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
    _, evidence, _ = leaves._endpoint_context(table, directory, "2024-10-31", entities)
    assert evidence["'000202001"]["record"] == 3
    assert _validate_fall_endpoints(directory, evidence) == evidence


def test_campus_witness_is_pinned_to_policy_and_input_even_when_reports_agree():
    from hashlib import sha256
    import json

    from teadata.campus_transfer_admission import preparation_witness

    cfg = load_config()
    source = cfg.resolve_dataset(leaves.DATASET, 2025, reader_profile="source", exact=True)
    policy = cfg.value_policies[source.metadata.value_policy]
    witness = {"status": "PASS", "raw_count_token_differences": 0, "input_sha256": "a" * 64}
    qualifications = {"prepared_source_qualified": True, "canonical_edges_qualified": True, "raw_string_readback_equal": True}
    report = {"status": "prepared", **qualifications, "source_validation": witness,
              "policy_sha256": sha256(json.dumps(policy.to_dict(), sort_keys=True).encode()).hexdigest()}
    preparation = {"validation": {"status": "PASS", **qualifications}, "outputs": {}}
    assert preparation_witness(2025, "receipt", "a" * 64, "preparation", preparation, report, policy) == witness
    with pytest.raises(ValueError, match="directional witness"):
        preparation_witness(2025, "receipt", "b" * 64, "preparation", preparation, report, policy)
    with pytest.raises(ValueError, match="directional witness"):
        preparation_witness(2025, "receipt", "a" * 64, "preparation", preparation, {**report, "policy_sha256": "c" * 64}, policy)
    with pytest.raises(ValueError, match="directional witness"):
        preparation_witness(2025, "receipt", "a" * 64, "preparation", {}, report, policy)
    for malformed in (None, [], "PASS"):
        with pytest.raises(ValueError, match="directional witness"):
            preparation_witness(2025, "receipt", "a" * 64, "preparation", preparation,
                                {**report, "source_validation": malformed}, policy)


def test_reviewed_masking_hold_admits_directional_preparation_only():
    from copy import deepcopy
    from hashlib import sha256
    import json

    from teadata.campus_transfer_admission import _REVIEWED, preparation_witness

    reviewed = _REVIEWED[2023]
    policy = load_config().value_policies["campus_transfers"]
    directions = {"status": "HOLD", "input_sha256": reviewed.original, "status_mismatches": 4,
                  "raw_count_token_differences": 4, "value_mismatches": 2, "unknown_count_rows": 0,
                  "duplicate_out_keys": 0, "duplicate_in_keys": 0, "one_sided_pairs": 0,
                  "historical_district_mismatches": 0,
                  "examples": [{"source_record_out": i, "source_record_in": i + 10,
                                "count_status_out": "suppressed", "count_status_in": "observed"}
                               for i in (4, 3, 2, 1)]}
    qualifications = {"status": "HOLD", "prepared_source_qualified": True, "canonical_edges_qualified": False,
                      "raw_string_readback_equal": True}
    preparation = {"validation": qualifications, "outputs": {
        "prepared_source": {"sha256": reviewed.prepared}, "annotations": {"sha256": reviewed.annotations},
        "validation_report": {"sha256": reviewed.report}}}
    report = {**qualifications, "source_validation": directions,
              "policy_sha256": sha256(json.dumps(policy.to_dict(), sort_keys=True).encode()).hexdigest()}
    args = (2023, reviewed.acquisition, reviewed.original, reviewed.processing, preparation)
    result = preparation_witness(*args, report, policy)
    assert result["value_mismatches"] == 0 and directions["value_mismatches"] == 2
    assert [row["source_record_in"] for row in result["examples"]] == [11, 12, 13, 14]
    with pytest.raises(ValueError, match="reviewed masking-only hold"):
        preparation_witness(*args[:3], "another-preparation", preparation, report, policy)
    for altered in ({**directions, "one_sided_pairs": 1}, {**directions, "examples": directions["examples"][:3]},
                    {**directions, "historical_district_mismatches": 1}):
        with pytest.raises(ValueError, match="reviewed masking-only hold"):
            preparation_witness(*args, {**report, "source_validation": altered}, policy)
    wrong_artifact = deepcopy(preparation)
    wrong_artifact["outputs"]["validation_report"]["sha256"] = reviewed.annotations
    with pytest.raises(ValueError, match="reviewed masking-only hold"):
        preparation_witness(*args[:4], wrong_artifact, report, policy)


def test_obsolete_context_is_separate_from_fall_identity(endpoint_inputs, monkeypatch):
    from dataclasses import replace
    from datetime import date

    from teadata.askted_lifecycle import LifecycleEvent, LifecycleEventKind, LifecycleParticipant, LifecycleReport
    from teadata.dataset_metadata import ReferenceDocument

    table, directory, entities = endpoint_inputs
    table, historical, _ = leaves._endpoint_context(table, directory, "2024-10-31", entities)
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
    ("source-config.input", None, "Malformed supplemental response"),
])
def test_campus_replay_rejects_malformed_retained_config(filename, contents, message, monkeypatch):
    import json

    from teadata import source_parity_typed as replay
    from teadata.acquisition import AcquisitionError

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
    if contents is None:
        contents = (Path(__file__).resolve().parents[1] / "teadata/teadata_sources.yaml").read_bytes()

        def malformed_archive(*_):
            raise AcquisitionError(message)

        monkeypatch.setattr(replay, "_name_archive", malformed_archive)
    monkeypatch.setattr(replay, "_resource", lambda _store, identity, *_args: json.dumps(manifest).encode()
                        if identity == companion else contents)
    with pytest.raises(ValueError, match=message):
        replay._campus_history(object(), source, 2025, recipe, graph, [*receipts, companion_id],
                               {"filename": filename}, 4096, "f" * 64)
