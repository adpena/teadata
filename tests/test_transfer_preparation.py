"""Synthetic publisher-shaped source tests; these do not establish TEA values."""

import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import pytest

from teadata.dataset_metadata import DataPeriod, DatasetMetadata, ResolvedDataset
from teadata.manifest import processor_identity
from teadata.provenance import ProvenanceStore
from teadata.scripts.refresh_transfers import main
from teadata.transfer_sources import (
    TransferValidationError, compare_campus_sources, prepare_transfer_sources,
    read_transfer_table, validate_campus_directions, validate_district_rows,
)
from teadata.values import ValuePolicy


CAMPUS_HEADER = ["YEAR", "REPORT_REGION", "REPORT_DISTRICT", "REPORT_DISTRICT_NAME", "REPORT_CAMPUS",
    "REPORT_CAMPUS_NAME", "REPORT_NUMBER", "REPORT_TYPE", "LINE_GROUP_NUMBER", "DISTRICT_RES_OR_ATTEND",
    "DISTNAME_RES_OR_ATTEND", "CAMPUS_RES_OR_ATTEND", "CAMPNAME_RES_OR_ATTEND", "TRANSFERS_IN_OR_OUT"]
DISTRICT_HEADER = ["YEAR", "REPORT_REGION", "REPORT_REGION_NAME", "REPORT_DISTNAME_NUMBER", "REPORT_CHARTER_STATUS",
    "REPORT_NUMBER", "REPORT_TYPE", "LINE_GROUP_NUMBER", "DISTNAME_NUMBER_RES_ATTEND", "TRANSFERS_IN_OUT"]
NOTES = [
    " -999 indicates counts or percentages are not available (i.e. masked) to comply with the Family Educational Rights and Privacy Act (FERPA).",
    "Masked numbers are typically small although larger numbers may be masked to prevent imputation.",
    "The REPORT_NUMBER column and the LINE_GROUP_NUMBER column are added to assist sorting and filtering of rows in the file.",
]
POLICY = {"policy_id": "campus_transfers", "reference_id": "masking", "tokens": {
    "-999": {"status": "suppressed", "reason": "publisher_mask_no_documented_bounds"},
    ".": {"status": "missing", "reason": "separator_or_missing_count"}}}


def pair_rows(out="0", incoming=None):
    incoming = out if incoming is None else incoming
    return [
        ["2024-2025", "01", "001902", "Source ISD", "001902001", 'Name, with "quote"\nand newline',
         "200", "Transfers Out To", "100", "002902", "Destination ISD", "002902002", "Destination", out],
        ["2024-2025", "02", "002902", "Destination ISD", "002902002", "Destination",
         "100", "Transfers In From", "100", "001902", "Source ISD", "001902001", 'Name, with "quote"\nand newline', incoming],
    ]


def original(tmp_path, rows, *, grain="campus", legacy=False, filename="source.csv", report_scope=None,
             campus_count_header=CAMPUS_HEADER[-1], padded_annotations=False):
    path = tmp_path / filename
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        header = [*CAMPUS_HEADER[:-1], campus_count_header] if grain == "campus" else DISTRICT_HEADER
        if not legacy:
            preamble = ["T E X A S  E D U C A T I O N  A G E N C Y", "", f"{report_scope or grain.title()} Level Transfer Report",
                        "", "PEIMS Student Transfer Report", ""]
            if padded_annotations:
                writer.writerows([text, *[""] * (len(header) - 1)] for text in preamble)
            else:
                stream.write("\n".join(preamble) + "\n")
        writer.writerow(header)
        writer.writerows(rows)
        if not legacy:
            if padded_annotations:
                writer.writerows([text, *[""] * (len(header) - 1)] for text in ["", NOTES[0], "", NOTES[1], "", NOTES[2]])
            else:
                stream.write("\n" + "\n\n".join(NOTES) + "\n\n")
    return path, sha256(path.read_bytes()).hexdigest()


def read(path, digest, *, grain="campus", year=2025, layout="original", region=None):
    return read_transfer_table(path, grain=grain, year=year, policy=ValuePolicy.from_dict(POLICY),
        expected_sha256=digest, encoding="utf-8", layout=layout, region=region)


def test_campus_original_preserves_source_strings_notes_coordinates_and_zero(tmp_path):
    rows = pair_rows()
    rows.append(["2024-2025", "01", "001902", "Source ISD", "001902001", "Source",
                 "200", "", "200", "", "", "", "", "."])
    path, digest = original(tmp_path, rows)
    source = read(path, digest)
    assert source.table["REPORT_DISTRICT"].to_pylist() == ["001902", "002902", "001902"]
    assert source.table["REPORT_CAMPUS_NAME"][0].as_py() == rows[0][5]
    assert source.table["transfer_reporting_number"][0].as_py() == "'001902001"
    assert source.table["transfer_count_value"].to_pylist() == [0, 0, None]
    assert source.table["transfer_count_status"].to_pylist() == ["observed", "observed", "missing"]
    assert source.table["transfer_source_record"].to_pylist() == [8, 9, 10]
    assert source.table["transfer_first_line"].to_pylist() == [8, 10, 12]
    assert source.table["transfer_last_line"].to_pylist() == [9, 11, 12]
    assert [note["text"].strip() for note in source.annotations if note["kind"] == "footer"] == [note.strip() for note in NOTES]
    assert validate_campus_directions(source)["matching_observed"] == 1


@pytest.mark.parametrize("year", [2021, 2022])
def test_historical_campus_padding_preserves_literal_annotations_and_counts(tmp_path, year):
    rows = [[f"{year-1}-{year}", *row[1:]] for row in pair_rows("-999")]
    path, digest = original(tmp_path, rows, padded_annotations=True)
    source = read(path, digest, year=year)
    assert source.table["transfer_count_status"].to_pylist() == ["suppressed", "suppressed"]
    assert source.table["transfer_source_record"].to_pylist() == [8, 9]
    assert source.table["transfer_first_line"].to_pylist() == [8, 10]
    assert source.table["transfer_last_line"].to_pylist() == [9, 11]
    footnotes = [note for note in source.annotations if note["kind"] == "footer"]
    assert [next(csv.reader([note["text"]]))[0] for note in footnotes] == NOTES
    assert all(note["raw_sha256"] == sha256(note["text"].encode()).hexdigest() for note in footnotes)
    with pytest.raises(ValueError, match="preamble/header schema"):
        read(path, digest, year=2023)


def test_historical_campus_padding_rejects_content_after_footer_begins(tmp_path):
    rows = [["2020-2021", *row[1:]] for row in pair_rows()]
    path, _ = original(tmp_path, [rows[0], [""] * len(CAMPUS_HEADER), rows[1]], padded_annotations=True)
    with pytest.raises(ValueError, match="data row after transfer footer"):
        read(path, sha256(path.read_bytes()).hexdigest(), year=2021)


def test_2026_campus_original_requires_its_literal_count_header(tmp_path):
    rows = [["2025-2026", *row[1:]] for row in pair_rows("-999")]
    path, digest = original(tmp_path, rows, campus_count_header="TRANSFERS_IN_OUT")
    source = read(path, digest, year=2026)

    assert source.table["TRANSFERS_IN_OUT"].to_pylist() == ["-999", "-999"]
    assert "TRANSFERS_IN_OR_OUT" not in source.table.column_names
    assert source.table["transfer_count_status"].to_pylist() == ["suppressed", "suppressed"]
    assert source.table["transfer_count_value"].to_pylist() == [None, None]
    assert validate_campus_directions(source)["matching_suppressed"] == 1
    assert "TRANSFERS_IN_OUT" in next(note["text"] for note in source.annotations if note["kind"] == "header")
    with pytest.raises(ValueError, match="preamble/header schema"):
        read(path, digest)

    old_header, old_digest = original(tmp_path, rows, filename="old-header.csv")
    with pytest.raises(ValueError, match="preamble/header schema"):
        read(old_header, old_digest, year=2026)

    legacy_path, legacy_digest = original(tmp_path, rows, legacy=True, filename="legacy.csv")
    legacy = read(legacy_path, legacy_digest, year=2026, layout="legacy_prepared")
    assert legacy.table["TRANSFERS_IN_OR_OUT"].to_pylist() == ["-999", "-999"]
    comparison = compare_campus_sources(legacy, source)
    assert comparison["raw_count_token_differences"] == 0
    assert comparison["only_old"] == comparison["only_new"] == 0


def test_district_published_total_is_retained_independently_of_masked_detail(tmp_path):
    common = ["2024-2025", "04", "REGION IV HOUSTON", "Example ISD  -  001902", "Traditional ISD/CSD", "100", "Transfers In From"]
    rows = [[*common, "100", "Partner ISD  -  002902", "-999"],
            [*common, "200", "", "."], [*common, "200", "Total Transfers In", "7370"],
            [*common, "300", "", "."]]
    path, digest = original(tmp_path, rows, grain="district")
    source = read(path, digest, grain="district")
    assert source.table["transfer_row_kind"].to_pylist() == ["detail", "separator", "publisher_total", "separator"]
    assert source.table["transfer_count_value"].to_pylist() == [None, None, 7370, None]
    assert source.table["transfer_count_status"].to_pylist() == ["suppressed", "missing", "observed", "missing"]
    assert source.table["transfer_partner_number"].to_pylist() == ["'002902", None, None, None]
    assert source.table["REPORT_DISTNAME_NUMBER"][0].as_py() == common[3]
    assert validate_district_rows(source.table)["reporting_districts"] == 1
    with pytest.raises(ValueError, match="Campus pair"):
        validate_campus_directions(source)
    duplicate, duplicate_sha = original(tmp_path, [*rows, rows[0]], grain="district", filename="duplicate.csv")
    with pytest.raises(TransferValidationError) as caught:
        validate_district_rows(read(duplicate, duplicate_sha, grain="district").table)
    assert caught.value.report["duplicate_detail_keys"] == 1


@pytest.mark.parametrize("grain,scope,region", [
    ("district", "Region", "04"), ("district", "District", None), ("campus", "Campus", None),
])
def test_physical_publisher_title_matches_explicit_scope_independently_of_table_grain(tmp_path, grain, scope, region):
    rows = pair_rows() if grain == "campus" else [["2024-2025", "04", "REGION IV HOUSTON",
        "Example ISD  -  001902", "Traditional ISD/CSD", "100", "Transfers In From", "100", "Partner ISD  -  002902", "-999"]]
    path, digest = original(tmp_path, rows, grain=grain, report_scope=scope)
    assert read(path, digest, grain=grain, region=region).grain == grain
    wrong_region = None if region is not None else "04"
    with pytest.raises(ValueError, match="publisher title|Regional report selection"):
        read(path, digest, grain=grain, region=wrong_region)
    if region is not None:
        with pytest.raises(ValueError, match="different selected region"):
            read(path, digest, grain=grain, region="13")


def test_campus_published_total_keeps_optional_partner_ids_missing(tmp_path):
    rows = pair_rows()
    rows.append(["2024-2025", "01", "001902", "Source ISD", "001902001", "Source",
                 "200", "Transfers Out To", "200", "", "", "", "Total Transfers Out", "7370"])
    path, digest = original(tmp_path, rows)
    source = read(path, digest)
    assert source.table["transfer_row_kind"].to_pylist() == ["detail", "detail", "publisher_total"]
    assert source.table["transfer_partner_number"][-1].as_py() is None
    assert source.table["transfer_partner_district_number"][-1].as_py() is None
    assert source.table["CAMPNAME_RES_OR_ATTEND"][-1].as_py() == "Total Transfers Out"
    assert source.table["transfer_count_value"][-1].as_py() == 7370
    assert validate_campus_directions(source)["matching_observed"] == 1


@pytest.mark.parametrize("change,message", [
    (lambda rows: rows[0].__setitem__(0, "2023-2024"), "YEAR differs"),
    (lambda rows: rows[0].__setitem__(4, "campus001902001"), "Invalid source identifier"),
    (lambda rows: rows[0].__setitem__(11, ""), "missing historical source identifier"),
    (lambda rows: rows[0].__setitem__(9, ""), "missing historical source identifier"),
    (lambda rows: rows[0].__setitem__(2, "003902"), "district identifiers disagree"),
    (lambda rows: rows[0].__setitem__(8, "400"), "Unexpected transfer row"),
    (lambda rows: rows[0].__setitem__(13, "-1"), "invalid count token"),
    (lambda rows: rows[0].__setitem__(13, "1.5"), "invalid count token"),
])
def test_unexpected_source_values_fail_visibly_without_zero_coercion(tmp_path, change, message):
    rows = pair_rows()
    change(rows)
    path, digest = original(tmp_path, rows)
    with pytest.raises(ValueError, match=message):
        read(path, digest)


def test_changed_header_or_unrecognized_footnote_fails_without_silent_drop(tmp_path):
    path, _ = original(tmp_path, pair_rows())
    raw = path.read_text().replace('"YEAR"', '"different_year"')
    # csv.writer does not quote these names by default.
    raw = raw.replace("YEAR,REPORT_REGION", "different_year,REPORT_REGION")
    path.write_text(raw)
    with pytest.raises(ValueError, match="preamble/header"):
        read(path, sha256(path.read_bytes()).hexdigest())
    path, _ = original(tmp_path, pair_rows())
    path.write_text(path.read_text().replace(NOTES[1], "Unknown publisher annotation"))
    with pytest.raises(ValueError, match="Unexpected transfer publisher footnote"):
        read(path, sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize("scenario", ["value", "status", "one_sided", "duplicate", "unknown"])
def test_independent_direction_gate_rejects_conflicts_before_edge_projection(tmp_path, scenario):
    rows = pair_rows()
    if scenario == "value":
        rows[1][-1] = "1"
    elif scenario == "status":
        rows[1][-1] = "-999"
    elif scenario == "one_sided":
        rows[1][2], rows[1][4] = "003902", "003902003"
    elif scenario == "duplicate":
        rows.append(list(rows[0]))
    else:
        rows[0][-1] = rows[1][-1] = "."
    path, digest = original(tmp_path, rows)
    with pytest.raises(TransferValidationError) as caught:
        validate_campus_directions(read(path, digest), max_examples=2)
    report = caught.value.report
    assert report["status"] == "HOLD"
    if scenario == "duplicate":
        assert report["duplicate_out_keys"] == 1 and "matching_observed" not in report
    elif scenario == "unknown":
        assert report["unknown_count_rows"] == 2 and "matching_observed" not in report
    elif scenario == "one_sided":
        assert report["one_sided_pairs"] == 2 and report["matching_observed"] == 0
        assert len(report["examples"]) == 2
        assert all(row["source_record_in"] is None or row["source_record_out"] is None for row in report["examples"])
    else:
        assert report["value_mismatches"] == 1 and report["matching_observed"] == 0


def test_direction_count_lexemes_are_distinct_from_interpreted_equality(tmp_path):
    path, digest = original(tmp_path, pair_rows("1", "1.0"))
    report = validate_campus_directions(read(path, digest))
    assert report["status"] == "PASS" and report["matching_observed"] == 1
    assert report["raw_count_token_differences"] == 1


def test_old_prepared_comparison_separates_padding_and_actual_changed_counts(tmp_path):
    old_rows = pair_rows()
    for row in old_rows:
        for index in [1, 2, 4, 9, 11]:
            row[index] = str(int(row[index]))
    old_path, old_sha = original(tmp_path, old_rows, legacy=True, filename="old.csv")
    new_path, new_sha = original(tmp_path, pair_rows(), filename="new.csv")
    old = read(old_path, old_sha, layout="legacy_prepared")
    new = read(new_path, new_sha)
    report = compare_campus_sources(old, new)
    assert report["only_old"] == report["only_new"] == report["count_value_changes"] == 0
    assert report["identifier_lexeme_changes"]["REPORT_CAMPUS"] == 2
    assert report["identifier_lexeme_changes"]["REPORT_REGION"] == 2
    assert not any(report["identifier_value_changes"].values())
    assert not any(report["other_raw_field_changes"].values())
    changed_path, changed_sha = original(tmp_path, pair_rows("1"), filename="changed.csv")
    changed = compare_campus_sources(old, read(changed_path, changed_sha), max_examples=1)
    assert changed["count_value_changes"] == 2 and len(changed["changed_data_examples"]) == 1


def retained_source(tmp_path, *, rows=None):
    path, digest = original(tmp_path, pair_rows() if rows is None else rows)
    store = ProvenanceStore(tmp_path / "archive")
    identifier = store.record_acquisition(path, ResolvedDataset("campus_transfer_reports", "data_sources", 2025, 2025, "",
        DatasetMetadata(version="2025", period=DataPeriod("school_year", 2024, 2025))),
        downloaded_at=datetime.now(timezone.utc), requested_url="https://example.test/original.csv",
        response_headers={"Content-Type": "text/csv"})
    config = tmp_path / "policy.yaml"
    config.write_text(json.dumps({"value_policies": {"campus_transfers": {key: value for key, value in POLICY.items() if key != "policy_id"}}}))
    return store, identifier, digest, config, sha256(config.read_bytes()).hexdigest()


@pytest.mark.parametrize("failed_position", [0, 1])
def test_structural_failure_retains_failed_input_error_and_parsed_prefix(tmp_path, failed_position):
    store, _, _, config, config_sha = retained_source(tmp_path)
    rows = [["2024-2025", "04", "REGION IV HOUSTON", "Example ISD  -  001902", "Traditional ISD/CSD",
             "100", "Transfers In From", "100", "Partner ISD  -  002902", "-999"]]
    inputs = []
    for number in range(2):
        path, digest = original(tmp_path, rows, grain="district", filename=f"district-{number}.csv",
                                report_scope="Campus" if number == failed_position else "District")
        identifier = store.record_acquisition(path, ResolvedDataset("district_transfer_reports", "data_sources", 2025, 2025, "",
            DatasetMetadata(version="2025", period=DataPeriod("school_year", 2024, 2025))),
            downloaded_at=datetime.now(timezone.utc), requested_url="https://example.test/district.csv",
            response_headers={"Content-Type": "text/csv"})
        inputs.append((identifier, digest))
    failed_record = store.read_record(inputs[failed_position][0])
    result = prepare_transfer_sources(store, grain="district", year=2025, output=tmp_path / "structural-hold",
        inputs=inputs, encoding="utf-8", policy_config=config, policy_config_sha256=config_sha, value_policy="campus_transfers")
    assert result["status"] == "HOLD" and result["failed_phase"] == "structural_read"
    assert [item["status"] for item in result["inputs"]] == ["parsed"] * failed_position + ["failed"] + ["pending"] * (1 - failed_position)
    assert result["failed_input"]["record_id"] == inputs[failed_position][0]
    assert result["error"]["type"] == "ValueError" and "CSV record 3, physical line 3" in result["error"]["message"]
    persisted = json.loads((tmp_path / "structural-hold/PREPARATION.json").read_text())
    assert persisted == result
    processing = store.read_record(result["processing_record_id"])
    assert set(processing["outputs"]) == {"validation_report"}
    assert not result["prepared_source_qualified"] and result["canonical_edges_qualified"] is None
    assert not processing["validation"]["prepared_source_qualified"]
    assert all(identifier in processing["inputs"] for identifier, _ in inputs)
    assert not (tmp_path / "structural-hold/prepared-source.parquet").exists()
    assert store.read_record(inputs[failed_position][0]) == failed_record


def test_preparation_closes_real_raw_and_policy_edges_and_does_not_activate_catalog(tmp_path):
    store, identifier, digest, config, config_sha = retained_source(tmp_path)
    original_record = store.read_record(identifier)
    result = prepare_transfer_sources(store, grain="campus", year=2025, output=tmp_path / "prepared",
        inputs=[(identifier, digest)], encoding="utf-8", policy_config=config,
        policy_config_sha256=config_sha, value_policy="campus_transfers")
    assert result["status"] == "prepared" and result["raw_string_readback_equal"]
    assert result["prepared_source_qualified"] and result["canonical_edges_qualified"]
    assert not result["activation"] and not result["field_correctness_qualified"]
    closure = store.verify_record_closure([result["processing_record_id"]])
    processing = closure[result["processing_record_id"]]
    assert identifier in processing["inputs"] and closure[identifier] == original_record
    assert set(processing["outputs"]) == {"prepared_source", "annotations", "validation_report"}
    assert processing["code_version"] == processor_identity()
    assert result["inputs"][0]["encoding"]["selection"] == "explicit_argument"
    policies = [record for record in closure.values() if record.get("dataset") == "reference:transfer_policy_config"]
    assert len(policies) == 1 and policies[0]["method"] == "local_observation" and policies[0]["downloaded_at"] is None


def test_preparation_does_not_guess_a_missing_publisher_charset(tmp_path):
    store, identifier, digest, config, config_sha = retained_source(tmp_path)
    with pytest.raises(ValueError, match="explicit encoding"):
        prepare_transfer_sources(store, grain="campus", year=2025, output=tmp_path / "prepared",
            inputs=[(identifier, digest)], policy_config=config, policy_config_sha256=config_sha, value_policy="campus_transfers")
    assert not (tmp_path / "prepared").exists()


def test_explicit_legacy_comparison_accepts_unknown_requested_year_without_rewriting_event(tmp_path):
    store, identifier, digest, config, config_sha = retained_source(tmp_path)
    legacy, legacy_sha = original(tmp_path, pair_rows(), legacy=True, filename="legacy.csv")
    legacy_id = store.record_acquisition(legacy, ResolvedDataset("campus_transfer_reports", "data_sources", None, 2025, "",
        DatasetMetadata(version="2025", period=DataPeriod("school_year", 2024, 2025), notes="Original HTTP/date unknown")))
    before = store.read_record(legacy_id)
    result = prepare_transfer_sources(store, grain="campus", year=2025, output=tmp_path / "compared",
        inputs=[(identifier, digest)], encoding="utf-8", policy_config=config,
        policy_config_sha256=config_sha, value_policy="campus_transfers",
        compare_input=(legacy_id, legacy_sha), compare_encoding="utf-8")
    assert result["status"] == "prepared" and result["old_prepared_comparison"]["count_value_changes"] == 0
    assert store.read_record(legacy_id) == before
    assert before["requested_year"] is None and before["downloaded_at"] is None and before["method"] == "local_observation"
    with pytest.raises(ValueError, match="dataset/year/acquisition"):
        prepare_transfer_sources(store, grain="campus", year=2025, output=tmp_path / "not-a-new-original",
            inputs=[(legacy_id, legacy_sha)], encoding="utf-8", policy_config=config,
            policy_config_sha256=config_sha, value_policy="campus_transfers")


@pytest.mark.parametrize("out,incoming", [("0", "1"), ("-999", "10"), ("10", "-999"), ("-999", "11"), ("11", "-999")])
def test_edge_hold_retains_independent_reported_values_without_coalescing(tmp_path, out, incoming):
    import pyarrow.parquet as pq

    store, identifier, digest, config, config_sha = retained_source(tmp_path, rows=pair_rows(out, incoming))
    original_record = store.read_record(identifier)
    result = prepare_transfer_sources(store, grain="campus", year=2025, output=tmp_path / "held",
        inputs=[(identifier, digest)], encoding="utf-8", policy_config=config,
        policy_config_sha256=config_sha, value_policy="campus_transfers")
    processing = store.read_record(result["processing_record_id"])
    assert result["status"] == "HOLD" and processing["validation"]["status"] == "HOLD"
    assert result["prepared_source_qualified"] and not result["canonical_edges_qualified"]
    assert processing["validation"]["prepared_source_qualified"]
    assert not processing["validation"]["canonical_edges_qualified"]
    assert result["failed_phase"] == processing["validation"]["failed_phase"] == "source_validation"
    assert "one canonical edge" in result["error"]["message"]
    assert result["source_validation"]["status"] == "HOLD"
    assert result["raw_string_readback_equal"] and not result["field_correctness_qualified"] and not result["activation"]
    assert set(processing["outputs"]) == {"prepared_source", "annotations", "validation_report"}
    table = pq.read_table(tmp_path / "held/prepared-source.parquet")
    assert table["TRANSFERS_IN_OR_OUT"].to_pylist() == [out, incoming]
    assert table["transfer_count_value"].to_pylist() == [None if value == "-999" else int(value) for value in (out, incoming)]
    assert table["transfer_count_status"].to_pylist() == ["suppressed" if value == "-999" else "observed" for value in (out, incoming)]
    assert table["transfer_direction"].to_pylist() == ["out", "in"]
    assert table["transfer_source_record"].to_pylist() == [8, 9]
    assert table["transfer_first_line"].to_pylist() == [8, 10]
    assert table["transfer_last_line"].to_pylist() == [9, 11]
    assert table["transfer_source_input"].to_pylist() == [0, 0]
    closure = store.verify_record_closure([result["processing_record_id"]])
    assert closure[identifier] == original_record
    assert json.loads((tmp_path / "held/PREPARATION.json").read_text()) == result
    with pytest.raises(TransferValidationError):
        validate_campus_directions(read(store.verify_artifact(digest), digest))


@pytest.mark.parametrize("new_hold,legacy_hold", [(False, True), (True, False), (True, True)])
def test_optional_comparison_cannot_erase_observations_or_override_edge_hold(tmp_path, new_hold, legacy_hold):
    import pyarrow.parquet as pq

    rows = pair_rows("-999", "10") if new_hold else pair_rows()
    store, identifier, digest, config, config_sha = retained_source(tmp_path, rows=rows)
    old_path, old_sha = original(tmp_path, pair_rows("0", "1") if legacy_hold else pair_rows(), legacy=True, filename="old.csv")
    legacy_id = store.record_acquisition(old_path, ResolvedDataset("campus_transfer_reports", "data_sources", None, 2025, "",
        DatasetMetadata(version="2025", period=DataPeriod("school_year", 2024, 2025), notes="Original HTTP/date unknown")))
    result = prepare_transfer_sources(store, grain="campus", year=2025, output=tmp_path / "held",
        inputs=[(identifier, digest)], encoding="utf-8", policy_config=config,
        policy_config_sha256=config_sha, value_policy="campus_transfers",
        compare_input=(legacy_id, old_sha), compare_encoding="utf-8")
    processing = store.read_record(result["processing_record_id"])
    assert result["status"] == processing["validation"]["status"] == "HOLD"
    assert result["prepared_source_qualified"] and result["canonical_edges_qualified"] is (not new_hold)
    assert result["source_validation"]["status"] == ("HOLD" if new_hold else "PASS")
    assert result["failed_phase"] == ("source_validation" if new_hold else "legacy_comparison")
    assert result["old_source"]["status"] == ("failed" if legacy_hold else "parsed")
    if legacy_hold:
        assert result["old_source"]["error"]["type"] == "TransferValidationError"
        assert result["failed_validation"]["status"] == "HOLD"
    else:
        assert result["old_prepared_comparison"]["status"] == "compared"
    table = pq.read_table(tmp_path / "held/prepared-source.parquet")
    assert table["TRANSFERS_IN_OR_OUT"].to_pylist() == [row[-1] for row in rows]
    assert set(processing["outputs"]) == {"prepared_source", "annotations", "validation_report"}
    assert legacy_id in store.verify_record_closure([result["processing_record_id"]])


@pytest.mark.parametrize("edge_hold", [False, True])
def test_failed_native_readback_remains_diagnostic_only(tmp_path, monkeypatch, edge_hold):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = pair_rows("-999", "10") if edge_hold else pair_rows()
    store, identifier, digest, config, config_sha = retained_source(tmp_path, rows=rows)
    original_write = pq.write_table

    def corrupt_count(table, where, **kwargs):
        name = "transfer_count_value"
        changed = table.set_column(table.schema.get_field_index(name), name, pa.array([99, 99], type=pa.int64()))
        return original_write(changed, where, **kwargs)

    monkeypatch.setattr(pq, "write_table", corrupt_count)
    result = prepare_transfer_sources(store, grain="campus", year=2025, output=tmp_path / "held",
        inputs=[(identifier, digest)], encoding="utf-8", policy_config=config,
        policy_config_sha256=config_sha, value_policy="campus_transfers")
    processing = store.read_record(result["processing_record_id"])
    assert result["status"] == processing["validation"]["status"] == "HOLD"
    assert not result["prepared_source_qualified"] and not processing["validation"]["prepared_source_qualified"]
    assert "readback differs" in result["readback_error"]["message"]
    assert result["failed_phase"] == ("source_validation" if edge_hold else "native_readback")
    assert set(processing["outputs"]) == {"validation_report"}
    assert not (tmp_path / "held/prepared-source.parquet").exists()


def test_prepare_cli_preserves_edge_hold_exit_status_with_retained_observations(tmp_path, capsys):
    store, identifier, digest, config, config_sha = retained_source(tmp_path, rows=pair_rows("-999", "10"))
    assert main(["prepare", "--store", str(store.root), "--grain", "campus", "--year", "2025",
        "--input", identifier, digest, "--encoding", "utf-8", "--policy-config", str(config),
        "--policy-config-sha256", config_sha, "--policy", "campus_transfers", "--output", str(tmp_path / "cli")]) == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "HOLD" and summary["rows"] == 2
    processing = store.read_record(summary["processing_record_id"])
    assert processing["validation"]["prepared_source_qualified"] and not processing["validation"]["canonical_edges_qualified"]


def test_prepare_cli_uses_canonical_core_without_network_or_engine(tmp_path, capsys):
    store, identifier, digest, config, config_sha = retained_source(tmp_path)
    assert main(["prepare", "--store", str(store.root), "--grain", "campus", "--year", "2025",
        "--input", identifier, digest, "--encoding", "utf-8", "--policy-config", str(config),
        "--policy-config-sha256", config_sha, "--policy", "campus_transfers", "--output", str(tmp_path / "cli")]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "prepared"


def test_shared_reader_changes_are_visible_to_existing_processor_identity(monkeypatch):
    reader = Path(__file__).resolve().parents[1] / "teadata/source_csv.py"
    original_read = Path.read_bytes
    before = processor_identity()

    def modified(path):
        raw = original_read(path)
        return raw + b"\n# reader semantics changed\n" if path == reader else raw

    monkeypatch.setattr(Path, "read_bytes", modified)
    assert processor_identity() != before
