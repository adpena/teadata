import csv
from decimal import Decimal
from io import StringIO
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pyarrow as pa
import pytest

from teadata.dataset_metadata import EditionKey
from teadata.manifest import ContractError
from teadata.native_courses import (
    HOLD_CAMPUS_GEOGRAPHY, PASS_COURSE_CONTROLS, _campus_blocks, _compare_batch, _district_campus_witness, _replay,
)
from teadata.peims_course_sources import CSV_HEADERS, DATASET, course_csv_preamble
from teadata.scripts.verify_peims_courses import _HTML_HEADER
from teadata.verification import main


def _native_batch():
    fields = {
        "source_scope": ["STATE"],
        "aggregation_level": ["COUNTYREGIONDISTRICT"],
        "school_year": ["2025-2026"],
        "district_number": ["057829"],
        "district_name": ["A+ ACADEMY"],
        "charter_status": ["OPEN ENROLLMENT CHARTER"],
        "region_number": ["10"],
        "subject": ["ART"],
        "course_code": ["02510001"],
        "course_title": ["ART, KINDERGARTEN"],
        "grade_level": ["KINDERGARTEN"],
        "peims_source_record": [10],
        "teacher_head_count": [1],
        "peims_raw_teacher_head_count": ["1"],
        "peims_status_teacher_head_count": ["observed"],
        "teacher_fte_count": pa.array([Decimal("0.17")], type=pa.decimal128(16, 2)),
        "peims_raw_teacher_fte_count": ["0.17"],
        "peims_status_teacher_fte_count": ["observed"],
        "contract_teacher_head_count": [0],
        "peims_raw_contract_teacher_head_count": ["0"],
        "peims_status_contract_teacher_head_count": ["observed"],
        "contract_teacher_fte_count": pa.array([Decimal("0")], type=pa.decimal128(16, 2)),
        "peims_raw_contract_teacher_fte_count": ["0"],
        "peims_status_contract_teacher_fte_count": ["observed"],
        "course_enrollment_count": pa.array([None], type=pa.int64()),
        "peims_raw_course_enrollment_count": ["-999"],
        "peims_status_course_enrollment_count": ["suppressed"],
    }
    return pa.RecordBatch.from_pydict(fields)


@pytest.mark.parametrize(("column", "value", "message"), [
    ("district_number", "57829", "Course raw value differs"),
    ("peims_source_record", 11, "Course physical source locator differs"),
    ("peims_raw_teacher_fte_count", "0.18", "Course raw measure differs"),
    ("teacher_fte_count", Decimal("0.18"), "Course normalized value differs"),
    ("course_enrollment_count", 0, "Course normalized value differs"),
    ("peims_status_course_enrollment_count", "missing", "Course measure status differs"),
])
def test_course_replay_rejects_parquet_raw_type_and_status_drift(column, value, message):
    raw = [["STATE", "COUNTYREGIONDISTRICT", "2025-2026", "057829", "A+ ACADEMY", "OPEN ENROLLMENT CHARTER", "10",
            "ART", "02510001", "ART, KINDERGARTEN", "KINDERGARTEN", "1", "0.17", "0", "0", "-999"]]
    leaf = _native_batch()
    assert _compare_batch(raw, leaf, "district", 10) == 1
    i = leaf.schema.get_field_index(column)
    damaged = leaf.set_column(i, column, pa.array([value], type=leaf.schema.field(column).type))
    with pytest.raises(ContractError, match=message):
        _compare_batch(raw, damaged, "district", 10)


def test_course_replay_rejects_candidate_owned_replacement_receipt(monkeypatch):
    release, store, witnesses, config = Mock(), Mock(), Mock(), Mock()
    key = EditionKey(DATASET, "2026", "a" * 64, "district", "district", "district and course")
    ref = SimpleNamespace(contract={"inputs": ["a" * 64], "rows": 1})
    release.catalog.variables.return_value = [{"field": "course_enrollment_count"}]
    release.catalog.resolve_artifacts.return_value = [{"artifact": ref}]
    release.catalog.acquisitions.return_value = [{"record_id": "b" * 64}]
    monkeypatch.setattr("teadata.native_courses._admit", lambda *args: SimpleNamespace(receipt="c" * 64))
    with pytest.raises(ContractError, match="packaged source edition and scope"):
        _replay(release, store, witnesses, config, key, {})
    store.verify_record_closure.assert_not_called()


def test_campus_blocks_match_values_without_reassigning_conflicting_headings(monkeypatch):
    monkeypatch.setattr("teadata.native_courses._check_footer", lambda footer, year, fmt: None)
    common = ["019901102", "DEKALB EL", "TRADITIONAL ISD/CSD", "08", "OTHER", "SE000001", "SPECIAL EDUCATION, GENERIC",
              "ALL GRADE LEVELS", "1"]
    rows = [
        ["STATE", "COUNTYREGIONDISTRICTCAMPUS", "2025-2026", "019901", "DEKALB ISD", *common, "0.08", "0", "0", "-999"],
        ["STATE", "COUNTYREGIONDISTRICTCAMPUS", "2025-2026", "019905", "NEW BOSTON ISD", *common, "0.26", "0", "0", "0"],
    ]
    label = "DEKALB EL (019901102)"
    heading = "District: NEW BOSTON ISD (019905) - Region: 08"
    prefix = [[heading], ["Charter Status: TRADITIONAL ISD/CSD"], ["Campus Name/Campus", *_HTML_HEADER]]
    shown = [*prefix, [label, *rows[0][9:-1], "N/A"], *prefix, [label, *rows[1][9:]]]
    blocks = _campus_blocks(shown, 2026, label, rows)
    assert [(block["matched_source_district_number"], block["heading_matches_bulk"]) for block in blocks] == [
        ("019901", False), ("019905", True),
    ]
    assert all(block["displayed_district_heading"] == heading for block in blocks)
    shown[5] = ["Charter Status: OPEN ENROLLMENT CHARTER"]
    second = _campus_blocks(shown, 2026, label, rows)[1]
    assert second["heading_matches_bulk"] and not second["charter_status_matches_bulk"]
    shown[3][-1] = "0"
    with pytest.raises(ContractError, match="unique bulk district block"):
        _campus_blocks(shown, 2026, label, rows)


@pytest.mark.parametrize("drift", [None, "district", "scope", "mask", "omission", "duplicate"])
def test_district_campus_csv_matches_complete_literal_slice(monkeypatch, drift):
    raw = ["STATE", "COUNTYREGIONDISTRICTCAMPUS", "2025-2026", "019905", "NEW BOSTON ISD", "019901102",
           "DEKALB EL", "TRADITIONAL ISD/CSD", "08", "OTHER", "SE000001", "SPECIAL EDUCATION, GENERIC",
           "ALL GRADE LEVELS", "1", "0.26", "0", "0", "-999"]
    extra = [*raw[:10], "SE000005", *raw[11:]]
    raw_rows = [raw, extra]
    prefix = course_csv_preamble(2026, "campus")
    prefix[6] = ["019905 NEW BOSTON ISD BY CAMPUS"]
    prefix[8] = [*CSV_HEADERS["campus"][:3], *CSV_HEADERS["campus"][5:]]
    selected = [["DISTRICT", *row[1:3], *row[5:]] for row in reversed(raw_rows)]
    if drift == "district":
        prefix[6] = ["019901 DEKALB ISD BY CAMPUS"]
    if drift == "scope":
        selected[0][0] = "STATE"
    if drift == "mask":
        selected[0][-1] = "0"
    if drift == "omission":
        selected.pop()
    if drift == "duplicate":
        selected[0] = selected[1]
    text = StringIO(newline="")
    csv.writer(text).writerows([*prefix, *selected])
    opened = Mock(return_value=(text.getvalue(), {"receipt_id": "witness"}))
    monkeypatch.setattr("teadata.native_courses._open_district_witness", opened)
    monkeypatch.setattr("teadata.native_courses._check_footer", lambda *args: None)
    store = Mock()
    if drift is None:
        assert _district_campus_witness(store, "witness", 2026, "019905", raw_rows) == {
            "receipt_id": "witness", "source_district_number": "019905", "rows": 2, "compared_cells": 26,
        }
        opened.assert_called_once_with(store, "witness", 2026, "019905", "NEW BOSTON ISD", campus_csv=True)
    else:
        with pytest.raises(ContractError, match="Course"):
            _district_campus_witness(store, "witness", 2026, "019905", raw_rows)


@pytest.mark.parametrize("status", [PASS_COURSE_CONTROLS, HOLD_CAMPUS_GEOGRAPHY])
def test_course_controls_never_exit_success_for_a_campus_hold(monkeypatch, capsys, status):
    monkeypatch.setattr(sys, "argv", [
        "check", "--kind", "library", "--course-controls", "--native-release", "release.json",
        "--native-pin", "f" * 64, "--source-store", "provenance",
    ])
    monkeypatch.setattr("teadata.verification._validate_course_controls", lambda *args: {"status": status})
    if status == HOLD_CAMPUS_GEOGRAPHY:
        with pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == 2
    else:
        main()
    assert status in capsys.readouterr().out
