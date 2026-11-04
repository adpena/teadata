from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

import pyarrow as pa
import pytest

from teadata.dataset_metadata import EditionKey
from teadata.manifest import ContractError
from teadata.native_courses import _compare_batch, _replay
from teadata.peims_course_sources import DATASET


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
