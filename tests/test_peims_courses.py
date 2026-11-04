from copy import deepcopy
import csv
from decimal import Decimal
from hashlib import sha256
from io import StringIO

import pyarrow as pa
import pytest

from teadata import peims_courses as courses
from teadata.manifest import ContractError
from teadata.peims_course_sources import DATASET, course_csv_preamble
from teadata.teadata_config import load_config


@pytest.mark.parametrize(("year", "state_rows", "district_rows"), [
    (2023, 1386, None), (2024, 1397, 201999), (2025, 1422, 204303), (2026, 1453, 207751),
])
def test_packaged_course_sources_are_exact_and_opt_in(year, state_rows, district_rows):
    config = load_config()
    assert config.is_source_only(DATASET, year)
    assert config.column_schema(DATASET, year) is None
    state = courses._table(config, year, "state")
    assert state["layout"]["last_data_record"] - 9 == state_rows
    district = courses._table(config, year, "district")
    if district_rows is None:
        assert district == {}
    else:
        assert district["layout"]["last_data_record"] - 9 == district_rows
        assert district["input_sha256"] != state["input_sha256"]


def _raw_source(tmp_path, scope, records):
    config = load_config()
    table = deepcopy(courses._table(config, 2026, scope))
    original = config.resolve_dataset(DATASET, 2026, reader_profile="source", exact=True)
    stream = StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerows(course_csv_preamble(2026, scope))
    start = stream.getvalue().encode("cp1252")
    writer.writerows(records)
    raw = stream.getvalue().encode("cp1252")
    footer = b"\r\nReviewed footer\r\n"
    path = tmp_path / "source.csv"
    path.write_bytes(raw + footer)
    first = 10 + len(records)
    table["layout"]["last_data_record"] = first - 1
    table["layout"]["annotations"] = [
        {"first_record": 1, "last_record": 8, "sha256": sha256(b"".join(start.splitlines(keepends=True)[:8])).hexdigest()},
        {"first_record": first, "last_record": first + 1, "sha256": sha256(footer).hexdigest()},
    ]
    return path, courses._CourseSource(2026, scope, original, "", "", table, ())


@pytest.mark.parametrize("scope", ["state", "district"])
def test_native_course_batches_preserve_codes_precision_and_masks(tmp_path, scope):
    state = scope == "state"
    geo = ["STATE", "STATE", "2025-2026"] if state else [
        "STATE", "COUNTYREGIONDISTRICT", "2025-2026", "001902", "CAYUGA ISD", "TRADITIONAL ISD/CSD", "07",
    ]
    path, selection = _raw_source(tmp_path, scope, [
        [*geo, "ART", "00123456", "ART, GRADE 2", "GRADE 2", "1", "0.30", "0", "0", "-999"],
        [*geo, "ART", "00123457", "ART, GRADE 3", "GRADE 3", "0", "0", "1", "0.20", "0"],
    ])
    table = pa.Table.from_batches(list(courses._batches(path, selection)))
    assert table.schema.field("teacher_fte_count").type == pa.decimal128(16, 2)
    assert table["teacher_fte_count"].to_pylist() == [Decimal("0.30"), Decimal("0.00")]
    assert table["course_code"].to_pylist() == ["00123456", "00123457"]
    if not state:
        assert table["district_number"].to_pylist() == ["001902", "001902"]
        assert table["region_number"].to_pylist() == ["07", "07"]
    assert table["peims_source_record"].to_pylist() == [10, 11]
    assert table["peims_raw_teacher_fte_count"].to_pylist() == ["0.30", "0"]
    assert table["course_enrollment_count"].to_pylist() == [None, 0]
    assert table["peims_raw_course_enrollment_count"].to_pylist() == ["-999", "0"]
    assert table["peims_status_course_enrollment_count"].to_pylist() == ["suppressed", "observed"]


@pytest.mark.parametrize(("damage", "match"), [
    ("duplicate", "Duplicate PEIMS course code"), ("identity", "Conflicting PEIMS course district identity"),
    ("district", "Malformed district ID"), ("scope", "selected scope or year"), ("year", "selected scope or year"),
    ("precision", "numeric token"), ("headcount-mask", "numeric token"), ("footer", "raw preamble or footer"),
    ("reopened", "district group reappeared"),
])
def test_native_course_stream_rejects_mismatched_scope_identity_and_values(tmp_path, monkeypatch, damage, match):
    monkeypatch.setattr(courses, "_CSV_BLOCK_BYTES", 192)
    rows = [
        ["STATE", "COUNTYREGIONDISTRICT", "2025-2026", "001902", "CAYUGA ISD", "TRADITIONAL ISD/CSD", "07",
         "ART", "00123456", "ART, GRADE 2", "GRADE 2", "1", "0.30", "0", "0", "-999"],
        ["STATE", "COUNTYREGIONDISTRICT", "2025-2026", "001902", "CAYUGA ISD", "TRADITIONAL ISD/CSD", "07",
         "ART", "00123457", "ART, GRADE 3", "GRADE 3", "1", "0.30", "0", "0", "0"],
    ]
    replacements = {"duplicate": (8, "00123456"), "identity": (4, "OTHER ISD"), "district": (3, "1902"),
                    "scope": (1, "STATE"), "year": (2, "2024-2025"), "precision": (12, "0.301"),
                    "headcount-mask": (11, "-999")}
    if damage in replacements:
        index, value = replacements[damage]
        rows[1][index] = value
    if damage == "reopened":
        rows.insert(1, [*rows[0][:3], "001903", *rows[0][4:]])
    path, selection = _raw_source(tmp_path, "district", rows)
    if damage == "footer":
        path.write_bytes(path.read_bytes().replace(b"Reviewed footer", b"Changed footer"))
    with pytest.raises(ContractError, match=match):
        list(courses._batches(path, selection))
