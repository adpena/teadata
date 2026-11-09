"""Contract checks for College Credit framing and cohort-specific fields."""

import csv
from decimal import Decimal
from hashlib import sha256
from io import StringIO

import pyarrow as pa
import pytest

from teadata import peims_college_credit as credit
from teadata.manifest import ContractError
from teadata.teadata_config import load_config


def test_college_credit_catalog_retains_literal_header_and_cohort_breaks():
    config = load_config()
    expected_groups = ("cte_excluding_participants", "cte_all_program_groups", "cte_all_program_groups",
                       "cte_all_program_groups", "cte_excluding_explorers")
    for year, group in zip(credit.YEARS, expected_groups, strict=True):
        assert config.is_source_only(credit.DATASET, year)
        assert config.column_schema(credit.DATASET, year) is None
        for scope in credit.SCOPES:
            table = credit._table(config, year, scope)
            assert credit.matches_college_credit_header(table, year, scope)
            fields = {row["field"]: row["code"] for row in table["fields"]}
            assert fields[f"{group}_students_earning_college_credit_count"] == "Total CTE Students Earning College Credit"
            assert fields["publisher_separator"] == ""


@pytest.mark.parametrize(("change", "error"), [
    (None, None), ("width", "CSV width differs"), ("separator", "separator is populated"),
    ("wrong_mask", "Unreviewed College Credit token"), ("duplicate", "Duplicate College Credit course"),
    ("no_total", "complete publisher totals differ"),
])
def test_college_credit_projection_retains_ids_source_records_and_masks(tmp_path, change, error):
    config = load_config()
    table = credit._table(config, 2026, "district")
    raw = StringIO(newline="")
    writer = csv.writer(raw)
    writer.writerows([["TEXAS EDUCATION AGENCY"], ["College Credit Reports - Statewide District Totals"],
                      ["PEIMS data for 2025-2026"], [], [], [], [], []])
    preamble = raw.getvalue().encode("cp1252")
    header = [row["code"] for row in table["fields"]]
    writer.writerow(header)
    geo = ["07", "KILGORE", "001902", "CAYUGA ISD", "TRADITIONAL ISD/CSD"]
    course = [*geo, "00123456", "MATH, DUAL CREDIT", "-999", "-999", "-999.0", "", "0", "0", "0"]
    total = [*geo, "", "Total", "2", "7", "3.5", "", "4", "13", "3.3"]
    if change == "width":
        course.pop()
    if change == "separator":
        course[10] = "unexpected"
    if change == "wrong_mask":
        course[9] = "-999"
    if change == "duplicate":
        total[:9] = course[:9]
    if change == "no_total":
        total[5:7] = ["00123457", "OTHER COURSE"]
    writer.writerows([course, [""] * len(header), total])
    footer = b"Reviewed note\r\n"
    path = tmp_path / "retained.csv"
    path.write_bytes(raw.getvalue().encode("cp1252") + footer)
    table["layout"].update(last_data_record=12, annotations=[
        {"first_record": 1, "last_record": 8, "kind": "preamble", "sha256": sha256(preamble).hexdigest()},
        {"first_record": 13, "last_record": 13, "kind": "footer", "sha256": sha256(footer).hexdigest()},
    ])
    selection = credit._Source(2026, "district", config.resolve_dataset(credit.DATASET, 2026, reader_profile="source",
        exact=True), "", "", table, (), 2)
    if error:
        with pytest.raises(ContractError, match=error):
            list(credit._batches(path, selection))
        return
    actual = pa.Table.from_batches(list(credit._batches(path, selection)))
    average = "cte_excluding_explorers_average_credit_hours_per_student"
    assert actual["district_number"].to_pylist() == ["001902", "001902"]
    assert actual[credit.SOURCE_RECORD].to_pylist() == [10, 12]
    assert actual[credit.SERVICE_KEY].to_pylist() == ["00123456", "Total"]
    assert actual["students_earning_college_credit_count"].to_pylist() == [0, 4]
    assert actual[average].to_pylist() == [None, Decimal("3.5")]
    assert actual[f"peims_raw_{average}"].to_pylist() == ["-999.0", "3.5"]
    assert actual[f"peims_status_{average}"].to_pylist() == ["suppressed", "observed"]
