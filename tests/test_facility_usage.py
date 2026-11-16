"""Small source-boundary regressions; synthetic rows are not publisher proof."""
from decimal import Decimal
from xml.etree import ElementTree as ET

import pytest

from teadata.facility_usage import FIELDS, _cell, _value, project_rows, submission_table
from teadata.manifest import ContractError
from teadata.teadata_config import load_config


def source_row(ordinal=2, status="SUBMITTED"):
    cells = {c: (None, "absent") for c, *_ in FIELDS}
    cells.update(A=("Example ISD", "inlineStr"), B=("001902", "inlineStr"),
                 AE=("12", "n"), AF=(status, "inlineStr"))
    return ordinal, cells, True


def domains():
    return {key: set() for key in "ABCDEFG"}


def test_strings_are_raw_only_and_numeric_states_preserve_blank_zero_and_precision():
    ordinal, cells, contact = source_row()
    cells.update(F=("  NA ", "inlineStr"), G=("10058", "n"), L=("0", "n"),
                 R=("0.008500000000000001", "n"), AB=("-346", "n"), AD=(None, "n_empty"))
    table, support = project_rows([(ordinal, cells, contact)], domains())
    row = table.to_pylist()[0]
    assert row["property_id"] == row["raw_property_id"] == "  NA "
    assert row["street_address"] == "10058"
    assert support["fields"]["property_id"] == {"raw": "raw_property_id"}
    assert "status_property_id" not in table.column_names
    assert row["classroom_count"] == 0 and row["status_classroom_count"] == "observed"
    assert row["portable_count"] is None and row["status_portable_count"] == "missing"
    assert row["acreage"] == Decimal("0.008500000000000001")
    assert row["adjusted_instructional_capacity"] == Decimal(-346)
    assert row["reported_campus_label"] is None
    assert row["district_number"] == "'001902" and row["raw_district_number"] == "001902"


@pytest.mark.parametrize("raw,physical", [("0.0085000000000000001", "decimal128(24,18)"),
                                         ("10000000", "decimal128(19,12)"),
                                         ("1.5", "int32"), ("NaN", "decimal128(18,13)"), ("40000", "int16")])
def test_inexact_or_out_of_contract_numeric_tokens_fail_without_values_in_errors(raw, physical):
    with pytest.raises(ContractError, match="exact contract") as error:
        _value(raw, "n", physical, "Q", 2)
    assert raw not in str(error.value)


def test_duplicate_facility_records_keep_distinct_original_row_keys():
    first = source_row(2)
    second = source_row(3)
    table, _ = project_rows([first, second], domains())
    assert table.num_rows == 2 and table["publisher_worksheet_row"].to_pylist() == [2, 3]
    groups, support = submission_table(table)
    assert groups.num_rows == 1 and groups["source_group_row_count"].to_pylist() == [2]
    assert support["fields"]["submission_status"] == {"raw": "raw_submission_status"}


@pytest.mark.parametrize("payload", ["contact", "facility"])
def test_status_only_rows_cannot_hide_payload(payload):
    ordinal, cells, _ = source_row(status="INCOMPLETE")
    if payload == "facility":
        cells["L"] = ("0", "n")
    with pytest.raises(ContractError, match="status-only"):
        project_rows([(ordinal, cells, payload != "contact")], domains())


def test_status_only_absence_and_district_status_conflict():
    table, _ = project_rows([source_row(status="NOT SUBMITTED")], domains())
    assert table["record_kind"].to_pylist() == ["status_only"]
    assert table["maximum_instructional_capacity"].to_pylist() == [None]
    table, _ = project_rows([source_row(2), source_row(3, "INCOMPLETE")], domains())
    with pytest.raises(ContractError, match="inconsistent"):
        submission_table(table)


def test_unlisted_category_is_observed_text_and_flagged_separately():
    ordinal, cells, contact = source_row()
    cells["U"] = ("FREE TEXT", "inlineStr")
    table, _ = project_rows([(ordinal, cells, contact)], domains())
    assert table["shared_building_type_1"].to_pylist() == ["FREE TEXT"]
    assert table["quality_conformance_json"].to_pylist() == ['{"outside_validation_columns":["U"]}']


def test_public_free_text_email_fails_without_disclosing_it():
    with pytest.raises(ContractError, match="privacy review") as error:
        _value("test@example.invalid", "inlineStr", "string", "G", 2)
    assert "example.invalid" not in str(error.value)


def test_empty_xml_cells_and_formula_are_distinct():
    assert _cell(None) == (None, "absent")
    assert _cell(ET.fromstring('<c xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" t="n"/>')) == (None, "n_empty")
    assert _cell(ET.fromstring('<c xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" t="inlineStr"><is><t/></is></c>')) == ("", "inlineStr_empty")
    with pytest.raises(ContractError, match="formula"):
        _cell(ET.fromstring('<c xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><f>1+1</f></c>'))


def test_source_only_contract_retains_public_headers_and_exact_area_units():
    cfg = load_config()
    for dataset in ("facility_usage", "facility_usage_submission"):
        assert cfg.is_source_only(dataset, 2026)
        source = cfg.resolve_dataset(dataset, 2026, exact=True, reader_profile="source")
        fields = cfg.header_records(dataset, edition="2026", reader_profile="source")
        assert len(fields) == (31 if dataset == "facility_usage" else 4)
        assert all(row["code"] != "Contact Email Address" for row in fields)
        assert source.metadata.period.end_year == 2026
        with pytest.raises(ContractError, match="Reader profile"):
            cfg.load_df(dataset, 2026)
    metadata = cfg.resolve_dataset("facility_usage", 2026).metadata
    assert metadata.metrics["square_footage"].unit == "square_feet"
    assert metadata.metrics["acreage"].unit == "acres"
