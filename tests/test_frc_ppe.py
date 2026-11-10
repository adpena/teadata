from html import escape
from types import SimpleNamespace
from uuid import uuid4

import pytest

from teadata.column_schema import crosswalk_dataset, parse_column_schema, reader_options
from teadata.dataset_metadata import DataPeriod, EditionKey, NativeSourceRef
from teadata.entities import District
from teadata.frc_ppe import DATASET, PERIOD, SOURCE_ROW, UNIT, _BLANK_NOTE, _FOOTER, _FUND_HEAD, _PPE_HEAD, _TITLE, _bulk_rows, _projection, _selected_row
from teadata.manifest import ContractError
from teadata.query import Query
from teadata.release import DataRelease
from teadata.statewide import _frc_value
from teadata.teadata_config import load_config
from teadata.values import ValueStatus, measure


def _html(rows):
    return "<table>" + "".join("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>" for row in rows) + "</table>"


def test_packaged_frc_source_is_exact_and_opt_in():
    config = load_config()
    assert config.is_source_only(DATASET, 2025)
    assert config.column_schema(DATASET, 2025) is None
    source = config.resolve_dataset(DATASET, 2025, reader_profile="source", exact=True)
    assert source.metadata.period == DataPeriod("fiscal_year", 2024, 2025)
    assert source.metadata.metrics["reported_enrollment"].unit == "count"
    assert source.metadata.metrics["all_sources_per_pupil"].unit == UNIT
    assert source.metadata.value_policy == "frc_ppe_published"
    assert config.value_policies["frc_ppe_published"].tokens[""].status == ValueStatus.MISSING
    options = reader_options(source.reader, "source")
    assert set(options["acquisition_ids"]) == {"state", "district", "campus"}
    assert set(options["witness_ids"]) == {"district", "campus"}
    resource = config.column_contract(DATASET, 2025, reader_profile="source")
    assert resource is not None
    doc = parse_column_schema(resource["text"].encode())
    tables = crosswalk_dataset(doc, DATASET, "2025")["tables"]
    assert {t["table"]: t["layout"]["last_data_record"] - t["layout"]["first_data_record"] + 1 for t in tables} == {
        "state": 1, "district": 1202, "campus": 8866,
    }


def test_frc_published_state_denominator_requires_direct_report():
    values = ["5,530,672", "$12,147", "$8,399", "$2,502", "$10,900", "$747", "$500", "$1,247"]
    head = _html([_TITLE, ["2025 Federal Report Card"], ["State"]])
    section = '<h1>Part (x): Per-pupil Expenditure</h1> preceding fiscal year (2024-25) '
    end = '<h1>Part (xi): STAAR Alternate 2 Participation</h1>'
    prefix = [_FUND_HEAD, ["", *_PPE_HEAD]]
    body = _html([*prefix, ["Total", *values], _BLANK_NOTE])
    assert _selected_row(head + section + body + end) == (3, values)
    with pytest.raises(ContractError, match="geography"):
        _selected_row(head.replace("<td>State</td>", "<td>District</td>") + section + body + end)
    with pytest.raises(ContractError, match="one complete published total"):
        _selected_row(head + section + _html([*prefix, ["Total", *values], ["Total", *values], _BLANK_NOTE]) + end)


@pytest.mark.parametrize("damage", [None, "duplicate", "different_district", "ungrouped", "heading"])
def test_frc_campus_rows_follow_literal_district_groups(damage):
    prefix = [_TITLE, ["2025 Campus Level Per-Pupil Expenditure Summary Report"], ["State"], _FUND_HEAD,
              ["District", "District Name", "Campus", "Campus Name", *_PPE_HEAD]]
    ppe = ["404", "$11,177", "$5,696", "$3,788", "$9,484", "", "$1,317", "$1,694"]
    rows = [["084908", "HITCHCOCK ISD", "084908042", "CROSBY MIDDLE", *ppe], ["084908001", "HITCHCOCK H S", *ppe]]
    if damage == "duplicate":
        rows[1][0] = rows[0][2]
    if damage == "different_district":
        rows[1][0] = "084901001"
    if damage == "ungrouped":
        rows.pop(0)
    if damage == "heading":
        prefix[-1][-2] = "Site"
    text = _html([*prefix, *rows, *_FOOTER])
    if damage is None:
        assert _bulk_rows(text, "campus") == [(6, rows[0]), (7, [*rows[0][:2], *rows[1]])]
    else:
        with pytest.raises(ContractError, match="FRC PPE"):
            _bulk_rows(text, "campus")


def test_frc_per_pupil_projection_preserves_blank_and_signed_zeros():
    fields = [{"code": "Enrollment", "field": "reported_enrollment", "disposition": "fact"},
              {"code": "Total", "field": "all_sources_per_pupil", "disposition": "fact"}]
    rows = [(i, ["404", token]) for i, token in enumerate(("", "$0", "$-0", "$-117", "$1,000"), 6)]
    table, support = _projection(rows, fields)
    name = "all_sources_per_pupil"
    assert table[name].to_pylist() == [None, 0, 0, -117, 1000]
    assert table[f"frc_raw_{name}"].to_pylist() == ["", "$0", "$-0", "$-117", "$1,000"]
    assert table[f"frc_status_{name}"].to_pylist() == ["missing", *["observed"] * 4]
    assert table[SOURCE_ROW].to_pylist() == [6, 7, 8, 9, 10] and support["context"] == [SOURCE_ROW]
    for values in (["", "$0"], ["404", "-999"], ["404", "$12.45"], ["404", "$1234"]):
        with pytest.raises(ContractError, match="unreviewed token"):
            _projection([(6, values)], fields)
    assert measure("$-117", unit=UNIT).value == -117
    assert measure("$1,000", unit=UNIT).display() == "$1,000 per pupil"


def test_frc_comparison_requires_one_attachment_for_both_editions():
    local = EditionKey(DATASET, "2025", "1" * 64, "district", "district", "one row per district_number")
    state = EditionKey(DATASET, "2025", "2" * 64, "state", "state", "statewide total")
    release = object.__new__(DataRelease)
    release._v3 = False
    release.catalog = SimpleNamespace(native_sources=(
        NativeSourceRef("district.json", "a" * 64, "b" * 64, (local,)),
        NativeSourceRef("state.json", "c" * 64, "d" * 64, (state,)),
    ))
    query = Query([District(uuid4(), "Example ISD", district_number="084908")], repo=None)
    with pytest.raises(ContractError, match="both exact editions in one native attachment"):
        query.compare_statewide(release=release, local=local, statewide=state, period=PERIOD)


@pytest.mark.parametrize("attached", [False, True])
def test_frc_comparison_rejects_repackaged_native_manifest_before_metadata(attached):
    local = EditionKey(DATASET, "2025", "1" * 64, "district", "district", "one row per district_number")
    state = EditionKey(DATASET, "2025", "2" * 64, "state", "state", "statewide total")
    release = object.__new__(DataRelease)
    release._v3 = not attached
    if attached:
        release.catalog = SimpleNamespace(native_sources=(
            NativeSourceRef("unreviewed.json", "a" * 64, "b" * 64, (local, state)),
        ))
    else:
        release._manifest_sha256 = "a" * 64
    query = Query([District(uuid4(), "Example ISD", district_number="084908")], repo=None)
    with pytest.raises(ContractError, match="exact qualified FY2025 native manifest"):
        query.compare_statewide(release=release, local=local, statewide=state, period=PERIOD)


@pytest.mark.parametrize("raw,status,value,unit,valid", [
    ("", "missing", None, UNIT, True),
    ("$-0", "observed", 0, UNIT, True),
    ("$-117", "observed", -117, UNIT, True),
    ("5,530,672", "observed", 5530672, "count", True),
    ("$1000", "observed", 1000, UNIT, False),
    ("$1,000", "observed", 1001, UNIT, False),
    ("$0", "missing", None, UNIT, False),
    ("", "missing", None, "count", False),
    ("0", "observed", False, "count", False),
])
def test_frc_comparison_validates_published_token(raw, status, value, unit, valid):
    plan = SimpleNamespace(support={"selected": ("raw", "status")})
    row = {"selected": value, "raw": raw, "status": status}
    if valid:
        assert _frc_value(plan, row, "selected", unit) == (raw, status, value)
    else:
        with pytest.raises(ContractError, match="published token and status"):
            _frc_value(plan, row, "selected", unit)
