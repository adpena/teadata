from html import escape
from types import SimpleNamespace

import pytest

from teadata.manifest import ContractError
from teadata.native_staar_all_grades import (
    REVIEWED_MANIFEST, _GROUPS, _STANDARDS, _SUBJECTS, _WITNESSES,
    _named_rates, _rate_display, qualify_staar_all_grades,
)
from teadata.values import TokenRule, ValuePolicy, ValueStatus


def _campus_rows():
    geography = ["", "SchoolYear", "State", "District", "Campus"]
    rows = [["TEA"], ["2025-26 STAAR Performance"], ["AUSTIN H S (227901002)"], [],
            [*geography, *_GROUPS[1:]], []]
    for subject in _SUBJECTS:
        rows.append([subject])
        for standard in _STANDARDS:
            rows.append([standard, "2025", "99%", "99%", *(["99%"] * len(_GROUPS))])
            rows.append([standard, "2026", "80%", "79%", "0%", "*", "-", *(["72%"] * (len(_GROUPS) - 3))])
    return rows


def _html(rows):
    return "<table>" + "".join(
        "<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>" for row in rows
    ) + "</table>"


def test_staars_named_rates_use_the_selected_geography_year_and_groups():
    rates = _named_rates(_html(_campus_rows()), "campus", _WITNESSES["campus"])
    stem = "STAAR Performance, All Subjects, SY 2025-26, "
    criterion = ", % At Approaches GL Standard or Above"
    assert len(rates) == 210
    assert [rates[stem + group + criterion] for group in _GROUPS[:4]] == ["0%", "*", "-", "72%"]


@pytest.mark.parametrize(("damage", "message"), [
    ("wrong_geography", "geography"),
    ("wrong_column_order", "groups"),
    ("duplicate_panel", "duplicated"),
    ("missing_row", "missing rate"),
])
def test_staars_named_rates_reject_wrong_or_incomplete_reports(damage, message):
    rows = _campus_rows()
    if damage == "wrong_geography":
        rows[2] = ["AUSTIN H S (227901003)"]
    elif damage == "wrong_column_order":
        rows[4][5], rows[4][6] = rows[4][6], rows[4][5]
    elif damage == "duplicate_panel":
        rows.append(rows[-1])
    else:
        rows.pop()
    with pytest.raises(ContractError, match=message):
        _named_rates(_html(rows), "campus", _WITNESSES["campus"])


def test_staars_raw_token_keeps_a_real_zero_distinct_from_masked_and_missing():
    contract = SimpleNamespace(value_policy=ValuePolicy("test", {
        "-1": TokenRule(ValueStatus.SUPPRESSED, "Publisher masks the rate"),
    }))
    assert [_rate_display(value, contract) for value in ("0", "-1", "", "100")] == [
        ("0%", "observed"), ("*", "suppressed"), ("-", "missing"), ("100%", "observed"),
    ]
    for value in ("-3", "101", "00", " 0", None, 0):
        with pytest.raises(ContractError, match="unsupported raw rate token"):
            _rate_display(value, contract)


@pytest.mark.parametrize(("pin", "offline"), [("a" * 64, True), (REVIEWED_MANIFEST, False)])
def test_staars_qualification_fails_closed_before_opening_any_other_release(pin, offline):
    with pytest.raises(ContractError, match="exact reviewed offline manifest"):
        qualify_staar_all_grades(SimpleNamespace(manifest_sha256=pin, offline=offline), None)
