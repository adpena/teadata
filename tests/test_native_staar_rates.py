from html import escape
from types import SimpleNamespace

import pytest

from teadata.manifest import ContractError
from teadata.native_staar_rates import (
    ALL_GRADES_MANIFEST, END_OF_COURSE_MANIFEST, QUALIFICATIONS, _GROUPS,
    _STANDARDS, _SUBJECTS, _WITNESSES_BY_EDITION, _named_rates, _rate_display, qualify_staar_rates,
)
from teadata.values import TokenRule, ValuePolicy, ValueStatus


def _campus_rows(year="2026", table="staar_all_grades"):
    geography = ["", "SchoolYear", "State", "District", "Campus"]
    groups = _GROUPS[1:] if year == "2026" else (*_GROUPS[1:-1], "EB/EL (Current & Monitored)")
    rows = [["TEA"], [f"{int(year) - 1}-{year[-2:]} STAAR Performance"], ["AUSTIN H S (227901002)"], [],
            [*geography, *groups], []]
    for subject in _SUBJECTS[table]:
        rows.append([subject])
        for standard in _STANDARDS:
            rows.append([standard, str(int(year) - 1), "99%", "99%", *(["99%"] * len(_GROUPS))])
            rows.append([standard, year, "80%", "79%", "0%", "*", "-", *(["72%"] * (len(_GROUPS) - 3))])
    return rows


def _html(rows):
    return "<table>" + "".join(
        "<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>" for row in rows
    ) + "</table>"


@pytest.mark.parametrize(("year", "table", "subject"), [
    ("2024", "staar_end_of_course", "AR11"), ("2025", "staar_end_of_course", "AR11"),
    ("2026", "staar_end_of_course", "AR11"), ("2026", "staar_all_grades", "A001"),
])
def test_staar_named_rates_use_the_selected_geography_year_and_groups(year, table, subject):
    rates = _named_rates(_html(_campus_rows(year, table)), year, table, "campus", _WITNESSES_BY_EDITION[year]["campus"])
    assert len(rates) == 210
    assert [rates[f"{group}00{subject}S{year[-2:]}R"] for group in ("DA", "DB", "DH", "DW")] == ["0%", "*", "-", "72%"]


@pytest.mark.parametrize(("damage", "message"), [
    ("wrong_geography", "geography"),
    ("wrong_column_order", "groups"),
    ("duplicate_panel", "duplicated"),
    ("missing_row", "missing rate"),
])
def test_staar_named_rates_reject_wrong_or_incomplete_reports(damage, message):
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
        _named_rates(_html(rows), "2026", "staar_all_grades", "campus", _WITNESSES_BY_EDITION["2026"]["campus"])


def test_staar_named_rates_reject_other_family_and_wrong_historical_header():
    rows = _campus_rows("2024", "staar_end_of_course")
    witness = _WITNESSES_BY_EDITION["2024"]["campus"]
    with pytest.raises(ContractError, match="missing rate"):
        _named_rates(_html(rows), "2024", "staar_all_grades", "campus", witness)
    rows[4][-1] = _GROUPS[-1]
    with pytest.raises(ContractError, match="groups"):
        _named_rates(_html(rows), "2024", "staar_end_of_course", "campus", witness)


def test_staar_raw_token_keeps_a_real_zero_distinct_from_masked_and_missing():
    contract = SimpleNamespace(value_policy=ValuePolicy("test", {
        "-1": TokenRule(ValueStatus.SUPPRESSED, "Publisher masks the rate"),
    }))
    assert [_rate_display(value, contract) for value in ("0", "-1", "", "100")] == [
        ("0%", "observed"), ("*", "suppressed"), ("-", "missing"), ("100%", "observed"),
    ]
    for value in ("-3", "101", "00", " 0", None, 0):
        with pytest.raises(ContractError, match="unsupported raw rate token"):
            _rate_display(value, contract)


@pytest.mark.parametrize(("pin", "offline"), [("a" * 64, True), (ALL_GRADES_MANIFEST, False), (END_OF_COURSE_MANIFEST, False)])
def test_staar_qualification_fails_closed_before_opening_any_other_release(pin, offline):
    with pytest.raises(ContractError, match="exact reviewed offline manifest"):
        qualify_staar_rates(SimpleNamespace(manifest_sha256=pin, offline=offline), None)


def test_eoc_requires_every_edition_and_scope_and_all_methodology_roots():
    plan = QUALIFICATIONS[END_OF_COURSE_MANIFEST]
    refs = [SimpleNamespace(edition=SimpleNamespace(table=plan.table, scope=scope, edition=year),
                            contract={"rows": witness.rows})
            for year in plan.editions for scope, witness in _WITNESSES_BY_EDITION[year].items()]
    catalog = SimpleNamespace(to_dict=lambda: {"artifacts": range(len(refs))}, artifact_ref=refs.__getitem__)
    release = SimpleNamespace(manifest_sha256=END_OF_COURSE_MANIFEST, offline=True, catalog=catalog)
    named = {witness.receipt for year in plan.editions for witness in _WITNESSES_BY_EDITION[year].values()}

    def missing_record(roots):
        assert named < set(roots)
        raise FileNotFoundError("retained methodology")

    with pytest.raises(FileNotFoundError, match="methodology"):
        qualify_staar_rates(release, SimpleNamespace(verify_record_closure=missing_record))
    refs[-1] = refs[0]
    with pytest.raises(ContractError, match="edition and scope"):
        qualify_staar_rates(release, None)
