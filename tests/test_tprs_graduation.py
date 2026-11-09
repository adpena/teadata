from decimal import Decimal
from html import escape

import pytest

from teadata.manifest import ContractError
from teadata.tprs_graduation import parse_graduation_profile
from teadata.values import ValueStatus


def _campus_html() -> bytes:
    rows = [
        ["", "Texas Education Agency", ""],
        ["2025-26 Graduation Profile"],
        ["TEST CAMPUS (001902103) - TEST DISTRICT - TEST COUNTY"],
        ["2024-25", "2025-26"],
        ["", "CampusCount", "CampusPercent", "DistrictCount", "StateCount"],
        ["Graduates (2024-25 Annual Graduates)"],
        ["Total Graduates", "0", "0.0%", "36", "400,365"],
        ["By Ethnicity:"],
        *[[label, "-", "-", "0", "1"] for label in (
            "African American", "Hispanic", "White", "American Indian", "Asian", "Pacific Islander", "Two or More Races",
        )],
        ["By Graduation Type:"],
        *[[label, "-", "-", "0", "1"] for label in (
            "Minimum H.S. Program", "Recommended H.S. Program/Distinguished Achievement Program",
            "Foundation H.S. Program (No Endorsement)", "Foundation H.S. Program (Endorsement)",
            "Foundation H.S. Program (DLA)", "Texas First Early H.S. Completion Program (Texas First)",
        )],
        [""],
        ["Special Education Graduates", "?", "?", "1", "38,964"],
        ["Economically Disadvantaged Graduates", "-", "-", "1", "219,172"],
        ["Emergent Bilingual (EB)/English Learner (EL) Graduates", "-", "-", "1", "63,238"],
        ["At-Risk Graduates", "-", "-", "1", "181,093"],
        ["CTE Completers", "-", "-", "1", "148,057"],
        ["-", "Indicates there are no students in the group."],
        ["?", "Indicates that the data for this item were statistically improbable or were reported outside a reasonable range."],
        ["", "TEA | Analytics, Assessment, and Reporting | Performance Reporting", ""],
    ]
    def table(group, *, facts=False):
        summary = ' summary="Procedure Report: Detailed and/or summarized report"' if facts else ""
        return f"<table{summary}>" + "".join(
            "<tr>" + "".join(
                f'<td colspan="5">{escape(cell)}</td>' if facts and len(row) == 1 else f"<td>{escape(cell)}</td>"
                for cell in row
            ) + "</tr>" for row in group
        ) + "</table>"

    return "".join((table(rows[:3]), table(rows[3:4]), table(rows[4:-3], facts=True), table(rows[-3:-1]), table(rows[-1:]))).encode()


def test_graduation_profile_keeps_geography_roles_and_qualifies_tokens():
    report = parse_graduation_profile(_campus_html(), scope="campus", identifier="001902103")
    assert len(report.cells) == 76
    count, percent, district, state = report.cells[:4]
    assert (count.role, count.raw, count.value, count.status, count.row, count.column) == (
        ("local", "count"), "0", 0, ValueStatus.OBSERVED, 7, 2,
    )
    assert (percent.value, percent.status) == (Decimal("0.0"), ValueStatus.OBSERVED)
    assert (district.role, district.value) == (("district_benchmark", "count"), 36)
    assert (state.role, state.value) == (("state_benchmark", "count"), 400365)
    assert (report.cells[4].value, report.cells[4].status, report.cells[4].raw) == (None, ValueStatus.NOT_APPLICABLE, "-")
    special = next(cell for cell in report.cells if cell.label == "Special Education Graduates")
    assert (special.value, special.status, special.raw) == (None, ValueStatus.INVALID, "?")
    assert (report.cells[-1].label, report.cells[-1].value) == ("CTE Completers", 148057)


@pytest.mark.parametrize(("before", "after"), [
    (b"001902103", b"000000000"),
    (b"2024-25 Annual Graduates", b"2025-26 Annual Graduates"),
    (b"CampusPercent", b"StatePercent"),
    (b"CTE Completers", b"CTE Graduates"),
    (b"<td>?</td><td>?</td>", b"<td>*</td><td>?</td>"),
    (b"<td>?</td><td>?</td>", b"<td></td><td>?</td>"),
    (b"no students in the group", b"suppressed for disclosure"),
    (b"<td>Total Graduates</td><td>0</td>", b'<td>Total Graduates</td><td colspan="2">0</td>'),
    (b"<td>Total Graduates</td><td>0</td>", b'<td>Total Graduates</td><td rowspan="2">0</td>'),
    (b"<td>Total Graduates</td><td>0</td>", b'<td>Total Graduates</td><td><script type="application/json">0</script></td>'),
    (b"<td>Total Graduates</td><td>0</td>", b'<td>Total Graduates</td><td><span hidden>0</span></td>'),
    (b"<td>Total Graduates</td><td>0</td>", b'<td>Total Graduates</td><td style="display:none">0</td>'),
    (b"<tr><td>Total Graduates", b"<tr hidden><td>Total Graduates"),
    (b"<tr><td>CTE Completers", b"</table><table><tr><td>CTE Completers"),
])
def test_graduation_profile_rejects_unreviewed_identity_layout_and_tokens(before, after):
    with pytest.raises(ContractError):
        parse_graduation_profile(_campus_html().replace(before, after), scope="campus", identifier="001902103")
