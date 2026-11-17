from copy import deepcopy

import pytest

from teadata.educator_certification import ADMITTED, DETAIL_HEADERS, ROLLUP_HEADERS, _detail_diagnostics, _selected_rows
from teadata.manifest import ContractError


def _direct():
    row = ("2018-19", "All", "Teacher", "All", "All", 10)
    return row, {("Teacher", "2018-19"): (2, row)}


def test_certification_selects_the_complete_direct_tuple_without_detail_aggregation():
    row, witnesses = _direct()
    assert _selected_rows([ROLLUP_HEADERS, row], witnesses) == {("Teacher", "2018-19"): (2, row)}
    assert ADMITTED == {role: tuple(range(2016, 2027)) for role in ("Teacher", "Principal", "Superintendent")}


@pytest.mark.parametrize("index,value", [(0, "2019-20"), (1, "University"), (2, "Principal"),
                                        (3, "Female"), (4, "Other"), (5, 11), (5, True), (5, "10")])
def test_certification_rejects_every_changed_direct_cell(index, value):
    row, witnesses = _direct()
    changed = tuple(value if i == index else v for i, v in enumerate(row))
    with pytest.raises(ContractError, match="XML witness"):
        _selected_rows([ROLLUP_HEADERS, changed], witnesses)


def test_certification_rejects_duplicate_missing_reordered_and_wrong_width_rows():
    row, witnesses = _direct()
    for rows in ([ROLLUP_HEADERS, row, row], [ROLLUP_HEADERS],
                 [ROLLUP_HEADERS, row[:-1]], [ROLLUP_HEADERS, ("2024-25", *row[1:])]):
        with pytest.raises(ContractError):
            _selected_rows(rows, witnesses)
    moved = deepcopy(witnesses)
    moved["Teacher", "2018-19"] = (3, row)
    with pytest.raises(ContractError, match="XML witness"):
        _selected_rows([ROLLUP_HEADERS, row], moved)


@pytest.mark.parametrize("counts,outcomes,unique", [([], ["missing_detail", "sum_differs"], 0),
    ([10], ["matches"], 1), ([5], ["sum_differs"], 1), ([5, 5], ["duplicate_keys"], 1),
    ([6, 6], ["duplicate_keys", "sum_differs"], 1)])
def test_detail_diagnostics_report_observed_outcomes_without_qualifying_detail(counts, outcomes, unique):
    row, witnesses = _direct()
    direct = _selected_rows([ROLLUP_HEADERS, row], witnesses)
    detail = [("2018-19", "University", "Teacher", "Female", "Hispanic/Latino", count) for count in counts]
    diagnostic, = _detail_diagnostics([DETAIL_HEADERS, *detail], direct)
    assert diagnostic["outcomes"] == outcomes
    assert diagnostic["sum"] == sum(counts) and diagnostic["unique_keys"] == unique
    assert diagnostic["detail_qualified"] is False
    assert direct["Teacher", "2018-19"][1][-1] == 10


@pytest.mark.parametrize("detail", [("2018-19", "University", "Unknown", "Female", "Other", 10),
                                    ("2018-19", "University", "Teacher", "Female", "Other", "*"),
                                    ("2018-19", "University", "Teacher", "Female", "Other", True)])
def test_detail_diagnostic_refuses_malformed_values_without_changing_direct_totals(detail):
    row, witnesses = _direct()
    with pytest.raises(ContractError, match="diagnostic input"):
        _detail_diagnostics([DETAIL_HEADERS, detail], _selected_rows([ROLLUP_HEADERS, row], witnesses))
