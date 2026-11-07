import pytest

from teadata.educator_certification import DETAIL_HEADERS, ROLLUP_HEADERS, _selected_rows
from teadata.manifest import ContractError


def test_certification_admits_only_the_three_way_witness():
    key = ("Teacher", "2018-19")
    direct = ("2018-19", "All", "Teacher", "All", "All", 10)
    unknown = ("2018-19", "All", "Principal", "All", "All", 999)
    first = ("2018-19", "University", "Teacher", "Female", "Hispanic/Latino", 6)
    second = ("2018-19", "University", "Teacher", "Male", "Hispanic/Latino", 4)
    assert _selected_rows([ROLLUP_HEADERS, direct, unknown], [DETAIL_HEADERS, first, second],
                          {key: (2, 10)}) == {key: (2, direct)}
    with pytest.raises(ContractError, match="archive witness"):
        _selected_rows([ROLLUP_HEADERS, direct], [DETAIL_HEADERS, first, second], {key: (2, 11)})
    with pytest.raises(ContractError, match="detail keys"):
        _selected_rows([ROLLUP_HEADERS, direct], [DETAIL_HEADERS, first, first, second], {key: (2, 10)})
    with pytest.raises(ContractError, match="detail and rollup totals"):
        _selected_rows([ROLLUP_HEADERS, direct], [DETAIL_HEADERS, first], {key: (2, 10)})
