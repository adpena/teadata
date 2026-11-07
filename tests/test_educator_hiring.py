import pytest

from teadata.educator_hiring import (
    CATEGORIES, DISTRICT_DATASET, STATE_DATASET, _projection, _qualify, _read_rows,
)
from teadata.manifest import ContractError
from teadata.teadata_config import load_config


def district_row(count=0):
    return ("001902", 2026, "07", "Intern Certified", count, "Cayuga ISD", "Rural", "07-Kilgore", "2025-2026")


def test_hiring_projection_keeps_zero_raw_value_status_and_key():
    table, support = _projection([district_row()], [123], DISTRICT_DATASET)
    assert table["district_number"].to_pylist() == ["001902"]
    assert table["new_hires_count"].to_pylist() == [0]
    assert table["raw_new_hires_count"].to_pylist() == ["0"]
    assert table["status_new_hires_count"].to_pylist() == ["observed"]
    assert table["publisher_worksheet_row"].to_pylist() == [123]
    assert support["fields"]["new_hires_count"] == {
        "raw": "raw_new_hires_count", "status": "status_new_hires_count",
    }


@pytest.mark.parametrize(("index", "value"), [(0, "1902"), (1, 2025), (2, "7"), (3, "Alternative"),
                                            (4, -1), (4, None), (4, True), (4, "=1+1")])
def test_hiring_rejects_unreviewed_keys_population_or_tokens(index, value):
    row = list(district_row())
    row[index] = value
    with pytest.raises(ContractError):
        _projection([tuple(row)], [2], DISTRICT_DATASET)


def test_hiring_rejects_duplicate_district_category_without_aggregation():
    with pytest.raises(ContractError, match="key, category"):
        _projection([district_row(1), district_row(2)], [2, 3], DISTRICT_DATASET)


def test_hiring_category_comparison_uses_explicit_mapping_and_direct_state():
    values = (4426, 1806, 824, 4028, 981, 12096, 1344, 16071, 2570)
    total = 44146
    rows = []
    for (_, category, _), value in zip(CATEGORIES, values, strict=True):
        row = list(district_row(value))
        row[3] = category
        rows.append(tuple(row))
    district, _ = _projection(rows, list(range(2, 11)), DISTRICT_DATASET)
    state_row = ("2025-26", total, *values, *(value / total for value in values))
    state, _ = _projection([state_row], [12], STATE_DATASET)
    assert _qualify(district, state)["matched_categories"] == 9
    assert state["intern_certified_ratio"].to_pylist() == [0.10025823404158928]
    assert state["raw_intern_certified_ratio"].to_pylist() == ["0.10025823404158928"]
    damaged = state.to_pydict()
    damaged["reenterer_count"] = [16072]
    import pyarrow as pa
    with pytest.raises(ContractError, match="category or published ratio"):
        _qualify(district, pa.table(damaged))


def test_hiring_source_selection_excludes_historical_rows(tmp_path):
    from openpyxl import Workbook

    from teadata.educator_hiring import _DISTRICT_HEADERS

    path = tmp_path / "source.xlsx"
    book = Workbook()
    sheet = book.active
    sheet.title = "District Counts"
    sheet.append(_DISTRICT_HEADERS)
    prior = list(district_row(1))
    prior[1], prior[-1] = 2025, "2024-2025"
    sheet.append(prior)
    sheet.append(district_row(0))
    book.save(path)
    book.close()
    rows, locators = _read_rows(path, DISTRICT_DATASET)
    assert rows == [district_row(0)] and locators == [3]


def test_hiring_contract_is_source_only_and_keeps_product_and_measurement_years():
    cfg = load_config()
    for dataset, edition in ((DISTRICT_DATASET, "5"), (STATE_DATASET, "2026-04")):
        source = cfg.resolve_dataset(dataset, 2026, reader_profile="source", exact=True)
        assert source.metadata.version == edition
        assert source.metadata.period.end_year == 2026
        assert cfg.is_source_only(dataset, 2026)
        with pytest.raises(ContractError, match="Reader profile is required"):
            cfg.load_df(dataset, 2026)
    state = cfg.resolve_dataset(STATE_DATASET, 2026, reader_profile="source", exact=True)
    assert state.metadata.metrics["intern_certified_ratio"].unit == "ratio"
