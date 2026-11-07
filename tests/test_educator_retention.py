import pytest

from teadata.dataset_metadata import DataPeriod
from teadata.educator_retention import (
    COHORT_YEARS, DEMOGRAPHIC_DATASET, GENDER_DATASET, ROUTE_DATASET, VIEWS, _project, _validate_rows,
)
from teadata.manifest import ContractError
from teadata.teadata_config import load_config


def _rows():
    view = VIEWS[GENDER_DATASET]
    rows = [view.headers]
    for year in COHORT_YEARS:
        for name, size in ((view.total, 20), ("Female", 12), ("Male", 8)):
            observed = [(size // 2, 0.5)] * min(5, 2026 - year)
            cells = observed + [(None, None)] * (5 - len(observed))
            rows.append((f"{year - 1}-{year % 100:02d}", name, size, *(value for pair in cells for value in pair)))
    return view, rows


def test_retention_preserves_unreported_horizons_and_source_rows():
    view, rows = _rows()
    populations = _validate_rows(rows, view)
    table, _ = _project(populations[2025])
    assert table["retained_year_1_count"].to_pylist() == [10, 6, 4]
    assert table["retained_year_5_ratio"].to_pylist() == [None] * 3
    assert table["raw_retained_year_5_ratio"].to_pylist() == [""] * 3
    assert table["status_retained_year_5_ratio"].to_pylist() == ["missing"] * 3
    assert table["publisher_worksheet_row"].to_pylist() == [35, 36, 37]


@pytest.mark.parametrize(("index", "changes", "message"), [
    (1, {0: "2014"}, "cohort or group"),
    (1, {4: 0.6}, "count or ratio"),
    (2, {2: 10, **dict.fromkeys(range(3, 13, 2), 5)}, "direct total"),
    (34, {11: 0, 12: 0.0}, "beyond the reporting horizon"),
])
def test_retention_rejects_a_changed_publisher_contract(index, changes, message):
    view, rows = _rows()
    rows[index] = tuple(changes.get(column, value) for column, value in enumerate(rows[index]))
    with pytest.raises(ContractError, match=message):
        _validate_rows(rows, view)


def test_retention_config_is_source_only_and_holds_only_the_route_cohort():
    from teadata.column_schema import crosswalk_dataset, parse_column_schema

    config = load_config()
    for dataset in (ROUTE_DATASET, DEMOGRAPHIC_DATASET, GENDER_DATASET):
        source = config.resolve_dataset(dataset, 2026, reader_profile="source", exact=True)
        assert source.metadata.version == "2026-04"
        assert source.metadata.period == DataPeriod()
        assert config.is_source_only(dataset, 2026)
        schema = config.column_contract(dataset, 2026, reader_profile="source")
        assert schema is not None
        tables = crosswalk_dataset(parse_column_schema(schema["text"].encode()), dataset, "2026-04")["tables"]
        years = {item["metric_period"]["end_year"] for item in tables}
        assert years == set(COHORT_YEARS) - ({2018} if dataset == ROUTE_DATASET else set())
