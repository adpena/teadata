import pytest

from teadata.educator_hiring import (
    CATEGORIES, DISTRICT_DATASET, EPP_DATASET, STATE_DATASET, _epp_counts, _epp_reference, _projection, _qualify,
    _qualify_epp, _read_rows, _read_years, _selected_years,
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


@pytest.mark.parametrize(("index", "value", "valid"), [(5, "", True), (5, None, False), (5, 123, False),
                                                        (6, "", False), (7, "", False)])
def test_hiring_preserves_publisher_blank_name_without_weakening_other_labels(index, value, valid):
    row = list(district_row(2))
    row[index] = value
    if valid:
        table, _ = _projection([tuple(row)], [29650], DISTRICT_DATASET)
        assert table["district_name"].to_pylist() == [""]
        assert table["new_hires_count"].to_pylist() == [2]
        assert table["publisher_worksheet_row"].to_pylist() == [29650]
    else:
        with pytest.raises(ContractError, match="identity or label"):
            _projection([tuple(row)], [29650], DISTRICT_DATASET)


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
    for dataset, edition in ((DISTRICT_DATASET, "5"), (STATE_DATASET, "2026-04"), (EPP_DATASET, "5")):
        source = cfg.resolve_dataset(dataset, 2026, reader_profile="source", exact=True)
        assert source.metadata.version == edition
        assert source.metadata.period.kind == "unknown"
        records = cfg.header_records(dataset, edition=edition, reader_profile="source")
        assert {row["period"]["end_year"] for row in records if row["disposition"] == "fact"} == set(range(2016 if dataset == STATE_DATASET else 2010, 2027))
        assert cfg.is_source_only(dataset, 2026)
        with pytest.raises(ContractError, match="Reader profile is required"):
            cfg.load_df(dataset, 2026)
    state = cfg.resolve_dataset(STATE_DATASET, 2026, reader_profile="source", exact=True)
    assert state.metadata.metrics["intern_certified_ratio"].unit == "ratio"


@pytest.mark.parametrize("years", [(), (2009,), (2027,), (True,), (2026, 2026), (2025, "2026")])
def test_hiring_rejects_unsupported_or_ambiguous_measurement_years(years):
    with pytest.raises(ContractError, match="supported school-year"):
        _selected_years(years)


def test_hiring_reads_multiple_years_once_and_preserves_old_report_roster(tmp_path, monkeypatch):
    from openpyxl import Workbook
    import teadata.educator_hiring as hiring

    path = tmp_path / "source.xlsx"
    book = Workbook()
    sheet = book.active
    sheet.title = "District Counts"
    sheet.append(hiring._DISTRICT_HEADERS)
    prior = list(district_row(7))
    prior[0], prior[1], prior[-1] = "999999", 2016, "2015-2016"
    sheet.append(prior)
    sheet.append(district_row(0))
    book.save(path)
    book.close()
    opened = []
    original = hiring.load_workbook

    def track(*args, **kwargs):
        opened.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(hiring, "load_workbook", track)
    selected = _read_years(path, DISTRICT_DATASET, (2026, 2016))
    assert len(opened) == 1 and tuple(selected) == (2016, 2026)
    assert selected[2016] == ([tuple(prior)], [2])
    table, _ = _projection(*selected[2016], DISTRICT_DATASET, school_year=2016)
    assert table["district_number"].to_pylist() == ["999999"]
    assert table["school_year_end"].to_pylist() == [2016]
    with pytest.raises(ContractError, match="measurement year"):
        _projection(*selected[2016], DISTRICT_DATASET)


def test_hiring_missing_selected_year_raises_instead_of_returning_current_rows(tmp_path):
    from openpyxl import Workbook
    from teadata.educator_hiring import _DISTRICT_HEADERS

    path = tmp_path / "source.xlsx"
    book = Workbook()
    sheet = book.active
    sheet.title = "District Counts"
    sheet.append(_DISTRICT_HEADERS)
    sheet.append(district_row(1))
    book.save(path)
    book.close()
    with pytest.raises(ContractError, match="exact selected school-year"):
        _read_years(path, DISTRICT_DATASET, (2016, 2026))


def test_earlier_district_year_is_supported_but_state_year_is_not():
    row = list(district_row(2))
    row[1], row[-1] = 2013, "2012-2013"
    assert _projection([tuple(row)], [45], DISTRICT_DATASET, school_year=2013)[0].num_rows == 1
    with pytest.raises(ContractError, match="supported school-year"):
        _projection([], [], STATE_DATASET, school_year=2013)


def epp_row(count=2):
    return ("001902", 2013, "07", "Cayuga ISD", "Rural", "07", count, "Example EPP",
            "Intern Certified", "07-Kilgore", "2012-2013", "07")


def epp_workbook(tmp_path, rows):
    from openpyxl import Workbook
    from teadata.educator_hiring import _EPP_HEADERS

    path = tmp_path / "epp.xlsx"
    book = Workbook()
    sheet = book.active
    sheet.title = "District Counts Long"
    sheet.append(_EPP_HEADERS)
    for row in rows:
        sheet.append(row)
    book.save(path)
    book.close()
    return path


def test_epp_comparison_aggregates_only_distinct_publisher_keys_and_requires_exact_key_sets(tmp_path):
    from collections import Counter

    second = list(epp_row(3))
    second[7] = "Another EPP"
    counts = _epp_counts(epp_workbook(tmp_path, [epp_row(), second]), (2013,))
    row = list(district_row(5))
    row[1], row[-1] = 2013, "2012-2013"
    district, _ = _projection([tuple(row)], [45], DISTRICT_DATASET, school_year=2013)
    assert _qualify_epp(district, counts, 2013)["epp_comparison_keys"] == 1
    assert _qualify_epp(district, counts, 2013)["direct_state_comparison"] == "unavailable"
    extra_zero = Counter(counts)
    extra_zero[2013, "001903", "Intern Certified"] = 0
    assert extra_zero == counts  # Counter equality alone would hide this unmatched publisher key.
    for invalid in (extra_zero, Counter(), Counter({(2013, "001902", "Intern Certified"): 6})):
        with pytest.raises(ContractError, match="keys or counts"):
            _qualify_epp(district, invalid, 2013)


@pytest.mark.parametrize(("index", "value"), [(0, "1902"), (2, "7"), (5, ""), (7, ""), (6, -1),
                                            (6, True), (6, None), (8, "Alternative"), (10, "2013-2014")])
def test_epp_comparison_rejects_unreviewed_tokens_and_population(tmp_path, index, value):
    row = list(epp_row())
    row[index] = value
    with pytest.raises(ContractError, match="key, count or population"):
        _epp_counts(epp_workbook(tmp_path, [row]), (2013,))


def test_epp_comparison_rejects_duplicate_source_keys(tmp_path):
    with pytest.raises(ContractError, match="key, count or population"):
        _epp_counts(epp_workbook(tmp_path, [epp_row(), epp_row()]), (2013,))


def test_epp_native_rows_keep_region_placeholder_and_blank_name_without_collapse():
    first = list(epp_row(0))
    first[3], first[5], first[7], first[11] = '', 'NA', 'Na-Uncertified', 'NA'
    second = list(first)
    second[5], second[11] = '07', '07-Kilgore'
    table, support = _projection([tuple(first), tuple(second)], [8, 9], EPP_DATASET, school_year=2013)
    assert table['publisher_epp_region'].to_pylist() == ['NA', '07']
    assert table['reported_epp_assignment'].to_pylist() == ['Na-Uncertified'] * 2
    assert table['district_name'].to_pylist() == ['', '']
    assert table['raw_new_hires_count'].to_pylist() == ['0', '0']
    assert table['status_new_hires_count'].to_pylist() == ['observed', 'observed']
    assert table['publisher_worksheet_row'].to_pylist() == [8, 9]
    assert support['fields']['new_hires_count']['raw'] == 'raw_new_hires_count'


def test_epp_dictionary_does_not_guess_renamed_codes_or_institution_id():
    cfg = load_config()
    records = cfg.header_records(EPP_DATASET, edition='5', table='district_2013', reader_profile='source')
    documented = {row['code']: row for row in records if row['dictionary_status'] == 'documented'}
    assert set(documented) == {'schyr', 'region_epp', 'cert_cat', 'EPP_assigned', 'total_new_hires'}
    assert all(row['code'] == row['label'] for row in records)
    assert documented['EPP_assigned']['field'] == 'reported_epp_assignment'
    assert documented['EPP_assigned']['dictionary_occurrences'][0]['source_line'] == 9
    assert {row['field'] for row in records}.isdisjoint({'epp_id', 'preparation_route'})


def test_epp_reference_must_match_exact_publisher_metadata_before_opening_source():
    from dataclasses import replace
    from types import SimpleNamespace

    source = load_config().resolve_dataset(DISTRICT_DATASET, 2026, reader_profile="source", exact=True)
    ref, = [ref for ref in source.metadata.references if ref.reference_id == "hiring_epp_comparison"]
    for changed in (replace(ref, url="https://example.invalid/wrong.xlsx"), replace(ref, reviewed_sha256="f" * 64),
                    replace(ref, provenance_ids=("f" * 64,))):
        bad = replace(source, metadata=replace(source.metadata, references=(changed,)))
        with pytest.raises(ContractError, match="exact reviewed EPP"):
            _epp_reference(bad, SimpleNamespace())
