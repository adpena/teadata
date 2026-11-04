"""Published PEIMS source discovery without implicit hydration or year fallback."""

from pathlib import Path

import pytest

from teadata.column_schema import crosswalk_dataset, parse_column_schema, reader_options, table_input_sha256s
from teadata.manifest import ContractError
from teadata.peims_leaves import PRIMARY_DISABILITY
from teadata.provenance import ProvenanceStore, validate_record_graph, validate_reference_records
from teadata.teadata_config import load_config
from teadata.values import ValueStatus, measure


ENROLLMENT = "peims_student_enrollment_by_ethnicity"
PROGRAMS = "peims_student_programs_and_special_populations"


@pytest.mark.parametrize(
    ("dataset", "source_date", "widths", "row_counts", "total_field"),
    [
        (ENROLLMENT, "2026-03-06", (3, 8, 10), (7, 6873, 48971), "enrollment_by_ethnicity_count"),
        (PROGRAMS, "2026-03-13", (17, 21, 25), (1, 1202, 9113), "total_enrollment_count"),
    ],
)
def test_source_discovery_is_exact_and_excluded_from_retained_builds(
    dataset, source_date, widths, row_counts, total_field, monkeypatch
):
    cfg = load_config()

    def reject_read(*args, **kwargs):
        pytest.fail("Source-only registry opened data")

    monkeypatch.setattr(type(cfg), "_load_table", reject_read)
    selected = cfg.resolve_dataset(dataset, 2026, exact=True, reader_profile="source")
    assert selected.metadata.published_at == source_date
    assert selected.metadata.metrics[total_field].unit == "count"
    assert selected.metadata.period.label == "2025-2026"
    assert selected.resolved_year == 2026
    assert {reference.role for reference in selected.metadata.references} == {"source_notes", "masking_methodology", "schema"}
    assert not selected.metadata.provenance_ids
    acquisitions = reader_options(selected.reader, "source")["acquisition_ids"]
    assert set(acquisitions) == {"state", "district", "campus"}
    assert cfg.column_schema(dataset, 2026) is None
    provenance = ProvenanceStore(Path(__file__).resolve().parents[1] / "provenance")
    roots = list(acquisitions.values())
    roots.extend(identifier for reference in selected.metadata.references for identifier in reference.provenance_ids)
    graph = validate_record_graph(provenance.read_record, roots)
    validate_reference_records(selected, graph.__getitem__, verify_all_claimed=True)

    # Retained builds can request an older year without opening opt-in sources.
    for year in (2022, 2023, 2024, 2025, 2026, None):
        assert cfg.is_source_only(dataset, year)
    assert not cfg.is_source_only("accountability", 2025)
    with pytest.raises(KeyError):
        cfg.resolve_dataset(dataset, 2022, exact=True)
    with pytest.raises(ContractError, match="Reader profile is required"):
        cfg.load_df(dataset, 2026, exact=True)
    with pytest.raises(ContractError, match="requires a dedicated preparation workflow"):
        cfg.load_df(dataset, 2026, reader_profile="source", exact=True)

    contract = cfg.column_contract(dataset, 2026, reader_profile="source")
    assert contract is not None
    selected_crosswalk = crosswalk_dataset(parse_column_schema(contract["text"].encode()), dataset, "2026")
    tables = selected_crosswalk["tables"]
    assert [table["table"] for table in tables] == ["state", "district", "campus"]
    assert len({table["input_sha256"] for table in tables}) == 3
    for table, width, row_count in zip(tables, widths, row_counts, strict=True):
        receipt = graph[acquisitions[table["table"]]]
        assert receipt["dataset"] == dataset and receipt["resolved_year"] == 2026
        assert receipt["source_metadata"]["version"] == "2026"
        assert receipt["parameters"]["replayed_from"] in graph
        assert table_input_sha256s(table) == (receipt["artifact"]["sha256"],)
        layout = table["layout"]
        assert layout["last_data_record"] - layout["first_data_record"] + 1 == row_count
        fields = cfg.header_records(dataset, edition="2026", table=table["table"], reader_profile="source")
        assert len(fields) == width
        assert [field["position"] for field in fields] == list(range(1, width + 1))
        assert all(field["code"] == field["label"] and field["header_identity"] == "position" for field in fields)
        assert [field["period"]["end_year"] for field in fields if field["field"] == total_field] == [2026]
        assert all(field["period"] == {"kind": "unknown"} for field in fields if field["disposition"] == "identity")


def test_mixed_editions_never_hide_required_retained_sources():
    cfg = load_config()
    active = cfg.data_sources["accountability"].entries[2025]
    editions = cfg.data_sources[ENROLLMENT].entries
    editions[2025] = active
    assert not cfg.is_source_only(ENROLLMENT, 2025)
    assert cfg.is_source_only(ENROLLMENT, 2026)
    with pytest.raises(KeyError, match="no prior fallback"):
        cfg.is_source_only(ENROLLMENT, 2022)

    # A future active edition also prevents the dataset from being classified
    # as exclusively source-only for an unavailable earlier edition.
    del editions[2025]
    editions[2027] = active
    assert not cfg.is_source_only(ENROLLMENT, 2027)
    with pytest.raises(KeyError, match="no prior fallback"):
        cfg.is_source_only(ENROLLMENT, 2022)


@pytest.mark.parametrize(('dataset', 'year', 'rows'), [
    (ENROLLMENT, 2023, (7, 6911, 48497)), (ENROLLMENT, 2024, (7, 6912, 48716)),
    (ENROLLMENT, 2025, (7, 6910, 48850)), (PROGRAMS, 2023, (1, 1209, 9054)),
    (PROGRAMS, 2024, (1, 1207, 9088)), (PROGRAMS, 2025, (1, 1208, 9088)),
])
def test_historical_peims_bind_each_scope_and_preserve_literal_headers(dataset, year, rows):
    cfg = load_config()
    selected = cfg.resolve_dataset(dataset, year, exact=True, reader_profile='source')
    assert selected.metadata.published_at is None
    assert selected.metadata.period.label == f'{year - 1}-{year}'
    assert cfg.is_source_only(dataset, year) and cfg.column_schema(dataset, year) is None
    scopes = reader_options(selected.reader, 'source')['acquisition_ids']
    provenance = ProvenanceStore(Path(__file__).resolve().parents[1] / 'provenance')
    roots = [*scopes.values(), *(item for ref in selected.metadata.references for item in ref.provenance_ids)]
    graph = validate_record_graph(provenance.read_record, roots)
    validate_reference_records(selected, graph.__getitem__, verify_all_claimed=True)
    contract = cfg.column_contract(dataset, year, reader_profile='source')
    assert contract is not None
    tables = crosswalk_dataset(parse_column_schema(contract['text'].encode()), dataset, str(year))['tables']
    for table, count in zip(tables, rows, strict=True):
        record = graph[scopes[table['table']]]
        assert (record['dataset'], record['resolved_year']) == (dataset, year)
        assert table_input_sha256s(table) == (record['artifact']['sha256'],)
        layout = table['layout']
        assert layout['last_data_record'] - layout['first_data_record'] + 1 == count
        if dataset == PROGRAMS:
            label = 'EMERGENT BILINGUAL/ENGLISH LEARNER' if year < 2025 else 'EMERGENT BILINGUAL'
            assert [f['code'] for f in table['fields'] if f['field'] == 'emergent_bilingual_count'] == [label]


@pytest.mark.parametrize(("year", "district_rows"), [(2023, 1209), (2024, 1206), (2025, 1206), (2026, 1201)])
def test_primary_disability_discovers_exact_source_only_years(year, district_rows):
    cfg = load_config()
    selected = cfg.resolve_dataset(PRIMARY_DISABILITY, year, exact=True, reader_profile="source")
    assert cfg.is_source_only(PRIMARY_DISABILITY, year)
    assert selected.metadata.published_at == ("2026-04-09" if year == 2026 else None)
    assert selected.metadata.period.label == f"{year - 1}-{year}"
    assert len(selected.metadata.metrics) == 13
    assert "special_education_program_count" not in selected.metadata.metrics
    assert {reference.role for reference in selected.metadata.references} == {
        "data_dictionary", "masking_methodology", "schema", "source_notes"}
    acquisitions = reader_options(selected.reader, "source")["acquisition_ids"]
    assert set(acquisitions) == {"state", "district"}
    provenance = ProvenanceStore(Path(__file__).resolve().parents[1] / "provenance")
    roots = list(acquisitions.values())
    roots.extend(identifier for reference in selected.metadata.references for identifier in reference.provenance_ids)
    graph = validate_record_graph(provenance.read_record, roots)
    validate_reference_records(selected, graph.__getitem__, verify_all_claimed=True)
    contract = cfg.column_contract(PRIMARY_DISABILITY, year, reader_profile="source")
    assert contract is not None
    tables = crosswalk_dataset(parse_column_schema(contract["text"].encode()), PRIMARY_DISABILITY, str(year))["tables"]
    for table, rows in zip(tables, (1, district_rows), strict=True):
        layout = table["layout"]
        assert layout["last_data_record"] - layout["first_data_record"] + 1 == rows
        receipt = graph[acquisitions[table["table"]]]
        assert receipt["resolved_year"] == year
        assert table_input_sha256s(table) == (receipt["artifact"]["sha256"],)
        assert len([field for field in table["fields"] if field["dictionary_status"] == "documented"]) == 13


@pytest.mark.parametrize(
    ("dataset", "raw", "status", "expected"),
    [
        (ENROLLMENT, "<2,790", ValueStatus.SUPPRESSED, None),
        (ENROLLMENT, "-999", ValueStatus.INVALID, None),
        (PROGRAMS, "-999", ValueStatus.SUPPRESSED, None),
        (PROGRAMS, "<10", ValueStatus.SUPPRESSED, None),
        (PROGRAMS, "-1", ValueStatus.INVALID, None),
        (PROGRAMS, "0", ValueStatus.OBSERVED, 0),
        (PROGRAMS, "", ValueStatus.MISSING, None),
        (PRIMARY_DISABILITY, "-999", ValueStatus.SUPPRESSED, None),
        (PRIMARY_DISABILITY, "0", ValueStatus.OBSERVED, 0),
        (PRIMARY_DISABILITY, "-1", ValueStatus.INVALID, None),
        (PRIMARY_DISABILITY, "<10", ValueStatus.INVALID, None),
    ],
)
def test_peims_masking_is_report_specific(dataset, raw, status, expected):
    cfg = load_config()
    source = cfg.resolve_dataset(dataset, 2026, exact=True)
    assert source.metadata.value_policy is not None
    value = measure(raw, cfg.value_policies[source.metadata.value_policy], unit="count", minimum=0)
    assert (value.status, value.value, value.raw) == (status, expected, raw)
