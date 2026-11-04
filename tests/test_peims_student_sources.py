"""Published PEIMS source discovery without implicit hydration or year fallback."""

from pathlib import Path

import pytest

from teadata.column_schema import crosswalk_dataset, parse_column_schema
from teadata.manifest import ContractError
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
    assert {reference.role for reference in selected.metadata.references} == {"source_notes", "masking_methodology"}
    assert len(selected.metadata.provenance_ids) == 3
    assert cfg.column_schema(dataset, 2026) is None
    provenance = ProvenanceStore(Path(__file__).resolve().parents[1] / "provenance")
    roots = list(selected.metadata.provenance_ids)
    roots.extend(identifier for reference in selected.metadata.references for identifier in reference.provenance_ids)
    validate_record_graph(provenance.read_record, roots)
    validate_reference_records(selected, provenance.read_record, verify_all_claimed=True)

    # Retained builds can request an older year without attempting an
    # unavailable future-only source. Historical exact lookup still fails.
    for year in (2024, 2025, 2026, None):
        assert cfg.is_source_only(dataset, year)
    assert not cfg.is_source_only("accountability", 2025)
    with pytest.raises(KeyError):
        cfg.resolve_dataset(dataset, 2025, exact=True)
    with pytest.raises(ContractError, match="Reader profile is required"):
        cfg.load_df(dataset, 2026, exact=True)
    with pytest.raises(ContractError, match="requires a dedicated preparation workflow"):
        cfg.load_df(dataset, 2026, reader_profile="source", exact=True)

    contract = cfg.column_contract(dataset, 2026, reader_profile="source")
    selected_crosswalk = crosswalk_dataset(parse_column_schema(contract["text"].encode()), dataset, "2026")
    tables = selected_crosswalk["tables"]
    assert [table["table"] for table in tables] == ["state", "district", "campus"]
    assert len({table["input_sha256"] for table in tables}) == 3
    for table, width, row_count in zip(tables, widths, row_counts, strict=True):
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
        cfg.is_source_only(ENROLLMENT, 2024)

    # A future active edition also prevents the dataset from being classified
    # as exclusively source-only for an unavailable earlier edition.
    del editions[2025]
    editions[2027] = active
    assert not cfg.is_source_only(ENROLLMENT, 2027)
    with pytest.raises(KeyError, match="no prior fallback"):
        cfg.is_source_only(ENROLLMENT, 2024)


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
    ],
)
def test_peims_masking_is_report_specific(dataset, raw, status, expected):
    cfg = load_config()
    source = cfg.resolve_dataset(dataset, 2026, exact=True)
    value = measure(raw, cfg.value_policies[source.metadata.value_policy], unit="count", minimum=0)
    assert (value.status, value.value, value.raw) == (status, expected, raw)
