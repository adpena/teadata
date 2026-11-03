from dataclasses import FrozenInstanceError
from datetime import date
from hashlib import sha256
import json
from pathlib import Path

import pytest

from teadata.dataset_metadata import DataPeriod, DatasetMetadata, MetricDefinition, SourceQualifiers
from teadata.metric_bindings import MetricBinding
from teadata.teadata_config import Config, load_config
from teadata.values import MeasurementUnit


@pytest.mark.parametrize("unit", [None, *MeasurementUnit])
def test_metric_metadata_and_bindings_share_declared_units_with_string_roundtrip(unit):
    literal = unit.value if unit is not None else None
    definition = MetricDefinition.from_dict({"field": "reported_value", "unit": unit})
    binding = MetricBinding.from_dict({"dataset": "source", "field": "reported_value", "unit": unit})
    assert definition.to_dict() == {"field": "reported_value", "unit": literal}
    assert definition.unit is None or type(definition.unit) is str
    assert binding.unit is None or type(binding.unit) is str
    assert json.loads(json.dumps(definition.to_dict()))["unit"] == literal
    assert json.loads(json.dumps(binding.to_dict()))["unit"] == literal
    assert MetricDefinition.from_dict(definition.to_dict()) == definition
    assert MetricBinding.from_dict(binding.to_dict()) == binding


@pytest.mark.parametrize("unit", ["FTE", "headcount", "fraction", 12, True, [], {}])
def test_unknown_declared_units_are_rejected_by_both_metadata_consumers(unit):
    with pytest.raises(ValueError, match="unit"):
        MetricDefinition.from_dict({"field": "reported_value", "unit": unit})
    with pytest.raises(ValueError, match="unit"):
        MetricBinding.from_dict({"dataset": "source", "field": "reported_value", "unit": unit})


def test_metric_suffix_and_bounds_validate_through_the_same_fte_unit_authority():
    document = {"dataset": "source", "field": "reported_value", "unit_suffixes": {"_fte": "fte"},
                "unit_bounds": {"fte": {"minimum": 0, "maximum": None}}}
    binding = MetricBinding.from_dict(document)
    assert binding.bounds(MeasurementUnit.FTE) == (0, None)
    assert MetricBinding.from_dict(binding.to_dict()) == binding
    with pytest.raises(ValueError, match="suffix"):
        MetricBinding.from_dict({**document, "unit_suffixes": {"_fte": "unknown"}})
    with pytest.raises(ValueError, match="bounds"):
        MetricBinding.from_dict({**document, "unit_bounds": {"unknown": {"minimum": 0}}})


def test_absent_source_qualifiers_preserve_legacy_metadata_receipt_and_semantics_bytes():
    from teadata.manifest import _leaf_semantics, canonical_json

    expected = '{"identifiers":[],"metrics":{},"period":{"kind":"unknown"},"provenance_ids":[],"references":[],"version":"2025"}'
    for raw in ({"version": "2025"}, {"version": "2025", "source_qualifiers": None}):
        metadata = DatasetMetadata.from_dict(raw)
        assert metadata.source_qualifiers is None
        assert canonical_json(metadata.to_dict()) == expected
        receipt = {"source_metadata": metadata.to_dict()}
        assert canonical_json(receipt) == '{"source_metadata":' + expected + '}'
        old_semantics = b'{"estimation":null,"metrics":{},"periods":{"value":{"kind":"unknown"}}}'
        assert _leaf_semantics(metadata, [{"field": "value", "period": {"kind": "unknown"}}]) == sha256(old_semantics).hexdigest()


@pytest.mark.parametrize("qualifiers", [
    {"family": "sof", "publication_run_id": "046820", "row_run_column": "SOF_RUN_ID", "publisher_release_status": "Near Final (revised)"},
    {"family": "peims_financial", "financial_basis": "actual"},
    {"family": "peims_financial", "financial_basis": "budget"},
    {"family": "accountability", "publisher_release_status": "Final resubmission"},
])
def test_source_qualifiers_are_typed_frozen_and_preserve_publisher_wording(qualifiers):
    raw = {"version": "separate edition identity", "source_qualifiers": {**qualifiers, "reference_id": "publication"},
        "references": [{"reference_id": "publication", "role": "source_notes", "status": "available",
            "url": "https://example.test/publication", "reviewed_sha256": "a" * 64, "provenance_ids": ["b" * 64]}]}
    metadata = DatasetMetadata.from_dict(raw)
    assert isinstance(metadata.source_qualifiers, SourceQualifiers)
    assert metadata.source_qualifiers.to_dict() == raw["source_qualifiers"]
    assert DatasetMetadata.from_dict(metadata.to_dict()) == metadata
    assert metadata.version == "separate edition identity"
    with pytest.raises(FrozenInstanceError):
        setattr(metadata.source_qualifiers, "publisher_release_status", "changed by caller")


@pytest.mark.parametrize("family", ["sof", "peims_financial", "accountability"])
def test_unknown_source_qualifiers_do_not_infer_defaults(family):
    metadata = DatasetMetadata.from_dict({"version": "46820", "source_qualifiers": {"family": family}})
    assert metadata.source_qualifiers is not None
    assert metadata.source_qualifiers.to_dict() == {"family": family}
    assert metadata.source_qualifiers.publisher_release_status is None
    assert metadata.source_qualifiers.publication_run_id is None
    assert metadata.source_qualifiers.financial_basis is None


@pytest.mark.parametrize("qualifiers", [
    [], {"family": []}, {"family": "unknown"}, {"family": "sof", "latest": True},
    {"family": "sof", "financial_basis": None},
    {"family": "peims_financial", "publication_run_id": "46820"},
    {"family": "accountability", "row_run_column": None},
    {"family": "sof", "publication_run_id": True},
    {"family": "sof", "publication_run_id": 46820},
    {"family": "sof", "row_run_column": " "},
    {"family": "sof", "publisher_release_status": ""},
    {"family": "peims_financial", "financial_basis": "estimated"},
    {"family": "accountability", "publisher_release_status": False},
])
def test_source_qualifier_unknown_fields_types_and_family_mismatches_fail(qualifiers):
    with pytest.raises(ValueError):
        SourceQualifiers.from_dict(qualifiers)


def test_source_qualifier_claims_need_a_declared_evidence_reference():
    with pytest.raises(ValueError, match="evidence reference"):
        SourceQualifiers.from_dict({"family": "sof", "publication_run_id": "46820"})
    with pytest.raises(ValueError, match="existing evidence reference"):
        DatasetMetadata.from_dict({"source_qualifiers": {
            "family": "sof", "publication_run_id": "46820", "reference_id": "not-declared"}})


def profile_metadata_document(path):
    return {"value_policies": {"legacy": {"reference_id": "masking", "tokens": {
        "-1": {"status": "suppressed", "reason": "legacy interpretation"}}}},
        "data_sources": {"example": {2024: {"path": str(path), "metadata": {
            "version": "2024", "title": "Legacy view", "period": {"kind": "calendar_year", "year": 2023},
            "value_policy": "legacy", "references": [{"reference_id": "masking", "role": "masking_methodology", "status": "unreviewed"}],
            "metrics": {"old": {"field": "value", "unit": "count"}}},
            "reader": {"dtype": "str", "keep_default_na": False,
                "profiles": {"source": {"metadata": {"version": "2024"}}}}}}}}


def test_selected_profile_metadata_is_complete_uninterpreted_and_does_not_mutate_base(tmp_path):
    path = tmp_path / "values.csv"
    path.write_text("value\n-1\n.\n001\n")
    document = profile_metadata_document(path)
    cfg = Config.from_dict(document)
    before = cfg.resolve_dataset("example", 2024).to_dict()
    serialized = cfg.to_json()
    raw = cfg.resolve_dataset("example", 2024, reader_profile="source")
    assert raw.metadata.version == "2024" and raw.metadata.title is None
    assert raw.metadata.period.kind == "unknown" and raw.metadata.value_policy is None
    assert not raw.metadata.metrics and raw.metadata.estimation is None and not raw.metadata.references
    assert cfg.reader_references("example", 2024, reader_profile="source") == []
    assert cfg.resolve_dataset("example", 2024).to_dict() == before
    assert cfg.to_json() == serialized
    year, frame = cfg.load_df("example", 2024, reader_profile="source")
    assert year == 2024 and frame["value"].tolist() == ["-1", ".", "001"]
    assert cfg.resolve_dataset("example", 2024).metadata.value_policy == "legacy"


@pytest.mark.parametrize("damage", ["null", "list", "unknown_field", "different_version", "missing_version", "unknown_policy", "extra_references"])
def test_profile_metadata_invalid_or_competing_authorities_fail_before_file_io(tmp_path, monkeypatch, damage):
    document = profile_metadata_document(tmp_path / "not-read.csv")
    profile = document["data_sources"]["example"][2024]["reader"]["profiles"]["source"]
    if damage == "null":
        profile["metadata"] = None
    elif damage == "list":
        profile["metadata"] = []
    elif damage == "unknown_field":
        profile["metadata"]["raw_policy_bypass"] = True
    elif damage == "different_version":
        profile["metadata"]["version"] = "2023"
    elif damage == "missing_version":
        profile["metadata"].clear()
    elif damage == "unknown_policy":
        profile["metadata"]["value_policy"] = "not-declared"
    else:
        profile["references"] = []
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("Invalid profile touched source file"))
    with pytest.raises(ValueError):
        Config.from_dict(document)


@pytest.mark.parametrize("has_profiles", [False, True])
@pytest.mark.parametrize("method,message", [
    ("column_contract", "Invalid column schema for example"),
    ("column_schema", "Invalid column schema for example"),
    ("load_df", "Invalid reader contract for example"),
    ("header_records", "Invalid column schema for example"),
    ("header_mapping", "Invalid column schema for example"),
])
def test_missing_profile_preserves_public_contract_error_before_io(tmp_path, monkeypatch, has_profiles, method, message):
    from teadata.manifest import ContractError

    document = profile_metadata_document(tmp_path / "not-read.csv")
    if not has_profiles:
        document["data_sources"]["example"][2024]["reader"].pop("profiles")
    cfg = Config.from_dict(document)
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("Missing profile opened an artifact"))
    arguments: dict[str, str | int] = {"edition": "2024"} if method.startswith("header_") else {"year": 2024}
    if method == "header_mapping":
        arguments["table"] = "example"
    with pytest.raises(ContractError, match=message):
        getattr(cfg, method)("example", reader_profile="absent", **arguments)
    # The direct metadata APIs retain their original ValueError boundary.
    with pytest.raises(ValueError, match="Reader profile absent is not declared"):
        cfg.resolve_dataset("example", 2024, reader_profile="absent")
    with pytest.raises(ValueError, match="Reader profile absent is not declared"):
        cfg.reader_references("example", 2024, reader_profile="absent")


@pytest.mark.parametrize("method", ["column_contract", "column_schema", "load_df"])
def test_reader_boundaries_preserve_legacy_base_resolution_errors(tmp_path, method):
    cfg = Config.from_dict(profile_metadata_document(tmp_path / "not-read.csv"))
    for profile in (None, "source"):
        with pytest.raises(ValueError, match="section must be"):
            getattr(cfg, method)("example", 2024, section="unknown", reader_profile=profile)
        with pytest.raises(KeyError, match="Unknown data_sources dataset"):
            getattr(cfg, method)("absent", 2024, reader_profile=profile)


def test_generic_profile_dataframe_build_uses_same_selected_metadata(tmp_path):
    from teadata.manifest import BuildRecorder
    from teadata.provenance import ProvenanceStore
    path = tmp_path / "values.csv"
    path.write_text("value\n-1\n001\n")
    document = profile_metadata_document(path)
    store = ProvenanceStore(tmp_path / "provenance")
    cfg = Config.from_dict(document)
    identifier = store.record_acquisition(path, cfg.resolve_dataset("example", 2024, reader_profile="source"))
    document["data_sources"]["example"][2024]["reader"]["profiles"]["source"]["metadata"]["provenance_ids"] = [identifier]
    cfg = Config.from_dict(document)
    with BuildRecorder(cfg, store) as recorder:
        _, frame = cfg.load_df("example", 2024, reader_profile="source")
    assert frame["value"].tolist() == ["-1", "001"]
    assert "value_policy" not in recorder.datasets["example"]["metadata"]
    assert recorder.datasets["example"]["metadata"]["period"] == {"kind": "unknown"}
    assert cfg.resolve_dataset("example", 2024).metadata.value_policy == "legacy"


def test_selected_applied_policy_still_fails_shared_methodology_gate_before_io(tmp_path, monkeypatch):
    from teadata.manifest import BuildRecorder, ContractError
    from teadata.provenance import ProvenanceStore

    document = profile_metadata_document(tmp_path / "not-read.csv")
    entry = document["data_sources"]["example"][2024]
    entry["reader"]["profiles"]["source"]["metadata"] = dict(entry["metadata"])
    cfg = Config.from_dict(document)
    recorder = BuildRecorder(cfg, ProvenanceStore(tmp_path / "history"))
    monkeypatch.setattr("teadata.manifest.verify_source_receipts", lambda *a, **k: pytest.fail("Unreviewed policy touched an artifact"))
    monkeypatch.setattr(cfg, "_load_table", lambda *a, **k: pytest.fail("Unreviewed policy loaded source rows"))
    with recorder:
        with pytest.raises(ContractError, match="retained reviewed methodology"):
            cfg.load_df("example", 2024, reader_profile="source")


def test_legacy_paths_and_load_df_keep_their_existing_contract(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("value\n42\n")
    cfg = Config.from_dict(
        {"data_sources": {"example": {2024: str(source), "latest": 2024}}}
    )
    assert cfg.resolve("example", 2025) == (2024, str(source))
    year, frame = cfg.load_df("example", 2025)
    assert year == 2024 and frame.loc[0, "value"] == 42
    described = cfg.resolve_dataset("example", 2025)
    assert described.is_fallback is True
    assert described.metadata.period.kind == "unknown"
    assert described.metadata.period.label is None
    assert json.loads(cfg.to_json())["data_sources"]["example"]["2024"]["path"] == str(source)


@pytest.mark.parametrize("suffix", ["yaml", "toml"])
def test_rich_source_metadata_loads_from_both_config_formats(tmp_path, suffix):
    yaml_text = """data_sources:
  example:
    2024:
      path: data/example.csv
      metadata:
        title: Research dataset
        publisher: Example publisher
        source_url: https://example.org/catalog
        download_url: https://example.org/data.csv
        published_at: 2025-01-15
        notes: data/notes-is-not-a-path
        period:
          kind: school_year
          start_year: 2023
          end_year: 2024
"""
    toml_text = """[data_sources.example.2024]
path = "data/example.csv"
[data_sources.example.2024.metadata]
title = "Research dataset"
publisher = "Example publisher"
source_url = "https://example.org/catalog"
download_url = "https://example.org/data.csv"
published_at = 2025-01-15
notes = "data/notes-is-not-a-path"
[data_sources.example.2024.metadata.period]
kind = "school_year"
start_year = 2023
end_year = 2024
"""
    path = tmp_path / f"sources.{suffix}"
    path.write_text(yaml_text if suffix == "yaml" else toml_text)
    cfg = load_config(path)
    source = cfg.resolve_dataset("example", 2025)
    assert source.path == str(tmp_path / "data" / "example.csv")
    assert source.resolved_year == 2024
    assert source.metadata.period.label == "2023-2024"
    assert source.metadata.published_at == "2025-01-15"
    assert source.metadata.notes == "data/notes-is-not-a-path"
    assert source.metadata.source_url == "https://example.org/catalog"
    round_trip = Config.from_dict(json.loads(cfg.to_json()))
    assert round_trip.resolve_dataset("example", 2025).to_dict() == source.to_dict()


def test_school_year_is_explicit_and_not_inferred_from_catalog_key():
    cfg = Config.from_dict(
        {
            "data_sources": {
                "baseline": {
                    2014: {
                        "path": "baseline.csv",
                        "metadata": {
                            "period": {
                                "kind": "school_year",
                                "start_year": 2014,
                                "end_year": 2015,
                            }
                        },
                    }
                }
            }
        }
    )
    source = cfg.resolve_dataset("baseline", 2025)
    assert source.resolved_year == 2014
    assert source.metadata.period.label == "2014-2015"


def test_floating_current_source_has_no_invented_catalog_year():
    cfg = Config.from_dict(
        {
            "data_sources": {
                "directory": {
                    "current": {
                        "path": "https://example.org/directory.csv",
                        "metadata": {
                            "period": {"kind": "as_of", "as_of": date(2025, 9, 30)},
                        },
                    }
                }
            }
        }
    )
    source = cfg.resolve_dataset("directory", 2026)
    assert source.resolved_year is None
    assert source.is_fallback is None
    assert source.metadata.period.label == "2025-09-30"
    assert cfg.resolve("directory", 2026)[0] == 9999  # legacy API remains compatible


def test_legacy_current_url_and_latest_year_alias_remain_supported():
    cfg = Config.from_dict(
        {
            "data_sources": {
                "directory": {"current": "https://example.org/current.csv"},
                "annual": {
                    "latest": 2025,
                    2025: {"path": "annual.csv", "metadata": {"title": "Annual"}},
                },
            }
        }
    )
    assert cfg.resolve_dataset("directory", 2026).resolved_year is None
    assert cfg.resolve_dataset("annual", 2026).metadata.title == "Annual"


@pytest.mark.parametrize(
    "raw",
    [
        {"period": {"kind": "school_year", "start_year": 2024, "end_year": 2026}},
        {"period": {"kind": "calendar_year", "year": -1}},
        {"period": {"kind": "school_year", "start_year": True, "end_year": 2}},
        {"period": {"kind": "unknown", "year": 2025}},
        {"period": {"kind": "as_of", "as_of": "2025-02-30"}},
        {"published_at": "2025-01-01T12:00:00"},
        {"download_url": "file:source.csv"},
        {"download_url": "https://user:password@example.org/file.csv"},
        {"publisher": 42},
        {"provenance_ids": ["invalid"]},
        {"identifiers": [1]},
        {"downloaded_at": "2025-01-01"},  # observations belong in immutable receipts
    ],
)
def test_invalid_or_misplaced_metadata_is_rejected(raw):
    with pytest.raises(ValueError):
        DatasetMetadata.from_dict(raw)


def test_calendar_and_fiscal_periods_have_distinct_kinds():
    assert DataPeriod(kind="calendar_year", year=2025).label == "2025"
    assert (
        DataPeriod(kind="fiscal_year", start_year=2023, end_year=2024).label
        == "2023-2024"
    )


def test_structured_paths_keep_file_type_validation():
    with pytest.raises(ValueError, match="File-type validation"):
        Config.from_dict(
            {
                "data_sources": {"example": {2025: {"path": "wrong.xlsx"}}},
                "schema": {"data_sources": {"example": ["csv"]}},
            }
        )


def test_describe_cli_exposes_metadata_without_loading_the_data(tmp_path, capsys):
    from teadata.teadata_config import main

    path = tmp_path / "sources.yaml"
    path.write_text("data_sources:\n  example:\n    2024: missing.csv\n")
    main(["teadata-config", "describe", str(path), "data_sources", "example", "2025"])
    result = json.loads(capsys.readouterr().out)
    assert result["resolved_year"] == 2024
    assert result["metadata"]["period"]["kind"] == "unknown"


def test_existing_repository_configuration_does_not_claim_download_dates():
    cfg = load_config(
        Path(__file__).resolve().parents[1] / "teadata" / "teadata_sources.yaml"
    )
    for name in cfg.data_sources:
        source = cfg.resolve_dataset(name, 2025)
        assert "downloaded_at" not in source.metadata.to_dict()
