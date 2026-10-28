from datetime import date
import json
from pathlib import Path

import pytest

from teadata.dataset_metadata import DataPeriod, DatasetMetadata
from teadata.teadata_config import Config, load_config


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
