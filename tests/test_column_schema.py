from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import pytest

from teadata.column_schema import portable_schema_path
from teadata.manifest import ContractError, canonical_json
from teadata.teadata_config import Config, load_config


ROOT = Path(__file__).resolve().parents[1]


def schema_config(tmp_path, *, document=None, reader=None):
    directory = tmp_path / "schemas"
    directory.mkdir(exist_ok=True)
    schema = directory / "columns.json"
    schema.write_text(
        canonical_json(document or {"schema_version": 1, "columns": ["alpha", "beta"]})
        + "\n"
    )
    source = tmp_path / "source.csv"
    source.write_text("alpha,beta\n1,2\n")
    config = {
        "data_sources": {
            "sample": {
                2025: {
                    "path": str(source),
                    "metadata": {
                        "references": [
                            {
                                "reference_id": "columns",
                                "role": "schema",
                                "path": "schemas/columns.json",
                                "status": "available",
                                "reviewed_sha256": sha256(
                                    schema.read_bytes()
                                ).hexdigest(),
                            }
                        ],
                    },
                    "reader": {
                        "column_schema": "schemas/columns.json",
                        **(reader or {}),
                    },
                },
            },
        },
    }
    cfg = Config.from_dict(config)
    cfg.base_dir = tmp_path
    return cfg, config, schema, source


def test_schema_lookup_is_offline_ordered_and_defensive(tmp_path, monkeypatch):
    cfg, _, _, source = schema_config(tmp_path)
    source.unlink()
    monkeypatch.setattr(
        cfg, "_load_table", lambda *args, **kwargs: pytest.fail("Read source rows")
    )
    first = cfg.column_schema("sample", 2026)
    assert first == ["alpha", "beta"]
    first.append("injected")
    assert cfg.column_schema("sample") == ["alpha", "beta"]


def test_absent_optional_schema_returns_none(tmp_path):
    cfg = Config.from_dict(
        {"data_sources": {"sample": {2025: str(tmp_path / "absent.csv")}}}
    )
    assert cfg.column_schema("sample") is None


def test_load_df_validates_order_after_configured_renames(tmp_path):
    cfg, _, _, source = schema_config(
        tmp_path, reader={"columns": {"beta": "raw_beta"}}
    )
    source.write_text("alpha,raw_beta\n1,2\n")
    _, frame = cfg.load_df("sample")
    assert list(frame.columns) == ["alpha", "beta"]
    source.write_text("raw_beta,alpha\n2,1\n")
    with pytest.raises(ContractError, match="schema changed"):
        cfg.load_df("sample")


def test_changed_schema_bytes_fail_before_any_source_read(tmp_path, monkeypatch):
    cfg, _, schema, _ = schema_config(tmp_path)
    schema.write_text(
        canonical_json({"schema_version": 1, "columns": ["alpha", "renamed"]})
    )
    monkeypatch.setattr(
        cfg, "_load_table", lambda *args, **kwargs: pytest.fail("Read source rows")
    )
    with pytest.raises(ContractError, match="bytes differ"):
        cfg.column_schema("sample")
    with pytest.raises(ContractError, match="bytes differ"):
        cfg.load_df("sample")


@pytest.mark.parametrize(
    "document",
    [
        {"schema_version": 2, "columns": ["alpha"]},
        {"schema_version": True, "columns": ["alpha"]},
        {"schema_version": "1", "columns": ["alpha"]},
        {"schema_version": 1, "columns": "alpha"},
        {"schema_version": 1, "columns": ["alpha", 2]},
        {"schema_version": 1, "columns": ["alpha", " "]},
        {"schema_version": 1, "columns": ["alpha"], "rows": [[1]]},
    ],
)
def test_schema_resources_reject_invalid_documents(tmp_path, document):
    cfg, _, _, _ = schema_config(tmp_path, document=document)
    with pytest.raises(ContractError, match="Invalid column schema document"):
        cfg.column_schema("sample")


def test_duplicate_schema_columns_are_rejected(tmp_path):
    cfg, _, _, _ = schema_config(
        tmp_path, document={"schema_version": 1, "columns": ["alpha", "alpha"]}
    )
    with pytest.raises(ContractError, match="duplicate columns"):
        cfg.column_schema("sample")


def test_duplicate_json_keys_are_rejected_even_when_reviewed(tmp_path):
    _, config, schema, _ = schema_config(tmp_path)
    schema.write_text('{"schema_version":1,"columns":["alpha"],"columns":["beta"]}')
    config["data_sources"]["sample"][2025]["metadata"]["references"][0][
        "reviewed_sha256"
    ] = sha256(schema.read_bytes()).hexdigest()
    cfg = Config.from_dict(config)
    cfg.base_dir = tmp_path
    with pytest.raises(ContractError, match="duplicate JSON keys"):
        cfg.column_schema("sample")


@pytest.mark.parametrize(
    "relative",
    [
        "../outside.json",
        "/tmp/outside.json",
        "schemas/../../outside.json",
        "schemas/../columns.json",
        "C:/outside.json",
        "schemas\\columns.json",
        "./columns.json",
        "schemas//columns.json",
    ],
)
def test_schema_paths_must_be_portable_and_contained(tmp_path, relative):
    with pytest.raises(ValueError):
        portable_schema_path(tmp_path, relative)


def test_config_rejects_schema_symlink_escape(tmp_path):
    cfg, _, schema, _ = schema_config(tmp_path)
    outside = tmp_path.parent / (tmp_path.name + "-outside.json")
    outside.write_bytes(schema.read_bytes())
    schema.unlink()
    schema.symlink_to(outside)
    with pytest.raises(ContractError, match="escapes"):
        cfg.column_schema("sample")


@pytest.mark.parametrize(
    "change",
    [
        "missing_review",
        "unreviewed",
        "wrong_role",
        "no_reference",
        "ambiguous_reference",
    ],
)
def test_schema_requires_one_available_review_pin(tmp_path, change):
    _, config, _, _ = schema_config(tmp_path)
    references = config["data_sources"]["sample"][2025]["metadata"]["references"]
    if change == "missing_review":
        references[0].pop("reviewed_sha256")
    elif change == "unreviewed":
        references[0]["status"] = "unreviewed"
    elif change == "wrong_role":
        references[0]["role"] = "data_dictionary"
    elif change == "no_reference":
        references.clear()
    else:
        other = deepcopy(references[0])
        other["reference_id"] = "competing_columns"
        references.append(other)
    cfg = Config.from_dict(config)
    cfg.base_dir = tmp_path
    with pytest.raises(ContractError, match="one available byte-reviewed"):
        cfg.column_schema("sample")


def test_profiles_select_their_own_column_contracts(tmp_path):
    _, config, schema, source = schema_config(tmp_path)
    second = schema.with_name("second.json")
    second.write_text(canonical_json({"schema_version": 1, "columns": ["beta"]}))
    entry = config["data_sources"]["sample"][2025]
    entry["metadata"]["references"].append(
        {
            "reference_id": "columns_second",
            "role": "schema",
            "path": "schemas/second.json",
            "status": "available",
            "reviewed_sha256": sha256(second.read_bytes()).hexdigest(),
        }
    )
    entry["reader"]["profiles"] = {
        "second": {"column_schema": "schemas/second.json", "usecols": ["beta"]}
    }
    cfg = Config.from_dict(config)
    cfg.base_dir = tmp_path
    assert cfg.column_schema("sample") == ["alpha", "beta"]
    assert cfg.column_schema("sample", reader_profile="second") == ["beta"]
    assert list(cfg.load_df("sample", reader_profile="second")[1].columns) == ["beta"]
    with pytest.raises(ContractError, match="not declared"):
        cfg.column_schema("sample", reader_profile="missing")
    assert source.exists()


def test_profiles_without_a_default_schema_require_explicit_selection(tmp_path):
    _, config, _, _ = schema_config(tmp_path)
    reader = config["data_sources"]["sample"][2025]["reader"]
    path = reader.pop("column_schema")
    reader["profiles"] = {"named": {"column_schema": path}}
    cfg = Config.from_dict(config)
    cfg.base_dir = tmp_path
    assert cfg.column_schema("sample") is None
    with pytest.raises(ContractError, match="Reader profile is required"):
        cfg.load_df("sample")
    assert list(cfg.load_df("sample", reader_profile="named")[1].columns) == [
        "alpha",
        "beta",
    ]


@pytest.mark.parametrize("declared_schema", [True, False])
def test_duplicate_post_rename_columns_fail_closed(
    tmp_path, monkeypatch, declared_schema
):
    cfg, _, _, _ = schema_config(tmp_path, reader={"columns": {"alpha": "beta"}})
    if not declared_schema:
        cfg.data_sources["sample"].entries[2025].reader.pop("column_schema")
    monkeypatch.setattr(
        cfg,
        "_load_table",
        lambda *args, **kwargs: pd.DataFrame({"alpha": [1], "beta": [2]}),
    )
    with pytest.raises(ContractError, match="schema changed|duplicate_columns=True"):
        cfg.load_df("sample")


def test_retired_fingerprints_cannot_create_a_second_schema_authority(tmp_path):
    cfg, _, _, _ = schema_config(tmp_path, reader={"expected_columns_sha256": "a" * 64})
    with pytest.raises(ContractError, match="retired expected_columns_sha256"):
        cfg.column_schema("sample")
    with pytest.raises(ContractError, match="retired expected_columns_sha256"):
        cfg.load_df("sample")


def test_every_packaged_version_and_declared_profile_has_an_offline_schema(monkeypatch):
    config_path = ROOT / "teadata" / "teadata_sources.yaml"
    cfg = load_config(config_path)
    monkeypatch.setattr(
        cfg, "_load_table", lambda *args, **kwargs: pytest.fail("Read source rows")
    )
    count = 0
    paths = set()
    for section in ("data_sources", "spatial"):
        for dataset, year_map in getattr(cfg, section).items():
            for year, entry in year_map.entries.items():
                for profile in list(entry.reader.get("profiles", {})) or [None]:
                    columns = cfg.column_schema(
                        dataset, year, section=section, reader_profile=profile
                    )
                    assert columns and len(columns) == len(set(columns))
                    options = {
                        **entry.reader,
                        **entry.reader.get("profiles", {}).get(profile, {}),
                    }
                    assert "expected_columns_sha256" not in options
                    path = config_path.parent / options["column_schema"]
                    document = json.loads(path.read_bytes())
                    assert path.read_text() == canonical_json(document) + "\n"
                    assert set(document) == {"schema_version", "columns"}
                    paths.add(path)
                    count += 1
    assert count > 0
    assert paths == set((config_path.parent / "schemas").glob("*.json"))
