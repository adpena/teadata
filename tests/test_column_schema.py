from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import pytest

from teadata.column_schema import (
    header_records, header_rename_map, portable_schema_path, source_schema_fingerprint,
    table_header_identity, table_input_sha256s, validate_header_table,
)
from teadata.manifest import ContractError, canonical_json
from teadata.teadata_config import Config, load_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("case", ["unit_only", "explicit_unknown", "known_match", "field_conflict", "source_conflict", "unknown_table"])
def test_metric_period_uses_reviewed_table_only_without_a_period_override(case):
    from teadata.dataset_metadata import DataPeriod, DatasetMetadata, MetricDefinition

    table = input_table(input_sha256="a" * 64)
    table["fields"][0]["disposition"] = "identity"
    known = DataPeriod("school_year", 2015, 2016)
    table["metric_period"] = known.to_dict() if case != "unknown_table" else DataPeriod().to_dict()
    period = (DataPeriod() if case == "explicit_unknown" else
              known if case == "known_match" else
              DataPeriod("school_year", 2024, 2025) if case == "field_conflict" else None)
    metadata = DatasetMetadata(period=DataPeriod("school_year", 2024, 2025) if case == "source_conflict" else DataPeriod(),
                               metrics={"amount": MetricDefinition("value", unit="count", period=period)})
    document = {"schema_version": 2, "mapping_version": 1, "datasets": [
        {"dataset": "sample", "edition": "5", "source_only": True, "tables": [table]}]}
    if case in {"field_conflict", "source_conflict"}:
        with pytest.raises(ValueError, match="period"):
            header_records(document, dataset="sample", edition="5", metadata=metadata)
    else:
        identity, measure = header_records(document, dataset="sample", edition="5", metadata=metadata)
        assert identity["period"] == DataPeriod().to_dict()
        assert measure["period"] == (DataPeriod() if case in {"explicit_unknown", "unknown_table"} else known).to_dict()


def input_table(**inputs):
    labels, codes = ["Identifier", "Value"], ["ID", "VALUE"]
    return {"table": "statewide", **inputs,
        "source_schema_sha256": source_schema_fingerprint(labels, codes),
        "fields": [{"position": i, "code": code, "label": label, "field": field}
                   for i, (code, label, field) in enumerate(zip(codes, labels, ["identifier", "value"], strict=True), 1)]}


def test_position_identity_retains_literal_occurrences_and_refuses_ambiguous_rename():
    literal = "GF Capital Outlay (Object 61xx-64xx)"
    table = {"table": "actual", "input_sha256": "a" * 64, "header_identity": "position",
        "source_schema_sha256": source_schema_fingerprint([literal, literal], [literal, literal]),
        "fields": [{"position": index, "code": literal, "label": literal, "field": field}
            for index, field in enumerate(["capital_outlay_at_a", "capital_outlay_at_b"], 1)]}
    before = canonical_json(table)
    validate_header_table(table)
    assert table_header_identity(table) == "position" and canonical_json(table) == before
    document = {"schema_version": 2, "mapping_version": 1, "datasets": [{"dataset": "finance", "edition": "2024",
        "prepared_input_sha256": "b" * 64, "tables": [table]}]}
    rows = header_records(document, dataset="finance", edition="2024")
    assert [(row["position"], row["code"], row["header_identity"]) for row in rows] == [(1, literal, "position"), (2, literal, "position")]
    for source in ("code", "label"):
        with pytest.raises(ValueError, match="Ambiguous header"):
            header_rename_map(rows, from_=source)
    coded = deepcopy(table)
    coded.pop("header_identity")
    with pytest.raises(ValueError, match="Duplicate source code"):
        validate_header_table(coded)
    assert "header_identity" not in coded
    with pytest.raises(ValueError, match="requires code header identity"):
        validate_header_table(table, expected_identity="code")


@pytest.mark.parametrize("problem", ["unknown_mode", "duplicate_field", "bad_position", "synthetic_code", "coded_dictionary"])
def test_position_mode_does_not_relax_field_position_literal_or_dictionary_binding(problem):
    table = input_table(input_sha256="a" * 64)
    table["header_identity"] = "position"
    for field in table["fields"]:
        field["label"] = field["code"]
    table["source_schema_sha256"] = source_schema_fingerprint(["ID", "VALUE"], ["ID", "VALUE"])
    if problem == "unknown_mode":
        table["header_identity"] = "auto"
    elif problem == "duplicate_field":
        table["fields"][1]["field"] = "identifier"
    elif problem == "bad_position":
        table["fields"][1]["position"] = 1
    elif problem == "synthetic_code":
        table["fields"][1]["code"] = "VALUE_2"
    else:
        table["dictionary_records"] = []
    with pytest.raises(ValueError):
        validate_header_table(table)


@pytest.mark.parametrize("inputs", [{"input_sha256": "a" * 64}, {"input_sha256s": ["a" * 64, "b" * 64]}])
def test_one_input_view_preserves_original_wire_bytes_and_header_record_shape(inputs):
    table = input_table(**inputs)
    document = {"schema_version": 2, "mapping_version": 1, "datasets": [{"dataset": "sample",
        "edition": "2025", "prepared_input_sha256": "c" * 64, "tables": [table]}]}
    before = canonical_json(document)
    validate_header_table(table)
    expected = ("a" * 64,) if "input_sha256" in inputs else ("a" * 64, "b" * 64)
    assert table_input_sha256s(table) == expected
    rows = header_records(document, dataset="sample", edition="2025")
    assert canonical_json(document) == before
    if "input_sha256" in inputs:
        assert rows[0]["input_sha256"] == expected[0] and "input_sha256s" not in rows[0]
    else:
        assert rows[0]["input_sha256s"] == list(expected) and "input_sha256" not in rows[0]
        rows[0]["input_sha256s"].append("d" * 64)
        assert rows[1]["input_sha256s"] == list(expected)
        assert table_input_sha256s(table) == expected


@pytest.mark.parametrize("inputs", [
    {}, {"input_sha256": "a" * 64, "input_sha256s": ["a" * 64]},
    {"input_sha256": ["a" * 64]}, {"input_sha256s": "a" * 64}, {"input_sha256s": []},
    {"input_sha256s": ["a" * 64, "a" * 64]}, {"input_sha256s": ["a" * 64, True]},
    {"input_sha256s": ["a" * 64, "B" * 64]}, {"input_sha256s": ("a" * 64, "b" * 64)},
])
def test_reviewed_input_forms_reject_ambiguity_empty_duplicates_and_invalid_digests(inputs):
    with pytest.raises(ValueError):
        validate_header_table(input_table(**inputs))


@pytest.mark.parametrize("acquired", [["a" * 64], ["a" * 64, "c" * 64],
    ["a" * 64, "b" * 64, "c" * 64], ["b" * 64, "a" * 64]])
def test_multi_input_binding_requires_complete_ordered_acquisition_content_set(acquired):
    table = input_table(input_sha256s=["a" * 64, "b" * 64])
    with pytest.raises(ValueError, match="selected acquisition bytes/order"):
        table_input_sha256s(table, acquisition_sha256s=acquired)
    assert table_input_sha256s(table, acquisition_sha256s=["a" * 64, "b" * 64]) == ("a" * 64, "b" * 64)
    assert table_input_sha256s(input_table(input_sha256="a" * 64), acquisition_sha256s=acquired) == ("a" * 64,)


def test_multiple_inputs_cannot_share_one_physical_csv_layout():
    table = input_table(input_sha256s=["a" * 64, "b" * 64])
    table["layout"] = {"code_record": 1, "label_record": None, "first_data_record": 2,
                       "last_data_record": 2, "annotations": []}
    with pytest.raises(ValueError, match="physical CSV layout requires a single raw input"):
        validate_header_table(table)


def test_complete_2024_source_crosswalk_is_offline_and_keeps_legacy_projection(monkeypatch):
    cfg = load_config(ROOT / "teadata" / "teadata_sources.yaml")
    def forbidden(*args, **kwargs):
        pytest.fail("Header discovery read source facts")
    monkeypatch.setattr(cfg, "_load_table", forbidden)
    monkeypatch.setattr(cfg, "read_tapr_sources", forbidden)
    for scope, count, retained in [("campus", 407, 404), ("district", 420, 75)]:
        dataset = scope + "_tapr_student_staff_profile"
        records = cfg.header_records(dataset, edition="2024", reader_profile="source")
        assert len(records) == count
        assert len(cfg.column_schema(dataset, 2024)) == retained
        assert set(cfg.column_schema(dataset, 2024)) <= {record["field"] for record in records}
        identities = {record["field"] for record in records if record["disposition"] == "identity"}
        expected = {"district_number", "district_name"}
        if scope == "campus":
            expected |= {"campus_number", "campus_name"}
        assert identities == expected
        assert len(cfg.header_mapping(dataset, edition="2024", table="profile", reader_profile="source")) == count


def test_default_build_keeps_unselected_source_locators_portable_across_data_roots(tmp_path, monkeypatch):
    import yaml
    from teadata.dataset_metadata import DatasetMetadata, ResolvedDataset
    from teadata.manifest import BuildRecorder
    from teadata.provenance import ProvenanceStore

    _, raw, schema, _ = schema_config(tmp_path, reader={"profiles": {"source": {
        "tapr_source": {"scope": "campus", "tables": {"profile": "data/tapr/raw/original.csv"}}}}})
    raw["data_sources"]["sample"][2025]["path"] = "data/source.csv"
    store = ProvenanceStore(tmp_path / "provenance")
    schema_id = store.record_acquisition(schema, ResolvedDataset("reference:columns", "reference", None, None,
        str(schema), DatasetMetadata(title="Test columns")))
    raw["data_sources"]["sample"][2025]["metadata"]["references"][0]["provenance_ids"] = [schema_id]
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(raw))
    roots = [tmp_path / "root-a", tmp_path / "root-b"]
    for root in roots:
        root.mkdir()
        (root / "source.csv").write_text("alpha,beta\n1,2\n")
    monkeypatch.setenv("TEADATA_DATA_DIR", str(roots[0]))
    cfg = load_config(config)
    observed = cfg.resolve_dataset("sample", 2025)
    source_id = store.record_acquisition(observed.path, observed)
    raw["data_sources"]["sample"][2025]["metadata"]["provenance_ids"] = [source_id]
    config.write_text(yaml.safe_dump(raw))
    build_ids = []
    readers = []
    for root in roots:
        monkeypatch.setenv("TEADATA_DATA_DIR", str(root))
        cfg = load_config(config)
        with BuildRecorder(cfg, store) as recorder:
            cfg.load_df("sample", 2025)
        readers.append(recorder.datasets["sample"]["reader"])
        build_ids.append(recorder.catalog().build_id)
    assert build_ids[0] == build_ids[1]
    assert readers[0] == readers[1]
    assert readers[0]["profiles"]["source"]["tapr_source"]["tables"] == {"profile": "data/tapr/raw/original.csv"}
    assert not any(str(root) in canonical_json(readers[0]) for root in roots)


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
                profiles = list(entry.reader.get("profiles", {}))
                if entry.reader.get("column_schema") or not profiles:
                    profiles.insert(0, None)
                for profile in profiles:
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
                    assert document["schema_version"] in {1, 2}
                    paths.add(path)
                    count += 1
    assert count > 0
    assert paths == set((config_path.parent / "schemas").glob("*.json"))
