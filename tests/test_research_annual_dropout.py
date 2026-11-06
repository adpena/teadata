from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pyarrow as pa
import pytest
import yaml

from teadata import frc_ppe, research_graduation, tprs_students
from teadata.column_schema import reader_options
from teadata.column_schema import crosswalk_dataset, parse_column_schema, table_input_sha256s
from teadata.manifest import ContractError
from teadata.provenance import ProvenanceStore, validate_record_graph, validate_reference_records
from teadata.research_annual_dropout import (
    DATASET, KEYS, PERIOD, YEAR, _report_input, _report_table, _validate_keys, _validate_triples,
    materialize_research_annual_dropout,
)
from teadata.teadata_config import load_config


ROOT = Path(__file__).resolve().parents[1]
TRIPLE = ("denominator", "numerator", "rate")


@pytest.mark.parametrize(("dataset", "year", "producer"), [
    (DATASET, YEAR, materialize_research_annual_dropout),
    (frc_ppe.DATASET, frc_ppe.YEAR, frc_ppe.materialize_frc_ppe),
    (research_graduation.DATASET, research_graduation.YEAR, research_graduation.materialize_research_graduation),
    (tprs_students.DATASET, tprs_students.EDITION, tprs_students.materialize_students),
])
def test_receipt_only_producers_accept_relocated_contracts(tmp_path, monkeypatch, dataset, year, producer):
    packaged = ROOT / "teadata"
    source_config = (packaged / "teadata_sources.yaml").read_bytes()
    config = tmp_path / "teadata_sources.yaml"
    config.write_bytes(source_config)
    source = load_config().resolve_dataset(dataset, year, exact=True, reader_profile="source")
    schema = Path(reader_options(source.reader, "source")["column_schema"])
    (tmp_path / schema).parent.mkdir(parents=True)
    (tmp_path / schema).write_bytes((packaged / schema).read_bytes())
    store = ProvenanceStore(tmp_path / "provenance")

    def missing_retained_sources(_):
        raise FileNotFoundError("retained sources are required")

    monkeypatch.setattr(store, "verify_record_closure", missing_retained_sources)
    arguments = {"scopes": ["state"]}
    expected = "retained sources are required"
    if producer is tprs_students.materialize_students:
        arguments = {"intake": tmp_path / "intake.json", "intake_sha256": "0" * 64}
        expected = "intake.json"

    with pytest.raises(FileNotFoundError, match=expected):
        producer(store, config=config, config_sha256=sha256(source_config).hexdigest(),
                 output=tmp_path / "bundle", **arguments)
    changed = yaml.safe_load(source_config)
    changed["data_sources"][dataset][year]["metadata"]["title"] = "Unreviewed source title"
    config.write_text(yaml.safe_dump(changed))
    with pytest.raises(ContractError, match="packaged"):
        producer(store, config=config, config_sha256=sha256(config.read_bytes()).hexdigest(),
                 output=tmp_path / "bundle", **arguments)


def test_annual_dropout_opt_in_preserves_reviewed_raw_scope_contracts():
    config = load_config()
    source = config.resolve_dataset(DATASET, YEAR, exact=True, reader_profile="source")
    assert config.is_source_only(DATASET, YEAR) and config.column_schema(DATASET, YEAR) is None
    assert source.metadata.period == PERIOD and not source.metadata.metrics and source.metadata.value_policy is None
    resource = config.column_contract(DATASET, YEAR, reader_profile="source")
    assert resource is not None
    crosswalk = crosswalk_dataset(parse_column_schema(resource["text"].encode()), DATASET, str(YEAR))
    tables = {table["table"]: table for table in crosswalk["tables"]}
    assert set(tables) == set(KEYS)
    widths = {"state": 97, "region": 98, "county": 99, "district": 104, "campus": 106}
    input_counts = {"state": 3, "region": 60, "county": 1, "district": 1, "campus": 1}
    assert {scope: len(table["fields"]) for scope, table in tables.items()} == widths
    store = ProvenanceStore(ROOT / "provenance")
    options = source.reader["profiles"]["source"]
    for scope, table in tables.items():
        records = [store.read_record(key) for key in options["acquisition_ids"][scope]]
        assert len(records) == input_counts[scope]
        assert table["metric_period"] == PERIOD.to_dict()
        assert table_input_sha256s(table, acquisition_sha256s=[record["artifact"]["sha256"] for record in records])
        assert all(record["method"] == "download" and record["source_metadata"]["period"] == PERIOD.to_dict() for record in records)
        fields = {item["field"]: item for item in table["fields"]}
        assert set(KEYS[scope]) <= set(fields)
        assert ("calc_for_state_acct" in fields) == (scope in {"campus", "district"})
        prefix = {"state": "state_", "region": "region_", "campus": "camp_", "district": "dist_", "county": "cnty_"}[scope]
        assert fields[prefix + "fdmilr"] != fields[prefix + "milr"]
    refs = [rid for ref in source.metadata.references for rid in ref.provenance_ids]
    graph = validate_record_graph(store.read_record, refs)
    validate_reference_records(source, graph.__getitem__, verify_all_claimed=True)


def test_annual_dropout_triples_preserve_literal_masks_zero_and_bound():
    values = [("1,250", "10", "0.8"), ("20", "0", "0.0"), ("0", "0", "."), ("-1", "-1", "-1"),
              ("-3", "-3", "2.5"), ("-3", "-3", "<0.1"), ("2,000", "1", "<0.1"), ("16", "1", "6.3")]
    raw = pa.table(dict(zip(TRIPLE, zip(*values, strict=True), strict=True)))
    assert _validate_triples(raw, [TRIPLE]) == 3
    assert list(zip(*(raw[column].to_pylist() for column in TRIPLE), strict=True)) == values


@pytest.mark.parametrize(("values", "message"), [
    (("1000", "1", "<0.1"), "upper bound conflicts"),
    (("2,000", "0", "<0.1"), "upper bound conflicts"),
    (("<0.1", "0", "0.0"), "outside its published"),
    (("-3", "-1", "<0.1"), "mask disagree"),
    (("0", "0", "0.0"), "mask disagree"),
    (("-3", "-3", "100.1"), "exceeds 100"),
    (("16", "1", "6.2"), "differs from its observed"),
])
def test_annual_dropout_rejects_wrong_token_arithmetic_or_status(values, message):
    raw = pa.table({name: [value] for name, value in zip(TRIPLE, values, strict=True)})
    with pytest.raises(ContractError, match=message):
        _validate_triples(raw, [TRIPLE])


def test_annual_dropout_grain_allows_literal_federal_only_span_and_rejects_rebinding():
    columns = {"campus_number": ["001902001"] * 3, "district_number": ["001902"] * 3,
               "grade_span": ["78", "78", "712"], "calc_for_state_acct": ["No", "Yes", "No"]}
    _validate_keys(pa.table(columns), "campus")
    duplicate = {**columns, "grade_span": ["78", "78", "78"]}
    with pytest.raises(ContractError, match="grain is duplicated"):
        _validate_keys(pa.table(duplicate), "campus")
    with pytest.raises(ContractError, match="identities disagree"):
        _validate_keys(pa.table({**columns, "district_number": ["002902"] * 3}), "campus")
    with pytest.raises(ContractError, match="grade span"):
        _validate_keys(pa.table({**columns, "grade_span": ["7-8", "78", "712"]}), "campus")


def test_annual_dropout_report_group_and_grade_are_explicit():
    heading = ["Group", "Grade span", "Dropouts", "Students", "Rate (%)"]
    rows = [["Military-connected federal definition", "Grades 7-8", "1", "2,300", "<0.1"],
            ["", "Grades 9-12", "-3", "-3", "1.2"],
            ["Military-connected state definition", "Grades 7-8", "0", "400", "0.0"]]
    report = _report_table([heading, *rows], "prgm")
    assert report["Military-connected federal definition", "78"] == ("2,300", "1", "<0.1")
    assert report["Military-connected federal definition", "912"] == ("-3", "-3", "1.2")
    assert report["Military-connected state definition", "78"] == ("400", "0", "0.0")
    with pytest.raises(ContractError, match="group and grade must be unique"):
        _report_table([heading, *rows, rows[-1]], "prgm")


def test_annual_dropout_report_binds_post_controls_and_parent_before_file_read(monkeypatch, tmp_path):
    source = load_config().resolve_dataset(DATASET, YEAR, exact=True, reader_profile="source")
    receipt = source.reader["profiles"]["source"]["acquisition_ids"]["region"][0]
    selector = next(ref for ref in source.metadata.references if ref.reference_id == "region_selector").provenance_ids[0]
    store = ProvenanceStore(ROOT / "provenance")
    path = tmp_path / "reviewed"
    monkeypatch.setattr(store, "verify_artifact", lambda _: path)
    assert _report_input(store, receipt, "region", "01", "all", selector) == path
    changed = deepcopy(store.read_record(receipt))
    changed["parameters"]["form_controls"][-1][1] = "02"
    monkeypatch.setattr(store, "read_record", lambda _: changed)
    monkeypatch.setattr(store, "verify_artifact", lambda _: pytest.fail("Unbound report opened its body"))
    with pytest.raises(ContractError, match="request or selector lineage"):
        _report_input(store, receipt, "region", "01", "all", selector)
