from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pytest

from teadata.column_schema import crosswalk_dataset, parse_column_schema, table_input_sha256s
from teadata.manifest import ContractError
from teadata.provenance import ProvenanceStore
from teadata.research_graduation import DATASET, POLICY, YEAR, _keys, _projection
from teadata.teadata_config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_research_contract_is_opt_in_and_pins_all_physical_fields_and_inputs():
    config = load_config()
    assert config.is_source_only(DATASET, YEAR)
    source = config.resolve_dataset(DATASET, YEAR, exact=True, reader_profile="source")
    assert source.metadata.value_policy == POLICY
    resource = config.column_contract(DATASET, YEAR, reader_profile="source")
    assert resource is not None
    contract = parse_column_schema(resource["text"].encode())
    tables = {row["table"]: row for row in crosswalk_dataset(contract, DATASET, str(YEAR))["tables"]}
    assert {name: len(table["fields"]) for name, table in tables.items()} == {"state": 12, "district": 404, "campus": 604}
    store = ProvenanceStore(ROOT / "provenance")
    options = source.reader["profiles"]["source"]
    for scope, table in tables.items():
        records = [store.read_record(key) for key in options["acquisition_ids"][scope]]
        assert table_input_sha256s(table) == tuple(item["artifact"]["sha256"] for item in records)
        assert {item["dataset"] for item in records} == {DATASET}
        assert {item["source_metadata"]["version"] for item in records} == {str(YEAR)}


def test_research_graduation_projection_distinguishes_masks_absence_and_zero(monkeypatch):
    config = load_config()
    source = config.resolve_dataset(DATASET, YEAR, exact=True, reader_profile="source")
    resource = config.column_contract(DATASET, YEAR, reader_profile="source")
    table, = [row for row in crosswalk_dataset(parse_column_schema(resource["text"].encode()), DATASET, str(YEAR))["tables"]
              if row["table"] == "campus"]
    fields = [field["code"] for field in table["fields"]]
    values = dict.fromkeys(fields, "0")
    for suffix in ("ALL", "ECN", "LEP", "SPE"):
        values[f"CAMP_{suffix}D"] = "9"
        values[f"CAMP_{suffix}N_GRAD"] = "0"
        values[f"CAMP_{suffix}R_GRAD"] = "0.0"
    values.update(CAMP_ECND="-3", CAMP_ECNN_GRAD="-1", CAMP_ECNR_GRAD="-1",
                  CAMP_LEPD="0", CAMP_LEPR_GRAD=".", CAMP_ALLN_GRAD_AEA="NA", CAMP_ALLR_CMP4="N/A")
    for field in ("CAMP_ALLN_CONT_AEA", "CAMP_ALLN_TXCHSE_AEA", "CAMP_ALLN_DROP_AEA", "CAMP_ALLN_CMP4"):
        values[field] = "NA"
    monkeypatch.setattr("teadata.research_graduation._keys", lambda *_: None)
    units = {metric.field: metric.unit for metric in source.metadata.metrics.values()}

    def project():
        batch = pa.record_batch([pa.array([values[field]]) for field in fields], names=fields)
        return _projection(batch, table, "campus", units=units, policy=config.value_policies[POLICY])[0].to_pylist()[0]

    row = project()
    assert (row["camp_alln_grad"], row["camp_allr_grad"], row["research_status_camp_allr_grad"]) == (0, Decimal("0.0"), "observed")
    assert (row["camp_ecnd"], row["research_raw_camp_ecnd"], row["research_status_camp_ecnd"]) == (None, "-3", "suppressed")
    assert (row["camp_lepd"], row["camp_lepr_grad"], row["research_status_camp_lepr_grad"]) == (0, None, "not_applicable")
    assert (row["camp_alln_grad_aea"], row["research_raw_camp_alln_grad_aea"], row["research_status_camp_alln_grad_aea"]) == (
        None, "NA", "not_applicable")
    assert (row["camp_allr_cmp4"], row["research_raw_camp_allr_cmp4"]) == (None, "N/A")
    values["CAMP_ALLR_GRAD"] = "0.1"
    with pytest.raises(ContractError, match="differs from its observed cohort counts"):
        project()
    values["CAMP_ALLR_GRAD"] = "NA"
    with pytest.raises(ContractError, match="Unreviewed research graduation token"):
        project()
    values.update(CAMP_ALLR_GRAD="0.0", CAMP_ALLN_GRAD_AEA="1", CALC_FOR_STATE_ACCT="No", AEA="Yes")
    with pytest.raises(ContractError, match="outside the statutory AEA rows"):
        project()


def test_duplicate_or_disagreeing_campus_identity_cannot_pass():
    columns = {"district_number": pa.array(["001902", "001902"]), "campus_number": pa.array(["001902001", "001902001"]),
               "calc_for_state_acct": pa.array(["No", "No"]), "aea": pa.array(["No", "No"])}
    with pytest.raises(ContractError, match="entity-method grain"):
        _keys(columns, "campus")
    columns["campus_number"] = pa.array(["001902001", "002902001"])
    with pytest.raises(ContractError, match="literal district prefix"):
        _keys(columns, "campus")
