"""An entity selector is independent of source row uniqueness and source column spelling."""
from dataclasses import replace
from io import BytesIO
from uuid import uuid4

import pyarrow as pa
from pyarrow.dataset import dataset
import pytest

from teadata import Campus, DataEngine
from teadata.dataset_metadata import ArtifactRef, DataPeriod, EntityRowBinding
from teadata.export_provenance import ExportProvenance
from teadata.manifest import ContractError, canonical_json
from teadata.query import Query
from teadata.release import DataRelease
from teadata.source_views import _entity_key, _entity_predicate

from test_catalog_projections import attach_observation_child, make_edition_fixture, selected_catalog  # noqa: F401
from test_source_series import use_children


BINDING = EntityRowBinding("publisher_entity", "reporting campus")


def fixture(tmp_path, *, year=2025, binding=BINDING):
    return make_edition_fixture(tmp_path, families=("student",), normalized=True,
        amount_period=DataPeriod("school_year", year - 1, year).to_dict(), entity_binding=binding)


@pytest.mark.parametrize("raw", [None, {}, {"column": "id"}, {"column": "id", "role": " "},
                                  {"column": False, "role": "reporter"},
                                  {"column": "id", "role": "reporter", "fallback": "other"}])
def test_row_binding_rejects_ambiguous_or_incomplete_contracts(raw):
    with pytest.raises(ValueError, match="binding"):
        EntityRowBinding.from_dict(raw)


def test_contract_keeps_row_keys_and_rejects_foreign_selector_column(tmp_path):
    data = fixture(tmp_path)
    _, ref, _ = data["leaves"]["student"]
    assert ref.contract["key_columns"] == ["source_record"]
    assert _entity_key(ref) == "publisher_entity" and ref.entity_binding == BINDING
    contract = ref.contract
    contract["entity_binding"]["column"] = "missing_column"
    with pytest.raises(ValueError, match="declared source column"):
        replace(ref, contract_json=canonical_json(contract))
    old = ref.contract
    old.pop("entity_binding")
    with pytest.raises(ContractError, match="declared entity row binding"):
        _entity_key(replace(ref, contract_json=canonical_json(old)))
    assert ArtifactRef.from_dict(ref.to_dict()) == ref


def test_noncanonical_district_column_uses_six_digits_and_does_not_select_zero():
    native = dataset(pa.table({"reporter": ["12345", "'012345", "012345", "'123456", None],
                               "known": [10, 11, 12, 99, 100]}))
    predicate = _entity_predicate(native, "reporter", ("012345",), width=6)
    assert native.to_table(filter=predicate)["known"].to_pylist() == [10, 11, 12]
    assert native.to_table(filter=_entity_predicate(native, "reporter", (), width=6)).num_rows == 0
    with pytest.raises(ValueError, match="Arrow strings"):
        _entity_predicate(dataset(pa.table({"reporter": [12345]})), "reporter", ("012345",), width=6)


def test_normal_namespace_dsl_and_replay_use_one_lazy_binding(tmp_path, request, monkeypatch):
    data = fixture(tmp_path)
    child = DataRelease(data["root"], offline=True)
    parent = attach_observation_child(request.getfixturevalue("selected_catalog"), child)
    engine = DataEngine()
    engine.catalog = parent.catalog
    campus = Campus(uuid4(), None, "Unjoined fixture", "", False, campus_number="004")
    engine.add_campus(campus)
    engine.bind_release(parent)
    with monkeypatch.context() as guard:
        guard.setattr(DataRelease, "artifact", lambda *a, **k: pytest.fail("Planning opened a data leaf"))
        namespace = campus.campus_student
        assert namespace.variables() == ("amount", "campus_name", "campus_number")
        planned = namespace.at(25).select("amount")
        assert planned.provenance().to_dict()["selector"]["entity_binding"] == BINDING.to_dict()
    assert namespace.amount == 5
    rows = Query([campus], engine).observations("campus_student", "amount").to_df()
    assert rows["amount"].tolist() == [5]
    assert rows["publisher_entity"].tolist() == ["004"]
    selected = planned.query("amount == @amount", params={"amount": 5}, record=True)
    provenance = selected.provenance()
    assert ExportProvenance.from_dict(provenance.to_dict()).replay_table(selected.release).to_arrow().equals(
        selected.to_arrow())
    assert planned._for_entity_ids(("999999999",)).to_arrow().num_rows == 0


def test_multi_year_reporting_cohort_exports_and_binding_breaks(tmp_path, monkeypatch):
    children = {}
    for year in (2024, 2025):
        path = tmp_path / str(year)
        path.mkdir()
        children[year] = DataRelease(fixture(path, year=year)["root"], offline=True)
    parent = children[2024]
    use_children(parent, children, monkeypatch)
    series = parent.series("campus_student", "amount", scope="campus", years=[24, 25], ids=[1, "'004"])
    expected = [(2024, 0, "0"), (2024, 5, "5"), (2025, 0, "0"), (2025, 5, "5")]
    rows = series.to_arrow().to_pylist()
    assert [(row["measurement_school_year"], row["amount"], row["VALUE"]) for row in rows] == expected
    for method in ("to_csv", "to_excel"):
        output = BytesIO()
        getattr(series, method)(output)
        assert output.getvalue()
    for source in series.provenance()["sources"]:
        assert source["provenance"]["selector"]["entity_binding"] == BINDING.to_dict()
    with pytest.raises(ContractError, match="one row per"):
        series.pair_years(from_year=24, to_year=25)
    foreign = tmp_path / "changed-role"
    foreign.mkdir()
    children[2025] = DataRelease(fixture(foreign, binding=EntityRowBinding("publisher_entity", "partner campus"))["root"], offline=True)
    with pytest.raises(ContractError, match="fields, types or publisher keys"):
        parent.series("campus_student", "amount", scope="campus", years=[24, 25], ids=[4])


def test_sql_coverage_and_readback_use_declared_noncanonical_entity_column(tmp_path, monkeypatch):
    import sqlite3

    from sqlalchemy import Column, Integer, JSON, MetaData, String, Table, Text, create_engine, select

    children = {}
    for year in (2024, 2025):
        path = tmp_path / str(year)
        path.mkdir()
        children[year] = DataRelease(fixture(path, year=year)["root"], offline=True)
    parent = children[2024]
    use_children(parent, children, monkeypatch)
    series = parent.series("campus_student", "amount", scope="campus", years=[24, 25], ids=[4, 999])
    metadata = MetaData()
    receipts = Table("receipts", metadata, Column("id", String(64), primary_key=True), Column("receipt", JSON))
    facts = Table("facts", metadata, Column("import_id", String(64), nullable=False))
    for field in series.schema:
        facts.append_column(Column(field.name, Integer if pa.types.is_integer(field.type) else Text))
    legacy = getattr(sqlite3, "LEGACY_TRANSACTION_CONTROL", None)
    sql = create_engine("sqlite://", connect_args={"autocommit": legacy} if legacy is not None else {})
    metadata.create_all(sql)
    with sql.begin() as connection:
        result = series.to_sql(connection, facts, receipts=receipts, require_complete=False)
    assert result.rows_by_year == {2024: 1, 2025: 1}
    assert result.missing_entity_years == ((2024, "000000999"), (2025, "000000999"))
    with sql.connect() as connection:
        record = connection.execute(select(receipts.c.receipt)).scalar_one()
        assert record["selection"]["sources"][0]["provenance"]["selector"]["entity_binding"] == BINDING.to_dict()
        restored = pa.Table.from_batches(list(series.iter_sql_batches(connection, facts, receipts=receipts,
                                                                     import_id=result.import_id)))
    assert restored.to_pylist() == series.to_arrow().to_pylist()
