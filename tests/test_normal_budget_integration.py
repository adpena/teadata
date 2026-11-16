"""Opt-in checks of the reviewed budget family through an actual normal parent."""

from base64 import b64decode
import csv
from hashlib import sha256
from io import BytesIO, StringIO
import json
import os
from pathlib import Path

from openpyxl import load_workbook
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from sqlalchemy import BigInteger, Column, JSON, MetaData, String, Table, Text, create_engine, select

from teadata import DataEngine
from teadata.manifest import ContractError, canonical_json
from teadata.release import DataRelease, using_release


DATASET = "peims_financial_single_file_budgets"
BUDGET_PIN = "bcff73e52199f4d8e527db2d66a89f8ee3a825d3734e7902e16ab4f7824e7f38"
BASELINE_PIN = "803a0ae418b724167b320f8488284fb9f25a001bb04e6066e6f4299d496dc0d5"
FISCAL_PERIOD = {"kind": "fiscal_year", "start_year": 2025, "end_year": 2026}


@pytest.fixture(scope="module")
def normal_budget():
    names = ("TEADATA_NORMAL_BUDGET_RELEASE", "TEADATA_NORMAL_BUDGET_PIN", "TEADATA_NORMAL_BUDGET_BASELINE")
    values = {name: os.environ.get(name) for name in names}
    if not all(values.values()):
        pytest.skip("Actual retained normal-budget parent, pin and baseline are explicit opt-ins")
    parent = DataRelease(values[names[0]], expected_sha256=values[names[1]], offline=True)
    prior = DataRelease(values[names[2]], expected_sha256=BASELINE_PIN, offline=True)
    budget, = (ref for ref in parent.catalog.native_sources if ref.manifest_sha256 == BUDGET_PIN)
    keys = {key.table: key for key in budget.editions}
    assert set(keys) == {"district", "charter"}
    with using_release(parent):
        engine = DataEngine.load_default(indexes_enabled=False)
    return parent, prior, budget, keys, engine


@pytest.fixture(scope="module")
def budget_views(normal_budget):
    parent, _, _, keys, _ = normal_budget
    result = {}
    for population, key in keys.items():
        view = parent.table(key).select("budget_amount").query(
            "source_record >= @start and source_record < @stop", params={"start": 0, "stop": 256}, record=True,
        )
        table = view.to_arrow()
        assert 0 < table.num_rows <= 256
        assert table.schema.field("budget_amount").type == pa.int64()
        result[population] = view, table
    return result


def test_normal_budget_preserves_all_base_selections_and_prior_children(normal_budget):
    parent, prior, budget, keys, engine = normal_budget
    assert parent.catalog.to_dict()["datasets"] == prior.catalog.to_dict()["datasets"]
    assert len(parent.catalog.to_dict()["datasets"]) == 13
    assert [ref.to_dict() for ref in parent.catalog.native_sources[:-1]] == [
        ref.to_dict() for ref in prior.catalog.native_sources
    ]
    assert parent.catalog.native_sources[-1] == budget
    assert len(parent.catalog.native_sources) == 17
    assert len({key.dataset for ref in parent.catalog.native_sources for key in ref.editions}) == 22
    assert sum(len(ref.editions) for ref in parent.catalog.native_sources) == 206
    assert (len(engine.districts), len(engine.campuses)) == (1229, 12593)
    assert all(key.dataset == DATASET and key.scope == "district" and key.edition == "2026" for key in keys.values())


@pytest.mark.parametrize("population", ["district", "charter"])
def test_budget_exact_entity_namespace_keeps_parent_provenance(normal_budget, budget_views, population):
    parent, _, _, keys, engine = normal_budget
    view, table = budget_views[population]
    identifier = table["district_number"][0].as_py().lstrip("'")
    district, = engine >> ("district", identifier)
    original_meta, original_enrollment = dict(district.meta), district.enrollment
    sources = parent.for_entity(district, edition="2026")
    with pytest.raises(ContractError, match="ambiguous"):
        sources.dataset(DATASET)
    bound = sources.dataset(keys[population])
    selected = bound.select("budget_amount", where=pc.field("source_record") < 256)
    assert selected.num_rows > 0
    provenance = json.loads(selected.schema.metadata[b"teadata_provenance"])
    assert provenance["selector"]["entity_release"] == {
        "build_id": parent.build_id, "manifest_sha256": parent.manifest_sha256,
    }
    assert provenance["selector"]["entity_ids"] == [identifier]
    assert provenance["release"]["manifest_sha256"] == BUDGET_PIN
    assert provenance["source"]["report_period"] == FISCAL_PERIOD
    with pytest.raises(ContractError, match="Multiple published rows"):
        _ = bound.budget_amount
    with pytest.raises(ContractError, match="School-year selection"):
        district.dataset(DATASET).select("budget_amount", school_year=2026)
    assert district.meta == original_meta and district.enrollment == original_enrollment
    assert view.provenance().to_dict()["selector"]["entity_release"]["manifest_sha256"] == parent.manifest_sha256


def test_budget_standalone_context_stays_standalone(normal_budget):
    parent, _, budget, keys, _ = normal_budget
    child = parent.table(keys["district"]).release
    assert child.manifest_sha256 == budget.manifest_sha256
    origin = child.table(keys["district"]).select("budget_amount").provenance().to_dict()
    assert origin["selector"]["entity_release"] is None


@pytest.mark.parametrize("population", ["district", "charter"])
def test_budget_bounded_dataframe_csv_excel_and_sqlite_roundtrip(budget_views, tmp_path, population):
    view, table = budget_views[population]
    origin = view.provenance().to_dict()
    assert origin["source"]["report_period"] == FISCAL_PERIOD
    assert table.schema.metadata[b"teadata_provenance"] == canonical_json(origin).encode()
    frame = view.to_df()
    assert frame.attrs["teadata_provenance"] == origin
    assert frame["budget_amount"].tolist() == table["budget_amount"].to_pylist()
    assert all(int(raw) == value for raw, value in zip(
        table["raw_budget_amount"].to_pylist(), table["budget_amount"].to_pylist(), strict=True,
    ))

    output = BytesIO()
    view.to_csv(output)
    csv_rows = list(csv.DictReader(StringIO(output.getvalue().decode())))
    assert len(csv_rows) == table.num_rows
    assert [int(row["budget_amount"]) for row in csv_rows] == table["budget_amount"].to_pylist()
    assert [row["raw_budget_amount"] for row in csv_rows] == table["raw_budget_amount"].to_pylist()

    xlsx = BytesIO()
    view.to_excel(xlsx)
    book = load_workbook(xlsx, read_only=True, data_only=True)
    try:
        headers, *rows = book["Data"].values
        assert len(rows) == table.num_rows
        assert [row[headers.index("budget_amount")] for row in rows] == table["budget_amount"].to_pylist()
        parts = sorted((row for row in book["Provenance"].values
                        if row[0] == "Canonical record" and type(row[1]) is int), key=lambda row: row[1])
        assert json.loads(b64decode("".join(row[3] for row in parts), validate=True)) == origin
    finally:
        book.close()

    metadata = MetaData()
    receipts = Table("imports", metadata, Column("id", String(64), primary_key=True), Column("receipt", JSON))
    facts = Table("budget_" + population, metadata, Column("import_id", String(64), nullable=False))
    for field in table.schema:
        facts.append_column(Column(field.name, BigInteger if pa.types.is_integer(field.type) else Text))
    import_id = sha256(canonical_json(origin).encode()).hexdigest()
    database = create_engine("sqlite:///" + str(Path(tmp_path) / (population + ".sqlite")))
    try:
        with database.begin() as connection:
            metadata.create_all(connection)
            connection.execute(receipts.insert(), {"id": import_id, "receipt": origin})
            connection.execute(facts.insert(), [{"import_id": import_id, **row} for row in table.to_pylist()])
        with database.connect() as connection:
            restored = connection.execute(select(*[facts.c[name] for name in table.column_names]).order_by(
                facts.c.source_record,
            )).mappings().all()
            assert [dict(row) for row in restored] == sorted(table.to_pylist(), key=lambda row: row["source_record"])
            assert connection.execute(select(receipts.c.receipt)).scalar_one() == origin
    finally:
        database.dispose()
