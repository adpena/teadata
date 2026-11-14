"""Transport goldens for exact decimal facts; these are not TEA source qualification."""
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pyarrow as pa
import pytest
from sqlalchemy import Column, Float, Integer, JSON, MetaData, Numeric, String, Table, Text, create_engine, func, select

from teadata.manifest import ContractError
from teadata.persistence.series import _decimal_storage
from teadata.source_series import ObservationSeries
from test_source_series import releases


VALUES = [Decimal("987654.12345678901234567"), Decimal("-1.00000000000000001"), None, Decimal("0")]


def decimal_series(tmp_path, monkeypatch):
    parent = releases(tmp_path)[2025]
    original = parent.series("campus_student", "amount", scope="campus", years=[25])
    data = original.to_arrow()
    index = data.schema.get_field_index("amount")
    data = data.set_column(index, "amount", pa.array(VALUES, type=pa.decimal128(23, 17)))
    data = data.set_column(data.schema.get_field_index("VALUE"), "VALUE",
                           pa.array([str(value) if value is not None else "." for value in VALUES]))
    result = replace(original, schema=data.schema)
    prior = ObservationSeries.iter_batches

    def batches(self):
        if self is result:
            yield from data.to_batches()
        else:
            yield from prior(self)

    monkeypatch.setattr(ObservationSeries, "iter_batches", batches)
    return result


def tables(series, amount_type=Text()):
    import sqlite3

    metadata = MetaData()
    receipts = Table("receipts", metadata, Column("id", String(64), primary_key=True), Column("receipt", JSON))
    facts = Table("facts", metadata, Column("import_id", String(64)))
    for field in series.schema:
        datatype = amount_type if field.name == "amount" else Integer() if pa.types.is_integer(field.type) else Text()
        facts.append_column(Column(field.name, datatype))
    legacy = getattr(sqlite3, "LEGACY_TRANSACTION_CONTROL", None)
    engine = create_engine("sqlite://", connect_args={"autocommit": legacy} if legacy is not None else {})
    metadata.create_all(engine)
    return engine, facts, receipts


def test_exact_decimal_text_sql_and_excel_round_trip(tmp_path, monkeypatch):
    from io import BytesIO
    from openpyxl import load_workbook

    series = decimal_series(tmp_path, monkeypatch)
    engine, facts, receipts = tables(series)
    with engine.begin() as connection:
        result = series.to_sql(connection, facts, receipts=receipts)
    with engine.connect() as connection:
        record = connection.execute(select(receipts.c.receipt)).scalar_one()
        assert record["destination"]["decimal_storage"] == {"amount": {
            "mode": "text", "arrow_type": "decimal128(23, 17)", "sql_precision": None, "sql_scale": None}}
        stored = connection.execute(select(facts.c.amount)).scalars().all()
        assert stored == ["987654.12345678901234567", "-1.00000000000000001", None, "0.00000000000000000"]
        restored = pa.Table.from_batches(list(series.iter_sql_batches(connection, facts, receipts=receipts,
                                                                     import_id=result.import_id)))
    assert restored["amount"].to_pylist() == VALUES
    assert series.to_df()["amount"].tolist()[:2] == VALUES[:2]
    csv = BytesIO()
    series.to_csv(csv)
    assert b"987654.12345678901234567" in csv.getvalue() and b"-1.00000000000000001" in csv.getvalue()
    excel = BytesIO()
    series.to_excel(excel)
    book = load_workbook(excel, read_only=True, data_only=True)
    try:
        header, *rows = book["Data"].values
        column = header.index("amount")
        assert [row[column] for row in rows] == ["987654.12345678901234567", "-1.00000000000000001", None, "0E-17"]
        assert [Decimal(row[column]) if row[column] is not None else None for row in rows] == VALUES
    finally:
        book.close()


@pytest.mark.parametrize("datatype", [Numeric(23, 17), Numeric(23, 17, asdecimal=False), Float(), String(10)])
def test_unsafe_decimal_destinations_fail_before_any_receipt_or_rows(tmp_path, monkeypatch, datatype):
    series = decimal_series(tmp_path, monkeypatch)
    engine, facts, receipts = tables(series, datatype)
    with pytest.raises(ValueError, match="Exact decimal"), engine.begin() as connection:
        series.to_sql(connection, facts, receipts=receipts)
    with engine.connect() as connection:
        assert connection.execute(select(func.count()).select_from(receipts)).scalar_one() == 0
        assert connection.execute(select(func.count()).select_from(facts)).scalar_one() == 0


@pytest.mark.parametrize("corrupt", ["NaN", "Infinity", "not numeric", "1.000000000000000001", "1234567.1"])
def test_invalid_decimal_sql_rows_are_not_silently_rounded(tmp_path, monkeypatch, corrupt):
    series = decimal_series(tmp_path, monkeypatch)
    engine, facts, receipts = tables(series)
    with engine.begin() as connection:
        result = series.to_sql(connection, facts, receipts=receipts)
        connection.execute(facts.update().values(amount=corrupt))
    with engine.connect() as connection, pytest.raises(ContractError, match="cannot be decoded"):
        list(series.iter_sql_batches(connection, facts, receipts=receipts, import_id=result.import_id))


def test_native_numeric_requires_raw_driver_decimal_not_processed_float():
    from sqlalchemy.dialects.postgresql.psycopg import PGDialect_psycopg

    schema = pa.schema([pa.field("amount", pa.decimal128(23, 17))])
    series = SimpleNamespace(schema=schema)
    target = Table("facts", MetaData(), Column("amount", Numeric(23, 17)))
    calls = []

    class Result:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def scalar_one(self): return Decimal("-777777.77777777777777777")

    def execute(statement, parameters):
        calls.append((statement, parameters))
        assert parameters == {"teadata_decimal": Decimal("-777777.77777777777777777")}
        return Result()

    connection = SimpleNamespace(dialect=PGDialect_psycopg(), exec_driver_sql=execute)
    assert _decimal_storage(series, connection, target)["amount"]["mode"] == "numeric"
    assert len(calls) == 1
    Result.scalar_one = lambda self: -777777.7777777778
    with pytest.raises(ValueError, match="raw database driver"):
        _decimal_storage(series, connection, target)


@pytest.mark.parametrize("datatype", [Numeric(22, 17), Numeric(23, 16), Numeric(23, 17, asdecimal=False)])
def test_native_numeric_capacity_is_checked_before_driver_probe(datatype):
    from sqlalchemy.dialects.postgresql.psycopg import PGDialect_psycopg

    series = SimpleNamespace(schema=pa.schema([pa.field("amount", pa.decimal128(23, 17))]))
    target = Table("facts", MetaData(), Column("amount", datatype))
    connection = SimpleNamespace(dialect=PGDialect_psycopg(),
        exec_driver_sql=lambda *_: pytest.fail("Unsafe capacity reached the driver"))
    with pytest.raises(ValueError, match="Exact decimal"):
        _decimal_storage(series, connection, target)
