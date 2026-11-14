"""Stream pinned observations through an existing SQLAlchemy Core schema."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import sqlite3
from typing import TYPE_CHECKING, Any

from ..citations import citation
from ..identifiers import canonical_source_identifier
from ..manifest import ContractError, canonical_json

if TYPE_CHECKING:
    import pyarrow as pa
    from sqlalchemy import Connection, Table

    from ..source_series import ObservationSeries


@dataclass(frozen=True, slots=True)
class SeriesSqlExport:
    """Identity and observed coverage of one SQL materialization."""

    import_id: str
    rows_by_year: dict[int, int]
    missing_entity_years: tuple[tuple[int, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "import_id": self.import_id,
            "rows_by_year": {str(year): rows for year, rows in self.rows_by_year.items()},
            "missing_entity_years": [list(pair) for pair in self.missing_entity_years],
        }


class SeriesCoverageError(ContractError):
    """The selected publication has no rows for requested entity-year pairs."""

    def __init__(self, result: SeriesSqlExport):
        self.result = result
        pairs = result.missing_entity_years
        super().__init__(
            f"No published rows for {len(pairs)} requested entity-year pairs: {pairs[:5]}. "
            "Inspect result.missing_entity_years, or pass require_complete=False to record the gaps."
        )


def _autocommit(connection: Connection) -> bool:
    driver = connection.connection.driver_connection
    return (connection.get_execution_options().get("isolation_level") == "AUTOCOMMIT"
            or getattr(driver, "autocommit", None) is True
            or (isinstance(driver, sqlite3.Connection) and driver.isolation_level is None
                and not driver.in_transaction))


def _namespace(connection: Connection, table: Table) -> str:
    if table.schema is not None:
        if connection.dialect.name == "postgresql" and (table.schema == "pg_temp" or table.schema.startswith("pg_temp_")):
            raise ValueError(f"Use an explicit persistent schema for PostgreSQL table {table.name!r}.")
        return table.schema
    if connection.dialect.name == "sqlite":
        return "main"
    if connection.dialect.name == "postgresql":
        from sqlalchemy import text

        namespace = connection.execute(text("""
            SELECT namespace.nspname
            FROM pg_catalog.pg_class AS relation
            JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            WHERE relation.oid = pg_catalog.to_regclass(pg_catalog.quote_ident(:name))
            """), {"name": table.name}).scalar_one_or_none()
        if isinstance(namespace, str) and not namespace.startswith("pg_temp_"):
            return namespace
        raise ValueError(f"Use an explicit persistent schema for PostgreSQL table {table.name!r}.")
    raise ValueError("Specify schema names on both SQL tables for this connection.")


def _sql_tables(series: ObservationSeries, connection: Connection, observations: Table,
                receipts: Table) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if connection.get_execution_options().get("schema_translate_map"):
        raise ValueError("Use explicit SQL schemas without schema translation for series imports.")
    schema = _namespace(connection, observations)
    receipt_schema = _namespace(connection, receipts)
    if (schema == receipt_schema and observations.name == receipts.name) or not {"id", "receipt"} <= set(receipts.c.keys()):
        raise ValueError("Use a separate receipt table with 'id' and 'receipt' column keys.")
    fields = series.schema.names
    if "import_id" in fields:
        raise ValueError("The source projection conflicts with the reserved 'import_id' column key.")
    absent = (set(fields) | {"import_id"}) - set(observations.c.keys())
    if absent:
        raise ValueError(f"Observation table is missing required column keys: {sorted(absent)}.")
    destination = {
        "schema": schema,
        "table": observations.name,
        "columns": {key: observations.c[key].name for key in (*fields, "import_id")},
    }
    if decimal_storage := _decimal_storage(series, connection, observations):
        destination["decimal_storage"] = decimal_storage
    observation_options = {"schema_translate_map": {None: schema}} if observations.schema is None else {}
    receipt_options = {"schema_translate_map": {None: receipt_schema}} if receipts.schema is None else {}
    return destination, observation_options, receipt_options


def _decimal_storage(series: ObservationSeries, connection: Connection, observations: Table) -> dict[str, Any]:
    """Admit exact decimal transport, independently of SQLAlchemy's result conversion."""
    import pyarrow as pa
    from sqlalchemy import Float, Numeric, String

    storage = {}
    probed = set()
    for field in series.schema:
        if not pa.types.is_decimal(field.type):
            continue
        precision, scale = field.type.precision, field.type.scale
        target = observations.c[field.name].type
        hint = f"Exact decimal column {field.name!r} requires Text/String or a verified native Numeric destination."
        implementation = target.dialect_impl(connection.dialect)
        if implementation.bind_processor(connection.dialect) is not None:
            raise ValueError(hint + " The selected SQL type converts bound values; use Text.")
        if isinstance(target, String):
            required = max(precision - scale, 1) + max(scale, 0) + int(scale > 0) + 1
            if target.length is not None and (type(target.length) is not int or target.length < required):
                raise ValueError(hint + f" String capacity must be at least {required} characters.")
            mode = "text"
            sql_precision = sql_scale = None
        elif isinstance(target, Numeric) and not isinstance(target, Float):
            sql_precision, sql_scale = target.precision, target.scale
            unbounded = connection.dialect.name == "postgresql" and sql_precision is None and sql_scale is None
            capacity = (type(sql_precision) is int and type(sql_scale) is int
                        and sql_scale >= scale and sql_precision - sql_scale >= precision - scale)
            if (connection.dialect.name not in {"postgresql", "duckdb"} or target.asdecimal is not True
                    or not (unbounded or capacity)):
                raise ValueError(hint + " Native Numeric must preserve the declared precision/scale and return Decimal.")
            signature = (precision, scale, sql_precision, sql_scale)
            if signature not in probed:
                _probe_decimal(connection, target, precision, scale, hint)
                probed.add(signature)
            mode = "numeric"
        else:
            raise ValueError(hint)
        storage[field.name] = {"mode": mode, "arrow_type": str(field.type),
                               "sql_precision": sql_precision, "sql_scale": sql_scale}
    return storage


def _probe_decimal(connection: Connection, target: Any, precision: int, scale: int, hint: str) -> None:
    from sqlalchemy import bindparam, cast, select

    # Build the witness without the process Decimal context (which may have a
    # precision below Arrow's). Include the least significant admitted digit.
    expected = Decimal((1, (7,) * precision, -scale))
    statement = select(cast(bindparam("teadata_decimal", expected, type_=target), target))
    compiled = statement.compile(dialect=connection.dialect)
    parameters = compiled.construct_params()
    values = tuple(parameters[name] for name in compiled.positiontup) if compiled.positiontup else parameters
    try:
        with connection.exec_driver_sql(str(compiled), values) as result:
            actual = result.scalar_one()
    except Exception as error:
        raise ValueError(hint + " Native Decimal capability could not be verified; use Text.") from error
    if not isinstance(actual, Decimal) or not actual.is_finite() or actual != expected:
        raise ValueError(hint + " The raw database driver did not round-trip Decimal exactly; use Text.")


def _selection_without_citations(selection: dict[str, Any]) -> dict[str, Any]:
    return {**selection, "sources": [
        {**source, "provenance": {key: value for key, value in source["provenance"].items() if key != "citation"}}
        for source in selection["sources"]
    ]}


def _read_series(
    series: ObservationSeries,
    connection: Connection,
    observations: Table,
    *,
    receipts: Table,
    import_id: str,
    years: tuple[int, ...] | None,
) -> Iterator[pa.RecordBatch]:
    import pyarrow as pa
    from sqlalchemy import select

    if (not isinstance(import_id, str) or len(import_id) != 64
            or any(character not in "0123456789abcdef" for character in import_id)):
        raise ValueError("Use the 64-character import_id returned by to_sql.")
    if connection.dialect.name == "postgresql" and _autocommit(connection):
        raise ValueError("Disable PostgreSQL autocommit to stream SQL observations.")
    destination, observation_options, receipt_options = _sql_tables(series, connection, observations, receipts)
    with connection.execute(select(receipts.c.receipt).where(receipts.c.id == import_id),
                            execution_options=receipt_options) as result:
        document = result.scalar_one_or_none()
    if not isinstance(document, dict):
        raise ContractError("SQL import has no receipt for the requested import ID.")
    plan = {key: value for key, value in document.items() if key != "materialization"}
    if (type(plan.get("schema_version")) is not int or plan["schema_version"] != 1
            or sha256(canonical_json(plan).encode()).hexdigest() != import_id):
        raise ContractError("SQL receipt version or plan digest differs.")
    materialization = document.get("materialization")
    if not isinstance(materialization, dict) or materialization.get("import_id") != import_id:
        raise ContractError("SQL import receipt has no completed materialization record.")
    release = plan.get("release")
    if (not isinstance(release, dict) or release.get("build_id") != series.release.build_id
            or release.get("manifest_sha256") != series.release.manifest_sha256):
        raise ContractError("SQL import requires the exact pinned series release.")
    if plan.get("destination") != destination:
        raise ContractError("SQL import destination table or column mapping differs.")
    try:
        selection = _selection_without_citations(plan["selection"])
    except (AttributeError, KeyError, TypeError):
        raise ContractError("SQL receipt selection is invalid.") from None
    if canonical_json(selection) != canonical_json(_selection_without_citations(series.provenance())):
        raise ContractError("SQL import selection differs from the pinned series.")

    from sqlalchemy import Text, cast

    columns = [cast(observations.c[field.name], Text) if pa.types.is_decimal(field.type)
               else observations.c[field.name] for field in series.schema]
    statement = select(*columns).where(observations.c.import_id == import_id)
    if years is not None:
        statement = statement.where(observations.c.measurement_school_year.in_(years))
    allowed_years = frozenset(series.years if years is None else years)
    with connection.execute(statement, execution_options={**observation_options, "yield_per": 1024}) as result:
        for rows in result.partitions():
            arrays = []
            for field, values in zip(series.schema, zip(*rows, strict=True), strict=True):
                if ((not field.nullable and None in values)
                        or (pa.types.is_integer(field.type) and any(
                            value is not None and type(value) is not int for value in values))
                        or (field.name == "measurement_school_year" and any(
                            value not in allowed_years for value in values))):
                    raise ContractError(f"SQL observations differ from the expected type or year for {field.name!r}.")
                try:
                    if pa.types.is_decimal(field.type):
                        if any(value is not None and not isinstance(value, str) for value in values):
                            raise ValueError("Decimal SQL transport must return text")
                        values = tuple(Decimal(value) if value is not None else None for value in values)
                        if any(value is not None and not value.is_finite() for value in values):
                            raise ValueError("Decimal SQL transport must return finite values")
                    arrays.append(pa.array(values, type=field.type))
                except (pa.ArrowException, InvalidOperation, OverflowError, TypeError, ValueError) as error:
                    raise ContractError(f"SQL observations cannot be decoded as {field.name!r} ({field.type}).") from error
            yield pa.RecordBatch.from_arrays(arrays, schema=series.schema)


def _export_series(
    series: ObservationSeries,
    connection: Connection,
    observations: Table,
    *,
    receipts: Table,
    require_complete: bool = True,
) -> SeriesSqlExport:
    """Insert a series and its receipt in the caller's active transaction.

    The observation table requires a column key for each Arrow field and an
    `import_id` key. The receipt table requires `id` and a JSON `receipt`
    column. Give the receipt ID a primary key and reference it from observations.
    Physical column names can differ via SQLAlchemy's Column.key. Choose types
    that preserve the source values, especially text identifiers and raw tokens.

    Call inside `with engine.begin() as connection` and let any error propagate
    to roll back the receipt and rows. Use a transactional database connection.
    This function neither creates tables nor starts or commits a transaction.
    Repeating the same import into the same table raises the receipt's unique
    constraint. A different release or selection has a different import ID.
    """
    if type(require_complete) is not bool:
        raise TypeError("require_complete must be a boolean.")
    if not connection.in_transaction():
        raise ValueError("to_sql requires an active SQLAlchemy transaction.")
    driver = connection.connection.driver_connection
    if _autocommit(connection):
        raise ValueError("to_sql requires a connection with driver autocommit disabled.")
    if (connection.in_nested_transaction() and isinstance(driver, sqlite3.Connection)
            and getattr(driver, "autocommit", None) == getattr(sqlite3, "LEGACY_TRANSACTION_CONTROL", None)):
        raise ValueError("Use a top-level transaction for to_sql with SQLite's legacy transaction mode.")
    destination, observation_options, receipt_options = _sql_tables(series, connection, observations, receipts)
    if list(receipts.primary_key.columns.keys()) != ["id"]:
        raise ValueError("The receipt table must use 'id' as its primary key.")

    reference = citation(series.release)
    plan = {
        "schema_version": 1,
        "release": {
            "build_id": series.release.build_id,
            "manifest_sha256": series.release.manifest_sha256,
            "manifest_url": reference.manifest_url,
            "software_version": reference.software_version,
        },
        "selection": series.provenance(),
        "destination": destination,
        "citations": {"text": reference.text, "bibtex": reference.bibtex},
    }
    import_id = sha256(canonical_json(plan).encode()).hexdigest()
    connection.execute(receipts.insert(), {"id": import_id, "receipt": plan}, execution_options=receipt_options)

    rows_by_year = dict.fromkeys(series.years, 0)
    missing = {(year, identifier) for year in series.years for identifier in series.ids or ()}
    from ..source_views import _entity_key

    entity_key = _entity_key(series._sources[0][2].artifact) if series.ids is not None else None
    text_decimals = tuple(name for name, value in destination.get("decimal_storage", {}).items()
                          if value["mode"] == "text")
    width = 6 if series.scope == "district" else 9
    insert = observations.insert()
    for batch in series.iter_batches():
        rows = batch.to_pylist()
        if not rows:
            continue
        year = rows[0]["measurement_school_year"]
        rows_by_year[year] += len(rows)
        if missing:
            assert entity_key is not None
            for row in rows:
                identifier = canonical_source_identifier(row[entity_key], width=width)
                if identifier is not None:
                    missing.discard((year, identifier[1:]))
        for row in rows:
            for name in text_decimals:
                if row[name] is not None:
                    if not isinstance(row[name], Decimal) or not row[name].is_finite():
                        raise ContractError(f"Source decimal {name!r} is not a finite Decimal.")
                    row[name] = format(row[name], "f")
            row["import_id"] = import_id
        connection.execute(insert, rows, execution_options=observation_options)

    result = SeriesSqlExport(import_id, rows_by_year, tuple(sorted(missing)))
    if require_complete and missing:
        raise SeriesCoverageError(result)
    connection.execute(receipts.update().where(receipts.c.id == import_id),
                       {"receipt": {**plan, "materialization": result.to_dict()}}, execution_options=receipt_options)
    return result
