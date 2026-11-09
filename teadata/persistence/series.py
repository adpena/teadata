"""Stream pinned observations through an existing SQLAlchemy Core schema."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
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
    observation_options = {"schema_translate_map": {None: schema}} if observations.schema is None else {}
    receipt_options = {"schema_translate_map": {None: receipt_schema}} if receipts.schema is None else {}
    return destination, observation_options, receipt_options


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

    columns = [observations.c[name] for name in series.schema.names]
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
                    arrays.append(pa.array(values, type=field.type))
                except (pa.ArrowException, OverflowError, TypeError, ValueError) as error:
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
    entity_key = f"{series.scope}_number"
    width = 6 if series.scope == "district" else 9
    insert = observations.insert()
    for batch in series.iter_batches():
        rows = batch.to_pylist()
        if not rows:
            continue
        year = rows[0]["measurement_school_year"]
        rows_by_year[year] += len(rows)
        if missing:
            for row in rows:
                identifier = canonical_source_identifier(row[entity_key], width=width)
                if identifier is not None:
                    missing.discard((year, identifier[1:]))
        for row in rows:
            row["import_id"] = import_id
        connection.execute(insert, rows, execution_options=observation_options)

    result = SeriesSqlExport(import_id, rows_by_year, tuple(sorted(missing)))
    if require_complete and missing:
        raise SeriesCoverageError(result)
    connection.execute(receipts.update().where(receipts.c.id == import_id),
                       {"receipt": {**plan, "materialization": result.to_dict()}}, execution_options=receipt_options)
    return result
