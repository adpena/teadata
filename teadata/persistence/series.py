"""Stream pinned observations into an existing SQLAlchemy Core schema."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import sqlite3
from typing import TYPE_CHECKING, Any

from ..citations import citation
from ..identifiers import canonical_source_identifier
from ..manifest import ContractError, canonical_json

if TYPE_CHECKING:
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
    if (connection.get_execution_options().get("isolation_level") == "AUTOCOMMIT"
            or getattr(driver, "autocommit", None) is True
            or (isinstance(driver, sqlite3.Connection) and driver.isolation_level is None
                and not driver.in_transaction)):
        raise ValueError("to_sql requires a connection with driver autocommit disabled.")
    if (connection.in_nested_transaction() and isinstance(driver, sqlite3.Connection)
            and getattr(driver, "autocommit", None) == getattr(sqlite3, "LEGACY_TRANSACTION_CONTROL", None)):
        raise ValueError("Use a top-level transaction for to_sql with SQLite's legacy transaction mode.")
    fields = series.schema.names
    if "import_id" in fields:
        raise ValueError("The source projection conflicts with the reserved 'import_id' column key.")
    absent = (set(fields) | {"import_id"}) - set(observations.c.keys())
    if absent:
        raise ValueError(f"Observation table is missing required column keys: {sorted(absent)}.")
    if observations.key == receipts.key or not {"id", "receipt"} <= set(receipts.c.keys()):
        raise ValueError("Use a separate receipt table with 'id' and 'receipt' column keys.")
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
        "destination": {
            "schema": observations.schema,
            "table": observations.name,
            "columns": {key: observations.c[key].name for key in (*fields, "import_id")},
        },
        "citations": {"text": reference.text, "bibtex": reference.bibtex},
    }
    import_id = sha256(canonical_json(plan).encode()).hexdigest()
    connection.execute(receipts.insert(), {"id": import_id, "receipt": plan})

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
        connection.execute(insert, rows)

    result = SeriesSqlExport(import_id, rows_by_year, tuple(sorted(missing)))
    if require_complete and missing:
        raise SeriesCoverageError(result)
    connection.execute(receipts.update().where(receipts.c.id == import_id),
                       {"receipt": {**plan, "materialization": result.to_dict()}})
    return result
