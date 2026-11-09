"""Materialize a pinned public PEIMS series into a caller-owned SQLite database."""

import argparse
from pathlib import Path

from sqlalchemy import BigInteger, Column, ForeignKey, Integer, JSON, MetaData, String, Table, Text, URL, create_engine

from teadata.manifest import canonical_json
from teadata.release import DataRelease

MANIFEST = "https://github.com/adpena/teadata/releases/download/v0.0.143/peims-enrollment-2019-2022.json"
MANIFEST_SHA256 = "fdd86b5bf8e5c4b1e2a97523e3f651cf41281cf0f94cae5cfc7e0a97f0300791"
DATASET = "peims_student_enrollment_by_ethnicity"
FIELD = "enrollment_by_ethnicity_count"

metadata = MetaData()
receipts = Table(
    "research_imports", metadata,
    Column("id", String(64), primary_key=True),
    Column("receipt", JSON, nullable=False),
)
observations = Table(
    "campus_enrollment", metadata,
    Column("import_id", ForeignKey("research_imports.id"), primary_key=True),
    Column("measurement_school_year", Integer, primary_key=True),
    Column("campus_number", String(9), primary_key=True),
    Column("ethnicity", Text, primary_key=True),
    Column("school_year", Text, nullable=False),
    Column(FIELD, BigInteger),
    Column(f"peims_raw_{FIELD}", Text),
    Column(f"peims_status_{FIELD}", Text, nullable=False),
    Column("peims_source_record", BigInteger, nullable=False),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="SQLite database to create or append a new import to")
    parser.add_argument("--manifest", default=MANIFEST, help="Use the public manifest or a local copy with the same hash")
    parser.add_argument("--campus", action="append", help="Campus ID, repeat to select several")
    parser.add_argument("--year", action="append", help="Measurement school year, repeat to select several")
    options = parser.parse_args()

    release = DataRelease(options.manifest, expected_sha256=MANIFEST_SHA256)
    series = release.series(DATASET, FIELD, scope="campus",
                            years=options.year or [2020, 2021, 2022],
                            ids=options.campus or ["057829001", "227901002"])
    engine = create_engine(URL.create("sqlite", database=str(options.database)))
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("PRAGMA foreign_keys = ON")
            metadata.create_all(connection)
            result = series.to_sql(connection, observations, receipts=receipts)
        print(canonical_json(result.to_dict()))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
