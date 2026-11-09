"""SQL transactions, publisher-grain rows and pinned materialization receipts."""

from hashlib import sha256
import os
from pathlib import Path
import runpy
import sqlite3

import pytest
from sqlalchemy import Column, Integer, JSON, MetaData, String, Table, Text, create_engine, func, select
from sqlalchemy.exc import IntegrityError

from teadata.manifest import canonical_json
from teadata.persistence import SeriesCoverageError
from teadata.release import DataRelease
from test_source_series import releases, use_children


def test_sql_projection_preserves_support_and_transactional_coverage(tmp_path, monkeypatch):
    children = releases(tmp_path)
    parent = children[2023]
    use_children(parent, children, monkeypatch)
    series = parent.series("campus_student", "amount", scope="campus", years=[23, 24], ids=[2, 4])
    metadata = MetaData()
    receipts = Table("imports", metadata, Column("id", String(64), primary_key=True), Column("receipt", JSON))
    facts = Table("facts", metadata, Column("import_id", String(64), nullable=False))
    for name in series.schema.names:
        datatype = Integer if name in {"amount", "measurement_school_year"} else Text
        facts.append_column(Column("publisher_token" if name == "VALUE" else name, datatype, key=name))
    legacy = getattr(sqlite3, "LEGACY_TRANSACTION_CONTROL", None)
    engine = create_engine("sqlite://", connect_args={"autocommit": legacy} if legacy is not None else {})
    metadata.create_all(engine)

    with engine.connect() as connection:
        with pytest.raises(ValueError, match="active SQLAlchemy transaction"):
            series.to_sql(connection, facts, receipts=receipts)
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection, connection.begin():
        with pytest.raises(ValueError, match="autocommit disabled"):
            series.to_sql(connection, facts, receipts=receipts)
    with engine.begin() as connection, connection.begin_nested():
        with pytest.raises(ValueError, match="top-level transaction"):
            series.to_sql(connection, facts, receipts=receipts)
    with engine.begin() as connection:
        result = series.to_sql(connection, facts, receipts=receipts)
    assert result.rows_by_year == {2023: 2, 2024: 2}
    assert not result.missing_entity_years
    missing = parent.series("campus_student", "amount", scope="campus", years=[23, 24], ids=[4, 999])
    with pytest.raises(SeriesCoverageError) as error, engine.begin() as connection:
        missing.to_sql(connection, facts, receipts=receipts)
    assert error.value.result.missing_entity_years == ((2023, "000000999"), (2024, "000000999"))
    with pytest.raises(IntegrityError), engine.begin() as connection:
        series.to_sql(connection, facts, receipts=receipts)
    with engine.connect() as connection:
        document = connection.execute(select(receipts.c.receipt)).scalar_one()
        assert document.pop("materialization") == {
            "import_id": result.import_id, "rows_by_year": {"2023": 2, "2024": 2}, "missing_entity_years": [],
        }
        assert sha256(canonical_json(document).encode()).hexdigest() == result.import_id
        assert document["selection"] == series.provenance()
        assert document["release"]["manifest_sha256"] == parent.manifest_sha256
        assert document["destination"]["columns"]["VALUE"] == "publisher_token"
        assert str(parent.path) not in canonical_json(document)
        rows = connection.execute(select(facts.c.amount, facts.c.VALUE, facts.c.amount_status)
                                  .order_by(facts.c.measurement_school_year, facts.c.campus_number)).all()
        assert rows == [(None, "-999", "suppressed"), (6, "6", "observed"),
                        (None, "-999", "suppressed"), (7, "7", "observed")]
    with engine.begin() as connection:
        partial = missing.to_sql(connection, facts, receipts=receipts, require_complete=False)
        stored = connection.execute(select(receipts.c.receipt).where(receipts.c.id == partial.import_id)).scalar_one()
        assert stored["materialization"]["missing_entity_years"] == [[2023, "000000999"], [2024, "000000999"]]
        assert connection.scalar(select(func.count()).select_from(facts)) == 6
    engine.dispose()


def test_official_peims_sql_golden():
    """Independent 2021–22 TEA HTML witness, receipt 0183affa…1555e.

    The original is linked in docs/enrichment.md. Set TEADATA_SQL_GOLDEN_MANIFEST
    to the released manifest or a verified offline copy to run this check.
    """
    manifest = os.getenv("TEADATA_SQL_GOLDEN_MANIFEST")
    if not manifest:
        pytest.skip("Set TEADATA_SQL_GOLDEN_MANIFEST to the pinned 2019–22 PEIMS manifest")
    example = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/materialize_research_db.py"))
    release = DataRelease(manifest, expected_sha256=example["MANIFEST_SHA256"])
    series = release.series(example["DATASET"], example["FIELD"], scope="campus",
                            years=[2020, 2021, 2022], ids=[57829001, "'227901002"])
    engine = create_engine("sqlite://")
    facts = example["observations"]
    receipts = example["receipts"]
    field = example["FIELD"]
    try:
        with engine.begin() as connection:
            example["metadata"].create_all(connection)
            result = series.to_sql(connection, facts, receipts=receipts)
            assert result.rows_by_year == {2020: 12, 2021: 12, 2022: 12}
            assert not result.missing_entity_years
            rows = connection.execute(select(facts.c.ethnicity, facts.c[field], facts.c[f"peims_raw_{field}"],
                                             facts.c[f"peims_status_{field}"]).where(
                facts.c.campus_number == "057829001", facts.c.measurement_school_year == 2022)).all()
            assert set(rows) == {
                ("Asian", None, "<10", "suppressed"), ("Black or African American", 33, "33", "observed"),
                ("Hispanic/Latino", 796, "796", "observed"), ("Two or more races", None, "<10", "suppressed"),
                ("White", 17, "17", "observed"),
            }
    finally:
        engine.dispose()
