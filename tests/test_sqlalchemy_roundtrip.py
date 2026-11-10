"""Lossless source custody and bounded database I/O for the graph bridge."""

from hashlib import sha256
from importlib.resources import files
import json
import math
import os
import sqlite3
import uuid

import pytest
from sqlalchemy import Text, event, func, inspect, select, text
from sqlalchemy.engine import make_url

from teadata.classes import Campus, DataEngine, District
from teadata.identifiers import EntityIdentityError
from teadata.manifest import (
    CATALOG_TABLE,
    ContractError,
    DatasetCatalog,
    canonical_json,
    read_sqlite_catalog,
    write_sqlite_catalog,
)
from teadata.persistence import (
    CampusMetaRecord,
    CampusRecord,
    CampusTransferRecord,
    DistrictMetaRecord,
    DistrictRecord,
    create_engine,
    create_sessionmaker,
    ensure_schema,
    export_dataengine,
    import_dataengine,
)
from teadata.persistence.sqlalchemy_store import _meta_projection, _point_meta_dialect


@pytest.fixture
def database():
    engine = create_engine("sqlite:///:memory:")
    ensure_schema(engine)
    try:
        yield engine, create_sessionmaker(engine)
    finally:
        engine.dispose()


def graph(catalog, size=2):
    repo = DataEngine()
    repo.catalog = catalog
    with repo.bulk():
        for index in range(1, size + 1):
            district = District(
                id=uuid.UUID(int=index),
                name=f"District {index}",
                district_number=f"{index:06}",
            )
            district.meta = {
                "structured": [{"count": 2**53 + 1}],
                "zero": -0.0,
                "huge": 10**500,
            }
            repo.add_district(district)
            campus = Campus(
                id=uuid.UUID(int=index + 10000),
                district_id=district.id,
                name=f"Campus {index}",
                campus_number=f"{index:09}",
                charter_type="",
                is_charter=False,
            )
            campus.meta = {
                "nullable": None,
                "flag": True,
                "text": "period . and NUL\x00",
                "nested": {"students": [1]},
            }
            repo.add_campus(campus)
    if size >= 2:
        repo._xfers_out[uuid.UUID(int=10001)] = [(uuid.UUID(int=10002), None, True)]
    return repo


def changed_catalog(catalog):
    document = catalog.to_dict()
    document.pop("build_id")
    document["processor_sha256"] = "3" * 64
    document["build_id"] = sha256(canonical_json(document).encode()).hexdigest()
    return DatasetCatalog(document)


def test_partial_export_rejects_duplicate_identity_in_legacy_database(database, catalog):
    engine, factory = database
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP INDEX ix_districts_district_number_canon")
        connection.exec_driver_sql("CREATE INDEX ix_districts_district_number_canon ON districts(district_number_canon)")
    with factory.begin() as session:
        export_dataengine(graph(catalog, 1), session)
        session.execute(text("UPDATE districts SET district_number_canon = district_number"))

    collision = DataEngine()
    collision.catalog = catalog
    collision.add_district(District(uuid.uuid4(), "Collision", district_number="000001"))
    with pytest.raises(EntityIdentityError, match="Duplicate district number '000001'"):
        with factory.begin() as session:
            export_dataengine(collision, session)

    with factory() as session:
        restored = import_dataengine(session)
        assert [row.name for row in restored.districts.values()] == ["DISTRICT 1"]


def test_campus_export_keeps_identity_columns_independent_of_dynamic_metadata(database, catalog):
    _, factory = database
    repo = graph(catalog, 1)
    campus = next(iter(repo.campuses.values()))
    campus.meta.update(campus_number_canon="999999999", district_number_canon="999999")
    with factory.begin() as session:
        export_dataengine(repo, session)
        row = session.get(CampusRecord, campus.id)
        assert row is not None
        assert row.campus_number_canon == "'000000001"
        assert row.district_number_canon is None
        assert row.meta["campus_number_canon"] == "999999999"


@pytest.mark.parametrize("lazy", [False, True])
def test_catalog_uuid_raw_metadata_and_graph_roundtrip(database, catalog, lazy):
    engine, factory = database
    repo = graph(catalog)
    with factory.begin() as session:
        export_dataengine(repo, session, batch_size=1)
        assert not session.identity_map
    with factory() as session:
        restored = import_dataengine(session, lazy_meta=lazy, batch_size=1)
        assert not session.identity_map
        assert restored.catalog.to_dict() == catalog.to_dict()
        assert restored._xfers_out == repo._xfers_out
        for identifier, original in repo._districts.items():
            copy = restored._districts[identifier]
            assert copy._repo.catalog.build_id == catalog.build_id
            assert canonical_json(dict(copy.meta)) == canonical_json(original.meta)
            assert math.copysign(1, copy.meta["zero"]) == -1
        for identifier, original in repo._campuses.items():
            assert dict(restored._campuses[identifier].meta) == original.meta
    with engine.connect() as connection:
        assert connection.exec_driver_sql(
            "SELECT DISTINCT typeof(id) FROM districts"
        ).all() == [("text",)]
        values = (
            connection.execute(
                select(DistrictMetaRecord.value_numeric).where(
                    DistrictMetaRecord.key.in_(["huge", "structured.count"])
                )
            )
            .scalars()
            .all()
        )
        assert values and all(value is None for value in values)


def test_partial_default_preserves_unrelated_rows_and_replaces_selected_meta(
    database, catalog
):
    _, factory = database
    repo = graph(catalog)
    with factory.begin() as session:
        export_dataengine(repo, session)
    partial = graph(catalog, size=1)
    partial._districts[uuid.UUID(int=1)].name = "UPDATED NAME"
    partial._districts[uuid.UUID(int=1)].meta = {"new": [7]}
    with factory.begin() as session:
        export_dataengine(partial, session)
    with factory() as session:
        result = import_dataengine(session, lazy_meta=False)
        assert len(result._districts) == len(result._campuses) == 2
        assert result._districts[uuid.UUID(int=1)].name == "UPDATED NAME"
        assert result._districts[uuid.UUID(int=1)].meta == {"new": [7]}
        assert (
            result._districts[uuid.UUID(int=2)].meta
            == repo._districts[uuid.UUID(int=2)].meta
        )
        assert result._xfers_out == repo._xfers_out
        keys = (
            session.execute(
                select(DistrictMetaRecord.key).where(
                    DistrictMetaRecord.district_id == uuid.UUID(int=1)
                )
            )
            .scalars()
            .all()
        )
        assert keys == ["new"]


def test_release_mixing_refused_before_mutation_and_explicit_replace_clears_all_meta(
    database, catalog
):
    _, factory = database
    with factory.begin() as session:
        export_dataengine(graph(catalog), session)
    other = changed_catalog(catalog)
    with pytest.raises(ContractError, match="cannot mix"):
        with factory.begin() as session:
            export_dataengine(graph(other, 1), session)
    with factory() as session:
        assert import_dataengine(session).catalog.build_id == catalog.build_id
        assert session.scalar(select(func.count()).select_from(DistrictRecord)) == 2
    with factory.begin() as session:
        export_dataengine(
            graph(other, 1), session, replace=True, include_meta_entries=False
        )
    with factory() as session:
        assert import_dataengine(session).catalog.build_id == other.build_id
        assert session.scalar(select(func.count()).select_from(DistrictRecord)) == 1
        assert session.scalar(select(func.count()).select_from(DistrictMetaRecord)) == 0
        assert session.scalar(select(func.count()).select_from(CampusMetaRecord)) == 0


def test_lazy_prefetch_preserves_nested_raw_types_and_refuses_later_release(
    database, catalog
):
    _, factory = database
    with factory.begin() as session:
        export_dataengine(graph(catalog), session)
    with factory() as session:
        result = import_dataengine(session, prefetch_district_meta_keys=["structured"])
    district = result._districts[uuid.UUID(int=1)]
    assert district.meta["structured"] == [{"count": 2**53 + 1}]
    assert not district.meta._loaded
    with factory.begin() as session:
        export_dataengine(graph(changed_catalog(catalog), 1), session, replace=True)
    with pytest.raises(ContractError, match="catalog changed"):
        district.meta["zero"]
    assert not district.meta._loaded


def _assert_lazy_metadata_projection(engine, factory, catalog):
    repo = graph(catalog, 1)
    district_id = uuid.UUID(int=1)
    campus_id = uuid.UUID(int=10001)
    repo._campuses[campus_id].meta = {"overall_rating": "Not Rated: Minimum Size rules", "nullable": None}
    with factory.begin() as session:
        export_dataengine(repo, session, replace=True, include_meta_entries=False)

    statements = []

    def capture(connection, cursor, statement, parameters, context, executemany):
        if "JOIN teadata_catalog" in statement:
            statements.append(statement)

    event.listen(engine, "before_cursor_execute", capture)
    try:
        with factory() as session:
            restored = import_dataengine(session, prefetch_district_meta_keys=["zero"])
        district = restored._districts[district_id]
        campus = restored._campuses[campus_id]
        assert len(statements) == 1 and " -> " in statements[0]
        assert math.copysign(1, district.zero) == -1
        assert len(statements) == 1
        assert district.huge == 10**500
        assert district.meta["huge"] == 10**500
        assert len(statements) == 2 and " -> " in statements[-1]
        assert campus.nullable is None and "nullable" in campus.meta
        assert not hasattr(campus, "absent_meta_key")
        value = restored.catalog.measurement(campus, "overall_rating")
        assert value.value == "Not Rated: Minimum Size rules"
        assert value.source["build_id"] == catalog.build_id
        assert value.source["input_binding"] == "release_entity"
        assert not district.meta._loaded and not campus.meta._loaded
        assert all(" -> " in sql for sql in statements)
        district.structured[0]["extra"] = 42
        assert dict(district.meta)["structured"] == [{"count": 2**53 + 1, "extra": 42}]
    finally:
        event.remove(engine, "before_cursor_execute", capture)

    with factory.begin() as session:
        restored = import_dataengine(session)
        orm_row = session.get(DistrictRecord, district_id)
        orm_row.meta = dict(orm_row.meta, pending_value="uncommitted")
        district = restored._districts[district_id]
        with session.no_autoflush:
            assert not hasattr(district, "pending_value")
        assert district.pending_value == "uncommitted"
        assert not district.meta._loaded

    with engine.begin() as connection:
        ordinary = select(*_meta_projection(DistrictRecord.meta, ("huge", "missing"), engine.dialect.name))
        assert connection.execute(ordinary).one() == (str(10**500), None, None)
        json_document = r'{"plain":1,"plain":2,"plain\u0000":3,"other":"a\u0000b","odd.key":[null,-0.0]}'
        assignment = "CAST(:document AS JSON)" if engine.dialect.name == "postgresql" else ":document"
        connection.execute(text(f"UPDATE districts SET meta = {assignment}"), {"document": json_document})
    with factory() as session:
        district = import_dataengine(session, prefetch_district_meta_keys=["plain"])._districts[district_id]
        assert district.plain == 2
        assert district.meta["plain\0"] == 3
        assert district.other == "a\0b"
        assert district.meta["odd.key"][0] is None
        assert math.copysign(1, district.meta["odd.key"][1]) == -1


def test_lazy_sqlite_projects_point_reads_and_prefetch(database, catalog):
    engine, factory = database
    with engine.connect() as connection:
        if _point_meta_dialect(connection) is None:
            pytest.skip("SQLite JSON point projection is unavailable on this build")
    _assert_lazy_metadata_projection(engine, factory, catalog)


def test_deferred_session_uses_its_own_json_decoder(tmp_path, catalog):
    url = f"sqlite:///{tmp_path / 'meta.sqlite'}"
    engine = create_engine(url)

    def decode(text):
        value = json.loads(text)
        if isinstance(value, dict) and "huge" in value:
            value["huge"] = 17
        return value

    deferred_engine = create_engine(url, json_deserializer=decode)
    try:
        ensure_schema(engine)
        factory = create_sessionmaker(engine)
        with factory.begin() as session:
            export_dataengine(graph(catalog, 1), session)
        with factory() as session:
            repo = import_dataengine(session, meta_session_factory=create_sessionmaker(deferred_engine))
        assert repo._districts[uuid.UUID(int=1)].huge == 17
    finally:
        deferred_engine.dispose()
        engine.dispose()


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_batch_size_has_no_writes(database, catalog, size):
    _, factory = database
    with factory.begin() as session:
        with pytest.raises(ValueError, match="positive integer"):
            export_dataengine(graph(catalog), session, batch_size=size)
        assert session.scalar(select(func.count()).select_from(DistrictRecord)) == 0


def test_missing_source_catalog_is_refused(database):
    _, factory = database
    with factory.begin() as session:
        with pytest.raises(ContractError, match="catalog-bound"):
            export_dataengine(DataEngine(), session)
        assert (
            session.execute(text(f"SELECT COUNT(*) FROM {CATALOG_TABLE}")).scalar_one()
            == 0
        )
        with pytest.raises(ContractError, match="required dataset catalog"):
            import_dataengine(session)


@pytest.mark.parametrize("operation", ["write", "read"])
def test_catalog_refusal_does_not_autoflush_pending_caller_entities(database, catalog, operation):
    engine, factory = database
    with factory.begin() as session:
        export_dataengine(graph(catalog), session)
        if operation == "read":
            session.connection().execute(
                text(f"UPDATE {CATALOG_TABLE} SET build_id=:identity"),
                {"identity": "f" * 64},
            )
    statements = []

    def observe(connection, cursor, statement, parameters, context, executemany):
        statements.append(statement.lstrip().upper())

    with factory() as session:
        pending = DistrictRecord(id=uuid.UUID(int=99), name="Caller pending district")
        session.add(pending)
        event.listen(engine, "before_cursor_execute", observe)
        try:
            with pytest.raises(ContractError, match="cannot mix|projection mismatch"):
                if operation == "write":
                    export_dataengine(graph(changed_catalog(catalog), 1), session)
                else:
                    import_dataengine(session)
        finally:
            event.remove(engine, "before_cursor_execute", observe)
        assert pending in session.new
        assert inspect(pending).pending
        assert not any(sql.startswith(("INSERT", "UPDATE", "DELETE")) for sql in statements)
        assert session.connection().execute(
            select(func.count()).select_from(DistrictRecord)
        ).scalar_one() == 2
        session.rollback()


@pytest.mark.parametrize("already_bound", [False, True])
def test_accepted_catalog_still_flushes_caller_work_and_leaves_commit_to_caller(database, catalog, already_bound):
    _, factory = database
    if already_bound:
        with factory.begin() as session:
            export_dataengine(graph(catalog), session)
    with factory() as session:
        pending = DistrictRecord(id=uuid.UUID(int=99), name="Caller pending district")
        session.add(pending)
        export_dataengine(graph(catalog, 1), session)
        assert pending not in session.new
        assert inspect(pending).persistent
        assert session.connection().execute(
            select(func.count()).select_from(DistrictRecord)
        ).scalar_one() == (3 if already_bound else 2)
        session.rollback()
    with factory() as session:
        assert session.connection().execute(
            select(func.count()).select_from(DistrictRecord)
        ).scalar_one() == (2 if already_bound else 0)


def test_postgresql_roundtrip_preserves_full_publisher_rating(catalog):
    """Opt-in native PostgreSQL check: SQLite does not enforce VARCHAR widths."""
    url = os.environ.get("TEADATA_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("Set TEADATA_TEST_POSTGRES_URL to a fresh empty teadata_test_* database")
    destination = make_url(url)
    if destination.get_backend_name() != "postgresql" or not (
        destination.database and destination.database.startswith("teadata_test_")
    ):
        pytest.fail("Publisher-label integration requires an explicitly owned PostgreSQL test database")
    excerpts = json.loads(
        files("teadata").joinpath("reference_cases/publisher_2025.json").read_text()
    )
    case = next(c for c in excerpts["cases"] if c["entity"].get("number") == "220820102")
    label = "Not Rated: Minimum Size rules"  # Retained Summary!G8507, not a generated oracle.
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            assert inspect(connection).get_table_names() == [], "Test database must be empty"
        ensure_schema(engine)
        factory = create_sessionmaker(engine)
        repo = DataEngine()
        repo.catalog = catalog
        district = District(
            id=uuid.uuid5(uuid.NAMESPACE_URL, "publisher-test:220820"),
            name="ROCKETSHIP PUBLIC SCHOOLS",  # Campus Student record 7474, DISTNAME.
            district_number="220820",
        )
        campus = Campus(
            id=uuid.uuid5(uuid.NAMESPACE_URL, "publisher-test:220820102"),
            district_id=district.id,
            district_number="220820",
            campus_number="220820102",
            name=case["entity"]["name"],
            charter_type="OPEN ENROLLMENT CHARTER",
            is_charter=True,
            rating=label,
            meta={fact["field"]: fact["source"]["raw_token"] for fact in case["facts"]},
        )
        repo.add_district(district)
        repo.add_campus(campus)
        with factory.begin() as session:
            export_dataengine(repo, session)
        with engine.connect() as connection:
            stored, length = connection.execute(
                select(CampusRecord.rating, func.char_length(CampusRecord.rating))
            ).one()
            assert stored == label and length == 29
            for table in ("districts", "campuses"):
                rating_column = next(c for c in inspect(connection).get_columns(table) if c["name"] == "rating")
                assert isinstance(rating_column["type"], Text)
        for lazy in (False, True):
            with factory() as session:
                restored = import_dataengine(session, lazy_meta=lazy)
                actual = restored._campuses[campus.id]
                assert restored.catalog.build_id == catalog.build_id
                assert actual.rating == label
                assert actual.meta["overall_rating"] == label
                value = restored.catalog.measurement(actual, "overall_rating")
                assert value.value == value.raw == label
                assert value.status.value == "observed"
                assert value.source["period"] == {"kind": "calendar_year", "year": 2025}
                assert value.source["build_id"] == catalog.build_id
        with engine.connect() as connection:
            project_metadata = _point_meta_dialect(connection) == "postgresql"
        if project_metadata:
            from psycopg import Cursor
            from psycopg.types.json import set_json_loads

            def decode(payload):
                value = json.loads(payload)
                if isinstance(value, dict) and "overall_rating" in value:
                    value["overall_rating"] = "driver-decoded"
                return value

            with factory() as session:
                restored = import_dataengine(session)
                set_json_loads(decode, session.connection().connection.driver_connection)
                assert restored._campuses[campus.id].overall_rating == "driver-decoded"
            engine.dispose()

            class DecodingCursor(Cursor):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    set_json_loads(decode, self)

            custom_engine = create_engine(url, connect_args={"cursor_factory": DecodingCursor})
            try:
                with create_sessionmaker(custom_engine)() as session:
                    restored = import_dataengine(session)
                    assert restored._campuses[campus.id].overall_rating == "driver-decoded"
            finally:
                custom_engine.dispose()

            _assert_lazy_metadata_projection(engine, factory, catalog)
    finally:
        engine.dispose()


def test_unsafe_legacy_uuid_schema_is_refused_without_ddl_changes():
    engine = create_engine("sqlite:///:memory:")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE districts(id UUID PRIMARY KEY, name TEXT)"
            )
            connection.exec_driver_sql(
                "INSERT INTO districts VALUES ('00000000000000000000000000000001','legacy')"
            )
        before = inspect(engine).get_table_names()
        with pytest.raises(ContractError, match="Legacy SQLite UUID"):
            ensure_schema(engine)
        assert inspect(engine).get_table_names() == before
    finally:
        engine.dispose()


def test_batched_graph_io_does_not_hydrate_orm_or_query_per_entity(database, catalog):
    engine, factory = database
    statements = []

    def capture(connection, cursor, statement, parameters, context, executemany):
        statements.append((statement, executemany))

    event.listen(engine, "before_cursor_execute", capture)
    try:
        with factory.begin() as session:
            export_dataengine(
                graph(catalog, 513), session, batch_size=32, include_meta_entries=False
            )
            assert not session.identity_map
        selects = [
            sql for sql, _ in statements if sql.lstrip().upper().startswith("SELECT")
        ]
        assert len(selects) < 10  # A per-entity SELECT would exceed a thousand.
        assert any(many and "INSERT INTO districts" in sql for sql, many in statements)
        statements.clear()
        with factory() as session:
            restored = import_dataengine(session, lazy_meta=False, batch_size=32)
            assert len(restored._districts) == len(restored._campuses) == 513
            assert not session.identity_map
        selects = [
            sql for sql, _ in statements if sql.lstrip().upper().startswith("SELECT")
        ]
        assert len(selects) < 10  # Includes eager metadata in the entity reads.
    finally:
        event.remove(engine, "before_cursor_execute", capture)


def test_caller_rollback_covers_catalog_entities_and_meta(database, catalog):
    engine, factory = database

    def fail(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO campuses"):
            raise RuntimeError("injected database write failure")

    event.listen(engine, "before_cursor_execute", fail)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            with factory.begin() as session:
                export_dataengine(graph(catalog), session)
    finally:
        event.remove(engine, "before_cursor_execute", fail)
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(DistrictRecord)) == 0
        assert session.scalar(select(func.count()).select_from(CampusRecord)) == 0
        assert session.scalar(select(func.count()).select_from(DistrictMetaRecord)) == 0
        assert (
            session.execute(text(f"SELECT COUNT(*) FROM {CATALOG_TABLE}")).scalar_one()
            == 0
        )


def test_lazy_metadata_binds_only_the_build_id(database, catalog):
    engine, factory = database
    with factory.begin() as session:
        export_dataengine(graph(catalog), session)
    with factory() as session:
        repo = import_dataengine(session)
    lengths = []

    def observe(connection, cursor, statement, parameters, context, executemany):
        if "JOIN teadata_catalog" in statement:
            lengths.append(
                [len(value.encode()) for value in parameters if isinstance(value, str)]
            )

    event.listen(engine, "before_cursor_execute", observe)
    try:
        for entity in [*repo._districts.values(), *repo._campuses.values()]:
            dict(entity.meta)
    finally:
        event.remove(engine, "before_cursor_execute", observe)
    assert len(lengths) == 4
    assert all(max(values) == 64 for values in lengths)


def test_catalog_projection_mismatch_refuses_read_and_write(database, catalog):
    _, factory = database
    with factory.begin() as session:
        export_dataengine(graph(catalog), session)
    with factory.begin() as session:
        session.execute(
            text(f"UPDATE {CATALOG_TABLE} SET build_id=:identity"),
            {"identity": "f" * 64},
        )
    with factory() as session:
        with pytest.raises(ContractError, match="projection mismatch"):
            import_dataengine(session)
    with factory.begin() as session:
        with pytest.raises(ContractError, match="projection mismatch"):
            export_dataengine(graph(catalog, 1), session, replace=True)
        assert session.scalar(select(func.count()).select_from(DistrictRecord)) == 2


def test_legacy_catalog_read_compatibility_and_bounded_layout_refusal(
    tmp_path, catalog
):
    path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            f"CREATE TABLE {CATALOG_TABLE} (singleton INTEGER PRIMARY KEY, document TEXT NOT NULL)"
        )
        connection.execute(
            f"INSERT INTO {CATALOG_TABLE} VALUES (1,?)",
            (canonical_json(catalog.to_dict()),),
        )
    assert read_sqlite_catalog(path).build_id == catalog.build_id
    engine = create_engine(f"sqlite:///{path}")
    try:
        before = path.read_bytes()
        with pytest.raises(ContractError, match="bounded build ID fence"):
            ensure_schema(engine)
        assert path.read_bytes() == before
    finally:
        engine.dispose()
    modern = tmp_path / "modern.sqlite"
    with sqlite3.connect(modern) as connection:
        write_sqlite_catalog(connection, catalog)
    assert read_sqlite_catalog(modern).build_id == catalog.build_id
    with sqlite3.connect(modern) as connection:
        connection.execute(f"UPDATE {CATALOG_TABLE} SET build_id=?", ("f" * 64,))
    with pytest.raises(ContractError, match="projection mismatch"):
        read_sqlite_catalog(modern)


@pytest.mark.parametrize("key", ["bad\x00key", "é" * 256, "bad\ud800key"])
def test_optional_index_key_domain_is_explicit_and_raw_only_remains_lossless(
    database, catalog, key
):
    _, factory = database
    repo = graph(catalog)
    repo._districts[uuid.UUID(int=2)].meta = {key: {"raw": [1, None]}}
    with pytest.raises(ValueError, match="include_meta_entries=False"):
        with factory.begin() as session:
            export_dataengine(repo, session, batch_size=1)
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(DistrictRecord)) == 0
        assert (
            session.execute(text(f"SELECT COUNT(*) FROM {CATALOG_TABLE}")).scalar_one()
            == 0
        )
    with factory.begin() as session:
        export_dataengine(repo, session, include_meta_entries=False)
    with factory() as session:
        restored = import_dataengine(session, lazy_meta=False)
        assert restored._districts[uuid.UUID(int=2)].meta == {key: {"raw": [1, None]}}


def test_index_key_limit_counts_characters_not_utf8_bytes(database, catalog):
    _, factory = database
    repo = graph(catalog, 1)
    key = "é" * 255
    repo._districts[uuid.UUID(int=1)].meta = {key: 7}
    with factory.begin() as session:
        export_dataengine(repo, session)
    with factory() as session:
        assert session.scalar(select(DistrictMetaRecord.key)) == key


def test_all_catalog_readers_reject_extra_rows_before_entity_writes(database, catalog):
    _, factory = database
    with factory.begin() as session:
        export_dataengine(graph(catalog), session)
        connection = session.connection()
        connection.exec_driver_sql(f"DROP TABLE {CATALOG_TABLE}")
        connection.exec_driver_sql(
            f"CREATE TABLE {CATALOG_TABLE} (singleton INTEGER PRIMARY KEY, build_id TEXT NOT NULL, document TEXT NOT NULL)"
        )
        for singleton in (1, 2):
            session.execute(
                text(
                    f"INSERT INTO {CATALOG_TABLE} VALUES (:singleton,:build_id,:document)"
                ),
                {
                    "singleton": singleton,
                    "build_id": catalog.build_id,
                    "document": canonical_json(catalog.to_dict()),
                },
            )
    with factory() as session:
        with pytest.raises(ContractError, match="exactly one dataset catalog"):
            import_dataengine(session, lazy_meta=False)
    for replace in (False, True):
        with factory.begin() as session:
            with pytest.raises(ContractError, match="exactly one dataset catalog"):
                export_dataengine(graph(catalog, 1), session, replace=replace)
            assert session.scalar(select(func.count()).select_from(DistrictRecord)) == 2
            assert (
                session.execute(
                    text(f"SELECT COUNT(*) FROM {CATALOG_TABLE}")
                ).scalar_one()
                == 2
            )


@pytest.mark.parametrize(
    "model,orphan",
    [
        (
            CampusTransferRecord,
            {
                "source_id": uuid.UUID(int=10001),
                "destination_id": uuid.UUID(int=10001),
                "student_count": 37,
                "masked": False,
            },
        ),
        (
            DistrictMetaRecord,
            {
                "district_id": uuid.UUID(int=1),
                "key": "legacy_unknown",
                "ordinal": 0,
                "value_json": 37,
            },
        ),
        (
            CampusMetaRecord,
            {
                "campus_id": uuid.UUID(int=10001),
                "key": "legacy_unknown",
                "ordinal": 0,
                "value_json": 37,
            },
        ),
    ],
)
def test_unbound_orphan_rows_are_never_adopted_as_current_source(
    database, catalog, model, orphan
):
    engine, factory = database
    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 0
    with factory.begin() as session:
        session.connection().execute(model.__table__.insert(), [orphan])
    with factory() as session:
        before = session.execute(select(model.__table__)).all()
        assert len(before) == 1
    with pytest.raises(ContractError, match="unbound populated database"):
        with factory.begin() as session:
            export_dataengine(graph(catalog, 1), session)
    with factory() as session:
        assert session.execute(select(model.__table__)).all() == before
        assert session.scalar(select(func.count()).select_from(DistrictRecord)) == 0
        assert session.scalar(select(func.count()).select_from(CampusRecord)) == 0
        assert (
            session.execute(text(f"SELECT COUNT(*) FROM {CATALOG_TABLE}")).scalar_one()
            == 0
        )
    with factory.begin() as session:
        export_dataengine(graph(catalog, 1), session, replace=True)
    with factory() as session:
        restored = import_dataengine(session, lazy_meta=False)
        assert len(restored._districts) == len(restored._campuses) == 1
        assert not restored._xfers_out
        assert not session.scalar(
            select(DistrictMetaRecord.key).where(
                DistrictMetaRecord.key == "legacy_unknown"
            )
        )
        assert not session.scalar(
            select(CampusMetaRecord.key).where(CampusMetaRecord.key == "legacy_unknown")
        )
