from contextlib import closing
import gzip
import json
from enum import IntEnum
import sqlite3
import sys
from types import SimpleNamespace

import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import numpy as np

from teadata.analytical import (
    ANALYTICAL_METADATA, ANALYTICAL_ROLES, _cell_values, analytical_dataset,
    analytical_schema, measurement_from_row, presentation_from_row,
    raw_value_from_row, validate_artifact, presentation_record, presentation_values,
)
from teadata.analytical_build import build_analytical_artifacts
from teadata.entity_store import EntityStore
from teadata.manifest import ContractError, DatasetCatalog, canonical_json, file_identity, processor_identity, write_sqlite_catalog
from teadata.values import Measurement, ValueStatus, presentation_value
from teadata.teadata_config import canonical_campus_number, canonical_district_number
from teadata.provenance import ProvenanceStore
from teadata.release import DataRelease, publish_release, SNAPSHOT_MAGIC


def _typed(value):
    if isinstance(value, dict):
        return (dict, tuple((key, _typed(item)) for key, item in value.items()))
    if isinstance(value, list):
        return (list, tuple(map(_typed, value)))
    if isinstance(value, float):
        return (float, value.hex())
    return (type(value), value)


@pytest.mark.parametrize("raw", [None, "", "=SUM(A1)", False, 0, -0.0, 2**80,
                                    {"z": [None, True, -0.0, 2**80], "a": 1}, [".", 0]])
def test_exact_opaque_raw_and_presentation_transport(raw):
    row = _cell_values(raw, None)
    assert _typed(raw_value_from_row(row)) == _typed(raw)
    assert measurement_from_row(row) is None
    expected = canonical_json(raw) if isinstance(raw, (dict, list)) else raw
    assert _typed(presentation_from_row(row)) == _typed(expected)
    batch = pa.RecordBatch.from_pylist([row])
    assert [_typed(value) for value in presentation_values(batch)] == [_typed(expected)]


@pytest.mark.parametrize("status", list(ValueStatus))
def test_exact_measurement_bounds_source_and_signed_zero(status):
    raw = {"z": [1, -0.0], "a": True}
    value = Measurement(-0.0 if status == ValueStatus.OBSERVED else None, status,
                        raw, "documented_reason", "policy", 1, 9.0, "count",
                        '{"period":{"kind":"calendar_year","year":2025}}')
    restored = measurement_from_row(_cell_values(raw, value))
    assert _typed(restored.to_dict()) == _typed(value.to_dict())
    assert _typed(restored.raw) == _typed(raw)
    assert restored._source_json == value._source_json
    assert _typed(presentation_from_row(_cell_values(raw, value))) == _typed(presentation_value(value))
    assert type(restored.lower) is int and type(restored.upper) is float


def test_private_measurement_does_not_lose_original_raw():
    value = Measurement(None, ValueStatus.NOT_APPLICABLE, None, "outside_metric_population")
    row = _cell_values({"z": 1}, value)
    assert measurement_from_row(row).raw is None
    assert raw_value_from_row(row) == {"z": 1}


def _entities(path, catalog):
    enrollment = catalog.source_for_metric("enrollment", "campus")[1]
    rating = catalog.source_for_metric("rating", "campus")[1]
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript("""
            CREATE TABLE districts (id TEXT PRIMARY KEY, name TEXT, district_number TEXT,
                district_number_canon TEXT, enrollment, rating, aea, aea_raw, meta TEXT);
            CREATE TABLE campuses (id TEXT PRIMARY KEY, district_id TEXT, campus_number TEXT,
                campus_number_canon TEXT, name TEXT, is_charter, is_private, enrollment,
                rating, aea, aea_raw, grade_range TEXT, lon, lat, meta TEXT);
        """)
        connection.execute("INSERT INTO districts VALUES (?,?,?,?,?,?,?,?,?)", (
            "d", " ISD ", "012345", canonical_district_number("012345"), 50, "A", None, None,
            json.dumps({"district_opaque": {"z": [0, False], "a": -0.0}}),
        ))
        for number, private, meta in (
            ("012345678", False, {enrollment: -999, rating: "A", "opaque": {"z": [0, False], "a": -0.0}, "explicit_null": None}),
            ("012345679", True, {enrollment: 50, rating: "B", "wide": 2**80}),
        ):
            connection.execute("INSERT INTO campuses VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                number, "d", number, canonical_campus_number(number), " Campus ", False, private, 50,
                "C", None, None, " KG–12 ", -97.1, 30.2, json.dumps(meta),
            ))
        write_sqlite_catalog(connection, catalog)
        connection.commit()
    return path


@pytest.fixture
def analytical(tmp_path, catalog):
    source = _entities(tmp_path / "entities.sqlite", catalog)
    paths, diagnostics = build_analytical_artifacts(source, catalog, tmp_path / "model", batch_size=1)
    return source, paths, diagnostics


def test_standard_arrow_projection_exact_scalar_and_presence(analytical, catalog):
    source, paths, diagnostics = analytical
    assert set(paths) == ANALYTICAL_ROLES
    dataset = analytical_dataset(paths["campus_analytics"], catalog, "campus_analytics",
                                 entities_sha256=file_identity(source)["sha256"])
    assert isinstance(dataset, ds.Dataset)
    with EntityStore(source) as store:
        for row in dataset.to_table().to_pylist():
            record = store.get_campus(row["entity_id"], cache=False)
            context = SimpleNamespace(**record)
            if row["kind"] == "field":
                expected = catalog.measure_field(record["meta"].get(row["name"]), row["name"], scope="campus", entity=context)
                assert row["raw_present"] is (row["name"] in record["meta"])
                assert _typed(raw_value_from_row(row)) == _typed(record["meta"].get(row["name"]))
            else:
                expected = catalog.measurement(context, row["name"], scope="campus")
                assert row["raw_present"] is None
            actual = measurement_from_row(row)
            assert _typed(actual.to_dict() if actual else None) == _typed(expected.to_dict() if expected else None)
    projected = dataset.to_table(columns=["entity_id", "value_integer", "status", "period"],
                                 filter=(ds.field("kind") == "metric") & (ds.field("name") == "enrollment"))
    assert projected.num_rows == 2
    assert projected["status"].to_pylist()[1] == "not_applicable"
    assert diagnostics["runtime"]["campus"]["source_hydration_seconds"] >= 0
    assert diagnostics["scopes"]["campus"]["cells"] == dataset.count_rows()
    attributes = pq.read_table(paths["campus_attributes"]).to_pylist()
    assert attributes[0]["campus_name"] == "Campus"
    assert attributes[0]["campus_number"] == "012345678"
    assert attributes[0]["campus_latitude"] == 30.2


def test_source_metadata_is_research_readable_and_detached(analytical, catalog):
    _, paths, _ = analytical
    rows = pq.read_table(paths["analytical_sources"], filters=[("kind", "=", "metric")]).to_pylist()
    for row in rows:
        info = catalog.metric_metadata(row["name"], scope=row["scope"])
        assert row["dataset"] == info["dataset"]
        assert row["period"]["kind"] == info["period"]["kind"]
        assert row["source_url"] == info["metadata"].get("source_url")
        for actual, expected in zip(row["references"], info["metadata"]["references"], strict=True):
            assert all(actual[key] == value for key, value in expected.items())
            assert all(value is None for key, value in actual.items() if key not in expected)
        assert row["acquisitions"]
    field = catalog.source_for_metric("enrollment", "campus")[1]
    info = catalog.field_metadata(field)
    info["metadata"]["period"]["kind"] = "unknown"
    assert catalog.field_metadata(field)["metadata"]["period"]["kind"] != "unknown"
    assert catalog.field_metadata("opaque") is None


def test_no_final_manifest_hash_cycle_and_row_count_guard(analytical, catalog, tmp_path):
    _, paths, _ = analytical
    path = paths["campus_analytics"]
    footer = validate_artifact(path, catalog, "campus_analytics")
    assert "manifest_sha256" not in footer
    assert len(footer["entities_sha256"]) == 64
    assert footer["producer_inputs"]["python"]
    table = pq.read_table(path)
    bad = tmp_path / "bad.parquet"
    footer["rows"] += 1
    pq.write_table(table.replace_schema_metadata({ANALYTICAL_METADATA: canonical_json(footer).encode()}), bad)
    with pytest.raises(ContractError, match="row count"):
        validate_artifact(bad, catalog, "campus_analytics")
    with pytest.raises(ContractError, match="different role"):
        validate_artifact(path, catalog, "district_analytics")


def test_fresh_outputs_and_source_change_fail_closed(analytical, catalog, tmp_path, monkeypatch):
    import teadata.analytical_build as builder

    source, paths, _ = analytical
    with pytest.raises(ContractError, match="fresh"):
        build_analytical_artifacts(source, catalog, next(iter(paths.values())).parent)
    original = builder.entity_attributes
    changed = False

    def mutate(entity, **kwargs):
        nonlocal changed
        if not changed:
            with source.open("ab") as stream:
                stream.write(b"changed")
            changed = True
        return original(entity, **kwargs)

    monkeypatch.setattr(builder, "entity_attributes", mutate)
    target = tmp_path / "failed"
    with pytest.raises(ContractError, match="source changed"):
        build_analytical_artifacts(source, catalog, target)
    assert not list(target.iterdir())


def test_arrow_types_are_explicit():
    schema = analytical_schema("campus_analytics")
    assert str(schema.field("raw_integer").type) == "int64"
    assert str(schema.field("raw_integer_decimal").type) == "large_string"
    assert str(schema.field("raw_float").type) == "double"
    assert not any(str(field.type) == "object" for field in schema)
    with pytest.raises(ContractError, match="lacks required lanes"):
        list(presentation_values(pa.RecordBatch.from_pylist([{"wrong": 0}])))


def test_complete_release_binds_arrow_children_to_exact_source(analytical, catalog, tmp_path):
    source, paths, _ = analytical
    paths = {**paths, "entities": source}
    for role in ("maps", "boundaries"):
        path = tmp_path / (role + ".sqlite")
        with closing(sqlite3.connect(path)) as connection:
            write_sqlite_catalog(connection, catalog)
            connection.commit()
        paths[role] = path
    paths["snapshot"] = tmp_path / "snapshot.pkl.gz"
    with gzip.open(paths["snapshot"], "wb") as stream:
        stream.write(SNAPSHOT_MAGIC + catalog.build_id.encode() + b"\n")
    store = ProvenanceStore(tmp_path / "provenance")
    records = store.root / "records"
    records.mkdir(parents=True)
    for identifier, record in catalog.to_dict()["provenance"].items():
        (records / (identifier + ".json")).write_text(canonical_json(record) + "\n")
    record_id = store.record_processing(dataset="test_release", inputs=list(catalog.to_dict()["provenance"]), outputs=paths,
                                       processor="fixture", code_version="test",
                                       parameters={"build_id": catalog.build_id})
    manifest = publish_release(tmp_path / "published", catalog, paths,
                               processing={"record_id": record_id, "record": store.read_record(record_id)})
    release = DataRelease(manifest)
    assert json.loads(manifest.read_text())["schema_version"] == 2
    assert set(release.verify()) == set(paths)
    assert isinstance(release.analytical_dataset("campus"), ds.Dataset)
    assert release.analytical_dataset("sources").count_rows() > 0
    with pytest.raises(ContractError, match="Unknown analytical"):
        release.analytical_dataset("unknown")
    document = json.loads(manifest.read_text())
    del document["artifacts"]["campus_attributes"]
    manifest.write_text(json.dumps(document))
    with pytest.raises(ContractError, match="complete artifact"):
        DataRelease(manifest)


def test_actual_producer_is_separate_from_source_catalog_processor(analytical, catalog):
    _, paths, _ = analytical
    metadata = validate_artifact(paths["campus_analytics"], catalog, "campus_analytics")
    inputs = metadata["producer_inputs"]
    assert inputs["processor_sha256"] == processor_identity()
    assert inputs["processor_sha256"] != catalog.to_dict()["processor_sha256"]
    assert inputs["source_processor_sha256"] == catalog.to_dict()["processor_sha256"]
    assert inputs["lock_sha256"]


def test_percent_safe_source_uri_reads_exact_file_without_creating_a_wrong_file(tmp_path, catalog):
    directory = tmp_path / "directory#?%"
    directory.mkdir()
    source = _entities(directory / "entities#?%.sqlite", catalog)
    before = {path.name for path in directory.iterdir()}
    identity = file_identity(source)
    paths, _ = build_analytical_artifacts(source, catalog, tmp_path / "model")
    assert validate_artifact(paths["campus_analytics"], catalog, "campus_analytics")["entities_sha256"] == identity["sha256"]
    with EntityStore(source, immutable=True) as store:
        assert store.catalog.build_id == catalog.build_id
        assert store.get_campus("012345678")["name"] == " Campus "
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            store.connection.execute("DELETE FROM campuses")
    assert {path.name for path in directory.iterdir()} == before
    assert file_identity(source) == identity


def test_wal_introduced_during_hydration_refuses_all_outputs(tmp_path, catalog, monkeypatch):
    import teadata.analytical_build as builder

    source = _entities(tmp_path / "entities.sqlite", catalog)
    with closing(sqlite3.connect(source)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
    before = file_identity(source)
    writers = []
    original = builder.entity_attributes

    def introduce_wal(entity, **kwargs):
        if not writers:
            writer = sqlite3.connect(source)
            writers.append(writer)
            writer.execute("UPDATE campuses SET meta=?", ('{"changed":777}',))
            writer.commit()
            assert file_identity(source) == before
        return original(entity, **kwargs)

    monkeypatch.setattr(builder, "entity_attributes", introduce_wal)
    output = tmp_path / "failed"
    try:
        with pytest.raises(ContractError, match="checkpointed"):
            build_analytical_artifacts(source, catalog, output, batch_size=1)
        assert not list(output.iterdir())
    finally:
        for writer in writers:
            writer.close()


def test_requested_metric_scopes_are_preflighted_before_opening_source(tmp_path, catalog):
    document = catalog.to_dict()
    document["metric_bindings"]["enrollment"]["scopes"] = ["campus"]
    narrow = DatasetCatalog.create(datasets=document["datasets"], value_policies=document["value_policies"],
                                   processor=document["processor_sha256"], records=document["provenance"],
                                   environment=document["build_environment"], metric_bindings=document["metric_bindings"])
    with pytest.raises(ContractError):
        build_analytical_artifacts(tmp_path / "must-not-open.sqlite", narrow, tmp_path / "model")
    assert not (tmp_path / "model").exists()


def test_unsupported_json_number_refuses_instead_of_losing_field_presence(tmp_path, catalog):
    limit = sys.get_int_max_str_digits()
    if not limit:
        return  # CPython's explicitly disabled limit cannot reject this number.
    source = _entities(tmp_path / "entities.sqlite", catalog)
    numeric = '{"giant":' + "1" * (limit + 1) + '}'
    with closing(sqlite3.connect(source)) as connection:
        assert connection.execute("SELECT json_valid(?)", (numeric,)).fetchone()[0] == 1
        connection.execute("UPDATE campuses SET meta=?", (numeric,))
        connection.commit()
    with EntityStore(source, immutable=True) as store:
        with pytest.raises(ContractError, match="unsupported JSON"):
            store.get_campus("012345678")
    output = tmp_path / "failed"
    with pytest.raises(ContractError, match="unsupported JSON"):
        build_analytical_artifacts(source, catalog, output)
    assert not list(output.iterdir())


@pytest.mark.parametrize("payload", ["", "{", "[]", '"opaque"', "1", "true", '{"x":NaN}', '{"x":1e999}', b'\xff'])
def test_one_metadata_decoder_refuses_malformed_unsupported_or_nonobject_json(payload):
    from teadata.entity_store import _decode_json

    with pytest.raises(ContractError):
        _decode_json(payload)


def test_nullable_json_and_stringified_large_digits_remain_distinct():
    from teadata.entity_store import _decode_json

    assert all(_decode_json(value) == {} for value in (None, "null", b"null", memoryview(b"null")))
    text = "1" * 4301
    assert _decode_json(json.dumps({"giant": text})) == {"giant": text}
    assert _decode_json('{"zero":-0.0}')["zero"].hex() == "-0x0.0p+0"


def test_nullable_metadata_producer_keeps_metrics_and_does_not_invent_a_null_field(tmp_path, catalog):
    source = _entities(tmp_path / "entities.sqlite", catalog)
    with closing(sqlite3.connect(source)) as connection:
        connection.execute("UPDATE campuses SET meta='null'")
        connection.execute("UPDATE districts SET meta=NULL")
        connection.commit()
    paths, diagnostics = build_analytical_artifacts(source, catalog, tmp_path / "model")
    assert diagnostics["scopes"]["campus"]["fields"] == 0
    assert diagnostics["scopes"]["district"]["fields"] == 0
    assert pq.read_metadata(paths["campus_analytics"]).num_rows == 4
    assert pq.read_metadata(paths["district_analytics"]).num_rows == 2


def test_sparse_attribute_access_does_not_touch_unselected_properties():
    from teadata.attributes import attribute_value

    class Sentinel:
        name = " Named "

        def __getattr__(self, name):
            raise AssertionError(f"Unselected property accessed: {name}")

    assert attribute_value(Sentinel(), "campus_name", scope="campus") == "Named"
    assert attribute_value(Sentinel(), "district_name", scope="district") == " Named "


def test_presentation_matches_json_scalar_subclass_payload_without_conversion_overrides():
    class Code(IntEnum):
        VALUE = 12

    class Text(str):
        def __str__(self):
            return "wrong override"

    class Integer(int):
        def __int__(self):
            return 999

    class Real(float):
        def __float__(self):
            return 999.0

    for value in (False, True, Code.VALUE, ValueStatus.SUPPRESSED, Text("payload"),
                  Integer(12), Real(-0.0), np.float64(-0.0)):
        expected = json.loads(json.dumps(value, allow_nan=False))
        assert _typed(presentation_from_row(presentation_record(value))) == _typed(expected)
    with pytest.raises(TypeError):
        _cell_values(np.float64(-0.0), None)  # Raw source boundary remains strict.
    for value in (object(), (1,), np.int64(12), float("nan"), float("inf")):
        with pytest.raises((TypeError, ValueError)):
            presentation_record(value)
