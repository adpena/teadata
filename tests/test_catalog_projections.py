"""Catalog projections detach selected metadata without decoding the whole release."""

from copy import deepcopy
import gzip
from hashlib import sha256
import pickle
from pathlib import Path
from dataclasses import replace
import json
import subprocess
import sys
import threading
from importlib.metadata import version
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from teadata.engine import DataEngine
from teadata.manifest import ContractError, DatasetCatalog, canonical_json
from teadata.release import SNAPSHOT_MAGIC
from teadata.dataset_metadata import ArtifactRef, DataPeriod, EditionKey
from teadata.manifest import BuildRecorder, file_identity
from teadata.provenance import ProvenanceStore
from teadata.teadata_config import Config


@pytest.fixture
def selected_catalog():
    metadata = {"title": "Current description", "period": {"kind": "unknown"}}
    artifact = {"filename": "sample.csv", "sha256": "a" * 64, "size_bytes": 12}
    records = []
    for title, method, date in [("Earlier description", "download", "2024-01-01T00:00:00+00:00"),
                                ("Current description", "local_observation", None)]:
        records.append({"kind": "acquisition", "dataset": "sample", "section": "data_sources",
            "artifact": dict(artifact), "source_metadata": {**metadata, "title": title},
            "method": method, "downloaded_at": date,
            "parameters": {"selection": {"keys": ["A", "B"]}}})
    receipts = {sha256((canonical_json(r) + "\n").encode()).hexdigest(): r for r in records}
    first, second = receipts
    return DatasetCatalog.create(
        datasets={"sample": {"status": "loaded", "section": "data_sources",
            "metadata": {**metadata, "provenance_ids": [first, second, first]},
            "artifact": artifact, "reads": [{"columns": ["value"], "rows": 1}]}},
        records=receipts, value_policies={}, metric_bindings={}, processor="0" * 64,
    )


def test_selected_projections_preserve_order_unknown_dates_and_detach_every_nested_return(selected_catalog):
    catalog = selected_catalog
    original = catalog.to_dict()
    build_id = catalog.build_id
    sources = catalog.datasets
    observations = catalog.acquisitions("sample")
    assert [o["method"] for o in observations] == ["download", "local_observation", "download"]
    assert [o["downloaded_at"] for o in observations] == ["2024-01-01T00:00:00+00:00", None, "2024-01-01T00:00:00+00:00"]
    assert observations[0]["record_id"] == observations[2]["record_id"]
    sources["sample"]["reads"][0]["columns"].append("caller-only")
    sources["sample"]["metadata"]["period"]["kind"] = "changed by caller"
    observations[0]["artifact"]["sha256"] = "b" * 64
    observations[0]["parameters"]["selection"]["keys"].append("caller-only")
    observations[0]["source_metadata"]["title"] = "changed by caller"
    assert observations[2]["parameters"]["selection"]["keys"] == ["A", "B"]
    assert observations[2]["artifact"]["sha256"] == "a" * 64
    assert catalog.datasets == original["datasets"]
    assert catalog.acquisitions("sample")[0]["source_metadata"]["title"] == "Earlier description"
    assert catalog.to_dict() == original and catalog.build_id == build_id


def test_selected_projections_need_no_full_document_parse_or_file_io(selected_catalog, monkeypatch):
    expected_sources = selected_catalog.datasets
    expected_observations = selected_catalog.acquisitions("sample")
    with monkeypatch.context() as forbidden:
        forbidden.setattr("teadata.manifest.json.loads", lambda *a, **k: pytest.fail("whole document decoded"))
        forbidden.setattr(Path, "open", lambda *a, **k: pytest.fail("artifact/file I/O"))
        forbidden.setattr("builtins.open", lambda *a, **k: pytest.fail("artifact/file I/O"))
        forbidden.setattr(DataEngine, "from_snapshot", lambda *a, **k: pytest.fail("engine load"))
        assert selected_catalog.datasets == expected_sources
        assert selected_catalog.acquisitions("sample") == expected_observations
        assert selected_catalog.period_label("sample") == "Period not specified"
        with pytest.raises(ContractError, match="Dataset was not loaded in this release"):
            selected_catalog.acquisitions("not-loaded")


def test_derived_record_lookup_is_not_pickled_and_old_default_state_restores_it(selected_catalog):
    catalog = selected_catalog
    expected = catalog.to_dict()
    observations = catalog.acquisitions("sample")
    # A non-pickleable value in the derived lookup must never enter snapshot state.
    catalog._records["derived-only"] = lambda: None
    restored = pickle.loads(pickle.dumps(catalog))
    assert restored.to_dict() == expected and restored.acquisitions("sample") == observations
    assert "derived-only" not in restored._records
    legacy_state = catalog.__getstate__()
    assert "_records" not in legacy_state
    old_default_state = DatasetCatalog.__new__(DatasetCatalog)
    old_default_state.__setstate__(legacy_state)
    assert old_default_state.to_dict() == expected
    assert old_default_state.acquisitions("sample") == observations


@pytest.mark.parametrize("invalid", [False, True])
def test_snapshot_load_still_revalidates_canonical_document(tmp_path, selected_catalog, invalid):
    catalog = selected_catalog
    if invalid:
        document = catalog.to_dict()
        document["datasets"]["sample"]["metadata"]["title"] = "Unvalidated metadata"
        catalog._serialized = canonical_json(document)
    engine = DataEngine()
    engine.catalog = catalog
    path = tmp_path / "empty-engine.gz"
    with gzip.open(path, "wb") as stream:
        stream.write(SNAPSHOT_MAGIC + catalog.build_id.encode() + b"\n")
        pickle.dump(engine, stream, protocol=pickle.HIGHEST_PROTOCOL)
    if invalid:
        with pytest.raises(ContractError, match="catalog identity mismatch"):
            DataEngine.from_snapshot(path)
    else:
        loaded = DataEngine.from_snapshot(path)
        assert loaded.catalog is not catalog
        assert loaded.catalog.to_dict() == catalog.to_dict()
        assert loaded.catalog.acquisitions("sample") == catalog.acquisitions("sample")


def test_projection_optimization_does_not_weaken_receipt_validation(selected_catalog):
    document = selected_catalog.to_dict()
    identifier = next(iter(document["provenance"]))
    record = deepcopy(document["provenance"].pop(identifier))
    record["dataset"] = "unrelated"
    replacement = sha256((canonical_json(record) + "\n").encode()).hexdigest()
    document["provenance"][replacement] = record
    metadata = document["datasets"]["sample"]["metadata"]
    metadata["provenance_ids"] = [replacement if i == identifier else i for i in metadata["provenance_ids"]]
    document["build_id"] = sha256(canonical_json({k: v for k, v in document.items() if k != "build_id"}).encode()).hexdigest()
    with pytest.raises(ContractError, match="section, or event kind"):
        DatasetCatalog(document)


@pytest.fixture
def edition_fixture(tmp_path):
    """Two tiny native leaves built from the ordinary recorder and crosswalk."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from teadata.analytical import LEAF_METADATA
    from teadata.column_schema import source_schema_fingerprint

    store = ProvenanceStore(tmp_path / "provenance")
    specs, datasets, schemas = [], {}, {}
    for family in ("student", "staff"):
        dataset = "campus_" + family
        raw = tmp_path / (family + ".csv")
        raw.write_text("ID,NAME,VALUE\n001,One,5\n")
        raw_sha = file_identity(raw)["sha256"]
        labels, codes = ["Campus ID", "Campus name", "Amount"], ["ID", "NAME", "VALUE"]
        fields = [{"position": index, "code": code, "label": label, "field": field,
                   "disposition": "identity" if index < 3 else "fact",
                   "publisher_type": None, "publisher_description": None,
                   "dictionary_status": "not_applicable" if index < 3 else "not_documented"}
                  for index, (code, label, field) in enumerate(zip(
                      codes, labels,
                      ["campus_number", "campus_name", "amount"]), 1)]
        table = {"table": family, "input_sha256": raw_sha,
                 "source_schema_sha256": source_schema_fingerprint(labels, codes),
                 "fields": fields, "metric_period": {"kind": "unknown"}}
        schema_doc = {"schema_version": 2, "mapping_version": 1,
                      "datasets": [{"dataset": dataset, "edition": "2025", "source_only": True, "tables": [table]}]}
        resource = tmp_path / (family + "-schema.json")
        resource.write_text(canonical_json(schema_doc))
        schema_sha = file_identity(resource)["sha256"]
        schemas[schema_sha] = resource.read_text()
        datasets[dataset] = {2025: {"path": str(raw), "metadata": {
            "version": "2025", "value_policy": "numeric", "metrics": {
                "amount": {"field": "amount", "unit": "count", "period": {"kind": "calendar_year", "year": 2024}}}},
            "reader": {"profiles": {"source": {"column_schema": resource.name,
                "references": [{"reference_id": "columns", "role": "schema", "path": resource.name,
                    "reviewed_sha256": schema_sha}]}}}}}
        specs.append((family, dataset, raw, raw_sha, table))
    cfg = Config.from_dict({"data_sources": datasets, "value_policies": {"numeric": {}}})
    cfg.base_dir = tmp_path
    recorder = BuildRecorder(cfg, store)
    leaves = {}
    for family, dataset, raw, raw_sha, table in specs:
        source = cfg.resolve_dataset(dataset, 2025)
        acquisition = store.record_acquisition(raw, source)
        key = EditionKey(dataset, "2025", EditionKey.source_revision((raw_sha,)), "campus", family, "one row per campus")
        native = pa.schema([pa.field("campus_number", pa.string(), nullable=False),
                            pa.field("campus_name", pa.string()), pa.field("amount", pa.string())])
        recipe = {"operation": "tiny native fixture", "compression": "snappy"}
        toolchain = {"python": sys.version, "pyarrow": version("pyarrow")}
        contract = recorder.leaf_contract(key, source, native_schema=native.serialize().to_pybytes(),
            columns=native.names, key_columns=["campus_number"], rows=1, inputs=[raw_sha], recipe=recipe, toolchain=toolchain)
        leaf_id = sha256(canonical_json(contract).encode()).hexdigest()
        leaf = tmp_path / (family + ".parquet")
        pq.write_table(pa.Table.from_arrays([pa.array(["001"]), pa.array(["One"]), pa.array(["5"])],
            schema=native.with_metadata({LEAF_METADATA: canonical_json({"schema_version": 3, "leaf_contract_sha256": leaf_id}).encode()})), leaf)
        processing = store.record_processing(dataset=dataset, inputs=[acquisition], outputs={"leaf": leaf},
            processor="native-fixture", code_version=recorder.processor_sha256,
            parameters={"leaf_contract_sha256": leaf_id, "recipe": recipe, "toolchain": toolchain})
        identity = file_identity(leaf)
        ref = ArtifactRef(key, identity["sha256"], identity["size_bytes"], leaf.name,
                          canonical_json(contract), processing, "leaf")
        recorder.record_edition(key, source, artifacts=[ref], acquisition_ids=[acquisition])
        leaves[family] = (key, ref, leaf)
    catalog = recorder.catalog(schema_version=3)
    root = tmp_path / "root.json"
    root.write_text(canonical_json({"schema_version": 3, "catalog": catalog.to_dict()}))
    return {"catalog": catalog, "root": root, "leaves": leaves, "recorder": recorder, "store": store}


def rehash_catalog(document):
    document["build_id"] = sha256(canonical_json({k: v for k, v in document.items() if k != "build_id"}).encode()).hexdigest()
    return DatasetCatalog(document)


def test_edition_index_resolves_explicit_keys_periods_dependencies_and_detaches(edition_fixture, monkeypatch):
    catalog = edition_fixture["catalog"]
    key, ref, _ = edition_fixture["leaves"]["student"]
    with monkeypatch.context() as forbidden:
        forbidden.setattr(Path, "open", lambda *a, **k: pytest.fail("discovery file I/O"))
        forbidden.setattr("teadata.manifest.json.loads", lambda *a, **k: pytest.fail("catalog reparse"))
        assert len(list(catalog.editions())) == 2
        fields = list(catalog.variables(edition=key))
        fields[0]["period"]["kind"] = "caller-only"
        assert catalog.field_metadata("amount", edition=key)["unit"] == "count"
    selected, = catalog.resolve_artifacts([(key, "amount")], period=DataPeriod(kind="calendar_year", year=2024))
    assert selected["artifact"] == ref
    assert selected["columns"] == ("campus_number", "amount")  # name is not the identity key
    with pytest.raises(ContractError, match="period"):
        catalog.resolve_artifacts([(key, "amount")], period=DataPeriod(kind="calendar_year", year=2025))
    with pytest.raises(ContractError, match="ambiguous"):
        catalog.field_metadata("amount")
    assert catalog.field_metadata("amount", edition=key)["period"]["year"] == 2024
    restored = pickle.loads(pickle.dumps(catalog))
    assert restored.to_dict() == catalog.to_dict() and restored._field_resolutions == {}


def test_schema2_complete_discovery_is_explicitly_unavailable(selected_catalog):
    with pytest.raises(ContractError, match="unavailable"):
        list(selected_catalog.variables())
    with pytest.raises(ContractError, match="unavailable"):
        selected_catalog.resolve_artifacts([])


@pytest.mark.parametrize("damage", ["missing_field", "duplicate_field", "cycle", "foreign_column", "policy", "derivation", "revision", "key"])
def test_v3_well_hashed_contradictions_fail_before_io(edition_fixture, damage, monkeypatch):
    document = edition_fixture["catalog"].to_dict()
    fields = document["field_bindings"]
    artifact_id = fields[0]["artifact_id"]
    ref = document["artifacts"][artifact_id]
    if damage == "missing_field":
        fields.pop()
    elif damage == "duplicate_field":
        fields.append(deepcopy(fields[0]))
    elif damage == "cycle":
        fields[0]["dependencies"] = [[fields[0]["edition_id"], fields[0]["field"]]]
    elif damage == "foreign_column":
        fields[0]["column"] = "campus_name"
    elif damage == "policy":
        document["value_policies"]["numeric"]["tokens"] = {"-1": {"status": "missing"}}
    elif damage == "derivation":
        ref["processing_id"] = next(iter(document["provenance"]))
    elif damage == "revision":
        document["editions"][fields[0]["edition_id"]]["key"]["revision_sha256"] = "f" * 64
    else:
        ref["contract"]["key_columns"] = ["unknown"]
    with monkeypatch.context() as forbidden:
        forbidden.setattr(Path, "open", lambda *a, **k: pytest.fail("validation file I/O"))
        with pytest.raises(ContractError):
            rehash_catalog(document)


def test_new_root_metadata_reuses_leaf_but_transform_inputs_change_contract(edition_fixture):
    catalog = edition_fixture["catalog"]
    changed = catalog.to_dict()
    changed["description"] = "new root-only editorial metadata"
    newer = rehash_catalog(changed)
    key, ref, _ = edition_fixture["leaves"]["student"]
    assert newer.build_id != catalog.build_id and newer.artifact_ref(ref.identity) == ref
    assert key.revision_sha256 == EditionKey.source_revision(tuple(ref.contract["inputs"]))
    for field in ("recipe_sha256", "mapping_sha256", "policy_sha256"):
        contract = ref.contract
        contract[field] = "f" * 64
        altered = replace(ref, contract_json=canonical_json(contract))
        assert altered.identity != ref.identity and altered.edition == ref.edition
    assert EditionKey.source_revision(("f" * 64,)) != key.revision_sha256


@pytest.fixture
def catalog_http(tmp_path):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            target = tmp_path / self.path.lstrip("/")
            if not target.is_file():
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(target.read_bytes())

        def log_message(self, format, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield "http://127.0.0.1:" + str(server.server_port), requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_pinned_root_selected_transport_second_root_and_offline(edition_fixture, catalog_http, monkeypatch, tmp_path):
    from teadata.release import DataRelease
    base, requests = catalog_http
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    root = edition_fixture["root"]
    key, student, _ = edition_fixture["leaves"]["student"]
    _, staff, _ = edition_fixture["leaves"]["staff"]
    url = base + "/root.json"
    pin = file_identity(root)["sha256"]
    selected = DataRelease(url, expected_sha256=pin)
    assert requests == ["/root.json"]
    assert len(list(selected.catalog.variables())) == 6 and requests == ["/root.json"]
    assert selected.analytical_dataset(student.identity).to_table(columns=["amount"]).to_pydict() == {"amount": ["5"]}
    assert requests == ["/root.json", "/student.parquet"]
    offline = DataRelease(url, expected_sha256=pin, offline=True)
    assert offline.artifact(student.identity).is_file()
    with pytest.raises(ContractError, match="not cached"):
        offline.artifact(staff.identity)
    selected.artifact(staff.identity)
    assert requests[-1] == "/staff.parquet" and len(requests) == 3
    document = selected.catalog.to_dict()
    for ref in document["artifacts"].values():
        ref["location"] = base + "/" + ref["location"]
    document["description"] = "second root referencing old-release URLs"
    newer = rehash_catalog(document)
    second = tmp_path / "second.json"
    second.write_text(canonical_json({"schema_version": 3, "catalog": newer.to_dict()}))
    next_release = DataRelease(base + "/second.json", expected_sha256=file_identity(second)["sha256"])
    next_release.artifact(student.identity)
    assert requests[-1] == "/second.json" and len(requests) == 4
    with pytest.raises(ContractError, match="period"):
        next_release.catalog.resolve_artifacts([(key, "amount")], period=DataPeriod(kind="calendar_year", year=2026))
    cached_root = offline.path
    cached_root.write_bytes(b"corrupt")
    with pytest.raises(ContractError, match="trusted pin"):
        DataRelease(url, expected_sha256=pin, offline=True)
    cached_root.unlink()
    with pytest.raises(FileNotFoundError):
        DataRelease(url, expected_sha256=pin, offline=True)
    assert len(requests) == 4


def test_cached_sha_never_grants_foreign_descriptor_footer_acceptance(edition_fixture):
    from teadata.release import DataRelease
    selected = DataRelease(edition_fixture["root"])
    _, ref, _ = edition_fixture["leaves"]["student"]
    selected.artifact(ref.identity)
    contract = ref.contract
    contract["recipe_sha256"] = "f" * 64
    foreign = replace(ref, contract_json=canonical_json(contract))
    selected.catalog._artifact_refs[foreign.identity] = foreign  # adversarial descriptor after a physical-cache hit
    with pytest.raises(ContractError, match="selected leaf"):
        selected.artifact(foreign.identity)


def test_v3_fresh_process_metadata_loading_imports_no_native_or_engine(edition_fixture):
    script = """
import json, pathlib, sys
from teadata.release import DataRelease
release = DataRelease(sys.argv[1])
pathlib.Path.open = lambda *a, **k: (_ for _ in ()).throw(AssertionError('I/O after bootstrap'))
assert len(list(release.catalog.variables())) == 6
assert len(list(release.catalog.editions())) == 2
assert not {'pyarrow', 'duckdb', 'teadata.engine', 'sqlalchemy'} & set(sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", script, str(edition_fixture["root"])],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_valid_pure_graph_and_dependency_resolution_need_no_artifact_io(edition_fixture, monkeypatch):
    document = edition_fixture["catalog"].to_dict()
    student, _, _ = edition_fixture["leaves"]["student"]
    staff, _, _ = edition_fixture["leaves"]["staff"]
    amount = next(row for row in document["field_bindings"] if row["edition_id"] == student.identity and row["field"] == "amount")
    amount["dependencies"] = [[staff.identity, "amount"]]
    with monkeypatch.context() as forbidden:
        forbidden.setattr(Path, "open", lambda *a, **k: pytest.fail("pure graph opened artifact"))
        catalog = rehash_catalog(document)
        selected = catalog.resolve_artifacts([(student, "amount")])
        assert [row["artifact"].edition for row in selected] == [student, staff]
        assert all(row["columns"] == ("campus_number", "amount") for row in selected)
        assert len(catalog.header_records(student.dataset, edition_key=student)) == 3
        assert catalog.acquisitions(student.dataset, edition=student)[0]["downloaded_at"] is None
        with pytest.raises(ContractError, match="period"):
            catalog.resolve_artifacts([(student, "campus_number")], period=DataPeriod(kind="calendar_year", year=2025))


def test_partial_download_and_changed_cached_leaf_fail_without_fallback(edition_fixture, catalog_http, monkeypatch, tmp_path):
    from teadata.release import DataRelease
    base, requests = catalog_http
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    selected = DataRelease(base + "/root.json", expected_sha256=file_identity(edition_fixture["root"])["sha256"])
    _, student, _ = edition_fixture["leaves"]["student"]
    _, staff, staff_path = edition_fixture["leaves"]["staff"]
    staff_path.write_bytes(staff_path.read_bytes()[:32])
    with pytest.raises(ContractError, match="checksum or size"):
        selected.artifact(staff.identity)
    assert not (selected.cache / "objects" / staff.sha256).exists()
    assert list((selected.cache / "objects").iterdir()) == []
    path = selected.artifact(student.identity)
    path.write_bytes(b"corrupt")
    with pytest.raises(ContractError, match="checksum or size"):
        selected.artifact(student.identity)
    assert requests == ["/root.json", "/staff.parquet", "/student.parquet"]


def test_stream_download_accepts_legacy_full_identity_and_root_cache_keys_include_trust_options(edition_fixture, catalog_http, monkeypatch, tmp_path):
    from teadata.release import _download, _load_release
    base, requests = catalog_http
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    _, _, leaf = edition_fixture["leaves"]["student"]
    target = tmp_path / "downloaded.parquet"
    expected = file_identity(leaf)
    _download(base + "/student.parquet", target, expected=expected)  # legacy descriptor includes filename
    actual = file_identity(target)
    assert (actual["sha256"], actual["size_bytes"]) == (expected["sha256"], expected["size_bytes"])
    url, pin = base + "/root.json", file_identity(edition_fixture["root"])["sha256"]
    _load_release.cache_clear()
    try:
        online = _load_release(url, 0, pin, False)
        offline = _load_release(url, 0, pin, True)
        assert online is not offline and offline.offline is True
        with pytest.raises(ContractError, match="trusted pin"):
            _load_release(url, 0, "f" * 64, False)
        with pytest.raises(ContractError, match="expected manifest"):
            _load_release(url, 0, None, True)
        assert requests == ["/student.parquet", "/root.json"]
    finally:
        _load_release.cache_clear()


def test_native_footer_preserves_order_type_nullability_and_no_global_metadata(edition_fixture, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from teadata.analytical import validate_leaf_artifact
    _, ref, path = edition_fixture["leaves"]["student"]
    table = pq.read_table(path)
    for damage in ("nullable", "global", "order"):
        if damage == "nullable":
            schema = pa.schema([pa.field(field.name, field.type) for field in table.schema], metadata=table.schema.metadata)
            changed = table.cast(schema)
        elif damage == "global":
            changed = table.replace_schema_metadata({**table.schema.metadata, b"teadata.build_id": b"old root"})
        else:
            changed = table.select(["campus_name", "campus_number", "amount"])
        destination = tmp_path / (damage + ".parquet")
        pq.write_table(changed, destination)
        with pytest.raises(ContractError, match="selected leaf"):
            validate_leaf_artifact(destination, ref)


def test_exact_same_dataset_revisions_never_choose_latest_or_share_field_cache(edition_fixture, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from teadata.analytical import LEAF_METADATA
    from teadata.dataset_metadata import DatasetMetadata, ResolvedDataset

    catalog, store = edition_fixture["catalog"], edition_fixture["store"]
    original, ref, leaf = edition_fixture["leaves"]["student"]
    document = catalog.to_dict()
    old_entry = document["editions"][original.identity]
    resource = json.loads(document["column_schemas"][old_entry["column_schema_sha256"]])
    raw = tmp_path / "reissued.csv"
    raw.write_text("ID,NAME,VALUE\n001,One,7\n")
    raw_sha = file_identity(raw)["sha256"]
    metadata = DatasetMetadata.from_dict(old_entry["metadata"])
    metadata = replace(metadata, metrics={"amount": replace(metadata.metrics["amount"], unit="percent")})
    acquisition = store.record_acquisition(raw, ResolvedDataset(original.dataset, "data_sources", 2025, 2025, str(raw), metadata))
    key = replace(original, revision_sha256=EditionKey.source_revision((raw_sha,)))
    table = resource["datasets"][0]["tables"][0]
    table["input_sha256"] = raw_sha
    text = canonical_json(resource)
    schema_sha = sha256(text.encode()).hexdigest()
    contract = ref.contract
    contract.update(edition=key.to_dict(), inputs=[raw_sha], mapping_sha256=sha256(canonical_json(table).encode()).hexdigest())
    from teadata.column_schema import header_records
    from teadata.manifest import _leaf_semantics
    contract["semantics_sha256"] = _leaf_semantics(metadata, header_records(resource,
        dataset=key.dataset, edition=key.edition, table=key.table, metadata=metadata))
    identity = sha256(canonical_json(contract).encode()).hexdigest()
    native = pq.read_table(leaf).replace_schema_metadata({LEAF_METADATA: canonical_json({"schema_version": 3, "leaf_contract_sha256": identity}).encode()})
    output = tmp_path / "reissued.parquet"
    pq.write_table(native.set_column(2, "amount", pa.array(["7"])), output)
    old_processing = store.read_record(ref.processing_id)
    processing = store.record_processing(dataset=key.dataset, inputs=[acquisition], outputs={"leaf": output},
        processor=old_processing["processor"], code_version=old_processing["code_version"],
        parameters={**old_processing["parameters"], "leaf_contract_sha256": identity})
    actual = file_identity(output)
    newer = ArtifactRef(key, actual["sha256"], actual["size_bytes"], output.name, canonical_json(contract), processing, "leaf")
    document["column_schemas"][schema_sha] = text
    document["editions"][key.identity] = {"key": key.to_dict(), "metadata": metadata.to_dict(),
        "acquisition_ids": [acquisition], "column_schema_sha256": schema_sha}
    document["artifacts"][identity] = newer.to_dict()
    document["field_bindings"].extend({**row, "edition_id": key.identity, "artifact_id": identity}
        for row in list(document["field_bindings"]) if row["edition_id"] == original.identity)
    document["provenance"].update(store.verify_record_closure([processing]))
    selected = rehash_catalog(document)
    with pytest.raises(ContractError, match="ambiguous"):
        selected.metadata(key.dataset)
    with pytest.raises(ContractError, match="ambiguous"):
        selected.policy(key.dataset)
    assert selected.metadata(key.dataset, edition=key).metrics["amount"].unit == "percent"
    assert selected.field_metadata("amount", edition=original)["unit"] == "count"
    assert selected.field_metadata("amount", edition=key)["unit"] == "percent"
    assert len(selected._qualified_fields) == 2
    with pytest.raises(ContractError, match="ambiguous"):
        selected.header_records(key.dataset, edition="2025")
    assert selected.header_records(key.dataset, edition_key=key)[0]["input_sha256"] == raw_sha


def test_many_year_metadata_envelope_is_measured_without_native_imports(tmp_path, record_property):
    """Synthetic registered metadata, not a claim of built historical leaves."""
    import base64
    import pyarrow as pa
    from teadata.column_schema import header_records, source_schema_fingerprint
    from teadata.dataset_metadata import DatasetMetadata
    from teadata.manifest import _leaf_semantics

    editions, artifacts, bindings, records, schemas = {}, {}, [], {}, {}

    def receipt(value):
        identifier = sha256((canonical_json(value) + "\n").encode()).hexdigest()
        records[identifier] = value
        return identifier

    for year in range(2015, 2027):
        for scope in ("campus", "district"):
            columns = [scope + "_number", scope + "_name", *(f"value_{i:03}" for i in range(198))]
            ipc = pa.schema([(column, pa.string()) for column in columns]).serialize().to_pybytes()
            for family in ("student", "staff"):
                dataset = scope + "_tapr_" + family + "_profile"
                raw_sha = sha256((dataset + str(year)).encode()).hexdigest()
                metadata = DatasetMetadata(version=str(year))
                key = EditionKey(dataset, str(year), EditionKey.source_revision((raw_sha,)), scope, family, "one row per " + scope)
                fields = [{"position": i, "code": column.upper(), "label": column, "field": column,
                    "disposition": "identity" if i < 3 else "fact", "publisher_type": None, "publisher_description": None,
                    "dictionary_status": "not_applicable" if i < 3 else "not_documented"} for i, column in enumerate(columns, 1)]
                table = {"table": family, "input_sha256": raw_sha, "fields": fields, "metric_period": {"kind": "unknown"},
                    "source_schema_sha256": source_schema_fingerprint(columns, [column.upper() for column in columns])}
                resource = {"schema_version": 2, "mapping_version": 1, "datasets": [{"dataset": dataset,
                    "edition": str(year), "source_only": True, "tables": [table]}]}
                text = canonical_json(resource)
                schema_sha = sha256(text.encode()).hexdigest()
                schemas[schema_sha] = text
                acquisition = receipt({"schema_version": 1, "kind": "acquisition", "dataset": dataset,
                    "section": "data_sources", "requested_year": year, "resolved_year": year, "source_metadata": metadata.to_dict(),
                    "artifact": {"sha256": raw_sha, "size_bytes": 1, "filename": "synthetic-unopened.csv"}})
                recipe, toolchain = {"operation": "synthetic metadata only"}, {"python": "synthetic"}
                contract = {"edition": key.to_dict(), "schema": base64.b64encode(ipc).decode(), "schema_sha256": sha256(ipc).hexdigest(),
                    "columns": columns, "key_columns": [scope + "_number"], "rows": 1, "inputs": [raw_sha],
                    "mapping_sha256": sha256(canonical_json(table).encode()).hexdigest(), "policy_sha256": None,
                    "semantics_sha256": _leaf_semantics(metadata, header_records(resource, dataset=dataset, edition=str(year), metadata=metadata)),
                    "recipe_sha256": sha256(canonical_json(recipe).encode()).hexdigest(),
                    "producer": {"code_sha256": "2" * 64, "toolchain": toolchain}}
                leaf_id = sha256(canonical_json(contract).encode()).hexdigest()
                physical_sha = sha256((leaf_id + "declared synthetic output").encode()).hexdigest()
                processing = receipt({"schema_version": 1, "kind": "processing", "dataset": dataset,
                    "inputs": [acquisition], "processor": "synthetic", "code_version": "2" * 64,
                    "parameters": {"leaf_contract_sha256": leaf_id, "recipe": recipe, "toolchain": toolchain},
                    "outputs": {"leaf": {"sha256": physical_sha, "size_bytes": 1024, "filename": "synthetic-unopened.parquet"}}})
                ref = ArtifactRef(key, physical_sha, 1024, "synthetic-unopened.parquet", canonical_json(contract), processing, "leaf")
                editions[key.identity] = {"key": key.to_dict(), "metadata": metadata.to_dict(),
                    "acquisition_ids": [acquisition], "column_schema_sha256": schema_sha}
                artifacts[leaf_id] = ref.to_dict()
                bindings.extend({"edition_id": key.identity, "field": column, "artifact_id": leaf_id,
                                 "column": column, "dependencies": []} for column in columns)
    content = {"schema_version": 3, "editions": editions, "artifacts": artifacts, "field_bindings": bindings,
               "column_schemas": schemas, "provenance": records, "value_policies": {}, "description": "synthetic metadata envelope"}
    document = {**content, "build_id": sha256(canonical_json(content).encode()).hexdigest()}
    raw = canonical_json({"schema_version": 3, "catalog": document}).encode()
    path = tmp_path / "many-years.json"
    path.write_bytes(raw)
    script = """
import json, pathlib, resource, sys, tracemalloc
from teadata.manifest import DatasetCatalog
from teadata.release import DataRelease
from teadata.manifest import ContractError
raw = pathlib.Path(sys.argv[1]).read_bytes()
document = json.loads(raw)['catalog']
tracemalloc.start()
catalog = DatasetCatalog(document)
peak = tracemalloc.get_traced_memory()[1]
tracemalloc.stop()
assert not {'pyarrow', 'duckdb', 'teadata.engine', 'sqlalchemy'} & set(sys.modules)
over_cap = len(raw) > 4 * 1024 * 1024
try:
    DataRelease(sys.argv[1])
except ContractError as e:
    assert over_cap and 'too large' in str(e)
else:
    assert not over_cap
print(json.dumps({'root_bytes':len(raw), 'editions':len(list(catalog.editions())),
 'fields':sum(1 for _ in catalog.variables()), 'validation_incremental_tracemalloc_peak_bytes':peak,
 'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
 'root_exceeds_unchanged_4MiB_cap':over_cap, 'native_imports':False}))
"""
    result = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["editions"] == 48 and observed["fields"] == 9600
    record_property("synthetic_metadata_measurement", canonical_json(observed))


def catalog_with_methodology(edition_fixture, tmp_path, *, suppression=False):
    """Genuine tiny native model leaf and retained method; no scientific data claim."""
    import pyarrow.parquet as pq
    from teadata.analytical import LEAF_METADATA
    from teadata.column_schema import header_records
    from teadata.dataset_metadata import DatasetMetadata, ResolvedDataset
    from teadata.manifest import _leaf_semantics
    from teadata.values import ValuePolicy

    document = edition_fixture["catalog"].to_dict()
    store = edition_fixture["store"]
    key, ref, leaf = edition_fixture["leaves"]["student"]
    entry = document["editions"][key.identity]
    method = tmp_path / "reviewed-method.txt"
    method.write_text("Explicit synthetic policy/model methodology")
    method_sha = file_identity(method)["sha256"]
    method_id = store.record_acquisition(method, ResolvedDataset("method", "reference", None, None, str(method), DatasetMetadata()))
    metadata = DatasetMetadata.from_dict({**entry["metadata"],
        "references": [{"reference_id": "method", "role": "methodology", "path": method.name,
                        "status": "available", "reviewed_sha256": method_sha, "provenance_ids": [method_id]}],
        "estimation": {"model_id": "synthetic", "value_per_suppressed": 5,
                       "grain": "origin_campus_destination_campus", "description": "Synthetic test model", "reference_id": "method"}})
    policy = ValuePolicy.from_dict({"policy_id": "numeric", "reference_id": "method" if suppression else None,
        "tokens": {"-1": {"status": "suppressed", "reason": "synthetic suppression"}} if suppression else {}})
    contract = ref.contract
    schema = json.loads(document["column_schemas"][entry["column_schema_sha256"]])
    contract["semantics_sha256"] = _leaf_semantics(metadata,
        header_records(schema, dataset=key.dataset, edition=key.edition, table=key.table, metadata=metadata))
    contract["policy_sha256"] = sha256(canonical_json(policy.to_dict()).encode()).hexdigest()
    identity = sha256(canonical_json(contract).encode()).hexdigest()
    native = pq.read_table(leaf).replace_schema_metadata({LEAF_METADATA: canonical_json({"schema_version": 3, "leaf_contract_sha256": identity}).encode()})
    output = tmp_path / "method-reviewed.parquet"
    pq.write_table(native, output)
    previous = store.read_record(ref.processing_id)
    processing = store.record_processing(dataset=key.dataset, inputs=[*previous["inputs"], method_id], outputs={"leaf": output},
        processor=previous["processor"], code_version=previous["code_version"],
        parameters={**previous["parameters"], "leaf_contract_sha256": identity})
    actual = file_identity(output)
    selected = ArtifactRef(key, actual["sha256"], actual["size_bytes"], output.name, canonical_json(contract), processing, "leaf")
    return DatasetCatalog.create_editions(editions={key.identity: {**entry, "metadata": metadata.to_dict()}},
        artifacts={identity: selected.to_dict()},
        field_bindings=[{**value, "artifact_id": identity} for value in document["field_bindings"] if value["edition_id"] == key.identity],
        column_schemas=document["column_schemas"], records=store.verify_record_closure([processing]),
        value_policies={"numeric": policy.to_dict()})


@pytest.mark.parametrize("damage", ["suppression_without_reference", "unreviewed", "not_found", "no_review_sha", "no_receipt"])
def test_v3_methodology_states_cannot_claim_qualified_policy_or_model(edition_fixture, tmp_path, monkeypatch, damage):
    catalog = catalog_with_methodology(edition_fixture, tmp_path, suppression=damage == "suppression_without_reference")
    document = catalog.to_dict()
    entry, = document["editions"].values()
    if damage == "suppression_without_reference":
        document["value_policies"]["numeric"]["reference_id"] = None
    else:
        method, = entry["metadata"]["references"]
        if damage in {"unreviewed", "not_found"}:
            method["status"] = damage
        elif damage == "no_review_sha":
            method["reviewed_sha256"] = None
        else:
            method["provenance_ids"] = []
    with monkeypatch.context() as forbidden:
        forbidden.setattr(Path, "open", lambda *a, **k: pytest.fail("methodology validation opened artifact"))
        with pytest.raises(ContractError, match="methodology|acquisition receipt"):
            rehash_catalog(document)


def test_v3_genuinely_reviewed_model_and_policy_metadata_remain_accepted(edition_fixture, tmp_path):
    from teadata.release import DataRelease
    catalog = catalog_with_methodology(edition_fixture, tmp_path, suppression=True)
    key, = catalog._edition_keys.values()
    assert catalog.metadata(key.dataset).estimation.value_per_suppressed == 5
    assert catalog.policy(key.dataset).tokens["-1"].status.value == "suppressed"
    path = tmp_path / "reviewed-root.json"
    path.write_text(canonical_json({"schema_version": 3, "catalog": catalog.to_dict()}))
    selected = DataRelease(path)
    ref, = catalog._artifact_refs.values()
    assert selected.analytical_dataset(ref.identity).to_table(columns=["amount"]).to_pydict() == {"amount": ["5"]}


@pytest.mark.parametrize("entrypoint", ["charter_transfer_summary", "measurement", "measure_field", "measurements_frame"])
def test_v3_scalar_refusal_precedes_all_foreign_entity_and_repository_access(edition_fixture, monkeypatch, entrypoint):
    catalog = edition_fixture["catalog"]

    class ForeignEntity:
        def __getattribute__(self, name):
            pytest.fail("Foreign legacy entity/repository touched before refusal")

        def __iter__(self):
            pytest.fail("Foreign entity iterable consumed before refusal")
    entity = ForeignEntity()
    monkeypatch.setattr(catalog, "policy", lambda *a, **k: pytest.fail("policy touched before refusal"))
    calls = {
        "charter_transfer_summary": lambda: catalog.charter_transfer_summary(entity),
        "measurement": lambda: catalog.measurement(entity, "amount"),
        "measure_field": lambda: catalog.measure_field(7, "amount", entity=entity),
        "measurements_frame": lambda: catalog.measurements_frame(entity, ["amount"]),
    }
    with pytest.raises(ContractError, match="Schema3 scalar interpretation is unavailable"):
        calls[entrypoint]()


@pytest.mark.parametrize("root_version", [1, 2])
def test_legacy_roots_reject_catalog3_before_artifact_io(edition_fixture, tmp_path, monkeypatch, root_version):
    from teadata.release import ANALYTICAL_RELEASE_ROLES, DataRelease, ROLES
    catalog = edition_fixture["catalog"]
    roles = ROLES if root_version == 1 else ANALYTICAL_RELEASE_ROLES
    outputs = {role: {"filename": role + ".bin", "sha256": "e" * 64, "size_bytes": 1} for role in roles}
    record = {"schema_version": 1, "kind": "processing", "parameters": {"build_id": catalog.build_id}, "outputs": outputs}
    identity = sha256((canonical_json(record) + "\n").encode()).hexdigest()
    path = tmp_path / "mixed-root.json"
    path.write_text(canonical_json({"schema_version": root_version, "catalog": catalog.to_dict(), "artifacts": outputs,
                                   "processing": {"record_id": identity, "record": record}}))
    monkeypatch.setattr("teadata.release.read_sqlite_catalog", lambda *a: pytest.fail("legacy artifact opened"))
    monkeypatch.setattr("teadata.release.validate_artifact", lambda *a, **k: pytest.fail("legacy footer opened"))
    with pytest.raises(ContractError, match="Legacy release roots require a schema2 catalog"):
        DataRelease(path)


def test_root3_catalog2_and_legacy_publisher_catalog3_fail_without_publication(selected_catalog, edition_fixture, tmp_path, monkeypatch):
    from teadata.release import DataRelease, ROLES, publish_release
    path = tmp_path / "reverse-mixed.json"
    path.write_text(canonical_json({"schema_version": 3, "catalog": selected_catalog.to_dict()}))
    with pytest.raises(ContractError, match="Schema3 roots"):
        DataRelease(path)
    missing = tmp_path / "missing-catalog.json"
    missing.write_text(canonical_json({"schema_version": 1, "artifacts": {role: {} for role in ROLES}}))
    with pytest.raises(KeyError, match="catalog"):
        DataRelease(missing)  # keep the legacy missing-catalog error boundary unchanged
    output = tmp_path / "must-not-be-created"
    with monkeypatch.context() as forbidden:
        forbidden.setattr(Path, "mkdir", lambda *a, **k: pytest.fail("legacy publisher wrote before refusing format"))
        with pytest.raises(ContractError, match="Legacy release publication requires a schema2 catalog"):
            publish_release(output, edition_fixture["catalog"], {role: path for role in ROLES}, processing={})
    assert not output.exists()


@pytest.mark.parametrize("edge", ["source_metadata", "reference"])
def test_processing_dependencies_are_explicit_and_never_acquisition_metadata_edges(edition_fixture, tmp_path, monkeypatch, edge):
    from teadata.dataset_metadata import ReferenceDocument
    from teadata.provenance import validate_record_graph, validate_reference_records

    store = edition_fixture["store"]
    key, ref, _ = edition_fixture["leaves"]["student"]
    source = edition_fixture["recorder"].cfg.resolve_dataset(key.dataset, 2025)
    original = store.read_record(ref.processing_id)["inputs"][0]
    annotation = tmp_path / "typed-annotation.json"
    annotation.write_text('{"explicit":"dependency"}')
    processing = store.record_processing(dataset="annotation", inputs=[original, ref.processing_id],
        outputs={"annotation": annotation}, processor="test-explicit-processing", code_version="test")
    closed = store.verify_record_closure([processing])
    assert closed[processing]["kind"] == "processing" and closed[original]["kind"] == "acquisition"
    if edge == "source_metadata":
        metadata = replace(source.metadata, provenance_ids=(processing,))
    else:
        metadata = replace(source.metadata, references=(ReferenceDocument.from_dict({
            "reference_id": "annotation", "role": "source_notes", "status": "unreviewed", "provenance_ids": [processing]}),))
    invalid_source = replace(source, metadata=metadata)
    invalid = store.record_acquisition(source.path, invalid_source)
    graph = {**closed, invalid: store.read_record(invalid)}
    with monkeypatch.context() as forbidden:
        forbidden.setattr(Path, "open", lambda *a, **k: pytest.fail("pure typed graph opened an artifact"))
        assert validate_record_graph(closed.__getitem__, [processing])[processing]["inputs"] == [original, ref.processing_id]
        with pytest.raises(ContractError, match="acquisition"):
            validate_record_graph(graph.__getitem__, [invalid])
        with pytest.raises(ContractError, match="acquisition"):
            validate_reference_records(invalid_source, graph.__getitem__)  # also applies to edition metadata roots
    monkeypatch.setattr(store, "verify_artifact", lambda *a: pytest.fail("physical artifacts checked before typed-edge refusal"))
    with pytest.raises(ContractError, match="acquisition"):
        store.verify_record_closure([invalid])


@pytest.mark.parametrize("edge", ["source_metadata", "reference"])
def test_edition_metadata_roots_use_the_same_acquisition_only_gate(edition_fixture, monkeypatch, edge):
    document = edition_fixture["catalog"].to_dict()
    key, ref, _ = edition_fixture["leaves"]["student"]
    metadata = document["editions"][key.identity]["metadata"]
    if edge == "source_metadata":
        metadata["provenance_ids"] = [ref.processing_id]
    else:
        metadata["references"] = [{"reference_id": "annotation", "role": "source_notes",
            "status": "unreviewed", "provenance_ids": [ref.processing_id]}]
    # The embedded record graph is closed and typed on its own. It is this
    # edition metadata edge, not missing history or file I/O, that must fail.
    with monkeypatch.context() as forbidden:
        forbidden.setattr(Path, "open", lambda *a, **k: pytest.fail("edition metadata opened an artifact"))
        with pytest.raises(ContractError, match="acquisition"):
            rehash_catalog(document)
