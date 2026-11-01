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
from importlib.metadata import version
from io import BytesIO
from urllib.parse import urlparse

import pytest

from teadata.engine import DataEngine
from teadata.manifest import ContractError, DatasetCatalog, canonical_json
from teadata.release import SNAPSHOT_MAGIC
from teadata.dataset_metadata import ArtifactRef, ColumnSchemaRef, DataPeriod, DatasetMetadata, EditionKey, EditionMetadataRef, ReferenceDocument, ResolvedDataset
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
    from teadata.analytical import write_leaf_artifact
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
        schemas[schema_sha] = resource.read_bytes()
        schema_observation = store.record_acquisition(resource, ResolvedDataset(dataset + ":schema", "reference", None, None,
            str(resource), DatasetMetadata(title="Synthetic retained crosswalk")))
        (tmp_path / (schema_sha + ".json")).write_bytes(schemas[schema_sha])
        datasets[dataset] = {2025: {"path": str(raw), "metadata": {
            "version": "2025", "value_policy": "numeric", "metrics": {
                "amount": {"field": "amount", "unit": "count", "period": {"kind": "calendar_year", "year": 2024}}}},
            "reader": {"profiles": {"source": {"column_schema": resource.name,
                "references": [{"reference_id": "columns", "role": "schema", "path": resource.name,
                    "reviewed_sha256": schema_sha, "status": "available", "provenance_ids": [schema_observation]}]}}}}}
        profile = datasets[dataset][2025]["reader"]["profiles"]["source"]
        datasets[dataset][2025]["metadata"]["references"] = profile.pop("references")
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
        recipe = {"operation": "tiny native fixture", "compression": "zstd"}
        toolchain = {"python": sys.version, "pyarrow": version("pyarrow")}
        contract = recorder.leaf_contract(key, source, native_schema=native.serialize().to_pybytes(),
            columns=native.names, key_columns=["campus_number"], rows=1, inputs=[raw_sha], recipe=recipe, toolchain=toolchain)
        leaf_id = sha256(canonical_json(contract).encode()).hexdigest()
        leaf = tmp_path / (family + ".parquet")
        write_leaf_artifact(leaf, pa.Table.from_arrays([pa.array(["001"]), pa.array(["One"]), pa.array(["5"])],
            schema=native).to_batches(), contract=contract)
        processing = store.record_processing(dataset=dataset, inputs=[acquisition], outputs={"leaf": leaf},
            processor="native-fixture", code_version=recorder.processor_sha256,
            parameters={"leaf_contract_sha256": leaf_id, "recipe": recipe, "toolchain": toolchain})
        identity = file_identity(leaf)
        ref = ArtifactRef(key, identity["sha256"], identity["size_bytes"], leaf.name,
                          canonical_json(contract), processing, "leaf")
        recorder.record_edition(key, source, artifacts=[ref], acquisition_ids=[acquisition])
        leaves[family] = (key, ref, leaf)
    catalog = recorder.catalog(schema_version=3)
    catalog.verify_metadata()
    root = tmp_path / "root.json"
    root.write_text(canonical_json({"schema_version": 3, "catalog": catalog.to_dict()}))
    return {"catalog": catalog, "root": root, "leaves": leaves, "recorder": recorder, "store": store, "resources": schemas}


def rehash_catalog(document, resources=None):
    document["build_id"] = sha256(canonical_json({k: v for k, v in document.items() if k != "build_id"}).encode()).hexdigest()
    return DatasetCatalog(document, column_schema_loader=(lambda ref: resources[ref.sha256]) if resources is not None else None)


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
        acquisition = next(identifier for identifier, record in document["provenance"].items()
                           if record["kind"] == "acquisition")
        assert acquisition != ref["processing_id"]
        ref["processing_id"] = acquisition
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
    newer = rehash_catalog(changed, edition_fixture["resources"])
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
def catalog_transport(tmp_path, monkeypatch):
    """Exercise the real streamed downloader without opening any socket."""
    requests = []
    monkeypatch.setattr("teadata.release.user_cache_path", lambda name: tmp_path / "resource-cache")
    base = "https://synthetic.invalid"
    def open_response(url, timeout):
        assert url.startswith(base + "/") and timeout == 60
        path = urlparse(url).path
        requests.append(path)
        return BytesIO((tmp_path / path.lstrip("/")).read_bytes())
    monkeypatch.setattr("teadata.release.urlopen", open_response)
    return base, requests


def test_pinned_root_selected_transport_second_root_and_offline(edition_fixture, catalog_transport, monkeypatch, tmp_path):
    from teadata.release import DataRelease
    base, requests = catalog_transport
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    root = edition_fixture["root"]
    key, student, _ = edition_fixture["leaves"]["student"]
    _, staff, _ = edition_fixture["leaves"]["staff"]
    url = base + "/root.json"
    pin = file_identity(root)["sha256"]
    selected = DataRelease(url, expected_sha256=pin)
    assert requests == ["/root.json"]
    assert len(list(selected.catalog.editions())) == 2 and requests == ["/root.json"]
    assert len(list(selected.catalog.variables(edition=key))) == 3
    student_schema = selected.catalog.to_dict()["editions"][key.identity]["column_schema_sha256"]
    assert requests == ["/root.json", "/" + student_schema + ".json"]
    assert selected.analytical_dataset(student.identity).to_table(columns=["amount"]).to_pydict() == {"amount": ["5"]}
    assert requests == ["/root.json", "/" + student_schema + ".json", "/student.parquet"]
    offline = DataRelease(url, expected_sha256=pin, offline=True)
    assert offline.artifact(student.identity).is_file()
    with pytest.raises(ContractError, match="not cached"):
        offline.artifact(staff.identity)
    selected.artifact(staff.identity)
    assert requests[-1] == "/staff.parquet" and len(requests) == 5
    document = selected.catalog.to_dict()
    for ref in document["artifacts"].values():
        ref["location"] = base + "/" + ref["location"]
    document["description"] = "second root referencing old-release URLs"
    newer = rehash_catalog(document)
    second = tmp_path / "second.json"
    second.write_text(canonical_json({"schema_version": 3, "catalog": newer.to_dict()}))
    next_release = DataRelease(base + "/second.json", expected_sha256=file_identity(second)["sha256"])
    next_release.artifact(student.identity)
    assert requests[-1] == "/second.json" and len(requests) == 6
    with pytest.raises(ContractError, match="period"):
        next_release.catalog.resolve_artifacts([(key, "amount")], period=DataPeriod(kind="calendar_year", year=2026))
    cached_root = offline.path
    cached_root.write_bytes(b"corrupt")
    with pytest.raises(ContractError, match="trusted pin"):
        DataRelease(url, expected_sha256=pin, offline=True)
    cached_root.unlink()
    with pytest.raises(FileNotFoundError):
        DataRelease(url, expected_sha256=pin, offline=True)
    assert len(requests) == 6


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
assert len(list(release.catalog.variables())) == 6
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
        catalog = rehash_catalog(document, edition_fixture["resources"])
        selected = catalog.resolve_artifacts([(student, "amount")])
        assert [row["artifact"].edition for row in selected] == [student, staff]
        assert all(row["columns"] == ("campus_number", "amount") for row in selected)
        assert len(catalog.header_records(student.dataset, edition_key=student)) == 3
        assert catalog.acquisitions(student.dataset, edition=student)[0]["downloaded_at"] is None
        with pytest.raises(ContractError, match="period"):
            catalog.resolve_artifacts([(student, "campus_number")], period=DataPeriod(kind="calendar_year", year=2025))


def test_partial_download_and_changed_cached_leaf_fail_without_fallback(edition_fixture, catalog_transport, monkeypatch, tmp_path):
    from teadata.release import DataRelease
    base, requests = catalog_transport
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    selected = DataRelease(base + "/root.json", expected_sha256=file_identity(edition_fixture["root"])["sha256"])
    _, student, _ = edition_fixture["leaves"]["student"]
    _, staff, staff_path = edition_fixture["leaves"]["staff"]
    staff_path.write_bytes(staff_path.read_bytes()[:32])
    with pytest.raises(ContractError, match="checksum or size"):
        selected.artifact(staff.identity)
    assert not (selected.cache / "objects" / staff.sha256).exists()
    assert list((selected.cache / "objects").iterdir())  # selected metadata is independently cached
    path = selected.artifact(student.identity)
    path.write_bytes(b"corrupt")
    with pytest.raises(ContractError, match="checksum or size"):
        selected.artifact(student.identity)
    assert requests[0] == "/root.json" and [p for p in requests if p.endswith(".parquet")] == ["/staff.parquet", "/student.parquet"]


def test_stream_download_accepts_legacy_full_identity_and_root_cache_keys_include_trust_options(edition_fixture, catalog_transport, monkeypatch, tmp_path):
    from teadata.release import _download, _load_release
    base, requests = catalog_transport
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


@pytest.mark.parametrize('damage', ['schema', 'rows', 'iteration'])
def test_native_leaf_writer_drops_partial_output_and_never_replaces(edition_fixture, tmp_path, damage):
    import pyarrow.parquet as pq
    from teadata.analytical import write_leaf_artifact

    _, ref, original = edition_fixture['leaves']['student']
    table = pq.read_table(original).replace_schema_metadata(None)
    output = tmp_path / 'new-leaf.parquet'
    before = set(tmp_path.iterdir())
    if damage == 'schema':
        batches = table.select(['amount']).to_batches()
    elif damage == 'rows':
        batches = table.slice(0, 0).to_batches()
    else:
        def interrupted():
            yield from table.to_batches()
            raise RuntimeError('injected batch interruption')
        batches = interrupted()
    with pytest.raises((ContractError, RuntimeError)):
        write_leaf_artifact(output, batches, contract=ref.contract)
    assert not output.exists() and set(tmp_path.iterdir()) == before
    identity = file_identity(original)
    with pytest.raises(FileExistsError):
        write_leaf_artifact(original, table.to_batches(), contract=ref.contract)
    assert file_identity(original) == identity


def test_explicit_local_schema3_publisher_uses_one_registry_and_keeps_prior_root(edition_fixture, monkeypatch):
    from teadata.release import DataRelease, publish_release
    root = edition_fixture['root']
    root.unlink()  # hand-authored test root is replaced by the canonical publisher.
    published = publish_release(root.parent, edition_fixture['catalog'], schema_version=3)
    assert DataRelease(published, offline=True).verify()
    before = file_identity(published)
    with pytest.raises(FileExistsError):
        publish_release(root.parent, edition_fixture['catalog'], schema_version=3)
    assert file_identity(published) == before
    with pytest.raises(ContractError, match='only its catalog'):
        publish_release(root.parent, edition_fixture['catalog'], artifacts={}, schema_version=3)
    with pytest.raises(ContractError, match='schema2'):
        publish_release(root.parent, edition_fixture['catalog'])
    with pytest.raises(ContractError, match='explicit schema3'):
        publish_release(root.parent, edition_fixture['catalog'], schema_version=3.0)  # ty: ignore[invalid-argument-type]
    oversized = deepcopy(edition_fixture['catalog'].to_dict())
    oversized['description'] = 'x' * (4 * 1024 * 1024)
    monkeypatch.setattr(edition_fixture['catalog'], 'to_dict', lambda: oversized)
    with pytest.raises(ContractError, match='too large'):
        publish_release(root.parent / 'oversized', edition_fixture['catalog'], schema_version=3)
    assert not (root.parent / 'oversized').exists()


def test_exact_same_dataset_revisions_never_choose_latest_or_share_field_cache(edition_fixture, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from teadata.analytical import write_leaf_artifact
    from teadata.dataset_metadata import DatasetMetadata, ResolvedDataset

    catalog, store = edition_fixture["catalog"], edition_fixture["store"]
    original, ref, leaf = edition_fixture["leaves"]["student"]
    document = catalog.to_dict()
    old_entry = document["editions"][original.identity]
    resource = json.loads(catalog.column_schema_bytes(old_entry["column_schema_sha256"]))
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
    schema_path = tmp_path / (schema_sha + ".json")
    schema_path.write_text(text)
    schema_id = store.record_acquisition(schema_path, ResolvedDataset(key.dataset + ":schema", "reference", None, None,
        str(schema_path), DatasetMetadata()))
    metadata = replace(metadata, references=(ReferenceDocument(reference_id="columns", role="schema", status="available",
        path=schema_path.name, reviewed_sha256=schema_sha, provenance_ids=(schema_id,)),))
    contract = ref.contract
    contract.update(edition=key.to_dict(), inputs=[raw_sha], mapping_sha256=sha256(canonical_json(table).encode()).hexdigest())
    from teadata.column_schema import header_records
    from teadata.manifest import _leaf_semantics
    contract["semantics_sha256"] = _leaf_semantics(metadata, header_records(resource,
        dataset=key.dataset, edition=key.edition, table=key.table, metadata=metadata))
    identity = sha256(canonical_json(contract).encode()).hexdigest()
    native = pq.read_table(leaf).replace_schema_metadata(None)
    output = tmp_path / "reissued.parquet"
    write_leaf_artifact(output, native.set_column(2, "amount", pa.array(["7"])).to_batches(), contract=contract)
    old_processing = store.read_record(ref.processing_id)
    processing = store.record_processing(dataset=key.dataset, inputs=[acquisition], outputs={"leaf": output},
        processor=old_processing["processor"], code_version=old_processing["code_version"],
        parameters={**old_processing["parameters"], "leaf_contract_sha256": identity})
    actual = file_identity(output)
    newer = ArtifactRef(key, actual["sha256"], actual["size_bytes"], output.name, canonical_json(contract), processing, "leaf")
    document["column_schemas"][schema_sha] = ColumnSchemaRef(schema_sha, len(text.encode()), schema_path.name).to_dict()
    document["editions"][key.identity] = {"key": key.to_dict(), "metadata": metadata.to_dict(),
        "acquisition_ids": [acquisition], "column_schema_sha256": schema_sha}
    document["artifacts"][identity] = newer.to_dict()
    document["field_bindings"].extend({**row, "edition_id": key.identity, "artifact_id": identity}
        for row in list(document["field_bindings"]) if row["edition_id"] == original.identity)
    document["provenance"].update(store.verify_record_closure([processing, schema_id]))
    selected = rehash_catalog(document, {**edition_fixture["resources"], schema_sha: text.encode()})
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
                schemas[schema_sha] = ColumnSchemaRef(schema_sha, len(text.encode()), schema_sha + ".json").to_dict()
                (tmp_path / (schema_sha + ".json")).write_text(text)
                schema_id = receipt({"schema_version": 1, "kind": "acquisition", "dataset": dataset + ":schema", "section": "reference",
                    "requested_year": None, "resolved_year": None, "source_metadata": {},
                    "artifact": {"sha256": schema_sha, "size_bytes": len(text.encode()), "filename": schema_sha + ".json"}})
                metadata = replace(metadata, references=(ReferenceDocument(reference_id="columns", role="schema", status="available",
                    path=schema_sha + ".json", reviewed_sha256=schema_sha, provenance_ids=(schema_id,)),))
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
catalog = DatasetCatalog(document, column_schema_loader=lambda ref: pathlib.Path(sys.argv[1]).with_name(ref.location).read_bytes())
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
    from teadata.analytical import write_leaf_artifact
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
        "references": [*entry["metadata"]["references"], {"reference_id": "method", "role": "methodology", "path": method.name,
                        "status": "available", "reviewed_sha256": method_sha, "provenance_ids": [method_id]}],
        "estimation": {"model_id": "synthetic", "value_per_suppressed": 5,
                       "grain": "origin_campus_destination_campus", "description": "Synthetic test model", "reference_id": "method"}})
    policy = ValuePolicy.from_dict({"policy_id": "numeric", "reference_id": "method" if suppression else None,
        "tokens": {"-1": {"status": "suppressed", "reason": "synthetic suppression"}} if suppression else {}})
    contract = ref.contract
    schema = json.loads(edition_fixture["catalog"].column_schema_bytes(entry["column_schema_sha256"]))
    contract["semantics_sha256"] = _leaf_semantics(metadata,
        header_records(schema, dataset=key.dataset, edition=key.edition, table=key.table, metadata=metadata))
    contract["policy_sha256"] = sha256(canonical_json(policy.to_dict()).encode()).hexdigest()
    identity = sha256(canonical_json(contract).encode()).hexdigest()
    native = pq.read_table(leaf).replace_schema_metadata(None)
    output = tmp_path / "method-reviewed.parquet"
    write_leaf_artifact(output, native.to_batches(), contract=contract)
    previous = store.read_record(ref.processing_id)
    processing = store.record_processing(dataset=key.dataset, inputs=[*previous["inputs"], method_id], outputs={"leaf": output},
        processor=previous["processor"], code_version=previous["code_version"],
        parameters={**previous["parameters"], "leaf_contract_sha256": identity})
    actual = file_identity(output)
    selected = ArtifactRef(key, actual["sha256"], actual["size_bytes"], output.name, canonical_json(contract), processing, "leaf")
    return DatasetCatalog.create_editions(editions={key.identity: {**entry, "metadata": metadata.to_dict()}},
        artifacts={identity: selected.to_dict()},
        field_bindings=[{**value, "artifact_id": identity} for value in document["field_bindings"] if value["edition_id"] == key.identity],
        column_schemas={entry["column_schema_sha256"]: document["column_schemas"][entry["column_schema_sha256"]]},
        records={**document["provenance"], **store.verify_record_closure([processing])},
        value_policies={"numeric": policy.to_dict()}, column_schema_loader=lambda ref: edition_fixture["resources"][ref.sha256])


@pytest.mark.parametrize("damage", ["suppression_without_reference", "unreviewed", "not_found", "no_review_sha", "no_receipt"])
def test_v3_methodology_states_cannot_claim_qualified_policy_or_model(edition_fixture, tmp_path, monkeypatch, damage):
    catalog = catalog_with_methodology(edition_fixture, tmp_path, suppression=damage == "suppression_without_reference")
    document = catalog.to_dict()
    entry, = document["editions"].values()
    if damage == "suppression_without_reference":
        document["value_policies"]["numeric"]["reference_id"] = None
    else:
        method, = [r for r in entry["metadata"]["references"] if r["role"] == "methodology"]
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


@pytest.mark.parametrize("damage", ["inline", "boolean_size", "unknown", "oversize", "traversal", "foreign_sha", "no_observation"])
def test_external_schema_descriptors_are_strict_and_root_checkable(edition_fixture, damage, monkeypatch):
    from teadata.dataset_metadata import COLUMN_SCHEMA_MAX_BYTES
    document = edition_fixture["catalog"].to_dict()
    digest, descriptor = next(iter(document["column_schemas"].items()))
    if damage == "inline":
        document["column_schemas"][digest] = edition_fixture["resources"][digest].decode()
    elif damage == "boolean_size":
        descriptor["size_bytes"] = True
    elif damage == "unknown":
        descriptor["format"] = "parquet"
    elif damage == "oversize":
        descriptor["size_bytes"] = COLUMN_SCHEMA_MAX_BYTES + 1
    elif damage == "traversal":
        descriptor["location"] = "../schema.json"
    elif damage == "foreign_sha":
        descriptor["sha256"] = "f" * 64
    else:
        for entry in document["editions"].values():
            if entry["column_schema_sha256"] == digest:
                entry["metadata"]["references"][0]["provenance_ids"] = []
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("Root validation opened a resource"))
    with pytest.raises(ContractError):
        rehash_catalog(document)


def test_selected_resource_changes_invalidate_warm_metadata_and_pickle_trust(edition_fixture):
    from teadata.release import DataRelease
    selected = DataRelease(edition_fixture["root"])
    key, _, _ = edition_fixture["leaves"]["student"]
    assert len(selected.catalog.header_records(key.dataset, edition_key=key)) == 3
    restored_catalog = pickle.loads(pickle.dumps(selected.catalog))
    assert not restored_catalog._validated_editions and not restored_catalog._column_schema_buffers
    assert len(list(restored_catalog.editions())) == 2
    with pytest.raises(ContractError, match="explicit.*resolver"):
        list(restored_catalog.variables(edition=key))
    restored_release = pickle.loads(pickle.dumps(selected))
    assert not restored_release._column_schema_data and not restored_release._verified
    digest = selected.catalog.to_dict()["editions"][key.identity]["column_schema_sha256"]
    path = selected.path.with_name(digest + ".json")
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + b"!")
    for release in (selected, restored_release):
        with pytest.raises(ContractError, match="Column-schema resource checksum"):
            release.catalog.header_records(key.dataset, edition_key=key)


def damage_selected_mapping(document, edition):
    """Well-hashed synthetic contradiction; no physical artifact is changed."""
    artifact_id, ref = next((i, r) for i, r in document["artifacts"].items() if r["edition"]["dataset"] == edition.dataset)
    document["artifacts"].pop(artifact_id)
    ref["contract"]["mapping_sha256"] = "f" * 64
    changed = sha256(canonical_json(ref["contract"]).encode()).hexdigest()
    processing = deepcopy(document["provenance"][ref["processing_id"]])
    processing["parameters"]["leaf_contract_sha256"] = changed
    processing_id = sha256((canonical_json(processing) + "\n").encode()).hexdigest()
    document["provenance"][processing_id] = processing
    ref["processing_id"] = processing_id
    document["artifacts"][changed] = ref
    for binding in document["field_bindings"]:
        if binding["artifact_id"] == artifact_id:
            binding["artifact_id"] = changed
    return changed


@pytest.mark.parametrize("selection", ["direct", "dependency", "complete_verify", "publication"])
def test_crosswalk_semantics_precede_every_selected_leaf_transport(edition_fixture, catalog_transport, tmp_path, selection):
    from teadata.release import DataRelease, publish_release
    base, requests = catalog_transport
    document = edition_fixture["catalog"].to_dict()
    student, _, _ = edition_fixture["leaves"]["student"]
    staff, staff_ref, _ = edition_fixture["leaves"]["staff"]
    changed = damage_selected_mapping(document, student)
    if selection == "dependency":
        amount = next(row for row in document["field_bindings"] if row["edition_id"] == staff.identity and row["field"] == "amount")
        amount["dependencies"] = [[student.identity, "amount"]]
    catalog = rehash_catalog(document, edition_fixture["resources"])
    path = tmp_path / "contradiction.json"
    path.write_text(canonical_json({"schema_version": 3, "catalog": catalog.to_dict()}))
    release = DataRelease(base + "/contradiction.json", expected_sha256=file_identity(path)["sha256"])
    assert len(list(release.catalog.editions())) == 2 and requests == ["/contradiction.json"]
    with pytest.raises(ContractError, match="mapping or semantics"):
        if selection == "direct":
            release.artifact(changed)
        elif selection == "dependency":
            release.artifact(staff_ref.identity)
        elif selection == "complete_verify":
            release.verify()
        else:
            publish_release(tmp_path / "unpublished", catalog, schema_version=3)
    assert not any(name.endswith(".parquet") for name in requests)
    assert not (tmp_path / "unpublished" / "release.json").exists()


def test_metadata_download_is_exact_bounded_and_offline_selection_is_independent(edition_fixture, catalog_transport, tmp_path, monkeypatch):
    from teadata.release import DataRelease
    base, requests = catalog_transport
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "separate-cache"))
    url, pin = base + "/root.json", file_identity(edition_fixture["root"])["sha256"]
    selected = DataRelease(url, expected_sha256=pin)
    student, _, _ = edition_fixture["leaves"]["student"]
    staff, _, _ = edition_fixture["leaves"]["staff"]
    digest = selected.catalog.to_dict()["editions"][student.identity]["column_schema_sha256"]
    resource = tmp_path / (digest + ".json")
    raw = resource.read_bytes()
    resource.write_bytes(raw + b"overflow")
    with pytest.raises(ContractError, match="exceeds.*size"):
        list(selected.catalog.variables(edition=student))
    assert not (selected.cache / "objects" / digest).exists()
    resource.write_bytes(raw)
    assert len(list(selected.catalog.variables(edition=student))) == 3
    offline = DataRelease(url, expected_sha256=pin, offline=True)
    assert len(list(offline.catalog.variables(edition=student))) == 3
    before = requests[:]
    with pytest.raises(ContractError, match="not cached"):
        list(offline.catalog.variables(edition=staff))
    assert requests == before and not any(p.endswith(".parquet") for p in requests)


def test_wide_native_leaf_never_stitches_admitted_source_groups(edition_fixture, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from teadata.analytical import write_leaf_artifact
    from teadata.scripts.build_tapr_profiles import _prepared_batches
    from teadata.tapr import TaprReadBudget
    schema = pa.schema([(name, pa.string()) for name in ["campus_number", *(f"value_{i}" for i in range(79))]])
    table = pa.Table.from_arrays([pa.array(["abcdefgh"] * 10) for _ in schema], schema=schema)
    assert table.nbytes > 2048
    source = tmp_path / "wide-source.parquet"
    pq.write_table(table, source, row_group_size=2)
    contract = edition_fixture["leaves"]["student"][1].contract
    raw_schema = schema.serialize().to_pybytes()
    import base64
    contract.update(schema=base64.b64encode(raw_schema).decode(), schema_sha256=sha256(raw_schema).hexdigest(),
                    columns=schema.names, rows=10)
    budget = TaprReadBudget(batch_bytes=2048, batch_rows=2)
    output = tmp_path / "wide-leaf.parquet"
    with pq.ParquetFile(source) as native:
        assert next(native.iter_batches(batch_size=4096)).nbytes > budget.batch_bytes  # old stitching counterexample
        write_leaf_artifact(output, _prepared_batches(native, budget), contract=contract, batch_bytes=budget.batch_bytes)
    with pq.ParquetFile(output) as native:
        assert [native.metadata.row_group(i).num_rows for i in range(native.metadata.num_row_groups)] == [2] * 5
        assert all(batch.nbytes <= 2048 for batch in _prepared_batches(native, budget))
    with pq.ParquetFile(source) as native:
        with pytest.raises(ValueError, match="one admitted"):
            list(_prepared_batches(native, TaprReadBudget(batch_bytes=8, batch_rows=2)))
    oversized = tmp_path / "oversized-leaf.parquet"
    with pytest.raises(ContractError, match="admitted.*budget"):
        write_leaf_artifact(oversized, table.to_batches(), contract=contract, batch_bytes=2048)
    assert not oversized.exists()


def test_non_combined_group_requirement_rejects_multiple_arrow_chunks():
    import pyarrow as pa
    from types import SimpleNamespace
    from teadata.scripts.build_tapr_profiles import _prepared_batches
    from teadata.tapr import TaprReadBudget
    group = pa.table({"value": pa.chunked_array([["one"], ["two"]])})
    native = SimpleNamespace(metadata=SimpleNamespace(num_row_groups=1, row_group=lambda i: SimpleNamespace(num_rows=2)),
        read_row_group=lambda i, use_threads: group)
    with pytest.raises(ValueError, match="non-combined"):
        list(_prepared_batches(native, TaprReadBudget()))


def test_shared_crosswalk_is_fetched_once_for_two_selected_editions(edition_fixture, catalog_transport, tmp_path):
    from teadata.release import DataRelease
    base, requests = catalog_transport
    document = edition_fixture["catalog"].to_dict()
    datasets = [branch for raw in edition_fixture["resources"].values() for branch in json.loads(raw)["datasets"]]
    raw = canonical_json({"schema_version": 2, "mapping_version": 1, "datasets": datasets}).encode()
    digest = sha256(raw).hexdigest()
    resource = tmp_path / (digest + ".json")
    resource.write_bytes(raw)
    store = edition_fixture["store"]
    observation = store.record_acquisition(resource, ResolvedDataset("shared-schema", "reference", None, None,
        str(resource), DatasetMetadata()))
    document["provenance"][observation] = store.read_record(observation)
    for entry in document["editions"].values():
        entry["column_schema_sha256"] = digest
        entry["metadata"]["references"] = [ReferenceDocument(reference_id="columns", role="schema", status="available",
            reviewed_sha256=digest, path=resource.name, provenance_ids=(observation,)).to_dict()]
    document["column_schemas"] = {digest: ColumnSchemaRef(digest, len(raw), resource.name).to_dict()}
    catalog = rehash_catalog(document)
    path = tmp_path / "shared.json"
    path.write_text(canonical_json({"schema_version": 3, "catalog": catalog.to_dict()}))
    release = DataRelease(base + "/shared.json", expected_sha256=file_identity(path)["sha256"])
    assert requests == ["/shared.json"]
    for key, _, _ in edition_fixture["leaves"].values():
        assert len(list(release.catalog.variables(edition=key))) == 3
    assert requests == ["/shared.json", "/" + digest + ".json"]
    assert len(release.catalog._column_schemas) == 1


def test_exact_root_size_boundary_survives_externalization(edition_fixture, tmp_path):
    from teadata.release import DataRelease
    cap = 4 * 1024 * 1024
    document = edition_fixture["catalog"].to_dict()
    document["description"] = ""
    base = len((canonical_json({"schema_version": 3, "catalog": document}) + "\n").encode())
    document["description"] = "x" * (cap - base)
    catalog = rehash_catalog(document)
    raw = (canonical_json({"schema_version": 3, "catalog": catalog.to_dict()}) + "\n").encode()
    assert len(raw) == cap
    path = tmp_path / "exact-root-cap.json"
    path.write_bytes(raw)
    assert DataRelease(path).build_id == catalog.build_id
    path.write_bytes(raw + b" ")
    with pytest.raises(ContractError, match="too large"):
        DataRelease(path)


def test_metadata_publication_fsync_failure_removes_its_private_temporary(edition_fixture, tmp_path, monkeypatch):
    from teadata.release import publish_release
    original = file_identity(edition_fixture["root"])
    output = tmp_path / "fsync-failure"
    def fail_fsync(fd):
        raise OSError("injected schema fsync failure")
    monkeypatch.setattr("teadata.release.os.fsync", fail_fsync)
    with pytest.raises(OSError, match="injected schema"):
        publish_release(output, edition_fixture["catalog"], schema_version=3)
    assert output.is_dir() and list(output.iterdir()) == []
    assert file_identity(edition_fixture["root"]) == original


@pytest.mark.parametrize("collision", ["root", "case_root", "portable_root", "duplicate_resource", "unicode_resource", "artifact_resource", "duplicate_artifact"])
def test_metadata_destinations_preflight_before_every_resolution_and_write(edition_fixture, tmp_path, monkeypatch, collision):
    from teadata.release import publish_release
    document = edition_fixture["catalog"].to_dict()
    resources = list(document["column_schemas"].values())
    if collision in {"root", "case_root", "portable_root"}:
        resources[0]["location"] = {"root": "release.json", "case_root": "ReLeAsE.JsOn", "portable_root": "release.json."}[collision]
    elif collision in {"duplicate_resource", "unicode_resource"}:
        first, second = (("shared.json", "Shared.json") if collision == "duplicate_resource" else ("CAFÉ.json", "cafe\u0301.json"))
        resources[0]["location"], resources[1]["location"] = first, second
    elif collision == "artifact_resource":
        resources[0]["location"] = next(iter(document["artifacts"].values()))["location"]
    else:
        artifacts = list(document["artifacts"].values())
        artifacts[0]["location"] = artifacts[1]["location"] = "duplicate.parquet"
    catalog = rehash_catalog(document, edition_fixture["resources"])
    before = file_identity(edition_fixture["root"])
    monkeypatch.setattr(catalog, "verify_metadata", lambda: pytest.fail("Destination collision resolved metadata"))
    monkeypatch.setattr(catalog, "column_schema_bytes", lambda *a: pytest.fail("Destination collision read metadata"))
    output = tmp_path / "collision-output"
    with pytest.raises(ContractError, match="destinations collide"):
        publish_release(output, catalog, schema_version=3)
    assert not output.exists() and file_identity(edition_fixture["root"]) == before


@pytest.mark.parametrize("boundary", ["selected", "publisher"])
@pytest.mark.parametrize("entry", ["fifo", "symlink", "directory", "oversize", "replacement_fifo"])
def test_metadata_regular_file_admission_has_a_finite_filesystem_harness(edition_fixture, tmp_path, boundary, entry):
    """A regression to blocking FIFO open fails a bounded child, never hangs pytest."""
    script = r'''
from pathlib import Path
import os, sys
from teadata.manifest import ContractError
from teadata.release import DataRelease, publish_release
root, boundary, entry = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
release = DataRelease(root)
key = next(release.catalog.editions())
from teadata.dataset_metadata import EditionKey
edition = EditionKey.from_dict({k: key[k] for k in ('dataset','edition','revision_sha256','scope','table','grain')})
digest = release.catalog.to_dict()['editions'][key['edition_id']]['column_schema_sha256']
descriptor = next(r for r in release.catalog.column_schema_refs() if r.sha256 == digest)
target = root.with_name(descriptor.location)
# Publisher uses a fresh directory and in-memory verified input bytes, never the damaged destination as its loader.
if boundary == 'publisher':
    release.catalog.verify_metadata()
    resources = {r.sha256: release.catalog.column_schema_bytes(r.sha256) for r in release.catalog.column_schema_refs()}
    release.catalog._column_schema_loader = lambda ref: resources[ref.sha256]
    output = root.parent / 'filesystem-output'
    output.mkdir()
    target = output / descriptor.location
    target.write_bytes(resources[digest])
    action = lambda: publish_release(output, release.catalog, schema_version=3)
else:
    action = lambda: list(release.catalog.variables(edition=edition))
if entry == 'symlink':
    source = target.with_name('symlink-target.json')
    target.rename(source)
    target.symlink_to(source.name)
elif entry == 'directory':
    target.unlink(); target.mkdir()
elif entry == 'oversize':
    with target.open('ab') as stream: stream.write(b'exceeds the declared metadata size')
elif entry == 'fifo':
    target.unlink(); os.mkfifo(target, 0o600)
# Check the admission happens before read/open on known invalid entries, and the race open carries both flags.
original_open = os.open
calls = []
def guarded_open(path, flags, *a, **k):
    if Path(path) == target:
        calls.append(flags)
        assert entry == 'replacement_fifo', 'Known invalid metadata reached open before admission'
        assert flags & os.O_NOFOLLOW and flags & os.O_NONBLOCK
        target.unlink(); os.mkfifo(target, 0o600)
    return original_open(path, flags, *a, **k)
os.open = guarded_open
try:
    action()
except ContractError:
    pass
else:
    raise AssertionError('Invalid metadata was accepted')
assert len(calls) == (1 if entry == 'replacement_fifo' else 0)
if boundary == 'publisher':
    assert not (output / 'release.json').exists()
    # Every destination is admitted before any absent sidecar is installed.
    assert sorted(p.name for p in output.iterdir()) == sorted([target.name] + (['symlink-target.json'] if entry == 'symlink' else []))
print('bounded metadata rejection PASS')
'''
    result = subprocess.run([sys.executable, "-c", script, str(edition_fixture["root"]), boundary, entry],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "bounded metadata rejection PASS"


def test_metadata_regular_capture_reuses_only_same_file_signature(edition_fixture, monkeypatch):
    from teadata.release import DataRelease
    release = DataRelease(edition_fixture["root"])
    key, _, _ = edition_fixture["leaves"]["student"]
    assert len(list(release.catalog.variables(edition=key))) == 3
    import teadata.release as transport
    original = transport.os.open
    def no_repeat_open(path, *a, **k):
        pytest.fail("Warm metadata file was reopened instead of using verified stat cache")
    with monkeypatch.context() as warm:
        warm.setattr(transport.os, "open", no_repeat_open)
        assert len(list(release.catalog.variables(edition=key))) == 3
    descriptor = next(r for r in release.catalog.column_schema_refs()
                      if r.sha256 == release.catalog.to_dict()["editions"][key.identity]["column_schema_sha256"])
    path = release.path.with_name(descriptor.location)
    raw = path.read_bytes()
    path.write_bytes(raw)
    calls = []
    def count_open(path, *a, **k):
        calls.append(path)
        return original(path, *a, **k)
    monkeypatch.setattr(transport.os, "open", count_open)
    assert len(list(release.catalog.variables(edition=key))) == 3
    assert calls == [path]


@pytest.fixture
def edition_index_fixture(edition_fixture):
    """Explicit v3 migration using only the existing two tiny synthetic leaves."""
    catalog = edition_fixture["catalog"].externalize_editions()
    resources = {}
    for row in catalog.editions(include_metadata=False):
        identifier = row["edition_id"]
        raw = catalog.edition_metadata_bytes(identifier)
        descriptor = catalog.to_dict()["editions"][identifier]["metadata"]
        resources[descriptor["sha256"]] = raw
        edition_fixture["root"].with_name(descriptor["location"]).write_bytes(raw)
    root = edition_fixture["root"].with_name("index.json")
    root.write_text(canonical_json({"schema_version": 4, "catalog": catalog.to_dict()}) + "\n")
    return {**edition_fixture, "legacy": edition_fixture["catalog"], "catalog": catalog, "root": root,
            "children": resources}


def index_with_child(edition_index_fixture, key, change):
    """Rehash a deliberately contradictory child; never mutate its leaf or receipt."""
    fixture = edition_index_fixture
    document, resources = fixture["catalog"].to_dict(), dict(fixture["children"])
    ref = document["editions"][key.identity]["metadata"]
    child = json.loads(resources[ref["sha256"]])
    change(child)
    raw = (canonical_json(child) + "\n").encode()
    digest = sha256(raw).hexdigest()
    resources[digest] = raw
    document["editions"][key.identity]["metadata"] = EditionMetadataRef(digest, len(raw), digest + ".json").to_dict()
    document["build_id"] = sha256(canonical_json({k: v for k, v in document.items() if k != "build_id"}).encode()).hexdigest()
    return DatasetCatalog(document, column_schema_loader=lambda ref: fixture["resources"][ref.sha256],
                          edition_metadata_loader=lambda ref: resources[ref.sha256])


@pytest.mark.parametrize("damage", ["size_bool", "oversize", "unknown", "traversal", "credentials", "sha"])
def test_v4_metadata_resource_descriptor_is_strict(damage):
    from teadata.dataset_metadata import EDITION_METADATA_MAX_BYTES
    raw = {"sha256": "a" * 64, "size_bytes": 10, "location": "metadata.json"}
    if damage == "size_bool":
        raw["size_bytes"] = True
    elif damage == "oversize":
        raw["size_bytes"] = EDITION_METADATA_MAX_BYTES + 1
    elif damage == "unknown":
        raw["format"] = "parquet"
    elif damage == "traversal":
        raw["location"] = "../metadata.json"
    elif damage == "credentials":
        raw["location"] = "https://user:password@example.invalid/metadata.json"
    else:
        raw["sha256"] = "z" * 64
    with pytest.raises(ValueError):
        EditionMetadataRef.from_dict(raw)


def test_v4_index_discovery_and_construction_never_prefetch_children(edition_index_fixture, monkeypatch):
    fixture = edition_index_fixture
    calls = []
    def child(ref):
        calls.append(ref.sha256)
        return fixture["children"][ref.sha256]
    catalog = DatasetCatalog(fixture["catalog"].to_dict(), edition_metadata_loader=child,
                             column_schema_loader=lambda ref: fixture["resources"][ref.sha256])
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("Metadata discovery opened data"))
    full = catalog.editions()
    assert calls == []
    cheap = list(catalog.editions(include_metadata=False))
    assert len(cheap) == 2 and all("metadata" not in row for row in cheap) and calls == []
    first = next(full)
    assert len(calls) == 1 and "metadata" in first
    assert list(catalog.editions(dataset=first["dataset"])) == [first]
    assert len(set(calls)) == 1
    assert not catalog._edition_views[first["edition_id"]]._validated_editions  # no crosswalk needed for source metadata


def test_v4_migration_preserves_detached_public_projections_and_every_identity(edition_index_fixture):
    fixture = edition_index_fixture
    old, current = fixture["legacy"], fixture["catalog"]
    original = old.to_dict()
    assert current.datasets == old.datasets
    assert list(current.editions()) == list(old.editions())
    records, fields, artifacts = {}, [], {}
    for row in current.editions(include_metadata=False):
        identifier = row["edition_id"]
        child = json.loads(current.edition_metadata_bytes(identifier))
        records.update(child["provenance"])
        fields.extend(child["field_bindings"])
        artifacts.update(child["artifacts"])
        key = EditionKey.from_dict(child["edition"]["key"])
        assert current.metadata(key.dataset, edition=key) == old.metadata(key.dataset, edition=key)
        assert current.acquisitions(key.dataset, edition=key) == old.acquisitions(key.dataset, edition=key)
        assert current.policy(key.dataset, edition=key) == old.policy(key.dataset, edition=key)
        assert current.header_records(key.dataset, edition_key=key) == old.header_records(key.dataset, edition_key=key)
        assert list(current.variables(edition=key)) == list(old.variables(edition=key))
        assert current.field_metadata("amount", edition=key) == old.field_metadata("amount", edition=key)
        assert current.resolve_artifacts([(key, "amount"), (key, "campus_name")]) == old.resolve_artifacts([(key, "amount"), (key, "campus_name")])
    assert records == original["provenance"] and artifacts == original["artifacts"]
    assert sorted(fields, key=canonical_json) == sorted(original["field_bindings"], key=canonical_json)
    for artifact in original["artifacts"]:
        assert current.artifact_ref(artifact) == old.artifact_ref(artifact)
    detached = current.datasets
    detached[next(iter(detached))]["metadata"]["title"] = "foreign mutation"
    assert current.datasets == old.datasets and original == old.to_dict()


@pytest.mark.parametrize("damage", ["edition", "route", "missing_observation", "methodology", "foreign_field", "cycle"])
def test_v4_child_and_closed_dependency_contradictions_precede_leaf_io(edition_index_fixture, monkeypatch, damage):
    fixture = edition_index_fixture
    key, descriptor, _ = fixture["leaves"]["student"]
    def change(child):
        if damage == "edition":
            child["edition"]["key"]["edition"] = "2026"
        elif damage == "route":
            child["artifacts"] = {}
        elif damage == "missing_observation":
            child["edition"]["metadata"]["references"][0]["provenance_ids"] = []
        elif damage == "methodology":
            child["edition"]["metadata"]["estimation"] = {"method": "unknown"}
        else:
            amount = next(row for row in child["field_bindings"] if row["field"] == "amount")
            amount["dependencies"] = [[key.identity, "missing" if damage == "foreign_field" else "amount"]]
    catalog = index_with_child(fixture, key, change)
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("Contradictory metadata opened a leaf"))
    with pytest.raises(ContractError):
        catalog.artifact_ref(descriptor.identity)


def test_v4_selected_foreign_dependency_closure_uses_existing_resolver(edition_index_fixture, monkeypatch):
    fixture = edition_index_fixture
    student, ref, _ = fixture["leaves"]["student"]
    staff, _, _ = fixture["leaves"]["staff"]
    def change(child):
        next(row for row in child["field_bindings"] if row["field"] == "amount")["dependencies"] = [[staff.identity, "amount"]]
    catalog = index_with_child(fixture, student, change)
    calls = []
    original = catalog._edition_metadata_loader
    def capture(descriptor):
        calls.append(descriptor.sha256)
        return original(descriptor)
    catalog._edition_metadata_loader = capture
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("Resolver opened native data"))
    assert catalog.artifact_ref(ref.identity) == ref
    result = catalog.resolve_artifacts([(student, "amount"), (staff, "amount")])
    assert [row["artifact"].edition for row in result] == [student, staff]
    assert [row["columns"] for row in result] == [("campus_number", "amount"), ("campus_number", "amount")]
    assert len(set(calls)) == 2
    with pytest.raises(ContractError, match="period"):
        catalog.resolve_artifacts([(student, "campus_number")], period=DataPeriod(kind="calendar_year", year=2025))
    def foreign(child):
        next(row for row in child["field_bindings"] if row["field"] == "amount")["dependencies"] = [["f" * 64, "amount"]]
    with pytest.raises(ContractError, match="metadata"):
        index_with_child(fixture, student, foreign).artifact_ref(ref.identity)


def test_v4_pickle_drops_child_acceptance_and_rechecks_current_files(edition_index_fixture):
    from teadata.release import DataRelease
    fixture = edition_index_fixture
    release = DataRelease(fixture["root"], offline=True)
    key, ref, _ = fixture["leaves"]["student"]
    assert len(list(release.catalog.variables(edition=key))) == 3
    restored_catalog = pickle.loads(pickle.dumps(release.catalog))
    assert not restored_catalog._edition_views and not restored_catalog._edition_buffers
    assert len(list(restored_catalog.editions(include_metadata=False))) == 2
    with pytest.raises(ContractError, match="explicit.*resolver"):
        next(restored_catalog.editions())
    restored = pickle.loads(pickle.dumps(release))
    assert not restored._column_schema_data and not restored._verified
    child = release.catalog.to_dict()["editions"][key.identity]["metadata"]
    path = fixture["root"].with_name(child["location"])
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + b"!")
    for selected in (release, restored):
        with pytest.raises(ContractError, match="checksum"):
            selected.catalog.artifact_ref(ref.identity)


def test_v4_offline_remote_root_and_selected_children_share_existing_cache(edition_index_fixture, monkeypatch, tmp_path):
    from teadata.release import DataRelease
    import teadata.release as transport
    fixture = edition_index_fixture
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    root = fixture["root"].read_bytes()
    available = {"index.json": root, **{ref.location: fixture["children"][ref.sha256] for ref in fixture["catalog"].edition_metadata_refs()},
                 **{ref.location: fixture["resources"][ref.sha256] for ref in fixture["legacy"].column_schema_refs()}}
    requests = []
    def download(url, path, *, expected=None):
        requests.append(url)
        raw = available[url.rsplit("/", 1)[1]]
        assert expected is None or all(expected.get(k, v) == v for k, v in (("sha256", sha256(raw).hexdigest()), ("size_bytes", len(raw))))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    monkeypatch.setattr(transport, "_download", download)
    pin = sha256(root).hexdigest()
    release = DataRelease("https://example.invalid/index.json", expected_sha256=pin)
    assert len(list(release.catalog.editions(include_metadata=False))) == 2 and len(requests) == 1
    key, _, _ = fixture["leaves"]["student"]
    assert len(list(release.catalog.variables(edition=key))) == 3 and len(requests) == 3
    offline = DataRelease(release.source, expected_sha256=pin, offline=True)
    assert len(list(offline.catalog.variables(edition=key))) == 3 and len(requests) == 3
    staff, _, _ = fixture["leaves"]["staff"]
    with pytest.raises(ContractError, match="not cached"):
        list(offline.catalog.variables(edition=staff))
    release.path.unlink()
    with pytest.raises(FileNotFoundError):
        DataRelease(release.source, expected_sha256=pin, offline=True)


def test_v4_publication_is_explicit_complete_and_preserves_native_contracts(edition_index_fixture, tmp_path):
    from teadata.release import DataRelease, publish_release
    fixture = edition_index_fixture
    output = tmp_path / "published-v4"
    output.mkdir()
    for _, ref, leaf in fixture["leaves"].values():
        (output / ref.location).write_bytes(leaf.read_bytes())
    root = publish_release(output, fixture["catalog"], schema_version=4)
    release = DataRelease(root, offline=True)
    assert len(list(release.catalog.editions(include_metadata=False))) == 2
    assert set(release.verify()) == set(fixture["legacy"].to_dict()["artifacts"])
    for artifact, expected in fixture["legacy"].to_dict()["artifacts"].items():
        assert release.catalog.artifact_ref(artifact).to_dict() == expected
    with pytest.raises(FileExistsError):
        publish_release(output, fixture["catalog"], schema_version=4)
    with pytest.raises(ContractError, match="schema2"):
        publish_release(tmp_path / "default", fixture["catalog"])
    assert fixture["recorder"].catalog(schema_version=4).to_dict() == fixture["catalog"].to_dict()


@pytest.mark.parametrize("damage", ["child_root", "child_artifact", "sidecar_symlink", "sidecar_corrupt", "sidecar_directory"])
def test_v4_publication_admits_every_resource_before_writes(edition_index_fixture, tmp_path, damage):
    from teadata.release import publish_release
    fixture = edition_index_fixture
    document = fixture["catalog"].to_dict()
    first = next(iter(document["editions"]))
    if damage.startswith("child_"):
        document["editions"][first]["metadata"]["location"] = (
            "ReLeAsE.JsOn." if damage == "child_root" else next(iter(fixture["legacy"].to_dict()["artifacts"].values()))["location"])
        document["build_id"] = sha256(canonical_json({k: v for k, v in document.items() if k != "build_id"}).encode()).hexdigest()
    catalog = DatasetCatalog(document, edition_metadata_loader=lambda ref: fixture["children"][ref.sha256],
                             column_schema_loader=lambda ref: fixture["resources"][ref.sha256])
    output = tmp_path / "refused-v4"
    before = set()
    if damage.startswith("sidecar_"):
        output.mkdir()
        target = output / list(catalog.edition_metadata_refs())[-1].location
        if damage == "sidecar_directory":
            target.mkdir()
        elif damage == "sidecar_symlink":
            target.symlink_to(fixture["root"])
        else:
            target.write_bytes(b"corrupt")
        before = set(output.iterdir())
    with pytest.raises(ContractError):
        publish_release(output, catalog, schema_version=4)
    assert set(output.iterdir()) == before if output.exists() else not before
    assert not (output / "release.json").exists()


def test_v4_migration_rejects_unbound_original_provenance(edition_fixture):
    document = edition_fixture["catalog"].to_dict()
    raw = edition_fixture["root"].with_name("unrelated.txt")
    raw.write_text("unrelated")
    identifier = edition_fixture["store"].record_acquisition(raw, ResolvedDataset("unrelated", "reference", None, None,
        str(raw), DatasetMetadata(title="Unbound synthetic observation")))
    document["provenance"][identifier] = edition_fixture["store"].read_record(identifier)
    with pytest.raises(ContractError, match="account for every"):
        rehash_catalog(document, edition_fixture["resources"]).externalize_editions()


def test_v4_interleaved_input_preserves_selected_order_without_a_global_field_ledger(edition_fixture):
    document = edition_fixture["catalog"].to_dict()
    editions = list(document["editions"])
    groups = [[row for row in document["field_bindings"] if row["edition_id"] == edition] for edition in editions]
    document["field_bindings"] = [row for pair in zip(*groups) for row in pair]
    old = rehash_catalog(document, edition_fixture["resources"])
    current = old.externalize_editions()
    expected = []
    for row in current.editions(include_metadata=False):
        key = old._edition_keys[row["edition_id"]]
        assert list(current.variables(edition=key)) == list(old.variables(edition=key))
        expected.extend(old.variables(edition=key))
        fields = ["amount", "campus_name", "amount"]
        assert current.resolve_artifacts((key, field) for field in fields) == old.resolve_artifacts((key, field) for field in fields)
    assert list(current.variables()) == expected
    assert list(current.variables()) != list(old.variables())


def test_v4_same_policy_name_remains_edition_scoped_even_through_foreign_dependencies(edition_fixture, tmp_path):
    from teadata.provenance import validate_record_graph
    modeled = catalog_with_methodology(edition_fixture, tmp_path, suppression=True).to_dict()
    student = next(iter(modeled["editions"]))
    roots = [ref["processing_id"] for ref in modeled["artifacts"].values()]
    roots.extend(value for reference in modeled["editions"][student]["metadata"]["references"] for value in reference["provenance_ids"])
    modeled["provenance"] = validate_record_graph(modeled["provenance"].__getitem__, roots)
    student_index = rehash_catalog(modeled, edition_fixture["resources"]).externalize_editions()
    plain = edition_fixture["catalog"].to_dict()
    staff = next(key for key in plain["editions"] if key != student)
    plain["editions"] = {staff: plain["editions"][staff]}
    plain["artifacts"] = {key: ref for key, ref in plain["artifacts"].items() if ref["edition"]["dataset"] == "campus_staff"}
    plain["field_bindings"] = [row for row in plain["field_bindings"] if row["edition_id"] == staff]
    digest = plain["editions"][staff]["column_schema_sha256"]
    plain["column_schemas"] = {digest: plain["column_schemas"][digest]}
    roots = [ref["processing_id"] for ref in plain["artifacts"].values()]
    roots.extend(value for reference in plain["editions"][staff]["metadata"]["references"] for value in reference["provenance_ids"])
    plain["provenance"] = validate_record_graph(plain["provenance"].__getitem__, roots)
    staff_index = rehash_catalog(plain, edition_fixture["resources"]).externalize_editions()
    document, children = student_index.to_dict(), {}
    for index in (student_index, staff_index):
        for identifier in index.to_dict()["editions"]:
            raw = index.edition_metadata_bytes(identifier)
            if identifier == student:
                child = json.loads(raw)
                next(row for row in child["field_bindings"] if row["field"] == "amount")["dependencies"] = [[staff, "amount"]]
                raw = (canonical_json(child) + "\n").encode()
            digest = sha256(raw).hexdigest()
            children[digest] = raw
            document["editions"][identifier] = {"key": index.to_dict()["editions"][identifier]["key"],
                "metadata": EditionMetadataRef(digest, len(raw), digest + ".json").to_dict()}
        document["artifacts"].update(index.to_dict()["artifacts"])
    document["build_id"] = sha256(canonical_json({k: v for k, v in document.items() if k != "build_id"}).encode()).hexdigest()
    current = DatasetCatalog(document, edition_metadata_loader=lambda ref: children[ref.sha256],
                             column_schema_loader=lambda ref: edition_fixture["resources"][ref.sha256])
    student_key = EditionKey.from_dict(document["editions"][student]["key"])
    staff_key = EditionKey.from_dict(document["editions"][staff]["key"])
    assert current.policy(student_key.dataset, edition=student_key).tokens["-1"].status.value == "suppressed"
    assert not current.policy(staff_key.dataset, edition=staff_key).tokens
    assert len(current.resolve_artifacts([(student_key, "amount")])) == 2


def test_v4_direct_leaf_transport_waits_for_foreign_crosswalk_semantics(edition_index_fixture, tmp_path, monkeypatch):
    from teadata.release import DataRelease
    fixture = edition_index_fixture
    student, ref, _ = fixture["leaves"]["student"]
    staff, _, _ = fixture["leaves"]["staff"]
    def change(child):
        next(row for row in child["field_bindings"] if row["field"] == "amount")["dependencies"] = [[staff.identity, "amount"]]
    catalog = index_with_child(fixture, student, change)
    document = catalog.to_dict()
    for identifier, entry in document["editions"].items():
        fixture["root"].with_name(entry["metadata"]["location"]).write_bytes(catalog.edition_metadata_bytes(identifier))
    fixture["root"].write_text(canonical_json({"schema_version": 4, "catalog": document}))
    digest = fixture["legacy"].to_dict()["editions"][staff.identity]["column_schema_sha256"]
    schema = json.loads(fixture["resources"][digest])
    schema["datasets"][0]["tables"][0]["fields"][2]["label"] = "Different scientific meaning"
    # Corruption is sufficient to require rejection before ANY leaf transport;
    # existing v3 well-hashed mapping contradiction coverage uses this same validator.
    fixture["root"].with_name(digest + ".json").write_text(canonical_json(schema))
    release = DataRelease(fixture["root"])
    original = release._resource_path
    def selected(location, sha, size):
        assert not location.endswith(".parquet"), "Foreign dependency was not validated before leaf transport"
        return original(location, sha, size)
    monkeypatch.setattr(release, "_resource_path", selected)
    with pytest.raises(ContractError, match="checksum|size"):
        release.artifact(ref.identity)


@pytest.mark.parametrize("entry", ["symlink", "directory", "fifo", "oversize"])
def test_v4_selected_child_uses_shared_regular_bounded_capture(edition_index_fixture, monkeypatch, entry):
    import os
    from teadata.release import DataRelease
    fixture = edition_index_fixture
    key, _, _ = fixture["leaves"]["student"]
    descriptor = fixture["catalog"].to_dict()["editions"][key.identity]["metadata"]
    path = fixture["root"].with_name(descriptor["location"])
    if entry == "oversize":
        with path.open("ab") as stream:
            stream.write(b"extra")
    else:
        path.unlink()
        if entry == "symlink":
            path.symlink_to(fixture["root"])
        elif entry == "directory":
            path.mkdir()
        else:
            os.mkfifo(path, 0o600)
    release = DataRelease(fixture["root"])
    monkeypatch.setattr(os, "open", lambda *a, **k: pytest.fail("Invalid child entry reached open"))
    with pytest.raises(ContractError, match="regular|bounded"):
        release.catalog.metadata(key.dataset, edition=key)


def test_v4_fresh_process_index_discovery_loads_no_heavy_modules(edition_index_fixture):
    script = r'''
import sys
from pathlib import Path
from teadata.release import DataRelease
release = DataRelease(sys.argv[1])
Path.open = lambda *a, **k: (_ for _ in ()).throw(AssertionError('index-only discovery opened a child'))
assert len(list(release.catalog.editions(include_metadata=False))) == 2
iterator = release.catalog.editions()
assert not release.catalog._edition_children
for name in ('pyarrow', 'duckdb', 'pandas', 'sqlalchemy', 'teadata.engine'):
    assert name not in sys.modules, name
'''
    result = subprocess.run([sys.executable, "-c", script, str(edition_index_fixture["root"])],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_v4_complete_root_cap_still_precedes_child_resolution(edition_index_fixture, tmp_path, monkeypatch):
    from teadata.release import DataRelease, publish_release
    cap = 4 * 1024 * 1024
    document = edition_index_fixture["catalog"].to_dict()
    size = len((canonical_json({"schema_version": 4, "catalog": document}) + "\n").encode())
    document["description"] = "x" * (cap - size)
    document["build_id"] = sha256(canonical_json({k: v for k, v in document.items() if k != "build_id"}).encode()).hexdigest()
    raw = (canonical_json({"schema_version": 4, "catalog": document}) + "\n").encode()
    assert len(raw) == cap
    path = tmp_path / "v4-exact-cap.json"
    path.write_bytes(raw)
    release = DataRelease(path)
    assert len(list(release.catalog.editions(include_metadata=False))) == 2
    path.write_bytes(raw + b" ")
    with pytest.raises(ContractError, match="too large"):
        DataRelease(path)
    document["description"] += "x"
    document["build_id"] = sha256(canonical_json({k: v for k, v in document.items() if k != "build_id"}).encode()).hexdigest()
    catalog = DatasetCatalog(document)
    monkeypatch.setattr(catalog, "_edition_child", lambda *a: pytest.fail("Over-cap root resolved a child"))
    with pytest.raises(ContractError, match="too large"):
        publish_release(tmp_path / "over-cap", catalog, schema_version=4)
    assert not (tmp_path / "over-cap").exists()


def test_v4_scalar_refusal_precedes_even_selected_child_io(edition_index_fixture, monkeypatch):
    catalog = edition_index_fixture["catalog"]
    monkeypatch.setattr(catalog, "_edition_metadata_loader", lambda *a: pytest.fail("Scalar API loaded child metadata"))
    with pytest.raises(ContractError, match="Schema4 scalar interpretation"):
        catalog.measure_field("amount", "5")
