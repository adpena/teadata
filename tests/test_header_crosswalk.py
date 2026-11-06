from copy import deepcopy
from hashlib import sha256
from importlib.resources import files
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

from teadata.column_schema import (
    header_records, header_rename_map, parse_column_schema, schema_columns,
    source_schema_fingerprint, validate_selected_crosswalk,
)
from teadata.dataset_metadata import DatasetMetadata, ResolvedDataset
from teadata.entities import District
from teadata.manifest import BuildRecorder, ContractError, DatasetCatalog, canonical_json
from teadata.provenance import ProvenanceStore
from teadata.query import Query
from teadata.teadata_config import Config


ROOT = Path(__file__).resolve().parents[1]
RESOURCE = "schemas/tapr--2025--header-crosswalk.json"


def crosswalk_document(input_sha, *, dataset="sample", label="Example Value"):
    labels, codes = ["9 Digit Campus Number", label], ["CAMPUS", "VALUE"]
    return {"schema_version": 2, "mapping_version": 1, "datasets": [{
        "dataset": dataset, "edition": "2025", "prepared_input_sha256": input_sha,
        "tables": [{"table": "student", "input_sha256": "1" * 64,
                    "source_schema_sha256": source_schema_fingerprint(labels, codes),
                    "fields": [{"position": i, "code": code, "label": human, "field": name}
                               for i, (code, human, name) in enumerate(zip(codes, labels, ["campus_number", "example_value"], strict=True), 1)]}],
    }]}


def build_bound_catalog(tmp_path, *, edition="2025", profile=None, bind_metric=False, repeated_field=None):
    source = tmp_path / "prepared.csv"
    source.write_text("campus_number,example_value\n001902001,-3\n")
    document = crosswalk_document(sha256(source.read_bytes()).hexdigest())
    document["datasets"][0]["edition"] = edition
    if repeated_field is not None:
        tables = document["datasets"][0]["tables"]
        first = tables[0]
        field = next(value for value in first["fields"] if value["field"] == repeated_field)
        tables.append({"table": "staff", "input_sha256": "2" * 64,
                       "source_schema_sha256": source_schema_fingerprint([field["label"]], [field["code"]]),
                       "fields": [{**field, "position": 1}]})
    schema = tmp_path / "schemas/crosswalk.json"
    schema.parent.mkdir()
    schema.write_text(canonical_json(document) + "\n")
    digest = sha256(schema.read_bytes()).hexdigest()
    store = ProvenanceStore(tmp_path / "provenance")
    reference_id = store.record_acquisition(schema, ResolvedDataset(
        "reference:columns", "reference", None, 2025, str(schema), DatasetMetadata(title="Test crosswalk"),
    ))
    raw = {"data_sources": {"sample": {2025: {
        "path": str(source), "metadata": {"version": edition, "references": [{
            "reference_id": "columns", "role": "schema", "path": "schemas/crosswalk.json",
            "title": "Test crosswalk", "status": "available", "reviewed_sha256": digest,
            "provenance_ids": [reference_id],
        }]}, "reader": {"column_schema": "schemas/crosswalk.json", "dtype": "str", "keep_default_na": False},
    }}}}
    if bind_metric:
        raw["data_sources"]["sample"][2025]["metadata"]["value_policy"] = "numeric"
        raw["value_policies"] = {"numeric": {"tokens": {}}}
        raw["metric_bindings"] = {"sample_value": {"dataset": "sample", "field": "example_value", "scopes": ["district"]}}
    if profile is not None:
        definition = raw["data_sources"]["sample"][2025]
        definition["reader"] = {"profiles": {profile: definition["reader"]}}
    cfg = Config.from_dict(raw)
    cfg.base_dir = tmp_path
    acquisition = store.record_acquisition(source, cfg.resolve_dataset("sample", 2025))
    raw["data_sources"]["sample"][2025]["metadata"]["provenance_ids"] = [acquisition]
    cfg = Config.from_dict(raw)
    cfg.base_dir = tmp_path
    with BuildRecorder(cfg, store) as build:
        cfg.load_df("sample", 2025, reader_profile=profile)
        catalog = build.catalog()
    return catalog, cfg, schema, source


@pytest.fixture
def bound_catalog(tmp_path):
    return build_bound_catalog(tmp_path)


@pytest.mark.parametrize("edition", ["2025", "publisher-release-A"])
@pytest.mark.parametrize("profile", [None, "named"])
def test_catalog_binds_selected_reader_and_opaque_edition(tmp_path, edition, profile):
    catalog, cfg, _, _ = build_bound_catalog(tmp_path, edition=edition, profile=profile)
    assert cfg.column_schema("sample", 2025, reader_profile=profile) == ["campus_number", "example_value"]
    source = catalog.datasets["sample"]
    assert source["resolved_year"] == 2025
    assert source["reads"][0]["profile"] == profile
    assert catalog.header_records("sample", edition=edition)[0]["edition"] == edition
    assert DatasetCatalog(catalog.to_dict()).header_mapping("sample", table="student")["VALUE"] == "example_value"


def test_catalog_persists_exact_resource_and_returns_detached_records(bound_catalog):
    catalog, cfg, schema, _ = bound_catalog
    digest = sha256(schema.read_bytes()).hexdigest()
    assert catalog.to_dict()["column_schemas"] == {digest: schema.read_text()}
    assert catalog.datasets["sample"]["column_schema_sha256"] == digest
    assert cfg.column_schema("sample", 2025) == ["campus_number", "example_value"]
    records = catalog.header_records("sample", edition="2025", table="student")
    assert records[0] == {"dataset": "sample", "edition": "2025", "table": "student", "mapping_version": 1,
                          "position": 1, "code": "CAMPUS", "label": "9 Digit Campus Number", "field": "campus_number",
                          "period": {"kind": "unknown"},
                          "input_sha256": "1" * 64, "source_schema_sha256": source_schema_fingerprint(
                              ["9 Digit Campus Number", "Example Value"], ["CAMPUS", "VALUE"])}
    records[0]["field"] = "changed"
    records[0]["period"]["kind"] = "changed"
    assert catalog.header_mapping("sample", table="student") == {"CAMPUS": "campus_number", "VALUE": "example_value"}
    assert catalog.header_records("sample", table="student")[0]["period"] == {"kind": "unknown"}
    assert cfg.header_records("sample", edition="2025", table="student")[0]["period"] == {"kind": "unknown"}


def test_query_uses_only_catalog_bound_bytes_after_files_are_gone(bound_catalog, monkeypatch):
    catalog, _, schema, source = bound_catalog
    schema.unlink()
    source.unlink()
    monkeypatch.setattr(Path, "read_bytes", lambda *_: pytest.fail("Query read external resource"))
    monkeypatch.setattr(Config, "column_contract", lambda *_args, **_kw: pytest.fail("Query read mutable config"))
    loaded = DatasetCatalog(catalog.to_dict())
    assert loaded.header_mapping("sample", edition="2025", table="student", from_="field", to="code") == {
        "campus_number": "CAMPUS", "example_value": "VALUE",
    }


def test_query_dataframe_restores_only_bound_headers(tmp_path):
    catalog, *_ = build_bound_catalog(tmp_path, bind_metric=True)
    repo = SimpleNamespace(catalog=catalog)
    district = District(uuid4(), "Test District", district_number="001902", _repo=repo)
    district.meta.update(example_value="-3", custom="literal")
    query = Query([district], repo=repo)
    frame = query.to_df(columns=["custom", "example_value"], restore_original_headers=True)
    assert frame.columns.to_list() == ["custom", "Example Value"]
    assert frame.to_numpy().tolist() == [["literal", "-3"]]
    assert frame.attrs["teadata"] == {"build_id": catalog.build_id, "publisher_headers": {
        "example_value": {"dataset": "sample", "edition": "2025", "table": "student",
                          "label": "Example Value", "code": "VALUE"}}}
    district.meta["Example Value"] = "collision"
    with pytest.raises(ContractError, match="ambiguous"):
        query.to_df(columns=["example_value", "Example Value"], restore_original_headers=True)
    district._repo = None
    with pytest.raises(ContractError, match="entities bound to the snapshot"):
        query.to_df(restore_original_headers=True)


@pytest.mark.parametrize("repeated", ["campus_number", "example_value"])
def test_snapshot_header_ambiguity_is_scoped_to_requested_field(tmp_path, repeated):
    catalog, *_ = build_bound_catalog(tmp_path, bind_metric=True, repeated_field=repeated)
    if repeated == "example_value":
        with pytest.raises(ContractError, match="ambiguous for 'example_value'"):
            catalog.bound_field_headers(["example_value"], scope="district")
    else:
        assert catalog.bound_field_headers(["example_value"], scope="district")["example_value"]["code"] == "VALUE"


def test_selected_edition_and_unknown_table_are_explicit(bound_catalog):
    catalog = bound_catalog[0]
    with pytest.raises(ContractError, match="edition unavailable"):
        catalog.header_records("sample", edition="2024")
    with pytest.raises(ContractError, match="table unavailable"):
        catalog.header_mapping("sample", table="staff")


def test_older_catalog_does_not_borrow_package_or_current_yaml(bound_catalog):
    document = bound_catalog[0].to_dict()
    document.pop("column_schemas")
    document["datasets"]["sample"].pop("column_schema_sha256")
    document["build_id"] = sha256(canonical_json({k: v for k, v in document.items() if k != "build_id"}).encode()).hexdigest()
    catalog = DatasetCatalog(document)
    with pytest.raises(ContractError, match="crosswalk unavailable"):
        catalog.header_records("sample")


def test_rename_maps_detect_reverse_ambiguity_and_external_collisions(bound_catalog):
    catalog = bound_catalog[0]
    with pytest.raises(ContractError, match="external destination"):
        catalog.header_mapping("sample", table="student", columns=["CAMPUS", "campus_number"])
    with pytest.raises(ContractError, match="duplicate"):
        catalog.header_mapping("sample", table="student", columns=["CAMPUS", "CAMPUS"])
    assert catalog.header_mapping("sample", table="student", columns=["VALUE", "external"]) == {
        "VALUE": "example_value",
    }
    document = crosswalk_document("0" * 64, label="9 Digit Campus Number")
    records = header_records(parse_column_schema(canonical_json(document).encode()), dataset="sample", edition="2025", table="student")
    with pytest.raises(ValueError, match="Ambiguous"):
        header_rename_map(records, from_="label", to="field")
    with pytest.raises(ValueError, match="Ambiguous"):
        header_rename_map(records, from_="code", to="label")


def test_partial_rename_mapping_is_usable_by_pandas_polars_and_arrow(bound_catalog):
    import pandas as pd
    import pyarrow as pa
    pl = pytest.importorskip("polars")
    values = {"VALUE": ["-3"], "external": ["unchanged"]}
    rename = bound_catalog[0].header_mapping("sample", table="student", columns=values.keys())
    assert list(pd.DataFrame(values).rename(columns=rename).columns) == ["example_value", "external"]
    assert pl.DataFrame(values).rename(rename).columns == ["example_value", "external"]
    table = pa.table(values)
    renamed = table.rename_columns([rename.get(name, name) for name in table.column_names])
    assert renamed.column_names == ["example_value", "external"]
    assert renamed.column("example_value").to_pylist() == ["-3"]


@pytest.mark.parametrize("change", ["position", "empty_code", "duplicate_code", "field_collision", "schema_hash", "dataset_edition"])
def test_resource_rejects_ambiguous_or_misbound_mapping_records(change):
    original = crosswalk_document("0" * 64)
    document = deepcopy(original)
    table = document["datasets"][0]["tables"][0]
    if change == "position":
        table["fields"][1]["position"] = 1
    elif change == "empty_code":
        table["fields"][1]["code"] = ""
    elif change == "duplicate_code":
        table["fields"][1]["code"] = " CAMPUS "
    elif change == "field_collision":
        table["fields"][1]["field"] = "campus_number"
    elif change == "schema_hash":
        table["fields"][1]["label"] = "Changed publisher label"
    else:
        document["datasets"].append(deepcopy(document["datasets"][0]))
    with pytest.raises(ValueError):
        parse_column_schema(canonical_json(document).encode())
    assert original["datasets"][0]["tables"][0]["fields"][1]["code"] == "VALUE"


@pytest.mark.parametrize("change", ["checksum", "reference", "columns", "input", "edition", "profile", "profile_schema"])
def test_embedded_crosswalk_is_bound_to_source_and_schema(bound_catalog, change):
    document = bound_catalog[0].to_dict()
    source = document["datasets"]["sample"]
    if change == "checksum":
        key = source["column_schema_sha256"]
        document["column_schemas"][key] += " "
    elif change == "reference":
        source["column_schema_sha256"] = "0" * 64
    elif change == "columns":
        source["reads"][0]["columns"].reverse()
    elif change == "edition":
        source["resolved_year"] = 2024
    elif change == "profile":
        source["reads"][0]["profile"] = "undeclared"
    elif change == "profile_schema":
        source["reader"]["profiles"] = {"other": {"column_schema": "schemas/unreviewed.json"}}
        source["reads"][0]["profile"] = "other"
    else:
        digest = source["column_schema_sha256"]
        mapping = json.loads(document["column_schemas"][digest])
        mapping["datasets"][0]["prepared_input_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="different prepared input"):
            validate_selected_crosswalk(mapping, dataset="sample", edition="2025",
                                        input_sha256=source["artifact"]["sha256"], columns=source["reads"][0]["columns"])
        return
    document["build_id"] = sha256(canonical_json({k: v for k, v in document.items() if k != "build_id"}).encode()).hexdigest()
    with pytest.raises(ContractError, match="(?i)(crosswalk|schema)"):
        DatasetCatalog(document)


def test_packaged_crosswalk_preserves_qualified_publisher_headers_and_current_fields():
    data = files("teadata").joinpath(RESOURCE).read_bytes()
    assert sha256(data).hexdigest() == "9304bb790dbb7322602e35c4d119081a8b7d375c2ab968cc33753aa429a9be4b"
    document = parse_column_schema(data)
    assert document["mapping_version"] == 3
    campus = header_records(document, dataset="campus_tapr_student_staff_profile", edition="2025", table="student")
    assert campus[0]["code"] == "CAMPUS" and campus[0]["label"] == "9 Digit Campus Number"
    assert campus[0]["field"] == "campus_number" and campus[0]["position"] == 1
    assert campus[0]["input_sha256"] == "27947f7ace1dba61aa81336302ebc39ca78d453b129a1f87d8448de1a21024fa"
    assert next(r for r in campus if r["code"] == "CPNTALLC")["field"] == "campus_2025_student_enrollment_all_students_count"
    for scope, size in (("campus", 330), ("district", 341)):
        dataset = scope + "_tapr_student_staff_profile"
        assert len(schema_columns(document, dataset=dataset, edition="2025")) == size
        fields = {row["code"]: row for row in header_records(document, dataset=dataset, edition="2025", table="student")}
        for code, name in (("IND", "American Indian"), ("ASI", "Asian")):
            for suffix, unit in (("C", "Count"), ("P", "Percent")):
                field = fields[scope[0].upper() + "PNT" + code + suffix]
                assert field["label"] == f"{scope.title()} 2025 Student Enrollment: {name} {unit}"
                assert field["field"] == f"{scope}_2025_student_enrollment_{name.lower().replace(' ', '_')}_{unit.lower()}"
    with pytest.raises(ValueError, match="unavailable"):
        header_records(document, dataset="campus_tapr_student_staff_profile", edition="2024")


def test_legacy_catalog_does_not_infer_new_package_crosswalk(catalog):
    # This valid legacy catalog has no frozen header resource. Its behavior must
    # not depend on whichever local release manifest a later build activates.
    assert not catalog.to_dict().get("column_schemas")
    with pytest.raises(ContractError, match="crosswalk unavailable"):
        catalog.header_records("campus_tapr_student_staff_profile", edition="2025")


def test_explicit_packaged_source_contract_is_usable_without_a_data_release(monkeypatch):
    from teadata import load_config
    monkeypatch.delenv("TEADATA_RELEASE", raising=False)
    monkeypatch.delenv("TEADATA_RELEASE_URL", raising=False)
    cfg = load_config()
    campus = cfg.header_records("campus_tapr_student_staff_profile", edition="2025")
    district = cfg.header_records("district_tapr_student_staff_profile", edition="2025")
    assert len(campus) == 334 and len(district) == 343
    assert len({r["field"] for r in campus}) == 330 and len({r["field"] for r in district}) == 341
    assert cfg.header_mapping("district_tapr_student_staff_profile", edition="2025", table="student")["DISTRICT"] == "district_number"
    with pytest.raises(ContractError, match="unavailable"):
        cfg.header_records("campus_tapr_student_staff_profile", edition="2026")
    with pytest.raises(ContractError, match="unavailable"):
        cfg.header_records("campus_tapr_student_staff_profile", edition="2024")
    with pytest.raises(ContractError, match="external destination"):
        cfg.header_mapping("campus_tapr_student_staff_profile", edition="2025", table="student", columns=["CAMPUS", "campus_number"])


def test_packaged_source_contract_import_and_discovery_open_metadata_only():
    code = '''
import builtins, importlib.abc, os, pathlib, socket, sqlite3, sys, urllib.request
heavy = ('teadata.classes', 'teadata.engine', 'teadata.entities', 'teadata.geometry', 'pyarrow', 'duckdb', 'sqlalchemy', 'pandas', 'numpy', 'geopandas', 'shapely')
class Trap(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if any(fullname == p or fullname.startswith(p + '.') for p in heavy): raise AssertionError('Heavy import: ' + fullname)
sys.meta_path.insert(0, Trap())
from teadata import load_config, DatasetCatalog
import teadata.teadata_config as config_module
root = pathlib.Path(config_module.__file__).parent
allowed = {(root / 'teadata_sources.yaml').resolve(), (root / 'schemas/tapr--2025--header-crosswalk.json').resolve()}
original = pathlib.Path.open
def guarded(self, *args, **kwargs):
 if self.resolve() not in allowed: raise AssertionError('Non-metadata open: ' + str(self))
 return original(self, *args, **kwargs)
pathlib.Path.open = guarded
def forbidden(*args, **kwargs): raise AssertionError('Data/network/database I/O')
socket.create_connection = forbidden
sqlite3.connect = forbidden
urllib.request.urlopen = forbidden
os.environ.pop('TEADATA_RELEASE', None)
os.environ.pop('TEADATA_RELEASE_URL', None)
os.environ.pop('TEADATA_DATA_DIR', None)
cfg = load_config()
campus = cfg.header_records('campus_tapr_student_staff_profile', edition='2025')
district = cfg.header_records('district_tapr_student_staff_profile', edition='2025')
assert len(campus) + len(district) == 677
assert len({r['field'] for r in campus}) == 330
assert len({r['field'] for r in district}) == 341
assert cfg.header_mapping('campus_tapr_student_staff_profile', edition='2025', table='student')['CAMPUS'] == 'campus_number'
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in heavy)
'''
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_metadata_public_import_and_discovery_do_not_load_heavy_modules(bound_catalog, tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(canonical_json(bound_catalog[0].to_dict()))
    code = '''
import builtins, importlib.abc, json, pathlib, socket, sqlite3, sys, urllib.request
heavy = ('teadata.classes', 'teadata.engine', 'teadata.entities', 'teadata.geometry', 'pyarrow', 'duckdb', 'sqlalchemy', 'pandas', 'numpy', 'geopandas', 'shapely')
class Trap(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if any(fullname == p or fullname.startswith(p + '.') for p in heavy): raise AssertionError('Heavy import: ' + fullname)
sys.meta_path.insert(0, Trap())
from teadata import DatasetCatalog, DataPeriod
import teadata
document = json.loads(pathlib.Path(sys.argv[1]).read_text())
def forbidden(*args, **kwargs): raise AssertionError('Query performed I/O')
builtins.open = forbidden
pathlib.Path.open = forbidden
pathlib.Path.read_bytes = forbidden
pathlib.Path.read_text = forbidden
socket.create_connection = forbidden
sqlite3.connect = forbidden
urllib.request.urlopen = forbidden
catalog = DatasetCatalog(document)
assert catalog.header_mapping('sample', table='student')['CAMPUS'] == 'campus_number'
assert catalog.header_records('sample')[0]['edition'] == '2025'
assert 'DataEngine' in dir(teadata)
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in heavy)
'''
    result = subprocess.run([sys.executable, "-B", "-c", code, str(catalog_path)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_lazy_heavy_exports_preserve_existing_snapshot_class_paths():
    from teadata import DataEngine, Campus, District
    from teadata import classes
    assert DataEngine is classes.DataEngine and Campus is classes.Campus and District is classes.District
    assert all(cls.__module__ == "teadata.classes" for cls in (DataEngine, Campus, District))
