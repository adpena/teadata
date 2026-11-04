import gzip
import importlib
import os
import pickle
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from teadata.scripts import paths


SCRIPT_MODULES = [
    "build_campus_peims_financials_statewide_data",
    "build_tapr_profiles",
    "download_campus_peims_financials",
    "flatten_districts_geojson",
    "geocode_tepsac_data",
    "rename_tapr_campus_headers",
    "rename_tepsac_headers",
]


def test_default_data_directory_follows_relocated_package(monkeypatch, tmp_path):
    package = tmp_path / "another checkout" / "teadata"
    monkeypatch.setattr(paths, "__file__", str(package / "scripts" / "paths.py"))
    monkeypatch.delenv("TEADATA_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    assert (
        paths.data_path("tapr", "input.csv") == package / "data" / "tapr" / "input.csv"
    )
    assert not package.exists()


@pytest.mark.parametrize("override", ["~/datasets", "$DATASET_ROOT", "relative-data"])
def test_data_directory_override_expands_paths(monkeypatch, tmp_path, override):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    monkeypatch.setenv("DATASET_ROOT", str(tmp_path / "external"))
    monkeypatch.setenv("TEADATA_DATA_DIR", override)
    expected = {
        "~/datasets": tmp_path / "user" / "datasets",
        "$DATASET_ROOT": tmp_path / "external",
        "relative-data": tmp_path / "relative-data",
    }[override]

    assert paths.data_path("tapr") == expected / "tapr"


@pytest.mark.parametrize("module", SCRIPT_MODULES)
def test_cli_help_without_source_data_from_another_cwd(tmp_path, module):
    env = dict(os.environ, TEADATA_DATA_DIR=str(tmp_path / "missing-data"))
    result = subprocess.run(
        [sys.executable, "-m", f"teadata.scripts.{module}", "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert not list(tmp_path.iterdir())


def test_excel_example_help_does_not_require_excel(tmp_path):
    script = Path(__file__).resolve().parents[1] / "examples" / "restyle_existing_excel.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "--sheet-name" in result.stdout
    assert not list(tmp_path.iterdir())


def test_geocoder_help_does_not_import_processing_dependencies(tmp_path):
    probe = """
import sys

class RejectProcessingImports:
    def find_spec(self, fullname, *_):
        if fullname.split(".")[0] in {"aiohttp", "geopy", "pandas", "requests", "tqdm"}:
            raise AssertionError(f"Geocoder help imported {fullname}")

sys.meta_path.insert(0, RejectProcessingImports())
from teadata.scripts.geocode_tepsac_data import main
main(["--help"])
"""
    result = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "--output" in result.stdout


@pytest.mark.parametrize(
    "module",
    [
        "download_campus_peims_financials",
        "geocode_tepsac_data",
        "rename_tepsac_headers",
    ],
)
def test_importing_scripts_does_not_read_source_files(monkeypatch, tmp_path, module):
    monkeypatch.setenv("TEADATA_DATA_DIR", str(tmp_path / "missing-data"))
    read_csv = Mock(side_effect=AssertionError("Import must not read data"))
    read_excel = Mock(side_effect=AssertionError("Import must not read data"))
    monkeypatch.setattr(pd, "read_csv", read_csv)
    monkeypatch.setattr(pd, "read_excel", read_excel)

    importlib.reload(importlib.import_module(f"teadata.scripts.{module}"))

    read_csv.assert_not_called()
    read_excel.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("explicit_paths", [False, True])
def test_tepsac_conversion_uses_selected_paths(monkeypatch, tmp_path, explicit_paths):
    from teadata.scripts.rename_tepsac_headers import main

    root = tmp_path / "datasets"
    source = root / "tepsac" / "tepsac_geocoded.csv"
    source.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "School Name": "Example School",
                "School Number": 123,
                "Grade Low": "K",
                "Grade High": "12",
                "Enrollment": 100,
                "School Full Address": "Example address",
                "School Website": "",
                "School Accreditations": "",
                "District Number": 456,
                "Best Lat": 30.0,
                "Best Lng": -97.0,
                "Best Provider": "fixture",
            }
        ]
    ).to_csv(source, index=False)
    monkeypatch.chdir(tmp_path)
    if explicit_paths:
        monkeypatch.setenv("TEADATA_DATA_DIR", str(tmp_path / "unused"))
        output = tmp_path / "output" / "schools.csv"
        main(["--input", str(source), "--output", str(output)])
    else:
        monkeypatch.setenv("TEADATA_DATA_DIR", str(root))
        output = root / "tepsac" / "tepsac_geocoded_FINAL.csv"
        main([])

    result = pd.read_csv(output)
    assert result.loc[0, "school_name"] == "Example School"
    assert result.loc[0, "school_number"] == "'000000123"
    assert result.loc[0, "district_number"] == "'000456"


def test_geocoder_uses_explicit_paths_with_stubbed_providers(monkeypatch, tmp_path):
    from teadata.scripts import geocode_tepsac_data as script

    source = tmp_path / "schools.xlsx"
    output = tmp_path / "output" / "geocoded.csv"
    pd.DataFrame(
        [
            {
                "Closed": False,
                "School Street": "123 Example St",
                "School City": "Austin",
                "School State": "TX",
                "School Zip": "78701",
            }
        ]
    ).to_excel(source, index=False)

    async def results(addresses):
        return {
            address: {"arcgis": {"lat": 30.0, "lng": -97.0, "score": 99}}
            for address in addresses
        }

    monkeypatch.setattr(script, "build_results_map_with_progress", results)
    monkeypatch.setenv("TEADATA_DATA_DIR", str(tmp_path / "unused"))
    script.main(["--input", str(source), "--output", str(output)])

    result = pd.read_csv(output)
    assert result.loc[0, "Best Provider"] == "arcgis"
    assert result.loc[0, "Best Lat"] == 30.0


@pytest.mark.parametrize("write_failure", [False, True, "missing_reference_store"])
def test_tapr_builder_accepts_external_input_and_new_output_dirs(tmp_path, monkeypatch, write_failure):
    from hashlib import sha256
    import json
    import pyarrow.parquet as pq
    from teadata.column_schema import source_schema_fingerprint
    from teadata.manifest import canonical_json
    from teadata.scripts import build_tapr_profiles as script
    from teadata.teadata_config import Config

    raw = tmp_path / "raw"
    raw.mkdir()
    datasets, entries = [], {}
    for scope, kind, number, code, label in [
        ("campus", "Campus", "000000123", "CAMPUS", "9 Digit Campus Number"),
        ("district", "District", "000456", "DISTRICT", "6 Digit County District Number"),
    ]:
        dataset = scope + "_tapr_student_staff_profile"
        labels, codes, fields = [label, "Example Value"], [code, "VALUE"], [scope + "_number", "example_value"]
        path = raw / f"2025 {kind} Student Information.csv"
        path.write_text(f"{label},Example Value\n{code},VALUE\n{number},42\n")
        paths = {"student": str(path)}
        tables = [{"table": "student", "input_sha256": sha256(path.read_bytes()).hexdigest(),
            "source_schema_sha256": source_schema_fingerprint(labels, codes),
            "fields": [{"position": i, "code": c, "label": text, "field": name}
                       for i, (c, text, name) in enumerate(zip(codes, labels, fields, strict=True), 1)]}]
        datasets.append({"dataset": dataset, "edition": "2025", "prepared_input_sha256": "0" * 64, "tables": tables})
        entries[dataset] = {2025: {"path": str(tmp_path / (scope + "-retained-reference.xlsx")),
            "metadata": {"version": "2025"},
            "reader": {"column_schema": "headers.json", "profiles": {"source": {"tapr_source": {"scope": scope, "tables": paths}}}}}}
    schema = tmp_path / "headers.json"
    schema.write_text(canonical_json({"schema_version": 2, "mapping_version": 1, "datasets": datasets}) + "\n")
    for versions in entries.values():
        versions[2025]["metadata"]["references"] = [{"reference_id": "columns", "role": "schema", "status": "available",
            "path": "headers.json", "reviewed_sha256": sha256(schema.read_bytes()).hexdigest()}]
        if write_failure == "missing_reference_store":
            versions[2025]["metadata"]["references"][0]["provenance_ids"] = ["1" * 64]
    cfg = Config.from_dict({"data_sources": entries})
    cfg.base_dir = tmp_path
    monkeypatch.setattr(script, "load_config", lambda _: cfg)
    output_dir = tmp_path / "new-output"
    if write_failure == "missing_reference_store":
        with pytest.raises(ValueError, match="explicit reference_store"):
            script.main(["--edition", "2025", "--input-dir", str(raw), "--output-dir", str(output_dir)])
        assert not output_dir.exists()
        assert not list(tmp_path.glob(".tapr-preparation-*"))
        return
    if write_failure:
        def fail(*args, **kwargs):
            raise OSError("simulated artifact write failure")
        monkeypatch.setattr(pq, "ParquetWriter", fail)
        with pytest.raises(OSError, match="simulated artifact write failure"):
            script.main(["--edition", "2025", "--input-dir", str(raw), "--output-dir", str(output_dir)])
        assert not output_dir.exists()
        assert not list(tmp_path.glob(".tapr-preparation-*"))
        return
    script.main(["--edition", "2025", "--input-dir", str(raw), "--output-dir", str(output_dir)])
    assert pq.read_table(output_dir / "campus.parquet").to_pylist() == [{"campus_number": "000000123", "example_value": "42"}]
    assert pq.read_table(output_dir / "district.parquet").to_pylist() == [{"district_number": "000456", "example_value": "42"}]
    receipt = json.loads((output_dir / "preparation.json").read_text())
    assert receipt["release_activation"] is False
    assert receipt["scopes"]["campus"]["identity_fields"] == ["campus_number"]
    from teadata.provenance import ProvenanceStore
    store = ProvenanceStore(output_dir / "provenance")
    for scope in ("campus", "district"):
        assert receipt["scopes"][scope]["source_columns_accounted"] == 2
        assert receipt["scopes"][scope]["local_reference_closure_verified"] is True
        closure = store.verify_record_closure([receipt["scopes"][scope]["processing_record_id"]])
        for record in closure.values():
            for reference in record.get("source_metadata", {}).get("references", []):
                assert reference["provenance_ids"]
                assert all(identifier in closure for identifier in reference["provenance_ids"])
    assert not list(output_dir.glob("*.xlsx"))


@pytest.mark.parametrize("scope", ["campus", "district"])
def test_packaged_2025_combined_tapr_profile_rejects_before_raw_io(tmp_path, monkeypatch, scope):
    import pyarrow.parquet as pq
    from teadata.provenance import ProvenanceStore
    from teadata.scripts.build_tapr_profiles import prepare_profiles
    from teadata.tapr import TaprSourceRequest
    from teadata.teadata_config import load_config

    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported combined preparation must reject before raw I/O or eager fallback")

    cfg = load_config()
    monkeypatch.setattr(TaprSourceRequest, "open", forbidden)
    monkeypatch.setattr(cfg, "read_tapr_sources", forbidden)
    monkeypatch.setattr(pq, "ParquetWriter", forbidden)
    monkeypatch.setattr(ProvenanceStore, "record_acquisition", forbidden)
    unread = tmp_path / "unread-source"
    combined = scope + "_tapr_student_staff_profile"
    assert len(cfg.tapr_source_requests(combined, 2025, input_dir=unread)) == 2
    for family in ("student_profile", "staff_profile"):
        request, = cfg.tapr_source_requests(scope + "_tapr_" + family, 2025, input_dir=unread)
        assert request.dataset == scope + "_tapr_" + family
        assert request.edition == "2025" and request.key_field == scope + "_number"

    output = tmp_path / "new" / "cohort"
    with pytest.raises(ValueError, match="one physical table per source; no eager merge fallback"):
        prepare_profiles(cfg, edition="2025", scopes=[scope], input_dir=unread, output_dir=output)
    assert not unread.exists() and not output.parent.exists()
    assert not list(tmp_path.rglob(".tapr-preparation-*"))


def test_snapshot_writer_only_writes_staging(monkeypatch, tmp_path, catalog):
    from teadata import load_data
    build_cache = tmp_path / "build"
    build_cache.mkdir()
    monkeypatch.setattr(load_data, "_repo_cache_dir", lambda: build_cache)
    monkeypatch.setattr(load_data, "_prepare_repo_for_pickle", lambda _: None)
    repo = SimpleNamespace(example="snapshot fixture", catalog=catalog)
    path = load_data._save_repo_snapshot(repo, "districts.geojson", "campuses.geojson")
    with gzip.open(path, "rb") as stream:
        stream.readline()
        assert pickle.load(stream).example == repo.example
    assert list(tmp_path.rglob("*.gz")) == [path]
