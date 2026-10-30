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
    "build_tapr_2025_profiles",
    "download_campus_peims_financials",
    "flatten_districts_geojson",
    "geocode_tepsac_data",
    "rename_tapr_campus_headers",
    "rename_tapr_district_headers",
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


def test_downloader_uses_explicit_paths_without_loading_snapshot(monkeypatch, tmp_path):
    from teadata.scripts import download_campus_peims_financials as script

    source = tmp_path / "statewide.csv"
    output_dir = tmp_path / "reports"
    pd.DataFrame(
        {"campus_number": [123, 456], "operating_payroll_gf": [None, 100]}
    ).to_csv(source, index=False)
    monkeypatch.setenv("TEADATA_DATA_DIR", str(tmp_path / "unused"))
    monkeypatch.setattr(script.time, "sleep", lambda _: None)
    downloads = []

    def download(url, destination, session):
        downloads.append(url)
        destination.write_bytes(b"download fixture")

    monkeypatch.setattr(script, "download_file", download)
    script.cli(["--input", str(source), "--output-dir", str(output_dir)])

    assert (
        output_dir / "2023-2024 PEIMS_Campus_Data_000000123.xlsx"
    ).read_bytes() == b"download fixture"
    assert len(downloads) == 1
    assert "which_camp=000000123" in downloads[0]


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


def test_tapr_builder_accepts_external_input_and_output_dirs(tmp_path, monkeypatch):
    from hashlib import sha256
    from teadata.column_schema import source_schema_fingerprint
    from teadata.scripts import build_tapr_2025_profiles as script

    raw = tmp_path / "raw"
    raw.mkdir()
    for kind, number, label in [
        ("Campus", "000000123", "9 Digit Campus Number"),
        ("District", "000456", "6 Digit County District Number"),
    ]:
        for group in ("Student", "Staff"):
            code = "CAMPUS" if kind == "Campus" else "DISTRICT"
            (raw / f"2025 {kind} {group} Information.csv").write_text(
                f"{label},Example Value\n{code},VALUE\n{number},42\n"
            )
    def mappings(_cfg, dataset):
        campus = dataset.startswith("campus_")
        kind = "Campus" if campus else "District"
        labels = ["9 Digit Campus Number" if campus else "6 Digit County District Number", "Example Value"]
        codes = ["CAMPUS" if campus else "DISTRICT", "VALUE"]
        fields = ["campus_number" if campus else "district_number", "example_value"]
        return {group.lower(): {"table": group.lower(),
                "input_sha256": sha256((raw / f"2025 {kind} {group} Information.csv").read_bytes()).hexdigest(),
                "source_schema_sha256": source_schema_fingerprint(labels, codes),
                "fields": [{"position": i, "code": code, "label": label, "field": field}
                           for i, (code, label, field) in enumerate(zip(codes, labels, fields, strict=True), 1)]}
                for group in ("Student", "Staff")}
    monkeypatch.setattr(script, "_header_tables", mappings)
    output_dir = tmp_path / "output"
    script.main(["--input-dir", str(raw), "--output-dir", str(output_dir)])

    campus = pd.read_excel(output_dir / "CAMPPROF_2024-2025_FINAL.xlsx")
    district = pd.read_excel(output_dir / "DISTPROF_2024-2025_FINAL.xlsx")
    assert campus.loc[0, "campus_number"] == 123
    assert district.loc[0, "district_number"] == 456
    assert len(list(output_dir.glob("*.xlsx"))) == 4


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
