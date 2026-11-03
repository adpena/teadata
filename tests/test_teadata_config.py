from typing import Any

import pytest

from teadata.teadata_config import Config, SpatialSelection, load_config, load_config_bytes


def test_load_table_json_respects_explicit_lines_kwarg(tmp_path):
    path = tmp_path / "records.json"
    path.write_text('[{"alpha": 1}, {"alpha": 2}]', encoding="utf-8")

    cfg = Config()
    df = cfg._load_table(str(path), lines=False)

    assert list(df["alpha"]) == [1, 2]


@pytest.mark.parametrize('suffix,raw', [
    ('.yaml', b'data_sources:\r\n  sample:\r\n    2025:\r\n      path: inputs/old.csv\r\n      metadata: {version: "2025"}\r\n'),
    ('.toml', b'[data_sources.sample.2025]\r\npath = "inputs/old.csv"\r\n[data_sources.sample.2025.metadata]\r\nversion = "2025"\r\n'),
    ('', b'data_sources: {sample: {2025: {path: inputs/old.csv, metadata: {version: "2025"}}}}'),
])
def test_captured_config_uses_same_format_original_base_and_fallback_without_reopening(tmp_path, monkeypatch, suffix, raw):
    path = tmp_path / ('original' + suffix)
    path.write_bytes(raw)
    original = load_config(path)
    path.unlink()
    monkeypatch.setattr(type(path), 'read_bytes', lambda self: pytest.fail('Captured Config reopened a path'))
    monkeypatch.setattr(type(path), 'read_text', lambda self, *a, **k: pytest.fail('Captured Config reopened a path'))
    captured = load_config_bytes(raw, path=path)
    assert captured.to_json() == original.to_json()
    assert captured.base_dir == tmp_path.resolve()
    source = captured.resolve_dataset('sample', 2026, strict=True)
    assert source.path == str(tmp_path / 'inputs/old.csv')
    assert source.requested_year == 2026 and source.resolved_year == 2025


@pytest.mark.parametrize("strict", [False, True])
def test_exact_catalog_year_is_opt_in_and_preserves_prior_fallback(tmp_path, strict):
    path = tmp_path / "values.csv"
    path.write_text("value\n0\n-999\n")
    cfg = Config.from_dict({"data_sources": {"sample": {2025: str(path)}}})
    assert cfg.resolve("sample", 2026, strict=strict) == (2025, str(path))
    assert cfg.resolve_dataset("sample", 2026, strict=strict).is_fallback is True
    assert cfg.data_sources["sample"].resolve(2026, strict=strict) == (2025, str(path))
    for resolve in (cfg.resolve, cfg.resolve_dataset):
        with pytest.raises(KeyError, match="No exact data"):
            resolve("sample", 2026, strict=strict, exact=True)
    year, frame = cfg.load_df("sample", 2025, exact=True)
    assert year == 2025 and frame["value"].tolist() == [0, -999]
    assert cfg.resolve_dataset("sample", 2025, exact=True).is_fallback is False


@pytest.mark.parametrize("method", ["resolve", "resolve_dataset", "load_df"])
def test_exact_missing_or_undated_selection_fails_before_source_io(tmp_path, monkeypatch, method):
    cfg = Config.from_dict({"data_sources": {
        "annual": {2025: str(tmp_path / "absent.csv")},
        "floating": {"current": str(tmp_path / "current.csv")},
    }})
    monkeypatch.setattr(cfg, "_load_table", lambda *a, **k: pytest.fail("Unexpected source read"))
    resolve = getattr(cfg, method)
    for dataset in ("annual", "floating"):
        with pytest.raises(KeyError, match="No exact data"):
            resolve(dataset, 2026, exact=True)
        with pytest.raises(ValueError, match="explicit catalog year"):
            resolve(dataset, None, exact=True)
        with pytest.raises(ValueError, match="dated catalog year"):
            resolve(dataset, 9999, exact=True)


def spatial_document(tmp_path) -> dict[str, Any]:
    def entry(dataset, year):
        return {"path": str(tmp_path / f"{dataset}-{year}.csv"),
                "metadata": {"title": f"Fixture {dataset}", "version": str(year),
                             "period": {"kind": "school_year", "start_year": year - 1, "end_year": year}},
                "reader": {"dtype": "str", "keep_default_na": False}}
    return {"data_sources": {"report": {2025: entry("report", 2025)}},
            "spatial": {"districts": {2025: entry("districts", 2025), 2026: entry("districts", 2026)},
                        "campuses": {2025: entry("campuses", 2025)}}}


def test_independent_spatial_requests_retain_actual_years_and_metadata_in_build(tmp_path):
    from pathlib import Path
    from teadata.manifest import BuildRecorder
    from teadata.provenance import ProvenanceStore

    document = spatial_document(tmp_path)
    document["options"] = {"spatial_years": {"districts": 2026, "campuses": 2025}}
    cfg = Config.from_dict(document)
    assert cfg.resolve_dataset("districts", 2025, section="spatial").resolved_year == 2025
    store = ProvenanceStore(tmp_path / "provenance")
    selections: list[tuple[str, str, SpatialSelection]] = [
        ("data_sources", "report", {"year": 2025, "exact": False}),
        *(("spatial", dataset, cfg.spatial_selection(dataset, 2025)) for dataset in ("districts", "campuses"))]
    for section, dataset, request in selections:
        source = cfg.resolve_dataset(dataset, section=section, **request)
        Path(source.path).write_text("value\n0\n-999\n")
        identifier = store.record_acquisition(source.path, source)
        document[section][dataset][source.resolved_year]["metadata"]["provenance_ids"] = [identifier]
    cfg = Config.from_dict(document)
    with BuildRecorder(cfg, store) as build:
        for section, dataset, request in selections:
            year, frame = cfg.load_df(dataset, section=section, **request)
            assert year == request["year"] and frame["value"].tolist() == ["0", "-999"]
        catalog = build.catalog()
    for dataset, year in (("report", 2025), ("districts", 2026), ("campuses", 2025)):
        retained = build.datasets[dataset]
        assert retained["requested_year"] == retained["resolved_year"] == year
        assert catalog.metadata(dataset).version == str(year)
        assert catalog.metadata(dataset).period.to_dict() == {
            "kind": "school_year", "start_year": year - 1, "end_year": year}
        assert retained["reads"][0]["rows"] == 2
        assert catalog.acquisitions(dataset)[0]["downloaded_at"] is None


@pytest.mark.parametrize("reporting_year,expected,fallback", [(2025, 2025, False), (2027, 2026, True), (None, 2026, None)])
def test_spatial_default_keeps_reporting_year_prior_fallback_and_latest(tmp_path, reporting_year, expected, fallback):
    cfg = Config.from_dict(spatial_document(tmp_path))
    request = cfg.spatial_selection("districts", reporting_year)
    assert request == {"year": reporting_year, "exact": False}
    source = cfg.resolve_dataset("districts", section="spatial", **request)
    assert source.requested_year == reporting_year and source.resolved_year == expected
    assert source.is_fallback is fallback
    assert "spatial_years" not in cfg.options


@pytest.mark.parametrize("value", [True, False, "2026", 2026.0, None, [], {}])
def test_spatial_override_requires_integer_without_coercion_before_source_io(tmp_path, monkeypatch, value):
    document = spatial_document(tmp_path)
    document["options"] = {"spatial_years": {"districts": value}}
    monkeypatch.setattr(Config, "_load_table", lambda *a, **k: pytest.fail("Invalid selection read source rows"))
    with pytest.raises(ValueError, match="integer catalog year"):
        Config.from_dict(document)


@pytest.mark.parametrize("overrides", [None, [], "2026", {"report": 2025}, {"district": 2026}])
def test_invalid_spatial_mapping_or_nonspatial_keys_fail_before_build(tmp_path, overrides):
    document = spatial_document(tmp_path)
    document["options"] = {"spatial_years": overrides}
    with pytest.raises(ValueError, match="spatial_years"):
        Config.from_dict(document)


@pytest.mark.parametrize("year,error,message", [
    (2027, KeyError, "No exact data"), (2008, KeyError, "Requested year"),
    (9999, ValueError, "dated catalog year"),
])
def test_spatial_override_reuses_exact_year_validation_and_never_falls_back(tmp_path, year, error, message):
    document = spatial_document(tmp_path)
    document["options"] = {"spatial_years": {"districts": year}}
    with pytest.raises(error, match=message):
        Config.from_dict(document)


def test_mutated_spatial_override_is_checked_even_before_another_layer_is_selected(tmp_path):
    cfg = Config.from_dict(spatial_document(tmp_path))
    cfg.options["spatial_years"] = {"districts": 2027}
    with pytest.raises(KeyError, match="No exact data"):
        cfg.spatial_selection("campuses", 2025)


def test_build_path_guard_rejects_prior_geometry_path_with_explicit_override(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from teadata import load_data
    from teadata.manifest import ContractError

    document = spatial_document(tmp_path)
    document["options"] = {"spatial_years": {"districts": 2026}}
    cfg = Config.from_dict(document)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(load_data, "sys", SimpleNamespace(flags=SimpleNamespace(hash_randomization=0)))
    monkeypatch.setattr(load_data, "YEAR", 2025)
    monkeypatch.setattr(load_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(load_data, "_build_repo", lambda *args: pytest.fail("Mismatched paths reached a build"))
    with pytest.raises(ContractError, match="Build paths must match"):
        load_data.load_repo(document["spatial"]["districts"][2025]["path"],
                            document["spatial"]["campuses"][2025]["path"])


def test_geometry_reads_keep_admitted_snapshot_when_config_file_changes_after_path_guard(tmp_path, monkeypatch):
    from copy import deepcopy
    from pathlib import Path
    from types import SimpleNamespace
    import yaml
    from teadata import load_data
    from teadata.manifest import ACTIVE_BUILD
    from teadata.provenance import ProvenanceStore

    document = spatial_document(tmp_path)
    document["options"] = {"spatial_years": {"districts": 2026, "campuses": 2025}}
    cfg = Config.from_dict(document)
    store = ProvenanceStore(tmp_path / "provenance")
    for section in ("data_sources", "spatial"):
        for dataset, editions in document[section].items():
            for year, entry in editions.items():
                source = cfg.resolve_dataset(dataset, year, section=section)
                Path(source.path).write_text(f"DISTRICT_C,value\n123456,{year}\n")
                entry["metadata"]["provenance_ids"] = [store.record_acquisition(source.path, source)]
    config_path = tmp_path / "sources.yaml"
    config_path.write_text(yaml.safe_dump(document))
    original_build = load_data._build_repo
    original_table = Config._load_table
    reads, retained = [], {}

    class EntityConstructionReached(Exception):
        pass

    def change_config_after_guard(*args, **kwargs):
        changed = deepcopy(document)
        changed["options"]["spatial_years"]["districts"] = 2025
        config_path.write_text(yaml.safe_dump(changed))
        return original_build(*args, **kwargs)

    def read_table(self, path, **kwargs):
        frame = original_table(self, path, **kwargs)
        reads.append((Path(path).name, frame["value"].tolist()))
        return frame

    def stop_before_entities():
        build = ACTIVE_BUILD.get()
        assert build is not None
        retained.update({name: (source["requested_year"], source["resolved_year"], source["metadata"]["version"])
                         for name, source in build.datasets.items()})
        raise EntityConstructionReached

    # Startup hash-seed enforcement has separate real-interpreter tests.
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("TEADATA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("TEADATA_PROVENANCE_DIR", str(tmp_path / "provenance"))
    monkeypatch.setattr(load_data, "sys", SimpleNamespace(flags=SimpleNamespace(hash_randomization=0)))
    monkeypatch.setattr(load_data, "CFG", str(config_path))
    monkeypatch.setattr(load_data, "YEAR", 2025)
    monkeypatch.setattr(load_data, "DataEngine", lambda: SimpleNamespace(bulk=stop_before_entities))
    monkeypatch.setattr(load_data, "_build_repo", change_config_after_guard)
    monkeypatch.setattr(Config, "_load_table", read_table)
    with pytest.raises(EntityConstructionReached):
        load_data.load_repo(document["spatial"]["districts"][2026]["path"],
                            document["spatial"]["campuses"][2025]["path"])
    assert reads == [("districts-2026.csv", ["2026"]), ("campuses-2025.csv", ["2025"])]
    assert retained == {"districts": (2026, 2026, "2026"), "campuses": (2025, 2025, "2025")}
    assert load_config(config_path).spatial_selection("districts", 2025) == {"year": 2025, "exact": True}
