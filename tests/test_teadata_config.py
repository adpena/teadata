import pytest

from teadata.teadata_config import Config, load_config, load_config_bytes


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
