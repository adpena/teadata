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
