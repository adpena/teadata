from teadata.teadata_config import Config


def test_load_table_json_respects_explicit_lines_kwarg(tmp_path):
    path = tmp_path / "records.json"
    path.write_text('[{"alpha": 1}, {"alpha": 2}]', encoding="utf-8")

    cfg = Config()
    df = cfg._load_table(str(path), lines=False)

    assert list(df["alpha"]) == [1, 2]
