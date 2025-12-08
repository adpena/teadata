import gzip
import pickle

from teadata import DataEngine
from teadata.load_data import _load_repo_snapshot, _save_repo_snapshot


def test_from_snapshot_handles_gzip_extension(tmp_path):
    engine = DataEngine()
    gz_path = tmp_path / "repo_snapshot.pkl.gz"
    with gzip.open(gz_path, "wb") as f:
        pickle.dump(engine, f, protocol=pickle.HIGHEST_PROTOCOL)

    loaded = DataEngine.from_snapshot(gz_path)

    assert isinstance(loaded, DataEngine)


def test_from_snapshot_sniffs_gzip_magic_without_extension(tmp_path):
    engine = DataEngine()
    gz_path = tmp_path / "repo_snapshot.pkl"
    with gzip.open(gz_path, "wb") as f:
        pickle.dump(engine, f, protocol=pickle.HIGHEST_PROTOCOL)

    loaded = DataEngine.from_snapshot(gz_path)

    assert isinstance(loaded, DataEngine)


def test_save_repo_snapshot_writes_gzip(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    districts_fp = tmp_path / "districts.csv"
    campuses_fp = tmp_path / "campuses.csv"
    districts_fp.write_text("districts")
    campuses_fp.write_text("campuses")

    engine = DataEngine()
    _save_repo_snapshot(engine, str(districts_fp), str(campuses_fp), {})

    snap = tmp_path / ".cache" / f"repo_{districts_fp.stem}_{campuses_fp.stem}.pkl"
    snap_gz = snap.with_suffix(snap.suffix + ".gz")

    assert snap.exists()
    assert snap_gz.exists()

    snap.unlink()  # force gzip-only path
    reloaded = _load_repo_snapshot(str(districts_fp), str(campuses_fp), {})

    assert isinstance(reloaded, DataEngine)
