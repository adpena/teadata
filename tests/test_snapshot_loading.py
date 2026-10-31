import gzip
import pickle

from teadata import DataEngine
from teadata.release import SNAPSHOT_MAGIC
from teadata.load_data import _save_repo_snapshot


def test_from_snapshot_handles_gzip_extension(tmp_path, catalog):
    engine = DataEngine()
    engine.catalog = catalog
    gz_path = tmp_path / "repo_snapshot.pkl.gz"
    with gzip.open(gz_path, "wb") as f:
        f.write(SNAPSHOT_MAGIC + catalog.build_id.encode() + b"\n")
        pickle.dump(engine, f, protocol=pickle.HIGHEST_PROTOCOL)

    loaded = DataEngine.from_snapshot(gz_path)

    assert isinstance(loaded, DataEngine)


def test_from_snapshot_sniffs_gzip_magic_without_extension(tmp_path, catalog):
    engine = DataEngine()
    engine.catalog = catalog
    gz_path = tmp_path / "repo_snapshot.pkl"
    with gzip.open(gz_path, "wb") as f:
        f.write(SNAPSHOT_MAGIC + catalog.build_id.encode() + b"\n")
        pickle.dump(engine, f, protocol=pickle.HIGHEST_PROTOCOL)

    loaded = DataEngine.from_snapshot(gz_path)

    assert isinstance(loaded, DataEngine)


def test_save_repo_snapshot_is_deterministic(monkeypatch, tmp_path, catalog):
    monkeypatch.chdir(tmp_path)
    engine = DataEngine()
    engine.catalog = catalog
    path = _save_repo_snapshot(engine, "districts.csv", "campuses.csv")
    before = path.read_bytes()
    assert path.suffix == ".gz"
    assert _save_repo_snapshot(engine, "districts.csv", "campuses.csv").read_bytes() == before
    assert DataEngine.from_snapshot(path).catalog.build_id == catalog.build_id
