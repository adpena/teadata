import gzip
import pickle
from uuid import uuid4

import pytest

from teadata import Campus, DataEngine, District
from teadata.identifiers import EntityIdentityError
from teadata.manifest import ContractError
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


@pytest.mark.parametrize("bulk", [False, True])
def test_registration_never_replaces_uuid_or_canonical_identity(bulk):
    from contextlib import nullcontext

    repo = DataEngine()
    district = District(uuid4(), "original district", district_number="011901")
    campus = Campus(uuid4(), district.id, "original campus", "", False, campus_number="011901001")
    with repo.bulk() if bulk else nullcontext():
        repo.add_district(district)
        repo.add_campus(campus)
        repo.add_district(district)
        repo.add_campus(campus)
        repo.clear_indexes()
        with pytest.raises(EntityIdentityError, match="Duplicate District ID"):
            repo.add_district(District(district.id, "replacement", district_number="999999"))
        with pytest.raises(EntityIdentityError, match="Duplicate Campus ID"):
            repo.add_campus(Campus(campus.id, district.id, "replacement", "", False, campus_number="011901002"))
        with pytest.raises(EntityIdentityError, match="Campus and district share entity ID"):
            repo.add_campus(Campus(district.id, district.id, "collision", "", False, campus_number="011901003"))
        with pytest.raises(EntityIdentityError, match="Duplicate canonical district number '011901'"):
            repo.add_district(District(uuid4(), "duplicate", district_number="11901"))
        with pytest.raises(EntityIdentityError, match="Duplicate canonical campus lookup key '011901001'"):
            repo.add_campus(Campus(uuid4(), district.id, "duplicate", "", False, campus_number="11901001"))
        with pytest.raises(EntityIdentityError, match="already belongs to another engine"):
            DataEngine().add_campus(campus)
        with pytest.raises(EntityIdentityError, match="registered District.district_number"):
            district.district_number = "011902"
        with pytest.raises(EntityIdentityError, match="registered Campus.campus_number"):
            campus.campus_number = "011901002"
    assert list(repo.campuses.values()) == [campus]
    assert list(repo.districts.values()) == [district]
    assert repo.campus_by_key("011901001") is campus
    assert (repo >> ("district", "011901")).first() is district
    assert repo.campuses_in(district) == [campus]


def test_snapshot_rejects_ambiguous_historical_district_rows(tmp_path, catalog):
    repo = DataEngine()
    repo.catalog = catalog
    district = District(uuid4(), "first", district_number="11901")
    repo.add_district(district)
    duplicate = District(uuid4(), "second", district_number="011901")
    duplicate._repo = repo
    repo._districts[duplicate.id] = duplicate  # Historical unchecked storage.
    path = tmp_path / "corrupt.pkl"
    with path.open("wb") as stream:
        stream.write(SNAPSHOT_MAGIC + catalog.build_id.encode() + b"\n")
        pickle.dump(repo, stream)
    with pytest.raises(ContractError, match="Duplicate canonical district number '011901'"):
        DataEngine.from_snapshot(path)


def test_snapshot_rejects_foreign_entity_owner(tmp_path, catalog):
    repo = DataEngine()
    repo.catalog = catalog
    foreign = DataEngine()
    district = District(uuid4(), "first", district_number="11901")
    foreign.add_district(district)
    repo._districts[district.id] = district
    path = tmp_path / "foreign.pkl"
    with path.open("wb") as stream:
        stream.write(SNAPSHOT_MAGIC + catalog.build_id.encode() + b"\n")
        pickle.dump(repo, stream)
    with pytest.raises(ContractError, match="different registry owner"):
        DataEngine.from_snapshot(path)


def test_engine_union_keeps_detached_identity_and_lazy_metadata():
    from teadata.persistence.sqlalchemy_store import LazyMetaDict

    loads = []
    left = DataEngine()
    district = District(uuid4(), "left", district_number="11901")
    district.district_number = "11902"
    district.meta = LazyMetaDict(lambda: loads.append("loaded") or {"field": 1})
    left.add_district(district)
    assert district.district_number_canon == "011902"
    assert (left >> ("district", "011902")).first() is district
    with pytest.raises(EntityIdentityError, match="Invalid district number"):
        District(uuid4(), "invalid", district_number="011902001")

    union = left | DataEngine()
    merged = union[district.id]
    assert loads == []
    assert district._repo is left and merged._repo is union
    merged.meta["field"] = 2
    assert loads == ["loaded"] and district.meta["field"] == 1
