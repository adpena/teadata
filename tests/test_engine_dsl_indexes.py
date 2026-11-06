from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import math
import pickle
import sys

import uuid
from threading import Event

import pytest


from teadata.classes import Campus, DataEngine, District
from teadata.engine import _SphericalCampusIndex
from teadata.geometry import haversine_miles
from teadata.query import Query



def _district(name: str, number: str) -> District:
    return District(
        id=uuid.uuid4(),
        name=name,
        district_number=number,
        enrollment=1000,
        rating="A",
    )


def _campus(
    district: District,
    *,
    name: str,
    campus_number: str,
    lon: float,
    lat: float,
    is_charter: bool = False,
) -> Campus:
    return Campus(
        id=uuid.uuid4(),
        district_id=district.id,
        name=name,
        charter_type="Charter" if is_charter else "Public",
        is_charter=is_charter,
        district_number=district.district_number,
        campus_number=campus_number,
        location=(lon, lat),
    )


def test_district_number_is_unique_and_name_returns_all_matches():
    repo = DataEngine()
    d1 = _district("Alpha ISD", "123")
    d2 = _district("Alpha ISD", "'000124")
    d3 = _district("Beta ISD", "456")
    repo.add_district(d1)
    repo.add_district(d2)
    repo.add_district(d3)

    number_hits = repo >> ("district", 123)
    assert [d.id for d in number_hits] == [d1.id]

    exact_hits = repo >> ("district", "ALPHA ISD")
    assert [d.id for d in exact_hits] == [d1.id, d2.id]

    wildcard_hits = repo >> ("district", "ALPHA*")
    assert [d.id for d in wildcard_hits] == [d1.id, d2.id]


def test_campus_lookup_by_exact_name_returns_all_matches():
    repo = DataEngine()
    d1 = _district("Gamma ISD", "111111")
    d2 = _district("Delta ISD", "222222")
    repo.add_district(d1)
    repo.add_district(d2)
    c1 = _campus(
        d1,
        name="Central High",
        campus_number="111111001",
        lon=-97.0,
        lat=32.0,
    )
    c2 = _campus(
        d2,
        name="Central High",
        campus_number="222222001",
        lon=-97.1,
        lat=32.1,
    )
    repo.add_campus(c1)
    repo.add_campus(c2)

    hits = repo >> ("campus", "CENTRAL HIGH")
    assert [c.id for c in hits] == [c1.id, c2.id]


def test_nearest_campuses_charter_only_returns_empty_when_no_charters():
    repo = DataEngine()
    d1 = _district("Echo ISD", "333333")
    repo.add_district(d1)
    repo.add_campus(
        _campus(
            d1,
            name="Echo Middle",
            campus_number="333333001",
            lon=-97.0,
            lat=32.0,
        )
    )

    hits = repo.nearest_campuses(-97.0, 32.0, limit=3, charter_only=True)
    assert hits == []


def test_nearest_campuses_respects_max_miles():
    repo = DataEngine()
    d1 = _district("Foxtrot ISD", "444444")
    repo.add_district(d1)
    near = _campus(
        d1,
        name="Near Campus",
        campus_number="444444001",
        lon=0.0,
        lat=0.0,
    )
    far = _campus(
        d1,
        name="Far Campus",
        campus_number="444444002",
        lon=1.0,
        lat=1.0,
    )
    repo.add_campus(near)
    repo.add_campus(far)

    hits = repo.nearest_campuses(0.0, 0.0, limit=5, max_miles=10.0, geodesic=True)
    assert [c.id for c in hits] == [near.id]


@pytest.mark.parametrize("backend", ["scipy", "numpy", "scan"])
def test_great_circle_queries_at_texas_latitude(monkeypatch, backend):
    if backend == "numpy":
        monkeypatch.setitem(sys.modules, "scipy.spatial", None)
    repo = DataEngine(indexes_enabled=backend != "scan")
    district = _district("Charter ISD", "012345")
    repo.add_district(district)

    def add(number, lon, lat):
        campus = _campus(
            district, name=number, campus_number=number, lon=lon, lat=lat, is_charter=True,
        )
        campus.school_type = "Elementary"
        repo.add_campus(campus)
        return campus

    origin = (-97.0, 32.0)
    seed = add("012345001", *origin)
    north = add("012345002", -97.0, 32.095)
    east = add("012345003", -96.895, 32.0)
    tied = add("012345004", -96.895, 32.0)
    edge = haversine_miles(*origin, -96.895, 32.0)
    assert 6.1 < edge < 6.3
    assert repo.knn_campuses(*origin, 20) == [seed, east, tied, north]
    assert repo.knn_campuses(*origin, 2) == [seed, east]
    assert repo.radius_campuses(*origin, edge) == [seed, east, tied]
    assert repo.radius_campuses(*origin, math.nextafter(edge, -math.inf)) == [seed]
    assert repo.nearest_campuses(*origin, limit=2, max_miles=edge) == [seed, east]
    assert repo.radius_campuses(*origin, math.inf) == [seed, east, tied, north]
    assert repo.nearest_charter_same_type([seed])[str(seed.id)] == {"match": east, "miles": edge}
    cohort = Query([seed, north, east, tied], repo)
    assert (cohort >> ("radius", origin, edge)).to_list() == [seed, east, tied]

    projected = add("012345005", 800, 800)
    assert repo.nearest_campus(801, 800) is projected
    assert repo.radius_campuses(801, 800, 5) == []
    assert repo.knn_campuses(*origin, 20) == [seed, east, tied, north]


@pytest.mark.parametrize("bulk", [False, True])
def test_warmed_spatial_queries_see_new_and_relocated_campuses(bulk):
    repo = DataEngine()
    district = _district("Charter ISD", "012345")
    repo.add_district(district)

    def make(number, lon, charter):
        campus = _campus(
            district, name=number, campus_number=number, lon=lon, lat=32.0, is_charter=charter,
        )
        campus.school_type = "Elementary"
        return campus

    seed, closer = make("012345001", -97.1, False), make("012345002", -97.0, True)
    repo.add_campus(seed)
    assert repo.nearest_campus(-97, 32) is seed
    assert repo.nearest_campus(-97, 32, charter_only=True) is None
    assert repo.nearest_charter_same_type([seed])[str(seed.id)]["match"] is None
    with repo.bulk() if bulk else nullcontext():
        repo.add_campus(closer)
        assert repo.nearest_campus(-97, 32) is closer
    assert repo.nearest_campus(-97, 32, charter_only=True) is closer
    assert repo.nearest_charter_same_type([seed])[str(seed.id)]["match"] is closer

    closer.point = (-98, 32)
    assert repo.nearest_campus(-97, 32) is seed
    closer.school_type = "Middle"
    assert repo.nearest_charter_same_type([seed])[str(seed.id)]["match"] is None
    closer.is_charter = False
    assert repo.nearest_campus(-97, 32, charter_only=True) is None
    assert repo._campus_spatial_caches is not None
    restored = pickle.loads(pickle.dumps(repo))
    assert "_campus_spatial_caches" not in vars(restored)
    assert restored.nearest_campus(-97, 32) is restored.campuses[seed.id]


def test_saved_query_branches_and_transfer_predicate_errors(monkeypatch):
    repo = DataEngine()
    district = _district("Transfer ISD", "012345")
    first = _campus(district, name="First", campus_number="012345001", lon=-97, lat=32)
    second = _campus(district, name="Second", campus_number="012345002", lon=-97, lat=32)
    cohort = Query([second, first], repo)
    assert (cohort >> ("sort", lambda campus: campus.name)).to_list() == [first, second]
    assert (cohort >> (lambda campus: campus is second)).to_list() == [second]
    assert cohort.to_list() == [second, first]

    monkeypatch.setattr(repo, "transfers_out", lambda campus: [(None, None, True)])
    row = (first, None, None, True)
    rows = Query([row], repo)
    assert (rows >> ("where_to", lambda campus: campus is None)).to_list() == [row]
    assert (Query([first], repo) >> ("transfers_out", lambda campus: campus is None)).to_list() == [row]
    for branch, operator in ((rows, "where_to"), (cohort, "transfers_out")):
        with pytest.raises(AttributeError, match="bogus"):
            branch >> (operator, lambda campus: campus.bogus)


@pytest.mark.parametrize("same_type", [False, True])
def test_slow_index_build_cannot_republish_after_location_change(monkeypatch, same_type):
    repo = DataEngine()
    district = _district("Charter ISD", "012345")
    repo.add_district(district)
    far = _campus(
        district, name="Far", campus_number="012345001", lon=-97.1, lat=32, is_charter=True,
    )
    near = Campus(
        id=uuid.uuid4(), district_id=district.id, name="Near",
        district_number=district.district_number, campus_number="012345002", charter_type="Charter", is_charter=True,
    )
    source = _campus(
        district, name="Source", campus_number="012345003", lon=-97, lat=32,
    )
    for campus in (far, near, source):
        campus.school_type = "Elementary"
        repo.add_campus(campus)
    near.location = (-97.0, 32.0)
    start, finish = Event(), Event()
    build = _SphericalCampusIndex.build

    def pause_once(entries):
        if not start.is_set():
            start.set()
            assert finish.wait(timeout=5)
        return build(entries)

    monkeypatch.setattr(_SphericalCampusIndex, "build", staticmethod(pause_once))

    def nearest():
        if same_type:
            return repo.nearest_charter_same_type([source])[str(source.id)]["match"]
        return repo.nearest_campus(-97, 32, charter_only=True)

    with ThreadPoolExecutor(max_workers=1) as workers:
        old_query = workers.submit(nearest)
        try:
            assert start.wait(timeout=5)
            near.location = (-98.0, 32.0)
        finally:
            finish.set()
        assert old_query.result(timeout=5) is near
    assert nearest() is far
    assert math.isnan(haversine_miles(-97, 32, math.nan, 32))
