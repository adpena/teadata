from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import math
import pickle
import sys

import uuid
from threading import Event

import pytest


from teadata import EntitySelectionError
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


def test_district_number_and_exact_name_disambiguation():
    repo = DataEngine()
    d1 = _district("Alpha ISD", "123")
    d2 = _district("Alpha ISD", "'000124")
    d3 = _district("Beta ISD", "456")
    repo.add_district(d1)
    repo.add_district(d2)
    repo.add_district(d3)

    number_hits = repo >> ("district", 123)
    assert [d.id for d in number_hits] == [d1.id]

    with pytest.raises(EntitySelectionError, match="Multiple exact matches") as error:
        repo >> ("district", "aLpHa iSd")
    assert [item["id"] for item in error.value.to_dict()["candidates"]] == ["000123", "000124"]
    assert repo.district_named(" beta isd ") is d3
    with pytest.raises(EntitySelectionError, match="Did you mean") as error:
        repo.district_named("Bta ISD")
    assert ("BETA ISD", "000456") in error.value.candidates

    wildcard_hits = repo >> ("district", "ALPHA*")
    assert [d.id for d in wildcard_hits] == [d1.id, d2.id]


def test_campus_exact_names_use_live_registry_and_wildcards_return_multiple_matches():
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

    hits = repo >> ("campus", "central*")
    assert [c.id for c in hits] == [c1.id, c2.id]
    repo.clear_indexes()
    with pytest.raises(EntitySelectionError, match="Multiple exact matches"):
        repo >> ("campus", "central high")
    c2.name = "OTHER CENTRAL HIGH"
    assert list(repo >> ("campus", "central high")) == [c1]
    assert list(repo >> ("campus", "central*")) == [c1]
    assert list(repo >> ("campus", "other*")) == [c2]
    assert repo.campus_by_name("other central high") is c2


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


@pytest.mark.parametrize("indexed", [True, False])
def test_stock_within_respects_holes_boundaries_and_geometry_updates(indexed):
    from shapely.geometry import MultiPolygon, Polygon, box

    engine = DataEngine(indexes_enabled=indexed)
    outer = [(-99, 30), (-95, 30), (-95, 34), (-99, 34)]
    hole = [(-98, 31), (-96, 31), (-96, 33), (-98, 33)]
    geometry = MultiPolygon([Polygon(outer, [hole]), box(-94, 30, -93, 31)])
    district = _district("Geography ISD", "012345")
    district.boundary = geometry
    other = _district("Other ISD", "012346")
    engine.add_district(district)
    engine.add_district(other)

    def campus(code, point, owner=other, charter=False):
        result = _campus(owner, name=code, campus_number=code, lon=point[0], lat=point[1], is_charter=charter)
        engine.add_campus(result)
        return result

    interior = campus("012346001", (-98.5, 32))
    boundary = campus("012346002", (-99, 32), charter=True)
    in_hole = campus("012345003", (-97, 32), owner=district)
    hole_boundary = campus("012346004", (-98, 32))
    island = campus("012346005", (-93.5, 30.5))
    private = Campus(uuid.uuid4(), None, "Private", "Private", True, is_private=True,
                     source_namespace="tefa", source_id="example", location=(-98.5, 32))
    engine.add_campus(private)

    assert list(engine >> ("within", geometry)) == [interior, island, private]
    assert list(engine >> ("within", district, False, True)) == [interior, boundary, hole_boundary, island, private]
    assert list(engine >> ("within", district, True, True)) == [boundary]
    assert engine.charter_campuses_within(district) == [boundary]
    assert engine.private_campuses_within(district) == [private]
    assert len(engine.campuses_in(district)) == 1
    assert list((engine >> ("district", "012345")) >> ("within", None, None)) == [interior, island, private]
    interior.point = (-97, 32)
    assert engine.campuses_within(geometry) == [island, private]

    with pytest.raises(ValueError, match="Boolean"):
        engine >> ("within", district, "charter")
    with pytest.raises(ValueError, match="outside WGS84"):
        engine.campuses_within(box(190, 30, 191, 31))
    with pytest.raises(ValueError, match="Boundary geometry unavailable"):
        engine.campuses_within(other)

    district.boundary = outer
    assert engine.campuses_within(district) == [interior, in_hole, hole_boundary, private]


def test_digit_bearing_names_and_legacy_numeric_selectors():
    engine = DataEngine()
    district = _district("Innovation 2 ISD", "012345")
    campus = _campus(district, name="Innovation 2 Academy", campus_number="012345001", lon=-98, lat=32)
    engine.add_district(district)
    engine.add_campus(campus)
    for selector in ("Innovation 2 ISD", "Innovation 2%", "012345-001", "’012345", "'12345.0", 12345):
        assert list(engine >> ("district", selector)) == [district]
    for selector in ("Innovation 2 Academy", "Innovation 2*", "'12345001.0", 12345001):
        assert list(engine >> ("campus", selector)) == [campus]
    with pytest.raises(EntitySelectionError, match="No exact match"):
        engine >> ("district", "Unknown 012345 ISD")
    assert list(engine >> ("campus", "999999999")) == []
    with pytest.raises(ValueError, match="Invalid district number"):
        engine >> ("district", True)
    with pytest.raises(ValueError, match="take expects positional arguments: n"):
        (engine >> ("district", "012345")) >> ("take",)


@pytest.mark.parametrize("indexed", [True, False])
def test_engine_selectors_keep_governance_geometry_and_namespace_separate(indexed):
    from array import array as coordinates
    from numpy import array
    from pandas import Series
    from shapely.geometry import box

    engine = DataEngine(indexes_enabled=indexed)
    district = District(uuid.uuid4(), "Example ISD", district_number="001001", boundary=box(-98, 30, -97, 31))
    charter = District(uuid.uuid4(), "Example Charter", district_number="001801", meta={"is_charter": True})
    empty = District(uuid.uuid4(), "Empty ISD", district_number="001002", boundary=box(-95, 30, -94, 31))
    for item in (district, charter, empty):
        engine.add_district(item)

    inside = _campus(district, name="Inside", campus_number="001001001", lon=-97.5, lat=30.5)
    outside = _campus(district, name="Outside", campus_number="001001002", lon=-96.5, lat=30.5)
    missing = Campus(uuid.uuid4(), district.id, "Missing point", "", False, campus_number="001001003")
    charter_in = _campus(charter, name="Charter inside", campus_number="001801001", lon=-97.7, lat=30.5, is_charter=True)
    charter_edge = _campus(charter, name="Charter edge", campus_number="001801002", lon=-97, lat=30.5, is_charter=True)
    private = Campus(uuid.uuid4(), None, "Private edge", "", False, is_private=True,
                     source_namespace="tefa", source_id="K/Ké", location=(-97, 30.5))
    for campus in (inside, outside, missing, charter_in, charter_edge, private):
        engine.add_campus(campus)

    isd_query = engine >> ("district", "`001001-001")
    assert list(isd_query) == [district]
    assert list(engine >> ("district", "example%")) == [district, charter]
    assert list(engine >> ("district", 999999)) == []
    assert list(engine >> ("campus", "tefa:K%2FK%C3%A9")) == [private]
    assert list(engine >> ("campus", "tefa:absent")) == []
    assert list(engine >> ("campus", "'001801001")) == [charter_in]
    assert list(engine >> ("campus", 1001999)) == []
    assert list(engine >> ("campuses_in", isd_query)) == [inside, outside, missing]
    assert list(engine >> ("campuses_in", charter)) == [charter_in, charter_edge]
    assert list(engine >> ("campuses_in", empty)) == []
    assert list(engine >> ("campuses_in", Query([], engine))) == []
    assert list(engine >> ("within", district)) == [inside, charter_in]
    assert list(engine >> ("within", district, "campuses", True)) == [inside, charter_in, charter_edge, private]
    assert list(engine >> ("within", empty)) == []
    assert list(engine >> ("charters_within", district)) == [charter_in, charter_edge]
    assert list(engine >> ("privates_within", district)) == [private]
    assert list(engine >> ("nearest", inside)) == [inside]
    for pair in (array([-97.5, 30.5]), coordinates("d", [-97.5, 30.5]), Series([-97.5, 30.5])):
        assert list(engine >> ("nearest", pair)) == [inside]
    assert list(engine >> ("nearest", (-97, 30.5), 10, 0)) == [charter_edge, private]
    assert list(engine >> ("nearest_charter", private, 10, 0)) == [charter_edge]
    with pytest.raises(ValueError, match="Boundary geometry unavailable"):
        engine >> ("within", charter)
    with pytest.raises(ValueError, match="expects a District"):
        engine >> ("campuses_in", inside)
    with pytest.raises(ValueError, match="seed item with geometry"):
        engine >> ("nearest", missing)


@pytest.mark.parametrize("name,max_args", [
    ("district", 1), ("campus", 1), ("campuses_in", 1), ("within", 3),
    ("charters_within", 1), ("privates_within", 1), ("nearest", 3), ("nearest_charter", 3),
])
def test_engine_selector_arity_is_checked_before_dispatch(name, max_args):
    engine = DataEngine()
    for operands in ((), (None,) * (max_args + 1)):
        with pytest.raises(ValueError, match=f"{name} expects positional arguments"):
            engine >> (name, *operands)


@pytest.mark.parametrize("selector,message", [
    (([], None), "selector name must be a string"),
    (("unknown", None), "Unsupported engine selector"),
    (("district", True), "Invalid district number"),
    (("campus", False), "Invalid campus number"),
    (("nearest", (-97, 30), True), "nearest limit must be an integer"),
    (("nearest_charter", (-97, 30), 1, "10"), "nearest max_miles must be a number"),
    (("nearest", ("-97", 30)), "longitude must be a number"),
    (("nearest", None), "seed item with geometry"),
])
def test_engine_selectors_reject_invalid_operands(selector, message):
    with pytest.raises(ValueError, match=message):
        DataEngine() >> selector
