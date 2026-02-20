import uuid

from teadata.classes import Campus, DataEngine, District


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


def test_district_lookup_by_number_and_name_returns_all_matches():
    repo = DataEngine()
    d1 = _district("Alpha ISD", "123")
    d2 = _district("Alpha ISD", "'000123")
    d3 = _district("Beta ISD", "456")
    repo.add_district(d1)
    repo.add_district(d2)
    repo.add_district(d3)

    number_hits = repo >> ("district", 123)
    assert [d.id for d in number_hits] == [d1.id, d2.id]

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
