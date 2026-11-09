"""Literal public Query results for each advertised form and alias."""

from typing import NamedTuple
from uuid import UUID

import pandas as pd
import pytest
from shapely.geometry import box

from teadata import Campus, DataEngine, District
from teadata.query import Query, describe_operators
from teadata.values import ValuePolicy



@pytest.fixture
def cohort():
    engine = DataEngine()
    district = District(UUID(int=1), "Golden ISD", district_number="001001",
                        boundary=box(-98, 30, -97, 31))
    engine.add_district(district)
    origin = (-97.6, 30.5)
    public_point = (-97.2, 30.5)
    for number, name, location, grades in (
        (1, "ORIGIN", origin, "PK–2"),
        (2, "CHARTER", origin, "3–5"),
        (3, "PUBLIC", public_point, "6–8"),
        (4, "NO POINT", None, None),
    ):
        engine.add_campus(Campus(UUID(int=number + 1), district.id, name, "", number == 2,
                                 campus_number=f"00100100{number}", location=location,
                                 grade_range=grades, school_type="Elementary"))
    engine.add_campus(Campus(UUID(int=6), None, "BOUNDARY", "", False, is_private=True,
                             source_namespace="tefa", source_id="boundary", location=(-97, 30.5)))
    engine.apply_transfers_from_dataframe(pd.DataFrame({
        "REPORT_CAMPUS": [1001001, "001001001"],
        "CAMPUS_RES_OR_ATTEND": ["'001001002", 1001003],
        "TRANSFERS_IN_OR_OUT": ["-999", "0"],
    }), value_policy=ValuePolicy.from_dict({
        "policy_id": "test_mask",
        "tokens": {"-999": {"status": "suppressed", "reason": "fixture"}},
    }))
    return engine


def names(result):
    return [item.name for item in result]


def is_origin(campus):
    return campus.name == "ORIGIN"


def transfer_rows(result):
    return [(start.name, end.name, count, masked) for start, end, count, masked in result]


def match_rows(result):
    return [(seed.name, match.name if match else None, miles) for seed, match, miles in result]


# All values below are chosen before dispatch. In particular the same point is
# zero miles away, the outer edge is excluded by contains, and masks are not 0.
FORMS = {
    ("campuses_in", "campuses_in", "tuple", ()): (
        "district", ("campuses_in",), names, ["ORIGIN", "CHARTER", "PUBLIC", "NO POINT"]),
    ("filter", "filter", "tuple", ("predicate",)): (
        "all", ("filter", is_origin), names, ["ORIGIN"]),
    ("filter", "filter", "callable", ("predicate",)): (
        "all", is_origin, names, ["ORIGIN"]),
    ("filter", "where", "tuple", ("predicate",)): (
        "all", ("where", is_origin), names, ["ORIGIN"]),
    ("take", "take", "tuple", ("n",)): (
        "all", ("take", -1), names, ["ORIGIN", "CHARTER", "PUBLIC"]),
    ("sort", "sort", "tuple", ("key", "descending")): (
        "all", ("sort", lambda c: c.name == "PUBLIC", True), names, ["PUBLIC", "ORIGIN", "CHARTER", "NO POINT"]),
    ("map", "map", "tuple", ("function",)): (
        "all", ("map", lambda c: c.name[0]), list, ["O", "C", "P", "N"]),
    ("map", "select", "tuple", ("function",)): (
        "all", ("select", lambda c: c.name[-1]), list, ["N", "R", "C", "T"]),
    ("distinct", "distinct", "tuple", ("key",)): (
        "all", ("distinct", lambda c: c.is_charter), names, ["ORIGIN", "CHARTER"]),
    ("grade_overlap", "grade_overlap", "tuple", ("grades",)): (
        "all", ("grade_overlap", "2–3"), names, ["ORIGIN", "CHARTER"]),
    ("grade_overlap", "grade_overlap", "tuple", ("low", "high")): (
        "all", ("grade_overlap", 2, 3), names, ["ORIGIN", "CHARTER"]),
    ("grade_overlap", "grade_overlaps", "tuple", ("grades",)): (
        "all", ("grade_overlaps", "3–3"), names, ["CHARTER"]),
    ("grade_overlap", "grade_overlaps", "tuple", ("low", "high")): (
        "all", ("grade_overlaps", 3, 3), names, ["CHARTER"]),
    ("grade_overlap", "grade_range_overlap", "tuple", ("grades",)): (
        "all", ("grade_range_overlap", "5–6"), names, ["CHARTER", "PUBLIC"]),
    ("grade_overlap", "grade_range_overlap", "tuple", ("low", "high")): (
        "all", ("grade_range_overlap", 5, 6), names, ["CHARTER", "PUBLIC"]),
    ("nearest", "nearest", "tuple", ("seed", "limit", "max_miles")): (
        "public", ("nearest",), names, ["PUBLIC"]),
    ("nearest_charter", "nearest_charter", "tuple", ("seed", "limit", "max_miles")): (
        "origin", ("nearest_charter", None, 1, 0), names, ["CHARTER"]),
    ("radius", "radius", "tuple", ("seed", "miles", "limit", "charter_only")): (
        "public", ("radius", None, 100, 1, False), names, ["PUBLIC"]),
    ("knn", "knn", "tuple", ("seed", "k", "charter_only")): (
        "public", ("knn", None, 1, True), names, ["CHARTER"]),
    ("within", "within", "tuple", ("seed", "target", "charter_only", "covers")): (
        "district", ("within", None, None, False, True), lambda result: sorted(names(result)),
        ["BOUNDARY", "CHARTER", "ORIGIN", "PUBLIC"]),
    ("privates_within", "privates_within", "tuple", ()): (
        "district", ("privates_within",), names, ["BOUNDARY"]),
    ("nearest_charter_same_type", "nearest_charter_same_type", "tuple", ("k",)): (
        "origin", ("nearest_charter_same_type",), match_rows, [("ORIGIN", "CHARTER", 0.0)]),
    ("nearest_charter_transfer_destination", "nearest_charter_transfer_destination", "tuple", ()): (
        "all", ("nearest_charter_transfer_destination",), match_rows,
        [("ORIGIN", "CHARTER", 0.0), ("CHARTER", None, None), ("PUBLIC", None, None), ("NO POINT", None, None)]),
    ("transfers_out", "transfers_out", "tuple", ("charter_only",)): (
        "origin", ("transfers_out", True), transfer_rows, [("ORIGIN", "CHARTER", None, True)]),
    ("transfers_out", "transfers_out", "tuple", ("predicate",)): (
        "origin", ("transfers_out", lambda c: c.name == "PUBLIC"), transfer_rows, [("ORIGIN", "PUBLIC", 0, False)]),
    ("transfers_in", "transfers_in", "tuple", ()): (
        "charter", ("transfers_in",), transfer_rows, [("ORIGIN", "CHARTER", None, True)]),
    ("where_to", "where_to", "tuple", ("predicate",)): (
        "outbound", ("where_to", lambda c: c.name == "PUBLIC"), transfer_rows, [("ORIGIN", "PUBLIC", 0, False)]),
}

OBSERVATIONS = ("observations", "observations", "tuple", ("dataset", "school_year", "fields", "scope", "edition"))
STATEWIDE = ("compare_statewide", "compare_statewide", "tuple", (
    "release", "local", "statewide", "metric", "dimensions", "school_year"))


def test_every_discovered_form_has_a_literal_golden():
    advertised = {
        (operator["name"], command, form["syntax"], tuple(arg["name"] for arg in form["arguments"]))
        for operator in describe_operators()["operators"]
        for form in operator["forms"]
        for command in (operator["name"], *operator["aliases"])
        if form["syntax"] == "tuple" or command == operator["name"]
    }
    assert advertised == {*FORMS, OBSERVATIONS, STATEWIDE}


@pytest.mark.parametrize("seed,command,project,expected", [
    pytest.param(*case, id=f"{spelling}-{syntax}-{','.join(arguments)}")
    for (_, spelling, syntax, arguments), case in FORMS.items()
])
def test_entity_query_golden(cohort, seed, command, project, expected):
    if seed in {"origin", "outbound", "public", "charter"}:
        identifiers = {"origin": 1001001, "outbound": 1001001, "public": 1001003, "charter": 1001002}
        query = cohort >> ("campus", identifiers[seed])
        if seed == "outbound":
            query = query >> ("transfers_out",)
    else:
        query = cohort >> ("district", "Golden ISD")
        if seed == "all":
            query = query >> ("campuses_in",)
    before = list(query)
    assert project(query >> command) == expected
    assert list(query) == before


def test_explicit_spatial_seed_cannot_fall_back_to_another_campus(cohort):
    origin = cohort >> ("campus", 1001001)
    no_point = cohort >> ("campus", 1001004)
    for invalid_seed in ("not a point", no_point, Query([], cohort)):
        with pytest.raises(ValueError, match="seed item with geometry"):
            origin >> ("nearest", invalid_seed)


def test_point_bearing_seed_takes_precedence_over_tuple_payload(cohort):
    class Site(NamedTuple):
        name: str
        point: tuple[float, float]

    origin = cohort >> ("campus", 1001001)
    assert names(origin >> ("nearest", Site("Public site", (-97.2, 30.5)))) == ["PUBLIC"]


def test_zero_radius_includes_the_point_and_honors_charter_flag(cohort):
    origin = cohort >> ("campus", 1001001)
    assert names(origin >> ("radius", None, 0, None, True)) == ["CHARTER"]


@pytest.mark.parametrize("command,message", [
    (("take", 1, "ignored"), "take expects positional arguments"),
    (("take", True), "take count must be an integer"),
    (("take", 1.5), "take count must be an integer"),
    (("sort", None, "false"), "sort descending must be a Boolean"),
    (("sort", None, 1), "sort descending must be a Boolean"),
    (("select", False), "map function must be callable"),
    (("where",), "where expects positional arguments"),
    (("where_to", False), "where_to predicate must be callable"),
    (("transfers_out", None), "transfers_out predicate"),
    (("transfers_out", "false"), "transfers_out predicate"),
    (("radius", (-97, 30), 5, None, "false"), "radius charter_only must be a Boolean"),
    (("knn", (-97, 30), 1, "false"), "knn charter_only must be a Boolean"),
    (("radius", (-97, 30), True), "radius miles must be a number"),
    (("nearest", (-97, 30), 1, "5"), "nearest max_miles must be a number"),
    (("knn", ("-97", 30), 1), "longitude must be a number"),
    (([],), "Query operator name must be a string"),
])
def test_malformed_queries_fail_even_on_empty_inputs(command, message):
    with pytest.raises(ValueError, match=message):
        Query([], DataEngine()) >> command
