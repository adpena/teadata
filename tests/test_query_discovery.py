import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import get_type_hints
from uuid import uuid4

import pytest

from teadata.entities import Campus, District
from teadata.engine import DataEngine
from teadata.query import Query, describe_operators


def test_grade_overlap_reports_an_incomplete_custom_span_before_filtering():
    bounds = SimpleNamespace(grade_range_code_spans=[(None, 4)])
    with pytest.raises(ValueError, match="requires both grade bounds"):
        Query([], DataEngine()) >> ("grade_overlap", bounds)


def test_discovery_describes_query_operators_without_sharing_mutable_state():
    description = describe_operators()
    assert description["schema_version"] == 1 and description["receiver"] == "Query"
    expected = json.dumps(description, allow_nan=False)
    operators = {operator["name"]: operator for operator in description["operators"]}
    assert len(operators) == len(description["operators"]) == 20
    assert set(operators) == {
        "campuses_in", "compare_statewide", "distinct", "filter", "grade_overlap",
        "knn", "map", "nearest", "nearest_charter", "nearest_charter_same_type",
        "nearest_charter_transfer_destination", "observations", "privates_within",
        "radius", "sort", "take", "transfers_in", "transfers_out", "where_to", "within",
    }
    aliases = {alias: operator["name"] for operator in operators.values() for alias in operator["aliases"]}
    assert aliases == {
        "where": "filter", "select": "map", "grade_overlaps": "grade_overlap",
        "grade_range_overlap": "grade_overlap",
    }
    assert sum(len(operator["aliases"]) for operator in operators.values()) == len(aliases)
    predicate = operators["filter"]
    assert [form["syntax"] for form in predicate["forms"]] == ["tuple", "callable"]
    assert predicate["forms"][1]["arguments"][0]["types"][0]["python_only"]
    predicate["aliases"].clear()
    predicate["forms"][0]["arguments"][0]["types"].clear()
    assert json.dumps(describe_operators(), allow_nan=False) == expected

    values = Query([1, 2], DataEngine())
    assert list(values >> (lambda number: number > 1)) == [2]
    assert list(values >> ("where", lambda number: number > 1) >> ("select", str)) == ["2"]
    assert list(values >> ("sort", None, True)) == [2, 1]
    campus = SimpleNamespace(grade_range="7–8")
    selection = Query([campus], DataEngine())
    for alias in ("grade_overlaps", "grade_range_overlap"):
        assert list(selection >> (alias, 7, 8)) == [campus]


def test_series_discovery_matches_the_release_method_signature():
    from inspect import Parameter, signature

    from teadata.release import DataRelease
    from teadata.source_series import ObservationSeries

    method, = describe_operators()["release_methods"]
    assert method["name"] == "series" and method["receiver"] == "DataRelease"
    assert method["result"] == "ObservationSeries"
    assert {"to_arrow", "to_df", "provenance", "iter_batches"} <= set(method["result_methods"])
    operations = {operation["name"]: operation for operation in method["result_operations"]}
    assert operations.keys() == {"pair_years", "to_sql"}
    for name, operation in operations.items():
        assert name in method["result_methods"]
        assert [arg["name"] for arg in operation["arguments"]] == list(
            signature(getattr(ObservationSeries, name)).parameters)[1:]
    assert operations["pair_years"]["result"] == "ObservationPair"
    assert operations["to_sql"]["result"] == "SeriesSqlExport"
    arguments = method["arguments"]
    parameters = tuple(signature(DataRelease.series).parameters.values())[1:]
    assert [item["name"] for item in arguments] == [parameter.name for parameter in parameters]
    kinds = {Parameter.POSITIONAL_OR_KEYWORD: "positional", Parameter.VAR_POSITIONAL: "varargs",
             Parameter.KEYWORD_ONLY: "keyword_only"}
    for item, parameter in zip(arguments, parameters, strict=True):
        assert item["kind"] == kinds[parameter.kind]
        assert item["optional"] is (parameter.default is not Parameter.empty)
        if item["optional"]:
            assert item["default"] == parameter.default



def test_discovery_fresh_import_does_not_load_release_or_read_data():
    script = r'''
import importlib.abc, json, pathlib, socket, sqlite3, sys, urllib.request

class Trap(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"teadata.release", "teadata.engine", "sqlalchemy"}:
            raise AssertionError("Unexpected import: " + fullname)

def forbidden(*args, **kwargs):
    raise AssertionError("Discovery attempted I/O")

sys.meta_path.insert(0, Trap())
socket.create_connection = sqlite3.connect = urllib.request.urlopen = forbidden
from teadata.query import describe_operators
pathlib.Path.open = forbidden
description = json.loads(json.dumps(describe_operators(), allow_nan=False))
assert description["receiver"] == "Query"
assert description["release_methods"][0]["name"] == "series"
'''
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_dispatch_reads_the_tuple_operator_once():
    class ReadOnce(tuple):
        def __getitem__(self, index):
            if index == 0:
                assert not getattr(self, "read", False)
                self.read = True
            return super().__getitem__(index)

    assert list(Query([1, 2], DataEngine()) >> ReadOnce(("take", 1))) == [1]


def test_discovery_identifies_positional_forms_and_terminal_results(monkeypatch):
    operators = {operator["name"]: operator for operator in describe_operators()["operators"]}
    observations = operators["observations"]
    assert observations["result"] == {"type": "SourceTable", "alternate_type": "PartitionedSourceTable"}
    _, year, fields, scope, edition = observations["forms"][0]["arguments"]
    assert year["nullable"] and not year["optional"] and "default" not in year
    assert "2024-25" in year["description"] and "2025 or 25" in year["description"]
    assert year["types"] == [{"type": "integer", "python_only": False},
                             {"type": "string", "python_only": False}]
    assert fields["types"][0]["type"] == "nonempty list or tuple of distinct nonempty strings"
    assert scope["optional"] and scope["default"] is None and scope["choices"] == ["district", "campus"]
    assert edition["types"] == [{"type": "EditionKey", "python_only": True}]
    comparison = operators["compare_statewide"]
    assert comparison["result"] == {"type": "Query", "item_type": "StatewideComparison"}
    assert [argument["types"][0] for argument in comparison["forms"][0]["arguments"][:3]] == [
        {"type": "DataRelease", "python_only": True},
        {"type": "EditionKey", "python_only": True},
        {"type": "EditionKey", "python_only": True},
    ]
    comparison_year = comparison["forms"][0]["arguments"][-1]
    assert comparison_year["name"] == "school_year" and comparison_year["types"] == [
        {"type": "integer", "python_only": False}, {"type": "string", "python_only": False}]
    assert comparison_year["optional"] and comparison_year["nullable"] and comparison_year["default"] is None
    nearest = operators["nearest_charter"]["forms"][0]["arguments"]
    assert [argument["name"] for argument in nearest] == ["seed", "limit", "max_miles"]
    assert [item["python_only"] for item in nearest[0]["types"]] == [False, True, True]
    for name in ("radius", "knn"):
        seed = operators[name]["forms"][0]["arguments"][0]
        assert seed["nullable"] and not seed["optional"] and "coordinate pair" in seed["description"]
    sort_key = operators["sort"]["forms"][0]["arguments"][0]
    assert sort_key["nullable"] and not sort_key["optional"]
    transfer_forms = operators["transfers_out"]["forms"]
    assert transfer_forms[0]["arguments"][0]["default"] is False
    assert transfer_forms[1]["arguments"][0]["types"][0]["python_only"]
    assert operators["privates_within"]["forms"] == [{"syntax": "tuple", "arguments": []}]
    assert operators["within"]["availability"] == "built_in"

    engine = DataEngine()
    witness = object()

    def nearest_campuses(lon, lat, *, limit, charter_only, max_miles, geodesic):
        assert (lon, lat, limit, charter_only, max_miles, geodesic) == (-95.0, 30.0, 3, True, 20.5, True)
        return [witness]

    monkeypatch.setattr(engine, "nearest_campuses", nearest_campuses)
    arguments = {"seed": (-95.0, 30.0), "limit": 3, "max_miles": 20.5}
    command = ("nearest_charter", *(arguments[argument["name"]] for argument in nearest))
    assert list(Query([], engine) >> command) == [witness]


def test_discovered_containment_forms_match_stock_engine_support():
    geometry = pytest.importorskip("shapely.geometry")
    engine = DataEngine()
    boundary = geometry.box(-96, 29, -94, 31)
    district = District(uuid4(), "District", district_number="012345", boundary=boundary)
    private = Campus(
        uuid4(), None, "Private", "Private", False, is_private=True,
        source_namespace="tefa", source_id="example", location=(-95, 30),
    )
    engine.add_district(district)
    engine.add_campus(private)

    selection = engine >> ("district", "012345")
    assert list(selection >> ("privates_within",)) == [private]
    assert list(selection >> ("within", None, district)) == [private]
    assert list(engine >> ("within", district, "campuses")) == [private]
    assert get_type_hints(DataEngine.campuses_within)["return"] == list[Campus]
    with pytest.raises(ValueError, match="Boolean"):
        selection >> ("within", None, district, "false")
    with pytest.raises(ValueError, match="within expects positional arguments"):
        selection >> ("within", None, district, False, True, "extra")


def test_discovered_outbound_boolean_filters_to_charters(monkeypatch):
    engine = DataEngine()
    origin = Campus(uuid4(), None, "Origin", "Public", False, campus_number="012345001")
    charter = Campus(uuid4(), None, "Charter", "Charter", True, campus_number="012346001")
    public = Campus(uuid4(), None, "Public", "Public", False, campus_number="012347001")
    monkeypatch.setattr(engine, "transfers_out", lambda campus: [(charter, None, True), (public, 12, False)])

    selection = Query([origin], engine)
    assert list(selection >> ("transfers_out", True)) == [(origin, charter, None, True)]
    assert list(selection >> ("transfers_out", False)) == [
        (origin, charter, None, True), (origin, public, 12, False),
    ]
