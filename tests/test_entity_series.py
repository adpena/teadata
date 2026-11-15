"""Entity namespaces keep the existing multi-year planner and export lineage."""

import csv
from io import BytesIO, StringIO
import pickle
from shutil import copyfile
from uuid import uuid4

import pytest

from teadata import Campus, DataEngine, District
from teadata.dataset_metadata import DataPeriod, EditionKey, NativeSourceRef
from teadata.manifest import ContractError
from teadata.query import describe_operators
from teadata.release import DataRelease
from teadata.source_views import DatasetFields, TPRSFields

from test_catalog_projections import attach_observation_child, make_edition_fixture, selected_catalog  # noqa: F401
from test_entity_tprs import FIELD, bound  # noqa: F401
from test_source_series import excel_export
from test_tprs_student_comparison import prepared_students  # noqa: F401


@pytest.fixture
def longitudinal(tmp_path, request):
    children = {}
    for year, count in ((2023, 9007199254740991), (2024, 9007199254740992), (2025, 9007199254740993)):
        directory = tmp_path / str(year)
        directory.mkdir()
        source = make_edition_fixture(directory, families=(f"year{year}",), dataset_name="campus_student", normalized=True,
            unique_entity_key=True, observed_amount=count,
            amount_period=DataPeriod("school_year", year - 1, year).to_dict())
        children[year] = DataRelease(source["root"], offline=True)
    # All three children are genuinely attached to one normal parent. Distinct
    # table filenames preserve adjacent immutable sources without mock routing.
    first = children[2023]
    refs = []
    for year, child in children.items():
        if year == 2023:
            continue
        for source in child.path.parent.iterdir():
            if source.is_file() and source != child.path:
                destination = first.path.parent / source.name
                if destination.exists():
                    assert destination.read_bytes() == source.read_bytes()
                else:
                    copyfile(source, destination)
        name = f"native-{year}.json"
        copyfile(child.path, first.path.parent / name)
        keys = tuple(EditionKey.from_dict({name: row[name] for name in EditionKey.__dataclass_fields__})
                     for row in child.catalog.editions(include_metadata=False))
        refs.append(NativeSourceRef(name, child.manifest_sha256, child.build_id, keys))
    release = attach_observation_child(request.getfixturevalue("selected_catalog"), first, additional_refs=refs)
    engine = DataEngine()
    engine.catalog = release.catalog
    district = District(uuid4(), "Synthetic district", district_number="000001")
    engine.add_district(district)
    campuses = []
    for identifier in range(1, 5):
        campus = Campus(uuid4(), district.id, "Synthetic campus", "", False,
                        campus_number=f"'{identifier:09d}")
        engine.add_campus(campus)
        campuses.append(campus)
    engine.bind_release(release)
    return engine, release, campuses


def test_entity_series_is_lazy_exact_and_keeps_full_precision_exports(longitudinal, monkeypatch):
    engine, release, campuses = longitudinal
    campus = campuses[3]
    before = pickle.dumps(engine)
    namespace = campus.campus_student
    with monkeypatch.context() as planning:
        planning.setattr(DataRelease, "artifact", lambda *a, **k: pytest.fail("Series planning opened a leaf"))
        series = namespace.series("AMOUNT", years=[25, "2022-23", 2024])
        expected = release.series("campus_student", "amount", scope="campus", years=[25, 23, 24], ids=[4])
        assert series.provenance() == expected.provenance()
        assert series.years == (2025, 2023, 2024)
        assert series.ids == ("000000004",)
        assert namespace.school_year is None
    counts = [9007199254740993, 9007199254740991, 9007199254740992]
    assert series.to_arrow()["amount"].to_pylist() == counts
    assert series.to_df()["amount"].tolist() == counts
    stream = BytesIO()
    series.to_csv(stream)
    rows = list(csv.DictReader(StringIO(stream.getvalue().decode())))
    assert [row["amount"] for row in rows] == list(map(str, counts))
    data = excel_export(series, BytesIO())
    index = data[0].index("amount")
    assert [row[index] for row in data[1:]] == list(map(str, counts))
    assert pickle.dumps(engine) == before
    assert all(campus.meta == {} for campus in campuses)


@pytest.mark.parametrize("index,raw,status", [(0, "0", "observed"), (1, "-999", "suppressed"), (2, ".", "missing")])
def test_entity_series_keeps_zero_suppression_and_missing_distinct(longitudinal, index, raw, status):
    _, _, campuses = longitudinal
    rows = campuses[index].campus_student.series("amount", years=[23, 24, 25]).to_arrow().to_pylist()
    assert len(rows) == 3
    assert {row["VALUE"] for row in rows} == {raw}
    assert {row["amount_status"] for row in rows} == {status}
    assert {row["amount"] for row in rows} == ({0} if index == 0 else {None})


@pytest.mark.parametrize("years", [None, [], "25", 25, [False], [25, "2024-25"]])
def test_entity_series_bad_years_are_actionable_without_leaf_reads(longitudinal, monkeypatch, years):
    _, _, campuses = longitudinal
    monkeypatch.setattr(DataRelease, "artifact", lambda *a, **k: pytest.fail("Bad years opened a leaf"))
    with pytest.raises(ValueError, match="years|school_year"):
        campuses[3].campus_student.series("amount", years=years)


def test_entity_series_refuses_unavailable_year_names_and_single_year_pin(longitudinal, monkeypatch):
    _, _, campuses = longitudinal
    namespace = campuses[3].campus_student
    monkeypatch.setattr(DataRelease, "artifact", lambda *a, **k: pytest.fail("Invalid selection opened a leaf"))
    with pytest.raises(ContractError, match="Available years"):
        namespace.series("amount", years=[22, 25])
    with pytest.raises(KeyError, match="Did you mean.*amount"):
        namespace.series("amont", years=[25])
    pinned = namespace.at(25)
    with pytest.raises(ValueError, match=r"pinned.*2025.*\.at\(None\)"):
        pinned.series("amount", years=[25])
    assert pinned.school_year == 2025 and namespace.school_year is None
    assert pinned.at(None).series("amount", years=[25]).years == (2025,)


def test_tprs_and_generic_facades_share_one_bound_series_planner(request, monkeypatch):
    engine, release, _, district, campus = request.getfixturevalue("bound")
    before = pickle.dumps(engine)
    monkeypatch.setattr("teadata.release.current_release", lambda: pytest.fail("Followed a later global release"))
    with monkeypatch.context() as planning:
        planning.setattr(DataRelease, "artifact", lambda *a, **k: pytest.fail("TPRS series planning opened a leaf"))
        for entity in (district, campus):
            tprs = entity.tprs.series(FIELD.upper(), years=["2025–26"])
            generic = entity.dataset("TPRS_STUDENT_MEMBERSHIP_ENROLLMENT_REPORTS").series(FIELD, years=[26])
            assert tprs.provenance() == generic.provenance()
    assert district.tprs.series(FIELD, years=[26]).to_arrow()[FIELD].to_pylist() == [0]
    assert pickle.dumps(engine) == before
    assert "series" in describe_operators()["entity_access"]["methods"]
    assert "unpinned" in describe_operators()["entity_access"]["year_policy"]
    assert DatasetFields.series is TPRSFields.series
