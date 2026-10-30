"""Small publisher-cell oracles, independent of generated releases/round trips."""

import json
from importlib.resources import files
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from teadata import Campus, DataEngine, District
from teadata.enrichment import campuses, districts
from teadata.transfer_summary import charter_transfer_observations


EXCERPTS = json.loads(
    files("teadata").joinpath("reference_cases/publisher_2025.json").read_text()
)


@pytest.mark.parametrize("module", [campuses, districts])
@pytest.mark.parametrize("index", [[1, 4, 9, 12], [1, 1, 4, 4]])
def test_text_cleanup_keeps_row_identity_and_nontext_types(module, index):
    frame = pd.DataFrame(
        {
            "rating": [" A ", " C ", " ", None],
            "count": pd.array([0, 2**60 + 1, None, 12], dtype="Int64"),
            "flag": pd.array([False, True, None, False], dtype="boolean"),
            "fraction": [-0.0, 0.0, 1.5, 2.0],
        },
        index=pd.Index(index, name="publisher_row"),
    )
    original = frame.drop(columns="rating").copy(deep=True)
    module.clean_value_columns(frame, list(frame.columns))
    assert frame.index.equals(original.index)
    assert frame["rating"].iloc[:2].tolist() == ["A", "C"]
    assert frame["rating"].iloc[2:].isna().all()
    assert_frame_equal(frame.drop(columns="rating"), original)
    assert frame["fraction"].iloc[0].hex() == "-0x0.0p+0"


def test_mixed_publisher_summary_rows_attach_ratings_to_the_right_campus(monkeypatch):
    rows = EXCERPTS["accountability_rows"]
    # Header is Excel row 1; pandas retains source offsets after district rows
    # are filtered out. Deliberate gaps expose label-vs-position assignment.
    frame = pd.DataFrame(rows, index=[row["excel_row"] - 2 for row in rows])
    monkeypatch.setattr(
        campuses,
        "load_config",
        lambda _: SimpleNamespace(load_df=lambda *args, **kwargs: (2025, frame.copy())),
    )
    entities = {
        row["Campus Number"]: SimpleNamespace(
            campus_number=row["Campus Number"], meta={}, rating=None
        )
        for row in rows
        if row["Campus Number"] is not None
    }
    resolved_year, updated = campuses._apply_campus_enrichment_table(
        SimpleNamespace(_campuses=entities),
        "unused-excerpt-config",
        "accountability",
        2025,
        rename={"2025 Overall Rating": "overall_rating"},
        select=["overall_rating"],
        aliases={"overall_rating": "rating"},
        reader_kwargs={"reader_profile": "ratings"},
    )
    assert (resolved_year, updated) == (2025, 3)
    assert {key: entity.rating for key, entity in entities.items()} == {
        "015834001": "A",  # Summary!G448
        "220820102": "Not Rated: Minimum Size rules",  # Summary!G8507
        "227901180": "A",  # Summary!G9440
    }
    assert all(entity.meta["overall_rating"] == entity.rating for entity in entities.values())


@pytest.mark.parametrize("case", EXCERPTS["cases"][:4], ids=lambda c: c["case_id"])
def test_publisher_values_statuses_units_and_reference_periods(catalog, case):
    scope = case["entity"]["type"]
    entity = SimpleNamespace(
        **{f"{scope}_number": case["entity"]["number"]},
        is_private=False,
        meta={fact["field"]: fact["source"]["raw_token"] for fact in case["facts"]},
    )
    for fact in case["facts"]:
        expected = fact["expected"]
        value = catalog.measurement(entity, expected["metric"], scope=scope)
        assert value.value == expected["value"], fact
        assert value.status.value == expected["status"], fact
        # CPET wildcard fields have no complete source-field unit dictionary
        # yet. Keep publisher units in the fixture as unresolved metadata truth,
        # rather than treating an unknown API unit as the desired answer.
        if expected["metric"] in EXCERPTS["release_validation"]["unit_metrics"]:
            assert value.unit == expected["unit"], fact
        assert value.raw == fact["source"]["raw_token"], fact
        assert value.source["period"] == expected["period"], fact
        if expected["status"] == "suppressed":
            assert value.numeric is None
            assert value.lower is None and value.upper is None


def test_kiker_midpoint_is_per_masked_destination_campus_not_charter_district(catalog):
    case = EXCERPTS["cases"][1]
    repo = DataEngine()
    repo.catalog = catalog
    for number, name in [("227901", "AUSTIN ISD"), ("015834", "BASIS TEXAS"), ("227829", "VALOR")]:
        repo.add_district(District(id=uuid5(NAMESPACE_URL, number), name=name, district_number=number))
    origin = Campus(
        id=uuid5(NAMESPACE_URL, "227901180"),
        district_id=uuid5(NAMESPACE_URL, "227901"),
        district_number="227901",
        campus_number="227901180",
        name="KIKER EL",
        charter_type="",
        is_charter=False,
    )
    repo.add_campus(origin)
    for flow in case["flows"]:
        repo.add_campus(Campus(
            id=uuid5(NAMESPACE_URL, flow["destination_number"]),
            district_id=uuid5(NAMESPACE_URL, flow["destination_governing_district"]),
            district_number=flow["destination_governing_district"],
            campus_number=flow["destination_number"],
            name=flow["destination_name"],
            charter_type=flow["charter_type_raw"],
            is_charter=True,
        ))
    repo.apply_transfers_from_dataframe(
        pd.DataFrame([
            {"REPORT_CAMPUS": "227901180", "CAMPUS_RES_OR_ATTEND": flow["destination_number"],
             "TRANSFERS_IN_OR_OUT": flow["raw_token"], "REPORT_TYPE": "Transfers Out To"}
            for flow in case["flows"]
        ]),
        value_policy=catalog.policy("campus_transfer_reports"),
    )
    summary = charter_transfer_observations(origin)
    assert summary.destination_count == 4
    assert summary.aggregate.observed_sum == 52
    assert summary.aggregate.suppressed_count == 3
    assert summary.aggregate.exact_total is None
    result = catalog.metadata("campus_transfer_reports").estimation.estimate(summary.aggregate)
    assert result["status"] == "estimated"
    assert result["observed_sum"] == 52
    assert result["estimated_contribution"] == 15
    assert result["value"] == 67  # 52 + 3 * 5, handwritten scenario arithmetic.


def test_ace_private_school_public_tapr_is_not_applicable(catalog):
    case = EXCERPTS["cases"][4]
    assert case["source_cells"]["Type"]["raw_token"] == "Private School"
    # This tests population applicability only. The released public-district
    # affiliation is a separate unresolved acceptance case, never golden truth.
    entity = SimpleNamespace(is_private=True, campus_number=None, meta={})
    for metric in ("enrollment", "overall_rating"):
        value = catalog.measurement(entity, metric)
        assert value.status.value == "not_applicable"
        assert value.reason == "outside_metric_population"
        assert value.numeric is None and value.value is None
