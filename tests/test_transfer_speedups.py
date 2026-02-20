import uuid

import pandas as pd

from teadata.classes import Campus, DataEngine, District
from teadata.load_data import _group_transfer_rows


def _make_district(name: str, district_number: str) -> District:
    return District(
        id=uuid.uuid4(),
        name=name,
        district_number=district_number,
        enrollment=1000,
        rating="A",
    )


def _make_campus(
    district: District,
    *,
    name: str,
    campus_number: str,
    lon: float,
    lat: float,
) -> Campus:
    return Campus(
        id=uuid.uuid4(),
        district_id=district.id,
        name=name,
        charter_type="Public",
        is_charter=False,
        district_number=district.district_number,
        campus_number=campus_number,
        location=(lon, lat),
    )


def test_group_transfer_rows_matches_pandas_groupby():
    df = pd.DataFrame(
        {
            "campus_campus_number": [
                "'011901001",
                "'011901001",
                "'011901001",
                "'011901002",
            ],
            "to_campus_number": [
                "'011901101",
                "'011901101",
                "'011901102",
                "'011901101",
            ],
            "count": [10, 5, 1, 3],
            "masked": [False, True, False, False],
        }
    )

    expected = (
        df.groupby(["campus_campus_number", "to_campus_number"], dropna=False)
        .agg({"count": "sum", "masked": "max"})
        .reset_index()
        .sort_values(["campus_campus_number", "to_campus_number"], kind="stable")
        .reset_index(drop=True)
    )
    actual = (
        _group_transfer_rows(df)
        .sort_values(["campus_campus_number", "to_campus_number"], kind="stable")
        .reset_index(drop=True)
    )

    assert actual.to_dict(orient="records") == expected.to_dict(orient="records")


def test_apply_transfers_from_dataframe_preserves_masked_and_missing_counts():
    repo = DataEngine()
    district = _make_district("Alpha ISD", "11901")
    repo.add_district(district)
    src = _make_campus(
        district,
        name="Source Campus",
        campus_number="11901001",
        lon=-97.0,
        lat=30.0,
    )
    dst = _make_campus(
        district,
        name="Destination Campus",
        campus_number="11901002",
        lon=-97.1,
        lat=30.1,
    )
    repo.add_campus(src)
    repo.add_campus(dst)

    df = pd.DataFrame(
        [
            {
                "REPORT_TYPE": "Transfers Out To",
                "REPORT_CAMPUS": 11901001,
                "CAMPUS_RES_OR_ATTEND": 11901002,
                "TRANSFERS_IN_OR_OUT": 10,
            },
            {
                "REPORT_TYPE": "Transfers Out To",
                "REPORT_CAMPUS": 11901001,
                "CAMPUS_RES_OR_ATTEND": 11901002,
                "TRANSFERS_IN_OR_OUT": -999,
            },
            {
                "REPORT_TYPE": "Transfers Out To",
                "REPORT_CAMPUS": 11901001,
                "CAMPUS_RES_OR_ATTEND": 11901002,
                "TRANSFERS_IN_OR_OUT": "not-a-number",
            },
            {
                "REPORT_TYPE": "Transfers Out To",
                "REPORT_CAMPUS": 11901001,
                "CAMPUS_RES_OR_ATTEND": 999999999,
                "TRANSFERS_IN_OR_OUT": 7,
            },
            {
                "REPORT_TYPE": "Transfers Out To",
                "REPORT_CAMPUS": 999999998,
                "CAMPUS_RES_OR_ATTEND": 11901002,
                "TRANSFERS_IN_OR_OUT": 3,
            },
            {
                "REPORT_TYPE": "Transfers Out To",
                "REPORT_CAMPUS": 999999998,
                "CAMPUS_RES_OR_ATTEND": 999999999,
                "TRANSFERS_IN_OR_OUT": 2,
            },
            {
                "REPORT_TYPE": "Transfers Out To",
                "REPORT_CAMPUS": "invalid",
                "CAMPUS_RES_OR_ATTEND": 11901002,
                "TRANSFERS_IN_OR_OUT": 4,
            },
            {
                "REPORT_TYPE": "Transfers In From",
                "REPORT_CAMPUS": 11901001,
                "CAMPUS_RES_OR_ATTEND": 11901002,
                "TRANSFERS_IN_OR_OUT": 50,
            },
        ]
    )

    updated = repo.apply_transfers_from_dataframe(df)

    assert updated == 1
    assert repo._xfers_missing == {"src": 1, "dst": 1, "either": 1}

    out_edges = repo.transfers_out(src)
    assert len(out_edges) == 3
    assert [count for _, count, _ in out_edges] == [10, None, None]
    assert [masked for _, _, masked in out_edges] == [False, True, False]

    in_edges = repo.transfers_in(dst)
    assert len(in_edges) == 3
    assert [count for _, count, _ in in_edges] == [10, None, None]
    assert [masked for _, _, masked in in_edges] == [False, True, False]
