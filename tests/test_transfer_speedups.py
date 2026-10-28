from teadata.teadata_config import load_config
import uuid

import pandas as pd
import pytest

from teadata.classes import Campus, DataEngine, District
from teadata.load_data import _group_transfer_rows
from teadata.manifest import ContractError


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


def test_transfer_projection_preserves_missing_zero_and_suppression():
    df = pd.DataFrame(
        {
            "campus_campus_number": [
                "'011901001",
                "'011901001",
                "'011901001",
                "'011901001",
            ],
            "to_campus_number": [
                "'011901101",
                "'011901102",
                "'011901103",
                "'011901104",
            ],
            "count": [10, None, 0, None],
            "masked": [False, True, False, False],
        }
    )

    actual = _group_transfer_rows(df)
    assert actual["count"].tolist() == [10, None, 0, None]
    assert actual["status"].tolist() == [
        "observed",
        "suppressed",
        "observed",
        "missing",
    ]


def test_transfer_projection_rejects_duplicate_pair_grain():
    frame = pd.DataFrame(
        {
            "campus_campus_number": ["123456001"] * 2,
            "to_campus_number": ["123456002"] * 2,
            "count": [12, 12],
            "masked": [False, False],
        }
    )
    with pytest.raises(ContractError, match="Duplicate transfer pair"):
        _group_transfer_rows(frame)


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
    suppressed_destination = _make_campus(
        district,
        name="Suppressed destination",
        campus_number="11901003",
        lon=-97.2,
        lat=30.2,
    )
    missing_destination = _make_campus(
        district,
        name="Missing destination",
        campus_number="11901004",
        lon=-97.3,
        lat=30.3,
    )
    repo.add_campus(suppressed_destination)
    repo.add_campus(missing_destination)

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
                "CAMPUS_RES_OR_ATTEND": 11901003,
                "TRANSFERS_IN_OR_OUT": -999,
            },
            {
                "REPORT_TYPE": "Transfers Out To",
                "REPORT_CAMPUS": 11901001,
                "CAMPUS_RES_OR_ATTEND": 11901004,
                "TRANSFERS_IN_OR_OUT": None,
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
                "REPORT_TYPE": "Transfers In From",
                "REPORT_CAMPUS": 11901001,
                "CAMPUS_RES_OR_ATTEND": 11901002,
                "TRANSFERS_IN_OR_OUT": 50,
            },
        ]
    )

    updated = repo.apply_transfers_from_dataframe(
        df,
        value_policy=load_config("teadata/teadata_sources.yaml").value_policies[
            "campus_transfers"
        ],
    )

    assert updated == 1
    assert repo._xfers_missing == {"src": 1, "dst": 1, "either": 1}

    out_edges = repo.transfers_out(src)
    assert len(out_edges) == 3
    assert {
        campus.campus_number: (count, masked) for campus, count, masked in out_edges
    } == {
        "'011901002": (10, False),
        "'011901003": (None, True),
        "'011901004": (None, False),
    }

    in_edges = repo.transfers_in(dst)
    assert len(in_edges) == 1
    assert [count for _, count, _ in in_edges] == [10]
    assert [masked for _, _, masked in in_edges] == [False]
