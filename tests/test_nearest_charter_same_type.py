import uuid

import pytest

from teadata.classes import Campus, DataEngine


def _campus(
    *,
    name: str,
    point: tuple[float, float],
    school_type: str,
    is_charter: bool,
    is_private: bool = False,
) -> Campus:
    campus = Campus(
        id=uuid.uuid4(),
        district_id=uuid.uuid4(),
        name=name,
        charter_type="Charter" if is_charter else "None",
        is_charter=is_charter,
        is_private=is_private,
        school_type=school_type,
    )
    campus.point = point
    return campus


@pytest.mark.parametrize("indexes_enabled", [False, True])
def test_nearest_charter_same_type_skips_self_and_private_sources(indexes_enabled):
    repo = DataEngine(indexes_enabled=indexes_enabled)

    source_charter = _campus(
        name="Source Charter",
        point=(-97.0, 32.0),
        school_type="Elementary",
        is_charter=True,
    )
    peer_charter = _campus(
        name="Peer Charter",
        point=(-97.01, 32.0),
        school_type="Elementary",
        is_charter=True,
    )
    far_charter = _campus(
        name="Far Charter",
        point=(-97.5, 32.0),
        school_type="Elementary",
        is_charter=True,
    )
    regular_source = _campus(
        name="Regular Source",
        point=(-97.0005, 32.0),
        school_type="Elementary",
        is_charter=False,
    )
    private_source = _campus(
        name="Private Source",
        point=(-97.02, 32.0),
        school_type="Elementary",
        is_charter=False,
        is_private=True,
    )

    for campus in (
        source_charter,
        peer_charter,
        far_charter,
        regular_source,
        private_source,
    ):
        repo.add_campus(campus)

    result = repo.nearest_charter_same_type(
        [source_charter, regular_source, private_source]
    )

    source_match = result[str(source_charter.id)]
    assert source_match["match"] is not None
    assert source_match["match"].id == peer_charter.id
    assert source_match["miles"] is not None
    assert source_match["miles"] > 0

    regular_match = result[str(regular_source.id)]
    assert regular_match["match"] is not None
    assert regular_match["match"].id == source_charter.id
    assert regular_match["miles"] is not None
    assert regular_match["miles"] > 0

    assert result[str(private_source.id)] == {"match": None, "miles": None}


@pytest.mark.parametrize("indexes_enabled", [False, True])
def test_nearest_charter_same_type_returns_none_when_only_self_candidate(
    indexes_enabled,
):
    repo = DataEngine(indexes_enabled=indexes_enabled)
    source_charter = _campus(
        name="Only Charter",
        point=(-97.0, 32.0),
        school_type="Middle",
        is_charter=True,
    )
    repo.add_campus(source_charter)

    result = repo.nearest_charter_same_type([source_charter])

    assert result[str(source_charter.id)] == {"match": None, "miles": None}
