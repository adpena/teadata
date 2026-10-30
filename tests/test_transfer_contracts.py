from hashlib import sha256
import math
import pickle
import sqlite3
import uuid

import pandas as pd
import pytest

from teadata import Campus, DataEngine, District
from teadata.identifiers import canonical_source_identifier
from teadata.load_data import (
    build_closures_map_payload,
    build_campus_profile_map_payload,
)
from teadata.manifest import (
    ContractError,
    DatasetCatalog,
    canonical_json,
    write_sqlite_catalog,
)
from teadata.release import SNAPSHOT_MAGIC
from teadata.transfer_summary import charter_transfer_observations
from teadata.values import ValuePolicy, ValueStatus, measure


@pytest.fixture
def transfer_repo():
    repo = DataEngine()
    district = District(id=uuid.uuid4(), name="Example ISD", district_number="123456")
    repo.add_district(district)
    campuses = []
    for index, name, charter in [
        (1, "Origin", False),
        (2, "Destination", True),
        (3, "Other destination", True),
    ]:
        campus = Campus(
            id=uuid.uuid4(),
            district_id=district.id,
            district_number="123456",
            campus_number=f"123456{index:03}",
            name=name,
            charter_type="Charter" if charter else "",
            is_charter=charter,
            enrollment=None,
        )
        repo.add_campus(campus)
        campuses.append(campus)
    return repo, district, campuses


def frame(destination="123456002", count=12):
    return pd.DataFrame(
        {
            "REPORT_CAMPUS": ["123456001"],
            "CAMPUS_RES_OR_ATTEND": [destination],
            "TRANSFERS_IN_OR_OUT": [count],
        }
    )


@pytest.mark.parametrize("district_type", ["isd", "charter"])
def test_map_human_labels_with_colons_are_not_identity_inputs(transfer_repo, catalog, district_type):
    repo, district, campuses = transfer_repo
    repo.catalog = catalog
    campuses[0].name = "Example: East Campus"
    campuses[1].name = "Example: Charter Campus"
    repo.apply_transfers_from_dataframe(frame(), value_policy=catalog.policy("campus_transfer_reports"))
    payload = build_closures_map_payload(repo, district, district_type=district_type)
    assert payload["aisdCampuses"][0]["name"] == "Example: East Campus"
    assert payload["charterCampuses"][0]["name"] == "Example: Charter Campus"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("'001902001", "'001902001"),
        (1902001, "'001902001"),
        (1902001.0, "'001902001"),
        ("1902001.000", "'001902001"),
        (None, None),
        ("", None),
        (float("nan"), None),
        (pd.NA, None),
    ],
)
def test_source_identifiers_preserve_valid_codes_and_true_missing(value, expected):
    assert canonical_source_identifier(value, width=9) == expected


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        "invalid",
        "123456001.5",
        123456001.5,
        "123456001-extra",
        "1234560010",
        "000000000",
        0,
        -12,
        "nan",
        "None",
        float("inf"),
        [123456001],
        [float("nan")],
    ],
)
def test_source_identifiers_reject_malformed_nonblank_cells(value):
    with pytest.raises(ValueError):
        canonical_source_identifier(value, width=9)


@pytest.mark.parametrize(
    "destination",
    [None, "", "invalid", "123456002.5", "1234560020", True, 0, float("nan"), pd.NA],
)
def test_invalid_transfer_input_is_atomic_and_never_silently_dropped(
    transfer_repo, destination
):
    repo, _, campuses = transfer_repo
    repo.apply_transfers_from_dataframe(frame())
    before = list(repo.transfers_out(campuses[0]))
    invalid = pd.concat([frame(), frame(destination)], ignore_index=True)
    with pytest.raises(ContractError, match="campus identifier"):
        repo.apply_transfers_from_dataframe(invalid)
    assert repo.transfers_out(campuses[0]) == before


def test_duplicate_transfer_pairs_fail_without_overwriting_existing_facts(
    transfer_repo,
):
    repo, _, campuses = transfer_repo
    repo.apply_transfers_from_dataframe(frame(count=7))
    with pytest.raises(ContractError, match="Duplicate transfer pair"):
        repo.apply_transfers_from_dataframe(
            pd.concat([frame(), frame()], ignore_index=True)
        )
    assert repo.transfers_out(campuses[0])[0][1] == 7


def test_compatibility_properties_share_observation_summary_and_source_reason(
    transfer_repo,
):
    repo, _, campuses = transfer_repo
    policy = ValuePolicy.from_dict(
        {
            "policy_id": "source-policy",
            "tokens": {
                "-999": {
                    "status": "suppressed",
                    "reason": "source-specific suppression",
                }
            },
        }
    )
    repo.apply_transfers_from_dataframe(
        pd.concat([frame(count=12), frame("123456003", -999)], ignore_index=True),
        value_policy=policy,
    )
    origin = campuses[0]
    result = charter_transfer_observations(origin)
    assert result.destination_count == origin.num_charter_transfer_destinations == 2
    assert (
        result.aggregate.suppressed_count
        == origin.num_charter_transfer_destinations_masked
        == 1
    )
    assert (
        result.aggregate.observed_sum
        == origin.total_unmasked_charter_transfers_out
        == 12
    )
    value = repo._transfer_measurements[(origin.id, campuses[2].id)]
    assert value.reason == "source-specific suppression"
    assert value.raw == -999


@pytest.mark.parametrize(
    "raw,status", [(None, "missing"), (0, "observed"), (-999, "suppressed")]
)
def test_missing_enrollment_and_location_do_not_exclude_transfer_facts(
    transfer_repo, catalog, raw, status
):
    repo, district, campuses = transfer_repo
    repo.catalog = catalog
    policy = catalog.policy("campus_transfer_reports")
    repo.apply_transfers_from_dataframe(frame(count=raw), value_policy=policy)
    payload = build_closures_map_payload(repo, district, district_type="isd")
    edge = payload["aisdCampuses"][0]["transfersOut"][0]
    assert edge["count"] == (0 if raw == 0 else None)
    assert edge["masked"] == (status == "suppressed")
    assert edge["status"] == status
    assert edge["measurement"]["raw"] == raw
    if status == "suppressed":
        assert edge["measurement"]["reason"] == policy.tokens["-999"].reason
    assert payload["charterCampuses"][0]["lat"] is None
    campus_payload = build_campus_profile_map_payload(repo, campuses[0])
    assert campus_payload["charterDestinations"][0]["measurement"]["status"] == status


def test_transfer_query_failures_do_not_publish_empty_successful_maps(
    transfer_repo, catalog, monkeypatch
):
    repo, district, _ = transfer_repo
    repo.catalog = catalog

    def failed(*args):
        raise RuntimeError("source transfer query failed")

    monkeypatch.setattr(repo, "transfers_out", failed)
    with pytest.raises(RuntimeError, match="source transfer query failed"):
        build_closures_map_payload(repo, district, district_type="isd")


def test_boundary_attachment_rejects_legacy_and_mixed_catalogs_atomically(
    tmp_path, catalog
):
    repo = DataEngine()
    repo.catalog = catalog
    good = tmp_path / "good.sqlite"
    with sqlite3.connect(good) as connection:
        write_sqlite_catalog(connection, catalog)
    repo.attach_boundary_store(good)
    document = catalog.to_dict()
    document.pop("build_id")
    document["processor_sha256"] = "a" * 64
    document["build_id"] = sha256(canonical_json(document).encode()).hexdigest()
    other = tmp_path / "other.sqlite"
    with sqlite3.connect(other) as connection:
        write_sqlite_catalog(connection, DatasetCatalog(document))
    with pytest.raises(ContractError, match="different data release"):
        repo.attach_boundary_store(other)
    legacy = tmp_path / "legacy.sqlite"
    with sqlite3.connect(legacy) as connection:
        connection.execute("CREATE TABLE boundaries (district_number TEXT, wkb BLOB)")
    with pytest.raises(ContractError, match="lacks"):
        repo.attach_boundary_store(legacy)
    assert repo._boundary_store is not None
    assert repo._boundary_store.path == good


def test_snapshot_constructor_preserves_bound_release_and_all_transfer_facts(
    tmp_path, transfer_repo, catalog
):
    repo, district, campuses = transfer_repo
    repo.catalog = catalog
    source = pd.concat([frame(count=-999), frame("123456004", 4)], ignore_index=True)
    repo.apply_transfers_from_dataframe(
        source, value_policy=catalog.policy("campus_transfer_reports")
    )
    snapshot = tmp_path / "repo.pkl"
    with snapshot.open("wb") as output:
        output.write(SNAPSHOT_MAGIC + catalog.build_id.encode() + b"\n")
        pickle.dump(repo, output)

    restored = DataEngine(snapshot=snapshot)
    origin = restored.campuses[campuses[0].id]
    assert restored.catalog.build_id == catalog.build_id
    assert restored.transfers_out(origin)[0][1:] == (None, True)
    assert restored._transfer_measurements[(origin.id, campuses[1].id)].raw == -999
    assert restored._xfers_unresolved == repo._xfers_unresolved
    assert restored._xfers_missing == repo._xfers_missing
    assert origin._repo is restored
    assert restored.districts[district.id]._repo is restored


@pytest.mark.parametrize("raw", ["9" * 400, 10**400, math.inf, -math.inf, "1e999"])
def test_invalid_numeric_values_return_status_without_overflow(raw):
    value = measure(raw, unit="count")
    assert value.status == ValueStatus.INVALID
    assert value.numeric is None
