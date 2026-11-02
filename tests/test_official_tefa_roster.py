"""Adapter contracts using small, exact rows from the retained official snapshot.

These exercise transport/normalization, not independent publisher truth. Release
acceptance cases are transcribed separately in the canonical reference cases.
"""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import uuid

import pandas as pd
import pytest

from teadata import DataEngine
from teadata.load_data import _add_tefa_entities, _load_private_entities
from teadata.manifest import ContractError
from teadata.teadata_config import load_config


ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests/fixtures/tefa/official-school-listings-2026-27.json"


@pytest.fixture
def cases():
    return json.loads(FIXTURE.read_text())["cases"]


@pytest.fixture
def cfg(tmp_path, cases):
    config = load_config(ROOT / "teadata/teadata_sources.yaml")
    current = config.data_sources["tefa"].entries[2027]
    path = tmp_path / "official-listings.json"
    path.write_text(json.dumps([case["record"] for case in cases.values()]))
    config.data_sources["tefa"].entries[2027] = replace(current, path=str(path))
    return config


def normalized_frame(cfg, records):
    mapping = cfg.resolve_dataset("tefa").reader["columns"]
    return pd.DataFrame(records).rename(columns={raw: name for name, raw in mapping.items()})


def test_real_reader_retains_every_namespaced_value_and_exact_identity(cfg, cases, catalog):
    year, frame = cfg.load_df("tefa", None)
    assert year == 2027
    assert list(frame.columns) == cfg.column_schema("tefa", 2027)
    repo = DataEngine()
    repo.catalog = catalog
    assert _add_tefa_entities(repo, frame, version="2026-27") == len(cases)
    mapping = cfg.resolve_dataset("tefa").reader["columns"]
    assert len(mapping) == 16
    assert catalog.datasets["tefa"]["reads"][0]["columns"] == list(mapping)
    assert list(catalog.metadata("tefa").identifiers) == ["tefa_listing_id"]
    for case in cases.values():
        raw = case["record"]
        campus = repo.campus_by_key("tefa:" + raw["id"])
        assert campus is not None
        assert campus.source_id == raw["id"] and campus.name == raw["name"].upper()
        assert campus.id == uuid.uuid5(uuid.NAMESPACE_URL, "teadata:private:tefa:" + raw["id"])
        assert campus.campus_number is None and campus.district_number is None and campus.district_id is None
        assert campus.enrollment is None
        assert {name: campus.meta[name] for name in mapping} == {name: raw[key] for name, key in mapping.items()}
        assert not {"id", "name", "type", "location", "tefa_id", "raw_id"} & campus.meta.keys()
        assert campus.tefa_listing_id == raw["id"]
        assert (repo >> ("campus", campus.lookup_key)).to_dicts()[0]["tefa_listing_id"] == raw["id"]
        assert campus.to_dict()["id"] == str(campus.id)
        assert catalog.measurement(campus, "enrollment").status.value == "not_applicable"


def test_real_service_arrays_grades_and_coordinate_exceptions(cfg, cases):
    repo = DataEngine()
    _load_private_entities(repo, cfg)

    def get(key):
        campus = repo.campus_by_key("tefa:" + cases[key]["record"]["id"])
        assert campus is not None
        return campus

    for key in ("local_prek", "mixed_online_first", "mixed_local_first"):
        campus = get(key)
        raw = cases[key]["record"]
        assert campus.location == (raw["location"]["lng"], raw["location"]["lat"])
        assert campus.meta["tefa_listing_service_type"] == raw["serviceType"]
        assert campus.meta["private_listing_location_status"] == "source_provided_unverified"
    assert get("local_prek").grade_range == "Pre-K"
    assert get("local_prek").school_type == "Early Childhood"
    assert get("mixed_online_first").grade_range == "6-12"
    assert get("online_only").grade_range == "9-12"
    assert get("online_only").school_type == "Private School"
    for key, status in (("online_only", "online_only"), ("missing_location", "missing"),
                        ("conflicting_tx_longitude", "invalid_source_coordinates")):
        assert get(key).location is None and get(key).point is None
        assert get(key).meta["private_listing_location_status"] == status
    assert get("conflicting_tx_longitude").meta["tefa_listing_location"]["lng"] > 0
    assert get("reversed_grade").grade_range is None
    assert get("reversed_grade").meta["private_listing_grade_status"] == "inconsistent"
    assert get("reversed_grade").meta["display_grade_range"] == cases["reversed_grade"]["record"]["displayGradeRange"]
    first, second = get("same_name_address_1"), get("same_name_address_2")
    assert first.name == second.name and first.meta["tefa_listing_address"] == second.meta["tefa_listing_address"]
    assert first.id != second.id and first.source_id != second.source_id
    assert get("missing_location").meta["school_website"] is None
    assert get("local_prek").meta["school_website"] == cases["local_prek"]["record"]["website"]


def test_private_selection_is_independent_of_public_build_year(cfg, monkeypatch):
    from teadata import load_data

    monkeypatch.setattr(load_data, "YEAR", 2025)
    assert cfg.resolve_dataset("tefa", 2025).metadata.period.kind == "unknown"
    assert cfg.resolve_dataset("tefa", None).resolved_year == 2027
    with pytest.raises(ContractError, match="profile is required"):
        cfg.load_df("tefa", 2025)
    repo = DataEngine()
    assert _load_private_entities(repo, cfg) == 9
    assert {campus.meta["private_source_version"] for campus in repo._campuses.values()} == {"2026-27"}


@pytest.mark.parametrize("field,value", [
    ("vendorType", "Other"), ("vendorTypes", ["Other"]),
    ("type", "Public School"), ("costOptions", "EFA-Pending"),
])
def test_only_explicit_eligible_private_school_tags_are_selected(cfg, cases, field, value):
    selected = deepcopy(cases["local_prek"]["record"])
    excluded = deepcopy(cases["online_only"]["record"])
    excluded[field] = value
    repo = DataEngine()
    assert _add_tefa_entities(repo, normalized_frame(cfg, [selected, excluded]), version="2026-27") == 1
    assert repo.campus_by_key("tefa:" + excluded["id"]) is None
    with pytest.raises(ContractError, match="no eligible"):
        _add_tefa_entities(DataEngine(), normalized_frame(cfg, [excluded]), version="2026-27")


@pytest.mark.parametrize("field,value", [
    ("serviceType", "local"), ("vendorTypes", "Schools"),
    ("serviceType", ["unreviewed"]), ("serviceType", [None]),
])
def test_unreviewed_array_shapes_fail_before_mutation(cfg, cases, field, value):
    valid = deepcopy(cases["local_prek"]["record"])
    invalid = deepcopy(cases["online_only"]["record"])
    invalid[field] = value
    repo = DataEngine()
    with pytest.raises(ContractError, match="array|review"):
        _add_tefa_entities(repo, normalized_frame(cfg, [valid, invalid]), version="2026-27")
    assert not repo._campuses


def test_duplicate_ids_and_generic_metadata_contract_are_rejected(cfg, cases):
    raw = cases["local_prek"]["record"]
    repo = DataEngine()
    with pytest.raises(ContractError, match="duplicate"):
        _add_tefa_entities(repo, normalized_frame(cfg, [raw, raw]), version="2026-27")
    assert not repo._campuses
    with pytest.raises(ContractError, match="namespaced"):
        _add_tefa_entities(repo, normalized_frame(cfg, [raw]).assign(id=raw["id"]), version="2026-27")
    assert not repo._campuses


@pytest.mark.parametrize("location", [
    {"lat": True, "lng": -96}, {"lat": "33", "lng": -96},
    {"lat": 91, "lng": -96}, {"lat": 33, "lng": -181}, {"lat": 33}, "unreviewed",
])
def test_invalid_coordinate_shapes_remain_raw_without_physical_point(cfg, cases, location):
    raw = deepcopy(cases["local_prek"]["record"])
    raw["location"] = location
    repo = DataEngine()
    _add_tefa_entities(repo, normalized_frame(cfg, [raw]), version="2026-27")
    campus = repo.campus_by_key("tefa:" + raw["id"])
    assert campus is not None
    assert campus.location is None
    assert campus.meta["tefa_listing_location"] == location
    assert campus.meta["private_listing_location_status"] == "invalid_source_coordinates"
