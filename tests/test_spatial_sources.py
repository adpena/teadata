"""Source-contract checks; actual GIS integrity and release qualification are separate."""

import json
from hashlib import sha256
from pathlib import Path

from teadata.provenance import ProvenanceStore
from teadata.teadata_config import load_config


ROOT = Path(__file__).resolve().parents[1] / "teadata"


def test_spatial_products_are_selected_independently_from_reporting_year():
    cfg = load_config(ROOT / "teadata_sources.yaml")
    for dataset, year, period, filename, receipt in (
        ("districts", 2026, (2025, 2026), "tea-districts-2026-20261103.geojson",
         "97be381420c20552e5baab45a68ed4b4c823ecff1d3945c075df1bd040796852"),
        ("campuses", 2025, (2024, 2025), "tea-campuses-2025-20261103.geojson",
         "ed8e190d5969357f60166ed061c55526eb09d4ef604d17b463fc82c77e1a0233"),
    ):
        source = cfg.resolve_dataset(dataset, section="spatial", **cfg.spatial_selection(dataset, 2025))
        assert source.requested_year == source.resolved_year == year
        assert source.metadata.version == str(year)
        assert source.metadata.period.to_dict() == {
            "kind": "school_year", "start_year": period[0], "end_year": period[1]}
        assert Path(source.path).name == filename and receipt in source.metadata.provenance_ids
        assert source.metadata.publisher == "Texas Education Agency"
    report = cfg.resolve_dataset("accountability", 2025)
    assert report.requested_year == report.resolved_year == 2025
    legacy = cfg.resolve_dataset("districts", 2025, section="spatial", exact=True)
    assert legacy.metadata.period.kind == "unknown" and legacy.metadata.source_url is None
    assert "Original download timestamp and acquisition URL" in legacy.metadata.notes


def test_campus_revision_changes_only_the_two_removed_reader_fields():
    cfg = load_config(ROOT / "teadata_sources.yaml")
    old_path = Path(__file__).parent / "fixtures" / "spatial" / "spatial--campuses--2025--default.json"
    old_raw = old_path.read_bytes()
    historical = ProvenanceStore(ROOT.parent / "provenance").read_record(
        "ae9ce2eb05c52d84875fdc00e689b5d117a2dfc040203723480d9938b1a955e4")
    assert historical["artifact"] == {
        "filename": old_path.name, "sha256": sha256(old_raw).hexdigest(), "size_bytes": len(old_raw)}
    old = json.loads(old_raw)["columns"]
    current = cfg.column_schema("campuses", 2025, section="spatial")
    assert len(old) == 120 and len(current) == 118
    assert current == [name for name in old if name not in {"USER_School_Status", "USER_Update_Date"}]
    assert {"USER_School_Number", "USER_District_Number", "USER_School_Status_Date", "geometry"} <= set(current)
    assert cfg.column_schema("districts", 2026, section="spatial") == json.loads(
        (ROOT / "schemas" / "spatial--districts--2025--default.json").read_text())["columns"]


def test_source_references_do_not_promote_unreviewed_dictionary_or_masking_claims():
    cfg = load_config(ROOT / "teadata_sources.yaml")
    for dataset in ("districts", "campuses"):
        source = cfg.resolve_dataset(dataset, section="spatial", **cfg.spatial_selection(dataset, 2025))
        references = {ref.reference_id: ref for ref in source.metadata.references}
        assert references["dictionary"].status == references["masking"].status == "unreviewed"
        assert references["publisher_item"].role == references["publisher_service"].role == "source_notes"
        assert references["publisher_layer"].role == "schema"
        for name in ("publisher_item", "publisher_service", "publisher_layer"):
            assert references[name].status == "available"
            assert references[name].provenance_ids and len(references[name].reviewed_sha256) == 64
    assert "208902" in cfg.resolve_dataset("districts", 2026, section="spatial").metadata.notes
    campus = cfg.resolve_dataset("campuses", 2025, section="spatial")
    assert "286 null source points" in campus.metadata.notes
    assert "source update dates remain unknown" in campus.metadata.notes
