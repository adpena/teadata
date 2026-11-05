import csv
from dataclasses import replace
from hashlib import sha256
from uuid import uuid4

import pytest

from teadata.column_schema import source_schema_fingerprint
from teadata.dataset_metadata import DatasetMetadata, EditionKey, ResolvedDataset
from teadata.entities import Campus, District
from teadata.manifest import ContractError, canonical_json
from teadata.peims_courses import DATASET as COURSES
from teadata.provenance import ProvenanceStore
from teadata.query import Query
from teadata.release import DataRelease
from teadata.scripts.build_tapr_profiles import prepare_profiles
from teadata.source_parity import validate_source_parity
from teadata.statewide import _reviewed_tapr
from teadata.teadata_config import Config

FAMILY = "college_readiness_components"
METRIC = "ccmr_annual_graduates_dual_credit_all_students_rate"
PERIOD = {"kind": "school_year", "start_year": 2023, "end_year": 2024}


@pytest.fixture
def comparison_release(tmp_path, monkeypatch, request):
    family = getattr(request, "param", FAMILY)
    store = ProvenanceStore(tmp_path / "source-store")
    masking = tmp_path / "masking.txt"
    masking.write_text("Synthetic reviewed rate masking methodology")
    masking_id = store.record_acquisition(masking, ResolvedDataset("reference:masking", "reference", 2025,
                                          2025, str(masking), DatasetMetadata()))
    definitions = {}
    for scope, observations in (
        ("district", [["227901", "35.6"], ["001901", "0"], ["001902", "-1"], ["001903", ""]]),
        ("campus", [["057909002", "33.8"]]),
        ("state", [["25.1"]]),
    ):
        dataset = f"{scope}_tapr_{family}"
        path = tmp_path / f"{scope}.csv"
        key = f"{scope}_number" if scope != "state" else None
        labels, codes, fields = (["ID"], [scope.upper()], [key]) if key else ([], [], [])
        labels += ["Dual credit rate"]
        codes += ["RATE"]
        fields += ["dual_credit_rate"]
        metrics = {METRIC: {"field": "dual_credit_rate", "unit": "percent"}}
        table_period = PERIOD
        if family == "student_profile":
            table_period = {"kind": "unknown"}
            metrics[METRIC]["period"] = PERIOD
            labels += ["Membership count"]
            codes += ["COUNT"]
            fields += ["membership_count"]
            observations = [row + ["1200" if scope == "state" else "105"] for row in observations]
            metrics["membership_count"] = {
                "field": "membership_count", "unit": "count",
                "period": {"kind": "school_year", "start_year": 2024, "end_year": 2025},
            }
        with path.open("w", newline="") as stream:
            csv.writer(stream).writerows([labels, codes, *observations])
        mapping = {
            "table": family, "input_sha256": sha256(path.read_bytes()).hexdigest(),
            "source_schema_sha256": source_schema_fingerprint(labels, codes),
            "metric_period": table_period,
            "layout": {"label_record": 1, "code_record": 2, "first_data_record": 3,
                       "last_data_record": 2 + len(observations), "annotations": []},
            "fields": [{"position": i, "code": code, "label": label, "field": field,
                        "disposition": "identity" if field == key else "fact",
                        "publisher_type": None, "publisher_description": None,
                        "dictionary_status": "not_applicable" if field == key else "not_documented"}
                       for i, (code, label, field) in enumerate(zip(codes, labels, fields, strict=True), 1)],
        }
        schema = tmp_path / f"{scope}.json"
        schema.write_text(canonical_json({"schema_version": 2, "mapping_version": 1, "datasets": [
            {"dataset": dataset, "edition": "2025", "source_only": True, "tables": [mapping]}]}) + "\n")
        original = store.record_acquisition(path, ResolvedDataset(dataset, "data_sources", 2025, 2025,
                                             str(path), DatasetMetadata(version="2025")))
        headers = store.record_acquisition(schema, ResolvedDataset(dataset + ":schema", "reference", 2025, 2025,
                                            str(schema), DatasetMetadata()))
        definitions[dataset] = {2025: {"path": str(path), "metadata": {
            "version": "2025", "period": table_period, "provenance_ids": [original],
            "metrics": metrics,
            "references": [
                {"reference_id": "columns_source", "role": "schema", "path": schema.name,
                 "status": "available", "reviewed_sha256": sha256(schema.read_bytes()).hexdigest(),
                 "provenance_ids": [headers]},
                {"reference_id": "masking", "role": "masking_methodology", "path": masking.name,
                 "status": "available", "reviewed_sha256": sha256(masking.read_bytes()).hexdigest(),
                 "provenance_ids": [masking_id]}]},
            "reader": {"profiles": {"source": {"column_schema": schema.name, "tapr_source": {
                "scope": scope, "tables": {family: str(path)}}}}}}}
    cfg = Config.from_dict({"data_sources": definitions})
    cfg.base_dir = tmp_path
    prepared = tmp_path / "prepared"
    result = prepare_profiles(cfg, edition="2025", family=family, scopes=["district", "campus", "state"],
                              input_dir=None, output_dir=prepared, reference_store=store)
    selected = {entry["dataset"]: (entry["processing_record_id"], () if scope == "state" else (f"{scope}_number",))
                for scope, entry in result["scopes"].items()}
    output = tmp_path / "cohort"
    built = prepare_profiles(cfg, edition="2025", scopes=[], input_dir=None, output_dir=output,
                             reference_store=ProvenanceStore(prepared / "provenance"), prepared_inputs=selected)
    release = DataRelease(output / "release.json", offline=True)
    editions = {entry["key"]["scope"]: EditionKey.from_dict(entry["key"]) for entry in built["datasets"].values()}
    assert built["datasets"][editions["state"].dataset]["key_columns"] == []
    assert len(release.verify()) == 3
    monkeypatch.setattr("teadata.statewide.load_config", lambda: cfg)
    _reviewed_tapr.cache_clear()
    yield release, editions
    _reviewed_tapr.cache_clear()


def test_statewide_operator_keeps_exact_values_and_distinguishes_unavailable(comparison_release):
    release, editions = comparison_release
    districts = [District(uuid4(), str(number), district_number=number)
                 for number in ("227901", "001901", "001902", "001903", "999999")]
    query = Query(districts, repo=None)
    result = query >> ("compare_statewide", release, editions["district"], editions["state"], METRIC)
    assert query.all() == districts
    assert [(row.local_raw, row.local_status, row.difference_pp) for row in result] == [
        ("35.6", "observed", 10.5), ("0", "observed", -25.1),
        ("-1", "suppressed", None), ("", "missing", None), (None, "unavailable", None),
    ]
    assert result.first().period.to_dict() == PERIOD
    assert result.first().statewide_edition == editions["state"]
    assert result.to_dicts(columns=["statewide_percent"])[0] == {"statewide_percent": 25.1}
    campus = Campus(uuid4(), None, "Garland", "", False, campus_number="057909002")
    assert Query([campus], repo=None).compare_statewide(
        release=release, local=editions["campus"], statewide=editions["state"], metric=METRIC
    ).first().difference_pp == 8.7
    private = Campus(uuid4(), None, "Private school", "", False, is_private=True,
                     source_namespace="tefa", source_id="057909002")
    row = Query([private], repo=None).compare_statewide(
        release=release, local=editions["campus"], statewide=editions["state"], metric=METRIC
    ).first()
    assert (row.identifier, row.local_status, row.difference_pp) == ("tefa:057909002", "not_applicable", None)
    with pytest.raises(ContractError, match="scope"):
        Query([campus], repo=None) >> ("compare_statewide", release, editions["district"], editions["state"], METRIC)
    with pytest.raises(ContractError, match="matched"):
        query >> ("compare_statewide", release, editions["district"], replace(editions["state"], edition="2026"), METRIC)
    with pytest.raises(ContractError, match="reviewed numeric"):
        query >> ("compare_statewide", release, editions["district"], editions["state"], "an_unreviewed_rate")


def test_statewide_operator_rejects_a_rebound_metric_before_leaf_reads(comparison_release, monkeypatch):
    release, editions = comparison_release
    metadata = release.catalog.metadata

    def rebound(dataset, *, edition):
        selected = metadata(dataset, edition=edition)
        if edition.scope == "district":
            return replace(selected, metrics={METRIC: replace(selected.metrics[METRIC], field="unrelated_rate")})
        return selected

    monkeypatch.setattr(release.catalog, "metadata", rebound)
    monkeypatch.setattr(release, "analytical_dataset", lambda *a: pytest.fail("Unreviewed rate opened a leaf"))
    query = Query([District(uuid4(), "Austin", district_number="227901")], repo=None)
    with pytest.raises(ContractError, match="reviewed numeric"):
        query >> ("compare_statewide", release, editions["district"], editions["state"], METRIC)


def test_raw_parity_replays_the_statewide_row_without_a_geographic_key(comparison_release):
    release, editions = comparison_release
    result = validate_source_parity(release, ProvenanceStore(release.path.parent / "provenance"),
                                   edition_ids=[editions["state"].identity])
    assert result["status"] == "PASS_RAW_PARITY"
    assert (result["coverage"]["raw_cells"], result["coverage"]["compared_cells"]) == (1, 2)


@pytest.mark.parametrize("comparison_release", ["student_profile"], indirect=True)
def test_student_profile_comparison_keeps_counts_and_distinct_periods(comparison_release):
    release, editions = comparison_release
    districts = [District(uuid4(), "Observed", district_number="227901"),
                 District(uuid4(), "Uninterpreted", district_number="001902")]
    rows = list(Query(districts, repo=None).compare_statewide(
        release=release, local=editions["district"], statewide=editions["state"]))
    rate, count, unknown, _ = rows
    assert (rate.unit, rate.period.end_year, rate.difference_pp) == ("percent", 2024, 10.5)
    assert (count.unit, count.period.end_year, count.local_value, count.statewide_value) == ("count", 2025, 105, 1200)
    assert count.difference_pp is None
    assert (unknown.local_raw, unknown.local_status, unknown.difference) == ("-1", "uninterpreted", None)


def test_campus_course_geography_hold_precedes_any_leaf_or_catalog_read():
    campus = Campus(uuid4(), None, "School", "", False, campus_number="057909002")
    local = EditionKey(COURSES, "2025", "1" * 64, "campus", "campus", "reported campus course")
    statewide = EditionKey(COURSES, "2025", "2" * 64, "state", "state", "reported state course")
    with pytest.raises(ContractError, match="geography qualification"):
        Query([campus], repo=None).compare_statewide(
            release=None, local=local, statewide=statewide, dimensions={"course_code": "03100500"})
