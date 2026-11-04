import csv
from dataclasses import replace
from hashlib import sha256
from uuid import uuid4

import pytest

from teadata.column_schema import source_schema_fingerprint
from teadata.dataset_metadata import DatasetMetadata, EditionKey, ResolvedDataset
from teadata.entities import Campus, District
from teadata.manifest import ContractError, canonical_json
from teadata.provenance import ProvenanceStore
from teadata.query import Query
from teadata.release import DataRelease
from teadata.scripts.build_tapr_profiles import prepare_profiles
from teadata.statewide import _reviewed_grad2
from teadata.teadata_config import Config

FAMILY = "college_readiness_components"
METRIC = "ccmr_annual_graduates_dual_credit_all_students_rate"
PERIOD = {"kind": "school_year", "start_year": 2023, "end_year": 2024}


@pytest.fixture
def comparison_release(tmp_path, monkeypatch):
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
        dataset = f"{scope}_tapr_{FAMILY}"
        path = tmp_path / f"{scope}.csv"
        key = f"{scope}_number" if scope != "state" else None
        labels, codes, fields = (["ID"], [scope.upper()], [key]) if key else ([], [], [])
        labels += ["Dual credit rate"]
        codes += ["RATE"]
        fields += ["dual_credit_rate"]
        with path.open("w", newline="") as stream:
            csv.writer(stream).writerows([labels, codes, *observations])
        mapping = {
            "table": FAMILY, "input_sha256": sha256(path.read_bytes()).hexdigest(),
            "source_schema_sha256": source_schema_fingerprint(labels, codes),
            "metric_period": PERIOD,
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
            "version": "2025", "period": PERIOD, "provenance_ids": [original],
            "metrics": {METRIC: {"field": "dual_credit_rate", "unit": "percent"}},
            "references": [
                {"reference_id": "columns_source", "role": "schema", "path": schema.name,
                 "status": "available", "reviewed_sha256": sha256(schema.read_bytes()).hexdigest(),
                 "provenance_ids": [headers]},
                {"reference_id": "masking", "role": "masking_methodology", "path": masking.name,
                 "status": "available", "reviewed_sha256": sha256(masking.read_bytes()).hexdigest(),
                 "provenance_ids": [masking_id]}]},
            "reader": {"profiles": {"source": {"column_schema": schema.name, "tapr_source": {
                "scope": scope, "tables": {FAMILY: str(path)}}}}}}}
    cfg = Config.from_dict({"data_sources": definitions})
    cfg.base_dir = tmp_path
    prepared = tmp_path / "prepared"
    result = prepare_profiles(cfg, edition="2025", family=FAMILY, scopes=["district", "campus", "state"],
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
    _reviewed_grad2.cache_clear()
    yield release, editions
    _reviewed_grad2.cache_clear()


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
    with pytest.raises(ContractError, match="reviewed percentage"):
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
    with pytest.raises(ContractError, match="reviewed percentage"):
        query >> ("compare_statewide", release, editions["district"], editions["state"], METRIC)
