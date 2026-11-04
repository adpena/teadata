"""Observed join coverage and honest failure history on small synthetic sources."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from teadata import DataEngine, District
from teadata import enrichment, load_data
from teadata.enrichment import campuses, districts
from teadata.manifest import BuildRecorder, ContractError
from teadata.identifiers import EntityIdentityError
from teadata.provenance import ProvenanceStore
from teadata.teadata_config import load_config


@pytest.mark.parametrize("updated", [0, 7])
def test_optional_failures_and_unsupported_selections_are_not_dropped(monkeypatch, updated):
    def missing(*args):
        raise FileNotFoundError("No source file")

    monkeypatch.setattr(enrichment, "_registry", {
        "broken": SimpleNamespace(apply=missing),
        "legacy": SimpleNamespace(apply=lambda *a: {"year": 2024, "updated": updated}),
    })
    repo = DataEngine()
    stats = enrichment.run_enrichment(repo, "unused", 2025, datasets=["broken", "unknown", "legacy"])
    assert stats["broken"]["outcome"]["status"] == "failed"
    assert stats["broken"]["outcome"]["source_status"] == "missing"
    assert stats["unknown"]["outcome"]["status"] == "unsupported"
    assert stats["legacy"]["updated"] == updated and stats["legacy"]["year"] == 2024
    assert stats["legacy"]["outcome"]["status"] == "succeeded"
    assert stats["legacy"]["outcome"]["matched"] is None
    assert stats["legacy"]["outcome"]["expected"] is None
    assert stats["legacy"]["outcome"]["unmatched"] is None
    assert stats["legacy"]["outcome"]["not_applicable"] is None
    validation = load_data._release_validation(repo)
    assert validation["enrichment_outcomes"] == [value["outcome"] for value in stats.values()]
    assert validation["quality_status"] == "needs_review"
    assert enrichment.run_enrichment(repo, "unused", 2025, datasets=[]) == {}


def test_interrupts_are_not_reported_as_ordinary_optional_failures(monkeypatch):
    def interrupted(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(enrichment, "_registry", {"stop": SimpleNamespace(apply=interrupted)})
    with pytest.raises(KeyboardInterrupt):
        enrichment.run_enrichment(DataEngine(), "unused", 2025)


@pytest.mark.parametrize("scope,module,dataset,rows", [
    ("district", districts, "sample", "011901,1\n11901,2\n"),
    ("campus", campuses, "sample", "011901001,1\n11901001,2\n"),
    ("campus", campuses, "campus_peims_financials", "011901001,1\n11901001,2\n"),
])
def test_enrichment_rejects_canonical_duplicate_source_rows(tmp_path, scope, module, dataset, rows):
    path = tmp_path / "values.csv"
    path.write_text(f"{scope}_number,value\n{rows}")
    cfg = tmp_path / "sources.yaml"
    cfg.write_text(yaml.safe_dump({"data_sources": {dataset: {2025: str(path)}}}))
    plural = {"campus": "campuses", "district": "districts"}[scope]
    apply = getattr(module, f"enrich_{plural}_from_config")
    with pytest.raises(EntityIdentityError, match=f"{dataset} .*2 rows share duplicate {scope}_number"):
        apply(DataEngine(), str(cfg), dataset, 2025, select=["value"])


def test_enrichment_conflicts_leave_all_entities_unchanged(tmp_path):
    from uuid import uuid4

    path = tmp_path / "values.csv"
    path.write_text("district_number,value\n011901,1\n011902,2\n")
    cfg = tmp_path / "sources.yaml"
    cfg.write_text(yaml.safe_dump({"data_sources": {"sample": {2025: str(path)}}}))
    repo = DataEngine()
    first = District(uuid4(), "first", district_number="011901")
    second = District(uuid4(), "second", district_number="011902", meta={"value": 3})
    repo.add_district(first)
    repo.add_district(second)
    with pytest.raises(ValueError, match="sample .*conflicting existing field 'value'"):
        districts.enrich_districts_from_config(repo, str(cfg), "sample", 2025)
    assert first.meta == {} and second.meta == {"value": 3}


@pytest.mark.parametrize(("data", "message"), [
    ("district_number,value\n-011901,1\n", "invalid identifier"),
    ("district_number,value\n11901.5,1\n", "invalid identifier"),
    ("DISTRICT,district_number,value\n011901,011902,1\n", "conflicting identifier columns"),
])
def test_enrichment_rejects_malformed_or_conflicting_original_identifiers(tmp_path, data, message):
    from uuid import uuid4

    path = tmp_path / "values.csv"
    path.write_text(data)
    cfg = tmp_path / "sources.yaml"
    cfg.write_text(yaml.safe_dump({"data_sources": {"sample": {2025: str(path)}}}))
    repo = DataEngine()
    district = District(uuid4(), "first", district_number="011901")
    repo.add_district(district)
    with pytest.raises(EntityIdentityError, match=message):
        districts.enrich_districts_from_config(repo, str(cfg), "sample", 2025)
    assert district.meta == {}


@pytest.mark.parametrize(("alias", "error"), [
    ("district_number_canon", AttributeError),
    ("_district_number_canon", EntityIdentityError),
])
def test_enrichment_rejects_identity_and_read_only_aliases_before_writes(alias, error):
    from uuid import uuid4
    from teadata.enrichment.base import apply_entity_attributes

    district = District(uuid4(), "test", district_number="011901")
    with pytest.raises(error, match=alias):
        apply_entity_attributes([(district, {"value": 1})], dataset="sample", year=2025, aliases={"value": alias})
    assert district.meta == {}


def test_enrichment_checks_authoritative_metadata_instead_of_lazy_seed():
    from uuid import uuid4
    from teadata.enrichment.base import EnrichmentConflictError, apply_entity_attributes
    from teadata.persistence.sqlalchemy_store import LazyMetaDict

    district = District(uuid4(), "test", meta=LazyMetaDict(lambda: {"value": "B"}, seed={"value": "A"}))
    with pytest.raises(EnrichmentConflictError, match="conflicting existing field 'value'"):
        apply_entity_attributes([(district, {"value": "A"})], dataset="sample", year=2025, aliases={"value": "rating"})
    assert district.meta["value"] == "B" and district.rating is None


def test_accountability_selects_reporting_grain_before_district_join(monkeypatch):
    from uuid import uuid4

    frame = pd.DataFrame({"District Number": [11901, 11901], "Campus Number": [11901001, None],
                          "District": ["Test ISD", "Test ISD"], "overall_rating": ["C", "A"]})
    config = SimpleNamespace(load_df=lambda *args, **kwargs: (2025, frame))
    monkeypatch.setattr(districts, "load_config", lambda _: config)
    repo = DataEngine()
    district = District(uuid4(), "test", district_number="011901")
    repo.add_district(district)
    assert districts.enrich_districts_from_config(
        repo, "unused", "accountability", 2025, select=["overall_rating"],
        aliases={"overall_rating": "rating"}, reader_kwargs={"reader_profile": "ratings"},
    ) == (2025, 1)
    assert district.meta == {"overall_rating": "A"} and district.rating == "A"


@pytest.mark.parametrize("scope,module,dataset", [
    ("campus", campuses, "sample"),
    ("district", districts, "sample"),
    ("campus", campuses, "campus_peims_financials"),
])
@pytest.mark.parametrize("empty", [False, True])
def test_real_join_counts_preserve_zero_negative_and_absent_rows(tmp_path, scope, module, dataset, empty):
    key = scope + "_number"
    numbers = (["001902001", "001902002", "001902003"] if scope == "campus"
               else ["001902", "001903", "001904"])
    path = tmp_path / "values.csv"
    path.write_text(f"{key},value\n" + ("" if empty else f"{numbers[0]},0\n{numbers[1]},-999\n"))
    cfg_path = tmp_path / "sources.yaml"
    cfg_path.write_text(yaml.safe_dump({"data_sources": {dataset: {2025: str(path)}}}))
    entities = [SimpleNamespace(**{key: number}, meta={}) for number in (*numbers, None)]
    plural = "campuses" if scope == "campus" else "districts"
    repo = SimpleNamespace(**{"_" + plural: dict(enumerate(entities))})
    apply = getattr(module, f"enrich_{plural}_from_config")
    assert apply(repo, str(cfg_path), dataset, 2026, select=["value"]) == (2025, 0 if empty else 2)
    outcome, = repo._enrichment_outcomes
    assert outcome["requested_year"] == 2026 and outcome["resolved_year"] == 2025
    assert outcome["expected"] == 4 and outcome["missing_identifier"] == 1
    assert outcome["not_applicable"] == (0 if scope == "campus" else None)
    assert outcome["matched"] == (0 if empty else 2)
    assert outcome["unmatched"] == (3 if empty else 1)
    assert outcome["source_rows"] == (0 if empty else 2)
    assert outcome["source_status"] == ("empty" if empty else "available")
    assert outcome["status"] == ("unmatched" if empty else "partial")
    if not empty:
        assert entities[0].meta["value"] == 0
        assert entities[1].meta["value"] == -999
    assert "value" not in entities[2].meta


@pytest.mark.parametrize("dataset", ["campus_tapr_student_staff_profile", "campus_peims_financials"])
@pytest.mark.parametrize("all_private", [False, True])
def test_campus_joins_separate_private_population_from_public_coverage(tmp_path, monkeypatch, dataset, all_private):
    columns = campuses.DEFAULT_PEIMS_FINANCIAL_COLUMNS if dataset == "campus_peims_financials" else ["value"]
    path = tmp_path / "campuses.csv"
    path.write_text(
        ",".join(["campus_number", *columns]) + "\n"
        + "001902001," + ",".join("0" for _ in columns) + "\n"
        + "001902002," + ",".join("-999" for _ in columns) + "\n"
    )
    cfg_path = tmp_path / "sources.yaml"
    cfg_path.write_text(yaml.safe_dump({"data_sources": {dataset: {2025: str(path)}}}))
    private = SimpleNamespace(campus_number=None, is_private=True, source_namespace="tefa",
                              source_id="opaque-private", meta={"private_token": 19})
    public = [SimpleNamespace(campus_number=number, is_private=False, meta={})
              for number in ("001902001", "001902002", "001902003", None)]
    repo = SimpleNamespace(_campuses=dict(enumerate([private] if all_private else [*public, private])))
    result = enrichment.run_enrichment(repo, str(cfg_path), 2025, datasets=[dataset])[dataset]
    outcome = result["outcome"]
    assert outcome["not_applicable"] == 1
    assert outcome["expected"] == (0 if all_private else 4)
    assert outcome["matched"] == result["updated"] == (0 if all_private else 2)
    assert outcome["unmatched"] == outcome["missing_identifier"] == (0 if all_private else 1)
    assert outcome["status"] == ("not_applicable" if all_private else "partial")
    assert outcome["source_rows"] == 2
    assert private.meta == {"private_token": 19}
    if not all_private:
        assert public[0].meta[columns[0]] == 0 and public[1].meta[columns[0]] == -999
        assert public[2].meta == public[3].meta == {}
        return

    # Optional population exclusion cannot bypass the required zero-match gate.
    def apply(repo, config, name, year, **kwargs):
        if name == dataset:
            result = enrichment._registry[dataset].apply(repo, config, year)
            return result["year"], result["updated"]
        return 2025, 1

    monkeypatch.setattr(load_data, "CFG", str(cfg_path))
    monkeypatch.setattr(load_data, "YEAR", 2025)
    monkeypatch.setattr(load_data, "load_config", lambda _: SimpleNamespace(
        load_df=lambda *a: (2025, pd.DataFrame()), value_policies={"campus_transfers": None},
    ))
    monkeypatch.setattr(load_data, "enrich_districts_from_config", apply)
    monkeypatch.setattr(load_data, "enrich_campuses_from_config", apply)
    required = SimpleNamespace(_campuses={0: private}, apply_transfers_from_dataframe=lambda *a, **k: 1)
    with pytest.raises(ContractError, match="No campus records matched"):
        load_data.run_enrichments(required)
    failure = required._enrichment_outcomes[-1]
    assert failure["status"] == "failed" and failure["required"]
    assert failure["not_applicable"] == 1 and failure["missing_identifier"] == 0


@pytest.mark.parametrize("dataset,scope", [
    ("district_tapr_student_staff_profile", "district"),
    ("campus_tapr_student_staff_profile", "campus"),
    ("campus_peims_financials", "campus"),
])
def test_missing_source_join_column_is_failed_optional_and_raises_when_required(
    tmp_path, monkeypatch, dataset, scope,
):
    path = tmp_path / "missing-id.csv"
    path.write_text("value\n0\n")
    cfg_path = tmp_path / "sources.yaml"
    cfg_path.write_text(yaml.safe_dump({"data_sources": {dataset: {2025: str(path)}}}))
    optional = DataEngine()
    stats = enrichment.run_enrichment(optional, str(cfg_path), 2025, datasets=[dataset])
    outcome = stats[dataset]["outcome"]
    assert outcome["status"] == "failed" and outcome["error_type"] == "KeyError"
    assert "missing required join column: " + scope + "_number" in outcome["error"]
    assert outcome["expected"] is None and outcome["matched"] is None

    # Let the required pipeline reach the real selected helper; other stages
    # stand in for already successful prerequisites and do not open sources.
    def apply(repo, config, name, year, **kwargs):
        if name == dataset:
            return enrichment._registry[dataset].apply(repo, config, year)
        return 2025, 1

    monkeypatch.setattr(load_data, "CFG", str(cfg_path))
    monkeypatch.setattr(load_data, "YEAR", 2025)
    monkeypatch.setattr(load_data, "load_config", lambda _: SimpleNamespace(
        load_df=lambda *a: (2025, pd.DataFrame()), value_policies={"campus_transfers": None},
    ))
    monkeypatch.setattr(load_data, "enrich_districts_from_config", apply)
    monkeypatch.setattr(load_data, "enrich_campuses_from_config", apply)
    required = DataEngine()
    monkeypatch.setattr(required, "apply_transfers_from_dataframe", lambda *a, **k: 1)
    with pytest.raises(KeyError, match="missing required join column: " + scope + "_number"):
        load_data.run_enrichments(required)
    failure = required._enrichment_outcomes[-1]
    assert failure["dataset"] == dataset and failure["required"]
    assert failure["status"] == "failed" and failure["error_type"] == "KeyError"


def test_required_zero_match_retains_real_failed_processing_output(tmp_path, monkeypatch):
    path = tmp_path / "ratings.csv"
    path.write_text("district_number,overall_rating\n001902,A\n")
    cfg_path = tmp_path / "sources.yaml"
    document = {"data_sources": {"accountability": {2025: {"path": str(path)}}}}
    cfg_path.write_text(yaml.safe_dump(document))
    store = ProvenanceStore(tmp_path / "history")
    cfg = load_config(cfg_path)
    acquisition = store.record_acquisition(path, cfg.resolve_dataset("accountability", 2025))
    document["data_sources"]["accountability"][2025]["metadata"] = {"provenance_ids": [acquisition]}
    cfg_path.write_text(yaml.safe_dump(document))
    cfg = load_config(cfg_path)
    monkeypatch.setattr(load_data, "CFG", str(cfg_path))
    monkeypatch.setattr(load_data, "YEAR", 2025)
    monkeypatch.setattr(load_data, "_repo_cache_dir", lambda: tmp_path)
    # Select the real synthetic table without a named publisher profile.
    def enrich(repo, cfg_path, dataset, year, **kwargs):
        return districts.enrich_districts_from_config(repo, cfg_path, dataset, year)

    monkeypatch.setattr(load_data, "enrich_districts_from_config", enrich)
    repo = SimpleNamespace(_districts={0: SimpleNamespace(district_number="001903", meta={})})
    with BuildRecorder(cfg, store):
        with pytest.raises(ContractError, match="No district records matched") as failure:
            load_data.run_enrichments(repo)
    outcome, = repo._enrichment_outcomes
    assert outcome["required"] and outcome["status"] == "failed"
    assert outcome["matched"] == 0 and outcome["unmatched"] == 1
    record, = store.history("teadata_release")
    assert record["inputs"] == [acquisition]
    assert record["parameters"]["build_status"] == "failed"
    assert set(record["outputs"]) == {"enrichment_diagnostic"}
    archived = store.verify_artifact(record["outputs"]["enrichment_diagnostic"]["sha256"])
    assert json.loads(archived.read_text()) == record["validation"]
    assert record["validation"]["enrichment_outcomes"] == [outcome]
    assert any(record["record_id"] in note for note in failure.value.__notes__)
    assert store.read_record(acquisition)["downloaded_at"] is None
    with pytest.raises(ContractError, match="failed required enrichment"):
        load_data._release_validation(repo)


@pytest.mark.parametrize("recording_fails", [False, True])
def test_required_missing_source_keeps_original_exception_and_explicit_outcome(monkeypatch, recording_fails):
    failure = FileNotFoundError("Missing selected CSV")

    def missing(*args, **kwargs):
        raise failure

    monkeypatch.setattr(load_data, "load_config", lambda _: SimpleNamespace())
    monkeypatch.setattr(load_data, "enrich_districts_from_config", missing)
    if recording_fails:
        def unavailable(*args):
            raise OSError("No diagnostic storage")
        monkeypatch.setattr(load_data, "_retain_failed_enrichment", unavailable)
    repo = DataEngine()
    with pytest.raises(FileNotFoundError) as caught:
        load_data.run_enrichments(repo)
    assert caught.value is failure
    outcome, = repo._enrichment_outcomes
    assert outcome["required"] and outcome["status"] == "failed"
    assert outcome["source_status"] == "missing"
    assert outcome["expected"] is None and outcome["matched"] is None
    if recording_fails:
        assert any("Could not retain" in note for note in failure.__notes__)
    else:
        assert not hasattr(failure, "__notes__")


def test_legacy_required_counts_are_not_inferred_as_complete_coverage(monkeypatch):
    cfg = SimpleNamespace(load_df=lambda *a: (2025, pd.DataFrame()),
                          value_policies={"campus_transfers": None})
    monkeypatch.setattr(load_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(load_data, "enrich_districts_from_config", lambda *a, **k: (2025, 1))
    monkeypatch.setattr(load_data, "enrich_campuses_from_config", lambda *a, **k: (2025, 1))
    repo = DataEngine()
    monkeypatch.setattr(repo, "apply_transfers_from_dataframe", lambda *a, **k: 1)
    assert load_data.run_enrichments(repo) is None
    outcomes = load_data._release_validation(repo)["enrichment_outcomes"]
    assert len(outcomes) == 8 and all(value["required"] for value in outcomes)
    assert all(value["updated"] == 1 for value in outcomes)
    assert all(value["expected"] is None and value["matched"] is None for value in outcomes)
