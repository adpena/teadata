import csv
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from teadata.column_schema import source_schema_fingerprint
from teadata.dataset_metadata import DataPeriod, DatasetMetadata, EditionKey, ResolvedDataset
from teadata.entities import Campus, District
from teadata.manifest import ContractError, canonical_json
from teadata.peims_courses import DATASET as COURSES
from teadata.provenance import ProvenanceStore
from teadata.query import Query
from teadata.release import DataRelease
from teadata.scripts.build_tapr_profiles import prepare_profiles
from teadata.source_parity import validate_source_parity
from teadata.statewide import _reviewed_tapr
from teadata.staar_comparison import _source as _reviewed_staar_source
from teadata.staar_comparison import _bindings, _parse_bindings, staar_rate_contract
from teadata.teadata_config import Config, load_config

FAMILY = "college_readiness_components"
METRIC = "ccmr_annual_graduates_dual_credit_all_students_rate"
PERIOD = {"kind": "school_year", "start_year": 2023, "end_year": 2024}
STAAR_METRIC = "staar_performance_reading_language_arts_grade_3_all_students_percent_at_meets_gl_standard_or_above"
ALL_GRADES_METRIC = "staar_performance_all_subjects_all_students_percent_at_meets_gl_standard_or_above"
EOC_METRIC = "staar_performance_eoc_algebra_i_all_students_percent_at_meets_gl_standard_or_above"


@pytest.fixture
def comparison_release(tmp_path, monkeypatch, request):
    selection = getattr(request, "param", FAMILY)
    fiscal = selection == "student_profile_fiscal"
    family, year = selection if isinstance(selection, tuple) else ("student_profile" if fiscal else selection, 2025)
    staar = family == "staar_tested_grade_3"
    store = ProvenanceStore(tmp_path / "source-store")
    masking = tmp_path / "masking.txt"
    masking.write_text("Synthetic reviewed rate masking methodology")
    masking_id = store.record_acquisition(masking, ResolvedDataset("reference:masking", "reference", year,
                                          year, str(masking), DatasetMetadata()))
    definitions = {}
    scopes = [
        ("district", [["227901", "35.6"], ["001901", "0"], ["001902", "-1"], ["001903", ""]]),
        ("campus", [["057909002", "33.8"]]),
        ("state", [["25.1"]]),
    ]
    if staar:
        scopes.append(("region", [["13", "35.6"], ["03", "-1"], ["01", "0"], ["02", ""], ["04", "-3"]]))
    for scope, observations in scopes:
        dataset = f"{scope}_tapr_{family}"
        path = tmp_path / f"{scope}.csv"
        key = f"{scope}_number" if scope != "state" else None
        labels, codes, fields = (["ID"], [scope.upper()], [key]) if key else ([], [], [])
        labels += ["Dual credit rate"]
        codes += ["RATE"]
        fields += ["dual_credit_rate"]
        metrics = {METRIC: {"field": "dual_credit_rate", "unit": "percent"}}
        table_period = PERIOD
        if staar:
            table_period = {"kind": "school_year", "start_year": year - 1, "end_year": year}
            rate_label = ("Domain 1A: Meets Grade Level STD, Grade 3, All Students, STAAR Reading/ELA Rate"
                          if year == 2024 else
                          "STAAR Performance, Reading/Language Arts (Grade 3), SY 2024-25, All Students, % At Meets GL Standard or Above")
            rate_field = ("domain_1a_meets_grade_level_std_grade_3_all_students_staar_reading_ela_rate"
                          if year == 2024 else STAAR_METRIC)
            labels[-1] = (f"{scope.title()} {year} " if year == 2024 else "") + rate_label
            codes[-1] = scope[0].upper() + f"DA03ARE12{str(year)[-2:]}R"
            fields[-1] = f"{scope}_{year}_{rate_field}"
            labels += ["Number at criterion", "Number of tests"]
            codes += ["NUMERATOR", "DENOMINATOR"]
            fields += ["raw_numerator", "raw_denominator"]
            observations = [[*row[:-1], row[-1].split(".")[0], "-3", "-3"] for row in observations]
            metrics = {}
        if family == "student_profile":
            table_period = {"kind": "unknown"}
            metrics[METRIC]["period"] = PERIOD
            labels += ["Membership count"]
            codes += ["COUNT"]
            fields += ["membership_count"]
            observations = [row + ["1200" if scope == "state" else "105"] for row in observations]
            metrics["membership_count"] = {
                "field": "membership_count", "unit": "count",
                "period": {"kind": "fiscal_year" if fiscal else "school_year", "start_year": 2024, "end_year": 2025},
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
                        "publisher_type": "Number" if staar and field != key else None,
                        "publisher_description": label if staar and field != key else None,
                        "dictionary_status": "not_applicable" if field == key else "documented" if staar else "not_documented"}
                       for i, (code, label, field) in enumerate(zip(codes, labels, fields, strict=True), 1)],
        }
        schema = tmp_path / f"{scope}.json"
        schema.write_text(canonical_json({"schema_version": 2, "mapping_version": 1, "datasets": [
            {"dataset": dataset, "edition": str(year), "source_only": True, "tables": [mapping]}]}) + "\n")
        original = store.record_acquisition(path, ResolvedDataset(dataset, "data_sources", year, year,
                                             str(path), DatasetMetadata(version=str(year))))
        headers = store.record_acquisition(schema, ResolvedDataset(dataset + ":schema", "reference", year, year,
                                            str(schema), DatasetMetadata()))
        definitions[dataset] = {year: {"path": str(path), "metadata": {
            "version": str(year), "period": table_period, "provenance_ids": [original],
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
    if staar:
        binding = {"rates": {STAAR_METRIC: {"code": f"DA03ARE12{str(year)[-2:]}R",
                   "label": rate_label, "field": rate_field}}, "scopes": {}}
        for scope, _ in scopes:
            dataset = f"{scope}_tapr_{family}"
            source = cfg.resolve_dataset(dataset, year, exact=True, reader_profile="source")
            source_request, = cfg.tapr_source_requests(dataset, year)
            binding["scopes"][scope] = {
                "label_prefix": f"{scope.title()} {year} " if year == 2024 else "",
                "raw_sha256": source_request.mapping["input_sha256"],
                "mapping_sha256": sha256(canonical_json(source_request.mapping).encode()).hexdigest(),
                "metadata_sha256": [sha256(canonical_json(source.metadata.to_dict()).encode()).hexdigest()],
            }
        rules = {"-1": {"status": "suppressed", "reason": "Synthetic reviewed publisher mask"}}
        monkeypatch.setattr("teadata.staar_comparison._bindings", lambda: {
            "notes": [], "editions": {str(year): {
                "period": table_period, "rate_tokens": rules, "families": {family: binding},
                "methodology": [{"url": "https://example.org/synthetic-mask", "provenance_id": masking_id,
                                 "sha256": sha256(masking.read_bytes()).hexdigest()}]}}})
        monkeypatch.setattr("teadata.staar_comparison.load_config", lambda: cfg)
        _reviewed_staar_source.cache_clear()
    prepared = tmp_path / "prepared"
    result = prepare_profiles(cfg, edition=str(year), family=family, scopes=[scope for scope, _ in scopes],
                              input_dir=None, output_dir=prepared, reference_store=store)
    selected = {entry["dataset"]: (entry["processing_record_id"], () if scope == "state" else (f"{scope}_number",))
                for scope, entry in result["scopes"].items()}
    output = tmp_path / "cohort"
    built = prepare_profiles(cfg, edition=str(year), scopes=[], input_dir=None, output_dir=output,
                             reference_store=ProvenanceStore(prepared / "provenance"), prepared_inputs=selected)
    release = DataRelease(output / "release.json", offline=True)
    editions = {entry["key"]["scope"]: EditionKey.from_dict(entry["key"]) for entry in built["datasets"].values()}
    assert built["datasets"][editions["state"].dataset]["key_columns"] == []
    assert len(release.verify()) == len(scopes)
    monkeypatch.setattr("teadata.statewide.load_config", lambda: cfg)
    _reviewed_tapr.cache_clear()
    yield release, editions
    _reviewed_tapr.cache_clear()
    _reviewed_staar_source.cache_clear()


def test_statewide_golden(comparison_release):
    from teadata.engine import DataEngine
    from uuid import UUID

    release, editions = comparison_release
    engine = DataEngine()
    for number in (227901, 1901, 1902, 1903, 999999):
        engine.add_district(District(UUID(int=number), str(number), district_number=f"{number:06}"))
    districts = engine >> (lambda item: isinstance(item, District))
    rows = districts >> ("compare_statewide", release, editions["district"], editions["state"], METRIC, None, 24)
    assert [(row.local_raw, row.local_status, row.statewide_percent, row.difference_pp) for row in rows] == [
        ("35.6", "observed", 25.1, 10.5),
        ("0", "observed", 25.1, -25.1),
        ("-1", "suppressed", 25.1, None),
        ("", "missing", 25.1, None),
        (None, "unavailable", 25.1, None),
    ]
    with pytest.raises(ContractError, match="reviewed local grain and statewide singleton"):
        districts >> ("compare_statewide", release, editions["district"], editions["state"], METRIC, {"grade": "12"})


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


@pytest.mark.parametrize("comparison_release", ["student_profile"], indirect=True)
def test_exact_comparison_year_projects_only_matching_vintage(comparison_release, monkeypatch):
    release, editions = comparison_release
    district = District(uuid4(), "Observed", district_number="227901")
    query = Query([district], repo=None)
    local, state = editions["district"], editions["state"]
    open_leaf = release.analytical_dataset
    requested = []

    class ObserveColumns:
        def __init__(self, dataset):
            self.dataset = dataset

        def head(self, count, *, columns, filter):
            requested.append(columns)
            return self.dataset.head(count, columns=columns, filter=filter)

        def to_table(self, *, columns, filter):
            requested.append(columns)
            return self.dataset.to_table(columns=columns, filter=filter)

    monkeypatch.setattr(release, "analytical_dataset", lambda identity: ObserveColumns(open_leaf(identity)))
    count, = query >> ("compare_statewide", release, local, state, "*", {}, 25)
    assert (count.metric, count.period.end_year, count.local_value, count.statewide_value) == (
        "membership_count", 2025, 105, 1200)
    assert len(requested) == 2
    assert all("membership_count" in columns and "dual_credit_rate" not in columns for columns in requested)
    def unexpected_leaf(_identity):
        raise AssertionError("A rejected school year opened an observation leaf")

    monkeypatch.setattr(release, "analytical_dataset", unexpected_leaf)
    with pytest.raises(ContractError, match=r"covers 2023-2024 .*Requested school year ends in 2025"):
        query.compare_statewide(release=release, local=local, statewide=state, metric=METRIC, school_year=2025)
    with pytest.raises(ContractError, match="Available school-year end years: 2024, 2025"):
        query.compare_statewide(release=release, local=local, statewide=state, school_year=2026)
    for invalid in (False, "2024-26", 2025.0, -1, 10_000):
        with pytest.raises(ValueError, match="Invalid school_year.*2024-25"):
            query.compare_statewide(release=release, local=local, statewide=state, school_year=invalid)

    args = {"release": release, "local": local, "statewide": state, "school_year": "2024–25"}
    for argument, value, error, message in (
        ("local", "district", TypeError, "EditionKey.*edition_id removed"),
        ("statewide", {}, TypeError, "EditionKey"),
        ("release", "release.json", TypeError, "DataRelease"),
        ("metric", [], ValueError, "reviewed metric name"),
        ("metric", "typo", ContractError, "Examples of common metric names"),
        ("dimensions", {"course_code": 3100500}, ContractError, "literal publisher strings"),
    ):
        with pytest.raises(error, match=message):
            query.compare_statewide(**(args | {argument: value}))


@pytest.mark.parametrize("comparison_release", ["student_profile_fiscal"], indirect=True)
def test_comparison_does_not_treat_a_fiscal_year_as_a_school_year(comparison_release, monkeypatch):
    release, editions = comparison_release
    district = District(uuid4(), "Observed", district_number="227901")
    query = Query([district], repo=None)
    local, state = editions["district"], editions["state"]
    metric = "membership_count"
    fiscal = DataPeriod("fiscal_year", 2024, 2025)
    row, = query >> ("compare_statewide", release, local, state, metric, None, None, fiscal)
    assert row.period == fiscal and row.metric == metric

    def unexpected_leaf(_identity):
        raise AssertionError("A fiscal-year mismatch opened an observation leaf")

    monkeypatch.setattr(release, "analytical_dataset", unexpected_leaf)
    with pytest.raises(ValueError, match="Choose school_year or period"):
        query >> ("compare_statewide", release, local, state, metric, None, 2025, fiscal)
    with pytest.raises(TypeError, match="period must be a DataPeriod"):
        query.compare_statewide(release=release, local=local, statewide=state, period="2024-25")
    with pytest.raises(ContractError, match=r"Available periods: 2024-2025 \(fiscal_year\)"):
        query.compare_statewide(release=release, local=local, statewide=state, metric=metric,
                                period=DataPeriod("school_year", 2024, 2025))
    with pytest.raises(ContractError, match=r"covers 2024-2025 \(fiscal_year\)"):
        Query([district], repo=None).compare_statewide(
            release=release, local=editions["district"], statewide=editions["state"],
            metric="membership_count", school_year=2025)


def test_campus_course_geography_hold_precedes_any_leaf_or_catalog_read():
    campus = Campus(uuid4(), None, "School", "", False, campus_number="057909002")
    local = EditionKey(COURSES, "2025", "1" * 64, "campus", "campus", "reported campus course")
    statewide = EditionKey(COURSES, "2025", "2" * 64, "state", "state", "reported state course")
    with pytest.raises(ContractError, match="geography qualification"):
        Query([campus], repo=None).compare_statewide(
            release=None, local=local, statewide=statewide, dimensions={"course_code": "03100500"})


@pytest.mark.parametrize("comparison_release", ["staar_tested_grade_3", ("staar_tested_grade_3", 2024)], indirect=True)
def test_staar_published_rates_keep_masked_counts_and_literal_region_ids(comparison_release):
    release, editions = comparison_release
    rows = list(Query(["13", "03", "01", "02", "04", "20"], repo=None).compare_statewide(
        release=release, local=editions["region"], statewide=editions["state"], metric=STAAR_METRIC))
    assert [(row.identifier, row.local_raw, row.local_status, row.difference_pp) for row in rows] == [
        ("13", "35", "observed", 10), ("03", "-1", "suppressed", None),
        ("01", "0", "observed", -25), ("02", "", "missing", None),
        ("04", "-3", "uninterpreted", None), ("20", None, "unavailable", None),
    ]
    year = int(editions["region"].edition)
    assert all(row.scope == "region" and row.unit == "percent" and row.period.end_year == year for row in rows)
    district = District(uuid4(), "Austin", district_number="227901")
    row = Query([district], repo=None).compare_statewide(
        release=release, local=editions["district"], statewide=editions["state"], metric=STAAR_METRIC).first()
    assert (row.local_value, row.statewide_value, row.difference_pp) == (35, 25, 10)
    local, = release.catalog.resolve_artifacts([(editions["district"], "raw_numerator")])
    original = release.analytical_dataset(local["artifact"].identity).to_table(columns=["raw_numerator", "raw_denominator"])
    assert original.to_pylist() == [{"raw_numerator": "-3", "raw_denominator": "-3"}] * 4
    contract = staar_rate_contract(release, editions["region"])
    expected_field = ("domain_1a_meets_grade_level_std_grade_3_all_students_staar_reading_ela_rate"
                      if year == 2024 else STAAR_METRIC)
    assert contract.metadata.metrics[STAAR_METRIC].field == f"region_{year}_{expected_field}"
    legend, = [ref for ref in contract.metadata.references if ref.reference_id == contract.value_policy.reference_id]
    assert legend.provenance_ids and legend.reviewed_sha256 and legend.url
    assert contract.value_policy.tokens["-1"].status.value == "suppressed"


@pytest.mark.parametrize("comparison_release", ["staar_tested_grade_3"], indirect=True)
def test_staar_only_interprets_reviewed_integer_rate_spellings(comparison_release):
    from teadata.statewide import _compare_source

    release, editions = comparison_release
    statistics, local, _ = _compare_source(release, editions["region"], editions["state"], STAAR_METRIC, {})
    field = statistics[0].local_field
    for raw in ("-1.0", "35.6", ".", "101", "-3", " 35", "35.0", "035", "３５"):
        assert local.value({field: raw}, field, "percent") == (raw, "uninterpreted", None)
    for raw, value in (("0", 0), ("35", 35), ("100", 100)):
        assert local.value({field: raw}, field, "percent") == (raw, "observed", value)
    with pytest.raises(ContractError, match="reviewed numeric"):
        Query(["13"], repo=None).compare_statewide(
            release=release, local=editions["region"], statewide=editions["state"], metric="raw_numerator")


@pytest.mark.parametrize("comparison_release", ["staar_tested_grade_3"], indirect=True)
def test_staar_comparison_rejects_changed_source_period_and_metadata_before_reads(comparison_release, monkeypatch):
    release, editions = comparison_release
    metadata = release.catalog.metadata
    monkeypatch.setattr(release, "analytical_dataset", lambda *a: pytest.fail("Unreviewed source opened a leaf"))

    def changed(dataset, *, edition):
        value = metadata(dataset, edition=edition)
        return replace(value, notes="Changed source binding")

    monkeypatch.setattr(release.catalog, "metadata", changed)
    with pytest.raises(ContractError, match="source references and metadata"):
        Query(["13"], repo=None).compare_statewide(
            release=release, local=editions["region"], statewide=editions["state"], metric=STAAR_METRIC)


@pytest.mark.parametrize("identifier", ["3", " 03", "03 ", "00", "21", "０３", "03\n", 3, True])
def test_region_comparison_requires_an_exact_publisher_literal_before_reads(identifier):
    local = EditionKey("region_tapr_staar_tested_grade_3", "2025", "1" * 64,
                       "region", "staar_tested_grade_3", "one row per region_number")
    statewide = EditionKey("state_tapr_staar_tested_grade_3", "2025", "2" * 64,
                           "state", "staar_tested_grade_3", "one statewide observation")
    with pytest.raises(ContractError, match="literal education service center"):
        Query([identifier], repo=None).compare_statewide(release=None, local=local, statewide=statewide)


@pytest.mark.parametrize("year", [2024, 2025, 2026])
def test_packaged_staar_comparison_metadata_discovers_exact_scope_neutral_rates(year, monkeypatch):
    cfg = load_config()
    store = ProvenanceStore(Path(__file__).resolve().parents[1] / "provenance")
    monkeypatch.setattr(store, "verify_artifact", lambda *a, **k: pytest.fail("Metadata lookup opened raw bytes"))
    _reviewed_staar_source.cache_clear()
    total = 0
    for grade, expected in ((3, 84), (4, 84), (5, 126), (6, 84), (7, 84), (8, 168)):
        common = None
        for scope in ("state", "district", "campus", "region"):
            dataset = f"{scope}_tapr_staar_tested_grade_{grade}"
            request, = cfg.tapr_source_requests(dataset, year)
            key = EditionKey(dataset, str(year), EditionKey.source_revision((request.mapping["input_sha256"],)),
                             scope, f"staar_tested_grade_{grade}",
                             "one statewide observation" if scope == "state" else f"one row per {scope}_number")
            release = SimpleNamespace(catalog=SimpleNamespace(metadata=lambda name, *, edition:
                cfg.resolve_dataset(name, year, exact=True, reader_profile="source").metadata))
            contract = staar_rate_contract(release, key)
            metrics = contract.metadata.metrics
            assert len(metrics) == expected and all(metric.unit == "percent" and metric.period.end_year == year
                                                     for metric in metrics.values())
            assert {metric.field for metric in metrics.values()} <= {field["field"] for field in request.mapping["fields"]}
            if common is not None:
                assert set(metrics) == common
            common = set(metrics)
            legend = next(ref for ref in contract.metadata.references if ref.reference_id == "staar_rate_legend")
            record = store.read_record(legend.provenance_ids[0])
            assert record["artifact"]["sha256"] == legend.reviewed_sha256
            assert contract.value_policy.reference_id == legend.reference_id
            assert set(contract.value_policy.tokens) == {"-1"}
        total += expected
    assert total == 630
    _reviewed_staar_source.cache_clear()


@pytest.mark.parametrize(("year", "family", "metric", "source_field"), [
    (2025, "staar_all_grades", ALL_GRADES_METRIC, ALL_GRADES_METRIC),
    (2026, "staar_all_grades", ALL_GRADES_METRIC, ALL_GRADES_METRIC),
    (2024, "staar_end_of_course", EOC_METRIC, "domain_1a_meets_grade_level_std_all_students_eoc_algebra_i_rate"),
    (2025, "staar_end_of_course", EOC_METRIC, "eoc_algebra_i_all_students_percent_at_meets_gl_standard_or_above"),
    (2026, "staar_end_of_course", EOC_METRIC, EOC_METRIC),
])
def test_packaged_aggregate_staar_contract_uses_exact_scope_and_source(year, family, metric, source_field):
    cfg = load_config()
    release = SimpleNamespace(catalog=SimpleNamespace(metadata=lambda name, *, edition:
        cfg.resolve_dataset(name, year, exact=True, reader_profile="source").metadata))
    names = None
    for scope in ("state", "district", "campus", "region"):
        dataset = f"{scope}_tapr_{family}"
        source, = cfg.tapr_source_requests(dataset, year)
        key = EditionKey(dataset, str(year), EditionKey.source_revision((source.mapping["input_sha256"],)),
                         scope, family, "one statewide observation" if scope == "state" else f"one row per {scope}_number")
        contract = staar_rate_contract(release, key)
        metrics = contract.metadata.metrics
        assert len(metrics) == 210 and all("sat_act" not in name for name in metrics)
        assert names is None or names == set(metrics)
        assert metrics[metric].field == f"{scope}_{year}_{source_field}"
        assert metrics[metric].period.end_year == year
        assert contract.value_policy.tokens["-1"].status.value == "suppressed"
        if year < 2026 and family == "staar_end_of_course":
            legend, = (ref for ref in contract.metadata.references if ref.reference_id == contract.value_policy.reference_id)
            assert legend.url == f"https://rptsvr1.tea.texas.gov/perfreport/tapr/{year}/masking.html"
        with pytest.raises(ContractError, match="reviewed publisher revision"):
            staar_rate_contract(release, replace(key, revision_sha256="a" * 64))
        with pytest.raises(ContractError, match="No reviewed STAAR comparison"):
            staar_rate_contract(release, replace(key, edition="2023" if family == "staar_end_of_course" else "2024"))
        names = set(metrics)


@pytest.mark.parametrize("change", ["source_field", "scope_prefix"])
@pytest.mark.parametrize("comparison_release", [("staar_tested_grade_3", 2024)], indirect=True)
def test_2024_semantic_alias_requires_exact_source_field_and_scope_heading(comparison_release, monkeypatch, change):
    from teadata import staar_comparison

    release, editions = comparison_release
    bindings = deepcopy(staar_comparison._bindings())
    family = bindings["editions"]["2024"]["families"]["staar_tested_grade_3"]
    if change == "source_field":
        family["rates"][STAAR_METRIC]["field"] = STAAR_METRIC
    else:
        family["scopes"]["region"]["label_prefix"] = "District 2024 "
    monkeypatch.setattr("teadata.staar_comparison._bindings", lambda: bindings)
    monkeypatch.setattr(release, "analytical_dataset", lambda *a: pytest.fail("Incorrect alias opened observations"))
    _reviewed_staar_source.cache_clear()
    with pytest.raises(ContractError, match="publisher rate label and code"):
        Query(["13"], repo=None).compare_statewide(
            release=release, local=editions["region"], statewide=editions["state"], metric=STAAR_METRIC)


@pytest.mark.parametrize("change", ["raw", "label", "period"])
def test_staar_rate_bindings_reject_changed_packaged_source_before_catalog_reads(change, monkeypatch):
    cfg = load_config()
    request, = cfg.tapr_source_requests("district_tapr_staar_tested_grade_3", 2025)
    mapping = deepcopy(request.mapping)
    if change == "raw":
        mapping["input_sha256"] = "0" * 64
    elif change == "label":
        mapping["fields"][5]["label"] += " different population"
    else:
        mapping["metric_period"] = PERIOD
    monkeypatch.setattr(cfg, "tapr_source_requests", lambda *a, **k: (replace(request, mapping=mapping),))
    monkeypatch.setattr("teadata.staar_comparison.load_config", lambda: cfg)
    _reviewed_staar_source.cache_clear()
    with pytest.raises(ContractError, match="publisher source, mapping and period"):
        _reviewed_staar_source("district", "staar_tested_grade_3", "2025")
    _reviewed_staar_source.cache_clear()


@pytest.mark.parametrize("duplicate", ["json_key", "field", "code"])
def test_staar_comparison_resource_rejects_ambiguous_bindings(duplicate):
    if duplicate == "json_key":
        body = b'{"schema_version":1,"schema_version":1}'
    else:
        document = deepcopy(_bindings())
        rates = document["editions"]["2025"]["families"]["staar_tested_grade_3"]["rates"]
        first, second = list(rates)[:2]
        if duplicate == "field":
            rates[second]["field"] = rates[first]["field"]
        else:
            rates[second]["code"] = rates[first]["code"]
        body = json.dumps(document).encode()
    with pytest.raises(ContractError, match="Duplicate|unique|binding"):
        _parse_bindings(body)
