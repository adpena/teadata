from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import pandas as pd
import pytest

from teadata import DataEngine
from teadata.dataset_metadata import MetricDefinition
from teadata.load_data import _add_reporting_entities
from teadata.manifest import ContractError, DatasetCatalog, canonical_json
from teadata.metric_bindings import parse_bindings
from teadata.values import ValueStatus, measure


def revised(catalog, change):
    original = catalog.to_dict()
    document = deepcopy(original)
    change(document)
    # Metadata revisions need a matching synthetic local observation, just as a
    # real revised contract does. No production provenance files are modified.
    for name, source in document["datasets"].items():
        metadata = source["metadata"]
        if metadata == original["datasets"][name]["metadata"]:
            continue
        previous = document["provenance"][metadata["provenance_ids"][-1]]
        receipt = deepcopy(previous)
        receipt.update(
            method="local_observation",
            downloaded_at=None,
            requested_url=None,
            resolved_url=None,
            response_headers={},
        )
        receipt["source_metadata"] = {**deepcopy(metadata), "provenance_ids": []}
        identifier = sha256((canonical_json(receipt) + "\n").encode()).hexdigest()
        document["provenance"][identifier] = receipt
        metadata["provenance_ids"] = [identifier]
    document.pop("build_id", None)
    document["build_id"] = sha256(canonical_json(document).encode()).hexdigest()
    return DatasetCatalog(document)


def entity(catalog, value, *, metric="enrollment", bound=False):
    _, field = catalog.source_for_metric(metric, "campus")
    result = SimpleNamespace(
        campus_number="'123456001", meta={field: value}, is_private=False
    )
    if bound:
        result.source_build_id = catalog.build_id
    return result


def test_v1_catalog_cannot_be_reinterpreted_by_v2_runtime(catalog):
    with pytest.raises(ContractError, match="Unsupported dataset catalog schema"):
        revised(catalog, lambda document: document.update(schema_version=1))


def test_catalog_requires_its_own_registry_and_has_no_runtime_defaults(catalog):
    with pytest.raises(ContractError, match="release-bound metric interpretation"):
        revised(catalog, lambda document: document.pop("metric_bindings"))


def test_registry_is_release_authority_and_defensive_copy(catalog):
    def change(document):
        document["metric_bindings"]["enrollment"] = {
            "dataset": "accountability",
            "field": "student_count",
            "field_mode": "metric",
            "unit": "count",
            "minimum": 0,
            "scopes": ["campus"],
        }

    different = revised(catalog, change)
    original_dataset, original_field = catalog.source_for_metric("enrollment", "campus")
    new_dataset, new_field = different.source_for_metric("enrollment", "campus")
    caller = SimpleNamespace(
        campus_number="123456001", meta={original_field: 11, new_field: 22}
    )
    assert original_dataset == "campus_tapr_student_staff_profile"
    assert new_dataset == "accountability"
    assert catalog.measurement(caller, "enrollment").numeric == 11
    assert different.measurement(caller, "enrollment").numeric == 22
    assert different.build_id != catalog.build_id
    copy = catalog.to_dict()
    copy["metric_bindings"]["enrollment"] = "rating"
    assert catalog.measurement(caller, "enrollment").numeric == 11


def test_aliases_and_wildcards_resolve_declared_release_fields(catalog):
    changed = revised(
        catalog,
        lambda document: document["metric_bindings"].update(
            {
                "students": "enrollment",
                "students_again": "students",
                "profile.*": "tapr.*",
            }
        ),
    )
    assert changed.source_for_metric(
        "students_again", "campus"
    ) == changed.source_for_metric("enrollment", "campus")
    assert changed.source_for_metric(
        "profile.student_enrollment_all_students_count", "campus"
    ) == changed.source_for_metric("enrollment", "campus")
    with pytest.raises(ContractError, match="not a declared field"):
        changed.source_for_metric("profile.nonexistent_source_field", "campus")
    with pytest.raises(ContractError, match="Unknown canonical metric"):
        changed.source_for_metric("unknown.no_field", "campus")


@pytest.mark.parametrize(
    "registry", [{"broken": "missing"}, {"first": "second", "second": "first"}]
)
def test_invalid_aliases_are_rejected(registry):
    with pytest.raises(ValueError, match="alias"):
        parse_bindings(registry)


def test_semantic_definition_cannot_invent_a_field_absent_from_release_reads(catalog):
    def change(document):
        document["datasets"]["campus_tapr_student_staff_profile"]["metadata"][
            "metrics"
        ]["ghost"] = {
            "field": "never_read_field",
            "unit": "count",
        }

    changed = revised(catalog, change)
    with pytest.raises(ContractError, match="not a declared field|not.*read"):
        changed.source_for_metric("tapr.ghost", "campus")


def test_foreign_release_entities_are_rejected_for_both_known_origin_forms(catalog):
    different = revised(
        catalog, lambda document: document.update(processor_sha256="f" * 64)
    )
    attached = entity(catalog, 12)
    attached._repo = SimpleNamespace(catalog=catalog)
    with pytest.raises(ContractError, match="another data release"):
        different.measurement(attached, "enrollment")
    persisted = entity(catalog, 12, bound=True)
    with pytest.raises(ContractError, match="another data release"):
        different.measurement(persisted, "enrollment")
    _, field = different.source_for_metric("enrollment", "campus")
    with pytest.raises(ContractError, match="another data release"):
        different.measure_field(12, field, entity=persisted)


def test_conflicting_explicit_and_attached_release_origins_fail_closed(catalog):
    different = revised(
        catalog, lambda document: document.update(processor_sha256="c" * 64)
    )
    conflicted = entity(catalog, 12, bound=True)
    conflicted._repo = SimpleNamespace(catalog=different)
    with pytest.raises(ContractError, match="another data release|conflicting|origin"):
        catalog.measurement(conflicted, "enrollment")


def test_foreign_release_or_field_bound_measurements_are_rejected(catalog):
    different = revised(
        catalog, lambda document: document.update(processor_sha256="e" * 64)
    )
    value = catalog.measurement(entity(catalog, 12), "enrollment")
    with pytest.raises(ContractError, match="another release or field"):
        different.measurement(entity(different, value), "enrollment")
    _, financial_field = catalog.source_for_metric(
        "financials.transportation_af_per_student", "campus"
    )
    with pytest.raises(ContractError, match="another release or field"):
        catalog.measure_field(value, financial_field)


def test_caller_supplied_values_have_explicit_origin_and_defensive_provenance(catalog):
    value = catalog.measurement(entity(catalog, 12), "enrollment")
    assert value.source["input_binding"] == "caller_supplied"
    assert value.source["build_id"] == catalog.build_id
    assert value.source["dataset"] == "campus_tapr_student_staff_profile"
    origin = value.source
    origin["build_id"] = "forged"
    assert value.to_dict()["source"]["build_id"] == catalog.build_id
    loaded = catalog.measurement(entity(catalog, 12, bound=True), "enrollment")
    assert loaded.source["input_binding"] == "release_entity"


@pytest.mark.parametrize(
    "raw,status", [(-1, ValueStatus.SUPPRESSED), (1.5, ValueStatus.INVALID)]
)
def test_unbound_interpreted_values_are_redecoded_under_the_release_policy(
    catalog, raw, status
):
    incoming = measure(raw)
    assert incoming.status == ValueStatus.OBSERVED
    result = catalog.measurement(entity(catalog, incoming, bound=True), "enrollment")
    assert result.status == status
    assert result.numeric is None
    assert (
        result.policy_id
        == catalog.policy("campus_tapr_student_staff_profile").policy_id
    )


def test_field_decoder_uses_source_suppression_without_masking_financial_negatives(
    catalog,
):
    _, enrollment_field = catalog.source_for_metric("enrollment", "campus")
    _, financial_field = catalog.source_for_metric(
        "financials.transportation_af_per_student", "campus"
    )
    suppressed = catalog.measure_field(-1, enrollment_field)
    negative = catalog.measure_field(-1, financial_field)
    assert suppressed.status == ValueStatus.SUPPRESSED
    assert suppressed.numeric is None
    assert suppressed.raw == -1
    assert negative.status == ValueStatus.OBSERVED
    assert negative.numeric == -1
    assert negative.unit == "currency"
    assert catalog.measure_field("123456001", "campus_number") is None
    assert catalog.measure_field(1, "not_a_release_field") is None


def test_field_decoder_honors_public_population_with_entity_context(catalog):
    _, field = catalog.source_for_metric("enrollment", "campus")
    private = SimpleNamespace(campus_number="P001", is_private=True, meta={field: 123})
    result = catalog.measure_field(123, field, entity=private)
    assert result.status == ValueStatus.NOT_APPLICABLE
    assert result.numeric is None
    assert result.reason == "outside_metric_population"


def test_status_frame_preserves_zero_missing_and_suppressed_raw_values(catalog):
    rows = [entity(catalog, raw) for raw in [0, None, -1]]
    frame = catalog.measurements_frame(rows, ["enrollment"])
    assert frame["status"].tolist() == ["observed", "missing", "suppressed"]
    assert frame.iloc[0]["value"] == 0
    assert pd.isna(frame.iloc[1]["value"]) and pd.isna(frame.iloc[2]["value"])
    assert frame.iloc[2]["raw"] == -1
    assert frame["input_binding"].tolist() == ["caller_supplied"] * 3
    assert set(frame["build_id"]) == {catalog.build_id}
    assert frame.attrs["teadata"]["build_id"] == catalog.build_id


def test_omitted_metric_period_inherits_but_explicit_unknown_does_not(catalog):
    dataset, _ = catalog.source_for_metric("enrollment", "campus")
    definition = catalog.metadata(dataset).metrics[
        "student_enrollment_all_students_count"
    ]
    assert definition.period is None
    assert catalog.metric_period("enrollment") == catalog.metadata(dataset).period

    def change(document):
        document["datasets"][dataset]["metadata"]["metrics"][
            "student_enrollment_all_students_count"
        ]["period"] = {"kind": "unknown"}

    unknown = revised(catalog, change)
    assert unknown.metric_period("enrollment").kind == "unknown"
    assert unknown.measurement(entity(unknown, 12), "enrollment").source["period"] == {
        "kind": "unknown"
    }
    assert (
        unknown.metric_label("enrollment", "Enrollment")
        == "Period not specified Enrollment"
    )
    assert MetricDefinition.from_dict({"field": "example"}).period is None


def test_same_field_semantic_metrics_keep_their_distinct_effective_periods(catalog):
    dataset, raw_field = catalog.source_for_metric("enrollment", "campus")

    def change(document):
        metadata = document["datasets"][dataset]["metadata"]
        for name, year in [("prior_definition", 2024), ("current_definition", 2025)]:
            metadata["metrics"][name] = {
                "field": raw_field,
                "unit": "count",
                "period": {"kind": "calendar_year", "year": year},
            }
            document["metric_bindings"][name] = {
                "dataset": dataset,
                "field": name,
                "field_mode": "metric",
                "unit": "count",
                "scopes": ["campus"],
            }

    changed = revised(catalog, change)
    caller = SimpleNamespace(campus_number="123456001", meta={raw_field: 12})
    prior = changed.measurement(caller, "prior_definition")
    current = changed.measurement(caller, "current_definition")
    assert prior.source["period"] == {"kind": "calendar_year", "year": 2024}
    assert current.source["period"] == {"kind": "calendar_year", "year": 2025}


@pytest.mark.parametrize("status", ["unreviewed", "not_found"])
def test_suppression_requires_available_exact_reviewed_methodology(catalog, status):
    def change(document):
        references = document["datasets"]["campus_tapr_student_staff_profile"][
            "metadata"
        ]["references"]
        next(
            reference
            for reference in references
            if reference["reference_id"] == "masking"
        )["status"] = status

    with pytest.raises(
        ContractError, match="review.*methodology|methodology.*review|exact methodology"
    ):
        revised(catalog, change)


def test_changed_methodology_bytes_cannot_keep_an_old_review(catalog):
    def change(document):
        references = document["datasets"]["campus_tapr_student_staff_profile"][
            "metadata"
        ]["references"]
        next(
            reference
            for reference in references
            if reference["reference_id"] == "masking"
        )["reviewed_sha256"] = "d" * 64

    with pytest.raises(ContractError, match="reviewed methodology"):
        revised(catalog, change)


def roster_row(**overrides):
    return {
        "district_number": "123456",
        "district_name": "Example ISD",
        "campus_number": "123456001",
        "campus_name": "Example Campus",
        "charter_status": "No",
        "accountability_student_count": 12,
        **overrides,
    }


@pytest.mark.parametrize("campus_id", ["bad", "123456001.5", 123456001.5, True, 0])
def test_malformed_campus_roster_identifiers_cannot_become_district_totals(campus_id):
    repo = DataEngine()
    with pytest.raises(ContractError, match="invalid source identifier"):
        _add_reporting_entities(
            repo, pd.DataFrame([roster_row(campus_number=campus_id)])
        )
    assert not repo.districts


@pytest.mark.parametrize("charter", ["No", "Yes"])
def test_true_district_roster_summary_keeps_population_separate_from_enrollment(
    charter,
):
    repo = DataEngine()
    _add_reporting_entities(
        repo,
        pd.DataFrame(
            [
                roster_row(
                    campus_number=None,
                    campus_name=None,
                    charter_status=charter,
                    school_type="District",
                )
            ]
        ),
    )
    district = next(iter(repo.districts.values()))
    assert district.enrollment is None
    assert district.meta["accountability_student_count"] == 12
    assert not repo.campuses


@pytest.mark.parametrize("charter", ["No", "Yes"])
def test_reporting_campus_remains_available_without_spatial_or_enrollment_facts(
    charter,
):
    repo = DataEngine()
    added = _add_reporting_entities(
        repo, pd.DataFrame([roster_row(charter_status=charter)])
    )
    campus = next(iter(repo.campuses.values()))
    assert added == {"districts": 1, "campuses": 1}
    assert campus.campus_number == "'123456001"
    assert campus.is_charter == (charter == "Yes")
    assert campus.enrollment is None and campus.location is None
    assert campus.meta["accountability_student_count"] == 12


@pytest.mark.parametrize("missing", [None, "", float("nan"), pd.NA])
def test_named_campus_without_identifier_is_not_a_district_summary(missing):
    repo = DataEngine()
    with pytest.raises(ContractError, match="campus.*identifier|identifier.*campus"):
        _add_reporting_entities(repo, pd.DataFrame([roster_row(campus_number=missing)]))
