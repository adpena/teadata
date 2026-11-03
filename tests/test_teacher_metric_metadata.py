"""Reviewed teacher units and periods use the release's canonical metadata."""

from types import SimpleNamespace

import pytest

from teadata.manifest import ContractError
from teadata.teadata_config import load_config
from teadata.values import ValueStatus


# Raw values are synthetic and do not claim observations for a real district.
TEACHER_CASES = [
    ("campus", "staff_teacher_total_full_time_equiv_count", "CPSTTOFC", "fte", "1.5", 1.5),
    ("campus", "staff_teacher_experience_average", "CPSTEXPA", "years", "12.5", 12.5),
    ("campus", "staff_teacher_total_base_salary_average", "CPSTTOSA", "currency", "60000.25", 60000.25),
    ("campus", "staff_teacher_student_ratio", "CPSTKIDR", "ratio", "14.2", 14.2),
    ("district", "staff_teacher_total_full_time_equiv_count", "DPSTTOFC", "fte", "1.5", 1.5),
    ("district", "staff_teacher_experience_average", "DPSTEXPA", "years", "12.5", 12.5),
    ("district", "staff_teacher_total_base_salary_average", "DPSTTOSA", "currency", "60000.25", 60000.25),
    ("district", "staff_teacher_student_ratio", "DPSTKIDR", "ratio", "14.2", 14.2),
    ("district", "staff_teacher_turnover_ratio", "DPSTURNR", "percent", "17.9", 17.9),
]


@pytest.mark.parametrize("scope,name,code,unit,raw,expected", TEACHER_CASES)
def test_reviewed_teacher_metadata_controls_scalar_and_raw_field_decoders(
    catalog, scope, name, code, unit, raw, expected
):
    metric = "tapr." + name
    dataset, field = catalog.source_for_metric(metric, scope)
    assert dataset == f"{scope}_tapr_student_staff_profile"
    assert field == f"{scope}_2025_{name}"
    metadata = catalog.metric_metadata(metric, scope=scope)
    assert metadata["unit"] == unit
    assert metadata["population"] == "public"
    assert metadata["period"] == {
        "kind": "school_year", "start_year": 2024, "end_year": 2025
    }
    # This synthetic catalog fixture carries declared read fields rather than
    # packaged header resources. Check the actual source contract separately.
    source_config = load_config("teadata/teadata_sources.yaml")
    header, = [row for row in source_config.header_records(dataset, edition="2025", table="staff")
               if row["field"] == field]
    assert header["code"] == code
    source = SimpleNamespace(meta={field: raw}, is_private=False)
    value = catalog.measurement(source, metric, scope=scope)
    assert value.status == ValueStatus.OBSERVED
    assert value.numeric == expected and value.raw == raw and value.unit == unit
    assert value.source["period"] == metadata["period"]
    decoded = catalog.measure_field(raw, field, scope=scope, entity=source)
    assert decoded is not None
    assert (decoded.value, decoded.status, decoded.raw, decoded.unit) == (
        value.value, value.status, value.raw, value.unit
    )
    assert decoded.source["period"] == metadata["period"]
    assert source.meta[field] == raw


def test_teacher_turnover_is_percent_bounded_and_district_only(catalog):
    metric = "tapr.staff_teacher_turnover_ratio"
    _, field = catalog.source_for_metric(metric, "district")
    source = SimpleNamespace(meta={field: "100.1"}, is_private=False)
    result = catalog.measurement(source, metric, scope="district")
    assert result.status == ValueStatus.INVALID and result.numeric is None
    assert result.raw == "100.1" and result.unit == "percent"
    with pytest.raises(ContractError, match="not a declared field"):
        catalog.source_for_metric(metric, "campus")
