"""Typed source rules for the reviewed PEIMS Staff FTE and Salary reports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .values import ValuePolicy, ValueStatus

if TYPE_CHECKING:
    import pyarrow as pa


STAFF = "peims_staff_fte_and_salary"
STAFF_RECIPE = "reviewed_peims_staff_csv_projection"
STAFF_EMPLOYEE_COUNT = "staff_employee_count"
STAFF_EMPLOYEE_BASIS = "peims_employee_count_basis"
STAFF_AVERAGE_PAY = "staff_average_base_pay"
_BROKER = "https://rptsvr1.tea.texas.gov/cgi/sas/broker"
_FACT_UNITS = {"staff_fte": "fte", "staff_total_base_pay": "currency", STAFF_AVERAGE_PAY: "currency"}
_TOTAL_ROLES = (
    (" ", "TOTAL AUXILIARY STAFF"),
    (" ", "TOTAL PERSONNEL"),
    (" ", "TOTAL PROFESSIONAL STAFF"),
    ("ADMINISTRATIVE STAFF", "TOTAL ADMINISTRATIVE STAFF"),
    ("PARAPROFESSIONAL STAFF", "TOTAL PARAPROFESSIONAL STAFF"),
    ("SUPPORT STAFF", "TOTAL SUPPORT STAFF"),
    ("TEACHING STAFF", "TOTAL TEACHING STAFF"),
)


def staff_fact_units(year: int) -> dict[str, str]:
    return {**_FACT_UNITS, STAFF_EMPLOYEE_COUNT: "count"} if year == 2026 else _FACT_UNITS.copy()


def validate_staff_request(record: dict[str, Any], year: int, scope: str) -> None:
    """Require the actual TEA statewide or statewide-by-district CSV selection."""
    parameters = record["parameters"]
    controls = parameters.get("form_controls", [])
    key = dict(controls).get("key")
    selection = {"state": "ss", "district": "sd"}[scope]
    expected = [["_service", "marykay"], ["_program", "adhoc.addispatch.sas"], ["major", "pe"], ["minor", "b"],
                ["loop", "1"], ["linespg", "60"], ["charsln", "120"], ["endyear", str(year)[-2:]],
                ["selsumm", selection], ["key", key], ["format", "C"]]
    geography = "Statewide_Districts" if scope == "district" else "Statewide"
    filename = f"Staff Salary FTE Report_{geography}_{year - 1}-{year}.csv"
    if (key not in ("", "TYPE HERE") or controls != expected
            or parameters.get("http_method") != "POST" or parameters.get("http_status") != 200
            or parameters.get("form_url") != _BROKER
            or record["requested_url"] != _BROKER or record["resolved_url"] != _BROKER
            or record["response_headers"].get("content-disposition") != f'name="charlie";filename="{filename}"'):
        raise ValueError("PEIMS staffing receipt differs from the exact publisher scope or edition")


def staff_allows_unavailable_average(name: str, year: int, scope: str) -> bool:
    return name == STAFF_AVERAGE_PAY and scope == "district" and year in (2025, 2026)


def staff_values(raw: pa.ChunkedArray, policy: ValuePolicy, *, unit: str, allow_dot: bool) -> tuple[
    pa.ChunkedArray, pa.ChunkedArray
]:
    import pyarrow as pa
    import pyarrow.compute as arrow_compute

    pc: Any = arrow_compute
    if unit not in ("fte", "currency", "count"):
        raise ValueError("PEIMS staffing fact lacks its reviewed unit")
    observed = pc.match_substring_regex(raw, r"^[0-9]+\.[0-9]{2}$" if unit == "fte" else r"^[0-9]+$")
    status = ValueStatus.OBSERVED.value
    if allow_dot:
        rule = policy.tokens.get(".")
        if rule is None or rule.status != ValueStatus.MISSING:
            raise ValueError("PEIMS staffing average pay requires a reviewed unavailable status for '.'")
        accepted = pc.or_(observed, pc.equal(raw, "."))
    else:
        accepted = observed
    if pc.all(pc.fill_null(accepted, False)).as_py() is not True:
        raise ValueError("PEIMS staffing contains a token outside its reviewed unit or status")
    datatype = pa.decimal128(16, 2) if unit == "fte" else pa.int64()
    values = pc.cast(pc.if_else(observed, raw, pa.scalar(None, pa.string())), datatype)
    return values, pc.if_else(observed, status, ValueStatus.MISSING.value)


def staff_employee_basis(categories: pa.ChunkedArray, roles: pa.ChunkedArray) -> pa.ChunkedArray:
    import pyarrow as pa
    import pyarrow.compute as arrow_compute

    pc: Any = arrow_compute
    valid_totals = pa.array([f"{category}\t{role}" for category, role in _TOTAL_ROLES], type=pa.string())
    totals = pc.is_in(pc.binary_join_element_wise(categories, roles, "\t"), value_set=valid_totals)
    if pc.all(pc.fill_null(pc.equal(totals, pc.starts_with(roles, "TOTAL ")), False)).as_py() is not True:
        raise ValueError("PEIMS staffing contains an unreviewed employee-count total role")
    return pc.if_else(totals, "distinct_employees", "positions")
