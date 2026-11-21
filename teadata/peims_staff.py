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
STAFF_SELECTOR = "https://rptsvr1.tea.texas.gov/adhocrpt/adpeb.html"
_BROKER = "https://rptsvr1.tea.texas.gov/cgi/sas/broker"
_ARCHIVE_DATASET = "reference:peims_staff_fte_salary_history"
_ARCHIVE_RECEIPTS = {
    (2019, "district"): "7126cac0d8848963ec31bd737fd94abcb1149d10d2c7085c0a48e61b30446c08",
    (2019, "state"): "d091ac507808ed12dad33a5c811196ee8abc3d0fbefc6fdf31d567382ec3e0fa",
    (2020, "district"): "2deb3ef2c32f6c6182b0173dd77e860b36eea25b48f20a2c43538069f8b11afe",
    (2020, "state"): "d1059b1300a0245144c5baff06a242927b0f0df20d281c461e967ab430f26482",
    (2021, "district"): "efa7cad60617924cf68e77db8d8f8aa5254151f624942a67d5d8f687fafd93be",
    (2021, "state"): "420691bdf28065b33ef07622292481cd083f535272828bd21af1dd8bf9d90a73",
    (2022, "district"): "1bd210cc05bd976c1fb5e690123612ff8bd2f8da7e271c2c7e823ee357c3001d",
    (2022, "state"): "54b1d62ed8e5c93be3bfa7cabcaeddda118747eb6df7756642d885285c61a09a",
}
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


def matches_staff_source(record: dict[str, Any], receipt_id: str, year: int, scope: str) -> bool:
    """Recognize a regular acquisition or an exact reviewed archival receipt."""
    identity = record["dataset"], record["section"], record["source_metadata"].get("source_url")
    if identity == (STAFF, "data_sources", STAFF_SELECTOR):
        return True
    return (_ARCHIVE_RECEIPTS.get((year, scope)) == receipt_id
            and identity == (_ARCHIVE_DATASET, "reference", _BROKER))


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
