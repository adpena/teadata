"""Typed source rules for the reviewed PEIMS Staff FTE and Salary reports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .values import ValuePolicy, ValueStatus

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pyarrow as pa

    from .dataset_metadata import DatasetMetadata


STAFF = "peims_staff_fte_and_salary"
STAFF_RECIPE = "reviewed_peims_staff_csv_projection"
STAFF_EMPLOYEE_COUNT = "staff_employee_count"
STAFF_EMPLOYEE_BASIS = "peims_employee_count_basis"
STAFF_AVERAGE_PAY = "staff_average_base_pay"
STAFF_SELECTOR = "https://rptsvr1.tea.texas.gov/adhocrpt/adpeb.html"
_BROKER = "https://rptsvr1.tea.texas.gov/cgi/sas/broker"
_ARCHIVE_DATASET = "reference:peims_staff_fte_salary_history"
_ARCHIVE_RECEIPTS = {
    (2013, "state"): "e997fe3ea92291b64decbd93ccec107d1723ab6be561f214af5f9c9c96de332b",
    (2014, "state"): "32d16870627c1ec6614607bad0386b451f5e7c713b2c40933a77437d15f86bc8",
    (2015, "state"): "f0bf9bb376b62877819a9452d456680a207db989fdcf2ffb6a376b74ed30b3ae",
    (2016, "state"): "9068ae4ef2155d773fc1266e5917114e041928d2048f970f7c1eba896261d2e7",
    (2017, "state"): "11f364dd32ebf58e8e22fcedbe910b3139614344f3ead1eef4888f06e22a2b85",
    (2018, "state"): "8df5f1f682fc7b8c7d96d8ca372dc185180e957fe04d44f191c47dfc74543e22",
    (2019, "district"): "7126cac0d8848963ec31bd737fd94abcb1149d10d2c7085c0a48e61b30446c08",
    (2019, "state"): "d091ac507808ed12dad33a5c811196ee8abc3d0fbefc6fdf31d567382ec3e0fa",
    (2020, "district"): "2deb3ef2c32f6c6182b0173dd77e860b36eea25b48f20a2c43538069f8b11afe",
    (2020, "state"): "d1059b1300a0245144c5baff06a242927b0f0df20d281c461e967ab430f26482",
    (2021, "district"): "efa7cad60617924cf68e77db8d8f8aa5254151f624942a67d5d8f687fafd93be",
    (2021, "state"): "420691bdf28065b33ef07622292481cd083f535272828bd21af1dd8bf9d90a73",
    (2022, "district"): "1bd210cc05bd976c1fb5e690123612ff8bd2f8da7e271c2c7e823ee357c3001d",
    (2022, "state"): "54b1d62ed8e5c93be3bfa7cabcaeddda118747eb6df7756642d885285c61a09a",
}
_EARLY_GUIDES = {
    2013: "471d4c3178b9be740b1e58f7bd35ca852fc6a00ccacda08aba00e2b658110bd2",
    2014: "ecc71e7670a1ab29534feccd21f8b2abeb24d6dfa2a3756cd21f3015ca5c8cb8",
    2015: "fa24ae1263b4e75d44c1f8a306925f562ef345f4574babbd6bf7dc0b12657eb9",
    2016: "8839774aa15cc5b6eb8d44612d49497ba93022d73ca84b05574620ce2f7201a1",
    2017: "c56e4c8b2009df1e51f511ab69baf6076722ee246df323e1a0ea03ab355dd998",
    2018: "ce33f30ef956cdcf641773ece98baa8b436a51353c010e3cd1ded75383def243",
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


def staff_scopes(year: int) -> frozenset[str]:
    """Only the direct Texas reports are qualified for the first six years."""
    return frozenset({"state"}) if year in _EARLY_GUIDES else frozenset({"state", "district"})


def validate_early_staff_guide(metadata: DatasetMetadata, year: int,
                               records: Mapping[str, Any]) -> None:
    if (identifier := _EARLY_GUIDES.get(year)) is None:
        return
    url = (f"{_BROKER}?_service=marykay&_program=adhoc.std_download_selected_report.sas"
           f"&rpt_subject=about&fname=abtstf{year % 100:02}&ftype=html")
    references = [item for item in metadata.references if item.reference_id == "staff_definition"]
    record = records.get(identifier, {})
    if (len(references) != 1 or references[0].role != "data_dictionary"
            or references[0].status != "available" or references[0].url != url
            or references[0].provenance_ids != (identifier,)
            or record.get("kind") != "acquisition" or record.get("method") != "download"
            or record.get("requested_url") != url or record.get("resolved_url") != url
            or (record.get("requested_year"), record.get("resolved_year")) != (year, year)
            or record.get("artifact", {}).get("sha256") != references[0].reviewed_sha256):
        raise ValueError("Early PEIMS staffing requires the exact same-year publisher guide")


def matches_staff_source(record: dict[str, Any], receipt_id: str, year: int, scope: str) -> bool:
    """Recognize a regular acquisition or an exact reviewed archival receipt."""
    identity = record["dataset"], record["section"], record["source_metadata"].get("source_url")
    if year not in _EARLY_GUIDES and identity == (STAFF, "data_sources", STAFF_SELECTOR):
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
