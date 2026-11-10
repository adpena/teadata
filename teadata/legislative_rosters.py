"""Read the reviewed 2025–26 TLC legislative school district rosters offline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
import json
from pathlib import Path
import re

from .dataset_metadata import DataPeriod
from .identifiers import stored_district_number
from .provenance import ProvenanceStore


class LegislativePlan(StrEnum):
    TEXAS_HOUSE = "PLANH2316"
    TEXAS_SENATE = "PLANS2168"
    US_HOUSE_C2193 = "PLANC2193"
    US_HOUSE_C2333 = "PLANC2333"
    STATE_BOARD_OF_EDUCATION = "PLANE2106"


_C2193_START = datetime(2023, 1, 3, 17, tzinfo=timezone.utc)
CONGRESSIONAL_REPRESENTATION_CHANGE = datetime(2027, 1, 3, 17, tzinfo=timezone.utc)


def congressional_representation_plan(at: datetime) -> LegislativePlan:
    """Select from the reviewed Texas congressional plans by representation time."""
    if not isinstance(at, datetime) or at.utcoffset() is None:
        raise ValueError("Congressional representation requires a timezone-aware datetime")
    instant = at.astimezone(timezone.utc)
    if instant < _C2193_START:
        raise ValueError("No reviewed Texas congressional representation plan for this date")
    return (LegislativePlan.US_HOUSE_C2193 if instant < CONGRESSIONAL_REPRESENTATION_CHANGE
            else LegislativePlan.US_HOUSE_C2333)


_PLANS = {
    LegislativePlan.TEXAS_HOUSE: ("HOUSE", 150),
    LegislativePlan.TEXAS_SENATE: ("SENATE", 31),
    LegislativePlan.US_HOUSE_C2193: ("CONGRESSIONAL", 38),
    LegislativePlan.US_HOUSE_C2333: ("CONGRESSIONAL", 38),
    LegislativePlan.STATE_BOARD_OF_EDUCATION: ("SBOE", 15),
}
_PERIOD = DataPeriod("school_year", 2025, 2026)
_TLC = "https://data.capitol.texas.gov"
_RED160 = "reference:tlc_r160_school_districts"
_LOCATOR = "https://services2.arcgis.com/5MVN2jsqIrNZD4tP/arcgis/rest/services/School_Districts__2026/FeatureServer/0/query"
_DISTRICT = re.compile(r"DISTRICT ([1-9][0-9]*)")
_SCHOOL = re.compile(r"(.+) \((?:100|[0-9]{1,2})%\)")
_MAX_ROWS = 5000


@dataclass(frozen=True, slots=True)
class RosterSource:
    receipt: str
    sha256: str


@dataclass(frozen=True, slots=True)
class LegislativeDistrictSchool:
    legislative_district_number: int
    school_district_number: str
    source_label: str
    worksheet_row: int


@dataclass(frozen=True, slots=True)
class LegislativeSchoolRoster:
    plan: LegislativePlan
    school_boundary_period: DataPeriod
    census_year: int
    workbook: RosterSource
    district_locator: RosterSource
    rows: tuple[LegislativeDistrictSchool, ...]

    def district_ids(self, legislative_district_number: int) -> tuple[str, ...]:
        _, count = _PLANS[self.plan]
        if type(legislative_district_number) is not int or not 1 <= legislative_district_number <= count:
            raise ValueError(f"Choose a {self.plan} district number from 1 to {count}")
        return tuple(row.school_district_number for row in self.rows
                     if row.legislative_district_number == legislative_district_number)


def _download(record: Mapping, dataset: str, publisher: str, version: str, limit: int) -> bool:
    if record["kind"] != "acquisition":
        return False
    metadata = record["source_metadata"]
    parameters = record["parameters"]
    return (record["method"] == "download"
            and record["downloaded_at"] is not None and record["dataset"] == dataset
            and record["requested_year"] == record["resolved_year"] == _PERIOD.end_year
            and metadata["publisher"] == publisher and metadata["version"] == version
            and DataPeriod.from_dict(metadata["period"]) == _PERIOD
            and isinstance(record["requested_url"], str) and record["requested_url"] == record["resolved_url"]
            and parameters.get("http_method") == "GET" and parameters.get("http_status") == 200
            and 0 < record["artifact"]["size_bytes"] <= limit)


def _district_names(store: ProvenanceStore, source: RosterSource) -> dict[str, str]:
    from pyogrio.errors import DataSourceError
    from pyogrio.raw import read_arrow

    record = store.read_record(source.receipt)
    if (not _download(record, "districts", "Texas Education Agency", "2026", 128 * 1024**2)
            or record["section"] != "spatial" or record["artifact"]["sha256"] != source.sha256
            or re.fullmatch(r"[0-9a-f]{64}", source.sha256) is None
            or record["parameters"].get("form_url") != _LOCATOR
            or not record["requested_url"].startswith(_LOCATOR + "?")):
        raise ValueError("Expected the pinned TEA Locator district source for 2025–26")
    path = (store.root / "objects" / source.sha256).resolve()
    if path.stat().st_size != record["artifact"]["size_bytes"]:
        raise ValueError("The retained TEA Locator size differs from its receipt")
    if path.as_posix().startswith("/vsi"):
        raise ValueError("The retained TEA Locator object must be a local GeoJSON file")
    store.verify_artifact(source.sha256)
    try:
        _, table = read_arrow(f"GeoJSON:{path}", columns=["NAME", "DISTRICT_C"], read_geometry=False)
    except DataSourceError as error:
        raise ValueError("The retained TEA Locator object must be a local GeoJSON file") from error
    names: dict[str, str] = {}
    numbers: set[str] = set()
    for name, number in zip(table["NAME"].to_pylist(), table["DISTRICT_C"].to_pylist(), strict=True):
        if not isinstance(number, str):
            raise ValueError("Invalid TEA Locator school district ID")
        identifier = stored_district_number(number, None)
        if not isinstance(name, str) or not name.strip() or identifier is None or identifier != number:
            raise ValueError("Invalid TEA Locator school district identity")
        key = name.strip().casefold()
        if key in names or identifier in numbers:
            raise ValueError(f"Duplicate TEA Locator district name or ID {name!r} ({identifier})")
        names[key] = identifier
        numbers.add(identifier)
    if not names:
        raise ValueError("The TEA Locator source has no school districts")
    return names


def _workbook(store: ProvenanceStore, source: RosterSource, plan: LegislativePlan) -> Path:
    name = plan.lower()
    filename = f"{name}_r160_isdssy25-26.xls"
    record = store.read_record(source.receipt)
    if (not _download(record, _RED160, "Texas Legislative Council", plan, 4 * 1024**2)
            or record["section"] != "reference" or record["artifact"]["sha256"] != source.sha256
            or record["source_metadata"]["source_url"] != f"{_TLC}/dataset/{name}"
            or record["artifact"]["filename"] != filename):
        raise ValueError(f"Expected a pinned {plan} Red-160 workbook for 2025–26")
    parents = record["source_metadata"]["provenance_ids"]
    if len(parents) != 1:
        raise ValueError(f"Expected one TLC plan catalog receipt for {plan}")
    catalog = store.read_record(parents[0])
    if (not _download(catalog, _RED160, "Texas Legislative Council", plan, 2 * 1024**2)
            or catalog["section"] != "reference"
            or catalog["requested_url"] != f"{_TLC}/api/3/action/package_show?id={name}"):
        raise ValueError(f"Unexpected TLC plan catalog receipt for {plan}")
    records = store.verify_record_closure([source.receipt])
    if set(records) != {source.receipt, parents[0]}:
        raise ValueError(f"Unexpected extra inputs for the {plan} Red-160 workbook")
    document = json.loads(store.verify_artifact(catalog["artifact"]["sha256"]).read_bytes())
    entries = [item["url"] for item in document["result"]["resources"]
               if item["name"].casefold() == filename and item["format"].casefold() == "xls"]
    if (document["success"] is not True or document["result"]["name"] != name
            or entries != [record["requested_url"]]
            or not re.fullmatch(rf"{re.escape(_TLC)}/dataset/[0-9a-f-]{{36}}/resource/[0-9a-f-]{{36}}/download/{re.escape(filename)}",
                                entries[0])):
        raise ValueError(f"TLC plan catalog does not select the pinned {plan} workbook")
    path = store.verify_artifact(source.sha256)
    with path.open("rb") as stream:
        if stream.read(8) != bytes.fromhex("D0CF11E0A1B11AE1"):
            raise ValueError(f"Expected a legacy Excel workbook for {plan}")
    return path


def _parse_rows(rows: Sequence[tuple[str | None, ...]], plan: LegislativePlan,
                names: Mapping[str, str]) -> tuple[LegislativeDistrictSchool, ...]:
    chamber, count = _PLANS[plan]
    if not 7 < len(rows) <= _MAX_ROWS:
        raise ValueError(f"Unexpected {plan} Red-160 worksheet length")
    report_identity = rows[1][1]
    blank = (None, None, None, None, None)
    if (not (report_identity or "").startswith(f"Red-160\nData: 2020 Census, SY 25-26\n{plan}  ")
            or tuple(rows[:7]) != (
                blank,
                (None, report_identity, "District Population Analysis with School District Subtotals", None, None),
                blank, (None, None, f"{chamber} DISTRICTS - {plan}", None, None),
                blank, blank, (None, None, None, None, "Total"),
            )):
        raise ValueError(f"Unexpected {plan} Red-160 worksheet header")
    result = []
    pairs: set[tuple[int, str]] = set()
    seat = 0
    seat_has_school = True
    index = 7
    while index < len(rows):
        number = index + 1
        label, _, _, marker, _ = rows[index]
        label = label or ""
        if match := _DISTRICT.fullmatch(label):
            if (marker is not None or not seat_has_school or int(match[1]) != seat + 1
                    or [row[3] for row in rows[index + 1:index + 3]] != ["Total:", "<18:"]
                    or any(row[0] is not None for row in rows[index + 1:index + 3])):
                raise ValueError(f"Invalid {plan} constituency block at worksheet row {number}")
            seat += 1
            seat_has_school = False
            index += 3
            continue
        school = _SCHOOL.fullmatch(label)
        if (not seat or school is None or marker != "Total:" or index + 1 == len(rows)
                or rows[index + 1][0] is not None or rows[index + 1][3] != "<18:"):
            raise ValueError(f"Invalid {plan} school subtotal at worksheet row {number}")
        identifier = names.get(school[1].casefold())
        if identifier is None or (seat, identifier) in pairs:
            raise ValueError(f"Unknown or repeated {plan} school district {label!r} at worksheet row {number}")
        pairs.add((seat, identifier))
        result.append(LegislativeDistrictSchool(seat, identifier, label, number))
        seat_has_school = True
        index += 2
    if seat != count or not seat_has_school or {identifier for _, identifier in pairs} != set(names.values()):
        raise ValueError(f"Incomplete {plan} constituency or school district coverage")
    return tuple(result)


def read_legislative_school_rosters(
    store: ProvenanceStore, *, district_locator: RosterSource, workbooks: Mapping[LegislativePlan, RosterSource],
) -> tuple[LegislativeSchoolRoster, ...]:
    """Read exact published school relationships using the same-year Locator names.

    Install the scripts extra for Excel intake. This operation reads retained
    files only. The results describe the publisher's school subtotals under
    the selected plan, and carry no assertion about its effective dates.
    """
    import fastexcel

    if not workbooks or any(not isinstance(plan, LegislativePlan) for plan in workbooks):
        raise ValueError("Select one or more LegislativePlan members with pinned workbooks")
    names = _district_names(store, district_locator)
    result = []
    for plan, source in workbooks.items():
        book = fastexcel.read_excel(_workbook(store, source, plan))
        if book.sheet_names != ["Red160"]:
            raise ValueError(f"Unexpected {plan} workbook sheets")
        sheet = book.load_sheet("Red160", header_row=None, skip_rows=0,
                                dtypes="string", n_rows=_MAX_ROWS + 1).to_arrow()
        if sheet.num_columns != 24:
            raise ValueError(f"Unexpected {plan} worksheet column count")
        rows = tuple(zip(*(sheet.column(column).to_pylist() for column in (0, 1, 7, 3, 4)), strict=True))
        result.append(LegislativeSchoolRoster(plan, _PERIOD, 2020, source, district_locator,
                                             _parse_rows(rows, plan, names)))
    return tuple(result)
