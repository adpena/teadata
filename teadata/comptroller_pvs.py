"""Read county relationships from retained Final Property Value Study pages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from html.parser import HTMLParser
import re
from typing import Any, Literal, NamedTuple

from .dataset_metadata import DataPeriod
from .identifiers import canonical_source_identifier
from .provenance import ProvenanceStore, validate_record_graph


_HOST = "https://comptroller.texas.gov"
_REPORTS = r"/auto-data/PT2/PVS/([0-9]{4}[FP])/"
_COUNTY_PAGE = re.compile(rf"{_REPORTS}([0-9]{{3}})index\.php")
_WORKSHEET = re.compile(rf"{_REPORTS}([0-9]{{3}})([0-9]{{6}})1([DF-L])\.php")
_COUNTY_HEADING = re.compile(r"([0-9]{3}) (.+) County")
_DISTRICT_HEADING = re.compile(r"([0-9]{3})-([0-9]{3}) (.+)")
_COUNTIES = frozenset(f"{number:03}" for number in range(1, 255))
_PAGE_CODES = _COUNTIES - {"191"}
_MAX_BYTES = 1_000_000
_SUMMARY_LABEL = "School District Summary Worksheet"
_COUNTY_LIST_TITLE = "School District Findings by County"
_COUNTY_REPORT_TITLE = "SCHOOL DISTRICT PROPERTY VALUE STUDY (GOV’T. CODE §403.302)"
_ALTERNATE = {"template", "noscript"}
_EXCLUDED = _ALTERNATE | {"aside", "footer"}


@dataclass(frozen=True, slots=True)
class _Edition:
    year: int

    @property
    def index_url(self) -> str:
        return f"{_HOST}/taxes/property-tax/pvs/{self.year}f/index.php"

    @property
    def report_path(self) -> str:
        return f"/auto-data/PT2/PVS/{self.year}F/"

    @property
    def county_heading(self) -> str:
        return "h3" if self.year == 2022 else "h2"

    @property
    def period(self) -> DataPeriod:
        return DataPeriod("calendar_year", year=self.year)


@dataclass(frozen=True, slots=True)
class CountyDistrictFinding:
    county_code: str
    county_name: str
    district_id: str
    source_district_label: str
    worksheet_url: str
    county_page_receipt: str


@dataclass(frozen=True, slots=True)
class ComptrollerCountyRoster:
    period: DataPeriod
    index_receipt: str
    rows: tuple[CountyDistrictFinding, ...]

    def for_county(self, county_code: str) -> tuple[CountyDistrictFinding, ...]:
        """Select by three-digit Comptroller county code."""
        if county_code not in _COUNTIES:
            raise ValueError("Expected a Comptroller county code from '001' through '254'")
        return tuple(row for row in self.rows if row.county_code == county_code)

    def for_district(self, district_id: str | int) -> tuple[CountyDistrictFinding, ...]:
        """Select by district ID. A valid ID can be absent from the study."""
        key = canonical_source_identifier(district_id, width=6)
        if key is None:
            raise ValueError("Expected a six-digit school district ID")
        return tuple(row for row in self.rows if row.district_id == key[1:])


class _Element(NamedTuple):
    tag: str
    text: str
    href: str | None = None
    scope: Literal["outside", "main", "county_list"] = "main"


class _Page(HTMLParser):
    def __init__(self, html: str, *, county_heading: str | None = None) -> None:
        super().__init__(convert_charrefs=True)
        self.elements: list[_Element] = []
        self._heading: tuple[str, list[str]] | None = None
        self._anchor: tuple[str | None, list[str]] | None = None
        self._main = False
        self._main_seen = False
        self._excluded: list[str] = []
        self._county_heading = county_heading
        self._awaiting_county_list = False
        self._list_depth = 0
        self._list_item_has_link = False
        self.feed(html)
        self.close()
        if not self._main_seen or self._main or self._awaiting_county_list:
            raise ValueError("Expected one complete Comptroller main report")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._awaiting_county_list:
            if tag != "ul" or "three-columns" not in (dict(attrs).get("class") or "").split():
                raise ValueError("Expected the list immediately after the Comptroller county heading")
            self._awaiting_county_list = False
            self.elements.append(_Element("county_list", ""))
            self._list_depth = 1
            return
        if self._list_depth:
            expected = {1: "li", 2: "a"}.get(self._list_depth)
            if tag != expected or (tag == "li" and attrs) or (
                tag == "a" and (self._list_item_has_link or [name for name, _ in attrs] != ["href"])
            ):
                raise ValueError("Expected direct link entries in the Comptroller county list")
            self._list_item_has_link = tag == "a"
            self._list_depth += 1
        if tag == "main":
            if self._main_seen or self._anchor is not None or self._excluded:
                raise ValueError("Unexpected Comptroller main report boundary")
            self._main_seen = self._main = True
        elif tag in _EXCLUDED:
            if self._main and tag in _ALTERNATE:
                raise ValueError("Unexpected alternate content inside the Comptroller report")
            if self._main and (self._heading is not None or self._anchor is not None):
                raise ValueError("Comptroller report element crosses a sidebar or footer boundary")
            self._excluded.append(tag)
        elif tag == "ul" and self._county_heading is None and self._main and not self._excluded:
            if "three-columns" in (dict(attrs).get("class") or "").split():
                self._list_depth = 1
        elif tag in {"h1", "h2", "h3"} and self._main and not self._excluded:
            self._heading = (tag, [])
        elif tag == "a":
            hrefs = [value for name, value in attrs if name == "href"]
            if self._main and (len(hrefs) > 1 or self._anchor is not None):
                raise ValueError("Ambiguous Comptroller report anchor")
            self._anchor = (hrefs[0] if hrefs else None, [])
        elif tag == "br":
            self.handle_data(" ")

    def handle_data(self, data: str) -> None:
        if self._awaiting_county_list and data.strip():
            raise ValueError("Expected the list immediately after the Comptroller county heading")
        if self._heading is not None:
            self._heading[1].append(data)
        if self._anchor is not None:
            self._anchor[1].append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._awaiting_county_list:
            raise ValueError("Expected the list immediately after the Comptroller county heading")
        if tag == "main":
            if not self._main or self._heading is not None or self._anchor is not None or self._list_depth:
                raise ValueError("Unexpected Comptroller main report boundary")
            self._main = False
        if self._excluded and tag == self._excluded[-1]:
            if self._main and (self._heading is not None or self._anchor is not None):
                raise ValueError("Comptroller report element crosses a sidebar or footer boundary")
            self._excluded.pop()
        if self._heading is not None and tag == self._heading[0]:
            heading_text = " ".join("".join(self._heading[1]).split())
            self.elements.append(_Element(tag, heading_text))
            if tag == self._county_heading and heading_text == _COUNTY_LIST_TITLE:
                self._awaiting_county_list = True
            self._heading = None
        if self._anchor is not None and tag == "a":
            href, parts = self._anchor
            scope = "county_list" if self._list_depth else "main" if self._main and not self._excluded else "outside"
            self.elements.append(_Element(tag, " ".join("".join(parts).split()), href, scope))
            self._anchor = None
        if self._list_depth:
            if tag != ("ul", "li", "a")[self._list_depth - 1] or (
                tag == "li" and not self._list_item_has_link
            ):
                raise ValueError("Expected complete direct entries in the Comptroller county list")
            self._list_depth -= 1


def _index_counties(html: str, edition: _Edition) -> dict[str, str]:
    elements = _Page(html, county_heading=edition.county_heading).elements
    title = f"School District Property Value Study Government Code Section 403.302 {edition.year} Final Findings"
    if edition.year == 2022:
        title = title.upper()
    if [element.text for element in elements if element.tag == "h1"] != [title]:
        raise ValueError(f"Expected the {edition.year} Final School District Property Value Study index")
    if sum(element.tag == "county_list" for element in elements) != 1:
        raise ValueError("Expected one list under the Comptroller county heading")
    counties = {}
    for element in elements:
        if element.tag != "a":
            continue
        match = _COUNTY_PAGE.fullmatch(element.href or "")
        if element.scope != "county_list":
            if match is not None:
                raise ValueError("Comptroller county link falls outside the county list")
            continue
        if match is None or match[1] != f"{edition.year}F" or not element.text or match[2] in counties:
            raise ValueError("Malformed or repeated Comptroller county index link")
        counties[match[2]] = element.text
    if counties.keys() != _PAGE_CODES or counties["188"] != "Potter-Randall":
        raise ValueError("Incomplete or unrecognized Comptroller county index")
    return counties


def _county_rows(html: str, county_names: Mapping[str, str], receipt: str,
                 edition: _Edition) -> list[CountyDistrictFinding]:
    elements = _Page(html).elements
    subtitle = f"{edition.year} Final Findings"
    titles = [subtitle, _COUNTY_REPORT_TITLE, subtitle]
    if [element.text for element in elements if element.tag == "h1"] != titles:
        raise ValueError(f"Expected a {edition.year} Final Comptroller county report")
    county_code = ""
    pending: tuple[str, str] | None = None
    last_district: str | None = None
    counts: dict[str, int] = {}
    districts: set[tuple[str, str]] = set()
    rows: list[CountyDistrictFinding] = []
    for element in elements:
        if element.tag == "h2":
            if pending is not None:
                raise ValueError(f"Missing Comptroller worksheet for {pending[0]}")
            last_district = None
            if county := _COUNTY_HEADING.fullmatch(element.text):
                code, name = county.groups()
                if county_names.get(code) != name or code in counts:
                    raise ValueError(f"Unexpected or repeated Comptroller county heading {element.text!r}")
                county_code = code
                counts[code] = 0
            elif district := _DISTRICT_HEADING.fullmatch(element.text):
                pending = (district[1] + district[2], district[3])
        if element.tag != "a":
            continue
        href = element.href or ""
        worksheet = _WORKSHEET.fullmatch(href)
        if worksheet is None and element.text != _SUMMARY_LABEL:
            continue
        if worksheet is not None and worksheet[1] != f"{edition.year}F":
            raise ValueError(f"Comptroller worksheet belongs to a different edition: {href!r}")
        if worksheet is not None and worksheet[4] != "D" and element.text != _SUMMARY_LABEL:
            if element.scope == "main" and pending is None and (county_code, last_district) == worksheet.group(2, 3):
                continue
            raise ValueError(f"Orphan Comptroller district detail worksheet: {href!r}")
        if (element.scope != "main" or worksheet is None or worksheet[4] != "D" or element.text != _SUMMARY_LABEL
                or pending is None or (county_code, pending[0]) != worksheet.group(2, 3) or pending[0] == "000000"):
            raise ValueError(f"Comptroller worksheet does not match its county and district headings: {href!r}")
        district_id, label = pending
        if (county_code, district_id) in districts:
            raise ValueError(f"Repeated Comptroller district {district_id} in county {county_code}")
        districts.add((county_code, district_id))
        counts[county_code] += 1
        rows.append(CountyDistrictFinding(county_code, county_names[county_code], district_id,
                                          label, _HOST + href, receipt))
        last_district = district_id
        pending = None
    if pending is not None or counts.keys() != county_names.keys() or not all(counts.values()):
        raise ValueError("Incomplete Comptroller county or district blocks")
    return rows


def _source_html(store: ProvenanceStore, record: Mapping[str, Any], *, url: str,
                 parents: list[str], edition: _Edition) -> str:
    if record["kind"] != "acquisition":
        raise ValueError(f"Expected a Comptroller acquisition receipt for {url}")
    metadata, parameters, artifact = (record["source_metadata"], record["parameters"], record["artifact"])
    try:
        downloaded = datetime.fromisoformat(record["downloaded_at"])
        recorded = datetime.fromisoformat(record["recorded_at"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Expected valid Comptroller acquisition timestamps for {url}") from error
    status = parameters.get("http_status")
    if not (record["method"] == "download"
            and downloaded.utcoffset() == recorded.utcoffset() == timedelta(0) and downloaded <= recorded
            and downloaded.isoformat() == record["downloaded_at"] and recorded.isoformat() == record["recorded_at"]
            and record["dataset"] == "reference:comptroller_pvs_county_rosters"
            and record["section"] == "reference"
            and type(record["requested_year"]) is type(record["resolved_year"]) is int
            and record["requested_year"] == record["resolved_year"] == edition.year
            and record["requested_url"] == record["resolved_url"] == url
            and metadata["publisher"] == "Texas Comptroller of Public Accounts"
            and metadata["version"] == f"{edition.year}-final" and metadata["source_url"] == edition.index_url
            and metadata["period"] == edition.period.to_dict() and metadata["provenance_ids"] == parents
            and metadata["references"] == []
            and parameters.get("http_method") == "GET" and type(status) is int and status == 200
            and 0 < artifact["size_bytes"] <= _MAX_BYTES):
        raise ValueError(f"Expected a direct {edition.year} Final Comptroller acquisition for {url}")
    # The graph validates artifact IDs. Hash the bounded bytes that will be parsed.
    digest = artifact["sha256"]
    with (store.root / "objects" / digest).open("rb") as stream:
        data = stream.read(artifact["size_bytes"] + 1)
    if len(data) != artifact["size_bytes"] or sha256(data).hexdigest() != digest:
        raise ValueError(f"Comptroller artifact size or checksum mismatch for {url}")
    return data.decode("utf-8")


def read_comptroller_county_roster(
    store: ProvenanceStore, *, index_receipt: str, county_page_receipts: Mapping[str, str], tax_year: int = 2025,
) -> ComptrollerCountyRoster:
    """Read a complete 2022–2025 Final county roster from a pinned capture.

    This offline reader returns taxable-property reporting relationships. The
    county codes are Comptroller codes, and the rows contain no geometry or
    campus assignments. All county receipts must point to the selected index.
    """
    if type(tax_year) is not int or not 2022 <= tax_year <= 2025:
        raise ValueError("Choose a Comptroller Final tax year from 2022 through 2025")
    edition = _Edition(tax_year)
    if county_page_receipts.keys() != _PAGE_CODES:
        raise ValueError(f"Supply all 253 county page receipts for the {tax_year} Final study")
    selected = {index_receipt, *county_page_receipts.values()}
    if len(selected) != len(_PAGE_CODES) + 1:
        raise ValueError("Comptroller index and county page receipts must be distinct")
    records = validate_record_graph(store.read_record, sorted(selected))
    if records.keys() != selected:
        raise ValueError("Comptroller receipts contain unexpected inputs")
    index_html = _source_html(store, records[index_receipt], url=edition.index_url, parents=[], edition=edition)
    counties = _index_counties(index_html, edition)
    rows: list[CountyDistrictFinding] = []
    for code, name in counties.items():
        receipt = county_page_receipts[code]
        url = f"{_HOST}{edition.report_path}{code}index.php"
        names = {"188": "Potter", "191": "Randall"} if code == "188" else {code: name}
        html = _source_html(store, records[receipt], url=url, parents=[index_receipt], edition=edition)
        rows.extend(_county_rows(html, names, receipt, edition))
    rows.sort(key=lambda row: (row.county_code, row.district_id))
    return ComptrollerCountyRoster(edition.period, index_receipt, tuple(rows))
