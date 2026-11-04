"""Retain exact campus PEIMS financial actual workbooks and publisher rosters."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from email.message import Message
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import build_opener
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile, ZipFile

from teadata.acquisition import AcquiredSource, AcquisitionClient, AcquisitionError, NoRedirect
from teadata.dataset_metadata import DataPeriod, DatasetMetadata, ReferenceDocument, ResolvedDataset, SourceQualifiers
from teadata.financial_leaves import NS, _package, _xml
from teadata.provenance import ProvenanceStore


DATASET = "campus_peims_financials"
BROKER = "https://rptsvr1.tea.texas.gov/cgi/sas/broker"
REPORTS = "https://rptsvr1.tea.texas.gov/school.finance/forecasting/financial_reports"
EDITIONS = (2024, 2025)
COLUMN_LABELS = ("General Fund", "%", "Per Student", "All Funds", "%", "Per Student")
SECTION_LABELS = {
    "A6": "Expenditures by Object (Objects 6100-6600)",
    "A11": "Expenditures by Function (Objects 6100-6400 Only)",
    "A28": "Program expenditures by Program (Objects 6100-6400 only)",
}


class _Selector(HTMLParser):
    def __init__(self, field: str):
        super().__init__()
        self.field = field
        self.matches = 0
        self.inside = False
        self.options: list[str] = []
        self.programs: list[str] = []
        self.headings: list[str] = []
        self.heading: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "select":
            self.inside = attributes.get("name") == self.field
            self.matches += int(self.inside)
        elif tag == "option" and self.inside:
            self.options.append(attributes.get("value") or "")
        elif tag == "input" and attributes.get("name") == "_program":
            self.programs.append(attributes.get("value") or "")
        elif tag in {"h2", "h3"}:
            self.heading = []

    def handle_endtag(self, tag):
        if tag == "select":
            self.inside = False
        elif tag in {"h2", "h3"} and self.heading is not None:
            self.headings.append(" ".join("".join(self.heading).split()))
            self.heading = None

    def handle_data(self, data):
        if self.heading is not None:
            self.heading.append(data)


def _broker_url(program: str, **params: str) -> str:
    return f"{BROKER}?" + urlencode({"_service": "marykay", "_debug": "0", **params, "_program": program})


def _parse_selector(path: Path, *, field: str, program: str, heading: str, district: str = "") -> tuple[str, ...]:
    parser = _Selector(field)
    parser.feed(path.read_text(encoding="utf-8-sig"))
    parser.close()
    pattern = rf"{district}[0-9]{{{3 if district else 6}}}"
    if (parser.matches != 1 or parser.inside or parser.programs != [program]
            or parser.headings.count(heading) != 1 or not parser.options
            or len(set(parser.options)) != len(parser.options)
            or any(re.fullmatch(pattern, option) is None for option in parser.options)):
        raise AcquisitionError(f"Invalid or changed TEA {field} roster for {district or heading}")
    return tuple(parser.options)


def _validate_attachment(headers, fiscal_year: str) -> None:
    message = Message()
    message["Content-Disposition"] = headers.get("content-disposition", "")
    if (message.get_content_disposition() != "attachment"
            or message.get_filename() != f"{fiscal_year} PEIMS_Campus_Data.xlsx"):
        raise AcquisitionError(f"Expected the {fiscal_year} PEIMS campus XLSX attachment")


def _validate_workbook(path: Path, fiscal_year: str, campus: str) -> None:
    """Validate headings and identity from OOXML without interpreting data cells."""
    try:
        with ZipFile(path) as archive:
            sheets, strings = _package(archive)
            if len(sheets) != 1:
                raise ValueError("Expected one worksheet")
            sheet = _xml(archive, next(iter(sheets.values())))
        cells = {}
        for cell in sheet.iter(NS + "c"):
            coordinate = cell.attrib["r"]
            if coordinate in cells or cell.find(NS + "f") is not None:
                raise ValueError("Unexpected duplicate cell or formula")
            cells[coordinate] = cell

        def label(coordinate: str) -> str:
            cell = cells[coordinate]
            if cell.get("t") == "inlineStr":
                return "".join(node.text or "" for node in cell.iter(NS + "t"))
            index = cell.findtext(NS + "v", "")
            if cell.get("t") != "s" or not re.fullmatch(r"[0-9]+", index):
                raise ValueError(f"Expected a string at {coordinate}")
            return strings[int(index)]

        title = label("A1").splitlines()
        if (len(title) != 3 or title[0] != f"{fiscal_year} PEIMS ACTUAL FINANCIAL DATA BY CAMPUS"
                or not re.fullmatch(rf"TOTALS FOR .+ \({campus}\)", title[1])
                or not title[2].strip()):
            raise ValueError("Actual edition or campus identity differs")
        if not label("A3").startswith("Total Enrolled Membership:"):
            raise ValueError("Unexpected membership label")
        if tuple(" ".join(label(f"{column}5").split()) for column in "BCDEFG") != COLUMN_LABELS:
            raise ValueError("Unexpected financial column labels")
        for coordinate, expected in SECTION_LABELS.items():
            if " ".join(label(coordinate).split()) != expected:
                raise ValueError(f"Unexpected financial section heading at {coordinate}")
    except (BadZipFile, IndexError, KeyError, ParseError, ValueError) as error:
        raise AcquisitionError(f"Invalid {fiscal_year} campus actual workbook for {campus}: {error}") from error


def acquire_campus_financials(
    client: AcquisitionClient, year: int, districts: Sequence[str], campuses: Sequence[str] = (), *, refresh: bool = False,
) -> dict:
    if year not in EDITIONS or not districts:
        raise ValueError("Select a supported fiscal end year and at least one district")
    for ids, length, name in ((districts, 6, "district"), (campuses, 9, "campus")):
        if len(set(ids)) != len(ids) or any(re.fullmatch(rf"[0-9]{{{length}}}", value) is None for value in ids):
            raise ValueError(f"Select unique {length}-digit {name} IDs")
    if any(campus[:6] not in districts for campus in campuses):
        raise ValueError("Every selected campus must belong to a selected district")

    fiscal_year = f"{year - 1}-{year}"
    suffix = str(year)[-2:]
    index = f"{REPORTS}/{str(year - 1)[-2:]}{suffix}_new_camp_actual.html"
    campus_selector_program = f"sfadhoc.new_pop_campusbox_actual{suffix}.sas"
    workbook_program = f"sfadhoc.new_Campus_actual{suffix}.sas"
    metadata = DatasetMetadata(
        title=f"{fiscal_year} PEIMS Campus Financial Actual Reports",
        publisher="Texas Education Agency", version=str(year), source_url=index,
        period=DataPeriod(kind="fiscal_year", start_year=year - 1, end_year=year),
    )
    source = ResolvedDataset(DATASET, "data_sources", year, year, "", metadata)
    prior = {} if refresh else {
        record["requested_url"]: record
        for record in client.store.history(DATASET)
        if record["kind"] == "acquisition" and record["method"] == "download"
        and record["requested_year"] == record["resolved_year"] == year
        and record["requested_url"] == record["resolved_url"]
        and record["parameters"].get("adapter") == DATASET
        and record["parameters"].get("adapter_revision") == 1
    }
    fetched = False

    def acquire(
        url: str, filename: str, validate: Callable[[Path], None], *,
        scope: dict[str, str], parent: AcquiredSource | None = None,
        validate_response: Callable[[Mapping[str, str]], None] | None = None,
    ) -> AcquiredSource:
        nonlocal fetched
        dependencies = (parent.record_id,) if parent is not None else ()
        cached = prior.get(url)
        if cached is not None and tuple(cached["source_metadata"]["provenance_ids"]) == dependencies:
            path = client.store.verify_artifact(cached["artifact"]["sha256"])
            if validate_response is not None:
                validate_response(cached["response_headers"])
            validate(path)
            return AcquiredSource(cached["record_id"], path)
        if fetched:
            time.sleep(0.5)
        fetched = True
        return client.fetch(
            url, replace(source, metadata=replace(metadata, download_url=url, provenance_ids=dependencies)),
            filename=filename, validate=validate, validate_response=validate_response,
            parameters={"adapter": DATASET, "adapter_revision": 1, **scope},
        )

    publisher_districts: tuple[str, ...] = ()

    def validate_index(path: Path) -> None:
        nonlocal publisher_districts
        publisher_districts = _parse_selector(
            path, field="who_list", program=campus_selector_program,
            heading=f"Select District for Financial Actual Campus {fiscal_year} Report:",
        )

    index_receipt = acquire(index, f"{fiscal_year}-district-index.html", validate_index, scope={"role": "district_index"})
    missing_districts = sorted(set(districts).difference(publisher_districts))
    if missing_districts:
        raise AcquisitionError(f"Districts absent from publisher's {fiscal_year} actual roster: {', '.join(missing_districts)}")
    index_hash = client.store.read_record(index_receipt.record_id)["artifact"]["sha256"]
    metadata = replace(
        metadata,
        references=(ReferenceDocument("publisher_selector", "source_notes", url=index,
                                      provenance_ids=(index_receipt.record_id,), reviewed_sha256=index_hash),),
        source_qualifiers=SourceQualifiers("peims_financial", reference_id="publisher_selector", financial_basis="actual"),
    )
    rosters = {}
    selected = set(campuses)
    for district in districts:
        publisher_campuses: tuple[str, ...] = ()

        def validate_roster(path: Path, district_id: str = district) -> None:
            nonlocal publisher_campuses
            publisher_campuses = _parse_selector(
                path, field="which_camp", program=workbook_program, heading="Select a Campus:", district=district_id,
            )

        url = _broker_url(campus_selector_program, who_list=district)
        roster = acquire(url, f"{fiscal_year}-{district}-campus-index.html", validate_roster,
                         scope={"role": "campus_index", "district_id": district}, parent=index_receipt)
        rosters[district] = (roster, publisher_campuses)
    listed = {campus for _, roster_ids in rosters.values() for campus in roster_ids}
    missing_campuses = sorted(selected - listed)
    if missing_campuses:
        raise AcquisitionError(f"Campuses absent from publisher's {fiscal_year} actual roster: {', '.join(missing_campuses)}")

    output = {}
    for district, (roster, roster_ids) in rosters.items():
        records = {}
        for campus in roster_ids:
            if selected and campus not in selected:
                continue
            url = _broker_url(workbook_program, single="N", batch="N", app="PUBLIC", ptype="E", which_camp=campus, search="dist")
            result = acquire(
                url, f"{fiscal_year}-actual-{campus}.xlsx", lambda path: _validate_workbook(path, fiscal_year, campus),
                scope={"role": "campus_workbook", "district_id": district, "campus_id": campus}, parent=roster,
                validate_response=lambda headers: _validate_attachment(headers, fiscal_year),
            )
            records[campus] = result.record_id
        output[district] = {"roster_receipt": roster.record_id, "listed_campuses": list(roster_ids), "workbook_receipts": records}
    return {"fiscal_end_year": year, "district_index_receipt": index_receipt.record_id,
            "publisher_district_count": len(publisher_districts), "districts": output}


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=EDITIONS, required=True, help="Fiscal year's ending year")
    parser.add_argument("--store", type=Path, required=True, help="Provenance archive directory")
    parser.add_argument("--district", action="append", required=True, help="Six-digit district ID, repeatable")
    parser.add_argument("--campus", action="append", default=[], help="Optional nine-digit campus selection, repeatable")
    parser.add_argument("--refresh", action="store_true", help="Fetch fresh originals while preserving prior receipts")
    args = parser.parse_args(argv)
    client = AcquisitionClient(ProvenanceStore(args.store), opener=build_opener(NoRedirect()), max_bytes=2 * 1024 * 1024)
    try:
        result = acquire_campus_financials(client, args.year, args.district, args.campus, refresh=args.refresh)
    except (OSError, URLError, ValueError, AcquisitionError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
