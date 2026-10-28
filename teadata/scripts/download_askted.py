"""Acquire a publisher-labelled AskTED archive without hardcoded form controls."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from email.message import Message
from hashlib import sha256
from html.parser import HTMLParser
from http.cookiejar import CookieJar
from pathlib import Path
import re
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, build_opener

from teadata.acquisition import AcquiredSource, AcquisitionClient, AcquisitionError
from teadata.dataset_metadata import DataPeriod, DatasetMetadata, ResolvedDataset
from teadata.provenance import ProvenanceStore


ARCHIVE_URL = "https://tealprod.tea.state.tx.us/Tea.AskTed.Web/Forms/ArchivedSchoolAndDistrictDataFiles.aspx"


@dataclass(frozen=True)
class Archive:
    end_year: int
    season: str
    declared_as_of: str
    event_target: str


class ArchivePage(HTMLParser):
    def __init__(self):
        super().__init__()
        self.fields: dict[str, str] = {}
        self.archives: list[Archive] = []
        self._row: list[str] = []
        self._link: list[str] | None = None
        self._target: str | None = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "input" and attrs.get("type", "").lower() == "hidden" and attrs.get("name"):
            self.fields[attrs["name"]] = attrs.get("value", "")
        if tag == "tr":
            self._row = []
        if tag == "a":
            match = re.fullmatch(
                r"javascript:__doPostBack\('([^']+)',''\)", attrs.get("href", "")
            )
            self._target = match[1] if match else None
            self._link = [] if match else None

    def handle_data(self, data):
        self._row.append(data)
        if self._link is not None:
            self._link.append(data)

    def handle_endtag(self, tag):
        if tag != "a" or self._link is None:
            return
        year = re.search(r"(\d{4})-(\d{4}) School Year", " ".join(self._row))
        label = re.fullmatch(r"(Fall|Spring) \(as of (\d{2}/\d{2}/\d{4})\)", "".join(self._link).strip())
        if year and label and self._target is not None and int(year[2]) == int(year[1]) + 1:
            self.archives.append(Archive(
                int(year[2]), label[1].lower(),
                datetime.strptime(label[2], "%m/%d/%Y").date().isoformat(),
                self._target,
            ))
        self._link = None
        self._target = None


def acquire_archive(store: ProvenanceStore, end_year: int, season: str) -> AcquiredSource:
    if season not in {"fall", "spring"}:
        raise ValueError("Archive season must be fall or spring")
    client = AcquisitionClient(store, opener=build_opener(HTTPCookieProcessor(CookieJar())))
    metadata = DatasetMetadata(title="AskTED archived school and district files", publisher="Texas Education Agency", source_url=ARCHIVE_URL)
    source = ResolvedDataset("public_school_directory", "data_sources", end_year, end_year, "", metadata)
    adapter_identity = {"adapter_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    page = client.fetch(ARCHIVE_URL, source, filename="askted-archive-index.html", parameters=adapter_identity)
    parser = ArchivePage()
    parser.feed(page.path.read_text(encoding="utf-8-sig"))
    matches = [item for item in parser.archives if item.end_year == end_year and item.season == season]
    if len(matches) != 1:
        raise AcquisitionError(f"Expected one published {end_year} {season} archive; found {len(matches)}")
    selected = matches[0]
    fields = {**parser.fields, "__EVENTTARGET": selected.event_target, "__EVENTARGUMENT": ""}
    from dataclasses import replace
    metadata = replace(metadata,
        title=f"AskTED {end_year - 1}-{end_year} {season} school and district directory",
        version=f"{end_year}-{season}",
        period=DataPeriod.from_dict({"kind": "as_of", "as_of": selected.declared_as_of}),
        download_url=ARCHIVE_URL,
        notes="The as-of date is the publisher's archive label. Row update dates must be audited before claiming a point-in-time snapshot. Enrollment fields carry their own reference periods.",
    )

    return client.fetch(
        ARCHIVE_URL, replace(source, metadata=metadata),
        filename=f"askted-{end_year}-{season}.csv", body=urlencode(fields).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        parameters={**adapter_identity, "archive_index_record": page.record_id, "school_year_end": end_year,
                    "season": season, "publisher_declared_as_of": selected.declared_as_of},
        validate=lambda path: validate_archive_columns(path, selected),
        validate_response=lambda headers: validate_archive_response(headers, selected),
    )


def validate_archive_response(headers, selected: Archive) -> None:
    """Require the attachment identity, independently of the requested form control."""
    message = Message()
    message["Content-Disposition"] = headers.get("content-disposition", "")
    filename = message.get_filename()
    expected = f"ArchivedSchoolAndDistrict{selected.season.title()}{selected.end_year}.csv"
    if message.get_content_disposition() != "attachment" or filename != expected:
        raise AcquisitionError("AskTED attachment does not match the selected year and season")
    if headers.get("content-type", "").split(";", 1)[0].strip().lower() != "text/csv":
        raise AcquisitionError("AskTED attachment is not CSV")


def validate_archive_columns(path: Path, selected: Archive) -> None:
    import csv
    with path.open(encoding="utf-8-sig", newline="") as stream:
        columns = next(csv.reader(stream), [])
    required = {"District Number", "District Name", "District Type", "School Number", "School Name", "School Status", "Charter Type"}
    # Both published seasons carry enrollment from October of the starting year.
    expected = {f"{scope} Enrollment as of Oct {selected.end_year - 1}" for scope in ("District", "School")}
    enrollment = {column for column in columns if "Enrollment as of" in column}
    if (len(columns) != len(set(columns)) or not required.issubset(columns)
            or enrollment != expected):
        raise AcquisitionError("AskTED directory columns do not match the selected school year")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True, help="School year's ending year")
    parser.add_argument("--season", choices=("fall", "spring"), required=True)
    parser.add_argument("--store", type=Path, default=Path("provenance"))
    args = parser.parse_args()
    acquired = acquire_archive(ProvenanceStore(args.store), args.year, args.season)
    print(f"receipt={acquired.record_id}\narchived_object={acquired.path}")


if __name__ == "__main__":
    main()
