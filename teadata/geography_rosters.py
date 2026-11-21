"""Build and read an offline school district roster projection by geography."""

from __future__ import annotations

from collections.abc import Generator, Iterable, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass
from hashlib import file_digest
import json
import os
from pathlib import Path
import sqlite3
from stat import S_ISREG
from tempfile import TemporaryDirectory
from typing import Literal, TypedDict

from ._sqlite import readonly_connection
from .askted_directory import AskTEDDirectory, DirectoryField, Season, read_askted_directory
from .dataset_metadata import DataPeriod
from .legislative_rosters import LegislativePlan, LegislativeSchoolRoster, RosterSource, read_legislative_school_rosters
from .manifest import canonical_json
from .provenance import ProvenanceStore

_ESC_FIELDS = (DirectoryField.ESC_REGION_SERVED, DirectoryField.ESC_REGION_PEIMS,
               DirectoryField.ESC_REGION_GEOGRAPHIC)
_APPLICATION_ID = 0x54475253
_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class GeographyRosterMember:
    district_number: str
    publisher_label: str | None
    source_records: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GeographyRosterIssue:
    district_number: str
    status: Literal["missing", "conflict"]
    candidate_regions: tuple[str, ...]
    source_records: tuple[int, ...]


class GeographyAcquisition(TypedDict):
    receipt: str
    sha256: str
    url: str
    retrieved_at: str
    publisher: str
    edition: str | None


@dataclass(frozen=True, slots=True)
class GeographyRosterCitation:
    period: DataPeriod
    census_year: int | None
    acquisitions: tuple[GeographyAcquisition, ...]


@dataclass(frozen=True, slots=True)
class GeographyRoster:
    relationship: LegislativePlan | DirectoryField
    geography_number: str
    citation: GeographyRosterCitation
    members: tuple[GeographyRosterMember, ...]


def _acquisition(store: ProvenanceStore, receipt: str) -> GeographyAcquisition:
    record = store.read_record(receipt)
    return {"receipt": receipt, "sha256": record["artifact"]["sha256"], "url": record["requested_url"],
            "retrieved_at": record["downloaded_at"], "publisher": record["source_metadata"]["publisher"],
            "edition": record["source_metadata"].get("version")}


def _create(connection: sqlite3.Connection) -> None:
    connection.executescript(f"""
        PRAGMA application_id = {_APPLICATION_ID};
        PRAGMA user_version = {_SCHEMA_VERSION};
        PRAGMA foreign_keys = ON;
        CREATE TABLE source (
            relationship TEXT PRIMARY KEY, citation TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE geography (
            relationship TEXT REFERENCES source NOT NULL,
            number TEXT NOT NULL,
            PRIMARY KEY (relationship, number)
        ) WITHOUT ROWID;
        CREATE TABLE member (
            relationship TEXT NOT NULL, number TEXT NOT NULL, district_number TEXT NOT NULL,
            publisher_label TEXT, source_records TEXT NOT NULL,
            PRIMARY KEY (relationship, number, district_number),
            FOREIGN KEY (relationship, number) REFERENCES geography
        ) WITHOUT ROWID;
        CREATE TABLE directory_issue (
            relationship TEXT REFERENCES source NOT NULL, district_number TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('missing', 'conflict')),
            candidate_regions TEXT NOT NULL, source_records TEXT NOT NULL,
            PRIMARY KEY (relationship, district_number)
        ) WITHOUT ROWID;
    """)


def _register(connection: sqlite3.Connection, relationship: str, numbers: Iterable[str], *,
              period: DataPeriod, acquisitions: tuple[GeographyAcquisition, ...], census_year: int | None = None) -> None:
    citation = {"period": period.to_dict(), "census_year": census_year, "acquisitions": acquisitions}
    connection.execute("INSERT INTO source VALUES (?, ?)", (relationship, canonical_json(citation)))
    connection.executemany("INSERT INTO geography VALUES (?, ?)", ((relationship, number) for number in numbers))


def _write_directory(connection: sqlite3.Connection, directory: AskTEDDirectory, store: ProvenanceStore) -> None:
    source = directory.source
    acquisitions = tuple(_acquisition(store, receipt) for receipt in (source.receipt, source.index_receipt))
    period = DataPeriod("as_of", as_of=source.publisher_declared_as_of)
    for field in _ESC_FIELDS:
        _register(connection, field, (f"{number:02}" for number in range(1, 21)),
                  period=period, acquisitions=acquisitions)
        for identifier, entry in directory.districts.items():
            codes = entry.values(field)
            if len(codes) == 1:
                records = tuple(row.record for row in entry.observations if row.fields[field].strip())
                connection.execute("INSERT INTO member VALUES (?, ?, ?, ?, ?)",
                                   (field, codes[0], identifier, entry.value(DirectoryField.NAME), canonical_json(records)))
            else:
                records = tuple(row.record for row in entry.observations)
                status = "conflict" if codes else "missing"
                connection.execute("INSERT INTO directory_issue VALUES (?, ?, ?, ?, ?)",
                                   (field, identifier, status, canonical_json(codes), canonical_json(records)))


def _write_legislative(connection: sqlite3.Connection, roster: LegislativeSchoolRoster, store: ProvenanceStore) -> None:
    numbers = sorted({row.legislative_district_number for row in roster.rows})
    workbook = store.read_record(roster.workbook.receipt)
    catalog, = workbook["source_metadata"]["provenance_ids"]
    acquisitions = tuple(_acquisition(store, receipt)
                         for receipt in (roster.workbook.receipt, catalog, roster.district_locator.receipt))
    _register(connection, roster.plan, map(str, numbers), period=roster.school_boundary_period,
              acquisitions=acquisitions, census_year=roster.census_year)
    connection.executemany("INSERT INTO member VALUES (?, ?, ?, ?, ?)",
                           ((roster.plan, str(row.legislative_district_number), row.school_district_number,
                             row.source_label, canonical_json((row.worksheet_row,))) for row in roster.rows))


def build_geography_rosters(store: ProvenanceStore, *, directory: RosterSource, year: int, season: Season,
                            district_locator: RosterSource, workbooks: Mapping[LegislativePlan, RosterSource],
                            output: Path) -> str:
    """Build a new immutable roster artifact and return its SHA-256 pin."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    archive = read_askted_directory(store, directory.receipt, year=year, season=season,
                                   expected_sha256=directory.sha256)
    legislative = read_legislative_school_rosters(store, district_locator=district_locator, workbooks=workbooks)
    with TemporaryDirectory(prefix=".geography-rosters-", dir=output.parent) as temp:
        path = Path(temp) / "rosters.sqlite3"
        with closing(sqlite3.connect(path)) as connection, connection:
            _create(connection)
            _write_directory(connection, archive, store)
            for roster in legislative:
                _write_legislative(connection, roster, store)
        with path.open("rb") as stream:
            pin = file_digest(stream, "sha256").hexdigest()
        os.link(path, output)
    return pin


class GeographyRosters:
    """Indexed reads over one verified local roster file."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def get(self, relationship: LegislativePlan | DirectoryField, number: int) -> GeographyRoster:
        esc = isinstance(relationship, DirectoryField) and relationship in _ESC_FIELDS
        if not (esc or isinstance(relationship, LegislativePlan)) or type(number) is not int or number < 1:
            raise ValueError("Choose an ESC assignment field or legislative plan and a positive geography number")
        key = f"{number:02}" if esc else str(number)
        params = (relationship, key)
        record = self._connection.execute("""SELECT source.citation FROM geography JOIN source USING (relationship)
            WHERE relationship = ? AND number = ?""", params).fetchone()
        if record is None:
            raise KeyError(f"Geography {key} is unavailable for {relationship}")
        citation = json.loads(record[0])
        rows = self._connection.execute("""SELECT district_number, publisher_label, source_records FROM member
            WHERE relationship = ? AND number = ? ORDER BY district_number""", params)
        return GeographyRoster(relationship, key, GeographyRosterCitation(DataPeriod.from_dict(citation["period"]),
            citation["census_year"], tuple(citation["acquisitions"])),
            tuple(GeographyRosterMember(identifier, label, tuple(json.loads(records))) for identifier, label, records in rows))

    def directory_issues(self, assignment: DirectoryField) -> tuple[GeographyRosterIssue, ...]:
        if not isinstance(assignment, DirectoryField) or assignment not in _ESC_FIELDS:
            raise ValueError("Choose a served, PEIMS reporting or geographic ESC assignment field")
        rows = self._connection.execute("""SELECT district_number, status, candidate_regions, source_records
            FROM directory_issue WHERE relationship = ? ORDER BY district_number""", (assignment,))
        return tuple(GeographyRosterIssue(identifier, status, tuple(json.loads(codes)), tuple(json.loads(records)))
                     for identifier, status, codes, records in rows)


@contextmanager
def open_geography_rosters(path: Path, *, expected_sha256: str) -> Generator[GeographyRosters]:
    """Verify exact local bytes once, then serve indexed read-only selections."""
    path = path.absolute()
    identity = _plain_database_identity(path)
    with path.open("rb") as stream:
        if (_identity(os.fstat(stream.fileno())) != identity
                or file_digest(stream, "sha256").hexdigest() != expected_sha256):
            raise ValueError("Geography roster file differs from its expected SHA-256")
        with closing(readonly_connection(path, immutable=True)) as connection:
            if (connection.execute("PRAGMA application_id").fetchone() != (_APPLICATION_ID,)
                    or connection.execute("PRAGMA user_version").fetchone() != (_SCHEMA_VERSION,)):
                raise ValueError("Unsupported geography roster file format")
            if (_identity(os.fstat(stream.fileno())) != identity or _plain_database_identity(path) != identity):
                raise ValueError("Geography roster file changed while opening")
            yield GeographyRosters(connection)


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _plain_database_identity(path: Path) -> tuple[int, int, int, int, int]:
    info = path.lstat()
    if not S_ISREG(info.st_mode) or any(os.path.lexists(str(path) + suffix) for suffix in ("-wal", "-shm", "-journal")):
        raise ValueError("Geography rosters require a regular, checkpointed SQLite artifact")
    return _identity(info)
