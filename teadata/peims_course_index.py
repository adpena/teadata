"""A pinned, rebuildable SQLite projection for bounded PEIMS course selections."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from io import StringIO
from itertools import chain
import os
from pathlib import Path
import re
import sqlite3
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, BinaryIO, Literal

from ._sqlite import verified_database
from .manifest import ContractError
from .peims_course_sources import DATASET, SOURCE_RECORD
from .source_predicates import _recorded_query
from .source_series import _YEAR_COLUMN, _append_series_sheet, _series_ids, _series_provenance
from .source_views import _CSVStream, _plan_export, _write_csv_bundle, _write_excel_rows

if TYPE_CHECKING:
    from openpyxl import Workbook

    from .export_provenance import ExportProvenance
    from .release import DataRelease
    from .source_views import SourceTable, _ExportSelection

FIELD = "course_enrollment_count"
RAW = "peims_raw_" + FIELD
STATUS = "peims_status_" + FIELD
_SCOPE = {"state": 0, "district": 1}
_SCOPE_COLUMNS = {_YEAR_COLUMN, "school_year", "course_code", FIELD, RAW, STATUS, SOURCE_RECORD}
_SCOPE_TYPE = Literal["district", "state"]
_APPLICATION_ID = 0x54444349
_SCHEMA_VERSION = 1
_COURSE = re.compile(r"[A-Z0-9]{8}")
_DISTRICT = re.compile(r"[0-9]{6}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_FILTER = "course_code == @course"
_MAX_DISTRICTS = 40
_SCHEMA = """
CREATE TABLE release_pin (sha256 TEXT NOT NULL);
CREATE TABLE edition (scope INTEGER, year INTEGER, source_year TEXT NOT NULL,
    PRIMARY KEY (scope, year)) WITHOUT ROWID;
CREATE TABLE observation (scope INTEGER, year INTEGER, course TEXT, district INTEGER,
    source_row INTEGER NOT NULL, amount INTEGER, raw_override TEXT, suppressed INTEGER NOT NULL,
    PRIMARY KEY (scope, year, course, district)) WITHOUT ROWID;
"""


def _stored_row(row: dict[str, Any], scope: _SCOPE_TYPE, year: int, label: str) -> tuple[Any, ...]:
    district = row.get("district_number", "")
    course, source_row, amount, raw, status = (row[name] for name in ("course_code", SOURCE_RECORD, FIELD, RAW, STATUS))
    if (row["school_year"] != label or not isinstance(course, str) or not _COURSE.fullmatch(course)
            or type(source_row) is not int or source_row < 1
            or (scope == "district" and (not isinstance(district, str) or not _DISTRICT.fullmatch(district)
                                         or district == "000000"))):
        raise ContractError("Course index received an invalid publisher key")
    if (status == "observed" and type(amount) is int and isinstance(raw, str) and raw.isascii()
            and raw.isdecimal() and (raw.lstrip("0") or "0") == str(amount)):
        override = raw if raw != str(amount) else None
    elif status == "suppressed" and amount is None and raw == "-999":
        override = raw
    else:
        raise ContractError("Course index received an unsupported publisher value")
    return _SCOPE[scope], year, course, int(district or "0"), source_row, amount, override, int(status == "suppressed")


def _source_projection(view: SourceTable, scope: _SCOPE_TYPE) -> tuple[_ExportSelection, tuple[str, ...]]:
    selection = _plan_export(view.release, view.edition, restore_original_headers=False, selected_fields=(FIELD,),
                             include_support=True, period=view.measurement_period)
    columns = (_YEAR_COLUMN, *selection.columns)
    expected = _SCOPE_COLUMNS | ({"district_number"} if scope == "district" else set())
    if set(columns) != expected or len(columns) != len(expected):
        raise ContractError("Course index requires the reviewed publisher projection")
    return selection, columns


def build_course_index(release: DataRelease, path: str | Path) -> None:
    """Compile district and direct Texas observations from an exact verified release."""
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(path)
    with TemporaryDirectory(dir=path.parent, prefix=".course-index-") as staging:
        database = Path(staging) / "courses.sqlite3"
        try:
            with closing(sqlite3.connect(database)) as db, db:
                db.execute(f"PRAGMA application_id={_APPLICATION_ID}")
                db.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
                db.executescript(_SCHEMA)
                db.execute("INSERT INTO release_pin VALUES (?)", (release.manifest_sha256,))
                for scope in ("state", "district"):
                    years = release.observation_years(DATASET, FIELD, scope=scope)[FIELD]
                    if not years:
                        raise ContractError(f"No PEIMS course enrollment editions for {scope}")
                    for year in years:
                        view = release.observations(DATASET, FIELD, scope=scope, school_year=year)
                        projection, _ = _source_projection(view, scope)
                        period = view.measurement_period
                        if period is None or (label := period.label) is None:
                            raise ContractError("Course edition requires an explicit measurement period")
                        db.execute("INSERT INTO edition VALUES (?, ?, ?)", (_SCOPE[scope], year, label))
                        row_count = 0
                        with closing(view.iter_batches()) as batches:
                            for batch in batches:
                                db.executemany("INSERT INTO observation VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                               (_stored_row(row, scope, year, label) for row in batch.to_pylist()))
                                row_count += batch.num_rows
                        if row_count != projection.artifact.contract["rows"]:
                            raise ContractError("Course index row count differs from the pinned edition")
                if db.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                    raise ContractError("Course index failed SQLite integrity verification")
                db.commit()
                db.execute("VACUUM")
        except sqlite3.Error as error:
            raise ContractError("Could not build the course index") from error
        # The checkpointed product is visible only after every edition succeeds.
        os.link(database, path)


@dataclass(frozen=True, slots=True)
class CourseEnrollment:
    """One published enrollment observation with its original token and row."""

    year: int
    district: str | None
    school_year: str
    code: str
    enrollment: int | None
    raw: str
    status: Literal["observed", "suppressed"]
    source_record: int

    def values(self, columns: tuple[str, ...]) -> tuple[int | str | None, ...]:
        row = {_YEAR_COLUMN: self.year, "district_number": self.district, "school_year": self.school_year,
               "course_code": self.code, FIELD: self.enrollment, RAW: self.raw, STATUS: self.status,
               SOURCE_RECORD: self.source_record}
        return tuple(row[name] for name in columns)


@dataclass(frozen=True, slots=True)
class CourseEnrollmentSelection:
    """A bounded result with the same provenance as the ordinary source series."""

    scope: _SCOPE_TYPE
    years: tuple[int, ...]
    districts: tuple[str, ...] | None
    columns: tuple[str, ...]
    rows: tuple[CourseEnrollment, ...]
    _sources: tuple[tuple[int, ExportProvenance], ...]

    def to_dict(self) -> dict[str, Any]:
        return _series_provenance(DATASET, (FIELD,), self.scope, self.years, self.districts, self._sources)

    def _append_sheet(self, workbook: Workbook, *, rows: int | None = None) -> None:
        _append_series_sheet(workbook, self.to_dict(), self._sources, rows=rows)

    def to_excel(self, path_or_buf: str | Path | BinaryIO) -> None:
        _write_excel_rows(self, (self.columns,), len(self.columns), len(self.rows),
                          (row.values(self.columns) for row in self.rows), path_or_buf)

    def to_csv_bundle(self, path_or_buf: str | Path | BinaryIO) -> None:
        def write_data(output: _CSVStream) -> None:
            line = StringIO(newline="")
            writer = csv.writer(line)
            for row in chain((self.columns,), (record.values(self.columns) for record in self.rows)):
                writer.writerow(row)
                output.write(line.getvalue().encode("utf-8"))
                line.seek(0)
                line.truncate()

        _write_csv_bundle(self, write_data, path_or_buf)


def select_course_enrollment(release: DataRelease, path: str | Path, *, expected_sha256: str, scope: _SCOPE_TYPE,
                             years: Sequence[int], course: str,
                             districts: Sequence[int | str] | None = None) -> CourseEnrollmentSelection:
    """Read a single course, at most two years and at most 40 district identifiers."""
    if scope not in _SCOPE or not isinstance(course, str) or not _COURSE.fullmatch(course):
        raise ValueError("Select district or state scope and an eight-character TEA course code")
    if not isinstance(expected_sha256, str) or not _DIGEST.fullmatch(expected_sha256):
        raise ValueError("PEIMS course index requires a pinned SHA-256")
    if not isinstance(years, Sequence) or isinstance(years, (str, bytes)):
        raise ValueError("School years must be a sequence of integer ending years")
    years = tuple(years)
    if not 1 <= len(years) <= 2 or any(type(year) is not int for year in years) or len(set(years)) != len(years):
        raise ValueError("Select one or two distinct school years by their integer ending years")
    if districts is not None:
        if not isinstance(districts, Sequence) or isinstance(districts, (str, bytes)):
            raise ValueError("District IDs must be a sequence")
        districts = tuple(districts)
    if (scope == "district" and (districts is None or not 1 <= len(districts) <= _MAX_DISTRICTS)
            or scope == "state" and districts is not None):
        raise ValueError("Select one to 40 district IDs or the direct Texas report")
    ids = _series_ids(districts, scope) if districts is not None else None
    keys = tuple(map(int, ids)) if ids is not None else (0,)
    markers = ",".join("?" for _ in keys)
    sources = []
    results = []
    columns = None
    try:
        with verified_database(Path(path), expected_sha256=expected_sha256, application_id=_APPLICATION_ID,
                               schema_version=_SCHEMA_VERSION, label="PEIMS course index", reuse_verification=True) as db:
            db.execute("PRAGMA trusted_schema=OFF")
            if db.execute("SELECT sha256 FROM release_pin").fetchmany(2) != [(release.manifest_sha256,)]:
                raise ContractError("PEIMS course index belongs to a different data release")
            for year in years:
                view = release.observations(DATASET, FIELD, scope=scope, school_year=year)
                projection, current = _source_projection(view, scope)
                if columns is not None and columns != current:
                    raise ContractError("Course index projection differs between selected years")
                columns = current
                receipt = projection.provenance(
                    view.release, period=view.measurement_period, entity_build=view._entity_build, entity_ids=ids,
                    opaque_filter=False, query_recipes=(_recorded_query(_FILTER, {"course": course}),),
                )
                sources.append((year, receipt))
                edition = db.execute("SELECT source_year FROM edition WHERE scope=? AND year=?", (_SCOPE[scope], year)).fetchone()
                if edition is None or view.measurement_period is None or edition[0] != view.measurement_period.label:
                    raise ContractError("PEIMS course index is missing the pinned edition")
                for district, record, amount, raw, suppressed in db.execute(
                    "SELECT district, source_row, amount, raw_override, suppressed FROM observation "
                    f"WHERE scope=? AND year=? AND course=? AND district IN ({markers}) ORDER BY source_row",
                    (_SCOPE[scope], year, course, *keys),
                ):
                    results.append(CourseEnrollment(
                        year, f"{district:06d}" if scope == "district" else None, edition[0], course, amount,
                        str(amount) if raw is None else raw, "suppressed" if suppressed else "observed", record,
                    ))
    except sqlite3.Error as error:
        raise ContractError("Could not read the pinned course index") from error
    assert columns is not None
    return CourseEnrollmentSelection(scope, years, ids, columns, tuple(results), tuple(sources))
