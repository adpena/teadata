"""Qualify the final 2025–26 School FIRST and university Charter FIRST workbooks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from html import unescape
from importlib.metadata import version
from pathlib import Path
import platform
import re
import sys
from tempfile import TemporaryDirectory
from typing import Any, NamedTuple
from zipfile import ZipFile

from openpyxl import load_workbook
from openpyxl.utils.datetime import from_excel
import pyarrow as pa
import pyarrow.parquet as pq

from .analytical import write_leaf_artifact
from .column_schema import crosswalk_dataset, parse_column_schema, reader_options
from .dataset_metadata import ArtifactRef, DataPeriod, DatasetMetadata, EditionKey, ResolvedDataset
from .financial_leaves import NS, _literal_batch, _literal_token, _package, _xml
from .identifiers import canonical_source_identifier
from .manifest import BuildRecorder, ContractError, canonical_json
from .native_witnesses import _Rows
from .provenance import ProvenanceStore, validate_reference_records
from .release import DataRelease, _install_candidate, publish_release
from .teadata_config import Config, load_config_bytes
from .values import ValueStatus

EDITION = "2026"
PERIOD = DataPeriod("fiscal_year", 2024, 2025)
_DATASET_PREFIX = "first_financial_accountability_"
_CONFIG = Path(__file__).with_name("teadata_sources.yaml")
_HEADERS = ("District#", "District_Name", "Status", "Rating", "Score", "Last Updated Date", "Region")
_FIELDS = ("district_number", "district_name", "first_status", "first_rating", "first_score",
           "first_row_updated", "first_region")
_DISTRICT_GRADES = {
    "A = Superior Achievement": "A", "B = Above Standard Achievement": "B",
    "C = Meets Standard Achievement": "C", "F = Substandard Achievement": "F",
}
_PASS_FAIL = {"Pass": "Pass", "Fail": "Fail"}
_VALUE_POLICY = "first_university_rating_2026"
_SUMMARY = "first_statewide_summary"
_SUPPORT = {
    "fields": {
        "district_number": {"raw": "first_raw_district_number"},
        "first_score": {"raw": "first_raw_score", "status": "first_score_status"},
        "first_row_updated": {"raw": "first_raw_row_updated"},
        "first_region": {"raw": "first_raw_region"},
    },
    "context": ["first_source_row", "first_source_sheet"],
}


@dataclass(frozen=True)
class _Population:
    scope: str
    sheet: str
    workbook_title: str
    roster_heading: str
    roster_page_count: int
    year_control: tuple[str, str]

    @property
    def dataset(self) -> str:
        return _DATASET_PREFIX + self.scope

    @property
    def university(self) -> bool:
        return self.scope == "university_charter"


_POPULATIONS = (
    _Population(
        "district", "FIRST Ratings", "2025-2026 FIRST Ratings — Final",
        "2025-2026 Ratings Based on School Year 2024-2025 Data - District Status Summary",
        21, ("ctl00$AppBar$ddlYear", "2024"),
    ),
    _Population(
        "university_charter", "CharterFIRST Ratings", "2025-2026 CharterFIRST Ratings — Final      (University Charters)",
        "2025-2026 Ratings Based on Fiscal Year 2025 Data - University Charter School Status Summary",
        1, ("ctl00$ddlYear", "2025"),
    ),
)


class _Rating(NamedTuple):
    district_number: str
    district_name: str
    first_status: str
    first_rating: str
    first_score: int | None
    first_row_updated: datetime
    first_region: int
    first_source_row: int
    first_source_sheet: str
    first_raw_district_number: str
    first_raw_score: str | None
    first_score_status: str
    first_raw_row_updated: str
    first_raw_region: str


class _RosterRows(_Rows):
    def handle_starttag(self, tag, attrs):
        super().handle_starttag(tag, attrs)
        if tag == "img" and self.cell is not None:
            attributes = dict(attrs)
            status = attributes.get("alt")
            if status == "Status":
                image = attributes.get("src") or ""
                status = {"passed.jpg": "Passed", "failed.jpg": "Failed"}.get(image.rsplit("/", 1)[-1])
            if status in ("Passed", "Failed"):
                self.cell.append(status)


def _literal_rows(path: Path, population: _Population) -> pa.RecordBatch:
    with ZipFile(path) as archive:
        sheets, strings = _package(archive, literal=True)
        if tuple(sheets) != (population.sheet,):
            raise ContractError(f"Unexpected FIRST worksheet for {population.scope}")
        xml = _xml(archive, sheets[population.sheet], literal=True)
        rows = xml.findall(NS + "sheetData/" + NS + "row")
        merges = [cell.attrib["ref"] for cell in xml.findall(NS + "mergeCells/" + NS + "mergeCell")]
        if (len(rows) < 3 or merges != ["A1:G1"] or rows[0].get("r") != "1" or rows[1].get("r") != "2"
                or len(rows[0]) == 0 or rows[0][0].get("r") != "A1" or _literal_token(rows[0][0], strings) != population.workbook_title
                or [cell.get("r") for cell in rows[1]] != [f"{column}2" for column in "ABCDEFG"]
                or tuple(_literal_token(cell, strings) for cell in rows[1]) != _HEADERS
                or any(cell.tag != NS + "c" or re.fullmatch(r"[A-G][0-9]+", cell.get("r", "")) is None
                       for row in rows for cell in row)):
            raise ContractError(f"Unexpected FIRST workbook framing for {population.scope}")
        return _literal_batch(rows[1:], strings, list(_HEADERS), 2, len(rows) - 2)


def _rating(population: _Population, number: int, row: tuple, tokens: tuple, epoch: datetime) -> _Rating:
    cdn, name, status, rating, score, updated, region = row
    grades = _PASS_FAIL if population.university else _DISTRICT_GRADES
    grade = grades.get(rating)
    if (type(cdn) is not int or not isinstance(name, str) or not name.strip()
            or grade is None or status != ("Fail" if grade in ("F", "Fail") else "Pass")
            or type(region) is not int or not 1 <= region <= 20 or not isinstance(updated, datetime)
            or (score is not None if population.university else type(score) is not int or not 0 <= score <= 100)):
        raise ContractError(f"Invalid FIRST {population.scope} source row {number}")
    identifier = canonical_source_identifier(cdn, width=6)
    raw_cdn, raw_name, raw_status, raw_rating, raw_score, raw_date, raw_region = tokens
    if (identifier is None or raw_cdn is None or raw_date is None or raw_region is None
            or (int(raw_cdn), raw_name, raw_status, raw_rating, None if raw_score is None else int(raw_score),
                from_excel(float(raw_date), epoch=epoch), int(raw_region)) != row):
        raise ContractError(f"Literal and typed FIRST workbook values differ at {population.scope} row {number}")
    score_status = ValueStatus.NOT_APPLICABLE if population.university else ValueStatus.OBSERVED
    return _Rating(identifier[1:], name, status, rating, score, updated, region, number, population.sheet,
                   raw_cdn, raw_score, score_status.value, raw_date, raw_region)


def _workbook(path: Path, population: _Population, expected_rows: int) -> dict[str, _Rating]:
    literal = _literal_rows(path, population)
    if literal.num_rows != expected_rows:
        raise ContractError(f"FIRST workbook row count differs for {population.scope}")
    records: dict[str, _Rating] = {}
    with path.open("rb") as stream:
        book = load_workbook(stream, read_only=True, data_only=False, keep_links=False)
        try:
            sheet, = book.worksheets
            rows = sheet.values
            if sheet.title != population.sheet or next(rows)[0] != population.workbook_title or next(rows) != _HEADERS:
                raise ContractError(f"Typed FIRST worksheet framing differs for {population.scope}")
            for number, (row, raw) in enumerate(zip(rows, literal.to_pylist(), strict=True), 3):
                entry = _rating(population, number, row, tuple(raw[header] for header in _HEADERS), book.epoch)
                if entry.district_number in records:
                    raise ContractError(f"Duplicate FIRST workbook district {entry.district_number}")
                records[entry.district_number] = entry
        finally:
            book.close()
    return records


def _check_roster(population: _Population, pages: list[str], records: dict[str, _Rating]) -> None:
    seen: set[str] = set()
    for content in pages:
        if population.roster_heading not in " ".join(content.split()) or "Final Ratings" not in content:
            raise ContractError(f"Unexpected FIRST public roster for {population.scope}")
        for row in _RosterRows(content).rows:
            if not row or re.fullmatch(r"[0-9]{6}", row[0]) is None:
                continue
            key = row[0]
            entry = records.get(key)
            if entry is None or key in seen:
                raise ContractError(f"Unknown or repeated FIRST public roster district {key}")
            seen.add(key)
            icon = "Passed" if entry.first_status == "Pass" else "Failed"
            expected = ([key, str(entry.first_region), entry.district_name, entry.first_rating, icon]
                        if population.university else [key, entry.district_name, entry.first_rating, icon])
            if row != expected:
                raise ContractError(f"FIRST roster disagrees with workbook for {key}")
    if seen != records.keys():
        raise ContractError(f"FIRST public roster is incomplete for {population.scope}")


def _summary_controls(content: str) -> tuple[dict[str, int], dict[str, int]]:
    districts = {}
    for row in _Rows(content).rows:
        if len(row) == 5 and row[0][:1] in _DISTRICT_GRADES.values() and row[0][1:4] == " – ":
            grade = row[0][0]
            if grade in districts:
                raise ContractError(f"Duplicate FIRST newsroom district control for grade {grade}")
            districts[grade] = int(row[1].replace(",", ""))
    phrase = ("all five charter schools operated by a public institution of higher education "
              "received a final FIRST rating of Pass")
    if districts.keys() != set(_DISTRICT_GRADES.values()) or phrase not in " ".join(
        unescape(re.sub(r"<[^>]+>", "", content)).split()
    ):
        raise ContractError("FIRST newsroom lacks the district and university charter controls")
    return districts, {"Pass": 5}


def _receipt(record: dict, population: _Population, url: str, parameters: dict, *,
             requested_url: str | None = None) -> None:
    metadata = record.get("source_metadata", {})
    if (record.get("kind") != "acquisition" or record.get("dataset") != population.dataset
            or record.get("method") != "download" or record.get("requested_url") != (requested_url or url)
            or record.get("resolved_url") != url or record.get("requested_year") != int(EDITION)
            or record.get("resolved_year") != int(EDITION) or metadata.get("version") != EDITION
            or metadata.get("period") != PERIOD.to_dict() or metadata.get("title") != population.roster_heading
            or any(record.get("parameters", {}).get(key) != value
                   for key, value in {"http_status": 200, **parameters}.items())):
        raise ContractError(f"Unexpected FIRST acquisition receipt for {population.scope}")


def _admit(config: Config, store: ProvenanceStore, population: _Population) -> tuple[
    ResolvedDataset, str, str, set[str], dict[str, _Rating], dict[str, int]
]:
    dataset = population.dataset
    source = config.resolve_dataset(dataset, int(EDITION), reader_profile="source", exact=True)
    metadata = source.metadata
    options = reader_options(source.reader, "source")
    resource = config.column_contract(dataset, int(EDITION), reader_profile="source")
    policy = config.value_policies.get(metadata.value_policy) if metadata.value_policy else None
    if (not config.is_source_only(dataset, int(EDITION)) or not metadata.source_url or metadata.version != EDITION
            or metadata.period != PERIOD or metadata.published_at != "2026-11-05" or resource is None
            or options.get("worksheet") != population.sheet or len(metadata.provenance_ids) != 1
            or metadata.value_policy != (_VALUE_POLICY if population.university else None)
            or (population.university and (policy is None or policy.reference_id != _SUMMARY or set(policy.tokens) != {""}
                                          or policy.tokens[""].status != ValueStatus.NOT_APPLICABLE))):
        raise ContractError(f"Unexpected FIRST source configuration for {dataset}")
    roster = options.get("roster_receipts", [])
    if not isinstance(roster, list) or len(roster) != population.roster_page_count or len(set(roster)) != len(roster):
        raise ContractError(f"Unexpected FIRST roster receipt set for {dataset}")
    refs = {ref.reference_id: ref for ref in metadata.references}
    if refs.keys() != {"first_crosswalk", "first_final_notice", _SUMMARY, "first_basis_and_rules"} or any(
        len(ref.provenance_ids) != 1 or not ref.reviewed_sha256 for ref in refs.values()
    ):
        raise ContractError(f"Missing exact FIRST references for {dataset}")
    workbook_id, = metadata.provenance_ids
    roots = {workbook_id, *roster, *(ref.provenance_ids[0] for ref in refs.values())}
    graph = store.verify_record_closure(sorted(roots))
    validate_reference_records(source, graph.__getitem__, verify_all_claimed=True)
    url = metadata.source_url
    workbook = graph[workbook_id]
    _receipt(workbook, population, url, {"http_method": "POST", "scope": population.scope,
                                         "selection_controls": dict([population.year_control])})
    workbook_sha = workbook["artifact"]["sha256"]
    mapping, = crosswalk_dataset(parse_column_schema(resource["text"].encode()), dataset, EDITION)["tables"]
    if (mapping["table"] != population.sheet or mapping["input_sha256"] != workbook_sha
            or mapping["metric_period"] != PERIOD.to_dict()
            or tuple((f["code"], f["label"], f["field"], f["disposition"]) for f in mapping["fields"]) != tuple(
                (header, header, name, "identity" if name in _FIELDS[:2] else "fact")
                for header, name in zip(_HEADERS, _FIELDS, strict=True)
            )):
        raise ContractError(f"Unexpected FIRST crosswalk for {dataset}")
    read = store.verify_artifact
    entries = _workbook(read(workbook_sha), population, mapping["layout"]["last_data_record"] - 2)
    pages = []
    for page, identifier in enumerate(roster, 1):
        record = graph[identifier]
        if population.university:
            _receipt(record, population, url, {"http_method": "POST", "selector_control": "ctl00$ddlNonAdmin",
                                              "selector_value": "DistStatusUniversity"},
                     requested_url=url.partition("?")[0])
        else:
            _receipt(record, population, url, {"http_method": "GET" if page == 1 else "POST", "page": page,
                                              "role": "public_roster"})
        pages.append(read(record["artifact"]["sha256"]).read_text(encoding="utf-8"))
    _check_roster(population, pages, entries)
    summary = graph[refs[_SUMMARY].provenance_ids[0]]
    controls = _summary_controls(read(summary["artifact"]["sha256"]).read_text(encoding="utf-8"))
    grades = _PASS_FAIL if population.university else _DISTRICT_GRADES
    counts = Counter(grades[entry.first_rating] for entry in entries.values())
    if counts != controls[population.university]:
        raise ContractError(f"FIRST workbook counts disagree with the applicable newsroom control for {dataset}")
    return source, workbook_id, workbook_sha, roots, entries, dict(counts)


def _project(entries: dict[str, _Rating]) -> pa.Table:
    types = (pa.string(), pa.string(), pa.string(), pa.string(), pa.int16(), pa.timestamp("ms"), pa.int8(),
             pa.int32(), pa.string(), pa.string(), pa.string(), pa.string(), pa.string(), pa.string())
    return pa.table(dict(zip(_Rating._fields,
        (pa.array(values, type=kind) for values, kind in zip(zip(*entries.values()), types, strict=True)), strict=True)))


def materialize_first_ratings(store: ProvenanceStore, output: Path) -> dict[str, Any]:
    """Build both reconciled populations together from retained publisher evidence."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("FIRST candidate requires a new exclusive output directory")
    raw_config = _CONFIG.read_bytes()
    config = load_config_bytes(raw_config, path=_CONFIG)
    district, university = admitted = [_admit(config, store, population) for population in _POPULATIONS]
    if district[4].keys() & university[4].keys():
        raise ContractError("FIRST workbook district IDs overlap the university charter population")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".first-ratings-", dir=output.parent) as directory:
        stage = Path(directory) / "bundle"
        stage.mkdir(mode=0o700)
        target = ProvenanceStore(stage / "provenance")
        for identifier in sorted(district[3] | university[3]):
            target.copy_record_from(store, identifier)
        captured = Path(directory) / _CONFIG.name
        captured.write_bytes(raw_config)
        config_id = target.record_acquisition(captured, ResolvedDataset(
            "reference:first:config", "reference", int(EDITION), int(EDITION), "",
            DatasetMetadata(title="Exact FIRST source configuration", version=EDITION),
        ))
        recorder = BuildRecorder(config, target)
        toolchain = {"python": sys.version, "pyarrow": version("pyarrow"), "openpyxl": version("openpyxl"),
                     "platform": platform.platform(), "architecture": platform.machine()}
        results, projections = [], {}
        for population, (source, workbook_id, digest, roots, entries, counts) in zip(_POPULATIONS, admitted, strict=True):
            table = _project(entries)
            key = EditionKey(population.dataset, EDITION, EditionKey.source_revision((digest,)), "district",
                             population.sheet, "one publisher row per district number")
            recipe = {"operation": "first_final_ratings", "worksheet": population.sheet,
                      "population": population.scope, "raw_cell_encoding": "decoded_openxml_cell_token"}
            contract = recorder.leaf_contract(key, source, native_schema=table.schema.serialize().to_pybytes(),
                columns=table.column_names, key_columns=["district_number"], rows=table.num_rows,
                inputs=[digest], recipe=recipe, toolchain=toolchain, column_support=_SUPPORT)
            contract_json = canonical_json(contract)
            leaf = stage / f"{sha256(contract_json.encode()).hexdigest()}.parquet"
            physical = write_leaf_artifact(leaf, table.to_batches(max_chunksize=512), contract=contract, batch_bytes=8 * 1024**2)
            if not table.equals(pq.read_table(leaf).replace_schema_metadata(None), check_metadata=True):
                raise ContractError("FIRST native readback differs from publisher projection")
            processing = target.record_processing(dataset=population.dataset, inputs=[*sorted(roots), config_id],
                outputs={"leaf": leaf}, processor="teadata.first_ratings", code_version=recorder.processor_sha256,
                parameters={"leaf_contract_sha256": leaf.stem, "recipe": recipe, "toolchain": toolchain},
                validation={"literal_and_typed_workbook_parity": True, "public_roster_parity": True,
                            "applicable_newsroom_controls_match": True, "exact_native_projection": True,
                            "rows": table.num_rows, "release_activation": False})
            artifact = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name,
                                   contract_json, processing, "leaf")
            recorder.record_edition(key, source, artifacts=[artifact], acquisition_ids=[workbook_id])
            results.append({"edition": key.to_dict(), "rows": table.num_rows, "ratings": counts,
                            "processing_record_id": processing})
            projections[key] = table
        catalog = recorder.catalog(schema_version=4)
        manifest = publish_release(stage, catalog, schema_version=4)
        release = DataRelease(manifest, offline=True)
        release.verify()
        for key, table in projections.items():
            public = release.table(key).to_arrow().select(table.column_names)
            if not public.equals(table):
                raise ContractError("FIRST public Arrow readback differs from publisher projection")
        result = {"status": "unpublished_native_first_ratings", "build_id": catalog.build_id,
                  "manifest_sha256": release.manifest_sha256, "editions": results, "release_activation": False}
        (stage / "MATERIALIZATION.json").write_text(canonical_json(result) + "\n", encoding="utf-8")
        _install_candidate(stage, output)
    return result
