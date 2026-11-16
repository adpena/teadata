"""Offline, exact-token district facility observations from the retained TEA revision.

Publisher records are not campus entities or corrected capacity measurements.
Contact cells are never materialized; original bytes remain private evidence.
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pyarrow as pa

from .analytical import write_leaf_artifact
from .column_schema import crosswalk_dataset, parse_column_schema
from .dataset_metadata import ArtifactRef, DatasetMetadata, EditionKey, ResolvedDataset
from .identifiers import canonical_source_identifier
from .manifest import BuildRecorder, ContractError, canonical_json, file_identity
from .provenance import ProvenanceStore, validate_reference_records
from .release import _install_candidate, publish_release
from .teadata_config import load_config_bytes

WORKBOOK_SHA = "a71e0919f2451e2343be2600c8bccf2cea72fffa21641d4a241ce18c6528b944"
WORKBOOK_RECEIPT = "40c0a709aa04f5ef871292982535b08fe9fc7a7dcf51da3e1cd1da6033241e48"
EXTRACTION_RECEIPT = "74f9b06dd69a379c0e084fc0d1b3b0918ba8275f07b06aa42eb29d6e67f01b58"
PUBLISHER_ROOTS = (WORKBOOK_RECEIPT,
    "c077f0203f1deec279cc11aacc191bf90494d7bf659104e20c56b6e4137a57cd",
    "9e802b447cfb7c3ad53a031a25c0b8ac1b4f67ef06e3883366d3105fcc1bf361")
DATASETS = ("facility_usage", "facility_usage_submission")
SHEET = "Facilities Data"
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
CONFIG = Path(__file__).with_name("teadata_sources.yaml")
EMAIL = re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
# Coordinate, canonical field, physical type, unit, literal publisher heading.
FIELDS = (
    ('A', 'district_name', 'string', 'text', 'Name of the district'),
    ('B', 'district_number', 'string', 'text', 'County District Number'),
    ('D', 'submission_type', 'string', 'text', 'Type of Submission'),
    ('E', 'active_used_facility', 'string', 'text', 'Active used facility'),
    ('F', 'property_id', 'string', 'text', 'Property ID'),
    ('G', 'street_address', 'string', 'text', 'Street Address'),
    ('H', 'city_state', 'string', 'text', 'City, State'),
    ('I', 'zip_code', 'string', 'text', 'Zip Code'),
    ('J', 'reported_org_code', 'string', 'text', 'TSDS/PEIMS Org Code'),
    ('K', 'building_type_recoded', 'string', 'text', 'Type of Building (Recoded)'),
    ('L', 'classroom_count', 'int32', 'count', 'Number of Classrooms'),
    ('M', 'original_year_completed', 'int16', 'years', 'Original Year Completed'),
    ('N', 'most_recent_renovation_year', 'int16', 'years', 'Year of Most Recent Renovation Completed'),
    ('O', 'houses_portables', 'string', 'text', 'Does the facility house portables'),
    ('P', 'portable_count', 'int32', 'count', 'How many Portables'),
    ('Q', 'square_footage', 'decimal128(19,12)', 'square_feet', 'Square Footage'),
    ('R', 'acreage', 'decimal128(24,18)', 'acres', 'Acreage'),
    ('S', 'ownership_status', 'string', 'text', 'Ownership Status'),
    ('T', 'shared_usage', 'string', 'text', 'Shared Usage'),
    ('U', 'shared_building_type_1', 'string', 'text', 'Shared Building Type 1 (Recoded)'),
    ('V', 'shared_building_type_2', 'string', 'text', 'Shared Building Type 2 (Recoded)'),
    ('W', 'shared_building_type_3', 'string', 'text', 'Shared Building Type 3 (Recoded)'),
    ('X', 'shared_building_type_4', 'string', 'text', 'Shared Building Type 4 (Recoded)'),
    ('Y', 'facility_usage', 'string', 'text', 'Facility Usage'),
    ('Z', 'grades_for_facility', 'string', 'text', 'Grades for this Facility'),
    ('AA', 'maximum_instructional_capacity', 'decimal128(18,13)', 'count', 'Maximum Instructional Capacity'),
    ('AB', 'adjusted_instructional_capacity', 'decimal128(18,13)', 'count', 'Adjusted Instructional Capacity'),
    ('AC', 'reported_current_enrollment', 'decimal128(9,1)', 'count', 'Total Current Enrollment'),
    ('AD', 'reported_campus_label', 'string', 'text', 'Campus Name (Campus ID)'),
    ('AE', 'map_esc_region', 'int8', None, 'ESC Region (Map)'),
    ('AF', 'submission_status', 'string', 'text', 'Submission Status'),
 )
SUBMISSION_COORDS = ("A", "B", "AE", "AF")
CONTEXT = ("publisher_worksheet", "publisher_worksheet_row", "record_kind",
           "source_cell_types_json", "quality_conformance_json")
GROUP_CONTEXT = ("publisher_worksheet", "source_group_row_count", "source_group_min_row",
                 "source_group_max_row", "source_group_rows_sha256")
CATEGORY_COLUMNS = {"D": "E", "E": "D", "K": "A", "O": "D", "S": "C", "T": "D",
                    "U": "F", "V": "F", "W": "F", "X": "F", "Y": "B", "Z": "G"}
LIMITS = ["As-published 2025-26 bulk revision; correction coverage, finality and publication date unknown.",
          "Facility-reported enrollment mixes Fall Snapshot and average daily students served.",
          "No campus relationship, corrected series, utilization or additive capacity qualification.",
          "Status-only and absent collection districts do not establish zero facilities."]


def physical_type(name: str) -> pa.DataType:
    if name.startswith("decimal128("):
        precision, scale = map(int, name[11:-1].split(","))
        return pa.decimal128(precision, scale)
    return {"string": pa.string(), "int8": pa.int8(), "int16": pa.int16(), "int32": pa.int32()}[name]


def _cell(cell: ET.Element | None) -> tuple[str | None, str]:
    if cell is None:
        return None, "absent"
    if cell.find("s:f", NS) is not None:
        raise ContractError("Facility source contains an unexpected formula")
    kind = cell.get("t", "n")
    if kind == "inlineStr":
        inline = cell.find("s:is", NS)
        text = "" if inline is None else "".join(inline.itertext())
        return text, "inlineStr_empty" if text == "" else "inlineStr"
    if kind == "n":
        value = cell.findtext("s:v", None, NS)
        return value, "n_empty" if value in (None, "") else "n"
    raise ContractError("Facility source contains an unreviewed XML cell type")


def _cells(row: ET.Element) -> dict[str, ET.Element]:
    result = {}
    for cell in row:
        address = cell.get("r", "")
        match = re.fullmatch(r"([A-Z]+)([1-9][0-9]*)", address)
        if not match or match[1] in result or match[2] != row.get("r"):
            raise ContractError("Facility XML cell coordinates are inconsistent")
        result[match[1]] = cell
    return result


def _worksheet(archive: ZipFile, path: str) -> list[ET.Element]:
    if archive.getinfo(path).file_size > 64 * 1024**2:
        raise ContractError("Facility worksheet exceeds the reviewed bounded reader")
    return ET.fromstring(archive.read(path)).findall("s:sheetData/s:row", NS)


def read_workbook(path: Path) -> tuple[list[tuple[int, dict[str, tuple[str | None, str]], bool]], dict[str, set[str]]]:
    """Read original cells without retaining contact values in the returned rows."""
    with ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook.find("s:sheets", NS)
        if sheets is None:
            raise ContractError("Facility workbook sheet directory is missing")
        names = [sheet.get("name") for sheet in sheets]
        if names != ["README", SHEET, "Validation", "Lookup"]:
            raise ContractError("Facility workbook sheet identity differs")
        rows = _worksheet(archive, "xl/worksheets/sheet2.xml")
        expected = {coord: header for coord, _, _, _, header in FIELDS} | {"C": "Contact Email Address"}
        if not rows or {coord: _cell(cell)[0] for coord, cell in _cells(rows[0]).items()} != expected:
            raise ContractError("Facility workbook original 32-column header differs")
        output = []
        previous = 1
        for row in rows[1:]:
            ordinal = int(row.attrib["r"])
            if ordinal != previous + 1:
                raise ContractError("Facility worksheet rows are missing, repeated or reordered")
            previous = ordinal
            cells = _cells(row)
            if set(cells) - expected.keys():
                raise ContractError("Facility source has an unexpected column")
            contact_blank = _cell(cells.get("C"))[0] in (None, "")
            values = {coord: _cell(cells.get(coord)) for coord, *_ in FIELDS}
            output.append((ordinal, values, contact_blank))
        domains: dict[str, set[str]] = {column: set() for column in "ABCDEFG"}
        for row in _worksheet(archive, "xl/worksheets/sheet3.xml")[1:]:
            for coord, cell in _cells(row).items():
                value = _cell(cell)[0]
                if coord not in domains:
                    raise ContractError("Facility validation domains have changed")
                if value not in (None, ""):
                    domains[coord].add(value)
        return output, domains


def _value(raw: str | None, kind: str, physical: str, coordinate: str, ordinal: int) -> Any:
    if raw in (None, ""):
        return None
    if physical == "string":
        if EMAIL.search(raw):
            raise ContractError(f"Public facility text requires privacy review at {coordinate}{ordinal}")
        if coordinate == "B":
            try:
                value = canonical_source_identifier(raw, width=6)
            except ValueError:
                raise ContractError(f"Facility district identifier is invalid at row {ordinal}") from None
            if value is None:
                raise ContractError("Facility district key is missing")
            return value
        return raw
    if kind != "n":
        raise ContractError(f"Facility numeric cell type differs at {coordinate}{ordinal}")
    try:
        value = Decimal(raw)
        if not value.is_finite():
            raise ValueError
        sign, digits, exponent = value.as_tuple()
        if not isinstance(exponent, int):
            raise ValueError
        if physical.startswith("decimal"):
            dtype = physical_type(physical)
            if max(0, -exponent) > dtype.scale or max(0, len(digits) + exponent) > dtype.precision - dtype.scale:
                raise ValueError
            # Exact Arrow rescaling pads zeros only; no quantize/rounding context.
            scalar = pa.scalar(value, type=dtype).as_py()
            if scalar != value:
                raise ValueError
            return value
        if value != value.to_integral_value():
            raise ValueError
        integer = int(value)
        if pa.scalar(integer, type=physical_type(physical)).as_py() != integer:
            raise ValueError
        return integer
    except (InvalidOperation, ValueError, OverflowError, pa.ArrowException):
        raise ContractError(f"Facility numeric token is outside the exact contract at {coordinate}{ordinal}") from None


def project_rows(rows, domains: dict[str, set[str]]) -> tuple[pa.Table, dict[str, Any]]:
    records = []
    for ordinal, cells, contact_blank in rows:
        status = cells["AF"][0]
        if status not in {"SUBMITTED", "INCOMPLETE", "NOT SUBMITTED"}:
            raise ContractError(f"Facility submission status differs at row {ordinal}")
        status_only = status != "SUBMITTED"
        if any(cells[c][0] in (None, "") for c in ("A", "B", "AE", "AF")):
            raise ContractError("Facility district roster fields must be populated")
        if status_only and (not contact_blank or any(cells[c][0] not in (None, "")
                            for c, *_ in FIELDS if c not in SUBMISSION_COORDS)):
            raise ContractError("A status-only district contains facility payload")
        record: dict[str, Any] = {}
        violations = []
        for coord, name, physical, _, _ in FIELDS:
            raw, kind = cells[coord]
            record[name] = _value(raw, kind, physical, coord, ordinal)
            record["raw_" + name] = raw
            if physical != "string":
                record["status_" + name] = "missing" if raw in (None, "") else "observed"
            if raw not in (None, "") and coord in CATEGORY_COLUMNS:
                values = [v.strip() for v in raw.split(",")] if coord == "Z" else [raw]
                if not all(value in domains[CATEGORY_COLUMNS[coord]] for value in values):
                    violations.append(coord)
        record.update(publisher_worksheet=SHEET, publisher_worksheet_row=ordinal,
                      record_kind="status_only" if status_only else "facility_observation",
                      source_cell_types_json=canonical_json({c: cells[c][1] for c, *_ in FIELDS}),
                      quality_conformance_json=canonical_json({"outside_validation_columns": sorted(violations)}))
        records.append(record)
    return _table(records, FIELDS, CONTEXT)


def _table(records: list[dict[str, Any]], fields, context) -> tuple[pa.Table, dict[str, Any]]:
    types: dict[str, pa.DataType] = {}
    support: dict[str, Any] = {"fields": {}, "context": list(context)}
    for _, name, physical, _, _ in fields:
        types[name] = physical_type(physical)
        types["raw_" + name] = pa.string()
        support["fields"][name] = {"raw": "raw_" + name}
        if physical != "string":
            types["status_" + name] = pa.string()
            support["fields"][name]["status"] = "status_" + name
    for name in context:
        types[name] = pa.int32() if name in {"publisher_worksheet_row", "source_group_row_count",
                                           "source_group_min_row", "source_group_max_row"} else pa.string()
    return pa.Table.from_pylist(records, schema=pa.schema(list(types.items()))), support


def submission_table(facilities: pa.Table) -> tuple[pa.Table, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    names = [name for coord, name, *_ in FIELDS if coord in SUBMISSION_COORDS]
    selected = [*names, *("raw_" + name for name in names), "status_map_esc_region", "publisher_worksheet_row"]
    for row in facilities.select(selected).to_pylist():
        groups.setdefault(row["district_number"], []).append(row)
    result = []
    for identifier, rows in sorted(groups.items()):
        for name in selected[:-1]:
            if len({row[name] for row in rows}) != 1 or rows[0][name] is None:
                raise ContractError("Facility district grouping has inconsistent source values")
        ordinals = sorted(row["publisher_worksheet_row"] for row in rows)
        if len(set(ordinals)) != len(ordinals):
            raise ContractError("Facility source row locators are duplicated")
        record = {name: rows[0][name] for name in selected[:-1]}
        record.update(publisher_worksheet=SHEET, source_group_row_count=len(rows),
                      source_group_min_row=ordinals[0], source_group_max_row=ordinals[-1],
                      source_group_rows_sha256=sha256(canonical_json(ordinals).encode()).hexdigest())
        result.append(record)
    return _table(result, [f for f in FIELDS if f[0] in SUBMISSION_COORDS], GROUP_CONTEXT)


def source_inputs(store: ProvenanceStore) -> Path:
    records = store.verify_record_closure(PUBLISHER_ROOTS)
    raw = records[WORKBOOK_RECEIPT]
    extraction = records[EXTRACTION_RECEIPT]
    if (raw.get("derived_from") != EXTRACTION_RECEIPT or raw["artifact"]["sha256"] != WORKBOOK_SHA
            or raw.get("method") != "local_observation"
            or extraction["outputs"]["workbook"]["sha256"] != WORKBOOK_SHA):
        raise ContractError("Facility retained workbook extraction lineage differs")
    return store.verify_artifact(WORKBOOK_SHA)


def materialize_facility_usage(store: ProvenanceStore, output: Path) -> dict[str, Any]:
    """Build both tables offline; never activate a normal release or upload raw data."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("Facility candidate requires a new exclusive directory")
    source_path = source_inputs(store)
    rows, domains = read_workbook(source_path)
    facilities, facility_support = project_rows(rows, domains)
    submission, submission_support = submission_table(facilities)
    if (facilities.num_rows != 17834 or submission.num_rows != 1017
            or Counter(facilities["record_kind"].to_pylist()) != {"facility_observation": 17763, "status_only": 71}
            or Counter(submission["submission_status"].to_pylist()) != {"SUBMITTED": 946, "INCOMPLETE": 36, "NOT SUBMITTED": 35}):
        raise ContractError("Facility retained source population differs")
    config_bytes = CONFIG.read_bytes()
    cfg = load_config_bytes(config_bytes, path=CONFIG)
    sources = {name: cfg.resolve_dataset(name, 2026, reader_profile="source", exact=True) for name in DATASETS}
    roots = list(PUBLISHER_ROOTS)
    for dataset, source in sources.items():
        if (not cfg.is_source_only(dataset, 2026) or source.metadata.version != "2026"
                or source.reader["profiles"]["source"]["acquisition_id"] != WORKBOOK_RECEIPT):
            raise ContractError("Facility source configuration differs")
        validate_reference_records(source, store.read_record)
        resource = cfg.column_contract(dataset, 2026, reader_profile="source")
        if resource is None:
            raise ContractError("Facility public header mapping is missing")
        mapping = crosswalk_dataset(parse_column_schema(resource["text"].encode()), dataset, "2026")["tables"]
        expected_fields = FIELDS if dataset == DATASETS[0] else [f for f in FIELDS if f[0] in SUBMISSION_COORDS]
        if (len(mapping) != 1 or mapping[0]["input_sha256"] != WORKBOOK_SHA
                or [f["field"] for f in mapping[0]["fields"]] != [f[1] for f in expected_fields]
                or [f["code"] for f in mapping[0]["fields"]] != [f[4] for f in expected_fields]):
            raise ContractError("Facility public header mapping differs from reviewed projection")
        roots.extend(rid for ref in source.metadata.references for rid in ref.provenance_ids)
    roots = list(dict.fromkeys(roots))
    store.verify_record_closure(roots)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".facility-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir(mode=0o700)
        target = ProvenanceStore(stage / "provenance")
        for rid in roots:
            target.copy_record_from(store, rid)
        captured = Path(temporary) / "teadata_sources.yaml"
        captured.write_bytes(config_bytes)
        config_id = target.record_acquisition(captured, ResolvedDataset("reference:facility_usage:config", "reference",
            2026, 2026, "", DatasetMetadata(title="Exact facility source configuration", version="2026")))
        recorder = BuildRecorder(cfg, target)
        toolchain = {"python": sys.version, "pyarrow": version("pyarrow")}
        records = []
        facility_processing = None
        for dataset, table, support in ((DATASETS[0], facilities, facility_support),
                                        (DATASETS[1], submission, submission_support)):
            key = EditionKey(dataset, "2026", EditionKey.source_revision((WORKBOOK_SHA,)), "district", "district",
                             "one original worksheet row per district" if dataset == DATASETS[0]
                             else "one publisher submission status per district")
            keys = ["district_number"] + (["publisher_worksheet_row"] if dataset == DATASETS[0] else [])
            chosen = FIELDS if dataset == DATASETS[0] else [f for f in FIELDS if f[0] in SUBMISSION_COORDS]
            recipe = {"operation": dataset, "worksheet": SHEET, "collection_school_year_end": 2026,
                      "original_coordinates": {name: coord for coord, name, *_ in chosen},
                      "excluded_source_columns": ["C"], "exclusion_reason": "submission_contact_not_public_product",
                      "limits": LIMITS}
            inputs = [*roots, config_id]
            if facility_processing is not None:
                inputs.append(facility_processing)
                recipe.update(group_by="district_number", require_equal=["A", "B", "AE", "AF"],
                              group_row_hash="sha256(canonical_json(sorted(original_worksheet_rows)))",
                              source_facility_processing=facility_processing)
            contract = recorder.leaf_contract(key, sources[dataset], native_schema=table.schema.serialize().to_pybytes(),
                columns=table.column_names, key_columns=keys, rows=table.num_rows, inputs=[WORKBOOK_SHA],
                recipe=recipe, toolchain=toolchain, column_support=support)
            contract_json = canonical_json(contract)
            leaf = stage / (sha256(contract_json.encode()).hexdigest() + ".parquet")
            physical = write_leaf_artifact(leaf, table.to_batches(max_chunksize=512), contract=contract, batch_bytes=8 * 1024**2)
            processing = target.record_processing(dataset=dataset, inputs=inputs, outputs={"leaf": leaf},
                processor="teadata.facility_usage", code_version=recorder.processor_sha256,
                parameters={"recipe": recipe, "leaf_contract_sha256": sha256(contract_json.encode()).hexdigest(),
                            "toolchain": toolchain}, validation={"rows": table.num_rows, "release_activation": False})
            artifact = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name, contract_json, processing, "leaf")
            recorder.record_edition(key, sources[dataset], artifacts=[artifact], acquisition_ids=[WORKBOOK_RECEIPT])
            records.append({"key": key.to_dict(), "artifact_id": artifact.identity, "leaf_sha256": artifact.sha256,
                            "leaf_size_bytes": artifact.size_bytes, "processing_id": processing, "rows": table.num_rows})
            facility_processing = processing
        catalog = recorder.catalog(schema_version=4)
        manifest = publish_release(stage, catalog, schema_version=4)
        result = {"status": "UNPUBLISHED_FACILITY_CHILD_REQUIRES_INDEPENDENT_QUALIFICATION",
                  "manifest_sha256": file_identity(manifest)["sha256"], "build_id": catalog.build_id,
                  "editions": records, "publisher_roots": list(PUBLISHER_ROOTS), "limits": LIMITS,
                  "release_activation": False}
        (stage / "MATERIALIZATION.json").write_text(canonical_json(result) + "\n")
        _install_candidate(stage, output)
    return result
