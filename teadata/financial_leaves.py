"""Explicit offline financial workbook leaves; no active Config or release changes.

Config selects literal worksheets and the existing reviewed column contract
owns every publisher field. Literal mode preserves original numeric tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import distribution, version
from io import BytesIO
import json
from pathlib import Path
import platform
import posixpath
import re
import shutil
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Literal, NotRequired, Sequence, TypedDict
import xml.etree.ElementTree as ET
from xml.parsers.expat import EXPAT_VERSION
import zipfile

from .column_schema import HeaderIdentity, crosswalk_dataset, parse_column_schema, reader_options, table_header_identity, table_input_sha256s
from .dataset_metadata import ArtifactRef, DatasetMetadata, EditionKey, MetricDefinition, ResolvedDataset
from .identifiers import canonical_source_identifiers
from .manifest import BuildRecorder, canonical_json, file_identity
from .provenance import ProvenanceStore
from .values import MeasurementUnit, ValuePolicy, ValueStatus, measure

if TYPE_CHECKING:
    import pyarrow as pa

    from .teadata_config import Config


NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
OFFICE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"
RID = "{" + OFFICE_REL + "}id"
MAX_ROWS = 20000
MAX_COLUMNS = 256
MAX_SOURCE_BYTES = 16 * 1024**2
MAX_XML_BYTES = 16 * 1024**2
MAX_BATCH_BYTES = 64 * 1024**2
MAX_LITERAL_CELLS = 1_000_000
_WORKSHEET_TARGET = re.compile(r"(?:/xl/)?worksheets/[A-Za-z0-9][A-Za-z0-9_.-]*\.xml")
_CELL_ADDRESS = re.compile(r"([A-Z]{1,3})([1-9][0-9]{0,6})")
_ESCAPED_TEXT = re.compile(r"_x[0-9a-f]{4}_", re.IGNORECASE)
_NUMERIC_TOKEN = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


class _HeaderField(TypedDict):
    """Typed view of fields already validated by the canonical crosswalk parser."""

    position: int
    code: str
    label: str
    field: str
    disposition: NotRequired[Literal["identity", "fact"]]


@dataclass(frozen=True, slots=True)
class _TableSelection:
    table: str
    sheet: str
    header_row: int
    scope: Literal["district", "campus", "state"]
    grain: str
    key_columns: tuple[str, ...]
    identifier_field: str
    header_identity: HeaderIdentity
    fields: tuple[_HeaderField, ...]
    uninterpreted_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _WorkbookSelection:
    source: ResolvedDataset
    edition: str
    source_sha256: str
    source_size_bytes: int
    tables: tuple[_TableSelection, ...]
    notes_sheets: tuple[str, ...]
    literal_cells: bool


def _xml(archive: zipfile.ZipFile, member: str, *, literal: bool = False):
    if archive.getinfo(member).file_size > MAX_XML_BYTES:
        raise ValueError("Financial OOXML member exceeds 16 MiB admission")
    raw = archive.read(member)
    if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
        raise ValueError("Financial OOXML contains a forbidden declaration")
    if literal:
        text = raw.decode("utf-8-sig")
        declaration = re.match(r"<\?xml\b[^?]*", text)
        encoding = re.search(r"""encoding\s*=\s*["']([^"']+)["']""", declaration[0]) if declaration else None
        if "\x00" in text or (encoding and encoding[1].lower() != "utf-8"):
            raise ValueError("Literal financial OOXML requires UTF-8 worksheet XML")
        return ET.fromstring(text)
    return ET.fromstring(raw)


def _require_element_only(element: ET.Element) -> None:
    whitespace = " \t\r\n"
    if ((element.text and element.text.strip(whitespace))
            or any(child.tail and child.tail.strip(whitespace) for child in element)):
        raise ValueError("Unexpected mixed content in literal financial OOXML")


def _cell_text(root: ET.Element) -> str:
    _require_element_only(root)
    children = list(root)
    if len(children) == 1 and children[0].tag == NS + "t" and not len(children[0]):
        value = children[0].text or ""
        if _ESCAPED_TEXT.search(value):
            raise ValueError("Unsupported literal financial Excel text escape")
        return value
    if children and all(run.tag == NS + "r" for run in children):
        values = []
        for run in children:
            _require_element_only(run)
            content = list(run)
            if content and content[0].tag == NS + "rPr":
                content = content[1:]
            if len(content) != 1 or content[0].tag != NS + "t" or len(content[0]):
                raise ValueError("Unsupported literal financial rich-text run")
            values.append(content[0].text or "")
        value = "".join(values)
        if _ESCAPED_TEXT.search(value):
            raise ValueError("Unsupported literal financial Excel text escape")
        return value
    raise ValueError("Unsupported literal financial shared or inline text")


def _package(archive: zipfile.ZipFile, *, literal: bool = False) -> tuple[dict[str, str], list[str]]:
    entries = archive.infolist()
    if (len(entries) > 4096 or len({entry.filename for entry in entries}) != len(entries)
            or sum(entry.file_size for entry in entries) > 256 * 1024**2
            or any(entry.flag_bits & 1 or entry.compress_type not in (0, 8) for entry in entries)):
        raise ValueError("Financial workbook exceeds bounded unencrypted OOXML admission")
    if literal:
        if len({entry.filename.lower() for entry in entries}) != len(entries):
            raise ValueError("Case-ambiguous literal financial ZIP member names")
        package = _xml(archive, "_rels/.rels", literal=True)
        office = [item.attrib for item in package if item.get("Type") == OFFICE_REL + "/officeDocument"]
        if (package.tag != PACKAGE_REL + "Relationships" or len(office) != 1
                or any(item.tag != PACKAGE_REL + "Relationship" for item in package)
                or len({item.get("Id") for item in package}) != len(package)
                or office[0].get("TargetMode", "Internal") != "Internal"
                or office[0].get("Target") not in {"xl/workbook.xml", "/xl/workbook.xml"}):
            raise ValueError("Literal financial workbook requires the canonical package relationship")
    relation_rows = _xml(archive, "xl/_rels/workbook.xml.rels", literal=literal)
    relations = {item.attrib["Id"]: item.attrib for item in relation_rows}
    if literal:
        shared = [relation for relation in relations.values() if relation["Type"] == OFFICE_REL + "/sharedStrings"]
        has_shared = "xl/sharedStrings.xml" in archive.namelist()
        if (relation_rows.tag != PACKAGE_REL + "Relationships"
                or any(item.tag != PACKAGE_REL + "Relationship" for item in relation_rows)
                or len(relations) != len(relation_rows) or len(shared) != int(has_shared)
                or (shared and (shared[0].get("TargetMode", "Internal") != "Internal"
                                or shared[0]["Target"] not in {"sharedStrings.xml", "/xl/sharedStrings.xml"}))):
            raise ValueError("Unsupported literal financial workbook relationships")
    sheets = {}
    workbook = _xml(archive, "xl/workbook.xml", literal=literal)
    if literal and (workbook.tag != NS + "workbook" or len(workbook.findall(NS + "sheets")) != 1):
        raise ValueError("Unsupported literal financial workbook sheet directory")
    for sheet in workbook.findall(NS + "sheets/" + NS + "sheet"):
        relation = relations[sheet.attrib[RID]]
        target = relation["Target"]
        part = posixpath.normpath(target.lstrip("/") if target.startswith("/") else "xl/" + target)
        if (relation.get("TargetMode") == "External" or not part.startswith("xl/")
                or part not in archive.namelist() or sheet.attrib["name"] in sheets
                or (literal and (part in sheets.values() or not part.startswith("xl/worksheets/")
                                 or _WORKSHEET_TARGET.fullmatch(target) is None
                                 or relation.get("Type") != OFFICE_REL + "/worksheet"
                                 or relation.get("TargetMode", "Internal") != "Internal"))):
            raise ValueError("Financial workbook has a duplicate or external worksheet")
        sheets[sheet.attrib["name"]] = part
    if not sheets or len(sheets) > 64:
        raise ValueError("Financial workbook requires 1..64 explicit worksheets")
    strings = []
    if "xl/sharedStrings.xml" in archive.namelist():
        root = _xml(archive, "xl/sharedStrings.xml", literal=literal)
        if len(root) > 100000:
            raise ValueError("Financial shared-string metadata exceeds admission")
        if literal and (root.tag != NS + "sst" or any(node.tag != NS + "si" for node in root)):
            raise ValueError("Unsupported literal financial shared-string table")
        strings = [_cell_text(node) if literal else "".join(node.itertext()) for node in root]
    return sheets, strings


def _sheet_frame(archive: zipfile.ZipFile, member: str, strings: list[str], *, header_row: int,
                 codes: list[str], literal: bool = False) -> tuple[int, list[ET.Element]]:
    """Validate the original worksheet framing before either value reader runs."""
    root = _xml(archive, member, literal=literal)
    if (root.find(".//" + NS + "f") is not None or root.find(NS + "mergeCells/" + NS + "mergeCell") is not None
            or root.find(".//" + NS + "c[@t='e']") is not None):
        raise ValueError("Financial fact worksheets cannot silently consume formulas, errors or merged cells")
    rows = root.findall(NS + "sheetData/" + NS + "row")
    if literal:
        data = root.findall(NS + "sheetData")
        if (root.tag != NS + "worksheet" or len(data) != 1 or len(data[0]) != len(rows)
                or any(cell.tag != NS + "c" for row in rows for cell in row)
                or sum(len(row) for row in rows) != sum(1 for _ in root.iter(NS + "c"))):
            raise ValueError("Unaccounted literal financial worksheet cells")
        _require_element_only(data[0])
        for row in rows:
            _require_element_only(row)
    ordinals = [row.attrib.get("r", "") for row in rows]
    if literal and any(not re.fullmatch(r"[1-9][0-9]{0,6}", ordinal) for ordinal in ordinals):
        raise ValueError("Invalid literal financial row ordinal")
    numbers = [int(ordinal) for ordinal in ordinals]
    if (not numbers or numbers != sorted(set(numbers)) or numbers[-1] - header_row > MAX_ROWS
            or header_row not in numbers):
        raise ValueError("Financial worksheet rows are missing, duplicated, unordered or exceed admission")
    if any(number < header_row for number in numbers):
        raise ValueError("Unaccounted financial preamble rows precede the declared header")
    # This is only a structural payload guard. Native Calamine/Arrow owns all
    # financial value extraction. Explicit column selection must not hide data
    # in an otherwise unnamed tail column.
    for cell in root.iter(NS + "c"):
        reference = _CELL_ADDRESS.fullmatch(cell.get("r", ""))
        if reference is None:
            raise ValueError("Financial source cell has an invalid physical coordinate")
        column = 0
        for character in reference[1]:
            column = column * 26 + ord(character) - 64
        if literal and column > 16384:
            raise ValueError("Literal financial cell exceeds the Excel worksheet width")
        if column > len(codes):
            inline = cell.find(NS + "is")
            if (literal and len(cell)) or cell.findtext(NS + "v") not in (None, "") or (inline is not None and "".join(inline.itertext()) != ""):
                raise ValueError("Financial source contains unclaimed physical column payload")
    header = rows[numbers.index(header_row)]
    observed = []
    header_addresses = set()
    for cell in header.findall(NS + "c"):
        if literal:
            address = cell.attrib["r"]
            match = _CELL_ADDRESS.fullmatch(address)
            if match is None or int(match[2]) != header_row or address in header_addresses:
                raise ValueError("Duplicate or mismatched literal financial header coordinate")
            header_addresses.add(address)
        value, kind = cell.findtext(NS + "v"), cell.get("t")
        if literal:
            value = _literal_token(cell, strings)
        elif kind == "s":
            if value is None or not value.isdecimal() or int(value) >= len(strings):
                raise ValueError("Invalid financial shared-string header index")
            value = strings[int(value)]
        elif kind == "inlineStr":
            inline = cell.find(NS + "is")
            value = "".join(inline.itertext()) if inline is not None else None
        if value is not None and value != "":
            number, letters = len(observed) + 1, ""
            while number:
                number, remainder = divmod(number - 1, 26)
                letters = chr(65 + remainder) + letters
            if cell.get("r") != letters + str(header_row):
                raise ValueError("Financial header positions differ from their physical source columns")
            observed.append(value)
    # Calamine omits styled-empty tails. Its complete native schema below must
    # still equal these exact labels; a populated unnamed column is not dropped.
    if observed != codes or not 0 < len(codes) <= MAX_COLUMNS:
        raise ValueError("Financial publisher headers differ from the reviewed ordered fields")
    return numbers[-1] - header_row, rows


def _literal_token(cell: ET.Element, strings: list[str]) -> str | None:
    _require_element_only(cell)
    kind = cell.get("t", "n")
    payload = list(cell)
    if kind == "inlineStr" and len(payload) == 1 and payload[0].tag == NS + "is":
        return _cell_text(payload[0])
    if not payload and kind == "n":
        return None
    if len(payload) != 1 or payload[0].tag != NS + "v" or len(payload[0]):
        raise ValueError("Unsupported literal financial cell payload")
    value = payload[0].text
    if kind == "n" and (value is None or _NUMERIC_TOKEN.fullmatch(value)):
        return value
    if kind == "s" and value is not None and re.fullmatch(r"[0-9]{1,6}", value) and int(value) < len(strings):
        return strings[int(value)]
    raise ValueError("Unsupported literal financial cell type or value")


def _literal_batch(rows: list[ET.Element], strings: list[str], names: list[str], header_row: int,
                   row_count: int) -> pa.RecordBatch:
    import pyarrow as pa

    dense_cells = row_count * len(names)
    if dense_cells > MAX_LITERAL_CELLS or len(rows) != row_count + 1:
        raise ValueError("Literal financial worksheet exceeds cell admission or has a physical row gap")
    projected: list[list[str | None]] = [[None] * row_count for _ in names]
    allocated = 5 * (dense_cells + len(names))
    for row_index, row in enumerate(rows[1:]):
        ordinal = row_index + header_row + 1
        if int(row.attrib["r"]) != ordinal:
            raise ValueError("Literal financial worksheet has a physical row gap")
        seen: set[str] = set()
        for cell in row.findall(NS + "c"):
            reference = cell.attrib["r"]
            match = _CELL_ADDRESS.fullmatch(reference)
            if match is None or int(match[2]) != ordinal or reference in seen:
                raise ValueError("Duplicate or mismatched literal financial cell coordinate")
            seen.add(reference)
            column = 0
            for character in match[1]:
                column = 26 * column + ord(character) - 64
            value = _literal_token(cell, strings)
            if value is not None:
                allocated += len(value.encode("utf-8"))
                if allocated > MAX_BATCH_BYTES:
                    raise ValueError("Literal financial worksheet exceeds expanded string byte admission")
            if column <= len(names):
                projected[column - 1][row_index] = value
    return pa.RecordBatch.from_arrays([pa.array(column, type=pa.string()) for column in projected], names=names)


def _selection(cfg: Config, dataset: str, year: int, recorder: BuildRecorder) -> _WorkbookSelection:
    source = cfg.resolve_dataset(dataset, year, reader_profile="source", exact=True)
    if source.metadata.version is None:
        raise ValueError("Financial sources require an explicit publisher edition")
    identity = recorder.before_read(source, reader_profile="source")
    if not 0 < identity["size_bytes"] <= MAX_SOURCE_BYTES:
        raise ValueError("Financial workbook exceeds the 16 MiB original-source admission")
    options = reader_options(source.reader, "source")
    workbook = options.get("workbook")
    if (not isinstance(workbook, dict) or not {"tables", "notes_sheets"} <= set(workbook)
            or set(workbook) - {"tables", "notes_sheets", "cell_text"}
            or workbook.get("cell_text", "calamine") not in ("calamine", "ooxml_literal")
            or not isinstance(workbook["tables"], dict) or not workbook["tables"]
            or not isinstance(workbook["notes_sheets"], list)
            or any(not isinstance(name, str) or not name for name in workbook["notes_sheets"])
            or len(set(workbook["notes_sheets"])) != len(workbook["notes_sheets"])):
        raise ValueError("Financial reader requires explicit tables and notes_sheets in its Config profile")
    resource = cfg.column_contract(dataset, year, reader_profile="source")
    if resource is None:
        raise ValueError("Financial workbook requires a reviewed publisher-header resource")
    selected = crosswalk_dataset(parse_column_schema(resource["text"].encode()), dataset, source.metadata.version)
    tables = selected["tables"]
    if set(workbook["tables"]) != {table["table"] for table in tables}:
        raise ValueError("Financial worksheet choices must account for every reviewed table")
    choices = []
    for table in tables:
        if table_input_sha256s(table) != (identity["sha256"],):
            raise ValueError("Financial reviewed headers belong to a different original workbook")
        declaration = workbook["tables"][table["table"]]
        fields = {field["field"]: field for field in table["fields"]}
        required = {"sheet", "header_row", "scope", "grain", "key_columns", "identifier_field"}
        if (not isinstance(declaration, dict)
                or not required <= set(declaration) or set(declaration) - required - {"uninterpreted_fields"}
                or any(not isinstance(declaration.get(name), str) or not declaration[name].strip() for name in ("sheet", "grain"))
                or type(declaration["header_row"]) is not int or not 1 <= declaration["header_row"] <= 100
                or declaration["scope"] not in {"district", "campus", "state"}
                or not isinstance(declaration["key_columns"], list) or not declaration["key_columns"]
                or any(not isinstance(name, str) or name not in fields for name in declaration["key_columns"])
                or len(set(declaration["key_columns"])) != len(declaration["key_columns"])
                or any(fields[name].get("disposition") != "identity" for name in declaration["key_columns"])
                or declaration["identifier_field"] not in declaration["key_columns"]
                or declaration["identifier_field"] not in source.metadata.identifiers
                or fields[declaration["identifier_field"]].get("disposition") != "identity"):
            raise ValueError("Financial table requires explicit existing fields, entity scope and genuine row keys")
        uninterpreted = declaration.get("uninterpreted_fields", [])
        if (not isinstance(uninterpreted, list)
                or any(not isinstance(name, str) or name not in fields for name in uninterpreted)
                or len(set(uninterpreted)) != len(uninterpreted)):
            raise ValueError("Financial uninterpreted_fields requires distinct existing fact names")
        for name in uninterpreted:
            definitions = [metric for metric in source.metadata.metrics.values() if metric.field == name]
            if (fields[name].get("disposition", "fact") != "fact"
                    or name in declaration["key_columns"] or name in source.metadata.identifiers
                    or not definitions or any(metric.unit is not None for metric in definitions)):
                raise ValueError("Financial uninterpreted fields require explicit unknown-unit facts, not identifiers or known units")
        if any(field["code"] != field["label"] for field in table["fields"]):
            raise ValueError("Workbook fields require their literal single publisher-header representation")
        # Canonical metadata owns effective periods and units, including unknowns.
        cfg.header_records(dataset, edition=source.metadata.version, table=table["table"], reader_profile="source")
        choices.append(_TableSelection(table=table["table"], sheet=declaration["sheet"], header_row=declaration["header_row"],
            scope=declaration["scope"], grain=declaration["grain"], key_columns=tuple(declaration["key_columns"]),
            identifier_field=declaration["identifier_field"], header_identity=table_header_identity(table), fields=tuple(table["fields"]),
            uninterpreted_fields=tuple(uninterpreted)))
    return _WorkbookSelection(source=source, edition=source.metadata.version, source_sha256=identity["sha256"],
        source_size_bytes=identity["size_bytes"], tables=tuple(choices), notes_sheets=tuple(workbook["notes_sheets"]),
        literal_cells=workbook.get("cell_text") == "ooxml_literal")


def _project(batch: pa.RecordBatch, selection: _TableSelection, source: ResolvedDataset,
             policy: ValuePolicy) -> tuple[pa.Table, dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.compute as arrow_compute

    pc: Any = arrow_compute
    if batch.num_rows < 1 or batch.num_rows > MAX_ROWS or batch.nbytes > MAX_BATCH_BYTES:
        raise ValueError("Financial native projection is empty or exceeds bounded row/byte admission")
    names = [field["field"] if selection.header_identity == "position" else field["code"] for field in selection.fields]
    if batch.schema.names != names:
        raise ValueError("Financial native schema has an unexpected header, unnamed field or missing column")
    context = ["financial_source_sheet", "financial_source_row"]
    if any(field["field"].startswith(("financial_raw_", "financial_status_")) or field["field"] in context for field in selection.fields):
        raise ValueError("Financial publisher names collide with reserved support columns")
    definitions: dict[str, MetricDefinition] = {}
    for metric in source.metadata.metrics.values():
        if metric.field in definitions and definitions[metric.field].unit != metric.unit:
            raise ValueError("Financial field has contradictory canonical units")
        definitions[metric.field] = metric
    columns, support = {}, {}
    for field in selection.fields:
        name = field["field"]
        raw = batch.column(field["position"] - 1) if selection.header_identity == "position" else batch[field["code"]]
        if not (pa.types.is_string(raw.type) or pa.types.is_large_string(raw.type)):
            raise ValueError("Financial raw support requires the explicit native string projection")
        if name == selection.identifier_field:
            value = canonical_source_identifiers(raw, width=9 if selection.scope == "campus" else 6) if selection.scope != "state" else raw
            if pc.any(pc.or_(pc.is_null(value), pc.fill_null(pc.equal(pc.utf8_trim_whitespace(value), ""), False))).as_py():
                raise ValueError("Financial source has a missing or malformed publisher entity identifier/footer")
            columns[name] = value
            columns["financial_raw_" + name] = raw
            support[name] = {"raw": "financial_raw_" + name}
        elif field.get("disposition", "fact") == "identity":
            columns[name] = raw
        elif name in selection.uninterpreted_fields:
            columns[name] = raw
        else:
            definition = definitions.get(name)
            unit = definition.unit if definition is not None else None
            if unit is None or MeasurementUnit(unit) == MeasurementUnit.TEXT:
                raise ValueError("Financial numeric facts need explicitly reviewed canonical units")
            unique = pc.unique(raw)
            if len(unique) > 4096:
                raise ValueError("Financial field exceeds bounded distinct-token interpretation")
            interpreted = [measure(token, policy, unit=unit) for token in unique.to_pylist()]
            if any(value.status == ValueStatus.INVALID for value in interpreted):
                raise ValueError("Financial source contains an invalid value under its explicit policy/unit")
            indices = pc.index_in(raw, value_set=unique)
            numbers = [value.numeric for value in interpreted]
            integral = all(value is None or isinstance(value, int) for value in numbers)
            if not integral and any(isinstance(value, int) and int(float(value)) != value for value in numbers):
                raise ValueError("Financial mixed numeric representation would lose integer precision")
            columns[name] = pc.take(pa.array(numbers, type=pa.int64() if integral else pa.float64()), indices)
            columns["financial_raw_" + name] = raw
            columns["financial_status_" + name] = pc.take(pa.array([value.status.value for value in interpreted], type=pa.string()), indices)
            support[name] = {"raw": "financial_raw_" + name, "status": "financial_status_" + name}
    columns[context[0]] = pa.repeat(selection.sheet, batch.num_rows)
    columns[context[1]] = pa.array(range(selection.header_row + 1, selection.header_row + 1 + batch.num_rows), type=pa.int64())
    table = pa.table(columns)
    if table.nbytes > MAX_BATCH_BYTES:
        raise ValueError("Financial projected table exceeds the admitted native byte budget")
    keys = list(selection.key_columns)
    if (any(table[name].null_count or pc.any(pc.equal(pc.utf8_trim_whitespace(table[name]), "")).as_py() for name in keys)
            or table.select(keys).group_by(keys).aggregate([]).num_rows != table.num_rows):
        raise ValueError("Financial publisher row keys are missing or duplicate")
    return table, {"fields": support, "context": context}


def materialize_financial_workbooks(store: ProvenanceStore, *, config: Path, config_sha256: str,
                                   selections: Sequence[tuple[str, int]], output: Path) -> dict[str, Any]:
    """Create adjacent unpublished leaves from explicit reviewed original sources."""
    from .analytical import write_leaf_artifact
    from .release import publish_release
    from .teadata_config import load_config_bytes

    if (not selections or len(set(selections)) != len(selections)
            or any(not isinstance(dataset, str) or not dataset or type(year) is not int for dataset, year in selections)):
        raise ValueError("Select distinct explicit financial dataset/catalog-year pairs")
    with config.open("rb") as stream:
        raw_config = stream.read(2 * 1024**2 + 1)
    if len(raw_config) > 2 * 1024**2 or sha256(raw_config).hexdigest() != config_sha256:
        raise ValueError("Financial Config differs from its exact pin or exceeds 2 MiB")
    cfg = load_config_bytes(raw_config, path=config)
    admission = BuildRecorder(cfg, store)
    admitted = [_selection(cfg, dataset, year, admission) for dataset, year in selections]
    if output.exists() or output.is_symlink():
        raise FileExistsError("Financial candidate requires a new exclusive output directory")
    import pyarrow.parquet as pq

    common_toolchain = {"python": sys.version, "pyarrow": version("pyarrow"),
                        "platform": platform.platform(), "architecture": platform.machine()}
    calamine_toolchain = {}
    if any(not item.literal_cells for item in admitted):
        reader_distribution = distribution("fastexcel")
        native = [file_identity(Path(str(reader_distribution.locate_file(entry))))["sha256"]
                  for entry in reader_distribution.files or () if entry.suffix in {".so", ".pyd", ".dylib"}]
        if not native:
            raise ValueError("Financial native reader extension identity is unavailable")
        calamine_toolchain = {"fastexcel": reader_distribution.version,
                              "fastexcel_native_sha256": sha256(canonical_json(native).encode()).hexdigest()}
    lock = Path(__file__).parents[1] / "uv.lock"
    if lock.is_file():
        common_toolchain["lock_sha256"] = file_identity(lock)["sha256"]
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".financial-leaves-", dir=output.parent))
    stage.chmod(0o700)
    try:
        target_store = ProvenanceStore(stage / "provenance")
        captured = stage / "source-config.input"
        with captured.open("xb") as stream:
            stream.write(raw_config)
        captured.chmod(0o600)
        config_id = target_store.record_acquisition(captured, ResolvedDataset("reference:financial_leaf_config", "reference", None, None,
            "", DatasetMetadata(title="Exact reviewed financial workbook source Config")))
        recorder = BuildRecorder(cfg, target_store)
        result: dict[str, Any] = {"status": "unpublished_native_financial_candidate", "field_correctness_qualified": False,
                                  "release_activation": False, "editions": []}
        for item in admitted:
            source = item.source
            roots = list(source.metadata.provenance_ids) + [identifier for reference in source.metadata.references for identifier in reference.provenance_ids]
            for identifier in roots:
                target_store.copy_record_from(store, identifier)
            policy_id = source.metadata.value_policy
            policy = cfg.value_policies.get(policy_id) if policy_id is not None else None
            if policy is None:
                raise ValueError("Financial source requires its explicitly selected canonical ValuePolicy")
            toolchain = {**common_toolchain, **({"expat": EXPAT_VERSION} if item.literal_cells else calamine_toolchain)}
            original: str | BytesIO = source.path
            if item.literal_cells:
                with Path(source.path).open("rb") as stream:
                    payload = stream.read(MAX_SOURCE_BYTES + 1)
                if len(payload) != item.source_size_bytes or sha256(payload).hexdigest() != item.source_sha256:
                    raise ValueError("Literal financial workbook bytes differ from the pinned source")
                original = BytesIO(payload)
                del payload
            with zipfile.ZipFile(original) as archive:
                sheets, strings = _package(archive, literal=item.literal_cells)
                declared = [value.sheet for value in item.tables] + list(item.notes_sheets)
                if len(set(declared)) != len(declared) or set(declared) != set(sheets):
                    raise ValueError("Financial source has unaccounted, duplicate or missing worksheets")
                reader = None
                if not item.literal_cells:
                    import fastexcel

                    reader = fastexcel.read_excel(Path(source.path))
                for selection in item.tables:
                    codes = [field["code"] for field in selection.fields]
                    names = [field["field"] for field in selection.fields] if selection.header_identity == "position" else codes
                    expected_rows, rows = _sheet_frame(archive, sheets[selection.sheet], strings,
                        header_row=selection.header_row, codes=codes, literal=item.literal_cells)
                    if reader is None:
                        batch = _literal_batch(rows, strings, names, selection.header_row, expected_rows)
                        del rows
                    else:
                        del rows
                        if selection.header_identity == "position":
                            batch = reader.load_sheet_eager(selection.sheet, header_row=None, skip_rows=selection.header_row,
                                column_names=[field["field"] for field in selection.fields],
                                use_columns=[field["position"] - 1 for field in selection.fields],
                                n_rows=MAX_ROWS + 1, dtypes="string", dtype_coercion="strict")
                        else:
                            batch = reader.load_sheet_eager(selection.sheet, header_row=selection.header_row - 1,
                                n_rows=MAX_ROWS + 1, dtypes="string", dtype_coercion="strict")
                    if batch.num_rows != expected_rows:
                        raise ValueError("Financial native row framing differs from physical worksheet coordinates")
                    projected, support = _project(batch, selection, source, policy)
                    key = EditionKey(source.dataset, item.edition, EditionKey.source_revision((item.source_sha256,)),
                        selection.scope, selection.table, selection.grain)
                    recipe = {"operation": "explicit_financial_workbook_projection", "config_sha256": config_sha256,
                        "worksheet": {"sheet": selection.sheet, "header_row": selection.header_row, "scope": selection.scope,
                            "grain": selection.grain, "key_columns": list(selection.key_columns), "identifier_field": selection.identifier_field,
                            "header_identity": selection.header_identity,
                            **({"uninterpreted_fields": list(selection.uninterpreted_fields)} if selection.uninterpreted_fields else {})},
                        "notes_sheets": list(item.notes_sheets), "value_policy": policy.to_dict(),
                        "representation": ("literal OOXML numeric tokens and decoded worksheet text"
                                           if item.literal_cells else "Calamine native strings, not original OOXML lexemes"),
                        "batch_rows": 1024,
                        "numeric_representation": "source unformatted value under reviewed stored-value units; no scaling",
                        "totals_or_ratios_computed": False, "formulas_recalculated": False,
                        "cli_sha256": file_identity(Path(__file__).parent / "scripts/refresh_financials.py")["sha256"]}
                    contract = recorder.leaf_contract(key, source, native_schema=projected.schema.serialize().to_pybytes(),
                        columns=projected.column_names, key_columns=selection.key_columns, rows=projected.num_rows,
                        inputs=[item.source_sha256], recipe=recipe, toolchain=toolchain, column_support=support)
                    leaf_id = sha256(canonical_json(contract).encode()).hexdigest()
                    leaf = stage / (leaf_id + ".parquet")
                    physical = write_leaf_artifact(leaf, projected.to_batches(max_chunksize=1024), contract=contract, batch_bytes=MAX_BATCH_BYTES)
                    if not projected.equals(pq.read_table(leaf).replace_schema_metadata(None), check_metadata=True):
                        raise ValueError("Financial leaf readback differs from its native source projection")
                    processing = target_store.record_processing(dataset=source.dataset, inputs=[*roots, config_id], outputs={"leaf": leaf},
                        processor="teadata.scripts.refresh_financials.materialize", code_version=recorder.processor_sha256,
                        parameters={"leaf_contract_sha256": leaf_id, "recipe": recipe, "toolchain": toolchain},
                        validation={"native_projection_readback_equal": True, "rows": projected.num_rows,
                                    "field_correctness_qualified": False, "release_activation": False})
                    ref = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name, canonical_json(contract), processing, "leaf")
                    recorder.record_edition(key, source, artifacts=[ref], acquisition_ids=source.metadata.provenance_ids)
                    result["editions"].append({"key": key.to_dict(), "artifact_id": leaf_id,
                        "rows": projected.num_rows, "processing_record_id": processing})
                    del batch, projected
            reader = None
            if isinstance(original, BytesIO):
                original.close()
            after = file_identity(source.path)
            if after["sha256"] != item.source_sha256 or after["size_bytes"] != item.source_size_bytes:
                raise ValueError("Original financial workbook changed during materialization")
        catalog = recorder.catalog(schema_version=3)
        manifest = publish_release(stage, catalog, schema_version=3)
        result.update(build_id=catalog.build_id, manifest=file_identity(manifest))
        with (stage / "MATERIALIZATION.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=True, allow_nan=False)
            stream.write("\n")
        if output.exists() or output.is_symlink():
            raise FileExistsError("Financial output appeared before completion")
        stage.rename(output)
        return result
    finally:
        if stage.exists():
            shutil.rmtree(stage)
