"""Replay unpublished PEIMS course leaves against retained CSV and HTML reports."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
import csv
from hashlib import sha256
from importlib.resources import files
from itertools import islice
import json
from pathlib import Path
import re
from typing import Any, Literal
from urllib.parse import urlencode

import pyarrow as pa
import pyarrow.compute as arrow_compute
import pyarrow.parquet as pq

from .dataset_metadata import DataPeriod, EditionKey, LeafColumnSupport
from .manifest import ContractError, canonical_json
from .peims_course_sources import BROKER, CSV_HEADERS, DATASET, ENCODING, SELECTOR, course_csv_preamble
from .peims_courses import COURSE_FACT_UNITS, COURSE_RECIPE, SOURCE_RECORD, _admit
from .provenance import ProvenanceStore
from .release import DataRelease
from .scripts.verify_peims_courses import _HTML_HEADER, _check_footer, _course_rows, _html_rows, _open_report
from .teadata_config import Config, load_config
from .values import ValueStatus

pc: Any = arrow_compute
PASS_COURSE_CONTROLS = "PASS_REVIEWED_PEIMS_COURSE_CONTROLS"
HOLD_CAMPUS_GEOGRAPHY = "HOLD_PEIMS_CAMPUS_GEOGRAPHY"
_BATCH_ROWS = 4096
_MAX_LINE_BYTES = 8192
_GEO_FIELDS = {
    "state": (),
    "district": ("district_number", "district_name", "charter_status", "region_number"),
    "campus": ("source_district_number", "source_district_name", "campus_number", "campus_name",
               "charter_status", "region_number"),
}
_IDENTITY_FIELDS = ("school_year", "source_scope", "aggregation_level")
_COURSE_FIELDS = ("subject", "course_code", "course_title", "grade_level")
_MEASURE_FIELDS = tuple(COURSE_FACT_UNITS)


def _raw_batches(path: Path, year: int, scope: str, count: int) -> Iterator[list[list[str]]]:
    headings = CSV_HEADERS[scope]
    with path.open("rb") as stream:
        def lines() -> Iterator[str]:
            while line := stream.readline(_MAX_LINE_BYTES + 1):
                if len(line) > _MAX_LINE_BYTES:
                    raise ContractError("Course CSV physical record exceeds its replay limit")
                yield line.decode(ENCODING)

        records = csv.reader(lines(), strict=True)
        prefix = list(islice(records, 9))
        if prefix != course_csv_preamble(year, scope) or records.line_num != 9:
            raise ContractError(f"Course {scope} CSV preamble or header differs")
        remaining = count
        while remaining:
            size = min(_BATCH_ROWS, remaining)
            start = records.line_num
            batch = list(islice(records, size))
            if len(batch) != size or records.line_num != start + size or any(len(row) != len(headings) for row in batch):
                raise ContractError("Course CSV record framing or row count differs")
            yield batch
            remaining -= size
        footer = list(islice(records, 33))
        if len(footer) > 32:
            raise ContractError("Course CSV footer exceeds its physical record limit")
        _check_footer([row for row in footer if row], year, "csv")


def _typed_measure(raw: pa.Array, status: pa.Array, value: pa.Array, index: int) -> None:
    suppressed = pc.equal(raw, "-999")
    if index != 4 and pc.any(suppressed).as_py():
        raise ContractError(f"Unexpected course masking token for {_MEASURE_FIELDS[index]}")
    expected_status = pc.if_else(suppressed, ValueStatus.SUPPRESSED.value, ValueStatus.OBSERVED.value)
    if not status.equals(expected_status):
        raise ContractError(f"Course measure status differs for {_MEASURE_FIELDS[index]}")
    if index in (1, 3):
        if value.type != pa.decimal128(16, 2):
            raise ContractError(f"Course FTE is not fixed precision for {_MEASURE_FIELDS[index]}")
        pattern = r"^[0-9]+(?:\.[0-9]{2})?$"
    else:
        if value.type != pa.int64():
            raise ContractError(f"Course count is not int64 for {_MEASURE_FIELDS[index]}")
        pattern = r"^[0-9]+$"
    if not pc.all(pc.or_(suppressed, pc.match_substring_regex(raw, pattern))).as_py():
        raise ContractError(f"Course raw value is unreviewed for {_MEASURE_FIELDS[index]}")
    expected_value = pc.cast(pc.if_else(suppressed, None, raw), value.type, safe=True)
    if not value.equals(expected_value):
        raise ContractError(f"Course normalized value differs for {_MEASURE_FIELDS[index]}")


def _compare_batch(rows: list[list[str]], native: pa.RecordBatch, scope: str, first: int) -> int:
    if native.num_rows != len(rows):
        raise ContractError("Course raw and native batch sizes differ")
    geo_fields = _GEO_FIELDS[scope]
    raw = list(zip(*rows, strict=True))
    identity = (raw[2], raw[0], raw[1], *raw[3:3 + len(geo_fields) + len(_COURSE_FIELDS)])
    for name, tokens in zip((*_IDENTITY_FIELDS, *geo_fields, *_COURSE_FIELDS), identity, strict=True):
        expected = pa.array(tokens, type=pa.string())
        if not native[name].equals(expected):
            raise ContractError(f"Course raw value differs for {name} at source line {first}")
    if not native[SOURCE_RECORD].equals(pa.array(range(first, first + len(rows)), type=pa.int64())):
        raise ContractError(f"Course physical source locator differs at line {first}")
    masked = 0
    for index, (name, tokens) in enumerate(zip(_MEASURE_FIELDS, raw[-len(_MEASURE_FIELDS):], strict=True)):
        archived = pa.array(tokens, type=pa.string())
        if not native[f"peims_raw_{name}"].equals(archived):
            raise ContractError(f"Course raw measure differs for {name} at source line {first}")
        _typed_measure(archived, native[f"peims_status_{name}"], native[name], index)
        masked += pc.sum(pc.equal(archived, "-999")).as_py()
    return masked


def _open_selected_witness(store: ProvenanceStore, identifier: str, year: int, selection: str,
                           method: Literal["GET", "POST"], controls: list[tuple[str, str]]) -> tuple[str, dict[str, Any]]:
    record = store.read_record(identifier)
    artifact = record["artifact"]
    parameters = record["parameters"]
    metadata = record["source_metadata"]
    url = f"{BROKER}?{urlencode(controls)}" if method == "GET" else BROKER
    content_type = "application/x-www-form-urlencoded" if method == "POST" else None
    if (record["kind"] != "acquisition" or record["method"] != "download" or record["dataset"] != DATASET
            or record["section"] != "data_sources" or (record["requested_year"], record["resolved_year"]) != (year, year)
            or (record["requested_url"], record["resolved_url"], parameters.get("form_url")) != (url, url, BROKER)
            or parameters.get("http_method") != method or parameters.get("http_status") != 200
            or parameters.get("form_controls") != [list(field) for field in controls]
            or parameters.get("form_content_type") != content_type
            or metadata["source_url"] != SELECTOR or metadata["version"] != str(year)
            or metadata["period"] != DataPeriod("school_year", year - 1, year).to_dict()
            or metadata["publisher"] != "Texas Education Agency" or not record["downloaded_at"]
            or record["response_headers"]["content-type"].split(";")[0] != "text/html"
            or not 0 < artifact["size_bytes"] <= 8 * 1024**2):
        raise ContractError(f"Course witness does not bind {year} and {selection}")
    source = store.verify_artifact(artifact["sha256"])
    if source.stat().st_size != artifact["size_bytes"]:
        raise ContractError("Course witness byte count differs")
    text = source.read_text(encoding=ENCODING)
    headings = re.findall(r"<h1\b[^>]*>\s*([^<]*?)\s*</h1>", text, re.IGNORECASE)
    if headings != [f"{year - 1}-{year} Teacher FTE Counts and Course Enrollment Reports"]:
        raise ContractError("Course witness edition differs")
    return text, {"receipt_id": identifier, "sha256": artifact["sha256"], "downloaded_at": record["downloaded_at"]}


def _district_witness(store: ProvenanceStore, identifier: str, year: int, district: str, name: str,
                      charter: str, region: str) -> tuple[list[list[str]], dict[str, Any]]:
    from .native_witnesses import _Rows

    controls = [
        ("_service", "marykay"), ("prev_htrefer", " "), ("referrer", " "), ("PageTitle", "."),
        ("RptClass", "FteEnroll"), ("SchoolYr", f"{year % 100:02}"), ("report", "DistrictDistrict"),
        ("format", "html"), ("sequence", "&sequence."), ("_debug", "0"),
        ("_program", "adhoc.std_driver2.sas"), ("district", f"{district} {name}"),
    ]
    text, evidence = _open_selected_witness(store, identifier, year, district, "POST", controls)
    table = _Rows(text).rows
    label = f"{name} ({district})"
    if table[:3] != [[f"Districtwide for {label}"], [f"Charter Status: {charter}"],
                     ["District Name/District", "Region", *_HTML_HEADER]]:
        raise ContractError("Course district witness geography or headings differ")
    footer_at = next((i for i, row in enumerate(table[3:], 3) if len(row) != 2 + len(_HTML_HEADER)), len(table))
    _check_footer(table[footer_at:], year, "html")
    visible = table[3:footer_at]
    if (not visible or visible[0][:2] != [label, region]
            or any(row[0] not in (label, "") or row[1] not in (region, "") for row in visible)):
        raise ContractError("Course district witness row identity differs")
    return [row[2:] for row in visible], evidence


def _campus_blocks(table: list[list[str]], year: int, campus_label: str,
                   originals: list[list[str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[list[str]]] = defaultdict(list)
    for row in originals:
        grouped[row[3]].append(row)
    expected = {district: _course_rows([row[9:] for row in raw], "csv") for district, raw in grouped.items()}
    header = ["Campus Name/Campus", *_HTML_HEADER]
    blocks: list[dict[str, Any]] = []
    offset = 0
    while offset + 2 < len(table) and table[offset + 2] == header:
        heading, charter, _ = table[offset:offset + 3]
        match = re.fullmatch(r"District: (.+) \(([0-9]{6})\) - Region: ([0-9]{2})",
                             heading[0] if len(heading) == 1 else "")
        if match is None or len(charter) != 1 or not charter[0].startswith("Charter Status: "):
            raise ContractError("Course campus witness district or charter heading differs")
        start = offset + 3
        stop = next((i for i in range(start, len(table)) if len(table[i]) != len(header)), len(table))
        visible = table[start:stop]
        if (not visible or visible[0][0] != campus_label
                or any(row[0] not in (campus_label, "") for row in visible)):
            raise ContractError("Course campus witness row identity differs")
        displayed = _course_rows([row[1:] for row in visible], "html")
        matches = [district for district, original in expected.items() if displayed == original]
        if len(matches) != 1:
            raise ContractError("Course campus witness values do not identify a unique bulk district block")
        district = matches[0]
        original_rows = grouped[district]
        context = original_rows[0][4:9]
        if any(row[4:9] != context for row in original_rows):
            raise ContractError("Course campus bulk context varies within the matched district block")
        name, _, _, status, region = context
        blocks.append({"displayed_district_heading": heading[0], "matched_source_district_number": district,
                       "heading_matches_bulk": match.groups() == (name, district, region),
                       "charter_status_matches_bulk": charter == [f"Charter Status: {status}"],
                       "rows": len(displayed)})
        del expected[district]
        offset = stop
    _check_footer(table[offset:], year, "html")
    if not blocks or expected:
        raise ContractError("Course campus witness does not cover all selected bulk district blocks")
    return blocks


def _campus_witness(store: ProvenanceStore, identifier: str, year: int, campus: str,
                    originals: list[list[str]]) -> dict[str, Any]:
    from .native_witnesses import _Rows

    if not originals:
        raise ContractError(f"Course campus witness {campus} is absent from the bulk source")
    names = {row[6] for row in originals}
    if len(names) != 1:
        raise ContractError(f"Course campus witness {campus} has multiple names in the bulk source")
    (name,) = names
    controls = [
        ("_service", "marykay"), ("prev_htrefer", " "), ("referrer", " "), ("PageTitle", "."),
        ("RptClass", "FteEnroll"), ("SchoolYr", f"{year % 100:02}"), ("report", "CampusCampus"),
        ("format", "html"), ("_debug", "0"), ("_program", "adhoc.std_driver2.sas"), ("campus", f"{campus} {name}"),
    ]
    text, evidence = _open_selected_witness(store, identifier, year, campus, "GET", controls)
    blocks = _campus_blocks(_Rows(text).rows, year, f"{name} ({campus})", originals)
    return {**evidence, "campus_number": campus, "rows": sum(block["rows"] for block in blocks),
            "source_district_heading_conflicts": sum(not block["heading_matches_bulk"] for block in blocks),
            "blocks": blocks}


def _replay(release: DataRelease, store: ProvenanceStore, witnesses: ProvenanceStore,
            config: Config, key: EditionKey, review: dict[str, Any]) -> dict[str, Any]:
    variables = list(release.catalog.variables(edition=key))
    (plan,) = release.catalog.resolve_artifacts((key, row["field"]) for row in variables)
    ref = plan["artifact"]
    (acquisition,) = release.catalog.acquisitions(key.dataset, edition=key)
    receipt_id = acquisition["record_id"]
    contract = ref.contract
    year = int(key.edition)
    reviewed = _admit(config, config, store, year, key.scope)
    metadata = release.catalog.metadata(key.dataset, edition=key)
    if (receipt_id != reviewed.receipt or [reviewed.digest] != contract["inputs"]
            or contract["rows"] != reviewed.rows or metadata != reviewed.source.metadata
            or contract["mapping_sha256"] != sha256(canonical_json(reviewed.table).encode()).hexdigest()
            or release.catalog.policy(key.dataset, edition=key) != config.value_policies.get(metadata.value_policy)):
        raise ContractError("Course candidate differs from the packaged source edition and scope")
    graph = store.verify_record_closure([ref.processing_id])
    processing = graph[ref.processing_id]
    parameters = processing["parameters"]
    recipe = parameters["recipe"]
    output = processing["outputs"][ref.output]
    support = LeafColumnSupport.from_contract(contract)
    configs = [graph[identifier]["artifact"]["sha256"] for identifier in processing["inputs"]
               if graph[identifier]["kind"] == "acquisition"
               and graph[identifier]["dataset"] == "reference:peims_course_config"]
    if (processing["processor"] != "teadata.peims_courses" or recipe["operation"] != COURSE_RECIPE
            or recipe["encoding"] != ENCODING or configs != [recipe["config_sha256"]]
            or parameters["leaf_contract_sha256"] != ref.identity
            or sha256(canonical_json(recipe).encode()).hexdigest() != contract["recipe_sha256"]
            or processing["code_version"] != contract["producer"]["code_sha256"]
            or parameters["toolchain"] != contract["producer"]["toolchain"]
            or receipt_id not in processing["inputs"]
            or not set(reviewed.references).issubset(processing["inputs"])
            or graph[receipt_id] != {k: v for k, v in acquisition.items() if k != "record_id"}
            or [acquisition["artifact"]["sha256"]] != contract["inputs"]
            or output["sha256"] != ref.sha256 or output["size_bytes"] != ref.size_bytes
            or support is None or support.context != (SOURCE_RECORD,)
            or {f.field: (f.raw, f.status, f.auxiliary) for f in support.fields}
            != {name: (f"peims_raw_{name}", f"peims_status_{name}", ()) for name in _MEASURE_FIELDS}):
        raise ContractError("Course native leaf does not bind its reviewed source, processing or support contract")
    headings = release.catalog.header_records(key.dataset, edition_key=key)
    names = ("source_scope", "aggregation_level", "school_year",
             *_GEO_FIELDS[key.scope], *_COURSE_FIELDS, *_MEASURE_FIELDS)
    if (contract["key_columns"] != reviewed.keys or metadata.period != DataPeriod("school_year", year - 1, year)
            or {name: item.unit for name, item in metadata.metrics.items()} != COURSE_FACT_UNITS
            or [(row["code"], row["field"]) for row in headings] != list(zip(CSV_HEADERS[key.scope], names, strict=True))):
        raise ContractError("Course native leaf grain, crosswalk or period differs")

    source = store.verify_artifact(reviewed.digest)
    selections: dict[str, list[list[str]]] = {name: [] for name in review} if key.scope != "state" else {}
    selection_position = 5 if key.scope == "campus" else 3
    rows = masked = 0
    with pq.ParquetFile(release.artifact(ref.identity)) as leaf:
        original = _raw_batches(source, year, key.scope, contract["rows"])
        for raw, native in zip(original, leaf.iter_batches(batch_size=_BATCH_ROWS, use_threads=False), strict=True):
            masked += _compare_batch(raw, native, key.scope, rows + 10)
            rows += len(raw)
            for row in raw if selections else ():
                if (selected := selections.get(row[selection_position])) is not None:
                    selected.append(row)
    if rows != contract["rows"]:
        raise ContractError("Course native and source record counts differ")

    controls: list[dict[str, Any]] = []
    if key.scope == "state":
        html, evidence = _open_report(witnesses, review["html"], year, "html")
        published = _course_rows(_html_rows(html, year), "html")
        original_rows = [row[3:] for block in _raw_batches(source, year, key.scope, rows) for row in block]
        if evidence["sha256"] in contract["inputs"] or published != _course_rows(original_rows, "csv"):
            raise ContractError("Course statewide HTML witness differs from the native leaf's source")
        controls.append({**evidence, "rows": rows})
    for district, original_rows in selections.items() if key.scope == "district" else ():
        if not original_rows:
            raise ContractError(f"Course district witness {district} is absent from the bulk source")
        _, name, charter, region = original_rows[0][3:7]
        displayed, evidence = _district_witness(witnesses, review[district], year, district, name, charter, region)
        if (evidence["sha256"] in contract["inputs"] or any(row[4:7] != [name, charter, region] for row in original_rows)
                or _course_rows(displayed, "html") != _course_rows([row[7:] for row in original_rows], "csv")):
            raise ContractError(f"Course district HTML witness {district} differs from the native leaf's source")
        controls.append({**evidence, "district_number": district, "rows": len(displayed)})
    for campus, original_rows in selections.items() if key.scope == "campus" else ():
        witness = _campus_witness(witnesses, review[campus], year, campus, original_rows)
        if witness["sha256"] in contract["inputs"]:
            raise ContractError("Course campus publisher witness is not independent of the bulk source")
        controls.append(witness)
    result = {"edition": key.to_dict(), "artifact_id": ref.identity, "source_receipt": receipt_id,
              "rows": rows, "masked_cells": masked, "raw_cells": rows * len(CSV_HEADERS[key.scope]),
              "publisher_witnesses": controls,
              "publisher_compared_cells": sum(item["rows"] for item in controls) * len(_MEASURE_FIELDS)}
    if key.scope == "campus":
        result.update(raw_parity="PASS_RAW_PARITY", qualification=HOLD_CAMPUS_GEOGRAPHY,
                      qualification_reason="Publisher district-field semantics and repeated HTML headings remain unresolved")
    return result


def qualify_courses(release: DataRelease, store: ProvenanceStore, witnesses: ProvenanceStore) -> dict[str, Any]:
    """Require independent source decoding and the declared publisher selections."""
    reviews = json.loads(files("teadata").joinpath("reference_cases/native_peims_courses.json").read_text())
    editions = [EditionKey.from_dict({k: v for k, v in row.items() if k != "edition_id"})
                for row in release.catalog.editions(include_metadata=False)]
    coverage = {(key.edition, key.scope) for key in editions}
    if (not editions or len(coverage) != len(editions)
            or any(key.dataset != DATASET or key.scope != key.table or key.scope not in reviews
                   or key.edition not in reviews[key.scope] for key in editions)):
        raise ContractError("Only unique course editions with reviewed publisher witnesses are supported")
    release.verify()
    config = load_config()
    results = [_replay(release, store, witnesses, config, key, reviews[key.scope][key.edition]) for key in editions]
    status = HOLD_CAMPUS_GEOGRAPHY if any(key.scope == "campus" for key in editions) else PASS_COURSE_CONTROLS
    return {"status": status, "manifest_sha256": release.manifest_sha256, "build_id": release.build_id,
            "release_activation": False, "rows": sum(item["rows"] for item in results),
            "raw_cells": sum(item["raw_cells"] for item in results),
            "masked_cells": sum(item["masked_cells"] for item in results),
            "publisher_compared_cells": sum(item["publisher_compared_cells"] for item in results),
            "editions": results}
