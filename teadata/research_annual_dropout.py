"""Build receipt-pinned raw TEA Accountability Research annual dropout leaves."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import platform
import re
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import parse_qsl, urlsplit
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.compute as compute
import pyarrow.parquet as pq

from .analytical import write_leaf_artifact
from .column_schema import crosswalk_dataset, parse_column_schema, reader_options, table_input_sha256s
from .dataset_metadata import ArtifactRef, DataPeriod, DatasetMetadata, EditionKey, ResolvedDataset
from .manifest import BuildRecorder, ContractError, canonical_json
from .native_witnesses import _Rows
from .provenance import ProvenanceStore, validate_reference_records
from .release import _install_candidate, publish_release
from .teadata_config import load_config, load_config_bytes

DATASET = "accountability_research_annual_dropout"
YEAR = 2025
PERIOD = DataPeriod("school_year", start_year=2024, end_year=2025)
KEYS = {
    "state": ("grade_span",),
    "region": ("region_number", "grade_span"),
    "county": ("county_number", "grade_span"),
    "district": ("district_number", "grade_span", "calc_for_state_acct"),
    "campus": ("campus_number", "grade_span", "calc_for_state_acct"),
}
_BOOKS = {"county": "cnty_", "district": "dist_", "campus": "camp_"}
_GRADES = {"Grades 7-8": "78", "Grades 9-12": "912", "Grades 7-12": "712"}
_REGIONS = tuple(f"{number:02d}" for number in range(1, 21))
_REPORTS = {
    "all": "Annual Dropout Rates, by Grade Span,",
    "demo": "Annual Dropout Rates, by Race/Ethnicity, Economic Status, Gender, and Grade Span,",
    "prgm": "Annual Dropout Rates, by Program Participation, Student Characteristic, and Grade Span,",
}
_STATISTICS = ("Students", "Dropouts", "Rate (%)")
_BROKER = "https://rptsvr1.tea.texas.gov/cgi/sas/broker"
_LOCAL_WITNESSES = {"district": "057909", "campus": "057909002"}
_SOURCE_ROW = "research_source_row"
_SOURCE_GRADE = "research_grade_label"
_BATCH_ROWS = 512
pc: Any = compute


def _original(store: ProvenanceStore, receipt: str, *, filename: str, workbook: bool = False) -> dict[str, Any]:
    record = store.read_record(receipt)
    source, artifact, parameters = record["source_metadata"], record["artifact"], record["parameters"]
    endpoint = urlsplit(record.get("requested_url") or "")
    if (record["kind"] != "acquisition" or record["method"] != "download" or not record["downloaded_at"]
            or record["requested_year"] != YEAR or record["resolved_year"] != YEAR
            or record["dataset"] != (DATASET if workbook else f"{DATASET}:{filename}")
            or source["version"] != str(YEAR) or source["period"] != PERIOD.to_dict()
            or source["publisher"] != "Texas Education Agency" or artifact["filename"] != filename
            or not 0 < artifact["size_bytes"] <= (24 if workbook else 2) * 1024**2
            or endpoint.scheme != "https" or endpoint.netloc not in {"tea.texas.gov", "rptsvr1.tea.texas.gov"}
            or record["requested_url"] != record["resolved_url"] or parameters.get("http_status") != 200):
        raise ContractError("Annual dropout original does not match the reviewed edition and role")
    return record


def _report_input(store: ProvenanceStore, receipt: str, scope: str, identity: str | None, kind: str,
                  parent: str) -> Path:
    suffix = "" if kind == "all" else f"-{kind}"
    prefix = "esc" if scope == "region" else scope
    name = f"{prefix}-{identity}{suffix}.html" if identity else f"state{suffix.replace('-', '_')}.html"
    record = _original(store, receipt, filename=name)
    endpoint = urlsplit(record["requested_url"])
    parameters = record["parameters"]
    if scope == "state":
        valid_request = (parameters.get("http_method") == "GET" and not endpoint.query
                         and endpoint.netloc == "rptsvr1.tea.texas.gov"
                         and endpoint.path == f"/acctres/drop_annual/2425/{name}")
    else:
        if identity is None:
            raise ContractError("Annual dropout local report requires its publisher identity")
        program = {"region": "regsum", "district": "distsum", "campus": "campsum"}[scope]
        program += suffix.replace("-", "_")
        controls = [("_service", "marykay"), ("_program", "acctres.armast.sas"),
                    ("prgopt", f"drop_annual/2425/{program}.sas"), ("year4", "2425"),
                    ("year2", "25"), ("topic", "dropout")]
        controls += [("rnum", identity)] if scope == "region" else [("search", scope), ("namenum", identity)]
        if kind == "all":
            valid_request = (parameters.get("http_method") == "POST" and not endpoint.query
                             and parameters.get("form_controls") == [list(pair) for pair in controls])
        else:
            observed = parse_qsl(endpoint.query, keep_blank_values=True)
            valid_request = (parameters.get("http_method") == "GET" and len(observed) == len(controls)
                             and dict(observed) == dict(controls))
        valid_request = valid_request and endpoint._replace(query="").geturl() == _BROKER
    if not valid_request or record["source_metadata"]["provenance_ids"] != [parent]:
        raise ContractError("Annual dropout report request or selector lineage differs")
    return store.verify_artifact(record["artifact"]["sha256"])


def _workbook(path: Path, scope: str, mapping: Mapping[str, Any]) -> pa.RecordBatch:
    import fastexcel

    with ZipFile(path) as archive:
        members = archive.infolist()
        names = {member.filename for member in members}
        if (len(names) != len(members) or len(members) > 80 or "xl/workbook.xml" not in names
                or any(name.endswith("vbaProject.bin") for name in names)
                or sum(member.file_size for member in members) > 512 * 1024**2):
            raise ContractError("Annual dropout workbook exceeds its reviewed XLSX container shape")
    reader = fastexcel.read_excel(path)
    sheet = f"{scope.title()}_Dropout_2425"
    if reader.sheet_names != ["Overview", "Data_Dictionary", sheet]:
        raise ContractError("Annual dropout workbook sheet order differs")
    overview = reader.load_sheet_eager("Overview", header_row=None, dtypes="string", dtype_coercion="strict")[0].to_pylist()
    if (len(overview) != 7 or overview[0] != f"Annual Dropout Rates, by Gradespan and {scope.title()}, Texas Public Schools, 2024-25"
            or overview[6] != "File created September 2026"
            or not isinstance(overview[3], str) or any(note not in overview[3] for note in ("-1", "-3", "A dot (.)"))
            or (scope != "county" and (not isinstance(overview[2], str) or any(note not in overview[2] for note in
                ("federal reporting purposes", "statutory exclusions applied for state accountability"))))):
        raise ContractError("Annual dropout workbook overview, period or masking notes differ")
    dictionary = reader.load_sheet_eager("Data_Dictionary", header_row=4, dtypes="string", dtype_coercion="strict")
    if dictionary.schema.names != ["Variable Name", "Description", "PEIMS Element ID"]:
        raise ContractError("Annual dropout workbook dictionary header differs")
    fields = mapping["fields"]
    definitions = {field["code"]: field["publisher_description"] for field in fields}
    observed = [(code, description) for code, description in zip(dictionary[0].to_pylist(), dictionary[1].to_pylist(), strict=True)
                if code in definitions]
    if len(observed) != len(definitions) or dict(observed) != definitions:
        raise ContractError("Annual dropout dictionary does not bind every physical field")
    rows = mapping["layout"]["last_data_record"] - 1
    batch = reader.load_sheet_eager(sheet, dtypes="string", dtype_coercion="strict", n_rows=rows + 1)
    if batch.schema.names != list(definitions) or batch.num_rows != rows:
        raise ContractError("Annual dropout workbook headers or row count differ")
    return batch


def _report_table(rows: list[list[str]], kind: str) -> dict[tuple[str, str], tuple[str, str, str]]:
    header = ["Grade span", "Dropouts", "Students", "Rate (%)"]
    if kind != "all":
        header.insert(0, "Group")
    if rows.count(header) != 1:
        raise ContractError("Annual dropout report table headings differ")
    result = {}
    group = "All students" if kind == "all" else ""
    for row in rows[rows.index(header) + 1:]:
        if len(row) != len(header) or row[-4] not in _GRADES:
            continue
        if kind != "all":
            group = row[0] or group
        grade, numerator, denominator, rate = row[-4:]
        key = group, _GRADES[grade]
        if not group or key in result:
            raise ContractError("Annual dropout report group and grade must be unique")
        result[key] = (denominator, numerator, rate)
    return result


def _report_columns(mapping: Mapping[str, Any]) -> dict[str, tuple[str, str, str]]:
    groups = {}
    for field in mapping["fields"]:
        if field["disposition"] == "fact":
            group, statistic = field["label"].rsplit(" / ", 1)
            fields = groups.setdefault(group, {})
            if statistic in fields:
                raise ContractError("Annual dropout report crosswalk repeats a group/statistic")
            fields[statistic] = field["field"]
    if len(groups) != 32 or any(set(fields) != set(_STATISTICS) for fields in groups.values()):
        raise ContractError("Annual dropout report crosswalk lacks a complete group/statistic projection")
    return {group: (fields["Students"], fields["Dropouts"], fields["Rate (%)"]) for group, fields in groups.items()}


def _direct(store: ProvenanceStore, receipts: Sequence[str], scope: str, mapping: Mapping[str, Any],
            parent: str) -> pa.Table:
    groups = _report_columns(mapping)
    identities = (None,) if scope == "state" else _REGIONS
    if len(receipts) != 3 * len(identities):
        raise ContractError("Annual dropout direct report set is incomplete")
    records = []
    for index, identity in enumerate(identities):
        values = {}
        page_set = receipts[3 * index:3 * index + 3]
        for kind, receipt in zip(_REPORTS, page_set, strict=True):
            path = _report_input(store, receipt, scope, identity, kind, parent if kind == "all" else page_set[0])
            rows = _Rows(path.read_text(encoding="iso-8859-1")).rows
            heading = [_REPORTS[kind] + (" 2024-25," if identity else "")]
            if rows.count(heading) != 1:
                raise ContractError("Annual dropout direct report period heading differs")
            start = rows.index(heading)
            geography = rows[start + 1]
            if (len(geography) != 1 or (identity is None and geography[0] != "Texas Public Schools, 2024-25")
                    or (identity is not None and not geography[0].startswith(f"ESC Region {identity} - "))):
                raise ContractError("Annual dropout direct report geography differs")
            report = _report_table(rows[start + 1:], kind)
            if len(report) != {"all": 3, "demo": 33, "prgm": 60}[kind] or values.keys() & report.keys():
                raise ContractError("Annual dropout report group coverage differs")
            values.update(report)
        if set(values) != {(group, grade) for group in groups for grade in _GRADES.values()}:
            raise ContractError("Annual dropout direct report group/grade crosswalk differs")
        for grade_label, grade in _GRADES.items():
            row = {"grade_span": grade, _SOURCE_GRADE: grade_label}
            if identity is not None:
                row["region_number"] = identity
            for group, fields in groups.items():
                row.update(zip(fields, values[group, grade], strict=True))
            records.append(row)
    schema = pa.schema([(field["field"], pa.string()) for field in mapping["fields"]] + [(_SOURCE_GRADE, pa.string())])
    return pa.Table.from_pylist(records, schema=schema)


def _validate_keys(table: pa.Table, scope: str) -> None:
    for name, pattern in (("county_number", r"[0-9]{3}"), ("district_number", r"[0-9]{6}"),
                          ("campus_number", r"[0-9]{9}"), ("region_number", r"(?:0[1-9]|1[0-9]|20)")):
        if name in table.column_names and any(not isinstance(value, str) or not re.fullmatch(pattern, value)
                                              or not int(value) for value in table[name].to_pylist()):
            raise ContractError(f"Annual dropout has an invalid literal {name}")
    if scope == "campus" and any(campus[:6] != district or campus.endswith("000") for campus, district in
                                 zip(table["campus_number"].to_pylist(), table["district_number"].to_pylist(), strict=True)):
        raise ContractError("Annual dropout campus and district identities disagree")
    keys = table.select(KEYS[scope]).to_pylist()
    if any(row["grade_span"] not in _GRADES.values() or row.get("calc_for_state_acct", "No") not in {"No", "Yes"} for row in keys):
        raise ContractError("Annual dropout grade span or accountability variant is unknown")
    if len(keys) != len({tuple(row.values()) for row in keys}):
        raise ContractError("Annual dropout publisher grain is duplicated")


def _validate_triples(table: pa.Table, triples: Sequence[tuple[str, str, str]]) -> int:
    """Check tokens and observed arithmetic while leaving every raw cell unchanged."""
    checked = 0
    for fields in triples:
        d_raw, n_raw, rate = (table[name] for name in fields)
        count_pattern = r"^(?:[0-9]{1,8}|[1-9][0-9]{0,2}(?:,[0-9]{3}){1,2}|-1|-3)\z"
        rate_pattern = r"^(?:[0-9]{1,3}\.[0-9]|-1|\.|<0\.1)\z"
        for raw, pattern in ((d_raw, count_pattern), (n_raw, count_pattern), (rate, rate_pattern)):
            if raw.null_count or not pc.all(pc.match_substring_regex(raw, pattern)).as_py():
                raise ContractError("Annual dropout has a token outside its published count or rate domain")
        primary, complementary = (pc.equal(d_raw, token) for token in ("-1", "-3"))
        dot, bound = (pc.equal(rate, token) for token in (".", "<0.1"))
        d, n = (pc.cast(pc.replace_substring(raw, ",", ""), pa.int64()) for raw in (d_raw, n_raw))
        if (not pc.all(pc.equal(primary, pc.equal(n_raw, "-1"))).as_py()
                or not pc.all(pc.equal(primary, pc.equal(rate, "-1"))).as_py()
                or not pc.all(pc.equal(complementary, pc.equal(n_raw, "-3"))).as_py()
                or not pc.all(pc.equal(dot, pc.equal(d, 0))).as_py()
                or pc.any(pc.and_(dot, pc.not_equal(n, 0))).as_py()
                or pc.any(pc.and_(pc.invert(pc.or_(primary, complementary)), pc.greater(n, d))).as_py()):
            raise ContractError("Annual dropout numerator, denominator and mask disagree")
        comparable = pc.and_(pc.greater(d, 0), pc.greater_equal(n, 0))
        bounded = pc.and_(comparable, bound)
        if pc.any(pc.and_(bounded, pc.or_(pc.less_equal(n, 0), pc.greater_equal(pc.multiply(n, 1000), d)))).as_py():
            raise ContractError("Annual dropout published upper bound conflicts with observed counts")
        scalar = pc.invert(pc.or_(pc.or_(primary, dot), bound))
        rate_tenths = pc.cast(pc.replace_substring(pc.filter(rate, scalar), ".", ""), pa.int64())
        if pc.any(pc.greater(rate_tenths, 1000)).as_py():
            raise ContractError("Annual dropout rate exceeds 100 percent")
        usable = pc.filter(comparable, scalar)
        observed_d, observed_n = (pc.filter(raw, pc.and_(scalar, comparable)) for raw in (d, n))
        rounded = pc.divide(pc.add(pc.multiply(observed_n, 2000), observed_d), pc.multiply(observed_d, 2))
        if pc.any(pc.not_equal(pc.filter(rate_tenths, usable), rounded)).as_py():
            raise ContractError("Annual dropout rate differs from its observed published counts")
        checked += len(rounded)
    return checked


def _witnesses(store: ProvenanceStore, ids: Sequence[str], scope: str, batch: pa.RecordBatch,
               book_groups: Mapping[str, tuple[str, str, str]], selector: str) -> int:
    entity = _LOCAL_WITNESSES[scope]
    selected = pa.Table.from_batches([batch]).filter(pc.equal(batch[scope.upper()], entity)).to_pylist()
    expected = {(row["CALC_FOR_STATE_ACCT"], row["GRADESPAN"], group): tuple(row[field.upper()] for field in fields)
                for row in selected for group, fields in book_groups.items()}
    if len(ids) != len(_REPORTS) or not expected:
        raise ContractError("Annual dropout witness coverage is incomplete")
    seen = {}
    for kind, receipt in zip(_REPORTS, ids, strict=True):
        source = _report_input(store, receipt, scope, entity, kind, selector if kind == "all" else ids[0])
        raw = source.read_text(encoding="iso-8859-1")
        rows = _Rows(raw).rows
        if ("The rates displayed in Table 1 are calculated for federal reporting purposes." not in raw
                or "The rates in Table 2 are calculated with statutory exclusions applied for state accountability" not in raw
                or rows.count(["Table 1"]) != 1 or rows.count(["Table 2"]) != 1):
            raise ContractError("Annual dropout named report method framing differs")
        first, second = rows.index(["Table 1"]), rows.index(["Table 2"])
        for method, section in (("No", rows[first:second]), ("Yes", rows[second:])):
            if (not any(row == [_REPORTS[kind] + " 2024-25,"] or row[0].startswith(_REPORTS[kind] + " 2024-25,")
                        for row in section if row)
                    or not any(f"{scope.title()} Number {entity}," in cell for row in section for cell in row)):
                raise ContractError("Annual dropout named report period or geography differs")
            for (group, grade), triple in _report_table(section, kind).items():
                key = method, grade, group.casefold()
                normalized = tuple(token.replace(",", "") for token in triple)
                if key in seen or key not in expected or normalized != tuple(token.replace(",", "") for token in expected[key]):
                    raise ContractError("Annual dropout named report differs from its workbook")
                seen[key] = triple
    if seen.keys() != expected.keys():
        raise ContractError("Annual dropout named report omitted a reviewed workbook cell")
    return len(seen) * 3


def materialize_research_annual_dropout(store: ProvenanceStore, *, config: Path, config_sha256: str,
                                       scopes: Sequence[str], output: Path) -> dict[str, Any]:
    """Build a new opt-in bundle of raw source cells from the retained originals."""
    if not scopes or len(set(scopes)) != len(scopes) or set(scopes) - KEYS.keys():
        raise ValueError("Select distinct annual dropout source scopes")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Annual dropout candidate requires a new output directory")
    with config.open("rb") as stream:
        raw = stream.read(2 * 1024**2 + 1)
    if len(raw) > 2 * 1024**2 or sha256(raw).hexdigest() != config_sha256:
        raise ContractError("Annual dropout Config differs from its trusted pin")
    cfg, canonical = load_config_bytes(raw, path=config), load_config()
    source = cfg.resolve_dataset(DATASET, YEAR, exact=True, reader_profile="source")
    reviewed = canonical.resolve_dataset(DATASET, YEAR, exact=True, reader_profile="source")
    resource = cfg.column_contract(DATASET, YEAR, reader_profile="source")
    if (source.to_dict() != reviewed.to_dict() or not cfg.is_source_only(DATASET, YEAR)
            or resource is None or resource != canonical.column_contract(DATASET, YEAR, reader_profile="source")
            or source.metadata.period != PERIOD or source.metadata.metrics or source.metadata.value_policy is not None):
        raise ContractError("Annual dropout requires the packaged exact raw source and mapping")
    tables = {table["table"]: table for table in crosswalk_dataset(parse_column_schema(resource["text"].encode()), DATASET, str(YEAR))["tables"]}
    if set(tables) != set(KEYS):
        raise ContractError("Annual dropout source contract scopes differ")
    state_groups = _report_columns(tables["state"])
    options = reader_options(source.reader, "source")
    references = {ref.reference_id: ref for ref in source.metadata.references}
    reference_ids = [rid for ref in references.values() for rid in ref.provenance_ids]
    graph = store.verify_record_closure(reference_ids)
    validate_reference_records(source, graph.__getitem__, verify_all_claimed=True)
    toolchain = {"python": platform.python_version(), "pyarrow": version("pyarrow"), "fastexcel": version("fastexcel"),
                 "platform": platform.platform(), "architecture": platform.machine()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".research-annual-dropout-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir(mode=0o700)
        target = ProvenanceStore(stage / "provenance")
        recorder = BuildRecorder(cfg, target)
        config_input = Path(temporary) / "source-config.input"
        config_input.write_bytes(raw)
        config_receipt = target.record_acquisition(config_input, ResolvedDataset(f"{DATASET}:config", "reference",
            YEAR, YEAR, "", DatasetMetadata(title="Exact annual dropout source Config", version=str(YEAR))))
        results = []
        for scope in scopes:
            mapping = tables[scope]
            receipts = options["acquisition_ids"][scope]
            witness_ids = options["witness_ids"].get(scope, [])
            roots = list(dict.fromkeys([*receipts, *witness_ids, *reference_ids]))
            for identifier in roots:
                target.copy_record_from(store, identifier)
            records = [target.read_record(identifier) for identifier in receipts]
            inputs = table_input_sha256s(mapping, acquisition_sha256s=[record["artifact"]["sha256"] for record in records])
            witness_cells = 0
            if scope in _BOOKS:
                receipt, = receipts
                record = _original(target, receipt, filename=f"{scope}-data-download-drop-2425.xlsx", workbook=True)
                parameters = record["parameters"]
                if (parameters.get("http_method") != "GET" or parameters.get("scope") != scope
                        or parameters.get("publisher_product") != "Accountability Research"
                        or parameters.get("source_family") != "annual dropout" or parameters.get("period_ending_year") != YEAR
                        or record["source_metadata"]["provenance_ids"] != list(references["publisher_index"].provenance_ids)):
                    raise ContractError("Annual dropout workbook request or publisher lineage differs")
                batch = _workbook(target.verify_artifact(record["artifact"]["sha256"]), scope, mapping)
                prefix = _BOOKS[scope]
                book_groups = {group.casefold(): (prefix + d.removeprefix("state_"), prefix + n.removeprefix("state_"),
                                                  prefix + r.removeprefix("state_"))
                               for group, (d, n, r) in state_groups.items()}
                triples = list(book_groups.values())
                if {name for triple in triples for name in triple} != {
                        field["field"] for field in mapping["fields"] if field["disposition"] == "fact"}:
                    raise ContractError("Annual dropout workbook does not bind the reviewed report groups")
                if scope in _LOCAL_WITNESSES:
                    selector, = references[f"{scope}_selector"].provenance_ids
                    witness_cells = _witnesses(target, witness_ids, scope, batch, book_groups, selector)
                table = pa.Table.from_batches([batch]).rename_columns([field["field"] for field in mapping["fields"]])
                table = table.append_column(_SOURCE_ROW, pa.array(range(2, table.num_rows + 2), type=pa.int64()))
                support = {"fields": {}, "context": [_SOURCE_ROW]}
            else:
                parent, = references["publisher_index" if scope == "state" else "region_selector"].provenance_ids
                table = _direct(target, receipts, scope, mapping, parent)
                triples = list(_report_columns(mapping).values())
                support = {"fields": {}, "context": [_SOURCE_GRADE]}
            _validate_keys(table, scope)
            rates_checked = _validate_triples(table, triples)
            key = EditionKey(DATASET, str(YEAR), EditionKey.source_revision(inputs), scope, scope,
                             "one publisher row per " + " and ".join(KEYS[scope]))
            recipe = {"operation": "explicit_research_annual_dropout_raw_projection", "scope": scope,
                      "config_sha256": config_sha256, "witnesses": witness_ids, "batch_rows": _BATCH_ROWS,
                      "values_interpreted": False}
            contract = recorder.leaf_contract(key, source, native_schema=table.schema.serialize().to_pybytes(),
                columns=table.column_names, key_columns=KEYS[scope], rows=table.num_rows, inputs=inputs,
                recipe=recipe, toolchain=toolchain, column_support=support)
            identity = sha256(canonical_json(contract).encode()).hexdigest()
            leaf = stage / f"{identity}.parquet"
            physical = write_leaf_artifact(leaf, table.to_batches(max_chunksize=_BATCH_ROWS), contract=contract,
                                           batch_bytes=8 * 1024**2)
            if not table.equals(pq.read_table(leaf).replace_schema_metadata(None), check_metadata=True):
                raise ContractError("Annual dropout Parquet readback differs from its raw projection")
            validation = {"raw_projection_cells": table.num_rows * len(mapping["fields"]), "rows": table.num_rows,
                          "observed_rate_checks": rates_checked, "independent_witness_cells": witness_cells}
            processing = target.record_processing(dataset=DATASET, inputs=[*roots, config_receipt], outputs={"leaf": leaf},
                processor="teadata.research_annual_dropout", code_version=recorder.processor_sha256,
                parameters={"leaf_contract_sha256": identity, "recipe": recipe, "toolchain": toolchain},
                validation={**validation, "native_raw_readback_equal": True, "release_activation": False})
            artifact = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name, canonical_json(contract), processing, "leaf")
            recorder.record_edition(key, source, artifacts=[artifact], acquisition_ids=receipts)
            results.append({"key": key.to_dict(), "processing_record_id": processing, **validation})
        catalog = recorder.catalog(schema_version=4)
        publish_release(stage, catalog, schema_version=4)
        result = {"status": "unpublished_native_research_annual_dropout_candidate", "build_id": catalog.build_id,
                  "release_activation": False, "editions": results}
        (stage / "MATERIALIZATION.json").write_text(canonical_json(result) + "\n", encoding="utf-8")
        _install_candidate(stage, output)
        return result
