"""Build explicit Class of 2025 TEA Accountability Research graduation leaves."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow as pa
import pyarrow.compute as compute
import pyarrow.parquet as pq

from .analytical import write_leaf_artifact
from .column_schema import crosswalk_dataset, parse_column_schema, reader_options, table_input_sha256s
from .dataset_metadata import STATEWIDE_SINGLETON_GRAIN, ArtifactRef, DataPeriod, DatasetMetadata, EditionKey, ResolvedDataset
from .manifest import BuildRecorder, ContractError, canonical_json
from .native_witnesses import _Rows
from .provenance import ProvenanceStore, validate_reference_records
from .release import _install_candidate, publish_release
from .teadata_config import load_config, load_config_bytes
from .values import ValuePolicy

DATASET = "accountability_research_four_year_graduation"
YEAR = 2025
CUTOFF = DataPeriod("as_of", as_of="2025-08-31")
SHEET = "Comp_2025_4yr"
POLICY = "research_graduation_class_2025"
SOURCE_ROW = "research_source_row"
_KEYS = {"state": [], "district": ["district_number", "calc_for_state_acct"],
         "campus": ["campus_number", "calc_for_state_acct"]}
_GROUPS = {"ALL": "All students", "ECN": "Economically disadvantaged",
           "LEP": "Emergent bilingual in last year", "SPE": "Special education"}
_STATE_REPORTS = {"state-all.html": ("ALL",), "state-demo.html": ("ECN",), "state-program.html": ("LEP", "SPE")}
_STATE_HEAD = "Grade 9 Four-Year Longitudinal Graduation and Dropout Rates"
_TRIPLE = ("D", "N_GRAD", "R_GRAD")
_AEA_CODES = ("CAMP_ALLN_GRAD_AEA", "CAMP_ALLN_CONT_AEA", "CAMP_ALLN_TXCHSE_AEA",
              "CAMP_ALLN_DROP_AEA", "CAMP_ALLN_CMP4", "CAMP_ALLR_CMP4")
_MAX_XLSX_BYTES = 16 * 1024**2
_BATCH_ROWS = 512
pc: Any = compute


def statistic_codes(scope: str, group: str) -> tuple[str, ...]:
    prefix = {"state": "STATE", "district": "DIST", "campus": "CAMP"}[scope]
    return tuple(f"{prefix}_{group}{suffix}" for suffix in _TRIPLE)


def _receipt_source(store: ProvenanceStore, receipt: str, *, raw: bool) -> Path:
    record = store.read_record(receipt)
    artifact = record["artifact"]
    expected_dataset = DATASET if raw else f"{DATASET}:{artifact['filename']}"
    if (record["kind"] != "acquisition" or record["method"] != "download"
            or record["dataset"] != expected_dataset or record["downloaded_at"] is None
            or record["source_metadata"]["publisher"] != "Texas Education Agency"
            or record["source_metadata"]["version"] != str(YEAR)
            or record["parameters"].get("cohort") != YEAR
            or record["parameters"].get("publisher_product") != "Accountability Research"
            or record["parameters"].get("http_method") != "GET" or record["parameters"].get("http_status") != 200
            or record["requested_url"] != record["resolved_url"]
            or not 0 < artifact["size_bytes"] <= _MAX_XLSX_BYTES):
        raise ContractError("Research graduation receipt is outside its reviewed source selection")
    return store.verify_artifact(artifact["sha256"])


def _xlsx(path: Path, scope: str, mapping: Mapping[str, Any]) -> pa.RecordBatch:
    import fastexcel

    reader = fastexcel.read_excel(path)
    if reader.sheet_names != ["Overview", "Data_Dictionary", SHEET]:
        raise ContractError("Research workbook worksheets differ from the reviewed edition")
    overview = reader.load_sheet_eager("Overview", header_row=None, dtypes="string", dtype_coercion="strict")[0]
    if (len(overview) != 7 or f"by {scope.title()}, Texas Public Schools, Class of 2025" not in overview[0].as_py()
            or "federal accountability purposes" not in overview[2].as_py()
            or "statutory exclusions applied for state accountability" not in overview[2].as_py()
            or any(note not in overview[3].as_py() for note in ("-1", "-3", "cross-masked", "A dot (.)"))
            or overview[6].as_py() != "File created September 2026"):
        raise ContractError("Research workbook edition, scope or masking notes differ")
    dictionary = reader.load_sheet_eager("Data_Dictionary", header_row=4, dtypes="string", dtype_coercion="strict")
    if dictionary.schema.names != ["Variable Name", "Description", "PEIMS Element IDa"]:
        raise ContractError("Research workbook dictionary headings differ")
    fields = mapping["fields"]
    codes = [field["code"] for field in fields]
    known = set(codes)
    definitions = [(code, description) for code, description in zip(dictionary[0].to_pylist(), dictionary[1].to_pylist(), strict=True)
                   if code in known]
    if len(definitions) != len(fields) or dict(definitions) != {field["code"]: field["publisher_description"] for field in fields}:
        raise ContractError("Research workbook dictionary does not account for every reviewed physical field")
    layout = mapping["layout"]
    count = layout["last_data_record"] - 1
    batch = reader.load_sheet_eager(SHEET, dtypes="string", dtype_coercion="strict", n_rows=count + 1)
    if batch.schema.names != codes or batch.num_rows != count:
        raise ContractError("Research workbook header order or row count differs from the reviewed edition")
    return batch


def _state_batch(paths: Sequence[Path], mapping: Mapping[str, Any]) -> pa.RecordBatch:
    values = {}
    for (name, groups), path in zip(_STATE_REPORTS.items(), paths, strict=True):
        rows = _Rows(path.read_text(encoding="iso-8859-1")).rows
        columns = ["Class", *["Number", "Rate (%)"] * 5]
        if name == "state-all.html":
            heading = [f"{_STATE_HEAD}, Texas Public Schools, Class of 2025"]
            if rows.count(heading) != 1 or rows.count(columns) != 1 or rows.index(columns) <= rows.index(heading):
                raise ContractError("Statewide overall graduation report edition or headings differ")
            published = rows[rows.index(columns) + 1:rows.index(columns) + 2]
        else:
            heading = [[f"{_STATE_HEAD},"], ["by Race/Ethnicity, Economic Status, and Gender," if name == "state-demo.html"
                                              else "by Program Participation and Student Characteristic,"],
                       ["Texas Public Schools, Class of 2025"]]
            index = rows.index(heading[0])
            if rows.count(heading[0]) != 1 or rows[index:index + 3] != heading or rows.count(["Group", *columns]) != 1:
                raise ContractError("Statewide graduation subgroup report edition or headings differ")
        for group in groups:
            selected = published if name == "state-all.html" else [row[1:] for row in rows if row and row[0] == _GROUPS[group]]
            if len(selected) != 1 or len(selected[0]) != len(columns):
                raise ContractError("Statewide report lacks a unique reviewed graduation row")
            values.update(zip(statistic_codes("state", group), selected[0][:3], strict=True))
    codes = [field["code"] for field in mapping["fields"]]
    if list(values) != codes:
        raise ContractError("Statewide report bindings differ from the reviewed crosswalk")
    return pa.record_batch([pa.array([values[code]], type=pa.string()) for code in codes], names=codes)


def _keys(columns: Mapping[str, pa.Array], scope: str) -> None:
    if scope == "state":
        return
    district_ids = columns["district_number"].to_pylist()
    if any(not isinstance(value, str) or len(value) != 6 or not value.isascii() or not value.isdecimal()
           or value == "000000" for value in district_ids):
        raise ContractError("Research graduation contains an invalid six-digit district ID")
    methods = columns["calc_for_state_acct"].to_pylist()
    identifiers = district_ids
    flags = [""] * len(methods)
    if scope == "campus":
        identifiers = columns["campus_number"].to_pylist()
        flags = columns["aea"].to_pylist()
        if any(not isinstance(value, str) or len(value) != 9 or not value.isascii() or not value.isdecimal()
               or value[:6] != parent or value.endswith("000") for value, parent in zip(identifiers, district_ids, strict=True)):
            raise ContractError("Research graduation campus IDs do not bind to the literal district prefix")
    pairs = {}
    census = Counter()
    for identifier, method, flag in zip(identifiers, methods, flags, strict=True):
        if method not in {"No", "Yes"} or flag not in {"", "No", "Yes"} or (identifier, method) in pairs:
            raise ContractError("Research graduation method, AEA flag or entity-method grain differs")
        pairs[identifier, method] = flag
        census[(method, flag)] += 1
    expected = {("No", ""): 1082, ("Yes", ""): 1082} if scope == "district" else {
        ("No", "No"): 1707, ("Yes", "No"): 1707, ("No", "Yes"): 238, ("Yes", "Yes"): 238, ("No", ""): 6,
    }
    if census != expected or any(pairs.get((identifier, "No" if method == "Yes" else "Yes")) != flag
                                 for identifier, method, flag in zip(identifiers, methods, flags, strict=True)
                                 if scope == "district" or flag):
        raise ContractError("Research graduation pairing coverage differs from the reviewed edition")


def _projection(batch: pa.RecordBatch, mapping: Mapping[str, Any], scope: str, *,
                units: Mapping[str, str | None], policy: ValuePolicy) -> tuple[pa.Table, dict[str, Any]]:
    fields = {field["code"]: field["field"] for field in mapping["fields"]}
    columns = dict(zip(fields.values(), batch.columns, strict=True))
    _keys(columns, scope)
    support = {}
    for field in fields.values():
        unit = units.get(field)
        if unit is None:
            continue
        if unit not in {"count", "percent"}:
            raise ContractError(f"Unreviewed research graduation unit in {field}")
        raw = columns[field]
        normalized = pc.replace_substring(raw, ",", "")
        numeric = pc.match_substring_regex(raw, r"^[0-9]+\.[0-9]$" if unit == "percent"
                                          else r"^(?:[0-9]+|[1-9][0-9]{0,2}(?:,[0-9]{3})+)$")
        status = pc.if_else(numeric, "observed", None)
        for token, rule in policy.tokens.items():
            if token not in {"na", "n/a"} or field.endswith(("_aea", "_cmp4")):
                status = pc.if_else(pc.equal(pc.ascii_lower(raw), token), rule.status.value, status)
        if status.null_count:
            raise ContractError(f"Unreviewed research graduation token in {field}")
        dtype = pa.decimal128(4, 1) if unit == "percent" else pa.int64()
        value = pc.cast(pc.if_else(numeric, normalized, None), dtype)
        if unit == "percent" and pc.any(pc.greater(value, 100)).as_py():
            raise ContractError(f"Research graduation percent exceeds 100 in {field}")
        columns[field] = value
        raw_key, status_key = f"research_raw_{field}", f"research_status_{field}"
        columns[raw_key], columns[status_key] = raw, status
        support[field] = {"raw": raw_key, "status": status_key}
    if scope == "campus":
        aea_eligible = pc.and_(pc.equal(columns["aea"], "Yes"), pc.equal(columns["calc_for_state_acct"], "Yes"))
        for code in _AEA_CODES:
            field = fields[code]
            status = columns[support[field]["status"]]
            if pc.any(pc.and_(pc.invert(aea_eligible), pc.not_equal(status, "not_applicable"))).as_py():
                raise ContractError("Research graduation contains an AEA fact outside the statutory AEA rows")
    for group in _GROUPS:
        denominator, numerator, rate = (columns[fields[code]] for code in statistic_codes(scope, group))
        jointly_observed = pc.and_(pc.and_(pc.is_valid(numerator), pc.is_valid(denominator)), pc.is_valid(rate))
        d, n, r = (pc.filter(value, jointly_observed) for value in (denominator, numerator, rate))
        if pc.any(pc.or_(pc.less_equal(d, 0), pc.greater(n, d))).as_py():
            raise ContractError("Research graduation numerator or denominator is outside the observed cohort")
        tenths = pc.divide(pc.add(pc.multiply(n, 2000), d), pc.multiply(d, 2))
        if pc.any(pc.not_equal(tenths, pc.cast(pc.multiply(r, 10), pa.int64()))).as_py():
            raise ContractError("Research graduation published rate differs from its observed cohort counts")
    context = []
    if scope != "state":
        columns[SOURCE_ROW] = pa.array(range(2, batch.num_rows + 2), type=pa.int64())
        context.append(SOURCE_ROW)
    return pa.table(columns), {"fields": support, "context": context}


def _witness(path: Path, batch: pa.RecordBatch, scope: str, entity: str) -> int:
    raw = path.read_text(encoding="iso-8859-1")
    if "The rates displayed in Table 1 are calculated for federal accountability purposes." not in raw:
        raise ContractError("Research witness does not identify the federal accountability table")
    report = _Rows(raw).rows
    local = {}
    key = "DISTRICT" if scope == "district" else "CAMPUS"
    selection = pc.equal(batch[key], entity)
    fields = ["CALC_FOR_STATE_ACCT", *statistic_codes(scope, "ALL")]
    if scope == "campus":
        fields.extend(["AEA", *_AEA_CODES])
    selected = pa.record_batch([pc.filter(batch[name], selection) for name in fields], names=fields)
    for row in selected.to_pylist():
        local[row["CALC_FOR_STATE_ACCT"]] = row
    if not local:
        raise ContractError("Research report witness identifier is missing in the workbook")
    total = 0
    for index, method in enumerate(("No", "Yes", "AEA"), 1):
        if method == "AEA" and scope != "campus":
            continue
        heading = [f"Table {index}"]
        if report.count(heading) != 1:
            raise ContractError("Research witness table is missing or repeated")
        start = report.index(heading)
        block = report[start:start + 9]
        if (block[1] != [f"{_STATE_HEAD}, Class of 2025,"]
                or not any(len(row) == 1 and f"{scope.title()} Number {entity}," in row[0] for row in block)):
            raise ContractError("Research witness class or geography differs")
        candidates = [i for i, row in enumerate(block) if row[:2] == ["Class", "Number"]]
        if len(candidates) != 1:
            raise ContractError("Research witness column headings differ")
        row = block[candidates[0] + 1]
        source = local.get(method if method != "AEA" else "Yes")
        if method == "AEA" and (source is None or source["AEA"] != "Yes"):
            continue
        if source is None:
            if any(row):
                raise ContractError("Research witness presents a missing workbook variant as observed")
            continue
        names = fields[1:4] if method != "AEA" else [fields[1], *fields[5:]]
        if [str(value).replace(",", "") for value in row[:len(names)]] != [
                str(source[name]).replace(",", "") for name in names]:
            raise ContractError("Research witness values differ from the workbook")
        total += len(names)
    return total


def materialize_research_graduation(store: ProvenanceStore, *, config: Path, config_sha256: str,
                                   scopes: Sequence[str], output: Path) -> dict[str, Any]:
    """Build a new exact source catalog without activating TAPR, TPRS, or the default snapshot."""
    if not scopes or len(set(scopes)) != len(scopes) or set(scopes) - _KEYS.keys():
        raise ValueError("Select distinct state, district or campus scopes")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Research graduation candidate requires a new output directory")
    with config.open("rb") as stream:
        raw = stream.read(2 * 1024**2 + 1)
    if len(raw) > 2 * 1024**2 or sha256(raw).hexdigest() != config_sha256:
        raise ContractError("Research graduation Config differs from its trusted pin")
    cfg = load_config_bytes(raw, path=config)
    canonical = load_config()
    source = cfg.resolve_dataset(DATASET, YEAR, exact=True, reader_profile="source")
    reviewed = canonical.resolve_dataset(DATASET, YEAR, exact=True, reader_profile="source")
    resource = cfg.column_contract(DATASET, YEAR, reader_profile="source")
    policy = cfg.value_policies.get(POLICY)
    # This reader opens pinned acquisitions, not the resolved path placeholder.
    if (replace(source, path=reviewed.path) != reviewed or not cfg.is_source_only(DATASET, YEAR)
            or resource is None or resource != canonical.column_contract(DATASET, YEAR, reader_profile="source")
            or policy is None or policy != canonical.value_policies.get(POLICY) or source.metadata.value_policy != POLICY):
        raise ContractError("Research graduation requires the packaged exact source, mapping and value policy")
    tables = crosswalk_dataset(parse_column_schema(resource["text"].encode()), DATASET, str(YEAR))["tables"]
    units = {metric.field: metric.unit for metric in source.metadata.metrics.values()}
    metrics = {metric.field: metric for metric in source.metadata.metrics.values()}
    for table in tables:
        fields = {field["code"]: field["field"] for field in table["fields"]}
        scope = table["table"]
        for group in _GROUPS:
            for code, unit in zip(statistic_codes(scope, group), ("count", "count", "percent"), strict=True):
                if fields[code] not in metrics or (units[fields[code]], metrics[fields[code]].period) != (unit, CUTOFF):
                    raise ContractError("Research graduation field units or diploma cutoff differ")
    options = reader_options(source.reader, "source")
    references = [identifier for ref in source.metadata.references for identifier in ref.provenance_ids]
    graph = store.verify_record_closure(references)
    validate_reference_records(source, graph.__getitem__, verify_all_claimed=True)
    toolchain = {"python": sys.version, "pyarrow": version("pyarrow"), "fastexcel": version("fastexcel"),
                 "platform": platform.platform(), "architecture": platform.machine()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".research-graduation-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir(mode=0o700)
        target = ProvenanceStore(stage / "provenance")
        recorder = BuildRecorder(cfg, target)
        config_input = Path(temporary) / "source-config.input"
        config_input.write_bytes(raw)
        config_receipt = target.record_acquisition(config_input, ResolvedDataset(f"{DATASET}:config", "reference",
            YEAR, YEAR, "", DatasetMetadata(title="Exact Class of 2025 graduation source Config", version=str(YEAR))))
        retained = set()
        results = []
        for table in tables:
            scope = table["table"]
            if scope not in scopes:
                continue
            receipts = options["acquisition_ids"][scope]
            witnesses = options["witness_ids"].get(scope, {})
            roots = list(dict.fromkeys([*receipts, *references, *witnesses.values()]))
            for identifier in roots:
                if identifier not in retained:
                    target.copy_record_from(store, identifier)
                    retained.add(identifier)
            paths = [_receipt_source(target, identifier, raw=True) for identifier in receipts]
            hashes = [path.name for path in paths]
            inputs = table_input_sha256s(table, acquisition_sha256s=hashes)
            if scope == "state":
                if [target.read_record(identifier)["artifact"]["filename"] for identifier in receipts] != list(_STATE_REPORTS):
                    raise ContractError("Research statewide source reports are out of order")
                batch = _state_batch(paths, table)
            else:
                path, = paths
                batch = _xlsx(path, scope, table)
            checked = sum(_witness(_receipt_source(target, identifier, raw=False), batch, scope, entity)
                          for entity, identifier in witnesses.items())
            projected, support = _projection(batch, table, scope, units=units, policy=policy)
            keys = _KEYS[scope]
            grain = STATEWIDE_SINGLETON_GRAIN if not keys else "one publisher row per " + " and ".join(keys)
            key = EditionKey(DATASET, str(YEAR), EditionKey.source_revision(inputs), scope, scope, grain)
            recipe = {"operation": "explicit_research_graduation_projection", "config_sha256": config_sha256,
                      "raw_representation": "publisher HTML text" if scope == "state" else "Calamine string cells",
                      "scope": scope, "witnesses": dict(witnesses), "batch_rows": _BATCH_ROWS}
            contract = recorder.leaf_contract(key, source, native_schema=projected.schema.serialize().to_pybytes(),
                columns=projected.column_names, key_columns=keys, rows=projected.num_rows, inputs=inputs,
                recipe=recipe, toolchain=toolchain, column_support=support)
            identity = sha256(canonical_json(contract).encode()).hexdigest()
            leaf = stage / f"{identity}.parquet"
            physical = write_leaf_artifact(leaf, projected.to_batches(max_chunksize=_BATCH_ROWS), contract=contract,
                                           batch_bytes=8 * 1024**2)
            if not projected.equals(pq.read_table(leaf).replace_schema_metadata(None), check_metadata=True):
                raise ContractError("Research graduation Arrow readback differs from its source projection")
            processing = target.record_processing(dataset=DATASET, inputs=[*roots, config_receipt], outputs={"leaf": leaf},
                processor="teadata.research_graduation", code_version=recorder.processor_sha256,
                parameters={"leaf_contract_sha256": identity, "recipe": recipe, "toolchain": toolchain},
                validation={"native_projection_readback_equal": True, "rows": projected.num_rows,
                            "independent_witness_cells": checked, "release_activation": False})
            artifact = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name, canonical_json(contract), processing, "leaf")
            recorder.record_edition(key, source, artifacts=[artifact], acquisition_ids=receipts)
            results.append({"key": key.to_dict(), "rows": projected.num_rows, "publisher_fields": len(table["fields"]),
                            "witness_cells": checked, "processing_record_id": processing})
        if len(results) != len(scopes):
            raise ContractError("Research graduation source contract omitted a selected scope")
        catalog = recorder.catalog(schema_version=4)
        publish_release(stage, catalog, schema_version=4)
        result = {"status": "unpublished_native_research_graduation_candidate", "build_id": catalog.build_id,
                  "release_activation": False, "editions": results}
        (stage / "MATERIALIZATION.json").write_text(canonical_json(result) + "\n", encoding="utf-8")
        _install_candidate(stage, output)
        return result
