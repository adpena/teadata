"""Reviewed Attendance and Chronic Absenteeism rates from named 2025 TPRS reports."""

from __future__ import annotations

from decimal import Decimal
from hashlib import sha256
from importlib.resources import files
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import sys
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

from .analytical import write_leaf_artifact
from .column_schema import source_schema_fingerprint
from .dataset_metadata import STATEWIDE_SINGLETON_GRAIN, ArtifactRef, DatasetMetadata, EditionKey, ResolvedDataset
from .manifest import BuildRecorder, ContractError, canonical_json
from .native_witnesses import _Rows
from .provenance import ProvenanceStore
from .release import DataRelease, _install_candidate, publish_release
from .teadata_config import load_config_bytes


DATASET = "tprs_attendance_chronic_absence"
RESOURCE = "methodologies/attendance-chronic-2025.json"
_SCOPES = {"state": "S", "region": "R", "district": "D", "campus": "C"}
_PERCENT = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9])?%")
_LEGEND = ("Indicates there are no students in the group.",
           "Indicates results are masked due to small numbers to protect student confidentiality.",
           "Indicates that the data for this item were statistically improbable or were reported outside a reasonable range.")


def definition() -> dict[str, Any]:
    return json.loads(files("teadata").joinpath(RESOURCE).read_bytes())


def _percent(raw: str, grid: dict[str, Any]) -> tuple[Decimal | None, str]:
    if raw in grid["tokens"]:
        return None, grid["tokens"][raw]
    if not isinstance(raw, str) or _PERCENT.fullmatch(raw) is None:
        raise ContractError("Attendance requires a published percent or its exact reviewed legend token")
    value = Decimal(raw[:-1])
    if not 0 <= value <= 100:
        raise ContractError("Attendance percentage is outside the reviewed 0–100 range")
    return value, "observed"


def read_attendance_report(raw: bytes, *, scope: str, identifier: str, url: str) -> dict[int, dict[str, Any]]:
    """Bind local populations without turning benchmark columns into local values."""
    grid = definition()
    patterns = {"state": "S", "region": r"(?:0[1-9]|1[0-9]|20)", "district": r"[0-9]{6}", "campus": r"[0-9]{9}"}
    if scope not in patterns or re.fullmatch(patterns[scope], identifier) is None:
        raise ContractError("Attendance requires a literal publisher geography identifier")
    locator = urlsplit(url)
    query = parse_qs(locator.query, keep_blank_values=True)
    expected = {"_service": ["marykay"], "_program": ["perfrept.perfmast.sas"],
                "ccyy": ["2025"], "id": [identifier], "lev": [_SCOPES[scope]],
                "prgopt": ["reports/tapr/attendgrad.sas"]}
    if (locator.scheme != "https" or locator.netloc != "rptsvr1.tea.texas.gov" or locator.path != "/cgi/sas/broker"
            or locator.fragment or any(query.get(key) != value for key, value in expected.items())
            or set(query) - set(expected) - {"_debug"} or len(raw) > 2 * 1024**2):
        raise ContractError("Attendance report request has a different publisher, scope, identifier or source edition")
    rows = _Rows(raw.decode("latin-1")).rows
    flattened = " ".join(" ".join(row) for row in rows)
    if (any(phrase not in flattened for phrase in _LEGEND)
            or "2024-25 reporting now reflects Attendance and Chronic Absenteeism data from the 2024-25 and 2023-24 school years." not in flattened):
        raise ContractError("Attendance report lacks the reviewed period and disclosure statements")
    result = {year: {} for year in grid["measurement_years"]}
    for section, prefix in grid["sections"].items():
        positions = [index for index, row in enumerate(rows) if row == [section]]
        if len(positions) != 1:
            raise ContractError("Attendance report has missing or repeated section headings")
        index, = positions
        header = next((row for row in reversed(rows[:index]) if row and row[0] == ""), None)
        local = grid["local_column"][scope]
        expected_headers = grid["population_headers"]
        if (header is None or len(header) != local + 11 or header[1] != "State"
                or header[local + 1:] != expected_headers):
            raise ContractError("Attendance population or benchmark column grid differs")
        if scope == "state" and (identifier != "S" or header[local] != "State"):
            raise ContractError("Attendance state report has a different local column")
        if scope == "region":
            region = re.fullmatch(r"Region\s*([0-9]{1,2})", header[local])
            if region is None or int(region[1]) != int(identifier):
                raise ContractError("Attendance regional column differs from the selected region")
        if scope in {"district", "campus"} and header[local] != scope.title():
            raise ContractError("Attendance local geography role differs")
        for offset, year in enumerate(grid["measurement_years"], 1):
            row = rows[index + offset]
            label = f"{year - 1}-{year % 100:02d}"
            if len(row) != len(header) or row[0] != label:
                raise ContractError("Attendance report has a different actual measurement period or row width")
            for position, population in enumerate(grid["populations"]):
                column = local + position
                field = prefix + "_" + population + "_percent"
                value, status = _percent(row[column], grid)
                result[year][field] = value
                result[year]["attendance_raw_" + field] = row[column]
                result[year]["attendance_status_" + field] = status
                result[year]["attendance_row_" + field] = index + offset + 1
                result[year]["attendance_column_" + field] = column + 1
    return result


def read_attendance_intake(store: ProvenanceStore, document: dict[str, Any]) -> dict[tuple[str, int], pa.Table]:
    """Require complete direct Texas and all twenty ESC reports for this admission."""
    grid = definition()
    reports = document.get("reports", [])
    expected = {("state", "S"), *(("region", f"{region:02d}") for region in range(1, 21))}
    selected = {(item["scope"], item["identifier"]) for item in reports}
    if len(reports) != len(expected) or selected != expected:
        raise ContractError("Attendance admission requires direct Texas and all twenty ESC regions; district/campus delivery remains held")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for item in sorted(reports, key=lambda item: (item["scope"], item["identifier"])):
        record = store.read_record(item["receipt"])
        if (record["kind"] != "acquisition" or record["method"] != "download"
                or record.get("parameters", {}).get("http_method") != "GET"
                or record.get("parameters", {}).get("http_status") != 200
                or record["requested_url"] != record["resolved_url"]):
            raise ContractError("Attendance input lacks its exact original publisher acquisition")
        raw = store.verify_artifact(record["artifact"]["sha256"]).read_bytes()
        parsed = read_attendance_report(raw, scope=item["scope"], identifier=item["identifier"], url=record["requested_url"])
        for year, row in parsed.items():
            identity = {"region_number": item["identifier"]} if item["scope"] == "region" else {}
            grouped.setdefault((item["scope"], year), []).append(
                {**identity, **row, "attendance_report_receipt": item["receipt"],
                 "attendance_report_sha256": record["artifact"]["sha256"]})
    tables = {}
    fields = [prefix + "_" + population + "_percent" for prefix in grid["sections"].values() for population in grid["populations"]]
    for key, rows in grouped.items():
        data = {}
        for column in rows[0]:
            values = [row[column] for row in rows]
            kind = pa.decimal128(4, 1) if column in fields else None
            data[column] = pa.array(values, type=kind)
        tables[key] = pa.table(data)
    for golden in grid["goldens"]:
        table = tables[golden["scope"], golden["year"]]
        index = table["region_number"].to_pylist().index(golden["identifier"]) if golden["scope"] == "region" else 0
        if table["attendance_raw_" + golden["field"]][index].as_py() != golden["raw"]:
            raise ContractError("Attendance differs from the independently frozen publisher report golden")
    return tables


def _support() -> dict[str, Any]:
    grid = definition()
    fields = [prefix + "_" + population + "_percent" for prefix in grid["sections"].values() for population in grid["populations"]]
    return {"fields": {field: {"raw": "attendance_raw_" + field, "status": "attendance_status_" + field,
        "auxiliary": ["attendance_row_" + field, "attendance_column_" + field]} for field in fields},
        "context": ["attendance_report_receipt", "attendance_report_sha256"]}


def _configuration(store: ProvenanceStore, stage: Path, intake: dict[str, Any]) -> bytes:
    grid = definition()
    receipts = [item["receipt"] for item in sorted(intake["reports"], key=lambda item: (item["scope"], item["identifier"]))]
    hashes = [store.read_record(identifier)["artifact"]["sha256"] for identifier in receipts]
    grid_path = stage / "attendance-chronic-semantic-grid.json"
    grid_path.write_bytes(files("teadata").joinpath(RESOURCE).read_bytes())
    grid_id = store.record_acquisition(grid_path, ResolvedDataset("reference:attendance_chronic_grid", "reference", 2025,
        2025, "", DatasetMetadata(title="Reviewed attendance periods, population grid and disclosure rules")))
    glossary = store.read_record(grid["glossary_receipt"])
    if glossary["artifact"]["sha256"] != grid["glossary_sha256"]:
        raise ContractError("Attendance glossary differs from its reviewed publisher bytes")
    references = [{"reference_id": "semantic_grid", "role": "methodology", "status": "available",
        "title": "2025 TPRS attendance semantic grid", "reviewed_sha256": sha256(grid_path.read_bytes()).hexdigest(),
        "path": grid_path.name,
        "provenance_ids": [grid_id]}, {"reference_id": "glossary", "role": "glossary", "status": "available",
        "title": "Comprehensive TPRS glossary, attendance and chronic absence", "url": glossary["requested_url"],
        "reviewed_sha256": grid["glossary_sha256"], "provenance_ids": [grid["glossary_receipt"]]}]
    for item in sorted(intake["reports"], key=lambda item: (item["scope"], item["identifier"])):
        original = store.read_record(item["receipt"])
        references.append({"reference_id": f"publisher_{item['scope']}_{item['identifier']}", "role": "source_notes",
            "status": "available", "title": f"Original {item['scope']} {item['identifier']} publisher report",
            "url": original["requested_url"], "reviewed_sha256": original["artifact"]["sha256"],
            "provenance_ids": [item["receipt"]]})
    mappings = []
    metrics = {}
    for scope in ("state", "region"):
        for year in grid["measurement_years"]:
            fields: list[dict[str, Any]] = [{"field": "region_number", "code": "Region", "label": "Region", "disposition": "identity",
                "dictionary_status": "not_applicable", "publisher_type": None,
                "publisher_description": None}] if scope == "region" else []
            for section, prefix in grid["sections"].items():
                for population in grid["populations"]:
                    field = prefix + "_" + population + "_percent"
                    label = section + " / " + population.replace("_", " ")
                    fields.append({"field": field, "code": label, "label": label, "disposition": "fact",
                        "dictionary_status": "not_documented", "publisher_type": None, "publisher_description": None})
                    metrics[field] = {"field": field, "unit": "percent"}
            fields = [{"position": index, **field} for index, field in enumerate(fields, 1)]
            labels = [field["label"] for field in fields]
            mappings.append({"table": f"{scope}_attendance_{year}", "header_identity": "position",
                "input_sha256s": hashes, "source_schema_sha256": source_schema_fingerprint(labels, labels),
                "metric_period": {"kind": "school_year", "start_year": year - 1, "end_year": year}, "fields": fields})
    schema = stage / "attendance-chronic-crosswalk.json"
    schema.write_text(canonical_json({"schema_version": 2, "mapping_version": 1, "datasets": [
        {"dataset": DATASET, "edition": "2025", "source_only": True, "tables": mappings}]}) + "\n")
    schema_id = store.record_acquisition(schema, ResolvedDataset("reference:attendance_chronic_crosswalk", "reference",
        2025, 2025, "", DatasetMetadata(title="Selected named attendance report crosswalk")))
    references.append({"reference_id": "schema", "role": "schema", "status": "available", "path": schema.name,
                       "reviewed_sha256": sha256(schema.read_bytes()).hexdigest(), "provenance_ids": [schema_id]})
    metadata = {"title": "2025 TPRS Attendance and Chronic Absenteeism, direct Texas and all ESC regions",
        "version": "2025", "publisher": "Texas Education Agency", "period": {"kind": "unknown"},
        "identifiers": ["region_number"], "metrics": metrics, "references": references,
        "notes": "Original publisher edition2025. Field periods are actual2024–25 and2023–24. "
                 "Twenty ESC and direct Texas reports retain all eleven local published populations. "
                 "District/campus delivery remains held. Benchmarks are retained in the original report and are not local subgroup values. "
                 "Native-source acquisitions are local re-observations of retained bytes; exact original download records remain in references."}
    rebound = []
    for identifier in receipts:
        original = store.read_record(identifier)
        path = store.verify_artifact(original["artifact"]["sha256"])
        rebound.append(store.record_acquisition(path, ResolvedDataset(DATASET, "data_sources", 2025, 2025, "",
            DatasetMetadata.from_dict(metadata)), parameters={"original_publisher_receipt": identifier}))
    metadata["provenance_ids"] = rebound
    return (canonical_json({"data_sources": {DATASET: {2025: {"path": str(store.verify_artifact(hashes[0])),
        "metadata": metadata, "reader": {"profiles": {"source": {"column_schema": schema.name}}}}}}}) + "\n").encode()


def materialize_attendance_chronic(store: ProvenanceStore, *, intake: Path, intake_sha256: str, output: Path) -> dict[str, Any]:
    raw_intake = intake.read_bytes()
    if sha256(raw_intake).hexdigest() != intake_sha256:
        raise ContractError("Attendance intake differs from its selected sealed identity")
    document = json.loads(raw_intake)
    tables = read_attendance_intake(store, document)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".attendance-chronic-", dir=output.parent) as directory:
        stage = Path(directory) / "bundle"
        stage.mkdir()
        config_bytes = _configuration(store, stage, document)
        config_path = stage / "source-config.input"
        config_path.write_bytes(config_bytes)
        config = load_config_bytes(config_bytes, path=config_path)
        source = config.resolve_dataset(DATASET, 2025, exact=True, reader_profile="source")
        roots = {*source.metadata.provenance_ids, *(identifier for reference in source.metadata.references for identifier in reference.provenance_ids)}
        target = ProvenanceStore(stage / "provenance")
        for identifier in sorted(roots):
            target.copy_record_from(store, identifier)
        config_id = target.record_acquisition(config_path, ResolvedDataset("reference:attendance_chronic_config", "reference",
            2025, 2025, "", DatasetMetadata(title="Exact attendance/chronic source configuration")))
        intake_id = target.record_acquisition(intake, ResolvedDataset("reference:attendance_chronic_intake", "reference", 2025,
            2025, "", DatasetMetadata(title="Complete state and twenty-ESC publisher report selection")))
        recorder = BuildRecorder(config, target)
        toolchain = {"python": sys.version, "pyarrow": version("pyarrow"), "platform": platform.platform(),
                     "architecture": platform.machine()}
        hashes = tuple(dict.fromkeys(store.read_record(identifier)["artifact"]["sha256"] for identifier in source.metadata.provenance_ids))
        for (scope, year), table in tables.items():
            grain = STATEWIDE_SINGLETON_GRAIN if scope == "state" else "one publisher row per region_number"
            key = EditionKey(DATASET, "2025", EditionKey.source_revision(hashes), scope, f"{scope}_attendance_{year}", grain)
            recipe = {"operation": "named_attendance_chronic_rates", "intake_sha256": intake_sha256,
                      "measurement_school_year": year, "scope": scope, "benchmark_columns_are_local": False}
            contract = recorder.leaf_contract(key, source, native_schema=table.schema.serialize().to_pybytes(),
                columns=table.column_names, key_columns=["region_number"] if scope == "region" else [],
                rows=table.num_rows, inputs=list(hashes), recipe=recipe, toolchain=toolchain, column_support=_support())
            serialized = canonical_json(contract)
            leaf = stage / (sha256(serialized.encode()).hexdigest() + ".parquet")
            physical = write_leaf_artifact(leaf, table.to_batches(), contract=contract, batch_bytes=4 * 1024**2)
            if not table.equals(pq.read_table(leaf).replace_schema_metadata(None)):
                raise ContractError("Attendance typed Parquet differs from publisher rates")
            processing = target.record_processing(dataset=DATASET, inputs=[*sorted(roots), config_id, intake_id], outputs={"leaf": leaf},
                processor="teadata.attendance_chronic", code_version=recorder.processor_sha256,
                parameters={"recipe": recipe, "leaf_contract_sha256": leaf.stem, "toolchain": toolchain},
                validation={"published_percentages_retained": True, "release_activation": False})
            ref = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name, serialized, processing, "leaf")
            recorder.record_edition(key, source, artifacts=[ref], acquisition_ids=source.metadata.provenance_ids)
        manifest = publish_release(stage, recorder.catalog(schema_version=4), schema_version=4)
        release = DataRelease(manifest, offline=True)
        release.verify()
        result = {"status": "PREPARED_ATTENDANCE_CHRONIC", "manifest_sha256": release.manifest_sha256,
                  "build_id": release.build_id, "scopes": ["state", "region"], "measurement_years": [2024, 2025],
                  "held_scopes": ["district", "campus"]}
        (stage / "MATERIALIZATION.json").write_text(canonical_json(result) + "\n")
        _install_candidate(stage, output)
    return result


def retained_attendance_evidence(release: DataRelease, store: ProvenanceStore) -> list[str]:
    return sorted(store.verify_record_closure([
        release.catalog.artifact_ref(identifier).processing_id for identifier in release.catalog.to_dict()["artifacts"]]))


def qualify_attendance_chronic(release: DataRelease, store: ProvenanceStore) -> dict[str, Any]:
    release.verify()
    refs = [release.catalog.artifact_ref(identifier) for identifier in release.catalog.to_dict()["artifacts"]]
    expected_keys = {(scope, f"{scope}_attendance_{year}") for scope in ("state", "region") for year in (2024, 2025)}
    if len(refs) != 4 or {(ref.edition.scope, ref.edition.table) for ref in refs} != expected_keys or not release.offline:
        raise ContractError("Attendance requires the exact complete direct Texas/twenty-ESC offline selection")
    intake_ids = set()
    for ref in refs:
        records = store.verify_record_closure([ref.processing_id])
        processing = records[ref.processing_id]
        embedded = json.loads(release.catalog.edition_metadata_bytes(ref.edition.identity))["provenance"]
        intake_ids.update(identifier for identifier, record in records.items()
                          if record["dataset"] == "reference:attendance_chronic_intake")
        if (processing["processor"] != "teadata.attendance_chronic"
                or processing["code_version"] != ref.contract["producer"]["code_sha256"]
                or processing["parameters"]["toolchain"] != ref.contract["producer"]["toolchain"]
                or ref.edition.dataset != DATASET or ref.edition.edition != "2025"
                or records != embedded
                or ref.contract["column_support"] != _support()):
            raise ContractError("Attendance native field or processing lineage differs")
    if len(intake_ids) != 1:
        raise ContractError("Attendance requires one exact complete publisher intake")
    intake_id, = intake_ids
    document = json.loads(store.verify_artifact(store.read_record(intake_id)["artifact"]["sha256"]).read_bytes())
    tables = read_attendance_intake(store, document)
    for ref in refs:
        year = int(ref.edition.table.rsplit("_", 1)[1])
        expected = tables[ref.edition.scope, year]
        fields = list(_support()["fields"])
        periods = release.catalog._field_periods(ref.edition, tuple(fields))
        metadata = release.catalog.metadata(DATASET, edition=ref.edition)
        references = {reference.reference_id: reference for reference in metadata.references}
        if (any(period.kind != "school_year" or period.end_year != year for period in periods.values())
                or any(metadata.metrics[field].unit != "percent" for field in fields)
                or references["glossary"].reviewed_sha256 != definition()["glossary_sha256"]
                or references["semantic_grid"].reviewed_sha256 != sha256(files("teadata").joinpath(RESOURCE).read_bytes()).hexdigest()):
            raise ContractError("Attendance field period, unit, glossary or semantic grid differs")
        physical = release.analytical_dataset(ref.identity).to_table().replace_schema_metadata(None)
        public = release.table(ref.edition).to_arrow().select(expected.column_names).replace_schema_metadata(None)
        if not physical.equals(expected) or not public.equals(expected):
            raise ContractError("Attendance native/public rates, masks, source rows or benchmark roles differ")
    return {"status": "PASS_REVIEWED_NATIVE", "manifest_sha256": release.manifest_sha256,
            "build_id": release.build_id, "reports": 21, "periods": [2024, 2025], "held_scopes": ["district", "campus"]}
