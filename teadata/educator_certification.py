"""Materialize qualified statewide initial certification totals."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from contextlib import closing
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
from typing import Any

from openpyxl import load_workbook
import pyarrow as pa
import pyarrow.parquet as pq

from .analytical import write_leaf_artifact
from .column_schema import crosswalk_dataset, parse_column_schema, reader_options
from .dataset_metadata import (
    STATEWIDE_SINGLETON_GRAIN,
    ArtifactRef,
    DataPeriod,
    DatasetMetadata,
    EditionKey,
    MetricDefinition,
    ResolvedDataset,
)
from .educator_retention import SOURCE_WITNESS_RECEIPT, _source_witness
from .manifest import BuildRecorder, ContractError, canonical_json, file_identity
from .provenance import ProvenanceStore, validate_reference_records
from .release import _install_candidate, publish_release
from .teadata_config import load_config_bytes
from .values import ValueStatus

REPORT_YEAR = 2026
EDITION = "2026-09"
WORKBOOK_RECEIPT = "4970586447fe1ade12160e9833ce9d836e3dd062f4657f7d85e5841a5cae576c"
WORKBOOK_SHA = "148a8e20669aba384d0e398d093829f6db9b95a5d2391bd5f38d3051fb0c56bb"
WORKBOOK_URL = "https://tea4avcastro.tea.state.tx.us/oess/edrs/newly-certified-by-role-dashboards/downloads/newly-certified-by-role-workbook.xlsx"
ADMITTED = {role: tuple(range(2016, 2027)) for role in ("Teacher", "Principal", "Superintendent")}
# These are revisions between two direct statewide totals. Their detailed
# publisher rows still have ambiguous repeated keys.
PUBLISHER_REVISIONS = {("Teacher", "2020-21"): (26085, 26086), ("Teacher", "2023-24"): (17797, 17801)}
ROLLUP_SHEET = "Rolled up Data"
ROLLUP_HEADERS = ("Academic Year", "Preparation Route", "Role", "Gender", "Ethnicity", "Count Certified")
DETAIL_HEADERS = ("Academic Year", "Preparation Route", "Role", "Gender", "Race/Ethnicity", "Count Certified")
FIELDS = ("academic_year", "preparation_route", "certification_role", "gender", "ethnicity", "certification_count")
SOURCE_ROW = "publisher_worksheet_row"
COUNT = FIELDS[-1]
RAW = f"raw_{COUNT}"
STATUS = f"status_{COUNT}"
METRICS = {COUNT: MetricDefinition(COUNT, unit="count")}
CONFIG = Path(__file__).with_name("teadata_sources.yaml")
_SUPPORT = {"fields": {COUNT: {"raw": RAW, "status": STATUS}}, "context": [SOURCE_ROW]}
_SCHEMA = pa.schema([
    *(pa.field(name, pa.string(), nullable=False) for name in FIELDS[:-1]),
    pa.field(COUNT, pa.int64(), nullable=False),
    pa.field(RAW, pa.string(), nullable=False),
    pa.field(STATUS, pa.string(), nullable=False),
    pa.field(SOURCE_ROW, pa.int32(), nullable=False),
])


def _dataset(role: str) -> str:
    return f"educator_state_initial_certification_{role.lower()}"


def _year_label(year: int) -> str:
    return f"{year - 1}-{year % 100:02d}"


def _selected_rows(rollups: Iterable[tuple],
                   witnesses: dict[tuple[str, str], tuple[int, tuple]]) -> dict[tuple[str, str], tuple[int, tuple]]:
    rollup_iter = iter(rollups)
    if next(rollup_iter, None) != ROLLUP_HEADERS:
        raise ContractError("Certification worksheet headers differ from the reviewed workbook")
    direct = {}
    for physical_row, row in enumerate(rollup_iter, 2):
        if len(row) != len(ROLLUP_HEADERS):
            raise ContractError("Certification rollup width differs from the reviewed workbook")
        year, route, role, gender, ethnicity, count = row
        key = role, year
        if (key not in witnesses or key in direct or (route, gender, ethnicity) != ("All", "All", "All")
                or type(count) is not int or count < 0 or witnesses[key] != (physical_row, row)):
            raise ContractError(f"Certification rollup differs from the independent XML witness for {key}")
        direct[key] = physical_row, row
    if direct.keys() != witnesses.keys():
        raise ContractError("Certification direct row roster differs from the independent XML witness")
    return direct


def _detail_diagnostics(details: Iterable[tuple], direct: dict[tuple[str, str], tuple[int, tuple]]) -> list[dict[str, Any]]:
    """Describe actual unqualified detail; never use it to produce a total."""
    iterator = iter(details)
    if next(iterator, None) != DETAIL_HEADERS:
        raise ContractError("Certification detail headers differ from the reviewed workbook")
    grouped: dict[tuple[str, str], list[tuple]] = {key: [] for key in direct}
    for row in iterator:
        if (len(row) != 6 or any(not isinstance(v, str) or not v for v in row[:5])
                or type(row[-1]) is not int or row[-1] < 0 or (row[2], row[0]) not in grouped):
            raise ContractError("Certification detail diagnostic input changed")
        grouped[row[2], row[0]].append(row)
    result = []
    for (role, year), (_, row) in direct.items():
        values = grouped[role, year]
        keys = Counter(value[:5] for value in values)
        duplicates, total = sum(n - 1 for n in keys.values()), sum(value[-1] for value in values)
        outcomes = ([] if values else ["missing_detail"]) + (["duplicate_keys"] if duplicates else [])
        if total != row[-1]:
            outcomes.append("sum_differs")
        result.append({"role": role, "year": year, "rows": len(values), "unique_keys": len(keys),
                       "duplicate_rows": duplicates, "sum": total, "direct_total": row[-1],
                       "outcomes": outcomes or ["matches"], "detail_qualified": False})
    return result


def _proof(store: ProvenanceStore, receipt: str) -> dict[str, Any]:
    proof = _source_witness(store, receipt)["certification"]
    if (receipt != SOURCE_WITNESS_RECEIPT
            or proof["workbook_receipt"] != WORKBOOK_RECEIPT
            or proof["workbook_sha256"] != WORKBOOK_SHA
            or set(proof["reports"]) != set(ADMITTED)):
        raise ContractError("Certification source witness scope changed")
    return proof


def materialize_educator_certification(store: ProvenanceStore, output: Path, *,
                                       archive_receipt: str) -> dict[str, Any]:
    """Build an exclusive local candidate from the exact pinned publisher sources."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("Certification candidate requires a new exclusive output directory")
    proof = _proof(store, archive_receipt)
    record = store.read_record(WORKBOOK_RECEIPT)
    if (record["kind"] != "acquisition" or record["artifact"]["sha256"] != WORKBOOK_SHA
            or record.get("requested_url") != WORKBOOK_URL or record.get("resolved_url") != WORKBOOK_URL
            or record.get("source_metadata", {}).get("version") != EDITION):
        raise ContractError("Certification workbook acquisition changed")

    config_bytes = CONFIG.read_bytes()
    config = load_config_bytes(config_bytes, path=CONFIG)
    sources = {}
    witnesses = {(item["values"][2], item["values"][0]): (item["row"], tuple(item["values"]))
                 for item in proof["rows"]}
    if len(witnesses) != 33 or len(proof["rows"]) != 33:
        raise ContractError("Certification requires all 33 independent direct-row witnesses")
    table_dictionaries = []
    roots = {WORKBOOK_RECEIPT, archive_receipt}
    for role, years in ADMITTED.items():
        dataset = _dataset(role)
        report = proof["reports"][role]
        source = config.resolve_dataset(dataset, REPORT_YEAR, reader_profile="source", exact=True)
        options = reader_options(source.reader, "source")
        resource = config.column_contract(dataset, REPORT_YEAR, reader_profile="source")
        if (not config.is_source_only(dataset, REPORT_YEAR) or source.metadata.version != EDITION
                or source.metadata.period != DataPeriod() or source.metadata.source_url != WORKBOOK_URL
                or source.metadata.metrics != METRICS or options.get("acquisition_id") != WORKBOOK_RECEIPT
                or options.get("worksheet") != ROLLUP_SHEET or resource is None):
            raise ContractError(f"Certification source configuration differs for {role}")
        archive = next((item for item in source.metadata.references if item.reference_id == "certification_role_archive"), None)
        if archive is None or (archive.role, archive.url, archive.reviewed_sha256, archive.provenance_ids) != (
            "methodology", report["archive_url"], report["artifact"]["sha256"], (report["receipt"],)
        ):
            raise ContractError(f"Certification archive reference differs for {role}")
        archive_record = store.read_record(report["receipt"])
        if (archive_record["kind"] != "acquisition" or archive_record["artifact"] != report["artifact"]
                or archive_record.get("requested_url") != report["archive_url"]):
            raise ContractError(f"Certification PDF acquisition differs for {role}")
        validate_reference_records(source, store.read_record, verify_all_claimed=True)
        roots.update(identifier for ref in source.metadata.references for identifier in ref.provenance_ids)
        tables = crosswalk_dataset(parse_column_schema(resource["text"].encode()), dataset, EDITION)["tables"]
        by_table = {mapping["table"]: mapping for mapping in tables}
        if len(tables) != len(years) or by_table.keys() != {f"state_{year}" for year in years}:
            raise ContractError(f"Certification crosswalk scope differs for {role}")
        for year in years:
            mapping = by_table[f"state_{year}"]
            table_dictionaries.append(mapping["dictionary_records"])
            fields = tuple((field["code"], field["label"], field["field"], field["disposition"]) for field in mapping["fields"])
            expected_fields = tuple((header, header, field, "fact" if field == COUNT else "identity")
                                    for header, field in zip(ROLLUP_HEADERS, FIELDS, strict=True))
            if (mapping["input_sha256"] != WORKBOOK_SHA or fields != expected_fields
                    or mapping["metric_period"] != DataPeriod("school_year", year - 1, year).to_dict()):
                raise ContractError(f"Certification crosswalk fields or period differ for {role} {year}")
            label = _year_label(year)
            matches = [item for item in report["comparisons"] if item["year"] == label]
            if len(matches) != 1:
                raise ContractError(f"Certification archive witness is missing or repeated for {role} {year}")
            witness = matches[0]
            archive_count, workbook_count = witness["archive_total"], witness["workbook_total"]
            expected = PUBLISHER_REVISIONS.get((role, label), (workbook_count, workbook_count))
            outcome = witness["outcome"]
            if (witnesses.get((role, label)) != (witness["workbook_row"], (label, "All", role, "All", "All", workbook_count))
                    or (2019 <= year <= 2024 and ((archive_count, workbook_count) != expected
                        or outcome != ("publisher_edition_difference" if (role, label) in PUBLISHER_REVISIONS else "matches")
                        or len(witness["archive_tables"]) != 2
                        or any(item["total"] != archive_count for item in witness["archive_tables"])))
                    or (not 2019 <= year <= 2024 and (archive_count is not None or outcome != "unavailable"
                                                     or witness["archive_tables"]))):
                raise ContractError(f"Certification archive count does not reconcile for {role} {year}")
        sources[role] = source
    store.verify_record_closure(sorted(roots))
    with store.verify_artifact(WORKBOOK_SHA).open("rb") as file, closing(
        load_workbook(file, read_only=True, data_only=False, keep_links=False)
    ) as book:
        rows = iter(book["Data Dictionary"].values)
        if next(rows, None) != ("Field Name", "Description", "Coding"):
            raise ContractError("Certification workbook dictionary headers differ")
        definitions = [
            {"ordinal": index, "code": code, "publisher_type": coding, "publisher_description": description,
             "source_sha256": WORKBOOK_SHA, "source_line": index + 1, "source_column": 0}
            for index, (code, description, coding) in enumerate(rows, 1)
        ]
        if (len(definitions) != len(ROLLUP_HEADERS)
                or {item["code"] for item in definitions} != set(ROLLUP_HEADERS)
                or any(dictionary != definitions for dictionary in table_dictionaries)):
            raise ContractError("Certification field definitions differ from the exact workbook dictionary")
        records = _selected_rows(book[ROLLUP_SHEET].values, witnesses)
        diagnostics = _detail_diagnostics(book["Data"].values, records)
        if diagnostics != proof["detail_diagnostics"]:
            raise ContractError("Certification detail diagnostics differ from the independent XML observation")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".educator-certification-", dir=output.parent) as directory:
        stage = Path(directory) / "bundle"
        stage.mkdir(mode=0o700)
        target = ProvenanceStore(stage / "provenance")
        for identifier in sorted(roots):
            target.copy_record_from(store, identifier)
        config_path = Path(directory) / "teadata_sources.yaml"
        config_path.write_bytes(config_bytes)
        config_id = target.record_acquisition(config_path, ResolvedDataset("reference:educator_certification:config",
            "reference", REPORT_YEAR, REPORT_YEAR, "", DatasetMetadata(title="Exact educator certification source configuration",
                                                                        version=EDITION)))
        recorder = BuildRecorder(config, target)
        toolchain = {"python": sys.version, "pyarrow": version("pyarrow"), "openpyxl": version("openpyxl"),
                     "platform": platform.platform(), "architecture": platform.machine()}
        editions = []
        for role, years in ADMITTED.items():
            source = sources[role]
            dataset = _dataset(role)
            for year in years:
                label = _year_label(year)
                number, row = records[role, label]
                revision = PUBLISHER_REVISIONS.get((role, label))
                comparison_row = next(item for item in proof["reports"][role]["comparisons"] if item["year"] == label)
                detail = next(item for item in diagnostics if (item["role"], item["year"]) == (role, label))
                comparison: dict[str, Any] = {
                    "archive_comparison": comparison_row["outcome"],
                    "archive_total_matches": None if comparison_row["outcome"] == "unavailable" else revision is None,
                    "detail_diagnostic": detail,
                    "detail_qualified": False,
                    "independent_xml_witness_matches": True,
                }
                if revision is not None:
                    comparison["publisher_edition_difference"] = {
                        "november_2024_archive_total": revision[0],
                        "september_2026_workbook_total": revision[1],
                    }
                values = (*row, str(row[-1]), ValueStatus.OBSERVED.value, number)
                table = pa.Table.from_arrays(
                    [pa.array([value], type=field.type) for value, field in zip(values, _SCHEMA, strict=True)], schema=_SCHEMA)
                key = EditionKey(dataset, EDITION, EditionKey.source_revision((WORKBOOK_SHA,)), "state", f"state_{year}",
                                 STATEWIDE_SINGLETON_GRAIN)
                recipe = {"operation": "educator_certification", "worksheet": ROLLUP_SHEET,
                          "publisher_role": role, "school_year_end": year}
                contract = recorder.leaf_contract(key, source, native_schema=table.schema.serialize().to_pybytes(),
                    columns=table.column_names, key_columns=[], rows=1, inputs=[WORKBOOK_SHA], recipe=recipe,
                    toolchain=toolchain, column_support=_SUPPORT)
                contract_json = canonical_json(contract)
                leaf = stage / f"{sha256(contract_json.encode()).hexdigest()}.parquet"
                physical = write_leaf_artifact(leaf, table.to_batches(), contract=contract, batch_bytes=8 * 1024**2)
                if not table.equals(pq.read_table(leaf).replace_schema_metadata(None), check_metadata=True):
                    raise ContractError("Certification native readback differs from its publisher row")
                local_refs = {identifier for ref in source.metadata.references for identifier in ref.provenance_ids}
                processing = target.record_processing(dataset=dataset, inputs=[*sorted(local_refs | {WORKBOOK_RECEIPT, archive_receipt}), config_id],
                    outputs={"leaf": leaf}, processor="teadata.educator_certification", code_version=recorder.processor_sha256,
                    parameters={"leaf_contract_sha256": leaf.stem, "recipe": recipe, "toolchain": toolchain},
                    validation={**comparison, "publisher_worksheet_row": number, "exact_native_projection": True,
                                "rows": 1, "release_activation": False})
                artifact = ArtifactRef(key, physical["sha256"], physical["size_bytes"], leaf.name,
                                       contract_json, processing, "leaf")
                recorder.record_edition(key, source, artifacts=[artifact], acquisition_ids=[WORKBOOK_RECEIPT])
                editions.append({"edition": key.to_dict(), "role": role, "school_year_end": year,
                                 "publisher_worksheet_row": number, "processing_record_id": processing})
        catalog = recorder.catalog(schema_version=4)
        manifest = publish_release(stage, catalog, schema_version=4)
        result = {"status": "unpublished_native_educator_certification", "build_id": catalog.build_id,
                  "manifest_sha256": file_identity(manifest)["sha256"], "editions": editions,
                  "archive_readback_receipt": archive_receipt, "release_activation": False}
        (stage / "MATERIALIZATION.json").write_text(canonical_json(result) + "\n", encoding="utf-8")
        _install_candidate(stage, output)
    return result
