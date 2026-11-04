"""Replay reviewed financial account leaves and separate TEA PWR controls."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from hashlib import sha256
from importlib.resources import files
from io import TextIOWrapper
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any
from zipfile import BadZipFile, ZipFile

import pyarrow as pa
import pyarrow.compute as arrow_compute
import pyarrow.csv as csv
import pyarrow.parquet as pq

from .column_schema import reader_options, table_input_sha256s
from .dataset_metadata import DataPeriod
from .manifest import ContractError
from .peims_actuals import (
    BUDGET_DATASET, DATASET, _ACTUALS, _BUDGETS, _MAX_ARCHIVE_BYTES, _MAX_MEMBER_BYTES, _FinancialBasis, _tables,
)
from .teadata_config import load_config

if TYPE_CHECKING:
    from .dataset_metadata import ArtifactRef, ReferenceDocument
    from .provenance import ProvenanceStore
    from .release import DataRelease
    from .teadata_config import Config


pc: Any = arrow_compute
_DISTRICT = "district_number"
_PERIOD = DataPeriod("fiscal_year", 2025, 2026)
_COVERAGE = {("charter", "AF"): 182, ("charter", "GF"): 182, ("district", "AF"): 1019, ("district", "GF"): 1019}
PASS_ACTUAL_PWR_CONTROLS = "PASS_REVIEWED_ACTUAL_PWR_CONTROLS"
_ACTUAL_PWR = {
    2023: ("2beb638ee1da5351a69e715b8dc6ac30dedfe8284c44ecdead5895efd8cd4bfa", 1014),
    2024: ("f62a3201b892c354edcac791126075b1e8be4b27ec5e060d56aedb38e4bc14fe", 1013),
    2025: ("f8a3128a833512452dfaef9af8030f07854f7f02fe8c2fa69d7f157c5d109b23", 1013),
}
_ACTUAL_PROGRAMS = ["23", "33", "43"]
_DYSLEXIA_PIC = "37"
_ACTUAL_EXCLUDED_FUNCTIONS = ["71", "81", "91", "93", "97", "99"]
_CAPITAL_PROJECTS_FUND = "699"
_GENERAL_FUNDS = ["100", "101", "199"]
_OBJECT_MIN, _OBJECT_MAX = "6100", "6499"


def is_reviewed_budget(manifest_sha256: str) -> bool:
    review = json.loads(files("teadata").joinpath("reference_cases/native_budgets.json").read_text())
    return manifest_sha256 in review["manifest_sha256s"]


def _source_hash(path: Path, member: str, headings: list[str]) -> str:
    digest = sha256()
    with ZipFile(path) as archive:
        entry, = archive.infolist()
        if entry.filename != member or not 0 < entry.file_size <= _MAX_MEMBER_BYTES:
            raise ContractError("Reviewed financial archive member differs")
        with TextIOWrapper(archive.open(entry), encoding="ascii", newline=None) as source:
            header = ",".join(headings) + "\n"
            if source.readline(len(header) + 1) != header:
                raise ContractError("Reviewed financial publisher header differs")
            while block := source.read(2 * 1024**2):
                digest.update(block.encode("ascii"))
    return digest.hexdigest()


def _accumulate_budget_control(table: pa.Table, population: str, totals: Counter[tuple[str, str]]) -> None:
    if population == "district":
        fund, function, obj = ("fund_code", "function_code", "object_code")
        general_fund = _GENERAL_FUNDS
    else:
        fund, function, obj = ("nonprofit_asset_code", "nonprofit_function_code", "nonprofit_object_code")
        general_fund = [*_GENERAL_FUNDS, "420"]
    selected = table.filter(pc.and_(pc.is_in(table[function], value_set=pa.array(["11", "95"])),
                                   _operating_objects(table[obj])))
    for group, rows in (
        ("AF", selected), ("GF", selected.filter(pc.is_in(selected[fund], value_set=pa.array(general_fund)))),
    ):
        _accumulate(rows, _BUDGETS.amount, group, totals)


def _operating_objects(column: pa.ChunkedArray) -> pa.ChunkedArray:
    return pc.and_(pc.greater_equal(column, _OBJECT_MIN), pc.less_equal(column, _OBJECT_MAX))


def _accumulate(table: pa.Table, amount: str, group: str, totals: Counter[tuple[str, str]]) -> None:
    sums = table.select([_DISTRICT, amount]).group_by(_DISTRICT, use_threads=False).aggregate([(amount, "sum")])
    for row in sums.to_pylist():
        totals[row[_DISTRICT], group] += row[f"{amount}_sum"]


def _replay_accounts(release: DataRelease, store: ProvenanceStore, ref: ArtifactRef, basis: _FinancialBasis,
                     collect: Callable[[pa.Table], None]) -> tuple[dict[str, Any], set[str]]:
    key = ref.edition
    acquisition, = release.catalog.acquisitions(basis.dataset, edition=key)
    raw_sha, = ref.contract["inputs"]
    descriptor = acquisition["artifact"]
    if descriptor["sha256"] != raw_sha or not 0 < descriptor["size_bytes"] <= _MAX_ARCHIVE_BYTES:
        raise ContractError("Reviewed financial original differs from its leaf binding")
    graph = store.verify_record_closure([ref.processing_id])
    processing = graph[ref.processing_id]
    parameters = processing["parameters"]
    recipe = parameters["recipe"]
    identifier = acquisition["record_id"]
    if (processing["processor"] != "teadata.peims_actuals"
            or recipe["operation"] != f"peims_single_file_{basis.name}s_projection"
            or parameters["leaf_contract_sha256"] != ref.identity or identifier not in processing["inputs"]
            or graph[identifier] != {name: value for name, value in acquisition.items() if name != "record_id"}):
        raise ContractError("Reviewed financial processing receipt differs from the candidate")
    fields = release.catalog.header_records(key.dataset, edition_key=key)
    headings = [item["code"] for item in fields]
    raw_fields = [basis.raw if item["field"] == basis.amount else item["field"] for item in fields]
    source = _source_hash(store.verify_artifact(raw_sha), recipe["archive_member"], headings)

    replay = sha256()
    csv_options = csv.WriteOptions(include_header=False, quoting_style="none")
    entities: set[str] = set()
    rows = zeros = negatives = 0
    with pq.ParquetFile(release.artifact(ref.identity)) as parquet:
        for batch in parquet.iter_batches(batch_size=32_768, use_threads=False):
            table = pa.Table.from_batches([batch])
            amount = table[basis.amount]
            validity = pc.and_kleene(pc.equal(pc.cast(table[basis.raw], pa.int64()), amount),
                                    pc.equal(table[basis.status], "observed"))
            locations = pc.equal(table["source_record"], pa.array(range(rows + 2, rows + len(batch) + 2)))
            if not pc.all(pc.fill_null(pc.and_kleene(validity, locations), False)).as_py():
                raise ContractError("Reviewed financial values, statuses or physical row locations differ")
            output = pa.BufferOutputStream()
            csv.write_csv(table.select(raw_fields), output, csv_options)
            replay.update(output.getvalue())
            collect(table)
            entities.update(pc.unique(table[_DISTRICT]).to_pylist())
            rows += len(batch)
            zeros += pc.sum(pc.equal(amount, 0)).as_py()
            negatives += pc.sum(pc.less(amount, 0)).as_py()
    if rows != ref.contract["rows"] or replay.hexdigest() != source:
        raise ContractError("Reviewed financial raw values or row count differ")
    return ({"edition_id": key.identity, "edition": key.to_dict(), "status": "PASS_RAW_PARITY",
             "rows": rows, "raw_cells": rows * len(fields), "compared_cells": rows * (len(fields) + 3),
             "zero_amounts": zeros, "negative_amounts": negatives, "source_data_lf_sha256": source,
             "source_receipt": identifier}, entities)


def _workbook_control(store: ProvenanceStore, reference: ReferenceDocument, scopes: dict[str, str],
                      totals: Counter[tuple[str, str]]) -> dict[str, Any]:
    from openpyxl import load_workbook

    identifier, = reference.provenance_ids
    record = store.read_record(identifier)
    artifact = record["artifact"]
    if (record["kind"] != "acquisition" or record["method"] != "download" or record["section"] != "reference"
            or record["dataset"] != "reference:peims_financial_budgets:budget_pwr"
            or record["requested_url"] != reference.url or record["source_metadata"]["period"] != _PERIOD.to_dict()
            or record["parameters"].get("http_method") != "GET" or record["parameters"].get("http_status") != 200
            or artifact["sha256"] != reference.reviewed_sha256 or not 0 < artifact["size_bytes"] <= 16 * 1024**2):
        raise ContractError("Independent Budget PWR receipt differs from the reviewed publisher report")
    path = store.verify_artifact(artifact["sha256"])
    coverage: Counter[tuple[str, str]] = Counter()
    with path.open("rb") as stream:
        workbook = load_workbook(stream, read_only=True, keep_links=False)
        try:
            for group in ("GF", "AF"):
                records = iter(workbook[f"25_26 {group} Bud"].rows)
                headings = [cell.value for cell in next(records)]
                title = f"{group} Instruction (Function 11,95)"
                if headings[:3].count("DISTRICT NUMBER") != 1 or headings[2] != "DISTRICT NUMBER" or headings.count(title) != 1:
                    raise ContractError("Independent Budget PWR headings differ")
                column = headings.index(title)
                seen: set[str] = set()
                for row in records:
                    entity, amount = row[2].value, row[column]
                    if entity in seen or entity not in scopes or row[0].value != "2026 Budget PWR":
                        raise ContractError("Independent Budget PWR population or edition differs")
                    if amount.data_type != "n" or type(amount.value) is not int or amount.value != totals[entity, group]:
                        raise ContractError(f"Independent Budget PWR amount differs for {entity} {group}")
                    seen.add(entity)
                    coverage[scopes[entity], group] += 1
                if seen != scopes.keys():
                    raise ContractError("Independent Budget PWR does not cover every reviewed account entity")
        finally:
            workbook.close()
    if dict(coverage) != _COVERAGE:
        raise ContractError("Independent Budget PWR does not cover the reviewed district and charter populations")
    return {"receipt_id": identifier, "source_url": reference.url, "functions": ["11", "95"],
            "objects": "6100–6499", "compared_cells": sum(coverage.values()),
            "coverage": {f"{scope}:{group}": count for (scope, group), count in sorted(coverage.items())}}


def qualify_budget(release: DataRelease, parity_store: ProvenanceStore, witness_store: ProvenanceStore) -> dict[str, Any]:
    try:
        catalog = release.catalog.to_dict()
        refs = [release.catalog.artifact_ref(identifier) for identifier in catalog["artifacts"]]
        if (catalog["schema_version"] != 4 or len(refs) != 2
                or {ref.edition.identity for ref in refs} != catalog["editions"].keys()
                or {ref.edition.table for ref in refs} != {"district", "charter"}
                or any((ref.edition.dataset, ref.edition.edition, ref.edition.scope) != (BUDGET_DATASET, "2026", "district")
                       for ref in refs)):
            raise ContractError("Budget candidate does not contain the exact reviewed editions and leaves")
        release.verify()
        totals: Counter[tuple[str, str]] = Counter()
        scopes: dict[str, str] = {}
        results = []
        witnesses = []
        for ref in sorted(refs, key=lambda item: item.edition.table):
            key = ref.edition
            metadata = release.catalog.metadata(BUDGET_DATASET, edition=key)
            qualifiers = metadata.source_qualifiers
            if metadata.period != _PERIOD or qualifiers is None or qualifiers.financial_basis != "budget":
                raise ContractError("Reviewed budget fiscal period or basis differs")
            witness, = (reference for reference in metadata.references if reference.reference_id == "budget_pwr")
            witnesses.append(witness)
            result, entities = _replay_accounts(release, parity_store, ref, _BUDGETS,
                                                lambda table: _accumulate_budget_control(table, key.table, totals))
            if scopes.keys() & entities:
                raise ContractError("Reviewed district and charter budget populations overlap")
            scopes.update(dict.fromkeys(entities, key.table))
            results.append(result)
        if witnesses[0] != witnesses[1] or witnesses[0].reviewed_sha256 in {digest for ref in refs for digest in ref.contract["inputs"]}:
            raise ContractError("Budget PWR does not independently cover both reviewed populations")
        publisher = _workbook_control(witness_store, witnesses[0], scopes, totals)
    except (BadZipFile, OSError, StopIteration, ValueError, KeyError, TypeError, IndexError) as error:
        raise ContractError(f"Reviewed budget evidence replay failed: {error}") from error
    return {"status": "PASS_REVIEWED_NATIVE", "build_id": release.build_id, "manifest_sha256": release.manifest_sha256,
            "structural_integrity": True, "qualification_scope": "budget account fidelity and PWR instruction controls",
            "parity": {"status": "PASS_RAW_PARITY", "editions": results,
                       "coverage": {"checked_editions": len(results),
                                    "raw_cells": sum(item["raw_cells"] for item in results),
                                    "compared_cells": sum(item["compared_cells"] for item in results)}},
            "publisher_witnesses": publisher}


def _accumulate_actual_control(table: pa.Table, totals: Counter[tuple[str, str]],
                               dyslexia_totals: Counter[tuple[str, str]]) -> None:
    operating = pc.and_(
        _operating_objects(table["object_code"]),
        pc.invert(pc.is_in(table["function_code"], value_set=pa.array(_ACTUAL_EXCLUDED_FUNCTIONS))),
    )
    for codes, measure_totals in ((_ACTUAL_PROGRAMS, totals), ([_DYSLEXIA_PIC], dyslexia_totals)):
        selected = table.filter(pc.and_(operating, pc.is_in(table["program_intent_code"], value_set=pa.array(codes))))
        _accumulate(selected.filter(pc.is_in(selected["fund_code"], value_set=pa.array(_GENERAL_FUNDS))),
                    _ACTUALS.amount, "GF", measure_totals)
        # The reviewed All Funds presentations exclude this fund. Preserve its
        # signed contribution independently; the raw account leaf is never changed.
        capital_projects = pc.equal(selected["fund_code"], _CAPITAL_PROJECTS_FUND)
        _accumulate(selected.filter(pc.invert(capital_projects)), _ACTUALS.amount, "AF", measure_totals)
        _accumulate(selected.filter(capital_projects), _ACTUALS.amount, "excluded_fund_699", measure_totals)


def _actual_workbook_control(store: ProvenanceStore, year: int, entities: set[str],
                             totals: Counter[tuple[str, str]], dyslexia_totals: Counter[tuple[str, str]]) -> dict[str, Any]:
    from openpyxl import load_workbook

    identifier, independent_count = _ACTUAL_PWR[year]
    record = store.read_record(identifier)
    artifact = record["artifact"]
    url = record["requested_url"]
    metadata = record["source_metadata"]
    if (record["kind"] != "acquisition" or record["method"] != "download" or record["section"] != "reference"
            or record["resolved_url"] != url or metadata.get("download_url", metadata.get("source_url")) != url
            or record["parameters"].get("http_method") != "GET" or record["parameters"].get("http_status") != 200
            or record["response_headers"].get("content-type", "").split(";", 1)[0]
            != "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            or not 0 < artifact["size_bytes"] <= 16 * 1024**2):
        raise ContractError("Actual PWR receipt differs from the reviewed publisher acquisition")
    path = store.verify_artifact(artifact["sha256"])
    if path.stat().st_size != artifact["size_bytes"]:
        raise ContractError("Actual PWR workbook size differs from its receipt")
    fund_groups = {}
    dyslexia_fund_groups = {}
    rosters: dict[str, dict[str, str]] = {}
    with path.open("rb") as stream:
        workbook = load_workbook(stream, read_only=True, keep_links=False)
        try:
            for group in ("GF", "AF"):
                tab_name = f"{(year - 1) % 100:02}_{year % 100:02} {group} Act"
                sheet = "2024 AF Act" if year == 2024 and group == "AF" else tab_name
                records = iter(workbook[sheet].rows)
                headings = [cell.value for cell in next(records)]
                title = f"{group} Students with Disabilities (PICs {','.join(_ACTUAL_PROGRAMS)})"
                dyslexia_title = f"{group} Dyslexia or Related Disorder Services (PIC {_DYSLEXIA_PIC})"
                type_heading = "DISTRICT TYPE" if year == 2023 else "District Type"
                if (headings[1:5] != ["TAB NAME", "DISTRICT NUMBER", "DISTRICT NAME", type_heading]
                        or any(headings.count(name) != 1 for name in ("DISTRICT NUMBER", title, dyslexia_title))):
                    raise ContractError("Actual PWR headings differ")
                controls = ((headings.index(title), totals), (headings.index(dyslexia_title), dyslexia_totals))
                roster: dict[str, str] = {}
                types: Counter[str] = Counter()
                for row in records:
                    entity, population = row[2].value, row[4].value
                    if (row[0].value != f"{year} Actual PWR" or row[1].value != tab_name
                            or not isinstance(entity, str) or re.fullmatch(r"[0-9]{6}", entity) is None
                            or entity == "000000" or entity in roster
                            or population not in {"INDEPENDENT", "COMMON", "CHARTER"}):
                        raise ContractError("Actual PWR population or edition differs")
                    roster[entity] = population
                    types[population] += 1
                    if population == "INDEPENDENT":
                        for column, measure_totals in controls:
                            amount = row[column]
                            if (amount.data_type != "n" or type(amount.value) is not int
                                    or amount.value != measure_totals[entity, group]):
                                raise ContractError(f"Actual PWR amount differs for FY{year} {entity}: {headings[column]}")
                if (types["INDEPENDENT"] != independent_count
                        or {entity for entity, population in roster.items() if population != "CHARTER"} != entities):
                    raise ContractError("Actual PWR district account population differs from the publisher roster")
                rosters[group] = roster
                fund_groups[group] = {"sheet": sheet, "publisher_tab_name": tab_name, "column": title,
                                      "compared_cells": types["INDEPENDENT"],
                                      "common_unchecked": types["COMMON"], "charter_unchecked": types["CHARTER"]}
                dyslexia_fund_groups[group] = {"column": dyslexia_title, "compared_cells": types["INDEPENDENT"]}
        finally:
            workbook.close()
    if rosters["GF"] != rosters["AF"]:
        raise ContractError("Actual PWR General Fund and All Funds rosters differ")
    independent = {entity for entity, population in rosters["AF"].items() if population == "INDEPENDENT"}
    return {"fiscal_end_year": year, "receipt_id": identifier, "source_url": url, "source_sha256": artifact["sha256"],
            "retrieved_at": record["downloaded_at"],
            "fund_groups": fund_groups, "compared_cells": 2 * independent_count,
            "independent_districts_with_nonzero_excluded_fund_699": sum(
                bool(totals[entity, "excluded_fund_699"]) for entity in independent),
            "dyslexia_pic37": {"fund_groups": dyslexia_fund_groups, "compared_cells": 2 * independent_count,
                               "independent_districts_with_nonzero_excluded_fund_699": sum(
                                   bool(dyslexia_totals[entity, "excluded_fund_699"]) for entity in independent)}}


def _check_actual_binding(config: Config, release: DataRelease, ref: ArtifactRef, year: int) -> None:
    key = ref.edition
    table = _tables(config, year, _ACTUALS)[key.table]
    source = config.resolve_dataset(DATASET, year, reader_profile="source", exact=True)
    selection = reader_options(source.reader, "source")["archives"][key.table]
    metadata = release.catalog.metadata(DATASET, edition=key)
    qualifiers = metadata.source_qualifiers
    amount = metadata.metrics.get(_ACTUALS.amount)
    acquisition, = release.catalog.acquisitions(DATASET, edition=key)
    headings = release.catalog.header_records(DATASET, edition_key=key)
    field_keys = ("code", "field", "disposition", "position")
    if (metadata.period != DataPeriod("fiscal_year", year - 1, year) or metadata.version != str(year)
            or qualifiers is None or qualifiers.family != "peims_financial" or qualifiers.financial_basis != "actual"
            or amount is None or amount.unit != "currency" or amount.period not in (None, metadata.period)
            or metadata.value_policy is not None or acquisition["record_id"] != selection["receipt_id"]
            or ref.contract["rows"] != selection["rows"]
            or tuple(ref.contract["inputs"]) != table_input_sha256s(table)
            or [tuple(field[name] for name in field_keys) for field in headings]
            != [tuple(field[name] for name in field_keys) for field in table["fields"]]):
        raise ContractError("Actual account period, basis, source binding or crosswalk differs")


def qualify_actual_pwr(release: DataRelease, parity_store: ProvenanceStore,
                       witness_store: ProvenanceStore) -> dict[str, Any]:
    """Replay only the reviewed FY2023–FY2025 Independent ISD operating controls."""
    try:
        catalog = release.catalog.to_dict()
        if catalog["schema_version"] != 4:
            raise ContractError("Actual PWR controls require a schema-4 account bundle")
        refs = [release.catalog.artifact_ref(identifier) for identifier in catalog["artifacts"]]
        config = load_config()
        release.verify()
        results = []
        witnesses = {}
        for year in _ACTUAL_PWR:
            selected = [ref for ref in refs if (ref.edition.dataset, ref.edition.edition, ref.edition.scope, ref.edition.table)
                        == (DATASET, str(year), "district", "district")]
            if len(selected) != 1:
                raise ContractError(f"Actual PWR controls require exactly one FY{year} district account leaf")
            ref = selected[0]
            _check_actual_binding(config, release, ref, year)
            totals: Counter[tuple[str, str]] = Counter()
            dyslexia_totals: Counter[tuple[str, str]] = Counter()
            result, entities = _replay_accounts(release, parity_store, ref, _ACTUALS,
                                                lambda table: _accumulate_actual_control(table, totals, dyslexia_totals))
            results.append(result)
            witness = _actual_workbook_control(witness_store, year, entities, totals, dyslexia_totals)
            if witness["source_sha256"] in ref.contract["inputs"]:
                raise ContractError("Actual PWR workbook is not a separate publisher presentation")
            witnesses[str(year)] = witness
    except (BadZipFile, OSError, StopIteration, ValueError, KeyError, TypeError, IndexError) as error:
        raise ContractError(f"Reviewed Actual PWR control replay failed: {error}") from error
    return {"status": PASS_ACTUAL_PWR_CONTROLS, "build_id": release.build_id, "manifest_sha256": release.manifest_sha256,
            "structural_integrity": True, "qualification_scope": "Independent ISD Actual PWR operating dollar controls",
            "release_activation": False, "signed_audit_reconciliation": False,
            "parity": {"status": "PASS_RAW_PARITY", "editions": results,
                       "coverage": {"checked_editions": len(results),
                                    "raw_cells": sum(item["raw_cells"] for item in results),
                                    "compared_cells": sum(item["compared_cells"] for item in results)}},
            "publisher_witnesses": {"editions": witnesses,
                                    "compared_cells": sum(item["compared_cells"] for item in witnesses.values()),
                                    "dyslexia_pic37": {"program_intent_code": _DYSLEXIA_PIC, "compared_cells": sum(
                                        item["dyslexia_pic37"]["compared_cells"] for item in witnesses.values())},
                                    "program_intent_codes": _ACTUAL_PROGRAMS,
                                    "objects": f"{_OBJECT_MIN}–{_OBJECT_MAX}",
                                    "excluded_functions": _ACTUAL_EXCLUDED_FUNCTIONS,
                                    "general_funds": _GENERAL_FUNDS,
                                    "empirical_all_funds_excluded_fund": _CAPITAL_PROJECTS_FUND}}
