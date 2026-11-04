"""Replay reviewed budget-account leaves and separate TEA PWR controls."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from importlib.resources import files
from io import TextIOWrapper
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zipfile import BadZipFile, ZipFile

import pyarrow as pa
import pyarrow.compute as arrow_compute
import pyarrow.csv as csv
import pyarrow.parquet as pq

from .dataset_metadata import DataPeriod
from .manifest import ContractError
from .peims_actuals import BUDGET_DATASET, _MAX_ARCHIVE_BYTES, _MAX_MEMBER_BYTES

if TYPE_CHECKING:
    from .dataset_metadata import ArtifactRef, ReferenceDocument
    from .provenance import ProvenanceStore
    from .release import DataRelease


pc: Any = arrow_compute
_AMOUNT = "budget_amount"
_RAW_AMOUNT = "raw_budget_amount"
_AMOUNT_STATUS = "budget_amount_status"
_DISTRICT = "district_number"
_PERIOD = DataPeriod("fiscal_year", 2025, 2026)
_COVERAGE = {("charter", "AF"): 182, ("charter", "GF"): 182, ("district", "AF"): 1019, ("district", "GF"): 1019}


def is_reviewed_budget(manifest_sha256: str) -> bool:
    review = json.loads(files("teadata").joinpath("reference_cases/native_budgets.json").read_text())
    return manifest_sha256 in review["manifest_sha256s"]


def _source_hash(path: Path, member: str, headings: list[str]) -> str:
    digest = sha256()
    with ZipFile(path) as archive:
        entry, = archive.infolist()
        if entry.filename != member or not 0 < entry.file_size <= _MAX_MEMBER_BYTES:
            raise ContractError("Reviewed budget archive member differs")
        with TextIOWrapper(archive.open(entry), encoding="ascii", newline=None) as source:
            header = ",".join(headings) + "\n"
            if source.readline(len(header) + 1) != header:
                raise ContractError("Reviewed budget publisher header differs")
            while block := source.read(2 * 1024**2):
                digest.update(block.encode("ascii"))
    return digest.hexdigest()


def _accumulate_budget_control(table: pa.Table, population: str, totals: Counter[tuple[str, str]]) -> None:
    if population == "district":
        fund, function, obj = ("fund_code", "function_code", "object_code")
        general_fund = ["100", "101", "199"]
    else:
        fund, function, obj = ("nonprofit_asset_code", "nonprofit_function_code", "nonprofit_object_code")
        general_fund = ["100", "101", "199", "420"]
    selected = table.filter(pc.and_(pc.is_in(table[function], value_set=pa.array(["11", "95"])),
                                   pc.and_(pc.greater_equal(table[obj], "6100"), pc.less_equal(table[obj], "6499"))))
    for group, rows in (
        ("AF", selected), ("GF", selected.filter(pc.is_in(selected[fund], value_set=pa.array(general_fund)))),
    ):
        sums = rows.select([_DISTRICT, _AMOUNT]).group_by(_DISTRICT, use_threads=False).aggregate([(_AMOUNT, "sum")])
        for row in sums.to_pylist():
            totals[(row[_DISTRICT], group)] += row[f"{_AMOUNT}_sum"]


def _replay_budget(release: DataRelease, store: ProvenanceStore, ref: ArtifactRef,
                   totals: Counter[tuple[str, str]]) -> tuple[dict[str, Any], set[str]]:
    key = ref.edition
    acquisition, = release.catalog.acquisitions(BUDGET_DATASET, edition=key)
    raw_sha, = ref.contract["inputs"]
    descriptor = acquisition["artifact"]
    if descriptor["sha256"] != raw_sha or not 0 < descriptor["size_bytes"] <= _MAX_ARCHIVE_BYTES:
        raise ContractError("Reviewed budget original differs from its leaf binding")
    graph = store.verify_record_closure([ref.processing_id])
    processing = graph[ref.processing_id]
    parameters = processing["parameters"]
    recipe = parameters["recipe"]
    identifier = acquisition["record_id"]
    if (processing["processor"] != "teadata.peims_actuals"
            or recipe["operation"] != "peims_single_file_budgets_projection"
            or parameters["leaf_contract_sha256"] != ref.identity or identifier not in processing["inputs"]
            or graph[identifier] != {name: value for name, value in acquisition.items() if name != "record_id"}):
        raise ContractError("Reviewed budget processing receipt differs from the candidate")
    fields = release.catalog.header_records(key.dataset, edition_key=key)
    headings = [item["code"] for item in fields]
    raw_fields = [_RAW_AMOUNT if item["field"] == _AMOUNT else item["field"] for item in fields]
    source = _source_hash(store.verify_artifact(raw_sha), recipe["archive_member"], headings)

    replay = sha256()
    csv_options = csv.WriteOptions(include_header=False, quoting_style="none")
    entities: set[str] = set()
    rows = zeros = negatives = 0
    with pq.ParquetFile(release.artifact(ref.identity)) as parquet:
        for batch in parquet.iter_batches(batch_size=32_768, use_threads=False):
            table = pa.Table.from_batches([batch])
            amount = table[_AMOUNT]
            validity = pc.and_kleene(pc.equal(pc.cast(table[_RAW_AMOUNT], pa.int64()), amount),
                                    pc.equal(table[_AMOUNT_STATUS], "observed"))
            locations = pc.equal(table["source_record"], pa.array(range(rows + 2, rows + len(batch) + 2)))
            if not pc.all(pc.fill_null(pc.and_kleene(validity, locations), False)).as_py():
                raise ContractError("Reviewed budget values, statuses or physical row locations differ")
            output = pa.BufferOutputStream()
            csv.write_csv(table.select(raw_fields), output, csv_options)
            replay.update(output.getvalue())
            _accumulate_budget_control(table, key.table, totals)
            entities.update(pc.unique(table[_DISTRICT]).to_pylist())
            rows += len(batch)
            zeros += pc.sum(pc.equal(amount, 0)).as_py()
            negatives += pc.sum(pc.less(amount, 0)).as_py()
    if rows != ref.contract["rows"] or replay.hexdigest() != source:
        raise ContractError("Reviewed budget raw values or row count differ")
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
            result, entities = _replay_budget(release, parity_store, ref, totals)
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
