"""Exact scoped source tables from TEA's public CTE workbooks."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
import json
from typing import TYPE_CHECKING, Any
from zipfile import ZipFile

from .financial_leaves import MAX_SOURCE_BYTES, _literal_batch, _package, _sheet_frame
from .identifiers import canonical_source_identifier, canonical_source_identifiers
from .manifest import ContractError

if TYPE_CHECKING:
    from .dataset_metadata import ArtifactRef
    from .provenance import ProvenanceStore
    from .release import DataRelease


MAX_WORKSHEET_ROWS = 60000
SCOPES = ("state", "region", "district")
CTE_WORKBOOK_DATASETS = frozenset({
    "cte_status", "cte_programs_of_study", "cte_ibc_annual", "cte_ibc_graduates", "cte_ibc_first_earned",
})
_ID_WIDTHS = {"district_number": 6, "region_number": 2}
_SOURCE_ROW = "worksheet_row"


@lru_cache(maxsize=1)
def cte_workbook_qualifications() -> dict[str, Any]:
    resource = files("teadata").joinpath("reference_cases/native_cte_workbooks.json")
    return json.loads(resource.read_bytes())


def _source_support(fields) -> dict[str, Any]:
    return {"fields": {field: {"raw": "raw_" + field} for field in _ID_WIDTHS if field in fields},
            "context": [_SOURCE_ROW]}


def cte_workbook_export_contract(release: DataRelease, ref: ArtifactRef, headers: list[dict[str, Any]]) -> dict[str, str]:
    """Allow original headers only for the pinned raw workbook projection."""
    key = ref.edition
    plan = cte_workbook_qualifications().get(release.manifest_sha256)
    if plan is None or (plan["dataset"], plan["edition"]) != (key.dataset, key.edition):
        raise ContractError("Original CTE headers require the reviewed workbook edition")
    if key.scope not in plan.get("published_scopes", SCOPES) or key.scope != key.table:
        raise ContractError("Original CTE headers require the reviewed worksheet scope")
    case = plan["scopes"][key.scope]
    fields = [row["field"] for row in headers]
    support = _source_support(fields)
    if (fields != case["fields"] or [row["code"] for row in headers] != case["headers"]
            or tuple(fields) != ref.public_columns or ref.contract["key_columns"] != case["keys"]
            or ref.contract.get("column_support") != support):
        raise ContractError("Original CTE headers require the reviewed workbook field order and raw support")
    return {field: "raw_" + field for field in support["fields"]}


def read_cte_sheet(archive: ZipFile, member: str, strings: list[str], headers: list[str], fields: list[str]):
    """Use the bounded literal reader and retain publisher text and numeric lexemes."""
    row_count, rows = _sheet_frame(archive, member, strings, header_row=1, codes=headers, literal=True,
                                  max_rows=MAX_WORKSHEET_ROWS)
    return _literal_batch(rows, strings, fields, 1, row_count)


def cte_workbook_projection(raw):
    """Project directory keys while preserving their publisher spelling."""
    import pyarrow as pa

    columns = dict(zip(raw.schema.names, raw.columns, strict=True))
    for field, width in _ID_WIDTHS.items():
        if field not in columns:
            continue
        source = columns[field]
        name = "raw_" + field
        columns[name] = source
        columns[field] = canonical_source_identifiers(source, width=width)
    columns[_SOURCE_ROW] = pa.array(range(2, 2 + raw.num_rows), type=pa.int64())
    return pa.table(columns), _source_support(raw.schema.names)


def _check_workbook_cell(field: str, value: object, row: dict[str, Any]) -> bool:
    native = row[field]
    if field in _ID_WIDTHS:
        original = row["raw_" + field]
        token = str(value) if value is not None else None
        return original == token and native == canonical_source_identifier(token, width=_ID_WIDTHS[field])
    if type(value) is float:
        # The independent reader decodes Excel's numeric representation. The
        # original XML check separately enforces the unrounded lexical value.
        return isinstance(native, str) and float(native) == value
    return native == (None if value is None else str(value))


def _workbook_scope(release: DataRelease, ref, archive: ZipFile, sheets: dict[str, str], strings: list[str],
                    book, case: dict[str, Any]) -> dict[str, int]:
    import pyarrow as pa

    sheet = case["sheet"]
    scope = ref.edition.scope
    headers, fields = case["headers"], case["fields"]
    raw = read_cte_sheet(archive, sheets[sheet], strings, headers, fields)
    native = release.analytical_dataset(ref.identity)
    projection, _ = cte_workbook_projection(raw)
    if (raw.num_rows != case["rows"] or tuple(ref.contract["key_columns"]) != tuple(case["keys"])
            or tuple(native.schema.names) != tuple(projection.column_names)
            or not native.to_table().replace_schema_metadata(None).equals(projection, check_metadata=True)):
        raise ContractError(f"CTE workbook {ref.edition.dataset}/{scope} original XML replay differs")
    if case["keys"] and (any(projection[name].null_count for name in case["keys"])
            or raw.num_rows != projection.select(case["keys"]).group_by(case["keys"]).aggregate([]).num_rows):
        raise ContractError(f"CTE workbook {ref.edition.dataset}/{scope} contains missing or duplicate source keys")
    worksheet = book[sheet]
    worksheet.reset_dimensions()
    rows = worksheet.iter_rows(max_col=len(fields), values_only=True)
    if next(rows) != tuple(headers):
        raise ContractError(f"CTE workbook {ref.edition.dataset}/{scope} independent headers differ")
    for ordinal, (decoded, leaf) in enumerate(zip(rows, projection.to_pylist(), strict=True), 2):
        if (leaf[_SOURCE_ROW] != ordinal
                or not all(_check_workbook_cell(field, value, leaf)
                           for field, value in zip(fields, decoded, strict=True))):
            raise ContractError(f"CTE workbook {ref.edition.dataset}/{scope} independent row {ordinal} differs")
    # Cross-check the native public schema, independent of the Arrow writer.
    if not all(pa.types.is_string(projection.schema.field(field).type) for field in fields):
        raise ContractError("CTE original text columns must preserve their workbook type contract")
    return {"rows": raw.num_rows, "cells": raw.num_rows * len(fields)}


def qualify_cte_workbook(release: DataRelease, store: ProvenanceStore) -> dict[str, Any]:
    plan = cte_workbook_qualifications().get(release.manifest_sha256)
    if not release.offline or plan is None:
        raise ContractError("CTE workbook qualification requires its reviewed offline manifest")
    from openpyxl import load_workbook

    record = store.verify_record_closure([plan["publisher_receipt"]])[plan["publisher_receipt"]]
    if (record["kind"] != "acquisition" or record["method"] != "download" or not record["downloaded_at"]
            or record["artifact"]["sha256"] != plan["sha256"]
            or not 0 < record["artifact"]["size_bytes"] <= MAX_SOURCE_BYTES):
        raise ContractError("CTE workbook publisher receipt differs")
    payload = store.verify_artifact(plan["sha256"])
    refs = [release.catalog.artifact_ref(key) for key in release.catalog.to_dict()["artifacts"]]
    expected_scopes = set(plan.get("published_scopes", SCOPES))
    if ({ref.edition.scope for ref in refs} != expected_scopes or len(refs) != len(expected_scopes)
            or any((ref.edition.dataset, ref.edition.edition) != (plan["dataset"], plan["edition"])
                   or list(ref.contract["inputs"]) != [plan["sha256"]] for ref in refs)):
        raise ContractError("CTE workbook scope, edition or source identities differ")
    with ZipFile(payload) as archive, payload.open("rb") as stream:
        sheets, strings = _package(archive, literal=True)
        if set(sheets) != {plan["about"], *(case["sheet"] for case in plan["scopes"].values())}:
            raise ContractError("CTE workbook has an undeclared sheet")
        book = load_workbook(stream, read_only=True, data_only=False, keep_links=False)
        try:
            checked = {ref.edition.scope: _workbook_scope(release, ref, archive, sheets, strings, book,
                        plan["scopes"][ref.edition.scope]) for ref in refs}
        finally:
            book.close()
    return {"status": "PASS_REVIEWED_NATIVE", "manifest_sha256": release.manifest_sha256,
            "build_id": release.build_id, "publisher_receipt": plan["publisher_receipt"], "scopes": checked,
            "limits": "Exact public workbook fields and separate populations. No inference of masked values, "
                      "cross-category totals or equivalence across report years."}
