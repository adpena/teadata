"""Offline qualification of the reviewed FY2025 finance and TAPR staff release."""

from __future__ import annotations

from decimal import Decimal
from hashlib import sha256
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from . import frc_ppe
from .column_schema import crosswalk_dataset, parse_column_schema
from .dataset_metadata import DataPeriod
from .manifest import ContractError
from .native_witnesses import _BROKER, _Rows
from .source_parity import validate_source_parity
from .values import ValueStatus

if TYPE_CHECKING:
    from .dataset_metadata import ArtifactRef, ReferenceDocument
    from .provenance import ProvenanceStore
    from .release import DataRelease


REVIEWED_MANIFEST = "1c5cc0185fafefca1304eb8c2d71e6c5f95819c672f1fd022c08b0d05a096065"
_STAFF_ENTITIES = {"campus": "084908042", "district": "084908"}
_STAFF_PERIOD = DataPeriod("school_year", 2024, 2025)
_STAFF_MAX_BYTES = 1024**2


def _reviewed_leaves(release: DataRelease) -> tuple[dict[str, ArtifactRef], dict[str, ArtifactRef]]:
    if release.manifest_sha256 != REVIEWED_MANIFEST or not release.offline:
        raise ContractError("Finance and staff qualification requires the exact reviewed offline release")
    index = release.catalog.to_dict()
    refs = [release.catalog.artifact_ref(identifier) for identifier in index["artifacts"]]
    finance = {ref.edition.scope: ref for ref in refs if ref.edition.dataset == frc_ppe.DATASET}
    staff = {ref.edition.scope: ref for ref in refs if ref.edition.dataset == f"{ref.edition.scope}_tapr_staff_profile"}
    if (index["schema_version"] != 4 or len(refs) != 5 or len(index["editions"]) != 5
            or set(finance) != {"state", "district", "campus"} or set(staff) != _STAFF_ENTITIES.keys()
            or any(ref.edition.edition != "2025" for ref in refs)):
        raise ContractError("Finance and staff release does not contain exactly the five reviewed scopes")
    return finance, staff


def _staff_reference(release: DataRelease, ref: ArtifactRef) -> ReferenceDocument:
    metadata = release.catalog.metadata(ref.edition.dataset, edition=ref.edition)
    reference, = (item for item in metadata.references if item.reference_id == "published_staff_comparison")
    return reference


def retained_finance_staff_evidence(release: DataRelease, store: ProvenanceStore) -> list[str]:
    """Rehash the exact source closure before using a composed-release receipt."""
    finance, staff = _reviewed_leaves(release)
    roots = [ref.processing_id for ref in (*finance.values(), *staff.values())]
    roots.extend(identifier for ref in staff.values() for identifier in _staff_reference(release, ref).provenance_ids)
    return sorted(store.verify_record_closure(roots))


def _finance_parity(release: DataRelease, store: ProvenanceStore, refs: dict[str, ArtifactRef]) -> list[dict[str, Any]]:
    catalog = release.catalog
    acquisitions = {}
    for scope, ref in refs.items():
        acquisition, = catalog.acquisitions(frc_ppe.DATASET, edition=ref.edition)
        acquisitions[scope] = acquisition["record_id"]
    results = []
    for scope, ref in sorted(refs.items()):
        key = ref.edition
        metadata = catalog.metadata(key.dataset, edition=key)
        policy = catalog.policy(key.dataset, edition=key)
        if (metadata.period != frc_ppe.PERIOD or set(policy.tokens) != {""}
                or policy.tokens[""].status != ValueStatus.MISSING):
            raise ContractError("FRC PPE replay requires the reviewed fiscal year and blank policy")
        schema, = (reference.reviewed_sha256 for reference in metadata.references if reference.role == "schema")
        if schema is None:
            raise ContractError("FRC PPE replay requires a reviewed crosswalk")
        document = parse_column_schema(catalog.column_schema_bytes(schema))
        mapping, = (table for table in crosswalk_dataset(document, key.dataset, key.edition)["tables"]
                    if table["table"] == scope)
        graph = store.verify_record_closure([ref.processing_id])
        processing = graph[ref.processing_id]
        recipe = processing["parameters"]["recipe"]
        witnesses = recipe["witness_receipts"]
        receipt = acquisitions[scope]
        if (processing != catalog._processing_record(ref) or processing["processor"] != "teadata.frc_ppe"
                or recipe["operation"] != "frc_ppe_published_totals" or receipt not in processing["inputs"]
                or len(witnesses) != (0 if scope == "state" else 1)
                or any(witness not in processing["inputs"] for witness in witnesses)):
            raise ContractError("FRC PPE replay differs from the original processing and witness bindings")
        expected, support = frc_ppe._scope_projection(
            store, metadata, mapping, scope, receipt, acquisitions["state"], witnesses[0] if witnesses else None,
        )
        actual = release.analytical_dataset(ref.identity).to_table().replace_schema_metadata(None)
        if support != ref.contract["column_support"] or not actual.equals(expected, check_metadata=True):
            raise ContractError(f"FRC PPE {scope} values, statuses or source rows differ from publisher replay")
        results.append({
            "scope": scope, "edition_id": key.identity, "source_receipt": receipt, "rows": actual.num_rows,
            "compared_cells": actual.num_rows * actual.num_columns, "witness_receipts": witnesses,
            "publisher_witness_cells": len(frc_ppe._PPE_HEAD) * len(witnesses),
        })
    return results


def _staff_cells(content: str, scope: str, entity: str) -> dict[str, tuple[str, str]]:
    rows = _Rows(content).rows
    district = scope == "district"
    labels = ["Count", "Percent", "Count", "Percent"] if district else ["Count/Average", "Percent", "District", "State"]
    if (len(rows) < 6 or rows[:2] != [["", "Texas Education Agency", ""], ["2024-25 Staff Information"]]
            or len(rows[2]) != 1 or f"({entity})" not in rows[2][0]
            or rows[5] != ["Staff Information", *labels]):
        raise ContractError("TAPR Staff Information report period or geography differs")
    professional, = (i for i, row in enumerate(rows) if len(row) == 5 and row[0] == "Professional Staff")
    support, = (i for i, row in enumerate(rows) if len(row) == 5 and row[0] == "Professional Support")
    experience = rows.index(["Average Years Experience of Campus Staff"])
    salary_by_experience = rows.index(["Average Teacher Salary by Years of Experience (regular duties only)"])
    salary = rows.index(["Average Actual Salaries (regular duties only)"])

    def value(label: str, group: list[list[str]]) -> str:
        token, = (row[1] for row in group if len(row) > 1 and row[0] == label)
        return token

    cells = {
        "teacher_total_full_time_equiv_count": ("fte", value("Teachers", rows[professional + 1:support])),
        "teacher_experience_average": ("years", value("Teachers", rows[experience + 1:salary_by_experience])),
        "teacher_total_base_salary_average": ("currency", value("Teachers", rows[salary + 1:salary + 2])),
        "teacher_student_ratio": ("ratio", value("Number of Students per Teacher", rows)),
    }
    if district:
        cells["teacher_turnover_ratio"] = ("percent", value("Turnover Rate for Teachers", rows))
    return cells


def _staff_witness(release: DataRelease, store: ProvenanceStore, ref: ArtifactRef) -> dict[str, Any]:
    from pyarrow.dataset import field

    key = ref.edition
    entity = _STAFF_ENTITIES[key.scope]
    controls = {"_debug": "0", "_program": "perfrept.perfmast.sas", "_service": "marykay", "ccyy": "2025",
                "id": entity, "lev": key.scope[0].upper(), "prgopt": "reports/tapr/staff.sas"}
    url = f"{_BROKER}?{urlencode(controls)}"
    catalog = release.catalog
    reference = _staff_reference(release, ref)
    receipt, = reference.provenance_ids
    record = store.verify_record_closure([receipt])[receipt]
    artifact, parameters = record["artifact"], record["parameters"]
    if (reference.url != url or reference.reviewed_sha256 != artifact["sha256"]
            or artifact["sha256"] in ref.contract["inputs"] or not 0 < artifact["size_bytes"] <= _STAFF_MAX_BYTES
            or record["kind"] != "acquisition" or record["method"] != "download" or record["section"] != "reference"
            or record["dataset"] != f"reference:tapr_staff_witness_{key.scope}" or not record["downloaded_at"]
            or record["source_metadata"]["publisher"] != "Texas Education Agency"
            or (record["requested_year"], record["resolved_year"]) != (2025, 2025)
            or (record["requested_url"], record["resolved_url"]) != (url, url)
            or parameters["http_method"] != "GET" or parameters["http_status"] != 200
            or parameters["scope"] != key.scope or parameters["controls"] != controls
            or record["response_headers"].get("content-type", "").lower() != "text/html; charset=iso-8859-1"):
        raise ContractError("TAPR staff witness receipt differs from its reviewed selection")
    with store.verify_artifact(artifact["sha256"]).open("rb") as stream:
        raw = stream.read(_STAFF_MAX_BYTES + 1)
    if len(raw) != artifact["size_bytes"] or sha256(raw).hexdigest() != artifact["sha256"]:
        raise ContractError("TAPR staff witness bytes differ from the reviewed receipt")
    cells = _staff_cells(raw.decode("iso-8859-1"), key.scope, entity)
    variables = {row["field"]: row for row in catalog.variables(edition=key)}
    fields = {f"{key.scope}_2025_staff_{name}": value for name, value in cells.items()}
    key_field = f"{key.scope}_number"
    row, = release.analytical_dataset(ref.identity).head(
        2, columns=list(fields), filter=field(key_field) == entity,
    ).to_pylist()
    checks = {}
    for name, (unit, published) in fields.items():
        variable = variables[name]
        native = row[name]
        numeric = r"[0-9]+(?:\.[0-9]+)?"
        display = r"\$[0-9]{1,3}(?:,[0-9]{3})*" if unit == "currency" else (
            numeric + "%" if unit == "percent" else numeric
        )
        if (variable["unit"] != unit or variable["period"] != _STAFF_PERIOD.to_dict()
                or not isinstance(native, str) or re.fullmatch(numeric, native) is None
                or re.fullmatch(display, published) is None
                or Decimal(native) != Decimal(published.removeprefix("$").removesuffix("%").replace(",", ""))):
            raise ContractError(f"TAPR staff witness value, unit or period differs for {entity} {name}")
        checks[name] = {"unit": unit, "period": variable["period"], "native_raw": native, "publisher_value": published}
    return {"scope": key.scope, "edition_id": key.identity, "entity": entity, "receipt": receipt, "fields": checks}


def qualify_finance_staff(release: DataRelease, store: ProvenanceStore) -> dict[str, Any]:
    """Replay exact native leaves and report separately which semantics were witnessed."""
    finance, staff = _reviewed_leaves(release)
    release.verify()
    try:
        finance_parity = _finance_parity(release, store, finance)
        staff_parity = validate_source_parity(
            release, store, edition_ids=[ref.edition.identity for ref in staff.values()], verify_export=True,
        )
        status = "PASS_REVIEWED_NATIVE" if staff_parity["status"] == "PASS_RAW_PARITY" else "HOLD"
        reports = [_staff_witness(release, store, staff[scope]) for scope in sorted(staff)] if status != "HOLD" else []
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ContractError(f"Finance or staff source replay failed: {error}") from error
    return {
        "status": status, "build_id": release.build_id, "manifest_sha256": release.manifest_sha256,
        "structural_integrity": True, "finance_parity": finance_parity, "staff_parity": staff_parity,
        "publisher_witnesses": {"staff": reports},
        "staff_semantic_coverage": "Only the fields in the independent staff comparisons have reviewed units and periods.",
    }
