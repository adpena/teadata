"""Independent XML-to-native qualification of the exact facility source family."""
from __future__ import annotations

from collections import Counter, defaultdict
import base64
from decimal import Decimal
from hashlib import sha256
from importlib.resources import files
import json
from pathlib import Path
import re
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pyarrow as pa

from .dataset_metadata import EditionKey
from .manifest import ContractError, canonical_json
from .provenance import ProvenanceStore
from .release import DataRelease


def review_template() -> dict[str, Any]:
    return json.loads(files("teadata").joinpath("reference_cases/native_facility_usage.json").read_text())


def reviewed_facility_usage(pin: str) -> dict[str, Any] | None:
    plan = review_template()
    return plan if pin == plan["manifest_sha256"] else None


def capture_facility_identity(release: DataRelease) -> dict[str, Any]:
    """Capture actual custody identities; never derive semantic expected values."""
    plan = review_template()
    document = release.catalog.to_dict()
    plan.update(manifest_sha256=release.manifest_sha256, build_id=release.build_id, editions={})
    for artifact_id, edition_id in document["artifacts"].items():
        ref = release.catalog.artifact_ref(artifact_id)
        plan["editions"][edition_id] = {"entry": document["editions"][edition_id], "artifact": ref.to_dict()}
    return plan


def validate_facility_identity(release: DataRelease, plan: dict[str, Any]) -> None:
    document = release.catalog.to_dict()
    if (plan["schema_version"] != 1 or release.manifest_sha256 != plan["manifest_sha256"]
            or release.build_id != plan["build_id"] or document["schema_version"] != 4
            or len(plan["editions"]) != 2 or document["editions"].keys() != plan["editions"].keys()):
        raise ContractError("Facility reviewed manifest/build/edition identity differs")
    seen = set()
    for edition_id, case in plan["editions"].items():
        if document["editions"][edition_id] != case["entry"]:
            raise ContractError("Facility edition metadata identity differs")
        matches = [aid for aid, eid in document["artifacts"].items() if eid == edition_id]
        if len(matches) != 1 or release.catalog.artifact_ref(matches[0]).to_dict() != case["artifact"]:
            raise ContractError("Facility artifact or processing identity differs")
        key = EditionKey.from_dict(case["entry"]["key"])
        if (key.scope != "district" or key.edition != "2026" or key.table != "district"
                or key.revision_sha256 != EditionKey.source_revision((plan["source_sha256"],))):
            raise ContractError("Facility source scope/revision differs")
        ref = release.catalog.artifact_ref(matches[0])
        fields = [f for f in plan["physical_contract"] if key.dataset == "facility_usage"
                  or f["coordinate"] in ("A", "B", "AE", "AF")]
        schema = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(ref.contract["schema"], validate=True)))
        support = ref.contract["column_support"]
        expected_support = {}
        for field in fields:
            name, physical = field["name"], field["type"]
            if physical.startswith("decimal128("):
                dtype = pa.decimal128(*map(int, physical[11:-1].split(",")))
            else:
                dtype = {"string": pa.string(), "int8": pa.int8(), "int16": pa.int16(), "int32": pa.int32()}[physical]
            expected_support[name] = {"raw": "raw_" + name}
            if physical != "string":
                expected_support[name]["status"] = "status_" + name
            if schema.field(name).type != dtype:
                raise ContractError("Facility fixed physical field type differs")
            metadata = release.catalog.field_metadata(name, scope="district", edition=key)
            if field["disposition"] == "fact" and (metadata is None or metadata["period"] != {
                    "kind": "school_year", "start_year": 2025, "end_year": 2026}):
                raise ContractError("Facility field collection-period binding differs")
        if support["fields"] != expected_support or set(ref.public_columns) != {f["name"] for f in fields}:
            raise ContractError("Facility public fields or raw/value-status support differs")
        seen.add(key.dataset)
    if seen != {"facility_usage", "facility_usage_submission"} or len(document["artifacts"]) != 2:
        raise ContractError("Facility source family is incomplete")


def _raw_rows(path: Path):
    """Independent source reader: no producer parser, projection or normalizer calls."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with ZipFile(path) as archive:
        domains = defaultdict(set)
        for row in ET.fromstring(archive.read("xl/worksheets/sheet3.xml")).iter(ns + "row"):
            if row.attrib["r"] == "1":
                continue
            for cell in row:
                inline = cell.find(ns + "is")
                text = ("" if inline is None else "".join(inline.itertext())) if cell.get("t") == "inlineStr" else cell.findtext(ns + "v")
                if text not in (None, ""):
                    domains[re.sub(r"[0-9]", "", cell.attrib["r"])].add(text)
        rows = []
        header = {}
        for row in ET.fromstring(archive.read("xl/worksheets/sheet2.xml")).iter(ns + "row"):
            values, kinds = {}, {}
            contact_blank = True
            for cell in row:
                coord = re.sub(r"[0-9]", "", cell.attrib["r"])
                if cell.find(ns + "f") is not None:
                    raise ContractError("Facility qualification found an unexpected formula")
                kind = cell.get("t", "n")
                if kind == "inlineStr":
                    inline = cell.find(ns + "is")
                    text = "" if inline is None else "".join(inline.itertext())
                    cell_kind = "inlineStr_empty" if text == "" else "inlineStr"
                elif kind == "n":
                    text = cell.findtext(ns + "v")
                    cell_kind = "n_empty" if text in (None, "") else "n"
                else:
                    raise ContractError("Facility qualification found an unreviewed XML type")
                if row.attrib["r"] == "1":
                    header[coord] = text
                elif coord == "C":
                    contact_blank = text in (None, "")
                else:
                    values[coord], kinds[coord] = text, cell_kind
            if row.attrib["r"] != "1":
                rows.append((int(row.attrib["r"]), values, kinds, contact_blank))
        return header, rows, domains


def _typed(raw: str | None, field: dict[str, Any]):
    if raw in (None, ""):
        return None
    if field["type"] == "string":
        if field["coordinate"] == "B":
            if re.fullmatch(r"[0-9]{6}", raw) is None or int(raw) == 0:
                raise ContractError("Facility independent district token check failed")
            return "'" + raw
        return raw
    value = Decimal(raw)
    if not value.is_finite():
        raise ContractError("Facility independent numeric check found nonfinite data")
    if field["type"].startswith("int"):
        if value != int(value):
            raise ContractError("Facility independent integral token check failed")
        return int(value)
    return value


def _check_fields(native: dict[str, Any], values: dict[str, str | None], fields: list[dict[str, Any]], row: int):
    for field in fields:
        name, coord = field["name"], field["coordinate"]
        raw = values.get(coord)
        expected = _typed(raw, field)
        if native.get("raw_" + name) != raw or native.get(name) != expected or type(native.get(name)) is not type(expected):
            raise ContractError(f"Facility raw/typed source mismatch at {coord}{row}")
        if field["type"] == "string":
            if "status_" + name in native:
                raise ContractError("Facility strings must use raw-only support")
        elif native.get("status_" + name) != ("missing" if raw in (None, "") else "observed"):
            raise ContractError(f"Facility numeric value status differs at {coord}{row}")


def _rows(release: DataRelease, artifact_id: str):
    for batch in release.analytical_dataset(artifact_id).to_batches(batch_size=512, use_threads=False):
        yield from batch.to_pylist()


def qualify_facility_usage(release: DataRelease, processing_store: ProvenanceStore,
                            publisher_store: ProvenanceStore, plan: dict[str, Any]) -> dict[str, Any]:
    validate_facility_identity(release, plan)
    release.verify()
    publisher = publisher_store.verify_record_closure(plan["publisher_roots"])
    refs = {release.catalog.artifact_ref(aid).edition.dataset: release.catalog.artifact_ref(aid)
            for aid in release.catalog.to_dict()["artifacts"]}
    processing = processing_store.verify_record_closure([ref.processing_id for ref in refs.values()])
    source = publisher[plan["publisher_roots"][0]]
    if (source.get("method") != "local_observation" or source["artifact"]["sha256"] != plan["source_sha256"]
            or source.get("derived_from") != "74f9b06dd69a379c0e084fc0d1b3b0918ba8275f07b06aa42eb29d6e67f01b58"):
        raise ContractError("Facility publisher extraction chain differs")
    rawpath = publisher_store.verify_artifact(plan["source_sha256"])
    header, rawrows, domains = _raw_rows(rawpath)
    fields = plan["physical_contract"]
    if header != {f["coordinate"]: f["header"] for f in fields} | {"C": "Contact Email Address"}:
        raise ContractError("Facility original header mapping differs")
    facilities = refs["facility_usage"]
    submitted = refs["facility_usage_submission"]
    expected = plan["expected"]
    if facilities.contract["rows"] != expected["facility_rows"] or submitted.contract["rows"] != expected["submission_rows"]:
        raise ContractError("Facility exact table populations differ")
    projection = release.catalog.resolve_artifacts(((submitted.edition, "submission_status"),))
    if len(projection) != 1 or projection[0]["artifact"].identity != submitted.identity:
        raise ContractError("Submission query unexpectedly expands to another leaf")
    for ref in refs.values():
        rec = processing[ref.processing_id]
        if (rec["outputs"]["leaf"]["sha256"] != ref.sha256
                or rec["parameters"]["leaf_contract_sha256"] != sha256(canonical_json(ref.contract).encode()).hexdigest()):
            raise ContractError("Facility processing contract/leaf binding differs")
    group_record = processing[submitted.processing_id]
    recipe = group_record["parameters"]["recipe"]
    if (facilities.processing_id not in group_record["inputs"]
            or recipe.get("source_facility_processing") != facilities.processing_id
            or recipe.get("group_by") != "district_number" or recipe.get("require_equal") != ["A", "B", "AE", "AF"]
            or recipe.get("original_coordinates") != {f["name"]: f["coordinate"] for f in fields if f["coordinate"] in ("A", "B", "AE", "AF")}):
        raise ContractError("Facility submission processing lineage or grouping recipe differs")
    native = _rows(release, facilities.identity)
    groups = defaultdict(list)
    category_cells = Counter()
    category_rows = 0
    status_only = 0
    observed_goldens = set()
    mapping = {"D": "E", "E": "D", "K": "A", "O": "D", "S": "C", "T": "D",
               "U": "F", "V": "F", "W": "F", "X": "F", "Y": "B", "Z": "G"}
    email = re.compile(r"[^\s@]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
    for position, (ordinal, values, kinds, contact_blank) in enumerate(rawrows, 2):
        if ordinal != position:
            raise ContractError("Facility original row ordinal coverage differs")
        row = next(native, None)
        if row is None:
            raise ContractError("Facility native records end before publisher source")
        _check_fields(row, values, fields, ordinal)
        non_submitted = values["AF"] != "SUBMITTED"
        if non_submitted:
            status_only += 1
            if (not contact_blank or values["AF"] not in ("INCOMPLETE", "NOT SUBMITTED")
                    or any(v not in (None, "") for c, v in values.items() if c not in ("A", "B", "AE", "AF"))):
                raise ContractError("Facility status-only source payload differs")
        violations = []
        for coord, domain in mapping.items():
            raw = values.get(coord)
            tokens = [x.strip() for x in raw.split(",")] if coord == "Z" and raw else [raw]
            if raw not in (None, "") and any(token not in domains[domain] for token in tokens):
                violations.append(coord)
                category_cells[coord] += 1
        category_rows += bool(violations)
        context = {"publisher_worksheet": "Facilities Data", "publisher_worksheet_row": ordinal,
                   "record_kind": "status_only" if non_submitted else "facility_observation",
                   "source_cell_types_json": canonical_json({f["coordinate"]: kinds.get(f["coordinate"], "absent") for f in fields}),
                   "quality_conformance_json": canonical_json({"outside_validation_columns": sorted(violations)})}
        if any(row[k] != v for k, v in context.items()):
            raise ContractError(f"Facility source locator or context differs at row {ordinal}")
        if any(email.search(value) for value in row.values() if isinstance(value, str)):
            raise ContractError(f"Facility public text privacy check failed at row {ordinal}")
        for coord, raw in values.items():
            address = f"{coord}{ordinal}"
            if address in plan["goldens"]:
                if raw != plan["goldens"][address]:
                    raise ContractError(f"Facility frozen publisher example differs at {address}")
                observed_goldens.add(address)
        # Absent-cell goldens must be checked too.
        for address, raw in plan["goldens"].items():
            match = re.fullmatch(r"([A-Z]+)([0-9]+)", address)
            if match is not None and int(match[2]) == ordinal:
                if values.get(match[1]) != raw:
                    raise ContractError(f"Facility frozen publisher example differs at {address}")
                observed_goldens.add(address)
        groups[values["B"]].append((ordinal, {c: values.get(c) for c in ("A", "B", "AE", "AF")}))
    if (next(native, None) is not None or len(rawrows) != expected["facility_rows"] or status_only != 71
            or category_cells != expected["category_cells"] or category_rows != expected["category_rows"]
            or observed_goldens != plan["goldens"].keys()):
        raise ContractError("Facility full source coverage, quality flags or publisher examples differ")
    native_groups = _rows(release, submitted.identity)
    statuses = Counter()
    for identifier, source_rows in sorted(groups.items()):
        values = {}
        for coord in ("A", "B", "AE", "AF"):
            choices = {row[coord] for _, row in source_rows}
            if len(choices) != 1 or None in choices or "" in choices:
                raise ContractError("Facility independent district status grouping is inconsistent")
            values[coord] = choices.pop()
        row = next(native_groups, None)
        if row is None:
            raise ContractError("Facility submission records end before source groups")
        chosen = [f for f in fields if f["coordinate"] in ("A", "B", "AE", "AF")]
        _check_fields(row, values, chosen, source_rows[0][0])
        ordinals = sorted(ordinal for ordinal, _ in source_rows)
        context = {"publisher_worksheet": "Facilities Data", "source_group_row_count": len(ordinals),
                   "source_group_min_row": ordinals[0], "source_group_max_row": ordinals[-1],
                   "source_group_rows_sha256": sha256(canonical_json(ordinals).encode()).hexdigest()}
        if any(row[k] != v for k, v in context.items()):
            raise ContractError("Facility submission original-row grouping lineage differs")
        statuses[values["AF"]] += 1
    if next(native_groups, None) is not None or len(groups) != expected["submission_rows"] or statuses != expected["statuses"]:
        raise ContractError("Facility submission status population differs")
    return {"status": "PASS_REVIEWED_NATIVE", "manifest_sha256": release.manifest_sha256,
            "build_id": release.build_id, "source_rows": len(rawrows), "districts": len(groups),
            "status_only_rows": status_only, "category_cells": dict(category_cells), "category_rows": category_rows,
            "publisher_examples": len(observed_goldens), "processing_records": len(processing),
            "publisher_records": len(publisher), "limits": plan["limits"], "qualification_scope": "as_published_district_records"}
