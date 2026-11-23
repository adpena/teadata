"""Qualify exact archival STAAR and accountability CCMR source tables."""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from importlib.resources import files
import json
import re
from typing import TYPE_CHECKING, Any

from .manifest import ContractError
from .source_parity import validate_source_parity

if TYPE_CHECKING:
    from .provenance import ProvenanceStore
    from .release import DataRelease


@lru_cache(maxsize=1)
def archive_qualifications() -> dict[str, Any]:
    """The reference cells come from named publisher HTML/PDF, never from the leaf."""
    resource = files("teadata").joinpath("reference_cases/native_tabular_archives.json")
    return json.loads(resource.read_bytes())


def archive_publisher_roots(plan: dict[str, Any]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        identifier for scope in plan["scopes"].values()
        for identifier in (scope["original"], *(case["receipt"] for case in scope["witnesses"]))
    ))


def _staar_display(raw: str) -> str:
    if raw in {"", "."}:
        return "-"
    if raw == "-1":
        return "*"
    if re.fullmatch(r"(?:100|[1-9]?[0-9])", raw):
        return raw
    raise ContractError(f"Unexpected historical STAAR token {raw!r} in named report check")


def _ccmr_matches(raw: str, display: str, *, rounded: tuple[str, str] | None) -> bool:
    token = display.replace(",", "").removesuffix("%")
    if rounded is not None:
        return (raw, token) == rounded
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", raw) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", token):
        return False
    return Decimal(raw) == Decimal(token)


def qualify_tabular_archive(release: DataRelease, store: ProvenanceStore) -> dict[str, Any]:
    """Check exact edition and row counts, named publisher cells and all original values."""
    plan = archive_qualifications().get(release.manifest_sha256)
    if not release.offline or plan is None:
        raise ContractError("Archive qualification requires an exact reviewed offline manifest")
    from pyarrow import dataset as ds

    family, year = plan["family"], plan["edition"]
    refs = [release.catalog.artifact_ref(key) for key in release.catalog.to_dict()["artifacts"]]
    scopes = plan["scopes"]
    if (len(refs) != len(scopes) or {ref.edition.scope for ref in refs} != scopes.keys()
            or any((ref.edition.table, ref.edition.edition) != (family, year) for ref in refs)):
        raise ContractError("Archive manifest has a different edition or set of scopes")
    publisher = store.verify_record_closure(archive_publisher_roots(plan))
    checked = []
    for ref in refs:
        scope = ref.edition.scope
        spec = scopes[scope]
        original_hash = publisher[spec["original"]]["artifact"]["sha256"]
        if ref.contract["rows"] != spec["rows"] or ref.contract["inputs"][0] != original_hash:
            raise ContractError(f"Archive data differs for {year} {scope}")
        for case in spec["witnesses"]:
            record = publisher[case["receipt"]]
            if (record["kind"] != "acquisition" or record["method"] != "download"
                    or record["downloaded_at"] is None or not record["resolved_url"]):
                raise ContractError(f"Archive comparator lacks publisher acquisition for {year} {scope}")
            key, identifier = f"{scope}_number", case["entity"]
            where = None if scope == "state" else ds.field(key).isin([identifier, "'" + identifier])
            fields = case["cells"]
            selection = release.analytical_dataset(ref.identity).to_table(
                columns=[code.lower() for code in fields], filter=where)
            if selection.num_rows != 1:
                raise ContractError(f"Archive comparator missing or duplicated for {year} {scope} {identifier}")
            row, = selection.to_pylist()
            for code, display in fields.items():
                raw = row[code.lower()]
                if family == "staar_all_grades":
                    matches = _staar_display(raw) == display.removesuffix("%")
                else:
                    rounded = case.get("rounded", {}).get(code)
                    matches = _ccmr_matches(raw, display, rounded=tuple(rounded) if rounded else None)
                if not matches:
                    raise ContractError(
                        f"Archive publisher cell differs for {year} {scope} {identifier} {code}: {raw!r}, {display!r}")
            checked.append({"edition": year, "scope": scope, "entity": identifier,
                            "publisher_receipt": case["receipt"], "compared_cells": len(fields)})
    parity = validate_source_parity(release, store, edition_ids=[ref.edition.identity for ref in refs],
                                    working_bytes=512 * 1024**2, verify_export=True)
    return {"status": "PASS_REVIEWED_NATIVE" if parity["status"] == "PASS_RAW_PARITY" else "HOLD",
            "manifest_sha256": release.manifest_sha256, "build_id": release.build_id,
            "publisher_witnesses": checked, "raw_parity": parity,
            "limits": "Independent publisher cells establish only the checked fields and periods. Other fields "
                      "retain original strings. Editions do not imply longitudinal equivalence."}
