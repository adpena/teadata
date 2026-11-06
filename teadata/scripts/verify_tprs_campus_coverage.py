"""Verify the campus coverage of exact retained 2025–26 TPRS release leaves."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from teadata.column_schema import _unique_object
from teadata.manifest import ContractError, canonical_json
from teadata.provenance import ProvenanceStore
from teadata.release import DataRelease
from teadata.scripts.acquire_tprs_campuses import _inventory_receipts, statewide_rosters
from teadata.tprs_statewide_intake import (
    CAMPUS_COHORT_OPERATION, CAMPUS_PARTITION_OPERATION, CAMPUS_PARTITION_REPORTS,
    campus_partition_ids, student_report_resources,
)
from teadata.tprs_students import DATASET, EDITION, _DIGEST


def _selection(release: DataRelease, roster_receipts: Mapping[str, str],
               rosters: Mapping[str, set[str]]) -> tuple[str, dict[str, set[str]], int | None]:
    catalog = release.catalog.to_dict()
    if catalog["schema_version"] != 4 or len(catalog["artifacts"]) != 1 or len(catalog["editions"]) != 1:
        raise ContractError("Campus coverage requires one schema4 edition and one artifact per manifest")
    (artifact_id, edition_id), = catalog["artifacts"].items()
    if edition_id not in catalog["editions"]:
        raise ContractError("Campus release artifact does not route to its selected edition")
    artifact = release.catalog.artifact_ref(artifact_id)
    key = artifact.edition
    if ((key.dataset, key.edition, key.scope, key.table, key.grain)
            != (DATASET, str(EDITION), "campus", "campus", "one row per campus_number")
            or artifact.contract["key_columns"] != ["campus_number"] or "://" in artifact.location):
        raise ContractError("Campus coverage requires the exact local 2026 student campus edition")
    record = release.catalog._processing_record(artifact)
    recipe = record["parameters"]["recipe"]
    if recipe["operation"] not in (CAMPUS_COHORT_OPERATION, CAMPUS_PARTITION_OPERATION):
        raise ContractError("Campus coverage requires a roster-bound cohort or partition")
    refs = student_report_resources(record)
    roster = json.loads(release._column_schema_bytes(refs["roster"]), object_pairs_hook=_unique_object)
    partition = recipe.get("campus_partition")
    regions = [partition["region"]] if partition else recipe["campus_regions"]
    if (not isinstance(roster, dict) or set(roster) != set(regions)
            or any(not isinstance(roster[region], dict)
                   or roster[region].get("receipt") != roster_receipts[region] for region in regions)):
        raise ContractError("Campus release roster receipts differ from the statewide roster inventory")
    ordinal = partition["ordinal"] if partition else None
    members = {region: rosters[region] if ordinal is None else set(campus_partition_ids(rosters[region], ordinal))
               for region in regions}
    reports, _ = release._student_report_ledger(artifact, refs)
    if [report.identifier for report in reports] != sorted(set().union(*members.values())):
        raise ContractError("Campus release report keys differ from the statewide roster inventory")
    return key.identity, members, ordinal


def verify_campus_coverage(store: ProvenanceStore, roster_receipts: Mapping[str, str],
                           releases: Sequence[tuple[Path, str]]) -> dict[str, Any]:
    """Verify exact leaf identities and count their disjoint publisher roster coverage."""
    sources = tuple((path, pin) for path, pin in releases)
    roster_receipts = dict(roster_receipts)
    if any(not isinstance(path, Path) or "://" in str(path) for path, _ in sources):
        raise ContractError("Campus coverage requires local release manifests")
    if any(not isinstance(pin, str) or not _DIGEST.fullmatch(pin) for _, pin in sources):
        raise ContractError("Campus release requires an exact manifest SHA-256 pin")
    if len({pin for _, pin in sources}) != len(sources):
        raise ContractError("Duplicate campus release manifest pin")
    rosters = statewide_rosters(store, roster_receipts)
    selected: dict[str, list[dict[str, Any]]] = {region: [] for region in rosters}
    covered: dict[str, set[str]] = {region: set() for region in rosters}
    edition_ids: set[str] = set()
    for path, pin in sorted(sources, key=lambda pair: pair[1]):
        edition, members, ordinal = _selection(
            DataRelease(path, expected_sha256=pin, offline=True), roster_receipts, rosters,
        )
        if edition in edition_ids:
            raise ContractError("Duplicate campus release edition identity")
        edition_ids.add(edition)
        for region, keys in members.items():
            if covered[region] & keys:
                raise ContractError(f"Campus releases overlap in region {region}")
            covered[region].update(keys)
            selected[region].append({"manifest_sha256": pin, "edition_id": edition, "ordinal": ordinal})

    # Reopen one root at a time so metadata and artifact caches stay bounded.
    for path, pin in sources:
        DataRelease(path, expected_sha256=pin, offline=True).verify()

    regions_result = {}
    for region, roster in rosters.items():
        verified = len(covered[region])
        complete = verified == len(roster)
        partition_count = (len(roster) + CAMPUS_PARTITION_REPORTS - 1) // CAMPUS_PARTITION_REPORTS
        selected_ordinals = {item["ordinal"] for item in selected[region]}
        regions_result[region] = {
            "roster_receipt": roster_receipts[region], "eligible": len(roster), "verified": verified,
            "missing": len(roster) - verified, "complete": complete, "releases": selected[region],
            "missing_ordinals": [] if complete else sorted(set(range(partition_count)) - selected_ordinals),
        }
    eligible = sum(map(len, rosters.values()))
    verified = sum(map(len, covered.values()))
    return {
        "schema_version": 1, "status": "VERIFIED_CAMPUS_RELEASE_COVERAGE",
        "dataset": DATASET, "edition": str(EDITION), "eligible": eligible, "verified": verified,
        "missing": eligible - verified, "complete": eligible == verified, "regions": regions_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path, help="Pinned 20-region roster receipt inventory")
    parser.add_argument("--release", action="append", nargs=2, default=[], metavar=("MANIFEST", "SHA256"))
    args = parser.parse_args()
    try:
        if any("://" in path for path, _ in args.release):
            raise ContractError("Campus coverage requires local release manifests")
        if not (args.store / "records").is_dir() or not (args.store / "objects").is_dir():
            raise ContractError("Select the existing retained publisher store")
        result = verify_campus_coverage(ProvenanceStore(args.store), _inventory_receipts(args.inventory),
                                        [(Path(path), pin) for path, pin in args.release])
    except (ContractError, OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Campus coverage verification failed: {error}\n")
    print(canonical_json(result))


if __name__ == "__main__":
    main()
