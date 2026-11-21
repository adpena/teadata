"""Compare the core of two pinned TEA Data releases without rebuilding either."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from teadata.manifest import ContractError, canonical_json
from teadata.release import DataRelease

_CATALOG_SELECTION_KEYS = {"build_id", "native_sources"}
_PROCESSING_INPUT_KEYS = ("dataset", "processor", "code_version", "inputs")


def _different_keys(before: Mapping[str, Any], after: Mapping[str, Any]) -> set[str]:
    return (set(before) ^ set(after)) | {
        key for key in before.keys() & after.keys()
        if canonical_json(before[key]) != canonical_json(after[key])
    }


def compare_cores(before: DataRelease, after: DataRelease, *, verify_equal_artifacts: bool = False) -> dict[str, Any]:
    """Compare input identity, attached editions and declared artifact bytes separately."""
    catalogs = before.catalog.to_dict(), after.catalog.to_dict()
    if any(document["schema_version"] != 2 for document in catalogs):
        raise ContractError("Compare two normal entity releases with schema 2 catalogs.")
    if not before.offline or not after.offline:
        raise ContractError("Select both releases in offline mode.")

    core_changes = _different_keys(*catalogs) - _CATALOG_SELECTION_KEYS
    old_record, new_record = before.processing["record"], after.processing["record"]
    inputs_changed = _different_keys(
        {key: old_record[key] for key in _PROCESSING_INPUT_KEYS if key in old_record},
        {key: new_record[key] for key in _PROCESSING_INPUT_KEYS if key in new_record},
    )
    old_parameters, new_parameters = old_record["parameters"], new_record["parameters"]
    inputs_changed.update(f"parameters.{key}" for key in
                          _different_keys(old_parameters, new_parameters) - _CATALOG_SELECTION_KEYS)
    old_refs, new_refs = before.catalog.native_sources, after.catalog.native_sources
    prefix_retained = new_refs[:len(old_refs)] == old_refs
    appended = new_refs[len(old_refs):] if prefix_retained else ()
    existing_keys = {(key.dataset, key.edition, key.scope, key.table) for ref in old_refs for key in ref.editions}
    appended_keys = [(key.dataset, key.edition, key.scope, key.table) for ref in appended for key in ref.editions]
    additive = bool(appended) and not (existing_keys & set(appended_keys)) and len(set(appended_keys)) == len(appended_keys)

    # These descriptors have already been matched to the processing receipts
    # and constrained to the release's exact role set by the readers.
    old_outputs, new_outputs = before._artifacts, after._artifacts
    roles = {}
    for role in sorted(old_outputs.keys() | new_outputs.keys()):
        old = old_outputs.get(role)
        new = new_outputs.get(role)
        same = old is not None and new is not None and all(old[key] == new[key] for key in ("sha256", "size_bytes"))
        verified = False
        if verify_equal_artifacts and same:
            before.artifact(role)
            after.artifact(role)
            verified = True
        roles[role] = {
            "before_bytes": old["size_bytes"] if old is not None else None,
            "after_bytes": new["size_bytes"] if new is not None else None,
            "declared_same_bytes": same,
            "verified_under_both_releases": verified,
        }
    return {
        "before": {"manifest_sha256": before.manifest_sha256, "build_id": before.build_id},
        "after": {"manifest_sha256": after.manifest_sha256, "build_id": after.build_id},
        "changed_core_catalog_sections": sorted(core_changes),
        "changed_core_processing_inputs": sorted(inputs_changed),
        "retains_existing_native_sources": prefix_retained,
        "adds_distinct_native_editions": additive,
        "appended_native_sources": [ref.manifest_sha256 for ref in appended],
        "same_catalog_build_id": before.build_id == after.build_id,
        "artifacts": roles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path, help="Retained normal entity manifest.")
    parser.add_argument("after", type=Path, help="Retained candidate entity manifest.")
    parser.add_argument("--before-sha256", required=True)
    parser.add_argument("--after-sha256", required=True)
    parser.add_argument("--verify-equal-artifacts", action="store_true",
                        help="Run each identical declared output through both release readers.")
    args = parser.parse_args()
    try:
        before = DataRelease(args.before, expected_sha256=args.before_sha256, offline=True)
        after = DataRelease(args.after, expected_sha256=args.after_sha256, offline=True)
        result = compare_cores(before, after, verify_equal_artifacts=args.verify_equal_artifacts)
    except (ContractError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
