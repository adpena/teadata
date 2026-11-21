"""Detached, metadata-only projections of one pinned release and its changes."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

from .dataset_metadata import EditionKey
from .manifest import ContractError

if TYPE_CHECKING:
    from .release import DataRelease


class ReleaseIdentity(TypedDict):
    source: str
    manifest_sha256: str
    build_id: str
    core_manifest_sha256: NotRequired[str]


class CatalogInventory(TypedDict):
    schema_version: Literal[1]
    release: ReleaseIdentity
    include_variables: bool
    metadata_inspected: bool
    base_inputs: list[dict[str, Any]]
    attachments: list[dict[str, Any]]
    editions: list[dict[str, Any]]
    metadata_resources: list[dict[str, Any]]
    limitations: list[str]


class CatalogChanges(TypedDict):
    added: list[dict[str, Any]]
    removed: list[dict[str, Any]]
    changed: list[dict[str, Any]]


class CatalogDiff(TypedDict):
    schema_version: Literal[1]
    before: ReleaseIdentity
    after: ReleaseIdentity
    include_variables: bool
    metadata_inspected: bool
    release_changed: bool
    changes: dict[str, CatalogChanges]
    limitations: list[str]


_LIMITATIONS = [
    "Catalog coverage is not a new acquisition, value validation, qualification or deployment approval.",
    "Publisher editions are not measurement years; use explicit field periods when metadata is included.",
    "Partitions and revisions remain distinct; this report does not establish scientific comparability.",
    "Index mode does not inspect attached semantic metadata or observations. An empty diff is not proof of their equality.",
    "Variable mode reads source/schema metadata, never observation leaves. Legacy base inputs have no complete native field index here.",
]


def _validate_flag(include_variables: bool) -> None:
    if type(include_variables) is not bool:
        raise TypeError("include_variables must be True or False, not a string or integer.")


def _identity(release: DataRelease) -> ReleaseIdentity:
    result: ReleaseIdentity = {"source": release.source, "manifest_sha256": release.manifest_sha256,
                               "build_id": release.build_id}
    if release.core_manifest_sha256 != release.manifest_sha256:
        result["core_manifest_sha256"] = release.core_manifest_sha256
    return result


def _index(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        identifier = row[key]
        if identifier in result:
            raise ContractError(f"Catalog inventory has duplicate {key}: {identifier}. Keep distinct source identities.")
        result[identifier] = row
    return result


def _reader(release: DataRelease) -> DataRelease:
    from .release import DataRelease

    return DataRelease(release.source if release.remote else release.path, expected_sha256=release.manifest_sha256,
                       offline=release.offline, cache_dir=release.cache)


def _edition(reader: DataRelease, key: EditionKey, include_variables: bool) -> dict[str, Any]:
    row: dict[str, Any] = {"edition_id": key.identity, **key.to_dict()}
    if include_variables:
        try:
            row["metadata"] = reader.catalog.metadata(key.dataset, edition=key).to_dict()
            row["variables"] = sorted(reader.catalog.variables(edition=key), key=lambda item: item["field"])
        finally:
            # This reader is disposable and owned here, never the caller's catalog.
            reader.catalog._clear_edition_caches()
            reader._column_schema_data.clear()
    return row


def catalog_inventory(release: DataRelease, *, include_variables: bool = False) -> CatalogInventory:
    _validate_flag(include_variables)
    base = []
    if not release._v3:
        for name, source in release.catalog.datasets.items():
            metadata = source["metadata"]
            row = {"dataset": name, "section": source["section"], "status": source["status"],
                   "artifact": source["artifact"], "period": metadata.get("period", {"kind": "unknown"})}
            if include_variables:
                row["metadata"] = metadata
            base.append(row)
    attachments = [ref.to_dict() for ref in release.native_sources]
    resources = ([ref.to_dict() for ref in release.catalog.edition_metadata_refs()]
                 if getattr(release.catalog, "_v4", False) else [])
    rows = []
    if release._v3:
        keys = [EditionKey.from_dict({name: row[name] for name in EditionKey.__dataclass_fields__})
                for row in release.catalog.editions(include_metadata=False)]
        reader = _reader(release) if include_variables else release
        rows.extend(_edition(reader, key, include_variables) for key in keys)
    else:
        for ref in release.native_sources:
            if include_variables:
                # One transient parent/child per attachment bounds decoded metadata lifetime.
                parent = _reader(release)
                child = parent._native_edition_release(ref.editions[0])
            else:
                child = release  # Index projection does not dereference this child.
            rows.extend(_edition(child, key, include_variables) for key in ref.editions)
    def ordered(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        return sorted(_index(items, key).values(), key=lambda item: item[key])

    return {"schema_version": 1, "release": _identity(release),
        "include_variables": include_variables, "metadata_inspected": include_variables,
        "base_inputs": ordered(base, "dataset"), "attachments": ordered(attachments, "location"),
        "editions": ordered(rows, "edition_id"), "metadata_resources": ordered(resources, "location"),
        "limitations": list(_LIMITATIONS)}


def catalog_diff(before: DataRelease, after: DataRelease, *, include_variables: bool = False) -> CatalogDiff:
    from .release import DataRelease

    _validate_flag(include_variables)
    if not isinstance(after, DataRelease):
        raise TypeError("other must be a pinned DataRelease. Open its manifest before comparing catalogs.")
    left = catalog_inventory(before, include_variables=include_variables)
    right = catalog_inventory(after, include_variables=include_variables)
    changes: dict[str, CatalogChanges] = {}
    sections = (("base_inputs", "dataset", left["base_inputs"], right["base_inputs"]),
                ("attachments", "location", left["attachments"], right["attachments"]),
                ("editions", "edition_id", left["editions"], right["editions"]),
                ("metadata_resources", "location", left["metadata_resources"], right["metadata_resources"]))
    for name, key, old_rows, new_rows in sections:
        old, new = _index(old_rows, key), _index(new_rows, key)
        changes[name] = {"added": [new[item] for item in sorted(new.keys()-old.keys())],
                         "removed": [old[item] for item in sorted(old.keys()-new.keys())],
                         "changed": [{"key": item, "before": old[item], "after": new[item]}
                                     for item in sorted(old.keys() & new.keys()) if old[item] != new[item]]}
    return {"schema_version": 1, "before": left["release"], "after": right["release"],
            "include_variables": include_variables, "metadata_inspected": include_variables,
            "release_changed": before.manifest_sha256 != after.manifest_sha256,
            "changes": changes, "limitations": list(_LIMITATIONS)}
