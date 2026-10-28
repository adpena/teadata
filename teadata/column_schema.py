"""Portable, byte-reviewed ordered column contracts without source rows."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


def reader_options(reader: Mapping[str, Any], profile: str | None) -> dict[str, Any]:
    """Select the same reader contract for schema lookup and table loading."""
    options = dict(reader)
    profiles = options.pop("profiles", {})
    if not isinstance(profiles, Mapping):
        raise ValueError("Reader profiles must be a mapping")
    if profile is not None:
        if profile not in profiles or not isinstance(profiles[profile], Mapping):
            raise ValueError(f"Reader profile {profile} is not declared")
        options.update(profiles[profile])
    if "expected_columns_sha256" in options:
        raise ValueError(
            "Replace retired expected_columns_sha256 with a column_schema resource"
        )
    return options


def portable_schema_path(base_dir: Path, relative: str) -> Path:
    """Resolve a portable path without environment, absolute, or symlink escapes."""
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or ":" in relative
    ):
        raise ValueError("Column schema must use a portable relative path")
    portable = PurePosixPath(relative)
    if (
        portable.is_absolute()
        or ".." in portable.parts
        or relative != portable.as_posix()
    ):
        raise ValueError("Column schema must stay within the source contract directory")
    root = base_dir.resolve()
    path = root.joinpath(*portable.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Column schema path escapes the source contract directory")
    return path


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Column schema contains duplicate JSON keys")
        result[key] = value
    return result


def load_column_schema(
    base_dir: Path, relative: str, reviewed_sha256: str
) -> list[str]:
    """Read exact reviewed bytes and validate the complete schema document."""
    path = portable_schema_path(base_dir, relative)
    data = path.read_bytes()
    if sha256(data).hexdigest() != reviewed_sha256:
        raise ValueError("Column schema bytes differ from the reviewed contract")
    document = json.loads(data, object_pairs_hook=_unique_object)
    if (
        not isinstance(document, dict)
        or set(document) != {"schema_version", "columns"}
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or not isinstance(document["columns"], list)
        or any(
            not isinstance(column, str) or not column.strip()
            for column in document["columns"]
        )
    ):
        raise ValueError("Invalid column schema document")
    columns = document["columns"]
    if len(columns) != len(set(columns)):
        raise ValueError("Column schema contains duplicate columns")
    return list(columns)
