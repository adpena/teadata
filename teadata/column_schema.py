"""Portable, byte-reviewed ordered column contracts without source rows."""

from __future__ import annotations

from hashlib import sha256
import json
import keyword
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping


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


def source_schema_fingerprint(labels: list[str], codes: list[str]) -> str:
    """Identify the exact ordered human-label/code headers, including whitespace."""
    data = json.dumps({"labels": labels, "codes": codes}, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=True) + "\n"
    return sha256(data.encode()).hexdigest()


def _checksum(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_header_table(table: Mapping[str, Any]) -> None:
    """Validate one explicit mapping; never synthesize or deduplicate names."""
    if (not isinstance(table, dict)
            or set(table) != {"table", "input_sha256", "source_schema_sha256", "fields"}
            or not isinstance(table["table"], str) or not table["table"].strip()
            or not _checksum(table["input_sha256"]) or not _checksum(table["source_schema_sha256"])
            or not isinstance(table["fields"], list) or not table["fields"]):
        raise ValueError("Invalid header crosswalk table")
    seen_codes, seen_fields = set(), set()
    for position, field in enumerate(table["fields"], 1):
        if (not isinstance(field, dict) or set(field) != {"position", "code", "label", "field"}
                or type(field["position"]) is not int or field["position"] != position
                or not isinstance(field["code"], str) or not field["code"].strip()
                or not isinstance(field["label"], str) or not field["label"].strip()
                or not isinstance(field["field"], str)):
            raise ValueError(f"Invalid header crosswalk column {position}")
        name = field["field"]
        if re.fullmatch(r"[a-z_][a-z0-9_]*", name) is None or keyword.iskeyword(name):
            raise ValueError(f"Invalid canonical header in column {position}: {name!r}; ASCII snake_case is required")
        if field["code"].strip() in seen_codes:
            raise ValueError(f"Duplicate source code in header column {position}: {field['code']!r}")
        if name in seen_fields:
            raise ValueError(f"Canonical header collision in column {position}: {name!r}")
        seen_codes.add(field["code"].strip())
        seen_fields.add(name)
    if source_schema_fingerprint([f["label"] for f in table["fields"]],
                                 [f["code"] for f in table["fields"]]) != table["source_schema_sha256"]:
        raise ValueError("Header crosswalk source-schema checksum mismatch")


def parse_column_schema(data: bytes) -> dict[str, Any]:
    """Validate either retained flat columns or an explicit qualified crosswalk."""
    document = json.loads(data, object_pairs_hook=_unique_object)
    if not isinstance(document, dict) or type(document.get("schema_version")) is not int:
        raise ValueError("Invalid column schema document")
    if document["schema_version"] == 1:
        if (set(document) != {"schema_version", "columns"}
                or not isinstance(document["columns"], list)
                or any(not isinstance(c, str) or not c.strip() for c in document["columns"])):
            raise ValueError("Invalid column schema document")
        if len(document["columns"]) != len(set(document["columns"])):
            raise ValueError("Column schema contains duplicate columns")
    elif document["schema_version"] == 2:
        if (set(document) != {"schema_version", "mapping_version", "datasets"}
                or type(document["mapping_version"]) is not int or document["mapping_version"] < 1
                or not isinstance(document["datasets"], list) or not document["datasets"]):
            raise ValueError("Invalid column schema document")
        seen = set()
        for dataset in document["datasets"]:
            if (not isinstance(dataset, dict)
                    or set(dataset) != {"dataset", "edition", "prepared_input_sha256", "tables"}
                    or any(not isinstance(dataset[k], str) or not dataset[k].strip() for k in ("dataset", "edition"))
                    or not _checksum(dataset["prepared_input_sha256"])
                    or not isinstance(dataset["tables"], list) or not dataset["tables"]):
                raise ValueError("Invalid header crosswalk dataset")
            identity = dataset["dataset"], dataset["edition"]
            if identity in seen:
                raise ValueError("Duplicate header crosswalk dataset edition")
            seen.add(identity)
            tables, shared = set(), {}
            for table in dataset["tables"]:
                validate_header_table(table)
                if table["table"] in tables:
                    raise ValueError("Duplicate header crosswalk table")
                tables.add(table["table"])
                for field in table["fields"]:
                    old = shared.setdefault(field["field"], (field["code"], field["label"]))
                    if old != (field["code"], field["label"]):
                        raise ValueError("Conflicting shared crosswalk field definitions")
    else:
        raise ValueError("Invalid column schema document")
    return document


def load_column_contract(base_dir: Path, relative: str, reviewed_sha256: str) -> dict[str, str]:
    """Return verified exact resource text for binding into a build catalog."""
    data = portable_schema_path(base_dir, relative).read_bytes()
    if sha256(data).hexdigest() != reviewed_sha256:
        raise ValueError("Column schema bytes differ from the reviewed contract")
    parse_column_schema(data)
    return {"path": relative, "sha256": reviewed_sha256, "text": data.decode("utf-8")}


def crosswalk_dataset(document: Mapping[str, Any], dataset: str, edition: str) -> dict[str, Any]:
    if document.get("schema_version") == 2:
        for item in document["datasets"]:
            if item["dataset"] == dataset and item["edition"] == edition:
                return item
    raise ValueError(f"Header crosswalk unavailable for {dataset!r}, edition {edition!r}")


def schema_columns(document: Mapping[str, Any], *, dataset: str | None = None,
                   edition: str | None = None) -> list[str]:
    if document["schema_version"] == 1:
        return list(document["columns"])
    if dataset is None or edition is None:
        raise ValueError("Header crosswalk requires an explicit dataset and edition")
    selected = crosswalk_dataset(document, dataset, edition)
    return list(dict.fromkeys(field["field"] for table in selected["tables"] for field in table["fields"]))


def validate_selected_crosswalk(document: Mapping[str, Any], *, dataset: str, edition: str,
                                input_sha256: str, columns: list[str]) -> None:
    selected = crosswalk_dataset(document, dataset, edition)
    if selected["prepared_input_sha256"] != input_sha256:
        raise ValueError("Header crosswalk belongs to a different prepared input")
    if schema_columns(document, dataset=dataset, edition=edition) != columns:
        raise ValueError("Header crosswalk differs from the selected prepared schema")


def header_records(document: Mapping[str, Any], *, dataset: str, edition: str,
                   table: str | None = None) -> list[dict[str, Any]]:
    selected = crosswalk_dataset(document, dataset, edition)
    tables = [item for item in selected["tables"] if table is None or item["table"] == table]
    if not tables:
        raise ValueError(f"Header crosswalk table unavailable: {table!r}")
    return [{**field, "dataset": dataset, "edition": edition, "table": item["table"],
             "mapping_version": document["mapping_version"], "input_sha256": item["input_sha256"],
             "source_schema_sha256": item["source_schema_sha256"]}
            for item in tables for field in item["fields"]]


def header_rename_map(records: list[dict[str, Any]], *, from_: str = "code", to: str = "field",
                      columns: Iterable[str] | None = None) -> dict[str, str]:
    """Produce an ordinary rename dictionary, refusing ambiguous destinations."""
    if from_ not in {"code", "label", "field"} or to not in {"code", "label", "field"}:
        raise ValueError("Header mapping requires code, label or field")
    result: dict[str, str] = {}
    destinations = set()
    for record in records:
        source, target = record[from_], record[to]
        if source in result or target in destinations:
            raise ValueError("Ambiguous header mapping; select one table and distinct source/destination headers")
        result[source] = target
        destinations.add(target)
    if columns is not None:
        if isinstance(columns, (str, bytes)):
            raise ValueError("External headers must be a sequence of column names")
        names = list(columns)
        if any(not isinstance(name, str) for name in names) or len(names) != len(set(names)):
            raise ValueError("External headers contain invalid or duplicate names")
        renamed = [result.get(name, name) for name in names]
        if len(renamed) != len(set(renamed)):
            raise ValueError("Header mapping collides with an external destination column")
        selected = set(names)
        result = {source: target for source, target in result.items() if source in selected}
    return result


def load_column_schema(base_dir: Path, relative: str, reviewed_sha256: str, *,
                       dataset: str | None = None, edition: str | None = None) -> list[str]:
    contract = load_column_contract(base_dir, relative, reviewed_sha256)
    return schema_columns(parse_column_schema(contract["text"].encode()), dataset=dataset, edition=edition)
