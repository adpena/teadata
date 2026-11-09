"""Canonical provenance for one planned native source export."""

from __future__ import annotations

from base64 import b64encode
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import TYPE_CHECKING, Any

from .citations import citation
from .dataset_metadata import DataPeriod, EditionKey, validate_http_url
from .manifest import ContractError, canonical_json

if TYPE_CHECKING:
    from pathlib import Path
    from typing import BinaryIO

    from openpyxl import Workbook
    from openpyxl.worksheet._write_only import WriteOnlyWorksheet

    from .dataset_metadata import ArtifactRef
    from .release import DataRelease
    from .source_views import SourceTable


def _provenance_sheet(workbook: Workbook) -> WriteOnlyWorksheet:
    sheet = workbook.create_sheet("Provenance")
    sheet.freeze_panes = "A2"
    for name, width in zip("ABCDEFGH", (24, 52, 30, 23, 24, 48, 70, 55), strict=True):
        sheet.column_dimensions[name].width = width
    return sheet


def _public_url(value: str | None) -> str | None:
    try:
        validate_http_url(value)
    except ValueError:
        return None
    return value


def describe_export(release: DataRelease, artifact: ArtifactRef, fields: tuple[str, ...],
                    columns: tuple[str, ...], headers: tuple[tuple[str, ...], ...], *,
                    period: DataPeriod | None, entity_build: tuple[str, str] | None,
                    entity_ids: tuple[str, ...] | None, opaque_filter: bool,
                    query_replay: dict[str, Any] | None = None) -> ExportProvenance:
    """Project the admitted catalog, without opening the data leaf."""
    edition = artifact.edition
    artifact_id = artifact.identity
    catalog = release.catalog
    source = catalog.metadata(edition.dataset, edition=edition)
    output = set(columns)
    requested = set(fields)
    definitions = catalog._export_field_headers(artifact, columns)
    support_for = defaultdict(list)
    support = artifact.column_support
    key_columns = set(artifact.contract["key_columns"])
    for column, definition in definitions.items():
        definition.update(field=column, selected=column in requested, policy_id=source.value_policy)
    for entry in support.fields if support is not None else ():
        if entry.field not in output:
            continue
        for role, target in (("raw", entry.raw), ("status", entry.status)):
            if target in output:
                support_for[target].append({"field": entry.field, "role": role})
        for target in entry.auxiliary:
            if target in output:
                support_for[target].append({"field": entry.field, "role": "auxiliary"})
    for column in support.context if support is not None else ():
        if column not in output:
            continue
        support_for[column].append({"field": None, "role": "context"})
    qualifiers = source.source_qualifiers
    if qualifiers is not None and qualifiers.row_run_column in output:
        support_for[qualifiers.row_run_column].append({"field": None, "role": "row_run_column"})

    def column_role(name: str) -> str:
        if name in key_columns:
            return "key"
        if name in definitions:
            return "field"
        return "support" if any(item["field"] is not None for item in support_for.get(name, ())) else "context"

    projection = [
        {
            "position": index + 1,
            "column": name,
            "headers": [row[index] for row in headers],
            "role": column_role(name),
            "fields": [definitions[name]] if name in definitions else [],
            "support_for": sorted(support_for.get(name, ()), key=lambda field: (field["field"] or "", field["role"])),
        }
        for index, name in enumerate(columns)
    ]
    acquisitions = sorted(catalog.acquisitions(edition.dataset, edition=edition), key=lambda record: record["record_id"])
    receipts = [{
        "record_id": record["record_id"],
        "method": record["method"],
        "downloaded_at": record.get("downloaded_at") if record["method"] == "download" else None,
        "recorded_at": record.get("recorded_at"),
        "requested_url": _public_url(record.get("requested_url")),
        "resolved_url": _public_url(record.get("resolved_url")),
        "artifact_sha256": record["artifact"]["sha256"],
    } for record in acquisitions]
    references = []
    for reference in sorted(source.references, key=lambda reference: (reference.role, reference.reference_id)):
        item = reference.to_dict()
        item.pop("path", None)
        item["url"] = _public_url(reference.url)
        references.append(item)
    document = {
        "schema_version": 1,
        "release": {"build_id": release.build_id, "manifest_sha256": release.manifest_sha256},
        "edition": edition.to_dict(),
        "artifact": {"id": artifact_id, "sha256": artifact.sha256, "processing_receipt": artifact.processing_id},
        "selector": {
            "measurement_period": period.to_dict() if period is not None else None,
            "entity_ids": sorted(set(entity_ids)) if entity_ids is not None else None,
            "entity_release": dict(zip(("build_id", "manifest_sha256"), entity_build, strict=True)) if entity_build else None,
            "opaque_filter": opaque_filter,
        },
        "source": {
            "title": source.title, "publisher": source.publisher, "url": _public_url(source.source_url),
            "download_url": _public_url(source.download_url),
            "report_period": source.period.to_dict(), "published_at": source.published_at,
            "revised_at": source.revised_at, "publisher_citation": source.citation,
            "qualifiers": qualifiers.to_dict() if qualifiers else None,
            "policy": catalog.policy(edition.dataset, edition=edition).to_dict() if source.value_policy else None,
            "references": references,
        },
        "columns": projection,
        "edition_acquisitions": receipts,
    }
    if source.notes:
        document["source"]["notes"] = source.notes
    if query_replay is not None:
        document["selector"]["query_replay"] = query_replay
    description_sha256 = sha256(canonical_json(document).encode()).hexdigest()
    return ExportProvenance(canonical_json({**document, "description_sha256": description_sha256,
                                            "citation": citation(release).text}))


def _download_summary(acquisitions: list[dict[str, Any]]) -> str:
    days = sorted({row["downloaded_at"][:10] for row in acquisitions
                   if row["method"] == "download" and row["downloaded_at"]})
    if not days:
        return "Download date unrecorded."
    if len(days) == 1:
        summary = f"Downloaded {days[0]} UTC."
    else:
        summary = f"Downloaded {days[0]} through {days[-1]} UTC ({len(days)} observed dates)."
    if any(row["method"] != "download" or not row["downloaded_at"] for row in acquisitions):
        summary += " Some input download dates are unrecorded."
    return summary


def _publisher_data_citation(document: dict[str, Any], download_summary: str) -> str:
    source, edition = document["source"], document["edition"]
    title = source["title"] or edition["dataset"]
    segments = [f"{title} ({edition['edition']}, {edition['scope']}) [Data set]."]
    if source["publisher"]:
        segments.insert(0, f"{source['publisher']}.")
    if source["revised_at"]:
        segments.append(f"Publisher revision {source['revised_at']}.")
    if url := source["url"] or source["download_url"]:
        segments.append(url)
    segments.extend((download_summary, f"Artifact SHA-256 {document['artifact']['sha256']}."))
    return " ".join(segments)


@dataclass(frozen=True, slots=True)
class ExportProvenance:
    """A detached export description, independent of how the rows are written."""

    _json: str

    @classmethod
    def from_dict(cls, document: dict[str, Any]) -> ExportProvenance:
        """Read a detached provenance record and check its description digest."""
        if not isinstance(document, dict):
            raise ContractError("Export provenance must be a record")
        body = {key: value for key, value in document.items() if key not in {"citation", "description_sha256"}}
        if (type(body.get("schema_version")) is not int or body["schema_version"] != 1
                or sha256(canonical_json(body).encode()).hexdigest() != document.get("description_sha256")):
            raise ContractError("Export provenance version or description digest differs")
        return cls(canonical_json(document))

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh JSON-compatible representation of the export selection."""
        return json.loads(self._json)

    @property
    def restore_original_headers(self) -> bool:
        """Return the saved header mode for a recorded source query."""
        selector = self.to_dict()["selector"]
        return selector.get("query_replay", {}).get("restore_original_headers") is True

    def replay_table(self, release: DataRelease) -> SourceTable:
        """Restore a recorded selection against its exact open native release."""
        from dataclasses import replace

        from .source_predicates import restore_query

        document = self.from_dict(self.to_dict()).to_dict()
        if document.get("release") != {"build_id": release.build_id, "manifest_sha256": release.manifest_sha256}:
            raise ContractError("Replay requires the exact native release")
        selector = document.get("selector")
        if not isinstance(selector, dict) or selector.get("opaque_filter") is not False:
            raise ContractError("Export provenance contains an opaque filter")
        recipe = selector.get("query_replay")
        if (not isinstance(recipe, dict) or set(recipe) != {"version", "fields", "restore_original_headers", "queries"}
                or type(recipe["version"]) is not int or recipe["version"] != 1
                or type(recipe["restore_original_headers"]) is not bool
                or not isinstance(recipe["queries"], list) or not recipe["queries"]):
            raise ContractError("Export provenance has no supported query replay recipe")
        fields = recipe["fields"]
        identifiers = selector.get("entity_ids")
        entity = selector.get("entity_release")
        if (fields is not None and (not isinstance(fields, list) or any(type(field) is not str for field in fields))
                or identifiers is not None and (not isinstance(identifiers, list)
                                                or any(type(identifier) is not str for identifier in identifiers))
                or entity is not None and (not isinstance(entity, dict) or set(entity) != {"build_id", "manifest_sha256"}
                                           or any(type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value)
                                                  for value in entity.values()))):
            raise ContractError("Invalid recorded source selection")
        try:
            table = release.table(EditionKey.from_dict(document["edition"]))
            period = selector.get("measurement_period")
            table = replace(table, _period=DataPeriod.from_dict(period) if period is not None else None,
                            _entity_build=(entity["build_id"], entity["manifest_sha256"]) if entity else None)
            if fields is not None:
                table = table.select(*fields)
            if identifiers is not None:
                table = table._for_entity_ids(tuple(identifiers))
            for recorded in recipe["queries"]:
                expression, params = restore_query(recorded)
                table = table.query(expression, params=params, record=True)
            rebuilt = table.provenance(restore_original_headers=recipe["restore_original_headers"]).to_dict()
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("Invalid recorded source selection") from error
        if rebuilt["description_sha256"] != document["description_sha256"]:
            raise ContractError("Recorded source selection differs from the pinned release")
        # Bind the filters against the admitted schema without opening the data.
        from pyarrow import ArrowException, Table
        from pyarrow.dataset import dataset

        from .analytical import _leaf_schema

        artifact = release.catalog.artifact_ref(rebuilt["artifact"]["id"])
        schema = _leaf_schema(artifact.contract)
        empty = dataset(Table.from_batches([], schema=schema))
        try:
            for query in table._queries:
                empty.scanner(filter=query(schema))
        except (ArrowException, KeyError, TypeError, ValueError) as error:
            raise ContractError("Recorded filter does not match the pinned source schema") from error
        return table

    @property
    def publisher_data_citation(self) -> str:
        """Cite the publisher edition, exact artifact and known UTC download dates."""
        document = self.to_dict()
        return _publisher_data_citation(document, _download_summary(document["edition_acquisitions"]))

    def to_excel(self, path_or_buf: str | Path | BinaryIO) -> None:
        """Write the same Provenance worksheet as a standalone companion workbook."""
        from .source_views import _excel_workbook

        with _excel_workbook(path_or_buf) as workbook:
            self._append_sheet(workbook)

    def _append_sheet(self, workbook: Workbook, *, rows: int | None = None,
                      sheet: WriteOnlyWorksheet | None = None) -> None:
        from openpyxl.cell import WriteOnlyCell
        from openpyxl.styles import Alignment, Font, PatternFill

        from .source_views import validate_excel_text

        if sheet is None:
            sheet = _provenance_sheet(workbook)
        header_fill = PatternFill("solid", fgColor="132840")
        header_font = Font(bold=True, color="FFFFFF")
        wrapped = Alignment(vertical="top", wrap_text=True)

        def append(values, *, heading=False):
            cells = []
            for value in values:
                if isinstance(value, str):
                    try:
                        validate_excel_text(value)
                    except ValueError:
                        value = "See canonical record below"
                    cell = WriteOnlyCell(sheet, value=value)
                    cell.data_type = "s"
                    cell.alignment = wrapped
                    if heading:
                        cell.fill = header_fill
                        cell.font = header_font
                    cells.append(cell)
                else:
                    cells.append(value)
            sheet.append(cells)

        document = self.to_dict()
        source, edition = document["source"], document["edition"]
        append(("TEA Data export provenance",), heading=True)
        append(("Dataset", source["title"], edition["dataset"]))
        append(("Publisher", source["publisher"], source["url"]))
        if source["download_url"]:
            append(("Publisher download", source["download_url"]))
        append(("Report edition", edition["edition"], DataPeriod.from_dict(source["report_period"]).label))
        append(("Scope", edition["scope"], edition["grain"]))
        append(("Release build", document["release"]["build_id"]))
        append(("Manifest SHA-256", document["release"]["manifest_sha256"]))
        append(("Artifact SHA-256", document["artifact"]["sha256"]))
        append(("Processing receipt", document["artifact"]["processing_receipt"]))
        append(("Description SHA-256", document["description_sha256"]))
        if rows is not None:
            append(("Rows exported", rows))
        append(("Citation", document["citation"]))
        download_summary = _download_summary(document["edition_acquisitions"])
        append(("Publisher data citation", _publisher_data_citation(document, download_summary)))
        append(("Input download dates", download_summary))
        if source["publisher_citation"]:
            append(("Publisher citation", source["publisher_citation"]))
        if source.get("notes"):
            append(("Source notes", source["notes"]))
        if source["published_at"] or source["revised_at"]:
            append(("Published / revised", source["published_at"], source["revised_at"]))
        if source["qualifiers"]:
            for name, value in source["qualifiers"].items():
                append((name.replace("_", " ").capitalize(), value))
        if source["policy"]:
            append(("Value policy", source["policy"]["policy_id"]))
        if document["selector"]["opaque_filter"]:
            append(("Filter", "Selection includes a filter that cannot be replayed from this record."))
        elif recipe := document["selector"].get("query_replay"):
            for query in recipe["queries"]:
                append(("Recorded filter", query["expression"]))
        append(("Column", "Export header", "Publisher code", "Measurement period", "Unit", "Binding", "Publisher definition"), heading=True)
        for column in document["columns"]:
            header = " / ".join(column["headers"])
            for field in column["fields"] or [None]:
                roles = [column["role"]]
                if field:
                    roles.append("selected" if field["selected"] else "retained")
                roles.extend(item["role"].replace("_", " ") for item in column["support_for"]
                             if item["role"] != column["role"])
                binding = ", ".join(roles)
                if field:
                    binding = f'{field["field"]} ({binding})'
                append((column["position"], header, field["code"] if field else None,
                        DataPeriod.from_dict(field["period"]).label if field else None, field["unit"] if field else None,
                        binding, field["publisher_description"] if field else None))
        append(("Edition acquisitions", "Receipt", "Method", "Downloaded at", "Recorded at",
                "Requested URL", "Resolved URL", "Artifact SHA-256"), heading=True)
        for row in document["edition_acquisitions"]:
            append(("", row["record_id"], row["method"], row["downloaded_at"], row["recorded_at"],
                    row["requested_url"], row["resolved_url"], row["artifact_sha256"]))
        append(("References", "Title", "Role", "Review status", "", "Source URL"), heading=True)
        for row in source["references"]:
            append((row["reference_id"], row.get("title"), row["role"], row["status"], None, row["url"]))
        append(("Canonical record", "Part", "Parts", "Base64-encoded UTF-8 JSON"), heading=True)
        encoded = b64encode(self._json.encode()).decode("ascii")
        parts = range(0, len(encoded), 12_000)
        for number, offset in enumerate(parts, 1):
            append(("Canonical record", number, len(parts), encoded[offset:offset + 12_000]))
