"""Project catalog-bound source fields without adding them to entity metadata."""

from __future__ import annotations

import csv
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from io import StringIO
from math import isfinite
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, BinaryIO, Generator, Iterator

from .dataset_metadata import ArtifactRef, EditionKey
from .identifiers import canonical_source_identifiers
from .manifest import ContractError

if TYPE_CHECKING:
    import pyarrow as pa
    from pyarrow.dataset import Expression

    from .release import DataRelease


def _projection(release: DataRelease, edition: EditionKey, fields: tuple[str, ...]) -> tuple[ArtifactRef, tuple[str, ...]]:
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("Select one or more distinct source fields")
    plans = release.catalog.resolve_artifacts((edition, name) for name in fields)
    if len(plans) != 1:
        raise ContractError("Projection requires fields from one native table")
    return plans[0]["artifact"], plans[0]["columns"]


@dataclass(frozen=True, slots=True)
class SourceFields:
    release: DataRelease
    edition: EditionKey
    identifier: str

    def variables(self, pattern: str | re.Pattern[str] | None = None) -> tuple[dict[str, Any], ...]:
        """Discover fields in this exact source, optionally by regex on their names."""
        match = re.compile(pattern) if pattern is not None else None
        return tuple(row for row in self.release.catalog.variables(edition=self.edition)
                     if match is None or match.search(row["field"]))

    def select(self, *fields: str, where: Expression | None = None) -> pa.Table:
        """Project selected fields, native keys and their available source support."""
        import pyarrow.compute as arrow_compute
        from pyarrow.dataset import field

        pc: Any = arrow_compute
        artifact, columns = _projection(self.release, self.edition, fields)
        entity_key = self.edition.scope + "_number"
        if entity_key not in artifact.contract["key_columns"]:
            raise ContractError("Source table has no matching entity key")
        native = self.release.analytical_dataset(artifact.identity)
        keys = native.to_table(columns=[entity_key])[entity_key]
        width = 6 if self.edition.scope == "district" else 9
        normalized = canonical_source_identifiers(keys, width=width)
        aliases = pc.unique(pc.filter(keys, pc.equal(normalized, "'" + self.identifier)))
        predicate = field(entity_key).isin(aliases)
        if where is not None:
            predicate = predicate & where
        return native.to_table(columns=list(columns), filter=predicate)

    def __getitem__(self, name: str):
        metadata = self.release.catalog.field_metadata(name, scope=self.edition.scope, edition=self.edition)
        if metadata is None:
            raise KeyError(name)
        table = self.select(name)
        if table.num_rows != 1:
            raise ContractError("Scalar access requires exactly one published row for this entity")
        return table[metadata["column"]][0].as_py()

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


@dataclass(frozen=True, slots=True)
class EntitySources:
    release: DataRelease
    scope: str
    identifier: str
    edition: str

    def datasets(self) -> tuple[EditionKey, ...]:
        """List exact source identities using only the release index."""
        return tuple(EditionKey.from_dict({name: row[name] for name in EditionKey.__dataclass_fields__})
                     for row in self.release.catalog.editions(scope=self.scope, include_metadata=False)
                     if row["edition"] == self.edition)

    def dataset(self, name: str | EditionKey) -> SourceFields:
        choices = [key for key in self.datasets() if key == name or key.dataset == name]
        if not choices:
            raise KeyError(name)
        if len(choices) != 1:
            raise ContractError("Source name is ambiguous; select an exact EditionKey")
        return SourceFields(self.release, choices[0], self.identifier)

    def __getitem__(self, name: str) -> SourceFields:
        return self.dataset(name)

    def __getattr__(self, name: str) -> SourceFields:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self.dataset(name)
        except KeyError as error:
            raise AttributeError(name) from error


@dataclass(frozen=True, slots=True)
class _TableExport:
    native: Any
    schema: pa.Schema
    headers: tuple[tuple[str, ...], ...]
    restore_original_headers: bool
    rows: int

    def batches(self) -> Iterator[pa.RecordBatch]:
        for batch in self.native.to_batches(columns=self.schema.names, batch_size=256, use_threads=False):
            if self.restore_original_headers and any(column.null_count for column in batch.columns):
                raise ContractError("A raw TAPR export cannot substitute blanks for null values")
            yield batch


def prepare_export(release: DataRelease, edition: EditionKey, *, restore_original_headers: bool) -> _TableExport:
    import pyarrow as pa
    from .tapr import TAPR_SOURCE_KEYS, qualified_tapr_family

    if type(restore_original_headers) is not bool:
        raise ValueError("Original headers must be explicitly enabled or disabled")

    headers = release.catalog.header_records(edition.dataset, edition_key=edition)
    fields = tuple(row["field"] for row in headers)
    artifact, _ = _projection(release, edition, fields)
    original_error = "Original label and code headers require a complete raw TAPR table"
    if restore_original_headers:
        try:
            qualified_tapr_family(edition.dataset, TAPR_SOURCE_KEYS.get(edition.scope))
        except ValueError as error:
            raise ContractError(original_error) from error
        if artifact.column_support is not None or fields != artifact.public_columns:
            raise ContractError(original_error)
    native = release.analytical_dataset(artifact.identity)
    schema = pa.schema([native.schema.field(name) for name in fields])
    rows = (fields,)
    if restore_original_headers:
        if not all(pa.types.is_string(field.type) or pa.types.is_large_string(field.type) for field in schema):
            raise ContractError(original_error)
        rows = tuple(tuple(row[kind] for row in headers) for kind in ("label", "code"))
    return _TableExport(native, schema, rows, restore_original_headers, artifact.contract["rows"])


def write_csv(table: _TableExport, path_or_buf: str | Path | BinaryIO) -> None:
    from pyarrow.csv import CSVWriter, WriteOptions

    with _export_stream(path_or_buf) as stream:
        if table.restore_original_headers:
            labels = StringIO(newline="")
            csv.writer(labels, lineterminator="\r\n").writerows(table.headers)
            stream.write(labels.getvalue().encode("utf-8"))
        with CSVWriter(stream, table.schema, write_options=WriteOptions(include_header=not table.restore_original_headers)) as writer:
            for batch in table.batches():
                writer.write_batch(batch)


@contextmanager
def _export_stream(path_or_buf: str | Path | BinaryIO) -> Generator[BinaryIO]:
    if not isinstance(path_or_buf, (str, Path)):
        yield path_or_buf
        return
    path = Path(path_or_buf)
    if os.path.lexists(path):
        raise FileExistsError(path)
    with TemporaryDirectory(dir=path.parent, prefix=".teadata-export-") as directory:
        staged = Path(directory) / "output"
        with staged.open("xb") as stream:
            yield stream
        os.link(staged, path)


def write_excel(table: _TableExport, path_or_buf: str | Path | BinaryIO) -> None:
    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell

    if len(table.schema) > 16_384 or table.rows + len(table.headers) > 1_048_576:
        raise ValueError("Source exceeds Excel worksheet limits; use CSV for this table")
    with _export_stream(path_or_buf) as stream:
        workbook = Workbook(write_only=True)
        sheet = workbook.create_sheet("Data")

        def cells(values):
            for value in values:
                if (isinstance(value, Decimal) or type(value) is int and abs(value) >= 10**15
                        or isinstance(value, float) and (not isfinite(value) or value != float(format(value, ".15g")))):
                    value = str(value)
                if isinstance(value, str):
                    if len(value) > 32_767:
                        raise ValueError("Source text exceeds Excel cell limits; use CSV for this table")
                    cell = WriteOnlyCell(sheet, value=value)
                    cell.data_type = "s"
                    yield cell
                else:
                    yield value

        try:
            for row in table.headers:
                sheet.append(list(cells(row)))
            for batch in table.batches():
                for row in zip(*(column.to_pylist() for column in batch.columns), strict=True):
                    sheet.append(list(cells(row)))
            workbook.save(stream)
        except BaseException as error:
            # Closing the workbook alone does not remove an unsaved sheet's XML.
            writer = getattr(sheet, "_writer", None)
            if writer is not None:
                try:
                    if not sheet.closed:
                        sheet.close()
                except Exception as cleanup_error:
                    error.add_note(f"Closing the Excel worksheet failed: {cleanup_error}")
                finally:
                    try:
                        try:
                            writer.close()
                        finally:
                            if os.path.exists(writer.out):
                                writer.cleanup()
                    except Exception as cleanup_error:
                        error.add_note(f"Removing the Excel scratch file failed: {cleanup_error}")
            raise
        finally:
            workbook.close()
