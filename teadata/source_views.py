"""Project catalog-bound source fields without adding them to entity metadata."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from decimal import Decimal
from io import StringIO
from math import isfinite
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, BinaryIO, Generator, Iterator, overload

from .dataset_metadata import ArtifactRef, EditionKey
from .identifiers import canonical_source_identifiers
from .manifest import ContractError

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa
    from pyarrow.dataset import Expression, Scanner

    from .release import DataRelease


def _projection(release: DataRelease, edition: EditionKey, fields: tuple[str, ...]) -> tuple[ArtifactRef, tuple[str, ...]]:
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("Select one or more distinct source fields")
    plans = release.catalog.resolve_artifacts((edition, name) for name in fields)
    if len(plans) != 1:
        raise ContractError("Projection requires fields from one native table")
    return plans[0]["artifact"], plans[0]["columns"]


@dataclass(frozen=True, slots=True, eq=False)
class SourceTable:
    """A lazy projection and filter of one exact published edition."""

    release: DataRelease
    edition: EditionKey
    _fields: tuple[str, ...] | None = None
    _predicate: Expression | None = None

    def select(self, *fields: str) -> SourceTable:
        """Select distinct public fields and retain their key and source columns."""
        if not fields or any(not isinstance(name, str) or not name for name in fields) or len(set(fields)) != len(fields):
            raise ValueError("Select one or more distinct source fields")
        return replace(self, _fields=fields)

    def where(self, predicate: Expression) -> SourceTable:
        """Narrow rows with an Arrow expression evaluated against the selected leaf."""
        from pyarrow.dataset import Expression

        if not isinstance(predicate, Expression):
            raise TypeError("Filter requires an Arrow expression")
        return replace(self, _predicate=predicate if self._predicate is None else self._predicate & predicate)

    def query(self, expression: str, *, params: Mapping[str, object] | None = None) -> SourceTable:
        """Add a bounded query string with explicit parameters to this lazy scan."""
        from .source_predicates import compile_predicate

        return self.where(compile_predicate(expression, params))

    @overload
    def __getitem__(self, key: str) -> Expression: ...

    @overload
    def __getitem__(self, key: list[str] | tuple[str, ...] | Expression) -> SourceTable: ...

    def __getitem__(self, key: str | list[str] | tuple[str, ...] | Expression) -> Expression | SourceTable:
        from pyarrow.dataset import field

        if isinstance(key, str):
            return field(key)
        if isinstance(key, (list, tuple)):
            return self.select(*key)
        return self.where(key)

    def _prepare(self, *, restore_original_headers: bool = False) -> _TableExport:
        return prepare_export(self.release, self.edition, restore_original_headers=restore_original_headers,
                              selected_fields=self._fields, predicate=self._predicate, include_support=True)

    def to_arrow(self) -> pa.Table:
        """Materialize the filtered projection with its keys and source support."""
        return self._prepare().to_arrow()

    def to_df(self, *, restore_original_headers: bool = False) -> pd.DataFrame:
        """Materialize the selected rows as an ordinary pandas DataFrame."""
        return self._prepare(restore_original_headers=restore_original_headers).to_df()

    def to_csv(self, path_or_buf: str | Path | BinaryIO, *, restore_original_headers: bool = False) -> None:
        """Stream the selected rows to CSV."""
        write_csv(self._prepare(restore_original_headers=restore_original_headers), path_or_buf)

    def to_excel(self, path_or_buf: str | Path | BinaryIO, *, restore_original_headers: bool = False) -> None:
        """Stream the selected rows to XLSX."""
        write_excel(self._prepare(restore_original_headers=restore_original_headers), path_or_buf)


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
    scanner: Scanner
    headers: tuple[tuple[str, ...], ...]
    restore_original_headers: bool
    rows: int
    lineage: dict[str, Any]

    @property
    def schema(self) -> pa.Schema:
        return self.scanner.projected_schema

    def batches(self) -> Iterator[pa.RecordBatch]:
        for batch in self.scanner.to_batches():
            if self.restore_original_headers and any(column.null_count for column in batch.columns):
                raise ContractError("A raw TAPR export cannot substitute blanks for null values")
            yield batch

    def to_arrow(self) -> pa.Table:
        import pyarrow as pa

        return pa.Table.from_batches(self.batches(), schema=self.schema)

    def to_df(self) -> pd.DataFrame:
        import pandas as pd

        frame = self.to_arrow().to_pandas(types_mapper=pd.ArrowDtype)
        if self.restore_original_headers:
            frame.columns = pd.MultiIndex.from_arrays(self.headers)
        frame.attrs["teadata"] = self.lineage
        return frame


def prepare_export(release: DataRelease, edition: EditionKey, *, restore_original_headers: bool,
                   selected_fields: tuple[str, ...] | None = None, predicate: Expression | None = None,
                   include_support: bool = False) -> _TableExport:
    import pyarrow as pa
    from .tapr import TAPR_SOURCE_KEYS, qualified_tapr_family

    if type(restore_original_headers) is not bool:
        raise ValueError("Original headers must be explicitly enabled or disabled")

    headers = release.catalog.header_records(edition.dataset, edition_key=edition) if (
        selected_fields is None or restore_original_headers) else []
    fields = tuple(row["field"] for row in headers) if headers else selected_fields or ()
    artifact, projected = _projection(release, edition, fields if selected_fields is None else selected_fields)
    columns = projected if include_support else fields
    original_error = "Original label and code headers require a complete raw TAPR table"
    if restore_original_headers:
        try:
            qualified_tapr_family(edition.dataset, TAPR_SOURCE_KEYS.get(edition.scope))
        except ValueError as error:
            raise ContractError(original_error) from error
        if artifact.column_support is not None or fields != artifact.public_columns:
            raise ContractError(original_error)
        if selected_fields is not None:
            selected = set(columns)
            headers = [row for row in headers if row["field"] in selected]
            columns = tuple(row["field"] for row in headers)
            if set(columns) != selected:
                raise ContractError(original_error)
        else:
            columns = fields
    native = release.analytical_dataset(artifact.identity)
    scanner = native.scanner(columns=list(columns), filter=predicate, batch_size=256, use_threads=False)
    rows = (columns,)
    if restore_original_headers:
        if not all(pa.types.is_string(field.type) or pa.types.is_large_string(field.type) for field in scanner.projected_schema):
            raise ContractError(original_error)
        rows = tuple(tuple(row[kind] for row in headers) for kind in ("label", "code"))
    lineage = {"build_id": release.build_id, "edition": edition.to_dict(), "artifact_sha256": artifact.sha256,
               "manifest_sha256": release.manifest_sha256}
    return _TableExport(scanner, rows, restore_original_headers, artifact.contract["rows"], lineage)


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
def _export_stream(path_or_buf: str | Path | BinaryIO) -> Generator[BinaryIO, None, None]:
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

    max_rows = 1_048_576 - len(table.headers)
    if len(table.schema) > 16_384 or (table.rows > max_rows and table.scanner.count_rows() > max_rows):
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
