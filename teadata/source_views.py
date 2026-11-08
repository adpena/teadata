"""Project catalog-bound source fields without adding them to entity metadata."""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping
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

from .dataset_metadata import ArtifactRef, DataPeriod, EditionKey
from .identifiers import canonical_source_identifiers
from .manifest import ContractError

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa
    from openpyxl import Workbook
    from pyarrow.dataset import Dataset, Expression, Scanner

    from .export_provenance import ExportProvenance
    from .release import DataRelease


def _projection(release: DataRelease, edition: EditionKey, fields: tuple[str, ...], *,
                period: DataPeriod | None = None) -> tuple[ArtifactRef, tuple[str, ...]]:
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("Select one or more distinct source fields")
    plans = release.catalog.resolve_artifacts(((edition, name) for name in fields), period=period)
    if len(plans) != 1:
        raise ContractError("Projection requires fields from one native table")
    return plans[0]["artifact"], plans[0]["columns"]


def _entity_key(artifact: ArtifactRef) -> str:
    scope = artifact.edition.scope
    key = scope + "_number"
    if scope not in {"district", "campus"} or key not in artifact.contract["key_columns"]:
        raise ContractError("Source table has no matching entity key")
    return key


def _entity_predicate(native: Dataset, entity_key: str, identifiers: tuple[str, ...]) -> Expression:
    import pyarrow as pa
    import pyarrow.compute as arrow_compute
    from pyarrow.dataset import field, scalar

    pc: Any = arrow_compute
    if not identifiers:
        return scalar(False)
    key_type = native.schema.field(entity_key).type
    if not (pa.types.is_string(key_type) or pa.types.is_large_string(key_type)):
        raise ValueError("Source identifier batches require Arrow strings")
    selected = pa.array(["'" + identifier for identifier in identifiers], type=key_type)
    width = 6 if entity_key == "district_number" else 9
    aliases = set()
    for batch in native.to_batches(columns=[entity_key], batch_size=65_536, use_threads=False):
        keys = batch.column(0)
        normalized = canonical_source_identifiers(keys, width=width)
        aliases.update(pc.unique(pc.filter(keys, pc.is_in(normalized, value_set=selected))).to_pylist())
    return field(entity_key).isin(pa.array(sorted(aliases), type=key_type))


@dataclass(frozen=True, slots=True, eq=False)
class SourceTable:
    """A lazy projection and filter of one exact published edition."""

    release: DataRelease
    edition: EditionKey
    _fields: tuple[str, ...] | None = None
    _predicate: Expression | None = None
    _queries: tuple[Callable[[pa.Schema], Expression], ...] = ()
    _period: DataPeriod | None = None
    _entity_build: tuple[str, str] | None = None
    _entity_ids: tuple[str, ...] | None = None

    @property
    def measurement_period(self) -> DataPeriod | None:
        """The resolved measurement period of an observation view."""
        return self._period

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
        from .source_predicates import prepare_predicate

        return replace(self, _queries=(*self._queries, prepare_predicate(expression, params)))

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
                              selected_fields=self._fields, predicate=self._predicate, queries=self._queries,
                              include_support=True, period=self._period, entity_build=self._entity_build,
                              entity_ids=self._entity_ids)

    def _for_entity_ids(self, identifiers: tuple[str, ...]) -> SourceTable:
        artifact, _ = _projection(self.release, self.edition, self._fields or (), period=self._period)
        _entity_key(artifact)
        return replace(self, _entity_ids=identifiers)

    def to_arrow(self) -> pa.Table:
        """Materialize the filtered projection with its keys and source support."""
        return self._prepare().to_arrow()

    def to_df(self, *, restore_original_headers: bool = False) -> pd.DataFrame:
        """Materialize the selected rows as an ordinary pandas DataFrame."""
        return self._prepare(restore_original_headers=restore_original_headers).to_df()

    def provenance(self, *, restore_original_headers: bool = False) -> ExportProvenance:
        """Describe this exact export selection without opening its data leaf."""
        selection = _plan_export(self.release, self.edition, restore_original_headers=restore_original_headers,
                                 selected_fields=self._fields, include_support=True, period=self._period)
        return selection.provenance(self.release, period=self._period, entity_build=self._entity_build,
                                    entity_ids=self._entity_ids, opaque_filter=bool(self._queries) or self._predicate is not None)

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
        artifact, columns = _projection(self.release, self.edition, fields)
        entity_key = _entity_key(artifact)
        native = self.release.analytical_dataset(artifact.identity)
        predicate = _entity_predicate(native, entity_key, (self.identifier,))
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
    rows_hint: int | None
    lineage: dict[str, Any]
    provenance: Callable[[], ExportProvenance]

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
            frame.columns = pd.Index(self.headers[0]) if len(self.headers) == 1 else pd.MultiIndex.from_arrays(self.headers)
        frame.attrs["teadata"] = self.lineage
        frame.attrs["teadata_provenance"] = self.provenance().to_dict()
        frame.attrs["teadata_export"] = {"rows": len(frame)}
        return frame


@dataclass(frozen=True, slots=True)
class _ExportSelection:
    artifact: ArtifactRef
    fields: tuple[str, ...]
    columns: tuple[str, ...]
    headers: tuple[tuple[str, ...], ...]

    def provenance(self, release: DataRelease, *, period: DataPeriod | None,
                   entity_build: tuple[str, str] | None, entity_ids: tuple[str, ...] | None,
                   opaque_filter: bool) -> ExportProvenance:
        from .export_provenance import describe_export

        if entity_ids is not None:
            _entity_key(self.artifact)
        return describe_export(release, self.artifact, self.fields, self.columns, self.headers,
                               period=period, entity_build=entity_build, entity_ids=entity_ids,
                               opaque_filter=opaque_filter)


def _plan_export(release: DataRelease, edition: EditionKey, *, restore_original_headers: bool,
                 selected_fields: tuple[str, ...] | None, include_support: bool,
                 period: DataPeriod | None) -> _ExportSelection:
    import pyarrow as pa

    from .analytical import _leaf_schema
    from .tapr import TAPR_SOURCE_KEYS, qualified_tapr_family

    if type(restore_original_headers) is not bool:
        raise ValueError("Original headers must be explicitly enabled or disabled")

    headers = release.catalog.header_records(edition.dataset, edition_key=edition) if (
        selected_fields is None or restore_original_headers) else []
    fields = tuple(row["field"] for row in headers) if headers else selected_fields or ()
    requested = fields if selected_fields is None else selected_fields
    artifact, projected = _projection(release, edition, requested, period=period)
    columns = projected if include_support else fields
    original_error = "Original headers require a complete raw TAPR table or reviewed code-only RDA district table"
    header_kinds = ()
    if restore_original_headers:
        from .rda_native import NATIVE_DATASET as rda_dataset

        header_kinds = release.catalog._source_header_kinds(edition)
        if not (edition.dataset == rda_dataset and edition.scope == edition.table == "district"
                and header_kinds == ("code",)):
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
        schema = _leaf_schema(artifact.contract)
        if not all(pa.types.is_string(schema.field(name).type) or pa.types.is_large_string(schema.field(name).type)
                   for name in columns):
            raise ContractError(original_error)
    output_headers = (columns,) if not restore_original_headers else tuple(
        tuple(row[kind] for row in headers) for kind in header_kinds)
    return _ExportSelection(artifact, requested, columns, output_headers)


def prepare_export(release: DataRelease, edition: EditionKey, *, restore_original_headers: bool,
                   selected_fields: tuple[str, ...] | None = None, predicate: Expression | None = None,
                   queries: tuple[Callable[[pa.Schema], Expression], ...] = (),
                   include_support: bool = False, period: DataPeriod | None = None,
                   entity_build: tuple[str, str] | None = None,
                   entity_ids: tuple[str, ...] | None = None) -> _TableExport:
    selection = _plan_export(release, edition, restore_original_headers=restore_original_headers,
                             selected_fields=selected_fields, include_support=include_support, period=period)
    return _open_export(release, selection, restore_original_headers=restore_original_headers,
                        predicate=predicate, queries=queries, period=period, entity_build=entity_build, entity_ids=entity_ids)


def _open_export(
    release: DataRelease, selection: _ExportSelection, *, restore_original_headers: bool = False,
    predicate: Expression | None = None, queries: tuple[Callable[[pa.Schema], Expression], ...] = (),
    period: DataPeriod | None = None, entity_build: tuple[str, str] | None = None,
    entity_ids: tuple[str, ...] | None = None,
) -> _TableExport:
    artifact = selection.artifact
    edition = artifact.edition
    opaque_filter = bool(queries) or predicate is not None
    entity_key = _entity_key(artifact) if entity_ids is not None else None
    native = release.analytical_dataset(artifact.identity)
    if entity_key is not None and entity_ids is not None:
        expression = _entity_predicate(native, entity_key, entity_ids)
        predicate = expression if predicate is None else predicate & expression
    for query in queries:
        expression = query(native.schema)
        predicate = expression if predicate is None else predicate & expression
    rows_hint = artifact.contract["rows"] if predicate is None else None
    scanner = native.scanner(columns=list(selection.columns), filter=predicate, batch_size=256, use_threads=False)
    lineage: dict[str, Any] = {"build_id": release.build_id, "edition": edition.to_dict(), "artifact_sha256": artifact.sha256,
                               "manifest_sha256": release.manifest_sha256}
    if period is not None:
        lineage["measurement_period"] = period.to_dict()
    if entity_build is not None:
        lineage.update(entity_build_id=entity_build[0], entity_manifest_sha256=entity_build[1])

    def provenance() -> ExportProvenance:
        return selection.provenance(release, period=period, entity_build=entity_build,
                                    entity_ids=entity_ids, opaque_filter=opaque_filter)

    return _TableExport(scanner, selection.headers, restore_original_headers, rows_hint, lineage, provenance)


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


# SpreadsheetML can decode literal escape tokens even in inline text cells.
_UNSAFE_EXCEL_TEXT = re.compile(r"[\x00-\x08\x0b-\x0d\x0e-\x1f\ud800-\udfff\ufffe\uffff]|_x[0-9a-fA-F]{4}_")


def validate_excel_text(value: str) -> None:
    """Refuse text the XLSX writer would truncate, normalize or encode as invalid XML."""
    if len(value) > 32_767:
        raise ValueError("Source text exceeds Excel cell limits")
    if _UNSAFE_EXCEL_TEXT.search(value):
        raise ValueError("Source text contains characters the Excel writer cannot preserve")


@contextmanager
def _excel_workbook(path_or_buf: str | Path | BinaryIO) -> Generator[Workbook, None, None]:
    from openpyxl import Workbook

    with _export_stream(path_or_buf) as stream:
        workbook = Workbook(write_only=True)
        try:
            yield workbook
            workbook.save(stream)
        except BaseException as error:
            for sheet in workbook.worksheets:
                # Closing the workbook alone leaves unsaved worksheet XML.
                writer = getattr(sheet, "_writer", None)
                if writer is None:
                    continue
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


def write_excel(table: _TableExport, path_or_buf: str | Path | BinaryIO) -> None:
    from openpyxl.cell import WriteOnlyCell

    max_rows = 1_048_576 - len(table.headers)
    row_limit_error = "Source exceeds Excel worksheet limits; use CSV for this table"
    if len(table.schema) > 16_384 or (table.rows_hint is not None and table.rows_hint > max_rows):
        raise ValueError(row_limit_error)
    provenance = table.provenance()
    with _excel_workbook(path_or_buf) as workbook:
        sheet = workbook.create_sheet("Data")

        def cells(values):
            for value in values:
                if (isinstance(value, Decimal) or type(value) is int and abs(value) >= 10**15
                        or isinstance(value, float) and (not isfinite(value) or value != float(format(value, ".15g")))):
                    value = str(value)
                if isinstance(value, str):
                    try:
                        validate_excel_text(value)
                    except ValueError as error:
                        raise ValueError(f"{error}. Use CSV for this table.") from error
                    cell = WriteOnlyCell(sheet, value=value)
                    cell.data_type = "s"
                    yield cell
                else:
                    yield value

        for row in table.headers:
            sheet.append(list(cells(row)))
        rows = 0
        for batch in table.batches():
            rows += batch.num_rows
            if rows > max_rows:
                raise ValueError(row_limit_error)
            for row in zip(*(column.to_pylist() for column in batch.columns), strict=True):
                sheet.append(list(cells(row)))
        provenance._append_sheet(workbook, rows=rows)
