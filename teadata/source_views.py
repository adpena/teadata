"""Project catalog-bound source fields without adding them to entity metadata."""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from decimal import Decimal
from io import StringIO
import json
from math import isfinite
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, BinaryIO, Generator, Iterator, overload

from .dataset_metadata import ArtifactRef, DataPeriod, EditionKey
from .identifiers import canonical_source_identifiers
from .manifest import ContractError, canonical_json

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa
    from openpyxl import Workbook
    from pyarrow.dataset import Dataset, Expression, Scanner

    from .export_provenance import ExportProvenance
    from .release import DataRelease
    from .source_predicates import PreparedPredicate


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
    _queries: tuple[PreparedPredicate, ...] = ()
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

    def query(self, expression: str, *, params: Mapping[str, object] | None = None,
              record: bool = False) -> SourceTable:
        """Add a bounded query string with explicit parameters to this lazy scan."""
        from .source_predicates import prepare_predicate

        return replace(self, _queries=(*self._queries, prepare_predicate(expression, params, record=record)))

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
                                    entity_ids=self._entity_ids, opaque_filter=self._predicate is not None,
                                    queries=self._queries)

    def to_csv(self, path_or_buf: str | Path | BinaryIO, *, restore_original_headers: bool = False) -> None:
        """Stream the selected rows to CSV."""
        write_csv(self._prepare(restore_original_headers=restore_original_headers), path_or_buf)

    def to_excel(self, path_or_buf: str | Path | BinaryIO, *, restore_original_headers: bool = False) -> None:
        """Stream the selected rows to XLSX."""
        write_excel(self._prepare(restore_original_headers=restore_original_headers), path_or_buf)


@dataclass(frozen=True, slots=True, eq=False)
class PartitionedSourceTable:
    """One measurement period and campus cohort across exact source partitions."""

    partitions: tuple[SourceTable, ...]

    def __post_init__(self) -> None:
        if len(self.partitions) < 2 or self.partitions[0].measurement_period is None:
            raise ValueError("A partitioned observation requires at least two pinned partitions and a measurement period")
        first = self.partitions[0]
        if len(set(self.editions)) != len(self.partitions) or any(
            part.release.manifest_sha256 != first.release.manifest_sha256
            or part.measurement_period != first.measurement_period
            or part._entity_build != first._entity_build
            or part.edition.scope != "campus" or not part._entity_ids
            for part in self.partitions
        ):
            raise ValueError("A partitioned observation requires distinct campus partitions from one release and period")

    @property
    def editions(self) -> tuple[EditionKey, ...]:
        return tuple(part.edition for part in self.partitions)

    @property
    def measurement_period(self) -> DataPeriod:
        period = self.partitions[0].measurement_period
        assert period is not None
        return period

    def select(self, *fields: str) -> PartitionedSourceTable:
        return PartitionedSourceTable(tuple(part.select(*fields) for part in self.partitions))

    def where(self, predicate: Expression) -> PartitionedSourceTable:
        return PartitionedSourceTable(tuple(part.where(predicate) for part in self.partitions))

    def query(self, expression: str, *, params: Mapping[str, object] | None = None,
              record: bool = False) -> PartitionedSourceTable:
        from .source_predicates import prepare_predicate

        query = prepare_predicate(expression, params, record=record)
        return PartitionedSourceTable(tuple(replace(part, _queries=(*part._queries, query)) for part in self.partitions))

    @overload
    def __getitem__(self, key: str) -> Expression: ...

    @overload
    def __getitem__(self, key: list[str] | tuple[str, ...] | Expression) -> PartitionedSourceTable: ...

    def __getitem__(self, key: str | list[str] | tuple[str, ...] | Expression) -> Expression | PartitionedSourceTable:
        from pyarrow.dataset import field

        if isinstance(key, str):
            return field(key)
        if isinstance(key, (list, tuple)):
            return self.select(*key)
        return self.where(key)

    def _prepare(self, *, restore_original_headers: bool = False) -> _TableExport:
        import pyarrow as pa
        from pyarrow.dataset import dataset

        from .analytical import _leaf_schema

        plans = tuple(_plan_export(part.release, part.edition, restore_original_headers=restore_original_headers,
                                   selected_fields=part._fields, include_support=True, period=part.measurement_period)
                      for part in self.partitions)
        first = plans[0]
        native = _leaf_schema(first.artifact.contract)
        schema = pa.schema([native.field(name) for name in first.columns])
        first_support = first.artifact.column_support
        for part, plan in zip(self.partitions, plans, strict=True):
            native = _leaf_schema(plan.artifact.contract)
            support = plan.artifact.column_support
            if (plan.fields != first.fields or plan.headers != first.headers or plan.columns != first.columns
                    or (support is None) != (first_support is None)
                    or (support is not None and first_support is not None and any(
                        support.projection(field) != first_support.projection(field) for field in first.fields))
                    or not schema.equals(pa.schema([native.field(name) for name in plan.columns]), check_metadata=True)):
                raise ContractError("Campus partitions have incompatible fields or source types. Read their exact editions separately.")
            if part._predicate is not None or part._queries:
                try:
                    empty = dataset(pa.Table.from_batches([], schema=native))
                    empty.scanner(filter=_bind_queries(part._predicate, part._queries, native))
                except (pa.ArrowException, KeyError, TypeError, ValueError) as error:
                    raise ContractError(f"Filter does not match campus partition {part.edition.revision_sha256[:12]}: {error}") from error
        scanner = _PartitionScan(schema, tuple(zip(self.partitions, plans, strict=True)))

        def provenance() -> PartitionedExportProvenance:
            return PartitionedExportProvenance(self.measurement_period, tuple(
                plan.provenance(part.release, period=part.measurement_period, entity_build=part._entity_build,
                                entity_ids=part._entity_ids, opaque_filter=part._predicate is not None, queries=part._queries)
                for part, plan in scanner.partitions
            ))

        return _TableExport(scanner, first.headers, restore_original_headers, None,
                            {"measurement_period": self.measurement_period.to_dict(),
                             "editions": [key.to_dict() for key in self.editions]}, provenance)

    def to_arrow(self) -> pa.Table:
        return self._prepare().to_arrow()

    def to_df(self, *, restore_original_headers: bool = False) -> pd.DataFrame:
        return self._prepare(restore_original_headers=restore_original_headers).to_df()

    def provenance(self, *, restore_original_headers: bool = False) -> PartitionedExportProvenance:
        result = self._prepare(restore_original_headers=restore_original_headers).provenance()
        assert isinstance(result, PartitionedExportProvenance)
        return result

    def to_csv(self, path_or_buf: str | Path | BinaryIO, *, restore_original_headers: bool = False) -> None:
        write_csv(self._prepare(restore_original_headers=restore_original_headers), path_or_buf)

    def to_excel(self, path_or_buf: str | Path | BinaryIO, *, restore_original_headers: bool = False) -> None:
        write_excel(self._prepare(restore_original_headers=restore_original_headers), path_or_buf)


@dataclass(frozen=True, slots=True)
class PartitionedExportProvenance:
    """Keep each partition's canonical export record and its selected entity IDs."""

    measurement_period: DataPeriod
    partitions: tuple[ExportProvenance, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "operation": "partitioned_source_observations",
                "measurement_period": self.measurement_period.to_dict(),
                "partitions": [part.to_dict() for part in self.partitions]}

    def to_excel(self, path_or_buf: str | Path | BinaryIO) -> None:
        with _excel_workbook(path_or_buf) as workbook:
            self._append_sheet(workbook)

    def _append_sheet(self, workbook: Workbook, *, rows: int | None = None) -> None:
        from .export_provenance import _provenance_sheet

        sheet = _provenance_sheet(workbook)
        sheet.append(("TEA Data partitioned export provenance", self.measurement_period.label))
        if rows is not None:
            sheet.append(("Total rows exported", rows))
        for number, partition in enumerate(self.partitions, 1):
            sheet.append(())
            sheet.append(("Source partition", number))
            partition._append_sheet(workbook, sheet=sheet)


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
        table = SourceTable(self.release, self.edition, _entity_ids=(self.identifier,)).select(*fields)
        if where is not None:
            table = table.where(where)
        return table.to_arrow()

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
        if attachments := self.release.catalog.native_sources:
            return tuple(key for source in attachments for key in source.editions
                         if key.scope == self.scope and key.edition == self.edition)
        return tuple(EditionKey.from_dict({name: row[name] for name in EditionKey.__dataclass_fields__})
                     for row in self.release.catalog.editions(scope=self.scope, include_metadata=False)
                     if row["edition"] == self.edition)

    def dataset(self, name: str | EditionKey) -> SourceFields:
        choices = [key for key in self.datasets() if key == name or key.dataset == name]
        if not choices:
            raise KeyError(name)
        if len(choices) != 1:
            raise ContractError("Source name is ambiguous; select an exact EditionKey")
        (key,) = choices
        return SourceFields(self.release.table(key).release, key, self.identifier)

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
    scanner: Scanner | _PartitionScan
    headers: tuple[tuple[str, ...], ...]
    restore_original_headers: bool
    rows_hint: int | None
    lineage: dict[str, Any]
    provenance: Callable[[], ExportProvenance | PartitionedExportProvenance]

    @property
    def schema(self) -> pa.Schema:
        return self.scanner.projected_schema

    def batches(self) -> Iterator[pa.RecordBatch]:
        for batch in self.scanner.to_batches():
            if self.restore_original_headers and any(column.null_count for column in batch.columns):
                raise ContractError("A publisher-header export cannot substitute blanks for null values")
            yield batch

    def to_arrow(self) -> pa.Table:
        import pyarrow as pa

        schema = self.schema
        if isinstance(self.scanner, _PartitionScan):
            schema = schema.with_metadata({b"teadata_provenance": canonical_json(self.provenance().to_dict()).encode()})
        return pa.Table.from_batches(self.batches(), schema=schema)

    def to_df(self) -> pd.DataFrame:
        import pandas as pd

        table = self.to_arrow()
        frame = table.to_pandas(types_mapper=pd.ArrowDtype)
        if self.restore_original_headers:
            frame.columns = pd.Index(self.headers[0]) if len(self.headers) == 1 else pd.MultiIndex.from_arrays(self.headers)
        frame.attrs["teadata"] = self.lineage
        if isinstance(self.scanner, _PartitionScan):
            frame.attrs["teadata_provenance"] = json.loads((table.schema.metadata or {})[b"teadata_provenance"])
        else:
            frame.attrs["teadata_provenance"] = self.provenance().to_dict()
        frame.attrs["teadata_export"] = {"rows": len(frame)}
        return frame


@dataclass(frozen=True, slots=True)
class _ExportSelection:
    artifact: ArtifactRef
    fields: tuple[str, ...]
    columns: tuple[str, ...]
    headers: tuple[tuple[str, ...], ...]
    selected_fields: tuple[str, ...] | None
    restore_original_headers: bool
    publisher_columns: tuple[str, ...] | None = None

    def provenance(self, release: DataRelease, *, period: DataPeriod | None,
                   entity_build: tuple[str, str] | None, entity_ids: tuple[str, ...] | None,
                   opaque_filter: bool, queries: tuple[PreparedPredicate, ...] = ()) -> ExportProvenance:
        from .export_provenance import describe_export

        if entity_ids is not None:
            _entity_key(self.artifact)
        records = [query.recipe_json for query in queries]
        opaque_filter = opaque_filter or any(record is None for record in records)
        replay = None
        if records and not opaque_filter:
            replay = {
                "version": 1, "fields": self.selected_fields,
                "restore_original_headers": self.restore_original_headers,
                "queries": [json.loads(record) for record in records if record is not None],
            }
        return describe_export(release, self.artifact, self.fields, self.columns, self.headers,
                               period=period, entity_build=entity_build, entity_ids=entity_ids,
                               opaque_filter=opaque_filter, query_replay=replay,
                               publisher_columns=self.publisher_columns)


@dataclass(frozen=True, slots=True)
class _PartitionScan:
    projected_schema: pa.Schema
    partitions: tuple[tuple[SourceTable, _ExportSelection], ...]

    def to_batches(self) -> Iterator[pa.RecordBatch]:
        for part, plan in self.partitions:
            export = _open_export(part.release, plan, restore_original_headers=plan.restore_original_headers,
                                  predicate=part._predicate, queries=part._queries, period=part.measurement_period,
                                  entity_build=part._entity_build, entity_ids=part._entity_ids)
            for batch in export.batches():
                yield batch.replace_schema_metadata(None)


def _plan_export(release: DataRelease, edition: EditionKey, *, restore_original_headers: bool,
                 selected_fields: tuple[str, ...] | None, include_support: bool,
                 period: DataPeriod | None) -> _ExportSelection:
    import pyarrow as pa

    if type(restore_original_headers) is not bool:
        raise ValueError("Original headers must be explicitly enabled or disabled")

    headers = release.catalog.header_records(edition.dataset, edition_key=edition) if (
        selected_fields is None or restore_original_headers) else []
    fields = tuple(row["field"] for row in headers) if headers else selected_fields or ()
    requested = fields if selected_fields is None else selected_fields
    artifact, projected = _projection(release, edition, requested, period=period)
    columns = projected if include_support else fields
    original_error = ("Original headers require a complete raw TAPR table or reviewed code-only RDA district table"
                      " or reviewed PEIMS course or College Credit table")
    header_kinds = ()
    publisher_columns = None
    if restore_original_headers:
        from .analytical import _leaf_schema
        from .peims_college_credit import DATASET as credit_dataset
        from .peims_course_sources import DATASET as course_dataset
        from .rda_native import NATIVE_DATASET as rda_dataset
        from .tapr import TAPR_SOURCE_KEYS, qualified_tapr_family

        if edition.dataset in {course_dataset, credit_dataset}:
            scope = edition.scope
            selected = set(projected)
            table = release.catalog._source_header_table(edition)
            if edition.dataset == course_dataset:
                from .peims_course_sources import CSV_HEADERS, course_column_support, matches_course_header
                from .peims_courses import course_key_columns

                valid = scope in CSV_HEADERS and matches_course_header(table, scope)
                support, keys = course_column_support(), course_key_columns(scope) if valid else []
                family = "course"
            else:
                from .peims_college_credit import (SCOPES, SERVICE_FIELDS, YEARS, college_credit_keys,
                    college_credit_support, matches_college_credit_header)

                family = "College Credit"
                if scope not in SCOPES or edition.edition not in tuple(map(str, YEARS)):
                    raise ContractError("Original PEIMS College Credit headers require a reviewed year and scope")
                year = int(edition.edition)
                valid = matches_college_credit_header(table, year, scope)
                support, keys = college_credit_support(year, scope), college_credit_keys(scope)
                selected.update(SERVICE_FIELDS)
            if (scope != edition.table or not valid
                    or fields != artifact.public_columns or artifact.contract.get("column_support") != support
                    or artifact.contract["key_columns"] != keys):
                raise ContractError(f"Original PEIMS {family} headers require the reviewed positional and raw column contract")
            headers = [row for row in headers if row["field"] in selected]
            publisher_columns = tuple(row["field"] for row in headers)
            raw = {field: entry["raw"] for field, entry in support["fields"].items()}
            columns = tuple(raw.get(field, field) for field in publisher_columns)
            header_kinds = ("code",)
        else:
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
    return _ExportSelection(artifact, requested, columns, output_headers, selected_fields,
                            restore_original_headers, publisher_columns)


def prepare_export(release: DataRelease, edition: EditionKey, *, restore_original_headers: bool,
                   selected_fields: tuple[str, ...] | None = None, predicate: Expression | None = None,
                   queries: tuple[PreparedPredicate, ...] = (),
                   include_support: bool = False, period: DataPeriod | None = None,
                   entity_build: tuple[str, str] | None = None,
                   entity_ids: tuple[str, ...] | None = None) -> _TableExport:
    selection = _plan_export(release, edition, restore_original_headers=restore_original_headers,
                             selected_fields=selected_fields, include_support=include_support, period=period)
    return _open_export(release, selection, restore_original_headers=restore_original_headers,
                        predicate=predicate, queries=queries, period=period, entity_build=entity_build, entity_ids=entity_ids)


def _open_export(
    release: DataRelease, selection: _ExportSelection, *, restore_original_headers: bool = False,
    predicate: Expression | None = None, queries: tuple[PreparedPredicate, ...] = (),
    period: DataPeriod | None = None, entity_build: tuple[str, str] | None = None,
    entity_ids: tuple[str, ...] | None = None,
) -> _TableExport:
    artifact = selection.artifact
    edition = artifact.edition
    opaque_filter = predicate is not None
    entity_key = _entity_key(artifact) if entity_ids is not None else None
    native = release.analytical_dataset(artifact.identity)
    if entity_key is not None and entity_ids is not None:
        expression = _entity_predicate(native, entity_key, entity_ids)
        predicate = expression if predicate is None else predicate & expression
    if queries:
        predicate = _bind_queries(predicate, queries, native.schema)
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
                                    entity_ids=entity_ids, opaque_filter=opaque_filter, queries=queries)

    return _TableExport(scanner, selection.headers, restore_original_headers, rows_hint, lineage, provenance)


def _bind_queries(predicate: Expression | None, queries: tuple[PreparedPredicate, ...],
                  schema: pa.Schema) -> Expression | None:
    for query in queries:
        expression = query(schema)
        predicate = expression if predicate is None else predicate & expression
    return predicate


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
