"""Project catalog-bound source fields without adding them to entity metadata."""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from decimal import Decimal
from hashlib import sha256
from io import RawIOBase, StringIO
import json
from math import isfinite
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, BinaryIO, Generator, Iterator, Literal, Protocol, Self, TypeVar, overload

from .dataset_metadata import ArtifactRef, DataPeriod, EditionKey
from .identifiers import canonical_source_identifiers
from .manifest import ContractError, canonical_json
from .tprs_partitions import TPRS_PARTITION_DATASETS as _TPRS_DATASETS

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa
    from openpyxl import Workbook
    from pyarrow.dataset import Dataset, Expression, Scanner

    from .export_provenance import ExportProvenance
    from .release import DataRelease
    from .source_predicates import PreparedPredicate
    from .source_series import ObservationSeries


_RangeBound = int | float | Decimal | str
_RangeInclusive = Literal["both", "left", "right", "neither"]
_T = TypeVar("_T")


def _range_query(field: str, lower: _RangeBound, upper: _RangeBound,
                 inclusive: _RangeInclusive) -> tuple[str, dict[str, object]]:
    if not isinstance(field, str) or not field.isascii() or not field.isidentifier():
        raise ValueError("Range filters require one canonical source field name")
    if inclusive not in ("both", "left", "right", "neither"):
        raise ValueError("Range inclusive must be 'both', 'left', 'right', or 'neither'")
    if type(lower) not in (int, float, Decimal, str) or type(upper) not in (int, float, Decimal, str):
        raise TypeError("Range bounds must be numbers or publisher text")
    left = "<=" if inclusive in ("both", "left") else "<"
    right = "<=" if inclusive in ("both", "right") else "<"
    return f"@lower {left} {field} {right} @upper", {"lower": lower, "upper": upper}


def _preview_limit(n: int) -> int:
    if type(n) is not int or n < 0:
        raise ValueError("Preview row limit must be a nonnegative integer")
    return n


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
    if scope in {"district", "campus"} and (binding := artifact.entity_binding) is not None:
        return binding.column
    key = scope + "_number"
    if scope not in {"district", "campus"} or key not in artifact.contract["key_columns"]:
        raise ContractError("Source table has no declared entity row binding or canonical entity key. "
                            "Use release.observations(...) to inspect the source rows and their published identifiers.")
    return key


def _entity_predicate(native: Dataset, entity_key: str, identifiers: tuple[str, ...], *, width: int) -> Expression:
    import pyarrow as pa
    from pyarrow.dataset import field, scalar

    if not identifiers:
        return scalar(False)
    key_type = native.schema.field(entity_key).type
    if not (pa.types.is_string(key_type) or pa.types.is_large_string(key_type)):
        raise ValueError("Source identifier batches require Arrow strings")
    batches = native.to_batches(columns=[entity_key], batch_size=65_536,
                               use_threads=False, **_dataset_scan_options())
    aliases = _entity_aliases(batches, key_type, identifiers, width=width)
    return field(entity_key).isin(aliases)


def _entity_aliases(batches, key_type, identifiers: tuple[str, ...], *, width: int):
    import pyarrow.compute as arrow_compute
    from .arrow_io import primitive_array

    pc: Any = arrow_compute  # Arrow registers these kernels dynamically.
    selected = primitive_array(["'" + identifier for identifier in identifiers], key_type)
    aliases = set()
    try:
        for batch in batches:
            keys = batch.column(0)
            normalized = canonical_source_identifiers(keys, width=width)
            aliases.update(pc.unique(pc.filter(keys, pc.is_in(normalized, value_set=selected))).to_pylist())
    finally:
        if close := getattr(batches, "close", None):
            close()
    return primitive_array(sorted(aliases), key_type)


def _dataset_scan_options() -> dict[str, Any]:
    from pyarrow.dataset import ParquetFragmentScanOptions

    return {"batch_readahead": 0, "fragment_readahead": 0, "cache_metadata": False,
            "fragment_scan_options": ParquetFragmentScanOptions(
                pre_buffer=False, use_buffered_stream=True, buffer_size=65_536)}


def _provenance_batches(batches: Generator[pa.RecordBatch, None, None],
                        provenance: Mapping[str, Any]) -> Generator[pa.RecordBatch, None, None]:
    """Attach the pinned receipt without copying columns; forward early close."""
    import pyarrow as pa

    try:
        encoded = canonical_json(provenance).encode()
        original_schema, schema = None, None
        for batch in batches:
            if original_schema is None or not original_schema.equals(batch.schema, check_metadata=True):
                original_schema = batch.schema
                schema = original_schema.with_metadata({**(original_schema.metadata or {}), b"teadata_provenance": encoded})
            yield pa.RecordBatch.from_arrays(batch.columns, schema=schema)
    finally:
        batches.close()


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

    def between(self, field: str, lower: _RangeBound, upper: _RangeBound, *,
                inclusive: _RangeInclusive = "both", record: bool = False) -> SourceTable:
        """Filter one field by exact bounds with independently selectable endpoints."""
        expression, params = _range_query(field, lower, upper, inclusive)
        return self.query(expression, params=params, record=record)

    @overload
    def __getitem__(self, key: str) -> Expression: ...

    @overload
    def __getitem__(self, key: list[str] | tuple[str, ...] | Expression) -> SourceTable: ...

    def __getitem__(self, key: str | list[str] | tuple[str, ...] | Expression) -> Expression | SourceTable:
        if isinstance(key, str):
            from pyarrow.dataset import field
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

    def iter_batches(self) -> Generator[pa.RecordBatch, None, None]:
        """Stream selected columns with provenance, opening the leaf on first pull.

        Close the generator when stopping early. Empty streams have no batch
        metadata; provenance() also provides the receipt without scanning rows.
        """
        export = self._prepare()
        yield from _provenance_batches(export.batches(), export.provenance().to_dict())

    def to_arrow(self) -> pa.Table:
        """Materialize the filtered projection with its keys and source support."""
        return self._prepare().to_arrow()

    def head(self, n: int = 5) -> pa.Table:
        """Materialize at most n rows in scan order as an Arrow preview."""
        limit = _preview_limit(n)
        return self._prepare().to_arrow(preview_rows=limit)

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

    def to_csv_bundle(self, path_or_buf: str | Path | BinaryIO, *, restore_original_headers: bool = False) -> None:
        """Stream CSV, canonical provenance and member checksums to one ZIP file."""
        write_csv_bundle(self._prepare(restore_original_headers=restore_original_headers), path_or_buf)

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

    def between(self, field: str, lower: _RangeBound, upper: _RangeBound, *,
                inclusive: _RangeInclusive = "both", record: bool = False) -> PartitionedSourceTable:
        """Filter one field by exact bounds across the selected partitions."""
        expression, params = _range_query(field, lower, upper, inclusive)
        return self.query(expression, params=params, record=record)

    @overload
    def __getitem__(self, key: str) -> Expression: ...

    @overload
    def __getitem__(self, key: list[str] | tuple[str, ...] | Expression) -> PartitionedSourceTable: ...

    def __getitem__(self, key: str | list[str] | tuple[str, ...] | Expression) -> Expression | PartitionedSourceTable:
        if isinstance(key, str):
            from pyarrow.dataset import field
            return field(key)
        if isinstance(key, (list, tuple)):
            return self.select(*key)
        return self.where(key)

    def _prepare(self, *, restore_original_headers: bool = False) -> _TableExport:
        import pyarrow as pa

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
                from pyarrow.dataset import dataset
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

    def iter_batches(self) -> Generator[pa.RecordBatch, None, None]:
        """Stream pinned partitions in order with the complete selection receipt.

        Close the generator when stopping early to avoid opening later leaves.
        """
        export = self._prepare()
        yield from _provenance_batches(export.batches(), export.provenance().to_dict())

    def to_arrow(self) -> pa.Table:
        return self._prepare().to_arrow()

    def head(self, n: int = 5) -> pa.Table:
        """Materialize at most n rows in partition and scan order."""
        limit = _preview_limit(n)
        return self._prepare().to_arrow(preview_rows=limit)

    def to_df(self, *, restore_original_headers: bool = False) -> pd.DataFrame:
        return self._prepare(restore_original_headers=restore_original_headers).to_df()

    def provenance(self, *, restore_original_headers: bool = False) -> PartitionedExportProvenance:
        result = self._prepare(restore_original_headers=restore_original_headers).provenance()
        assert isinstance(result, PartitionedExportProvenance)
        return result

    def to_csv(self, path_or_buf: str | Path | BinaryIO, *, restore_original_headers: bool = False) -> None:
        write_csv(self._prepare(restore_original_headers=restore_original_headers), path_or_buf)

    def to_csv_bundle(self, path_or_buf: str | Path | BinaryIO, *, restore_original_headers: bool = False) -> None:
        """Stream selected partitions and their exact provenance to one ZIP file."""
        write_csv_bundle(self._prepare(restore_original_headers=restore_original_headers), path_or_buf)

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


class _UnknownSourceName(KeyError):
    """Only a missing namespace name becomes an AttributeError."""


def _unknown_source_name(name: str, names: Iterable[str], context: str, discovery: str) -> _UnknownSourceName:
    from .selectors import _field_suggestions

    matches = _field_suggestions(name, names) if len(name) <= 256 else ()
    hint = f" Did you mean {', '.join(map(repr, matches))}?" if matches else ""
    return _UnknownSourceName(f"Unknown {context} {name!r}.{hint} See {discovery} for exact names.")


@dataclass(frozen=True, slots=True)
class SourceFields:
    release: DataRelease
    edition: EditionKey
    identifier: str
    _entity_build: tuple[str, str] | None = None

    def variables(self, pattern: str | re.Pattern[str] | None = None) -> tuple[dict[str, Any], ...]:
        """Discover fields in this exact source, optionally by regex on their names."""
        match = re.compile(pattern) if pattern is not None else None
        return tuple(row for row in self.release.catalog.variables(edition=self.edition)
                     if match is None or match.search(row["field"]))

    def select(self, *fields: str, where: Expression | None = None) -> pa.Table:
        """Project selected fields, native keys and their available source support."""
        table = SourceTable(self.release, self.edition, _entity_ids=(self.identifier,),
                            _entity_build=self._entity_build).select(*fields)
        if where is not None:
            table = table.where(where)
        return table.to_arrow()

    def __getitem__(self, name: str):
        if not isinstance(name, str):
            raise TypeError("Source field selection requires a field name")
        catalog = self.release.catalog
        metadata = catalog.field_metadata(name, scope=self.edition.scope, edition=self.edition)
        if metadata is None:
            matches = catalog._observation_fields(self.edition, (name,))
            if not matches:
                raise _unknown_source_name(name, catalog._edition_field_names(self.edition), "source field", "variables()")
            name = matches[name]
            metadata = catalog.field_metadata(name, scope=self.edition.scope, edition=self.edition)
            if metadata is None:
                raise ContractError(f"Source field {name!r} is missing its catalog binding")
        table = self.select(name)
        if table.num_rows != 1:
            selection = f"{self.edition.scope} {self.identifier} in {self.edition.dataset}, edition {self.edition.edition}"
            if table.num_rows == 0:
                raise ContractError(f"No published row for {selection}. Check the publisher roster.")
            raise ContractError(f"Multiple published rows for {selection}. Use select() with a row filter.")
        return table[metadata["column"]][0].as_py()

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except _UnknownSourceName as error:
            raise AttributeError(*error.args) from error


@dataclass(frozen=True, slots=True)
class EntitySources:
    release: DataRelease
    scope: str
    identifier: str
    edition: str

    def datasets(self) -> tuple[EditionKey, ...]:
        """List exact source identities using only the release index."""
        if attachments := self.release.native_sources:
            return tuple(key for source in attachments for key in source.editions
                         if key.scope == self.scope and key.edition == self.edition)
        return tuple(EditionKey.from_dict({name: row[name] for name in EditionKey.__dataclass_fields__})
                     for row in self.release.catalog.editions(scope=self.scope, include_metadata=False)
                     if row["edition"] == self.edition)

    def dataset(self, name: str | EditionKey) -> SourceFields:
        keys = self.datasets()
        if isinstance(name, str):
            choices = [key for key in keys if key.dataset == name]
            if not choices:
                folded = name.casefold()
                choices = [key for key in keys if key.dataset.casefold() == folded]
            if not choices:
                raise _unknown_source_name(name, (key.dataset for key in keys), "source dataset", "datasets()")
        elif isinstance(name, EditionKey):
            choices = [key for key in keys if key == name]
            if not choices:
                raise _UnknownSourceName(f"Exact source edition {name.identity} is unavailable. See datasets().")
        else:
            raise TypeError("Source dataset selection requires a name or exact EditionKey")
        if len(choices) > 1 and isinstance(name, str) and self.scope == "campus":
            from .tprs_partitions import STUDENT_DATASET, TPRS_PARTITION_DATASETS, derived_campus_edition

            dataset = choices[0].dataset
            if dataset in TPRS_PARTITION_DATASETS and all(key.dataset == dataset for key in choices):
                routed = self.release.tprs_student_campus_index(edition=self.edition).edition_for(self.identifier)
                if routed is None:
                    raise ContractError(
                        f"No admitted {dataset} report for campus {self.identifier} in publisher edition "
                        f"{self.edition}. Check that edition's publisher roster; see datasets() for source choices."
                    )
                if dataset != STUDENT_DATASET:
                    routed = derived_campus_edition(routed, dataset)
                if routed not in choices:
                    raise ContractError("Campus report route is outside the selected dataset editions")
                choices = [routed]
        if len(choices) != 1:
            raise ContractError("Source name is ambiguous; select an exact EditionKey")
        (key,) = choices
        table = self.release.table(key)
        return SourceFields(table.release, key, self.identifier, _entity_build=table._entity_build)

    def __getitem__(self, name: str) -> SourceFields:
        return self.dataset(name)

    def __getattr__(self, name: str) -> SourceFields:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self.dataset(name)
        except _UnknownSourceName as error:
            raise AttributeError(*error.args) from error



@dataclass(frozen=True, slots=True)
class _SourceNameIndex:
    context: tuple[Any, ...]
    names: tuple[tuple[str, tuple[str, ...]], ...]
    files: tuple[tuple[Path, str, int, tuple[int, ...]], ...]

    def matches_files(self) -> bool:
        from .dataset_metadata import ColumnSchemaRef
        from .release import _capture_column_schema, _matches_metadata_files

        if not self.files:
            return False
        try:
            if getattr(os, "O_NOFOLLOW", 0):
                return _matches_metadata_files(tuple((path, stamp) for path, _, _, stamp in self.files))
            for path, digest, size, _ in self.files:
                # Without POSIX identity guarantees, verify bytes instead of
                # treating Windows pathname timestamps as a content identity.
                _capture_column_schema(path, ColumnSchemaRef(digest, size, path.name))
            return True
        except (ContractError, OSError):
            return False

    def projection(self) -> dict[str, set[str]]:
        return {dataset: set(names) for dataset, names in self.names}


@dataclass(frozen=True, slots=True)
class _EntityFields:
    """Shared metadata and observation resolver; no entity payload copies."""

    release: DataRelease
    scope: str
    identifier: str
    school_year: int | None = None

    @property
    def _datasets(self) -> frozenset[str]:
        raise NotImplementedError

    @property
    def _label(self) -> str:
        return "source"

    def _indexes(self) -> dict[str, set[str]]:
        selected = tuple(sorted(self._datasets))
        cache_key = self.scope, selected
        context = (self.scope, selected, self.release.manifest_sha256,
                   self.release.catalog.build_id, self.release.source, self.release.path,
                   self.release.cache, getattr(self.release, "_cache_dir", None), self.release.offline)
        cached = getattr(self.release, "_source_name_indexes", {}).get(cache_key)
        if cached is not None and cached.context == context and cached.matches_files():
            return cached.projection()
        index = self._read_indexes(context)
        if index.files and not index.matches_files():
            raise ContractError("Source metadata changed while discovering fields. Retry with verified source artifacts.")
        if self.scope in {"district", "campus"} and index.files and len(index.files) <= 256:
            entries = dict(getattr(self.release, "_source_name_indexes", {}))
            entries.pop(cache_key, None)
            entries[cache_key] = index
            while len(entries) > 8:
                entries.pop(next(iter(entries)))
            self.release._source_name_indexes = entries
        return index.projection()

    def _read_indexes(self, context: tuple[Any, ...]) -> _SourceNameIndex:
        from .dataset_metadata import ColumnSchemaRef
        from .release import _capture_column_schema

        groups = []
        if self.release._v3:
            groups.append(tuple(EditionKey.from_dict({name: row[name] for name in EditionKey.__dataclass_fields__})
                                for row in self.release.catalog.editions(scope=self.scope, include_metadata=False)
                                if row["dataset"] in self._datasets))
        else:
            groups.extend(tuple(key for key in source.editions
                                if key.scope == self.scope and key.dataset in self._datasets)
                          for source in self.release.native_sources)
        names: dict[str, set[str]] = {}
        files: dict[Path, tuple[str, int, tuple[int, ...]]] = {}
        for keys in groups:
            if not keys:
                continue
            child = self.release if self.release._v3 else self.release._native_edition_release(keys[0])
            try:
                for key in keys:
                    names.setdefault(key.dataset, set()).update(child.catalog._edition_field_names(key))
                resources = [(path, digest, stamp[4], stamp)
                             for digest, (path, stamp, _) in child._column_schema_data.items()]
                if child is not self.release:
                    size = child.path.stat().st_size
                    _, stamp = _capture_column_schema(child.path, ColumnSchemaRef(child.manifest_sha256, size, child.path.name))
                    resources.append((child.path, child.manifest_sha256, size, stamp))
                for path, digest, size, stamp in resources:
                    if path in files and files[path][:2] != (digest, size):
                        raise ContractError("Source metadata routes conflicting identities to one file")
                    files[path] = digest, size, stamp
            finally:
                if child is not self.release:
                    # These children belong only to discovery. Bound loaders
                    # otherwise retain large parsed graphs until cyclic GC.
                    child.catalog._clear_edition_caches()
                    child._column_schema_data.clear()
        return _SourceNameIndex(context, tuple((dataset, tuple(sorted(values))) for dataset, values in sorted(names.items())),
                              tuple((path, *identity) for path, identity in sorted(files.items())))

    def variables(self, pattern: str | re.Pattern[str] | None = None) -> tuple[str, ...]:
        """List canonical names from metadata indexes, without opening data leaves."""
        match = re.compile(pattern) if pattern is not None else None
        return tuple(sorted({name for names in self._indexes().values() for name in names
                             if match is None or match.search(name)}))

    def _resolve(self, fields: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
        from .dataset_metadata import _validate_observation_selection
        from .selectors import _folded_field_names

        _validate_observation_selection(self._label, fields, self.scope, None)
        indexes = self._indexes()
        if not indexes:
            raise ContractError(
                f"No qualified {self._label} native families are attached for {self.scope}. "
                "Inspect release.native_sources; configured sources are not automatically admitted."
            )
        names = {name for values in indexes.values() for name in values}
        resolved = []
        for name, matches in _folded_field_names(fields, names).items():
            if name in matches:
                resolved.append(name)
            elif len(matches) == 1:
                resolved.append(matches[0])
            elif matches:
                raise ContractError(f"Ambiguous {self._label} field {name!r}. Use exact spelling: {matches}.")
            else:
                raise _unknown_source_name(name, names, f"{self._label} field", f"{self._label.lower()}.variables()")
        if len(set(resolved)) != len(resolved):
            raise ValueError(f"{self._label} fields must resolve to distinct canonical names")
        owners = {name: [dataset for dataset, values in indexes.items() if name in values] for name in resolved}
        ambiguous = {name: datasets for name, datasets in owners.items() if len(datasets) > 1}
        if ambiguous:
            raise ContractError(
                f"{self._label} fields require one unambiguous native family; conflicting names: {ambiguous}. "
                "Use release.for_entity(..., edition=...).dataset(exact_key) to choose a source explicitly."
            )
        choices = [dataset for dataset, values in indexes.items() if all(name in values for name in resolved)]
        if len(choices) != 1:
            raise ContractError(
                f"{self._label} fields require one unambiguous native family; matching families: {choices}. "
                "Select fields from one family, or use release.for_entity(..., edition=...).dataset(exact_key)."
            )
        return choices[0], tuple(resolved)

    def at(self, school_year: int | str | None) -> Self:
        """Select a measurement school year; None selects the latest eligible year."""
        from .dataset_metadata import _parse_school_year

        return replace(self, school_year=_parse_school_year(school_year))

    def years(self, field: str) -> tuple[int, ...]:
        """List admitted measurement years for a field, separately from row presence."""
        dataset, (name,) = self._resolve((field,))
        years = self.release.observation_years(dataset, name, scope=self.scope)[name]
        if not years:
            self.release.observations(dataset, name, scope=self.scope)
        return years

    def select(self, *fields: str, school_year: int | str | None = None) -> SourceTable:
        """Plan an entity projection with its original raw/status/provenance support."""
        from .dataset_metadata import _parse_school_year

        year = self.school_year if school_year is None else _parse_school_year(school_year)
        dataset, names = self._resolve(fields)
        table = self.release._entity_observations(dataset, names, school_year=year, scope=self.scope,
                                                   edition=None, identifiers=(self.identifier,))
        if not isinstance(table, SourceTable):
            raise ContractError("One entity cannot resolve to multiple source partitions")
        return table

    def provenance(self, *fields: str) -> ExportProvenance:
        """Describe the pinned selection without opening its data leaf."""
        return self.select(*fields).provenance()

    def series(self, *fields: str, years: Sequence[int | str]) -> ObservationSeries:
        """Plan explicit measurement years using the existing bound-source series."""
        if self.school_year is not None:
            raise ValueError(
                f"This namespace is pinned to school year {self.school_year}. "
                "Call .at(None).series(..., years=[...]) to select explicit series years."
            )
        dataset, names = self._resolve(fields)
        return self.release.series(dataset, *names, scope=self.scope, years=years, ids=(self.identifier,))

    def __getitem__(self, name: str):
        view = self.select(name)
        assert view._fields is not None
        metadata = view.release.catalog.field_metadata(view._fields[0], scope=self.scope, edition=view.edition)
        if metadata is None:
            raise ContractError(f"{self._label} field {name!r} is missing its catalog binding")
        table = view.to_arrow()
        if table.num_rows != 1:
            kind = "No" if table.num_rows == 0 else "Multiple"
            raise ContractError(
                f"{kind} published rows for {self.scope} {self.identifier} in {view.edition.dataset}, "
                f"measurement period {view.measurement_period}. "
                "Use select(...).query(...) to select source rows explicitly; check the publisher roster for absent rows."
            )
        return table[metadata["column"]][0].as_py()

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except _UnknownSourceName as error:
            raise AttributeError(*error.args) from error

    def __dir__(self) -> list[str]:
        return sorted(set(object.__dir__(self)) | set(self.variables()))


@dataclass(frozen=True, slots=True)
class TPRSFields(_EntityFields):
    """Compatible TPRS facade over the shared catalog-bound resolver."""

    @property
    def _datasets(self) -> frozenset[str]:
        return _TPRS_DATASETS

    @property
    def _label(self) -> str:
        return "TPRS"


@dataclass(frozen=True, slots=True)
class DatasetFields(_EntityFields):
    """One attached source dataset; measurement school years remain explicit."""

    dataset: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset.strip():
            raise ValueError("DatasetFields requires an exact attached dataset name")

    @property
    def _datasets(self) -> frozenset[str]:
        return frozenset({self.dataset})

    @property
    def _label(self) -> str:
        return self.dataset


@dataclass(frozen=True, slots=True)
class _TableExport:
    scanner: Scanner | _PartitionScan | _ParquetScan
    headers: tuple[tuple[str, ...], ...]
    restore_original_headers: bool
    rows_hint: int | None
    lineage: dict[str, Any]
    provenance: Callable[[], ExportProvenance | PartitionedExportProvenance]

    @property
    def schema(self) -> pa.Schema:
        return self.scanner.projected_schema

    def batches(self) -> Generator[pa.RecordBatch, None, None]:
        batches = self.scanner.to_batches()
        try:
            for batch in batches:
                if self.restore_original_headers and any(column.null_count for column in batch.columns):
                    raise ContractError("A publisher-header export cannot substitute blanks for null values")
                yield batch
        finally:
            if close := getattr(batches, "close", None):
                close()

    def to_arrow(self, *, preview_rows: int | None = None) -> pa.Table:
        import pyarrow as pa

        metadata = {**(self.schema.metadata or {}),
                    b"teadata_provenance": canonical_json(self.provenance().to_dict()).encode()}
        if preview_rows is not None:
            metadata[b"teadata_preview"] = canonical_json({
                "limit": preview_rows, "provenance_scope": "full_selection",
            }).encode()
        schema = self.schema.with_metadata(metadata)
        with _owned_iterator(self.batches()) as batches:
            selected = batches if preview_rows is None else _prefix_batches(batches, preview_rows)
            return pa.Table.from_batches(selected, schema=schema)

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
                   opaque_filter: bool, queries: tuple[PreparedPredicate, ...] = (),
                   query_recipes: tuple[str, ...] = ()) -> ExportProvenance:
        from .export_provenance import describe_export

        if entity_ids is not None:
            _entity_key(self.artifact)
        records = [*(query.recipe_json for query in queries), *query_recipes]
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
            batches = export.batches()
            try:
                for batch in batches:
                    yield batch.replace_schema_metadata(None)
            finally:
                batches.close()


def _projected_schema(schema: pa.Schema, columns: tuple[str, ...]) -> pa.Schema:
    import pyarrow as pa

    return pa.schema([schema.field(name) for name in columns], metadata=schema.metadata)


def _require_unchanged_stream(stream: BinaryIO, signature: tuple[int, ...]) -> None:
    state = os.fstat(stream.fileno())
    if (state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns) != signature:
        raise ContractError("Native artifact changed during its owned read")


@dataclass(frozen=True, slots=True)
class _ParquetScan:
    """One descriptor-owned plain projection; no handles survive preparation."""

    release: DataRelease
    artifact: ArtifactRef
    columns: tuple[str, ...]
    entity_ids: tuple[str, ...] | None
    projected_schema: pa.Schema

    @classmethod
    def prepare(cls, release: DataRelease, artifact: ArtifactRef, columns: tuple[str, ...],
                entity_ids: tuple[str, ...] | None) -> _ParquetScan:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if release.catalog.artifact_ref(artifact.identity) != artifact:
            raise ContractError("Native projection descriptor changed before preparation")
        with release._open_artifact(artifact.identity) as (source, signature):
            with pq.ParquetFile(source, pre_buffer=False, buffer_size=65_536) as parquet:
                schema = _projected_schema(parquet.schema_arrow, columns)
                if entity_ids:
                    key_type = parquet.schema_arrow.field(_entity_key(artifact)).type
                    if not (pa.types.is_string(key_type) or pa.types.is_large_string(key_type)):
                        raise ValueError("Source identifier batches require Arrow strings")
                _require_unchanged_stream(source, signature)
        return cls(release, artifact, columns, entity_ids, schema)

    def to_batches(self) -> Generator[pa.RecordBatch, None, None]:
        import pyarrow.compute as arrow_compute
        import pyarrow.parquet as pq

        pc: Any = arrow_compute  # Arrow registers these kernels dynamically.
        if self.release.catalog.artifact_ref(self.artifact.identity) != self.artifact:
            raise ContractError("Native projection descriptor changed before consumption")
        with self.release._open_artifact(self.artifact.identity) as (source, signature):
            with pq.ParquetFile(source, pre_buffer=False, buffer_size=65_536) as parquet:
                if not _projected_schema(parquet.schema_arrow, self.columns).equals(
                    self.projected_schema, check_metadata=True,
                ):
                    raise ContractError("Native projection schema changed before consumption")
                if self.entity_ids == ():
                    return
                columns, aliases, key = self.columns, None, None
                row_groups = None
                if self.entity_ids is not None:
                    from .arrow_io import primitive_array

                    key = _entity_key(self.artifact)
                    key_type = parquet.schema_arrow.field(key).type
                    row_groups, spellings = [], set()
                    # Validate every key before emitting data, including keys
                    # in groups that do not contain a selected entity.
                    for group in range(parquet.metadata.num_row_groups):
                        matched = _entity_aliases(
                            parquet.iter_batches(batch_size=65_536, columns=[key],
                                                 row_groups=[group], use_threads=False),
                            key_type, self.entity_ids,
                            width=6 if self.artifact.edition.scope == "district" else 9,
                        )
                        _require_unchanged_stream(source, signature)
                        if len(matched):
                            row_groups.append(group)
                            spellings.update(matched.to_pylist())
                    aliases = primitive_array(sorted(spellings), key_type)
                    if not len(aliases):
                        return
                    if key not in columns:
                        columns = (*columns, key)
                batches = parquet.iter_batches(batch_size=256, columns=list(columns),
                                               row_groups=row_groups, use_threads=False)
                try:
                    for batch in batches:
                        if aliases is not None:
                            assert key is not None
                            batch = batch.filter(pc.is_in(batch.column(key), value_set=aliases)).select(self.columns)
                        _require_unchanged_stream(source, signature)
                        if not batch.schema.equals(self.projected_schema, check_metadata=True):
                            raise ContractError("Native projection returned a different schema")
                        if batch.num_rows:
                            yield batch
                finally:
                    if close := getattr(batches, "close", None):
                        close()


def _plan_export(release: DataRelease, edition: EditionKey, *, restore_original_headers: bool,
                 selected_fields: tuple[str, ...] | None, include_support: bool,
                 period: DataPeriod | None) -> _ExportSelection:
    if type(restore_original_headers) is not bool:
        raise ValueError("Original headers must be explicitly enabled or disabled")

    headers = release.catalog.header_records(edition.dataset, edition_key=edition) if (
        selected_fields is None or restore_original_headers) else []
    fields = tuple(row["field"] for row in headers) if headers else selected_fields or ()
    requested = fields if selected_fields is None else selected_fields
    artifact, projected = _projection(release, edition, requested, period=period)
    columns = projected if include_support else fields
    if include_support and not restore_original_headers and (binding := artifact.entity_binding) is not None:
        columns = tuple(dict.fromkeys((*artifact.contract["key_columns"], binding.column, *columns)))
    original_error = ("Original headers require a complete raw TAPR table or reviewed code-only RDA district table"
                      " or reviewed PEIMS course or College Credit table")
    header_kinds = ()
    publisher_columns = None
    if restore_original_headers:
        import pyarrow as pa

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
    # Parquet's dotted-path selectors can differ from an Arrow expression's
    # literal field-name semantics. Keep those unusual names on the existing
    # Dataset executor, alongside explicit predicates and queries.
    plain = (predicate is None and not queries and not any("." in name for name in selection.columns)
             and (entity_key is None or "." not in entity_key))
    if plain:
        scanner = _ParquetScan.prepare(release, artifact, selection.columns, entity_ids)
        rows_hint = artifact.contract["rows"] if entity_ids is None else None
    else:
        native = release.analytical_dataset(artifact.identity)
        if entity_key is not None and entity_ids is not None:
            expression = _entity_predicate(native, entity_key, entity_ids,
                                           width=6 if edition.scope == "district" else 9)
            predicate = expression if predicate is None else predicate & expression
        if queries:
            predicate = _bind_queries(predicate, queries, native.schema)
        rows_hint = artifact.contract["rows"] if predicate is None else None
        scanner = native.scanner(columns=list(selection.columns), filter=predicate, batch_size=256,
                                 use_threads=False, **_dataset_scan_options())
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


class _WriterProvenance(Protocol):
    def _append_sheet(self, workbook: Workbook, *, rows: int | None = None) -> None: ...

    def to_dict(self) -> dict[str, Any]: ...


class _StreamingExport(Protocol):
    @property
    def schema(self) -> pa.Schema: ...

    @property
    def headers(self) -> tuple[tuple[str, ...], ...]: ...

    @property
    def restore_original_headers(self) -> bool: ...

    @property
    def rows_hint(self) -> int | None: ...

    def batches(self) -> Iterator[pa.RecordBatch]: ...

    def provenance(self) -> _WriterProvenance: ...


class _CSVStream(Protocol):
    @property
    def closed(self) -> bool: ...

    def write(self, data: Any, /) -> int: ...


@contextmanager
def _owned_iterator(iterator: Iterator[_T]) -> Generator[Iterator[_T], None, None]:
    """Close even a retained iterator when encoding or a consumer fails."""
    try:
        yield iterator
    finally:
        if close := getattr(iterator, "close", None):
            close()


def _prefix_batches(batches: Iterator[pa.RecordBatch], remaining: int) -> Iterator[pa.RecordBatch]:
    if not remaining:
        return
    for batch in batches:
        if batch.num_rows >= remaining:
            yield batch if batch.num_rows == remaining else batch.slice(0, remaining)
            return
        yield batch
        remaining -= batch.num_rows


def write_csv(table: _StreamingExport, path_or_buf: str | Path | BinaryIO) -> None:
    with _export_stream(path_or_buf) as stream:
        _write_csv_stream(table, stream)


def _write_csv_stream(table: _StreamingExport, stream: _CSVStream) -> None:
    from pyarrow.csv import CSVWriter, WriteOptions

    if table.restore_original_headers:
        labels = StringIO(newline="")
        csv.writer(labels, lineterminator="\r\n").writerows(table.headers)
        stream.write(labels.getvalue().encode("utf-8"))
    with CSVWriter(stream, table.schema, write_options=WriteOptions(include_header=not table.restore_original_headers)) as writer:
        with _owned_iterator(table.batches()) as batches:
            for batch in batches:
                writer.write_batch(batch)


class _DigestWriter(RawIOBase):
    """Hash the exact bytes accepted by one ZIP member, without owning it."""

    def __init__(self, stream: _CSVStream):
        super().__init__()
        self.stream = stream
        self.digest = sha256()
        self.size = 0

    def writable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.size

    def write(self, data: Any) -> int:
        written = self.stream.write(data)
        if written != len(data):
            raise OSError("Incomplete CSV bundle member write")
        self.digest.update(data)
        self.size += written
        return written


def write_csv_bundle(table: _StreamingExport, path_or_buf: str | Path | BinaryIO) -> None:
    """One streaming traversal and one atomic publication for data plus lineage.

    Member hashes describe these exported bytes, not scientific comparability or
    publisher approval. Source and selection identities remain the existing
    canonical provenance record. The caller owns non-path output streams.
    """
    _write_csv_bundle(table.provenance(), lambda stream: _write_csv_stream(table, stream), path_or_buf)


def _write_csv_bundle(provenance_record: _WriterProvenance, write_data: Callable[[_CSVStream], None],
                      path_or_buf: str | Path | BinaryIO) -> None:
    from zipfile import ZIP_DEFLATED, ZipFile

    provenance = canonical_json(provenance_record.to_dict()).encode("utf-8")
    with _export_stream(path_or_buf) as stream, ZipFile(stream, "w", compression=ZIP_DEFLATED,
                                                     compresslevel=6, allowZip64=True) as archive:
        with archive.open("data.csv", "w", force_zip64=True) as member, _DigestWriter(member) as writer:
            write_data(writer)
            data_identity = {"sha256": writer.digest.hexdigest(), "size_bytes": writer.size}
        archive.writestr("provenance.json", provenance)
        manifest = {"schema_version": 1, "format": "teadata-csv-bundle",
                    "members": {"data.csv": data_identity,
                                "provenance.json": {"sha256": sha256(provenance).hexdigest(),
                                                    "size_bytes": len(provenance)}}}
        archive.writestr("manifest.json", canonical_json(manifest).encode("utf-8"))


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


def write_excel(table: _StreamingExport, path_or_buf: str | Path | BinaryIO) -> None:
    def data_rows():
        with _owned_iterator(table.batches()) as batches:
            for batch in batches:
                yield from zip(*(column.to_pylist() for column in batch.columns), strict=True)

    _write_excel_rows(table.provenance(), table.headers, len(table.schema), table.rows_hint, data_rows(), path_or_buf)


def _write_excel_rows(provenance: _WriterProvenance, headers: tuple[tuple[str, ...], ...], width: int,
                      rows_hint: int | None, data_rows: Iterator[Sequence[Any]],
                      path_or_buf: str | Path | BinaryIO) -> None:
    from openpyxl.cell import WriteOnlyCell

    max_rows = 1_048_576 - len(headers)
    row_limit_error = "Source exceeds Excel worksheet limits; use CSV for this table"
    if width > 16_384 or (rows_hint is not None and rows_hint > max_rows):
        raise ValueError(row_limit_error)
    with _owned_iterator(data_rows), _excel_workbook(path_or_buf) as workbook:
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

        for row in headers:
            sheet.append(list(cells(row)))
        rows = 0
        for rows, row in enumerate(data_rows, 1):
            if rows > max_rows:
                raise ValueError(row_limit_error)
            sheet.append(list(cells(row)))
        provenance._append_sheet(workbook, rows=rows)
