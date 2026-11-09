"""Lazy multi-year retrieval of exact published observation views."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Integral
from typing import TYPE_CHECKING, Any

from .dataset_metadata import EditionKey, _parse_school_year, _validate_observation_selection
from .identifiers import canonical_source_identifier, canonical_source_identifiers
from .manifest import ContractError, canonical_json
from .selectors import _EntityNames
from .source_views import SourceTable, _entity_key, _open_export, _plan_export

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa
    from sqlalchemy import Connection, Table

    from .export_provenance import ExportProvenance
    from .persistence import SeriesSqlExport
    from .release import DataRelease
    from .source_views import _ExportSelection


_YEAR_COLUMN = "measurement_school_year"
_PROVENANCE_KEY = b"teadata_provenance"


def _required_year(value: int | str) -> int:
    year = _parse_school_year(value)
    if year is None:
        raise ValueError("Series requires an explicit school_year, such as 25 or '2024-25'.")
    return year


def _series_ids(ids: Sequence[int | str], scope: str) -> tuple[str, ...]:
    width = 6 if scope == "district" else 9
    message = f"ids must contain distinct, nonzero {scope} numbers of up to {width} digits."
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)) or not ids:
        raise ValueError(message)
    result = []
    for item in ids:
        try:
            if not isinstance(item, (Integral, str)):
                raise ValueError
            identifier = canonical_source_identifier(item, width=width)
            if identifier is None:
                raise ValueError
        except ValueError:
            raise ValueError(f"{message} Invalid ID: {item!r}.") from None
        result.append(identifier[1:])
    if len(set(result)) != len(result):
        raise ValueError(f"{message} Multiple inputs resolve to the same ID.")
    return tuple(result)


def _registry_ids(registry: DataRelease, names: tuple[str, ...], scope: str) -> tuple[str, ...]:
    from .entity_store import EntityStore

    with EntityStore(registry.artifact("entities"), immutable=True) as store:
        if scope == "district":
            directory = _EntityNames(((row["name"], row["district_number"])
                                      for row in store.iter_districts(include_meta=False) if row["district_number"]),
                                     str)
        else:
            directory = _EntityNames(((row["name"], row["lookup_key"])
                                      for row in store.iter_campuses(include_meta=False) if not row["is_private"]),
                                     str)
        return _series_ids([directory.resolve(name, scope=scope) for name in names], scope)


@dataclass(frozen=True, slots=True, eq=False)
class ObservationSeries:
    """Pinned source views with a common physical projection and explicit years."""

    dataset: str
    fields: tuple[str, ...]
    scope: str
    ids: tuple[str, ...] | None
    _sources: tuple[tuple[int, SourceTable, _ExportSelection, ExportProvenance], ...]
    schema: pa.Schema
    _registry: tuple[str, str, tuple[str, ...]] | None
    _release: DataRelease

    @classmethod
    def _plan(
        cls, release: DataRelease, dataset: str, fields: tuple[str, ...], *, scope: str,
        years: Sequence[int | str], ids: Sequence[int | str] | None,
        editions: Mapping[int | str, EditionKey] | None, names: Sequence[str] | None,
        entity_release: DataRelease | None,
    ) -> ObservationSeries:
        import pyarrow as pa

        from .analytical import _leaf_schema

        _validate_observation_selection(dataset, fields, scope, None)
        if not isinstance(years, Sequence) or isinstance(years, (str, bytes)) or not years:
            raise ValueError("years must be a nonempty sequence of school years, such as [23, 24, 25].")
        requested = tuple(map(_required_year, years))
        if len(requested) != len(set(requested)):
            raise ValueError("years must be distinct after resolving school-year spellings.")
        from .release import DataRelease

        if ids is not None and names is not None:
            raise ValueError("Select either ids or names.")
        if (ids is not None or names is not None) and scope not in ("district", "campus"):
            raise ValueError("ids and names can select public district or campus rows. Omit them for other scopes.")
        if entity_release is not None and names is None:
            raise ValueError("entity_release is only used with names.")
        cohort = _series_ids(ids, scope) if ids is not None else None
        registry = None
        selected_names: tuple[str, ...] = ()
        if names is not None:
            if (not isinstance(names, Sequence) or isinstance(names, (str, bytes)) or not names
                    or any(not isinstance(name, str) or not name.strip() for name in names)):
                raise ValueError("names must be a nonempty sequence of nonblank entity names.")
            selected_names = tuple(names)
            registry = release if entity_release is None else entity_release
            if not isinstance(registry, DataRelease):
                raise TypeError("entity_release must be an entity DataRelease.")
            if registry._v3:
                raise ContractError("Name selection requires an entity release. Pass entity_release or use ids.")

        pins: dict[int, EditionKey] = {}
        if editions is not None:
            if not isinstance(editions, Mapping):
                raise TypeError("editions must map requested school years to EditionKey objects.")
            for spelling, key in editions.items():
                year = _required_year(spelling)
                _validate_observation_selection(dataset, fields, scope, key)
                if not isinstance(key, EditionKey):
                    raise TypeError("editions must map requested school years to EditionKey objects.")
                if year in pins or year not in requested or key.dataset != dataset or key.scope != scope:
                    raise ValueError(f"Edition pin for {year} must uniquely match a requested year, dataset and scope.")
                pins[year] = key

        options = list(release._observation_year_options(dataset, fields, scope, None)) if len(pins) < len(requested) else []
        views = []
        common_structure = None
        output_fields: dict[str, pa.Field] = {}
        column_roles = {}
        present: dict[str, int] = {}
        planned = []
        for year in requested:
            pin = pins.get(year)
            candidates = list(release._observation_year_options(dataset, fields, scope, pin)) if pin is not None else options
            view = release._select_observation_view(fields, scope, year, candidates)
            if planned and view._fields != planned[0][1]._fields:
                raise ContractError(f"Source field names differ in school year {year}. "
                                    "Read each year separately until a reviewed mapping is available.")
            planned.append((year, view))
        if registry is not None:
            cohort = _registry_ids(registry, selected_names, scope)
        for year, view in planned:
            selected = _plan_export(view.release, view.edition, restore_original_headers=False,
                                    selected_fields=view._fields, include_support=True,
                                    period=view.measurement_period)
            native = _leaf_schema(selected.artifact.contract)
            if cohort is not None:
                key_type = native.field(_entity_key(selected.artifact)).type
                if not (pa.types.is_string(key_type) or pa.types.is_large_string(key_type)):
                    raise ContractError(f"Series ID selection requires a string publisher key for school year {year}.")
                view = replace(view, _entity_ids=cohort)
            source = selected.provenance(view.release, period=view.measurement_period, entity_build=view._entity_build,
                                         entity_ids=cohort, opaque_filter=False)
            roles = {
                column["column"]: (
                    column["role"],
                    tuple(field["field"] for field in column["fields"]),
                    tuple((item["field"], item["role"]) for item in column["support_for"]),
                )
                for column in source.to_dict()["columns"]
            }
            structure = (tuple(selected.artifact.contract["key_columns"]), view.edition.grain,
                         {name: role for name, role in roles.items() if role[0] != "context"})
            if common_structure is None:
                common_structure = structure
            elif structure != common_structure:
                raise ContractError(f"Series fields, types or publisher keys for school year {year} differ from "
                                    f"school year {requested[0]}. Select compatible fields or read each year separately.")
            if _YEAR_COLUMN in selected.columns:
                raise ContractError(f"Series year column {_YEAR_COLUMN!r} conflicts with school year {year} source columns.")
            for name in selected.columns:
                field = native.field(name)
                if name in output_fields:
                    if roles[name] != column_roles[name] or not field.equals(output_fields[name], check_metadata=True):
                        raise ContractError(f"Series column {name!r} has incompatible roles or types in school year {year}. "
                                            "Select compatible fields or read each year separately.")
                else:
                    output_fields[name] = field
                    column_roles[name] = roles[name]
                present[name] = present.get(name, 0) + 1
            views.append((year, view, selected, source))
        schema = pa.schema([
            pa.field(_YEAR_COLUMN, pa.int32(), nullable=False),
            *(field.with_nullable(True) if present[field.name] < len(requested) else field
              for field in output_fields.values()),
        ])
        registry_identity = None if registry is None else (registry.build_id, registry.manifest_sha256, selected_names)
        selected_fields = views[0][1]._fields
        assert selected_fields is not None
        return cls(dataset, selected_fields, scope, cohort, tuple(views), schema, registry_identity, release)

    @property
    def release(self) -> DataRelease:
        """The pinned root release used to select this series' editions."""
        return self._release

    @property
    def years(self) -> tuple[int, ...]:
        """Canonical ending years in request order. Format as four digits for literal reuse."""
        return tuple(year for year, *_ in self._sources)

    def for_year(self, school_year: int | str) -> SourceTable:
        """Return the pinned source view for one selected year."""
        target = _required_year(school_year)
        for year, view, _, _ in self._sources:
            if year == target:
                return view
        raise ContractError(f"School year {target} is outside this series. Selected years: {self.years}.")

    def pair_years(self, *, from_year: int | str, to_year: int | str) -> ObservationPair:
        """Plan a two-year alignment of publisher rows by public entity ID."""
        return ObservationPair._plan(self, _required_year(from_year), _required_year(to_year))

    def provenance(self) -> dict[str, Any]:
        """Describe each pinned source view and the exact series selector without scanning rows."""
        result: dict[str, Any] = {
            "schema_version": 1,
            "operation": "school_year_series",
            "dataset": self.dataset,
            "fields": list(self.fields),
            "scope": self.scope,
            "school_years": list(self.years),
            "ids": list(self.ids) if self.ids is not None else None,
            "year_column": _YEAR_COLUMN,
            "cross_year_comparability": "unassessed",
            "unavailable_context": "null when the column is absent from that edition's projection",
            "sources": [{"school_year": year, "provenance": source.to_dict()} for year, _, _, source in self._sources],
        }
        if self._registry is not None:
            build_id, manifest_sha256, names = self._registry
            assert self.ids is not None
            result["name_selection"] = {
                "build_id": build_id, "manifest_sha256": manifest_sha256,
                "resolved": [{"name": name, "id": identifier}
                             for name, identifier in zip(names, self.ids, strict=True)],
            }
        return result

    def iter_batches(self) -> Iterator[pa.RecordBatch]:
        """Scan one selected year at a time, with the measurement year on every row."""
        import pyarrow as pa

        fields = tuple(self.schema)[1:]
        for year, view, selected, _ in self._sources:
            scalar = pa.scalar(year, type=pa.int32())
            offsets = {name: position for position, name in enumerate(selected.columns)}
            export = _open_export(view.release, selected, period=view.measurement_period,
                                  entity_build=view._entity_build, entity_ids=self.ids)
            for batch in export.batches():
                columns = [batch.column(offsets[field.name]) if field.name in offsets
                           else pa.nulls(batch.num_rows, type=field.type) for field in fields]
                yield pa.RecordBatch.from_arrays([pa.repeat(scalar, batch.num_rows), *columns], schema=self.schema)

    def to_arrow(self) -> pa.Table:
        """Collect the series as an Arrow table with its source provenance."""
        import pyarrow as pa

        schema = self.schema.with_metadata({_PROVENANCE_KEY: canonical_json(self.provenance()).encode()})
        return pa.Table.from_batches(self.iter_batches(), schema=schema)

    def to_df(self) -> pd.DataFrame:
        """Collect the series with nullable Arrow types and per-year source provenance."""
        import json

        import pandas as pd

        table = self.to_arrow()
        frame = table.to_pandas(types_mapper=pd.ArrowDtype)
        frame.attrs["teadata_provenance"] = json.loads((table.schema.metadata or {})[_PROVENANCE_KEY])
        return frame

    def to_sql(self, connection: Connection, observations: Table, *, receipts: Table,
               require_complete: bool = True) -> SeriesSqlExport:
        """Stream into caller-defined SQLAlchemy tables with a pinned JSON receipt.

        Install `teadata[database]`. Use an active transactional connection
        (for example `with engine.begin() as connection`) and let errors
        propagate for rollback. Observation column keys match `series.schema`
        plus `import_id`. The receipt table has `id` as its primary key and a
        JSON `receipt` column. This method does not create tables or commit.
        Explicitly selected IDs require a row for every selected year by
        default. Pass `require_complete=False` to record missing pairs.
        """
        from .persistence.series import _export_series

        return _export_series(self, connection, observations, receipts=receipts, require_complete=require_complete)

    def iter_sql_batches(self, connection: Connection, observations: Table, *, receipts: Table,
                         import_id: str, years: Sequence[int | str] | None = None) -> Iterator[pa.RecordBatch]:
        """Read a matching SQL import, preserving the series schema and raw support columns.

        Supply the import ID from `to_sql` and optionally a subset of the
        series years. Batches have no guaranteed row order. Consume or close
        the iterator before closing the connection. The receipt checks the
        import plan; use a trusted database for the stored observations.
        """
        from .persistence.series import _read_series

        selected = None
        if years is not None:
            if not isinstance(years, Sequence) or isinstance(years, (str, bytes)) or not years:
                raise ValueError("Pass a nonempty sequence of school years.")
            selected = tuple(_required_year(year) for year in years)
            if len(set(selected)) != len(selected) or not set(selected) <= set(self.years):
                raise ValueError(f"Select distinct years from {self.years}.")
        return _read_series(self, connection, observations, receipts=receipts, import_id=import_id, years=selected)


@dataclass(frozen=True, slots=True, eq=False)
class ObservationPair:
    """Two exact school years aligned on public entity identifiers."""

    from_year: int
    to_year: int
    _series: ObservationSeries
    _endpoints: tuple[tuple[int, SourceTable, _ExportSelection, ExportProvenance], ...]
    _key: str
    schema: pa.Schema

    @classmethod
    def _plan(cls, series: ObservationSeries, from_year: int, to_year: int) -> ObservationPair:
        import pyarrow as pa

        from .analytical import _leaf_schema

        if from_year >= to_year:
            raise ValueError("pair_years requires from_year earlier than to_year.")
        by_year = {source[0]: source for source in series._sources}
        if from_year not in by_year or to_year not in by_year:
            raise ContractError(f"Both endpoints must be in this series. Selected years: {series.years}.")
        if series.scope not in {"district", "campus"}:
            raise ContractError("pair_years supports public district and campus observations.")
        endpoints = (by_year[from_year], by_year[to_year])
        key = f"{series.scope}_number"
        fields = [pa.field(key, pa.string(), nullable=False)]
        for prefix, (year, _, selection, _) in zip(("from", "to"), endpoints, strict=True):
            keys = set(selection.artifact.contract["key_columns"])
            if keys not in ({key}, {key, "school_year"}) or not keys <= set(selection.columns):
                raise ContractError(f"pair_years requires one row per {series.scope} and school year. "
                                    f"Publisher keys in {year}: {sorted(keys)}.")
            native = _leaf_schema(selection.artifact.contract)
            for name in keys:
                dtype = native.field(name).type
                if not (pa.types.is_string(dtype) or pa.types.is_large_string(dtype)):
                    raise ContractError(f"pair_years requires a string {name} key in school year {year}.")
            fields.append(pa.field(f"{prefix}_present", pa.bool_(), nullable=False))
            fields.extend(native.field(name).with_name(f"{prefix}_{name}").with_nullable(True)
                          for name in selection.columns)
        names = [field.name for field in fields]
        if len(names) != len(set(names)):
            raise ContractError("pair_years output columns conflict with a publisher field.")
        return cls(from_year, to_year, series, endpoints, key, pa.schema(fields))

    def provenance(self) -> dict[str, Any]:
        """Describe alignment and the two pinned source projections without reading them."""
        selector = self._series.provenance()
        result = {
            "schema_version": 1,
            "operation": "school_year_pair",
            "dataset": selector["dataset"],
            "scope": selector["scope"],
            "fields": selector["fields"],
            "from_year": self.from_year,
            "to_year": self.to_year,
            "ids": selector["ids"],
            "cohort": "requested_ids_in_order" if self._series.ids is not None else "sorted_union_of_publisher_ids",
            "alignment_key": self._key,
            "key_normalization": "bare_zero_padded_public_id",
            "row_validation": "both_complete_endpoint_projections",
            "cross_year_comparability": "unassessed",
            "historical_entity_continuity": "unassessed",
            "endpoints": {
                prefix: {"school_year": year, "present_column": f"{prefix}_present",
                         "columns": {name: f"{prefix}_{name}" for name in selection.columns},
                         "provenance": source.to_dict()}
                for prefix, (year, _, selection, source) in zip(("from", "to"), self._endpoints, strict=True)
            },
        }
        if "name_selection" in selector:
            result["name_selection"] = selector["name_selection"]
        return result

    def _read_endpoint(self, endpoint: tuple[int, SourceTable, _ExportSelection, ExportProvenance]
                       ) -> tuple[pa.Table, pa.ChunkedArray]:
        import pyarrow as pa
        import pyarrow.compute as arrow_compute

        pc: Any = arrow_compute
        year, view, selected, _ = endpoint
        period = view.measurement_period
        assert period is not None
        # Read each complete endpoint once. A cohort may not hide invalid keys.
        table = _open_export(view.release, selected, period=period, entity_build=view._entity_build).to_arrow()
        width = 6 if self._key == "district_number" else 9
        try:
            normalized = canonical_source_identifiers(table[self._key], width=width)
        except ValueError as error:
            raise ContractError(f"Invalid {self._key} in school year {year}: {error}") from error
        ids = pc.utf8_slice_codeunits(normalized, start=1).cast(pa.string())
        if ids.null_count or len(pc.unique(ids)) != table.num_rows:
            raise ContractError(f"Null or duplicate canonical {self._key} in school year {year}.")
        if "school_year" in selected.artifact.contract["key_columns"]:
            published = table["school_year"]
            if published.null_count or pc.any(pc.not_equal(published, period.label)).as_py():
                raise ContractError(f"Publisher school_year key does not match {period.label}.")
        return table, ids

    def to_arrow(self) -> pa.Table:
        """Read the two selected leaves and align their source values and statuses."""
        import pyarrow as pa
        import pyarrow.compute as arrow_compute

        pc: Any = arrow_compute
        before, after = (self._read_endpoint(endpoint) for endpoint in self._endpoints)
        if self._series.ids is not None:
            cohort = pa.array(self._series.ids, type=pa.string())
        else:
            union = pc.unique(pa.chunked_array([*before[1].chunks, *after[1].chunks], type=pa.string()))
            cohort = pc.take(union, pc.sort_indices(union))
        columns = [cohort]
        for table, ids in (before, after):
            offsets = pc.index_in(cohort, value_set=ids)
            columns.extend([pc.is_valid(offsets), *table.take(offsets).columns])
        schema = self.schema.with_metadata({_PROVENANCE_KEY: canonical_json(self.provenance()).encode()})
        return pa.Table.from_arrays(columns, schema=schema)

    def to_df(self) -> pd.DataFrame:
        """Return paired rows with nullable types and both endpoints' provenance."""
        import pandas as pd

        table = self.to_arrow()
        frame = table.to_pandas(types_mapper=pd.ArrowDtype)
        frame.attrs["teadata_provenance"] = self.provenance()
        return frame
