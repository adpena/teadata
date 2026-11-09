"""Query pipeline utilities and operator dispatch."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from operator import index
from typing import (
    Any, Callable, Dict, List, Literal, Mapping, NotRequired, Optional,
    Protocol, Sequence, SupportsFloat, SupportsIndex, Tuple, TYPE_CHECKING, TypedDict, Unpack,
)

if TYPE_CHECKING:
    from .dataset_metadata import EditionKey
    from .engine import DataEngine
    from .release import DataRelease
    from .source_views import PartitionedSourceTable, SourceTable

from .entities import (
    Campus,
    District,
    EntityList,
    EntityMap,
    ReadOnlyEntityView,
    is_charter,
)
from .geometry import point_xy
from .grades import coerce_grade_bounds, coerce_grade_spans

__all__ = [
    "Query",
    "describe_operators",
    "EntityList",
    "EntityMap",
    "ReadOnlyEntityView",
    "unwrap_query",
]


class _ArgumentOptions(TypedDict, total=False):
    default: str | int | float | bool | None
    description: str
    choices: list[str]


class _ArgumentType(TypedDict):
    type: str
    python_only: bool


class _Argument(_ArgumentOptions):
    name: str
    types: list[_ArgumentType]
    optional: bool
    nullable: bool


class _Form(TypedDict):
    syntax: Literal["tuple", "callable"]
    arguments: list[_Argument]


class _Result(TypedDict):
    type: Literal["Query", "SourceTable"]
    alternate_type: NotRequired[Literal["PartitionedSourceTable"]]
    item_type: NotRequired[str]
    tuple_fields: NotRequired[list[dict[str, str]]]


class _Operator(TypedDict):
    name: str
    aliases: list[str]
    description: str
    availability: Literal["built_in", "custom_repository"]
    forms: list[_Form]
    result: _Result
    requires: NotRequired[str]
    errors: NotRequired[list[dict[str, str]]]


class _MethodArgument(_Argument):
    kind: Literal["positional", "varargs", "keyword_only"]


class _SeriesMethod(TypedDict):
    name: Literal["pair_years"]
    receiver: Literal["ObservationSeries"]
    description: str
    arguments: list[_MethodArgument]
    result: Literal["ObservationPair"]
    result_methods: list[str]


class _ReleaseMethod(TypedDict):
    name: str
    receiver: Literal["DataRelease"]
    description: str
    arguments: list[_MethodArgument]
    result: Literal["ObservationSeries"]
    result_methods: list[str]
    result_operations: list[_SeriesMethod]
    discovery: list[str]


class _Discovery(TypedDict):
    schema_version: Literal[1]
    receiver: Literal["Query"]
    operators: list[_Operator]
    release_methods: list[_ReleaseMethod]


class _ExportPreset(TypedDict):
    column_order: Sequence[str]
    rename: NotRequired[Mapping[str, str]]


class _CampusContainment(Protocol):
    def __call__(self, target: object, *, charter_only: bool, covers: bool) -> list[Campus]: ...


def _validate_within_flags(charter_only: object, covers: object) -> None:
    if type(charter_only) is not bool or type(covers) is not bool:
        raise ValueError("within expects Boolean charter_only and covers flags")


@dataclass(frozen=True)
class _RegisteredOperator:
    handler: Callable[[Query, tuple], Any]
    description: _Operator
    tuple_arities: frozenset[int]


def _integer_operand(value: object, label: str) -> int:
    if isinstance(value, SupportsIndex) and not isinstance(value, bool):
        try:
            return index(value)
        except TypeError:
            pass
    raise ValueError(f"{label} must be an integer")


def _boolean_operand(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a Boolean")
    return value


def _number_operand(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (SupportsFloat, SupportsIndex)):
        raise ValueError(f"{label} must be a number")
    return float(value)


def _callable_operand(value: object, label: str) -> Callable[..., Any]:
    if not callable(value):
        raise ValueError(f"{label} must be callable")
    return value


_OPERATORS: dict[str, _RegisteredOperator] = {}
_ALIASES: dict[str, str] = {}


def describe_operators() -> _Discovery:
    """Describe Query shifts and related source access without loading a release.

    Return detached JSON data. Each form lists operands in positional order
    after the operator name, or the bare predicate for callable syntax.
    """
    return {
        "schema_version": 1,
        "receiver": "Query",
        "operators": [deepcopy(operator.description) for operator in _OPERATORS.values()],
        "release_methods": deepcopy(_RELEASE_METHODS),
    }


def unwrap_query(obj: Any) -> Any:
    """If ``obj`` is a Query return its first item; otherwise return ``obj``."""

    if isinstance(obj, Query):
        return obj.first()
    return obj


class Query:
    """Lightweight, chainable view over a list of repository objects."""

    _PRESETS: dict[str, _ExportPreset] = {
        "nearest_charter": {
            "column_order": [
                "campus_name",
                "campus_district_number",
                "campus_campus_number",
                "campus_school_type",
                "match_name",
                "match_district_number",
                "match_campus_number",
                "match_school_type",
                "distance_miles",
            ],
            "rename": {"distance_miles": "miles_to_match"},
        },
        "nearest_charter_same_type": {
            "column_order": [
                "campus_name",
                "campus_district_number",
                "campus_campus_number",
                "campus_school_type",
                "match_name",
                "match_district_number",
                "match_campus_number",
                "match_school_type",
                "distance_miles",
            ],
            "rename": {"distance_miles": "miles_to_match"},
        },
        "nearest_charter_transfer_destination": {
            "column_order": [
                "campus_name",
                "campus_district_number",
                "campus_campus_number",
                "to_name",
                "to_district_number",
                "to_campus_number",
                "to_school_type",
                "count",
                "masked",
            ]
        },
        "transfers_out": {
            "column_order": [
                "campus_name",
                "campus_district_number",
                "campus_campus_number",
                "to_name",
                "to_district_number",
                "to_campus_number",
                "to_school_type",
                "count",
                "masked",
            ]
        },
    }

    def __init__(self, items: List[Any], repo: DataEngine):
        self._items = list(items)
        self._repo = repo

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def to_list(self) -> List[Any]:
        return list(self._items)

    def all(self) -> List[Any]:
        return list(self._items)

    def first(self) -> Optional[Any]:
        return self._items[0] if self._items else None

    def __getattr__(self, name: str):
        if not self._items:
            raise AttributeError(
                f"'Query' object has no attribute '{name}' (empty result set)"
            )
        return getattr(self._items[0], name)

    def __getitem__(self, idx: int) -> Any:
        return self._items[idx]

    def __bool__(self) -> bool:
        return bool(self._items)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        cls = self.__class__.__name__
        n = len(self._items)
        sample = self._items[0].__class__.__name__ if n else "None"
        return f"<{cls} len={n} first={sample}>"

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def to_dicts(
        self,
        *,
        include_meta: bool = True,
        include_geometry: bool = False,
        columns: list[str] | None = None,
    ) -> list[dict]:
        """Materialize the current query items as a list of dictionaries."""

        columns_set = set(columns) if columns is not None else None

        def _include_column(name: str) -> bool:
            return columns_set is None or name in columns_set

        def _allowed_keys(prefix: str) -> set[str] | None:
            if columns_set is None:
                return None
            prefix_len = len(prefix)
            allowed = {col[prefix_len:] for col in columns_set if col.startswith(prefix)}
            return allowed

        def _obj_to_basic_dict(
            obj, *, prefix: str = "", allowed_keys: set[str] | None = None
        ) -> dict:
            if hasattr(obj, "to_dict"):
                d = obj.to_dict(
                    include_meta=include_meta, include_geometry=include_geometry
                )
            else:
                try:
                    d = dict(vars(obj))
                except Exception:
                    d = {"value": obj}
            if allowed_keys is not None:
                d = {k: v for k, v in d.items() if k in allowed_keys}
            if prefix:
                return {f"{prefix}{k}": v for k, v in d.items()}
            return d

        rows = []
        for item in self._items:
            if isinstance(item, tuple):
                is_ncst = (
                    len(item) == 3
                    and getattr(item[0].__class__, "__name__", "") == "Campus"
                    and (
                        item[1] is None
                        or getattr(item[1].__class__, "__name__", "") == "Campus"
                    )
                )
                if is_ncst:
                    campus, match, miles = item
                    row = {}
                    row.update(
                        _obj_to_basic_dict(
                            campus, prefix="campus_", allowed_keys=_allowed_keys("campus_")
                        )
                    )
                    if match is not None:
                        row.update(
                            _obj_to_basic_dict(
                                match,
                                prefix="match_",
                                allowed_keys=_allowed_keys("match_"),
                            )
                        )
                    else:
                        if _include_column("match_id"):
                            row["match_id"] = None
                        if _include_column("match_name"):
                            row["match_name"] = None
                        if _include_column("match_enrollment"):
                            row["match_enrollment"] = None
                        if _include_column("match_rating"):
                            row["match_rating"] = None
                        if _include_column("match_school_type"):
                            row["match_school_type"] = None
                        if _include_column("match_district_number"):
                            row["match_district_number"] = None
                        if _include_column("match_campus_number"):
                            row["match_campus_number"] = None
                    if _include_column("distance_miles"):
                        row["distance_miles"] = miles
                    if columns_set is not None:
                        row = {k: v for k, v in row.items() if k in columns_set}
                    rows.append(row)
                    continue

                is_transfers = (
                    len(item) == 4
                    and getattr(item[0].__class__, "__name__", "") == "Campus"
                    and (
                        item[1] is None
                        or getattr(item[1].__class__, "__name__", "") == "Campus"
                    )
                )
                if is_transfers:
                    campus, to_campus, count, masked = item
                    row = {}
                    row.update(
                        _obj_to_basic_dict(
                            campus, prefix="campus_", allowed_keys=_allowed_keys("campus_")
                        )
                    )
                    if to_campus is not None:
                        row.update(
                            _obj_to_basic_dict(
                                to_campus,
                                prefix="to_",
                                allowed_keys=_allowed_keys("to_"),
                            )
                        )
                    else:
                        if _include_column("to_id"):
                            row["to_id"] = None
                        if _include_column("to_name"):
                            row["to_name"] = None
                        if _include_column("to_enrollment"):
                            row["to_enrollment"] = None
                        if _include_column("to_rating"):
                            row["to_rating"] = None
                        if _include_column("to_school_type"):
                            row["to_school_type"] = None
                        if _include_column("to_district_number"):
                            row["to_district_number"] = None
                        if _include_column("to_campus_number"):
                            row["to_campus_number"] = None
                    if _include_column("count"):
                        row["count"] = count
                    if _include_column("masked"):
                        row["masked"] = masked
                    if columns_set is not None:
                        row = {k: v for k, v in row.items() if k in columns_set}
                    rows.append(row)
                    continue

                row = {}
                for i, elem in enumerate(item):
                    row.update(
                        _obj_to_basic_dict(
                            elem,
                            prefix=f"p{i}_",
                            allowed_keys=_allowed_keys(f"p{i}_"),
                        )
                    )
                if columns_set is not None:
                    row = {k: v for k, v in row.items() if k in columns_set}
                rows.append(row)
                continue

            row = _obj_to_basic_dict(item)
            if columns_set is not None:
                row = {k: v for k, v in row.items() if k in columns_set}
            rows.append(row)

        return rows

    def to_df(
        self,
        columns: list[str] | None = None,
        *,
        include_meta: bool = True,
        include_geometry: bool = False,
        column_order: Sequence[str] | None = None,
        rename: Dict[str, str] | None = None,
        preset: str | None = None,
        restore_original_headers: bool = False,
    ):
        """Materialize the current query items as a pandas DataFrame.

        Parameters
        ----------
        columns : list[str] | None
            Optional subset of columns to retain.
        column_order : Sequence[str] | None
            Preferred ordering applied after materialisation. Missing columns are ignored.
        rename : Dict[str, str] | None
            Column rename mapping applied via DataFrame.rename.
        preset : str | None
            Named preset (e.g., 'nearest_charter_same_type', 'transfers_out') providing
            a default ordering/rename configuration for common tuple outputs.
        restore_original_headers : bool
            Use reviewed publisher labels for snapshot-bound source fields. Other
            columns keep their canonical names. Requires a single entity scope.
        """

        try:
            import pandas as pd
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "pandas is required for .to_df(); install pandas to use this feature"
            ) from exc

        if type(restore_original_headers) is not bool:
            raise ValueError("Original headers must be explicitly enabled or disabled")
        if restore_original_headers and (rename or preset is not None):
            raise ValueError("Original headers cannot be combined with rename or preset")
        data = self.to_dicts(
            include_meta=include_meta,
            include_geometry=include_geometry,
            columns=columns,
        )
        df = pd.DataFrame(data)

        config_order: Sequence[str] | None = None
        config_rename: Dict[str, str] = {}
        if preset is not None:
            key = preset.lower()
            if key not in self._PRESETS:
                raise ValueError(f"Unknown preset '{preset}' for Query.to_df()")
            preset_cfg = self._PRESETS[key]
            config_order = preset_cfg.get("column_order")
            config_rename = dict(preset_cfg.get("rename", {}))

        if rename:
            config_rename.update(rename)
        if config_rename:
            df = df.rename(columns=config_rename)

        order = list(column_order or [])
        if config_order:
            order = (
                list(config_order)
                if not order
                else list(order) + [c for c in config_order if c not in order]
            )
        if columns:
            order = (
                list(columns)
                if not order
                else list(order) + [c for c in columns if c not in order]
            )
        if order:
            cols = [c for c in order if c in df.columns]
            cols += [c for c in df.columns if c not in cols]
            df = df[cols]
        if restore_original_headers:
            from .manifest import ContractError

            catalog = getattr(self._repo, "catalog", None)
            if catalog is None:
                raise ContractError("Original headers require a bound snapshot catalog")
            if all(isinstance(item, District) for item in self._items):
                scope = "district"
            elif all(isinstance(item, Campus) for item in self._items):
                scope = "campus"
            else:
                raise ContractError("Original headers require a single district or campus scope")
            if any(catalog._entity_build_id(item) != catalog.build_id for item in self._items):
                raise ContractError("Original headers require entities bound to the snapshot")
            headers = catalog.bound_field_headers(df.columns, scope=scope)
            labels = [headers[col]["label"] if col in headers else col for col in df.columns]
            if len(set(labels)) != len(labels):
                raise ContractError("Original labels are ambiguous in this selection")
            df.columns = labels
            df.attrs["teadata"] = {"build_id": catalog.build_id, "publisher_headers": headers}
        return df

    # ------------------------------------------------------------------
    # Dispatch implementation
    # ------------------------------------------------------------------
    def __rshift__(self, op):
        if callable(op):
            return Query(self._items, self._repo)._op_filter_callable(op)
        if not (isinstance(op, tuple) and op):
            raise ValueError(f"Unsupported >> operation on Query: {op!r}")

        key = op[0]
        if not isinstance(key, str):
            raise ValueError("Query operator name must be a string")
        operator = _OPERATORS.get(_ALIASES.get(key, key))
        if operator is None:
            raise ValueError(f"Unsupported Query op: {op!r}")
        if len(op) - 1 not in operator.tuple_arities:
            forms = [
                ", ".join(f"[{arg['name']}]" if arg["optional"] else arg["name"] for arg in form["arguments"])
                or "no arguments"
                for form in operator.description["forms"] if form["syntax"] == "tuple"
            ]
            raise ValueError(f"{key} expects positional arguments: {' or '.join(forms)}")
        return operator.handler(Query(self._items, self._repo), op)

    # ---------------------- individual handlers ----------------------
    def observations(self, dataset: str, *fields: str, school_year: int | str | None = None, scope: str | None = None,
                     edition: EditionKey | None = None, release: DataRelease | None = None) -> SourceTable | PartitionedSourceTable:
        """Plan published rows for the selected public entities and a measurement school year."""
        from .dataset_metadata import _parse_school_year, _validate_observation_selection
        from .manifest import ContractError
        from .release import DataRelease, current_release

        year = _parse_school_year(school_year)
        if release is not None and not isinstance(release, DataRelease):
            raise TypeError("release must be a DataRelease opened from a release manifest, or None.")
        if scope is not None and (not isinstance(scope, str) or scope not in ("district", "campus")):
            raise ValueError("scope must be 'district', 'campus', or None to infer from a nonempty Query.")
        identifiers = []
        for item in self._items:
            if isinstance(item, District):
                item_scope, identifier = "district", item.district_number_canon
            elif isinstance(item, Campus) and not item.is_private:
                item_scope, identifier = "campus", item.lookup_key
            else:
                raise ValueError("Observation queries require public districts or campuses")
            if scope is None:
                scope = item_scope
            if scope != item_scope or not identifier:
                raise ValueError("Observation queries require one entity scope and exact TEA numbers")
            identifiers.append(identifier)
        if scope is None:
            raise ValueError("An empty Query needs scope='district' or scope='campus'.")
        _validate_observation_selection(dataset, fields, scope, edition)
        catalog = getattr(self._repo, "catalog", None)
        if catalog is None or not catalog.native_sources:
            raise ContractError("The query's entity build has no attached native sources")
        if any(catalog._entity_build_id(item) != catalog.build_id for item in self._items):
            raise ContractError("Observation queries require entities bound to the snapshot")
        parent = release if release is not None else current_release()
        if parent.build_id != catalog.build_id:
            raise ContractError("Observation queries require the release for their entity build")
        return parent._entity_observations(dataset, fields, school_year=year, scope=scope, edition=edition,
                                           identifiers=tuple(dict.fromkeys(identifiers)))

    def _op_observations(self, op: tuple) -> SourceTable | PartitionedSourceTable:
        if not isinstance(op[3], (tuple, list)):
            raise ValueError("Use ('observations', dataset, school_year, ('field',)) with optional scope and edition. "
                             "The field names must be a list or tuple of distinct nonblank strings.")
        return self.observations(op[1], *op[3], school_year=op[2], scope=op[4] if len(op) >= 5 else None,
                                 edition=op[5] if len(op) == 6 else None)

    def compare_statewide(self, *, release: DataRelease, local: EditionKey,
                          statewide: EditionKey, metric: str = "*",
                          dimensions: Mapping[str, str] | None = None, school_year: int | str | None = None) -> Query:
        """Project one or all reviewed statistics against publisher statewide values."""
        from .statewide import compare_statewide

        return Query(compare_statewide(self._items, release=release, local=local,
                                       statewide=statewide, metric=metric, dimensions=dimensions,
                                       school_year=school_year), self._repo)

    def _op_compare_statewide(self, op: tuple):
        return self.compare_statewide(release=op[1], local=op[2], statewide=op[3],
                                      metric=op[4] if len(op) >= 5 else "*",
                                      dimensions=op[5] if len(op) >= 6 else None,
                                      school_year=op[6] if len(op) == 7 else None)

    def _op_filter_callable(self, predicate: Callable[[Any], bool]):
        self._items = [o for o in self._items if predicate(o)]
        return self

    def _op_filter(self, op: tuple):
        pred = _callable_operand(op[1], "filter predicate")
        return self._op_filter_callable(pred)

    def _op_take(self, op: tuple):
        n = _integer_operand(op[1], "take count")
        self._items = self._items[:n]
        return self

    def _op_sort(self, op: tuple):
        keyfunc = None if op[1] is None else _callable_operand(op[1], "sort key")
        reverse = _boolean_operand(op[2], "sort descending") if len(op) >= 3 else False
        self._items.sort(key=keyfunc, reverse=reverse)
        return self

    def _op_map(self, op: tuple):
        func = _callable_operand(op[1], "map function")
        self._items = [func(o) for o in self._items]
        return self

    def _op_distinct(self, op: tuple):
        keyfunc = _callable_operand(op[1], "distinct key")
        seen = set()
        out = []
        for obj in self._items:
            key = keyfunc(obj)
            if key in seen:
                continue
            seen.add(key)
            out.append(obj)
        self._items = out
        return self

    def _op_grade_overlap(self, op: tuple):
        if len(op) == 2:
            low, high = coerce_grade_bounds(op[1])
        else:
            low, high = coerce_grade_bounds(op[1], op[2])
        if low is None and high is None:
            raise ValueError("grade_overlap received an empty/unknown grade span")
        if low is None or high is None:
            raise ValueError("grade_overlap requires both grade bounds")

        want_low, want_high = low, high

        def _item_grade_spans(obj: Any) -> list[tuple[Optional[int], Optional[int]]]:
            return coerce_grade_spans(obj)

        def _overlaps(obj: Any) -> bool:
            spans = _item_grade_spans(obj)
            for low_val, high_val in spans:
                if low_val is None or high_val is None:
                    continue
                if not (want_high < low_val or high_val < want_low):
                    return True
            return False

        self._items = [o for o in self._items if _overlaps(o)]
        return self

    def _op_campuses_in(self, op: tuple):
        campuses: List[Any] = []
        for item in self._items:
            if isinstance(item, District):
                campuses.extend(self._repo.campuses_in(item))
            elif isinstance(item, Campus):
                campuses.append(item)
        self._items = campuses
        return self

    def _op_privates_within(self, op: tuple):
        if any(not isinstance(item, District) for item in self._items):
            raise ValueError("privates_within requires a district query")
        campuses = {campus.id: campus for district in self._items
                    for campus in self._repo.private_campuses_within(district)}
        self._items = list(campuses.values())
        return self

    def _resolve_seed_coords(self, seed: object) -> Tuple[float, float]:
        if seed is None and self._items:
            seed = self._items[0]
        seed = unwrap_query(seed)
        xy = point_xy(getattr(seed, "point", None))
        if xy is not None:
            return xy
        if isinstance(seed, (list, tuple)) and len(seed) == 2:
            return _number_operand(seed[0], "longitude"), _number_operand(seed[1], "latitude")
        raise ValueError("Spatial query requires coordinates or a seed item with geometry")

    def _op_nearest(self, op: tuple):
        charter = op[0] == "nearest_charter"
        coords = self._resolve_seed_coords(op[1] if len(op) >= 2 else None)
        n = _integer_operand(op[2], "nearest limit") if len(op) >= 3 else 1
        max_m = _number_operand(op[3], "nearest max_miles") if len(op) >= 4 and op[3] is not None else None
        items = self._repo.nearest_campuses(
            coords[0],
            coords[1],
            limit=n,
            charter_only=charter,
            max_miles=max_m,
            geodesic=True,
        )
        return Query(items, self._repo)

    def _op_radius(self, op: tuple):
        coords = self._resolve_seed_coords(op[1])
        miles = _number_operand(op[2], "radius miles")
        limit = _integer_operand(op[3], "radius limit") if len(op) >= 4 and op[3] is not None else None
        charter_only = _boolean_operand(op[4], "radius charter_only") if len(op) >= 5 else False
        items = self._repo.radius_campuses(
            coords[0], coords[1], miles, charter_only=charter_only, limit=limit
        )
        return Query(items, self._repo)

    def _op_knn(self, op: tuple):
        coords = self._resolve_seed_coords(op[1])
        k = _integer_operand(op[2], "knn k")
        charter_only = _boolean_operand(op[3], "knn charter_only") if len(op) >= 4 else False
        items = self._repo.knn_campuses(
            coords[0], coords[1], k, charter_only=charter_only
        )
        return Query(items, self._repo)

    def _op_within(self, op: tuple):
        seed = unwrap_query(op[1])
        target = unwrap_query(op[2])
        charter_only = op[3] if len(op) >= 4 else False
        covers = op[4] if len(op) >= 5 else False
        _validate_within_flags(charter_only, covers)
        if target is None:
            if seed is None and self._items:
                seed = self._items[0]
            if seed is None:
                raise ValueError(
                    "within requires a target polygon/district or a seed item in the chain"
                )
            target = seed
        contain: _CampusContainment = getattr(self._repo, "campuses_within")
        return Query(contain(target, charter_only=charter_only, covers=covers), self._repo)

    def _op_nearest_charter_same_type(self, op: tuple):
        k = _integer_operand(op[1], "nearest_charter_same_type k") if len(op) >= 2 else 1
        campuses: List[Any] = []
        for item in self._items:
            if isinstance(item, Campus):
                campuses.append(item)
            elif isinstance(item, District):
                campuses.extend(self._repo.campuses_in(item))
        if not campuses:
            self._items = []
            return self
        res = self._repo.nearest_charter_same_type(campuses, k=k)
        out = []
        for c in campuses:
            r = res.get(str(c.id), {"match": None, "miles": None})
            out.append((c, r["match"], r["miles"]))
        self._items = out
        return self

    def _op_nearest_charter_transfer_destination(self, op: tuple):
        campuses: List[Any] = []
        for item in self._items:
            if isinstance(item, Campus):
                campuses.append(item)
            elif isinstance(item, District):
                campuses.extend(self._repo.campuses_in(item))
        if not campuses:
            self._items = []
            return self
        res = self._repo.nearest_charter_transfer_destination(campuses)
        out = []
        for c in campuses:
            r = res.get(str(c.id), {"match": None, "miles": None})
            out.append((c, r["match"], r["miles"]))
        self._items = out
        return self

    def _op_transfers_out(self, op: tuple):
        charter_only = False
        pred_to = None
        if len(op) >= 2:
            arg = op[1]
            if isinstance(arg, bool):
                charter_only = arg
            else:
                pred_to = _callable_operand(arg, "transfers_out predicate (or pass a Boolean)")

        campuses = [
            it
            for it in self._items
            if isinstance(it, Campus)
        ]
        rows = []
        for c in campuses:
            for to_c, cnt, masked in self._repo.transfers_out(c):
                if charter_only and not (to_c is not None and is_charter(to_c)):
                    continue
                if pred_to is not None and not pred_to(to_c):
                    continue
                rows.append((c, to_c, cnt, masked))
        self._items = rows
        return self

    def _op_transfers_in(self, op: tuple):
        campuses = [
            it
            for it in self._items
            if isinstance(it, Campus)
        ]
        rows = []
        for c in campuses:
            for from_c, cnt, masked in self._repo.transfers_in(c):
                rows.append((from_c, c, cnt, masked))
        self._items = rows
        return self

    def _op_where_to(self, op: tuple):
        pred = _callable_operand(op[1], "where_to predicate")
        new_items = []
        for it in self._items:
            if isinstance(it, tuple) and len(it) >= 2:
                to_c = it[1]
                if pred(to_c):
                    new_items.append(it)
        self._items = new_items
        return self


def _argument(name: str, *types: str, python_types: tuple[str, ...] = (), nullable: bool = False,
              **options: Unpack[_ArgumentOptions]) -> _Argument:
    return {
        "name": name,
        "types": [
            *(_ArgumentType(type=kind, python_only=False) for kind in types),
            *(_ArgumentType(type=kind, python_only=True) for kind in python_types),
        ],
        "optional": "default" in options,
        "nullable": nullable,
        **options,
    }


def _form(*arguments: _Argument, syntax: Literal["tuple", "callable"] = "tuple") -> _Form:
    return {"syntax": syntax, "arguments": list(arguments)}


def _register(name: str, handler: Callable[[Query, tuple], Any], description: str, *,
              forms: tuple[_Form, ...] = (), aliases: tuple[str, ...] = (), result: _Result,
              availability: Literal["built_in", "custom_repository"] = "built_in",
              requires: str = "", errors: tuple[tuple[str, str], ...] = ()) -> None:
    operator: _Operator = {
        "name": name, "aliases": list(aliases), "description": description,
        "availability": availability,
        "forms": list(forms) or [_form()], "result": result,
    }
    if requires:
        operator["requires"] = requires
    if errors:
        operator["errors"] = [{"type": kind, "condition": condition} for kind, condition in errors]
    arities = set()
    for form in operator["forms"]:
        if form["syntax"] == "tuple":
            arguments = form["arguments"]
            required = max((i for i, arg in enumerate(arguments, start=1) if not arg["optional"]), default=0)
            arities.update(range(required, len(arguments) + 1))
    _OPERATORS[name] = _RegisteredOperator(handler, operator, frozenset(arities))
    _ALIASES.update(dict.fromkeys(aliases, name))


_INPUT_RESULT: _Result = {"type": "Query", "item_type": "input item"}
_CAMPUS_RESULT: _Result = {"type": "Query", "item_type": "Campus"}
_NEAREST_MATCH_RESULT: _Result = {
    "type": "Query", "item_type": "tuple",
    "tuple_fields": [
        {"name": "campus", "type": "Campus"},
        {"name": "match", "type": "Campus | None"},
        {"name": "miles", "type": "float | None"},
    ],
}


def _transfer_result(*, incoming: bool) -> _Result:
    return {
        "type": "Query", "item_type": "tuple",
        "tuple_fields": [
            {"name": "from_campus", "type": "Campus | None" if incoming else "Campus"},
            {"name": "to_campus", "type": "Campus" if incoming else "Campus | None"},
            {"name": "count", "type": "int | None"},
            {"name": "masked", "type": "bool"},
        ],
    }


_PREDICATE = _argument("predicate", python_types=("Callable[[item], bool]",))
_SEED = _argument(
    "seed", "[longitude, latitude]", python_types=("Point-bearing object", "Query"), nullable=True,
    description="A Query seed uses its first item. Pass None to use the first input item's point or coordinate pair.",
)
_NEAREST_FORM = _form(
    _argument("seed", "[longitude, latitude]", python_types=("Point-bearing object", "Query"),
              default=None, nullable=True, description="Omit or pass None to use the first input item's point."),
    _argument("limit", "integer", default=1),
    _argument("max_miles", "number", default=None, nullable=True),
)
_CHARTER_ONLY = _argument("charter_only", "boolean", default=False)

_register(
    "filter", Query._op_filter, "Keep items for which a Python predicate is true.",
    forms=(_form(_PREDICATE), _form(_PREDICATE, syntax="callable")), aliases=("where",), result=_INPUT_RESULT,
)
_register(
    "take", Query._op_take, "Keep the first n items using Python slice semantics.",
    forms=(_form(_argument("n", "integer")),), result=_INPUT_RESULT,
)
_register(
    "sort", Query._op_sort, "Sort items stably.",
    forms=(_form(_argument("key", python_types=("Callable[[item], comparable]",), nullable=True,
                           description="Pass None to use natural ordering."),
                 _argument("descending", "boolean", default=False)),), result=_INPUT_RESULT,
)
_register(
    "map", Query._op_map, "Apply a Python function to each item.",
    forms=(_form(_argument("function", python_types=("Callable[[item], result]",))),),
    aliases=("select",), result={"type": "Query", "item_type": "mapped item"},
)
_SCHOOL_YEAR_INPUT = (
    "Accepts an ending year such as 2025 or 25 and strings such as '2024-25' or '2024–2025'. "
    "Integers 0–99 and two-digit strings mean 2000–2099. A four-digit string selects the literal year."
)

_RELEASE_METHODS: list[_ReleaseMethod] = [{
    "name": "series",
    "receiver": "DataRelease",
    "description": "Plan several exact source fields across explicit measurement school years.",
    "arguments": [
        {"kind": "positional", **_argument("dataset", "nonblank string")},
        {"kind": "varargs", **_argument("fields", "one or more distinct nonblank source field names")},
        {"kind": "keyword_only", **_argument("scope", "publisher scope string")},
        {"kind": "keyword_only", **_argument("years", "nonempty sequence of distinct school years",
                                             description=_SCHOOL_YEAR_INPUT)},
        {"kind": "keyword_only", **_argument("ids", "sequence of integer or string district or campus IDs",
                                             default=None, nullable=True,
                                             description="IDs are zero-padded. A leading Excel apostrophe is optional.")},
        {"kind": "keyword_only", **_argument("editions", python_types=("Mapping[school year, EditionKey]",),
                                             default=None, nullable=True)},
        {"kind": "keyword_only", **_argument("names", "sequence of exact district or campus names",
                                             default=None, nullable=True,
                                             description="Case-insensitive. Use IDs to disambiguate names.")},
        {"kind": "keyword_only", **_argument("entity_release", python_types=("DataRelease",),
                                             default=None, nullable=True,
                                             description="Entity directory for names. Required for native-only releases.")},
    ],
    "result": "ObservationSeries",
    "result_methods": ["for_year", "iter_batches", "to_arrow", "to_df", "provenance", "pair_years"],
    "result_operations": [{
        "name": "pair_years",
        "receiver": "ObservationSeries",
        "description": "Align two selected measurement years by public ID for one-row district or campus sources.",
        "arguments": [
            {"kind": "keyword_only", **_argument("from_year", "integer", "string", description=_SCHOOL_YEAR_INPUT)},
            {"kind": "keyword_only", **_argument("to_year", "integer", "string", description=_SCHOOL_YEAR_INPUT)},
        ],
        "result": "ObservationPair",
        "result_methods": ["provenance", "to_arrow", "to_df"],
    }],
    "discovery": ["catalog.editions", "catalog.variables", "observation_years",
                  "DataEngine.district_named", "DataEngine.campus_named"],
}]

_register(
    "compare_statewide", Query._op_compare_statewide, "Compare each input item and selected metric with Texas.",
    forms=(_form(
        _argument("release", python_types=("DataRelease",)),
        _argument("local", python_types=("EditionKey",)),
        _argument("statewide", python_types=("EditionKey",)),
        _argument("metric", "string", default="*", description="Use * for all reviewed metrics."),
        _argument("dimensions", "mapping of string to string", default=None, nullable=True),
        _argument("school_year", "integer", "string", default=None, nullable=True,
                  description=_SCHOOL_YEAR_INPUT + " Omit or pass None for every reviewed period in the pinned editions."),
    ),),
    result={"type": "Query", "item_type": "StatewideComparison"},
    requires="Matching reviewed editions in one release. Use districts, campuses, or literal region codes for supported STAAR editions.",
    errors=(("ValueError", "The school year or metric has an invalid form."),
            ("TypeError", "The release or either edition has the wrong object type."),
            ("ContractError", "The scope, editions, periods, metrics or selected publisher observations are incompatible.")),
)
_register(
    "observations", Query._op_observations, "Select published observations for one entity scope and measurement year.",
    forms=(_form(
        _argument("dataset", "nonblank string"),
        _argument("school_year", "integer", "string", nullable=True,
                  description=_SCHOOL_YEAR_INPUT + " Use None only when the selected fields share their latest year "
                              "and one publisher edition provides them together."),
        _argument("fields", "nonempty list or tuple of distinct nonempty strings"),
        _argument("scope", "string", choices=["district", "campus"], default=None, nullable=True,
                  description="Infer from input items when omitted. An empty Query needs an explicit scope."),
        _argument("edition", python_types=("EditionKey",), default=None, nullable=True),
    ),),
    result={"type": "SourceTable", "alternate_type": "PartitionedSourceTable"},
    requires="One scope of public district or campus entities bound to a release with the attached native source. "
             "Roster-backed TPRS Student campus queries route across partitions within one report edition.",
    errors=(("ValueError", "Invalid operands, mixed scopes, or missing exact entity identifiers."),
            ("TypeError", "The edition has the wrong object type."),
            ("ContractError", "The entity build or release does not match, or the source, year or edition cannot resolve.")),
)
_register(
    "distinct", Query._op_distinct, "Keep the first item for each key returned by a Python function.",
    forms=(_form(_argument("key", python_types=("Callable[[item], hashable]",))),), result=_INPUT_RESULT,
)
_register(
    "grade_overlap", Query._op_grade_overlap, "Keep items whose published grade spans overlap the selected grade bounds.",
    forms=(
        _form(_argument("grades", "grade token or range", "grade bounds or spans",
                        python_types=("Object with grade range attributes",))),
        _form(_argument("low", "grade token", nullable=True), _argument("high", "grade token", nullable=True)),
    ),
    aliases=("grade_overlaps", "grade_range_overlap"), result=_INPUT_RESULT,
    errors=(("ValueError", "The grade bounds are incomplete or unrecognized."),),
)
_register(
    "campuses_in", Query._op_campuses_in, "Expand districts to their non-private campuses and keep campus inputs.",
    requires="District or campus input items.", result=_CAMPUS_RESULT,
)
_register(
    "privates_within", Query._op_privates_within, "Select private campuses inside or on the input districts' boundaries.",
    requires="District input items with reported boundary geometry.", result=_CAMPUS_RESULT,
    errors=(("ValueError", "The input or district boundary is invalid."),),
)
_register(
    "nearest", Query._op_nearest, "Find up to limit campuses nearest to a seed.",
    forms=(_NEAREST_FORM,), result=_CAMPUS_RESULT,
    errors=(("ValueError", "No seed coordinates or first input point can be resolved."),),
)
_register(
    "nearest_charter", Query._op_nearest, "Find up to limit charter campuses nearest to a seed.",
    forms=(_NEAREST_FORM,), result=_CAMPUS_RESULT,
    errors=(("ValueError", "No seed coordinates or first input point can be resolved."),),
)
_register(
    "radius", Query._op_radius, "Find campuses within an inclusive great-circle distance.",
    forms=(_form(_SEED, _argument("miles", "number"),
                 _argument("limit", "integer", default=None, nullable=True), _CHARTER_ONLY),),
    result=_CAMPUS_RESULT,
)
_register(
    "knn", Query._op_knn, "Find up to k campuses in increasing great-circle distance.",
    forms=(_form(_SEED, _argument("k", "integer"), _CHARTER_ONLY),), result=_CAMPUS_RESULT,
)
_register(
    "within", Query._op_within, "Find campuses physically inside a reported district boundary or polygon.",
    forms=(_form(
        _argument("seed", python_types=("District", "Polygon", "MultiPolygon", "Query"), nullable=True),
        _argument("target", python_types=("District", "Polygon", "MultiPolygon", "Query"), nullable=True,
                  description="Use None to resolve the seed or first input item."),
        _CHARTER_ONLY, _argument("covers", "boolean", default=False,
                                 description="Include points on the polygon boundary."),
    ),),
    requires="A valid WGS84 polygon or a district with reported boundary geometry.",
    errors=(("ValueError", "The flags, target or supplied boundary are invalid."),),
    result=_CAMPUS_RESULT,
)
_register(
    "nearest_charter_same_type", Query._op_nearest_charter_same_type, "Return one nearest same-type peer charter per campus.",
    forms=(_form(_argument("k", "integer", default=1,
                           description="Accepted for compatibility. The result has at most one peer per campus.")),),
    requires="Campus inputs or districts to expand.", result=_NEAREST_MATCH_RESULT,
)
_register(
    "nearest_charter_transfer_destination", Query._op_nearest_charter_transfer_destination,
    "Return the nearest charter that receives transfers from each input campus.",
    requires="Campus inputs or districts to expand. A campus without a matching destination has a null match and distance.",
    result=_NEAREST_MATCH_RESULT,
)
_register(
    "transfers_out", Query._op_transfers_out, "Return the outbound campus transfer rows.",
    forms=(_form(_CHARTER_ONLY),
           _form(_argument("predicate", python_types=("Callable[[Campus | None], bool]",)))),
    requires="Campus inputs. Any destination predicate must handle the values it receives.",
    result=_transfer_result(incoming=False),
)
_register(
    "transfers_in", Query._op_transfers_in, "Return the inbound campus transfer rows.",
    requires="Campus inputs.", result=_transfer_result(incoming=True),
)
_register(
    "where_to", Query._op_where_to, "Filter tuple rows by applying a Python predicate to their second item.",
    forms=(_form(_argument("predicate", python_types=("Callable[[destination], bool]",))),),
    requires="Tuple rows with a second item, such as campus transfer destinations.", result=_INPUT_RESULT,
)
