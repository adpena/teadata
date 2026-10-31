#!/usr/bin/env python3
"""
teadata_config.py

Config loader + validator + one-line loader + cross-dataset joiner for TEA projects.

Features:
- YAML/TOML config with per-year dataset entries (>= 2009) and schema/file-type validation
- Resolve best available file for a given year (exact else nearest prior)
- Pythonic sugar: cfg["tapr", 2018], cfg["spatial","districts",2024], cfg.tapr, cfg.peims, etc.
- Auto-load decorator and helpers: return pandas/GeoPandas DataFrames based on file extension
- District Number normalization and cross-dataset join on normalized "district_number"
- Save joined results to Parquet and/or DuckDB
- CLI:
    - init <out.yaml>
    - resolve <cfg> <section> <dataset> <year>
    - report <cfg> [--json] [--min N] [--max N]
    - join <cfg> <year> [--datasets tapr,peims,...] [--parquet out.parquet] [--duckdb out.duckdb --table name]
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List, Sequence, Mapping, Iterable, Literal
from pathlib import Path
from urllib.parse import urlparse
import os
import sys
import json
import re

from teadata.dataset_metadata import DatasetDefinition, DatasetMetadata, ResolvedDataset
from teadata.values import ValuePolicy
from teadata.metric_bindings import MetricBinding, parse_bindings, bindings_document

YEAR_MIN_DEFAULT = 2009

# ------------------------------
# Loading utilities (YAML/TOML)
# ------------------------------


def _load_yaml(text: str) -> dict:
    try:
        import yaml  # PyYAML
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required to read .yaml/.yml configs. pip install pyyaml"
        ) from e
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping (dict).")
    return data


def _load_toml(text: str) -> dict:
    # Prefer stdlib tomllib on 3.11+, fallback to tomli
    try:
        import tomllib

        data = tomllib.loads(text)  # pyright: ignore[reportAttributeAccessIssue]
    except Exception:
        try:
            import tomli

            data = tomli.loads(text)
        except Exception as e:
            raise RuntimeError(
                "tomllib (3.11+) or tomli is required to read .toml configs."
            ) from e
    if not isinstance(data, dict):
        raise ValueError("TOML root must be a mapping (dict).")
    return data


def _detect_and_load(path: Path) -> dict:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in (".yaml", ".yml"):
        return _load_yaml(text)
    if suffix == ".toml":
        return _load_toml(text)
    # Last resort: try YAML first, then TOML
    try:
        return _load_yaml(text)
    except Exception:
        return _load_toml(text)


# ------------------------------
# Helpers
# ------------------------------


def _expand_path(value: str) -> str:
    """Expand ~ and $ENV in a path-like string, but leave URLs untouched."""
    if isinstance(value, str) and ("://" not in value):
        return os.path.expandvars(os.path.expanduser(value))
    return value


def _resolve_source_path(value: str, base_dir: Path) -> str:
    """Resolve one runtime locator without mutating portable reader metadata."""
    if value.strip().isdigit() or "://" in value:
        return value
    path = Path(_expand_path(value))
    if path.is_absolute():
        return str(path)
    if path.parts and path.parts[0] == "data" and os.environ.get("TEADATA_DATA_DIR"):
        return str(Path(_expand_path(os.environ["TEADATA_DATA_DIR"])).resolve().joinpath(*path.parts[1:]))
    return str(base_dir / path)


def _rewrite_relative_paths(raw: dict, base_dir: Path) -> dict:
    """Rewrite relative file paths in data_sources/spatial to be relative to base_dir."""

    def rewrite_value(val: Any) -> Any:
        if isinstance(val, dict):
            updated = dict(val)
            if "path" in updated:
                updated["path"] = rewrite_value(updated["path"])
            return updated
        if not isinstance(val, str):
            return val
        return _resolve_source_path(val, base_dir)

    def rewrite_section(section: dict) -> dict:
        out: dict = {}
        for key, mapping in section.items():
            if not isinstance(mapping, dict):
                out[key] = mapping
                continue
            out[key] = {k: rewrite_value(v) for k, v in mapping.items()}
        return out

    updated = dict(raw)
    for section_key in ("data_sources", "spatial"):
        section = updated.get(section_key)
        if isinstance(section, dict):
            updated[section_key] = rewrite_section(section)
    return updated


def _coerce_year_key(k: Any) -> Optional[int]:
    """Accept int keys, or strings like '2015' -> 2015."""
    if isinstance(k, int):
        return k
    if isinstance(k, str) and k.isdigit():
        return int(k)
    return None


def _basename_and_ext(path_or_url: str) -> tuple[str, str]:
    """Return lowercase (basename, extension including dot). Works for local paths and URLs with querystrings."""
    if "://" in path_or_url:
        p = urlparse(path_or_url)
        base = os.path.basename(p.path.lower())
    else:
        base = os.path.basename(path_or_url.lower())
    _, ext = os.path.splitext(base)
    return base, ext


# ------------------------------
# Data model & validation
# ------------------------------


@dataclass
class YearMap:
    """Mapping of year -> location (path or URL)."""

    entries: Dict[int, DatasetDefinition] = field(default_factory=dict)
    year_min: int = YEAR_MIN_DEFAULT

    @staticmethod
    def from_raw(
        raw: Dict[str, Any], *, section: str, dataset: str, year_min: int
    ) -> "YearMap":
        if not isinstance(raw, dict):
            raise ValueError(
                f"[{section}.{dataset}] must be a mapping of year->path/url"
            )
        entries: Dict[int, DatasetDefinition] = {}
        for k, v in raw.items():
            y = _coerce_year_key(k)
            if y is None:
                if k not in {"latest", "current"}:
                    raise ValueError(
                        f"[{section}.{dataset}] invalid key '{k}' (expected year like '2017')"
                    )
                if isinstance(v, (int, str)) and str(v).isdigit():
                    y2 = int(v)
                    if y2 < year_min:
                        raise ValueError(
                            f"[{section}.{dataset}] year {y2} < {year_min}"
                        )
                    continue
                # A floating current source has no known catalog year.
                y = 9999
            if y < year_min:
                raise ValueError(f"[{section}.{dataset}] year {y} < {year_min}")
            if isinstance(v, dict):
                unknown = set(v) - {"path", "metadata", "reader"}
                if (
                    unknown
                    or not isinstance(v.get("path"), str)
                    or not v["path"].strip()
                ):
                    raise ValueError(
                        f"[{section}.{dataset}.{y}] requires path and optional metadata/reader"
                    )
                reader = v.get("reader", {})
                if not isinstance(reader, dict):
                    raise ValueError("reader must be a mapping")
                entries[y] = DatasetDefinition(
                    _expand_path(v["path"]),
                    DatasetMetadata.from_dict(v.get("metadata")),
                    reader,
                )
            elif isinstance(v, (str, int, float)):
                entries[y] = DatasetDefinition(_expand_path(str(v)))
            else:
                raise ValueError(
                    f"[{section}.{dataset}.{y}] value must be a path or URL string"
                )
        if not entries:
            raise ValueError(f"[{section}.{dataset}] has no valid year entries")
        return YearMap(entries=entries, year_min=year_min)

    def to_dict(self) -> Dict[int, Any]:
        return {year: definition.to_dict() for year, definition in self.entries.items()}

    def available_years(self) -> List[int]:
        return sorted(y for y in self.entries.keys() if y != 9999)

    def resolve(self, year: int, *, strict: bool = False) -> Optional[Tuple[int, str]]:
        if year < self.year_min:
            if strict:
                raise KeyError(f"Requested year {year} < {self.year_min}")
            return None
        if year in self.entries:
            return year, self.entries[year].path
        prior_years = [
            y for y in self.entries.keys() if isinstance(y, int) and y <= year
        ]
        if prior_years:
            best = max(prior_years)
            return best, self.entries[best].path
        if not strict and 9999 in self.entries:
            return 9999, self.entries[9999].path
        if strict:
            raise KeyError(
                f"No data available for {year} (and no prior years) in this dataset."
            )
        return None


# ---------- Schema model ----------

_ALLOWED_EXT_GROUPS: Mapping[str, Sequence[str]] = {
    "csv": (".csv",),
    "parquet": (".parquet",),
    "json": (".json",),
    "geojson": (".geojson", ".json"),
    "gpkg": (".gpkg",),
    "xlsx": (".xlsx", ".xls"),
    "feather": (".feather",),
    "shp": (".shp",),
}


def _normalize_kind(kind: str) -> str:
    return kind.strip().lower()


@dataclass
class SchemaHints:
    """
    Expected file types per dataset/layer.

    Example:
      schema:
        data_sources:
          tapr: ["parquet","csv"]
          peims: ["parquet"]
        spatial:
          districts: ["parquet","geojson","gpkg"]
          campuses: ["parquet","geojson","gpkg"]
    """

    data_sources: Dict[str, List[str]] = field(default_factory=dict)
    spatial: Dict[str, List[str]] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SchemaHints":
        def coerce(section: str) -> Dict[str, List[str]]:
            raw = d.get(section, {}) or {}
            if not isinstance(raw, dict):
                raise ValueError(
                    f"[schema.{section}] must be a mapping of dataset->list[str]"
                )
            out: Dict[str, List[str]] = {}
            for ds, kinds in raw.items():
                if isinstance(kinds, str):
                    kinds = [kinds]
                if not (
                    isinstance(kinds, list) and all(isinstance(k, str) for k in kinds)
                ):
                    raise ValueError(
                        f"[schema.{section}.{ds}] must be a string or list of strings"
                    )
                out[ds] = [_normalize_kind(k) for k in kinds]
            return out

        return SchemaHints(
            data_sources=coerce("data_sources"),
            spatial=coerce("spatial"),
        )

    def allowed_exts_for(self, section: str, dataset: str) -> List[str]:
        """Return flat list of allowed extensions for a dataset/layer; empty means 'no restriction'."""
        mapping = self.data_sources if section == "data_sources" else self.spatial
        kinds = mapping.get(dataset, [])
        exts: List[str] = []
        for k in kinds:
            group = _ALLOWED_EXT_GROUPS.get(k)
            if group:
                exts.extend(group)
        # dedupe preserve order
        seen = set()
        out: List[str] = []
        for e in exts:
            if e not in seen:
                seen.add(e)
                out.append(e)
        return out


# ---------- Config root ----------


@dataclass
class Config:
    data_sources: Dict[str, YearMap] = field(default_factory=dict)
    spatial: Dict[str, YearMap] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)
    schema: SchemaHints = field(default_factory=SchemaHints)
    year_min: int = YEAR_MIN_DEFAULT
    value_policies: Dict[str, ValuePolicy] = field(default_factory=dict)
    metric_bindings: Dict[str, MetricBinding | str] = field(default_factory=dict)
    base_dir: Path | None = field(default=None, repr=False)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Config":
        if not isinstance(d, dict):
            raise ValueError("Config must be a mapping at the top level.")
        if d.get("contract_version", 1) != 1:
            raise ValueError("Unsupported dataset contract version")

        year_min = int(d.get("year_min", YEAR_MIN_DEFAULT))
        data_sources_raw = d.get("data_sources", {})
        spatial_raw = d.get("spatial", {})
        options = d.get("options", {})
        schema_raw = d.get("schema", {})

        if not isinstance(data_sources_raw, dict):
            raise ValueError("[data_sources] must be a mapping.")
        if not isinstance(spatial_raw, dict):
            raise ValueError("[spatial] must be a mapping.")
        if not isinstance(options, dict):
            raise ValueError("[options] must be a mapping.")
        if not isinstance(schema_raw, dict):
            raise ValueError("[schema] must be a mapping (may be empty).")

        data_sources: Dict[str, YearMap] = {}
        for dataset, yearmap_raw in data_sources_raw.items():
            data_sources[dataset] = YearMap.from_raw(
                yearmap_raw, section="data_sources", dataset=dataset, year_min=year_min
            )

        spatial: Dict[str, YearMap] = {}
        for layer, yearmap_raw in spatial_raw.items():
            spatial[layer] = YearMap.from_raw(
                yearmap_raw, section="spatial", dataset=layer, year_min=year_min
            )

        if set(data_sources) & set(spatial):
            raise ValueError(
                "Dataset identifiers must be unique across data_sources and spatial"
            )
        schema = SchemaHints.from_dict(schema_raw)

        policies_raw = d.get("value_policies", {})
        if not isinstance(policies_raw, dict):
            raise ValueError("value_policies must be a mapping")
        policies = {}
        for name, definition in policies_raw.items():
            if (
                not isinstance(definition, dict)
                or definition.get("policy_id", name) != name
            ):
                raise ValueError(
                    "Value-policy identifiers must match their registry keys"
                )
            policies[name] = ValuePolicy.from_dict({"policy_id": name, **definition})

        cfg = Config(
            data_sources=data_sources,
            spatial=spatial,
            options=options,
            schema=schema,
            year_min=year_min,
            value_policies=policies,
            metric_bindings=parse_bindings(d.get("metric_bindings")),
        )
        cfg.validate_file_types()  # fail fast
        for catalog in (cfg.data_sources, cfg.spatial):
            for year_map in catalog.values():
                for definition in year_map.entries.values():
                    policy_id = definition.metadata.value_policy
                    if policy_id and policy_id not in cfg.value_policies:
                        raise ValueError(f"Unknown dataset value policy: {policy_id}")
                    if policy_id:
                        reference = cfg.value_policies[policy_id].reference_id
                        if reference and reference not in {
                            ref.reference_id for ref in definition.metadata.references
                        }:
                            raise ValueError(
                                f"Value policy {policy_id} refers to missing source document {reference}"
                            )
        return cfg

    # ---------- Sugar ----------
    def __getitem__(self, key):
        """
        cfg["tapr", 2018] -> (year, path)
        cfg["spatial", "districts", 2024] -> (year, path)
        """
        if not isinstance(key, tuple):
            raise KeyError("Use cfg[(dataset, year)] or cfg[(section, dataset, year)]")
        if len(key) == 2:
            dataset, year = key
            return self.resolve(dataset, int(year), section="data_sources")
        elif len(key) == 3:
            section, dataset, year = key
            return self.resolve(dataset, int(year), section=section)
        else:
            raise KeyError("Key must be (dataset, year) or (section, dataset, year)")

    @property
    def tapr(self):
        return self.data_sources.get("tapr")

    @property
    def peims(self):
        return self.data_sources.get("peims")

    @property
    def askted(self):
        return self.data_sources.get("askted")

    @property
    def finance(self):
        return self.data_sources.get("finance")

    @property
    def districts(self):
        return self.spatial.get("districts")

    @property
    def campuses(self):
        return self.spatial.get("campuses")

    # ---------- Core APIs ----------
    def resolve(
        self,
        dataset: str,
        year: int | None,
        *,
        strict: bool = False,
        section: str = "data_sources",
    ) -> Tuple[int, str]:
        catalog = self.data_sources if section == "data_sources" else self.spatial
        if dataset not in catalog:
            raise KeyError(f"{section} does not contain dataset '{dataset}'.")
        if year is None:
            years = catalog[dataset].available_years()
            year = max(years) if years else 9999
        resolved = catalog[dataset].resolve(year, strict=strict)
        if not resolved:
            raise KeyError(
                f"No entry for {dataset} in {section} for year {year} (and no prior fallback)."
            )
        return resolved

    def to_json(self) -> str:
        return json.dumps(
            {
                "year_min": self.year_min,
                "contract_version": 1,
                "value_policies": {
                    name: policy.to_dict()
                    for name, policy in self.value_policies.items()
                },
                "metric_bindings": bindings_document(self.metric_bindings),
                "data_sources": {k: v.to_dict() for k, v in self.data_sources.items()},
                "spatial": {k: v.to_dict() for k, v in self.spatial.items()},
                "options": self.options,
                "schema": {
                    "data_sources": self.schema.data_sources,
                    "spatial": self.schema.spatial,
                },
            },
            indent=2,
            sort_keys=True,
        )

    def resolved_metric_bindings(self) -> dict:
        """Freeze build-time source choices before embedding the metric registry."""
        bindings = bindings_document(self.metric_bindings)
        for name, binding in list(bindings.items()):
            if isinstance(binding, dict) and "{private_dataset}" in binding["dataset"]:
                selected = self.options.get("private_dataset")
                if not selected:
                    del bindings[name]
                else:
                    binding["dataset"] = binding["dataset"].replace("{private_dataset}", str(selected))
        return bindings

    def resolve_dataset(
        self,
        dataset: str,
        year: int | None = None,
        *,
        strict: bool = False,
        section: str = "data_sources",
    ) -> ResolvedDataset:
        """Describe a configured source, preserving the actual resolved catalog year.

        This describes configuration, not evidence that a dataset was loaded into
        an engine or snapshot. Legacy entries have an unknown reference period.
        """
        if section not in {"data_sources", "spatial"}:
            raise ValueError("section must be data_sources or spatial")
        catalog = self.data_sources if section == "data_sources" else self.spatial
        if dataset not in catalog:
            raise KeyError(f"Unknown {section} dataset: {dataset}")
        selection_year = year
        if selection_year is None:
            years = catalog[dataset].available_years()
            selection_year = max(years) if years else 9999
        resolved_year, path = self.resolve(
            dataset, selection_year, strict=strict, section=section
        )
        definition = catalog[dataset].entries[resolved_year]
        return ResolvedDataset(
            dataset=dataset,
            section=section,
            requested_year=year,
            resolved_year=None if resolved_year == 9999 else resolved_year,
            path=path,
            metadata=definition.metadata,
            reader=definition.reader,
        )

    # ---------- Validation ----------
    def validate_file_types(self) -> None:
        problems: List[str] = []

        def check_section(section: str, catalog: Dict[str, YearMap]):
            for ds, ymap in catalog.items():
                allowed = self.schema.allowed_exts_for(section, ds)
                if not allowed:
                    continue
                for y, definition in ymap.entries.items():
                    base, ext = _basename_and_ext(definition.path)
                    tag = "latest" if y == 9999 else str(y)
                    if ext not in allowed:
                        problems.append(
                            f"[{section}.{ds}.{tag}] '{base}' has extension '{ext}', expected one of {allowed}"
                        )

        check_section("data_sources", self.data_sources)
        check_section("spatial", self.spatial)

        if problems:
            msg = "File-type validation failed:\n  - " + "\n  - ".join(problems)
            raise ValueError(msg)

    # ---------- Availability report ----------
    def availability_report(
        self, *, year_min: Optional[int] = None, year_max: Optional[int] = None
    ) -> Dict[str, Any]:
        yr_min = int(year_min) if year_min is not None else self.year_min
        declared_years: List[int] = []
        for m in list(self.data_sources.values()) + list(self.spatial.values()):
            declared_years.extend([y for y in m.entries.keys() if y != 9999])
        yr_max = (
            int(year_max)
            if year_max is not None
            else (max(declared_years) if declared_years else yr_min)
        )

        def section_report(catalog: Dict[str, YearMap]) -> Dict[str, Any]:
            section_out: Dict[str, Any] = {}
            for ds, ymap in catalog.items():
                years = sorted([y for y in ymap.entries.keys() if y != 9999])
                present_set = set(years)
                expected_years = list(range(yr_min, yr_max + 1))
                gaps = [y for y in expected_years if y not in present_set]
                section_out[ds] = {
                    "min_declared": min(years) if years else None,
                    "max_declared": max(years) if years else None,
                    "count": len(years),
                    "years": years,
                    "gaps": gaps,
                }
            return section_out

        return {
            "range": {"min": yr_min, "max": yr_max},
            "data_sources": section_report(self.data_sources),
            "spatial": section_report(self.spatial),
        }

    # ---------- Auto-resolve & auto-load ----------
    def _load_table(
        self, path: str, *, prefer_geopandas: bool = False, **reader_kwargs
    ):
        """
        Load a file into pandas (or GeoPandas if spatial and available).
        Detects loader by extension.
        """
        import pandas as pd

        base, ext = _basename_and_ext(path)

        # Spatial formats
        if ext in (".geojson", ".gpkg", ".shp"):
            if prefer_geopandas:
                try:
                    import geopandas as gpd  # optional
                except Exception as e:
                    raise RuntimeError(
                        "geopandas required to load spatial formats (.geojson/.gpkg/.shp)"
                    ) from e
                return gpd.read_file(
                    path,
                    **{k: v for k, v in reader_kwargs.items() if k not in {"dtype"}},
                )
            else:
                # Fallback: load geometry as plain JSON/CSV if possible
                try:
                    import geopandas as gpd

                    return gpd.read_file(path)
                except Exception:
                    raise RuntimeError("Install geopandas to load spatial files.")

        # Tabular formats
        if ext == ".parquet":
            return pd.read_parquet(path, **reader_kwargs)
        if ext == ".feather":
            return pd.read_feather(path, **reader_kwargs)
        if ext == ".csv":
            return pd.read_csv(path, **reader_kwargs)
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(path, **reader_kwargs)
        if ext == ".json":
            # Respect explicit caller preference; otherwise try NDJSON first.
            json_kwargs = dict(reader_kwargs)
            if "lines" in json_kwargs:
                return pd.read_json(path, **json_kwargs)
            try:
                return pd.read_json(path, lines=True, **json_kwargs)
            except ValueError:
                return pd.read_json(path, **json_kwargs)

        raise ValueError(
            f"Unsupported file extension for loader: {ext} (from '{base}')"
        )

    def column_contract(
        self,
        dataset: str,
        year: int | None = None,
        *,
        section: str = "data_sources",
        reader_profile: str | None = None,
    ) -> dict[str, str] | None:
        """Load the exact reviewed reader resource without source-row I/O."""
        from .column_schema import load_column_contract, reader_options
        from .manifest import ContractError

        source = self.resolve_dataset(dataset, year, section=section)
        try:
            options = reader_options(source.reader, reader_profile)
            resource = options.get("column_schema")
            if resource is None:
                return None
            references = [
                reference
                for reference in self.reader_references(dataset, year, section=section, reader_profile=reader_profile)
                if reference.role == "schema" and reference.path == resource
            ]
            if (
                len(references) != 1
                or references[0].status != "available"
                or references[0].reviewed_sha256 is None
            ):
                raise ValueError(
                    "Column schema requires one available byte-reviewed schema reference"
                )
            if self.base_dir is None:
                raise ValueError("Column schema requires a source contract directory")
            return load_column_contract(
                self.base_dir, resource, references[0].reviewed_sha256
            )
        except (OSError, ValueError) as error:
            raise ContractError(f"Invalid column schema for {dataset}: {error}") from error

    def reader_references(self, dataset: str, year: int | None = None, *,
                          section: str = "data_sources", reader_profile: str | None = None):
        """Describe selected source-profile references without changing retained metadata."""
        from .column_schema import reader_options
        from .dataset_metadata import ReferenceDocument

        source = self.resolve_dataset(dataset, year, section=section)
        options = reader_options(source.reader, reader_profile)
        extra = options.get("references", [])
        if not isinstance(extra, list) or any(not isinstance(item, dict) for item in extra):
            raise ValueError("Reader references must be reference-document records")
        selected = [*source.metadata.references, *(ReferenceDocument.from_dict(item) for item in extra)]
        if len({item.reference_id for item in selected}) != len(selected):
            raise ValueError("Reader reference identifiers must be unique")
        return selected

    def column_schema(
        self, dataset: str, year: int | None = None, *, section: str = "data_sources",
        reader_profile: str | None = None,
    ) -> list[str] | None:
        """Derive ordered reader columns from the same reviewed resource."""
        from .column_schema import parse_column_schema, schema_columns
        from .manifest import ContractError

        contract = self.column_contract(dataset, year, section=section, reader_profile=reader_profile)
        if contract is None:
            return None
        source = self.resolve_dataset(dataset, year, section=section)
        try:
            return schema_columns(parse_column_schema(contract["text"].encode()),
                                  dataset=source.dataset, edition=source.metadata.version)
        except ValueError as error:
            raise ContractError(f"Invalid column schema for {dataset}: {error}") from error

    def header_records(self, dataset: str, *, edition: str,
                       table: str | None = None, reader_profile: str | None = None) -> list[dict[str, Any]]:
        """Return reviewed source-contract mappings without any dataset reads.

        This explicitly selects this configuration's contract, independently of
        a data release. Editions are exact metadata versions, never prior-year
        fallbacks or guesses from filenames.
        """
        from .column_schema import parse_column_schema, header_records
        from .manifest import ContractError

        choices = [(section, year) for section in ("data_sources", "spatial")
                   for name, versions in getattr(self, section).items() if name == dataset
                   for year, definition in versions.entries.items() if definition.metadata.version == edition]
        if len(choices) != 1:
            raise ContractError(f"Source-contract header crosswalk edition unavailable or ambiguous: {dataset!r}, {edition!r}")
        section, year = choices[0]
        contract = self.column_contract(dataset, year, section=section, reader_profile=reader_profile)
        if contract is None:
            raise ContractError(f"Source-contract header crosswalk unavailable: {dataset!r}")
        try:
            return header_records(parse_column_schema(contract["text"].encode()),
                                  dataset=dataset, edition=edition, table=table)
        except ValueError as error:
            raise ContractError(str(error)) from error

    def header_mapping(self, dataset: str, *, edition: str, table: str,
                       from_: str = "code", to: str = "field",
                       columns: Iterable[str] | None = None, reader_profile: str | None = None) -> dict[str, str]:
        """Return an ordinary rename map from this explicit source contract."""
        from .column_schema import header_rename_map
        from .manifest import ContractError

        try:
            return header_rename_map(self.header_records(dataset, edition=edition, table=table, reader_profile=reader_profile),
                                     from_=from_, to=to, columns=columns)
        except ValueError as error:
            raise ContractError(str(error)) from error

    def read_tapr_sources(self, dataset: str, year: int, *, input_dir: str | Path | None = None):
        """Explicit full-edition raw preparation, independent of released facts.

        The retained release recorder deliberately continues reading its exact
        prepared inputs. Raw preparation has its own real processing receipt;
        it cannot masquerade as a schema2 release build or replace those bytes.
        """
        from .column_schema import crosswalk_dataset, parse_column_schema, reader_options
        from .manifest import ACTIVE_BUILD, ContractError
        from .tapr import read_tapr_table

        if ACTIVE_BUILD.get() is not None:
            raise ContractError("Prepare full TAPR sources explicitly before a retained-format release build")
        source = self.resolve_dataset(dataset, year)
        if source.resolved_year != year or source.metadata.version != str(year):
            raise ContractError("Full TAPR source preparation requires an exact declared edition")
        options = reader_options(source.reader, "source")
        selection = options.get("tapr_source")
        if (not isinstance(selection, dict) or set(selection) != {"scope", "tables"}
                or selection["scope"] not in {"campus", "district"}
                or not isinstance(selection["tables"], dict)
                or not selection["tables"]
                or any(not isinstance(value, str) or not value or "://" in value
                       for value in selection["tables"].values())):
            raise ContractError("Full TAPR source profile requires an explicit scope and local table paths")
        contract = self.column_contract(dataset, year, reader_profile="source")
        if contract is None:
            raise ContractError("Full TAPR source preparation requires a reviewed crosswalk")
        selected = crosswalk_dataset(parse_column_schema(contract["text"].encode()), dataset, source.metadata.version)
        if set(selection["tables"]) != {table["table"] for table in selected["tables"]}:
            raise ContractError("Full TAPR source paths do not account for every reviewed table")
        result = []
        for table in selected["tables"]:
            path = Path(selection["tables"][table["table"]])
            if input_dir is not None:
                path = Path(input_dir) / path.name
            else:
                path = Path(_resolve_source_path(str(path), self.base_dir or Path.cwd()))
            result.append(read_tapr_table(path, mapping=table, key_field=selection["scope"] + "_number",
                                          dataset=dataset, edition=source.metadata.version))
        return result

    def load_df(
        self,
        dataset: str,
        year: int | None = None,
        *,
        section: str = "data_sources",
        reader_profile: str | None = None,
        **reader_kwargs,
    ):
        """
        Resolve the dataset for the year and load as DataFrame (or GeoDataFrame if spatial).
        For CSV loads, you can pass dtype=... etc through reader_kwargs.
        """
        from .column_schema import reader_options
        from .manifest import ACTIVE_BUILD, ContractError

        source = self.resolve_dataset(dataset, year, section=section)
        try:
            options = reader_options(source.reader, reader_profile)
        except ValueError as error:
            raise ContractError(f"Invalid reader contract for {dataset}: {error}") from error
        expected_schema = self.column_schema(
            dataset, year, section=section, reader_profile=reader_profile
        )
        if (
            reader_profile is None
            and source.reader.get("profiles")
            and expected_schema is None
        ):
            raise ContractError(
                f"Reader profile is required for {dataset}; no default column schema is declared"
            )
        options.pop("column_schema", None)
        options.pop("references", None)
        raw_tapr = options.pop("tapr_source", None)
        if raw_tapr is not None:
            from .tapr import merge_tapr_tables

            if reader_profile != "source" or reader_kwargs:
                raise ContractError("Full TAPR preparation uses its exact source profile without reader overrides")
            selected_year = year if year is not None else source.resolved_year
            if selected_year is None:
                raise ContractError("Full TAPR source preparation requires an exact declared edition")
            tables = self.read_tapr_sources(dataset, selected_year)
            frame = merge_tapr_tables(tables, key_field=raw_tapr["scope"] + "_number").to_pandas()
            if list(frame.columns) != expected_schema:
                raise ContractError("Full TAPR source preparation differs from its complete crosswalk")
            return source.resolved_year, frame
        columns = options.pop("columns", {})
        required = options.pop("required_columns", [])
        options.update(reader_kwargs)
        field_year = source.metadata.field_year
        options = {
            key: value.format(field_year=field_year)
            if isinstance(value, str) and "{field_year}" in value
            else value
            for key, value in options.items()
        }
        build = ACTIVE_BUILD.get()
        identity = build.before_read(source) if build is not None else None
        prefer_geo = section == "spatial"
        frame = self._load_table(source.path, prefer_geopandas=prefer_geo, **options)
        rename = {
            raw.format(field_year=field_year): name for name, raw in columns.items()
        }
        if rename:
            frame = frame.rename(columns=rename)
        if expected_schema is not None and list(frame.columns) != expected_schema:
            raise ContractError(
                f"Source schema changed for {dataset}; review its dictionary, parser, and contract"
            )
        missing = set(required) - set(frame.columns)
        if missing or frame.columns.duplicated().any():
            raise ContractError(
                f"Source schema mismatch for {dataset}; missing={sorted(missing)}, duplicate_columns={bool(frame.columns.duplicated().any())}"
            )
        if build is not None:
            assert identity is not None
            build.after_read(
                source,
                identity,
                len(frame),
                reader_profile,
                {"options": options, "columns": columns, "required_columns": required},
                list(frame.columns),
            )
        return source.resolved_year, frame


# ---------- Decorator: auto-resolve + auto-load ----------


def auto_load(section="data_sources", **default_reader_kwargs):
    """
    Decorator: method(cfg, dataset, year, **kwargs) receives (resolved_year, df) instead of (dataset, year).
    Extra kwargs to the method are preserved. Reader kwargs can be overridden at call-time.

    Usage:
        @auto_load("data_sources")
        def do_something(cfg, resolved, *, reader_kwargs=None): ...
    """

    def decorator(func):
        from functools import wraps

        @wraps(func)
        def wrapper(
            cfg: Config,
            dataset: str,
            year: int,
            *args,
            reader_kwargs: Optional[dict] = None,
            **kwargs,
        ):
            rkw = dict(default_reader_kwargs)
            if reader_kwargs:
                rkw.update(reader_kwargs)
            resolved_year, df = cfg.load_df(dataset, int(year), section=section, **rkw)
            return func(cfg, (resolved_year, df), *args, **kwargs)

        return wrapper

    return decorator


# ---------- District number normalization & joining ----------

_DEFAULT_DISTRICT_ALIASES: List[str] = [
    # common possibilities across TEA sources; extend as needed
    "District Number",
    "DistrictNumber",
    "DISTRICT",
    "DISTRICT_ID",
    "DISTRICT NUMBER",
    "DISTRICT_NBR",
    "DIST_NBR",
    "DistNbr",
    "district_number",
    "districtnumber",
    "district_id",
    "County-District Number",
    "CountyDistrictNumber",
    "county_district",
    "CDN",
    "cds",
    "CDN Number",
    "LEA",
    "LEA_ID",
    "LEA Code",
]

_DISTRICT_RE = re.compile(r"(\d+)")

# ---------- Campus number normalization ----------

# Common possibilities across TEA sources for campus identifiers
_DEFAULT_CAMPUS_ALIASES: List[str] = [
    "Campus Number",
    "CampusNumber",
    "CAMPUS",
    "CAMPUS_ID",
    "CAMPUS NUMBER",
    "CAMPUS_NBR",
    "CAMP_NBR",
    "CampNbr",
    "campus_number",
    "campusnumber",
    "campus_id",
    "CDC",
    "CDC Number",
    "Campus Code",
    "CDS",
    "CDCN",
]

# Reuse the same digit-run regex
_CAMPUS_RE = _DISTRICT_RE


def normalize_campus_number_value(x: Any) -> Optional[str]:
    """
    Normalize any representation of a campus number into a 9-digit zero-padded string (e.g., '011901001').
    Handles:
      - ints/floats ('11901001' / 11901001.0) -> '011901001'
      - strings with apostrophes or spaces ("'011901001", " 11901001 ") -> '011901001'
      - strings with non-digits, keep numeric run ("011901001-XX") -> '011901001'
    Returns None for missing/empty.
    """
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    # Remove Excel-style leading apostrophe
    if s.startswith(("'", "’", "`")):
        s = s[1:].strip()
    # If float-ish like "11901001.0"
    try:
        if re.fullmatch(r"\d+(\.0+)?", s):
            s = str(int(float(s)))
    except Exception:
        pass
    # Extract first digit run
    m = _CAMPUS_RE.search(s)
    if not m:
        return None
    digits = m.group(1)
    # Zero-pad to 9
    return digits.zfill(9)


def normalize_campus_number_column(
    df,
    aliases: Iterable[str] = _DEFAULT_CAMPUS_ALIASES,
    *,
    new_col: str = "campus_number",
):
    """
    Find the first existing alias in df columns, normalize it to 9-digit string into `new_col`.
    Leaves original column intact.
    """
    import pandas as pd

    # Case-insensitive lookup
    lower_map = {c.lower(): c for c in df.columns}
    col_name = None
    for a in aliases:
        if a in df.columns:
            col_name = a
            break
        if a.lower() in lower_map:
            col_name = lower_map[a.lower()]
            break
    if col_name is None and new_col in df.columns:
        col_name = new_col

    if col_name is None:
        # No alias found: create empty and return
        df[new_col] = pd.Series([None] * len(df), dtype="object")
        return df, None

    # Normalize
    df[new_col] = df[col_name].map(normalize_campus_number_value)
    return df, col_name


def normalize_district_number_value(x: Any) -> Optional[str]:
    """
    Normalize any representation of a district number into a 6-digit zero-padded string (e.g., '011901').
    Handles:
      - ints/floats ('11901' / 11901.0) -> '011901'
      - strings with apostrophes or spaces ("'011901", " 11901 ") -> '011901'
      - strings with non-digits, keep numeric run ("011901-001") -> '011901'
    Returns None for missing/empty.
    """
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    # Remove Excel-style leading apostrophe
    if s.startswith(("'", "’", "`")):
        s = s[1:].strip()
    # If float-ish like "11901.0"
    try:
        # Be careful not to nuke leading zeros; only use numeric path when the string is plain number.
        if re.fullmatch(r"\d+(\.0+)?", s):
            s = str(int(float(s)))
    except Exception:
        pass
    # Extract first digit run
    m = _DISTRICT_RE.search(s)
    if not m:
        return None
    digits = m.group(1)
    # Zero-pad to 6
    return digits.zfill(6)


def normalize_district_number_column(
    df,
    aliases: Iterable[str] = _DEFAULT_DISTRICT_ALIASES,
    *,
    new_col: str = "district_number",
):
    """
    Find the first existing alias in df columns, normalize it to 6-digit string into `new_col`.
    Leaves original column intact.
    """
    import pandas as pd

    # Case-insensitive lookup
    lower_map = {c.lower(): c for c in df.columns}
    col_name = None
    for a in aliases:
        if a in df.columns:
            col_name = a
            break
        if a.lower() in lower_map:
            col_name = lower_map[a.lower()]
            break
    if col_name is None and new_col in df.columns:
        col_name = new_col

    if col_name is None:
        # No alias found: create empty and return
        df[new_col] = pd.Series([None] * len(df), dtype="object")
        return df, None

    # Normalize
    df[new_col] = df[col_name].map(normalize_district_number_value)
    return df, col_name


def canonical_campus_number(value: Any) -> Optional[str]:
    """
    Return the repository-preferred representation of a campus number: a
    9-digit, zero-padded string with a leading apostrophe. Returns ``None`` for
    empty inputs.
    """

    normalized = normalize_campus_number_value(value)
    if normalized is None:
        return None
    return f"'{normalized}"


def canonical_district_number(value: Any) -> Optional[str]:
    """
    Return the repository-preferred representation of a district number: a
    6-digit, zero-padded string with a leading apostrophe. Returns ``None`` for
    empty inputs.
    """

    normalized = normalize_district_number_value(value)
    if normalized is None:
        return None
    return f"'{normalized}"


def join_datasets_on_district(
    cfg: Config,
    year: int,
    *,
    datasets: Optional[List[str]] = None,
    aliases: Iterable[str] = _DEFAULT_DISTRICT_ALIASES,
    how: Literal["left", "right", "inner", "outer", "left_anti", "right_anti"] = "outer",
    keep: Optional[Dict[str, List[str]]] = None,
    reader_overrides: Optional[Dict[str, dict]] = None,
):
    """
    Join selected datasets for `year` (resolving to nearest <= year) on normalized 'district_number'.
    - datasets: defaults to ALL cfg.data_sources keys (order matters for suffixing).
    - aliases: column-name candidates to find district number in each dataset.
    - how: 'outer' by default (so you see gaps).
    - keep: optional dict dataset->list of columns to keep (besides the district key);
            if None, keeps all columns (and suffixes to avoid collisions).
    - reader_overrides: dataset->reader_kwargs to pass to loader.

    Returns: (resolved_years: dict[str,int], df)
    """
    import pandas as pd

    ds_list = datasets or list(cfg.data_sources.keys())
    resolved_years: Dict[str, int] = {}
    loaded: List[Tuple[str, pd.DataFrame]] = []

    for ds in ds_list:
        rkw = (reader_overrides or {}).get(ds, {})
        y, df = cfg.load_df(ds, year, section="data_sources", **rkw)
        resolved_years[ds] = y
        # Normalize district key
        df, found_col = normalize_district_number_column(df, aliases=aliases)
        # Optionally subselect columns
        if keep and ds in keep:
            cols = list(
                dict.fromkeys(
                    ["district_number"] + [c for c in keep[ds] if c in df.columns]
                )
            )
            df = df[cols]
        else:
            # Ensure we don't explode duplicates before merges
            pass
        # Suffix non-key columns to avoid collisions (except if keep explicitly trims them)
        nonkey = [c for c in df.columns if c != "district_number"]
        df = df.rename(columns={c: f"{c}__{ds}" for c in nonkey})
        loaded.append((ds, df))

    # Reduce outer-join on district_number
    def _merge(a, b):
        _, left = a
        ds_b, right = b
        return (ds_b, left.merge(right, on="district_number", how=how))

    if not loaded:
        raise ValueError("No datasets selected to join.")
    # Start merge chain
    _, merged = loaded[0]
    for item in loaded[1:]:
        _, merged = _merge((None, merged), item)

    # Sort columns: key first
    cols_sorted = ["district_number"] + [
        c for c in merged.columns if c != "district_number"
    ]
    merged = merged[cols_sorted]

    return resolved_years, merged


# ---------- Campus join ----------


def join_datasets_on_campus(
    cfg: Config,
    year: int,
    *,
    datasets: Optional[List[str]] = None,
    aliases: Iterable[str] = _DEFAULT_CAMPUS_ALIASES,
    how: Literal["left", "right", "inner", "outer", "left_anti", "right_anti"] = "outer",
    keep: Optional[Dict[str, List[str]]] = None,
    reader_overrides: Optional[Dict[str, dict]] = None,
):
    """
    Join selected datasets for `year` (resolving to nearest <= year) on normalized 'campus_number'.
    - datasets: defaults to ALL cfg.data_sources keys (order matters for suffixing).
    - aliases: column-name candidates to find campus number in each dataset.
    - how: 'outer' by default (so you see gaps).
    - keep: optional dict dataset->list of columns to keep (besides the campus key);
            if None, keeps all columns (and suffixes to avoid collisions).
    - reader_overrides: dataset->reader_kwargs to pass to loader.

    Returns: (resolved_years: dict[str,int], df)
    """
    import pandas as pd

    ds_list = datasets or list(cfg.data_sources.keys())
    resolved_years: Dict[str, int] = {}
    loaded: List[Tuple[str, pd.DataFrame]] = []

    for ds in ds_list:
        rkw = (reader_overrides or {}).get(ds, {})
        y, df = cfg.load_df(ds, year, section="data_sources", **rkw)
        resolved_years[ds] = y
        # Normalize campus key
        df, found_col = normalize_campus_number_column(df, aliases=aliases)
        # Optionally subselect columns
        if keep and ds in keep:
            cols = list(
                dict.fromkeys(
                    ["campus_number"] + [c for c in keep[ds] if c in df.columns]
                )
            )
            df = df[cols]
        # Suffix non-key columns to avoid collisions (except if keep explicitly trims them)
        nonkey = [c for c in df.columns if c != "campus_number"]
        df = df.rename(columns={c: f"{c}__{ds}" for c in nonkey})
        loaded.append((ds, df))

    if not loaded:
        raise ValueError("No datasets selected to join.")

    # Reduce outer-join on campus_number
    _, merged = loaded[0]
    for item in loaded[1:]:
        _, right = item
        merged = merged.merge(right, on="campus_number", how=how)

    # Sort columns: key first
    cols_sorted = ["campus_number"] + [
        c for c in merged.columns if c != "campus_number"
    ]
    merged = merged[cols_sorted]

    return resolved_years, merged


def save_parquet(
    df, path: str, *, engine: str = "pyarrow", compression: str = "snappy"
):
    df.to_parquet(path, engine=engine, compression=compression, index=False)
    return path


def save_duckdb(df, db_path: str, *, table: str = "teadata_year"):
    import duckdb

    con = duckdb.connect(db_path)
    con.register("df_tmp", df)
    con.execute(
        f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM df_tmp WITH NO DATA;"
    )
    con.execute(f"INSERT INTO {table} SELECT * FROM df_tmp;")
    con.close()
    return db_path, table


# ------------------------------
# Public API
# ------------------------------


def load_config(path: str | Path | None = None) -> Config:
    """Load an explicit source contract, defaulting to the packaged metadata.

    This never selects or fetches a data release. Source files need not exist
    for schema/crosswalk discovery; rows are read only by load_df.
    """
    p = Path(path) if path is not None else Path(__file__).with_name("teadata_sources.yaml")
    if not p.exists():
        raise FileNotFoundError(p)
    raw = _detect_and_load(p)
    raw = _rewrite_relative_paths(raw, p.resolve().parent)
    cfg = Config.from_dict(raw)
    cfg.base_dir = p.resolve().parent
    return cfg


# ------------------------------
# CLI helpers
# ------------------------------

_TEMPLATE_YAML = """\
# TEA Data configuration (YAML)
# Each dataset/layer maps catalog years (2009+) to a file path or URL.
# Entries can also be mappings with path and metadata; reference periods must be
# declared explicitly. A catalog key alone does not establish a school year.
# 'schema' declares expected file types per dataset/layer for validation.

year_min: 2009

data_sources:
  tapr:
    2009: data/tapr/2009.csv
    2015: data/tapr/2015.csv
    2020: data/tapr/2020.parquet
    2023: https://example.org/tapr/2023.parquet
    latest: 2023
  peims:
    2009: data/peims/2009.csv
    2019: data/peims/2019.parquet
    2024: data/peims/2024.parquet
  askted:
    2012: data/askted/2012.json
    2021: data/askted/2021.json
    current: https://example.org/askted/current.json
  finance:
    2009: data/finance/2009.xlsx
    2016: data/finance/2016.parquet
    2022: data/finance/2022.parquet

spatial:
  districts:
    2013: shapes/districts_2013.geojson
    2018: shapes/districts_2018.gpkg
    2024: shapes/districts_2024.parquet
  campuses:
    2013: shapes/campuses_2013.geojson
    2018: shapes/campuses_2018.gpkg
    2024: shapes/campuses_2024.parquet

# Declare expected file types to catch mistakes (extensions are validated).
schema:
  data_sources:
    tapr: ["parquet", "csv"]
    peims: ["parquet", "csv"]
    askted: ["json"]
    finance: ["parquet", "xlsx"]
  spatial:
    districts: ["parquet", "geojson", "gpkg"]
    campuses: ["parquet", "geojson", "gpkg"]

options:
  cache: true
  log_level: INFO
"""


def _cmd_init(out_path: str) -> None:
    p = Path(out_path)
    if p.exists():
        print(f"Refusing to overwrite existing file: {p}", file=sys.stderr)
        sys.exit(2)
    p.write_text(_TEMPLATE_YAML, encoding="utf-8")
    print(f"Wrote starter config with schema hints: {p}")


def _cmd_resolve(cfg_path: str, section: str, dataset: str, year: int) -> None:
    cfg = load_config(cfg_path)
    if section not in {"data_sources", "spatial"}:
        print("section must be 'data_sources' or 'spatial'", file=sys.stderr)
        sys.exit(2)
    resolved_year, path = cfg.resolve(dataset, year, section=section)
    print(
        json.dumps(
            {
                "section": section,
                "dataset": dataset,
                "request_year": year,
                "resolved_year": resolved_year,
                "path": path,
            },
            indent=2,
        )
    )


def _format_report_table(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    rng = report["range"]
    lines.append(f"Coverage report: {rng['min']}–{rng['max']}\n")

    def section_block(title: str, data: Dict[str, Any]):
        lines.append(title)
        lines.append("-" * len(title))
        for ds, info in sorted(data.items()):
            gaps = info["gaps"]
            min_decl = info["min_declared"]
            max_decl = info["max_declared"]
            lines.append(
                f"{ds}: {info['count']} years (min={min_decl}, max={max_decl})"
            )
            if gaps:
                # compress gaps into runs
                runs = []
                start = prev = gaps[0]
                for y in gaps[1:]:
                    if y == prev + 1:
                        prev = y
                    else:
                        runs.append((start, prev))
                        start = prev = y
                runs.append((start, prev))
                run_txt = ", ".join([f"{a}-{b}" if a != b else f"{a}" for a, b in runs])
                lines.append(f"  gaps: {run_txt}")
            else:
                lines.append("  gaps: (none)")
        lines.append("")

    section_block("DATA SOURCES", report["data_sources"])
    section_block("SPATIAL", report["spatial"])
    return "\n".join(lines)


def _cmd_report(
    cfg_path: str, *, as_json: bool, year_min: Optional[int], year_max: Optional[int]
) -> None:
    cfg = load_config(cfg_path)
    rep = cfg.availability_report(year_min=year_min, year_max=year_max)
    if as_json:
        print(json.dumps(rep, indent=2))
    else:
        print(_format_report_table(rep))


def _cmd_join(
    cfg_path: str,
    year: int,
    *,
    datasets: Optional[List[str]],
    out_parquet: Optional[str],
    out_duckdb: Optional[str],
    duckdb_table: str,
) -> None:
    cfg = load_config(cfg_path)
    resolved_years, df = join_datasets_on_district(cfg, year, datasets=datasets)
    meta = {
        "requested_year": year,
        "resolved_years": resolved_years,
        "rows": len(df),
        "cols": len(df.columns),
    }
    print(json.dumps(meta, indent=2))
    if out_parquet:
        save_parquet(df, out_parquet)
        print(f"Wrote Parquet: {out_parquet}")
    if out_duckdb:
        save_duckdb(df, out_duckdb, table=duckdb_table)
        print(f"Appended to DuckDB: {out_duckdb} (table={duckdb_table})")


def main(argv: List[str]) -> None:
    if len(argv) >= 2 and argv[1] == "init":
        if len(argv) != 3:
            print("Usage: teadata_config.py init <output.yaml>", file=sys.stderr)
            sys.exit(2)
        _cmd_init(argv[2])
        return

    if len(argv) >= 2 and argv[1] in {"resolve", "describe"}:
        if len(argv) != 6:
            print(
                "Usage: teadata_config.py resolve <config.(yaml|toml)> <section> <dataset> <year>",
                file=sys.stderr,
            )
            print(
                "Example: teadata_config.py resolve config.yaml data_sources tapr 2017",
                file=sys.stderr,
            )
            sys.exit(2)
        _, _, cfg_path, section, dataset, year_s = argv
        try:
            year = int(year_s)
        except ValueError:
            print("year must be an integer like 2021", file=sys.stderr)
            sys.exit(2)
        if argv[1] == "describe":
            print(
                json.dumps(
                    load_config(cfg_path)
                    .resolve_dataset(dataset, year, section=section)
                    .to_dict(),
                    indent=2,
                )
            )
        else:
            _cmd_resolve(cfg_path, section, dataset, year)
        return

    if len(argv) >= 2 and argv[1] == "report":
        if len(argv) < 3:
            print(
                "Usage: teadata_config.py report <config.(yaml|toml)> [--json] [--min N] [--max N]",
                file=sys.stderr,
            )
            sys.exit(2)
        cfg_path = argv[2]
        as_json = "--json" in argv

        def _get_flag_val(flag: str) -> Optional[int]:
            if flag in argv:
                i = argv.index(flag)
                if i + 1 < len(argv):
                    try:
                        return int(argv[i + 1])
                    except ValueError:
                        print(f"{flag} expects an integer", file=sys.stderr)
                        sys.exit(2)
            return None

        year_min = _get_flag_val("--min")
        year_max = _get_flag_val("--max")
        _cmd_report(cfg_path, as_json=as_json, year_min=year_min, year_max=year_max)
        return

    if len(argv) >= 2 and argv[1] == "join":
        # teadata_config.py join cfg.yaml 2021 [--datasets tapr,peims,finance] [--parquet out.parquet] [--duckdb out.duckdb --table name]
        if len(argv) < 4:
            print(
                "Usage: teadata_config.py join <config.(yaml|toml)> <year> [--datasets a,b,c] [--parquet out.parquet] [--duckdb out.duckdb --table name]",
                file=sys.stderr,
            )
            sys.exit(2)
        cfg_path = argv[2]
        try:
            year = int(argv[3])
        except ValueError:
            print("year must be an integer like 2021", file=sys.stderr)
            sys.exit(2)
        datasets = None
        if "--datasets" in argv:
            i = argv.index("--datasets")
            if i + 1 < len(argv):
                datasets = [s.strip() for s in argv[i + 1].split(",") if s.strip()]
        out_parquet = None
        if "--parquet" in argv:
            i = argv.index("--parquet")
            if i + 1 < len(argv):
                out_parquet = argv[i + 1]
        out_duckdb = None
        duckdb_table = "teadata_year"
        if "--duckdb" in argv:
            i = argv.index("--duckdb")
            if i + 1 < len(argv):
                out_duckdb = argv[i + 1]
        if "--table" in argv:
            i = argv.index("--table")
            if i + 1 < len(argv):
                duckdb_table = argv[i + 1]
        _cmd_join(
            cfg_path,
            year,
            datasets=datasets,
            out_parquet=out_parquet,
            out_duckdb=out_duckdb,
            duckdb_table=duckdb_table,
        )
        return

    print("Usage:", file=sys.stderr)
    print("  teadata_config.py init <output.yaml>", file=sys.stderr)
    print(
        "  teadata_config.py resolve <config.(yaml|toml)> <section> <dataset> <year>",
        file=sys.stderr,
    )
    print(
        "  teadata_config.py describe <config.(yaml|toml)> <section> <dataset> <year>",
        file=sys.stderr,
    )
    print(
        "  teadata_config.py report <config.(yaml|toml)> [--json] [--min N] [--max N]",
        file=sys.stderr,
    )
    print(
        "  teadata_config.py join <config.(yaml|toml)> <year> [--datasets a,b,c] [--parquet out.parquet] [--duckdb out.duckdb --table name]",
        file=sys.stderr,
    )
    sys.exit(2)


def cli() -> None:
    main(sys.argv)


if __name__ == "__main__":
    cli()
