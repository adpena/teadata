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
from typing import Dict, Any, Optional, Tuple, List, Sequence, Mapping, Iterable
from pathlib import Path
from urllib.parse import urlparse
import os
import sys
import json
import re

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


def _rewrite_relative_paths(raw: dict, base_dir: Path) -> dict:
    """Rewrite relative file paths in data_sources/spatial to be relative to base_dir."""

    def rewrite_value(val: Any) -> Any:
        if not isinstance(val, str):
            return val
        stripped = val.strip()
        if stripped.isdigit() or "://" in stripped:
            return val
        expanded = _expand_path(val)
        path = Path(expanded)
        if path.is_absolute():
            return str(path)
        return str(base_dir / path)

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

    entries: Dict[int, str] = field(default_factory=dict)
    year_min: int = YEAR_MIN_DEFAULT

    @staticmethod
    def from_raw(
        raw: Dict[str, Any], *, section: str, dataset: str, year_min: int
    ) -> "YearMap":
        if not isinstance(raw, dict):
            raise ValueError(
                f"[{section}.{dataset}] must be a mapping of year->path/url"
            )
        entries: Dict[int, str] = {}
        for k, v in raw.items():
            y = _coerce_year_key(k)
            if y is None:
                # allow sugar: 'latest' (int year) or 'current' (direct path/URL)
                if k in {"latest", "current"}:
                    if isinstance(v, (int, str)) and str(v).isdigit():
                        y2 = int(v)
                        if y2 < year_min:
                            raise ValueError(
                                f"[{section}.{dataset}] year {y2} < {year_min}"
                            )
                        entries[y2] = entries.get(y2, "")
                    else:
                        entries[9999] = _expand_path(str(v))  # synthetic latest path
                    continue
                raise ValueError(
                    f"[{section}.{dataset}] invalid key '{k}' (expected year like '2017')"
                )
            if y < year_min:
                raise ValueError(f"[{section}.{dataset}] year {y} < {year_min}")
            if not isinstance(v, (str, int, float)):
                raise ValueError(
                    f"[{section}.{dataset}.{y}] value must be a path or URL string"
                )
            entries[y] = _expand_path(str(v))
        entries = {y: p for y, p in entries.items() if p}
        if not entries:
            raise ValueError(f"[{section}.{dataset}] has no valid year entries")
        return YearMap(entries=entries, year_min=year_min)

    def available_years(self) -> List[int]:
        return sorted(y for y in self.entries.keys() if y != 9999)

    def resolve(self, year: int, *, strict: bool = False) -> Optional[Tuple[int, str]]:
        if year < self.year_min:
            if strict:
                raise KeyError(f"Requested year {year} < {self.year_min}")
            return None
        if year in self.entries:
            return year, self.entries[year]
        prior_years = [
            y for y in self.entries.keys() if isinstance(y, int) and y <= year
        ]
        if prior_years:
            best = max(prior_years)
            return best, self.entries[best]
        if not strict and 9999 in self.entries:
            return 9999, self.entries[9999]
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

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Config":
        if not isinstance(d, dict):
            raise ValueError("Config must be a mapping at the top level.")

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

        schema = SchemaHints.from_dict(schema_raw)

        cfg = Config(
            data_sources=data_sources,
            spatial=spatial,
            options=options,
            schema=schema,
            year_min=year_min,
        )
        cfg.validate_file_types()  # fail fast
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
        year: int,
        *,
        strict: bool = False,
        section: str = "data_sources",
    ) -> Tuple[int, str]:
        catalog = self.data_sources if section == "data_sources" else self.spatial
        if dataset not in catalog:
            raise KeyError(f"{section} does not contain dataset '{dataset}'.")
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
                "data_sources": {k: v.entries for k, v in self.data_sources.items()},
                "spatial": {k: v.entries for k, v in self.spatial.items()},
                "options": self.options,
                "schema": {
                    "data_sources": self.schema.data_sources,
                    "spatial": self.schema.spatial,
                },
            },
            indent=2,
            sort_keys=True,
        )

    # ---------- Validation ----------
    def validate_file_types(self) -> None:
        problems: List[str] = []

        def check_section(section: str, catalog: Dict[str, YearMap]):
            for ds, ymap in catalog.items():
                allowed = self.schema.allowed_exts_for(section, ds)
                if not allowed:
                    continue
                for y, path in ymap.entries.items():
                    base, ext = _basename_and_ext(path)
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

    def load_df(
        self, dataset: str, year: int, *, section: str = "data_sources", **reader_kwargs
    ):
        """
        Resolve the dataset for the year and load as DataFrame (or GeoDataFrame if spatial).
        For CSV loads, you can pass dtype=... etc through reader_kwargs.
        """
        resolved_year, path = self.resolve(dataset, year, section=section)
        prefer_geo = section == "spatial"
        return resolved_year, self._load_table(
            path, prefer_geopandas=prefer_geo, **reader_kwargs
        )


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
    how: str = "outer",
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
    how: str = "outer",
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


def load_config(path: str | Path) -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    raw = _detect_and_load(p)
    raw = _rewrite_relative_paths(raw, p.resolve().parent)
    cfg = Config.from_dict(raw)
    return cfg


# ------------------------------
# CLI helpers
# ------------------------------

_TEMPLATE_YAML = """\
# TEA Data configuration (YAML)
# Each dataset/layer maps school years (2009+) to a file path or URL.
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

    if len(argv) >= 2 and argv[1] == "resolve":
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
        "  teadata_config.py report <config.(yaml|toml)> [--json] [--min N] [--max N]",
        file=sys.stderr,
    )
    print(
        "  teadata_config.py join <config.(yaml|toml)> <year> [--datasets a,b,c] [--parquet out.parquet] [--duckdb out.duckdb --table name]",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main(sys.argv)
