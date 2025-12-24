# Configuration & Resolution

All external data locations and schema expectations are declared in a single YAML (or TOML) config.

## Config Structure

Highlights:

- **Per-dataset per-year** entries (2009+)
- **`latest`** and **`current`** shortcuts
- **Schema hints** (allowed file extensions per dataset)
- **Spatial layers** declared alongside tabular sources

Example (excerpt):

```yaml
year_min: 2009

data_sources:
  accountability:
    2025: data/accountability/2025-enhanced-statewide-summary-after-2023-appeal.xlsx
    latest: 2025

spatial:
  districts:
    2025: data/shapes/Current_Districts_2025.geojson
  campuses:
    2025: data/shapes/Schools_2024_to_2025.geojson

schema:
  data_sources:
    accountability: ["parquet","csv","xlsx"]
  spatial:
    districts: ["geojson","gpkg","parquet"]
    campuses: ["geojson","gpkg","parquet"]
```

## Programmatic Access

```python
from teadata.teadata_config import load_config
cfg = load_config("teadata_sources.yaml")

# Resolve “best” file for a year (exact match or nearest prior)
resolved_year, path = cfg["data_sources", "accountability", 2025]
print(resolved_year, path)

# Quick availability report
print(cfg.availability_report())
```

## Cross-dataset Joins

Use the provided helpers to normalize `district_number` and join TAPR/PEIMS/etc. without fighting column name variations.
