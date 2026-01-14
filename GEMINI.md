# teadata Project Context

## Project Overview

**teadata** is a high-performance, spatially-aware Python framework designed for the comprehensive analysis and modeling of Texas public education data (TEA). It provides a robust, object-oriented interface for handling large-scale geographic and demographic datasets.

The framework employs a "snapshot-based" architecture: instead of repeatedly processing raw data, it loads a pre-built, optimized binary snapshot (pickle/gzip) containing the complete state of Districts, Campuses, and Geometries into memory. This facilitates near-instantaneous query execution and complex spatial operations.

**Key Technologies:**
*   **Language:** Python 3.11+
*   **Data Manipulation:** Pandas, NumPy, DuckDB, PyArrow
*   **Geospatial:** Shapely 2.0+, GeoPandas, PyOgrio, SciPy (KDTree)
*   **Build System:** `uv` (recommended), `setuptools`
*   **Testing/Quality:** `pytest`, `ruff`, `ty`

## Architecture & Core Concepts

### 1. DataEngine (`teadata.engine.DataEngine`)
The central hub of the library.
*   **Loading:** Loads data from a snapshot file (defaulting to `.cache/repo_*.pkl`).
*   **Querying:** Provides a fluent interface (`>>`) for filtering districts and campuses.
*   **Spatial:** Manages spatial indexes (`cKDTree` for points, `STRtree` for geometries) to perform fast nearest-neighbor and point-in-polygon queries.
*   **Enrichment:** Facilitates attaching external datasets (finance, accountability ratings) to entities.

### 2. Domain Entities (`teadata.entities`)
*   **`District`**: Represents a school district (ISD). Contains geometry (Polygon/MultiPolygon), enrollment, rating, and metadata.
*   **`Campus`**: Represents a school. Contains location (Point), grade levels, school type, and metadata.
*   **`EntityMap` / `EntityList`**: Specialized collections for holding these objects, providing Pandas-like methods (`.to_df()`, `.value_counts()`).

### 3. Fluent Query DSL
The `>>` operator is overloaded on `DataEngine` to support concise queries:
```python
# Get a district by number
district = engine >> ("district", "101902")

# Get all campuses in that district
campuses = engine >> ("campuses_in", district)

# Find 5 nearest campuses to a point
nearest = engine >> ("nearest", (lon, lat), 5)
```

### 4. Data Sources (`teadata/teadata_sources.yaml`)
Defines the mapping between logical data keys (e.g., `tapr`, `peims`) and physical file paths or URLs.

## Directory Structure

*   `teadata/`: Main package source code.
    *   `engine.py`: Core `DataEngine` logic.
    *   `entities.py`: `District` and `Campus` dataclasses.
    *   `teadata_sources.yaml`: Data source configuration.
    *   `scripts/`: Utilities for downloading/processing raw data.
*   `examples/`: Example scripts and Jupyter notebooks demonstrating usage.
*   `tests/`: `pytest` test suite.
*   `docs/`: MkDocs documentation source.
*   `.cache/`: Snapshot and store artifacts that must be committed.

## Development Workflow

### Versioning Conventions
The project follows a modified Semantic Versioning (SemVer) approach:
*   **Major (X.0.0):** Significant architectural changes, API-breaking updates, or major framework migrations.
*   **Minor (0.X.0):** New features, significant new data source integrations, or changes to the DataEngine query DSL.
*   **Patch (0.0.X):** Bug fixes, minor logic updates, or metadata improvements.
*   **Data Refresh (0.0.Xy):** When a simple data refresh occurs (e.g., running `load_data.py` to pick up latest TEA releases) without significant code changes, append an extra digit (e.g., `0.0.7` -> `0.0.71`).

**Release tag policy:**
*   Tags always use the thousandths place (e.g., `v0.0.101`, `v0.0.102`). If no tags exist, start at `v0.0.101`.
*   Keep only the three most recent tags/releases; delete older tags and their release assets everywhere (GitHub releases included).

**When to increment:**
*   Increment the version BEFORE running a release build or distributing a new snapshot.
*   Always increment the version when `load_data.py` is executed for a production data refresh.

### Installation
The project uses `uv` for dependency management and command execution; do not use bare `python` or `pip`.

```bash
# Recommended
uv sync
```

### Running Tests
```bash
uv run pytest
```

### Linting & Formatting
```bash
uv run ruff check .
uv run ty check .
```

### Building Documentation
```bash
mkdocs serve
```

## Common Tasks

*   **Loading Data:**
    ```python
    from teadata import DataEngine
    # Tries to find the latest snapshot automatically
    engine = DataEngine.from_snapshot()
    ```

*   **Accessing Attributes:**
    Entities use dynamic attribute access via `__getattr__` to expose data stored in their `.meta` dictionary, allowing for flexible schema evolution.

*   **Adding New Data:**
    1.  Add the source definition to `teadata/teadata_sources.yaml`.
    2.  Use/Modify scripts in `teadata/scripts/` to process the raw data into the snapshot format.
