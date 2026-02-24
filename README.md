# TEA Data Engine (`teadata`)

`teadata` is a snapshot-first Python engine for Texas education data.
It provides:

- `District` and `Campus` domain models
- a fluent query DSL using `>>`
- geospatial lookups (nearest charter, campuses in district boundaries, private-school overlap)
- config-driven enrichment from TAPR, accountability, transfers, PEIMS financials, and closure datasets
- sidecar sqlite stores for fast boundary/map/entity lookup

## Installation

### PyPI

```bash
pip install teadata
```

### Development (recommended)

```bash
git clone https://github.com/adpena/teadata.git
cd teadata
uv sync --all-extras
```

## Quick Start

```python
from teadata import DataEngine

# Preferred runtime path: load the latest discovered snapshot.
engine = DataEngine.from_snapshot(search=True)

# District lookup by district number, campus number, or name.
aldine = engine.get_district("101902")
print(aldine.name)

# Campuses physically inside district boundaries.
for campus in aldine.campuses[:5]:
    print(campus.name, campus.campus_number)
```

## Public API Surface

Primary imports:

```python
from teadata import DataEngine, District, Campus
```

Core behaviors:

- `DataEngine.from_snapshot(...)` supports `.pkl` and `.pkl.gz` snapshots and multiple payload shapes.
- Snapshot discovery checks explicit paths, env vars, package `.cache`, and parent `.cache` directories.
- `District` and `Campus` support dynamic metadata attributes through `meta`.
- `Campus.to_dict()` always includes `percent_enrollment_change` (numeric when available, otherwise `"N/A"`).

## Snapshot and Asset Behavior

`teadata` is intentionally cache-first.

Artifacts typically used at runtime:

- `repo_*.pkl` / `repo_*.pkl.gz` (engine snapshot)
- `boundaries_*.sqlite` (boundary WKB sidecar)
- `map_payloads_*.sqlite` (map payload sidecar)
- `entities_*.sqlite` (entity lookup sidecar)

If snapshot/store files are Git LFS pointers or missing locally, runtime asset resolvers can fetch real files when URL env vars are provided.

### Environment Variables

- `TEADATA_SNAPSHOT`: explicit snapshot path.
- `TEADATA_SNAPSHOT_URL`: URL used when snapshot candidate is missing or a Git LFS pointer.
- `TEADATA_BOUNDARY_STORE`: explicit boundary sqlite path.
- `TEADATA_BOUNDARY_STORE_URL`: URL fallback for boundary store.
- `TEADATA_MAP_STORE`: explicit map sqlite path.
- `TEADATA_MAP_STORE_URL`: URL fallback for map store.
- `TEADATA_ENTITY_STORE`: explicit entity sqlite path.
- `TEADATA_ENTITY_STORE_URL`: URL fallback for entity store.
- `TEADATA_ASSET_CACHE_DIR`: override cache directory used for downloaded assets.
- `TEADATA_DISABLE_INDEXES`: disable default spatial acceleration indexes.
- `TEADATA_LOG_MEMORY`: enable memory snapshot logging.

## Query DSL

`DataEngine` and `Query` chains use `>>`.

```python
# Resolve district then expand to district-operated campuses.
q = engine >> ("district", "ALDINE ISD") >> ("campuses_in",)

# Filter, sort, and take.
top = (
    q
    >> ("filter", lambda c: (c.enrollment or 0) > 1000)
    >> ("sort", lambda c: c.enrollment or 0, True)
    >> ("take", 10)
)

rows = top.to_df(columns=["name", "campus_number", "enrollment"])
```

Supported lookup semantics include:

- case-insensitive district and campus name matching
- wildcard patterns (`*`, `?`, SQL-like `%`/`_`)
- normalized district number handling (for example `"123"` and `"'000123"`)

Spatial and transfer helpers include:

- nearest-campus/nearest-charter queries
- `nearest_charter_same_type(...)`
- transfer graph methods such as `transfers_out(...)` / `transfers_in(...)`

## Enrichment Pipeline

`teadata/enrichment` provides registered enrichers for district and campus datasets.

Included enrichers cover:

- district accountability and district TAPR profile data
- campus accountability, TAPR profile/historical enrollment, PEIMS financials
- planned closure overlays
- charter network augmentation

Pipeline behavior is fault-tolerant by design: dataset-level failures are generally logged and do not hard-stop the full build.

## Data Build Pipeline

`teadata/load_data.py` builds a full `DataEngine` and updates cached artifacts.

```bash
uv run python -m teadata.load_data
```

At a high level, it:

1. resolves year-aware source paths from `teadata/teadata_sources.yaml`
2. warm-loads compatible snapshot cache when signatures match
3. otherwise builds districts/campuses from spatial files
4. applies enrichment datasets
5. writes snapshot + sqlite sidecars back to `.cache/`

## Config and CLI (`teadata-config`)

`teadata/teadata_config.py` provides YAML/TOML config loading, year resolution, schema checks, and dataset joins.

CLI entrypoint:

```bash
uv run teadata-config --help
```

Subcommands:

- `init <out.yaml>`
- `resolve <cfg> <section> <dataset> <year>`
- `report <cfg> [--json] [--min N] [--max N]`
- `join <cfg> <year> [--datasets a,b,c] [--parquet out.parquet] [--duckdb out.duckdb --table t]`

## Testing

```bash
uv run pytest
```

Current tests cover:

- snapshot gzip and fallback loading
- query DSL semantics and chaining
- nearest charter behavior and transfer grouping
- store discovery and asset-cache behavior
- entity serialization invariants (`percent_enrollment_change`)

## PyPI Size Limits and Current Packaging Status

PyPI defaults currently documented at:

- per-file upload limit: `100 MB`
- total project limit: `10 GB`

Reference: <https://docs.pypi.org/project-management/storage-limits/>

Current `teadata` release artifacts for `0.0.118` are above the per-file limit:

- wheel: `dist/teadata-0.0.118-py3-none-any.whl` about `448 MB`
- sdist: `dist/teadata-0.0.118.tar.gz` about `446 MB`

These exceed the default 100 MB file cap because large `.cache` snapshot/store artifacts are packaged into both distributions.

## Release Policy

- Versioning uses thousandths-place tags (`v0.0.101`, `v0.0.102`, ...).
- Keep only the 3 most recent release tags/assets.

## License

Business Source License 1.1. See [LICENSE](LICENSE).
