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

Source datasets are separate from the checkout; an existing snapshot is enough
for runtime use. Maintenance scripts default to `teadata/data/` beside the
installed package, independent of the current working directory. Set
`TEADATA_DATA_DIR` to use another dataset tree, or pass the scripts' explicit
input/output options. See [Configuration & Resolution](docs/config.md#maintenance-script-paths)
for examples and onboarding details.

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

- `DataEngine.from_snapshot(...)` loads catalog-bound snapshots with verified release framing.
- One release manifest selects and verifies every artifact; independent discovery is retired.
- `District` and `Campus` support dynamic metadata attributes through `meta`.
- `Campus.to_dict()` always includes `percent_enrollment_change` (numeric when available, otherwise `"N/A"`).

## Snapshot and Asset Behavior

`teadata` is intentionally cache-first.

Source definitions can include dataset-specific reference periods, publisher
details, URLs, citations, and provenance record identifiers. Use
`cfg.resolve_dataset(...)` to query configured metadata; use `ProvenanceStore`
to preserve immutable acquisition and processing history with archived file
bytes. Unknown historical download dates remain unknown. See
[dataset metadata and provenance](docs/config.md#dataset-metadata-and-provenance).

Artifacts typically used at runtime:

- `snapshot_<build-id>.pkl.gz` (engine snapshot)
- `boundaries_*.sqlite` (boundary WKB sidecar)
- `maps_<build-id>.sqlite` (map payload sidecar)
- `entities_*.sqlite` (entity lookup sidecar)

The authoritative `release.json` binds the snapshot and sidecars to one catalog and verified hashes. Hydrate tracked assets with Git LFS, or configure an immutable remote release URL. Metadata, provenance, masking, estimates, and rebuild instructions are documented in [Dataset contracts and data releases](docs/data-releases.md).

### Environment Variables

- `TEADATA_RELEASE`: local release manifest.
- `TEADATA_RELEASE_URL`: immutable remote release manifest.
- `TEADATA_DATA_DIR`: source data root for scripts and configured `data/...` inputs.
- `TEADATA_CONFIG`, `TEADATA_PROVENANCE_DIR`, `TEADATA_OUTPUT_DIR`: build locations.
- `TEADATA_DISABLE_INDEXES`: disable spatial acceleration indexes.
- `TEADATA_LOG_MEMORY`: log memory snapshots.

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
2. verifies and records selected source files and schema contracts
3. builds districts/campuses from verified inputs
4. applies enrichment datasets
5. validates a staged snapshot and sidecars, then activates `teadata/.cache/release.json`

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

Before packaging trim, `0.0.118` artifacts were above the per-file limit:

- wheel: about `448 MB`
- sdist: about `446 MB`

Current slimmed artifacts are below the limit:

- wheel: `dist/teadata-0.0.118-py3-none-any.whl` about `74 MB`
- sdist: `dist/teadata-0.0.118.tar.gz` about `72 MB`

The Python distribution includes code, source contracts, and local methodology documents. Large data artifacts are versioned separately in Git LFS and selected through a release manifest. Source checkouts can use their hydrated `teadata/.cache/release.json`; installed wheels require `TEADATA_RELEASE` or `TEADATA_RELEASE_URL` pointing to an available bundle. Publishing code does not refresh production data.

## Release Policy

- Versioning uses thousandths-place tags (`v0.0.101`, `v0.0.102`, ...).
- Keep only the 3 most recent release tags/assets.

## License

Apache License 2.0. See [LICENSE](LICENSE).
