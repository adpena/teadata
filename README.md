# TEA Data Engine (`teadata`)

`teadata` is a snapshot-first Python engine for Texas education data.
It provides:

- `District` and `Campus` domain models
- a fluent query DSL using `>>`
- geospatial lookups (nearest charter, campuses in district boundaries, private-school overlap)
- config-driven enrichment from TAPR, accountability, transfers, PEIMS financials, and closure datasets
- sidecar sqlite stores for fast boundary/map/entity lookup

## Installation

Python 3.11 through 3.15 are supported. Development and local checks default to
the exact Python version in `.python-version` (currently 3.14.10).
`TEADATA_CHECK_PYTHON` overrides that default for compatibility checks. The lock
retains interpreter-specific versions where dependency compatibility requires
them. On macOS 3.15, the optional documentation tools build
Watchdog's native FSEvents extension and require the
[documented Xcode tools](https://pypi.org/project/watchdog/#dependencies).
PyYAML's supported pure Python implementation is tested on 3.15.

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

Source datasets are separate from the checkout; a verified release bundle is enough
for runtime use. Maintenance scripts default to `teadata/data/` beside the
installed package, independent of the current working directory. Set
`TEADATA_DATA_DIR` to use another dataset tree, or pass the scripts' explicit
input/output options. See [Configuration & Resolution](docs/config.md#maintenance-script-paths)
for examples and onboarding details.

### Dependency security

`pyproject.toml` publishes security floors for the core and each optional extra;
`uv.lock` retains the reviewed versions across supported interpreters. Validate
the selected environment with `scripts/check`; changing the lock also requires
a new producer-bound data release before publishing replacement artifacts.

The notebook extra uses Bleach 6.4 for its published fixes. Its opt-in
`parse_email=True` linkifier still has a known
[regex denial-of-service issue](https://github.com/mozilla/bleach/security/advisories/GHSA-g75f-g53v-794x).
TEA Data does not enable that path, and its nbconvert dependency uses Bleach's
HTML/CSS sanitization. The upstream limitation remains relevant to notebook
code that explicitly enables email linkification; upgrading Bleach alone does
not resolve it.

### Existing relational stores

Publisher rating categories are stored as `TEXT`, including full labels such as
`Not Rated: Minimum Size rules`. Existing PostgreSQL stores with `VARCHAR(16)`
rating columns need a caller-owned schema migration widening **both**
`districts.rating` and `campuses.rating` to `TEXT`, or a fresh destination.
`ensure_schema()` creates missing tables; it does not alter existing columns.
TEA Data does not migrate user databases automatically or truncate these labels.
Exports leave transaction commit and rollback to the caller.

### Campus identity and affiliation

`Campus.lookup_key` and `DataEngine.campus_by_key()` use nine-digit TEA campus
numbers for public schools and source-qualified keys such as
`tefa:rec1kXXvIucG7UkoA` for private schools. Private source IDs retain their exact
case and prefixes; URI-component encoding preserves other characters. The
shared identifiers helper validates the encoded ASCII key's 1024-byte limit.
`campus_number` is null for private schools. Their `source_namespace` and
`source_id` are explicit fields, and their governing `district_id` and
`district_number` are null. A directory's reported district is retained separately
as an unverified directory claim, never as governance.

Public campus UUIDs, numbers, and URLs remain unchanged. Private UUIDs derive
from their source-qualified identity. Existing entity, map, and snapshot bundles
using fabricated private TEA numbers require a source-bound rebuild; no numeric
alias is provided. Analytical schema 2 uses the exact campus lookup key as
`entity_id`, with separate nullable TEA-number and source-identity attributes.
Existing relational destinations also need the caller-owned identity-schema
migration or a fresh destination; `ensure_schema()` does not alter old tables.

Private comparisons use `private_campuses_within()`, which includes points inside
or on district boundaries. The former private district-membership operator is
retired. Private profile maps report every covering traditional public district
(`INDEPENDENT` or `COMMON`) and the boundary dataset/version; overlaps are not
collapsed to a chosen parent. Charter service areas are not attendance-boundary
comparators. Schools without coordinates remain queryable, with no geographic
comparison or guessed district.

The optional native PostgreSQL regression uses `TEADATA_TEST_POSTGRES_URL`
pointing to a fresh empty `teadata_test_*` database. It is separate from the
default SQLite checks because SQLite does not enforce declared `VARCHAR` widths.

## Quick Start

```python
from teadata import DataEngine

# Load the configured, hash-verified release.
engine = DataEngine.load_default()

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

Required source, schema, methodology, and transfer-join failures stop publication.
The previous release remains active when a staged build fails. Geometry coverage
and pending source review remain explicit in the processing receipt.

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
scripts/check
scripts/check --verify-latest
```

Local checks are the primary verification path. They run Ruff, ty, deptry, and
pytest in a locked, isolated uv environment without replacing the development
`.venv`. Logs and immutable per-run JSON receipts stay in ignored `.local/checks/`.
Verification rejects receipts after the Git revision, source, lockfile, check
policy, tool/dependency environment, or captured logs change. A commit requires
a fresh run even when it contains previously checked working-tree contents.

After rebuilding changed serialized data, also verify the retained release:

```bash
scripts/check --release teadata/.cache/release.json
scripts/check --release teadata/.cache/release.json --verify-latest
TEADATA_CHECK_PYTHON=3.11 scripts/check
```

Code-only receipts do not certify raw inputs, data quality, or data artifacts.
Release checks verify all selected artifact hashes and bindings and reject a
manifest built from different processing code or lockfile. Use the same Python
selection when verifying an earlier run. Checks make no implicit data refresh.

An optional `scripts/check --install-hook` installs a pre-push code-check guard.
It verifies a current passing receipt or runs checks and then preserves Git LFS
pre-push processing. The guard requires a clean checkout so an uncommitted fix
cannot qualify a different pushed revision. Pushed branches/tags must point to
the checked HEAD; check out other revisions before pushing them. Custom hooks or `core.hooksPath` are left untouched; compose
the check command with those hooks manually. Check subprocesses cannot consume
the Git LFS ref updates from the hook's stdin. Inherited pytest control variables
are removed so shell settings cannot silently reduce the required suite.

Hosted checks remain available through manual GitHub Actions dispatch; pushes,
pull requests, and schedules do not consume hosted minutes automatically. A
passing local receipt is command evidence, not a signed attestation or an
independent review approval. Adversarial review of the final diff remains
required before landing.

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
