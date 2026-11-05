# TEA Data Engine

**Unofficial Python toolkit for Texas public education data — spatially-aware, object-oriented, fast.**

Load a catalog-bound release, then query its district and campus objects with a fluent `>>` operator.

---

## Quick start

### 1) Clone & install (editable)

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/adpena/teadata.git
cd teadata
# Install with uv (recommended)
uv sync
```

### 2) Install an application dependency

Select a published version and retain your application's lockfile:

```bash
uv add "teadata==0.0.130"
```

### 3) Load the configured release

Set `TEADATA_RELEASE` to a local release manifest or `TEADATA_RELEASE_URL` to an
immutable remote release manifest. Python packages contain code and metadata,
with data bundles distributed separately. For example:

```bash
export TEADATA_RELEASE_URL="https://github.com/adpena/teadata/releases/download/v0.0.128/release.json"
export TEADATA_RELEASE_SHA256="9dd062128273cbb90f3b9c952b072d74a629de6ecbcff8065c4295b511eaa9aa"
```

A source checkout may instead have a local `teadata/.cache/release.json`;
ordinary wheel installs do not include that bundle. A release pinned by `using_release()` takes
precedence. The manifest selects the snapshot and sidecars; selected artifact
hashes are checked when they are loaded.

```python
from teadata import DataEngine

engine = DataEngine.load_default()

print(len(engine.districts), len(engine.campuses))
# Counts depend on the selected release.
```

`DataEngine()` creates an empty engine by default (`autoload=False`).
`DataEngine.from_snapshot()` without a path uses the same configured release.
An explicit `from_snapshot(path)` is for a trusted local catalog-bound snapshot.

### 4) Look up a district

```python
# 101902 is a six-digit TEA district number.
aldine = (engine >> ("district", "101902")).first()
if aldine is not None:
    print(aldine.name, aldine.rating)
    for campus in aldine.campuses:
        print(campus.name, campus.rating)

# Name matching is case-insensitive and can return multiple districts.
for district in (engine >> ("district", "Northside ISD")):
    print(district.district_number, district.name)
```

The district operation returns a `Query`. `.first()` selects the first match or
returns `None`; iterate the query to inspect all matches for an ambiguous name
or wildcard. `district.campuses` lists campuses assigned to that district.

## What you get

- **Rich domain objects**: `District` with polygon boundaries; `Campus` with point geometry and canonical IDs.
- **Fluent query language**: chain filters/transforms with `>>`.
- **Spatial acceleration**: nearest-k, containment, and district/campus lookups.
- **Config-driven ingestion**: add datasets by editing YAML/TOML; get schema checks and year-based resolution.
- **Enrichment**: attach new fields (e.g., accountability, finance) as dynamic attributes — and alias into canonical ones.
- **Reproducible caches**: build once into a `.pkl` (and `.pkl.gz`) “repo snapshot”, reload instantly next time.
- **Private school coverage**: ingest Texas private schools when the dataset is present, flagged via `Campus.is_private` for easy filtering.
