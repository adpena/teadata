# TEA Data Engine

**Unofficial Python toolkit for Texas public education data — spatially-aware, object-oriented, fast.**

It ships a cache-first “data repo” you can load in seconds, then query with clean, Pythonic primitives (including a fluent `>>` operator).

---

## Quick start

### 1) Clone & install (editable)

```bash
git clone https://github.com/adpena/teadata.git
cd teadata
# Install with uv (recommended)
uv sync
# OR with pip
pip install -e .
```

### 2) Install from GitHub (for Render/CI)

If you are using this in a Django application deployed on **Render**, add it to your `requirements.txt`:

```text
teadata @ git+https://github.com/adpena/teadata.git
```

### 3) Load the prebuilt snapshot

The engine will automatically discover a `.pkl` or `.pkl.gz` snapshot in either:

- `./.cache/` under your repository root, or
- `<site-packages>/teadata/.cache/` shipping with the library.

```python
from teadata import DataEngine

# Fast-path: load the latest discovered snapshot
engine = DataEngine.from_snapshot(search=True)

print(len(engine.districts), len(engine.campuses))
# -> e.g. 1207 9739
```

> If you prefer the implicit path: `engine = DataEngine()` also tries to auto-load a snapshot.

### 3) First query in 10 seconds

```python
# Retrieve district by TEA campus number (integer, digits only, or 6-digit left-zero padded string with leading apostrophe all work)
aldine = engine.get_district("101902")

# Retrieve district by district name (case insensitive - returns multiple results if there are multiple districts with the same name, such as "Northside ISD")
aldine = engine.get_district("Aldine ISD")

print(aldine)                 # District(...)
print(aldine.rating)          # canonical (may be enriched via alias)
print(aldine.overall_rating_2025)  # enriched attribute, if present

# Iterate campuses physically inside the district
for c in aldine.campuses:
    print(c.name, c.rating)
```

## What you get

- **Rich domain objects**: `District` with polygon boundaries; `Campus` with point geometry and canonical IDs.
- **Fluent query language**: chain filters/transforms with `>>`.
- **Spatial acceleration**: nearest-k, containment, and district/campus lookups.
- **Config-driven ingestion**: add datasets by editing YAML/TOML; get schema checks and year-based resolution.
- **Enrichment**: attach new fields (e.g., accountability, finance) as dynamic attributes — and alias into canonical ones.
- **Reproducible caches**: build once into a `.pkl` (and `.pkl.gz`) “repo snapshot”, reload instantly next time.
- **Private school coverage**: ingest Texas private schools when the dataset is present, flagged via `Campus.is_private` for easy filtering.
