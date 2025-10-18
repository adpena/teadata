# TEA Data Engine

Unofficial Python toolkit for Texas public education data — spatially-aware, object-oriented, fast.  
It ships a cache-first “data repo” you can load in seconds, then query with clean, Pythonic primitives (including a fluent `>>` operator).

---

## Contents

- [Quick start](#quick-start)
- [What you get](#what-you-get)
- [Data models](#data-models)
- [Query system (the `>>` operator)](#query-system-the--operator)
- [Spatial tricks (`coords`, `within`, nearest)](#spatial-tricks-coords-within-nearest)
- [Enrichment pipeline](#enrichment-pipeline)
- [Config & data resolution](#config--data-resolution)
- [Performance notes](#performance-notes)
- [Repo layout](#repo-layout)
- [FAQ](#faq)
- [License](#license)

---

## Quick start

### 1) Clone & install (editable)

```bash
git clone https://github.com/adpena/teadata.git
cd teadata
pip install -e .
```

> The editable install makes `import teadata` available and lets you iterate quickly.

### 2) Load the prebuilt snapshot

The engine will automatically discover a `.pkl` snapshot in either:
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

---

## What you get

- **Rich domain objects**: `District` with polygon boundaries; `Campus` with point geometry and canonical IDs.
- **Fluent query language**: chain filters/transforms with `>>` (details below).
- **Spatial acceleration**: nearest-k, containment, and district/campus lookups.
- **Config-driven ingestion**: add datasets by editing YAML/TOML; get schema checks and year-based resolution.
- **Enrichment**: attach new fields (e.g., accountability, finance) as dynamic attributes — and alias into canonical ones.
- **Reproducible caches**: build once into a `.pkl` “repo snapshot”, reload instantly next time.
- **Private school coverage**: ingest Texas private schools when the dataset is present, flagged via `Campus.is_private` for easy filtering.

---

## Data models

### `District`

Key fields & behavior:

- `id: UUID`
- `name: str`
- `district_number: str` — normalized **six-digit**, zero-padded, may be stored with a **leading apostrophe** to match TEA data where needed.
- `rating: Optional[str]` — canonical; can be synced from enrichment via alias
- `boundary: shapely.Polygon | MultiPolygon`
- **Computed / accessors**
  - `campuses: list[Campus]` — all campuses physically inside the boundary (spatial join is precomputed during repo build)
  - `nearest_campuses(...)`
- `charter_campuses`: campuses flagged as charters (excludes private schools)
  - Dynamic attributes from enrichment (e.g., `overall_rating_2025`)

### `Campus`

- `id: UUID`
- `campus_number: str` — normalized **nine-digit** TEA code (zero-padded; apostrophe-safe)
- `name: str`
- `charter_type: str`
- `is_charter: bool`
- `is_private: bool` — True for private schools loaded from the new dataset
- `enrollment: Optional[int]`
- `type: Optional[str]` — e.g. `"OPEN ENROLLMENT CHARTER"`
- `point: shapely.Point`
- `coords: tuple[float, float]` — `(lon, lat)` for convenience
- `district_id: UUID` — back-link to parent
- `district: District` — property resolving the back-link
- Dynamic enrichment fields (e.g., `overall_rating_2025`)

> **Normalization helpers** exist for both district/campus numbers; they’ll accept ints, strings with/without apostrophes, and coerce to canonical text forms.

---

## Query system (the `>>` operator)

The `Query` object wraps lists of `District` or `Campus` and supports fluent chaining via `>>`:

```python
# Start a query from a district TEA number, expand to campuses, then score
district_q = engine >> ("district", "101902")

top5 = (
    district_q
    >> ("campuses_in",)
    >> ("filter", lambda c: (c.enrollment or 0) > 1000)
    >> ("sort", lambda c: c.enrollment or 0, True)  # True -> descending
    >> ("take", 5)
    >> ("map", lambda c: (c.name, c.enrollment))
)

print(top5)  # list[(name, enrollment)]
```

Common operators:

- `>> ("campuses_in",)` — expand the current district query into its campuses
- `>> ("filter", predicate)` or `>> (lambda x: ...)`
- `>> ("sort", key_fn, descending: bool=False)`
- `>> ("take", n)`
- `>> ("nearest_charter", coords, mile_limit, k)` — from repo-level query context
- `first()` — take first element
- Attribute fall-through: `Query.attr` proxies to `first().attr` for convenience.

---

## Spatial tricks (`coords`, `within`, nearest)

### Using `coords`

Every campus exposes `coords` as `(lon, lat)`. This is handy for distance-based queries and pipelines:

```python
campus_q = (engine >> ("district", "101902")) >> ("campuses_in",)
c0 = campus_q.first()
k_nearest = engine.nearest_campuses(*c0.coords, limit=5)  # list[(Campus, miles)]
```

> The engine keeps a fast spatial index (STRtree, Shapely 2.x) for nearest-k and containment probes.

### `within` and `charters_within`

Two common helpers:

```python
# All campuses within a district boundary (inferred from the district query)
inside = ((engine >> ("district", "101902")) >> ("within", None)).to_list()

# Charter-only campuses within the same boundary (excludes private schools)
charters = ((engine >> ("district", "101902")) >> ("within", None, True)).to_list()

# Equivalent imperative helpers
inside_alt = engine.within(aldine, items="campuses")
charters_alt = engine.charter_campuses_within(aldine)
```

These use polygon containment and are validated by a built-in slow-path check to avoid false negatives.

### Nearest

```python
# Five nearest charters within 10 linear miles of a target point (private schools are automatically excluded)
pt = (-95.36, 29.83)  # (lon, lat)
nearest_charters = engine.nearest_campuses(
    pt[0],
    pt[1],
    limit=5,
    max_miles=10,
    charter_only=True,
)
```

---

## Enrichment pipeline

Bring in external datasets (accountability, finance, etc.) and attach fields directly to objects.

### Districts

```python
from teadata.enrichment.districts import enrich_districts_from_config
from teadata.teadata_config import load_config

CFG = load_config("teadata_sources.yaml")
year, updated = enrich_districts_from_config(
    engine, CFG, dataset="accountability", year=2025,
    select=["2025 Overall Rating"],
    rename={"2025 Overall Rating": "overall_rating_2025"},
    aliases={"overall_rating_2025": "rating"},  # also write canonical slot
)
print(f"Enriched {updated} districts from {year}")
```

### Campuses

```python
from teadata.enrichment.campuses import enrich_campuses_from_config

year, updated = enrich_campuses_from_config(
    engine, CFG, dataset="accountability", year=2025,
    select=["2025 Overall Rating"],
    rename={"2025 Overall Rating": "overall_rating_2025"},
    aliases={"overall_rating_2025": "rating"},
)
print(f"Enriched {updated} campuses from {year}")
```

### Snapshots

After enrichment, persist a reproducible repo:

```python
engine.save_snapshot(".cache/repo_<tag>.pkl")
# Later:
engine2 = DataEngine.from_snapshot(".cache/repo_<tag>.pkl")
```

Snapshots are versioned with a content signature. The loader can discover and pick the latest automatically with `search=True`.

### Relational persistence (PostgreSQL via SQLAlchemy)

Prefer a shared database over large pickle blobs?  Install the optional extra and
use the new SQLAlchemy bridge:

```bash
pip install "teadata[database]"
```

```python
from teadata import DataEngine
from teadata.persistence import (
    available_meta_keys,
    create_engine,
    create_sessionmaker,
    ensure_schema,
    export_dataengine,
    fetch_meta_values,
    import_dataengine,
)

# 1. Build or load an in-memory repo as usual
repo = DataEngine.from_snapshot(search=True)

# 2. Connect to PostgreSQL (any SQLAlchemy URL works)
engine = create_engine("postgresql+psycopg://user:pass@host:5432/teadata")
ensure_schema(engine)  # creates tables on first run
Session = create_sessionmaker(engine)

# 3. Persist everything: districts, campuses, enrichment meta, transfers, geometry
with Session() as session:
    export_dataengine(repo, session)
    session.commit()

# 4. Hydrate a fresh DataEngine straight from SQL when serving web/API requests
with Session() as session:
    repo_from_db = import_dataengine(
        session,
        lazy_meta=True,  # defer meta JSON until accessed (saves RAM per request)
        prefetch_campus_meta_keys=["accountability.rating"],  # optional targeted warm-up
    )

# Lazy meta hydration will automatically spin up lightweight ad-hoc sessions
# using the same engine when the originating Session is closed.  If you prefer
# to manage that lifecycle yourself, pass a callable that yields a new Session:
# repo_from_db = import_dataengine(session, meta_session_factory=Session)

# Inspect which enrichment columns are available before fetching values
with Session() as session:
    top_keys = available_meta_keys(session, entity="campus")[:5]
    print("First few enrichment keys:", top_keys)

    # Pull just a handful of enrichment columns without materializing the full JSON
    preview = fetch_meta_values(
        session,
        entity="campus",
        keys=["accountability.rating", "finance.per_pupil_spend"],
    )
    if preview:
        print(next(iter(preview.values())))
```

The ORM models keep canonical UUIDs and TEA numbers indexed for quick lookup,
serialize `meta` dictionaries as JSONB (or JSON on non-Postgres backends),
explode scalar metadata into `district_meta` / `campus_meta` tables with typed
columns for sorting/filtering, and store campus-to-campus transfer edges in a
dedicated join table so existing graph helpers keep working.

`import_dataengine` now keeps enrichment payloads lazy by default: districts and
campuses hydrate with lightweight `meta` dictionaries that fetch their JSON only
when a request accesses an enrichment key.  This keeps per-request memory close
to the bare domain objects instead of eagerly allocating tens of megabytes of
metadata.  When you do need specific enrichment columns for a bulk analysis,
use `available_meta_keys` to discover what exists and `fetch_meta_values` or the
`prefetch_*` arguments to warm just those keys.

Want to query enrichment fields directly?  Example Postgres snippets:

```sql
-- Count campuses by an enrichment key promoted out of campus.meta
SELECT value_text AS accountability_rating, COUNT(*)
FROM campus_meta
WHERE key = 'accountability.rating'
GROUP BY value_text
ORDER BY value_text;

-- Numeric sort / filter on scalar meta
SELECT campus_id, value_numeric
FROM campus_meta
WHERE key = 'finance.per_pupil_spend'
ORDER BY value_numeric DESC
LIMIT 20;
```

---

## Config & data resolution

All external data locations and schema expectations are declared in a single YAML (or TOML) config.  
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

**Programmatic access**

```python
from teadata.teadata_config import load_config
cfg = load_config("teadata_sources.yaml")

# Resolve “best” file for a year (exact match or nearest prior)
resolved_year, path = cfg["data_sources", "accountability", 2025]
print(resolved_year, path)

# Quick availability report
print(cfg.availability_report())
```

**Cross-dataset joins**

Use the provided helpers to normalize `district_number` and join TAPR/PEIMS/etc. without fighting column name variations.

---

## Performance notes

- **Shapely 2.x + STRtree** powers nearest and containment queries at scale.
- **One-time precomputation** during repo build: district ➜ campuses index, ID maps, clean normalizations.
- **Cache everything**: Snapshots persist the derived indices; reloads are **milliseconds**, not minutes.
- **Slow-path sanity check** on spatial containment can be toggled, and only runs when the fast path returns 0 where >0 is expected.

---

## Repo layout

- `teadata/classes.py` — models (`District`, `Campus`, `DataEngine`), query object, spatial index.
- `teadata/load_data.py` / `teadata/load_data2.py` — examples/scripts to build and snapshot repos from raw sources.
- `teadata/teadata_config.py` — config reader/validator (YAML/TOML), year-resolution, schema validation.
- `teadata/enrichment/` — modular enrichment functions (districts, campuses, charter networks, etc.).
- `.cache/` — binary snapshots (`repo_*.pkl`) for instant loads.

---

## FAQ

**Q: Where do snapshots live?**  
A: Prefer the package-internal `.cache/` for the “best known” snapshot. The loader also searches your repo root `.cache/`.

**Q: Does `district_number` need an apostrophe?**  
A: Normalizers accept either digits-only or apostrophe-prefixed strings and standardize internally.

**Q: Can I chain spatial + attribute filters?**  
A: Yes — use `>>` to compose: `engine >> ("within", d, "campuses") >> (lambda c: c.rating in {"A","B"})`.

---

## License

MIT. See `LICENSE`.

---

If you build cool analyses or add new enrichments (finance, demographics, staffing), PRs welcome. This toolkit is meant to cut hours of wrangling to minutes and make room for the hard, interesting questions.
