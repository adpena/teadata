# Enrichment Pipeline

Bring in external datasets (accountability, finance, etc.) and attach fields directly to objects.

## Districts

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

## Campuses

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

## Snapshots

After enrichment, persist a reproducible repo:

```python
engine.save_snapshot(".cache/repo_<tag>.pkl")
# Later:
engine2 = DataEngine.from_snapshot(".cache/repo_<tag>.pkl")
```

Snapshots are versioned with a content signature. The loader can discover and pick the latest automatically with `search=True`.

## Relational Persistence

Prefer a shared database over large pickle blobs? Install the optional extra and use the SQLAlchemy bridge:

```bash
uv add "teadata[database]"
```

### Export to SQL

```python
from teadata import DataEngine
from teadata.persistence import (
    create_engine,
    create_sessionmaker,
    ensure_schema,
    export_dataengine,
)

# 1. Build or load an in-memory repo as usual
repo = DataEngine.from_snapshot(search=True)

# 2. Connect to PostgreSQL or SQLite using your configured connection URL.
import os
engine = create_engine(os.environ["TEADATA_DATABASE_URL"])
ensure_schema(engine)  # creates tables on first run
Session = create_sessionmaker(engine)

# 3. Persist everything: districts, campuses, enrichment meta, transfers, geometry
with Session.begin() as session:
    export_dataengine(repo, session, batch_size=256)
```

The default upserts the supplied entities and transfer edges and preserves
unrelated rows. Use `replace=True` explicitly to replace the entire destination,
including its metadata index rows. A destination is bound to one validated
dataset catalog. Partial exports from another release are refused; the catalog,
entity rows, metadata, and transfers commit or roll back together. Exports require
a catalog-bound `DataEngine`; loading a retained snapshot provides one.

Writes use bounded SQLAlchemy Core batches and native SQLite/PostgreSQL upserts.
They do not hydrate ORM records, query every entity, or read the destination's
entire transfer history. `batch_size` controls additional row buffers; the input
`DataEngine` itself still contains its object graph.

### Import from SQL

```python
from teadata.persistence import import_dataengine

# Hydrate a fresh DataEngine straight from SQL
with Session() as session:
    repo_from_db = import_dataengine(
        session,
        lazy_meta=True,  # defer meta JSON until accessed (saves RAM per request)
        prefetch_campus_meta_keys=["source_field_name"],  # optional exact raw key
        batch_size=256,
    )
```

`import_dataengine` streams database mappings directly into the returned graph
without building a second full ORM graph. Eager metadata is read with each entity;
lazy metadata loads when needed and refuses a changed source release or deleted
entity. Prefetch preserves complete nested raw values; flattened metadata index
keys are a separate query API. The caller controls transaction isolation for a
consistent import while other clients write. Lazy reads observe the current rows
of the bound release and do not retain a database snapshot indefinitely.
Lazy metadata is a standard mutable mapping. Use `dict(entity.meta)` or
`entity.to_dict()` at a JSON boundary; a custom JSON encoder is unnecessary.
An active caller Session is reused for lazy reads, including uncommitted writes.
After that transaction ends, each deferred read owns and closes its new Session.
The catalog document remains authoritative; a validated 64-byte build ID column
fences lazy reads without retransmitting the full provenance catalog each time.
Older two-column database catalog layouts must be recreated and refilled. Reading
historical immutable release artifacts remains supported.

Raw metadata uses JSON rather than PostgreSQL JSONB so signed zero and integer
values survive round trips. Approximate numeric lookup columns are populated only
when a value can be represented exactly; raw JSON remains authoritative. Strings
containing NUL retain their exact JSON value and have no PostgreSQL TEXT lookup
projection. Optional flattened metadata index keys must be valid UTF-8, have at
most 255 characters, and contain no NUL. An unsupported key raises an actionable
error; roll back the export and use `include_meta_entries=False` to preserve
arbitrary raw JSON without that optional index. Keys are never silently dropped
or encoded into a different name. SQLite UUID columns use SQLAlchemy's
portable `Uuid` type with text affinity. Older databases with NUMERIC-affinity
UUIDs or JSONB columns must be recreated and refilled from the catalog-bound
source: those representations may already have lost information. `ensure_schema`
refuses those layouts and does not silently alter or repair existing databases.

### Querying Metadata in SQL

Want to query enrichment fields directly? Example Postgres snippets:

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
