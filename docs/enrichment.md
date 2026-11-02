# Enrichment Pipeline

Bring in external datasets (accountability, finance, etc.) and attach fields directly to objects.

## Districts

```python
from teadata.enrichment.districts import enrich_districts_from_config
CFG = "teadata/teadata_sources.yaml"
year, updated = enrich_districts_from_config(
    engine, CFG, dataset="accountability", year=2025,
    select=["overall_rating"],
    aliases={"overall_rating": "rating"},  # also write canonical slot
    reader_kwargs={"reader_profile": "ratings", "exact": True},
)
print(f"Enriched {updated} districts from {year}")
```

## Campuses

```python
from teadata.enrichment.campuses import enrich_campuses_from_config

year, updated = enrich_campuses_from_config(
    engine, CFG, dataset="accountability", year=2025,
    select=["overall_rating"],
    aliases={"overall_rating": "rating"},
    reader_kwargs={"reader_profile": "ratings", "exact": True},
)
print(f"Enriched {updated} campuses from {year}")
```

## Outcomes and Coverage

`run_enrichment()` runs optional registered enrichers. Its returned mapping keeps
each successful enricher's existing result keys (such as `year` and `updated`)
and adds `outcome`. Failed and unregistered selections also receive entries:

```python
from teadata.enrichment import run_enrichment

stats = run_enrichment(engine, CFG, 2025, datasets=["campus_accountability"])
outcome = stats["campus_accountability"]["outcome"]
print(outcome["status"], outcome["matched"], outcome["unmatched"])
```

The campus/district table joins report `expected` repository objects visited,
`matched` objects receiving a source row, `unmatched` objects with a usable
identifier but no matching row, and `missing_identifier` objects that cannot be
joined. They also retain the requested/resolved catalog years and observed
`source_rows`. These are object-join counts, not an expected statewide reporting
population, proof that every source row was used, or validation of every field.
An unmatched object does not imply a zero, inapplicability, or a publisher error.
Numeric zero and negative source tokens retain their original values and policies.

Statuses distinguish `succeeded`, `partial`, `unmatched`, `unsupported`, and
`failed`. Missing source files are labelled `source_status="missing"`; a loaded
zero-row table is `empty`. A registered legacy enricher that returns only an
update count has unknown (`None`) coverage counts, even when it succeeds.
Unknown source availability remains `unknown`. Error text is limited to 4,096
characters, with `error_truncated=True` when shortened. Interrupts still propagate.

Outcomes remain on the engine and are copied into the existing release processing
record's `validation.enrichment_outcomes`. Required enrichments still raise on
failure or zero matches, and failed required outcomes block release validation.
During an active build with successfully verified input receipts, a required
failure archives its actual outcome JSON (maximum 2 MiB) as a diagnostic output
of a failed processing record. That record does not publish a release and does
not become a source acquisition. If diagnostic retention fails, the original
exception is preserved with a note. Outside an active build, the caller receives
the exception and engine outcomes; no processing receipt is claimed. A stopped
pipeline reports attempted stages only and does not imply later stages ran.

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
