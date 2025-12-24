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
pip install "teadata[database]"
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

# 2. Connect to PostgreSQL (any SQLAlchemy URL works)
engine = create_engine("postgresql+psycopg://user:pass@host:5432/teadata")
ensure_schema(engine)  # creates tables on first run
Session = create_sessionmaker(engine)

# 3. Persist everything: districts, campuses, enrichment meta, transfers, geometry
with Session() as session:
    export_dataengine(repo, session)
    session.commit()
```

### Import from SQL

```python
from teadata.persistence import import_dataengine

# Hydrate a fresh DataEngine straight from SQL
with Session() as session:
    repo_from_db = import_dataengine(
        session,
        lazy_meta=True,  # defer meta JSON until accessed (saves RAM per request)
        prefetch_campus_meta_keys=["accountability.rating"],  # optional targeted warm-up
    )
```

`import_dataengine` now keeps enrichment payloads lazy by default: districts and campuses hydrate with lightweight `meta` dictionaries that fetch their JSON only when a request accesses an enrichment key.

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
