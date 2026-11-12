# Enrichment Pipeline

Bring in external datasets (accountability, finance, etc.) and attach fields directly to objects.

The [source coverage table](source-coverage.md) lists the entity build inputs,
their loaded periods, and their configured editions. The builder records its
exact sources. Separate native source tables are available through `DataRelease`.

## Observations across school years

An entity identifies a district or campus. An observation also needs its
measurement period, source revision, and the publisher's row keys. For example,
annual dropout has separate rows for grade span and reporting calculation. A
single value in an entity's metadata cannot represent those distinctions.

An entity build can attach native releases through `options.native_sources` in
its build configuration. A pinned selection needs only its manifest location
and SHA-256:

```yaml
options:
  native_sources:
    - location: staff-native.json
      manifest_sha256: <exact SHA-256 of the retained manifest bytes>
```

The builder derives the build ID and complete edition keys from that exact
manifest. Existing four-field entries (`location`, `manifest_sha256`, `build_id`,
`editions`) remain supported and must match the manifest. Use an immutable
HTTP(S) URL or a distinct JSON filename adjacent to the entity manifest. Stage
the complete native bundle before building; HTTP inputs must already be cached.
The builder verifies it offline and includes the expanded pins and edition keys
in the entity build ID and processing receipt. Selection does not download data
or confer semantic qualification.

The final release gate selects the same reviewed qualification plan used for a
standalone native release. Keep the retained `provenance` store next to each
attached manifest, including its independent publisher witnesses. The builder's
structural checks do not replace that semantic replay. See the
[reviewed source families and evidence requirements](data-releases.md#qualifying-reviewed-native-bundles).
Other attachments remain on hold until they have a reviewed qualification path.

The observation planner resolves selected fields by measurement period before
opening a data file. Execution retains the native row keys, raw values and
suppression status. An entity query filters those rows by selected TEA numbers.
The historical population and geography still require their own dated sources.
See [school-year selection](querying.md#school-years-and-source-editions).

`series.to_csv(path)` and `series.to_excel(path)` stream one selected year at a
time through the same writers as source tables. They retain the measurement-year
column, publisher raw tokens, normalized values and statuses. Excel includes the
complete series selection and each year's citations and source pins in its
Provenance worksheet. CSV needs the existing JSON sidecar:

```python
import json
from pathlib import Path

series.to_csv("observations.csv")
Path("observations.provenance.json").write_text(
    json.dumps(series.provenance(), indent=2), encoding="utf-8"
)
series.to_excel("observations.xlsx")
```

Exports do not establish cross-year comparability. Series use a common schema,
including null context columns absent from individual editions. For a publisher's
original header layout, select one exact year, for example
`series.for_year(25).to_csv("publisher.csv", restore_original_headers=True)`.
The same option applies to that source table's DataFrame and Excel exports.
Writers preserve decimal precision (as text in Excel when necessary), reject
existing destination paths and publish files atomically. Caller-owned binary
streams remain open; a stream can contain partial output if writing fails.

### Lazy dataset access on entities

An entity in a release-bound engine discovers attached, qualified dataset names
without opening data leaves. Registration alone does not add a dataset here.
Names are case insensitive; unknown names give suggestions, never fuzzy selection.

```python
district.available_datasets()
staff = district.dataset("peims_staff_fte_and_salary")
# The same view, when no metadata key conflicts:
staff = district.peims_staff_fte_and_salary
staff.variables(r"^staff_")
staff.years("staff_fte")
selected = staff.at(26).select("staff_fte", "staff_average_base_pay").query(
    "staff_category == 'TEACHING STAFF' and staff_role == 'TOTAL TEACHING STAFF'",
    record=True,
)
selected.to_df()  # Raw tokens, value statuses and provenance are retained.
selected.provenance()
```

These are immutable views over the same observation planner and artifacts used
by `Query.observations`, not copies in `meta`. `.tprs` remains a compatible facade
over its reviewed Student Information and class-size families. Single-row fields
support `entity.<dataset>.<field>` and `view[field]`; multiple rows require explicit
selection and filtering. They are never silently summed or reduced to one row.
An absent entity row is an error rather than an older observation. Private school
entities cannot acquire public campus observations through this interface.

The default resolves the latest eligible measurement school year for all selected
fields together; `.at(25)` requests 2024–25 exactly. An unavailable year lists the
available years. Fiscal and calendar periods currently require the existing exact
`release.for_entity(entity, edition=publisher_edition).dataset(edition_key)` path;
the convenience facade reports that limitation before opening data leaves.
Publisher edition must never stand in for measurement year.

`entity.available_datasets()` reads parent descriptors only. `view.variables()`
reads verified metadata for that dataset, with bounded runtime caches excluded
from snapshots. Scalar access reads the selected source artifact; this is not a
promise of remote row-level transfers or an indexed point lookup. Regular
`dir(entity)` and whole-mapping operations retain their existing mapping behavior.
`describe_operators()["entity_access"]` documents this interface without opening
a release; available dataset names and fields come from the selected release.

Existing direct metadata attributes and local edits remain supported. Native
fields do not automatically become unqualified scalar aliases. A metadata key
that conflicts with a dataset namespace remains accessible through
`entity.meta[key]`; explicit `entity.dataset(name)` bypasses the collision. A cold
`LazyMetaDict` without a safe keys-only reader also requires that explicit form,
including instead of implicit `.tprs` access. SQL keys-only discovery refuses
unsupported encodings or projections rather than hydrating the full payload.
Arbitrary third-party mappings retain their own key-iteration behavior.

DataFrame attributes and the Excel provenance worksheet carry export lineage.
CSV carries the selected values; retain `selected.provenance().to_dict()` beside
it to preserve the same source and query identity.

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

The campus/district table joins report `expected` eligible repository objects,
`matched` objects receiving a source row, `unmatched` objects with a usable
identifier but no matching row, and `missing_identifier` objects that cannot be
joined. They also retain the requested/resolved catalog years and observed
`source_rows`. These are object-join counts, not an expected statewide reporting
population, proof that every source row was used, or validation of every field.
An unmatched object does not imply a zero, inapplicability, or a publisher error.
Numeric zero and negative source tokens retain their original values and policies.

Campus-number joins exclude explicitly private campuses from the eligible
population and count them separately as `not_applicable`, not as missing public
identifiers. Public objects without a usable identifier remain
`missing_identifier`; no identifier or population is inferred from an opaque
source ID. A completed campus join reports `not_applicable=0` when none were
excluded; enrichers that do not observe this dimension retain `None`.

Statuses distinguish `succeeded`, `partial`, `unmatched`, `not_applicable`, `unsupported`, and
`failed`. Missing source files are labelled `source_status="missing"`; a loaded
zero-row table is `empty`. A registered legacy enricher that returns only an
update count has unknown (`None`) coverage counts, even when it succeeds.
Unknown source availability remains `unknown`. Error text is limited to 4,096
characters, with `error_truncated=True` when shortened. Interrupts still propagate.
An optional all-private campus join reports `not_applicable` with zero eligible
objects. Required enrichments still reject zero matches, including this case.

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

### Materialize selected observations into your own tables

Use `DataRelease.series()` to select source fields, measurement school years
and, optionally, campus or district IDs. Inspect `series.schema` and
`series.provenance()` before opening the observations. Use
`series.to_sql(connection, observations, receipts=receipts)` with your
SQLAlchemy Core tables and an active transaction. The same contract appears
in `describe_operators()["release_methods"]`.

Run the [SQLite example](../examples/materialize_research_db.py) from this
checkout. It uses public release assets and a database path you choose.

```bash
uv run --extra database python examples/materialize_research_db.py research.sqlite
sqlite3 research.sqlite 'SELECT measurement_school_year, campus_number, COUNT(*) FROM campus_enrollment GROUP BY 1, 2 ORDER BY 1, 2;'
```

The observation table uses each projected Arrow field as a SQLAlchemy column
key, plus an `import_id` referencing the receipt's primary key. Use text
columns for identifiers and original publisher tokens. A SQLAlchemy
`Column(key=...)` can map a source key to a different SQL column name. The
example's fact key combines the import, measurement year, campus and
ethnicity. Adapt keys and types to the publisher's grain. TEA Data inserts
one native batch at a time using bound parameters. It preserves support
columns, including raw tokens and status, and leaves DDL, indexes, commit
and rollback to the caller. Let errors propagate out of
`with engine.begin() as connection` to roll back an incomplete import.
With SQLite's legacy transaction mode, use a top-level transaction.

The JSON receipt stores the exact root build and manifest checksum, edition
and artifact identities, field definitions, publisher links, acquisition
dates where known, selector, citations and observed row counts by year.
The returned `SeriesSqlExport.to_dict()` is JSON-ready. An absent
requested entity-year raises `SeriesCoverageError` by default. Its
`result.missing_entity_years` identifies the gaps. For a deliberately sparse
panel, pass `require_complete=False` to store only published rows and include
the missing pairs in the receipt. Suppressed rows remain present with a SQL
`NULL` numeric value and their original token. Duplicate imports into the
same table raise on the receipt's primary key.

Read the same import back as typed Arrow batches using the original series
and the import ID returned by `to_sql`:

```python
with engine.connect() as connection:
    for batch in series.iter_sql_batches(connection, observations, receipts=receipts,
                                         import_id=result.import_id, years=[2022]):
        print(batch.num_rows)
```

The year filter runs in SQL. The reader checks the release, selection and
physical destination against the stored plan, then decodes the stored columns
using the series schema. All raw and status columns stay available. Use a
trusted database because the plan digest covers provenance and table mapping,
while the stored rows remain under database control. Row order is unspecified.
Unqualified tables resolve to `main` in SQLite. In PostgreSQL, the reader and
writer resolve persistent tables on the connection's search path and use
their physical schema names. Supply explicit names to bypass name lookup.
Earlier receipts that recorded no physical schema require a new export before
using this reader. Leave PostgreSQL autocommit disabled for cursor streaming.

The example's 2021–22 observations for A+ Academy Elementary (`057829001`)
were checked against [TEA's campus report](https://rptsvr1.tea.texas.gov/cgi/sas/broker?_service=marykay&_program=adhoc.addispatch.sas&endyear=22&major=st&minor=e&format=w&selsumm=nc&linespg=60&charsln=120&grouping=e&loop=2&key=057829001&_debug=0),
retrieved November 9, 2026, under acquisition receipt
`0183affaf59a4bed506785d6febf31e3095cea562ddbefd4dfe24866dd15755e`.
The five checked cells include Asian enrollment suppressed as `<10` and
White enrollment observed as `17`. To run that golden against the original
published native artifact or a verified local copy, set
`TEADATA_SQL_GOLDEN_MANIFEST` to the manifest and run
`uv run --extra database pytest tests/test_series_sql.py`.

#### Query from Django, Flask, or SQL

Build the database outside web requests. For Django, use a dedicated database
alias and reviewed unmanaged models for separately built tables. Django's
`inspectdb --database <alias>` can scaffold model fields. Review the primary
key and use read-only database credentials or SQLite `mode=ro` to enforce
read-only access. Django's `managed = False` controls migrations. Use a
management command only if the app needs an app-specific rebuild workflow.
See the [TEA Data app's production example](https://github.com/adpena/teadata-app/blob/main/docs/data-release-migration.md#read-only-profile-snapshot)
and [Django's database integration guide](https://docs.djangoproject.com/en/6.1/howto/legacy-databases/).

In Flask, use a [Flask-SQLAlchemy bind](https://flask-sqlalchemy.palletsprojects.com/en/stable/binds/)
for the same tables or open them with SQLAlchemy Core. Raw SQL clients can
query the fact table and join its `import_id` to the receipt. PostgreSQL
deployments can supply their own transactional SQLAlchemy connection, driver
and schema. The same example rows and rollback behavior were checked in
SQLite and PostgreSQL.

For DuckDB, [query the Arrow stream directly](https://duckdb.org/docs/stable/guides/python/sql_on_arrow).
Register a new reader for each query that consumes it. Keep
`series.provenance()` with any result you save.

```python
import duckdb
import pyarrow as pa

with duckdb.connect() as connection:
    reader = pa.RecordBatchReader.from_batches(series.schema, series.iter_batches())
    connection.register("observations", reader)
    counts = connection.sql(
        "SELECT measurement_school_year, COUNT(*) AS observations "
        "FROM observations GROUP BY 1 ORDER BY 1"
    ).fetchall()
```

### Round-trip the canonical entity graph

#### Export to SQL

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

#### Import from SQL

```python
from teadata.persistence import import_dataengine

# Hydrate a fresh DataEngine straight from SQL
with Session() as session:
    repo_from_db = import_dataengine(
        session,
        lazy_meta=True,  # defer metadata and project individual keys where supported
        prefetch_campus_meta_keys=["source_field_name"],  # optional exact raw key
        batch_size=256,
    )
```

`import_dataengine` streams database mappings directly into the returned graph
without building a second full ORM graph. Eager metadata is read with each entity;
lazy metadata refuses a changed source release or deleted entity on deferred reads.
Dot access and bracket access both use the same mapping. For ASCII identifier
keys, SQLite 3.38 or later with JSON support and PostgreSQL with UTF-8 server
and client encodings and the
default psycopg 3 cursors and JSON loader can return just the selected JSON
value. Prefetch projects up to 16 such keys in one bounded query per entity
type. Both paths preserve JSON null, large integers and nested
values. Compatibility paths read the original document for unusual keys,
unsupported engines or encodings, custom JSON decoders, or rows where native
extraction cannot preserve the original value. Deferred sessions from a
different bind also use the original reader. The flattened index remains a
separate query API. The caller controls transaction isolation for a consistent
import while other clients write. Lazy reads observe the bound release's rows
at query time. Previously read values can already be in memory.
Lazy metadata is a standard mutable mapping. Use `dict(entity.meta)` or
`entity.to_dict()` at a JSON boundary. Whole-mapping operations still materialize
the document. For names alone, `entity.meta.iter_keys()` on a lazy mapping lists
exact top-level names in lexical order. A supported SQL reader returns just
the names. It uses the original reader for documents or decoders that need it.
Use `repo_from_db.catalog.measurement(entity, metric)` for an interpreted value
and its catalog provenance by metric name, or
`repo_from_db.catalog.field_measurement(entity, field)` by exact field name.
The field form raises `teadata.manifest.FieldMeasurementError` when the catalog
cannot interpret the key. The exception gives a reason and available alternatives,
including spelling suggestions for unknown fields. A bound field with a missing
observation instead returns a `Measurement` with status `missing`. The exact
field method reads the requested field alone. A named metric may have a
catalog-declared fallback to an entity attribute.
Within one dataset, an explicit metric binding takes precedence over
a family wildcard. Use the metric form to choose between conflicting
interpretations at the same precedence.
The metric lookup stays sparse. The catalog may also check source, scope, and
population attributes before interpreting it.

On a lazy SQL import, mapping writes and deletions are local edits. Measurements
of edited fields carry `input_binding="caller_supplied"` while keeping the
catalog's interpretation rules. A deleted field stays caller-supplied when a
named metric uses its fallback. Unchanged fields retain their release binding.
With a supported SQL point reader, the first edit to an ASCII identifier
checks the source row and build. Assignment uses a scalar query. Deleting a
source key checks its current presence instead of using a prefetched value.
Later assignments to that mapping need no database read. A failed delete leaves a stale
prefetch alone until a full read. The active reader can fall back to raw JSON
when it cannot project that key exactly. Full conversion uses the source row
visible at that read, keeps its surviving key order, and appends local keys
that are new or were deleted and reinserted. Use a consistent transaction for
a fixed source view across reads. Public copies and operations such as
`popitem()` still materialize the mapping.
Copies and pickle round trips keep known edits. Changes made through the
backing dict or a nested value are detected by comparing their current content
with the value before it was shared. Eager SQL imports retain their ordinary
dict interface and do not track caller edits.

The catalog-bound SQL writer rejects edited lazy imports before touching the
destination. Reload the original release for a release export. Use
`entity.to_dict()` to export your working data separately. For same-database
replacement, read the source eagerly before beginning the replacement.

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

#### Querying Metadata in SQL

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
