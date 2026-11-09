# Query System

The `Query` object wraps lists of `District` or `Campus` and supports fluent chaining via the `>>` operator.

## Basics

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

Each shift returns a result from the original selection. Assign the returned
query to keep a filtered or sorted branch. Branches share the same entity
objects, so editing an entity is visible through any branch that contains it.
Transfer predicates propagate exceptions. A predicate can explicitly accept
or reject a missing destination.

## Common Operators

- `districts >> ("campuses_in",)` expands districts to their non-private member campuses.
- `engine >> ("within", boundary[, charter_only[, covers]])` finds campuses geographically inside a district boundary or a WGS84 Polygon or MultiPolygon. The chained form is `districts >> ("within", None, None)`, which uses the first input district. General containment excludes boundary points unless `covers=True`.
- `districts >> ("privates_within",)` selects private campuses inside or on reported district boundaries. The engine entry form is `engine >> ("privates_within", district)`. Geographic containment is separate from institutional governance, and border points can match multiple districts.
- `>> ("filter", predicate)` or `>> (lambda x: ...)`
- `>> ("sort", key_fn, descending: bool=False)`
- `>> ("take", n)` takes an integer and uses Python slice semantics, including zero and negative counts.
- `>> ("radius", (longitude, latitude), miles)` — campuses within an inclusive great-circle distance. Radius and KNN use valid longitude and latitude coordinates.
- `>> ("nearest_charter", coords, limit, max_miles)` selects nearby charter campuses. `limit` defaults to one.
- `first()` — take first element
- Attribute fall-through: `Query.attr` proxies to `first().attr` for convenience.

!!! tip "Chaining"
    You can mix spatial and attribute filters easily:
    
    ```python
    engine >> ("charters_within", district) >> (lambda c: c.rating in {"A", "B"})
    ```

## Discover Query operators

`describe_operators()` returns a versioned, JSON-ready description of the
operators you can use after creating a `Query`. It lists names, aliases,
supported positional forms, argument types and defaults, and result shapes.
Read it without loading a release or constructing an engine.

```python
import json
from teadata.query import describe_operators

print(json.dumps(describe_operators(), indent=2))
```

The result has `schema_version`, `receiver`, `operators` and
`release_methods` keys. A tuple
form lists its arguments after the operator name. A callable form represents
`query >> predicate`. Arguments list their supported type alternatives.
An alternative with `python_only: true` needs a live Python value, such as
a callable, an `EditionKey` or a `DataRelease`. `optional` means the slot
may be omitted, and `nullable` means you can pass `None` explicitly. For
example, the `observations` tuple requires the school year slot and accepts
`None` there. `availability: "custom_repository"` means a Query operator
needs a method outside the stock engine.

The operator list covers Query shifts. `release_methods` describes
`DataRelease.series()` with argument positions, types, defaults and result
methods. The `DataEngine` entry syntax and the
[`SourceTable.query()` expression syntax](#write-a-query-string) are separate.
Chained operators report missing and extra positional arguments. Use `True`
or `False` in flag slots, integers in count slots, and numeric coordinates
and distances. For `nearest`, `nearest_charter`, `radius`, and `knn`, pass
`None` as the seed to use the first input item. An explicit seed must resolve
to coordinates. `within` resolves a district or polygon as described above.
Use `release.catalog.editions()`, `release.catalog.variables()` and
`release.observation_years()` to find datasets, fields and periods in your
selected release.

## School years and source editions

Select by ending year. `2025`, `"2025"`, `25`, `"25"`, `"2024-25"`
and `"2024–2025"` all select 2024–25. Integers 0–99 and exactly
two-digit strings mean 2000–2099. A four-digit string selects the
literal year, including `"0025"` for a period ending in year 25.
A label requires a four-digit start and a consecutive ending year.
For example, `"1999-00"` resolves to 2000. These spellings work in
the observation and statewide-comparison methods and tuple operators.

`Query` starts from entities in the loaded snapshot. When the entity build pins
a native source, `observations` selects its fields by measurement school year.
Pass a supported year spelling for an exact selection. Omit it to use the
latest school year when every selected field has that same latest year and one
edition provides them together. The result is a lazy `SourceTable`.

```python
dataset = "tprs_student_membership_enrollment_reports"
field = "membership_total_students_count"
years = release.observation_years(dataset, field, scope="district")[field]
districts = engine >> ("district", "Austin ISD")
observations = districts.observations(dataset, field, school_year=2026)
period = observations.measurement_period
frame = observations.to_df()

# Read the direct publisher statewide observation from the same entity release.
texas = release.observations(dataset, field, school_year=2026, scope="state").to_df()
```

The equivalent chain operation is `districts >> ("observations", dataset,
2026, (field,))`. Use `None` in the year slot for latest. Empty entity queries
require an explicit `scope`. An unavailable year raises with the available
measurement years for each field. Use an exact `EditionKey` when several
eligible editions match. To reopen a saved entity
build, pass its pinned `DataRelease` as `release=` or enter its `using_release`
context. The DataFrame lineage records both the entity manifest and the native
manifest, edition, artifact and measurement period.

The same year selection works directly on a separately published native
`DataRelease`. Filter its source keys to reach historical entities absent from
the current snapshot. For several years, iterate the desired values from
`observation_years`. To pass each year back literally, use `f"{year:04d}"`.
Check each edition's definitions before comparing.
Discovery and planning may read selected catalog metadata, while execution
opens the data leaf. Online releases can fetch that metadata and the selected
leaf into their verified cache. Offline releases require those bytes locally.
The 2025 annual dropout edition measures school year 2024–25, even though
TEA published it in 2026. Check field periods separately from the report
period. See [entity build inputs](source-coverage.md) for snapshot coverage.


## Retrieve a time series

Use `release.series()` to collect several published fields across explicit
measurement years. Pass `ids` for a fixed public district or campus cohort.
An ID can be an integer or a string with or without an Excel apostrophe.
District IDs are padded to six digits and campus IDs to nine. Omit both
`ids` and `names` to read the full publisher roster.
This example requires a release containing all three PEIMS editions.

```python
series = release.series(
    "peims_special_education_by_primary_disability",
    "primary_specific_learning_disability_count",
    "primary_autism_count",
    scope="district",
    years=[13, "2018–19", 2022],
    ids=[227901, "'246909"],
)
frame = series.to_df()
one_year = series.for_year("2018–19")  # The same pinned SourceTable interface
```

Use `names=["Austin ISD"]` in place of `ids` for case-insensitive exact
name selection. An entity release uses its own directory by default.
With a native-only release, pass a specific entity release as
`entity_release`. This opens the verified entity directory, while the
observation tables remain lazy. Provenance identifies that directory and
records the requested names and resolved IDs. Use the directory edition
appropriate to your question. The directory match alone establishes no
historical name or affiliation.

Exact engine selectors `engine.district_named(name)` and
`engine.campus_named(name)`, and the exact-name forms of
`engine >> ("district", name)` and `engine >> ("campus", name)`, raise
`EntitySelectionError` on a miss or on multiple exact matches.
`error.to_dict()` returns a reason and candidate names with IDs. The exception message
shows up to five name and ID pairs. Structured ambiguity errors retain
all exact matches. Suggestions account for common punctuation and word-order
differences, preserve numbered names, and leave the original labels intact.
Use a wildcard for a search that can return several matches. A valid absent
numeric ID still yields an empty engine query.

Source fields in `observations()`, `observation_years()` and `series()`
accept case-insensitive names. Observation tables and series preserve the
canonical spelling. The `observation_years()` dictionary keys preserve
the requested spelling. Selection errors distinguish an unknown field from a field that is
unavailable for the requested school year. Suggested spellings are advisory.
A series requires the same canonical field in each selected edition.
Renamed fields need a reviewed mapping that accounts for changes in method,
population and denominator.

The output adds `measurement_school_year` and retains each source's key,
raw values, and status columns. It preserves year order and publisher row
order within each year. An ID with no published row for a year remains
part of the requested cohort in provenance. Edition-specific context
columns are null where the source lacks them. Select compatible
field types, source grain, and value support across years. A requested year with unavailable
fields or multiple revisions raises with the available choices. Pass
`editions={2019: key}` to pin a revision for that year.

`iter_batches()` scans the selected source leaves one year at a time.
`to_arrow()` and `to_df()` collect the selected data. The Arrow schema
metadata and DataFrame attrs carry `teadata_provenance`, including the
pinned source, period and selection for each year. Call `provenance()`
to inspect that record before opening data leaves. Use
`release.observation_years()` to discover field coverage and
`describe_operators()["release_methods"]` for the input contract.
Format discovered years as four digits when passing them back literally.
Review publisher
definitions and historical identity before calculating changes.

## Align two years by public ID

Use `series.pair_years(from_year=13, to_year=2022)` to align two years
already in the series. Both editions must publish one row per district or
campus for the selected year. The returned plan exposes `schema` and
`provenance()` before it opens data. Call `to_arrow()` or `to_df()` to
read the two endpoints.

The output retains both publishers' raw values and key columns, plus any
status columns the source provides, with `from_` and `to_` prefixes. The leading district or campus number is
the normalized alignment key. The two presence flags distinguish an
absent row from a published row with suppressed or missing values. A
requested ID remains in the output even when absent from both years.
Without an ID or name cohort, the output is the sorted union of the two
publisher rosters.

Alignment keeps source editions pinned. It checks both complete endpoint
projections for duplicate canonical IDs before selecting the cohort. When
a publisher has a physical `school_year` key, the supported spelling is the
bound full period, such as `2021-2022`. Pairing makes no determination about
historical entity continuity or whether changes in the measure are
comparable. Inspect both source definitions before calculating change.

## Select an entity's published source fields

An explicit native release can expose fields in dataset namespaces. Start
with a public district or campus from the engine and an exact release edition.

```python
sources = release.for_entity(district, edition="2026")
student = sources.tprs_student_membership_enrollment_reports
count = student.membership_total_students_count
fields = student.variables(r"^membership_total_students_")
table = student.select(*(entry["field"] for entry in fields))
```

`sources.datasets()` lists exact source identities without opening data files.
`variables()` reads the chosen source's catalog metadata. `select()` reads a
single native table and retains its keys and available raw and status columns.
Use `student["field_name"]` when a field name conflicts with a method.
Pass an exact key from `sources.datasets()` to `sources.dataset(key)` when a
name resolves to more than one table or revision. Scalar access requires one published row for
the entity. For a table with more than one row per entity, pass a PyArrow
expression to `select(..., where=...)` to narrow its grain.

For a partitioned TPRS Student release, prepare a campus index once and reuse
it for lookups. It reads the release's pinned report ledgers without opening
the underlying data tables.

```python
import pyarrow.dataset as ds

campus_index = release.tprs_student_campus_index(edition="2026")
campus_number = "001902103"
key = campus_index.edition_for(campus_number)
if key is not None:
    table = release.observations(
        "tprs_student_membership_enrollment_reports",
        "membership_total_students_count",
        school_year=2026,
        scope="campus",
        edition=key,
    )
    rows = table.where(ds.field("campus_number") == campus_number).to_arrow()
```

`edition` is the publisher's four-digit report edition. It does not set the
measurement year of every field. Pass the returned exact key and the
measurement year to `release.observations()`. A missing route means the
selected release did not admit a Student report for that campus in this
edition; it does not establish that the campus did not exist or that a
measure was zero. An index belongs to one release hash; rebuild it when
switching releases.

These views leave existing entity properties and `meta` keys in place. A raw
source field returns its published token. A typed field returns its published
projection, with missing or masked values represented by `None`. Select its
source support to distinguish those cases.

`Query.to_df(restore_original_headers=True)` uses reviewed publisher labels
where the bound snapshot supplies them. Unmapped fields keep their canonical
names. The DataFrame's `attrs["teadata"]` records the build and restored
label/code pairs. These describe the columns, including when callers have
edited entity values. Use this mode with queries of a single bound entity
scope. Duplicate output labels raise an error. To restore the headers of an
entire source table, use the [native export methods](data-releases.md#exact-retained-source-parity).

## Filter a source table before loading it into pandas

Use `release.table(edition_key)` for a lazy view of one native edition. For
example, with the district edition of the 2025–26 TPRS student source:

```python
table = release.table(edition_key)
count = table["membership_total_students_count"]
selected = table[(count >= 1_000) & (count < 20_000)][["membership_total_students_count"]]
frame = selected.to_df()
```

Column expressions use Arrow comparisons, `&`, `|`, `~`, `.isin()` and
`.is_null()`. A comparison excludes null values. Select a typed field to keep
its raw token and status in the result alongside its numeric value. Raw-only
tables compare the publisher's literal strings. A predicate may use columns
in the native table that are absent from the result. Unknown fields fail when
the scan binds. `select()` and `where()` offer named equivalents for column
selection and filtering. Each call returns a new view. `to_arrow()`, `to_df()`,
`to_csv()` and `to_excel()` execute the scan and read only the selected leaf.

The DataFrame records its manifest, edition and artifact identity in
`attrs["teadata"]`. Its `attrs["teadata_provenance"]` and the Excel
`Provenance` worksheet describe the exported columns, publisher codes and
definitions, measurement periods, report edition, value policy, and available
source references. The acquisition entries describe the edition and the
processing receipt identifies the exported artifact. Excel reports the
observed row count separately from the source description. The DataFrame
reports it in `attrs["teadata_export"]["rows"]`.

For raw TAPR tables, `restore_original_headers=True` restores the reviewed
headers in publisher order, including single-row codes where the crosswalk
specifies that layout. CSV retains its ordinary tabular layout. Write a
companion workbook using the same header mode as the data export.

```python
selected.to_csv("membership.csv")
selected.provenance().to_excel("membership-provenance.xlsx")
record = selected.provenance().to_dict()
```

`provenance()` can read release catalog resources but leaves the data leaf
unopened. Use `release.provenance(edition)` with a direct `release.to_csv()`
export. The worksheet's numbered canonical record rows contain the same
JSON in base64-encoded UTF-8 for machine readback. The description checksum
covers the structured record excluding the runtime citation and checksum.
Keep filter expressions and parameters with a file export if you need to
replay them. The record marks these filters as opaque and omits their values.

## Write a query string

The same scan can take a bounded query string with explicit `@` parameters.

```python
frame = table.query(
    "@minimum <= membership_total_students_count < @maximum",
    params={"minimum": 1_000, "maximum": 20_000},
).select("membership_total_students_count").to_df()
```

Use comparisons, `and`, `or`, `not`, `in`, `not in`, `is None` and
`is not None`. Field names are canonical identifiers. Membership follows
Arrow's `.isin()` behavior. To exclude nulls from a negated membership test,
also add `field is not None`. Query strings and total parameter use each have
an 8 KiB limit. Parameters accept only scalar values and bounded membership
lists. The syntax rejects function calls, attribute access, arithmetic and
implicit local variables.

Numeric literals retain their exact digits for integer and decimal columns.
For a threshold between two representable values, comparisons include the
same rows as an exact comparison against the original threshold. Membership
matches only representable values. Pass `Decimal` values for decimal
parameters. To compare a decimal column with a floating-point column, use an
explicit cast in an Arrow filter.

After materialization, the result supports pandas `.loc` and boolean
indexing. Use [pandas `.query()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html)
only with trusted expressions on that DataFrame because it can execute Python code.

## Compare a published measure with Texas

`Query.compare_statewide()` selects the publisher's statewide observation for
the exact local source edition. Results retain both edition identities, the
measurement period, raw tokens, value statuses and numeric values. A percent
comparison exposes its difference in percentage points as `difference_pp`.

Pass `school_year=2025` to select measurements for 2024–25 within the
pinned editions. A wildcard selects the reviewed metrics for that year.
A named metric must cover that same school year. An unavailable year
raises `ContractError` before the comparison reads observation data.
Omit the year or pass `None` to keep the reviewed periods of the selected
metrics. The tuple form accepts the year after the dimensions argument.

```python
result = districts >> ("compare_statewide", release, local, texas, "*", None, 2025)
```

Prepared [2026 TPRS student bundles](config.md#selected-2026-tprs-student-membership-and-enrollment-reports)
use this same interface for all 168 reviewed Membership/Enrollment count and
percent fields. Select the local and direct Texas editions from one explicit
bundle; engine entities supply identifiers only, not observations from their
own build. For example, with Austin ISD included in the selected intake:

```python
from teadata import DataRelease
from teadata.dataset_metadata import EditionKey

students = DataRelease("prepared/tprs-students-2026/release.json", offline=True)

def student_edition(scope):
    entry, = students.catalog.editions(
        dataset="tprs_student_membership_enrollment_reports",
        scope=scope, include_metadata=False,
    )
    return EditionKey.from_dict({name: entry[name] for name in EditionKey.__dataclass_fields__})

result = (engine >> ("district", "227901")).compare_statewide(
    release=students, local=student_edition("district"),
    statewide=student_edition("state"),
    metric="membership_total_students_count",
).first()
print(result.local_raw, result.statewide_raw, result.difference)
```

Both editions cover 2025–26 and use the same population and measure: Membership is not
Enrollment. Published percentages use Total Students within that same
population. Count differences remain counts; percent differences are percentage
points. The Texas observation comes from its state report, never a sum of the
selected local reports or an embedded benchmark. Omit `metric` for all 168 fields.

Selected coverage remains explicit. An entity absent from the bundle is
`unavailable`; an omitted campus grade row remains `missing`, with a null raw
token, rather than zero. Private campuses are `not_applicable`. Masked and
no-students tokens retain their separate statuses; a difference requires two
observed values. CTE, grouped disability, lagged discipline and embedded
benchmarks remain outside this comparison contract. No regional or longitudinal
comparison, default engine activation, or statewide local-entity coverage is implied.

For tested-grade STAAR, prepare a grade bundle containing the local and state
scopes using the [source preparation workflow](config.md#tested-grade-staar-sources).
Select exact catalog editions from that bundle. The example below uses a
2025–26 Grade 3 bundle and an existing engine containing Austin ISD. Set
`edition` and the bundle path together to select another prepared edition;
`2024`, `2025` and `2026` describe the 2023–24, 2024–25 and 2025–26
measurement years, respectively.

```python
from teadata import DataRelease
from teadata.dataset_metadata import EditionKey

edition = "2026"
release = DataRelease(f"prepared/staar-{edition}-grade-3/release.json", offline=True)

def edition_for(scope):
    dataset = f"{scope}_tapr_staar_tested_grade_3"
    entry, = (
        row for row in release.catalog.editions(dataset=dataset, include_metadata=False)
        if row["edition"] == edition
    )
    return EditionKey.from_dict({key: value for key, value in entry.items() if key != "edition_id"})

metric = (
    "staar_performance_reading_language_arts_grade_3_all_students_"
    "percent_at_meets_gl_standard_or_above"
)
comparison = (engine >> ("district", "227901")).compare_statewide(
    release=release,
    local=edition_for("district"),
    statewide=edition_for("state"),
    metric=metric,
)
result = comparison.first()
print(result.local_raw, result.local_status, result.statewide_raw)
print(result.period, result.difference_pp)
```

Omit `metric` to select all reviewed measures common to the two editions.
Tested-grade bindings cover 630 reviewed published-rate selectors across the
six grades in each of 2023–24, 2024–25 and 2025–26. These selectors have stable
names across the three editions; each binding retains its own publisher code,
original header, source field, period and source identity. A stable selector
does not establish that a student group's definition is comparable over time.
The raw source catalog retains numerator and denominator
fields alongside these rates. A separately published observed rate can be
compared when its underlying counts are masked.

The same interface supports 210 published rates for `staar_all_grades` in
2024–25 and 2025–26, and 210 per year for `staar_end_of_course` in
2023–24, 2024–25 and 2025–26.
Select the local and direct state editions from a release that contains
both scopes. For example, use
`staar_performance_all_subjects_all_students_percent_at_meets_gl_standard_or_above`
with the All Grades editions. These bindings cover the named report groups
within the STAAR accountability subset. SAT/ACT and the all-tested reports
are separate populations. The published
[All Grades (v0.0.133)](https://github.com/adpena/teadata/releases/tag/v0.0.133)
manifest contains all four 2025–26 scopes. The
[End-of-Course (v0.0.132)](https://github.com/adpena/teadata/releases/tag/v0.0.132)
manifest contains all four scopes for each of the three source editions.

Education service centers use their literal two-digit identifiers. A regional
comparison uses the same interface with a query of region codes and a bundle
containing the regional and statewide editions.

```python
from teadata.query import Query

region = Query(["13"], repo=None).compare_statewide(
    release=release,
    local=edition_for("region"),
    statewide=edition_for("state"),
    metric=metric,
).first()
print(region.identifier, region.scope, region.difference_pp)
```

Inspect the reviewed rate definitions and their interpretation references
before reading observations. This uses the same contract as the comparison.

```python
from teadata.staar_comparison import staar_rate_contract

district_edition = edition_for("district")
contract = staar_rate_contract(release, district_edition)
print(contract.metadata.period)
print(contract.metadata.metrics[metric])
for reference in contract.metadata.references:
    print(reference.reference_id, reference.url, reference.provenance_ids)
print(contract.value_policy)
print(release.catalog.metadata(district_edition.dataset, edition=district_edition).references)
```

For 2025–26 End-of-Course, `staar_rate_legend` is the interpretation
reference for the 210 reviewed rate fields. In the named TPRS reports
checked, `*` matched raw rate `-1`, which the comparison marks `suppressed`.
The report's `-` matched an empty raw rate string. The comparison preserves
that empty string and marks it `missing`. A numeric rate can remain observed
when the raw component counts are `-3`. The comparison does not infer those
counts or a masking threshold.

The raw bulk edition's `masking` reference is still `unreviewed`. TEA's
[TPRS glossary](https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/comprehensive-tprs-glossary.pdf)
dates the linked TAPR masking rules to 2024–25. The reviewed rate legend
does not govern the entire 2025–26 bulk table or other products.

The source must contain both selected editions. Each comparison requires the
same grade, school year, subject, student group and criterion. Suppression,
missing values and unavailable entities retain their individual statuses.
Observed zero is numeric. A difference is available when both rates are
observed. The population is the STAAR accountability subset described by
[TEA's TPRS guidance](https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/texas-performance-reporting-system).

## Look up names in a dated AskTED directory

The [offline archive reader](spatial.md#archived-administrative-geography)
accepts a pinned receipt and season. It resolves names case-insensitively
within that directory. An ambiguous or suggested name reports its ID
and, where the selected edition has a single usable value, its site city
and administrative county.

Pass other verified directories explicitly as `aliases_from` to search
names observed in those editions as well. Results retain the exact names,
source receipts and record numbers. Repeated occurrences for one ID are
one candidate. A name used by different IDs remains ambiguous.
Place labels come from the selected directory. A historical ID absent
from that directory has no inferred city or county. Resolve names or
cohorts first, then pass the IDs explicitly to `series()` for the desired
performance-report periods.
