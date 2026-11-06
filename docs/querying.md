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

## Common Operators

- `>> ("campuses_in",)` — expand the current district query into its district-operated campuses (excludes charters and private schools)
- `>> ("privates_within", district)` — private schools inside or on the reported district boundary (`covers`); geographic containment is separate from institutional governance, and border points can match multiple districts
- `>> ("filter", predicate)` or `>> (lambda x: ...)`
- `>> ("sort", key_fn, descending: bool=False)`
- `>> ("take", n)`
- `>> ("nearest_charter", coords, mile_limit, k)` — from repo-level query context
- `first()` — take first element
- Attribute fall-through: `Query.attr` proxies to `first().attr` for convenience.

!!! tip "Chaining"
    You can mix spatial and attribute filters easily:
    
    ```python
    engine >> ("within", d, "campuses") >> (lambda c: c.rating in {"A","B"})
    ```

## School years and source editions

`Query` starts from entities in the loaded snapshot. When the entity build pins
a native source, `observations` selects its fields by measurement school year.
Pass the ending year as an integer for an exact selection. Omit it to use the
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
`observation_years` and check each edition's definitions before comparing.
Discovery and planning may read selected catalog metadata, while execution
opens the data leaf. Online releases can fetch that metadata and the selected
leaf into their verified cache. Offline releases require those bytes locally.
The 2025 annual dropout edition measures school year 2024–25, even though
TEA published it in 2026. Check field periods separately from the report
period. See [entity build inputs](source-coverage.md) for snapshot coverage.

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

These views leave existing entity properties and `meta` keys in place. A raw
source field returns its published token. A typed field returns its published
projection, with missing or masked values represented by `None`. Select its
source support to distinguish those cases.

`Query.to_df(restore_original_headers=True)` uses reviewed publisher labels
where the bound snapshot supplies them. Unmapped fields keep their canonical
names. The DataFrame's `attrs["teadata"]` records the build and restored
label/code pairs. These describe the columns, including when callers have
edited entity values. Use this mode with queries of a single bound entity
scope. Duplicate output labels raise an error. To export both header rows for an
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
`attrs["teadata"]`. For raw TAPR tables, `restore_original_headers=True`
restores the selected labels and codes in publisher order. CSV and Excel
contain the selected rows and columns. Keep the manifest and filter with a
file export to reproduce the subset.

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

For 2025–26, the same interface also supports 210 published rates each for
`staar_all_grades` and `staar_end_of_course`. Select the local and direct
state editions from a release that contains both scopes. For example, use
`staar_performance_all_subjects_all_students_percent_at_meets_gl_standard_or_above`
with the All Grades editions. These bindings cover the named report groups
within the STAAR accountability subset. SAT/ACT and the all-tested reports
are separate populations. The published
[All Grades (v0.0.133)](https://github.com/adpena/teadata/releases/tag/v0.0.133)
and [End-of-Course (v0.0.132)](https://github.com/adpena/teadata/releases/tag/v0.0.132)
manifests each contain all four 2025–26 scopes.

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

contract = staar_rate_contract(release, edition_for("district"))
print(contract.metadata.period)
print(contract.metadata.metrics[metric])
for reference in contract.metadata.references:
    print(reference.reference_id, reference.url, reference.provenance_ids)
print(contract.value_policy)
```

The source must contain both selected editions. Each comparison requires the
same grade, school year, subject, student group and criterion. Suppression,
missing values and unavailable entities retain their individual statuses.
Observed zero is numeric. A difference is available when both rates are
observed. The population is the STAAR accountability subset described by
[TEA's TPRS guidance](https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/texas-performance-reporting-system).
