# Configuration & Resolution

All external data locations and schema expectations are declared in a single YAML (or TOML) config.

## Config Structure

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

## Programmatic Access

```python
from teadata.teadata_config import load_config
cfg = load_config("teadata_sources.yaml")

# Resolve “best” file for a year (exact match or nearest prior)
resolved_year, path = cfg["data_sources", "accountability", 2025]
print(resolved_year, path)

# Quick availability report
print(cfg.availability_report())
```

### Independent spatial editions in a build

The reporting year and geometry editions can differ. Once the corresponding
spatial entries have been acquired and qualified, select their exact catalog
keys in the existing options mapping, for example:

```yaml
options:
  spatial_years:
    districts: 2026
    campuses: 2025
```

This example does not activate or qualify either source. Each override must be
an integer (not a Boolean) and name an existing spatial dataset and exact dated
catalog entry. Missing editions fail before source reads; an override never
falls back to an earlier geometry edition. Without an override, the build keeps
its existing reporting-year selection and nearest-prior fallback.

The loader, supplied-path guard and module CLI share
`cfg.spatial_selection(dataset, reporting_year)`, which returns the `year` and
`exact` arguments for the existing `resolve_dataset` and `load_df` APIs. Direct
calls to those APIs continue to honor their explicit arguments. Build metadata
retains each source's actual requested/resolved catalog year, publisher version
and declared reference period; a catalog year is not a geometry vintage or a
publication date. The supplied-path guard, recorder and geometry loader use the
same loaded configuration snapshot, so changing the configuration file during a
build does not silently switch the selected geometry edition.

## Cross-dataset Joins

Use the provided helpers to normalize `district_number` and join TAPR/PEIMS/etc. without fighting column name variations.

## Dataset Metadata and Provenance

Dataset entries accept both the existing path/URL shorthand and a structured
entry. The same structure works in YAML and TOML:

```yaml
data_sources:
  campus_tapr_student_staff_profile:
    2025:
      path: data/tapr/CAMPPROF_2024-2025_FINAL.xlsx
      metadata:
        title: TAPR campus student and staff profile
        publisher: Texas Education Agency
        period:
          kind: school_year
          start_year: 2024
          end_year: 2025
    latest: 2025
```

```toml
[data_sources.campus_tapr_student_staff_profile.2025]
path = "data/tapr/CAMPPROF_2024-2025_FINAL.xlsx"
[data_sources.campus_tapr_student_staff_profile.2025.metadata]
title = "TAPR campus student and staff profile"
publisher = "Texas Education Agency"
[data_sources.campus_tapr_student_staff_profile.2025.metadata.period]
kind = "school_year"
start_year = 2024
end_year = 2025
```

The numeric catalog key selects a configured version. It is distinct from the
reference period and publisher's publication/revision dates. In particular, an
existing catalog key does not establish whether it means a school year's start
or end. Existing entries retain an explicit `unknown` period until reviewed.

Supported metadata fields:

| Field | Meaning |
| --- | --- |
| `title`, `publisher`, `version` | Dataset identity and the publisher's version label |
| `period` | `school_year`/`fiscal_year` with consecutive `start_year` and `end_year`; `calendar_year` with `year`; `as_of` with an ISO date; or `unknown` |
| `source_url`, `download_url` | The publisher's catalog/landing page and the direct source-file URL |
| `published_at`, `revised_at` | Known publisher dates or timezone-aware ISO timestamps |
| `license`, `citation`, `identifiers` | Verified reuse/citation information and identifiers such as DOIs |
| `notes` | Source-specific definitions, caveats, suppression rules, or other research context |
| `provenance_ids` | SHA-256 identifiers of immutable acquisition receipts; typed processing dependencies belong in processing inputs |
| `references` | Per-version dictionary, glossary, masking, column schema, or other documentation with URL, availability/review status, exact review hash, and acquisition receipts |
| `metrics` | Physical field, unit, and optional per-metric reference period |
| `value_policy`, `estimation` | Source token interpretation and a separately versioned estimation assumption |

Omit facts that are unknown. Do not infer download dates from file modification
times, today's date, filenames, or a deployment date. HTTP `Last-Modified` is an
observed response header and does not establish the publisher's release date.
Metadata URLs must use HTTP(S) without embedded credentials. Unknown fields and
inconsistent periods are rejected to catch mistakes before a build.

### Querying Declared Metadata

```python
from teadata.teadata_config import load_config

cfg = load_config("teadata/teadata_sources.yaml")
source = cfg.resolve_dataset("campus_tapr_student_staff_profile", 2026)
print(source.resolved_year)          # 2025, if that is the selected fallback
print(source.metadata.period.label)  # "2024-2025" for the structured example
print(source.is_fallback)            # True
metadata_for_json = source.to_dict()
```

`resolve()` and `load_df()` keep returning `(resolved_year, path_or_dataframe)`.
Their default selection allows the nearest prior catalog year. `strict=True`
on `resolve()` or `resolve_dataset()` preserves that prior-year fallback; it
rejects unavailable/undated fallback rather than requiring an exact year.
Use `exact=True` on `resolve()`, `resolve_dataset()`, or `load_df()` to require
an explicit numeric catalog year and raise before source I/O if it is absent:

```python
source = cfg.resolve_dataset("campus_tapr_student_staff_profile", 2025, exact=True)
year, frame = cfg.load_df("campus_tapr_student_staff_profile", 2025, exact=True)
```

An exact catalog key does not establish a metric's reference period. Read the
selected source metadata and per-metric periods separately; an undated `current`
entry cannot satisfy an exact-year request.

`to_json()` preserves structured metadata and the shorthand for legacy entries.
An undated `current` entry has `resolved_year=None` and `is_fallback=None` in the
new API; its legacy synthetic `9999` year is not presented as a real year.

```bash
uv run teadata-config describe teadata/teadata_sources.yaml \
  data_sources campus_tapr_student_staff_profile 2026
```

These APIs describe configured sources. Runtime consumers use `current_release().catalog`, which describes successfully loaded inputs and is bound to every artifact in the release. See [Dataset contracts and data releases](data-releases.md) for researcher APIs, masking policies, estimates, validation, and rollout.

### Immutable Acquisition and Processing History

`ProvenanceStore` preserves JSON receipts and exact file bytes in a portable
directory. Receipt identifiers and artifact identifiers are SHA-256 hashes.
Existing records and objects are never overwritten; reads verify receipt hashes
and `verify_artifact()` verifies archived bytes. Each real acquisition records
its own timestamp even when the URL or downloaded contents remain the same.

```python
from teadata import ProvenanceStore

store = ProvenanceStore("provenance")
source = cfg.resolve_dataset("campus_tapr_student_staff_profile", 2025)

# Observe an existing file without claiming to know when it was downloaded.
record_id = store.record_acquisition(source.path, source)
record = store.read_record(record_id)
assert record["downloaded_at"] is None
archived_file = store.verify_artifact(record["artifact"]["sha256"])
history = store.history(source.dataset)
```

A downloader should call `record_acquisition()` after a successful transfer and
supply its observed, timezone-aware `downloaded_at`, requested/final URLs, and
response headers. Receipts record the actual byte count and SHA-256, declared
source metadata at that time, selected catalog year, request parameters, and
an observation timestamp. Only ETag, Last-Modified, Content-Type,
Content-Encoding, Content-Length, and Content-Disposition headers are retained.

Transformations link to their input receipts, retaining the processing code
version, parameters, validation results, and each output artifact:

```python
processed_id = store.record_processing(
    dataset=source.dataset,
    inputs=[record_id],
    outputs={"normalized": "output/normalized.csv"},
    processor="my_pipeline.normalize",
    code_version="git:<exact-commit>",
    parameters={"catalog_year": source.resolved_year},
    validation={"schema": "passed", "row_count": 123},  # actual check results
)
```

When validation results are absent, the receipt stores `null`, not a passing
result. Input records must exist and verify successfully. Several source files
can feed a processing record, and later processing can reference that record to
preserve the complete transformation chain. A publisher replacing a file at the
same URL produces a new artifact hash while the earlier bytes remain archived.

The registry layout is `records/<receipt-sha256>.json` and
`objects/<content-sha256>`. Move or publish these together to retain resolvable
history. The repository ignores `provenance/objects/` to keep large source
archives out of ordinary Git history; preserve those objects in your durable
artifact storage alongside the versioned JSON receipts. The API performs no
downloads or background refreshes on its own.
Existing downloaders and builders need explicit integration; old artifacts do
not gain verified provenance merely because their configuration is extended.

## Maintenance Script Paths

Before observing or pinning a packaged schema resource, serialize it with
`teadata.manifest.canonical_json(document)` plus one newline and run the existing
`test_every_packaged_version_and_declared_profile_has_an_offline_schema` check.

Catalogs that bind a reviewed header crosswalk expose the publisher's source
codes, exact human labels and canonical Python field names without reading
source datasets or current YAML:

```python
records = catalog.header_records(
    "campus_tapr_student_staff_profile", edition="2025", table="student"
)
rename = catalog.header_mapping(
    "campus_tapr_student_staff_profile", edition="2025", table="student",
    from_="code", to="field", columns=df.columns,
)
renamed = df.rename(columns=rename)
```

For renaming your own data without a data release, use the explicit packaged
source-contract entry point instead. No raw datasets or cache artifacts are
required, and `TEADATA_RELEASE` is not consulted:

```python
from teadata import load_config

contract = load_config()  # Packaged teadata_sources.yaml; an explicit path is also accepted.
records = contract.header_records(
    "district_tapr_student_staff_profile", edition="2025", table="student"
)
rename = contract.header_mapping(
    "district_tapr_student_staff_profile", edition="2025", table="student",
    from_="code", to="field", columns=df.columns,
)
```

This selects the packaged source contract's reviewed mapping, which identifies
the exact edition and retained raw input SHA. It does not claim that those
facts belong to an installed data release. Release-bound catalog queries stay
independent and never fall back to this configuration.

Records are detached ordinary dictionaries suitable for a pandas/Polars table
or `pyarrow.Table.from_pylist(records)`. They include dataset, selected edition,
table, position, exact code/label, canonical field, input and header-schema
hashes, and mapping version. Rename dictionaries also support `label` and
reverse directions. Ambiguous mappings, duplicate external headers and
collisions with unchanged destination columns raise errors. Source codes are
recorded metadata, not alternate canonical query-field authorities.
Passing `columns` also limits the dictionary to headers actually present, so
partial tables work with libraries that reject unknown rename keys.

The same reviewed resource in `DatasetDefinition.reader.column_schema` drives
ETL and derives prepared column order. `BuildRecorder` binds its exact bytes
once into the catalog, together with the selected input, edition and read
schema. Lookup never falls back to package files or a newer configuration.
Older catalogs, including published 0.0.126, explicitly report the crosswalk
unavailable; the new contract requires a separately qualified future bundle.
Unreviewed years/datasets are not inferred. A retained-input comparison and a
new local observation do not establish an original download or historical
transformation event. Source layout and naming remain separate from masking
policies and measurement interpretation.

Maintenance scripts use the package-adjacent `teadata/data/` directory by
default. Moving a checkout or invoking a script from another working directory
does not change that default. Raw source datasets are ignored by Git and must
be supplied separately; the tracked snapshots are sufficient for normal runtime
queries but do not contain every original source workbook.

Set `TEADATA_DATA_DIR` to point the scripts at an existing source-data tree:

```bash
export TEADATA_DATA_DIR="$HOME/datasets/teadata"
uv run python -m teadata.scripts.rename_tepsac_headers
uv run python -m teadata.scripts.build_tapr_profiles \
  --edition 2024 --reference-store provenance --output-dir prepared/tapr-2024
```

The variable names the directory containing `tapr/`, `shapes/`, `tepsac/`, and
`campus_peims_financials/`. `~` and environment variables in this value are
expanded; a relative value is resolved from the invoking working directory.
The default and the override use the same layout. For an installed package,
prefer an external writable data directory over writing inside site-packages.

Each data-processing script also accepts explicit paths, which take precedence
over the default data directory:

```bash
uv run python -m teadata.scripts.rename_tapr_campus_headers \
  --input raw/campus.csv --reference raw/data_elements.xlsx \
  --output output/campus.xlsx
uv run python -m teadata.scripts.download_campus_peims_financials \
  --year 2025 --store provenance --district 227901 --campus 227901002
uv run python -m teadata.scripts.geocode_tepsac_data \
  --input raw/schools.xlsx --output output/geocoded.csv
uv run python -m teadata.scripts.flatten_districts_geojson \
  --input districts.geojson --output districts_flat.geojson
uv run python -m teadata.scripts.build_tapr_profiles \
  --edition 2025 --family student_profile --input-dir raw/ \
  --reference-store provenance --output-dir prepared/tapr-2025-students
uv run python -m teadata.scripts.build_tapr_profiles \
  --edition 2025 --family staff_profile --input-dir raw/ \
  --reference-store provenance --output-dir prepared/tapr-2025-staff
```

Campus PEIMS financial actuals acquisition supports fiscal years ending in 2024
and 2025. It retains the publisher's district selector, the campus roster for
each selected district, and the original individual-campus Excel workbooks in
the provenance store. Pass `--district` more than once to select multiple
districts. Omit `--campus` to acquire every campus listed for the selected
districts. The command returns receipt IDs and distinguishes listed campuses
from acquired workbooks. Rerun the same selection to resume from verified
receipts. Use `--refresh` to retain a new observation alongside earlier bytes.
Uncached acquisitions run serially with a pause between requests.
This command acquires source material. Field interpretation, masking review,
and catalog or release activation are separate steps.

To materialize a complete FY2025 campus intake offline, save the acquisition
command's JSON and compute its SHA-256. Pass that pin to the existing financial
refresh command with a new output directory.

```bash
uv run python -m teadata.scripts.refresh_financials \
  --store provenance --campus-intake fy2025-campus-intake.json \
  --campus-intake-sha256 "$INTAKE_SHA256" --output prepared/fy2025-campus-actuals
```

The campus mode verifies the receipt graph, exact year, financial basis,
and complete campus roster for every selected district. It accepts the
reviewed FY2025 campus workbook format. `campus_reports.parquet` retains
the original A3 membership text and reported count separately from
`campus_financial_facts.parquet`, which retains each amount, share, or
per-student cell's raw lexeme, section, literal line label and occurrence,
column role, worksheet coordinates, and original receipt. `MATERIALIZATION.json`
identifies the verified readback and the separate processing receipt in the
selected provenance store. The processing receipt lists the campus report
notes, raw account dictionary, and related definitions as publisher reference
links. Their semantic review is separate from this raw projection.

The candidate retains source zeros and negatives as published. It records a
hold reason for normalization, including a distinct hold for per-student
cells when reported membership is zero. The output does not change the active
catalog or the historical FY2024 CSV workflow. Campus intake and the financial
command's district workbook `--config` mode run separately.

The [FY2025 dollar field qualification](financial-sources.md#qualify-selected-fy2025-dollar-fields)
uses a caller-supplied witness case and runs separately from workbook admission.
Its verified schema-4 candidate is separate from the active snapshot and the
profile supplement's `--native-release` gate.

The TAPR command prepares every declared publisher column in the selected
single-table family for the requested geographies into a **new** directory.
It validates all inputs before staging any output. Its
Parquet files retain raw strings, source crosswalks, annotation records and
immutable acquisition/processing receipts. Existing directories and retained
reference workbooks are never replaced. These outputs are source preparations,
not an activated release or typed scientific measurements. The old year-specific
builder and the 75-column district renamer are retired. Supported single-table
families use this explicit preparation path.

`--reference-store` supplies the canonical acquisition receipts and exact bytes
for established reference links. The producer retains only their selected
immutable closure, preserving record IDs and observation dates. A missing
claimed receipt or artifact fails preparation. Newly observed crosswalks bind
their own receipts; unreviewed link-only documents remain explicitly not
acquired. Portable raw-source locators remain unchanged in public reader
metadata and resolve against the configured data directory only when selected.

Coverage refers to the declared inputs: 2024 `PROF` contains 407 campus and 420
district columns, including 34 lagged annual-graduate fields in each scope.
The 2025 `STUD`/`STAF` source preparation remains separate from `GRAD`
Graduation Profile. Select the latter explicitly:

```bash
uv run python -m teadata.scripts.build_tapr_profiles \
  --edition 2025 --family graduation_profile --input-dir raw/ \
  --reference-store provenance --output-dir prepared/tapr-2025-grad
```

`campus_tapr_graduation_profile` and `district_tapr_graduation_profile` declare
editions 2024 and 2025, measuring 2022–2023 and 2023–2024 annual graduates.
The 2024 period comes from the annual glossary, pages 15–16. Its publisher
dictionary heading instead says 2023–2024; the source notes retain that
contradiction alongside the exact CSV labels, selection page and references.
The current 2024 GRAD observations remain independent of retained 2024 PROF;
their shared-cell literal `.` to blank changes remain visible in the audit.
They retain 42 metrics
and four/two canonical identity/history attributes in separate wide leaves;
identity/name observation dates remain unknown. Their source-only crosswalk has
no historical prepared-input hash and requires the explicit `source` profile.
The source metadata and availability report describe this declared coverage,
not inclusion in a released catalog. `cfg.is_source_only(dataset, 2025)` reads
that flag from the reviewed crosswalk without opening source data. GRAD1–4 readiness/assessment families and
other annual TAPR families are outside this preparation target.

All 42 GRAD source codes and exact human labels are published in CSV records
2 and 1. The acquired publisher dictionary documents only 39 metric codes per
scope. `CA0GS24N`, `CA0GM24R`, `CA0GS24R` and their district `D` counterparts
remain independent fields with `dictionary_status: not_documented`, null type
and description, and the original CSV-header reference. Documented dictionary
descriptions are retained without treating a similar label or equal value as
semantic equivalence. Every metric uses the verified 2023–2024 reference period;
units remain explicitly unknown until individually reviewed. Raw blanks and
zeros are preserved without assigning a missingness or suppression cause.
The producer requires and retains the original acquisition receipts and bytes
for source-only datasets; a new local observation does not replace download
history. GRAD is never automatically merged with STUD/STAF or entity metadata.

The current 2024 STUD and STAF observations have separate source-only datasets:
`campus_tapr_student_profile`, `district_tapr_student_profile`,
`campus_tapr_staff_profile` and `district_tapr_staff_profile`. Select
`--edition 2024 --family student_profile` or `--family staff_profile` with the
same producer, input directory and canonical reference store. Each produces
one independent table per scope: STUD has 243/242 campus/district columns and
STAF has 106/118. These observations do not replace retained PROF or fill fields
absent from the selected family. PROF retains its 407/420 columns, including
38/42 codes absent from the current three-family union.

The mixed STUD/STAF tables have an explicit unknown reference period, separate
from publisher edition 2024. Source-reviewed mobility and attrition percent
fields declare 2022–2023, supported by their current dictionary descriptions
and annual glossary page 32. Total enrollment count and total staff FTE declare
the 2023–2024 context documented on pages 28 and 34. Enrollment is an October
snapshot, not an annual cumulative count; no exact observation date is invented.
Staff FTE is not headcount and its unit remains unknown in the current vocabulary.
Other periods and units remain unknown. The one header projection uses exact registered
`MetricDefinition.field` matches: explicit field periods refine unknown table
periods; an omitted registered period uses selected dataset metadata; an
explicit unknown stays unknown. Identities and undeclared fields never inherit
a report year. Conflicting known table/field periods or conflicting definitions
fail before source rows are opened. Config and a frozen catalog use this same
projection, with the catalog consulting only its own bound metadata.

The new crosswalks preserve all ordered dictionary occurrences, including
identical repeated district definitions, and retain dictionary-only codes
without creating CSV columns. Campus staff `CPSCTOFC` remains undocumented;
dictionary-only `CPSCTOST` is not an alias. Original HTTP receipts retain their
original composite dataset names. The reviewed `csv_headers` source-notes
reference binds exact raw bytes to each independent derivation without
rewriting those receipts. Raw dots, blanks, zeros and the six observed new
positive numeric lexemes remain distinct; no revision cause is inferred.

The `source` reader profile exposes complete header contracts without reading
source rows, for example
`cfg.header_records("district_tapr_student_staff_profile", edition="2024", reader_profile="source")`.
`cfg.load_df(..., reader_profile="source")` explicitly reads the full raw edition;
the default reader continues to use its exact retained prepared workbook.
Raw source preparation cannot run inside a retained-format release build.
An explicit profile may declare a complete `metadata` record. Selecting it with
`cfg.resolve_dataset(..., reader_profile="source")` replaces the base
`DatasetMetadata` through the canonical parser; no base fields or references
are merged. Its version must match the declared source version, and references
belong only to that record, not a second profile-level `references` list.
Header lookup, source preparation and build recording select this same record.
Calls without `reader_profile` retain the base metadata and reader behavior.

The 2024 full PROF profiles select raw-only metadata: unknown period, no metric
definitions, no estimation and no applied value policy. Complete raw strings,
including masking-looking tokens, remain uninterpreted. The retained annual
dictionary, glossary and masking references provide context, not blanket
qualification of an indicator's scientific meaning. A catalog with no selected
policy rejects policy interpretation; it does not synthesize a raw-token
policy or numeric status. Legacy default metadata and its methodology gates
remain unchanged. The fresh source observation IDs retain their genuine HTTP
dates; the older local observations are not rewritten or newly qualified.
Source identities and historical names are lossless period-specific attributes,
not additional metric keys or automatic `.meta` hydration. No current geometry
roster limits source coverage. Unknown field units and reference periods remain
unknown; the existing qualified metric bindings retain their lagged periods.

The 2024 `campus_tapr_reference` / `district_tapr_reference` and
`campus_tapr_dropout_attendance` / `district_tapr_dropout_attendance` branches
are also source-only. Select `reader_profile="source"` explicitly to discover
their complete 22/12 and 166/164 ordered fields. One reviewed crosswalk binds
the exact two CSV header records, finite data ranges, original raw checksums
and dictionary occurrences. Human-readable field names retain a leading label
year after its literal Campus/District word; `county_number` and `esc_region`
omit only the label's digit-width naming prefix. The original codes, labels
and years remain available through `header_records` / `header_mapping`.

All 54 undocumented DROP_ATT fact columns per scope remain present with unknown
type and definition. Every field period is unknown, with no metric, estimation
or applied value policy; interpretation therefore fails rather than assigning
a status from `N`/`D`/`R`, a label year, or a masking-looking string. REF dictionary
Type/Length values are literal publisher metadata, not scientific types, units
or CSV-value limits. These current-interface families are independent of legacy
GRAD. Their original HTTP dates and annual context receipts remain immutable;
the schema has its own genuine local observation. This registration supplies
contracts, not prepared outputs, released/default facts or app activation.

The 2024 `campus_tapr_four_year_longitudinal_rate` and
`district_tapr_four_year_longitudinal_rate` source profiles retain the complete
228/226-column COMP4 observations: 224 facts per scope and their original
campus/district identity columns. Use the explicit `source` profile and the
existing `four_year_longitudinal_rate` preparation family. The single curated
crosswalk preserves both exact CSV header records, finite data ranges and raw
checksums. Code-based `header_mapping()` distinguishes repeated labels; a
label-only rename remains ambiguous and is rejected.

Each dictionary occurrence exposes its literal anchored `publisher_context`.
The 182 ordinary fact names stay unchanged; the 42 federal names per scope
include the exact section title, `4-Year Federal Graduation Rate Without
Exclusions (Gr 9-12)`, as a naming qualifier. Original codes, labels,
descriptions and source anchors remain discoverable. Literal 2023 text does
not establish the reference window of publisher edition 2024. The retained
annual [glossary](https://rptsvr1.tea.texas.gov/perfreport/tapr/2024/glossary.pdf)
and [masking page](https://rptsvr1.tea.texas.gov/perfreport/tapr/2024/masking.html)
are linked as available references with their original acquisition receipts.
Two all-students Class of 2023 graduate rates per scope have reviewed metadata.
Both are percentages with an August 31, 2023 diploma cutoff. TEA's ordinary
longitudinal and federal graduation rates use different cohort inclusion rules.
The retained reports show 95.2% ordinary and 93.1% federal for Garland High
School, and 93.1% ordinary and 90.9% federal for Austin ISD. Marathon's campus and
district report columns both show asterisks in these fields, corresponding to
`-1` in the raw download under TEA's small-group masking rule. Blank download
tokens remain unexplained. These checks apply to the two graduate rate fields in
each scope. All other field periods and units remain unknown. Preparation keeps
all source values as strings, applies no value policy or estimation, and retains
identity attributes as source history. Original acquisition dates stay intact,
and the generated schema has its own local observation.

The 2024 `campus_tapr_staar_school_progress` and
`district_tapr_staar_school_progress` declarations require the explicit `source`
profile. Their single curated crosswalk retains all 2,212/2,210 physical columns:
2,208 returned facts per scope and four/two canonical identity attributes. Both
logical CSV headers retain their exact labels and codes; the first header spans
four physical lines. Names use the existing sanitizer on literal labels, with
no aliases or code suffixes. The original raw observations and dates stay intact.

Each table's existing `dictionary_records` retains all 2,400 ordered publisher
occurrences and source anchors. The 192 dictionary members absent from each
raw response remain dictionary metadata: 96 complete definitions and 96
code-only observations. `header_records()` and `header_mapping()` expose the
actual columns only. Compute dictionary-only membership from these two existing
collections; it does not create empty value columns or missing/masked statuses.
No cause or publisher-family completeness is inferred from the difference.

Literal edition/year text, descriptive `Number` types and linked annual
glossary/masking documents do not assign field-specific scientific periods,
units, token rules, applicability or comparability. Those remain unreviewed,
with no value policy, metric definitions or estimation. Original dictionary
section context is unknown and is not synthesized. The charter-transfer
midpoint never applies to these fields. These declarations are source metadata
only; actual School Progress preparation is held for the separately reviewed
bounded producer migration. No released/default catalog or app behavior is
activated by this source contract.
Registration does not prepare, publish or activate these sources.

The 2025 outcome originals are also explicit `source` opt-ins:

| Dataset / preparation family suffix | Scope | Publisher file | Returned columns | Ordered dictionary occurrences |
| --- | --- | --- | ---: | ---: |
| `four_year_longitudinal_rate` | Campus | COMP4 | 228 | 224 |
| `four_year_longitudinal_rate` | District | COMP4 | 226 | 224 |
| `four_year_longitudinal_rate` | State | COMP4 | 182 | 224 |
| `college_career_military_readiness` | Campus | GRAD1 | 279 | 396 |
| `college_career_military_readiness` | District | GRAD1 | 277 | 396 |
| `college_career_military_readiness` | State | GRAD1 overall CCMR selection | 33 | 396 |
| `college_readiness_components` | Campus | GRAD2 | 125 | 165 |
| `college_readiness_components` | District | GRAD2 | 123 | 165 |
| `college_readiness_components` | State | GRAD2 | 121 | 165 |

Prefix each dataset with `campus_tapr_`, `district_tapr_`, or `state_tapr_` by scope. The sources
retain the exact CSV labels, codes, 9,084 campus rows and 1,208 district rows
per family, and every dictionary occurrence. COMP4
labels identify Class of 2024; the applicable glossary describes entry into
ninth grade in 2020–21 and a diploma cutoff of August 31, 2024. Its ordinary
longitudinal and federal without-exclusions populations retain separate code
identities and context-qualified names. GRAD1 and GRAD2 describe annual
graduates labelled 2024, not current student Membership or Outcomes Bonus.
GRAD2 AP/IB criteria and dual-credit readiness describe attainment. Graduate
course participation measures a separate outcome. Repeated denominator
dictionary contexts remain discoverable without duplicating physical columns.

Two COMP4 Class of 2024 rates at campus and district scope have a reviewed percent unit and an
August 31, 2024 diploma cutoff. The Austin ISD attendance and graduation report
matches the ordinary longitudinal rate (`DAGC4X24R`, 92.3%) and the federal rate
without exclusions (`DAGC424R`, 90.2%). Garland High School confirms the campus
rates (`CAGC4X24R`, 93.4%, and `CAGC424R`, 91.7%). Marathon's two campus rates
show asterisks in the report and `-1` in the CSV. Sam Houston Middle School's
two campus rates show dashes in the report and blank CSV fields. The reason
for other blanks remains unknown. The two graduation cohorts follow different
inclusion rules.

The state COMP4 source retains all 182 offered fields and every dictionary
occurrence. The statewide selection offers `SAGC424R` but no `SAGC4X24R`.
The dictionary repeats the available code under both longitudinal and federal
headings. The [Garland High School Report Card](https://rptsvr1.tea.texas.gov/cgi/sas/broker?_debug=0&_program=perfrept.perfmast.sas&_service=marykay&ccyy=2025&id=057909002&lev=C&prgopt=reports%2Fsrc%2Fsrc.sas)
names the Class of 2024 federal rate without exclusions and gives Texas at
90.7%. The [state longitudinal table](https://rptsvr1.tea.texas.gov/acctres/completion/2024/state.html)
corroborates the CSV's 367,549 graduates and 405,019 cohort members.
`SAGC424R` retains its original longitudinal label. TEA's
[completion and dropout report](https://tea.texas.gov/reports-and-data/school-performance/accountability-research/dropcomp-2023-24.pdf)
explains that state accountability exclusions do not apply to statewide rates
and confirms 367,549 and 90.7% in the Federal Report Card cross-tabulation
(PDF pages 24, 144 and 150).
`Query.compare_statewide` supports the exact federal metric across the three
editions. An ordinary statewide benchmark using the local state-accountability
exclusions remains unbound.
Verify the retained publisher witnesses with:

```sh
uv run python -m teadata.scripts.verify_tapr_state_grad2 --family COMP4 --store provenance
```

Five district GRAD1 and GRAD2 rates have a reviewed percent unit and
2023–24 annual-graduate period. The same-edition Austin ISD TPRS report
matches the raw downloads for each.

| Family | Rate | All students | Economically disadvantaged |
| --- | --- | --- | --- |
| GRAD1 | CCMR | `DA0GP24R` (87.8%) | |
| GRAD2 | AP/IB exam criterion | `DA0GY24R` (36.4%) | `DE0GY24R` (16.0%) |
| GRAD2 | Dual-credit criterion | `DA0GUA24R` (35.6%) | `DE0GUA24R` (24.4%) |

Six campus rates have the same reviewed unit and annual-graduate period. Garland
High School, Jack C Hays High School and Pflugerville High School match the
original CSV rates in same-edition TPRS. The Garland High School report
shows these figures.

| Family | Rate | All students | Economically disadvantaged |
| --- | --- | --- | --- |
| GRAD1 | CCMR | `CA0GP24R` (92.9%) | `CE0GP24R` (92.7%) |
| GRAD2 | AP/IB exam criterion | `CA0GY24R` (32.7%) | `CE0GY24R` (26.3%) |
| GRAD2 | Dual-credit criterion | `CA0GUA24R` (33.8%) | `CE0GUA24R` (36.3%) |

The CCMR report also shows asterisks for Austin ISD's American Indian rates for
all three measures. Those cells match `-1` in `DI0GP24R`, `DI0GY24R`, and
`DI0GUA24R`. The reports do not provide independent numerator or denominator
counts. Family periods and remaining field periods and units stay unknown. The
source reader preserves blanks and `-1` and applies no value policy or estimation.
The district rows carry no State or regional aggregate. The reviewed rates are
available through the explicit source profile and do not enter the default release.

The `state_tapr_college_readiness_components` source preserves one statewide
publisher row without a geographic key. It covers 2023–24 annual graduates
in the 2024–25 edition. The 121 CSV tokens and all 165 ordered dictionary
occurrences are retained. The public statewide CCMR report corroborates 55
displayed rates across five measures and 11 groups. The CSV supplies the
numerators and denominators. The catalog binds the all-students and
economically disadvantaged counts and rates with their units and period.
Dual-credit and OnRamps criteria can overlap. Use the published combined
criterion for the union. The statewide source supports raw preparation,
explicit eager reading, and a native singleton leaf. Use `-` as the key
argument in `--prepared-input` for the state source. The singleton is
identified by its exact edition and has no physical geographic key.

Replay the statewide evidence from a populated provenance store:

```sh
uv run python -m teadata.scripts.verify_tapr_state_grad2 --store provenance
uv run python -m teadata.scripts.verify_tapr_state_grad2 --family GRAD1 --store provenance
```

The statewide GRAD1 selection retains all 33 requested overall CCMR fields
for 2023–24 annual graduates and the complete publisher dictionary. All
11 displayed rates match the separate statewide report. TEA's full GRAD1
response ends with a SAS execution error. The comparison uses the separate
successful overall CCMR download.

Compare a district or campus query against the state with a pinned native
release containing the two exact GRAD2 editions:

```python
from teadata.dataset_metadata import EditionKey

def select_grad2(scope):
    dataset = f"{scope}_tapr_college_readiness_components"
    row, = (row for row in release.catalog.editions(dataset=dataset, include_metadata=False)
            if row["edition"] == "2025")
    return EditionKey.from_dict({name: value for name, value in row.items() if name != "edition_id"})

comparison = (engine >> ("district", "Austin ISD")) >> (
    "compare_statewide", release, select_grad2("district"), select_grad2("state"),
    "ccmr_annual_graduates_dual_credit_economically_disadvantaged_rate",
)
print(comparison.first().difference_pp)
```

Omit the metric or pass `"*"` to compare every common reviewed statistic in
the selected editions. The same operator accepts exact editions of the
following families when the release contains both scopes.

| Publisher family | Local scopes | Required dimensions |
| --- | --- | --- |
| TAPR 2025 GRAD2 | District, campus | None |
| TAPR 2025 GRAD1 overall CCMR | District, campus | None |
| TAPR 2025 COMP4 federal graduation rate | District, campus | None |
| TAPR 2025 STUD | District, campus | None |
| PEIMS Student Program and Special Populations | District, campus | None |
| PEIMS Student Enrollment by Ethnicity | District, campus | `ethnicity`, such as `"Hispanic/Latino"` |
| PEIMS Primary Disability | District | None |
| PEIMS Staff FTE and Salary | District | `staff_category` and `staff_role`, as published |
| PEIMS Teacher FTE and Course Enrollment | District | `course_code`, as published |

Pass dimensions as a sixth tuple element or the `dimensions` keyword.
The result includes local and statewide values, their raw tokens and statuses,
their unit and period, and local minus statewide. Percent comparisons also
expose `difference_pp`. Counts, currency and FTE retain their own units.
Course comparisons use the separate district and statewide reports for the
same course code and school year. Course enrollment counts memberships at
the PEIMS reporting date. Campus course comparison remains on geography hold.
The 2023 and 2024 Developmental Delay interpretation hold excludes those
observations from comparison.

GRAD2 exposes the reviewed AP/IB and dual-credit criteria rates for all
graduates and economically disadvantaged graduates. Each uses 2023–24
annual graduates in the selected geography.
The reviewed TEA rate policy preserves `-1` as suppressed. Blank tokens,
absent local rows, and uninterpreted tokens carry separate statuses and
yield no gap. TAPR is not applicable to private campuses. Resolve ambiguous
source revisions explicitly before comparing. These native leaves are opt-in.

The 2024–25 STUD state selection retains 228 columns outside class size and
241 ordered dictionary entries. The full-state selection returned a SAS error.
The independent statewide report corroborates 72 comparison fields covering
Membership, Enrollment, mobility and disciplinary placements. The latter two
carry 2023–24 periods, and a wildcard comparison retains each field's period.
Special education, gifted and talented, dyslexia, emergent bilingual and
Section 504 have separate Membership and Enrollment measures. Mobility uses
the publisher's campus-based calculation at each scope.
The `daep_count` and `daep_percent` selectors follow the bulk shorthand.
The glossary's disciplinary placements measure includes both disciplinary
alternative education and juvenile justice alternative education programs.

The state CSV and dictionary label `SPNTINDC/P` as Asian, while the independent
report places those values in American Indian Enrollment. Both fields retain
their original labels and codes and stay outside the comparison contract.
Attrition remains raw in the state edition while the glossary's cohort conflict
is open. The separate `SPNTASIC/P` pair matches published Asian Enrollment.
[TEA statewide Student Information](https://rptsvr1.tea.texas.gov/cgi/sas/broker?_debug=0&_program=perfrept.perfmast.sas&_service=marykay&ccyy=2025&id=S&lev=S&prgopt=reports%2Ftapr%2Fstudent.sas)

```sh
uv run python -m teadata.scripts.verify_tapr_state_grad2 --family STUD --store provenance
```

Prepare each family separately with the existing bounded native producer:

```sh
uv run --locked --offline --python 3.14.10 python -m teadata.scripts.build_tapr_profiles \
  --config teadata/teadata_sources.yaml --edition 2025 --scope district \
  --family college_career_military_readiness \
  --reference-store <canonical-provenance-store> \
  --input-dir <owned-named-input-view> --output-dir <new-output-directory>
```

Select either of the other literal family names from the table to prepare that
original instead. The named input files must match the configured basenames
and reviewed raw checksums; the output directory must not exist. The original
HTTP receipt is retained through the `csv_headers` reference, even when its
immutable acquisition dataset name predates this family declaration. Preparation
adds actual local observations and processing receipts without rewriting that
history or activating a default build.

Raw TAPR intake uses one logical-record scanner for both the eager
`read_tapr_table()` API and bounded source preparation. Quoted multiline
headers and values, original whitespace, UTF8 BOMs, CR/LF separators and
reviewed annotation ranges remain lossless. Batch results are provisional
until the reader reaches validated EOF; early close or any validation failure
does not expose a successful source summary.

`Config.tapr_source_requests()` validates the same source metadata without
opening raw files. A request's `open(budget=TaprReadBudget(...))` context
manager yields Arrow batches. Import the budget from `teadata.tapr` and set
`batch_bytes`, `record_bytes`, `metadata_bytes`, `key_bytes` and `batch_rows`
when an explicit intake envelope is needed. Admission counts original record
bytes, native string payload/offsets, retained header/annotation text and the
narrow global identity index. These limits bound the admitted working data;
they are not a promise about Python, Arrow or codec peak RSS. Fact values
remain exact strings with no masking or numeric interpretation.

The raw `prepare_profiles(..., budget=...)` path validates every selected
source before opening a writer or archive, then processes scopes sequentially
into private Parquet row groups bounded by admitted batches. It verifies exact
strings, schema and row groups against a further bounded source scan before
publishing the complete generation. It currently requires one physical table
per source and rejects a multi-table source before reading its raw rows;
there is no eager merge fallback. The declared 2024 source entries each have
one physical table. The legacy 2025 `campus_tapr_student_staff_profile` and
`district_tapr_student_staff_profile` source profiles each declare two physical
tables and are unsupported by this bounded producer. Select the separate
2025 `student_profile` and `staff_profile` families, as in the explicit commands
above, with a new output directory for each. Their existing source profiles
are independent contracts; this does not rewrite the historical combined
configuration or choose an eager join automatically.

Public eager `read_tapr_sources()` and `merge_tapr_tables()` remain available
for explicit multi-table reading and merging. That API is separate from the
bounded producer and is never its fallback. Prepared-leaf intake is separate
and rejects a raw-source budget rather than implying it applies there.

One-header 2024 inputs use their complete Name/Type/Length/Label dictionaries;
2025 inputs declare exact label and code header records. Any preamble, note,
blank gap or footer must have reviewed CSV-record coordinates and an exact
content hash in the source contract. Removed notes are retained as annotations.
Undeclared records, duplicate identities, changed dictionaries, header collisions
and conflicting shared fields fail. Shared-field merging is limited to the same
qualified TAPR dataset and edition; it does not choose between observations
from different datasets, editions or methodologies.

Use `--help` to inspect options without loading source files or contacting
download/geocoding services. Importing the download and geocoding modules does
not start processing. Explicit relative paths are resolved from the invoking
working directory. Download and geocoding commands make network requests when
run normally.

Historical refresh uses one explicit CLI and the same acquisition/provenance
store. It never updates Config or replaces originals. A public GET, for example:

```bash
uv run python -m teadata.scripts.refresh_tapr acquire \
  --url https://rptsvr1.tea.texas.gov/perfreport/tapr/2024/masking.html \
  --method GET --section reference --dataset tapr_2024_masking_reference --edition 2024 \
  --metadata masking-metadata.json --metadata-sha256 METADATA_SHA256 --no-redirects \
  --filename masking.html --store provenance

uv run python -m teadata.scripts.refresh_tapr replay \
  --record ORIGINAL_RECORD_ID --store provenance
```

`acquire` and `replay` are the only network commands. Both default to three
transport attempts, a 45-second socket timeout and a 32 MiB response limit.
Retries start the transfer from the beginning and repeat the same request.
The client backs off for timeouts, interrupted responses and temporary HTTP
errors. It honors `Retry-After` and defers if the wait exceeds 30 seconds.
Use `--timeout 90 --attempts 4` for a slow endpoint, or `--attempts 1`
for a single try. Set `--max-retry-wait` to change the wait limit.
The receipt records the attempt that completed the transfer. Publisher error
pages returned with HTTP 200 still require separate source qualification.

For POST, supply the observed content type
and repeat `--field NAME VALUE` in original source order, including repeated
names and empty values. The command records these controls and request limits.
POST controls use UTF-8; other charset declarations are rejected before HTTP.
It never invents form parameters, counts flags or private/authentication
headers. Legacy POST receipts missing request content type require an explicit
`--content-type application/x-www-form-urlencoded`. Replay retains predecessor
history through the same provenance reference graph, even when bytes change.
Reconstructed form URLs must exactly match the original requested URL. A legacy
GET without form metadata reuses its exact retained URL.

Every new `acquire` requires explicit SHA-pinned `DatasetMetadata` JSON. For
example, `masking-metadata.json` can contain
`{"title":"2024 TAPR masking reference","version":"2024","period":{"kind":"unknown"}}`.
Its version must exactly equal `--edition`. `--section reference` explicitly
labels a documentation observation; it does not select a configured dataset.
Data acquisitions use the default `--section data_sources` and additionally
require `--config CONFIG --config-sha256 CONFIG_SHA256 --year YEAR`. The dataset
must exist in that Config, resolve to exactly that year, and have the exact
edition. Prior-year fallback is rejected by this acquisition consumer; ordinary
Config selection semantics are unchanged. Config validates identity/year/version
only: its period, metrics, policies, references and source rows are never
inherited or loaded. The supplied observation metadata remains independent.

Metadata and Config inputs are each capped at 2 MiB; their captured bytes are
SHA-checked, parsed and retained without rereading the original file. Config
format and relative-base handling use the same canonical loader as `load_config`.
The input files become local-observation receipts linked through the acquired
record's metadata `provenance_ids`; this adds input lineage, not a replacement
download date. Supply `--reference-store STORE` when selected reference receipts
live elsewhere, or retain them in `--store`. Only explicitly claimed metadata
and reference closures are verified and copied before HTTP. Missing, changed,
wrong-URL or wrong-SHA documentation fails before the request. No reference is
fetched implicitly. Use `--no-redirects` when authorization covers only the
literal request URL. These are socket/response bounds, not a whole-batch deadline;
an operator's separately selected outer deadline still applies. `replay` retains
the original observation metadata and is unchanged; it does not apply aliases or
silently correct historical labels.

Inspect retained family menus and variable-selection forms before composing the
next explicit request:

```bash
uv run python -m teadata.scripts.refresh_tapr form \
  --source-store provenance --record FORM_ACQUISITION_ID \
  --encoding iso-8859-1 --store provenance --output form-observations.json
```

This offline inspection records source-order control and option attributes,
labels, literal method/action values, tag coordinates, and fieldset/optgroup
context. A shared context table uses parent indices; controls reference that
table instead of copying their ancestors. It preserves repeated control names
and empty values. Missing methods
stay unknown. A missing form end tag is reported as `closed: false`; ambiguous
nested forms, controls or duplicate attributes fail for source review. Display
labels normalize presentation whitespace; textarea content stays exact.
The typed `teadata.tapr_forms.extract_tapr_forms()` API provides the same snapshots.

The report resolves anchor URLs and literal `window.open` first arguments
against the retained acquisition's observed URL. It never executes JavaScript,
evaluates disabled/checked/select submission rules, infers selected variables,
submits a form, or downloads data. Computed event handlers remain literal
attributes for review. A local observation without a recorded HTTP URL keeps
relative targets unresolved. The original bytes and acquisition dates remain
authoritative; a new processing receipt records this inspection.
Inputs use the same explicit positive `--max-bytes` bound and SHA/length checks
as dictionary inspection, and existing output paths are never overwritten.
`--max-nodes` bounds the complete form projection (default 20,000 observed forms,
controls, options, labels, links and context nodes). `--max-output-bytes` bounds
streamed JSON staging for both form and dictionary inspection (default 8 MiB).
Oversized output is discarded before publishing a processing receipt or report;
larger reviewed inspections require explicit limits. Retained inputs must be
regular files, are opened without following a final symlink or blocking on a
FIFO, and must retain the same file identity throughout capture.

Inspect a retained TAPR HTML dictionary offline through the same command owner:

```bash
uv run python -m teadata.scripts.refresh_tapr dictionary \
  --source-store provenance --record DICTIONARY_ACQUISITION_ID \
  --encoding iso-8859-1 --store provenance --output dictionary-observations.json
```

This appends a processing receipt without changing Config or acquiring data.
`--max-bytes` is a positive retained-input limit (default 2 MiB); a separately
reviewed larger dictionary can use an explicit larger value. Advertised and
actual sizes are bounded before copying, and the capped parsed bytes must
match the selected acquisition's SHA and length.
The typed `teadata.tapr_dictionary.extract_tapr_dictionary()` API retains every
ordered occurrence, including duplicate codes and independently unknown type
or description components. It preserves an exact decoded HTML slice; archived
raw bytes remain the identity authority. Rendered text normalizes presentation
whitespace only. Anchors bind the raw SHA to the outer tooltip start tag using
one-based lines and zero-based decoded-character columns.
Supported `p`, `div` and `br` boundaries separate rendered text. Ambiguous
dictionary-relevant duplicate class attributes fail instead of hiding a code.
The optional `Length:` heading is retained separately as literal
`publisher_length`, including leading zeros; it is not folded into publisher
type, cast to a number, treated as a scientific unit, or enforced as a maximum
CSV-value length. Missing length stays unknown and leaves the legacy column
record shape unchanged. The retained 2024 REF tooltips use this fourth heading;
the existing three-heading and code-only forms remain supported.

The existing `dictionary_records` column contract retains these observations.
Only a complete occurrence can document a field; separate partial occurrences
are never combined into a definition. Conflicting known components fail review.
Length is supplemental metadata, not a requirement for a complete definition.
Known length conflicts fail through the same occurrence validation, and an
explicit source-field length must agree with a known observed length. Every
occurrence remains available in the detached header projection without filling
missing field metadata from another occurrence.
An optional `publisher_context` occurrence object preserves a literal section
`heading` and its `source_line`/`source_column` in the same raw dictionary SHA.
The typed parser exposes this as frozen `TaprPublisherContext`; `column_record()`
omits the key when the context is unknown. The heading anchor locates the section
cell start, precedes the tooltip, and uses the same line/column conventions.
Only a closed, single-cell `linecontent` row with positive `colspan` establishes
a section. Contiguous heading rows before data form one block whose first title
is primary; an auxiliary class label does not replace that title. Context is
table-scoped, and ambiguous transitions omit context or reject a previously
claimed known-context row. It never supplies periods, units, masking policies
or methodological equivalence. `header_records()` exposes a detached copy in
`dictionary_occurrences`; field mappings do not acquire another context authority.
Existing records without this key remain unchanged and accepted. New context
documents require the updated reader; older strict readers may reject them.
Context does not change the original raw-header fingerprint.
Source comparisons retain these observations in `source_structure`; their
`dictionary_definition_equal` flag compares literal field definition components,
not section context or methodological equivalence.
Dictionary-only codes do not become source columns or establish availability,
aliases, units or reference periods. Existing `header_records()` and
`header_rename_map()` remain the code/label/canonical-field mapping APIs.
The 2025 `student_profile` and `staff_profile` branches are declared independently
for both scopes through the same explicit `source` profile and preparation
command. Student retains 243/242 campus/district columns and the exact pinned
raw revision; Staff retains the fresh 106/118-column observations, including all
15/17 additional columns. The reviewed Staff proposal moved unchanged to
`schemas/tapr--2025--staff-profile-crosswalk.json`; its proposal path was removed.
The Student contract derives its code/label/canonical-field mapping from the
unchanged historical crosswalk and adds the actual ordered 2025 dictionary
observations. Neither branch inherits 2024 periods, policies or metric bindings:
table/field periods and scientific units remain unknown. Identity/history
attributes stay lossless without creating extra entity metadata keys.

Use `--edition 2025 --family student_profile` or `--family staff_profile` with
the existing producer. Raw paths remain portable and `--input-dir` selects a
local retained-input directory; provenance stores remain explicit local inputs.
These Config declarations do not qualify a real preparation run, new data
release, released defaults, app activation or schema3. The historical 2025
composite and its older Staff revision remain independent reproducible evidence,
without fallback, replacement or a current-source alias. An unavailable route
for 2025 PROF does not establish that the source never existed.

The selected raw, form, annual dictionary, glossary and masking references bind
genuine immutable acquisition receipts. Fresh HTTP dates remain known; earlier
local-download dates remain unknown. The original STUD logical-label typo is
preserved alongside its explicit correction processing receipt; it is not an
automatic alias or replacement acquisition. Config's acquisition-only fields do
not express a processing-input edge. Pass a reviewed dependency explicitly with
repeatable `--processing-input DATASET RECORD_ID` to the existing producer; each
dataset must be selected by that invocation's family/scopes. The producer verifies
that each root is a processing receipt with a complete closure before reading
source rows, then copies that closure and includes the typed ID in that dataset's
new processing inputs. It neither infers applicability nor converts processing
records into acquisitions. The same STUD correction can be explicitly selected
for each affected scope; its complete original graph remains intact. Real
producer qualification still requires separately reviewed selected inputs and
execution evidence; these declarations and synthetic tests do not supply it.

Preparation accepts canonical annual integer editions exactly registered in
Config, rather than a fixed year allowlist. The existing reader requires the
resolved year and metadata version to match exactly before opening source rows;
nearest-prior fallback remains prohibited. This does not declare or acquire real
2026 inputs or introduce a general opaque-edition model.

Compare two retained originals offline using their exact acquisition IDs and
independently reviewed crosswalks. Substitute the actual IDs and reviewed SHAs:

```bash
uv run python -m teadata.scripts.refresh_tapr compare \
  --scope district --store comparison-provenance --output comparison.json \
  --old-store original-provenance --old-record OLD_RECORD_ID \
  --old-schema contracts/old-student.json --old-schema-sha256 OLD_SCHEMA_SHA256 \
  --old-dataset district_tapr_student_profile --old-edition 2024 \
  --old-table student_profile \
  --new-store refreshed-provenance --new-record NEW_RECORD_ID \
  --new-schema contracts/new-student.json --new-schema-sha256 NEW_SCHEMA_SHA256 \
  --new-dataset district_tapr_student_profile --new-edition 2024 \
  --new-table student_profile
```

Original input drift against either reviewed contract blocks keyed comparison
and retains a diagnostic report and original bytes. Reviewed header/layout,
dictionary, period and historical-key changes are reported separately from
cell differences. Old-only/new-only fields are source-schema absence, distinct
from blank, masked, missing or not-applicable cells. A shared annual code or
similar name/value does not establish a rename or methodological equivalence.
Retained PROF and its unmatched 38/42 campus/district fields remain independent.

Raw tokens are preserved, and scientific statuses default to unbound. Optional
`--old-policy CODE POLICY_ID` / `--new-policy CODE POLICY_ID` select individual
publisher fields from a SHA-pinned `--policy-config` and
`--policy-config-sha256`. A caller selection does not establish applicability;
the exact policy and its reference are recorded. Status projection uses the
existing scalar authority once per distinct token, capped at 4096 per field,
then native gathering. Decimal-spelling equality uses native exact decimals,
without floats or rounding, inside a 38-integer/38-fraction-digit envelope.
Other spellings remain unclassified. Selected non-observed statuses are never
used in numeric comparison; unbound spelling equality is not measured-value
equality or arithmetic. Reports publish atomically to new paths and retain
their processing/closure identity. Key/token examples are bounded and disclose
when the complete set is not emitted. Changed-observation examples include the
canonical entity key, old/new raw tokens and statuses, in key order within each
publisher code; exact totals and originals remain available.

### Unreleased opt-in edition catalog reader

Schema 3 is an explicit contract/reader API. Default bundle writers still emit
schemas 1/2, and published 0.0.126 and app defaults are unchanged. The opt-in
source producer below writes a new local candidate; it does not activate a
released default or establish all-family/year coverage.

`BuildRecorder.leaf_contract(...)` derives interpretation and selected mapping
identities from the same Config/crosswalk authorities. A native writer supplies
metadata-free Arrow IPC schema bytes, ordered physical columns and **explicit
ordered key columns**, row count, input hashes, recipe and actual toolchain.
The contract excludes final file SHA, processing receipt and root/footer
metadata. After writing the `teadata.leaf` footer and recording the file through
`ProvenanceStore.record_processing`, whose parameters bind `leaf_contract_sha256`,
`recipe` and `toolchain`, register the exact `ArtifactRef` with
`record_edition(...)`; `catalog(schema_version=3)` is opt-in. The source revision
is derived from ordered upstream bytes, independently of edition text and leaf
transform identity. These are separate from observation dates and transport URLs.

Schema 3 stores each exact crosswalk once as an external `ColumnSchemaRef`
(`sha256`, positive `size_bytes`, adjacent filename or HTTP(S) `location`). Its
canonical original JSON bytes and genuine schema observation remain unchanged;
it is not a Parquet `ArtifactRef`. Each metadata resource is capped at 8 MiB.
The unpublished inline schema 3 prototype is retired: current schema 3 rejects
inline or mixed representations. Archived prototypes remain immutable evidence;
re-emit a new candidate from accepted bounded preparations when needed. Published
schemas 1/2 keep their original inline representation and readers.

On a loaded schema 3 `DatasetCatalog`, `editions(...)` and source metadata are
root-only projections. `variables(...)`, crosswalk headers, field metadata and
`resolve_artifacts([(edition_key, field), ...])` lazily resolve the selected
crosswalk and validate its complete table/mapping/semantics before returning
field descriptors or physical projections. Dependencies are validated before
any selected leaf request, including direct `artifact(artifact_id)` calls.
Unfiltered `variables()` explicitly discovers all registered metadata resources;
none of these operations opens data leaves or imports Arrow. A standalone catalog
needs an explicit resource resolver for selected metadata; it never consults
package configuration implicitly. Verified resource bytes share the existing
release download/cache transport, with parsed metadata held only as transient
state. Pickles carry neither cached acceptance nor an implicit resolver. An explicit `period=DataPeriod(...)`
requires exact known coverage; unknown fields remain discoverable without
invented units/policies. Scalar interpretation is not added by this native
reader. `metadata`, `policy`, `field_metadata` and crosswalk projections require
an exact `EditionKey` when a unique match cannot be established. Schema 2 does
not invent a complete index by consulting current configuration.

`DataRelease(uri, expected_sha256=pin, offline=True)` verifies the selected root
in the existing URI cache slot and forbids root, metadata-resource and leaf downloads. A missing or
corrupt pinned root fails even with leaves cached. `artifact(artifact_id)` and
`analytical_dataset(artifact_id)` open only selected schema 3 leaves, validating
their own contract, native types/nullability, row count and grain. Legacy role
and global-footer checks remain strict for schemas 1/2. The root limit remains
4 MiB; the index covers its registered editions, with no claim that arbitrary
future history fits this in-memory envelope. Externalization alone does not
prove a candidate fits: publication admits the complete final canonical root,
including every field binding, native contract and provenance record. `verify()`
explicitly validates all metadata and opens every registered leaf; publication
performs that complete local offline check before exposing its root.

### Unreleased edition metadata resources (schema 4)

For cheap discovery, use the same iterator with metadata explicitly excluded:

```python
for edition in release.catalog.editions(include_metadata=False):
    print(edition["dataset"], edition["edition"], edition["edition_id"])
```

Schema 4 is an explicit, unpublished representation. Its small root contains
exact `EditionKey`/`EditionMetadataRef` entries and artifact-ID-to-edition routes.
Each child contains that edition's complete metadata, native contracts, qualified
field bindings, policies and original receipt closure. An `EditionMetadataRef`
uses the existing SHA/size/location contract and a 4 MiB per-child limit;
crosswalks retain their independent 8 MiB limit. All resources use the same
verified release transport/cache. The complete canonical root remains capped at
4 MiB. These limits do not establish a memory or arbitrary-history capacity claim.

The default `editions()` result shape is unchanged and includes metadata. It is
a generator: creating it performs no child I/O; iteration loads the selected
children. `metadata()`, `datasets`, unfiltered `variables()` and complete
`column_schema_refs()` discovery may read all relevant child metadata, never
native leaves. Broad `variables()` discovery uses edition-index order, then the
original field order within each edition; selected field/header/projection order
is preserved. Internal index discovery uses `include_metadata=False`.

Selected field/header and direct artifact resolution close foreign edition
dependencies before running the same catalog validator and selected crosswalk
checks. Missing qualified dependencies, cycles, foreign metadata and unsupported
period requests fail before leaf opening. Policies remain scoped to their exact
edition, even when historical policy names repeat. There are no policy aliases,
implicit current-Config lookups or new scalar interpretation APIs.

`resolve_artifacts()` shares dependency validation within each call and reuses
each selected edition's validated view. One ordered walk preserves the original
request/dependency discovery order, native key columns and duplicate elimination;
membership sets avoid repeated scans of growing projections. It uses the already
validated artifact IDs without rehashing full native contracts per field. This
state is local to the call. Selected child/schema resources are still rechecked
before returning, including when a lazy selection iterable changes a file during
resolution. Catalog bytes, build IDs, pickle state and native data files are
unchanged. This reduces metadata work; it establishes no measured latency, RSS
or runtime resource qualification.

Explicit migration is `schema3_catalog.externalize_editions()`; alternatively,
`DatasetCatalog.create_editions(..., schema_version=4)` and
`BuildRecorder.catalog(schema_version=4)` opt in after ordinary v3 validation.
Migration preserves native artifact IDs/contracts/locations, fields, source
revision, original schema bytes and every receipt. It rejects unbound provenance
or missing genuine schema observations; callers must review any historical
reference additions explicitly before migration. It creates only in-memory
metadata resources. `publish_release(..., schema_version=4)` preflights all local
destinations and existing sidecars, then performs complete metadata and native
leaf verification before exposing a new root. Cached parsed children and leaf
acceptance are transient; unpickling a release reopens its pinned root and
reverifies selected resources. A standalone catalog pickle needs explicit
resolvers again. Published schemas 1/2, their default writers and snapshot state
remain unchanged. Explicit schema 3 remains available until its actual consumers
have qualified an explicit migration; no representation is inferred or retried.

### Unreleased raw-wide source selection

The same `build_tapr_profiles` owner can consume one or more explicitly selected
accepted source preparations for one registered annual edition. Each selected
Config source profile must declare one complete reviewed physical table and its
scope. Logical family uses the existing qualified TAPR source identity; it need
not equal the physical table name (the retained composite's table is `profile`).
Supply exact processing receipts and explicit physical keys. For example, after
the ordinary source-preparation commands, select both GRAD scopes:

```bash
uv run python -m teadata.scripts.build_tapr_profiles \
  --edition 2024 --config reviewed-sources.yaml --reference-store selected-provenance \
  --prepared-input campus_tapr_graduation_profile CAMPUS_GRAD_PROCESSING_ID campus_number \
  --prepared-input district_tapr_graduation_profile DISTRICT_GRAD_PROCESSING_ID district_number \
  --output-dir new-2024-raw-wide-candidate
```

All selected typed closures, exact Config editions, reviewed source/header
contracts, upstream raw identities and declared processing dependencies validate
before native input opening. Explicit keys must match the existing TAPR
publisher-code contract; native types, complete columns, row count and key
uniqueness are checked before activation. Prepared mode requires the current
bounded source producer's exact code identity and complete `source_read_budget`;
legacy or missing admission requires source re-preparation. Every source row
group remains one admitted, non-combined Arrow batch and one output row group.
The producer never concatenates groups into a fixed 4096-row read. It checks
native batch bytes, rows and retained canonical keys against that budget; paired
readback preserves the same exact group boundaries. An unexpected chunked group
is rejected rather than split or combined. These are payload/index admission
limits, not guarantees about Parquet codecs, native allocator overhead or RSS.
Preparation row claims must first match the finite reviewed
data-record interval after its explicit annotation exclusions; missing or empty
layouts cannot establish retained rows. The producer verifies exact raw-string
readback, registers every selected source field, validates its root through `publish_release(...,
schema_version=3)`, and renames only a complete fresh staging directory. Existing
outputs are never replaced. `--processing-input DATASET RECORD_ID` remains
available for additional explicitly reviewed typed dependencies; inherited
preparation dependencies remain in the closure without relabeling acquisitions.

Prepared input datasets own the selection: the producer neither enumerates all
configured datasets nor implicitly adds another scope/family. A single source
is sufficient; STUD/STAF use the same path after accepted bounded source
preparation. No
declared table or field is silently dropped; ambiguous/multiple physical-table
declarations fail before native I/O. CLI `--family` and `--scope` belong to CSV
intake and cannot be combined with `--prepared-input`; their defaults do not
filter prepared datasets. Programmatic `prepared_inputs` likewise owns selection
instead of the legacy `family`, `scopes` and `input_dir` parameters. Existing
exact edition, scientific methodology and closure gates still apply: this does
not qualify the retained PROF policy's unreviewed masking reference or activate
any source/default.

Use one bounded edition/cohort root at a time, with 2024 before 2025 and later
reviewed annual editions. This is a raw-wide selection path: edition text is
not a measurement period, and unknown units, policies and meanings remain
unknown. It does not implement qualified batch measurement normalization or
longitudinal comparability. Select an exact edition/field using the catalog,
then use the returned `ArtifactRef` with `DataRelease.analytical_dataset` and
ordinary Arrow `scanner(columns=..., filter=..., batch_size=...)`. Root metadata
discovery opens no leaf; selected reads use the existing per-SHA verified cache
and pinned-root offline behavior. No entity engine, analytical EAV conversion,
source reacquisition or all-history manifest is introduced.

Relative `data/...` paths in the source contract also honor `TEADATA_DATA_DIR`. Other relative paths remain relative to the configuration file. `uv run python -m teadata.load_data` reads the selected contracts, stages a complete release, and activates `teadata/.cache/release.json` after validation. There is no second cache mirror or developer-checkout copy.

The Excel styling example takes a workbook path rather than referring to an
unrelated local project:

```bash
uv run python examples/restyle_existing_excel.py report.xlsx --sheet-name Sheet1
```

Styling requires Excel and the optional `xlwings` package. The campus profile
example's county-boundary default follows `TEADATA_DATA_DIR`; callers can still
pass an explicit `counties_path`.

### Tested-grade STAAR sources

The source profiles `{state,region,district,campus}_tapr_staar_tested_grade_{3..8}`
select exact 2026 tested-grade STAAR extracts for the accountability subset.
Each grade is a separate family and has one publisher table per geography.
The state profile reads TEA's statewide row directly. The six grades share
the same sets of 20 education service center, 1,202 district and 9,105 campus
IDs in the retained exports.
These counts describe the publisher's extracts, which can contain blank or
withheld results for a particular grade or group.

The 2024–25 source profiles select edition 2025 and also include
`region_tapr_staar_tested_grade_{3..8}`. The six grades retain one Texas row,
20 education service center rows, 1,208 district rows and 9,084 campus rows
per grade. Use `--edition 2025` for these files. In both editions, regional
keys are TEA's literal two-character codes `01` through `20`.
Each 2025 geography retains its own dictionary, a named TEA report and the
[2024–25 TAPR masking methodology](https://rptsvr1.tea.texas.gov/perfreport/tapr/2025/masking.html).
The raw reader preserves those tokens without interpreting them.

Prepare a grade using its exact retained CSVs and reference receipts.

```bash
uv run python -m teadata.scripts.build_tapr_profiles \
  --edition 2026 --family staar_tested_grade_3 \
  --scope state --scope region --scope district --scope campus --input-dir raw/ \
  --reference-store provenance --output-dir prepared/staar-2026-grade-3
```

Use the resulting processing receipts with the raw-wide selection above to
build a separate bundle for a grade. Select the bundle and geography explicitly.
The source retains publisher column codes, dictionary definitions and raw
strings, including blanks, `-1` and `-3`. These profiles define no masking policy,
normalized metrics, or automatic statewide comparison. They are outside the
default snapshot and entity enrichment.

See [TEA's TPRS glossary](https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/comprehensive-tprs-glossary.pdf#page=2)
for the tested-grade population and how it differs from the all-tested reports.

The same source profile and preparation command support these additional exact
2026 families. Substitute the family name in `--family`. Each family retains its
own publisher file and dictionary for Texas, education service centers,
districts and campuses.

| Family | Publisher file | Fact columns per scope |
| --- | --- | ---: |
| `staar_grades_3_to_8` | `STAAR_GR38` | 300 |
| `staar_all_grades` | `STAAR_ALL` | 945 |
| `staar_end_of_course` | `STAAR_EOC` | 1,134 |
| `staar_school_progress` | `STAAR_SP` | 2,304 |
| `staar_participation` | `PART` | 2,106 |
| `staar_bilingual_esl_performance` | `BIL1` | 527 |
| `staar_bilingual_esl_growth` | `BIL2` | 270 |

`STAAR_ALL` is the all-grades publisher file in the accountability reporting
product. For the separate all-tested population, see TEA's all-tested reports.
The bilingual and ESL dictionaries leave 72 performance fields and 36 growth
fields per scope undocumented. The crosswalk retains those fields and their
raw values with explicit dictionary status. Some publisher codes recur across
families. Keep those source identities when selecting a bundle for analysis.
For the participation and school-progress files, prepare and publish one
geography per candidate to stay within the schema 3 manifest size limit.
The school-progress crosswalk pins both records of TEA's wrapped label header
as original preamble bytes and uses the third record as the code header.
It retains the original label text and links the smaller publisher exports
that independently reproduced every fact cell.

### Annual and longitudinal graduation-program sources

The explicit 2024 `source` profiles retain the publisher's Annual and
Longitudinal RHSP/DAP/FHSP-E/FHSP-DLA/Texas First-DLA responses as raw strings.
Their dataset names use the scope prefix followed by
`tapr_annual_rhsp_dap_fhsp_e_fhsp_dla_texas_first_dla` or
`tapr_longitudinal_rhsp_dap_fhsp_e_fhsp_dla_texas_first_dla`.

| Publisher group | Campus columns | District columns | Dictionary evidence per scope |
| --- | ---: | ---: | --- |
| Annual | 200 | 198 | 210 ordered occurrences; 14 returned fields lack definitions |
| Longitudinal | 158 | 156 | 204 ordered occurrences; 22 code-only entries have no returned column |

The responses contain 9,082 campus rows and 1,207 district rows per group.
Identity columns occur once in each source table. Exact raw labels, codes,
canonical names and dictionary contexts remain available through the existing
header APIs; dictionary-only entries remain in `dictionary_records` without
fabricated value bindings. Absence in this response does not establish
noncollection or absence in another edition.

Publisher edition 2024 and the literal 2023 labels remain distinct. Reference
periods and units are unknown, field-specific masking applicability is
unreviewed, and no value policy, metric definitions or estimation is applied.
These opt-ins do not activate a default reader, release or app enrichment.

### Five- and six-year extended longitudinal graduation rates

The 2024 `source` opt-ins use the exact suffixes
`tapr_5_year_extended_longitudinal_rate` and
`tapr_6_year_extended_longitudinal_rate`, each with `campus_` or `district_`.
Each selected response retains 200 campus columns or 198 district columns,
including its identity columns once, and 9,082 campus or 1,207 district rows.
All 224 ordered dictionary occurrences per scope remain available; 28 have no
column in that exact acquired response and receive no fabricated value binding.

The raw labels and dictionary descriptions state 2022 for the five-year source
and 2021 for the six-year source, while the selection-form federal group labels
state 2023. These publisher statements remain unresolved and do not establish a
measurement period. Only colliding federal canonical names include the observed
federal dictionary heading; the original labels, codes and contexts are retained.
Publisher Length, units and reference periods remain unknown, field-specific
masking applicability is unreviewed, and no value policy or estimation is applied.
Source absence does not establish noncollection. These profiles preserve raw
strings without activating a default reader, release or app enrichment.

Checksum-verified release SQLite artifacts are read from the checkpointed main file: default map, entity and boundary lookups ignore unverified WAL sidecars. Explicit paths and caller-owned `EntityStore`/`BoundaryStore` instances remain ordinary read-only SQLite readers and can observe committed WAL state. `DataRelease` retains checksum/catalog validation and stat-cache ownership; native Parquet/edition resources keep their existing validation path.
