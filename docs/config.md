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
| `provenance_ids` | SHA-256 identifiers of immutable acquisition/processing receipts |
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
  --input statewide.csv --output-dir reports/
uv run python -m teadata.scripts.build_campus_peims_financials_statewide_data \
  --input-dir reports/ --output output/statewide.csv
uv run python -m teadata.scripts.geocode_tepsac_data \
  --input raw/schools.xlsx --output output/geocoded.csv
uv run python -m teadata.scripts.flatten_districts_geojson \
  --input districts.geojson --output districts_flat.geojson
uv run python -m teadata.scripts.build_tapr_profiles \
  --edition 2025 --input-dir raw/ --reference-store provenance --output-dir prepared/tapr-2025
```

The TAPR command prepares every declared publisher column for both scopes into
a **new** directory, validating all inputs before staging any output. Its
Parquet files retain raw strings, source crosswalks, annotation records and
immutable acquisition/processing receipts. Existing directories and retained
reference workbooks are never replaced. These outputs are source preparations,
not an activated release or typed scientific measurements. The old year-specific
builder and the 75-column district renamer are retired; 2024 and 2025 use this
same explicit preparation path.

`--reference-store` supplies the canonical acquisition receipts and exact bytes
for established reference links. The producer retains only their selected
immutable closure, preserving record IDs and observation dates. A missing
claimed receipt or artifact fails preparation. Newly observed crosswalks bind
their own receipts; unreviewed link-only documents remain explicitly not
acquired. Portable raw-source locators remain unchanged in public reader
metadata and resolve against the configured data directory only when selected.

Coverage refers to the declared inputs: 2024 `PROF` contains 407 campus and 420
district columns, including 34 lagged annual-graduate fields in each scope.
The current 2025 inputs are `STUD` and `STAF` only. They contain no graduate
fields; the separate publisher `GRAD` Graduation Profile needs its own audited
input/header contract before claiming equivalent annual-profile coverage.

The `source` reader profile exposes complete header contracts without reading
source rows, for example
`cfg.header_records("district_tapr_student_staff_profile", edition="2024", reader_profile="source")`.
`cfg.load_df(..., reader_profile="source")` explicitly reads the full raw edition;
the default reader continues to use its exact retained prepared workbook.
Raw source preparation cannot run inside a retained-format release build.
Source identities and historical names are lossless period-specific attributes,
not additional metric keys or automatic `.meta` hydration. No current geometry
roster limits source coverage. Unknown field units and reference periods remain
unknown; the existing qualified metric bindings retain their lagged periods.

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

Relative `data/...` paths in the source contract also honor `TEADATA_DATA_DIR`. Other relative paths remain relative to the configuration file. `uv run python -m teadata.load_data` reads the selected contracts, stages a complete release, and activates `teadata/.cache/release.json` after validation. There is no second cache mirror or developer-checkout copy.

The Excel styling example takes a workbook path rather than referring to an
unrelated local project:

```bash
uv run python examples/restyle_existing_excel.py report.xlsx --sheet-name Sheet1
```

Styling requires Excel and the optional `xlwings` package. The campus profile
example's county-boundary default follows `TEADATA_DATA_DIR`; callers can still
pass an explicit `counties_path`.
