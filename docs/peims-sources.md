# PEIMS student reports

The library registers two 2025–26 PEIMS report families with independent state,
district and campus scopes. Each scope has its own exact source acquisition,
publisher header crosswalk, and native Parquet leaf. These are October 2025
PEIMS Enrollment observations. TPRS Membership is a separate population.

| Dataset key | Publisher release | State rows | District rows | Campus rows |
| --- | --- | ---: | ---: | ---: |
| `peims_student_enrollment_by_ethnicity` | [Student Enrollment Reports, ethnicity grouping](https://rptsvr1.tea.texas.gov/adhocrpt/adste.html), March 6, 2026 | 7 | 6,873 | 48,971 |
| `peims_student_programs_and_special_populations` | [Student Program and Special Populations Reports](https://rptsvr1.tea.texas.gov/adhocrpt/adspr.html), March 13, 2026 | 1 | 1,202 | 9,113 |

The [PEIMS Standard Reports listing](https://tea.texas.gov/data-reports/student-data/standard-reports/peims-standard-reports)
gives publication dates and leaves finality unspecified. The population and
masking documents are in `Config.reader_references` and in the built catalog.
The checked-in `source` profile lists exact scope receipt IDs. Each receipt
links to the earlier publisher observation and preserves its original request
method, controls, filename, and download time.

The ethnicity report contains one row per published ethnicity and geography.
For a direct total use `total_enrollment_count` from the matching program table.
Program counts can overlap. The native leaves retain literal six-digit district
and nine-digit campus IDs, including leading zeros. Count fields carry their
publisher token, normalized value, status, and source record locator. The
per-report policy preserves true zero and tokens such as `<2,790` and `-999`.
Suppression produces a null value with status `suppressed`.

## Build and read a candidate

Materialization is offline. `TEADATA_PROVENANCE_DIR` must contain the Config's
complete acquisition closure and exact source bytes. The output must be a new
directory. This workflow leaves the packaged source-only discovery profiles
available and builds a separate bundle for explicit catalog reads.

```sh
PEIMS_CONFIG=teadata/teadata_sources.yaml
uv run --locked --offline python -m teadata.scripts.refresh_peims \
  --store "$TEADATA_PROVENANCE_DIR" \
  --config "$PEIMS_CONFIG" \
  --config-sha256 "$(shasum -a 256 "$PEIMS_CONFIG" | cut -d ' ' -f1)" \
  --source peims_student_enrollment_by_ethnicity 2026 \
  --source peims_student_programs_and_special_populations 2026 \
  --output "$PEIMS_CANDIDATE_DIR"
```

```python
from teadata.dataset_metadata import EditionKey
from teadata.release import DataRelease

release = DataRelease("path/to/candidate/release.json", offline=True)
dataset = "peims_student_programs_and_special_populations"
entry, = (row for row in release.catalog.editions(dataset=dataset)
          if row["edition"] == "2026" and row["scope"] == "state")
edition = EditionKey(entry["dataset"], entry["edition"], entry["revision_sha256"],
                     entry["scope"], entry["table"], entry["grain"])
selection, = release.catalog.resolve_artifacts([(edition, "gifted_and_talented_program_count")])
table = release.analytical_dataset(selection["artifact"].identity).to_table(
    columns=list(selection["columns"]))
acquisitions = release.catalog.acquisitions(dataset, edition=edition)
```

The reader verifies the entire bounded source file and its reviewed preamble,
header, footer, edition, scope and row keys. Arrow reads a bounded CSV body and
converts each count column in native code. Only distinct special tokens require
Python policy interpretation. Parquet row groups are limited to 4,096 rows and
8 MiB of Arrow data. Each original is limited to 8 MiB and 60,000 physical
records. The producer checks each written row group against its projection.

## Publisher comparisons

The six public files were downloaded again on November 4, 2026 and matched the
earlier reviewed bytes. Independent CSV decoding matched all 66,167 rows and
797,799 original cells, including 48,465 suppressed count values. Independent
checks of seven TEA HTML report selections matched 78 values across both
families. Three examples establish the exact population, a leading-zero key,
and the difference between a true zero and suppression.

| Publisher witness | 2025–26 PEIMS observation |
| --- | --- |
| [Austin ISD ethnicity report](../provenance/records/03288b78e7ef6ed8faf33cdc710669d07e05611467ad17bc6b5f09ab54809293.json) | Asian 3,446 and American Indian or Alaska Native 123 |
| [Del Valle ISD program report](../provenance/records/d57f7ea3daa2d3df48a9acdbc1200e10857acbb525feeeab51eeafca234a70c7.json) and [Texas program report](../provenance/records/8678c92b4c16e804a20d2a26d4dbd5d9a7ec78204186148540a6d343be9243fc.json) | G/T 557 of 11,811 Enrollment in Del Valle and 489,027 of 5,467,642 Enrollment statewide |
| [A+ Academy EL ethnicity report](../provenance/records/ec75b056b4c3538f0ed415427e3d71bd0d29ff337f582efebf6d3cc0a79ad07c.json) and [program report](../provenance/records/1007442d96b05eb9839f36a7b23fcb7b4bfa055c27a5eebaddf76efe2db0bcf3.json) | Campus `057829001`, district `057829`, Asian count `<10`, and an observed military-connected count of zero |
