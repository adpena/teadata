# PEIMS student reports

The library registers two PEIMS report families for 2022–23 through 2025–26,
with independent state, district and campus scopes. Each scope has its own
exact source acquisition, publisher header crosswalk, and native Parquet leaf.
These are October PEIMS Enrollment observations. TPRS Membership is a separate
population.

| Dataset key | Publisher release | State rows | District rows | Campus rows |
| --- | --- | ---: | ---: | ---: |
| `peims_student_enrollment_by_ethnicity` | [Student Enrollment Reports, ethnicity grouping](https://rptsvr1.tea.texas.gov/adhocrpt/adste.html), March 6, 2026 | 7 | 6,873 | 48,971 |
| `peims_student_programs_and_special_populations` | [Student Program and Special Populations Reports](https://rptsvr1.tea.texas.gov/adhocrpt/adspr.html), March 13, 2026 | 1 | 1,202 | 9,113 |

The [PEIMS Standard Reports listing](https://tea.texas.gov/data-reports/student-data/standard-reports/peims-standard-reports)
gives the most recent publication dates and leaves finality unspecified. The
three earlier extracts retain their acquisition timestamps and exact edition.
The program reports for 2022–23 and 2023–24 label the emergent bilingual count
`EMERGENT BILINGUAL/ENGLISH LEARNER`. That literal heading remains visible in
the crosswalk for each year. The population and
masking documents are in `Config.reader_references` and in the built catalog.
The checked-in `source` profile lists exact scope receipt IDs. Each receipt
preserves its request method, controls, filename, and download time. A replay
receipt links to the earlier observation when available.

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

The 18 historical leaves cover 197,674 rows across the two families and
three scopes. The offline qualification gate compares their typed and raw
cells with the retained bulk files and checks 189 published counts against
18 separately selected Texas, Austin ISD and A+ Academy EL reports. Historical
campus ethnicity reports omit the overall count. Use the program report for
that count and retain suppressed categories without summing them.

Use the historical manifest pinned in
[`native_peims_historical.json`](../teadata/reference_cases/native_peims_historical.json)
with the [native release gate](data-releases.md#qualifying-reviewed-native-bundles).
The manifest's adjacent `provenance` directory supplies raw parity evidence.
The separate source store passed with `--source-store` supplies the individual
publisher reports.

## Teacher FTE and course enrollment source review

TEA's [standard report listing](https://tea.texas.gov/data-reports/student-data/standard-reports/peims-standard-reports)
records March 16, 2026 for the latest edition. The 2025–26 report footer also
records the July 20 addition of Engineering courses. The September 16
[report criteria](https://rptsvr1.tea.texas.gov/adhocrpt/Standard_Reports/About/About_TeacherFTE_Enroll.html)
document a separate September recreation of the 2012–13 through 2025–26
reports to eliminate duplicate student counts associated with assistant or
support teachers. Pin the dated acquisition when comparing extracts from
before and after that revision. TEA does not specify a finality status here.

Each statewide edition has one observation per published course code, with
the subject and intended grade range attached. The student count measures
course membership at the PEIMS reporting date. Course membership differs
from full-year completion and does not measure eligibility or exam taking.
A student can appear in multiple courses. Teacher FTE includes charter partner
teachers starting in 2023–24, so comparisons with earlier years need that
definition change. The publisher uses `-999` in CSV and `N/A` in HTML for
masked cells, including some larger values masked to prevent imputation.

The November 4, 2026 acquisition retains the
[selector](../provenance/records/67a39bd274e492426111703a32278ad84a63cb5db037963779e18d5147da80a1.json)
and [criteria](../provenance/records/3f0a9a11c39e00dbada1e7017aba382afe7b859bd142a00c0d3b2125752f6b10.json)
with the exact request controls, hashes and retrieval times. The following
offline comparison requires the archived objects under the provenance store's
`objects/` directory. To replay from another checkout, use the archive or
reissue each receipt's recorded form request and verify the returned byte hash
and size before reusing the receipt. Changed bytes need a new acquisition
record. The command compares 5,658 statewide rows and 28,290 measure cells,
including 306 masked cells, across two presentations from TEA. It also checks
the reviewed masking, grade and revision notes in each edition's footer.

```sh
uv run --locked --offline python -m teadata.scripts.verify_peims_courses --store provenance \
  --pair 2023 4619f15fd40e8a6ae875d9a1c548542a46a7a2f550152f825b5861b22b273164 0b982beddda26f2d4515264bca782d4bff3c408ed246c6cfbff8c7fb894c189e \
  --pair 2024 0f4645a4f6919b48ed9653ad6ae8fa07abb500cb74891edc0fe45faa49b0f683 1ff4dbd6bcd81695d36dbb129b35d83b19571f4f22d7c1952af53633c9f21e1a \
  --pair 2025 dbe04d144f99c9d54f6f98b01b89a85b2b04f2ac46c9d58b5621d2d9f6ad3a2c 6e3333bc9663abf5d33685838a6ecc77207c1b7159c399b67ca2af60250e3669 \
  --pair 2026 b83eb582021312eaf322d97e0ea081085258d9f89e26e11cbdd9c2efa8b82cc1 cea5fd8baa3c740914856de275dc29a33508b3dafbcf51d9bf4b7370b84cc99c
```

An [individual campus extract](../provenance/records/50bdbb47f990375b35b64c5fa2e92c92edff30d4c267a5388b814854a78c97b6.json)
returned 59 course rows for A+ Academy Elementary (`057829001`) in 2023–24.
The [selection receipt](../provenance/records/a5d05dd3d90060f054d41962f18d8df08e73ff37ee0f1477469102597256632e.json)
records how TEA resolved that campus number. This establishes that a historical
campus report can be retrieved. Campus remains outside the native candidate.

### Build and verify a course candidate

The source profile admits separate state and district leaves. It preserves the
publisher course code, leading-zero district number, raw measure tokens and
physical source row. Head counts and course membership use integers, and FTE
uses fixed-precision decimals. The reviewed `-999` course-membership token has
a null value and `suppressed` status. Recorded zeros remain observed.
TEA defines the measures in its criteria page. The crosswalk's native storage
types reflect the reviewed CSV tokens because TEA does not publish formal
types for this export.

| School year | State course rows | District course rows | District IDs |
| --- | ---: | ---: | ---: |
| 2022–23 | 1,386 | — | — |
| 2023–24 | 1,397 | [201,999](../provenance/records/93c96e9c71b2201e09ba1abde6b0c8f3fb1da2c4befa71d662c1766ff2dda0d2.json) | 1,209 |
| 2024–25 | 1,422 | [204,303](../provenance/records/7963ec85cdfdce4e02849ec53520caed1df31588fd3c0c2b26161cc061c5773c.json) | 1,207 |
| 2025–26 | 1,453 | [207,751](../provenance/records/c54740974d39fdbc3c461120aba68db87b3af07fa3615eb91c868bd2d1c0ae81.json) | 1,203 |

`TEADATA_PROVENANCE_DIR` must contain the selected acquisition and reference
closures. Build into a new directory and pin its manifest before verification.

```sh
PEIMS_CONFIG=teadata/teadata_sources.yaml
uv run --locked --offline python -m teadata.scripts.refresh_peims_courses \
  --store "$TEADATA_PROVENANCE_DIR" \
  --config "$PEIMS_CONFIG" \
  --config-sha256 "$(shasum -a 256 "$PEIMS_CONFIG" | cut -d ' ' -f1)" \
  --edition 2023 state --edition 2024 state --edition 2024 district \
  --edition 2025 state --edition 2025 district --edition 2026 state --edition 2026 district \
  --output "$PEIMS_CANDIDATE_DIR"

scripts/check --course-controls --native-release "$PEIMS_CANDIDATE_DIR/release.json" \
  --native-pin "$PEIMS_MANIFEST_SHA256" --source-store "$TEADATA_PROVENANCE_DIR"
```

The course check independently decodes the retained CSV and compares all raw
fields, physical row locators, typed measures and statuses with the candidate.
It also compares the four state leaves with separately requested TEA HTML
reports and checks individual district reports for Hays CISD across all three
district years and A+ Academy in 2025–26. The [publisher witness inventory](../teadata/reference_cases/native_peims_courses.json)
identifies the exact HTML receipts. District publisher agreement is scoped to
those selections. These checks support explicit catalog reads from the
unpublished candidate. Dataset activation and release publication are separate.
