# Entity enrichment sources and years

This table describes the published [v0.0.128 entity release](https://github.com/adpena/teadata/releases/tag/v0.0.128), build ID `17061ef34da36b4c968e0d162cfeb2ba0dfddd64fe4fd8067a785b3dd2bca685`. It includes all 13 recorded inputs, including roster, boundary, and reference inputs used alongside enrichment. Check the manifest for the release you actually load. Its catalog records the requested and resolved edition, source period, and provenance for each input.

| Source ID | Use in the entity build | Period in v0.0.128 | Editions configured for the entity pipeline |
| --- | --- | --- | --- |
| `accountability` | District and campus population and accountability ratings | Calendar 2025 | 2025 |
| `district_tapr_student_staff_profile` | District student and staff fields | School year 2024–25 | 2024, 2025 |
| `campus_tapr_student_staff_profile` | Campus student and staff fields | School year 2024–25 | 2024, 2025 |
| `campus_tapr_historical_enrollment` | Campus historical enrollment | School year 2014–15 | 2015 |
| `campus_transfer_reports` | Campus transfer edges | School year 2024–25 | 2025 |
| `campus_peims_financials` | Campus financial actuals | Fiscal year 2023–24 | 2024 |
| `campus_planned_closures` | Campus closure overlay | October 3, 2025 snapshot | 2025 |
| `charter_reference` | Charter identity and network reference | September 30, 2025 snapshot | 2025 |
| `all_districts_reference` | District reference | Reference period unknown | 2025 |
| `all_districts_reference2` | Additional district reference | Reference period unknown | 2025 |
| `districts` | District geometry | School year 2025–26 | 2025, 2026 |
| `campuses` | Campus geometry | School year 2024–25 | 2025 |
| `tefa` | Private school finder listings | School year 2026–27 | 2025, 2027 |

A configured edition means the [source configuration](../teadata/teadata_sources.yaml) defines a reader for that version. Loading this entity release supplies the periods in the third column. The TAPR profile editions `2024` and `2025` refer to school years 2023–24 and 2024–25. Some individual fields measure a different period than the report. Use field metadata when the measurement year matters. Fiscal years, calendar years, and dated reference snapshots keep their own period types.

The release verifier recognizes a separate 2026 candidate with TEA's exact
preliminary accountability workbook and 2025–26 campus transfer preparation.
It checks four rating witnesses, including the literal minimum-size rating for
a reporting-only campus, and the applicable 2025 TAPR witnesses. The 2025
release baseline remains available for its exact source. The 2026 scalar check
pins the transfer preparation and original CSV receipt; qualifying the transfer
edges also requires the separately retained preparation and readback evidence.
[TEA's preliminary accountability release](https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/2026-accountability-rating-system)

## Separately published source tables

A native source release holds a verified table with its original row keys, source revision, and field definitions. You can query and export it directly through `DataRelease`. These tables are currently separate from the default entity enrichment build. See [published TAPR and TPRS supplements](data-releases.md#tapr-and-tprs-supplements) and [annual dropout](research-annual-dropout.md) for how to use them.

Annual dropout is available for school year 2024–25 across campus, district, county, region, and Texas. A campus or district can have several rows for the same school year, distinguished by grade span and calculation variant. Its published values are raw tokens. The 2025–26 TPRS membership and enrollment candidate covers 1,202 districts from the publisher roster and one direct Texas report. It has 168 qualified fields per scope. Campus reports have a separate qualification gate.

The reviewed [PEIMS primary-disability candidates](data-releases.md#peims-primary-disability-counts)
cover districts and Texas for 2012–13 through 2025–26. Four historical editions
include a separate same-year TEA district classification where the PEIMS
report predates its charter column.

TEA's AskTED archive provides fall and spring files from 2018–19 through 2025–26. The live directory provides a daily observation. Use these files to discover district and campus identifiers, retaining the archive edition or acquisition date for each occurrence. Use each performance report's own roster to establish its reporting population. [AskTED archive index](https://tealprod.tea.state.tx.us/Tea.AskTed.Web/Forms/ArchivedSchoolAndDistrictDataFiles.aspx) · [AskTED directory](https://tealprod.tea.state.tx.us/tea.askted.web/Forms/Home.aspx)

[School-year selection and source identity](querying.md#school-years-and-source-editions)
