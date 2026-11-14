# Entity enrichment sources and years

The [v0.0.153 normal release](https://github.com/adpena/teadata/releases/tag/v0.0.153)
composes ten native bundles, 12 dataset IDs and 132 edition/scope/partition keys,
alongside 13 unchanged base inputs. It includes PEIMS ethnicity for
2018–19 through 2025–26 and programs for 2022–23 through 2025–26,
statewide by district and campus plus direct Texas, and
district transfer reports for 2023–24 through 2025–26. The primary-disability
2012–13–2025–26 and staff FTE/salary 2022–23–2025–26 families remain available.
The [integration inventory](source-integration-status.json) pins this exact
selection. Library publication does not change deployed profiles or an
unconfigured wheel's default data; select the release explicitly.

PEIMS ethnicity preserves one row per reported group. Use the matching programs
report's direct total rather than reconstructing it from masked categories.
Program counts overlap. District transfer rows preserve direction, reporting
geography, published totals and pair details; they are not campus transfer edges
or TEFA participation. Their current per-entity shortcut lacks a reviewed role
binding; use the canonical source table with an explicit reporting/partner filter.
Inspect each source's roster and limitations before pooling.

## Attendance source additions under qualification

The following November 14, 2026 additions have retained-source replay checks but
are not part of the published v0.0.153 parent above. Standalone qualification,
normal-parent composition, publication and application activation are distinct.
The current work does not imply complete TAPR/TPRS domain coverage.

| Dataset | Actual measurement years | Publisher edition | Qualified candidate coverage | Remaining scope |
| --- | --- | --- | --- | --- |
| `summary_of_finances_attendance` | 2025–26 | Run 47422, Near Final | 1,243 LEAs in each of input/output LPE and DPE tables; six exact-decimal attendance fields | Direct state/sector figures remain separate retained reports; no synthesized state total. Other finance fields remain unqualified. |
| `tprs_attendance_chronic_absence` | 2023–24 and 2024–25 | Named report edition 2025 | Direct Texas and all 20 ESC regions; attendance and chronic-absence percentages for 11 published populations | Complete current district/campus corpus and its partitioned access remain held. Newer report editions do not establish newer measurement years. |

See the [attendance source contract](attendance-sources.md) for definitions,
precision, missingness and exact acquisition versus observation dates. Enrollment,
Membership and funding attendance equivalents are separate populations and units.

Use `entity.peims_special_education_by_primary_disability` or
`entity.dataset("peims_special_education_by_primary_disability")` for lazy field
discovery and exact-year selection. Historical IDs absent from the current entity
snapshot remain accessible through `release.series(..., ids=[...], years=[...])`;
current names and geometry do not become historical facts. These are October
primary-disability enrollment snapshots, distinct from program enrollment and
TAPR Membership. Older Developmental Delay interpretation remains held; preserved
counts do not establish comparability, diagnoses, service intensity or cost.
The product has district, county, region and state reports; this release attaches
district and direct Texas only, and no campus product is implied.

Staff data retain category and role rows, decimals, observed zeros and missing
values. Select those dimensions explicitly; overlapping categories cannot be
summed as unique staff. See [series exports](enrichment.md#observations-across-school-years)
for CSV, DataFrame and Excel provenance and original-header behavior.

For the current registration-to-release gap, see the dated
[source integration inventory](source-integration-status.md) and its
[machine-readable edition/scope comparison](source-integration-status.json).
The v0.0.128 table below remains a historical release description.

This table describes the [v0.0.128 entity release](https://github.com/adpena/teadata/releases/tag/v0.0.128) used in the Quick Start, build ID `17061ef34da36b4c968e0d162cfeb2ba0dfddd64fe4fd8067a785b3dd2bca685`. It includes all 13 recorded inputs, including roster, boundary, and reference inputs used alongside enrichment. The catalog for each selected release records the requested and resolved edition, source period, and provenance.

| Source ID | Use in the entity build | Period in v0.0.128 | Editions in the tracked source configuration |
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

## Statewide district and campus sources

When TEA publishes statewide district and campus products, each is authoritative
at its own grain. Verify coverage separately against each product's reporting
roster and exact edition, and report any gaps by scope. Use a direct Texas
report for [statewide comparisons](querying.md#compare-a-published-measure-with-texas) when the
measure, population definition, and measurement period match.

### Facility usage: acquired, integration pending

The [2025–26 facility usage review](facility-usage.md) documents TEA's statewide
district collection: 17,763 facility records from 946 submitting districts and
71 status-only district rows. TEA says charter schools are not currently required
to report. The data are facility records, with some publisher-supplied campus links,
not a campus enrollment or capacity series. Source bytes and definitions have
been inspected; this source is not yet registered, qualified, or attached to
v0.0.152. Unresolved correction lineage, mixed enrollment populations and shared
facilities constrain the next normal-pipeline integration.

### Historical archive routes

The historical backfill targets all 14 school years from 2012–13 through
2025–26. Track each source family and geography separately for every year.
An archive listing establishes publisher availability. Retained files,
field qualification, and query availability are separate stages.
Consult the selected release's catalog for queryable editions and fields.

| Source | Publisher archive and example editions | Published scopes and limits |
| --- | --- | --- |
| TAPR | [TEA's TAPR archive](https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/texas-academic-performance-reports) lists every edition from 2012–13 through 2024–25 and schedules the formal 2025–26 edition for December 2026. | Prefer statewide district and campus files, ESC regions, and direct Texas tables where offered. Use each edition's data element reference and masking instructions. Keep rolling TPRS observations separate from the formal TAPR edition. |
| Historical TPRS | [TPRS archive](https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/texas-performance-reporting-system-archives) lists editions starting in 2012–13 | District, campus, region, and Texas reports. Check domain and bulk availability for each edition independently of TAPR. |
| PEIMS standard reports | [TEA product index](https://tea.texas.gov/data-reports/student-data/standard-reports/peims-standard-reports) lists years separately for each product. See the [primary-disability series](data-releases.md#peims-primary-disability-counts) for one historical example. | Enrollment and programs have campus, district, and Texas products. The primary-disability series has district, county, region, and Texas scope. |
| Accountability | [2013](https://rptsvr1.tea.texas.gov/perfreport/account/2013/) and [2014](https://rptsvr1.tea.texas.gov/perfreport/account/2014/) final ratings, manuals, and data download links | District and campus ratings, with separate region and Texas summaries. Calendar-year systems and rating scales can change. |
| Longitudinal graduation and annual leavers | [TEA's completion, graduation and dropout data index](https://tea.texas.gov/data-reports/school-performance/accountability-research/completion-graduation-and-dropout/completion-graduation-and-dropout-data) | Four-, five-, and six-year graduation are separate longitudinal cohorts and follow-up windows. Annual leavers and annual graduates describe separate reporting populations. Check the annual archive for campus, district, county, ESC, and direct Texas routes. |
| Student grade-level retention | [TEA's grade-level retention data index](https://tea.texas.gov/data-reports/school-performance/accountability-research/grade-level-retention/grade-level-retention-data-search) | Compare spring grade with the following fall. The denominator includes students who return or graduate. Bulk campus, district and, where offered, county files are separate from printed Texas and ESC reports. |
| PEIMS financial actuals and budgets | District and charter [single-file archives](https://tea.texas.gov/about-tea/state-funding/state-funding-reports-and-data/peims-single-file-financial-data-downloads) include both years. The [financial reports index](https://tea.texas.gov/about-tea/state-funding/state-funding-reports-and-data/peims-financial-standard-reports) also lists both years of campus actuals. | District and charter account files are separate populations. The historical reports organized by campus require a district selection. Retain direct Texas figures from their own published reports. See [financial sources](financial-sources.md). |
| Summary of Finances | Verified [School District State Aid Reports](https://tealprod.tea.state.tx.us/Tea.FspReports.Web/) district and charter downloads for 2012–13 and 2013–14 | These summaries are direct statewide totals for separate sectors. TEA also offers individual district and charter reports. Keep the run-list date, workbook update date, run, status, and any row-level run IDs separately. |
| Geography | [TEA's ArcGIS archive](https://schoolsdata2-db440-tea-texas.opendata.arcgis.com/) and historical locator layers for [2012–13](https://schoolsdata2-tea-texas.opendata.arcgis.com/datasets/school2012) and [2013–14](https://schoolsdata2-tea-texas.opendata.arcgis.com/datasets/school2013) | Campus points and district polygons have separate vintages. Preserve records with absent coordinates and investigate duplicate campus identifiers before using points for joins. Establish historical polygon coverage independently. |
| Discipline and transfers | [Discipline summary archive](https://tea.texas.gov/data-reports/student-data/discipline-data-products/discipline-reports) and [district transfer reports](https://rptsvr1.tea.texas.gov/adhocrpt/Standard_Reports/Transfer_Reports/district_transfer_report.html) reach the starting years | Discipline offers statewide district, campus, and Texas downloads. TEA documents [statewide campus transfer CSVs](https://rptsvr1.tea.texas.gov/adhocrpt/Standard_Reports/About/about_transfer_reports.html) beginning in 2020–21. |
| Educator workforce | [TEA educator reports](https://tea.texas.gov/data-reports/educator-data/educator-reports-and-data) and [version 5 hiring workbooks](https://tea4avcastro.tea.state.tx.us/oess/edrs/regional-dashboards/) | The local hiring producer supports district totals and reported EPP detail from 2009–10 through 2025–26, and direct Texas observations from 2015–16 through 2025–26. The local statewide continuous retention candidate covers first-teaching cohorts 2013–14 through 2024–25 by demographics and gender, and by preparation route except 2017–18. Pathway-at-entry retention, employed certification and newly certified supply have separate qualification gaps. [Educator workforce coverage](educator-workforce-sources.md) |

For TAPR, use the master reference for file codes, the report dictionary to
locate printed cells, the glossary for the measure and population, and the
masking guide for the table's published tokens. The old Profile masking
guides use `-2` for values outside a reasonable range. Performance and
advanced count downloads have separate rules.

| Edition | Field and report references | Definitions and disclosure |
| --- | --- | --- |
| 2012–13 TAPR | [Master reference](https://rptsvr1.tea.texas.gov/perfreport/tapr/2013/xplore/taprref.html) · [Report dictionary](https://rptsvr1.tea.texas.gov/perfreport/tapr/2013/datadict.pdf) | [Glossary](https://rptsvr1.tea.texas.gov/perfreport/tapr/2013/glossary.pdf) · [Masking guide](https://rptsvr1.tea.texas.gov/perfreport/tapr/2013/masking.html) |
| 2013–14 TAPR | [Master reference](https://rptsvr1.tea.texas.gov/perfreport/tapr/2014/xplore/taprref.html) · [Report dictionary](https://rptsvr1.tea.texas.gov/perfreport/tapr/2014/datadict.pdf) | [Glossary](https://rptsvr1.tea.texas.gov/perfreport/tapr/2014/glossary.pdf) · [Masking guide](https://rptsvr1.tea.texas.gov/perfreport/tapr/2014/masking.html) |
| 2024–25 TAPR | [Edition index](https://rptsvr1.tea.texas.gov/perfreport/tapr/2025/index.html) · [Bulk download selector](https://rptsvr1.tea.texas.gov/perfreport/tapr/tapr_dd_download.html?year=2025) | [Glossary](https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/2024-25-comprehensive-tapr-glossary-0.pdf) · [Masking guide](https://rptsvr1.tea.texas.gov/perfreport/tapr/2025/masking.html) |
| 2025–26 TPRS observations | [2026–27 user guide and publication schedule](https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/tprs-user-guide-2026.pdf) | [Comprehensive glossary](https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/comprehensive-tprs-glossary.pdf) and the disclosure legend on the exact report. |

The 2025–26 TPRS Graduation Profile measures **2024–25 annual graduates**.
The named-report intake reader verifies the selected geography and keeps
the displayed benchmark cells separate from the local count and percent.
The campus display includes district and Texas benchmark counts. The
district display includes Texas counts and percents. The direct Texas
report provides the statewide observation. The report legend identifies
`-` as no students in the group and `?` as a value outside a reasonable
range. The reader preserves both raw tokens and keeps them distinct from
an observed zero. [Named Texas report](https://rptsvr1.tea.texas.gov/cgi/sas/broker?_service=marykay&_program=perfrept.perfmast.sas&_debug=0&ccyy=2026&id=S&lev=S&prgopt=reports%2Ftapr%2Fgradprofile.sas)

The named layout has been checked against selected campus, district and
Texas reports. Statewide intake and release activation remain open. The
linked Excel reports in the checked sample declare a worksheet range
that omits the final CTE Completers row. TEA's separate Annual Leavers
report publishes 400,377 graduate records and warns that students can
appear more than once. TPRS prints 400,365 annual graduates. Keep the
two source totals separate. Neither is the four-year longitudinal
graduation count. [Annual Leavers state report](https://rptsvr1.tea.texas.gov/acctres/leavers/2425/state.html) · [Class of 2025 graduation definitions](research-graduation.md)

PEIMS [submission standards](https://tea.texas.gov/data-reports/data-submission/peims/peims-data-standards/peims-data-standards)
document input elements and codes. Public aggregate products have their own
definitions and disclosure rules, such as the [primary-disability report criteria](https://rptsvr1.tea.texas.gov/adhocrpt/Standard_Reports/About/abtspeced.html)
and the legend on each report. The [PEIMS financial file dictionary](financial-sources.md)
documents a separate layout for account data.

Keep school years, measurement periods, and publication editions distinct.
Check keys against each edition's own roster, including entities without
geometry. Verify units, masking, and comparability at the field level before
binding a historical series. Use publisher Texas observations where available.
The 2020–21 through 2022–23 regional TAPR Profile downloads spell ESC codes
with a leading apostrophe. The reader keeps that spelling in source tables
and CSV exports. It uses the two-digit code when validating keys and rejects
duplicate codes even when their source spellings differ.

In the 2024–25 Student Information report, mobility, student attrition, and
disciplinary placements describe 2023–24. Campus and district metadata record
that period for the reviewed count and percent fields. The
[annual attrition workbook](https://tea.texas.gov/data-reports/school-performance/accountability-research/completion-graduation-and-dropout/2023-24-annual-attrition-rates_0.xlsx)
independently corroborates the reviewed direct Texas count (765,408) and rate
(18.0%). It measures whether eligible students enrolled in fall 2023–24
returned to the same campus in fall 2024–25; a student can stay in the same
district and count as a nonreturner. The TAPR glossary prose names the prior
fall interval. The workbook and the exact report observations support the
2023–24 binding; their reporting rosters and disclosure tokens are separate.
The [2024–25 TAPR Staff Information glossary, page 37](https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/2024-25-comprehensive-tapr-glossary-0.pdf#page=37)
defines teacher turnover against fall 2023–24 teacher FTE, with fall 2024–25
as the comparison. Teachers who stay employed by the district in a
non-teaching role can count toward turnover. The report does not show
this measure at campus level. The denominator is separate from the
fall 2024–25 total teacher FTE. Average actual salary covers regular
duties; supplemental pay is outside the measure. Retain the printed
rate alongside its published numerator and denominator, whose displayed
precision can differ.

The separate 2024–25 campus and district Staff Information source editions
carry explicit periods and units for teacher FTE, average teacher experience,
average regular-duty salary, and students per teacher FTE. The district
edition also binds the published turnover percentage. Independent TEA
reports for Hitchcock ISD and Crosby Middle verify the selected columns.
Other fields retain their raw publisher tokens and unknown measurement
periods. Read these editions explicitly for source analysis.
[Hitchcock ISD Staff Information](https://rptsvr1.tea.texas.gov/cgi/sas/broker?_debug=0&_program=perfrept.perfmast.sas&_service=marykay&ccyy=2025&id=084908&lev=D&prgopt=reports%2Ftapr%2Fstaff.sas)
· [Crosby Middle Staff Information](https://rptsvr1.tea.texas.gov/cgi/sas/broker?_debug=0&_program=perfrept.perfmast.sas&_service=marykay&ccyy=2025&id=084908042&lev=C&prgopt=reports%2Ftapr%2Fstaff.sas)

The 2024–25 regional Student Information and regional and statewide Staff
Information source profiles retain the original two-header CSVs and exact
publisher dictionaries. Each regional file has all 20 ESC identifiers. The
regional Student profile binds 2024–25 Membership and Enrollment totals,
race counts and percentages, and special education counts and percentages.
Total mobility and attrition use 2023–24. All 38 selected fields match the
separate TEA Student Information reports for ESCs 06 and 16. The annual
attrition workbook also matches both attrition fields for all 20 ESCs.
A campus change within the same region can count as attrition. Regional
values use TEA's own regional reporting population.

Direct Texas Student Information also has a reviewed 2023–24 edition.
It retains all 240 source columns and binds 76 measures to their exact
periods. Mobility, disciplinary placements and attrition describe 2022–23.
The disciplinary placements display includes DAEP and JJAEP. Attrition
observes return to the same campus in fall 2023–24. The annual attrition
workbook corroborates the published count and rate. The linked masking
guide has a 2022–23 page title and 2023–24 body. The raw source has no
table-wide value policy. [TEA 2023–24 reports and references](https://rptsvr1.tea.texas.gov/perfreport/tapr/2024/index.html).

The statewide Staff file is one direct Texas observation. Regional and statewide
Staff bind the 2024–25 school year and published units for teacher FTE, average
professional experience, average regular-duty salary, students per teacher
FTE, and teacher turnover. The five values match separate TEA Staff reports
for direct Texas and ESCs 06 and 16. Each observation keeps its publisher
scope and raw CSV token. The turnover rate uses the fall 2023–24 baseline and
fall 2024–25 comparison. The remaining fields have unbound semantics, and
these source profiles carry no table-wide value policy. Campus and district
enrichment use their separately configured inputs.
[TEA TAPR data download](https://rptsvr1.tea.texas.gov/perfreport/tapr/tapr_dd_download.html?year=2025)

## Published 2026 entity snapshot

The explicit [v0.0.136 entity snapshot](data-releases.md#2026-entity-snapshot-and-tprs-student-data)
updates `accountability` to calendar 2026 preliminary ratings and
`campus_transfer_reports` to school year 2025–26. The other 11 resolved inputs,
including both 2024–25 TAPR student and staff profiles, retain their source
artifacts. The scalar verifier checks four rating witnesses and selected
2025 TAPR witnesses. The separate [transfer edge readback](https://github.com/adpena/teadata/releases/download/v0.0.136/ENTITY_READBACK.json)
reports observed and suppressed edge counts. The manifest pins the preparation and
original CSV receipt. The entity build receipt records pending broader source review.
See [TEA's preliminary accountability release](https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/2026-accountability-rating-system).

The v0.0.136 snapshot also pins a 2025–26 TPRS Membership and Enrollment
attachment with 168 reviewed fields per scope for 1,202 reporting districts
and one direct Texas report. It has no campus edition. Query its fields by
[school year](querying.md#school-years-and-source-editions). The attachment
leaves the entity snapshot's TAPR fields at their original periods.

## Results Driven Accountability

The explicit `teadata.rda` reader covers the 2024 and 2025 Results Driven
Accountability Indicator 1 STAAR passing-rate files for bilingual education,
other special populations, and special education. The reader requires the
matching original district CSV. It checks the publisher bytes before reading
and returns one Arrow row per district and subject.

```python
from teadata.rda import rda_indicator1_metadata, read_rda_indicator1

source = rda_indicator1_metadata(2025, "04")
district_rates = read_rda_indicator1("2025-D-04-original.csv", edition=2025, area="04")
texas_mathematics = source["fields"][0]["direct_texas"]
```

The metadata supplies the district download controls, data dictionary,
technical appendix, masking rules, and direct Texas report. Texas values
come from the matching state report. District single-year rates and pooled
multi-year rates occupy separate columns. Where TEA reports only a pooled
rate, the single-year value remains absent even when the current numerator
and denominator are visible. The raw publisher token, disclosure status,
and field code remain available. A null raw pooled value indicates an
omitted column, while a literal period indicates an available column with
no pooled rate on that row.

Build a local native candidate for the 2025 special-education area from the
retained provenance store and Poppler's `pdftotext`.

```sh
uv run --locked python -m teadata.scripts.refresh_rda_indicator1 \
  --store "$TEADATA_PROVENANCE_DIR" --output "$RDA_CANDIDATE_DIR"
```

The candidate has a district table with 1,208 publisher rows and a Texas
table with four subjects. Open its `release.json` using the manifest hash
in `MATERIALIZATION.json`. Select the exact district or state `EditionKey`
from `DataRelease.catalog.editions()`. The district leaf contains 18
selected source columns in the original CSV order, including literal
district and ESC codes. Its exports accept `restore_original_headers=True`
for the original CSV code row. The separate Texas leaf retains the printed
rates and comma-formatted counts from page 5 of the state PDF. The native
leaves carry raw values. Use `read_rda_indicator1()` for typed values and
disclosure statuses. Pooled rates have an unresolved measurement window.

[TEA district data and dictionaries](https://rptsvr1.tea.texas.gov/pbm/download.html) ·
[TEA state reports](https://tea.texas.gov/assessment-and-accountability/monitoring-and-interventions/rda/rda-state-reports/rda-state-reports)

## Educator hiring

The local native producer registers `educator_district_new_hires` from
dashboard version `5` and `educator_state_new_hires` from the separate
`2026-04` workbook. `educator_district_new_hires_by_epp` holds the separate
version 5 workbook's district detail. District measurement years run from 2009–10 through 2025–26;
direct Texas observations begin in 2015–16. The default producer selection
remains 2025–26. `--all-years` includes every available year for each scope,
and repeated `--school-year` selects ending years explicitly. An earlier
selection builds district tables and records Texas as unavailable for those
years; it does not fabricate or backfill a state observation.
The district table has one row per district, year and published category.
It includes charters in the publisher's reporting roster. The state table
holds one direct Texas observation with nine category counts, their
fractional ratios and the statewide total. The producer checks every
district category sum against that report where it is available.

```sh
uv run --locked python -m teadata.scripts.refresh_educator_hiring \
  --store "$TEADATA_PROVENANCE_DIR" --output "$EDUCATOR_HIRING_CANDIDATE_DIR"
# Or select all available history in a different, new directory:
uv run --locked python -m teadata.scripts.refresh_educator_hiring \
  --store "$TEADATA_PROVENANCE_DIR" --output "$EDUCATOR_HIRING_HISTORY_DIR" --all-years
```

Open the candidate's `release.json` as a `DataRelease` and select its
measurement year explicitly.

```python
district_hires = release.observations(
    "educator_district_new_hires", "new_hires_count",
    school_year=2026, scope="district",
).to_df()
texas_hires = release.observations(
    "educator_state_new_hires", "total_new_hires_count",
    school_year=2026, scope="state",
).to_df()
available = release.observation_years(
    "educator_district_new_hires", "new_hires_count", scope="district",
)
historical_hires = release.observations(
    "educator_district_new_hires", "new_hires_count",
    school_year=2013, scope="district",
).to_df()
```

Available years reflect the selected release, not every year in the original
workbook. Omitting `school_year` chooses the latest shared measurement year
for the requested fields and scope. An unavailable year raises an error with
the available years. Discovery and planning read metadata; materialization
reads the selected local Parquet leaf. Installation does not fetch this
history automatically, and this producer performs no network requests.

Every year, the separate EPP-disaggregated workbook must have exactly the same district/year/category
keys and grouped counts, including explicit zero keys. This establishes
cross-product consistency; both products may share upstream processing.
For 2015–16 onward, every primary district category sum is also compared with
its direct Texas count and published fraction.
Neither that check nor a sum of district counts establishes an independently
published Texas observation for an unavailable year. EPP detail is queryable
at its reported five-field grain. See [educator workforce source notes](educator-workforce-sources.md)
for its worksheet, dictionary gaps, query example, and interpretation limits.

The report edition is distinct from each table's measurement year. The whole
workbook has no single measurement period, so that source-level period is
unknown; each fact's reviewed table period is explicit. Historical tables
retain their publisher's reporting IDs, names and row coordinates, including
districts absent from the latest reporting roster. The 2026 table identities
remain unchanged. This is the retrospective version 5 history; it does not
reconstruct what the publisher originally reported in each historical year.
Some historical rows have a valid district ID and count but a literal blank
district name. Preserve that blank; do not substitute a present-day name.
CSV and DataFrame outputs retain the empty string; Excel displays an empty
text cell, which some spreadsheet readers return as `None`.

Counts preserve integer types, source values, disclosure status and Excel
row coordinates. State ratios retain the publisher's fractions with
unit `ratio`. For example, 0.10025823404158928 represents approximately
10.0258 percent. Source-table exports preserve the raw fraction text.
Excel also writes a fraction as text when its numeric cells would lose
precision. An absent district/category row remains absent.
CSV includes selected fields and their raw/status support; retain the separate
`view.provenance().to_dict()` descriptor with the CSV for its source bindings,
references, notes and release identity. Excel embeds a Provenance worksheet.
The original-header export option remains restricted to its explicitly
supported raw-source families; it is not yet enabled for these typed hiring
tables.

The categories describe certification or entry pathway at hire. They include
re-entering teachers and teachers previously working part-time. Preparation
route, EPP supply and statewide continuous teacher retention have separate
definitions. TEA's regional dashboard history attributes its changes to
certification date logic, the 0.5 FTE threshold and the part-time pathway
to version 4 (March 2025). Version 5 adds 2025–26. This history applies
to the regional dashboard; the state workbook was issued separately.
Historical products require their release and revision identities.

The linked regional documentation workbook is dated January 2023 and the
business-rules PDF April 2025. Their definitions are retained with those
dates and their documented gaps. A link from the current dashboard does
not establish that an older document describes every current field.
[TEA hiring methodology and version history](https://tea4avcastro.tea.state.tx.us/oess/edrs/regional-dashboards/)

## Separately published source tables

A native source release holds a verified table with its original row keys, source revision, and field definitions. You can query and export it directly through `DataRelease`. These tables are currently separate from the default entity enrichment build. See [published TAPR and TPRS supplements](data-releases.md#tapr-and-tprs-supplements) and [annual dropout](research-annual-dropout.md) for how to use them.

Annual dropout is available for school year 2024–25 across campus, district, county, region, and Texas. A campus or district can have several rows for the same school year, distinguished by grade span and calculation variant. Its published values are raw tokens.

The reviewed [PEIMS primary-disability candidates](data-releases.md#peims-primary-disability-counts)
cover districts and Texas for 2012–13 through 2025–26. Four historical editions
include a separate same-year TEA district classification where the PEIMS
report predates its charter column.

The dated AskTED fall and spring download controls cover 2018–19 through 2025–26.
The same publisher site separately lists Texas School Directory editions
back to 1999–2000. The live directory provides a daily observation. Keep the
edition or acquisition date with each district or campus ID occurrence,
and use each performance report's own roster to establish its population.
[AskTED archive index](https://tealprod.tea.state.tx.us/Tea.AskTed.Web/Forms/ArchivedSchoolAndDistrictDataFiles.aspx) · [AskTED directory](https://tealprod.tea.state.tx.us/tea.askted.web/Forms/Home.aspx)

The retained statewide AskTED lifecycle exports are separate from the
term directories. `teadata.askted_lifecycle.read_lifecycle_report` reads
one verified public campus, charter campus, district closure, or district
reorganization CSV from its acquisition receipt and exact object hash.
The typed events preserve publisher order, logical CSV record numbers,
printed names and dates, and explicit annexation roles. A reused ID can
have multiple observations. Charter campus records overlap public campus
records. Keep both source identities when reconciling events. The raw
January 1, 1800 date remains unresolved. The report print date does not
establish the date of any event. [AskTED reports](https://tealprod.tea.state.tx.us/Tea.AskTed.Web/Forms/ReportSelection.aspx)

[School-year selection and source identity](querying.md#school-years-and-source-editions)

## 2025–26 TPRS class size

`teadata.tprs_class_size.read_class_size_report` reads one retained Student
Information report using its acquisition receipt and exact content SHA-256.
It preserves the local class-size token and its HTML cell coordinates.
A campus report can omit the table. The publisher's dash means no students
and the question mark flags an improbable or out-of-range result. Use the
direct Texas report for statewide values. The reader is the offline
preparation step for the queryable class-size attachment.
[TPRS glossary, p. 42](https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/comprehensive-tprs-glossary.pdf#page=42).

`class_size_projection` converts reports read through that receipt-verified
reader into a typed Arrow table for one geography. It keeps a campus row
when the source report omits the table and distinguishes that case from a
missing grade or subject within a table. The Texas table contains one
direct statewide report. The Arrow projection is available offline.
Querying it through a TEA Data release requires catalog and source admission.

The offline producer can call
`teadata.tprs_statewide_intake.project_reviewed_class_sizes(release, store, scope=...)`
on the exact reviewed Student Information attachment. It reads only the
selected scope's reports from the pinned ledgers and checks the complete
publisher roster for campus scope. It requires the retained acquisition
objects. This step produces a candidate table and its column support for
the existing native source pipeline.

Build an unpublished district and Texas native candidate with the original
report acquisitions and the retained source references:

```sh
uv run --locked python -m teadata.scripts.refresh_tprs_class_sizes \
  --student-attachment /path/to/reviewed-student-release.json \
  --store /path/to/provenance --output /path/to/new-candidate
```

The candidate can be opened locally with `DataRelease`. The default build
provides district and Texas editions and uses the direct Texas report for
state values. Add `--include-campus` with the exact reviewed complete Student
Information attachment to also build the campus partitions. The producer
keeps the attachment's original partition boundaries and verifies each
native table against its Arrow projection. The 2025–26 campus roster has
9,105 reports, of which 612 omit the class-size table. Those reports still
have a row with missing measurements and the reason `table_not_published`.

Use [an exact campus edition key](querying.md#campus-class-size-observations)
with the stand-alone v0.0.147 manifest or a matching offline candidate.

To qualify the published v0.0.147 child offline, stage its manifest and
assets together with the reviewed `tprs-student-2026-campus-complete.release.json`
from v0.0.138 and its edition metadata and roster descriptors. The Student
descriptors supply the report census and partition boundaries. The retained
provenance store must contain the class-size processing closures, including
the original Student HTML and roster responses, and the three named Student
Information workbook acquisition closures from `teadata.tprs_students.definition()`.

```sh
uv run --locked --offline python -m teadata.verification --kind library \
  --native-release /path/to/stage/tprs-class-size-2026.release.json \
  --native-pin 1512f0ef1a8b1df41f969c683c3ec1dab0d607499dddb589127c6f5a41b9af4e \
  --source-store /path/to/stage/provenance
```

The gate compares all class-size leaves with the retained HTML and checks
local class-size cells from three independent workbook exports. For an
entity release that attaches this child, stage the same evidence beside the
local child manifest, with its store named `provenance`. The composed release
check uses the same qualification and rechecks the evidence when verifying
an existing receipt.

## School FIRST financial accountability

The final **2025–26** FIRST rating edition was released November 5, 2026 and
measures **fiscal year 2024–25**. It is separate from academic accountability.
A worksheet's last-updated column records the publisher's row update, which
can predate the final release. See the [final release notice](https://tea.texas.gov/taa-letters/official-release-final-2025-2026-first-ratings)
and [TEA's explanation of the financial basis and indicators](https://tea.texas.gov/taa-letters/preliminary-2025-26-first-ratings).

| Source ID | Geographic scope and population | Offline qualification |
| --- | --- | --- |
| `first_financial_accountability_district` | District scope, 1,018 School FIRST districts | Final workbook and complete public roster. 840 A, 107 B, 60 C, 11 F. Scores are 0–100 points. TEA can assign F regardless of score. |
| `first_financial_accountability_university_charter` | District scope, five charters operated by public higher education institutions | Final workbook and complete public roster. All five received Pass. Numeric score is not applicable. |

Both populations are available as native district observations in
[TEA Data v0.0.146](https://github.com/adpena/teadata/releases/tag/v0.0.146).
Their workbook, public roster,
and announcement controls were checked separately. The 165 graded
non-university charters remain on hold. The final export, roster, and
Charter FIRST statistics agree with one another, but the [TEA newsroom table](https://tea.texas.gov/about-tea/newsroom/tea-releases-final-2025-2026-financial-accountability-ratings)
reports one fewer B and one more C. The reason for that discrepancy remains
unverified. No combined total or campus rating is admitted.

To replay the exact retained acquisitions into a new unpublished directory,
run `uv run --locked python -m teadata.scripts.refresh_first_ratings --store
<retained-store> --output <new-directory>`. The producer checks the archived
workbooks, roster pages and scoped newsroom counts offline, then verifies
both leaf artifacts and their Arrow readback.

The production verification command admits the exact reviewed candidate
through `--native-release` with its manifest SHA-256 and adjacent
`provenance` store. It uses the same checks when that candidate is attached
to an entity release. Verification replays the archived configuration,
crosswalk, workbooks, roster pages, and newsroom controls against the
physical and public Arrow tables. The admitted child is currently
unpublished.
