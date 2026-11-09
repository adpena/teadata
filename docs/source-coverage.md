# Entity enrichment sources and years

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

The 2024–25 regional Student Information and regional and statewide Staff
Information source profiles retain the original two-header CSVs and exact
publisher dictionaries. Each regional file has all 20 ESC identifiers. The
statewide Staff file is one direct Texas observation. These source profiles
preserve publisher tokens and leave measurement periods and value policies
unbound. Campus and district enrichment use their separately configured inputs.
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
