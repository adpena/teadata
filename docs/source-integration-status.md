# Source integration status

The packaged selector identifies **normal data release 0.0.186**. This is a
frozen analysis release. Software publication and application deployment are
separate gates. Historical requests keep their pins.
The [machine inventory](source-integration-status-0.0.186.json), generated from
the composed 0.0.186 manifest and current source configuration on November 23,
2026, records
128 registered IDs, 321 configured editions,
13 base inputs and 88 native attachments covering 59 dataset IDs and
590 edition/scope keys. Registration, acquisition, qualification and query
access are separate statuses.

- Manifest: `e82f01bc23ef7b812a76d02963f36d5148ed1336967d68e645697407b3fbdca5`.
- Entity core manifest: `2a19e22760af8da443af29a38d01d7d9c13f65095fd739e78bb7a0a1dc345f1b`.
- Build: `31038e45f324b824e1d7f27dee4e1c6e001d5d770013bd710c1e9eda7ca0d368`.
- Processing record: `270ef47f9e8f3a2d7ce5057ae7c8c73061c4db88123cb64f47b2522cb7cb43bc`.

## Direct Texas CCMR

Version 0.0.186 adds eight statewide rating editions in
`state_accountability_ccmr`. Query `scope="state"` and the school year of the
annual graduates. Available cohort years end in 2017, 2018 and 2020–25.
The fields retain the graduate count, displayed credit and reported percentage
from the direct Texas report. The 2018 rating edition displays fractional
credit. Consult the edition-specific rules before interpreting changes.
[Source coverage and methods](source-coverage.md#source-editions-periods-and-exceptions)

## Historical STAAR, accountability detail and career readiness workbooks

Version 0.0.185 added 139 edition and scope combinations. Forty historical
STAAR2 tables cover Texas, ESCs, districts and campuses. Sixteen
accountability CCMR tables cover district and campus rating editions
2018, 2019 and 2021–26. Six 2026 detail tables cover Student Achievement
STAAR Performance, Academic Growth and Closing the Gaps at district and
campus scope. Replay compared
159,242,291 original CSV cells against each prepared table, final leaf and original-header export. Named publisher reports supplied
3,565 independent checks.

The 26 qualified CTE workbooks add 77 edition and scope combinations.
Literal OOXML and an independent workbook reader compared
2,643,140 cells across 301,909 rows.
Mixed school years, interim layouts, repeated certification keys and
unassigned ESC issues are described in the
[coverage and source exceptions](source-coverage.md#source-editions-periods-and-exceptions).

## Historical Student Programs and primary disability geography

The Programs backfill adds 24 Texas, district and campus editions from
2014–15 through 2021–22. The primary disability geography source adds 28
editions from 2012–13 through 2025–26. The shared verification dispatch
replayed 1,985,368 raw cells, checked 327 values from named Programs reports
and reconciled 51,912 county-by-ESC and ESC values against separate TEA
HTML reports. Raw markers, normalized status and source row are retained.
As published, a county may appear in several ESCs. See the
[coverage and classification notes](source-coverage.md#accountability-assessment-readiness-and-special-education).

## PEIMS teacher FTE and course enrollment

Three attachments cover statewide district and directly reported Texas data
from 2012–13 through 2025–26. The course code and scope are part of the grain, so
the same student or teacher can contribute to several courses. Native leaves
retain the raw publisher tokens, value statuses and source row locators.
Campus course affiliations remain under review.
See the [course source notes](peims-sources.md#build-and-verify-a-course-candidate).

## Historical PEIMS staffing

The 2018–19 through 2021–22 Staff FTE and Salary reports add 210,306
publisher rows across eight district and direct Texas editions. FTE uses
exact scale-two decimals. Published total and average base pay retain their
nominal dollar units. This source has no campus scope. Six earlier direct
Texas editions extend statewide staffing to 2012–13 and add 396 verified
publisher rows. The corresponding district editions remain outside coverage
while undetermined publisher dot tokens are investigated.
See the [staff source notes](peims-staff.md).

## TPRS CTE Membership

The two 2025–26 editions cover 1,202 districts and the direct Texas report.
The all-grade and grades 9–12 measures preserve separate denominators,
original report cells and rate statuses. The 97 district grades 9–12 rates
with zero denominators report `not_applicable`. See the
[coverage and district query example](source-coverage.md#accountability-assessment-readiness-and-special-education).

## STAAR publisher assessment rates

The normal release includes All Grades from 2012–13 through 2025–26 for
years with spring testing, and End-of-Course from 2023–24, at campus,
district, ESC region and direct Texas scopes. The statewide comparison
helper still uses its separately reviewed recent-year rate bindings. Independent named records were reconciled
for each included report edition and scope. Other physical columns keep
their source and export identity. Cross-year scientific comparability and
inferred denominators remain outside this qualification.

## Campus transfer report coverage

The normal catalog covers all six statewide campus transfer reports from
2020–21 through 2025–26. Release 0.0.172 added 2022–23 and 2023–24 with 279,340
and 297,734 publisher rows. All 27 earlier native attachments and 13 base
inputs keep their source identities. The default at campus scope remains
2025–26.

Each observation keeps the reporting campus, partner campus, published
direction, raw count token, value status and original row locator. Archived
AskTED records support historical reporting IDs that are absent from the
current registry. The retained district reports match 364 and 372 fully
observed directional district groups, respectively. Four pairs per edition
have different masking across directions. The reconciled edges remain on
hold, and the source rows retain each publisher observation. See the
[transfer source guide](transfer-sources.md) for scope and limitations.

## District transfer report history

The normal district namespace now offers 14 school years, 2012–13 through
2025–26, for the publisher's directional district transfer reports. The
11 added historical editions preserve all 389,183 rows from the separately
published v0.0.141 bundle, including raw count tokens, statuses, report directions,
publisher totals and separators. The reporting district is the entity bound to
each source row. Report rows do not establish a single statewide transfer count
or reconciled district-pair edges.

Qualification replayed the retained original rows and checked 12,073 rows from
20 independently selected TEA reports. For 2022–23, campus detail summed to the
same count in 364 fully visible directional district-pair groups. Groups with suppression
were excluded from that comparison, and the separate campus-edge hold remains.
The normal release preserves the 22 earlier attachment identities. The new child
is qualified separately, with an offline normal-parent query and export check.

## Retained SOF DPE coverage

`summary_of_finances_dpe` adds **486 fields: 462 numeric quantities and 24
literal code/text fields**, from the 2025–26 Near Final publication run 47422.
The source-wide roster has 1,243 LEA recipients in the first three tables;
I&S has 1,030. The roster includes 20 education service centers and Bowie County,
which are preserved as source recipients rather than invented district entities.
The normal entity snapshot contains 1,229 districts; it is not the source roster.

The full raw inventory preserves all 593 facts and 729,857 original cells.
The admitted subset passed exact source/value/status/coordinate replay for
598,560 cells and 110 frozen original-report display/workbook-cell pairs from
two district/run reports. These are distinct checks: the latter verify exact
retained text and raw-cell pairs, not recomputed report-rounding parity.
No publisher formula is certified or repaired. Source acquisitions were reused;
the original download dates remain in provenance.

Units distinguish currency, funding rates, headcounts, FTE and attendance
equivalents. Full source precision is retained without binary-float conversion.
Each field has its actual bound period; earlier tax, insurance and prior-law
observations do not inherit the funding year. Counts, codes, missing values and
observed zero remain separate. **104 fields remain held**, with exact reasons;
three attendance fields keep their existing canonical owner. See the
[field definitions, admission ledger and limits](sof-dpe-sources.md).

The existing catalog, lazy entity dataset namespace, case-insensitive field
selection, explicit/default year resolution and standard exports expose this
family through the normal parent. Actual Cayuga, Austin and charter A+ Academy
entity checks preserve child and normal-parent provenance. Representative real
DataFrame, CSV, Excel and SQLite exports preserve exact values and citations.
The SQLite check used exact decimal text storage and Arrow readback; it does
not establish another database dialect or every possible query.

All 21 preexisting native source pins and 13 base inputs are unchanged; their
matching source qualifications are reused. Current 179-field Student, Class
Size and 2024–25 continuity sources remain unchanged. The
[complete Student field accounting](tprs-student-field-status.md) separately
accounts for every retained physical cell and the precise remaining holds.
Mobility/discipline year conflicts, CTE token/denominator questions and publisher
blanks are not silently resolved or substituted. No general adjusted refined
ADA is filled from an SSRA-specific field.

Authenticated production continuity export remains unverified. Anonymous health,
Policy downloads and runtime identity checks do not establish that workflow.
Application adoption belongs to the application rollout, not this source proof.

## Qualified native coverage

The table lists exact publisher/report editions and **known bound field
periods**, not a promise that every raw column has a qualified metric. A school
year ending 2026 means 2025–26. Multiple periods in a source can describe lagged
fields. `Unspecified` stays unresolved; a stable name does not establish
comparability. Inspect the machine inventory or catalog field metadata for
individual definitions, units, population, revisions and source links.

| Native dataset | Report editions | Scopes | Bound field periods |
| --- | --- | --- | --- |
| `campus_tapr_staar_all_grades` | 2026 | campus | SY ending 2026 |
| `campus_tapr_staar_end_of_course` | 2024–2026 | campus | SY ending 2024–2026 |
| `campus_tapr_staff_profile` | 2025 | campus | SY ending 2025 |
| `campus_transfer_reports` | 2021–2026 | campus | SY ending 2021–2026 |
| `district_tapr_staar_all_grades` | 2026 | district | SY ending 2026 |
| `district_tapr_staar_end_of_course` | 2024–2026 | district | SY ending 2024–2026 |
| `district_tapr_staff_profile` | 2025 | district | SY ending 2025 |
| `district_transfer_reports` | 2013–2026 | district | SY ending 2013–2026 |
| `educator_district_new_hires_by_epp` | 5 | district | SY ending 2010–2026 |
| `educator_district_new_hires` | 5 | district | SY ending 2010–2026 |
| `educator_state_initial_certification_principal` | 2026-09 | state | Academic years ending 2016–2026 |
| `educator_state_initial_certification_superintendent` | 2026-09 | state | Academic years ending 2016–2026 |
| `educator_state_initial_certification_teacher` | 2026-09 | state | Academic years ending 2016–2026 |
| `educator_state_new_hires` | 2026-04 | state | SY ending 2016–2026 |
| `educator_state_retention_by_demographic_group` | 2026-04 | state | First-teaching cohorts ending 2014–2025 |
| `educator_state_retention_by_gender` | 2026-04 | state | First-teaching cohorts ending 2014–2025 |
| `educator_state_retention_by_preparation_route` | 2026-04 | state | First-teaching cohorts ending 2014–2017, 2019–2025 |
| `facility_usage_submission` | 2026 | district | SY ending 2026 |
| `facility_usage` | 2026 | district | SY ending 2026 |
| `first_financial_accountability_district` | 2026 | district | FY ending 2025 |
| `first_financial_accountability_university_charter` | 2026 | district | FY ending 2025 |
| `frc_per_pupil_expenditure` | 2025 | campus, district, state | FY ending 2025 |
| `peims_college_credit` | 2022–2026 | district, region, state | SY ending 2022–2026 |
| `peims_financial_single_file_budgets` | 2026 | district | FY ending 2026 |
| `peims_special_education_by_primary_disability` | 2013–2026 | district, state | SY ending 2013–2026 |
| `peims_staff_fte_and_salary` | 2019–2026 | district | SY ending 2019–2026 |
| `peims_staff_fte_and_salary` | 2013–2026 | state | SY ending 2013–2026 |
| `peims_student_enrollment_by_ethnicity` | 2019–2026 | campus, district, state | SY ending 2019–2026 |
| `peims_student_programs_and_special_populations` | 2023–2026 | campus, district, state | SY ending 2023–2026 |
| `peims_teacher_course_enrollment` | 2013–2026 | district, state | SY ending 2013–2026 |
| `region_tapr_staar_all_grades` | 2026 | region | SY ending 2026 |
| `region_tapr_staar_end_of_course` | 2024–2026 | region | SY ending 2024–2026 |
| `region_tapr_staff_profile` | 2025 | region | SY ending 2025 |
| `region_tapr_student_profile` | 2025 | region | SY ending 2024–2025 |
| `spp_timely_initial_evaluation` | 2025–2026 | district | SY ending 2024–2025 |
| `state_tapr_staar_all_grades` | 2026 | state | SY ending 2026 |
| `state_tapr_staar_end_of_course` | 2024–2026 | state | SY ending 2024–2026 |
| `state_tapr_staff_profile` | 2025 | state | SY ending 2025 |
| `summary_of_finances_attendance` | 47422 | district | SY ending 2026 |
| `summary_of_finances_dpe` | 47422 Near Final | district-keyed LEA recipients | Field-specific school/tax years and as-of dates; see contract |
| `tprs_attendance_chronic_absence` | 2025 | region, state | SY ending 2024–2025 |
| `tprs_class_size` | 2026 | campus, district, state | SY ending 2026 |
| `tprs_student_continuity` | 2026 | campus, district, state | Baseline SY ending 2025; return/processing fall 2025 |
| `tprs_student_membership_enrollment_reports` | 2026 | campus, district, state | SY ending 2026 |

State means the directly published Texas scope, not a sum of ESCs or districts.
Regional, county and district reporting populations remain separate. Historical
source rosters can contain entities absent from the current entity snapshot;
source rows remain queryable without inventing historical entity attributes.

## Base entity inputs

These declared source periods describe the entity build. They do not assign
one common measurement year to every attached field or make a current address,
name or geometry historically accurate.

| Base dataset | Declared source period | Build status |
| --- | --- | --- |
| `accountability` | Calendar 2026 | loaded |
| `all_districts_reference` | Unspecified | loaded |
| `all_districts_reference2` | Unspecified | loaded |
| `campus_peims_financials` | FY ending 2024 | loaded |
| `campus_planned_closures` | As of 2025-10-03 | loaded |
| `campus_tapr_historical_enrollment` | SY ending 2015 | loaded |
| `campus_tapr_student_staff_profile` | SY ending 2025 | loaded |
| `campus_transfer_reports` | SY ending 2026 | loaded |
| `campuses` | SY ending 2025 | loaded |
| `charter_reference` | As of 2025-09-30 | loaded |
| `district_tapr_student_staff_profile` | SY ending 2025 | loaded |
| `districts` | SY ending 2026 | loaded |
| `tefa` | SY ending 2027 | loaded |

## Limits and remaining integration work

- The current Student counts and five composition percentages have measurement year 2025–26. Lagged or differently
  defined fields are not silently advanced to that year. Supported frozen
  168-, 174- and 179-field comparison profiles remain separate; mixed or unknown
  profiles fail clearly. No cross-year scientific comparability is inferred.
- The 2017–18 preparation-route retention cohort stays held: the rules example
  and workbook disagree. Five route rows are excluded. Other admitted views
  do not repair or reconstruct them. Unreported follow-up horizons remain
  missing, with no older-value fallback.
- Certification detail remains unqualified because of repeated keys, absent
  detail and sum differences. Only published All/All/All totals are admitted.
  Two teacher totals differ from archived PDF editions; the latest publisher
  totals are used with explicit edition notes. Their cause remains unresolved.
- Initial certification, hiring, staff snapshots, continuous retention and
  turnover concern different populations and periods. Cohort ratios are
  publisher fractions. Overlapping views and certification roles are not additive.
- Other registered families still need source-specific acquisition, dictionaries,
  semantic qualification and normal-release integration. Priority gaps include
  further TAPR/TPRS and STAAR domains; earlier transfer years; PEIMS financial
  actuals; and broader student, readiness and finance history. Registration is
  not acquisition, qualification, release or query availability.
- Additional TPEIR and TEA research products need their own availability and
  source-contract review. This configuration/release comparison does not
  inventory unregistered producers or establish their absence from the project.
- Native primary-disability coverage reaches 2012–13 for district/Texas, but
  this is not comprehensive 2012–13 coverage for every source and scope.
  County/region integration and source-specific campus absences remain explicit.
- Preserve suppression, missingness, invalid, not-applicable and observed-zero
  states. No protected counts are reconstructed; no harmonization or scientific
  comparability is inferred from matching column names.

Use [source coverage](source-coverage.md), [enrichment and year selection](enrichment.md),
[release selection](data-releases.md), [educator workforce](educator-workforce-sources.md),
[PEIMS staff](peims-staff.md), [finance](financial-sources.md) and
[facility usage](facility-usage.md) for source-specific contracts.

## Reproduce and inspect

```sh
uv run --locked python scripts/source_inventory.py \
  --release "$TEADATA_MANIFEST_PATH" --sha256 "$TEADATA_MANIFEST_SHA256" \
  --config teadata/teadata_sources.yaml --include-variables
```

This reads verified catalog metadata, not observation leaves. The default index
mode omits variable metadata. `DataRelease.catalog_inventory()` and
`catalog_diff()` expose the same discoverable inventory without hydrating the
corpus. Native attachments join the normal pipeline through exact manifest pins;
there is no second enrichment or per-dataset entity-property list.

Without a year, observations resolve the latest eligible period once for the
requested dataset/scope in the selected release. Explicit unavailable years
raise helpful errors; they never substitute another year. Multi-row sources
retain their publisher keys and require selection instead of arbitrary scalars.
Year-aware state observations use the normal release; entity Query observations
remain district/campus scoped. This release does not introduce a State entity.

Frozen prior inventories remain available for
[0.0.154](source-integration-status-0.0.154.json),
[0.0.155](source-integration-status-0.0.155.json),
[0.0.157](source-integration-status-0.0.157.json),
[0.0.158](source-integration-status-0.0.158.json),
[0.0.159](source-integration-status-0.0.159.json),
[0.0.160](source-integration-status-0.0.160.json),
[0.0.161](source-integration-status-0.0.161.json),
[0.0.163](source-integration-status-0.0.163.json),
[0.0.175](source-integration-status-0.0.175.json), and
[0.0.185](source-integration-status-0.0.185.json).
The previous narrative is preserved in Git history; those historical counts
must not be reused as the current selected release.
