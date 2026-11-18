# Source integration status

The packaged selector identifies **normal data release 0.0.166**. This is a
frozen release, not a live TEA feed. Software publication and application adoption
are separate gates. Historical selections retain their exact release.
The [machine inventory](source-integration-status-0.0.166.json) was generated from
the qualified normal parent and current configuration on November 18, 2026:
127 registered IDs, 279 configured editions,
13 base inputs and 21 native attachments covering 32 dataset IDs and
302 edition/scope keys. Registration does not establish
acquisition, qualification or query access; the source-only IDs not attached
by name are not a count of missing measures.

- Manifest SHA-256: `e34cce994f2db88b5a2dd0eb3d6950412d0bd7439603675018503844f4da471a`.
- Build: `63290448a80ffa9fd959f4217dcb2015f882d51e426bde264b95e292486c6514`.
- Actual normal producer: `9513698` (full commit retained in the build record).
- Processing identity: `81f41daec4f9ca09a17f4d720b576be4994c6e9d0db7f2e97f401eafc2fb9ebb`.

## What this release adds

The new `tprs_student_continuity` family supplies 22 campus and 24 district/state
fields for published attrition, grade retention and district/state underreporting.
Its report edition is **2026**; the measurement/baseline year is **2024–25**,
with return/processing observations in fall 2025. It does not replace current
2025–26 Membership or Enrollment, and it is not a dropout or individual-student
tracking dataset. Campus underreporting is unavailable. Retention populations
remain separate; counts and denominators are not inferred from percentages.

Coverage is 9,105 reporting campuses, 1,202 reporting districts and the direct
Texas report, preserving all 25 original source partitions. The 229,182 logical
field slots include observed values and explicit missing, suppressed and not
applicable states. Missing grade rows/sections have null source coordinates and
factual absence reasons, never invented eligibility explanations or zeros.
Names preserve `2024_25`; cross-year harmonized aliases need separate review.
See [the definitions, populations and scope notes](tprs-continuity-sources.md).

The complete retained continuity projection passed original-source raw/value/
status/locator replay across all 10,308 reports. Three independently requested
HTML/XLSX pairs supplied 62 published local golden cells and eight absent-grade
bindings. Actual source comparisons checked 24 district and 22 campus metrics
against the direct Texas report and frozen raw goldens. Comparisons preserve
count units and percentage points and require the same exact qualified child.
These are frozen-source checks; no fresh acquisition of all TEA rows is claimed.
Original download times remain in the acquisition receipts.

The qualified normal parent passed its nine artifact integrity/build/schema
checks and bounded actual-entity discovery, dot-path/year, source-golden,
statewide-comparison and DataFrame/CSV/Excel provenance readback. Exact source
replay was reused by unchanged child identity; the other 20 native attachments
and all 13 base inputs retain their previous qualifications. Their source
families were not replayed merely to repeat matching evidence. Current 179-field
Student and Class Size sources remain unchanged. Application cache/deployment
adoption is recorded in the application rollout, not inferred from this release.

The full-report inventory separately accounts for 4,709,972 physical data cells
before Class Size or the actual report end across 10,308 reports. That is layout evidence, not qualification of every
cell. Remaining mobility/disciplinary period conflicts, CTE dots and blank
disability Enrollment/total-percent observations remain held. The four SOF DPE
sheets have exact raw fidelity and accepted dictionary bindings; their additional
590 numeric semantics remain held. No general adjusted refined ADA was filled
from a Support Staff Retention field. See [Student source notes](attendance-sources.md)
and [the SOF formula clarification](financial-sources.md#sof-202526-dpe-source-accounting-and-an-unresolved-formula-question).

## Qualified native coverage

The table lists exact publisher/report editions and **known bound field
periods**, not a promise that every raw column has a qualified metric. A school
year ending 2026 means 2025–26. Multiple periods in a source can describe lagged
fields. `Unspecified` stays unresolved; a stable name does not establish
comparability. Inspect the machine inventory or catalog field metadata for
individual definitions, units, population, revisions and source links.

| Native dataset | Report editions | Scopes | Bound field periods |
| --- | --- | --- | --- |
| `campus_tapr_staff_profile` | 2025 | campus | SY ending 2025 |
| `district_tapr_staff_profile` | 2025 | district | SY ending 2025 |
| `district_transfer_reports` | 2024–2026 | district | SY ending 2024–2026 |
| `educator_district_new_hires` | 5 | district | SY ending 2010–2026 |
| `educator_district_new_hires_by_epp` | 5 | district | SY ending 2010–2026 |
| `educator_state_initial_certification_principal` | 2026-09 | state | Academic years ending 2016–2026 |
| `educator_state_initial_certification_superintendent` | 2026-09 | state | Academic years ending 2016–2026 |
| `educator_state_initial_certification_teacher` | 2026-09 | state | Academic years ending 2016–2026 |
| `educator_state_new_hires` | 2026-04 | state | SY ending 2016–2026 |
| `educator_state_retention_by_demographic_group` | 2026-04 | state | First-teaching cohorts ending 2014–2025 |
| `educator_state_retention_by_gender` | 2026-04 | state | First-teaching cohorts ending 2014–2025 |
| `educator_state_retention_by_preparation_route` | 2026-04 | state | First-teaching cohorts ending 2014–2017, 2019–2025 |
| `facility_usage` | 2026 | district | SY ending 2026 |
| `facility_usage_submission` | 2026 | district | SY ending 2026 |
| `first_financial_accountability_district` | 2026 | district | FY ending 2025 |
| `first_financial_accountability_university_charter` | 2026 | district | FY ending 2025 |
| `frc_per_pupil_expenditure` | 2025 | campus, district, state | FY ending 2025 |
| `peims_college_credit` | 2022–2026 | district, region, state | SY ending 2022–2026 |
| `peims_financial_single_file_budgets` | 2026 | district | FY ending 2026 |
| `peims_special_education_by_primary_disability` | 2013–2026 | district, state | SY ending 2013–2026 |
| `peims_staff_fte_and_salary` | 2023–2026 | district, state | SY ending 2023–2026 |
| `peims_student_enrollment_by_ethnicity` | 2019–2026 | campus, district, state | SY ending 2019–2026 |
| `peims_student_programs_and_special_populations` | 2023–2026 | campus, district, state | SY ending 2023–2026 |
| `region_tapr_staff_profile` | 2025 | region | SY ending 2025 |
| `region_tapr_student_profile` | 2025 | region | SY ending 2024–2025 |
| `spp_timely_initial_evaluation` | 2025–2026 | district | SY ending 2024–2025 |
| `state_tapr_staff_profile` | 2025 | state | SY ending 2025 |
| `summary_of_finances_attendance` | 47422 | district | SY ending 2026 |
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
[0.0.161](source-integration-status-0.0.161.json) and
[0.0.163](source-integration-status-0.0.163.json).
The previous narrative is preserved in Git history; those historical counts
must not be reused as the current selected release.
