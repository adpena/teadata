# Source integration status

The packaged selector identifies **normal data release 0.0.165**. Software, data
publication and application adoption are separate; an explicitly pinned older
release does not change when the reader is upgraded. The [machine inventory](source-integration-status-0.0.165.json)
was regenerated from the actual selected release on November 17, 2026. It lists
126 registered IDs, 278 configured editions, 13 base inputs and 20 native
attachments covering 31 dataset IDs and 277 edition/scope keys. The 85 registered
source-only IDs not attached by name are not 85 missing measures or proof that
publisher data are unavailable.

- Manifest SHA-256: `e32c15b1f782f7a10f97b2158c4560a0c161d2c47fec5448987a9d55fe4c5685`.
- Build: `92d82a7b54f4b3f6b8ea9850e31897a746f456d7542cf8e74396d0dc6fab636c`.
- Actual normal producer: `e24277068834b5a6279764e643c1e46ab6f8194a`.
- Processing identity: `7c2b10997f6797801a5c93c412a9d79ec4472195029095ae7e33202737d3f3f7`.

## What this release adds

The TPRS Student projection now has **179 qualified fields**. It preserves all
174 existing fields and appends five disability-composition Membership
percentages. Coverage remains 9,105 reporting campuses, 1,202 reporting districts
and the direct Texas report in 2025–26: 10,308 rows across 25 editions. The new
five fields add 51,540 observations, including publisher value statuses. No
measurement year or additional TPRS domain is added by this release.

Their denominator is **Total Students with Disabilities in Membership**, not all
students. It is declared in source metadata, variable/header discovery and the
shared DataFrame/CSV/Excel provenance. Selecting a percentage does not read its
denominator's observations. Excel adds a Denominator column to the Provenance
field table only when needed. Broad disability group names follow the glossary:
Intellectual Disabilities includes learning disability, developmental delay and
traumatic brain injury, not only the specific intellectual-disability category.
Texas disability Membership 901,935 and Special Education program Membership
901,932 remain distinct source observations.

The normal producer reused all retained acquisitions. Every original 174 field,
raw token and status matched across all 10,308 rows. One shared scientific gate
replayed the complete new 179-field projection, its locators and raw-HOLD audit.
Three genuine HTML/XLSX pairs also matched all 1,298 inventoried data cells
before Class Size. Physical blank XML cells are distinguished from absent cells
and zero. These are frozen-source checks, not fresh live reacquisition of every
TEA record. Original download dates remain in provenance.

Bounded actual-parent checks verified all nine parent artifacts, replacement-only
wiring, metadata-only discovery/year resolution, actual district/campus dot
access, and all 179 district/Texas comparisons. The five new percentages and
real campus masked, zero and missing cases passed DataFrame, CSV and Excel
readback with the same shared provenance. Older 174-field descriptors, source
meaning, description hash, CSV bytes and XLSX data remain unchanged. A new
reconstruction truthfully cites software 0.0.165; frozen saved files still cite
0.0.164. The retained saved selection had no supported public query-replay
recipe, so reconstruction is not described as successful `replay_table` use.

The other 19 native attachments and 13 base inputs are unchanged and retain
their earlier qualifications. They were not scientifically replayed merely to
repeat matching evidence. Class-size source bytes are unchanged. The full-report
inventory exposes unsupported cells without turning them into another dataset:
blank disability Enrollment/total-percent cells, undocumented CTE dots and
unreviewed lagged/tail measures remain held. See [Student source notes](attendance-sources.md)
and [the SOF documentation/formula question](financial-sources.md#sof-202526-dpe-source-accounting-and-an-unresolved-formula-question).

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
