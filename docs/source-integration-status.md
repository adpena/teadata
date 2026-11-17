# Source integration status — November 17, 2026 (UTC)

The packaged selector is pinned to **normal data release 0.0.164**. Software and data
versions are separate identities: installing a newer reader does not silently
replace an explicitly pinned historical release. This is a dated release
inventory, not a live TEA feed. Qualification, public data publication, software
publication and application adoption are separate gates.

The [machine inventory](source-integration-status-0.0.164.json) was generated
from the exact retained manifest and verified metadata. It records **13 base
inputs, 20 native attachments, 31 native dataset IDs and 277 edition/scope
keys**. The configuration registers 126 IDs and 278 report editions; **85
registered source-only IDs remain unattached by name**. Those are integration
gaps, not 85 missing measures or evidence that the publisher has no data.

- Manifest SHA-256: `b2a46d2e690f4aa36b97f27788b4d5eeb7ed10aa322cbb884a54c1f8598ee64d`.
- Build: `69faad22d961a1ba428916d8f240814ae1ab59d0c04b5012e1adb2dae9cbd021`.
- Actual producer: `c2d4cfe3bf732e000d447caca706b5ca92c8f4ba`.
- Processing identity: `12afcbc0662288cb0c2660ef4a3bfa81cdbe9f411d8367c41c3bdbf1c4775054`.

## What this release adds

The TPRS Student attachment now exposes **174 qualified fields**, adding six
Membership disability counts to the original 168-field projection. The 25
editions cover 23 campus partitions, 1,202 reporting districts and the direct
Texas observation: **9,105 campus rows and 10,308 rows overall**. Six new fields
therefore add 61,848 observations, with raw tokens and value statuses retained.
This is the reviewed current Student Information projection, not every TPRS
domain or every raw report cell. The 13 base inputs and other 19 native
attachments are unchanged; their earlier source qualifications are reused.

These counts use TEA's broad disability groupings. Intellectual Disabilities
includes learning disability, developmental delay and traumatic brain injury,
not only the specific intellectual-disability category. Disability percentages
and Enrollment variants remain held. Texas reports 901,935 in total disability
Membership and 901,932 in Special Education program Membership; both source
observations remain distinct. No suppressed count is reconstructed or forced
to reconcile. See [student definitions and source holds](attendance-sources.md)
and [same-profile statewide comparisons](querying.md#compare-a-published-measure-with-texas).

The retained producer census reproduced all original 168 values, raw tokens
and statuses for all 10,308 rows. The shared scientific source gate also
replayed all 10,308 original reports against the complete 174-field source,
checking raw/status/locator equality and the raw-HOLD audit. Historical
168-field Texas replay and the named workbook witnesses passed under both
captured contracts. These checks do not cover every earlier standalone
source bundle. Three named publisher HTML/XLSX witness pairs
support the reviewed field interpretation. These are retained-source checks,
not a fresh live re-acquisition of every record. Source download times remain
available in provenance; no finality is inferred from the report edition.

Bounded checks against the actual new normal parent verified all nine parent
artifact hashes/build bindings, metadata-only discovery and year availability,
actual entity identities, district/campus `.tprs` access, all 174 district/Texas
comparisons and an older pinned 168-field comparison. Six new counts round-trip
through DataFrame, CSV and Excel with the same canonical provenance. Real
campus masked, observed-zero and absent-grade states remain explicit in the
new parent readback; the corresponding three-format export receipts are
reused for those unchanged native rows. The existing class-size route and
its source bytes are unchanged.
The unchanged native families and source census were not replayed merely to
repeat their existing receipts. This inventory itself establishes metadata
coverage only, not semantic or live-source verification.

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

- The new Student counts have measurement year 2025–26. Lagged or differently
  defined fields are not silently advanced to that year. Supported frozen
  168-field and 174-field comparison profiles remain separate; mixed or unknown
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
