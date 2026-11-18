# Source integration status

The packaged selector identifies **normal data release 0.0.167**. This is a
frozen analysis release, not a live TEA feed. Software publication and
application deployment are separate gates; historical requests keep their pins.
The [machine inventory](source-integration-status-0.0.167.json), generated from
the qualified normal parent on November 18, 2026, records
127 registered IDs, 279 configured editions,
13 base inputs and 22 native attachments covering 33 dataset IDs and
306 edition/scope keys. Registration is not
acquisition, semantic qualification or query access.

- Manifest: `ecc89a8dea775defa5c3e81911d059fe9eb6af09d0ccb65112a5f1be5e43573a`.
- Build: `df8c457b82a6029591b4d835fff70a3ddc2b64e2b70389d72754aeb421c8c421`.
- Processing identity: `3804ddc10606b294b837636e92138b90ad788bd04e1643b5e886860d33375cd0`.

## What this release adds

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
[0.0.161](source-integration-status-0.0.161.json) and
[0.0.163](source-integration-status-0.0.163.json).
The previous narrative is preserved in Git history; those historical counts
must not be reused as the current selected release.
