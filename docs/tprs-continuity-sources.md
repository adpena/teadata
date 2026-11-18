# TPRS student continuity cohorts

`tprs_student_continuity`, publisher edition 2026, interprets selected local
observations in the same retained 2025–26 Student Information reports used by
the Membership/Enrollment and class-size projections. It does not reacquire
reports, duplicate current Membership/Enrollment measures, or alter the
historical 168/174/179-field contracts.

The measurement period names the **2024–25 baseline cohort or prior grade
year**. Return and processing occur in fall 2025 during 2025–26; underreporting
uses the glossary's school-start window ending September 26, 2025. Edition
2026 and measurement year ending 2025 are separate selectors.

The family has 24 district/state fields and 22 campus fields:

- `attrition_2024_25_count` and `attrition_2024_25_percent`: the published
  adjusted fall-2024 cohort that did not return to the same campus in fall
  2025. Source exclusions concern next-grade availability and inactive
  campuses; retained students, including grade 12, remain. District/state
  rates aggregate eligible campus numerators and denominators. These are not
  dropout measures.
- `retention_2024_25_{non_special_education,special_education}_{kindergarten,grade_1,...,grade_9}_percent`:
  published same-grade return rates, separated by the source populations.
  The denominator is prior-year students who return the following year or
  graduate. No retained-student or denominator counts are inferred.
- `underreported_students_2024_25_count` and
  `underreported_students_2024_25_percent`: district/state leaver-reporting
  data quality for prior-year grades 7–12 students. The exact TxCHSE, prior
  graduate, other Texas district and return-window exclusions are in the
  pinned methodology. These fields are unavailable in the campus schema.

The complete source definitions, population wording, local physical columns,
source pins and literal original HTML/XLSX goldens are in
`teadata/methodologies/tprs-student-continuity-2026.json`. Supported
MetricDefinition properties carry field, period, unit and denominator.
Descriptions remain in that methodology and source notes. Percentage values
are percentage points, with a one-decimal roundtrip to the original token.
Count values are integral; original comma formatting remains in raw support.

`*` remains suppressed and `-` means the publisher's no-students/not-applicable
state. A documented `?` is null/invalid with its original reason. Unknown
tokens and context-inappropriate `**` fail closed. Published blanks, absent
grade rows and omitted retention sections remain distinct missing states;
they never become zeros or inferred eligibility exclusions. Every published
cell retains its raw token and one-based parsed HTML row/column. Missing rows
or sections have no invented coordinates.

The existing structural Student inventory supplies tail boundaries and cell
coordinates. Continuity is a separate versioned interpretation over that
inventory. Mobility, disciplinary placements, CTE dots, blank disability
Enrollment and blank total-disability percent remain outside this family.

The original campus acquisition closure is 16,442,743 canonical bytes, above
the unchanged 4 MiB native-edition metadata cap. The child therefore preserves
the original bounded Student campus partitions, plus district and direct
Texas editions. A continuity campus key keeps the Student source revision and
uses `table="campus-" + student.identity`; one shared typed translator and the
existing immutable Student index supply entity/series routing. No provenance
is removed and no source partitions are rebalanced. Standalone multi-partition
children require exact edition keys; automatic entity routing uses a normal
parent with the matching Student attachment.

After exact scientific admission and normal parent composition, the existing
catalog, entity dataset namespaces, `.tprs`, query and export paths expose the
family. For this edition, `entity.tprs.at(2025)` and `.at("2024-25")` select
continuity, while `.at(2026)` must report that its admitted year is 2025.
Select current Membership/Enrollment and continuity separately because a
namespace selection resolves one native family. Explicit publisher-edition
access continues to use `release.for_entity(entity, edition="2026")`.
Direct state access uses
`release.observations("tprs_student_continuity", field, scope="state", school_year=2025)`;
there is no state entity facade or inferred statewide aggregation.

Source-only configuration registration, a sealed local native candidate,
complete original-source replay and literal goldens, exact scientific
dispatch admission, normal parent integration, packaged default activation
and downstream application adoption are separate gates. The producer emits
`release_activation:false` and `field_correctness_qualified:false`.
`qualify_continuity` requires an explicit exact child pin and reports its
literal source replay separately from independent review and publication.
It verifies all retained report requests, local tokens, normalized values,
statuses, missing reasons, physical coordinates, scope-specific fields,
original partition selection, roster closure and the three independently
requested workbook witnesses. It does not manufacture expected values from
candidate output. The existing full-report census is layout evidence rather
than numerical admission of this projection.
