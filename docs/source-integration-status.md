# Source integration status — November 12, 2026

This is a dated inventory of the tracked source configuration at code commit
`2705e2d85158fd22fd3f0614fd8c89bcb60d6092`, compared with the app-selected
[v0.0.148 parent release](https://github.com/adpena/teadata/releases/download/v0.0.148/release.json).
Software 0.0.150 is separate from that data release. The installed package's
configuration matched the checkout's SHA-256. No raw data or qualification
suite was reacquired or rerun for this metadata audit.

The configuration contains **124 dataset IDs and 276 editions**. Of these,
**111 IDs are preparation/source-only contracts**; **105 are absent as named
native sources from the selected parent**. These include scope and domain
variants, not 105 independent TEA systems. Some measures already occur in the
combined TAPR entity profiles. Absence of a separate source ID does not establish
that every corresponding measure is missing. Conversely, registration does not
establish acquisition, interpretation, qualification or availability in a release.

The parent has 13 base inputs and four attached bundles exposing seven native
dataset IDs. Six of those IDs are in the preparation-only configuration; the
seventh, class size, is derived from retained TPRS reports. The complete
[machine-readable inventory](source-integration-status.json) records exact
registered periods, source flags and attached edition/scope identities. Its
configuration digest and parent hash pin this comparison; it is not a live feed.

## Consequential integration gaps

| Group not attached under its registered source IDs | Registered edition keys | What needs finishing |
| --- | --- | --- |
| PEIMS staff FTE and salary | 2023–2026 | Attach the reviewed district/Texas leaves and exercise generic entity, filtered-role and export access. Four years of retained parity and eight publisher-report witnesses exist; that is not release activation. |
| PEIMS primary disability | 2013–2026 | Integrate the detailed district/Texas series, retaining suppression and classification transitions. Existing TAPR special education figures remain a distinct population. |
| PEIMS enrollment by ethnicity; programs and special populations | 2019–2026; 2023–2026 | Complete year/scope-specific qualification and release integration; preserve Enrollment versus Membership and all publisher tokens. |
| PEIMS teacher/course enrollment; college credit | 2023–2026; 2022–2026 | Preserve the course/program row dimensions and denominators through normal lazy access. |
| PEIMS single-file financial actuals; budgets | 2013–2025; 2026 | Resolve field/unit qualification and formula changes; retain fiscal years, sectors, actual/budget basis and account grain. These are not the already attached FRC per-pupil table. |
| District transfer reports | 2013–2026 | Finish the direct district-report path and historical classification holds. The entity release already has separate campus transfer edges. |
| STAAR domains across campus/district/region/Texas | 2024–2026, varying by domain | Complete qualified source binding and release delivery for performance, participation, growth, tested grades, EOC and bilingual/ESL domains. Edition is not necessarily assessment year. |
| Graduation, dropout, readiness and college-readiness components | 2024/2025, varying by source | Preserve annual versus longitudinal populations, graduation exclusions, overlapping criteria and actual cohorts. |
| Educator hiring, retention and initial certification | Registered 2026 editions containing source-specific historical series | Deliver the existing multi-row role, route, EPP, demographic and cohort contracts; do not interpret edition 2026 as a single observation year. |
| SPP timely initial evaluation; RDA special education Indicator 1 | 2024–2025; 2025 | Integrate the exact reviewed scope/period variants, preserving pooled versus single-year rates. |
| Separate TAPR student, regional/state staff and reference profiles | Mostly 2024/2025 | Reconcile overlap with existing base fields and attach qualified additional periods/scopes without duplication. |

The separate PEIMS Superintendent Salary product is **not registered** in this
configuration. It needs its own contract: district ID is not a unique person-row
key, numeric zero can mean salary not submitted, and the latest file contains
TEA's explicit Harlandale misreporting notice. Its semantics cannot be inherited
from Staff FTE and Salary or initial-certification counts.

## What is attached already

| Native dataset | Attached edition and geography |
| --- | --- |
| TPRS Student Information/Membership/Enrollment | 2026: district, direct Texas and 23 campus partitions |
| TAPR Staff Information | 2025: campus and district, with selected qualified teacher measures |
| FRC per-pupil expenditure | 2025 report edition: campus, district and direct Texas; inspect fiscal period per measure |
| School FIRST | 2026 district and university-charter editions; financial accountability is distinct from academic accountability |
| TPRS class size | 2026: district, direct Texas and 23 campus partitions |

This table establishes attachment from manifest/catalog metadata. It does not
claim every source column has qualified semantics or that every direct entity
attribute/query combination was exercised in this audit.

## Default and selected releases

The checked-in source checkout's `teadata/.cache/release.json` has build
`973bfac9a388ccf22670ffe64ced1b29adeea6060991c214f84885f3c8d8f8b7`, 13 base
inputs and no native attachments. It is not the app's selected parent. The
installed 0.0.150 wheel inspected on Mini had **no embedded default manifest**;
do not infer that it automatically selects v0.0.148. An explicit configured
release controls what is available, and explicit unavailable years must fail.

## Completion rule and integration direction

Count a source as delivered only for the exact released editions/scopes whose
qualified fields can be discovered and queried through the normal catalog-bound
access path, with missingness, periods and export provenance preserved. Retained
source tables and qualification evidence are useful intermediate work; they are
not completion of entity integration.

Reuse the existing Config, EditionKey, ArtifactRef, FieldBinding and lazy metadata
mapping contracts. The normal pipeline should consume a qualified source's
catalog and expose its fields automatically. Avoid adding another manual list
of entity properties, query operators or export special cases per dataset.
Multi-row sources retain their dimensions and explicit selection; automatic
integration must not silently aggregate or overwrite scalar attributes.

At the audited code revision, normal builds deliberately excluded `source_only`
inputs unless separately attached, and the parent-release verifier dispatched
only four native groups while standalone qualification supported more reviewed
families. The subsequent verifier consolidation makes both paths select the
same reviewed source plan and binds receipt evidence separately to each child
manifest. It adds no attachments, published data years or entity fields, so the
inventory counts above remain unchanged. Normal release composition and a shared
lazy entity resolver still need finishing. Preserve publisher evidence and the
approval boundary; an unreviewed registration must not become an automatic
publication merely to clear this inventory.

Normal builds now accept an exact manifest location and SHA-256 and derive the
complete source descriptor from its catalog index. This removes manual copying
of build IDs and edition lists; it does not change this frozen inventory or
activate any source in an existing release. The offline builder checks structural
integrity before constructing entities. Semantic qualification remains at the
final shared release gate, and publication is a separate decision. See
[normal source composition](enrichment.md#observations-across-school-years).

Standalone/programmatic producers outside the default YAML, such as additional
finance and AskTED workflows, require a separate coverage check. The counts here
do not establish their absence from the repository or their publisher availability.

## Reproduce the metadata comparison

Use the inspected code/configuration and a retained v0.0.148 manifest with its
assets. This reads source contracts and attachment descriptors, not observations:

```python
from teadata.release import DataRelease
from teadata.teadata_config import load_config

config = load_config()
configured = config.data_sources | config.spatial
source_only = {
    name for section in ("data_sources", "spatial")
    for name, editions in getattr(config, section).items()
    if any(config.is_source_only(name, year, section=section)
           for year in editions.available_years())
}
release = DataRelease("/path/to/retained/release.json", offline=True,
    expected_sha256="e485c94e68242a466533b5b94203ca91834f1149de1b49d0ea70c6b52be8a308")
attached = {key.dataset for ref in release.catalog.native_sources for key in ref.editions}
print(len(configured), sum(len(item.available_years()) for item in configured.values()))
print(len(source_only), len(source_only - attached))
```

The expected counts for this frozen comparison are `124, 276` and `111, 105`.
The machine inventory's configuration hash must match before reusing those
numbers. Recompute them after changing configuration or the selected release.

## Complete preparation-only dataset inventory

Edition keys below describe registered contracts, not a promise of qualified
data for every cell. “Not attached” does not mean no standalone candidate exists.

| Dataset ID | Registered editions | Present in selected parent |
| --- | --- | --- |
| `accountability_research_annual_dropout` | 2025 | Not attached |
| `accountability_research_four_year_graduation` | 2025 | Not attached |
| `campus_tapr_5_year_extended_longitudinal_rate` | 2024 | Not attached |
| `campus_tapr_6_year_extended_longitudinal_rate` | 2024 | Not attached |
| `campus_tapr_annual_rhsp_dap_fhsp_e_fhsp_dla_texas_first_dla` | 2024 | Not attached |
| `campus_tapr_college_career_military_readiness` | 2025 | Not attached |
| `campus_tapr_college_readiness_components` | 2025 | Not attached |
| `campus_tapr_dropout_attendance` | 2024, 2025 | Not attached |
| `campus_tapr_four_year_longitudinal_rate` | 2024, 2025 | Not attached |
| `campus_tapr_graduation_profile` | 2024, 2025 | Not attached |
| `campus_tapr_longitudinal_rhsp_dap_fhsp_e_fhsp_dla_texas_first_dla` | 2024 | Not attached |
| `campus_tapr_reference` | 2024 | Not attached |
| `campus_tapr_staar_all_grades` | 2025, 2026 | Not attached |
| `campus_tapr_staar_bilingual_esl_growth` | 2026 | Not attached |
| `campus_tapr_staar_bilingual_esl_performance` | 2026 | Not attached |
| `campus_tapr_staar_end_of_course` | 2024–2026 | Not attached |
| `campus_tapr_staar_grades_3_to_8` | 2026 | Not attached |
| `campus_tapr_staar_participation` | 2024–2026 | Not attached |
| `campus_tapr_staar_school_progress` | 2024–2026 | Not attached |
| `campus_tapr_staar_tested_grade_3` | 2024–2026 | Not attached |
| `campus_tapr_staar_tested_grade_4` | 2024–2026 | Not attached |
| `campus_tapr_staar_tested_grade_5` | 2024–2026 | Not attached |
| `campus_tapr_staar_tested_grade_6` | 2024–2026 | Not attached |
| `campus_tapr_staar_tested_grade_7` | 2024–2026 | Not attached |
| `campus_tapr_staar_tested_grade_8` | 2024–2026 | Not attached |
| `campus_tapr_staff_profile` | 2024, 2025 | 2025: campus |
| `campus_tapr_student_profile` | 2024, 2025 | Not attached |
| `district_tapr_5_year_extended_longitudinal_rate` | 2024 | Not attached |
| `district_tapr_6_year_extended_longitudinal_rate` | 2024 | Not attached |
| `district_tapr_annual_rhsp_dap_fhsp_e_fhsp_dla_texas_first_dla` | 2024 | Not attached |
| `district_tapr_college_career_military_readiness` | 2025 | Not attached |
| `district_tapr_college_readiness_components` | 2025 | Not attached |
| `district_tapr_dropout_attendance` | 2024, 2025 | Not attached |
| `district_tapr_four_year_longitudinal_rate` | 2024, 2025 | Not attached |
| `district_tapr_graduation_profile` | 2024, 2025 | Not attached |
| `district_tapr_longitudinal_rhsp_dap_fhsp_e_fhsp_dla_texas_first_dla` | 2024 | Not attached |
| `district_tapr_reference` | 2024 | Not attached |
| `district_tapr_staar_all_grades` | 2025, 2026 | Not attached |
| `district_tapr_staar_bilingual_esl_growth` | 2026 | Not attached |
| `district_tapr_staar_bilingual_esl_performance` | 2026 | Not attached |
| `district_tapr_staar_end_of_course` | 2024–2026 | Not attached |
| `district_tapr_staar_grades_3_to_8` | 2026 | Not attached |
| `district_tapr_staar_participation` | 2024–2026 | Not attached |
| `district_tapr_staar_school_progress` | 2024–2026 | Not attached |
| `district_tapr_staar_tested_grade_3` | 2024–2026 | Not attached |
| `district_tapr_staar_tested_grade_4` | 2024–2026 | Not attached |
| `district_tapr_staar_tested_grade_5` | 2024–2026 | Not attached |
| `district_tapr_staar_tested_grade_6` | 2024–2026 | Not attached |
| `district_tapr_staar_tested_grade_7` | 2024–2026 | Not attached |
| `district_tapr_staar_tested_grade_8` | 2024–2026 | Not attached |
| `district_tapr_staff_profile` | 2024, 2025 | 2025: district |
| `district_tapr_student_profile` | 2024, 2025 | Not attached |
| `district_transfer_reports` | 2013–2026 | Not attached |
| `educator_district_new_hires` | 2026 | Not attached |
| `educator_district_new_hires_by_epp` | 2026 | Not attached |
| `educator_state_initial_certification_principal` | 2026 | Not attached |
| `educator_state_initial_certification_superintendent` | 2026 | Not attached |
| `educator_state_initial_certification_teacher` | 2026 | Not attached |
| `educator_state_new_hires` | 2026 | Not attached |
| `educator_state_retention_by_demographic_group` | 2026 | Not attached |
| `educator_state_retention_by_gender` | 2026 | Not attached |
| `educator_state_retention_by_preparation_route` | 2026 | Not attached |
| `first_financial_accountability_district` | 2026 | 2026: district |
| `first_financial_accountability_university_charter` | 2026 | 2026: district |
| `frc_per_pupil_expenditure` | 2025 | 2025: campus, district, state |
| `peims_college_credit` | 2022–2026 | Not attached |
| `peims_financial_single_file_actuals` | 2013–2025 | Not attached |
| `peims_financial_single_file_budgets` | 2026 | Not attached |
| `peims_special_education_by_primary_disability` | 2013–2026 | Not attached |
| `peims_staff_fte_and_salary` | 2023–2026 | Not attached |
| `peims_student_enrollment_by_ethnicity` | 2019–2026 | Not attached |
| `peims_student_programs_and_special_populations` | 2023–2026 | Not attached |
| `peims_teacher_course_enrollment` | 2023–2026 | Not attached |
| `rda_special_education_indicator1` | 2025 | Not attached |
| `region_tapr_dropout_attendance` | 2024, 2025 | Not attached |
| `region_tapr_staar_all_grades` | 2025, 2026 | Not attached |
| `region_tapr_staar_bilingual_esl_growth` | 2026 | Not attached |
| `region_tapr_staar_bilingual_esl_performance` | 2026 | Not attached |
| `region_tapr_staar_end_of_course` | 2024–2026 | Not attached |
| `region_tapr_staar_grades_3_to_8` | 2026 | Not attached |
| `region_tapr_staar_participation` | 2024–2026 | Not attached |
| `region_tapr_staar_school_progress` | 2024–2026 | Not attached |
| `region_tapr_staar_tested_grade_3` | 2024–2026 | Not attached |
| `region_tapr_staar_tested_grade_4` | 2024–2026 | Not attached |
| `region_tapr_staar_tested_grade_5` | 2024–2026 | Not attached |
| `region_tapr_staar_tested_grade_6` | 2024–2026 | Not attached |
| `region_tapr_staar_tested_grade_7` | 2024–2026 | Not attached |
| `region_tapr_staar_tested_grade_8` | 2024–2026 | Not attached |
| `region_tapr_staff_profile` | 2025 | Not attached |
| `region_tapr_student_profile` | 2025 | Not attached |
| `spp_timely_initial_evaluation` | 2024, 2025 | Not attached |
| `state_tapr_college_career_military_readiness` | 2025 | Not attached |
| `state_tapr_college_readiness_components` | 2025 | Not attached |
| `state_tapr_dropout_attendance` | 2024, 2025 | Not attached |
| `state_tapr_four_year_longitudinal_rate` | 2025 | Not attached |
| `state_tapr_staar_all_grades` | 2025, 2026 | Not attached |
| `state_tapr_staar_bilingual_esl_growth` | 2026 | Not attached |
| `state_tapr_staar_bilingual_esl_performance` | 2026 | Not attached |
| `state_tapr_staar_end_of_course` | 2024–2026 | Not attached |
| `state_tapr_staar_grades_3_to_8` | 2026 | Not attached |
| `state_tapr_staar_participation` | 2024–2026 | Not attached |
| `state_tapr_staar_school_progress` | 2024–2026 | Not attached |
| `state_tapr_staar_tested_grade_3` | 2024–2026 | Not attached |
| `state_tapr_staar_tested_grade_4` | 2024–2026 | Not attached |
| `state_tapr_staar_tested_grade_5` | 2024–2026 | Not attached |
| `state_tapr_staar_tested_grade_6` | 2024–2026 | Not attached |
| `state_tapr_staar_tested_grade_7` | 2024–2026 | Not attached |
| `state_tapr_staar_tested_grade_8` | 2024–2026 | Not attached |
| `state_tapr_staff_profile` | 2025 | Not attached |
| `state_tapr_student_profile` | 2024, 2025 | Not attached |
| `tprs_student_membership_enrollment_reports` | 2026 | 2026: campus, district, state |
