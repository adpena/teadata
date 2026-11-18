# 2025–26 TPRS Student Information field coverage

As observed November 18, 2026, normal release **0.0.166** contains the 179-field
Membership/Enrollment projection for 9,105 reporting campuses, 1,202 reporting
districts and the direct Texas report. It also contains 22 campus and 24
district/state [continuity fields](tprs-continuity-sources.md), measured on
the 2024–25 baseline cohort with fall-2025 return/processing. These are distinct
periods and populations. Class size is another qualified attachment.
Those exact Student attachments remain unchanged in normal release 0.0.167;
the machine inventory keeps its original 0.0.166 observation identity.

The retained complete census covers **1,386 cell groups and 4,709,972 physical
cells before Class Size**. The table below overlays its older parser
dispositions with the actual selected catalog, so already released continuity
observations are not mistakenly counted as unfinished. Suppressed values,
missing grades, blanks and observed zero remain different states.

| Disposition | Physical cells | Meaning and action |
| --- | ---: | --- |
| Live selected local fields | 1,585,672 | Use the normal catalog and each value's status. |
| Embedded/context cells | 1,981,354 | Retain publisher context; compare using direct geography observations. |
| Publisher blanks | 813,090 | No numeric observation is published; never fill by subtraction or older values. |
| Mobility period conflict | 226,776 | Resolve report heading 2024–25 versus glossary denominator 2025–26. |
| Disciplinary definition conflict | 20,616 | Resolve the 2024–25 label versus the glossary's 2023–24 action-code list. |
| CTE count bindings requiring scope-specific qualification | 18,210 | Two campus Membership count fields are implementable; they are not yet admitted. |
| CTE percentage denominator not yet established | 18,210 | Verify definitions and denominator, especially the grades 9–12 restriction. |
| Publisher no-students token on unadmitted CTE bindings | 46,040 | Preserve `-`; do not convert it to zero or infer a numeric observation. |
| Undocumented CTE dots | 4 | Obtain publisher meaning before numerical interpretation. |

This is field accounting, not a fresh download or a claim that every retained
cell is an eligible measure. The [machine-readable inventory](tprs-student-field-status-0.0.166.json)
records each group's scope, label, population/quantity where established,
source tokens/examples, live field/period or precise hold reason, and next
action. Its exact source-census identity is recorded as `census_sha256` in the
JSON, separately from the selected release and the observation date.

Published blank disability Enrollment cells and total-disability percentages
do not become inferred values. Campus report variants omit 469 retention
sections and 612 Class Size sections. Qualified continuity retains 97,554
absent-grade bindings and 9,380 absent-section bindings as explicit missing
states with no manufactured coordinates. These are publication facts, not
assumed eligibility exclusions.

The catalog read used to resolve live coverage was offline and opened metadata,
not observations. Original-source replay and retained HTML/XLSX witnesses have
separate receipts. **Authenticated production continuity export remains
unverified**; anonymous health/Policy checks and a runtime version readback do
not establish that user workflow.
