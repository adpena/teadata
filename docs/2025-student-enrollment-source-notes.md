# 2025 student enrollment race labels

Status: source correction and independent selected-source comparisons are
prepared; corrected-artifact qualification and release rebuild are pending.
This note does not activate new app data or modify published 0.0.126.

The retained 2025 STUD CSV headers and dictionary descriptions label both
`CPNTINDC/P` and `CPNTASIC/P` as Asian, with the corresponding district codes
`DPNTINDC/P` and `DPNTASIC/P` repeating that conflict. The first pair has labels
such as `District 2025 Student Enrollment: Asian Count`; the second uses
`District 2024-25 Asian Students Count`. The acquired labels are evidence, not
authoritative corrected definitions.

The selected-original audit (`9af21848ba2327290b5a49d850bb9d05cab2c38c06cf3361931ed6cee716985f`,
34,337 bytes) records all seven racial count/percentage pairs for membership and
enrollment, plus both totals, for district `227901` and campus `227901119`.
Its CSV and dictionary observations are bound to these original acquisitions:

| Original | SHA-256 | Bytes |
| --- | --- | ---: |
| Campus STUD CSV | `7e0e762a8f2ba50687fd8c6a59e8052d5740ba57abbedecddfca064bbd77548e` | 6,979,601 |
| District STUD CSV | `8ccd14fb2eaff8c73d94960950cc359d44bf4b81edb721975edf6795b8995811` | 1,057,804 |
| Campus STUD dictionary HTML | `01b60e8eec23e5c158206d796c0ce82c713417e5f822f1ab81354199814ef209` | 212,356 |
| District STUD dictionary HTML | `693d84e0fa2285918c0f0d437a8fbfdefa72b66c1a7acd8ddac94165b536568d` | 214,388 |

The respective acquisition receipt IDs are
`3dec9c6e933d9f88c6efacc6f00f8669f8dda0e9d3c51ee9c176f910011e7b35`,
`cdef4db2551805ed9aa901243df73d91843d158d14426592607e842ce7036231`,
`114edf213e7dd527c3167b5e6400feeeeae37ab272bcd9541e771d21c8171bd4`, and
`237fa1b42ce2d30f53bb56c7079a147c41b11e94aefb68927fca27e783cd9208`.
The original CSV receipts retain their `student_studf_profile` logical-name typo;
this correction does not relabel those receipts.

The [official 2025 Austin ISD Student Information report](https://rptsvr1.tea.texas.gov/cgi/sas/broker?_debug=0&_program=perfrept.perfmast.sas&_service=marykay&ccyy=2025&id=227901&lev=D&prgopt=reports/tapr/student.sas)
was acquired separately, as was the [official Highland Park EL report](https://rptsvr1.tea.texas.gov/cgi/sas/broker?_debug=0&_program=perfrept.perfmast.sas&_service=marykay&ccyy=2025&id=227901119&lev=C&prgopt=reports/tapr/student.sas).
The saved public HTML and acquisition records are retained in the canonical
ProvenanceStore and selected in `teadata/reference_cases/publisher_2025.json`:

| Report | Artifact SHA-256 | Bytes | Acquisition record |
| --- | --- | ---: | --- |
| Austin ISD | `b3a9383958500d32c16e39e8b9dfc12d633ed50b53894dcd4bc6510e567ee534` | 105,720 | `a83c507d701b9b77a9128ca7c8d62d1d229cdfabe8bb0f1cdcdbf8a6090a5202` |
| Highland Park EL | `7dbe9ffb7c7fcc30d2555d464d7a73deb2028608071e25fe34d14fb89e075fa3` | 98,724 | `b7405cf94a0fc80f3b9f5faa44610870acfcc4ff1e18a68d8bddfac08e254176` |

The 60 reviewed demographic values include all seven races' counts and
percentages for membership and enrollment, plus both totals, in each scope.
District Asian membership is 3,443 and enrollment is 3,448; American Indian is
125 in both populations. Highland Park membership is 648 and enrollment is 649.
The report columns establish these separate populations; staff demographics are
outside this correction. These rolling-report observations do not establish the
historical acquisition or transformation event for a prepared workbook.

## Canonical correction

The two 2025 crosswalk resources change only these canonical field destinations
for both campus and district, and increment `mapping_version` to 2:

| Exact source pair | Canonical field suffixes |
| --- | --- |
| `CPNTINDC/P`, `DPNTINDC/P` | `2025_student_enrollment_american_indian_count/percent` |
| `CPNTASIC/P`, `DPNTASIC/P` | `2025_student_enrollment_asian_count/percent` |

Each field retains its `campus_` or `district_` prefix. Literal `label`,
`publisher_description`, dictionary occurrences, source codes, column positions,
source hashes and prepared-input hashes remain unchanged. The existing `label`
is the publisher label; `field` is the corrected canonical destination. Neither
field spelling nor publisher edition assigns a new measurement period or policy.

The existing composite reader's `columns` contract maps corrected canonical
names to the immutable prepared workbook's old headers. Its ordinary
simultaneous rename and schema check perform the normalization; the raw `source`
profile explicitly clears that map. A future processing receipt must record the
effective reader and the evidence used. No compatibility alias or converter is
introduced.

## Lagged Student Information periods

The retained 2025 reports explicitly label Mobility, Student Attrition and
Students with Disciplinary Placements as 2023-24. Per scope, this covers eleven
mobility groups with count/percentage pairs, the attrition count/percentage pair,
and the disciplinary-placement count/percentage pair. Explicit metric periods
now cover all 52 fields across campus and district in both composite and
source-only STUD contracts. The four existing composite all-student percentage
overrides remain. Of the 48 additional composite bindings, 44 previously inherited
2024-25; four disciplinary fields were previously unbound by scalar APIs.

These are reported reference periods, not a blanket change to the table year or
proof of every cohort definition. The official 2025 glossary's attrition prose
and formula disagree about the fall-to-fall cohort. Retention and teacher
turnover also need explicit cross-year cohort review. Their unknown source-only
periods are preserved. Synthetic catalog and analytical-export regressions check
period transport for all reviewed fields; they do not establish TEA fact values.

## Qualification and migration

The existing `source_notes` references now bind the actual scope-specific report
acquisitions. Independent comparisons cover the selected original CSV/dictionary
observations, both official reports, selected 0.0.126 prepared-workbook cells and
the retained 0.0.126 entities. The reference JSON holds the reviewed literal
expectations and source coordinates. The historical 0.0.118 helper's execution
and prepared-file derivation remain unknown.

The bounded lineage regression uses exact three-record CSV fixtures, changing
only fixture input hashes and row extents in cloned source contracts. It retains
all headers, codes, canonical destinations and dictionary occurrences, then runs
canonical source preparation to Parquet, ordinary Config enrichment and actual
EntityStore persistence before comparing all 60 measurements. A separate XLSX
regression checks the four old-header renames per scope through `Config.load_df`.
Its local fixture observations do not claim original HTTP events. This verifies
the exercised processing path; a new full corrected catalog and artifact bundle
still need their own processing receipts and retained-release checks.

The old standard `*_2025_student_enrollment_asian_*` fields and the separate
`*_2024_25_asian_students_*` fields must not be silently combined or aliased.
Their corrected destinations change the data contract. A coordinated new
catalog, entities, snapshot, analytical artifacts, maps and app cache generation
must select the new build; old published catalogs remain reproducible. Exports
must retain original publisher labels alongside corrected canonical fields and
this source-note context. Passing code checks alone does not qualify the data.
