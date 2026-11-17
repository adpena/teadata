# Attendance and population sources

Select the source and measurement year before comparing values. TPRS Membership,
PEIMS Enrollment, published attendance/chronic-absence percentages and SOF average
daily attendance equivalents are distinct measures. A report edition or finance
run is not a measurement year. The selected release's catalog records the exact
fields, available scopes and source receipts; no source lookup falls back to an
older nonmissing value.

Both narrowly qualified families below entered the v0.0.154 normal release
and remain attached in v0.0.157 through the shared lazy entity/source machinery.
The current manifest is recorded in [source integration status](source-integration-status.md).
Candidate construction, publisher-source qualification, normal access/export
acceptance, publication and application activation remain separate stages.

## Current population

The published baseline for this extension, v0.0.164, retains 2025–26 TPRS Membership and
Enrollment at district, campus and direct Texas scopes, PEIMS programs for
2022–23 through 2025–26 and ethnicity enrollment for 2018–19 through 2025–26.
The TPRS acquisition contains 9,105 admitted campus reports in 23 partitions;
that coverage is for the selected Student Information reports, not every TPRS
domain. PEIMS ethnicity keeps one row per published group, and overlapping
program counts are not additive. Use the direct program total where offered.

Retained normal-parent checks separately exercised public districts, a charter
district, public and charter campuses, direct Texas, suppressed ethnicity cells
and DataFrame/CSV/Excel/SQL exports. Texas 2025–26 Membership was 5,454,059 and
Enrollment 5,467,642. These are retained-source checks; they do not establish
that every released record was freshly reacquired from TEA.

### Disability counts and composition: distinct denominators and source holds

The v0.0.164 normal release includes six Membership counts: the published disability
total and five primary-disability groups. Intellectual Disabilities groups ID,
learning disability, developmental delay and traumatic brain injury; it is not
an ID-only count. These counts need no percentage denominator. The separate
179-field source contract adds the five published disability-group Membership
percentages. Their denominator is **Total Students with Disabilities in
Membership**, not all students. Its shared source gate replayed all 10,308
retained reports on November 17, 2026. Normal release publication and application
adoption are separate gates; consult the [selected release inventory](source-integration-status.md).
The 168- and 174-field captured contracts remain unchanged.

Total-disability percent and disability Enrollment cells are blank in the
three publisher witnesses; no corresponding observations are manufactured.
CTE dots lack a documented interpretation and stay held. Lagged disciplinary,
mobility, attrition, retention and data-quality cells need their own definitions,
periods and denominators. The retained glossary has inconsistent years in its
discipline and mobility explanations. No silent period correction is applied.
The core projection stops at Mobility; the separate full-report inventory
accounts for all data cells before Class Size without admitting tail measures.
Class Size continues through its existing qualified reader.

On November 17, 2026, the existing producer materialized all 24 retained intakes:
9,105 campuses, 1,202 districts and direct Texas. All 174 fields passed producer
readback; every original field, raw token and status matched the prior Parquet
across 10,308 rows. The six added fields provide 61,848 observations, including
suppressed and unavailable states. No report was downloaded again for that run.
These are checks against frozen publisher extracts, not fresh live verification.
The three independent HTML/XLSX witnesses remain a distinct check.

The full inventory matched all 1,298 data cells in the three retained exports:
476 campus, 548 district and 274 Texas cells. Physical empty XML cells remain
distinct from absent cells and numeric zero. These witnesses contain 79, 78 and
39 explicit empty XML cells, respectively, and no absent data-cell addresses.
That finding describes these named witnesses, not every campus workbook.
The raw HTML/XLSX and download receipts remain authoritative.

```python
from teadata.tprs_student_inventory import inventory_student_report

audit = inventory_student_report(
    retained_html_bytes, scope="district", identifier="227901",
    contract=captured_student_grid,
)
for cell in audit.cells:
    if cell.disposition == "held":
        print(cell.section, cell.label, cell.row, cell.column, cell.raw, cell.reason)
```

Inventory coordinates are one-based parsed HTML table coordinates. Whitespace
normalization follows the existing HTML reader; original bytes retain styling,
merged-cell and whitespace evidence. Inventory is an audit, not another entity
dataset or a shortcut around source qualification. A changed or unrecognized
layout raises an error. No network lookup or fallback year is performed.

The published normal release and live app remain unchanged until normal-release
integration and app ingestion pass. A source-only candidate is not a selected
normal release. Derived grid and crosswalk references retain their actual
processing receipt through typed local-acquisition edges; they are not publisher
downloads. Invalid reference types now fail before report parsing or staging.

## Named attendance and chronic absence

The qualified `tprs_attendance_chronic_absence` source uses TEA's named **2025**
Attendance and Graduation reports, which explicitly publish **2024–25** and
**2023–24** attendance/chronic-absence rows. The similarly named retained bulk
fields for edition 2025 contain the older measurement year; they are not
substitutes for the newer named-report row. A bounded retry of the newer bulk
endpoint still returned the publisher's SAS error.

Each admitted local geography retains the all-students rate and ten subgroups:
African American, Hispanic, White, American Indian, Asian, Pacific Islander,
two or more races, special education, economically disadvantaged and EB/EL.
Neighboring state, region or district benchmark columns remain in the original
report; they are not relabeled as local subgroup observations. Student groups
may overlap. The source currently qualifies direct Texas and all 20 ESC regions.
District/campus parsing is implemented and tested, but complete current local
coverage and generic partition routing remain held.

Chronic absence describes unduplicated K–12 students enrolled for at least ten
days who were absent at least ten percent of their enrolled days. Attendance
Rate describes days present divided by days in Membership for grades 1–12.
The published percentages are retained; neither numerator nor denominator is
reconstructed. The comprehensive glossary and the report's year-specific
period and legend statements are pinned in the methodology.

The parser distinguishes `*` (suppressed), `-` (no students in the group;
not applicable), `?` (invalid/out-of-range), observed zero and missing source
rows. Unknown or blank source tokens fail admission. Rates use Arrow
`decimal128(4, 1)`, with exact raw tokens and report row/column locators.
Current retained controls include Texas 18.0% and Region 04 17.8% chronic
absence for 2024–25; the adjacent Texas 2023–24 row is 19.0%.

## SOF attendance equivalents

The qualified `summary_of_finances_attendance` source projects only six selected
attendance fields from the statewide 2025–26 bulk workbook, publication run
47422, **Near Final**. It retains all 1,243 LEA rows in each original worksheet
`inputs_lpe`, `inputs_dpe`, `outputs_lpe` and `outputs_dpe`, including charter
school districts. Each worksheet's row-run distribution is preserved: a
publication run does not mean every row was calculated in that run.

| Field suffix | Meaning | Availability in this source |
| --- | --- | --- |
| `ada_tot_refined` | Refined Average Daily Attendance | 1,236 populated rows and seven missing per input table; 27 observed zeros remain zero. |
| `ada_adj_tot_refined` | General Adjusted Total Refined ADA | All 1,243 input rows are missing. No substitute or zero is supplied. |
| `adj_avg_ada` | The SAS-calculated adjusted ADA for HB 2's Support Staff Retention Allotment under section 48.1581 | A separate output calculation. It is not the missing general adjusted-ADA input. |

Canonical field names retain the worksheet prefix, such as
`inputs_lpe_ada_tot_refined`, `inputs_dpe_ada_tot_refined` and
`outputs_lpe_adj_avg_ada`. LPE and DPE remain different estimates, with original
input codes 2 and 1. A request across physical tables must select separate views;
it does not join or combine estimates implicitly. The unit
`attendance_equivalent` permits fractional funding attendance and is not a
student headcount, staffing FTE or class size.

Original OOXML numeric strings go directly into Decimal and Arrow decimals,
without a float conversion. Input fields use `decimal128(21, 15)` and outputs
`decimal128(23, 17)`. For example, the publisher token
`543.68900000000008` is preserved exactly. Raw tokens, token-specific scale,
original sheet/row/column, row run, school year and estimate basis accompany
the selected values. Overflow, excess scale or an unreviewed token fails.
The full eight-sheet finance candidate is not qualified by these six fields.
Direct state/sector reports are separate observations, not sums reconstructed
from these LEA rows.

The run 47422 dictionary's bytes and independent origin observation are retained.
The new dictionary receipt is explicitly a **local re-observation**. Its original
download time was not recorded, and the later observation time is not presented
as a download time. Original report/download receipts remain separately linked.

## Preparation, access and export

Use the existing explicit preparation entry point with a retained provenance
store and a new output directory. It never acquires TEA data or publishes:

```sh
uv run --extra scripts python -m teadata.scripts.refresh_attendance sof \
  --store /path/to/provenance --dictionary /path/to/dictionary.xlsx \
  --dictionary-origin /path/to/origin-observation.json --output /path/to/new-sof-child
uv run --extra scripts python -m teadata.scripts.refresh_attendance chronic \
  --store /path/to/provenance --intake /path/to/intake.json \
  --intake-sha256 '<exact intake SHA-256>' --output /path/to/new-attendance-child
```

The normal enrichment build attaches exact, independently qualified child pins
through `options.native_sources`. These two sources first appeared in the
v0.0.154 parent and remain in v0.0.157; they are absent from v0.0.153. With the
current pinned release selected, use the same lazy `entity.dataset(name)`, dataset
dot path, `observations()` and `series()` planner as other sources. Explicit
unavailable years fail; current chronic-absence latest is 2024–25 while SOF and
population sources resolve to 2025–26. Region sources use their published region
keys; a district entity does not become a region observation.

DataFrames retain Arrow-backed decimals. CSV keeps exact text and uses the
common provenance sidecar; Excel writes Decimal values as text and includes the
Provenance worksheet. SQL callers use a sufficient text destination for portable
exact decimals or a verified native Decimal destination. SQLite NUMERIC/Float
must not silently round the source. Precision and provenance survive the export;
scientific comparability across years still requires source review.
