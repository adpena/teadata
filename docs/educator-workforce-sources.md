# Educator workforce

The three hiring products below first joined normal release 0.0.158 and remain
attached to the selected normal 0.0.163 data release. The packaged default pin
selects that current data release; explicitly select historical 0.0.158 when
reproducing its original parent. TAPR staff fields are separate. Normal 0.0.163
also attaches the qualified retention and direct certification families below:
68 editions and 184 published rows, with the stated holds retained. See the
[current inventory](source-integration-status.md) for exact parent and child coverage.

## Hiring in normal release 0.0.158

The normal release adds **45 tables**: district hiring and reported EPP detail
for **2009–10 through 2025–26**, and direct Texas for **2015–16 through 2025–26**.
Retained-source replay checked all **83,478 district rows, 168,585 EPP rows and
11 Texas rows**, publisher fields, raw/status companions and worksheet
coordinates. It also checked nine frozen publisher-row examples, exact
district/EPP key sets including zeros, and direct Texas category counts and
ratios. The composed parent passed the existing canonical data validator for
all 15 native attachments. A selected actual-parent workflow checked entity
namespaces, DSL, historical rows and Arrow, DataFrame, CSV, Excel and SQLite
exports. This is retained-source reproduction, not fresh re-acquisition or
independent confirmation of every original historical definition.

The hiring child remains pinned to
`8989e9860aadd5d43bcb30bcc40aab2af59f230429638fb0723a1496ca821626`.
Its exact bytes are now an attachment of normal parent
`be9803cf7c27cff1020b5ec958bf1158d960a9ad574750d7242959ee051092f7`.
The parent producer was software **0.0.157**, source
`f0b25ec83d90963a14ea6a0589193e8e7579d78e`; software **0.0.158** is its
subsequent consumer. That historical identity does not change when later normal
parents retain the child. Current default selection follows the packaged pin;
application adoption is a separate deployment. See [release selection](data-releases.md).

There is no campus hiring product. Historical source IDs remain queryable even
when absent from the latest entity roster. Retrospective publisher names do not
establish historical entity names or continuity. Preserve category, reported
EPP region and assignment dimensions; a single district/year can have multiple
published rows, so scalar access refuses to choose one. The dated dictionary
limitations below remain in force. Retention and certification are separate
families with their existing holds.

## Reported EPP assignments

The [TEA version 5 district-and-EPP workbook](https://tea4avcastro.tea.state.tx.us/oess/edrs/regional-dashboards/downloads/new-hires-by-school-system-and-epp-views-4-and-5.xlsx)
contains the `District Counts Long` worksheet. The local native product is
`educator_district_new_hires_by_epp`, with district measurement years ending
2010–2026. Its key is the reported district number, school year, certification
category, EPP region and EPP assignment. `new_hires_count` is an integer count
at that key. The raw count, status and worksheet row remain available.

`reported_epp_assignment` includes literal publisher placeholders such as
`NA-Re-enterer`, `Na-Uncertified` and `NA-Previously Part-Time`. They are
observed labels and their associated counts remain observed. A label or
reported region does not identify an institution, current program, preparation
route, or individual teacher. The workbook supplies no EPP organization ID
column and no directly published campus or statewide EPP-disaggregated rows.
Do not join labels to current institution or district directories implicitly.

```python
view = release.observations(
    "educator_district_new_hires_by_epp", "new_hires_count",
    scope="district", school_year=2013,
)
data = view.query("district_number == @district", params={"district": "227901"}).to_df()
```

Omit the year for the latest eligible observation in the selected release.
This is the retrospective version 5 series, not a reconstruction of each
original report vintage. Both district products retain their own historical
reporting rosters. Grouping EPP detail to district/year/category matches
every primary district key and count, including explicit-zero keys. This is
a comparison of separate publisher products that may share upstream inputs.
The direct Texas workbook separately checks aggregate category counts and
ratios for 2015–16 onward; it does not supply an EPP breakdown.

The [regional dictionary workbook](https://tea4avcastro.tea.state.tx.us/oess/edrs/regional-dashboards/downloads/newly-certified-and-new-teacher-hires-data-documentation.xlsx)
is labeled January 2023. Its `New Hires by School Sys & EPP` sheet has eleven
entries at rows 2–12. Five codes match the actual version 5 headers literally:

| Workbook code | Dictionary row | Native field |
|---|---:|---|
| `schyr` | 3 | `school_year_end` |
| `region_epp` | 5 | `publisher_epp_region` |
| `cert_cat` | 8 | `certification_at_hire_category` |
| `EPP_assigned` | 9 | `reported_epp_assignment` |
| `total_new_hires` | 10 | `new_hires_count` |

The crosswalk retains exact code, description and coding text for all eleven
dictionary entries with the original workbook hash, worksheet and row
anchors. Dashboard titles remain in column B of the retained workbook. Seven
actual headers have no literal match: `district`, `region`, `distname`,
`geotype`, `region_2`, `Year` and `region_epp2`. Dictionary fields such as
`district_code`, `District_name` and `county_org_cd` are not aliases added
to the current data. The dictionary's `org_name` row pairs an EPP title with
a temporary-certification description and numeric coding. It remains
unbound evidence. No TEA values or documentation have been repaired.

The [April 2025 business rules](https://tea4avcastro.tea.state.tx.us/oess/edrs/regional-dashboards/downloads/newly-certified-and-new-teacher-hires-business-rules.pdf)
and the January 2023 dictionary predate version 5. Their dated definitions
are retained without assuming they describe every later category or label.
The selected release carries these limitations and links in native export
provenance. Excel embeds the descriptor; keep the separate
`view.provenance().to_dict()` record with a CSV. Original-header exports are
not enabled for these typed hiring tables.

## Statewide continuous teacher retention

The normal release admits three views of first-teaching cohorts from the
April 2026 workbook, covering 2013–14 through 2024–25.

| Dataset | Reported grouping | Cohorts admitted in normal 0.0.163 |
|---|---|---|
| `educator_state_retention_by_preparation_route` | Initial standard certification preparation route | 2013–14 through 2024–25, excluding 2017–18 |
| `educator_state_retention_by_demographic_group` | Publisher race and ethnicity group | 2013–14 through 2024–25 |
| `educator_state_retention_by_gender` | Publisher gender group | 2013–14 through 2024–25 |

The cohort is the teacher's first school year as a regular classroom teacher
after initial standard certification in the preceding academic year. TEA
uses teachers of record with at least 0.5 cumulative FTE. Year 1 counts
continuous employment through the school year after the cohort. Years 2–5
extend that interval. Teaching in another Texas public school system can
count, and a break in employment ends continuous retention. Rates are
publisher fractions. The three views share statewide totals, so adding them
would count the same teachers multiple times. [TEA workbook](https://tea4avcastro.tea.state.tx.us/oess/edrs/teacher-retention-dashboard/downloads/Teacher_Retention_Workbook.xlsx) ·
[TEA business rules](https://tea4avcastro.tea.state.tx.us/oess/edrs/teacher-retention-dashboard/downloads/Business_Rules_Document_Retention.pdf)

```python
view = release.observations(
    "educator_state_retention_by_demographic_group",
    "first_year_teacher_count", "retained_year_1_count", "retained_year_1_ratio",
    scope="state", school_year=2024,
)
hispanic_latino = view.query("publisher_group == @group", params={"group": "Hispanic / Latino"})
data = hispanic_latino.to_df()
```

Without a year, the API selects cohort 2024–25 for all fields, including a
Year 5 request. The workbook has only Year 1 for that cohort. Year 5 has
values through cohort 2020–21. Unreported horizons preserve the source blank
and a missing status. The typed export keeps the decoded numeric value,
its text and the worksheet row. The retained workbook preserves literal
Excel storage. CSV users can save the same provenance descriptor as a
separate file. Excel includes it in the metadata sheet.

The compatible `school_year` period is a **first-teaching cohort selector**,
not the observation year for every follow-up field. Cohort 2024–25 Year 1
describes employment through 2025–26; its missing Year 5 is unreported.
Source and edition notes carry this distinction in machine provenance, the
CSV companion descriptor and XLSX Source notes.

The business rules' 2017–18 out-of-state Year 4 example gives 56.44%.
The workbook gives 1,786 of 3,168, which is 56.38% rounded to two places.
The 2017–18 preparation-route edition remains held. TEA's 2023–24
Hispanic/Latino Year 1 example agrees at the reported precision, 91.6%
from 4,965 of 5,422. The separate pathway-at-entry worksheet is also held
because its published rates and counts do not consistently reconcile.
The workbook provides statewide views only.

Build the reviewed cohort editions offline in a new directory.

```sh
uv run --locked python -m teadata.scripts.refresh_educator_retention \
  --store "$TEADATA_PROVENANCE_DIR" --output "$EDUCATOR_RETENTION_CANDIDATE_DIR"
```

Repeat `--cohort-year` to choose explicit first-teaching school-year ending
years. A 2018 selection builds the demographic and gender editions.

## Initial standard certification by role

The September 2026 TEA workbook directly publishes 33 statewide role/year
totals. The materializer selects those cells, rather than summing its detail.

| Dataset | Academic years in normal 0.0.163 |
|---|---|
| `educator_state_initial_certification_teacher` | 2015–16 through 2025–26 |
| `educator_state_initial_certification_principal` | 2015–16 through 2025–26 |
| `educator_state_initial_certification_superintendent` | 2015–16 through 2025–26 |

Each leaf preserves the physical workbook row, academic year, reported role,
three literal `All` dimensions, observed status and raw count. Independent
fixed-coordinate worksheet XML witnesses bind every complete six-cell tuple
before materialization. The two archived All Routes tables per role cover
2018–19 through 2023–24: sixteen role/year totals match, and the two teacher
totals below are edition differences. The other fifteen role-years have no
archived comparison. Missing archive coverage is not reported as agreement.

The compatible `school_year` selector means the publisher's certification
academic period, **September 1–August 31**. The workbook says the report was
created **August 25, 2026**. That date does not establish a data cutoff or an
"as of" date for the counts; completed coverage and finality through August 31
are unknown, including 2025–26. These statements are carried in machine export
descriptions, CSV companion descriptors and XLSX Source notes. Counts concern
people receiving an initial standard certificate for the reported role.
Employment, hiring and retention use separate sources, and the three role
populations are separate and not additive.

No certification detail is qualified by this total-cell selection. Teacher
detail repeats keys from 2020–21 onward. Principal and superintendent detail
is absent for 2015–16 through 2019–20. Their 2020–21 detail sums are 1,599 and
395, against directly published totals 1,837 and 553; the archived reports
separately show Alternative counts 238 and 158. Their latest 2025–26 detail
sums are 1,881 and 367, against direct totals 1,932 and 387. These discrepancies
remain unexplained. Detail diagnostics record actual missing, repeated-key and
sum outcomes; neither a matching sum nor absence of a known revision marks
detail as qualified. No row deduplication or aggregate repair is performed.

```python
view = release.observations(
    "educator_state_initial_certification_principal",
    "certification_count", scope="state", school_year=2024,
)
print(view.to_df())
```

Build the scoped candidate offline from the retained store and the exact
reviewed XML/PDF source-witness observation. The existing CLI argument name
`--archive-receipt` is retained; it now binds that complete source observation,
not an invented replacement of the unavailable historical receipt.

```sh
uv run --locked python -m teadata.scripts.refresh_educator_certification \
  --store "$TEADATA_PROVENANCE_DIR" --archive-receipt "$CERTIFICATION_ARCHIVE_RECEIPT" \
  --output "$EDUCATOR_CERTIFICATION_CANDIDATE_DIR"
```

The crosswalk records the workbook's six dictionary definitions and source
row numbers. Its published coding labels remain separate from the physical
rollup types. For example, the dictionary labels gender and ethnicity as
numeric while the rollup cells use the literal string `All`.

### Errata and publisher edition differences

| Teacher certification year | November 2024 TEA report, pages 2 and 4 | September 2026 TEA workbook | Workbook row |
|---|---:|---:|---:|
| 2020–21 | 26,085 | 26,086 | 17 |
| 2023–24 | 17,797 | 17,801 | 26 |

Both editions report a statewide all-route total directly. TEA Data uses
the newer published workbook totals and preserves the older report in
provenance. The November 2024 methodology explains that TEA sometimes
updates counts when it corrects records or changes how it identifies the
eligible population. The reason for these two revisions remains unknown.
The workbook repeats detail keys in these years, so preparation-route and
demographic breakdowns remain unqualified along with all other detail in this
family. XLSX provenance worksheets, CSV companions and machine-readable export
descriptions carry the edition differences.

Sources: [TEA's September 2026 workbook](https://tea4avcastro.tea.state.tx.us/oess/edrs/newly-certified-by-role-dashboards/downloads/newly-certified-by-role-workbook.xlsx)
and [November 2024 teacher report](https://tea.texas.gov/data-reports/educator-data/newly-certified-teacher-demographics-by-preparation-route-1819-2324.pdf).
