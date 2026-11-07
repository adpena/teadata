# Educator workforce: reported EPP assignments

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

The local candidate admits three views of first-teaching cohorts from the
April 2026 workbook, covering 2013–14 through 2024–25.

| Dataset | Reported grouping | Cohorts admitted locally |
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
