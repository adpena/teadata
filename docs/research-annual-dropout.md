# 2024–25 annual dropout

TEA Accountability Research reports students who dropped out during the school
year and students cumulatively enrolled during that year. The public source
includes the grade spans 7–8, 9–12 and 7–12. Its September 2026 publication date
is separate from the 2024–25 measurement period. [Publisher downloads and lookup](https://tea.texas.gov/data-reports/school-performance/accountability-research/completion-graduation-and-dropout/annual-dropout-data-2024-25) · [TEA methodology](https://tea.texas.gov/data-reports/school-performance/accountability-research/dropcomp-2024-25.pdf#page=20)

| Scope | Native row key | Published coverage |
| --- | --- | --- |
| Texas | `grade_span` | 3 direct observations |
| ESC region | `region_number`, `grade_span` | 60 rows across all 20 regions |
| County | `county_number`, `grade_span` | 756 rows across 252 published county codes |
| District | `district_number`, `grade_span`, `calc_for_state_acct` | 6,845 rows across 1,171 districts |
| Campus | `campus_number`, `grade_span`, `calc_for_state_acct` | 19,067 rows across 4,298 campuses |

The district and campus `No` rows use the federal reporting calculation.
The `Yes` rows use the calculation with statutory exclusions for state
accountability. One district and nine campus grade-span pairs have only the
federal observation. The county workbook omits codes `131` and `151`. These
coverage gaps remain absent. Texas and ESC rates come from the direct reports,
and county rates come from the county workbook. The build keeps each published
grade span and does not compute statewide or combined-grade values.

All 32 groups retain the publisher's denominator, numerator and rate text.
`-1` represents small-count masking, `-3` represents complementary count masking,
and `.` denotes a rate unavailable because the group has no students. A
published numeric rate survives even when both counts are cross-masked. A rate
printed as `<0.1` preserves the strict upper bound of 0.1 percentage points in
the raw string. Scalar interpretation remains unavailable for this source.
The separate federal and state military-connected groups retain their own
fields. [Publisher glossary and masking](https://rptsvr1.tea.texas.gov/acctres/gloss2425.html#msk)

Select `accountability_research_annual_dropout` and `reader_profile="source"`.
Build against the retained public provenance store and an exact packaged Config.

```sh
RESEARCH_CONFIG=teadata/teadata_sources.yaml
uv run --locked --offline --extra scripts python -m teadata.scripts.refresh_research_annual_dropout \
  --store "$TEADATA_PROVENANCE_DIR" \
  --config "$RESEARCH_CONFIG" \
  --config-sha256 "$(shasum -a 256 "$RESEARCH_CONFIG" | cut -d ' ' -f1)" \
  --scope state --scope region --scope county --scope district --scope campus \
  --output "$RESEARCH_CANDIDATE_DIR"
```

Pin the resulting manifest and select an exact `EditionKey` from
`DataRelease.catalog.editions()`. The `grade_span` keys are the literal workbook
codes `78`, `912` and `712`. Texas and ESC leaves also retain the corresponding
display label as source context. Workbook rows carry their original worksheet
row number. Read the exact raw fields through the native source view or the
edition's Arrow dataset. For example, this reads Austin ISD's state-accountability
grade 9–12 rate as a raw publisher token from the [v0.0.135 release](https://github.com/adpena/teadata/releases/tag/v0.0.135):

```python
from teadata.dataset_metadata import EditionKey
from teadata.release import DataRelease

release = DataRelease(
    "https://github.com/adpena/teadata/releases/download/v0.0.135/annual-dropout-2024-25.release.json",
    expected_sha256="cd9221718c3077d3de6f571e7c383435f71fa71cf30c9136f67e404ed5bed2f0",
)
edition, = release.catalog.editions(scope="district", include_metadata=False)
key = EditionKey.from_dict({name: value for name, value in edition.items() if name != "edition_id"})
table = release.table(key)
austin = table.where(
    (table["district_number"] == "227901")
    & (table["grade_span"] == "912")
    & (table["calc_for_state_acct"] == "Yes")
).select("dist_allr").to_df()
```

The frame includes the row keys and source locator. The value in `dist_allr`
is the string `"1.9"` (percent), not a numeric estimate derived from masked counts.
