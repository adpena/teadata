# Class of 2025 four-year graduation

The Class of 2025 begins with students entering grade 9 in 2021–22 and adds eligible transfers into that cohort. The graduation cutoff is August 31, 2025. TEA uses fall 2025 records to resolve the cohort and continuation status. TEA released the September 2026 Accountability Research workbooks with embedded data dictionaries and disclosure notes. [Publisher index](https://tea.texas.gov/data-reports/school-performance/accountability-research/completion-graduation-and-dropout/four-year-graduation-and-dropout-data-class-2025) · [TEA processing documentation](https://tea.texas.gov/data-reports/school-performance/accountability-research/dropcomp-4yr-processing-class-2025.pdf)

| Scope | Grain | Coverage |
| --- | --- | --- |
| District | Six-digit district ID and `calc_for_state_acct` | 2,164 rows across 1,082 districts |
| Campus | Nine-digit campus ID and `calc_for_state_acct` | 3,896 rows across 1,951 campuses. Six IDs have only the federal row. |
| State | One directly reported Texas observation | 388,460 graduates among 426,137 cohort members, or 91.2% |

For district and campus rows, `No` identifies the federal calculation and `Yes` identifies the statutory state calculation. TEA describes separate campus and district exclusion rules for each. The publisher applies neither set of local accountability exclusions to the reported Texas longitudinal rate. The statewide Federal Report Card cross-tabulation corroborates the published numerator and rate. The 92.8% rate TEA estimates under state-accountability exclusions is a counterfactual. [Texas cohort table](https://rptsvr1.tea.texas.gov/acctres/completion/2025/state.html) · [TEA annual report, PDF pages 144 and 150](https://tea.texas.gov/reports-and-data/school-performance/accountability-research/dropcomp-2024-25.pdf#page=144)

The campus workbook retains the original `AEA` flag. AEA completion includes graduates, continuers and TxCHSE recipients, with specific treatment of prior dropouts. It is a separate measure from graduation. The six absent statutory campus observations remain absent. A published `0` is an observed count. `-1` and `-3` are small-count and complementary suppression tokens, `.` indicates a group with no students, and `NA` or `N/A` are preserved as unavailable AEA fields. The archive retains raw cells for every field. Native typed projections cover graduation counts and rates for all students, economically disadvantaged students, students classified emergent bilingual in their last year, and special-education students. Native typed AEA fields cover the all-student completion series. The state projection selects four distinct rows from the three directly published state tables. The original pages retain the other state groups.

Select the exact `accountability_research_four_year_graduation` source with `reader_profile="source"`. The intake validates exact originals, full workbook header and dictionary coverage, local keys, complete-case arithmetic, and selected reports from TEA's separate Research lookup. TPRS campus parity and default activation are separate gates.

```sh
RESEARCH_CONFIG=teadata/teadata_sources.yaml
uv run --locked --offline --extra scripts python -m teadata.scripts.refresh_research_graduation \
  --store "$TEADATA_PROVENANCE_DIR" \
  --config "$RESEARCH_CONFIG" \
  --config-sha256 "$(shasum -a 256 "$RESEARCH_CONFIG" | cut -d ' ' -f1)" \
  --scope district --scope campus --scope state --output "$RESEARCH_CANDIDATE_DIR"
```

Pin the resulting `release.json` and select the exact scope with `DataRelease.catalog.editions()`. Filter district or campus reads on the published identifier and accountability method. The workbook row locator starts at 2 because row 1 is the publisher header. Use source references and the published group labels for statewide rows.
