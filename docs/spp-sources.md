# SPP timely initial evaluation

`spp_timely_initial_evaluation` registers the Texas Education Agency's public
regional district detail workbooks for school years 2023–24 and 2024–25. Their
publisher report editions are 2025 and 2026, respectively. These are
pre-clarification district results for Indicator 11A, Timely Initial Evaluation.
The data measure initial evaluations within the state timeline for children
whose parents provided consent, subject to the reporting rules and exceptions.
The published target is 100%. District coverage across all 20 education
service center regions does not supply an official statewide or regional
aggregate. [TEA public reporting and editions](https://tea.texas.gov/special-populations-and-support/special-education/data-and-reports/local-educational-agency-public-reporting) · [TEA 11A calculation](https://tea.texas.gov/special-populations-and-support/review-and-support/sppi-11a-calculation-and-example.pdf)

The school-year ending year selects the source configuration. The native
catalog's edition key uses TEA's report edition. The catalog's measure
period preserves the underlying school year. The reports are source-only and
separate from the active snapshot. [2025 publisher guide](https://tea.texas.gov/special-populations-and-support/special-education/about-the-2025-lea-spp-indicator-report-ffy-2023-0.pdf) · [2026 publisher guide](https://tea.texas.gov/special-populations-and-support/special-education/about-the-2026-lea-spp-indicator-report.pdf)

## Values and source locations

The `Ind11` workbook sheet provides the consent count, timely evaluation count,
reported percentage and Met Target flag. Counts are nullable integers, the
percentage is an exact decimal on the 0–100 scale, and the target is a nullable
Boolean. Each measure has separate raw-token and status support. `-999` counts
are suppressed. A literal `NR` means the publisher did not report a value.
The reports do not distinguish its underlying reason at the individual cell.
A blank or absent count remains missing. An observed zero remains zero.
A visible rate or target does not license inference of a suppressed count.
[TEA workbook selector and masking rules](https://rptsvr1.tea.texas.gov/idea/index.html)

The catalog exposes the original header and its workbook position. Native
technical fields identify the cell representation, `spp_source_sheet` and
one-based `spp_source_record`. `spp_source_input` is a zero-based index into the
edition's acquisition receipts, ordered by ESC region 01–20. Together they
identify the exact source cell. Each receipt identifies the original workbook,
download URL, SHA-256 and acquisition time.

For observed count pairs, the publisher rate is checked at the number of
decimal places in its raw token. Across the reviewed editions, 21 rates that
differ from direct rounding match a two-stage check through two decimal
places. This describes the observed values. TEA's internal rounding method
is unknown. The producer keeps the original published rate and records the
check outcome in `spp_display_consistency`. An unexplained difference blocks
materialization. In two 2023–24 rows, a displayed 100.0% and a published target
flag of No are consistent with the exact count pair. Use the independently
published flag for compliance questions. [TEA reading guide](https://tea.texas.gov/special-populations-and-support/special-education/about-the-2025-lea-spp-indicator-report-ffy-2023-0.pdf)

## Reproduce an unpublished native candidate

The configured source contains 20 regional workbook receipts for each
edition. Supply their retained original objects and exact reviewed reference
objects in one ProvenanceStore. The producer runs offline and writes a separate
candidate with two district leaves, 1,212 district rows per edition. The
rosters have four different district IDs in each edition. Select the
intersection explicitly for longitudinal work.

```sh
uv run --locked python -m teadata.scripts.refresh_spp \
  --store /path/to/retained/provenance \
  --config teadata/teadata_sources.yaml --config-sha256 EXACT_CONFIG_SHA256 \
  --year 2024 --year 2025 --output /path/to/new/spp-candidate
```

The command verifies receipt bindings, physical source layout and native leaf
readback. It reports display-check coverage without activating the result.
Independent public district reports can corroborate the reported percentage
and Met Target. These separate presentations may show different decimal
precision. The underlying count pair is checked directly against the retained
regional workbook.
