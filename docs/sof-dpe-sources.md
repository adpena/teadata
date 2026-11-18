# Summary of Finances District Payment Estimate

This source family preserves selected TEA funding quantities as published. It
does not calculate entitlement, revenue, receipts, expenditures or funding
adequacy. The source is the **2025–26 Near Final publication run 47422**,
downloaded November 17, 2026. Publication run and each district row's run are
different provenance fields. A workbook can legitimately contain several row
runs; selecting the publication does not rewrite them.

The official [State Funding page](https://tea.texas.gov/about-tea/state-funding/state-funding)
links the [statewide SOF workbook](https://tea.texas.gov/about-tea/state-funding/fy2025-2026-sof-data.xlsx)
and its [run-specific dictionary](https://tea.texas.gov/about-tea/state-funding/state-funding-reports-and-data/fy2025-2026-sof-data-dictionary-runid-47422.xlsx).
The retained captures freeze those bytes; a future publisher change is another
source revision, even if TEA keeps the same URL or Final label.

## Coverage and admission

The complete raw DPE inventory has four tables, 593 fact occurrences and
729,857 cells. `inputs_dpe`, `outputs_dpe` and `mo_asahe_dpe` each have 1,243
LEA rows. `is_asahe_dpe` has 1,030; its 213 absent LEAs are not zero-valued rows.
The other four LPE worksheets remain in the original workbook and are not
relabeled as notes or silently substituted for DPE.

The six-digit source-key scope is `district`, but the publisher's roster is
broader than school districts: 21 input recipients outside the selected district
entity snapshot are named as 20 education service centers and Bowie County.
They remain source LEA rows, with their publisher names and keys; they are not
manufactured school district entities. The native table's grain is explicitly
one publisher LEA row. A school district cohort must use its selected district
IDs rather than treating all 1,243 recipients as school districts or summing
nonadditive funding-rate parameters.

The rosters overlap at 1,222 district IDs. Seven district entities have no row
in this DPE source edition. That absence is not a reported zero and must not be
filled from another release or year.

The reviewed interpretation selects **486 fields: 462 numeric quantities and
24 literal code/text fields**. It excludes 104 held fields and three attendance
fields already owned by `summary_of_finances_attendance`. Codes remain strings,
including numeric-looking codes. No undocumented boolean meaning is inferred.
The complete admission and hold ledger, dictionary descriptions, report-line
references and source coordinates are in
[`sof-dpe-47422-v1.json`](../teadata/methodologies/sof-dpe-47422-v1.json).

Units distinguish dollars, funding rates, headcounts, FTE, weighted FTE, ADA
and weighted ADA. None is interchangeable with Membership or Enrollment.
Each field retains its own period: a school year, tax/calendar year or explicit
as-of date. Current funding-calculation observations do not establish the
vintage of every underlying input or the receipt of a state payment. Earlier
insurance, tax and prior-law inputs are not relabeled as current-year measures.

Original numeric lexemes are decoded directly into integers or exact Arrow
decimals, including source precision extending to 19 decimal places. Negative
amounts and observed zero stay observed. Absent cells and stored blank text
remain missing with their distinct original raw values. There is no Excel
display scaling, binary-float intermediate, rounding or masked reconstruction.
Native support includes raw/status, original worksheet, one-based source row
and column, estimate basis and row run. The standard export machinery carries
this lineage; high-precision workbook output must preserve the exact value
rather than imply Excel's numeric precision is unlimited.

## Unresolved source questions

- Cayuga's `MENTOR_ALLOT` is absent in the XLSX but its matching district/run
  report displays `$0`. That field is held; the report display is not used to
  fill the source blank. The retained audit checked 120 selected report-display
  comparisons: 119 matched at displayed precision and this one did not. This
  is separate from exact workbook-cell replay.
- Facilities lines 16 and 28 have an unresolved published formula question.
  The known dependency closure includes lines 17, 18, **27**, 29, 30 and 34,
  `ASAHE_46_071_ADJ`, `EDA_STATE_SHARE_ADJ` and `TOT_STATE_AID`. Those amounts
  remain held. No intended correction or alternative entitlement is inferred.
- Other holds concern ambiguous units/periods, obsolete or internal fields,
  staff descriptions accompanied by fractional values, five-plus versus
  greater-than-five teacher thresholds and repeated state/county context.
  Their raw values remain retained; they are not independently additive
  district measurements or a replacement for direct Texas observations.

`RDSPD_AMT_DIFF` is a published per-student funding-rate parameter derived from
statewide quantities, not a district total. Its currency-per-pupil unit and
definition must accompany it; it must never be summed across district rows.

## Preparation and release boundary

Use the existing offline financial command with a retained provenance store:

```sh
uv run python -m teadata.scripts.refresh_financials \
  --sof-dpe --store /path/to/retained/provenance --output /path/to/new/candidate
```

The command refuses an existing output. It emits an **unpublished candidate**,
not a qualified or activated release. Original-cell replay, exact scientific
dispatch admission, independent review, normal-parent composition, immutable
publication and application adoption are separate steps. The command does not
change a selected release by itself. The normal 0.0.167 selector is finalized
only after its separate integration qualification.

After normal integration, `summary_of_finances_dpe` uses the same lazy entity
source namespaces, native tables, discovery, DSL and exports as other attached
datasets. It adds no entity store or parallel query API. Select an exact
district `EditionKey` from `release.for_entity(district, edition="47422")`
for report-edition access. School-year selection applies only to compatible
field measurement periods; it must not turn tax years or as-of dates into
school-year observations.
