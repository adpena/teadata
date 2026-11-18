# Financial workbook source candidates

## Reviewed FY2026 budget accounts and normal integration

The retained budget child
`bcff73e52199f4d8e527db2d66a89f8ee3a825d3734e7902e16ab4f7824e7f38`
passed the canonical reviewed native gate on Python 3.14.10 and 3.15.2. It
contains both FY2026 populations: 1,337,887 district rows and 88,400 charter
rows, preserving all 12,836,583 original account cells. Independent Budget PWR
instruction controls compare 2,402 AF/GF values over 1,019 district and 182
charter IDs. This qualifies budget account fidelity and those instruction
controls, not audited actuals, every financial interpretation or funding adequacy.
Signed integer amounts, zero, account codes, raw/status support and source row
locators remain explicit. FY2026 here is fiscal 2025–26, not a school-year label.

Normal 159 does not attach this child. The qualified normal160 composition adds
its exact pin while preserving all thirteen base inputs and sixteen existing
children. The resulting parent passed all seventeen canonical source gates.
Six actual-parent checks passed both population selections, fiscal and scalar
guards, normal provenance and bounded Arrow, DataFrame, CSV, Excel and
transactional SQLAlchemy SQLite roundtrips. Software 0.0.160 packages the
[same normal-parent manifest and pin](source-integration-status.md) as its
default. Public delivery and application adoption remain separate scopes.

Once that normal release is selected, use its entity namespace and an exact
`EditionKey` from `release.for_entity(district, edition="2026").datasets()`.
The two populations share dataset, scope and edition, so select `table="district"`
or `table="charter"` explicitly. Use the source's `select` with account predicates
or `release.table(exact_key).select("budget_amount").query(...)` for the existing
lazy DSL. New selections retain both the normal parent and source child in
provenance. A plain scalar refuses multiple account rows. The school-year
`.at(...)` resolver intentionally refuses this fiscal period; do not relabel it
or silently choose a population. Exports use canonical headers and preserve
the Excel worksheet row limit. A bounded SQLite roundtrip uses caller-owned
SQLAlchemy tables and a transaction, not a `SourceTable.to_sql` method.

The narrowly qualified [SOF attendance projection](attendance-sources.md#sof-attendance-equivalents)
is separate from the broad financial candidates described below. Its run 47422
Near Final source has four original LPE/DPE input/output tables and six selected
attendance fields. Literal OOXML decimals, mixed physical row runs, estimate
roles, missing general adjusted ADA and SSRA-specific output ADA remain distinct.
Qualifying those fields does not qualify the other finance fields or establish
revenue, expenditures, audited costs or funding adequacy. Check the selected
normal parent's coverage before assuming the new candidate is attached.

The [DPE quantity interpretation](sof-dpe-sources.md) separately accounts for
all 593 raw DPE facts and selects a reviewed subset with exact decimal storage,
field-specific periods and explicit holds. Its producer and scientific replay
do not activate or replace the selected normal release automatically.

The [PEIMS single-file archive](https://tea.texas.gov/about-tea/state-funding/state-funding-reports-and-data/peims-single-file-financial-data-downloads)
publishes separate district and charter account files. Historical packaging
varies. The FY2014 and FY2018 district budget ZIPs include a charter CSV
alongside the district CSV, and some historical charter budget links return
CSV directly. The current [single-file candidate reader](data-releases.md#peims-single-file-financial-budgets)
requires a ZIP containing exactly one member, the reviewed CSV. These
multi-member ZIPs and direct CSVs require reader support as well as review
of their headers, fiscal period, and population. The [publisher dictionary](https://tea.texas.gov/data-reports/financial-reports/school-finance-reports-and-data/reports-and-downloadable-data/datadictionary.pdf)
documents the account codes.

The dictionary's first page identifies the product as audited actuals. Its
`ACTAMT` paragraph on page 2 describes budget amounts and an as-of date.
The `FUNDYEAR` layout lists a small integer of length 2. Its narrative
describes one-character submissions displayed as four characters. Determine
budget versus actual and field representation from the exact edition,
archive label, and bytes. The `ACTAMT` entry explicitly allows negative
amounts.

`teadata.scripts.refresh_financials` uses one offline native producer for explicitly reviewed financial workbook editions. It does not edit active source configuration, select a latest year, activate an engine, derive a state aggregate, or publish a data release. Its unpublished candidate retains `field_correctness_qualified: false` and `release_activation: false`; structural admission and readback are separate from scientific qualification.

The existing Config source profile owns worksheet selection. The existing `column_schema` resource owns ordered literal publisher headers and canonical fields. An example source-profile fragment is:

```yaml
column_schema: reviewed-financial-columns.json
workbook:
  tables:
    actual_general_fund:
      sheet: "literal publisher worksheet name"
      header_row: 1
      scope: district
      grain: "one publisher district row per worksheet"
      key_columns: [district_number]
      identifier_field: district_number
  notes_sheets: ["literal publisher Notes worksheet"]
```

All original worksheets must be accounted for exactly once. Table names must match the reviewed crosswalk. Each table binds the exact original acquisition SHA, with real header positions, one canonical field per real publisher header, and honest dictionary-review status. Physical worksheet names and an embedded `TAB NAME` value are distinct observations. No year-pattern substitution, SOF header alias, fuzzy mapping, or automatic cross-year formula equivalence is used.

To produce a selected subset, keep every data table in the Config and crosswalk
and explicitly list the unused logical table IDs in `workbook.unselected_tables`.
For example, a DPE-only SOF candidate can list `inputs_lpe`, `outputs_lpe`,
`mo_asahe_lpe`, and `is_asahe_lpe` there while retaining all eight table
declarations. The producer checks the physical framing and exact headers of
every declared worksheet. It decodes and emits facts and editions only for the
selected tables. An unselected data worksheet is not a notes sheet, and its
fact values are not qualified by these header checks. Unknown/duplicate table
IDs and a selection leaving no output table are rejected. The original workbook
remains a provenance input; the explicit selection is frozen in processing
recipes, and `MATERIALIZATION.json` accounts for unselected physical tables.
Omitting the option retains the existing all-table behavior. This selection
does not weaken the normal release's complete native-child attachment check.

Publisher-label-only worksheets can opt into `header_identity: position` **in the existing crosswalk table**. This permits repeated literal header occurrences while ordered physical positions and canonical field IDs remain unique. Codes and labels both retain the exact publisher text; only canonical Python field IDs may distinguish occurrences, for example `_at_ad` and `_at_ca`. The native read explicitly skips the verified physical header, assigns reviewed canonical column names and selects verified physical indexes. It never treats Calamine's automatic duplicate-name mangling as publisher metadata. `catalog.header_records` retains each occurrence and position; code/label-to-name lookup still rejects ambiguity. Coded tables retain their default strict behavior, and TAPR's coded readers, comparisons, dictionaries and merges reject position mode.

`DatasetMetadata.metrics` supplies explicit per-field units and reference periods; a table with mixed enrollment, tax and fiscal periods can keep its table period unknown and declare the field periods separately. Numeric facts require reviewed units. Existing `SourceQualifiers` distinguishes SOF publication run/status and its actual physical row-run column from PEIMS actual/budget basis. Mixed SOF row runs need not equal the publication run, and a run column is not added to business keys to force its selection.

The existing canonical `MeasurementUnit` validates declared units. The numeric path rejects unknown/text units and retains fractional `fte` as FTE rather than coercing it to headcount. This capability does not invent an FTE column in PWR or establish a source's actual unit.

An individual table can explicitly opt into `uninterpreted_fields: [canonical_fact_name]` in its existing Config workbook declaration. Every selected name must be a real publisher-bound **fact** with an explicit `MetricDefinition` whose unit is unknown (`None`); identifiers, row keys, identity fields and known-unit facts are rejected. This option preserves the field directly as native Arrow strings, including range tokens, dots, negative values and nulls, with the existing sheet/row context. It does not call `measure`, infer a value/status, create duplicate raw columns, scale a number or reclassify a fact as identity. A null unit alone does not enable this behavior: without the explicit option, the numeric path still rejects unknown units. Unknown periods remain unknown, and unrelated fields still require their reviewed numeric units.

These uninterpreted facts remain discoverable in `catalog.variables(edition=key)` with `unit: null`, and `catalog.resolve_artifacts([(key, field)])` selects the string plus existing row keys and sheet/row context. There is no normalized raw/status support for that field. Do not treat a null, dot or range as zero, infer a suppression status, or apply numeric comparisons/scaling before reviewing its actual representation. In a schema3 candidate, the exact opt-in is accessible through the selected variable's `artifact_id`: `ref = catalog.artifact_ref(artifact_id)`, then `catalog.to_dict()["provenance"][ref.processing_id]["parameters"]["recipe"]["worksheet"]["uninterpreted_fields"]`. The contract binds that canonical recipe through `ref.contract["recipe_sha256"]`. This uses the existing processing record, not another field registry. The option is omitted from recipes when unused, preserving the default contract shape. Retaining a raw fact does not qualify its unit, period, definition or correctness.

For exact stored numeric tokens, set `cell_text: ooxml_literal` in the
existing `workbook` profile. This mode preserves the numeric text in the
original worksheet XML, decodes supported shared and inline text, and
distinguishes an empty string from an absent cell. It rejects unhandled
cell types, malformed framing, and workbooks over its resource limits.
The literal reader caps each dense worksheet at one million cells and
checks expanded string bytes before allocating Arrow arrays. It uses
the original stored tokens as inputs to the same financial
projection and leaves units and display formats to the reviewed source
contract. The default retains the Calamine string representation. A
Calamine string may round an original stored numeric token, so exact
value parity requires a separate source check.

Units must describe the **stored, unformatted values**. An Excel percent format can display a stored fraction as percentage points; currency display formats do not establish a monetary scale. This producer performs no style-based conversion, multiplier, percentage-point conversion, rounding or year-based transformation. Such conversions are unsupported. Keep a stored fraction as a reviewed ratio, or defer the field until its representation is established; do not label an unconverted fraction as percentage points. Literal tax/program header changes do not establish a conversion or equivalence.

The default mode uses the optional `fastexcel` dependency. Calamine reads
admitted worksheets as Arrow strings, which may differ from original
OOXML numeric tokens. Literal mode reads bounded worksheet XML and
constructs the Arrow strings directly. Both retain original workbook
bytes in provenance, reject formulas, error cells and merged fact cells,
and retain worksheet, row and column positions. Neither mode
recalculates formulas. Arrow operates on columns, and `values.measure`
interprets at most 4,096 distinct tokens per reviewed numeric field.

Normalized numeric fields retain explicit raw and status support in the same immutable leaf. Selecting a count through `catalog.resolve_artifacts` automatically includes its raw/status and sheet/row context; SOF's declared run column is included by the existing qualifier projection. Native schema3/4 access does not create scalar `Measurement` APIs. Use canonical `values.measure(raw, policy, unit=...)` when interpreting an explicitly selected raw value. Zero and negative financial values remain distinct from missing, not applicable and suppressed tokens. No protected value is filled from a total or another report.

The producer checks physical headers, exact projected native schema, positive bounded row counts, actual publisher entity identifiers and unique declared row keys. Unknown nonempty headers, unnamed data, unexpected footers and malformed keys fail rather than disappearing. A bounded OOXML structural guard rejects payload outside reviewed physical columns before native selection; this also prevents an explicit index list from hiding populated unnamed tails. Styled-empty trailing cells have no value to exclude. Resource admission is 16 MiB original compressed bytes, 256 MiB package expansion, 16 MiB per XML member, 20000 rows, 256 publisher columns, and 64 MiB projected Arrow bytes per table. These are admission limits, not an RSS or whole-process duration guarantee.

Each real fact worksheet becomes one native leaf through existing `EditionKey`, `BuildRecorder.leaf_contract`, `column_support`, `write_leaf_artifact`, `ArtifactRef` and processing receipts. Notes stay source-reference evidence, not fabricated research measures. The processing recipe captures the literal selection, exact Config, policy, native reader/extension identities, CLI/code identity and toolchain. Native readback is checked before the candidate becomes visible. A statewide file of district records remains district-grain; no district sum, average percentage or fake state identifier creates a publisher state total.

```sh
uv run --locked --extra scripts python -m teadata.scripts.refresh_financials \
  --store /path/to/canonical/provenance \
  --config /path/to/reviewed/sources.json --config-sha256 EXACT_SHA256 \
  --source reviewed_financial_dataset 2024 \
  --source reviewed_financial_dataset 2025 \
  --output /path/to/new-exclusive-candidate
```

Actual PWR2024/2025 inspection found narrowed 2025 AF program-code labels while GF retained broader labels, matching changes in only four Notes cells. These observations do not establish semantic equivalence or a publisher error. The original source contracts, reviewed references and later independent field-level examples must resolve what can be compared. Future 2026 workbooks use this same producer only after their own exact sheet/header/unit/period/source contracts are reviewed; a year label alone establishes none of those facts.

<a id="sof-202526-dpe-source-accounting-and-an-unresolved-formula-question"></a>

### SOF 2025–26 DPE: source accounting and an unresolved formula question

The retained statewide publication run **47422, Near Final** contains four DPE
worksheets: inputs, outputs, M&O ASAHE and I&S ASAHE. They contain 593 fact
occurrences: 591 publisher numeric fields and two text fields. Three attendance
facts remain owned by `summary_of_finances_attendance`. The separately reviewed
`summary_of_finances_dpe` projection qualifies 486 further fields (462 numeric
quantities and 24 literal codes/text), with 104 explicit holds. The raw contract
preserves all 593. A dictionary match alone does not establish numerical units,
measurement periods or correctness. See [the admission contract and limits](sof-dpe-sources.md)
and the selected release inventory for normal integration and app-adoption status.

The retained raw DPE candidate binds all 611 physical headers (593 facts and 18
identity/context columns) to the exact dictionary while leaving LPE mappings
unchanged. It accounts for all eight original sheets and emits only the four
DPE editions. An independent original-OOXML comparison matched all 729,857
DPE fact cells, including literal nulls, integer/decimal strings and row runs.
That receipt establishes raw fidelity and dictionary identity. It is separate
from the later 486-field semantic/source replay and 110 selected original-report
literal goldens. Qualification of that subset does not qualify the 104 holds,
certify publisher formulas or establish application deployment.

The first three sheets each report 1,243 LEA recipients, including education
service centers and Bowie County; these are not 1,243 school districts. The
I&S sheet reports 1,030 recipients; 213 absent rows must not become zeros. Preserve each row's
actual SOF run separately from publication run 47422 and retain Payment Cycle 2
without inventing a named interpretation. The separate 2026–27 Preliminary
run 47562 is a different funding edition.

On November 17, 2026, a direct read of the pinned original workbook and dictionary
confirmed this documentation/formula reconciliation question in `is_asahe_dpe`:

| District | Source row | Line 16: published / dictionary line 5 + line 15 | Line 28: published / dictionary line 5 + line 27 |
| --- | ---: | ---: | ---: |
| Cayuga ISD (001902) | 2 | 1,516,508 / 1,832,213 | 1,471,268 / 1,786,973 |
| Austin ISD (227901) | 916 | 242,137,652 / 229,297,732 | 236,172,085 / 223,332,165 |

Both physical headers say “Total State and Local Revenue for Debt Service.”
Keep column U / line 16 and column AG / line 28 distinct. The descriptions are in
`Variable Listing!D600` and `D598`, respectively. Cayuga's row run is 47223;
Austin's is 47422. These are exact original integer tokens and arithmetic.
Fresh official facilities-detail reports for these same district/run pairs
agree with the six checked LPE/DPE dollar cells and repeat both printed
formulas. This rules out a workbook-only parsing mismatch for those checked
cells; it does not establish which tax-collection base TEA intends to use.

An exact-decimal comparison of all 1,030 I&S rows found that each of lines 16
and 28 equals **LPE line 5 plus the DPE aid term** in 1,028 districts. Using
the displayed DPE line 5 instead matches 145 districts; 144 match both bases.
Louise ISD (241906) matches the DPE formula alone (published 0 versus LPE-base
50). Devers ISD (146903) matches neither (published 0 versus DPE-base 800 or
LPE-base 1,017). These are observed arithmetic patterns, not authorized formula
definitions or corrections. Both exceptions retain their original values.
The named official reports are available through the SOF report's own
“ASAHE for Facilities” link (`rpt=2`); their acquisition timestamps, hashes,
report headings and displayed-cell comparisons are retained separately.
No value was corrected by that reconciliation. The uncertainty also applies
to dependent lines 17–18, 27, 29–30 and 34 and the downstream ASAHE_46_071_ADJ,
EDA_STATE_SHARE_ADJ and TOT_STATE_AID amounts. Retain source descriptions and values; do not silently
substitute a different base in either formula.

The [TEA SOF download service](https://tealprod.tea.state.tx.us/fsp/Reports/ReportSelection.aspx)
is the publisher authority. Frozen workbook SHA-256:
`5026e6cb7533dfc157b2371108fd0ecfc54f3792bb63f5b3341acec90ccee715`;
dictionary SHA-256:
`9d018f14dd81e2bed5f896b621c5e86f64caa61b8dbc48a7373fe4dbaa575db0`.
The original dictionary download time is unrecorded; its later custody
observation is not substituted for that time. A separate November 17, 2026
reacquisition returned the identical workbook and dictionary bytes and retains
its own download time. This question blocks semantic
admission of the affected descriptions/formulas, not retention of their source
bytes or the separately qualified attendance fields.

Both annual GF/AF sheets also repeat their literal Capital Outlay label at AD1 and CA1. Their Notes crosswalk gives different later column pointers (CC1 and CB1); those pointers must not override the actual headers or establish equal values. Keep both physical observations and their raw/status support. PWR tax-rate denominators/scales and metric periods remain source-review questions; do not force a fractional, per-ADA or per-valuation measure into `count` or `ratio` to satisfy the current canonical unit vocabulary.

## Qualify selected FY2025 dollar fields

Use a reviewed, caller-supplied JSON case and its trusted SHA-256 to compare
campus All Funds Total Expenditures and district All Funds PIC 38 against
independent TEA reports. The campus field uses the object-section All Funds
amount at E7. The district field uses column BN in the statewide PWR workbook
and matches the individual district's All Funds amount. The case supplies
`schema_version` `1`, `fiscal_end_year`
`2025`, `campus_preparation_receipt`, `cohort_intake_sha256`,
`district_pwr_receipt`, `district_selector_receipt`, `about_actual_receipt`,
`accounting_guide_receipt`, `raw_dictionary_receipt`, and `districts`. Each
district entry uses an exact six-digit ID and two acquisition receipt IDs named
`campus_pdf` and `district_html`.

```sh
uv run --locked --extra scripts python -m teadata.scripts.refresh_financials \
  --store /path/to/retained/provenance \
  --qualification-case /path/to/reviewed/witness-selection.json \
  --qualification-case-sha256 EXACT_SHA256 \
  --output /path/to/new/qualified-actuals
```

The command requires Poppler `pdftotext`. It checks the exact campus roster and
all selected original and witness dollar cells, preserves signed source values,
and verifies the completed candidate by reading its leaves back. Qualification
applies to those two fields and the witnessed scope. It does not include
per-student denominators, award amounts, or statewide coverage inferred from a
district selection. The output and supplied store retain the case and scope
provenance, so use private locations for private project selections.
