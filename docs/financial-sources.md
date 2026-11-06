# Financial workbook source candidates

The [PEIMS single-file archive](https://tea.texas.gov/about-tea/state-funding/state-funding-reports-and-data/peims-single-file-financial-data-downloads)
publishes separate district and charter account files. Historical packaging
varies. The FY2014 and FY2018 district budget ZIPs include a charter CSV
alongside the district CSV, and some historical charter budget links return
CSV directly. The current [single-file candidate reader](data-releases.md#peims-single-file-financial-budgets)
requires a ZIP containing exactly one member, the reviewed CSV. These
multi-member ZIPs and direct CSVs require reader support as well as review
of their headers, fiscal period, and population. The [publisher dictionary](https://tea.texas.gov/data-reports/financial-reports/school-finance-reports-and-data/reports-and-downloadable-data/datadictionary.pdf)
documents the account codes.

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

Publisher-label-only worksheets can opt into `header_identity: position` **in the existing crosswalk table**. This permits repeated literal header occurrences while ordered physical positions and canonical field IDs remain unique. Codes and labels both retain the exact publisher text; only canonical Python field IDs may distinguish occurrences, for example `_at_ad` and `_at_ca`. The native read explicitly skips the verified physical header, assigns reviewed canonical column names and selects verified physical indexes. It never treats Calamine's automatic duplicate-name mangling as publisher metadata. `catalog.header_records` retains each occurrence and position; code/label-to-name lookup still rejects ambiguity. Coded tables retain their default strict behavior, and TAPR's coded readers, comparisons, dictionaries and merges reject position mode.

`DatasetMetadata.metrics` supplies explicit per-field units and reference periods; a table with mixed enrollment, tax and fiscal periods can keep its table period unknown and declare the field periods separately. Numeric facts require reviewed units. Existing `SourceQualifiers` distinguishes SOF publication run/status and its actual physical row-run column from PEIMS actual/budget basis. Mixed SOF row runs need not equal the publication run, and a run column is not added to business keys to force its selection.

The existing canonical `MeasurementUnit` validates declared units. The numeric path rejects unknown/text units and retains fractional `fte` as FTE rather than coercing it to headcount. This capability does not invent an FTE column in PWR or establish a source's actual unit.

An individual table can explicitly opt into `uninterpreted_fields: [canonical_fact_name]` in its existing Config workbook declaration. Every selected name must be a real publisher-bound **fact** with an explicit `MetricDefinition` whose unit is unknown (`None`); identifiers, row keys, identity fields and known-unit facts are rejected. This option preserves the field directly as native Arrow strings, including range tokens, dots, negative values and nulls, with the existing sheet/row context. It does not call `measure`, infer a value/status, create duplicate raw columns, scale a number or reclassify a fact as identity. A null unit alone does not enable this behavior: without the explicit option, the numeric path still rejects unknown units. Unknown periods remain unknown, and unrelated fields still require their reviewed numeric units.

These uninterpreted facts remain discoverable in `catalog.variables(edition=key)` with `unit: null`, and `catalog.resolve_artifacts([(key, field)])` selects the string plus existing row keys and sheet/row context. There is no normalized raw/status support for that field. Do not treat a null, dot or range as zero, infer a suppression status, or apply numeric comparisons/scaling before reviewing its actual representation. In a schema3 candidate, the exact opt-in is accessible through the selected variable's `artifact_id`: `ref = catalog.artifact_ref(artifact_id)`, then `catalog.to_dict()["provenance"][ref.processing_id]["parameters"]["recipe"]["worksheet"]["uninterpreted_fields"]`. The contract binds that canonical recipe through `ref.contract["recipe_sha256"]`. This uses the existing processing record, not another field registry. The option is omitted from recipes when unused, preserving the default contract shape. Retaining a raw fact does not qualify its unit, period, definition or correctness.

Units must describe the **stored, unformatted values**. An Excel percent format can display a stored fraction as percentage points; currency display formats do not establish a monetary scale. This producer performs no style-based conversion, multiplier, percentage-point conversion, rounding or year-based transformation. Such conversions are unsupported. Keep a stored fraction as a reviewed ratio, or defer the field until its representation is established; do not label an unconverted fraction as percentage points. Literal tax/program header changes do not establish a conversion or equivalence.

The native reader is the optional `fastexcel` dependency, imported only when materialization is requested. Calamine reads complete admitted fact tables into Arrow strings. These are native reader representations, **not original OOXML numeric lexemes**. Original workbook bytes remain in canonical provenance; source sheet, physical row and publisher-column positions locate the originals. The producer inspects bounded OOXML structure, rejects formulas, error cells and merged fact cells, and never recalculates formulas. It does not perform Python Excel-cell value ETL or multiprocessing. Arrow operates on columns; the existing `values.measure` authority interprets at most 4096 distinct tokens per numeric field under the selected policy and unit.

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
