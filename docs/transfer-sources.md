# Transfer source intake

`teadata.scripts.refresh_transfers acquire` retains original TEA regional district
transfer CSVs through `AcquisitionClient`, ordered `acquire_form` controls, and
`ProvenanceStore`. It does not activate configuration, prepare a table, change an
engine, or qualify a report's values. The command is intended for explicit finite
batches, with one sequential request per selected region and no redirects or
automatic retries. The maximum byte cap is 8 MiB per response. The timeout passed
to urllib is at most 20 seconds for socket operations; it is not a hard request
or whole-batch wall-clock deadline. An owning caller may impose an additional
batch deadline.

The SHA-pinned original [regional selector](https://rptsvr1.tea.texas.gov/adhocrpt/Standard_Reports/Transfer_Reports/region_district_transfer_report.html)
owns the offered year labels, region IDs, GET action, and ordered controls. Year
2025 is labeled 2024-2025 in the retained form; a year control alone is not a
publication or observation date. Unknown original/revision/license dates stay
unknown. A changed form requires review, rather than guessed controls.

```sh
uv run --locked --offline python -m teadata.scripts.refresh_transfers acquire \
  --store /absolute/path/to/provenance \
  --form-record 31108c9b98a10b765108d1a82b24e2499dd6d9945d8fa4f90812261463f600a8 \
  --form-sha256 f03735f20d82ed6e86fa4b07eea90d705e2efc7d56833ddb02a9faa1dea84b60 \
  --year 2025 --all-offered-regions --output /absolute/path/to/new-batch
```

Alternatively supply `--regions 01 13` with exact offered IDs. The store must
already contain the specified form's complete canonical receipt/object closure.
`--resume REGION RECEIPT` reuses only an explicitly named original HTTP receipt
matching the selected form, year, region, period, and transport bounds, with
verified retained bytes. A fresh output directory is required. There is no
automatic discovery of old downloads or continuation of a failed request.

`BATCH_INDEX.json` is a private, atomically updated projection of original
receipts. It remains `HOLD` after failure or interruption; interruptions are
persisted and immediately re-raised, and later regions remain pending. `PASS`
means all selected acquisitions passed the exact request and CSV
media-type checks. It does not certify schema, counts, masking, completeness of
district coverage, or equality to another report. A rejected response remains a
truthful original acquisition, with its ID retained in the HOLD index. Original
CSV headings and footnotes are not removed. CSV decoding and structural
preparation are separate; a missing response charset must not be silently
reported as a publisher-declared encoding.

## Meaning and independent report grains

The retained [TEA transfer definitions](https://rptsvr1.tea.texas.gov/adhocrpt/Standard_Reports/About/about_transfer_reports.html)
describe residence-area to actual attendance, not a student's previous school or
a longitudinal move. District reports concern cross-district transfers; campus
reports can include different campus residence areas within one district.
Charter schools are not residence campuses/districts in these reports, so charter
Out is publisher not applicable. Private schools are excluded. These definitions
are retained by acquisition
`e853850f4f9931afc5e755de31ff3f78cfd2fa5c1f9b21f6049152d3728f40c1`.

Publisher district TOTAL rows must remain distinct from pair details. Suppressed
children must not be summed to replace a published total, or subtracted from it
to reveal a suppressed value. The existing `campus_transfers` value policy retains
`-999` as suppressed and `.` as missing; the publisher notes allow larger counts
to be masked to prevent inference. The separately labeled campus-pair midpoint
scenario does not establish district totals or permit a second imputation.

## Retained campus directional audit

TEA offers statewide campus CSVs beginning in 2020–21. The 2020–21 and
2021–22 originals pad titles, blank records, and footnotes to the full
CSV width. The reader accepts that exact historical shape, preserves
the literal annotation records and source coordinates, and verifies
the title and masking text before preparing the data. A structural
read leaves directional equivalence as a separate qualification step.

The older retained, locally observed campus CSV has source SHA
`acd304693b0e93865d12add2909bd015e7c5f1e4c04370a75202ad4117a35cce`
and acquisition
`d8fb8c735fdc89749c1593c494c8f73f6c8ef966ab885771af68717650e3871b`.
Its original HTTP observation date remains unknown. Root's bounded native audit
report SHA
`18781594bfda940ab89f0469ec5d571720a0e27187e131fa14a6ee592ce8b7d6`
found 150,365 independently reported rows in each direction: 14,215 observed and
136,150 suppressed pairs matched In against inverted Out. It found no one-sided
pairs, duplicates, count/status or raw count-token differences, historical
district-ID mismatches, or invalid endpoint IDs. Another 15,927 blank-direction
group-200 rows were missing-value separators. This is parsed directional
equivalence for that source, not byte identity of every field or proof about a
newer response.

Canonical-edge qualification validates that relationship before selecting
one campus edge. Source preparation retains independent reported directions and
raw row locators in its audit, and fails visibly on one-sided, duplicate, count,
status, or historical district-ID conflicts. The fresh original publisher campus
CSV, SHA `ea11eaf275f6c5abb79ead6213ded0223b17b70511798a4dc81f8abe03936372`,
receipt `785606f825deb15011a31afaa03bcdb17683021f9be062ef28b93fe0392ea8ee`,
has different bytes. Its separate retained preparation and old/new comparison
are documented below; they do not turn the older audit into proof of raw-byte
identity.

## Retained 2026-11-03 preparation results

The two district preparations each consumed all 20 acquired regional originals.
Their observed table coverage was:

| Publisher school year | Table rows | Detail rows | Publisher TOTAL rows | Reporting district IDs observed |
| --- | ---: | ---: | ---: | ---: |
| 2023-2024 | 41,801 | 35,192 | 2,203 | 1,206 |
| 2024-2025 | 43,179 | 36,564 | 2,205 | 1,208 |

Both passed structural/year/region checks with zero duplicate detail keys,
duplicate total keys, or reporting districts assigned to multiple report
regions. Publisher totals were retained independently, without computation from
masked details. These counts describe the responses' observed reporting IDs;
they do not prove equality to every expected LEA or to the current entity catalog.
The canonical processing records are
`20b4640c31a135b55c435725f50f75d8798f769ad2790a20b10447d182cfd1da`
for 2023-2024 and
`a79a1f84b46c02b4929f3ec0dc942fa2adce8d6e3ca9f60539bf6ede31549fb5`
for 2024-2025.

The fresh campus original prepared 316,657 table rows. Its independent In and
Out reports each contained 150,365 pairs: 14,215 observed and 136,150 suppressed.
The direction gate found zero unknown-count rows, duplicate or one-sided pairs,
count/status or historical district-ID mismatches, or raw count-token differences.
Across 300,730 paired detail observations, the old-prepared/fresh-original
comparison found zero one-sided rows or changed counts, statuses, historical
district IDs, raw count tokens, names or other compared non-ID fields. Raw ID
lexemes differed while their normalized IDs remained equal. Original quoting,
headings, footnotes and byte identities remain distinct; non-detail row contents
are outside that pair-detail comparison. The campus processing record is
`87624230d536d19f63a4583ce62eacbaf6399d79607ecd20784f7138a49e0e2a`.

All three preparations passed native readback equality. This verifies the
retained source projection, not every field's external correctness or exhaustive
LEA coverage. Their reports retain `field_correctness_qualified: false` and
`activation: false`; these source preparations are not an activated release.

The independently acquired 2023-2024 campus original has SHA
`40df05470d7a2cf117ec793ab24a85c5e70b62338b6ca1785a6a8643c354ef34`
and acquisition
`5349c45b292fc5d2e4d564412a9917ace24100b1883afdb1c34f3bd636bc690b`.
The original preparation
`7cbae23f47643cc952635d95eb065dff8e3d4073328954f8305d5645e78e1398`
parsed 297,734 rows and found 140,975 pairs per direction, with zero duplicate,
one-sided or historical district-ID conflicts. Four oriented pairs had different
masking between the reported In and Out views. Root's original-row audit, SHA
`4d8efd2adff35e3420114b830af05819531776cb8f226f81563de6b7d466f3a4`,
verified all eight source rows. These observations establish a reporting-view
difference, not its cause or a publisher error. A count visible in one view must
not fill a suppressed cell in the other. That immutable preparation remains
HOLD; the source-only retention behavior below requires a new preparation and
does not change the original record or activate this edition.

## 2025–26 source preparation

All 20 regional district originals prepared successfully. They contain 44,327
rows, including 37,736 detail rows and 2,197 publisher totals, across 1,201
reporting district codes. These are the district reports' own observations.
The processing receipt is
`02977bde453c093ff624ff81537cf9c26d163aed6434e7045239273e1a3e0b18`.
[TEA regional district reports](https://rptsvr1.tea.texas.gov/adhocrpt/Standard_Reports/Transfer_Reports/region_district_transfer_report.html)

The 2025–26 statewide campus original uses `TRANSFERS_IN_OUT` where the
2024–25 original uses `TRANSFERS_IN_OR_OUT`. The preceding 13 headings,
report titles and masking footnotes agree. The 2026 reader retains the literal
header and projects its counts through the same masking policy. The original
acquisition is `f7feea636a5bb61447566389cc135413798918cd70ec2bd3200058460fdf417d`,
with raw SHA-256
`40726ad2d508f383cddc09861e3efbbeabf9e40f1fbd0687a46bf59ea1a7a3f0`.
[TEA statewide campus report](https://rptsvr1.tea.texas.gov/adhocrpt/Standard_Reports/Transfer_Reports/campus_transfer_report.html)

The campus preparation retains 321,375 rows across 8,850 reporting campuses.
Its direction check found 152,781 unique pairs in each view, with 14,864
observed and 137,917 suppressed pairs agreeing across views. The check covers
pair identities, report-time districts, count values, statuses and raw count
tokens. The processing receipt is
`7d651084d10803842bc7557f16ba604140051423967956c88952675c3e560798`.
Both preparations passed native readback and remain at the source preparation
stage. Field-level qualification and release activation are pending.

## Offline structural preparation

`prepare` consumes either explicitly pinned source acquisition/SHA pairs or one
SHA-pinned completed regional batch index. District parsing is independent of
partitioning: an independently acquired statewide district original, if found,
can use `--input` without a regional index. No speculative statewide endpoint is
assumed. All 20 regional originals have been acquired for each year from
2022–23 through 2025–26. Compare their district coverage separately against the expected
scope. Individual district examples provide additional publisher witnesses.

The 2020–21 regional district originals put the six-digit ID in parentheses at the
end of the label. Preparation reads that final ID as text and retains the
full original label. A shorter or missing ID fails preparation.

For a regional batch:

```sh
uv run --locked --offline python -m teadata.scripts.refresh_transfers prepare \
  --store /absolute/path/to/provenance --grain district --year 2024 \
  --batch-index /absolute/path/to/2024-batch/BATCH_INDEX.json \
  --batch-index-sha256 64a8075743db3719c0ef2fe60873429858f13b987cfbea51bf6c03296ec8d9bd \
  --encoding utf-8 --policy-config /absolute/path/to/teadata_sources.yaml \
  --policy-config-sha256 ACTUAL_CONFIG_SHA256 --policy campus_transfers \
  --output /absolute/path/to/new-preparation
```

The explicit codec is an operator selection when retained MIME has no charset;
the preparation records the original header and selection basis. A conflicting
MIME/argument selection fails. The shared byte-framing primitive supports UTF-8,
ASCII, ISO-8859-1 and Windows-1252; UTF-16/32 and other unsupported newline
encodings fail visibly. UTF-8 is not inferred from an absent header. The command
above is a recipe; the retained runs reported above passed strict UTF-8 decoding
as an explicit operator selection. Their original CSV MIME headers did not
declare a charset, so this is not a publisher-declared encoding claim.

For the fresh campus original, use `--grain campus --year 2025 --input
785606f825deb15011a31afaa03bcdb17683021f9be062ef28b93fe0392ea8ee
ea11eaf275f6c5abb79ead6213ded0223b17b70511798a4dc81f8abe03936372` and
`--compare-legacy-input
d8fb8c735fdc89749c1593c494c8f73f6c8ef966ab885771af68717650e3871b
acd304693b0e93865d12add2909bd015e7c5f1e4c04370a75202ad4117a35cce
--compare-encoding utf-8`. The separately admitted legacy input has a header-first
layout; its removed notes and unknown acquisition dates are not reconstructed.

The reader preserves all original 10 district or 14 campus columns as strings.
Table grain and physical report scope are distinct: an explicit regional
district selection requires `Region Level Transfer Report`; an individual
district input requires `District Level Transfer Report`, and a campus input
requires `Campus Level Transfer Report`. A different title is not silently
accepted as another scope.
It checks the selected school-year label and, for regional inputs, the actual
selected region. A bounded structural CSV scan retains original titles, blank
records, exact footnotes and physical/logical row coordinates; Arrow performs
whole-column identity and numeric/status interpretation. District combined
labels require an anchored six-digit identifier; the raw label stays intact.
Unexpected headers, row kinds, malformed IDs, invalid counts and unknown notes
fail visibly. Zero remains observed zero. Suppression never becomes zero or an
estimate. Publisher TOTAL rows remain independent of detail rows.

Input admission is 64 MiB per retained source, with 1 MiB logical records,
8 MiB native batches and 2 MiB annotations. These are buffer/input admissions,
not a whole-process memory guarantee. Campus direction validation and comparison
use native joins; they do not hydrate the entity engine or geocode records.

`prepared-source.parquet` retains every table row, including separators and
published totals, plus namespaced normalized IDs/statuses and row locators.
`transfer_source_input` indexes the corresponding original receipt in
`PREPARATION.json`; `ANNOTATIONS.json` retains the original non-table text and
its raw-record hashes. Native readback must equal the written source projection.
The campus gate rejects duplicate, one-sided, unknown-count or conflicting
count/status/historical-district observations. Raw count-token spelling changes
are reported separately from interpreted equality. The old/new comparison uses
normalized direction/pair keys, reports one-sided rows and changed counts,
statuses, historical districts and other raw fields, and distinguishes identifier
padding changes. It does not infer unchanged values from the prior audit.

Readback-verified observations and their validation/annotation reports are retained with
`ProvenanceStore.record_processing`, using actual source receipts and captured
policy/index/schema inputs as typed parents. Structural reads and legacy
comparison reads retain the attempted input, error and parsed-prefix status in
`PREPARATION.json` on failure. Structural/schema failures or failed native
readback retain only a HOLD diagnostic processing record, without a
`prepared_source` output; a parsed prefix is not a prepared dataset.

New preparations distinguish two narrow outcomes in the same report and
processing validation. `prepared_source_qualified: true` means every selected
original passed structural parsing and the full prepared table passed exact
native readback. It does not qualify external field correctness.
`canonical_edges_qualified` reports the campus direction gate's agreement:
`true` for PASS, `false` for HOLD, and `null` when no campus edge gate ran
(including district reports). It does not override a readback failure.
An edge HOLD retains the complete independent directional observations,
annotations, raw tokens, statuses and locators when their preparation passes
readback; it does not select or reconcile one edge. Overall status remains
HOLD and the CLI exits 2. An optional legacy-comparison failure also retains
valid new-source observations, and cannot replace an earlier edge HOLD or make
it PASS. Both outcomes retain `field_correctness_qualified: false` and
`activation: false`. Existing successful campus preparations remain
`prepared`/PASS; district source-grain validation and native admission remain
unchanged. Earlier immutable preparation records keep their original contracts
and are not rewritten to add these flags. Original
acquisition events, timestamps and metadata remain unchanged. The processing
identity includes the shared `source_csv.py`; the TAPR comparison also fingerprints
that dependency explicitly after its extraction.

These outputs remain unpublished source preparations. Acquisition-only metadata
fields cannot be used for processing IDs. The explicit native step below binds
the existing typed processing/artifact authority; it does not activate datasets,
change the default source configuration, or qualify a release scientifically.

## Offline district native candidates

`refresh_transfers materialize` accepts exact annual preparation processing IDs
and a reviewed, SHA-pinned source Config. The packaged `teadata_sources.yaml`
includes source discovery for the 2022–23 through 2025–26 district reports.
The builder requires every offered region from the preparation's captured
index/form, in source order. Each annual batch contains all 20 regional originals.
The result contains one statewide district logical table and native leaf per
selected year. Regions are physical acquisition partitions.

The reviewed source profile must enumerate every original SHA in the ordered
header contract, retain genuine schema/masking-reference acquisitions, identify
the exact school-year period, and declare the one transfer-count field's units.
The publisher's ten literal headers are the only research-variable entries.
Their canonical fields can bind the existing normalized values; unchanged
columns receive lossless canonical renames. All 23 prepared columns and all
detail, publisher-total and separator rows remain. The single `column_support`
contract retains raw/status/locator columns without inventing publisher headers.
Selected counts include their raw token, status and row kind automatically;
extracted district IDs retain the original combined name/number labels.

The store must contain the prepared artifacts and their complete original
receipt and object closure. For the retained inputs, use the packaged source
Config and its exact SHA-256, and choose a new candidate directory:

```sh
uv run --locked --offline python -m teadata.scripts.refresh_transfers materialize \
  --store "$TRANSFER_PROVENANCE_STORE" \
  --config teadata/teadata_sources.yaml \
  --config-sha256 "$TRANSFER_REVIEWED_CONFIG_SHA256" \
  --prepared-input 2023 f884f3ebdd177607d6ac2b417dec9664c19a98349bb775b0ce22fcea06381f37 \
  --prepared-input 2024 20b4640c31a135b55c435725f50f75d8798f769ad2790a20b10447d182cfd1da \
  --prepared-input 2025 a79a1f84b46c02b4929f3ec0dc942fa2adce8d6e3ca9f60539bf6ede31549fb5 \
  --prepared-input 2026 02977bde453c093ff624ff81537cf9c26d163aed6434e7045239273e1a3e0b18 \
  --output "$TRANSFER_NEW_CANDIDATE_DIR"
```

For 2022–23, all 20 regional originals had the same ten ordered headings and
the same masking notes. The local candidate retained 40,703 source rows,
including 34,076 directional details, 2,209 published totals, and 4,418
separators across 1,209 reporting districts. Offline source parity checked all
rows and found no differences through preparation and the native table. Each
of the 17,038 incoming details had a reciprocal outgoing detail with the same
raw count or masking token. These checks describe the regional reports and do
not establish full LEA roster coverage. Separately, all 201 rows in the
individual reports for Austin ISD, Hutto ISD, and Harmony Public Schools –
Houston North matched in source order, including the original count tokens
and published totals. Those three witnesses were retrieved on November 8,
2026, through TEA's [District Transfer Report selector](https://rptsvr1.tea.texas.gov/adhocrpt/Standard_Reports/Transfer_Reports/district_transfer_report.html).
The edition remains unpublished.

To read the candidate directly, open its manifest and choose an explicit
edition and field. The catalog resolves the leaf and includes the raw count,
status and source row locator alongside the normalized count.

```python
from teadata.dataset_metadata import EditionKey
from teadata.release import DataRelease

release = DataRelease("path/to/candidate/release.json", offline=True)
entry, = (row for row in release.catalog.editions(dataset="district_transfer_reports")
          if row["edition"] == "2024")
edition = EditionKey(entry["dataset"], entry["edition"], entry["revision_sha256"],
                     entry["scope"], entry["table"], entry["grain"])
selection, = release.catalog.resolve_artifacts([(edition, "student_transfer_count")])
rows = release.analytical_dataset(selection["artifact"].identity).to_batches(
    columns=list(selection["columns"]))
```

Historical preparations are admitted against their own recorded producer/CLI,
policy, toolchain, validation and exact outputs; equality with current global
source code is not required. Missing original lineage remains unknown. The new
leaf processing record has the original preparation and reviewed source-control
acquisitions as typed parents. `ArtifactRef.processing_id`, `BuildRecorder` and
the ordinary edition publisher create an adjacent unpublished candidate with
composite `(transfer_source_input, transfer_source_record)` keys. They create no
parallel provenance DAG, active release pointer or engine state.

Admission is 2 MiB for each control/report JSON, 16 MiB per prepared artifact,
8 MiB for the compact row-key index, 4096 rows per native batch, and 8 MiB per
emitted batch. These are input/buffer checks, not a whole-process RSS or duration
guarantee. Native readback must equal the accepted preparation after only the
declared column projection. The earlier preparations' `field_correctness_qualified`
value is false; PASS/readback and native materialization do not upgrade it or
establish all-LEA coverage. The packaged entries expose the source contracts
without adding district rows to the default engine or app. Engine/app consumer
cutover, field correctness and release qualification remain separate reviewed steps.

## Compare campus observations with district reports

`compare-campus-district` compares cross-district directional details from a
retained campus preparation and an exact manifest-pinned district release. It
verifies the campus preparation's original provenance and the selected district
leaf. The school years must agree.

```sh
uv run --locked --offline python -m teadata.scripts.refresh_transfers compare-campus-district \
  --store "$TRANSFER_PROVENANCE_STORE" --year 2026 \
  --campus-preparation 7d651084d10803842bc7557f16ba604140051423967956c88952675c3e560798 \
  --district-release "$DISTRICT_RELEASE_MANIFEST" --district-pin "$DISTRICT_MANIFEST_SHA256"
```

It excludes same-district campus details, publisher totals, and separator rows.
It sums a directional district pair only when every published campus detail
and the district count are observed. Coverage categories are mutually
exclusive. When both levels are suppressed the shared group is counted under
district suppression. The check does not reconstruct masked counts.

| School year | Exact observed groups | District-only masked groups |
| --- | ---: | ---: |
| 2023–24 | 372 | 6 |
| 2024–25 | 420 | 0 |
| 2025–26 | 418 | 12 |

`PASS_OBSERVED_TRANSFER_ROLLUP` covers only this visible subset. A disagreement,
a campus-only cross-district group, a district-only observed group, or no
comparable groups yields `HOLD`. The 2023–24 campus preparation's canonical
edge status remains `HOLD` because of the directional masking differences
described above. The comparison can use its separately accepted source rows
without upgrading that status.


### Campus native source check

Use `scripts/check --source-parity` with the exact native candidate
manifest, retained source store, and campus edition ID. The streaming
check compares each publisher column and its typed projection against both
the prepared table and native leaf. It checks recorded historical campus
locators against the pinned fall AskTED directory. An independent recheck
of the companion registry's missing campus roster is a separate step.

If `materialize-campus` finds unresolved historical endpoints, it emits a
JSON `HOLD_HISTORICAL_TRANSFER_ENDPOINTS` result and exits with status 2.
The result groups endpoint IDs by reason and gives distinct affected
detail-row counts, with reporting and partner counts shown separately.
The candidate directory is created only after admission succeeds.

The 2024–25 source view can also carry separate AskTED obsolete dates and
logical record locators for the reporting or partner endpoint. A dated entry
in the pinned All Public Schools Closed report must match the fall directory's
campus ID and name. The report's print date, its obsolete date, the fall
archive's as-of date, and the transfer measurement period describe different
events. The obsolete date supplies historical context. It does not prove why
an endpoint was absent from the companion registry, change any transfer
count, or change the current campus roster. Empty obsolete-date columns mean
the workflow attached no qualifying evidence from this report.

### Publisher values, calculated sums and estimates

Keep a TEA-published pair count or total distinct from a TEA Data
calculation. A sum of observed charter-destination counts is derived
from the available pair counts. It can omit transfers to masked
destinations. The estimated total adds a midpoint of five per eligible
masked destination at the model's grain. Cite both the publisher inputs
and the [estimation methodology](https://github.com/adpena/teadata/blob/main/teadata/methodologies/charter-transfer-midpoint.md)
where the estimate appears.
