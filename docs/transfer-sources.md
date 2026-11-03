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

Structural preparation validates that relationship before selecting
one canonical campus edge. It must retain independent reported directions and
raw row locators in its audit, and fail visibly on one-sided, duplicate, count,
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

## Offline structural preparation

`prepare` consumes either explicitly pinned source acquisition/SHA pairs or one
SHA-pinned completed regional batch index. District parsing is independent of
partitioning: an independently acquired statewide district original, if found,
can use `--input` without a regional index. No speculative statewide endpoint is
assumed. All 20 regional originals have been acquired for each of 2023-2024 and
2024-2025; acquisition alone does not prove every expected district has report
rows. Any E3 district examples are additional goldens, not the statewide scope.

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

Successful outputs and their validation/annotation reports are retained with
`ProvenanceStore.record_processing`, using actual source receipts and captured
policy/index/schema inputs as typed parents. Structural reads and legacy
comparison reads retain the attempted input, error and parsed-prefix status in
`PREPARATION.json` on failure. Structural or validation failures retain a HOLD
diagnostic processing record with no `prepared_source` output; a parsed prefix
is not published as a prepared dataset. Original
acquisition events, timestamps and metadata remain unchanged. The processing
identity includes the shared `source_csv.py`; the TAPR comparison also fingerprints
that dependency explicitly after its extraction.

These outputs are unpublished source preparations. `BuildRecorder` currently
requires an acquisition matching the exact bytes of a selected retained input;
its acquisition-only metadata fields cannot be used for processing IDs. A later
activation must explicitly bind the existing typed processing/artifact authority
and pass release qualification. This command creates no parallel release
manifest, source catalog, engine state or active configuration.
