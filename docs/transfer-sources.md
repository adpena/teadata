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

Future structural preparation must validate that relationship before selecting
one canonical campus edge. It must retain independent reported directions and
raw row locators in its audit, and fail visibly on one-sided, duplicate, count,
status, or historical district-ID conflicts. The fresh original publisher campus
CSV, SHA `ea11eaf275f6c5abb79ead6213ded0223b17b70511798a4dc81f8abe03936372`,
receipt `785606f825deb15011a31afaa03bcdb17683021f9be062ef28b93fe0392ea8ee`,
has different bytes and requires its own structural preparation and comparison;
unchanged values have not been established by the older audit.
