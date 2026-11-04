# Geographic source-quality observations, November 2026

**Status: source review in progress.** These observations cover TEA responses
acquired November 3–4, 2026 UTC for the prospective 0.0.128 source refresh.
Published 0.0.127 is the comparison release. Publisher edition, school year
and HTTP download time remain separate facts.

## Sources and scope

| Product | Declared period | Exact retained acquisition |
| --- | --- | --- |
| [TEA School Districts 2026](https://services2.arcgis.com/5MVN2jsqIrNZD4tP/arcgis/rest/services/School_Districts__2026/FeatureServer/0?f=pjson) | School year 2025–26 | [District HTTP receipt](https://github.com/adpena/teadata/blob/main/provenance/records/97be381420c20552e5baab45a68ed4b4c823ecff1d3945c075df1bd040796852.json), raw SHA-256 `8f92d9210ef23f6ab3aa5ec455960b11675dab5c8132fb0c9d0f66001cd9ceb4` |
| [TEA Schools 2024 to 2025](https://services2.arcgis.com/5MVN2jsqIrNZD4tP/arcgis/rest/services/Schools_2024_to_2025/FeatureServer/0?f=pjson) | School year 2024–25 | [Campus HTTP receipt](https://github.com/adpena/teadata/blob/main/provenance/records/ed8e190d5969357f60166ed061c55526eb09d4ef604d17b463fc82c77e1a0233.json), raw SHA-256 `4826bcd28674d043c3e02433a420104d47ac40ec920572df5cf21b1107377f4f` |

The receipts retain the exact successful query URLs and controls, including the
requested `outSR=4326`, and the actual download timestamps. TEA's district
metadata attributes upstream GIS work to the Texas Legislative Council; its
campus metadata identifies an AskTED export. Those statements do not establish
the original HTTP lineage of older local files, a legal boundary determination,
or independently verified campus coordinates.

On November 4, 2026 the official locator still used the 2025–26 district
polygons and 2024–25 campus points shown above. The public service and item
indexes checked that day exposed those as the latest dedicated statewide
locator layers. [TEA Geographic Data](https://tea.texas.gov/families-and-students/school-district-locator/geographic-data) ·
[Locator map configuration](https://www.arcgis.com/sharing/rest/content/items/15ce858b4ad5408480950db704fd455a/data?f=json)

## Snyder ISD boundary

The raw district properties identify `208902` as **Snyder ISD** (`NAME`), with
`DISTRICT` equal to `208-902`. The native reader comparison reports the same GEOS
diagnostic in both the historical and newly acquired representations:

```text
Ring Self-intersection[-100.815812515043 32.7617015573616]
```

A fresh native-projection Esri JSON query on November 4 returned a
self-intersection in the 1,682-vertex outer ring at easting `923691.1325` and
northing `1176792.9503`. The separate WGS84 GeoJSON query reproduced the
diagnostic above. The defect is present in the publisher's served geometry.
Its effect on the legal boundary is unknown. Hold the polygon from map
rendering pending review and preserve the source geometry.
[Native Snyder query](https://services2.arcgis.com/5MVN2jsqIrNZD4tP/arcgis/rest/services/School_Districts__2026/FeatureServer/0/query?where=DISTRICT_C%3D%27208902%27&outFields=DISTRICT_C%2CNAME&returnGeometry=true&f=json)

## Campus rows without source geometry

Of 9,739 campus-source rows, 286 have null geometry: **2.94%**. All 286 retain
`Status=U` and `Score=0`. [Esri documents `U` as an unmatched geocoding
result](https://doc.arcgis.com/en/arcgis-online/reference/geocode.htm). This is
consistent with an unmatched result; it does not identify why the TEA workflow
did not produce coordinates or show that every affected row is an error.

| Publisher district type | Null-point rows |
| --- | ---: |
| INDEPENDENT | 261 |
| CHARTER | 24 |
| COMMON | 1 |

| Publisher instruction type | Null-point rows |
| --- | ---: |
| REGULAR INSTRUCTIONAL | 263 |
| ALTERNATIVE INSTRUCTIONAL | 11 |
| DAEP INSTRUCTIONAL | 7 |
| JJAEP INSTRUCTIONAL | 5 |

There are 285 nonempty site street-address strings that do not match the
analysis's PO-box pattern. Of the 286 rows, 85 have PO-box-pattern mailing
addresses; all 85 also have nonempty, non-PO-box-pattern site addresses.
Across all 286, `IN_Address` matches `USER_School_Site_Street_Address` after
trimming, case normalization and whitespace normalization. The observed input
therefore does not support blaming the separate PO-box mailing addresses.

These are string comparisons, not independent address verification. They do not
prove that 285 addresses can be geocoded, that the schools are currently active,
or that every listing represents an operating physical campus. In particular,
regular-instruction labels do not establish current operational status.

The following values were checked directly in the retained GeoJSON properties;
each example has null geometry, `Status=U` and `Score=0`:

| Campus code | Source name | Site street address | Site city |
| --- | --- | --- | --- |
| 001906002 | NECHES H S | `1509 ACR 346` | PALESTINE |
| 015827007 | SST SCHERTZ | `20845 TECHNOLOGY AVE` | SCHERTZ |
| 015906001 | RANDOLPH H S | `BLDG 1225` | RANDOLPH A F B |

Randolph's building-style base address illustrates a plausible addressing
complication, not a demonstrated cause of its unmatched result. None of these
examples justifies guessing a coordinate, using a city centroid or plotting a
missing point at zero.

## 2026 accountability and the October map layers

Joining the 2026 preliminary accountability roster to the dedicated layers by
district and campus code yielded 1,016 usable district polygons across 1,202
reporting districts. Another 185 lacked a polygon, including 181 with charter
district evidence in the locator, and Snyder was invalid. Of 9,105 reporting
campuses, 8,756 had a usable locator point, 272 had a locator row with null
geometry and `Status=U`, `Score=0`, and 77 were absent from the 2024–25 locator
layer. The 286 null points in the preceding section use the entire locator
layer as their denominator. [2026 Accountability Rating System](https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/2026-accountability-rating-system)

TEA Performance Reporting's separate October 2026 map layers have 1,195
district points and 8,963 campus points. They provide display points for 346
of the 349 reporting campuses without a usable dedicated locator point.
Among reporting campuses with usable points in both layers, 3,180 pairs differ
by more than 100 metres and 608 differ by more than one kilometre. The October
item dates describe publication of the map layers. The underlying coordinate
vintage and derivation are unresolved. Hold these points as corroboration
pending review of location lineage and discrepancies, and keep district
display points separate from district boundaries. [October campus layer](https://www.arcgis.com/home/item.html?id=3ac5b236caca4979bbdf091a5f983058) ·
[October district layer](https://www.arcgis.com/home/item.html?id=bc436027f8e1424ba52e534cf5fb1d10)

## AskTED campus directory reconciliation

AskTED is TEA's public organizational directory for public schools and
districts, including charters and state-agency schools. TEA's search distinguishes
school type, instruction type, magnet status, residential facility status and
alternative education accountability. The separate Campus Enrollment Type
taxonomy concerns admissions and attendance, such as zoned, selective or
special assignment. Keep these attributes separate when binding source fields.
[AskTED description](https://tea.texas.gov/families-and-students/secure-applications/teal-applications-reference-program-descriptions) ·
[AskTED search](https://tealprod.tea.state.tx.us/Tea.AskTed.Web/Forms/SearchScreen.aspx) ·
[Campus Enrollment Type FAQ](https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/campus-enrollment-type-faq)

For performance measures, use the TPRS glossary's definitions and reporting
populations. Use the TPRS guide for its report navigation and release schedule.
AskTED's archive describes a separate organizational source.
[TPRS glossary](https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/comprehensive-tprs-glossary.pdf) ·
[TPRS guide](https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/tprs-user-guide-2026.pdf)

| 2025–26 AskTED export | Publisher archive label | Acquired November 4, 2026 UTC | Rows | Raw SHA-256 | Acquisition receipt |
| --- | --- | --- | ---: | --- | --- |
| Fall | October 31, 2025 | 08:52:05 | 12,442 | `dd3dd5779f45b77894657c3925bc4fbcde513a771686a54ef5c6ca50d9a381c2` | [Fall receipt](https://github.com/adpena/teadata/blob/main/provenance/records/fda9ca2e041f93e875def0d0b4253b36eb496a7c32437c5ca91fcbb49271819f.json) |
| Spring | March 31, 2026 | 08:55:37 | 12,481 | `f7df51bd645a0801d38bec8e4048a674f16b058b6310dd1fc6a569368b07e2db` | [Spring receipt](https://github.com/adpena/teadata/blob/main/provenance/records/d0c9ec40cc8738a3b7325e7d44ab6701eaad93b8b6561a65769d036c4386e735.json) |

Every row's `Update Date` is later than its archive label. Both files label
their enrollment columns October 2025 and omit Campus Enrollment Type.
Treat the archive label, row update time and enrollment reference month as
distinct observations. [Publisher archive](https://tealprod.tea.state.tx.us/Tea.AskTed.Web/Forms/ArchivedSchoolAndDistrictDataFiles.aspx) ·
[Acquisition contract](data-releases.md#rebuild-from-retained-inputs)

Each export has 82 repeated canonical campus codes. The same 48 overlap the
2026 preliminary accountability roster in both exports, with exactly two
AskTED rows per campus. Each pair already has identical raw identifiers with
the leading apostrophe and nine campus or six district digits. The duplicate
rows exist before normalization. The accountability roster has one row per
campus code.

All 48 pairs differ on NCES school ID, site address, grade range and
enrollment. Forty differ on school name. The populated NCES row is Active in
every pair, while its blank-NCES partner is Active in 29 and Under Construction
in 19. The reviewed public guidance supplies no precedence for these
conflicting rows. Preserve both and hold conflicting directory attributes
pending a publisher rule or source-specific resolution.

## Planned application behavior

When a focal campus lacks usable coordinates, a transfer-partners-only map may
be useful **only when actual retained transfer data and locatable partner rows
exist**. It should identify the missing focal location, show those partners
without a focal marker or connecting edges, and preserve the distinction
between observed transfers and geographic comparison. If that evidence is
absent, show a purposeful no-map explanation with source details. An unavailable
map must not imply zero schools or zero transfers. This behavior is agreed
source work in progress; implementation, runtime and deployment proof remain
separate.

## Audit evidence and limits

The [canonical comparison processing receipt](https://github.com/adpena/teadata/blob/main/provenance/records/0f2e648dd64ebb46e0b5ffe1deb55e46f567798c1c215959eb493a4ed9a92e9c.json)
binds the reader comparison's inputs, outputs and toolchain. It is an audit
record, not a data-transform parent or a raw-source acquisition receipt. Its
interpreted native comparison is not an exhaustive lexical GeoJSON diff.

The later address and instruction breakdown was recomputed directly from the
retained raw files by a separately archived analysis. Its
[canonical location-quality audit receipt](https://github.com/adpena/teadata/blob/main/provenance/records/a617b55c8a5e7c7cc427ee5c3e75c185a47e3518d488b58ca0b6164810897beb.json)
binds the exact source receipts, analysis code, toolchain, summary and all 286
observed null-point rows. The comparison receipt alone does not attest that
later analysis. Its string normalization is for these named comparisons only;
the archived row values and source coordinates remain unchanged.

The retained raw rows, acquisition receipts and these explicit analysis limits
own the evidence. This note is their explanation; it neither classifies all
missing points as confirmed publisher errors nor substitutes for current
release and application qualification.
