# Geographic source-quality observations, November 2026

**Status: SOURCE IN PROGRESS — not deployed.** These observations concern the
retained TEA responses acquired on 2026-11-03 UTC and the prospective 0.0.128
source refresh. They do not change published 0.0.127 or establish release
qualification. Publisher edition, school year and HTTP download time remain
separate facts.

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

## Snyder ISD boundary

The raw district properties identify `208902` as **Snyder ISD** (`NAME`), with
`DISTRICT` equal to `208-902`. The native reader comparison reports the same GEOS
diagnostic in both the historical and newly acquired representations:

```text
Ring Self-intersection[-100.815812515043 32.7617015573616]
```

This identifies a geometric validity issue in the compared representations. It
does not establish that the legal boundary is wrong or determine whether the
issue originates upstream or during export/conversion. Comparing the native
ArcGIS geometry with the upstream representation is a reasonable technical
review step. No repair, simplification or replacement has been applied.

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
