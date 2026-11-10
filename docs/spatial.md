# Spatial Tools

The engine keeps a fast spatial index (STRtree, Shapely 2.x) for nearest-k and containment probes.

## Geographic source editions in 0.0.128

The source contracts select TEA's district product labelled 2026 for school year
2025–26 and its dedicated campus product labelled 2025 for school year 2024–25.
The latter is a newly acquired revision of the same edition, not a newer school
year. These independent geometry selections leave a reporting-year 2025 build's
tabular selection unchanged. Source configuration does not itself rebuild,
qualify or activate a data release; published 0.0.127 artifacts remain immutable.

The retained TEA item, service and layer references distinguish acquisition host
from upstream lineage. District metadata credits Texas Legislative Council GIS
and a TIGER/Line base with later updates; campus metadata identifies an AskTED
export without establishing its geocoding accuracy or methodology. Layer field
metadata is schema evidence, not a complete dictionary, masking policy or reuse
license. The original HTTP URL and download dates of the historical local inputs
remain unknown. New HTTP receipts do not establish that earlier lineage.

The campus revision's 118-column reader contract omits `USER_School_Status` and
`USER_Update_Date` from the historical 120-column contract. The historical schema
is retained. A missing source update date remains unknown; download time,
item-modification time and school-status date are separate observations.

The acquired district revision retains the pre-existing self-intersecting
polygon for district 208902. The campus revision retains 286 null source points.
Neither is silently repaired, geocoded, replaced with a centroid or filled with
zero. The separate native comparison reports four changed district boundaries
and distinguishes exact coordinates from a 1e-9-degree comparison criterion;
that criterion is not a claim of source accuracy. The
[canonical comparison receipt](https://github.com/adpena/teadata/blob/main/provenance/records/0f2e648dd64ebb46e0b5ffe1deb55e46f567798c1c215959eb493a4ed9a92e9c.json)
binds the comparison inputs, reports and toolchain. It is audit evidence, not a
data transformation or a raw-source acquisition receipt. Build and release
qualification must retain these limits.

## Using `coords`

Every campus exposes `coords` as `(lon, lat)`. This is handy for distance-based queries and pipelines:

```python
campus_q = (engine >> ("district", "101902")) >> ("campuses_in",)
c0 = campus_q.first()
k_nearest = engine.nearest_campuses(*c0.coords, limit=5)  # list[Campus]
```

## Charter and private campus containment

Use the engine's geographic helpers with a district selection:

```python
districts = engine >> ("district", "101902")
charters = engine >> ("charters_within", districts)
privates = districts >> ("privates_within",)
all_inside = engine >> ("within", districts)
including_border = districts >> ("within", None, None, False, True)
```

These use reported district geometry. For member campuses, use
`districts >> ("campuses_in",)`. General containment uses a district or a WGS84 Shapely Polygon or
MultiPolygon. It excludes border points by default. Pass `covers=True`
to `engine.campuses_within()` or use the final positional flag in the
tuple form to include them. Charter and private containment include the
boundary.

## Nearest Neighbors

```python
# Five nearest charters within 10 linear miles of a target point (private schools are automatically excluded)
pt = (-95.36, 29.83)  # (lon, lat)
nearest_charters = engine.nearest_campuses(
    pt[0],
    pt[1],
    limit=5,
    max_miles=10,
    charter_only=True,
)
```

## Roadmap for geographic cohorts and profiles

The planned library interface will let researchers discover geographic
entities and use their relationships to select district or campus cohorts.
Each relationship should carry its geographic edition, effective date,
assignment method and source. Join to published observations by public ID
and measurement period. Keep the period and geography vintage visible in
results, exports and app profiles.

| Geography | Source to qualify | Distinct relationships to expose |
| --- | --- | --- |
| County | [Comptroller property value study](https://comptroller.texas.gov/taxes/property-tax/pvs/index.php), [Census TIGER/Line archive](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-geodatabase-file.2013.html), and [TEA geography](https://tea.texas.gov/families-and-students/school-district-locator/geographic-data) | Taxable-property reporting, campus location, district office location, and school district boundary overlap |
| Education service center | [TEA ESC map](https://tea.texas.gov/school-and-district-leaders/other-services/education-service-centers/education-service-centers-map) and [AskTED definitions](https://tea.texas.gov/reports-and-data/school-performance/accountability-research/exploring-askted.pdf) and [archive index](https://tealprod.tea.state.tx.us/Tea.AskTed.Web/Forms/ArchivedSchoolAndDistrictDataFiles.aspx) | District services, PEIMS reporting, office region, and physical campus region |
| Texas House, Texas Senate, U.S. House, and State Board of Education | [Texas Legislative Council district plans](https://redistricting.capitol.texas.gov/Current-districts) and [plan history](https://redistricting.capitol.texas.gov/history) | Campus location and school district boundary overlap under a specified plan and effective date |
| U.S. Senate | [U.S. Senate representation](https://www.senate.gov/about/origins-foundations/senate-and-constitution/two-senators-per-state.htm) | Texas statewide geography, with both seats represented on one state profile |

Start with county and ESC cohort selection. Verify historical
availability edition by edition, including the corresponding TEA campus
locations, district boundaries and AskTED directory. Keep unmatched
locations and boundaries visible. A district may overlap several
geographies, and a campus may be outside its governing district's
boundary. The AskTED archive provides separate site and mailing city
fields. Site city can help distinguish identical names in a dated
directory. It does not establish geographic containment. For charters, use verified campus points and published district
relationships where available. The federal [NCES district boundary archive](https://nces.ed.gov/programs/edge/Geographic/DistrictBoundaries),
[school locations](https://nces.ed.gov/programs/edge/Geographic/SchoolLocations),
and [geographic relationship files](https://nces.ed.gov/programs/edge/Geographic/RelationshipFiles)
are historical candidates and cross-checks. Confirm the school year,
entity crosswalk and coverage before using a federal edition alongside
TEA data.

Build legislative profiles from the same library relationships. Keep the
effective representation date separate from the election date and the
observation year. A district number can describe different boundaries
under different plans. Label profiles as campus-location summaries or
boundary-overlap summaries and use official geography-level totals when
available. Area overlap cannot allocate student counts or establish where
students live.

Default each profile relationship to its latest qualified publisher edition
and show its date. Keep the property-tax county roster, site locations,
positive-area boundary overlap and the different AskTED ESC assignments
separate. A fresh edition for one relationship does not update the others.

Use `congressional_representation_plan(at)` with a timezone-aware datetime
when selecting from the two reviewed U.S. House plans. The suggested profile
note is “These boundaries reflect current representation. Texas’s new
congressional map takes effect for representation at noon Eastern on
January 3, 2027.” PLANC2193 applies before that instant. PLANC2333, used in
the 2026 election, applies from that instant under the reviewed schedule.
The constitutional start of the new terms is separate from the House’s
swearing-in schedule. [Texas Redistricting current districts](https://redistricting.capitol.texas.gov/Current-districts) · [Twentieth Amendment](https://www.archives.gov/founding-docs/amendments-11-27#amend20)

### Boundary sources retained for profiles

The retained [2026 Census TIGER/Line county layer](https://www.census.gov/geographies/mapping-files/2026/geo/tiger-line-file.html)
has all 254 Texas county polygons. The county-name reconciliation matches
all 254 Census GEOIDs to the Comptroller codes in the 2025 Final property
value study. The polygon and taxable-property relationship remain separate.

The [TEA School District Locator ESC layer](https://services2.arcgis.com/5MVN2jsqIrNZD4tP/arcgis/rest/services/ESC2024/FeatureServer/0)
has all 20 region polygons. Its region IDs match the served-region codes in
the spring 2026 AskTED archive. Served-region membership still comes from the
directory. The polygon only establishes map geography.

The [Texas Legislative Council shapefiles](https://redistricting.capitol.texas.gov/Current-districts)
for PLANH2316, PLANS2168, PLANE2106, PLANC2193, and PLANC2333 match the
constituency IDs in the corresponding Red-160 rosters. Together they have
272 plan-specific polygons. These checks establish boundary coverage and
identifier alignment. They do not assign school districts or campuses by
overlay. Profile deployment requires a separately pinned projection.

### Published school district rosters for legislative plans

The Texas Legislative Council's Red-160 workbooks list school district
subtotals by legislative constituency. The reviewed 2025–26 workbooks cover
Texas House, Texas Senate, the State Board of Education, and both congressional
plans. The library resolves the literal school names to six-digit TEA IDs
using the same-year Locator name and ID columns. County qualifiers in the
publisher's school names disambiguate identical names. A school district may
appear in several constituencies.

```python
from teadata.legislative_rosters import (
    LegislativePlan, RosterSource, read_legislative_school_rosters,
)

house, = read_legislative_school_rosters(
    store,
    district_locator=RosterSource(locator_receipt, locator_sha256),
    workbooks={LegislativePlan.TEXAS_HOUSE: RosterSource(workbook_receipt, workbook_sha256)},
)
ids = house.district_ids(1)
```

Use the `scripts` extra for workbook intake. Supply retained receipt IDs and
artifact hashes from the same provenance store. Each relationship retains the
original workbook label and one-based worksheet row. The 2025–26 school
boundary year, 2020 Census population vintage, and effective representation
date are distinct. The workbook percentages remain part of the literal
source label. The reader loads Locator attributes only. Use separately pinned
polygons to draw profile maps. [Legislative Council plans and resources](https://redistricting.capitol.texas.gov/Current-districts)

### Comptroller taxable-property county rosters

The 2025 Final School District Property Value Study lists school districts by
county for taxable-property reporting. The retained edition has 1,549 county
and district pairs across 254 counties and 1,013 school districts. The index
links to 253 county pages because Potter (188) and Randall (191) share one
page. Both county sections have their own worksheets. Districts such as
Amarillo ISD (188901) can appear in both sections.
[Comptroller 2025 Final index](https://comptroller.texas.gov/taxes/property-tax/pvs/2025f/index.php)
· [Potter and Randall report](https://comptroller.texas.gov/auto-data/PT2/PVS/2025F/188index.php)

```python
from teadata.comptroller_pvs import read_comptroller_county_roster

roster = read_comptroller_county_roster(
    store, index_receipt=index_receipt, county_page_receipts=page_receipts, tax_year=2025
)
counties = roster.for_district("248902")  # Wink-Loving ISD in Loving and Winkler
districts = roster.for_county("191")     # Randall County
```

The offline reader uses a provenance store with the index and its 253 exact
page captures. Pass a mapping from each three-digit Comptroller page code to
its acquisition receipt. The index receipt and each row's county receipt
locate the original bytes and download timestamps in the store. The row
includes the county code and name, district ID, publisher label with
whitespace normalized, and direct worksheet URL. Counties use Comptroller
codes. The selected calendar tax year is separate from a school year. The
reader supports Final tax years 2022 through 2025 and defaults to 2025.
Select the matching index and page receipts for each year.

The relationships describe what the publisher included in this study. A
valid district ID can be absent. Use separate sources for school-year
administrative assignments, campus locations, and district boundary overlaps.
Earlier published editions require their own source validation. The
2016–2018 reports group Potter and Randall under one appraisal district and
do not establish the separate county relationships. The current workflow
reads retained pages through the library. App profiles require a separately
qualified geographic release and projection.
[Comptroller archive](https://comptroller.texas.gov/taxes/property-tax/pvs/index.php)
· [2018 Potter–Randall report](https://comptroller.texas.gov/taxes/property-tax/pvs/2018f/188index.php)

### Archived administrative geography

Use `read_askted_directory()` with a local provenance store, the exact
acquisition receipt and its expected SHA-256. The reader verifies the
linked archive page, season, publisher-declared date and attachment
before binding rows. The date is the archive's label. Rows may reflect
later directory updates.

```python
from teadata.askted_directory import DirectoryField, read_askted_directory

directory = read_askted_directory(
    store, receipt, year=2026, season="fall", expected_sha256=sha256,
)
cohort = directory.cohort(
    "campus", field=DirectoryField.ESC_REGION_SERVED, value=13,
)
ids = cohort.ids
```

Choose the ESC field explicitly. Served, PEIMS and geographic refer to the
district's service, reporting and primary office assignments. Campus
selections inherit those values from the full parent district entry. The
same applies to administrative county. Site city and mailing city have
separate publisher fields for the district office and campus. The reader
keeps literal cells and logical CSV record numbers. It compares codes
after padding them, and reports missing or conflicting assignments
separately. See the [AskTED definitions](https://tea.texas.gov/reports-and-data/school-performance/accountability-research/exploring-askted.pdf)
and [TEA geographic report notes](https://rptsvr1.tea.texas.gov/adhocrpt/adgeo.html).

### Reported geography and calculated geography

Use `administrative_county_name` for the AskTED publisher field.
A future `locator_point_county_fips` field would identify the county
covering the published TEA Locator point. This is a calculated value.
Its provenance needs the exact point and polygon editions and the
containment rule. TEA may use a mailing address when a physical address
is unavailable. A point on a shared boundary, overlapping polygons or a
missing point requires an explicit status. District boundary overlap
answers a separate question. See the [TEA Locator methodology](https://tea.texas.gov/families-and-students/school-district-locator/geographic-data)
and [TxDOT county boundary service](https://maps.dot.state.tx.us/arcgis/rest/services/Boundaries/MapServer/1).

### Package a pinned archive

Run `python -m teadata.askted_leaves --help` for the local builder.
It uses a selected publisher archive and its checksum to produce a
standalone native release. Use `DataRelease.table(edition)` with an
exact fall or spring edition key. The archive date is the publisher's
as-of label. Each leaf keeps selected raw publisher cells and the
logical CSV record number. The district projection includes campus
rows from the original, and the campus projection includes records
with a campus ID. Repeated IDs and blank strings remain visible.
Catalog positions describe the projection order. The processing
receipt records the original CSV column numbers and ordered header
digest. For the grouped directory and its conflict handling, use
`read_askted_directory()`.

Candidate validation checks the archived input and the native
projection. Standalone candidate packaging leaves the current entity
release and its dataset selection alone. The generic raw-source
parity checker still reports this source as uncovered.

Before using a calculated campus county, verify that the coordinate
comes from a physical site. Retain an explicit unavailable status for
missing site addresses, missing points, and unverified input lineage.
The Census Bureau has a dated Texas county layer for January 1, 2025.
Its vintage does not establish the date or physical address of any
campus point. [Census ACS 2025 county layer](https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/37)


### November 2026 campus map layer

TEA Performance Reporting also publishes a November campus point layer.
Its ArcGIS item was modified on November 6, 2026. The item does not
specify a school year or establish physical-address lineage. The
statewide layer has 8,932 unique campus IDs with latitude and longitude
that match the point geometry. It overlaps 8,856 campus IDs in the
2024–25 Locator, including 269 whose Locator point is missing.
Among campuses with points in both sources, 610 pairs differ by more
than a kilometer. These differences do not establish which position
is more accurate.

Keep the November points as a separate publisher map edition for
source review. Before selecting a point for physical-site enrichment,
verify that it represents the campus site. Leave missing or unverified
sites unavailable and do not geocode mailing addresses.
[TEA November campus map layer](https://services2.arcgis.com/5MVN2jsqIrNZD4tP/arcgis/rest/services/districts_schools_2026_1105_C/FeatureServer/0)
