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
k_nearest = engine.nearest_campuses(*c0.coords, limit=5)  # list[(Campus, miles)]
```

## `within` and `charters_within`

Two common helpers:

```python
# All campuses within a district boundary (inferred from the district query)
inside = ((engine >> ("district", "101902")) >> ("within", None)).to_list()

# Charter-only campuses within the same boundary (excludes private schools)
charters = ((engine >> ("district", "101902")) >> ("within", None, True)).to_list()

# Equivalent imperative helpers
inside_alt = engine.within(aldine, items="campuses")
charters_alt = engine.charter_campuses_within(aldine)

# Private-school campuses (Campus.is_private) within the boundary
privates_alt = engine.private_campuses_within(aldine)
privates_query = repo >> ("privates_within", dist)
```

These use polygon containment and are validated by a built-in slow-path check to avoid false negatives.

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
