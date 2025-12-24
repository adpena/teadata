# Spatial Tools

The engine keeps a fast spatial index (STRtree, Shapely 2.x) for nearest-k and containment probes.

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
