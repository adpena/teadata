# Data Models

## `District`

Key fields & behavior:

- `id: UUID`
- `name: str`
- `district_number: str` — normalized **six-digit**, zero-padded, may be stored with a **leading apostrophe** to match TEA data where needed.
- `rating: Optional[str]` — canonical; can be synced from enrichment via alias
- `boundary: shapely.Polygon | MultiPolygon`
- **Computed / accessors**
  - `campuses: list[Campus]` — all campuses physically inside the boundary (spatial join is precomputed during repo build)
  - `nearest_campuses(...)`
- `charter_campuses`: campuses flagged as charters (excludes private schools)
  - Dynamic attributes from enrichment (e.g., `overall_rating_2025`)

## `Campus`

- `id: UUID`
- `campus_number: str` — normalized **nine-digit** TEA code (zero-padded; apostrophe-safe)
- `name: str`
- `charter_type: str`
- `is_charter: bool`
- `is_private: bool` — True for private schools loaded from the new dataset
- `enrollment: Optional[int]`
- `type: Optional[str]` — e.g. `"OPEN ENROLLMENT CHARTER"`
- `point: shapely.Point`
- `coords: tuple[float, float]` — `(lon, lat)` for convenience
- `district_id: UUID` — back-link to parent
- `district: District` — property resolving the back-link
- Dynamic enrichment fields (e.g., `overall_rating_2025`)

> **Normalization helpers** exist for both district/campus numbers; they’ll accept ints, strings with/without apostrophes, and coerce to canonical text forms.
