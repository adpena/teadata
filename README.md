# TEA Data Engine

The **TEA Data Engine** is a Python-based toolkit for working with Texas public education datasets.  
It provides object-oriented access to rich, spatially-aware data about school districts and campuses, 
making it easy for developers to explore, enrich, and analyze educational data.

---

## 🚀 Quick Start

Clone the repository and install dependencies:

```bash
git clone https://github.com/adpena/teadata.git
cd teadata
pip install -r requirements.txt
```

A cached snapshot (`.cache/teadata.pkl`) is already included in the repository so you can start immediately without downloading source datasets.

```python
from classes import DataEngine

# Initialize engine from cache
engine = DataEngine()

# Explore a district by number
aldine = engine.get_district("101902")
print(aldine.name)                  # Aldine ISD
print(aldine.overall_rating_2025)   # Example: 'C'

# Explore campuses
for campus in aldine.campuses:
    print(campus.name, campus.rating)
```

---

## 📊 Models

### `District`
Represents a Texas school district with properties:
- `id` (UUID)
- `district_number` (normalized string, zero-padded where needed)
- `name`
- `enrollment`
- `type` (e.g., charter, traditional ISD)
- `rating` (canonical rating, plus enrichments like `overall_rating_2025`)
- Spatial geometry (district boundaries)

Methods and behaviors:
- `.campuses`: returns all `Campus` objects within this district boundary
- `.nearest_campuses()`: query by location to find closest campuses
- `.enrich()`: dynamically attach new attributes from external datasets

### `Campus`
Represents a school campus with properties:
- `id` (UUID)
- `campus_number` (normalized to nine-digit TEA format with optional leading apostrophe)
- `name`
- `enrollment`
- `type`
- `rating` (canonical or enriched, e.g., `overall_rating_2025`)
- Spatial geometry (campus location point)

Methods and behaviors:
- `.district`: reference to parent District
- `.enrich()`: dynamically attach new attributes from external datasets

---

## 🧠 Why Object-Oriented?

This engine uses Python’s OOP and dynamic attribute resolution to make working with education data *Pythonic* and intuitive. By encapsulating data and behavior within `District` and `Campus` objects, the TEA Data Engine enables a rich, expressive interface that feels natural to Python developers.

### Dynamic Attributes & Enrichment

Attributes such as enrollment, ratings, or finance data can be **dynamically attached** to objects at runtime via the enrichment system. This means you can access newly added data just like any native attribute without modifying class definitions or managing complex joins:

```python
print(district.overall_rating_2025)   # Access enriched accountability rating
print(campus.finance_per_pupil)       # Access enriched finance data
```

This flexibility supports iterative workflows where new datasets become available, and the engine adapts seamlessly.

### Operator Overloading for Natural Comparisons

Districts and campuses implement comparison operators based on meaningful attributes like enrollment or ratings. This allows straightforward and readable comparisons:

```python
if campus1.enrollment > campus2.enrollment:
    print(f"{campus1.name} has more students than {campus2.name}")

if district1.rating == 'A' and district2.rating != 'A':
    print(f"{district1.name} outperforms {district2.name}")
```

This operator overloading reduces boilerplate, enabling idiomatic expressions that read like natural language.

### Query Chaining with the `>>` Operator

A unique feature is the **`>>` operator**, enabling **query chaining** to compose filters, transformations, and traversals in a clean, pipeline style. This approach enhances readability and simplifies complex queries:

```python
# From a district, get campuses, then filter by rating 'A'
top_campuses = aldine >> 'campuses' >> (lambda c: c.rating == 'A')
for campus in top_campuses:
    print(campus.name, campus.rating)

# Chain multiple filters: campuses with enrollment > 500 and rating 'B' or better
filtered = aldine >> 'campuses' >> (lambda c: c.enrollment > 500) >> (lambda c: c.rating in ['A', 'B'])
```

Under the hood, this chaining uses Python's `__rshift__` method to apply successive operations to collections or attributes, creating expressive, concise workflows without intermediate variables or nested loops.

### Spatial Queries Made Easy

Because each object includes spatial geometry, you can perform geospatial queries naturally:

```python
# Find nearest 5 campuses to a coordinate
nearest = engine.nearest_campuses(lon=-95.37, lat=29.76, k=5)
for campus, distance in nearest:
    print(f"{campus.name} is {distance:.1f} meters away")

# Get campuses inside a district boundary
campuses_in_district = aldine.campuses
print(f"{aldine.name} has {len(campuses_in_district)} campuses")
```

Spatial operations are integrated into the object model, enabling location-based analyses without external GIS tools.

### Real-World Patterns Inspired by `load_data.py` and `load_data2.py`

In practice, these patterns enable workflows like:

- Accessing enriched attributes transparently:

```python
district = engine.get_district("101902")
print(district.overall_rating_2025)  # Enriched rating
print(district.finance_per_pupil)    # Enriched finance metric
```

- Comparing districts or campuses directly:

```python
districts = engine.all_districts()
top_districts = [d for d in districts if d.rating == 'A' and d.enrollment > 10000]
top_districts.sort(key=lambda d: d.enrollment, reverse=True)
```

- Combining spatial and attribute filters:

```python
# Campuses within Aldine ISD that are charters and have rating 'A'
charter_a_campuses = aldine >> 'campuses' >> (lambda c: "CHARTER" in c.type.upper()) >> (lambda c: c.rating == 'A')
```

- Chaining multiple enrichment and filtering operations:

```python
# After enriching districts with finance data, filter by per pupil spending and rating
affordable_high_rating = (engine.all_districts()
                         >> (lambda d: d.finance_per_pupil < 10000)
                         >> (lambda d: d.rating in ['A', 'B']))
```

### Concise, Readable Advanced Workflows

The design of the TEA Data Engine emphasizes code clarity and brevity, even for complex analyses. Instead of juggling dataframes or raw dictionaries, you work with rich objects that combine data, behavior, and spatial context. This leads to scripts and notebooks that are:

- **Easier to read and maintain**: Clear attribute access and chaining express intent directly.
- **More powerful**: Leverage Python’s full language features—functions, lambdas, comprehensions.
- **Highly extensible**: Add new datasets or enrichments without refactoring core logic.

In sum, the TEA Data Engine’s object-oriented, Pythonic design empowers data scientists, analysts, and developers to build advanced educational data workflows that are both expressive and efficient.

---

## 🌍 Spatial Power

Every district and campus object includes geospatial data. This allows:
- **Boundary lookups** (e.g., find all charter campuses in Aldine ISD)
- **Campus-in-district matching** (determine which district a campus belongs to)
- **Proximity queries** (e.g., nearest 5 campuses to a given coordinate)

```python
aldine = engine.get_district("101902")
print(len(aldine.campuses))  # Count campuses inside Aldine ISD

nearest = engine.nearest_campuses(lon=-95.37, lat=29.76, k=5)
for campus, distance in nearest:
    print(campus.name, distance)
```

---

## 🛠️ Enrichment System

The enrichment system lets you wire in external datasets from `teadata_sources.yaml` and `teadata_config.py`.  
Examples include accountability ratings, finance reports, or demographic summaries.

- Districts and campuses can be enriched with new attributes (`overall_rating_2025`, `finance_per_pupil`, etc.).
- Aliases let you map enriched attributes into canonical ones (e.g., writing `overall_rating_2025` into `.rating`).

The `.cache` snapshot saves enriched results, so reloads are instant and reproducible.

---

## 📖 Example Workflows

### Find all charter campuses in a district
```python
aldine = engine.get_district("101902")
for campus in aldine.campuses:
    if "CHARTER" in campus.type.upper():
        print(campus.name, campus.rating)
```

### Query nearest campuses to a coordinate
```python
nearest = engine.nearest_campuses(lon=-95.35, lat=29.85, k=3)
for campus, distance in nearest:
    print(f"{campus.name} is {distance} meters away")
```

### Enrichment example
```python
print(aldine.rating)  # canonical
print(aldine.overall_rating_2025)  # enriched
```

---

## 🗂️ Repository Structure

- `classes.py`: Core object models (`District`, `Campus`, `DataEngine`)
- `load_data.py` / `load_data2.py`: Build repo from raw datasets or cached snapshot
- `enrichment/`: Modular enrichment logic (districts, campuses, datasets)
- `.cache/`: Stores prebuilt `.pkl` snapshots for instant loading
- `shapes/`: GeoJSON boundaries for spatial analysis (not included in repo by default)
- `accountability/`: Source spreadsheets and reports (not included in repo by default)

---

## 📝 License

MIT License. See `LICENSE` for details.
