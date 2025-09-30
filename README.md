# TEA Data Engine

The TEA Data Engine is a Python-based toolkit for working with Texas public education datasets.

It provides object-oriented access to rich, spatially-aware data about school districts and campuses, making it easy for developers to explore, enrich, and analyze educational data.

---

## 🚀 Quick Start

Getting started is straightforward. First, clone the repository to your local machine:

```bash
git clone https://github.com/adpena/teadata.git
cd teadata
```

Then, install the package in editable mode:

```bash
pip install -e .
```

Next, import the library and initialize the engine:

```python
from teadata import DataEngine

# Initialize the engine directly from cache for instant access
engine = DataEngine()

# Quickly retrieve a district by its unique number
aldine = engine.get_district("101902")
print(aldine.name)                   # Aldine ISD
print(aldine.overall_rating_2025)   # Access enriched accountability rating instantly

# Iterate through all campuses within the district effortlessly
for campus in aldine.campuses:
    print(campus.name, campus.rating)
```

This setup avoids unnecessary data wrangling and lets you focus on analysis immediately.

---

## 📊 Models

At the heart of the TEA Data Engine are two rich, spatially-aware models designed to mirror the real world:

### `District`

A `District` represents a Texas school district with comprehensive attributes:

- **Unique ID & District Number**: Normalized identifiers including zero-padding for consistency.
- **Name & Enrollment**: Key demographic and identifying info.
- **Type**: Classifies the district (e.g., charter, traditional ISD).
- **Ratings**: Both canonical and dynamically enriched ratings like `overall_rating_2025`.
- **Spatial Geometry**: Precise district boundaries enabling advanced geospatial queries.

The `District` class is packed with powerful behaviors:

- `.campuses`: Fetch all campuses within the district boundary with a simple attribute access.
- `.nearest_campuses()`: Perform location-based queries to find closest campuses, supporting proximity analyses.
- `.enrich()`: Dynamically attach new attributes from external datasets, allowing your data to evolve seamlessly.

### `Campus`

Each `Campus` is modeled with equal richness:

- **Unique ID & Campus Number**: Normalized to the official nine-digit TEA format.
- **Name, Enrollment, Type**: Core descriptive attributes.
- **Ratings**: Access canonical or enriched ratings effortlessly.
- **Spatial Geometry**: Precise campus location points for spatial analysis.

Campus objects link back to their parent district through `.district`, enabling easy navigation across hierarchical data.

Both models are designed to be extensible and dynamic, supporting complex workflows without sacrificing clarity or performance.

---

## 🧠 Why Object-Oriented?

The TEA Data Engine leverages Python’s OOP principles to deliver a developer-friendly, highly expressive API that feels natural and intuitive.

### Dynamic Attributes & Enrichment

Forget rigid schemas or cumbersome joins. The engine’s enrichment system lets you **inject new attributes on the fly**, turning raw datasets into rich, multidimensional objects:

```python
print(district.overall_rating_2025)   # Seamless access to enriched accountability data
print(campus.finance_per_pupil)       # Instantly available finance metrics
```

This dynamic enrichment means your analyses can evolve as new datasets arrive, without rewriting core logic or managing complex merges.

### Operator Overloading for Natural Comparisons

Districts and campuses implement Python’s comparison operators based on meaningful attributes like enrollment or ratings, enabling elegant and readable code:

```python
if campus1.enrollment > campus2.enrollment:
    print(f"{campus1.name} has more students than {campus2.name}")

if district1.rating == 'A' and district2.rating != 'A':
    print(f"{district1.name} outperforms {district2.name}")
```

This reduces boilerplate and lets you write code that reads like plain English, enhancing maintainability and reducing errors.

### Query Chaining with the `>>` Operator

One of the toolkit’s coolest features is the **`>>` operator**, which enables **fluent query chaining** to compose filters, transformations, and traversals in a clean, pipeline style:

```python
# From a district, retrieve campuses, then filter by rating 'A'
top_campuses = aldine >> 'campuses' >> (lambda c: c.rating == 'A')
for campus in top_campuses:
    print(campus.name, campus.rating)

# Chain multiple filters: campuses with enrollment > 500 and rating 'B' or better
filtered = aldine >> 'campuses' >> (lambda c: c.enrollment > 500) >> (lambda c: c.rating in ['A', 'B'])
```

This expressive syntax dramatically simplifies complex queries, eliminating nested loops or intermediate variables. It encourages concise, readable code that scales naturally as your analysis grows.

---

## 🌍 Spatial Power

Every district and campus object includes detailed geospatial data, unlocking a world of location-based insights:

- **Boundary Lookups**: Easily find all charter campuses inside a district with a simple filter.
- **Campus-in-District Matching**: Determine which district a campus belongs to based on spatial containment.
- **Proximity Queries**: Find the nearest campuses to any coordinate, enabling targeted outreach or resource planning.

```python
aldine = engine.get_district("101902")
print(f"{aldine.name} has {len(aldine.campuses)} campuses")

nearest = engine.nearest_campuses(lon=-95.37, lat=29.76, k=5)
for campus, distance in nearest:
    print(f"{campus.name} is {distance:.1f} meters away")
```

Spatial operations are seamlessly integrated into the data model, so you can conduct advanced geospatial analyses without external GIS tools or complicated workflows.

---

## 🛠️ Enrichment System

The enrichment system is a game-changer, letting you wire in external datasets from configuration files like `teadata_sources.yaml` and `teadata_config.py` with minimal effort.

- **Dynamic Attribute Injection**: Add new data fields like accountability ratings, finance metrics, or demographic summaries directly onto District and Campus objects.
- **Alias Mapping**: Map enriched attributes into canonical fields (e.g., `overall_rating_2025` into `.rating`) for consistent access.
- **Cache Snapshots**: Save enriched datasets as `.pkl` snapshots for lightning-fast reloads and reproducible analyses.

This modular enrichment architecture means your data ecosystem can grow organically, supporting new research questions and deliverables without rewriting foundational code.

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
print(aldine.rating)  # canonical rating
print(aldine.overall_rating_2025)  # enriched accountability rating
```

### Complex chained query example

```python
# Campuses within Aldine ISD that are charters and have rating 'A'
charter_a_campuses = aldine >> 'campuses' \
                             >> (lambda c: "CHARTER" in c.type.upper()) \
                             >> (lambda c: c.rating == 'A')
for campus in charter_a_campuses:
    print(campus.name, campus.rating)
```

### Combining enrichment and filtering

```python
# After enriching districts with finance data, filter by per pupil spending and rating
affordable_high_rating = (engine.all_districts()
                         >> (lambda d: d.finance_per_pupil < 10000)
                         >> (lambda d: d.rating in ['A', 'B']))
for district in affordable_high_rating:
    print(district.name, district.finance_per_pupil, district.rating)
```

These examples highlight how the TEA Data Engine’s design encourages clear, concise, and powerful data exploration.

---

## ⏳ Time Savings & Research Power

The TEA Data Engine isn’t just a toolkit — it’s a **productivity powerhouse** that transforms how you conduct education data research:

- **Accelerate Data Preparation**: Eliminate hours of manual data cleaning, joining, and formatting. The engine’s built-in normalization and enrichment handle these automatically.
- **Simplify Complex Queries**: The `>>` operator and dynamic attributes let you write expressive, maintainable code that would otherwise require verbose SQL or nested loops.
- **Enable New Analyses**: Spatial awareness and dynamic enrichment unlock research questions that were previously too time-consuming or complex, such as proximity-based resource allocation or multi-dimensional performance comparisons.
- **Improve Reproducibility**: Cached snapshots and modular enrichment pipelines ensure your workflows are consistent and easy to share.
- **Reduce Cognitive Load**: By encapsulating data and behavior within intuitive objects, the engine lets you focus on insights rather than data plumbing.

Compared to spreadsheets or ad hoc scripts, the TEA Data Engine dramatically reduces errors, saves development time, and opens doors to richer, more nuanced educational analyses. It’s the ideal tool for analysts, policymakers, and developers who want to move fast without sacrificing rigor.

---

## 🗂️ Repository Structure

The repository is thoughtfully organized to support extensibility and ease of use:

- `classes.py`: Core object models including `District`, `Campus`, and `DataEngine` that form the foundation of the toolkit.
- `load_data.py` / `load_data2.py`: Scripts to build the repository from raw datasets or cached snapshots, facilitating flexible data ingestion workflows.
- `enrichment/`: Modular enrichment logic organized by entity type (districts, campuses) and dataset, enabling easy addition of new data sources.
- `.cache/`: Stores prebuilt `.pkl` snapshots for instant loading and reproducibility.
- `shapes/`: GeoJSON boundaries for spatial analysis (not included in repo by default to save space).
- `accountability/`: Source spreadsheets and reports (not included by default), supporting transparency and traceability.

This structure supports clean separation of concerns, encourages modular development, and makes it easy to extend or customize the engine for your unique needs.

---

## 📝 License

MIT License. See `LICENSE` for details.

---

With the TEA Data Engine, you’re equipped with a cutting-edge toolkit that makes Texas education data more accessible, analyzable, and actionable than ever before. Experience faster workflows, richer insights, and a new level of research power—try it today!
