from teadata import DataEngine

# Initialize the DataEngine, this will use the cached .pkl snapshot if available
repo = DataEngine()

# Inspect how many objects are loaded
print(f"Loaded {len(repo.districts)} districts and {len(repo.campuses)} campuses")

# Pick one district by name
aldine = repo.districts["Aldine ISD"]

print("\nExample district:")
print(aldine)

# Access enriched attributes (dot-syntax works if enrichment logic is wired correctly)
print("Canonical rating:", aldine.rating)
print("Enriched rating 2025:", getattr(aldine, "overall_rating_2025", None))

# Explore campuses inside the district
print(f"\nAldine has {len(aldine.campuses)} campuses. First few:")
for campus in list(aldine.campuses)[:5]:
    print("-", campus.name, "| rating:", campus.rating)