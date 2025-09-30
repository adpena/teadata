from teadata import DataEngine

repo = DataEngine.from_snapshot(search=True)
print(len(repo.districts), len(repo.campuses))
aldine = (repo >> ("district", "ALDINE ISD")).first()
print("Example district:", aldine)

# Inspect how many objects are loaded
print(f"Loaded {len(repo.districts)} districts and {len(repo.campuses)} campuses")

# Pick one district by name
aldine = (repo >> ("district", "ALDINE ISD")).first()

print("\nExample district:")
print(aldine)

# Access enriched attributes (dot-syntax works if enrichment logic is wired correctly)
print("Canonical rating:", aldine.rating)
print("Enriched rating 2025:", getattr(aldine, "overall_rating_2025", None))

# Explore campuses inside the district
print(f"\nAldine has {len(aldine.campuses)} campuses. First few:")
for campus in list(aldine.campuses)[:5]:
    print("-", campus.name, "| rating:", campus.rating)
