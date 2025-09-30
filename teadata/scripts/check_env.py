# scripts/check_env.py
import importlib, sys

mods = [
    "numpy",
    "pandas",
    "shapely",
    "geopandas",
    "pyogrio",
    "pyarrow",
    "duckdb",
    "yaml",
    "openpyxl",
]
print(f"Python: {sys.version}")
for m in mods:
    try:
        mod = importlib.import_module(m if m != "yaml" else "yaml")
        ver = getattr(mod, "__version__", "unknown")
        print(f"{m:10} {ver}")
    except Exception as e:
        print(f"{m:10} MISSING ({e})")
        sys.exit(1)
print("✓ environment looks good")
