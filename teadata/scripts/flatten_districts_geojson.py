import argparse
import json
from typing import List

import geopandas as gpd  # type: ignore[import-untyped]
from shapely.geometry import shape, mapping, MultiPolygon, Polygon  # type: ignore[import-untyped]
from shapely.validation import make_valid  # type: ignore[import-untyped]

# --- Quick Settings for IDE runs (edit these and just press Run) ---
DEFAULT_INPUT = "/Users/adpena/PycharmProjects/teadata/teadata/data/shapes/Current_Districts_2025.geojson"
DEFAULT_OUTPUT = "/Users/adpena/PycharmProjects/teadata/teadata/data/shapes/Current_Districts_2025_flat.geojson"
DEFAULT_FILTER_FIELD = "NAME"
DEFAULT_FILTER_VALUE = "AUSTIN ISD"


def flatten_and_filter_geojson(
    input_path: str,
    output_path: str,
    filter_field: str | None = None,
    filter_value: str | None = None,
) -> None:
    """
    Flatten nested geometries into (Multi)Polygon and optionally filter by a property/value.

    Parameters
    ----------
    input_path : str
        Path to the source GeoJSON.
    output_path : str
        Path to write the cleaned/flattened GeoJSON.
    filter_field : str | None
        Feature property name to filter on (e.g., "NAME"). Case-insensitive comparison.
    filter_value : str | None
        Value to match in `filter_field` (e.g., "AUSTIN ISD"). Case-insensitive comparison.
    """
    # Load raw GeoJSON data
    with open(input_path, "r") as f:
        data = json.load(f)

    features = data.get("features", [])

    clean_features: List[dict] = []
    for feat in features:
        # Optional filtering by attribute (properties) BEFORE geometry work for speed
        if filter_field and filter_value:
            props = feat.get("properties", {})
            if str(props.get(filter_field, "")).upper() != str(filter_value).upper():
                continue

        try:
            geom = shape(feat["geometry"]) if feat.get("geometry") else None
            if geom is None or geom.is_empty:
                continue

            # Normalize to Polygon/MultiPolygon
            if geom.geom_type == "GeometryCollection":
                parts = [
                    g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")
                ]
                if not parts:
                    continue
                geom = MultiPolygon(parts) if len(parts) > 1 else parts[0]
            elif geom.geom_type == "Polygon":
                # Rebuild from exterior ring to avoid nested ring array issues
                geom = Polygon(geom.exterior)
            elif geom.geom_type != "MultiPolygon":
                # Skip unsupported geometry types
                continue

            # --- Repair invalid/self-intersecting geometries ---
            geom = make_valid(geom)

            # If repair returns a GeometryCollection, keep only polygonal parts
            if geom.geom_type == "GeometryCollection":
                polys = [
                    g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")
                ]
                if not polys:
                    continue
                geom = MultiPolygon(polys) if len(polys) > 1 else polys[0]

            # Recheck normalization after repair
            if geom.geom_type == "Polygon":
                geom = Polygon(geom.exterior)
            elif geom.geom_type != "MultiPolygon":
                # Skip if still not polygonal
                continue

            feat["geometry"] = mapping(geom)
            clean_features.append(feat)
        except Exception:
            # Skip any irreparable feature
            continue

    # Write back to GeoJSON
    out_fc = {"type": "FeatureCollection", "features": clean_features}
    with open(output_path, "w") as f:
        json.dump(out_fc, f)

    # Verify with GeoPandas (optional but helpful)
    gdf = gpd.read_file(output_path)
    print(f"✅ Flattened successfully with {len(gdf)} feature(s).")
    print("Columns:", gdf.columns.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flatten and optionally filter a districts GeoJSON."
    )
    parser.add_argument(
        "--input",
        "-i",
        dest="input_path",
        required=False,
        default=DEFAULT_INPUT,
        help="Path to input GeoJSON",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output_path",
        required=False,
        default=DEFAULT_OUTPUT,
        help="Path to output GeoJSON",
    )
    parser.add_argument(
        "--filter-field",
        dest="filter_field",
        default=DEFAULT_FILTER_FIELD,
        help="Property field to filter on (e.g., NAME)",
    )
    parser.add_argument(
        "--filter-value",
        dest="filter_value",
        default=DEFAULT_FILTER_VALUE,
        help="Value to match for the filter field (e.g., AUSTIN ISD)",
    )

    args = parser.parse_args()

    # Auto-run: write full flattened AND a district-only file for IDE runs
    # 1) Full flattened (no filter)
    flatten_and_filter_geojson(
        input_path=args.input_path,
        output_path=args.output_path,
        filter_field=None,
        filter_value=None,
    )

    # 2) Filtered district-only file (e.g., Austin ISD)
    #    If you change DEFAULT_FILTER_FIELD/DEFAULT_FILTER_VALUE above, this will follow.
    filtered_output = args.output_path.replace(
        ".geojson", f"_{args.filter_value.replace(' ', '_').upper()}.geojson"
    )
    flatten_and_filter_geojson(
        input_path=args.input_path,
        output_path=filtered_output,
        filter_field=args.filter_field,
        filter_value=args.filter_value,
    )

    print("\nFiles written:")
    print(" - Full flattened:", args.output_path)
    print(" - District-only :", filtered_output)
