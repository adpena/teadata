"""Build a campus summary table with accountability + PEIMS metrics.

This mirrors the workflow described in the README response thread and avoids
`KeyError: 'district_number'` by keeping the campus-sourced district number
column during the district merge.
"""

from __future__ import annotations

import pandas as pd

from teadata import DataEngine


BASE_COLUMNS = [
    "campus_id",
    "campus_name",
    "district_name",
    "county_code",
    "governance_type",
    "grade_levels_served",
    "school_type_label",
    "rating_2025",
    "pct_econ_disadv",
    "pct_special_ed",
    "pct_attrition",
    "pct_mobility",
    "pct_at_risk",
    "pct_emergent_bilingual",
    "pct_beginning_teachers",
    "County",
]


def build_campus_summary() -> pd.DataFrame:
    repo = DataEngine.from_snapshot(search=True)

    campus_df = repo.campuses.to_df(include_geometry=True)
    district_df = repo.districts.to_df()[["id", "name"]].rename(
        columns={"id": "district_id_merge", "name": "district_name"}
    )

    campus_df = campus_df.merge(
        district_df,
        left_on="district_id",
        right_on="district_id_merge",
        how="left",
    ).drop(columns=["district_id_merge"])

    campus_df = campus_df.rename(
        columns={
            "name": "campus_name",
            "campus_number": "campus_id",
            "overall_rating_2025": "rating_2025",
            "grade_range": "grade_levels_served",
            "school_type": "school_type_label",
            "campus_2024_student_enrollment_econ_disadv_percent": "pct_econ_disadv",
            "campus_2024_student_enrollment_special_ed_percent": "pct_special_ed",
            "campus_2024_student_membership_2022_23_attrition_all_students_percent": "pct_attrition",
            "campus_2024_student_membership_2023_mobility_all_students_percent": "pct_mobility",
            "campus_2024_student_enrollment_at_risk_percent": "pct_at_risk",
            "campus_2024_student_enrollment_el_percent": "pct_emergent_bilingual",
            "campus_2024_staff_teacher_beginning_full_time_equiv_percent": "pct_beginning_teachers",
        }
    )

    campus_df["governance_type"] = campus_df["is_charter"].map(
        {True: "Charter", False: "District"}
    )
    campus_df["county_code"] = (
        campus_df["district_number"].str.replace("'", "", regex=False).str[:3]
    )

    numeric_cols = [
        "pct_econ_disadv",
        "pct_special_ed",
        "pct_attrition",
        "pct_mobility",
        "pct_at_risk",
        "pct_emergent_bilingual",
        "pct_beginning_teachers",
    ]
    campus_df[numeric_cols] = campus_df[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    # Derive 'County' from campus point geometry using the Texas county boundaries.
    campus_df = add_county_from_geometry(campus_df)

    return campus_df


# Helper: add County via spatial join to Texas county boundaries
def add_county_from_geometry(
    campus_df: pd.DataFrame,
    counties_path: str = "teadata/data/shapes/Texas_County_Boundaries_Detailed_7317697762830183947.geojson",
) -> pd.DataFrame:
    """
    Add a 'County' column by locating each campus point within Texas county polygons.

    Logic:
    - If a GeoSeries 'geometry' exists and contains shapely Points, use it.
    - Else, try to build Points from common lat/lng column names.
    - Read the counties geojson, align CRS, and spatial-join.
    - County name column is auto-detected from common field names.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except Exception as e:
        raise RuntimeError("geopandas and shapely are required for county mapping") from e

    # Determine or construct campus geometry (assumed WGS84 lon/lat).
    geom_col = None
    temp_geom_built = False
    lower_name_map = {c.lower(): c for c in campus_df.columns}

    # Prefer explicit geometry columns that may already contain shapely Points.
    for key in ("geometry", "geometry_point", "point", "location", "coords"):
        if key in lower_name_map:
            geom_col = lower_name_map[key]
            break

    if geom_col is None:
        # Try common latitude/longitude column pairs
        candidate_pairs = [
            ("longitude", "latitude"),
            ("lon", "lat"),
            ("lng", "lat"),
            ("x", "y"),
            ("campus_longitude", "campus_latitude"),
        ]
        lng_col, lat_col = None, None
        for lo, la in candidate_pairs:
            if lo in campus_df.columns and la in campus_df.columns:
                lng_col, lat_col = lo, la
                break
        if lng_col and lat_col:
            # Build Points from lon/lat
            campus_df = campus_df.copy()
            campus_df["_tmp_geom"] = [
                Point(float(lo), float(la)) if pd.notna(lo) and pd.notna(la) else None
                for lo, la in zip(campus_df[lng_col], campus_df[lat_col])
            ]
            geom_col = "_tmp_geom"
            temp_geom_built = True

    elif geom_col == lower_name_map.get("geometry_point"):
        # Geometry provided as (lon, lat) tuples from the repository snapshot.
        campus_df = campus_df.copy()
        campus_df["_tmp_geom"] = [
            Point(float(coords[0]), float(coords[1]))
            if isinstance(coords, (tuple, list))
            and len(coords) == 2
            and pd.notna(coords[0])
            and pd.notna(coords[1])
            else None
            for coords in campus_df[geom_col]
        ]
        geom_col = "_tmp_geom"
        temp_geom_built = True

    if geom_col is None:
        # No geometry available — return with County as NA
        campus_df = campus_df.copy()
        campus_df["County"] = pd.NA
        return campus_df

    # Promote to GeoDataFrame
    gcamp = gpd.GeoDataFrame(campus_df.copy(), geometry=campus_df[geom_col], crs="EPSG:4326")

    # Read counties; fall back to /mnt/data path if the project-relative path isn't found.
    try:
        counties = gpd.read_file(counties_path)
    except Exception:
        try:
            counties = gpd.read_file(
                "/mnt/data/Texas_County_Boundaries_Detailed_7317697762830183947.geojson"
            )
        except Exception:
            campus_df = campus_df.copy()
            campus_df["County"] = pd.NA
            if temp_geom_built:
                campus_df = campus_df.drop(columns=[geom_col], errors="ignore")
            return campus_df

    # Ensure CRS alignment
    if counties.crs is None:
        counties = counties.set_crs("EPSG:4326")
    counties = counties.to_crs(gcamp.crs)

    # Detect the county name / FIPS fields commonly used
    name_candidates = ["NAME", "Name", "county", "County", "CNTY_NM", "COUNTY_NAM", "COUNTY_NM", "COUNTYNAME"]
    fips_candidates = ["GEOID", "GEOID10", "COUNTYFP", "FIPS", "FIPS_CODE", "CNTY_FIPS"]

    county_name_col = next((c for c in name_candidates if c in counties.columns), None)
    county_fips_col = next((c for c in fips_candidates if c in counties.columns), None)

    # Keep only necessary columns for the join
    keep_cols = ["geometry"]
    if county_name_col:
        keep_cols.append(county_name_col)
    if county_fips_col:
        keep_cols.append(county_fips_col)
    counties = counties[keep_cols].copy()

    # Spatial join
    # Use 'within' primarily; if polygons have minor slivers, 'intersects' will still match points.
    joined = gpd.sjoin(gcamp, counties, how="left", predicate="within")
    if joined[county_name_col if county_name_col else "geometry_right"].isna().any():
        joined = gpd.sjoin(gcamp, counties, how="left", predicate="intersects")

    # Write the unified 'County' and optional 'county_fips'
    if county_name_col:
        joined["County"] = joined[county_name_col]
    else:
        joined["County"] = pd.NA
    if county_fips_col and "county_fips" not in joined.columns:
        joined["county_fips"] = joined[county_fips_col]

    # Drop temporary geometry if we created it
    out = pd.DataFrame(joined.drop(columns=["geometry"], errors="ignore"))
    if temp_geom_built:
        out = out.drop(columns=[geom_col], errors="ignore")

    return out


def split_by_governance(campus_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    district_campuses = campus_df.loc[
        campus_df["governance_type"] == "District", BASE_COLUMNS
    ].copy()
    charter_campuses = campus_df.loc[
        campus_df["governance_type"] == "Charter", BASE_COLUMNS
    ].copy()
    return district_campuses, charter_campuses


if __name__ == "__main__":
    campus_summary = build_campus_summary()
    district_tab, charter_tab = split_by_governance(campus_summary)

    district_tab.to_excel("campus_stats_district.xlsx", index=False)
    charter_tab.to_excel("campus_stats_charter.xlsx", index=False)

    print("Wrote campus_stats_district.xlsx and campus_stats_charter.xlsx")
