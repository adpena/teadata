"""Build a campus summary table with accountability + PEIMS metrics.

This mirrors the workflow described in the README response thread and avoids
`KeyError: 'district_number'` by keeping the campus-sourced district number
column during the district merge.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from teadata import DataEngine

from restyle_existing_excel import style_existing_excel


# Adjust these values when running from an IDE. They are ignored when the module is
# imported elsewhere, but provide a convenient toggle when executing the script
# directly via "Run" or "Debug".
RUN_CONFIG: dict[str, object] = {
    # Set to a float such as 75 to require a minimum percent economically
    # disadvantaged. Leave as None to include all campuses.
    "min_pct_econ_disadv": None,
    # Choose from "ab"/"high" (A or B campuses) or "df"/"low" (D or F campuses).
    # You may also provide a list like ["ab", "df"] or leave as None to disable
    # rating-based filtering.
    "rating_filter": None,
}


BASE_COLUMNS = [
    "campus_id",
    "campus_name",
    "district_name",
    "county",
    "governance_type",
    "grade_levels_served",
    "school_type_label",
    "is_magnet",
    "rating_2025",
    "pct_econ_disadv",
    "pct_special_ed",
    "pct_attrition",
    "pct_mobility",
    "pct_at_risk",
    "pct_emergent_bilingual",
    "pct_beginning_teachers",
]


def build_campus_summary(
    min_pct_econ_disadv: float | None = None,
    rating_filter: str | Iterable[str] | None = None,
) -> pd.DataFrame:
    repo = DataEngine.from_snapshot(search=True)

    campus_df = repo.campuses.to_df(include_geometry=True)

    # Exclude private campuses
    if "is_private" in campus_df.columns:
        campus_df = campus_df[~campus_df["is_private"].fillna(False)]

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

    if "is_magnet" not in campus_df.columns:
        campus_df["is_magnet"] = pd.NA

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

    if min_pct_econ_disadv is not None and "pct_econ_disadv" in campus_df.columns:
        campus_df = campus_df[campus_df["pct_econ_disadv"].ge(min_pct_econ_disadv)]

    if rating_filter:
        rating_groups: dict[str, set[str]] = {
            "ab": {"A", "B"},
            "a/b": {"A", "B"},
            "high": {"A", "B"},
            "df": {"D", "F"},
            "d/f": {"D", "F"},
            "low": {"D", "F"},
            "a": {"A"},
            "b": {"B"},
            "d": {"D"},
            "f": {"F"},
        }

        if isinstance(rating_filter, str):
            rating_keys = {rating_filter}
        else:
            rating_keys = set(rating_filter)

        normalized_keys = {key.strip().lower() for key in rating_keys if key}
        if not normalized_keys:
            raise ValueError("rating_filter provided but no valid keys found")

        allowed_ratings: set[str] = set()
        for key in normalized_keys:
            if key not in rating_groups:
                raise ValueError(
                    "rating_filter must be one of 'ab', 'df', 'high', 'low', or an iterable of these"
                )
            allowed_ratings.update(rating_groups[key])

        if "rating_2025" not in campus_df.columns:
            raise ValueError("rating_filter requested but 'rating_2025' column is missing")

        ratings_normalized = campus_df["rating_2025"].astype(str).str.upper()
        campus_df = campus_df[ratings_normalized.isin(allowed_ratings)]

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
        campus_df["county"] = pd.NA
        return campus_df

    # Promote to GeoDataFrame
    gcamp = gpd.GeoDataFrame(campus_df.copy(), geometry=campus_df[geom_col], crs="EPSG:4326")

    # Read counties; fall back to data path if the project-relative path isn't found.
    try:
        counties = gpd.read_file(counties_path)
    except Exception:
        try:
            counties = gpd.read_file(
                "/Users/adpena/PycharmProjects/teadata/teadata/data/shapes/Texas_County_Boundaries_Detailed_7317697762830183947.geojson"
            )
        except Exception:
            campus_df = campus_df.copy()
            campus_df["county"] = pd.NA
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
        joined["county"] = joined[county_name_col]
    else:
        joined["county"] = pd.NA
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


def main(
    *,
    min_pct_econ_disadv: float | None = None,
    rating_filter: str | Iterable[str] | None = None,
) -> None:
    campus_summary = build_campus_summary(
        min_pct_econ_disadv=min_pct_econ_disadv,
        rating_filter=rating_filter,
    )
    district_tab, charter_tab = split_by_governance(campus_summary)

    # Rename columns for output readability
    rename_map = {
        "campus_id": "Campus Number",
        "campus_name": "Campus Name",
        "district_name": "District Name",
        "county": "County",
        "governance_type": "District/Charter",
        "grade_levels_served": "Grades Served",
        "school_type_label": "School Type",
        "is_magnet": "Is Magnet",
        "rating_2025": "2025 Overall Rating",
        "pct_econ_disadv": "2023-24 % Economically Disadvantaged",
        "pct_special_ed": "2023-24 % Special Education",
        "pct_at_risk": "2023-24 % At-Risk",
        "pct_emergent_bilingual": "2023-24 % Emergent Bilingual (EL)",
        "pct_beginning_teachers": "2023-24 % Beginning Teachers",
        "pct_attrition": "2023-24 % Attrition (2022-23)",
        "pct_mobility": "2023-24 % Mobility (2023)",
    }

    # Apply renaming and enforce the order
    district_tab = district_tab.rename(columns=rename_map)[[rename_map[c] for c in BASE_COLUMNS]]
    charter_tab = charter_tab.rename(columns=rename_map)[[rename_map[c] for c in BASE_COLUMNS]]

    # ✅ Sort by District Name then Campus Name
    district_tab = district_tab.sort_values(by=["District Name", "Campus Name"], ascending=[True, True])
    charter_tab = charter_tab.sort_values(by=["District Name", "Campus Name"], ascending=[True, True])

    district_tab.to_excel("campus_stats_district.xlsx", index=False)
    charter_tab.to_excel("campus_stats_charter.xlsx", index=False)

    print("Wrote campus_stats_district.xlsx and campus_stats_charter.xlsx")

    style_existing_excel(
        "campus_stats_district.xlsx",  # use the same absolute path we just wrote
        sheet_name="Sheet1",
        table_style="TableStyleMedium2",
        open_after=False,
    )

    style_existing_excel(
        "campus_stats_charter.xlsx",  # use the same absolute path we just wrote
        sheet_name="Sheet1",
        table_style="TableStyleMedium2",
        open_after=False,
    )


if __name__ == "__main__":
    main(
        min_pct_econ_disadv=RUN_CONFIG.get("min_pct_econ_disadv"),
        rating_filter=RUN_CONFIG.get("rating_filter"),
    )
