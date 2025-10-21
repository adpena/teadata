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
]


def build_campus_summary() -> pd.DataFrame:
    repo = DataEngine.from_snapshot(search=True)

    campus_df = repo.campuses.to_df()
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

    return campus_df


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
