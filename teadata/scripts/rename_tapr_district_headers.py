from pprint import pprint

import pandas as pd
import re


def sanitize_header(s: str) -> str:
    """Convert a column header to a Python-safe identifier, mirroring the
    normalization applied to TAPR columns in this script.
    """
    s = str(s)
    s = s.lower()
    # remove punctuation we don't want to translate
    s = s.replace(":", "").replace("(", "").replace(")", "")
    # translate some symbols to words
    s = s.replace("&", " and ")
    s = s.replace("%", " percent ")
    s = s.replace("#", " number ")
    s = s.replace(">", " gt ")
    s = s.replace("<", " lt ")
    # unify dashes and slashes to underscores (includes en/em dashes)
    s = re.sub(r"[\\/–—\-]+", "_", s)
    # replace dots and whitespace with underscores
    s = re.sub(r"[.]", "_", s)
    s = re.sub(r"\s+", "_", s)
    # remove any remaining disallowed chars
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    # squeeze repeats and trim
    s = re.sub(r"__+", "_", s)
    return s.strip("_")


def restore_labels_from_reference(
    cols: list[str], reference_df: pd.DataFrame
) -> list[str]:
    """Given sanitized column names and the TAPR data elements dataframe,
    return the original, human-friendly Labels with capitalization/punctuation
    when possible. Columns that do not match a known Label are left as-is.

    This uses the *sanitized Label* as the lookup key to reverse the mapping.
    If multiple Labels sanitize to the same key, the first occurrence wins.
    """
    # Build lookup: sanitized label -> original label
    lookup: dict[str, str] = {}
    for lbl in reference_df["Label"].astype(str).tolist():
        key = sanitize_header(lbl)
        if key not in lookup:
            lookup[key] = lbl
        # if there is a collision, keep the first-seen label deterministically
    # Map provided columns back to labels when we have a match
    return [lookup.get(c, c) for c in cols]


INPUT_FILE = (
    "/Users/adpena/PycharmProjects/teadata/teadata/data/tapr/raw/DISTPROF_2023-2024.csv"
)
REFERENCE_FILE = "/Users/adpena/PycharmProjects/teadata/teadata/data/tapr/raw/DISTPROF_2023-2024_data_elements.xlsx"
OUTPUT_FILE = "/Users/adpena/PycharmProjects/teadata/teadata/data/tapr/DISTPROF_2023-2024_FINAL.xlsx"


def main():
    # Options to control workflow
    DO_SANITIZE = True
    DO_RESTORE = False
    FILTER = True

    tapr_df = pd.read_csv(INPUT_FILE)
    tapr_reference_df = pd.read_excel(REFERENCE_FILE)

    rename_dict = {
        "DISTRICT": "district_number",
        "CPETALLC": "campus_2015_student_enrollment_all_students_count",
        # "CAMPNAME": "campus_name",
        # "DISTRICT": "district_number",
        # "DISTNAME": "district_name",
    }

    try:
        auto_rename_dict = dict(
            zip(tapr_reference_df["Name"], tapr_reference_df["Label"])
        )
    except Exception:
        auto_rename_dict = dict(
            zip(tapr_reference_df["NAME"], tapr_reference_df["LABEL"])
        )

    tapr_df = tapr_df.rename(columns=rename_dict)
    tapr_df = tapr_df.rename(columns=auto_rename_dict)

    # Desired columns (sanitized identifiers)
    cols_sanitized = [
        "district_number",
        "district_2024_student_enrollment_all_students_count",
        "district_2024_student_enrollment_african_american_percent",
        "district_2024_student_enrollment_hispanic_percent",
        "district_2024_student_enrollment_white_percent",
        "district_2024_student_enrollment_american_indian_percent",
        "district_2024_student_enrollment_asian_percent",
        "district_2024_student_enrollment_pacific_islander_percent",
        "district_2024_student_enrollment_two_or_more_races_percent",
        "district_2024_student_enrollment_econ_disadv_percent",
        "district_2024_student_enrollment_non_educationally_disadv_percent",
        "district_2024_student_enrollment_section_504_percent",
        "district_2024_student_enrollment_el_percent",
        "district_2024_student_enrollment_dyslexia_percent",
        "district_2024_student_enrollment_foster_care_percent",
        "district_2024_student_enrollment_homeless_percent",
        "district_2024_student_enrollment_immigrant_percent",
        "district_2024_student_enrollment_migrant_percent",
        "district_2024_student_enrollment_title_i_percent",
        "district_2024_student_enrollment_at_risk_percent",
        "district_2024_student_enrollment_bilingual_esl_percent",
        "district_2024_student_enrollment_gifted_and_talented_percent",
        "district_2024_student_enrollment_special_ed_percent",
        "district_2023_daep_percent",
        "district_2024_student_membership_2022_23_attrition_all_students_percent",
        "district_2024_student_membership_2023_mobility_all_students_percent",
        "district_2024_staff_teacher_total_full_time_equiv_count",
        "district_2024_staff_support_total_full_time_equiv_count",
        "district_2024_staff_school_admin_total_full_time_equiv_count",
        "district_2024_staff_central_admin_total_full_time_equiv_count",
        "district_2024_staff_educ_aide_total_full_time_equiv_count",
        "district_2024_staff_ssa_educational_aides_full_time_equiv_count",
        "district_2024_staff_counselor_total_part_time_equiv_count",
        "district_2024_staff_counselor_total_full_time_equiv_count",
        "district_2024_staff_librarian_total_part_time_equiv_count",
        "district_2024_staff_librarian_total_full_time_equiv_count",
        "district_2024_staff_teacher_regular_program_full_time_equiv_count",
        "district_2024_staff_teacher_career_and_technical_prgms_full_time_equiv_count",
        "district_2024_staff_teacher_bilingual_program_full_time_equiv_count",
        "district_2024_staff_teacher_special_education_full_time_equiv_count",
        "district_2024_staff_teacher_turnover_ratio",
        "district_2024_staff_teacher_tenure_average",
        "district_2024_staff_teacher_experience_average",
        "district_2024_staff_teacher_student_ratio",
        "district_2024_staff_professional_total_full_time_equiv_percent",
        "district_2024_staff_teacher_total_full_time_equiv_percent",
        "district_2024_staff_support_total_full_time_equiv_percent",
        "district_2024_staff_school_admin_total_full_time_equiv_percent",
        "district_2024_staff_central_admin_total_full_time_equiv_percent",
        "district_2024_staff_educ_aide_total_full_time_equiv_percent",
        "district_2024_staff_auxiliary_total_full_time_equiv_percent",
        "district_2024_staff_teacher_no_degree_full_time_equiv_percent",
        "district_2024_staff_teacher_ba_degree_full_time_equiv_percent",
        "district_2024_staff_teacher_ms_degree_full_time_equiv_percent",
        "district_2024_staff_teacher_ph_degree_full_time_equiv_percent",
        "district_2024_staff_teacher_beginning_full_time_equiv_percent",
        "district_2024_staff_teacher_1_5_years_full_time_equiv_percent",
        "district_2024_staff_teacher_6_10_years_full_time_equiv_percent",
        "district_2024_staff_teacher_11_20_years_full_time_equiv_percent",
        "district_2024_staff_teacher_21_30_years_full_time_equiv_percent",
        "district_2024_staff_teacher_gt_30_years_full_time_equiv_percent",
        "district_2024_staff_teacher_regular_program_full_time_equiv_percent",
        "district_2024_staff_teacher_career_and_technical_prgms_full_time_equiv_percent",
        "district_2024_staff_teacher_bilingual_program_full_time_equiv_percent",
        "district_2024_staff_teacher_special_education_full_time_equiv_percent",
        "district_2024_staff_teacher_beginning_base_salary_average",
        "district_2024_staff_teacher_1_5_years_base_salary_average",
        "district_2024_staff_teacher_6_10_years_base_salary_average",
        "district_2024_staff_teacher_11_20_years_base_salary_average",
        "district_2024_staff_teacher_21_30_years_base_salary_average",
        "district_2024_staff_teacher_gt_30_years_base_salary_average",
        "district_2024_staff_teacher_total_base_salary_average",
        "district_2024_staff_support_total_base_salary_average",
        "district_2024_staff_school_admin_total_base_salary_average",
        "district_2024_staff_central_admin_total_base_salary_average",
    ]
    # Map desired sanitized names to human Labels using the reference file
    cols_labels = restore_labels_from_reference(cols_sanitized, tapr_reference_df)

    def _are_cols_sanitized(colnames: list[str]) -> bool:
        return all(c == sanitize_header(c) for c in colnames)

    if DO_SANITIZE:
        # 1) Sanitize current headers
        tapr_df.columns = [sanitize_header(c) for c in tapr_df.columns]
        # 2) Filter using sanitized names (to avoid KeyError before sanitize)
        if FILTER:
            missing = [c for c in cols_sanitized if c not in tapr_df.columns]
            if missing:
                print("Warning: missing expected columns (sanitized):", missing)
            tapr_df = tapr_df[[c for c in cols_sanitized if c in tapr_df.columns]]
        pprint(list(tapr_df.columns))
        tapr_df.to_excel(OUTPUT_FILE, index=False)

    elif DO_RESTORE:
        # If we want labels, filter with labels when headers are not sanitized,
        # otherwise filter with sanitized and then restore.
        if FILTER:
            if _are_cols_sanitized(list(tapr_df.columns)):
                desired = [c for c in cols_sanitized if c in tapr_df.columns]
            else:
                desired = [c for c in cols_labels if c in tapr_df.columns]
            missing = [
                c
                for c in (
                    cols_labels
                    if not _are_cols_sanitized(list(tapr_df.columns))
                    else cols_sanitized
                )
                if c not in tapr_df.columns
            ]
            if missing:
                print("Warning: missing expected columns:", missing)
            tapr_df = tapr_df[desired]
        # Finally, restore human-friendly Labels if headers are sanitized
        if _are_cols_sanitized(list(tapr_df.columns)):
            tapr_df.columns = restore_labels_from_reference(
                list(tapr_df.columns), tapr_reference_df
            )
        pprint(list(tapr_df.columns))
        tapr_df.to_excel(OUTPUT_FILE, index=False)

    # pprint(tapr_df.columns.tolist())
    # exit()


if __name__ == "__main__":
    main()
