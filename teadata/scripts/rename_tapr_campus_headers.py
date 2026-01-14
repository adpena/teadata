from pprint import pprint

import pandas as pd  # type: ignore[import-untyped]
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
    "/Users/adpena/PycharmProjects/teadata/teadata/data/tapr/OLD/CAMPPROF_2014-2015.csv"
)
REFERENCE_FILE = "/Users/adpena/PycharmProjects/teadata/teadata/data/tapr/OLD/CAMPPROF_2014-2015_data_elements.xlsx"
OUTPUT_FILE = "/Users/adpena/PycharmProjects/teadata/teadata/data/tapr/CAMPPROF_2014-2015_FINAL.xlsx"


def main():
    # Options to control workflow
    DO_SANITIZE = True
    DO_RESTORE = False
    FILTER = True

    tapr_df = pd.read_csv(INPUT_FILE)
    tapr_reference_df = pd.read_excel(REFERENCE_FILE)

    rename_dict = {
        "CAMPUS": "campus_number",
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

    cols = rename_dict.values()

    if FILTER is True:
        tapr_df = tapr_df[cols]

    if DO_SANITIZE:
        # Normalize column names to be Python-safe identifiers (using the same logic as sanitize_header)
        tapr_df.columns = [sanitize_header(c) for c in tapr_df.columns]
        pprint(list(tapr_df.columns))
        tapr_df.to_excel(OUTPUT_FILE, index=False)
    elif DO_RESTORE:
        # Restore human-friendly Labels from the sanitized headers
        tapr_df.columns = restore_labels_from_reference(
            list(tapr_df.columns), tapr_reference_df
        )
        pprint(list(tapr_df.columns))
        tapr_df.to_excel(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
