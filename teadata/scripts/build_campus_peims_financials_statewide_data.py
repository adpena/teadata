# mypy: ignore-errors

from pathlib import Path
import re
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from teadata.teadata_config import normalize_campus_number_value

# ---- CONFIG ----
# Directory containing the Excel workbooks
directory_url = "/Users/adpena/PycharmProjects/teadata/teadata/data/campus_peims_financials/campus_reports"

# Output (optional): set to None to skip saving
OUTPUT_CSV = "/Users/adpena/PycharmProjects/teadata/teadata/data/campus_peims_financials/2023-2024 Campus PEIMS Actual Financials Statewide.csv"

# Mapping of base variable name -> 1-based Excel row number
BASE_ROW_MAP = {
    "total_enrolled_membership": 3,
    "operating_payroll": 8,
    "other_operating": 9,
    "non_operating": 10,
    "instruction": 13,
    "instructional_res_media": 14,
    "curriculum_staff_develop": 15,
    "instructional_leadership": 16,
    "school_leadership": 17,
    "guidance_counseling": 18,
    "social_work": 19,
    "health_services": 20,
    "transportation": 21,
    "food": 22,
    "extracurricular": 23,
    "plant_maint_operation": 24,
    "security_monitoring": 25,
    "data_processing": 26,
    "fund_raising": 27,
    "regular_program": 30,
    "gifted_and_talented": 31,
    "career_and_technical": 32,
    "students_w_disabilities": 33,
    "state_compensatory_ed": 34,
    "bilingual_ed": 35,
    "early_education_allotment": 36,
    "dyslexia_or_related_disorder_serv": 37,
    "ccmr": 38,
    "athletic_programming": 39,
    "unallocated": 40,
}

# Column selections (1-based Excel columns):
# B, C, D = General Fund amount, %, Per Student
# E, F, G = All Funds amount, %, Per Student
COLS = {
    "gf": 2,  # column B
    "gf_perc": 3,  # column C
    "gf_per_student": 4,  # column D
    "af": 5,  # column E
    "af_perc": 6,  # column F
    "af_per_student": 7,  # column G
}

# -----------------


def extract_campus_number(path: Path) -> str | None:
    """Return the 9-digit run from the filename, or None if not found."""
    m = re.search(r"(\d{9})", path.name)
    return m.group(1) if m else None


def _parse_membership_value(x) -> int | None:
    """Extract an integer from strings like 'Total Student Membership: 1,234'.
    Returns None if no digits are found."""
    if x is None:
        return None
    s = str(x)
    # if there's a colon, take the right-hand side
    if ":" in s:
        s = s.split(":", 1)[1]
    # remove commas, spaces, currency, etc.
    s = s.replace(",", "").replace("$", "").strip()
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def scrape_file(xlsx_path: Path) -> dict:
    """Read required cells from the first sheet and return a flat record dict."""
    # Read as raw table (no headers). Default sheet (index 0) is used.
    df = pd.read_excel(xlsx_path, header=None)

    rec: dict[str, object] = {
        "source_path": str(xlsx_path.resolve()),
        "campus_number": extract_campus_number(xlsx_path),
    }

    # --- Special case: total_enrolled_membership is a merged/text cell on its own row ---
    mem_row = BASE_ROW_MAP["total_enrolled_membership"] - 1
    try:
        row_vals = df.iloc[mem_row, :].tolist()
        # first non-empty cell in the row
        first_non_empty = next(
            (v for v in row_vals if pd.notna(v) and str(v).strip() != ""), None
        )
        rec["total_enrolled_membership"] = _parse_membership_value(first_non_empty)
    except Exception:
        rec["total_enrolled_membership"] = None

    # Translate 1-based row/column to 0-based iloc indices
    for base, row1 in BASE_ROW_MAP.items():
        if base == "total_enrolled_membership":
            continue  # handled separately above
        r = row1 - 1
        try:
            rec[f"{base}_gf"] = df.iat[r, COLS["gf"] - 1]
        except Exception:
            rec[f"{base}_gf"] = None
        try:
            rec[f"{base}_gf_perc"] = df.iat[r, COLS["gf_perc"] - 1]
        except Exception:
            rec[f"{base}_gf_perc"] = None
        try:
            rec[f"{base}_gf_per_student"] = df.iat[r, COLS["gf_per_student"] - 1]
        except Exception:
            rec[f"{base}_gf_per_student"] = None
        try:
            rec[f"{base}_af"] = df.iat[r, COLS["af"] - 1]
        except Exception:
            rec[f"{base}_af"] = None
        try:
            rec[f"{base}_af_perc"] = df.iat[r, COLS["af_perc"] - 1]
        except Exception:
            rec[f"{base}_af_perc"] = None
        try:
            rec[f"{base}_af_per_student"] = df.iat[r, COLS["af_per_student"] - 1]
        except Exception:
            rec[f"{base}_af_per_student"] = None

    return rec


def scrape_file_safe(fp: Path) -> dict:
    """Wrapper that never raises; returns an error field on failure."""
    try:
        return scrape_file(fp)
    except Exception as e:
        return {
            "source_path": str(fp.resolve()),
            "campus_number": normalize_campus_number_value(extract_campus_number(fp)),
            "error": str(e),
        }


def main() -> None:
    base = Path(directory_url)
    files = sorted(base.glob("*.xlsx"))

    records: list[dict] = []

    # Parallel scrape with a progress bar
    if files:
        with ProcessPoolExecutor() as ex:
            futures = [ex.submit(scrape_file_safe, fp) for fp in files]
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc="Scraping", unit="file"
            ):
                records.append(fut.result())
    else:
        print("No .xlsx files found.")

    df = pd.DataFrame(records)
    df = pd.DataFrame(records)
    df["campus_number"] = df["campus_number"].map(normalize_campus_number_value)

    print(df.head())
    print(f"\nRows: {len(df)}  Columns: {len(df.columns)}")

    if OUTPUT_CSV:
        outp = Path(OUTPUT_CSV)
        outp.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outp, index=False)
        print(f"Saved -> {outp}")


if __name__ == "__main__":
    main()
