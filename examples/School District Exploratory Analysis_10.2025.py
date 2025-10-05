#!/usr/bin/env python
# coding: utf-8

# In[1]:


from teadata import DataEngine

# If needed:
from teadata.classes import haversine_miles, inspect_object
import pandas as pd
from pathlib import Path

# Set option to display all rows and columns
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# Enter the district name or district number of the school district you wish to analyze
DISTRICT_NAME = "Austin ISD"

# Instantiate DataEngine object, a layer over the District and Campus objects
repo = DataEngine.from_snapshot(search=True)

# Inspect how many objects are loaded
print(
    f"Loaded {len(repo.districts)} districts/charters and {len(repo.campuses)} campuses"
)

# Select a district using either the district name (case insensitive) or district number (any format works - integer and string with or without a leading apostrophe are both acceptable)
district = (repo >> ("district", DISTRICT_NAME)).first()
print("Example district:", district.name, district.district_number)

rows = (
    repo
    >> ("campuses_in", district)
    >> ("where", lambda x: (x.enrollment or 0) > 0)
    >> ("where", lambda x: x.facing_closure is True)
    >> ("sort", lambda x: x.name or "", False)
    # >> ("take", 5)
    # >> ("transfers_out", True)  # Option 1: charter_only=True baked in
    >> (
        "nearest_charter_transfer_destination",
    )  # yields list of (campus, match, miles)
)

for closure_campus in rows:
    match = closure_campus[1]
    distance = closure_campus[2]
    closure_campus = closure_campus[0]

    print(
        closure_campus.campus_number,
        closure_campus.name,
        closure_campus.enrollment,
        closure_campus.rating,
        closure_campus.aea,
        closure_campus.grade_range,
    )

df = rows.to_df()

# pprint(list(df.columns))

cols = {
    "campus_number": "Campus Number",
    "name": "Campus Name",
    "enrollment": "Enrollment as of Oct 2024",
    "overall_rating_2025": "2025 Overall Rating",
    "aea": "AEA",
    "grade_range": "Grade Range",
    "percent_enrollment_change": "2014-15 to 2024-2025 Percent Enrollment Change",
    "num_charter_transfer_destinations": "Number of Charter Campuses Receiving Student Transfers - TOTAL",
    "num_charter_transfer_destinations_masked": "Number of Charter Campuses Receiving Student Transfers - MASKED",
    "total_unmasked_charter_transfers_out": "Total Unmasked Charter Transfers Out",
    "closure_date": "Proposed Closure Date",
    "campus_2024_student_membership_2023_mobility_all_students_percent": "2024 Student Membership: 2023 Mobility All Students Percent",
    "campus_2024_student_membership_2022_23_attrition_all_students_percent": "2024 Student Membership: 2022-23 Attrition All Students Percent",
    "campus_2024_student_enrollment_african_american_percent": "2024 Student Enrollment: African American Percent",
    "campus_2024_student_enrollment_hispanic_percent": "2024 Student Enrollment: Hispanic Percent",
    "campus_2024_student_enrollment_econ_disadv_percent": "2024 Student Enrollment: Econ Disadvantaged Percent",
    "campus_2024_student_enrollment_section_504_percent": "2024 Student Enrollment: Section 504 Percent",
    "campus_2024_student_enrollment_el_percent": "2024 Student Enrollment: EL Percent",
    "campus_2024_student_enrollment_dyslexia_percent": "2024 Student Enrollment: Dyslexia Percent",
    "campus_2024_student_enrollment_foster_care_percent": "2024 Student Enrollment: Foster Care Percent",
    "campus_2024_student_enrollment_homeless_percent": "2024 Student Enrollment: Homeless Percent",
    "campus_2024_student_enrollment_immigrant_percent": "2024 Student Enrollment: Immigrant Percent",
    "campus_2024_student_enrollment_migrant_percent": "2024 Student Enrollment: Migrant Percent",
    "campus_2024_student_enrollment_title_i_percent": "2024 Student Enrollment: Title I Percent",
    "campus_2024_student_enrollment_at_risk_percent": "2024 Student Enrollment: At Risk Percent",
    "campus_2024_student_enrollment_bilingual_esl_percent": "2024 Student Enrollment: Bilingual ESL Percent",
    "campus_2024_student_enrollment_special_ed_percent": "2024 Student Enrollment: Special Ed Percent",
    "campus_2024_staff_teacher_beginning_full_time_equiv_percent": "2024 Staff: Teacher Beginning FTE Percent",
    "campus_2024_staff_teacher_1_5_years_full_time_equiv_percent": "2024 Staff: Teacher 1-5 Years FTE Percent",
    "campus_2024_staff_teacher_no_degree_full_time_equiv_percent": "2024 Staff: Teacher No Degree FTE Percent",
    "campus_2024_staff_teacher_ba_degree_full_time_equiv_percent": "2024 Staff: Teacher BA Degree FTE Percent",
    "campus_2024_staff_teacher_student_ratio": "2024 Staff: Teacher to Student Ratio",
}


# 1) Build ordered lists as before, but rely on the actual columns to exist
campus_order = [f"campus_{k}" for k in cols if f"campus_{k}" in df.columns]
match_order = [f"match_{k}" for k in cols if f"match_{k}" in df.columns]

# 2) Handle distance column(s) gracefully (pick first present)
distance_candidates = ["distance_miles", "distance", "miles"]
distance_col = next((c for c in distance_candidates if c in df.columns), None)
distance_order = [distance_col] if distance_col else []

# 3) Final ordered columns: campus_* then match_* then distance
ordered_cols = campus_order + match_order + distance_order

# 4) Filter & copy
_df = df[ordered_cols].copy()

# 5) Create a rename map directly from ordered_cols and the cols dict
rename_map = {}
for col in ordered_cols:
    new_name = col  # fallback to original if no match
    if col.startswith("campus_"):
        key = col[len("campus_") :]
        if key in cols:
            new_name = cols[key]
    elif col.startswith("match_"):
        key = col[len("match_") :]
        if key in cols:
            new_name = f"NEAREST CHARTER {cols[key]}"
    elif col in ("distance_miles", "distance", "miles"):
        new_name = "Distance in Miles"
    rename_map[col] = new_name

# 6) Apply renames
df = _df.rename(columns=rename_map)

print(len(df.columns))

# 7) Export (use an absolute path and verify headers after save)
OUTPUT_XLSX = str(
    Path("Austin ISD Consolidation Plan_Exploratory Analysis_10.2025.xlsx").resolve()
)

df.to_excel(OUTPUT_XLSX, index=False, sheet_name="Sheet1")

# Sanity check: read back just the header row from the file we wrote
try:
    saved_headers = pd.read_excel(
        OUTPUT_XLSX, sheet_name="Sheet1", nrows=0
    ).columns.tolist()
    print("Saved file:", OUTPUT_XLSX)
    print("Headers after save (first 20):", saved_headers[:20])
except Exception as e:
    print("[warn] Could not verify saved headers:", e)
# In[11]:


import pandas as pd
import xlwings as xw
from pathlib import Path


def style_existing_excel(
    path: str,
    sheet_name: str = "Sheet1",
    table_name: str = "DataTbl",
    table_style: str = "TableStyleMedium9",
    open_after: bool = True,
    header_row: int = 1,
    max_col_width: float = 40,
    min_col_width: float = 10,
):
    """
    Post-process an existing .xlsx with xlwings on macOS:
      - ensure/convert used range to an Excel Table with style
      - wrap header, set header height
      - freeze panes (top header_row)
      - autofit columns then cap/min widths
      - apply simple number formats (ints, floats, percents by name)
    """
    path = str(Path(path).resolve())

    # Optional: use pandas just to infer number formats by dtype
    try:
        df_types = pd.read_excel(path, sheet_name=sheet_name)
    except Exception:
        df_types = None

    app = xw.App(visible=open_after, add_book=False)
    try:
        book = xw.Book(path)
        if sheet_name not in [s.name for s in book.sheets]:
            raise ValueError(f"Sheet {sheet_name!r} not found in workbook.")
        sht = book.sheets[sheet_name]

        # Used block starting at header_row
        top_left = sht.range((header_row, 1))
        used = top_left.expand("table")
        if used.rows.count < 2 or used.columns.count < 1:
            # Nothing to format (requires at least headers + one data row)
            book.save(path)
            if not open_after:
                book.close()
            return

        # --- macOS-safe Table creation via high-level API (fresh table over current range) ---
        # Capture the current header text exactly as written in the sheet
        header = sht.range((header_row, 1), (header_row, used.columns.count))
        current_headers = [c.value for c in header.columns[0]]

        # Remove any existing tables to avoid stale header schemas from prior runs
        try:
            for t in list(sht.tables):
                try:
                    t.delete()
                except Exception:
                    pass
        except Exception:
            pass

        # Create a new table over the exact used range and enforce style
        tbl = sht.tables.add(source=used, name=table_name, table_style_name=table_style)
        if tbl.name != table_name:
            tbl.name = table_name
        tbl.table_style = table_style

        # Reapply header cell values explicitly to prevent Excel from auto-normalizing
        try:
            tbl.header_row_range.value = [current_headers]
        except Exception:
            try:
                # Fallback: write directly to the header range if table API path fails
                header.value = current_headers
            except Exception:
                pass
        # --------------------------------------------------------------------------

        # Freeze header row AND first two columns (A & B) — macOS AppleScript backend
        # Place the selection at the first unfrozen cell: row just below header, column after two frozen columns (i.e., C)
        sht.range((header_row + 1, 3)).select()
        try:
            win = app.api.active_window  # macOS name
            # Use .set(...) for AppleScript-backed attributes
            win.split_row.set(header_row)
            win.split_column.set(2)  # freeze first two columns
            win.freeze_panes.set(True)
        except Exception:
            # Fallback via the sheet's workbook app handle
            try:
                win = sht.book.app.api.active_window
                win.split_row.set(header_row)
                win.split_column.set(2)
                win.freeze_panes.set(True)
            except Exception:
                # Non-fatal: continue without freezing panes
                pass

        # Wrap the header row text & give it breathing room
        header = sht.range((header_row, 1), (header_row, used.columns.count))
        try:
            header.api.wrap_text.set(True)  # macOS AppleScript name
        except Exception:
            # If on a different backend, xlwings may still map WrapText:
            try:
                header.api.WrapText = True
            except Exception:
                pass
        sht.range(f"{header_row}:{header_row}").row_height = 30

        # Number formats (dtype-driven + name heuristic for percents)
        if df_types is not None and not df_types.empty:
            excel_headers = [c.value for c in header.columns[0] if c.value is not None]
            for i, col_name in enumerate(excel_headers, start=1):
                rng = sht.range((header_row + 1, i), (used.last_cell.row, i))
                # dtype-driven
                if col_name in df_types.columns:
                    s = df_types[col_name]
                    if pd.api.types.is_integer_dtype(s):
                        rng.number_format = "0"
                    elif pd.api.types.is_float_dtype(s):
                        rng.number_format = "0.00"
                # name heuristic
                if "percent" in str(col_name).lower() or "%" in str(col_name):
                    rng.number_format = "0.0%"
        else:
            # Fallback: only name-based percents
            excel_headers = [c.value for c in header.columns[0] if c.value is not None]
            for i, col_name in enumerate(excel_headers, start=1):
                if "percent" in str(col_name).lower() or "%" in str(col_name):
                    rng = sht.range((header_row + 1, i), (used.last_cell.row, i))
                    rng.number_format = "0.0%"

        # --- Column sizing: wrap header FIRST, then autofit, then cap by data-only width ---
        # Excel's AutoFit can still consider header text; to avoid overly wide columns from long headers,
        # we compute a target width from the data cells only and cap the AutoFit result at that target.
        used.columns.autofit()

        # Helper: flatten 2D values to a 1D list safely
        def _flatten_2d(vals):
            if vals is None:
                return []
            if isinstance(vals, list):
                out = []
                for r in vals:
                    if isinstance(r, list):
                        out.extend(r)
                    else:
                        out.append(r)
                return out
            return [vals]

        for i in range(1, used.columns.count + 1):
            # Data-only range (exclude header row)
            data_rng = sht.range((header_row + 1, i), (used.last_cell.row, i))
            vals = _flatten_2d(data_rng.value)
            # Compute max printable length in this column (ignoring None)
            try:
                max_len = max((len(str(v)) for v in vals if v is not None), default=0)
            except Exception:
                max_len = 0
            # Target width: based on data only, within [min_col_width, max_col_width]
            target_width = max(min_col_width, min(max_col_width, max_len + 2))

            col_letter = xw.utils.col_name(i)
            # Current width after AutoFit (may include header influence)
            current_width = (
                sht.range(f"{col_letter}:{col_letter}").column_width or target_width
            )
            try:
                current_width = float(current_width)
            except Exception:
                current_width = target_width

            # Final width: the smaller of AutoFit and our data-only target
            final_width = min(current_width, target_width)
            sht.range(f"{col_letter}:{col_letter}").column_width = final_width

        # Final step: autofit row heights (after all column sizing & wrapping)
        try:
            used.rows.autofit()
            # Keep header row comfortably readable
            hdr_rng = sht.range(f"{header_row}:{header_row}")
            try:
                hrh = float(hdr_rng.row_height or 0)
            except Exception:
                hrh = 0.0
            if hrh < 30:
                hdr_rng.row_height = 30
        except Exception:
            # Non-fatal if the backend doesn't support autofit rows
            pass

        # Center the header horizontally and vertically (target table header if present)
        align_targets = [header]
        try:
            # If a Table exists, its header_row_range is the authoritative header formatting target
            if "tbl" in locals() and tbl is not None:
                try:
                    align_targets.append(tbl.header_row_range)
                except Exception:
                    pass
        except Exception:
            pass

        for tgt in align_targets:
            try:
                # macOS AppleScript backend prefers .set with string values
                tgt.api.horizontal_alignment.set("center")
                tgt.api.vertical_alignment.set("center")
                continue
            except Exception:
                pass
            try:
                # Cross-platform constants path
                from xlwings.constants import HAlign, VAlign

                tgt.api.HorizontalAlignment = HAlign.xlHAlignCenter
                tgt.api.VerticalAlignment = VAlign.xlVAlignCenter
                continue
            except Exception:
                pass
            try:
                # Numeric fallback: xlCenter == -4108
                tgt.api.HorizontalAlignment = -4108
                tgt.api.VerticalAlignment = -4108
            except Exception:
                # Last resort: try capitalized strings (some AppleScript shims accept these)
                try:
                    tgt.api.horizontal_alignment.set("Center")
                    tgt.api.vertical_alignment.set("Center")
                except Exception:
                    pass

        final_path = str(
            Path(path).with_name(f"{Path(path).stem}_FINAL{Path(path).suffix}")
        )
        book.save(final_path)
        print("Formatted Excel saved to:", final_path)
        if not open_after:
            book.close()
    finally:
        if not open_after:
            app.quit()


# In[12]:


style_existing_excel(
    OUTPUT_XLSX,  # use the same absolute path we just wrote
    sheet_name="Sheet1",
    table_name="SchoolClosures",
    table_style="TableStyleMedium2",
    open_after=True,
)
