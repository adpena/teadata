import pandas as pd  # type: ignore[import-untyped]
import re

tepsac_fp = (
    "/Users/adpena/PycharmProjects/teadata/teadata/data/tepsac/tepsac_geocoded.csv"
)

tepsac_df = pd.read_csv(tepsac_fp)

# Filter and reorder columns before renaming
cols = [
    "School Name",
    "School Number",
    "Grade Low",
    "Grade High",
    "Enrollment",
    "School Full Address",
    "School Website",
    "School Accreditations",
    "District Number",
    "Best Lat",
    "Best Lng",
    "Best Provider",
]
tepsac_df = tepsac_df[cols]


def to_valid_identifier(name: str) -> str:
    name = name.lower()
    name = re.sub(r"\W+", "_", name)
    name = name.strip("_")
    if name and name[0].isdigit():
        name = "col_" + name
    return name


tepsac_df.columns = [to_valid_identifier(col) for col in tepsac_df.columns]

print(tepsac_df.head().to_string())

print(tepsac_df.columns.tolist())

tepsac_df["school_number"] = (
    tepsac_df["school_number"]
    .astype(str)
    .str.zfill(9)  # pad with leading zeros up to 9 digits
    .radd("'")  # prepend a single apostrophe
)

tepsac_df["district_number"] = (
    tepsac_df["district_number"]
    .astype(str)
    .str.zfill(6)  # pad with leading zeros up to 6 digits
    .radd("'")  # prepend a single apostrophe
)

tepsac_df.to_csv(
    "/Users/adpena/PycharmProjects/teadata/teadata/data/tepsac/tepsac_geocoded_FINAL.csv",
    index=False,
)
