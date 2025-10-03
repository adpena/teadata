from pprint import pprint

import pandas as pd

tapr_df = pd.read_csv(
    "/Users/adpena/PycharmProjects/teadata/teadata/data/tapr/CAMPPROF_2023-2024.csv"
)
tapr_reference_df = pd.read_excel(
    "/Users/adpena/PycharmProjects/teadata/teadata/data/tapr/CAMPPROF_2023-2024_data_elements.xlsx"
)

rename_dict = {
    "CAMPUS": "campus_number",
    "CAMPNAME": "campus_name",
    "DISTRICT": "district_number",
    "DISTNAME": "district_name",
}

auto_rename_dict = dict(zip(tapr_reference_df["Name"], tapr_reference_df["Label"]))

tapr_df = tapr_df.rename(columns=rename_dict)

tapr_df = tapr_df.rename(columns=auto_rename_dict)

# Normalize column names to be Python-safe identifiers
# - lowercase
# - map common symbols to words
# - replace dashes/slashes/backslashes & various dashes with underscores
# - drop colons and parentheses
# - replace dots and spaces with underscores
# - collapse anything non [a-z0-9_] to underscores
# - collapse multiple underscores, trim edges
tapr_df.columns = (
    tapr_df.columns.str.lower()
    # remove punctuation we don't want to translate
    .str.replace(":", "", regex=False)
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
    # translate some symbols to words
    .str.replace("&", " and ", regex=False)
    .str.replace("%", " percent ", regex=False)
    .str.replace("#", " number ", regex=False)
    .str.replace(">", " gt ", regex=False)
    .str.replace("<", " lt ", regex=False)
    # unify dashes and slashes to underscores
    .str.replace(r"[\\/–—\-]+", "_", regex=True)
    # replace dots and whitespace with underscores
    .str.replace(r"[.]", "_", regex=True)
    .str.replace(r"\s+", "_", regex=True)
    # remove any remaining disallowed chars
    .str.replace(r"[^a-z0-9_]+", "_", regex=True)
    # squeeze repeats and trim
    .str.replace(r"__+", "_", regex=True)
    .str.strip("_")
)

pprint(list(tapr_df.columns))

tapr_df.to_excel(
    "/Users/adpena/PycharmProjects/teadata/teadata/data/tapr/CAMPPROF_2023-2024_FINAL.xlsx",
    index=False,
)
