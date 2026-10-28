import argparse
from pathlib import Path
import pandas as pd  # type: ignore[import-untyped]
import re

from teadata.scripts.paths import data_path


def to_valid_identifier(name: str) -> str:
    name = name.lower()
    name = re.sub(r"\W+", "_", name)
    name = name.strip("_")
    if name and name[0].isdigit():
        name = "col_" + name
    return name


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Normalize TEPSAC column headers.")
    parser.add_argument(
        "--input", type=Path, default=data_path("tepsac", "tepsac_geocoded.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=data_path("tepsac", "tepsac_geocoded_FINAL.csv")
    )
    args = parser.parse_args(argv)
    tepsac_df = pd.read_csv(args.input)

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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tepsac_df.to_csv(
        args.output,
        index=False,
    )


if __name__ == "__main__":
    main()
