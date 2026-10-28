import argparse
import requests
from pathlib import Path
from typing import Iterable
import time

import pandas as pd

from teadata.scripts.paths import data_path


def build_url(campus_number: str) -> str:
    """Constructs the public download URL for a given TEA campus number."""
    return (
        "https://rptsvr1.tea.texas.gov/cgi/sas/broker?_service=marykay&_debug=0"
        "&single=N&batch=N&app=PUBLIC&ptype=E&_program=sfadhoc.new_Campus_actual24.sas"
        f"&which_camp={campus_number}&search=dist"
    )


def download_file(
    url: str, dest: Path, session: requests.Session, chunk_size: int = 1 << 14
) -> None:
    """Stream a file from URL to dest with basic error handling."""
    resp = session.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)


def main(
    nums: Iterable[str],
    force: bool = False,
    rate_limit_delay: float = 0.5,
    download_dir: Path | None = None,
) -> None:
    download_dir = (
        download_dir
        if download_dir is not None
        else data_path("campus_peims_financials", "campus_reports")
    )
    download_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    total = len(nums)
    successes = 0
    failures = 0

    if force:
        print("Force mode enabled: existing files will be re-downloaded.")

    for i, camp_num in enumerate(nums, start=1):
        url = build_url(camp_num)
        output_file = download_dir / f"2023-2024 PEIMS_Campus_Data_{camp_num}.xlsx"

        # Skip if already downloaded and non-empty, unless force is True
        if output_file.exists() and output_file.stat().st_size > 0 and not force:
            print(f"[{i}/{total}] ✅ Already exists, skipping: {output_file}")
            continue

        print(f"[{i}/{total}] ↓ Downloading {url} -> {output_file}")
        try:
            download_file(url, output_file, session)
            print(f"    ✅ Saved: {output_file}")
            successes += 1
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            failures += 1

        time.sleep(rate_limit_delay)

    print(f"Done. {successes} succeeded, {failures} failed out of {total}.")


def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download missing campus PEIMS financial reports."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=data_path(
            "campus_peims_financials",
            "2023-2024 Campus PEIMS Actual Financials Statewide.csv",
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_path("campus_peims_financials", "campus_reports"),
    )
    parser.add_argument(
        "--force", action="store_true", help="Download reports even if already present."
    )
    args = parser.parse_args(argv)
    df = pd.read_csv(args.input)
    df = df[df["operating_payroll_gf"].isnull()].copy()
    df["campus_number"] = (
        df["campus_number"].astype(str).str.extract(r"(\d+)")[0].str.zfill(9)
    )
    main(df["campus_number"].unique(), force=args.force, download_dir=args.output_dir)


if __name__ == "__main__":
    cli()
