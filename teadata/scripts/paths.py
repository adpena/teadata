"""Portable source-data defaults for maintenance scripts and examples."""

import os
from pathlib import Path


def data_path(*parts: str) -> Path:
    """Resolve beneath TEADATA_DATA_DIR, or the package's adjacent data directory.

    Relative overrides are interpreted from the caller's working directory.
    Without an override, defaults are independent of the working directory.
    This helper does not create directories or require source data to exist.
    """
    root = os.environ.get("TEADATA_DATA_DIR")
    base = (
        Path(os.path.expandvars(root)).expanduser()
        if root
        else Path(__file__).resolve().parents[1] / "data"
    )
    return base.joinpath(*parts).resolve()
