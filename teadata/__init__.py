# teadata/__init__.py
from importlib.metadata import PackageNotFoundError, version
import sys
import warnings

_MINIMUM_PYTHON = (3, 11)
_REQUIRED_DEPENDENCIES = {
    "pandas": "2.1",
    "numpy": "1.26",
}

_OPTIONAL_DEPENDENCIES = {
    "shapely": "2.0",
    "geopandas": "0.14",
}

if sys.version_info < _MINIMUM_PYTHON:
    raise RuntimeError(f"Python >= {'.'.join(map(str, _MINIMUM_PYTHON))} is required.")


def _gte(installed: str, required: str) -> bool:
    from packaging import version as pv

    return pv.parse(installed) >= pv.parse(required)


_required_issues: list[str] = []
for pkg, minv in _REQUIRED_DEPENDENCIES.items():
    try:
        v = version(pkg)
    except PackageNotFoundError:
        _required_issues.append(f"{pkg}>={minv} (not installed)")
        continue
    if not _gte(v, minv):
        _required_issues.append(f"{pkg}>={minv} (found {v})")

if _required_issues:
    raise ImportError(
        "teadata requires the following dependencies: "
        + ", ".join(_required_issues)
    ) from None


_optional_issues: list[str] = []
for pkg, minv in _OPTIONAL_DEPENDENCIES.items():
    try:
        v = version(pkg)
    except PackageNotFoundError:
        _optional_issues.append(f"{pkg}>={minv} (not installed)")
        continue
    if not _gte(v, minv):
        _optional_issues.append(f"{pkg}>={minv} (found {v})")

if _optional_issues:
    warnings.warn(
        "Optional geospatial dependencies are missing or out of date: "
        + ", ".join(_optional_issues)
        + ". Certain geometry features may be unavailable.",
        RuntimeWarning,
        stacklevel=2,
    )


try:
    __version__ = version("teadata")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .classes import DataEngine, District, Campus

__all__ = ["DataEngine", "District", "Campus", "__version__"]
