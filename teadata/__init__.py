# teadata/__init__.py
from importlib.metadata import version, PackageNotFoundError
import sys

from .classes import DataEngine

REQS = {
    "python": (3, 11),
    "shapely": "2.0",
    "geopandas": "0.14",
    "pandas": "2.1",
    "numpy": "1.26",
}

if sys.version_info < REQS["python"]:
    raise RuntimeError(f"Python >= {'.'.join(map(str, REQS['python']))} is required.")


def _gte(installed: str, required: str) -> bool:
    from packaging import version as pv

    return pv.parse(installed) >= pv.parse(required)


def _check(name: str, minimum: str):
    try:
        v = version(name)
    except PackageNotFoundError:
        raise RuntimeError(f"{name} >= {minimum} is required but not installed.")
    if not _gte(v, minimum):
        raise RuntimeError(f"{name} >= {minimum} is required (found {v}).")


for pkg, minv in REQS.items():
    if pkg == "python":
        continue
    _check(pkg, minv)


from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("teadata")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .classes import DataEngine, District, Campus

__all__ = ["DataEngine", "District", "Campus", "__version__"]
