# teadata/__init__.py
from importlib.metadata import PackageNotFoundError, version
from importlib import import_module
import sys
from typing import TYPE_CHECKING, Any
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
        "teadata requires the following dependencies: " + ", ".join(_required_issues)
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

if TYPE_CHECKING:
    from .classes import DataEngine, District, Campus
    from .dataset_metadata import DataPeriod, DatasetMetadata, ResolvedDataset
    from .provenance import ProvenanceStore
    from .manifest import DatasetCatalog
    from .release import DataRelease, current_release
    from .teadata_config import load_config
    from .values import Measurement, ValueStatus, ValuePolicy, EstimationPolicy, measure, aggregate


_EXPORT_MODULES = {
    **dict.fromkeys(("DataEngine", "District", "Campus"), ".classes"),
    **dict.fromkeys(("DataPeriod", "DatasetMetadata", "ResolvedDataset"), ".dataset_metadata"),
    "ProvenanceStore": ".provenance",
    "DatasetCatalog": ".manifest",
    **dict.fromkeys(("DataRelease", "current_release"), ".release"),
    "load_config": ".teadata_config",
    **dict.fromkeys(("Measurement", "ValueStatus", "ValuePolicy", "EstimationPolicy", "measure", "aggregate"), ".values"),
}


def __getattr__(name: str) -> Any:
    """Load public symbols on demand; metadata discovery does not load the engine.

    Heavy entity exports still use the existing classes module so this step
    preserves the released snapshot module paths until their planned retirement.
    """
    module = _EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _EXPORT_MODULES.keys())

__all__ = [
    "DataEngine",
    "District",
    "Campus",
    "__version__",
    "DataPeriod",
    "DatasetMetadata",
    "ResolvedDataset",
    "ProvenanceStore",
    "DatasetCatalog",
    "DataRelease",
    "current_release",
    "load_config",
    "Measurement",
    "ValueStatus",
    "ValuePolicy",
    "EstimationPolicy",
    "measure",
    "aggregate",
]
