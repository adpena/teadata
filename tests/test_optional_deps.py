import importlib
import types

import pytest


def test_import_without_optional_dependencies(monkeypatch):
    module = importlib.import_module("teadata")

    from importlib import metadata as im

    def fake_version(name):
        if name in {"shapely", "geopandas"}:
            raise im.PackageNotFoundError
        return "9999"

    monkeypatch.setattr(im, "version", fake_version)

    reloaded = importlib.reload(module)

    assert isinstance(reloaded, types.ModuleType)
    assert hasattr(reloaded, "DataEngine")


def test_import_requires_core_dependencies(monkeypatch):
    module = importlib.import_module("teadata")

    from importlib import metadata as im

    def fake_version(name):
        if name in {"pandas", "numpy"}:
            raise im.PackageNotFoundError
        return "9999"

    monkeypatch.setattr(im, "version", fake_version)

    with pytest.raises(ImportError) as excinfo:
        importlib.reload(module)

    assert "pandas" in str(excinfo.value)
    assert "numpy" in str(excinfo.value)
