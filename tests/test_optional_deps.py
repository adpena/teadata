import importlib
import types


def test_import_without_optional_dependencies(monkeypatch):
    module = importlib.import_module("teadata")

    from importlib import metadata as im

    def fake_version(name):
        if name in {"shapely", "geopandas", "pandas", "numpy"}:
            raise im.PackageNotFoundError
        return "9999"

    monkeypatch.setattr(im, "version", fake_version)

    reloaded = importlib.reload(module)

    assert isinstance(reloaded, types.ModuleType)
    assert hasattr(reloaded, "DataEngine")
