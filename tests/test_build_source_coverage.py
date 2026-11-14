"""Private-source latest editions are not public school-year selections."""
from types import SimpleNamespace
from typing import cast

from teadata.load_data import _expected_build_sources
from teadata.teadata_config import Config


def test_future_private_edition_does_not_require_an_older_private_file():
    calls = []

    def source_only(name, year):
        calls.append((name, year))
        if name == "tefa":
            raise KeyError("TEFA begins in2027; public build is2026")
        return name == "retained_only"

    cfg = SimpleNamespace(data_sources={"tefa": {}, "accountability": {}, "retained_only": {}},
                          spatial={"districts": {}, "campuses": {}},
                          options={"private_dataset": "tefa"}, is_source_only=source_only)
    assert _expected_build_sources(cast(Config, cfg), 2026) == {
        "tefa", "accountability", "districts", "campuses"}
    assert calls == [("accountability", 2026), ("retained_only", 2026)]
