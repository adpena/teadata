from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from teadata import __version__, citation


def test_repository_citation_agrees_with_installed_metadata():
    cff = yaml.safe_load((Path(__file__).parents[1] / "CITATION.cff").read_text(encoding="utf-8"))
    with patch("teadata.release.current_release", side_effect=AssertionError("implicit data access")):
        software = citation()

    assert software.authors == tuple(
        f"{author['given-names']} {author['family-names']}" for author in cff["authors"]
    )
    assert software.repository_url == cff["url"]
    assert software.software_version == __version__
    assert software.text == (
        f"{', '.join(software.authors)}. {cff['title']} (version {__version__}) [Computer software].\n"
        f"{software.repository_url}"
    )
    assert f"title = {{{{{cff['title']}}}}}" in software.bibtex
    assert f"note = {{Version {__version__}}}" in software.bibtex
    assert software.build_id is None


def test_direct_citation_requires_a_real_build_identity():
    with pytest.raises(ValueError, match="SHA-256 build ID"):
        replace(citation(), build_id="data},\n@misc{other")
