from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
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


@pytest.mark.parametrize("name", [
    "release.json",
    "peims-student-profile-2023-2026.release.json",
    "native_2026_v2.release.json",
    "2026-profile.release.json",
    "a" * 242 + ".release.json",
])
def test_citation_retains_safe_official_versioned_manifest_url_without_data_access(name):
    url = f"{citation().repository_url}/releases/download/v0.0.131/{name}"
    selected = SimpleNamespace(source=url, build_id="1" * 64)
    with patch("teadata.release.current_release", side_effect=AssertionError("implicit data access")):
        reference = citation(selected)
    assert reference.manifest_url == url
    assert reference.build_id == selected.build_id
    assert url in reference.text
    assert url.replace("_", r"\_") in reference.bibtex


@pytest.mark.parametrize("suffix", [
    "unrelated.json",
    ".release.json",
    "_hidden.release.json",
    "-hidden.release.json",
    "private.extra.release.json",
    "folder/native.release.json",
    "../native.release.json",
    r"folder\native.release.json",
    "native%2Fprivate.release.json",
    "native%2eprivate.release.json",
    "native.release.json?token=private",
    "native.release.json#private",
    "nativé.release.json",
    "a" * 243 + ".release.json",
])
def test_citation_omits_unsupported_named_manifest_spelling(suffix):
    url = f"{citation().repository_url}/releases/download/v0.0.131/{suffix}"
    reference = citation(SimpleNamespace(source=url, build_id="1" * 64))
    assert reference.manifest_url is None
    assert reference.build_id == "1" * 64
    assert url not in reference.text


@pytest.mark.parametrize("source", [
    "https://evil.example/adpena/teadata/releases/download/v0.0.131/native.release.json",
    "https://github.com@evil.example/adpena/teadata/releases/download/v0.0.131/native.release.json",
    "https://github.com/adpena/other/releases/download/v0.0.131/native.release.json",
    "https://github.com/adpena/teadata/releases/latest/download/native.release.json",
    "https://github.com/adpena/teadata/releases/download/main/native.release.json",
    "https://github.com/adpena/teadata/releases/download/v0.0.131-rc1/native.release.json",
])
def test_citation_keeps_repository_and_version_url_guard_for_named_manifests(source):
    reference = citation(SimpleNamespace(source=source, build_id="1" * 64))
    assert reference.manifest_url is None
    assert reference.build_id == "1" * 64
    assert source not in reference.text
