"""The packaged selector chooses one pinned normal release without weakening offline use."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from unittest.mock import Mock

import pytest

import teadata.engine as engine_module
import teadata.release as release_module
from teadata.manifest import ContractError
from test_release import catalog as catalog, release as release  # noqa: F401


DEFAULT_URL = "https://github.com/adpena/teadata/releases/download/v0.0.160/release.json"


@pytest.fixture(autouse=True)
def isolated_selection(monkeypatch):
    for name in ("TEADATA_RELEASE", "TEADATA_RELEASE_URL", "TEADATA_RELEASE_SHA256", "TEADATA_RELEASE_OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    token = release_module._ACTIVE_RELEASE.set(None)
    release_module._load_release.cache_clear()
    yield
    release_module._load_release.cache_clear()
    release_module._ACTIVE_RELEASE.reset(token)


@pytest.fixture
def selector(tmp_path, monkeypatch):
    monkeypatch.setattr(release_module, "__file__", str(tmp_path / "release.py"))
    return tmp_path / "default_release.json"


def write_selector(path, **changes):
    value = {"schema_version": 1, "url": DEFAULT_URL, "sha256": "a" * 64}
    value.update(changes)
    path.write_text(json.dumps(value))
    return value


def test_packaged_selector_accepts_exact_versioned_pin(selector):
    write_selector(selector)
    assert release_module._packaged_default_release() == (DEFAULT_URL, "a" * 64)


def test_shipped_default_matches_the_qualified_normal_release_inventory():
    url, pin = release_module._packaged_default_release()
    selected = re.fullmatch(r"https://github\.com/adpena/teadata/releases/download/v([0-9]+\.[0-9]+\.[0-9]+)/release\.json", url)
    assert selected is not None
    assert len(pin) == 64 and all(character in "0123456789abcdef" for character in pin)
    inventory = json.loads((Path(__file__).resolve().parents[1] / "docs" /
                            f"source-integration-status-{selected[1]}.json").read_bytes())["selected_catalog"]
    assert inventory["release"]["source"] == url
    assert inventory["release"]["manifest_sha256"] == pin
    assert inventory["base_inputs"]  # Normal entity release, rather than a standalone native bundle.


@pytest.mark.parametrize("change", [
    {"schema_version": True}, {"schema_version": 1.0}, {"schema_version": "1"}, {"schema_version": 2},
    {"url": None}, {"url": "http://github.com/adpena/teadata/releases/download/v0.0.160/release.json"},
    {"url": "https://user:password@github.com/adpena/teadata/releases/download/v0.0.160/release.json"},
    {"url": DEFAULT_URL + "#fragment"}, {"url": DEFAULT_URL + "?download=1"},
    {"url": DEFAULT_URL + "\n"}, {"url": DEFAULT_URL.replace("adpena/teadata", "other/teadata")},
    {"url": DEFAULT_URL.replace("v0.0.160", "latest")},
    {"sha256": None}, {"sha256": True}, {"sha256": 1}, {"sha256": "A" * 64},
    {"sha256": "a" * 63}, {"sha256": "g" * 64}, {"sha256": "a" * 64 + "\n"},
    {"unexpected": "ignored"},
])
def test_packaged_selector_rejects_ambiguous_or_unfinalized_values(selector, change):
    write_selector(selector, **change)
    with pytest.raises(ContractError):
        release_module._packaged_default_release()


@pytest.mark.parametrize("body", [b"", b"{", b"[]", b"null", b"{}", b" " * (64 * 1024 + 1)])
def test_packaged_selector_rejects_missing_malformed_and_oversized_resource(selector, body):
    selector.write_bytes(body)
    with pytest.raises(ContractError):
        release_module._packaged_default_release()


def test_packaged_selector_missing_resource_is_contract_error(selector):
    assert not selector.exists()
    with pytest.raises(ContractError):
        release_module._packaged_default_release()


def test_unfinalized_default_fails_without_cache_or_network_fallback(selector, monkeypatch):
    write_selector(selector, sha256=None)
    loader = Mock(side_effect=AssertionError("Unfinalized selector must not open any release"))
    network = Mock(side_effect=AssertionError("Unfinalized selector must not fetch"))
    monkeypatch.setattr(release_module, "_load_release", loader)
    monkeypatch.setattr(release_module, "urlopen", network)
    with pytest.raises(ContractError):
        release_module.current_release()
    loader.assert_not_called()
    network.assert_not_called()
    monkeypatch.undo()


def test_context_and_explicit_historical_source_bypass_unfinalized_default(release, selector, monkeypatch):
    write_selector(selector, sha256=None)
    pin = sha256(release.read_bytes()).hexdigest()
    historical = release_module.DataRelease(release, expected_sha256=pin, offline=True)
    helper = Mock(side_effect=AssertionError("Explicit release must not consult packaged default"))
    monkeypatch.setattr(release_module, "_packaged_default_release", helper)
    monkeypatch.setenv("TEADATA_RELEASE", str(release))
    monkeypatch.setenv("TEADATA_RELEASE_URL", DEFAULT_URL)
    monkeypatch.setenv("TEADATA_RELEASE_SHA256", pin)
    monkeypatch.setenv("TEADATA_RELEASE_OFFLINE", "1")
    assert release_module.current_release().manifest_sha256 == historical.manifest_sha256
    monkeypatch.setenv("TEADATA_RELEASE", str(release.parent / "missing.json"))
    with release_module.using_release(historical):
        assert release_module.current_release() is historical
    helper.assert_not_called()


def test_explicit_url_keeps_its_own_pin(selector, tmp_path, monkeypatch):
    write_selector(selector, sha256=None)
    helper = Mock(side_effect=AssertionError("Explicit URL must bypass default"))
    monkeypatch.setattr(release_module, "_packaged_default_release", helper)
    monkeypatch.setenv("TEADATA_RELEASE_URL", "https://example.invalid/historical/release.json")
    monkeypatch.setenv("TEADATA_RELEASE_SHA256", "b" * 64)
    monkeypatch.setenv("TEADATA_RELEASE_OFFLINE", "1")
    monkeypatch.setattr(release_module, "user_cache_path", lambda _: tmp_path / "empty-cache")
    network = Mock(side_effect=AssertionError("Offline source must not fetch"))
    monkeypatch.setattr(release_module, "urlopen", network)
    with pytest.raises((ContractError, FileNotFoundError)):
        release_module.current_release()
    helper.assert_not_called()
    network.assert_not_called()


def test_default_pin_cannot_be_replaced_by_stray_environment_sha(selector, monkeypatch):
    write_selector(selector)
    monkeypatch.setenv("TEADATA_RELEASE_SHA256", "b" * 64)
    network = Mock(side_effect=AssertionError("Mismatched default pin must fail before fetch"))
    monkeypatch.setattr(release_module, "urlopen", network)
    with pytest.raises(ContractError):
        release_module.current_release()
    network.assert_not_called()


@pytest.mark.parametrize("cached", [False, True])
def test_default_offline_missing_or_corrupt_cache_never_fetches(selector, tmp_path, monkeypatch, cached):
    write_selector(selector)
    cache = tmp_path / "cache" / "releases"
    if cached:
        (cache / "manifests").mkdir(parents=True)
        (cache / "manifests" / (sha256(DEFAULT_URL.encode()).hexdigest() + ".json")).write_bytes(b"corrupt")
    monkeypatch.setattr(release_module, "user_cache_path", lambda _: cache.parent)
    monkeypatch.setenv("TEADATA_RELEASE_OFFLINE", "1")
    network = Mock(side_effect=AssertionError("Offline default must never fetch"))
    monkeypatch.setattr(release_module, "urlopen", network)
    with pytest.raises((ContractError, FileNotFoundError)):
        release_module.current_release()
    network.assert_not_called()


def test_default_online_selection_then_warm_offline_and_loaded_engine(release, selector, tmp_path, monkeypatch):
    pin = sha256(release.read_bytes()).hexdigest()
    write_selector(selector, sha256=pin)
    monkeypatch.setattr(release_module, "user_cache_path", lambda _: tmp_path / "cache")
    def response(url, *, timeout):
        assert timeout > 0
        filename = url.removeprefix(DEFAULT_URL.removesuffix("release.json"))
        return BytesIO((release.parent / filename).read_bytes())
    monkeypatch.setattr(release_module, "urlopen", response)
    selected = release_module.current_release()
    assert selected.source == DEFAULT_URL and selected.manifest_sha256 == pin
    selected.verify()
    release_module._load_release.cache_clear()
    monkeypatch.setenv("TEADATA_RELEASE_OFFLINE", "1")
    network = Mock(side_effect=AssertionError("Warm offline default must not fetch"))
    monkeypatch.setattr(release_module, "urlopen", network)
    restored = release_module.current_release()
    assert restored.offline and restored.manifest_sha256 == pin
    assert restored.verify().keys() == selected.verify().keys()
    engine = engine_module.DataEngine.load_default(indexes_enabled=False)
    assert engine.catalog.build_id == restored.build_id
    network.assert_not_called()


def test_empty_engine_does_not_select_or_download_default(monkeypatch):
    loader = Mock(side_effect=AssertionError("Empty constructor must not load data"))
    monkeypatch.setattr(engine_module, "current_release", loader)
    engine = engine_module.DataEngine()
    assert not engine.districts and not engine.campuses
    loader.assert_not_called()
