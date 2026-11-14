"""Catalog discovery compares exact metadata without opening observations."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from teadata.dataset_metadata import DataPeriod
from teadata.manifest import ContractError, canonical_json
from teadata.release import DataRelease, publish_release
from teadata.release_inventory import _index

from test_catalog_projections import attach_observation_child, make_edition_fixture
from test_catalog_projections import selected_catalog as selected_catalog


def native(directory, version=4, *, year=2025, families=("student", "staff")):
    directory.mkdir()
    fixture = make_edition_fixture(directory, normalized=True, families=families,
        amount_period=DataPeriod("school_year", year-1, year).to_dict())
    root = fixture["root"]
    if version == 4:
        root = publish_release(directory, fixture["catalog"].externalize_editions(), schema_version=4)
        root = root.rename(directory/"native-root.json")
    return DataRelease(root, offline=True)


@pytest.mark.parametrize("version", [3, 4])
def test_index_is_detached_complete_and_performs_no_io(tmp_path, monkeypatch, version):
    release = native(tmp_path/"native", version)
    expected = release.catalog_inventory()
    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", lambda *a, **k: pytest.fail("Index opened a file"))
        guard.setattr(DataRelease, "artifact", lambda *a, **k: pytest.fail("Index opened observations"))
        guard.setattr(DataRelease, "_resource_path", lambda *a, **k: pytest.fail("Index opened metadata"))
        assert release.catalog_inventory() == expected
        assert not release.catalog_diff(release)["release_changed"]
    assert len(expected["editions"]) == 2
    assert expected["metadata_inspected"] is False
    assert not expected["base_inputs"] and not expected["attachments"]
    assert bool(expected["metadata_resources"]) is (version == 4)
    expected["editions"][0]["dataset"] = "caller mutation"
    expected["limitations"].clear()
    assert release.catalog_inventory()["editions"][0]["dataset"] != "caller mutation"
    assert release.catalog_inventory()["limitations"]


@pytest.mark.parametrize("version", [3, 4])
def test_variable_mode_preserves_caller_caches_and_field_periods(tmp_path, monkeypatch, version):
    release = native(tmp_path/"native", version, year=2024)
    before = {key: value.copy() if isinstance(value, dict) else value
              for key, value in vars(release.catalog).items()}
    calls = []
    original = DataRelease._resource_path

    def resource(self, location, digest, size):
        calls.append(location)
        assert not location.endswith((".parquet", ".sqlite", ".gz"))
        return original(self, location, digest, size)

    monkeypatch.setattr(DataRelease, "_resource_path", resource)
    monkeypatch.setattr(DataRelease, "artifact", lambda *a, **k: pytest.fail("Read observations"))
    result = release.catalog_inventory(include_variables=True)
    assert result["metadata_inspected"] is True and calls
    assert vars(release.catalog) == before and not release._column_schema_data
    for row in result["editions"]:
        assert row["edition"] == "2025"
        amount, = (field for field in row["variables"] if field["field"] == "amount")
        assert amount["period"]["end_year"] == 2024
        assert amount["support"]["raw"] == "VALUE"
    json.dumps(result, allow_nan=False)


def test_same_edition_metadata_revision_is_visible_without_inventing_years(tmp_path):
    before = native(tmp_path/"before", year=2024)
    after = native(tmp_path/"after", year=2025)
    index = before.catalog_diff(after)
    assert index["release_changed"] and not index["metadata_inspected"]
    assert not index["changes"]["editions"]["added"] and not index["changes"]["editions"]["removed"]
    assert index["changes"]["metadata_resources"]["added"]
    full = before.catalog_diff(after, include_variables=True)
    assert len(full["changes"]["editions"]["changed"]) == 2
    assert all(item["before"]["edition_id"] == item["after"]["edition_id"]
               for item in full["changes"]["editions"]["changed"])


def test_normal_parent_index_and_attachment_change_without_child_reads(tmp_path, selected_catalog, monkeypatch):
    child = native(tmp_path/"native", version=3)
    parent = attach_observation_child(selected_catalog, child)
    doc = json.loads(parent.path.read_bytes())
    source = doc["catalog"]["native_sources"][0]
    source["manifest_sha256"] = "a"*64
    content = {key: value for key, value in doc["catalog"].items() if key != "build_id"}
    doc["catalog"]["build_id"] = sha256(canonical_json(content).encode()).hexdigest()
    record = doc["processing"]["record"]
    record["parameters"].update(build_id=doc["catalog"]["build_id"], native_sources=doc["catalog"]["native_sources"])
    doc["processing"]["record_id"] = sha256((canonical_json(record)+"\n").encode()).hexdigest()
    changed = parent.path.with_name("changed.json")
    changed.write_text(canonical_json(doc))
    after = DataRelease(changed, offline=True)
    with monkeypatch.context() as guard:
        guard.setattr(DataRelease, "_native_edition_release", lambda *a, **k: pytest.fail("Opened attachment"))
        report = parent.catalog_diff(after)
    assert len(parent.catalog_inventory()["base_inputs"]) == 1
    assert len(report["changes"]["attachments"]["changed"]) == 1
    assert not any(report["changes"]["editions"].values())


def test_attached_variables_use_normal_metadata_path_without_observations(tmp_path, selected_catalog, monkeypatch):
    child = native(tmp_path/"native", version=4)
    parent = attach_observation_child(selected_catalog, child)
    monkeypatch.setattr(DataRelease, "artifact", lambda *a, **k: pytest.fail("Opened observations"))
    result = parent.catalog_inventory(include_variables=True)
    assert len(result["base_inputs"]) == 1 and len(result["editions"]) == 2
    assert result["attachments"][0]["manifest_sha256"] == child.manifest_sha256
    assert result["editions"][0]["variables"]
    assert not getattr(parent, "_column_schema_data", {})


def test_relative_source_keeps_admitted_tree_after_working_directory_changes(tmp_path, monkeypatch):
    selected = native(tmp_path/"native")
    monkeypatch.chdir(tmp_path)
    relative = DataRelease(Path("native")/selected.path.name, offline=True)
    outside = tmp_path/"outside"
    outside.mkdir()
    monkeypatch.chdir(outside)
    report = relative.catalog_inventory(include_variables=True)
    assert len(report["editions"]) == 2 and report["metadata_inspected"]


def test_added_removed_editions_are_complete_and_no_missing_is_zero(tmp_path):
    before = native(tmp_path/"before", families=("student",))
    after = native(tmp_path/"after", families=("staff",))
    report = before.catalog_diff(after)
    assert report["changes"]["editions"]["removed"][0]["dataset"] == "campus_student"
    assert report["changes"]["editions"]["added"][0]["dataset"] == "campus_staff"
    assert not report["changes"]["editions"]["changed"]


def test_variable_mode_refuses_missing_or_changed_metadata(tmp_path):
    release = native(tmp_path/"native")
    ref = release.catalog.edition_metadata_refs()[0]
    (release.path.parent/ref.location).unlink()
    assert release.catalog_inventory()["metadata_inspected"] is False
    with pytest.raises((ContractError, FileNotFoundError)):
        release.catalog_inventory(include_variables=True)


@pytest.mark.parametrize("flag", ["false", "true", 0, 1, None])
def test_invalid_options_are_explicit_before_io(tmp_path, flag):
    release = native(tmp_path/"native")
    with pytest.raises(TypeError, match="True or False"):
        release.catalog_inventory(include_variables=flag)
    with pytest.raises(TypeError, match="True or False"):
        release.catalog_diff(release, include_variables=flag)
    with pytest.raises(TypeError, match="pinned DataRelease"):
        release.catalog_diff("release.json")


def test_duplicate_keys_fail_instead_of_collapsing_partitions():
    rows = [{"edition_id": "a", "dataset": "first"}, {"edition_id": "a", "dataset": "second"}]
    with pytest.raises(ContractError, match="duplicate edition_id"):
        _index(deepcopy(rows), "edition_id")


@pytest.mark.parametrize("online", [False, True])
def test_registration_cli_is_explicitly_offline_and_groups_exact_periods(tmp_path, monkeypatch, capsys, online):
    import sys

    from scripts import source_inventory

    selected = native(tmp_path/"native", year=2024)
    config = tmp_path/"sources.yaml"
    config.write_text("data_sources:\n  campus_student:\n    2025: unused.csv\n  not_attached:\n    2024: older.csv\n")
    arguments = ["source_inventory", "--release", str(selected.path), "--sha256", selected.manifest_sha256,
                 "--config", str(config), "--include-variables"]
    if online:
        arguments.append("--online")
    monkeypatch.setattr(sys, "argv", arguments)
    constructor = source_inventory.DataRelease
    calls = []

    def open_release(*args, **kwargs):
        calls.append(kwargs)
        return constructor(*args, **kwargs)

    monkeypatch.setattr(source_inventory, "DataRelease", open_release)
    monkeypatch.setattr(DataRelease, "artifact", lambda *a, **k: pytest.fail("CLI opened observations"))
    source_inventory.main()
    report = json.loads(capsys.readouterr().out)
    assert calls[0]["offline"] is (not online)
    assert report["summary"]["registered_ids"] == 2
    assert report["unestablished_by_this_report"]
    record, = (row for row in report["registration"] if row["dataset"] == "campus_student")
    assert record["registered_editions"] == [2025] and record["attached_report_editions"] == ["2025"]
    row, = (row for row in report["selected_catalog"]["editions"] if row["dataset"] == "campus_student")
    amount, = (group for group in row["field_periods"] if "amount" in group["fields"])
    assert amount["period"]["end_year"] == 2024  # Not the report edition.
    assert "variables" not in row


def test_registration_cli_missing_cache_explains_offline_recovery(tmp_path, monkeypatch, capsys):
    import sys

    from scripts import source_inventory

    config = tmp_path/"sources.yaml"
    config.write_text("data_sources:\n  source:\n    2025: unused.csv\n")
    monkeypatch.setattr(sys, "argv", ["source_inventory", "--release", str(tmp_path/"missing.json"),
        "--sha256", "a"*64, "--config", str(config)])
    with pytest.raises(SystemExit) as error:
        source_inventory.main()
    assert error.value.code == 2
    result = capsys.readouterr()
    assert not result.out and "Offline is the default" in result.err and "--online" in result.err


def test_registration_digest_binds_the_config_bytes_actually_parsed(tmp_path, monkeypatch):
    from argparse import Namespace

    from scripts import source_inventory

    release = native(tmp_path/"native")
    config = tmp_path/"sources.yaml"
    original = b"data_sources:\n  original:\n    2025: unused.csv\n"
    config.write_bytes(original)
    method = DataRelease.catalog_inventory

    def concurrent_edit(self, **kwargs):
        config.write_text("data_sources:\n  changed:\n    2026: different.csv\n")
        return method(self, **kwargs)

    monkeypatch.setattr(DataRelease, "catalog_inventory", concurrent_edit)
    result = source_inventory._report(Namespace(config=config, release=release.path,
        sha256=release.manifest_sha256, online=False, cache_dir=None, include_variables=False))
    assert result["registration"][0]["dataset"] == "original"
    assert result["config_sha256"] == sha256(original).hexdigest()
    assert result["config_sha256"] != sha256(config.read_bytes()).hexdigest()
