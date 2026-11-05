"""Exercise source custody and cleanup in the maintainer packaging command."""

from pathlib import Path
import json
import os
import subprocess

import pytest

from scripts import build_release as builder


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname="teadata"\nversion="0.0.131"\n')
    (root / ".python-version").write_text("3.14.10\n")
    (root / ".gitignore").write_text("build/\nprivate.txt\n")
    (root / "source.py").write_text('VALUE = "tracked source"\n')
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.org",
                    "commit", "--quiet", "-m", "Source"], check=True)
    (root / "build/lib").mkdir(parents=True)
    (root / "build/lib/obsolete.py").write_text("OBSOLETE = True\n")
    (root / "private.txt").write_text("ignored private input\n")
    return root


def fake_build(monkeypatch, root, *, failure=False, move_source=False, on_build=None):
    original = subprocess.run
    calls = []

    def run(command, **kwargs):
        if command[:2] == ["uv", "build"]:
            calls.append((command, kwargs))
            source = Path(kwargs["cwd"])
            assert (source / "source.py").read_bytes() == (root / "source.py").read_bytes()
            assert (source / ".python-version").read_text() == "3.14.10\n"
            assert not any((source / name).exists() for name in ("build", "private.txt", ".git"))
            assert kwargs["env"]["GIT_LFS_SKIP_SMUDGE"] == "1"
            assert kwargs["env"]["UV_NO_ENV_FILE"] == "1"
            assert kwargs["env"]["SOURCE_DATE_EPOCH"].isdigit()
            if on_build is not None:
                on_build(command, kwargs)
            if move_source:
                (root / "source.py").write_text("VALUE = 'new work'\n")
            if failure:
                raise subprocess.CalledProcessError(1, command)
            destination = Path(command[command.index("--out-dir") + 1])
            destination.mkdir()
            for name in ("teadata-0.0.131-py3-none-any.whl", "teadata-0.0.131.tar.gz"):
                (destination / name).write_bytes(name.encode())
            return subprocess.CompletedProcess(command, 0)
        return original(command, **kwargs)

    monkeypatch.setattr(builder.subprocess, "run", run)
    return calls


def test_build_uses_clean_archive_and_leaves_no_temporary_source(repository, tmp_path, monkeypatch):
    calls = fake_build(monkeypatch, repository)
    manifest = builder.build_release(repository, tmp_path / "dist")
    result = json.loads(manifest.read_text())
    assert len(calls) == 1 and result["lfs_smudge_disabled"] is True
    assert {row["filename"] for row in result["artifacts"]} == {
        "teadata-0.0.131-py3-none-any.whl", "teadata-0.0.131.tar.gz"}
    assert (repository / "build/lib/obsolete.py").is_file()
    assert (repository / "private.txt").is_file()
    assert not list(tmp_path.glob(".teadata-build-*"))
    assert not list((tmp_path / "dist").glob(".teadata-build-*"))


def test_dirty_source_is_refused_before_build(repository, tmp_path, monkeypatch):
    calls = fake_build(monkeypatch, repository)
    (repository / "source.py").write_text("VALUE = 'uncommitted work'\n")
    with pytest.raises(ValueError, match="clean committed"):
        builder.build_release(repository, tmp_path / "dist")
    assert calls == [] and not (tmp_path / "dist").exists()


def test_existing_artifact_is_preserved(repository, tmp_path, monkeypatch):
    calls = fake_build(monkeypatch, repository)
    output = tmp_path / "dist"
    output.mkdir()
    existing = output / "teadata-0.0.131-py3-none-any.whl"
    existing.write_bytes(b"previous reviewed artifact")
    with pytest.raises(FileExistsError, match="Keep existing"):
        builder.build_release(repository, output)
    assert existing.read_bytes() == b"previous reviewed artifact" and calls == []


def test_wrong_commit_is_refused_before_build(repository, tmp_path, monkeypatch):
    calls = fake_build(monkeypatch, repository)
    with pytest.raises(ValueError, match="exact clean"):
        builder.build_release(repository, tmp_path / "dist", expected_commit="0" * 40)
    assert calls == [] and not (tmp_path / "dist").exists()


def test_concurrent_artifact_is_preserved_and_our_partial_output_is_removed(repository, tmp_path, monkeypatch):
    fake_build(monkeypatch, repository)
    original = os.link

    def race(source, target):
        if source.name.endswith(".tar.gz"):
            target.write_bytes(b"concurrent reviewed artifact")
            raise FileExistsError(target)
        original(source, target)

    monkeypatch.setattr(builder.os, "link", race)
    output = tmp_path / "dist"
    with pytest.raises(FileExistsError):
        builder.build_release(repository, output)
    assert [path.name for path in output.iterdir()] == ["teadata-0.0.131.tar.gz"]
    assert (output / "teadata-0.0.131.tar.gz").read_bytes() == b"concurrent reviewed artifact"
    assert not list(tmp_path.glob(".teadata-build-*"))


@pytest.mark.parametrize("replace_link", [False, True])
def test_link_interruption_and_replacement_keep_destination_ownership(
    repository, tmp_path, monkeypatch, replace_link,
):
    fake_build(monkeypatch, repository)
    original = os.link

    def fault(source, target):
        if source.name.endswith(".tar.gz"):
            raise KeyboardInterrupt
        original(source, target)
        if replace_link:
            target.unlink()
            target.write_bytes(b"another build's file")
        else:
            raise KeyboardInterrupt

    monkeypatch.setattr(builder.os, "link", fault)
    output = tmp_path / "dist"
    with pytest.raises(KeyboardInterrupt):
        builder.build_release(repository, output)
    files = list(output.iterdir())
    assert [path.name for path in files] == (["teadata-0.0.131-py3-none-any.whl"] if replace_link else [])
    if replace_link:
        assert files[0].read_bytes() == b"another build's file"
    assert not list(tmp_path.glob(".teadata-build-*"))


def test_build_records_captured_constraints_even_when_the_external_file_changes(repository, tmp_path, monkeypatch):
    from hashlib import sha256
    constraints = tmp_path / "backend.txt"
    original = b"setuptools==84.2.1\n"
    constraints.write_bytes(original)

    def change_external(command, _kwargs):
        captured = Path(command[command.index("--build-constraints") + 1])
        assert captured != constraints and captured.read_bytes() == original
        constraints.write_text("setuptools==1.0\n")

    fake_build(monkeypatch, repository, on_build=change_external)
    manifest = builder.build_release(repository, tmp_path / "dist", constraints=constraints)
    assert json.loads(manifest.read_text())["build_constraints"]["sha256"] == sha256(original).hexdigest()


@pytest.mark.parametrize("content", ["-c another.txt\n", "setuptools @ https://example.org/backend.whl\n"])
def test_constraints_require_self_contained_registry_requirements(repository, tmp_path, monkeypatch, content):
    calls = fake_build(monkeypatch, repository)
    constraints = tmp_path / "backend.txt"
    constraints.write_text(content)
    with pytest.raises(ValueError):
        builder.build_release(repository, tmp_path / "dist", constraints=constraints)
    assert calls == [] and list((tmp_path / "dist").iterdir()) == []
    assert not list(tmp_path.glob(".teadata-build-*"))


@pytest.mark.parametrize("failure,move_source", [(True, False), (False, True)])
def test_failed_or_moving_source_does_not_expose_distributions(
    repository, tmp_path, monkeypatch, failure, move_source,
):
    fake_build(monkeypatch, repository, failure=failure, move_source=move_source)
    with pytest.raises((subprocess.CalledProcessError, ValueError)):
        builder.build_release(repository, tmp_path / "dist")
    assert list((tmp_path / "dist").iterdir()) == []
    assert not list(tmp_path.glob(".teadata-build-*"))
