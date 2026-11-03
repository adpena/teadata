"""Closeout must fail visibly for the drift it is intended to prevent."""
import importlib.util
from pathlib import Path
import subprocess

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/check_workspace_closeout.py"
SPEC = importlib.util.spec_from_file_location("workspace_closeout", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repository(tmp_path):
    remote, repo = tmp_path / "remote.git", tmp_path / "project"
    subprocess.run(["git", "init", "--bare", "--initial-branch=main", str(remote)], capture_output=True, check=True)
    subprocess.run(["git", "init", "--initial-branch=main", str(repo)], capture_output=True, check=True)
    (repo / "source.txt").write_text("first\n")
    git(repo, "add", "source.txt")
    git(repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false", "commit", "-m", "Fixture")
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "-u", "origin", "main")
    return repo, remote


def codes(result):
    return {item["code"] for item in result["issues"]}


def test_clean_main_matches_live_origin(repository):
    repo, _ = repository
    assert audit.audit_workspace([repo], [repo.parent])["status"] == "passed"


def test_extra_branch_worktree_and_untracked_work_cannot_pass(repository):
    repo, _ = repository
    extra = repo.parent / "project-review"
    git(repo, "worktree", "add", "-b", "review", str(extra))
    (repo / "unfinished.txt").write_text("must be preserved\n")
    result = audit.audit_repository(repo)
    assert {"extra_worktrees", "local_branches", "unfinished_changes"} <= codes(result)
    assert (repo / "unfinished.txt").read_text() == "must be preserved\n"
    assert extra.is_dir()


def test_live_remote_divergence_is_not_hidden_by_stale_tracking_ref(repository):
    repo, remote = repository
    other = repo.parent / "other"
    subprocess.run(["git", "clone", str(remote), str(other)], capture_output=True, check=True)
    (other / "source.txt").write_text("second\n")
    git(other, "add", "source.txt")
    git(other, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false", "commit", "-m", "Advance")
    git(other, "push", "origin", "main")
    before = git(repo, "rev-parse", "refs/remotes/origin/main")
    result = audit.audit_repository(repo)
    assert {"head_differs_from_live_main", "tracking_main_differs_from_live_main"} <= codes(result)
    assert git(repo, "rev-parse", "refs/remotes/origin/main") == before  # Audit does not fetch.


def test_remote_feature_branch_is_unresolved(repository):
    repo, _ = repository
    git(repo, "push", "origin", "main:refs/heads/review")
    assert "remote_branches" in codes(audit.audit_repository(repo))


def test_independent_clone_is_detected_without_deleting_it(repository):
    repo, remote = repository
    duplicate = repo.parent / "project-duplicate"
    subprocess.run(["git", "clone", str(remote), str(duplicate)], capture_output=True, check=True)
    result = audit.audit_workspace([repo], [repo.parent])
    assert "additional_repository_clones" in codes(result)
    assert duplicate.is_dir()
