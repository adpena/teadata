#!/usr/bin/env python3
"""Read-only closeout audit. Never merge, fetch, push, reset or delete anything."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any


def git(root: Path, *arguments: str) -> str:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes -o ConnectTimeout=5")
    completed = subprocess.run(["git", "-C", str(root), *arguments], env=env,
        capture_output=True, timeout=15, check=False)
    if completed.returncode:
        raise RuntimeError(f"Git {' '.join(arguments[:2])} failed with exit {completed.returncode}")
    if len(completed.stdout) > 2 * 1024**2:
        raise RuntimeError("Git response exceeded the audit admission")
    return os.fsdecode(completed.stdout)


def worktrees(root: Path) -> list[dict[str, str]]:
    records = []
    for block in git(root, "worktree", "list", "--porcelain", "-z").split("\0\0"):
        record = {}
        for item in block.split("\0"):
            if item:
                key, _, value = item.partition(" ")
                record[key] = value
        if record:
            records.append(record)
    return records


def audit_repository(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    result: dict[str, Any] = {"repository": str(root), "issues": []}
    issues = result["issues"]
    try:
        top = Path(git(root, "rev-parse", "--show-toplevel").strip()).resolve()
        common = Path(git(root, "rev-parse", "--path-format=absolute", "--git-common-dir").strip()).resolve()
        result.update(common_git_directory=str(common), checkout=str(top))
        if top != root:
            issues.append({"code": "not_repository_root", "path": str(top)})
        trees = worktrees(root)
        local = git(root, "for-each-ref", "--format=%(refname:short)", "refs/heads").splitlines()
        live = [line.split("\t", 1) for line in git(root, "ls-remote", "--heads", "origin").splitlines()]
        remote = {ref.removeprefix("refs/heads/"): commit for commit, ref in live}
        head = git(root, "rev-parse", "HEAD").strip()
        branch = git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
        status = git(root, "status", "--porcelain=v1", "-z", "--untracked-files=normal")
        result.update(worktrees=trees, local_branches=local, remote_branches=remote,
            head=head, branch=branch, dirty=bool(status), live_main=remote.get("main"))
        if len(trees) != 1:
            issues.append({"code": "extra_worktrees", "count": len(trees)})
        if local != ["main"]:
            issues.append({"code": "local_branches", "branches": local})
        if set(remote) != {"main"}:
            issues.append({"code": "remote_branches", "branches": sorted(remote)})
        if branch != "main":
            issues.append({"code": "checkout_not_main", "branch": branch})
        if status:
            issues.append({"code": "unfinished_changes", "action": "Reconcile tracked and untracked changes; do not discard them to pass."})
        if remote.get("main") != head:
            issues.append({"code": "head_differs_from_live_main"})
        for ref, code in (("refs/heads/main", "local_main_differs_from_live_main"),
                          ("refs/remotes/origin/main", "tracking_main_differs_from_live_main")):
            if git(root, "rev-parse", ref).strip() != remote.get("main"):
                issues.append({"code": code})
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        issues.append({"code": "audit_incomplete", "reason": str(error)})
    result["status"] = "passed" if not issues else "unresolved"
    return result


def audit_workspace(repositories: list[Path], scan_roots: list[Path]) -> dict[str, Any]:
    results = [audit_repository(root) for root in repositories]
    known = {item["common_git_directory"] for item in results if "common_git_directory" in item}
    issues: list[dict[str, Any]] = []
    if len(known) != len(results):
        issues.append({"code": "repository_set_incomplete_or_duplicated"})
    names = tuple(root.name for root in repositories)
    clones = []
    seen = set()
    for scan_root in scan_roots:
        try:
            children = list(scan_root.resolve(strict=True).iterdir())
            if len(children) > 10000:
                raise RuntimeError("Scan root exceeds the bounded immediate-directory admission")
            for path in children:
                if not path.name.startswith(names) or not (path / ".git").is_dir():
                    continue
                common = git(path, "rev-parse", "--path-format=absolute", "--git-common-dir").strip()
                if common not in known and common not in seen:
                    seen.add(common)
                    clones.append({"path": str(path), "common_git_directory": common,
                        "worktrees": worktrees(path), "local_branches": git(path, "for-each-ref", "--format=%(refname:short)", "refs/heads").splitlines()})
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            issues.append({"code": "clone_scan_incomplete", "reason": str(error)})
    if clones:
        issues.append({"code": "additional_repository_clones", "clones": clones})
    return {"schema": "workspace-closeout-audit-v1", "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not issues and all(item["status"] == "passed" for item in results) else "unresolved",
        "repositories": results, "issues": issues,
        "limits": "Read-only live Git audit. No source, evidence or release deletion; no test, deployment or artifact qualification is implied."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", action="append", type=Path, required=True, help="Canonical checkout; repeat for each task repository")
    parser.add_argument("--scan-root", action="append", type=Path, default=[], help="Inspect immediate children for additional task-repository clones")
    args = parser.parse_args()
    result = audit_workspace(args.repo, args.scan_root)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
