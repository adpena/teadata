#!/usr/bin/env python3
"""Build software distributions from an exact clean Git commit using uv."""

from __future__ import annotations

import argparse
from hashlib import file_digest
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
import tomllib

from packaging.requirements import Requirement


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def _clean_commit(root: Path, expected: str | None) -> str:
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("Select the repository root")
    commit = _git(root, "rev-parse", "HEAD")
    if (expected is not None and commit != expected) or _git(root, "status", "--porcelain"):
        raise ValueError("Build requires the exact clean committed source")
    return commit


def _identity(path: Path) -> dict[str, str | int]:
    with path.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    return {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": digest}


def build_release(root: Path, output: Path, *, expected_commit: str | None = None,
                  constraints: Path | None = None) -> Path:
    """Preserve ignored build trees and publish only newly built distributions.

    A private temporary Git archive excludes ignored files and disables LFS
    smudging. The checkout is checked again before exposing artifacts. This
    builds packages only; tests, data qualification and publication are separate.
    """
    root, output = root.resolve(), output.resolve()
    commit = _clean_commit(root, expected_commit)
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    if project["name"] != "teadata":
        raise ValueError("This maintainer command builds TEA Data")
    version = project["version"]
    if not isinstance(version, str) or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) is None:
        raise ValueError("Select an explicit numeric release version")
    filenames = (f"teadata-{version}-py3-none-any.whl", f"teadata-{version}.tar.gz")
    manifest_name = f"teadata-{version}.artifacts.json"
    if any((output / name).exists() or (output / name).is_symlink() for name in (*filenames, manifest_name)):
        raise FileExistsError("Keep existing distributions; select a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "UV_NO_ENV_FILE": "1",
                   "SOURCE_DATE_EPOCH": _git(root, "show", "-s", "--format=%ct", commit)}
    with TemporaryDirectory(prefix=".teadata-build-", dir=output) as temporary:
        stage = Path(temporary)
        archive, source, distributions = stage / "source.tar", stage / "source", stage / "dist"
        source.mkdir()
        subprocess.run(["git", "-C", str(root), "archive", "--format=tar", "--output", str(archive), commit],
                       env=environment, check=True)
        archive_identity = _identity(archive)
        # The archive is emitted by Git from the selected trusted commit.
        shutil.unpack_archive(archive, source, filter="data")
        command = ["uv", "build", "--no-sources", "--out-dir", str(distributions)]
        captured_constraints = None
        constraints_identity = None
        if constraints is not None:
            constraints = constraints.resolve()
            body = constraints.read_bytes()
            # Capture a self-contained set of registry requirements; nested
            # files and local/URL requirements need their own input custody.
            for line in body.decode("utf-8").splitlines():
                line = line.split("#", 1)[0].strip()
                if line and Requirement(line).url is not None:
                    raise ValueError("Use self-contained named registry build constraints")
            captured_constraints = stage / "build-constraints.txt"
            captured_constraints.write_bytes(body)
            constraints_identity = _identity(captured_constraints)
            command.extend(("--build-constraints", str(captured_constraints)))
        subprocess.run(command, cwd=source, env=environment, check=True)
        if captured_constraints is not None and _identity(captured_constraints) != constraints_identity:
            raise ValueError("Captured build constraints changed during the build")
        actual = {path.name for path in distributions.iterdir() if path.name != ".gitignore"}
        if actual != set(filenames) or any(not (distributions / name).is_file()
                                          or (distributions / name).is_symlink() for name in filenames):
            raise ValueError("Build did not produce exactly the expected wheel and source distribution")
        _clean_commit(root, commit)
        report = {
            "software_commit": commit, "software_version": version,
            "source_archive": archive_identity, "lfs_smudge_disabled": True,
            "source_date_epoch": environment["SOURCE_DATE_EPOCH"],
            "uv": subprocess.check_output(["uv", "--version"], text=True).strip(),
            "build_constraints": constraints_identity,
            "artifacts": [_identity(distributions / name) for name in filenames],
            "scope": "Software packaging only; not tests, source-data qualification or publication.",
        }
        manifest = stage / manifest_name
        manifest.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
        # Exclusive destinations prevent overwriting an artifact created by
        # another build after the initial check. Staging shares this filesystem.
        created: list[tuple[Path, int, int]] = []
        try:
            for path in (*(distributions / name for name in filenames), manifest):
                target = output / path.name
                identity = path.lstat()
                created.append((target, identity.st_dev, identity.st_ino))
                try:
                    os.link(path, target)
                except FileExistsError:
                    # This attempt did not create the colliding destination.
                    created.pop()
                    raise
        except BaseException:
            for target, device, inode in reversed(created):
                if target.exists() and (target.lstat().st_dev, target.lstat().st_ino) == (device, inode):
                    target.unlink()
            raise
    return output / manifest_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--commit")
    parser.add_argument("--build-constraints", type=Path)
    arguments = parser.parse_args()
    print(build_release(arguments.root, arguments.out_dir, expected_commit=arguments.commit,
                        constraints=arguments.build_constraints))


if __name__ == "__main__":
    main()
