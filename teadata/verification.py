"""Local verification and content-bound receipts shared by TEA Data projects.

Receipts are evidence of commands run locally, not signatures or independent
review approvals. Source and log hashes prevent accidental reuse after changes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
import uuid


SCHEMA_VERSION = 1
IGNORED_SOURCE_PARTS = frozenset({
    ".git", ".local", ".venv", ".cache", "__pycache__", ".pytest_cache", ".ruff_cache",
})
TEST_ENVIRONMENT = {
    "DATABASE_URL": "", "DEBUG": "1", "SECRET_KEY": "local-verification-only",
    "PYTHON_DOTENV_DISABLED": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "TZ": "UTC",
    "PYTHONOPTIMIZE": "0", "PYTHONNOUSERSITE": "1",
}
GIT_LFS_PRE_PUSH = '''#!/bin/sh
command -v git-lfs >/dev/null 2>&1 || { printf >&2 "\\n%s\\n\\n" "This repository is configured for Git LFS but 'git-lfs' was not found on your path. If you no longer wish to use Git LFS, remove this hook by deleting the 'pre-push' file in the hooks directory (set by 'core.hookspath'; usually '.git/hooks')."; exit 2; }
git lfs pre-push "$@"
'''


class ReceiptError(RuntimeError):
    """A receipt cannot establish a passing check of the selected source."""


@dataclass(frozen=True)
class Check:
    name: str
    argv: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "argv": list(self.argv)}


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *arguments], stdin=subprocess.DEVNULL, stderr=subprocess.PIPE)


def source_identity(root: Path) -> dict[str, Any]:
    """Hash Git-visible source, including unstaged, staged, deleted, and new files.

    Raw inputs and .cache artifacts are intentionally outside ordinary code
    checks. A release check covers its selected manifest and verified artifacts.
    Output directories never participate, even before ignore rules are installed.
    """
    root = root.resolve()
    names = set(_git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").split(b"\0"))
    files = []
    for raw in sorted(names):
        if not raw:
            continue
        name = os.fsdecode(raw)
        relative = Path(name)
        if (
            any(part in IGNORED_SOURCE_PARTS or part.endswith(".egg-info") for part in relative.parts)
            or relative.parts[0] in {"build", "dist", "data"}
            or relative.parts[:2] in {("teadata", "data"), ("provenance", "objects")}
        ):
            continue
        path = root / relative
        if path.is_symlink():
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root) or not resolved.is_file():
                raise ReceiptError(f"Source symlink must resolve to a file inside the repository: {name}")
            identity = {"kind": "symlink", "target_sha256": sha256(os.fsencode(os.readlink(path))).hexdigest(),
                        "sha256": _digest(resolved)}
        elif path.is_file():
            identity = {"kind": "file", "sha256": _digest(path), "executable": bool(path.stat().st_mode & 0o111)}
        elif not path.exists():
            identity = {"kind": "deleted"}
        else:
            raise ReceiptError(f"Unsupported source entry: {name}")
        files.append({"path": relative.as_posix(), **identity})
    lockfile = root / "uv.lock"
    return {
        "head": _git(root, "rev-parse", "HEAD").decode().strip(),
        "content_sha256": sha256(_canonical(files)).hexdigest(),
        "file_count": len(files),
        "lock_sha256": _digest(lockfile) if lockfile.is_file() else None,
    }


def default_checks(kind: str, *, release: str | None = None) -> list[Check]:
    if kind == "library":
        checks = [
            Check("ruff", ("@python", "-m", "ruff", "check", ".")),
            Check("types", ("@python", "-m", "ty", "check", ".")),
            Check("dependencies", ("@python", "-m", "deptry", ".")),
            Check("pytest", ("@python", "-m", "pytest")),
        ]
    elif kind == "app":
        checks = [
            Check("ruff", ("@python", "-m", "ruff", "check", ".")),
            Check("migrations", ("@python", "manage.py", "makemigrations", "--check", "--dry-run")),
            Check("map-javascript", ("node", "--test", "static/js/tests/profile_map_shared.test.cjs")),
            Check("pytest", ("@python", "-m", "pytest")),
        ]
    else:
        raise ValueError("Project kind must be library or app")
    if release is not None:
        checks.append(Check("data-release", ("@python", "-m", "teadata.verification", "--validate-release", release, "--kind", kind)))
    return checks


def _environment_identity(python: str, kind: str, root: Path) -> dict[str, Any]:
    script = """import hashlib, importlib.metadata, importlib.util, json, platform
from pathlib import Path
spec = importlib.util.find_spec('teadata')
digest = hashlib.sha256()
if spec and spec.origin:
 root = Path(spec.origin).parent
 for path in sorted(root.rglob('*')):
  if path.is_file() and path.suffix in {'.py', '.yaml', '.md', '.json'} and not any(part in {'data', '.cache', '__pycache__'} for part in path.relative_to(root).parts):
   digest.update(path.relative_to(root).as_posix().encode() + b'\\0' + path.read_bytes())
print(json.dumps({"python": platform.python_version(), "implementation": platform.python_implementation(),
 "system": platform.system(), "release": platform.release(), "architecture": platform.machine(),
 "teadata_source_sha256": digest.hexdigest() if spec else None,
 "packages": sorted((item.metadata.get('Name', '').lower(), item.version)
                    for item in importlib.metadata.distributions())}, sort_keys=True))
"""
    environment = json.loads(subprocess.check_output([python, "-c", script], cwd=root, env=_test_environment(), stdin=subprocess.DEVNULL))
    for name in ("uv", "node") if kind == "app" else ("uv",):
        try:
            environment[name] = subprocess.check_output([name, "--version"], stdin=subprocess.DEVNULL, stderr=subprocess.STDOUT).decode().strip()
        except (OSError, subprocess.CalledProcessError):
            environment[name] = None
    return environment


def _input_identity(root: Path, *, kind: str, paired_library: Path | None, release: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": kind, "mode": "paired" if paired_library else "locked",
        "project": source_identity(root), "runner_sha256": _digest(Path(__file__)),
        "test_environment": TEST_ENVIRONMENT,
    }
    if paired_library is not None:
        result["library"] = source_identity(paired_library)
    if release is not None:
        path = Path(release).expanduser()
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise ReceiptError("A release check requires a retained local manifest")
        if path.stat().st_size > 4 * 1024 * 1024:
            raise ReceiptError("Release manifest is too large")
        result["release_manifest_sha256"] = _digest(path)
        # Capture bytes as inputs here; the data-release command owns semantic
        # validation of the catalog, processing receipt, and artifact bindings.
        # A later receipt verification must also reject changed artifact bytes.
        document = json.loads(path.read_text())
        if not isinstance(document, dict):
            raise ReceiptError("Release manifest must be a JSON object")
        artifacts = document.get("artifacts", {})
        if not isinstance(artifacts, dict) or not artifacts:
            raise ReceiptError("Release manifest has no artifacts")
        identities = {}
        for role, artifact in artifacts.items():
            filename = artifact.get("filename") if isinstance(artifact, dict) else None
            if not isinstance(filename, str) or not filename or Path(filename).name != filename or filename in {".", ".."} or "\\" in filename:
                raise ReceiptError("Release artifact names must be portable filenames")
            asset = path.parent / filename
            identities[role] = {"filename": filename, "sha256": _digest(asset), "size_bytes": asset.stat().st_size}
        result["release_artifacts"] = identities
    return result


def _write_document(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(_canonical(document))
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def _run_check(check: Check, root: Path, log: Path, replacements: Mapping[str, str], environment: Mapping[str, str], timeout: float) -> dict[str, Any]:
    argv = [replacements.get(item, item) for item in check.argv]
    started = time.monotonic()
    interrupted = timed_out = False
    with log.open("wb") as stream:
        log.chmod(0o600)
        try:
            process = subprocess.Popen(argv, cwd=root, env=dict(environment), stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _stop_process(process)
                returncode = 124
            except KeyboardInterrupt:
                interrupted = True
                _stop_process(process)
                returncode = 130
        except OSError as error:
            stream.write(f"Unable to start check: {type(error).__name__}: {error}\n".encode())
            returncode = 127
    return {
        **check.to_dict(), "returncode": returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "timed_out": timed_out, "interrupted": interrupted,
        "log": log.name, "log_sha256": _digest(log),
    }


def _test_environment() -> dict[str, str]:
    excluded = ("TEADATA_", "GOOGLE_", "RESEARCH_GOOGLE_", "EMAIL_", "REPROQ_", "PYTEST_", "PYTHON")
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(excluded) and key not in {
            "DATABASE_URL", "DJANGO_SETTINGS_MODULE", "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV",
            "NODE_OPTIONS", "NODE_PATH",
        }
    }
    return {**environment, **TEST_ENVIRONMENT}


def _try_environment_identity(python: str, kind: str, root: Path) -> dict[str, Any] | None:
    try:
        return _environment_identity(python, kind, root)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def run_checks(
    root: Path,
    *,
    kind: str,
    paired_library: Path | None = None,
    release: str | None = None,
    checks: Sequence[Check] | None = None,
    timeout: float = 900,
) -> Path:
    root = root.resolve()
    paired_library = paired_library.resolve() if paired_library is not None else None
    before = _input_identity(root, kind=kind, paired_library=paired_library, release=release)
    policy = list(checks) if checks is not None else default_checks(kind, release=release)
    if not policy or len({check.name for check in policy}) != len(policy):
        raise ValueError("Checks must be nonempty and uniquely named")
    checks_root = root / ".local" / "checks"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
    directory = checks_root / run_id
    directory.mkdir(parents=True, mode=0o700)
    environment = _test_environment()
    python = sys.executable
    setup = []
    if paired_library is not None:
        check_python = Path(".local/check-env") / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        python = str(root / check_python)
        environment["UV_PROJECT_ENVIRONMENT"] = str(root / ".local/check-env")
        setup = [
            Check("setup-locked", ("uv", "sync", "--locked", "--python", "3.12")),
            Check("setup-paired", ("uv", "pip", "install", "--python", "@python", "--no-deps", "--editable", "@library")),
        ]
    replacements = {"@python": python, "@library": str(paired_library or "")}
    results = []
    environment_before = None
    for index, check in enumerate([*setup, *policy]):
        if index == len(setup):
            environment_before = _try_environment_identity(python, kind, root)
        print(f"[{check.name}] running", flush=True)
        result = _run_check(check, root, directory / f"{index:02d}-{check.name}.log", replacements, environment, timeout)
        results.append(result)
        print(f"[{check.name}] {'passed' if result['returncode'] == 0 else 'FAILED'} ({result['duration_seconds']}s)", flush=True)
        if result["returncode"] != 0:
            break
    try:
        after = _input_identity(root, kind=kind, paired_library=paired_library, release=release)
    except (OSError, ReceiptError, subprocess.CalledProcessError) as error:
        after = {"unavailable": type(error).__name__}
    actual_environment = _try_environment_identity(python, kind, root)
    passed = len(results) == len(setup) + len(policy) and all(item["returncode"] == 0 for item in results)
    receipt = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "scope": "code-and-selected-release" if release else "code-only; raw inputs and data artifacts are not certified",
        "policy": [check.to_dict() for check in policy], "setup_policy": [check.to_dict() for check in setup],
        "inputs": before, "inputs_after": after, "source_stable": before == after,
        "environment": environment_before, "environment_after": actual_environment,
        "checks": results,
        "status": "passed" if passed and before == after and environment_before is not None and environment_before == actual_environment else "failed",
    }
    document = {**receipt, "record_id": sha256(_canonical(receipt)).hexdigest()}
    path = directory / "receipt.json"
    _write_document(path, document)
    _write_document(checks_root / f"latest-{kind}-{'paired' if paired_library else 'locked'}.json",
                    {"receipt": f"{run_id}/receipt.json", "record_id": document["record_id"]})
    print(f"Receipt: {path}", flush=True)
    if before != after:
        print("Source changed during checks; this receipt cannot be used for landing.", file=sys.stderr)
    return path


def verify_receipt(
    path: Path,
    root: Path,
    *,
    kind: str,
    paired_library: Path | None = None,
    release: str | None = None,
    checks: Sequence[Check] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    path = path.resolve()
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ReceiptError("Receipt is too large")
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise ReceiptError("Receipt must be a JSON object")
    record = {key: value for key, value in document.items() if key != "record_id"}
    if document.get("record_id") != sha256(_canonical(record)).hexdigest() or record.get("schema_version") != SCHEMA_VERSION:
        raise ReceiptError("Receipt integrity or schema mismatch")
    if record.get("status") != "passed" or not record.get("source_stable") or record.get("inputs") != record.get("inputs_after"):
        raise ReceiptError("Receipt does not contain a complete stable passing run")
    current = _input_identity(root, kind=kind, paired_library=paired_library, release=release)
    if record.get("inputs") != current:
        raise ReceiptError("Receipt is stale for this revision, source, lockfile, runner, or release")
    policy = list(checks) if checks is not None else default_checks(kind, release=release)
    setup = [Check("setup-locked", ("uv", "sync", "--locked", "--python", "3.12")),
             Check("setup-paired", ("uv", "pip", "install", "--python", "@python", "--no-deps", "--editable", "@library"))] if paired_library else []
    if record.get("policy") != [item.to_dict() for item in policy] or record.get("setup_policy") != [item.to_dict() for item in setup]:
        raise ReceiptError("Receipt does not match the current check policy")
    expected = [*setup, *policy]
    results = record.get("checks", [])
    if len(results) != len(expected):
        raise ReceiptError("Receipt is missing required checks")
    for check, result in zip(expected, results):
        if any(result.get(key) != value for key, value in check.to_dict().items()) or result.get("returncode") != 0:
            raise ReceiptError("Receipt contains a failed or different check")
        name = result.get("log")
        if not isinstance(name, str) or Path(name).name != name or "\\" in name:
            raise ReceiptError("Invalid receipt log path")
        log = path.parent / name
        if log.is_symlink() or not log.is_file() or _digest(log) != result.get("log_sha256"):
            raise ReceiptError("Check log is missing or changed")
    python = str(root / ".local/check-env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")) if paired_library else sys.executable
    if record.get("environment") != _environment_identity(python, kind, root):
        raise ReceiptError("Dependency or tool environment changed since the run")
    if record.get("environment") != record.get("environment_after"):
        raise ReceiptError("Dependency or tool environment changed during checks")
    return document


def install_pre_push_hook(root: Path, *, kind: str, paired: bool = False) -> Path:
    """Install only over no hook, this exact hook, or Git LFS's generated hook.

    Custom hook paths and custom hooks require their owner to compose the check
    command manually. No existing custom hook is discarded or silently bypassed.
    """
    root = root.resolve()
    configured = subprocess.run(["git", "-C", str(root), "config", "--get", "core.hooksPath"],
                                stdin=subprocess.DEVNULL, capture_output=True)
    if configured.returncode == 0:
        raise ReceiptError("A custom core.hooksPath is configured; retain it and chain scripts/check manually")
    if configured.returncode != 1:
        raise ReceiptError("Could not inspect the Git hook configuration")
    path = Path(_git(root, "rev-parse", "--git-path", "hooks/pre-push").decode().strip())
    if not path.is_absolute():
        path = root / path
    mode = " --paired" if paired else ""
    existing = path.read_text() if path.is_file() and not path.is_symlink() else None
    attributes = root / ".gitattributes"
    uses_lfs = kind == "library" or existing == GIT_LFS_PRE_PUSH or (attributes.is_file() and "filter=lfs" in attributes.read_text())
    body = f'''#!/bin/sh
# TEA Data local verification hook v1
set -eu
project_root=$(git rev-parse --show-toplevel)
head_revision=$(git rev-parse HEAD)
git ls-files -v | while IFS= read -r entry; do
  case "$entry" in
    [a-zS]' '*)
      printf >&2 '%s\\n' 'Hidden index flags prevent committed-source verification; use a clean checkout without assume-unchanged or skip-worktree.'
      exit 1
      ;;
  esac
done
pending_source=$(git status --porcelain --untracked-files=normal)
if [ -n "$pending_source" ]; then
  printf >&2 '%s\\n' 'Commit or isolate pending changes before pushing; local checks must cover the committed revision.'
  exit 1
fi
push_ref_updates=$(mktemp)
trap 'rm -f "$push_ref_updates"' EXIT
cat > "$push_ref_updates"
while read -r local_ref local_sha remote_ref remote_sha; do
  case "$local_sha" in
    *[!0]*)
      pushed_revision=$(git rev-parse --verify "$local_sha^{{}}")
      if [ "$pushed_revision" != "$head_revision" ]; then
        printf >&2 '%s\\n' 'Check out and verify each revision before pushing it; this guard covers HEAD only.'
        exit 1
      fi
      ;;
  esac
done < "$push_ref_updates"
"$project_root/scripts/check"{mode} --verify-latest </dev/null || "$project_root/scripts/check"{mode} </dev/null
test "$head_revision" = "$(git rev-parse HEAD)"
'''
    if uses_lfs:
        body += 'git lfs pre-push "$@" < "$push_ref_updates"\n'
    if path.is_symlink() or (path.exists() and existing not in {body, GIT_LFS_PRE_PUSH}):
        raise ReceiptError("An existing custom pre-push hook is present; retain it and chain scripts/check manually")
    if existing == GIT_LFS_PRE_PUSH:
        backup = root / ".local/checks/pre-push.before-local-checks"
        backup.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not backup.exists():
            with backup.open("xb") as stream:
                stream.write(existing.encode())
            backup.chmod(0o600)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(body)
    try:
        temporary.chmod(0o755)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _validate_release(path: str, kind: str) -> None:
    from teadata.release import DataRelease
    release = DataRelease(Path(path).expanduser())
    release.verify()
    if kind == "library":
        from teadata.manifest import processor_identity
        if release.catalog.to_dict()["processor_sha256"] != processor_identity():
            raise ReceiptError("Release was built from different processing code or lockfile; rebuild it")
    print(f"Verified data release {release.build_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("library", "app"), required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--paired-library", type=Path)
    parser.add_argument("--release", help="Retained local release manifest to verify, in addition to code checks")
    parser.add_argument("--verify", type=Path, help="Verify an existing receipt instead of executing checks")
    parser.add_argument("--verify-latest", action="store_true")
    parser.add_argument("--install-hook", action="store_true", help="Install an optional pre-push code-check hook, preserving Git LFS")
    parser.add_argument("--timeout", type=float, default=900, help="Maximum seconds for each check")
    parser.add_argument("--validate-release", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.validate_release:
        _validate_release(args.validate_release, args.kind)
        return
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.paired_library and args.kind != "app":
        parser.error("Paired mode applies to the app")
    if args.verify and args.verify_latest:
        parser.error("Choose --verify or --verify-latest")
    if args.install_hook and (args.verify or args.verify_latest or args.release):
        parser.error("Install the hook separately from verification or release checks")
    try:
        if args.install_hook:
            path = install_pre_push_hook(args.root, kind=args.kind, paired=args.paired_library is not None)
            print(f"Installed local check hook: {path}")
            return
        if args.verify_latest:
            mode = "paired" if args.paired_library else "locked"
            checks_root = args.root / ".local/checks"
            pointer = json.loads((checks_root / f"latest-{args.kind}-{mode}.json").read_text())
            relative = Path(pointer["receipt"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ReceiptError("Invalid latest receipt path")
            args.verify = checks_root / relative
        if args.verify:
            document = verify_receipt(args.verify, args.root, kind=args.kind, paired_library=args.paired_library, release=args.release)
            print(f"Verified local checks for {document['inputs']['project']['head']}")
        else:
            path = run_checks(args.root, kind=args.kind, paired_library=args.paired_library, release=args.release, timeout=args.timeout)
            if json.loads(path.read_text())["status"] != "passed":
                raise SystemExit(1)
    except (ReceiptError, OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Local verification failed: {error}\n")


if __name__ == "__main__":
    main()
