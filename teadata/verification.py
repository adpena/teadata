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
from typing import TYPE_CHECKING, Any, Mapping, Sequence
import uuid

if TYPE_CHECKING:
    from teadata.release import DataRelease


SCHEMA_VERSION = 1
IGNORED_SOURCE_PARTS = frozenset({
    ".git", ".local", ".venv", ".cache", "__pycache__", ".pytest_cache", ".ruff_cache",
})
TEST_ENVIRONMENT = {
    "DATABASE_URL": "", "DEBUG": "0", "SECRET_KEY": "local-verification-only",
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


def _git_environment() -> dict[str, str]:
    # Git hooks export repository-local context that overrides an explicit -C.
    # Git owns this list; preserve unrelated transport and global settings.
    local = set(subprocess.check_output(["git", "rev-parse", "--local-env-vars"],
                                       stdin=subprocess.DEVNULL, stderr=subprocess.PIPE).decode().splitlines())
    return {key: value for key, value in os.environ.items() if key not in local}


def _git(root: Path, *arguments: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *arguments], env=_git_environment(),
                                   stdin=subprocess.DEVNULL, stderr=subprocess.PIPE)


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
            Check("frontend-javascript", (
                "node", "--test", "static/js/tests/profile_map_shared.test.cjs",
                "static/js/tests/workspace_preview.test.cjs", "frontend/tests/adapter.test.mjs",
                "frontend/tests/build.test.mjs",
            )),
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
        key: value for key, value in _git_environment().items()
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


def _controller_python_version() -> str:
    return ".".join(str(part) for part in sys.version_info[:3])


def _latest_pointer(checks_root: Path, kind: str, *, paired: bool, version: str, release: str | None) -> Path:
    mode = "paired" if paired else "locked"
    name = f"latest-{kind}-{mode}-{version}"
    if release is not None:
        name += f"-release-{sha256(os.fsencode(release)).hexdigest()}"
    return checks_root / f"{name}.json"


def _check_runtime(root: Path, *, paired: bool) -> tuple[str, str, Path | None, list[Check]]:
    """One interpreter/setup authority for executing and verifying checks."""
    version = _controller_python_version()
    if not paired:
        return version, sys.executable, None, []
    environment = root / ".local" / f"check-env-{version}"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    setup = [
        Check("setup-locked", ("uv", "sync", "--locked", "--python", version)),
        Check("setup-paired", ("uv", "pip", "install", "--python", "@python", "--no-deps", "--editable", "@library")),
        Check("setup-dependencies", ("uv", "pip", "check", "--python", "@python")),
    ]
    return version, str(python), environment, setup


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
    version, python, project_environment, setup = _check_runtime(root, paired=paired_library is not None)
    if project_environment is not None:
        environment["UV_PROJECT_ENVIRONMENT"] = str(project_environment)
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
            if check.name == "setup-dependencies":
                print("Paired dependency check failed. Coordinate the app lockfile and sibling library metadata before running checks again; inspect the setup-dependencies log.", file=sys.stderr)
            break
    try:
        after = _input_identity(root, kind=kind, paired_library=paired_library, release=release)
    except (OSError, ReceiptError, subprocess.CalledProcessError) as error:
        after = {"unavailable": type(error).__name__}
    actual_environment = _try_environment_identity(python, kind, root)
    passed = len(results) == len(setup) + len(policy) and all(item["returncode"] == 0 for item in results)
    receipt = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "python_version": version,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "scope": "code-and-selected-release" if release else "code-only; raw inputs and data artifacts are not certified",
        "policy": [check.to_dict() for check in policy], "setup_policy": [check.to_dict() for check in setup],
        "inputs": before, "inputs_after": after, "source_stable": before == after,
        "environment": environment_before, "environment_after": actual_environment,
        "checks": results,
        "status": "passed" if passed and before == after and environment_before is not None and environment_before == actual_environment and environment_before.get("python") == version else "failed",
    }
    document = {**receipt, "record_id": sha256(_canonical(receipt)).hexdigest()}
    path = directory / "receipt.json"
    _write_document(path, document)
    # Focused callers retain their receipt path; they must not displace the
    # default-policy receipt used by --verify-latest and the push guard.
    if checks is None:
        _write_document(_latest_pointer(checks_root, kind, paired=paired_library is not None,
                                        version=version, release=release),
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
    version, python, _, setup = _check_runtime(root, paired=paired_library is not None)
    if record.get("python_version") != version:
        raise ReceiptError("Receipt was checked with a different Python runtime")
    current = _input_identity(root, kind=kind, paired_library=paired_library, release=release)
    if record.get("inputs") != current:
        raise ReceiptError("Receipt is stale for this revision, source, lockfile, runner, or release")
    policy = list(checks) if checks is not None else default_checks(kind, release=release)
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
    current_environment = _environment_identity(python, kind, root)
    if record.get("environment") != current_environment:
        raise ReceiptError("Dependency or tool environment changed since the run")
    if current_environment.get("python") != version:
        raise ReceiptError("Check environment uses a different Python runtime")
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
                                env=_git_environment(), stdin=subprocess.DEVNULL, capture_output=True)
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


def _publisher_reference() -> dict[str, Any]:
    from importlib.resources import files

    return json.loads(files("teadata").joinpath("reference_cases/publisher_2025.json").read_text())


def _validate_publisher_cases(release: DataRelease) -> dict[str, Any]:
    """Compare a small reviewed publisher baseline with actual stored records."""
    from types import SimpleNamespace

    from teadata.entity_store import EntityStore
    from teadata.manifest import ContractError

    reference = _publisher_reference()
    coverage = reference["release_validation"]
    selected = {case["case_id"]: case for case in reference["cases"]}
    cases = [selected[key] for key in coverage["case_ids"]]
    if not cases:
        raise ReceiptError("Publisher reference coverage is empty")
    datasets = release.catalog.datasets
    source_ids = {fact["source"]["source_id"] for case in cases for fact in case["facts"]}
    for source_id in sorted(source_ids):
        source = reference["sources"][source_id]
        selected_input = source["release_input"]
        artifact = datasets.get(selected_input["dataset"], {}).get("artifact", {})
        if artifact.get("sha256") != selected_input["sha256"]:
            raise ReceiptError(f"Publisher reference source changed: {selected_input['dataset']}; manually review its cases")
    headers = {}
    for case in cases:
        for fact in case["facts"]:
            locator = fact["source"]
            if "code" not in locator:
                continue
            source = reference["sources"][locator["source_id"]]
            dataset = source["release_input"]["dataset"]
            if dataset not in headers:
                try:
                    headers[dataset] = release.catalog.header_records(dataset)
                except ContractError as error:
                    raise ReceiptError(f"Publisher reference source lineage unavailable: {dataset}") from error
            matches = [row for row in headers[dataset] if row["field"] == fact["field"]]
            if len(matches) != 1:
                raise ReceiptError(f"Publisher reference {case['case_id']} source lineage field missing or ambiguous: {fact['field']}")
            expected_lineage = {"code": locator["code"], "label": locator["header_row1"],
                                "position": locator["column_1_based"], "input_sha256": source["sha256"]}
            for component, expected in expected_lineage.items():
                if matches[0].get(component) != expected:
                    raise ReceiptError(f"Publisher reference {case['case_id']} source lineage {component} differs: {fact['field']}")
    facts_checked = attributes_checked = 0
    with EntityStore(release.artifact("entities"), immutable=True) as store:
        for case in cases:
            scope = case["entity"]["type"]
            getter = store.get_campus if scope == "campus" else store.get_district
            record = getter(case["entity"]["number"], include_meta=True, cache=False)
            if record is None:
                raise ReceiptError(f"Publisher reference entity is absent: {case['case_id']}")
            context = SimpleNamespace(**record)
            expected_by_metric = {fact["expected"]["metric"]: fact["expected"] for fact in case["facts"]}
            # Cases may cover different publisher measures. A race/population
            # case must not invent a rating merely to join this bounded gate.
            attributes = case.get("attribute_metrics", coverage["attribute_metrics"])
            if not isinstance(attributes, dict) or any(metric not in expected_by_metric for metric in attributes.values()):
                raise ReceiptError(f"Publisher reference {case['case_id']} has invalid attribute expectations")
            for attribute, metric in attributes.items():
                expected = expected_by_metric[metric]["value"]
                if record.get(attribute) != expected or type(record.get(attribute)) is not type(expected):
                    raise ReceiptError(f"Publisher reference {case['case_id']} attribute {attribute} differs")
                attributes_checked += 1
            for fact in case["facts"]:
                expected = fact["expected"]
                metric = expected["metric"]
                value = release.catalog.measurement(context, metric, scope=scope)
                origin = value.source or {}
                components = {
                    "value": value.value == expected["value"] and type(value.value) is type(expected["value"]),
                    "status": value.status.value == expected["status"],
                    "raw": value.raw == fact["source"]["raw_token"],
                    "period": origin.get("period") == expected["period"],
                    "dataset": origin.get("dataset") == reference["sources"][fact["source"]["source_id"]]["release_input"]["dataset"],
                    "field": origin.get("field") == fact["field"],
                }
                if metric in coverage["unit_metrics"]:
                    components["unit"] = value.unit == expected["unit"]
                for component in ("reason", "lower", "upper"):
                    if component in expected:
                        components[component] = getattr(value, component) == expected[component]
                for component, matches in components.items():
                    if not matches:
                        raise ReceiptError(f"Publisher reference {case['case_id']} metric {metric} {component} differs")
                facts_checked += 1
    return {"cases": len(cases), "measurements": facts_checked, "attributes": attributes_checked,
            "uncovered": coverage["uncovered"]}


def _validate_release_provenance(release: DataRelease) -> int:
    """Gate the embedded record graph without opening historical artifact bytes."""
    from teadata.manifest import ContractError
    from teadata.provenance import validate_record_graph

    document = release.catalog.to_dict()
    records = dict(document["provenance"])
    roots = set(records)
    for source in document.get("datasets", {}).values():
        metadata = source["metadata"]
        roots.update(metadata.get("provenance_ids", ()))
        for reference in metadata.get("references", ()):
            roots.update(reference.get("provenance_ids", ()))
    if document["schema_version"] == 2:
        processing = release.processing
        records[processing["record_id"]] = processing["record"]
        roots.add(processing["record_id"])
    try:
        return len(validate_record_graph(records.__getitem__, sorted(roots)))
    except (KeyError, TypeError, ValueError, ContractError) as error:
        raise ReceiptError(f"Release provenance record graph is incomplete or invalid: {error}") from error


def _validate_tefa_cases(release: DataRelease) -> dict[str, Any]:
    """Check the acquired official listing snapshot, not the legacy workbook."""
    from importlib.resources import files
    from types import SimpleNamespace

    from teadata.entity_store import EntityStore
    from teadata.provenance import validate_record_graph

    reference = json.loads(files("teadata").joinpath("reference_cases/tefa_2026_27.json").read_text())
    document = release.catalog.to_dict()
    source = document.get("datasets", {}).get("tefa")
    if source is None:
        if any(entry.get("key", {}).get("dataset") == "tefa" for entry in document.get("editions", {}).values()):
            raise ReceiptError("TEFA private source edition lacks this reviewed scalar acceptance contract")
        return {"status": "not_present", "cases": 0,
                "uncovered": ["No TEFA dataset; private listing acceptance was not checked."]}
    expected = reference["source"]
    metadata = source["metadata"]
    source_fields = {"status": "loaded", "section": "data_sources", "resolved_year": expected["resolved_year"]}
    metadata_fields = {key: expected[key] for key in ("publisher", "source_url", "version", "period")}
    metadata_fields.update(download_url=expected["request_url"], identifiers=["tefa_listing_id"])
    if (any(source.get(key) != value for key, value in source_fields.items())
            or source.get("requested_year") not in (None, expected["requested_year"])
            or any(metadata.get(key) != value for key, value in metadata_fields.items())
            or any(source.get("artifact", {}).get(key) != value for key, value in expected["artifact"].items())):
        raise ReceiptError("TEFA private source metadata or artifact differs from the reviewed official snapshot")
    reads = source.get("reads", [])
    if (source.get("reader", {}).get("columns") != reference["reader_columns"] or not reads
            or any(read.get("columns") != list(reference["reader_columns"]) for read in reads)
            or not any(item.get("role") == "schema" and item.get("status") == "available"
                       and item.get("reviewed_sha256") == expected["column_schema_sha256"]
                       for item in metadata.get("references", []))):
        raise ReceiptError("TEFA private source reader/schema differs from the reviewed 16-field contract")
    try:
        records = document["provenance"]
        closure = validate_record_graph(records.__getitem__, metadata["provenance_ids"])
        observation = closure[expected["acquisition_record_id"]]
    except (KeyError, TypeError, ValueError) as error:
        raise ReceiptError("TEFA private source lacks its verified official HTTP acquisition") from error
    http_fields = {"kind": "acquisition", "method": "download", "dataset": "tefa", "section": "data_sources",
                   "requested_year": expected["requested_year"], "resolved_year": expected["resolved_year"],
                   "requested_url": expected["request_url"], "resolved_url": expected["request_url"]}
    parameters = {"http_method": expected["http_method"], "http_status": 200,
                  "form_url": expected["form_url"], "form_controls": expected["form_controls"]}
    if (any(observation.get(key) != value for key, value in http_fields.items())
            or any(observation.get("parameters", {}).get(key) != value for key, value in parameters.items())
            or any(observation.get("artifact", {}).get(key) != value for key, value in expected["artifact"].items())):
        raise ReceiptError("TEFA private source HTTP acquisition binding differs")

    raw_fields = measurements = 0
    entity_ids = set()
    with EntityStore(release.artifact("entities"), immutable=True) as store:
        for case in reference["cases"]:
            projected, raw = case["expected"], case["raw"]
            record = store.get_campus(projected["lookup_key"], include_meta=True, cache=False)
            if record is None:
                raise ReceiptError(f"TEFA private case is absent: {case['case_id']}")
            core = {key: projected[key] for key in ("lookup_key", "source_namespace", "source_id", "name", "grade_range", "lat", "lon")}
            core.update(source_build_id=release.build_id, is_private=True, is_charter=False,
                        campus_number=None, campus_number_canon=None, district_id=None, district_name=None,
                        district_number=None, district_number_canon=None, enrollment=None, rating=None)
            for key, value in core.items():
                if record.get(key) != value or type(record.get(key)) is not type(value):
                    raise ReceiptError(f"TEFA private {case['case_id']} core {key} differs")
            identity = record.get("id")
            if not isinstance(identity, str) or not identity or identity in entity_ids:
                raise ReceiptError(f"TEFA private {case['case_id']} stored identity is missing or collapsed")
            entity_ids.add(identity)
            meta = record.get("meta", {})
            for field, original in reference["reader_columns"].items():
                if field not in meta or _canonical(meta[field]) != _canonical(raw[original]):
                    raise ReceiptError(f"TEFA private {case['case_id']} raw field {field} differs")
                raw_fields += 1
            derived = {key: projected[key] for key in ("private_listing_location_status", "private_listing_grade_status", "school_full_address")}
            derived.update(private_source_dataset="tefa", private_source_version=expected["version"],
                           school_website=raw["website"], display_grade_range=raw["displayGradeRange"])
            for key, value in derived.items():
                if key not in meta or _canonical(meta[key]) != _canonical(value):
                    raise ReceiptError(f"TEFA private {case['case_id']} presentation {key} differs")
            for metric, dataset in reference["private_metrics"].items():
                value = release.catalog.measurement(SimpleNamespace(**record), metric, scope="campus")
                origin = value.source or {}
                if (value.status.value != "not_applicable" or value.value is not None or value.raw is not None
                        or value.reason != "outside_metric_population" or origin.get("build_id") != release.build_id
                        or origin.get("dataset") != dataset):
                    raise ReceiptError(f"TEFA private {case['case_id']} public metric {metric} is not release-bound N/A")
                measurements += 1
    return {"status": "checked", "cases": len(reference["cases"]), "raw_fields": raw_fields,
            "private_measurements": measurements, "source_acquisition": expected["acquisition_record_id"],
            "uncovered": reference["limits"]}


def _validate_release(path: str, kind: str) -> None:
    from teadata.release import DataRelease
    release = DataRelease(Path(path).expanduser())
    if release.catalog.to_dict()["schema_version"] != 2:
        raise ReceiptError("Edition catalogs require --native-release with an exact pin and retained source store")
    provenance_records = _validate_release_provenance(release)
    release.verify()
    publisher_checks = _validate_publisher_cases(release)
    private_checks = _validate_tefa_cases(release)
    if kind == "library":
        from teadata.manifest import processor_identity
        if release.catalog.to_dict()["processor_sha256"] != processor_identity():
            raise ReceiptError("Release was built from different processing code or lockfile; rebuild it")
    print(f"Verified data release {release.build_id}")
    print(f"Verified provenance record graph: {provenance_records} records")
    print("Publisher example coverage: " + json.dumps(publisher_checks, sort_keys=True))
    print("TEFA private example coverage: " + json.dumps(private_checks, sort_keys=True))


def _retained_store(path: Path):
    from teadata.provenance import ProvenanceStore

    if not (path / "records").is_dir() or not (path / "objects").is_dir():
        raise ReceiptError("An existing retained provenance store is required")
    return ProvenanceStore(path)


def _validate_native_release(path: Path, pin: str, source_store: Path) -> dict[str, Any]:
    from teadata.dataset_metadata import EditionKey
    from teadata.manifest import ContractError, canonical_json
    from teadata.native_financial_accounts import is_reviewed_budget, qualify_budget
    from teadata.native_staff_spp import qualify_staff_spp, reviewed_staff_spp
    from teadata.native_witnesses import (
        reviewed_composed_disability_references,
        reviewed_historical_peims_references,
        reviewed_native_references,
        reviewed_primary_disability_references,
        validate_native_witnesses,
    )
    from teadata.release import DataRelease
    from teadata.source_parity import validate_source_parity

    try:
        release = DataRelease(path, expected_sha256=pin, offline=True)
        if is_reviewed_budget(release.manifest_sha256):
            return qualify_budget(release, _retained_store(release.path.parent / "provenance"),
                                  _retained_store(source_store))
        staffing_or_evaluation = reviewed_staff_spp(release.manifest_sha256)
        if staffing_or_evaluation is not None:
            return qualify_staff_spp(release, _retained_store(release.path.parent / "provenance"),
                                     _retained_store(source_store), staffing_or_evaluation)
        primary_disability = reviewed_primary_disability_references(release.manifest_sha256)
        historical_peims = reviewed_historical_peims_references(release.manifest_sha256)
        inline_references = primary_disability if historical_peims is None else historical_peims
        references = reviewed_native_references() if inline_references is None else inline_references
        inline_catalog = inline_references is not None
        composed_disability = {} if inline_catalog else reviewed_composed_disability_references(release.manifest_sha256)
        if references.keys() & composed_disability.keys():
            raise ReceiptError("Reviewed native profile constituents have overlapping editions")
        references = references | composed_disability
        index = release.catalog.to_dict()
        if index["schema_version"] != (3 if inline_catalog else 4) or index["editions"].keys() != references.keys():
            raise ReceiptError("Native release does not contain exactly the reviewed editions and leaves")
        routes = ({identifier: EditionKey.from_dict(value["edition"]).identity
                   for identifier, value in index["artifacts"].items()} if inline_catalog else index["artifacts"])
        if routes != {entry["artifact"]: edition for edition, entry in references.items()}:
            raise ReceiptError("Native release does not contain exactly the reviewed editions and leaves")
        for edition, entry in references.items():
            artifact = release.catalog.artifact_ref(entry["artifact"])
            metadata = index["editions"][edition]["metadata"]
            if edition in composed_disability:
                # Externalization changes the descriptor identity while preserving the reviewed metadata.
                key = artifact.edition
                metadata = release.catalog.metadata(key.dataset, edition=key).to_dict()
            metadata_sha256 = (sha256(canonical_json(metadata).encode()).hexdigest()
                               if inline_catalog or edition in composed_disability else metadata["sha256"])
            if (artifact.sha256 != entry["sha256"] or artifact.processing_id != entry["processing_id"]
                    or metadata_sha256 != entry["metadata_sha256"]):
                raise ReceiptError("Native leaf, metadata or processing receipt differs from the reviewed identity")
        release.verify()
        store = _retained_store(source_store)
        parity_store = _retained_store(release.path.parent / "provenance") if inline_catalog else store
        parity = validate_source_parity(release, parity_store, edition_ids=sorted(references))
        if parity["status"] != "PASS_RAW_PARITY":
            return {"status": "HOLD", "build_id": release.build_id, "manifest_sha256": release.manifest_sha256,
                    "structural_integrity": True, "parity": parity, "publisher_witnesses": None}
        witnesses = validate_native_witnesses(release, store, references)
    except ContractError as error:
        raise ReceiptError(str(error)) from error
    return {"status": "PASS_REVIEWED_NATIVE", "build_id": release.build_id, "manifest_sha256": release.manifest_sha256,
            "structural_integrity": True, "parity": parity, "publisher_witnesses": witnesses}


def _validate_actual_pwr_controls(path: Path, pin: str, source_store: Path) -> dict[str, Any]:
    from teadata.manifest import ContractError
    from teadata.native_financial_accounts import qualify_actual_pwr
    from teadata.release import DataRelease

    try:
        release = DataRelease(path, expected_sha256=pin, offline=True)
        return qualify_actual_pwr(release, _retained_store(release.path.parent / "provenance"),
                                  _retained_store(source_store))
    except ContractError as error:
        raise ReceiptError(str(error)) from error


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
    parser.add_argument("--source-parity", type=Path, help="Offline raw fidelity check for selected supported native release editions")
    parser.add_argument("--native-release", type=Path, help="Qualify an exact reviewed native bundle offline")
    parser.add_argument("--native-pin", help="Trusted manifest SHA-256 for --native-release")
    parser.add_argument("--actual-pwr-controls", action="store_true",
                        help="Check only the reviewed FY2023–FY2025 district Actual PWR operating-dollar controls")
    parser.add_argument("--source-store", type=Path, help="Existing retained ProvenanceStore for source or native checks")
    parser.add_argument("--source-edition", action="append", default=[], help="Exact edition ID for --source-parity; repeat explicitly")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.paired_library and args.kind != "app":
        parser.error("Paired mode applies to the app")
    if args.verify and args.verify_latest:
        parser.error("Choose --verify or --verify-latest")
    if args.actual_pwr_controls and (args.native_release is None or args.source_parity is not None):
        parser.error("Actual PWR controls require --native-release and run separately from source parity")
    if args.install_hook and (args.verify or args.verify_latest or args.release):
        parser.error("Install the hook separately from verification or release checks")
    if args.source_parity or args.native_release:
        if (args.source_store is None or args.validate_release or args.release or args.verify
                or args.verify_latest or args.install_hook or args.paired_library):
            parser.error("Source and native checks require --source-store and run separately from other modes")
        if args.source_parity and (not args.source_edition or args.native_release or args.native_pin):
            parser.error("Source parity requires explicit --source-edition; run separately from native qualification")
        if args.native_release and (not args.native_pin or args.source_edition or args.kind != "library"):
            parser.error("Native qualification requires --kind library and --native-pin and selects its exact reviewed editions")
    elif args.source_store is not None or args.source_edition or args.native_pin:
        parser.error("Source options require --source-parity or --native-release")
    try:
        if args.source_parity:
            from teadata.release import DataRelease
            from teadata.source_parity import validate_source_parity

            report = validate_source_parity(DataRelease(args.source_parity, offline=True),
                _retained_store(args.source_store), edition_ids=args.source_edition)
            print(json.dumps(report, sort_keys=True, ensure_ascii=False, allow_nan=False))
            if report["status"] != "PASS_RAW_PARITY":
                raise SystemExit(2)
            return
        if args.native_release:
            if args.actual_pwr_controls:
                from teadata.native_financial_accounts import PASS_ACTUAL_PWR_CONTROLS

                report = _validate_actual_pwr_controls(args.native_release, args.native_pin, args.source_store)
                print(json.dumps(report, sort_keys=True, ensure_ascii=False, allow_nan=False))
                if report["status"] != PASS_ACTUAL_PWR_CONTROLS:
                    raise SystemExit(2)
                return
            report = _validate_native_release(args.native_release, args.native_pin, args.source_store)
            print(json.dumps(report, sort_keys=True, ensure_ascii=False, allow_nan=False))
            if report["status"] != "PASS_REVIEWED_NATIVE":
                raise SystemExit(2)
            return
        if args.validate_release:
            _validate_release(args.validate_release, args.kind)
            return
        if args.install_hook:
            path = install_pre_push_hook(args.root, kind=args.kind, paired=args.paired_library is not None)
            print(f"Installed local check hook: {path}")
            return
        pointer_record_id = None
        if args.verify_latest:
            checks_root = args.root / ".local/checks"
            pointer_path = _latest_pointer(checks_root, args.kind, paired=args.paired_library is not None,
                                           version=_controller_python_version(), release=args.release)
            pointer = json.loads(pointer_path.read_text())
            if (not isinstance(pointer, dict) or not isinstance(pointer.get("receipt"), str)
                    or not isinstance(pointer.get("record_id"), str)):
                raise ReceiptError("Invalid latest receipt pointer")
            relative = Path(pointer["receipt"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ReceiptError("Invalid latest receipt path")
            pointer_record_id = pointer["record_id"]
            args.verify = checks_root / relative
        if args.verify:
            document = verify_receipt(args.verify, args.root, kind=args.kind, paired_library=args.paired_library, release=args.release)
            if pointer_record_id is not None and pointer_record_id != document["record_id"]:
                raise ReceiptError("Latest receipt pointer does not match the verified record")
            print(f"Verified local checks for {document['inputs']['project']['head']}")
        else:
            path = run_checks(args.root, kind=args.kind, paired_library=args.paired_library, release=args.release, timeout=args.timeout)
            if json.loads(path.read_text())["status"] != "passed":
                raise SystemExit(1)
    except (ReceiptError, OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Local verification failed: {error}\n")


if __name__ == "__main__":
    main()
