import json
from hashlib import sha256
import subprocess
import sys

import pytest

from teadata.verification import Check, GIT_LFS_PRE_PUSH, ReceiptError, install_pre_push_hook, run_checks, source_identity, verify_receipt
from teadata.verification import _controller_python_version, _environment_identity, _test_environment


@pytest.fixture
def project(tmp_path, monkeypatch):
    # Keep receipt-policy tests independent of concurrent package development.
    # The paired integration smoke exercises the real environment capture.
    monkeypatch.setattr("teadata.verification._environment_identity", lambda *args: {"python": _controller_python_version()})
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Local verification test"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "verification@example.invalid"], check=True)
    (tmp_path / ".gitignore").write_text(".local/\n")
    (tmp_path / "example.py").write_text("VALUE = 1\n")
    (tmp_path / "uv.lock").write_text("version = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "--quiet", "-m", "Test source"], check=True)
    return tmp_path


def passing():
    return [Check("example", ("@python", "-c", "print('checked')"))]


def paired_routing(monkeypatch, version, *, environment_version=None):
    """Capture setup routing without provisioning a product environment."""
    monkeypatch.setattr("teadata.verification._controller_python_version", lambda: version)
    probes = []
    calls = []

    def identity(python, kind, root):
        probes.append(python)
        return {"python": environment_version or version}

    def check(check, root, log, replacements, environment, timeout):
        calls.append((check, replacements.copy(), environment.copy()))
        log.write_text("owned setup-policy fixture\n")
        return {**check.to_dict(), "returncode": 0, "duration_seconds": 0,
                "timed_out": False, "interrupted": False,
                "log": log.name, "log_sha256": sha256(log.read_bytes()).hexdigest()}

    monkeypatch.setattr("teadata.verification._environment_identity", identity)
    monkeypatch.setattr("teadata.verification._run_check", check)
    return calls, probes


@pytest.mark.parametrize("version", ["3.14.9", "3.14.10", "3.15.1"])
def test_paired_setup_and_verification_share_actual_controller_runtime(project, monkeypatch, version):
    calls, probes = paired_routing(monkeypatch, version)
    receipt = run_checks(project, kind="app", paired_library=project, checks=passing())
    result = verify_receipt(receipt, project, kind="app", paired_library=project, checks=passing())
    assert result["status"] == "passed" and result["python_version"] == version
    assert calls[0][0].argv == ("uv", "sync", "--locked", "--python", version)
    environment = project / ".local" / f"check-env-{version}"
    import os
    python = str(environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    assert all(call[1]["@python"] == python for call in calls)
    assert all(call[2]["UV_PROJECT_ENVIRONMENT"] == str(environment) for call in calls)
    assert probes == [python, python, python]
    assert str(project) not in receipt.read_text()
    assert sys.executable not in receipt.read_text()


def test_paired_setup_cannot_pass_when_uv_uses_the_wrong_runtime(project, monkeypatch):
    paired_routing(monkeypatch, "3.15.1", environment_version="3.12.15")
    receipt = run_checks(project, kind="app", paired_library=project, checks=passing())
    result = json.loads(receipt.read_text())
    assert all(check["returncode"] == 0 for check in result["checks"])
    assert result["status"] == "failed"
    with pytest.raises(ReceiptError, match="passing"):
        verify_receipt(receipt, project, kind="app", paired_library=project, checks=passing())


def test_a_different_controller_runtime_cannot_reuse_a_paired_receipt(project, monkeypatch):
    paired_routing(monkeypatch, "3.14.9")
    receipt = run_checks(project, kind="app", paired_library=project, checks=passing())
    monkeypatch.setattr("teadata.verification._controller_python_version", lambda: "3.15.1")
    with pytest.raises(ReceiptError, match="different Python runtime"):
        verify_receipt(receipt, project, kind="app", paired_library=project, checks=passing())


def test_passing_receipt_is_bound_to_source_logs_and_policy(project):
    before = source_identity(project)
    receipt = run_checks(project, kind="library", checks=passing())
    assert source_identity(project) == before
    result = verify_receipt(receipt, project, kind="library", checks=passing())
    assert result["status"] == "passed"
    assert result["scope"].startswith("code-only")
    assert result["checks"][0]["returncode"] == 0
    assert str(project) not in receipt.read_text()
    assert receipt.stat().st_mode & 0o777 == 0o600


def test_source_and_lockfile_changes_reject_receipt(project):
    receipt = run_checks(project, kind="library", checks=passing())
    (project / "example.py").write_text("VALUE = 2\n")
    with pytest.raises(ReceiptError, match="stale"):
        verify_receipt(receipt, project, kind="library", checks=passing())
    (project / "example.py").write_text("VALUE = 1\n")
    (project / "uv.lock").write_text("version = 2\n")
    with pytest.raises(ReceiptError, match="stale"):
        verify_receipt(receipt, project, kind="library", checks=passing())


def test_new_untracked_source_and_new_commit_reject_receipt(project):
    receipt = run_checks(project, kind="library", checks=passing())
    (project / "new.py").write_text("VALUE = 2\n")
    with pytest.raises(ReceiptError, match="stale"):
        verify_receipt(receipt, project, kind="library", checks=passing())
    (project / "new.py").unlink()
    subprocess.run(["git", "-C", str(project), "commit", "--quiet", "--allow-empty", "-m", "Another revision"], check=True)
    with pytest.raises(ReceiptError, match="stale"):
        verify_receipt(receipt, project, kind="library", checks=passing())


def test_receipt_and_log_tampering_are_rejected(project):
    receipt = run_checks(project, kind="library", checks=passing())
    original = receipt.read_bytes()
    document = json.loads(original)
    document["checks"][0]["returncode"] = 7
    receipt.write_text(json.dumps(document))
    with pytest.raises(ReceiptError, match="integrity"):
        verify_receipt(receipt, project, kind="library", checks=passing())
    receipt.write_bytes(original)
    (receipt.parent / document["checks"][0]["log"]).write_text("different output")
    with pytest.raises(ReceiptError, match="log"):
        verify_receipt(receipt, project, kind="library", checks=passing())


def test_failure_has_durable_evidence_and_never_verifies(project):
    checks = [Check("failure", ("@python", "-c", "import sys; print('evidence'); sys.exit(7)"))]
    receipt = run_checks(project, kind="library", checks=checks)
    result = json.loads(receipt.read_text())
    assert result["status"] == "failed"
    assert result["checks"][0]["returncode"] == 7
    assert (receipt.parent / result["checks"][0]["log"]).read_text() == "evidence\n"
    with pytest.raises(ReceiptError, match="passing"):
        verify_receipt(receipt, project, kind="library", checks=checks)


def test_changing_source_during_checks_invalidates_an_otherwise_passing_run(project):
    checks = [Check("mutation", ("@python", "-c", "from pathlib import Path; Path('example.py').write_text('VALUE = 2\\n')"))]
    receipt = run_checks(project, kind="library", checks=checks)
    result = json.loads(receipt.read_text())
    assert result["checks"][0]["returncode"] == 0
    assert result["status"] == "failed" and not result["source_stable"]


def test_different_check_policy_cannot_reuse_a_passing_receipt(project):
    receipt = run_checks(project, kind="library", checks=passing())
    with pytest.raises(ReceiptError, match="policy"):
        verify_receipt(receipt, project, kind="library", checks=[Check("other", ("@python", "-c", "pass"))])


def test_missing_tool_and_timeout_are_recorded_as_failures(project):
    checks = [Check("missing", ("teadata-nonexistent-check-command",))]
    receipt = run_checks(project, kind="library", checks=checks)
    assert json.loads(receipt.read_text())["checks"][0]["returncode"] == 127
    checks = [Check("timeout", ("@python", "-c", "import time; time.sleep(60)"))]
    receipt = run_checks(project, kind="library", checks=checks, timeout=0.1)
    result = json.loads(receipt.read_text())["checks"][0]
    assert result["returncode"] == 124 and result["timed_out"]


def test_ordinary_code_identity_excludes_raw_and_generated_data(project):
    before = source_identity(project)
    for name in [".local/result.json", ".cache/snapshot.pkl", "teadata/data/source.csv", "provenance/objects/hash"]:
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("retained data")
    assert source_identity(project) == before


def test_external_source_symlink_cannot_hide_unverified_code(project, tmp_path_factory):
    external = tmp_path_factory.mktemp("external") / "code.py"
    external.write_text("VALUE = 1\n")
    (project / "linked.py").symlink_to(external)
    with pytest.raises(ReceiptError, match="symlink"):
        source_identity(project)


def test_check_environment_does_not_inherit_live_service_configuration(project, monkeypatch):
    monkeypatch.setenv("TEADATA_RELEASE_URL", "https://example.invalid/private-release")
    monkeypatch.setenv("DATABASE_URL", "production-database")
    checks = [Check("isolated", ("@python", "-c", "import os; assert os.environ['DATABASE_URL'] == ''; assert 'TEADATA_RELEASE_URL' not in os.environ"))]
    receipt = run_checks(project, kind="library", checks=checks)
    assert json.loads(receipt.read_text())["status"] == "passed"


def test_hook_installer_preserves_git_lfs_and_original_stdin(project):
    path = project / ".git/hooks/pre-push"
    path.write_text(GIT_LFS_PRE_PUSH)
    installed = install_pre_push_hook(project, kind="library")
    body = installed.read_text()
    assert '--verify-latest </dev/null' in body
    assert 'git lfs pre-push "$@"' in body
    assert (project / ".local/checks/pre-push.before-local-checks").read_text() == GIT_LFS_PRE_PUSH
    assert install_pre_push_hook(project, kind="library").read_text() == body


def test_hook_installer_refuses_custom_hooks_and_custom_hook_paths(project):
    path = project / ".git/hooks/pre-push"
    path.write_text("#!/bin/sh\nprintf 'custom'\n")
    with pytest.raises(ReceiptError, match="custom pre-push"):
        install_pre_push_hook(project, kind="app")
    assert path.read_text() == "#!/bin/sh\nprintf 'custom'\n"
    path.unlink()
    subprocess.run(["git", "-C", str(project), "config", "core.hooksPath", "custom-hooks"], check=True)
    with pytest.raises(ReceiptError, match="custom core.hooksPath"):
        install_pre_push_hook(project, kind="app")


def test_receipt_rejects_a_changed_dependency_or_tool_environment(project, monkeypatch):
    receipt = run_checks(project, kind="library", checks=passing())
    monkeypatch.setattr("teadata.verification._environment_identity", lambda *args: {"python": "changed-runtime"})
    with pytest.raises(ReceiptError, match="environment changed"):
        verify_receipt(receipt, project, kind="library", checks=passing())


def test_release_receipt_becomes_stale_when_only_an_artifact_changes(project):
    cache = project / ".cache"
    cache.mkdir()
    (cache / "entities.sqlite").write_bytes(b"original artifact")
    manifest = cache / "release.json"
    manifest.write_text(json.dumps({"artifacts": {"entities": {"filename": "entities.sqlite"}}}))
    receipt = run_checks(project, kind="library", checks=passing(), release=".cache/release.json")
    (cache / "entities.sqlite").write_bytes(b"changed artifact")
    with pytest.raises(ReceiptError, match="stale"):
        verify_receipt(receipt, project, kind="library", checks=passing(), release=".cache/release.json")


def test_release_input_cannot_capture_an_artifact_outside_its_directory(project):
    manifest = project / "release.json"
    manifest.write_text(json.dumps({"artifacts": {"entities": {"filename": "../outside"}}}))
    with pytest.raises(ReceiptError, match="portable"):
        run_checks(project, kind="library", checks=passing(), release="release.json")


def test_pytest_environment_cannot_silently_select_a_smaller_suite(project, monkeypatch):
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k only_the_easy_test")
    monkeypatch.setenv("PYTEST_PLUGINS", "caller_injected_plugin")
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    checks = [Check("full-policy", ("@python", "-c", "import os; assert not any(name.startswith('PYTEST_') for name in os.environ)"))]
    receipt = run_checks(project, kind="library", checks=checks)
    assert json.loads(receipt.read_text())["status"] == "passed"


def test_inherited_pytest_filter_cannot_hide_a_failing_required_test(project, monkeypatch):
    (project / "test_required.py").write_text(
        "def test_easy():\n    assert True\n\ndef test_hard():\n    assert False, 'required failure'\n"
    )
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k test_easy")
    checks = [Check("pytest", ("@python", "-m", "pytest", "-q", "test_required.py"))]
    receipt = run_checks(project, kind="library", checks=checks)
    document = json.loads(receipt.read_text())
    assert document["status"] == "failed"
    assert document["checks"][0]["returncode"] == 1
    assert "1 failed, 1 passed" in (receipt.parent / document["checks"][0]["log"]).read_text()


def test_environment_probe_uses_the_same_python_import_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "unrelated"))
    captured = []
    def output(argv, **kwargs):
        if "-c" in argv:
            captured.append(set(kwargs["env"]))
            return b'{}'
        return b'tool version'
    monkeypatch.setattr(subprocess, "check_output", output)
    _environment_identity(sys.executable, "library", tmp_path)
    assert captured == [set(_test_environment())]
    assert "PYTHONPATH" not in captured[0]


def test_installed_library_schema_bytes_participate_in_environment_identity(tmp_path):
    package = tmp_path / "teadata"
    package.mkdir()
    (package / "__init__.py").write_text("")
    schemas = package / "schemas"
    schemas.mkdir()
    resource = schemas / "columns.json"
    resource.write_text('{"schema_version":1,"columns":["before"]}')
    before = _environment_identity(sys.executable, "app", tmp_path)
    resource.write_text('{"schema_version":1,"columns":["after"]}')
    after = _environment_identity(sys.executable, "app", tmp_path)
    assert before["teadata_source_sha256"] != after["teadata_source_sha256"]


def test_inherited_python_optimization_cannot_disable_required_helper_assertions(project, monkeypatch):
    (project / "required_helper.py").write_text("def check():\n    assert False, 'required invariant'\n")
    (project / "test_required.py").write_text("from required_helper import check\ndef test_required():\n    check()\n")
    monkeypatch.setenv("PYTHONOPTIMIZE", "1")
    checks = [Check("pytest", ("@python", "-m", "pytest", "-q", "test_required.py"))]
    receipt = run_checks(project, kind="library", checks=checks)
    document = json.loads(receipt.read_text())
    assert document["status"] == "failed"
    assert document["checks"][0]["returncode"] == 1
    assert "required invariant" in (receipt.parent / document["checks"][0]["log"]).read_text()
    monkeypatch.setenv("PYTHONOPTIMIZE", "0")
    with pytest.raises(ReceiptError, match="passing"):
        verify_receipt(receipt, project, kind="library", checks=checks)


def test_runtime_startup_injection_is_removed_and_assertions_enabled(monkeypatch):
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONOPTIMIZE", "PYTHONWARNINGS", "NODE_OPTIONS", "NODE_PATH"):
        monkeypatch.setenv(name, "caller startup control")
    environment = _test_environment()
    assert environment["PYTHONOPTIMIZE"] == "0"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert not {"PYTHONPATH", "PYTHONHOME", "PYTHONWARNINGS", "NODE_OPTIONS", "NODE_PATH"} & environment.keys()


def test_push_hook_cannot_certify_uncommitted_fixes_instead_of_the_pushed_revision(project):
    scripts = project / "scripts"
    scripts.mkdir()
    checker = scripts / "check"
    checker.write_text("#!/bin/sh\nprintf 'checked committed source\\n'\n")
    checker.chmod(0o755)
    subprocess.run(["git", "-C", str(project), "add", "scripts/check"], check=True)
    subprocess.run(["git", "-C", str(project), "commit", "--quiet", "-m", "Add local checker"], check=True)
    hook = install_pre_push_hook(project, kind="app")
    (project / "uncommitted-fix.py").write_text("VALUE = 'not in pushed commit'\n")
    result = subprocess.run([str(hook)], cwd=project, capture_output=True, text=True)
    assert result.returncode == 1
    assert "committed revision" in result.stderr
    assert "checked committed source" not in result.stdout
    (project / "uncommitted-fix.py").unlink()
    result = subprocess.run([str(hook)], cwd=project, capture_output=True, text=True)
    assert result.returncode == 0
    assert "checked committed source" in result.stdout


def test_push_hook_binds_pushed_refs_and_preserves_exact_git_lfs_input(project, monkeypatch):
    scripts = project / "scripts"
    scripts.mkdir()
    checker = scripts / "check"
    checker.write_text("#!/bin/sh\nexit 0\n")
    checker.chmod(0o755)
    previous = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project).decode().strip()
    subprocess.run(["git", "-C", str(project), "add", "scripts/check"], check=True)
    subprocess.run(["git", "-C", str(project), "commit", "--quiet", "-m", "Add local checker"], check=True)
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project).decode().strip()
    binary = project / ".local" / "bin"
    binary.mkdir(parents=True)
    lfs = binary / "git-lfs"
    lfs.write_text('#!/bin/sh\ncat > "$TEADATA_TEST_LFS_INPUT"\n')
    lfs.chmod(0o755)
    import os
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ["PATH"])
    captured = project / ".local" / "lfs-input"
    monkeypatch.setenv("TEADATA_TEST_LFS_INPUT", str(captured))
    hook = install_pre_push_hook(project, kind="library")
    refs = f"refs/heads/main {current} refs/heads/main {previous}\n"
    result = subprocess.run([str(hook), "origin", "example.invalid"], cwd=project, input=refs, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert captured.read_text() == refs
    captured.unlink()
    other = f"refs/heads/other {previous} refs/heads/other {'0' * 40}\n"
    result = subprocess.run([str(hook), "origin", "example.invalid"], cwd=project, input=other, capture_output=True, text=True)
    assert result.returncode == 1
    assert "HEAD only" in result.stderr
    assert not captured.exists()


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_push_hook_rejects_hidden_worktree_fixes_without_changing_index_flags(project, flag):
    scripts = project / "scripts"
    scripts.mkdir()
    checker = scripts / "check"
    checker.write_text("#!/bin/sh\nprintf 'checked source\\n'\n")
    checker.chmod(0o755)
    subprocess.run(["git", "add", "scripts/check"], cwd=project, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "Add local checker"], cwd=project, check=True)
    subprocess.run(["git", "update-index", flag, "example.py"], cwd=project, check=True)
    (project / "example.py").write_text("VALUE = 'hidden fix not committed'\n")
    assert subprocess.check_output(["git", "status", "--porcelain"], cwd=project) == b""
    before = subprocess.check_output(["git", "ls-files", "-v"], cwd=project)
    hook = install_pre_push_hook(project, kind="app")
    result = subprocess.run([str(hook)], cwd=project, capture_output=True, text=True)
    assert result.returncode == 1
    assert "Hidden index flags" in result.stderr
    assert "checked source" not in result.stdout
    assert subprocess.check_output(["git", "ls-files", "-v"], cwd=project) == before
