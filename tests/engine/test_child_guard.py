"""Model-free child-process guard tests."""

from __future__ import annotations

import importlib.util
import ast
import json
import os
import subprocess
import sys
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: A read-only reference copy, used only for its constants at collection time.
_GUARD_MODULE = _load("q3f_guard_reference", "ironmule/child_guard.py")


def test_q3f_operation_and_blocker_sets_are_exact():
    guard = _load("test_q3f_guard_module", "ironmule/child_guard.py")
    assert guard.BLOCKER_TOKENS == (
        "mlx", "llama", "ollama", "vllm", "gemma", "qwen",
        "q3c", "q3d", "ironmule", "huggingface",
    )


def test_q3f_static_scan_starts_at_actual_child_surface():
    guard = _load("test_q3f_guard_surface", "ironmule/child_guard.py")
    visited = guard.assert_source_surface(ROOT / "ironmule" / "ab.py")
    assert ("ironmule.model_identity", "resolve_model_source") in visited
    assert ("ironmule.model_identity", "scan_local_cache") in visited


@pytest.mark.parametrize("injection", [
    "\n    subprocess.Popen(['/bin/true'])\n",
    "\n    import forbidden_runtime\n",
])
def test_q3f_recursive_scan_rejects_indirect_forbidden_child_path(tmp_path, injection):
    guard = _load("test_q3f_guard_recursive", "ironmule/child_guard.py")
    source_root = tmp_path / "ironmule"
    shutil.copytree(ROOT / "ironmule", source_root)
    tune_path = source_root / "tune.py"
    source = tune_path.read_text()
    tune_path.write_text(source.replace("    model, tokenizer = load(source)\n",
                                       injection + "    model, tokenizer = load(source)\n", 1))
    with pytest.raises(guard.GuardInstallationError):
        guard.assert_source_surface(source_root / "ab.py")


@pytest.mark.parametrize("needle", [
    "    local = Path(model_id).expanduser()\n",
    "    from huggingface_hub import scan_cache_dir\n",
])
def test_q3f_recursive_scan_reaches_model_identity_functions(tmp_path, needle):
    guard = _load("test_q3f_guard_identity_recursive", "ironmule/child_guard.py")
    source_root = tmp_path / "ironmule"
    shutil.copytree(ROOT / "ironmule", source_root)
    identity_path = source_root / "model_identity.py"
    source = identity_path.read_text()
    identity_path.write_text(source.replace(needle, "    subprocess.Popen(['/bin/true'])\n" + needle, 1))
    with pytest.raises(guard.GuardInstallationError):
        guard.assert_source_surface(source_root / "ab.py")


def test_q3f_bootstrap_installs_guard_before_package_import_without_mlx(tmp_path):
    tree = ast.parse((ROOT / "ironmule" / "ab.py").read_text())
    bootstrap_node = next(node for node in tree.body
                          if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "CHILD_BOOTSTRAP"
                                  for target in node.targets))
    bootstrap = ast.literal_eval(bootstrap_node.value)
    fake_package = tmp_path / "ironmule"
    fake_package.mkdir()
    (fake_package / "__init__.py").write_text("")
    (fake_package / "ab.py").write_text(
        "import atexit, subprocess\n"
        "from ironmule import child_guard\n"
        "def _after():\n"
        "    subprocess.run(['/bin/true'], check=True)\n"
        "atexit.register(_after)\n"
        "def _child(spec):\n"
        "    return {'guard_active': child_guard.is_installed(), 'spec': spec}\n"
    )
    spec_json = json.dumps({"model": "never-imported"})
    guard_path = ROOT / "ironmule" / "child_guard.py"
    ab_path = ROOT / "ironmule" / "ab.py"
    completed = subprocess.run(
        [sys.executable, "-c", bootstrap, spec_json, str(guard_path), str(ab_path)],
        cwd=tmp_path, env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(tmp_path)},
        capture_output=True, text=True, check=False, timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    marker = next(line[2:] for line in completed.stdout.splitlines() if line.startswith("@@"))
    assert json.loads(marker) == {"guard_active": True, "spec": {"model": "never-imported"}}


@pytest.mark.parametrize("operation", sorted(_GUARD_MODULE.OPERATION_SET))
def test_q3f_guard_blocks_and_records_every_process_operation_in_isolated_child(operation):
    module_path = ROOT / "ironmule" / "child_guard.py"
    operation_code = {
        "subprocess.Popen": "import subprocess; subprocess.Popen(['/bin/true'])",
        "os.system": "os.system('true')",
        "os.fork": "os.fork()",
        "os.forkpty": "os.forkpty()",
        "os.posix_spawn": "os.posix_spawn('/bin/true', ['true'], os.environ.copy())",
        "os.posix_spawnp": "os.posix_spawnp('true', ['true'], os.environ.copy())",
        "os.setsid": "os.setsid()",
        "os.setpgid": "os.setpgid(os.getpid(), os.getpid())",
    }[operation]
    code = (
        "import importlib.util, json, os; "
        f"s=importlib.util.spec_from_file_location('g', {str(module_path)!r}); "
        "g=importlib.util.module_from_spec(s); s.loader.exec_module(g); "
        "g.install(); "
        f"\ntry: {operation_code}\nexcept g.GuardViolation: pass\n"
        "print(json.dumps(g.ledger(), sort_keys=True))"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, check=False, timeout=10)
    assert completed.returncode == 0, completed.stderr
    ledger = json.loads(completed.stdout)
    assert ledger == {"version": "ironmule.q3f_child_guard.v1", "installed": True,
                      "events": [{"event": operation, "operation": operation,
                                  "monotonic": ledger["events"][0]["monotonic"], "blocked": True}]}


def test_q3f_guard_unavailable_operation_rolls_back_wrappers_in_isolated_child():
    module_path = ROOT / "ironmule" / "child_guard.py"
    code = (
        "import importlib.util, json, os, subprocess; "
        f"s=importlib.util.spec_from_file_location('g', {str(module_path)!r}); "
        "g=importlib.util.module_from_spec(s); s.loader.exec_module(g); "
        "original=subprocess.Popen; os.setsid=None; failed=False; "
        "\ntry: g.install()\nexcept g.GuardInstallationError: failed=True\n"
        "print(json.dumps({'failed':failed, 'popen_restored':subprocess.Popen is original, 'setsid_none':os.setsid is None}))"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, check=False, timeout=10)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "failed": True, "popen_restored": True, "setsid_none": True,
    }


@pytest.mark.integration
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS kqueue/ps process identity (DATA3: fails on Kaggle Linux)")
def test_q3f_native_kqueue_monitor_detects_libc_fork():
    module_path = ROOT / "ironmule" / "child_guard.py"
    code = (
        "import ctypes, importlib.util, json, os, time; "
        f"s=importlib.util.spec_from_file_location('g', {str(module_path)!r}); "
        "g=importlib.util.module_from_spec(s); s.loader.exec_module(g); g.install(); "
        "libc=ctypes.CDLL(None); libc.fork.restype=ctypes.c_int; pid=libc.fork(); "
        "os._exit(0) if pid == 0 else None; detected=False; "
        "\nfor _ in range(20):\n"
        "    try: g.ledger()\n"
        "    except g.GuardViolation: detected=True; break\n"
        "    time.sleep(0.01)\n"
        "os.waitpid(pid, 0); print(json.dumps({'detected': detected, 'native_events': bool(g.failure_marker().get('native_events'))}))"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, check=False, timeout=10)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"detected": True, "native_events": True}
