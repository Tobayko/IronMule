"""Model-free Q3f guard and same-UID attribution tests."""

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


q3b = _load("test_q3f_q3b", "research/q3b_residual_swap_canary.py")


def _snapshot(rows, comm, stamp):
    return {
        "monotonic": stamp, "command_ok": True, "parse_ok": True,
        "records": rows, "gone_pids": [], "enrichment": [], "error": None,
        "comm": {"monotonic": stamp + 0.01, "command_ok": True,
                  "parse_ok": True, "records": comm, "error": None},
    }


def _fixtures(args: str = "/bin/sleep 20", *, second_args: str | None = None):
    root = {"pid": 1, "ppid": 0, "pgid": 1, "uid": 0, "stat": "Ss",
            "start": "root", "args": "/sbin/launchd"}
    row = {"pid": 3, "ppid": 1, "pgid": 3, "sid": 3, "uid": 501,
           "stat": "S", "start": "new", "args": args}
    other = dict(row, args=second_args if second_args is not None else args)
    comm = {"pid": 3, "comm": "sleep"}
    other_comm = dict(comm)
    return [_snapshot([root, row], [dict(comm), {"pid": 1, "comm": "launchd"}], 1.0),
            _snapshot([root, other], [other_comm, {"pid": 1, "comm": "launchd"}], 2.0)]


def _identity():
    return {"worker_pid": 2, "parent_pid": 1, "worker_ancestor_pids": [1],
            "pgid": 2, "sid": 2, "uid": 501,
            "known_process_starts": {"2": "worker"},
            "known_process_ppids": {"2": 1},
            "known_process_sids": {"2": 2},
            "known_process_pgids": {"2": 2}}


def _guard_proof():
    return ([{"pid": 2, "guard": {"version": "ironmule.q3f_child_guard.v1",
                                  "installed": True, "events": []}}],
            [{"pid": 2, "ppid": 1, "pgid": 2, "sid": 2, "uid": 501,
              "start": "worker", "callback_monotonic": 0.5,
              "guard_version": "ironmule.q3f_child_guard.v1", "guard_event_count": 0}])


def test_q3f_operation_and_blocker_sets_are_exact():
    guard = _load("test_q3f_guard_module", "ironmule/q3f_child_guard.py")
    assert q3b.Q3F_GUARD_OPERATIONS == guard.OPERATION_SET
    assert q3b.Q3F_BLOCKER_TOKENS == guard.BLOCKER_TOKENS
    assert q3b.Q3F_BLOCKER_TOKENS == (
        "mlx", "llama", "ollama", "vllm", "gemma", "qwen",
        "q3c", "q3d", "ironmule", "huggingface",
    )


def test_q3f_static_scan_starts_at_actual_child_surface():
    guard = _load("test_q3f_guard_surface", "ironmule/q3f_child_guard.py")
    visited = guard.assert_source_surface(ROOT / "ironmule" / "ab.py")
    assert ("ironmule.model_identity", "resolve_model_source") in visited
    assert ("ironmule.model_identity", "scan_local_cache") in visited


@pytest.mark.parametrize("injection", [
    "\n    subprocess.Popen(['/bin/true'])\n",
    "\n    import forbidden_runtime\n",
])
def test_q3f_recursive_scan_rejects_indirect_forbidden_child_path(tmp_path, injection):
    guard = _load("test_q3f_guard_recursive", "ironmule/q3f_child_guard.py")
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
    guard = _load("test_q3f_guard_identity_recursive", "ironmule/q3f_child_guard.py")
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
        "from ironmule import q3f_child_guard\n"
        "def _after():\n"
        "    subprocess.run(['/bin/true'], check=True)\n"
        "atexit.register(_after)\n"
        "def _child(spec):\n"
        "    return {'guard_active': q3f_child_guard.is_installed(), 'spec': spec}\n"
    )
    spec_json = json.dumps({"model": "never-imported"})
    guard_path = ROOT / "ironmule" / "q3f_child_guard.py"
    ab_path = ROOT / "ironmule" / "ab.py"
    completed = subprocess.run(
        [sys.executable, "-c", bootstrap, spec_json, str(guard_path), str(ab_path)],
        cwd=tmp_path, env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(tmp_path)},
        capture_output=True, text=True, check=False, timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    marker = next(line[2:] for line in completed.stdout.splitlines() if line.startswith("@@"))
    assert json.loads(marker) == {"guard_active": True, "spec": {"model": "never-imported"}}


@pytest.mark.parametrize("operation", sorted(q3b.Q3F_GUARD_OPERATIONS))
def test_q3f_guard_blocks_and_records_every_process_operation_in_isolated_child(operation):
    module_path = ROOT / "ironmule" / "q3f_child_guard.py"
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
    module_path = ROOT / "ironmule" / "q3f_child_guard.py"
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
def test_q3f_native_kqueue_monitor_detects_libc_fork():
    module_path = ROOT / "ironmule" / "q3f_child_guard.py"
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


def test_q3f_stable_external_process_is_attributed_unrelated():
    guard_proof, child_ledger = _guard_proof()
    unresolved, unrelated, reasons = q3b._classify_unrelated_new_processes(
        _fixtures(), _identity(),
        [{"pid": 1, "start": "root", "uid": 0, "sid": 1, "pgid": 1},
         {"pid": 2, "start": "worker", "uid": 501, "sid": 2, "pgid": 2}],
        [2], competing=None, guard_proof=guard_proof, child_ledger=child_ledger,
    )
    assert unresolved == []
    assert len(unrelated) == 1 and unrelated[0]["pid"] == 3
    assert reasons == []


@pytest.mark.parametrize("snapshots", [
    _fixtures("/usr/bin/python3 gemma-worker"),
    _fixtures(second_args="/bin/sleep changed"),
])
def test_q3f_unrelated_attribution_fails_closed_for_model_or_changed_row(snapshots):
    guard_proof, child_ledger = _guard_proof()
    unresolved, unrelated, reasons = q3b._classify_unrelated_new_processes(
        snapshots, _identity(),
        [{"pid": 1, "start": "root", "uid": 0, "sid": 1, "pgid": 1},
         {"pid": 2, "start": "worker", "uid": 501, "sid": 2, "pgid": 2}],
        [2], competing=None, guard_proof=guard_proof, child_ledger=child_ledger,
    )
    assert not unrelated and reasons


@pytest.mark.parametrize("mutation", [
    lambda row, identity, known: row.update(pgid=identity["pgid"]),
    lambda row, identity, known: row.update(sid=identity["sid"]),
    lambda row, identity, known: (known.append(row["pid"])),
    lambda row, identity, known: identity["worker_ancestor_pids"].append(row["pid"]),
    lambda row, identity, known: row.update(stat="Z"),
])
def test_q3f_attribution_rejects_group_session_descendant_ancestor_and_zombie(mutation):
    snapshots = _fixtures()
    identity = _identity()
    known = [2]
    guard_proof, child_ledger = _guard_proof()
    for snapshot in snapshots:
        row = next(item for item in snapshot["records"] if item["pid"] == 3)
        mutation(row, identity, known)
    unresolved, unrelated, reasons = q3b._classify_unrelated_new_processes(
        snapshots, identity,
        [{"pid": 1, "start": "root", "uid": 0, "sid": 1, "pgid": 1},
         {"pid": 2, "start": "worker", "uid": 501, "sid": 2, "pgid": 2}],
        known, competing=None, guard_proof=guard_proof, child_ledger=child_ledger,
    )
    assert not unrelated and reasons


def test_q3f_attribution_rejects_foreign_uid_and_malformed_comm():
    guard_proof, child_ledger = _guard_proof()
    snapshots = _fixtures()
    for snapshot in snapshots:
        next(item for item in snapshot["records"] if item["pid"] == 3)["uid"] = 42
    unresolved, unrelated, _ = q3b._classify_unrelated_new_processes(
        snapshots, _identity(),
        [{"pid": 1, "start": "root", "uid": 0, "sid": 1, "pgid": 1},
         {"pid": 2, "start": "worker", "uid": 501, "sid": 2, "pgid": 2}],
        [2], competing=None, guard_proof=guard_proof, child_ledger=child_ledger,
    )
    assert unresolved == [] and unrelated == []
    snapshots = _fixtures()
    snapshots[1]["comm"]["records"].append({"pid": 3, "comm": "other"})
    unresolved, unrelated, reasons = q3b._classify_unrelated_new_processes(
        snapshots, _identity(),
        [{"pid": 1, "start": "root", "uid": 0, "sid": 1, "pgid": 1},
         {"pid": 2, "start": "worker", "uid": 501, "sid": 2, "pgid": 2}],
        [2], competing=None, guard_proof=guard_proof, child_ledger=child_ledger,
    )
    assert not unrelated and reasons


@pytest.mark.integration
def test_q3f_real_cleanup_keeps_external_process_alive():
    worker = None
    unrelated = None
    try:
        baseline = q3b._capture_process_baseline()
        assert baseline["valid"] is True
        worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"],
                                  start_new_session=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True)
        identity = q3b._capture_worker_identity(worker, baseline=baseline)
        unrelated = subprocess.Popen(["/bin/sleep", "20"], start_new_session=True)
        worker_pid = identity["worker_pid"]
        guard_proof = [{"pid": worker_pid, "guard": {"version": "ironmule.q3f_child_guard.v1",
                                                      "installed": True, "events": []}}]
        child_ledger = [{"pid": worker_pid, "ppid": identity["parent_pid"],
                         "pgid": identity["pgid"], "sid": identity["sid"], "uid": identity["uid"],
                         "start": identity["known_process_starts"][str(worker_pid)],
                         "callback_monotonic": 0.5,
                         "guard_version": "ironmule.q3f_child_guard.v1", "guard_event_count": 0}]
        evidence = q3b._cleanup_worker_evidence(
            worker, identity, global_inventory=True,
            guard_proof=guard_proof, child_ledger=child_ledger)
        assert evidence["verification"]["group_gone"] is True
        assert evidence["verification"]["new_processes"] == [[], []]
        assert any(item["pid"] == unrelated.pid
                   for item in evidence["verification"]["unrelated_new_processes"])
        assert unrelated.poll() is None
    finally:
        if unrelated is not None and unrelated.poll() is None:
            unrelated.terminate()
            unrelated.wait(timeout=5)
        if worker is not None and worker.poll() is None:
            worker.kill()
            worker.wait(timeout=5)
