import ast
import importlib.util
import json
import os
import signal
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


q3b = _load("test_q3d_q3b", "research/q3b_residual_swap_canary.py")
q3d = _load("test_q3d_gate", "research/q3d_stability_gate.py")


ROOT_ROW = "1 0 1 1 0 S 00:00:01 /sbin/launchd\n"
WORKER_ROW = "42 1 42 42 501 S 00:00:02 /usr/bin/python worker\n"
CHILD_ROW = "43 42 42 42 501 S 00:00:03 /usr/bin/python child\n"


class _Process:
    pid = 42

    def __init__(self, *, reaped=True):
        self.returncode = 0 if reaped else None
        self.reaped = reaped
        self.waits = 0

    def wait(self, timeout=None):
        self.waits += 1
        if not self.reaped:
            raise subprocess.TimeoutExpired("worker", timeout)

    def poll(self):
        return self.returncode

    def communicate(self, timeout=None):
        return "", ""


def _sequence(*outputs):
    values = iter(outputs)

    def run(_command):
        return next(values)

    return run


def _capture_identity(monkeypatch, post_outputs, *, reaped=True, kill=None):
    process = _Process(reaped=reaped)
    monkeypatch.setattr(q3b.os, "killpg", kill or (lambda *_args: None))
    baseline_runner = _sequence(ROOT_ROW + WORKER_ROW + CHILD_ROW, ROOT_ROW + WORKER_ROW + CHILD_ROW)
    baseline = q3b._capture_process_baseline(baseline_runner)
    identity = q3b._capture_worker_identity(process,
                                             run=_sequence(ROOT_ROW + WORKER_ROW + CHILD_ROW),
                                             baseline=baseline)
    evidence = q3b._cleanup_worker_evidence(
        process, identity, run=_sequence(ROOT_ROW + WORKER_ROW + CHILD_ROW, *post_outputs))
    return identity, evidence


def test_cleanup_identity_captures_parent_and_descendant_pids(monkeypatch):
    identity, evidence = _capture_identity(monkeypatch, [ROOT_ROW, ROOT_ROW])
    assert identity["worker_pid"] == 42 and identity["parent_pid"] == 1
    assert identity["pgid"] == identity["sid"] == 42 and identity["uid"] == 501
    assert identity["known_descendant_pids"] == [42, 43]
    assert identity["known_process_starts"] == {"42": "00:00:02", "43": "00:00:03"}
    assert identity["known_process_sids"] == {"42": 42, "43": 42}
    assert identity["spawn_baseline"]["valid"] is True
    assert evidence["worker_reaped"] is True
    assert evidence["verification"]["group_gone"] is True
    assert evidence["verification"]["snapshot_count"] == 2
    assert evidence["verification"]["independent"] is True


def test_cleanup_permission_error_is_retained_but_resolved_when_group_empty(monkeypatch):
    def deny(_pid, sig):
        if sig == signal.SIGTERM:
            raise PermissionError("not owner")

    _, evidence = _capture_identity(monkeypatch, [ROOT_ROW, ROOT_ROW], kill=deny)
    assert evidence["verification"]["group_gone"] is True
    assert evidence["unresolved_errors"] == []
    assert evidence["signal_attempts"][0]["status"] == "permission_error"
    assert "SIGTERM:permission_error" in evidence["resolved_errors"]


def test_cleanup_rejects_escaped_descendant_and_foreign_uid_member(monkeypatch):
    escaped = "1 0 1 1 0 S 00:00:01 /sbin/launchd\n43 1 42 42 999 S 00:00:03 /usr/bin/child\n"
    _, evidence = _capture_identity(monkeypatch, [escaped, escaped, escaped])
    assert evidence["verification"]["group_gone"] is False
    assert any("descendant" in item or "group" in item for item in evidence["unresolved_errors"])
    assert evidence["verification"]["members"][0][0]["uid"] == 999


def test_cleanup_rejects_zombie_and_unreaped_leader(monkeypatch):
    zombie = "1 0 1 1 0 S 00:00:01 /sbin/launchd\n42 1 42 42 501 Z 00:00:02 /usr/bin/worker\n"
    _, evidence = _capture_identity(monkeypatch, [zombie, zombie, zombie], reaped=True)
    assert evidence["verification"]["group_gone"] is False
    assert "zombie worker or descendant remains" in evidence["unresolved_errors"]
    _, evidence = _capture_identity(monkeypatch, [ROOT_ROW, ROOT_ROW], reaped=False)
    assert evidence["verification"]["group_gone"] is False
    assert "worker leader was not reaped" in evidence["unresolved_errors"]


@pytest.mark.parametrize("output, needle", [
    ("1 0 1 1 0 S", "row malformed"),
    ("1 0 1 1 0 S 00:00:01 /sbin/launchd\n42 77 42 42 501 S 00:00:02 /worker\n", "parent link missing"),
    ("1 2 1 1 0 S 00:00:01 /one\n2 1 1 1 0 S 00:00:02 /two\n", "ancestry cycle"),
])
def test_cleanup_ps_parser_fails_closed_for_unknown_tree(output, needle):
    parsed = q3b._parse_cleanup_ps_snapshot(output)
    assert parsed["valid"] is False
    assert needle in parsed["error"]


def test_cleanup_race_with_first_snapshot_member_is_not_called_gone(monkeypatch):
    escaped = "1 0 1 1 0 S 00:00:01 /sbin/launchd\n43 42 42 42 501 S 00:00:03 /usr/bin/child\n"
    _, evidence = _capture_identity(monkeypatch, [escaped, ROOT_ROW])
    assert evidence["verification"]["group_gone"] is False


def test_cleanup_escalates_term_to_kill_when_reaped_leader_has_member(monkeypatch):
    member = ROOT_ROW + "43 1 42 42 501 S 00:00:03 /usr/bin/child\n"
    signals = []
    def killpg(pid, sig):
        signals.append((pid, sig))
    _, evidence = _capture_identity(monkeypatch, [member, ROOT_ROW, ROOT_ROW], kill=killpg)
    assert signals == [(42, signal.SIGTERM), (42, signal.SIGKILL)]
    assert evidence["verification"]["group_gone"] is True
    assert [item["signal"] for item in evidence["signal_attempts"]] == ["SIGTERM", "SIGKILL"]


def test_cleanup_kills_known_reparented_descendant_and_rejects_start_reuse(monkeypatch):
    escaped = ROOT_ROW + "43 1 999 42 501 S 00:00:03 /usr/bin/child\n"
    killed = []
    monkeypatch.setattr(q3b.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    _, evidence = _capture_identity(monkeypatch, [escaped, ROOT_ROW, ROOT_ROW])
    assert killed == [(43, signal.SIGKILL)]
    assert evidence["verification"]["group_gone"] is True
    reused = ROOT_ROW + "43 1 999 42 501 S 99:99:99 /usr/bin/child\n"
    killed.clear()
    _, evidence = _capture_identity(monkeypatch, [reused, ROOT_ROW])
    assert killed == []
    assert evidence["verification"]["group_gone"] is False
    assert "known PID start identity changed or was reused" in evidence["unresolved_errors"]


def test_cleanup_rejects_reused_pid_with_changed_sid_or_group(monkeypatch):
    reused = ROOT_ROW + "43 1 999 999 501 S 00:00:03 /usr/bin/child\n"
    killed = []
    monkeypatch.setattr(q3b.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    _, evidence = _capture_identity(monkeypatch, [reused, reused, reused])
    assert killed == []
    assert evidence["verification"]["group_gone"] is False
    assert "new same-UID process appeared after worker spawn baseline" in evidence["unresolved_errors"]


def test_cleanup_stat_unknown_fails_closed():
    parsed = q3b._parse_cleanup_ps_snapshot("1 0 1 0 ? 00:00:01 /sbin/launchd\n")
    assert parsed["valid"] is False


def test_worker_identity_rejects_root_and_foreign_uid(monkeypatch):
    process = _Process()
    full = ROOT_ROW + WORKER_ROW + CHILD_ROW
    baseline = q3b._capture_process_baseline(_sequence(full))
    monkeypatch.setattr(q3b.os, "getuid", lambda: 0)
    with pytest.raises(q3b.CanaryRefused, match="root execution"):
        q3b._capture_worker_identity(process, run=_sequence(full), baseline=baseline)
    monkeypatch.setattr(q3b.os, "getuid", lambda: 501)
    foreign = ROOT_ROW + "42 1 42 42 999 S 00:00:02 /usr/bin/python worker\n"
    foreign_baseline = q3b._capture_process_baseline(_sequence(foreign))
    with pytest.raises(q3b.CanaryRefused, match="UID"):
        q3b._capture_worker_identity(process, run=_sequence(foreign), baseline=foreign_baseline)


def test_cleanup_unknown_first_snapshot_still_attempts_safe_known_orphan_kill(monkeypatch):
    malformed = "not-a-ps-row\n"
    orphan = ROOT_ROW + "43 1 999 42 501 S 00:00:03 /usr/bin/child\n"
    killed = []
    monkeypatch.setattr(q3b.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    _, evidence = _capture_identity(monkeypatch, [malformed, orphan, ROOT_ROW, ROOT_ROW])
    assert killed == [(43, signal.SIGKILL)]
    assert evidence["verification"]["group_gone"] is False
    assert evidence["unresolved_errors"]


def _fake_gate(*, values=None, fail_at=None):
    now = [0.0]
    calls = []
    values = iter(values if values is not None else [100] * q3d.SAMPLE_COUNT)

    def clock():
        return now[0]

    def sleeper(duration):
        calls.append(duration)
        now[0] += duration

    def reader():
        index = len(calls)
        if fail_at is not None and index >= fail_at:
            raise RuntimeError("sample failed")
        return next(values)

    post = {"power_source": "AC", "low_power_mode": False, "thermal_state": "nominal",
            "swap_used_bytes": 100, "memory_free_percent": 50}
    pre = {"passed": True, "checks": {key: True for key in q3d._preflight_check_names()}, "environment": post,
           "model_cache_identity": {"model_id": q3d.MODEL_ID, "model_revision": q3d.EXPECTED_REVISION,
                                     "model_manifest_sha256": "a" * 64},
           "git": {"clean": True, "commit": "b" * 40}, "git_commit": "b" * 40,
           "untracked": {"passed": True}, "loadavg": {"samples": [1.0, 1.0, 1.0], "max": 1.0, "spread": 0.0, "passed": True},
           "runtime_code_sha256": "c" * 64, "preregistration_matches": True,
           "installed_memory_bytes": 1000}
    evidence = q3d.run_stability_gate(
        pre, q3b_module=object(), clock=clock, sleeper=sleeper,
        swap_reader=reader, environment_reader=lambda: post, process_checker=lambda: None,
        load_reader=lambda: {"samples": [1.0, 1.0, 1.0], "max": 1.0, "spread": 0.0, "passed": True})
    return evidence, calls


def test_stability_gate_uses_exact_t0_plus_one_second_schedule():
    evidence, calls = _fake_gate()
    assert evidence["passed"] is True
    assert len(evidence["samples"]) == 61
    assert len(evidence["commands"]) == 61
    assert evidence["sample_offsets"] == pytest.approx(list(range(61)))
    assert calls == [1.0] * 60
    assert evidence["elapsed_seconds"] == pytest.approx(60.0)
    assert evidence["highwater_delta_bytes"] == 0
    assert q3d.validate_stability_evidence(evidence) == (True, "ok")


def test_stability_gate_rejects_nonzero_swap_and_retains_partial_evidence():
    evidence, _ = _fake_gate(values=[100] + [101] * 60)
    assert evidence["passed"] is False
    assert evidence["highwater_delta_bytes"] == 1
    assert "swap high-water increase was not exactly zero" in evidence["errors"]


def test_stability_gate_stops_on_unknown_sample_without_retry():
    evidence, _ = _fake_gate(fail_at=3)
    assert evidence["passed"] is False
    assert len(evidence["samples"]) == 3
    assert len(evidence["commands"]) == 4
    assert evidence["commands"][-1]["known"] is False


def test_stability_plan_and_preregistration_are_frozen_and_exact():
    generated = q3d.plan(ROOT)
    assert generated["sample_count"] == 61
    assert generated["scheduled_samples"] == 60
    assert generated["gate_deadline_seconds"] == 90.0
    assert generated["outer_max_seconds"] == 720.0
    assert generated["terminal_reserve_seconds"] == 30.0
    assert generated["output_cap_bytes"] == 512 * 1024
    assert q3d._preregistration_matches()


def test_stability_output_is_exclusive_and_private(tmp_path):
    path = tmp_path / "gate.json"
    q3d._write_exclusive(path, b"{}\n")
    assert path.read_bytes() == b"{}\n"
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        q3d._write_exclusive(path, b"overwrite\n")


def _main_preflight():
    return {"passed": True, "checks": {key: True for key in q3d._preflight_check_names()}, "environment": {
        "power_source": "AC", "low_power_mode": False, "thermal_state": "nominal",
        "memory_free_percent": 50, "swap_used_bytes": 100},
        "model_cache_identity": {"model_id": q3d.MODEL_ID, "model_revision": q3d.EXPECTED_REVISION,
                                  "model_manifest_sha256": "a" * 64},
        "git": {"clean": True, "commit": "b" * 40}, "git_commit": "b" * 40,
        "untracked": {"passed": True}, "loadavg": {"samples": [1.0, 1.0, 1.0],
        "max": 1.0, "spread": 0.0, "passed": True}, "runtime_code_sha256": "c" * 64,
        "preregistration_matches": True, "installed_memory_bytes": 1000}


def test_main_never_invokes_q3c_after_preflight_or_gate_failure(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(q3d, "_invoke_q3c_once", lambda *_args, **_kwargs: calls.append(True))
    monkeypatch.setattr(q3d, "preflight", lambda **_kwargs: {"passed": False, "checks": {}})
    output = tmp_path / "failed.json"
    assert q3d.main(["--execute", "--output", str(output), "--q3c-output", str(tmp_path / "q3c.json")]) == 2
    assert calls == []
    assert json.loads(output.read_text())["status"] == "FAILED"
    summary = output.with_name(output.stem + "_summary.json")
    assert json.loads(summary.read_text())["q3c"]["invoked"] is False

    monkeypatch.setattr(q3d, "preflight", lambda **_kwargs: _main_preflight())
    monkeypatch.setattr(q3d, "run_stability_gate", lambda *_args, **_kwargs: {"passed": False})
    output = tmp_path / "gate-failed.json"
    assert q3d.main(["--execute", "--output", str(output), "--q3c-output", str(tmp_path / "q3c2.json")]) == 2
    assert calls == []


def test_q3c_invocation_is_one_exact_bounded_subprocess_call(tmp_path, monkeypatch):
    calls = []

    class Completed:
        returncode = 0
        stdout = "q3c"
        stderr = ""

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        q3c_output.write_text("{}\n")
        return Completed()

    q3c_output = tmp_path / "q3c.json"
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("SECRET_TOKEN", "do-not-persist")
    result = q3d._invoke_q3c_once(ROOT, q3c_output, expected_commit="b" * 40,
                                  expected_identity={"model_id": q3d.MODEL_ID,
                                                     "model_revision": q3d.EXPECTED_REVISION,
                                                     "model_manifest_sha256": "a" * 64}, runner=runner)
    assert result["status"] == "FAILED"
    assert "schema" in result["error"]
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == [q3d.sys.executable, str(ROOT / "research/q3c_performance_replication.py"),
                       "--execute", "--output", str(q3c_output)]
    assert kwargs["timeout"] == q3d.Q3C_WRAPPER_TIMEOUT_SECONDS
    assert kwargs["capture_output"] is True and kwargs["text"] is True and kwargs["check"] is False
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert kwargs["env"]["PYTHONPATH"] == str(ROOT)
    assert kwargs["env"]["HF_HUB_CACHE"] == str(tmp_path / "cache")
    assert "SECRET_TOKEN" not in result["offline_env"]


def test_q3c_communicate_exception_routes_through_cleanup_v2(monkeypatch, tmp_path):
    calls = []
    nested_calls = []

    class Process:
        pid = 991
        returncode = None

        def communicate(self, timeout=None):
            raise OSError("pipe broke")

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    class Policy:
        _run_text = staticmethod(lambda _command: "")

        @staticmethod
        def _capture_process_baseline():
            return {"valid": True, "digest": "d" * 64, "identities": []}

        @staticmethod
        def _capture_worker_identity(process, baseline=None):
            return {"worker_pid": process.pid, "parent_pid": 1, "pgid": process.pid, "uid": 501,
                    "sid": process.pid, "known_descendant_pids": [process.pid],
                    "known_process_starts": {str(process.pid): "00:00:01"},
                    "known_process_sids": {str(process.pid): process.pid},
                    "spawn_baseline": baseline or {"valid": True, "digest": "d" * 64, "identities": []}}

        @staticmethod
        def _cleanup_ps_snapshot(_run):
            return {"monotonic": 1.0, "command_ok": True, "parse_ok": True,
                    "records": [{"pid": 1, "ppid": 0, "pgid": 1, "sid": 1, "uid": 0,
                                 "stat": "S", "start": "00:00:01", "args": "root"}], "error": None}

        @staticmethod
        def _cleanup_worker_evidence(process, identity):
            calls.append(("outer", process.pid, identity))
            return {"schema": "ironmule.cleanup.v2", "verification": {"group_gone": True}}

    monkeypatch.setattr(q3d, "_load_q3c", lambda _root: object())
    monkeypatch.setattr(q3d, "_load_q3b", lambda _root: Policy)
    monkeypatch.setattr(q3d, "_cleanup_nested_q3c_workers", lambda *args: nested_calls.append(True) or {"known": True, "errors": []})
    monkeypatch.setattr(q3d.subprocess, "Popen", lambda *args, **kwargs: Process())
    result = q3d._invoke_q3c_once(ROOT, tmp_path / "q3c.json")
    assert result["status"] == "FAILED"
    assert result["cleanup"]["schema"] == "ironmule.cleanup.v2"
    assert calls and calls[0][0] == "outer"
    assert nested_calls == [True]


def test_nested_q3c_worker_inventory_is_terminated_by_verified_group(monkeypatch):
    root_row = {"pid": 1, "ppid": 0, "pgid": 1, "sid": 1, "uid": 0, "stat": "S",
                "start": "00:00:01", "args": "/sbin/launchd"}
    nested_row = {"pid": 50, "ppid": 1, "pgid": 50, "sid": 50, "uid": 501, "stat": "S",
                  "start": "00:00:02", "args": "python q3c_performance_replication.py --phase-worker"}
    first = q3b._parse_cleanup_ps_snapshot(
        "1 0 1 1 0 S 00:00:01 /sbin/launchd\n"
        "50 1 50 50 501 S 00:00:02 python q3c_performance_replication.py --phase-worker\n")
    assert first["valid"] is True
    first_record = {"monotonic": 1.0, "command_ok": True, "parse_ok": True,
                    "records": first["records"], "error": None}
    second_record = {"monotonic": 2.0, "command_ok": True, "parse_ok": True,
                     "records": [root_row], "error": None}
    snapshots = iter([
        first_record,
        second_record,
    ])
    class Policy:
        @staticmethod
        def _cleanup_ps_snapshot(_run):
            return next(snapshots)
    killed = []
    monkeypatch.setattr(q3d.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    result = q3d._cleanup_nested_q3c_workers(
        Policy, 501, lambda _command: "",
        {"valid": True, "identities": [{"pid": 1, "start": "00:00:01", "uid": 0, "sid": 1, "pgid": 1}]})
    assert killed == [(50, signal.SIGTERM)]
    assert result["known"] is True and result["errors"] == []


def test_nested_q3c_worker_missing_sid_or_args_is_unknown():
    class Policy:
        @staticmethod
        def _cleanup_ps_snapshot(_run):
            return {"monotonic": 1.0, "command_ok": True, "parse_ok": True,
                    "records": [{"pid": 1, "ppid": 0, "pgid": 1, "uid": 0,
                                 "stat": "S", "start": "00:00:01", "args": "root"}], "error": None}
    result = q3d._cleanup_nested_q3c_workers(Policy, 501, lambda _command: "")
    assert result["known"] is False


def test_nested_preexisting_q3c_lookalike_is_not_killed(monkeypatch):
    row = {"pid": 50, "ppid": 1, "pgid": 50, "sid": 50, "uid": 501, "stat": "S",
           "start": "00:00:02", "args": "python q3c_performance_replication.py --phase-worker"}
    root = {"pid": 1, "ppid": 0, "pgid": 1, "sid": 1, "uid": 0, "stat": "S",
            "start": "00:00:01", "args": "root"}
    values = iter([{"monotonic": 1.0, "command_ok": True, "parse_ok": True,
                    "records": [root, row], "error": None},
                   {"monotonic": 2.0, "command_ok": True, "parse_ok": True,
                    "records": [root, row], "error": None}])
    class Policy:
        @staticmethod
        def _cleanup_ps_snapshot(_run):
            return next(values)
    killed = []
    monkeypatch.setattr(q3d.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    result = q3d._cleanup_nested_q3c_workers(
        Policy, 501, lambda _command: "",
        {"valid": True, "identities": [{"pid": 1, "start": "00:00:01", "uid": 0, "sid": 1, "pgid": 1},
                                          {"pid": 50, "start": "00:00:02", "uid": 501, "sid": 50, "pgid": 50}]})
    assert killed == []
    assert result["groups"] == []


def test_main_invokes_q3c_once_only_after_stability_pass(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(q3d, "preflight", lambda **_kwargs: _main_preflight())
    passing_gate, _ = _fake_gate()
    monkeypatch.setattr(q3d, "run_stability_gate", lambda *_args, **_kwargs: passing_gate)
    monkeypatch.setattr(q3d, "_invoke_q3c_once", lambda root, output, **_kwargs: calls.append((root, output)) or {"invoked": True, "status": "FAILED"})
    output = tmp_path / "gate-pass.json"
    q3c_output = tmp_path / "q3c-pass.json"
    assert q3d.main(["--execute", "--output", str(output), "--q3c-output", str(q3c_output)]) == 2
    assert calls == [(ROOT, q3c_output)]
    summary = json.loads(output.with_name(output.stem + "_summary.json").read_text())
    assert summary["invoked"] is True
    assert summary["gate_raw"]["sha256"]


def test_main_persists_failed_summary_when_q3c_invocation_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(q3d, "preflight", lambda **_kwargs: _main_preflight())
    passing_gate, _ = _fake_gate()
    monkeypatch.setattr(q3d, "run_stability_gate", lambda *_args, **_kwargs: passing_gate)
    def explode(*_args, **_kwargs):
        raise RuntimeError("runner unavailable")
    monkeypatch.setattr(q3d, "_invoke_q3c_once", explode)
    output = tmp_path / "gate.json"
    assert q3d.main(["--execute", "--output", str(output)]) == 2
    summary = json.loads((tmp_path / "gate_summary.json").read_text())
    assert summary["status"] == "Q3C_FAILED"
    assert summary["q3c"]["invoked"] is True


def test_stability_module_parent_has_no_forbidden_imports():
    tree = ast.parse((ROOT / "research/q3d_stability_gate.py").read_text())
    forbidden = {"ironmule", "mlx", "mlx_lm"}
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module.split(".")[0])
    assert not forbidden.intersection(imported)
