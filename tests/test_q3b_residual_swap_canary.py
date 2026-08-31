import importlib.util
import json
import os
import signal
import threading
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("q3b_residual_swap_canary", ROOT / "research/q3b_residual_swap_canary.py")
assert SPEC.loader is not None
q3b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q3b)


def _environment(**overrides):
    value = {
        "swap_used_bytes": 1000,
        "memory_free_percent": 50,
        "power_source": "AC",
        "low_power_mode": False,
        "thermal_state": "nominal",
        "loadavg": {"passed": True},
        "competing_model_process": None,
    }
    value.update(overrides)
    return value


def test_claude_exception_is_exact_bundle_only():
    desktop = f"123 100 8.0 {q3b.CLAUDE_DESKTOP_EXECUTABLE} --type=renderer\n"
    assert q3b.competing_model_process(lambda _command: desktop) is None
    quoted = f'123 100 8.0 "{q3b.CLAUDE_DESKTOP_EXECUTABLE}" --type=renderer\n'
    assert q3b.competing_model_process(lambda _command: quoted) is None
    assert q3b.competing_model_process(lambda _command: "123 100 0.0 /usr/local/bin/claude --serve\n") == "unverified Claude process activity detected"
    assert q3b.competing_model_process(lambda _command: "123 100 0.0 /Applications/Other.app/Claude\n") == "unverified Claude process activity detected"
    assert q3b.competing_model_process(lambda _command: f"123 100 0.0 {q3b.CLAUDE_DESKTOP_EXECUTABLE}X\n") == "unverified Claude process activity detected"


def test_competing_inference_process_is_blocked_and_malformed_fails_closed():
    assert q3b.competing_model_process(lambda _command: "123 10 0.0 llama-server --model local\n") == "competing model activity detected"
    assert q3b.competing_model_process(lambda _command: "broken") == "process inventory malformed"


def test_load_gate_uses_canary_thresholds():
    assert q3b.loadavg_gate(iter([7.9, 8.0, 7.0]).__next__, sleeper=lambda _seconds: None)["passed"]
    assert not q3b.loadavg_gate(iter([7.9, 8.1, 7.0]).__next__, sleeper=lambda _seconds: None)["passed"]
    assert not q3b.loadavg_gate(iter([7.0, 9.1, 7.0]).__next__, sleeper=lambda _seconds: None)["passed"]


def test_stage_gate_enforces_residual_swap_and_memory_thresholds():
    assert q3b._stage_gate(_environment(), 1000, peak=500, rss=500, installed=1000, max_swap_used_bytes=1000)["passed"]
    assert q3b._stage_gate(_environment(memory_free_percent=19), 1000, peak=500, rss=500, installed=1000, max_swap_used_bytes=1000)["passed"] is False
    assert q3b._stage_gate(_environment(), 1000, peak=500, rss=500, installed=1000, max_swap_used_bytes=1000 + q3b.SWAP_DELTA_LIMIT_BYTES + 1)["passed"] is False
    assert q3b._stage_gate(_environment(), 1000, peak=601, rss=500, installed=1000, max_swap_used_bytes=1000)["passed"] is False


def test_stage_gate_requires_known_pressure_swap_and_rss():
    assert not q3b._stage_gate(_environment(memory_free_percent=None), 1000, peak=500, rss=500, installed=1000, max_swap_used_bytes=1000)["passed"]
    assert not q3b._stage_gate(_environment(swap_used_bytes=None), 1000, peak=500, rss=500, installed=1000, max_swap_used_bytes=None)["passed"]
    assert not q3b._stage_gate(_environment(swap_used_bytes=None), 1000, peak=500, rss=500, installed=1000, max_swap_used_bytes=1000)["passed"]
    assert not q3b._stage_gate(_environment(), 1000, peak=None, rss=500, installed=1000, max_swap_used_bytes=1000)["passed"]


def test_stage_gate_checks_prior_child_reap():
    assert q3b.competing_model_process(lambda _command: "4321 10 0.0 python child\n", absent_pids=(4321,)) == "prior model child was not reaped"


def test_preregistration_sha_matches_before_execution():
    fields = q3b.PREREGISTRATION_SHA.read_text().split()
    assert q3b._preregistration_matches()
    assert fields[1] == q3b.PREREGISTRATION.name


def test_dry_run_does_not_create_output(capsys, tmp_path):
    output = tmp_path / "q3b.json"
    assert q3b.main(["--output", str(output)]) == 0
    assert not output.exists()
    plan = json.loads(capsys.readouterr().out)
    assert plan["experiment"] == q3b.EXPERIMENT_ID
    assert plan["plan"]["performance_valid"] is False
    assert plan["plan"]["promotion_allowed"] is False
    assert plan["plan"]["swap_sample_limit"] == q3b.MAX_SWAP_SAMPLES
    assert plan["plan"]["swap_sample_max_gap_seconds"] == q3b.MAX_SWAP_SAMPLE_GAP_SECONDS
    assert plan["estimated_wall_seconds"] <= q3b.PILOT_DEADLINE_SECONDS


def test_existing_output_is_rejected_without_preflight(monkeypatch, tmp_path, capsys):
    output = tmp_path / "existing.json"
    output.write_text("existing")
    monkeypatch.setattr(q3b, "preflight", lambda **_kwargs: pytest.fail("preflight must not run"))
    assert q3b.main(["--execute", "--output", str(output)]) == 2
    assert "output path" in capsys.readouterr().err


def _stage_result(stage, *, tokens=None):
    tokens = tokens or [1]
    physical = [0] + tokens
    arm = {
        "total_ns": [1.0, 2.0, 3.0], "prefill_ns": [1.0, 2.0, 3.0], "decode_ns": [1.0, 2.0, 3.0],
        "logical_tokens": tokens, "logical_tokens_per_repeat": [tokens] * 3,
        "physical_tokens_per_repeat": [physical] * 3,
        "token_counts": [{"logical": len(tokens), "physical": len(physical)}] * 3,
        "stop_reasons": ["length"] * 3, "capacities": [4] * 3, "deterministic": True,
        "decode_steps": len(physical) - 1, "prompt_tokens": 1, "mlx_peak_bytes": 100,
    }
    summary = {"n": 1, "median": 2.0, "min": 2.0, "max": 2.0, "p95": 2.0, "stdev": 0.0}
    return {
        "stage": stage, "arms": {stage: q3b.ARMS[stage]}, "processes": 1, "repeats": 3, "warmup": 1,
        "raw": [{"pid": 10 if stage == "baseline" else 11, "arms": {stage: arm}, "order": [stage], "mlx_peak_bytes": 100}],
        "per_arm": {stage: {"total_ns": summary, "prefill_ns": summary, "decode_ns": summary}},
        "token_identity": True, "token_count_identity": True, "stop_reason_identity": True,
        "deterministic": True, "reference_tokens": tokens, "ratios": {},
        "binding": {"model_id": q3b.MODEL_ID, "model_revision": q3b.EXPECTED_REVISION,
                     "model_manifest_sha256": "a" * 64, "runtime_code_sha256": "b" * 64},
        "child_rss_peak_bytes": 100, "swap_samples": [1000, 1000],
        "swap_sample_times": [1.0, 1.25], "swap_sample_offsets": [0.0, 0.25],
        "sampler_errors": [], "max_swap_used_bytes": 1000,
    }


def _preflight_for_test():
    return {
        "passed": True, "identity": {"model_id": q3b.MODEL_ID, "model_revision": q3b.EXPECTED_REVISION, "model_manifest_sha256": "a" * 64},
        "installed_memory_bytes": 1000, "environment": {"swap_used_bytes": 1000},
    }


def test_two_stage_run_stops_before_candidate_after_baseline_gate_failure(monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(q3b, "preflight", lambda **_kwargs: _preflight_for_test())
    monkeypatch.setattr(q3b, "_start_stage", lambda stage, *_args: (started.append(stage) or (_stage_result(stage), [])))
    monkeypatch.setattr(q3b, "_post_environment", lambda *_args: _environment(memory_free_percent=19))
    output = tmp_path / "failed.json"
    assert q3b.main(["--execute", "--output", str(output)]) == 2
    assert started == ["baseline"]
    assert json.loads(output.read_text())["status"] == "FAILED"


def test_cross_stage_token_identity_requires_logical_physical_counts_and_stops():
    baseline = _stage_result("baseline", tokens=[1])
    candidate = _stage_result("candidate", tokens=[2])
    assert q3b.validate_stage_result(baseline, "baseline")[0]
    assert q3b.validate_stage_result(candidate, "candidate")[0]
    assert baseline["reference_tokens"] != candidate["reference_tokens"]
    identities = q3b.cross_stage_identity(baseline, candidate)
    assert identities["logical_tokens"] is False


@pytest.mark.parametrize("field", ["capacities", "decode_steps", "prompt_tokens", "deterministic"])
def test_cross_stage_identity_rejects_metadata_mismatch(field):
    baseline = _stage_result("baseline")
    candidate = _stage_result("candidate")
    if field == "capacities":
        candidate["raw"][0]["arms"]["candidate"][field] = [5, 4, 4]
    elif field == "decode_steps":
        candidate["raw"][0]["arms"]["candidate"][field] += 1
    elif field == "prompt_tokens":
        candidate["raw"][0]["arms"]["candidate"][field] += 1
    else:
        candidate["deterministic"] = False
    assert q3b.cross_stage_identity(baseline, candidate)[field] is False


def test_swap_evidence_requires_matching_monotonic_times_and_bounded_gaps():
    result = _stage_result("baseline")
    result["swap_sample_times"] = [1.0, 1.0 + q3b.MAX_SWAP_SAMPLE_GAP_SECONDS + 0.01]
    assert not q3b.validate_stage_result(result, "baseline")[0]
    result = _stage_result("baseline")
    result["swap_sample_offsets"] = [0.0]
    assert not q3b.validate_stage_result(result, "baseline")[0]
    result = _stage_result("baseline")
    result["swap_sample_offsets"] = [0.0, q3b.MAX_SWAP_SAMPLE_GAP_SECONDS + 0.01]
    assert not q3b.validate_stage_result(result, "baseline")[0]


def test_exclusive_result_write_is_0600_and_non_overwriting(tmp_path):
    path = tmp_path / "result.json"
    q3b._write_exclusive(path, b"{}\n")
    assert path.read_bytes() == b"{}\n"
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        q3b._write_exclusive(path, b"overwrite\n")


def test_worker_cleanup_reaps_without_unconditional_second_kill(monkeypatch):
    class Process:
        pid = 4321
        def __init__(self):
            self.returncode = None
        def wait(self, timeout=None):
            self.returncode = 0
        def communicate(self, timeout=None):
            return "", ""
        def poll(self):
            return self.returncode
    signals = []
    def killpg(pid, sig):
        if sig == 0:
            raise ProcessLookupError
        signals.append((pid, sig))
    monkeypatch.setattr(q3b.os, "killpg", killpg)
    assert q3b._cleanup_worker(Process()) == []
    assert signals == [(4321, q3b.signal.SIGTERM)]


def test_live_safety_capture_is_bounded_single_shot_and_injectable():
    events = []
    kills = []
    state = {}
    samples = [1000, 1000 + q3b.SWAP_DELTA_LIMIT_BYTES + 1]
    event = q3b._capture_live_safety(
        reason="swap_delta_exceeded", samples=samples, sample_times=[1.0, 1.25],
        sample_offsets=[0.0, 0.25], sampler_errors=[], lock=threading.Lock(), state=state,
        marker_writer=events.append, kill_group=lambda pid, sig: kills.append((pid, sig)),
    )
    assert event == events[0] == state["event"]
    assert kills == [(os.getpgrp(), signal.SIGTERM)]
    assert set(event) == {"reason", "samples", "times", "offsets", "errors"}
    assert "argv" not in json.dumps(event)
    assert q3b._capture_live_safety(
        reason="second", samples=[1], sample_times=[2.0], sample_offsets=[1.0],
        sampler_errors=["ignored"], lock=threading.Lock(), state=state,
        marker_writer=events.append, kill_group=lambda pid, sig: kills.append((pid, sig)),
    ) is event
    assert len(events) == 1 and len(kills) == 1


def test_worker_start_highwater_refuses_before_model_child(monkeypatch, capsys):
    expected = {
        "identity": {"model_id": q3b.MODEL_ID, "model_revision": q3b.EXPECTED_REVISION,
                     "model_manifest_sha256": "a" * 64},
        "runtime_code_sha256": "b" * 64, "stage": "baseline", "initial_swap": 1000,
        "installed_memory": 1000,
    }
    monkeypatch.setattr(q3b, "_read_capability", lambda: expected)
    monkeypatch.setattr(q3b, "runtime_code_sha256", lambda *_args, **_kwargs: "b" * 64)
    monkeypatch.setattr(q3b, "_read_swap_sample", lambda _run: 1000 + q3b.SWAP_DELTA_LIMIT_BYTES + 1)
    monkeypatch.setenv("IRONMULE_Q3B_WORKER_DEADLINE", "9999999999")
    imported = []
    original = __import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("ironmule", "mlx", "mlx_lm")):
            imported.append(name)
            raise AssertionError(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)
    assert q3b._stage_worker(kill_group=lambda *_args: pytest.fail("pre-child must not kill")) == 2
    assert imported == []
    failure = json.loads(next(line[2:] for line in capsys.readouterr().out.splitlines() if line.startswith("@@")))
    assert "worker-start swap delta exceeded" in failure["failure"]


def test_parent_safety_marker_preserves_evidence_and_reaps(monkeypatch):
    class Process:
        pid = 4321
        returncode = 2

        def communicate(self, timeout=None):
            event = {"reason": "swap_sampler_error", "samples": [1], "times": [1.0],
                     "offsets": [0.0], "errors": ["periodic: RuntimeError"]}
            return "@SAFETY " + json.dumps(event) + "\n", "diagnostic"

    cleanup = []
    monkeypatch.setattr(q3b.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(q3b, "runtime_code_sha256", lambda: "b" * 64)
    monkeypatch.setattr(q3b, "_cleanup_worker", lambda process: cleanup.append(process.pid) or [])
    result, _ = q3b._start_stage("baseline", {"model_id": q3b.MODEL_ID,
                                                "model_revision": q3b.EXPECTED_REVISION,
                                                "model_manifest_sha256": "a" * 64},
                                  0, 1000, q3b.time.monotonic() + 60)
    assert result["failure"] == "stage worker live safety abort: swap_sampler_error"
    assert result["safety_event"]["errors"] == ["periodic: RuntimeError"]
    assert result["partial_evidence"]["samples"] == [1]
    assert result["group_gone"] is True
    assert cleanup == [4321]


def test_final_swap_delta_is_terminal_safety_event_with_injected_kill():
    events = []
    signals = []
    state = {}
    event = q3b._finalize_stage_safety(
        initial_swap=1000, samples=[1000, 1000 + q3b.SWAP_DELTA_LIMIT_BYTES + 1],
        sample_times=[1.0, 1.25], sample_offsets=[0.0, 0.25], sampler_errors=[],
        lock=threading.Lock(), state=state, final_error=None, marker_writer=events.append,
        kill_group=lambda pid, sig: signals.append((pid, sig)),
    )
    assert event["reason"] == "final_swap_delta_exceeded"
    assert events == [event]
    assert signals == [(os.getpgrp(), signal.SIGTERM)]
    assert state["event"] is event


def test_final_swap_read_error_is_terminal_safety_event():
    events = []
    signals = []
    event = q3b._finalize_stage_safety(
        initial_swap=1000, samples=[1000], sample_times=[1.0], sample_offsets=[0.0],
        sampler_errors=["worker-final: RuntimeError: sampler failed"],
        lock=threading.Lock(), state={}, final_error=RuntimeError("sampler failed"),
        marker_writer=events.append, kill_group=lambda pid, sig: signals.append(sig),
    )
    assert event["reason"] == "final_swap_sampler_error"
    assert event["errors"]
    assert signals == [signal.SIGTERM]


def test_safety_term_failure_escalates_to_kill_and_succeeds():
    signals = []

    def kill_group(_pid, sig):
        signals.append(sig)
        if sig == signal.SIGTERM:
            raise OSError("TERM denied")

    event = q3b._capture_live_safety(
        reason="final_swap_delta_exceeded", samples=[1], sample_times=[1.0],
        sample_offsets=[0.0], sampler_errors=[], lock=threading.Lock(), state={},
        marker_writer=lambda _event: None, kill_group=kill_group,
    )
    assert event["reason"] == "final_swap_delta_exceeded"
    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_safety_term_and_kill_failure_is_loud():
    signals = []

    def kill_group(_pid, sig):
        signals.append(sig)
        raise OSError(str(sig))

    with pytest.raises(q3b.CanaryRefused, match="group kill failed"):
        q3b._capture_live_safety(
            reason="final_swap_delta_exceeded", samples=[1], sample_times=[1.0],
            sample_offsets=[0.0], sampler_errors=[], lock=threading.Lock(), state={},
            marker_writer=lambda _event: None, kill_group=kill_group,
        )
    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_top_level_import_surface_has_no_model_imports():
    import ast
    tree = ast.parse((ROOT / "research/q3b_residual_swap_canary.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] not in {"ironmule", "mlx", "mlx_lm"} for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in {"ironmule", "mlx", "mlx_lm"}


def test_direct_stage_worker_refuses_before_ironmule_import(monkeypatch):
    imported = []
    original = __import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("ironmule", "mlx", "mlx_lm")):
            imported.append(name)
            raise AssertionError(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)
    assert q3b._stage_worker() == 2
    assert imported == []
