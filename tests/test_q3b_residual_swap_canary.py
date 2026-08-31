import importlib.util
import json
import os
import signal
import sys
import threading
import types
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


def _inventories(args, comm):
    # Keep the older focused fixtures concise while supplying the strict
    # parent-aware ps schema required by the gate.  The test process is the
    # synthetic root; fixture PIDs remain independent siblings.
    normalized = []
    for line in args.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4:
            normalized.append(f"{parts[0]} 0 {parts[1]} {parts[2]} {parts[3]}")
        else:
            normalized.append(line)
    self_pid = os.getpid()
    if not any(line.split(None, 1)[0:1] == [str(self_pid)] for line in normalized):
        normalized.insert(0, f"{self_pid} 0 0 0.0 /usr/bin/python test")
    comm_lines = comm.splitlines()
    if not any(line.split(None, 1)[0:1] == [str(self_pid)] for line in comm_lines):
        comm_lines.insert(0, f"{self_pid} python")
    args = "\n".join(normalized) + "\n"
    comm = "\n".join(comm_lines) + "\n"

    def run(command):
        return comm if command[-1].endswith("comm=") else args
    return run


def test_claude_exception_requires_verified_bundle_and_exact_contents_boundary(monkeypatch):
    monkeypatch.setattr(q3b, "_trusted_claude_bundle", lambda: True)
    args = f"123 100 8.0 {q3b.CLAUDE_DESKTOP_EXECUTABLE} --type=renderer\n"
    comm = f"123 {q3b.CLAUDE_DESKTOP_EXECUTABLE}\n"
    assert q3b.competing_model_process(_inventories(args, comm)) is None
    helper_comm = "123 /Applications/Claude.app/Contents/Frameworks/Claude Helper.app/Contents/MacOS/Claude Helper\n"
    helper_args = "123 100 8.0 '/Applications/Claude.app/Contents/Frameworks/Claude Helper.app/Contents/MacOS/Claude Helper' --type=renderer\n"
    assert q3b.competing_model_process(_inventories(helper_args, helper_comm)) is None
    crashpad_comm = "123 /Applications/Claude.app/Contents/Frameworks/Claude Helper.app/Contents/Resources/crashpad_handler\n"
    crashpad_args = "123 100 8.0 /Applications/Claude.app/Contents/Frameworks/Claude Helper.app/Contents/Resources/crashpad_handler\n"
    assert q3b.competing_model_process(_inventories(crashpad_args, crashpad_comm)) is None
    outside = "123 100 0.0 /Applications/Claude.app.evil/Contents/MacOS/Claude\n"
    outside_comm = "123 /Applications/Claude.app.evil/Contents/MacOS/Claude\n"
    assert q3b.competing_model_process(_inventories(outside, outside_comm)) == "unverified Claude process activity detected"


def test_claude_bundle_trust_fails_closed_on_verify_or_metadata_failure(monkeypatch):
    class Completed:
        def __init__(self, returncode=0, stdout="", stderr=""):
            self.returncode, self.stdout, self.stderr = returncode, stdout, stderr

    metadata = "\n".join([
        "Identifier=com.anthropic.claudefordesktop",
        "TeamIdentifier=Q6L2SF6YDW",
        "Authority=Developer ID Application: Anthropic PBC (Q6L2SF6YDW)",
        "Authority=Developer ID Certification Authority",
    ])
    calls = []
    kwargs_by_call = []
    def runner(command, **kwargs):
        calls.append(command)
        kwargs_by_call.append(kwargs)
        return Completed(stderr="ok" if "--verify" in command else metadata)
    assert q3b._trusted_claude_bundle(runner=runner)
    assert calls[0] == [q3b.CLAUDE_CODESIGN, "--verify", "--deep", "--strict", q3b.CLAUDE_DESKTOP_BUNDLE]
    assert calls[1] == [q3b.CLAUDE_CODESIGN, "-dv", "--verbose=4", q3b.CLAUDE_DESKTOP_BUNDLE]
    assert [call["timeout"] for call in kwargs_by_call] == [q3b.CLAUDE_CODESIGN_TIMEOUT_SECONDS] * 2

    monkeypatch.setattr(q3b.subprocess, "run", lambda *_args, **_kwargs: Completed(returncode=1))
    assert not q3b._trusted_claude_bundle()
    bad_metadata = metadata.replace("Q6L2SF6YDW", "WRONGTEAM", 1)
    calls.clear()
    def bad_runner(command, **_kwargs):
        calls.append(command)
        return Completed(stderr="ok" if "--verify" in command else bad_metadata)
    assert not q3b._trusted_claude_bundle(runner=bad_runner)

    class TimeoutRunner:
        def __call__(self, *_args, **_kwargs):
            raise q3b.subprocess.TimeoutExpired(cmd=q3b.CLAUDE_CODESIGN, timeout=5.0)

    assert not q3b._trusted_claude_bundle(runner=TimeoutRunner())

    class ErrorRunner:
        def __call__(self, *_args, **_kwargs):
            raise OSError("codesign unavailable")

    assert not q3b._trusted_claude_bundle(runner=ErrorRunner())


def test_untrusted_and_generic_claude_processes_are_blocked(monkeypatch):
    monkeypatch.setattr(q3b, "_trusted_claude_bundle", lambda: False)
    args = f"123 100 0.0 {q3b.CLAUDE_DESKTOP_EXECUTABLE} --type=renderer\n"
    comm = f"123 {q3b.CLAUDE_DESKTOP_EXECUTABLE}\n"
    assert q3b.competing_model_process(_inventories(args, comm)) == "unverified Claude process activity detected"
    generic_args = "123 100 0.0 /usr/local/bin/claude --serve\n"
    generic_comm = "123 /usr/local/bin/claude\n"
    assert q3b.competing_model_process(_inventories(generic_args, generic_comm)) == "unverified Claude process activity detected"
    other_args = "123 100 0.0 /Applications/Other.app/Claude\n"
    other_comm = "123 /Applications/Other.app/Claude\n"
    assert q3b.competing_model_process(_inventories(other_args, other_comm)) == "unverified Claude process activity detected"
    prefix_args = f"123 100 0.0 {q3b.CLAUDE_DESKTOP_EXECUTABLE}X\n"
    prefix_comm = f"123 {q3b.CLAUDE_DESKTOP_EXECUTABLE}X\n"
    assert q3b.competing_model_process(_inventories(prefix_args, prefix_comm)) == "unverified Claude process activity detected"


def test_competing_inference_process_is_blocked_and_malformed_fails_closed():
    output = "123 10 0.0 llama-server --model local\n"
    assert q3b.competing_model_process(_inventories(output, "123 llama-server\n")) == "competing model activity detected"
    assert q3b.competing_model_process(_inventories(output, "123")) == "process inventory malformed"


def test_process_inventory_allows_new_extra_comm_pid():
    args = "123 10 0.0 /usr/bin/python worker.py\n"
    comm = "123 python\n456 /usr/bin/new-process\n"
    assert q3b.competing_model_process(_inventories(args, comm)) is None


def test_process_inventory_allows_irrelevant_missing_comm_pid():
    args = "123 10 0.0 /usr/bin/python worker.py\n"
    comm = "456 /usr/bin/new-process\n"
    assert q3b.competing_model_process(_inventories(args, comm)) is None


def test_process_inventory_allows_relevant_missing_pid_proven_gone(monkeypatch):
    monkeypatch.setattr(q3b, "_trusted_claude_bundle", lambda: True)
    args = f"123 10 0.0 {q3b.CLAUDE_DESKTOP_EXECUTABLE} --type=renderer\n"
    comm = "456 /usr/bin/new-process\n"
    assert q3b.competing_model_process(
        _inventories(args, comm), pid_probe=lambda pid: "gone"
    ) is None


@pytest.mark.parametrize("status", ["alive", "unknown"])
def test_process_inventory_rejects_relevant_missing_pid_unproven(status):
    args = f"123 10 0.0 {q3b.CLAUDE_DESKTOP_EXECUTABLE} --type=renderer\n"
    comm = "456 /usr/bin/new-process\n"
    assert q3b.competing_model_process(
        _inventories(args, comm), pid_probe=lambda pid: status
    ) == "process inventory pid map mismatch"


def _exact_inventories(args, comm):
    def run(command):
        return comm if command[-1].endswith("comm=") else args
    return run


def test_process_inventory_uses_exact_ancestry_before_model_token_checks():
    current = os.getpid()
    parent, root, sibling = current + 1, current + 2, current + 3
    args = "\n".join([
        f"{current} {parent} 10 0.0 /usr/bin/python test",
        f"{parent} {root} 10 0.0 /usr/bin/python ancestor --mlx --gemma",
        f"{root} 0 10 0.0 /usr/bin/python root --qwen",
        f"{sibling} {root} 10 0.0 /usr/bin/python sibling --gemma",
    ]) + "\n"
    comm = "\n".join([
        f"{current} python",
        f"{parent} python",
        f"{root} python",
        f"{sibling} python",
    ]) + "\n"
    assert q3b.competing_model_process(_exact_inventories(args, comm)) == \
        "competing model activity detected"
    sibling_args = "\n".join(
        line for line in args.splitlines()
        if f"{sibling} {root} 10 0.0 /usr/bin/python sibling --gemma" not in line
    ) + "\n"
    assert q3b.competing_model_process(_exact_inventories(sibling_args, comm)) is None


def test_process_inventory_requires_current_pid_and_complete_parent_chain():
    current = os.getpid()
    missing_self = f"{current + 1} 0 10 0.0 python --gemma\n"
    assert q3b.competing_model_process(
        _exact_inventories(missing_self, f"{current + 1} python\n")
    ) == "process inventory ancestry malformed"
    missing_parent = f"{current} {current + 1} 10 0.0 python\n"
    assert q3b.competing_model_process(
        _exact_inventories(missing_parent, f"{current} python\n")
    ) == "process inventory ancestry malformed"


def test_process_inventory_rejects_ancestry_cycle_and_invalid_ppid():
    current = os.getpid()
    cycle = f"{current} {current + 1} 10 0.0 python\n{current + 1} {current} 10 0.0 python\n"
    comm = f"{current} python\n{current + 1} python\n"
    assert q3b.competing_model_process(_exact_inventories(cycle, comm)) == \
        "process inventory ancestry malformed"
    assert q3b._parse_process_args_inventory(
        f"-1 0 10 0.0 python\n"
    ) == "process inventory malformed"
    assert q3b._parse_process_args_inventory(
        f"{current} -1 10 0.0 python\n"
    ) == "process inventory malformed"


def test_process_inventory_rejects_malformed_argv_quoting():
    current = os.getpid()
    args = f"{current} 0 10 0.0 'unterminated\n"
    assert q3b._parse_process_args_inventory(args) == "process inventory malformed"


def test_process_inventory_accepts_argv_with_spaces():
    current = os.getpid()
    args = f"{current} 0 10 0.0 '/usr/bin/model helper' --model local\n"
    parsed = q3b._parse_process_args_inventory(args)
    assert isinstance(parsed, dict)
    assert parsed[current][3] == "'/usr/bin/model helper' --model local"


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
    assert q3b.competing_model_process(_inventories("4321 10 0.0 python child\n", "4321 python\n"), absent_pids=(4321,)) == "prior model child was not reaped"


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
        "raw": [{"pid": 10 if stage == "baseline" else 11, "arms": {stage: arm}, "order": [stage], "mlx_peak_bytes": 100,
                 "guard": {"version": "ironmule.q3f_child_guard.v1", "installed": True, "events": []}}],
        "per_arm": {stage: {"total_ns": summary, "prefill_ns": summary, "decode_ns": summary}},
        "token_identity": True, "token_count_identity": True, "stop_reason_identity": True,
        "deterministic": True, "reference_tokens": tokens, "ratios": {},
        "binding": {"model_id": q3b.MODEL_ID, "model_revision": q3b.EXPECTED_REVISION,
                     "model_manifest_sha256": q3b.EXPECTED_MODEL_MANIFEST_SHA256, "runtime_code_sha256": "b" * 64},
        "child_rss_peak_bytes": 100, "swap_samples": [1000, 1000],
        "swap_sample_times": [1.0, 1.25], "swap_sample_offsets": [0.0, 0.25],
        "sampler_errors": [], "max_swap_used_bytes": 1000,
    }


def _preflight_for_test():
    return {
        "passed": True, "identity": {"model_id": q3b.MODEL_ID, "model_revision": q3b.EXPECTED_REVISION, "model_manifest_sha256": q3b.EXPECTED_MODEL_MANIFEST_SHA256},
        "installed_memory_bytes": 1000, "environment": {"swap_used_bytes": 1000},
    }


def test_preflight_rejects_formally_valid_but_wrong_model_manifest(monkeypatch):
    monkeypatch.setattr(q3b, "system_environment", lambda _run: {
        "power_source": "AC", "low_power_mode": False, "thermal_state": "nominal",
        "swap_used_bytes": 1000, "memory_free_percent": 50})
    monkeypatch.setattr(q3b, "installed_memory_bytes", lambda _run: 1000)
    monkeypatch.setattr(q3b, "loadavg_gate", lambda *_args, **_kwargs: {"passed": True})
    monkeypatch.setattr(q3b, "competing_model_process", lambda _run: None)
    monkeypatch.setattr(q3b, "_git_binding", lambda *_args: {"clean": True, "commit": "b" * 40})
    monkeypatch.setattr(q3b, "_preregistration_matches", lambda: True)
    monkeypatch.setattr(q3b, "runtime_code_sha256", lambda *_args: "c" * 64)
    result = q3b.preflight(
        identity_resolver=lambda _root: {"model_id": q3b.MODEL_ID,
                                         "model_revision": q3b.EXPECTED_REVISION,
                                         "model_manifest_sha256": "f" * 64})
    assert result["checks"]["model_identity_exact"] is False
    assert result["passed"] is False


def test_worker_capability_rejects_formally_valid_but_wrong_model_manifest(monkeypatch):
    nonce = "nonce"
    expected = {"identity": {"model_id": q3b.MODEL_ID,
                              "model_revision": q3b.EXPECTED_REVISION,
                              "model_manifest_sha256": "f" * 64},
                "runtime_code_sha256": "c" * 64, "stage": "baseline",
                "initial_swap": 1000, "installed_memory": 1000}
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, json.dumps({"nonce": nonce, "expected": expected}).encode())
    finally:
        os.close(write_fd)
    monkeypatch.setenv("IRONMULE_Q3B_CAP_FD", str(read_fd))
    monkeypatch.setenv("IRONMULE_Q3B_CAP_NONCE", nonce)
    monkeypatch.setenv("IRONMULE_Q3B_EXPECTED", json.dumps(expected))
    try:
        with pytest.raises(q3b.CanaryRefused, match="capability"):
            q3b._read_capability()
    finally:
        try:
            os.close(read_fd)
        except OSError:
            pass


def test_stage_validator_rejects_formally_valid_but_wrong_model_manifest():
    result = _stage_result("baseline")
    result["binding"]["model_manifest_sha256"] = "f" * 64
    assert q3b.validate_stage_result(result, "baseline")[0] is False


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


def test_descriptive_timing_is_finite_labeled_and_non_gating():
    baseline = _stage_result("baseline", tokens=[1])
    candidate = _stage_result("candidate", tokens=[1])
    candidate["raw"][0]["arms"]["candidate"]["total_ns"] = [1.0, 1.0, 1.0]
    candidate["raw"][0]["arms"]["candidate"]["prefill_ns"] = [1.0, 1.0, 1.0]
    candidate["raw"][0]["arms"]["candidate"]["decode_ns"] = [1.0, 1.0, 1.0]
    candidate["per_arm"]["candidate"] = {
        metric: {"n": 1, "median": 1.0, "min": 1.0, "max": 1.0, "p95": 1.0, "stdev": 0.0}
        for metric in ("total_ns", "prefill_ns", "decode_ns")
    }
    descriptive = q3b.descriptive_timing(baseline, candidate)
    assert descriptive["descriptive_only"] is True
    assert descriptive["performance_valid"] is False
    assert descriptive["order_confounded"] is True
    assert descriptive["statistical_confidence"] == "none"
    assert descriptive["stages"]["baseline"]["median_total_ms"] == 2e-6
    assert descriptive["stages"]["baseline"]["median_prefill_ms"] == 2e-6
    assert descriptive["stages"]["baseline"]["median_decode_ms"] == 2e-6
    assert descriptive["stages"]["baseline"]["logical_output_tokens"] == 1
    assert descriptive["stages"]["baseline"]["physical_output_tokens"] == 2
    assert descriptive["stages"]["baseline"]["total_output_tokens_per_s"] == pytest.approx(5e8)
    assert descriptive["stages"]["baseline"]["decode_steps_per_s"] == pytest.approx(5e8)
    assert descriptive["stages"]["candidate"]["total_output_tokens_per_s"] == pytest.approx(1e9)
    assert descriptive["stages"]["candidate"]["decode_steps_per_s"] == pytest.approx(1e9)
    for metric in ("total_ms", "prefill_ms", "decode_ms",
                   "total_output_tokens_per_s", "decode_steps_per_s"):
        comparison = descriptive["comparison"]["candidate_over_baseline"][metric]
        assert comparison["ratio"] == (0.5 if metric.endswith("_ms") else 2.0)
        assert comparison["percent_faster"] == (50.0 if metric.endswith("_ms") else 100.0)
        assert comparison["direction"] == ("lower_is_better" if metric.endswith("_ms") else "higher_is_better")
    assert "confidence_interval" not in descriptive
    assert "winner" not in descriptive
    assert "promotion_allowed" not in descriptive


def test_descriptive_timing_refuses_incomplete_stage():
    with pytest.raises(ValueError, match="complete candidate"):
        q3b.descriptive_timing(_stage_result("baseline"), {"stage": "candidate"})


def test_descriptive_timing_does_not_change_pass_status(monkeypatch, tmp_path):
    monkeypatch.setattr(q3b, "preflight", lambda **_kwargs: _preflight_for_test())
    monkeypatch.setattr(q3b, "_start_stage", lambda stage, *_args: (_stage_result(stage), []))
    monkeypatch.setattr(q3b, "_post_environment", lambda *_args: _environment())
    output = tmp_path / "passed.json"
    assert q3b.main(["--execute", "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["status"] == "SAFETY_CANARY_PASS"
    assert result["performance_valid"] is False
    assert result["descriptive_timing"]["descriptive_only"] is True


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
                     "model_manifest_sha256": q3b.EXPECTED_MODEL_MANIFEST_SHA256},
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
                                                "model_manifest_sha256": q3b.EXPECTED_MODEL_MANIFEST_SHA256},
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
    activated = []
    original = __import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("ironmule", "mlx", "mlx_lm")):
            imported.append(name)
            raise AssertionError(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)
    monkeypatch.setattr(q3b, "_activate_exact_repo_root", lambda: activated.append(True))
    assert q3b._stage_worker() == 2
    assert imported == []
    assert activated == []


def test_valid_capability_activates_exact_root_after_worker_checks(monkeypatch):
    expected = {
        "identity": {"model_id": q3b.MODEL_ID, "model_revision": q3b.EXPECTED_REVISION,
                     "model_manifest_sha256": q3b.EXPECTED_MODEL_MANIFEST_SHA256},
        "runtime_code_sha256": "b" * 64, "stage": "baseline", "initial_swap": 1000,
        "installed_memory": 1000,
    }
    monkeypatch.setattr(q3b, "_read_capability", lambda: expected)
    monkeypatch.setattr(q3b, "runtime_code_sha256", lambda *_args, **_kwargs: "b" * 64)
    monkeypatch.setattr(q3b, "_read_swap_sample", lambda _run: 1000)
    monkeypatch.setenv("IRONMULE_Q3B_WORKER_DEADLINE", "9999999999")
    original_path = list(sys.path)
    monkeypatch.setattr(sys, "path", original_path)
    activated = []
    original_activate = q3b._activate_exact_repo_root

    def tracking_activate():
        root = original_activate()
        activated.append(root)
        return root

    monkeypatch.setattr(q3b, "_activate_exact_repo_root", tracking_activate)
    monkeypatch.setattr(q3b, "_load_q3a_helpers",
                        lambda: (_ for _ in ()).throw(q3b.CanaryRefused("test stop")))
    assert q3b._stage_worker() == 2
    assert activated == [ROOT]
    assert sys.path[0] == str(ROOT)


def test_exact_repo_root_rejects_shadow_spec(monkeypatch, tmp_path):
    shadow_package = tmp_path / "ironmule"
    shadow_package.mkdir()
    shadow_init = shadow_package / "__init__.py"
    shadow_init.write_text("# shadow\n")
    shadow_spec = types.SimpleNamespace(
        origin=str(shadow_init), submodule_search_locations=[str(shadow_package)]
    )
    monkeypatch.setattr(q3b.importlib.util, "find_spec", lambda _name: shadow_spec)
    with pytest.raises(q3b.CanaryRefused, match="outside the exact repository root"):
        q3b._activate_exact_repo_root()


def test_exact_repo_root_rejects_preloaded_foreign_module(monkeypatch, tmp_path):
    foreign = types.ModuleType("ironmule")
    foreign.__file__ = str(tmp_path / "ironmule.py")
    monkeypatch.setitem(sys.modules, "ironmule", foreign)
    with pytest.raises(q3b.CanaryRefused, match="preloaded foreign"):
        q3b._activate_exact_repo_root()
