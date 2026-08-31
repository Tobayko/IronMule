import ast
import builtins
import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("q3a_path_interaction", ROOT / "research/q3a_path_interaction.py")
assert SPEC.loader is not None
q3a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q3a)


def _safe_runner(command):
    key = tuple(command)
    name = Path(command[0]).name
    if name == "pmset" and key[-2:] == ("-g", "batt"):
        return "Now drawing from 'AC Power'"
    if name == "pmset" and key[-2:] == ("-g", "lowpowermode"):
        return "lowpowermode 0"
    if name == "pmset" and key[-2:] == ("-g", "therm"):
        return "No thermal warning level has been recorded\nNo performance warning level has been recorded"
    if name == "sysctl" and key[-2:] == ("-n", "vm.swapusage"):
        return "total = 4096.00M used = 128.00M free = 3968.00M"
    if name == "sysctl" and key[-2:] == ("-n", "hw.memsize"):
        return str(16 * 1024**3)
    if name == "ps":
        return "1 100 0.0 python idle\n"
    if name == "git" and "rev-parse" in key:
        return "a" * 40 + "\n"
    if name == "git":
        return ""
    raise AssertionError(command)


def _identity(_root):
    return {"model_id": q3a.MODEL_ID, "model_revision": q3a.EXPECTED_REVISION, "model_manifest_sha256": "a" * 64}


def test_plan_is_dry_run_safe_and_exactly_binds_the_two_arms(capsys, tmp_path):
    output = tmp_path / "q3a.json"
    assert q3a.main(["--output", str(output)]) == 0
    assert not output.exists()
    plan = json.loads(capsys.readouterr().out)
    assert plan["plan"]["baseline"] == q3a.Q2_INCUMBENT
    assert plan["plan"]["candidate"] == q3a.Q3A_CANDIDATE
    assert plan["estimated_wall_seconds"] == 270


def test_execute_rejects_existing_output_before_preflight(monkeypatch, tmp_path, capsys):
    output = tmp_path / "existing.json"
    output.write_text("already here")
    monkeypatch.setattr(q3a, "preflight", lambda: (_ for _ in ()).throw(AssertionError("preflight must not run")))
    assert q3a.main(["--execute", "--output", str(output)]) == 2
    assert "output path" in capsys.readouterr().err


def test_direct_worker_refuses_before_any_ironmule_or_mlx_import(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("ironmule", "mlx", "mlx_lm")):
            raise AssertionError(f"forbidden import before capability: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert q3a._worker() == 2


def test_worker_timeout_stops_group_after_successful_term_wait(monkeypatch):
    class FakeProcess:
        pid = 4321
        returncode = None

        def __init__(self):
            self.calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(["worker"], timeout, output="")
            return "", ""

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

    process = FakeProcess()
    signals = []
    monkeypatch.setattr(q3a.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(q3a.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    gone = iter([False, True])
    monkeypatch.setattr(q3a, "_wait_process_group_gone", lambda _pid: next(gone))
    result, markers = q3a._start_worker(_identity(None), q3a.time.monotonic() + 20)
    assert "timeout" in result["failure"]
    assert [sig for _, sig in signals] == [q3a.signal.SIGTERM, q3a.signal.SIGKILL]
    assert signals[0][0] == process.pid


def test_worker_timeout_does_not_kill_group_that_disappeared(monkeypatch):
    class FakeProcess:
        pid = 4322
        returncode = None

        def __init__(self):
            self.calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(["worker"], timeout, output="")
            return "", ""

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

    process = FakeProcess()
    signals = []
    monkeypatch.setattr(q3a.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(q3a.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(q3a, "_wait_process_group_gone", lambda _pid: True)
    result, _ = q3a._start_worker(_identity(None), q3a.time.monotonic() + 20)
    assert "timeout" in result["failure"]
    assert [sig for _, sig in signals] == [q3a.signal.SIGTERM]


def test_preflight_requires_identity_environment_load_and_clean_git():
    result = q3a.preflight(run=_safe_runner, identity_resolver=_identity, load_sample=lambda: 1.0)
    assert result["passed"] is True
    assert result["checks"]["model_identity_exact"] is True
    assert result["checks"]["loadavg_gate"] is True
    assert result["checks"]["git_clean_and_bound"] is True


def test_thermal_parser_is_strict_and_accepts_current_nominal_sentence():
    assert q3a._thermal_nominal("No thermal warning level has been recorded\nNo performance warning level has been recorded")
    assert q3a._thermal_nominal("No thermal warning level has been recorded\nNo performance warning level has been recorded\nNo CPU power status is available\nCPU_Speed_Limit = 0\npressure = 0")
    assert not q3a._thermal_nominal("No thermal warning level has been recorded")
    assert not q3a._thermal_nominal("Thermal level = 0")
    assert not q3a._thermal_nominal("No thermal warning level has been recorded\nNo performance warning level has been recorded\nCPU_Speed_Limit = 100")
    assert not q3a._thermal_nominal("No thermal warning level has been recorded\nNo performance warning level has been recorded\nThrottle = unknown")
    assert not q3a._thermal_nominal("not nominal")
    assert not q3a._thermal_nominal("thermal level = unknown")


def test_load_and_process_gates_fail_closed():
    sleeps = []
    assert q3a.loadavg_gate(iter([1.0, 1.2, 1.5]).__next__, sleeper=sleeps.append)["passed"]
    assert sleeps == [1.0, 1.0]
    assert not q3a.loadavg_gate(iter([1.0, 2.1, 1.0]).__next__, sleeper=lambda _seconds: None)["passed"]
    assert q3a.competing_model_process(lambda _command: "broken") == "process inventory malformed"
    assert q3a.competing_model_process(lambda _command: "123 1000000 0.0 ollama serve\n") == "competing model activity detected"
    assert q3a.competing_model_process(lambda _command: "123 10 0.0 llama-server --model local\n") == "competing model activity detected"
    assert q3a.competing_model_process(lambda _command: "123 10 0.0 mlx_lm.server\n") == "competing model activity detected"
    assert q3a.competing_model_process(lambda _command: "123 100 2.0 llama.cpp worker\n") == "competing model activity detected"
    assert q3a.competing_model_process(lambda _command: "123 100 2.0 claude worker\n") == "competing model activity detected"
    assert q3a.competing_model_process(lambda _command: "123 10 0.0 claude idle\n") is None
    assert q3a.competing_model_process(lambda _command: "123 100 nope claude worker\n") == "process inventory malformed"


def _worker_result():
    arms = {}
    for name in ("q2_incumbent", "fused_argmax_path"):
        logical = [[1, 2, 3] for _ in range(q3a.REPEATS)]
        physical = [[0, 1, 2, 3] for _ in range(q3a.REPEATS)]
        arms[name] = {
            "total_ns": [100.0 + i for i in range(q3a.REPEATS)], "prefill_ns": [60.0 + i for i in range(q3a.REPEATS)], "decode_ns": [40.0 + i for i in range(q3a.REPEATS)],
            "logical_tokens": logical[0], "logical_tokens_per_repeat": logical, "physical_tokens_per_repeat": physical,
            "token_counts": [{"logical": 3, "physical": 4} for _ in range(q3a.REPEATS)], "stop_reasons": ["length"] * q3a.REPEATS,
            "capacities": [384] * q3a.REPEATS, "deterministic": True, "decode_steps": 3, "prompt_tokens": 322, "mlx_peak_bytes": 100,
        }
    raw = []
    for index in range(q3a.PROCESSES):
        order = ["q2_incumbent", "fused_argmax_path"] if index % 2 == 0 else ["fused_argmax_path", "q2_incumbent"]
        raw.append({"pid": index + 10, "arms": arms, "order": order, "mlx_peak_bytes": 100})
    markers = [{"index": index, "pid": index + 10, "order": raw[index]["order"], "arms": ["fused_argmax_path", "q2_incumbent"]} for index in range(q3a.PROCESSES)]
    summaries = {"total_ns": {"n": 6, "median": 103.0, "min": 103.0, "max": 103.0, "p95": 103.0, "stdev": 0.0}, "prefill_ns": {"n": 6, "median": 63.0, "min": 63.0, "max": 63.0, "p95": 63.0, "stdev": 0.0}, "decode_ns": {"n": 6, "median": 43.0, "min": 43.0, "max": 43.0, "p95": 43.0, "stdev": 0.0}}
    ratio = {"median_ratio": 1.0, "ci_low": 1.0, "ci_high": 1.0, "pairs": [1.0] * 6}
    return {"arms": {"q2_incumbent": q3a.Q2_INCUMBENT, "fused_argmax_path": q3a.Q3A_CANDIDATE}, "processes": 6, "repeats": 7, "warmup": 2, "raw": raw, "per_arm": {name: dict(summaries) for name in arms}, "token_identity": True, "token_count_identity": True, "stop_reason_identity": True, "deterministic": True, "reference_tokens": [1, 2, 3], "ratios": {"fused_argmax_path/q2_incumbent": {metric: ratio for metric in ("total_ns", "prefill_ns", "decode_ns")}}, "binding": {"model_id": q3a.MODEL_ID, "model_revision": q3a.EXPECTED_REVISION, "model_manifest_sha256": "a" * 64, "runtime_code_sha256": q3a.runtime_code_sha256()}, "rss_peak_bytes": 100, "progress_markers": markers}


def _before_after():
    identity = {"model_id": q3a.MODEL_ID, "model_revision": q3a.EXPECTED_REVISION, "model_manifest_sha256": "a" * 64}
    return ({"peak_ceiling_bytes": 1000, "identity": identity, "environment": {"swap_used_bytes": 100}}, {"swap_used_bytes": 50, "power_source": "AC", "low_power_mode": False, "thermal_state": "nominal", "loadavg": {"passed": True}, "competing_model_process": None})


def test_result_validation_is_exact_and_fail_closed():
    before, after = _before_after()
    result = _worker_result()
    assert q3a.validate_result(result, before, after, result["binding"]["runtime_code_sha256"])["passed"]
    assert q3a.validate_result({}, before, after, "a" * 64)["passed"] is False
    malformed = dict(result, rss_peak_bytes=float("nan"))
    assert q3a.validate_result(malformed, before, after, result["binding"]["runtime_code_sha256"])["passed"] is False


def test_huge_json_integers_fail_closed_without_overflow():
    before, after = _before_after()
    huge = 10 ** 10000
    assert q3a._finite(huge) is False
    timing = _worker_result()
    timing["raw"][0]["arms"]["q2_incumbent"]["total_ns"][0] = huge
    assert q3a.validate_result(timing, before, after, timing["binding"]["runtime_code_sha256"])["passed"] is False
    ratio = _worker_result()
    ratio["ratios"]["fused_argmax_path/q2_incumbent"]["total_ns"]["median_ratio"] = huge
    assert q3a.validate_result(ratio, before, after, ratio["binding"]["runtime_code_sha256"])["passed"] is False


def test_reported_summary_and_ratio_must_match_raw_samples():
    before, after = _before_after()
    forged_ratio = _worker_result()
    for child in forged_ratio["raw"]:
        child["arms"]["fused_argmax_path"]["total_ns"] = [120.0] * q3a.REPEATS
    forged_ratio["ratios"]["fused_argmax_path/q2_incumbent"]["total_ns"]["median_ratio"] = 0.99
    assert q3a.validate_result(forged_ratio, before, after, forged_ratio["binding"]["runtime_code_sha256"])["passed"] is False
    forged_summary = _worker_result()
    forged_summary["per_arm"]["q2_incumbent"]["total_ns"]["median"] = 99.0
    assert q3a.validate_result(forged_summary, before, after, forged_summary["binding"]["runtime_code_sha256"])["passed"] is False


@pytest.mark.parametrize("field", ["token_identity", "token_count_identity", "stop_reason_identity"])
def test_identity_flags_are_recomputed_across_all_raw_arms(field):
    before, after = _before_after()
    result = _worker_result()
    for child in result["raw"]:
        arm = child["arms"]["fused_argmax_path"]
        if field == "token_identity":
            arm["logical_tokens"] = [1, 2, 4]
            arm["logical_tokens_per_repeat"] = [[1, 2, 4] for _ in range(q3a.REPEATS)]
        elif field == "token_count_identity":
            arm["physical_tokens_per_repeat"] = [[0, 1, 2, 3, 4] for _ in range(q3a.REPEATS)]
            arm["token_counts"] = [{"logical": 3, "physical": 5} for _ in range(q3a.REPEATS)]
            arm["decode_steps"] = 4
        else:
            arm["stop_reasons"] = ["eos"] * q3a.REPEATS
    assert result[field] is True
    assert q3a.validate_result(result, before, after, result["binding"]["runtime_code_sha256"])["passed"] is False


def test_deterministic_flag_is_recomputed_from_per_repeat_physical_data():
    before, after = _before_after()
    result = _worker_result()
    arm = result["raw"][0]["arms"]["fused_argmax_path"]
    arm["physical_tokens_per_repeat"][1] = [0, 1, 2, 9]
    assert result["deterministic"] is True
    assert q3a.validate_result(result, before, after, result["binding"]["runtime_code_sha256"])["passed"] is False


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [(0.98, 0.994, "GAIN"), (1.006, 1.02, "LOSS"), (0.996, 1.004, "PRACTICALLY_NEUTRAL"), (0.99, 1.01, "INCONCLUSIVE")],
)
def test_analysis_is_preregistered_and_conservative(low, high, expected):
    result = _worker_result()
    ratio = result["ratios"]["fused_argmax_path/q2_incumbent"]
    for metric in ratio.values():
        metric["ci_low"], metric["ci_high"], metric["median_ratio"] = low, high, (low + high) / 2
    assert q3a.analyze_ratio(result)["classification"] == expected


def test_exclusive_result_write_is_private_and_non_overwriting(tmp_path):
    path = tmp_path / "result.json"
    q3a._write_exclusive(path, b"{}\n")
    assert path.read_bytes() == b"{}\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        q3a._write_exclusive(path, b"overwrite\n")


def test_top_level_module_imports_are_stdlib_only():
    tree = ast.parse((ROOT / "research/q3a_path_interaction.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] not in {"ironmule", "mlx", "mlx_lm"} for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in {"ironmule", "mlx", "mlx_lm"}
