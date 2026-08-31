import copy
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("q3c_performance_replication", ROOT / "research/q3c_performance_replication.py")
assert SPEC.loader is not None
q3c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q3c)


class _FakeAB:
    @staticmethod
    def summarise(values):
        ordered = sorted(values)
        median = ordered[len(ordered) // 2]
        return {"n": len(values), "median": median, "min": min(values), "max": max(values),
                "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], "stdev": 0.0}

    @staticmethod
    def paired_ratio(candidate, baseline, **_kwargs):
        pairs = [c / b for c, b in zip(candidate, baseline)]
        ratio = statistics.median(pairs)
        return {"median_ratio": ratio, "ci_low": ratio, "ci_high": ratio, "pairs": pairs}

    @staticmethod
    def validate_result(result, *, processes, repeats, warmup, expected_arms, baseline, candidate):
        if (set(result) != {"arms", "processes", "repeats", "warmup", "raw", "per_arm",
                            "token_identity", "token_count_identity", "stop_reason_identity",
                            "deterministic", "reference_tokens", "ratios"}
                or result["arms"] != expected_arms or result["processes"] != processes
                or result["repeats"] != repeats or result["warmup"] != warmup
                or len(result["raw"]) != processes):
            return False, "fake-top-fields"
        pids = set()
        for index, child in enumerate(result["raw"]):
            expected_order = [baseline, candidate] if index % 2 == 0 else [candidate, baseline]
            if (set(child) != {"pid", "arms", "order", "mlx_peak_bytes"}
                    or child["order"] != expected_order or set(child["arms"]) != {baseline, candidate}
                    or not isinstance(child["pid"], int) or child["pid"] in pids):
                return False, "fake-child-fields"
            pids.add(child["pid"])
            for name in (baseline, candidate):
                arm = child["arms"][name]
                if any(len(arm[key]) != repeats for key in ("total_ns", "prefill_ns", "decode_ns", "logical_tokens_per_repeat", "physical_tokens_per_repeat", "token_counts", "stop_reasons", "capacities")):
                    return False, "fake-repeat-fields"
        return True, None


q3c._q3b_runtime = lambda: type("Q3b", (), {"ab": _FakeAB()})()


def _cleanup_evidence(pid, child_pids=None):
    child_pids = list(child_pids or [pid])
    known_pids = sorted(set([pid] + child_pids))
    baseline_identities = [{"pid": pid, "start": "00:00:01", "uid": 501, "sid": pid, "pgid": pid}]
    baseline_digest = hashlib.sha256(json.dumps(baseline_identities, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    snapshots = [
        {"monotonic": 1.0, "command_ok": True, "parse_ok": True, "records": [], "gone_pids": [], "enrichment": [], "comm": {"monotonic": 1.01, "command_ok": True, "parse_ok": True, "records": [{"pid": 1, "comm": "launchd"}], "error": None}, "error": None},
        {"monotonic": 1.25, "command_ok": True, "parse_ok": True, "records": [], "gone_pids": [], "enrichment": [], "comm": {"monotonic": 1.26, "command_ok": True, "parse_ok": True, "records": [{"pid": 1, "comm": "launchd"}], "error": None}, "error": None},
    ]
    return {"schema": "ironmule.cleanup.v2",
            "identity": {"worker_pid": pid, "parent_pid": 1, "worker_ancestor_pids": [1], "pgid": pid, "sid": pid, "uid": 501,
                         "known_descendant_pids": known_pids, "known_process_starts": {str(item): "00:00:01" for item in known_pids},
                         "known_process_ppids": {str(item): (1 if item == pid else pid) for item in known_pids},
                         "known_process_sids": {str(item): pid for item in known_pids},
                         "known_process_pgids": {str(item): pid for item in known_pids},
                         "uid_invariant": {"owner_uid": 501, "worker_uid": 501, "same_non_root": True},
                         "spawn_baseline": {"valid": True, "digest": baseline_digest,
                                            "identities": baseline_identities}},
            "worker_reaped": True,
            "signal_attempts": [{"signal": "SIGTERM", "status": "not_found"}],
            "descendant_kill_attempts": [],
            "verification": {"method": "two_independent_ps_snapshots", "pre_signal_snapshot": snapshots[0], "pre_escalation_snapshot": snapshots[0], "snapshots": snapshots,
                              "members": [[], []], "leader": [[], []], "descendants": [[], []], "new_processes": [[], []],
                              "unrelated_new_processes": [], "attribution_reasons": [], "guard_proof": [{"pid": item, "guard": {"version": "ironmule.q3f_child_guard.v1", "installed": True, "events": []}} for item in child_pids], "child_ledger": [{"pid": item, "ppid": (1 if item == pid else pid), "pgid": pid, "sid": pid, "uid": 501, "start": "00:00:01", "callback_monotonic": 1.0 + index, "guard_version": "ironmule.q3f_child_guard.v1", "guard_event_count": 0} for index, item in enumerate(child_pids)], "snapshot_count": 2, "snapshot_gap_seconds": 0.25,
                              "independent": True, "global_process_inventory": {"enabled": True, "known": True, "competing": None}, "group_gone": True},
            "resolved_errors": ["SIGTERM:not_found"], "unresolved_errors": [],
            "guard_proof": [{"pid": item, "guard": {"version": "ironmule.q3f_child_guard.v1", "installed": True, "events": []}} for item in child_pids],
            "child_ledger": [{"pid": item, "ppid": (1 if item == pid else pid), "pgid": pid, "sid": pid, "uid": 501, "start": "00:00:01", "callback_monotonic": 1.0 + index, "guard_version": "ironmule.q3f_child_guard.v1", "guard_event_count": 0} for index, item in enumerate(child_pids)]}


def _result(phase="R", incumbent_time=0.8568e9, candidate_time=0.8569e9):
    candidate = q3c.PHASE_CANDIDATE[phase]
    other_time = incumbent_time if phase == "R" else candidate_time
    arms = copy.deepcopy(q3c.PHASE_ARMS[phase])
    names = list(arms)
    raw = []
    for index in range(q3c.PROCESSES):
        child_arms = {}
        for name in names:
            duration = 1e9 if name == "baseline" else other_time
            tokens = [4, 5]
            physical = [0, 4, 5]
            child_arms[name] = {
                "total_ns": [duration] * q3c.REPEATS,
                "prefill_ns": [duration / 2] * q3c.REPEATS,
                "decode_ns": [duration / 2] * q3c.REPEATS,
                "logical_tokens": tokens,
                "logical_tokens_per_repeat": [tokens] * q3c.REPEATS,
                "physical_tokens_per_repeat": [physical] * q3c.REPEATS,
                "token_counts": [{"logical": 2, "physical": 3}] * q3c.REPEATS,
                "stop_reasons": ["length"] * q3c.REPEATS,
                "capacities": [32] * q3c.REPEATS,
                "deterministic": True,
                "decode_steps": 2,
                "prompt_tokens": q3c.PROMPT_TOKENS,
                "mlx_peak_bytes": 100,
            }
        order = [names[0], names[1]] if index % 2 == 0 else [names[1], names[0]]
        pid = 1000 + index + (0 if phase == "R" else 100)
        raw.append({"pid": pid, "arms": child_arms,
                    "order": order, "mlx_peak_bytes": 100,
                    "guard": {"version": "ironmule.q3f_child_guard.v1", "installed": True, "events": []}})
    worker_pid = 9000 + (0 if phase == "R" else 100)
    child_ledger = [{"pid": child["pid"], "ppid": worker_pid, "pgid": worker_pid, "sid": worker_pid, "uid": 501,
                     "start": "00:00:01", "callback_monotonic": float(index + 1),
                     "guard_version": "ironmule.q3f_child_guard.v1", "guard_event_count": 0}
                    for index, child in enumerate(raw)]
    child_start_markers = [{"index": index, "pid": child["pid"], "ledger": child_ledger[index],
                            "row": {"pid": child["pid"], "ppid": worker_pid, "pgid": worker_pid,
                                    "sid": worker_pid, "uid": 501, "stat": "S",
                                    "start": "00:00:01", "args": "/usr/bin/python child"},
                            "comm": {"pid": child["pid"], "comm": "python"}}
                           for index, child in enumerate(raw)]
    ab = _FakeAB()
    per_arm = {name: {metric: q3c._summarise([statistics.median(child["arms"][name][metric]) for child in raw])
                      for metric in ("total_ns", "prefill_ns", "decode_ns")}
               for name in names}
    # Keep this fixture independent of implementation aggregation details.
    ratios = {f"{names[1]}/{names[0]}": {metric: ab.paired_ratio(
        [statistics.median(child["arms"][names[1]][metric]) for child in raw],
        [statistics.median(child["arms"][names[0]][metric]) for child in raw])
        for metric in ("total_ns", "prefill_ns", "decode_ns")}}
    result = {"phase": phase, "phase_plan": q3c.phase_plan(phase), "arms": arms,
              "processes": q3c.PROCESSES, "repeats": q3c.REPEATS, "warmup": q3c.WARMUP,
              "raw": raw, "per_arm": per_arm, "token_identity": True,
              "token_count_identity": True, "stop_reason_identity": True, "deterministic": True,
              "reference_tokens": [4, 5], "ratios": ratios,
              "binding": {"model_id": q3c.MODEL_ID, "model_revision": q3c.EXPECTED_REVISION,
                          "model_manifest_sha256": q3c.EXPECTED_MODEL_MANIFEST_SHA256, "runtime_code_sha256": q3c.runtime_code_sha256()},
              "child_rss_peak_bytes": 100, "swap_samples": [100, 100],
              "swap_sample_times": [1.0, 1.25], "swap_sample_offsets": [0.0, 0.25],
              "sampler_errors": [], "max_swap_used_bytes": 100, "phase_initial_swap_bytes": 100,
              "derived_rates": {}, "child_ledger": child_ledger,
              "child_start_markers": child_start_markers,
              "cleanup": _cleanup_evidence(worker_pid,
                                            [child["pid"] for child in raw])}
    result["derived_rates"] = q3c.derive_rates(result, candidate)
    return result


def test_dry_run_is_stdlib_safe_and_does_not_write(tmp_path, capsys):
    output = tmp_path / "q3c.json"
    assert q3c.main(["--output", str(output)]) == 0
    assert not output.exists()
    plan = json.loads(capsys.readouterr().out)
    assert plan["plan"]["phases"][0]["processes"] == 6
    assert plan["plan"]["phases"][0]["order"] == [["baseline", "incumbent"], ["incumbent", "baseline"]] * 3
    assert plan["plan"]["post_phase_seconds"] == q3c.POST_PHASE_SECONDS
    assert plan["plan"]["final_reserve_seconds"] == q3c.FINAL_RESERVE_SECONDS
    assert plan["plan"]["sampler_interval_seconds"] == q3c.SAMPLE_INTERVAL_SECONDS
    assert plan["plan"]["max_sampler_samples"] == q3c.MAX_SWAP_SAMPLES
    assert plan["plan"]["swap_delta_limit_bytes"] == q3c.SWAP_DELTA_LIMIT_BYTES
    assert plan["plan"]["worker_output_max_bytes"] == q3c.MAX_WORKER_OUTPUT


def test_parent_validation_and_rates_do_not_import_ironmule_or_mlx(monkeypatch):
    import builtins
    original = builtins.__import__
    imported = []
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"ironmule", "mlx", "mlx_lm"}:
            imported.append(name)
            raise AssertionError(name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    result = _result("R")
    assert q3c.validate_phase_result(result, "R")[0]
    assert imported == []


def test_phase_plan_is_exact():
    assert q3c.phase_plan("R")["arms"]["baseline"] == q3c.BASE
    assert q3c.phase_plan("R")["arms"]["incumbent"] == q3c.INCUMBENT
    assert q3c.phase_plan("N")["arms"]["candidate"] == q3c.CANDIDATE
    with pytest.raises(ValueError):
        q3c.phase_plan("X")


def test_cleanup_validator_accepts_valid_session_different_from_process_group():
    identity = {"uid": 501, "known_descendant_pids": [42],
                "known_process_starts": {"42": "00:00:02"},
                "known_process_sids": {"42": 42},
                "known_process_pgids": {"42": 42}}
    records = [{"pid": 1, "ppid": 0, "pgid": 1, "uid": 0, "stat": "S",
                "start": "7:00AM", "args": "/sbin/launchd"},
               {"pid": 42, "ppid": 1, "pgid": 42, "uid": 501, "sid": 42,
                "stat": "S", "start": "00:00:02", "args": "/worker"},
               {"pid": 43, "ppid": 1, "pgid": 43, "uid": 501, "sid": 1,
                "stat": "S", "start": "00:00:03", "args": "/other"}]
    assert q3c._cleanup_snapshot_records_valid(records, identity)


def test_phase_validator_rejects_formally_valid_but_wrong_model_manifest():
    result = _result("R")
    result["binding"]["model_manifest_sha256"] = "f" * 64
    valid, reason = q3c.validate_phase_result(result, "R")
    assert valid is False
    assert reason == "model/runtime binding mismatch"


def test_q3c_preflight_requires_pinned_model_manifest(monkeypatch):
    class Runtime:
        COMMANDS = {"git": "/usr/bin/git"}
        _run_text = staticmethod(lambda _command: "")
        _deadline_runner = staticmethod(lambda _deadline, run: run)
        system_environment = staticmethod(lambda _run: {
            "power_source": "AC", "low_power_mode": False,
            "thermal_state": "nominal", "swap_used_bytes": 1000,
            "memory_free_percent": 50})
        installed_memory_bytes = staticmethod(lambda _run: 1000)
        loadavg_gate = staticmethod(lambda **_kwargs: {"passed": True})
        _git_binding = staticmethod(lambda *_args: {"clean": True, "commit": "b" * 40})
        competing_model_process = staticmethod(lambda _run: None)

    monkeypatch.setattr(q3c, "_q3b_runtime", lambda: Runtime)
    monkeypatch.setattr(q3c, "_untracked_runtime_inventory", lambda *_args: {"passed": True})
    wrong = {"model_id": q3c.MODEL_ID, "model_revision": q3c.EXPECTED_REVISION,
             "model_manifest_sha256": "f" * 64}
    monkeypatch.setattr(Runtime, "resolve_exact_local_identity", staticmethod(lambda _root: wrong), raising=False)
    rejected = q3c.preflight(root=ROOT)
    assert rejected["checks"]["model_identity_exact"] is False
    assert rejected["passed"] is False
    correct = {**wrong, "model_manifest_sha256": q3c.EXPECTED_MODEL_MANIFEST_SHA256}
    monkeypatch.setattr(Runtime, "resolve_exact_local_identity", staticmethod(lambda _root: correct), raising=False)
    accepted = q3c.preflight(root=ROOT)
    assert accepted["checks"]["model_identity_exact"] is True
    assert accepted["passed"] is True


def test_strict_phase_validation_rejects_forged_order_and_missing_evidence():
    result = _result("R")
    assert q3c.validate_phase_result(result, "R")[0], q3c.validate_phase_result(result, "R")
    result["raw"][1]["order"] = ["baseline", "incumbent"]
    assert not q3c.validate_phase_result(result, "R")[0]


@pytest.mark.parametrize("field", ["ppid", "start"])
def test_child_start_marker_cannot_forge_identity_maps(field):
    result = _result("R")
    marker = result["child_start_markers"][0]
    if field == "ppid":
        marker["row"][field] = 999999
    else:
        marker["row"][field] = "changed-start"
    assert q3c.validate_phase_result(result, "R")[0] is False


def test_legacy_cleanup_evidence_never_passes_phase_validation():
    result = _result("R")
    result["cleanup"] = {"worker_group_gone": True, "cleanup_errors": [], "child_cleanup_errors": []}
    assert q3c.validate_phase_result(result, "R")[0] is False
    result = _result("R")
    result.pop("derived_rates")
    assert not q3c.validate_phase_result(result, "R")[0]


def test_strict_phase_validation_rejects_huge_or_mutated_timing():
    result = _result("R")
    result["raw"][0]["arms"]["baseline"]["total_ns"] = [10 ** 1000] * q3c.REPEATS
    assert not q3c.validate_phase_result(result, "R")[0]
    result = _result("R")
    result["raw"][0]["arms"]["incumbent"]["prompt_tokens"] += 1
    assert not q3c.validate_phase_result(result, "R")[0]
    result = _result("R")
    result["binding"]["runtime_code_sha256"] = "c" * 64
    assert not q3c.validate_phase_result(result, "R")[0]


@pytest.mark.parametrize("mutate", [
    lambda result: result["raw"][0].__setitem__("pid", 0),
    lambda result: result["raw"][0].__setitem__("mlx_peak_bytes", 0),
    lambda result: result["raw"][0]["arms"]["baseline"].__setitem__("mlx_peak_bytes", 0),
    lambda result: result["raw"][0]["arms"]["baseline"]["physical_tokens_per_repeat"].__setitem__(0, []),
    lambda result: result["raw"][0]["arms"]["baseline"]["token_counts"].__setitem__(0, {"logical": True, "physical": 3}),
])
def test_ab_validation_rejects_invalid_identity_and_repeat_evidence(mutate):
    result = _result("R")
    mutate(result)
    assert not q3c.validate_phase_result(result, "R")[0]


def test_ab_validation_requires_exact_arm_and_metric_sets():
    result = _result("R")
    result["per_arm"]["unexpected"] = {}
    assert not q3c.validate_phase_result(result, "R")[0]
    result = _result("R")
    result["per_arm"]["baseline"]["extra"] = {}
    assert not q3c.validate_phase_result(result, "R")[0]


def test_ab_validation_summary_uses_strict_types():
    result = _result("R")
    result["per_arm"]["baseline"]["total_ns"]["n"] = 6.0
    assert not q3c.validate_phase_result(result, "R")[0]


def test_ab_validation_rejects_python_coercion_surrogates_and_bounded_timing():
    result = _result("R")
    result["arms"]["incumbent"]["readback_every"] = 2.0
    assert not q3c.validate_phase_result(result, "R")[0]
    result = _result("R")
    result["arms"]["incumbent"]["speculate_k"] = False
    assert not q3c.validate_phase_result(result, "R")[0]
    result = _result("R")
    result["processes"] = 6.0
    assert not q3c.validate_phase_result(result, "R")[0]
    result = _result("R")
    result["raw"][0]["arms"]["baseline"]["total_ns"][0] = 10 ** 100
    assert not q3c.validate_phase_result(result, "R")[0]


def test_safety_marker_keeps_defensive_completed_children_without_argv():
    records = []
    lock = q3c.threading.Lock()
    child = {"pid": 7, "arms": {"baseline": {"total_ns": [1]}}}
    q3c._record_completed_child(records, lock, child)
    child["arms"]["baseline"]["total_ns"][0] = 99
    emitted = []
    q3c._write_bounded_safety_marker(
        {"reason": "swap_delta_exceeded", "samples": [1, 2], "times": [0.0, 0.25],
         "offsets": [0.0, 0.25], "errors": []}, records, lock, emitted.append)
    assert emitted[0]["partial_children"] == [{"pid": 7, "arms": {"baseline": {"total_ns": [1]}}}]
    assert "argv" not in json.dumps(emitted[0], sort_keys=True)


def test_capability_json_rejects_non_finite_constants(monkeypatch):
    monkeypatch.setenv("IRONMULE_Q3C_CAP_FD", "3")
    monkeypatch.setenv("IRONMULE_Q3C_CAP_NONCE", "nonce")
    monkeypatch.setenv("IRONMULE_Q3C_EXPECTED", "{}")
    monkeypatch.setattr(q3c.os, "read", lambda _fd, _size: b'{"nonce":"nonce","expected":NaN}')
    monkeypatch.setattr(q3c.os, "close", lambda _fd: None)
    with pytest.raises(q3c.Q3cRefused):
        q3c._read_capability()


def test_worker_markers_reject_non_finite_constants_and_cleanup(monkeypatch):
    class Process:
        pid = 4324
        returncode = 0

        def communicate(self, timeout=None):
            return '@@{"value":NaN}\n', ""

    cleaned = []
    policy = type("Policy", (), {"_cleanup_worker": staticmethod(lambda process: cleaned.append(process.pid) or [])})()
    monkeypatch.setattr(q3c, "_q3b_runtime", lambda: policy)
    monkeypatch.setattr(q3c.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(q3c, "runtime_code_sha256", lambda: "b" * 64)
    outcome = q3c._start_phase("R", {"model_id": q3c.MODEL_ID,
                                    "model_revision": q3c.EXPECTED_REVISION,
                                    "model_manifest_sha256": q3c.EXPECTED_MODEL_MANIFEST_SHA256}, 100, 1000,
                                q3c.time.monotonic() + 60)
    assert outcome["failure"] == "phase worker result JSON malformed"
    assert cleaned == []
    assert outcome["cleanup"]["unresolved_errors"] == ["legacy cleanup evidence rejected"]


def _v2_cleanup(pid):
    baseline_identities = [{"pid": pid, "start": "00:00:01", "uid": 501, "sid": pid, "pgid": pid}]
    baseline_digest = hashlib.sha256(json.dumps(baseline_identities, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    snapshots = [{"monotonic": 1.0, "command_ok": True, "parse_ok": True, "records": [], "gone_pids": [], "enrichment": [], "comm": {"monotonic": 1.01, "command_ok": True, "parse_ok": True, "records": [{"pid": 1, "comm": "launchd"}], "error": None}, "error": None},
                 {"monotonic": 1.25, "command_ok": True, "parse_ok": True, "records": [], "gone_pids": [], "enrichment": [], "comm": {"monotonic": 1.26, "command_ok": True, "parse_ok": True, "records": [{"pid": 1, "comm": "launchd"}], "error": None}, "error": None}]
    return {"schema": "ironmule.cleanup.v2",
            "identity": {"worker_pid": pid, "parent_pid": 1, "worker_ancestor_pids": [1], "pgid": pid, "sid": pid, "uid": 501,
                         "known_descendant_pids": [pid], "known_process_starts": {str(pid): "00:00:01"},
                         "known_process_ppids": {str(pid): 1},
                         "known_process_sids": {str(pid): pid},
                         "known_process_pgids": {str(pid): pid},
                         "uid_invariant": {"owner_uid": 501, "worker_uid": 501, "same_non_root": True},
                         "spawn_baseline": {"valid": True, "digest": baseline_digest,
                                            "identities": baseline_identities}},
            "worker_reaped": True, "signal_attempts": [{"signal": "SIGTERM", "status": "not_found"}],
            "descendant_kill_attempts": [],
            "verification": {"method": "two_independent_ps_snapshots", "pre_signal_snapshot": snapshots[0], "pre_escalation_snapshot": snapshots[0], "snapshots": snapshots,
                              "members": [[], []], "leader": [[], []], "descendants": [[], []], "new_processes": [[], []],
                              "unrelated_new_processes": [], "attribution_reasons": [], "guard_proof": [{"pid": pid, "guard": {"version": "ironmule.q3f_child_guard.v1", "installed": True, "events": []}}], "child_ledger": [{"pid": pid, "ppid": 1, "pgid": pid, "sid": pid, "uid": 501, "start": "00:00:01", "callback_monotonic": 0.5, "guard_version": "ironmule.q3f_child_guard.v1", "guard_event_count": 0}], "snapshot_count": 2, "snapshot_gap_seconds": 0.25, "independent": True,
                              "global_process_inventory": {"enabled": True, "known": True, "competing": None}, "group_gone": True}, "resolved_errors": ["SIGTERM:not_found"],
            "unresolved_errors": [], "guard_proof": [{"pid": pid, "guard": {"version": "ironmule.q3f_child_guard.v1", "installed": True, "events": []}}],
            "child_ledger": [{"pid": pid, "ppid": 1, "pgid": pid, "sid": pid, "uid": 501, "start": "00:00:01", "callback_monotonic": 0.5, "guard_version": "ironmule.q3f_child_guard.v1", "guard_event_count": 0}]}


@pytest.mark.parametrize("stream_mode", ["success", "safety", "timeout", "communication"])
def test_start_phase_wires_v2_cleanup_for_every_worker_outcome(monkeypatch, stream_mode):
    events = []

    class Process:
        pid = 4390
        returncode = 0

        def communicate(self, timeout=None):
            events.append("communicate")
            if stream_mode == "timeout":
                raise q3c.subprocess.TimeoutExpired("worker", timeout)
            if stream_mode == "communication":
                raise KeyboardInterrupt
            if stream_mode == "safety":
                self.returncode = 2
                return '@SAFETY {"reason":"swap_delta_exceeded"}\n', ""
            return "@@{}\n", ""

    class Policy:
        @staticmethod
        def _capture_worker_identity(process):
            events.append("capture")
            return {"worker_pid": process.pid, "parent_pid": 1, "pgid": process.pid, "sid": process.pid, "uid": 501,
                    "known_descendant_pids": [process.pid], "known_process_starts": {str(process.pid): "00:00:01"},
                    "known_process_sids": {str(process.pid): process.pid},
                    "known_process_pgids": {str(process.pid): process.pid}}

        @staticmethod
        def _cleanup_worker_evidence(process, identity):
            events.append("cleanup")
            return _v2_cleanup(process.pid)

    monkeypatch.setattr(q3c, "_q3b_runtime", lambda: Policy)
    monkeypatch.setattr(q3c.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(q3c, "runtime_code_sha256", lambda: "b" * 64)
    outcome = q3c._start_phase("R", {"model_id": q3c.MODEL_ID,
                                    "model_revision": q3c.EXPECTED_REVISION,
                                    "model_manifest_sha256": q3c.EXPECTED_MODEL_MANIFEST_SHA256}, 100, 1000,
                                q3c.time.monotonic() + 60)
    assert outcome["cleanup"]["schema"] == "ironmule.cleanup.v2"
    assert outcome["cleanup"]["unresolved_errors"] == []
    assert events == ["capture", "communicate", "cleanup"]


def test_success_group_alive_uses_structured_cleanup_error(monkeypatch):
    class Process:
        pid = 4325
        returncode = 0

        def communicate(self, timeout=None):
            return "@@{}\n", ""

    policy = type("Policy", (), {
        "_cleanup_worker": staticmethod(lambda _process: (_ for _ in ()).throw(KeyboardInterrupt)),
    })()
    monkeypatch.setattr(q3c, "_q3b_runtime", lambda: policy)
    monkeypatch.setattr(q3c.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(q3c, "_group_gone", lambda _pid: False)
    monkeypatch.setattr(q3c, "runtime_code_sha256", lambda: "b" * 64)
    outcome = q3c._start_phase("R", {"model_id": q3c.MODEL_ID,
                                    "model_revision": q3c.EXPECTED_REVISION,
                                    "model_manifest_sha256": q3c.EXPECTED_MODEL_MANIFEST_SHA256}, 100, 1000,
                                q3c.time.monotonic() + 60)
    assert outcome["cleanup"]["verification"]["group_gone"] is False
    assert outcome["cleanup"]["unresolved_errors"] == ["legacy cleanup evidence rejected"]


def test_untracked_inventory_allows_only_known_licensed_fixture(tmp_path):
    policy = type("Q3b", (), {"COMMANDS": {"git": "/usr/bin/git"}})()
    allowed = q3c._untracked_runtime_inventory(
        tmp_path, lambda _command: "research/data/squad-dev-v1.1.json\n", policy)
    assert allowed["passed"] is True
    blocked = q3c._untracked_runtime_inventory(
        tmp_path, lambda _command: "research/q3c_performance_replication.py\n", policy)
    assert blocked["passed"] is False
    assert blocked["unexpected"] == ["research/q3c_performance_replication.py"]


def test_rate_formula_uses_physical_tokens_and_same_repeat_duration():
    result = _result("N")
    rates = result["derived_rates"]["process_medians"]["baseline"]
    assert rates["physical_output_tokens_per_s"][0] == pytest.approx(3.0)
    assert rates["decode_steps_per_s"][0] == pytest.approx(4.0)
    assert result["derived_rates"]["formula"]["physical_output_tokens_per_s"].startswith("len(physical")


def test_historical_decision_requires_all_replication_gates_and_no_promotion():
    replication = _result("R", incumbent_time=0.8568e9)
    candidate = _result("N", candidate_time=0.8569e9)
    decision = q3c.decide(replication, candidate)
    assert decision["status"] == "COMPLETE_PASS"
    assert decision["phase_R_reproduced"] is True
    assert decision["phase_N_preserved"] is True
    assert decision["promotion_allowed"] is False
    replication["ratios"]["incumbent/baseline"]["total_ns"]["ci_high"] = 1.01
    assert q3c.decide(replication, candidate)["phase_R_reproduced"] is False


def test_directional_ci_percentages_reverse_time_endpoints_and_keep_rate_order():
    replication = _result("R", incumbent_time=0.85e9)
    candidate = _result("N", candidate_time=0.84e9)
    decision = q3c.decide(replication, candidate)
    time_value = decision["metrics"]["time_percent_faster"]["total_ns"]
    assert time_value["percent_ci_low"] <= time_value["percent_faster"] <= time_value["percent_ci_high"]
    rate_value = decision["metrics"]["rate_percent_faster"]["physical_output_tokens_per_s"]
    assert rate_value["percent_ci_low"] <= rate_value["percent_faster"] <= rate_value["percent_ci_high"]


def test_cross_phase_identity_reports_pid_reuse():
    replication = _result("R")
    candidate = _result("N")
    candidate["raw"][0]["pid"] = replication["raw"][0]["pid"]
    assert q3c.cross_phase_identity(replication, candidate)["pids_disjoint"] is False


def test_nonzero_worker_marker_is_failure_and_cleanup_is_proven(monkeypatch):
    class Process:
        pid = 4321
        returncode = 1

        def communicate(self, timeout=None):
            return "@@{}\n", "worker failed"

    cleaned = []
    policy = type("Policy", (), {"_cleanup_worker": staticmethod(lambda process: cleaned.append(process.pid) or [])})()
    monkeypatch.setattr(q3c, "_q3b_runtime", lambda: policy)
    monkeypatch.setattr(q3c.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(q3c, "runtime_code_sha256", lambda: "b" * 64)
    outcome = q3c._start_phase("R", {"model_id": q3c.MODEL_ID,
                                    "model_revision": q3c.EXPECTED_REVISION,
                                    "model_manifest_sha256": q3c.EXPECTED_MODEL_MANIFEST_SHA256}, 100, 1000,
                                q3c.time.monotonic() + 60)
    assert outcome["failure"].startswith("phase worker exited")
    assert outcome["cleanup"]["verification"]["group_gone"] is False
    assert cleaned == []


def test_phase_communication_base_exception_is_cleaned_and_reaped(monkeypatch):
    class Process:
        pid = 4322
        returncode = 1

        def communicate(self, timeout=None):
            raise KeyboardInterrupt

    cleaned = []
    policy = type("Policy", (), {"_cleanup_worker": staticmethod(lambda process: cleaned.append(process.pid) or [])})()
    monkeypatch.setattr(q3c, "_q3b_runtime", lambda: policy)
    monkeypatch.setattr(q3c.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(q3c, "runtime_code_sha256", lambda: "b" * 64)
    outcome = q3c._start_phase("R", {"model_id": q3c.MODEL_ID,
                                    "model_revision": q3c.EXPECTED_REVISION,
                                    "model_manifest_sha256": q3c.EXPECTED_MODEL_MANIFEST_SHA256}, 100, 1000,
                                q3c.time.monotonic() + 60)
    assert outcome["failure"] == "phase communication failed"
    assert outcome["cleanup"]["verification"]["group_gone"] is False
    assert cleaned == []


def test_phase_communication_requires_text_streams_and_cleans_up(monkeypatch):
    class Process:
        pid = 4323
        returncode = 0

        def communicate(self, timeout=None):
            return b"@@{}\n", ""

    cleaned = []
    policy = type("Policy", (), {"_cleanup_worker": staticmethod(lambda process: cleaned.append(process.pid) or [])})()
    monkeypatch.setattr(q3c, "_q3b_runtime", lambda: policy)
    monkeypatch.setattr(q3c.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(q3c, "runtime_code_sha256", lambda: "b" * 64)
    outcome = q3c._start_phase("R", {"model_id": q3c.MODEL_ID,
                                    "model_revision": q3c.EXPECTED_REVISION,
                                    "model_manifest_sha256": q3c.EXPECTED_MODEL_MANIFEST_SHA256}, 100, 1000,
                                q3c.time.monotonic() + 60)
    assert outcome["failure"] == "phase worker communication returned non-text output"
    assert outcome["cleanup"]["verification"]["group_gone"] is False
    assert cleaned == []


def test_failed_final_resource_gate_cannot_be_overwritten_by_recovery(monkeypatch, tmp_path):
    r_result, n_result = _result("R"), _result("N")
    calls = []
    def stage_gate(*_args, **_kwargs):
        calls.append(True)
        return {"passed": len(calls) < 3}
    policy = type("Q3b", (), {
        "_post_environment": staticmethod(lambda *_args: {}),
        "_stage_gate": staticmethod(stage_gate),
    })()
    monkeypatch.setattr(q3c, "_q3b_runtime", lambda: policy)
    monkeypatch.setattr(q3c, "preflight", lambda **_kwargs: {"passed": True, "identity": {
        "model_id": q3c.MODEL_ID, "model_revision": q3c.EXPECTED_REVISION,
        "model_manifest_sha256": q3c.EXPECTED_MODEL_MANIFEST_SHA256}, "runtime_code_sha256": q3c.runtime_code_sha256(),
        "environment": {"swap_used_bytes": 1000}, "installed_memory_bytes": 1000})
    monkeypatch.setattr(q3c, "_start_phase", lambda phase, *_args: r_result if phase == "R" else n_result)
    monkeypatch.setattr(q3c, "validate_phase_result", lambda *_args: (True, "ok"))
    output = tmp_path / "final-failure.json"
    assert q3c.main(["--execute", "--output", str(output)]) == 2
    record = json.loads(output.read_text())
    assert record["status"] == "FAILED"
    assert "decision" not in record


def test_phase_r_failure_stops_phase_n_and_retains_partial_evidence(monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(q3c, "preflight", lambda **_kwargs: {"passed": True, "identity": {},
                                                               "environment": {"swap_used_bytes": 1000},
                                                               "installed_memory_bytes": 1000})
    def start(phase, *_args):
        started.append(phase)
        return {"failure": "phase timeout", "partial_evidence": {"raw": [{"pid": 7}]}}
    monkeypatch.setattr(q3c, "_start_phase", start)
    output = tmp_path / "failure.json"
    assert q3c.main(["--execute", "--output", str(output)]) == 2
    assert started == ["R"]
    record = json.loads(output.read_text())
    assert record["phases"][0]["partial_evidence"]["raw"][0]["pid"] == 7
    assert record["fallback"] == "BASE/current incumbent"


def test_phase_r_criteria_miss_is_recorded_but_does_not_skip_phase_n(monkeypatch, tmp_path):
    started = []
    r_result = _result("R")
    r_result["ratios"]["incumbent/baseline"]["total_ns"]["ci_high"] = 1.01
    n_result = _result("N")
    q3b = type("Q3b", (), {
        "_post_environment": staticmethod(lambda *_args: {}),
        "_stage_gate": staticmethod(lambda *_args, **_kwargs: {"passed": True}),
    })()
    monkeypatch.setattr(q3c, "_q3b_runtime", lambda: q3b)
    monkeypatch.setattr(q3c, "preflight", lambda **_kwargs: {"passed": True, "identity": {
        "model_id": q3c.MODEL_ID, "model_revision": q3c.EXPECTED_REVISION,
        "model_manifest_sha256": q3c.EXPECTED_MODEL_MANIFEST_SHA256}, "runtime_code_sha256": q3c.runtime_code_sha256(),
        "environment": {"swap_used_bytes": 1000}, "installed_memory_bytes": 1000})
    monkeypatch.setattr(q3c, "_start_phase", lambda phase, *_args: started.append(phase) or (r_result if phase == "R" else n_result))
    monkeypatch.setattr(q3c, "validate_phase_result", lambda *_args: (True, "ok"))
    output = tmp_path / "criteria.json"
    assert q3c.main(["--execute", "--output", str(output)]) == 2
    assert started == ["R", "N"]
    record = json.loads(output.read_text())
    assert record["phase_gates"][0]["performance"]["passed"] is False
    assert record["status"] == "CRITERIA_MISS"
    assert record["decision"]["phase_N_preserved"] is True


def test_phase_n_time_boundary_is_explicit_failure_evidence(monkeypatch, tmp_path):
    r_result = _result("R")
    policy = type("Q3b", (), {
        "_post_environment": staticmethod(lambda *_args: {}),
        "_stage_gate": staticmethod(lambda *_args, **_kwargs: {"passed": True}),
    })()
    monkeypatch.setattr(q3c, "_q3b_runtime", lambda: policy)
    monkeypatch.setattr(q3c, "preflight", lambda **_kwargs: {"passed": True, "identity": {
        "model_id": q3c.MODEL_ID, "model_revision": q3c.EXPECTED_REVISION,
        "model_manifest_sha256": q3c.EXPECTED_MODEL_MANIFEST_SHA256}, "runtime_code_sha256": q3c.runtime_code_sha256(),
        "environment": {"swap_used_bytes": 1000}, "installed_memory_bytes": 1000})
    monkeypatch.setattr(q3c, "_start_phase", lambda phase, *_args: r_result)
    monkeypatch.setattr(q3c, "validate_phase_result", lambda *_args: (True, "ok"))
    clock = iter((0.0, 0.0, q3c.PHASE_MAX_SECONDS + 1.0))
    monkeypatch.setattr(q3c.time, "monotonic", lambda: next(clock))
    output = tmp_path / "n-not-started.json"
    assert q3c.main(["--execute", "--output", str(output)]) == 2
    record = json.loads(output.read_text())
    assert record["phases"][1]["phase"] == "N"
    assert record["phases"][1]["failure_evidence"]["reason"] == "phase deadline exhausted before worker start"
