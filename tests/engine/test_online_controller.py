"""CPU-only behavioral checks for executed learning and independent qualification."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import threading

import pytest

from ironmule_controller import (
    ControllerConfig, ControllerContext, ExecutionOutcome, OnlineController,
)

ROOT = Path(__file__).resolve().parents[2]
IDENTITY = {"backend": "test-cpu.v1", "model": "numeric.v1", "plan": "strict.v1"}


@pytest.fixture(scope="module")
def native_library(tmp_path_factory):
    cargo = shutil.which("cargo")
    if cargo is None:
        candidate = Path.home() / ".cargo" / "bin" / "cargo"
        cargo = str(candidate) if candidate.is_file() else None
    if cargo is None:
        pytest.skip("Rust toolchain unavailable; native controller not tested")
    target = tmp_path_factory.mktemp("native-controller-build")
    build = subprocess.run([cargo, "build", "--offline", "--release", "--manifest-path",
                            str(ROOT / "native" / "online_controller" / "Cargo.toml"),
                            "--target-dir", str(target)], capture_output=True, text=True, timeout=120)
    assert build.returncode == 0, build.stdout + build.stderr
    extension = ".dylib" if sys.platform == "darwin" else ".dll" if os.name == "nt" else ".so"
    libraries = list((target / "release").glob("*" + extension))
    assert len(libraries) == 1
    return libraries[0]


def config(**overrides):
    return ControllerConfig(min_train=2, freeze_every=4, min_pairs=64, max_trials=8,
                            exploration_ppm=50_000, trial_budget=6, **overrides)


def context(**overrides):
    return ControllerContext(identity=IDENTITY, request_count=4, max_tokens=8,
                             workload={"phase": "test"}, **overrides)


def outcome(action, *, cost=None, digest="same-output", resource=True, fallbacks=0):
    value = {"executed_action": action, "tokens": (11, 12, 99), "stop": "eos"}
    elapsed = (1.0 if action == 0 else 0.1) if cost is None else cost
    return ExecutionOutcome(value=value, signature=digest, elapsed_s=elapsed,
                            ttft_s=(elapsed / 2,) * 4, latencies_s=(elapsed,) * 4,
                            generated_tokens=12, fallback_count=fallbacks, resource_ok=resource)


def teach(controller, execute, calls=12000, context_factory=context):
    for index in range(calls):
        controller.run(context_factory(), execute)
        if index % 16 == 15:
            assert controller.flush()
        if controller.snapshot().get("promoted", 0):
            break
    assert controller.flush()
    return controller.snapshot()


def test_real_executions_train_freeze_and_promote_automatically(native_library, tmp_path):
    observed = []
    with OnlineController(IDENTITY, library_path=native_library,
                          checkpoint_path=tmp_path / "state.json", config=config()) as controller:
        initial = controller.snapshot()
        def execute(action):
            observed.append(action)
            return outcome(action)
        final = teach(controller, execute)
        assert set(observed) == {0, 1}, "alternative labels require alternative execution"
        assert final["updates"] > initial["updates"]
        assert final["promoted"] >= 1
        assert final["active_version"] > initial["active_version"]
        assert 1 in final["active_actions"]
        assert final["service"]["comparisons"] >= 64
        assert final["updates"] < len(observed), "holdout executions cannot train their candidate"
        assert controller.run(context(), execute)["tokens"] == (11, 12, 99)


def test_mismatched_holdout_output_faults_path_and_cannot_promote(native_library, tmp_path):
    with OnlineController(IDENTITY, library_path=native_library,
                          checkpoint_path=tmp_path / "state.json", config=config()) as controller:
        final = teach(controller, lambda action: outcome(action, digest=f"different-{action}"), calls=1000)
        assert final["promoted"] == 0
        assert final["fault_mask"] & 2
        assert all(action == 0 for action in final["active_actions"])


def test_slower_candidate_never_qualifies(native_library):
    with OnlineController(IDENTITY, library_path=native_library, config=config()) as controller:
        final = teach(controller, lambda action: outcome(action, cost=1.0 + action), calls=2000)
        assert final["promoted"] == 0
        assert final["updates"] > 0
        assert all(action == 0 for action in final["active_actions"])


def test_separately_measured_losing_candidate_is_rejected(native_library):
    with OnlineController(IDENTITY, library_path=native_library, config=config()) as controller:
        def execute(action):
            # Ordinary executed training data initially favors action one. The
            # independent frozen comparison then observes its changed cost.
            holdout = controller.last_decision.get("comparison", False)
            result = outcome(action, cost=1.1 if action and holdout else 0.1 if action else 1.0)
            return replace(result, ttft_s=(0.5,) * 4, latencies_s=(1.0,) * 4) if holdout else result
        final = teach(controller, execute)
        assert final["promoted"] == 0
        assert final["rejected"] > 0
        assert all(action == 0 for action in final["active_actions"])


def test_restart_preserves_updates_and_persistent_operator_kill(native_library, tmp_path):
    checkpoint = tmp_path / "state.json"
    controller = OnlineController(IDENTITY, library_path=native_library,
                                  checkpoint_path=checkpoint, config=config())
    for _ in range(40):
        controller.run(context(), outcome)
    assert controller.flush()
    count = controller.snapshot()["updates"]
    controller.kill()
    controller.close()
    with OnlineController(IDENTITY, library_path=native_library,
                          checkpoint_path=checkpoint, config=config()) as restored:
        final = restored.snapshot()
        assert final["updates"] == count
        assert final["killed"]
        assert restored.run(context(), outcome)["executed_action"] == 0
        assert restored.snapshot()["updates"] == count


@pytest.mark.parametrize("corruption", ["digest", "identity", "syntax", "non_mapping"])
def test_invalid_checkpoint_fails_closed(native_library, tmp_path, corruption):
    checkpoint = tmp_path / "state.json"
    with OnlineController(IDENTITY, library_path=native_library,
                          checkpoint_path=checkpoint, config=config()) as controller:
        controller.run(context(), outcome)
        controller.flush()
        controller.save()
    payload = json.loads(checkpoint.read_text())
    if corruption == "non_mapping":
        checkpoint.write_text("[]")
    elif corruption == "syntax":
        checkpoint.write_text("{broken")
    else:
        payload["sha256" if corruption == "digest" else "identity"] = "foreign"
        checkpoint.write_text(json.dumps(payload))
    with OnlineController(IDENTITY, library_path=native_library,
                          checkpoint_path=checkpoint, config=config()) as restored:
        final = restored.snapshot()
        assert final["promoted"] == 0
        assert final["updates"] == 0
        assert restored.run(context(), outcome)["executed_action"] == 0
        assert final["last_error"]


def test_frozen_evaluation_has_no_updates_trials_or_exploration(native_library):
    with OnlineController(IDENTITY, library_path=native_library,
                          config=replace(config(), frozen=True)) as controller:
        before = controller.snapshot()
        actions = []
        for _ in range(100):
            value = controller.run(context(), lambda action: (actions.append(action), outcome(action))[1])
            assert value["executed_action"] == 0
        controller.flush()
        after = controller.snapshot()
        assert actions == [0] * 100
        assert after["updates"] == before["updates"]
        assert after["trials_started"] == before["trials_started"]


def test_ineligible_grouped_profile_cannot_execute(native_library):
    with OnlineController(IDENTITY, library_path=native_library, config=config()) as controller:
        actions = []
        for _ in range(100):
            controller.run(context(eligible_mask=1), lambda action: (actions.append(action), outcome(action))[1])
        controller.flush()
        assert actions == [0] * 100


def test_resource_failure_cannot_qualify_faster_execution(native_library):
    with OnlineController(IDENTITY, library_path=native_library, config=config()) as controller:
        final = teach(controller, lambda action: outcome(action, resource=action == 0), calls=1000)
        assert final["promoted"] == 0
        assert all(action == 0 for action in final["active_actions"])


def test_foreign_context_is_reference_only(native_library):
    with OnlineController(IDENTITY, library_path=native_library, config=config()) as controller:
        foreign = ControllerContext(identity={**IDENTITY, "backend": "different"},
                                    request_count=4, max_tokens=8, workload={})
        before = controller.snapshot()["updates"]
        for _ in range(50):
            assert controller.run(foreign, outcome)["executed_action"] == 0
        controller.flush()
        assert controller.snapshot()["updates"] == before


def test_full_learning_queue_preserves_complete_service_accounting(native_library, tmp_path):
    # A deliberately paused consumer fills the actual bounded queue. The user
    # work still executes and its counters remain independent of retained labels.
    with OnlineController(IDENTITY, library_path=native_library,
                          checkpoint_path=tmp_path / "paused.json",
                          config=config(queue_capacity=1), start_worker=False) as controller:
        for _ in range(10):
            assert controller.run(context(), outcome)["tokens"] == (11, 12, 99)
        status = controller.snapshot()
        assert status["learning_queue"] == 1
        assert status["service"]["groups"] == 10
        assert status["service"]["completed_requests"] == 40
        assert status["service"]["learning_dropped"] >= 9
        assert status["service"]["pending_groups"] == 0


def test_drift_withdraws_policy_without_clearing_persistent_kill(native_library, tmp_path):
    checkpoint = tmp_path / "drift.json"
    with OnlineController(IDENTITY, library_path=native_library,
                          checkpoint_path=checkpoint, config=config()) as controller:
        trained = teach(controller, outcome)
        assert trained["promoted"]
        controller.fault(1, kind=1)
        withdrawn = controller.snapshot()
        assert all(action == 0 for action in withdrawn["active_actions"])
        controller.kill()
        controller.fault(1, kind=1)
        assert controller.snapshot()["killed"]
    with OnlineController(IDENTITY, library_path=native_library,
                          checkpoint_path=checkpoint, config=config()) as restored:
        assert restored.snapshot()["killed"]
        assert restored.run(context(), outcome)["executed_action"] == 0


def test_baseline_exception_records_failure_without_correctness_fault(native_library):
    with OnlineController(IDENTITY, library_path=native_library, config=config()) as controller:
        def failed(_action):
            raise RuntimeError("controlled backend failure")
        with pytest.raises(RuntimeError, match="controlled backend failure"):
            controller.run(context(), failed)
        assert controller.flush()
        status = controller.snapshot()
        assert status["fault_mask"] == 0
        assert status["service"]["errors"] == 1
        assert status["service"]["pending_groups"] == 0


def test_run_after_close_rejects_before_execution(native_library):
    controller = OnlineController(IDENTITY, library_path=native_library, config=config())
    controller.close()
    calls = []
    with pytest.raises(RuntimeError, match="closed"):
        controller.run(context(), lambda action: (calls.append(action), outcome(action))[1])
    assert calls == []


def test_mixed_recovery_is_observed_but_never_used_as_pure_training(native_library):
    with OnlineController(IDENTITY, library_path=native_library, config=config()) as controller:
        before = controller.snapshot()["updates"]
        controller.run(context(), lambda action: outcome(action, fallbacks=1))
        assert controller.flush()
        status = controller.snapshot()
        assert status["updates"] == before
        assert status["service"]["completed_requests"] == 4
        assert status["last_decision"]["actual_action"] == "mixed_recovery"
        assert status["last_decision"]["propensity"] is None


def test_performance_limit_failure_is_not_correctness_fault(native_library):
    with OnlineController(IDENTITY, library_path=native_library, config=config()) as controller:
        controller.run(context(), lambda action: outcome(action, cost=60.0))
        assert controller.flush()
        status = controller.snapshot()
        assert status["fault_mask"] == 0
        assert status["promoted"] == 0


def test_periodic_checkpoint_rate_limit_preserves_explicit_save_and_kill(native_library, tmp_path):
    checkpoint = tmp_path / "rate-limited.json"
    settings = config(checkpoint_every=1, checkpoint_interval_s=3600)
    with OnlineController(IDENTITY, library_path=native_library,
                          checkpoint_path=checkpoint, config=settings) as controller:
        for _ in range(5):
            controller.run(context(), outcome)
            assert controller.flush()
        assert controller.snapshot()["service"]["checkpoints"] == 0
        updates = controller.snapshot()["updates"]
        assert controller.save() == checkpoint
        assert controller.snapshot()["service"]["checkpoints"] == 1
        controller.kill()
        assert controller.snapshot()["service"]["checkpoints"] == 2
    with OnlineController(IDENTITY, library_path=native_library,
                          checkpoint_path=checkpoint, config=settings) as restored:
        assert restored.snapshot()["killed"]
        assert restored.snapshot()["updates"] == updates


def four_action_contract():
    from ironmule_product.engine_bridge import serving_profile_contract
    return serving_profile_contract({"compiled_fixed_cache": False,
                                     "head_skip_prefill": False, "readback_every": 1})


def checked_controller(native_library, **kwargs):
    return OnlineController(IDENTITY, library_path=native_library,
                            config=config(verify_exploration=True, seed=15),
                            action_contract=four_action_contract(), **kwargs)


def accumulate_reference_credit(controller):
    for _ in range(19):
        controller.run(context(eligible_mask=1), outcome)
        assert controller.flush()


def test_four_registered_actions_verify_before_delivering_reference(native_library):
    with checked_controller(native_library) as controller:
        accumulate_reference_credit(controller)
        calls = []
        selected = controller.run(context(eligible_mask=15),
                                  lambda action: (calls.append(action), outcome(action))[1])
        assert controller.flush()
        decision = controller.last_decision
        assert decision["kind"] == 2
        assert decision["proposed_action"] in (1, 2, 3)
        assert calls == [0, decision["proposed_action"]]
        assert selected["executed_action"] == 0
        assert decision["returned_action"] == decision["actual_action"] == 0
        assert decision["observation_action"] == decision["proposed_action"]
        assert decision["correctness"] == "matched"
        assert controller.snapshot()["service"]["exploration_checks"] == 1
        assert controller.snapshot()["promoted"] == 0


def test_four_action_mismatch_quarantines_all_alternatives_and_persists(native_library, tmp_path):
    state = tmp_path / "four-actions.json"
    with checked_controller(native_library, checkpoint_path=state) as controller:
        accumulate_reference_credit(controller)
        result = controller.run(context(eligible_mask=15),
                                lambda action: outcome(action, digest=f"state-{action}"))
        assert controller.flush()
        assert result["executed_action"] == 0
        assert controller.snapshot()["fault_mask"] == 0b1110
        assert controller.last_decision["observation_action"] is None
        controller.save()
    with checked_controller(native_library, checkpoint_path=state) as restored:
        assert restored.snapshot()["fault_mask"] == 0b1110
        assert restored.run(context(eligible_mask=15), outcome)["executed_action"] == 0


def test_comparison_between_optimized_actions_recovers_fresh_action_zero(native_library):
    import ironmule_controller as bridge
    with checked_controller(native_library) as controller:
        # Controlled ABI tickets exercise mismatch recovery for active/candidate
        # IDs two and three without inventing comparative training evidence.
        a = bridge._Decision(0, 1, 2, 2, 3, 0, 4, 1.0)
        aa = bridge._Decision(0, 1, 2, 2, 4, 0, 4, 1.0)
        b = bridge._Decision(0, 1, 2, 3, 5, 0, 4, 1.0)
        controller.last_decision = {"propensity": 1.0}
        calls = []
        import time
        result = controller._compare(context(eligible_mask=15),
                                     lambda action: (calls.append(action), outcome(action, digest=f"state-{action}"))[1],
                                     ((a, aa, b), 0), time.perf_counter_ns())
        assert calls == [2, 2, 3, 0]
        assert result.value["executed_action"] == 0
        assert controller.last_decision["returned_action"] == 0
        assert controller.snapshot()["fault_mask"] == 0b1110
        assert controller.snapshot()["updates"] == 0


@pytest.mark.parametrize("protocol", ["strict_v1", "adverse_control_v2", "sequential_v3", "sequential_noise_v4"])
@pytest.mark.parametrize("failure", ["relative_ttft", "relative_latency", "output", "caller", "absolute_ttft", "resource"])
def test_comparison_audit_distinguishes_output_and_timing_gates(native_library, monkeypatch, failure, protocol):
    import ironmule_controller as bridge
    with OnlineController(IDENTITY, library_path=native_library,
                          config=config(verify_exploration=True, seed=15, qualification_protocol=protocol),
                          action_contract=four_action_contract()) as controller:
        # Captured preparation is not submitted to native code as fabricated tickets.
        prepared = []
        monkeypatch.setattr(controller, "_enqueue", lambda items: prepared.extend(items))
        decisions = tuple(bridge._Decision(100 + i, 1, 2, action, 3 + i, 0, 4, 1.0)
                          for i, action in enumerate((0, 0, 2)))
        baseline = outcome(0, cost=0.2)
        candidate = outcome(2, cost=0.1)
        if failure == "relative_ttft":
            candidate = replace(candidate, ttft_s=(0.11,) * 4)
        elif failure == "relative_latency":
            candidate = replace(candidate, latencies_s=(0.22,) * 4)
        elif failure == "absolute_ttft":
            candidate = replace(candidate, ttft_s=(2.1,) * 4)
        elif failure == "output":
            candidate = replace(candidate, signature="different-state")
        elif failure == "resource":
            candidate = replace(candidate, resource_ok=False)
        controller.last_decision = {"propensity": 1.0}
        started = time.perf_counter_ns() - (3_000_000_000 if failure == "caller" else 0)
        result = controller._compare(context(eligible_mask=15),
                                     lambda action: candidate if action == 2 else baseline,
                                     (decisions, 1), started)
        audit = controller.last_decision
        assert audit["qualification_protocol"] == protocol
        assert audit["comparison_score_semantics"]["protocol"] == protocol
        assert result is baseline
        assert audit["comparison_order"] == ["B", "AA", "A"]
        assert audit["comparison_caller_slo_ok"] == (failure != "caller")
        assert audit["comparison_caller_slo_check_s"] >= (3 if failure == "caller" else 0)
        assert audit["comparison_relative_ttft_ok"] == (None if failure == "output" else failure not in ("relative_ttft", "absolute_ttft"))
        assert audit["comparison_relative_latency_ok"] == (None if failure == "output" else failure != "relative_latency")
        arms = audit["comparison_arms"]
        assert [arm["role"] for arm in arms] == ["A", "AA", "B"]
        b = arms[2]
        assert b["elapsed_s"] == candidate.elapsed_s
        assert b["ttft_s"] == list(candidate.ttft_s)
        assert b["latencies_s"] == list(candidate.latencies_s)
        assert b["status"] == "ok"
        assert b["signature_digest"] == controller._signature(candidate)
        assert b["signature_matched"] == (failure != "output")
        assert b["resource_ok_before_gates"] == (failure != "resource")
        assert b["resource_ok_after_gates"] == (failure == "output")
        assert b["resource_ok_for_completion"] == (failure == "output")
        assert b["prepared_for_learning"] is True
        assert b["prepared_completion"] == {name: getattr(prepared[2], name) for name, _ in prepared[2]._fields_}
        assert b["prepared_completion"]["status"] == (4 if failure == "output" else 0)
        assert b["prepared_completion"]["correctness"] == (0 if failure == "output" else 1)
        assert b["prepared_completion"]["cost"] == controller.config.latency_limit_s
        assert controller.snapshot()["updates"] == 0


def test_comparison_audit_bounds_vectors_and_retains_partial_interruption(native_library, monkeypatch):
    import ironmule_controller as bridge
    with checked_controller(native_library) as controller:
        prepared = []
        monkeypatch.setattr(controller, "_enqueue", lambda items: prepared.extend(items))
        decisions = tuple(bridge._Decision(100 + i, 1, 2, action, 3 + i, 0, 4, 1.0)
                          for i, action in enumerate((0, 0, 2)))
        large = replace(outcome(0), ttft_s=(float("nan"),) + (0.1,) * 39, latencies_s=(0.2,) * 40)
        calls = []
        def execute(action):
            calls.append(action)
            if len(calls) == 2:
                raise KeyboardInterrupt
            return large
        controller.last_decision = {"propensity": 1.0}
        with pytest.raises(KeyboardInterrupt):
            controller._compare(context(eligible_mask=15), execute, (decisions, 0), time.perf_counter_ns())
        audit = controller.last_decision
        assert audit["comparison_order"] == ["A", "AA"]
        assert audit["comparison_caller_slo_check_s"] is None
        a, aa, b = audit["comparison_arms"]
        assert a["ttft_s_count"] == 40 and a["ttft_s_truncated"] is True
        assert len(a["ttft_s"]) == 32 and a["ttft_s"][0] is None
        assert a["latencies_s_count"] == 40 and len(a["latencies_s"]) == 32
        assert a["signature_matched"] is None
        assert aa["elapsed_s"] is None and b["ttft_s_count"] is None
        assert all(arm["prepared_completion"]["status"] == 4 for arm in (a, aa, b))
        json.dumps(audit, allow_nan=False)


def test_explicit_offline_budget_qualifies_with_real_callbacks_and_expires_before_restart(native_library, tmp_path):
    checkpoint = tmp_path / "offline.json"
    calls = []
    def create():
        return OnlineController(IDENTITY, library_path=native_library, checkpoint_path=checkpoint,
                                config=ControllerConfig(verify_exploration=True, seed=15),
                                action_contract=four_action_contract())
    with create() as controller:
        assert controller.config.min_train == 8 and controller.config.min_pairs == 64
        assert controller.snapshot()["offline_training"] == dict(granted=0, used=0, remaining=0, active=0)
        with controller.offline_training(max_extra_comparison_executions=128):
            assert controller.snapshot()["offline_training"] == dict(granted=128, used=0, remaining=128, active=1)
            for _ in range(3000):
                controller.run(context(eligible_mask=5), lambda action: (calls.append(action), outcome(action))[1])
                assert controller.flush()
                state = controller.snapshot()
                if controller.last_decision.get("comparison"):
                    assert controller.last_decision["comparison_funding"] == "offline_training"
                    assert len(controller.last_decision["comparison_arms"]) == 3
                if state["promoted"]:
                    break
            assert state["promoted"] == 1
            assert state["offline_training"]["used"] == 128
            assert state["offline_training"]["remaining"] == 0
            assert state["offline_training"]["active"] == 0
            assert controller.run(context(eligible_mask=5), outcome)["executed_action"] == 2
            assert controller.last_decision["kind"] == 1
            assert controller.flush()
        assert controller.snapshot()["offline_training"] == dict(granted=128, used=128, remaining=0, active=0)
        with pytest.raises(RuntimeError, match="grant rejected"):
            with controller.offline_training(max_extra_comparison_executions=2):
                pass
        controller.save()
    with create() as restored:
        state = restored.snapshot()
        assert state["offline_training"] == dict(granted=0, used=0, remaining=0, active=0)
        assert state["promoted"] == 1 and 2 in state["active_actions"]
        restored.kill()
    with create() as killed:
        assert killed.snapshot()["killed"] == 1
        with pytest.raises(RuntimeError, match="live|killed"):
            with killed.offline_training(max_extra_comparison_executions=2):
                pass
    assert set(calls) == {0, 2}


def test_offline_budget_is_thread_scoped_and_revoked_on_exception(native_library, monkeypatch):
    with checked_controller(native_library) as controller:
        with pytest.raises(ValueError, match="deliberate"):
            with controller.offline_training(max_extra_comparison_executions=8):
                monkeypatch.setattr(controller, "_status_unlocked", lambda: {
                    "candidate_version": 1, "credits": 2, "last_pair_id": 0, "killed": 0})
                calls = []
                monkeypatch.setattr(controller._lib, "imc_begin_training_pair", lambda *_args: (calls.append("offline"), 3)[1])
                monkeypatch.setattr(controller._lib, "imc_begin_pair", lambda *_args: (calls.append("live"), 3)[1])
                native_context = controller._native_context(context(eligible_mask=5))
                controller._choose(native_context)
                thread = threading.Thread(target=lambda: controller._choose(native_context))
                thread.start()
                thread.join(timeout=5)
                assert not thread.is_alive()
                assert calls == ["offline", "live"]
                raise ValueError("deliberate")
        assert controller.snapshot()["offline_training"] == dict(granted=8, used=0, remaining=0, active=0)


@pytest.mark.parametrize("budget", [True, 0, 1, 3, 32770, 2.0])
def test_offline_budget_rejects_invalid_grants_without_spending_authority(native_library, budget):
    with checked_controller(native_library) as controller:
        with pytest.raises(ValueError, match="even integer"):
            with controller.offline_training(max_extra_comparison_executions=budget):
                pass
        assert controller.snapshot()["offline_training"]["granted"] == 0


def test_offline_context_rejects_frozen_closed_and_quarantined_controllers(native_library):
    with OnlineController(IDENTITY, library_path=native_library, config=config(frozen=True)) as frozen:
        with pytest.raises(RuntimeError, match="live"):
            with frozen.offline_training(max_extra_comparison_executions=2):
                pass
    controller = checked_controller(native_library)
    controller._quarantined = True
    with pytest.raises(RuntimeError, match="live"):
        with controller.offline_training(max_extra_comparison_executions=2):
            pass
    controller._quarantined = False
    controller.close()
    with pytest.raises(RuntimeError, match="live"):
        with controller.offline_training(max_extra_comparison_executions=2):
            pass


def test_library_without_optional_training_exports_preserves_default_operation(native_library, monkeypatch):
    import ironmule_controller as bridge
    original = bridge.C.PyDLL
    class LegacyLibrary:
        def __init__(self, path):
            self.lib = original(path)
        def __getattr__(self, name):
            if name in {"imc_training_begin", "imc_training_end", "imc_training_status", "imc_begin_training_pair"}:
                raise AttributeError(name)
            return getattr(self.lib, name)
    monkeypatch.setattr(bridge.C, "PyDLL", LegacyLibrary)
    with OnlineController(IDENTITY, library_path=native_library, config=config()) as controller:
        assert controller.run(context(eligible_mask=1), outcome)["executed_action"] == 0
        assert controller.flush()
        assert controller.snapshot()["updates"] == 1
        assert controller.snapshot()["offline_training"] is None
        with pytest.raises(RuntimeError, match="does not support"):
            with controller.offline_training(max_extra_comparison_executions=2):
                pass


def test_sequential_protocol_promotes_a_clear_winner_after_eleven_windows(native_library, tmp_path):
    checkpoint = tmp_path / "sequential.json"
    settings = ControllerConfig(verify_exploration=True, seed=15, qualification_protocol="sequential_v3")
    assert settings.objective()["schema"] == "ironmule.complete_group_objective.v3"
    assert settings.objective()["qualification"]["max_samples"] == 64
    windows = 0
    with OnlineController(IDENTITY, library_path=native_library, checkpoint_path=checkpoint, config=settings,
                          action_contract=four_action_contract()) as controller:
        with controller.offline_training(max_extra_comparison_executions=128):
            for _ in range(3000):
                controller.run(context(eligible_mask=5), outcome)
                assert controller.flush()
                windows += bool(controller.last_decision.get("comparison"))
                state = controller.snapshot()
                if state["promoted"]:
                    break
        assert state["promoted"] == 1 and 2 in state["active_actions"]
        assert windows == 11 and state["offline_training"]["used"] == 22
        assert controller._envelope_binding()["schema"] == "ironmule.online_controller.v3"
    with OnlineController(IDENTITY, library_path=native_library, checkpoint_path=checkpoint, config=settings,
                          action_contract=four_action_contract()) as restored:
        assert restored.snapshot()["checkpoint_rejected"] is False
        assert 2 in restored.snapshot()["active_actions"]


def _calls_until_promotion(native_library, directed):
    settings = ControllerConfig(verify_exploration=True, seed=15, qualification_protocol="sequential_v3",
                                directed_training=directed)
    with OnlineController(IDENTITY, library_path=native_library, config=settings,
                          action_contract=four_action_contract()) as controller:
        with controller.offline_training(max_extra_comparison_executions=256):
            for calls in range(1, 3001):
                controller.run(context(eligible_mask=5), outcome)
                assert controller.flush()
                if controller.last_decision.get("comparison_funding") == "offline_training_directed_label":
                    assert controller.last_decision["returned_action"] == 0  # checked, reference delivered
                if controller.snapshot()["promoted"]:
                    return calls, controller.snapshot()
    raise AssertionError("no promotion")


def test_directed_training_reaches_promotion_in_far_fewer_calls(native_library):
    random_calls, _ = _calls_until_promotion(native_library, False)
    directed_calls, state = _calls_until_promotion(native_library, True)
    assert directed_calls * 2 < random_calls, (directed_calls, random_calls)
    assert 2 in state["active_actions"]
    assert ControllerConfig(directed_training=True) != ControllerConfig()
    with pytest.raises(ValueError, match="booleans"):
        ControllerConfig(directed_training=1)


def test_comparison_floor_repeats_arms_and_still_promotes(native_library):
    calls = []
    settings = ControllerConfig(verify_exploration=True, seed=15, qualification_protocol="sequential_v3",
                                directed_training=True, comparison_floor_s=0.35)
    with OnlineController(IDENTITY, library_path=native_library, config=settings,
                          action_contract=four_action_contract()) as controller:
        with controller.offline_training(max_extra_comparison_executions=256):
            for _ in range(3000):
                before = len(calls)
                controller.run(context(eligible_mask=5),
                               lambda action: (calls.append(action), outcome(action, cost=0.1 if action else 0.12))[1])
                assert controller.flush()
                if controller.last_decision.get("comparison"):
                    reps = controller.last_decision["comparison_repetitions"]
                    assert reps == 3 and len(calls) - before == 3 * reps  # ceil(0.35 / 0.12)
                if controller.snapshot()["promoted"]:
                    break
        assert 2 in controller.snapshot()["active_actions"]
        assert controller._envelope_binding()["config"]["comparison_floor_s"] == 0.35
    with pytest.raises(ValueError, match="comparison_floor_s"):
        ControllerConfig(comparison_floor_s=-1)


def test_inconsistent_repetitions_break_the_signature(native_library):
    with OnlineController(IDENTITY, library_path=native_library, config=ControllerConfig()) as controller:
        digests = iter(("a", "b"))
        merged = controller._execute_repeated(lambda action: outcome(action, digest=next(digests)), 0, 2)
        assert merged.signature == {"inconsistent_repetitions": [controller._signature(outcome(0, digest="a")),
                                                                 controller._signature(outcome(0, digest="b"))]}
        same = controller._execute_repeated(lambda action: outcome(action, cost=0.2), 0, 3)
        assert same.elapsed_s == pytest.approx(0.6) and same.ttft_s == pytest.approx((0.1,) * 4)


@pytest.mark.parametrize("view,accepted", [("per_request", False), ("delivered", True)])
def test_delivered_view_judges_grouping_by_what_the_caller_receives(native_library, monkeypatch, view, accepted):
    import ironmule_controller as bridge
    settings = config(verify_exploration=True, seed=15, qualification_protocol="sequential_v3", latency_view=view)
    with OnlineController(IDENTITY, library_path=native_library, config=settings,
                          action_contract=four_action_contract()) as controller:
        prepared = []
        monkeypatch.setattr(controller, "_enqueue", lambda items: prepared.extend(items))
        decisions = tuple(bridge._Decision(100 + i, 1, 2, action, 3 + i, 0, 4, 1.0)
                          for i, action in enumerate((0, 0, 1)))
        sequential = replace(outcome(0, cost=1.0), latencies_s=(0.25, 0.5, 0.75, 1.0), ttft_s=(0.1, 0.35, 0.6, 0.85))
        grouped = replace(outcome(1, cost=0.8), latencies_s=(0.8,) * 4, ttft_s=(0.2,) * 4)  # all done near the end
        controller.last_decision = {"propensity": 1.0}
        controller._compare(context(), lambda action: sequential if action == 0 else grouped,
                            (decisions, 0), time.perf_counter_ns())
        assert bool(prepared[2].resource_ok) is accepted
    assert ("relative_gate_view" in ControllerConfig(latency_view="delivered").objective()) is True
    assert "relative_gate_view" not in ControllerConfig().objective()
    with pytest.raises(ValueError, match="latency_view"):
        ControllerConfig(latency_view="fast")


def test_directed_training_flag_off_keeps_legacy_envelope(native_library):
    with OnlineController(IDENTITY, library_path=native_library) as controller:
        assert "directed_training" not in controller._envelope_binding()["config"]
        assert "comparison_floor_s" not in controller._envelope_binding()["config"]
        assert "latency_view" not in controller._envelope_binding()["config"]


@pytest.mark.parametrize("protocol", [None, True, 2, "adverse", "strict_v2"])
def test_qualification_protocol_requires_explicit_supported_name(protocol):
    with pytest.raises(ValueError, match="qualification protocol"):
        ControllerConfig(qualification_protocol=protocol)


def test_strict_objective_and_default_envelope_preserve_legacy_binding(native_library):
    old_objective = {"schema": "ironmule.complete_group_objective.v1",
                     "cost": "elapsed_seconds / completed_requests; hard TTFT/latency gates",
                     "ttft_limit_s": 2.0, "latency_limit_s": 30.0,
                     "memory_limit_bytes": None, "min_gain": 0.05, "max_regression": 0.05}
    strict = ControllerConfig()
    assert strict.objective() == old_objective
    with OnlineController(IDENTITY, library_path=native_library, config=strict) as controller:
        binding = controller._envelope_binding()
        assert "qualification_protocol" not in binding["config"]
        assert "objective" not in binding
        assert binding["schema"] == "ironmule.online_controller.v1"
        assert controller.snapshot()["qualification_protocol"] == "strict_v1"
    adverse = ControllerConfig(qualification_protocol="adverse_control_v2")
    assert adverse.objective()["schema"] == "ironmule.complete_group_objective.v2"
    contract = adverse.objective()["qualification"]
    assert contract["fixed_samples"] == 64 and contract["unstable_ratio_score"] == 1.05
    assert "full raw" in contract["noise_score"]
    assert "nominal alpha" in contract["uncertainty_scope"]
    assert adverse.objective() != old_objective


@pytest.mark.parametrize("missing", [{"imc_create_with_protocol", "imc_restore_with_protocol"},
                                      {"imc_restore_with_protocol"}])
def test_protocol_exports_are_optional_for_strict_but_required_together_for_v2(native_library, monkeypatch, missing):
    import ironmule_controller as bridge
    original = bridge.C.PyDLL
    class LegacyLibrary:
        def __init__(self, path):
            self.lib = original(path)
        def __getattr__(self, name):
            if name in missing:
                raise AttributeError(name)
            return getattr(self.lib, name)
    monkeypatch.setattr(bridge.C, "PyDLL", LegacyLibrary)
    with OnlineController(IDENTITY, library_path=native_library) as strict:
        assert strict.run(context(eligible_mask=1), outcome)["executed_action"] == 0
        assert strict.flush()
        assert strict.snapshot()["updates"] == 1
    with pytest.raises(RuntimeError, match="does not support adverse_control_v2"):
        OnlineController(IDENTITY, library_path=native_library,
                         config=ControllerConfig(qualification_protocol="adverse_control_v2"))


@pytest.mark.parametrize("protocol,other,magic", [
    ("strict_v1", "adverse_control_v2", b"IMCSTATE01"),
    ("adverse_control_v2", "strict_v1", b"IMCSTATE02"),
    ("sequential_v3", "adverse_control_v2", b"IMCSTATE03"),
    ("sequential_noise_v4", "sequential_v3", b"IMCSTATE04"),
])
def test_protocol_checkpoint_preserves_same_mode_rejects_cross_mode_and_drops_offline_rights(
        native_library, tmp_path, protocol, other, magic):
    import base64
    checkpoint = tmp_path / "protocol.json"
    settings = ControllerConfig(qualification_protocol=protocol)
    with OnlineController(IDENTITY, library_path=native_library, checkpoint_path=checkpoint, config=settings) as controller:
        controller.run(context(eligible_mask=1), outcome)
        assert controller.flush()
        with controller.offline_training(max_extra_comparison_executions=2):
            assert controller.snapshot()["offline_training"]["active"] == 1
            controller.save()  # Even a checkpoint written during the grant cannot convey it.
            encoded = json.loads(checkpoint.read_text())
            assert base64.b64decode(encoded["payload"]).startswith(magic)
            assert "offline_training" not in encoded
            if protocol != "strict_v1":
                assert encoded["config"]["qualification_protocol"] == protocol
                assert encoded["objective"] == settings.objective()
        count = controller.snapshot()["updates"]
    with OnlineController(IDENTITY, library_path=native_library, checkpoint_path=checkpoint, config=settings) as restored:
        state = restored.snapshot()
        assert state["updates"] == count and state["checkpoint_rejected"] is False
        assert state["offline_training"] == dict(granted=0, used=0, remaining=0, active=0)
        assert restored.hardware_knowledge()["qualification_protocol"] == protocol
    original_bytes = checkpoint.read_bytes()
    with OnlineController(IDENTITY, library_path=native_library, checkpoint_path=checkpoint,
                          config=ControllerConfig(qualification_protocol=other)) as foreign:
        assert foreign.snapshot()["checkpoint_rejected"] is True
        assert foreign.run(context(), outcome)["executed_action"] == 0
        assert foreign.snapshot()["offline_training"] == dict(granted=0, used=0, remaining=0, active=0)
    assert checkpoint.read_bytes() == original_bytes


@pytest.mark.parametrize("protocol,expected_pairs", [("strict_v1", 1), ("adverse_control_v2", 64),
                                                     ("sequential_v3", 21)])
def test_noisy_controls_retain_raw_times_and_cannot_qualify_under_either_protocol(native_library, protocol, expected_pairs):
    settings = ControllerConfig(verify_exploration=True, seed=15, qualification_protocol=protocol)
    with OnlineController(IDENTITY, library_path=native_library, config=settings,
                          action_contract=four_action_contract()) as controller:
        reference_calls = [0]
        audits = []
        def execute(action):
            cost = None
            if controller.last_decision.get("comparison") and action == 0:
                cost = 1.0 if reference_calls[0] % 2 == 0 else 0.8
                reference_calls[0] += 1
            return outcome(action, cost=cost)
        with controller.offline_training(max_extra_comparison_executions=128):
            for _ in range(3000):
                controller.run(context(eligible_mask=5), execute)
                assert controller.flush()
                state = controller.snapshot()
                if controller.last_decision.get("comparison"):
                    audits.append(dict(controller.last_decision))
                    if len(audits) < expected_pairs:
                        assert state["rejected"] == 0 and state["pairs_completed"] == len(audits)
                if state["rejected"]:
                    break
            assert state["rejected"] == 1 and state["promoted"] == 0
            assert len(audits) == expected_pairs
            assert state["offline_training"]["used"] == 2 * expected_pairs
            assert state["invalid_trials"] == 0
            assert all(action == 0 for action in state["active_actions"])
            assert all(audit["correctness"] == "matched" for audit in audits)
            for audit in audits:
                arms = audit["comparison_arms"]
                assert sorted(arm["elapsed_s"] for arm in arms) == [0.1, 0.8, 1.0]
                assert all(arm["prepared_completion"]["resource_ok"] == 1 for arm in arms)
                assert audit["comparison_score_semantics"]["protocol"] == protocol
            if protocol != "strict_v1":
                assert state["mean_ratio"] == pytest.approx(1.05)
                assert state["aa_noise"] == pytest.approx(2 / 9)
                assert state["upper_bound"] == 1.0


def test_unavailable_alternate_signature_cannot_train_or_fault(native_library):
    with checked_controller(native_library) as controller:
        accumulate_reference_credit(controller)
        before = controller.snapshot()["updates"]
        result = controller.run(context(eligible_mask=15),
                                lambda action: outcome(action, digest=None if action else "reference"))
        assert controller.flush()
        assert result["executed_action"] == 0
        assert controller.snapshot()["updates"] == before
        assert controller.snapshot()["fault_mask"] == 0
        assert controller.last_decision["observation_action"] is None


def test_action_mask_cannot_address_unregistered_profiles(native_library):
    with checked_controller(native_library) as controller:
        for invalid in (16, 17, 31, 0, 2):
            assert controller.run(context(eligible_mask=invalid), outcome)["executed_action"] == 0
        assert controller.flush()
        assert controller.snapshot()["updates"] == 0


def test_changed_four_action_semantics_reject_saved_state(native_library, tmp_path):
    state = tmp_path / "profiles.json"
    with checked_controller(native_library, checkpoint_path=state) as controller:
        controller.save()
    changed = four_action_contract()
    changed["profiles"][2] = changed["definitions"][2]["profile_id"] = "changed.sequential.v2"
    with OnlineController(IDENTITY, library_path=native_library, checkpoint_path=state,
                          config=config(verify_exploration=True, seed=15), action_contract=changed) as restored:
        assert restored.snapshot()["checkpoint_rejected"]
        assert restored.run(context(eligible_mask=15), outcome)["executed_action"] == 0


def test_duplicate_effective_profile_definition_is_rejected(native_library):
    contract = four_action_contract()
    contract["definitions"][3]["knobs"] = dict(contract["definitions"][1]["knobs"])
    with pytest.raises(ValueError, match="duplicate effective"):
        OnlineController(IDENTITY, library_path=native_library,
                         config=config(verify_exploration=True), action_contract=contract)


@pytest.mark.parametrize("bad_reference", ["status", "resource"])
def test_checked_alternative_does_not_train_against_failed_reference(native_library, bad_reference):
    with checked_controller(native_library) as controller:
        accumulate_reference_credit(controller)
        before = controller.snapshot()["updates"]
        def execute(action):
            value = outcome(action)
            if action == 0:
                value = replace(value, status="error") if bad_reference == "status" else replace(value, resource_ok=False)
            return value
        controller.run(context(eligible_mask=15), execute)
        assert controller.flush()
        assert controller.snapshot()["updates"] == before
        assert controller.snapshot()["fault_mask"] == 0
        assert controller.last_decision["observation_action"] is None


def activated_four_action_controller(native_library):
    controller = checked_controller(native_library)
    def execute(action):
        return outcome(action, cost=0.05 if action == 2 else 1.0)
    final = teach(controller, execute, calls=8000, context_factory=lambda: context(eligible_mask=15))
    assert final["active_actions"][4] == 2
    return controller, execute


def test_activated_four_action_exception_returns_reference_once_and_withdraws(native_library):
    controller, execute = activated_four_action_controller(native_library)
    try:
        before = controller.snapshot()["updates"]
        calls = []
        def failed(action):
            calls.append(action)
            if action == 2:
                raise RuntimeError("controlled optimized failure")
            return execute(action)
        result = controller.run(context(eligible_mask=15), failed)
        assert controller.flush()
        assert calls == [2, 0]
        assert result["executed_action"] == 0
        assert controller.snapshot()["active_actions"][4] == 0
        assert controller.snapshot()["updates"] == before
        assert controller.snapshot()["fault_mask"] == 0
        assert controller.last_decision["recovery_reason"] == "optimized_execution_failed"
    finally:
        controller.close()


def test_activated_four_action_mixed_fallback_withdraws_without_training(native_library):
    controller, execute = activated_four_action_controller(native_library)
    try:
        before = controller.snapshot()["updates"]
        controller.run(context(eligible_mask=15), lambda action: replace(execute(action), fallback_count=1))
        assert controller.flush()
        assert controller.snapshot()["active_actions"][4] == 0
        assert controller.snapshot()["updates"] == before
        assert controller.snapshot()["fault_mask"] == 0
    finally:
        controller.close()


def test_late_old_epoch_fallback_does_not_withdraw_requalified_same_action(native_library):
    import ironmule_controller as bridge
    controller, execute = activated_four_action_controller(native_library)
    try:
        before = controller.snapshot()
        old = bridge._Decision(1, before["active_version"] - 1, 0, 2, 2, 0, 4, 0.01)
        # Only the epoch guard is under test; this controlled old ticket is not
        # enqueued as new evidence or counted as another completed execution.
        completion = controller._completion(old, replace(execute(2), fallback_count=1),
                                            context(eligible_mask=15))
        assert completion.status == 4
        after = controller.snapshot()
        assert after["active_version"] == before["active_version"]
        assert after["active_actions"][4] == 2
        assert after["updates"] == before["updates"]
    finally:
        controller.close()


@pytest.mark.parametrize("failure", ["status", "resource", "missing_latency"])
def test_failed_checked_candidate_has_no_training_label(native_library, failure):
    with checked_controller(native_library) as controller:
        accumulate_reference_credit(controller)
        before = controller.snapshot()["updates"]
        def execute(action):
            value = outcome(action)
            if action:
                if failure == "status":
                    value = replace(value, status="timeout")
                elif failure == "resource":
                    value = replace(value, resource_ok=False)
                else:
                    value = replace(value, latencies_s=())
            return value
        value = controller.run(context(eligible_mask=15), execute)
        assert controller.flush()
        assert value["executed_action"] == 0
        assert controller.snapshot()["updates"] == before
        assert controller.snapshot()["fault_mask"] == 0
        assert controller.last_decision["observation_action"] is None


def test_noise_protocol_contract_describes_raw_scoring():
    contract = ControllerConfig(qualification_protocol="sequential_noise_v4").objective()["qualification"]
    assert contract["unstable_ratio_score"] is None and "every window" in contract["ratio_score"]
    assert "e-process" in contract["sequential_test"]
    assert ControllerConfig(qualification_protocol="sequential_noise_v4").objective()["schema"].endswith(".v4")
