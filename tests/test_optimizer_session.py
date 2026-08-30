from __future__ import annotations

import tempfile
import unittest
import time
import os
from pathlib import Path

from friday_optimizer.readiness import HardwareLease, ProbeSnapshot, ReadinessPolicy
from friday_optimizer.profiles import AtomicProfileStore, OptimizerProfile
from friday_optimizer.session import (
    AdapterResult, InvalidTransition, PromotionAuthorization, SessionController,
    SessionError, SessionState, StageSpec, SubprocessStageRunner,
)


class Clock:
    def __init__(self): self.value = 0.0
    def __call__(self): return self.value
    def sleep(self, seconds): self.value += seconds


def good():
    return ProbeSnapshot(
        timestamp=0, ac_connected=True, low_power=False, swap_used_bytes=1,
        memory_available_bytes=10000, memory_total_bytes=20000, load_1m=.1,
        cpu_percent=1, workload_active=False, process_tree_readable=True,
    )


class Probe:
    def sample(self): return good()


class Lease:
    def __init__(self, fingerprint="hardware-v1"): self.held = False; self.fingerprint = fingerprint
    def acquire(self): self.held = True; return self
    def validate(self): return self.held
    def release(self): self.held = False


class StageGate:
    def __init__(self): self.used = set()
    def verify_and_consume_authorization(self, spec, session_id):
        key = (spec.stage, spec.authorization_nonce)
        if key in self.used:
            return False
        if spec.execute_authorized is not True or spec.authorization_session_id != session_id or not spec.authorization_tag:
            return False
        self.used.add(key)
        return True


class Adapter:
    test_only = True
    def __init__(self, test="qualified", canary="pass", profile_version=0, rollback="pass", deactivate="pass", events=None, activation="activated"): self.test_result = test; self.canary_result = canary; self.profile_version = profile_version; self.rollback_result = rollback; self.deactivate_result = deactivate; self.activation_result = activation; self.events = events if events is not None else []
    def calibrate(self, **_): return AdapterResult("ok")
    def test(self, **_): return AdapterResult(self.test_result, {"profile_id": "candidate", "fingerprint": "hardware-v1", "profile_version": self.profile_version})
    def activate(self, **_): self.events.append("runtime_activate"); return AdapterResult("activated")
    def canary(self, **_): return AdapterResult(self.canary_result)
    def rollback(self, **_): self.events.append("runtime_rollback"); return AdapterResult(self.rollback_result)
    def deactivate(self, **_): self.events.append("runtime_deactivate"); return AdapterResult(self.deactivate_result)


class SessionTests(unittest.TestCase):
    def make_with(self, adapter, *, contract, authorization):
        controller = self.make(adapter=adapter, auto=True)
        controller.profile_contract = contract
        controller.promotion_authorization = authorization
        return controller

    def make(self, adapter=None, auto=True):
        self.clock = Clock()
        adapter = adapter or Adapter()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        executable = Path(directory.name) / "helper"
        canary = getattr(adapter, "canary_result", "pass")
        rollback = getattr(adapter, "rollback_result", "pass")
        deactivate = getattr(adapter, "deactivate_result", "pass")
        activation = getattr(adapter, "activation_result", "activated")
        version = getattr(adapter, "profile_version", 0)
        missing = adapter.__class__.__name__ == "MissingVersion"
        event_file = Path(directory.name) / "events"
        executable.write_text(
            "#!/bin/sh\n"
            "stage=\"$1\"\necho \"$stage\" >> \"$4\"\n"
            "case \"$stage\" in\n"
            "calibrate) echo '{\"outcome\":\"ok\"}' ;;\n"
            f"test) echo '{{\"outcome\":\"qualified\",\"payload\":{{\"profile_id\":\"candidate\",\"fingerprint\":\"hardware-v1\"{'' if missing else f',\"profile_version\":{version}'}}}}}' ;;\n"
            f"activate) echo '{{\"outcome\":\"{activation}\"}}' ;;\n"
            f"canary) echo '{{\"outcome\":\"{canary}\"}}' ;;\n"
            f"rollback) echo '{{\"outcome\":\"{rollback}\"}}' ;;\n"
            f"deactivate) echo '{{\"outcome\":\"{deactivate}\"}}' ;;\n"
            "esac\n"
        )
        os.chmod(executable, 0o700)
        digest = SubprocessStageRunner._file_sha256(str(executable))
        runner = SubprocessStageRunner(allowlisted_executables={str(executable): digest}, clock=self.clock)
        specs = {name: StageSpec(str(executable), (name, "", str(version), str(event_file)), env={}, execute_authorized=True, stage=name, authorization_session_id="session", authorization_nonce=name, authorization_tag="test-tag") for name in ("calibrate", "test", "activate", "canary", "rollback", "deactivate")}
        gate = StageGate()
        controller = SessionController(
            probe=Probe(), lease=Lease(), stage_runner=runner, stage_specs=specs,
            readiness_policy=ReadinessPolicy(sample_interval_seconds=0),
            sleeper=self.clock.sleep, clock=self.clock, auto_activate=auto,
            stage_authorization_gate=gate,
        )
        controller._event_file = event_file
        return controller

    def test_explicit_start_and_duration_bounds(self):
        controller = self.make(auto=False)
        with self.assertRaises(SessionError): controller.request(5)
        with self.assertRaises(ValueError): controller.request(4, user_started=True)
        controller.request(5, user_started=True)
        self.assertEqual(controller.state, SessionState.REQUESTED)
        self.assertEqual(controller.deadline, 300)

    def test_qualified_can_wait_for_activation(self):
        controller = self.make(auto=False)
        self.assertEqual(controller.run(5, user_started=True), SessionState.ACTIVATION_PENDING)
        self.assertEqual([a.to_state for a in controller.transitions], [
            SessionState.REQUESTED, SessionState.WAITING, SessionState.CALIBRATING,
            SessionState.TESTING, SessionState.QUALIFIED, SessionState.ACTIVATION_PENDING,
        ])

    def test_waiting_polls_until_readiness_then_starts(self):
        class ProbeSequence:
            def __init__(self): self.calls = 0
            def sample(self):
                self.calls += 1
                return ProbeSnapshot(
                    timestamp=0, ac_connected=True, low_power=False, swap_used_bytes=1,
                    memory_available_bytes=10000, memory_total_bytes=20000,
                    load_1m=(2 if self.calls <= 3 else .1), cpu_percent=1,
                    workload_active=(self.calls <= 3), process_tree_readable=True,
                )
        controller = self.make(auto=False)
        controller.probe = ProbeSequence()
        self.assertEqual(controller.run(5, user_started=True), SessionState.ACTIVATION_PENDING)
        self.assertGreater(controller.probe.calls, 3)

    def test_promotion_requires_explicit_authorization(self):
        controller = self.make(auto=True)
        self.assertEqual(controller.run(5, user_started=True), SessionState.ACTIVATION_PENDING)
        self.assertFalse(any(isinstance(item, AdapterResult) for item in []))
        auth = PromotionAuthorization(True, "session", "nonce-session", "hardware-v1", 0, 1000)
        class Contract:
            def validate_activation(self, **_): return True
            def activate(self, **_): return True
            def rollback(self, **_): return True
        controller = self.make_with(Adapter(), contract=Contract(), authorization=auth)
        self.assertEqual(controller.run(5, user_started=True), SessionState.ACTIVE)

    def test_unverified_run_stage_adapter_is_rejected(self):
        class Unverified:
            def run_stage(self, *_args, **_kwargs): return AdapterResult("ok")
        with self.assertRaises(SessionError):
            SessionController(probe=Probe(), lease=Lease(), adapter=Unverified())

    def test_missing_profile_version_cannot_activate(self):
        class MissingVersion(Adapter):
            def test(self, **_): return AdapterResult("qualified", {"profile_id": "candidate", "fingerprint": "hardware-v1"})
        controller = self.make(adapter=MissingVersion(), auto=False)
        self.assertEqual(controller.run(5, user_started=True), SessionState.BASELINE)

    def test_real_profile_contract_rolls_back_after_canary_failure(self):
        self.clock = Clock()
        with tempfile.TemporaryDirectory() as directory:
            store = AtomicProfileStore(Path(directory) / "profiles.json")
            store.save(OptimizerProfile("base", "hardware-v1", "baseline", {}, True).with_hash(), baseline=True)
            store.save(OptimizerProfile("candidate", "hardware-v1", "candidate", {}, True).with_hash())
            version = store.load()["version"]
            auth = PromotionAuthorization(True, "session", "nonce-store", "hardware-v1", 0, 1000)
            controller = self.make_with(Adapter(canary="fail", profile_version=version), contract=store, authorization=auth)
            self.assertEqual(controller.run(5, user_started=True), SessionState.BASELINE)
            selected = store.select("hardware-v1")
            self.assertTrue(selected.rollback_latched)
            self.assertEqual(selected.profile.profile_id, "base")

    def test_rollback_cas_conflict_never_claims_baseline(self):
        class Conflict:
            def validate_activation(self, **_): return True
            def activate(self, **_): return True
            def current_version(self): return 2
            def rollback(self, **_): raise RuntimeError("CAS conflict")
        self.clock = Clock()
        controller = self.make_with(Adapter(canary="fail", profile_version=1), contract=Conflict(), authorization=PromotionAuthorization(True, "session", "nonce-conflict", "hardware-v1", 0, 1000))
        self.assertEqual(controller.run(5, user_started=True), SessionState.ACTIVATION_UNCERTAIN)
        self.assertTrue(controller.no_recommendation)

    def test_runtime_rollback_precedes_store_rollback_and_both_are_required(self):
        events = []
        class Contract:
            def validate_activation(self, **_): return True
            def activate(self, **_): events.append("store_activate"); return True
            def current_version(self): return 2
            def rollback(self, **_): events.append("store_rollback"); return True
        adapter = Adapter(canary="fail", profile_version=1, events=events)
        self.clock = Clock()
        controller = self.make_with(adapter, contract=Contract(), authorization=PromotionAuthorization(True, "session", "nonce-order", "hardware-v1", 0, 1000))
        self.assertEqual(controller.run(5, user_started=True), SessionState.BASELINE)
        self.assertIn("rollback", controller._event_file.read_text().splitlines())
        self.assertIn("store_rollback", events)

    def test_runtime_rollback_failure_never_claims_baseline(self):
        class Contract:
            def validate_activation(self, **_): return True
            def activate(self, **_): return True
            def current_version(self): return 2
            def rollback(self, **_): raise AssertionError("must not touch store")
        self.clock = Clock()
        controller = self.make_with(Adapter(canary="fail", rollback="fail", profile_version=1), contract=Contract(), authorization=PromotionAuthorization(True, "session", "nonce-runtime-fail", "hardware-v1", 0, 1000))
        self.assertEqual(controller.run(5, user_started=True), SessionState.ACTIVATION_UNCERTAIN)

    def test_store_activation_failure_rolls_runtime_back_first(self):
        events = []
        class Contract:
            def validate_activation(self, **_): return True
            def activate(self, **_): events.append("store_activate"); raise RuntimeError("store CAS")
            def rollback(self, **_): events.append("store_rollback"); return True
        self.clock = Clock()
        controller = self.make_with(Adapter(events=events, profile_version=1), contract=Contract(), authorization=PromotionAuthorization(True, "session", "nonce-store-fail", "hardware-v1", 0, 1000))
        self.assertEqual(controller.run(5, user_started=True), SessionState.BASELINE)
        self.assertIn("rollback", controller._event_file.read_text().splitlines())
        self.assertIn("store_rollback", events)

    def test_activation_error_still_attempts_runtime_cleanup(self):
        events = []
        class Contract:
            def validate_activation(self, **_): return True
            def activate(self, **_): return True
            def rollback(self, **_): events.append("store_rollback"); return True
        self.clock = Clock()
        controller = self.make_with(Adapter(activation="error", profile_version=1), contract=Contract(), authorization=PromotionAuthorization(True, "session", "nonce-activation-error", "hardware-v1", 0, 1000))
        self.assertEqual(controller.run(5, user_started=True), SessionState.BASELINE)
        self.assertIn("rollback", controller._event_file.read_text().splitlines())

    def test_subprocess_stage_is_hard_deadline_bounded(self):
        executable = "/usr/bin/python3"
        runner = SubprocessStageRunner(allowlisted_executables={executable: SubprocessStageRunner._file_sha256(executable)})
        result = runner.run(StageSpec(executable, ("-c", "import time; time.sleep(2)"), env={}, execute_authorized=True, authorization_session_id="s", authorization_nonce="n", authorization_tag="tag"), deadline=time.monotonic() + .05)
        self.assertEqual(result.outcome, "timeout")

    def test_stage_without_explicit_execution_authorization_is_blocked(self):
        executable = "/usr/bin/python3"
        runner = SubprocessStageRunner(allowlisted_executables={executable: SubprocessStageRunner._file_sha256(executable)})
        with self.assertRaises(SessionError):
            runner.run(StageSpec(executable, ("-c", "print('must-not-run')"), env={}), deadline=time.monotonic() + 5)

    def test_session_without_authorization_gate_blocks_before_runner(self):
        controller = self.make(auto=False)
        controller.stage_authorization_gate = None
        self.assertEqual(controller.run(5, user_started=True), SessionState.BASELINE)
        self.assertFalse(controller._event_file.exists())

    def test_subprocess_stage_kills_output_bomb(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = str(Path(directory) / "bomb")
            Path(executable).write_text("#!/bin/sh\nprintf '%*s' 1000000 x\n")
            os.chmod(executable, 0o700)
            runner = SubprocessStageRunner(allowlisted_executables={executable: SubprocessStageRunner._file_sha256(executable)}, max_output_bytes=1024)
            result = runner.run(StageSpec(executable, (), env={}, execute_authorized=True, authorization_session_id="s", authorization_nonce="n", authorization_tag="tag"), deadline=time.monotonic() + 5)
            self.assertEqual(result.outcome, "error")
            self.assertEqual(result.reason, "output_truncated")

    def test_foreign_verified_runner_is_rejected(self):
        class Foreign:
            verified = True
            def run(self, *_args, **_kwargs): return AdapterResult("ok")
        with self.assertRaises(SessionError):
            SessionController(probe=Probe(), lease=Lease(), stage_runner=Foreign())
        class Unverified:
            def run_stage(self, *_args, **_kwargs): return AdapterResult("ok")
        with self.assertRaises(SessionError):
            SessionController(probe=Probe(), lease=Lease(), adapter=Unverified())

    def test_release_error_is_recorded_without_overwriting_baseline(self):
        self.clock = Clock()
        class BrokenRelease(Lease):
            def release(self): self.held = False; raise RuntimeError("release")
        controller = self.make(adapter=Adapter(), auto=False)
        controller.lease = BrokenRelease()
        self.assertEqual(controller.run(5, user_started=True), SessionState.ACTIVATION_PENDING)
        self.assertEqual(controller.release_error, "RuntimeError")
        self.assertIn("lease_release_error:RuntimeError", controller.audit_errors)

    def test_foreign_load_mid_run_falls_back(self):
        class ChangingProbe:
            def __init__(self): self.count = 0
            def sample(self):
                self.count += 1
                return good() if self.count < 7 else ProbeSnapshot(
                    timestamp=0, ac_connected=True, low_power=False, swap_used_bytes=1,
                    memory_available_bytes=10000, memory_total_bytes=20000, load_1m=2,
                    cpu_percent=99, workload_active=True, process_tree_readable=True,
                )
        controller = self.make()
        controller.probe = ChangingProbe()
        self.assertEqual(controller.run(5, user_started=True), SessionState.BASELINE)
        self.assertTrue(controller.no_recommendation)

    def test_deadline_boundary_is_baseline(self):
        controller = self.make(auto=False)
        controller.request(5, user_started=True)
        controller.clock.value = controller.deadline
        self.assertEqual(controller.run(), SessionState.BASELINE)

    def test_invalid_transition_is_rejected(self):
        controller = self.make()
        with self.assertRaises(InvalidTransition):
            controller._transition(SessionState.ACTIVE, "bad")


if __name__ == "__main__": unittest.main()
