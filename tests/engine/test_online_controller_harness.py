"""Scripted process/snapshot checks; no device or model execution occurs."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import io
import json
import os
import subprocess
import sys
import tarfile
from types import ModuleType, SimpleNamespace

import pytest

from experiments.controller1.prepare import build_snapshot
from experiments.controller1.kaggle_runner import extract_verified
from tools import online_controller_eval as harness


def arguments(tmp_path):
    seed = tmp_path / "seed.json"
    seed.write_text('{"scripted":true}')
    return argparse.Namespace(output=tmp_path / "evaluation.json", model="scripted-model",
                              revision="scripted-revision", seed_checkpoint=seed,
                              groups=1, repeats=1, library=None, timeout=0.2)


@pytest.mark.skipif(os.name != "posix", reason="owned POSIX process-group contract")
def test_timeout_reaps_owned_worker_and_preserves_partial_artifacts(monkeypatch, tmp_path):
    args = arguments(tmp_path)
    original = subprocess.Popen
    owned = []

    def spawn(_argv, **kwargs):
        if "--single-arm" not in _argv:
            return original(_argv, **kwargs)
        child = original([sys.executable, "-c",
                          "import subprocess,sys,time; "
                          "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                          "print('owned-child-pid='+str(p.pid),flush=True); time.sleep(60)"], **kwargs)
        owned.append(child)
        return child

    monkeypatch.setattr(harness.subprocess, "Popen", spawn)
    with pytest.raises(subprocess.TimeoutExpired):
        harness.run_mlx(args)
    result = json.loads(args.output.read_text())
    assert result["status"] == "failed"
    assert result["failure"]["owned_process_reaped"]
    assert owned[0].poll() is not None
    artifacts = tmp_path / "evaluation-arms"
    assert (artifacts / "reference-0.json").read_bytes() == args.seed_checkpoint.read_bytes()
    log = (artifacts / "reference-0.log").read_text()
    assert "owned-child-pid=" in log
    child_pid = int(log.split("owned-child-pid=")[1].splitlines()[0])
    # A terminated orphan may briefly remain a zombie pending its OS reaper.
    probe = subprocess.run(["ps", "-o", "stat=", "-p", str(child_pid)],
                           text=True, capture_output=True, timeout=5)
    assert probe.returncode in (0, 1) and not probe.stderr, probe.stderr
    status = probe.stdout.strip()
    assert not status or status.startswith("Z")


@pytest.mark.skipif(os.name != "posix", reason="owned POSIX process-group contract")
def test_failed_later_arm_retains_completed_arm_and_its_checkpoint(monkeypatch, tmp_path):
    args = arguments(tmp_path)
    args.timeout = 5
    original = subprocess.Popen

    def spawn(argv, **kwargs):
        arm = argv[argv.index("--single-arm") + 1]
        output = argv[argv.index("--output") + 1]
        code = ("import json,pathlib,sys; "
                "p=pathlib.Path(sys.argv[1]); "
                "p.write_text(json.dumps({'identity':{'backend':'scripted'},'raw':[],"
                "'outer_wall_s':1.0,'status':'complete'})); "
                "print('scripted arm diagnostic',flush=True); "
                "sys.exit(int(sys.argv[2]))")
        return original([sys.executable, "-c", code, output, "0" if arm == "reference" else "1"], **kwargs)

    monkeypatch.setattr(harness.subprocess, "Popen", spawn)
    with pytest.raises(RuntimeError, match="evaluation invalid"):
        harness.run_mlx(args)
    result = json.loads(args.output.read_text())
    assert result["status"] == "failed"
    assert len(result["runs"]) == 1
    assert result["failure"]["arm"] == "simple"
    assert (tmp_path / "evaluation-arms/reference-0-out.json").is_file()
    assert (tmp_path / "evaluation-arms/simple-0-out.json").is_file()
    assert (tmp_path / "evaluation-arms/simple-0.log").is_file()


def test_snapshot_includes_untracked_shim_and_exact_drivers(tmp_path):
    snapshot = tmp_path / "snapshot.tar.gz"
    result = build_snapshot(snapshot)
    assert result["cloud_started"] is False
    extracted = tmp_path / "source"
    manifest = extract_verified(snapshot, extracted)
    for path in ("ironmule/online_controller.py", "ironmule_controller.py",
                 "tools/online_controller_eval.py", "experiments/controller1/prepare.py",
                 "experiments/controller1/kaggle_runner.py"):
        assert path in manifest["files"]
        assert (extracted / path).is_file()


@pytest.mark.parametrize("attack", ["duplicate", "unlisted", "escape"])
def test_unverified_snapshot_members_cannot_change_executed_source(tmp_path, attack):
    snapshot = tmp_path / "invalid.tar.gz"
    manifest = {"schema": "ironmule.controller1-snapshot.v1", "files": {},
                "base_commit": "test", "local_source_included": True}
    with tarfile.open(snapshot, "w:gz") as archive:
        entries = [("snapshot-manifest.json", json.dumps(manifest).encode())]
        if attack == "duplicate":
            entries.append(entries[0])
        elif attack == "unlisted":
            entries.append(("unverified.py", b"print('unverified')"))
        else:
            entries.append(("../outside.py", b"unverified"))
        for name, payload in entries:
            item = tarfile.TarInfo(name)
            item.size = len(payload)
            archive.addfile(item, io.BytesIO(payload))
    with pytest.raises(ValueError):
        extract_verified(snapshot, tmp_path / "source")
    assert not (tmp_path / "outside.py").exists()


def test_control2_configuration_keeps_qualification_and_workload_prospective():
    spec = harness.load_control2_spec(harness.ROOT / "experiments/controller1/control2_spec.json")
    settings = harness.control2_config(spec)
    assert settings.verify_exploration is True
    assert settings.min_pairs == 64
    assert settings.seed == 15
    assert settings.exploration_ppm == 50_000
    assert spec["performance_claim"] is False
    assert spec["execution_budget"]["maximum_new_probe_measurements"] == 1
    assert [harness.control2_group(spec, i) for i in range(3)] == [
        {"requests": 1, "length": 8}, {"requests": 2, "length": 8}, {"requests": 1, "length": 8}]
    assert harness.control2_config(spec, frozen=True).frozen


def test_hardware_cli_rejects_synthetic_adapter_without_device_import(monkeypatch, tmp_path):
    before = set(sys.modules)
    monkeypatch.setattr(sys, "argv", ["online_controller_eval.py", "--hardware-learning",
                                     "--output", str(tmp_path / "unused.json")])
    with pytest.raises(SystemExit) as rejected:
        harness.main()
    assert rejected.value.code == 2
    assert "mlx.core" not in set(sys.modules) - before
    assert not (tmp_path / "unused.json").exists()


def test_hardware_cli_rejects_changed_workload_before_device_import(monkeypatch, tmp_path):
    before = set(sys.modules)
    monkeypatch.setattr(sys, "argv", ["online_controller_eval.py", "--backend", "mlx",
                                     "--hardware-learning", "--groups", "9",
                                     "--output", str(tmp_path / "unused.json")])
    with pytest.raises(SystemExit) as rejected:
        harness.main()
    assert rejected.value.code == 2
    assert "mlx.core" not in set(sys.modules) - before


def test_simple_hardware_warmup_and_measurement_use_same_profile(monkeypatch):
    from types import SimpleNamespace
    from ironmule_product.engine_bridge import serving_profile_contract
    spec = harness.load_control2_spec(harness.ROOT / "experiments/controller1/control2_spec.json")
    definitions = serving_profile_contract({"compiled_fixed_cache": False,
                                           "head_skip_prefill": False})["definitions"]
    seen = []
    runtime = SimpleNamespace(serve=lambda _: pytest.fail("simple profile was not applied"))
    monkeypatch.setattr(harness, "mlx_requests", lambda _runtime, _index, group: [None] * group["requests"])
    def execute(_runtime, _requests, definition, *, capture_state):
        seen.append((definition["profile_id"], capture_state))
        return [], 1.0, "checked-in-pilot"
    monkeypatch.setattr(harness, "execute_registered_profile", execute)
    # The exact helper used by mlx_arm is called for warmup and measurement.
    for _phase in ("warmup", "measurement"):
        for index in range(2):
            harness.run_arm_group(runtime, "simple", index, spec=spec,
                                  definitions=definitions, simple_action=3)
    assert seen == [("core.sequential.v1", False), ("core.grouped4.v1", False)] * 2


def training_spec(tmp_path, training):
    spec = harness.load_control2_spec(harness.ROOT / "experiments/controller1/control2_spec.json")
    spec["experiment"] = "explicit-scripted-training-test"
    spec["training"] = training
    path = tmp_path / "training-spec.json"
    path.write_text(json.dumps(spec))
    return path


@pytest.mark.parametrize("training", [
    None, {}, {"groups": True, "request_count_cycle": [1], "max_tokens": 8},
    {"groups": 0, "request_count_cycle": [1], "max_tokens": 8},
    {"groups": 4097, "request_count_cycle": [1], "max_tokens": 8},
    {"groups": 1, "request_count_cycle": [], "max_tokens": 8},
    {"groups": 1, "request_count_cycle": [True], "max_tokens": 8},
    {"groups": 1, "request_count_cycle": [5], "max_tokens": 8},
    {"groups": 1, "request_count_cycle": [0], "max_tokens": 8},
    {"groups": 1, "request_count_cycle": [1], "max_tokens": False},
    {"groups": 1, "request_count_cycle": [1], "max_tokens": 65},
    {"groups": 1, "request_count_cycle": [1], "max_tokens": 0},
    {"groups": 1, "request_count_cycle": [1], "max_tokens": 8, "unknown": 1},
    {"groups": 1, "request_count_cycle": [1], "max_tokens": 8, "scope": True},
    {"groups": 1, "request_count_cycle": [1], "max_tokens": 8, "report_every_groups": 0},
    {"groups": 1, "request_count_cycle": [1], "max_tokens": 8, "report_every_groups": 65},
    {"groups": 1, "request_count_cycle": [1], "max_tokens": 8, "report_every_groups": True},
])
def test_training_spec_rejects_invalid_or_unbounded_inputs(tmp_path, training):
    path = training_spec(tmp_path, training)
    with pytest.raises(ValueError, match="bounded real training"):
        harness.load_control2_spec(path)


def test_training_spec_accepts_closed_bounds_and_optional_descriptions(tmp_path):
    training = {"groups": 4096, "request_count_cycle": [1, 2, 4], "max_tokens": 64,
                "scope": "scripted validation", "qualification": "fixed gates", "report_every_groups": 64}
    assert harness.load_control2_spec(training_spec(tmp_path, training))["training"] == training


@pytest.mark.parametrize("budget", [True, 0, 1, 3, 1026, 32770, 2.0])
def test_training_spec_rejects_invalid_offline_comparison_authority(tmp_path, budget):
    training = {"groups": 3, "request_count_cycle": [1], "max_tokens": 8,
                "offline_extra_comparison_executions": budget}
    with pytest.raises(ValueError, match="offline comparison budget"):
        harness.load_control2_spec(training_spec(tmp_path, training))


def test_training_spec_accepts_exact_fixed_qualification_family_budget(tmp_path):
    training = {"groups": 3, "request_count_cycle": [1], "max_tokens": 8,
                "offline_extra_comparison_executions": 1024}
    assert harness.load_control2_spec(training_spec(tmp_path, training))["training"] == training


@pytest.mark.parametrize("protocol", [None, True, 2, "adverse", "strict_v2"])
def test_harness_rejects_unknown_qualification_protocol(tmp_path, protocol):
    spec = json.loads((harness.ROOT / "experiments/controller1/control2_spec.json").read_text())
    spec["qualification_protocol"] = protocol
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="qualification protocol"):
        harness.load_control2_spec(path)


@pytest.mark.parametrize("protocol", ["strict_v1", "adverse_control_v2", "sequential_v3"])
def test_harness_uses_same_declared_protocol_for_training_and_frozen_replay(tmp_path, protocol):
    spec = json.loads((harness.ROOT / "experiments/controller1/control2_spec.json").read_text())
    spec["qualification_protocol"] = protocol
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(spec))
    loaded = harness.load_control2_spec(path)
    assert harness.control2_config(loaded).qualification_protocol == protocol
    assert harness.control2_config(loaded, frozen=True).qualification_protocol == protocol
    assert harness.control2_config(loaded).min_train == 8
    assert harness.control2_config(loaded).min_pairs == 64
    loaded.pop("qualification_protocol")
    assert harness.control2_config(loaded).qualification_protocol == "strict_v1"


def scripted_preparation(monkeypatch, tmp_path, *, training=None, promote=False, failure=None):
    """Explicit process/controller doubles; no MLX import or model execution."""
    from ironmule_product.engine_bridge import serving_profile_contract
    package = ModuleType("ironmule")
    package.__path__ = []
    hw = ModuleType("ironmule.hw")
    hw.measure = lambda: {"explicit_scripted": True}
    hw.apply_cuda_graph_defaults = lambda: {}
    service = ModuleType("ironmule.service")
    service.Runtime = SimpleNamespace(load=lambda *_args, **_kwargs: None)
    service.InteractiveMode = service.ThroughputMode = SimpleNamespace
    service.runtime_identity = lambda _runtime: controller.identity
    package.hw = hw
    monkeypatch.setitem(sys.modules, "ironmule", package)
    monkeypatch.setitem(sys.modules, "ironmule.hw", hw)
    monkeypatch.setitem(sys.modules, "ironmule.service", service)
    args = arguments(tmp_path)
    args.timeout = 60
    args.seed_checkpoint.unlink()
    args.probe_cache_dir = tmp_path / "probe"
    args.control2_spec = (training_spec(tmp_path, training) if training is not None
                          else harness.ROOT / "experiments/controller1/control2_spec.json")
    spec = harness.load_control2_spec(args.control2_spec)
    connection_groups = spec["rust_execution_connection"]["reference_only_credit_groups"] + 1
    flushes = []
    controllers = []
    definitions = serving_profile_contract({"compiled_fixed_cache": False,
                                            "head_skip_prefill": False})

    class ScriptedController:
        identity = {"backend": "explicit-scripted", "grouping_supported": True}
        action_contract = definitions
        def __init__(self, _identity=None, *, checkpoint_path, **_kwargs):
            self.path = checkpoint_path
            self.updates = self.promoted = self.killed = self.fault_mask = self.mismatches = 0
            self.counts = [0] * 24
            self.means = [0.0] * 24
            self.pending = []
            self.closed = False
            self.last_decision = {}
            self.offline = dict(granted=0, used=0, remaining=0, active=0)
            if self.path.exists():
                saved = json.loads(self.path.read_text())
                for key in ("updates", "promoted", "killed", "counts", "means"):
                    setattr(self, key, saved[key])
            controllers.append(self)
        def snapshot(self):
            state = {key: 0 for key in ("active_version", "fallback_version", "candidate_version", "frozen_after",
                "decisions", "observations", "dropped", "overridden", "abandoned", "rejected", "invalid_trials",
                "last_pair_id", "trials_started", "credits", "pending")}
            state.update({"active_actions": [0] * 6, "fallback_actions": [0] * 6, "candidate_actions": [0] * 6,
                    "updates": self.updates, "promoted": self.promoted, "killed": self.killed,
                    "fault_mask": self.fault_mask, "checkpoint_rejected": False,
                    "offline_training": dict(self.offline),
                    "working_counts": list(self.counts), "working_means": list(self.means),
                    "service": {"mismatches": self.mismatches, "deadline_violations": 0,
                                "groups": 0, "requests": 0, "comparisons": 0, "exploration_checks": 0}})
            return state
        @contextmanager
        def offline_training(self, *, max_extra_comparison_executions):
            assert self.offline["granted"] == 0
            self.offline.update(granted=max_extra_comparison_executions,
                                remaining=max_extra_comparison_executions, active=1)
            try:
                yield self
            finally:
                self.offline.update(remaining=0, active=0)
        def flush(self, **_kwargs):
            flushes.append(len(self.pending))
            for requests in self.pending:
                self.updates += 1
                cell = (min(requests, 3) - 1) * 2
                self.counts[cell * 4] += 1
                self.means[cell * 4] = float(self.counts[cell * 4])
            if promote and len(self.pending) > 1:
                self.promoted = 1
            self.pending.clear()
            return True
        def save(self):
            self.path.write_text(json.dumps({key: getattr(self, key) for key in
                ("updates", "promoted", "killed", "counts", "means")}))
        def hardware_knowledge(self):
            return {"source": "explicit scripted test", "updates": self.updates}
        def kill(self):
            self.killed = 1
            self.save()
        def close(self):
            if not self.closed:
                self.save()
                self.closed = True

    controller = ScriptedController(checkpoint_path=args.seed_checkpoint)
    class ScriptedRuntime:
        def __init__(self):
            self.calls = 0
            self.groups = []
            self.fallbacks = 0
            self.telemetry = SimpleNamespace(fallbacks=0, snapshot=lambda: {
                "fallbacks": self.fallbacks, "correctness_errors": 0})
        def enable_hardware_learning(self, *_args, **_kwargs):
            return controller
        def serve(self, requests):
            index = self.calls - connection_groups
            self.calls += 1
            self.groups.append(len(requests))
            controller.last_decision = {"action": 0, "actual_action": 0, "correctness": "unchecked"}
            if controller.offline["active"] and controller.offline["remaining"] >= 2:
                controller.offline["used"] += 2
                controller.offline["remaining"] -= 2
            if self.calls == connection_groups:
                controller.last_decision.update(verification=True, correctness="matched", returned_action=0)
            if failure == "error" and index == 1:
                raise RuntimeError("scripted training failure")
            if failure == "mismatch" and index == 1:
                controller.mismatches += 1  # Quiet reference recovery still invalidates training.
            if failure == "fallback" and index == 1:
                self.fallbacks = 1
            if failure == "deadline" and index == 0:
                monkeypatch.setattr(harness.time, "monotonic", lambda: 1e30)
            controller.pending.append(len(requests))
            return requests
        def close(self):
            controller.close()
    runtime = ScriptedRuntime()
    record = {"backend": "explicit-scripted-probe"}
    monkeypatch.setattr(hw, "probe", lambda **_: record, raising=False)
    monkeypatch.setattr(service.Runtime, "load", lambda *_args, **_kwargs: runtime)
    monkeypatch.setattr(harness, "OnlineController", ScriptedController)
    def requests(_runtime, index, group):
        return [SimpleNamespace(rid=f"test-{index}-{i}", prompt_ids=[1] * (133 if group["length"] == 1 else 18),
                                tokens=[3], stop_reason="length", text="scripted",
                                metrics={"physical_generated_tokens": 1, "visible_generated_tokens": 1})
                for i in range(group["requests"])]
    monkeypatch.setattr(harness, "mlx_requests", requests)
    monkeypatch.setattr(harness, "execute_registered_profile", lambda _runtime, rows, _definition:
                        (rows, 0.001, "explicit-scripted-equal-output"))
    return args, runtime, controller, connection_groups, flushes


def test_optional_real_training_uses_existing_serve_with_only_final_drain(monkeypatch, tmp_path):
    training = {"groups": 3, "request_count_cycle": [1, 2, 4], "max_tokens": 8}
    args, runtime, controller, connection, flushes = scripted_preparation(
        monkeypatch, tmp_path, training=training)
    report = harness.prepare_control2_checkpoint(args)
    saved = report["training"]
    assert runtime.groups[-3:] == [1, 2, 4]
    assert runtime.calls == connection + 3
    assert flushes == [1] * connection + [3]
    assert saved["synthetic_labels"] is False
    assert saved["fresh_evaluation_required"] is True
    assert saved["status"] == "complete" and saved["drained"] is True
    assert saved["update_delta"] == 3
    assert saved["working_counts_changed"] and saved["working_means_changed"]
    assert saved["selected_action_counts"] == {"0": 3}
    assert saved["returned_action_counts"] == {"0": 3}
    assert saved["executed_action_counts"] == {"0": 3}
    assert report["learned_state"]["updates"] == connection + 3
    assert report["restart_preserved_updates"] and report["operator_kill_survived_restart"]
    assert report["restart_preserved_learning_state"] is True
    assert report["restart_preserved_policy"] is True
    assert all("native_progress" in row for row in saved["raw"])
    assert saved["report_every_groups"] == 1
    assert report["seed_scope"] == "trained cost model; reference active, no qualified promotion"
    assert json.loads(args.seed_checkpoint.read_text())["updates"] == connection + 3


def test_training_retains_genuine_promotion_without_making_a_performance_claim(monkeypatch, tmp_path):
    args, _, _, _, _ = scripted_preparation(monkeypatch, tmp_path,
        training={"groups": 3, "request_count_cycle": [1, 2, 4], "max_tokens": 8}, promote=True)
    report = harness.prepare_control2_checkpoint(args)
    assert report["learned_state"]["promoted"] == 1
    assert report["training"]["promotions"] == 1
    assert "prospective qualification" in report["seed_scope"]
    assert report["performance_claim"] is False and report["qualification_evidence"] is False


def test_original_control2_has_no_training_or_additional_drain(monkeypatch, tmp_path):
    args, runtime, _, connection, flushes = scripted_preparation(monkeypatch, tmp_path)
    report = harness.prepare_control2_checkpoint(args)
    assert "training" not in report
    assert runtime.calls == connection and flushes == [1] * connection
    assert report["seed_scope"] == "observed, unqualified reference policy"


def test_explicit_offline_training_drains_each_real_group_and_revokes_before_seed(monkeypatch, tmp_path):
    args, _, controller, connection, flushes = scripted_preparation(monkeypatch, tmp_path,
        training={"groups": 3, "request_count_cycle": [1], "max_tokens": 8,
                  "offline_extra_comparison_executions": 8})
    original_flush = controller.flush
    measured_flushes = []
    clock = [0.0]
    monkeypatch.setattr(harness.time, "perf_counter", lambda: clock[0])
    def delayed_flush(**kwargs):
        if controller.offline["active"]:
            clock[0] += 0.25  # Explicit synthetic clock increment, no performance evidence.
            measured_flushes.append(controller.offline["used"])
        return original_flush(**kwargs)
    monkeypatch.setattr(controller, "flush", delayed_flush)
    report = harness.prepare_control2_checkpoint(args)
    training = report["training"]
    assert flushes == [1] * connection + [1, 1, 1, 0]
    assert measured_flushes == [2, 4, 6, 6]
    assert training["offline_group_drain_in_outer_elapsed"] is True
    assert training["offline_budget_before"] == dict(granted=0, used=0, remaining=0, active=0)
    assert training["offline_budget_granted"] == dict(granted=8, used=0, remaining=8, active=1)
    assert training["offline_budget_after"] == dict(granted=8, used=6, remaining=0, active=0)
    assert [row["outer_elapsed_s"] for row in training["raw"]] == [0.25] * 3
    assert [row["native_progress"]["offline_training"]["used"] for row in training["raw"]] == [2, 4, 6]
    assert report["restart_offline_rights_absent"] is True
    assert report["restart_offline_training"] == dict(granted=0, used=0, remaining=0, active=0)
    assert report["learned_state"]["offline_training"]["active"] == 0
    assert "offline" not in json.loads(args.seed_checkpoint.read_text())


def test_explicit_offline_training_error_revokes_remaining_authority(monkeypatch, tmp_path):
    args, _, controller, connection, flushes = scripted_preparation(monkeypatch, tmp_path,
        training={"groups": 3, "request_count_cycle": [1], "max_tokens": 8,
                  "offline_extra_comparison_executions": 8}, failure="error")
    with pytest.raises(RuntimeError, match="scripted training failure"):
        harness.prepare_control2_checkpoint(args)
    report = json.loads(args.output.read_text())
    assert flushes == [1] * connection + [1]
    budget = report["training"]["controller_after"]["offline_training"]
    assert budget == dict(granted=8, used=4, remaining=0, active=0)
    assert controller.offline["active"] == 0
    assert report["training"]["raw"][-1]["outer_elapsed_s"] >= 0


@pytest.mark.parametrize("failure,exception", [("error", RuntimeError), ("mismatch", RuntimeError),
                                              ("fallback", RuntimeError), ("deadline", TimeoutError)])
def test_training_error_keeps_progress_and_never_runs_intermediate_flush(monkeypatch, tmp_path, failure, exception):
    args, _, _, connection, flushes = scripted_preparation(monkeypatch, tmp_path,
        training={"groups": 3, "request_count_cycle": [1, 2, 4], "max_tokens": 8}, failure=failure)
    with pytest.raises(exception):
        harness.prepare_control2_checkpoint(args)
    report = json.loads(args.output.read_text())
    assert report["status"] == report["training"]["status"] == "failed"
    assert report["training"]["drained"] is False
    assert report["training"]["raw"][0]["status"] == "completed"
    assert "controller_after" in report["training"]
    assert flushes == [1] * connection


def test_fresh_evaluation_carries_drained_training_seed_scope(monkeypatch, tmp_path):
    args = arguments(tmp_path)
    args.timeout = 5
    args.hardware_learning = True
    args.probe_cache_dir = None
    args.control2_spec = training_spec(tmp_path, {"groups": 3, "request_count_cycle": [1, 2, 4], "max_tokens": 8})
    args.pilot_record = tmp_path / "preparation.json"
    scope = "trained policy with prospective qualification; fresh evaluation required"
    args.pilot_record.write_text(json.dumps({
        "status": "complete", "training": {"status": "complete", "drained": True},
        "seed_checkpoint_sha256": harness.hashlib.sha256(args.seed_checkpoint.read_bytes()).hexdigest(),
        "seed_scope": scope,
    }))
    original = subprocess.Popen
    def spawn(argv, **kwargs):
        arm = argv[argv.index("--single-arm") + 1]
        output = argv[argv.index("--output") + 1]
        code = ("import json,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(json.dumps({"
                "'arm':sys.argv[2],'identity':{'backend':'explicit-scripted'},"
                "'controller_before':{'service':{},'killed':0,'fault_mask':0,'checkpoint_rejected':False},"
                "'controller_after':{'service':{},'killed':0,'fault_mask':0,'checkpoint_rejected':False},"
                "'raw':[{'signature':'equal','workload':{'requests':1},'completed_requests':1,"
                "'telemetry':{},'decision':{}}],'outer_wall_s':1.0,'status':'complete'}))")
        return original([sys.executable, "-c", code, output, arm], **kwargs)
    monkeypatch.setattr(harness.subprocess, "Popen", spawn)
    result = harness.run_mlx(args)
    assert result["seed_scope"] == scope
    assert result["status"] == "complete"
    assert result["performance_claim"] is False


def scripted_arm(monkeypatch, tmp_path):
    args, runtime, controller, _, _ = scripted_preparation(monkeypatch, tmp_path,
        training={"groups": 3, "request_count_cycle": [1, 2, 4], "max_tokens": 8})
    args.hardware_learning = True
    args.pilot_record = tmp_path / "preparation.json"
    args.pilot_record.write_text(json.dumps({
        "identity": controller.identity,
        "spec_sha256": harness.hashlib.sha256(args.control2_spec.read_bytes()).hexdigest(),
        "simple_action": 0, "seed_scope": "explicit scripted trained seed",
    }))
    return args, runtime, controller


@pytest.mark.parametrize("phase", ["warmup", "measurement"])
def test_quiet_baseline_return_after_mismatch_invalidates_arm_and_keeps_row(monkeypatch, tmp_path, phase):
    args, runtime, controller = scripted_arm(monkeypatch, tmp_path)
    original = runtime.serve
    target = 0 if phase == "warmup" else 2
    def serve(work):
        current = runtime.calls
        rows = original(work)
        if current == target:
            controller.mismatches += 1
        return rows  # Delivered output remains the same reference signature.
    runtime.serve = serve
    with pytest.raises(RuntimeError, match="mismatch"):
        harness.mlx_arm(args, "online", args.seed_checkpoint)
    partial = json.loads(args.output.read_text())
    assert partial["status"] == "failed"
    assert partial["controller_after"]["service"]["mismatches"] == 1
    if phase == "warmup":
        assert len(partial["warmups"]) == 1 and partial["raw"] == []
    else:
        assert len(partial["warmups"]) == 2 and len(partial["raw"]) == 1
        assert partial["raw"][0]["signature"]
        assert partial["raw"][0]["completed_requests"] == 1


def test_async_fault_after_final_drain_invalidates_complete_delivered_arm(monkeypatch, tmp_path):
    args, runtime, controller = scripted_arm(monkeypatch, tmp_path)
    original = controller.flush
    def flush(**kwargs):
        complete = original(**kwargs)
        controller.fault_mask = 4
        return complete
    controller.flush = flush
    with pytest.raises(RuntimeError, match="faulted"):
        harness.mlx_arm(args, "online", args.seed_checkpoint)
    partial = json.loads(args.output.read_text())
    assert partial["status"] == "failed"
    assert len(partial["raw"]) == 1 and len(partial["warmups"]) == 2
    assert partial["controller_after"]["fault_mask"] == 4


def test_faulted_initial_controller_executes_no_evaluation_work(monkeypatch, tmp_path):
    args, runtime, controller = scripted_arm(monkeypatch, tmp_path)
    controller.killed = 1
    with pytest.raises(RuntimeError, match="killed"):
        harness.mlx_arm(args, "online", args.seed_checkpoint)
    partial = json.loads(args.output.read_text())
    assert runtime.calls == 0 and partial["status"] == "failed"
    assert partial["controller_before"]["killed"] == 1


@pytest.mark.parametrize("field", ["fallbacks", "correctness_errors", "plan_switch_attempts"])
def test_shared_hardware_health_check_rejects_each_execution_error(field):
    state = {"service": {}, "killed": 0, "fault_mask": 0, "checkpoint_rejected": False}
    sample = {"workload": {"requests": 1}, "completed_requests": 1,
              "telemetry": {field: 1}, "decision": {}}
    assert harness.hardware_execution_failure(state, state, sample=sample)


def test_training_reports_returned_reference_separately_from_executed_alternative(monkeypatch, tmp_path):
    args, runtime, controller, connection, _ = scripted_preparation(monkeypatch, tmp_path,
        training={"groups": 3, "request_count_cycle": [1, 2, 4], "max_tokens": 8})
    original = runtime.serve
    def serve(work):
        rows = original(work)
        if runtime.calls > connection:
            controller.last_decision.update(proposed_action=2, verification=True,
                correctness="matched", returned_action=0, executed_actions=[0, 2])
        return rows
    runtime.serve = serve
    report = harness.prepare_control2_checkpoint(args)
    assert report["training"]["selected_action_counts"] == {"2": 3}
    assert report["training"]["returned_action_counts"] == {"0": 3}
    assert report["training"]["executed_action_counts"] == {"0": 3, "2": 3}
    assert "actual_action_counts" not in report["training"]


def test_comparison_audit_selects_only_current_tickets_for_repeated_workload():
    decision = {"ticket": 10, "comparison": True, "candidate_version": 3,
                "workload_digest": "same-workload"}
    def event(ticket, action, candidate=3):
        return {"ticket": ticket, "action": action, "candidate_version": candidate,
                "workload_digest": "same-workload", "comparison": True, "executed": True}
    state = {"events": [event(1, 0), event(2, 0), event(3, 2),
                         event(10, 0), event(11, 0), event(12, 2), event(20, 3),
                         dict(decision, action=0)]}  # Outer decision event reuses A's ticket.
    actions, error = harness.recorded_actions(state, decision)
    assert actions == [0, 0, 2] and error is None
    state["events"].pop(4)  # A historical same-workload event cannot fill the current gap.
    assert harness.recorded_actions(state, decision) == ([], "current comparison execution audit incomplete")


def test_comparison_audit_does_not_invent_execution_from_missing_or_duplicate_records():
    decision = {"ticket": 10, "comparison": True, "candidate_version": 3,
                "workload_digest": "same-workload"}
    assert harness.recorded_actions({}, decision)[0] == []
    assert harness.recorded_actions({}, {"comparison": True})[0] == []
    event = {"ticket": 10, "action": 0, "candidate_version": 3,
             "workload_digest": "same-workload", "comparison": True, "executed": True}
    assert harness.recorded_actions({"events": [event, event, event]}, decision)[0] == []


def test_native_progress_preserves_unavailable_fields_without_fabricated_zero():
    progress = harness.native_progress({"updates": 8, "active_actions": [0] * 6})
    assert progress["updates"] == 8
    assert progress["candidate_version"] is None
    assert progress["pairs_completed"] is None
    assert progress["credits"] is None


@pytest.mark.parametrize("cadence,expected", [(1, [1, 2, 3, 4, 5]), (2, [2, 4])])
def test_training_report_cadence_keeps_every_raw_row_and_final_result(monkeypatch, tmp_path, cadence, expected):
    args, _, _, _, _ = scripted_preparation(monkeypatch, tmp_path,
        training={"groups": 5, "request_count_cycle": [1], "max_tokens": 8,
                  "report_every_groups": cadence})
    writes = []
    original = type(args.output).write_text
    def observed_write(path, data, *a, **kw):
        if path == args.output:
            writes.append(json.loads(data))
        return original(path, data, *a, **kw)
    monkeypatch.setattr(type(args.output), "write_text", observed_write)
    report = harness.prepare_control2_checkpoint(args)
    running = [len(row["training"]["raw"]) for row in writes
               if "training" in row and row["training"]["status"] == "running"
               and row["training"]["raw"]]
    assert running == expected
    assert len(report["training"]["raw"]) == 5
    assert json.loads(args.output.read_text())["training"]["completed_groups"] == 5


def test_policy_transition_saves_immediately_between_cadence_boundaries(monkeypatch, tmp_path):
    args, runtime, controller, connection, _ = scripted_preparation(monkeypatch, tmp_path,
        training={"groups": 5, "request_count_cycle": [1], "max_tokens": 8,
                  "report_every_groups": 4})
    original_serve = runtime.serve
    def serve(work):
        rows = original_serve(work)
        if runtime.calls == connection + 3:
            controller.promoted = 1
        return rows
    runtime.serve = serve
    writes = []
    original = type(args.output).write_text
    def observed_write(path, data, *a, **kw):
        if path == args.output:
            row = json.loads(data)
            if "training" in row and row["training"]["status"] == "running":
                writes.append(len(row["training"]["raw"]))
        return original(path, data, *a, **kw)
    monkeypatch.setattr(type(args.output), "write_text", observed_write)
    report = harness.prepare_control2_checkpoint(args)
    assert writes == [0, 3, 4]
    assert report["training"]["raw"][2]["native_progress"]["promoted"] == 1


def test_training_error_flushes_all_buffered_rows_without_waiting_for_cadence(monkeypatch, tmp_path):
    args, _, _, _, _ = scripted_preparation(monkeypatch, tmp_path,
        training={"groups": 3, "request_count_cycle": [1], "max_tokens": 8,
                  "report_every_groups": 64}, failure="error")
    with pytest.raises(RuntimeError, match="scripted training failure"):
        harness.prepare_control2_checkpoint(args)
    report = json.loads(args.output.read_text())
    assert report["status"] == "failed"
    assert len(report["training"]["raw"]) == 2
    assert report["training"]["raw"][0]["status"] == "completed"
    assert report["training"]["raw"][1]["status"] == "failed"


@pytest.mark.parametrize("field", ["active_actions", "fallback_actions", "active_version",
                                   "fallback_version", "killed", "fault_mask"])
def test_restart_refuses_changed_active_or_fallback_policy(monkeypatch, tmp_path, field):
    args, _, controller, _, _ = scripted_preparation(monkeypatch, tmp_path,
        training={"groups": 3, "request_count_cycle": [1], "max_tokens": 8})
    cls = type(controller)
    original = cls.snapshot
    def snapshot(self):
        state = original(self)
        if self.path.name.endswith("-restart.json"):
            state[field] = [2] * 6 if field.endswith("actions") else 1
        return state
    monkeypatch.setattr(cls, "snapshot", snapshot)
    with pytest.raises(RuntimeError, match="restart lost the active or fallback"):
        harness.prepare_control2_checkpoint(args)
    report = json.loads(args.output.read_text())
    assert report["status"] == "failed"
    assert report["restart_preserved_policy"] is False


def test_restart_preserves_nonreference_active_policy_as_well_as_updates(monkeypatch, tmp_path):
    args, _, controller, _, _ = scripted_preparation(monkeypatch, tmp_path,
        training={"groups": 3, "request_count_cycle": [1], "max_tokens": 8}, promote=True)
    cls, original = type(controller), type(controller).snapshot
    def snapshot(self):
        state = original(self)
        if self.promoted:
            state["active_version"] = 1
            state["active_actions"] = [2] + [0] * 5
        return state
    monkeypatch.setattr(cls, "snapshot", snapshot)
    report = harness.prepare_control2_checkpoint(args)
    assert report["learned_state"]["active_actions"][0] == 2
    assert report["restart_preserved_policy"] is True


def test_fresh_cuda_arm_validates_cache_under_existing_graph_defaults(monkeypatch, tmp_path):
    from ironmule import hw, service
    import mlx.core as mx

    monkeypatch.delenv("MLX_MAX_OPS_PER_BUFFER", raising=False)
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(mx, "cuda", SimpleNamespace(is_available=lambda: True), raising=False)
    monkeypatch.setattr(mx, "device_info", lambda: {"compute_capability_major": 7})
    monkeypatch.setattr(harness, "load_control2_spec", lambda _: {
        "target": {"reference_tuned_profile": False}})
    cache_reads = []
    def cached_probe(**options):
        assert options["allow_measure"] is False
        if os.environ.get("MLX_MAX_OPS_PER_BUFFER") != "400":
            raise RuntimeError("stored CUDA graph binding does not match fresh process")
        cache_reads.append(options)
        return {}
    def stop_before_model(*args, **kwargs):
        raise RuntimeError("cache admitted; model execution intentionally not started")
    monkeypatch.setattr(hw, "probe", cached_probe)
    monkeypatch.setattr(service.Runtime, "load", stop_before_model)
    args = SimpleNamespace(hardware_learning=True, control2_spec=tmp_path / "spec.json",
                           probe_cache_dir=tmp_path, model="scripted", revision="fixed")
    with pytest.raises(RuntimeError, match="cache admitted"):
        harness.mlx_arm(args, "online", tmp_path / "seed.json")
    assert len(cache_reads) == 1


def readiness_setup(monkeypatch, tmp_path):
    args, runtime, controller, _, _ = scripted_preparation(monkeypatch, tmp_path,
        training={"groups": 3, "request_count_cycle": [1, 2, 4], "max_tokens": 8})
    spec = harness.load_control2_spec(args.control2_spec)
    spec["readiness"] = {"calls_per_profile_shape": 2, "startup_wall_limit_s": 180,
                         "scope": "before admitted traffic"}
    spec["pilot"]["warmups_per_profile"] = 0
    spec["evaluation"].update(warmups_per_arm=0, request_count_cycle=[1, 2, 4])
    args.control2_spec.write_text(json.dumps(spec))
    runtime.backend = SimpleNamespace(capacity_for=lambda lengths, cap: ((max(lengths) + cap + 8 + 63) // 64) * 64)
    runtime.telemetry = SimpleNamespace(fallbacks=0)
    calls = []
    def execute(_runtime, requests, definition):
        calls.append((len(requests), len(requests[0].prompt_ids), definition["profile_id"],
                      tuple(request.rid for request in requests)))
        for row in requests:
            generated = 1 if len(row.prompt_ids) == 133 else 8
            row.metrics.update(physical_generated_tokens=generated, visible_generated_tokens=generated)
        runtime.telemetry.snapshot = lambda: {
            "fallbacks": 0, "correctness_errors": 0, "plan_switch_attempts": 0,
            "per_request": [{"service_ttft_ms": 100.0, "latency_ms": 300.0} for _ in requests]}
        return requests, 0.3, "explicit-scripted-output-and-state"
    monkeypatch.setattr(harness, "execute_registered_profile", execute)
    return args, runtime, controller, spec, calls


def test_readiness_uses_declared_shapes_and_preserves_learning_state(monkeypatch, tmp_path):
    _, runtime, controller, spec, calls = readiness_setup(monkeypatch, tmp_path)
    record = {}
    harness.run_readiness(runtime, controller, spec, phase="preparation",
        startup_started=harness.time.perf_counter(), within_budget=lambda: None, record=record, save=lambda: None)
    assert record["status"] == "ready" and record["no_learning"] is True
    assert record["controller_before"] == record["controller_after"]
    assert record["service_before"] == record["service_after"]
    assert len(calls) == 22
    assert [(row["requests"], row["length"], row["reference_only"]) for row in record["shapes"]] == [
        (2, 8, False), (1, 1, True), (1, 8, False), (4, 8, False)]
    assert all("sequential" in profile for count, _, profile, _ in calls if count == 1)
    assert all(profile == "current.sequential.v1" for _, length, profile, _ in calls if length == 133)
    assert [row["planned_capacity"] for row in record["raw"] if row["prompt_tokens"] == [133]] == [192, 192]
    assert all(calls[index][3] == calls[index + 1][3] for index in range(0, len(calls), 2))
    assert [row["length"] for row in harness.readiness_shapes(spec, "evaluation")] == [8, 8, 8]


@pytest.mark.parametrize("failure", ["slow-repeat", "mismatch", "missing-timing", "no-decode", "learning"])
def test_readiness_failure_keeps_completed_calls_without_retry(monkeypatch, tmp_path, failure):
    _, runtime, controller, spec, calls = readiness_setup(monkeypatch, tmp_path)
    original = harness.execute_registered_profile
    def execute(*args):
        rows, elapsed, digest = original(*args)
        if failure == "mismatch" and len(calls) == 3:
            digest = "different-scripted-terminal-state"
        if failure == "no-decode":
            rows[0].metrics["physical_generated_tokens"] = 1
        if failure in ("slow-repeat", "missing-timing"):
            runtime.telemetry.snapshot = lambda: {"fallbacks": 0, "correctness_errors": 0,
                "plan_switch_attempts": 0, "per_request": [{"service_ttft_ms": None if failure == "missing-timing" else 3000,
                                                          "latency_ms": 4000} for _ in rows]}
        if failure == "learning":
            controller.updates += 1
        return rows, elapsed, digest
    monkeypatch.setattr(harness, "execute_registered_profile", execute)
    record = {}
    with pytest.raises(RuntimeError):
        harness.run_readiness(runtime, controller, spec, phase="preparation",
            startup_started=harness.time.perf_counter(), within_budget=lambda: None, record=record, save=lambda: None)
    assert record["status"] == "failed" and record["raw"]
    if failure == "slow-repeat":
        assert len(calls) == 2
    if failure == "learning":
        assert record["no_learning"] is False


def test_startup_excess_is_retained_without_conflating_completion_and_ttft(monkeypatch, tmp_path):
    _, runtime, controller, spec, calls = readiness_setup(monkeypatch, tmp_path)
    original = harness.execute_registered_profile
    def execute(*args):
        rows, _, digest = original(*args)
        first = len(calls) == 1
        runtime.telemetry.snapshot = lambda: {"fallbacks": 0, "correctness_errors": 0,
            "plan_switch_attempts": 0, "per_request": [{"service_ttft_ms": 20000 if first else 100,
                                                       "latency_ms": 20000 if first else 3000} for _ in rows]}
        return rows, 20.0 if first else 3.0, digest
    monkeypatch.setattr(harness, "execute_registered_profile", execute)
    record = {}
    harness.run_readiness(runtime, controller, spec, phase="preparation",
        startup_started=harness.time.perf_counter(), within_budget=lambda: None, record=record, save=lambda: None)
    assert record["status"] == "ready" and record["first_call_slo_excess_calls"] == 1
    assert record["raw"][0]["elapsed_s"] == 20.0
    assert record["raw"][1]["elapsed_s"] == 3.0 and record["raw"][1]["admitted_slo_met"] is True


def test_readiness_cannot_admit_an_empty_generated_workload(monkeypatch, tmp_path):
    _, runtime, controller, spec, calls = readiness_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(harness, "mlx_requests", lambda *args: [])
    record = {}
    with pytest.raises(RuntimeError, match="declared shape"):
        harness.run_readiness(runtime, controller, spec, phase="preparation",
                              startup_started=harness.time.perf_counter(), within_budget=lambda: None,
                              record=record, save=lambda: None)
    assert calls == [] and record["status"] == "failed"


def test_preparation_readiness_precedes_traffic_and_does_not_repeat_pilot_warmups(monkeypatch, tmp_path):
    args, runtime, _, _, calls = readiness_setup(monkeypatch, tmp_path)
    original_serve = runtime.serve
    def serve(work):
        assert len(calls) >= 22
        return original_serve(work)
    runtime.serve = serve
    report = harness.prepare_control2_checkpoint(args)
    assert report["readiness"]["status"] == "ready" and report["readiness"]["no_learning"] is True
    assert len(report["pilot"]) == 12
    assert report["probe_wall_s"] >= 0 and report["model_load_s"] >= 0
    assert report["complete_preparation_wall_s"] >= report["readiness"]["setup_elapsed_s"]


def test_fresh_arm_uses_same_readiness_without_credit_shapes_or_learning_warmups(monkeypatch, tmp_path):
    args, runtime, controller, _, calls = readiness_setup(monkeypatch, tmp_path)
    args.hardware_learning = True
    args.pilot_record = tmp_path / "preparation.json"
    args.pilot_record.write_text(json.dumps({"identity": controller.identity, "source_hashes": {},
        "spec_sha256": harness.hashlib.sha256(args.control2_spec.read_bytes()).hexdigest(),
        "simple_action": 0, "seed_scope": "explicit scripted trained seed"}))
    original_serve = runtime.serve
    def serve(work):
        assert len(calls) == 20
        return original_serve(work)
    runtime.serve = serve
    report = harness.mlx_arm(args, "online", args.seed_checkpoint)
    assert report["status"] == "complete" and report["readiness"]["no_learning"] is True
    assert report["warmups"] == [] and len(report["raw"]) == 1
    assert all(length == 18 for _, length, _, _ in calls)
    assert report["setup_execution_drain_checkpoint_close_s"] >= report["readiness"]["setup_elapsed_s"]


@pytest.mark.parametrize("field,value", [("calls_per_profile_shape", True), ("calls_per_profile_shape", 1),
                                        ("startup_wall_limit_s", False), ("startup_wall_limit_s", 241)])
def test_readiness_spec_rejects_invalid_budget_or_repetition_count(monkeypatch, tmp_path, field, value):
    args, _, _, spec, _ = readiness_setup(monkeypatch, tmp_path)
    spec["readiness"][field] = (spec["execution_budget"]["owned_child_deadline_s"] + 1
                                if value == 241 else value)
    args.control2_spec.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="bounded readiness"):
        harness.load_control2_spec(args.control2_spec)
