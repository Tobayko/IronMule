"""Exercise the real C++ solver, later hardware comparisons and preemption at the serving boundary."""
import json
import threading
import time
from types import SimpleNamespace

import pytest

from ironmule_product import placement, placement_devices
from ironmule_product import placement_training as train
from ironmule_product.device_pool import DevicePool
from ironmule_product.placement_native import NativeModel
from ironmule_product.types import GenerationRequest


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    return NativeModel(tmp_path_factory.mktemp("native"))


def trace_for(native):
    binding = {"devices": ["0", "cpu"], "trainer": native.version}
    plan = train.workload([], 2)
    trace = {"binding": binding, "identity": train.signature(binding), "schema": train.SCHEMA,
             "started": time.time(), "plan": plan, "trials": []}
    for _ in range(train.ROUNDS):
        for case in plan:
            size, cap, queues = case
            readings = []
            for arm, q in enumerate(queues):
                cost = (10 + size / 64 + cap / 8) * (1 + q) * (4 if arm == 0 else 1)
                readings.append([cost, cost / 2, cap])
            trace["trials"].append({"case": case, "arms": readings, "outputs": ["a" * 64] * 2})
    return trace


def test_fit_and_later_paired_holdouts(native):
    trace = trace_for(native)
    report = train.qualify(native, trace)
    assert report["qualified"] and all(r["changed_routes"] for r in report["holdout"])
    assert train.decision(native, report["weights"], 200, 32, [0, 0]) == 1
    # An adverse later round cannot affect the fitted weights or activate the policy.
    for row in trace["trials"][-train.CASES:]:
        row["arms"][1][0] *= 100
    rejected = train.qualify(native, trace)
    assert rejected["weights"] == report["weights"] and not rejected["qualified"]


def test_first_token_output_count_and_binding_are_gates(native):
    t = trace_for(native)
    for row in t["trials"][-train.CASES:]:
        row["arms"][1][2] = 1
    assert not train.qualify(native, t)["qualified"]
    t = trace_for(native)
    for row in t["trials"]:
        row["arms"][0][1] = .001
    assert not train.qualify(native, t)["qualified"]
    t["binding"]["devices"].reverse()
    with pytest.raises(ValueError):
        train.qualify(native, t)
    with pytest.raises(ValueError):
        native.fit([(1, 10, 0, float("nan"))])


def test_scope_and_clock(native, capsys):
    t = trace_for(native)
    assert train.covered(t["plan"], 500, 32, [0, 0])
    assert not train.covered(t["plan"], 500, 32, [5, 0])
    assert not train.covered(t["plan"], 8192, 32, [0, 0])
    train.main(["estimate", "--devices", "3", "--seconds-per-arm", "8", "--startup-minutes", "15"])
    report = json.loads(capsys.readouterr().out)
    assert report["worker_measurements"] == 120
    assert 31 < report["estimated_total_minutes"] < 33


class Worker:
    def __init__(self):
        self.spec = SimpleNamespace(model_id="small", revision="gguf-abcd")
        self.store = SimpleNamespace(root=None)
        self.backend = SimpleNamespace(thinking=True)
        self.open = {}
        self.cancelled = []
        self.started = threading.Event()

    def accepting(self):
        return True

    def holds(self, rid):
        return rid in self.open

    def outstanding(self):
        return len(self.open)

    foreground_outstanding = outstanding

    def stream(self, request):
        gate = threading.Event()
        self.open[request.request_id] = gate
        self.started.set()
        def output():
            gate.wait(2)
            yield {"type": "done", "ttft_ns": 1000000, "service_ns": 2000000, "completion_tokens": 1}
        return output()

    def cancel(self, rid):
        gate = self.open.pop(rid, None)
        if gate:
            gate.set()
            self.cancelled.append(rid)

    def close(self):
        for rid in list(self.open):
            self.cancel(rid)


def create_controller(tmp_path, monkeypatch, names=("cpu",)):
    monkeypatch.setattr(placement, "room_for_probes", lambda _: True)
    workers = [Worker() for _ in names]
    pool = DevicePool(workers, list(names))
    opts = {d: {"ggml_gate": {"key": "verified-" + d, "passes": True}} for d in names}
    learner = placement.Placement(pool, opts, tmp_path, idle_seconds=1800)
    pool.hardware_placement = learner
    return pool, learner


def test_arrival_preempts_probe_and_preserves_request(tmp_path, monkeypatch):
    pool, learner = create_controller(tmp_path, monkeypatch)
    worker = pool.shards[0]
    learner.idle_since -= 4000
    def run_probe():
        with pytest.raises(placement.Interrupted):
            learner.step()
    task = threading.Thread(target=run_probe)
    task.start()
    try:
        assert worker.started.wait(5)
        req = GenerationRequest("small", (("user", "private foreground text"),), 8)
        events = pool.stream(req)
        task.join(3)
        assert not task.is_alive()
        assert learner.status()["paired_trials"] == 0
        assert learner.status()["interruptions"] == 1 and req.request_id not in worker.cancelled
        worker.cancel(req.request_id)
        assert next(events)["type"] == "done"
        learner.control("pause")
        assert "private foreground" not in learner.path.read_text()
    finally:
        pool.close()


def test_restore_regrade_drift_and_persistent_pause(tmp_path, monkeypatch):
    pool, learner = create_controller(tmp_path, monkeypatch, ("0", "cpu"))
    trace = trace_for(learner.native)
    trace.update(binding=learner.binding, identity=train.signature(learner.binding))
    trace["comparison"] = {"qualified": True, "weights": [[999]]}  # saved verdict is never trusted
    with learner.lock:
        learner.trace = trace
    learner._save()
    assert learner.path.stat().st_mode & 0o077 == 0
    pool.close()
    pool, learner = create_controller(tmp_path, monkeypatch, ("0", "cpu"))
    try:
        assert learner.status()["active"]
        order, prediction = learner.order([0, 1], (96, 8), [0, 0])
        assert order == [1, 0]
        for _ in range(3):
            list(learner.watch(iter([{"type": "done", "service_ns": 10**12}]), prediction))
        assert not learner.status()["active"] and json.loads(learner.path.read_text())["withdrawn"]
        learner.control("pause")
    finally:
        pool.close()
    pool, learner = create_controller(tmp_path, monkeypatch, ("0", "cpu"))
    try:
        assert learner.status()["phase"] == "paused" and not learner.status()["active"]
        learner.control("resume")
        assert not learner.status()["active"]
    finally:
        pool.close()


def test_cuda_replicas_split_and_cpu(monkeypatch):
    from ironmule import ggml_engine as engine
    gib = 1 << 30
    rows = [{"provider": "cuda", "free_memory": 4 * gib}] * 2
    monkeypatch.delenv("IRONMULE_GGML_GPU_LAYERS", raising=False)
    monkeypatch.setattr(engine, "ggml_devices", lambda: rows)
    monkeypatch.setattr(engine, "memory_bytes", lambda: (8 * gib, 10 * gib))
    monkeypatch.setattr(engine, "gpu_memory_bytes", lambda _: (4 * gib, 4 * gib))
    monkeypatch.setattr(placement_devices, "cuda_devices", lambda: ["GPU-a", "GPU-b"])
    assert placement_devices.candidates(gib) == ["GPU-a", "GPU-b", "cpu"]
    assert placement_devices.candidates(6 * gib) == ["local", "cpu"]
    assert placement_devices.candidates(10 * gib) == []
    monkeypatch.setenv("IRONMULE_GGML_GPU_LAYERS", "0")
    assert placement_devices.candidates(gib) == ["cpu"]
    assert placement_devices.worker_environment("cpu")["IRONMULE_GGML_GPU_LAYERS"] == "0"
    assert placement_devices.worker_environment("GPU-b")["CUDA_VISIBLE_DEVICES"] == "GPU-b"


def test_equal_length_different_response_or_stop_rejects_v2(native):
    trace = trace_for(native)
    trace["trials"][-train.CASES]["outputs"][1] = "b" * 64
    result = train.qualify(native, trace)
    assert not result["qualified"]
    assert result["holdout"][-1]["equal_output_counts"] and not result["holdout"][-1]["matching_responses"]
    del trace["trials"][-1]["outputs"]
    with pytest.raises(ValueError):
        train.qualify(native, trace)
    trace["schema"] = train.LEGACY_SCHEMA
    old = train.qualify(native, trace)
    assert old["protocol"] == train.LEGACY_SCHEMA and old["holdout"][-1]["matching_responses"] is None


def test_idle_shortens_only_when_recent_quiet_windows_fit_and_a_preemption_backs_off(tmp_path, monkeypatch):
    pool, learner = create_controller(tmp_path, monkeypatch)
    try:
        learner.idle_seconds = 15
        learner.idle_gaps.extend([6] * 7)
        assert learner.status()["idle_delay_seconds"] == 15
        learner.idle_gaps.append(6)
        learner.last_trial_s = 3
        assert learner.status()["idle_delay_seconds"] == 2
        learner.last_trial_s = 8
        assert learner.status()["idle_delay_seconds"] == 15
        learner.last_trial_s = 3
        learner.inflight["probe"] = pool.shards[0]
        req = GenerationRequest("small", (("user", "new foreground request"),))
        learner.arrival(req)
        assert learner.status()["idle_delay_seconds"] == 15
    finally:
        pool.close()


def test_rocm_replicas_use_physical_pci_child_identity_and_local_busy_sensors(tmp_path, monkeypatch):
    import ironmule_devices
    from ironmule import ggml_engine
    from ironmule_product.errors import InvalidRequest

    gib = 1 << 30
    first, second = "0000:31:00.0", "0000:4a:00.0"
    rows = [{"name": f"ROCm{i}", "provider": "rocm", "pci": bus, "free_memory": 4 * gib}
            for i, bus in enumerate((first, second))]
    monkeypatch.setattr(ggml_engine, "ggml_devices", lambda: rows)
    monkeypatch.setattr(ironmule_devices, "ggml_devices", lambda: rows)
    monkeypatch.setattr(ggml_engine, "memory_bytes", lambda: (2 * gib, 2 * gib))
    monkeypatch.delenv("IRONMULE_GGML_GPU_LAYERS", raising=False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "3,1")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,2,4,5")
    replicas = ["rocm:" + first, "rocm:" + second]
    assert placement_devices.candidates(gib) == [*replicas, "cpu"]
    assert placement_devices.candidates(6 * gib) == ["local"]
    env = placement_devices.worker_environment(replicas[1])
    assert env["HIP_VISIBLE_DEVICES"] == "1" and "ROCR_VISIBLE_DEVICES" not in env
    with pytest.raises(InvalidRequest):
        placement_devices.verify_rocm_worker(second)
    monkeypatch.setattr(ironmule_devices, "ggml_devices", lambda: [rows[0]])
    with pytest.raises(InvalidRequest):
        placement_devices.verify_rocm_worker(second)
    placement_devices.verify_rocm_worker(first)
    for bus, load in ((first, "0"), (second, "99")):
        (tmp_path / bus).mkdir()
        (tmp_path / bus / "gpu_busy_percent").write_text(load)
    assert placement_devices.accelerators_quiet(replicas[:1], pci_root=tmp_path) == []
    assert placement_devices.accelerators_quiet(replicas, pci_root=tmp_path) is None
    (tmp_path / first / "gpu_busy_percent").unlink()
    assert placement_devices.accelerators_quiet(replicas[:1], pci_root=tmp_path) is None


def test_cost_forecast_uses_remaining_cases_and_asymmetric_workers(native, capsys):
    t = trace_for(native)
    for i, row in enumerate(t["trials"]):
        row["wall_s"] = 12 if i % 8 == 7 else 1
    t["trials"] = t["trials"][:-1]
    assert train.remaining_idle_seconds(t) == 13
    train.main(["estimate", "--seconds-per-worker", "4", "60"])
    assert json.loads(capsys.readouterr().out)["estimated_idle_minutes"] == pytest.approx(43.58333)


def test_native_reward_fit_uses_own_paired_rewards_and_freezes_before_holdout(native):
    t = trace_for(native)
    t["binding"]["reward_trainer"] = native.reward_version
    t["identity"] = train.signature(t["binding"])
    first = native.fit_rewards(t["trials"][:16])
    assert native.choose_rewards(first, 96, 8, [0, 0]) == 1
    # Quicker answers that differ from the reference should lose the reward.
    for row in t["trials"][:16]:
        if row["case"][2][0] == 0:
            row["outputs"][1] = "b" * 64
    changed = native.fit_rewards(t["trials"][:16])
    assert native.choose_rewards(changed, 96, 8, [0, 0]) == 0
    a = train.qualify(native, t)
    assert len(a["selection"]) == 4 and a["selection"][-1]["method"] == "reward_policy"
    for row in t["trials"][-train.CASES:]:
        row["arms"][1][0] *= 1000
    b = train.qualify(native, t)
    for key in ("selection", "method", "weights"):
        assert a[key] == b[key]
    assert a.get("reward_weights") == b.get("reward_weights")


def test_first_start_preserves_existing_server_settings_and_lists_an_unregistered_gguf(tmp_path):
    from ironmule_product.state import ProductStore
    from ironmule_product.types import ModelSpec

    store = ProductStore(tmp_path)
    settings = store.setup("desktop")
    assert store.setup("server", only_if_missing=True) == settings
    pool = DevicePool([Worker()], ["cpu"])
    pool.store = store
    pool.spec = ModelSpec("gguf/example", "gguf-revision", str(tmp_path / "model.gguf"), 1024)
    pool.health = lambda: {"ready": True}
    assert pool.models() == [{"id": "gguf/example", "object": "model", "owned_by": "local",
                              "revision": "gguf-revision", "loaded": True}]
    assert store.models() == []


def test_recurrent_foreground_queue_shapes_reach_the_idle_plan_and_long_backlogs_keep_reference(native):
    repeated = [(2, 1)] * 6 + [(1, 2)] * 4 + [(6, 0)] * 12
    plan = train.workload([], 2, repeated)
    assert {tuple(queues) for _, _, queues in plan} == {(0, 0), (2, 1), (1, 2)}
    assert not train.covered(plan, 96, 8, [6, 0])
    t = trace_for(native)
    t["plan"] = plan
    for i, row in enumerate(t["trials"]):
        row["case"] = plan[i % train.CASES]
    train.check_trace(t)
    assert train.qualify(native, t)["holdout"]
    t["plan"][0][2][0] = 5
    with pytest.raises(ValueError):
        train.check_trace(t)


def test_paired_probes_hold_background_work_on_other_lanes_and_release_it(tmp_path, monkeypatch):
    pool, learner = create_controller(tmp_path, monkeypatch, ("cpu", "local"))
    worker, other = pool.shards
    def target(request):
        assert other.outstanding() == 2
        yield {"type": "token", "text": "correct"}
        yield {"type": "done", "service_ns": 2000000, "ttft_ns": 1000000,
               "completion_tokens": 1, "finish_reason": "stop"}
    monkeypatch.setattr(worker, "stream", target)
    try:
        times, digest = learner._probe(0, [96, 8, [0, 2]], learner.serial, 1)
        assert times == [2, 1, 1] and len(digest) == 64
        assert other.outstanding() == 0 and len(other.cancelled) == 2
    finally:
        pool.close()
