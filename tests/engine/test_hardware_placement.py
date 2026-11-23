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
            trace["trials"].append({"case": case, "arms": readings})
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
    learner.last_arrival -= 4000
    task = threading.Thread(target=learner.step)
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
    learner.path.write_text(json.dumps(trace))
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
