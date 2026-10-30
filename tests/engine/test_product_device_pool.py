"""HW1: one worker per GPU behind join-the-shortest-queue placement (no model, no GPU).

The end-to-end cases start `stab1_fake_worker.py` through the real `MLXWorkerClient` and
`ProductService`, so placement, pinning and the per-device logs are checked on the real
scheduler and protocol. Nothing here measures speed.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

import ironmule_product.backend as backend_module
from ironmule_product import cli
from ironmule_product.backend import MLXWorkerClient
from ironmule_product.device_pool import DevicePool, cuda_devices, select_devices, start_pool
from ironmule_product.errors import BackendUnavailable, InvalidRequest, Overloaded
from ironmule_product.service import STABILITY_LOG, ProductService
from ironmule_product.state import ProductStore
from ironmule_product.types import GenerationRequest, ModelSpec

FAKE = Path(__file__).with_name("stab1_fake_worker.py")


def test_devices_come_from_the_inherited_visibility_first():
    never = lambda *a, **k: pytest.fail("nvidia-smi must not run when CUDA_VISIBLE_DEVICES is set")
    assert cuda_devices({"CUDA_VISIBLE_DEVICES": "2, 3"}, never) == ["2", "3"]
    assert cuda_devices({"CUDA_VISIBLE_DEVICES": ""}, never) == []  # an empty value hides every GPU


def test_devices_otherwise_come_from_nvidia_smi():
    listed = SimpleNamespace(returncode=0, stdout="0\n1\n")
    assert cuda_devices({}, lambda *a, **k: listed) == ["0", "1"]
    assert cuda_devices({}, lambda *a, **k: SimpleNamespace(returncode=9, stdout="0\n")) == []

    def missing(*_a, **_k):
        raise FileNotFoundError("nvidia-smi")

    assert cuda_devices({}, missing) == []


def test_device_selection_refuses_what_would_share_or_miss_a_card():
    assert select_devices("all", ["0", "1"]) == ["0", "1"]
    assert select_devices("1", ["0", "1"]) == ["1"]
    for value, available, message in (("all", [], "no CUDA device"), ("0,0", ["0"], "twice"),
                                      ("2", ["0", "1"], "does not show"), ("0,", ["0"], "indices"),
                                      ("", ["0"], "indices")):
        with pytest.raises(InvalidRequest, match=message):
            select_devices(value, available)


def test_a_worker_names_at_most_one_device():
    spec = ModelSpec("local/test", "revision", "/nonexistent", 1)
    for bad in ("0,1", "", "0 1", 1, "a" * 65):
        with pytest.raises(ValueError, match="one CUDA device"):
            MLXWorkerClient(spec, cuda_device=bad)
    assert MLXWorkerClient(spec, cuda_device="GPU-5f2c-01").cuda_device == "GPU-5f2c-01"
    assert MLXWorkerClient(spec).cuda_device is None


class _Shard:
    """The part of ProductService a pool uses, with scripted load and admission."""

    def __init__(self, name: str, load: int = 0, accepting: bool = True, full: bool = False):
        self.name, self.load, self._accepting, self.full = name, load, accepting, full
        self.spec, self.store = "spec", SimpleNamespace(root=None, models=lambda: [])
        self.stability_log = self.serving_log = None
        self.admitted: list[str] = []
        self.cancelled: list[str] = []
        self.closed = False

    def accepting(self):
        return self._accepting

    def outstanding(self):
        return self.load + len(self.admitted)

    def holds(self, request_id):
        return request_id in self.admitted

    def stream(self, request):
        if self.full:
            raise Overloaded("request queue is full; retry later")
        self.admitted.append(request.request_id)
        return iter([{"type": "done", "shard": self.name}])

    def cancel(self, request_id):
        if request_id in self.admitted:
            self.cancelled.append(request_id)
            return True
        return False

    def health(self):
        return {"service": "ironmule", "ready": self._accepting, "mode": "server", "execution": "exact",
                "backend": "mlx_lm_reference", "loaded_model": "local/test", "queued_requests": self.load,
                "active_requests": len(self.admitted), "completed_requests": 1, "failed_requests": 0,
                "cancelled_requests": 0, "optimization": {}, "device": "gpu", "worker": {"state": "ready"},
                "in_flight": None, "groups_started": 0, "oldest_queued_s": None}

    def close(self):
        self.closed = True


def _request(request_id: str, content: str = "hello", max_tokens: int = 4) -> GenerationRequest:
    return GenerationRequest("local/test", (("user", content),), max_tokens=max_tokens, request_id=request_id,
                             stream=True)


def test_placement_joins_the_shortest_queue_with_ties_to_the_lower_index():
    first, second = _Shard("a"), _Shard("b")
    pool = DevicePool([first, second], ["0", "1"])
    for index in range(4):
        list(pool.stream(_request(f"r{index}")))
    assert first.admitted == ["r0", "r2"] and second.admitted == ["r1", "r3"]
    busy, idle = _Shard("a", load=3), _Shard("b", load=1)
    pool = DevicePool([busy, idle], ["0", "1"])
    assert [event["shard"] for event in pool.stream(_request("x"))] == ["b"]


def test_a_full_queue_overflows_to_the_next_device_and_only_then_refuses():
    full, open_ = _Shard("a", full=True), _Shard("b", load=5)
    pool = DevicePool([full, open_], ["0", "1"])
    assert [event["shard"] for event in pool.stream(_request("x"))] == ["b"]
    pool = DevicePool([_Shard("a", full=True), _Shard("b", full=True)], ["0", "1"])
    with pytest.raises(Overloaded):
        pool.stream(_request("y"))


def test_a_device_that_does_not_accept_is_skipped_and_none_is_unavailable():
    down, up = _Shard("a", accepting=False), _Shard("b", load=9)
    pool = DevicePool([down, up], ["0", "1"])
    assert [event["shard"] for event in pool.stream(_request("x"))] == ["b"]
    pool = DevicePool([_Shard("a", accepting=False)], ["0"])
    with pytest.raises(BackendUnavailable):
        pool.stream(_request("y"))


def test_ids_are_unique_across_devices_and_cancel_reaches_the_holder():
    first, second = _Shard("a"), _Shard("b")
    pool = DevicePool([first, second], ["0", "1"])
    pool.stream(_request("r0"))
    pool.stream(_request("r1"))
    with pytest.raises(InvalidRequest, match="duplicate"):
        pool.stream(_request("r1"))
    assert pool.cancel("r1") is True and second.cancelled == ["r1"] and first.cancelled == []
    assert pool.cancel("unknown") is False


def test_health_sums_the_devices_and_learning_stays_off():
    pool = DevicePool([_Shard("a", load=2), _Shard("b", accepting=False)], ["0", "1"])
    health = pool.health()
    assert health["ready"] is True and health["queued_requests"] == 2 and health["completed_requests"] == 2
    assert [device["cuda_device"] for device in health["devices"]] == ["0", "1"]
    assert [device["ready"] for device in health["devices"]] == [True, False]
    assert health["dispatch"] == "join_shortest_queue" and "worker" not in health
    assert pool.learning_status()["mode"] == "off"
    with pytest.raises(InvalidRequest, match="one per GPU"):
        pool.learning_control("pause")
    pool.close()
    assert all(shard.closed for shard in pool.shards)


def _fake_workers(tmp_path: Path, monkeypatch) -> tuple[ProductStore, ModelSpec, list[dict]]:
    real_popen = subprocess.Popen
    started: list[dict] = []

    def fake_popen(args, *popen_args, **popen_kwargs):
        if len(args) > 3 and str(args[3]).endswith("worker.py"):
            started.append({"visible": popen_kwargs["env"].get("CUDA_VISIBLE_DEVICES")})
            args = [sys.executable, "-u", str(FAKE), *args[4:]]
        return real_popen(args, *popen_args, **popen_kwargs)

    monkeypatch.setattr(backend_module.subprocess, "Popen", fake_popen)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    store = ProductStore(tmp_path / "state")
    store.setup("server")
    store.set_request_timeout(120)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    spec = ModelSpec("local/test", "revision", str(snapshot.resolve()), 1)
    store.register_model(spec)
    return store, spec, started


def _text(events: list[dict]) -> str:
    return "".join(event.get("text", "") for event in events if event.get("type") == "token")


def test_two_pinned_workers_share_concurrent_requests_and_answer_alike(tmp_path, monkeypatch):
    store, spec, started = _fake_workers(tmp_path, monkeypatch)
    pool = start_pool(store, spec, ["0", "1"],
                      lambda device: MLXWorkerClient(spec, execution_variant="current_engine", cuda_device=device),
                      ProductService)
    try:
        assert sorted(entry["visible"] for entry in started) == ["0", "1"]
        results: dict[str, list[dict]] = {}
        streams = [(f"r{index}", pool.stream(_request(f"r{index}", "slow:0.05"))) for index in range(4)]
        threads = [threading.Thread(target=lambda rid=rid, events=events: results.__setitem__(rid, list(events)))
                   for rid, events in streams]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert all(results[rid][-1]["type"] == "done" for rid, _ in streams)
        assert len({_text(results[rid]) for rid, _ in streams}) == 1  # same prompt, same answer, either card
        health = pool.health()
        assert [device["completed_requests"] for device in health["devices"]] == [2, 2]
        assert health["completed_requests"] == 4 and health["ready"] is True
        # A lost worker comes back on its own card (STAB1 through the pool's backend factory).
        replacement = pool.shards[1].make_backend()
        assert replacement.cuda_device == "1"
    finally:
        pool.close()
    log = [json.loads(line) for line in (store.root / STABILITY_LOG).read_text().splitlines()]
    ends = [record for record in log if record["event"] == "end"]
    assert sorted(record["device"] for record in ends) == ["0", "0", "1", "1"]
    assert all((store.root / f"serving-log.cuda-{device}.jsonl").exists() for device in ("0", "1"))
    assert not (store.root / "serving-log.jsonl").exists()  # the single-worker dreamer sees none of it


def test_a_worker_without_a_device_inherits_the_environment(tmp_path, monkeypatch):
    store, spec, started = _fake_workers(tmp_path, monkeypatch)
    backend = MLXWorkerClient(spec, execution_variant="current_engine")
    backend.start()
    backend.close()
    assert started == [{"visible": None}]


def test_the_pool_starts_all_devices_or_none(tmp_path, monkeypatch):
    store, spec, _ = _fake_workers(tmp_path, monkeypatch)
    made: list[MLXWorkerClient] = []

    def make(device):
        backend = MLXWorkerClient(spec, execution_variant="current_engine", cuda_device=device)
        if device == "1":
            backend.start = lambda **_k: (_ for _ in ()).throw(BackendUnavailable("no memory"))
        made.append(backend)
        return backend

    with pytest.raises(BackendUnavailable, match="CUDA device 1"):
        start_pool(store, spec, ["0", "1"], make, ProductService)
    assert not made[0].ready  # the device that did come up is stopped again


def test_serve_refuses_gpus_without_a_model(tmp_path):
    with pytest.raises(SystemExit):
        cli.serve(["--no-model", "--gpus", "all", "--state-dir", str(tmp_path / "state")])


def test_health_names_the_knobs_each_card_serves_with_and_whether_they_are_class_a(tmp_path, monkeypatch):
    from ironmule_product.knob_profile import kp1_knobs

    store, spec, _ = _fake_workers(tmp_path, monkeypatch)
    requested = {"knobs": kp1_knobs(), "source": "qualified:KP1-run1"}
    pool = start_pool(store, spec, ["0", "1"], lambda device: MLXWorkerClient(
        spec, execution_variant="current_engine", cuda_device=device, engine_knobs=requested), ProductService)
    try:
        assert pool.health()["knobs"] == {"source": "class_a:qualified:KP1-run1", "values": kp1_knobs(),
                                          "class_a": True}
    finally:
        pool.close()
    plain = start_pool(store, spec, ["0"], lambda device: MLXWorkerClient(
        spec, execution_variant="current_engine", cuda_device=device), ProductService)
    try:
        assert "knobs" not in plain.health()  # the fake reports no knobs unless asked; nothing is invented
    finally:
        plain.close()
