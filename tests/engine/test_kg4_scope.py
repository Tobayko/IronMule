from types import SimpleNamespace

import pytest

from ironmule import ggml_engine, kg4_scope as kg
from ironmule_product.backend import MLXWorkerClient
from ironmule_product.types import ModelSpec
from test_ggml_engine import _stub_engine


class Masks:
    def __init__(self):
        self.q4 = self.q6 = 0
        self.ironmule_kg_set_mask = lambda v: setattr(self, "q4", v)
        self.ironmule_kg_set_mask6 = lambda v: setattr(self, "q6", v)
        self.ironmule_kg_get_mask = lambda: self.q4
        self.ironmule_kg_get_mask6 = lambda: self.q6


def scoped_engine():
    engine, _, _ = _stub_engine(seqs=16)
    engine.settings = {"kv_unified": True}
    lib = Masks()
    scope = kg.Kg4Scope(lib, [6, 7, 8], engine)
    engine.kg4_scope = scope
    seen = []
    native = engine._decode

    def decode(rows):
        seen.append(lib.q4)
        return native(rows)

    engine._decode = decode
    return engine, lib, seen


def jobs(limits):
    return [ggml_engine.Job([1 + i, 2, 3], limit) for i, limit in enumerate(limits)]


def test_wide_low_wide_groups_and_straggler_on_one_loaded_model():
    engine, lib, seen = scoped_engine()
    for limits, expected in (([4, 4, 4, 4, 4, 4, 4, 1, 1, 1], {0}), ([2] * 8, {224}), ([3] * 10, {0})):
        seen.clear()
        assert all(job.reason == "length" for job in engine.run(jobs(limits)))
        assert set(seen) == expected and lib.q4 == lib.q6 == 0
    assert engine.kg4_scope.low_runs == 1 and engine.kg4_scope.wide_runs == 2


def test_live_joins_disable_before_compute_and_tail_cannot_reenable():
    engine, lib, seen = scoped_engine()
    joins = iter([[], jobs([1] * 3)])
    engine.run(jobs([5] * 7 + [2]), admit=lambda room: next(joins, []))
    assert seen[0] == 224 and seen[-1] == 0
    assert set(seen[seen.index(0):]) == {0}
    assert lib.q4 == lib.q6 == 0 and engine.kg4_scope.wide_runs == 1


def test_failed_dispatch_releases_mask_and_changed_geometry_stays_stock():
    engine, lib, seen = scoped_engine()
    native = engine._decode

    def fail(rows):
        assert lib.q4 == 224
        raise RuntimeError("decode failed")

    engine._decode = fail
    with pytest.raises(RuntimeError, match="decode failed"):
        engine.run(jobs([2] * 8))
    assert lib.q4 == lib.q6 == 0
    engine._decode = native
    engine.seq_ctx *= 2
    engine.run(jobs([2] * 8))
    assert set(seen) == {0} and not engine.kg4_scope.status()["geometry_matches"]
    seen.clear()
    engine.seq_ctx //= 2
    engine.run(jobs([2] * 8))
    assert set(seen) == {224} and engine.kg4_scope.status()["geometry_matches"]


def test_calibration_locally_rejects_a_losing_bucket_and_never_tests_q6(monkeypatch):
    engine, _, _ = scoped_engine()
    engine.gpu_layers = 999
    lib = Masks()
    monkeypatch.setenv("GGML_CUDA_DISABLE_GRAPHS", "1")
    monkeypatch.setattr(kg.ctypes, "CDLL", lambda path: lib)
    monkeypatch.setattr(ggml_engine, "file_sha256", lambda path: {"model": kg.MODEL_SHA256, "plugin": kg.PLUGIN_SHA256}[path])
    monkeypatch.setattr(ggml_engine, "ggml_devices", lambda _: [{"provider": "cuda", "description": "NVIDIA L40S"}])
    tried = []

    def price(buckets, repeats):
        assert repeats == 1 and buckets == (6, 7, 8) and lib.q6 == 0
        tried.append(lib.q4)
        return {6: .98, 7: .96, 8: .8} if lib.q4 else dict.fromkeys(buckets, 1.)

    engine._step_prices = price
    scope, info = kg.Kg4Scope.calibrate(engine, "model", "plugin")
    assert tried == [0, 224, 224, 0, 0, 224, 224, 0]
    assert lib.q4 == lib.q6 == 0 and scope.buckets == (7, 8)
    assert info["tokens_served"] == [7, 8] and info["q6k_tokens_served"] == []
    for gpu_layers, graphs in ((999, "0"), (32, "1")):
        monkeypatch.setenv("GGML_CUDA_DISABLE_GRAPHS", graphs)
        engine.gpu_layers = gpu_layers
        with pytest.raises(ValueError, match="scoped KG4"):
            kg.Kg4Scope.calibrate(engine, "model", "plugin")


def test_worker_placement_requires_current_process_initial_load_and_complete_line():
    client = MLXWorkerClient(ModelSpec("m", "r", "/tmp/test", 1))
    worker = SimpleNamespace(pid=123)
    client._process = worker

    class Pipe:
        def __init__(self, chunks):
            self.chunks = iter(chunks)

        def read(self, count):
            return next(self.chunks, b"")

    marker = b"load_tensors: offloaded 65/65 layers to GPU\n"
    # A stale worker, an unrequested diagnostic, a non-native prefix or a partial line prove nothing.
    for parts, owner in (([marker], SimpleNamespace(pid=123)), ([marker], None),
                         ([b"x" * 1000, marker], worker), ([marker[:-1]], worker)):
        client._drain_stderr(Pipe(parts), owner)
        assert client.ggml_kg4_placement is None
    client._ready_payload = {"ready": True}
    client._drain_stderr(Pipe([marker]), worker)
    assert client.ggml_kg4_placement is None
    client._ready_payload = None
    client._drain_stderr(Pipe([b"loader start\n", marker[:12], marker[12:32], marker[32:]]), worker)
    assert client.ggml_kg4_placement == {"worker_pid": 123, "offloaded_layers": [65, 65]}
    client._process = None
    client._mark_unusable()
    assert client.ggml_kg4_placement is None
