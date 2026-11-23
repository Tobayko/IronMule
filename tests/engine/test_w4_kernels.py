"""W4 cache identity and corruption recovery."""
from ironmule import w4_kernels as wk


def test_w4_cache_verifies_artifacts_and_device_binding(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    gguf = tmp_path / "small.gguf"
    gguf.write_bytes(b"model")
    identity = {"cc": "90", "driver": "fixture", "nvcc": "toolkit", "devices": [{"uuid": "first"}]}
    calls = []
    monkeypatch.setattr(wk, "nvcc_path", lambda: "nvcc")
    monkeypatch.setattr(wk, "compute_capability", lambda: "90")
    monkeypatch.setattr(wk, "fingerprint", lambda: dict(identity))
    def build(out, _arch):
        path = out / "binary"
        path.write_bytes(b"compiled for " + identity["devices"][0]["uuid"].encode())
        return path
    def tune(*args, **kwargs):
        calls.append(1)
        return {"entries": [{"type": 12, "N": 32, "K": 256, "B": 8, "variant": 1, "ksplit": 1}]}
    monkeypatch.setattr(wk, "build", build)
    monkeypatch.setattr(wk, "tune", tune)
    def ensure():
        return wk.ensure(str(gguf), tmp_path / "cache")
    with ThreadPoolExecutor(2) as workers:
        first, second = list(workers.map(lambda _: ensure(), range(2)))
    assert first == second and len(calls) == 1
    first[1].write_text("damaged")
    assert ensure() == first and len(calls) == 2
    first[0].write_bytes(b"damaged")
    assert ensure() == first and len(calls) == 3
    identity["devices"] = [{"uuid": "new"}]
    assert ensure() != first and len(calls) == 4
    monkeypatch.setattr(wk, "tune", lambda *a, **k: {"entries": []})
    identity["devices"] = [{"uuid": "unqualified"}]
    assert ensure() is None
    monkeypatch.setattr(wk, "tune", lambda *a, **k: 1 / 0)
    assert ensure() is None
