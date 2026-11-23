"""W4R tuner table (ironmule/w4_kernels.py): which token counts the measured widths cover."""
from ironmule import w4_kernels as wk


def test_the_measured_widths_cover_every_tile_count_up_to_32_tokens():
    # a token count is served by the measurement of its tile count (8 tokens a tile); 17-24 tokens are three tiles
    assert {-(-t // 8) for t in wk.TOKENS if t > 1} == {1, 2, 3, 4}


def test_every_token_count_from_1_to_32_gets_a_table_line():
    entries = [{"type": wk.TYPE_Q4_K, "N": 25600, "K": 5120, "B": b, "variant": 7, "ksplit": 1} for b in wk.TOKENS]
    lines = wk.table_lines({"entries": entries}).split()
    served = sorted(int(lines[i + 3]) for i in range(0, len(lines), 6))
    assert served == list(range(1, 33))
    without_24 = [e for e in entries if e["B"] != 24]  # the tuner before the fix
    lines = wk.table_lines({"entries": without_24}).split()
    assert not {int(lines[i + 3]) for i in range(0, len(lines), 6)} & set(range(17, 25))


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
