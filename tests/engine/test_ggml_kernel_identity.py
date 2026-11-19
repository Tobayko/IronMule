"""GGUF cache identities for stock, generated kernels and native device placement."""
import json
import subprocess
from types import SimpleNamespace

from ironmule import ggml_binary
from ironmule import ggml_engine as ge


def test_gate_key_with_stable_native_runtime_binds_offload_split_and_kernel(monkeypatch, tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"test GGUF")
    library = tmp_path / "custom.so"
    monkeypatch.setattr(ge, "ggml_devices", lambda: ())
    monkeypatch.setattr(ggml_binary, "identity", lambda: {"sha256": "fixed-native-runtime"})
    monkeypatch.delenv("IRONMULE_KG_LIB", raising=False)
    monkeypatch.delenv("IRONMULE_KG5", raising=False)
    monkeypatch.setenv("IRONMULE_GGML_SPLIT", "layer")
    monkeypatch.setenv("IRONMULE_GGML_GPU_LAYERS", "0")

    def key():
        return ge.gate_key(str(gguf), str(gguf))

    stock_cpu = key()
    monkeypatch.setenv("IRONMULE_GGML_GPU_LAYERS", "999")
    stock_gpu = key()
    monkeypatch.setenv("IRONMULE_GGML_SPLIT", "row")
    split_gpu = key()
    monkeypatch.setenv("IRONMULE_KG_LIB", str(library))
    unresolved = key()
    assert ge._kernel_digest() is None
    library.write_bytes(b"one")
    first = key()
    library.write_bytes(b"two")
    second = key()
    assert len({stock_cpu, stock_gpu, split_gpu, unresolved, first, second}) == 6


def test_width_cache_requires_current_generated_library_bytes(monkeypatch, tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"model")
    library = tmp_path / "custom.so"
    monkeypatch.setenv("IRONMULE_KG_LIB", str(library))
    monkeypatch.setattr(ge, "machine_key", lambda: "fixed")
    monkeypatch.setattr(ge, "gpu_layers", lambda: 0)
    monkeypatch.setattr(ge, "memory_bytes", lambda: (16 << 30, 16 << 30))
    calls = []

    def run(argv, **kwargs):
        calls.append(argv[2])
        data = {"cell_bytes": 1024}
        if argv[2] == ge.WIDTH_CHILD:
            data.update(rates={str(w): 1.0 for w in ge.WIDTHS}, step_s={})
        return SimpleNamespace(returncode=0, stdout=json.dumps(data), stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    cache = tmp_path / "cache"
    library.write_bytes(b"one")
    ge.choose_sequences(str(gguf), cache=cache)
    ge.choose_sequences(str(gguf), cache=cache)
    assert len(calls) == 2
    library.write_bytes(b"two")
    ge.choose_sequences(str(gguf), cache=cache)
    assert len(calls) == 4
    library.unlink()
    ge.choose_sequences(str(gguf), cache=cache)
    ge.choose_sequences(str(gguf), cache=cache)
    assert len(calls) == 8
    assert len(list(cache.glob("width-*.json"))) == 2
