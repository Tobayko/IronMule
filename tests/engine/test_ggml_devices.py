"""Bind the adaptive GGML server to real providers and memory in its worker."""
import json
import os

import pytest

import ironmule_devices as devices
from ironmule import ggml_engine as engine
from tools import build_ggml


def test_inventory_recognizes_google_tpu_pci_not_arbitrary_google_device(tmp_path, monkeypatch):
    root = tmp_path / "bus/pci/devices"
    for slot, vendor, chip in [("01", "0x1ae0", "0x0063"), ("02", "0x1ae0", "0xffff"), ("03", "0x1002", "0x744c")]:
        node = root / slot
        node.mkdir(parents=True)
        for name, value in (("vendor", vendor), ("device", chip), ("class", "0x030000")):
            (node / name).write_text(value)
    monkeypatch.setenv("PJRT_DEVICE", "TPU")
    rows = devices.host_accelerators(tmp_path)
    assert [(x["source"], x["vendor"]) for x in rows] == [("pci", "tpu"), ("pci", "amd"),
                                                         ("allocation-environment", "tpu")]
    assert rows[0]["model"] == "v5e"


def test_gpu_cache_uses_provider_model_and_masks_but_not_free_memory(monkeypatch):
    monkeypatch.setattr(devices, "host_accelerators", list)
    monkeypatch.setattr(devices, "_nvidia_identity", list)
    row = {"provider": "rocm", "description": "RX 7900", "type": "gpu", "total_memory": 24, "free_memory": 23}
    base = devices.runtime_binding((row,))
    assert base == devices.runtime_binding(({**row, "free_memory": 3},))
    assert base != devices.runtime_binding(({**row, "provider": "cuda"},))
    assert base != devices.runtime_binding(({**row, "description": "Radeon VII"},))
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1")
    assert base != devices.runtime_binding((row,))


def test_rocm_widths_use_registry_vram_and_skip_cuda_probes(monkeypatch, tmp_path):
    rows = ({"provider": "rocm", "name": "ROCm0", "free_memory": 8 << 30, "total_memory": 12 << 30},)
    monkeypatch.setattr(engine, "ggml_devices", lambda L=None: rows)
    assert engine.gpu_memory_bytes() == (8 << 30, 12 << 30)
    assert engine.gpu_memory_bytes("0") is None
    assert not engine.cuda_kernel_device()
    gguf = tmp_path / "model.gguf"
    gguf.touch()
    monkeypatch.setattr(engine, "gpu_layers", lambda: 999)
    monkeypatch.setattr(engine, "gpu_memory_bytes", lambda device: None)
    refused = engine.choose_sequences(str(gguf))
    assert refused["sequences"] == 1 and refused["memory"] == "gpu-unknown"


def test_selected_cuda_uuid_is_resolved_by_the_selected_child_registry(monkeypatch):
    import subprocess
    from types import SimpleNamespace
    rows = [{"provider": "cuda", "free_memory": 4, "total_memory": 8},
            {"provider": "cuda", "free_memory": 10, "total_memory": 24}]
    monkeypatch.setattr(engine, "ggml_devices", lambda: rows)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    def run(*_, env, **kwargs):
        assert env["CUDA_VISIBLE_DEVICES"] == "GPU-board2"
        return SimpleNamespace(stdout=json.dumps(rows[1:]))
    monkeypatch.setattr(subprocess, "run", run)
    assert engine.gpu_memory_bytes("GPU-board2") == (10, 24)


def test_hip_build_uses_local_upstream_backend_and_excludes_cuda_patches(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(build_ggml, "fetch", lambda dest: dest)
    monkeypatch.setattr(build_ggml, "patch", lambda tree, names: calls.append(names) or names)
    monkeypatch.setattr(build_ggml, "run", lambda cmd, **kw: calls.append(cmd))
    build_ggml.main(["--dest", str(tmp_path), "--rocm", "gfx1100,gfx942"])
    assert calls[0] == ["fa1", "row1", "am1"]
    assert "-DGGML_HIP=ON" in calls[1] and "-DAMDGPU_TARGETS=gfx1100;gfx942" in calls[1]
    assert "-DGGML_CUDA=ON" not in calls[1]
    with pytest.raises(SystemExit):
        build_ggml.main(["--rocm", "75"])


def test_rocm_target_detection_discards_cpu_and_deduplicates_visible_chips(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(build_ggml.shutil, "which", lambda name: "/opt/rocm/bin/" + name)
    def run(command, **kwargs):
        assert command == ["/opt/rocm/bin/rocm_agent_enumerator", "-t", "GPU"]
        return SimpleNamespace(stdout="gfx000\ngfx942:sramecc+:xnack-\ngfx1100\ngfx942\n")
    monkeypatch.setattr(build_ggml.subprocess, "run", run)
    assert build_ggml.rocm_targets() == "gfx1100,gfx942"


def test_qualification_and_worker_each_receive_the_same_single_cuda_device(monkeypatch, tmp_path):
    import os
    from argparse import Namespace
    from types import SimpleNamespace

    from ironmule_product import backend, cli
    from ironmule_product.errors import BackendUnavailable
    from ironmule_product.types import ModelSpec

    args = Namespace(gguf=tmp_path / "m.gguf", draft_gguf=None, speculate="off", agent="on", model=None)
    spec = ModelSpec("local/m", "deadbeef", str(tmp_path / "m.gguf"), 123)
    store = SimpleNamespace(root=tmp_path)
    sizes = {"GPU-a": 4, "GPU-b": 16}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-b,GPU-a")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "")
    monkeypatch.setenv("IRONMULE_GGML_SEQS", "8")
    monkeypatch.setenv("IRONMULE_KG_LIB", "/initial.so")

    def qualify(command, *, env, input, **kw):
        assert command[-1] == "ironmule_product.pinned_ggml"
        assert env["ROCR_VISIBLE_DEVICES"] == "" and json.loads(input)["spec"] == spec.as_dict()
        card = env["CUDA_VISIBLE_DEVICES"]
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            "ggml_gate": {"passes": True, "card": card}, "engine_configuration": "ggml",
            "ggml_gguf": str(args.gguf), "ggml_sequences": sizes[card], "ggml_kernel_library": f"/{card}.so"}))
    seen = []
    def launch(command, *, env, **kwargs):
        card = env["CUDA_VISIBLE_DEVICES"]
        gate = json.loads(command[command.index("--ggml-gate") + 1])
        seen.append((card, gate["card"], env["IRONMULE_GGML_SEQS"], env["IRONMULE_KG_LIB"]))
        raise OSError("stop after capturing the worker environment")
    monkeypatch.setattr(cli.subprocess, "run", qualify)
    monkeypatch.setattr(backend.subprocess, "Popen", launch)
    for card, capacity in sizes.items():
        options = cli._pinned_ggml_options(card, args, spec, store)
        worker = backend.MLXWorkerClient(spec, cuda_device=card, execution_variant="current_engine", **options)
        assert worker.batch_capacity == capacity
        with pytest.raises(BackendUnavailable):
            worker.start()
    assert seen == [("GPU-a", "GPU-a", "4", "/GPU-a.so"), ("GPU-b", "GPU-b", "16", "/GPU-b.so")]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-b,GPU-a"
    assert os.environ["IRONMULE_KG_LIB"] == "/initial.so"


def test_unpinned_qualification_restores_parent_paths_on_failure_or_success(monkeypatch):
    import os

    from ironmule_product import cli

    monkeypatch.delenv("IRONMULE_KG_LIB", raising=False)
    monkeypatch.setenv("IRONMULE_GGML_SEQS", "4")
    def measure(*args, pinned):
        assert not pinned
        os.environ["IRONMULE_KG_LIB"] = "/only-this-model.so"
        os.environ["IRONMULE_GGML_SEQS"] = "32"
        return {"ggml_sequences": 32}
    monkeypatch.setattr(cli, "_measure_ggml_options", measure)
    assert cli._local_ggml_options(None, None, None)["ggml_sequences"] == 32
    assert "IRONMULE_KG_LIB" not in os.environ and os.environ["IRONMULE_GGML_SEQS"] == "4"


def test_cold_tune_respects_a_worker_configured_with_two_sequences(monkeypatch):
    from types import SimpleNamespace
    widths = []
    class SmallEngine:
        seqs = 2
        memory = None
        L = SimpleNamespace(llama_memory_clear=lambda *args: None)
        def __init__(self, *args, **kwargs):
            pass
        def tokenize(self, _):
            return [1, 2, 3]
        def _decode(self, rows):
            width = max(seq for _, _, seq, _ in rows) + 1
            assert width <= self.seqs
            widths.append(width)
        def close(self):
            pass
    monkeypatch.setattr(engine, "GgmlEngine", SmallEngine)
    report = engine.autotune("model", steps=1, repeats=1)
    assert {1, 2} == set(widths)
    assert "2" in report["measured"][0]["tokens_per_s"]


def test_pinning_refuses_zero_gpu_layers_and_all_layers_normalizes(monkeypatch):
    from ironmule_product import cli
    from ironmule_product.errors import InvalidRequest

    monkeypatch.setattr(engine, "available", lambda: None)
    monkeypatch.setattr(engine, "gpu_offload", lambda L=None: True)
    monkeypatch.setattr(engine, "GPU_LAYERS", "0")
    with pytest.raises(InvalidRequest, match="selects the CPU"):
        cli._local_ggml_options(None, None, None, pinned=True)
    monkeypatch.setattr(engine, "GPU_LAYERS", "-1")
    assert engine.gpu_layers() == 999
    monkeypatch.setattr(engine, "GPU_LAYERS", "-3")
    with pytest.raises(ValueError, match="nonnegative"):
        engine.gpu_layers()


def test_failed_generated_kernel_requalifies_real_stock_mode(monkeypatch, tmp_path, capsys):
    from ironmule import ggml_engine, ggml_gate_diagnostics
    from ironmule_product import cli

    monkeypatch.setenv('IRONMULE_KG_LIB', '/local/candidate.so')
    monkeypatch.delenv('IRONMULE_KG5', raising=False)
    calls = []

    def gate(snapshot, gguf, *, cache):
        assert (snapshot, gguf, cache) == ('model.gguf', 'model.gguf', tmp_path)
        candidate = os.environ.get('IRONMULE_KG_LIB')
        calls.append(candidate)
        if candidate:
            return {'passes': False, 'diagnostic': {'schema': ggml_gate_diagnostics.SCHEMA,
                    'stage': 'candidate', 'code': 'cuda_kernel_error'}}
        return {'passes': True, 'ratio': 1.0, 'settings': {'flash_attn': False}}

    monkeypatch.setattr(ggml_engine, 'gate', gate)
    checked = cli._qualify_ggml_kernels('model.gguf', 'model.gguf', tmp_path)
    assert calls == ['/local/candidate.so', None]
    assert checked['kernel_fallback']['serving'] == 'stock'
    assert checked['settings'] == {'flash_attn': False}
    assert [row['passes'] for row in checked['kernel_fallback']['attempts']] == [False, True]
    assert 'cuda_kernel_error' in capsys.readouterr().out


@pytest.mark.parametrize(("hook", "configured", "enabled", "expected"), [
    (False, "", "1", ""), (False, "/explicit.so", "1", ""),
    (True, "", "1", "/generated.so"), (None, "", "1", "/generated.so"),
    (True, "/explicit.so", "1", "/explicit.so"), (None, "", "0", ""),
])
def test_qualifies_only_kernels_the_cuda_backend_can_consume(monkeypatch, tmp_path, hook, configured, enabled, expected):
    from types import SimpleNamespace

    from ironmule import ggml_binary, kernel_gen
    from ironmule_product import cli

    monkeypatch.setattr(engine, "available", lambda: None)
    monkeypatch.setattr(engine, "gpu_layers", lambda: 999)
    monkeypatch.setattr(engine, "cuda_kernel_device", lambda: True)
    monkeypatch.setattr(ggml_binary, "cuda_kg1_hook", lambda: hook if enabled == "1" else pytest.fail("disabled"))
    generated, qualified = [], []

    def generate(*args, **kwargs):
        generated.append(True)
        return "/generated.so"

    def qualify(*args):
        qualified.append(os.environ.get("IRONMULE_KG_LIB") or "")
        return {"passes": True}

    monkeypatch.setattr(kernel_gen, "ensure", generate)
    monkeypatch.setattr(cli, "_qualify_ggml_kernels", qualify)
    monkeypatch.setenv("IRONMULE_GGML_SEQS", "1")
    monkeypatch.setenv("IRONMULE_KG", enabled)
    monkeypatch.setenv("IRONMULE_KG_LIB", configured)
    args = SimpleNamespace(gguf=tmp_path / "model.gguf", draft_gguf=None, speculate="off", agent="on", model=None)
    spec = SimpleNamespace(snapshot_path=args.gguf, revision="pinned")
    options = cli._local_ggml_options(args, spec, SimpleNamespace(root=tmp_path))
    assert qualified == [expected] and options["ggml_kernel_library"] == expected
    assert bool(generated) == (expected == "/generated.so")
    assert os.environ["IRONMULE_KG_LIB"] == configured
    if hook is False:
        assert options["ggml_gate"]["kernel_generation"]["reason"] == "cuda_backend_has_no_kg1_hook"
    elif enabled == "0":
        assert options["ggml_gate"]["kernel_generation"]["reason"] == "IRONMULE_KG=0"


def test_stock_or_independent_gate_failure_never_admits_or_retries(monkeypatch, tmp_path):
    from ironmule import ggml_engine, ggml_gate_diagnostics
    from ironmule_product import cli
    from ironmule_product.errors import InvalidRequest

    calls = []
    def failed(snapshot, gguf, *, cache):
        calls.append(os.environ.get('IRONMULE_KG_LIB'))
        return {'passes': False, 'diagnostic': {'schema': ggml_gate_diagnostics.SCHEMA,
                'stage': 'reference', 'code': 'out_of_memory'}}

    monkeypatch.setattr(ggml_engine, 'gate', failed)
    monkeypatch.setenv('IRONMULE_KG_LIB','/local/candidate.so')
    with pytest.raises(InvalidRequest, match='reference: out_of_memory'):
        cli._qualify_ggml_kernels('model.gguf', 'model.gguf', tmp_path)
    assert calls == ['/local/candidate.so']
    monkeypatch.delenv('IRONMULE_KG_LIB')
    with pytest.raises(InvalidRequest, match='reference: out_of_memory'):
        cli._qualify_ggml_kernels('model.gguf', 'model.gguf', tmp_path)
    assert calls == ['/local/candidate.so', None]
