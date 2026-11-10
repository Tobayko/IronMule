"""CPU6 step 2: the stacked decode plan on MLX's CPU device (`ironmule/dense_cpu.py`).

Tiny random Llama and Qwen 3 models. A down projection of K=1536 is the shape where OpenBLAS gave a row alone and
the same row among others different bits (development data in BACKLOG CPU6); the plan pads every product to one
width, so the probe and the served tokens must be identical stacked and alone, on the stock BLAS and, where this
machine can build it, on OpenBLAS (in a child process with the shim preloaded).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models import llama, qwen3

from ironmule import cpu_blas, dense_cpu
from ironmule.dense_plan import DenseBackend
from ironmule.plans import StrictOneShotPlan

ROOT = Path(__file__).resolve().parents[2]
LLAMA = {"model_type": "llama", "hidden_size": 192, "num_hidden_layers": 3, "intermediate_size": 1536,
         "num_attention_heads": 6, "num_key_value_heads": 2, "vocab_size": 300, "rms_norm_eps": 1e-5,
         "rope_theta": 10000.0, "tie_word_embeddings": True}
QWEN3 = {"model_type": "qwen3", "hidden_size": 128, "num_hidden_layers": 2, "intermediate_size": 1536,
         "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32, "vocab_size": 300, "rms_norm_eps": 1e-6,
         "rope_theta": 1000000.0, "tie_word_embeddings": True, "max_position_embeddings": 512}


def tiny(config: dict, seed: int = 0, quantize: bool = False):
    mx.set_default_device(mx.cpu)
    mx.random.seed(seed)
    module = llama if config["model_type"] == "llama" else qwen3
    model = module.Model(module.ModelArgs.from_dict(config))
    model.update(nn.utils.tree_map(lambda p: p * 3 if p.ndim == 2 else p, model.parameters()))
    if quantize:
        nn.quantize(model, group_size=32, bits=4)
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("config", [LLAMA, QWEN3], ids=["llama", "qwen3"])
def test_the_probe_finds_every_width_identical_to_one_row_and_agrees_with_mlx_lm(config):
    plan, record = dense_cpu.install(tiny(config), slots=8, capacity=64)
    assert record["stacked_rows_identical"] is True
    assert record["reference_rel_error"] <= dense_cpu.REFERENCE_TOLERANCE
    assert record["top1_equal"] is True
    assert record["kernels"] == "cpu_float32" and record["slots"] == 8 and record["capacity"] == 64
    assert plan.head.dtype == mx.float32 and plan.keys[0].dtype == mx.float32
    # SP2: a window of positions of one slot, alone and stacked, is bit-identical to one step at a time
    assert record["verify_windows_identical"] is True and plan.verifies_windows is True


def test_a_quantised_checkpoint_is_dequantized_to_float32_first():
    model = tiny(LLAMA, quantize=True)
    plan, record = dense_cpu.install(model, slots=4, capacity=32)
    assert record["dequantized_layers"] > 0
    assert not any(isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)) for _, m in model.named_modules())
    assert plan.trunk.layers[0].mlp.down_proj.weight.dtype == mx.float32


def _serve(backend: DenseBackend, prompts: list[list[int]], steps: int) -> list[list[int]]:
    states, tokens = [], []
    for prompt in prompts:
        state, first = backend.prefill(prompt, StrictOneShotPlan(), backend.plan.capacity)
        states.append(state)
        tokens.append([first])
    for _ in range(steps // backend.readback):
        chains = backend.steps_group(states, [t[-1] for t in tokens], backend.readback)
        backend.complete_chains(chains)
        for i, chain in enumerate(chains):
            for handle in chain:
                token, states[i] = backend.read(handle)
                tokens[i].append(token)
    for state in states:
        backend.release(state)
    return tokens


@pytest.mark.parametrize("config", [LLAMA, QWEN3], ids=["llama", "qwen3"])
def test_each_request_gets_the_same_tokens_stacked_as_alone(config):
    plan, _ = dense_cpu.install(tiny(config, seed=1), slots=8, capacity=64)
    backend = DenseBackend(plan, eos_ids=())
    prompts = [[(r * 37 + j * 11 + 5) % 300 for j in range(2 + r)] for r in range(8)]
    alone = [_serve(backend, [prompt], 8)[0] for prompt in prompts]
    assert _serve(backend, prompts, 8) == alone
    assert _serve(backend, prompts[3:6], 8) == alone[3:6]
    assert backend.free_slots() == 8


def test_the_plan_refuses_what_it_was_not_built_for_and_more_rows_than_slots():
    from mlx_lm.models import gemma3_text

    from tests.engine.test_dense_gemma import TINY

    gemma = gemma3_text.Model(gemma3_text.ModelArgs.from_dict(TINY))
    assert "llama, qwen2, qwen3" in dense_cpu.refuse(gemma)
    with pytest.raises(ValueError, match="serves llama"):
        dense_cpu.install(gemma)
    plan = dense_cpu.CpuDensePlan(tiny(LLAMA), slots=2, capacity=16)
    with pytest.raises(ValueError, match="at most 2 rows"):
        plan.linear(mx.zeros((3, 192)), plan.head)
    with pytest.raises(ValueError, match="own slot position"):  # SP2: one slot, distinct positions is a window
        plan.forward(mx.array([1, 2], mx.uint32), mx.array([0, 0], mx.int32), mx.array([2, 2], mx.int32))
    with pytest.raises(ValueError, match="past its slot"):
        plan.forward(mx.array([1], mx.uint32), mx.array([0], mx.int32), mx.array([16], mx.int32))


def test_the_slot_size_is_the_largest_tier_that_fits_half_the_memory():
    model = tiny(LLAMA)
    dense_cpu.to_float32(model)
    per_token = 2 * 3 * 2 * 32 * 4  # layers, kv heads, head dim, float32
    weights = sum(v.size * 4 for _, v in __import__("mlx.utils", fromlist=["x"]).tree_flatten(model.parameters()))
    total = int((weights + 8 * 2048 * per_token) / dense_cpu.MEMORY_SHARE) + 1
    assert dense_cpu.slot_capacity(model, 8, total=total) == 2048
    assert dense_cpu.slot_capacity(model, 8, total=total - 2) == 1024
    assert dense_cpu.slot_capacity(model, 8, total=1) is None


def test_the_bridge_installs_the_cpu_plan_on_mlx_s_cpu_device():
    from ironmule_product.engine_bridge import install_dense

    plan, record = install_dense(tiny(QWEN3), None, slots=4)
    assert isinstance(plan, dense_cpu.CpuDensePlan) and record["kernels"] == "cpu_float32" and plan.slots == 4


CHILD = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, sys.argv[1])
    from tests.engine.test_dense_cpu import LLAMA, tiny, _serve
    from ironmule import cpu_blas, dense_cpu
    from ironmule.dense_plan import DenseBackend
    plan, record = dense_cpu.install(tiny(LLAMA, seed=2), slots=8, capacity=64)
    backend = DenseBackend(plan, eos_ids=())
    prompts = [[(r * 41 + j * 13 + 7) % 300 for j in range(3 + r)] for r in range(8)]
    alone = [_serve(backend, [p], 8)[0] for p in prompts]
    print(json.dumps({"serving": cpu_blas.active()["serving"], "identical": record["stacked_rows_identical"],
                      "served_alike": _serve(backend, prompts, 8) == alone}))
""")


def test_on_openblas_too_the_stacked_rows_are_identical_to_one_row():
    variables, record = cpu_blas.environment()
    if not variables:
        pytest.skip(f"OpenBLAS shim unavailable here: {record.get('reason')}")
    done = subprocess.run([sys.executable, "-c", CHILD, str(ROOT)], env={**os.environ, **variables},
                          capture_output=True, text=True, timeout=600, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    assert json.loads(done.stdout.strip().splitlines()[-1]) == {"serving": "openblas", "identical": True,
                                                                "served_alike": True}


def test_through_the_product_eight_requests_at_once_get_their_own_answers(tmp_path, monkeypatch):
    """The service, the client, the worker process and the bridge's own `install_dense` (a tiny Llama in
    `dense_cpu_worker.py`): eight requests offered together get the answers each gets alone, on OpenBLAS where this
    machine builds the shim, and health says what serves."""
    import threading

    import ironmule_product.backend as backend_module
    from ironmule_product.backend import MLXWorkerClient
    from ironmule_product.service import ProductService
    from ironmule_product.state import ProductStore
    from ironmule_product.types import GenerationRequest, ModelSpec
    from tests.engine import dense_cpu_worker

    real_popen = subprocess.Popen

    def popen(args, *rest, **kwargs):
        if len(args) > 3 and str(args[3]).endswith("worker.py"):
            args = [sys.executable, "-u", dense_cpu_worker.__file__, *args[4:]]
        return real_popen(args, *rest, **kwargs)

    monkeypatch.setattr(backend_module.subprocess, "Popen", popen)
    monkeypatch.setenv("DENSE_CPU_FAMILY", "llama")
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("DENSE_CPU_LEDGER", str(ledger))
    monkeypatch.delenv("DENSE_CPU_FAULTS", raising=False)
    blas = "openblas" if cpu_blas.environment()[0] else None
    store = ProductStore(tmp_path / "state")
    store.setup("server")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    spec = ModelSpec("local/test", "revision", str(snapshot.resolve()), 1)
    store.register_model(spec)
    texts = [f"question number {i} " * (1 + i % 3) for i in range(8)]

    def start() -> ProductService:
        backend = MLXWorkerClient(spec, execution_variant="current_engine", engine_configuration="dense_stacked",
                                  cpu_blas=blas)
        ready = backend.start()
        assert ready["engine"]["dense_plan"]["kernels"] == "cpu_float32"
        return ProductService(store, backend=backend, spec=spec)

    def ask(service, rid, text, out):
        request = GenerationRequest("local/test", (("user", text),), max_tokens=12, request_id=rid, stream=True)
        out[rid] = "".join(e.get("text", "") for e in service.stream(request) if isinstance(e, dict))

    alone: dict[str, str] = {}
    service = start()
    try:
        for i, text in enumerate(texts):
            ask(service, f"a{i}", text, alone)
    finally:
        service.close()
    assert max(row["rows"] for row in map(json.loads, ledger.read_text().splitlines()) if row["op"] == "step") == 1
    ledger.unlink()
    together: dict[str, str] = {}
    service = start()
    try:
        threads = [threading.Thread(target=ask, args=(service, f"t{i}", text, together)) for i, text in enumerate(texts)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=300)
        health = service.health()
    finally:
        service.close()
    assert [together[f"t{i}"] for i in range(8)] == [alone[f"a{i}"] for i in range(8)]
    stacked = [row["rows"] for row in map(json.loads, ledger.read_text().splitlines()) if row["op"] == "step"]
    assert max(stacked) > 1  # they did share steps
    assert all(alone.values()) and health["failed_requests"] == 0 and health["execution"] == "numeric@dense"
    if blas:
        assert health["cpu_blas"] == {"requested": "openblas", "serving": "openblas", "reason": None}


def test_the_prefill_head_runs_at_one_row_and_matches_mlx_lm_s_last_position():
    """Plan version 2: the prefill's head is one unpadded row (a prefill never shares a step), close to mlx-lm's own
    last-position logits on the same float32 weights, and the first token stacked is still the first token alone."""
    from mlx_lm.models.cache import make_prompt_cache

    model = tiny(LLAMA, seed=4)
    plan, record = dense_cpu.install(model, slots=8, capacity=64)
    assert record["plan_version"] >= 2
    ids = mx.array([[5, 9, 17, 33, 2]])
    mine = plan.prefill(ids, make_prompt_cache(model))
    theirs = model(ids, cache=make_prompt_cache(model))[0, -1:]
    assert mine.shape == (1, 300)
    assert float(mx.max(mx.abs(mine - theirs)) / mx.max(mx.abs(theirs))) < 1e-4
