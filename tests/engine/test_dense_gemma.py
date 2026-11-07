"""KB2's Gemma 3 dense plan, checked off CUDA: its decode step against mlx-lm's own Gemma 3 on a tiny random
4-bit model, with FA2's semantics in plain MLX (`reference_attention`). The CUDA kernels and their bit-exactness
are checked on the hardware (`tools/cuda_probe.py --plan kb2`)."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from mlx_lm.models import gemma3_text  # noqa: E402
from mlx_lm.models.cache import KVCache  # noqa: E402

from ironmule import dense_gemma as dg  # noqa: E402

TINY = {"model_type": "gemma3_text", "hidden_size": 64, "num_hidden_layers": 6, "intermediate_size": 128,
        "num_attention_heads": 4, "head_dim": 32, "num_key_value_heads": 2, "vocab_size": 256,
        "sliding_window": 8, "sliding_window_pattern": 3, "query_pre_attn_scalar": 32,
        "rope_scaling": {"factor": 8.0, "rope_type": "linear"}}


def tiny_model(seed: int = 0) -> gemma3_text.Model:
    mx.random.seed(seed)
    model = gemma3_text.Model(gemma3_text.ModelArgs.from_dict(TINY))
    model.sanitize({})  # no lm_head weight: the head is tied, as in mlx-community's Gemma 3
    model.update(nn.utils.tree_map(lambda p: (p * 4).astype(mx.bfloat16) if p.ndim == 2 else p.astype(mx.bfloat16),
                                   model.parameters()))
    nn.quantize(model, group_size=32, bits=4)
    mx.eval(model.parameters())
    return model


def plan_of(model, slots=4, capacity=32):
    return dg.GemmaDensePlan(model, slots=slots, capacity=capacity, attention=dg.reference_attention, native=False)


def reference_decode(model, prompt, steps):
    """mlx-lm's own Gemma 3 (its rotating sliding-window caches), greedy: the tokens and each step's logits."""
    cache = model.make_cache()
    logits = model(mx.array(prompt)[None], cache=cache)[0, -1].astype(mx.float32)
    tokens, rows = [int(mx.argmax(logits).item())], [logits]
    for _ in range(steps):
        logits = model(mx.array([[tokens[-1]]]), cache=cache)[0, -1].astype(mx.float32)
        rows.append(logits)
        tokens.append(int(mx.argmax(logits).item()))
    return tokens, rows


def plan_decode(plan, prompts, feeds, steps):
    """The plan with every prompt in its own slot, stacked, fed `feeds[r]` (the reference's tokens)."""
    first = []
    for slot, prompt in enumerate(prompts):
        cache = plan.new_cache()
        first.append(plan.prefill(mx.array(prompt)[None], cache)[0])
        plan.write_prefill(slot, [c.state for c in cache])
    rows = [[f] for f in first]
    slots = mx.arange(len(prompts), dtype=mx.int32)
    for t in range(steps):
        tokens = mx.array([feeds[r][t] for r in range(len(prompts))])
        positions = mx.array([len(p) + t for p in prompts], dtype=mx.int32)
        logits = plan.forward(tokens, slots, positions)
        for r in range(len(prompts)):
            rows[r].append(logits[r])
    return rows


def close(a, b, tol=0.08):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return float(mx.max(mx.abs(a - b)).item()) <= tol * max(1.0, float(mx.max(mx.abs(b)).item()))


def test_decode_matches_mlx_lm_across_the_window():
    """Prompts shorter and longer than the window, decoded past it: the plan's logits follow mlx-lm's own model
    (to bfloat16 noise; the reference attention is not FA2's arithmetic) and greedy picks agree."""
    model = tiny_model()
    prompts = [[3, 5, 7], [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]]  # 3 and 11 tokens, window 8
    steps = 12
    references = [reference_decode(model, p, steps) for p in prompts]
    rows = plan_decode(plan_of(model), prompts, [ref[0] for ref in references], steps)
    for r, (tokens, logits) in enumerate(references):
        assert len(rows[r]) == len(logits)
        for t, (mine, theirs) in enumerate(zip(rows[r], logits)):
            assert close(mine, theirs), (r, t)
        picks = [int(mx.argmax(row).item()) for row in rows[r]]
        assert sum(a == b for a, b in zip(picks, tokens)) >= len(tokens) - 1


def test_window_limits_what_a_sliding_layer_reads():
    """A key outside the window must not move the output: poisoning slot positions before the window leaves a
    sliding layer's attention unchanged and changes a global layer's."""
    q = mx.random.normal((1, 4, 32)).astype(mx.bfloat16)
    keys = mx.random.normal((1 * 2 * 32, 32)).astype(mx.bfloat16)
    values = mx.random.normal((1 * 2 * 32, 32)).astype(mx.bfloat16)
    poisoned_v = values.reshape(1, 2, 32, 32)
    poisoned_v = mx.concatenate([mx.full((1, 2, 4, 32), 100.0, mx.bfloat16), poisoned_v[:, :, 4:]], axis=2).reshape(-1, 32)
    args = (mx.array([12]), mx.array([0]), 2, 32, 32 ** -0.5)
    sliding = dg.reference_attention(q, keys, values, *args[:-1], args[-1], 8)
    assert mx.array_equal(sliding, dg.reference_attention(q, keys, poisoned_v, *args[:-1], args[-1], 8))
    assert not mx.array_equal(dg.reference_attention(q, keys, values, *args[:-1], args[-1], 0),
                              dg.reference_attention(q, keys, poisoned_v, *args[:-1], args[-1], 0))


def test_slots_and_positions_are_independent():
    """The same prompt in another slot, beside other rows, gives the same logits as alone (the reference
    attention reads each row separately, so this checks the slot and position bookkeeping)."""
    model = tiny_model(1)
    prompt = [9, 8, 7, 6, 5]
    plan = plan_of(model)
    alone = plan_decode(plan, [prompt], [[1, 2, 3, 4]], 4)[0]
    plan = plan_of(model)
    stacked = plan_decode(plan, [[4, 4], prompt, [1, 2, 3, 4, 5, 6, 7, 8, 9]], [[5] * 4, [1, 2, 3, 4], [6] * 4], 4)[1]
    for a, b in zip(alone, stacked):
        assert close(a, b, tol=0.02)


def test_refusals():
    model = tiny_model()
    assert dg.refuse(model, native=False) is None
    assert "native" in dg.refuse(model, native=True)
    qwen_like = type("M", (), {"args": type("A", (), {"model_type": "qwen3"})()})()
    assert "Gemma 3" in dg.refuse(qwen_like)
    with pytest.raises(ValueError):
        plan_of(model).forward(mx.zeros((9,), mx.uint32), mx.zeros((9,), mx.int32), mx.zeros((9,), mx.int32))


def test_prefill_cache_is_in_position_order_past_the_window():
    """`new_cache` keeps every position (plain caches), so a prompt longer than the window still fills its slot."""
    model = tiny_model()
    plan = plan_of(model)
    cache = plan.new_cache()
    plan.prefill(mx.array(list(range(1, 15)))[None], cache)
    assert all(isinstance(c, KVCache) and c.state[0].shape[2] == 14 for c in cache)
    plan.write_prefill(2, [c.state for c in cache])


def test_slot_capacity_picks_the_largest_tier_that_fits():
    model = tiny_model()
    assert dg.slot_capacity(model, None) == dg.SLOT_TIERS[-1]
    assert dg.slot_capacity(model, {"total_memory": 16 * 1024 ** 3}) == dg.SLOT_TIERS[0]
    assert dg.slot_capacity(model, {"total_memory": 1024}) is None


def test_the_products_dense_backend_serves_the_gemma_plan():
    """`DenseBackend` (the executor's backend behind `--engine dense`) takes the Gemma plan as it is: its prefill
    uses the plan's own position-ordered cache, and stacked steps follow the plan's own forward."""
    from ironmule.dense_plan import DenseBackend
    from ironmule.plans import StrictOneShotPlan

    model = tiny_model(2)
    backend = DenseBackend(plan_of(model), eos_ids=(255,), readback=2)
    prompts = [[3, 4, 5], list(range(20, 32))]  # the second is past the window
    states, firsts = zip(*(backend.prefill(p, StrictOneShotPlan(), backend.capacity_for([len(p)], 8)) for p in prompts))
    assert backend.free_slots() == 2 and [s.offset for s in states] == [3, 12]
    chains = backend.steps_group(list(states), list(firsts), 3)
    backend.complete_chains(chains)
    picks = [[backend.read(h)[0] for h in chain] for chain in chains]
    plan = plan_of(model)
    expected = plan_decode(plan, prompts, [[f, *p[:2]] for f, p in zip(firsts, picks)], 3)
    for r in range(2):
        assert [int(mx.argmax(row).item()) for row in expected[r][1:]] == picks[r]
    for state in states:
        backend.release(state)
    assert backend.free_slots() == 4


def test_the_product_bridge_picks_the_gemma_plan_for_a_4bit_gemma(monkeypatch):
    """`--engine dense` on a 4-bit Gemma 3 installs the native row kernel and then KB2's plan; any other model
    goes to KB1's plan, which refuses what it was not built for."""
    from ironmule import cuda_native, dense_plan
    from ironmule_product import engine_bridge

    calls = []
    monkeypatch.setattr(cuda_native, "install", lambda model, info: calls.append("native") or {"modules": 1})
    monkeypatch.setattr(dg, "refuse", lambda model, native=True: None if not native or "native" in calls
                        else "install the native plan first")
    monkeypatch.setattr(dg, "install", lambda model, info, slots=8: calls.append(("gemma", slots))
                        or ("plan", {"capacity": 1024}))
    monkeypatch.setattr(dense_plan, "install", lambda model, info, slots=8: calls.append(("qwen", slots))
                        or ("qplan", {}))
    plan, record = engine_bridge.install_dense(tiny_model(), {"total_memory": 1})
    assert plan == "plan" and calls == ["native", ("gemma", 8)]
    assert record["model_family"] == "gemma3" and record["native"] == {"modules": 1}
    calls.clear()
    engine_bridge.install_dense(tiny_model(), {"total_memory": 1}, slots=6)  # AG5: the agent's width
    assert calls == ["native", ("gemma", 6)]
    monkeypatch.setattr(dg, "refuse", lambda model, native=True: "the Gemma dense plan serves Gemma 3 checkpoints only")
    calls.clear()
    assert engine_bridge.install_dense(object(), None, slots=5)[0] == "qplan" and calls == [("qwen", 5)]


class _Tok:
    """A stand-in tokenizer inside the tiny model's vocabulary (for the gate text and speed prompts)."""

    eos_token_id = 255

    def encode(self, text):
        return [ord(c) % 200 + 1 for c in text]


def test_fa2_v2_launches_a_warp_per_query_head_and_v1_is_unchanged(monkeypatch):
    """KB3: version 2 runs G warps where version 1 ran one (G query heads per KV head); version 1's launch and
    identity (KB2's evidence is bound to it) do not move, and the plan still serves version 1 until KB3-0 passes."""
    import pytest

    launched = []

    def kernel(name, scale):
        def run(inputs, template, grid, threadgroup, output_shapes, output_dtypes):
            launched.append((name, grid[0], dict(template)))
            return [mx.zeros(shape, dtype=dtype) for shape, dtype in zip(output_shapes, output_dtypes)]
        return run

    monkeypatch.setattr(dg, "_kernel", kernel)
    b, kv, h, d, capacity = 3, 8, 16, 256, 1024
    q = mx.zeros((b, h, d), dtype=mx.bfloat16)
    pool = mx.zeros((4 * kv * capacity, d), dtype=mx.bfloat16)
    lengths, slots = mx.array([5, 300, 1024]), mx.array([0, 1, 2])
    for version in (1, 2):
        dg.fa2(q, pool, pool, lengths, slots, kv, capacity, 0.0625, 0, version=version)
    nc = capacity // dg.FA2_CHUNK
    (one, grid_one, _), _, (two, grid_two, template), _ = launched
    assert (one, two) == ("partial", "partial_v2") and template["G"] == 2
    assert grid_one == -(-(b * kv * nc * 32) // 256) * 256
    assert grid_two == -(-(b * kv * 2 * nc * 32) // 256) * 256
    with pytest.raises(ValueError, match="unknown FA2 version"):
        dg.fa2(q, pool, pool, lengths, slots, kv, capacity, 0.0625, 0, version=4)
    assert dg.FA2_VERSION == 1
    assert dg.KERNEL_SHA256 == "54b1d3a2e975305d03a81915bc1f4d67be9332ddfdc3487e82b0b70d36dc1c7f"
    assert dg.KERNEL_V2_SHA256 != dg.KERNEL_SHA256


def test_fa2_v3_launches_like_v2_from_its_own_source(monkeypatch):
    """KB3-1: version 3 is version 2's layout with the row loop bounded by the chunk's valid rows; it gets its own
    kernel and identity, and the plan still serves version 1."""
    launched = []

    def kernel(name, scale):
        def run(inputs, template, grid, threadgroup, output_shapes, output_dtypes):
            launched.append((name, grid[0]))
            return [mx.zeros(shape, dtype=dtype) for shape, dtype in zip(output_shapes, output_dtypes)]
        return run

    monkeypatch.setattr(dg, "_kernel", kernel)
    q = mx.zeros((1, 16, 256), dtype=mx.bfloat16)
    pool = mx.zeros((8 * 1024, 256), dtype=mx.bfloat16)
    for version in (2, 3):
        dg.fa2(q, pool, pool, mx.array([12]), mx.array([0]), 8, 1024, 0.0625, 0, version=version)
    assert [name for name, _ in launched] == ["partial_v2", "combine", "partial_v3", "combine"]
    assert launched[0][1] == launched[2][1], "the same warps as version 2"
    assert len({dg.KERNEL_SHA256, dg.KERNEL_V2_SHA256, dg.KERNEL_V3_SHA256}) == 3 and dg.FA2_VERSION == 1
    assert "for (int r = 0; r < count; ++r)" in dg.PARTIAL_V3 and "if (second)" in dg.PARTIAL_V3
