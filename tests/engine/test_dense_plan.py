"""KB1's `dense` numeric plan in the library: refusals, the stacked forward, and its executor through `Runtime`
(no GPU: the CUDA kernels are replaced by CPU stand-ins with the same contract -- each row on its own, attention
reading only up to a row's length in its own slot). Hardware exactness and speed are KB1-1's and KB1-2's."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from ironmule import dense_plan as dp  # noqa: E402
from ironmule.runtime import Engine, Knobs  # noqa: E402
from ironmule.service import DenseStackedMode, Request, Runtime  # noqa: E402


class _Tokenizer:
    eos_token_id = 255
    eos_token_ids = {255}

    def encode(self, text):
        return [ord(c) % 200 + 1 for c in text]

    def decode(self, ids):
        return "".join(chr(32 + i % 90) for i in ids)


def _tiny(head_dim=128):
    from mlx_lm.models import qwen3

    args = qwen3.ModelArgs(model_type="qwen3", hidden_size=128, num_hidden_layers=2, intermediate_size=256,
                           num_attention_heads=2, rms_norm_eps=1e-6, vocab_size=256, num_key_value_heads=1,
                           max_position_embeddings=512, rope_theta=10000.0, head_dim=head_dim, tie_word_embeddings=True)
    model = qwen3.Model(args)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


def _rows(x, w, rows=None):
    flat = x.reshape(-1, x.shape[-1])
    out = mx.concatenate([(flat[i:i + 1].astype(mx.float32) @ w.astype(mx.float32).T).astype(mx.bfloat16)
                          for i in range(flat.shape[0])])
    return out.reshape(*x.shape[:-1], w.shape[0])


def _attention(q, keys, values, lengths, slots, kv_heads, capacity):
    rows, heads, dim = q.shape
    groups = heads // kv_heads
    k4, v4 = keys.reshape(-1, kv_heads, capacity, dim), values.reshape(-1, kv_heads, capacity, dim)
    out = []
    for b in range(rows):
        n, s = int(lengths[b].item()), int(slots[b].item())
        kk = mx.repeat(k4[s, :, :n, :], groups, axis=0).astype(mx.float32)
        vv = mx.repeat(v4[s, :, :n, :], groups, axis=0).astype(mx.float32)
        scores = (q[b].astype(mx.float32)[:, None, :] @ kk.transpose(0, 2, 1))[:, 0, :] * dim ** -0.5
        out.append((mx.softmax(scores, axis=-1)[:, None, :] @ vv)[:, 0, :])
    return mx.stack(out).astype(mx.bfloat16)


def _plan(model, slots=8, capacity=128):
    return dp.DensePlan(model, slots=slots, capacity=capacity, linear=_rows, attention=_attention)


def test_the_plan_refuses_what_it_was_not_built_or_measured_for():
    model = _tiny()
    assert dp.refuse(model) is None
    assert "head dim 128" in dp.refuse(_tiny(head_dim=64))
    other = _tiny()
    other.args.model_type = "llama"
    assert "Qwen 3" in dp.refuse(other)
    import mlx.nn as nn

    quantized = _tiny()
    nn.quantize(quantized, group_size=64, bits=4)
    assert "unquantized" in dp.refuse(quantized)
    with pytest.raises(ValueError, match="compute capability 8"):
        dp.install(model, None)  # no CUDA device here


def test_stacked_rows_in_their_own_slots_give_each_row_its_answer_alone_close_to_the_model():
    from mlx_lm.models.cache import make_prompt_cache

    model = _tiny()
    plan = _plan(model)
    prompts = [[5, 9, 13, 17, 21], [3, 7, 11, 15, 19, 23, 27, 31, 35], [8, 6, 4]]
    firsts = []
    for slot, ids in enumerate(prompts):
        cache = make_prompt_cache(model)
        logits = model(mx.array(ids)[None], cache=cache)
        plan.write_prefill(7 - slot, [c.state for c in cache])  # slots in no particular order
        firsts.append(int(mx.argmax(logits[0, -1]).item()))
        expected = model(mx.array([[firsts[-1]]]), cache=cache)[0, -1]
        alone = plan.forward(mx.array([firsts[-1]], mx.uint32), mx.array([7 - slot]), mx.array([len(ids)]))
        expected = expected.astype(mx.float32)  # a numeric plan: close to the model, not bit-equal
        assert float(mx.max(mx.abs(alone[0] - expected)).item()) < 0.05 * float(mx.max(mx.abs(expected)).item())
    # every slot was written again at the same positions by the single steps above, so stack the same step
    stacked = plan.forward(mx.array(firsts, mx.uint32), mx.array([7, 6, 5]), mx.array([len(p) for p in prompts]))
    for slot, ids in enumerate(prompts):
        alone = plan.forward(mx.array([firsts[slot]], mx.uint32), mx.array([7 - slot]), mx.array([len(ids)]))
        assert mx.array_equal(stacked[slot], alone[0]).item()
    assert dp.probe(plan)["stacked_rows_identical"]


def _runtime(model, plan):
    return Runtime(Engine(model, _Tokenizer(), Knobs()), _Tokenizer(), mode=DenseStackedMode(plan))


def test_the_executor_answers_each_request_as_it_would_alone_and_returns_every_slot():
    model = _tiny()
    plan = _plan(model)
    runtime = _runtime(model, plan)
    prompts = [[5, 9, 13, 17, 21], [3, 7, 11, 15, 19, 23, 27, 31, 35], [8, 6, 4], [40, 41, 42, 43], [99] * 12]
    alone = [runtime.serve([Request(prompt_ids=ids, max_tokens=9)])[0].tokens for ids in prompts]
    together = [r.tokens for r in runtime.serve([Request(prompt_ids=ids, max_tokens=9) for ids in prompts])]
    assert together == alone
    assert runtime.backend.free_slots() == plan.slots
    assert max(runtime.telemetry.realised_widths) == len(prompts)


def test_a_failed_stacked_step_restarts_inside_the_plan_and_changes_no_answer():
    model = _tiny()
    plan = _plan(model)
    runtime = _runtime(model, plan)
    prompts = [[5, 9, 13], [3, 7, 11, 15, 19], [8, 6, 4, 2]]
    expected = [r.tokens for r in runtime.serve([Request(prompt_ids=ids, max_tokens=7) for ids in prompts])]
    original = runtime.backend.step_group
    calls = {"n": 0}

    def flaky(states, tokens):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("injected")
        return original(states, tokens)

    runtime.backend.step_group = flaky
    again = [r.tokens for r in runtime.serve([Request(prompt_ids=ids, max_tokens=7) for ids in prompts])]
    assert again == expected and runtime.telemetry.fallbacks == 1
    assert runtime.backend.free_slots() == plan.slots


def test_a_cancelled_request_frees_its_slot_and_the_others_keep_their_answers():
    from ironmule.executor import TokenStream

    model = _tiny()
    plan = _plan(model)
    runtime = _runtime(model, plan)
    prompts = [[5, 9, 13], [3, 7, 11, 15, 19], [8, 6, 4, 2]]
    expected = [r.tokens for r in runtime.serve([Request(prompt_ids=ids, max_tokens=8) for ids in prompts])]
    seen = []
    stream = TokenStream(seen.append, cancelled=lambda: len(seen) >= 3)
    requests = [Request(prompt_ids=ids, max_tokens=8) for ids in prompts]
    requests[1].stream = stream
    results = runtime.serve(requests)
    assert results[1].stop_reason == "cancelled" and len(results[1].tokens) == 3
    assert [results[0].tokens, results[2].tokens] == [expected[0], expected[2]]
    assert runtime.backend.free_slots() == plan.slots


def test_a_request_that_does_not_fit_a_slot_is_refused_before_any_work():
    model = _tiny()
    runtime = _runtime(model, _plan(model, capacity=16))
    with pytest.raises(ValueError, match="do not fit"):
        runtime.serve([Request(prompt_ids=[1] * 12, max_tokens=8)])
    assert runtime.backend.free_slots() == 8
