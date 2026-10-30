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

    mx.random.seed(7)  # fixed weights: random ones can put two logits within MLX CPU's batch-dependent rounding
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



def test_the_prefill_sends_only_the_last_position_through_the_plans_own_head():
    """Plan version 2: the first token comes from the decode steps' head at one row, not a head over every
    prompt position (17 % of a long prompt's prefill on the T4, HW4); close to the model's own last logits."""
    from mlx_lm.models.cache import make_prompt_cache

    model = _tiny()
    plan = _plan(model)
    rows = []
    original = plan.linear

    def counting(x, w, rows_=None):
        if w is plan.head:
            rows.append(x.size // x.shape[-1])
        return original(x, w)

    plan.linear = counting
    ids = mx.array([[5, 9, 13, 17, 21, 25, 29]])
    logits = plan.prefill(ids, make_prompt_cache(model))
    assert rows == [1] and logits.shape == (1, 256) and logits.dtype == mx.float32
    expected = model(ids, cache=make_prompt_cache(model))[0, -1].astype(mx.float32)
    assert float(mx.max(mx.abs(logits[0] - expected)).item()) < 0.05 * float(mx.max(mx.abs(expected)).item())
    assert dp.PLAN_VERSION == 3


def test_a_prompt_longer_than_a_prefill_chunk_is_prefilled_chunk_by_chunk_and_a_shorter_one_in_one_call(monkeypatch):
    """Plan version 3: a prompt up to `PREFILL_CHUNK` positions is prefilled in one call, exactly as in version 2;
    a longer one chunk by chunk into the same cache, as mlx-lm's own generation prefills, close to one call."""
    from mlx_lm.models.cache import make_prompt_cache

    model = _tiny()
    plan = _plan(model)
    calls, original = [], plan.trunk
    plan.trunk = lambda ids, cache: calls.append(ids.shape[1]) or original(ids, cache=cache)
    monkeypatch.setattr(dp, "PREFILL_CHUNK", 4)
    plan.prefill(mx.array([[5, 9, 13, 17]]), make_prompt_cache(model))
    assert calls == [4]
    calls.clear()
    ids = mx.array([[5, 9, 13, 17, 21, 25, 29, 33, 37, 41]])
    cache = make_prompt_cache(model)
    chunked = plan.prefill(ids, cache)
    assert calls == [4, 4, 2] and cache[0].offset == 10
    monkeypatch.setattr(dp, "PREFILL_CHUNK", 2048)
    whole = plan.prefill(ids, make_prompt_cache(model))
    # Not bit-identical (other matmul shapes round bf16 activations otherwise: 0.4 % on Metal in CI), but close.
    assert float(mx.max(mx.abs(chunked - whole)).item()) < 0.02 * float(mx.max(mx.abs(whole)).item())


def test_the_slot_size_is_the_largest_tier_the_card_holds_beside_the_model():
    """A function of the card and the model alone, so both cards of a machine hold the same requests: on a T4
    (16,106,127,360 bytes as MLX reports it) beside Qwen 3 1.7B bf16 (28 layers, 8 KV heads, 3.44 GB of
    weights) H11's rule gives a slot the product's whole 8,192-token context; the plan keeps 2,048 until H11 (a)
    passes."""
    from types import SimpleNamespace

    class Stand:
        args = SimpleNamespace(num_key_value_heads=8, head_dim=128)
        model = SimpleNamespace(layers=[None] * 28)

        def __init__(self, weights):
            self.weights = weights

        def parameters(self):
            return {"weights": SimpleNamespace(nbytes=self.weights)}

    def card(total):
        return {"total_memory": total, "compute_capability_major": 7}

    t4, tiers = 16_106_127_360, dp.H11_TIERS
    assert dp.slot_capacity(Stand(3_441_149_952), card(t4), tiers=tiers) == 8192  # HW3's measured weights
    assert dp.slot_capacity(Stand(3_441_149_952), card(t4)) == 2048  # until H11 (a) admits the larger slots
    per_token = 2 * 28 * 8 * 128 * 2
    edge = int(dp.SLOT_HEADROOM * t4) - 8 * 8192 * per_token - dp.PREFILL_RESERVE  # the most weights 8192 allows
    assert dp.slot_capacity(Stand(edge), card(t4), tiers=tiers) == 8192
    assert dp.slot_capacity(Stand(edge + 1), card(t4), tiers=tiers) == 4096
    assert dp.slot_capacity(Stand(int(dp.SLOT_HEADROOM * t4)), card(t4), tiers=tiers) is None
    assert dp.slot_capacity(Stand(1), None, tiers=tiers) == 2048  # a card that reports no memory: the smallest

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
    original = runtime.backend.steps_group  # every stacked step, chained or not, goes through it
    calls = {"n": 0}

    def flaky(states, tokens, count):
        calls["n"] += 1
        if calls["n"] == 2 and len(states) > 1:
            raise RuntimeError("injected")
        return original(states, tokens, count)

    runtime.backend.steps_group = flaky
    again = [r.tokens for r in runtime.serve([Request(prompt_ids=ids, max_tokens=7) for ids in prompts])]
    assert again == expected and runtime.telemetry.fallbacks == 1
    assert runtime.backend.free_slots() == plan.slots



def test_a_whole_executor_failure_replays_in_the_same_slots_and_frees_them_after(monkeypatch):
    """`Runtime`'s safety net: the executor's `finally` returns every slot, then the sequential path replays the
    sessions in the slots they still name. That is safe only because the replay prefills nothing, so no other
    request can take a slot in between; this pins it."""
    model = _tiny()
    plan = _plan(model)
    runtime = _runtime(model, plan)
    prompts = [[5, 9, 13], [3, 7, 11, 15, 19], [8, 6, 4, 2]]
    expected = [r.tokens for r in runtime.serve([Request(prompt_ids=ids, max_tokens=7) for ids in prompts])]
    prefills, original_prefill = [], runtime.backend.prefill
    sweeps, original_sweep = {"n": 0}, dp.DenseStackedExecutor._sweep

    def failing_sweep(self):
        sweeps["n"] += 1
        if sweeps["n"] == 2:  # between rounds, outside the per-group fallback
            raise RuntimeError("injected")
        original_sweep(self)

    monkeypatch.setattr(dp.DenseStackedExecutor, "_sweep", failing_sweep)
    monkeypatch.setattr(runtime.backend, "prefill", lambda *a: prefills.append(a) or original_prefill(*a))
    again = runtime.serve([Request(prompt_ids=ids, max_tokens=7) for ids in prompts])
    assert [r.tokens for r in again] == expected and runtime.telemetry.fallbacks == 1
    assert "executor: RuntimeError: injected" in runtime.telemetry.fallback_reasons
    assert len(prefills) == len(prompts) and runtime.backend.free_slots() == plan.slots

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


def test_the_product_bridge_serves_the_dense_plan_at_width_eight_and_lets_joins_take_free_slots():
    from types import SimpleNamespace

    from ironmule_product.engine_bridge import CurrentEngineBridge

    model = _tiny()
    plan = _plan(model)
    runtime = _runtime(model, plan)
    identity = SimpleNamespace(to_dict=lambda: {"identity_sha256": "a" * 64})
    bridge = CurrentEngineBridge(runtime, identity=identity, knobs=Knobs(), profile_available=False,
                                 profile_source="dense_numeric_plan", configuration="dense_stacked",
                                 historical_configuration_candidate=None, dense_plan={"kernel_sha256": dp.KERNEL_SHA256})
    prompts = [[5, 9, 13], [3, 7, 11, 15, 19], [8, 6, 4, 2], [40, 41], [99] * 6, [12, 14], [21, 23, 25], [30] * 9]
    alone = [bridge.generate(ids, 6).tokens for ids in prompts]
    streamed = []
    joins = [([50, 51, 52], 6, streamed.append, None)]  # the worker always streams a joiner
    seen = {"slots": None}

    def admit(capacity):
        seen["slots"] = bridge.free_slots()
        rows, joins[:] = list(joins), []
        return rows

    results = bridge.generate_many(prompts[:7], 6, admit=admit)
    assert [r.tokens for r in results[:7]] == alone[:7]
    assert results[7].tokens == streamed == bridge.generate([50, 51, 52], 6).tokens  # the joiner, as alone
    assert seen["slots"] is not None and bridge.free_slots() == plan.slots
    assert results[0].metadata["dense_plan"]["kernel_sha256"] == dp.KERNEL_SHA256
    eight = bridge.generate_many(prompts, 6)
    assert [r.tokens for r in eight] == alone and eight[0].metadata["max_realised_width"] == 8
    with pytest.raises(ValueError, match="one-shot"):
        bridge.generate_many(prompts[:2], 6, boundaries=[[1], None])


class _Detokenizer:
    def __init__(self):
        self.pending = ""

    def reset(self):
        self.pending = ""

    def add_token(self, token):
        self.pending += chr(32 + token % 90)

    def finalize(self):
        pass

    @property
    def last_segment(self):
        segment, self.pending = self.pending, ""
        return segment


class _ChatTokenizer(_Tokenizer):
    @property
    def detokenizer(self):
        return _Detokenizer()

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return self.encode(messages[0]["content"])


def _worker_world(monkeypatch, slots=8):
    """A real dense bridge (tiny model, CPU stand-ins) behind the worker's batch path, its events captured."""
    import queue
    from types import SimpleNamespace

    from ironmule_product import worker
    from ironmule_product.engine_bridge import CurrentEngineBridge

    model = _tiny()
    plan = _plan(model, slots=slots)
    identity = SimpleNamespace(to_dict=lambda: {"identity_sha256": "a" * 64})
    bridge = CurrentEngineBridge(_runtime(model, plan), identity=identity, knobs=Knobs(), profile_available=False,
                                 profile_source="dense_numeric_plan", configuration="dense_stacked",
                                 historical_configuration_candidate=None, dense_plan={"kernel_sha256": dp.KERNEL_SHA256})
    sent = []
    monkeypatch.setattr(worker, "_emit", sent.append)
    monkeypatch.setattr(worker, "_DETOKENIZER_POOLS", {})
    monkeypatch.setattr(worker, "JOINS", queue.Queue())
    return worker, bridge, plan, _ChatTokenizer(), sent


def _chat(rid, text, max_tokens=6):
    return {"request_id": rid, "model": "m", "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens, "stream": True}


def test_the_worker_streams_a_lone_dense_batch_and_lets_a_request_join_it(monkeypatch):
    """H8 run 1: the client sends the dense engine lone streamed batches (it streams batches), so the worker must
    run its groups live -- streamed, joinable -- as it does the current profile's, not refuse them."""
    import threading

    worker, bridge, plan, tokenizer, sent = _worker_world(monkeypatch)

    worker.JOINS.put({"type": "join_batch", "batch_id": "batch-1", "requests": [_chat("c", "joined later")]})
    command = {"type": "generate_batch", "batch_id": "batch-1", "variant": "current_engine",
               "requests": [_chat("a", "a lone stream")]}
    worker._run_engine_batch(command, "m", tokenizer, bridge, {}, set(), threading.Lock())
    assert not [event for event in sent if event["type"] == "error"], sent
    for rid, text in (("a", "a lone stream"), ("c", "joined later")):
        streamed = [event["token_id"] for event in sent if event["type"] == "token" and event["request_id"] == rid]
        assert streamed == bridge.generate(tokenizer.encode(text), 6).tokens
    done = [event for event in sent if event["type"] == "done"]
    assert sorted(d["request_id"] for d in done) == ["a", "c"]
    assert all(d["engine"]["delivery"] == "streamed" for d in done)
    assert sent[-1]["type"] == "batch_done" and bridge.free_slots() == plan.slots



def test_a_request_too_long_for_a_slot_is_rejected_alone_and_the_group_runs(monkeypatch):
    """A prompt plus output limit beyond the dense plan's slot failed its whole group (and the client then retired
    the worker); it is now rejected alone before any token, a joiner too, and the others are answered as before."""
    import threading

    worker, bridge, plan, tokenizer, sent = _worker_world(monkeypatch)
    long_text = "word " * plan.capacity
    worker.JOINS.put({"type": "join_batch", "batch_id": "batch-1",
                      "requests": [_chat("d", long_text), _chat("c", "joined later")]})
    command = {"type": "generate_batch", "batch_id": "batch-1", "variant": "current_engine",
               "requests": [_chat("a", "short one"), _chat("b", long_text)]}
    worker._run_engine_batch(command, "m", tokenizer, bridge, {}, set(), threading.Lock())
    assert not [event for event in sent if event["type"] == "error"], sent
    rejected = [event for event in sent if event["type"] == "rejected"]
    assert [(e["request_id"], e["reason"], e["limit"]) for e in rejected] == [
        ("b", "engine_limit", plan.capacity), ("d", "engine_limit", plan.capacity)]
    assert not [e for e in sent if e.get("request_id") in ("b", "d") and e["type"] != "rejected"]
    for rid, text in (("a", "short one"), ("c", "joined later")):
        streamed = [event["token_id"] for event in sent if event["type"] == "token" and event["request_id"] == rid]
        assert streamed == bridge.generate(tokenizer.encode(text), 6).tokens, rid
    assert sent[-1] == {**sent[-1], "type": "batch_done", "request_ids": ["a", "b", "d", "c"]}
    assert bridge.free_slots() == plan.slots


def test_a_joiner_that_finds_every_slot_taken_takes_the_first_one_a_member_frees(monkeypatch):
    """KB1: with both slots of a two-slot plan taken, a joiner waits inside the run and takes the slot the
    short member frees, instead of waiting for the whole run to end."""
    import threading

    worker, bridge, plan, tokenizer, sent = _worker_world(monkeypatch, slots=2)
    runs, original = [], bridge.generate_many
    monkeypatch.setattr(bridge, "generate_many", lambda *a, **k: runs.append(len(a[0])) or original(*a, **k))
    worker.JOINS.put({"type": "join_batch", "batch_id": "batch-1", "requests": [_chat("c", "waits for a slot")]})
    command = {"type": "generate_batch", "batch_id": "batch-1", "variant": "current_engine",
               "requests": [_chat("a", "short one", 2), _chat("b", "a longer answer", 12)]}
    worker._run_engine_batch(command, "m", tokenizer, bridge, {}, set(), threading.Lock())
    assert not [event for event in sent if event["type"] == "error"], sent
    assert runs == [2]  # one run: the joiner did not wait for it to end
    tokens = [event["request_id"] for event in sent if event["type"] == "token"]  # streamed as read back
    assert tokens.index("c") > tokens.index("a") and tokens.index("c") < len(tokens) - 1 - tokens[::-1].index("b")
    for rid, text, limit in (("a", "short one", 2), ("b", "a longer answer", 12), ("c", "waits for a slot", 6)):
        streamed = [event["token_id"] for event in sent if event["type"] == "token" and event["request_id"] == rid]
        assert streamed == bridge.generate(tokenizer.encode(text), limit).tokens, rid
    assert bridge.free_slots() == plan.slots


class _InProcessWorker:
    """The real worker's batch path behind `MLXWorkerClient.complete_batch`'s contract, in this process: the
    service's group code on one side, `worker._run_engine_batch` with a real dense bridge on the other, the
    JSON events passed between them unchanged (H9: two fakes, one per side, hid that slot reuse never ran)."""

    streams_batches = True
    reuses_slots = True
    ready = True

    def __init__(self, worker, bridge, tokenizer, capacity):
        self.worker, self.bridge, self.tokenizer, self.batch_capacity = worker, bridge, tokenizer, capacity
        self.joined_after: list[tuple[str, int]] = []  # (joiner, members already done when it joined)
        self.order: list[str] = []  # whose token, in the order the worker sent them

    def can_batch_requests(self, requests):
        return True

    def complete_batch(self, requests, *, cancels, timeout, on_event=None, joins=None):
        import queue
        import threading

        events: queue.Queue = queue.Queue()
        self.worker._emit = events.put
        command = {"type": "generate_batch", "batch_id": "batch-1", "variant": "current_engine",
                   "requests": [request.as_dict() for request in requests]}
        members = [request.request_id for request in requests]
        run = threading.Thread(target=self.worker._run_engine_batch,
                               args=(command, "m", self.tokenizer, self.bridge, {}, set(), threading.Lock()))
        run.start()
        done: dict[str, dict] = {}
        rows: dict[str, list] = {rid: [] for rid in members}
        while True:
            try:
                event = events.get(timeout=0.01)
            except queue.Empty:
                event = None
            if event is not None:
                if event["type"] == "batch_done":
                    break
                assert event["type"] != "error", event
                rows[event["request_id"]].append(event)
                if event["type"] == "token":
                    self.order.append(event["request_id"])
                if event["type"] == "done":
                    done[event["request_id"]] = event
                on_event(event["request_id"], event)
            for request, _ in (joins() if joins is not None else []):
                self.joined_after.append((request.request_id, len(done)))
                members.append(request.request_id)
                rows[request.request_id] = []
                self.worker.JOINS.put({"type": "join_batch", "batch_id": "batch-1", "requests": [request.as_dict()]})
        run.join(timeout=30)
        return [{"request_id": rid, "events": rows[rid], "metadata": done[rid]["engine"]} for rid in members]


def test_a_dense_group_admits_a_waiting_request_when_a_member_finishes_through_the_real_service_and_worker(
        monkeypatch):
    from ironmule_product.service import _Session
    from ironmule_product.serving_policy import ServingPolicy
    from ironmule_product.types import GenerationRequest
    from tests.engine.test_product_group_scheduler import _service

    worker, bridge, plan, tokenizer, _ = _worker_world(monkeypatch, slots=2)
    backend = _InProcessWorker(worker, bridge, tokenizer, capacity=2)
    service = _service(backend)
    service.policy = ServingPolicy()  # the default (join off, SERVE8): the dense plan takes newcomers anyway

    def session(rid, text, limit):
        request = GenerationRequest("m", (("user", text),), max_tokens=limit, stream=True, request_id=rid)
        return _Session(request, None)

    short, long_, waiting = session("a", "short one", 2), session("b", "a longer answer", 12), session("c", "waits", 6)
    service._pending.put(waiting)
    service._perform_batch([short, long_])
    assert backend.joined_after == [("c", 1)]  # admitted once one member had finished, not before
    last_b = len(backend.order) - 1 - backend.order[::-1].index("b")
    assert "c" in backend.order and backend.order.index("c") < last_b  # inside the long member's run
    assert service._completed == 3 and service._failed == 0
    for item, text, limit in ((short, "short one", 2), (long_, "a longer answer", 12), (waiting, "waits", 6)):
        frames = []
        while not item.events.empty():
            frames.append(item.events.get_nowait())
        tokens = [f["token_id"] for f in frames if isinstance(f, dict) and f.get("type") == "token"]
        assert tokens == bridge.generate(tokenizer.encode(text), limit).tokens, item.request.request_id
    assert bridge.free_slots() == plan.slots


def test_h11_compares_slot_sizes_bit_for_bit_times_them_and_records_long_prompts_on_cpu_stand_ins(tmp_path):
    """H11 (a)'s tool on CPU stand-ins (it checks the tool, measures nothing): the same requests in slots of two
    sizes, the same tokens and logits; ABBA timing with an A/A pair per size; long prompts in the larger pool."""
    import json

    from tools import cuda_probe as probe

    class Chat(_ChatTokenizer):
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return self.encode(messages[0]["content"])

    def plans(model, capacities):
        return {c: dp.DensePlan(model, capacity=c, linear=_rows, attention=_attention) for c in capacities}

    text = [(i * 7) % 200 + 1 for i in range(400)]
    gate = {name: [(text[i * 20:i * 20 + prefix], text[i * 20 + prefix:i * 20 + prefix + 9]) for i in range(4)]
            for name, prefix in (("3k", 30), ("6k", 60))}
    report = probe.probe_h11_report("tiny", tmp_path / "h11.json", smoke=True, loaded=(_tiny(), Chat()), plans=plans,
                                    capacities=(128, 256), long=(40, 120, 300), gate=gate)
    assert "error" not in report["h11"], report["h11"]
    assert report["auto_capacity"] == report["h11_tier"] == 2048  # no card memory reported on the CPU
    result, analysis = report["h11"], report["analysis"]
    assert all(e["tokens_identical"] and e["logit_steps_identical"] == e["logit_steps"]
               for e in result["exact"].values())
    assert analysis["(a1) answers identical"] is True
    assert sorted((r["capacity"], r["run"]) for r in result["timing"]) == [(128, 0), (128, 1), (256, 0), (256, 1)]
    assert set(result["long"]["prefill_s"]) == {"40", "120"}  # 300 does not fit a 256-token slot
    assert result["long"]["long_row_tokens"] == 120 and "one_long_row_over_short" in analysis
    long_row = result["long"]["long_row_exact"]  # one long row beside seven short ones, against each apart
    assert long_row["tokens_identical"] and long_row["logit_steps_identical"] == long_row["logit_steps"] == 8 * 4
    a2 = analysis["(a2) larger slot per step"]
    assert a2["verdict"] in ("harm", "gain", "no resolved difference") and set(a2["per_width"]) == {"1", "2", "4", "8"}
    quality = analysis["(a4) quality at long context"]
    assert set(quality["groups"]) == {"3k", "6k"} and all(len(result["gate"][g]["plan"]) == 4 for g in quality["groups"])
    assert analysis["(a3) memory"] is None  # no card memory on the CPU: nothing passes by default
    assert analysis["ship_8192"] is False
    assert json.loads((tmp_path / "h11.json").read_text())["schema"] == "ironmule.cuda_probe.h11.v1"
