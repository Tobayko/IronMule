"""SERVE1: chained readback in the serving path must equal step-wise serving exactly."""
import mlx.core as mx
import mlx.nn as nn
import pytest

from ironmule.executor import AsyncGroupedB1Executor, SequentialExecutor, build_sessions
from ironmule.plans import StrictOneShotPlan
from ironmule.runtime import Engine, Knobs
from ironmule.service import MLXBackend, Request
from ironmule.telemetry import Telemetry


@pytest.fixture(scope="module")
def tiny_model():
    # Restored after the module: a leaked CPU default made test_cuda_native compare CPU stock
    # against GPU results whenever both files shared an xdist worker (TEST1, as in B53).
    before = mx.default_device()
    mx.set_default_device(mx.cpu)
    from mlx_lm.models.llama import Model, ModelArgs
    args = ModelArgs(model_type="llama", hidden_size=64, num_hidden_layers=2, intermediate_size=128,
                     num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5, vocab_size=96,
                     tie_word_embeddings=True)
    mx.random.seed(7)
    model = Model(args)
    nn.quantize(model, group_size=32, bits=4)
    mx.eval(model.parameters())
    yield model
    mx.set_default_device(before)


def serve(model, every, prompt, max_tokens, eos, speculate_k=0):
    engine = Engine(model, None, Knobs(compiled_fixed_cache=True, readback_every=every, speculate_k=speculate_k))
    backend = MLXBackend(engine, eos)
    telemetry = Telemetry()
    capacity = backend.capacity_for([len(prompt)], max_tokens)
    sessions = build_sessions([Request(prompt_ids=prompt, max_tokens=max_tokens, plan=StrictOneShotPlan())],
                              backend, telemetry, capacity)
    SequentialExecutor(backend, telemetry).run(sessions, capacity)
    session = sessions[0]
    offset = len(session.prompt_ids) + len(session.tokens) - 1
    assert int(session.state["position"]["offset"].item()) == offset
    return session.tokens, session.stop_reason, session.metrics.visible_generated_tokens, \
        backend.kv_hash(session.state, offset)


@pytest.mark.parametrize("every", [2, 4, 8])
def test_chained_readback_equals_stepwise_at_the_length_limit(tiny_model, every):
    prompt = [3, 9, 27, 81, 5, 6, 7]
    reference = serve(tiny_model, 1, prompt, 13, eos=(95,))
    assert reference[1] == "length"
    assert serve(tiny_model, every, prompt, 13, eos=(95,)) == reference


@pytest.mark.parametrize("every", [2, 3, 8])
def test_chained_readback_stops_at_an_eos_inside_a_batch(tiny_model, every):
    prompt = [3, 9, 27, 81, 5, 6, 7]
    tokens = serve(tiny_model, 1, prompt, 20, eos=(95,))[0]
    eos = (tokens[5],)  # a token the model really emits mid-answer
    reference = serve(tiny_model, 1, prompt, 20, eos=eos)
    assert reference[1] == "eos" and len(reference[0]) == tokens.index(eos[0]) + 1
    assert serve(tiny_model, every, prompt, 20, eos=eos) == reference


def serve_group(model, every, requests, eos):
    engine = Engine(model, None, Knobs(compiled_fixed_cache=True, readback_every=every))
    backend = MLXBackend(engine, eos)
    telemetry = Telemetry()
    capacity = backend.capacity_for([len(p) for p, _ in requests], max(m for _, m in requests))
    sessions = build_sessions([Request(prompt_ids=p, max_tokens=m, plan=StrictOneShotPlan(), rid=str(i))
                               for i, (p, m) in enumerate(requests)], backend, telemetry, capacity)
    AsyncGroupedB1Executor(backend, telemetry, max_width=4).run(sessions, capacity)
    rows = []
    for session in sessions:
        offset = len(session.prompt_ids) + len(session.tokens) - 1
        assert int(session.state["position"]["offset"].item()) == offset
        rows.append((session.tokens, session.stop_reason, backend.kv_hash(session.state, offset)))
    return rows


@pytest.mark.parametrize("every", [2, 4, 8])
def test_grouped_chains_equal_grouped_steps_with_mixed_limits_and_eos(tiny_model, every):
    requests = [([3, 9, 27, 81, 5, 6, 7], 11), ([4, 8, 15, 16, 23, 42], 17), ([1, 2, 3], 5)]
    probe = serve(tiny_model, 1, requests[1][0], 17, eos=(95,))[0]
    eos = (probe[6],)
    reference = serve_group(tiny_model, 1, requests, eos)
    assert {row[1] for row in reference} == {"eos", "length"}
    assert serve_group(tiny_model, every, requests, eos) == reference


def generate(model, speculate_k, prompt, max_tokens, eos, every=4):
    engine = Engine(model, None, Knobs(compiled_fixed_cache=True, readback_every=every, speculate_k=speculate_k))
    return engine.generate(prompt, max_tokens, eos)


@pytest.mark.parametrize("prompt", [[3, 9, 27, 81, 5, 6, 7] * 3, [11, 4, 60, 2, 33, 70, 18, 9]])
@pytest.mark.parametrize("speculate_k", [2, 4])
def test_draft_gated_speculation_equals_plain_greedy(tiny_model, prompt, speculate_k):
    """SPEC1: plain steps without a draft, chained backoff after a miss, exact either way."""
    plain = generate(tiny_model, 0, prompt, 40, eos=(999,))  # outside the vocabulary: full length
    assert len(plain["logical_tokens"]) == 40
    assert generate(tiny_model, speculate_k, prompt, 40, eos=(999,))["logical_tokens"] == plain["logical_tokens"]
    stop = (plain["logical_tokens"][9],)
    plain_stop = generate(tiny_model, 0, prompt, 40, eos=stop)["logical_tokens"]
    assert plain_stop[-1] == stop[0]
    assert generate(tiny_model, speculate_k, prompt, 40, eos=stop)["logical_tokens"] == plain_stop


@pytest.mark.parametrize("prompt", [[3, 9, 27, 81, 5, 6, 7] * 3, [11, 4, 60, 2, 33, 70, 18, 9]])
@pytest.mark.parametrize("speculate_k", [2, 4])
def test_served_speculation_keeps_every_token_and_the_offset_contract(tiny_model, prompt, speculate_k):
    """SERVE2: tokens, stop and visible count equal step-wise serving; the terminal offset holds.

    The cache bits do not: a wide verify computes accepted positions' keys and values with a
    different matmul shape than one-token steps, so they may differ in the last bits. Only
    token-exactness is claimed for speculation.
    """
    full = serve(tiny_model, 1, prompt, 30, eos=(999,))
    assert serve(tiny_model, 4, prompt, 30, eos=(999,), speculate_k=speculate_k)[:3] == full[:3]
    stop = (full[0][7],)
    reference = serve(tiny_model, 1, prompt, 30, eos=stop)
    assert reference[1] == "eos"
    assert serve(tiny_model, 4, prompt, 30, eos=stop, speculate_k=speculate_k)[:3] == reference[:3]
