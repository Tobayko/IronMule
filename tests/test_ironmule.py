"""One runnable check per piece of non-trivial logic. CPU only, no model download."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from ironmule import hw, runtime
from ironmule.fast import fuse_projections
from ironmule.runtime import BASELINE, Engine, Knobs


def test_fingerprint_reacts_to_hardware_and_is_stable():
    facts = hw.static_facts()
    assert hw.fingerprint(facts) == hw.fingerprint(facts)
    assert hw.fingerprint(dict(facts, gpu_cores=999)) != hw.fingerprint(facts)


def test_fusion_is_bit_identical():
    mx.set_default_device(mx.cpu)
    from mlx_lm.models.gemma3_text import Gemma3Model, ModelArgs

    args = ModelArgs(model_type="gemma3_text", hidden_size=64, num_hidden_layers=4,
                     intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
                     head_dim=16, vocab_size=128, sliding_window=8, sliding_window_pattern=2)
    model = Gemma3Model(args)
    nn.quantize(model, group_size=32, bits=4)
    mx.eval(model.parameters())

    tokens = mx.array([[3, 9, 27, 81, 5, 6]])
    reference = model(tokens)
    mx.eval(reference)

    assert fuse_projections(model, check_version=False) == 4
    fused = model(tokens)
    mx.eval(fused)
    assert mx.array_equal(fused, reference).item()


def test_fusion_refuses_an_unverified_library():
    with pytest.raises(Exception):
        fuse_projections(object(), check_version=True)


def test_capacity_is_sized_to_the_workload():
    stub = Engine.__new__(Engine)
    stub.knobs = BASELINE
    assert stub._capacity(322, 32) == 384
    stub.knobs = Knobs(capacity_slack=128)
    assert stub._capacity(322, 32) == 512


def test_state_leaves_are_order_stable():
    assert runtime._leaves({"b": [1, 2], "a": 3}) == [3, 1, 2]


def test_reload_is_only_needed_for_model_mutating_knobs():
    from dataclasses import replace
    assert Engine.needs_reload(BASELINE, replace(BASELINE, fuse_projections=True))
    assert Engine.needs_reload(BASELINE, replace(BASELINE, wired_fraction=0.5))
    assert not Engine.needs_reload(BASELINE, replace(BASELINE, readback_every=8))
    assert not Engine.needs_reload(BASELINE, replace(BASELINE, compiled_fixed_cache=True))


def test_prefix_cache_plan_is_a_function_of_the_prompt_alone():
    """A cache hit must not be able to change where the prefill splits."""
    from ironmule.runtime import PrefixCache

    cache = PrefixCache([5, 6, 7])
    assert cache.matches([5, 6, 7, 8, 9])
    assert not cache.matches([5, 6, 8])
    assert cache.boundary([5, 6, 7, 8, 9]) == 3
    assert cache.boundary([5, 6]) == 2, "a prompt shorter than the prefix splits at its end"
    assert cache.get(384) is None, "an empty cache never hits"
    assert cache.hits == 0 and cache.misses == 0


def test_prefix_cache_hands_out_references_not_copies():
    """slice_update is functional, so sharing arrays with a live snapshot is safe."""
    import mlx.core as mx
    from ironmule.runtime import PrefixCache

    mx.set_default_device(mx.cpu)
    cache = PrefixCache([1, 2])
    keys = mx.zeros((1, 2, 8, 4))
    cache.put({"layers": [{"keys": keys, "values": keys}]}, capacity=8)
    borrowed = cache.get(8)
    assert borrowed is not None and cache.get(16) is None, "capacity must match"
    borrowed[0]["keys"] = mx.ones((1, 2, 8, 4))          # what a decode step does
    assert mx.array_equal(cache.get(8)[0]["keys"], keys).item(), "snapshot was mutated"
