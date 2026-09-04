"""One runnable check per piece of non-trivial logic. CPU only, no model download."""

from types import SimpleNamespace

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


def test_all_kv_state_conversion_preserves_gemma_contract():
    """Gemma's KV/rotating-KV cache becomes the established fixed-shape state."""
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    mx.set_default_device(mx.cpu)
    full = KVCache()
    full.keys = mx.arange(24, dtype=mx.float32).reshape(1, 2, 3, 4)
    full.values = full.keys + 100
    rotating = RotatingKVCache(max_size=8)
    rotating.keys = full.keys
    rotating.values = full.values
    state = runtime._fixed_state_from_standard([full, rotating], used=3, capacity=8)

    assert all(set(layer) == {"keys", "values"} for layer in state["layers"])
    assert state["position"]["offset"].item() == 3
    assert all(layer["keys"].shape == (1, 2, 8, 4) for layer in state["layers"])
    assert mx.array_equal(state["layers"][0]["keys"][..., :3, :], full.keys).item()
    assert mx.array_equal(state["layers"][1]["values"][..., :3, :], full.values).item()


def test_mixed_cache_state_round_trip_preserves_types_and_arrays():
    from mlx_lm.models.cache import ArraysCache, KVCache

    mx.set_default_device(mx.cpu)
    recurrent_a = [mx.ones((1, 3, 4)), mx.arange(8, dtype=mx.float32).reshape(1, 2, 4)]
    recurrent_b = [mx.full((1, 2), 7), mx.zeros((1, 5, 6))]
    first = ArraysCache(size=2)
    first.cache = list(recurrent_a)
    kv = KVCache()
    kv.keys = mx.arange(24, dtype=mx.float32).reshape(1, 2, 3, 4)
    kv.values = kv.keys + 1
    last = ArraysCache(size=2)
    last.cache = list(recurrent_b)

    state = runtime._fixed_state_from_standard([first, kv, last], used=3, capacity=8)
    assert [runtime._state_layer_kind(layer) for layer in state["layers"]] == [
        "arrays", "kv", "arrays"]
    assert all(set(layer) in ({"arrays"}, {"keys", "values"}) for layer in state["layers"])

    restored = runtime._caches_from_state(state, capacity=8)
    assert isinstance(restored[0], ArraysCache) and isinstance(restored[1], runtime.FixedKVCache)
    assert isinstance(restored[2], ArraysCache)
    assert restored[0].cache is not state["layers"][0]["arrays"]
    assert restored[2].cache is not state["layers"][2]["arrays"]
    for got, want in zip(restored[0].cache, recurrent_a):
        assert mx.array_equal(got, want).item()
    for got, want in zip(restored[2].cache, recurrent_b):
        assert mx.array_equal(got, want).item()
    assert mx.array_equal(restored[1].keys[..., :3, :], kv.keys).item()
    assert mx.array_equal(restored[1].values[..., :3, :], kv.values).item()


def test_prefix_and_reset_copy_recurrent_inner_lists_and_hybrid_hash():
    from ironmule.service import MLXBackend

    mx.set_default_device(mx.cpu)
    arrays = [mx.ones((1, 2)), mx.zeros((1, 3))]
    source = {"position": {"offset": mx.array(4, dtype=mx.int32)},
              "layers": [{"arrays": arrays}]}
    cache = runtime.PrefixCache([1])
    cache.put(source, capacity=8)
    borrowed = cache.get(8)
    assert borrowed[0]["arrays"] is not arrays
    borrowed[0]["arrays"].append(mx.ones((1, 4)))
    assert len(cache.get(8)[0]["arrays"]) == 2

    backend = MLXBackend(None, ())
    reset = backend.reset_state(source, offset=2)
    assert reset["layers"][0]["arrays"] is not source["layers"][0]["arrays"]
    assert reset["position"]["offset"].item() == 2
    reset["layers"][0]["arrays"][0] = mx.full((1, 2), 2)
    assert backend.kv_hash(source, offset=4) != backend.kv_hash(reset, offset=2)


@pytest.mark.parametrize("fused, expected", [(False, 1), (True, 2)])
def test_mlx_backend_step_honours_fused_argmax_contract(fused, expected):
    from ironmule.service import MLXBackend

    mx.set_default_device(mx.cpu)

    class Engine:
        knobs = SimpleNamespace(fused_argmax=fused)

        @staticmethod
        def _body(capacity, width):
            def body(input_ids, state):
                if fused:
                    return mx.array([[expected]], dtype=mx.int32), {"next": state}
                logits = mx.array([[[0.0, 3.0, 1.0]]])
                return logits, {"next": state}
            return body

    backend = MLXBackend(Engine(), ())
    output, pick = backend.step({"state": 1}, token=7, capacity=64)
    assert int(pick.item()) == expected
    assert output[1]["next"] == {"state": 1}


@pytest.mark.parametrize("speculate_k", [0, 2])
def test_engine_generate_stops_on_prefill_eos(speculate_k):
    from ironmule.runtime import Engine

    class Scalar:
        def reshape(self, shape):
            return self

        def item(self):
            return 99

    engine = Engine.__new__(Engine)
    engine._closed = False
    engine.knobs = SimpleNamespace(speculate_k=speculate_k, as_dict=lambda: {})
    engine.prefix_cache = None
    engine._capacity = lambda prompt_len, max_tokens: 64
    engine._prefill = lambda prompt_ids, capacity: ({}, Scalar())

    def unexpected_decode(*args, **kwargs):
        raise AssertionError("prefill EOS must skip normal and speculative decode")

    engine._decode = unexpected_decode
    engine._decode_speculative = unexpected_decode
    result = engine.generate([1], max_tokens=3, eos_ids=(99,))

    assert result["physical_tokens"] == [99]
    assert result["logical_tokens"] == [99]
    assert result["visible_tokens"] == []
    assert result["decode_ns"] == 0


def test_unknown_cache_type_is_rejected():
    with pytest.raises(TypeError, match="unsupported model cache type"):
        runtime._cache_kinds([object()])


def test_arrays_cache_metadata_is_rejected_instead_of_reset():
    from mlx_lm.models.cache import ArraysCache

    mx.set_default_device(mx.cpu)
    cache = ArraysCache(size=2)
    cache.cache = [mx.zeros((1, 2)), mx.zeros((1, 3))]
    cache.lengths = mx.array([2])
    with pytest.raises(ValueError, match="lengths/left_padding"):
        runtime._fixed_state_from_standard([cache], used=1, capacity=8)

    cache.lengths = None
    cache.left_padding = mx.array([0])
    with pytest.raises(ValueError, match="lengths/left_padding"):
        runtime._empty_fixed_state(8, [cache])


def test_hybrid_hash_preserves_bfloat16_raw_bits():
    from ironmule.service import MLXBackend

    mx.set_default_device(mx.cpu)
    first = mx.array([[1.0, 2.0]], dtype=mx.bfloat16)
    second = mx.array([[3.0, 4.0]], dtype=mx.bfloat16)
    state = {"position": {"offset": mx.array(1, dtype=mx.int32)},
             "layers": [{"arrays": [first, second]}]}
    backend = MLXBackend(None, ())
    original = backend.kv_hash(state, offset=1)
    changed = {"position": state["position"],
               "layers": [{"arrays": [first, second + mx.array(2, dtype=mx.bfloat16)]}]}
    assert backend.kv_hash(changed, offset=1) != original
