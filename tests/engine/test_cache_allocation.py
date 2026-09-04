"""B25: does anything reallocate the KV cache during decode?

The predecessor project localised 4.4263% of correlated marginal decode cost to cache
growth copies and never got to measure it, because back then the cache grew. This
runtime allocates a fixed-shape cache once per `serve()`, so the cost should be gone.
"Should be" is not a measurement, and the answer decides whether 4.4% is sitting there
unclaimed or whether the entry closes. This asserts the mechanism directly, without a
model: writing token after token into the cache must not grow the allocation.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from ironmule.runtime import FixedKVCache  # noqa: E402


CAPACITY = 256
HEADS, DIM = 4, 8
STEPS = 64
# MLX may retain a small allocator header/cache bookkeeping delta. This fixed test-only
# tolerance is in addition to exactly one keys+values cache copy, not a mechanism change.
ALLOCATOR_METADATA_TOLERANCE_BYTES = 4096


def _cache():
    shape = (1, HEADS, CAPACITY, DIM)
    state = {"keys": mx.zeros(shape, dtype=mx.float16),
             "values": mx.zeros(shape, dtype=mx.float16)}
    position = {"offset": mx.array(0, dtype=mx.int32)}
    return FixedKVCache(state, position, CAPACITY), position


def _write_one_token(cache, position, step):
    one = mx.ones((1, HEADS, 1, DIM), dtype=mx.float16)
    keys, values = cache.update_and_fetch(one, one)
    position["offset"] = mx.array(step + 1, dtype=mx.int32)
    mx.eval(keys, values, position["offset"])


def test_fixed_cache_does_not_grow_while_decoding():
    cache, position = _cache()

    # Warm up: first writes allocate scratch that is not part of the steady state.
    for step in range(8):
        _write_one_token(cache, position, step)
    mx.clear_cache()
    settled = mx.get_active_memory()

    for step in range(8, STEPS):
        _write_one_token(cache, position, step)
    mx.eval(cache.keys, cache.values)
    after = mx.get_active_memory()

    grown = after - settled
    budget = (CAPACITY * HEADS * DIM * 2 * 2
              + ALLOCATOR_METADATA_TOLERANCE_BYTES)  # one copy + allocator metadata
    assert grown <= budget, (
        f"{STEPS - 8} decode steps grew active memory by {grown} B "
        f"(more than one full cache copy plus {ALLOCATOR_METADATA_TOLERANCE_BYTES} B "
        f"allocator metadata, threshold {budget} B) -- the fixed-shape cache is "
        f"being reallocated after all"
    )


def test_cache_keeps_its_shape_across_the_whole_sequence():
    """A reallocation that kept memory flat but changed shape would still be one."""
    cache, position = _cache()
    shape = cache.keys.shape
    for step in range(STEPS):
        _write_one_token(cache, position, step)
        assert cache.keys.shape == shape
        assert cache.values.shape == shape


def test_paired_ab_measures_peak_memory_per_arm(monkeypatch):
    """Every arm loads its own model into one process; the peak must be reset between.

    MLX's peak is a process-wide high-water mark. Read once after the arm loop, the
    second arm inherits the first arm's peak and the number is no longer about the arm
    it is filed under -- which is how a 4.15 GB single-arm pilot turns into 6+ GB with
    nothing having grown.
    """
    import importlib

    from ironmule import ab
    from ironmule.runtime import Knobs

    tune_module = importlib.import_module("ironmule.tune")
    resets = []
    peaks = iter([4_000_000_000, 4_100_000_000])

    class FakeEngine:
        def generate(self, ids, max_tokens, eos):
            return {"total_ns": 1, "prefill_ns": 1, "decode_ns": 1,
                    "logical_tokens": [1, 2], "physical_tokens": [1, 2],
                    "capacity": 64}

    monkeypatch.setattr(tune_module, "load_engine", lambda *a, **k: (FakeEngine(), object()))
    monkeypatch.setattr(tune_module, "prompt_ids", lambda tok, prompt: [1, 2, 3])
    monkeypatch.setattr(tune_module, "_eos_ids", lambda tok: (0,))
    monkeypatch.setattr(mx, "reset_peak_memory", lambda: resets.append(True))
    monkeypatch.setattr(mx, "get_peak_memory", lambda: next(peaks))

    spec = {"order": ["a", "b"],
            "arms": {"a": Knobs().as_dict(), "b": Knobs().as_dict()},
            "warmup": 0, "repeats": 1, "max_tokens": 2}
    out = ab._child(spec)

    assert len(resets) == 2, "the peak must be reset once per arm, not once per process"
    assert out["arms"]["a"]["mlx_peak_bytes"] == 4_000_000_000
    assert out["arms"]["b"]["mlx_peak_bytes"] == 4_100_000_000
    assert out["mlx_peak_bytes"] == 4_100_000_000
    assert out["arms"]["a"]["logical_tokens_per_repeat"] == [[1, 2]]
    assert out["arms"]["a"]["physical_tokens_per_repeat"] == [[1, 2]]
    assert out["arms"]["a"]["token_counts"] == [{"logical": 2, "physical": 2}]
    assert out["arms"]["a"]["stop_reasons"] == ["length"]
    assert out["arms"]["a"]["capacities"] == [64]
