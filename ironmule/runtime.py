"""Tunable greedy decode engine.

Every knob is a switch the autotuner may flip; none of them may change a token.
The knobs collect the mechanisms that the predecessor project measured one at a time
(fixed-shape KV cache plus `mx.compile`, greedy selection inside the compiled
body, batched stop-token readback, last-position-only prefill projection) and
add the two this fork introduces: projection fusion and a right-sized cache.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from typing import Any

import mlx.core as mx

from . import fast


@dataclass(frozen=True)
class Knobs:
    fuse_projections: bool = False
    compiled_fixed_cache: bool = False
    fused_argmax: bool = False
    head_skip_prefill: bool = False
    prefill_into_fixed: bool = False
    readback_every: int = 1
    speculate_k: int = 0           # 0 -> no speculation; else draft width per forward
    speculate_ngram: int = 3       # longest prompt n-gram used to propose a draft
    capacity_slack: int = 0        # 0 -> capacity is auto sized to the workload
    wired_fraction: float = 0.0    # 0 -> leave the MLX wired limit alone

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def key(self) -> str:
        return "|".join(f"{k}={v}" for k, v in sorted(self.as_dict().items()))


BASELINE = Knobs()


class FixedKVCache:
    """Constant-shape KV cache: the precondition for compiling the decode step."""

    def __init__(self, state: dict[str, Any], position: dict[str, Any], capacity: int):
        self._state, self._position, self._capacity = state, position, capacity

    @property
    def offset(self):
        return self._position["offset"]

    @property
    def keys(self):
        return self._state["keys"]

    @property
    def values(self):
        return self._state["values"]

    def update_and_fetch(self, keys, values):
        zero = mx.array(0, dtype=self._position["offset"].dtype)
        starts = mx.stack((zero, zero, self._position["offset"], zero))
        self._state["keys"] = mx.slice_update(self._state["keys"], keys, start_indices=starts, axes=(0, 1, 2, 3))
        self._state["values"] = mx.slice_update(self._state["values"], values, start_indices=starts, axes=(0, 1, 2, 3))
        return self._state["keys"], self._state["values"]

    def make_mask(self, n_tokens: int, *, window_size: int | None = None, return_array: bool = False):
        del return_array
        dtype = self._position["offset"].dtype
        slots = mx.arange(self._capacity, dtype=dtype)
        queries = self._position["offset"] + mx.arange(n_tokens, dtype=dtype)
        mask = (slots[None, :] <= queries[:, None]) & (slots[None, :] < self._position["offset"] + n_tokens)
        if window_size is not None:
            mask = mask & (slots[None, :] >= queries[:, None] - window_size + 1)
        return mask[None, None, :, :]


class PrefixCache:
    """Snapshot of a declared shared prompt prefix.

    Reuse here is bit exact rather than approximately exact, and the reason is the
    execution plan, not the arithmetic. The engine always feeds the request tail as
    its own chunk, so the sequence of forwards is identical whether the prefix came
    out of this cache or was recomputed a moment ago. Only a plan change can move a
    token, and the plan does not change. E9 measured `max|delta| = 0` against a
    chunked baseline across 12 requests and every decode step, while the same reuse
    against a *single shot* baseline flipped 2 tokens in 254 steps (E8).

    The consequence is deliberate and must be stated: a chunked plan does not
    produce the same tokens as a single-shot plan (E9 measured up to 4.31 logits
    apart). Enabling this is therefore a plan decision a caller makes, never
    something the autotuner may flip on its own.
    """

    def __init__(self, prefix_ids):
        self.prefix_ids = tuple(prefix_ids)
        self._snapshot = None
        self._capacity = None
        self.hits = 0
        self.misses = 0

    def boundary(self, prompt_ids) -> int:
        """Where this plan splits the prefill. A function of the prompt alone."""
        return min(len(self.prefix_ids), len(prompt_ids))

    def matches(self, prompt_ids) -> bool:
        return tuple(prompt_ids[:len(self.prefix_ids)]) == self.prefix_ids

    def get(self, capacity: int):
        if self._snapshot is None or self._capacity != capacity:
            return None
        # slice_update is functional, so handing out references is safe: advancing a
        # copy of this structure never writes through to the stored arrays.
        return _copy_state_layers(self._snapshot)

    def put(self, state, capacity: int) -> None:
        self._snapshot = _copy_state_layers(state["layers"])
        self._capacity = capacity


def _text(model):
    """The text stack. Gemma 3 ships behind a multimodal wrapper; other models do not."""
    inner = getattr(model, "language_model", None)
    return model if inner is None else inner


def _trunk(model):
    """Everything up to but excluding the output projection."""
    return _text(model).model


def _project(model, hidden):
    """The output projection, honouring tied embeddings."""
    text = _text(model)
    if getattr(text, "tie_word_embeddings", False):
        return text.model.embed_tokens.as_linear(hidden)
    return text.lm_head(hidden)


def _cache_kinds(cache: list | tuple) -> list[str]:
    """Classify the MLX-LM cache contract, rejecting unknown implementations."""
    if not isinstance(cache, (list, tuple)):
        raise TypeError(f"model cache must be a list or tuple, got {type(cache).__name__}")
    from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

    kinds = []
    for item in cache:
        if isinstance(item, ArraysCache):
            if item.lengths is not None or item.left_padding is not None:
                raise ValueError(
                    "ArraysCache lengths/left_padding metadata is unsupported in runtime state")
            kinds.append("arrays")
        elif isinstance(item, (KVCache, RotatingKVCache)):
            kinds.append("kv")
        else:
            raise TypeError(f"unsupported model cache type: {type(item).__name__}")
    return kinds


def _state_layer_kind(layer: dict[str, Any]) -> str:
    """Infer a state layer kind from its shape; marker leaves are forbidden."""
    if set(layer) == {"keys", "values"}:
        return "kv"
    if set(layer) == {"arrays"} and isinstance(layer["arrays"], list):
        return "arrays"
    raise TypeError("unsupported cache state layer; expected keys/values or arrays")


def _copy_state_layers(layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy Python containers while retaining immutable MLX array leaves."""
    copied = []
    for layer in layers:
        kind = _state_layer_kind(layer)
        if kind == "kv":
            copied.append({"keys": layer["keys"], "values": layer["values"]})
        else:
            copied.append({"arrays": list(layer["arrays"])})
    return copied


def _state_is_hybrid(state: dict[str, Any]) -> bool:
    kinds = {_state_layer_kind(layer) for layer in state["layers"]}
    return "arrays" in kinds


def _cache_leaves(cache: list | tuple) -> list[Any]:
    """Return cache arrays for evaluation without assuming every cache has keys."""
    leaves = []
    for item, kind in zip(cache, _cache_kinds(cache)):
        if kind == "kv":
            leaves.extend((item.keys, item.values))
        else:
            leaves.extend(value for value in item.cache if value is not None)
    return leaves


def _caches_from_state(state: dict[str, Any], capacity: int) -> list[Any]:
    """Recreate MLX-LM cache objects from the marker-free runtime state tree."""
    from mlx_lm.models.cache import ArraysCache

    caches = []
    for layer in state["layers"]:
        kind = _state_layer_kind(layer)
        if kind == "kv":
            caches.append(FixedKVCache(layer, state["position"], capacity))
        else:
            cache = ArraysCache(size=len(layer["arrays"]))
            cache.cache = list(layer["arrays"])
            caches.append(cache)
    return caches


def _state_from_caches(caches: list[Any], position: dict[str, Any]) -> dict[str, Any]:
    """Capture cache mutations while retaining each layer's native state shape."""
    layers = []
    from mlx_lm.models.cache import ArraysCache

    for cache in caches:
        if isinstance(cache, FixedKVCache):
            layers.append({"keys": cache.keys, "values": cache.values})
        elif isinstance(cache, ArraysCache):
            layers.append({"arrays": list(cache.cache)})
        else:  # defensive: cache objects are created by _caches_from_state
            raise TypeError(f"unsupported runtime cache type: {type(cache).__name__}")
    return {"position": position, "layers": layers}


def _empty_fixed_state(capacity: int, template: list) -> dict[str, Any]:
    kinds = _cache_kinds(template)
    layers = []
    for layer, kind in zip(template, kinds):
        if kind == "kv":
            keys = layer.keys
            shape = (keys.shape[0], keys.shape[1], capacity, keys.shape[3])
            layers.append({"keys": mx.zeros(shape, dtype=keys.dtype),
                           "values": mx.zeros(shape, dtype=keys.dtype)})
        else:
            if any(value is None for value in layer.cache):
                raise ValueError("recurrent cache must be initialized before conversion")
            layers.append({"arrays": [mx.zeros_like(value) for value in layer.cache]})
    return {"position": {"offset": mx.array(0, dtype=mx.int32)}, "layers": layers}


def _fixed_state_from_standard(cache: list, used: int, capacity: int) -> dict[str, Any]:
    kinds = _cache_kinds(cache)
    layers = []
    start = mx.array((0, 0, 0, 0), dtype=mx.int32)
    for layer, kind in zip(cache, kinds):
        if kind == "arrays":
            if any(value is None for value in layer.cache):
                raise ValueError("recurrent cache must be initialized before conversion")
            layers.append({"arrays": list(layer.cache)})
            continue
        keys, values = layer.keys, layer.values
        if keys.shape[2] < used or used > capacity:
            raise ValueError("prompt does not fit the fixed capacity")
        shape = (keys.shape[0], keys.shape[1], capacity, keys.shape[3])
        layers.append({
            "keys": mx.slice_update(mx.zeros(shape, dtype=keys.dtype), keys[..., :used, :],
                                    start_indices=start, axes=(0, 1, 2, 3)),
            "values": mx.slice_update(mx.zeros(shape, dtype=values.dtype), values[..., :used, :],
                                      start_indices=start, axes=(0, 1, 2, 3)),
        })
    return {"position": {"offset": mx.array(used, dtype=mx.int32)}, "layers": layers}


def _lookup_draft(sequence: list[int], ngram: int, k: int) -> list[int]:
    """Continuation that followed the most recent earlier occurrence of the current tail.

    Longest tail first, so a specific match beats a generic one. Returns at most k
    tokens and an empty list when nothing matches.
    """
    for size in range(ngram, 1, -1):
        if len(sequence) <= size:
            continue
        tail = sequence[-size:]
        for start in range(len(sequence) - size - 1, -1, -1):
            if sequence[start:start + size] == tail:
                draft = sequence[start + size:start + size + k]
                if draft:
                    return draft
    return []


def _leaves(value):
    if isinstance(value, dict):
        return [leaf for key in sorted(value) for leaf in _leaves(value[key])]
    if isinstance(value, (list, tuple)):
        return [leaf for child in value for leaf in _leaves(child)]
    return [value]


class Engine:
    """One loaded model plus one knob setting. Reload only when the knobs need it."""

    def __init__(self, model, tokenizer, knobs: Knobs = BASELINE):
        self.model, self.tokenizer, self.knobs = model, tokenizer, knobs
        self._compiled = None
        self._compiled_capacity = None
        self.prefix_cache: PrefixCache | None = None
        if knobs.fuse_projections:
            fast.fuse_projections(model)
        if knobs.wired_fraction > 0:
            from .hw import static_facts
            mx.set_wired_limit(int(static_facts()["memory_bytes"] * knobs.wired_fraction))

    @staticmethod
    def needs_reload(old: Knobs, new: Knobs) -> bool:
        """Fusion and the wired limit mutate global/model state; the rest is per call."""
        return (old.fuse_projections != new.fuse_projections
                or old.wired_fraction != new.wired_fraction)

    def _capacity(self, prompt_len: int, max_tokens: int) -> int:
        needed = prompt_len + max_tokens + self.knobs.speculate_k + self.knobs.capacity_slack
        return ((needed + 63) // 64) * 64

    def _body(self, capacity: int, width: int):
        if self._compiled is not None and self._compiled_capacity == (capacity, width):
            return self._compiled
        fused = self.knobs.fused_argmax
        model, trunk = self.model, _trunk(self.model)

        def body(input_ids, state):
            if _state_is_hybrid(state):
                caches = _caches_from_state(state, capacity)
            else:
                # Keep the established all-KV compiled graph and state tree intact.
                caches = [FixedKVCache(layer, state["position"], capacity)
                          for layer in state["layers"]]
            logits = _project(model, trunk(input_ids, cache=caches))
            if _state_is_hybrid(state):
                layers = _state_from_caches(caches, state["position"])["layers"]
            else:
                layers = [{"keys": c.keys, "values": c.values} for c in caches]
            new_state = {"position": {"offset": state["position"]["offset"] + input_ids.shape[1]},
                         "layers": layers}
            if fused:
                return mx.argmax(logits, axis=-1), new_state
            return logits, new_state

        self._compiled = mx.compile(body, shapeless=False) if self.knobs.compiled_fixed_cache else body
        self._compiled_capacity = (capacity, width)
        return self._compiled

    def _feed(self, state, ids: list[int], capacity: int):
        """One prefill chunk into an existing fixed-shape state."""
        if _state_is_hybrid(state):
            caches = _caches_from_state(state, capacity)
        else:
            caches = [FixedKVCache(layer, state["position"], capacity)
                      for layer in state["layers"]]
        hidden = _trunk(self.model)(mx.array(ids)[None, :], cache=caches)
        used = int(state["position"]["offset"].item()) + len(ids)
        position = {"offset": mx.array(used, dtype=mx.int32)}
        if _state_is_hybrid(state):
            return (_state_from_caches(caches, position), hidden)
        return ({"position": position,
                 "layers": [{"keys": c.keys, "values": c.values} for c in caches]}, hidden)

    def _empty_state(self, capacity: int):
        probe = self.model.make_cache()
        kinds = _cache_kinds(probe)  # validate before invoking the model with the cache
        _trunk(self.model)(mx.array([[0]]), cache=probe)
        if all(kind == "kv" for kind in kinds):
            mx.eval([c.keys for c in probe])
        else:
            mx.eval(*_cache_leaves(probe))
        return _empty_fixed_state(capacity, probe)

    def _prefill_chunked(self, prompt_ids: list[int], capacity: int, cache: "PrefixCache"):
        """The declared plan: chunk 1 up to the boundary, then the request tail.

        Identical work whether chunk 1 is served from the snapshot or computed, so
        a cache hit cannot change a token.
        """
        split = cache.boundary(prompt_ids)
        head, tail = prompt_ids[:split], prompt_ids[split:]
        stored = cache.get(capacity) if cache.matches(prompt_ids) else None
        if stored is not None:
            cache.hits += 1
            state = {"position": {"offset": mx.array(split, dtype=mx.int32)}, "layers": stored}
        else:
            cache.misses += 1
            state, _ = self._feed(self._empty_state(capacity), head, capacity)
            mx.eval(*_leaves(state))
            if cache.matches(prompt_ids):
                cache.put(state, capacity)
        if tail:
            state, hidden = self._feed(state, tail, capacity)
        else:
            state, hidden = self._feed(state, head[-1:], capacity)
        logits = _project(self.model, hidden[:, -1:, :] if self.knobs.head_skip_prefill else hidden)
        token = mx.argmax(logits[:, -1, :], axis=-1).reshape((1, 1))
        mx.eval(token, *_leaves(state))
        mx.synchronize()
        return state, token

    def _prefill(self, prompt_ids: list[int], capacity: int):
        if self.prefix_cache is not None:
            return self._prefill_chunked(prompt_ids, capacity, self.prefix_cache)
        ids = mx.array(prompt_ids)[None, :]
        if self.knobs.prefill_into_fixed:
            probe = self.model.make_cache()
            kinds = _cache_kinds(probe)
            _ = _trunk(self.model)(mx.array([[prompt_ids[0]]]), cache=probe)
            if all(kind == "kv" for kind in kinds):
                mx.eval([c.keys for c in probe])
            else:
                mx.eval(*_cache_leaves(probe))
            state = _empty_fixed_state(capacity, probe)
            if all(_state_layer_kind(layer) == "kv" for layer in state["layers"]):
                caches = [FixedKVCache(layer, state["position"], capacity)
                          for layer in state["layers"]]
            else:
                caches = _caches_from_state(state, capacity)
            hidden = _trunk(self.model)(ids, cache=caches)
            position = {"offset": mx.array(len(prompt_ids), dtype=mx.int32)}
            if all(_state_layer_kind(layer) == "kv" for layer in state["layers"]):
                state = {"position": position,
                         "layers": [{"keys": c.keys, "values": c.values} for c in caches]}
            else:
                state = _state_from_caches(caches, position)
        else:
            cache = self.model.make_cache()
            _cache_kinds(cache)
            hidden = _trunk(self.model)(ids, cache=cache)

        if self.knobs.head_skip_prefill:
            logits = _project(self.model, hidden[:, -1:, :])
        else:
            logits = _project(self.model, hidden)

        token = mx.argmax(logits[:, -1, :], axis=-1).reshape((1, 1))
        if not self.knobs.prefill_into_fixed:
            mx.eval(logits)
            state = _fixed_state_from_standard(cache, len(prompt_ids), capacity)
        mx.eval(token, *_leaves(state))
        mx.synchronize()
        return state, token

    def _picks(self, out):
        """Greedy choice per input position, shape (1, width)."""
        return out[0] if self.knobs.fused_argmax else mx.argmax(out[0], axis=-1)

    def _decode(self, state, token, max_tokens, eos, capacity):
        body = self._body(capacity, 1)
        physical = [int(token.reshape((-1,)).item())]
        every = max(1, self.knobs.readback_every)
        pending: list[Any] = []
        for step in range(max_tokens - 1):
            out = body(token, state)
            picks = self._picks(out)
            token, state = picks[:, -1:], out[1]
            pending.append(token)
            if len(pending) == every or step == max_tokens - 2:
                mx.eval(*pending, *_leaves(state))
                mx.synchronize()
                physical.extend(int(item.reshape((-1,)).item()) for item in pending)
                if any(value in eos for value in physical[-len(pending):]):
                    break
                pending = []
        return physical, 0

    def _decode_speculative(self, state, token, prompt_ids, max_tokens, eos, capacity):
        """Prompt-lookup speculation. Exactly greedy: a draft token is kept only when
        it equals what the model itself chose for that position."""
        if _state_is_hybrid(state):
            raise ValueError("speculative decoding is unsupported for hybrid cache state")
        width = self.knobs.speculate_k + 1
        body = self._body(capacity, width)
        first = int(token.reshape((-1,)).item())
        physical = [first]
        sequence = list(prompt_ids) + [first]
        offset = len(prompt_ids) + 1
        current, drafted, accepted_total = first, 0, 0

        while len(physical) < max_tokens:
            draft = _lookup_draft(sequence, self.knobs.speculate_ngram, self.knobs.speculate_k)
            padded = (draft + [current] * self.knobs.speculate_k)[:self.knobs.speculate_k]
            out = body(mx.array([[current] + padded]), state)
            picks = self._picks(out)
            state = out[1]
            mx.eval(picks, *_leaves(state))
            mx.synchronize()
            chosen = picks.reshape((-1,)).tolist()

            accepted = [chosen[0]]
            for i in range(1, width):
                if i - 1 < len(draft) and draft[i - 1] == chosen[i - 1]:
                    accepted.append(chosen[i])
                else:
                    break
            accepted = accepted[:max_tokens - len(physical)]

            drafted += len(draft)
            accepted_total += len(accepted) - 1
            physical.extend(accepted)
            sequence.extend(accepted)
            offset += len(accepted)
            current = accepted[-1]
            # Roll the cache back over the rejected draft; the mask hides the stale slots.
            state["position"]["offset"] = mx.array(offset - 1, dtype=mx.int32)
            if any(value in eos for value in accepted):
                break
        return physical, (accepted_total / drafted if drafted else 0.0)

    def generate(self, prompt_ids: list[int], max_tokens: int, eos_ids: tuple[int, ...]) -> dict[str, Any]:
        """Greedy decode. Returns tokens plus a timing breakdown."""
        capacity = self._capacity(len(prompt_ids), max_tokens)

        started = time.perf_counter_ns()
        state, token = self._prefill(prompt_ids, capacity)
        prefill_ns = time.perf_counter_ns() - started

        first = int(token.reshape((-1,)).item())
        if first in eos_ids:
            physical, acceptance, decode_ns = [first], 0.0, 0
        else:
            started = time.perf_counter_ns()
            if self.knobs.speculate_k > 0:
                physical, acceptance = self._decode_speculative(
                    state, token, prompt_ids, max_tokens, eos_ids, capacity)
            else:
                physical, acceptance = self._decode(state, token, max_tokens, eos_ids, capacity)
            decode_ns = time.perf_counter_ns() - started

        logical = []
        for value in physical:
            logical.append(value)
            if value in eos_ids:
                break
        return {
            "physical_tokens": physical,
            "logical_tokens": logical,
            "visible_tokens": [t for t in logical if t not in eos_ids],
            "prefill_ns": prefill_ns,
            "decode_ns": decode_ns,
            "total_ns": prefill_ns + decode_ns,
            "capacity": capacity,
            "acceptance": acceptance,
            "prefix_cache_hits": self.prefix_cache.hits if self.prefix_cache else 0,
            "knobs": self.knobs.as_dict(),
        }


def _self_check() -> None:
    """Knob bookkeeping and capacity sizing, without loading a model."""
    assert BASELINE.readback_every == 1
    assert Knobs(fuse_projections=True).key() != BASELINE.key()
    assert Engine.needs_reload(BASELINE, replace(BASELINE, fuse_projections=True))
    assert not Engine.needs_reload(BASELINE, replace(BASELINE, readback_every=8))

    class Stub(Engine):
        def __init__(self, knobs):
            self.knobs = knobs

    assert Stub(BASELINE)._capacity(322, 32) == 384, "capacity must round up to the next 64"
    assert Stub(Knobs(capacity_slack=200))._capacity(322, 32) == 576
    assert Stub(Knobs(speculate_k=4))._capacity(322, 32) == 384

    cache = PrefixCache([5, 6, 7])
    assert cache.matches([5, 6, 7, 8, 9]) and not cache.matches([5, 6, 8])
    assert cache.boundary([5, 6, 7, 8, 9]) == 3, "plan splits at the declared prefix"
    assert cache.boundary([5, 6]) == 2, "a short prompt splits at its own end"
    assert cache.get(384) is None, "an empty cache never hits"

    tokens = [7, 1, 2, 3, 9, 9, 1, 2, 3]
    assert _lookup_draft(tokens, 3, 2) == [9, 9], "must continue the most recent match"
    assert _lookup_draft(tokens, 3, 0) == [], "k=0 proposes nothing"
    assert _lookup_draft([5, 6], 3, 2) == [], "no history, no draft"
    assert _lookup_draft([4, 4, 4, 4], 2, 2) == [4], "a draft is cut off by the end of history"
    assert _leaves({"b": [1, 2], "a": 3}) == [3, 1, 2], "state leaves must be order stable"
    print("runtime self-check ok")


if __name__ == "__main__":
    _self_check()
