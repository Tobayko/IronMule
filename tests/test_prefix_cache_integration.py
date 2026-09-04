"""Comprehensive integration and unit tests for stateful PrefixCache support in friday_serve.

Tests:
1. Direct PrefixCache functionality (matching, boundary, capacity caching).
2. IronMuleBackend prefix cache lifecycle:
   - set_prefix_cache creates PrefixCache and sets engine.prefix_cache on new and existing engines.
   - set_prefix_cache(None) clears prefix cache.
   - generate() transparently forwards prefix_cache_hits.
3. Server prefix cache delegation:
   - set_prefix_cache encodes string prompt or forwards raw token IDs.
   - set_prefix_cache(None) forwards None.
   - Backend without set_prefix_cache raises AttributeError.
4. End-to-end cache miss on first call, cache hit on subsequent call with shared prefix:
   - 100% token fidelity and correct metadata propagation.
"""

from __future__ import annotations

import unittest
from typing import Any, Mapping, Sequence

from friday_calibrate.profile import DeviceProfile, KnobVerdict
from friday_evidence.canonical import canonical_sha256
from friday_serve.ironmule_backend import IronMuleBackend
from friday_serve.server import BASELINE_PLAN, DEVICE_PROFILE_PLAN, Server

# ironmule worktree is placed on sys.path by tests/conftest.py
from ironmule.runtime import BASELINE, Knobs, PrefixCache


def make_profile(*verified: str, model_id: str = "test-model", revision: str = "rev1") -> DeviceProfile:
    verdicts = [
        KnobVerdict(k, "verified" if k in verified else "failed", 6, 0.9, 0.88, 0.93, True)
        for k in ("head_skip", "fixed_compiled", "bundled_readback")
    ]
    return DeviceProfile(
        profile_id="device-prefix-test",
        model_id=model_id,
        model_revision=revision,
        hardware_sha256="a" * 64,
        environment_sha256="b" * 64,
        mde=0.006,
        knobs=tuple(verdicts),
    )


class FakeEngine:
    """Simulates an IronMule Engine with prefix caching behavior."""

    def __init__(self, model: Any, tokenizer: Any, knobs: Knobs = BASELINE) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.knobs = knobs
        self.prefix_cache: PrefixCache | None = None
        self.calls: list[list[int]] = []

    def generate(
        self, prompt_ids: Sequence[int], max_tokens: int, eos_ids: Sequence[int]
    ) -> dict[str, Any]:
        self.calls.append(list(prompt_ids))
        prompt_tuple = tuple(prompt_ids)
        hits_before = self.prefix_cache.hits if self.prefix_cache is not None else 0

        if self.prefix_cache is not None:
            capacity = 128
            cached = self.prefix_cache.get(capacity)
            if cached is not None and self.prefix_cache.matches(prompt_tuple):
                self.prefix_cache.hits += 1
            else:
                self.prefix_cache.misses += 1
                fake_state = {"layers": [{"keys": "k", "values": "v"}]}
                self.prefix_cache.put(fake_state, capacity)

        # Deterministic tokens derived from prompt
        tokens = [(prompt_ids[0] if prompt_ids else 1) * 10 + i for i in range(max_tokens)]
        return {
            "physical_tokens": tokens,
            "logical_tokens": tokens,
            "visible_tokens": [t for t in tokens if t not in eos_ids],
            "prefill_ns": 400_000 if (self.prefix_cache and self.prefix_cache.hits > hits_before) else 2_000_000,
            "decode_ns": 1_000_000,
            "total_ns": 3_000_000,
            "capacity": 128,
            "acceptance": None,
            "prefix_cache_hits": self.prefix_cache.hits if self.prefix_cache else 0,
            "knobs": self.knobs.as_dict(),
        }


class FakeTokenizer:
    def __init__(self) -> None:
        self.eos_token_id = 999

    def apply_chat_template(self, messages, add_generation_prompt=True):
        content = messages[0]["content"]
        return [ord(c) for c in content]

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, token_ids: Sequence[int]) -> str:
        return f"decoded({len(token_ids)})"


class MockBackend:
    """Mock backend implementing GenerationBackend protocol with set_prefix_cache."""

    def __init__(self, model_id: str = "test-model", revision: str = "rev1") -> None:
        self.model_id = model_id
        self.model_revision = revision
        self.prefix_cache: PrefixCache | None = None
        self.calls: list[dict[str, Any]] = []

    def encode(self, prompt: str) -> list[int]:
        return [ord(c) for c in prompt]

    def set_prefix_cache(self, prefix_ids: Sequence[int] | None) -> None:
        if prefix_ids is None:
            self.prefix_cache = None
        elif isinstance(prefix_ids, PrefixCache):
            self.prefix_cache = prefix_ids
        else:
            self.prefix_cache = PrefixCache(list(prefix_ids))

    def generate(
        self, token_ids: Sequence[int], max_tokens: int, knobs: Mapping[str, Any]
    ) -> dict[str, Any]:
        prompt_tuple = tuple(token_ids)
        hits_before = self.prefix_cache.hits if self.prefix_cache is not None else 0

        if self.prefix_cache is not None:
            capacity = 128
            cached = self.prefix_cache.get(capacity)
            if cached is not None and self.prefix_cache.matches(prompt_tuple):
                self.prefix_cache.hits += 1
            else:
                self.prefix_cache.misses += 1
                fake_state = {"layers": [{"keys": "k", "values": "v"}]}
                self.prefix_cache.put(fake_state, capacity)

        tokens = [(token_ids[0] if token_ids else 1) * 10 + i for i in range(max_tokens)]
        self.calls.append({"token_ids": list(token_ids), "knobs": dict(knobs)})
        return {
            "logical_tokens": tokens,
            "text": f"text_for_{tokens[0]}",
            "prefill_ns": 300_000 if (self.prefix_cache and self.prefix_cache.hits > hits_before) else 2_000_000,
            "decode_ns": 1_000_000,
            "prefix_cache_hits": self.prefix_cache.hits if self.prefix_cache else 0,
            "knobs": dict(knobs),
        }


class PrefixCacheDirectTest(unittest.TestCase):
    """Direct tests for PrefixCache semantics."""

    def test_prefix_cache_matches_and_boundary(self) -> None:
        cache = PrefixCache([1, 2, 3])
        self.assertTrue(cache.matches([1, 2, 3, 4, 5]))
        self.assertFalse(cache.matches([1, 2, 4, 5]))
        self.assertFalse(cache.matches([1, 2]))
        self.assertEqual(cache.boundary([1, 2, 3, 4]), 3)
        self.assertEqual(cache.boundary([1, 2]), 2)

    def test_prefix_cache_get_put_capacity(self) -> None:
        cache = PrefixCache([1, 2, 3])
        self.assertIsNone(cache.get(128))

        state = {"layers": [{"keys": [1], "values": [2]}]}
        cache.put(state, 128)

        self.assertIsNotNone(cache.get(128))
        self.assertIsNone(cache.get(256), "different capacity must be a miss")


class IronMuleBackendPrefixCacheTest(unittest.TestCase):
    """Tests for PrefixCache lifecycle in IronMuleBackend."""

    def setUp(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.backend = IronMuleBackend(
            model="fake_model",
            tokenizer=self.tokenizer,
            model_id="test-model",
            model_revision="rev1",
        )
        self.backend._Engine = FakeEngine

    def test_initial_state_has_no_prefix_cache(self) -> None:
        self.assertIsNone(self.backend.prefix_cache)

    def test_set_prefix_cache_before_engine_instantiation(self) -> None:
        prefix = [10, 20, 30]
        self.backend.set_prefix_cache(prefix)
        self.assertIsNotNone(self.backend.prefix_cache)
        self.assertEqual(self.backend.prefix_cache.prefix_ids, (10, 20, 30))

        engine = self.backend._engine({})
        self.assertIs(engine.prefix_cache, self.backend.prefix_cache)

    def test_set_prefix_cache_propagates_to_existing_engines(self) -> None:
        engine1 = self.backend._engine({})
        engine2 = self.backend._engine({"head_skip_prefill": True})
        self.assertIsNone(engine1.prefix_cache)
        self.assertIsNone(engine2.prefix_cache)

        self.backend.set_prefix_cache([1, 2, 3])
        self.assertIs(engine1.prefix_cache, self.backend.prefix_cache)
        self.assertIs(engine2.prefix_cache, self.backend.prefix_cache)
        self.assertEqual(engine1.prefix_cache.prefix_ids, (1, 2, 3))

    def test_clear_prefix_cache_clears_all_engines(self) -> None:
        self.backend.set_prefix_cache([1, 2, 3])
        engine = self.backend._engine({})
        self.assertIsNotNone(engine.prefix_cache)

        self.backend.set_prefix_cache(None)
        self.assertIsNone(self.backend.prefix_cache)
        self.assertIsNone(engine.prefix_cache)

    def test_set_prefix_cache_accepts_instance(self) -> None:
        cache = PrefixCache([5, 6, 7])
        self.backend.set_prefix_cache(cache)
        self.assertIs(self.backend.prefix_cache, cache)

    def test_generate_transparently_passes_prefix_cache_hits(self) -> None:
        self.backend.set_prefix_cache([10, 20])
        res1 = self.backend.generate([10, 20, 30], 4, {})
        self.assertEqual(res1["prefix_cache_hits"], 0)
        self.assertEqual(len(res1["logical_tokens"]), 4)
        self.assertIn("text", res1)

        res2 = self.backend.generate([10, 20, 40], 4, {})
        self.assertEqual(res2["prefix_cache_hits"], 1)

    def test_generate_fallback_prefix_cache_hits_if_missing_from_engine(self) -> None:
        class IncompleteEngine(FakeEngine):
            def generate(self, prompt_ids, max_tokens, eos_ids):
                out = super().generate(prompt_ids, max_tokens, eos_ids)
                del out["prefix_cache_hits"]
                return out

        self.backend._Engine = IncompleteEngine
        self.backend.set_prefix_cache([1, 2])
        res = self.backend.generate([1, 2, 3], 4, {})
        self.assertIn("prefix_cache_hits", res)
        self.assertEqual(res["prefix_cache_hits"], 0)


class ServerPrefixCacheDelegationTest(unittest.TestCase):
    """Tests for Server.set_prefix_cache delegation to backend."""

    def test_set_prefix_cache_with_string_encodes_prompt(self) -> None:
        backend = MockBackend()
        server = Server(backend, make_profile())
        server.set_prefix_cache("abc")

        self.assertIsNotNone(backend.prefix_cache)
        self.assertEqual(backend.prefix_cache.prefix_ids, (ord("a"), ord("b"), ord("c")))

    def test_set_prefix_cache_with_ids(self) -> None:
        backend = MockBackend()
        server = Server(backend, make_profile())
        server.set_prefix_cache([10, 20, 30])

        self.assertIsNotNone(backend.prefix_cache)
        self.assertEqual(backend.prefix_cache.prefix_ids, (10, 20, 30))

    def test_set_prefix_cache_with_none(self) -> None:
        backend = MockBackend()
        server = Server(backend, make_profile())
        server.set_prefix_cache([1, 2])
        self.assertIsNotNone(backend.prefix_cache)

        server.set_prefix_cache(None)
        self.assertIsNone(backend.prefix_cache)

    def test_backend_without_set_prefix_cache_raises_attribute_error(self) -> None:
        class IncompatibleBackend:
            model_id = "test"
            model_revision = "rev"

            def encode(self, p):
                return [1]

            def generate(self, t, m, k):
                return {}

        server = Server(IncompatibleBackend(), make_profile())
        with self.assertRaises(AttributeError) as caught:
            server.set_prefix_cache([1, 2, 3])
        self.assertIn("does not implement set_prefix_cache", str(caught.exception))


class EndToEndPrefixCacheHitAndMissTest(unittest.TestCase):
    """End-to-end integration tests: cache miss on 1st call, cache hit on 2nd call, token identity."""

    def test_end_to_end_hit_miss_and_token_identity(self) -> None:
        backend = MockBackend(model_id="test-model", revision="rev1")
        prof = make_profile("head_skip")
        server = Server(backend, prof)

        # Set prefix cache for "SYSTEM: "
        prefix_str = "SYSTEM: "
        server.set_prefix_cache(prefix_str)

        # Request 1: Shared prefix + user query A
        prompt1 = prefix_str + "Question 1"
        gen1 = server.generate(prompt1, max_tokens=4)

        # First call must be a Cache Miss
        self.assertEqual(gen1.prefix_cache_hits, 0, "1st request must be a cache miss")
        self.assertEqual(len(gen1.tokens), 4)
        self.assertEqual(gen1.token_sha256, canonical_sha256(list(gen1.tokens)))
        self.assertEqual(gen1.plan, DEVICE_PROFILE_PLAN)

        # Request 2: Shared prefix + user query B
        prompt2 = prefix_str + "Question 2"
        gen2 = server.generate(prompt2, max_tokens=4)

        # Second call must be a Cache Hit
        self.assertEqual(gen2.prefix_cache_hits, 1, "2nd request with shared prefix must be a cache hit")
        self.assertEqual(len(gen2.tokens), 4)
        self.assertEqual(gen2.token_sha256, canonical_sha256(list(gen2.tokens)))

        # Request 3: Non-matching prompt (no prefix)
        unrelated_prompt = "OTHER PREFIX: Question 3"
        gen3 = server.generate(unrelated_prompt, max_tokens=4)

        # Non-matching request must NOT increase hits
        self.assertEqual(gen3.prefix_cache_hits, 1, "Unrelated prompt must not count as a hit")

        # Request 4: Another prompt sharing the original prefix
        prompt4 = prefix_str + "Question 4"
        gen4 = server.generate(prompt4, max_tokens=4)

        # Must be another Cache Hit
        self.assertEqual(gen4.prefix_cache_hits, 2, "Repeated shared prefix must increment hits to 2")

        # Request 5: Repeat prompt 1 exactly — tokens must be 100% bit-identical
        gen1_repeat = server.generate(prompt1, max_tokens=4)
        self.assertEqual(gen1_repeat.tokens, gen1.tokens, "Token identity gate: must match 100%")
        self.assertEqual(gen1_repeat.token_sha256, gen1.token_sha256)
        self.assertEqual(gen1_repeat.prefix_cache_hits, 3)


if __name__ == "__main__":
    unittest.main()
