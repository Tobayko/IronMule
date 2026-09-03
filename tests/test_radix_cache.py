"""Unit tests for Radix-Tree Global Prefix Caching."""

from __future__ import annotations

import unittest
from friday_serve.radix_cache import RadixCache, RadixNode


class TestRadixCache(unittest.TestCase):
    def test_basic_insert_and_match(self):
        cache = RadixCache(max_tokens=1000)
        tokens = (10, 20, 30, 40)
        dummy_state = {"kv": "state_1"}

        cache.insert(tokens, dummy_state)
        matched_len, state, nodes = cache.match_prefix((10, 20, 30, 40))

        self.assertEqual(matched_len, 4)
        self.assertEqual(state, dummy_state)
        self.assertEqual(cache.hits, 1)

    def test_tag_mismatch_is_a_miss(self):
        # A KV state built under one knob signature must not be served to a
        # request with a different one.
        cache = RadixCache(max_tokens=1000)
        head_skip = ("rev1", (("head_skip_prefill", True),))
        baseline = ("rev1", ())
        cache.insert((10, 20, 30, 40), {"kv": "head_skip"}, tag=head_skip)

        miss_len, miss_state, _ = cache.match_prefix((10, 20, 30, 40, 50), tag=baseline)
        self.assertEqual(miss_len, 0)
        self.assertIsNone(miss_state)

        hit_len, hit_state, _ = cache.match_prefix((10, 20, 30, 40, 50), tag=head_skip)
        self.assertEqual(hit_len, 4)
        self.assertEqual(hit_state, {"kv": "head_skip"})

    def test_partial_prefix_match(self):
        cache = RadixCache(max_tokens=1000)
        system_prompt = (1, 2, 3, 4, 5)
        dummy_state = {"kv": "system_prompt_state"}

        cache.insert(system_prompt, dummy_state)

        # Incoming request with same system prompt + user query (99, 100)
        query = (1, 2, 3, 4, 5, 99, 100)
        matched_len, state, nodes = cache.match_prefix(query)

        self.assertEqual(matched_len, 5)
        self.assertEqual(state, dummy_state)
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.tokens_saved, 5)

    def test_branch_splitting(self):
        cache = RadixCache(max_tokens=1000)
        cache.insert((10, 20, 30, 40), {"state": "A"})
        cache.insert((10, 20, 50, 60), {"state": "B"})

        # Match branch A
        len_a, state_a, _ = cache.match_prefix((10, 20, 30, 40))
        self.assertEqual(len_a, 4)
        self.assertEqual(state_a, {"state": "A"})

        # Match branch B
        len_b, state_b, _ = cache.match_prefix((10, 20, 50, 60))
        self.assertEqual(len_b, 4)
        self.assertEqual(state_b, {"state": "B"})

        # Match common prefix only
        len_c, state_c, _ = cache.match_prefix((10, 20, 99, 100))
        self.assertEqual(len_c, 0)  # intermediate node has no state set unless inserted

    def test_eviction_under_budget(self):
        # Set tiny budget of 15 tokens
        cache = RadixCache(max_tokens=15)
        cache.insert((1, 2, 3, 4, 5, 6, 7, 8), {"state": "first"})
        cache.insert((10, 20, 30, 40, 50, 60, 70, 80), {"state": "second"})

        self.assertLessEqual(cache.total_cached_tokens, 15)

    def test_no_match(self):
        cache = RadixCache(max_tokens=1000)
        cache.insert((1, 2, 3), {"state": "test"})
        matched_len, state, _ = cache.match_prefix((9, 8, 7))
        self.assertEqual(matched_len, 0)
        self.assertIsNone(state)
        self.assertEqual(cache.misses, 1)


if __name__ == "__main__":
    unittest.main()
