"""Comprehensive tests for streaming execution engine and generator in friday_serve.

Validates:
1. Exact Token Identity: tokens from stream_generate() match generate() bit-for-bit.
2. TTFT (Time To First Token) measurement and pacing:
   - First token emitted immediately with is_first=True and prefill_ns > 0.
   - Subsequent tokens emitted in readback chunks with is_first=False.
   - Final 'done' event contains total tokens, decode_ns, total_ns, and knobs.
3. EOS recognition:
   - When first token is EOS: immediate exit after first token + done event (no decode loop).
   - When EOS occurs during decode: stream terminates at EOS without subsequent tokens.
   - Exact max_tokens bounds (max_tokens=1, max_tokens=N).
4. Stateful Prefix Caching:
   - Prefix cache hit increments hits counter and propagates to TTFT and done events.
5. Server stream_generate integration:
   - Scope and circuit breaker gating.
   - Latched circuit breaker on backend failure during streaming.
   - Verification of authorised knobs and rejection of unapplied knobs.
   - Adaptive RL-Controller reward feedback on stream completion.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import mlx.core as mx

from friday_calibrate.profile import DeviceProfile, KnobVerdict
from friday_runtime_core.breaker import PersistentLatch
from friday_runtime_core.controller import RuntimeExecutionError
from friday_serve.dispatch import knobs_for
from friday_serve.ironmule_backend import IronMuleBackend
from friday_serve.rl_controller import AdaptiveRLController
from friday_serve.server import BASELINE_PLAN, DEVICE_PROFILE_PLAN, Server

# Ensure ironmule worktree is in path for tests
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IRONMULE = PROJECT_ROOT / ".worktrees" / "friday-optimizer-ironmule"
if str(IRONMULE) not in sys.path:
    sys.path.insert(0, str(IRONMULE))

from ironmule.runtime import BASELINE, Knobs, PrefixCache  # noqa: E402

MODEL = "mlx-community/gemma-3-4b-it-4bit"
REVISION = "rev1"


def make_profile(*verified: str, model_id: str = MODEL, revision: str = REVISION) -> DeviceProfile:
    verdicts = [
        KnobVerdict(k, "verified" if k in verified else "failed", 6, 0.9, 0.88, 0.93, True)
        for k in ("head_skip", "fixed_compiled", "bundled_readback")
    ]
    return DeviceProfile(
        profile_id="device-stream-test",
        model_id=model_id,
        model_revision=revision,
        hardware_sha256="a" * 64,
        environment_sha256="b" * 64,
        mde=0.006,
        knobs=tuple(verdicts),
    )


class MockTokenizer:
    def __init__(self, eos_ids: tuple[int, ...] = (2, 100)) -> None:
        self.eos_token_id = eos_ids[0]
        self.eos_token_ids = eos_ids

    def apply_chat_template(self, messages, add_generation_prompt=True):
        content = messages[0]["content"]
        return [ord(c) for c in content]

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, token_ids: Sequence[int]) -> str:
        return f"tok[{','.join(str(t) for t in token_ids)}]"


class StreamTestEngine:
    """Deterministic MLX-backed Engine implementation for stream verification."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        knobs: Knobs = BASELINE,
        *,
        token_sequence: list[int] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.knobs = knobs
        self.prefix_cache: PrefixCache | None = None
        self.token_sequence = (
            list(token_sequence) if token_sequence is not None else [10, 20, 30, 40, 50, 60, 70, 80]
        )
        self._pos = 0

    def _capacity(self, prompt_len: int, max_tokens: int) -> int:
        needed = prompt_len + max_tokens + self.knobs.speculate_k + self.knobs.capacity_slack
        return ((needed + 63) // 64) * 64

    def _prefill(self, prompt_ids: list[int], capacity: int):
        if self.prefix_cache is not None:
            split = self.prefix_cache.boundary(prompt_ids)
            stored = self.prefix_cache.get(capacity) if self.prefix_cache.matches(prompt_ids) else None
            if stored is not None:
                self.prefix_cache.hits += 1
            else:
                self.prefix_cache.misses += 1
                if self.prefix_cache.matches(prompt_ids):
                    self.prefix_cache.put({"layers": [{"keys": mx.array([0]), "values": mx.array([0])}]}, capacity)

        self._pos = 0
        first_token = self.token_sequence[0] if self.token_sequence else 10
        token = mx.array([[first_token]])
        state = {
            "position": {"offset": mx.array(len(prompt_ids), dtype=mx.int32)},
            "layers": [{"keys": mx.array([0]), "values": mx.array([0])}],
        }
        return state, token

    def _picks(self, out):
        return out[0]

    def _body(self, capacity: int, width: int):
        def body(input_ids, state):
            self._pos += 1
            idx = self._pos
            val = self.token_sequence[idx] if idx < len(self.token_sequence) else 99
            if width > 1:
                picks = mx.array([[val] * width])
            else:
                picks = mx.array([[val]])
            new_state = {
                "position": {"offset": state["position"]["offset"] + input_ids.shape[1]},
                "layers": state["layers"],
            }
            return picks, new_state

        return body

    def generate(
        self, prompt_ids: list[int], max_tokens: int, eos_ids: tuple[int, ...]
    ) -> dict[str, Any]:
        capacity = self._capacity(len(prompt_ids), max_tokens)
        state, token = self._prefill(prompt_ids, capacity)
        first = int(token.reshape((-1,)).item())
        if first in eos_ids:
            physical, decode_ns = [first], 0
        else:
            physical = [first]
            body = self._body(capacity, 1)
            pending: list[Any] = []
            curr = token
            every = max(1, self.knobs.readback_every)
            for step in range(max_tokens - 1):
                out = body(curr, state)
                picks = self._picks(out)
                curr, state = picks[:, -1:], out[1]
                pending.append(curr)
                if len(pending) == every or step == max_tokens - 2:
                    mx.eval(*pending)
                    mx.synchronize()
                    physical.extend(int(item.reshape((-1,)).item()) for item in pending)
                    if any(v in eos_ids for v in physical[-len(pending):]):
                        break
                    pending = []
            decode_ns = 100_000

        logical = []
        for v in physical:
            logical.append(v)
            if v in eos_ids:
                break

        return {
            "physical_tokens": physical,
            "logical_tokens": logical,
            "visible_tokens": [t for t in logical if t not in eos_ids],
            "prefill_ns": 200_000,
            "decode_ns": decode_ns,
            "total_ns": 200_000 + decode_ns,
            "capacity": capacity,
            "acceptance": 0.0,
            "prefix_cache_hits": self.prefix_cache.hits if self.prefix_cache else 0,
            "knobs": self.knobs.as_dict(),
        }


class TestStreamBackend(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = MockTokenizer(eos_ids=(2, 100))
        self.backend = IronMuleBackend(
            model=object(),
            tokenizer=self.tokenizer,
            model_id=MODEL,
            model_revision=REVISION,
        )
        self.tokens = [10, 20, 30, 40, 50, 60, 70, 80]
        # Replace default engine factory with test engine
        self.backend._Engine = lambda m, t, knobs: StreamTestEngine(
            m, t, knobs, token_sequence=self.tokens
        )

    def test_token_identity_greedy_baseline(self) -> None:
        """Tokens from stream_generate must match generate() exactly in greedy baseline."""
        prompt = [1, 2, 3]
        max_tokens = 6
        knobs = {}

        gen_result = self.backend.generate(prompt, max_tokens, knobs)
        expected_tokens = gen_result["logical_tokens"]

        stream_events = list(self.backend.stream_generate(prompt, max_tokens, knobs))
        streamed_tokens: list[int] = []
        for event in stream_events:
            if event["type"] == "token":
                if event["is_first"]:
                    streamed_tokens.append(event["token"])
                else:
                    streamed_tokens.extend(event["tokens"])

        self.assertEqual(streamed_tokens, expected_tokens)
        self.assertEqual(len(streamed_tokens), max_tokens)

        # Final done event validation
        done_event = stream_events[-1]
        self.assertEqual(done_event["type"], "done")
        self.assertEqual(done_event["total_tokens"], max_tokens)
        self.assertEqual(done_event["logical_tokens"], expected_tokens)
        self.assertGreater(done_event["decode_ns"], 0)
        self.assertEqual(done_event["total_ns"], stream_events[0]["prefill_ns"] + done_event["decode_ns"])

    def test_token_identity_batched_readback(self) -> None:
        """Token identity holds under readback_every=3 chunking."""
        prompt = [5, 6, 7]
        max_tokens = 7
        knobs = {"readback_every": 3}

        # Clear cached engine to use updated knobs
        self.backend._engines.clear()
        gen_result = self.backend.generate(prompt, max_tokens, knobs)
        expected_tokens = gen_result["logical_tokens"]

        self.backend._engines.clear()
        stream_events = list(self.backend.stream_generate(prompt, max_tokens, knobs))
        streamed_tokens: list[int] = []
        chunk_sizes: list[int] = []

        for event in stream_events:
            if event["type"] == "token":
                if event["is_first"]:
                    streamed_tokens.append(event["token"])
                else:
                    streamed_tokens.extend(event["tokens"])
                    chunk_sizes.append(len(event["tokens"]))

        self.assertEqual(streamed_tokens, expected_tokens)
        # With max_tokens=7: token 0 is first (1). Remaining 6 tokens come in chunks of 3 and 3
        self.assertEqual(chunk_sizes, [3, 3])

    def test_ttft_and_pacing(self) -> None:
        """First token is yielded immediately with is_first=True and valid TTFT timing."""
        prompt = [1, 2]
        max_tokens = 4
        events = list(self.backend.stream_generate(prompt, max_tokens, {}))

        first_event = events[0]
        self.assertEqual(first_event["type"], "token")
        self.assertTrue(first_event["is_first"])
        self.assertEqual(first_event["token"], self.tokens[0])
        self.assertGreater(first_event["prefill_ns"], 0)
        self.assertEqual(first_event["prefix_cache_hits"], 0)

        # Subsequent token events
        for event in events[1:-1]:
            self.assertEqual(event["type"], "token")
            self.assertFalse(event["is_first"])
            self.assertIn("tokens", event)

        # Last event
        done_event = events[-1]
        self.assertEqual(done_event["type"], "done")
        self.assertEqual(done_event["total_tokens"], 4)

    def test_eos_immediate_on_first_token(self) -> None:
        """When first token is EOS, generator yields first token and done immediately."""
        eos_seq = [2, 10, 20, 30]  # token 2 is EOS
        self.backend._Engine = lambda m, t, knobs: StreamTestEngine(
            m, t, knobs, token_sequence=eos_seq
        )
        self.backend._engines.clear()

        events = list(self.backend.stream_generate([1, 2], max_tokens=10, knobs={}))
        self.assertEqual(len(events), 2)

        # Event 1: first token
        self.assertEqual(events[0]["type"], "token")
        self.assertTrue(events[0]["is_first"])
        self.assertEqual(events[0]["token"], 2)

        # Event 2: done
        self.assertEqual(events[1]["type"], "done")
        self.assertEqual(events[1]["total_tokens"], 1)
        self.assertEqual(events[1]["decode_ns"], 0)
        self.assertEqual(events[1]["logical_tokens"], [2])

    def test_eos_mid_stream(self) -> None:
        """When EOS appears during decode, stream halts exactly at EOS."""
        eos_seq = [10, 20, 2, 40, 50]  # token 2 is EOS at index 2
        self.backend._Engine = lambda m, t, knobs: StreamTestEngine(
            m, t, knobs, token_sequence=eos_seq
        )
        self.backend._engines.clear()

        events = list(self.backend.stream_generate([1, 2], max_tokens=10, knobs={"readback_every": 1}))
        tokens_emitted: list[int] = []
        for e in events:
            if e["type"] == "token":
                tokens_emitted.append(e["token"])

        self.assertEqual(tokens_emitted, [10, 20, 2])
        done_event = events[-1]
        self.assertEqual(done_event["type"], "done")
        self.assertEqual(done_event["total_tokens"], 3)
        self.assertEqual(done_event["logical_tokens"], [10, 20, 2])

    def test_prefix_cache_hits_in_streaming(self) -> None:
        """Prefix cache hits reflect in first token and done events."""
        prefix_prompt = [42, 43, 44]
        self.backend.set_prefix_cache(prefix_prompt)

        # First request with prefix prompt -> cache miss
        events1 = list(self.backend.stream_generate(prefix_prompt + [99], max_tokens=3, knobs={}))
        self.assertEqual(events1[0]["prefix_cache_hits"], 0)
        self.assertEqual(events1[-1]["prefix_cache_hits"], 0)

        # Second request with matching prefix -> cache hit
        events2 = list(self.backend.stream_generate(prefix_prompt + [100], max_tokens=3, knobs={}))
        self.assertEqual(events2[0]["prefix_cache_hits"], 1)
        self.assertEqual(events2[-1]["prefix_cache_hits"], 1)

    def test_max_tokens_single_token(self) -> None:
        """max_tokens=1 yields first token and done immediately."""
        events = list(self.backend.stream_generate([1, 2], max_tokens=1, knobs={}))
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["type"], "token")
        self.assertTrue(events[0]["is_first"])
        self.assertEqual(events[1]["type"], "done")
        self.assertEqual(events[1]["total_tokens"], 1)
        self.assertEqual(events[1]["decode_ns"], 0)

    def test_speculative_stream_generate(self) -> None:
        """stream_generate with speculate_k > 0 yields tokens with draft matching."""
        prompt = [1, 2, 3, 4, 1, 2]
        knobs = {"speculate_k": 2, "speculate_ngram": 2}
        self.backend._Engine = lambda m, t, k: StreamTestEngine(
            m, t, k, token_sequence=[10, 20, 30, 40, 50, 60]
        )
        self.backend._engines.clear()

        events = list(self.backend.stream_generate(prompt, max_tokens=5, knobs=knobs))
        streamed_tokens: list[int] = []
        for e in events:
            if e["type"] == "token":
                if e["is_first"]:
                    streamed_tokens.append(e["token"])
                else:
                    streamed_tokens.extend(e["tokens"])

        self.assertEqual(len(streamed_tokens), 5)
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["total_tokens"], 5)


class FakeServerBackend:
    """Fake backend for testing Server.stream_generate gating and breaker."""

    def __init__(self, *, fail_in_stream: bool = False, lies_about_knobs: bool = False) -> None:
        self.model_id = MODEL
        self.model_revision = REVISION
        self.fail_in_stream = fail_in_stream
        self.lies_about_knobs = lies_about_knobs
        self.calls: list[dict[str, Any]] = []

    def encode(self, prompt: str) -> list[int]:
        return [ord(c) for c in prompt]

    def generate(self, token_ids: Sequence[int], max_tokens: int, knobs: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "logical_tokens": list(range(max_tokens)),
            "text": "test",
            "prefill_ns": 100,
            "decode_ns": 200,
            "knobs": dict(knobs),
        }

    def stream_generate(
        self, token_ids: Sequence[int], max_tokens: int, knobs: Mapping[str, Any]
    ) -> Iterator[dict[str, Any]]:
        self.calls.append(dict(knobs))
        if self.fail_in_stream:
            raise RuntimeError("Metal GPU fault")

        reported_knobs = {} if self.lies_about_knobs else dict(knobs)
        yield {
            "type": "token",
            "token": 1,
            "tokens": [1],
            "text": "t1",
            "is_first": True,
            "prefill_ns": 100_000,
            "prefix_cache_hits": 0,
        }

        for i in range(1, max_tokens):
            yield {
                "type": "token",
                "token": i + 1,
                "tokens": [i + 1],
                "text": f"t{i+1}",
                "is_first": False,
            }

        yield {
            "type": "done",
            "total_tokens": max_tokens,
            "decode_ns": 200_000,
            "total_ns": 300_000,
            "knobs": reported_knobs,
            "prefix_cache_hits": 0,
            "logical_tokens": list(range(1, max_tokens + 1)),
        }


class TestServerStreamGenerate(unittest.TestCase):
    def test_verified_profile_dispatches_knobs(self) -> None:
        """Server dispatches verified knobs to backend during stream_generate."""
        backend = FakeServerBackend()
        profile = make_profile("head_skip", "fixed_compiled")
        server = Server(backend, profile)

        events = list(server.stream_generate("hello", max_tokens=4))
        self.assertEqual(backend.calls[-1], {"head_skip_prefill": True, "compiled_fixed_cache": True})
        done_event = events[-1]
        self.assertEqual(done_event["type"], "done")
        self.assertEqual(done_event["plan"], DEVICE_PROFILE_PLAN)

    def test_out_of_scope_falls_back_to_baseline(self) -> None:
        """Out of scope request falls back to baseline greedy plan."""
        backend = FakeServerBackend()
        server = Server(backend, make_profile("head_skip"))

        events = list(server.stream_generate("hello", max_tokens=4, temperature=0.7))
        self.assertEqual(backend.calls[-1], {})
        done_event = events[-1]
        self.assertEqual(done_event["plan"], BASELINE_PLAN)
        self.assertEqual(done_event["reason"], "sampling_out_of_scope")

    def test_supports_token_sequence_prompt(self) -> None:
        """Prompt can be provided directly as Sequence[int]."""
        backend = FakeServerBackend()
        server = Server(backend, make_profile("head_skip"))

        events = list(server.stream_generate([65, 66, 67], max_tokens=3))
        self.assertEqual(len([e for e in events if e["type"] == "token"]), 3)

    def test_circuit_breaker_latches_on_stream_failure(self) -> None:
        """Failure during streaming latches circuit breaker on optimised path."""
        store: list[str] = []
        latch = lambda: PersistentLatch(  # noqa: E731
            load=lambda: store[0] if store else None, append=store.append
        )
        failing_backend = FakeServerBackend(fail_in_stream=True)
        server = Server(failing_backend, make_profile("head_skip"), latch=latch())

        with self.assertRaises(RuntimeExecutionError) as caught:
            list(server.stream_generate("hello", max_tokens=4))

        self.assertIn("circuit breaker latched", str(caught.exception))
        self.assertEqual(store, ["RuntimeError"])

        # Subsequent call restarts on baseline
        ok_backend = FakeServerBackend()
        restarted = Server(ok_backend, make_profile("head_skip"), latch=latch())
        events = list(restarted.stream_generate("hello", max_tokens=4))
        self.assertEqual(ok_backend.calls[-1], {})
        self.assertEqual(events[-1]["plan"], BASELINE_PLAN)
        self.assertEqual(events[-1]["reason"], "circuit_breaker_latched")

    def test_engine_lying_about_knobs_trips_breaker(self) -> None:
        """When engine lies about applied knobs, breaker trips."""
        lying_backend = FakeServerBackend(lies_about_knobs=True)
        server = Server(lying_backend, make_profile("head_skip"))

        with self.assertRaises(RuntimeExecutionError) as caught:
            list(server.stream_generate("hello", max_tokens=4))

        self.assertIn("circuit breaker latched", str(caught.exception))
        self.assertIn("authorised knob was not applied", str(caught.exception.__cause__))

    def test_rl_controller_updates_reward_after_stream(self) -> None:
        """AdaptiveRLController receives reward observation after complete stream."""
        backend = FakeServerBackend()
        profile = make_profile("head_skip")
        rl = AdaptiveRLController()

        server = Server(backend, profile, rl_controller=rl)
        self.assertEqual(rl.history, [])

        # Initially, baseline action is chosen (reward 0.0)
        list(server.stream_generate("hello", max_tokens=4))
        self.assertEqual(len(rl.history), 1)
        self.assertEqual(rl.history[0]["action"], "baseline")
        self.assertEqual(rl.history[0]["reward"], 0.0)

        # Train full_optimized action to have high reward
        rl.observe_reward("full_optimized", MODEL, 5, 4, 1.0)

        # Now full_optimized is selected, producing reward 0.15 for verified knobs
        list(server.stream_generate("hello", max_tokens=4))
        self.assertEqual(len(rl.history), 3)
        self.assertEqual(rl.history[-1]["action"], "full_optimized")
        self.assertEqual(rl.history[-1]["reward"], 0.15)


if __name__ == "__main__":
    unittest.main()
