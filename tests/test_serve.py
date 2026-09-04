"""friday_serve: the request path, its scope check, and its refusal to guess.

Every test here runs against a fake engine. The point is not that generation
works — mlx does that — but that the *decisions* around it hold: an unverified
knob stays off, an out-of-scope request falls back, and a failed optimised path
stays failed across a restart.
"""

from __future__ import annotations

import unittest

from friday_calibrate.profile import DeviceProfile, KnobVerdict
from friday_runtime_core.breaker import PersistentLatch
from friday_runtime_core.controller import RuntimeExecutionError
from friday_serve.dispatch import explain, knobs_for
from friday_serve.scope import in_calibrated_scope, observe
from friday_serve.server import BASELINE_PLAN, DEVICE_PROFILE_PLAN, Server

MODEL = "mlx-community/gemma-3-4b-it-4bit"
REVISION = "abc123"


def profile(*verified: str, model_id: str = MODEL, revision: str = REVISION) -> DeviceProfile:
    verdicts = []
    for knob in ("head_skip", "fixed_compiled", "bundled_readback"):
        if knob in verified:
            verdicts.append(KnobVerdict(knob, "verified", 6, 0.9, 0.88, 0.93, True))
        else:
            verdicts.append(KnobVerdict(knob, "failed", 6, 1.0, 0.98, 1.02, True, "no gain"))
    return DeviceProfile(
        profile_id="device-test",
        model_id=model_id,
        model_revision=revision,
        hardware_sha256="a" * 64,
        environment_sha256="b" * 64,
        mde=0.006,
        knobs=tuple(verdicts),
    )


class Backend:
    """A fake engine that reports the knobs it was given, like IronMule does."""

    def __init__(self, *, fail_with=None, lies_about_knobs=False, extra_tokens=0):
        self.model_id = MODEL
        self.model_revision = REVISION
        self.calls = []
        self._fail_with = fail_with
        self._lies = lies_about_knobs
        self._extra = extra_tokens

    def encode(self, prompt):
        return [7] * max(1, len(prompt))

    def generate(self, token_ids, max_tokens, knobs):
        self.calls.append(dict(knobs))
        if self._fail_with is not None:
            raise self._fail_with
        reported = {} if self._lies else dict(knobs)
        return {
            "logical_tokens": list(range(max_tokens + self._extra)),
            "text": "answer",
            "prefill_ns": 1_000_000,
            "decode_ns": 2_000_000,
            "knobs": reported,
        }


class DispatchTest(unittest.TestCase):
    def test_only_verified_knobs_reach_the_engine(self) -> None:
        self.assertEqual(knobs_for(profile("head_skip")), {"head_skip_prefill": True})
        self.assertEqual(
            knobs_for(profile("head_skip", "bundled_readback")),
            {"head_skip_prefill": True, "readback_every": 8},
        )
        self.assertEqual(knobs_for(profile()), {})
        self.assertEqual(knobs_for(None), {})

    def test_dispatch_is_selectable_per_phase(self) -> None:
        both = profile("head_skip", "fixed_compiled")
        self.assertEqual(knobs_for(both, phases=("prefill",)), {"head_skip_prefill": True})
        self.assertEqual(
            knobs_for(both, phases=("decode",)), {"compiled_fixed_cache": True}
        )

    def test_explain_says_why_each_knob_is_off(self) -> None:
        described = explain(profile("head_skip"))
        self.assertTrue(described["knobs"]["head_skip"]["active"])
        self.assertFalse(described["knobs"]["fixed_compiled"]["active"])
        self.assertEqual(described["knobs"]["fixed_compiled"]["reason"], "no gain")
        self.assertEqual(explain(None)["reason"], "no_device_profile")


class ScopeTest(unittest.TestCase):
    def test_scope_comes_from_the_tokens_not_from_a_claim(self) -> None:
        scope = observe(
            model_id=MODEL, model_revision=REVISION, token_ids=[1, 2, 3], output_tokens=32
        )
        self.assertEqual(scope.prompt_tokens, 3)
        self.assertEqual(len(scope.prompt_sha256), 64)

    def test_an_underivable_request_has_no_scope(self) -> None:
        for kwargs in (
            {"token_ids": []},
            {"token_ids": "not tokens"},
            {"token_ids": [1, -2]},
            {"token_ids": [1, True]},
            {"output_tokens": 0},
            {"output_tokens": True},
            {"model_revision": ""},
        ):
            base = {
                "model_id": MODEL,
                "model_revision": REVISION,
                "token_ids": [1, 2],
                "output_tokens": 32,
            }
            base.update(kwargs)
            self.assertIsNone(observe(**base), kwargs)

    def test_a_different_model_or_revision_is_out_of_scope(self) -> None:
        current = profile()
        for kwargs, expected in (
            ({"model_id": "other/model"}, "model_mismatch"),
            ({"model_revision": "def456"}, "model_revision_mismatch"),
            ({"temperature": 0.7}, "sampling_out_of_scope"),
            ({"batch": 4}, "batch_out_of_scope"),
        ):
            base = {
                "model_id": MODEL,
                "model_revision": REVISION,
                "token_ids": [1, 2],
                "output_tokens": 32,
            }
            base.update(kwargs)
            allowed, reason = in_calibrated_scope(observe(**base), current)
            self.assertFalse(allowed)
            self.assertEqual(reason, expected)

    def test_a_profile_from_another_machine_is_out_of_scope(self) -> None:
        import dataclasses

        current = dataclasses.replace(profile("head_skip"), machine_sha256="c" * 64)
        scope = observe(
            model_id=MODEL, model_revision=REVISION, token_ids=[1, 2, 3], output_tokens=32
        )
        allowed, reason = in_calibrated_scope(scope, current)
        self.assertFalse(allowed)
        self.assertEqual(reason, "machine_mismatch")

        # a profile with the live machine hash still serves
        from friday_runtime_core.provenance import machine_sha256

        here = dataclasses.replace(profile("head_skip"), machine_sha256=machine_sha256())
        self.assertTrue(in_calibrated_scope(scope, here)[0])

    def test_prompt_content_does_not_narrow_the_scope(self) -> None:
        """Token identity is a property of the computation, not of one prompt."""

        current = profile("head_skip")
        first = observe(
            model_id=MODEL, model_revision=REVISION, token_ids=[1, 2, 3], output_tokens=32
        )
        second = observe(
            model_id=MODEL, model_revision=REVISION, token_ids=[9, 9], output_tokens=8
        )
        self.assertTrue(in_calibrated_scope(first, current)[0])
        self.assertTrue(in_calibrated_scope(second, current)[0])


class ServeTest(unittest.TestCase):
    def test_a_verified_profile_dispatches_its_knobs(self) -> None:
        backend = Backend()
        server = Server(backend, profile("head_skip", "fixed_compiled"))
        result = server.generate("hello", 4)
        self.assertEqual(result.plan, DEVICE_PROFILE_PLAN)
        self.assertEqual(
            backend.calls[-1], {"head_skip_prefill": True, "compiled_fixed_cache": True}
        )
        self.assertEqual(len(result.tokens), 4)
        self.assertEqual(len(result.token_sha256), 64)

    def test_without_a_profile_the_baseline_serves(self) -> None:
        backend = Backend()
        result = Server(backend, None).generate("hello", 4)
        self.assertEqual(result.plan, BASELINE_PLAN)
        self.assertEqual(result.reason, "no_device_profile")
        self.assertEqual(backend.calls[-1], {})

    def test_a_profile_that_verified_nothing_serves_the_baseline(self) -> None:
        backend = Backend()
        result = Server(backend, profile()).generate("hello", 4)
        self.assertEqual(result.plan, BASELINE_PLAN)
        self.assertEqual(result.reason, "no_verified_knob")
        self.assertEqual(backend.calls[-1], {})

    def test_an_out_of_scope_request_serves_the_baseline(self) -> None:
        backend = Backend()
        server = Server(backend, profile("head_skip"))
        result = server.generate("hello", 4, temperature=0.7)
        self.assertEqual(result.plan, BASELINE_PLAN)
        self.assertEqual(result.reason, "sampling_out_of_scope")
        self.assertEqual(backend.calls[-1], {})

    def test_an_engine_that_did_not_apply_the_knob_is_a_failure(self) -> None:
        backend = Backend(lies_about_knobs=True)
        server = Server(backend, profile("head_skip"))
        with self.assertRaises(RuntimeExecutionError) as caught:
            server.generate("hello", 4)
        self.assertIn("circuit breaker latched", str(caught.exception))
        self.assertIn("authorised knob was not applied", str(caught.exception.__cause__))
        self.assertEqual(server.controller.circuit_reason, "RuntimeExecutionError")

    def test_more_tokens_than_requested_is_a_failure(self) -> None:
        server = Server(Backend(extra_tokens=3), profile("head_skip"))
        with self.assertRaises(RuntimeExecutionError):
            server.generate("hello", 4)

    def test_an_optimised_failure_latches_and_the_next_process_starts_on_baseline(self) -> None:
        store: list[str] = []
        latch = lambda: PersistentLatch(  # noqa: E731
            load=lambda: store[0] if store else None, append=store.append
        )
        first = Server(Backend(fail_with=ValueError("kernel")), profile("head_skip"), latch=latch())
        with self.assertRaises(RuntimeExecutionError):
            first.generate("hello", 4)
        self.assertEqual(store, ["ValueError"])

        backend = Backend()
        restarted = Server(backend, profile("head_skip"), latch=latch())
        result = restarted.generate("hello", 4)
        self.assertEqual(result.plan, BASELINE_PLAN)
        self.assertEqual(result.reason, "circuit_breaker_latched")
        self.assertEqual(backend.calls[-1], {})

    def test_a_baseline_failure_does_not_latch(self) -> None:
        server = Server(Backend(fail_with=ValueError("kernel")), profile())
        with self.assertRaises(RuntimeExecutionError) as caught:
            server.generate("hello", 4)
        self.assertIn("baseline path failed", str(caught.exception))
        self.assertIsNone(server.controller.circuit_reason)

    def test_explain_reports_the_breaker(self) -> None:
        server = Server(Backend(fail_with=ValueError("kernel")), profile("head_skip"))
        with self.assertRaises(RuntimeExecutionError):
            server.generate("hello", 4)
        self.assertEqual(server.explain()["circuit_reason"], "ValueError")


class LatchWiringTest(unittest.TestCase):
    def test_cli_latch_returns_a_persistent_latch(self) -> None:
        # _latch had no return statement, so serve got a MemoryLatch and the
        # breaker never survived a restart.
        import tempfile
        from pathlib import Path

        from friday_serve import cli

        with tempfile.TemporaryDirectory() as d:
            latch = cli._latch(Path(d) / "device-profile.sqlite3")
        self.assertIsInstance(latch, PersistentLatch)


if __name__ == "__main__":
    unittest.main()
