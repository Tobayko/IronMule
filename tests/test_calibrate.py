"""friday_calibrate: the decision logic, run offline against a fake engine.

The hardware path is one thin function (`build_runner`); everything that decides
a verdict takes a `run(knobs) -> Sample` callable, so the part that can be wrong
in a way that matters is testable without a GPU.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from friday_calibrate import plan
from friday_calibrate.profile import (
    CALIBRATED_KNOBS,
    HISTORY,
    DeviceProfile,
    KnobVerdict,
    ProfileError,
    newest_profile,
)
from friday_calibrate.runner import (
    CalibrationError,
    Sample,
    calibrate,
    draft_width_curve,
    noise_mde,
    paired_arms,
    verdict_for,
)
from friday_runtime_core.history import RuntimeHistory
from friday_runtime_core.provenance import ProvenanceSpec, collect_provenance


def engine(*, gains=None, identity_break=False, jitter=0.0):
    """A deterministic fake engine: each knob multiplies decode or prefill time."""

    gains = gains or {}
    state = {"calls": 0}

    def run(knobs):
        state["calls"] += 1
        ttft, tps = 1.7851, 71.0
        for name, value in knobs.items():
            factor = gains.get(name)
            if factor is None:
                continue
            if name in ("head_skip_prefill",):
                ttft *= factor
            else:
                tps /= factor
        # A tiny deterministic wobble keeps the paired ratios from being exactly equal.
        wobble = 1.0 + jitter * ((state["calls"] % 3) - 1)
        digest = "b" * 64 if (identity_break and state["calls"] % 2 == 0) else "a" * 64
        return Sample(
            ttft_seconds=ttft * wobble, decode_tps=tps / wobble, tokens=32, token_sha256=digest
        )

    return run


class PlanTest(unittest.TestCase):
    def test_the_plan_is_printable_and_budgeted_without_hardware(self) -> None:
        described = plan.as_dict()
        names = [step["name"] for step in described["steps"]]
        self.assertEqual(names[0], "aa_noise")
        self.assertIn("knob:head_skip", names)
        self.assertLess(described["gpu_seconds_estimate"], 30 * 60 * 0.15)
        self.assertFalse(described["formal_claim"])

    def test_prefill_step_size_is_declared_not_calibrated_with_a_reason(self) -> None:
        described = plan.as_dict()
        self.assertIn("prefill_step_size", described["not_calibrated"])
        self.assertNotIn("prefill_step_size", plan.KNOB_TO_ENGINE)


class PairingTest(unittest.TestCase):
    def test_order_alternates_so_neither_arm_is_always_warmer(self) -> None:
        baseline, candidate, breaks = paired_arms(engine(), {"compiled_fixed_cache": True}, pairs=4)
        self.assertEqual((), breaks)
        self.assertEqual([s.order for s in baseline], ["AB", "BA", "AB", "BA"])
        self.assertEqual(len(candidate), 4)

    def test_a_single_identity_break_ends_the_run(self) -> None:
        baseline, _candidate, breaks = paired_arms(
            engine(identity_break=True), {"compiled_fixed_cache": True}, pairs=6
        )
        self.assertEqual(breaks, ("token_identity_broken:pair_0",))
        self.assertEqual(baseline, [])


class VerdictTest(unittest.TestCase):
    def test_a_real_gain_is_verified(self) -> None:
        run = engine(gains={"compiled_fixed_cache": 0.90}, jitter=0.002)
        verdict = verdict_for("fixed_compiled", run, pairs=6, mde=0.01)
        self.assertEqual(verdict.verdict, "verified")
        self.assertLess(verdict.ci_high, 1.0)
        self.assertTrue(verdict.token_identical)
        self.assertEqual(verdict.phase, "decode")

    def test_no_gain_is_not_verified(self) -> None:
        run = engine(gains={"compiled_fixed_cache": 1.0}, jitter=0.002)
        verdict = verdict_for("fixed_compiled", run, pairs=6, mde=0.01)
        self.assertNotEqual(verdict.verdict, "verified")
        self.assertTrue(verdict.reason)

    def test_a_regression_is_not_verified(self) -> None:
        run = engine(gains={"compiled_fixed_cache": 1.15}, jitter=0.002)
        self.assertNotEqual(
            verdict_for("fixed_compiled", run, pairs=6, mde=0.01).verdict, "verified"
        )

    def test_an_identity_break_fails_the_knob_and_never_verifies_it(self) -> None:
        run = engine(gains={"compiled_fixed_cache": 0.5}, identity_break=True)
        verdict = verdict_for("fixed_compiled", run, pairs=6, mde=0.01)
        self.assertEqual(verdict.verdict, "failed")
        self.assertIn("token_identity_broken", verdict.reason)

    def test_a_knob_without_an_engine_counterpart_is_not_applicable(self) -> None:
        verdict = verdict_for("prefill_step_size", engine(), pairs=6, mde=0.01)
        self.assertEqual(verdict.verdict, "not_applicable")
        self.assertIn("one forward", verdict.reason)


class NoiseTest(unittest.TestCase):
    def test_aa_noise_becomes_the_mde(self) -> None:
        spread, mde = noise_mde(engine(jitter=0.003), pairs=6)
        self.assertEqual(spread, mde)
        self.assertGreater(mde, 0.0)
        self.assertLess(mde, 0.5)

    def test_an_aa_identity_break_is_a_harness_fault_not_a_verdict(self) -> None:
        with self.assertRaises(CalibrationError):
            noise_mde(engine(identity_break=True), pairs=6)


class WidthCurveTest(unittest.TestCase):
    def test_the_curve_covers_the_bandit_action_space(self) -> None:
        curve = draft_width_curve(engine(gains={"speculate_k": 1.1}))
        self.assertEqual(sorted(curve), [0, 2, 3, 4, 8])
        self.assertTrue(all(value > 0 for value in curve.values()))


class ProfileTest(unittest.TestCase):
    def _profile(self, **overrides) -> DeviceProfile:
        base = dict(
            profile_id="device-test",
            model_id="m",
            model_revision="r",
            hardware_sha256="a" * 64,
            environment_sha256="b" * 64,
            mde=0.006,
            aa_noise=0.003,
            knobs=(
                KnobVerdict("head_skip", "verified", 6, 0.877, 0.86, 0.89, True),
                KnobVerdict("fixed_compiled", "failed", 6, 1.01, 0.99, 1.03, True, "no gain"),
                KnobVerdict("prefill_step_size", "not_applicable", reason="single forward"),
            ),
        )
        base.update(overrides)
        return DeviceProfile(**base)

    def test_only_verified_knobs_are_offered_to_serving(self) -> None:
        profile = self._profile()
        self.assertEqual(profile.verified_knobs(), ("head_skip",))
        self.assertEqual(profile.verified_knobs("decode"), ())
        self.assertFalse(profile.is_verified("fixed_compiled"))
        self.assertFalse(profile.is_verified("bundled_readback"))  # never measured
        self.assertIn("bundled_readback", profile.unverified())

    def test_a_verified_verdict_cannot_be_claimed_without_its_evidence(self) -> None:
        for kwargs in (
            {"token_identical": False, "ratio": 0.9, "ci_high": 0.95},
            {"token_identical": True, "ratio": None, "ci_high": None},
            {"token_identical": True, "ratio": 0.99, "ci_high": 1.02},
            {"token_identical": True, "ratio": 0.9, "ci_high": 0.95, "pairs": 0},
        ):
            fields = {"pairs": 6, "ci_low": 0.85, **kwargs}
            with self.assertRaises(ProfileError):
                KnobVerdict("head_skip", "verified", **fields)

    def test_an_unregistered_knob_cannot_enter_a_profile(self) -> None:
        with self.assertRaises(ProfileError):
            KnobVerdict("kv_cache_quantisation", "verified", 6, 0.5, 0.4, 0.6, True)

    def test_the_report_round_trips_and_the_digest_is_checked(self) -> None:
        profile = self._profile()
        report = profile.as_report("cal-1")
        self.assertEqual(DeviceProfile.from_report(report).verified_knobs(), ("head_skip",))
        tampered = dict(report)
        tampered["mde"] = 0.5
        with self.assertRaises(ProfileError):
            DeviceProfile.from_report(tampered)

    def test_a_profile_survives_the_hash_chain(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "profile.sqlite3"
            provenance = collect_provenance(
                ProvenanceSpec(
                    runtime_id=HISTORY.runtime_id,
                    code_directories=("friday_calibrate",),
                    spec_files=("AGENTS.md",),
                ),
                require_clean=False,
            )
            with RuntimeHistory.open(HISTORY, path, initialize=True) as history:
                history.persist(self._profile().as_report("cal-1"), provenance)
            with RuntimeHistory.open(HISTORY, path, read_only=True) as history:
                with history.read_transaction():
                    rows = history.verified_records()
            self.assertEqual(newest_profile(rows).verified_knobs(), ("head_skip",))
            self.assertIsNone(newest_profile([]))

    def test_a_fabricated_profile_without_aa_noise_is_skipped_for_serving(self) -> None:
        # tools/autotune.py wrote verified verdicts with no A/A noise run into
        # the sealed chain; newest_profile must fall back past them.
        with TemporaryDirectory() as directory:
            path = Path(directory) / "profile.sqlite3"
            provenance = collect_provenance(
                ProvenanceSpec(
                    runtime_id=HISTORY.runtime_id,
                    code_directories=("friday_calibrate",),
                    spec_files=("AGENTS.md",),
                ),
                require_clean=False,
            )
            good = self._profile().as_report("cal-real")
            fabricated = self._profile(
                profile_id="device-fake", aa_noise=None
            ).as_report("cal-fake")
            with RuntimeHistory.open(HISTORY, path, initialize=True) as history:
                history.persist(good, provenance)
                history.persist(fabricated, provenance)
            with RuntimeHistory.open(HISTORY, path, read_only=True) as history:
                with history.read_transaction():
                    rows = history.verified_records()
            chosen = newest_profile(rows)
            self.assertEqual(chosen.profile_id, "device-test")
            self.assertEqual(chosen.verified_knobs(), ("head_skip",))


class CalibrateTest(unittest.TestCase):
    def test_a_full_calibration_produces_one_verdict_per_knob(self) -> None:
        run = engine(
            gains={"head_skip_prefill": 0.88, "compiled_fixed_cache": 0.93, "readback_every": 1.0},
            jitter=0.002,
        )
        profile = calibrate(
            run,
            {"model_id": "m", "model_revision": "r"},
            hardware_sha256="a" * 64,
            environment_sha256="b" * 64,
            pairs=6,
            profile_id="device-test",
        )
        self.assertEqual(
            sorted(v.knob for v in profile.knobs), sorted(CALIBRATED_KNOBS)
        )
        self.assertIn("head_skip", profile.verified_knobs("prefill"))
        self.assertIn("fixed_compiled", profile.verified_knobs("decode"))
        # readback_every had no effect in this fake engine, so it must not verify.
        self.assertNotIn("bundled_readback", profile.verified_knobs())
        self.assertEqual(profile.verdict_for("prefill_step_size").verdict, "not_applicable")
        self.assertGreater(profile.mde, 0.0)
        self.assertEqual(sorted(profile.width_curve), [0, 2, 3, 4, 8])


if __name__ == "__main__":
    unittest.main()


def test_on_break_hands_out_both_sequences_and_changes_nothing():
    """The dump must be free: same pairs, same digests, same verdict with and
    without ``on_break``. Without this, a run that fails to reproduce an
    identity break could always blame the changed measuring core."""

    from friday_calibrate.runner import Sample, paired_arms

    def engine(diverge_at):
        calls = {"n": 0}

        def run(knobs):
            calls["n"] += 1
            speculative = bool(knobs)
            tokens = (1, 2, 3) if not (speculative and calls["n"] >= diverge_at) else (1, 2, 4)
            digest = "a" * 64 if tokens == (1, 2, 3) else "b" * 64
            return Sample(1.0, 32.0, len(tokens), digest, token_ids=tokens)

        return run

    seen = []
    plain = paired_arms(engine(99), {"speculate_k": 2}, pairs=4)
    dumped = paired_arms(
        engine(99), {"speculate_k": 2}, pairs=4,
        on_break=lambda index, left, right: seen.append(index),
    )
    assert [s.as_dict() for s in plain[0]] == [s.as_dict() for s in dumped[0]]
    assert [s.as_dict() for s in plain[1]] == [s.as_dict() for s in dumped[1]]
    assert plain[2] == dumped[2] == ()
    assert seen == []

    captured = []
    broken = paired_arms(
        engine(2), {"speculate_k": 2}, pairs=4,
        on_break=lambda index, left, right: captured.append((index, left.token_ids, right.token_ids)),
    )
    assert broken[2] == ("token_identity_broken:pair_0",)
    assert captured == [(0, (1, 2, 3), (1, 2, 4))]
