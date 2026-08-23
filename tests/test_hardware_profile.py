import json
import tempfile
import unittest
from pathlib import Path

from friday_hardware import HardwareProfile, ProfileError, sample_budget


def profile(**overrides):
    """The measured 4B shape, unless a test needs to vary something."""

    base = dict(
        device="Apple M1 Max",
        model_id="mlx-community/gemma-3-4b-it-4bit",
        bits=4,
        group_size=64,
        layers=34,
        weight_gb=2.1832,
        per_layer_ms=0.16669,
        ms_per_gigabyte=2.79005,
        width_ms={1: 14.503, 2: 21.611, 3: 27.753, 4: 36.372, 5: 42.004,
                  6: 78.744, 8: 82.009, 16: 87.752, 32: 86.517,
                  48: 150.701, 64: 146.064},
        regression_widths=(6, 8, 48),
        measured_at="2026-08-23",
    )
    base.update(overrides)
    return HardwareProfile(**base)


class ProfileIdentityTests(unittest.TestCase):
    def test_refuses_a_configuration_it_never_saw(self):
        p = profile()
        p.require("mlx-community/gemma-3-4b-it-4bit", 4, 64)
        for wrong in (
            ("mlx-community/gemma-3-1b-it-4bit", 4, 64),
            ("mlx-community/gemma-3-4b-it-4bit", 8, 64),
            ("mlx-community/gemma-3-4b-it-4bit", 4, 32),
        ):
            self.assertFalse(p.applies_to(*wrong))
            with self.assertRaises(ProfileError):
                p.require(*wrong)

    def test_rejects_malformed_profiles(self):
        for bad in (
            dict(layers=0),
            dict(weight_gb=0.0),
            dict(per_layer_ms=-1.0),
            dict(width_ms={}),
            dict(width_ms={0: 1.0}),
            dict(segment_safety=0.0),
            dict(segment_safety=1.5),
            # A regression at a width that was never measured is a contradiction.
            dict(regression_widths=(7,)),
        ):
            with self.assertRaises(ProfileError):
                profile(**bad)


class WidthChoiceTests(unittest.TestCase):
    def test_returns_the_requested_width_when_it_is_fine(self):
        width, _ = profile().choose_width(4)
        self.assertEqual(width, 4)

    def test_never_returns_less_than_requested(self):
        p = profile()
        for requested in (1, 2, 3, 5, 7, 9, 17, 33):
            width, _ = p.choose_width(requested)
            self.assertGreaterEqual(width, requested)

    def test_avoids_measured_regressions(self):
        p = profile()
        for requested in range(1, 49):
            width, _ = p.choose_width(requested)
            self.assertNotIn(width, p.regression_widths)

    def test_takes_free_positions_when_a_wider_width_costs_no_more(self):
        # 16 costs 87.752 ms and 32 costs 86.517 ms: twice the positions, less time.
        width, reason = profile().choose_width(16)
        self.assertEqual(width, 32)
        self.assertIn("free", reason)

    def test_reports_when_the_request_exceeds_what_was_measured(self):
        width, reason = profile().choose_width(500)
        self.assertEqual(width, 64)
        self.assertIn("several passes", reason)

    def test_refuses_a_nonsensical_request(self):
        with self.assertRaises(ProfileError):
            profile().choose_width(0)

    def test_refuses_when_every_width_regressed(self):
        p = profile(width_ms={4: 10.0, 8: 30.0}, regression_widths=(4, 8))
        with self.assertRaises(ProfileError):
            p.usable_widths()


class SegmentationTests(unittest.TestCase):
    def test_segment_stays_inside_the_continuous_limit(self):
        p = profile()
        for width in p.width_ms:
            steps = p.steps_per_segment(width, 6.0)
            self.assertGreaterEqual(steps, 1)
            self.assertLessEqual(steps * p.width_ms[width] / 1000.0, 6.0)

    def test_a_cheaper_width_fits_more_steps(self):
        p = profile()
        self.assertGreater(p.steps_per_segment(1, 6.0), p.steps_per_segment(32, 6.0))

    def test_refuses_an_unmeasured_width(self):
        with self.assertRaises(ProfileError):
            profile().steps_per_segment(12, 6.0)

    def test_plan_covers_every_requested_token(self):
        plan = profile().plan(items=32, max_new_tokens=240, continuous_limit_s=6.0)
        self.assertEqual(plan.width, 32)
        self.assertGreaterEqual(plan.segments * plan.steps_per_segment, 240)
        self.assertGreater(plan.estimated_seconds, 0.0)
        self.assertIn("width", plan.as_dict())

    def test_plan_refuses_zero_tokens(self):
        with self.assertRaises(ProfileError):
            profile().plan(items=1, max_new_tokens=0, continuous_limit_s=6.0)


class CostModelTests(unittest.TestCase):
    def test_single_token_cost_matches_the_two_terms(self):
        p = profile()
        self.assertAlmostEqual(
            p.single_token_ms(), 34 * 0.16669 + 2.1832 * 2.79005, places=9
        )

    def test_dispatch_is_about_half_of_the_measured_4b_step(self):
        shares = profile().cost_shares()
        self.assertAlmostEqual(shares["dispatch_share"] + shares["bandwidth_share"], 1.0)
        self.assertTrue(0.4 < shares["dispatch_share"] < 0.55, shares)

    def test_a_slower_device_shifts_the_bottleneck_to_bandwidth(self):
        p = profile()
        fast = p.project(bandwidth_gb_s=358.4, per_layer_ms=0.16669)
        slow = p.project(bandwidth_gb_s=50.0, per_layer_ms=0.16669)
        self.assertLess(fast["bandwidth_share"], slow["bandwidth_share"])
        self.assertGreater(slow["bandwidth_share"], 0.8)
        # The fixed term damps the ratio: seven times less bandwidth is not seven
        # times the latency.
        self.assertLess(slow["ms_per_token"], fast["ms_per_token"] * 7.2)

    def test_projection_requires_both_device_parameters(self):
        p = profile()
        for bad in ({"bandwidth_gb_s": 0.0, "per_layer_ms": 0.2},
                    {"bandwidth_gb_s": 100.0, "per_layer_ms": 0.0}):
            with self.assertRaises(ProfileError):
                p.project(**bad)


class PersistenceTests(unittest.TestCase):
    def test_round_trip_preserves_every_decision(self):
        original = profile()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "profile.json"
            original.save(path)
            restored = HardwareProfile.load(path)
        self.assertEqual(restored.as_dict(), original.as_dict())
        self.assertEqual(restored.choose_width(16), original.choose_width(16))
        self.assertEqual(
            restored.plan(items=8, max_new_tokens=100, continuous_limit_s=6.0),
            original.plan(items=8, max_new_tokens=100, continuous_limit_s=6.0),
        )

    def test_unreadable_and_malformed_files_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "absent.json"
            with self.assertRaises(ProfileError):
                HardwareProfile.load(missing)
            broken = Path(tmp) / "broken.json"
            broken.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ProfileError):
                HardwareProfile.load(broken)
            incomplete = Path(tmp) / "incomplete.json"
            incomplete.write_text(json.dumps({"device": "x"}), encoding="utf-8")
            with self.assertRaises(ProfileError):
                HardwareProfile.load(incomplete)


class SampleBudgetTests(unittest.TestCase):
    def test_never_lands_in_the_range_that_measured_worse_than_one(self):
        for want in (True, False):
            self.assertNotIn(sample_budget(want_accuracy=want), range(5, 16))

    def test_accuracy_costs_more_samples(self):
        self.assertGreater(
            sample_budget(want_accuracy=True), sample_budget(want_accuracy=False)
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class SpeculationTests(unittest.TestCase):
    def test_break_even_rises_with_draft_length(self):
        p = profile()
        bars = [p.speculation_break_even(k) for k in (1, 2, 3, 4)]
        self.assertEqual(bars, sorted(bars), bars)
        # Measured 4B curve: one drafted token needs about half, four about
        # three quarters.
        self.assertTrue(0.4 < bars[0] < 0.55, bars)
        self.assertTrue(0.7 < bars[3] < 0.8, bars)

    def test_a_flatter_curve_lowers_the_bar(self):
        steep = profile(width_ms={1: 10.0, 2: 20.0}, regression_widths=())
        flat = profile(width_ms={1: 10.0, 2: 11.0}, regression_widths=())
        self.assertGreater(
            steep.speculation_break_even(1), flat.speculation_break_even(1)
        )

    def test_a_free_wider_pass_needs_no_acceptance_at_all(self):
        free = profile(width_ms={1: 10.0, 2: 9.5}, regression_widths=())
        self.assertEqual(free.speculation_break_even(1), 0.0)

    def test_an_impossible_curve_reports_impossible(self):
        hopeless = profile(width_ms={1: 10.0, 2: 30.0}, regression_widths=())
        self.assertEqual(hopeless.speculation_break_even(1), 1.0)

    def test_speedup_agrees_with_the_break_even_it_reports(self):
        p = profile()
        for k in (1, 2, 3, 4):
            bar = p.speculation_break_even(k)
            self.assertGreater(p.speculation_speedup(k, min(1.0, bar + 0.05)), 1.0)
            self.assertLess(p.speculation_speedup(k, max(0.0, bar - 0.05)), 1.0)

    def test_speculation_refuses_unmeasured_and_nonsensical_input(self):
        p = profile()
        with self.assertRaises(ProfileError):
            p.speculation_break_even(0)
        with self.assertRaises(ProfileError):
            p.speculation_break_even(11)   # width 12 was never measured
        with self.assertRaises(ProfileError):
            p.speculation_speedup(1, 1.5)
        self.assertEqual(p.speculation_speedup(0, 0.9), 1.0)
