"""The pre-screen that decides whether a prompt family may enter candidate 5.

Its job is to be cheap and to refuse. Admitting a family with a degenerate
position costs a gated hardware run that ends in an identity break saying
nothing about chunking — which is exactly what candidates 1 and 2 spent.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "prefill_step_size"))

from screen import (  # noqa: E402
    MEASURED_PERTURBATION,
    GapError,
    degenerate_threshold,
    screen,
)


def gaps(values):
    return [{"position": index, "gap": value} for index, value in enumerate(values)]


class ScreenTest(unittest.TestCase):
    def test_p2s_own_prompt_is_rejected(self) -> None:
        """The screen has to catch the workload that already broke."""

        payload = json.loads(
            (ROOT / "experiments" / "identity_forensics" / "logit_gap.json").read_text()
        )
        result = screen(payload["reference_gaps"])
        self.assertEqual(result.verdict, "degenerate")
        self.assertIn(10, result.degenerate_positions)
        self.assertEqual(result.min_gap, 0.5)

    def test_half_of_p2s_positions_are_degenerate_not_one(self) -> None:
        """The plan says "a" degenerate position; the measurement says eight."""

        payload = json.loads(
            (ROOT / "experiments" / "identity_forensics" / "logit_gap.json").read_text()
        )
        result = screen(payload["reference_gaps"])
        self.assertEqual(result.positions, 16)
        self.assertEqual(len(result.degenerate_positions), 8)

    def test_a_family_with_wide_gaps_is_admissible(self) -> None:
        result = screen(gaps([12.0] * 16))
        self.assertEqual(result.verdict, "admissible")
        self.assertEqual(result.degenerate_positions, ())
        self.assertTrue(result.as_dict()["admissible"])

    def test_one_narrow_position_is_enough_to_reject(self) -> None:
        values = [12.0] * 16
        values[7] = 0.4
        result = screen(gaps(values))
        self.assertEqual(result.verdict, "degenerate")
        self.assertEqual(result.degenerate_positions, (7,))

    def test_the_threshold_follows_the_measured_perturbation(self) -> None:
        self.assertEqual(degenerate_threshold(), MEASURED_PERTURBATION + 1.0)
        wide = screen(gaps([3.0] * 16), perturbation=1.0)
        narrow = screen(gaps([3.0] * 16), perturbation=5.0)
        self.assertEqual(wide.verdict, "admissible")
        self.assertEqual(narrow.verdict, "degenerate")

    def test_thin_evidence_is_not_a_pass(self) -> None:
        result = screen(gaps([12.0] * 3))
        self.assertEqual(result.verdict, "insufficient_evidence")
        self.assertFalse(result.as_dict()["admissible"])

    def test_malformed_evidence_is_refused_not_guessed(self) -> None:
        for payload in ([{"position": 0}], [{"gap": "wide"}], [{"gap": -1.0}], "gaps"):
            with self.assertRaises(GapError):
                screen(payload)


if __name__ == "__main__":
    unittest.main()
