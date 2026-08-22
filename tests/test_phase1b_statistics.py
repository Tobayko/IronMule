from __future__ import annotations

import unittest

from friday_phase1b.constants import BASELINE_NAMES
from friday_phase1b.statistics import (
    StatisticsError,
    hierarchical_ratio,
    select_baseline,
    session_ratio,
)


class Phase1BStatisticsTests(unittest.TestCase):
    def test_hierarchical_identity_and_known_win(self) -> None:
        identity = [([100.0] * 31, [100.0] * 31) for _ in range(3)]
        aa = hierarchical_ratio(identity, seed=0xB16B2AA0)
        self.assertEqual(aa["ratio"], 1.0)
        self.assertEqual(aa["ci95_low"], 1.0)
        self.assertEqual(aa["ci95_high"], 1.0)
        win = hierarchical_ratio(
            [([90.0] * 31, [100.0] * 31) for _ in range(3)], seed=0xB16B3AB0
        )
        self.assertAlmostEqual(win["ratio"], 0.9)
        self.assertLess(win["ci95_high"], 1.0)

    def test_baseline_tie_uses_frozen_precedence(self) -> None:
        sessions = []
        for _ in range(3):
            values = {name: [100.0] * 15 for name in BASELINE_NAMES}
            values["eager_transparent"] = [99.8] * 15
            sessions.append(values)
        selected = select_baseline(sessions)
        self.assertEqual(selected["selected"], "fast_rms_norm")
        self.assertIn("eager_transparent", selected["eligible"])

    def test_invalid_timing_is_rejected(self) -> None:
        with self.assertRaises(StatisticsError):
            session_ratio([0.0], [1.0])
        with self.assertRaises(StatisticsError):
            hierarchical_ratio([([1.0], [1.0])], seed=1)


if __name__ == "__main__":
    unittest.main()
