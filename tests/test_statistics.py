import unittest

from friday_h0.constants import AA_BOOTSTRAP_SEEDS
from friday_h0.statistics import (
    StatisticsError,
    aa_gate,
    hierarchical_bootstrap,
    iqr,
    mad,
    median,
    session_ratio,
    set_ratio,
)


def aa_sessions(scale=1.0):
    return [
        [(100.0 + block, (100.0 + block) * scale) for block in range(30)]
        for _ in range(3)
    ]


class StatisticsTests(unittest.TestCase):
    def test_robust_statistics(self):
        self.assertEqual(median([3, 1, 2]), 2.0)
        self.assertEqual(median([1, 2, 3, 4]), 2.5)
        self.assertEqual(mad([1, 2, 3]), 1.0)
        self.assertEqual(iqr([1, 2, 3, 4]), 1.5)
        self.assertAlmostEqual(set_ratio([0.9, 1.0, 1.1]), 1.0)

    def test_session_ratio_direction_and_replay(self):
        self.assertAlmostEqual(session_ratio([(10, 12), (20, 24)]), 1.2)
        first = hierarchical_bootstrap(aa_sessions(), AA_BOOTSTRAP_SEEDS["characterization"], replicates=10_000)
        second = hierarchical_bootstrap(aa_sessions(), AA_BOOTSTRAP_SEEDS["characterization"], replicates=10_000)
        self.assertEqual(first, second)

    def test_aa_gate_tie_and_invalid(self):
        result = aa_gate(aa_sessions(), aa_sessions())
        self.assertEqual(result["classification"], "tie")
        self.assertTrue(result["engineering_equivalence_gate"])
        self.assertEqual(result["sets"]["characterization"]["bootstrap_replicates"], 10_000)
        self.assertEqual(result["sets"]["characterization"]["prng"], "splitmix64_v1")
        self.assertEqual(result["prng"], "splitmix64_v1")
        invalid = aa_gate(aa_sessions(1.10), aa_sessions(1.10))
        self.assertEqual(invalid["classification"], "h0_invalid")
        with self.assertRaises(StatisticsError):
            aa_gate(aa_sessions()[:2], aa_sessions())
        with self.assertRaises(StatisticsError):
            aa_gate([[pair for pair in session[:29]] for session in aa_sessions()], aa_sessions())

    def test_invalid_data_is_rejected(self):
        with self.assertRaises(StatisticsError):
            median([])
        with self.assertRaises(StatisticsError):
            session_ratio([(1, 0)])
