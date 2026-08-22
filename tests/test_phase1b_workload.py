from __future__ import annotations

import unittest

import numpy as np

from friday_phase1b.constants import CASE_SEEDS
from friday_phase1b.workload import make_fixture


class Phase1BWorkloadTests(unittest.TestCase):
    def test_full_cancellation_fixture_is_deterministic_and_bit_zero(self) -> None:
        fixture = make_fixture(np, "cancellation", CASE_SEEDS["cancellation"])
        self.assertEqual(
            fixture.digest,
            "92f76cb32c546a26d1c7a67ee6fa1717b617237bdf28f803083d1302381cc440",
        )
        self.assertTrue(fixture.x.flags.c_contiguous)
        self.assertTrue(fixture.residual.flags.c_contiguous)
        self.assertTrue(fixture.weight.flags.c_contiguous)
        self.assertTrue(fixture.oracle.flags.c_contiguous)
        self.assertTrue(np.all(fixture.oracle.view(np.uint16) == 0))


if __name__ == "__main__":
    unittest.main()
