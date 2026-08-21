"""Offline regression tests for roofline generation pacing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from measure_roofline import pace_generation  # noqa: E402


class FakeGuard:
    def __init__(self) -> None:
        self.breaks = 0

    def required_break(self) -> None:
        self.breaks += 1


class RooflinePacingTests(unittest.TestCase):
    def test_first_generation_does_not_wait(self) -> None:
        guard = FakeGuard()

        pace_generation(guard, 0)

        self.assertEqual(guard.breaks, 0)

    def test_every_later_generation_requires_a_break(self) -> None:
        guard = FakeGuard()

        for index in range(1, 7):
            pace_generation(guard, index)

        self.assertEqual(guard.breaks, 6)

    def test_negative_index_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pace_generation(FakeGuard(), -1)


if __name__ == "__main__":
    unittest.main()
