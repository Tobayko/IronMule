"""friday_serve.throttle: the step-back policy, decided without a Mac.

``classify`` is a pure function of (load, idle floor, host facts), so the part
that can be wrong in a way that matters is testable offline. Synthetic snapshots
are the allowed kind here: this is decision logic, and nothing below claims a
performance, hardware or model result.
"""

from __future__ import annotations

import unittest

from friday_serve.throttle import (
    GENTLE_EXCESS_CORES,
    MAX_PAUSE_SECONDS,
    MIN_SAMPLES,
    MINIMAL_EXCESS_CORES,
    Level,
    Throttle,
    classify,
)

WARM = {"ac_connected": True, "low_power": False, "cpu_speed_limit": 100}


def level(load, idle, facts=None, samples=MIN_SAMPLES):
    return classify(load, idle, dict(facts or WARM), samples=samples)


class ClassifyTest(unittest.TestCase):
    def test_an_idle_mac_runs_at_full_speed(self) -> None:
        got = level(5.4, 5.3)
        self.assertEqual(got.name, "full")
        self.assertEqual(got.pause_for(0.09), 0.0)

    def test_the_threshold_is_excess_over_this_machines_idle_floor(self) -> None:
        # The same absolute load is idle on one machine and busy on another;
        # only the distance from the learned floor separates them. This is the
        # G1 finding: a fixed max_load_1m of 0.75 is unreachable at an idle
        # floor of 4.0-6.0.
        self.assertEqual(level(5.4, 5.3).name, "full")
        self.assertEqual(level(5.4, 0.2).name, "minimal")

    def test_foreign_load_steps_back_by_degree(self) -> None:
        idle = 4.0
        self.assertEqual(level(idle + GENTLE_EXCESS_CORES, idle).name, "gentle")
        self.assertEqual(level(idle + MINIMAL_EXCESS_CORES, idle).name, "minimal")

    def test_battery_and_low_power_are_their_own_reasons(self) -> None:
        battery = level(4.0, 4.0, {**WARM, "ac_connected": False})
        self.assertEqual((battery.name, battery.reason), ("gentle", "on_battery"))
        saving = level(4.0, 4.0, {**WARM, "low_power": True})
        self.assertEqual((saving.name, saving.reason), ("gentle", "low_power_mode"))

    def test_thermal_pressure_outranks_an_idle_reading(self) -> None:
        hot = level(4.0, 4.0, {**WARM, "cpu_speed_limit": 70})
        self.assertEqual((hot.name, hot.reason), ("minimal", "thermal_pressure"))

    def test_an_unknown_host_is_treated_politely_not_optimistically(self) -> None:
        self.assertEqual(level(None, 4.0).reason, "load_unreadable")
        self.assertEqual(level(4.0, 4.0, samples=MIN_SAMPLES - 1).reason, "idle_load_unknown")


class LevelTest(unittest.TestCase):
    def test_the_width_cap_never_reaches_zero(self) -> None:
        # A server that admits nothing looks hung; minimal still runs one.
        self.assertEqual(Level("minimal", 1.0, 0.0).width_cap(8), 1)
        self.assertEqual(Level("gentle", 0.15, 0.5).width_cap(8), 4)
        self.assertEqual(Level("gentle", 0.15, 0.5).width_cap(1), 1)
        self.assertEqual(Level("full", 0.0, 1.0).width_cap(8), 8)

    def test_the_pause_is_a_share_of_the_work_not_a_fixed_sleep(self) -> None:
        # A fixed sleep fires once per readback bundle, so it means an
        # eight-fold different slowdown at readback_every=8 than at 1.
        gentle = Level("gentle", 0.15, 0.5)
        self.assertAlmostEqual(gentle.pause_for(0.090), 0.0135)
        self.assertAlmostEqual(gentle.pause_for(0.720), 0.108)

    def test_a_pathological_bundle_cannot_become_a_stall(self) -> None:
        self.assertEqual(Level("minimal", 1.0, 0.0).pause_for(10.0), MAX_PAUSE_SECONDS)


class ThrottleTest(unittest.TestCase):
    def _throttle(self, loads, facts=None, **kwargs):
        series = list(loads)
        return Throttle(
            load_reader=lambda: series.pop(0) if series else None,
            facts_reader=lambda: dict(facts or WARM),
            **kwargs,
        )

    def test_the_idle_floor_is_the_minimum_of_the_window(self) -> None:
        throttle = self._throttle([6.0, 5.0, 4.2, 9.0], max_width=4)
        for _ in range(4):
            throttle.sample()
        self.assertEqual(throttle.as_dict()["idle_load"], 4.2)
        self.assertEqual(throttle.as_dict()["load_1m"], 9.0)

    def test_a_busy_mac_narrows_the_batch_and_pauses(self) -> None:
        throttle = self._throttle([4.0] * (MIN_SAMPLES + 1), max_width=8)
        for _ in range(MIN_SAMPLES):
            throttle.sample()
        self.assertEqual(throttle.level.name, "full")
        self.assertEqual(throttle.width_cap(), 8)
        throttle._load_reader = lambda: 4.0 + MINIMAL_EXCESS_CORES
        throttle.sample()
        self.assertEqual(throttle.level.name, "minimal")
        self.assertEqual(throttle.width_cap(), 1)

    def test_switched_off_it_reports_full_and_never_pauses(self) -> None:
        throttle = self._throttle([99.0] * 5, enabled=False, max_width=4)
        throttle.sample()
        self.assertEqual(throttle.level.name, "full")
        self.assertEqual(throttle.level.pause_for(0.09), 0.0)
        self.assertEqual(throttle.width_cap(), 4)

    def test_the_caller_keeps_its_own_width(self) -> None:
        # A process-wide throttle built for 4 must not cap a batcher built for 8.
        throttle = self._throttle([4.0] * 3, enabled=False, max_width=4)
        throttle.sample()
        self.assertEqual(throttle.width_cap(8), 8)

    def test_the_expensive_probe_is_not_read_on_every_sample(self) -> None:
        # Polling pmset hard would burn the CPU this module exists to give back.
        calls = {"n": 0}

        def facts():
            calls["n"] += 1
            return dict(WARM)

        throttle = Throttle(load_reader=lambda: 4.0, facts_reader=facts, slow_interval=30.0)
        throttle.sample(now=0.0)
        throttle.sample(now=1.0)
        throttle.sample(now=29.0)
        self.assertEqual(calls["n"], 1)
        throttle.sample(now=31.0)
        self.assertEqual(calls["n"], 2)


if __name__ == "__main__":
    unittest.main()
