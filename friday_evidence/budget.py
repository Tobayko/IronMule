"""Fail-closed runtime accounting for the approved H1/H2 hardware budgets."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

from .registry import BudgetPolicy, DEFAULT_BUDGET_POLICY


class BudgetError(RuntimeError):
    """A measurement exceeded or could not satisfy a registered budget."""


class BudgetGuard:
    """Track wall/GPU work, bounded continuous load, duty cycle, and cooldowns."""

    def __init__(
        self,
        policy: BudgetPolicy = DEFAULT_BUDGET_POLICY,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.policy = policy
        self._clock = clock
        self._sleeper = sleeper
        self._started = clock()
        self._last_candidate_finished: float | None = None
        self._gpu_events: list[tuple[float, float]] = []
        self._last_gpu_end: float | None = None
        self._continuous_gpu_seconds = 0.0
        self.gpu_work_seconds = 0.0
        self.max_continuous_gpu_seconds = 0.0
        self.cooldown_seconds = 0.0
        self.required_break_seconds = 0.0

    def check_wall(self) -> None:
        if self._clock() - self._started > self.policy.wall_limit_s:
            raise BudgetError("wall budget exceeded")

    def _verified_sleep(self, seconds: float, label: str) -> float:
        before = self._clock()
        self._sleeper(seconds)
        elapsed = self._clock() - before
        if elapsed + 1e-9 < seconds:
            raise BudgetError(f"{label} did not elapse in real time")
        return elapsed

    def before_candidate(self) -> None:
        self.check_wall()
        if self._last_candidate_finished is None:
            return
        remaining = self.policy.candidate_cooldown_s - (
            self._clock() - self._last_candidate_finished
        )
        if remaining > 0:
            self.cooldown_seconds += self._verified_sleep(remaining, "candidate cooldown")
        self._continuous_gpu_seconds = 0.0
        self.check_wall()

    def finish_candidate(self) -> None:
        self._last_candidate_finished = self._clock()
        self.check_wall()

    def required_break(self) -> None:
        self.required_break_seconds += self._verified_sleep(
            self.policy.required_break_s, "required break"
        )
        self._continuous_gpu_seconds = 0.0
        self.check_wall()

    def record_gpu(self, seconds: float) -> None:
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise BudgetError("GPU work must be a finite nonnegative number")
        seconds = float(seconds)
        continuous = seconds
        if not math.isfinite(seconds) or seconds < 0:
            raise BudgetError("GPU work must be a finite nonnegative number")
        if not math.isfinite(continuous) or continuous < 0:
            raise BudgetError("continuous GPU work must be finite and nonnegative")
        if continuous > self.policy.continuous_gpu_limit_s:
            raise BudgetError("continuous GPU work budget exceeded")

        now = self._clock()
        event_start = now - seconds
        if (
            self._last_gpu_end is not None
            and event_start - self._last_gpu_end >= self.policy.required_break_s
        ):
            self._continuous_gpu_seconds = 0.0
        self._continuous_gpu_seconds += continuous
        if self._continuous_gpu_seconds > self.policy.continuous_gpu_limit_s:
            raise BudgetError("continuous GPU work budget exceeded")
        self.gpu_work_seconds += seconds
        self.max_continuous_gpu_seconds = max(
            self.max_continuous_gpu_seconds, self._continuous_gpu_seconds
        )
        if self.gpu_work_seconds > self.policy.gpu_work_limit_s:
            raise BudgetError("GPU work budget exceeded")

        self._gpu_events.append((event_start, now))
        self._last_gpu_end = now
        cutoff = now - self.policy.duty_window_s
        self._gpu_events = [event for event in self._gpu_events if event[1] >= cutoff]
        rolling_gpu = sum(
            max(0.0, end - max(start, cutoff)) for start, end in self._gpu_events
        )
        if rolling_gpu > self.policy.duty_window_s * self.policy.duty_cycle_limit:
            raise BudgetError("rolling GPU duty-cycle budget exceeded")
        self.check_wall()

    def summary(self) -> dict[str, float]:
        return {
            "gpu_work_seconds": round(self.gpu_work_seconds, 6),
            "max_continuous_gpu_seconds": round(self.max_continuous_gpu_seconds, 6),
            "cooldown_seconds": round(self.cooldown_seconds, 6),
            "required_break_seconds": round(self.required_break_seconds, 6),
            "wall_seconds": round(self._clock() - self._started, 6),
            "gpu_work_limit_seconds": self.policy.gpu_work_limit_s,
            "continuous_gpu_limit_seconds": self.policy.continuous_gpu_limit_s,
            "duty_cycle_limit": self.policy.duty_cycle_limit,
            "wall_limit_seconds": self.policy.wall_limit_s,
            "candidate_cooldown_seconds": self.policy.candidate_cooldown_s,
            "required_break_limit_seconds": self.policy.required_break_s,
        }


__all__ = ["BudgetError", "BudgetGuard"]
