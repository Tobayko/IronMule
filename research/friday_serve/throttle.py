"""Step back while the Mac belongs to someone else.

Serving reads no live system state today: the environment is tuned once at
startup from ``hw.memsize`` and nothing after that asks whether the machine is
still idle. This module adds the one missing input and two ways to act on it.

**It is not a knob.** It never touches :class:`ironmule.runtime.Knobs`, so it
cannot reach the authorised-knob marker check (``server._check_marker``,
``batcher._admit_session``) and cannot invalidate a device profile. It changes
*when* work happens and how many requests run at once, never *how* a token is
computed. Pacing is therefore output-identical by construction; the admission
cap rides on the batcher's existing, already-variable group width.

**Why the load threshold is relative.** ``ReadinessPolicy.max_load_1m = 0.75``
compares against the raw one-minute average, which on a ten-core machine means
7.5 % utilisation while the measured idle floor here is 4.0-6.0 — a limit that
was never once reached (``BACKLOG.md`` G1). A fixed number cannot work across
machines, so the floor is learned instead: the lowest load seen in a sliding
window is this machine's idle, and only the excess above it counts as somebody
else's work. That policy is untouched — it still gates *measurement* runs, where
fail-closed is the right answer.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

#: How often the cheap signal (load average) is read.
SAMPLE_SECONDS = 3.0
#: How often the expensive signals (``pmset``) are read. Power source and
#: thermal pressure change on human timescales, and polling them hard would
#: spend the very CPU this module exists to give back.
SLOW_SAMPLE_SECONDS = 30.0
#: Sliding window for the idle floor: 10 minutes at the default cadence. Long
#: enough to survive a build, short enough to follow a machine that gains a new
#: background daemon.
WINDOW = 200
#: Below this many samples the idle floor is not yet knowable — a single load
#: reading cannot tell "this Mac idles at 5.4" from "this Mac is busy at 5.4".
MIN_SAMPLES = 20

#: Excess load, in whole busy cores, above this machine's learned idle floor.
GENTLE_EXCESS_CORES = 0.75
MINIMAL_EXCESS_CORES = 2.0

# How much of the work just finished is handed back, as a fraction of its own
# duration. Relative on purpose: a fixed pause fires once per readback bundle,
# so the same number means an eight-fold different slowdown at readback_every=8
# than at 1, and a different one again on a faster machine or a smaller model.
# Measured (experiments/throttle_effect): a fixed 60 ms cost 84 % of decode
# throughput to buy 9 % for the foreign job — the right ratio to control is the
# duty cycle, not the sleep.
GENTLE_GIVE_BACK = 0.15
MINIMAL_GIVE_BACK = 1.0
#: A pathological bundle must not turn into a visible stall.
MAX_PAUSE_SECONDS = 0.25


@dataclass(frozen=True)
class Level:
    """One step-back setting and the observation that produced it."""

    name: str
    give_back: float
    width_share: float
    reason: str = ""

    def width_cap(self, max_width: int) -> int:
        """Never below one: a server that admits nothing reads as hung."""

        return max(1, int(max_width * self.width_share))

    def pause_for(self, work_seconds: float) -> float:
        """How long to stand aside after a piece of work of this length."""

        if self.give_back <= 0.0 or work_seconds <= 0.0:
            return 0.0
        return min(MAX_PAUSE_SECONDS, work_seconds * self.give_back)

    def because(self, reason: str) -> "Level":
        return Level(self.name, self.give_back, self.width_share, reason)


FULL = Level("full", 0.0, 1.0, "machine_idle")
GENTLE = Level("gentle", GENTLE_GIVE_BACK, 0.5)
MINIMAL = Level("minimal", MINIMAL_GIVE_BACK, 0.0)
OFF = Level("full", 0.0, 1.0, "throttle_disabled")


def _load_1m() -> float | None:
    try:
        return float(os.getloadavg()[0])
    except (OSError, ValueError):
        return None


def _power_and_thermal() -> dict[str, Any]:
    """AC, low-power and thermal pressure, or empty when unreadable.

    Reuses the readiness probe rather than re-parsing ``pmset``: it is
    stdlib-only, bounded by a timeout and an output cap, and already tested.
    """

    facts: dict[str, Any] = {}
    try:
        from friday_optimizer.readiness import MacSystemProbe, probe_thermal_status
    except Exception:
        return facts
    try:
        snapshot = MacSystemProbe().sample()
        facts["ac_connected"] = snapshot.ac_connected
        facts["low_power"] = snapshot.low_power
    except Exception:
        pass
    try:
        # `cpu_speed_limit` is a number macOS reports; the sibling `throttled`
        # flag is inferred from prose in the same output and is the weaker of
        # the two signals.
        facts["cpu_speed_limit"] = probe_thermal_status().get("cpu_speed_limit", 100)
    except Exception:
        pass
    return facts


def classify(
    load: float | None,
    idle: float | None,
    facts: dict[str, Any],
    *,
    samples: int,
) -> Level:
    """The whole policy, as a pure function so it can be tested without a Mac.

    The load average counts runnable work, so its unit is already "busy cores"
    and the thresholds do not scale with the core count: one core taken from you
    is one core taken from you on an M1 and on an M3 Max alike.
    """

    if facts.get("cpu_speed_limit", 100) < 100:
        return MINIMAL.because("thermal_pressure")
    if facts.get("ac_connected") is False:
        return GENTLE.because("on_battery")
    if facts.get("low_power") is True:
        return GENTLE.because("low_power_mode")
    if load is None:
        return GENTLE.because("load_unreadable")
    if samples < MIN_SAMPLES or idle is None:
        # Erring towards polite: one quiet minute at startup costs a tenth of
        # the decode rate, while erring the other way hands full speed to a
        # machine that is already busy — the case this exists for.
        return GENTLE.because("idle_load_unknown")
    excess = load - idle
    if excess >= MINIMAL_EXCESS_CORES:
        return MINIMAL.because("foreign_load_high")
    if excess >= GENTLE_EXCESS_CORES:
        return GENTLE.because("foreign_load")
    return FULL


class Throttle:
    """Samples the host in the background, publishes one :class:`Level`."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_width: int = 4,
        load_reader: Callable[[], float | None] = _load_1m,
        facts_reader: Callable[[], dict[str, Any]] = _power_and_thermal,
        interval: float = SAMPLE_SECONDS,
        slow_interval: float = SLOW_SAMPLE_SECONDS,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_width = max(1, int(max_width))
        self._load_reader = load_reader
        self._facts_reader = facts_reader
        self._interval = float(interval)
        self._slow_interval = float(slow_interval)
        self._loads: deque[float] = deque(maxlen=WINDOW)
        self._facts: dict[str, Any] = {}
        self._level = OFF if not self.enabled else GENTLE
        self._last_slow = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> "Throttle":
        if not self.enabled or self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._loop, name="friday-throttle", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.sample()
            except Exception:
                # A throttle that crashes the process it is meant to be gentle
                # with has failed at its one job; an unreadable host just means
                # the previous level stands.
                pass
            self._stop.wait(self._interval)

    # -- the one step ---------------------------------------------------------
    def sample(self, *, now: float | None = None) -> Level:
        """Read the host once and republish the level. Also the test entry."""

        moment = time.monotonic() if now is None else now
        if moment - self._last_slow >= self._slow_interval or not self._facts:
            self._facts = self._facts_reader()
            self._last_slow = moment
        load = self._load_reader()
        if load is not None:
            self._loads.append(load)
        self._level = classify(
            load,
            min(self._loads) if self._loads else None,
            self._facts,
            samples=len(self._loads),
        )
        return self._level

    # -- what serving asks ----------------------------------------------------
    @property
    def level(self) -> Level:
        return OFF if not self.enabled else self._level

    def width_cap(self, max_width: int | None = None) -> int:
        """Cap the caller's own width. The caller passes it rather than the
        throttle remembering it: a process-wide default configured for four
        would otherwise silently cap a batcher built for eight."""

        return self.level.width_cap(self.max_width if max_width is None else max_width)

    def pause(self, work_seconds: float) -> None:
        """Stand aside in proportion to the work just finished.

        Called right after the decode loop has synchronised with Metal, so the
        GPU is idle for the duration and the caller passes how long the bundle
        it just completed took.
        """

        seconds = self.level.pause_for(work_seconds)
        if seconds > 0.0:
            time.sleep(seconds)

    def as_dict(self) -> dict[str, Any]:
        level = self.level
        return {
            "enabled": self.enabled,
            "level": level.name,
            "reason": level.reason,
            "give_back_pct": round(level.give_back * 100, 1),
            "width_cap": self.width_cap(),
            "max_width": self.max_width,
            "idle_load": round(min(self._loads), 2) if self._loads else None,
            "load_1m": round(self._loads[-1], 2) if self._loads else None,
            "samples": len(self._loads),
        }


_GLOBAL: Throttle | None = None


def get_global_throttle() -> Throttle:
    """The process-wide throttle, matching ``telemetry.get_global_tracker``."""

    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = Throttle(enabled=False)
    return _GLOBAL


def set_global_throttle(throttle: Throttle) -> Throttle:
    global _GLOBAL
    _GLOBAL = throttle
    return throttle


__all__ = [
    "FULL",
    "GENTLE",
    "MINIMAL",
    "Level",
    "Throttle",
    "classify",
    "get_global_throttle",
    "set_global_throttle",
]
