"""AG4 step 2: two serving paths on one card, and the agent routes every request between them (BACKLOG AG4).

AG4-Q run 2 measured on a T4 with Qwen 3 1.7B that neither path wins everywhere. The `float16` reference path made
57.7 tokens a second for a request alone, with its first token after 0.06 s. The stacked `dense` engine made 290.9
at width 8, but 47.3 alone, and its first token came much later. So the agent keeps both loaded and sends each
request where it is expected to finish first: the single path while it is free, and the stacked engine once
requests queue.

For each path it learns online, by recursive least squares, what a token costs as a function of how many requests
share the path (`seconds per token = a + b * (sharing - 1)`), and the running mean of the time to the first token
and of the tokens a request makes. It starts from AG4's measurement on this card, so the first request is already
priced. A decision is arithmetic over two paths, a few microseconds. Ties go to the first path, so a quiet server
behaves as the single path alone. It never changes how a path computes: each path's answers are its own, and both
passed AG4's gate.
"""

from __future__ import annotations

import copy
import math
import statistics
import threading
import time
from typing import Any, Iterator, Sequence

FORGET = 0.995      # as the agent's world model: recent requests weigh more, so drift is tracked
MEAN_RATE = 0.05    # running means of first-token seconds and tokens per request


class PathCost:
    """One path's learned prices: per-token seconds at a sharing level, first-token seconds, tokens a request."""

    def __init__(self, capacity: int, step_s: float, step_s_per_row: float, first_s: float, tokens: float = 128.0):
        self.capacity = max(1, int(capacity))
        # In milliseconds per token, where recursive least squares' unit observation noise is about right; the
        # prior variance (10 ms^2) lets the measured prices hold for a few requests and then follow the data.
        self.theta = [float(step_s) * 1000, float(step_s_per_row) * 1000]
        self.p = [[10.0, 0.0], [0.0, 10.0]]
        self.first_s, self.tokens, self.seen = float(first_s), float(tokens), 0

    def step(self, sharing: int) -> float:
        sharing = max(1, min(self.capacity, sharing))
        return max(1e-6, (self.theta[0] + self.theta[1] * (sharing - 1)) / 1000)

    def expected(self, outstanding: int, tokens: float) -> float:
        """Seconds until a new request that finds `outstanding` others here finishes: its wait for a free row (whole
        services of the rows ahead), its first token and its tokens at the sharing it finds."""
        sharing = min(self.capacity, outstanding + 1)
        rounds = max(0, outstanding + 1 - self.capacity)
        service = self.first_s + self.tokens * self.step(self.capacity)
        wait = math.ceil(rounds / self.capacity) * service if rounds else 0.0
        return wait + self.first_s + tokens * self.step(sharing)

    def learn(self, sharing: int, tokens: int, decode_s: float, first_s: float | None) -> None:
        if tokens > 1 and decode_s > 0:
            x = [1.0, float(max(1, min(self.capacity, sharing)) - 1)]
            y = decode_s / (tokens - 1) * 1000
            px = [self.p[0][0] * x[0] + self.p[0][1] * x[1], self.p[1][0] * x[0] + self.p[1][1] * x[1]]
            gain_den = FORGET + x[0] * px[0] + x[1] * px[1]
            k = [px[0] / gain_den, px[1] / gain_den]
            err = y - (self.theta[0] * x[0] + self.theta[1] * x[1])
            self.theta = [self.theta[0] + k[0] * err, max(0.0, self.theta[1] + k[1] * err)]
            self.p = [[(self.p[i][j] - k[i] * px[j]) / FORGET for j in range(2)] for i in range(2)]
        if first_s is not None and first_s >= 0 and sharing <= 1:  # a first token that did not wait in a queue
            self.first_s += MEAN_RATE * (first_s - self.first_s)
        if tokens > 0:
            self.tokens += MEAN_RATE * (tokens - self.tokens)
        self.seen += 1

    def status(self) -> dict[str, Any]:
        return {"capacity": self.capacity, "step_ms": round(self.theta[0], 3),
                "step_ms_per_row": round(self.theta[1], 3), "first_ms": round(self.first_s * 1000, 1),
                "tokens": round(self.tokens, 1), "requests": self.seen}


class PathRouter:
    """Routes each request to the path where it is expected to finish first; learns from every finished one."""

    def __init__(self, names: Sequence[str], costs: Sequence[PathCost]):
        if len(names) != len(costs) or len(names) < 2:
            raise ValueError("the router needs one cost model per path, and at least two paths")
        self.names, self.costs = list(names), list(costs)
        self.decisions = [0] * len(names)
        self._lock = threading.Lock()
        self._decision_ns = 0
        self._decision_ns_total = self._decision_ns_max = self._decisions_timed = 0
        self.errors, self.last_error = 0, None
        self._errors_lock = threading.Lock()

    def _timed(self, started: int) -> None:  # the caller holds `_lock`
        self._decision_ns = time.perf_counter_ns() - started
        self._decision_ns_total += self._decision_ns
        self._decision_ns_max = max(self._decision_ns_max, self._decision_ns)
        self._decisions_timed += 1

    def _failed(self, where: str, exc: BaseException) -> None:
        """A decision or learning error never reaches a request: it is counted and shown in `/health`."""
        with self._errors_lock:
            self.errors += 1
            self.last_error = f"{where}: {type(exc).__name__}: {exc}"[:300]

    @classmethod
    def from_setup(cls, setup: dict[str, Any], stacked_width: int | None = None) -> "PathRouter":
        """From AG4's measurement on this card: the single path's step and first token, the stacked engine's step
        at one row and at its width (AG5: the agent's chosen width, else 8)."""
        stacked_width = stacked_width or setup.get("width") or 8
        rows = list(setup["stacked"]["speed"].values())
        mean = lambda key: sum(r[key] for r in rows) / len(rows)  # noqa: E731
        one = sum(r["plan_step_s"]["1"] for r in rows) / len(rows)
        wide = sum(r["plan_step_s"][str(stacked_width)] for r in rows) / len(rows)
        single = PathCost(1, mean("base_step_s"), 0.0, mean("base_first_s"))
        stacked = PathCost(stacked_width, one, max(0.0, (wide - one) / (stacked_width - 1)), mean("plan_first_s"))
        return cls(["single", "stacked"], [single, stacked])

    def choose(self, outstanding: Sequence[int], max_tokens: int) -> int:
        started = time.perf_counter_ns()
        with self._lock:
            try:
                tokens = [min(float(max_tokens), cost.tokens) for cost in self.costs]
                expected = [cost.expected(n, t) for cost, n, t in zip(self.costs, outstanding, tokens)]
                best = min(range(len(expected)), key=lambda i: (expected[i], i))
            except Exception as exc:  # noqa: BLE001 - a failed decision serves on the first path
                self._failed("choose", exc)
                best = 0
            self.decisions[best] += 1
            self._timed(started)
            return best

    def observe(self, index: int, sharing: int, tokens: int, decode_s: float, first_s: float | None) -> None:
        with self._lock:
            self.costs[index].learn(sharing, tokens, decode_s, first_s)

    def watch(self, index: int, sharing: int, events: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Pass a path's stream through, timing its first token and its end, and learn from the finished request."""
        started = time.monotonic()
        first = None
        for event in events:
            try:
                if first is None and event.get("type") == "token":
                    first = time.monotonic() - started
                if event.get("type") == "done" and first is not None:
                    end = time.monotonic() - started
                    self.observe(index, sharing, int(event.get("completion_tokens") or 0), end - first, first)
            except Exception as exc:  # noqa: BLE001 - learning never breaks the stream it watches
                self._failed("learn", exc)
            yield event

    def status(self) -> dict[str, Any]:
        with self._lock:
            out = {"paths": {name: cost.status() for name, cost in zip(self.names, self.costs)},
                   "decisions": dict(zip(self.names, self.decisions)),
                   "decision_us": round(self._decision_ns / 1000, 2),
                   "decision_us_mean": round(self._decision_ns_total / max(1, self._decisions_timed) / 1000, 2),
                   "decision_us_max": round(self._decision_ns_max / 1000, 2)}
        with self._errors_lock:
            out.update(errors=self.errors, last_error=self.last_error)
        return out


# ---------------------------------------------------------------- AG6: the serving policy, learned by dreaming

MODES = ("route", "stacked", "single")
DREAM_EVERY = 16      # finished requests between dreams
DREAM_MIN = 32        # logged requests before the first dream
DREAM_WINDOW = 64     # the last finished requests a dream replays
DREAM_GAIN = 1.05     # a mode must replay at least this many times the current mode's tokens per second
DREAM_TTFT = 1.1      # and a mean first token at most this many times the current mode's
DREAM_TIE = 0.98      # with no such winner: a mode with at least this share of the current mode's tokens per second
DREAM_SOONER = 0.8    # whose mean first token is at most this many times the current mode's, the earliest one,
DREAM_QUIET = 1 / 16  # and only on observed quiet traffic: at most this share of the window overlapped (below)


class _Path:
    """One path in the replay: its learned prices, its queue of arrivals and its running rows."""

    def __init__(self, cost: PathCost):
        self.cost, self.queue, self.active = cost, [], []

    def outstanding(self) -> int:
        return len(self.queue) + len(self.active)

    def admissible(self, clock: float) -> bool:
        return bool(self.queue) and self.queue[0][0] <= clock and len(self.active) < self.cost.capacity

    def runnable(self, clock: float) -> bool:
        return bool(self.active) or self.admissible(clock)


def replay(costs: Sequence[PathCost], arrivals: Sequence[tuple[float, int]], mode: str) -> dict[str, float]:
    """Dream: `arrivals` (seconds from the first, tokens made) served under `mode` in the learned world model of the
    two paths (index 0 the single path, 1 the stacked engine). Both paths run on one card, so they share its time:
    while both have work the card serves them a unit each in turn (a newcomer's first token, or one decode step of
    all of a path's rows at the learned price for that many rows). Tokens per second over the span and the mean
    first token, as the hardware runs score a cell."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    paths = [_Path(costs[0]), _Path(costs[1])]
    pending = sorted(arrivals)
    outs: list[dict[str, Any]] = []
    clock, turn, index = 0.0, 0, 0

    def arrive(upto: float) -> None:
        nonlocal index
        while index < len(pending) and pending[index][0] <= upto:
            arrival, tokens = pending[index]
            index += 1
            out: dict[str, Any] = {"arrival": arrival, "tokens": tokens}
            outs.append(out)
            if mode == "route":  # as `PathRouter.choose`: each path's learned tokens a request, its outstanding work
                here = [path.outstanding() for path in paths]
                chosen = min(range(2), key=lambda i: (costs[i].expected(here[i], costs[i].tokens), i))
            else:
                chosen = 1 if mode == "stacked" else 0
            paths[chosen].queue.append([arrival, tokens, out])

    def unit(path: _Path) -> float:
        """Run one unit of `path` from `clock`; return its seconds."""
        if path.admissible(clock):
            arrival, tokens, out = path.queue.pop(0)
            out["first"] = clock + path.cost.first_s
            if tokens <= 1:
                out["end"] = out["first"]
            else:
                path.active.append([tokens - 1, out])
            return path.cost.first_s
        seconds = path.cost.step(len(path.active))
        for row in path.active:
            row[0] -= 1
            if row[0] <= 0:
                row[1]["end"] = clock + seconds
        path.active = [row for row in path.active if row[0] > 0]
        return seconds

    while index < len(pending) or any(path.outstanding() for path in paths):
        arrive(clock)
        ready = [i for i in range(2) if paths[i].runnable(clock)]
        if not ready:
            clock = max(clock, min([pending[index][0]] if index < len(pending) else [clock]
                                   + [path.queue[0][0] for path in paths if path.queue]))
            if index >= len(pending) and not any(path.queue for path in paths):
                break
            continue
        if len(ready) == 1:
            chosen = ready[0]
        else:
            chosen, turn = turn, 1 - turn  # the card serves the two paths a unit each, in turn
        clock += unit(paths[chosen])
    span = max(out["end"] for out in outs) - min(out["arrival"] for out in outs)
    return {"tokens_per_s": sum(out["tokens"] for out in outs) / span if span > 0 else 0.0,
            "ttft_mean_s": statistics.fmean(out["first"] - out["arrival"] for out in outs)}


def observed_quiet(window: Sequence[tuple[float, float]], earlier: Sequence[tuple[float, float]] = (),
                   in_flight: int = 0, missing: int = 0, minimum: int = DREAM_MIN) -> dict[str, Any]:
    """Whether the traffic the router served was quiet, from what it observed alone, never from the model's prices.

    `window` holds the served intervals, (arrived, ended) on the router's clock, of the finished requests a dream
    replays. `earlier` holds those finished before them. A request of the window overlapped if it arrived while
    another observed request was being served: some other interval with arrived_i <= arrived_j < ended_i
    (half-open, so a request that arrives the instant another ends did not overlap; two that arrive together
    both did). It is quiet if at most floor(n * DREAM_QUIET) of the window's n requests overlapped, so at most 4
    of 64 or 2 of 32. The observation must be complete: at least `minimum` requests, none in flight, and none that
    ended without its answer inside the window (`missing`). Otherwise it is not quiet."""
    n = len(window)
    out: dict[str, Any] = {"quiet": False, "requests": n, "overlapped": None, "allowed": math.floor(n * DREAM_QUIET)}
    if n < minimum or in_flight or missing:
        out["reason"] = ("too few observed requests" if n < minimum else
                         "requests still in flight" if in_flight else "requests ended without their answer")
        return out
    served = list(earlier) + list(window)
    overlapped = sum(1 for j, (arrived, _) in enumerate(window, start=len(earlier))
                     if any(i != j and start <= arrived < end for i, (start, end) in enumerate(served)))
    out.update(overlapped=overlapped, quiet=overlapped <= out["allowed"],
               reason="observed quiet" if overlapped <= out["allowed"] else "observed overlap")
    return out


def dream(costs: Sequence[PathCost], arrivals: Sequence[tuple[float, int]], current: str,
          quiet: dict[str, Any] | None = None) -> dict[str, Any]:
    """Replay every mode and name the one to serve: the most tokens per second among the modes whose mean first token
    is at most `DREAM_TTFT` of the current mode's, if it beats the current mode by `DREAM_GAIN`.

    Else, on quiet traffic only, the earliest first tokens at the same throughput: among the modes with at least
    `DREAM_TIE` of the current mode's tokens per second, the one with the earliest mean first token, if it is at
    most `DREAM_SOONER` of the current mode's (ties: more tokens per second, then `MODES` order). On quiet traffic
    every mode's tokens per second is the arrival rate, so a mode with ten times later first tokens stayed (review
    of 2026-11-08). Quiet is `quiet`, the caller's observation of the served traffic (`observed_quiet`); without
    one it is not quiet. On quiet traffic a first token is the path's learned single-request price. Under load the
    replay's first tokens are unreliable: in AG6-H's bursts at load 0.8 it priced the router's 25 % earlier than
    stacking's, where the card measured stacking's 5x earlier. The 5 % to move for throughput and the 2 % that may
    be given up for latency keep the two rules from taking turns."""
    results = {mode: replay(costs, arrivals, mode) for mode in MODES}
    now = results[current]
    allowed = [mode for mode in MODES if results[mode]["ttft_mean_s"] <= DREAM_TTFT * now["ttft_mean_s"]]
    best = max(allowed, key=lambda mode: (results[mode]["tokens_per_s"], mode == current))
    if results[best]["tokens_per_s"] >= DREAM_GAIN * now["tokens_per_s"]:
        chosen, rule = best, "tokens per second"
    else:
        level = [mode for mode in MODES if results[mode]["tokens_per_s"] >= DREAM_TIE * now["tokens_per_s"]]
        soonest = min(level, key=lambda mode: (results[mode]["ttft_mean_s"], -results[mode]["tokens_per_s"],
                                               MODES.index(mode)))
        observed = bool(quiet and quiet.get("quiet"))
        sooner = observed and results[soonest]["ttft_mean_s"] <= DREAM_SOONER * now["ttft_mean_s"]
        chosen, rule = (soonest, "first token") if sooner else (current, "kept")
    return {"mode": chosen, "rule": rule, "replayed": results, "requests": len(arrivals), "quiet": quiet}


class PolicyRouter(PathRouter):
    """AG6: the router's per-request choice inside a serving mode the agent learns by dreaming. Every
    `DREAM_EVERY` finished requests it replays its last `DREAM_WINDOW` arrivals under each mode in its learned world
    model and moves to the mode that serves them best (`dream`). A decision stays a lookup, or in `route` the
    router's arithmetic; the dream runs after a request has finished, not inside a decision."""

    def __init__(self, names: Sequence[str], costs: Sequence[PathCost], mode: str = "route"):
        super().__init__(names, costs)
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        self.mode, self.switches, self.dreams, self.last_dream = mode, [], 0, None
        self._log: list[tuple[float, int, float]] = []  # (arrived, tokens, ended) of finished requests, observed
        self._finished = 0
        self._in_flight = 0
        self._missing: list[float] = []  # arrival times of requests that ended without their answer
        self._dream_ns_total = self._dream_ns_max = 0  # the background evaluation's cost, apart from decisions

    def choose(self, outstanding: Sequence[int], max_tokens: int) -> int:
        if self.mode == "route":
            return super().choose(outstanding, max_tokens)
        started = time.perf_counter_ns()
        with self._lock:
            best = 1 if self.mode == "stacked" else 0
            self.decisions[best] += 1
            self._timed(started)
            return best

    def watch(self, index: int, sharing: int, events: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        from .errors import EngineLimitExceeded

        arrived = time.monotonic()
        with self._lock:
            self._in_flight += 1
        answered = rerouted = False
        try:
            for event in super().watch(index, sharing, events):
                try:
                    if event.get("type") == "done" and not answered:
                        answered = True
                        with self._lock:
                            self._in_flight -= 1
                        self.finished(arrived, int(event.get("completion_tokens") or 0), time.monotonic())
                except Exception as exc:  # noqa: BLE001 - a failed dream keeps the mode; the answer goes out
                    self._failed("dream", exc)
                yield event
        except EngineLimitExceeded:
            rerouted = True  # the pool serves it on the other path, which this router watches again
            raise
        finally:
            if not answered:
                with self._lock:
                    self._in_flight -= 1
                    if not rerouted:  # failed or cancelled: its interval is missing from what was observed
                        self._missing = (self._missing + [arrived])[-2 * DREAM_WINDOW:]

    def finished(self, arrived: float, tokens: int, ended: float | None = None) -> None:
        """Log one finished request, with its observed interval, and dream when it is time."""
        with self._lock:
            self._log = (self._log + [(arrived, tokens, arrived if ended is None else ended)])[-2 * DREAM_WINDOW:]
            self._finished += 1
            window = self._log[-DREAM_WINDOW:]
            if self._finished % DREAM_EVERY or len(window) < DREAM_MIN:
                return
            first = min(arrival for arrival, _, _ in window)
            arrivals = [(arrival - first, max(1, tokens)) for arrival, tokens, _ in window]
            quiet = observed_quiet([(a, e) for a, _, e in window], [(a, e) for a, _, e in self._log[:-DREAM_WINDOW]],
                                   in_flight=self._in_flight, missing=sum(1 for t in self._missing if t >= first),
                                   minimum=DREAM_MIN)
            if ended is None:  # a caller without observed intervals: nothing shows the traffic was quiet
                quiet = {**quiet, "quiet": False, "reason": "no observed intervals"}
            costs, mode = copy.deepcopy(self.costs), self.mode  # the prices as they stand; learning goes on
        started = time.perf_counter_ns()
        result = dream(costs, arrivals, mode, quiet)
        took = time.perf_counter_ns() - started
        with self._lock:
            self.dreams += 1
            self._dream_ns_total += took
            self._dream_ns_max = max(self._dream_ns_max, took)
            self.last_dream = {**result, "at_request": self._finished, "dream_ms": round(took / 1e6, 2)}
            if result["mode"] != self.mode:
                self.switches.append({"at_request": self._finished, "from": self.mode, "to": result["mode"],
                                      "rule": result["rule"], "at_s": time.monotonic()})
                self.mode = result["mode"]

    def status(self) -> dict[str, Any]:
        out = super().status()
        with self._lock:
            out.update(mode=self.mode, dreams=self.dreams, switches=list(self.switches), last_dream=self.last_dream,
                       dream_ms_mean=round(self._dream_ns_total / max(1, self.dreams) / 1e6, 2),
                       dream_ms_max=round(self._dream_ns_max / 1e6, 2))
        return out
