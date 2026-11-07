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

import math
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
            tokens = [min(float(max_tokens), cost.tokens) for cost in self.costs]
            expected = [cost.expected(n, t) for cost, n, t in zip(self.costs, outstanding, tokens)]
            best = min(range(len(expected)), key=lambda i: (expected[i], i))
            self.decisions[best] += 1
            self._decision_ns = time.perf_counter_ns() - started
            return best

    def observe(self, index: int, sharing: int, tokens: int, decode_s: float, first_s: float | None) -> None:
        with self._lock:
            self.costs[index].learn(sharing, tokens, decode_s, first_s)

    def watch(self, index: int, sharing: int, events: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Pass a path's stream through, timing its first token and its end, and learn from the finished request."""
        started = time.monotonic()
        first = None
        for event in events:
            if first is None and event.get("type") == "token":
                first = time.monotonic() - started
            if event.get("type") == "done":
                end = time.monotonic() - started
                tokens = int(event.get("completion_tokens") or 0)
                if first is not None:
                    self.observe(index, sharing, tokens, end - first, first)
            yield event

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {"paths": {name: cost.status() for name, cost in zip(self.names, self.costs)},
                    "decisions": dict(zip(self.names, self.decisions)),
                    "decision_us": round(self._decision_ns / 1000, 2)}
