"""RL2-H: does a whole-machine serving decision have state-dependent headroom (BACKLOG RL2)?

A learner can only beat the best fixed setting where different states want different settings.
RSI3-LIVE-P3 and OPT1 lost to the best fixed setting, so before a whole-machine learner is trained
this measures the upper bound any policy could reach, on the same recorded traces, against the A/A
noise of the reference (as OPT2-CAL did for the background-thread decision).

Fixed before data (BACKLOG RL2-H, 2026-10-28):

  arms      every setting a learner could pick: placement across GPUs (`shortest`, HW1's
            join-the-shortest-queue; `fill`, the lowest-index GPU with fewer than `width` requests
            outstanding, else the shortest) times the live learner's schedule catalogue (group width
            1, 2, 4, 8; gather wait 0, 50, 150 ms; width 1 never waits). The reference is
            `shortest/w8/g0`, the stored default policy at the tuned engine's group capacity of 8. With
            one GPU placement does not exist and only the schedule is measured.
  levels    offered load at 0.2, 0.6 and 0.95 of the reference's capacity (requests completed per
            second from a burst of 16 per GPU), equal weights.
  traces    per level, six traces of 16 Poisson arrivals with fixed seeds; prompt and output limit
            (64 or 128 tokens) are drawn from the same seed. Every arm replays the same traces.
  order     per (trace, level) block: the reference, every other arm in an order rotated by the
            block and reversed on odd blocks, the reference again (an A/A pair spanning the block).
  cost      per request, the live learner's reward (docs/LIVE_LEARNING.md): seconds in the system
            from the scheduled arrival + time to first content, or (1 + 1) x deadline + 1 s if it
            failed, timed out or was refused; a cell's cost is the mean over its trace.
  headroom  oracle = sum over levels of weight x the cheapest arm there; fixed = the cheapest single
            arm over all levels; H = 1 - oracle / fixed. In sample, and held out: arms chosen on the
            other traces, scored on the left-out one. Per decision, the oracle may vary only that
            decision; the other stays at the best fixed arm's value.
  noise     the 95th percentile (nearest rank) of the A/A pairs' relative differences.
  verdict   a decision has headroom when the 2.5 % bootstrap bound of its held-out H (2000 resamples
            of the held-out traces, seed 0) exceeds the noise; otherwise a fixed policy ships and no
            learner is trained for that decision. That fixed policy is the reference unless the best
            fixed arm is cheaper by more than the noise (wording added after run 1; the rule is unchanged).
  validity  for each request, every arm's answer digest must agree (grouped batch-1 is exact); a
            mismatch, a non-finite cost, a failed calibration or fewer than two complete traces per
            level makes the run invalid, and an invalid run releases no learner.

Run 2 (`--plan h2`, BACKLOG RL2-H2, fixed 2026-10-28 before its data) keeps arms' meaning, cost,
order, headroom, noise, verdict and validity, and changes the traffic so queues and long requests
occur, which run 1's Poisson traffic with uniform lengths never produced:

  states    `quiet`: run 1's light level (Poisson at 0.2 of capacity, short prompts, 64 or 128
            tokens); `bursts`: four bursts of four arrivals, each burst inside 0.2 s, burst starts
            Poisson at 0.6 of capacity / 4; `spike`: all 16 arrivals inside 1 s. In `bursts` and
            `spike` half the prompts carry a fixed generic context of about a thousand tokens and the
            output limit is 32 or 256 tokens. Equal weights.
  arms      the ten `shortest` schedules and `fill` without a gather wait at every width (14):
            `fill` with a wait lost by 27 % or more at every level of run 1.
  traces    five per state, seeds distinct from run 1's.

Run 3 (`--plan h3`, BACKLOG RL3, fixed 2026-10-28 before its data) keeps run 2's traffic, cost,
order, headroom, noise, verdict and validity and measures two decisions that use the output limit,
which is known when a request arrives (`long`: at least 128 tokens):

  placement `shortest` (HW1); `segregate`: long requests to the last GPU, short ones to the first;
            `spill`: `segregate` unless the preferred GPU has at least four more requests outstanding
            than the other, then the other.
  order     `fifo` (the service's own); `sjf`: the waiting request with the smallest output limit
            starts first, ties and requests waiting longer than 10 s by arrival.
  arms      the six combinations with the reference schedule (width 8, no wait).

The report keeps token counts, times and SHA-256 digests of answers, never prompt or answer text.
To run it on Kaggle (T4 x2) from a pushed commit:

    python tools/rl2_headroom.py --write-kernel kernel --owner <user> --commit <sha> [--plan h2|h3]
    kaggle kernels push -p kernel --accelerator NvidiaTeslaT4
    kaggle kernels output <user>/ironmule-rl2-headroom[-h2|-h3] -p results
    python tools/rl2_headroom.py --analyse results/rl2_headroom_report.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Iterable

MODEL = "mlx-community/Qwen3-1.7B-bf16"
SMOKE_MODEL = "Qwen/Qwen3-0.6B"
WIDTHS = (1, 2, 4, 8)
GATHERS_MS = (0, 50, 150)
REFERENCE = "shortest/w8/g0"
LOADS = (0.2, 0.6, 0.95)
WEIGHTS = (1 / 3, 1 / 3, 1 / 3)
TRACES = 6
REQUESTS = 16
MAX_TOKENS = (64, 128)
RESAMPLES = 2000
DECISIONS = ("joint", "placement", "schedule", "order")
LONG_TOKENS = 128   # run 3: an output limit at least this long is `long`
SPILL_GAP = 4       # run 3: `spill` leaves the preferred GPU when it has this many more requests outstanding
SJF_AGE_S = 10.0    # run 3: a request waiting longer than this starts by arrival under `sjf`
# Generic questions; the report keeps only their index, never the text or the answers.
PROMPTS = ("Explain how a refrigerator keeps food cold.", "What causes the seasons on Earth?",
           "Describe how a bicycle gear system works.", "Why is the sky blue during the day?",
           "Summarize the water cycle in a few sentences.", "How does a compiler differ from an interpreter?",
           "What is the difference between weather and climate?", "Explain what a hash table is.",
           "How do vaccines train the immune system?", "Why do metals conduct electricity?",
           "Describe the life cycle of a star like the Sun.", "What is photosynthesis?",
           "How does a binary search work?", "Explain the greenhouse effect.",
           "What does a CPU cache do?", "How do airplanes stay in the air?")
H2_STATES = ("quiet", "bursts", "spike")
H2_TRACES = 5
H2_MAX_TOKENS = (32, 256)
# Run 2's long prompts: generic filler of about a thousand tokens before the question, never user text.
CONTEXT = "Background notes, not needed for the answer. " + " ".join(PROMPTS * 8)


# ---------------------------------------------------------------- the plan, fixed before data

def arm_name(placement: str, width: int, gather_ms: int) -> str:
    return f"{placement}/w{width}/g{gather_ms}"


def parse_arm(name: str) -> tuple[str, int, int]:
    placement, width, gather = name.split("/")[:3]
    return placement, int(width[1:]), int(gather[1:])


def arm_order(name: str) -> str:
    """Run 3's fourth part: `fifo` (omitted) or `sjf`."""
    parts = name.split("/")
    return parts[3] if len(parts) > 3 else "fifo"


def _parts(name: str) -> dict[str, Any]:
    placement, width, gather = parse_arm(name)
    return {"placement": placement, "schedule": (width, gather), "order": arm_order(name)}


def arms(gpus: int) -> list[str]:
    """Every setting, the reference first; width 1 never waits, and one GPU has no placement."""
    schedules = [(w, g) for w in WIDTHS for g in GATHERS_MS if w > 1 or g == 0]
    placements = ("shortest", "fill") if gpus > 1 else ("shortest",)
    names = [arm_name(p, w, g) for p in placements for w, g in schedules]
    names.remove(REFERENCE)
    return [REFERENCE, *names]


def arms_h2(gpus: int) -> list[str]:
    """Run 2: every `shortest` schedule, and `fill` only without a gather wait."""
    return [name for name in arms(gpus) if parse_arm(name)[0] == "shortest" or parse_arm(name)[2] == 0]


def arms_h3(gpus: int) -> list[str]:
    """Run 3: placement by output limit and shortest-limit-first order, at the reference schedule."""
    placements = ("shortest", "segregate", "spill") if gpus > 1 else ("shortest",)
    return [REFERENCE + ("/sjf" if order == "sjf" else "") if placement == "shortest" else
            f"{placement}/w8/g0" + ("/sjf" if order == "sjf" else "")
            for placement in placements for order in ("fifo", "sjf")]


def trace(level: int, index: int, rate: float, requests: int = REQUESTS,
          max_tokens: tuple[int, ...] | None = None) -> list[dict[str, Any]]:
    """Poisson arrivals at `rate` per second with prompt and output limit from the same fixed seed."""
    rng = random.Random(1000 * level + index)
    offset, items = 0.0, []
    for _ in range(requests):
        items.append({"offset_s": round(offset, 6), "prompt": rng.randrange(len(PROMPTS)),
                      "max_tokens": rng.choice(max_tokens or MAX_TOKENS)})
        offset += rng.expovariate(rate)
    return items


def trace_h2(state: int, index: int, capacity: float, requests: int = REQUESTS,
             max_tokens: tuple[int, ...] | None = None) -> list[dict[str, Any]]:
    """Run 2's traffic for one state (see the module docstring), from a fixed seed."""
    rng = random.Random(2000 * state + index)
    if state == 0:
        return [{**item, "long": False} for item in trace(0, 100 + index, 0.2 * capacity, requests, max_tokens)]
    offsets: list[float] = []
    start = 0.0
    while len(offsets) < requests:
        if state == 1:
            offsets += [start + rng.uniform(0, 0.2) for _ in range(min(4, requests - len(offsets)))]
            start += rng.expovariate(0.6 * capacity / 4)
        else:
            offsets.append(rng.uniform(0, 1.0))
    first = min(offsets)
    return [{"offset_s": round(offset - first, 6), "prompt": rng.randrange(len(PROMPTS)), "long": rng.random() < 0.5,
             "max_tokens": rng.choice(max_tokens or H2_MAX_TOKENS)} for offset in sorted(offsets)]


def block_order(names: list[str], block: int) -> list[tuple[str, int]]:
    """(arm, run): the reference, the others rotated by the block (reversed on odd blocks), the reference again."""
    others = names[1:]
    shift = block % len(others) if others else 0
    others = others[shift:] + others[:shift]
    if block % 2:
        others.reverse()
    return [(names[0], 0), *((name, 0) for name in others), (names[0], 1)]


# ---------------------------------------------------------------- analysis (no hardware)

def request_cost(request: dict[str, Any], deadline_s: float) -> float:
    """The live learner's reward contract: time in system + time to first content, or the failure penalty."""
    if request.get("outcome") != "ok":
        return (1 + 1.0) * deadline_s + 1.0
    first = request["first_s"] if request.get("first_s") is not None else request["end_s"]  # no visible text
    return float(request["end_s"]) + float(first)


def _cell_costs(rows: list[dict[str, Any]], deadline_s: float) -> dict[tuple[int, int, str, int], float]:
    return {(row["level"], row["trace"], row["arm"], row["run"]):
            statistics.fmean(request_cost(r, deadline_s) for r in row["requests"]) for row in rows}


def _choose(table: dict[tuple[int, int, str], float], levels: list[int], traces: list[int],
            names: list[str], decision: str) -> tuple[dict[int, str], str]:
    """Per level, the cheapest arm the decision may vary, and the cheapest single arm, on `traces`."""
    def mean(level: int, name: str) -> float:
        return statistics.fmean(table[level, t, name] for t in traces)

    fixed = min(names, key=lambda n: (sum(WEIGHTS[level] * mean(level, n) for level in levels), n))
    held = {part: value for part, value in _parts(fixed).items() if part != decision}
    allowed = (list(names) if decision == "joint" else
               [n for n in names if all(_parts(n)[part] == value for part, value in held.items())])
    return {level: min(allowed, key=lambda n: (mean(level, n), n)) for level in levels}, fixed


def _score(table: dict[tuple[int, int, str], float], levels: list[int], traces: list[int],
           oracle: dict[int, str], fixed: str) -> tuple[float, float]:
    """Mean weighted cost of the oracle's arms and of the fixed arm over `traces`."""
    def cost(pick: Any) -> float:
        return statistics.fmean(sum(WEIGHTS[level] * table[level, t, pick(level)] for level in levels)
                                for t in traces)

    return cost(lambda level: oracle[level]), cost(lambda level: fixed)


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def analyse(report: dict[str, Any], resamples: int = RESAMPLES, seed: int = 0) -> dict[str, Any]:
    """Headroom per decision, in sample and held out, against the reference's A/A noise; and validity."""
    rows, deadline = report.get("rows", []), float(report.get("deadline_s") or 120)
    names = report.get("arms") or []
    levels = list(range(len(report.get("states") or report.get("rates") or LOADS)))
    problems: list[str] = []
    if report.get("calibration", {}).get("failed"):
        problems.append("calibration failed: " + str(report["calibration"]["failed"]))
    digests: dict[tuple[int, int, int], set[str]] = {}
    for row in rows:
        for request in row["requests"]:
            if request.get("outcome") == "ok" and request.get("digest"):
                digests.setdefault((row["level"], row["trace"], request["index"]), set()).add(request["digest"])
    mismatched = sorted(key for key, found in digests.items() if len(found) > 1)
    if mismatched:
        problems.append(f"{len(mismatched)} requests answered differently across arms (first: {list(mismatched[0])})")
    cells = _cell_costs(rows, deadline)
    if any(not math.isfinite(value) for value in cells.values()):
        problems.append("a cell cost is not finite")
    complete = [t for t in sorted({row["trace"] for row in rows})
                if all((level, t, name, 0) in cells and (level, t, REFERENCE, 1) in cells
                       for level in levels for name in names)]
    if len(complete) < 2 or not names:
        problems.append(f"{len(complete)} complete traces; at least two are needed in every level")
    out: dict[str, Any] = {"schema": "ironmule.rl2_headroom.analysis.v1", "arms": len(names),
                           "complete_traces": complete, "weights": list(WEIGHTS),
                           "mismatched_requests": len(mismatched)}
    if problems:
        return {**out, "valid": False, "problems": problems, "verdict": "invalid: no learner is released"}
    table = {(level, t, name): cells[level, t, name, 0] for level in levels for t in complete for name in names}
    pairs = [(cells[level, t, REFERENCE, 0], cells[level, t, REFERENCE, 1]) for level in levels for t in complete]
    relative = [abs(a - b) / ((a + b) / 2) for a, b in pairs]
    noise = _quantile(relative, 0.95)
    rng = random.Random(seed)
    decisions: dict[str, Any] = {}
    varying = [part for part in DECISIONS[1:] if len({repr(_parts(n)[part]) for n in names}) > 1]
    # One varying part is the joint decision itself; parts are reported apart only when two or more vary.
    for decision in ("joint", *varying) if len(varying) > 1 else ("joint",):
        oracle, fixed = _choose(table, levels, complete, names, decision)
        o, f = _score(table, levels, complete, oracle, fixed)
        held = []
        for t in complete:  # leave one trace out: choose on the others, score on it
            train = [u for u in complete if u != t]
            held.append(_score(table, levels, [t], *_choose(table, levels, train, names, decision)))
        h_out = 1 - sum(x for x, _ in held) / sum(y for _, y in held)
        boot = []
        for _ in range(resamples):
            sample = [held[rng.randrange(len(held))] for _ in held]
            boot.append(1 - sum(x for x, _ in sample) / sum(y for _, y in sample))
        low = _quantile(boot, 0.025)
        decisions[decision] = {
            "fixed": fixed, "oracle": {str(level): oracle[level] for level in levels},
            "headroom_in_sample": 1 - o / f, "headroom_held_out": h_out,
            "headroom_held_out_ci95": [low, _quantile(boot, 0.975)],
            "verdict": "headroom: train and qualify a learner" if low > noise else "no headroom: a fixed policy ships"}
    reference = sum(WEIGHTS[level] * statistics.fmean(table[level, t, REFERENCE] for t in complete) for level in levels)
    best = decisions["joint"]["fixed"]
    best_cost = sum(WEIGHTS[level] * statistics.fmean(table[level, t, best] for t in complete) for level in levels)
    per_level = {str(level): {name: statistics.fmean(table[level, t, name] for t in complete) for name in names}
                 for level in levels}
    learned = None
    if LEARNED in names:  # run 4 (BACKLOG RL4-H): the frozen policy against every fixed arm, paired by trace
        fixed_arms = [n for n in names if n != LEARNED]
        rival = min(fixed_arms, key=lambda n: (sum(WEIGHTS[level] * statistics.fmean(table[level, t, n] for t in complete)
                                                   for level in levels), n))
        per_trace = [(sum(WEIGHTS[level] * table[level, t, LEARNED] for level in levels),
                      sum(WEIGHTS[level] * table[level, t, rival] for level in levels)) for t in complete]
        boot = []
        for _ in range(resamples):
            sample = [per_trace[rng.randrange(len(per_trace))] for _ in per_trace]
            boot.append(1 - sum(x for x, _ in sample) / sum(y for _, y in sample))
        gain = 1 - sum(x for x, _ in per_trace) / sum(y for _, y in per_trace)
        low = _quantile(boot, 0.025)
        learned = {"rival": rival, "gain": gain, "gain_ci95": [low, _quantile(boot, 0.975)],
                   "verdict": ("the learned policy beats the best fixed arm" if low > noise
                               else "the learned policy does not beat the best fixed arm")}
    return {**out, "valid": True, "problems": [], "aa_noise": noise, "aa_pairs": len(pairs),
            "reference_cost": reference, "best_fixed": best, "best_fixed_vs_reference": best_cost / reference,
            "decisions": decisions, "mean_cost_per_level": per_level, "learned": learned,
            "verdict": ("headroom for: " + ", ".join(d for d, v in decisions.items() if v["verdict"].startswith("headroom"))
                        if any(v["verdict"].startswith("headroom") for v in decisions.values())
                        else f"no headroom: the fixed policy {best} ships, no learner" if best_cost / reference < 1 - noise
                        else f"no headroom: the reference stays ({best} is within the A/A noise of it), no learner")}


# ---------------------------------------------------------------- run 4's learned policy (BACKLOG RL4)

LONG_PROMPT_CHARS = 1000  # a request whose last message is longer counts as a long prompt
PLACEMENT_FEATURES = 8
ORDER_FEATURES = 4


def _is_long(request: Any) -> bool:
    return request.max_tokens >= LONG_TOKENS


def placement_features(request: Any, cards: list[Any], card: int) -> list[float]:
    """For one candidate card: its load and long work, crossed with whether this request is long.
    `request` has `max_tokens` and `long_prompt`; each card's `outstanding()` lists such requests."""
    outstanding = cards[card].outstanding()
    load = len(outstanding) / 8.0
    longs = sum(_is_long(r) for r in outstanding) / 8.0
    shorts = load - longs
    long_request = 1.0 if _is_long(request) else 0.0
    last = 1.0 if card == len(cards) - 1 else 0.0
    return [load, long_request * longs, (1 - long_request) * longs, long_request * shorts,
            (1 - long_request) * shorts, long_request * last, (1 - long_request) * last,
            float(request.long_prompt) * load]


def order_features(waiting: list[Any]) -> list[float]:
    longs = sum(_is_long(r) for r in waiting)
    shorts = len(waiting) - longs
    return [1.0, len(waiting) / 8.0, float(longs > 0 and shorts > 0), longs / max(1, len(waiting))]


@dataclass
class LearnedPolicy:
    """A softmax over cards per request and a logistic `sjf` choice per group start (trained by
    `tools/rl4_agent.py`); the harness applies it greedily."""
    placement: list[float]
    order: list[float]

    @classmethod
    def zero(cls) -> "LearnedPolicy":
        return cls([0.0] * PLACEMENT_FEATURES, [0.0] * ORDER_FEATURES)

    def to_json(self) -> dict[str, Any]:
        body = {"schema": "ironmule.rl4.policy.v1", "placement": self.placement, "order": self.order,
                "features": {"placement": PLACEMENT_FEATURES, "order": ORDER_FEATURES}}
        return {**body, "sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "LearnedPolicy":
        body = {k: v for k, v in data.items() if k != "sha256"}
        if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != data.get("sha256"):
            raise ValueError("policy digest does not match its weights")
        if body.get("schema") != "ironmule.rl4.policy.v1":
            raise ValueError("unknown policy schema")
        return cls(list(body["placement"]), list(body["order"]))

    def card_probabilities(self, request: Any, cards: list[Any]) -> list[float]:
        scores = [sum(w * x for w, x in zip(self.placement, placement_features(request, cards, c)))
                  for c in range(len(cards))]
        top = max(scores)
        exps = [math.exp(s - top) for s in scores]
        return [e / sum(exps) for e in exps]

    def sjf_probability(self, waiting: list[Any]) -> float:
        z = sum(w * x for w, x in zip(self.order, order_features(waiting)))
        return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))

    def best_card(self, request: Any, cards: list[Any]) -> int:
        p = self.card_probabilities(request, cards)
        return max(range(len(p)), key=lambda i: (p[i], -i))

    def uses_sjf(self, waiting: list[Any]) -> bool:
        return self.sjf_probability(waiting) > 0.5

    def greedy(self) -> tuple[Any, Any]:
        """(place, order) callables in `tools/rl4_agent.py`'s simulator signature."""
        return (lambda request, cards, now: self.best_card(request, cards),
                lambda waiting, now: "sjf" if self.uses_sjf(waiting) else "fifo")


@dataclass
class _View:
    """What the learned policy reads of a live request."""
    max_tokens: int
    long_prompt: bool

    @classmethod
    def of(cls, request: Any) -> "_View":
        return cls(request.max_tokens, len(request.messages[-1][1]) > LONG_PROMPT_CHARS)


class _CardView:
    def __init__(self, requests: list[_View]):
        self._requests = requests

    def outstanding(self) -> list[_View]:
        return self._requests


def _card_views(shards: list[Any]) -> list[_CardView]:
    views = []
    for shard in shards:
        with shard._lock:
            views.append(_CardView([_View.of(session.request) for session in shard._sessions.values()]))
    return views


# ---------------------------------------------------------------- measurement (hardware)

def _placement_pool(pool: Any) -> Any:
    """The started pool, whose placement can also `fill`: the lowest-index GPU with fewer than
    `fill_width` requests outstanding, else HW1's shortest queue. Only this tool uses `fill`."""
    from ironmule_product.device_pool import DevicePool

    class PlacementPool(DevicePool):
        placement, fill_width, learned_policy = "shortest", 8, None
        _local = threading.local()  # the request being placed, per calling thread

        def stream(self, request: Any) -> Any:
            self._local.request = request
            return super().stream(request)

        def _order(self) -> list[int]:
            order = super()._order()
            if self.placement == "fill":
                room = sorted(index for index in order if self.shards[index].outstanding() < self.fill_width)
                return room + [index for index in order if index not in room]
            if self.placement == "learned" and len(order) > 1 and self.learned_policy is not None:
                request = getattr(self._local, "request", None)
                if request is not None:
                    choice = self.learned_policy.best_card(_View.of(request), _card_views(self.shards))
                    if choice in order:
                        return [choice] + [index for index in order if index != choice]
                return order
            if self.placement not in ("segregate", "spill") or len(order) < 2:
                return order
            request = getattr(self._local, "request", None)
            long = request is not None and request.max_tokens >= LONG_TOKENS
            preferred = len(self.shards) - 1 if long else 0
            if preferred not in order:
                return order
            if self.placement == "spill":
                other = min((index for index in order if index != preferred),
                            key=lambda index: (self.shards[index].outstanding(), index))
                if self.shards[preferred].outstanding() - self.shards[other].outstanding() >= SPILL_GAP:
                    preferred = other
            return [preferred] + [index for index in order if index != preferred]

    return PlacementPool(pool.shards, pool.devices)


def ordered_service() -> type:
    """`ProductService` whose waiting requests can start shortest output limit first (run 3's `sjf`);
    with `order = "fifo"` it is the service unchanged. Only this tool uses `sjf`."""
    import queue

    from ironmule_product.service import ProductService

    class OrderedService(ProductService):
        order, learned_policy = "fifo", None  # `policy` is the service's own ServingPolicy

        def _sort_waiting(self) -> None:  # the caller holds self._lock
            while True:
                try:
                    self._carry.append(self._pending.get_nowait())
                except queue.Empty:
                    break
            now = time.monotonic_ns()
            sjf = self.order == "sjf" or (self.order == "learned" and self.learned_policy is not None and
                                          self.learned_policy.uses_sjf([_View.of(s.request) for s in self._carry]))

            def key(session: Any) -> tuple[bool, int, int]:
                if not sjf:  # run 4's learned order chose arrival order for this group
                    return (False, 0, session.submitted_ns)
                aged = now - session.submitted_ns > SJF_AGE_S * 1e9
                return (not aged, 0 if aged else session.request.max_tokens, session.submitted_ns)

            waiting = sorted(self._carry, key=key)
            self._carry.clear()
            self._carry.extend(waiting)

        def _next_session(self) -> Any:
            if self.order not in ("sjf", "learned"):
                return super()._next_session()
            with self._lock:
                self._sort_waiting()
                if self._carry:
                    return self._carry.popleft()
            try:
                first = self._pending.get(timeout=0.1)
            except queue.Empty:
                return None
            with self._lock:
                self._carry.append(first)
                self._sort_waiting()
                return self._carry.popleft()

        def _group_for(self, first: Any, width: int | None = None, gather_ms: int | None = None) -> list[Any]:
            if self.order in ("sjf", "learned"):
                with self._lock:
                    self._sort_waiting()
            return super()._group_for(first, width, gather_ms)

    return OrderedService


def set_arm(pool: Any, name: str) -> None:
    from ironmule_product.serving_policy import ServingPolicy

    placement, width, gather = parse_arm(name)
    pool.placement, pool.fill_width = placement, width
    for shard in pool.shards:
        shard.policy = ServingPolicy(max_width=width, gather_ms=gather, source="rl2-h")
        shard.order = arm_order(name)


def wait_idle(pool: Any, timeout_s: float = 120) -> bool:
    until = time.monotonic() + timeout_s
    while time.monotonic() < until:
        health = pool.health()
        if health.get("queued_requests", 0) == 0 and health.get("active_requests", 0) == 0:
            return True
        time.sleep(0.05)
    return False


def replay(pool: Any, model: str, items: list[dict[str, Any]], deadline_s: float) -> list[dict[str, Any]]:
    """Send each request at its scheduled offset through the pool; times are from the scheduled arrival."""
    from ironmule_product.errors import Overloaded, ProductError, RequestTimeout
    from ironmule_product.types import GenerationRequest

    results: list[dict[str, Any]] = [{} for _ in items]
    start = time.monotonic() + 0.05

    def client(index: int, item: dict[str, Any]) -> None:
        scheduled = start + item["offset_s"]
        out: dict[str, Any] = {"index": index, "outcome": "failed", "first_s": None, "end_s": None,
                               "digest": None, "tokens": 0}
        ids: list[Any] = []
        prompt = (CONTEXT + "\n\n" if item.get("long") else "") + PROMPTS[item["prompt"]]
        request = GenerationRequest(model, (("user", prompt),), max_tokens=item["max_tokens"], stream=True)
        try:
            for event in pool.stream(request):
                if event.get("type") == "token":
                    if out["first_s"] is None and event.get("text"):
                        out["first_s"] = time.monotonic() - scheduled
                    ids.append(event.get("token_id") if event.get("token_id") is not None else event.get("text"))
                elif event.get("type") == "done":
                    out["outcome"], out["tokens"] = "ok", int(event.get("completion_tokens") or 0)
        except Overloaded:
            out["outcome"] = "rejected"
        except RequestTimeout:
            out["outcome"] = "timeout"
        except ProductError as exc:
            out["error"] = exc.code
        out["end_s"] = time.monotonic() - scheduled
        if out["outcome"] == "ok":
            out["digest"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        results[index] = out

    threads = []
    for index, item in enumerate(items):
        delay = start + item["offset_s"] - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        thread = threading.Thread(target=client, args=(index, item), daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join(deadline_s + 60)
    return results


def calibrate(pool: Any, model: str, deadline_s: float, max_tokens: tuple[int, ...] | None = None) -> dict[str, Any]:
    """Requests completed per second under the reference from a burst of 16 per GPU at once."""
    set_arm(pool, REFERENCE)
    burst = [{**item, "offset_s": 0.0} for item in trace(9, 0, 1.0, 16 * len(pool.shards), max_tokens)]
    done = replay(pool, model, burst, deadline_s)
    ok = [r for r in done if r["outcome"] == "ok"]
    outcomes: dict[str, int] = {}
    for r in done:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    out: dict[str, Any] = {"requests": len(done), "ok": len(ok), "outcomes": outcomes}
    if len(ok) < len(done) / 2 or not ok:
        return {**out, "failed": "fewer than half of the calibration burst completed"}
    return {**out, "makespan_s": max(r["end_s"] for r in ok), "capacity_rps": len(ok) / max(r["end_s"] for r in ok)}


def measure(pool: Any, model: str, *, deadline_s: float, traces: int | None = None, requests: int = REQUESTS,
            loads: tuple[float, ...] = LOADS, names: list[str] | None = None,
            max_tokens: tuple[int, ...] | None = None, plan: str = "h1", policy: Any = None,
            on_row: Any = None) -> dict[str, Any]:
    """The plan above on a started pool: calibration, then every (trace, level) block in order.
    `names` and `max_tokens` shrink it for a smoke run; a measurement keeps the defaults."""
    pool = _placement_pool(pool)
    pool.learned_policy = policy
    for shard in pool.shards:
        shard.learned_policy = policy
    names = names or {"h2": arms_h2, "h3": arms_h3}.get(plan, arms)(len(pool.shards))
    if any(rl_part == "learned" for name in names for rl_part in (parse_arm(name)[0], arm_order(name))) \
            and policy is None:
        raise ValueError("a learned arm needs a policy")
    warm = replay(pool, model, [{"offset_s": 0.0, "prompt": 0, "max_tokens": 8}] * len(pool.shards), deadline_s)
    report: dict[str, Any] = {"plan": plan, "arms": names, "devices": list(pool.devices), "deadline_s": deadline_s,
                              "policy": None if policy is None else policy.to_json(),
                              "warmup_ok": sum(r["outcome"] == "ok" for r in warm), "rows": []}
    report["calibration"] = calibrate(pool, model, deadline_s, max_tokens)
    if report["calibration"].get("failed"):
        return report
    capacity = report["calibration"]["capacity_rps"]
    if plan in ("h2", "h3", "h4"):
        report["states"] = list(H2_STATES)
        traffic = [lambda t, s=s: trace_h2(s, t, capacity, requests, max_tokens) for s in range(len(H2_STATES))]
    else:
        report["rates"] = [load * capacity for load in loads]
        traffic = [lambda t, level=level, rate=rate: trace(level, t, rate, requests, max_tokens)
                   for level, rate in enumerate(report["rates"])]
    block = 0
    for t in range(traces if traces is not None else H2_TRACES if plan in ("h2", "h3", "h4") else TRACES):
        for level, make in enumerate(traffic):
            items = make(t)
            for name, run in block_order(names, block):
                if not wait_idle(pool):
                    report["stopped"] = f"the pool did not drain before block {block}"
                    return report
                set_arm(pool, name)
                started = time.time()
                row = {"level": level, "trace": t, "arm": name, "run": run, "block": block,
                       "started_unix": started, "requests": replay(pool, model, items, deadline_s)}
                report["rows"].append(row)
                if on_row is not None:
                    on_row(report)
            block += 1
    return report


def _ironmule(*arguments: str, state: Path, timeout: float = 600) -> dict[str, Any]:
    code = "import sys, ironmule_cli; sys.exit(ironmule_cli.main(sys.argv[1:]))"
    try:
        done = subprocess.run([sys.executable, "-c", code, *arguments, "--state-dir", str(state)],
                              capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"returncode": None, "error": f"{type(exc).__name__}: {exc}"[:400]}
    return {"returncode": done.returncode, "stdout": done.stdout[-2000:], "stderr": done.stderr[-2000:]}


def environment() -> dict[str, Any]:
    from importlib import metadata

    versions = {}
    for name in ("ironmule", "mlx", "mlx-lm", "mlx-cuda"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    try:
        gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        gpus = ""
    return {"python": platform.python_version(), "machine": platform.machine(), "cpus": os.cpu_count(),
            "versions": versions, "gpus": gpus.strip().splitlines()}


def arms_h4(best_fixed: str) -> list[str]:
    """Run 4 (BACKLOG RL4-H): the reference, run 3's best fixed arm and the learned policy."""
    return list(dict.fromkeys([REFERENCE, best_fixed, LEARNED]))


LEARNED = "learned/w8/g0/learned"


def run(model: str, work: Path, *, smoke: bool, plan: str = "h1", on_row: Any = None,
        policy: Any = None, names: list[str] | None = None) -> dict[str, Any]:
    """Register `model`, start one worker per GPU (the tuned engine's grouped path) and measure."""
    from huggingface_hub import snapshot_download

    from ironmule_product.backend import MLXWorkerClient
    from ironmule_product.device_pool import cuda_devices, start_pool
    from ironmule_product.state import ProductStore

    revision = Path(snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                                             "*.jinja", "tokenizer*"])).name
    state = work / "rl2h-state"
    setup = _ironmule("setup", "--mode", "server", state=state)
    added = _ironmule("models", "add", model, state=state)
    if added.get("returncode") != 0:
        return {"model": model, "revision": revision, "failed": "models add", "setup": setup, "added": added}
    store = ProductStore(state)
    spec = store.model(model)
    devices = cuda_devices() or [None]
    pool = start_pool(store, spec, devices, lambda device: MLXWorkerClient(
        spec, execution_variant="current_engine", cuda_device=device), ordered_service())
    try:
        deadline = float(store.settings()["request_timeout_s"] or 120)
        smoke_names = (arms_h3(len(devices))[:2] if plan == "h3" else
                       [n for n in arms(len(devices)) if parse_arm(n)[1:] in ((8, 0), (1, 0))])
        sizes = {"traces": 2, "requests": 3, "max_tokens": (4, 8), "names": smoke_names} if smoke else {}
        if names is not None:
            sizes["names"] = names
        result = measure(pool, model, deadline_s=deadline, plan=plan, policy=policy, on_row=on_row, **sizes)
    finally:
        pool.close()
    return {"model": model, "revision": revision, **result}


REPOSITORY = "https://github.com/Tobayko/IronMule"
_BOOTSTRAP = '''"""IronMule RL2-H on this Kaggle session; written by tools/rl2_headroom.py."""
from __future__ import annotations
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "ironmule[cuda] @ git+{repository}@{commit}"], check=True)
'''


def write_kernel(directory: Path, owner: str, commit: str, plan: str = "h1", extra: list[str] | None = None) -> Path:
    """A private script kernel that installs IronMule at `commit` and runs this file's `plan` (with
    `extra` arguments, such as run 4's frozen policy); the report lands in the kernel's output as
    `rl2_headroom_report.json`."""
    if not commit or any(c not in "0123456789abcdef" for c in commit) or len(commit) != 40:
        raise ValueError("commit must be a full 40-character lowercase SHA, so the kernel runs exactly it")
    directory.mkdir(parents=True, exist_ok=True)
    body = (Path(__file__).read_text().replace('\nif __name__ == "__main__":\n    sys.exit(main())\n', "")
            .replace("from __future__ import annotations\n", "", 1))
    arguments = ["--out", "/kaggle/working/rl2_headroom_report.json", "--plan", plan, *(extra or [])]
    entry = f"\nsys.exit(main({arguments!r}))\n"
    name = "ironmule-rl2-headroom" + ("" if plan == "h1" else f"-{plan}")
    (directory / "run.py").write_text(_BOOTSTRAP.format(repository=REPOSITORY, commit=commit) + body + entry)
    metadata = {"id": f"{owner}/{name}", "title": name, "code_file": "run.py",
                "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
                "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=1) + "\n")
    return directory


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="write the measurement report here")
    parser.add_argument("--model", default=None)
    parser.add_argument("--plan", choices=("h1", "h2", "h3", "h4"), default="h1", help="run 1's to 4's fixed plan")
    parser.add_argument("--policy-json", help="run 4: the frozen learned policy (JSON text or @path)")
    parser.add_argument("--best-fixed", help="run 4: run 3's best fixed arm")
    parser.add_argument("--smoke", action="store_true", help="tiny sizes: checks this tool, measures nothing")
    parser.add_argument("--analyse", type=Path, metavar="REPORT", help="print the analysis of a finished report")
    parser.add_argument("--write-kernel", type=Path, metavar="DIR",
                        help="write a Kaggle kernel folder instead of running (needs --owner and --commit)")
    parser.add_argument("--owner")
    parser.add_argument("--commit")
    args = parser.parse_args(None if argv is None else list(argv))
    if args.write_kernel is not None:
        if not args.owner or not args.commit:
            parser.error("--write-kernel needs --owner and --commit")
        extra = []
        if args.plan == "h4":
            if not args.policy_json or not args.best_fixed:
                parser.error("--plan h4 needs --policy-json and --best-fixed")
            text = Path(args.policy_json[1:]).read_text() if args.policy_json.startswith("@") else args.policy_json
            LearnedPolicy.from_json(json.loads(text))  # refuse a policy whose digest does not match
            extra = ["--best-fixed", args.best_fixed, "--policy-json", json.dumps(json.loads(text), sort_keys=True)]
        print(json.dumps({"kernel": str(write_kernel(args.write_kernel, args.owner, args.commit, args.plan, extra))}))
        return 0
    if args.analyse is not None:
        report = json.loads(args.analyse.read_text())
        print(json.dumps(analyse(report.get("rl2h", report)), indent=1, sort_keys=True))
        return 0
    if args.out is None:
        parser.error("--out is required to run")
    report: dict[str, Any] = {"schema": "ironmule.rl2_headroom.v1", "plan": args.plan, "smoke": args.smoke,
                              "started_unix": time.time(), "env": environment()}

    def save(partial: dict[str, Any] | None = None) -> None:
        if partial is not None:
            report["rl2h"] = partial
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")

    save()
    work = Path(tempfile.mkdtemp(prefix="ironmule-rl2h-"))
    model = args.model or (SMOKE_MODEL if args.smoke else MODEL)
    policy = names = None
    if args.plan == "h4":
        if not args.policy_json or not args.best_fixed:
            parser.error("--plan h4 needs --policy-json and --best-fixed")
        text = Path(args.policy_json[1:]).read_text() if args.policy_json.startswith("@") else args.policy_json
        policy = LearnedPolicy.from_json(json.loads(text))
        names = arms_h4(args.best_fixed)
    report["rl2h"] = run(model, work, smoke=args.smoke, plan=args.plan, on_row=save, policy=policy, names=names)
    report["analysis"] = analyse(report["rl2h"])
    report["finished_unix"] = time.time()
    save()
    print(json.dumps({"report": str(args.out), "verdict": report["analysis"]["verdict"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
