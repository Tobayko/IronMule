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

X1 (`--plan x1`, BACKLOG X1, fixed 2026-10-28 before its data) is not a headroom run: run 3 was invalid because
one short request was answered differently across arms, and X1 checks on each card whether the KV capacity, the
group or the card moves a request's tokens (`exactness_on_device`, decided by `x1_summary`).

RL4-S (`--plan h5`, BACKLOG RL4-S, fixed 2026-10-29 before its data) keeps run 4's arms (the reference, run 3's
best fixed arm and the frozen learned policy), cost, order, noise, verdict and validity, and replays only `spike`
traffic, on 30 traces (indices 5-34) whose seeds no earlier run and no training used.

RL6 (`--plan h6`, BACKLOG RL6, fixed 2026-10-29 before its data) compares the reference with shortest-job-first at
widths 1, 2 and 4 and a frozen policy that also picks each group's width, on `bursts` and `spike` (12 traces each,
indices 35-46); the analysis adds each fixed arm's paired gain over the reference.

The report keeps token counts, times and SHA-256 digests of answers, never prompt or answer text.
To run it on Kaggle (T4 x2) from a pushed commit:

    python tools/rl2_headroom.py --write-kernel kernel --owner <user> --commit <sha> [--plan h2|h3|x1]
    kaggle kernels push -p kernel --accelerator NvidiaTeslaT4
    kaggle kernels output <user>/ironmule-rl2-headroom[-h2|-h3|-x1] -p results
    python tools/rl2_headroom.py --analyse results/rl2_headroom_report.json
"""

from __future__ import annotations

import argparse
import base64
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
H5_STATE = 2                       # RL4-S: `spike` only, where run 3's A/A pairs stayed within 1.4 %
H5_TRACES = tuple(range(5, 35))    # 30 traces whose seeds no earlier run and no training used
H6_STATES = (1, 2)                 # RL6: `bursts` and `spike`, where width and order matter in the model
H6_TRACES = tuple(range(35, 47))   # 12 traces per state, seeds unused by every earlier run and training
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
        # RL6: each fixed arm against the reference, paired by trace (after run 4's draws, which stay as they were)
        learned["fixed_vs_reference"] = {}
        for arm in fixed_arms:
            if arm == REFERENCE:
                continue
            paired = [(sum(WEIGHTS[level] * table[level, t, arm] for level in levels),
                       sum(WEIGHTS[level] * table[level, t, REFERENCE] for level in levels)) for t in complete]
            boot = []
            for _ in range(resamples):
                sample = [paired[rng.randrange(len(paired))] for _ in paired]
                boot.append(1 - sum(x for x, _ in sample) / sum(y for _, y in sample))
            learned["fixed_vs_reference"][arm] = {"gain": 1 - sum(x for x, _ in paired) / sum(y for _, y in paired),
                                                  "gain_ci95": [_quantile(boot, 0.025), _quantile(boot, 0.975)]}
    return {**out, "valid": True, "problems": [], "aa_noise": noise, "aa_pairs": len(pairs),
            "reference_cost": reference, "best_fixed": best, "best_fixed_vs_reference": best_cost / reference,
            "decisions": decisions, "mean_cost_per_level": per_level, "learned": learned,
            "verdict": ("headroom for: " + ", ".join(d for d, v in decisions.items() if v["verdict"].startswith("headroom"))
                        if any(v["verdict"].startswith("headroom") for v in decisions.values())
                        else f"no headroom: the fixed policy {best} ships, no learner" if best_cost / reference < 1 - noise
                        else f"no headroom: the reference stays ({best} is within the A/A noise of it), no learner")}


# ---------------------------------------------------------------- run 4's learned policy (BACKLOG RL4)

LONG_PROMPT_CHARS = 1000  # a request whose last message is longer counts as a long prompt
PLACEMENT_FEATURES = 10
ORDER_FEATURES = 4
WIDTH_CHOICES = (1, 2, 4, 8)  # RL6: the group widths a policy may start, from the waiting queue's features


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
    idle = 1.0 if not outstanding else 0.0
    work = sum(r.max_tokens for r in outstanding) / 1024.0  # output tokens still owed on this card
    return [load, long_request * longs, (1 - long_request) * longs, long_request * shorts,
            (1 - long_request) * shorts, long_request * last, (1 - long_request) * last,
            float(request.long_prompt) * load, idle, work]


def order_features(waiting: list[Any]) -> list[float]:
    longs = sum(_is_long(r) for r in waiting)
    shorts = len(waiting) - longs
    return [1.0, len(waiting) / 8.0, float(longs > 0 and shorts > 0), longs / max(1, len(waiting))]


@dataclass
class LearnedPolicy:
    """A softmax over cards per request and a logistic `sjf` choice per group start (trained by
    `tools/rl4_agent.py`); the harness applies it greedily. With `width` (RL6) it also picks each
    group's width from `WIDTH_CHOICES`, a softmax over the waiting queue's `order_features`;
    without it the group width is the arm's."""
    placement: list[float]
    order: list[float]
    width: list[float] | None = None  # len(WIDTH_CHOICES) rows of ORDER_FEATURES weights

    @classmethod
    def zero(cls, widths: bool = False) -> "LearnedPolicy":
        return cls([0.0] * PLACEMENT_FEATURES, [0.0] * ORDER_FEATURES,
                   [0.0] * (len(WIDTH_CHOICES) * ORDER_FEATURES) if widths else None)

    def to_json(self) -> dict[str, Any]:
        body: dict[str, Any] = {"schema": "ironmule.rl4.policy.v1", "placement": self.placement, "order": self.order,
                                "features": {"placement": PLACEMENT_FEATURES, "order": ORDER_FEATURES}}
        if self.width is not None:  # a v1 policy keeps its exact body and digest
            body.update(schema="ironmule.rl4.policy.v2", width=self.width, widths=list(WIDTH_CHOICES))
        return {**body, "sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "LearnedPolicy":
        body = {k: v for k, v in data.items() if k != "sha256"}
        if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != data.get("sha256"):
            raise ValueError("policy digest does not match its weights")
        if body.get("schema") == "ironmule.rl4.policy.v1":
            return cls(list(body["placement"]), list(body["order"]))
        if body.get("schema") == "ironmule.rl4.policy.v2" and body.get("widths") == list(WIDTH_CHOICES):
            return cls(list(body["placement"]), list(body["order"]), list(body["width"]))
        raise ValueError("unknown policy schema")

    def width_probabilities(self, waiting: list[Any]) -> list[float]:
        f = order_features(waiting)
        scores = [sum(w * x for w, x in zip(self.width[c * ORDER_FEATURES:(c + 1) * ORDER_FEATURES], f))
                  for c in range(len(WIDTH_CHOICES))]
        top = max(scores)
        exps = [math.exp(v - top) for v in scores]
        return [e / sum(exps) for e in exps]

    def best_width(self, waiting: list[Any]) -> int:
        p = self.width_probabilities(waiting)
        return WIDTH_CHOICES[max(range(len(p)), key=lambda i: (p[i], -i))]

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

    def greedy_width(self) -> Any:
        """The simulator's per-group width callable, or None when the arm's width applies."""
        return None if self.width is None else (lambda waiting, now: self.best_width(waiting))


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
                    policy = self.learned_policy
                    if self.order == "learned" and policy is not None and policy.width is not None:
                        # RL6: the policy picks this group's width from the queue as the group starts
                        width = policy.best_width([_View.of(s.request) for s in [first, *self._carry]])
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
    if plan == "h5":
        report["states"] = [H2_STATES[H5_STATE]]
        traffic = [lambda t: trace_h2(H5_STATE, t, capacity, requests, max_tokens)]
    elif plan == "h6":
        report["states"] = [H2_STATES[s] for s in H6_STATES]
        traffic = [lambda t, s=s: trace_h2(s, t, capacity, requests, max_tokens) for s in H6_STATES]
    elif plan in ("h2", "h3", "h4"):
        report["states"] = list(H2_STATES)
        traffic = [lambda t, s=s: trace_h2(s, t, capacity, requests, max_tokens) for s in range(len(H2_STATES))]
    else:
        report["rates"] = [load * capacity for load in loads]
        traffic = [lambda t, level=level, rate=rate: trace(level, t, rate, requests, max_tokens)
                   for level, rate in enumerate(report["rates"])]
    block = 0
    indices = (range(traces) if traces is not None else H5_TRACES if plan == "h5" else H6_TRACES if plan == "h6"
               else range(H2_TRACES if plan in ("h2", "h3", "h4") else TRACES))
    for t in indices:
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


def arms_h6() -> list[str]:
    """RL6: the reference, shortest-job-first at widths 1, 2 and 4, and the width-learning policy."""
    return [REFERENCE, "shortest/w1/g0/sjf", "shortest/w2/g0/sjf", "shortest/w4/g0/sjf", LEARNED]


LEARNED = "learned/w8/g0/learned"

# X1 (BACKLOG X1, after run 3 was invalid): request 14 of run 3's `bursts` trace 1, a short prompt,
# was answered one way under every `shortest` arm and another under every `segregate`/`spill` arm.
X1_TRACE = (1, 1)
X1_GROUP = (12, 13, 14, 15)  # the burst it arrived in: three long prompts and itself
X1_PROBE = 14
X1_REPEATS = 3
X1_CAPACITIES = (1024, 1088)  # besides each prompt's own: a power of two, and run 3's long-prompt group at 32 tokens
X1_CHILD = "IRONMULE_X1_CHILD"
X1_REVISION = "9cd6692855d3e06772228e9a962b2606359b2d24"  # the revision of MODEL that run 3 served


def _x1_ids(tokenizer: Any, item: dict[str, Any]) -> list[int]:
    """The prompt as `replay` sends it and the worker renders it (chat template, thinking on)."""
    prompt = (CONTEXT + "\n\n" if item.get("long") else "") + PROMPTS[item["prompt"]]
    return list(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                              tokenize=True))


def _token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(list(tokens)).encode()).hexdigest()  # as `replay` digests token ids


def exactness_on_device(model: str, smoke: bool = False) -> dict[str, Any]:
    """X1 on this process's one device, with the engine a worker without a tuned profile runs (BASELINE
    knobs): the probe's tokens served alone and in run 3's group through `Runtime.serve` (grouped
    batch-1, one KV capacity per group), and every short prompt decoded directly at its own capacity
    and at each of `X1_CAPACITIES`, with the largest logit difference before the first differing token."""
    from huggingface_hub import snapshot_download
    import mlx.core as mx
    from mlx_lm import load

    from ironmule.plans import StrictOneShotPlan
    from ironmule.runtime import BASELINE, Engine
    from ironmule.service import Request, Runtime, ThroughputMode

    path = Path(snapshot_download(model, revision=X1_REVISION if model == MODEL else None,
                                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja",
                                                  "tokenizer*"]))
    loaded, tokenizer = load(str(path))
    runtime = Runtime(Engine(loaded, tokenizer, BASELINE), tokenizer, mode=ThroughputMode(max_width=4))
    backend, repeats, limit = runtime.backend, 1 if smoke else X1_REPEATS, 4 if smoke else 32
    items = trace_h2(*X1_TRACE, 1.0)  # prompts, lengths and limits do not depend on the capacity
    group = [{**items[i], "max_tokens": min(items[i]["max_tokens"], limit) if smoke else items[i]["max_tokens"]}
             for i in X1_GROUP]
    probe = group[X1_GROUP.index(X1_PROBE)]

    def serve(members: list[dict[str, Any]]) -> list[str]:
        requests = [Request(prompt_ids=_x1_ids(tokenizer, m), max_tokens=m["max_tokens"]) for m in members]
        return [_token_digest(result.tokens) for result in runtime.serve(requests)]

    def decode(ids: list[int], capacity: int) -> tuple[list[int], list[Any]]:
        state, token = backend.prefill(ids, StrictOneShotPlan(), capacity)
        tokens, logits = [token], []
        for _ in range(limit - 1):
            out, pick = backend.step(state, tokens[-1], capacity)
            mx.eval(out[0], pick)
            state = out[1]
            logits.append(out[0][:, -1, :].astype(mx.float32))
            tokens.append(int(pick.reshape((-1,)).item()))
        return tokens, logits

    try:
        alone = [serve([probe])[0] for _ in range(repeats)]
        grouped = [serve(group)[X1_GROUP.index(X1_PROBE)] for _ in range(repeats)]
        sweep = []
        for prompt in range(2 if smoke else len(PROMPTS)):
            ids = _x1_ids(tokenizer, {"prompt": prompt, "long": False})
            own = backend.capacity_for([len(ids)], limit)
            base_tokens, base_logits = decode(ids, own)
            row: dict[str, Any] = {"prompt": prompt, "own_capacity": own, "digest": _token_digest(base_tokens),
                                   "at": {}}
            for capacity in X1_CAPACITIES:
                tokens, logits = decode(ids, capacity)
                first = next((i for i, (a, b) in enumerate(zip(base_tokens, tokens)) if a != b), None)
                same_input = base_logits[: (len(base_logits) if first is None else max(0, first - 1))]
                row["at"][str(capacity)] = {
                    "equal": tokens == base_tokens, "first_difference": first,
                    "max_logit_difference": max((float(mx.max(mx.abs(a - b)).item())
                                                 for a, b in zip(same_input, logits)), default=0.0),
                    "digest": _token_digest(tokens)}
            sweep.append(row)
    finally:
        runtime.close()
    return {"probe": X1_PROBE, "revision": path.name, "alone": alone, "grouped": grouped, "sweep": sweep,
            "long_capacity": backend.capacity_for([max(len(_x1_ids(tokenizer, m)) for m in group)], limit)}


def x1_summary(devices: dict[str, dict[str, Any]], run3_digests: dict[str, str] | None = None) -> dict[str, Any]:
    """X1's decision (fixed in BACKLOG X1 before its data): does the KV capacity, the group, or the
    card move a request's tokens on this hardware?"""
    found = {name: result for name, result in devices.items() if not result.get("failed")}
    capacity_moves = sorted({(name, row["prompt"], capacity) for name, result in found.items()
                             for row in result["sweep"] for capacity, cell in row["at"].items() if not cell["equal"]})
    group_moves = [name for name, result in found.items() if set(result["alone"]) != set(result["grouped"])]
    unstable = [name for name, result in found.items()
                if len(set(result["alone"])) > 1 or len(set(result["grouped"])) > 1]
    digests = {name: [row["digest"] for row in result["sweep"]] for name, result in found.items()}
    cards_agree = len({json.dumps(d) for d in digests.values()}) <= 1
    if len(found) < len(devices) or not found:
        verdict = "incomplete: a device failed"
    elif unstable:
        verdict = "nondeterministic: the same request on the same device was answered differently"
    elif capacity_moves:
        verdict = "the KV capacity moves tokens on this hardware: a request's capacity must not depend on its group"
    elif group_moves:
        verdict = "the group moves tokens beyond its capacity: grouped batch-1 is not exact here"
    elif not cards_agree:
        verdict = "the cards answer differently"
    else:
        verdict = "not reproduced in-process: the product path differs from Runtime.serve"
    out = {"verdict": verdict, "capacity_moves": [list(m) for m in capacity_moves], "group_moves": group_moves,
           "unstable": unstable, "cards_agree": cards_agree}
    if run3_digests:  # which of run 3's two answers each way of serving reproduces
        out["run3"] = {name: {"alone": [run3_digests.get(d, "neither") for d in result["alone"]],
                              "grouped": [run3_digests.get(d, "neither") for d in result["grouped"]]}
                       for name, result in found.items()}
    return out


# run 3's two answers to request 14 of `bursts` trace 1 (SHA-256 of the token ids, first ten hex digits)
X1_RUN3 = {"3f93c7c2f9": "shortest arms", "3826c1eacc": "segregate and spill arms, and the same request in `spike`"}


def run_x1(model: str, out: Path, smoke: bool = False) -> dict[str, Any]:
    """One child process per visible CUDA device, each pinned to it as HW1 pins a worker. A Kaggle
    kernel runs this file as a notebook cell, so the child gets the source the kernel embedded."""
    from ironmule_product.device_pool import cuda_devices

    child_file = out.with_name("x1_child.py")
    child_file.write_text(globals().get("_EMBEDDED_SOURCE") or Path(__file__).read_text())
    children = {}
    for device in cuda_devices() or [None]:
        name = device or "default"
        target = out.with_name(f"x1_device_{name}.json")
        env = {**os.environ, X1_CHILD: str(target), "IRONMULE_X1_MODEL": model,
               "IRONMULE_X1_SMOKE": "1" if smoke else ""}
        if device is not None:
            env["CUDA_VISIBLE_DEVICES"] = device
        children[name] = (target, subprocess.Popen([sys.executable, str(child_file)], env=env))
    results = {}
    for name, (target, child) in children.items():
        code = child.wait()
        results[name] = (json.loads(target.read_text()) if code == 0 and target.exists()
                         else {"failed": f"exit {code}"})
    run3 = {digest: label for result in results.values() if not result.get("failed")
            for digest in result["alone"] + result["grouped"] for key, label in X1_RUN3.items()
            if digest.startswith(key)}
    return {"devices": results, "summary": x1_summary(results, run3)}


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
    source = Path(__file__).read_text()
    body = (source.replace('\nif __name__ == "__main__":\n    sys.exit(main())\n', "")
            .replace("from __future__ import annotations\n", "", 1))
    if plan == "x1":  # `run_x1` starts its per-device children from this file's own source
        encoded = base64.b64encode(source.encode()).decode()
        body = f"_EMBEDDED_SOURCE = __import__('base64').b64decode({encoded!r}).decode()\n" + body
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
    child = os.environ.get(X1_CHILD)
    if child:  # an X1 child pinned to one device by `run_x1`
        result = exactness_on_device(os.environ["IRONMULE_X1_MODEL"], smoke=bool(os.environ.get("IRONMULE_X1_SMOKE")))
        Path(child).write_text(json.dumps(result, sort_keys=True) + "\n")
        return 0
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="write the measurement report here")
    parser.add_argument("--model", default=None)
    parser.add_argument("--plan", choices=("h1", "h2", "h3", "h4", "h5", "h6", "x1"), default="h1",
                        help="run 1's to 4's fixed plan, RL4-S's or RL6's run, or X1's exactness diagnostic")
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
        if args.plan in ("h4", "h5", "h6"):
            if not args.policy_json or (not args.best_fixed and args.plan != "h6"):
                parser.error(f"--plan {args.plan} needs --policy-json" + ("" if args.plan == "h6" else " and --best-fixed"))
            text = Path(args.policy_json[1:]).read_text() if args.policy_json.startswith("@") else args.policy_json
            LearnedPolicy.from_json(json.loads(text))  # refuse a policy whose digest does not match
            extra = (["--best-fixed", args.best_fixed] if args.plan != "h6" else []) + \
                ["--policy-json", json.dumps(json.loads(text), sort_keys=True)]
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
    if args.plan == "x1":
        report["x1"] = run_x1(model, args.out, smoke=args.smoke)
        report["finished_unix"] = time.time()
        save()
        print(json.dumps({"report": str(args.out), "verdict": report["x1"]["summary"]["verdict"]}), flush=True)
        return 0
    policy = names = None
    if args.plan in ("h4", "h5", "h6"):
        if not args.policy_json or (not args.best_fixed and args.plan != "h6"):
            parser.error(f"--plan {args.plan} needs --policy-json" + ("" if args.plan == "h6" else " and --best-fixed"))
        text = Path(args.policy_json[1:]).read_text() if args.policy_json.startswith("@") else args.policy_json
        policy = LearnedPolicy.from_json(json.loads(text))
        names = arms_h6() if args.plan == "h6" else arms_h4(args.best_fixed)
    report["rl2h"] = run(model, work, smoke=args.smoke, plan=args.plan, on_row=save, policy=policy, names=names)
    report["analysis"] = analyse(report["rl2h"])
    report["finished_unix"] = time.time()
    save()
    print(json.dumps({"report": str(args.out), "verdict": report["analysis"]["verdict"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
