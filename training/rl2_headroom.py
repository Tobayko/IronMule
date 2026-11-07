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

LD1-V (`--plan h7`, BACKLOG LD1-V, fixed 2026-10-29 before its data) runs the product's own learned dispatch (LD1)
with a given policy against the product's reference and RL6's narrow shortest-first schedules on all three states
(8 traces each, indices 47-54); a product cell whose dispatch was not active makes the run invalid.

H8 (`--plan h8`, BACKLOG KB1, fixed 2026-10-29 before its data) compares the product's default on this setup (the
tuned engine with RL6's qualified schedule) with the dense numeric plan (`--engine dense`) at width 8, in arrival
order and shortest first, on all three states (8 traces, indices 55-62). Each of four rounds starts both engines'
HW1 pools in turn, the first one alternating; answers must agree within an engine, not across engines.

The report keeps token counts, times and SHA-256 digests of answers, never prompt or answer text.
To run it on Kaggle (T4 x2) from a pushed commit:

    python training/rl2_headroom.py --write-kernel kernel --owner <user> --commit <sha> [--plan h2|h3|x1]
    kaggle kernels push -p kernel --accelerator NvidiaTeslaT4
    kaggle kernels output <user>/ironmule-rl2-headroom[-h2|-h3|-x1] -p results
    python training/rl2_headroom.py --analyse results/rl2_headroom_report.json
"""

from __future__ import annotations

import argparse
import base64
import copy
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
H7_TRACES = tuple(range(47, 55))   # LD1-V: 8 traces per state, all three states, seeds unused so far
PRODUCT = "product/w8/g0"          # LD1-V: the product's own learned dispatch (LD1) deciding every group
H2_MAX_TOKENS = (32, 256)
# Run 2's long prompts: generic filler of about a thousand tokens before the question, never user text.
CONTEXT = "Background notes, not needed for the answer. " + " ".join(PROMPTS * 8)


def prompt_text(item: dict[str, Any]) -> str:
    """A trace item's prompt: its question, behind the background notes once (`long`) or `context` times (H11);
    an AG2 item carries a real instruction prompt as `text`."""
    if item.get("text"):
        return item["text"]
    return "".join(CONTEXT + "\n\n" for _ in range(item.get("context") or (1 if item.get("long") else 0))) \
        + PROMPTS[item["prompt"]]


# ---------------------------------------------------------------- the plan, fixed before data

def arm_name(placement: str, width: int, gather_ms: int) -> str:
    return f"{placement}/w{width}/g{gather_ms}"


def arm_engine(name: str) -> str:
    """H8: an arm may name its engine (`dense:shortest/w8/g0`); without one it is the tuned engine."""
    return name.split(":", 1)[0] if ":" in name else "tuned"


def _bare(name: str) -> str:
    return name.split(":", 1)[-1]


def parse_arm(name: str) -> tuple[str, int, int]:
    placement, width, gather = _bare(name).split("/")[:3]
    return placement, int(width[1:]), int(gather[1:])


def arm_order(name: str) -> str:
    """Run 3's fourth part: `fifo` (omitted) or `sjf`."""
    parts = _bare(name).split("/")
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
    idle = [row for row in rows if row["arm"] == PRODUCT and not (row.get("dispatch") or {}).get("active")]
    if idle:  # LD1-V: a product cell without an active learned dispatch was the reference in disguise
        problems.append(f"{len(idle)} product cells ran without an active learned dispatch")
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
    challenger = LEARNED if LEARNED in names else PRODUCT if PRODUCT in names else None  # LD1-V: the product's
    if challenger is not None:  # run 4 (BACKLOG RL4-H): the frozen policy against every fixed arm, paired by trace
        fixed_arms = [n for n in names if n != challenger]
        rival = min(fixed_arms, key=lambda n: (sum(WEIGHTS[level] * statistics.fmean(table[level, t, n] for t in complete)
                                                   for level in levels), n))
        per_trace = [(sum(WEIGHTS[level] * table[level, t, challenger] for level in levels),
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
        # LD1-V: the challenger against the reference, paired by trace (after the draws above)
        if challenger == PRODUCT:
            paired = [(sum(WEIGHTS[level] * table[level, t, challenger] for level in levels),
                       sum(WEIGHTS[level] * table[level, t, REFERENCE] for level in levels)) for t in complete]
            boot = []
            for _ in range(resamples):
                sample = [paired[rng.randrange(len(paired))] for _ in paired]
                boot.append(1 - sum(x for x, _ in sample) / sum(y for _, y in sample))
            learned["vs_reference"] = {"gain": 1 - sum(x for x, _ in paired) / sum(y for _, y in paired),
                                       "gain_ci95": [_quantile(boot, 0.025), _quantile(boot, 0.975)]}
        # LD1-V: the challenger against the rival in each state, for the no-harm check (after the draws above)
        learned["per_state"] = {}
        for level in levels:
            paired = [(table[level, t, challenger], table[level, t, rival]) for t in complete]
            boot = []
            for _ in range(resamples):
                sample = [paired[rng.randrange(len(paired))] for _ in paired]
                boot.append(1 - sum(x for x, _ in sample) / sum(y for _, y in sample))
            learned["per_state"][str(level)] = {"gain": 1 - sum(x for x, _ in paired) / sum(y for _, y in paired),
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
    `training/rl4_agent.py`); the harness applies it greedily. With `width` (RL6) it also picks each
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
        """(place, order) callables in `training/rl4_agent.py`'s simulator signature."""
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

        def _order(self, request: Any = None) -> list[int]:
            if self.placement == "product":  # LD1-V: the product's own learned dispatch places it
                return super()._order(request)
            order = super()._order()  # without a request: the product's own learned dispatch (LD1) stays out
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
    pool.attach(getattr(pool, "product_dispatch", None) if placement == "product" else None)
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
                               "digest": None, "tokens": 0, "context": item.get("context") or 0}
        ids: list[Any] = []
        request = GenerationRequest(model, (("user", prompt_text(item)),), max_tokens=item["max_tokens"], stream=True)
        finish = None
        try:
            for event in pool.stream(request):
                if event.get("type") == "token":
                    if out["first_s"] is None and event.get("text"):
                        out["first_s"] = time.monotonic() - scheduled
                    ids.append(event.get("token_id") if event.get("token_id") is not None else event.get("text"))
                    if len(ids) == 2:  # KP1: a new decode body's compile delays the second token, not the first
                        out["second_s"] = time.monotonic() - scheduled
                elif event.get("type") == "done":
                    out["outcome"], out["tokens"] = "ok", int(event.get("completion_tokens") or 0)
                    out["prompt_tokens"] = event.get("prompt_tokens")
                    finish = event.get("finish_reason")
                    engine = event.get("engine") if isinstance(event.get("engine"), dict) else {}
                    if engine.get("decode_bodies") is not None:
                        out["bodies"], out["device"] = engine["decode_bodies"], engine.get("cuda_device")
        except Overloaded:
            out["outcome"] = "rejected"
        except RequestTimeout:
            out["outcome"] = "timeout"
        except ProductError as exc:
            out["error"] = exc.code
        except Exception as exc:  # noqa: BLE001 - any other failure is still this request's outcome
            out["error"] = type(exc).__name__
        out["end_s"] = time.monotonic() - scheduled
        if out["outcome"] == "ok":  # the tokens, and since KP1 how the answer ended
            out["digest"] = hashlib.sha256(json.dumps([ids, finish, out["tokens"]]).encode()).hexdigest()
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


def calibrate(pool: Any, model: str, deadline_s: float, max_tokens: tuple[int, ...] | None = None,
              items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Requests completed per second under the reference from a burst of 16 per GPU at once (`items`, AG2: the
    workload's own requests instead of the fixed questions)."""
    set_arm(pool, REFERENCE)
    burst = [{**item, "offset_s": 0.0} for item in (items or trace(9, 0, 1.0, 16 * len(pool.shards), max_tokens))]
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
            on_row: Any = None, dispatch: Any = None) -> dict[str, Any]:
    """The plan above on a started pool: calibration, then every (trace, level) block in order.
    `names` and `max_tokens` shrink it for a smoke run; a measurement keeps the defaults."""
    pool = _placement_pool(pool)
    pool.learned_policy = policy
    pool.product_dispatch = dispatch  # LD1-V: attached only while the product arm runs
    for shard in pool.shards:
        shard.learned_policy = policy
    names = names or ({"h7": lambda gpus: arms_h7(), "h2": arms_h2, "h3": arms_h3}.get(plan, arms))(len(pool.shards))
    if any(rl_part == "learned" for name in names for rl_part in (parse_arm(name)[0], arm_order(name))) \
            and policy is None:
        raise ValueError("a learned arm needs a policy")
    warm = replay(pool, model, [{"offset_s": 0.0, "prompt": 0, "max_tokens": 8}] * len(pool.shards), deadline_s)
    if PRODUCT in names and (dispatch is None or not dispatch.active):
        raise ValueError("the product arm needs an active learned dispatch: " + str(getattr(dispatch, "reason", None)))
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
    elif plan == "h7":
        report["states"] = list(H2_STATES)
        traffic = [lambda t, s=s: trace_h2(s, t, capacity, requests, max_tokens) for s in range(len(H2_STATES))]
    elif plan in ("h2", "h3", "h4"):
        report["states"] = list(H2_STATES)
        traffic = [lambda t, s=s: trace_h2(s, t, capacity, requests, max_tokens) for s in range(len(H2_STATES))]
    else:
        report["rates"] = [load * capacity for load in loads]
        traffic = [lambda t, level=level, rate=rate: trace(level, t, rate, requests, max_tokens)
                   for level, rate in enumerate(report["rates"])]
    block = 0
    indices = (range(traces) if traces is not None else
               {"h5": H5_TRACES, "h6": H6_TRACES, "h7": H7_TRACES}.get(plan)
               or range(H2_TRACES if plan in ("h2", "h3", "h4") else TRACES))
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
                if name == PRODUCT:  # LD1-V validity: the learned dispatch really decided this cell
                    row["dispatch"] = pool.health().get("learned_dispatch")
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


def arms_h7() -> list[str]:
    """LD1-V: the product's reference, its learned dispatch, and RL6's narrow shortest-first schedules."""
    return [REFERENCE, PRODUCT, "shortest/w2/g0/sjf", "shortest/w1/g0/sjf"]


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
    return list(tokenizer.apply_chat_template([{"role": "user", "content": prompt_text(item)}], add_generation_prompt=True,
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
        dispatch = None
        if plan == "h7":  # LD1-V: the product's LearnedDispatch with this policy; a measurement needs no evidence
            from ironmule_product.learned_dispatch import LearnedDispatch, Policy

            dispatch = LearnedDispatch(Policy.from_json(policy.to_json()), {"experiment": "LD1-V measurement"}, None)
            names = names if names is not None else arms_h7()
            sizes["names"] = names
        result = measure(pool, model, deadline_s=deadline, plan=plan, policy=None if plan == "h7" else policy,
                         on_row=on_row, dispatch=dispatch, **sizes)
        if dispatch is not None:
            result.update(dispatch_policy=policy.to_json(), dispatch_status=dispatch.status())
    finally:
        pool.close()
    return {"model": model, "revision": revision, **result}


# ---------------------------------------------------------------- H8: the dense plan through the product (BACKLOG KB1)

H8_TRACES = range(55, 63)       # fresh seeds: no earlier run, training or simulation used them
H8_ROUNDS = 4                   # each round runs both engines' pools in turn, the first one alternating
ENGINE_CONFIGURATIONS = {"tuned": "current_profile", "dense": "dense_stacked"}


def arms_h8() -> list[str]:
    """Today's product default on this setup (RL6's qualified schedule on the tuned engine) and the dense plan at
    width 8, in arrival order and shortest first. Within an engine the first arm runs twice per block (A/A)."""
    return ["tuned:shortest/w2/g0/sjf", "dense:shortest/w8/g0", "dense:shortest/w8/g0/sjf"]


def measure_h8(start: Any, model: str, *, deadline_s: float, traces: Iterable[int] = H8_TRACES,
               rounds: int = H8_ROUNDS, requests: int = REQUESTS, max_tokens: tuple[int, ...] | None = None,
               on_row: Any = None, names: list[str] | None = None, engines: tuple[str, ...] = ("tuned", "dense"),
               plan: str = "h8", stress: Any = None, traffic: Any = None,
               warm_extra: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """`start(engine)` returns a started pool for that engine. Calibration runs once on the tuned pool; then each
    round measures its traces on both engines, every (trace, state) block an engine's arms in HW3's block order.
    KP1 reuses it with its own arms and engines, and `stress(round, capacity)`, when given, adds one recorded
    cell per engine and round (not part of any decision) after that engine's blocks. H11 passes its own
    `traffic(state, trace, capacity, requests, max_tokens)` and `warm_extra` requests sent once per card in the
    warm-up."""
    names = names or arms_h8()
    traffic = traffic or trace_h2
    traces = list(traces)
    per_round = -(-len(traces) // rounds)
    report: dict[str, Any] = {"plan": plan, "arms": names, "states": list(H2_STATES), "deadline_s": deadline_s,
                              "rows": [], "engines": {}, "stress_rows": []}
    capacity = None
    block = 0
    for r in range(rounds):
        chunk = traces[r * per_round:(r + 1) * per_round]
        if not chunk:
            break
        shift = r % len(engines)  # the first engine rotates by round (two engines: they alternate)
        order = list(engines[shift:]) + list(engines[:shift])
        for engine in order:
            pool = _placement_pool(start(engine))
            try:
                # Run 2: both engines warm with the same burst of eight mixed-length requests per card, so neither
                # pays its first compiles (bodies per capacity; the plan's kernels compile in its load probe) in
                # a timed cell.
                warm = replay(pool, model, [{"offset_s": 0.0, "prompt": i % len(PROMPTS), "long": i % 2 == 1,
                                             "max_tokens": 8} for i in range(8 * len(pool.shards))]
                              + [dict(item) for item in (warm_extra or []) for _ in pool.shards], deadline_s)
                report["engines"].setdefault(engine, []).append(
                    {"round": r, "warmup_ok": sum(w.get("outcome") == "ok" for w in warm),
                     "warmup_failed": sum(w.get("outcome") not in COSTED_OUTCOMES for w in warm),
                     "execution": pool.health().get("execution"), "devices": list(pool.devices),
                     "engine": _engine_metadata(pool)})
                if report["engines"][engine][-1]["warmup_failed"]:  # run 1: a pool that fails is not measured
                    report["stopped"] = f"the {engine} pool failed a warm-up request"
                    report["stopped_health"] = [shard.health() for shard in getattr(pool, "shards", [pool])]
                    return report
                if capacity is None:
                    if engine != "tuned":  # calibrate on the reference engine first, so traffic never depends on the plan
                        pool.close()
                        pool = _placement_pool(start("tuned"))
                    report["calibration"] = calibrate(pool, model, deadline_s, max_tokens)
                    if report["calibration"].get("failed"):
                        return report
                    capacity = report["calibration"]["capacity_rps"]
                    if engine != "tuned":
                        pool.close()
                        pool = _placement_pool(start(engine))
                engine_names = [n for n in names if arm_engine(n) == engine]
                for t in chunk:
                    for level in range(len(H2_STATES)):
                        items = traffic(level, t, capacity, requests, max_tokens)
                        for name, run in block_order(engine_names, block):
                            if not wait_idle(pool):
                                report["stopped"] = f"the {engine} pool did not drain before block {block}"
                                return report
                            set_arm(pool, name)
                            joined_before = pool.health().get("joined_requests") or 0
                            row = {"level": level, "trace": t, "arm": name, "run": run, "block": block, "round": r,
                                   "engine": engine, "execution": pool.health().get("execution"),
                                   "started_unix": time.time(), "requests": replay(pool, model, items, deadline_s)}
                            row["joined"] = (pool.health().get("joined_requests") or 0) - joined_before  # H10
                            report["rows"].append(row)
                            if on_row is not None:
                                on_row(report)
                        block += 1
                if stress is not None:
                    if not wait_idle(pool):
                        report["stopped"] = f"the {engine} pool did not drain before its stress cell"
                        return report
                    set_arm(pool, engine_names[0])
                    report["stress_rows"].append({"round": r, "engine": engine, "arm": engine_names[0],
                                                  "requests": replay(pool, model, stress(r, capacity), deadline_s)})
                    if on_row is not None:
                        on_row(report)
            finally:
                pool.close()
    return report


COSTED_OUTCOMES = ("ok", "timeout", "rejected")  # H8 run 2: anything else is a defect of the path under test


def _engine_metadata(pool: Any) -> list[dict[str, Any]]:
    """What each card's worker reported it runs: knobs, where they came from, the plan (no request data)."""
    rows = []
    for shard in getattr(pool, "shards", [pool]):
        payload = getattr(getattr(shard, "backend", None), "ready_payload", None) or {}
        engine = payload.get("engine") or {}
        rows.append({key: engine.get(key) for key in ("configuration", "profile_source", "selected_knobs",
                                                       "dense_plan")})
    return rows


def analyse_h8(report: dict[str, Any], resamples: int = RESAMPLES, seed: int = 0) -> dict[str, Any]:
    """H8's decisions, as fixed in BACKLOG before data."""
    rows, deadline = report.get("rows", []), float(report.get("deadline_s") or 120)
    names = report.get("arms") or arms_h8()
    levels = list(range(len(report.get("states") or H2_STATES)))
    problems: list[str] = []
    if report.get("calibration", {}).get("failed") or "stopped" in report:
        problems.append("calibration failed or the run stopped")
    digests: dict[tuple[str, int, int, int], set[str]] = {}
    for row in rows:
        for request in row["requests"]:
            if request.get("outcome") == "ok" and request.get("digest"):
                digests.setdefault((row["engine"], row["level"], row["trace"], request["index"]), set()).add(request["digest"])
    mismatched = sorted(key for key, found in digests.items() if len(found) > 1)
    if mismatched:
        problems.append(f"{len(mismatched)} requests answered differently across arms of one engine")
    plain = [row for row in rows if row["engine"] == "dense" and row.get("execution") != "numeric@dense"]
    if plain:
        problems.append(f"{len(plain)} dense cells did not run the dense plan")
    # Fixed after run 1's data and before run 2's (run 1: every dense request failed with a product error and
    # the run still read valid): a request that fails, or ends without an outcome, is a defect of the path under
    # test, not a cost; only timeouts and rejections are costs.
    failures: dict[str, dict[str, int]] = {}
    for row in rows:
        for request in row["requests"]:
            if request.get("outcome") not in COSTED_OUTCOMES:
                code = str(request.get("error") or request.get("outcome") or "no outcome")
                failures.setdefault(row["engine"], {})[code] = failures.setdefault(row["engine"], {}).get(code, 0) + 1
    if failures:
        problems.append(f"{sum(sum(c.values()) for c in failures.values())} requests failed: {failures}")
    cells = _cell_costs(rows, deadline)
    complete = [t for t in sorted({row["trace"] for row in rows})
                if all((level, t, name, 0) in cells for level in levels for name in names)
                and all((level, t, names[0], 1) in cells and (level, t, names[1], 1) in cells for level in levels)]
    if len(complete) < 2:
        problems.append(f"{len(complete)} complete traces; at least two are needed")
    out: dict[str, Any] = {"schema": "ironmule.rl2_headroom.h8.analysis.v1", "complete_traces": complete,
                           "mismatched_requests": len(mismatched)}
    if problems:
        return {**out, "valid": False, "problems": problems}
    pairs = [(cells[level, t, name, 0], cells[level, t, name, 1]) for name in names[:2] for level in levels
             for t in complete]
    noise = _quantile([abs(a - b) / ((a + b) / 2) for a, b in pairs], 0.95)
    rng = random.Random(seed)

    def paired(arm: str, against: str, level: int | None = None) -> dict[str, Any]:
        def cost(name: str, t: int) -> float:
            return (cells[level, t, name, 0] if level is not None
                    else sum(WEIGHTS[lv] * cells[lv, t, name, 0] for lv in levels))
        values = [(cost(arm, t), cost(against, t)) for t in complete]
        boot = []
        for _ in range(resamples):
            sample = [values[rng.randrange(len(values))] for _ in values]
            boot.append(1 - sum(x for x, _ in sample) / sum(y for _, y in sample))
        return {"gain": 1 - sum(x for x, _ in values) / sum(y for _, y in values),
                "gain_ci95": [_quantile(boot, 0.025), _quantile(boot, 0.975)]}

    reference = names[0]
    dense_arms = [n for n in names if arm_engine(n) == "dense"]
    weighted = {n: sum(WEIGHTS[lv] * statistics.fmean(cells[lv, t, n, 0] for t in complete) for lv in levels)
                for n in names}
    best = min(dense_arms, key=lambda n: (weighted[n], n))
    pooled = paired(best, reference)
    per_state = {str(level): paired(best, reference, level) for level in levels}
    agree = sum(len(digests.get(("dense", level, t, i), set()) & digests.get(("tuned", level, t, i), set())) > 0
                for (engine, level, t, i) in digests if engine == "dense")
    return {**out, "valid": True, "problems": [], "aa_noise": noise, "aa_pairs": len(pairs), "best_dense": best,
            "(1) dense beats today's default": {**pooled, "passes": pooled["gain_ci95"][0] > noise},
            "(2) no harm in any state": {"per_state": per_state,
                                         "passes": all(v["gain_ci95"][0] > -noise for v in per_state.values())},
            "each_dense_arm": {n: paired(n, reference) for n in dense_arms},
            "sjf_within_dense": paired(dense_arms[1], dense_arms[0]) if len(dense_arms) > 1 else None,
            "mean_cost_per_level": {str(lv): {n: statistics.fmean(cells[lv, t, n, 0] for t in complete) for n in names}
                                    for lv in levels},
            "answers_same_as_today": agree, "answers_compared": sum(1 for key in digests if key[0] == "dense")}


def run_h8(model: str, work: Path, *, smoke: bool, on_row: Any = None) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    from ironmule_product.backend import DENSE_STARTUP_TIMEOUT, MLXWorkerClient
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

    def start(engine: str) -> Any:  # one startup limit for both: startup is not in the cost (run 2)
        return start_pool(store, spec, devices, lambda device: MLXWorkerClient(
            spec, execution_variant="current_engine", engine_configuration=ENGINE_CONFIGURATIONS[engine],
            cuda_device=device, startup_timeout=DENSE_STARTUP_TIMEOUT), ordered_service())

    deadline = float(store.settings()["request_timeout_s"] or 120)
    preflight = dense_preflight(str(spec.snapshot_path), devices[0])
    if not preflight.get("ok"):
        return {"model": model, "revision": revision, "failed": "dense preflight", "preflight": preflight}
    sizes = {"traces": [H8_TRACES[0]], "rounds": 1, "requests": 3, "max_tokens": (4, 8)} if smoke else {}
    result = measure_h8(start, model, deadline_s=deadline, on_row=on_row, **sizes)
    return {"model": model, "revision": revision, "preflight": preflight, **result}


_PREFLIGHT = r"""
import json, sys, time, traceback
out = {"ok": False}
try:
    import mlx.core as mx
    from mlx_lm import load
    from ironmule.dense_plan import install
    from ironmule.runtime import Engine, Knobs
    from ironmule.service import DenseStackedMode, Request, Runtime
    model, tokenizer = load(sys.argv[1])
    started = time.perf_counter()
    plan, out["record"] = install(model, mx.device_info() if mx.cuda.is_available() else None)
    out["install_s"] = time.perf_counter() - started
    runtime = Runtime(Engine(model, tokenizer, Knobs()), tokenizer, mode=DenseStackedMode(plan))
    prompts = [tokenizer.encode(text) for text in json.loads(sys.argv[2])]
    alone = [runtime.serve([Request(prompt_ids=ids, max_tokens=16)])[0].tokens for ids in prompts]
    started = time.perf_counter()
    together = [r.tokens for r in runtime.serve([Request(prompt_ids=ids, max_tokens=16) for ids in prompts])]
    out["together_s"] = time.perf_counter() - started
    out["together_equals_alone"] = together == alone
    out["free_slots_after"] = runtime.backend.free_slots()
    out["ok"] = out["together_equals_alone"] and out["free_slots_after"] == plan.slots
except Exception:
    out["error"] = traceback.format_exc(limit=12)
print(json.dumps(out))
"""


def dense_preflight(snapshot: str, device: str | None, timeout_s: float = 900.0) -> dict[str, Any]:
    """H8 run 2: before any pool starts, the dense plan installs on one GPU in a process of its own and serves
    three requests together as it serves them alone; on failure the run stops with the error (run 1 spent its
    session on a path that never served). The traceback stays in the private kernel output."""
    environment = dict(os.environ, **({"CUDA_VISIBLE_DEVICES": device} if device is not None else {}))
    try:
        done = subprocess.run([sys.executable, "-c", _PREFLIGHT, snapshot, json.dumps(list(PROMPTS[:3]))],
                              capture_output=True, text=True, timeout=timeout_s, env=environment)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"no answer in {timeout_s:.0f} s"}
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    try:
        return json.loads(lines[-1])
    except (IndexError, ValueError):
        return {"ok": False, "error": f"exit {done.returncode}", "stderr_tail": done.stderr[-4000:]}


# ---------------------------------------------------------------- KP1, fixed in BACKLOG before data

KP1_TRACES = range(63, 75)      # fresh seeds: no earlier run, training or simulation used them
KP1_ROUNDS = 4                  # three traces per round; each round starts both engines' pools, alternating
KP1_CAPACITIES = (64, 128, 256, 384, 512, 768, 1024, 1280, 1536, 2048, 3072, 4096, 5120, 6144, 7168, 8192)
KP1_STEPS = 24                  # decode steps per preflight comparison, fed the baseline's tokens
STRESS_MAX_TOKENS = tuple(40 + 64 * k for k in range(16))  # 16 distinct capacities for one short prompt
ENGINE_CONFIGURATIONS["kp"] = "current_profile"            # the tuned engine with KP1's knobs


def arms_kp1() -> list[str]:
    """Today's default (RL6's qualified schedule on the tuned engine) without and with KP1's knobs; each runs
    twice per block (A/A for both engines)."""
    return ["tuned:shortest/w2/g0/sjf", "kp:shortest/w2/g0/sjf"]


def trace_stress(index: int, capacity: float) -> list[dict[str, Any]]:
    """KP1's recorded stress cell: quiet arrivals, short prompts, one output limit per distinct capacity, so more
    capacities are active than the engine keeps decode bodies."""
    rng = random.Random(900 + index)
    limits = list(STRESS_MAX_TOKENS)
    rng.shuffle(limits)
    return [{**item, "long": False, "max_tokens": limit}
            for item, limit in zip(trace(0, 900 + index, 0.2 * capacity, len(limits)), limits)]


_KP1_PREFLIGHT = r"""
import json, sys, time, traceback
out = {"ok": False}
try:
    from mlx_lm import load
    from ironmule.runtime import Knobs
    from ironmule_product.knob_profile import compare_class_a
    spec = json.loads(sys.argv[2])
    model, tokenizer = load(sys.argv[1])

    def ids_of(text):
        return list(tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True))

    def cap(n, m):
        return ((n + m + 8 + 63) // 64) * 64

    produced = set()
    for text in spec["prompts"]:
        for long in (False, True):
            n = len(ids_of((spec["context"] + "\n\n" if long else "") + text))
            produced |= {cap(n, m) for m in spec["max_tokens"] + ([] if long else spec["stress"])}
    prompts = {"short": ids_of(spec["prompts"][0]), "long": ids_of(spec["context"] + "\n\n" + spec["prompts"][0])}
    started = time.perf_counter()
    out = compare_class_a(model, tokenizer, Knobs(**spec["knobs"]), prompts,
                          sorted(set(spec["capacities"]) | produced), spec["steps"])
    out.update(produced=sorted(produced), seconds=time.perf_counter() - started)
except Exception:
    out["error"] = traceback.format_exc(limit=12)
print(json.dumps(out))
"""


def kp1_preflight(snapshot: str, device: str | None, knobs: dict[str, Any], timeout_s: float = 1800.0) -> dict[str, Any]:
    """KP1's kill 1, HW4's method in a process of its own on one card: prefill logits, 24 decode steps fed the
    baseline's tokens and the final state, bit for bit, baseline against `knobs`, on a short and a long prompt at
    every capacity in `KP1_CAPACITIES` and every capacity this run's traffic produces."""
    spec = {"prompts": list(PROMPTS), "context": CONTEXT, "max_tokens": list(H2_MAX_TOKENS),
            "stress": list(STRESS_MAX_TOKENS), "capacities": list(KP1_CAPACITIES), "steps": KP1_STEPS,
            "knobs": knobs}
    environment = dict(os.environ, **({"CUDA_VISIBLE_DEVICES": device} if device is not None else {}))
    try:
        done = subprocess.run([sys.executable, "-c", _KP1_PREFLIGHT, snapshot, json.dumps(spec)],
                              capture_output=True, text=True, timeout=timeout_s, env=environment)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"no answer in {timeout_s:.0f} s"}
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    try:
        return json.loads(lines[-1])
    except (IndexError, ValueError):
        return {"ok": False, "error": f"exit {done.returncode}", "stderr_tail": done.stderr[-4000:]}


def run_kp1(model: str, work: Path, *, smoke: bool, on_row: Any = None) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    from ironmule_product.backend import DENSE_STARTUP_TIMEOUT, MLXWorkerClient
    from ironmule_product.device_pool import cuda_devices, start_pool
    from ironmule_product.knob_profile import kp1_knobs
    from ironmule_product.state import ProductStore

    revision = Path(snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                                             "*.jinja", "tokenizer*"])).name
    state = work / "rl2h-state"
    _ironmule("setup", "--mode", "server", state=state)
    added = _ironmule("models", "add", model, state=state)
    if added.get("returncode") != 0:
        return {"model": model, "revision": revision, "failed": "models add", "added": added}
    store = ProductStore(state)
    spec = store.model(model)
    devices = cuda_devices() or [None]
    knobs = kp1_knobs()
    preflight = kp1_preflight(str(spec.snapshot_path), devices[0], knobs)
    if not preflight.get("ok"):
        return {"model": model, "revision": revision, "failed": "kp1 preflight (kill 1)", "preflight": preflight}

    def start(engine: str) -> Any:  # the same startup limit for both engines; startup is not in the cost
        extra = {"engine_knobs": {"knobs": knobs, "source": "KP1-candidate"}} if engine == "kp" else {}
        return start_pool(store, spec, devices, lambda device: MLXWorkerClient(
            spec, execution_variant="current_engine", engine_configuration="current_profile", cuda_device=device,
            startup_timeout=DENSE_STARTUP_TIMEOUT, **extra), ordered_service())

    deadline = float(store.settings()["request_timeout_s"] or 120)
    sizes = ({"traces": [KP1_TRACES[0]], "rounds": 1, "requests": 3, "max_tokens": (4, 8)} if smoke else
             {"traces": KP1_TRACES, "rounds": KP1_ROUNDS})
    result = measure_h8(start, model, deadline_s=deadline, on_row=on_row, names=arms_kp1(), engines=("tuned", "kp"),
                        plan="kp1", stress=None if smoke else trace_stress, **sizes)
    return {"model": model, "revision": revision, "preflight": preflight, "knobs": knobs, **result}


def analyse_kp1(report: dict[str, Any], resamples: int = RESAMPLES, seed: int = 0) -> dict[str, Any]:
    """KP1's kills, validity and decisions, as fixed in BACKLOG before data. A cell's cost is the mean of its
    two runs. The A/A control applies the same estimator to an arm's first run against its second; KP's bound
    must clear the larger of those two gains in size, and 0."""
    from ironmule_product.knob_profile import BASELINE_KNOBS, kp1_knobs

    rows, deadline = report.get("rows", []), float(report.get("deadline_s") or 120)
    names = report.get("arms") or arms_kp1()
    levels = list(range(len(report.get("states") or H2_STATES)))
    problems: list[str] = []
    if report.get("calibration", {}).get("failed") or "stopped" in report:
        problems.append("calibration failed or the run stopped")
    if not (report.get("preflight") or {}).get("ok"):
        problems.append("the preflight did not pass (kill 1)")
    expected = {"tuned": BASELINE_KNOBS, "kp": kp1_knobs()}
    for engine, starts in (report.get("engines") or {}).items():
        for record in starts:
            if any(card.get("selected_knobs") != expected.get(engine) for card in record.get("engine") or [{}]):
                problems.append(f"a {engine} pool did not run its arm's knobs")
                break
    failures: dict[str, dict[str, int]] = {}
    digests: dict[tuple[int, int, int], set[str]] = {}
    for row in rows:
        for request in row["requests"]:
            if request.get("outcome") not in COSTED_OUTCOMES:
                code = str(request.get("error") or request.get("outcome") or "no outcome")
                failures.setdefault(row["engine"], {})[code] = failures.setdefault(row["engine"], {}).get(code, 0) + 1
            elif request.get("outcome") == "ok" and request.get("digest"):
                digests.setdefault((row["level"], row["trace"], request["index"]), set()).add(request["digest"])
    if failures:
        problems.append(f"{sum(sum(c.values()) for c in failures.values())} requests failed: {failures}")
    mismatched = sorted(key for key, found in digests.items() if len(found) > 1)
    cells = _cell_costs(rows, deadline)
    complete = [t for t in sorted({row["trace"] for row in rows})
                if all((level, t, name, run) in cells for level in levels for name in names for run in (0, 1))]
    if len(complete) < 2:
        problems.append(f"{len(complete)} complete traces; at least two are needed")
    out: dict[str, Any] = {"schema": "ironmule.rl2_headroom.kp1.analysis.v1", "complete_traces": complete,
                           "mismatched_requests": len(mismatched), "failures": failures,
                           "kill 2 (answers differ on the product path)": bool(mismatched)}
    if mismatched:
        problems.append(f"kill 2: {len(mismatched)} requests answered differently across engines or runs")
    if problems:
        return {**out, "valid": False, "problems": problems}
    pairs = [(cells[level, t, name, 0], cells[level, t, name, 1]) for name in names for level in levels
             for t in complete]
    noise = _quantile([abs(a - b) / ((a + b) / 2) for a, b in pairs], 0.95)
    rng = random.Random(seed)

    def paired(cost: Any) -> dict[str, Any]:
        values = [cost(t) for t in complete]
        boot = []
        for _ in range(resamples):
            sample = [values[rng.randrange(len(values))] for _ in values]
            boot.append(1 - sum(x for x, _ in sample) / sum(y for _, y in sample))
        return {"gain": 1 - sum(x for x, _ in values) / sum(y for _, y in values),
                "gain_ci95": [_quantile(boot, 0.025), _quantile(boot, 0.975)]}

    def mean_cost(name: str, t: int, levels_: list[int], runs: tuple[int, ...] = (0, 1)) -> float:
        return sum(WEIGHTS[lv] * statistics.fmean(cells[lv, t, name, run] for run in runs) for lv in levels_) \
            / sum(WEIGHTS[lv] for lv in levels_)

    tuned, kp = names
    controls = {name: paired(lambda t, n=name: (mean_cost(n, t, levels, (1,)), mean_cost(n, t, levels, (0,))))
                for name in names}
    floor = max(0.0, *(abs(c["gain"]) for c in controls.values()))  # what the estimator finds between identical runs
    pooled = paired(lambda t: (mean_cost(kp, t, levels), mean_cost(tuned, t, levels)))
    per_state = {str(lv): paired(lambda t, lv=lv: (mean_cost(kp, t, [lv]), mean_cost(tuned, t, [lv]))) for lv in levels}
    seconds = {engine: [r["second_s"] - r["first_s"] for row in rows if row["engine"] == engine
                        for r in row["requests"] if r.get("second_s") is not None and r.get("first_s") is not None]
               for engine in ("tuned", "kp")}
    stress = {engine: [statistics.fmean(request_cost(r, deadline) for r in row["requests"])
                       for row in report.get("stress_rows", []) if row["engine"] == engine] for engine in ("tuned", "kp")}

    def bodies(rows_: list[dict[str, Any]], engine: str) -> dict[str, dict[str, int]]:
        top: dict[str, dict[str, int]] = {}
        for row in rows_:
            if row["engine"] != engine:
                continue
            for r in row["requests"]:
                if isinstance(r.get("bodies"), dict):
                    card = top.setdefault(str(r.get("device")), {"built": 0, "evicted": 0})
                    for key in card:
                        card[key] = max(card[key], int(r["bodies"].get(key, 0)))
        return top

    return {**out, "valid": True, "problems": [], "aa_noise": noise, "aa_pairs": len(pairs), "aa_control": controls,
            "aa_floor": floor,
            "(1) KP beats the default": {**pooled, "passes": pooled["gain_ci95"][0] > floor},
            "(2) no harm in any state": {"per_state": per_state,
                                         "passes": all(v["gain_ci95"][0] > -noise for v in per_state.values())},
            "mean_cost_per_level": {str(lv): {n: statistics.fmean(mean_cost(n, t, [lv]) for t in complete)
                                              for n in names} for lv in levels},
            "second_token_s": {e: {"median": statistics.median(v), "p95": _quantile(v, 0.95)} if v else None
                               for e, v in seconds.items()},
            "stress_cost": {e: statistics.fmean(v) if v else None for e, v in stress.items()},
            "stress_gain": (1 - statistics.fmean(stress["kp"]) / statistics.fmean(stress["tuned"])
                            if stress["kp"] and stress["tuned"] else None),
            "decode_bodies_last_seen": {e: bodies(rows + report.get("stress_rows", []), e) for e in ("tuned", "kp")}}


# ---------------------------------------------------------------- H9, fixed in BACKLOG before data

H9_TRACES = range(75, 87)       # fresh seeds: no earlier run, training or simulation used them
H9_ROUNDS = 3                   # a cyclic Latin square: each engine starts its round's pools once in each position


def arms_h9() -> list[str]:
    """The default, the same with KP1's knobs, and the dense plan; each runs twice per block (A/A for all)."""
    return ["tuned:shortest/w2/g0/sjf", "kp:shortest/w2/g0/sjf", "dense:shortest/w8/g0/sjf"]


def run_h9(model: str, work: Path, *, smoke: bool, on_row: Any = None) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    from ironmule_product.backend import DENSE_STARTUP_TIMEOUT, MLXWorkerClient
    from ironmule_product.device_pool import cuda_devices, start_pool
    from ironmule_product.knob_profile import kp1_knobs
    from ironmule_product.state import ProductStore

    revision = Path(snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                                             "*.jinja", "tokenizer*"])).name
    state = work / "rl2h-state"
    _ironmule("setup", "--mode", "server", state=state)
    added = _ironmule("models", "add", model, state=state)
    if added.get("returncode") != 0:
        return {"model": model, "revision": revision, "failed": "models add", "added": added}
    store = ProductStore(state)
    spec = store.model(model)
    devices = cuda_devices() or [None]
    knobs = kp1_knobs()
    preflight = kp1_preflight(str(spec.snapshot_path), devices[0], knobs)
    if not preflight.get("ok"):
        return {"model": model, "revision": revision, "failed": "kp1 preflight (kill 1)", "preflight": preflight}
    dense = dense_preflight(str(spec.snapshot_path), devices[0])
    engines, names = ("tuned", "kp", "dense"), arms_h9()
    if not dense.get("ok"):  # the dense part cannot run; the KP part, whose kill 1 passed, still does
        engines, names = ("tuned", "kp"), arms_h9()[:2]

    def start(engine: str) -> Any:  # one startup limit for every engine; startup is not in the cost
        extra = {"engine_knobs": {"knobs": knobs, "source": "KP1-candidate"}} if engine == "kp" else {}
        return start_pool(store, spec, devices, lambda device: MLXWorkerClient(
            spec, execution_variant="current_engine", engine_configuration=ENGINE_CONFIGURATIONS[engine],
            cuda_device=device, startup_timeout=DENSE_STARTUP_TIMEOUT, **extra), ordered_service())

    deadline = float(store.settings()["request_timeout_s"] or 120)
    sizes = ({"traces": [H9_TRACES[0]], "rounds": 1, "requests": 3, "max_tokens": (4, 8)} if smoke else
             {"traces": H9_TRACES, "rounds": H9_ROUNDS})
    result = measure_h8(start, model, deadline_s=deadline, on_row=on_row, names=names, engines=engines, plan="h9",
                        stress=None if smoke else trace_stress, **sizes)
    return {"model": model, "revision": revision, "preflight": preflight, "dense_preflight": dense, "knobs": knobs,
            **result}


def analyse_h9(report: dict[str, Any], resamples: int = RESAMPLES, seed: int = 0) -> dict[str, Any]:
    """H9's two parts, as fixed in BACKLOG before data: KP1's rules on `tuned` against `kp`, then H8's on the
    dense plan against the default as it would ship after them."""
    tuned, kp, dense = arms_h9()
    exact = {"tuned", "kp"}
    kp_part = analyse_kp1({**report, "arms": [tuned, kp], "rows": [r for r in report.get("rows", []) if r["engine"] in exact],
                           "stress_rows": [r for r in report.get("stress_rows", []) if r["engine"] in exact],
                           "engines": {e: v for e, v in (report.get("engines") or {}).items() if e in exact}},
                          resamples, seed)
    kp_ships = bool(kp_part.get("valid") and kp_part["(1) KP beats the default"]["passes"]
                    and kp_part["(2) no harm in any state"]["passes"])
    reference = kp if kp_ships else tuned
    if not (report.get("dense_preflight") or {}).get("ok") or not any(r["engine"] == "dense" for r in report.get("rows", [])):
        dense_part: dict[str, Any] = {"valid": False, "problems": ["the dense preflight did not pass or no dense cell ran"]}
        against = {}
    else:
        dense_part = analyse_h8({**report, "arms": [reference, dense]}, resamples, seed)
        against = {}  # descriptive: the plan against both, so the switch of reference can be seen not to decide it
        for other in (tuned, kp):
            part = analyse_h8({**report, "arms": [other, dense]}, resamples, seed)
            if part.get("valid"):
                against[other] = part["(1) dense beats today's default"]
    return {"schema": "ironmule.rl2_headroom.h9.analysis.v1", "kp_ships": kp_ships, "dense_reference": reference,
            "kp": kp_part, "dense": dense_part, "dense_against_each": against,
            "dense_recommended": bool(dense_part.get("valid") and dense_part["(1) dense beats today's default"]["passes"]
                                      and dense_part["(2) no harm in any state"]["passes"])}


# ---------------------------------------------------------------- H10, fixed in BACKLOG before data

H10_TRACES = range(87, 99)      # fresh seeds: no earlier run, training or simulation used them
H10_ROUNDS = 4


def arms_h10() -> list[str]:
    """The default against the dense plan with slot reuse at H9's width and at RL8's."""
    return ["tuned:shortest/w2/g0/sjf", "dense:shortest/w8/g0/sjf", "dense:shortest/w4/g0/sjf"]


def run_h10(model: str, work: Path, *, smoke: bool, on_row: Any = None) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    from ironmule_product.backend import DENSE_STARTUP_TIMEOUT, MLXWorkerClient
    from ironmule_product.device_pool import cuda_devices, start_pool
    from ironmule_product.state import ProductStore

    revision = Path(snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                                             "*.jinja", "tokenizer*"])).name
    state = work / "rl2h-state"
    _ironmule("setup", "--mode", "server", state=state)
    added = _ironmule("models", "add", model, state=state)
    if added.get("returncode") != 0:
        return {"model": model, "revision": revision, "failed": "models add", "added": added}
    store = ProductStore(state)
    spec = store.model(model)
    devices = cuda_devices() or [None]
    preflight = dense_preflight(str(spec.snapshot_path), devices[0])
    if not preflight.get("ok"):
        return {"model": model, "revision": revision, "failed": "dense preflight", "preflight": preflight}

    def start(engine: str) -> Any:  # one startup limit for both engines; startup is not in the cost
        return start_pool(store, spec, devices, lambda device: MLXWorkerClient(
            spec, execution_variant="current_engine", engine_configuration=ENGINE_CONFIGURATIONS[engine],
            cuda_device=device, startup_timeout=DENSE_STARTUP_TIMEOUT), ordered_service())

    deadline = float(store.settings()["request_timeout_s"] or 120)
    sizes = ({"traces": [H10_TRACES[0]], "rounds": 1, "requests": 3, "max_tokens": (4, 8)} if smoke else
             {"traces": H10_TRACES, "rounds": H10_ROUNDS})
    result = measure_h8(start, model, deadline_s=deadline, on_row=on_row, names=arms_h10(), engines=("tuned", "dense"),
                        plan="h10", **sizes)
    return {"model": model, "revision": revision, "preflight": preflight, **result}


def analyse_h10(report: dict[str, Any], resamples: int = RESAMPLES, seed: int = 0) -> dict[str, Any]:
    """H10, as fixed in BACKLOG before data: H8's decisions for the better dense arm, slot reuse as a validity
    condition, and RL8's rule for the plan's width."""
    part = analyse_h8({**report, "arms": arms_h10()}, resamples, seed)
    joined = {lv: sum(row.get("joined") or 0 for row in report.get("rows", [])
                      if row["engine"] == "dense" and row["level"] == lv)
              for lv in range(len(report.get("states") or H2_STATES))}
    problems = list(part.get("problems", []))
    if not (report.get("preflight") or {}).get("ok"):
        problems.append("the dense preflight did not pass")
    if joined.get(1, 0) + joined.get(2, 0) == 0:
        problems.append("no dense cell of `bursts` or `spike` reported a joined request: slot reuse did not engage")
    out = {**part, "schema": "ironmule.rl2_headroom.h10.analysis.v1", "joined_per_state": joined}
    if problems:
        return {**out, "valid": False, "problems": problems}
    w8, w4 = arms_h10()[1:]
    width = part["sjf_within_dense"]  # analyse_h8 pairs its second dense arm against its first: w4 against w8
    return {**out, "w4_over_w8": width, "recommended_width": "w4" if width["gain_ci95"][0] > part["aa_noise"] else "w8",
            "recommended": bool(part["(1) dense beats today's default"]["passes"]
                                and part["(2) no harm in any state"]["passes"])}


H11_TRACES = range(99, 107)     # fresh seeds: no earlier run, training or simulation used them
H11_ROUNDS = 2
H11_CONTEXTS = (2, 4, 7)        # the background notes this often: about 2,050, 4,060 and 7,090 prompt tokens
H11_LONG_EVERY = 8              # one request in eight, at least one per trace, sits behind a long context


def arms_h11() -> list[str]:
    """The default against the dense plan with slots that hold the product's whole context (H11)."""
    return ["tuned:shortest/w2/g0/sjf", "dense:shortest/w8/g0/sjf"]


def trace_h11(state: int, index: int, capacity: float, requests: int = REQUESTS,
              max_tokens: tuple[int, ...] | None = None) -> list[dict[str, Any]]:
    """H10's traffic for one state, with one request in eight (at least one) behind a long context."""
    items = trace_h2(state, index, capacity, requests, max_tokens)
    rng = random.Random(3000 * state + index)
    long = set(rng.sample(range(len(items)), max(1, round(len(items) / H11_LONG_EVERY))))
    return [{**item, "context": rng.choice(H11_CONTEXTS)} if i in long else item for i, item in enumerate(items)]


def run_h11(model: str, work: Path, *, smoke: bool, on_row: Any = None) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    from ironmule_product.backend import DENSE_STARTUP_TIMEOUT, MLXWorkerClient
    from ironmule_product.device_pool import cuda_devices, start_pool
    from ironmule_product.state import ProductStore

    revision = Path(snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                                             "*.jinja", "tokenizer*"])).name
    state = work / "rl2h-state"
    _ironmule("setup", "--mode", "server", state=state)
    added = _ironmule("models", "add", model, state=state)
    if added.get("returncode") != 0:
        return {"model": model, "revision": revision, "failed": "models add", "added": added}
    store = ProductStore(state)
    spec = store.model(model)
    devices = cuda_devices() or [None]
    preflight = dense_preflight(str(spec.snapshot_path), devices[0])
    if not preflight.get("ok"):
        return {"model": model, "revision": revision, "failed": "dense preflight", "preflight": preflight}

    def start(engine: str) -> Any:  # one startup limit for both engines; startup is not in the cost
        return start_pool(store, spec, devices, lambda device: MLXWorkerClient(
            spec, execution_variant="current_engine", engine_configuration=ENGINE_CONFIGURATIONS[engine],
            cuda_device=device, startup_timeout=DENSE_STARTUP_TIMEOUT), ordered_service())

    deadline = float(store.settings()["request_timeout_s"] or 120)
    sizes = ({"traces": [H11_TRACES[0]], "rounds": 1, "requests": 3, "max_tokens": (4, 8)} if smoke else
             {"traces": H11_TRACES, "rounds": H11_ROUNDS})
    result = measure_h8(start, model, deadline_s=deadline, on_row=on_row, names=arms_h11(), engines=("tuned", "dense"),
                        plan="h11", traffic=trace_h11,
                        warm_extra=[{"offset_s": 0.0, "prompt": 0, "context": max(H11_CONTEXTS), "max_tokens": 8}],
                        **sizes)
    return {"model": model, "revision": revision, "preflight": preflight, **result}


def analyse_h11(report: dict[str, Any], resamples: int = RESAMPLES, seed: int = 0) -> dict[str, Any]:
    """H11 (b), as fixed in BACKLOG before data (amended before data after review): H8's decisions on traffic
    with long prompts. Valid only when both engines served long requests with no failed or missing outcome (a
    timeout is a cost, a rejection for length a broken fit rule) and the dense workers' slots held the whole
    context. Short and long requests are reported apart per engine and state, and so is the offered work."""
    part = analyse_h8({**report, "arms": arms_h11()}, resamples, seed)
    problems = list(part.get("problems", []))
    deadline = float(report.get("deadline_s") or 120)
    if not (report.get("preflight") or {}).get("ok"):
        problems.append("the dense preflight did not pass")
    slots = sorted({(worker.get("dense_plan") or {}).get("capacity") for entry in report.get("engines", {}).get("dense", [])
                    for worker in entry.get("engine") or []}, key=str)
    if slots != [8192]:
        problems.append(f"the dense workers' slots were {slots}, not 8,192 tokens")
    outcomes: dict[str, dict[str, int]] = {}
    split: dict[str, dict[str, dict[str, Any]]] = {}
    offered: dict[str, dict[str, int]] = {}
    for row in report.get("rows", []):
        for request in row["requests"]:
            kind = "long" if request.get("context") else "short"
            if kind == "long":
                counts = outcomes.setdefault(row["engine"], {})
                key = str(request.get("error") or request.get("outcome") or "no outcome")
                counts[key] = counts.get(key, 0) + 1
            if row.get("run") != 0:
                continue
            cell = split.setdefault(row["engine"], {}).setdefault(str(row["level"]), {}).setdefault(
                kind, {"requests": 0, "cost": 0.0, "first_s": [], "timeouts": 0})
            cell["requests"] += 1
            cell["cost"] += request_cost(request, deadline)
            cell["timeouts"] += request.get("outcome") == "timeout"
            if request.get("first_s") is not None:
                cell["first_s"].append(request["first_s"])
            if row["engine"] == "tuned" and request.get("outcome") == "ok":
                work = offered.setdefault(str(row["level"]), {"prompt_tokens": 0, "completion_tokens": 0})
                work["prompt_tokens"] += int(request.get("prompt_tokens") or 0)
                work["completion_tokens"] += int(request.get("tokens") or 0)
    for engine in split.values():
        for state in engine.values():
            for cell in state.values():
                cell["mean_cost"] = cell.pop("cost") / max(1, cell["requests"])
                firsts = cell.pop("first_s")
                cell["mean_first_s"] = statistics.fmean(firsts) if firsts else None
    if set(outcomes) != {"tuned", "dense"} or any(
            sum(counts.values()) == 0 or set(counts) - set(COSTED_OUTCOMES) for counts in outcomes.values()):
        problems.append(f"long requests did not all reach an outcome in both engines: {outcomes}")
    joined = sum(row.get("joined") or 0 for row in report.get("rows", []) if row["engine"] == "dense")
    out = {**part, "schema": "ironmule.rl2_headroom.h11.analysis.v1", "dense_slots": slots,
           "long_outcomes": outcomes, "short_and_long": split, "offered_work_per_state": offered,
           "dense_joined": joined}
    if problems:
        return {**out, "valid": False, "problems": problems}
    return {**out, "recommended_up_to_context_limit": bool(part["(1) dense beats today's default"]["passes"]
                                                          and part["(2) no harm in any state"]["passes"])}

REPOSITORY = "https://github.com/Tobayko/IronMule"
# ---------------------------------------------------------------- AG1-V: the agent learning online on hardware

AG1_LEARN_TRACES = range(107, 143)  # 36 fresh seeds for the online phase: no earlier run, training or simulation used them
AG1_EVAL_TRACES = range(143, 151)   # 8 fresh seeds per state for the held-out comparison
AG1_PAUSE_S = 120.0                 # idle after each learning trace, in which the agent plans, explores and replays
AG1_ARMS = ("reference", "agent")
# V on a T4 (the default), and V-CPU: the same harness on a CPU, fixed in BACKLOG AG1 before its data
AG1_PROFILES = {
    "t4": {"model": MODEL, "variant": "current_engine", "compute_dtype": None, "max_tokens": None,
           "learn": AG1_LEARN_TRACES, "evaluate": AG1_EVAL_TRACES, "pause_s": AG1_PAUSE_S, "timeout_s": None,
           "short_prompts": False},
    "cpu": {"model": "mlx-community/SmolLM2-135M-Instruct-8bit", "variant": "reference", "compute_dtype": "dequantize",
            "max_tokens": (4, 32), "learn": range(107, 122), "evaluate": range(143, 147), "pause_s": 360.0,
            "timeout_s": 1800, "short_prompts": True},
}
# BACKLOG AG1, V-CPU-2: V-CPU with 40 online traces on fresh seeds, so the sequential test can reach its later looks
AG1_PROFILES["cpu2"] = {**AG1_PROFILES["cpu"], "learn": range(151, 191), "evaluate": range(191, 195)}
# BACKLOG AG1, V2: V with 64-request traces (four replay windows each) and time to replay them, on fresh seeds
# and three tiers on the same held-out traces: the default (`w8/fifo`), the best fixed schedule earlier runs found
# (RL6's `w2/sjf`) and the agent, measured even when the agent qualified nothing (then it serves the default)
AG1_PROFILES["t4b"] = {**AG1_PROFILES["t4"], "requests": 64, "learn": range(195, 219), "evaluate": range(219, 223),
                       "pause_s": 480.0, "fixed": (2, True)}
# BACKLOG AG1, V3: V2 with a 900 s deadline so a 64-request spike completes, 1,200 s idle so a trace's four windows
# replay, at most 16 online traces ending after the one in which the agent qualified, and 3 held-out traces a state
AG1_PROFILES["t4c"] = {**AG1_PROFILES["t4b"], "learn": range(223, 239), "evaluate": range(239, 242),
                       "pause_s": 1200.0, "timeout_s": 900, "stop_on_qualify": True}
# BACKLOG AG1, V4: the agent with busy-period windows (V3's replay judge scored a burst's first 16 arrivals); the
# fixed tier is `w8/sjf`, the best fixed schedule on whole traces in V3's world model; sized to the GPU quota left
AG1_PROFILES["t4d"] = {**AG1_PROFILES["t4c"], "learn": range(242, 254), "evaluate": range(254, 256),
                       "pause_s": 800.0, "fixed": (8, True)}
# BACKLOG AG1, V5-CPU: V-CPU with V4's agent (busy-period windows) on a Kaggle CPU session, which spends no GPU
# quota; at most 30 online traces ending after the one in which the agent qualified, the three tiers with the fixed
# `w1/sjf` (capacity 1: the order is the only decision), 4 held-out traces a state, fresh seeds
AG1_PROFILES["cpu3"] = {**AG1_PROFILES["cpu"], "learn": range(256, 286), "evaluate": range(286, 290),
                        "stop_on_qualify": True, "fixed": (1, True), "accelerator": "cpu"}
# BACKLOG AG2: real instruction prompts (Dolly 15k, fetched at run time) with one output limit for all, so answers end
# at EOS and "shortest limit first" is arrival order; the agent may order by its learned answer-length prediction.
# The fixed tier is the best limit-ordered schedule its action space holds at capacity 1, `w1/sjf`.
DOLLY_URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
AG2_PROMPT_CHARS = 600
AG1_PROFILES["cpu-lp"] = {**AG1_PROFILES["cpu3"], "max_tokens": (128,), "prompts": "dolly", "learn": range(290, 320),
                          "evaluate": range(320, 324), "short_prompts": False}
# BACKLOG AG2, the T4 gate: the owner's Gemma 3 12B under the owner's `native` plan (the reference worker, capacity 1),
# on both T4s of the session with an agent per card (owner request: multi-GPU), Dolly prompts with one limit of 512;
# first as a smoke that fixes the full run's sizes
AG1_PROFILES["t4-g12-lp"] = {**AG1_PROFILES["t4"], "model": "mlx-community/gemma-3-12b-it-4bit", "variant": "reference",
                             "compute_dtype": "native", "max_tokens": (512,), "prompts": "dolly",
                             "learn": range(324, 354), "evaluate": range(354, 358), "timeout_s": 1800,
                             "stop_on_qualify": True, "fixed": (1, True), "cards": 2}
# BACKLOG AG2, V-G12: the full run the gate sized (limit 256 halves the time per answer and keeps 40-256 tokens of
# spread; 32 requests a trace give each card about 16, one replay window a trace; 360 s idle fits its triplet)
AG1_PROFILES["t4-g12-v"] = {**AG1_PROFILES["t4-g12-lp"], "max_tokens": (256,), "requests": 32,
                            "learn": range(358, 398), "evaluate": range(398, 400), "pause_s": 360.0}


def dolly_prompts(work: Path) -> tuple[list[str], str]:
    """AG2: Dolly 15k's instructions (with their context, cut at `AG2_PROMPT_CHARS`), shuffled by a fixed seed, and the
    file's digest. Online traces draw from the first 12,000, held-out traces from the rest, calibration from the end."""
    import urllib.request

    path = work / "dolly.jsonl"
    if not path.exists():
        urllib.request.urlretrieve(DOLLY_URL, path)
    data = path.read_bytes()
    prompts = []
    for line in data.decode().splitlines():
        if line.strip():
            row = json.loads(line)
            text = (row["instruction"] + ("\n" + row["context"] if row.get("context") else "")).strip()
            prompts.append(text[:AG2_PROMPT_CHARS])
    random.Random(15).shuffle(prompts)
    return prompts, hashlib.sha256(data).hexdigest()


def lp_items(items: list[dict[str, Any]], prompts: list[str], trace_index: int, held_out: bool,
             limit: int) -> list[dict[str, Any]]:
    """AG2: a trace's arrivals with real prompts and one output limit; the prompt pool is split so held-out traces
    never reuse an online prompt."""
    pool = prompts[12_000:-64] if held_out else prompts[:12_000]
    rng = random.Random(7_000 + trace_index)
    return [{**item, "long": False, "max_tokens": limit, "text": pool[rng.randrange(len(pool))]} for item in items]


class _PredictingDispatch:
    """AG2's held-out arm: the frozen policy deciding through the LD1 hooks, ordering by the frozen length model."""

    def __init__(self, dispatch: Any, lengths: Any):
        self._dispatch, self._lengths = dispatch, lengths

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dispatch, name)

    def order_key(self, request: Any) -> float | None:
        from ironmule_product.length_model import features

        policy = self._dispatch.policy
        if policy is None or policy.key != "predicted":
            return None
        return min(float(request.max_tokens), self._lengths.predict(features(request), int(request.max_tokens)))


def run_ag1(model: str, work: Path, *, smoke: bool, on_row: Any = None, profile: str = "t4") -> dict[str, Any]:
    """BACKLOG AG1, V: one worker on the first GPU, the product's own service, the agent attached as `--agent on`
    attaches it. The agent learns online from 36 traces of RL2-H2's states with a pause after each; then its serving
    policy is frozen and compared with the reference schedule on held-out traces (reference, agent, reference per
    block), the agent's policy deciding through the product's LearnedDispatch hooks (a measurement needs no
    evidence)."""
    from huggingface_hub import snapshot_download

    from ironmule_product.backend import MLXWorkerClient
    from ironmule_product.cli import _agent
    from ironmule_product.device_pool import cuda_devices, start_pool
    from ironmule_product.learned_dispatch import LearnedDispatch
    from ironmule_product.serving_policy import ServingPolicy
    from ironmule_product.service import ProductService
    from ironmule_product.state import ProductStore

    setup = AG1_PROFILES[profile]
    revision = Path(snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                                             "*.jinja", "tokenizer*"])).name
    state = work / "rl2h-state"
    _ironmule("setup", "--mode", "server", state=state)
    added = _ironmule("models", "add", model, state=state)
    if added.get("returncode") != 0:
        return {"model": model, "revision": revision, "failed": "models add", "added": added}
    store = ProductStore(state)
    if smoke or setup["timeout_s"]:  # on a CPU the server mode's 120 s cuts answers off
        store.set_request_timeout(setup["timeout_s"] or 1800)
    spec = store.model(model)
    devices = (cuda_devices() or [None])[:int(setup.get("cards", 1))]  # AG2: an agent per card
    pool = start_pool(store, spec, devices, lambda device: MLXWorkerClient(
        spec, execution_variant=setup["variant"], compute_dtype=setup["compute_dtype"], cuda_device=device),
        ProductService)
    shards = list(pool.shards)
    learn, evaluate = (([setup["learn"][0]] * 3, [setup["evaluate"][0]]) if smoke
                       else (list(setup["learn"]), list(setup["evaluate"])))
    pause = 1.0 if smoke else setup["pause_s"]
    sizes = ({"requests": 3, "max_tokens": (4, 8)} if smoke else
             {key: setup[key] for key in ("max_tokens", "requests") if setup.get(key)})
    report: dict[str, Any] = {"plan": "ag1", "profile": profile, "arms": list(AG1_ARMS), "states": list(H2_STATES),
                              "learning": [], "rows": [], "devices": list(devices)}
    try:
        deadline = float(store.settings()["request_timeout_s"] or 120)
        report["deadline_s"] = deadline
        warm = replay(pool, model, [{"offset_s": 0.0, "prompt": i % len(PROMPTS), "long": i % 2 == 1, "max_tokens": 8}
                                    for i in range(8)], deadline)
        report["warmup_ok"] = sum(w.get("outcome") == "ok" for w in warm)
        prompts = None
        if setup.get("prompts") == "dolly":  # AG2
            prompts, report["prompts_sha256"] = dolly_prompts(work)
            report["prompts"] = {"source": DOLLY_URL, "count": len(prompts), "limit": setup["max_tokens"][0]}
        calibration_items = None if prompts is None else [
            {"offset_s": 0.0, "prompt": 0, "long": False, "max_tokens": setup["max_tokens"][0], "text": text}
            for text in prompts[-64:][:16 * len(pool.shards)]]
        report["calibration"] = calibrate(pool, model, deadline, sizes.get("max_tokens"), calibration_items)
        if report["calibration"].get("failed"):
            return {"model": model, "revision": revision, **report}
        capacity = report["calibration"]["capacity_rps"]

        def traffic(level: int, t: int) -> list[dict[str, Any]]:
            items = trace_h2(level, t, capacity, sizes.get("requests", REQUESTS), sizes.get("max_tokens"))
            if prompts is not None:  # AG2: real prompts, one limit, answers ending at EOS
                return lp_items(items, prompts, t, t in setup["evaluate"], setup["max_tokens"][0])
            # V-CPU: a 1,030-token prompt takes tens of seconds to prefill on a slow CPU (amended before data)
            return [{**item, "long": False} for item in items] if setup["short_prompts"] else items

        reference = ServingPolicy(max_width=8, gather_ms=0, source="ag1-reference")  # the default for this setup
        agents = []
        for each, device in zip(shards, devices):
            each.policy = reference  # before the agent is built: its binding names the schedule it is measured against
            agents.append(_agent(store, each, spec, card=device if len(shards) > 1 else None))
            each.agent = each.dispatch = agents[-1]
        from ironmule_product.agent import share_lengths

        share_lengths(agents)  # AG2: answer lengths depend on the model, not the card
        agent = agents[0]
        started = time.monotonic()
        for k, t in enumerate(learn):  # the online phase: the agent decides, learns and qualifies by itself
            level = k % len(H2_STATES)
            items = traffic(level, t)
            row = {"level": level, "trace": t, "started_s": time.monotonic() - started,
                   "serving": agent.status()["serving"], "requests": replay(pool, model, items, deadline)}
            wait_idle(pool)
            time.sleep(pause)
            status = agent.status()
            row["agent"] = {key: status.get(key) for key in ("serving", "gain", "gain_ci95", "world_model",
                                                           "candidate", "lengths", "decisions", "decision_us",
                                                           "counters", "planner")}
            if len(agents) > 1:  # AG2: every card's agent, the first one's also under "agent" as before
                row["agents"] = [{key: s.get(key) for key in ("serving", "gain", "candidate", "lengths", "counters")}
                                 for s in (a.status() for a in agents)]
            report["learning"].append(row)
            if on_row is not None:
                on_row(report)
            if setup.get("stop_on_qualify") and all(a.policy is not None for a in agents):  # V3: held-out needs nothing more
                report["online_ended_after"] = t
                break
        from ironmule_product.length_model import LengthModel

        finals = [{**a.status(), "history": list(a.history), "evidence": a.evidence, "last_plan": a.last_plan}
                  for a in agents]
        policies = [a.policy for a in agents]
        lengths_by_card = [LengthModel.from_json(a.lengths.to_json()) for a in agents]  # AG2: frozen with the policy
        policy = policies[0]
        report["agent_final"] = finals[0]
        if len(agents) > 1:
            report["agents_final"] = finals
        for each in shards:
            each.agent = each.dispatch = None  # detach first: the scheduler starts no new idle work for it
        until = time.monotonic() + 600
        while any(getattr(each, "_replay_state", None) is not None for each in shards) and time.monotonic() < until:
            time.sleep(0.1)  # a replay window still running ends before the comparison starts
        for a in agents:
            a.close()
        if policy is None and smoke:  # a smoke still walks the comparison, with a fixed schedule
            from ironmule_product.agent import fixed_policy
            from ironmule_product.learned_dispatch import Policy as DispatchPolicy

            policy = DispatchPolicy.from_json(fixed_policy(2, True))
            policies = [policy] * len(agents)
        tiers = setup.get("fixed") is not None
        if len(agents) > 1 and policy is None and any(p is not None for p in policies):
            policy = next(p for p in policies if p is not None)  # a card that qualified something counts
        if policy is None and not tiers:
            report["frozen"] = None
            report["stopped"] = "the agent qualified no schedule in the online phase; nothing to compare"
            return {"model": model, "revision": revision, **report}
        report["frozen"] = None if policy is None else {"policy_sha256": policy.sha256, "evidence": agent.evidence}
        if len(agents) > 1 and policy is not None:
            report["frozen"]["cards"] = [None if p is None else {"policy_sha256": p.sha256, "evidence": a.evidence}
                                         for p, a in zip(policies, agents)]
        if policy is None:  # V2: the tiers are still measured; the agent serves the default, as it did online
            report["stopped"] = "the agent qualified no schedule in the online phase; its tier is the default's"
        def frozen_dispatch(p: Any, frozen_lengths: Any) -> Any:
            if p is None:
                return None
            made = LearnedDispatch(p, {"experiment": "AG1-V measurement"}, None)
            return _PredictingDispatch(made, frozen_lengths) if p.key == "predicted" else made
        dispatches = [frozen_dispatch(p, ln) for p, ln in zip(policies, lengths_by_card)]
        dispatch = dispatches[0]
        arms = list(AG1_ARMS)
        if tiers:
            from ironmule_product.agent import fixed_policy
            from ironmule_product.learned_dispatch import Policy as DispatchPolicy

            fixed = LearnedDispatch(DispatchPolicy.from_json(fixed_policy(*setup["fixed"])),
                                    {"experiment": "AG1-V2 fixed tier"}, None)
            arms = ["reference", "fixed", "agent"]
            report["arms"] = arms
        block = 0
        for t in evaluate:  # the held-out comparison, LD1-V's block order: reference, agent, reference
            for level in range(len(H2_STATES)):
                items = traffic(level, t)
                for name, run in block_order(arms, block):
                    if not wait_idle(pool):
                        report["stopped"] = f"the pool did not drain before block {block}"
                        return {"model": model, "revision": revision, **report}
                    for each, own in zip(shards, dispatches):
                        each.policy = reference
                        each.dispatch = own if name == "agent" else fixed if name == "fixed" else None
                    report["rows"].append({"level": level, "trace": t, "arm": name, "run": run, "block": block,
                                           "requests": replay(pool, model, items, deadline)})
                    if on_row is not None:
                        on_row(report)
                block += 1
        for each in shards:
            each.dispatch = None
        report["dispatch_status"] = dispatch.status() if dispatch is not None else None
    finally:
        pool.close()
    return {"model": model, "revision": revision, **report}


def analyse_ag1(report: dict[str, Any], resamples: int = RESAMPLES, seed: int = 0) -> dict[str, Any]:
    """AG1, V, as fixed in BACKLOG before data. Validity: every held-out request answered with the same tokens in all
    three cells of its block and none failed. Noise: the nearest-rank 95th percentile of the A/A pairs (the
    reference's two runs). (1) The agent beats the reference: the gain over the reference (the mean of its two
    runs), paired by trace with the three states weighted equally, has a 2.5 % bound above zero and above the A/A
    floor (KP1's rule). (2) No harm: each state's bound is above minus the noise."""
    deadline = float(report.get("deadline_s") or 120)
    rng = random.Random(seed)
    cells: dict[tuple[int, int, str, int], list[dict[str, Any]]] = {}
    for row in report.get("rows", []):
        cells[row["level"], row["trace"], row["arm"], row["run"]] = row["requests"]
    blocks = sorted({(lv, t) for lv, t, _, _ in cells})
    problems, mismatched = [], 0
    complete = []
    for lv, t in blocks:
        trio = [cells.get((lv, t, "reference", 0)), cells.get((lv, t, "agent", 0)), cells.get((lv, t, "reference", 1))]
        if "fixed" in (report.get("arms") or ()):  # V2's tier between the default and the agent
            trio.append(cells.get((lv, t, "fixed", 0)))
        if any(c is None for c in trio):
            problems.append(f"block {lv}/{t} is incomplete")
            continue
        if any(r.get("outcome") != "ok" for c in trio for r in c):
            problems.append(f"block {lv}/{t} has a request that did not complete")
        for requests in zip(*trio):
            if len({r.get("digest") for r in requests}) != 1:
                mismatched += 1
        complete.append((lv, t))

    def cost(lv: int, t: int, arm: str, run: int) -> float:
        return statistics.fmean(request_cost(r, deadline) for r in cells[lv, t, arm, run])

    def reference(lv: int, t: int) -> float:
        return (cost(lv, t, "reference", 0) + cost(lv, t, "reference", 1)) / 2

    def paired(pairs_of: Any, traces: list[int]) -> dict[str, Any]:
        pairs = [pairs_of(t) for t in traces]
        if not pairs:
            return {"gain": None, "gain_ci95": [None, None]}
        boot = []
        for _ in range(resamples):
            sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
            boot.append(1 - sum(x for x, _ in sample) / sum(y for _, y in sample))
        return {"gain": 1 - sum(x for x, _ in pairs) / sum(y for _, y in pairs),
                "gain_ci95": [_quantile(boot, 0.025), _quantile(boot, 0.975)]}

    levels = sorted({lv for lv, _ in complete})
    traces = sorted({t for lv, t in complete if all((lv2, t) in complete for lv2 in levels)})
    weighted = lambda arm_cost, t: sum(arm_cost(lv, t) for lv in levels) / len(levels)  # noqa: E731
    noise = 0.0
    aa = [abs(1 - cost(lv, t, "reference", 1) / cost(lv, t, "reference", 0)) for lv, t in complete]
    if aa:
        noise = sorted(aa)[max(0, math.ceil(0.95 * len(aa)) - 1)]
    control = paired(lambda t: (weighted(lambda lv, t2: cost(lv, t2, "reference", 1), t),
                                weighted(lambda lv, t2: cost(lv, t2, "reference", 0), t)), traces)
    floor = abs(control["gain"]) if control["gain"] is not None else 0.0
    pooled = paired(lambda t: (weighted(lambda lv, t2: cost(lv, t2, "agent", 0), t), weighted(reference, t)), traces)
    per_state = {str(lv): paired(lambda t, lv=lv: (cost(lv, t, "agent", 0), reference(lv, t)),
                                 sorted({t for lv2, t in complete if lv2 == lv})) for lv in levels}
    valid = not problems and mismatched == 0 and len(traces) >= 2
    passes = valid and pooled["gain"] is not None and pooled["gain_ci95"][0] > max(0.0, floor)
    no_harm = valid and all(v["gain_ci95"][0] is not None and v["gain_ci95"][0] > -noise for v in per_state.values())
    learning = report.get("learning", [])
    return {"schema": "ironmule.rl2_headroom.ag1.analysis.v1", "valid": valid, "problems": problems,
            "mismatched_requests": mismatched, "complete_traces": traces, "aa_noise": noise, "aa_floor": floor,
            "aa_control": control, "(1) the agent beats the reference": {**pooled, "passes": passes},
            "(2) no harm in any state": {"per_state": per_state, "passes": no_harm},
            "frozen": report.get("frozen"), "stopped": report.get("stopped"),
            **({"tiers": {
                "fixed over the default": paired(lambda t: (weighted(lambda lv, t2: cost(lv, t2, "fixed", 0), t),
                                                            weighted(reference, t)), traces),
                "agent over the default": pooled,
                "agent over fixed": paired(lambda t: (weighted(lambda lv, t2: cost(lv, t2, "agent", 0), t),
                                                      weighted(lambda lv, t2: cost(lv, t2, "fixed", 0), t)), traces)}}
               if "fixed" in (report.get("arms") or ()) and traces else {}),
            "online": {"traces": len(learning),
                       "serving_by_trace": [row.get("agent", {}).get("serving") for row in learning],
                       "first_learned_trace": next((k for k, row in enumerate(learning)
                                                    if str(row.get("agent", {}).get("serving", "")).startswith("learned")),
                                                   None),
                       "counters": (report.get("agent_final") or {}).get("counters"),
                       "world_model": (report.get("agent_final") or {}).get("world_model"),
                       "decision_us": (report.get("agent_final") or {}).get("decision_us")},
            "verdict": ("invalid" if report.get("rows") and not valid else
                        "nothing qualified online" if report.get("frozen") is None else
                        "the agent beats the reference" if passes and no_harm else
                        "the agent does not beat the reference")}


# ---------------------------------------------------------------- AG4-R: the agent's two paths, routed (BACKLOG AG4)

AG4R_ARMS = ("single", "router", "stacked", "frozen")  # the single path forced, the agent's router, the stacked
# engine forced, and the same router frozen at its startup prices (added 2026-11-05, before any AG4-R data)
AG4R_LOADS = (0.8, 3.0)                       # arrival rate over the single path's calibrated capacity
AG4R_TRACES = range(400, 406)                 # fresh seeds: 6 traces per state and load
AG4R_LIMIT = 256


class ForcedRoute:
    """A router that always picks one path: the single path alone, or the stacked engine alone, on the same two
    loaded workers as the agent's router."""

    def __init__(self, index: int):
        self.index = index

    def choose(self, outstanding: Any, max_tokens: int) -> int:
        return self.index

    def watch(self, index: int, sharing: int, events: Any) -> Any:
        return events

    def status(self) -> dict[str, Any]:
        return {"forced": self.index}


class FrozenRouter:
    """The agent's router with the prices it starts from, never learning: the same decisions as the learning router
    would make before its first finished request. Against it, the learning router shows what learning while serving
    adds on the card (AG4-R's learning comparison)."""

    def __init__(self, router: Any):
        self.inner = type(router)(router.names, [copy.deepcopy(cost) for cost in router.costs])

    def choose(self, outstanding: Any, max_tokens: int) -> int:
        return self.inner.choose(outstanding, max_tokens)

    def watch(self, index: int, sharing: int, events: Any) -> Any:
        return events

    def status(self) -> dict[str, Any]:
        return {**self.inner.status(), "frozen": True}


def measure_ag4r(pool: Any, model: str, router: Any, prompts: list[str] | None, deadline_s: float, *,
                 loads: tuple[float, ...] = AG4R_LOADS, traces: Any = AG4R_TRACES, requests: int = 32,
                 limit: int = AG4R_LIMIT, on_row: Any = None) -> dict[str, Any]:
    """Every block (state, load, trace) runs the single path, the router, the stacked engine and the frozen router
    (rotated) and the single path again, on the same arrivals; the router keeps learning across blocks, the frozen
    one keeps the prices the router started from."""
    def items_for(level: int, load: float, t: int, capacity: float) -> list[dict[str, Any]]:
        items = trace_h2(level, t, load * capacity, requests, (limit,))
        return lp_items(items, prompts, t, True, limit) if prompts else items

    frozen = FrozenRouter(router)  # before the router has seen a finished request
    pool.router = ForcedRoute(0)
    calibration_items = None if not prompts else [
        {"offset_s": 0.0, "prompt": 0, "long": False, "max_tokens": limit, "text": text} for text in prompts[-16:]]
    burst = [{**item, "offset_s": 0.0} for item in (calibration_items or trace(9, 0, 1.0, 16, (limit,)))]
    started = time.monotonic()
    done = replay(pool, model, burst, deadline_s)
    makespan = max(r["end_s"] or 0.0 for r in done) or (time.monotonic() - started)
    capacity = sum(r["outcome"] == "ok" for r in done) / makespan
    report: dict[str, Any] = {"plan": "ag4r", "arms": list(AG4R_ARMS), "states": list(H2_STATES), "loads": list(loads),
                              "calibration": {"capacity_rps": capacity, "makespan_s": makespan,
                                              "ok": sum(r["outcome"] == "ok" for r in done)},
                              "deadline_s": deadline_s, "rows": []}
    routes = {"single": ForcedRoute(0), "stacked": ForcedRoute(1), "router": router, "frozen": frozen}
    block = 0
    for load in loads:
        for level in range(len(H2_STATES)):
            for t in traces:
                items = items_for(level, load, t, capacity)
                others = list(AG4R_ARMS[1:])
                turn = block % len(others)
                order = [("single", 0), *((name, 0) for name in others[turn:] + others[:turn]), ("single", 1)]
                for arm, run in order:
                    pool.router = routes[arm]
                    wait_idle(pool)
                    requests_out = replay(pool, model, items, deadline_s)
                    for item, row in zip(items, requests_out):
                        row["offset_s"] = item["offset_s"]
                    report["rows"].append({"level": level, "load": load, "trace": t, "arm": arm, "run": run,
                                           "block": block, "requests": requests_out})
                    if on_row is not None:
                        on_row(report)
                block += 1
    pool.router = router
    report["router"] = router.status()
    report["frozen"] = frozen.status()
    return report


def _cell(requests: list[dict[str, Any]], censor: dict[str, float] | None = None) -> dict[str, float] | None:
    """Tokens per second over the span and first-token times of one cell; None if a request failed. With `censor`
    (AG4-C run 3, the default path only) a failed request counts as if it had produced its whole token limit, first
    token and last, at the deadline: never later than it really would have, so it can only flatter that path."""
    if censor is not None:
        requests = [r if r.get("outcome") == "ok" and r.get("first_s") is not None else
                    {**r, "outcome": "ok", "first_s": censor["deadline_s"], "end_s": censor["deadline_s"],
                     "tokens": censor["limit"]} for r in requests]
    ok = [r for r in requests if r.get("outcome") == "ok" and r.get("first_s") is not None]
    if len(ok) != len(requests):
        return None
    span = max(r["offset_s"] + r["end_s"] for r in ok) - min(r["offset_s"] for r in ok)
    first = sorted(r["first_s"] for r in ok)
    return {"tokens_per_s": sum(r["tokens"] for r in ok) / span, "ttft_mean_s": statistics.fmean(first),
            "ttft_p95_s": first[max(0, math.ceil(0.95 * len(first)) - 1)]}


def analyse_ag4r(report: dict[str, Any], resamples: int = RESAMPLES, seed: int = 0) -> dict[str, Any]:
    """AG4-R as fixed in BACKLOG before data. Pass: the router's tokens per second over the single path's (the mean
    of its two runs), paired by block, has a 2.5 % bound above 0; and its mean and 95th-percentile first token, pooled,
    are no worse than the single path's beyond the A/A noise (the single path's second run against its first)."""
    rng = random.Random(seed)
    cells: dict[tuple[int, str, int], dict[str, float] | None] = {}
    for row in report.get("rows", []):
        cells[row["block"], row["arm"], row["run"]] = _cell(row["requests"])
    blocks = sorted({b for b, _, _ in cells})
    arms = report.get("arms") or AG4R_ARMS[:3]  # a report from before the frozen arm has three
    complete = [b for b in blocks if cells.get((b, "single", 1)) and all(cells.get((b, arm, 0)) for arm in arms)]
    problems = [f"block {b} is incomplete or has a failed request" for b in blocks if b not in complete]
    out: dict[str, Any] = {"schema": "ironmule.rl2_headroom.ag4r.analysis.v1", "complete_blocks": len(complete),
                           "problems": problems, "router": report.get("router")}
    if len(complete) < 2:
        return {**out, "valid": False, "verdict": "invalid: fewer than two complete blocks"}

    def single(b: int, key: str) -> float:
        return (cells[b, "single", 0][key] + cells[b, "single", 1][key]) / 2

    def ratio(arm: str, key: str, blocks_: list[int], base: str = "single") -> dict[str, Any]:
        pairs = [(cells[b, arm, 0][key], single(b, key) if base == "single" else cells[b, base, 0][key])
                 for b in blocks_]
        boot = []
        for _ in range(resamples):
            sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
            boot.append(sum(x for x, _ in sample) / sum(y for _, y in sample))
        return {"ratio": sum(x for x, _ in pairs) / sum(y for _, y in pairs),
                "ci95": [_quantile(boot, 0.025), _quantile(boot, 0.975)]}

    aa = {key: sorted(abs(cells[b, "single", 1][key] / cells[b, "single", 0][key] - 1) for b in complete)
          for key in ("tokens_per_s", "ttft_mean_s", "ttft_p95_s")}
    noise = {key: values[max(0, math.ceil(0.95 * len(values)) - 1)] for key, values in aa.items()}
    tps = ratio("router", "tokens_per_s", complete)
    ttft = {key: ratio("router", key, complete) for key in ("ttft_mean_s", "ttft_p95_s")}
    gain = tps["ci95"][0] > 1.0
    no_worse = all(v["ratio"] <= 1 + noise[key] for key, v in ttft.items())
    out.update(valid=not problems, aa_noise=noise, router_tokens_per_s=tps, router_ttft=ttft,
               stacked_tokens_per_s=ratio("stacked", "tokens_per_s", complete),
               stacked_ttft={key: ratio("stacked", key, complete) for key in ("ttft_mean_s", "ttft_p95_s")},
               per_load={str(load): {"router_tokens_per_s": ratio("router", "tokens_per_s", [
                   b for b in complete if any(r["block"] == b and r["load"] == load for r in report["rows"])])}
                   for load in report.get("loads", [])},
               passes={"tokens_per_s": gain, "ttft_no_worse": no_worse})
    out["verdict"] = ("invalid" if problems else "pass: the router ships as the agent's setup" if gain and no_worse
                      else "fail: the router does not ship")
    if "frozen" in arms:
        # Learning while serving, fixed 2026-11-05 before any AG4-R data: the learning router over the frozen one,
        # paired by block. It decides only whether learning adds a measured gain; the router's ship rule is above.
        late = complete[len(complete) // 2:]
        learn_tps = ratio("router", "tokens_per_s", complete, "frozen")
        learn_ttft = {key: ratio("router", key, complete, "frozen") for key in ("ttft_mean_s", "ttft_p95_s")}
        learn_gain = learn_tps["ci95"][0] > 1.0
        learn_no_worse = all(v["ratio"] <= 1 + noise[key] for key, v in learn_ttft.items())
        out["learning"] = {
            "router_over_frozen_tokens_per_s": learn_tps, "router_over_frozen_ttft": learn_ttft,
            "late_half_tokens_per_s": ratio("router", "tokens_per_s", late, "frozen") if late else None,
            "frozen_over_single_tokens_per_s": ratio("frozen", "tokens_per_s", complete),
            "passes": {"tokens_per_s": learn_gain, "ttft_no_worse": learn_no_worse}, "frozen": report.get("frozen"),
            "verdict": ("invalid" if problems else "pass: learning while serving adds a measured gain"
                        if learn_gain and learn_no_worse else "fail: no measured gain from learning while serving")}
    return out


def run_ag4r(model: str, work: Path, *, smoke: bool, on_row: Any = None) -> dict[str, Any]:
    """AG4-R on one card: the product's own `choose_setup`, then, if it says `route`, both paths in their own workers
    and the three arms through the product on the same arrivals."""
    from huggingface_hub import snapshot_download

    from ironmule.plan_choice import choose_setup
    from ironmule.tune import resolve_local_model
    from ironmule_product.backend import MLXWorkerClient
    from ironmule_product.cli import _engine_options
    from ironmule_product.device_pool import cuda_devices, start_paths
    from ironmule_product.path_router import PathRouter
    from ironmule_product.service import ProductService
    from ironmule_product.state import ProductStore

    revision = Path(snapshot_download(model, revision=X1_REVISION if model == MODEL else None,
                                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja",
                                                      "tokenizer*"])).name
    identity = resolve_local_model(model, revision=revision).identity
    setup = choose_setup(model, "kaggle-ag4r", identity.identity_sha256)
    report: dict[str, Any] = {"plan": "ag4r", "model": model, "revision": revision, "setup": setup}
    if setup["engine"] != "route":
        return {**report, "stopped": f"the agent's setup is {setup['engine']}, not route: {setup['reason']}"}
    state = work / "rl2h-state"
    _ironmule("setup", "--mode", "server", state=state)
    added = _ironmule("models", "add", model, state=state)
    if added.get("returncode") != 0:
        return {**report, "failed": "models add", "added": added}
    store = ProductStore(state)
    store.set_request_timeout(1800)
    spec = store.model(model)
    device = (cuda_devices() or [None])[0]
    makers = [lambda: MLXWorkerClient(spec, compute_dtype=setup["compute_dtype"], cuda_device=device),
              lambda: MLXWorkerClient(spec, execution_variant="current_engine", cuda_device=device,
                                      **_engine_options("dense"))]
    router = PathRouter.from_setup(setup)
    pool = start_paths(store, spec, makers, [f"{device}:single", f"{device}:stacked"], ProductService, router)
    try:
        prompts, digest = dolly_prompts(work)
        report["prompts_sha256"] = digest
        measured = measure_ag4r(pool, model, router, prompts, 1800.0,
                                **({"loads": (3.0,), "traces": [AG4R_TRACES[0]], "requests": 4, "limit": 16}
                                   if smoke else {}), on_row=lambda partial: on_row({**report, **partial})
                                if on_row else None)
    finally:
        pool.close()
    return {**report, **measured}


# ---------------------------------------------------------------- AG4-C: the agent's own setup on a CPU (BACKLOG AG4)

AG4C_MODEL = "mlx-community/SmolLM2-135M-Instruct-8bit"  # V-CPU's model; Qwen3-0.6B-4bit took 9 s a token here
AG4C_TRACES = range(410, 412)                   # fresh seeds: 2 traces per state
AG4C_LOAD = 0.8                                 # arrival rate over the default path's calibrated capacity
AG4C_REQUESTS = 4
AG4C_LIMIT = 16


def measure_ag4c(pool: Any, model: str, deadline_s: float, *, traces: Any = AG4C_TRACES, load: float = AG4C_LOAD,
                 requests: int = AG4C_REQUESTS, limit: int = AG4C_LIMIT, levels: Any = None,
                 on_row: Any = None, capacity: float | None = None, short: bool = False,
                 first_block: int = 0) -> dict[str, Any]:
    """Every block (state, trace) runs the default path, the agent's chosen path and the default path again, on the
    same arrivals, both paths loaded side by side (path 0 the default, path 1 the agent's choice). AG5-G passes a
    `capacity` calibrated once per phase, `short` (the question alone, no context) and its running block number."""
    pool.router = ForcedRoute(0)

    def shorten(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{**item, "long": False} for item in items] if short else items

    calibration = None
    if capacity is None:
        burst = shorten([{**item, "offset_s": 0.0} for item in trace(9, 0, 1.0, requests, (limit,))])
        done = replay(pool, model, burst, deadline_s)
        makespan = max(r["end_s"] or 0.0 for r in done) or 1.0
        capacity = sum(r["outcome"] == "ok" for r in done) / makespan
        calibration = {"capacity_rps": capacity, "makespan_s": makespan, "ok": sum(r["outcome"] == "ok" for r in done)}
    report: dict[str, Any] = {"plan": "ag4c", "arms": ["default", "agent"], "states": list(H2_STATES),
                              "calibration": calibration or {"capacity_rps": capacity, "given": True},
                              "deadline_s": deadline_s, "rows": []}
    block = first_block
    for level in (range(len(H2_STATES)) if levels is None else levels):
        for t in traces:
            items = shorten(trace_h2(level, t, load * capacity, requests, (limit,)))
            for arm, run, index in (("default", 0, 0), ("agent", 0, 1), ("default", 1, 0)):
                pool.router = ForcedRoute(index)
                wait_idle(pool, timeout_s=deadline_s)
                rows = replay(pool, model, items, deadline_s)
                for item, row in zip(items, rows):
                    row["offset_s"] = item["offset_s"]
                report["rows"].append({"level": level, "trace": t, "arm": arm, "run": run, "block": block,
                                       "requests": rows})
                if on_row is not None:
                    on_row(report)
            block += 1
    return report


def analyse_ag4c(report: dict[str, Any], resamples: int = RESAMPLES, seed: int = 0) -> dict[str, Any]:
    """AG4-C as fixed in BACKLOG before data: the agent's path over the default (the mean of its two runs), paired
    by block; tokens per second must have a 2.5 % bound above 1, and the mean and 95th-percentile first token must be
    no worse than the default's beyond the A/A noise of the default's two runs."""
    rng = random.Random(seed)
    censor = report.get("censor_default")  # run 3 on: the default path's failures are censored, never the agent's
    cells = {(row["block"], row["arm"], row["run"]): _cell(row["requests"], censor if row["arm"] == "default" else None)
             for row in report.get("rows", [])}
    censored = sum(r.get("outcome") != "ok" for row in report.get("rows", []) if row["arm"] == "default"
                   for r in row["requests"]) if censor else 0
    blocks = sorted({b for b, _, _ in cells})
    complete = [b for b in blocks if all(cells.get(k) for k in ((b, "default", 0), (b, "default", 1), (b, "agent", 0)))]
    out: dict[str, Any] = {"schema": "ironmule.rl2_headroom.ag4c.analysis.v1", "complete_blocks": len(complete),
                           "censored_default_requests": censored,
                           "setup": {k: (report.get("setup") or {}).get(k) for k in ("engine", "compute_dtype",
                                                                                     "reason")}}
    if report.get("stopped"):
        return {**out, "valid": False, "verdict": f"stopped: {report['stopped']}"}
    problems = [f"block {b} is incomplete or has a failed request" for b in blocks if b not in complete]
    if len(complete) < 2:
        return {**out, "valid": False, "problems": problems, "verdict": "invalid: fewer than two complete blocks"}

    def ratio(key: str) -> dict[str, Any]:
        pairs = [(cells[b, "agent", 0][key], (cells[b, "default", 0][key] + cells[b, "default", 1][key]) / 2)
                 for b in complete]
        boot = []
        for _ in range(resamples):
            sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
            boot.append(sum(x for x, _ in sample) / sum(y for _, y in sample))
        return {"ratio": sum(x for x, _ in pairs) / sum(y for _, y in pairs),
                "ci95": [_quantile(boot, 0.025), _quantile(boot, 0.975)]}

    noise = {}
    for key in ("tokens_per_s", "ttft_mean_s", "ttft_p95_s"):
        values = sorted(abs(cells[b, "default", 1][key] / cells[b, "default", 0][key] - 1) for b in complete)
        noise[key] = values[max(0, math.ceil(0.95 * len(values)) - 1)]
    tps = ratio("tokens_per_s")
    ttft = {key: ratio(key) for key in ("ttft_mean_s", "ttft_p95_s")}
    gain = tps["ci95"][0] > 1.0
    no_worse = all(v["ratio"] <= 1 + noise[key] for key, v in ttft.items())
    out.update(valid=not problems, problems=problems, aa_noise=noise, agent_tokens_per_s=tps, agent_ttft=ttft,
               passes={"tokens_per_s": gain, "ttft_no_worse": no_worse})
    out["verdict"] = ("invalid" if problems else "pass: the agent's own setup beats the default here"
                      if gain and no_worse else "fail")
    return out


AG4C_RUN2 = {"traces": range(412, 414), "levels": (1, 2)}  # post hoc to run 1: bursts and spike, fresh seeds
# Run 3, post hoc to run 2 (whose default path missed the deadline in 2 of 4 blocks): fresh seeds, a 3,600 s deadline,
# and the default path's failures censored at it (`_cell`); an agent failure still voids its block.
AG4C_RUN3 = {"traces": range(414, 416), "levels": (1, 2)}
AG4C_RUN3_DEADLINE_S = 3600.0


def run_ag4c(model: str, work: Path, *, smoke: bool, on_row: Any = None) -> dict[str, Any]:
    """AG4-C on this machine's CPU: the product's own `choose_setup`, then the default path and the agent's path
    side by side through the product, on the same arrivals."""
    from huggingface_hub import snapshot_download

    from ironmule.plan_choice import choose_setup
    from ironmule.tune import resolve_local_model
    from ironmule_product.backend import MLXWorkerClient
    from ironmule_product.device_pool import start_paths
    from ironmule_product.service import ProductService
    from ironmule_product.state import ProductStore

    revision = Path(snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                                             "*.jinja", "tokenizer*"])).name
    identity = resolve_local_model(model, revision=revision).identity
    setup = choose_setup(model, "ag4c-" + platform.machine(), identity.identity_sha256)
    report: dict[str, Any] = {"plan": "ag4c", "model": model, "revision": revision, "setup": setup}
    if setup["engine"] != "reference" or setup["compute_dtype"] is None:
        return {**report, "stopped": f"the agent kept {setup['engine']} with {setup['compute_dtype']}: "
                                     f"{setup['reason']}"}
    state = work / "rl2h-state"
    _ironmule("setup", "--mode", "server", state=state)
    added = _ironmule("models", "add", model, state=state)
    if added.get("returncode") != 0:
        return {**report, "failed": "models add", "added": added}
    store = ProductStore(state)
    deadline = 7200.0 if smoke else AG4C_RUN3_DEADLINE_S
    store.set_request_timeout(int(deadline))
    if not smoke:  # fixed before run 3's data; partial reports carry it too
        report["censor_default"] = {"deadline_s": deadline, "limit": AG4C_LIMIT}
    spec = store.model(model)
    makers = [lambda: MLXWorkerClient(spec), lambda: MLXWorkerClient(spec, compute_dtype=setup["compute_dtype"])]
    pool = start_paths(store, spec, makers, ["cpu:default", "cpu:agent"], ProductService, ForcedRoute(0))
    try:
        measured = measure_ag4c(pool, model, deadline, **({"traces": [AG4C_TRACES[0]], "requests": 2, "limit": 8}
                                                         if smoke else AG4C_RUN3),
                                on_row=(lambda partial: on_row({**report, **partial})) if on_row else None)
    finally:
        pool.close()
    return {**report, **measured}


# ---------------------------------------------------------------- AG6: the dreamer checked against AG4-R run 1

AG6_CHECK_WITHIN = 0.95   # the named mode's measured tokens per second, at least this share of the best arm's
AG6_CHECK_SHARE = 0.8     # in at least this share of the complete blocks
AG6_CHECK_RATIO = 0.25    # and the replayed stacked-over-single ratio, pooled, within 25 % of the measured one


def check_ag6(report: dict[str, Any]) -> dict[str, Any]:
    """AG6's check in simulation, as fixed in BACKLOG AG6 before it ran: on each complete AG4-R block, the dreamer
    (from the prices the agent measured at startup) names a mode for the block's arrivals and token counts; that
    mode's measured arm must be the best arm by tokens per second, or within 5 % of it, in 80 % of the blocks, and
    the replayed stacked-over-single ratio, pooled, within 25 % of the measured one. A model check on data that
    already exists, not evidence for a claim."""
    from ironmule_product.path_router import PolicyRouter, dream

    costs = PolicyRouter.from_setup(report["setup"]).costs
    cells = {(row["block"], row["arm"], row["run"]): row for row in report.get("rows", [])}
    blocks = sorted({b for b, _, _ in cells})
    arm_of = {"route": "router", "stacked": "stacked", "single": "single"}
    rows, hits = [], 0
    replayed_single = replayed_stacked = measured_single = measured_stacked = 0.0
    for b in blocks:
        needed = [(b, "single", 0), (b, "single", 1), (b, "router", 0), (b, "stacked", 0)]
        measured = {key: _cell(cells[key]["requests"]) if key in cells else None for key in needed}
        if any(value is None for value in measured.values()):
            continue
        base = cells[b, "single", 0]["requests"]
        arrivals = [(r["offset_s"], max(1, int(r["tokens"]))) for r in base]
        result = dream(costs, arrivals, "route")
        tps = {"single": (measured[b, "single", 0]["tokens_per_s"] + measured[b, "single", 1]["tokens_per_s"]) / 2,
               "router": measured[b, "router", 0]["tokens_per_s"], "stacked": measured[b, "stacked", 0]["tokens_per_s"]}
        best = max(tps, key=tps.get)
        named = arm_of[result["mode"]]
        hit = tps[named] >= AG6_CHECK_WITHIN * tps[best]
        hits += hit
        replayed_single += result["replayed"]["single"]["tokens_per_s"]
        replayed_stacked += result["replayed"]["stacked"]["tokens_per_s"]
        measured_single += tps["single"]
        measured_stacked += tps["stacked"]
        rows.append({"block": b, "load": cells[b, "single", 0]["load"], "level": cells[b, "single", 0]["level"],
                     "named": named, "best": best, "hit": hit, "measured": tps,
                     "replayed": {mode: value["tokens_per_s"] for mode, value in result["replayed"].items()}})
    if not rows:
        return {"verdict": "no complete block to check", "blocks": 0}
    replayed_ratio, measured_ratio = replayed_stacked / replayed_single, measured_stacked / measured_single
    share = hits / len(rows)
    close = abs(replayed_ratio / measured_ratio - 1) <= AG6_CHECK_RATIO
    return {"blocks": len(rows), "named_mode_hits": hits, "share": share, "replayed_stacked_over_single": replayed_ratio,
            "measured_stacked_over_single": measured_ratio, "ratio_within": close, "rows": rows,
            "verdict": "pass" if share >= AG6_CHECK_SHARE and close else "fail: repair the world model first"}


# ---------------------------------------------------------------- AG6-H: the learning agent on one T4

AG6H_ARMS = ("single", "router", "agent", "stacked")   # the middle three rotate; `single` runs again at the end
AG6H_LOADS = (0.8, 3.0)
AG6H_LEVELS = (1, 2)                                  # bursts and spike
AG6H_TRACES = range(440, 444)                         # fresh seeds: 4 traces per state and load
AG6H_REQUESTS = 32
AG6H_LIMIT = 256
AG6H_BUDGET_S = 10 * 3600.0                           # no new block after 10 hours (AG4-R run 1 was cut at 12)
AG6H_MIN_BLOCKS = 4


def measure_ag6h(pool: Any, model: str, router: Any, agent: Any, prompts: list[str] | None, deadline_s: float, *,
                 loads: tuple[float, ...] = AG6H_LOADS, levels: tuple[int, ...] = AG6H_LEVELS,
                 traces: Any = AG6H_TRACES, requests: int = AG6H_REQUESTS, limit: int = AG6H_LIMIT,
                 budget_s: float = AG6H_BUDGET_S, on_row: Any = None) -> dict[str, Any]:
    """AG6-H (BACKLOG AG6, fixed before its data): every block (trace, load, state) runs the single path, then the
    router, the learning agent and the stacked engine (rotated), then the single path again, on the same arrivals.
    The router and the agent keep learning across blocks; the agent's mode is recorded around each of its cells.
    Blocks run trace by trace, so a time cut keeps every load and state; none starts after `budget_s`."""
    def items_for(level: int, load: float, t: int, capacity: float) -> list[dict[str, Any]]:
        items = trace_h2(level, t, load * capacity, requests, (limit,))
        return lp_items(items, prompts, t, True, limit) if prompts else items

    started = time.monotonic()
    pool.router = ForcedRoute(0)
    calibration_items = None if not prompts else [
        {"offset_s": 0.0, "prompt": 0, "long": False, "max_tokens": limit, "text": text} for text in prompts[-16:]]
    burst = [{**item, "offset_s": 0.0} for item in (calibration_items or trace(9, 0, 1.0, 16, (limit,)))]
    done = replay(pool, model, burst, deadline_s)
    makespan = max(r["end_s"] or 0.0 for r in done) or (time.monotonic() - started)
    capacity = sum(r["outcome"] == "ok" for r in done) / makespan
    report: dict[str, Any] = {"plan": "ag6h", "arms": list(AG6H_ARMS), "states": list(H2_STATES), "loads": list(loads),
                              "levels": list(levels), "calibration": {"capacity_rps": capacity, "makespan_s": makespan,
                                                                     "ok": sum(r["outcome"] == "ok" for r in done)},
                              "deadline_s": deadline_s, "budget_s": budget_s, "rows": []}
    routes = {"single": ForcedRoute(0), "stacked": ForcedRoute(1), "router": router, "agent": agent}
    block = 0
    for t in traces:
        for load in loads:
            for level in levels:
                if time.monotonic() - started > budget_s:
                    report["time_cut"] = {"after_blocks": block, "elapsed_s": time.monotonic() - started}
                    break
                items = items_for(level, load, t, capacity)
                others = list(AG6H_ARMS[1:])
                turn = block % len(others)
                order = [("single", 0), *((name, 0) for name in others[turn:] + others[:turn]), ("single", 1)]
                for arm, run in order:
                    pool.router = routes[arm]
                    wait_idle(pool, timeout_s=deadline_s)
                    before = agent.mode if arm == "agent" else None
                    requests_out = replay(pool, model, items, deadline_s)
                    for item, row in zip(items, requests_out):
                        row["offset_s"] = item["offset_s"]
                    row = {"level": level, "load": load, "trace": t, "arm": arm, "run": run, "block": block,
                           "requests": requests_out}
                    if arm == "agent":
                        row["agent"] = {"mode_before": before, "mode_after": agent.mode,
                                        "switches": len(agent.switches), "dreams": agent.dreams}
                    report["rows"].append(row)
                    if on_row is not None:
                        on_row(report)
                block += 1
            if "time_cut" in report:
                break
        if "time_cut" in report:
            break
    pool.router = router
    health = pool.health()
    report.update(router=router.status(), agent=agent.status(), fallbacks=(health.get("paths") or {}).get("fallbacks"),
                  elapsed_s=time.monotonic() - started)
    return report


def analyse_ag6h(report: dict[str, Any], resamples: int = RESAMPLES, seed: int = 0) -> dict[str, Any]:
    """AG6-H as fixed in BACKLOG AG6 before its data. Pass: the agent's tokens per second over the router's, paired
    by block, has a 2.5 % bound above 1 (2,000 resamples), and its mean and 95th-percentile first token are no worse
    than the router's beyond the A/A noise of the two single runs. A failed request voids its block; fewer than 4
    complete blocks make the run invalid. Recorded: the agent over the single path and over the stacked engine,
    the other arms over the single path, the agent's modes, and the pool's fallbacks."""
    rng = random.Random(seed)
    cells = {(row["block"], row["arm"], row["run"]): _cell(row["requests"]) for row in report.get("rows", [])}
    blocks = sorted({b for b, _, _ in cells})
    arms = report.get("arms") or list(AG6H_ARMS)
    complete = [b for b in blocks if cells.get((b, "single", 1)) and all(cells.get((b, arm, 0)) for arm in arms)]
    out: dict[str, Any] = {"schema": "ironmule.rl2_headroom.ag6h.analysis.v1", "complete_blocks": len(complete),
                           "voided_blocks": [b for b in blocks if b not in complete],
                           "time_cut": report.get("time_cut"), "fallbacks": report.get("fallbacks"),
                           "setup": {k: (report.get("setup") or {}).get(k) for k in ("engine", "compute_dtype",
                                                                                     "width", "reason")}}
    if report.get("stopped"):
        return {**out, "valid": False, "verdict": f"stopped: {report['stopped']}"}
    if len(complete) < AG6H_MIN_BLOCKS:
        return {**out, "valid": False, "verdict": f"invalid: fewer than {AG6H_MIN_BLOCKS} complete blocks"}

    def value(b: int, arm: str, key: str) -> float:
        if arm == "single":
            return (cells[b, "single", 0][key] + cells[b, "single", 1][key]) / 2
        return cells[b, arm, 0][key]

    def ratio(arm: str, base: str, key: str, blocks_: list[int]) -> dict[str, Any] | None:
        if not blocks_:
            return None
        pairs = [(value(b, arm, key), value(b, base, key)) for b in blocks_]
        boot = []
        for _ in range(resamples):
            sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
            boot.append(sum(x for x, _ in sample) / sum(y for _, y in sample))
        return {"ratio": sum(x for x, _ in pairs) / sum(y for _, y in pairs),
                "ci95": [_quantile(boot, 0.025), _quantile(boot, 0.975)]}

    aa = {key: sorted(abs(cells[b, "single", 1][key] / cells[b, "single", 0][key] - 1) for b in complete)
          for key in ("tokens_per_s", "ttft_mean_s", "ttft_p95_s")}
    noise = {key: values[max(0, math.ceil(0.95 * len(values)) - 1)] for key, values in aa.items()}
    tps = ratio("agent", "router", "tokens_per_s", complete)
    ttft = {key: ratio("agent", "router", key, complete) for key in ("ttft_mean_s", "ttft_p95_s")}
    gain = tps["ci95"][0] > 1.0
    no_worse = all(v["ratio"] <= 1 + noise[key] for key, v in ttft.items())
    where = {(row["block"]): (row["load"], row["level"]) for row in report.get("rows", [])}
    by = {}
    for load in report.get("loads", []):
        for level in report.get("levels", []):
            part = [b for b in complete if where.get(b) == (load, level)]
            by[f"load {load}, {H2_STATES[level]}"] = {
                "blocks": len(part), "agent_over_router": ratio("agent", "router", "tokens_per_s", part),
                "agent_over_single": ratio("agent", "single", "tokens_per_s", part),
                "stacked_over_single": ratio("stacked", "single", "tokens_per_s", part)}
    modes = [{"block": row["block"], **row["agent"]} for row in report.get("rows", []) if row.get("agent")]
    out.update(valid=True, aa_noise=noise, agent_over_router_tokens_per_s=tps, agent_over_router_ttft=ttft,
               passes={"tokens_per_s": gain, "ttft_no_worse": no_worse},
               recorded={"agent_over_single": ratio("agent", "single", "tokens_per_s", complete),
                         "agent_over_stacked": ratio("agent", "stacked", "tokens_per_s", complete),
                         "router_over_single": ratio("router", "single", "tokens_per_s", complete),
                         "stacked_over_single": ratio("stacked", "single", "tokens_per_s", complete),
                         "agent_ttft_over_single": {key: ratio("agent", "single", key, complete)
                                                    for key in ("ttft_mean_s", "ttft_p95_s")},
                         "by_load_and_state": by, "agent_modes": modes,
                         "agent": report.get("agent"), "router": report.get("router")})
    out["verdict"] = ("pass: the learning agent beats today's router on this card" if gain and no_worse
                      else "fail: no measured gain over today's router")
    return out


def run_ag6h(model: str, work: Path, *, smoke: bool, on_row: Any = None) -> dict[str, Any]:
    """AG6-H on one card: the product's own `choose_setup`; if it says `route`, both paths in their own workers, and
    the four arms through the product on the same arrivals."""
    from huggingface_hub import snapshot_download

    from ironmule.plan_choice import choose_setup
    from ironmule.tune import resolve_local_model
    from ironmule_product.backend import MLXWorkerClient
    from ironmule_product.cli import _engine_options
    from ironmule_product.device_pool import cuda_devices, start_paths
    from ironmule_product.path_router import PathRouter, PolicyRouter
    from ironmule_product.service import ProductService
    from ironmule_product.state import ProductStore

    revision = Path(snapshot_download(model, revision=X1_REVISION if model == MODEL else None,
                                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja",
                                                      "tokenizer*"])).name
    identity = resolve_local_model(model, revision=revision).identity
    setup = choose_setup(model, "kaggle-ag6h", identity.identity_sha256)
    report: dict[str, Any] = {"plan": "ag6h", "model": model, "revision": revision, "setup": setup}
    if on_row is not None:
        on_row(report)
    if setup["engine"] != "route":
        return {**report, "stopped": f"the agent's setup is {setup['engine']}, not route: {setup['reason']}"}
    state = work / "rl2h-state"
    _ironmule("setup", "--mode", "server", state=state)
    added = _ironmule("models", "add", model, state=state)
    if added.get("returncode") != 0:
        return {**report, "failed": "models add", "added": added}
    store = ProductStore(state)
    store.set_request_timeout(1800)
    spec = store.model(model)
    device = (cuda_devices() or [None])[0]
    makers = [lambda: MLXWorkerClient(spec, compute_dtype=setup["compute_dtype"], cuda_device=device),
              lambda: MLXWorkerClient(spec, execution_variant="current_engine", cuda_device=device,
                                      **_engine_options("dense", setup.get("width")))]
    router, agent = PathRouter.from_setup(setup), PolicyRouter.from_setup(setup)
    try:
        pool = start_paths(store, spec, makers, [f"{device}:single", f"{device}:stacked"], ProductService, router)
    except Exception as exc:  # noqa: BLE001 - recorded with the agent's setup
        return {**report, "failed": f"the paths did not start: {exc!r}"[:500]}
    try:
        prompts, digest = dolly_prompts(work)
        report["prompts_sha256"] = digest
        measured = measure_ag6h(pool, model, router, agent, prompts, 1800.0,
                                **({"loads": (3.0,), "levels": (2,), "traces": [AG6H_TRACES[0]], "requests": 4,
                                    "limit": 16} if smoke else {}),
                                on_row=(lambda partial: on_row({**report, **partial})) if on_row else None)
    finally:
        pool.close()
    return {**report, **measured}


# ---------------------------------------------------------------- AG5-G: the agent's own width, Gemma 3 on T4s

AG5G_MODEL = "mlx-community/gemma-3-12b-it-4bit"
AG5G_REVISION = "86cc6a8dedbc456dd0e4af01a9d09f396f77e558"  # KB2's revision
AG5G_LOADS = (0.8, 3.0)        # AG4-R's loads, over the default path's calibrated capacity
AG5G_LEVELS = (1, 2)           # AG4-C's states: bursts and spike
AG5G_TRACES = (430,)           # a fresh seed, run in both phases
AG5G_REQUESTS = 12
AG5G_LIMIT = 256
AG5G_DEADLINE_S = 3600.0       # AG4-C run 3's deadline and censoring of the default path


def run_ag5g(model: str, work: Path, *, smoke: bool, on_row: Any = None) -> dict[str, Any]:
    """AG5-G (BACKLOG AG5, fixed before its data): the product's own `choose_setup` decides; then the default path
    (the native plan, one request at a time) and the agent's setup (the native plan with the stacked engine at the
    agent's width) side by side on two cards, through the product, on the same arrivals; phase 2 swaps the cards."""
    from huggingface_hub import snapshot_download

    from ironmule.plan_choice import choose_setup
    from ironmule.tune import resolve_local_model
    from ironmule_product.backend import MLXWorkerClient
    from ironmule_product.cli import _engine_options
    from ironmule_product.device_pool import cuda_devices, start_paths
    from ironmule_product.service import ProductService
    from ironmule_product.state import ProductStore

    revision = Path(snapshot_download(model, revision=AG5G_REVISION if model == AG5G_MODEL else None,
                                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja",
                                                      "tokenizer*"])).name
    report: dict[str, Any] = {"plan": "ag5g", "model": model, "revision": revision}
    devices = cuda_devices()
    if len(devices) < 2:
        return {**report, "stopped": f"AG5-G needs two cards, this session has {len(devices)}"}
    identity = resolve_local_model(model, revision=revision).identity
    setup = choose_setup(model, "kaggle-ag5g", identity.identity_sha256)
    report["setup"] = setup
    if on_row is not None:
        on_row(report)  # the agent's decision is kept whatever happens next
    if setup["engine"] != "dense" or not setup.get("width"):
        return {**report, "stopped": f"the agent's setup is {setup['engine']}, not the stacked engine: "
                                     f"{setup['reason']}"}
    width = int(setup["width"])
    state = work / "rl2h-state"
    _ironmule("setup", "--mode", "server", state=state)
    added = _ironmule("models", "add", model, state=state)
    if added.get("returncode") != 0:
        return {**report, "failed": "models add", "added": added}
    store = ProductStore(state)
    deadline = 600.0 if smoke else AG5G_DEADLINE_S
    store.set_request_timeout(int(deadline))
    report.update(censor_default={"deadline_s": deadline, "limit": AG5G_LIMIT}, width=width, rows=[], phases=[],
                  arms=["default", "agent"], states=list(H2_STATES), deadline_s=deadline)
    spec = store.model(model)
    sizes = {"requests": 2, "limit": 8} if smoke else {"requests": AG5G_REQUESTS, "limit": AG5G_LIMIT}
    loads, levels, traces = ((AG5G_LOADS[-1],), AG5G_LEVELS[:1], AG5G_TRACES) if smoke else (
        AG5G_LOADS, AG5G_LEVELS, AG5G_TRACES)
    for phase, (default_card, agent_card) in enumerate(((devices[0], devices[1]), (devices[1], devices[0]))):
        makers = [lambda card=default_card: MLXWorkerClient(spec, compute_dtype="native", cuda_device=card),
                  lambda card=agent_card: MLXWorkerClient(spec, execution_variant="current_engine", cuda_device=card,
                                                          **_engine_options("dense", width))]
        try:
            pool = start_paths(store, spec, makers, [f"{default_card}:default", f"{agent_card}:agent"],
                               ProductService, ForcedRoute(0))
        except Exception as exc:  # noqa: BLE001 - recorded: the stacked engine has not served Gemma before
            report["failed"] = f"phase {phase}: the paths did not start: {exc!r}"[:500]
            return report
        try:
            capacity = None
            for load in loads:
                measured = measure_ag4c(pool, model, deadline, traces=traces, load=load, levels=levels,
                                        capacity=capacity, short=True, first_block=len({r["block"] for r in
                                                                                         report["rows"]}),
                                        **sizes)
                if capacity is None:
                    capacity = measured["calibration"]["capacity_rps"]
                    report["phases"].append({"phase": phase, "default_card": default_card, "agent_card": agent_card,
                                             "calibration": measured["calibration"],
                                             "execution": pool.health().get("execution")})
                for row in measured["rows"]:
                    report["rows"].append({**row, "phase": phase, "load": load})
                if on_row is not None:
                    on_row(report)
        finally:
            pool.close()
    return report


def analyse_ag5g(report: dict[str, Any], resamples: int = RESAMPLES, seed: int = 0) -> dict[str, Any]:
    """AG5-G as fixed in BACKLOG AG5 before its data: AG4-C's analysis unchanged over all blocks (it decides), and
    the same ratio per load and per phase, recorded only."""
    out = analyse_ag4c(report, resamples, seed)
    out["schema"] = "ironmule.rl2_headroom.ag5g.analysis.v1"
    setup = report.get("setup") or {}
    out["width"] = report.get("width")
    out["curve_tokens_per_s"] = ((setup.get("decision") or {}).get("tokens_per_second") or {}).get("by_width")
    out["agent_setup_reason"] = setup.get("reason")
    if report.get("stopped") or not report.get("rows"):
        return out
    recorded = {}
    for key, value in (("load", lambda row: row.get("load")), ("phase", lambda row: row.get("phase"))):
        for each in sorted({value(row) for row in report["rows"]}):
            part = analyse_ag4c({**report, "rows": [row for row in report["rows"] if value(row) == each]},
                                resamples, seed)
            recorded[f"{key} {each}"] = {"tokens_per_s": part.get("agent_tokens_per_s"),
                                         "ttft": part.get("agent_ttft"), "complete_blocks": part.get("complete_blocks")}
    out["recorded"] = recorded
    return out


_BOOTSTRAP = '''"""IronMule RL2-H on this Kaggle session; written by training/rl2_headroom.py."""
from __future__ import annotations
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "ironmule[{extra}] @ git+{repository}@{commit}"], check=True)
'''


def write_kernel(directory: Path, owner: str, commit: str, plan: str = "h1", extra: list[str] | None = None,
                 suffix: str = "", cpu: bool = False) -> Path:
    """A private script kernel that installs IronMule at `commit` and runs this file's `plan` (with
    `extra` arguments, such as run 4's frozen policy); the report lands in the kernel's output as
    `rl2_headroom_report.json`. With `cpu` it is a CPU session (MLX's CPU backend, no GPU and no GPU quota)."""
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
    name = "ironmule-rl2-headroom" + ("" if plan == "h1" else f"-{plan}") + suffix
    extra_name = "cpu" if cpu else "cuda"
    (directory / "run.py").write_text(_BOOTSTRAP.format(repository=REPOSITORY, commit=commit, extra=extra_name)
                                      + body + entry)
    metadata = {"id": f"{owner}/{name}", "title": name, "code_file": "run.py",
                "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": not cpu,
                "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
                "model_sources": [], **({} if cpu else {"machine_shape": "NvidiaTeslaT4"})}
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
    parser.add_argument("--plan", choices=("h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "kp1", "h9", "h10", "h11",
                                           "ag1", "ag4r", "ag4c", "ag5g", "ag6h", "x1"),
                        default="h1",
                        help="run 1's to 4's fixed plan, RL4-S's, RL6's or LD1-V's run, or X1's exactness diagnostic")
    parser.add_argument("--policy-json", help="run 4: the frozen learned policy (JSON text or @path)")
    parser.add_argument("--best-fixed", help="run 4: run 3's best fixed arm")
    parser.add_argument("--smoke", action="store_true", help="tiny sizes: checks this tool, measures nothing")
    parser.add_argument("--profile", choices=sorted(AG1_PROFILES), default="t4",
                        help="--plan ag1: V on a T4 (default) or V-CPU on this machine's CPU (BACKLOG AG1)")
    parser.add_argument("--analyse", type=Path, metavar="REPORT", help="print the analysis of a finished report")
    parser.add_argument("--check-ag6", type=Path, metavar="REPORT",
                        help="AG6's check in simulation: the dreamer against an AG4-R report's measured arms")
    parser.add_argument("--write-kernel", type=Path, metavar="DIR",
                        help="write a Kaggle kernel folder instead of running (needs --owner and --commit)")
    parser.add_argument("--owner")
    parser.add_argument("--commit")
    args = parser.parse_args(None if argv is None else list(argv))
    if args.write_kernel is not None:
        if not args.owner or not args.commit:
            parser.error("--write-kernel needs --owner and --commit")
        extra = []
        if args.plan in ("h4", "h5", "h6", "h7"):
            needs_fixed = args.plan in ("h4", "h5")
            if not args.policy_json or (needs_fixed and not args.best_fixed):
                parser.error(f"--plan {args.plan} needs --policy-json" + (" and --best-fixed" if needs_fixed else ""))
            text = Path(args.policy_json[1:]).read_text() if args.policy_json.startswith("@") else args.policy_json
            LearnedPolicy.from_json(json.loads(text))  # refuse a policy whose digest does not match
            extra = (["--best-fixed", args.best_fixed] if needs_fixed else []) + \
                ["--policy-json", json.dumps(json.loads(text), sort_keys=True)]
        suffix = ""
        if args.plan == "ag1" and args.profile != "t4":  # e.g. V2's profile, as its own kernel
            extra, suffix = extra + ["--profile", args.profile], f"-{args.profile}"
        cpu = (args.plan == "ag1" and AG1_PROFILES[args.profile].get("accelerator") == "cpu") or args.plan == "ag4c"
        print(json.dumps({"kernel": str(write_kernel(args.write_kernel, args.owner, args.commit, args.plan, extra,
                                                     suffix, cpu=cpu))}))
        return 0
    if args.check_ag6 is not None:
        report = json.loads(args.check_ag6.read_text())
        print(json.dumps(check_ag6(report.get("rl2h", report)), indent=1, sort_keys=True, default=str))
        return 0
    if args.analyse is not None:
        report = json.loads(args.analyse.read_text())
        inner = report.get("rl2h", report)
        judge = {"h8": analyse_h8, "kp1": analyse_kp1, "h9": analyse_h9, "h10": analyse_h10,
                 "h11": analyse_h11, "ag1": analyse_ag1, "ag4r": analyse_ag4r,
                 "ag4c": analyse_ag4c, "ag5g": analyse_ag5g, "ag6h": analyse_ag6h}.get(inner.get("plan"), analyse)
        print(json.dumps(judge(inner), indent=1, sort_keys=True))
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
    if args.plan == "ag4c" and args.model is None:
        model = AG4C_MODEL
    if args.plan == "ag5g" and args.model is None:
        model = AG5G_MODEL
    if args.plan == "x1":
        report["x1"] = run_x1(model, args.out, smoke=args.smoke)
        report["finished_unix"] = time.time()
        save()
        print(json.dumps({"report": str(args.out), "verdict": report["x1"]["summary"]["verdict"]}), flush=True)
        return 0
    if args.plan in ("h8", "kp1", "h9", "h10", "h11", "ag1", "ag4r", "ag4c", "ag5g", "ag6h"):
        run, judge = {"h8": (run_h8, analyse_h8), "kp1": (run_kp1, analyse_kp1), "h9": (run_h9, analyse_h9),
                      "h10": (run_h10, analyse_h10), "h11": (run_h11, analyse_h11),
                      "ag1": (run_ag1, analyse_ag1), "ag4r": (run_ag4r, analyse_ag4r),
                      "ag4c": (run_ag4c, analyse_ag4c), "ag5g": (run_ag5g, analyse_ag5g),
                      "ag6h": (run_ag6h, analyse_ag6h)}[args.plan]
        extra = {"profile": args.profile} if args.plan == "ag1" else {}
        if args.plan == "ag1" and args.model is None and not args.smoke:
            model = AG1_PROFILES[args.profile]["model"]
        report["rl2h"] = run(model, work, smoke=args.smoke, on_row=save, **extra)
        report["analysis"] = judge(report["rl2h"])
        report["finished_unix"] = time.time()
        save()
        print(json.dumps({"report": str(args.out), "analysis": {k: v for k, v in report["analysis"].items()
                                                                 if k != "mean_cost_per_level"}}, default=str), flush=True)
        return 0
    policy = names = None
    if args.plan in ("h4", "h5", "h6", "h7"):
        needs_fixed = args.plan in ("h4", "h5")
        if not args.policy_json or (needs_fixed and not args.best_fixed):
            parser.error(f"--plan {args.plan} needs --policy-json" + (" and --best-fixed" if needs_fixed else ""))
        text = Path(args.policy_json[1:]).read_text() if args.policy_json.startswith("@") else args.policy_json
        policy = LearnedPolicy.from_json(json.loads(text))
        names = {"h6": arms_h6, "h7": arms_h7}[args.plan]() if not needs_fixed else arms_h4(args.best_fixed)
    report["rl2h"] = run(model, work, smoke=args.smoke, plan=args.plan, on_row=save, policy=policy, names=names)
    report["analysis"] = analyse(report["rl2h"])
    report["finished_unix"] = time.time()
    save()
    print(json.dumps({"report": str(args.out), "verdict": report["analysis"]["verdict"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
