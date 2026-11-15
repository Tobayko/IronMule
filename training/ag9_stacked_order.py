"""AG9 step S: does the order of joins inside the stacked engine have headroom? (BACKLOG AG9, in simulation).

The CPU stacked engine (CPU6) serves up to 8 requests in fixed slots; when a slot frees, the service gives it to
the oldest waiting request. Which request takes it changes no answer (CPU6: rows invariant to slot and company at a
fixed width), so order is the one decision left to an online agent there. This replays RL2-H2's three traffic
states through a model of that engine, priced by the agent's own CPU6-A2 measurement, under five orders, and
applies the rule fixed in the entry before the data.

Engine model (fixed in the entry): a request that takes a slot is prefilled before the next decode step, and every
member waits for it; its first token comes with its prefill. A decode step with `n` members costs the measured
`plan_step_s[n]` of the short-prompt row, moved towards the long-prompt row by the share of members whose prompt is
long. A prefill costs the plan's measured first token, linear in prompt tokens between the short and the long
prompt. Capacity, for `quiet`'s arrival rate, is 16 over the span `fifo` takes for 16 short requests of limit 96
arriving at once.

    python training/ag9_stacked_order.py --out ag9s.json
    python training/ag9_stacked_order.py --prices cpu6a2_report.json --out ag9s.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable

SLOTS = 8
TRACES = 40
REQUESTS = 16
RESAMPLES = 2000
STATES = ("quiet", "bursts", "spike")
VARIANTS = ("to_limit", "early")
ORDERS = ("fifo", "sjf", "ljf", "sjf*", "ljf*")
SHORT_PROMPT, LONG_PROMPT = 11, 328
THROUGHPUT_BOUND, FIRST_BOUND = 1.03, 1.02  # headroom: tokens/s 2.5 % bound above, first token 97.5 % at most
FIRST_HEADROOM, THROUGHPUT_FLOOR = 0.80, 0.98

# The agent's own measurement in CPU6-A2 (2026-11-10, this container, SmolLM2-135M-Instruct-8bit rev 0f0d9b8,
# OpenBLAS, plan version 2), `setup.stacked.speed` of its report.
CPU6A2_PRICES = {
    "short": {"plan_first_s": 0.09314180000001215,
              "plan_step_s": {"1": 0.06465526499960106, "2": 0.07279454950003128, "3": 0.0733999579997544,
                              "4": 0.08835833000011917, "5": 0.0811222070005897, "6": 0.0872466600012558,
                              "7": 0.09837623649946181, "8": 0.09203845250067388}},
    "long": {"plan_first_s": 1.4954592079993745,
             "plan_step_s": {"1": 0.08160856650010828, "2": 0.07492590699985158, "3": 0.10012071450000803,
                             "4": 0.09022767299939005, "5": 0.09575461950043973, "6": 0.10880968600031338,
                             "7": 0.12229219200071384, "8": 0.11580994550058676}},
}


class Prices:
    def __init__(self, speed: dict[str, Any]):
        self.short = {int(k): float(v) for k, v in speed["short"]["plan_step_s"].items()}
        self.long = {int(k): float(v) for k, v in speed["long"]["plan_step_s"].items()}
        self.first_short = float(speed["short"]["plan_first_s"])
        self.first_long = float(speed["long"]["plan_first_s"])

    def prefill(self, prompt: int) -> float:
        per = (self.first_long - self.first_short) / (LONG_PROMPT - SHORT_PROMPT)
        return max(1e-4, self.first_short + per * (prompt - SHORT_PROMPT))

    def step(self, members: int, long_share: float) -> float:
        n = max(1, min(SLOTS, members))
        return self.short[n] + long_share * (self.long[n] - self.short[n])


def _key(order: str) -> Callable[[dict[str, Any]], tuple]:
    if order == "fifo":
        return lambda r: (r["arrive"], r["id"])
    field = "limit" if order in ("sjf", "ljf") else "length"
    sign = 1 if order.startswith("sjf") else -1
    return lambda r: (sign * r[field], r["arrive"], r["id"])


def simulate(trace: list[dict[str, Any]], order: str, prices: Prices, slots: int = SLOTS) -> dict[str, float]:
    """One trace through the engine model under `order`: tokens per second and mean first token."""
    key = _key(order)
    pending = sorted(trace, key=lambda r: (r["arrive"], r["id"]))
    waiting: list[dict[str, Any]] = []
    active: list[list[Any]] = []  # [remaining tokens, long]
    now, i, tokens, first, end = 0.0, 0, 0, [], []

    def admit() -> None:
        nonlocal i
        while i < len(pending) and pending[i]["arrive"] <= now:
            waiting.append(pending[i])
            i += 1

    while i < len(pending) or waiting or active:
        admit()
        if not waiting and not active:
            now = pending[i]["arrive"]
            continue
        while waiting and len(active) < slots:
            pick = min(waiting, key=key)
            waiting.remove(pick)
            now += prices.prefill(pick["prompt"])
            first.append(now - pick["arrive"])
            tokens += pick["length"]
            if pick["length"] > 1:
                active.append([pick["length"] - 1, pick["prompt"] > SHORT_PROMPT])
            else:
                end.append(now)
            admit()
        if active:
            now += prices.step(len(active), sum(1 for a in active if a[1]) / len(active))
            for member in active:
                member[0] -= 1
            end.extend(now for member in active if member[0] <= 0)
            active = [member for member in active if member[0] > 0]
    span = max(end) - min(r["arrive"] for r in trace)
    return {"tokens_per_s": tokens / span, "mean_first_s": sum(first) / len(first), "span_s": span}


def capacity(prices: Prices) -> float:
    burst = [{"id": k, "arrive": 0.0, "prompt": SHORT_PROMPT, "limit": 96, "length": 96} for k in range(REQUESTS)]
    return REQUESTS / simulate(burst, "fifo", prices)["span_s"]


def trace(state: str, variant: str, seed: int, rate: float) -> list[dict[str, Any]]:
    rng = random.Random(f"ag9-{state}-{variant}-{seed}")
    if state == "quiet":
        times, t = [], 0.0
        for _ in range(REQUESTS):
            t += rng.expovariate(0.2 * rate)
            times.append(t)
    elif state == "bursts":
        times, t = [], 0.0
        for _ in range(4):
            t += rng.expovariate(0.6 * rate / 4)
            times += [t + rng.uniform(0.0, 0.2) for _ in range(4)]
    else:
        times = [rng.uniform(0.0, 1.0) for _ in range(REQUESTS)]
    out = []
    for k, arrive in enumerate(sorted(times)):
        long = state != "quiet" and rng.random() < 0.5
        limit = rng.choice((64, 128) if state == "quiet" else (32, 256))
        length = limit if variant == "to_limit" else max(1, round(limit * rng.uniform(0.25, 1.0)))
        out.append({"id": k, "arrive": arrive, "prompt": LONG_PROMPT if long else SHORT_PROMPT, "limit": limit,
                    "length": length})
    return out


def _interval(values: list[float], seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(RESAMPLES))
    return sum(values) / len(values), means[int(0.025 * RESAMPLES)], means[int(0.975 * RESAMPLES) - 1]


def run(prices: Prices, traces: int = TRACES) -> dict[str, Any]:
    rate = capacity(prices)
    cells: dict[str, Any] = {}
    for state in STATES:
        for variant in VARIANTS:
            rows = []
            for seed in range(traces):
                t = trace(state, variant, seed, rate)
                rows.append({order: simulate(t, order, prices) for order in ORDERS})
            summary = {}
            for order in ORDERS[1:]:
                tps = [r[order]["tokens_per_s"] / r["fifo"]["tokens_per_s"] for r in rows]
                fst = [r[order]["mean_first_s"] / r["fifo"]["mean_first_s"] for r in rows]
                summary[order] = {"tokens_per_s": _interval(tps), "mean_first": _interval(fst)}
            cells[f"{state}/{variant}"] = {"fifo_tokens_per_s": sum(r["fifo"]["tokens_per_s"] for r in rows) / traces,
                                           "fifo_mean_first_s": sum(r["fifo"]["mean_first_s"] for r in rows) / traces,
                                           "orders": summary}
    return {"protocol": "ag9-s", "capacity_rps": rate, "traces": traces, "cells": cells}


def analyse(report: dict[str, Any]) -> dict[str, Any]:
    """The rule fixed in BACKLOG AG9 before the data."""
    headroom, not_worse = [], {}
    for cell, data in report["cells"].items():
        for order, s in data["orders"].items():
            tps, fst = s["tokens_per_s"], s["mean_first"]
            if tps[1] > THROUGHPUT_BOUND and fst[2] <= FIRST_BOUND:
                headroom.append({"order": order, "cell": cell, "kind": "throughput"})
            if fst[2] < FIRST_HEADROOM and tps[1] >= THROUGHPUT_FLOOR:
                headroom.append({"order": order, "cell": cell, "kind": "first_token"})
            not_worse.setdefault(order, []).append(tps[1] >= THROUGHPUT_FLOOR and fst[2] <= FIRST_BOUND)
    if not headroom:
        decision = "no headroom: no learner for the order inside the stacked engine (Tier 0)"
    else:
        fixed = sorted({h["order"] for h in headroom if all(not_worse[h["order"]])})
        decision = (f"fixed rule: {', '.join(fixed)} (hardware test, no learner)" if fixed
                    else "states want different orders: step 2 (a learned order with slot reuse)")
    return {"headroom": headroom, "not_worse_everywhere": {o: all(v) for o, v in not_worse.items()},
            "decision": decision}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prices", type=Path, help="a CPU6 report whose setup.stacked.speed prices the engine")
    parser.add_argument("--traces", type=int, default=TRACES)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    speed = (json.loads(args.prices.read_text())["setup"]["stacked"]["speed"] if args.prices else CPU6A2_PRICES)
    report = run(Prices(speed), args.traces)
    report["analysis"] = analyse(report)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["analysis"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
