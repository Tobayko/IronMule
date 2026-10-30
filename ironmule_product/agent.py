"""AG1: the IronMule agent. It decides how this server schedules its requests, learns from every request
it serves, and serves what it learned only after that beat the incumbent on this machine (BACKLOG AG1).

**What it decides**, in microseconds per decision, in the LD1 policy format (`learned_dispatch.Policy`):
at each group start, shortest output limit first or arrival order, and the group's width (1, 2, 4 or 8,
within the backend's capacity). It never changes what a request answers: every request keeps its own
greedy answer on the grouped batch-1 path, and a replayed comparison checks that it did.

**How it learns**, in four steps that repeat for as long as the server runs:

1. *Observe.* Every finished group, real or replayed, updates a world model by recursive least squares:
   prefill seconds per member and per thousand prompt tokens, decode seconds per round and per further
   session in a round, as the pool model of `tools/rl4_agent.py` (RL4, RL8) prices them. Every finished
   request goes into a traffic memory of metadata (arrival, prompt tokens, output limit, tokens made);
   no prompt or answer text is kept.
2. *Plan.* While the server is idle, a planner process (its own process, so serving threads keep the
   GIL) trains a REINFORCE policy in a discrete-event model of this server, priced by step 1, on traffic
   resampled from the memory, with RL8's harm weight against the best fixed schedule in the model. The
   candidate is the learned policy if the model prefers it, else the best fixed schedule; nothing is
   proposed that the model does not price below the incumbent.
3. *Qualify.* While idle, arrivals recorded after the candidate was planned are replayed through the real
   backend with filler prompts: incumbent, candidate, incumbent again (A, B, AA, or B, A, AA). The candidate
   passes when its paired gain's lower 2.5 % bootstrap bound is above zero and above what the same estimator
   finds between the two incumbent runs (KP1's "every resolved gain" rule), and every replayed answer matched
   across the three; there is no minimum effect size. A width the model has never seen is measured in a
   replay window, never on a user's request. A triplet's arms may run in separate idle spells.
4. *Serve and watch.* A qualified policy decides every group. Two consecutive live windows with failed
   requests, three decision errors, a later comparison it loses, or the operator withdraws it; a withdrawn
   policy's digest is recorded and never serves again under this binding. Learning goes on, and the next
   candidate must qualify again.

It is off by default (`ironmule serve --agent on`), refused together with `--learning` and
`--dispatch-policy` (one decision owner), and keeps the reference schedule whenever it cannot prove better.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterable

from .learned_dispatch import (LONG_PROMPT_CHARS, LONG_TOKENS, ORDER_FEATURES, PLACEMENT_FEATURES, Policy,
                               SJF_AGE_S, View, WIDTH_CHOICES, order_features)

STATE_SCHEMA = "ironmule.agent.state.v2"
EVIDENCE_SCHEMA = "ironmule.agent.evidence.v1"
PLAN_SCHEMA = "ironmule.agent.plan.v1"
ENGINE_WIDTH = 4            # ironmule.executor.MAX_GROUP_WIDTH: sessions one grouped decode round serves
PRIOR = (0.05, 0.2, 0.0, 0.025, 0.004)  # prefill s/member, s/1000 prompt tokens, s/group; decode s/round, s/session
FORGET = 0.995              # recursive least squares forgetting: recent groups weigh more, so drift is tracked
MAX_TRAFFIC = 2048          # arrivals the memory keeps (metadata only)
MIN_GROUPS = 12             # groups the world model needs before it may plan
MIN_ARRIVALS = 32           # arrivals the memory needs before it may plan
FIDELITY = 0.25             # the planner runs only while the model's median relative error is below this (RL4)
MIN_SUPPORT = 3             # groups of a width before the model's price for it is trusted; else explored
MAX_EXPLORATIONS = 2        # replay windows that may measure one unpriced width
REPLAN_AFTER = 64           # new arrivals before a fresh plan when no candidate waits
QUIET_S = 5.0               # idle seconds before a replay window may start (RSI3-LIVE)
TRACE = 16                  # arrivals per replay window
MAX_REPLAY_TOKENS = 512     # a replayed output limit is capped here
MAX_PREEMPTIONS = 3         # demand cut a held-out window this often: the next one is used instead
MIN_TRIPLETS = 6
MAX_TRIPLETS = 24
LOOKS = (6, 12, 18, 24)     # a comparison is judged only at these numbers of triplets (group-sequential)
ALPHAS = (0.001, 0.002, 0.004, 0.018)  # one-sided error spent at each look (O'Brien-Fleming-like), 2.5 % in all
RESAMPLES = 4000
MONITOR_WINDOW = 24         # live requests per monitoring window
MAX_ERRORS = 3
SAVE_AFTER_S = 30.0
TABLE_QUEUE = 16            # queues up to this length are decided from a compiled table, longer ones by the policy
PLANNER = {"episodes": 3000, "seeds": 2, "lr": 0.05, "harm_weight": 5.0, "validation": 40, "held_out": 80,
           "trace": TRACE, "scale": (0.8, 1.25)}  # gaps scaled log-symmetrically: 0.5-1.5 weighted heavy load


# ---------------------------------------------------------------- the world model

class _RLS:
    """Recursive least squares with forgetting, for a model linear in its features. The covariance stays
    symmetric with a positive diagonal (else it is reopened, keeping the fit) and its trace at or below its start:
    in AG1 S run 1 it lost both after about 2,600 groups and the prefill fit ran off."""

    def __init__(self, theta: Iterable[float], spread: Iterable[float] | float = 1e2):
        self.theta = [float(x) for x in theta]
        n = len(self.theta)
        self.spread = [float(v) for v in spread] if isinstance(spread, (list, tuple)) else [float(spread)] * n
        self.p = [[self.spread[i] if i == j else 0.0 for j in range(n)] for i in range(n)]
        self.n = 0

    def reopen(self) -> None:
        """Forget how sure the fit is, not what it found: the next groups move it as at the start."""
        n = len(self.theta)
        self.p = [[self.spread[i] if i == j else 0.0 for j in range(n)] for i in range(n)]

    def predict(self, x: list[float]) -> float:
        return sum(t * v for t, v in zip(self.theta, x))

    def update(self, x: list[float], y: float, forget: float = FORGET) -> None:
        n = len(x)
        px = [sum(self.p[i][j] * x[j] for j in range(n)) for i in range(n)]
        denominator = forget + sum(x[i] * px[i] for i in range(n))
        if denominator <= 0:
            return
        gain = [v / denominator for v in px]
        error = y - self.predict(x)
        theta = [t + g * error for t, g in zip(self.theta, gain)]
        p = [[(self.p[i][j] - gain[i] * px[j]) / forget for j in range(n)] for i in range(n)]
        p = [[(p[i][j] + p[j][i]) / 2 for j in range(n)] for i in range(n)]
        if (not all(math.isfinite(v) for v in theta + [v for row in p for v in row])
                or any(p[i][i] <= 0 for i in range(n))):
            self.reopen()  # a numerical failure keeps the last finite fit
            return
        trace, limit = sum(p[i][i] for i in range(n)), sum(self.spread)
        if trace > limit:
            p = [[v * limit / trace for v in row] for row in p]
        self.theta, self.p = theta, p
        self.n += 1

    def to_json(self) -> dict[str, Any]:
        return {"theta": self.theta, "p": self.p, "n": self.n, "spread": self.spread}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "_RLS":
        model = cls(data["theta"], data.get("spread", 1e2))
        model.p = [[float(v) for v in row] for row in data["p"]]
        model.n = int(data["n"])
        return model


def decode_rounds(tokens: Iterable[int], engine_width: int = ENGINE_WIDTH) -> tuple[int, int]:
    """(rounds, further sessions summed over rounds) for a group whose members make `tokens` tokens each:
    the first token comes with each member's prefill, then rounds of at most `engine_width` sessions rotate."""
    remaining = [t - 1 for t in tokens if t > 1]
    rounds = further = 0
    while remaining:
        served = remaining[:engine_width]
        rounds += 1
        further += len(served) - 1
        remaining = remaining[len(served):] + [r - 1 for r in served if r > 1]
    return rounds, further


class WorldModel:
    """What this server's groups cost, refitted after every group it runs."""

    def __init__(self, engine_width: int = ENGINE_WIDTH):
        self.engine_width = engine_width
        # the per-group term starts sure of zero: groups of one cannot tell it from the per-member term, so
        # without wider groups to separate them the split wandered (AG1 S2, run 2)
        self.prefill = _RLS(PRIOR[:3], spread=(1e2, 1e2, 1e-2))
        self.decode = _RLS(PRIOR[3:])
        self.groups: dict[int, int] = {}
        self.errors: deque[float] = deque(maxlen=50)
        # each part's own recent errors: a part is reopened only when it is the one that went wrong
        self.part_errors: dict[str, deque[float]] = {"prefill": deque(maxlen=50), "decode": deque(maxlen=50)}
        self.since_reopen, self.reopened = {"prefill": 0, "decode": 0}, 0

    def params(self) -> dict[str, float]:
        a, b, g = self.prefill.theta
        c, d = self.decode.theta
        return {"prefill_s": max(0.0, a), "prefill_s_per_ktoken": max(0.0, b), "group_s": max(0.0, g),
                "step_s": max(1e-4, c), "step_s_per_session": max(0.0, d)}

    def predict_span(self, members: list[dict[str, Any]]) -> float:
        p = self.params()
        prefill = p["group_s"] + sum(p["prefill_s"] + p["prefill_s_per_ktoken"] * m["prompt_tokens"] / 1000
                                     for m in members)
        rounds, further = decode_rounds([m["tokens"] for m in members], self.engine_width)
        return prefill + p["step_s"] * rounds + p["step_s_per_session"] * further

    def observe(self, members: list[dict[str, Any]]) -> bool:
        """One finished group (each member: start, first, end in seconds, prompt_tokens, tokens, ok). A group
        with a failed member or inconsistent times teaches nothing. Returns whether it was learned. When a part's
        prices stop fitting, the machine changed or the fit went wrong. Prefill and decode errors are tracked
        apart: the part whose own recent median error passes `FIDELITY` is reopened (at most once per
        window of its errors), so it re-learns while the part that still fits keeps its precision (reopening
        both after a decode-only change let the prefill fit wander 10.4 %, AG1 S2 run 2)."""
        if not members or any(not m.get("ok") or m["tokens"] < 1 or m["first"] < m["start"] or m["end"] < m["first"]
                              for m in members):
            return False
        members = sorted(members, key=lambda m: m["first"])
        start = min(m["start"] for m in members)
        span = max(m["end"] for m in members) - start
        predicted = self.predict_span(members)
        self.errors.append(abs(predicted - span) / max(span, 1e-3))
        p = self.params()
        prefill_time = members[-1]["first"] - start
        prefill_guess = p["group_s"] + sum(p["prefill_s"] + p["prefill_s_per_ktoken"] * m["prompt_tokens"] / 1000
                                           for m in members)
        self.part_errors["prefill"].append(abs(prefill_guess - prefill_time) / max(prefill_time, 1e-3))
        rounds, further = decode_rounds([m["tokens"] for m in members], self.engine_width)
        if rounds:
            decode_time = max(m["end"] for m in members) - members[-1]["first"]
            decode_guess = p["step_s"] * rounds + p["step_s_per_session"] * further
            self.part_errors["decode"].append(abs(decode_guess - decode_time) / max(decode_time, 1e-3))
        for part, fit in (("prefill", self.prefill), ("decode", self.decode)):
            errors = self.part_errors[part]
            self.since_reopen[part] += 1
            if (len(errors) >= errors.maxlen and statistics.median(errors) > FIDELITY
                    and self.since_reopen[part] >= errors.maxlen):
                fit.reopen()  # this part went wrong (the machine changed or its fit did); the other keeps its own
                self.since_reopen[part], self.reopened = 0, self.reopened + 1
                errors.clear()
        previous = start
        for k, member in enumerate(members):  # each member's own prefill: since the one before had its first token;
            # the first member's also holds what the group costs once (rendering it, the worker round trip)
            self.prefill.update([1.0, member["prompt_tokens"] / 1000.0, 1.0 if k == 0 else 0.0],
                                member["first"] - previous)
            previous = member["first"]
        if rounds:
            self.decode.update([float(rounds), float(further)], max(m["end"] for m in members) - members[-1]["first"])
        width = len(members)
        self.groups[width] = self.groups.get(width, 0) + 1
        return True

    def fidelity(self) -> float | None:
        return statistics.median(self.errors) if len(self.errors) >= 5 else None

    def supported(self, width: int) -> bool:
        """Whether groups as wide as `width` were seen often enough to trust the linear price there (a round
        serves at most `engine_width` sessions, so wider groups cost what the rounds they rotate through cost)."""
        if width == 1:
            return self.groups.get(1, 0) >= MIN_SUPPORT
        return sum(n for w, n in self.groups.items() if w >= min(width, self.engine_width)) >= MIN_SUPPORT

    def to_json(self) -> dict[str, Any]:
        return {"engine_width": self.engine_width, "prefill": self.prefill.to_json(), "decode": self.decode.to_json(),
                "groups": {str(k): v for k, v in self.groups.items()}, "errors": list(self.errors),
                "part_errors": {k: list(v) for k, v in self.part_errors.items()}, "reopened": self.reopened}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "WorldModel":
        model = cls(int(data["engine_width"]))
        model.prefill, model.decode = _RLS.from_json(data["prefill"]), _RLS.from_json(data["decode"])
        model.groups = {int(k): int(v) for k, v in data["groups"].items()}
        model.errors.extend(float(e) for e in data["errors"])
        for part, errors in (data.get("part_errors") or {}).items():
            model.part_errors[part].extend(float(e) for e in errors)
        model.reopened = int(data.get("reopened", 0))
        return model


# ---------------------------------------------------------------- the model of this server

@dataclass
class Job:
    """One request in the model: what a decision sees (output limit, long prompt) and what it costs."""
    index: int
    arrive: float
    max_tokens: int
    long_prompt: bool
    prompt_tokens: int
    tokens: int
    first: float | None = None
    end: float | None = None


def _sjf_key(now: float) -> Callable[[Job], tuple]:
    """The service's shortest-first order (`ProductService._order_waiting`): aged requests by arrival."""
    return lambda r: (now - r.arrive <= SJF_AGE_S, r.max_tokens if now - r.arrive <= SJF_AGE_S else 0, r.arrive)


def simulate(params: dict[str, float], jobs: list[Job], sjf: Callable[[list[Job]], bool],
             width: Callable[[list[Job]], int], capacity: int, engine_width: int = ENGINE_WIDTH) -> float:
    """One card that runs one group at a time, members prefilled one after another, decode rounds of at most
    `engine_width` sessions, each member ending at its own last token (member release, H9). Returns the mean
    request cost: time in system plus time to first token, the live learner's contract."""
    pending = sorted(jobs, key=lambda j: (j.arrive, j.index))
    waiting: list[Job] = []
    now, i = 0.0, 0
    while i < len(pending) or waiting:
        while i < len(pending) and pending[i].arrive <= now:
            waiting.append(pending[i])
            i += 1
        if not waiting:
            now = pending[i].arrive
            continue
        if sjf(waiting):
            waiting.sort(key=_sjf_key(now))
        size = max(1, min(capacity, width(waiting)))
        group, waiting = waiting[:size], waiting[size:]
        t = now + params.get("group_s", 0.0)
        remaining = {}
        for job in group:
            t += params["prefill_s"] + params["prefill_s_per_ktoken"] * job.prompt_tokens / 1000
            job.first = t
            remaining[job.index] = job.tokens - 1
            if remaining[job.index] <= 0:
                job.end = t
        active = [j for j in group if remaining[j.index] > 0]
        while active:
            served = active[:engine_width]
            t += params["step_s"] + params["step_s_per_session"] * (len(served) - 1)
            for job in served:
                remaining[job.index] -= 1
                if remaining[job.index] == 0:
                    job.end = t
            active = active[len(served):] + [j for j in served if remaining[j.index] > 0]
        now = t
    return statistics.fmean((j.end - j.arrive) + (j.first - j.arrive) for j in jobs) if jobs else 0.0


def fixed_policy(width: int, sjf: bool) -> dict[str, Any]:
    """A fixed schedule in the LD1 v2 policy format: the bias feature picks the order and the width."""
    order = [20.0 if sjf else -20.0] + [0.0] * (ORDER_FEATURES - 1)
    rows = []
    for choice in WIDTH_CHOICES:
        rows += [20.0 if choice == width else 0.0] + [0.0] * (ORDER_FEATURES - 1)
    return policy_json([0.0] * PLACEMENT_FEATURES, order, rows)


def policy_json(placement: list[float], order: list[float], width: list[float]) -> dict[str, Any]:
    body = {"schema": "ironmule.rl4.policy.v2", "placement": placement, "order": order,
            "features": {"placement": PLACEMENT_FEATURES, "order": ORDER_FEATURES},
            "width": width, "widths": list(WIDTH_CHOICES)}
    return {**body, "sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()}


def _probabilities(weights: list[float], features: list[float], rows: int) -> list[float]:
    scores = [sum(w * x for w, x in zip(weights[r * ORDER_FEATURES:(r + 1) * ORDER_FEATURES], features))
              for r in range(rows)]
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    return [e / sum(exps) for e in exps]


def _views(waiting: list[Job]) -> list[View]:
    return [View(j.max_tokens, j.long_prompt) for j in waiting]


def _deciders(policy: Policy, capacity: int) -> tuple[Callable, Callable]:
    return (lambda waiting: policy.uses_sjf(_views(waiting)),
            lambda waiting: min(capacity, policy.best_width(_views(waiting)) or capacity))


def _fixed(width: int, sjf: bool) -> tuple[Callable, Callable]:
    """The reference schedule's deciders: its own width (any, not only a catalogue width) and order."""
    return (lambda waiting: sjf), (lambda waiting: width)


def resample(arrivals: list[list[Any]], rng: random.Random, size: int, scale: tuple[float, float]) -> list[Job]:
    """A window of `size` consecutive recorded arrivals, its gaps scaled by one factor drawn from `scale`
    (lighter and heavier load than recorded), as model jobs. An arrival is [offset s, prompt tokens,
    long prompt, output limit, tokens made]."""
    start = rng.randrange(max(1, len(arrivals) - size + 1))
    window = arrivals[start:start + size]
    factor = rng.uniform(*scale)
    t0 = window[0][0]
    return [Job(k, (a[0] - t0) * factor, int(a[3]), bool(a[2]), int(a[1]), max(1, int(a[4])))
            for k, a in enumerate(window)]


def _arms(capacity: int) -> dict[str, dict[str, Any]]:
    return {f"w{w}/{'sjf' if s else 'fifo'}": fixed_policy(w, s)
            for w in WIDTH_CHOICES if w <= max(1, capacity) for s in (False, True)}


def _mean_cost(params: dict[str, float], windows: list[list[Job]], deciders: tuple[Callable, Callable],
               capacity: int, engine_width: int) -> float:
    sjf, width = deciders
    return statistics.fmean(simulate(params, [Job(j.index, j.arrive, j.max_tokens, j.long_prompt, j.prompt_tokens,
                                                  j.tokens) for j in window], sjf, width, capacity, engine_width)
                            for window in windows)


def plan(payload: dict[str, Any]) -> dict[str, Any]:
    """Step 2: train a candidate in the model of this server. Pure and deterministic for its payload, so it
    runs in a planner process and a test alike."""
    started = time.process_time()
    settings = {**PLANNER, **(payload.get("settings") or {})}
    params, arrivals = payload["model"], payload["arrivals"]
    capacity, engine_width = int(payload["capacity"]), int(payload.get("engine_width", ENGINE_WIDTH))
    rng = random.Random(int(payload.get("seed", 0)))
    size, scale = int(settings["trace"]), tuple(settings["scale"])
    validation = [resample(arrivals, random.Random(30_000 + k), size, scale) for k in range(settings["validation"])]
    held_out = [resample(arrivals, random.Random(50_000 + k), size, scale) for k in range(settings["held_out"])]
    arms = _arms(capacity)
    allowed = [w for w in WIDTH_CHOICES if w <= max(1, capacity) and w in payload.get("widths", WIDTH_CHOICES)]
    arms = {name: arm for name, arm in arms.items() if int(name[1:].split("/")[0]) in allowed}
    arm_costs = {name: _mean_cost(params, validation, _deciders(Policy.from_json(arm), capacity), capacity,
                                  engine_width) for name, arm in arms.items()}
    guard_name = min(arm_costs, key=lambda n: (arm_costs[n], n))
    guard = Policy.from_json(arms[guard_name])
    excluded = set(payload.get("excluded") or ())  # behaviours rejected against this incumbent or withdrawn
    open_arms = [n for n in arm_costs if behaviour(arms[n]) not in excluded]
    fixed_name = min(open_arms, key=lambda n: (arm_costs[n], n)) if open_arms else None
    runs = []
    for seed in range(int(settings["seeds"])):
        weights, curve = _train(params, arrivals, capacity, engine_width, guard, allowed, settings,
                                random.Random(rng.randrange(1 << 30) + seed))
        candidate = Policy.from_json(policy_json([0.0] * PLACEMENT_FEATURES, weights[0], weights[1]))
        runs.append((_mean_cost(params, validation, _deciders(candidate, capacity), capacity, engine_width), seed,
                     weights, curve))
    best = min(runs, key=lambda r: (r[0], r[1]))
    learned = policy_json([0.0] * PLACEMENT_FEATURES, best[2][0], best[2][1])
    reference = payload["reference"]
    rivals = {"learned": learned}
    if fixed_name is not None:
        rivals["best_fixed"] = arms[fixed_name]
    if payload.get("incumbent") is not None:
        rivals["incumbent"] = payload["incumbent"]
    costs = {name: _mean_cost(params, held_out, _deciders(Policy.from_json(p), capacity), capacity, engine_width)
             for name, p in rivals.items()}
    costs["reference"] = _mean_cost(params, held_out, _fixed(min(capacity, int(reference["width"])),
                                                             bool(reference["sjf"])), capacity, engine_width)
    eligible = [name for name in ("learned", "best_fixed") if name in rivals and behaviour(rivals[name]) not in excluded]
    choice = min(eligible, key=lambda name: (costs[name], name)) if eligible else None
    incumbent = costs.get("incumbent", costs["reference"])
    proposed = rivals[choice] if choice is not None and costs[choice] < incumbent else None
    step = max(1, len(best[3]) // 50)
    return {"schema": PLAN_SCHEMA, "candidate": proposed, "choice": choice if proposed else None,
            "best_fixed": fixed_name, "guard": guard_name, "excluded": sorted(excluded), "model_costs": costs,
            "arm_costs": arm_costs,
            "model_gain_vs_incumbent": 1 - costs[choice] / incumbent if choice is not None and incumbent else 0.0,
            "curve": [statistics.fmean(best[3][k:k + step]) for k in range(0, len(best[3]), step)],
            "episodes": int(settings["episodes"]) * int(settings["seeds"]), "widths": allowed,
            "cpu_s": time.process_time() - started}


def _train(params: dict[str, float], arrivals: list[list[Any]], capacity: int, engine_width: int, guard: Policy,
           allowed: list[int], settings: dict[str, Any], rng: random.Random) -> tuple[tuple[list, list], list]:
    """REINFORCE with a running-mean baseline (as `tools/rl4_agent.train`), order and width heads only; each
    episode also pays the harm weight times what it lost to the guard schedule on the same traffic (RL8)."""
    order = [0.0] * ORDER_FEATURES
    width = [0.0] * (len(WIDTH_CHOICES) * ORDER_FEATURES)
    for r, choice in enumerate(WIDTH_CHOICES):
        if choice not in allowed:
            width[r * ORDER_FEATURES] = -50.0  # a width the backend cannot run is never sampled
    guard_sjf, guard_width = _deciders(guard, capacity)
    baseline, curve = None, []
    lr, harm = float(settings["lr"]), float(settings["harm_weight"])
    for _ in range(int(settings["episodes"])):
        jobs = resample(arrivals, rng, int(settings["trace"]), tuple(settings["scale"]))
        grad_o = [0.0] * ORDER_FEATURES
        grad_w = [0.0] * len(width)

        def sjf(waiting: list[Job]) -> bool:
            f = order_features(_views(waiting))
            z = sum(w * x for w, x in zip(order, f))
            q = 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))
            chosen = rng.random() < q
            for k in range(ORDER_FEATURES):
                grad_o[k] += ((1.0 if chosen else 0.0) - q) * f[k]
            return chosen

        def pick(waiting: list[Job]) -> int:
            f = order_features(_views(waiting))
            p = _probabilities(width, f, len(WIDTH_CHOICES))
            chosen = rng.choices(range(len(p)), weights=p)[0]
            for c in range(len(p)):
                for k in range(ORDER_FEATURES):
                    grad_w[c * ORDER_FEATURES + k] += ((1.0 if c == chosen else 0.0) - p[c]) * f[k]
            return WIDTH_CHOICES[chosen]

        copy = [Job(j.index, j.arrive, j.max_tokens, j.long_prompt, j.prompt_tokens, j.tokens) for j in jobs]
        value = -simulate(params, copy, sjf, pick, capacity, engine_width)
        rival = simulate(params, jobs, guard_sjf, guard_width, capacity, engine_width)
        value -= harm * max(0.0, -value - rival)
        if baseline is None:
            baseline = value
        advantage = (value - baseline) / max(1e-6, abs(baseline))
        baseline += 0.05 * (value - baseline)
        order = [w + lr * advantage * g for w, g in zip(order, grad_o)]
        width = [w + lr * advantage * g for w, g in zip(width, grad_w)]
        curve.append(-value)
    return (order, width), curve


# ---------------------------------------------------------------- qualification

def window_cost(rows: list[dict[str, Any]], deadline_s: float) -> float:
    """Mean request cost of one replayed window: time in system plus time to first token, or the penalty."""
    costs = []
    for row in rows:
        if row.get("outcome") != "ok":
            costs.append(2.0 * deadline_s + 1.0)
        else:
            first = row["ttft_s"] if row.get("ttft_s") is not None else row["e2e_s"]
            costs.append(row["e2e_s"] + first)
    return statistics.fmean(costs) if costs else 0.0


def _nearest_rank(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)] if ordered else 0.0


def _paired(pairs: list[tuple[float, float]], resamples: int,
            quantiles: tuple[float, ...] = (0.025, 0.975)) -> tuple[float, list[float]]:
    """1 - sum(x) / sum(y) over the pairs, with bootstrap quantiles (seed 0), as KP1's estimator."""
    rng = random.Random(0)
    boot = []
    for _ in range(resamples):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        boot.append(1 - sum(x for x, _ in sample) / sum(y for _, y in sample))
    boot.sort()
    return (1 - sum(x for x, _ in pairs) / sum(y for _, y in pairs),
            [boot[min(resamples - 1, max(0, int(q * resamples) - (1 if q > 0.5 else 0)))] for q in quantiles])


def judge(triplets: list[dict[str, float]], resamples: int = RESAMPLES) -> dict[str, Any]:
    """The candidate's gain over the incumbent (B against the mean of A and AA, paired by triplet), against
    the floor: the size of what the same estimator finds between the incumbent's two runs (AA against A).
    Judged only at the `LOOKS` (group-sequential): `pass` when the gain's lower bound at that look's share of
    `ALPHAS` (2.5 % in all, strict early and lenient late) is above zero and the floor; `fail` when the upper
    97.5 % bound cannot reach the floor, or the last look passed; else `open`. Judging after every triplet with
    an uncorrected bound let a null candidate pass (AG1 S, run 5); an even split cost too much power at the last
    look (run 6). The 95 % interval and the A/A noise (nearest-rank 95th percentile of
    |1 - AA/A|) are reported beside it."""
    n = len(triplets)
    alpha = _alpha(n)
    gain, (bound, low, high) = _paired([(t["b"], (t["a"] + t["aa"]) / 2) for t in triplets], resamples,
                                       (alpha, 0.025, 0.975))
    control, _ = _paired([(t["aa"], t["a"]) for t in triplets], resamples)
    floor = max(0.0, abs(control))
    verdict = "open"
    if n in LOOKS or n >= MAX_TRIPLETS:
        if bound > floor:
            verdict = "pass"
        elif high <= floor or n >= MAX_TRIPLETS:
            verdict = "fail"
    return {"verdict": verdict, "triplets": n, "gain": gain, "gain_ci95": [low, high], "bound": bound,
            "floor": floor, "noise": _nearest_rank([abs(1 - t["aa"] / t["a"]) for t in triplets], 0.95)}


def _alpha(n: int) -> float:
    """The one-sided error spent at a look after `n` triplets (the last look's between looks and beyond)."""
    return ALPHAS[LOOKS.index(n)] if n in LOOKS else ALPHAS[-1]


def harm(triplets: list[dict[str, float]], resamples: int = RESAMPLES) -> dict[str, Any]:
    """An audit of the serving schedule (B) against the reference (A, AA): `harm` when, at a look, the upper
    bound of its gain at 1 minus that look's alpha is below zero: it loses to the reference with the corrected
    bound."""
    gain, (low, high) = _paired([(t["b"], (t["a"] + t["aa"]) / 2) for t in triplets], resamples,
                                (0.025, 1 - _alpha(len(triplets))))
    n = len(triplets)
    verdict = "open"
    if n in LOOKS or n >= MAX_TRIPLETS:
        verdict = "harm" if high < 0 else ("confirmed" if n >= MAX_TRIPLETS else "open")
    return {"verdict": verdict, "triplets": n, "gain": gain, "low": low, "high": high}


def compile_table(policy: Policy) -> dict[tuple[int, int], tuple[bool, int | None]]:
    """Every decision the policy can make for a queue of up to `TABLE_QUEUE` requests. Its features see only how
    many wait and how many of them are long (`order_features`), so (waiting, long) keys every decision exactly,
    and serving one is a dictionary lookup instead of two dot products and a softmax."""
    table = {}
    for n in range(1, TABLE_QUEUE + 1):
        for longs in range(n + 1):
            views = [View(LONG_TOKENS, False)] * longs + [View(1, False)] * (n - longs)
            table[n, longs] = (policy.uses_sjf(views), policy.best_width(views))
    return table


def behaviour(policy_body: dict[str, Any]) -> str:
    """What a policy does, as a digest of its decisions on every queue of 1-16 requests with any number of long
    ones: two learned policies with different weights but the same decisions are the same schedule, so one
    that lost is not tried again under a new digest."""
    policy = Policy.from_json(policy_body)
    decisions = []
    for n in range(1, 17):
        for longs in range(n + 1):
            views = [View(LONG_TOKENS * 2, False)] * longs + [View(LONG_TOKENS // 4, False)] * (n - longs)
            decisions.append([policy.uses_sjf(views), policy.best_width(views)])
    return hashlib.sha256(json.dumps(decisions).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- the agent in the server

class Agent:
    """The decision owner of a service's schedule. The service reads `active`, `sjf` and `width` exactly as it
    reads an LD1 dispatch, feeds `observe_group` after every group and `busy` when work arrives, and calls
    `idle` while nothing waits."""

    def __init__(self, root: str | Path, binding: dict[str, Any], *, capacity: int, reference: dict[str, Any],
                 engine_width: int = ENGINE_WIDTH, deadline_s: float = 120.0, device: str | None = None,
                 clock: Callable[[], float] = time.monotonic, planner_command: list[str] | None = None,
                 planner_settings: dict[str, Any] | None = None):
        self.root = Path(root)
        self.binding = dict(binding)
        self.binding_sha256 = hashlib.sha256(json.dumps(self.binding, sort_keys=True).encode()).hexdigest()
        self.path = self.root / f"state-{self.binding_sha256[:16]}.json"
        self.capacity, self.reference, self.engine_width = max(1, capacity), dict(reference), engine_width
        self.deadline_s, self.device, self.clock = deadline_s, device, clock
        self.planner_command = planner_command or [sys.executable, "-m", "ironmule_product.agent", "plan"]
        self.planner_settings = dict(planner_settings or {})
        self._planner: subprocess.Popen | None = None
        self._planner_files: tuple[Path, Path] | None = None
        self._planner_stopped = False
        self._plan_since = 0.0
        self._last_activity = clock()
        self._dirty_since: float | None = None
        self._lock = threading.Lock()
        self._save_lock = threading.Lock()  # one writer of the state file at a time (idle and close)
        self._closed = False
        self._table: dict[tuple[int, int], tuple[bool, int | None]] | None = None  # the serving policy, compiled
        self._table_sha256: str | None = None
        self.reason: str | None = None
        self._reset()
        self._load()

    def _reset(self) -> None:
        """Everything this binding learns, at its start."""
        self.world = WorldModel(self.engine_width)
        self.traffic: deque[list[Any]] = deque(maxlen=MAX_TRAFFIC)  # [arrive s, prompt tokens, long, limit, tokens]
        self.policy: Policy | None = None
        self.evidence: dict[str, Any] | None = None
        self.candidate: dict[str, Any] | None = None  # {"policy", "plan", "since", "triplets", "used"}
        self.withdrawn: list[dict[str, Any]] = []
        self.history: deque[dict[str, Any]] = deque(maxlen=50)
        self.explored: dict[int, int] = {}
        # behaviours rejected against each incumbent ("reference" while none serves), until the model reopens
        self.rejected: dict[str, list[str]] = {}
        self.audit: dict[str, Any] | None = None  # the serving schedule replayed against the reference
        self.rejected_epoch = 0
        self.last_plan: dict[str, Any] | None = None
        self.paused = False
        self.errors = 0
        self.decisions = {"order": 0, "width": 0}
        self.decision_ns = 0
        self.counters = {"groups": 0, "replayed_groups": 0, "plans": 0, "explorations": 0, "triplets": 0,
                         "qualified": 0, "rejected": 0, "withdrawn": 0}
        self.arrivals_at_plan = 0
        self.arrivals = 0
        self._monitor: list[bool] = []
        self._failed_windows = 0

    # ---- decisions (the service's scheduler thread; microseconds) ----

    @property
    def active(self) -> bool:
        return self.policy is not None and not self.paused

    def _decide(self, kind: str, decide: Callable[[Policy], Any]) -> Any:
        policy = self.policy
        if policy is None or self.paused:
            return None
        started = time.perf_counter_ns()
        try:
            value = decide(policy)
        except Exception as exc:  # noqa: BLE001 - a policy fault never stops serving
            with self._lock:
                self.errors += 1
                if self.errors >= MAX_ERRORS:
                    self._withdraw(f"{self.errors} errors while deciding (last: {exc!r})")
            return None
        with self._lock:
            self.decisions[kind] += 1
            self.decision_ns += time.perf_counter_ns() - started
        return value

    def _lookup(self, policy: Policy, waiting: list[Any]) -> tuple[bool, int | None]:
        """The policy's (sjf, width) for this queue: from the compiled table when it holds the queue."""
        if self._table is None or self._table_sha256 != policy.sha256:
            self._table, self._table_sha256 = compile_table(policy), policy.sha256
        entry = self._table.get((len(waiting), sum(1 for r in waiting if r.max_tokens >= LONG_TOKENS)))
        if entry is not None:
            return entry
        views = [View.of(r) for r in waiting]
        return policy.uses_sjf(views), policy.best_width(views)

    def sjf(self, waiting: list[Any]) -> bool | None:
        return self._decide("order", lambda p: self._lookup(p, waiting)[0])

    def width(self, waiting: list[Any]) -> int | None:
        def decide(policy: Policy) -> int | None:
            width = self._lookup(policy, waiting)[1]
            return None if width is None else min(self.capacity, width)  # a width-less policy keeps the service's
        return self._decide("width", decide)

    def card(self, request: Any, cards: list[list[Any]]) -> int | None:
        return None  # placement across GPUs stays with the pool's dispatch

    # ---- observation ----

    def busy(self) -> None:
        """Work arrived: replay never starts now, and on a CPU the planner yields the cores to serving."""
        self._last_activity = self.clock()
        if self.device == "cpu" and self._planner is not None and not self._planner_stopped:
            try:
                self._planner.send_signal(signal.SIGSTOP)
                self._planner_stopped = True
            except (OSError, ValueError):
                pass

    def observe_group(self, members: list[dict[str, Any]], replay: bool = False) -> None:
        """A finished group (start, first, end s; prompt_tokens, prompt_chars, max_tokens, tokens, ok). Live groups
        also feed the traffic memory and the live monitor; replayed ones teach the world model only."""
        with self._lock:
            if not replay:  # a replayed group is the agent's own idle work, not demand
                self._last_activity = self.clock()
            if self.world.observe(members):
                self.counters["replayed_groups" if replay else "groups"] += 1
            if not replay:
                for m in members:
                    if m.get("ok"):
                        self.traffic.append([m["arrive"], int(m["prompt_tokens"]),
                                             int(m.get("prompt_chars", 0)) > LONG_PROMPT_CHARS, int(m["max_tokens"]),
                                             int(m["tokens"])])
                        self.arrivals += 1
                    if self.active:
                        self._monitor.append(bool(m.get("ok")))
                self._watch()
            self._dirty()

    def _watch(self) -> None:
        """Two consecutive live windows with a failed request withdraw the serving policy (ordinary
        observations may withdraw, never qualify)."""
        while len(self._monitor) >= MONITOR_WINDOW:
            window, self._monitor = self._monitor[:MONITOR_WINDOW], self._monitor[MONITOR_WINDOW:]
            self._failed_windows = self._failed_windows + 1 if not all(window) else 0
            if self._failed_windows >= 2:
                self._withdraw("two consecutive live windows with failed requests")
                self._failed_windows = 0

    # ---- idle work (scheduler thread, nothing waiting) ----

    def idle(self, preempted: Callable[[], bool], replay: Callable | None = None) -> None:
        """Collect a finished plan or start one, explore an unpriced width, run one qualification triplet (or,
        with no candidate, one audit triplet of the serving schedule), save. Every step returns as soon as
        `preempted()` says demand arrived."""
        if self._closed:
            return
        if self._planner is not None and self._planner_stopped:
            try:
                self._planner.send_signal(signal.SIGCONT)
            except (OSError, ValueError):
                pass
            self._planner_stopped = False
        self._collect_plan()
        if not self.paused:
            quiet = self.clock() - self._last_activity >= QUIET_S
            if quiet and replay is not None and not preempted():
                if not self._explore(replay, preempted) and not preempted():
                    if self.candidate is not None:
                        self._qualify(replay, preempted)
                    else:
                        self._audit(replay, preempted)
            if self._planner is None and self.candidate is None:
                self._start_plan()
        self.save()

    def _explore(self, replay: Callable, preempted: Callable[[], bool]) -> bool:
        """One replay window at a width the model has not priced yet, every recorded arrival of the window
        queued at once so that the groups are that wide (a measurement, not a comparison); at most
        `MAX_EXPLORATIONS` per width. Returns whether it ran one."""
        unpriced = [w for w in WIDTH_CHOICES if w <= self.capacity and not self.world.supported(w)
                    and self.explored.get(w, 0) < MAX_EXPLORATIONS]
        trace = self._trace(self.traffic, since=None)
        if not unpriced or trace is None:
            return False
        width = unpriced[0]
        result = replay([(0.0, chars, tokens) for _, chars, tokens in trace],
                        lambda count, waiting, rate, now: (width, 0, False), preempted)
        if result is not None:
            with self._lock:
                self.explored[width] = self.explored.get(width, 0) + 1
                self.counters["explorations"] += 1
                self._event("explored", width=width)
                self._dirty()
        return True

    def _trace(self, arrivals: Iterable[list[Any]], since: float | None,
               skip: int = 0) -> list[tuple[float, int, int]] | None:
        """TRACE consecutive recorded arrivals (after `since`, past the first `skip`) as the replay window's
        (offset s, prompt characters, output limit), with their recorded gaps: the service skips the idle
        stretches instead. Capping the gaps, as RSI3-LIVE does, turned quiet traffic into bursts, and
        capping output limits at 128 halved the long requests, so qualification favoured a schedule that
        loses on the recorded traffic (AG1 S, run 3). Prompt characters stand in for the recorded tokens at
        about four characters a token, and long prompts keep their class."""
        rows = sorted(a for a in arrivals if since is None or a[0] > since)[skip:skip + TRACE]
        if len(rows) < TRACE:
            return None
        out = []
        for a in rows:
            chars = max(LONG_PROMPT_CHARS + 1, a[1] * 4) if a[2] else min(LONG_PROMPT_CHARS, max(1, a[1] * 4))
            out.append((round(a[0] - rows[0][0], 6), chars, min(int(a[3]), MAX_REPLAY_TOKENS)))
        return out

    def _arm(self, policy: Policy | None) -> Callable:
        """A replay window's choice per group: the reference schedule, or a policy's order and width."""
        if policy is None:
            width, sjf = int(self.reference["width"]), bool(self.reference["sjf"])
            return lambda count, waiting, rate, now: (width, int(self.reference.get("gather_ms", 0)), sjf)

        def choose(count: int, waiting: list[tuple[int, Any]], rate: float, now: int) -> tuple[int, int, bool]:
            views = [View.of(request) for _, request in waiting]
            return min(self.capacity, policy.best_width(views) or self.capacity), 0, policy.uses_sjf(views)
        return choose

    def _triplet(self, state: dict[str, Any], replay: Callable, preempted: Callable[[], bool],
                 a: Policy | None, b: Policy | None) -> dict[str, Any] | None:
        """Run what is left of one A, B, AA triplet (always in that order, so B sits in the middle of any drift)
        on the next held-out trace of `state` (arrivals recorded after `state["since"]`). An arm that demand cut
        runs again in the next idle spell and the arms already run are kept (cost, whether a request failed, a
        digest per answer; AG1 S, run 2), bound to the serving schedule they ran against: if that changed or
        the agent was paused meanwhile, the kept arms are dropped. Returns the finished arms, else None."""
        trace = self._trace(self.traffic, since=state["since"], skip=state["used"] * TRACE)
        if trace is None:
            return None
        bound = [self._incumbent_key(), self.paused]
        done = state.setdefault("partial", {})
        if done.get("bound", bound) != bound:
            done.clear()
        done["bound"] = bound
        for arm in ("a", "b", "aa"):
            if arm in done:
                continue
            result = replay(trace, self._arm(b if arm == "b" else a), preempted)
            with self._lock:
                if [self._incumbent_key(), self.paused] != bound:
                    state["partial"] = {}  # the serving schedule changed under this triplet
                    return None
                if result is None:  # demand came first: this arm runs again in the next idle spell
                    state["preempted"] = state.get("preempted", 0) + 1
                    if state["preempted"] >= MAX_PREEMPTIONS:  # this window never fits the idle spells
                        state["used"], state["preempted"], state["partial"] = state["used"] + 1, 0, {}
                    self._dirty()
                    return None
                done[arm] = {"cost": window_cost(result["rows"], self.deadline_s),
                             "failed": any(row.get("outcome") != "ok" for row in result["rows"]),
                             "answers": [hashlib.sha256(text.encode()).hexdigest()[:16] for text in result["texts"]]}
                state["preempted"] = 0
                self._dirty()
        with self._lock:
            state["used"], state["preempted"], state["partial"] = state["used"] + 1, 0, {}
            return done

    def _qualify(self, replay: Callable, preempted: Callable[[], bool]) -> None:
        """One A/B/AA triplet of the candidate (B) against the serving schedule on arrivals recorded after the
        candidate was planned; judged at the looks."""
        candidate = self.candidate
        if candidate is None:
            return
        done = self._triplet(candidate, replay, preempted, self.policy, Policy.from_json(candidate["policy"]))
        if done is None:
            return
        with self._lock:
            if self.candidate is not candidate:
                return
            self.counters["triplets"] += 1
            if not (done["a"]["answers"] == done["b"]["answers"] == done["aa"]["answers"]):
                self._reject("a replayed answer differed between the arms")
                return
            if done["b"]["failed"]:
                self._reject("a replayed request failed under the candidate")
                return
            if done["a"]["failed"] or done["aa"]["failed"]:
                return  # a failure under the incumbent says nothing about the candidate: the triplet is not counted
            candidate["triplets"].append({k: done[k]["cost"] for k in ("a", "b", "aa")})
            verdict = judge(candidate["triplets"])
            candidate["verdict"] = verdict
            if verdict["verdict"] == "pass":
                self._promote(candidate, verdict)
            elif verdict["verdict"] == "fail":
                self._reject(f"its gain {verdict['gain']:.1%} (95 % {verdict['gain_ci95'][0]:.1%} to "
                             f"{verdict['gain_ci95'][1]:.1%}) did not clear the A/A floor {verdict['floor']:.1%}")
            self._dirty()

    def _audit(self, replay: Callable, preempted: Callable[[], bool]) -> None:
        """While no candidate waits, the serving schedule (B) is replayed against the reference (A, AA) on
        arrivals recorded after it began serving, up to `MAX_TRIPLETS` per schedule. If it loses with the
        corrected bound it is withdrawn: comparative evidence may withdraw, as ordinary observations may.
        This also catches a schedule that has become slower without failing anything."""
        audit = self.audit
        if self.policy is None or audit is None or audit.get("incumbent") != self.policy.sha256 \
                or audit.get("verdict") in ("harm", "confirmed"):
            return
        done = self._triplet(audit, replay, preempted, None, self.policy)
        if done is None:
            return
        with self._lock:
            if self.audit is not audit or self.policy is None or self.policy.sha256 != audit["incumbent"]:
                return
            if any(done[arm]["failed"] for arm in ("a", "aa")) or \
                    not (done["a"]["answers"] == done["b"]["answers"] == done["aa"]["answers"]):
                return  # not a valid audit triplet; a differing answer is the candidate check's business
            audit["triplets"].append({k: done[k]["cost"] for k in ("a", "b", "aa")})
            verdict = harm(audit["triplets"])
            audit["verdict"] = verdict["verdict"]
            self.counters["audits"] = self.counters.get("audits", 0) + 1
            if verdict["verdict"] == "harm":
                self._withdraw(f"it lost to the reference in replays: {verdict['gain']:.1%} "
                               f"(bound {verdict['high']:.1%})")
            elif verdict["verdict"] == "confirmed":
                self._event("audited", policy=audit["incumbent"][:12], gain=verdict["gain"], low=verdict["low"])
            self._dirty()

    def _promote(self, candidate: dict[str, Any], verdict: dict[str, Any]) -> None:
        policy = Policy.from_json(candidate["policy"])
        body = {"schema": EVIDENCE_SCHEMA, "policy_sha256": policy.sha256, "binding_sha256": self.binding_sha256,
                "incumbent_sha256": None if self.policy is None else self.policy.sha256, "verdict": "pass",
                "gain": verdict["gain"], "gain_ci95": verdict["gain_ci95"], "floor": verdict["floor"],
                "noise": verdict["noise"],
                "triplets": candidate["triplets"], "plan": {k: candidate["plan"].get(k) for k in
                                                            ("choice", "best_fixed", "model_costs", "episodes")},
                "unix": time.time()}
        self.policy, self.evidence = policy, {**body, "sha256": hashlib.sha256(
            json.dumps(body, sort_keys=True).encode()).hexdigest()}
        self.candidate, self.errors, self._monitor, self._failed_windows = None, 0, [], 0
        self.audit = {"incumbent": policy.sha256, "since": max((a[0] for a in self.traffic), default=0.0),
                      "used": 0, "triplets": [], "verdict": "open"}
        self.counters["qualified"] += 1
        self.arrivals_at_plan = self.arrivals
        self._event("qualified", policy=policy.sha256[:12], gain=verdict["gain"], ci95=verdict["gain_ci95"],
                    floor=verdict["floor"], choice=candidate["plan"].get("choice"))

    def _incumbent_key(self) -> str:
        return "reference" if self.policy is None else self.policy.sha256

    def _excluded(self) -> list[str]:
        """Behaviours the planner may not propose: what lost to the serving schedule already, and anything
        withdrawn. Proposing a loser again and again, under its own digest or a learned twin's, until one
        comparison passes by chance would turn noise into a qualification (AG1 S, runs 4 and 5). A reopened
        world model (the machine changed) clears what lost, not what was withdrawn."""
        if self.rejected_epoch != self.world.reopened:
            self.rejected, self.rejected_epoch = {}, self.world.reopened
        withdrawn = [w["behaviour"] for w in self.withdrawn if w.get("behaviour")]
        return sorted(set(self.rejected.get(self._incumbent_key(), ())) | set(withdrawn))

    def _reject(self, reason: str) -> None:
        if self.candidate is not None:
            self._excluded()
            self.rejected.setdefault(self._incumbent_key(), []).append(behaviour(self.candidate["policy"]))
        self.counters["rejected"] += 1
        self._event("rejected", policy=self.candidate["policy"]["sha256"][:12] if self.candidate else None,
                    reason=reason)
        self.candidate = None
        self.arrivals_at_plan = self.arrivals

    def withdraw(self, reason: str) -> None:
        with self._lock:
            self._withdraw(reason)
            self._dirty()

    def _withdraw(self, reason: str) -> None:
        if self.policy is None:
            return
        try:
            signature = behaviour(_policy_body(self.policy))
        except Exception:  # noqa: BLE001 - a policy too broken to describe is still withdrawn by its digest
            signature = None
        self.withdrawn.append({"policy_sha256": self.policy.sha256, "behaviour": signature, "reason": reason,
                               "unix": time.time()})
        self.audit = None
        self.counters["withdrawn"] += 1
        self._event("withdrawn", policy=self.policy.sha256[:12], reason=reason)
        self.policy, self.evidence, self.reason = None, None, f"withdrawn: {reason}"

    def control(self, action: str) -> dict[str, Any]:
        """`off` withdraws the serving policy and pauses learning; `on` resumes learning (a withdrawn policy
        does not come back: only a new qualification serves again)."""
        if action not in ("off", "on"):
            raise ValueError('action must be "off" or "on"')
        with self._lock:
            if action == "off":
                self._withdraw("the operator turned it off")
                self.paused = True
            else:
                self.paused = False
            self._event("operator", action=action)
            self._dirty()
        if action == "off":
            self._stop_planner()
        return self.status()

    # ---- the planner process ----

    def _start_plan(self) -> None:
        if self._closed:
            return
        fidelity = self.world.fidelity()
        if (sum(self.world.groups.values()) < MIN_GROUPS or len(self.traffic) < MIN_ARRIVALS
                or fidelity is None or fidelity > FIDELITY
                or (self.counters["plans"] and self.arrivals - self.arrivals_at_plan < REPLAN_AFTER)):
            return
        arrivals = sorted(self.traffic, key=lambda a: a[0])
        payload = {"model": self.world.params(), "arrivals": arrivals, "capacity": self.capacity,
                   "engine_width": self.engine_width, "seed": self.counters["plans"],
                   "widths": [w for w in WIDTH_CHOICES if w <= self.capacity and self.world.supported(w)],
                   "reference": self.reference, "settings": self.planner_settings, "excluded": self._excluded(),
                   "incumbent": None if self.policy is None else _policy_body(self.policy)}
        self.root.mkdir(parents=True, exist_ok=True)
        source, target = self.root / "plan-in.json", self.root / "plan-out.json"
        target.unlink(missing_ok=True)
        source.write_text(json.dumps(payload))
        try:
            self._planner = subprocess.Popen(self.planner_command + [str(source), str(target)],
                                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                             stderr=subprocess.DEVNULL, preexec_fn=_lower_priority)
        except OSError as exc:
            self.reason = f"the planner did not start: {exc}"
            return
        self._planner_files, self._plan_since = (source, target), arrivals[-1][0]
        self._event("planning", arrivals=len(arrivals), model=self.world.params())

    def _collect_plan(self) -> None:
        process = self._planner
        if process is None or process.poll() is None:
            return
        self._planner = None
        source, target = self._planner_files
        try:
            result = json.loads(target.read_text())
            if result.get("schema") != PLAN_SCHEMA:
                raise ValueError("unknown plan schema")
        except (OSError, ValueError) as exc:
            self._event("plan_failed", reason=str(exc), code=process.returncode)
            self.arrivals_at_plan = self.arrivals
            return
        with self._lock:
            self.counters["plans"] += 1
            self.arrivals_at_plan = self.arrivals
            proposed = result.get("candidate")
            if proposed is not None and (any(w["policy_sha256"] == proposed["sha256"] for w in self.withdrawn)
                                         or behaviour(proposed) in self._excluded()):
                proposed = None  # a withdrawn policy never serves again, a rejected one is not tried again
            self.last_plan = {k: result.get(k) for k in ("choice", "best_fixed", "excluded", "model_costs", "arm_costs",
                                                          "model_gain_vs_incumbent", "curve", "episodes", "cpu_s")}
            if proposed is not None:
                self.candidate = {"policy": proposed, "plan": self.last_plan, "since": self._plan_since,
                                  "triplets": [], "used": 0}
            self._event("planned", candidate=None if proposed is None else proposed["sha256"][:12],
                        choice=result.get("choice"), model_gain=result.get("model_gain_vs_incumbent"),
                        cpu_s=result.get("cpu_s"))
            self._dirty()

    def _stop_planner(self) -> None:
        process, self._planner = self._planner, None
        if process is not None and process.poll() is None:
            try:
                if self._planner_stopped:
                    process.send_signal(signal.SIGCONT)
                process.terminate()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()

    # ---- state ----

    def _event(self, kind: str, **fields: Any) -> None:
        self.history.append({"event": kind, "unix": round(time.time(), 3), **fields})

    def _dirty(self) -> None:
        if self._dirty_since is None:
            self._dirty_since = self.clock()

    def save(self, force: bool = False) -> None:
        if self._dirty_since is None or (not force and self.clock() - self._dirty_since < SAVE_AFTER_S):
            return
        with self._save_lock:
            self._save()

    def _save(self) -> None:
        with self._lock:
            state = {"schema": STATE_SCHEMA, "binding": self.binding, "binding_sha256": self.binding_sha256,
                     "world": self.world.to_json(), "traffic": list(self.traffic),
                     "policy": None if self.policy is None else _policy_body(self.policy),
                     "evidence": self.evidence, "candidate": self.candidate, "withdrawn": self.withdrawn,
                     "history": list(self.history), "counters": self.counters, "paused": self.paused,
                     "arrivals": self.arrivals, "arrivals_at_plan": self.arrivals_at_plan,
                     "last_plan": self.last_plan, "explored": {str(k): v for k, v in self.explored.items()},
                     "rejected": self.rejected, "rejected_epoch": self.rejected_epoch, "audit": self.audit}
            self._dirty_since = None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            partial = self.path.with_name(f"{self.path.name}.partial-{os.getpid()}-{threading.get_ident()}")
            fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as stream:
                json.dump(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            partial.replace(self.path)
        except OSError as exc:
            self.reason = f"its state was not saved: {exc}"

    def _load(self) -> None:
        """Resume this binding's learning; a foreign, corrupt or unknown state starts fresh, and a serving
        policy resumes only with evidence that names it and this binding."""
        if not self.path.exists():
            return
        try:
            state = json.loads(self.path.read_text())
            if state.get("schema") != STATE_SCHEMA or state.get("binding_sha256") != self.binding_sha256:
                raise ValueError("it belongs to another setup")
            self.world = WorldModel.from_json(state["world"])
            self.traffic.extend(state["traffic"])
            self.withdrawn = list(state["withdrawn"])
            self.history.extend(state["history"])
            self.counters.update(state["counters"])
            self.paused = bool(state["paused"])
            self.arrivals, self.arrivals_at_plan = int(state["arrivals"]), int(state["arrivals_at_plan"])
            self.candidate, self.last_plan = state.get("candidate"), state.get("last_plan")
            self.explored = {int(k): int(v) for k, v in (state.get("explored") or {}).items()}
            self.rejected = {str(k): list(v) for k, v in (state.get("rejected") or {}).items()}
            self.rejected_epoch = int(state.get("rejected_epoch", 0))
            self.audit = state.get("audit")
            if state.get("policy") is not None:
                policy, evidence = Policy.from_json(state["policy"]), state.get("evidence") or {}
                body = {k: v for k, v in evidence.items() if k != "sha256"}
                if (hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != evidence.get("sha256")
                        or body.get("policy_sha256") != policy.sha256 or body.get("verdict") != "pass"
                        or body.get("binding_sha256") != self.binding_sha256
                        or any(w["policy_sha256"] == policy.sha256 for w in self.withdrawn)):
                    raise ValueError("its serving policy is not proven for this setup")
                self.policy, self.evidence = policy, evidence
        except (OSError, ValueError, KeyError, TypeError) as exc:
            try:
                self.path.replace(self.path.with_name(f"{self.path.name}.rejected-{int(time.time())}"))
            except OSError:
                pass
            self._reset()
            self.reason = f"its saved state was set aside ({exc}); learning starts fresh"

    def close(self) -> None:
        """Stop for good: no idle work or plan starts after this, the planner is stopped, the state saved."""
        self._closed = True
        self._stop_planner()
        self._dirty()
        self.save(force=True)

    def status(self) -> dict[str, Any]:
        with self._lock:
            decisions = sum(self.decisions.values())
            candidate = self.candidate
            return {
                "agent": "AG1", "active": self.active, "paused": self.paused, "reason": self.reason,
                "serving": "reference" if self.policy is None else f"learned:{self.policy.sha256[:12]}",
                "policy_sha256": None if self.policy is None else self.policy.sha256,
                "gain": None if self.evidence is None else self.evidence["gain"],
                "gain_ci95": None if self.evidence is None else self.evidence["gain_ci95"],
                "world_model": {**self.world.params(), "groups": dict(self.world.groups),
                                "median_relative_error": self.world.fidelity(), "reopened": self.world.reopened},
                "traffic": len(self.traffic), "planner": "running" if self._planner is not None else "idle",
                "last_plan": self.last_plan,
                "candidate": None if candidate is None else {
                    "policy_sha256": candidate["policy"]["sha256"], "choice": candidate["plan"].get("choice"),
                    "triplets": len(candidate["triplets"]), "verdict": candidate.get("verdict")},
                "audit": None if self.audit is None else {
                    "policy_sha256": self.audit.get("incumbent"), "triplets": len(self.audit.get("triplets", ())),
                    "verdict": self.audit.get("verdict")},
                "decisions": dict(self.decisions), "errors": self.errors,
                "decision_us": None if not decisions else self.decision_ns / decisions / 1000,
                "counters": dict(self.counters), "history": list(self.history)[-10:]}


def _policy_body(policy: Policy) -> dict[str, Any]:
    return policy_json(list(policy.placement), list(policy.order), list(policy.width or ()))


def _lower_priority() -> None:
    try:
        os.nice(19)
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    """`python -m ironmule_product.agent plan IN.json OUT.json`: the planner process."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3 or args[0] != "plan":
        print("usage: python -m ironmule_product.agent plan IN.json OUT.json", file=sys.stderr)
        return 2
    result = plan(json.loads(Path(args[1]).read_text()))
    partial = Path(args[2] + ".partial")
    partial.write_text(json.dumps(result))
    partial.replace(args[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
