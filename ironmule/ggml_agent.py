"""AG10: the IronMule agent inside the ggml engine -- it learns online, dreams, and switches the engine's configuration.

The ggml engine (`ggml_engine.py`) serves a GGUF with llama.cpp's kernels. Which configuration serves best -- how many
sequences run at once, an 8-bit or a 16-bit KV cache, and so which width fits beside the model -- depends on the
machine and on the traffic. SEQ1 chose the width once, at full load, by a fixed doubling rule. This agent chooses it
the way the project's main goal asks (BACKLOG, "The main goal"): it decides in microseconds, learns while it serves,
and serves only configurations that passed the quality gate on this machine.

* **World model, learned online.** Per configuration, a step's seconds are linear in what the step carries:
  `a + b * decode rows + c * prompt tokens / 1000 + d * attended KV tokens / 1000`, fitted by recursive least squares
  with forgetting (the AG1 `_RLS`) from every step the engine serves. The start values come from the engine's own
  measurement at one row and at full load (`ggml_engine.measure_widths`).
* **Traffic memory.** Every finished request leaves (gap since the previous arrival, prompt tokens, answer tokens);
  no text. Persisted per machine, model and engine version.
* **Dreaming.** While the engine is idle the agent replays windows of recorded traffic (gaps scaled by 0.5-1.5, as
  AG1 does) through a simulation of the engine's continuous batching -- prompts join a running step, one call per
  step, a finished request frees its sequence -- priced by each configuration's world model, with common random
  windows for all configurations. Objective (owner, 2026-11-03): tokens per second first, and the mean first-token
  time no worse than the incumbent's (tolerance `TTFT_TOLERANCE`).
* **Policy improvement and its guard.** A candidate replaces the incumbent only when the bootstrap lower bound
  (2.5 %) of its dreamed tokens-per-second gain over the windows is above `MARGIN` and its first tokens are not
  later. The switch happens in an idle spell (the context is rebuilt on the loaded model). After it, the live steps
  keep updating the candidate's world model; when a re-dream on the corrected model no longer shows a gain above
  zero, the agent reverts and records the candidate as rejected for this binding (it is tried again only after
  `RETRY_AFTER` new requests, since traffic can change). Every decision and its reason is in `status()`.
* **Decisions in microseconds.** The hot path reads one attribute (`current`) and appends one tuple per step; the
  least-squares updates and every simulation run in the idle thread.

Configurations differ in arithmetic only where the quality gate let them (an 8-bit KV cache serves only after it
passed `ggml_engine.gate` within the owner's bound); this module never widens what the gate allowed.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Sequence

VERSION = 2  # 2: prefill anchored, computed prompt tokens in the traffic memory
MEMORY = 4096  # recorded arrivals kept
WINDOW = 32  # arrivals per dreamed window
WINDOWS = 48  # dreamed windows per decision
MIN_ARRIVALS = 48  # dream only with this much recorded traffic
MARGIN = 0.03  # a switch needs a 2.5 % bound of the dreamed gain above 3 %
TTFT_TOLERANCE = 1.02  # mean first token at most 2 % later than the incumbent's
RETRY_AFTER = 512  # a rejected candidate may be dreamed again after this many new requests
FORGET = 0.995
RECENT = 256  # arrivals that count as the current traffic
LIVE_MIN = 200  # live steps before a configuration is priced by its own live model
FACTOR_MIN = 20  # live steps before its own live factor prices it
FEATURES = 4  # 1, decode rows, prompt tokens / 1000, attended KV tokens / 1000


class StepCost:
    """Recursive least squares for one configuration's step seconds; see the module docstring."""

    def __init__(self, theta: Sequence[float], spread: float = 1.0):
        self.theta = [float(v) for v in theta]
        self.spread = float(spread)
        self.p = [[self.spread if i == j else 0.0 for j in range(FEATURES)] for i in range(FEATURES)]
        self.n = 0

    @staticmethod
    def features(rows: int, prompt: int, kv: int) -> list[float]:
        return [1.0, float(rows), prompt / 1000.0, kv / 1000.0]

    def predict(self, rows: int, prompt: int, kv: int) -> float:
        x = self.features(rows, prompt, kv)
        return max(1e-5, sum(t * v for t, v in zip(self.theta, x)))

    def update(self, rows: int, prompt: int, kv: int, seconds: float) -> None:
        x = self.features(rows, prompt, kv)
        px = [sum(self.p[i][j] * x[j] for j in range(FEATURES)) for i in range(FEATURES)]
        denominator = FORGET + sum(x[i] * px[i] for i in range(FEATURES))
        if denominator <= 0 or not math.isfinite(seconds):
            return
        gain = [v / denominator for v in px]
        error = seconds - sum(t * v for t, v in zip(self.theta, x))
        theta = [t + g * error for t, g in zip(self.theta, gain)]
        p = [[(self.p[i][j] - gain[i] * px[j]) / FORGET for j in range(FEATURES)] for i in range(FEATURES)]
        p = [[(p[i][j] + p[j][i]) / 2 for j in range(FEATURES)] for i in range(FEATURES)]
        if not all(math.isfinite(v) for v in theta + [v for row in p for v in row]) or any(
                p[i][i] <= 0 for i in range(FEATURES)):
            self.p = [[self.spread if i == j else 0.0 for j in range(FEATURES)] for i in range(FEATURES)]
            return
        cap = [min(1.0, math.sqrt(self.spread / p[i][i])) for i in range(FEATURES)]  # variance at most its start
        self.p = [[p[i][j] * cap[i] * cap[j] for j in range(FEATURES)] for i in range(FEATURES)]
        self.theta, self.n = theta, self.n + 1

    def to_json(self) -> dict[str, Any]:
        return {"theta": self.theta, "p": self.p, "n": self.n, "spread": self.spread}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "StepCost":
        model = cls(data["theta"], data.get("spread", 1.0))
        model.p, model.n = [[float(v) for v in row] for row in data["p"]], int(data.get("n", 0))
        return model

    @classmethod
    def from_measurement(cls, width: int, step_one: float, step_full: float, prefill_per_1k: float,
                         kv_per_1k: float = 0.0) -> "StepCost":
        """Start values from the engine's own measurement: one row and `width` rows at a short context, and what a
        full step paid per thousand attended KV tokens at a longer one."""
        per_row = max(0.0, (step_full - step_one) / max(1, width - 1))
        return cls([max(1e-5, step_one - per_row), per_row, max(0.0, prefill_per_1k), max(0.0, kv_per_1k)])


class Scaled:
    """A start model times the machine factor: how a configuration that has not served enough is priced."""

    def __init__(self, base: StepCost, factor: float):
        self.base, self.factor = base, factor

    def predict(self, rows: int, prompt: int, kv: int) -> float:
        return self.base.predict(rows, prompt, kv) * self.factor


# --- the engine, simulated ---------------------------------------------------------------------------------------------

def simulate(cost: StepCost, width: int, arrivals: Sequence[tuple], seq_ctx: int = 2048,
             trace: list[tuple[int, int, int, float]] | None = None) -> dict[str, float]:
    """The engine's continuous batching on `arrivals` [(time, prompt tokens, answer tokens[, prompt tokens computed])],
    priced by `cost` (a prompt's kept prefix costs no prefill but is attended to): in each
    step every waiting request that has arrived takes a free sequence and its prompt goes into the step's one call
    (up to `seq_ctx` prompt tokens, at least one prompt), every running request gets one token, a finished request
    frees its sequence. Returns tokens, seconds from the first arrival to the last token, and mean first-token time."""
    pending = sorted((a[0], a[1], a[2], a[3] if len(a) > 3 else a[1]) for a in arrivals)
    i, now, free = 0, (pending[0][0] if pending else 0.0), width
    waiting: deque[tuple[float, int, int, int]] = deque()
    active: list[list[float]] = []  # [remaining tokens, context tokens]
    tokens, ttft = 0, []
    while i < len(pending) or waiting or active:
        while i < len(pending) and pending[i][0] <= now:
            waiting.append(pending[i])
            i += 1
        if not waiting and not active:
            now = pending[i][0]
            continue
        joined, prompt = [], 0
        while waiting and free and (not joined or prompt + waiting[0][3] <= seq_ctx):
            request = waiting.popleft()
            joined.append(request)
            prompt += request[3]
            free -= 1
        if not joined and not active:  # nothing fits until something frees (cannot happen with width >= 1)
            now = pending[i][0] if i < len(pending) else now + 1e-3
            continue
        kv = int(sum(a[1] for a in active))
        seconds = cost.predict(len(active), prompt, kv)
        now += seconds
        if trace is not None:
            trace.append((len(active), prompt, kv, seconds))
        survivors = []
        for job in active:
            job[0] -= 1
            job[1] += 1
            tokens += 1
            if job[0] > 0:
                survivors.append(job)
            else:
                free += 1
        for arrived, prompt_tokens, answer, _ in joined:
            ttft.append(now - arrived)
            tokens += 1
            if answer > 1:
                survivors.append([answer - 1, prompt_tokens + 1])
            else:
                free += 1
        active = survivors
    span = max(1e-9, now - (pending[0][0] if pending else 0.0))
    return {"tokens": tokens, "seconds": span, "tok_s": tokens / span,
            "ttft": sum(ttft) / len(ttft) if ttft else 0.0}


def windows(memory: Sequence[tuple[float, int, int]], rng: random.Random, count: int = WINDOWS,
            size: int = WINDOW) -> list[list[tuple[float, int, int]]]:
    """Windows of `size` consecutive recorded arrivals, gaps scaled by one factor in 0.5-1.5 per window; three in four
    start among the last `RECENT` arrivals, so a change in the traffic shows in the dream soon after it happens."""
    out = []
    last = max(1, len(memory) - size + 1)
    for _ in range(count):
        start = rng.randrange(max(0, last - RECENT), last) if rng.random() < 0.75 else rng.randrange(0, last)
        factor, t, window = rng.uniform(0.5, 1.5), 0.0, []
        for row in memory[start:start + size]:
            gap, prompt, answer = row[:3]
            t += gap * factor
            computed = int(row[3]) if len(row) > 3 else int(prompt)
            window.append((t, max(1, int(prompt)), max(1, int(answer)), max(1, min(int(prompt), computed))))
        out.append(window)
    return out


def dream(configs: dict[str, dict[str, Any]], costs: dict[str, StepCost], incumbent: str,
          memory: Sequence[tuple[float, int, int]], rng: random.Random, resamples: int = 1000) -> dict[str, Any]:
    """Every configuration on the same windows; per candidate the mean relative tokens-per-second gain over the
    incumbent with a bootstrap 2.5 % bound, and its mean first-token ratio."""
    sampled = windows(memory, rng)
    runs = {name: [simulate(costs[name], int(c["seqs"]), w, int(c.get("seq_ctx", 2048))) for w in sampled]
            for name, c in configs.items()}
    base = runs[incumbent]
    out: dict[str, Any] = {"windows": len(sampled), "candidates": {}}
    for name, rows in runs.items():
        gains = [r["tok_s"] / b["tok_s"] - 1.0 for r, b in zip(rows, base)]
        boots = sorted(sum(gains[rng.randrange(len(gains))] for _ in gains) / len(gains) for _ in range(resamples))
        ttft_b = sum(b["ttft"] for b in base) / len(base)
        ttft_c = sum(r["ttft"] for r in rows) / len(rows)
        out["candidates"][name] = {"gain": sum(gains) / len(gains), "low": boots[int(0.025 * resamples)],
                                   "ttft_ratio": (ttft_c / ttft_b) if ttft_b > 0 else 1.0,
                                   "tok_s": sum(r["tok_s"] for r in rows) / len(rows)}
    return out


# --- the agent ---------------------------------------------------------------------------------------------------------

class GgmlAgent:
    """See the module docstring. `configs`: name -> {"seqs", "settings", "seq_ctx", and the measured "step_one",
    "step_full", "prefill_per_1k"}; every configuration here has passed the quality gate."""

    def __init__(self, root: str | Path | None, binding: dict[str, Any], configs: dict[str, dict[str, Any]],
                 incumbent: str, seed: int = 0):
        if incumbent not in configs:
            raise ValueError("the incumbent must be one of the configurations")
        self.configs = {k: dict(v) for k, v in configs.items()}
        self.binding = dict(binding)
        key = hashlib.sha256(json.dumps([binding, sorted(configs), VERSION], sort_keys=True).encode()).hexdigest()[:20]
        self.path = Path(root) / "ggml-agent" / f"{key}.json" if root is not None else None
        self.costs = {name: StepCost.from_measurement(int(c["seqs"]), float(c["step_one"]), float(c["step_full"]),
                                                      float(c.get("prefill_per_1k", 0.0)), float(c.get("kv_per_1k", 0.0)))
                      for name, c in configs.items()}
        # the startup measurement's points, folded in at every idle call: live steps of one kind of traffic (all rows
        # busy at long contexts) cannot tell rows from attended tokens apart, and without these points a model fitted
        # there drifts where it has seen nothing (AG10 S, run 1: no way back to flash attention for short chats)
        self.anchors = {name: anchors_of(c) for name, c in configs.items()}
        # each configuration's start model alone (anchors only), and how the live machine compares with it: a model
        # fitted on live steps against one fitted on the startup measurement compares two machines, not two
        # configurations (seen live on 2026-11-13: the measured ones looked 8 % cheaper). A configuration with fewer
        # than LIVE_MIN live steps is priced as its start model times the factor the served ones show.
        self.base = {}
        for name, points in self.anchors.items():
            model = StepCost(self.costs[name].theta)
            for _ in range(40):
                for rows, prompt, kv, seconds in points:
                    model.update(rows, prompt, kv, seconds)
            self.base[name] = model
        self.live = {name: 0 for name in configs}
        self.factors = {name: 1.0 for name in configs}  # live seconds over the start model's, per configuration
        self.memory: deque[tuple[float, int, int]] = deque(maxlen=MEMORY)
        self.current = incumbent  # the hot path reads this
        self.trial: dict[str, Any] | None = None  # {"from": name, "since": requests}
        self.rejected: dict[str, int] = {}  # name -> request count at rejection
        self.requests = 0
        self.history: list[dict[str, Any]] = []
        self.pending_switch: str | None = None
        self._steps: list[tuple[str, int, int, int, float]] = []
        self._last_arrival: float | None = None
        self._lock = threading.Lock()
        self._rng = random.Random(seed)
        self._load()

    # hot path ---------------------------------------------------------------------------------------------------------
    def observe_step(self, rows: int, prompt: int, kv: int, seconds: float) -> None:
        self._steps.append((self.current, rows, prompt, kv, seconds))

    def observe_request(self, arrived: float, prompt: int, answer: int, computed: int | None = None) -> None:
        """One finished request: when it arrived, its prompt and answer length, and how many prompt tokens were
        computed (the rest came from a kept prefix)."""
        with self._lock:
            gap = 0.0 if self._last_arrival is None else max(0.0, arrived - self._last_arrival)
            self._last_arrival = arrived if self._last_arrival is None else max(self._last_arrival, arrived)
            self.memory.append((gap, int(prompt), int(answer), int(prompt if computed is None else computed)))
            self.requests += 1

    # idle work --------------------------------------------------------------------------------------------------------
    def learn(self) -> int:
        """Fold the buffered steps into the world models; returns how many."""
        steps, self._steps = self._steps, []
        for name, rows, prompt, kv, seconds in steps:
            if name in self.costs and (rows or prompt):
                self.costs[name].update(rows, prompt, kv, seconds)
                self.live[name] += 1
                expected = self.base[name].predict(rows, prompt, kv)
                if expected > 0 and seconds > 0:  # in log space, a step at a time
                    rate = 0.2 if self.live[name] <= 10 else 0.02
                    self.factors[name] = math.exp((1 - rate) * math.log(self.factors[name])
                                                  + rate * math.log(seconds / expected))
        for name, points in self.anchors.items():
            for rows, prompt, kv, seconds in points:
                self.costs[name].update(rows, prompt, kv, seconds)
        return len(steps)

    def idle(self) -> str | None:
        """Learn, dream, and decide. Returns a configuration to switch to, or None."""
        self.learn()
        with self._lock:
            memory = list(self.memory)
        if len(memory) < MIN_ARRIVALS or len(self.configs) < 2 or self.pending_switch is not None:
            return None
        prices = self.prices()
        result = dream(self.configs, prices, self.current, memory, self._rng)
        decision: dict[str, Any] = {"at": time.time(), "requests": self.requests, "incumbent": self.current,
                                    "dream": result}
        if self.trial is not None:  # the switch is on trial: keep it only while the corrected model still shows a gain
            back = self.trial["from"]
            since = memory[-(self.requests - self.trial["since"]):] if self.requests > self.trial["since"] else []
            if len(since) < 2 * WINDOW:  # judged on the traffic it served, not on what came before the switch
                decision.update(action="watch", reason="the switch is on trial; too little traffic since it")
                self._remember(decision)
                return None
            mine = dream({self.current: self.configs[self.current], back: self.configs[back]}, prices, back,
                         since, self._rng)["candidates"][self.current]
            if mine["gain"] <= 0.0 or mine["ttft_ratio"] > TTFT_TOLERANCE:
                self.rejected[self.current] = self.requests
                decision.update(action="revert", to=back, reason="the live steps corrected the model; no gain left")
                self.trial = None
                self.pending_switch = back
            elif self.requests - self.trial["since"] >= 2 * MIN_ARRIVALS:
                decision.update(action="keep", reason="the switch held on the live steps")
                self.trial = None
            else:
                decision.update(action="watch", reason="the switch is on trial")
            self._remember(decision)
            return self.pending_switch
        best, best_low = None, MARGIN
        for name, row in result["candidates"].items():
            if name == self.current or (name in self.rejected and self.requests - self.rejected[name] < RETRY_AFTER):
                continue
            if row["low"] > best_low and row["ttft_ratio"] <= TTFT_TOLERANCE:
                best, best_low = name, row["low"]
        if best is None:
            decision.update(action="stay", reason="no configuration beats the incumbent in the dream")
        else:
            decision.update(action="switch", to=best, reason="dreamed gain above the margin, first tokens not later")
            self.trial = {"from": self.current, "since": self.requests}
            self.pending_switch = best
        self._remember(decision)
        return self.pending_switch

    def machine_factor(self, besides: str | None = None) -> float:
        """How the live machine compares with the startup measurement: the factor of the configuration that served
        most (other than `besides`), 1.0 before any served."""
        served = [(n, k) for k, n in self.live.items() if k != besides and n > 0]
        return self.factors[max(served)[1]] if served else 1.0

    def prices(self) -> dict[str, Any]:
        """What each configuration's steps cost in a dream: its live model once it has served LIVE_MIN steps; its
        start model times its own live factor once it has served a few (FACTOR_MIN); else its start model times the
        machine factor the most-served configuration shows."""
        out: dict[str, Any] = {}
        for name in self.configs:
            if self.live[name] >= LIVE_MIN:
                out[name] = self.costs[name]
            elif self.live[name] >= FACTOR_MIN:
                out[name] = Scaled(self.base[name], self.factors[name])
            else:
                out[name] = Scaled(self.base[name], self.machine_factor(name))
        return out

    def switched(self, name: str) -> None:
        """The engine now serves `name`."""
        self.current, self.pending_switch = name, None
        self._save()

    def status(self) -> dict[str, Any]:
        return {"version": VERSION, "serving": self.current, "configs": {k: {"seqs": v["seqs"],
                                                                           "settings": v.get("settings")}
                                                                       for k, v in self.configs.items()},
                "requests": self.requests, "trial": self.trial, "rejected": sorted(self.rejected),
                "live_factor": {k: round(v, 4) for k, v in self.factors.items()}, "live_steps": dict(self.live),
                "world_model": {k: {"theta": [round(t, 6) for t in c.theta], "steps": c.n}
                                for k, c in self.costs.items()},
                "last_decisions": [{k: v for k, v in d.items() if k != "dream"} for d in self.history[-5:]]}

    # persistence ------------------------------------------------------------------------------------------------------
    def _remember(self, decision: dict[str, Any]) -> None:
        self.history.append(decision)
        self.history = self.history[-50:]
        self._save()

    def _save(self) -> None:
        if self.path is None:
            return
        state = {"version": VERSION, "binding": self.binding, "current": self.current, "rejected": self.rejected,
                 "factors": self.factors, "live": self.live,
                 "requests": self.requests, "memory": list(self.memory), "trial": self.trial,
                 "costs": {k: c.to_json() for k, c in self.costs.items()}, "history": self.history[-20:]}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state))
            tmp.replace(self.path)
        except OSError:
            pass

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            state = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return
        if state.get("version") != VERSION or state.get("binding") != self.binding:
            return  # another machine, model or engine: start fresh
        self.memory.extend(tuple(row) for row in state.get("memory", []))
        self.costs.update({k: StepCost.from_json(v) for k, v in state.get("costs", {}).items() if k in self.configs})
        self.rejected = {k: int(v) for k, v in state.get("rejected", {}).items() if k in self.configs}
        self.requests = int(state.get("requests", 0))
        self.factors.update({k: float(v) for k, v in state.get("factors", {}).items() if k in self.configs})
        self.live.update({k: int(v) for k, v in state.get("live", {}).items() if k in self.configs})
        self.history = list(state.get("history", []))
        if state.get("current") in self.configs:  # resume what served, unless it was on a trial it had not passed
            self.current = state["trial"]["from"] if state.get("trial") else state["current"]


def anchors_of(config: dict[str, Any]) -> list[tuple[int, int, int, float]]:
    """(rows, prompt tokens, attended KV tokens, seconds) the startup measurement observed for a configuration."""
    width, short, long = int(config["seqs"]), int(config.get("short_ctx", 64)), int(config.get("long_ctx", 0))
    points = [(1, 0, short, float(config["step_one"])), (width, 0, width * short, float(config["step_full"]))]
    if config.get("prefill_per_1k"):  # a prompt of 1,000 tokens alone: without it a configuration that never served
        # would dream its prefill free (seen live on 2026-11-13: it switched to the narrower width for that)
        points.append((0, 1000, 0, float(config["prefill_per_1k"])))
    if config.get("step_long") is not None and long > short:
        points.append((width, 0, width * long, float(config["step_long"])))
    elif config.get("kv_per_1k"):
        points.append((width, 0, width * 1024, float(config["step_full"]) + float(config["kv_per_1k"]) * width
                       * (1024 - short) / 1000.0))
    return points


def configs_from_measurement(measured: Iterable[dict[str, Any]], seq_ctx: int) -> dict[str, dict[str, Any]]:
    """Configurations from `ggml_engine.choose_sequences`'s measured rows: one per (settings, width)."""
    out = {}
    for row in measured:
        settings = {k: bool(v) for k, v in dict(row["settings"]).items() if v}
        for width, full in row["step_s"].items():
            one = row.get("step_one_s", {}).get(width, full)
            long = row.get("step_long_s", {}).get(width)
            extra = (int(row.get("long_ctx", 0)) - int(row.get("short_ctx", 0))) * int(width) / 1000.0
            kv = max(0.0, (float(long) - float(full)) / extra) if long is not None and extra > 0 else 0.0
            name = f"w{width}" + "".join(f"-{tag}" for key, tag in (("flash_attn", "fa"), ("kv_unified", "uni"),
                                                                     ("kv_f32", "f32"), ("kv_q8", "kv8"))
                                         if settings.get(key))
            prefill = row.get("prefill_per_1k", 0.0)
            prefill = prefill.get(width, 0.0) if isinstance(prefill, dict) else prefill
            out[name] = {"seqs": int(width), "settings": settings, "seq_ctx": seq_ctx, "step_full": float(full),
                         "step_one": float(one), "kv_per_1k": kv, "prefill_per_1k": float(prefill),
                         "short_ctx": int(row.get("short_ctx", 64)), "long_ctx": int(row.get("long_ctx", 0)),
                         "step_long": None if long is None else float(long)}
    return out


__all__ = ["GgmlAgent", "StepCost", "configs_from_measurement", "dream", "simulate", "windows"]
