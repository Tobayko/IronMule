"""IronMule Pilot: a small decision network that runs a server's LLM engine on the hardware it finds.

Every engine step the Pilot looks at the machine and the traffic -- how many cores this process may use, how many of
them other programs keep busy, what a step costs on this machine (measured), how many requests run and wait, how long
their prompts and contexts are -- and decides four things for the next step:

* how many CPU threads the step uses (on a shared server the default of "all cores" can make every step 20-50x slower:
  llama.cpp's threads wait at barriers for the one a foreign process pushed off its core; docs/LAB.md, PI1);
* how many requests run at once;
* how many prompt tokens may join a step (first tokens of new requests against the pace of running ones);
* whether the draft model may draft (DM1).

The network is a two-layer MLP (20 inputs, 2 x 128 tanh units, four softmax heads), about 22,000 weights. It was
trained from scratch by reinforcement learning in a dreamed world -- a world model of serving built from measurements --
and never on a production server; its weights ship as `pilot_weights.npz` (open weights). An engine whose layers run on a
GPU uses `pilot_weights_gpu.npz` instead (PI3): weights trained in a dream world with a GPU's measured costs, whose
decisions there were checked on a T4, while their thread decisions on CPUs were not as steady as the CPU weights'. The network takes a few tens
of microseconds in numpy; with the CPU load from `/proc` a decision takes about 0.17-0.19 ms in the running server (PI1, PI2), about 0.11 ms
beside a GPU, where no CPU load is read (PI3). It never changes
what the model answers: every decision is a scheduling or threading choice.

Safety: without the weights file, with features outside the ranges it was trained on, or with `IRONMULE_PILOT=0`, the
engine keeps its own defaults. `status()` reports what the Pilot decided and why it is or is not active.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

THREAD_FRACTIONS = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
CAPS = (1, 2, 4, 8, 16, 32, 64)
BUDGETS = (32, 128, 512, 2048, 1 << 30)  # PI2: 32 and 128 let short prompts start one or a few per step
DRAFTS = (0, 1)
HEADS = (len(THREAD_FRACTIONS), len(CAPS), len(BUDGETS), len(DRAFTS))
OBS = 20
WEIGHTS = Path(__file__).with_name("pilot_weights.npz")
WEIGHTS_GPU = Path(__file__).with_name("pilot_weights_gpu.npz")  # PI3: the weights for an engine on a GPU
VERSION = 2  # 2 (PI2): five prompt budgets, the shared pool honours them
_HEAD_SLICES = tuple((sum(HEADS[:i]), sum(HEADS[:i + 1])) for i in range(len(HEADS)))


def usable_cores() -> int:
    """CPUs this process may run on: its affinity, narrowed by a cgroup CPU quota (containers)."""
    try:
        cores = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cores = os.cpu_count() or 1
    for path in ("/sys/fs/cgroup/cpu.max",):
        try:
            quota, period = Path(path).read_text().split()[:2]
            if quota != "max":
                cores = min(cores, max(1, math.ceil(int(quota) / int(period))))
        except (OSError, ValueError):
            pass
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0:
            cores = min(cores, max(1, math.ceil(quota / period)))
    except (OSError, ValueError):
        pass
    return max(1, cores)


class CpuLoad:
    """Cores kept busy by other programs: the machine's busy time minus this process's, over the last interval.

    It keeps `/proc/stat` and `/proc/self/stat` open and rereads them with `pread`: about 50 microseconds right after
    an engine step, against about 190 for opening and reading them afresh. A background reader would take nothing from
    the step's path, but it cost 4.5 % idle with 8 requests on 4 cores: its wake-ups push a llama.cpp thread off its
    core (docs/LAB.md, PI1 decision time)."""

    def __init__(self, scope: str | None = None) -> None:
        scope = scope or os.environ.get("IRONMULE_PILOT_CPU_LOAD", "host")
        if scope not in ("host", "affinity"):
            raise ValueError("IRONMULE_PILOT_CPU_LOAD must be host or affinity")
        self.cpus = frozenset(os.sched_getaffinity(0)) if scope == "affinity" else None
        self.scope_binding = {"reader": "linux-affinity-v1", "cpus": sorted(self.cpus)} if self.cpus else None
        if self.cpus:
            self._labels = {f"cpu{cpu}".encode() for cpu in self.cpus}
            self._size = min(1 << 20, max(4096, (max(self.cpus) + 2) * 192))
        self.hz = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        self._fds: tuple[int, int] | None = None
        try:
            self._fds = (os.open("/proc/stat", os.O_RDONLY), os.open("/proc/self/stat", os.O_RDONLY))
        except OSError:
            pass
        self._last = self._read()
        self.foreign = math.nan if self.cpus else 0.0

    def close(self) -> None:
        if self._fds is not None:
            for fd in self._fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._fds = None

    def _read(self) -> tuple[float, float, float] | None:
        if self._fds is None:
            return None
        try:
            if self.cpus:
                busy = 0
                found = set()
                snapshot = os.pread(self._fds[0], self._size, 0)
                for line in snapshot.rpartition(b"\n")[0].splitlines():
                    name, _, counters = line.partition(b" ")
                    if name in self._labels:
                        ticks = [int(x) for x in counters.split()]
                        if len(ticks) < 8:
                            return None
                        # User and nice already include guest ticks. Idle and iowait are idle.
                        busy += sum(ticks[:3]) + sum(ticks[5:8])
                        found.add(name)
                        if len(found) == len(self._labels):
                            break
                if found != self._labels:
                    return None
            else:
                parts = os.pread(self._fds[0], 512, 0).split(b"\n", 1)[0].split()[1:]
                values = [float(v) for v in parts]
                busy = sum(values) - values[3] - (values[4] if len(values) > 4 else 0.0)  # historical host reader
            own = os.pread(self._fds[1], 4096, 0).rsplit(b")", 1)[1].split()
            mine = float(own[11]) + float(own[12])  # utime + stime of this process (all its threads)
            return time.monotonic(), busy, mine
        except (OSError, ValueError, IndexError):
            return None

    def update(self, min_interval: float = 0.05) -> float:
        if self.cpus:
            try:
                current = os.sched_getaffinity(0)
            except OSError:
                current = set()
            if current != self.cpus:
                self.foreign, self._last = math.nan, None
                return self.foreign
        if self._last is not None and time.monotonic() - self._last[0] < min_interval:
            return self.foreign  # read /proc at most every 50 ms
        now = self._read()
        if now is None or self._last is None:
            self._last = now
            if self.cpus:
                self.foreign = math.nan
            return self.foreign
        dt = now[0] - self._last[0]
        if dt < min_interval:
            return self.foreign
        busy = (now[1] - self._last[1]) / self.hz / dt
        mine = (now[2] - self._last[2]) / self.hz / dt
        self._last = now
        self.foreign = max(0.0, busy - mine)
        return self.foreign


def predict(weights, features):
    """Score Pilot actions on the CPU, also used for offline fitting and thread proposals."""
    w1, b1, w2, b2, w3, b3 = weights
    h = np.tanh(w1 @ np.asarray(features, dtype=np.float32) + b1)
    return w3 @ np.tanh(w2 @ h + b2) + b3


class Pilot:
    """The decision network and the features it reads. `engine` is a `GgmlEngine`; `costs` holds the step costs this
    machine measured at startup (seconds): `step_one` (one row), `row` (each further row), `prefill_1k` (1,000 prompt
    tokens), `kv_1k` (1,000 attended tokens)."""

    @staticmethod
    def startup_profile(heads, features, names, qualified, reference, *, in_support, margin_error):
        """Select one already qualified complete profile before native graphs are captured."""
        if reference not in qualified or len(names) < 2 or not heads:
            raise ValueError("an eligible reference and competing profile scores are required")
        started = time.perf_counter_ns()
        selected, proposal, scores, margin = reference, None, [], None
        reason = "unknown hardware or workload"
        if in_support:
            scores = [predict(weights, features).tolist() for weights in heads]
            if any(len(s) != len(names) or not all(math.isfinite(x) for x in s) for s in scores):
                reason = "invalid prediction"
            else:
                ranks = [sorted(range(len(names)), key=score.__getitem__, reverse=True) for score in scores]
                proposal = names[ranks[0][0]]
                margin = min(score[r[0]] - score[r[1]] for score, r in zip(scores, ranks, strict=True))
                if any(names[r[0]] != proposal for r in ranks) or not margin > margin_error >= 0:
                    reason = "unresolved model margin"
                elif proposal not in qualified:
                    reason = "whole profile lacks local qualification"
                else:
                    selected, reason = proposal, "qualified model choice"
        return {"profile": selected, "proposal": proposal, "scores": scores, "margin": margin,
                "margin_error": margin_error, "reason": reason, "cpu_us": (time.perf_counter_ns() - started) / 1000}

    def __init__(self, engine: Any, costs: dict[str, float], *, max_cap: int, gpu: bool = False,
                 weights: Path | None = None):
        self.engine, self.gpu, self.max_cap = engine, bool(gpu), int(max_cap)
        self.costs = {k: float(v) for k, v in costs.items()}
        self.cores = usable_cores()
        self.load = CpuLoad("host" if self.gpu else None)
        # Shipped weights always see the original host feature, even during a scoped trial.
        self.base_load = CpuLoad("host") if self.load.scope_binding else self.load
        self.reason: str | None = None
        self.w = None
        default = WEIGHTS_GPU if self.gpu and WEIGHTS_GPU.exists() else WEIGHTS
        path = weights or Path(os.environ.get("IRONMULE_PILOT_WEIGHTS") or default)
        self.weights_path = path
        self.weights_name = path.name
        self.learning = None
        self.last_fallback = False
        if os.environ.get("IRONMULE_PILOT", "1") == "0":
            self.reason = "turned off (IRONMULE_PILOT=0)"
        elif not path.exists():
            self.reason = f"no weights at {path.name}"
        else:
            z = np.load(path)
            if tuple(int(h) for h in z["heads"]) != HEADS or int(z["obs"]) != OBS:
                self.reason = "the weights are for another Pilot version"
            else:
                self.w = [z[k].astype(np.float32) for k in ("w1", "b1", "w2", "b2", "w3", "b3")]
        self.cur = [len(THREAD_FRACTIONS) - 1, len(CAPS) - 1, len(BUDGETS) - 1, 1]
        self.last_step = self.costs.get("step_one", 0.1)
        self.arr_ema = self.tok_ema = 0.0
        self.acc_seen = 0.5
        self.decisions = 0
        self.decide_us = 0.0
        self.out_of_range = 0
        self.history: list[dict[str, Any]] = []

    @property
    def active(self) -> bool:
        return self.w is not None

    # --- the features (the dream world computes the same) -------------------------------------------------------------
    def threads_for(self, index: int) -> int:
        return max(1, round(self.cores * THREAD_FRACTIONS[index]))

    def _uncontended(self, threads: int, rows: int) -> float:
        c = self.costs
        return (c["step_one"] + c["row"] * max(0, rows - 1)) * (self.cores / max(1, threads)) ** 0.85 \
            if not self.gpu else c["step_one"] + c["row"] * max(0, rows - 1)

    def features(self, running: int, queued: int, waiting_prompt_tokens: int, mean_ctx: float) -> np.ndarray:
        return np.array(self._features(running, queued, waiting_prompt_tokens, mean_ctx, self.load), dtype=np.float32)

    def _features(self, running: int, queued: int, waiting_prompt_tokens: int, mean_ctx: float,
                  load: CpuLoad) -> list[float]:
        c, C = self.costs, float(self.cores)
        a1 = max(1e-4, c["step_one"])
        T = self.threads_for(self.cur[0])
        seen = 0.0 if self.gpu else load.update()
        slow = self.last_step / max(1e-6, self._uncontended(T, max(1, running)))
        return [
            math.log2(C) / 6, float(self.gpu), float(getattr(self.engine, "drafter", None) is not None),
            math.log10(a1) + 1.5, math.log10(max(1e-6, c["row"]) / a1) + 1.5,
            math.log10(max(1e-6, c["prefill_1k"]) / a1) * 0.5, math.log10(max(1e-6, c["kv_1k"]) / a1) + 1,
            min(seen, C) / C, math.log(min(100.0, max(0.2, slow))) / 2, T / C,
            running / max(1, self.max_cap), math.log1p(queued) / 4, math.log1p(waiting_prompt_tokens) / 10,
            math.log1p(mean_ctx) / 8, math.log1p(self.arr_ema * a1 * 100) / 4, math.log1p(self.tok_ema * a1) / 2,
            self.cur[1] / (len(CAPS) - 1), self.cur[2] / (len(BUDGETS) - 1), self.acc_seen,
            math.log2(max(1, self.max_cap)) / 6]

    # --- the decision ------------------------------------------------------------------------------------------------
    def decide(self, running: int, queued: int, waiting_prompt_tokens: int, mean_ctx: float) -> dict[str, Any]:
        """Threads, running cap, prompt budget and drafting for the next step."""
        started = time.perf_counter()
        if self.base_load is not self.load:
            self.load.update()  # bind the plain-decode chooser's observed regime to this affinity
        f = self._features(running, queued, waiting_prompt_tokens, mean_ctx, self.base_load)
        self.last_fallback = False
        if self.w is None:
            choice = self.cur
        elif not all(math.isfinite(v) and abs(v) <= 4.0 for v in f):
            self.last_fallback = True
            self.out_of_range += 1
            choice = [len(THREAD_FRACTIONS) - 1, len(CAPS) - 1, len(BUDGETS) - 1, 1]
        else:
            logits = predict(self.w, f).tolist()
            choice = [max(range(b - a), key=lambda i, a=a: logits[a + i]) for a, b in _HEAD_SLICES]
        decision = self.apply(choice)
        self.decisions += 1
        self.decide_us += (time.perf_counter() - started) * 1e6
        return decision

    def apply(self, choice: list[int]) -> dict[str, Any]:
        self.cur = list(choice)
        return {"threads": self.threads_for(choice[0]), "cap": min(CAPS[choice[1]], self.max_cap),
                "budget": BUDGETS[choice[2]], "draft": bool(DRAFTS[choice[3]])}

    def observe(self, seconds: float, produced: int, arrived: int, accepted_drafts: float | None = None) -> None:
        """After a step: its seconds, the tokens it made, the requests that arrived meanwhile."""
        self.last_step = max(1e-5, seconds)
        a = 1 - math.exp(-seconds / 5.0)
        self.arr_ema += a * (arrived / self.last_step - self.arr_ema)
        self.tok_ema += a * (produced / self.last_step - self.tok_ema)
        if accepted_drafts is not None:
            self.acc_seen += 0.05 * (accepted_drafts - self.acc_seen)

    def close(self) -> None:
        self.load.close()
        if self.base_load is not self.load:
            self.base_load.close()

    def status(self) -> dict[str, Any]:
        return {"version": VERSION, "active": self.active, "reason": self.reason, "weights": self.weights_name,
                "cores": self.cores,
                "foreign_cores": round(self.base_load.foreign, 2),
                **({"cpu_load_scope": self.load.scope_binding,
                    "scoped_foreign_cores": round(self.load.foreign, 2) if math.isfinite(self.load.foreign) else None}
                   if self.load.scope_binding else {}),
                "decision": self.apply(self.cur) if self.active else None,
                "decisions": self.decisions,
                "decide_us_mean": round(self.decide_us / self.decisions, 1) if self.decisions else None,
                "out_of_range": self.out_of_range,
                **({"thread_learning": self.learning.status()} if self.learning is not None else {})}


__all__ = [
    "BUDGETS",
    "CAPS",
    "HEADS",
    "OBS",
    "THREAD_FRACTIONS",
    "WEIGHTS",
    "WEIGHTS_GPU",
    "CpuLoad",
    "Pilot",
    "usable_cores",
]
