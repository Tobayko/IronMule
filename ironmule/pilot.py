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
and never on a production server; its weights ship as `pilot_weights.npz` (open weights). The network takes a few tens
of microseconds in numpy; with the CPU-load read a decision took about 0.5 ms in a busy server (PI1). It never changes
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
BUDGETS = (256, 1024, 4096, 1 << 30)
DRAFTS = (0, 1)
HEADS = (len(THREAD_FRACTIONS), len(CAPS), len(BUDGETS), len(DRAFTS))
OBS = 20
WEIGHTS = Path(__file__).with_name("pilot_weights.npz")
VERSION = 1


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
    """Cores kept busy by other programs: the machine's busy time minus this process's, over the last interval."""

    def __init__(self) -> None:
        self.hz = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        self._last = self._read()
        self.foreign = 0.0

    @staticmethod
    def _read() -> tuple[float, float, float] | None:
        try:
            parts = Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]
            values = [float(v) for v in parts]
            busy = sum(values) - values[3] - (values[4] if len(values) > 4 else 0.0)  # all but idle and iowait
            own = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
            mine = float(own[11]) + float(own[12])  # utime + stime of this process (all its threads)
            return time.monotonic(), busy, mine
        except (OSError, ValueError, IndexError):
            return None

    def update(self, min_interval: float = 0.05) -> float:
        if self._last is not None and time.monotonic() - self._last[0] < min_interval:
            return self.foreign  # read /proc at most every 50 ms: a decision stays in the microseconds
        now = self._read()
        if now is None or self._last is None:
            self._last = now
            return self.foreign
        dt = now[0] - self._last[0]
        if dt < min_interval:
            return self.foreign
        busy = (now[1] - self._last[1]) / self.hz / dt
        mine = (now[2] - self._last[2]) / self.hz / dt
        self._last = now
        self.foreign = max(0.0, busy - mine)
        return self.foreign


class Pilot:
    """The decision network and the features it reads. `engine` is a `GgmlEngine`; `costs` holds the step costs this
    machine measured at startup (seconds): `step_one` (one row), `row` (each further row), `prefill_1k` (1,000 prompt
    tokens), `kv_1k` (1,000 attended tokens)."""

    def __init__(self, engine: Any, costs: dict[str, float], *, max_cap: int, gpu: bool = False,
                 weights: Path | None = None):
        self.engine, self.gpu, self.max_cap = engine, bool(gpu), int(max_cap)
        self.costs = {k: float(v) for k, v in costs.items()}
        self.cores = usable_cores()
        self.load = CpuLoad()
        self.reason: str | None = None
        self.w = None
        path = weights or Path(os.environ.get("IRONMULE_PILOT_WEIGHTS") or WEIGHTS)
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
        c, C = self.costs, float(self.cores)
        a1 = max(1e-4, c["step_one"])
        T = self.threads_for(self.cur[0])
        seen = 0.0 if self.gpu else self.load.update()
        slow = self.last_step / max(1e-6, self._uncontended(T, max(1, running)))
        return np.array([
            math.log2(C) / 6, float(self.gpu), float(getattr(self.engine, "drafter", None) is not None),
            math.log10(a1) + 1.5, math.log10(max(1e-6, c["row"]) / a1) + 1.5,
            math.log10(max(1e-6, c["prefill_1k"]) / a1) * 0.5, math.log10(max(1e-6, c["kv_1k"]) / a1) + 1,
            min(seen, C) / C, math.log(min(100.0, max(0.2, slow))) / 2, T / C,
            running / max(1, self.max_cap), math.log1p(queued) / 4, math.log1p(waiting_prompt_tokens) / 10,
            math.log1p(mean_ctx) / 8, math.log1p(self.arr_ema * a1 * 100) / 4, math.log1p(self.tok_ema * a1) / 2,
            self.cur[1] / (len(CAPS) - 1), self.cur[2] / (len(BUDGETS) - 1), self.acc_seen,
            math.log2(max(1, self.max_cap)) / 6], dtype=np.float32)

    # --- the decision ------------------------------------------------------------------------------------------------
    def decide(self, running: int, queued: int, waiting_prompt_tokens: int, mean_ctx: float) -> dict[str, Any]:
        """Threads, running cap, prompt budget and drafting for the next step."""
        started = time.perf_counter()
        x = self.features(running, queued, waiting_prompt_tokens, mean_ctx)
        if self.w is None:
            return self.apply(self.cur)
        if not np.all(np.isfinite(x)) or np.abs(x).max() > 4.0:  # outside what it was trained on: keep defaults
            self.out_of_range += 1
            return self.apply([len(THREAD_FRACTIONS) - 1, len(CAPS) - 1, len(BUDGETS) - 1, 1])
        w1, b1, w2, b2, w3, b3 = self.w
        h = np.tanh(w1 @ x + b1)
        h = np.tanh(w2 @ h + b2)
        logits = w3 @ h + b3
        choice, off = [], 0
        for n in HEADS:
            choice.append(int(np.argmax(logits[off:off + n])))
            off += n
        self.decide_us += (time.perf_counter() - started) * 1e6
        self.decisions += 1
        return self.apply(choice)

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

    def status(self) -> dict[str, Any]:
        return {"version": VERSION, "active": self.active, "reason": self.reason, "cores": self.cores,
                "foreign_cores": round(self.load.foreign, 2), "decision": self.apply(self.cur) if self.active else None,
                "decisions": self.decisions,
                "decide_us_mean": round(self.decide_us / self.decisions, 1) if self.decisions else None,
                "out_of_range": self.out_of_range}


__all__ = ["BUDGETS", "CAPS", "HEADS", "OBS", "Pilot", "THREAD_FRACTIONS", "CpuLoad", "usable_cores"]
