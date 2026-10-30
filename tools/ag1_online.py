"""AG1: the product's agent learning online, against a simulated server (BACKLOG AG1, step S).

The agent here is `ironmule_product.agent.Agent` unchanged: its world model, its planner process, its replay
qualification and its decisions. Only the server around it is simulated: one card that runs one group at a
time, members prefilled one after another, decode rounds of at most four sessions, each member ending at
its own last token; priced by RL8's fit to H9's hardware run (Qwen3 1.7B bf16 on a Tesla T4, tuned engine)
unless `--machine` names another price. Every group's time carries a slowdown drawn per window (the GPU's
state, which is what makes an A/A pair differ) and per round (lognormal), so a replayed comparison has noise.

Live traffic cycles through RL2-H2's states (quiet, bursts, spike; one trace of 16 requests each), with a
pause between traces. While the queue is empty and the next arrival is at least the agent's quiet time
away, the agent's `idle` runs, with a replay that simulates the trace on the same machine and is preempted,
as the real one is, when an arrival falls inside it. The same arrivals are also run under the reference
schedule alone (the counterfactual), so each window's cost under the agent can be compared with what the
reference would have cost on that window. `--drift-at` changes the machine mid-run (for example to a CPU,
where a further session in a round costs as much as the first), to show the agent re-learning.

    python tools/ag1_online.py --traces 300 --out ag1_online.json [--drift-at 150 --drift cpu]

This is a simulation of the server, not a hardware result: what it can show is that the loop learns, plans,
qualifies and re-learns on its own, and how long that takes in traffic; what the agent gains on a machine
is measured on that machine (AG1-V).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rl2_headroom as rl2  # noqa: E402
from ironmule_product import agent as ag  # noqa: E402
from ironmule_product.learned_dispatch import SJF_AGE_S  # noqa: E402

# RL8's fit to H9 (tuned engine, member release): prefill 0.146 s + 2.22 ms per prompt token, a round
# 18.7 ms + 15.7 ms per further session. `cpu`: one CPU core's worth, a further session costing a full step.
MACHINES = {
    "t4": {"prefill_s": 0.1456, "prefill_s_per_ktoken": 2.2214, "step_s": 0.01871, "step_s_per_session": 0.01568},
    "cpu": {"prefill_s": 0.40, "prefill_s_per_ktoken": 6.0, "step_s": 0.06, "step_s_per_session": 0.06},
}
CAPACITY_RPS = 0.5205   # RL2-H's calibration for one card (H9), which RL2-H2's states are scaled by
WINDOW_SIGMA = 0.03     # the machine's state per window
ROUND_SIGMA = 0.02
PAUSE_S = 240.0         # between live traces: the idle time the agent works in
PRIOR_WRONG = True      # the agent starts with its generic prior, not the machine's price


class Machine:
    def __init__(self, params: dict[str, float], seed: int):
        self.params, self.rng = dict(params), random.Random(seed)

    def run(self, jobs: list[dict[str, Any]], sjf: Callable, width: Callable, capacity: int,
            start: float, stop_at: float = math.inf) -> tuple[list[list[dict[str, Any]]], float, bool]:
        """Serve `jobs` (arrive, prompt_tokens, max_tokens, tokens, chars) from `start`; returns the groups it ran
        (each member with start, first, end), the time it ended, and whether an arrival at `stop_at` cut it short
        (a preempted replay). Decisions see only the output limit and the prompt class."""
        p = self.params
        slow = math.exp(self.rng.gauss(0, WINDOW_SIGMA))
        pending = sorted(jobs, key=lambda j: j["arrive"])
        waiting: list[dict[str, Any]] = []
        groups, now, i = [], start, 0
        while i < len(pending) or waiting:
            while i < len(pending) and pending[i]["arrive"] <= now:
                waiting.append(pending[i])
                i += 1
            if not waiting:
                now = pending[i]["arrive"]
                continue
            if now >= stop_at:
                return groups, now, True
            views = [_Request(j) for j in waiting]
            if sjf(views):
                waiting.sort(key=lambda j: (now - j["arrive"] <= SJF_AGE_S,
                                            j["max_tokens"] if now - j["arrive"] <= SJF_AGE_S else 0, j["arrive"]))
            size = max(1, min(capacity, width([_Request(j) for j in waiting])))
            group, waiting = waiting[:size], waiting[size:]
            t = now
            members = []
            for job in group:
                t += (p["prefill_s"] + p["prefill_s_per_ktoken"] * job["prompt_tokens"] / 1000) * slow
                members.append({**job, "start": now, "first": t, "end": t})
            remaining = {k: m["tokens"] - 1 for k, m in enumerate(members)}
            active = [k for k in remaining if remaining[k] > 0]
            while active:
                served = active[:ag.ENGINE_WIDTH]
                t += (p["step_s"] + p["step_s_per_session"] * (len(served) - 1)) * slow * math.exp(
                    self.rng.gauss(0, ROUND_SIGMA))
                for k in served:
                    remaining[k] -= 1
                    if remaining[k] == 0:
                        members[k]["end"] = t
                active = active[len(served):] + [k for k in served if remaining[k] > 0]
            if t > stop_at:  # a real arrival cancels a replay at once
                return groups, stop_at, True
            groups.append(members)
            now = t
        return groups, now, False


class _Request:
    """What the agent's decisions read of a simulated request (`learned_dispatch.View.of`)."""

    def __init__(self, job: dict[str, Any]):
        self.max_tokens = job["max_tokens"]
        self.messages = (("user", "x" * job["chars"]),)


def cost(members: list[dict[str, Any]]) -> float:
    return statistics.fmean((m["end"] - m["arrive"]) + (m["first"] - m["arrive"]) for m in members)


def live_traces(traces: int, seed: int) -> list[list[dict[str, Any]]]:
    out = []
    for k in range(traces):
        state = k % len(rl2.H2_STATES)
        items = rl2.trace_h2(state, seed + k, CAPACITY_RPS)
        out.append([{"arrive": it["offset_s"], "prompt_tokens": 1030 if it["long"] else 24,
                     "chars": 4200 if it["long"] else 100, "max_tokens": it["max_tokens"],
                     "tokens": it["max_tokens"], "state": state} for it in items])
    return out


def run(traces: int = 300, seed: int = 90_000, machine: str = "t4", drift_at: int | None = None,
        drift: str = "cpu", episodes: int = 1500, root: Path | None = None, log: Callable = print) -> dict[str, Any]:
    root = Path(root or tempfile.mkdtemp(prefix="ag1-online-"))
    params = MACHINES[machine]
    server, counterfactual = Machine(params, seed), Machine(params, seed + 1)
    replayer = Machine(params, seed + 2)
    clock = [0.0]
    agent = ag.Agent(root / "agent", {"simulated": machine, "seed": seed}, capacity=8,
                     reference={"width": 8, "sjf": False, "gather_ms": 0}, clock=lambda: clock[0],
                     planner_settings={"episodes": episodes, "seeds": 2})
    reference_sjf, reference_width = (lambda w: False), (lambda w: 8)

    def replay(trace: list[tuple[float, int, int]], choose: Callable, preempted: Callable) -> dict[str, Any] | None:
        jobs = [{"arrive": clock[0] + offset, "prompt_tokens": max(1, chars // 4), "chars": chars,
                 "max_tokens": tokens, "tokens": tokens} for offset, chars, tokens in trace]
        decide = lambda views: choose(len(views), [(0, v) for v in views[:8]], 0.0, 0)  # noqa: E731
        groups, end, cut = replayer.run(jobs, lambda v: decide(v)[2], lambda v: decide(v)[0], 8, clock[0],
                                        stop_at=next_arrival[0])
        if cut:
            clock[0] = max(clock[0], next_arrival[0])
            return None
        clock[0] = end
        for members in groups:
            agent.observe_group([{**m, "ok": True, "prompt_chars": m["chars"]} for m in members], replay=True)
        rows = [{"ttft_s": m["first"] - m["arrive"], "e2e_s": m["end"] - m["arrive"], "tokens": m["tokens"],
                 "outcome": "ok"} for members in groups for m in members]
        # grouped batch-1 keeps each answer; the real server compares the texts
        return {"rows": rows, "texts": ["same"] * len(rows), "span_s": end - jobs[0]["arrive"]}

    windows, t, end = [], 0.0, -ag.QUIET_S
    next_arrival = [0.0]
    logged: dict[str, int] = {}
    for k, trace in enumerate(live_traces(traces, seed)):
        if drift_at is not None and k == drift_at:
            server.params = counterfactual.params = replayer.params = MACHINES[drift]
            log(json.dumps({"trace": k, "drift": drift}))
        jobs = [{**j, "arrive": t + j["arrive"]} for j in trace]
        next_arrival[0] = jobs[0]["arrive"]
        # idle before this trace, once the agent's quiet time has passed: it may plan, explore and qualify
        clock[0] = max(clock[0], end + ag.QUIET_S)
        while clock[0] < next_arrival[0]:
            before = clock[0]
            agent.idle(lambda: clock[0] >= next_arrival[0], replay=replay)
            if agent._planner is not None:
                agent._planner.wait()  # the planner's CPU time is not the server's time
                agent.idle(lambda: clock[0] >= next_arrival[0], replay=None)
                continue
            if clock[0] == before:
                break  # nothing left to do before the traffic comes
        clock[0] = max(clock[0], next_arrival[0])
        serving = agent.status()["serving"]
        groups, end, _ = server.run(jobs, lambda v: bool(agent.sjf(v)) if agent.active else reference_sjf(v),
                                    lambda v: (agent.width(v) or 8) if agent.active else reference_width(v),
                                    8, clock[0])
        base, _, _ = counterfactual.run(jobs, reference_sjf, reference_width, 8, clock[0])
        for members in groups:
            agent.observe_group([{**m, "ok": True, "prompt_chars": m["chars"]} for m in members])
        clock[0] = end
        live = [m for g in groups for m in g]
        windows.append({"trace": k, "state": rl2.H2_STATES[trace[0]["state"]], "serving": serving,
                        "agent_cost": cost(live), "reference_cost": cost([m for g in base for m in g]),
                        "t_s": jobs[0]["arrive"]})
        t = end + PAUSE_S
        if k % 30 == 0 or agent.counters != logged:
            logged = dict(agent.counters)
            log(json.dumps({"trace": k, "serving": serving, "events": logged,
                            "world": {k2: round(v, 4) for k2, v in agent.world.params().items()}}))
    status = agent.status()
    agent.close()
    return {"schema": "ironmule.ag1.online.v1", "machine": machine, "drift_at": drift_at, "drift": drift,
            "traces": traces, "seed": seed, "episodes": episodes, "windows": windows,
            "history": list(agent.history), "counters": agent.counters, "final": status}


def summarise(result: dict[str, Any]) -> dict[str, Any]:
    """Per phase of the run (until the first qualification, after it, and after the drift), the agent's mean
    cost against the reference's on the same windows."""
    windows = result["windows"]
    phases: dict[str, list[dict[str, Any]]] = {}
    for w in windows:
        drifted = result["drift_at"] is not None and w["trace"] >= result["drift_at"]
        name = ("after drift, " if drifted else "") + ("reference serves" if w["serving"] == "reference"
                                                        else "learned serves")
        phases.setdefault(name, []).append(w)
    out = {}
    for name, rows in phases.items():
        a, r = sum(w["agent_cost"] for w in rows), sum(w["reference_cost"] for w in rows)
        out[name] = {"windows": len(rows), "agent": a / len(rows), "reference": r / len(rows), "gain": 1 - a / r}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--traces", type=int, default=300)
    parser.add_argument("--seed", type=int, default=90_000)
    parser.add_argument("--machine", choices=sorted(MACHINES), default="t4")
    parser.add_argument("--drift-at", type=int, default=None)
    parser.add_argument("--drift", choices=sorted(MACHINES), default="cpu")
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    started = time.time()
    result = run(args.traces, args.seed, args.machine, args.drift_at, args.drift, args.episodes)
    result["summary"] = summarise(result)
    result["wall_s"] = time.time() - started
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
