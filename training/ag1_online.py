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

    python training/ag1_online.py --traces 300 --out ag1_online.json [--drift-at 150 --drift cpu]

`--prompts FILE.jsonl` (AG2) gives the requests real prompt texts ({"prompt", "tokens"} per line, outside the
repository) with one output limit for all (`--limit`), each answer ending at its own recorded length, as chat
answers end at EOS. The agent's length model then learns online from those texts, and every window is also run
under each fixed schedule that orders by the limit, so the summary can name the best of them in hindsight.

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
    # S2: the T4's prefill, with rounds where a further session costs four base steps, so narrow groups win
    "narrow": {"prefill_s": 0.1456, "prefill_s_per_ktoken": 2.2214, "step_s": 0.010, "step_s_per_session": 0.040},
    # AG2 V-G12: Gemma 3 12B 4 bit under `native` on one T4, the reference worker at capacity 1 (the gate's fit:
    # 0.49 s prefill, 47.2 ms a step; its prompt-token slope from 5 groups is not trusted, 1 s per 1,000 is assumed)
    "t4-g12": {"prefill_s": 0.4886, "prefill_s_per_ktoken": 1.0, "step_s": 0.04715, "step_s_per_session": 0.04715},
}
# Presets that mirror a hardware protocol's sizes, for a dry run of its online phase (`--profile`)
PROFILES = {
    # AG2 V-G12 per card: 32 requests a trace over two cards is 16 a card at half the two cards' calibrated rate
    # (0.135 requests a second), limit 256, 360 s idle, at most 40 traces; the other card's answers teach the
    # shared length model (`twin`)
    "v-g12": {"machine": "t4-g12", "capacity": 1, "requests": 16, "rate": 0.135 / 2, "pause_s": 360.0,
              "traces": 40, "limit": 256, "twin": True, "episodes": 3000},
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
            start: float, stop_at: float = math.inf, skip_idle: bool = False,
            order_value: Callable | None = None) -> tuple[list[list[dict[str, Any]]], float, bool]:
        """Serve `jobs` (arrive, prompt_tokens, max_tokens, tokens, chars) from `start`; returns the groups it ran
        (each member with start, first, end), the time it ended, and whether an arrival at `stop_at` cut it short
        (a preempted replay). Decisions see only the output limit and the prompt class. `skip_idle` releases the
        next arrival at once while the server is idle, moving the rest up, as the service's replay does.
        `order_value(job)` is what shortest-first sorts by (AG2); by default the output limit."""
        value = order_value or (lambda j: j["max_tokens"])
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
                if skip_idle and pending[i]["arrive"] > now:
                    ahead = pending[i]["arrive"] - now
                    for job in pending[i:]:
                        job["arrive"] -= ahead
                else:
                    now = pending[i]["arrive"]
                continue
            if now >= stop_at:
                return groups, now, True
            views = [_Request(j) for j in waiting]
            if sjf(views):
                waiting.sort(key=lambda j: (now - j["arrive"] <= SJF_AGE_S,
                                            value(j) if now - j["arrive"] <= SJF_AGE_S else 0, j["arrive"]))
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
        self.messages = (("user", job.get("prompt") or "x" * job["chars"]),)
        self.request_id = job.get("id")


def cost(members: list[dict[str, Any]]) -> float:
    return statistics.fmean((m["end"] - m["arrive"]) + (m["first"] - m["arrive"]) for m in members)


def live_traces(traces: int, seed: int, prompts: list[dict[str, Any]] | None = None,
                limit: int = 512, requests: int = rl2.REQUESTS,
                rate: float = CAPACITY_RPS) -> list[list[dict[str, Any]]]:
    out = []
    draw = random.Random(seed)
    for k in range(traces):
        state = k % len(rl2.H2_STATES)
        items = rl2.trace_h2(state, seed + k, rate, requests)
        jobs = []
        for i, it in enumerate(items):
            job = {"arrive": it["offset_s"], "prompt_tokens": 1030 if it["long"] else 24,
                   "chars": 4200 if it["long"] else 100, "max_tokens": it["max_tokens"],
                   "tokens": it["max_tokens"], "state": state, "id": f"live-{k}-{i}"}
            if prompts:  # AG2: a real prompt, one limit for all, the answer ending at its own length
                p = prompts[draw.randrange(len(prompts))]
                job.update(prompt=p["prompt"], chars=len(p["prompt"]), prompt_tokens=max(1, len(p["prompt"]) // 4),
                           max_tokens=limit, tokens=max(1, min(limit, int(p["tokens"]))))
            jobs.append(job)
        out.append(jobs)
    return out


FIXED = [(w, s) for w in (1, 2, 4, 8) for s in (False, True)]  # hindsight comparators, ordered by the limit


def run(traces: int = 300, seed: int = 90_000, machine: str = "t4", drift_at: int | None = None,
        drift: str = "cpu", episodes: int = 1500, root: Path | None = None, log: Callable = print,
        prompts: list[dict[str, Any]] | None = None, limit: int = 512, capacity: int = 8,
        pause_s: float = PAUSE_S, requests: int = rl2.REQUESTS, rate: float = CAPACITY_RPS,
        twin: bool = False) -> dict[str, Any]:
    """`capacity`, `pause_s`, `requests` and `rate` size the server and its traffic. `twin` (AG2, two cards) has a
    second card's answers to the same traffic teach the shared length model after every trace, as
    `agent.share_lengths` does; that card's own agent, replays and qualification are not simulated."""
    root = Path(root or tempfile.mkdtemp(prefix="ag1-online-"))
    params = MACHINES[machine]
    server, counterfactual = Machine(params, seed), Machine(params, seed + 1)
    replayer = Machine(params, seed + 2)
    fixed = {f"w{w}/{'sjf' if s else 'fifo'}": Machine(params, seed + 3 + k) for k, (w, s) in enumerate(FIXED)} \
        if prompts else {}
    clock = [0.0]
    agent = ag.Agent(root / "agent", {"simulated": machine, "seed": seed}, capacity=capacity,
                     reference={"width": 8, "sjf": False, "gather_ms": 0}, clock=lambda: clock[0],
                     planner_settings={"episodes": episodes, "seeds": 2})
    reference_sjf, reference_width = (lambda w: False), (lambda w: 8)

    def replay(trace: list[tuple], choose: Callable, preempted: Callable) -> dict[str, Any] | None:
        jobs = []
        for row in trace:  # (offset, chars, tokens to make[, limit, prediction]) as the service's replay takes them
            offset, chars, tokens = row[:3]
            limit, predicted = (row[3], row[4]) if len(row) > 4 else (tokens, tokens)
            jobs.append({"arrive": clock[0] + offset, "prompt_tokens": max(1, chars // 4), "chars": chars,
                         "max_tokens": limit, "tokens": tokens, "predicted": predicted})
        decide = lambda views: choose(len(views), [(0, v) for v in views[:8]], 0.0, 0)  # noqa: E731
        key = [None]

        def sjf(views: list) -> bool:
            decision = decide(views)
            key[0] = decision[3] if len(decision) > 3 else "limit"
            return decision[2]
        groups, end, cut = replayer.run(jobs, sjf, lambda v: decide(v)[0], 8, clock[0],
                                        stop_at=next_arrival[0], skip_idle=True,
                                        order_value=lambda j: j["predicted"] if key[0] == "predicted"
                                        else j["max_tokens"])
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
    events: list[dict[str, Any]] = []  # every agent event with the trace it happened before (its status keeps 50)

    def collect(trace: int) -> None:
        for event in list(agent.history):
            if not any(event is seen["event_ref"] for seen in events[-60:]):
                events.append({"event_ref": event, "trace": trace, **event})
    twins = live_traces(traces, seed + 50_000, prompts, limit, requests, rate) if twin and prompts else None
    for k, trace in enumerate(live_traces(traces, seed, prompts, limit, requests, rate)):
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
        collect(k)
        serving = agent.status()["serving"]
        active = agent.active

        def order_value(job: dict[str, Any]) -> float:  # AG2: the agent's key when its policy orders by prediction
            value = agent.order_key(_Request(job)) if active else None
            return job["max_tokens"] if value is None else value
        groups, end, _ = server.run(jobs, lambda v: bool(agent.sjf(v)) if active else reference_sjf(v),
                                    lambda v: (agent.width(v) or 8) if active else reference_width(v),
                                    capacity, clock[0], order_value=order_value)
        base, _, _ = counterfactual.run(jobs, reference_sjf, reference_width, capacity, clock[0])
        fixed_costs = {}
        for (name, machine_), (w, s) in zip(fixed.items(), FIXED):
            if w > capacity:
                continue
            ran, _, _ = machine_.run(jobs, lambda v, s=s: s, lambda v, w=w: w, capacity, clock[0])
            fixed_costs[name] = cost([m for g in ran for m in g])
        for members in groups:
            observed = []
            for m in members:
                row = {**m, "ok": True, "prompt_chars": m["chars"]}
                if prompts:  # the service hands the agent each live request's length features
                    row["length"], row["request_id"] = agent.length_of(_Request(m)), m["id"]
                observed.append(row)
            agent.observe_group(observed)
        if twins is not None:  # the other card's answers, learned by the shared length model
            for job in twins[k]:
                feats, predicted = agent.length_of(_Request({**job, "id": f"twin-{job['id']}"}))
                with agent._lock:
                    agent._predicted.pop(f"twin-{job['id']}", None)
                agent.lengths.learn(feats, int(job["tokens"]), predicted)
        clock[0] = end
        live = [m for g in groups for m in g]
        windows.append({"trace": k, "state": rl2.H2_STATES[trace[0]["state"]], "serving": serving,
                        "agent_cost": cost(live), "reference_cost": cost([m for g in base for m in g]),
                        "t_s": jobs[0]["arrive"], **({"fixed_costs": fixed_costs} if fixed_costs else {})})
        t = end + pause_s
        if k % 30 == 0 or agent.counters != logged:
            logged = dict(agent.counters)
            log(json.dumps({"trace": k, "serving": serving, "events": logged,
                            "world": {k2: round(v, 4) for k2, v in agent.world.params().items()}}))
    collect(traces)
    status = agent.status()
    agent.close()
    return {"schema": "ironmule.ag1.online.v1", "machine": machine, "drift_at": drift_at, "drift": drift,
            "capacity": capacity, "pause_s": pause_s, "requests": requests, "rate": rate, "twin": twin,
            "prompts": len(prompts) if prompts else None, "limit": limit if prompts else None,
            "traces": traces, "seed": seed, "episodes": episodes, "windows": windows,
            "history": [{k: v for k, v in e.items() if k != "event_ref"} for e in events],
            "counters": agent.counters, "final": status}


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
        if all("fixed_costs" in w for w in rows):  # AG2: the best schedule ordered by the limit, in hindsight
            totals = {arm: sum(w["fixed_costs"][arm] for w in rows) for arm in rows[0]["fixed_costs"]}
            best = min(totals, key=totals.get)
            out[name].update(best_fixed=best, best_fixed_cost=totals[best] / len(rows),
                             gain_vs_best_fixed=1 - a / totals[best])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--traces", type=int, default=300)
    parser.add_argument("--seed", type=int, default=90_000)
    parser.add_argument("--machine", choices=sorted(MACHINES), default="t4")
    parser.add_argument("--drift-at", type=int, default=None)
    parser.add_argument("--drift", choices=sorted(MACHINES), default="cpu")
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--prompts", default=None, help="AG2: JSONL of {prompt, tokens}, kept outside the repository")
    parser.add_argument("--limit", type=int, default=512, help="AG2: the one output limit every request sends")
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None,
                        help="a hardware protocol's sizes (overrides --traces, --machine, --limit, --episodes)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    started = time.time()
    prompts = ([json.loads(line) for line in Path(args.prompts).read_text().splitlines() if line.strip()]
               if args.prompts else None)
    sizes = dict(PROFILES[args.profile]) if args.profile else {
        "machine": args.machine, "traces": args.traces, "limit": args.limit, "episodes": args.episodes}
    result = run(sizes.pop("traces"), args.seed, sizes.pop("machine"), args.drift_at, args.drift,
                 sizes.pop("episodes"), prompts=prompts, **sizes)
    result["summary"] = summarise(result)
    result["wall_s"] = time.time() - started
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
