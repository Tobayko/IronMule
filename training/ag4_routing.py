"""AG4 step 2 in simulation: the product's router between a single path and a stacked engine on one card.

Both paths are priced by a hardware measurement (AG4-Q run 2 on a T4 with Qwen 3 1.7B by default): the single path
serves one request at a time; the stacked engine serves up to eight, a step costing `step(n)`, and a newcomer's
prefill stalls the rows already running (slot reuse, H10). RL2-H2's three traffic states arrive at a load relative
to the single path's capacity, with answer lengths from a prompts file (outside the repository) or a fixed spread.
Three policies are compared on the same arrivals: the single path alone, the stacked engine alone, and the router
(`ironmule_product.path_router.PathRouter`, unchanged, learning from every finished request). A simulation, so no
hardware claim follows from it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rl2_headroom as rl2  # noqa: E402
from ironmule_product.path_router import PathRouter  # noqa: E402

# AG4-Q run 2 (BACKLOG AG4): float16 alone, and the dense engine at one and eight rows, averaged over its prompts
MEASURED = {"single_step_s": 0.0173, "single_first_s": 0.06, "stacked_first_s": 0.23,
            "stacked_step_s": {1: 0.0212, 8: 0.0282}, "width": 8}


def setup_of(measured: dict[str, Any]) -> dict[str, Any]:
    row = {"base_step_s": measured["single_step_s"], "base_first_s": measured["single_first_s"],
           "plan_first_s": measured["stacked_first_s"],
           "plan_step_s": {str(k): v for k, v in measured["stacked_step_s"].items()}}
    return {"stacked": {"speed": {"short": row, "long": dict(row)}}}


def simulate(jobs: list[dict[str, Any]], policy: str, measured: dict[str, Any], router: PathRouter | None = None
             ) -> list[dict[str, Any]]:
    """Serve `jobs` (arrive, tokens) by `policy` ("single", "stacked" or "router"); each job gets first and end."""
    width = measured["width"]
    one, wide = measured["stacked_step_s"][1], measured["stacked_step_s"][width]
    step = lambda n: one + (wide - one) * (n - 1) / (width - 1)  # noqa: E731
    pending = sorted(({**j} for j in jobs), key=lambda j: j["arrive"])
    single_free = 0.0                    # the single path: FIFO, one at a time
    single_ends: list[float] = []
    active: list[dict[str, Any]] = []    # the stacked engine's rows: remaining tokens
    waiting: list[dict[str, Any]] = []
    now, done, i = 0.0, [], 0
    while i < len(pending) or active or waiting:
        next_arrival = pending[i]["arrive"] if i < len(pending) else float("inf")
        if not active and not waiting:
            now = max(now, next_arrival)
        while i < len(pending) and pending[i]["arrive"] <= now:
            job = pending[i]
            i += 1
            single_ends = [e for e in single_ends if e > job["arrive"]]
            outstanding = [len(single_ends), len(active) + len(waiting)]
            path = {"single": 0, "stacked": 1}.get(policy)
            if path is None:
                path = router.choose(outstanding, 512)
            job["path"], job["sharing"] = path, outstanding[path] + 1
            if path == 0:
                start = max(single_free, job["arrive"])
                job["first"] = start + measured["single_first_s"]
                job["end"] = job["first"] + (job["tokens"] - 1) * measured["single_step_s"]
                single_free = job["end"]
                single_ends.append(job["end"])
                done.append(job)
                if router is not None:
                    router.observe(0, job["sharing"], job["tokens"], job["end"] - job["first"],
                                   job["first"] - job["arrive"] if job["sharing"] == 1 else None)
            else:
                waiting.append(job)
        if not active and not waiting:
            continue
        stall = 0.0
        while waiting and len(active) < width:  # a newcomer's prefill stalls the running rows
            job = waiting.pop(0)
            stall += measured["stacked_first_s"]
            job["first"], job["left"] = now + stall, job["tokens"] - 1
            active.append(job)
        now += stall + step(len(active))
        for job in list(active):
            job["left"] -= 1
            if job["left"] <= 0:
                job["end"] = now
                active.remove(job)
                done.append(job)
                if router is not None:
                    router.observe(1, job["sharing"], job["tokens"], job["end"] - job["first"],
                                   job["first"] - job["arrive"] if job["sharing"] == 1 else None)
    return done


def metrics(traces: list[list[dict[str, Any]]]) -> dict[str, float]:
    """Tokens per second over the traces' busy spans (each from its first arrival to its last answer), and the
    first-token and end times of every request."""
    span = sum(max(j["end"] for j in done) - min(j["arrive"] for j in done) for done in traces)
    done = [j for trace in traces for j in trace]
    ttft = sorted(j["first"] - j["arrive"] for j in done)
    return {"tokens_per_s": sum(j["tokens"] for j in done) / span, "ttft_mean_s": statistics.fmean(ttft),
            "ttft_p95_s": ttft[max(0, int(0.95 * len(ttft)) - 1)],
            "e2e_mean_s": statistics.fmean(j["end"] - j["arrive"] for j in done),
            "stacked_share": sum(j.get("path") == 1 for j in done) / len(done)}


def traffic(loads: list[float], traces: int, seed: int, lengths: list[int], measured: dict[str, Any],
            requests: int = 32) -> dict[float, list[list[dict[str, Any]]]]:
    """For each load (arrival rate over the single path's capacity), RL2-H2's states on fresh seeds."""
    mean_service = measured["single_first_s"] + statistics.fmean(lengths) * measured["single_step_s"]
    draw = random.Random(seed)
    out = {}
    for load in loads:
        out[load] = []
        for k in range(traces):
            items = rl2.trace_h2(k % 3, seed + k, load / mean_service, requests)
            out[load].append([{"arrive": it["offset_s"], "tokens": max(2, draw.choice(lengths))} for it in items])
    return out


def run(loads: list[float], traces: int = 30, seed: int = 70_000, lengths: list[int] | None = None,
        measured: dict[str, Any] | None = None) -> dict[str, Any]:
    measured = measured or MEASURED
    lengths = lengths or [40, 80, 120, 160, 200, 256]
    result = {}
    for load, traces_ in traffic(loads, traces, seed, lengths, measured).items():
        router = PathRouter.from_setup(setup_of(measured), measured["width"])  # learns across this load's traces
        rows = {name: [] for name in ("single", "stacked", "router")}
        for jobs in traces_:
            for name in rows:
                rows[name].append(simulate(jobs, name, measured, router if name == "router" else None))
        result[str(load)] = {name: metrics(done) for name, done in rows.items()}
        result[str(load)]["router_status"] = router.status()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--loads", default="0.3,0.8,1.5,3.0", help="arrival rate over the single path's capacity")
    parser.add_argument("--traces", type=int, default=30)
    parser.add_argument("--prompts", default=None, help='JSONL of {"tokens": ...}; answers capped at --limit')
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    lengths = None
    if args.prompts:
        lengths = [min(args.limit, int(json.loads(line)["tokens"])) for line in Path(args.prompts).read_text().splitlines()
                   if line.strip()]
    result = run([float(x) for x in args.loads.split(",")], args.traces, lengths=lengths)
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
