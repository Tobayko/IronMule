"""AG10 S: the ggml engine's agent (`ironmule/ggml_agent.py`) unchanged, against a simulated engine whose true step
costs come from KV1's measurement (Qwen3-4B Q8_0, 16 sequences, a 4-vCPU Xeon; docs/LAB.md) and differ from the
agent's start model, under traffic that changes regime: short chats, then long-context requests (RAG-like prompts of
1,200-1,700 tokens), then short chats again, each at a load that keeps the engine busy.

Every live window of 32 arrivals is served by the configuration the agent serves at that moment (the true cost model
prices it, the agent sees each step with 3 % lognormal noise), and also, for comparison only, by every fixed
configuration on the same arrivals. Between windows the agent gets one idle call. Reported per phase: tokens per
second of the agent's choice, of the fixed incumbent (SEQ1's choice, flash attention) and of the best fixed
configuration in hindsight, and the mean first-token time of each; plus every switch.

    python training/ag10_sim.py [--seeds 3] [--windows 60] [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ironmule.ggml_agent import GgmlAgent, StepCost, simulate  # noqa: E402

# KV1 (docs/LAB.md): one row and 16 rows at 64 tokens, and the extra seconds per thousand attended KV tokens at 16 rows
TRUE = {"w16-fa": StepCost.from_measurement(16, 0.093, 0.157, 2.4, 0.021),
        "w16": StepCost.from_measurement(16, 0.116, 0.213, 2.0, 0.013),
        "w8-fa": StepCost.from_measurement(8, 0.093, 0.127, 2.4, 0.021),
        "w8": StepCost.from_measurement(8, 0.116, 0.165, 2.0, 0.013)}
# what the startup measurement told the agent: each number off by up to 20 %
MEASURED = {"w16-fa": (16, {"flash_attn": True}, 0.093, 0.157, 0.026), "w16": (16, {}, 0.116, 0.213, 0.010),
            "w8-fa": (8, {"flash_attn": True}, 0.093, 0.127, 0.026), "w8": (8, {}, 0.116, 0.165, 0.010)}
PHASES = (("short", 20, 120), ("long", 1200, 1700), ("short again", 20, 120))


def configs(rng: random.Random) -> dict[str, dict]:
    out = {}
    for name, (seqs, settings, one, full, kv) in MEASURED.items():
        bias = lambda v: v * rng.uniform(0.85, 1.15)  # noqa: E731
        out[name] = {"seqs": seqs, "settings": settings, "seq_ctx": 2048, "step_one": bias(one),
                     "step_full": bias(full), "kv_per_1k": bias(kv), "prefill_per_1k": 2.2}
    return out


def traffic(rng: random.Random, low: int, high: int, n: int, gap: float) -> list[tuple[float, int, int]]:
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(1 / gap)
        out.append((t, rng.randint(low, high), rng.randint(40, 200)))
    return out


def run(seed: int, windows: int) -> dict:
    rng = random.Random(seed)
    agent = GgmlAgent(None, {"sim": seed}, configs(rng), "w16-fa", seed=seed)
    result = {"seed": seed, "phases": [], "switches": []}
    clock = 0.0
    for phase, low, high in PHASES:
        rows = {"agent": [], **{name: [] for name in TRUE}}
        for _ in range(windows):
            arrivals = [(t + clock, p, a) for t, p, a in traffic(rng, low, high, 32, 0.25)]
            serving = agent.current
            trace: list = []
            served = simulate(TRUE[serving], int(agent.configs[serving]["seqs"]), arrivals, trace=trace)
            for rows_, prompt, kv, seconds in trace:
                agent.observe_step(rows_, prompt, kv, seconds * math.exp(rng.gauss(0, 0.03)))
            for t, p, a in arrivals:
                agent.observe_request(t, p, a)
            rows["agent"].append(served)
            for name, cost in TRUE.items():
                rows[name].append(simulate(cost, int(agent.configs[name]["seqs"]), arrivals))
            clock = arrivals[-1][0] + 10.0
            before = agent.current
            target = agent.idle()
            if target is not None:
                agent.switched(target)
            if agent.current != before:
                result["switches"].append({"phase": phase, "from": before, "to": agent.current,
                                           "action": agent.history[-1].get("action")})
        summary = {name: {"tok_s": sum(r["tok_s"] for r in v) / len(v), "ttft": sum(r["ttft"] for r in v) / len(v)}
                   for name, v in rows.items()}
        best = max(TRUE, key=lambda n: summary[n]["tok_s"])
        result["phases"].append({"phase": phase, "agent": summary["agent"], "incumbent": summary["w16-fa"],
                                 "best_fixed": {"name": best, **summary[best]},
                                 "agent_vs_incumbent": summary["agent"]["tok_s"] / summary["w16-fa"]["tok_s"],
                                 "agent_vs_best": summary["agent"]["tok_s"] / summary[best]["tok_s"]})
    result["world_model"] = agent.status()["world_model"]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--windows", type=int, default=60)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    runs = [run(seed, args.windows) for seed in range(args.seeds)]
    for r in runs:
        print(json.dumps({"seed": r["seed"], "switches": r["switches"],
                          "phases": [{k: (round(v, 3) if isinstance(v, float) else
                                          ({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                                           if isinstance(v, dict) else v)) for k, v in p.items()}
                                     for p in r["phases"]]}))
    if args.out:
        args.out.write_text(json.dumps(runs, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
