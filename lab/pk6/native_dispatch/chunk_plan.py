"""Use real mixed calls on this device to choose a bounded shared-pool prefill trial."""

import argparse
from pathlib import Path
from statistics import median

from ironmule.ggml_agent import StepCost
from lab.pk6.mixed_progress import grade, learning, plan
from lab.pk6.mixed_progress.schedule import slices
from lab.pk6.native_dispatch.score import waves, read
from lab.pk6.warm_repeat import save


def simulate(model, lengths, budget, decode_s):
    pos = [0] * len(lengths)
    times = [[] for _ in lengths]
    now = 0.0
    while any(len(t) < 32 for t in times):
        running = [i for i, t in enumerate(times) if 0 < len(t) < 32]
        queued = sorted(
            (i for i in range(len(times)) if not times[i]),
            key=lambda i: lengths[i] - pos[i],
        )
        pieces = list(slices(queued, lengths, pos, budget - len(running)))
        prompt = sum(b - a for _, a, b in pieces)
        now += model.predict(len(running), prompt, 0) if prompt else decode_s
        for i in running:
            times[i].append(now)
        for i, _, end in pieces:
            pos[i] = end
            if end == lengths[i]:
                times[i].append(now)
    return plan.metrics(
        times, [t[-1] for t in times], [32] * len(times), [0] * len(times)
    )


def propose(primary):
    blocks = {}
    for arm in ("R1", "R2"):
        rows = waves(primary, arm)
        result = grade.bindings(primary, arm, rows)
        blocks[arm] = (
            [s for c in result["cells"] if c["repeat"] < 3 for s in c["mixed"]],
            result["trace_sha256"],
            rows,
        )
    if {w["seed"] for w in blocks["R1"][2]} & {w["seed"] for w in blocks["R2"][2]}:
        raise ValueError("overlapping input seeds")
    fitted = learning.fit(
        blocks["R1"][0], blocks["R2"][0], plan.plan()["step_cost_theta"]
    )
    model = StepCost.from_json(fitted["updated"])
    groups = [w for w in blocks["R1"][2] if w["width"] == 32 and w["repeat"] < 3]
    cadence = median(
        median(y - x for x, y in zip(r["chunk_times"], r["chunk_times"][1:]))
        for w in groups
        for r in w["requests"]
        if len(r["chunk_times"]) >= 2
    )
    actual = grade.summarize([grade.measure(w) for w in groups])
    alternatives = []
    for budget in (256, 384, 512, 768, 1024, 1536):
        samples = [
            simulate(
                model, [r["prompt_tokens"] for r in w["requests"]], budget, cadence
            )
            for w in groups
        ]
        scores = {k: median(s[k] for s in samples) for k in samples[0]}
        reg = max(
            actual[k] / v if k == "tok_s" else v / actual[k] for k, v in scores.items()
        )
        alternatives.append({"budget": budget, "forecast": scores, "worst_ratio": reg})
    best = min(alternatives, key=lambda r: r["worst_ratio"])
    meta = read(primary / "target.json")
    return {
        "hardware": meta["hardware"],
        "gguf": meta["gguf_sha256"],
        "rows": best["budget"],
        "source_traces": {a: b[1] for a, b in blocks.items()},
        "model": fitted,
        "simulated": alternatives,
        "scope": "One device and three 32-client discovery waves. Cold prefixes, simultaneous arrivals and constant decode cadence are forecast assumptions. R2 scores model error only.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("primary", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    save(a.out, propose(a.primary))
