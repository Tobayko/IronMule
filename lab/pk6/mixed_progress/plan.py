"""Choose one mixed prefill row budget from a past A100 lease. Forecasts only."""

import json
import math
from statistics import median

from ironmule.ggml_agent import StepCost
from lab.pk6.first_token.phases import HISTORICAL, RESULTS, clients
from lab.pk6.mixed_progress.schedule import slices


def metrics(times, ends, tokens, start):
    first = sorted(t[0] - s for t, s in zip(times, start))
    eighth = [t[7] - s for t, s in zip(times, start)]
    gaps = sorted(max(b - a for a, b in zip(t, t[1:])) for t in times)
    ix = math.ceil(0.95 * len(times)) - 1
    return {
        "first": median(first),
        "first_p95": first[ix],
        "eighth": median(eighth),
        "gap_p95": gaps[ix],
        "end_p95": sorted(e - s for e, s in zip(ends, start))[ix],
        "tok_s": sum(tokens) / (max(ends) - min(start)),
    }


def observed(label):
    records = clients(HISTORICAL, label)
    out = []
    for i in range(0, len(records), 32):
        wave = records[i : i + 32]
        out.append(
            metrics(
                [r["chunk_times"] for r in wave],
                [r["t_done"] for r in wave],
                [r["tokens"] for r in wave],
                [r["t_send"] for r in wave],
            )
        )
    return out


def typical(waves):
    return {k: median(row[k] for row in waves) for k in waves[0]}


def forecast(model, budget, decode_s, host_s):
    lengths = [33, 444] * 16
    positions = [0] * 32
    waiting = sorted(range(32), key=lengths.__getitem__)
    times = [[] for _ in lengths]
    now = host_s
    while any(len(t) < 32 for t in times):
        running = [i for i, t in enumerate(times) if 0 < len(t) < 32]
        chunks = list(slices(waiting, lengths, positions, budget - len(running)))
        prompts = sum(end - begin for _, begin, end in chunks)
        now += model.predict(0, prompts + len(running), 0) if prompts else decode_s
        for i in running:
            times[i].append(now)
        for i, _, end in chunks:
            positions[i] = end
            if end == lengths[i]:
                times[i].append(now)
        waiting = [i for i in waiting if positions[i] < lengths[i]]
    return metrics(times, [t[-1] for t in times], [32] * 32, [0] * 32)


def plan():
    directory = RESULTS / "a100-first-token"
    slopes, intercepts, periods, host = [], [], [], []
    for arm in ("C1", "C2"):
        native = json.loads((directory / f"native-{arm}.json").read_bytes())
        calls = native["calls"]
        for short, long in zip(calls[::2], calls[1::2], strict=True):
            slope = (long["seconds"] - short["seconds"]) / (
                long["rows"] - short["rows"]
            )
            slopes.append(slope)
            intercepts.append(short["seconds"] - slope * short["rows"])
        records = clients(directory, arm)
        for r in records:
            job = native["jobs"][r["server_id"]]
            periods.extend(
                b - a for a, b in zip(r["chunk_times"][8:], r["chunk_times"][9:])
            )
            host.append(job["seen"] - r["t_send"] + r["chunk_times"][0] - job["first"])
    model = StepCost([median(intercepts), 0.0, 1000 * median(slopes), 0.0])
    target = typical(observed("vllm"))
    reference = typical(observed("K1") + observed("K2"))
    options = []
    for budget in (512, 1024, 1536, 2048, 3072, 4096):
        costs = forecast(model, budget, median(periods), median(host))
        # Rank the largest relative miss, preserving the direction of throughput.
        ratios = [
            target[k] / v if k == "tok_s" else v / target[k] for k, v in costs.items()
        ]
        options.append(
            {"rows": budget, "forecast": costs, "worst_ratio_to_vllm": max(ratios)}
        )
    chosen = min(options, key=lambda o: o["worst_ratio_to_vllm"])
    return {
        "source": directory.name,
        "source_trace_sha256": [
            json.loads((directory / f"native-{arm}.json").read_bytes())["trace_sha256"]
            for arm in ("C1", "C2")
        ],
        "historical_measured": {"vllm": target, "ironmule": reference},
        "simulated": options,
        "selected_rows": chosen["rows"],
        "step_cost_theta": model.theta,
        "decode_s": median(periods),
        "host_s": median(host),
        "uncertainty": "Mixed prompt/decode kernels were unobserved. Decode uses measured streaming cadence. "
        "Forecasts assume 32 fixed output tokens, cold identical lengths and constant call overhead.",
    }


if __name__ == "__main__":
    print(json.dumps(plan(), indent=2, allow_nan=False))
