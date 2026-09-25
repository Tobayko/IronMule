"""BACKLOG1: the verdicts of PERF1-M, PERF1-T, PORT1-F, PERF1-N and PERF1-L, with the harness's rules.

The gate bootstrap uses quality.py's seed; on PERF1 run 5's 8B decode files it gives the published
ratio 0.997356 exactly and an interval that differs from the published one in the fifth decimal.

Usage: python backlog1_summary.py RESULTS_DIR [OUT.json]    (BACKLOG1 or BACKLOG2)
"""
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

BOUND = 1.005


def gate(results):
    stock = json.loads((results / "gate-qwen3-14b-stock-decode.json").read_text())
    kernel = json.loads((results / "gate-qwen3-14b-kernel-decode.json").read_text())
    pairs = list(zip(stock["chunk_nll"], kernel["chunk_nll"]))

    def ratio(sample):
        return math.exp(st.mean(k for _, k in sample) - st.mean(s for s, _ in sample))

    rng = random.Random(20260915)
    boot = sorted(ratio([rng.choice(pairs) for _ in pairs]) for _ in range(10000))
    nonfinite = stock["nonfinite"] + kernel["nonfinite"]
    low, high = boot[250], boot[9750]
    return {"chunks": len(pairs), "ratio": ratio(pairs), "interval": [low, high], "nonfinite": nonfinite,
            "routed": kernel.get("routed"), "verdict": "passes" if high <= BOUND and not nonfinite else "fails",
            "seconds": [stock["seconds"], kernel["seconds"]]}


def determinism(results):
    cross = json.loads((results / "cross-qwen35-9b-graphs0.json").read_text())
    runs = cross["runs"]
    reference = next((r["tokens"] for r in runs["stock"] if r.get("tokens")), None)
    stock = {r["rep"]: r["median_ms"] for r in runs["stock"] if r.get("median_ms")}
    arms = {}
    for name, rows in runs.items():
        done = [r for r in rows if r.get("tokens")]
        digests = {r["outputs_sha256"] for r in done}
        tokens = done[0]["tokens"] if done else None
        ratios = [r["median_ms"] / stock[r["rep"]] for r in done if r["rep"] in stock]
        arms[name] = {"processes": len(done), "failed": len(rows) - len(done),
                      "deterministic": len(digests) == 1 if done else None,
                      "identical_to_stock": (sum(a == b for a, b in zip(tokens, reference))
                                             if tokens and reference else None),
                      "ratios": [round(r, 4) for r in ratios],
                      "median_ratio": st.median(ratios) if ratios else None}
    return {"reps": cross["reps"], "arms": arms}


def tune(results):
    report = json.loads(next(results.glob("backlog*-result.json")).read_text())
    stage = report["stages"].get("tune_gemma3-4b")
    return None if stage is None else {"exit": stage["exit"], "seconds": stage.get("seconds"),
                                       "tail": stage.get("tail", "")[-600:]}


def probes(results):
    fusion = json.loads((results / "probe-perf1n.json").read_text())["results"]
    by_path = {}
    for row in fusion:
        entry = by_path.setdefault(row["path"], {"cases": 0, "bit_equal": 0, "max_abs_diff": 0.0})
        entry["cases"] += 1
        entry["bit_equal"] += row["bit_equal"]
        entry["max_abs_diff"] = max(entry["max_abs_diff"], row["max_abs_diff"])
    k32 = {}
    for arith in ("free", "pinned"):
        rows = json.loads((results / f"probe-perf1l-{arith}.json").read_text())["results"]
        k32[arith] = {"cases": len(rows), "bit_equal": sum(r["bit_equal"] for r in rows),
                      "max_rel_diff": max(r["max_rel_diff"] for r in rows)}
    return {"PERF1-N": by_path, "PERF1-L": k32}


def main(results, out=None):
    results = Path(results)
    verdicts = {}
    for name, fn in (("probes", probes), ("PERF1-M", gate), ("PERF1-T", determinism), ("PORT1-F", tune)):
        try:
            verdicts[name] = fn(results)
        except FileNotFoundError as missing:
            verdicts[name] = {"missing": str(missing.filename)}
    text = json.dumps(verdicts, indent=1)
    if out:
        Path(out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:3])
