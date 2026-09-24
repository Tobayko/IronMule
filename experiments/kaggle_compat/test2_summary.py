"""TEST2: judge each re-measured ratio against the published one, with the rules fixed before the run.

Usage: python test2_summary.py RESULTS_DIR [OUT.json]

Per model and arm: the per-repetition ratios against stock of the same repetition, their
median and range, the published ratio and the relative deviation; "reproduced" when the new
median lies within 5% of the published one, otherwise "replaced". Stock's own spread across
repetitions is this regime's noise, and its output digest across processes the model's
determinism. For an arm whose tokens differ from stock, the first differing position of each
request is reported as a description, not a verdict.
"""
import json
import statistics as st
import sys
from pathlib import Path

PUBLISHED = {
    ("qwen3-8b", "ironmule_native"): (0.2013, "PERF1 run 7"),
    ("qwen3-8b", "ironmule_fp32"): (0.5346, "PORT2 run 6"),
    ("qwen3-8b", "ironmule_fp16"): (0.3100, "PORT2 run 6"),
    ("qwen3-14b", "ironmule_native"): (0.2078, "PERF1 run 7"),
    ("qwen3-14b", "ironmule_fused_fp32"): (0.5157, "PORT2 run 4"),
    ("gptoss-20b", "ironmule_fp32"): (0.2818, "PORT2 run 6"),
    ("gptoss-20b", "ironmule_fp16"): (0.1991, "PORT2 run 6"),
}
TOLERANCE = 0.05


def first_divergence(a, b):
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def judge(key, cross):
    runs, summary = cross["runs"], cross["summary"]
    stock_walls = [r["median_ms"] for r in runs["stock"] if r.get("median_ms")]
    reference = next((r["tokens"] for r in runs["stock"] if r.get("tokens")), None)
    model = {
        "model_id": cross["model_id"], "revision": cross["revision"], "reps": cross["reps"],
        "stock_walls_s": [round(w / 1000, 2) for w in stock_walls],
        "stock_spread": (max(stock_walls) / min(stock_walls) - 1) if len(stock_walls) > 1 else None,
        "stock_deterministic": summary["stock"]["deterministic_across_processes"],
        "arms": {},
    }
    for name, s in summary.items():
        if name == "stock":
            continue
        pairs = s["ratio_vs_reference_per_rep"]
        tokens = next((r["tokens"] for r in runs[name] if r.get("tokens")), None)
        arm = {
            "ratios": [round(p, 4) for p in pairs], "median": s["median_ratio"],
            "min": min(pairs) if pairs else None, "max": max(pairs) if pairs else None,
            "speedup": 1 / s["median_ratio"] if s["median_ratio"] else None,
            "identical_requests": s["identical_requests"],
            "deterministic": s["deterministic_across_processes"], "failed": s["failed_processes"],
            "first_divergence": ([first_divergence(t, r) for t, r in zip(tokens, reference)]
                                 if tokens and reference else None),
        }
        published = PUBLISHED.get((key, name))
        if published and s["median_ratio"]:
            ratio, source = published
            arm.update(published=ratio, source=source, deviation=s["median_ratio"] / ratio - 1,
                       verdict="reproduced" if abs(s["median_ratio"] / ratio - 1) <= TOLERANCE else "replaced")
        model["arms"][name] = arm
    return model


def main(results, out=None):
    report = {}
    for path in sorted(Path(results).glob("cross-*.json")):
        key = path.stem.removeprefix("cross-")
        report[key] = judge(key, json.loads(path.read_text()))
    text = json.dumps(report, indent=1)
    if out:
        Path(out).write_text(text + "\n")
    for key, model in report.items():
        spread = model["stock_spread"]
        print(f"{key}: reps {model['reps']}, stock {model['stock_walls_s']} s "
              f"(spread {spread:.2%})" if spread is not None else f"{key}: reps {model['reps']}",
              f"stock deterministic: {model['stock_deterministic']}")
        for name, arm in model["arms"].items():
            print(f"  {name:22s} median {arm['median']:.4f} ({arm['speedup']:.2f}x) range "
                  f"[{arm['min']:.4f}; {arm['max']:.4f}] published {arm.get('published')} "
                  f"({arm.get('deviation', 0):+.1%}) -> {arm.get('verdict')}; identical "
                  f"{arm['identical_requests']}/6, first divergence {arm['first_divergence']}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
