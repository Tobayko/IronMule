"""PERF1 run 18 (PERF1-Z): judge each re-measured CUDA number with the rules its notebook fixed.

Usage: python run18_summary.py RESULTS_DIR [OUT.json]

Per `cross.py` arm: per-repetition wall ratios against a reference arm of the same repetition,
their median, and the published ratio; "reproduced" within 5%, otherwise "replaced". The
exact arms must also return stock's tokens in 6 of 6 requests. For Qwen 3 8B (`perf1.py e2e`)
the decode ratio against stock is judged the same way; absolute tok/s and TTFT are reported,
not judged, because T4 cells differ by up to 2x.
"""
import json
import statistics as st
import sys
from pathlib import Path

TOLERANCE = 0.05
# (model, arm, reference arm): (published ratio, source, exact arm)
CROSS = {
    ("12b", "ironmule_exact", "stock"): (0.9736, "PORT1", True),
    ("12b", "ironmule_fp32", "stock"): (0.4905, "PORT1", False),
    ("12b", "ironmule_native", "ironmule_fp32_plain"): (0.4011, "PERF1 run 17", False),
    ("12b", "ironmule_native_compiled", "ironmule_fp32_plain"): (0.3695, "PERF1 run 17", False),
    ("12b", "ironmule_native_compiled", "stock"): (0.4905 * 0.3695, "projection, PORT1 x run 17", False),
    ("1b", "ironmule", "stock"): (0.5502, "PORT1", True),
    ("4b", "ironmule_exact", "stock"): (0.9511, "PORT1", True),
}
QWEN_DECODE = (32.47 / 6.34, "PERF1 run 2, kernel+p16 over stock")


def verdict(new, published):
    return {"median": new, "published": published, "deviation": new / published - 1,
            "verdict": "reproduced" if abs(new / published - 1) <= TOLERANCE else "replaced"}


def main(results, out=None):
    results = Path(results)
    report = {"cross": [], "qwen3_8b": {}}
    for (model, arm, reference), (published, source, exact) in CROSS.items():
        cross = json.loads((results / f"cross-gemma3-{model}.json").read_text())
        walls = {name: {r["rep"]: r["median_ms"] for r in rows if r.get("median_ms")}
                 for name, rows in cross["runs"].items()}
        ratios = [walls[arm][rep] / walls[reference][rep] for rep in walls[arm] if rep in walls[reference]]
        row = {"model": model, "arm": arm, "reference": reference, "source": source,
               "ratios": [round(r, 4) for r in ratios], "identical_to_stock": cross["summary"][arm]["identical_requests"],
               "deterministic": cross["summary"][arm]["deterministic_across_processes"],
               **verdict(st.median(ratios), published)}
        if exact:
            row["exact_holds"] = row["identical_to_stock"] == 6
        report["cross"].append(row)
    arms = {name: json.loads((results / f"e2e-qwen3-8b-{name}.json").read_text())
            for name in ("stock", "kernel+p16", "kernel+p16-pinned")}
    stock = arms["stock"]
    for name, arm in arms.items():
        divergence = next((i for i, (a, b) in enumerate(zip(arm["tokens"], stock["tokens"])) if a != b), None)
        report["qwen3_8b"][name] = {"decode_tps_median": arm["decode_tps_median"],
                                    "ttft_ms_median": arm["ttft_ms_median"],
                                    "first_token_differing_from_stock": divergence}
    ratio = arms["kernel+p16"]["decode_tps_median"] / stock["decode_tps_median"]
    report["qwen3_8b"]["decode_ratio"] = {"source": QWEN_DECODE[1], **verdict(ratio, QWEN_DECODE[0])}
    text = json.dumps(report, indent=1)
    if out:
        Path(out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:3])
