"""OSS1: compare the expert routing and NLL that `moe_routing.py` recorded per dtype.

Usage: python oss1_summary.py RESULTS_DIR [OUT.json]

For every pair of recorded runs and every chunk: the share of (layer, position) cells whose
sorted top-k experts differ, and the NLL difference. For flipped cells against the first
bf16 run, where the first run's router margin lay: the margin is the gap between the k-th
and the (k+1)-th router logit, so a small one means the choice was nearly a tie.
"""
import json
import statistics as st
import sys
from itertools import combinations
from pathlib import Path


def load(results):
    runs = {}
    for path in sorted(Path(results).glob("routing-*.json")):
        runs[path.stem.removeprefix("routing-")] = json.loads(path.read_text())
    return runs


def compare(a, b):
    per_chunk = []
    for ra, rb in zip(a["rows"], b["rows"]):
        cells = flips = 0
        flipped_margins, all_margins = [], []
        for la, lb, ma in zip(ra["experts"], rb["experts"], ra["margins"]):
            for ea, eb, margin in zip(la, lb, ma):
                cells += 1
                all_margins.append(margin)
                if ea != eb:
                    flips += 1
                    flipped_margins.append(margin)
        per_chunk.append({
            "chunk": ra["chunk"], "flip_rate": flips / cells, "flips": flips, "cells": cells,
            "dnll": rb["nll"] - ra["nll"],
            "median_margin_all": st.median(all_margins),
            "median_margin_flipped": st.median(flipped_margins) if flipped_margins else None,
        })
    return per_chunk


def main(results, out=None):
    runs = load(results)
    report = {"runs": {name: {"dtype": run["dtype"], "nll": [r["nll"] for r in run["rows"]],
                              "chunks": [r["chunk"] for r in run["rows"]]} for name, run in runs.items()},
              "pairs": {}}
    for left, right in combinations(sorted(runs), 2):
        report["pairs"][f"{left}|{right}"] = compare(runs[left], runs[right])
    text = json.dumps(report, indent=1)
    if out:
        Path(out).write_text(text + "\n")
    for name, run in report["runs"].items():
        print(f"{name:8s} {run['dtype']:8s} chunks {run['chunks']} nll {[round(v, 4) for v in run['nll']]}")
    for pair, rows in report["pairs"].items():
        print(pair)
        for row in rows:
            flipped = row["median_margin_flipped"]
            print(f"  chunk {row['chunk']:3d} flips {row['flip_rate']:.4%} ({row['flips']}/{row['cells']}) "
                  f"dNLL {row['dnll']:+.4f} margin median all {row['median_margin_all']:.4f} "
                  f"flipped {flipped if flipped is None else round(flipped, 4)}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
