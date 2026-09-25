"""Judge `perf1.py nll` gate pairs: perplexity ratio candidate / stock with the harness's rules.

Usage: python gate_summary.py RESULTS_DIR KEY [OUT.json]

For every `gate-KEY-stock-PATH.json` in RESULTS_DIR and the one candidate file of the same key
and path: the ratio exp(mean candidate chunk NLL - mean stock chunk NLL) and a 10 000-sample
paired chunk bootstrap with quality.py's seed. A path passes when the upper bound (97.5th
percentile) is at most 1.005 and neither side has a non-finite value. The same computation as
`backlog1_summary.gate`, for any model key.
"""
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

BOUND = 1.005
SEED = 20260915


def judge(stock, candidate):
    pairs = list(zip(stock["chunk_nll"], candidate["chunk_nll"]))

    def ratio(sample):
        return math.exp(st.mean(c for _, c in sample) - st.mean(s for s, _ in sample))

    rng = random.Random(SEED)
    boot = sorted(ratio([rng.choice(pairs) for _ in pairs]) for _ in range(10000))
    nonfinite = stock["nonfinite"] + candidate["nonfinite"]
    low, high = boot[250], boot[9750]
    return {"arm": candidate["arm"], "chunks": len(pairs), "bos": [stock.get("bos"), candidate.get("bos")],
            "stock_perplexity": math.exp(st.mean(stock["chunk_nll"])),
            "candidate_perplexity": math.exp(st.mean(candidate["chunk_nll"])),
            "ratio": ratio(pairs), "interval": [low, high], "nonfinite": nonfinite,
            "verdict": "passes" if high <= BOUND and not nonfinite else "fails",
            "seconds": [stock["seconds"], candidate["seconds"]]}


def main(results, key, out=None):
    results = Path(results)
    report = {}
    for stock_path in sorted(results.glob(f"gate-{key}-stock-*.json")):
        path_mode = stock_path.stem.rsplit("-", 1)[-1]
        candidates = [p for p in results.glob(f"gate-{key}-*-{path_mode}.json") if p != stock_path]
        if len(candidates) != 1:
            report[path_mode] = {"missing": [p.name for p in candidates]}
            continue
        report[path_mode] = judge(json.loads(stock_path.read_text()), json.loads(candidates[0].read_text()))
    text = json.dumps(report, indent=1)
    if out:
        Path(out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:4])
