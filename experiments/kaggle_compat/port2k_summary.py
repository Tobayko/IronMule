"""PORT2-K: judge `quality.py` single-precision gates, BOS on every chunk, as the plan table does.

Usage: python port2k_summary.py RESULTS_DIR [OUT.json]

For every `quality-KEY-bf16.json` in RESULTS_DIR and each `quality-KEY-float32.json` /
`quality-KEY-float16.json` beside it: the perplexity ratio exp(mean plan NLL - mean bf16 NLL)
over the paired chunks and a 10 000-draw paired chunk bootstrap with the seed
`tests/test_numeric_plans.py` re-derives the table with. A plan passes when the upper bound
(97.5th percentile) is at most 1.005 and no NLL is non-finite; it is refused when the lower
bound lies above 1.005 (a measured ruin), as `numeric_plans.check` treats it.
"""
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

BOUND = 1.005
SEED = 20260916  # tests/test_numeric_plans.py BOOTSTRAP_SEED
DRAWS = 10000


def judge(bf16, other, key):
    rows = [(a["nll_bf16"], b[key]) for a, b in zip(bf16["rows"], other["rows"])]

    def ratio(sample):
        return math.exp(st.mean(b for _, b in sample) - st.mean(a for a, _ in sample))

    rng = random.Random(SEED)
    draws = sorted(ratio([rng.choice(rows) for _ in rows]) for _ in range(DRAWS))
    low, high = draws[250], draws[9750]
    nonfinite = sum(not math.isfinite(v) for pair in rows for v in pair)
    verdict = ("fails: non-finite" if nonfinite else "passes" if high <= BOUND
               else "refused: lower bound above the bound" if low > BOUND else "inconclusive")
    return {"chunks": len(rows), "bos": [bf16.get("bos"), other.get("bos")],
            "bf16_perplexity": math.exp(st.mean(a for a, _ in rows)),
            "plan_perplexity": math.exp(st.mean(b for _, b in rows)),
            "ratio": ratio(rows), "interval": [low, high], "nonfinite": nonfinite, "verdict": verdict}


def main(results, out=None):
    results = Path(results)
    report = {}
    for bf16_path in sorted(results.glob("quality-*-bf16.json")):
        key = bf16_path.name[len("quality-"):-len("-bf16.json")]
        bf16 = json.loads(bf16_path.read_text())
        for plan in ("float32", "float16"):
            other_path = results / f"quality-{key}-{plan}.json"
            if other_path.exists():
                report[f"{key}/{plan}"] = judge(bf16, json.loads(other_path.read_text()), f"nll_{plan}")
    text = json.dumps(report, indent=1)
    if out:
        Path(out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:3])
