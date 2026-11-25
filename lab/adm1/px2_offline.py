"""PX2 (2026-11-25): PX1's fixed limit against PX2's cost decision on the packed calls of earlier traces (offline).

Every large packed call of the traces is rebuilt from its "# job" lines (sequence, prompt, reused prefix); both rules
choose their level and the pods' fitted pass costs (16.45 ms + 0.978 ms a row under 128 rows, 91.15 ms + 0.2869 ms a
row from 128 on; 186 calls of pods 32/33) price the result.

  python lab/adm1/px2_offline.py lab/pk1/results/pod3[23]/raw/trace_*.txt
"""
import statistics as st
import sys

sys.path.insert(0, "lab/pf3")
from passes import calls_of  # noqa: E402

from ironmule.ggml_engine import PX1_ROWS, PassCosts, split_shapes  # noqa: E402

small = lambda r: (16.45 + 0.978 * r) / 1000  # noqa: E731
big = lambda r: (91.15 + 0.2869 * r) / 1000  # noqa: E731
costs = PassCosts({1: small(1), 8: small(8), 16: small(16), 32: small(32), 127: small(127), 128: big(128),
                   256: big(256), 1024: big(1024)})


def level_rows(c, level):
    return {s: r + max(0, c["keep"][s] - level) for s, r in c["seq_rows"].items()}


gains = {"px1": [], "px2": []}
for path in sys.argv[1:]:
    jobs = {}
    for line in open(path):
        if line.startswith("# job "):
            kv = dict(p.split("=", 1) for p in line[6:].split())
            jobs[(int(kv["seq"]), float(kv["prefill_start"]))] = int(kv["reused"])
    for c in calls_of(path):
        if c["rows"] < 300 or c["seqs"] < 2:
            continue
        c["keep"] = {s: jobs.get((s, c["start"]), 0) for s in c["seq_rows"]}
        if c["sec"] > 2.7 and c["rows"] < 8000:
            continue
        keeps = list(c["keep"].values())
        now = split_shapes(c["seq_rows"], 8192)
        now_s = costs.split(now)
        lo = min(keeps)
        ev = split_shapes(level_rows(c, lo), 8192)
        extra = sum(k - lo for k in keeps)
        p1 = ev if len(ev) < len(now) and extra <= PX1_ROWS * (len(now) - len(ev)) else now
        best, best_s = now, now_s
        for lev in sorted(set(keeps))[:-1]:
            sh = split_shapes(level_rows(c, lev), 8192)
            if costs.split(sh) < best_s:
                best, best_s = sh, costs.split(sh)
        p2 = best if now_s - best_s >= 0.002 else now
        g1, g2 = 1e3 * (now_s - costs.split(p1)), 1e3 * (now_s - costs.split(p2))
        gains["px1"].append(g1)
        gains["px2"].append(g2)
        print(f"{path.split('/')[-1]} rows {c['rows']} passes {len(now)} px1 {len(p1)} px2 {len(p2)} extra {extra} "
              f"saving px1 {g1:.0f} ms px2 {g2:.0f} ms (call {1e3 * c['sec']:.0f} ms)")
print({k: {"calls": len(v), "median_ms": round(st.median(v), 1), "mean_ms": round(st.mean(v), 1)}
       for k, v in gains.items()})
