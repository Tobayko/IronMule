"""PF3: the passes llama.cpp's split_equal makes of each packed prefill call, rebuilt from the engine trace (2026-11-24).

With one KV stream per sequence llama.cpp splits a call into ubatches (passes) sequentially: a pass takes the run of
consecutive sequence ids, from the first one with tokens left, that all still have tokens, and advances every one of
them by the fewest tokens any of them has left (capped at n_ubatch tokens). Every pass is a full trip through the
model (all weights read, PK2 dequantisation again). The trace's "# job" lines give each request's sequence id, prompt
length and reused prefix; the call lines give the native call's duration.

  python lab/pf3/passes.py TRACE [TRACE ...] [--ubatch 8192] [--min-rows 300]
"""
import argparse
import json
import statistics as st
from collections import defaultdict


def split_equal(seq_rows, n_ubatch):
    """[(rows of each pass, sequences in it)] for {seq: rows}, as llama.cpp's sequential split_equal."""
    left = dict(seq_rows)
    passes = []
    while any(v > 0 for v in left.values()):
        ids = sorted(s for s, v in left.items() if v > 0)
        run = [ids[0]]
        for s in ids[1:]:
            if s == run[-1] + 1:
                run.append(s)
            else:
                break
        step = min(left[s] for s in run)
        step = max(1, min(step, n_ubatch // len(run))) if len(run) <= n_ubatch else 1
        for s in run:
            left[s] -= step
        passes.append((step * len(run), len(run)))
    return passes


def calls_of(path):
    calls, jobs = [], []
    for line in open(path):
        if line.startswith("# job "):
            kv = dict(p.split("=", 1) for p in line[6:].split())
            jobs.append({"seq": int(kv["seq"]), "rows": int(kv["prompt"]) - int(kv["reused"]),
                         "prompt": int(kv["prompt"]), "reused": int(kv["reused"]), "start": float(kv["prefill_start"])})
        elif line.strip() and not line.startswith("#"):
            p = line.split()
            if len(p) > 5 and int(p[5]) > 0:
                calls.append({"start": float(p[0]), "sec": float(p[1]), "rows": int(p[2]), "seqs": int(p[3])})
    out = []
    for c in calls:
        js = [j for j in jobs if abs(j["start"] - c["start"]) < 1e-5]
        if sum(j["rows"] for j in js) != c["rows"]:
            continue  # a prompt split over calls, or a call without its jobs in this trace
        kmin = min(j["reused"] for j in js) if js else 0
        out.append({**c, "seq_rows": {j["seq"]: j["rows"] for j in js},
                    "seq_rows_uniform": {j["seq"]: j["prompt"] - kmin for j in js}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--ubatch", type=int, default=8192)
    ap.add_argument("--min-rows", type=int, default=300)
    a = ap.parse_args()
    rows = []
    for path in a.traces:
        for c in calls_of(path):
            if c["rows"] < a.min_rows:
                continue
            ps = split_equal(c["seq_rows"], a.ubatch)
            pu = split_equal(c["seq_rows_uniform"], a.ubatch)
            rows.append({"trace": path.split("/")[-1], "rows": c["rows"], "seqs": c["seqs"], "ms": round(1e3 * c["sec"], 1),
                         "passes": len(ps), "small_passes": sum(1 for r, _ in ps if r < 128),
                         "pass_rows": [r for r, _ in ps],
                         "uniform_rows": sum(c["seq_rows_uniform"].values()), "uniform_passes": len(pu)})
    for r in rows:
        print(json.dumps(r))
    by = defaultdict(list)
    for r in rows:
        if r["seqs"] >= 28:
            by[r["passes"]].append(r["ms"] / r["rows"] * 7400)
    big = [r for r in rows if r["seqs"] >= 28]
    if len(big) > 3:  # ms = a * rows + b * passes + c, least squares over the large calls
        import numpy as np
        X = np.array([[r["rows"], r["passes"], 1.0] for r in big])
        y = np.array([r["ms"] for r in big])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ coef
        pred = [coef[0] * r["uniform_rows"] + coef[1] * r["uniform_passes"] + coef[2] for r in big]
        print("FIT " + json.dumps({"ms_per_row": round(coef[0], 4), "ms_per_pass": round(coef[1], 2), "ms_const": round(coef[2], 1),
                                   "calls": len(big), "residual_median_abs_ms": round(float(np.median(np.abs(res))), 1),
                                   "residual_max_ms": round(float(np.max(res)), 1)}))
        d = [p_ - r["ms"] for p_, r in zip(pred, big)]
        print("UNIFORM_KEEP " + json.dumps({"extra_rows_median": st.median(r["uniform_rows"] - r["rows"] for r in big),
                                            "passes_now_median": st.median(r["passes"] for r in big),
                                            "passes_uniform_median": st.median(r["uniform_passes"] for r in big),
                                            "predicted_change_ms_median": round(st.median(d), 1),
                                            "predicted_change_ms_range": [round(min(d), 1), round(max(d), 1)]}))
    print("SUMMARY " + json.dumps({p: {"calls": len(v), "ms_per_7400_rows_median": round(st.median(v), 1),
                                       "min": round(min(v), 1), "max": round(max(v), 1)} for p, v in sorted(by.items())}))


if __name__ == "__main__":
    main()
