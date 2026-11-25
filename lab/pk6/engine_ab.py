"""PK6 (2026-11-25): PK6 off / on in one engine process, wave by wave (the native library reads IRONMULE_PK6 at every
graph), on the benchmark's 32-request prefill shape: 32 fresh prompts a wave (the KV cleared, one packed prefill),
then `--tokens` greedy tokens, so the wave covers the prefill and the decode steps after it. Order ABBA repeated;
prompts of their own for every wave; no profiler.

  IRONMULE_GGML_PACK=1 IRONMULE_GGML_UBATCH=8192 IRONMULE_CUBLAS_MIN_ROWS=128 \
    python lab/pk6/engine_ab.py MODEL.gguf OUT.jsonl [--pairs 8] [--tokens 32]

Per wave: the prefill call (ms, rows, from the engine trace), the wave's wall time and its tokens. SUMMARY: the
median paired difference (on - off) of the prefill and of the wave, and in how many pairs on was shorter.
"""
import argparse
import json
import os
import statistics as st
import tempfile
import time

os.environ.setdefault("IRONMULE_GGML_TRACE", tempfile.mktemp(suffix=".trace"))

from ironmule.ggml_engine import GgmlEngine, Job  # noqa: E402
from ironmule_product import compare  # noqa: E402

TPL = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("out")
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--n", type=int, default=32)
    a = ap.parse_args()
    e = GgmlEngine(a.gguf, seqs=32, seq_ctx=2048, flash_attn=True, speculate=True)
    trace = os.environ["IRONMULE_GGML_TRACE"]
    out = open(a.out, "w")
    order = []
    for k in range(a.pairs):
        order += [0, 1] if k % 2 == 0 else [1, 0]
    order = [0] + order  # one discarded wave first
    rows = []
    tokens_by_arm = {0: {}, 1: {}}
    for w, arm in enumerate(order):
        os.environ["IRONMULE_PK6"] = str(arm)
        seed = 1300 + ((w - 1) // 2 if w else 999)  # a pair shares its prompts; the discarded wave has its own
        prompts = compare.performance_prompts(seed, a.n, max(a.n, 8))[:a.n]
        jobs = [Job(e.tokenize(TPL.format(p)), a.tokens) for p in prompts]
        e._clear()
        e._kept.clear()
        e._free = list(range(e.seqs))
        t0 = time.monotonic()
        e.run(jobs)
        t1 = time.monotonic()
        calls = []
        for line in open(trace):
            if line.strip() and not line.startswith("#"):
                p = line.split()
                if t0 <= float(p[0]) <= t1 and int(p[5]) > 0:
                    calls.append((round(1e3 * float(p[1]), 2), int(p[2])))
        big = [c for c in calls if c[1] > 300]
        rec = {"wave": w, "pk6": arm, "pair": (w - 1) // 2 if w else None, "wall_ms": round(1e3 * (t1 - t0), 1),
               "prefill_ms": sum(c[0] for c in big), "prefill_rows": sum(c[1] for c in big), "calls": len(calls)}
        if w:
            tokens_by_arm[arm][rec["pair"]] = [list(j.tokens) for j in jobs]
            rows.append(rec)
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print("WAVE " + json.dumps(rec), flush=True)
    os.environ["IRONMULE_PK6"] = "0"
    summary = {}
    for key in ("prefill_ms", "wall_ms"):
        d = []
        for k in range(a.pairs):
            on = [r[key] for r in rows if r["pair"] == k and r["pk6"] == 1]
            off = [r[key] for r in rows if r["pair"] == k and r["pk6"] == 0]
            if on and off:
                d.append(on[0] - off[0])
        summary[key] = {"median_on_minus_off": round(st.median(d), 2), "on_shorter": f"{sum(x < 0 for x in d)}/{len(d)}",
                        "off_median": st.median(r[key] for r in rows if r["pk6"] == 0),
                        "on_median": st.median(r[key] for r in rows if r["pk6"] == 1)}
    same = [tokens_by_arm[0].get(k) == tokens_by_arm[1].get(k) for k in range(a.pairs)]
    summary["same_tokens_pairs"] = f"{sum(same)}/{len(same)}"
    out.write(json.dumps({"summary": summary}) + "\n")
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
