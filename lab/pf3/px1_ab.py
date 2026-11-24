"""PF3: PX1 against the reference in one engine process, same prompt work, controlled cache state (2026-11-24).

The engine as the server builds it (flash attention, one KV stream per sequence, 32 sequences, speculation on, packing
and n_ubatch 8192 from the environment, the W4R library and table of the server's state). One measured wave is the
HTTP benchmark's 32 prompts of one repeat (lab/pk1/http_bench.py's seeds) with 32-token answers, all handed to the
engine at once, so a wave is the packed prefill call(s) plus the decode steps; its time is the engine's wall time
around `run`, without profiler (the engine trace is on in both arms: one line per native call).

Cache state: before every measured wave the engine's KV is cleared and the previous repeat's 32 prompts run once (the
primer, 2-token answers, PX1 off), so each stream holds what the benchmark's preceding wave left there and the
measured prompts find the same kept prefixes (0-12 template tokens) in both arms. Order alternates in ABBA blocks.
The first wave after the engine start (no kept prefixes, PX1 has nothing to do) is timed once apart, as "cold".

Per wave: wall time, the passes of each packed call (trace "px1" notes), the packed calls' durations, and the answer
tokens; the arms' tokens are compared request by request (PX1 recomputes kept tokens, so small numeric differences
can change a token; the factual check runs in the HTTP stage).

  python lab/pf3/px1_ab.py MODEL.gguf OUT.jsonl [--pairs 12] [--width 32]
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
BASE_SEED = 20261119  # lab/pk1/http_bench.py: seed = base + repeat * 1009 + width


def prompts(e, rep, width):
    seed = BASE_SEED + rep * 1009 + width
    return [e.tokenize(TPL.format(p)) for p in compare.performance_prompts(seed, width, max(width, 8))][:width]


def trace_since(path, t0, t1):
    calls, notes = [], []
    for line in open(path):
        if line.startswith("# px1 "):
            p = line.split()
            if t0 <= float(p[-1]) <= t1:
                notes.append({"passes": int(p[3]), "even": int(p[5]), "extra": int(p[7]), "on": int(p[9])})
        elif line.strip() and not line.startswith("#"):
            p = line.split()
            if t0 <= float(p[0]) <= t1:
                calls.append({"ms": round(1e3 * float(p[1]), 2), "rows": int(p[2]), "prompt_rows": int(p[5])})
    return calls, notes


def wave(e, ids, limit, px1):
    e.px1 = px1
    jobs = [Job(list(x), limit) for x in ids]
    t0 = time.monotonic()
    e.run(jobs)
    t1 = time.monotonic()
    return t0, t1, jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("out")
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--limit", type=int, default=32)
    a = ap.parse_args()
    trace = os.environ["IRONMULE_GGML_TRACE"]
    t = time.perf_counter()
    e = GgmlEngine(a.gguf, seqs=32, seq_ctx=2048, flash_attn=True, speculate=True)
    head = {"start_s": round(time.perf_counter() - t, 2), "pack": e.pack, "pk2": e.pk2, "ubatch": os.environ.get(
        "IRONMULE_GGML_UBATCH"), "kg_tokens": (e.kg or {}).get("tokens_served")}
    print("ENGINE " + json.dumps(head, default=str), flush=True)
    out = open(a.out, "w")
    out.write(json.dumps({"engine": head}, default=str) + "\n")

    # cold: the first wave after the start, repeat 0's prompts, nothing kept
    t0, t1, jobs = wave(e, prompts(e, 0, a.width), a.limit, False)
    calls, notes = trace_since(trace, t0, t1)
    rec = {"kind": "cold", "rep": 0, "wall_s": round(t1 - t0, 4), "calls": [c for c in calls if c["prompt_rows"]],
           "px1": notes}
    out.write(json.dumps(rec) + "\n")
    print("COLD " + json.dumps(rec), flush=True)

    order = []
    for i in range(a.pairs):
        order += [(i, True), (i, False)] if i % 2 else [(i, False), (i, True)]  # ABBA
    tokens = {}
    walls = {}
    for i, arm in order:
        rep = 1 + i % 2  # the benchmark's repeats 1 and 2, each primed by the repeat before it
        e._clear()
        wave(e, prompts(e, rep - 1, a.width), 2, False)  # primer
        t0, t1, jobs = wave(e, prompts(e, rep, a.width), a.limit, arm)
        calls, notes = trace_since(trace, t0, t1)
        pre = [c for c in calls if c["prompt_rows"]]
        rec = {"kind": "ab", "pair": i, "rep": rep, "px1": arm, "wall_s": round(t1 - t0, 4),
               "prefill_ms": round(sum(c["ms"] for c in pre), 2), "prefill_calls": pre, "notes": notes,
               "reused": sum(j.reused for j in jobs), "tokens": [j.tokens for j in jobs]}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        tokens[(i, arm)] = rec["tokens"]
        walls[(i, arm)] = rec["wall_s"]
        print("AB " + json.dumps({k: v for k, v in rec.items() if k not in ("tokens", "prefill_calls")}), flush=True)

    d = [walls[(i, False)] - walls[(i, True)] for i in range(a.pairs)]
    same = sum(tokens[(i, False)][r] == tokens[(i, True)][r] for i in range(a.pairs) for r in range(a.width))
    summary = {"pairs": a.pairs, "saving_ms": sorted(round(1e3 * x, 1) for x in d),
               "saving_median_ms": round(1e3 * st.median(d), 1), "px1_faster": sum(x > 0 for x in d),
               "ref_wall_median_s": round(st.median(walls[(i, False)] for i in range(a.pairs)), 4),
               "px1_wall_median_s": round(st.median(walls[(i, True)] for i in range(a.pairs)), 4),
               "identical_answers": f"{same}/{a.pairs * a.width}", "px1_evened": getattr(e, "px1_evened", 0)}
    out.write(json.dumps({"summary": summary}) + "\n")
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
