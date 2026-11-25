"""PS1 (2026-11-25): one native pass for a packed prefill against the reference, in one engine process, same prompt
work and controlled cache state (after lab/pf3/px1_ab.py).

The engine as the server builds it (flash attention, one KV stream per sequence, 32 sequences, speculation on, packing
and n_ubatch from the environment, the W4R library and table of the server's state). A measured wave is the HTTP
benchmark's 32 prompts of one repeat with 32-token answers, all handed to the engine at once. Before every measured
wave the KV is cleared and the previous repeat's prompts run once (2-token answers, the reference arm), so the
streams hold the kept prefixes the benchmark's preceding wave leaves (0-12 template tokens) in every arm.

Arms (rotated every round): R = separate passes, PX2 evens the kept prefixes (the current default before PS1);
S = PS1, every kept prefix used. Per wave: wall time, the packed prefill call(s) (ms, rows), the PX notes (passes,
rows recomputed or kept), rows reused, answer tokens (compared request by request between the arms).

  python lab/ps1/engine_ab.py MODEL.gguf OUT.jsonl [--rounds 10] [--width 32]
"""
import argparse
import json
import os
import statistics as st
import tempfile
import time

os.environ.setdefault("IRONMULE_GGML_TRACE", tempfile.mktemp(suffix=".trace"))

from ironmule import ggml_engine as ge  # noqa: E402
from ironmule.ggml_engine import GgmlEngine, Job  # noqa: E402
from ironmule_product import compare  # noqa: E402

TPL = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
BASE_SEED = 20261119  # lab/pk1/http_bench.py: seed = base + repeat * 1009 + width
ARMS = {"R": {"PS1": "0"}, "S": {"PS1": "1"}}


def prompts(e, rep, width):
    seed = BASE_SEED + rep * 1009 + width
    return [e.tokenize(TPL.format(p)) for p in compare.performance_prompts(seed, width, max(width, 8))][:width]


def set_arm(name):
    os.environ["IRONMULE_PS1"] = ARMS[name]["PS1"]
    ge.PS1 = ARMS[name]["PS1"] == "1"


def trace_since(path, t0, t1):
    calls, notes = [], []
    for line in open(path):
        if line.startswith("# px1 "):
            p = line.split()
            if t0 <= float(p[-1]) <= t1:
                notes.append(" ".join(p[2:-1]))
        elif line.strip() and not line.startswith("#"):
            p = line.split()
            if t0 <= float(p[0]) <= t1:
                calls.append({"ms": round(1e3 * float(p[1]), 2), "rows": int(p[2]), "prompt_rows": int(p[5])})
    return calls, notes


def wave(e, ids, limit):
    jobs = [Job(list(x), limit) for x in ids]
    t0 = time.monotonic()
    e.run(jobs)
    return t0, time.monotonic(), jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("out")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--limit", type=int, default=32)
    a = ap.parse_args()
    trace = os.environ["IRONMULE_GGML_TRACE"]
    t = time.perf_counter()
    e = GgmlEngine(a.gguf, seqs=32, seq_ctx=2048, flash_attn=True, speculate=True)
    head = {"start_s": round(time.perf_counter() - t, 2), "pack": e.pack, "pk2": e.pk2,
            "ubatch": os.environ.get("IRONMULE_GGML_UBATCH"), "kg_tokens": (e.kg or {}).get("tokens_served")}
    print("ENGINE " + json.dumps(head, default=str), flush=True)
    out = open(a.out, "w")
    out.write(json.dumps({"engine": head}, default=str) + "\n")
    for name in ARMS:  # a cold wave per arm first (first graphs, buffers), not counted
        set_arm(name)
        e._clear()
        wave(e, prompts(e, 0, a.width), a.limit)
    names = list(ARMS)
    rec_by = {}
    for rnd in range(a.rounds):
        order = names[rnd % len(names):] + names[:rnd % len(names)]
        rep = 1 + rnd % 2  # the benchmark's repeats 1 and 2, each primed by the repeat before it
        for name in order:
            set_arm("R")
            e._clear()
            wave(e, prompts(e, rep - 1, a.width), 2)  # primer
            set_arm(name)
            t0, t1, jobs = wave(e, prompts(e, rep, a.width), a.limit)
            calls, notes = trace_since(trace, t0, t1)
            pre = [c for c in calls if c["prompt_rows"]]
            first = min(j.first_ns for j in jobs) / 1e9 - t0
            last_first = max(j.first_ns for j in jobs) / 1e9 - t0
            rec = {"round": rnd, "rep": rep, "arm": name, "wall_s": round(t1 - t0, 4),
                   "prefill_ms": round(sum(c["ms"] for c in pre), 2), "prefill_calls": pre,
                   "prompt_rows": sum(c["prompt_rows"] for c in pre), "reused": sum(j.reused for j in jobs),
                   "notes": notes, "first_token_s": [round(first, 4), round(last_first, 4)],
                   "tokens": [j.tokens for j in jobs]}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            rec_by[(rnd, name)] = rec
            print("AB " + json.dumps({k: v for k, v in rec.items() if k not in ("tokens",)}), flush=True)
    summary = {}
    for name in names:
        w = [rec_by[(r, name)]["wall_s"] for r in range(a.rounds)]
        summary[name] = {"wall_median_s": round(st.median(w), 4),
                         "prefill_median_ms": st.median(rec_by[(r, name)]["prefill_ms"] for r in range(a.rounds)),
                         "prompt_rows_median": st.median(rec_by[(r, name)]["prompt_rows"] for r in range(a.rounds)),
                         "reused_median": st.median(rec_by[(r, name)]["reused"] for r in range(a.rounds))}
    d = [rec_by[(r, "S")]["wall_s"] - rec_by[(r, "R")]["wall_s"] for r in range(a.rounds)]
    same = sum(rec_by[(r, "S")]["tokens"][k] == rec_by[(r, "R")]["tokens"][k] for r in range(a.rounds)
               for k in range(a.width))
    summary["S_vs_R"] = {"paired_ms_median": round(1e3 * st.median(d), 1), "faster": sum(x < 0 for x in d),
                         "rounds": a.rounds, "identical_answers": f"{same}/{a.rounds * a.width}"}
    out.write(json.dumps({"summary": summary}) + "\n")
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
