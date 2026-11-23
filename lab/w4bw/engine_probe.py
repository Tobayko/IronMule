"""Engine probe for the 2026-11-24 experiments: waves of the HTTP benchmark's prompts (compare.performance_prompts, the
chat template, 32-token answers) straight on the engine, at the given widths (e.g. 8,20,24,32: 20 and 24 load the
17-24-row steps the W4R table had no entry for). From the engine trace (IRONMULE_GGML_TRACE): the median decode call per
row count, every prefill call (rows, prompts, seconds), and per wave the time to the first token (median and slowest
request) and the wall time; the answers' tokens go to OUT for comparison between arms.

  python lab/w4bw/engine_probe.py MODEL.gguf OUT.json --widths 8,20,24,32 --seeds 0,1,2 [--label NAME]

The W4R library and table come from the environment (IRONMULE_KG_LIB, IRONMULE_W4_TABLE), the prompt-matmul threshold
from IRONMULE_CUBLAS_MIN_ROWS or the engine's own calibration (IRONMULE_PK2_CALIBRATION=equal|mix).
"""
import argparse
import json
import os
import statistics as st
import tempfile
import time

trace = tempfile.mktemp(suffix=".trace")
os.environ.setdefault("IRONMULE_GGML_TRACE", trace)
trace = os.environ["IRONMULE_GGML_TRACE"]

from ironmule.ggml_engine import GgmlEngine, Job  # noqa: E402
from ironmule_product import compare  # noqa: E402

TPL = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def calls_between(t0, t1):
    out = []
    for line in open(trace):
        if line.strip() and not line.startswith("#"):
            p = line.split()
            start, sec, rows = float(p[0]), float(p[1]), int(float(p[2]))
            if t0 <= start <= t1:
                out.append((start, sec, rows, int(p[3]) if len(p) > 3 else 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("out")
    ap.add_argument("--widths", default="8,32")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    e = GgmlEngine(a.gguf, seqs=32, seq_ctx=2048, flash_attn=True, speculate=False)
    meta = {"label": a.label, "pk2": getattr(e, "pk2", {}), "kg": getattr(e, "kg", None)}
    print("PROBE " + json.dumps(meta, default=str)[:600], flush=True)
    tokens, waves, steps = {}, [], {}
    for width in [int(w) for w in a.widths.split(",")]:
        for seed in [int(s) for s in a.seeds.split(",")]:
            prompts = compare.performance_prompts(seed, width, max(width, 8))[:width]
            jobs = [Job(e.tokenize(TPL.format(p)), 32) for p in prompts]
            t0 = time.monotonic()
            started_ns = time.monotonic_ns()
            e.run(jobs)
            t1 = time.monotonic()
            calls = calls_between(t0, t1)
            firsts = sorted((j.first_ns - started_ns) / 1e6 for j in jobs if j.first_ns)
            pre = [(c[2], c[3], round(c[1], 4)) for c in calls if c[2] > 64]
            for c in calls:
                if c[2] <= 64 and c[2] == c[3]:  # a decode step: one row per running sequence, no prompt rows
                    steps.setdefault(c[2], []).append(c[1])
            waves.append({"width": width, "seed": seed, "wall_s": round(t1 - t0, 4),
                          "ttft_p50_ms": round(firsts[len(firsts) // 2], 1) if firsts else None,
                          "ttft_max_ms": round(firsts[-1], 1) if firsts else None,
                          "tok_s": round(sum(len(j.tokens) for j in jobs) / (t1 - t0), 1), "prefill": pre})
            tokens[f"{width}-{seed}"] = [j.tokens for j in jobs]
            print("WAVE " + json.dumps(waves[-1]), flush=True)
    med = {str(r): round(1e3 * st.median(v), 3) for r, v in sorted(steps.items()) if len(v) >= 3}
    print("STEPS " + json.dumps({"label": a.label, "decode_ms_by_rows": med}), flush=True)
    json.dump({"meta": meta, "waves": waves, "decode_ms_by_rows": med, "tokens": tokens}, open(a.out, "w"),
              default=str)
    e.close()


if __name__ == "__main__":
    main()
