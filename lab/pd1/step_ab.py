"""PD1: the served decode step with and without producer-prepared activations, in one process (2026-11-24).

One engine as the server builds it (W4R library and table from IRONMULE_KG_LIB / IRONMULE_W4_TABLE, CUDA graphs from
the environment), the HTTP benchmark's prompts at 8, 16, 24 and 32 requests. The arms alternate inside the process:
PD1 is switched in the kernel library (ironmule_kg_pd1_set) and every CUDA graph is captured again
(ironmule_cuda_graphs_invalidate, patch pd1); after each switch one wave per width is run and discarded (it captures the
graphs), then the measured waves. From the engine trace: each wave's median decode call at the full width (every
sequence decoding, no prompt rows), the wave's wall time; the answers' tokens of each arm against the reference arm.
With IRONMULE_KG_PD1_CHECK=1 the library also compares every producer-written fp16 copy with prep_x's.

  python lab/pd1/step_ab.py MODEL.gguf OUT.json [--widths 8,16,24,32] [--seeds 3,4] [--rounds 3]
"""
import argparse
import ctypes
import json
import os
import statistics as st
import tempfile
import time
from pathlib import Path

os.environ.setdefault("IRONMULE_GGML_TRACE", tempfile.mktemp(suffix=".trace"))
TRACE = os.environ["IRONMULE_GGML_TRACE"]

from ironmule.ggml_engine import GgmlEngine, Job  # noqa: E402
from ironmule_product import compare  # noqa: E402

TPL = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def decode_calls(t0, t1, width):
    out = []
    for line in open(TRACE):
        if line.strip() and not line.startswith("#"):
            p = line.split()
            start = float(p[0])
            if t0 <= start <= t1 and int(p[2]) == width and int(p[3]) == width and len(p) >= 7 and int(p[5]) == 0:
                out.append(1e3 * float(p[1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("out")
    ap.add_argument("--widths", default="8,16,24,32")
    ap.add_argument("--seeds", default="3,4")
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    kg = ctypes.CDLL(os.environ["IRONMULE_KG_LIB"])
    kg.ironmule_kg_pd1_set.argtypes = [ctypes.c_int]
    folder = Path(os.environ["LLAMA_CPP_LIB_PATH"])
    e = GgmlEngine(a.gguf, seqs=32, seq_ctx=2048, flash_attn=True, speculate=False)
    cuda = ctypes.CDLL(str(next(folder.glob("libggml-cuda.so*"))))
    widths = [int(w) for w in a.widths.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    prompts = {(w, s): compare.performance_prompts(s, w, max(w, 8))[:w] for w in widths for s in seeds}
    rec = {"widths": widths, "seeds": seeds, "rounds": []}
    tokens = {}
    for r in range(a.rounds):
        arms = ("ref", "pd1") if r % 2 == 0 else ("pd1", "ref")
        for arm in arms:
            kg.ironmule_kg_pd1_set(1 if arm == "pd1" else 0)
            cuda.ironmule_cuda_graphs_invalidate()
            for w in widths:  # one discarded wave per width: graphs captured anew with this arm's choice
                e.run([Job(e.tokenize(TPL.format(p)), 32) for p in compare.performance_prompts(99, w, max(w, 8))[:w]])
            for w in widths:
                for s in seeds:
                    jobs = [Job(e.tokenize(TPL.format(p)), 32) for p in prompts[(w, s)]]
                    t0 = time.monotonic()
                    e.run(jobs)
                    t1 = time.monotonic()
                    calls = decode_calls(t0, t1, w)
                    row = {"round": r, "arm": arm, "width": w, "seed": s, "wall_s": round(t1 - t0, 4),
                           "decode_calls": len(calls), "decode_ms_median": round(st.median(calls), 3) if calls else None}
                    toks = [list(j.tokens) for j in jobs]
                    key = f"{w}:{s}"
                    ref = tokens.setdefault(("ref", key), toks) if arm == "ref" else tokens.get(("ref", key))
                    if arm == "pd1":
                        tokens.setdefault(("pd1", key), toks)
                    if ref is not None:
                        row["tokens_equal_to_ref"] = sum(x == y for a_, b_ in zip(toks, ref) for x, y in zip(a_, b_))
                        row["tokens"] = sum(len(t) for t in ref)
                    rec["rounds"].append(row)
                    print("STEP " + json.dumps(row), flush=True)
    counts = (ctypes.c_ulonglong * 2)()
    kg.ironmule_kg_pd1_counts(counts)
    rec["pd1_check"] = {"compared_halves": counts[0], "differing_halves": counts[1],
                        "enabled": os.environ.get("IRONMULE_KG_PD1_CHECK", "0")}
    print("CHECK " + json.dumps(rec["pd1_check"]), flush=True)
    summary = {}
    for w in widths:
        for arm in ("ref", "pd1"):
            ms = [x["decode_ms_median"] for x in rec["rounds"] if x["arm"] == arm and x["width"] == w and x["decode_ms_median"]]
            summary[f"{arm}_{w}"] = {"per_wave": ms, "median": round(st.median(ms), 3) if ms else None}
    rec["summary"] = summary
    print("SUMMARY " + json.dumps(summary), flush=True)
    Path(a.out).write_text(json.dumps(rec))


if __name__ == "__main__":
    main()
