"""PF2: what the GPU does in a packed 32-request prefill, first wave after the engine's start against later ones
(2026-11-24). Diagnostic, apart from the timed HTTP runs: CUPTI activity tracing through torch.profiler (GPU counters are
refused in RunPod containers; activity tracing works).

The engine as the server builds it (flash attention, one KV stream per sequence, 32 sequences, speculation on; the W4R
library, table and CUDA graphs from the environment). Each wave is the HTTP benchmark's 32 prompts of a fresh seed with
a one-token answer, so a wave is the gathered prefill call(s) and the first token only. Waves 0 (the first after the
start) and 3 run under the profiler, the others without it (their prefill calls are in the engine trace). For a profiled
wave: kernel time by group (GEMM, MMQ, weight dequantisation, activation conversion, attention, W4R, the rest), the GPU's
busy time and gaps inside the wave, host time before the first and after the last kernel, and the CUDA API calls that
mark first-time work (allocations, module loads, graph or attribute setup).

  python lab/pf2/prefill_gpu.py MODEL.gguf OUT_DIR [--waves 5]
"""
import argparse
import json
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("IRONMULE_GGML_TRACE", tempfile.mktemp(suffix=".trace"))

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function  # noqa: E402

from ironmule.ggml_engine import GgmlEngine, Job  # noqa: E402
from ironmule_product import compare  # noqa: E402

TPL = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
GROUPS = [("w4r", r"w4[a-z0-9]*<|prep_x|reduce_split|pd1_compare"), ("attention", r"flash_attn|fattn|soft_max"),
          ("gemm", r"gemm|xmma|cutlass|s16816|s1688|Kernel2|cublas"), ("mmq", r"mul_mat_q|mul_mat_vec|mmq|mmvq"),
          ("weight_dequant", r"dequantize|ironmule_pk5|q4_K|q6_K|Q4_K|Q6_K"),
          ("act_convert", r"convert_unary|cpy_f32_f16|to_fp16|cpy_flt|convert_"),
          ("norm", r"rms_norm"), ("rope", r"rope"), ("glu", r"swiglu|unary_gated|silu"), ("kv_write", r"set_rows"),
          ("elementwise", r"k_bin_bcast|add|mul")]
FIRST_TIME = re.compile(r"cudaMalloc|cudaFree|cuMemAlloc|cuMemFree|cuModule|cudaFuncSetAttribute|cudaGraphInstantiate|"
                        r"cudaGraphExecUpdate|cudaStreamCreate|cudaEventCreate|cublas|cuLibrary|cudaHostAlloc|cuMemCreate|"
                        r"cuMemMap")


def group(name):
    for g, pat in GROUPS:
        if re.search(pat, name):
            return g
    return "other"


def analyse(trace_json, label):
    ev = json.load(open(trace_json))["traceEvents"]
    region = [e for e in ev if e.get("cat") == "user_annotation" and e.get("name") == label]
    r0 = region[0]["ts"] if region else min(e["ts"] for e in ev if "ts" in e)
    r1 = r0 + (region[0]["dur"] if region else 0)
    kern = sorted([e for e in ev if e.get("cat") == "kernel" and "dur" in e], key=lambda e: e["ts"])
    groups, names = defaultdict(float), defaultdict(float)
    for e in kern:
        groups[group(e["name"])] += e["dur"]
        names[e["name"][:90]] += e["dur"]
    busy, gaps, end = 0.0, [], None
    for e in kern:
        s, f = e["ts"], e["ts"] + e["dur"]
        if end is None or s > end:
            if end is not None:
                gaps.append(s - end)
            busy += e["dur"]
            end = f
        elif f > end:
            busy += f - end
            end = f
    api = Counter()
    api_us = defaultdict(float)
    for e in ev:
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and FIRST_TIME.search(e.get("name", "")):
            api[e["name"]] += 1
            api_us[e["name"]] += e.get("dur", 0)
    slow_api = sorted(((e.get("dur", 0), e.get("name", ""), round((e["ts"] - r0) / 1e3, 1)) for e in ev
                       if e.get("cat") in ("cuda_runtime", "cuda_driver") and e.get("dur", 0) > 2000), reverse=True)[:12]
    out = {"label": label, "region_ms": round((r1 - r0) / 1e3, 2), "kernels": len(kern),
           "api_over_2ms": [[round(d / 1e3, 1), n, at] for d, n, at in slow_api],
           "kernel_ms": round(sum(e["dur"] for e in kern) / 1e3, 2), "busy_ms": round(busy / 1e3, 2),
           "gaps_ms": round(sum(gaps) / 1e3, 2), "gaps_over_50us": sum(1 for g in gaps if g > 50),
           "largest_gaps_ms": [round(g / 1e3, 2) for g in sorted(gaps)[-5:]],
           "host_before_first_kernel_ms": round((kern[0]["ts"] - r0) / 1e3, 2) if kern else None,
           "after_last_kernel_ms": round((r1 - (kern[-1]["ts"] + kern[-1]["dur"])) / 1e3, 2) if kern else None,
           "groups_ms": {g: round(v / 1e3, 2) for g, v in sorted(groups.items(), key=lambda x: -x[1])},
           "top_kernels_ms": {k: round(v / 1e3, 2) for k, v in sorted(names.items(), key=lambda x: -x[1])[:12]},
           "first_time_api": {k: [n, round(api_us[k] / 1e3, 2)] for k, n in api.most_common(12)}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("out", type=Path)
    ap.add_argument("--waves", type=int, default=5)
    ap.add_argument("--profile", default="0,3")
    ap.add_argument("--split", type=int, default=0, help="as the server's split bursts: the first N requests start the "
                    "wave alone, the others are admitted 40 ms later (after the gathering window closed)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.zeros(1, device="cuda")  # CUPTI on the device
    t = time.perf_counter()
    e = GgmlEngine(a.gguf, seqs=32, seq_ctx=2048, flash_attn=True, speculate=True)
    print("ENGINE " + json.dumps({"start_s": round(time.perf_counter() - t, 2), "pk2": e.pk2,
                                  "kg_tokens": (e.kg or {}).get("tokens_served")}, default=str), flush=True)
    prof_waves = {int(w) for w in a.profile.split(",")}
    trace = os.environ["IRONMULE_GGML_TRACE"]
    for w in range(a.waves):
        prompts = compare.performance_prompts(500 + w, 32, 32)
        jobs = [Job(e.tokenize(TPL.format(p)), 1) for p in prompts]
        label = f"wave{w}"
        t0 = time.monotonic()
        def go():
            if not a.split:
                return e.run(jobs)
            rest, began = list(jobs[a.split:]), time.monotonic()

            def admit(free):
                if rest and time.monotonic() - began > 0.04:
                    got = rest[:free]
                    del rest[:len(got)]
                    return got
                return []
            return e.run(jobs[:a.split], admit=admit)
        if w in prof_waves:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                with record_function(label):
                    go()
                torch.cuda.synchronize()
            path = a.out / f"{label}.trace.json"
            prof.export_chrome_trace(str(path))
            res = analyse(path, label)
            res["prompt_tokens"] = sum(len(j.ids) for j in jobs)
            print("GPU " + json.dumps(res), flush=True)
            path.unlink()  # large; the summary is kept
        else:
            go()
        t1 = time.monotonic()
        calls = []
        for line in open(trace):
            if line.strip() and not line.startswith("#"):
                p = line.split()
                if t0 <= float(p[0]) <= t1:
                    calls.append({"ms": round(1e3 * float(p[1]), 2), "rows": int(p[2]), "seqs": int(p[3]),
                                  **{kv[0]: float(kv[2:]) for kv in p[8:] if len(kv) > 2 and kv[1] == "="}})
        print("WAVE " + json.dumps({"wave": w, "profiled": w in prof_waves, "wall_s": round(t1 - t0, 4),
                                    "prompt_tokens": sum(len(j.ids) for j in jobs), "calls": calls}), flush=True)


if __name__ == "__main__":
    main()
