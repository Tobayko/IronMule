"""Kernel-level time breakdown of IronMule's GGML engine on a CUDA GPU, via CUPTI activity tracing (torch.profiler).

GPU performance counters may be locked in containers (ncu: ERR_NVGPUCTRPERM); activity tracing still records every
kernel the process launches, llama.cpp's included. Profiles one packed prompt burst and a run of decode steps.

  python lab/pk1/profile_engine.py --gguf MODEL.gguf --prompts 16 --prompt-len 444 --decode 8 --width 32
"""
from __future__ import annotations

import argparse
import time

import torch
from torch.profiler import ProfilerActivity, profile

from ironmule.ggml_engine import GgmlEngine


def table(prof, title, n=40):
    rows = []
    for e in prof.key_averages():
        t = getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
        if t > 0 and e.key and not e.key.startswith("ProfilerStep"):
            rows.append((t, e.count, e.key))
    rows.sort(reverse=True)
    total = sum(r[0] for r in rows)
    print(f"== {title}: {total / 1000:.1f} ms of kernels")
    for t, cnt, key in rows[:n]:
        print(f"{t / 1000:9.2f} ms {100 * t / total:5.1f}% x{cnt:<6d} {key[:110]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--prompts", type=int, default=16)
    p.add_argument("--prompt-len", type=int, default=444)
    p.add_argument("--decode", type=int, default=8)
    p.add_argument("--width", type=int, default=32)
    a = p.parse_args()
    torch.zeros(1, device="cuda")  # initialise CUPTI on the device
    e = GgmlEngine(a.gguf, seqs=a.width, seq_ctx=2048, flash_attn=True, speculate=False)
    bos = 1
    ids = e.tokenize(" ".join(f"Site {1000 + j} measured {20 + j % 60} units." for j in range(800)))
    group = [(s, None, 0) for s in range(a.prompts)]
    rows = [(ids[(s * a.prompt_len + j) % len(ids)], j, s, j == a.prompt_len - 1)
            for s in range(a.prompts) for j in range(a.prompt_len)]
    # warm-up
    e._clear()
    e._decode(rows if len(rows) <= e.batch_cap else rows[:e.batch_cap])
    e._clear()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        t = time.perf_counter()
        for start in range(0, len(rows), e.batch_cap):
            e._decode(rows[start:start + e.batch_cap])
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t
    print(f"prefill {len(rows)} rows in {prefill_s:.3f} s = {len(rows) / prefill_s:.0f} tok/s")
    table(prof, "prefill")
    pos = a.prompt_len
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        t = time.perf_counter()
        for k in range(a.decode):
            e._decode([(bos + 7, pos + k, s, True) for s in range(a.prompts)])
        torch.cuda.synchronize()
        dec_s = time.perf_counter() - t
    print(f"decode {a.decode} steps x {a.prompts} rows: {1000 * dec_s / a.decode:.2f} ms per step")
    table(prof, "decode")
    e.close()


if __name__ == "__main__":
    main()
