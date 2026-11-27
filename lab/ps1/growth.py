"""PS1 diagnostics (2026-11-25): packed prefills whose merged row count first rises, then repeats, in one engine process,
with the TL1/TL2 timestamps (IRONMULE_TL1, IRONMULE_TL2) and optionally CUPTI (torch.profiler) for chosen waves.

A wave is N of the benchmark's prompts (a fresh seed each, the KV cleared first, so its rows depend on N only) with a
one-token answer: one packed prefill call. The default sequence 8, 16, 16, 24, 24, 32, 32, 24, 32 makes every rise a
new maximum of the merged rows and every repeat a known shape. Per wave: rows, the prefill call (ms), and the TL1 /
TL2 lines inside it. --profile 0-based wave indices run under CUPTI (kernels, busy time, gaps, first-time API calls).

  IRONMULE_PS1=1 IRONMULE_TL1=tl1.txt IRONMULE_TL2=tl2.txt python lab/ps1/growth.py MODEL.gguf OUT.jsonl [--profile 3]
"""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("IRONMULE_GGML_TRACE", tempfile.mktemp(suffix=".trace"))

from ironmule.ggml_engine import GgmlEngine, Job  # noqa: E402
from ironmule_product import compare  # noqa: E402

TPL = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def lines_between(path, t0_us, t1_us, kind):
    out = []
    if not path or not os.path.exists(path):
        return out
    for line in open(path):
        p = line.split()
        if p and p[0] == kind and t0_us <= int(p[1]) <= t1_us:
            out.append(line.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("out")
    ap.add_argument("--counts", default="8,16,16,24,24,32,32,24,32")
    ap.add_argument("--profile", default="")
    a = ap.parse_args()
    prof_waves = {int(x) for x in a.profile.split(",") if x}
    if prof_waves:
        import torch
        torch.zeros(1, device="cuda")
    t = time.perf_counter()
    e = GgmlEngine(a.gguf, seqs=32, seq_ctx=2048, flash_attn=True, speculate=True)
    head = {"start_s": round(time.perf_counter() - t, 2), "ps1": os.environ.get("IRONMULE_PS1"),
            "warm": os.environ.get("IRONMULE_PS1_WARM"), "module_loading": os.environ.get("CUDA_MODULE_LOADING"),
            "pk2": e.pk2.get("min_rows") if e.pk2 else None}
    out = open(a.out, "w")
    out.write(json.dumps({"engine": head}) + "\n")
    print("ENGINE " + json.dumps(head), flush=True)
    trace = os.environ["IRONMULE_GGML_TRACE"]
    tl1, tl2 = os.environ.get("IRONMULE_TL1"), os.environ.get("IRONMULE_TL2")
    for w, n in enumerate(int(x) for x in a.counts.split(",")):
        prompts = compare.performance_prompts(700 + w, n, max(n, 8))[:n]
        jobs = [Job(e.tokenize(TPL.format(p)), 1) for p in prompts]
        e._clear()
        e._kept.clear()
        e._free = list(range(e.seqs))
        t0 = time.monotonic()
        t0_us = int(time.monotonic() * 1e6)
        if w in prof_waves:
            import torch
            from torch.profiler import ProfilerActivity, profile, record_function
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                with record_function(f"wave{w}"):
                    e.run(jobs)
                torch.cuda.synchronize()
            path = Path(a.out).with_suffix(f".wave{w}.trace.json")
            prof.export_chrome_trace(str(path))
            gpu = analyse(path, f"wave{w}")
            path.unlink()
        else:
            e.run(jobs)
            gpu = None
        t1 = time.monotonic()
        t1_us = int(t1 * 1e6)
        calls = []
        for line in open(trace):
            if line.strip() and not line.startswith("#"):
                p = line.split()
                if t0 <= float(p[0]) <= t1 and int(p[5]) > 0:
                    calls.append({"ms": round(1e3 * float(p[1]), 2), "rows": int(p[2])})
        rec = {"wave": w, "requests": n, "wall_s": round(t1 - t0, 4), "prefill": calls,
               "tl1": [x for x in lines_between(tl1, t0_us, t1_us, "D") if int(x.split()[4]) >= 1][-6:], "tl2": lines_between(tl2, t0_us, t1_us, "A"),
               **({"gpu": gpu} if gpu else {})}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print("WAVE " + json.dumps({k: v for k, v in rec.items() if k != "gpu"}), flush=True)
        if gpu:
            print("GPU " + json.dumps(gpu), flush=True)


def analyse(path, label):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pf2"))
    import prefill_gpu  # PF2's CUPTI summary (imports torch)
    return prefill_gpu.analyse(path, label)


if __name__ == "__main__":
    main()
