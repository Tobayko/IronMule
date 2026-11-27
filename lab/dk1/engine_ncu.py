"""DK1: one served 32-row decode wave under Nsight Compute (2026-11-24). The engine as engine_probe.py builds it (W4R
library and table from IRONMULE_KG_LIB / IRONMULE_W4_TABLE, CUDA graphs from the environment), a first 32-request wave
of the HTTP benchmark's prompts to warm up (graphs captured), then cudaProfilerStart around a second wave with other
prompts: run ncu with --profile-from-start off and a kernel filter, e.g.

  ncu --profile-from-start off --kernel-name regex:w4[df] --launch-skip 449 --launch-count 449 ... \
      python lab/dk1/engine_ncu.py MODEL.gguf

Without ncu it prints the second wave's decode calls (engine trace) as a check.
"""
import ctypes
import os
import sys
import tempfile
import time

os.environ.setdefault("IRONMULE_GGML_TRACE", tempfile.mktemp(suffix=".trace"))

from ironmule.ggml_engine import GgmlEngine, Job  # noqa: E402
from ironmule_product import compare  # noqa: E402

TPL = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def main():
    e = GgmlEngine(sys.argv[1], seqs=32, seq_ctx=2048, flash_attn=True, speculate=False)
    try:
        rt = ctypes.CDLL("libcudart.so.12")
        start, stop, sync = rt.cudaProfilerStart, rt.cudaProfilerStop, rt.cudaDeviceSynchronize
    except OSError:  # the driver API is always there
        rt = ctypes.CDLL("libcuda.so.1")
        start, stop, sync = rt.cuProfilerStart, rt.cuProfilerStop, rt.cuCtxSynchronize
    for seed, profile in ((0, False), (1, True)):
        jobs = [Job(e.tokenize(TPL.format(p)), 32) for p in compare.performance_prompts(seed, 32, 32)]
        if profile:
            sync()
            print("profiler start rc", start(), flush=True)
        t0 = time.monotonic()
        e.run(jobs)
        if profile:
            sync()
            print("profiler stop rc", stop(), flush=True)
        print(f"wave seed {seed}: {time.monotonic() - t0:.3f} s", flush=True)


if __name__ == "__main__":
    main()
