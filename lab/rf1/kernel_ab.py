"""RF1: a grid K-split W4R matmul with its partial sums reduced by reduce_split (reference) or inside the kernel by the
last block of each row range (RF1), in one process (2026-11-24). Each call is prep_x + matmul + the whole reduction
(im_w4_bench), eight weight copies in turn (DRAM, as served), alternating rounds after a warm-up; the outputs of the two
paths compared bit for bit.

  python lab/rf1/kernel_ab.py LIB.so --cases Q4_K:25600:5120:32:256:4,Q6_K:25600:5120:32:266:4 [--rounds 6]
"""
import argparse
import ctypes
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "w4bw"))
import bench  # noqa: E402

from ironmule import w4_kernels as wk  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lib")
    ap.add_argument("--cases", required=True, help="kind:K:N:B:variant:ksplit,...")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    lib = wk._lib(Path(a.lib))
    lib.ironmule_kg_rf1_set.argtypes = [ctypes.c_int]
    rng = np.random.default_rng(0)
    for case in a.cases.split(","):
        kind, K, N, B, vid, ks = case.split(":")
        K, N, B, vid, ks = int(K), int(N), int(B), int(vid), int(ks)
        raw = bench.random_blocks(kind, N, K, rng)
        x = rng.standard_normal((B, K)).astype(np.float32)
        ys = {}

        def run(arm, reps):
            lib.ironmule_kg_rf1_set(1 if arm == "rf1" else 0)
            y = np.zeros((B, N), dtype=np.float32)
            us = ctypes.c_double()
            rc = lib.im_w4_bench(vid, raw.ctypes.data, raw.nbytes, N, K, x.ctypes.data, B, ks, 8, reps, ctypes.byref(us),
                                 y.ctypes.data)
            ys[arm] = y
            return us.value if rc == 0 else None
        run("ref", 2000)
        run("rf1", 2000)
        times = {"ref": [], "rf1": []}
        for r in range(a.rounds):
            for arm in (("ref", "rf1") if r % 2 == 0 else ("rf1", "ref")):
                times[arm].append(run(arm, a.reps))
        same = bool(np.array_equal(ys["ref"].view(np.uint32), ys["rf1"].view(np.uint32)))
        out = {"case": case, "bit_identical": same, "max_abs_diff": float(np.max(np.abs(ys["ref"] - ys["rf1"])))}
        for arm, ts in times.items():
            ok = [t for t in ts if t is not None]
            out[arm] = {"us_median": round(st.median(ok), 2) if ok else None, "us": [round(t, 2) for t in ok]}
        print("KERNEL " + json.dumps(out), flush=True)
    lib.ironmule_kg_rf1_set(-1)


if __name__ == "__main__":
    main()
