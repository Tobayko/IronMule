"""DK1: several builds of the W4R library in one process, measured in turn (2026-11-24). GPU counters are refused in
RunPod containers, so v9's limits are located by timing; separate processes per build gave scattered numbers (a short
timing window right after start, an undefined L1 / shared-memory split). Here every build is loaded at once, the GPU is
kept busy for over a second first, and the arms alternate for several rounds of many calls each.

  python lab/dk1/abl_bench.py --variant 184 --ksplit 1 NAME=LIB.so NAME=LIB.so ...
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
    ap.add_argument("libs", nargs="+")
    ap.add_argument("--variant", type=int, default=184)
    ap.add_argument("--cfg", default="", help="warps:ks:nt instead of --variant, e.g. 2:30201:2 (v9, 2 warps, 2 tiles)")
    ap.add_argument("--ksplit", type=int, default=1)
    ap.add_argument("--shape", default="Q4_K:5120:25600")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=6)
    a = ap.parse_args()
    kind, K, N = a.shape.split(":")
    K, N, B = int(K), int(N), a.tokens
    rng = np.random.default_rng(0)
    raw = bench.random_blocks(kind, N, K, rng)
    x = rng.standard_normal((B, K)).astype(np.float32)
    y = np.zeros((B, N), dtype=np.float32)
    arms = {}
    for spec in a.libs:
        name, path = spec.split("=", 1)
        arms[name] = wk._lib(Path(path))
    if a.cfg:
        w, ks, nt = map(int, a.cfg.split(":"))
        a.variant = next(v["id"] for v in wk.variants(next(iter(arms.values())))
                         if (v["warps"], v["ks"], v["nt"]) == (w, ks, nt))

    def run(lib, reps):
        us = ctypes.c_double()
        rc = lib.im_w4_bench(a.variant, raw.ctypes.data, raw.nbytes, N, K, x.ctypes.data, B, a.ksplit, 8, reps,
                             ctypes.byref(us), y.ctypes.data)
        return us.value if rc == 0 else None
    for lib in arms.values():  # warm-up: about 0.2 s of kernels per build
        run(lib, 2000)
    times = {n: [] for n in arms}
    for r in range(a.rounds):
        order = list(arms) if r % 2 == 0 else list(arms)[::-1]
        for n in order:
            times[n].append(run(arms[n], a.reps))
    for n, ts in times.items():
        ok = [t for t in ts if t is not None]
        print(json.dumps({"arm": n, "shape": a.shape, "B": B, "variant": a.variant, "ksplit": a.ksplit,
                          "us_median": round(st.median(ok), 2) if ok else None, "us_min": round(min(ok), 2) if ok else None,
                          "us": [None if t is None else round(t, 2) for t in ts]}), flush=True)


if __name__ == "__main__":
    main()
