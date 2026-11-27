"""DK1: one W4R variant on one Qwen3-32B shape, for Nsight Compute (2026-11-24). Random blocks of the real format,
eight weight copies in turn (larger than the A100's 40 MB L2, so weights come from DRAM as in a served step), x
prepared by the library's prep_x just before the kernel (in L2, as in a served step).

  ncu --kernel-name regex:w4[df] --launch-skip 4 --launch-count 1 ... python lab/dk1/ncu_kernel.py LIB KIND K N B VARIANT KSPLIT
"""
import ctypes
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "w4bw"))
import bench  # noqa: E402

from ironmule import w4_kernels as wk  # noqa: E402


def main():
    lib_path, kind, K, N, B, vid, ks = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:8])
    lib = wk._lib(Path(lib_path))
    rng = np.random.default_rng(0)
    raw = bench.random_blocks(kind, N, K, rng)
    x = rng.standard_normal((B, K)).astype(np.float32)
    y = np.zeros((B, N), dtype=np.float32)
    us = ctypes.c_double()
    rc = lib.im_w4_bench(vid, raw.ctypes.data, raw.nbytes, N, K, x.ctypes.data, B, ks, 8, 3, ctypes.byref(us), y.ctypes.data)
    print(f"rc={rc} us={us.value:.2f} (with the profiler attached)", flush=True)


if __name__ == "__main__":
    main()
