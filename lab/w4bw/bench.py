"""W4R bandwidth bench: every variant of native/kernels/w4mma.cu on Qwen3-32B Q4_K_M's matrix shapes (random blocks
of the real formats), 1 / 8 / 32 tokens; the best per shape in GB/s and in % of the GPU's achievable read bandwidth
(lab/w4bw/bwprobe.cu), checked against a float64 decoding; then a whole decode step's matmuls, summed by count.

  python lab/w4bw/bench.py OUT_DIR [--tokens 1,8,32] [--source native/kernels/w4mma.cu]
"""
import argparse
import ctypes
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from ironmule import w4_kernels as wk
from ironmule.kernel_gen import compute_capability, nvcc_path

ROOT = Path(__file__).resolve().parents[2]
# (type, K, N, matrices per decode step) of Qwen3-32B Q4_K_M: 64 layers; attn_v and ffn_down half Q4_K, half Q6_K
SHAPES = [("Q4_K", 5120, 8192, 64), ("Q4_K", 5120, 1024, 64 + 32), ("Q4_K", 8192, 5120, 64),
          ("Q4_K", 5120, 25600, 128), ("Q4_K", 25600, 5120, 32), ("Q6_K", 25600, 5120, 32),
          ("Q6_K", 5120, 1024, 32), ("Q6_K", 5120, 151936, 1)]
BLOCK = {"Q4_K": 144, "Q6_K": 210}


def random_blocks(kind, n, k, rng):
    nb = k // 256
    raw = rng.integers(0, 256, size=(n, nb, BLOCK[kind]), dtype=np.uint8)
    d = (rng.random((n, nb)) * 0.01 + 0.001).astype(np.float16).view(np.uint8).reshape(n, nb, 2)
    if kind == "Q4_K":
        raw[..., 0:2] = d
        raw[..., 2:4] = (rng.random((n, nb)) * 0.01).astype(np.float16).view(np.uint8).reshape(n, nb, 2)
    else:
        raw[..., 208:210] = d
    return np.ascontiguousarray(raw.reshape(-1))


def build(out: Path, arch: str, source: Path) -> Path:
    nvcc = nvcc_path()
    lib = out / f"w4_{source.stem}_sm{arch}.so"
    cmd = [nvcc, "-O3", "-std=c++17", f"-arch=sm_{arch}", "-shared", "-Xcompiler", "-fPIC", "--cudart", "shared",
           "--expt-relaxed-constexpr", "-I", str(source.parent), "-o", str(lib), str(source)]
    subprocess.run(cmd, check=True)
    probe = out / f"bwprobe_sm{arch}.so"
    subprocess.run([nvcc, "-O3", f"-arch=sm_{arch}", "-shared", "-Xcompiler", "-fPIC", "--cudart", "shared",
                    "-o", str(probe), str(ROOT / "lab/w4bw/bwprobe.cu")], check=True)
    return lib


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--tokens", default="1,8,32")
    p.add_argument("--source", type=Path, default=ROOT / "native/kernels/w4mma.cu")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--only", default="", help="comma list of variant ids to measure (default: all)")
    p.add_argument("--shapes", default="", help="kind:K:N list to measure only these (e.g. Q4_K:5120:25600)")
    p.add_argument("--l2", action="store_true", help="one copy of the matrices that fit the L2 cache: the kernel's "
                   "compute-side ceiling (what limits it where DRAM is fast, as on an A100 at 32 tokens)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    arch = compute_capability()
    t0 = time.time()
    lib_path = build(a.out, arch, a.source)
    probe = ctypes.CDLL(str(a.out / f"bwprobe_sm{arch}.so"))
    probe.bw_probe.argtypes, probe.bw_probe.restype = [ctypes.c_size_t, ctypes.c_int], ctypes.c_double
    peak = probe.bw_probe(1 << 30, 6)
    print(json.dumps({"gpu_sm": arch, "achievable_read_gb_s": round(peak, 1), "compile_s": round(time.time() - t0)}),
          flush=True)
    lib = wk._lib(lib_path)
    vs = wk.variants(lib)
    only = {int(v) for v in a.only.split(",") if v}
    rng = np.random.default_rng(0)
    tokens = [int(t) for t in a.tokens.split(",")]
    best_all = []
    shapes = [s for s in SHAPES if not a.l2 or s[1] * s[2] // 256 * BLOCK[s[0]] < 30e6]
    if a.shapes:  # e.g. the full 5120 x 25600 gate/up matrix, which --l2 leaves out
        keep = {tuple(x.split(":")) for x in a.shapes.split(",")}
        shapes = [s for s in SHAPES if (s[0], str(s[1]), str(s[2])) in keep]
    for kind, K, N, count in shapes:
        code = wk.TYPE_Q4_K if kind == "Q4_K" else wk.TYPE_Q6_K
        raw = random_blocks(kind, N, K, rng)
        wbytes = raw.nbytes
        copies = 1 if a.l2 else max(2, min(8, int(2e9 // wbytes)))
        ref_rows = min(N, 2048)
        wref = wk.dequant(raw[:ref_rows * (K // 256) * BLOCK[kind]], kind, ref_rows, K)
        x = rng.standard_normal((max(tokens), K)).astype(np.float32)
        for B in tokens:
            ref = x[:B].astype(np.float64) @ wref.T
            y = np.zeros((B, N), dtype=np.float32)
            rows, outs = [], {}
            for v in vs:
                if v["type"] != code or v["nt"] != -(-B // 8) or (only and v["id"] not in only):
                    continue
                for ks in wk.KSPLITS:
                    if ks > 1 and (K // 256) < 2 * ks:
                        continue
                    us = ctypes.c_double()
                    rc = lib.im_w4_bench(v["id"], raw.ctypes.data, wbytes, N, K, x.ctypes.data, B, ks, copies, a.reps,
                                         ctypes.byref(us), y.ctypes.data)
                    if rc:
                        continue
                    got = y[:, :ref_rows].astype(np.float64)
                    err = float(np.linalg.norm(got - ref) / max(1e-30, np.linalg.norm(ref)))
                    if ks == 1 and v["ks"] % 10000 in range(0, 10000) and (v["ks"] // 10000) in (3, 8):
                        outs[(v["ks"] // 10000, v["ks"] % 10000, v["warps"], v["nt"])] = y.copy()  # v9 / PS, ksplit 1
                    if err <= wk.ERROR_BOUND:
                        rows.append((us.value, v["id"], ks, err, v))
            rows.sort(key=lambda r: r[0])
            plain = [r for r in rows if not r[4].get("pre")]
            pre = [r for r in rows if r[4].get("pre")]
            if not rows:
                print(json.dumps({"kind": kind, "K": K, "N": N, "B": B, "best": None}), flush=True)
                continue
            # PS (v12, code 8xxxx) must equal v9 (code 3xxxx) of the same configuration bit for bit
            pairs = [(k, (3,) + k[1:]) for k in outs if k[0] == 8 and (3,) + k[1:] in outs]
            identical = all(np.array_equal(outs[a], outs[b]) for a, b in pairs) if pairs else None
            us, vid, ks, err, v = rows[0]
            gbs = wbytes / us / 1e3
            rec = {"l2": a.l2, "kind": kind, "K": K, "N": N, "B": B, "count": count, "us": round(us, 2), "gb_s": round(gbs, 1),
                   "pct_peak": round(100 * gbs / peak, 1), "variant": vid, "ksplit": ks, "rel_err": round(err, 6),
                   "cfg": {k: v[k] for k in ("warps", "ks", "nt", "pre")},
                   # PS (v12) against the best without it, on the same matrix, x and copies
                   "best_plain_us": round(plain[0][0], 2) if plain else None,
                   "best_ps_us": round(pre[0][0], 2) if pre else None,
                   "ps_bit_identical_to_v9": identical, "ps_pairs": len(pairs),
                   "next": [(round(r[0], 2), r[1], r[2]) for r in rows[1:4]]}
            best_all.append(rec)
            print(json.dumps(rec), flush=True)
    for B in tokens:
        recs = [r for r in best_all if r["B"] == B]
        ms = sum(r["us"] * r["count"] for r in recs) / 1e3
        gb = sum(r["us"] * r["gb_s"] * r["count"] for r in recs) / 1e6
        print(json.dumps({"step_B": B, "matmul_ms": round(ms, 3), "gb": round(gb, 3),
                          "effective_gb_s": round(gb / ms * 1e3, 1), "pct_peak": round(100 * gb / ms * 1e3 / peak, 1)}),
              flush=True)
    (a.out / "bench.json").write_text(json.dumps({"peak": peak, "best": best_all}))


if __name__ == "__main__":
    main()
