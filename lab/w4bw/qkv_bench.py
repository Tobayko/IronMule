"""Q/K/V in isolation (2026-11-24): a Qwen3-32B layer's three projections back to back as a served step runs them (the
best W4R variant of each shape, one prepared x shared as PK3 does), against v13 (w4dq), which computes the segments as
one virtual matrix in one launch. Two layer kinds: 32 layers with v in Q4_K (q + k + v fused) and 32 with v in Q6_K
(q + k fused, v separate). Per token count the best fused configuration, its error against a float64 decoding and its
largest difference from the separate path, and the Q/K/V time of one decode step with and without fusion.

  python lab/w4bw/qkv_bench.py OUT_DIR [--tokens 8,16,24,32]     # also writes OUT_DIR/qkv.table for QK1
"""
import argparse
import ctypes
import json
import time
from pathlib import Path

import numpy as np

from ironmule import w4_kernels as wk
from ironmule.kernel_gen import compute_capability

import bench  # lab/w4bw/bench.py: random blocks of the real formats, the build

K, NQ, NKV = 5120, 8192, 1024
LAYERS_V4, LAYERS_V6 = 32, 32  # Qwen3-32B Q4_K_M: attn_v is Q4_K in half the layers, Q6_K in the others


def best_single(lib, vs, code, raw, n, x, B, copies, reps=20):
    y = np.zeros((B, n), dtype=np.float32)
    rows = []
    for v in vs:
        if v["type"] != code or v["nt"] != -(-B // 8) or v.get("pre"):
            continue
        for ks in wk.KSPLITS:
            if ks > 1 and (K // 256) < 2 * ks:
                continue
            us = ctypes.c_double()
            rc = lib.im_w4_bench(v["id"], raw.ctypes.data, raw.nbytes, n, K, x.ctypes.data, B, ks, copies, reps,
                                 ctypes.byref(us), y.ctypes.data)
            if rc == 0:
                rows.append((us.value, v["id"], ks))
    rows.sort()
    return rows[0] if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--tokens", default="8,24,32")
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    arch = compute_capability()
    lib_path = bench.build(a.out, arch, bench.ROOT / "native/kernels/w4mma.cu")
    lib = wk._lib(lib_path)
    lib.im_w4_qkv_variant_count.restype = ctypes.c_int
    lib.im_w4_qkv_variant.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.im_w4_qkv_bench.restype = ctypes.c_int
    P = ctypes.c_void_p  # every pointer declared: ctypes passes an undeclared int as 32 bits
    lib.im_w4_qkv_bench.argtypes = [ctypes.c_int, P, P, P, ctypes.c_long, P, ctypes.c_long, P, P, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), P, P]
    vs = wk.variants(lib)
    fvs = []
    for i in range(lib.im_w4_qkv_variant_count()):
        o = (ctypes.c_int * 4)()
        lib.im_w4_qkv_variant(i, o)
        fvs.append({"id": i, "warps": o[0], "ks": o[1], "nt": o[2], "mt": o[3]})
    rng = np.random.default_rng(0)
    q, k, v4 = (bench.random_blocks("Q4_K", n, K, rng) for n in (NQ, NKV, NKV))
    v6 = bench.random_blocks("Q6_K", NKV, K, rng)
    ref = {name: wk.dequant(raw, "Q4_K", n, K) if name != "v6" else wk.dequant(raw, "Q6_K", n, K)
           for name, raw, n in (("q", q[: 2048 * (K // 256) * 144], 2048), ("k", k, NKV), ("v4", v4, NKV), ("v6", v6, NKV))}
    copies = 8
    results = []
    for B in [int(t) for t in a.tokens.split(",")]:
        x = rng.standard_normal((B, K)).astype(np.float32)
        t0 = time.time()
        sq = best_single(lib, vs, wk.TYPE_Q4_K, q, NQ, x, B, copies)
        skv = best_single(lib, vs, wk.TYPE_Q4_K, k, NKV, x, B, copies)
        sv6 = best_single(lib, vs, wk.TYPE_Q6_K, v6, NKV, x, B, copies)
        for kind, segs in (("qkv", [(q, NQ, sq, "q"), (k, NKV, skv, "k"), (v4, NKV, skv, "v4")]),
                           ("qk", [(q, NQ, sq, "q"), (k, NKV, skv, "k")])):
            n = len(segs)
            w_host = (ctypes.c_void_p * n)(*[s[0].ctypes.data for s in segs])
            wbytes = (ctypes.c_size_t * n)(*[s[0].nbytes for s in segs])
            Ns = (ctypes.c_long * n)(*[s[1] for s in segs])
            sep_v = (ctypes.c_int * n)(*[s[2][1] for s in segs])
            sep_k = (ctypes.c_int * n)(*[s[2][2] for s in segs])
            total = sum(s[1] for s in segs)
            y_sep = np.zeros(B * total, dtype=np.float32)
            y_fused = np.zeros(B * total, dtype=np.float32)
            best = None
            for fv in fvs:
                if fv["nt"] != -(-B // 8):
                    continue
                for ks in (1, 2, 4):
                    us_sep, us_f = ctypes.c_double(), ctypes.c_double()
                    rc = lib.im_w4_qkv_bench(n, ctypes.cast(w_host, P), ctypes.cast(wbytes, P), ctypes.cast(Ns, P), K,
                                             x.ctypes.data, B, ctypes.cast(sep_v, P), ctypes.cast(sep_k, P), fv["id"], ks,
                                             copies, a.reps, ctypes.byref(us_sep), ctypes.byref(us_f),
                                             y_sep.ctypes.data, y_fused.ctypes.data)
                    if rc:
                        continue
                    errs, off = {}, 0
                    for raw, nn, _, name in segs:
                        got = y_fused[off: off + B * nn].reshape(B, nn).astype(np.float64)
                        rr = ref[name].shape[0]
                        want = x.astype(np.float64) @ ref[name].T
                        errs[name] = float(np.linalg.norm(got[:, :rr] - want) / np.linalg.norm(want))
                        off += B * nn
                    diff = float(np.max(np.abs(y_fused - y_sep)))
                    ok = all(e <= wk.ERROR_BOUND for e in errs.values())
                    row = {"fv": fv, "ksplit": ks, "us_sep": round(us_sep.value, 2), "us_fused": round(us_f.value, 2),
                           "rel_err": {kk: round(e, 6) for kk, e in errs.items()}, "max_abs_diff_vs_sep": diff,
                           "bit_identical": diff == 0.0}
                    if ok and (best is None or row["us_fused"] < best["us_fused"]):
                        best = row
            rec = {"B": B, "layer": kind, "sep_variants": [(s[3], s[2][1], s[2][2], round(s[2][0], 2)) for s in segs],
                   "best_fused": best}
            results.append(rec)
            print(json.dumps(rec), flush=True)
        v6_us = sv6[0] if sv6 else None
        A = [r for r in results if r["B"] == B and r["layer"] == "qkv"][-1]["best_fused"]
        Bk = [r for r in results if r["B"] == B and r["layer"] == "qk"][-1]["best_fused"]
        if A and Bk and v6_us:
            sep_ms = (LAYERS_V4 * A["us_sep"] + LAYERS_V6 * (Bk["us_sep"] + v6_us)) / 1e3
            fused_ms = (LAYERS_V4 * A["us_fused"] + LAYERS_V6 * (Bk["us_fused"] + v6_us)) / 1e3
            step = {"B": B, "qkv_step_ms_separate": round(sep_ms, 3), "qkv_step_ms_fused": round(fused_ms, 3),
                    "saved_ms": round(sep_ms - fused_ms, 3), "v6_us": round(v6_us, 2), "s": round(time.time() - t0)}
            results.append(step)
            print("STEP " + json.dumps(step), flush=True)
    (a.out / "qkv.json").write_text(json.dumps(results))
    # QK1's table (IRONMULE_KG_QKV_TABLE): "nseg ntot K B variant ksplit" for every token count of a measured tile count
    # where the fused launch beat the separate ones; never 1 token (the one-row kernels stay)
    lines = []
    for r in results:
        b = r.get("best_fused")
        if not b or b["us_fused"] >= b["us_sep"]:
            continue
        nseg, ntot = (3, NQ + 2 * NKV) if r["layer"] == "qkv" else (2, NQ + NKV)
        nt = -(-r["B"] // 8)
        for tok in range(max(2, 8 * (nt - 1) + 1), 8 * nt + 1):
            lines.append(f"{nseg} {ntot} {K} {tok} {b['fv']['id']} {b['ksplit']}")
    (a.out / "qkv.table").write_text("\n".join(lines) + "\n")
    print(json.dumps({"qkv_table_lines": len(lines)}), flush=True)


if __name__ == "__main__":
    main()
