"""W4MMA: IronMule's tensor-core kernels for GGUF Q4_K / Q6_K weights times 1-32 tokens, chosen on the GPU itself.

`native/kernels/w4mma.cu` holds a family of kernels that stream llama.cpp's own quant blocks through a multi-stage
shared-memory pipeline and multiply on mma.sync.m16n8k16 (fp16 in, fp32 accumulation). A variant fixes how the GPU is
addressed: warps per block, 16-row tiles per warp, super-blocks per pipeline stage, pipeline depth and the split of K
over blocks. This module compiles the family for the visible GPU, measures every variant on this model's own weight
matrices and token counts, checks each result against a float64 reference that decodes the format element by element,
and writes the fastest correct variant per (type, rows, columns, tokens) to a table. The library reads the table
(`IRONMULE_W4_TABLE`) and serves through llama.cpp's kg1 hook (`IRONMULE_KG_LIB`); the engine's own startup measurement
(`GgmlEngine.measure_kg`) then keeps it only for the token counts where the whole step got faster.

  python -m ironmule.w4_kernels --gguf MODEL.gguf --out DIR      # prints the library and table paths
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .kernel_gen import compile_cuda, compute_capability, fingerprint, nvcc_path, q4k_tensors

SOURCE = Path(__file__).resolve().parents[1] / "native" / "kernels" / "w4mma.cu"
TYPE_Q4_K, TYPE_Q6_K = 12, 14
TOKENS = (1, 8, 16, 32)  # measured; every token count is served by the measurement of its tile count
KSPLITS = (1, 2, 4, 8)  # 8: long-K matrices with few output rows (down, 25600 -> 5120) at 32 tokens
ERROR_BOUND = 4e-3  # relative L2 error against float64; fp16 activations and weights give about 3e-4
VERSION = 2  # 2: W4R, weights repacked once into tile order


def dequant(raw: Any, kind: str, n_rows: int, k: int):
    """float64 [n_rows, k] of GGUF Q4_K or Q6_K bytes, decoded as ggml's dequantize_row_* does."""
    import numpy as np

    nb = k // 256
    if kind == "Q4_K":
        b = np.frombuffer(bytes(raw), dtype=np.uint8).reshape(n_rows, nb, 144)
        d = b[..., 0:2].copy().view(np.float16).astype(np.float64)[..., 0]
        dmin = b[..., 2:4].copy().view(np.float16).astype(np.float64)[..., 0]
        sc = b[..., 4:16].astype(np.int64)
        qs = b[..., 16:144].astype(np.int64)
        scale = np.zeros((n_rows, nb, 8)); mins = np.zeros((n_rows, nb, 8))
        for j in range(8):
            if j < 4:
                s, m = sc[..., j] & 63, sc[..., j + 4] & 63
            else:
                s = (sc[..., j + 4] & 0xF) | ((sc[..., j - 4] >> 6) << 4)
                m = (sc[..., j + 4] >> 4) | ((sc[..., j] >> 6) << 4)
            scale[..., j], mins[..., j] = d * s, dmin * m
        out = np.zeros((n_rows, nb, 256))
        for grp in range(4):
            q = qs[..., 32 * grp:32 * grp + 32]
            out[..., 64 * grp:64 * grp + 32] = (q & 0xF) * scale[..., 2 * grp, None] - mins[..., 2 * grp, None]
            out[..., 64 * grp + 32:64 * grp + 64] = (q >> 4) * scale[..., 2 * grp + 1, None] - mins[..., 2 * grp + 1, None]
        return out.reshape(n_rows, k)
    if kind == "Q6_K":
        b = np.frombuffer(bytes(raw), dtype=np.uint8).reshape(n_rows, nb, 210)
        ql, qh = b[..., 0:128].astype(np.int64), b[..., 128:192].astype(np.int64)
        sc = b[..., 192:208].view(np.int8).astype(np.float64)
        d = b[..., 208:210].copy().view(np.float16).astype(np.float64)[..., 0]
        out = np.zeros((n_rows, nb, 256))
        for half in range(2):
            l_ = ql[..., 64 * half:64 * half + 64]
            h_ = qh[..., 32 * half:32 * half + 32]
            s_ = sc[..., 8 * half:8 * half + 8]
            for l in range(32):
                i = l // 16
                q1 = ((l_[..., l] & 0xF) | (((h_[..., l] >> 0) & 3) << 4)) - 32
                q2 = ((l_[..., l + 32] & 0xF) | (((h_[..., l] >> 2) & 3) << 4)) - 32
                q3 = ((l_[..., l] >> 4) | (((h_[..., l] >> 4) & 3) << 4)) - 32
                q4 = ((l_[..., l + 32] >> 4) | (((h_[..., l] >> 6) & 3) << 4)) - 32
                base = 128 * half + l
                out[..., base] = d * s_[..., i] * q1
                out[..., base + 32] = d * s_[..., i + 2] * q2
                out[..., base + 64] = d * s_[..., i + 4] * q3
                out[..., base + 96] = d * s_[..., i + 6] * q4
        return out.reshape(n_rows, k)
    raise ValueError(kind)


def build(out: Path, arch: str | None = None) -> Path:
    arch = arch or compute_capability()
    if arch is None or int(arch) < 80:
        raise RuntimeError("W4MMA needs an Ampere or newer NVIDIA GPU (sm_80+)")
    lib = out / f"w4mma_sm{arch}_v{VERSION}_{source_digest()[:12]}.so"
    if not lib.exists():
        compile_cuda(SOURCE.read_text(), lib, arch, include=SOURCE.parent)
    return lib


def source_digest() -> str:
    """The kernel source and the MQ1 headers it includes (native/kernels/mq1)."""
    h = hashlib.sha256(SOURCE.read_bytes())
    for path in sorted((SOURCE.parent / "mq1").rglob("*")):
        if path.is_file() and path.suffix in (".h", ".cuh", ".hpp"):
            h.update(path.relative_to(SOURCE.parent).as_posix().encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def _lib(path: Path):
    lib = ctypes.CDLL(str(path))
    lib.im_w4_variant_count.restype = ctypes.c_int
    lib.im_w4_variant.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.im_w4_bench.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_long, ctypes.c_long,
                                ctypes.c_void_p, ctypes.c_long, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.POINTER(ctypes.c_double), ctypes.c_void_p]
    lib.im_w4_bench.restype = ctypes.c_int
    return lib


def variants(lib) -> list[dict[str, int]]:
    out = []
    for i in range(lib.im_w4_variant_count()):
        v = (ctypes.c_int * 6)()
        lib.im_w4_variant(i, v)
        out.append({"id": i, "type": v[0], "warps": v[1], "ks": v[2], "nt": v[3]})
    return out


def tune(gguf: str, lib_path: Path, *, tokens=TOKENS, reps: int = 20, log=print, max_ref_rows: int = 4096,
         shapes=None, ksplits=KSPLITS, kinds=("Q4_K", "Q6_K")) -> dict:
    import numpy as np

    lib = _lib(lib_path)
    vs = variants(lib)
    rng = np.random.default_rng(0)
    entries, measured = [], []
    started = time.time()
    for kind, code in (("Q4_K", TYPE_Q4_K), ("Q6_K", TYPE_Q6_K)):
        if kind not in kinds:
            continue
        for (K, N), t in sorted(q4k_tensors(gguf, kind=kind).items()):
            if shapes is not None and (K, N) not in shapes:
                continue
            if N > 100000 and shapes is None and kind == "Q4_K":
                continue  # the token embedding is a row lookup, not a matmul; the Q6_K output matrix is tuned
            raw = np.ascontiguousarray(np.asarray(t["data"])).reshape(-1)
            wbytes = raw.nbytes
            copies = max(1, min(8, int(2e9 // max(1, wbytes))))  # beyond the 40 MB L2 cache
            ref_rows = min(N, max_ref_rows)
            bs = 144 if kind == "Q4_K" else 210
            wref = dequant(raw[:ref_rows * (K // 256) * bs], kind, ref_rows, K)
            x = rng.standard_normal((max(tokens), K)).astype(np.float32)
            for B in tokens:
                ref = x[:B].astype(np.float64) @ wref.T  # [B, ref_rows]
                y = np.zeros((B, N), dtype=np.float32)
                best = None
                for v in vs:
                    if v["type"] != code or v["nt"] != -(-B // 8):
                        continue
                    for ks in ksplits:
                        if ks > 1 and (K // 256) < 2 * ks:
                            continue
                        us = ctypes.c_double()
                        rc = lib.im_w4_bench(v["id"], raw.ctypes.data, wbytes, N, K, x.ctypes.data, B, ks, copies,
                                             reps, ctypes.byref(us), y.ctypes.data)
                        err = float("inf")
                        if rc == 0:
                            got = y[:, :ref_rows].astype(np.float64)
                            err = float(np.linalg.norm(got - ref) / max(1e-30, np.linalg.norm(ref)))
                        row = {"type": code, "K": K, "N": N, "B": B, "variant": v["id"], "ksplit": ks, "rc": rc,
                               "us": round(us.value, 2) if rc == 0 else None, "rel_err": err}
                        measured.append(row)
                        if rc == 0 and err <= ERROR_BOUND and (best is None or us.value < best["us"]):
                            best = row
                if best is not None:
                    gbs = wbytes / best["us"] / 1e3
                    entries.append({**best, "gb_s": round(gbs, 1)})
                    log(json.dumps({"w4_tuned": entries[-1], "config": vs[best["variant"]]}), flush=True)
    return {"version": VERSION, "fingerprint": fingerprint(), "entries": entries, "measured": measured,
            "tune_s": round(time.time() - started, 1), "error_bound": ERROR_BOUND}


def table_lines(result: dict) -> str:
    """Every token count 1-32 gets the variant measured at the nearest larger measured count of its tile count."""
    best = {(e["type"], e["N"], e["K"], e["B"]): e for e in result["entries"]}
    lines = []
    shapes = sorted({(e["type"], e["N"], e["K"]) for e in result["entries"]})
    for code, n, k in shapes:
        measured_b = sorted(b for (c, nn, kk, b) in best if (c, nn, kk) == (code, n, k))
        for tok in range(1, 33):
            cand = [b for b in measured_b if b >= tok and -(-b // 8) == -(-tok // 8)]
            if not cand:
                continue
            e = best[(code, n, k, cand[0])]
            lines.append(f"{code} {n} {k} {tok} {e['variant']} {e['ksplit']}")
    return "\n".join(lines) + "\n"


def ensure(gguf: str, cache: Path, log=print) -> tuple[Path, Path] | None:
    """The library and table for this GPU, compiler and GGUF, tuned once and reused."""
    from .ggml_engine import file_sha256

    if nvcc_path() is None:
        return None
    arch = compute_capability()
    if arch is None or int(arch) < 80:
        return None
    fp = fingerprint()
    key = hashlib.sha256(json.dumps({"fp": fp, "gguf": file_sha256(gguf), "v": VERSION,
                                     "src": source_digest()},
                                    sort_keys=True).encode()).hexdigest()[:24]
    cache.mkdir(parents=True, exist_ok=True)
    lib = build(cache, arch)
    table, record = cache / f"{key}.table", cache / f"{key}.json"
    if table.exists() and record.exists():
        return lib, table
    result = tune(gguf, lib, log=log)
    record.write_text(json.dumps(result))
    table.write_text(table_lines(result))
    return lib, table


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--gguf", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--quick", help="K:N,K:N shapes; tunes only those (1, 8, 32 tokens) and prints, no table")
    a = p.parse_args(argv)
    if a.quick:
        shapes = {tuple(int(v) for v in s.split(":")) for s in a.quick.split(",")}
        lib = build(a.out)
        res = tune(a.gguf, lib, tokens=(1, 8, 32), shapes=shapes)
        best = {}
        for r in res["measured"]:
            if r["rc"] == 0 and r["rel_err"] <= ERROR_BOUND:
                k = (r["type"], r["K"], r["N"], r["B"])
                best.setdefault(k, []).append(r)
        for k, rows in sorted(best.items()):
            rows.sort(key=lambda r: r["us"])
            print(k, [(r["variant"], r["ksplit"], r["us"]) for r in rows[:3]], flush=True)
        return 0
    got = ensure(a.gguf, a.out)
    if got is None:
        print("no CUDA toolkit or GPU older than sm_80")
        return 1
    print(f"export IRONMULE_KG_LIB={got[0]} IRONMULE_W4_TABLE={got[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
