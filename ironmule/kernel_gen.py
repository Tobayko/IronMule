"""KG1: CUDA kernels generated and tuned for the GPU they run on, for llama.cpp's Q4_K weights.

llama.cpp computes a quantized matmul of 1-8 tokens with one fixed kernel per weight type (`mul_mat_vec_q`), tuned once
for all GPUs. Here a family of kernels is generated from parameters, every member is checked and timed on this GPU with
this model's own weights, and the fastest one for each matrix shape and token count goes into a small library that
llama.cpp calls through the `kg1` hook (`native/ggml_patches/kg1`, `IRONMULE_KG_LIB`); for anything the library has no
kernel for, llama.cpp's own kernels run as before.

A kernel is (L, R, W, B): L lanes share one 256-weight block (each reads 128 / L bytes of quants at once), a warp
computes R output rows (the activations it loaded serve all of them), W warps per thread block, B tokens at once. The
activations are rounded to 8 bits per 32 values, as llama.cpp's own kernels do, and four multiply-adds run per
`__dp4a`. A candidate serves only if its error against a float64 reference (which decodes the format element by
element) is within 1.25x llama.cpp's own kernel's on the same data, and only where it is faster than llama.cpp's
kernel in the engine's own step (the engine measures that at startup and switches the library off per token count).

Measured on a Kaggle T4 (docs/LAB.md, KG1 v2): all Q4_K matmuls of a Qwen3-4B token 1.06x / 1.50x / 1.69x llama.cpp's
time at 1 / 4 / 8 tokens.

Needs `nvcc` (the CUDA toolkit llama.cpp was built with) on the machine; without it nothing is generated.
"""

from __future__ import annotations

import ctypes
import glob
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

KG_VERSION = 1
TYPE_Q4_K = 12  # GGML_TYPE_Q4_K
BATCHES = tuple(range(1, 9))  # from 9 tokens on llama.cpp's tensor-core kernel is faster (KG1 v2: 16.8 ms vs 42.4)
LANES, ROWS, WARPS = (8, 16, 32), (1, 2, 4, 8), (2, 4, 8)
ERROR_MARGIN = 1.25  # a kernel may be at most this much less exact than llama.cpp's own on the same data

DEVICE = r"""
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

struct block_q4_K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; };
static_assert(sizeof(block_q4_K) == 144, "Q4_K block");

__device__ __forceinline__ uint32_t sbyte(uint32_t y, uint32_t z, uint32_t w, int k) {
    const uint32_t word = k < 4 ? y : (k < 8 ? z : w);
    return (word >> (8 * (k & 3))) & 0xFF;
}
// the 6-bit scale and min of sub-block j (get_scale_min_k4 of ggml-quants.c) from bytes 4..15 of a block
__device__ __forceinline__ void scale_min(uint32_t y, uint32_t z, uint32_t w, int j, float &sc, float &m) {
    if (j < 4) {
        sc = (float)(sbyte(y, z, w, j) & 63); m = (float)(sbyte(y, z, w, j + 4) & 63);
    } else {
        sc = (float)((sbyte(y, z, w, j + 4) & 0xF) | ((sbyte(y, z, w, j - 4) >> 6) << 4));
        m = (float)((sbyte(y, z, w, j + 4) >> 4) | ((sbyte(y, z, w, j) >> 6) << 4));
    }
}

// activations to 8 bits per 32 values: one warp per 32 values
__global__ void quant_x(const float *__restrict__ x, int8_t *__restrict__ xq, float *__restrict__ xd, int n) {
    const int blk = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (blk * 32 >= n) return;
    const float v = x[blk * 32 + lane];
    float a = fabsf(v);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
    const float d = a / 127.f;
    xq[blk * 32 + lane] = (int8_t)(d == 0.f ? 0 : __float2int_rn(v / d));
    if (lane == 0) xd[blk] = d;
}

template <int L, int R, int W, int B>
__global__ void __launch_bounds__(W * 32) kq4k(const block_q4_K *__restrict__ wq, const int8_t *__restrict__ xq,
                                               const float *__restrict__ xd, float *__restrict__ y, int N, int K) {
    constexpr int G = 32 / L, BYTES = 128 / L, LPC = L / 4, WORDS = BYTES / 4;
    const int nb = K / 256;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row0 = (blockIdx.x * W + warp) * R;
    if (row0 >= N) return;
    const int g = lane / L, t = lane % L;
    const int c = t / LPC, l0 = (t % LPC) * BYTES;
    float acc[R][B];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int b = 0; b < B; b++) acc[r][b] = 0.f;
    for (int ib = g; ib < nb; ib += G) {
        int xl[B][WORDS], xh[B][WORDS];
        float dl[B], dh[B], sl[B], sh[B];
#pragma unroll
        for (int b = 0; b < B; b++) {
            const int8_t *pl = xq + (size_t)b * K + ib * 256 + 64 * c + l0, *ph = pl + 32;
            if constexpr (WORDS == 4) {
                const int4 a = __ldg(reinterpret_cast<const int4 *>(pl)), h = __ldg(reinterpret_cast<const int4 *>(ph));
                xl[b][0] = a.x; xl[b][1] = a.y; xl[b][2] = a.z; xl[b][3] = a.w;
                xh[b][0] = h.x; xh[b][1] = h.y; xh[b][2] = h.z; xh[b][3] = h.w;
            } else if constexpr (WORDS == 2) {
                const int2 a = __ldg(reinterpret_cast<const int2 *>(pl)), h = __ldg(reinterpret_cast<const int2 *>(ph));
                xl[b][0] = a.x; xl[b][1] = a.y; xh[b][0] = h.x; xh[b][1] = h.y;
            } else {
                xl[b][0] = __ldg(reinterpret_cast<const int *>(pl)); xh[b][0] = __ldg(reinterpret_cast<const int *>(ph));
            }
            const size_t sb = (size_t)b * (K / 32) + ib * 8 + 2 * c;
            dl[b] = __ldg(xd + sb); dh[b] = __ldg(xd + sb + 1);
            int il = 0, ih = 0;
#pragma unroll
            for (int w = 0; w < WORDS; w++) { il = __dp4a(0x01010101, xl[b][w], il); ih = __dp4a(0x01010101, xh[b][w], ih); }
            sl[b] = dl[b] * (float)il; sh[b] = dh[b] * (float)ih;
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
            if (row0 + r >= N) break;
            const block_q4_K *blk = wq + (size_t)(row0 + r) * nb + ib;
            const uint4 hdr = __ldg(reinterpret_cast<const uint4 *>(blk));
            const float d = __half2float(__ushort_as_half((unsigned short)(hdr.x & 0xFFFF)));
            const float dm = __half2float(__ushort_as_half((unsigned short)(hdr.x >> 16)));
            float sc1, m1, sc2, m2;
            scale_min(hdr.y, hdr.z, hdr.w, 2 * c, sc1, m1);
            scale_min(hdr.y, hdr.z, hdr.w, 2 * c + 1, sc2, m2);
            const float d1 = d * sc1, n1 = dm * m1, d2 = d * sc2, n2 = dm * m2;
            int q[WORDS];
            const uint8_t *qp = blk->qs + 32 * c + l0;
            if constexpr (WORDS == 4) {
                const int4 v = __ldg(reinterpret_cast<const int4 *>(qp)); q[0] = v.x; q[1] = v.y; q[2] = v.z; q[3] = v.w;
            } else if constexpr (WORDS == 2) {
                const int2 v = __ldg(reinterpret_cast<const int2 *>(qp)); q[0] = v.x; q[1] = v.y;
            } else {
                q[0] = __ldg(reinterpret_cast<const int *>(qp));
            }
#pragma unroll
            for (int b = 0; b < B; b++) {
                int il = 0, ih = 0;
#pragma unroll
                for (int w = 0; w < WORDS; w++) {
                    il = __dp4a(q[w] & 0x0F0F0F0F, xl[b][w], il);
                    ih = __dp4a((q[w] >> 4) & 0x0F0F0F0F, xh[b][w], ih);
                }
                acc[r][b] += (d1 * dl[b] * (float)il - n1 * sl[b]) + (d2 * dh[b] * (float)ih - n2 * sh[b]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int b = 0; b < B; b++) {
            float v = acc[r][b];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
            if (lane == 0 && row0 + r < N) y[(size_t)b * N + row0 + r] = v;
        }
}

typedef void (*launch_t)(const block_q4_K *, const int8_t *, const float *, float *, int, int, cudaStream_t);
template <int L, int R, int W, int B>
void launch(const block_q4_K *w, const int8_t *xq, const float *xd, float *y, int N, int K, cudaStream_t s) {
    const int rows = R * W;
    kq4k<L, R, W, B><<<(N + rows - 1) / rows, W * 32, 0, s>>>(w, xq, xd, y, N, K);
}
static void quantize(const float *x, int8_t *xq, float *xd, int n, cudaStream_t s) {
    quant_x<<<(n / 32 + 7) / 8, 256, 0, s>>>(x, xq, xd, n);
}
struct Variant { int L, R, W, B; launch_t fn; };
"""

TUNE = r"""
// float64 reference: one warp per output row, element by element (dequantize_row_q4_K of ggml-quants.c)
__global__ void ref64(const block_q4_K *__restrict__ wq, const float *__restrict__ x, double *__restrict__ y, int N,
                      int K, int B) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= N) return;
    const int nb = K / 256;
    for (int b = 0; b < B; b++) {
        double s = 0.0;
        for (int k = lane; k < K; k += 32) {
            const block_q4_K *blk = wq + (size_t)row * nb + k / 256;
            const int e = k % 256, chunk = e / 64, within = e % 64, j = 2 * chunk + (within >= 32), l = within % 32;
            const uint8_t *sc = blk->scales;
            int s6, m6;
            if (j < 4) { s6 = sc[j] & 63; m6 = sc[j + 4] & 63; }
            else { s6 = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); m6 = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
            const uint8_t byte = blk->qs[32 * chunk + l];
            const int q = within >= 32 ? byte >> 4 : byte & 0xF;
            s += ((double)__half2float(__ushort_as_half(blk->d)) * s6 * q -
                  (double)__half2float(__ushort_as_half(blk->dmin)) * m6) * (double)x[(size_t)b * K + k];
        }
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        if (lane == 0) y[(size_t)b * N + row] = s;
    }
}

extern "C" int kg_count() { return (int)VARIANTS.size(); }
extern "C" void kg_info(int i, int *out) { out[0] = VARIANTS[i].L; out[1] = VARIANTS[i].R; out[2] = VARIANTS[i].W; out[3] = VARIANTS[i].B; }

// time (us per matmul over `copies` weight copies, `reps` rounds) and error against float64 of variant i
extern "C" int kg_measure(int i, const void *w_host, int N, int K, const float *x_host, int copies, int reps,
                          double *us, double *rel_err) {
    const Variant &v = VARIANTS[i];
    const size_t wbytes = (size_t)N * (K / 256) * 144;
    std::vector<block_q4_K *> dw(copies);
    float *dx, *dy, *dxd; int8_t *dxq; double *dr;
    if (cudaMalloc(&dx, (size_t)v.B * K * 4) || cudaMalloc(&dy, (size_t)v.B * N * 4) || cudaMalloc(&dr, (size_t)v.B * N * 8) ||
        cudaMalloc(&dxq, (size_t)v.B * K) || cudaMalloc(&dxd, (size_t)v.B * K / 32 * 4)) return 1;
    for (int c = 0; c < copies; c++) {
        if (cudaMalloc(&dw[c], wbytes)) return 2;
        cudaMemcpy(dw[c], w_host, wbytes, cudaMemcpyHostToDevice);
    }
    cudaMemcpy(dx, x_host, (size_t)v.B * K * 4, cudaMemcpyHostToDevice);
    ref64<<<(N + 7) / 8, 256>>>(dw[0], dx, dr, N, K, v.B);
    quantize(dx, dxq, dxd, v.B * K, 0);
    v.fn(dw[0], dxq, dxd, dy, N, K, 0);
    if (cudaDeviceSynchronize() != cudaSuccess) return 3;
    std::vector<float> hy((size_t)v.B * N); std::vector<double> hr((size_t)v.B * N);
    cudaMemcpy(hy.data(), dy, hy.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hr.data(), dr, hr.size() * 8, cudaMemcpyDeviceToHost);
    double num = 0, den = 0;
    for (size_t k = 0; k < hy.size(); k++) { num += (hy[k] - hr[k]) * (hy[k] - hr[k]); den += hr[k] * hr[k]; }
    *rel_err = sqrt(num / den);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int r = 0; r < 3; r++) for (int c = 0; c < copies; c++) { quantize(dx, dxq, dxd, v.B * K, 0); v.fn(dw[c], dxq, dxd, dy, N, K, 0); }
    cudaEventRecord(e0);
    for (int r = 0; r < reps; r++) for (int c = 0; c < copies; c++) { quantize(dx, dxq, dxd, v.B * K, 0); v.fn(dw[c], dxq, dxd, dy, N, K, 0); }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    *us = 1000.0 * ms / ((double)reps * copies);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    for (int c = 0; c < copies; c++) cudaFree(dw[c]);
    cudaFree(dx); cudaFree(dy); cudaFree(dr); cudaFree(dxq); cudaFree(dxd);
    return cudaGetLastError() == cudaSuccess ? 0 : 4;
}
"""

LIBRARY = r"""
static uint32_t enabled_mask = 0xFFFFFFFFu;  // bit B-1: token count B is served here
extern "C" void ironmule_kg_set_mask(uint32_t mask) { enabled_mask = mask; }
extern "C" uint32_t ironmule_kg_get_mask() { return enabled_mask; }
extern "C" const char *ironmule_kg_table() { return TABLE_JSON; }

// the hook's entry: 0 when this library computed y = w x, non-zero for llama.cpp's own kernels
extern "C" int ironmule_kg_mul_mat(int type, const void *w, const float *x, float *y, int64_t N, int64_t K, int64_t B,
                                   cudaStream_t s) {
    if (type != 12 || K % 256 != 0 || B < 1 || B > 8 || !((enabled_mask >> (B - 1)) & 1u)) return 1;
    launch_t fn = nullptr;
    for (const Entry &e : ENTRIES)
        if (e.K == K && e.N == N && e.B == B) { fn = e.fn; break; }
    if (fn == nullptr) return 1;
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) return 1;
    static int8_t *xq[16]; static float *xd[16]; static size_t cap[16];
    const size_t need = (size_t)B * K;
    if (cap[dev] < need) {
        if (xq[dev]) { cudaFree(xq[dev]); cudaFree(xd[dev]); }
        if (cudaMalloc(&xq[dev], need) != cudaSuccess || cudaMalloc(&xd[dev], need / 32 * sizeof(float)) != cudaSuccess) {
            cap[dev] = 0; xq[dev] = nullptr; return 1;
        }
        cap[dev] = need;
    }
    quantize(x, xq[dev], xd[dev], (int)need, s);
    fn((const block_q4_K *)w, xq[dev], xd[dev], y, (int)N, (int)K, s);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}
"""


def variants(batches=BATCHES) -> list[tuple[int, int, int, int]]:
    return [(L, R, W, B) for B in batches for L, R, W in itertools.product(LANES, ROWS, WARPS) if R * B <= 64]


def tune_source(vs: list[tuple[int, int, int, int]]) -> str:
    rows = "\n".join(f"    {{{L}, {R}, {W}, {B}, launch<{L}, {R}, {W}, {B}>}}," for L, R, W, B in vs)
    return DEVICE + "static const std::vector<Variant> VARIANTS = {\n" + rows + "\n};\n" + TUNE


def library_source(table: dict[str, Any]) -> str:
    rows = "\n".join(f"    {{{e['K']}, {e['N']}, {e['B']}, launch<{e['L']}, {e['R']}, {e['W']}, {e['B']}>}},"
                     for e in table["entries"])
    text = json.dumps(table, sort_keys=True).replace("\\", "\\\\").replace('"', '\\"')
    return (DEVICE + "struct Entry { int K, N, B; launch_t fn; };\n"
            + "static const std::vector<Entry> ENTRIES = {\n" + rows + "\n};\n"
            + f'static const char *TABLE_JSON = "{text}";\n' + LIBRARY)


def nvcc_path() -> str | None:
    found = shutil.which("nvcc")
    if found:
        return found
    for path in sorted(glob.glob("/usr/local/cuda*/bin/nvcc"), reverse=True):
        return path
    return None


def compute_capability() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=30).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
    return out[0].replace(".", "") if out else None


def fingerprint() -> dict[str, Any]:
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        gpu = ""
    nvcc = nvcc_path()
    version = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1] \
        if nvcc else ""
    return {"gpu": gpu, "cc": compute_capability(), "nvcc": version, "kg": KG_VERSION}


def compile_cuda(source: str, out: Path, arch: str) -> None:
    nvcc = nvcc_path()
    if nvcc is None:
        raise RuntimeError("no nvcc: KG1 needs the CUDA toolkit")
    out.parent.mkdir(parents=True, exist_ok=True)
    src = out.with_suffix(".cu")
    src.write_text(source)
    done = subprocess.run([nvcc, "-O3", "-std=c++17", f"-arch=sm_{arch}", "-shared", "-Xcompiler", "-fPIC",
                           "--cudart", "shared", "-o", str(out), str(src)], capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f"nvcc failed: {done.stderr[-2000:]}")


def q4k_tensors(gguf_path: str, bindings: str | None = None) -> dict[tuple[int, int], dict[str, Any]]:
    """The model's Q4_K matrices by shape (K inputs, N outputs): how many, and one of them to tune on."""
    root = Path(bindings or os.environ.get("IRONMULE_GGML_BINDINGS", "")) / "vendor" / "llama.cpp" / "gguf-py"
    if root.exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import gguf  # llama.cpp's own reader

    shapes: dict[tuple[int, int], dict[str, Any]] = {}
    for t in gguf.GGUFReader(gguf_path).tensors:
        if t.tensor_type.name != "Q4_K" or len(t.shape) != 2:
            continue
        key = (int(t.shape[0]), int(t.shape[1]))
        entry = shapes.setdefault(key, {"count": 0, "name": t.name, "data": t.data})
        entry["count"] += 1
    return shapes


def tune(gguf_path: str, work: Path, *, reference_error: float = 6e-3, copies: int = 8, reps: int = 30,
         log=print) -> dict[str, Any]:
    """Every variant on this GPU with this model's own Q4_K matrices: the fastest correct one per shape and token count.
    `reference_error` is llama.cpp's own kernel's error against float64 on such data (KG1 v2 on a T4: 4.8-5.6e-3)."""
    import numpy as np

    arch = compute_capability()
    if arch is None:
        raise RuntimeError("no NVIDIA GPU")
    vs = variants()
    lib_path = work / f"kg_tune_sm{arch}_v{KG_VERSION}.so"
    started = time.time()
    if not lib_path.exists():
        compile_cuda(tune_source(vs), lib_path, arch)
    compile_s = time.time() - started
    lib = ctypes.CDLL(str(lib_path))
    lib.kg_measure.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                               ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    rng = np.random.default_rng(0)
    bound = ERROR_MARGIN * reference_error
    entries, measured = [], []
    for (K, N), t in sorted(q4k_tensors(gguf_path).items()):
        w = np.ascontiguousarray(np.asarray(t["data"]))
        x = rng.standard_normal((max(BATCHES), K)).astype(np.float32)
        for B in BATCHES:
            best = None
            for i, (L, R, W, b) in enumerate(vs):
                if b != B:
                    continue
                us, err = ctypes.c_double(), ctypes.c_double()
                code = lib.kg_measure(i, w.ctypes.data, N, K, x.ctypes.data, copies, reps, ctypes.byref(us),
                                      ctypes.byref(err))
                row = {"K": K, "N": N, "B": B, "L": L, "R": R, "W": W, "us": round(us.value, 2),
                       "rel_err": err.value, "code": code}
                measured.append(row)
                if code == 0 and err.value <= bound and (best is None or us.value < best["us"]):
                    best = row
            if best is not None:
                entries.append({k: best[k] for k in ("K", "N", "B", "L", "R", "W", "us")})
                log(json.dumps({"kg_tuned": entries[-1]}))
    return {"version": KG_VERSION, "fingerprint": fingerprint(), "arch": arch, "entries": entries,
            "error_bound": bound, "compile_s": round(compile_s, 1), "tune_s": round(time.time() - started, 1),
            "measured": measured}


def build(table: dict[str, Any], out: Path) -> Path:
    compile_cuda(library_source({k: v for k, v in table.items() if k != "measured"}), out, table["arch"])
    return out


def ensure(gguf_path: str, cache: Path, log=print) -> Path | None:
    """The library for this GPU, nvcc and model shapes, tuned and built once and reused after; None where it cannot be."""
    if nvcc_path() is None or compute_capability() is None:
        return None
    fp = fingerprint()
    shapes = sorted(q4k_tensors(gguf_path))
    if not shapes:
        return None
    key = hashlib.sha256(json.dumps({"fp": fp, "shapes": shapes}, sort_keys=True).encode()).hexdigest()[:16]
    out = cache / f"kg_{key}.so"
    if out.exists():
        return out
    table = tune(gguf_path, cache, log=log)
    (cache / f"kg_{key}.json").write_text(json.dumps(table, indent=1))
    if not table["entries"]:
        return None
    return build(table, out)


__all__ = ["BATCHES", "KG_VERSION", "build", "ensure", "fingerprint", "library_source", "q4k_tensors", "tune",
           "tune_source", "variants"]
