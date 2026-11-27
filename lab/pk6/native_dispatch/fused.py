"""Fused Q4_K/Q6_K unpack and FP16 tensor-core GEMM. Reads the original unmodified GGUF blocks."""

import triton
import triton.language as tl


@triton.jit
def fused_gemm(
    W,
    X,
    Y,
    B,
    N,
    K,
    TYPE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # Group M tiles so nearby thread blocks can reuse compressed weight tiles in L2.
    pid = tl.program_id(0)
    num_m = tl.cdiv(B, BM)
    num_n = tl.cdiv(N, BN)
    group = pid // (4 * num_n)
    start_m = group * 4
    size_m = tl.minimum(num_m - start_m, 4)
    mid = start_m + (pid % (4 * num_n)) % size_m
    nid = (pid % (4 * num_n)) // size_m
    m = mid * BM + tl.arange(0, BM)
    n = nid * BN + tl.arange(0, BN)
    r = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    # Row strides follow block_q4_K and block_q6_K in GGUF, including the fp16 metadata.
    for chunk in range(tl.cdiv(K, BK)):
        k = chunk * BK + r
        blocks = n[:, None] * (K // 256) + k[None, :] // 256
        if TYPE == 12:
            block = W + blocks * 144
            g = (k % 256) // 32
            base = (block + 4).to(tl.pointer_type(tl.uint8))
            lo = tl.load(base + g[None, :], mask=n[:, None] < N, other=0).to(tl.int32)
            top = tl.load(
                base + tl.where(g < 4, g + 4, g - 4)[None, :],
                mask=n[:, None] < N,
                other=0,
            ).to(tl.int32)
            mix = tl.load(
                base + tl.minimum(g + 4, 11)[None, :], mask=n[:, None] < N, other=0
            ).to(tl.int32)
            s = tl.where(g[None, :] < 4, lo & 63, (mix & 15) | ((top >> 6) << 4))
            z = tl.where(g[None, :] < 4, top & 63, (mix >> 4) | ((lo >> 6) << 4))
            d = tl.load(
                block.to(tl.pointer_type(tl.float16)), mask=n[:, None] < N, other=0
            ).to(tl.float32)
            mn = tl.load(
                (block + 2).to(tl.pointer_type(tl.float16)),
                mask=n[:, None] < N,
                other=0,
            ).to(tl.float32)
            q = tl.load(
                block + 16 + ((k % 256) // 64 * 32 + k % 32)[None, :],
                mask=n[:, None] < N,
                other=0,
            ).to(tl.int32)
            v = tl.where((g % 2)[None, :] == 1, q >> 4, q & 15)
            w = (v * s * d - z * mn).to(tl.float16)
        else:
            block = W + blocks * 210
            g = k % 128 // 32
            lo = tl.load(
                block + (k % 256 // 128 * 64 + (g % 2) * 32 + k % 32)[None, :],
                mask=n[:, None] < N,
                other=0,
            ).to(tl.int32)
            hi = tl.load(
                block + 128 + (k % 256 // 128 * 32 + k % 32)[None, :],
                mask=n[:, None] < N,
                other=0,
            ).to(tl.int32)
            q = (
                tl.where(g[None, :] < 2, lo & 15, lo >> 4)
                | (((hi >> (g * 2)[None, :]) & 3) << 4)
            ) - 32
            s = tl.load(
                (block + 192 + (k % 256 // 16)[None, :]).to(tl.pointer_type(tl.int8)),
                mask=n[:, None] < N,
                other=0,
            )
            d = tl.load(
                (block + 208).to(tl.pointer_type(tl.float16)),
                mask=n[:, None] < N,
                other=0,
            ).to(tl.float32)
            w = (q * s.to(tl.int32) * d).to(tl.float16)
        x = tl.load(X + m[:, None] * K + k[None, :], mask=m[:, None] < B, other=0).to(
            tl.float16
        )
        acc += tl.dot(x, tl.trans(w))
    tl.store(
        Y + m[:, None] * N + n[None, :], acc, mask=(m[:, None] < B) & (n[None, :] < N)
    )
