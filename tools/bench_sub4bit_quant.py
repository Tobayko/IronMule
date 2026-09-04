#!/usr/bin/env python3
"""Benchmark Sub-4-Bit Quantization on Apple Silicon M1 Max GPU.

Evaluates 2-bit, 3-bit, 4-bit, and 8-bit QuantGEMM in MLX:
- Decode Latency (Batch 1, memory bandwidth bound)
- Prefill Latency (Batch 128, compute bound)
- Effective UMA Memory Bandwidth (GB/s)
- Reconstruction Error (MSE, Cosine Similarity, Max Absolute Error)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mlx.core as mx
import mlx.nn as nn


def bench_quant_level(bits: int, group_size: int = 64, M: int = 1, K: int = 2560, N: int = 10240, guard=None):
    # Simulated MLP layer weight (K x N)
    w_fp16 = mx.random.normal(shape=(N, K), dtype=mx.float16) * 0.02
    x = mx.random.normal(shape=(M, K), dtype=mx.float16)

    # Reference FP16 forward
    ref_out = x @ w_fp16.T
    mx.eval(ref_out)

    # Quantize
    w_q, scales, biases = mx.quantize(w_fp16, group_size=group_size, bits=bits)
    mx.eval(w_q, scales, biases)

    # Measure memory footprint
    weight_bytes = w_q.nbytes + scales.nbytes + (biases.nbytes if biases is not None else 0)

    # Quantized forward pass
    def quant_forward():
        return mx.quantized_matmul(x, w_q, scales, biases, transpose=True, group_size=group_size, bits=bits)

    # Correctness check
    q_out = quant_forward()
    mx.eval(q_out)
    diff = mx.abs(ref_out - q_out)
    max_err = float(mx.max(diff).item())
    mse = float(mx.mean(diff ** 2).item())
    cos_sim = float((mx.sum(ref_out * q_out) / (mx.linalg.norm(ref_out) * mx.linalg.norm(q_out))).item())

    # Warmup — evaluate each iteration or MLX keeps the graph lazy and only the
    # last array is ever realised.
    for _ in range(10):
        out = quant_forward()
        mx.eval(out)
    mx.synchronize()

    # Benchmark iterations
    iters = 100
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        out = quant_forward()
        mx.eval(out)
    mx.synchronize()
    total_ns = time.perf_counter_ns() - t0
    lat_us = (total_ns / iters) / 1e3
    if guard is not None:
        guard.record_gpu(total_ns / 1e9)

    # Bandwidth in GB/s: (weight_bytes + input_bytes + output_bytes) / lat_sec
    data_bytes = weight_bytes + x.nbytes + out.nbytes
    bw_gbs = (data_bytes / (lat_us * 1e-6)) / 1e9

    # TFLOPS: 2 * M * K * N ops
    flops = 2.0 * M * K * N
    tflops = (flops / (lat_us * 1e-6)) / 1e12

    return {
        "bits": bits,
        "bytes_mb": weight_bytes / (1024**2),
        "latency_us": lat_us,
        "bw_gbs": bw_gbs,
        "tflops": tflops,
        "mse": mse,
        "max_err": max_err,
        "cos_sim": cos_sim,
    }


def main():
    from _bench import harness_preconditions

    guard = harness_preconditions()
    print("================================================================================")
    print("🔬 SUB-4-BIT QUANTIZATION & MEMORY ROOFLINE (M1 Max)")
    print("================================================================================")
    print("Layer Dimension: MLP Projection K=2560 -> N=10240 (Gemma 4B intermediate)")
    print("Unified Memory Theoretical Peak: 400 GB/s | Peak FP16 Compute: 10.4 TFLOPS\n")

    # --- Decode Regime (Batch M=1) ---
    print("--- 1. DECODE REGIME (Batch M=1, Memory-Bandwidth Bound) ---")
    print(f"{'Bits':<6} | {'Size':<10} | {'Latency':<10} | {'UMA Bandwidth':<15} | {'Cos Sim':<10} | {'MSE':<12}")
    print("-" * 75)

    decode_results = []
    for b in [8, 4, 3, 2]:
        res = bench_quant_level(bits=b, M=1, guard=guard)
        decode_results.append(res)
        print(
            f"{b:<6} | "
            f"{res['bytes_mb']:5.2f} MB   | "
            f"{res['latency_us']:6.1f} µs   | "
            f"{res['bw_gbs']:6.1f} GB/s       | "
            f"{res['cos_sim']:8.5f}   | "
            f"{res['mse']:10.6f}"
        )

    # --- Prefill Regime (Batch M=128) ---
    print("\n--- 2. PREFILL REGIME (Batch M=128, Compute Bound) ---")
    print(f"{'Bits':<6} | {'Size':<10} | {'Latency':<10} | {'Compute':<15} | {'Cos Sim':<10} | {'MSE':<12}")
    print("-" * 75)

    prefill_results = []
    for b in [8, 4, 3, 2]:
        res = bench_quant_level(bits=b, M=128, guard=guard)
        prefill_results.append(res)
        print(
            f"{b:<6} | "
            f"{res['bytes_mb']:5.2f} MB   | "
            f"{res['latency_us']:6.1f} µs   | "
            f"{res['tflops']:6.2f} TFLOPS     | "
            f"{res['cos_sim']:8.5f}   | "
            f"{res['mse']:10.6f}"
        )

    print("\n" + "=" * 75)
    print("📊 ROOFLINE FAZIT:")
    ratio_4b_to_3b = decode_results[1]["latency_us"] / decode_results[2]["latency_us"]
    ratio_4b_to_2b = decode_results[1]["latency_us"] / decode_results[3]["latency_us"]
    print(f"   4-Bit -> 3-Bit Decode Speedup: {ratio_4b_to_3b:.2f}x")
    print(f"   4-Bit -> 2-Bit Decode Speedup: {ratio_4b_to_2b:.2f}x")
    print("================================================================================")


if __name__ == "__main__":
    main()
