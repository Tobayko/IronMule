"""Time whole fused operators and compare outputs with source GGUF dequantization on the same channels."""

import argparse
from pathlib import Path
from statistics import median

import numpy as np
import torch
import triton

from ironmule.kernel_gen import q4k_tensors
from lab.pk6.mixed_progress.trial import board
from lab.pk6.native_dispatch.fused import fused_gemm
from lab.pk6.replay_http import checksum
from lab.pk6.warm_repeat import GGUF, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if checksum(args.gguf) != GGUF:
        raise ValueError("unexpected source weights")
    args.out.mkdir(mode=0o700, parents=True, exist_ok=False)
    result = []
    torch.manual_seed(530891)
    torch.backends.cuda.matmul.allow_tf32 = False
    for kind, q in (("Q4_K", 12), ("Q6_K", 14)):
        shapes = q4k_tensors(str(args.gguf), kind=kind)
        import gguf

        for (k, n), weight in shapes.items():
            if n > 65536:
                continue
            raw = weight["data"].reshape(n, -1)
            idx = np.linspace(0, n - 1, 64, dtype=int)
            dense = gguf.quants.dequantize(
                np.ascontiguousarray(raw[idx]), getattr(gguf.GGMLQuantizationType, kind)
            )
            reference = torch.tensor(dense, device="cuda", dtype=torch.float32)
            packed = torch.tensor(
                np.asarray(raw).copy(), device="cuda", dtype=torch.uint8
            )
            for b in (32, 143, 448):
                x = torch.randn((b, k), device="cuda")
                y = torch.empty((b, n), device="cuda")
                expected = x @ reference.T
                candidates = []
                for bm, bn, bk, warps in (
                    (32, 64, 32, 4),
                    (64, 64, 32, 4),
                    (64, 128, 64, 8),
                    (128, 128, 32, 8),
                ):

                    def invoke():
                        return fused_gemm[(triton.cdiv(b, bm) * triton.cdiv(n, bn),)](
                            packed,
                            x,
                            y,
                            b,
                            n,
                            k,
                            q,
                            bm,
                            bn,
                            bk,
                            num_warps=warps,
                            num_stages=3,
                        )

                    binary = invoke()
                    torch.cuda.synchronize()
                    error = y[:, idx] - expected
                    rms = float(
                        torch.linalg.vector_norm(error)
                        / torch.linalg.vector_norm(expected)
                    )
                    finite = bool(torch.isfinite(y).all())
                    if finite and rms < 0.004:
                        events = []
                        for _ in range(5):
                            a, e = (
                                torch.cuda.Event(enable_timing=True) for _ in range(2)
                            )
                            a.record()
                            invoke()
                            e.record()
                            events.append((a, e))
                        torch.cuda.synchronize()
                        ms = median(a.elapsed_time(e) for a, e in events)
                    else:
                        ms = None
                    candidates.append(
                        {
                            "tile": [bm, bn, bk, warps],
                            "relative_rms": rms,
                            "finite": finite,
                            "ms": ms,
                            "shared": binary.metadata.shared,
                            "registers": binary.n_regs,
                        }
                    )
                cell = {"type": kind, "k": k, "n": n, "b": b, "variants": candidates}
                save(args.out / f"cell-{q}-{n}-{k}-{b}.json", cell)
                result.append(cell)
    save(
        args.out / "result.json",
        {
            "board": board(),
            "gguf_sha256": GGUF,
            "triton_version": triton.__version__,
            "cells": result,
        },
    )


if __name__ == "__main__":
    main()
