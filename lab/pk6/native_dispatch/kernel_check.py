"""Compare the Ada plugin with dequantized source GGUF rows and report device time for its full operation."""

import argparse
import ctypes
from pathlib import Path
import statistics

import numpy as np

from ironmule.kernel_gen import q4k_tensors
from lab.pk6.mixed_progress.trial import board
from lab.pk6.replay_http import checksum
from lab.pk6.warm_repeat import GGUF, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import os

    os.environ["IRONMULE_LAB_FP8"] = "32"
    if checksum(args.gguf) != GGUF:
        raise ValueError("unexpected model")
    hw = board()
    torch.manual_seed(530891)
    torch.backends.cuda.matmul.allow_tf32 = False
    lib = ctypes.CDLL(str(args.lib.resolve()))
    call = lib.ironmule_kg_mul_mat
    ptr = ctypes.c_void_p
    call.argtypes = [
        ctypes.c_int,
        ptr,
        ptr,
        ptr,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ptr,
    ]
    call.restype = ctypes.c_int
    lib.im_prompt_dispatches.restype = ctypes.c_uint64
    args.out.mkdir(mode=0o700, parents=True, exist_ok=False)
    result = []
    for kind, number in (("Q4_K", 12), ("Q6_K", 14)):
        shapes = q4k_tensors(str(args.gguf), kind=kind)
        import gguf

        for (k, n), weight in shapes.items():
            if n > 65536:
                continue
            has_session = hasattr(lib, "im_prompt_begin")
            if has_session:
                lib.im_prompt_begin()
            raw = weight["data"].reshape(n, -1)
            # 64 output channels spread across the actual tensor, including its last row.
            indices = np.linspace(0, n - 1, 64, dtype=int)
            dense = gguf.quants.dequantize(
                np.ascontiguousarray(raw[indices]),
                getattr(gguf.GGMLQuantizationType, kind),
            )
            reference = torch.tensor(dense, device="cuda", dtype=torch.float32)
            weights = torch.tensor(
                np.asarray(raw).copy(), device="cuda", dtype=torch.uint8
            )
            for b in (32, 143, 448):
                x = torch.randn((b, k), device="cuda")
                y = torch.empty((b, n), device="cuda", dtype=torch.float32)
                expected = x @ reference.T
                stream = ptr(torch.cuda.current_stream().cuda_stream)

                def invoke():
                    return call(
                        number,
                        weights.data_ptr(),
                        x.data_ptr(),
                        y.data_ptr(),
                        n,
                        k,
                        b,
                        stream,
                    )

                before = lib.im_prompt_dispatches()
                rc = invoke()
                torch.cuda.synchronize()
                if rc:
                    result.append(
                        {
                            "type": kind,
                            "n": n,
                            "k": k,
                            "b": b,
                            "rc": rc,
                            "lt_status": lib.im_prompt_status(),
                        }
                    )
                    continue
                got = y[:, indices]
                error = got - expected
                starts = []
                ends = []
                for i in range(7):
                    a, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    a.record()
                    if invoke() != 0:
                        raise ValueError("kernel stopped dispatching")
                    e.record()
                    starts.append(a)
                    ends.append(e)
                torch.cuda.synchronize()
                result.append(
                    {
                        "type": kind,
                        "tensor": weight["name"],
                        "n": n,
                        "k": k,
                        "b": b,
                        "rc": 0,
                        "relative_rms": float(
                            torch.linalg.vector_norm(error)
                            / torch.linalg.vector_norm(expected)
                        ),
                        "absolute_max": float(error.abs().max()),
                        "finite": bool(torch.isfinite(got).all()),
                        "median_ms": statistics.median(
                            a.elapsed_time(e) for a, e in zip(starts, ends)
                        ),
                        "native_launches": lib.im_prompt_dispatches() - before,
                    }
                )
            if has_session:
                lib.im_prompt_end()
            save(args.out / f"shape-{number}-{n}-{k}.json", result[-3:])
    save(
        args.out / "result.json",
        {
            "hardware": hw,
            "weights": GGUF,
            "library_sha256": checksum(args.lib),
            "seed": 530891,
            "cells": result,
            "condition": "Real GGUF weights. Gaussian activations. 64 uniformly spaced channels checked; all channels computed.",
        },
    )


if __name__ == "__main__":
    main()
