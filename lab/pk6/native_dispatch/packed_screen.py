"""Fit a restricted Q4_K Marlin routing table from real native graphs and measured KV headroom."""

import argparse
import json
import math
import os
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ironmule.kernel_gen import q4k_tensors
from lab.pk6.mixed_progress import trial
from lab.pk6.native_dispatch import operators
from lab.pk6.replay_http import checksum
from lab.pk6.warm_repeat import GGUF, save

WIDTHS = (16, 24, 30, 32, 64, 128, 143, 256, 384, 432, 448, 449, 464, 512, 1024, 2048)
SEEDS = (913099, 347711)


def screen(model, library, check, compact, out):
    arith = json.loads(check.read_bytes())
    memory = json.loads(compact.read_bytes())
    hardware = trial.board()
    if (
        hardware != arith["hardware"]
        or hardware != memory["hardware"]
        or checksum(model) != arith["weights"]
        or arith["weights"] != GGUF
        or checksum(library) != arith["library_sha256"]
    ):
        raise ValueError("foreign hardware, GGUF or packed program")
    out.mkdir(exist_ok=False)
    spec = {
        "hardware": hardware,
        "gguf": GGUF,
        "plugin": checksum(library),
        "memory": memory,
        "widths": WIDTHS,
        "seeds": SEEDS,
        "arithmetic_sha256": checksum(check),
        "backend": checksum(Path(os.environ["LLAMA_CPP_LIB_PATH"]) / "libggml-cuda.so"),
    }
    save(out / "protocol.json", spec)
    shapes = {
        (c["n"], c["k"])
        for c in arith["cells"]
        if c["type"] == "Q4_K"
        and not c["rc"]
        and c["finite"]
        and c["relative_rms"] < 0.002
        and c["b"] == 448
    }
    measured = {}
    with tempfile.TemporaryDirectory(dir=out) as temp:
        tmp = Path(temp)
        exe, source = operators.build(tmp, model_lifetime=True)
        for seed in SEEDS:
            rng = random.Random(seed)
            blocks = [
                (k, n, t)
                for (k, n), t in q4k_tensors(str(model)).items()
                if (n, k) in shapes
            ]
            rng.shuffle(blocks)
            for k, n, t in blocks:
                weights, inputs = tmp / "weights", tmp / "input"
                t["data"].tofile(weights)
                np.random.default_rng(seed + n + k).standard_normal(
                    (max(WIDTHS), k), dtype=np.float32
                ).tofile(inputs)
                copies = min(
                    48, max(2, math.ceil((128 << 20) / weights.stat().st_size))
                )
                widths = list(WIDTHS)
                arms = ["stock", "marlin"]
                rng.shuffle(widths)
                rng.shuffle(arms)
                cells = {w: {} for w in widths}
                for arm in arms:
                    env = dict(
                        os.environ,
                        GGML_CUDA_DISABLE_GRAPHS="1",
                        REF_TYPE="12",
                        IRONMULE_CUBLAS_MIN_ROWS="0",
                        IRONMULE_KG_LIB=str(library) if arm == "marlin" else "",
                    )
                    proc = subprocess.run(
                        [
                            exe,
                            weights,
                            str(n),
                            str(k),
                            inputs,
                            tmp / arm,
                            str(copies),
                            "8",
                            *map(str, widths),
                        ],
                        text=True,
                        capture_output=True,
                        env=env,
                        check=False,
                    )
                    if proc.returncode:
                        raise RuntimeError(proc.stderr[-1000:])
                    records = [
                        json.loads(l)
                        for l in proc.stdout.splitlines()
                        if l.startswith("{")
                    ]
                    for w, timed, witness in zip(
                        widths, records[::2], records[1::2], strict=True
                    ):
                        if [timed[i] for i in ("B", "N", "K", "type")] != [
                            w,
                            n,
                            k,
                            12,
                        ] or not 0 < timed["us"] < math.inf:
                            raise ValueError("unmatched graph timing")
                        cells[w][arm] = {**witness, "us": timed["us"]}
                for w in widths:
                    ref = np.fromfile(tmp / f"stock_B{w}.bin", np.float32)
                    y = np.fromfile(tmp / f"marlin_B{w}.bin", np.float32)
                    if (
                        ref.size != 64 * w
                        or ref.shape != y.shape
                        or not np.linalg.norm(ref)
                    ):
                        raise ValueError("missing native output")
                    error = float(np.linalg.norm(y - ref) / np.linalg.norm(ref))
                    if not math.isfinite(error):
                        raise ValueError("nonfinite native output")
                    cells[w]["relative_rms_to_stock"] = error
                    measured.setdefault((n, k, w), []).append(cells[w])
                    for name in arms:
                        (tmp / f"{name}_B{w}.bin").unlink()
                save(
                    out / f"shape-{seed}-{n}-{k}.json",
                    {
                        "n": n,
                        "k": k,
                        "seed": seed,
                        "copies": copies,
                        "cells": cells,
                        "harness": source,
                    },
                )
    intervals = []
    for n, k, w in sorted(measured):
        rows = measured[n, k, w]
        if len(rows) != 2:
            raise ValueError("missing independent activation seed")
        if not all(
            r["marlin"]["graphs"] == r["marlin"]["dispatches"] > 0
            and r["relative_rms_to_stock"] < 0.04
            and r["marlin"]["us"] * 1.10 < r["stock"]["us"]
            for r in rows
        ):
            continue
        previous = next((v for v in reversed(WIDTHS) if v < w), None)
        if intervals and intervals[-1][:2] == [n, k] and intervals[-1][-1] == previous:
            intervals[-1][-1] = w
        else:
            intervals.append([n, k, w, w])
    save(
        out / "plan.json",
        {
            **spec,
            "ranges": intervals,
            "meaning": "Two native-graph seeds selected candidate intervals. Intermediate widths and whole-model performance require independent system qualification.",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("model", "library", "arithmetic", "memory", "out"):
        p.add_argument(field, type=Path)
    a = p.parse_args()
    screen(a.model, a.library, a.arithmetic, a.memory, a.out)
