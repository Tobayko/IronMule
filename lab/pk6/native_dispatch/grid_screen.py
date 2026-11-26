"""Screen native Q4_K launch grids against stock with disjoint weights and held-out activation seeds."""

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
from lab.pk6.mixed_progress.trial import board
from lab.pk6.native_dispatch import operators
from lab.pk6.replay_http import checksum
from lab.pk6.warm_repeat import GGUF, save

MODES = {"stock": 0, "xy": 1, "xy64": 2, "sk64": 3}
WIDTHS = (9, 16, 24, 32, 64, 128, 143, 256, 384, 432, 448, 449, 464, 512, 1024, 2048)
SEEDS = (323021, 661907)


def screen(gguf, built, out):
    native = built / "bin/libggml-cuda.so"
    manifest = json.loads((built / "build.json").read_bytes())
    if (
        checksum(native) != manifest["backend"]
        or board() != manifest["hardware"]
        or checksum(gguf) != GGUF
        or manifest["plan"] is not None
    ):
        raise ValueError("unverified screen binary or model")
    out.mkdir(exist_ok=False, parents=True)
    save(
        out / "protocol.json",
        {**manifest, "gguf": GGUF, "widths": WIDTHS, "modes": MODES, "seeds": SEEDS},
    )
    with tempfile.TemporaryDirectory(dir=out) as temp:
        tmp = Path(temp)
        exe, harness = operators.build(tmp, native_grid=True)
        for seed in SEEDS:
            rng = random.Random(seed)
            shapes = [
                (k, n, t) for (k, n), t in q4k_tensors(str(gguf)).items() if n <= 65536
            ]
            rng.shuffle(shapes)
            for k, n, t in shapes:
                w = tmp / "weights"
                t["data"].tofile(w)
                x = tmp / "activations"
                np.random.default_rng(seed + n + k).standard_normal(
                    (max(WIDTHS), k), dtype=np.float32
                ).tofile(x)
                copies = min(48, max(2, math.ceil((128 << 20) / w.stat().st_size)))
                widths = list(WIDTHS)
                modes = list(MODES)
                rng.shuffle(widths)
                rng.shuffle(modes)
                rows = {b: {} for b in widths}
                for mode in modes:
                    env = dict(
                        os.environ,
                        REF_TYPE="12",
                        IRONMULE_KG_LIB="",
                        IRONMULE_CUBLAS_MIN_ROWS="0",
                        GGML_CUDA_DISABLE_GRAPHS="1",
                        IRONMULE_GRID_BACKEND=str(native),
                        IRONMULE_LAB_MMQ_TILE=mode,
                        LD_LIBRARY_PATH=str(native.parent)
                        + ":"
                        + os.getenv("LD_LIBRARY_PATH", ""),
                    )
                    proc = subprocess.run(
                        [
                            exe,
                            w,
                            str(n),
                            str(k),
                            x,
                            tmp / mode,
                            str(copies),
                            "8",
                            *map(str, widths),
                        ],
                        env=env,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    if proc.returncode:
                        raise RuntimeError(proc.stderr[-1200:])
                    result = [
                        json.loads(l)
                        for l in proc.stdout.splitlines()
                        if l.startswith("{")
                    ]
                    for b, timed, witness in zip(
                        widths, result[::2], result[1::2], strict=True
                    ):
                        if [timed[i] for i in ("B", "N", "K", "type")] != [
                            b,
                            n,
                            k,
                            12,
                        ] or not 0 < timed["us"] < math.inf:
                            raise ValueError("wrong native operation")
                        rows[b][mode] = {**witness, "us": timed["us"]}
                for b in widths:
                    reference = np.fromfile(tmp / f"stock_B{b}.bin", np.float32)
                    if (
                        reference.size != b * 64
                        or not np.isfinite(reference).all()
                        or not np.linalg.norm(reference)
                    ):
                        raise ValueError("invalid stock result")
                    for mode in modes:
                        path = tmp / f"{mode}_B{b}.bin"
                        y = np.fromfile(path, np.float32)
                        error = float(
                            np.linalg.norm(y - reference) / np.linalg.norm(reference)
                        )
                        if not math.isfinite(error):
                            raise ValueError("invalid grid output")
                        rows[b][mode]["relative_rms"] = error
                        path.unlink()
                save(
                    out / f"shape-{seed}-{n}-{k}.json",
                    {
                        "n": n,
                        "k": k,
                        "seed": seed,
                        "cells": rows,
                        "source": harness,
                        "tensor": t["name"],
                        "weight_copies": copies,
                    },
                )
    save(out / "plan.json", select(out))


def select(root):
    protocol = json.loads((root / "protocol.json").read_bytes())
    observations = {}
    for f in root.glob("shape-*.json"):
        s = json.loads(f.read_bytes())
        shape = s["n"], s["k"]
        for b, choices in s["cells"].items():
            observations.setdefault((*shape, int(b)), {})[s["seed"]] = choices
    if len(observations) != 5 * len(WIDTHS) or any(
        set(b) != set(SEEDS) for b in observations.values()
    ):
        raise ValueError("incomplete grid study")
    ranges = []
    for (n, k, b), blocks in sorted(observations.items()):
        train, holdout = (blocks[s] for s in SEEDS)
        mode = min(MODES, key=lambda m: train[m]["us"])
        if mode == "stock" or not all(
            rows[mode]["dispatches"] == rows[mode]["graphs"] > 0
            and rows[mode]["relative_rms"] <= 1e-4
            and rows[mode]["us"] * 1.10 < rows["stock"]["us"]
            for rows in (train, holdout)
        ):
            continue
        # No interpolation beyond the witnessed row count.
        ranges.append([n, k, b, b, MODES[mode]])
    return {
        "hardware": protocol["hardware"],
        "reference_cuda": protocol["reference_cuda"],
        "gguf": protocol["gguf"],
        "screened_backend": protocol["backend"],
        "ranges": ranges,
        "scope": "Experimental Q4_K row counts confirmed by a second seed. The system gate still applies.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("gguf", "build", "out"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    screen(a.gguf, a.build, a.out)
