"""Time native GGML matmuls on disjoint target buffers, with device synchronization and a dispatch witness."""

import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile

import numpy as np

from ironmule.kernel_gen import q4k_tensors
from lab.pk6.mixed_progress.trial import board
from lab.pk6.native_dispatch.prepare import substitute
from lab.pk6.replay_http import checksum
from lab.pk6.warm_repeat import GGUF, save

WIDTHS = (64, 128, 143, 256, 448, 512, 1024, 2048)
ARMS = ("stock", "cublas", "fp8")


def build(directory, *, native_grid=False):
    tree = Path(os.environ["IRONMULE_GGML_BINDINGS"]) / "vendor/llama.cpp"
    library = Path(os.environ["LLAMA_CPP_LIB_PATH"])
    source = Path("lab/kg1/ref_ggml.c").read_text()
    source = substitute(
        source,
        "#include <stdio.h>",
        "#include <stdio.h>\n#include <dlfcn.h>\n#include <stdint.h>",
    )
    source = substitute(
        source,
        "        const double t0 = now();",
        """
        ggml_backend_synchronize(be);
        const char *library = getenv("IRONMULE_KG_LIB");
        void *handle = library && library[0] ? dlopen(library, RTLD_NOW) : NULL;
        uint64_t (*dispatches)(void) = handle ? dlsym(handle, "im_prompt_dispatches") : NULL;
        uint64_t before = dispatches ? dispatches() : 0;
        const double t0 = now();""",
    )
    source = substitute(
        source,
        "        const double us =",
        "        ggml_backend_synchronize(be);\n        const double us =",
    )
    # Keep evenly spread channels from every row to compare matched native outputs without GB of temporary files.
    source = substitute(
        source,
        "fwrite(y, sizeof(float), (size_t)N * B, f);",
        "for (int b=0; b<B; ++b) for (int j=0; j<64; ++j) "
        "fwrite(y + (size_t)b*N + (size_t)j*(N-1)/63, sizeof(float), 1, f);",
    )
    source = substitute(
        source,
        "        fflush(stdout);",
        """
        printf("{\\"dispatches\\":%llu,\\"graphs\\":%d}\\n",
               (unsigned long long)(dispatches ? dispatches()-before : 0), REPS*C);
        if (handle) dlclose(handle);
        fflush(stdout);""",
    )
    if native_grid:
        source = substitute(source, 'getenv("IRONMULE_KG_LIB")', 'getenv("IRONMULE_GRID_BACKEND")')
        source = substitute(source, '"im_prompt_dispatches"', '"im_mmq_tile_calls"')
    path = directory / "operator.c"
    path.write_text(source)
    binary = directory / "operator"
    subprocess.run(
        [
            "gcc",
            "-O2",
            str(path),
            "-o",
            str(binary),
            "-I" + str(tree / "ggml/include"),
            "-L" + str(library),
            "-Wl,-rpath," + str(library),
            "-lggml",
            "-lggml-base",
            "-lggml-cuda",
            "-ldl",
        ],
        check=True,
    )
    return binary, checksum(path)


def run(gguf, plugin, out):
    if checksum(gguf) != GGUF:
        raise ValueError("GGUF weights changed")
    out.mkdir(parents=True, exist_ok=False)
    cuda = Path(os.environ["LLAMA_CPP_LIB_PATH"]) / "libggml-cuda.so"
    spec = {
        "hardware": board(),
        "gguf": GGUF,
        "plugin": checksum(plugin),
        "ggml_cuda": checksum(cuda),
        "widths": WIDTHS,
        "arms": ARMS,
        "seeds": [31993, 19001],
        "repeats": 6,
    }
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        tmp = Path(tmp)
        binary, spec["harness_sha256"] = build(tmp)
        save(out / "protocol.json", spec)
        for seed in spec["seeds"]:
            rng = random.Random(seed)
            tensors = [
                (number, k, n, entry)
                for kind, number in (("Q4_K", 12), ("Q6_K", 14))
                for (k, n), entry in q4k_tensors(str(gguf), kind=kind).items()
                if n <= 65536
            ]
            rng.shuffle(tensors)
            for number, k, n, entry in tensors:
                weights = tmp / "weights"
                entry["data"].tofile(weights)
                x = tmp / "x"
                np.random.default_rng(seed + k + n).standard_normal(
                    (max(WIDTHS), k), dtype=np.float32
                ).tofile(x)
                copies = min(
                    48, max(2, math.ceil((128 << 20) / weights.stat().st_size))
                )
                arms = list(ARMS)
                rng.shuffle(arms)
                widths = list(WIDTHS)
                rng.shuffle(widths)
                cells = {}
                for arm in arms:
                    env = {
                        **os.environ,
                        "GGML_CUDA_DISABLE_GRAPHS": "1",
                        "REF_TYPE": str(number),
                        "IRONMULE_KG_LIB": str(plugin.resolve())
                        if arm == "fp8"
                        else "",
                        "IRONMULE_LAB_FP8": "32",
                        "IRONMULE_CUBLAS_MIN_ROWS": "32" if arm == "cublas" else "0",
                    }
                    cmd = [
                        binary,
                        weights,
                        str(n),
                        str(k),
                        x,
                        tmp / arm,
                        str(copies),
                        str(spec["repeats"]),
                        *map(str, widths),
                    ]
                    proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
                    if proc.returncode:
                        raise RuntimeError(proc.stderr[-1600:])
                    rows = [
                        json.loads(line)
                        for line in proc.stdout.splitlines()
                        if line.startswith("{")
                    ]
                    if len(rows) != 2 * len(widths):
                        raise ValueError("incomplete operator measurements")
                    for b, metric, witness in zip(
                        widths, rows[::2], rows[1::2], strict=True
                    ):
                        if [metric[x] for x in ("B", "N", "K", "type")] != [
                            b,
                            n,
                            k,
                            number,
                        ]:
                            raise ValueError("operator shape changed")
                        if arm == "fp8" and witness["dispatches"] != witness["graphs"]:
                            raise ValueError(
                                f"partial or missing FP8 dispatch at {(k, n, b)}: {witness}"
                            )
                        cells.setdefault(b, {})[arm] = {**witness, "us": metric["us"]}
                for b in widths:
                    ref = np.fromfile(tmp / f"stock_B{b}.bin", np.float32)
                    if (
                        ref.size != 64 * b
                        or not np.isfinite(ref).all()
                        or not np.linalg.norm(ref)
                    ):
                        raise ValueError("invalid native reference")
                    for arm in arms:
                        y = np.fromfile(tmp / f"{arm}_B{b}.bin", np.float32)
                        err = float(np.linalg.norm(y - ref) / np.linalg.norm(ref))
                        if not math.isfinite(err):
                            raise ValueError("invalid native candidate")
                        cells[b][arm]["relative_rms_to_stock"] = err
                        (tmp / f"{arm}_B{b}.bin").unlink()
                save(
                    out / f"shape-{seed}-{number}-{k}-{n}.json",
                    {
                        "seed": seed,
                        "type": number,
                        "n": n,
                        "k": k,
                        "name": entry["name"],
                        "matrix_count": entry["count"],
                        "copies": copies,
                        "cells": cells,
                    },
                )
    save(out / "completed.json", spec)


if __name__ == "__main__":
    import fcntl

    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("gguf", "lib", "out"):
        p.add_argument("--" + arg, type=Path, required=True)
    args = p.parse_args()
    with Path(
        os.environ.get("IRONMULE_LAB_GPU_LOCK", "/work/native-dispatch.lock")
    ).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run(args.gguf, args.lib, args.out)
