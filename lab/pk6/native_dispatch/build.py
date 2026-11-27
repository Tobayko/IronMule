"""Build one opt-in prompt plugin on its target and bind the output to native software and the unchanged GGUF."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from lab.pk6.mixed_progress.trial import board
from lab.pk6.replay_http import checksum
from lab.pk6.warm_repeat import GGUF, save


def packed_routes(plan):
    from lab.pk6.native_dispatch.packed_screen import WIDTHS

    memory = plan["memory"]
    limit = (
        plan["hardware"]["memory_mib"]
        - memory["measured_serving_peak_mib"]
        - memory["reserve_mib"]
    ) // 1024
    gib = memory["cache_gib"]
    if (
        memory["hardware"] != plan["hardware"]
        or type(gib) is not int
        or not 1 <= gib <= limit
        or memory["reserve_mib"] < 6144
    ):
        raise ValueError("invalid packed cache reservation")
    routes = []
    for n, k, lo, hi in plan["ranges"]:
        if (
            any(type(x) is not int for x in (n, k, lo, hi))
            or not all(0 < x <= 25600 and x % 256 == 0 for x in (n, k))
            or lo not in WIDTHS
            or hi not in WIDTHS
            or hi < lo
        ):
            raise ValueError("invalid measured packed shape")
        routes.extend("{" + f"{n},{k},{b}" + "}" for b in WIDTHS if lo <= b <= hi)
    if not routes:
        raise ValueError("empty measured packed route set")
    return (
        f"constexpr unsigned cache_gib = {gib};\n"
        "struct PackedRoute { int n,k,b; };\n"
        "constexpr PackedRoute measured_routes[] = {" + ",".join(routes) + "};\n"
    )


def compile_plugin(kind, output, model, policy=None):
    hardware = board()
    if checksum(model) != GGUF:
        raise ValueError("unrecognized GGUF artifact")
    here = Path(__file__).resolve().parent
    cuda = Path(os.environ["LLAMA_CPP_LIB_PATH"]) / "libggml-cuda.so"
    native = checksum(cuda)
    args = [
        "nvcc",
        "-O3",
        "-std=c++17",
        "-arch=sm_89",
        "--cudart",
        "shared",
        "-shared",
        "-Xcompiler=-fPIC",
    ]
    inputs = [here / "prompt_fp8.cu", here / "prompt_codec.cuh"]
    packed_header = None
    pool_cells = None
    if kind == "routed":
        if policy is None:
            raise ValueError("routed kernels require a fitted target policy")
        fit = json.loads((policy / "fit.json").read_bytes())
        if (
            fit["target"]["hardware"] != hardware
            or fit["target"]["gguf"] != GGUF
            or fit["target"]["ggml_cuda"] != native
            or checksum(policy / "target_rows.h") != fit["header_sha256"]
            or checksum(policy / "pilot.npz") != fit["weights_sha256"]
            or not fit["candidate_intervals"]
        ):
            raise ValueError("foreign or empty routing policy")
        args.extend(["-I" + str(policy.resolve()), str(here / "routed.cu")])
        inputs.extend(
            [
                here / "routed.cu",
                policy / "target_rows.h",
                policy / "pilot.npz",
                policy / "fit.json",
            ]
        )
    elif kind == "cached-half":
        inputs = [
            here / "half_prompt.cu",
            here / "weight_cache.cuh",
            here / "block_values.cuh",
        ]
        args.append(str(inputs[0]))
    elif kind in ("marlin", "packed-marlin"):
        include = here.parents[2] / "native/kernels"
        args.extend(
            [
                "--expt-relaxed-constexpr",
                "-I" + str(include),
                str(here / "marlin_prompt.cu"),
            ]
        )
        inputs = [
            here / "marlin_prompt.cu",
            *sorted(
                p
                for p in (include / "mq1").rglob("*")
                if p.suffix in (".h", ".cuh", ".hpp")
            ),
        ]
        if kind == "packed-marlin":
            if policy is None:
                raise ValueError("missing matched native-graph plan")
            plan = json.loads((policy / "plan.json").read_bytes())
            if (
                plan["hardware"] != hardware
                or plan["backend"] != native
                or plan["gguf"] != GGUF
            ):
                raise ValueError("foreign packed kernel plan")
            packed_header = packed_routes(plan)
            pool_cells = plan["memory"]["pool_cells"]
            inputs.append(policy / "plan.json")
            args.extend(["-DIM_MARLIN_ROUTED", "-I" + str(output.resolve())])
    else:
        if kind in ("int8", "cached-int8"):
            args.append("-DIM_PROMPT_INT8")
        if kind == "cached-int8":
            args.append("-DIM_PROMPT_CACHE")
            inputs.append(here / "weight_cache.cuh")
        args.append(str(inputs[0]))
    output.mkdir(parents=True, exist_ok=False)
    lib = output / (kind + ".so")
    if packed_header:
        header = output / "packed_routes.h"
        header.write_text(packed_header)
        inputs.append(header)
    if kind not in ("marlin", "packed-marlin"):
        args.extend(["-lcublasLt", "-lcublas"])
    started = time.monotonic()
    subprocess.run([*args, "-o", str(lib)], check=True)
    save(
        output / "build.json",
        {
            "hardware": hardware,
            "gguf": GGUF,
            "ggml_cuda": native,
            "kind": kind,
            "pool_cells": pool_cells,
            "library_sha256": checksum(lib),
            "sources": {p.name: checksum(p) for p in inputs},
            "seconds": time.monotonic() - started,
            "compiler": subprocess.check_output(
                ["nvcc", "--version"], text=True
            ).splitlines()[-1],
            "scope": "Experimental plugin. Separate model and product verification required before use.",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "kind",
        choices=[
            "fp8",
            "routed",
            "int8",
            "marlin",
            "packed-marlin",
            "cached-int8",
            "cached-half",
        ],
    )
    p.add_argument("output", type=Path)
    p.add_argument("gguf", type=Path)
    p.add_argument("--policy", type=Path)
    a = p.parse_args()
    compile_plugin(a.kind, a.output, a.gguf, a.policy)
