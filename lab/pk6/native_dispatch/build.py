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
    elif kind == "marlin":
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
    else:
        if kind in ("int8", "cached-int8"):
            args.append("-DIM_PROMPT_INT8")
        if kind == "cached-int8":
            args.append("-DIM_PROMPT_CACHE")
            inputs.append(here / "weight_cache.cuh")
        args.append(str(inputs[0]))
    output.mkdir(parents=True, exist_ok=False)
    lib = output / (kind + ".so")
    if kind != "marlin":
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
        choices=["fp8", "routed", "int8", "marlin", "cached-int8", "cached-half"],
    )
    p.add_argument("output", type=Path)
    p.add_argument("gguf", type=Path)
    p.add_argument("--policy", type=Path)
    a = p.parse_args()
    compile_plugin(a.kind, a.output, a.gguf, a.policy)
