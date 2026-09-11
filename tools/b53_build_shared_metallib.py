#!/usr/bin/env python3
"""Compile one shared `.metallib` from the unchanged B53 specialised kernel source.

The kernel body and its helper header are taken byte for byte from
`tools/b42_qmv_kernel.py`, exactly as that module hands them to MLX, by watching the call
rather than by editing the module. What this file adds is the function signature, because
MLX generates one at runtime and an offline compile needs it written out. The signature
binds the same arrays in the same order and declares the same three attribute parameters
the body reads; it contains no arithmetic.

The safe contract is kept on purpose. The offline compiler defaults to fast maths
(`__FAST_MATH__ 1`), while MLX compiles custom kernels with `MathMode::Safe`, so this
passes `-fmetal-math-mode=safe` and records that it did. Everything else about the
translation is recorded rather than assumed: compiler build, SDK, language version,
options, and the checksum of every intermediate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx  # noqa: E402

import b42_qmv_kernel as b42  # noqa: E402
from b52_automatic_selection import write_once  # noqa: E402

OUT_DIR = PROJECT_ROOT / ".friday-data" / "b53-metal"
FUNCTION = "b53_qmv_k3840"
MATH_MODE = "safe"

# Only a signature: the same arrays in the same order, and the three attributes the body
# reads. MLX writes an equivalent one at runtime; this one is written out so an offline
# compiler can see it.
WRAPPER = """
[[kernel]] void {function}(
    device const uint32_t* w [[buffer(0)]],
    device const T* scales [[buffer(1)]],
    device const T* biases [[buffer(2)]],
    device const T* x [[buffer(3)]],
    device const int* shape [[buffer(4)]],
    device T* out [[buffer(5)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {{
{body}
}}
"""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _capture(base: str, fixed_k: int | None) -> tuple[str, str]:
    """The exact source and header b42 hands to MLX, without touching that module."""

    seen = {}
    real = mx.fast.metal_kernel

    def record(**kwargs):
        seen["source"] = kwargs["source"]
        seen["header"] = kwargs.get("header") or ""
        return object()

    try:
        mx.fast.metal_kernel = record
        b42._kernel(base, fixed_k)
    finally:
        mx.fast.metal_kernel = real
    return seen["source"], seen["header"]


def _run(command: list[str]) -> dict:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {"command": " ".join(command), "returncode": completed.returncode,
            "stdout": completed.stdout.strip()[:2000],
            "stderr": completed.stderr.strip()[:2000]}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    body, header = _capture("qmv_k3840", 3840)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metal_source = ("#include <metal_stdlib>\nusing namespace metal;\n"
                    + header
                    + WRAPPER.format(function=FUNCTION, body=body))
    metal_path = OUT_DIR / "b53_shared.metal"
    air_path = OUT_DIR / "b53_shared.air"
    lib_path = OUT_DIR / "b53_shared.metallib"
    for path in (metal_path, air_path, lib_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite an existing artefact: {path}")
    metal_path.write_text(metal_source, encoding="utf-8")

    compile_flags = ["-std=metal4.0", f"-fmetal-math-mode={MATH_MODE}", "-c"]
    compile_step = _run(["xcrun", "-sdk", "macosx", "metal", *compile_flags,
                         str(metal_path), "-o", str(air_path)])
    if compile_step["returncode"] != 0:
        compile_flags = [f"-fmetal-math-mode={MATH_MODE}", "-c"]
        compile_step = _run(["xcrun", "-sdk", "macosx", "metal", *compile_flags,
                             str(metal_path), "-o", str(air_path)])
    link_step = _run(["xcrun", "-sdk", "macosx", "metallib", str(air_path),
                      "-o", str(lib_path)])
    built = compile_step["returncode"] == 0 and link_step["returncode"] == 0

    macros = subprocess.run(["xcrun", "-sdk", "macosx", "metal",
                             f"-fmetal-math-mode={MATH_MODE}", "-x", "metal", "-E", "-dM",
                             "/dev/null"], capture_output=True, text=True, check=False)
    selected = {line.split()[1]: line.split()[2] if len(line.split()) > 2 else ""
                for line in macros.stdout.splitlines()
                if line.startswith("#define") and any(
                    token in line for token in ("__METAL_VERSION__", "__FAST_MATH__",
                                                "__METAL_FAST_MATH__",
                                                "__METAL_PRECISE_MATH__"))}

    record = {
        "schema": "ironmule.b53_shared_metallib.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "build",
        "status": "built" if built else "failed",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "body_from": "tools/b42_qmv_kernel.py, captured as handed to MLX, unchanged",
            "body_sha256": _sha(body.encode()),
            "header_sha256": _sha(header.encode()),
            "wrapper_added_here": "a signature only: the same arrays in the same order "
                                  "plus the three attribute parameters the body reads. No "
                                  "arithmetic, no type change",
            "function": FUNCTION,
            "metal_file": {"path": str(metal_path.relative_to(PROJECT_ROOT)),
                           "sha256": _sha(metal_path.read_bytes()),
                           "bytes": metal_path.stat().st_size},
        },
        "translation": {
            "compiler": _run(["xcrun", "-sdk", "macosx", "metal", "--version"])["stdout"],
            "linker": _run(["xcrun", "-sdk", "macosx", "metallib",
                            "--version"])["stdout"],
            "sdk_version": _run(["xcrun", "-sdk", "macosx",
                                 "--show-sdk-version"])["stdout"],
            "flags": compile_flags,
            "math_mode": MATH_MODE,
            "why_math_mode_is_passed": "the offline compiler defaults to fast maths, MLX "
                                       "compiles custom kernels safe. Passing it keeps the "
                                       "contract the same on both sides",
            "predefined_macros": selected,
            "compile_step": compile_step,
            "link_step": link_step,
        },
        "artefacts": {
            "air": {"path": str(air_path.relative_to(PROJECT_ROOT)),
                    "sha256": _sha(air_path.read_bytes()) if air_path.exists() else None},
            "metallib": {"path": str(lib_path.relative_to(PROJECT_ROOT)),
                         "sha256": _sha(lib_path.read_bytes()) if lib_path.exists()
                         else None,
                         "bytes": lib_path.stat().st_size if lib_path.exists() else None},
        },
        "caveat": "one .metallib is not a guarantee of identical final GPU code: the "
                  "driver still specialises when a pipeline is created, and MLX's own "
                  "runtime-compiled kernel went through a different front end entirely",
        "device": {"platform": platform.platform(), "machine": platform.machine()},
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip(),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({"built": built, "function": FUNCTION,
                      "metallib": record["artefacts"]["metallib"],
                      "flags": compile_flags,
                      "compile_stderr": compile_step["stderr"][:400],
                      "link_stderr": link_step["stderr"][:400]}, indent=2, sort_keys=True))
    return 0 if built else 1


if __name__ == "__main__":
    raise SystemExit(main())
