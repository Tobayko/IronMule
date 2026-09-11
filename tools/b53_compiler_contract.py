#!/usr/bin/env python3
"""What the Metal compiler is actually told, and whether it changes the answer.

Three questions, in order.

1. **The contract as installed.** The wheel ships its own headers, so the default is
   readable rather than assumed: `mlx/backend/common/metal_kernel.h` declares
   `MathMode {Safe, Relaxed, Fast}` with `CompileOptions.math_mode = Safe`, and the
   binding's docstring says the same. This records the intended, the passed and the
   effective setting separately, and checks that the registry's digest covers exactly the
   specification that is handed to MLX.

2. **Can the construction tell the modes apart at all?** A tiny kernel whose result
   differs between safe and fast. Without that control, a separation test that finds
   nothing has no power and means nothing.

3. **The separation itself.** The same stored real bytes through the same source under
   three content-bound names: default, explicit safe, and fast as a diagnostic control.
   The library reference is not touched. A limited set of constructed finite inputs is
   added, marked synthetic: heavy cancellation, and values sitting on bfloat16 rounding
   boundaries.

`fast` is never proposed as a runtime setting. If explicit safe equals the default, that
removes one candidate for B53 and repairs nothing.
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

import mlx.core as mx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import b42_qmv_kernel as b42  # noqa: E402
from b52_automatic_selection import write_once  # noqa: E402
from ironmule import kernel_registry as kr  # noqa: E402

INPUTS = PROJECT_ROOT / ".friday-data" / "b53-inputs"
TARGETS = ("model_layers_0_self_attn_k_proj", "model_layers_0_self_attn_q_proj",
           "model_layers_0_mlp_gate_proj")
MODES = (("default", None), ("explicit_safe", {"math_mode": "safe"}),
         ("fast_control", {"math_mode": "fast"}))

# A long chain of multiply-adds is what actually separates the modes here: fast maths may
# contract and reassociate it. `exp(-inf)`, the example in the binding's docstring, turned
# out not to separate them on this build, and a control that cannot separate is worthless.
POWER_SOURCE = """
  uint i = thread_position_in_grid.x;
  float acc = 0.0f;
  float v = a[i];
  for (int k = 0; k < 64; k++) {
    float t = v * 1.0000001f + float(k) * 1e-7f;
    acc = acc + t * t - t * t * 0.9999999f;
  }
  out[i] = acc;
"""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def contract() -> dict:
    """Intended, passed and effective, kept apart, plus the registry cross-check."""

    header = (Path(mx.__file__).parent / "include" / "mlx" / "backend" / "common"
              / "metal_kernel.h")
    header_text = header.read_text()
    passed: dict[str, dict] = {}
    real = mx.fast.metal_kernel

    def record(**kwargs):
        passed[kwargs["name"]] = {
            "compile_options": kwargs.get("compile_options"),
            "header_sha256": _sha((kwargs.get("header") or "").encode()),
            "source_sha256": _sha(kwargs["source"].encode()),
            "input_names": list(kwargs["input_names"]),
            "output_names": list(kwargs["output_names"]),
            "ensure_row_contiguous": kwargs.get("ensure_row_contiguous"),
            "atomic_outputs": kwargs.get("atomic_outputs"),
        }
        return real(**kwargs)

    try:
        mx.fast.metal_kernel = record
        b42._kernel("qmv_port", None)
        b42._kernel("qmv_k3840", 3840)
    finally:
        mx.fast.metal_kernel = real

    registry = kr.registered()
    digest_matches = {}
    for name, spec in registry.items():
        if name not in passed:
            continue
        handed = passed[name]
        digest_matches[name] = {
            "identifier_reproduces": kr.identifier(spec) == name,
            "source_matches": _sha(spec["source"].encode()) == handed["source_sha256"],
            "header_matches": _sha(spec["header"].encode()) == handed["header_sha256"],
            "compile_options_match":
                (spec["compile_options"] or None) == handed["compile_options"],
            "input_names_match": spec["input_names"] == handed["input_names"],
            "output_names_match": spec["output_names"] == handed["output_names"],
        }

    return {
        "installed_header": {
            "path": str(header.resolve().relative_to(Path(mx.__file__).parents[3])),
            "sha256": _sha(header.read_bytes()),
            "declares_default_safe": "MathMode math_mode = MathMode::Safe;" in header_text,
            "modes": ["Safe = 0", "Relaxed = 1", "Fast = 2"],
        },
        "binding_docstring_says_default_safe":
            'Default: ``"safe"``' in (mx.fast.metal_kernel.__doc__ or ""),
        "intended_by_the_modules": {"b42_qmv_kernel": b42.COMPILE_OPTIONS},
        "passed_to_mlx": {name: row["compile_options"] for name, row in passed.items()},
        "effective": "safe for every kernel here: nothing is passed and the installed "
                     "default is Safe",
        "the_three_agree": all(row["compile_options"] is None for row in passed.values()),
        "registry_digest_against_what_was_passed": digest_matches,
        "registry_cross_check_clean": all(all(row.values())
                                          for row in digest_matches.values()),
        "other_inputs_to_the_translation": {
            "header": "one shared header per module, hashed into the identifier",
            "macros": "none are set by these modules",
            "language_version": "not selected by MLX's Python surface; the wheel decides",
            "fma_and_math_functions": "governed by the math mode, which is Safe here; "
                                      "MLX drives MTLCompileOptions.mathMode, the "
                                      "setMathMode: selector is present in libmlx.dylib",
        },
    }


def power_control() -> dict:
    """Prove the two modes can produce different bytes before trusting a null result."""

    values = mx.array([0.0, 1.0, 2.0, float("inf")], dtype=mx.float32)
    mx.eval(values)
    results = {}
    for label, options in MODES:
        kernel = kr.build(f"b53_power_{label}", source=POWER_SOURCE, input_names=["a"],
                          output_names=["out"], compile_options=options)
        out = kernel(inputs=[values], output_shapes=[(4,)], output_dtypes=[mx.float32],
                     grid=(4, 1, 1), threadgroup=(4, 1, 1))[0]
        mx.eval(out)
        results[label] = {"bytes_sha256": _sha(bytes(memoryview(out))),
                          "values": [float(v) for v in out.tolist()]}
    distinguishes = results["fast_control"]["bytes_sha256"] != results["default"]["bytes_sha256"]
    return {"per_mode": results,
            "safe_and_fast_differ": distinguishes,
            "reading": "if these do not differ, this small control cannot tell the modes "
                       "apart. The separation below then has to carry its own power, "
                       "which it does when its own fast arm differs"}


def _stored(name: str):
    weight = mx.load(str(INPUTS / f"weight_{name}.npy"))
    scales = mx.load(str(INPUTS / f"scales_{name}.npy"))
    biases = mx.load(str(INPUTS / f"biases_{name}.npy"))
    x = mx.load(str(INPUTS / f"activation_{name}.npy"))
    mx.eval(weight, scales, biases, x)
    return weight, scales, biases, x


def _capture_source(base: str, fixed_k: int | None) -> tuple[str, str]:
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


def _modal_kernels():
    """The same two sources under three modes, each with its own content-bound name."""

    built = {}
    for label, options in MODES:
        for base, fixed_k in (("qmv_port", None), ("qmv_k3840", 3840)):
            source, header = _capture_source(base, fixed_k)
            built[(label, base)] = kr.build(
                f"{base}_{label}", source=source, header=header,
                input_names=["w", "scales", "biases", "x", "shape"],
                output_names=["out"], ensure_row_contiguous=True,
                compile_options=options, template={"fixed_k": fixed_k})
    return built


def _run(kernel, w, scales, biases, x, out_features):
    groups = (out_features + 7) // 8
    return kernel(inputs=[w, scales, biases, x, b42.shape_array(3840, out_features)],
                  output_shapes=[(1, out_features)], output_dtypes=[x.dtype],
                  grid=(b42.SIMD_SIZE, b42.NUM_SIMDGROUPS * groups, 1),
                  threadgroup=(b42.SIMD_SIZE, b42.NUM_SIMDGROUPS, 1))[0]


def separation(built) -> dict:
    """Three modes over the stored real projections, against an unmodified library."""

    findings = {}
    for name in TARGETS:
        weight, scales, biases, x = _stored(name)
        out_features = int(weight.shape[0])
        reference = mx.quantized_matmul(x, weight, scales, biases, transpose=True,
                                        group_size=64, bits=4)
        mx.eval(reference)
        expected = bytes(memoryview(reference))
        row = {"out_features": out_features,
               "input_digests": {"weight": _sha(bytes(memoryview(weight))),
                                 "scales": _sha(bytes(memoryview(scales))),
                                 "biases": _sha(bytes(memoryview(biases))),
                                 "x": _sha(bytes(memoryview(x)))},
               "library_sha256": _sha(expected)}
        for (label, base), kernel in built.items():
            out = _run(kernel, weight, scales, biases, x, out_features)
            mx.eval(out)
            produced = bytes(memoryview(out))
            row[f"{base}_{label}"] = {
                "equals_library": produced == expected,
                "sha256": _sha(produced),
            }
        findings[name] = row
    return findings


def constructed(built, cases: int) -> dict:
    """Synthetic diagnosis only: cancellation and bfloat16 rounding boundaries."""

    findings = {}
    key = mx.random.key(20260910)
    for out_features in (2048, 4096, 15360):
        rows = []
        for index in range(cases):
            key, weight_key, vector_key = mx.random.split(key, 3)
            dense = mx.random.normal((out_features, 3840), key=weight_key).astype(mx.bfloat16)
            weight, scales, biases = mx.quantize(dense, group_size=64, bits=4)
            if index % 2 == 0:
                # cancellation: a vector with alternating signs against itself
                base = mx.random.normal((1, 3840), key=vector_key)
                x = (base - mx.mean(base)).astype(mx.bfloat16)
            else:
                # values a hair away from a bfloat16 rounding boundary
                base = mx.random.normal((1, 3840), key=vector_key)
                x = (base * mx.array(1.0 + 2.0 ** -9)).astype(mx.bfloat16)
            mx.eval(weight, scales, biases, x)
            reference = mx.quantized_matmul(x, weight, scales, biases, transpose=True,
                                            group_size=64, bits=4)
            mx.eval(reference)
            expected = bytes(memoryview(reference))
            disagreements = []
            for (label, base_name), kernel in built.items():
                out = _run(kernel, weight, scales, biases, x, out_features)
                mx.eval(out)
                if bytes(memoryview(out)) != expected:
                    disagreements.append(f"{base_name}_{label}")
            if disagreements:
                rows.append({"case": index, "kind": "cancellation" if index % 2 == 0
                             else "rounding_boundary", "disagreeing": disagreements})
        findings[str(out_features)] = {"cases": cases, "disagreements": rows}
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--constructed-cases", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    the_contract = contract()
    power = power_control()
    built = _modal_kernels()
    real = separation(built)
    synthetic = constructed(built, args.constructed_cases)

    fast_differs_on_real = any(
        row[f"{base}_fast_control"]["sha256"] != row[f"{base}_default"]["sha256"]
        for row in real.values() for base in ("qmv_port", "qmv_k3840")
    )
    modes_agree_on_real = all(
        row[f"{base}_{label}"]["sha256"] == row[f"{base}_default"]["sha256"]
        for row in real.values() for label, _ in MODES for base in ("qmv_port", "qmv_k3840")
    )
    safe_equals_default = all(
        row[f"{base}_explicit_safe"]["sha256"] == row[f"{base}_default"]["sha256"]
        for row in real.values() for base in ("qmv_port", "qmv_k3840")
    )
    all_equal_library = all(row[f"{base}_{label}"]["equals_library"]
                            for row in real.values()
                            for label, _ in MODES if label != "fast_control"
                            for base in ("qmv_port", "qmv_k3840"))
    synthetic_clean = not any(entry["disagreements"] for entry in synthetic.values())

    record = {
        "schema": "ironmule.b53_compiler_contract.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "diagnosis",
        "status": "measured",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "preregistration": "B53_compiler_contract_preregistration_20260910",
        "contract": the_contract,
        "power_control": power,
        "real_projections": real,
        "constructed_inputs": {
            "synthetic": True,
            "not_model_evidence": "these inputs are constructed to be hard, not to be "
                                  "representative; they say nothing about model output or "
                                  "about performance",
            "per_width": synthetic,
            "clean": synthetic_clean,
        },
        "explicit_safe_equals_default": safe_equals_default,
        "every_mode_agrees_on_real_data": modes_agree_on_real,
        "safe_modes_match_the_library": all_equal_library,
        "power": {
            "small_control_separates_modes": power["safe_and_fast_differ"],
            "separation_own_fast_arm_separates_modes": fast_differs_on_real,
            "has_power": power["safe_and_fast_differ"] or fast_differs_on_real,
            "reading": "the test can tell the modes apart if either control separates "
                       "them; the fast arm of the separation itself is the stronger one, "
                       "because it runs the very kernels under question",
        },
        "verdict": (
            "NO POWER" if not (power["safe_and_fast_differ"] or fast_differs_on_real)
            else "MODE DOES NOT SEPARATE SAFE FROM DEFAULT" if safe_equals_default
            else "MODE SEPARATES"),
        "what_this_settles": (
            "whether the math mode explains the observed disagreement. It does not repair "
            "B53 either way"
        ),
        "mlx_version": mx.__version__,
        "device": {"platform": platform.platform(), "machine": platform.machine()},
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip(),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({"verdict": record["verdict"],
                      "small_control_distinguishes": power["safe_and_fast_differ"],
                      "fast_arm_distinguishes": fast_differs_on_real,
                      "explicit_safe_equals_default": safe_equals_default,
                      "safe_modes_match_the_library": all_equal_library,
                      "constructed_clean": synthetic_clean,
                      "registry_cross_check_clean":
                          the_contract["registry_cross_check_clean"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
