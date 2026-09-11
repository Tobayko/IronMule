#!/usr/bin/env python3
"""Run the affected call path under Apple's shader validation, and prove it was on.

Three phases, because the environment has to be set before Metal initialises and because a
clean validator report is worthless unless the instrumentation can be shown to work.

  --capture   loads the model once, with no validation, and stores the packed weights,
              scales and biases of the target projections beside the real activations
              already captured, each with a checksum.
  --control   a child process that deliberately reads past the end of a buffer in a custom
              kernel. If the validator does not report that, it is not instrumenting
              custom kernels and no clean result may be claimed.
  --validate  a child process that runs the transcription and the specialised kernel on the
              stored real bytes and compares them against the library.

Each child is spawned with the diagnostic environment already in place, so nothing has
touched Metal by the time it is read. No timing is taken: validation changes execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DIAGNOSTIC_ENVIRONMENT = {
    "MTL_DEBUG_LAYER": "1",
    "MTL_SHADER_VALIDATION": "1",
    "MTL_SHADER_VALIDATION_REPORT_TO_STDERR": "1",
}
INPUTS = PROJECT_ROOT / ".friday-data" / "b53-inputs"
TARGETS = ("model_layers_0_self_attn_q_proj",     # width 4096
           "model_layers_0_self_attn_k_proj",     # width 2048
           "model_layers_0_mlp_gate_proj")        # width 15360

CONTROL_SOURCE = """
  uint i = thread_position_in_grid.x;
  // Deliberately one element past the end of `a`, which has four elements.
  out[i] = a[i + 4];
"""

VALIDATE_SOURCE = """
import json, sys
from pathlib import Path
sys.path.insert(0, %(root)r)
sys.path.insert(0, %(tools)r)
import mlx.core as mx
from b42_qmv_kernel import K3840, PORT, run

inputs = Path(%(inputs)r)
report = {}
for name in %(targets)r:
    weight = mx.load(str(inputs / ("weight_" + name + ".npy")))
    scales = mx.load(str(inputs / ("scales_" + name + ".npy")))
    biases = mx.load(str(inputs / ("biases_" + name + ".npy")))
    x = mx.load(str(inputs / ("activation_" + name + ".npy")))
    mx.eval(weight, scales, biases, x)
    out_features = int(weight.shape[0])
    reference = mx.quantized_matmul(x, weight, scales, biases, transpose=True,
                                    group_size=64, bits=4)
    again = mx.quantized_matmul(x, weight, scales, biases, transpose=True,
                                group_size=64, bits=4)
    ported = run(PORT, weight, scales, biases, x, out_features, 3840)
    special = run(K3840, weight, scales, biases, x, out_features, 3840)
    mx.eval(reference, again, ported, special)
    expected = bytes(memoryview(reference))
    report[name] = {
        "out_features": out_features,
        "library_against_library": bytes(memoryview(again)) == expected,
        "port_against_library": bytes(memoryview(ported)) == expected,
        "k3840_against_library": bytes(memoryview(special)) == expected,
    }
print("RESULT " + json.dumps(report, sort_keys=True))
"""

CONTROL_SCRIPT = """
import sys
sys.path.insert(0, %(root)r)
import mlx.core as mx
kernel = mx.fast.metal_kernel(name="b53_control_out_of_bounds", input_names=["a"],
                              output_names=["out"], source=%(source)r,
                              ensure_row_contiguous=True)
a = mx.zeros((4,), dtype=mx.float32)
out = kernel(inputs=[a], output_shapes=[(4,)], output_dtypes=[mx.float32],
             grid=(4, 1, 1), threadgroup=(4, 1, 1))[0]
mx.eval(out)
print("CONTROL_COMPLETED", out.tolist())
"""


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(model_id: str) -> dict:
    """Store the packed bytes of the target projections. No validation in this phase."""

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    manifest = json.loads((INPUTS / "manifest.json").read_text())
    by_file = {entry["file"]: entry for entry in manifest["activations"]}
    model, _ = load(model_id)
    stored = {}
    for name in TARGETS:
        entry = by_file[f"activation_{name}.npy"]
        # The capture registered paths relative to the language model, as it does.
        node = getattr(model, "language_model", model)
        for part in entry["module_path"].split("."):
            node = node[int(part)] if part.isdigit() else getattr(node, part)
        if not isinstance(node, nn.QuantizedLinear):
            raise SystemExit(f"{entry['module_path']} is not a quantised projection")
        stored[name] = {"module_path": entry["module_path"],
                        "out_features": int(node.weight.shape[0])}
        for label, array in (("weight", node.weight), ("scales", node.scales),
                             ("biases", node.biases)):
            target = INPUTS / f"{label}_{name}.npy"
            reused = target.exists()
            if not reused:
                mx.save(str(target), array)
            stored[name][label] = {"path": str(target), "sha256": _digest(target),
                                   "reused": reused}
    return stored


PROBE_SCRIPT = """
import sys
sys.path.insert(0, %(root)r)
import mlx.core as mx
source = "uint i = thread_position_in_grid.x; out[i] = a[i + %(offset)d];"
kernel = mx.fast.metal_kernel(name="b53_probe_%(offset)d", input_names=["a"],
                              output_names=["out"], source=source,
                              ensure_row_contiguous=True)
a = mx.zeros((4,), dtype=mx.float32)
out = kernel(inputs=[a], output_shapes=[(4,)], output_dtypes=[mx.float32],
             grid=(4, 1, 1), threadgroup=(4, 1, 1))[0]
mx.eval(out)
print("PROBE_COMPLETED", out.tolist())
"""

PROBE_CASES = (
    ("three_variables_small_overread", DIAGNOSTIC_ENVIRONMENT, 4),
    ("three_variables_far_overread", DIAGNOSTIC_ENVIRONMENT, 1 << 22),
    ("plus_global_memory_small_overread",
     {**DIAGNOSTIC_ENVIRONMENT, "MTL_SHADER_VALIDATION_GLOBAL_MEMORY": "1"}, 4),
    ("no_diagnostic_environment", {}, 4),
)


def probe(logs: Path) -> dict:
    """Can shader validation be shown to instrument a custom kernel at all?

    A read four elements past a four-element buffer stays inside the same page, so page
    protection cannot catch it; only buffer-length instrumentation can. A read four million
    elements past it leaves any plausible allocation, so a fault there shows page
    protection is working while saying nothing about instrumentation.
    """

    results = {}
    logs.mkdir(parents=True, exist_ok=True)
    for label, environment, offset in PROBE_CASES:
        completed = subprocess.run(
            [sys.executable, "-c", PROBE_SCRIPT % {"root": str(PROJECT_ROOT),
                                                   "offset": offset}],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
            env={**os.environ, **environment})
        path = logs / f"shader_validation_probe_{label}.txt"
        path.write_text(f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n"
                        f"{completed.stderr}", encoding="utf-8")
        results[label] = {
            "environment": environment,
            "read_offset_elements": offset,
            "returncode": completed.returncode,
            "completed": "PROBE_COMPLETED" in completed.stdout,
            "stderr": completed.stderr,
            "log": {"path": str(path), "sha256": _digest(path)},
            "mentions_shader_validation": "Shader Validation" in completed.stderr,
            "mentions_api_validation": "API Validation Enabled" in completed.stderr,
            "mentions_gpu_validation": "GPU Validation Enabled" in completed.stderr,
        }
    return results


def _child(script: str, label: str) -> dict:
    environment = {**os.environ, **DIAGNOSTIC_ENVIRONMENT}
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True,
                               text=True, env=environment, cwd=str(PROJECT_ROOT))
    return {"label": label, "returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr,
            "environment": DIAGNOSTIC_ENVIRONMENT}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", default="mlx-community/gemma-3-12b-it-4bit")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--probe", action="store_true",
                        help="ask whether shader validation instruments a custom kernel")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--logs", type=Path,
                        default=PROJECT_ROOT / ".friday-data" / "b53-evidence")
    args = parser.parse_args()

    if args.capture:
        stored = capture(args.model)
        print(json.dumps(stored, indent=2, sort_keys=True))
        return 0

    if args.out is None:
        parser.error("--out is required for a validating run")

    from b52_automatic_selection import write_once

    if args.probe:
        results = probe(args.logs)
        caught = {label: not row["completed"] or row["mentions_shader_validation"]
                  for label, row in results.items()}
        record = {
            "schema": "ironmule.b53_shader_validation_probe.v1",
            "experiment_id": args.out.stem,
            "agent": "claude",
            "kind": "diagnosis",
            "status": "measured",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "question": "does Apple's shader validation instrument a kernel built through "
                        "mx.fast.metal_kernel on this platform",
            "cases": {label: {k: v for k, v in row.items() if k != "stderr"}
                      for label, row in results.items()},
            "stderr_heads": {label: row["stderr"][:1200] for label, row in results.items()},
            "any_case_reported_shader_validation":
                any(row["mentions_shader_validation"] for row in results.values()),
            "small_overread_caught": caught["three_variables_small_overread"],
            "far_overread_caught": caught["three_variables_far_overread"],
            "instrumentation_proved":
                caught["three_variables_small_overread"]
                or caught["plus_global_memory_small_overread"],
            "reading": "a small overread stays inside the page, so only buffer-length "
                       "instrumentation can catch it. If it is not caught while a far "
                       "overread is, the platform is giving page protection and not "
                       "shader validation",
            "device": {"platform": platform.platform(), "machine": platform.machine()},
            "git_revision": subprocess.run(
                ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=False).stdout.strip(),
        }
        write_once(args.out, json.dumps(record, indent=2, sort_keys=True))
        print(json.dumps({label: {"returncode": row["returncode"],
                                  "completed": row["completed"],
                                  "shader_validation": row["mentions_shader_validation"]}
                          for label, row in results.items()}, indent=2, sort_keys=True))
        return 0

    control = _child(CONTROL_SCRIPT % {"root": str(PROJECT_ROOT),
                                       "source": CONTROL_SOURCE}, "out_of_bounds_control")
    validate = _child(VALIDATE_SOURCE % {"root": str(PROJECT_ROOT),
                                         "tools": str(PROJECT_ROOT / "tools"),
                                         "inputs": str(INPUTS),
                                         "targets": TARGETS}, "real_call_path")

    args.logs.mkdir(parents=True, exist_ok=True)
    log_files = {}
    for child in (control, validate):
        for stream in ("stdout", "stderr"):
            path = args.logs / f"shader_validation_{child['label']}_{stream}.txt"
            path.write_text(child[stream], encoding="utf-8")
            log_files[f"{child['label']}_{stream}"] = {"path": str(path),
                                                       "sha256": _digest(path),
                                                       "bytes": len(child[stream])}

    markers = ("Metal Shader Validation", "shader validation", "out of bounds",
               "Out of bounds", "GPU Software Fault", "validationErrorMessage")
    control_reported = any(marker in control["stderr"] for marker in markers) \
        or control["returncode"] != 0
    instrumented = control_reported

    result = {}
    for line in validate["stdout"].splitlines():
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])

    all_equal = bool(result) and all(all(v for k, v in row.items() if k != "out_features")
                                     for row in result.values())

    record = {
        "schema": "ironmule.b53_shader_validation.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "diagnosis",
        "status": "measured",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "environment_set_before_metal": DIAGNOSTIC_ENVIRONMENT,
        "instrumentation_proved": instrumented,
        "instrumentation_evidence": {
            "control_returncode": control["returncode"],
            "control_stderr_head": control["stderr"][:2000],
            "control_stdout_head": control["stdout"][:500],
        },
        "real_call_path": {
            "returncode": validate["returncode"],
            "targets": list(TARGETS),
            "comparisons": result,
            "all_equal": all_equal,
            "stderr_head": validate["stderr"][:4000],
        },
        "logs": log_files,
        "verdict": ("NO DIFFERENCE UNDER VALIDATION" if instrumented and all_equal
                    else "INSTRUMENTATION NOT PROVED" if not instrumented
                    else "DIFFERENCE UNDER VALIDATION"),
        "caveats": [
            "a clean validator does not prove the absence of every fault",
            "no timing was taken: validation changes execution and timing",
            "the comparison arm is not assumed correct; a difference names a "
            "disagreement, not a culprit",
        ],
        "device": {"platform": platform.platform(), "machine": platform.machine()},
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip(),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({"instrumentation_proved": instrumented,
                      "verdict": record["verdict"],
                      "comparisons": result}, indent=2, sort_keys=True))
    return 0 if instrumented else 1


if __name__ == "__main__":
    raise SystemExit(main())
