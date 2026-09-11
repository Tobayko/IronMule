#!/usr/bin/env python3
"""What makes MLX's own quantised matmul go wrong inside a process?

The reproduction showed the library arm sitting percent-level away from the float64 value
while both custom kernels tracked it. The library is stable within a run, so something
earlier in the process changes what `mx.quantized_matmul` computes. Each condition below
runs in its own fresh process and ends with the same comparison on the same stored bytes.

Conditions, each a suspect that the reproducing worker actually executed:

  baseline                  nothing before the comparison
  after_name_collision      two kernels registered under one name with different sources
                            and dispatched before one eval, which is the MLX defect the
                            registry guards against
  after_metal_kernel_patch  mx.fast.metal_kernel temporarily replaced and restored, as the
                            source-capture helpers do
  after_custom_kernel_run   an ordinary custom kernel built and dispatched first
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

from b52_automatic_selection import write_once  # noqa: E402

DUMP = (PROJECT_ROOT / ".friday-data" / "b53-dumps"
        / "1789022551358441000-23497-port_against_library-2048.npz")

PRELUDE = {
    "baseline": "",
    "after_name_collision": """
src_a = "uint i = thread_position_in_grid.x; out[i] = a[i] + 1.0f;"
src_b = "uint i = thread_position_in_grid.x; out[i] = a[i] + 100.0f;"
def build(source):
    return mx.fast.metal_kernel(name="b53_poison_probe", input_names=["a"],
                                output_names=["out"], source=source,
                                ensure_row_contiguous=True)
zeros = mx.zeros((4,), dtype=mx.float32)
first = build(src_a)(inputs=[zeros], output_shapes=[(4,)], output_dtypes=[mx.float32],
                     grid=(4,1,1), threadgroup=(4,1,1))[0]
second = build(src_b)(inputs=[zeros], output_shapes=[(4,)], output_dtypes=[mx.float32],
                      grid=(4,1,1), threadgroup=(4,1,1))[0]
mx.eval(first, second)
""",
    "after_metal_kernel_patch": """
real = mx.fast.metal_kernel
def record(**kwargs):
    return real(**kwargs)
mx.fast.metal_kernel = record
try:
    pass
finally:
    mx.fast.metal_kernel = real
""",
    "after_custom_kernel_run": """
kernel = mx.fast.metal_kernel(name="b53_plain_probe", input_names=["a"],
                              output_names=["out"],
                              source="uint i = thread_position_in_grid.x; out[i] = a[i] + 1.0f;",
                              ensure_row_contiguous=True)
zeros = mx.zeros((4,), dtype=mx.float32)
out = kernel(inputs=[zeros], output_shapes=[(4,)], output_dtypes=[mx.float32],
             grid=(4,1,1), threadgroup=(4,1,1))[0]
mx.eval(out)
""",
}

TEMPLATE = """
import json, sys
sys.path.insert(0, %(root)r)
sys.path.insert(0, %(tools)r)
import mlx.core as mx
import numpy as np
%(prelude)s
arrays = mx.load(%(dump)r)
w, s, b, x = arrays["weight"], arrays["scales"], arrays["biases"], arrays["x"]
mx.eval(w, s, b, x)
from b42_qmv_kernel import K3840, PORT, run
library = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4)
ported = run(PORT, w, s, b, x, int(w.shape[0]), 3840)
special = run(K3840, w, s, b, x, int(w.shape[0]), 3840)
deq = mx.dequantize(w, s, b, group_size=64, bits=4)
mx.eval(library, ported, special, deq)
rows = np.array(deq.astype(mx.float32), copy=True).astype(np.float64)
vec = np.array(x.astype(mx.float32), copy=True).reshape(-1).astype(np.float64)
exact = rows @ vec
den = np.maximum(np.abs(exact), 1e-30)
lib = np.array(library.astype(mx.float32), copy=True).reshape(-1)
ker = np.array(ported.astype(mx.float32), copy=True).reshape(-1)
print("RESULT " + json.dumps({
    "library_median_relative_error": float(np.median(np.abs(lib - exact) / den)),
    "kernel_median_relative_error": float(np.median(np.abs(ker - exact) / den)),
    "library_equals_kernels": bytes(memoryview(library)) == bytes(memoryview(ported)),
    "kernels_agree": bytes(memoryview(ported)) == bytes(memoryview(special)),
}))
"""


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    results = {}
    for label, prelude in PRELUDE.items():
        script = TEMPLATE % {"root": str(PROJECT_ROOT),
                             "tools": str(PROJECT_ROOT / "tools"),
                             "dump": str(DUMP), "prelude": prelude}
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                   text=True, cwd=str(PROJECT_ROOT))
        payload = {}
        for line in completed.stdout.splitlines():
            if line.startswith("RESULT "):
                payload = json.loads(line[len("RESULT "):])
        results[label] = {"returncode": completed.returncode, **payload,
                          "stderr": completed.stderr.strip()[:600]}

    baseline = results["baseline"].get("library_median_relative_error")
    guilty = [label for label, row in results.items()
              if label != "baseline"
              and row.get("library_median_relative_error", 0) > 10 * (baseline or 1e-9)]

    record = {
        "schema": "ironmule.b53_library_poisoning.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "diagnosis",
        "status": "measured",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "question": "what earlier action in a process makes mx.quantized_matmul return "
                    "percent-level wrong results on the stored failing input",
        "input": {"dump": str(DUMP.relative_to(PROJECT_ROOT)),
                  "sha256": hashlib.sha256(DUMP.read_bytes()).hexdigest()},
        "conditions": results,
        "conditions_that_broke_the_library": guilty,
        "each_condition_ran_in_a_fresh_process": True,
        "mlx_version": subprocess.run(
            [sys.executable, "-c", "import mlx.core as mx; print(mx.__version__)"],
            capture_output=True, text=True).stdout.strip(),
        "device": {"platform": platform.platform(), "machine": platform.machine()},
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip(),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({label: {k: v for k, v in row.items() if k != "stderr"}
                      for label, row in results.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
