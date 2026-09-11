#!/usr/bin/env python3
"""The B53 trigger, reduced to one line of test state.

`tests/engine/test_ironmule.py` calls `mx.set_default_device(mx.cpu)` eight times and
never restores it. Under `--dist loadfile` a worker that runs that file keeps the CPU as
its default for every later file it is given. `tests/test_qmv_k3840.py` then compares
`mx.quantized_matmul`, which follows the default device, against custom Metal kernels,
which can only run on the GPU. The two arms are no longer computing on the same device.

Three states, each in a fresh process, each ending in the same comparison on the same
stored failing bytes:

    clean       default device untouched
    triggered   the default device set to the CPU first, as the leaking test leaves it
    restored    set to the CPU and then put back, as the fix does

The claim is only established if the triggered state reproduces the stored bytes exactly.
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

DUMPS = PROJECT_ROOT / ".friday-data" / "b53-dumps"

PRELUDE = {
    "clean": "",
    "triggered": "mx.set_default_device(mx.cpu)\n",
    "restored": "mx.set_default_device(mx.cpu)\nmx.set_default_device(mx.gpu)\n",
    "triggered_via_the_leaking_test": (
        "import subprocess, sys\n"  # placeholder, replaced below
    ),
}

TEMPLATE = """
import hashlib, json, sys
sys.path.insert(0, %(root)r)
sys.path.insert(0, %(tools)r)
import mlx.core as mx
%(prelude)s
from b42_qmv_kernel import K3840, PORT, run
arrays = mx.load(%(dump)r)
w, s, b, x = arrays["weight"], arrays["scales"], arrays["biases"], arrays["x"]
mx.eval(w, s, b, x)
n = int(w.shape[0])
library = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4)
ported = run(PORT, w, s, b, x, n, 3840)
special = run(K3840, w, s, b, x, n, 3840)
mx.eval(library, ported, special)
sha = lambda a: hashlib.sha256(bytes(memoryview(a))).hexdigest()
print("RESULT " + json.dumps({
    "default_device": str(mx.default_device()),
    "library_sha256": sha(library),
    "ported_sha256": sha(ported),
    "special_sha256": sha(special),
    "library_equals_kernels": sha(library) == sha(ported),
}))
"""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import mlx.core as mx

    cases = {}
    for dump in sorted(DUMPS.glob("*.npz")):
        arrays = mx.load(str(dump))
        stored = {"library": _sha(bytes(memoryview(arrays["library_first"]))),
                  "kernels": _sha(bytes(memoryview(arrays["ported"])))}
        rows = {}
        for label, prelude in PRELUDE.items():
            if label == "triggered_via_the_leaking_test":
                continue
            script = TEMPLATE % {"root": str(PROJECT_ROOT),
                                 "tools": str(PROJECT_ROOT / "tools"),
                                 "dump": str(dump), "prelude": prelude}
            completed = subprocess.run([sys.executable, "-c", script],
                                       capture_output=True, text=True,
                                       cwd=str(PROJECT_ROOT))
            payload = {}
            for line in completed.stdout.splitlines():
                if line.startswith("RESULT "):
                    payload = json.loads(line[len("RESULT "):])
            rows[label] = {
                "returncode": completed.returncode,
                **payload,
                "reproduces_stored_library_bytes":
                    payload.get("library_sha256") == stored["library"],
                "kernels_match_stored": payload.get("ported_sha256") == stored["kernels"],
                "stderr": completed.stderr.strip()[:400],
            }
        cases[dump.name] = {"stored": stored, "states": rows}

    triggered_reproduces = all(
        row["states"]["triggered"]["reproduces_stored_library_bytes"]
        for row in cases.values())
    clean_is_clean = all(row["states"]["clean"]["library_equals_kernels"]
                         for row in cases.values())
    restored_is_clean = all(row["states"]["restored"]["library_equals_kernels"]
                            for row in cases.values())

    record = {
        "schema": "ironmule.b53_device_trigger.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "diagnosis",
        "status": "measured",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "question": "does one leaked default device reproduce the stored failing bytes",
        "trigger": "mx.set_default_device(mx.cpu), left unrestored",
        "where_it_comes_from": {
            "file": "tests/engine/test_ironmule.py",
            "calls": 8,
            "restores": 0,
            "why_it_reaches_another_file": "pytest-xdist runs whole files per worker, so a "
                                           "worker that ran this file keeps the CPU as its "
                                           "default for every later file it is given",
            "also_leaks_when_run_as_a_script": "ironmule/fast.py::_self_check, which is "
                                               "only reachable under __main__ and is not "
                                               "part of fuse_projections",
        },
        "mechanism": "mx.quantized_matmul follows the default device; a custom Metal "
                     "kernel can only run on the GPU. With the default on the CPU the two "
                     "arms compute on different devices, and MLX's CPU and GPU quantised "
                     "matmuls are not bit-identical",
        "cases": cases,
        "triggered_reproduces_stored_bytes": triggered_reproduces,
        "clean_state_agrees": clean_is_clean,
        "restoring_the_device_makes_it_clean": restored_is_clean,
        "verdict": ("TRIGGER ESTABLISHED" if triggered_reproduces and clean_is_clean
                    and restored_is_clean else "NOT ESTABLISHED"),
        "what_it_is_not": "a defect in either kernel, and not a defect in MLX. Two "
                          "different devices were compared as if they were one",
        "device": {"platform": platform.platform(), "machine": platform.machine()},
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip(),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({"verdict": record["verdict"],
                      "triggered_reproduces_stored_bytes": triggered_reproduces,
                      "clean_state_agrees": clean_is_clean,
                      "restoring_makes_it_clean": restored_is_clean}, indent=2,
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
