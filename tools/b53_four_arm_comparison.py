#!/usr/bin/env python3
"""Four ways to run one K=3840 projection, compared byte for byte.

    A  the unmodified MLX library call
    B  the existing mx.fast.metal_kernel call, unchanged
    C  a native MLX program that loads the shared .metallib by path through MLX's own
       library loader and pipeline cache
    D  a standalone Metal program with no MLX in the process, loading exactly the same
       file and the same function

A and B are untouched. C and D share the file, the function, the buffer order and the
thread and grid geometry, and each owns its buffers, logs their real lengths and offsets,
holds them until the command buffer reports completion, and checks that status before
reading anything.

Inputs are handed to C and D as raw bytes taken from the frozen corpus, so nothing is
reinterpreted on the way. Real and synthetic cases are reported apart.
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

from b42_qmv_kernel import K3840, run  # noqa: E402
from b52_automatic_selection import write_once  # noqa: E402

METAL_DIR = PROJECT_ROOT / ".friday-data" / "b53-metal"
REAL_DIR = PROJECT_ROOT / ".friday-data" / "b53-inputs"
SYNTHETIC_DIR = PROJECT_ROOT / ".friday-data" / "b53-corpus"
SCRATCH = METAL_DIR / "run"
LIBRARY = METAL_DIR / "b53_shared.metallib"
FUNCTION = "b53_qmv_k3840"
REAL_CASES = ("model_layers_0_self_attn_k_proj", "model_layers_0_self_attn_q_proj",
              "model_layers_0_mlp_gate_proj")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load(directory: Path, name: str, real: bool):
    prefix = "activation" if real else "activation"
    weight = mx.load(str(directory / f"weight_{name}.npy"))
    scales = mx.load(str(directory / f"scales_{name}.npy"))
    biases = mx.load(str(directory / f"biases_{name}.npy"))
    x = mx.load(str(directory / f"{prefix}_{name}.npy"))
    mx.eval(weight, scales, biases, x)
    return weight, scales, biases, x


def _write_bytes(path: Path, array) -> str:
    data = bytes(memoryview(array))
    path.write_bytes(data)
    return _sha(data)


def _external(binary: Path, case: str, paths: dict, out_features: int) -> dict:
    out_path = SCRATCH / f"{binary.name}_{case}.bin"
    completed = subprocess.run(
        [str(binary), str(LIBRARY), FUNCTION, str(paths["w"]), str(paths["scales"]),
         str(paths["biases"]), str(paths["x"]), str(out_path), str(out_features)],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    result = {"returncode": completed.returncode, "stdout": completed.stdout.strip(),
              "stderr": completed.stderr.strip()[:2000]}
    if completed.returncode == 0 and out_path.is_file():
        data = out_path.read_bytes()
        result["sha256"] = _sha(data)
        result["bytes"] = len(data)
        result["path"] = str(out_path.relative_to(PROJECT_ROOT))
    return result


def one_case(case: str, directory: Path, real: bool) -> dict:
    weight, scales, biases, x = _load(directory, case, real)
    out_features = int(weight.shape[0])
    SCRATCH.mkdir(parents=True, exist_ok=True)
    paths = {}
    digests = {}
    for label, array in (("w", weight), ("scales", scales), ("biases", biases), ("x", x)):
        path = SCRATCH / f"{case}_{label}.bin"
        digests[label] = _write_bytes(path, array)
        paths[label] = path

    a = mx.quantized_matmul(x, weight, scales, biases, transpose=True,
                            group_size=64, bits=4)
    b = run(K3840, weight, scales, biases, x, out_features, 3840)
    mx.eval(a, b)
    arm_a = _sha(bytes(memoryview(a)))
    arm_b = _sha(bytes(memoryview(b)))

    arm_c = _external(METAL_DIR / "arm_c", case, paths, out_features)
    arm_d = _external(METAL_DIR / "arm_d", case, paths, out_features)

    outputs = {"A_library": arm_a, "B_mlx_custom_kernel": arm_b,
               "C_mlx_loads_shared_metallib": arm_c.get("sha256"),
               "D_standalone_metal": arm_d.get("sha256")}
    distinct = sorted({value for value in outputs.values() if value})
    return {
        "case": case,
        "synthetic": not real,
        "out_features": out_features,
        "input_digests": digests,
        "outputs": outputs,
        "all_equal": len(distinct) == 1 and all(outputs.values()),
        "distinct_outputs": len(distinct),
        "arm_c": {k: v for k, v in arm_c.items() if k != "sha256"},
        "arm_d": {k: v for k, v in arm_d.items() if k != "sha256"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--contract-only", action="store_true",
                        help="run one real projection to check the call contracts")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((SYNTHETIC_DIR / "manifest.json").read_text())
    synthetic_cases = [entry["name"] for entry in manifest["cases"]]

    results = [one_case(REAL_CASES[0], REAL_DIR, real=True)]
    if not args.contract_only:
        for case in REAL_CASES[1:]:
            results.append(one_case(case, REAL_DIR, real=True))
        for case in synthetic_cases:
            results.append(one_case(case, SYNTHETIC_DIR, real=False))

    real_rows = [row for row in results if not row["synthetic"]]
    synthetic_rows = [row for row in results if row["synthetic"]]
    every_arm_ran = all(row["outputs"]["C_mlx_loads_shared_metallib"]
                        and row["outputs"]["D_standalone_metal"] for row in results)
    all_equal = all(row["all_equal"] for row in results)

    record = {
        "schema": "ironmule.b53_four_arm_comparison.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "diagnosis",
        "status": "measured",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "preregistration": "B53_isolated_metal_preregistration_20260910",
        "phase": "contract_check" if args.contract_only else "full_corpus",
        "shared_artefact": {
            "metallib": str(LIBRARY.relative_to(PROJECT_ROOT)),
            "sha256": _sha(LIBRARY.read_bytes()),
            "function": FUNCTION,
            "built_by": "B53_shared_metallib_20260910",
        },
        "binaries": {
            name: {"path": str((METAL_DIR / name).relative_to(PROJECT_ROOT)),
                   "sha256": _sha((METAL_DIR / name).read_bytes())}
            for name in ("arm_c", "arm_d")
        },
        "real_cases": real_rows,
        "synthetic_cases": synthetic_rows,
        "every_arm_ran": every_arm_ran,
        "all_four_agree": all_equal,
        "verdict": ("NO REPRODUCTION" if all_equal and every_arm_ran
                    else "ARM FAILED TO RUN" if not every_arm_ran
                    else "DIFFERENCE FOUND"),
        "reading": "four agreeing arms mean the disagreement did not reappear here. That "
                   "is not a solved B53, and a difference would have been a localised "
                   "finding rather than a proof of an MLX defect",
        "not_measured": "no timing, no performance claim, no new model run",
        "mlx_version": mx.__version__,
        "device": {"platform": platform.platform(), "machine": platform.machine()},
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip(),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({"verdict": record["verdict"], "cases": len(results),
                      "all_four_agree": all_equal,
                      "first_case": {"case": results[0]["case"],
                                     "outputs": results[0]["outputs"],
                                     "arm_c_stdout": results[0]["arm_c"].get("stdout"),
                                     "arm_d_stdout": results[0]["arm_d"].get("stdout")}},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
