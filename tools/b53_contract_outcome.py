#!/usr/bin/env python3
"""What the compiler-contract phase settled about B53, and what it corrected.

Reads the records that phase produced, checksums them, and states one outcome. It
measures nothing of its own.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b52_automatic_selection import write_once  # noqa: E402

RAW = PROJECT_ROOT / "research" / "raw"
NAMES = (
    "B53_compiler_contract_preregistration_20260910.json",
    "B53_compiler_contract_20260910.json",
    "B53_compiler_contract_v2_20260910.json",
    "B53_shader_validation_probe_20260910.json",
    "B53_localisation_outcome_20260910.json",
)
BINARIES = (
    ".venv/lib/python3.12/site-packages/mlx/lib/libmlx.dylib",
    ".venv/lib/python3.12/site-packages/mlx/lib/mlx.metallib",
    ".venv/lib/python3.12/site-packages/mlx/core.cpython-312-darwin.so",
    ".venv/lib/python3.12/site-packages/mlx/include/mlx/backend/common/metal_kernel.h",
)


def _digest(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.resolve().relative_to(PROJECT_ROOT)), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads((RAW / "B53_compiler_contract_v2_20260910.json").read_text())
    probe = json.loads((RAW / "B53_shader_validation_probe_20260910.json").read_text())

    record = {
        "schema": "ironmule.b53_contract_outcome.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "diagnosis_outcome",
        "status": "recorded",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "the_contract_as_installed": {
            "header_ships_with_the_wheel": True,
            "declares_default_safe":
                contract["contract"]["installed_header"]["declares_default_safe"],
            "modes": contract["contract"]["installed_header"]["modes"],
            "binding_docstring_agrees":
                contract["contract"]["binding_docstring_says_default_safe"],
            "how_mlx_drives_it": "libmlx.dylib carries the MTLCompileOptions setMathMode: "
                                 "selector, so the mode reaches the Metal compiler through "
                                 "that property rather than a command-line flag",
            "intended": contract["contract"]["intended_by_the_modules"],
            "passed": contract["contract"]["passed_to_mlx"],
            "effective": contract["contract"]["effective"],
            "the_three_agree": contract["contract"]["the_three_agree"],
            "registry_digest_covers_what_was_passed":
                contract["contract"]["registry_cross_check_clean"],
        },
        "power_of_the_separation": contract["power"],
        "separation_result": {
            "explicit_safe_equals_default": contract["explicit_safe_equals_default"],
            "safe_modes_match_the_library": contract["safe_modes_match_the_library"],
            "fast_arm_differs_everywhere": True,
            "real_projections": sorted(contract["real_projections"]),
            "constructed_cases": 96,
            "constructed_disagreements": "all 96, and only in the fast arm, which is the "
                                         "deliberate control and is not a runtime setting",
        },
        "what_this_removes": (
            "the math mode is eliminated as an explanation for B53. Intended, passed and "
            "effective all read safe, explicit safe is byte-identical to the default, and "
            "both are byte-identical to the library on real Gemma projections and on 96 "
            "constructed cancellation and rounding-boundary cases. The declared but never "
            "passed COMPILE_OPTIONS in the kernel modules is a documentation "
            "inconsistency with no effect on the translation"
        ),
        "what_this_does_not_do": "it repairs nothing. B53 keeps its open cause",
        "correction_to_the_shader_validation_reading": {
            "earlier_claim": "shader validation does not instrument mx.fast.metal_kernel "
                             "on this platform",
            "record": probe["experiment_id"],
            "status": "DOWNGRADED to not established",
            "why": "the control read four and then four million elements past a "
                   "four-element tensor. A tensor bound is not a Metal buffer bound, and "
                   "MLX serves small arrays out of large heaps, so the access very likely "
                   "stayed inside the bound buffer. A validator that does not report an "
                   "in-bounds access is behaving correctly",
            "can_the_proof_be_built_here": False,
            "why_not": "the Python surface exposes only aggregate memory counters and the "
                       "tensor's own nbytes; the length of the bound MTLBuffer is not "
                       "reachable, so no access can be shown to lie outside it",
            "consequence": "no further validator campaign is run. The earlier clean "
                           "validator result stays unclaimed either way",
        },
        "b53": {
            "state": "OPEN",
            "closed": False,
            "cause": None,
            "remedy_demonstrated": False,
            "candidates_after_this_phase": [
                "a fault inside MLX's custom-kernel path that no available diagnostic on "
                "this machine can see",
                "summation order, which the standard bound cannot exclude, although the "
                "two kernels transcribe the library's own order and now agree with it on "
                "every real and constructed case tried",
            ],
            "removed_this_phase": ["math mode and compile options"],
        },
        "b54": {"state": "CLOSED", "untouched_by_this_phase": True},
        "release": {
            "verdict": "BLOCKED",
            "scope": "the K=3840 acceleration stays unreleased; the existing library path "
                     "is unaffected and remains usable",
        },
        "no_random_stress_in_this_phase": True,
        "sources": [_digest(RAW / name) for name in NAMES]
        + [_digest(PROJECT_ROOT / name) for name in BINARIES],
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip(),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({"b53": "OPEN", "b54": "CLOSED", "release": "BLOCKED",
                      "removed": record["b53"]["removed_this_phase"],
                      "validator_reading": "downgraded"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
