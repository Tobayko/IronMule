#!/usr/bin/env python3
"""Three separate answers: did the install work, what did the comparison find, where is B53.

Reads the records this phase produced, checksums them, and keeps the three apart, because
a successful install is not a finding and agreeing arms are not a solved entry.
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
METAL = PROJECT_ROOT / ".friday-data" / "b53-metal"
NAMES = (
    "B53_metal_toolchain_audit_20260910.json",
    "B53_isolated_metal_preregistration_20260910.json",
    "B53_shared_metallib_20260910.json",
    "B53_four_arm_contract_20260910.json",
    "B53_four_arm_comparison_20260910.json",
)


def _digest(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.resolve().relative_to(PROJECT_ROOT)), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    build = json.loads((RAW / "B53_shared_metallib_20260910.json").read_text())
    full = json.loads((RAW / "B53_four_arm_comparison_20260910.json").read_text())
    contract = json.loads((RAW / "B53_four_arm_contract_20260910.json").read_text())

    record = {
        "schema": "ironmule.b53_isolated_outcome.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "diagnosis_outcome",
        "status": "recorded",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "installation": {
            "what_was_installed": "Metal Toolchain 17F109, 687.9 MB, by "
                                  "`xcodebuild -downloadComponent MetalToolchain`",
            "nothing_else_changed": ["no macOS update", "no Xcode update", "no MLX update",
                                     "no change of the active developer directory",
                                     "no other package", "no security setting"],
            "no_user_action_was_needed": "no privilege prompt and no licence confirmation "
                                         "appeared",
            "before_and_after": ".friday-data/b53-evidence/toolchain_before.txt and "
                                "toolchain_after.txt",
            "compiler": build["translation"]["compiler"],
            "linker": build["translation"]["linker"],
            "proved_usable_not_merely_present": {
                "how": "a minimal shader compiled to .air and .metallib, loaded from a "
                       "standalone program and run on the GPU",
                "expected": "inputs 1,2,3,4 plus seven",
                "observed": "8, 9, 10, 11",
            },
            "old_measurements_not_relabelled": "every earlier record keeps the toolchain "
                                               "state it was taken under",
        },
        "shared_artefact": {
            "metallib": build["artefacts"]["metallib"],
            "function": build["source"]["function"],
            "source": "the kernel body and helper header of tools/b42_qmv_kernel.py, "
                      "captured as handed to MLX and unchanged; only a function signature "
                      "was written out, because MLX generates one at runtime",
            "flags": build["translation"]["flags"],
            "math_mode": build["translation"]["math_mode"],
            "why_the_flag": build["translation"]["why_math_mode_is_passed"],
            "language_version": build["translation"]["predefined_macros"].get(
                "__METAL_VERSION__"),
            "caveat": build["caveat"],
        },
        "comparison": {
            "arms": {
                "A": "unmodified mx.quantized_matmul",
                "B": "unchanged mx.fast.metal_kernel call",
                "C": "native MLX program loading the shared file through "
                     "Device::get_library(name, path) and Device::get_kernel",
                "D": "standalone Metal program, no MLX in the process, same file and "
                     "function",
            },
            "contract_check": {
                "case": contract["real_cases"][0]["case"],
                "all_four_agree": contract["all_four_agree"],
                "bindings": contract["real_cases"][0]["arm_c"]["stdout"],
            },
            "real_cases": [{"case": row["case"], "out_features": row["out_features"],
                            "distinct_outputs": row["distinct_outputs"]}
                           for row in full["real_cases"]],
            "synthetic_cases": [{"case": row["case"], "out_features": row["out_features"],
                                 "distinct_outputs": row["distinct_outputs"]}
                                for row in full["synthetic_cases"]],
            "every_arm_ran": full["every_arm_ran"],
            "all_four_agree": full["all_four_agree"],
            "verdict": full["verdict"],
            "what_it_does_not_mean": "agreement is not a solved B53, and B against C also "
                                     "changes the translation path, so it was never a "
                                     "pure comparison of call paths",
        },
        "b53": {
            "state": "OPEN",
            "closed": False,
            "cause": None,
            "reason": "the disagreement did not reappear. Nine cases, four arms, one "
                      "distinct output each. That removes a call-path or loader "
                      "explanation for these inputs and adds nothing about the inputs "
                      "that failed, which are lost",
        },
        "b54": {"state": "CLOSED", "untouched_by_this_phase": True},
        "release": {"verdict": "BLOCKED", "scope": "the K=3840 acceleration stays "
                                                   "unreleased; the library path is "
                                                   "unaffected and remains usable"},
        "binaries": {name: _digest(METAL / name) for name in
                     ("arm_c", "arm_c.cpp", "arm_d", "arm_d.mm", "b53_shared.metal",
                      "b53_shared.metallib")},
        "sources": [_digest(RAW / name) for name in NAMES],
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip(),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({"installation": "SUCCESSFUL AND PROVED USABLE",
                      "comparison": full["verdict"],
                      "b53": "OPEN", "b54": "CLOSED", "release": "BLOCKED"},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
