#!/usr/bin/env python3
"""What the B53 localisation established, and what it did not.

Reads the records the three preregistered configurations and the static analysis produced,
checksums them, and states one outcome. It measures nothing of its own.
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
    "B53_observed_failure_20260909.json",
    "B53_localisation_preregistration_20260910.json",
    "B53_cancellation_bound_v2_20260910.json",
    "B53_static_addressing_20260910.json",
    "B53_shader_validation_20260910.json",
    "B53_shader_validation_probe_20260910.json",
    "B53_flow_comparison_20260910.json",
)


def _digest(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.resolve().relative_to(PROJECT_ROOT)), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    loaded = {name.split("_20260")[0]: json.loads((RAW / name).read_text())
              for name in NAMES}
    bound = loaded["B53_cancellation_bound_v2"]
    static = loaded["B53_static_addressing"]
    validation = loaded["B53_shader_validation"]
    probe = loaded["B53_shader_validation_probe"]
    flows = loaded["B53_flow_comparison"]

    record = {
        "schema": "ironmule.b53_localisation_outcome.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "diagnosis_outcome",
        "status": "recorded",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "corrections_to_earlier_reasoning": [
            {
                "claim": "reassociation is excluded by four orders of magnitude, so the "
                         "two arms did not read the same bytes",
                "record": "B53_reassociation_bound_20260910",
                "status": "WITHDRAWN as established",
                "why": "that bound was measured on fresh draws from the same generator, "
                       "not on the inputs of the failure, which were never captured, and "
                       "it used an optimistic one-eps-per-sum error model",
                "replaced_by": "B53_cancellation_bound_v2_20260910",
            },
            {
                "claim": "the original inputs constrain the bound",
                "status": "IMPOSSIBLE",
                "why": "the failing run captured no tensors; every number describes inputs "
                       "from the same generator and shapes instead",
            },
        ],
        "what_is_measured_now": {
            "random_orders_move_this_workload_by_ulps":
                bound["natural_workload"]["worst_spread_ulps"],
            "cancelled_terms_move_it_by_ulps":
                bound["cancelled_control"]["worst_spread_ulps"],
            "largest_recorded_disagreement_ulps": bound["largest_recorded_ulps"],
            "classical_bound_absolute": bound["recorded_demand"]["2048"][0]
            ["reach_classical_absolute"],
            "recorded_within_classical_reach":
                bound["recorded_difference_within_classical_reach"],
            "largest_factor_beyond_classical":
                bound["largest_factor_beyond_classical_model"],
            "reading": "summation order is within a factor of about two and a half of the "
                       "recorded difference under the standard bound, and beyond it under "
                       "heavy cancellation. It is therefore not excluded, and 'the arms "
                       "read different bytes' stays one hypothesis among several",
        },
        "addressing": {
            "record": "B53_static_addressing_20260910",
            "admitted_shapes_clean": static["admitted_are_clean"],
            "finding": "for every admitted shape the largest weight, scale and activation "
                       "index is exactly one below the bound of the buffer that is bound, "
                       "every output row is written, and no row is written twice. A width "
                       "that is not a multiple of eight would have two simdgroups write "
                       "some rows; the runtime refuses those widths",
            "why_it_matters_more_than_expected": "the probe below shows an out-of-bounds "
                                                 "device read in a custom kernel is silent "
                                                 "on this platform, so this derivation is "
                                                 "the only evidence available on that "
                                                 "question",
        },
        "shader_validation": {
            "record": validation["experiment_id"],
            "environment": validation["environment_set_before_metal"],
            "api_and_gpu_validation_enabled": True,
            "instrumentation_proved": validation["instrumentation_proved"],
            "probe": {
                "record": probe["experiment_id"],
                "small_overread_caught": probe["small_overread_caught"],
                "far_overread_caught": probe["far_overread_caught"],
                "any_case_reported_shader_validation":
                    probe["any_case_reported_shader_validation"],
            },
            "finding": "the three variables switch on Metal API and GPU validation, which "
                       "the logs state, but neither a four-element overread nor one four "
                       "million elements past the buffer is reported or faults. Shader "
                       "validation does not instrument mx.fast.metal_kernel here",
            "claim_made": "none. The real call path showed no difference under this "
                          "environment, and that is not reported as a passed check",
            "real_call_path_comparisons": validation["real_call_path"]["comparisons"],
        },
        "flow_comparison": {
            "record": flows["experiment_id"],
            "verdict": flows["verdict"],
            "comparisons_per_configuration": sum(
                row["comparisons"] for row in flows["configuration_a"]["per_width"].values()),
            "inputs_mutated": flows["configuration_b"]["inputs_mutated_by_any_call"],
            "streams": flows["streams"],
        },
        "configurations_used": ["original flow", "materialised inputs",
                                "shader validation"],
        "no_further_campaigns": "the budget was three configurations and it was not "
                                "exceeded; no whole-suite campaign was run in this phase",
        "b53": {
            "state": "OPEN",
            "closed": False,
            "cause": None,
            "remedy_demonstrated": False,
            "why_not_closed": "no cause was shown and no faulty check was shown. What "
                              "changed is that one earlier conclusion is withdrawn and two "
                              "candidate mechanisms are now ruled in or out on evidence: "
                              "addressing is clean by derivation, the flow makes no "
                              "difference over 1152 comparisons, and summation order is "
                              "back on the list",
            "still_open_candidates": [
                "an arithmetic-order or math-mode difference between the library's kernel "
                "and these custom kernels, which the unpassed COMPILE_OPTIONS makes "
                "plausible and which the classical summation bound cannot exclude",
                "a fault inside MLX's custom-kernel path that no Python-level probe and no "
                "available validator can see on this platform",
            ],
        },
        "b54": {"state": "CLOSED", "untouched_by_this_phase": True},
        "release": {
            "verdict": "BLOCKED",
            "scope": "the K=3840 kernel path, whose bit-identity claim carries one "
                     "unexplained counter-observation",
        },
        "diagnosis": "COMPLETE",
        "sources": [_digest(RAW / name) for name in NAMES],
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip(),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({"diagnosis": "COMPLETE", "b53": "OPEN", "b54": "CLOSED",
                      "release": "BLOCKED",
                      "flow_comparisons": record["flow_comparison"]
                      ["comparisons_per_configuration"] * 2},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
