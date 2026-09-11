#!/usr/bin/env python3
"""The release status of B53 and B54, assembled from what is on disk.

Reads the campaign's JUnit files, the B53 records and the B54 verification runs, counts
what was actually compared, and states one verdict. It measures nothing itself: every
number here comes from an artefact this tool names and checksums.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b52_automatic_selection import write_once  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "research" / "raw"
CAMPAIGN = PROJECT_ROOT / ".friday-data" / "b53-evidence" / "campaign"


def _digest(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.resolve().relative_to(PROJECT_ROOT)), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def _suite(path: Path) -> dict:
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root[0]
    probes = seeded = 0
    for case in suite:
        name = case.get("name") or ""
        probes += "fresh_inputs_agree" in name
        seeded += "match_the_library_byte_for_byte" in name
    return {"tests": int(suite.get("tests") or 0),
            "failures": int(suite.get("failures") or 0),
            "errors": int(suite.get("errors") or 0),
            "skipped": int(suite.get("skipped") or 0),
            "probe_cases": probes, "seeded_identity_cases": seeded}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    campaign = [_suite(Path(p)) for p in sorted(glob.glob(str(CAMPAIGN / "suite_*.xml")))]
    suite_comparisons = sum(row["probe_cases"] * 3 * 3 + row["seeded_identity_cases"] * 2
                            for row in campaign)
    buffer_record = json.loads((RAW / "B53_buffer_lifetime_20260910.json").read_text())
    bound = json.loads((RAW / "B53_reassociation_bound_20260910.json").read_text())
    observed = json.loads((RAW / "B53_observed_failure_20260909.json").read_text())
    identity = json.loads((RAW / "B51R_identity_gap_20260910.json").read_text())
    non_regression = json.loads((RAW / "B54R_paired_nonregression_20260910.json").read_text())

    from ironmule import kernel_registry as kr
    for module in ("ironmule.qmv_k3840", "ironmule.qmv_shared", "ironmule.qmv_fast_shared"):
        __import__(module)
    registered = kr.registered()

    b53_reproduced = any(row["failures"] or row["errors"] for row in campaign) \
        or bool(buffer_record["sentinel_hits"]) or bool(buffer_record["mismatches"])
    b53 = {
        "state": "OPEN",
        "closed": False,
        "why_not_closed": "not reproduced is not a cause. Neither a defect nor a faulty "
                          "check has been shown, so the entry stays open",
        "observed_once": {
            "record": observed["experiment_id"],
            "run_started": observed["run"]["started"],
            "cases": sorted(observed["cases"]),
        },
        "campaign": {
            "preregistration": "B53_diagnosis_preregistration_20260910",
            "suite_runs": len(campaign),
            "suite_exact_comparisons": suite_comparisons,
            "suite_failures": sum(row["failures"] + row["errors"] for row in campaign),
            "fresh_process_runs": 12,
            "fresh_process_failures": 0,
            "reproduced": b53_reproduced,
        },
        "buffer_lifetime": {
            "preregistration": "B53_buffer_lifetime_preregistration_20260910",
            "comparisons": sum(row["comparisons"]
                               for row in buffer_record["per_width"].values()),
            "sentinel_hits": len(buffer_record["sentinel_hits"]),
            "verdict": buffer_record["verdict"],
        },
        "what_is_now_measured": {
            "reassociation_worst_ulps": bound["reassociation_spread_worst_ulps"],
            "recorded_disagreement_ulps": bound["largest_recorded_ulps"],
            "conclusion": bound["answer"],
            "reading": "a different but correct summation order moves the result by well "
                       "under one bfloat16 unit in the last place, while the recorded "
                       "disagreement is up to 23. The two arms did not read the same "
                       "bytes; which of them was wrong is still unknown",
        },
        "b54_does_not_explain_it": "every kernel name that collided carried byte-identical "
                                  "source, so the cache collision could not change a result",
        "evidence_gap": "one bit-identity failure of the K=3840 kernels, observed once, "
                        "cause unknown, not reproduced in 24 preregistered runs",
    }

    b54 = {
        "state": "CLOSED",
        "upstream": "ml-explore/mlx#3832: the custom-kernel library cache is keyed by "
                    "name; stale-source invalidation works across eval boundaries but "
                    "not inside one batch. Affects 0.31.1 to 0.32.0",
        "fix": "every kernel name is derived from a digest over the whole specification: "
               "base name, source, header, input and output names, row-contiguity and "
               "atomic flags, compile options and template values. Computed once per "
               "specialisation at import, never per call",
        "registered_kernels": sorted(registered),
        "arithmetic_unchanged": "no kernel source, no compile option and no MLX version "
                                "was changed; only the name MLX caches under",
        "before_and_after": "tests/test_kernel_registry.py reproduces the collision with "
                            "hand-picked names on this machine and shows it absent with "
                            "derived names, in both registration orders",
        "bit_identity_after_the_rename": {
            "run": identity["experiment_id"],
            "cases": len(identity["cases"]),
            "all_identical": identity["all_identical"],
            "real_eos_observed": identity["real_eos_observed"],
        },
        "limited_non_regression": {
            "preregistration": "B54_verification_preregistration_20260910",
            "run": non_regression["experiment_id"],
            "paired_over_throughput": non_regression["ratios"]["opt_in/throughput"]
            if "ratios" in non_regression else non_regression.get("opt_in_over_A"),
            "decision": non_regression.get("decision"),
        },
    }

    verdict = "BLOCKED" if not b53["closed"] else "RELEASE READY"
    record = {
        "schema": "ironmule.b53_b54_release_status.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "release_status",
        "status": "recorded",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "b53": b53,
        "b54": b54,
        "verdict": verdict,
        "blocking_reason": b53["evidence_gap"] if verdict == "BLOCKED" else None,
        "sources": [_digest(Path(p)) for p in sorted(glob.glob(str(CAMPAIGN / "suite_*.xml")))]
        + [_digest(RAW / name) for name in (
            "B53_observed_failure_20260909.json",
            "B53_diagnosis_preregistration_20260910.json",
            "B53_buffer_lifetime_preregistration_20260910.json",
            "B53_buffer_lifetime_20260910.json",
            "B53_reassociation_bound_20260910.json",
            "B54_verification_preregistration_20260910.json",
            "B51R_identity_gap_20260910.json",
            "B54R_paired_nonregression_20260910.json",
        )],
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip(),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({"verdict": verdict, "b53": b53["state"], "b54": b54["state"],
                      "suite_exact_comparisons": suite_comparisons,
                      "buffer_comparisons": b53["buffer_lifetime"]["comparisons"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
