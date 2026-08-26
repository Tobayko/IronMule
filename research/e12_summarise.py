"""Turn E12 raw results into the classification, summary and failure records.

Comparison B needs one correction that the harness cannot make while running: each
arm decodes on its own argmax, so once the single-shot arm picks a different token
it is being compared against a different context and the logit difference stops
meaning anything. The honest B figure is the maximum difference up to and including
the first step at which the two plans still agree on the token.
"""

from __future__ import annotations

import glob
import json
import statistics
import sys
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"


def b_pre_divergence(rec: dict) -> tuple[float, int | None]:
    """max |delta| while both plans still agree, and the step where they part."""
    worst = rec["B_prefill_max_abs_diff"]
    for step in rec["steps"]:
        if step["token_single_shot"] != step["token_baseline"]:
            return worst, step["step"]
        worst = max(worst, step["B_max_abs_diff"])
    return worst, None


def summarise(paths: list[Path]) -> dict:
    runs, cases = [], []
    for path in paths:
        payload = json.loads(path.read_text())
        runs.append({"tag": payload["tag"], "pid": payload["pid"],
                     "aborted": payload["stage_aborted"],
                     "wall_seconds": payload["wall_seconds"], "path": str(path)})
        for case in payload["cases"]:
            if case.get("aborted"):
                cases.append({"tag": payload["tag"], **case})
                continue
            b_vals, div_steps = [], []
            for rec in case["requests"]:
                worst, step = b_pre_divergence(rec)
                b_vals.append(worst)
                div_steps.append(step)
            cases.append({
                "tag": payload["tag"], "pid": payload["pid"],
                "prefix_kind": case["prefix_kind"], "prefix_length": case["prefix_length"],
                "realised_prefix_tokens": case["realised_prefix_tokens"],
                "capacity": case["capacity"],
                "prompt_tokens_min": min(case["prompt_tokens"]),
                "prompt_tokens_max": max(case["prompt_tokens"]),
                "aligned_64": case["aligned_64"], "aligned_256": case["aligned_256"],
                "window_crossed_at_prefill": case["window_crossed_at_prefill"],
                "window_crossed_only_in_decode": case["window_crossed_only_in_decode"],
                "window_clips_in_mask": any(r["mask_baseline"]["window_clips"]
                                            for r in case["requests"]),
                "A_pass": case["A_pass"], "A_failures": case["A_failures"],
                "A_requests_exact": sum(r["exact_equal"] for r in case["requests"]),
                "A_hashes_all_equal": all(r.get("hashes_equal", False) for r in case["requests"]),
                "A_steps_compared": sum(len(r["steps"]) for r in case["requests"]),
                "B_max_abs_diff_pre_divergence": max(b_vals),
                "B_max_abs_diff_raw": case["B_max_abs_diff"],
                "B_requests_with_different_tokens": case["B_requests_with_different_tokens"],
                "B_first_divergence_steps": div_steps,
                "cold_ttft_ms": statistics.median(r["chunked_cold_ttft_ns"] for r in case["requests"]) / 1e6,
                "reuse_ttft_ms": statistics.median(r["chunked_reuse_ttft_ns"] for r in case["requests"]) / 1e6,
                "single_shot_ttft_ms": statistics.median(r["single_shot_ttft_ns"] for r in case["requests"]) / 1e6,
                "snapshot_build_ms": case["snapshot_build_ns"] / 1e6,
                "mlx_peak_gb": case["mlx_peak_bytes"] / 1e9,
            })

    ok = [c for c in cases if "A_pass" in c]
    failed = [c for c in ok if not c["A_pass"]]
    aborted = [c for c in cases if c.get("aborted")]

    classification = "INSUFFICIENT_EVIDENCE"
    if aborted:
        classification = "INSUFFICIENT_EVIDENCE"
    elif failed:
        below = [c for c in ok if c["prefix_length"] < 1024 and c["A_pass"]]
        classification = "DOMAIN_RESTRICTED" if below else "PLAN_INTERNAL_FAILURE"
    elif ok:
        classification = "PLAN_INTERNAL_EXACT"

    lengths_pass = sorted({c["prefix_length"] for c in ok if c["A_pass"]})
    return {
        "experiment": "E12",
        "runs": runs,
        "classification_comparison_A": classification,
        "classification_comparison_B": "PLAN_DIVERGENCE",
        "cases_total": len(cases), "cases_passed": len(ok) - len(failed),
        "cases_failed": len(failed), "cases_aborted": len(aborted),
        "total_steps_compared": sum(c["A_steps_compared"] for c in ok),
        "total_requests_compared": sum(c["A_requests_exact"] for c in ok),
        "lengths_passing": lengths_pass,
        "lengths_failing": sorted({c["prefix_length"] for c in failed}),
        "hashes_all_equal": all(c["A_hashes_all_equal"] for c in ok),
        "B_max_abs_diff_pre_divergence_overall": max((c["B_max_abs_diff_pre_divergence"] for c in ok), default=None),
        "B_below_window": max((c["B_max_abs_diff_pre_divergence"] for c in ok
                               if not c["window_crossed_at_prefill"]), default=None),
        "B_at_or_above_window": max((c["B_max_abs_diff_pre_divergence"] for c in ok
                                     if c["window_crossed_at_prefill"]), default=None),
        "B_requests_differing_total": sum(c["B_requests_with_different_tokens"] for c in ok),
        "cases": cases,
    }


def failures(paths: list[Path]) -> dict:
    out = []
    for path in paths:
        payload = json.loads(path.read_text())
        for case in payload["cases"]:
            for rec in case.get("requests", []):
                for failure in rec["failures"]:
                    out.append({"tag": payload["tag"], "prefix_kind": case["prefix_kind"],
                                "prefix_length": case["prefix_length"],
                                "request": rec["request"], **failure})
    return {"experiment": "E12", "failure_records": out, "count": len(out)}


def main(argv=None) -> int:
    pattern = str(RAW / "E12_results_*.json")
    paths = sorted(Path(p) for p in glob.glob(pattern))
    if not paths:
        print("no E12 result files")
        return 1
    summary = summarise(paths)
    (RAW / "E12_summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    (RAW / "E12_failures.json").write_text(json.dumps(failures(paths), indent=1, sort_keys=True, default=str))

    print(f"files: {[p.name for p in paths]}")
    print(f"Comparison A: {summary['classification_comparison_A']}  "
          f"{summary['cases_passed']}/{summary['cases_total']} cases, "
          f"{summary['total_requests_compared']} requests, "
          f"{summary['total_steps_compared']} decode steps compared")
    print(f"KV hashes all equal: {summary['hashes_all_equal']}")
    print(f"Comparison B (pre-divergence): below window {summary['B_below_window']}, "
          f"at/above window {summary['B_at_or_above_window']}, "
          f"{summary['B_requests_differing_total']} requests with different tokens")
    print(f"\n{'kind':10s}{'L':>6}{'cap':>6}{'win':>5}{'clip':>6}{'A':>6}{'fail':>6}"
          f"{'B_pre':>9}{'B_raw':>9}{'Bdiff':>7}{'cold':>9}{'reuse':>9}{'ratio':>7}")
    for c in summary["cases"]:
        if "A_pass" not in c:
            print(f"{c['prefix_kind']:10s}{c['prefix_length']:6d}  ABORTED {c['aborted']}")
            continue
        print(f"{c['prefix_kind']:10s}{c['prefix_length']:6d}{c['capacity']:6d}"
              f"{str(c['window_crossed_at_prefill'])[0]:>5}{str(c['window_clips_in_mask'])[0]:>6}"
              f"{('PASS' if c['A_pass'] else 'FAIL'):>6}{c['A_failures']:6d}"
              f"{c['B_max_abs_diff_pre_divergence']:9.4f}{c['B_max_abs_diff_raw']:9.2f}"
              f"{c['B_requests_with_different_tokens']:5d}/12"
              f"{c['cold_ttft_ms']:9.1f}{c['reuse_ttft_ms']:9.1f}"
              f"{c['reuse_ttft_ms']/c['cold_ttft_ms']:7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
