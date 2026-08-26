"""E13 primary analysis. Runs exactly once, on the completed frozen set.

Everything here follows research/raw/E13_preregistration.md sections 7 to 9.
Nothing in the decision rule, the margin or the metric is chosen here.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e13_plan_quality import MARGIN, RAW, analyse  # noqa: E402


def main() -> int:
    path = RAW / "E13_results_main.json"
    if not path.is_file():
        print("main results missing")
        return 1
    payload = json.loads(path.read_text())
    result = analyse(payload["strict"], payload["reusable"])
    result["wall_seconds"] = payload["wall_seconds"]
    result["mlx_peak_bytes"] = payload["mlx_peak_bytes"]
    result["excluded"] = payload["excluded"]

    d = result["paired_difference"]
    print("=" * 74)
    print(f"PRIMARY   containment accuracy, averaged over {result['contexts']} contexts "
          f"({result['questions']} questions)")
    print("=" * 74)
    print(f"  StrictOneShotPlan     {result['strict_accuracy']:.4f}")
    print(f"  ReusableSessionPlan   {result['reusable_accuracy']:.4f}")
    print(f"  paired difference     {d['mean']:+.4f}   "
          f"95% CI [{d['ci_low']:+.4f}; {d['ci_high']:+.4f}]   clusters={d['clusters']}")
    print(f"  non-inferiority margin -{MARGIN:.2f}: CI lower {d['ci_low']:+.4f} "
          f"{'>' if d['ci_low'] > -MARGIN else '<='} -{MARGIN:.2f}")
    print(f"\n  VERDICT: {result['verdict']}")

    print(f"\nDISCORDANCE (containment)")
    print(f"  strict correct / reusable wrong : {result['discordance']['strict_only']}")
    print(f"  reusable correct / strict wrong : {result['discordance']['reusable_only']}")
    print(f"  answer token divergence         : {result['divergent_questions']}/{result['questions']}"
          f" = {result['answer_divergence_rate']:.4f}")
    print(f"  divergences that changed correctness: {result['divergences_changing_correctness']}")

    print(f"\nSECONDARY  (analysed only after the primary)")
    for name, s, r, diff in (
            ("exact match", result["strict_em"], result["reusable_em"], result["em_difference"]),
            ("token F1", result["strict_f1"], result["reusable_f1"], result["f1_difference"])):
        print(f"  {name:12s} strict {s:.4f}  reusable {r:.4f}  diff {diff['mean']:+.4f} "
              f"CI [{diff['ci_low']:+.4f}; {diff['ci_high']:+.4f}]")
    if result["strict_nll_mean"] is not None:
        print(f"  answer NLL   strict {result['strict_nll_mean']:.4f}  "
              f"reusable {result['reusable_nll_mean']:.4f}  "
              f"(coverage {result['nll_coverage']:.2f} of questions)")

    print(f"\n  {'band':6s} {'n':>3} {'prefix tokens':>15} {'strict':>8} {'reuse':>8} {'diff':>9} {'95% CI':>22}")
    for band, b in result["bands"].items():
        span = f"{min(b['prefix_tokens'])}-{max(b['prefix_tokens'])}"
        interval = f"[{b['ci_low']:+.4f}; {b['ci_high']:+.4f}]"
        print(f"  {band:6s} {b['clusters']:3d} {span:>15} {b['strict_acc']:8.4f} "
              f"{b['reusable_acc']:8.4f} {b['mean']:+9.4f} {interval:>22}")

    strict_ttft = statistics.median(c["strict_ttft_ms"] for c in result["per_context"])
    reuse_ttft = statistics.median(c["reusable_ttft_ms"] for c in result["per_context"])
    strict_sess = statistics.median(c["strict_session_ms"] for c in result["per_context"])
    reuse_sess = statistics.median(c["reusable_session_ms"] for c in result["per_context"])
    print(f"\nPERFORMANCE (secondary)")
    print(f"  TTFT median     strict {strict_ttft:8.1f} ms   reusable {reuse_ttft:8.1f} ms   "
          f"ratio {reuse_ttft/strict_ttft:.4f}")
    print(f"  session median  strict {strict_sess:8.1f} ms   reusable {reuse_sess:8.1f} ms   "
          f"ratio {reuse_sess/strict_sess:.4f}")
    print(f"  MLX peak {result['mlx_peak_bytes']/1e9:.2f} GB, wall {result['wall_seconds']:.0f} s")

    result["performance"] = {"strict_ttft_ms": strict_ttft, "reusable_ttft_ms": reuse_ttft,
                             "ttft_ratio": reuse_ttft / strict_ttft,
                             "strict_session_ms": strict_sess, "reusable_session_ms": reuse_sess,
                             "session_ratio": reuse_sess / strict_sess}
    (RAW / "E13_summary.json").write_text(json.dumps(
        {k: v for k, v in result.items() if k not in ("discordant_cases", "divergences")},
        indent=1, sort_keys=True, default=str))
    (RAW / "E13_discordance.json").write_text(json.dumps(
        {"discordant_cases": result["discordant_cases"], "divergences": result["divergences"]},
        indent=1, sort_keys=True, default=str))
    print(f"\nwrote {RAW/'E13_summary.json'} and {RAW/'E13_discordance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
