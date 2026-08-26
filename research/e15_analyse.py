"""E15 analysis. Applies research/raw/E15_preregistration.md sections 9 and 10.

No threshold is chosen here. `completion_wait` is never called GPU time.
"""

from __future__ import annotations

import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e15_service import PRIMARY_WIDTH, RAW, SEED, WIDTHS  # noqa: E402

THETA = 0.10
MAIN_WORKLOADS = ["homogeneous", "heterogeneous", "staggered"]
LATENCY_INFLATION_LIMIT = 0.10


def pct(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def boot(pairs, seed=SEED):
    rng = random.Random(seed)
    meds = sorted(st.median([pairs[rng.randrange(len(pairs))] for _ in pairs])
                  for _ in range(10000))
    return {"value": st.median(pairs), "ci_low": meds[250], "ci_high": meds[9750],
            "n": len(pairs), "excludes_zero": meds[250] > 0 or meds[9750] < 0}


def select(runs, workload, plan, strategy):
    """One median value per process, so pairing is at process level."""
    out = []
    for run in runs:
        vals = [r["wall_ns"] for r in run["runs"]
                if r["workload"] == workload and r["plan"] == plan and r["strategy"] == strategy]
        if not vals:
            return []
        out.append(st.median(vals))
    return out


def main() -> int:
    payload = json.loads((RAW / "E15_results_main.json").read_text())
    runs = payload["runs"]
    result = {"experiment": "E15", "processes": len(runs), "theta": THETA,
              "aborted": payload.get("aborted")}

    workloads = sorted({r["workload"] for run in runs for r in run["runs"]})
    plans = sorted({r["plan"] for run in runs for r in run["runs"]})
    strategies = ["sequential"] + [f"grouped{w}" for w in WIDTHS]

    # ---- correctness first: it can veto everything ---------------------------
    reference = {(ref["workload"], ref["plan"]): ref["requests"]
                 for run in runs for ref in run["reference"]}
    failures = []
    for run in runs:
        for entry in run["runs"]:
            ref = reference.get((entry["workload"], entry["plan"]))
            if ref is None:
                continue
            for got, want in zip(entry["requests"], ref):
                if got["tokens"] != want["tokens"]:
                    failures.append({"kind": "tokens", "workload": entry["workload"],
                                     "plan": entry["plan"], "strategy": entry["strategy"],
                                     "rid": got["rid"]})
                elif got["token_count"] != want["token_count"]:
                    failures.append({"kind": "count", **{k: entry[k] for k in ("workload", "plan", "strategy")},
                                     "rid": got["rid"]})
                elif got["stop_reason"] != want["stop_reason"]:
                    failures.append({"kind": "stop_reason", **{k: entry[k] for k in ("workload", "plan", "strategy")},
                                     "rid": got["rid"]})
                elif got["kv_hash"] and want["kv_hash"] and got["kv_hash"] != want["kv_hash"]:
                    failures.append({"kind": "kv_state", **{k: entry[k] for k in ("workload", "plan", "strategy")},
                                     "rid": got["rid"]})
    order_checked = sum(1 for run in runs for e in run["runs"] if e["strategy"] == "grouped_reversed")
    order_failures = [f for f in failures if f["strategy"] == "grouped_reversed"]
    result["correctness"] = {
        "failures": failures[:40], "failure_count": len(failures),
        "order_independence_runs": order_checked,
        "order_independence_failures": len(order_failures),
        "checks_passed": len(failures) == 0}

    # ---- throughput and latency ---------------------------------------------
    table = {}
    for workload in workloads:
        for plan in plans:
            for strategy in strategies + ["grouped_reversed"]:
                entries = [e for run in runs for e in run["runs"]
                           if e["workload"] == workload and e["plan"] == plan
                           and e["strategy"] == strategy]
                if not entries:
                    continue
                walls = [e["wall_ns"] / 1e6 for e in entries]
                # Latency measured from the workload's defined arrival time, identically
                # for both strategies. The stored latency_ms starts at admission, which
                # in the sequential arm is when a request begins rather than when it
                # arrives, so it silently omits queueing and is not comparable.
                lat = [r["ttft_ms"] + sum(r["inter_token_ms"]) - r["arrival_ms"]
                       for e in entries for r in e["requests"] if r["ttft_ms"] is not None]
                lat_admission = [r["latency_ms"] for e in entries for r in e["requests"]]
                queue = [r["ttft_ms"] - r["arrival_ms"]
                         for e in entries for r in e["requests"] if r["ttft_ms"] is not None]
                ttft = [r["ttft_ms"] for e in entries for r in e["requests"] if r["ttft_ms"]]
                inter = [g for e in entries for r in e["requests"] for g in r["inter_token_ms"]]
                tokens = st.median(e["tokens_generated"] for e in entries) or 1
                prefill = st.median(run["prefill_ms"][f"{workload}/{plan}"][i]
                                    for run in runs for i in range(8))
                table[f"{workload}/{plan}/{strategy}"] = {
                    "wall_ms": st.median(walls),
                    "tokens_per_second": 1000 * tokens / st.median(walls),
                    "mean_realised_width": st.mean(e["mean_realised_width"] for e in entries),
                    "service_latency_p50": pct(lat, 0.50), "service_latency_p95": pct(lat, 0.95),
                    "admission_based_p50": pct(lat_admission, 0.50),
                    "admission_based_p95": pct(lat_admission, 0.95),
                    "full_latency_p50": pct(lat, 0.50) + prefill,
                    "full_latency_p95": pct(lat, 0.95) + prefill,
                    "queue_wait_p95": pct(queue, 0.95),
                    "ttft_p50": pct(ttft, 0.50) if ttft else None,
                    "inter_token_p50": pct(inter, 0.50), "inter_token_p95": pct(inter, 0.95),
                    "prefill_median_ms": prefill,
                }
    result["table"] = table

    # ---- gains --------------------------------------------------------------
    gains = {}
    for workload in workloads:
        for plan in plans:
            base = select(runs, workload, plan, "sequential")
            if not base:
                continue
            for width in WIDTHS:
                cand = select(runs, workload, plan, f"grouped{width}")
                if cand:
                    gains[f"{workload}/{plan}/W{width}"] = boot(
                        [1 - c / b for c, b in zip(cand, base)])
    result["gains"] = gains

    # ---- harness controls ---------------------------------------------------
    noise = st.median(c["total_ns"] for run in runs for c in run["controls"]["timer_noise"]) / 1e6
    seq_ms = [e["wall_ns"] / 1e6 for run in runs for e in run["runs"]
              if e["strategy"] == "sequential" and e["workload"] == "homogeneous"]
    seq_ms.sort()
    iqr = (pct(seq_ms, 0.75) - pct(seq_ms, 0.25)) / st.median(seq_ms)
    g1 = [k for k in gains if k.endswith("/W1")]
    w1_neutral = all(abs(gains[k]["value"]) < 0.05 for k in g1)
    result["controls"] = {"timer_noise_ms": noise, "relative_iqr_sequential": iqr,
                          "control_passed": iqr <= 0.10,
                          "w1_matches_sequential_within_5pct": w1_neutral}
    result["cold_start"] = {k: v for run in runs for k, v in run.get("cold_start", {}).items()}

    # ---- classification (frozen, ordered) -----------------------------------
    def qualifies(workload, plan):
        entry = gains.get(f"{workload}/{plan}/W{PRIMARY_WIDTH}")
        return entry is not None and entry["value"] >= THETA and entry["ci_low"] > 0

    all_main = all(qualifies(w, p) for w in MAIN_WORKLOADS for p in plans)
    homo_only = all(qualifies("homogeneous", p) for p in plans) and not all_main
    inflations = []
    for w in MAIN_WORKLOADS:
        for p in plans:
            base = table.get(f"{w}/{p}/sequential")
            cand = table.get(f"{w}/{p}/grouped{PRIMARY_WIDTH}")
            if base and cand:
                inflations.append((cand["full_latency_p95"] / base["full_latency_p95"] - 1,
                                   cand["service_latency_p95"] / base["service_latency_p95"] - 1))
    worst_full = max((a for a, _ in inflations), default=0.0)
    worst_service = max((b for _, b in inflations), default=0.0)
    result["latency_inflation"] = {"full_p95_worst": worst_full,
                                   "service_p95_worst": worst_service}

    if not result["correctness"]["checks_passed"]:
        verdict = "STATE_ISOLATION_FAILURE"
    elif not result["controls"]["control_passed"]:
        verdict = "INCONCLUSIVE"
    elif all_main and worst_full <= LATENCY_INFLATION_LIMIT:
        verdict = "ASYNC_B1_SERVICE_VIABLE"
    elif all_main:
        verdict = "THROUGHPUT_GAIN_WITH_LATENCY_COST"
    elif homo_only:
        verdict = "RAGGED_OR_ARRIVAL_SENSITIVE"
    else:
        verdict = "NO_SERVICE_GAIN"
    result["verdict"] = verdict

    # ---- best width per workload --------------------------------------------
    best = {}
    for workload in workloads:
        for plan in plans:
            candidates = [(gains[f"{workload}/{plan}/W{w}"]["value"], w) for w in WIDTHS
                          if f"{workload}/{plan}/W{w}" in gains]
            if candidates:
                best[f"{workload}/{plan}"] = max(candidates)[1]
    result["best_width"] = best

    # ---- report -------------------------------------------------------------
    print("=" * 96)
    print(f"E15  {len(runs)} fresh processes, workloads {workloads}, plans {plans}")
    print("=" * 96)
    for workload in workloads:
        for plan in plans:
            print(f"\n{workload} / {plan}")
            print(f"  {'strategy':18s} {'wall ms':>9} {'tok/s':>8} {'width':>6} "
                  f"{'svc p50':>9} {'svc p95':>9} {'queue p95':>10} {'itl p95':>9}")
            for strategy in strategies + ["grouped_reversed"]:
                row = table.get(f"{workload}/{plan}/{strategy}")
                if not row:
                    continue
                print(f"  {strategy:18s} {row['wall_ms']:9.1f} {row['tokens_per_second']:8.2f} "
                      f"{row['mean_realised_width']:6.2f} {row['service_latency_p50']:9.1f} "
                      f"{row['service_latency_p95']:9.1f} {row['queue_wait_p95']:10.2f} "
                      f"{row['inter_token_p95']:9.2f}")
            for width in WIDTHS:
                key = f"{workload}/{plan}/W{width}"
                if key in gains:
                    g = gains[key]
                    mark = "qualifies" if g["value"] >= THETA and g["ci_low"] > 0 else "-"
                    print(f"    G(W{width}) {g['value']*100:+7.2f}%  95% CI "
                          f"[{g['ci_low']*100:+.2f}%; {g['ci_high']*100:+.2f}%]  {mark}")

    print(f"\nHARNESS CONTROLS")
    print(f"  timer noise floor            {noise:.4f} ms")
    print(f"  relative IQR on sequential   {iqr:.4f} (ceiling 0.10)  "
          f"{'PASS' if result['controls']['control_passed'] else 'FAIL'}")
    print(f"  W=1 matches sequential ±5%   {w1_neutral}")
    print(f"  cold start (reported apart)  "
          f"{ {k: [round(x) for x in v['first_rounds_ms'][:3]] for k, v in list(result['cold_start'].items())[:2]} }")

    print(f"\nLATENCY (never derived by dividing group time by width)")
    print(f"  worst full-response p95 inflation at W{PRIMARY_WIDTH}   {worst_full*100:+.2f}% "
          f"(limit {LATENCY_INFLATION_LIMIT*100:.0f}%)")
    print(f"  worst service-phase p95 inflation at W{PRIMARY_WIDTH}   {worst_service*100:+.2f}%  "
          f"(stricter view, reported alongside)")
    print(f"  median latency moves the other way; per workload at W{PRIMARY_WIDTH}:")
    for w in MAIN_WORKLOADS + ["terse"]:
        for pl in plans:
            b = table.get(f"{w}/{pl}/sequential"); c = table.get(f"{w}/{pl}/grouped{PRIMARY_WIDTH}")
            if b and c:
                print(f"    {w:14s}/{pl:8s} p50 {b['service_latency_p50']:7.1f} -> "
                      f"{c['service_latency_p50']:7.1f} ms  ({c['service_latency_p50']/b['service_latency_p50']-1:+6.1%})   "
                      f"p95 {b['service_latency_p95']:7.1f} -> {c['service_latency_p95']:7.1f} ms  "
                      f"({c['service_latency_p95']/b['service_latency_p95']-1:+6.1%})")

    co = result["correctness"]
    print(f"\nCORRECTNESS  failures {co['failure_count']}   "
          f"order-independence runs {co['order_independence_runs']}, "
          f"failures {co['order_independence_failures']}")
    for f in co["failures"][:5]:
        print(f"    {f}")

    print(f"\nBEST WIDTH PER WORKLOAD  {best}")
    print(f"\nVERDICT: {verdict}")
    (RAW / "E15_summary.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    print(f"wrote {RAW/'E15_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
