"""E16 analysis. Applies research/raw/E16_preregistration.md sections 8, 9 and 10.

Criterion A1 is applied exactly as frozen, including where the pilot already showed
it is anchored to the wrong point. The shape diagnostic next to it is a reported
quantity, not a replacement.
"""

from __future__ import annotations

import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e16_replication import (ALL_WORKLOADS, E15_REFERENCE, GROWTH_LIMIT,  # noqa: E402
                             MAIN_WORKLOADS, PLANS, RAW, REPEAT_DRIFT_LIMIT, SEED, THETA)


def pct(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def boot(values, seed=SEED):
    rng = random.Random(seed)
    meds = sorted(st.median([values[rng.randrange(len(values))] for _ in values])
                  for _ in range(10000))
    return {"value": st.median(values), "ci_low": meds[250], "ci_high": meds[9750],
            "n": len(values), "spread": max(values) - min(values)}


def arm_wall(child, arm, repeat=None):
    vals = [r["wall_ns"] for r in child["runs"] if r["arm"] == arm
            and (repeat is None or r["repeat"] == repeat)]
    return st.median(vals) if vals else None


def latencies(child, arm):
    out = []
    for run in child["runs"]:
        if run["arm"] != arm:
            continue
        for r in run["requests"]:
            if r["ttft_ms"] is not None:
                out.append(r["ttft_ms"] + sum(r["inter_token_ms"]) - r["arrival_ms"])
    return out


def ttfts(child, arm):
    return [r["ttft_ms"] - r["arrival_ms"] for run in child["runs"] if run["arm"] == arm
            for r in run["requests"] if r["ttft_ms"] is not None]


def main() -> int:
    payload = json.loads((RAW / "E16_results_main.json").read_text())
    reps = payload["replicates"]
    result = {"experiment": "E16", "aborted": payload.get("aborted"), "theta": THETA}

    crashed = [r for r in reps if r.get("crashed")]
    ok = [r for r in reps if not r.get("crashed")]
    grouped_arm = "grouped4"

    # ---- correctness, including the check only real processes make possible ----
    failures = [f for r in ok for f in r["failures"]]
    determinism = {}
    for workload in ALL_WORKLOADS:
        for plan in PLANS:
            children = [r for r in ok if r["workload"] == workload and r["plan"] == plan]
            if children:
                signatures = {json.dumps(c["reference_tokens"]) for c in children}
                determinism[f"{workload}/{plan}"] = {"processes": len(children),
                                                     "identical": len(signatures) == 1}
    result["correctness"] = {
        "failure_count": len(failures), "failures": failures[:20],
        "cross_process_determinism": determinism,
        "all_deterministic": all(v["identical"] for v in determinism.values()),
        "passed": len(failures) == 0 and all(v["identical"] for v in determinism.values())}

    # ---- effect across real processes ---------------------------------------
    conditions = {}
    for workload in ALL_WORKLOADS:
        for plan in PLANS:
            children = [r for r in ok if r["workload"] == workload and r["plan"] == plan]
            if not children:
                continue
            per_process = []
            for c in children:
                seq, grp = arm_wall(c, "sequential"), arm_wall(c, grouped_arm)
                if seq and grp:
                    per_process.append(1 - grp / seq)
            first = [1 - arm_wall(c, grouped_arm, 0) / arm_wall(c, "sequential", 0)
                     for c in children if arm_wall(c, "sequential", 0)]
            last_r = max(r["repeat"] for c in children for r in c["runs"])
            last = [1 - arm_wall(c, grouped_arm, last_r) / arm_wall(c, "sequential", last_r)
                    for c in children if arm_wall(c, "sequential", last_r)]
            seq_lat = [x for c in children for x in latencies(c, "sequential")]
            grp_lat = [x for c in children for x in latencies(c, grouped_arm)]
            seq_ttft = [x for c in children for x in ttfts(c, "sequential")]
            grp_ttft = [x for c in children for x in ttfts(c, grouped_arm)]
            within = [st.pstdev([r["wall_ns"] for r in c["runs"] if r["arm"] == grouped_arm])
                      / st.mean([r["wall_ns"] for r in c["runs"] if r["arm"] == grouped_arm])
                      for c in children]
            conditions[f"{workload}/{plan}"] = {
                "processes": len(children), "G": boot(per_process),
                "G_per_process": per_process,
                "G_first_repeat": st.median(first) if first else None,
                "G_last_repeat": st.median(last) if last else None,
                "seq_p50": pct(seq_lat, 0.50), "seq_p95": pct(seq_lat, 0.95),
                "grp_p50": pct(grp_lat, 0.50), "grp_p95": pct(grp_lat, 0.95),
                "seq_ttft_p50": pct(seq_ttft, 0.50), "grp_ttft_p50": pct(grp_ttft, 0.50),
                "mean_realised_width": st.mean(r["mean_realised_width"] for c in children
                                               for r in c["runs"] if r["arm"] == grouped_arm),
                "within_process_cv": st.mean(within),
                "e15": E15_REFERENCE.get((workload, plan)),
            }
    result["conditions"] = conditions

    # ---- memory: A1, A2 as frozen; shape diagnostic reported alongside -------
    memory = []
    for c in ok:
        marks = {m["label"]: m for m in c["checkpoints"]}
        repeats = sorted(k for k in marks if k.startswith("after_repeat_"))
        warm, end = marks["after_warmup"], marks["process_end"]
        first_repeat = marks[repeats[0]]
        memory.append({
            "workload": c["workload"], "plan": c["plan"], "index": c["index"], "pid": c["pid"],
            "rss_start_mb": marks["process_start"]["rss_kb"] / 1024,
            "rss_after_load_mb": marks["after_model_load"]["rss_kb"] / 1024,
            "rss_after_warmup_mb": warm["rss_kb"] / 1024,
            "rss_end_mb": end["rss_kb"] / 1024,
            "rss_growth_from_warmup": end["rss_kb"] / warm["rss_kb"] - 1,
            "rss_growth_from_first_repeat": end["rss_kb"] / first_repeat["rss_kb"] - 1,
            "mx_active_growth_from_warmup": end["mx_active"] / warm["mx_active"] - 1,
            "mx_cache_growth_from_warmup": end["mx_cache"] / warm["mx_cache"] - 1,
            "mx_peak_gb": max(m["mx_peak"] for m in c["checkpoints"]) / 1e9,
            "compiled_bodies_end": end["compiled_bodies"],
            "py_blocks_growth_from_warmup": end["py_blocks"] / warm["py_blocks"] - 1,
        })
    result["memory"] = memory

    a1 = max(m["rss_growth_from_warmup"] for m in memory)
    a1_shape = max(m["rss_growth_from_first_repeat"] for m in memory)
    a2 = max(m["mx_active_growth_from_warmup"] for m in memory)
    drift = max(abs(c["G_first_repeat"] - c["G_last_repeat"]) for c in conditions.values()
                if c["G_first_repeat"] is not None and c["G_last_repeat"] is not None)
    checks = {"A1_rss_growth_from_warmup": a1, "A1_passed": a1 <= GROWTH_LIMIT,
              "A2_mx_active_growth": a2, "A2_passed": a2 <= GROWTH_LIMIT,
              "A3_repeat_drift": drift, "A3_passed": drift <= REPEAT_DRIFT_LIMIT,
              "shape_rss_growth_from_first_repeat": a1_shape,
              "max_mx_cache_growth": max(m["mx_cache_growth_from_warmup"] for m in memory),
              "max_py_blocks_growth": max(m["py_blocks_growth_from_warmup"] for m in memory),
              "compiled_bodies_max": max(m["compiled_bodies_end"] for m in memory)}
    result["accumulation_checks"] = checks

    # ---- classification (frozen, ordered) -----------------------------------
    def qualifies(key):
        c = conditions.get(key)
        return c is not None and c["G"]["value"] >= THETA and c["G"]["ci_low"] > 0

    required = [f"{w}/{p}" for w in MAIN_WORKLOADS for p in PLANS]
    all_qualify = all(qualifies(k) for k in required)
    complete = all(conditions.get(k, {}).get("processes", 0) >= 5 for k in required)
    accum_ok = checks["A1_passed"] and checks["A2_passed"] and checks["A3_passed"]
    shrunk = any(conditions[k]["e15"] is not None
                 and conditions[k]["e15"] - conditions[k]["G"]["value"] > 0.05
                 for k in required)

    if not result["correctness"]["passed"]:
        verdict = "CORRECTNESS_FAILURE"
    elif crashed or not complete or payload.get("aborted"):
        verdict = "INCONCLUSIVE"
    elif all_qualify and accum_ok and shrunk:
        verdict = "REPLICATED_WITH_SMALLER_EFFECT"
    elif all_qualify and accum_ok:
        verdict = "REPLICATED"
    elif all_qualify:
        verdict = "CONFOUNDED_BY_PROCESS_STATE"
    else:
        verdict = "NOT_REPLICATED"
    result["verdict"] = verdict

    # ---- report -------------------------------------------------------------
    print("=" * 100)
    print(f"E16  {len(ok)} completed replicates in {len({r['pid'] for r in ok})} distinct OS "
          f"processes, {len(crashed)} crashed, wall {payload['wall_seconds']:.0f}s")
    print("=" * 100)
    print(f"{'condition':26s} {'n':>2} {'G':>9} {'95% CI':>22} {'E15':>8} {'delta':>8} "
          f"{'width':>6} {'cv%':>6}")
    for key, c in conditions.items():
        g = c["G"]
        e15 = c["e15"]
        mark = "qualifies" if g["value"] >= THETA and g["ci_low"] > 0 else "-"
        print(f"{key:26s} {c['processes']:2d} {g['value']*100:+8.2f}% "
              f"[{g['ci_low']*100:+.2f}%;{g['ci_high']*100:+.2f}%] {e15*100:7.2f}% "
              f"{(g['value']-e15)*100:+7.2f}pp {c['mean_realised_width']:6.2f} "
              f"{c['within_process_cv']*100:5.2f}  {mark}")

    print(f"\nLATENCY, arrival based, sequential -> grouped4")
    print(f"{'condition':26s} {'p50':>22} {'p95':>22} {'TTFT p50':>20}")
    for key, c in conditions.items():
        print(f"{key:26s} {c['seq_p50']:8.1f} -> {c['grp_p50']:8.1f} "
              f"({c['grp_p50']/c['seq_p50']-1:+5.1%}) {c['seq_p95']:8.1f} -> {c['grp_p95']:8.1f} "
              f"({c['grp_p95']/c['seq_p95']-1:+5.1%}) {c['seq_ttft_p50']:7.1f} -> {c['grp_ttft_p50']:7.1f}")

    print(f"\nACCUMULATION CHECKS (A1 applied exactly as frozen)")
    print(f"  A1 RSS growth after warmup      {a1*100:+7.2f}%  limit {GROWTH_LIMIT*100:.0f}%   "
          f"{'PASS' if checks['A1_passed'] else 'FAIL'}")
    print(f"     shape: RSS growth after the FIRST repeat   {a1_shape*100:+7.2f}%   "
          f"(reported diagnostic, not a criterion)")
    print(f"  A2 MLX active growth            {a2*100:+7.2f}%  limit {GROWTH_LIMIT*100:.0f}%   "
          f"{'PASS' if checks['A2_passed'] else 'FAIL'}")
    print(f"  A3 effect drift first vs last   {drift*100:+7.2f}pp limit "
          f"{REPEAT_DRIFT_LIMIT*100:.0f}pp  {'PASS' if checks['A3_passed'] else 'FAIL'}")
    print(f"     MLX cache growth {checks['max_mx_cache_growth']*100:+.2f}%   "
          f"Python blocks growth {checks['max_py_blocks_growth']*100:+.3f}%   "
          f"compiled bodies {checks['compiled_bodies_max']}")

    print(f"\nMEMORY per replicate (MB), first three shown per condition")
    shown = set()
    for m in memory:
        key = f"{m['workload']}/{m['plan']}"
        if key in shown:
            continue
        shown.add(key)
        print(f"  {key:26s} start {m['rss_start_mb']:6.0f}  load {m['rss_after_load_mb']:6.0f}  "
              f"warmup {m['rss_after_warmup_mb']:6.0f}  end {m['rss_end_mb']:6.0f}  "
              f"mx peak {m['mx_peak_gb']:.2f} GB")

    co = result["correctness"]
    print(f"\nCORRECTNESS  failures {co['failure_count']}   "
          f"cross-process determinism all identical: {co['all_deterministic']}")
    for key, v in co["cross_process_determinism"].items():
        if not v["identical"]:
            print(f"    DIFFERS: {key} over {v['processes']} processes")

    print(f"\nVERDICT: {verdict}")
    (RAW / "E16_summary.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    print(f"wrote {RAW/'E16_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
