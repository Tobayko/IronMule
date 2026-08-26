"""E14 analysis. Applies research/raw/E14_preregistration.md sections 9 and 11.

Nothing here chooses a threshold; every number below is compared against a value
frozen before the measurement.
"""

from __future__ import annotations

import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e14_dispatch import (BATCHES, FLOOR_MS, KERNELS_INFERRED, RAW, SEQ_STEPS,  # noqa: E402
                          WIDTHS, BOOTSTRAP_SEED)


def ms(ns) -> float:
    return ns / 1e6


def linfit(xs, ys):
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n
    mean = sy / n
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - mean) ** 2 for y in ys)
    return intercept, slope, (1 - ss_res / ss_tot if ss_tot else 1.0)


def paired_ratio(num, den, resamples=10000, seed=BOOTSTRAP_SEED):
    pairs = [a / b for a, b in zip(num, den)]
    rng = random.Random(seed)
    meds = sorted(st.median([pairs[rng.randrange(len(pairs))] for _ in pairs])
                  for _ in range(resamples))
    return {"median_ratio": st.median(pairs), "ci_low": meds[int(0.025 * resamples)],
            "ci_high": meds[int(0.975 * resamples)], "pairs": pairs}


def per_process(runs, name, field="total_ns"):
    """Median of `field` for one arrangement, one value per process."""
    out = []
    for run in runs:
        samples = run["arrangements"].get(name)
        if not samples or field not in samples[0]:
            return []          # sequential arrangements carry per-step samples instead
        out.append(st.median(s[field] for s in samples))
    return out


def main() -> int:
    payload = json.loads((RAW / "E14_results_main.json").read_text())
    runs = payload["runs"]
    result = {"experiment": "E14", "processes": len(runs),
              "floor_ms": FLOOR_MS, "kernels_inferred": KERNELS_INFERRED}

    # ---- arrangement medians -------------------------------------------------
    table = {}
    for name in ["S"] + [f"W{w}" for w in WIDTHS] + [f"B{b}" for b in BATCHES] + \
                [f"U{b}" for b in BATCHES]:
        totals = per_process(runs, name)
        if not totals:
            continue
        row = {"total_ms": ms(st.median(totals)), "per_process_ms": [ms(t) for t in totals]}
        subs = per_process(runs, name, "submit_ns")
        gpus = per_process(runs, name, "gpu_ns")
        if subs:
            row["submit_ms"] = ms(st.median(subs))
            row["gpu_ms"] = ms(st.median(gpus))
        n = int(name[1:]) if name != "S" and len(name) > 1 else (SEQ_STEPS if name == "S" else 1)
        row["logical_tokens"] = n
        row["ms_per_logical_token"] = row["total_ms"] / n
        table[name] = row
    result["arrangements"] = table

    T1 = table["B1"]["total_ms"]
    result["T1_ms"] = T1

    # ---- fits ---------------------------------------------------------------
    fits = {}
    for prefix, sizes in (("B", BATCHES), ("W", WIDTHS)):
        xs = [s for s in sizes if f"{prefix}{s}" in table]
        ys = [table[f"{prefix}{s}"]["total_ms"] for s in xs]
        a, b, r2 = linfit(xs, ys)
        fits[prefix] = {"fixed_ms": a, "marginal_ms": b, "r2": r2, "x": xs, "y": ys,
                        "usable": r2 >= 0.9}
    result["fits"] = fits
    aB = fits["B"]["fixed_ms"]
    result["a_B_ms"] = aB
    result["residual_R_ms"] = aB - FLOOR_MS

    # ---- sync probe ---------------------------------------------------------
    deltas = [st.median(p["delta_sync_ns_per_step"] for p in run["sync_probe"]) for run in runs]
    result["delta_sync_ms"] = ms(st.median(deltas))

    # ---- positive control ---------------------------------------------------
    ks, ys = [], []
    for k in sorted(runs[0]["controls"], key=int):
        vals = [st.median(s["total_ns"] for s in run["controls"][k]) for run in runs]
        ks.append(int(k))
        ys.append(ms(st.median(vals)))
    c0, slope, r2 = linfit(ks, ys)
    result["control"] = {"k": ks, "total_ms": ys, "per_dispatch_us": slope * 1000,
                         "r2": r2, "passed": slope > 0 and r2 >= 0.9}

    # ---- submit/gpu split diagnostic (preregistration risk 2) ---------------
    split = [{"name": n, "submit_ms": table[n]["submit_ms"], "gpu_ms": table[n]["gpu_ms"]}
             for n in [f"B{b}" for b in BATCHES] if "submit_ms" in table.get(n, {})]
    if len(split) >= 2:
        gpu_growth = split[-1]["gpu_ms"] / split[0]["gpu_ms"]
        sub_growth = split[-1]["submit_ms"] / split[0]["submit_ms"]
        result["submit_split"] = {"rows": split, "gpu_growth": gpu_growth,
                                  "submit_growth": sub_growth,
                                  "reliable": sub_growth < 0.5 * gpu_growth}

    # ---- paired equal-work comparisons --------------------------------------
    paired = {}
    for b in BATCHES:
        if f"B{b}" in table and f"U{b}" in table and b > 1:
            paired[f"B{b}/U{b}"] = paired_ratio(per_process(runs, f"B{b}"),
                                                per_process(runs, f"U{b}"))
    result["paired"] = paired

    # ---- inter-token latency -------------------------------------------------
    steps = [ms(s["total_ns"]) for run in runs for sample in run["arrangements"]["S"]
             for s in sample["per_step"]]
    steps.sort()
    result["inter_token_latency"] = {
        "p50_ms": steps[len(steps) // 2],
        "p95_ms": steps[min(len(steps) - 1, int(0.95 * (len(steps) - 1)))],
        "n": len(steps)}
    iqr = (steps[int(0.75 * len(steps))] - steps[int(0.25 * len(steps))]) / steps[len(steps) // 2]
    result["relative_iqr_T1"] = iqr

    # ---- correctness ---------------------------------------------------------
    identity = [row for run in runs for row in run["prefill_logit_identity"]]
    result["batched_logits_bit_equal"] = all(row["all"] for row in identity)
    result["identity_rows"] = identity

    # ---- classification (frozen) --------------------------------------------
    tok1 = table["B1"]["ms_per_logical_token"]
    tok4 = table.get("B4", {}).get("ms_per_logical_token")
    cond1 = tok4 is not None and (tok1 - tok4) / tok1 >= 0.25
    cond2 = aB / T1 >= 0.60
    cond3 = (aB - FLOOR_MS) / aB >= 0.25 if aB > 0 else False
    cond4 = result["control"]["passed"]
    predicted = result["control"]["per_dispatch_us"] / 1000 * KERNELS_INFERRED
    residual_minus_sync = result["residual_R_ms"] - result["delta_sync_ms"]
    cond5 = predicted > 0 and (1 / 3) <= (residual_minus_sync / predicted) <= 3
    conds = {"1_per_token_falls_25pct": cond1, "2_fixed_majority": cond2,
             "3_residual_not_weights": cond3, "4_control_passed": cond4,
             "5_order_of_magnitude_INFERRED": cond5,
             "dispatch_predicted_ms": predicted,
             "residual_minus_sync_ms": residual_minus_sync}
    if not cond4 or result["relative_iqr_T1"] > 0.10:
        verdict = "INCONCLUSIVE"
    elif not cond1 or (aB > 0 and (aB - FLOOR_MS) / aB < 0.10):
        verdict = "DISPATCH_MECHANISM_NOT_SUPPORTED"
    elif all([cond1, cond2, cond3, cond4, cond5]):
        verdict = "DISPATCH_MECHANISM_SUPPORTED"
    else:
        verdict = "MIXED_MECHANISM"
    result["conditions"] = conds
    result["verdict"] = verdict

    # ---- report -------------------------------------------------------------
    print("=" * 78)
    print(f"E14  {len(runs)} fresh processes, context {payload['runs'][0]['context_tokens']} tokens, "
          f"capacity {payload['runs'][0]['capacity']}")
    print("=" * 78)
    print(f"{'arrangement':12s} {'total ms':>10} {'submit ms':>10} {'gpu ms':>9} "
          f"{'tokens':>7} {'ms/token':>10}")
    for name, row in table.items():
        print(f"{name:12s} {row['total_ms']:10.3f} {row.get('submit_ms', float('nan')):10.3f} "
              f"{row.get('gpu_ms', float('nan')):9.3f} {row['logical_tokens']:7d} "
              f"{row['ms_per_logical_token']:10.3f}")

    print(f"\nFITS")
    for prefix, f in fits.items():
        print(f"  {prefix}: t(x) = {f['fixed_ms']:.3f} + {f['marginal_ms']:.3f}*x   "
              f"R^2={f['r2']:.4f}  {'usable' if f['usable'] else 'NOT USABLE (R^2<0.9)'}")
    print(f"  weight-streaming floor F = {FLOOR_MS:.3f} ms  (2.18 GB / 324 GB/s, E4 MEASURED)")
    print(f"  fixed cost a_B = {aB:.3f} ms   residual R = a_B - F = {result['residual_R_ms']:.3f} ms")
    print(f"  per-step synchronisation delta_sync = {result['delta_sync_ms']:.3f} ms  (MEASURED)")

    c = result["control"]
    print(f"\nPOSITIVE CONTROL  K={c['k']}  totals={[round(v,3) for v in c['total_ms']]}")
    print(f"  per dispatch = {c['per_dispatch_us']:.3f} us   R^2={c['r2']:.4f}   "
          f"{'PASSED' if c['passed'] else 'FAILED'}   (MEASURED in the real graph)")

    if "submit_split" in result:
        s = result["submit_split"]
        print(f"\nSUBMIT/GPU SPLIT (preregistration risk 2)")
        print(f"  gpu grows {s['gpu_growth']:.2f}x, submit grows {s['submit_growth']:.2f}x "
              f"from B1 to B{BATCHES[-1]}")
        print(f"  split usable as CPU-side cost: {s['reliable']}")

    print(f"\nEQUAL LOGICAL WORK  (batched vs the same requests unbatched)")
    for name, r in paired.items():
        print(f"  {name:9s} ratio {r['median_ratio']:.4f}  95% CI [{r['ci_low']:.4f}; "
              f"{r['ci_high']:.4f}]  ({(1-r['median_ratio'])*100:+.1f}%)  n={len(r['pairs'])} processes")

    lat = result["inter_token_latency"]
    print(f"\nLATENCY vs THROUGHPUT  (never conflated)")
    print(f"  single-request inter-token latency  p50 {lat['p50_ms']:.3f} ms  "
          f"p95 {lat['p95_ms']:.3f} ms  (n={lat['n']})")
    for b in BATCHES:
        row = table.get(f"B{b}")
        if row:
            print(f"  B{b}: per-request latency {row['total_ms']:8.3f} ms   "
                  f"aggregate {1000*b/row['total_ms']:7.2f} tok/s   "
                  f"per-token {row['ms_per_logical_token']:7.3f} ms")

    print(f"\nCORRECTNESS  batched logits bit-identical to unbatched: "
          f"{result['batched_logits_bit_equal']}")
    print(f"relative IQR on the batch-1 step: {result['relative_iqr_T1']:.4f}")
    print(f"\nCONDITIONS")
    for key, value in conds.items():
        if isinstance(value, bool):
            print(f"  {key:34s} {value}")
    print(f"  dispatch predicted from INFERRED count: {predicted:.3f} ms")
    print(f"  residual minus measured sync          : {residual_minus_sync:.3f} ms")
    print(f"\nVERDICT: {verdict}")

    (RAW / "E14_summary.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    print(f"\nwrote {RAW/'E14_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
