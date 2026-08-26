"""E14b analysis. Applies research/raw/E14b_preregistration.md sections 6 and 9.

No threshold is chosen here; every comparison is against a value frozen before the
measurement. `completion_wait` is never renamed to GPU time.
"""

from __future__ import annotations

import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e14b_arms import BATCHES, RAW, SEED  # noqa: E402

THETA = 0.10
FIELDS = ["total_ns", "host_prep_ns", "submission_ns", "completion_wait_ns"]
PRIMARY_B = 4


def ms(ns):
    return ns / 1e6


def per_process(runs, arm, batch, field="total_ns"):
    out = []
    for run in runs:
        samples = [b[field] for b in run["blocks"] if b["arm"] == arm and b["batch"] == batch]
        if not samples:
            return []
        out.append(st.median(samples))
    return out


def gain(runs, better, worse, batch):
    """1 - T_better / T_worse, paired over processes."""
    num = per_process(runs, better, batch)
    den = per_process(runs, worse, batch)
    if not num or not den:
        return None
    pairs = [1 - a / b for a, b in zip(num, den)]
    rng = random.Random(SEED)
    meds = sorted(st.median([pairs[rng.randrange(len(pairs))] for _ in pairs])
                  for _ in range(10000))
    return {"gain": st.median(pairs), "ci_low": meds[250], "ci_high": meds[9750],
            "pairs": pairs, "excludes_zero": meds[250] > 0 or meds[9750] < 0}


def main() -> int:
    payload = json.loads((RAW / "E14b_results_main.json").read_text())
    runs = payload["runs"]
    batches = [b for b in BATCHES if per_process(runs, "A", b)]
    result = {"experiment": "E14b", "processes": len(runs), "batches": batches,
              "theta": THETA, "primary_batch": PRIMARY_B}

    # ---- table --------------------------------------------------------------
    table = {}
    for b in batches:
        for arm in "ABC":
            row = {}
            for field in FIELDS:
                vals = per_process(runs, arm, b, field)
                row[field.replace("_ns", "_ms")] = ms(st.median(vals)) if vals else None
            row["ms_per_request"] = row["total_ms"] / b
            row["tokens_per_second"] = 1000 * b / row["total_ms"]
            table[f"{arm}{b}"] = row
    result["table"] = table

    # ---- harness controls ---------------------------------------------------
    noise = ms(st.median(c["total_ns"] for run in runs for c in run["controls"]["timer_control"]))
    t_a1 = table["A1"]["total_ms"]
    a1 = [s for run in runs for s in
          [x["total_ns"] for x in run["blocks"] if x["arm"] == "A" and x["batch"] == 1]]
    a1.sort()
    iqr = (a1[int(0.75 * len(a1))] - a1[int(0.25 * len(a1))]) / a1[len(a1) // 2]
    forced_sync_visible = all(table[f"A{b}"]["total_ms"] > table[f"B{b}"]["total_ms"]
                              for b in batches if b > 1)
    controls = {"timer_noise_ms": noise, "noise_fraction_of_A1": noise / t_a1,
                "control1_passed": noise / t_a1 <= 0.05,
                "control2_forced_sync_visible": forced_sync_visible,
                "relative_iqr_A1": iqr,
                "b1_arms_within_2pct": max(abs(table[f"{a}1"]["total_ms"] / t_a1 - 1)
                                           for a in "ABC") <= 0.02}
    result["controls"] = controls

    # ---- gains --------------------------------------------------------------
    gains = {}
    for b in batches:
        if b == 1:
            continue
        gains[b] = {"G_B": gain(runs, "B", "A", b), "G_CB": gain(runs, "C", "B", b),
                    "G_C": gain(runs, "C", "A", b)}
    result["gains"] = gains

    # ---- correctness --------------------------------------------------------
    entries = [c for run in runs for c in run["correctness"]]
    result["correctness"] = {
        "prefill_logits_bit_equal": all(all(c["prefill_logits_bit_equal"]) for c in entries),
        "generated_tokens_equal": all(all(c["tokens_equal"]) for c in entries),
        "counts_equal": all(all(c["counts_equal"]) for c in entries),
        "sequences_compared": sum(len(c["tokens_equal"]) for c in entries),
        "divergent": [{"batch": c["batch"], "row": i,
                       "batched": c["batched_tokens"][i], "single": c["single_tokens"][i]}
                      for c in entries for i, ok in enumerate(c["tokens_equal"]) if not ok]}

    # ---- classification (frozen, ordered) -----------------------------------
    g = gains.get(PRIMARY_B)
    gb, gcb, gc = (g["G_B"], g["G_CB"], g["G_C"]) if g else (None, None, None)
    qual = lambda x: x is not None and x["gain"] >= THETA and x["ci_low"] > 0
    if not controls["control1_passed"] or controls["relative_iqr_A1"] > 0.10 or g is None:
        verdict = "INCONCLUSIVE"
    elif qual(gb) and qual(gcb):
        verdict = "MIXED_MECHANISM"
    elif qual(gb):
        verdict = "SUBMISSION_SYNC_AMORTIZATION_SUPPORTED"
    elif qual(gcb):
        verdict = "TRUE_BATCH_SHAPE_EFFECT_SUPPORTED"
    elif qual(gc):
        verdict = "MIXED_MECHANISM"
    else:
        verdict = "MECHANISM_NOT_SUPPORTED"
    result["verdict"] = verdict

    # ---- report -------------------------------------------------------------
    print("=" * 92)
    print(f"E14b  {len(runs)} fresh processes, context {runs[0]['context_tokens']} tokens, "
          f"capacity {runs[0]['capacity']}, {len(runs[0]['blocks'])} blocks/process")
    print("=" * 92)
    print(f"{'arm':5s} {'total ms':>10} {'host prep':>10} {'submission':>11} "
          f"{'compl wait':>11} {'ms/request':>11} {'tok/s':>8}")
    for b in batches:
        for arm in "ABC":
            r = table[f"{arm}{b}"]
            print(f"{arm}{b:<4d} {r['total_ms']:10.3f} {r['host_prep_ms']:10.3f} "
                  f"{r['submission_ms']:11.3f} {r['completion_wait_ms']:11.3f} "
                  f"{r['ms_per_request']:11.3f} {r['tokens_per_second']:8.2f}")
        print()

    print("HARNESS CONTROLS")
    print(f"  1 timer noise floor      {noise:.4f} ms = {noise/t_a1*100:.2f}% of A1   "
          f"{'PASS' if controls['control1_passed'] else 'FAIL'} (ceiling 5%)")
    print(f"  2 forced sync visible    A slower than B at every b>1: "
          f"{controls['control2_forced_sync_visible']}")
    print(f"  3 barrier before blocks  applied; b=1 arms agree within 2%: "
          f"{controls['b1_arms_within_2pct']}")
    print(f"    relative IQR on A1     {iqr:.4f} (ceiling 0.10)")

    print(f"\nMECHANISM SPLIT  (theta = {THETA}, intervals over {len(runs)} processes, coarse)")
    for b, entry in gains.items():
        print(f"  b={b}")
        for label, key, meaning in (("G_B  B vs A", "G_B", "submission + sync amortised"),
                                    ("G_CB C vs B", "G_CB", "additional true-batch shape effect"),
                                    ("G_C  C vs A", "G_C", "total")):
            x = entry[key]
            if x:
                mark = "qualifies" if x["gain"] >= THETA and x["ci_low"] > 0 else "-"
                print(f"    {label:12s} {x['gain']*100:+7.2f}%  95% CI "
                      f"[{x['ci_low']*100:+.2f}%; {x['ci_high']*100:+.2f}%]  {mark:10s} {meaning}")

    print(f"\nPER-REQUEST HOST COSTS  (does submission fall per request, or only the wait?)")
    print(f"  {'b':>3} {'A sub/req':>10} {'B sub/req':>10} {'C sub/req':>10} "
          f"{'A wait/req':>11} {'B wait/req':>11} {'C wait/req':>11}")
    for b in batches:
        print(f"  {b:3d} " + " ".join(
            f"{table[f'{a}{b}']['submission_ms']/b:10.3f}" for a in "ABC") + " " + " ".join(
            f"{table[f'{a}{b}']['completion_wait_ms']/b:11.3f}" for a in "ABC"))

    print(f"\nTHROUGHPUT vs SINGLE-REQUEST LATENCY  (never conflated)")
    for b in batches:
        c = table[f"C{b}"]
        print(f"  C{b}: aggregate {c['tokens_per_second']:7.2f} tok/s   "
              f"latency of the batch {c['total_ms']:8.3f} ms   "
              f"per request {c['ms_per_request']:7.3f} ms")

    co = result["correctness"]
    print(f"\nCORRECTNESS  ({co['sequences_compared']} sequences compared against their batch-1 run)")
    print(f"  prefill logits bit identical : {co['prefill_logits_bit_equal']}")
    print(f"  generated token IDs equal    : {co['generated_tokens_equal']}")
    print(f"  token counts equal           : {co['counts_equal']}")
    if co["divergent"]:
        for d in co["divergent"][:5]:
            print(f"    divergence b={d['batch']} row={d['row']}")

    print(f"\nVERDICT: {verdict}")
    (RAW / "E14b_summary.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    print(f"wrote {RAW/'E14b_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
