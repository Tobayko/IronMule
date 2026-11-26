"""Update a source-scoped experiment model after a real, rejected sampling qualification."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median

from lab.pk1 import followup_contract as protocol
from lab.pk1 import followup_trial as trial
from lab.pk1.restart_order import order_digest
from training.calibration_stability import sha
from training.code_author import write_once
from training.code_trial import read
from training.trial_release import require


def beta(successes, failures):
    a, b = 1 + successes, 1 + failures
    return {"alpha": a, "beta": b, "next_success_mean": a / (a + b)}


def plan(folder, order, control):
    """No provider access. Retain the prior qualified source when this exact program was rejected."""
    result, ctx, parent = (
        read(folder / "result.json"),
        read(folder / "context.json"),
        read(control),
    )
    require(
        order_digest(order) == order["order_id"] == ctx["order"]
        and result["passed"] is False
        and result["context"] == ctx
        and ctx["control"] == sha(control) == order["control_receipt_sha256"]
        and ctx["source"] == order["source_sha256"],
        "different program, parent or physical verdict",
    )
    require(
        "context.json" in result["artifacts"]
        and all(
            name == Path(name).name and sha(folder / name) == pin
            for name, pin in result["artifacts"].items()
        ),
        "unbound trial exports",
    )
    execution = {}
    pairs = []
    invalid = []
    aa = []
    for i in (0, 1):
        name = f"aa-{i}.json"
        if name in result["artifacts"]:
            row = read(folder / name)
            try:
                trial.validate(row, "reference", protocol.AA_SEED, ctx, parent)
            except (KeyError, ValueError, IndexError, TypeError):
                continue
            aa.append(row["cold_s"] + row["broker_to_process_s"])
    cohorts = {}
    for tag, seeds in (("mixed", protocol.MIXED), ("facts", protocol.FACTS)):
        comparisons = []
        for seed in seeds:
            files = [f"{tag}-{s}-{seed}.json" for s in ("reference", "candidate")]
            if not all(n in result["artifacts"] for n in files):
                continue
            ref, cand = (read(folder / name) for name in files)
            try:
                for label, r in (("reference", ref), ("candidate", cand)):
                    trial.validate(r, label, seed, ctx, parent, tag == "facts")
                score = trial.scores(ref, cand)
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                invalid.append({"seed": seed, "group": tag, "reason": str(exc)})
                continue
            comparisons.append(score)
            for label, r in (("reference", ref), ("candidate", cand)):
                entry = execution.setdefault(
                    label,
                    {
                        "verified_workers": 0,
                        "worker_pilot_decisions": 0,
                        "empty_effective_padding_tables": 0,
                        "cold": {
                            "waves": 0,
                            "steps": 0,
                            "padded": 0,
                            "pack_calls": 0,
                            "pack_rows": 0,
                        },
                        "warm": {
                            "waves": 0,
                            "steps": 0,
                            "padded": 0,
                            "pack_calls": 0,
                            "pack_rows": 0,
                        },
                    },
                )
                entry["verified_workers"] += 1
                entry["worker_pilot_decisions"] += r["pilot_decisions"]
                entry["empty_effective_padding_tables"] += int(
                    r["sampling"]["observed"]["startup"]["dispatch"] == {}
                )
                for phase in ("cold", "warm"):
                    entry[phase]["waves"] += 1
                    for name in ("steps", "padded", "pack_calls", "pack_rows"):
                        entry[phase][name] += r[phase]["geometry"][name]
            t = cand["sampling"]["observed"]["startup"]["telemetry"]
            key = "cold_speedup" if tag == "mixed" else "factual_cold"
            pairs.append(
                {
                    "seed": seed,
                    "group": tag,
                    "cold_rate_ratio": score[key],
                    "mode": t["mode"],
                    "reason": t["reason"],
                    "fresh_method_s": t["elapsed_s"],
                    "native_counts": [
                        list(v)
                        for v in sorted({tuple(v) for v in t["counts"].values()})
                    ],
                    "ref_init_s": ref["initialized_s"],
                    "cand_init_s": cand["initialized_s"],
                    "ref_first_wave_s": ref["cold"]["wall_s"]
                    if tag == "mixed"
                    else ref["factual_first"][0]["wall_s"],
                    "cand_first_wave_s": cand["cold"]["wall_s"]
                    if tag == "mixed"
                    else cand["factual_first"][0]["wall_s"],
                    "max_price_relative_error": score["sampling_price_relative_error"],
                }
            )
        cohorts[tag] = comparisons
    if "summary" in result:
        require(
            len(aa) == 2
            and len(cohorts["mixed"]) == len(protocol.MIXED)
            and len(cohorts["facts"]) == len(protocol.FACTS)
            and not trial.gate(
                cohorts["mixed"], cohorts["facts"], max(aa) / min(aa) - 1
            ),
            "terminal gate not reproduced",
        )
    # Prospective holdouts were replayed original traces. This posterior is for whether the
    # measurement strategy can stay adaptive, independent of any claim about whole-worker gain.
    blind = sum(r["qualifies_for_experiment"] for r in order["blind_results"])
    require(blind == len(order["blind_results"]), "unbound prior experiment selection")
    local = [p for p in pairs if p["mode"] == "adaptive"]
    fallback = [p for p in pairs if p["mode"] == "full"]
    negative = [p for p in pairs if p["cold_rate_ratio"] <= 1]
    unexpected = [p for p in negative if p["mode"] == "adaptive"]
    actions = []
    if unexpected:
        worst = max(unexpected, key=lambda p: p["cand_init_s"] - p["ref_init_s"])
        actions.append(
            {
                "kind": "attribute initialization during a naturally scheduled comparable reference startup",
                "seed": worst["seed"],
                "observed_init_difference_s": worst["cand_init_s"]
                - worst["ref_init_s"],
                "observed_first_wave_difference_s": worst["cand_first_wave_s"]
                - worst["ref_first_wave_s"],
                "work_already_performed": "Compared existing same-seed initialization, first-wave, worker and sampling clocks.",
                "unknown": "Reference method-specific time and internal I/O, CPU, GPU clock, residency and graph timings were not recorded in these uninstrumented workers. The causal source of the delay is unresolved.",
                "additional_reserved_compute_usd": 0,
                "new_target_job": False,
            }
        )
    if fallback:
        actions.append(
            {
                "kind": "retain natural ambiguity and fallback work as measured observations",
                "seeds": [p["seed"] for p in fallback],
                "work_already_performed": "Counted repeated fresh work from reported trial telemetry and priced its observed method duration.",
                "unknown": "Reusing partial timings under a changed interleaving, graph state or workload is untested.",
                "additional_reserved_compute_usd": 0,
                "new_target_job": False,
            }
        )
    if not actions:
        actions.append(
            {
                "kind": "retain available evidence and the actual gate failure",
                "new_target_job": False,
                "additional_reserved_compute_usd": 0,
            }
        )
    return {
        "schema": "ironmule.calibration-terminal-replan.v1",
        "order_id": order["order_id"],
        "source_sha256": ctx["candidate"],
        "parent_control_sha256": sha(control),
        "physical_result_sha256": sha(folder / "result.json"),
        "trial_elapsed_s": result["elapsed_s"],
        "previous_development_decision": "admit one source experiment from the prospectively confirmed sampling prediction",
        "development_decision": "retain qualified parent and close the exact rejected source experiment",
        "retry_order": False,
        "activate": False,
        "paid_job": None,
        "posterior_scope": "Beta(1,1) descriptive model; repeated measurements on the same physical device are source/target-specific. The prospective prefix-matches and measured native modes are identified separately.",
        "adaptive_regime_before_code": beta(blind, 0),
        "adaptive_regime_after_regular_source_arms": beta(
            blind + len(local), len(fallback)
        ),
        "measured_cold_win_probability": {
            "prior": beta(0, 0),
            "after": beta(len(pairs) - len(negative), len(negative)),
        },
        "actual_adaptive_regimes": len(local),
        "actual_full_fallback_regimes": len(fallback),
        "median_fresh_adaptive_method_s": median(p["fresh_method_s"] for p in local)
        if local
        else None,
        "median_full_fallback_method_s": median(p["fresh_method_s"] for p in fallback)
        if fallback
        else None,
        "paid_full_qualification_repeat_estimate_usd": order["simulation"][
            "estimated_qualification_usd"
        ],
        "accepted_production_gain": None,
        "validated_regular_pairs": pairs,
        "unusable_pairs": invalid,
        "retained_execution_counters": execution,
        "consumption_scope": "Cold/warm geometry counters record actual padding effects and packing work. Factual timing rows omit these counters. There is no first-read timestamp for padding, pass prices, pilot costs or public telemetry. A zero padding count cannot establish that calibration, graph warm-up or its effect on the scheduler can be omitted.",
        "information_decisions": actions,
    }


def record(state, order, control):
    receipt = plan(state / "trial", order, control)
    write_once(state / "post-terminal-replan.json", receipt)
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("state", "control"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    record(a.state, read(a.state / "order.json"), a.control)
