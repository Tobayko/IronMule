"""Price changes to observed, editable source sites using two protected post-merge serving starts."""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

from lab.pk1.followup_contract import contract_digest
from lab.pk1.restart_order import order_digest
from training.code_author import write_once
from training.code_trial import read
from training.experiment_planner import _historical_notes

PHASES = ("startup", "cold", "warm", "factual")


def plan(observations, control, weights, discovery_contract, checkout, archive, hourly):
    from lab.pk1.post_restart_verify import verify_pair

    if not math.isfinite(hourly) or hourly <= 0:
        raise ValueError("missing measured lease price")
    proof = verify_pair(observations, control, weights, discovery_contract, checkout)
    docs = read(discovery_contract)
    rows = [read(path) for path in observations]
    prices = defaultdict(lambda: [[0.] * len(PHASES) for _ in rows])
    exclusions = defaultdict(float)
    for index, row in enumerate(rows):
        for phase in row["phases"]:
            j = PHASES.index(phase["phase"])
            exclusions["unassigned measured seconds"] += phase["unassigned_s"]
            for site in phase["sites"]:
                key = (site["path"], site["function"], site["line"])
                if site["eligible"]:
                    prices[key][index][j] += site["self_s"]
                else:
                    exclusions[site["reason"]] += site["self_s"]
    broker = [r["broker_elapsed_s"] for r in rows]
    # Six alternating pairs, two A/A workers and two additional factual-first pairs.
    # Use complete observed worker cost rather than charging only the measured source phase.
    cost = 18 * max(broker)
    diagnostics = sum(broker)
    min_fraction = 1 - 1 / docs["minimum_rate_ratio"]
    scenarios = docs["scenarios"]
    ranking = []
    for (path, name, line), measured in prices.items():
        target = {"path": path, "class": "GgmlEngine", "function": name.rsplit(".", 1)[1], "line": line}
        notes = _historical_notes(target, archive)
        # A notebook match at a different method/path cannot penalize this particular implementation.
        failures = [n["id"] for n in notes if n["contains_negative_result"] and
                    (name.lower() in (n.get("title", "") + " " + n["excerpt"]).lower() or
                     path.lower() in n["excerpt"].lower() and target["function"].lower() in n["excerpt"].lower())]
        cold = [r["arm"]["cold_s"] for r in rows]
        scenarios_by_horizon = []
        for warm_count in scenarios["sustained_warm_waves_per_restart"]:
            # The completed cold wave already includes startup. One factual wave still runs per restart.
            exposed = [values[0] + values[1] + warm_count * values[2] + values[3] for values in measured]
            elapsed = [row["arm"]["cold_s"] + warm_count * row["arm"]["warm"]["wall_s"]
                       + row["arm"]["factual_timing"]["wall_s"] for row in rows]
            scenarios_by_horizon.append({"warm_waves": warm_count, "base_s": median(elapsed),
                    "conditional_saved_s": [max(0., reduce * min(exposed) - (max(exposed) - min(exposed))) / (1 + len(failures))
                        for reduce in scenarios["editable_time_reduction"]]})
        exposure = [v[0] + v[1] for v in measured]
        conservative = max(0., .25 * min(exposure) - (max(exposure) - min(exposure))) / (1 + len(failures))
        optimistic = .5 * min(exposure)
        typical = scenarios_by_horizon[1]["conditional_saved_s"][1]
        horizon = scenarios["future_starts_for_cost_sensitivity"][-1]
        # Both absolute cost and normalized correct-work rate must improve in the preregistered scenario.
        eligible = (conservative > min_fraction * max(cold) and typical > 0 and typical * horizon > cost + diagnostics)
        ranking.append({"target": target, "measured_exclusive_s_by_seed_phase": measured,
                        "phase_order": PHASES, "prior_rejections": failures, "historical_knowledge": notes,
                        "estimated_cold_saving_s": conservative, "optimistic_cold_saving_s": optimistic,
                        "cost_break_even_starts": math.ceil((cost + diagnostics) / typical) if typical else None,
                        "scenario_returns": scenarios_by_horizon, "eligible": eligible})
    ranking.sort(key=lambda v: (-int(v["eligible"]), -v["estimated_cold_saving_s"], -v["optimistic_cold_saving_s"], v["target"]["function"]))
    winner = ranking[0] if ranking else None
    common = {"base_commit": docs["base_commit"], "parent_order": docs["parent_order"], "control_receipt_sha256": docs["control_receipt_sha256"],
              "weights_sha256": docs["weights_sha256"], "source_sha256": docs["source_sha256"], "contract": contract_digest(),
              "observations": proof["observation_sha256"], "verification": proof, "candidates": ranking,
              "unassigned_and_protected_s_across_runs": dict(exclusions), "simulation_kind": "simulated",
              "lease_rate_usd_per_hour": hourly, "estimated_qualification_s": cost, "estimated_qualification_usd": cost * hourly / 3600,
              "measured_discovery_s": diagnostics, "discovery_usd": diagnostics * hourly / 3600,
              "required_cold_saving_s": min_fraction * max(r["arm"]["cold_s"] for r in rows),
              "scope": "Costs are marginal estimates for the existing lease. Future restart frequency and attainable source acceleration remain unmeasured."}
    if winner is None or not winner["eligible"]:
        promising = [a for a in ranking if a["optimistic_cold_saving_s"] > common["required_cold_saving_s"]]
        return {"schema": "ironmule.post-restart-deferred.v1", **common, "selected": None,
                "next_experiment": ({"kind": "replicate exclusive attribution before authoring", "target": promising[0]["target"]}
                                    if promising else {"kind": "none at measured editable source sites"}),
                "reason": "Insufficient net editable cold-work headroom after uncertainty and qualification cost."}
    target = winner["target"]
    order = {"schema": "ironmule.post-restart-source-order.v1", **common, "allowed_paths": [target["path"]],
             "target": target, "selected": target, "hypothesis": f"Reduce the independently measured exclusive work in GgmlEngine.{target['function']} while preserving all native work, readiness, checks, calls, observable output and fallback."}
    order["order_id"] = order_digest(order)
    return order


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observation", type=Path, action="append", required=True)
    for k in ("control", "weights", "contract", "checkout", "archive", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--hourly-usd", type=float, required=True)
    a = p.parse_args()
    write_once(a.out, plan(a.observation, a.control, a.weights, a.contract, a.checkout, a.archive, a.hourly_usd))


if __name__ == "__main__":
    main()
