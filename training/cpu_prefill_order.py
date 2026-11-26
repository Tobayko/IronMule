"""Prioritize host prefill experiments from completed serving waves and direct native phase timing."""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import re
import statistics
import subprocess
from pathlib import Path

from training.experiment_planner import (
    DEFERRED,
    EVALUATOR,
    ORDER,
    _fit,
    _historical_notes,
    _next_waves,
    _rollout,
    _validate,
    digest,
)
from training.pilot_threads import save


def read(path):
    return json.loads(path.read_bytes())


def plan(prior_path, waves_dir, attribution_path, host_decision_path, archive, root):
    prior, a, host = read(prior_path), read(attribution_path), read(host_decision_path)
    waves = sorted((read(p) for p in waves_dir.glob("wave-*.json")), key=lambda w: w["captured_ns"])
    for w in waves:
        _validate(w)
    if (not prior["scratch_measured_decisions"] or prior["schema"] != EVALUATOR or len(waves) != 4
            or a["schema"] != "ironmule.cpu-prefill-attribution.v1" or a["kind"] != "measured"
            or a["prior_sha256"] != hashlib.sha256(prior_path.read_bytes()).hexdigest()
            or any(a[k] != prior[k] for k in ("evaluator_sha256", "workload", "correct", "tokens"))
            or a["base_commit"] != prior["commit"] or a["correct"] != prior["completed"]
            or host.get("schema") != DEFERRED or host.get("binding") != prior["binding"]
            or host["completed_observations"] != [digest(w) for w in waves]
            or any(w["base_commit"] != a["base_commit"] or w["binding"] != prior["binding"] for w in waves)
            or any(a["binding"].get(k) != v for k, v in prior["binding"].items() if k != "env_sha256")):
        raise ValueError("incompatible attribution or serving evidence")
    context = {"base_commit": a["base_commit"], "binding": prior["binding"], "evaluator": EVALUATOR}
    path = "ironmule/ggml_engine.py"
    commit = a["base_commit"]
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("expected the measured full base commit")
    revision = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=root, capture_output=True)
    if revision.returncode == 0:
        code = revision.stdout
    else:
        # Shallow clones can replay a measurement from its archived immutable source.
        snapshot = host_decision_path.parent / "sources" / f"{commit}-ggml_engine.py.gz"
        if not snapshot.is_file():
            raise ValueError(f"missing measured source {commit}:{path} and its archive")
        code = gzip.decompress(snapshot.read_bytes())
    if hashlib.sha256(code).hexdigest() != a["source_sha256"] or a["method"] != "GgmlEngine._prefill":
        raise ValueError("source does not match the instrumented method")
    node = next(f for c in ast.parse(code).body if isinstance(c, ast.ClassDef) and c.name == "GgmlEngine"
                for f in c.body if isinstance(f, ast.FunctionDef) and f.name == "_prefill")
    target = {"path": path, "function": a["method"], "line": node.lineno}
    usable = prior["binding"]["cores"]
    shares = {}
    for sample, original, trace in zip(waves, prior["waves"], a["windows"], strict=True):
        if trace["output_sha256"] != original["output_sha256"] or trace["requests"] != sample["requests"]:
            raise ValueError("attribution changed fixed work")
        observed = sum(call["seconds"] for call in trace["prefill"] if call["threads"] < usable)
        share = observed / trace["wall_s"]
        if not 0 <= share <= 1:
            raise ValueError("invalid inclusive timing")
        shares.setdefault(sample["workload"], []).append(share)
    forecast = _next_waves(waves)
    model = _fit(waves)
    # Factor one quarter is a sensitivity scenario, never an observed or guaranteed speedup.
    wall = statistics.fmean(_rollout(w, model)["seconds"] for w in forecast)
    exposed = statistics.fmean(_rollout(w, model)["seconds"] * statistics.median(shares[w["workload"]])
                               for w in forecast)
    estimate = {"kind": "simulated", "next_workloads": [w["workload"] for w in forecast],
                "step_model": model.to_json(), "forecast_wall_s": wall, "conditional_savings_s": [0, .25 * exposed, .5 * exposed],
                "observed_submax_prefill_fraction_by_workload": shares, "minimum_fraction_for_release": 1 - 1 / 1.02,
                "six_pair_wall_estimate_s": 12 * (prior["total_elapsed_s"] + a["qualify_s"] - prior["scratch_qualification"]["search_and_warmup_s"]),
                "assumptions": "Only timed native prefill calls below available core count can benefit. Decode thread ratios do not predict prefill timing."}
    bound = {"attribution_sha256": hashlib.sha256(attribution_path.read_bytes()).hexdigest(),
             "host_decision_sha256": hashlib.sha256(host_decision_path.read_bytes()).hexdigest(),
             "largest_host_exclusive_time_fraction": host["simulation"]["highest_remaining_measured_self_fraction"],
             "prior_sha256": a["prior_sha256"], "wave_hashes": [digest(w) for w in waves], "head_uses": a["measured_choices"]}
    if not a["measured_choices"] or exposed / wall <= 1 - 1 / 1.02:
        return {"schema": DEFERRED, **context, "reason": "no active scratch choice or insufficient measured prefill headroom",
                "target": target, "observations": bound, "simulation": estimate}
    order = {"schema": ORDER, **context, "target": target, "allowed_paths": [path], "workload": prior["workload"]["phases"],
             "profile_sha256": [digest(w) for w in waves], "observations": bound, "simulation": estimate,
             "historical_experience": _historical_notes(target, archive),
             "hypothesis": "Reduce the measured single-sequence native prompt ingestion time by letting the CPU run "
                           "multi-token prefill with an independently selected thread count. Keep the existing "
                           "head and its qualified plain-decode thread profile active. The paired evaluator tests "
                           "whether a local thread switch pays its cost on the recorded CPU workload.",
             "constraints": [("Only change the specified method in the supplied engine source. Preserve its interface, "
                              "token work, cache, observer, order and outputs. Restore thread state after errors."),
                             ("The restricted deployment and measurements use the same four-core candidate and head. "
                              "Keep existing width, thread and backend qualifications intact on other targets."),
                             ("The unchanged serving evaluator times six matched pairs. Keep correctness, TTFT, RSS "
                              "and complete work fixed. Withhold the candidate if the four-thread rule passes the "
                              "same dominance gate against it. Report ties against the rule.")]}
    order["order_id"] = digest(order)
    return order


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("prior", "waves", "attribution", "host-decision", "archive", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    order = plan(args.prior, args.waves, args.attribution, args.host_decision, args.archive,
                 Path(__file__).resolve().parents[1])
    save(args.output, order)
    print(json.dumps({k: order[k] for k in ("schema", "order_id", "target", "reason") if k in order}))


if __name__ == "__main__":
    main()
