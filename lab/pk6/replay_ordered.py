"""Offline replay of source qualification, scratch training, deployment and rollback from original work."""
import argparse
import hashlib
import math
from pathlib import Path
from statistics import median
import tempfile

from lab.pk6 import ordered_grade as grade
from lab.pk6 import ordered_quality
from lab.pk6 import ordered_release as deploy
from training import scoped_release as release
from training.code_trial import read
from training.trial_release import require


def replay(folder):
    ordered = folder / "ordered"
    verified = deploy.qualification(ordered)
    proof = folder / "proofs"
    fresh = folder / "fresh"
    state = folder / "target-state"
    receipt = read(state / "control.json")
    pin = release.checksum(state / "control.json")
    activation = read(proof / "activation.json")
    require(receipt["qualification"] == verified and activation == read(state / "selection.json")
            and activation["arm"] == "candidate" and activation["receipt_sha256"] == pin
            and activation["target_check_sha256"] == release.checksum(state / "target-check.json"),
            "deployment state differs from the completed observations")
    require(hashlib.sha256(deploy.materialize(folder)).hexdigest() == verified["candidate_sha256"],
            "different generated source")
    with tempfile.TemporaryDirectory() as temp:
        fit = deploy.fit(Path(temp), ordered)
        require(fit["weights_sha256"] == release.checksum(folder / "ordered-head.npz")
                == read(folder / "fit.json")["weights_sha256"], "scratch re-fit differs")
    observed = read(folder / "numerical/observed.json")
    numeric = ordered_quality.validate(observed, verified["binding"]["runtime"]["settings"])
    target = read(state / "target-check.json")
    require(target["metrics"]["numerical_quality"] == numeric, "different quality receipt")

    def worker(name):
        row = read(fresh / (name + ".json"))
        require(row["sampled_trace_sha256"] == release.checksum(fresh / (name + ".trace")),
                "different native worker trace")
        grade.validate(row, row["arm"], row["seed"], bool(row["factual_first"]))
        deploy.bound(row, receipt)
        return row

    for pair in target["metrics"]["fresh"]:
        seed = pair["seed"]
        rows = [worker(f"fresh-{arm}-{seed}") for arm in ("reference", "candidate")]
        require(grade.price(*rows, seed, pair["facts_first"]) == pair and pair["rate"] >= 1.02
                and max(pair["warm_wall"], pair["warm_p95"]) <= 1.02
                and max(pair["rss_ratio"], pair["gpu_ratio"]) <= 1.05, "fresh target gate differs")
    require([p["seed"] for p in target["metrics"]["fresh"]] == [99301, 99401], "missing fresh families")
    random, trained = (read(proof / (k + ".json")) for k in ("random", "trained"))
    recovery = read(proof / "recovery.json")
    require(recovery == read(state / "rollback-exercise.json") and recovery["passed"] and recovery["receipt_sha256"] == pin,
            "different rollback record")
    before, after = recovery["before"], recovery["after"]
    deliveries = (random, trained, before, after)
    require(len({r["work_id"] for r in deliveries}) == 4, "reused delivery during the exercise")
    for report, arm in zip(deliveries, ("reference", "candidate", "candidate", "reference"), strict=True):
        raw = worker(report["work_id"])
        require(report["arm"] == raw["arm"] == arm and report["seed"] == raw["seed"]
                and report["source_sha256"] == raw["source_sha256"] and report["completed"] == [32,32,18]
                and report["work_sha256"] == grade.canonical([raw["cold"]["answers"],raw["warm"]["answers"],raw["factual"]])
                and report["complete_first_work_s"] <= report["complete_request_s"]
                and abs(report["complete_first_work_s"] - raw["cold_s"] - raw["broker_to_process_s"]
                        - report["outside_worker_s"]) < 1e-6, "different real delivery")
    require(random["work_sha256"] == trained["work_sha256"] and before["work_sha256"] == after["work_sha256"]
            and random["decision"]["ablate_weights"] and not trained["decision"]["ablate_weights"],
            "different work or mismatched weight ablation")
    pairs = verified["metrics"]["pairs"][:6]
    saved = median(p["reference_s"] - p["candidate_s"] for p in pairs)
    evaluation_s = sum(read(path)["ledger_elapsed_s"] for path in ordered.glob("*.json")
                       if path.name.startswith(("aa-", "candidate-", "reference-")))
    return {"pull_request": receipt["publication"]["pr"], "candidate": receipt["publication"]["candidate"],
            "measured_pairs": len(verified["metrics"]["pairs"]), "mixed_first_median_rate": median(p["rate"] for p in pairs),
            "mixed_first_median_paired_saving_s": saved, "controller_ms_random_trained": [r["decision"]["cpu_ms"] for r in (random,trained)],
            "first_delivery_with_broker_s_random_trained": [r["complete_first_work_s"] for r in (random,trained)],
            "complete_requests_s_random_trained": [r["complete_request_s"] for r in (random,trained)],
            "fresh_rates": [p["rate"] for p in target["metrics"]["fresh"]], "stock_quality": numeric,
            "qualification_workers_elapsed_s": evaluation_s, "qualification_worker_payback_starts": math.ceil(evaluation_s / saved),
            "weights_sha256": fit["weights_sha256"], "rollback_completed": recovery["passed"]}


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    print(json.dumps(replay(parser.parse_args().folder), indent=2, allow_nan=False))
