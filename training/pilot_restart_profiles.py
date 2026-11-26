"""Fit and run the CPU Pilot's whole-profile head for one scoped, qualified restart release."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import median
import time

import numpy as np

from ironmule.pilot import Pilot, predict

FAMILIES = {"mixed-first-32": 32, "factual-first-18": 18}
NAMES = ("reference", "candidate")
PARAMS = ("w1", "b1", "w2", "b2", "w3", "b3")
SCHEMA = "ironmule.restart-pilot.v1"


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def binding(order, verified):
    return digest([SCHEMA, order["order_id"], verified["reference_sha256"], verified["candidate_sha256"],
                   verified["evaluator_sha256"], verified["binding"]])


def features(family):
    # Only quantities known before construction of the first engine are included.
    return np.asarray([1., FAMILIES.get(family, 0) / 32], dtype=np.float32)


def make_heads(train, test, seed):
    from training.pilot_w4_profiles import fit_head

    # All six nearly identical mixed-first seeds form one training family. Both factual-first
    # seeds form one held-out family. Each pair keeps the same seed on both sides.
    mixed, facts = list(FAMILIES)
    times = {mixed: [1., math.exp(-median(math.log(r / c) for r, c in train))]}
    inputs = {mixed: features(mixed)}
    before = [fit_head(times, [mixed], seed + j, inputs, epochs=0) for j in range(3)]
    after = [fit_head(times, [mixed], seed + j, inputs) for j in range(3)]
    margin = max(abs(math.log(r / c) + math.log(times[mixed][1])) for r, c in train)
    decision = Pilot.startup_profile(after, features(facts), NAMES, NAMES, "reference", in_support=True, margin_error=margin)
    ablation = {}
    for family in FAMILIES:
        ablation[family] = {
            name: Pilot.startup_profile(ensemble, features(family), NAMES, NAMES, "reference",
                                        in_support=True, margin_error=margin)
            for name, ensemble in (("random_initialization", before), ("after_mixed_observations", after))}
    winner = min(range(2), key=lambda k: median(x[k] for x in test))
    predicted = NAMES.index(decision["profile"])
    return before, after, margin, {
        "train_family": mixed, "holdout_family": facts, "train_complete_pairs": len(train), "holdout_complete_pairs": len(test),
        "holdout_winner": NAMES[winner], "holdout_decision": decision,
        "holdout_correct_profile": decision["reason"] == "qualified model choice" and predicted == winner,
        "holdout_seconds": {"shipped_pilot": [r for r, _ in test], "fixed_use_cache": [c for _, c in test],
                            "scratch_pilot": [pair[predicted] for pair in test]},
        "scores_before": [predict(h, features(facts)).tolist() for h in before],
        "scores_after": [predict(h, features(facts)).tolist() for h in after],
        "matched_eligibility_ablation": ablation,
        "train_log_margin_error": margin,
        "uncertainty": "Training spread uses mixed-first pairs only. The independently tested factual-first family is the only additional admissible context. Future families fall back."}


def fit(args):
    from lab.pk1 import restart_release, restart_trial
    from training.code_author import write_once

    start = time.perf_counter()
    verified = restart_release.approved(args)  # Replays cold/warm/18-fact, controls, source, repeated targets and vendor identity.
    order = restart_trial.read(args.order)
    trial = args.trial
    def pair(prefix, i, field):
        vals = []
        for label in ("ref", "cand"):
            row = restart_trial.read(trial / f"{prefix}{label}-{i}.json")
            vals.append(field(row))
        return vals
    train = [pair("", i, lambda row: row["cold_s"]) for i in range(restart_trial.PAIRS)]
    holdout = [pair("facts-", i, lambda row: row["factual_first"][0]["completed_s"]) for i in range(2)]
    before, heads, margin, scored = make_heads(train, holdout, args.seed)
    arrays = {f"{tag}{i}_{key}": w for tag, ensemble in (("prior", before), ("head", heads))
              for i, h in enumerate(ensemble) for key, w in zip(PARAMS, h, strict=True)}
    key = binding(order, verified)
    out = io.BytesIO()
    np.savez(out, **arrays, scope=np.asarray(key), schema=np.asarray(SCHEMA), margin=np.asarray(margin),
             holdout_admitted=np.asarray(scored["holdout_correct_profile"]))
    content = out.getvalue()
    weights = args.weights
    weights.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation keeps the selected checkpoint immutable for this trial. An interrupted write
    # must never replace the bound target artifact.
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(dir=weights.parent, delete=False) as tmp:
        staged = Path(tmp.name)
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
    try:
        try:
            os.link(staged, weights)
        except FileExistsError:
            if weights.read_bytes() != content:
                raise ValueError("checkpoint bytes changed for this order")
    finally:
        staged.unlink(missing_ok=True)
    sha = hashlib.sha256(content).hexdigest()
    report = {"schema": SCHEMA, "weights_sha256": sha, "binding": key, "order": order["order_id"], "seed": args.seed,
              "parameters_per_head": sum(w.size for w in heads[0]), "training_pairs_s": train, "holdout_pairs_s": holdout,
              "base_commit": order["base_commit"], "trial_result_sha256": restart_trial.native.checksum(trial / "result.json"),
              **scored, "training_cpu_wall_s": time.perf_counter() - start}
    # CPU execution time is a measured observation, so preserve the first fit on replay.
    if args.report.exists():
        saved = restart_trial.read(args.report)
        if saved["weights_sha256"] != sha or saved["trial_result_sha256"] != report["trial_result_sha256"]:
            raise ValueError("previous fit does not match target observations")
        return saved
    write_once(args.report, report)
    return report


def select(checkpoint, receipt, *, family, cache_ready, active, previous=False, ablate_weights=False, expected_sha256):
    """One CPU-only dispatch on a verified control receipt. Unknown or withdrawn profiles use the baseline."""
    start = time.perf_counter_ns()
    reason = None
    try:
        data = Path(checkpoint).read_bytes()
        if len(data) > 1 << 20 or hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ValueError("checkpoint digest differs")
        q = receipt["qualification"]
        if receipt["adapter"] not in ("qwen-restart", "qwen-ordered-pk2") or q["passed"] is not True:
            raise ValueError("unqualified restart release")
        with np.load(io.BytesIO(data), allow_pickle=False) as saved:
            if str(saved["schema"]) != SCHEMA or str(saved["scope"]) != binding(receipt["order"], q):
                raise ValueError("different target, receipt or source")
            tag = "prior" if previous or ablate_weights else "head"
            heads = [[saved[f"{tag}{i}_{k}"] for k in PARAMS] for i in range(3)]
            margin = float(saved["margin"])
            supported = family == "mixed-first-32" or (family == "factual-first-18" and bool(saved["holdout_admitted"]))
        eligible = NAMES if cache_ready is True and active is True else ("reference",)
        result = Pilot.startup_profile(heads, features(family), NAMES, eligible, "reference",
                                       in_support=supported and not previous, margin_error=margin)
    except (KeyError, ValueError, OSError, TypeError, EOFError) as exc:
        reason = str(exc)
        result = {"profile": "reference", "reason": "unavailable qualified startup model", "scores": [], "cpu_us": 0.0}
    return {**result, "detail": reason, "family": family, "previous": previous, "ablate_weights": ablate_weights, "weights_sha256": expected_sha256,
            "cpu_ms": (time.perf_counter_ns() - start) / 1e6}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("order", "checkout", "trial", "candidate-source", "weights", "report"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--seed", default=18471, type=int)
    print(json.dumps(fit(p.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
