"""Calibrate a real reference head once, then independently requalify its exact regime in every arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from types import MethodType, SimpleNamespace


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, payload):
    with Path(path).open("x") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")


def hooks():
    from ironmule import ggml_engine
    from ironmule import pilot_learning as pl
    from training import pilot_threads as pt
    from training import serving_trial as st

    return ggml_engine, pl, pt, st


def instrument(probe, records):
    def recorded(engine, context, base, threads, seed, interrupted):
        result = probe(engine, context, base, threads, seed, interrupted)
        if result is not None and seed:
            seconds, tokens = result
            records.append(
                {
                    "context": context,
                    "base": base,
                    "threads": threads,
                    "seed": seed,
                    "seconds": seconds,
                    "tokens_sha256": hashlib.sha256(
                        json.dumps(tokens).encode()
                    ).hexdigest(),
                }
            )
        return result

    return recorded


def measured_margin(key, offset, observations):
    band, base, _, _ = map(int, key.split(":"))
    candidate = base + offset
    samples = observations[-12:]
    if len(samples) != 12:
        raise ValueError("incomplete native requalification")
    ratios, noise, margin = [], [], []
    for i in range(4):
        trio = samples[3 * i : 3 * i + 3]
        ar, br, repeat = trio if i % 2 == 0 else [trio[1], trio[0], trio[2]]
        if (
            [ar["threads"], br["threads"], repeat["threads"]] != [base, candidate, base]
            or any(
                x["base"] != base
                or x["context"] != 64 * band + 4 + 13 * i
                or x["seed"] != i + 1
                for x in trio
            )
            or len({x["tokens_sha256"] for x in trio}) != 1
            or any(not math.isfinite(x["seconds"]) or x["seconds"] <= 0 for x in trio)
        ):
            raise ValueError("unmatched native thread, context, output or work")
        mid = (ar["seconds"] + repeat["seconds"]) / 2
        delta = abs(ar["seconds"] - repeat["seconds"])
        ratios.append(br["seconds"] / mid)
        noise.append(delta / mid)
        margin.append((mid - br["seconds"] - delta) * 1e6 / 8)
    if (
        max(ratios) >= 0.97
        or 1 - statistics.median(ratios) <= max(noise)
        or min(margin) <= 0
    ):
        raise ValueError("native holdout did not meet its existing thresholds")
    return {
        "windows": 4,
        "candidate_threads": candidate,
        "max_ratio": max(ratios),
        "max_aa_fraction": max(noise),
        "min_gain_after_noise_us_per_step": min(margin),
    }


def verify_profile(profile, order):
    b = profile["binding"]
    head = b["scratch_thread_head"]
    if (
        profile["schema"] != "ironmule.cpu-matched-profile.v1"
        or profile["source_order_id"] != order["order_id"]
        or profile["calibrator_sha256"] != sha(__file__)
        or profile["reference_correct"] != 16
        or profile["reference_scratch_uses"] < 1
        or profile["runtime_source_sha256"]["ggml_engine.py"]
        != "aae9299715c89437116ac8aa6f9d65c18149ca148a0ba18878b29eaa3ba718b5"
        or b["provider"] != "cpu"
        or b["model_sha256"] != profile["model_sha256"]
        or head["weights_sha256"] != profile["head_sha256"]
        or head["learner_binding"] != profile["learner_binding"]
        or head["qualified_offsets"] != profile["weights"]
        or any(
            b[k] != v
            for k, v in order["binding"].items()
            if k not in ("env_sha256", "scratch_thread_head")
        )
        or any(
            head[k] != v
            for k, v in order["binding"]["scratch_thread_head"].items()
            if k != "qualified_offsets"
        )
        or set(profile["costs"]) != {"step_one", "row", "prefill_1k", "kv_1k"}
        or any(not math.isfinite(c) or c <= 0 for c in profile["costs"].values())
        or not profile["weights"]
    ):
        raise ValueError("invalid measured common CPU profile")
    for key, offset in profile["weights"].items():
        if (
            profile["visits"][key] < 64
            or measured_margin(key, offset, profile["probes"][key])
            != profile["margins"][key]
        ):
            raise ValueError("calibration does not reproduce its native evidence")


def capture(args):
    ge, pl, pt, st = hooks()
    order = read(args.order)
    before, captured, raw = {}, {}, []
    original_probe = pl.ThreadLearning._probe
    original_idle = pl.ThreadLearning.idle
    original_qualify = pt.qualify_candidate
    data = {}

    def idle(learning, interrupted):
        snapshot = dict(learning.visits)
        raw.clear()
        result = original_idle(learning, interrupted)
        if learning.last and learning.last.get("result") == "qualified":
            key = learning.last["key"]
            if snapshot.get(key, 0) >= pl.MIN_STEPS:
                before[key] = snapshot[key]
                captured[key] = list(raw[-12:])
        return result

    def qualify(engine, *a, **kw):
        start = dict(engine.pilot.costs)
        result = original_qualify(engine, *a, **kw)
        learning = engine.pilot.learning
        data.update(
            costs=start,
            weights=dict(learning.weights),
            margins=dict(learning.margins),
            visits=dict(before),
            probes=dict(captured),
            receipt=result,
            learner_binding=learning.binding,
        )
        return result

    pl.ThreadLearning._probe = instrument(original_probe, raw)
    pl.ThreadLearning.idle, pt.qualify_candidate = idle, qualify
    try:
        row = st.measure(options(args, order, seed=17351))
    finally:
        pl.ThreadLearning._probe = original_probe
        pl.ThreadLearning.idle, pt.qualify_candidate = original_idle, original_qualify
    b = row["binding"]
    if (
        b["provider"] != "cpu"
        or not data["weights"]
        or not row["scratch_measured_decisions"]
        or set(data["weights"]) != set(data["margins"])
        or set(data["weights"]) != set(data["visits"])
        or row["completed"] != row["correct"]
        or row["runtime_source_sha256"]["ggml_engine.py"]
        != order["observations"].get("reference_source_sha256", sha(ge.__file__))
        or any(
            b[k] != v
            for k, v in order["binding"].items()
            if k not in ("env_sha256", "scratch_thread_head")
        )
        or any(
            b["scratch_thread_head"][k] != v
            for k, v in order["binding"]["scratch_thread_head"].items()
            if k != "qualified_offsets"
        )
    ):
        raise ValueError(
            "reference did not produce a complete target-bound live calibration"
        )
    profile = {
        "schema": "ironmule.cpu-matched-profile.v1",
        "source_order_id": order["order_id"],
        "runtime_source_sha256": row["runtime_source_sha256"],
        "evaluator_sha256": row["evaluator_sha256"],
        "calibrator_sha256": sha(__file__),
        "binding": b,
        "model_sha256": sha(args.gguf),
        "head_sha256": sha(args.head),
        "reference_scratch_uses": row["scratch_measured_decisions"],
        "reference_correct": row["correct"],
        **data,
    }
    verify_profile(profile, order)
    return profile


def options(args, order, seed):
    return SimpleNamespace(
        gguf=args.gguf,
        threads=2,
        clients=4,
        seed=seed,
        tokens=3,
        waves=4,
        revision=order["base_commit"],
        require_provider="cpu",
        thread_candidate=args.head,
        profile_dir=None,
    )


def requalify(engine, gguf, head, state, profile):
    ge, pl, _, _ = hooks()
    started = time.monotonic()
    engine.pilot.costs = dict(profile["costs"])
    engine.pilot.last_step = engine.pilot.costs["step_one"]
    learning = pl.ThreadLearning(
        engine, state, sha(gguf), sha(engine.pilot.weights_path)
    )
    if (
        learning.binding != profile["learner_binding"]
        or ge.machine_key() != profile["binding"]["machine"]
        or sha(head) != profile["head_sha256"]
        or sha(gguf) != profile["model_sha256"]
    ):
        raise ValueError("different local thread-profile binding")
    learning._load_candidate(head)
    if learning.thread_candidate is None:
        raise ValueError("head unavailable")
    engine.pilot.learning = learning
    # Replay the *recorded visit counts* from the actual fresh calibration. They trigger a trial
    # of the one saved proposal. Only the original native gate can add local weights and margins.
    pending_history = dict(profile["weights"])
    fresh = {}
    for key, offset in pending_history.items():
        count = profile["visits"][key]
        if count < pl.MIN_STEPS:
            raise ValueError("no observed demand for this regime")
        learning.visits.clear()
        learning.visits[key] = count
        learning.proposals[key] = offset
        # The world/load sensor needs two observations separated by its native sampling interval.
        engine.pilot.load.update()
        time.sleep(0.06)
        raw = []
        probe = learning._probe
        learning._probe = MethodType(instrument(type(learning)._probe, raw), learning)
        try:
            learning.idle(lambda: False)
        finally:
            learning._probe = probe
        if (
            learning.weights.get(key) != offset
            or (learning.last or {}).get("result") != "qualified"
        ):
            raise ValueError(
                f"fresh native requalification did not admit {key}: {learning.last}"
            )
        fresh[key] = {
            "margin": dict(learning.margins[key]),
            "receipt": dict(learning.last),
            "probes": raw,
        }
    learning.visits.clear()
    learning.margins = {k: dict(v) for k, v in profile["margins"].items()}
    ordinary = learning.choose

    def choose(ctx, foreign, base, rows=1):
        picked = ordinary(ctx, foreign, base, rows)
        if picked != base:
            key = learning.last_regime
            if (
                learning.last_thread_choice[5]
                >= fresh[key]["margin"]["min_gain_after_noise_us_per_step"]
            ):
                # Both the common published cost/uncertainty allowance and the child's observed
                # allowance must pay for the real network invocation.
                learning.candidate_status["qualified_decisions"] -= 1
                learning.candidate_status["reference_decisions"] += 1
                learning.last_thread_choice = (
                    key,
                    base,
                    base,
                    rows,
                    "fresh local cost veto",
                    learning.last_thread_choice[5],
                    fresh[key]["margin"],
                )
                return base
        return picked

    learning.choose = choose
    b = {
        "learner_binding": learning.binding,
        "weights_sha256": sha(head),
        "qualified_offsets": dict(learning.weights),
    }
    return {
        "binding": b,
        "status": learning.status(),
        "search_and_warmup_s": time.monotonic() - started,
        "fresh": fresh,
        "traces": [],
        "idle_results": [s["receipt"] for s in fresh.values()],
    }


def verify_arm(row, profile, profile_sha, mode, order):
    verify_profile(profile, order)
    audit = row["matched_profile"]
    if (
        audit["profile_sha256"] != profile_sha
        or audit["published_costs"] != profile["costs"]
        or audit["published_offsets"] != profile["weights"]
        or audit["published_margins"] != profile["margins"]
        or row["binding"]["scratch_thread_head"]
        != profile["binding"]["scratch_thread_head"]
        or row["binding"]["scratch_thread_head"]["qualified_offsets"]
        != profile["weights"]
        or row["evaluator_sha256"] != profile["evaluator_sha256"]
        or audit["mode"] != mode
        or audit["matched_helper_sha256"] != profile["calibrator_sha256"]
        or set(audit["fresh"]) != set(profile["weights"])
    ):
        raise ValueError("untied calibration or evaluator")
    for key, observed in audit["fresh"].items():
        n = observed["margin"]
        rec = observed["receipt"]
        if (
            rec["result"] != "qualified"
            or rec["key"] != key
            or measured_margin(key, profile["weights"][key], observed["probes"]) != n
            or n["windows"] != 4
            or not all(math.isfinite(v) and v > 0 for v in rec["ratios"])
            or len(rec["ratios"]) != 4
            or len(rec["aa_spread"]) != 4
            or max(rec["ratios"]) >= 0.97
            or n["min_gain_after_noise_us_per_step"] <= 0
        ):
            raise ValueError("missing fresh evidence for a published action")
    if (mode == "rule") != (row["scratch_measured_decisions"] == 0):
        raise ValueError("missing actual scratch decision or four-thread challenger")


def arm(args):
    _, _, pt, st = hooks()
    order, profile = read(args.order), read(args.profile)
    if (
        profile["source_order_id"] != order["order_id"]
        or profile["calibrator_sha256"] != sha(__file__)
        or sha(pt.__file__)
        != profile["binding"]["scratch_thread_head"]["helper_sha256"]
    ):
        raise ValueError("profile and original controlled experiment differ")
    original = pt.qualify_candidate

    def fixed(engine, gguf, weights, state, **_):
        return requalify(engine, gguf, weights, state, profile)

    pt.qualify_candidate = fixed
    try:
        opts = options(args, order, args.seed)
        if args.mode == "rule":
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from cpu_four_thread_rule import measure

            row = measure(opts)
        else:
            row = st.measure(opts)
    finally:
        pt.qualify_candidate = original
    row["matched_profile"] = {
        "profile_sha256": sha(args.profile),
        "matched_helper_sha256": sha(__file__),
        "mode": args.mode,
        "published_costs": profile["costs"],
        "published_offsets": profile["weights"],
        "published_margins": profile["margins"],
        "fresh": row["scratch_qualification"]["fresh"],
    }
    verify_arm(row, profile, sha(args.profile), args.mode, order)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=("calibrate", "arm"))
    for name in ("order", "gguf", "head", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--mode", choices=("reference", "candidate", "rule"))
    p.add_argument("--profile", type=Path)
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    if args.operation == "arm" and not (args.mode and args.profile and args.seed):
        p.error("arm needs mode, profile, seed")
    write(args.out, capture(args) if args.operation == "calibrate" else arm(args))


if __name__ == "__main__":
    main()
