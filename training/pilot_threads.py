"""Collect comparable CPU plain-decode profiles, fit the Pilot thread head and replay grouped holdouts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import statistics
import time

import numpy as np

from ironmule.pilot import OBS, THREAD_FRACTIONS, predict


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def acquire(args):
    import platform
    import subprocess

    from ironmule.ggml_engine import GgmlEngine, file_sha256
    from ironmule.pilot_learning import ThreadLearning

    os.environ["IRONMULE_GGML_GPU_LAYERS"] = "0"
    e = GgmlEngine(str(args.gguf), seqs=args.slots, seq_ctx=512, speculate=False)
    try:
        p = e.pilot
        if not p or e.gpu_layers:
            raise ValueError("collection requires a live CPU Pilot")
        learning = ThreadLearning(e, args.state, file_sha256(str(args.gguf)), file_sha256(str(p.weights_path)))
        # All arms share a prompt, decode positions, reference, host and binary. Keep both orders.
        record = {"schema": 1, "binding": learning.binding, "profile": {
            **learning.policy.profile, "gguf": args.gguf.name, "platform": platform.platform(),
            "numpy": np.__version__, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "cache": "idle slots removed after each arm", "backend": "ggml-cpu", "cost_usd_incremental": 0,
        }, "groups": []}
        if args.extend:
            previous = json.loads(args.trials.read_text())
            if previous["binding"] != record["binding"]:
                raise ValueError("previous trials belong to another local engine")
            record = previous
        elif args.trials.exists():
            raise FileExistsError(f"pass --extend to add comparable trials to {args.trials}")
        pass_index = 1 + max((g.get("pass", 0) for g in record["groups"]), default=-1)
        rng = random.Random(9021 + pass_index)
        cases = [(band, rows) for band in range(1, 8) for rows in (1, 2)]
        rng.shuffle(cases)
        for band, rows in cases:
            context = band * 64 + 8
            features = p.features(rows, 0, 0, context)
            base = p.decide(rows, 0, 0, context)["threads"]
            load = min(p.cores, int(p.load.update() + 0.5))
            learning.replay_load, learning.replay_rows = load, rows
            group = {"group": f"ctx-bin-{band}", "pass": pass_index, "step_costs": p.costs, "context": context, "rows": rows, "foreign": load,
                     "base": base, "features": features.tolist(), "observations": []}
            actions = sorted({p.threads_for(i) for i in range(len(THREAD_FRACTIONS))})
            rng.shuffle(actions)
            # The first and reversed repeats counterbalance order. A partial series stays diagnostic only.
            for repetition, order in enumerate((actions, actions[::-1])):
                for action in order:
                    result = learning._probe(context, base, action, band * 7 + rows, lambda: False)
                    seconds, digest = None, None
                    if result:
                        seconds, tokens = result
                        digest = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
                    group["observations"].append({"threads": action, "seconds": seconds, "tokens_digest": digest,
                                                  "repetition": repetition})
            record["groups"].append(group)
            save(args.trials, record)
            print(group["group"], rows, flush=True)
    finally:
        e.close()


def examples(record):
    cores = record["profile"]["cores"]
    choices = [max(1, round(cores * f)) for f in THREAD_FRACTIONS]
    for group in record["groups"]:
        obs = group["observations"]
        tokens = {r["tokens_digest"] for r in obs}
        if len(tokens) != 1 or None in tokens:
            continue
        timings = {}
        for n in set(choices):
            repeats = [r for r in obs if r["threads"] == n]
            if len(repeats) != 2 or {r["repetition"] for r in repeats} != {0, 1}:
                break
            times = [r["seconds"] for r in repeats]
            if not all(s is not None and math.isfinite(s) and s > 0 for s in times):
                break
            timings[n] = statistics.mean(times)
        else:
            if group["base"] in timings:
                yield group, timings, choices


def train(rows, seed):
    """From random matrices, fit the same two tanh layers as the shipped Pilot to measured log speedups."""
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, OBS ** -0.5, (128, OBS))
    w2 = rng.normal(0, 128 ** -0.5, (128, 128))
    w3 = rng.normal(0, 128 ** -0.5, (len(THREAD_FRACTIONS), 128))
    b1, b2, b3 = np.zeros(128), np.zeros(128), np.zeros(len(w3))
    weights = [w1, b1, w2, b2, w3, b3]
    x = np.array([g["features"] for g, _, _ in rows])
    y = np.array([[math.log(t[g["base"]] / t[n]) for n in choices] for g, t, choices in rows])
    # Treat overlapping repeats as unresolved. They may be replayed descriptively, but their
    # timing fluctuations cannot teach a positive or negative reward for the alternative.
    mask = []
    for group, _, choices in rows:
        by_threads = {n: [r["seconds"] for r in group["observations"] if r["threads"] == n] for n in choices}
        ref = by_threads[group["base"]]
        mask.append([n == group["base"] or min(ref) > max(by_threads[n]) or max(ref) < min(by_threads[n])
                     for n in choices])
    mask = np.asarray(mask, dtype=bool)
    for _ in range(220):
        h1 = np.tanh(x @ w1.T + b1)
        h2 = np.tanh(h1 @ w2.T + b2)
        residual = mask * (h2 @ w3.T + b3 - y) / len(x)
        d2 = (residual @ w3) * (1 - h2 ** 2)
        d1 = (d2 @ w2) * (1 - h1 ** 2)
        grads = [d1.T @ x, d1.sum(0), d2.T @ h1, d2.sum(0), residual.T @ h2, residual.sum(0)]
        for param, grad in zip(weights, grads):
            param -= 0.008 * grad
    return [w.astype(np.float32) for w in weights]


def action(weights, group, choices):
    logits = predict(weights, group["features"])
    return choices[int(np.argmax(logits))]


def fit(args):
    record = json.loads(args.trials.read_text())
    data = list(examples(record))
    # Partition entire context families; neighboring prompts, both widths and arm repetitions stay together.
    first = [r for r in data if r[0]["context"] // 64 in (1, 2, 3)]
    new = [r for r in data if r[0]["context"] // 64 in (4, 5)]
    held = [r for r in data if r[0]["context"] // 64 in (6, 7)]
    if not first or not new or not held:
        raise ValueError("need complete matched pairs in each group split")
    fit_started = time.perf_counter()
    before = train(first, args.seed)
    weights = train(first + new, args.seed)
    fit_seconds = time.perf_counter() - fit_started
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    with args.weights.open("wb") as f:
        np.savez(f, **dict(zip(("w1", "b1", "w2", "b2", "w3", "b3"), weights)),
                 binding=record["binding"], version=1)
    changed = []
    for group, _, choices in first + new:
        previous = action(before, group, choices)
        selected = action(weights, group, choices)
        if selected != previous:
            changed.append({"group": group["group"], "rows": group["rows"], "base": group["base"],
                            "pass": group.get("pass", 0), "before": previous, "after": selected})
    result = []
    for group, timings, choices in held:
        selected = action(weights, group, choices)
        previous = action(before, group, choices)
        result.append({"group": group["group"], "rows": group["rows"], "selected": selected, "before": previous,
                       "pilot": group["base"], "scores_before": predict(before, group["features"]).tolist(),
                       "scores_after": predict(weights, group["features"]).tolist(),
                       "seconds": {"candidate": timings[selected], "shipped_pilot": timings[group["base"]],
                                   "one_thread": timings[1], "all_threads": timings[max(choices)]}})
    save(args.report, {"seed": args.seed, "backend": "numpy-cpu", "both_fits_seconds": fit_seconds, "parameters": sum(w.size for w in weights),
                       "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                       "trials_sha256": hashlib.sha256(args.trials.read_bytes()).hexdigest(),
                       "train_groups": sorted({r[0]["group"] for r in first + new}),
                       "excluded_groups_or_widths": len(record["groups"]) - len(data),
                       "new_experience_groups": sorted({r[0]["group"] for r in new}),
                       "training_decisions_changed": changed, "heldout": result, "interpretation": "Measured decode replay only; local qualification required."})


def busy(stop):
    block = b"contention fixture" * 4096
    while not stop.is_set():
        hashlib.sha256(block).digest()


def request(engine, prompt):
    from ironmule.ggml_engine import Job

    job = Job(prompt, 48)
    start = time.monotonic_ns()
    engine.run([job])
    return {"tokens": len(job.tokens), "reason": job.reason, "seconds": (time.monotonic_ns() - start) / 1e9,
            "ttft_s": (job.first_ns - start) / 1e9,
            "digest": hashlib.sha256(json.dumps(job.tokens).encode()).hexdigest()}


def qualify_candidate(engine, gguf: Path, weights: Path, state: Path, *, prompt_tokens=97):
    """Attach the checkpoint and freshly qualify its proposals on this real CPU worker before measurement."""
    from ironmule.ggml_engine import file_sha256
    from ironmule.pilot_learning import ThreadLearning

    pilot = engine.pilot
    if not pilot or engine.gpu_layers:
        raise ValueError("candidate needs a live CPU Pilot")
    learning = ThreadLearning(engine, state, file_sha256(str(gguf)), file_sha256(str(pilot.weights_path)))
    learning._load_candidate(Path(weights))
    if learning.thread_candidate is None:
        raise ValueError(f"unable to run candidate: {learning.candidate_status}")
    pilot.learning = learning
    ids = engine.tokenize("Continue the sequence by counting from one to ten.")
    prompt = (ids * (prompt_tokens // len(ids) + 1))[:prompt_tokens]
    traces = []
    search_started = time.monotonic()
    previous_reuse = engine.prefix_reuse
    engine.prefix_reuse = False
    try:
        for _ in range(12):
            traces.append(request(engine, prompt))
            learning.idle(lambda: False)
            if learning.weights:
                break
    finally:
        engine.prefix_reuse = previous_reuse
    return {"binding": {"learner_binding": learning.binding, "weights_sha256": file_sha256(str(weights)),
                        "qualified_offsets": dict(learning.weights)}, "status": json.loads(json.dumps(learning.status())),
            "search_and_warmup_s": time.monotonic() - search_started, "traces": traces}


def exercise(args):
    """Drive finite real engine requests through the shipped Pilot and the opt-in candidate's idle gate."""
    from ironmule.ggml_engine import GgmlEngine

    os.environ["IRONMULE_GGML_GPU_LAYERS"] = "0"
    engine = GgmlEngine(str(args.gguf), seqs=args.slots, seq_ctx=512, speculate=False)
    # Start the controlled foreign workload after native startup, as a serving load transition.
    ctx = multiprocessing.get_context("spawn")
    stop = ctx.Event()
    workers = []
    try:
        for _ in range(args.foreign_workers):
            worker = ctx.Process(target=busy, args=(stop,), daemon=True)
            worker.start()
            workers.append(worker)
        receipt = qualify_candidate(engine, args.gguf, args.weights, args.state, prompt_tokens=args.prompt_tokens)
        pilot = engine.pilot
        learning = pilot.learning
        engine.prefix_reuse = False
        ids = engine.tokenize("Continue the sequence by counting from one to ten.")
        prompt = (ids * (args.prompt_tokens // len(ids) + 1))[:args.prompt_tokens]
        # Same completed work, counterbalanced, with the actual decision and thread switching charged.
        pairs = []
        before = dict(learning.visits)
        for arm in ("shipped_pilot", "candidate_gate", "candidate_gate", "shipped_pilot"):
            pilot.learning = learning if arm == "candidate_gate" else None
            pairs.append({"arm": arm, **request(engine, prompt)})
        pilot.learning = learning
        active_calls = {key: learning.visits[key] - before.get(key, 0) for key in learning.weights}
        save(args.report, {"binding": learning.binding, "foreign_workers": len(workers), "prompt_tokens": len(prompt),
                           "search_and_warmup_s": receipt["search_and_warmup_s"],
                           "profile_binding": receipt["binding"],
                           "search_and_warmup": receipt["traces"], "qualification": receipt["status"], "paired_runs": pairs,
                           "qualified_thread_calls": active_calls, "pilot": pilot.status(),
                           "all_reference_answers_match": len({(t["digest"], t["tokens"], t["reason"])
                                                               for t in pairs}) == 1})
    finally:
        stop.set()
        for worker in workers:
            worker.join(timeout=3)
            if worker.is_alive():
                worker.terminate()
                worker.join()
        engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    collection = sub.add_parser("collect")
    collection.add_argument("--slots", type=int, choices=(2, 4), default=2)
    collection.add_argument("--extend", action="store_true")
    collection.add_argument("--gguf", type=Path, required=True)
    collection.add_argument("--trials", type=Path, required=True)
    collection.add_argument("--state", type=Path, required=True)
    fit_parser = sub.add_parser("fit")
    for key in ("trials", "weights", "report"):
        fit_parser.add_argument("--" + key, type=Path, required=True)
    fit_parser.add_argument("--seed", type=int, default=20261126)
    live = sub.add_parser("exercise")
    live.add_argument("--slots", type=int, choices=(2, 4), default=2)
    live.add_argument("--prompt-tokens", type=int, default=97, choices=range(64, 448), metavar="64..447")
    live.add_argument("--foreign-workers", type=int, choices=(0, 1, 2), default=0)
    for key in ("gguf", "state", "weights", "report"):
        live.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "collect":
        import fcntl

        args.trials.parent.mkdir(parents=True, exist_ok=True)
        with args.trials.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquire(args)
    elif args.cmd == "fit":
        fit(args)
    else:
        exercise(args)


if __name__ == "__main__":
    main()
