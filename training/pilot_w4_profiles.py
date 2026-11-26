"""Train the existing CPU Pilot's startup profile head from one local W4/width calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np

from ironmule.pilot import Pilot, predict
from lab.pk1 import native_trial as trial
from lab.portable_gpu.bench import save
from training.pilot_w4_probe import CONTEXTS
from training.pilot_w4_probe import WIDTHS as STREAM_WIDTHS

NAMES = ("stock", "q4", "q4_q6")
WIDTHS = (*range(1, 9), 16, 24, 32)
FIELDS = ("step_ms_off", "step_ms_on", "step_ms_q6k")
PARAMS = ("w1", "b1", "w2", "b2", "w3", "b3")


def read(path):
    return json.loads(Path(path).read_bytes())


def features(width):
    # For the startup evidence these are consecutive token rows on one sequence.
    return np.asarray([width / 32, math.log2(width) / 5], dtype=np.float32)


def sources(order_path, receipt_path):
    order, receipt = read(order_path), read(receipt_path)
    oid = hashlib.sha256(json.dumps({k: v for k, v in order.items() if k != "order_id"}, sort_keys=True).encode())
    b = order["binding"]
    calibration = receipt["binding"]
    if (oid.hexdigest() != order["order_id"] or trial.checksum(order_path.with_suffix(".profile.json")) != order["profile_sha256"]
            or receipt["outcome"] != "built" or receipt["binary"] != b["w4_sha256"] or receipt["table"] != b["table_sha256"]
            or trial.checksum(receipt_path) != b["tuning_receipt_sha256"]
            or any(b[k] != v for k, v in calibration.items())):
        raise ValueError("unbound tuning receipt or profile order")
    mask = b["mask"]
    # These three complete local modes coincide with the measured startup arms only on these exact masks.
    if set(mask["tokens_served"]) != set(WIDTHS) or set(mask["q6k_tokens_served"]) != set(WIDTHS):
        raise ValueError("this experimental lane requires both masks locally admitted at every calibrated width")
    times = {w: [float(mask[field][str(w)]) for field in FIELDS] for w in WIDTHS}
    if any(not math.isfinite(v) or v <= 0 for values in times.values() for v in values):
        raise ValueError("invalid complete-step times")
    return order, receipt["measurement"], times


def fit_head(times, widths, seed, inputs=None, epochs=550):
    rng = np.random.default_rng(seed)
    x = np.stack([features(w) if inputs is None else inputs[w] for w in widths])
    # Positive scores represent the log speedup of the complete profile versus stock.
    y = np.array([[math.log(times[w][0] / t) for t in times[w]] for w in widths])
    hidden = 16
    n = x.shape[1]
    weights = [rng.normal(0, n ** -.5, (hidden, n)), np.zeros(hidden),
               rng.normal(0, hidden ** -.5, (hidden, hidden)), np.zeros(hidden),
               rng.normal(0, hidden ** -.5, (y.shape[1], hidden)), np.zeros(y.shape[1])]
    w1, b1, w2, b2, w3, b3 = weights
    for _ in range(epochs):
        a = np.tanh(x @ w1.T + b1)
        b = np.tanh(a @ w2.T + b2)
        out = (b @ w3.T + b3 - y) / len(widths)
        db = (out @ w3) * (1 - b ** 2)
        da = (db @ w2) * (1 - a ** 2)
        grads = (da.T @ x, da.sum(0), db.T @ a, db.sum(0), out.T @ b, out.sum(0))
        for param, grad in zip(weights, grads, strict=True):
            param -= .025 * grad
    return [v.astype(np.float32) for v in weights]


def fit(args):
    started = time.perf_counter()
    order, kernel, times = sources(args.order, args.receipt)
    if args.contexts:
        return fit_contexts(args, order)
    first, new, calibration, test = list(range(1, 9)), [16], 24, 32
    seeds = [args.seed + i for i in range(3)]
    before = [fit_head(times, first, s) for s in seeds]
    heads = [fit_head(times, first + new, s) for s in seeds]
    # Widths sharing the same 8-row tile have never been split across subsets.
    # One independent calibration tile estimates uncertainty. Report its size explicitly.
    errors = []
    for head in heads:
        score = predict(head, features(calibration))
        best, nxt = np.argsort(-score)[:2]
        actual_margin = math.log(times[calibration][nxt] / times[calibration][best])
        errors.append(abs(float(score[best] - score[nxt]) - actual_margin))
    uncertainty = max(errors)
    decision = Pilot.startup_profile(heads, features(test), NAMES, NAMES, "q4_q6", in_support=True, margin_error=uncertainty)
    args.weights.parent.mkdir(exist_ok=True, parents=True)
    with args.weights.open("wb") as out:
        np.savez(out, **{f"head{i}_{key}": p for i, head in enumerate(heads) for key, p in zip(PARAMS, head, strict=True)},
                 order=order["order_id"], calibration_error=uncertainty, serving_seed=-1,
                 serving_features=features(32))
    raw = kernel["measured"]
    save(args.report, {"schema": "ironmule.w4-profile-fit.v1", "order": order["order_id"], "seed": args.seed,
                       "profile_sha256": order["profile_sha256"], "tuner_receipt_sha256": order["binding"]["tuning_receipt_sha256"],
                       "gguf_sha256": order["binding"]["gguf"], "gpu": order["binding"]["fp"]["gpu"],
                       "native_base": order["base_commit"], "step_ms_by_rows": {
                           str(w): dict(zip(NAMES, times[w], strict=True)) for w in WIDTHS},
                       "weights_sha256": trial.checksum(args.weights), "cpu_train_s": time.perf_counter() - started,
                       "train_widths": first, "new_widths": new, "margin_calibration_widths": [calibration],
                       "holdout_widths": [test], "parameters_per_head": sum(p.size for p in heads[0]),
                       "scores_before_new_experience": [predict(h, features(test)).tolist() for h in before],
                       "scores_after_new_experience": decision["scores"], "holdout": decision,
                       "measured_holdout_ms": dict(zip(NAMES, times[test], strict=True)),
                       "shipped_pilot_with_local_mask_ms": times[test][2], "fixed_stock_ms": times[test][0],
                       "complete_step_rows": len(times), "training_target": "log speedup of one whole native call",
                       "geometry": {"sequences": 1, "prefix_tokens": 64, "rows": "successive positions"},
                       "winner_labels": sorted({NAMES[int(np.argmin(t))] for t in times.values()}),
                       "kernel_attempts_diagnostic_only": len(raw),
                       "kernel_rejections": sum(r["rejection"] is not None for r in raw),
                       "uncertainty": "Four tile groups on one GPU, model and calibration. One calibration group. "
                                      "Held-out step medians do not certify a serving variant."})


def fit_contexts(args, order):
    """Learn a separate independent-stream head, keeping all repeats and widths of a context family together."""
    archive = read(args.contexts)
    names = {f"{i}-{name}.json" for i in (0, 1) for name in NAMES}
    if (archive["schema"] != "ironmule.w4-stream-contexts.v1" or archive["order"] != order["order_id"]
            or set(archive["source_sha256"]) != names
            or any(trial.checksum(args.contexts.parent / n) != d for n, d in archive["source_sha256"].items())):
        raise ValueError("incomplete local context collection")
    reports = [read(args.contexts.parent / n) for n in sorted(names)]
    if any(r["source_sha256"] != trial.checksum(Path(__file__).with_name("pilot_w4_probe.py"))
           or r["spec"]["order"] != order["order_id"] for r in reports):
        raise ValueError("context collection uses a different order or protocol")
    started = time.perf_counter()
    times, inputs = {}, {}
    for cell in archive["groups"]:
        c, w = cell["context"], cell["width"]
        if w == 1:  # the synthetic short lane is identical at every context label
            continue
        observed = [next(row for row in run["cells"] if (row["context"], row["width"]) == (c, w)) for run in reports]
        if any(any(r[k] != observed[0][k] for k in ("tokens", "warm_tokens", "prompt_lengths")) for r in observed):
            continue
        durations = [[r["seconds"] for r, run in zip(observed, reports, strict=True) if run["spec"]["name"] == name] for name in NAMES]
        if any(len(d) != 2 or not all(math.isfinite(t) and t > 0 for t in d) for d in durations):
            raise ValueError("invalid paired context measurements")
        medians = [float(np.mean(d)) for d in durations]
        winner = int(np.argmin(medians))
        if any(max(durations[winner]) >= min(d) for i, d in enumerate(durations) if i != winner):
            continue
        key = (c, w)
        lengths = observed[0]["prompt_lengths"]
        inputs[key] = np.asarray([w / 32, np.mean(lengths) / 512, (max(lengths) - min(lengths)) / 512], dtype=np.float32)
        times[key] = medians
    first = [k for k in times if k[0] in ("32", "96")]
    added = [k for k in times if k[0] == "224"]
    calibration = [k for k in times if k[0] == "512"]
    held = [k for k in times if k[0] == "serving_mix"]
    target = ("serving_mix", 32)
    if not all((first, added, calibration, held)) or target not in held:
        raise ValueError("insufficient output-matched context groups or unresolved holdout")
    seeds = range(args.seed, args.seed + 3)
    before = [fit_head(times, first, s, inputs) for s in seeds]
    heads = [fit_head(times, first + added, s, inputs) for s in seeds]
    errors = []
    for h in heads:
        for k in calibration:
            score = predict(h, inputs[k])
            best, runner_up = np.argsort(-score)[:2]
            errors.append(abs(float(score[best] - score[runner_up]) - math.log(times[k][runner_up] / times[k][best])))
    uncertainty = max(errors)
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    with args.weights.open("wb") as out:
        np.savez(out, **{f"head{i}_{key}": p for i, head in enumerate(heads) for key, p in zip(PARAMS, head, strict=True)},
                 order=order["order_id"], calibration_error=uncertainty, serving_seed=92741, serving_features=inputs[target])
    holdout = [{"group": c, "width": w, "complete_native_seconds": dict(zip(NAMES, times[c, w], strict=True)),
                "decision": Pilot.startup_profile(heads, inputs[c, w], NAMES, NAMES, "q4_q6",
                                                   in_support=True, margin_error=uncertainty)} for c, w in held]
    save(args.report, {"schema": "ironmule.w4-stream-profile-fit.v1", "order": order["order_id"], "seed": args.seed,
                       "sources_sha256": archive["source_sha256"], "weights_sha256": trial.checksum(args.weights),
                       "train_groups": first, "added_groups": added, "uncertainty_groups": calibration, "holdout": holdout,
                       "scores_before": [predict(h, inputs[target]).tolist() for h in before],
                       "scores_after": [predict(h, inputs[target]).tolist() for h in heads],
                       "cpu_train_s": time.perf_counter() - started, "reference": "q4_q6", "fixed_rule": "stock",
                       "uncertainty": "Two observations per arm and cell. Unknown workloads use the verified whole-profile reference."})


def load_heads(weights, order_id):
    with np.load(weights, allow_pickle=False) as saved:
        if str(saved["order"]) != order_id:
            raise ValueError("checkpoint belongs to another device or order")
        heads = [[saved[f"head{i}_{key}"] for key in PARAMS] for i in range(3)]
        error = float(saved["calibration_error"])
        seed = int(saved["serving_seed"])
        inputs = saved["serving_features"].copy()
    return heads, error, seed, inputs


def whole_profile(reference, name):
    q4, q6 = reference["mask_bits"]
    return {**reference, "mask_bits": {"stock": [0, 0], "q4": [q4, 0], "q4_q6": [q4, q6]}[name]}


def equal_conditions(row, reference):
    """Remove only the two explicitly treated mask registers when comparing the fixed-gate receipts."""
    profile = row["binding"]["profile"]
    return {**row["binding"], "profile": {**profile, "mask_bits": reference["mask_bits"]}}


def check_target(args, binding):
    from ironmule.ggml_binary import identity
    from ironmule.kernel_gen import fingerprint

    if (fingerprint() != binding["fp"] or identity() != binding["native"] or trial.checksum(args.gguf) != binding["gguf"]
            or trial.checksum(args.table) != binding["table_sha256"] or trial.checksum(args.reference_lib) != binding["w4_sha256"]
            or any(trial.checksum(args.checkout / k) != v for k, v in binding["serving_sources"].items())):
        raise ValueError("changed complete execution binding")


def check_pairs(directory, order, reference, name):
    from training.code_trial import PAIRS

    binding = order["binding"]
    treatment = {"binary": binding["w4_sha256"], "table": binding["table_sha256"]}
    pairs = []
    for i in range(PAIRS):
        pair = []
        for arm, bundled in (("ref", reference), ("cand", whole_profile(reference, name))):
            row = read(directory / f"{arm}-{i}.json")
            trial.validate_arm(row, binding, treatment, 92741 + i)
            trial.verify_outputs(row, 92741 + i)
            if row["binding"]["profile"] != bundled or row["binding"]["flags"] != trial.FLAGS or row.get("profiled") is not False:
                raise ValueError("unexpected treatment in profile qualification")
            pair.append({**row, "binding": equal_conditions(row, reference)})
        pairs.append(tuple(pair))
    return trial.judge_native(pairs)


def qualify(args):
    from training.code_author import write_once
    from training.code_trial import PAIRS

    order, _, _ = sources(args.order, args.receipt)
    check_target(args, order["binding"])
    reference = trial.validate_calibration(read(args.calibration), order["binding"])
    args.state.mkdir(mode=0o700, parents=True, exist_ok=True)
    setup = {"order": order["order_id"], "name": args.name, "reference": reference, "evaluator": trial.checksum(trial.__file__)}
    write_once(args.state / "setup.json", setup)
    for i in range(PAIRS):
        arms = ("ref", "cand") if i % 2 == 0 else ("cand", "ref")
        for arm in arms:
            target = args.state / f"{arm}-{i}.json"
            if not target.exists():
                trial.launch_arm(args, args.reference_lib, 92741 + i, target,
                                 reference if arm == "ref" else whole_profile(reference, args.name))
    result = check_pairs(args.state, order, reference, args.name)
    save(args.state / "qualification.json", {**setup, **result})


def serve(args):
    order, _, _ = sources(args.order, args.receipt)
    binding = order["binding"]
    state = args.state
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    check_target(args, binding)
    calibration = args.calibration or state / "calibration.json"
    if not calibration.is_file():
        trial.launch_arm(args, args.reference_lib, args.seed, calibration,
                         {"mask_bits": trial.mask_bits(binding["mask"])}, calibrate=True)
    profile = trial.validate_calibration(read(calibration), binding)
    treatment = {"binary": binding["w4_sha256"], "table": binding["table_sha256"]}
    certification = args.reference_arm or state / "reference.json"
    if not certification.is_file():
        trial.launch_arm(args, args.reference_lib, args.seed, certification, profile)
    ref = read(certification)
    trial.validate_arm(ref, binding, treatment, ref["workload"]["seed"])
    trial.verify_outputs(ref, ref["workload"]["seed"])
    if ref["binding"]["profile"] != profile or ref["binding"]["flags"] != trial.FLAGS or ref.get("profiled") is not False:
        raise ValueError("reference serving receipt does not qualify this profile")
    started = time.perf_counter()
    heads, error, supported_seed, inputs = load_heads(args.weights, order["order_id"])
    eligible = {"q4_q6": profile}
    for directory in args.qualified:
        proof = read(directory / "qualification.json")
        name = proof["name"]
        if (name not in ("q4", "stock") or proof["order"] != order["order_id"] or proof.get("passed") is not True
                or proof["reference"] != profile or proof["evaluator"] != trial.checksum(trial.__file__)
                or read(directory / "ref-0.json")["binding"] != ref["binding"]
                or check_pairs(directory, order, profile, name).get("passed") is not True):
            raise ValueError("alternate whole profile lacks the local paired serving qualification")
        eligible[name] = whole_profile(profile, name)
    # The incumbent's startup measurement submits 32 successive tokens of ONE sequence.
    # The serving wave submits up to 32 independent streams. Matching row counts alone cannot authorize it.
    decision = Pilot.startup_profile(heads, inputs, NAMES, eligible, "q4_q6",
                                     in_support=args.seed == supported_seed and supported_seed >= 0, margin_error=error)
    picked = eligible[decision["profile"]]
    selection_s = time.perf_counter() - started
    delivered = trial.launch_arm(args, args.reference_lib, args.seed, state / "delivery.json", picked)
    trial.validate_arm(delivered, binding, treatment, args.seed)
    trial.verify_outputs(delivered, args.seed)
    if delivered["binding"]["profile"] != picked or equal_conditions(delivered, profile) != ref["binding"]:
        raise ValueError("loaded serving profile differs from the certified CPU head choice")
    if args.seed == ref["workload"]["seed"] and any(ref[k] != delivered[k] for k in ("output_sha256", "fact_digest", "tokens")):
        raise ValueError("matched-seed deliveries changed")
    save(state / "result.json", {"schema": "ironmule.w4-startup-decision.v1", "order": order["order_id"],
         "weights_sha256": trial.checksum(args.weights), "selection": decision, "eligible": list(eligible),
         "workload": delivered["workload"], "serving_receipt": trial.checksum(state / "delivery.json"),
         "local_reference_receipt": trial.checksum(certification), "profile": picked,
         "selected_before_model_load": True, "checkpoint_and_decision_s": selection_s,
         "total_after_choice_s": delivered["total_elapsed_s"], "work_wall_s": delivered["wall_s"],
         "wall_with_head_s": delivered["wall_s"] + selection_s, "p95_ttft_with_head_s": delivered["ttft_s"] + selection_s,
         "completed": delivered["completed"], "tokens": delivered["tokens"], "facts_correct": sum(delivered["facts"]),
         "pilot_online_decisions": delivered["pilot_decisions"], "output_sha256": delivered["output_sha256"],
         "claim": "One startup selection and delivery. Relative performance evidence belongs to the separate paired gate."})


def collect(args):
    """Six fresh same-device arms in reversed order. Context families, widths and both repeats stay together."""
    order, _, _ = sources(args.order, args.receipt)
    b = order["binding"]
    check_target(args, b)
    state = args.state
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not args.calibration or args.worker_root is None:
        raise ValueError("requires frozen reference calibration and restricted RunPod worker")
    incumbent = trial.validate_calibration(read(args.calibration), b)
    script = Path(__file__).with_name("pilot_w4_probe.py")
    checks = []
    for rnd, order_names in enumerate((NAMES, tuple(reversed(NAMES)))):
        for name in order_names:
            spec = {"order": order["order_id"], "profile": whole_profile(incumbent, name), "name": name, "round": rnd,
                    "gguf": b["gguf"], "native": b["native"], "devices": b["fp"]["devices"],
                    "table": b["table_sha256"], "reference": b["w4_sha256"]}
            out = state / f"{rnd}-{name}.json"
            if not out.exists():
                with tempfile.TemporaryDirectory(dir=state) as tmp:
                    data = Path(tmp) / "probe.json"
                    data.write_text(json.dumps(spec, allow_nan=False))
                    mounts = [(args.table, "/kernel/dispatch.table"), (args.reference_lib, "/kernel/active.so"),
                              (script, "/kernel/probe.py"), (data, "/kernel/probe.json")]
                    env = {"IRONMULE_" + k: v for k, v in trial.FLAGS.items()}
                    trial.launch(args, out, ["/kernel/probe.py", "--gguf", "/models/" + args.gguf.name,
                                 "--input", "/kernel/probe.json", "--output", "/output/" + out.name],
                                 mounts, {**env, "IRONMULE_KG_LIB": "/kernel/active.so",
                                         "IRONMULE_W4_TABLE": "/kernel/dispatch.table"})
            row = read(out)
            if (row["schema"] != "ironmule.w4-context-probe.v1" or row["spec"] != spec
                    or row["source_sha256"] != trial.checksum(script)
                    or sorted((r["context"], r["width"]) for r in row["cells"]) != sorted((c, w) for c in CONTEXTS for w in STREAM_WIDTHS)
                    or any(not math.isfinite(r["seconds"]) or r["seconds"] <= 0 or len(r["tokens"]) != 8
                           or any(len(step) != r["width"] for step in r["tokens"]) for r in row["cells"])):
                raise ValueError("incomplete or changed context probe")
            checks.append(row)
    cells = []
    for c in CONTEXTS:
        for w in STREAM_WIDTHS:
            observed = [next(r for r in row["cells"] if (r["context"], r["width"]) == (c, w)) for row in checks]
            equivalent = all(all(r[k] == observed[0][k] for k in ("tokens", "warm_tokens", "prompt_lengths")) for r in observed)
            cells.append({"group": f"context-{c}", "context": c, "width": w, "tokens_identical": equivalent,
                          "prompt_lengths": observed[0]["prompt_lengths"],
                          "seconds": {n: [r["seconds"] for r, check in zip(observed, checks, strict=True)
                                         if check["spec"]["name"] == n] for n in NAMES},
                          "label_eligible_for_native_step": equivalent,
                          "serving_qualified": False})
    save(state / "contexts.json", {"schema": "ironmule.w4-stream-contexts.v1", "order": order["order_id"],
         "scope": "eight native greedy steps on independent streams; no tokenization or delivery time",
         "groups": cells, "source_sha256": {p.name: trial.checksum(p) for p in sorted(state.glob("[01]-*.json"))}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="op", required=True)
    train = sub.add_parser("fit")
    train.add_argument("--report", type=Path, required=True)
    train.add_argument("--contexts", type=Path, help="complete independent-stream counterbalanced collection")
    run = sub.add_parser("serve")
    gather = sub.add_parser("collect")
    gate = sub.add_parser("qualify")
    gate.add_argument("--name", choices=("q4", "stock"), required=True)
    for cmd in (run, gather, gate):
        for field in ("state", "gguf", "checkout", "reference-lib", "table", "worker-root"):
            cmd.add_argument("--" + field, type=Path, required=True)
        cmd.add_argument("--calibration", type=Path, required=cmd in (gather, gate))
        cmd.add_argument("--worker-uid", type=int, default=60000)
        cmd.add_argument("--timeout", type=int, default=900)
    run.add_argument("--reference-arm", type=Path)
    run.add_argument("--qualified", type=Path, action="append", default=[], help="directory of a complete paired whole-profile qualification")
    train.add_argument("--seed", type=int, default=5723)
    run.add_argument("--seed", type=int, default=92741)
    for cmd in (train, run, gather, gate):
        for name in ("order", "receipt"):
            cmd.add_argument("--" + name, type=Path, required=True)
    for cmd in (train, run):
        cmd.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    {"fit": fit, "serve": serve, "collect": collect, "qualify": qualify}[args.op](args)


if __name__ == "__main__":
    main()
