"""Attribute native development orders to measured CUDA device events on one tuned W4 build."""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from statistics import median

SOURCE = "native/kernels/w4mma.cu"
# Kernel source lines, resolved again at capture. Only editable kernels in the loaded W4 library qualify.
DEVICE_SYMBOL = re.compile(r"\(anonymous namespace\)::(w4\w+|reduce_split|prep_x)(<[^>]+>)?\(")


def activity(prof):
    times, counts = Counter(), Counter()
    for event in prof.events():
        if event.device_type.name == "CUDA" and event.device_time_total > 0:
            times[event.name] += event.device_time_total / 1e6
            counts[event.name] += 1
    return [{"symbol": name, "device_s": times[name], "calls": counts[name]}
            for name, _ in times.most_common()]


def ranked(samples, source, decode_steps):
    """A sensitivity replay of the observed 31-step decode tail of the PR57 benchmark."""
    runs = []
    for sample in samples:
        found = {}
        for event in sample:
            match = DEVICE_SYMBOL.search(event["symbol"])
            if match is None:
                continue
            name = match.group(1)
            line = re.search(r"^(?:__global__ void )?" + re.escape(name) + r"\(", source, re.MULTILINE)
            if line is None:
                continue
            site = (name, match.group(0).removeprefix("(anonymous namespace)::").removesuffix("("),
                    source.count("\n", 0, line.start()) + 1)
            found[site] = (event["device_s"] / decode_steps, event["calls"] / decode_steps)
        runs.append(found)
    if len(runs) < 2:
        raise ValueError("two warmed CUDA activity traces required")
    candidates = []
    for key in set.intersection(*(set(run) for run in runs)):
        seconds = [run[key][0] for run in runs]
        count = [run[key][1] for run in runs]
        if min(seconds) <= 0 or min(count) <= 0:
            continue
        candidates.append({"function": key[0], "specialization": key[1], "line": key[2], "path": SOURCE,
                           "kernel_seconds_per_step": seconds, "calls_per_step": count,
                           "simulated_31_step_saving_at_10_percent_s": median(seconds) * 31 * 0.10})
    candidates.sort(key=lambda c: (-c["simulated_31_step_saving_at_10_percent_s"], c["specialization"]))
    if not candidates:
        raise ValueError("no source-attributable W4 CUDA activity in both warmed decode traces")
    return candidates


def write_order(args, engine, samples, wall, root, commit):
    from ironmule.ggml_binary import identity
    from ironmule.ggml_engine import file_sha256
    from ironmule.kernel_gen import fingerprint
    from ironmule.w4_kernels import source_digest
    from lab.portable_gpu.bench import save

    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("profile requires the exact full source commit")
    if engine.gpu_layers != 999 or args.prompts not in engine.kg.get("tokens_served", []):
        save(args.order_out.with_suffix(".rejected.json"), {
            "kind": "rejected_native_diagnostic", "reason": "locally qualified width-32 W4 path unavailable",
            "base_commit": commit, "gpu_layers": engine.gpu_layers, "width": args.prompts,
            "kernel_selection": engine.kg, "w4": engine.w4, "settings": engine.settings,
            "events": samples, "wall_seconds_per_window": wall,
        })
        raise ValueError("W4 was not selected on this GPU for the profiled width")
    table = Path(os.environ["IRONMULE_W4_TABLE"])
    lib = Path(os.environ["IRONMULE_KG_LIB"])
    tuning = json.loads(table.with_name("measurement.json").read_text())
    binding = tuning["binding"].copy()
    native = identity()
    if (binding["src"] != source_digest() or binding["gguf"] != file_sha256(args.gguf)
            or binding["fp"] != fingerprint() or tuning["binary"] != file_sha256(str(lib))
            or tuning["table"] != file_sha256(str(table)) or not native.get("sha256")):
        raise ValueError("loaded binary, device, runtime, table, source or weights lack a matching receipt")
    from ironmule import ggml_engine, gguf_tokenizer, pilot

    serving_sources = {m.__name__.replace(".", "/") + ".py": file_sha256(m.__file__)
                       for m in (ggml_engine, gguf_tokenizer, pilot)}
    binding.update(native=native, serving_sources=serving_sources, tuning_receipt_sha256=file_sha256(str(table.with_name("measurement.json"))),
                   w4_sha256=tuning["binary"], table_sha256=tuning["table"],
                   source_sha256=file_sha256(str(root / SOURCE)), engine=engine.settings, mask=engine.kg)
    variants = [row for line in table.read_text().splitlines() if line.strip() and not line.startswith("#")
                and (row := list(map(int, line.split())))[3] == args.prompts]
    if not variants:
        raise ValueError("no tuned table entry for this width")
    report = {"base_commit": commit, "binding": binding, "width": args.prompts, "decode_steps": args.decode,
              "prompt_tokens_per_client": args.prompt_len, "kind": "measured_native_diagnostic",
              "reference_flags": {k: os.getenv("IRONMULE_" + k) for k in
                                  ("PS1", "PS1_WARM", "PK6", "CUDA_GRAPHS_ROWS", "GGML_PACK", "GGML_UBATCH")},
              "dispatch_rows_type_n_k_batch_variant_ksplit": variants,
              "device_events": samples, "wall_seconds_per_window": wall}
    profile = args.order_out.with_suffix(".profile.json")
    save(profile, report)
    alternatives = ranked(samples, (root / SOURCE).read_text(), args.decode)
    report = {"schema": "ironmule.native-code-experiment.v1", "base_commit": commit, "binding": binding,
              "target": alternatives[0], "allowed_paths": [SOURCE], "dreams": alternatives,
              "workload": {"decode_rows": args.prompts, "steps": args.decode},
              "evaluator": "ironmule.w4_target_trial.v1",
              "profile_sha256": file_sha256(str(profile)),
              "hypothesis": "Reduce the measured cost of this exact compiled kernel specialization. Preserve "
                            "the incumbent implementation as a separate variant and qualify on the bound device.",
              "historical_references": ["lab/pk1/results/pod28/a100-pod28-profile.log",
                                        "lab/pk1/results/pod30/a100_dk1c.log", "docs/LAB.md"],
              "simulation": {"kind": "simulated", "fraction": 0.10, "steps": 31,
                             "uncertainty": "A hypothetical shorter device interval need not reduce wall time. "
                                            "Attribution establishes a compiled symbol, not a matrix shape."}}
    report["order_id"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    save(args.order_out, report)
    print(f"Native order {report['order_id']} for {alternatives[0]['specialization']}")


def next_target(options, explored, wave_s, fraction):
    """Require a new source family to pay the whole-wave gate under the original simulation's prior."""
    if not 0 < fraction < 1:
        raise ValueError("invalid simulated reduction")
    choices = []
    for option in options:
        if option["function"] in explored:
            continue
        exposed = 31 * min(option["kernel_seconds_per_step"])
        saved = wave_s * (1 - 1 / 1.02)
        if exposed > 0 and fraction * exposed >= saved:
            choices.append({**option, "required_device_time_reduction": saved / exposed})
    return min(choices, key=lambda c: c["required_device_time_reduction"]) if choices else None


def reference_prices(path, order):
    """Price the next kernel against completed incumbent work on the original loaded device."""
    from lab.pk1.native_trial import checksum, validate_arm, verify_outputs

    bind = order["binding"]
    if path.is_file():
        ref = json.loads(path.read_text())
        seed = ref["workload"]["seed"]
        validate_arm(ref, bind, {"binary": bind["w4_sha256"], "table": bind["table_sha256"]}, seed)
        verify_outputs(ref, seed)
        return ref["wall_s"], ref["total_elapsed_s"], {"reference_sha256": checksum(path)}

    from lab.pk1 import restart_trial

    waves, workers, hashes = [], [], {}
    first = None
    for i in range(restart_trial.PAIRS):
        source = path / f"ref-{i}.json"
        row = json.loads(source.read_text())
        restart_trial.validate(row, restart_trial.SEED + i, expected_source=bind["serving_sources"]["ironmule/ggml_engine.py"])
        restart_trial.verify_binding(row, order)
        if row["reused"] or row["factual_first"] or (first is not None and row["binding"] != first):
            raise ValueError("need six matched incumbent mixed-first runs with fresh mask measurements")
        first = row["binding"]
        hashes[source.name] = checksum(source)
        waves.append(row["warm"]["wall_s"])
        workers.append(row["broker_elapsed_s"])
    return median(waves), median(workers), {"warm_reference_receipts": hashes, "warm_reference_waves_s": waves}


def replay_rejection(prior, trial, reference, output, seconds, hourly_usd, usd):
    import math
    import shutil

    from lab.pk1.native_trial import SCHEMA, checksum
    from lab.portable_gpu.bench import save

    result = json.loads((trial / "result.json").read_text())
    order = json.loads(prior.read_text())
    profile = prior.with_suffix(".profile.json")
    digest = hashlib.sha256(json.dumps({k: v for k, v in order.items() if k != "order_id"}, sort_keys=True).encode()).hexdigest()
    if (digest != order["order_id"] or result.get("passed") is not False or result["context"]["order"] != digest
            or result["context"]["base_commit"] != order["base_commit"] or result["schema"] != SCHEMA
            or checksum(profile) != order["profile_sha256"] or order["evaluator"] != SCHEMA):
        raise ValueError("expected a rejected trial of this profiled native order")
    wave_s, worker_s, reference_ids = reference_prices(reference, order)
    if not all(math.isfinite(v) and v > 0 for v in [seconds, hourly_usd, usd, wave_s, worker_s, result["elapsed_s"]]):
        raise ValueError("cost and time inputs must be finite and positive")
    history = [*order.get("prior_rejections", []), {"order": digest, "target": order["target"],
               "reason": result["reason"], "trial_sha256": checksum(trial / "result.json")}]
    excluded = {h["target"]["function"] for h in history}
    fraction = order["simulation"]["fraction"]
    target = next_target(order["dreams"], excluded, wave_s, fraction)
    # Fixed six alternating pairs. A short screening failure cannot set an unrealistically low time estimate.
    forecast_s = max(result["elapsed_s"], 12 * worker_s)
    cost = {"estimated_seconds": forecast_s, "estimated_gpu_usd": forecast_s * hourly_usd / 3600,
            "available_seconds": seconds, "available_usd": usd, "reference_wall_s": wave_s,
            **reference_ids, "hourly_gpu_usd": hourly_usd}
    if (target is None or forecast_s > seconds or cost["estimated_gpu_usd"] > usd
            or len(history) >= 2):
        projections = [{"specialization": opt["specialization"],
                        "simulated_whole_wave_saving_s": fraction * 31 * min(opt["kernel_seconds_per_step"])}
                       for opt in order["dreams"] if opt["function"] not in excluded]
        save(output, {"schema": "ironmule.native-search-stop.v1", "prior_rejections": history, "cost": cost,
                      "required_wave_saving_s": wave_s * (1 - 1 / 1.02), "simulated_device_reduction": fraction,
                      "unexplored_projections": projections,
                      "reason": "no new target above search sensitivity or remaining operational bounds"})
        return
    following = {**order, "target": target, "prior_rejections": history, "prospective_cost": cost,
                 "hypothesis": f"Investigate {target['specialization']} from the retained target-device profile. "
                               "Avoid the rejected source families and preserve the local dispatch ABI."}
    following.pop("order_id")
    following["order_id"] = hashlib.sha256(json.dumps(following, sort_keys=True).encode()).hexdigest()
    out_profile = output.with_suffix(".profile.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if profile.resolve() != out_profile.resolve():
        shutil.copyfile(profile, out_profile)
    save(output, following)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Replay one rejected native code experiment")
    for field in ("prior", "trial", "reference", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    for field in ("seconds", "hourly-usd", "usd"):
        parser.add_argument("--" + field, type=float, required=True)
    replay_rejection(**vars(parser.parse_args()))
