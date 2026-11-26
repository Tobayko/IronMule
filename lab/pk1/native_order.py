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
    binding.update(native=native, w4_sha256=tuning["binary"], table_sha256=tuning["table"],
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
