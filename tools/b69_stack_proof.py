#!/usr/bin/env python3
"""Does the `(4, 8)` geometry move the confirmed 12B stack? One arm per process, decided fairly.

`B66` measured the geometry beating `mx.quantized_matmul` by `8` to `10%` on the shape
carrying `48%` of `12B` decode, byte identical, twice, against an A/A arm at parity. Whether
that reaches a stack is unknown: the planned proof never started, and the authorised
exploratory run was blocked when swap in use rose `3.40 GB` during its first child. That
growth had a cause worth removing rather than tolerating — each child loaded the `12B` twice,
once per arm, so two model images passed through one process.

Here each child loads one model, runs one arm, and exits. A block is three such children:
the reference, the candidate, and the reference a second time under another name. The third
is the A/A control, and it is what says whether a few per cent means anything on the machine
this actually ran on.

**Nothing shipped changes.** `ironmule/qmv_k3840.py` keeps its qualified source, its default
geometry and its activation hold; the variant is built and installed by the harness, from
`tools/b66_stack_proof.py`, so the geometry under test is the same source `B66` measured.

**An isolated `8` to `10%` is not a stack gain and is never reported as one.** Only the
complete stack time of a class against that class's own confirmed reference counts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import resource
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

MODEL = "mlx-community/gemma-3-12b-it-4bit"
GEOMETRY = (4, 8)
BLOCKS = 6
REPEATS = 5
WARMUPS = 2
CHILD_TIMEOUT_S = 1800.0
DRIFT_LIMIT = 0.25
ARMS = ("reference", "candidate", "reference_aa")
MEASURED_SOURCES = ("tools/b69_stack_proof.py", "tools/b66_stack_proof.py",
                    "tools/b57_stack_composition.py", "ironmule/qmv_k3840.py",
                    "ironmule/runtime.py", "ironmule/service.py", "ironmule/plans.py")

#: The 12B classes whose confirmed stack runs ungrouped decode. The four throughput classes
#: are not tested: their confirmed stack is C, and ironmule/stacks.py keeps k3840_matvec and
#: the paired path apart at source level, so the candidate is not composable there.
CLASSES = (("single_short", 1, "strict", 32, "A"),
           ("single_long", 1, "strict", 128, "A"),
           ("session_warm", 3, "session_warm", 32, "B"))

PREREGISTRATION = {
    "experiment": "B69_stack_proof_k3840_geometry",
    "question": ("does the (4, 8) threadgroup geometry beat the confirmed complete 12B stack "
                 "of each class whose stack runs ungrouped decode"),
    "candidate_frozen": {
        "geometry": {"num_simdgroups": GEOMETRY[0], "results_per_simdgroup": GEOMETRY[1]},
        "source": ("built by tools/b66_stack_proof.py from ironmule/qmv_k3840.py's own "
                   "kernel body, which is the source B66 measured. No new geometry is "
                   "searched and none is tried if this one loses"),
        "shipped_code_untouched": ("ironmule/qmv_k3840.py keeps its qualified source, its "
                                   "default geometry and its activation hold"),
    },
    "references": {"single_short": "A", "single_long": "A", "session_warm": "B"},
    "classes_excluded": ("the four throughput classes. Their confirmed stack is C, the paired "
                         "path, and the source keeps the kernel apart from it"),
    "design": {
        "blocks": BLOCKS,
        "children_per_block": 3,
        "one_arm_per_process": ("each child loads one model, runs one arm and exits, so a "
                                "single 12B image is resident at a time. This is the fix for "
                                "the 3.40 GB swap growth that blocked the exploratory run, "
                                "and it changes nothing about what is compared"),
        "arm_order": "rotated by block index across (reference, candidate, reference_aa)",
        "aa_control": ("reference_aa is the reference arm under another name, in its own "
                       "fresh process. Its interval is the noise floor the candidate must "
                       "clear"),
        "repeats": REPEATS, "warmups": WARMUPS,
        "statistic": "per-block ratio against the reference child of the same block, "
                     "median, 95 per cent bootstrap over 10000 resamples",
        "load_and_compile": ("model load and first-call compilation are measured and "
                             "recorded separately, and are not inside the stack time"),
    },
    "readiness": (
        "the harness's own long-standing gate and nothing stricter: no other model process, "
        "and a 1-minute load average at or below 4.0 at start. The run may therefore begin "
        "on a moderately busy machine, and that is deliberate: the AB/BA rotation, the A/A "
        "arm and the 25 per cent drift gate are the instruments that decide whether such a "
        "machine produced a usable measurement. A stricter wait was tried twice for the "
        "earlier proof and never started"),
    "correctness": (
        "every admitted projection is compared byte for byte against mx.quantized_matmul on "
        "its own buffers before a single token is timed, and a projection that differs is "
        "left on the library path and recorded rather than timed. Across arms, token ids, "
        "physical counts and stop reasons must be identical. Any difference is NO_GO"),
    "resource_gate": (
        "B65 in full, from native probes: macOS pressure normal at every probe, free memory "
        "at or above 10 per cent, peak RSS at or below 60 per cent of installed, swap in use "
        "never above its value at session start. Plus zero fallbacks"),
    "drift": f"any block deviating more than {int(DRIFT_LIMIT * 100)} per cent from the median block blocks the verdict",
    "success_criterion": (
        "CONFIRMED for a class only if the candidate's 95 per cent interval against that "
        "class's confirmed reference lies entirely below 1.0, correctness holds and the "
        "resource and drift gates pass. An interval containing 1.0 is NO_USEFUL_GAIN"),
    "no_extension": ("the block count is fixed here. It is not raised to reach significance, "
                     "and no other geometry is tried if this one does not win"),
    "not_an_activation": ("a CONFIRMED result is written into silicon_profile as evidence. "
                          "It does not release the kernel's activation hold, change any "
                          "default, or enter a product profile"),
}

CHILD = r'''
import json, sys
sys.path.insert(0, {root!r})
sys.path.insert(0, {tools!r})
from b69_stack_proof import child_main
print("@@" + json.dumps(child_main(json.loads(sys.argv[1])), sort_keys=True, allow_nan=False), flush=True)
'''


def child_main(spec: dict) -> dict:
    """One arm, one resident model, then exit."""
    import mlx.core as mx
    from ironmule.hw import memory_pressure_level, vm_counters
    from ironmule.plans import StrictOneShotPlan
    from ironmule.service import InteractiveMode, Request, Runtime
    from ironmule.tune import load_profile, resolve_local_model

    loader = importlib.util.spec_from_file_location(
        "b57", str(Path(spec["root"]) / "tools" / "b57_stack_composition.py"))
    b57 = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(b57)
    installer = importlib.util.spec_from_file_location(
        "b66sp", str(Path(spec["root"]) / "tools" / "b66_stack_proof.py"))
    b66 = importlib.util.module_from_spec(installer)
    installer.loader.exec_module(b66)

    resolved = resolve_local_model(spec["model"])
    if load_profile(spec["model"], model_identity=resolved.identity) is None:
        raise RuntimeError("no confirmed profile for this model")

    def machine(label):
        counters = vm_counters() or {}
        return {"label": label, "load_average": list(os.getloadavg()),
                "memory_pressure_level": memory_pressure_level(),
                "swapouts": counters.get("swapouts"),
                "mlx_peak_bytes": int(mx.get_peak_memory())}

    mx.reset_peak_memory()
    # Load and install are outside the stack time and are reported on their own.
    load_started = time.perf_counter_ns()
    runtime = Runtime.load(spec["model"], mode=InteractiveMode())
    load_ns = time.perf_counter_ns() - load_started
    installed = None
    try:
        if spec["arm"] == "candidate":
            install_started = time.perf_counter_ns()
            installed = b66.install_variant(runtime.engine.model, tuple(spec["geometry"]))
            installed["install_ns"] = time.perf_counter_ns() - install_started
            if not installed["admitted"] or installed["mismatched"]:
                raise RuntimeError(
                    f"variant not installable: admitted={len(installed['admitted'])} "
                    f"mismatched={len(installed['mismatched'])}")

        session = runtime.session_plan(b57.DOCUMENT, name="b69")
        question_ids = [runtime.encode(b57.DOCUMENT + "\n\n" + q) for q in b57.QUESTIONS]
        plain_ids = [runtime.encode(q) for q in b57.QUESTIONS]

        per_class = {}
        for name, count, plan_kind, max_tokens, _stack in spec["classes"]:
            before = machine(f"before_{name}")

            def build():
                requests = []
                for index in range(count):
                    if plan_kind == "strict":
                        requests.append(Request(prompt_ids=list(plain_ids[index % len(plain_ids)]),
                                                max_tokens=max_tokens, plan=StrictOneShotPlan(),
                                                arrival_ms=0.0, objective="latency"))
                    else:
                        requests.append(Request(prompt_ids=list(question_ids[index % len(question_ids)]),
                                                max_tokens=max_tokens, plan=session,
                                                arrival_ms=0.0, objective="latency"))
                return requests

            warmup_started = time.perf_counter_ns()
            for _ in range(spec["warmups"]):
                runtime.serve(build())
            warmup_ns = time.perf_counter_ns() - warmup_started

            samples, tokens, stops = [], None, None
            for _ in range(spec["repeats"]):
                started = time.perf_counter_ns()
                results = runtime.serve(build())
                samples.append(time.perf_counter_ns() - started)
                order = [r.rid for r in results]
                seen = [list(map(int, next(r for r in results if r.rid == rid).tokens))
                        for rid in order]
                stop = [next(r for r in results if r.rid == rid).stop_reason for rid in order]
                if tokens is None:
                    tokens, stops = seen, stop
                elif seen != tokens or stop != stops:
                    raise RuntimeError(f"{name} is not deterministic within its own arm")
            snapshot = runtime.telemetry.snapshot()
            per_class[name] = {
                "wall_ns": statistics.median(samples), "samples_ns": samples,
                "tokens": tokens, "stop_reasons": stops,
                "aggregate_tokens_per_second": snapshot["aggregate_tokens_per_second"],
                "latency_p50_ms": snapshot["latency_p50_ms"],
                "service_ttft_p50_ms": snapshot["service_ttft_p50_ms"],
                "engine_ttft_p50_ms": snapshot["engine_ttft_p50_ms"],
                "peak_memory_bytes": snapshot["peak_memory_bytes"],
                "fallbacks": snapshot["fallbacks"],
                "warmup_ns_outside_the_measurement": warmup_ns,
                "machine_before": before, "machine_after": machine(f"after_{name}"),
            }
        return {"pid": os.getpid(), "arm": spec["arm"], "classes": per_class,
                "installed": installed,
                "model_load_ns_outside_the_measurement": load_ns,
                "mlx_peak_bytes": int(mx.get_peak_memory()),
                "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    finally:
        runtime.close()


def bootstrap(ratios, resamples: int = 10000, seed: int = 20260910) -> dict:
    rng = random.Random(seed)
    if not ratios:
        return {"median": None, "ci_low": None, "ci_high": None, "n": 0}
    draws = sorted(median([rng.choice(ratios) for _ in ratios]) for _ in range(resamples))
    return {"median": median(ratios), "ci_low": draws[int(0.025 * resamples)],
            "ci_high": draws[int(0.975 * resamples) - 1], "n": len(ratios), "ratios": ratios}


def source_binding() -> dict:
    digests, running = {}, hashlib.sha256()
    for name in MEASURED_SOURCES:
        data = (PROJECT_ROOT / name).read_bytes()
        digests[name] = hashlib.sha256(data).hexdigest()
        running.update(name.encode())
        running.update(data)
    return {"files": digests, "combined_sha256": running.hexdigest()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preregister", type=Path)
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once

    if args.preregister:
        write_once(args.preregister, json.dumps(
            {"preregistration": PREREGISTRATION, "source_binding": source_binding(),
             "written_at": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True))
        print(f"preregistration written to {args.preregister}")
        return 0

    import mlx.core as mx
    import mlx_lm
    from ironmule.hw import (fingerprint, installed_memory_bytes, memory_pressure_level,
                             static_facts, swap_used_bytes, vm_counters)
    from ironmule.tune import gpu_busy

    busy = gpu_busy()
    if busy:
        raise SystemExit(f"another model process is running, refusing to measure ({busy})")
    load_at_start = os.getloadavg()
    if load_at_start[0] > 4.0:
        raise SystemExit(f"1-minute load average is {load_at_start[0]:.2f}, above 4.0")

    total_pages = (installed_memory_bytes() or 0) // 16384

    def probe(label):
        counters = vm_counters() or {}
        free_pages = (counters.get("free_count", 0) + counters.get("speculative_count", 0)
                      + counters.get("inactive_count", 0))
        return {"label": label, "swapouts": counters.get("swapouts"),
                "swap_used_bytes": swap_used_bytes(),
                "memory_pressure_level": memory_pressure_level(),
                "load_average": list(os.getloadavg()),
                "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
                "memory_free_percent": (100.0 * free_pages / total_pages) if total_pages else None}

    samples = [probe("start")]
    blocks, failures = [], []
    for index in range(BLOCKS):
        shift = index % len(ARMS)
        order = ARMS[shift:] + ARMS[:shift]
        entry = {"block": index, "order": list(order), "children": {}}
        for arm in order:
            spec = {"model": MODEL, "arm": "reference" if arm == "reference_aa" else arm,
                    "geometry": list(GEOMETRY), "classes": [list(row) for row in CLASSES],
                    "repeats": REPEATS, "warmups": WARMUPS, "root": str(PROJECT_ROOT)}
            process = subprocess.Popen(
                [sys.executable, "-u", "-c",
                 CHILD.format(root=str(PROJECT_ROOT), tools=str(PROJECT_ROOT / "tools")),
                 json.dumps(spec)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                cwd=str(PROJECT_ROOT),
                env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
            try:
                stdout, stderr = process.communicate(timeout=CHILD_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            code = process.returncode
            marker = next((line[2:] for line in stdout.splitlines()
                           if line.startswith("@@")), None)
            if code != 0 or marker is None:
                failures.append({"block": index, "arm": arm, "returncode": code,
                                 "signalled": code is not None and code < 0,
                                 "signal_name": (signal.Signals(-code).name
                                                 if code is not None and code < 0 else None),
                                 "stderr_tail": stderr[-4000:]})
                break
            entry["children"][arm] = json.loads(marker)
            print(f"block {index} {arm} done", flush=True)
        blocks.append(entry)
        samples.append(probe(f"after_block_{index}"))
        if failures:
            break
    samples.append(probe("end"))

    complete = [entry for entry in blocks if set(entry["children"]) == set(ARMS)]
    differences, comparisons = [], {}
    reference_tokens = {}
    for name, *_rest in CLASSES:
        for entry in complete:
            base = entry["children"]["reference"]["classes"][name]
            reference_tokens.setdefault(name, base["tokens"])
            for arm in ARMS:
                row = entry["children"][arm]["classes"][name]
                if row["tokens"] != reference_tokens[name] or row["stop_reasons"] != base["stop_reasons"]:
                    differences.append({"block": entry["block"], "arm": arm, "class": name})
        comparisons[name] = {
            arm: bootstrap([entry["children"][arm]["classes"][name]["wall_ns"]
                            / entry["children"]["reference"]["classes"][name]["wall_ns"]
                            for entry in complete])
            for arm in ("candidate", "reference_aa")}

    block_walls = [sum(entry["children"][arm]["classes"][name]["wall_ns"] / 1e6
                       for arm in ARMS for name, *_r in CLASSES) for entry in complete]
    wall_median = median(block_walls) if block_walls else None
    disturbed = ([{"block": complete[i]["block"], "wall_ms": w,
                   "deviation": w / wall_median - 1.0}
                  for i, w in enumerate(block_walls)
                  if abs(w / wall_median - 1.0) > DRIFT_LIMIT] if wall_median else [])

    swaps = [s["swap_used_bytes"] for s in samples if s["swap_used_bytes"] is not None]
    frees = [s["memory_free_percent"] for s in samples if s["memory_free_percent"] is not None]
    levels = [s["memory_pressure_level"] for s in samples]
    peak_rss = max((s["peak_rss_bytes"] for s in samples), default=0)
    memory_total = installed_memory_bytes() or 0
    fallbacks = sum(entry["children"][arm]["classes"][name]["fallbacks"]
                    for entry in complete for arm in ARMS for name, *_r in CLASSES)
    gate_reasons = []
    if not swaps or max(swaps) > swaps[0]:
        gate_reasons.append("swap in use rose above its value at session start")
    if not frees or min(frees) < 10.0:
        gate_reasons.append("free memory below 10 per cent")
    if not all(level == 1 and level is not None for level in levels):
        gate_reasons.append("macOS memory pressure not normal at every probe, or unreadable")
    if memory_total and peak_rss > memory_total * 0.60:
        gate_reasons.append("combined peak child RSS above 60 per cent of installed memory")

    blocked = bool(failures) or bool(differences) or fallbacks or bool(disturbed) or bool(gate_reasons)
    winners = {name: row["candidate"] for name, row in comparisons.items()
               if row["candidate"]["ci_high"] is not None and row["candidate"]["ci_high"] < 1.0}
    if blocked:
        verdict = "BLOCKED"
    elif not complete:
        verdict = "BLOCKED"
    elif winners:
        verdict = "STACK_CONFIRMED"
    else:
        verdict = "NO_USEFUL_GAIN"

    record = {
        "experiment": PREREGISTRATION["experiment"],
        "preregistration": PREREGISTRATION,
        "source_binding": source_binding(),
        "environment": {"platform": platform.platform(), "mlx": mx.__version__,
                        "mlx_lm": mlx_lm.__version__, "fingerprint": fingerprint(),
                        "chip": static_facts().get("chip")},
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "load_average_at_start": list(load_at_start),
        "load_average_at_end": list(os.getloadavg()),
        "blocks": blocks,
        "complete_blocks": len(complete),
        "child_failures": failures,
        "correctness": {"differences": differences, "identical": not differences},
        "comparisons": comparisons,
        "confirmed_classes": sorted(winners),
        "stability": {"block_wall_ms": block_walls, "median_block_wall_ms": wall_median,
                      "disturbed_blocks": disturbed},
        "resource_gate": {"applied": "pressure", "passed": not gate_reasons,
                          "reasons": gate_reasons,
                          "swap_growth_above_start_bytes": (max(swaps) - swaps[0]) if swaps else None,
                          "min_memory_free_percent": min(frees) if frees else None,
                          "peak_child_rss_bytes": peak_rss},
        "resource_samples": samples,
        "fallbacks": fallbacks,
        "verdict": verdict,
        "reading_rule": ("an isolated 8 to 10 per cent kernel result is not a stack gain and "
                         "is not reported as one. Only the intervals below count"),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"verdict": verdict, "complete_blocks": len(complete),
                      "correctness_identical": not differences,
                      "resource_gate": record["resource_gate"]["passed"],
                      "disturbed_blocks": len(disturbed),
                      "comparisons": {k: {a: {"median": v["median"],
                                              "ci": [v["ci_low"], v["ci_high"]]}
                                          for a, v in row.items()}
                                      for k, row in comparisons.items()}},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
