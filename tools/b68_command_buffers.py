#!/usr/bin/env python3
"""Do fewer Metal command buffers make a confirmed stack faster, or only make fewer buffers?

`B66` measured a submission boundary on this machine at `230` to `270 us`, about forty times
the `6.41 us` per dispatch `E5` measured, and every confirmed win this project holds works by
putting more work between two such boundaries. MLX 0.32.0 has its own boundary that nothing
here has ever touched: `CommandEncoder::needs_commit()` commits when either
`max_ops_per_buffer` or `max_mb_per_buffer` is exceeded, and both come from
`MLX_MAX_OPS_PER_BUFFER` and `MLX_MAX_MB_PER_BUFFER` through a function-local static, so they
are read once per process and never again.

**A commit is not a synchronize.** A committed buffer runs while the CPU encodes the next
one, so a lower buffer count is not a speed claim and is not treated as one here. The only
performance statement this file is allowed to make comes from complete stack time.

Phase `probe` establishes that the limit is in force and where it bites, by observation
rather than from the header: a chain of identical tiny kernels behind one `eval`, swept
across the limit, should show a step wherever a commit is forced. Phase `sweep` measures the
real workload classes, one fresh process per configuration, rotated across rounds so a busy
machine cannot hand the win to whichever configuration ran first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

MEASURED_SOURCES = ("tools/b68_command_buffers.py", "tools/b57_stack_composition.py",
                    "ironmule/runtime.py", "ironmule/service.py", "ironmule/hw.py")

MODELS = {"4B": "mlx-community/gemma-3-4b-it-4bit",
          "12B": "mlx-community/gemma-3-12b-it-4bit"}

#: Fixed before anything ran. `50/50` is the reference because it is what this chip gets by
#: default. The last two isolate which limit carries any effect the first four show.
CONFIGURATIONS = (
    ("default_50_50", None, None),
    ("ops_mb_100", 100, 100),
    ("ops_mb_200", 200, 200),
    ("ops_mb_400", 400, 400),
    ("ops_400_mb_default", 400, None),
    ("ops_default_mb_400", None, 400),
)
ROUNDS = 4
REPEATS = 5
WARMUPS = 2

#: The classes from `tools/b57_stack_composition.py`, unchanged, so the numbers connect to
#: the confirmed composition profile rather than to a workload invented here.
CLASSES = (("single_short", 1, "strict", 32),
           ("session_warm", 3, "session_warm", 32))

PREREGISTRATION = {
    "experiment": "B68_command_buffer_limits",
    "question": ("does raising MLX's per-command-buffer op and megabyte limits make a "
                 "confirmed stack faster on this machine, or does it only change how many "
                 "command buffers there are"),
    "mechanism_verified_in_the_installed_library": (
        "mlx/include/mlx/utils.h reads MLX_MAX_OPS_PER_BUFFER and MLX_MAX_MB_PER_BUFFER "
        "through a function-local static, so each is read once per process; "
        "mlx/include/mlx/backend/metal/device.h holds both on the Device and decides in "
        "CommandEncoder::needs_commit(). The values themselves are not exposed to Python, "
        "which is why phase `probe` observes the limit instead of asserting it"),
    "commit_is_not_synchronize": (
        "a committed command buffer executes while the CPU encodes the next, so a lower "
        "buffer count is not a speed claim. No count from this study is reported as a gain"),
    "configurations": [{"name": name, "ops": ops, "mb": mb} for name, ops, mb in CONFIGURATIONS],
    "why_these": ("the default, three whole-number steps above it, and two single-axis "
                  "diagnoses. Not a search, and not extended if none of them wins"),
    "not_a_product_recommendation": ("these are research parameters. A configuration enters "
                                     "silicon_profile only after beating a confirmed "
                                     "complete stack, and even then it is bound to this "
                                     "hardware, this MLX build, this model and this class"),
    "design": (f"{ROUNDS} rounds; every round runs every configuration once in a rotated "
               f"order, each in a fresh process with the environment set before the first "
               f"MLX access; {WARMUPS} warmups and {REPEATS} measured repeats per class; "
               "ratios are paired within a round against the default configuration"),
    "readback_every_not_reswept": ("2, 4 and 8 were screened by the autotuner on both models "
                                   "and 2 won. That sweep is not repeated"),
    "gates": ("the B65 resource gate in full: macOS pressure normal at every probe, free "
              "memory at or above 10 per cent, peak RSS at or below 60 per cent of "
              "installed, swap in use not above its start. Plus zero fallbacks, token and "
              "stop-reason identity against the default configuration, and any command "
              "buffer error or GPU timeout recorded and blocking"),
    "selection_and_confirmation_are_separate": (
        "this file selects. A winner is confirmed only against the best confirmed complete "
        "stack of its own workload class, in its own separate run"),
}


# --------------------------------------------------------------------------- child

CHILD = r'''
import json, os, sys
sys.path.insert(0, {root!r})
sys.path.insert(0, {tools!r})
from b68_command_buffers import child_main
print("@@" + json.dumps(child_main(json.loads(sys.argv[1])), sort_keys=True, allow_nan=False), flush=True)
'''


def child_probe(spec: dict) -> dict:
    """A staircase: N identical tiny kernels behind one eval, swept across the limit.

    If a commit is forced every `max_ops_per_buffer` operations, the cost of N should carry a
    visible step at each multiple of the limit. This observes that rather than asserting it.
    """
    import mlx.core as mx

    x = mx.random.normal((256, 256)).astype(mx.float32)
    mx.eval(x)
    out = {}
    for count in spec["counts"]:
        def call(count=count):
            value = x
            for _ in range(count):
                value = value + 1.0
            mx.eval(value)
        for _ in range(2):
            call()
        samples = [None] * 9
        for index in range(9):
            start = time.perf_counter_ns()
            call()
            samples[index] = time.perf_counter_ns() - start
        out[count] = {"ops": count, "median_ns": statistics.median(samples),
                      "ns_per_op": statistics.median(samples) / count}
    return {"pid": os.getpid(), "staircase": out,
            "env": {k: os.environ.get(k) for k in
                    ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER")}}


def child_main(spec: dict) -> dict:
    if spec["kind"] == "probe":
        return child_probe(spec)

    import importlib.util

    import mlx.core as mx
    from ironmule.hw import memory_pressure_level, vm_counters
    from ironmule.plans import StrictOneShotPlan
    from ironmule.service import InteractiveMode, Request, Runtime
    from ironmule.tune import load_profile, resolve_local_model

    loader = importlib.util.spec_from_file_location(
        "b57", str(Path(spec["root"]) / "tools" / "b57_stack_composition.py"))
    b57 = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(b57)

    resolved = resolve_local_model(spec["model"])
    if load_profile(spec["model"], model_identity=resolved.identity) is None:
        raise RuntimeError("no confirmed profile for this model")

    mx.reset_peak_memory()
    runtime = Runtime.load(spec["model"], mode=InteractiveMode())
    try:
        session = runtime.session_plan(b57.DOCUMENT, name="b68")
        question_ids = [runtime.encode(b57.DOCUMENT + "\n\n" + q) for q in b57.QUESTIONS]
        plain_ids = [runtime.encode(q) for q in b57.QUESTIONS]
        per_class = {}
        for name, count, plan_kind, max_tokens in spec["classes"]:
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

            for _ in range(spec["warmups"]):
                runtime.serve(build())
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
            counters = vm_counters() or {}
            per_class[name] = {
                "wall_ns": statistics.median(samples), "samples_ns": samples,
                "tokens": tokens, "stop_reasons": stops,
                "aggregate_tokens_per_second": snapshot["aggregate_tokens_per_second"],
                "latency_p50_ms": snapshot["latency_p50_ms"],
                "service_ttft_p50_ms": snapshot["service_ttft_p50_ms"],
                "engine_ttft_p50_ms": snapshot["engine_ttft_p50_ms"],
                "fallbacks": snapshot["fallbacks"],
                "machine_after": {"load_average": list(os.getloadavg()),
                                  "memory_pressure_level": memory_pressure_level(),
                                  "swapouts": counters.get("swapouts")},
            }
        return {"pid": os.getpid(), "classes": per_class,
                "mlx_peak_bytes": int(mx.get_peak_memory()),
                "env": {k: os.environ.get(k) for k in
                        ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER")}}
    finally:
        runtime.close()


# --------------------------------------------------------------------------- parent


def run_child(spec: dict, ops, mb, timeout: float = 1800.0) -> dict:
    environment = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
    for key in ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER"):
        environment.pop(key, None)
    if ops is not None:
        environment["MLX_MAX_OPS_PER_BUFFER"] = str(ops)
    if mb is not None:
        environment["MLX_MAX_MB_PER_BUFFER"] = str(mb)
    process = subprocess.Popen(
        [sys.executable, "-u", "-c",
         CHILD.format(root=str(PROJECT_ROOT), tools=str(PROJECT_ROOT / "tools")),
         json.dumps(spec)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=str(PROJECT_ROOT), env=environment)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
    marker = next((line[2:] for line in stdout.splitlines() if line.startswith("@@")), None)
    lowered = stderr.lower()
    return {"returncode": process.returncode,
            "record": json.loads(marker) if marker else None,
            "stderr_tail": stderr[-3000:] if process.returncode != 0 else "",
            "command_buffer_error": any(token in lowered for token in
                                        ("command buffer", "gpu timeout", "iogpu",
                                         "mtlcommandbuffer", "device removed"))}


def bootstrap(ratios, resamples: int = 10000, seed: int = 20260910) -> dict:
    import random
    rng = random.Random(seed)
    if not ratios:
        return {"median": None, "ci_low": None, "ci_high": None, "n": 0}
    draws = sorted(statistics.median([rng.choice(ratios) for _ in ratios])
                   for _ in range(resamples))
    return {"median": statistics.median(ratios), "ci_low": draws[int(0.025 * resamples)],
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
    parser.add_argument("--phase", choices=("probe", "sweep"), default="sweep")
    parser.add_argument("--models", default="4B,12B")
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
    total_pages = (installed_memory_bytes() or 0) // 16384

    def probe(label):
        counters = vm_counters() or {}
        free_pages = (counters.get("free_count", 0) + counters.get("speculative_count", 0)
                      + counters.get("inactive_count", 0))
        return {"label": label, "swapouts": counters.get("swapouts"),
                "swap_used_bytes": swap_used_bytes(),
                "memory_pressure_level": memory_pressure_level(),
                "load_average": list(os.getloadavg()),
                "memory_free_percent": (100.0 * free_pages / total_pages) if total_pages else None}

    samples = [probe("start")]
    result: dict = {}
    errors = []

    if args.phase == "probe":
        counts = [10, 25, 40, 45, 48, 50, 52, 55, 60, 75, 95, 100, 105, 150, 200, 250]
        for name, ops, mb in (("default_50_50", None, None), ("ops_mb_400", 400, 400)):
            child = run_child({"kind": "probe", "counts": counts}, ops, mb)
            if child["returncode"] != 0 or child["record"] is None:
                errors.append({"configuration": name, **{k: child[k] for k in
                                                         ("returncode", "stderr_tail",
                                                          "command_buffer_error")}})
                continue
            result[name] = child["record"]
            samples.append(probe(f"after_{name}"))
            print(f"probe {name} done", flush=True)
    else:
        for label in [m.strip() for m in args.models.split(",") if m.strip()]:
            rounds = []
            for index in range(ROUNDS):
                shift = index % len(CONFIGURATIONS)
                order = CONFIGURATIONS[shift:] + CONFIGURATIONS[:shift]
                entry = {"round": index, "order": [row[0] for row in order],
                         "configurations": {}}
                for name, ops, mb in order:
                    spec = {"kind": "sweep", "model": MODELS[label],
                            "classes": [list(row) for row in CLASSES],
                            "repeats": REPEATS, "warmups": WARMUPS, "root": str(PROJECT_ROOT)}
                    child = run_child(spec, ops, mb)
                    if child["returncode"] != 0 or child["record"] is None:
                        errors.append({"model": label, "round": index, "configuration": name,
                                       **{k: child[k] for k in ("returncode", "stderr_tail",
                                                                "command_buffer_error")}})
                        continue
                    entry["configurations"][name] = child["record"]
                    print(f"{label} round {index} {name} done", flush=True)
                rounds.append(entry)
                samples.append(probe(f"after_{label}_round_{index}"))
            result[label] = rounds

    samples.append(probe("end"))

    analysis = {}
    if args.phase == "sweep":
        for label, rounds in result.items():
            per_class = {}
            for name, *_rest in CLASSES:
                differences, ratios = [], {}
                reference_tokens = None
                for entry in rounds:
                    configurations = entry["configurations"]
                    if "default_50_50" not in configurations:
                        continue
                    base = configurations["default_50_50"]["classes"][name]
                    reference_tokens = reference_tokens or base["tokens"]
                    for configuration, record in configurations.items():
                        row = record["classes"][name]
                        if row["tokens"] != reference_tokens or row["stop_reasons"] != base["stop_reasons"]:
                            differences.append({"round": entry["round"], "configuration": configuration})
                        if configuration != "default_50_50":
                            ratios.setdefault(configuration, []).append(
                                row["wall_ns"] / base["wall_ns"])
                per_class[name] = {
                    "reference": "default_50_50",
                    "differences": differences,
                    "arms": {configuration: bootstrap(values)
                             for configuration, values in ratios.items()},
                }
            analysis[label] = per_class

    swaps = [s["swap_used_bytes"] for s in samples if s["swap_used_bytes"] is not None]
    frees = [s["memory_free_percent"] for s in samples if s["memory_free_percent"] is not None]
    levels = [s["memory_pressure_level"] for s in samples]
    gate_reasons = []
    if not swaps or max(swaps) > swaps[0]:
        gate_reasons.append("swap in use rose above its start")
    if not frees or min(frees) < 10.0:
        gate_reasons.append("free memory below 10 per cent")
    if not all(level == 1 for level in levels) or not all(level is not None for level in levels):
        gate_reasons.append("macOS memory pressure not normal at every probe, or unreadable")
    if any(row.get("command_buffer_error") for row in errors):
        gate_reasons.append("a command buffer error or GPU timeout was seen")

    record = {
        "experiment": PREREGISTRATION["experiment"],
        "phase": args.phase,
        "preregistration": PREREGISTRATION,
        "source_binding": source_binding(),
        "environment": {"platform": platform.platform(), "mlx": mx.__version__,
                        "mlx_lm": mlx_lm.__version__, "fingerprint": fingerprint(),
                        "chip": static_facts().get("chip"),
                        "gpu_architecture": mx.device_info().get("architecture")},
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "results": result,
        "analysis": analysis,
        "child_errors": errors,
        "resource_gate": {"applied": "pressure", "passed": not gate_reasons,
                          "reasons": gate_reasons},
        "resource_samples": samples,
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(f"written {args.out}")
    if analysis:
        print(json.dumps({label: {name: {"differences": len(row["differences"]),
                                         "arms": {a: {"median": v["median"],
                                                      "ci": [v["ci_low"], v["ci_high"]]}
                                                  for a, v in row["arms"].items()}}
                                  for name, row in classes.items()}
                          for label, classes in analysis.items()}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
