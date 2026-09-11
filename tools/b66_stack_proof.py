#!/usr/bin/env python3
"""Does the geometry that won in isolation move a confirmed stack, or does it not transfer?

`B66`'s geometry axis measured the owned `K = 3840` matvec at `(4, 8)` -- four simdgroups,
eight output rows each -- running `8` to `10%` faster than `mx.quantized_matmul` on the shape
that carries `48%` of `12B` decode, byte identical, over two independent runs with an A/A
arm at parity. That is an isolated kernel number, and `E5` measured exactly such a number
failing to reach a pipelined graph: fusion's `6%` bandwidth prediction arrived as `0%` in
decode because the dispatches it added cancelled it.

So this asks the only question that counts. The reference is the confirmed stack for each
workload class on `12B`: `A` for `single_short` and `single_long`, `B` for `session_warm`.
The candidate is that same stack with the `K = 3840` projections served by the geometry
under test, and nothing else changed.

**The shipped kernel is not touched.** `tests/test_qmv_k3840_integration.py` holds a tripwire
that the shipped module must contain the `B42`/`B43` source verbatim, and an earlier attempt
to parameterise it in place tripped exactly that guard. The variant is built here, from the
same body, and installed here. `ironmule/qmv_k3840.py` keeps its activation hold and its
default geometry, and nothing in this file proposes releasing either.
"""

from __future__ import annotations

import argparse
import hashlib
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
PAIRS = 6                 # fresh processes, arm order alternating
REPEATS = 5
WARMUPS = 2
CHILD_TIMEOUT_S = 1800.0
MEASURED_SOURCES = ("tools/b66_stack_proof.py", "tools/b57_stack_composition.py",
                    "ironmule/qmv_k3840.py", "ironmule/runtime.py", "ironmule/service.py",
                    "ironmule/plans.py")

#: The classes on `12B` whose confirmed stack runs ungrouped decode. The paired path is the
#: confirmed stack everywhere else and the source keeps it apart from this kernel, so those
#: classes cannot host the candidate and are not measured here.
CLASSES = (
    ("single_short", 1, "strict", 32, "A"),
    ("single_long", 1, "strict", 128, "A"),
    ("session_warm", 3, "session_warm", 32, "B"),
)

PREREGISTRATION = {
    "experiment": "B66_stack_proof_k3840_geometry",
    "question": ("does the (4, 8) threadgroup geometry, measured 8 to 10 per cent faster "
                 "than the library in isolation, move the confirmed 12B stack for the "
                 "classes whose confirmed stack runs ungrouped decode"),
    "model": MODEL,
    "geometry_under_test": {"num_simdgroups": GEOMETRY[0], "results_per_simdgroup": GEOMETRY[1]},
    "shipped_geometry": {"num_simdgroups": 2, "results_per_simdgroup": 4},
    "classes": [{"class": name, "requests": count, "plan": plan, "max_tokens": tokens,
                 "confirmed_reference_stack": stack}
                for name, count, plan, tokens, stack in CLASSES],
    "classes_excluded": ("the four throughput classes, whose confirmed stack is C. C is the "
                         "paired path and ironmule/stacks.py keeps k3840_matvec apart from "
                         "it at source level, so the candidate cannot be composed there"),
    "design": (f"{PAIRS} fresh processes, arm order alternating between them, one model "
               f"load per arm, {WARMUPS} warmups and {REPEATS} measured repeats per class, "
               "paired per-process ratios, 10000-resample bootstrap"),
    "correctness": ("every admitted projection is checked byte for byte against "
                    "mx.quantized_matmul on its own buffers before a single token is timed, "
                    "and token ids, physical counts and stop reasons must be identical "
                    "between the arms of every process"),
    "adoption_rule": ("a class enters silicon_profile only if its 95 per cent interval "
                      "against its own confirmed reference stack lies entirely below 1.0 "
                      "and correctness and the B65 resource gate both pass"),
    "shipped_code_untouched": ("ironmule/qmv_k3840.py keeps its qualified source, its "
                               "default geometry and its activation hold. The variant is "
                               "built and installed by this harness only"),
    "not_an_activation": ("a confirmed win here is written into silicon_profile as evidence. "
                          "It does not release the kernel's hold, change any default, or "
                          "enter any product profile"),
}


# --------------------------------------------------------------------------- child


CHILD = r'''
import json, sys, time
sys.path.insert(0, {root!r})
sys.path.insert(0, {tools!r})
from b66_stack_proof import child_main
print("@@" + json.dumps(child_main(json.loads(sys.argv[1])), sort_keys=True, allow_nan=False), flush=True)
'''


def variant_kernel(simdgroups: int, results: int, fixed_k: int = 3840):
    """The owned kernel body at another geometry, named by the registry from its spec."""
    from ironmule import kernel_registry, qmv_k3840 as qmv

    source = qmv.BODY.format(
        num_simdgroups=simdgroups, results_per_simdgroup=results,
        pack_factor=qmv.PACK_FACTOR, bytes_per_pack=qmv.BYTES_PER_PACK,
        values_per_thread=qmv.VALUES_PER_THREAD, block_size=qmv.BLOCK_SIZE,
        group_size=qmv.GROUP_SIZE, scale_step_per_thread=qmv.SCALE_STEP_PER_THREAD,
        in_vec_size_decl=f"constexpr int in_vec_size = {fixed_k};",
        main_loop=qmv.FIXED_LOOP.replace("15", str(fixed_k // qmv.BLOCK_SIZE)), tail="")
    return kernel_registry.build(
        f"b66_stack_qmv_sg{simdgroups}_r{results}",
        input_names=["w", "scales", "biases", "x", "shape"], output_names=["out"],
        source=source, header=qmv.HEADER, ensure_row_contiguous=True,
        template={"fixed_k": fixed_k, "num_simdgroups": simdgroups,
                  "results_per_simdgroup": results})


def install_variant(model, geometry):
    """Swap every admissible K=3840 projection for the variant, and prove each one first.

    Returns the admitted paths and the byte-identity evidence. A projection whose variant
    output differs from the library on its own buffers is left on the library path and
    recorded, so a wrong kernel cannot become a timing.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from ironmule import qmv_k3840 as qmv

    simdgroups, results = geometry
    kernel = variant_kernel(simdgroups, results)
    rows_per_group = simdgroups * results
    admitted, declined, mismatched = [], [], []

    class VariantLinear(nn.Module):
        def __init__(self, source):
            super().__init__()
            self._source = source
            self.weight, self.scales, self.biases = source.weight, source.scales, source.biases
            self.group_size, self.bits = source.group_size, source.bits
            self.out_features = int(source.weight.shape[0])
            self.shape = qmv.shape_array(qmv.TARGET_K, self.out_features)
            self.groups = (self.out_features + rows_per_group - 1) // rows_per_group

        def __call__(self, x):
            if len(x.shape) != 3 or x.shape[0] != 1 or x.shape[1] != 1:
                return mx.quantized_matmul(x, self.weight, self.scales, self.biases,
                                           transpose=True, group_size=self.group_size,
                                           bits=self.bits)
            out = kernel(inputs=[self.weight, self.scales, self.biases, x[0], self.shape],
                         output_shapes=[(1, self.out_features)], output_dtypes=[x.dtype],
                         grid=(qmv.SIMD_SIZE, simdgroups * self.groups, 1),
                         threadgroup=(qmv.SIMD_SIZE, simdgroups, 1))[0]
            return out[None]

    def identical(module) -> bool:
        probe = mx.random.normal((1, 1, qmv.TARGET_K)).astype(module.scales.dtype)
        mx.eval(probe)
        want = mx.quantized_matmul(probe, module.weight, module.scales, module.biases,
                                   transpose=True, group_size=module.group_size,
                                   bits=module.bits)
        got = VariantLinear(module)(probe)
        mx.eval(want, got)
        return bytes(memoryview(mx.array(want))) == bytes(memoryview(mx.array(got)))

    def walk(parent, prefix=""):
        for name, child in list(parent.children().items()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, list):
                for index, item in enumerate(child):
                    if isinstance(item, nn.Module):
                        walk(item, f"{path}.{index}")
                continue
            if isinstance(child, nn.QuantizedLinear):
                columns = int(child.weight.shape[1]) * 32 // int(child.bits)
                if columns != qmv.TARGET_K:
                    continue
                if not qmv._admit_projection(child) or int(child.weight.shape[0]) % rows_per_group:
                    declined.append(path)
                    continue
                if not identical(child):
                    mismatched.append(path)
                    continue
                setattr(parent, name, VariantLinear(child))
                admitted.append(path)
            elif isinstance(child, nn.Module):
                walk(child, path)

    walk(model)
    return {"admitted": admitted, "declined": declined, "mismatched": mismatched,
            "geometry": {"num_simdgroups": simdgroups, "results_per_simdgroup": results}}


def child_main(spec: dict) -> dict:
    import importlib.util

    import mlx.core as mx
    from ironmule.hw import memory_pressure_level, vm_counters
    from ironmule.plans import ReusableSessionPlan, StrictOneShotPlan
    from ironmule.service import InteractiveMode, Request, Runtime
    from ironmule.tune import load_profile, resolve_local_model

    loader = importlib.util.spec_from_file_location(
        "b57", str(Path(spec["root"]) / "tools" / "b57_stack_composition.py"))
    b57 = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(b57)

    resolved = resolve_local_model(spec["model"])
    profile = load_profile(spec["model"], model_identity=resolved.identity)
    if profile is None:
        raise RuntimeError("no confirmed profile for this model; the reference stack needs one")

    out = {"pid": os.getpid(), "order": spec["order"], "arms": {}}
    for arm in spec["order"]:
        mx.reset_peak_memory()
        # `use_tuned_profile` is the default, so this is the confirmed stack's knob set.
        runtime = Runtime.load(spec["model"], mode=InteractiveMode())
        installed = None
        try:
            if arm == "candidate":
                installed = install_variant(runtime.engine.model,
                                            tuple(spec["geometry"]))
                if not installed["admitted"] or installed["mismatched"]:
                    raise RuntimeError(
                        f"variant not installable: admitted={len(installed['admitted'])} "
                        f"mismatched={len(installed['mismatched'])}")
            document_ids = runtime.encode(b57.DOCUMENT)
            session = runtime.session_plan(b57.DOCUMENT, name="b66")
            question_ids = [runtime.encode(b57.DOCUMENT + "\n\n" + q) for q in b57.QUESTIONS]
            plain_ids = [runtime.encode(q) for q in b57.QUESTIONS]

            def machine(label):
                counters = vm_counters() or {}
                return {"label": label, "load_average": list(os.getloadavg()),
                        "memory_pressure_level": memory_pressure_level(),
                        "swapouts": counters.get("swapouts"),
                        "mlx_peak_bytes": int(mx.get_peak_memory())}

            per_class = {}
            for name, count, plan_kind, max_tokens, _stack in spec["classes"]:
                before = machine(f"before_{name}")
                def build():
                    requests = []
                    for index in range(count):
                        if plan_kind == "strict":
                            requests.append(Request(prompt_ids=list(plain_ids[index % len(plain_ids)]),
                                                    max_tokens=max_tokens,
                                                    plan=StrictOneShotPlan(),
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
                    tokens = seen if tokens is None else tokens
                    stops = stop if stops is None else stops
                    if seen != tokens or stop != stops:
                        raise RuntimeError(f"{arm}/{name} is not deterministic within its own arm")
                snapshot = runtime.telemetry.snapshot()
                after = machine(f"after_{name}")
                per_class[name] = {
                    "machine_before": before, "machine_after": after,
                    "wall_ns": statistics.median(samples), "samples_ns": samples,
                    "tokens": tokens, "stop_reasons": stops,
                    "aggregate_tokens_per_second": snapshot["aggregate_tokens_per_second"],
                    "latency_p50_ms": snapshot["latency_p50_ms"],
                    "service_ttft_p50_ms": snapshot["service_ttft_p50_ms"],
                    "engine_ttft_p50_ms": snapshot["engine_ttft_p50_ms"],
                    "peak_memory_bytes": snapshot["peak_memory_bytes"],
                    "fallbacks": snapshot["fallbacks"],
                }
            out["arms"][arm] = {
                "classes": per_class,
                "installed": installed,
                "mlx_peak_bytes": int(mx.get_peak_memory()),
                "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            }
        finally:
            runtime.close()
    return out


# --------------------------------------------------------------------------- parent


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
    parser.add_argument("--exploratory-forced-load", action="store_true",
                        help="skip the START load check only, on explicit authorisation. "
                             "Every runtime gate stays: A/A, token and stop identity, the "
                             "per-projection byte check, drift, fallbacks and the B65 "
                             "resource gate. A run started this way can never be CONFIRMED")
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
    from ironmule.hw import (fingerprint, memory_pressure_level, static_facts,
                             swap_used_bytes, vm_counters)
    from ironmule.tune import gpu_busy

    busy = gpu_busy()
    if busy:
        raise SystemExit(f"another model process is running, refusing to measure ({busy})")
    load_at_start = os.getloadavg()
    if load_at_start[0] > 4.0 and not args.exploratory_forced_load:
        raise SystemExit(f"1-minute load average is {load_at_start[0]:.2f}, above 4.0")

    total_pages = (static_facts().get("memory_bytes") or 0) // 16384

    def probe(label):
        counters = vm_counters() or {}
        free_pages = (counters.get("free_count", 0) + counters.get("speculative_count", 0)
                      + counters.get("inactive_count", 0))
        return {"label": label, "swapouts": counters.get("swapouts"),
                "swapins": counters.get("swapins"),
                "pageouts": counters.get("pageouts"),
                "compressor_pages": counters.get("compressor_page_count"),
                "wired_pages": counters.get("wire_count"),
                "swap_used_bytes": swap_used_bytes(),
                "memory_pressure_level": memory_pressure_level(),
                "load_average": list(os.getloadavg()),
                # Native, and deliberately not the same quantity `memory_pressure -Q`
                # prints: free plus speculative plus inactive over installed pages.
                "memory_free_percent": (100.0 * free_pages / total_pages) if total_pages else None}

    samples = [probe("start")]
    children, failures = [], []
    for index in range(PAIRS):
        order = ["reference", "candidate"] if index % 2 == 0 else ["candidate", "reference"]
        spec = {"model": MODEL, "order": order, "geometry": list(GEOMETRY),
                "classes": [list(row) for row in CLASSES], "repeats": REPEATS,
                "warmups": WARMUPS, "root": str(PROJECT_ROOT)}
        process = subprocess.Popen(
            [sys.executable, "-u", "-c",
             CHILD.format(root=str(PROJECT_ROOT), tools=str(PROJECT_ROOT / "tools")),
             json.dumps(spec)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(PROJECT_ROOT),
            env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
        try:
            stdout, stderr = process.communicate(timeout=CHILD_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
        code = process.returncode
        marker = next((line[2:] for line in stdout.splitlines() if line.startswith("@@")), None)
        if code != 0 or marker is None:
            failures.append({"child": index, "returncode": code,
                             "signalled": code is not None and code < 0,
                             "signal_name": (signal.Signals(-code).name
                                             if code is not None and code < 0 else None),
                             "stderr_tail": stderr[-4000:]})
            break
        children.append(json.loads(marker))
        samples.append(probe(f"after_child_{index}"))
        print(f"child {index} done ({'/'.join(order)})", flush=True)
    samples.append(probe("end"))

    differences, comparisons = [], {}
    if children:
        for name, _c, _p, _t, _stack in CLASSES:
            for child in children:
                reference = child["arms"]["reference"]["classes"][name]
                candidate = child["arms"]["candidate"]["classes"][name]
                if candidate["tokens"] != reference["tokens"]:
                    differences.append({"pid": child["pid"], "class": name, "field": "tokens"})
                if candidate["stop_reasons"] != reference["stop_reasons"]:
                    differences.append({"pid": child["pid"], "class": name, "field": "stop_reason"})
            comparisons[name] = bootstrap([
                child["arms"]["candidate"]["classes"][name]["wall_ns"]
                / child["arms"]["reference"]["classes"][name]["wall_ns"] for child in children])

    swapouts = [s["swapouts"] for s in samples if s["swapouts"] is not None]
    swaps = [s["swap_used_bytes"] for s in samples if s["swap_used_bytes"] is not None]
    levels = [s["memory_pressure_level"] for s in samples]
    gate_ok = (bool(swaps) and max(swaps) <= swaps[0]
               and all(level == 1 for level in levels) and all(level is not None for level in levels))
    fallbacks = sum(arm["classes"][name]["fallbacks"] for child in children
                    for arm in child["arms"].values() for name, *_ in CLASSES)

    blocked = bool(failures) or bool(differences) or fallbacks or not gate_ok
    winners = {name: row for name, row in comparisons.items()
               if row["ci_high"] is not None and row["ci_high"] < 1.0}
    if args.exploratory_forced_load:
        # A run that skipped its readiness condition cannot be a confirmation, whatever it
        # measures. The label says so, and nothing downstream may quietly promote it.
        verdict = ("EXPLORATORY_INVALID" if blocked
                   else "EXPLORATORY_GAIN_UNDER_LOAD" if winners
                   else "EXPLORATORY_NO_GAIN_UNDER_LOAD")
    else:
        verdict = ("BLOCKED" if blocked
                   else "CHARACTERIZATION_GAIN" if winners else "NO_USEFUL_GAIN")

    record = {
        "experiment": PREREGISTRATION["experiment"],
        "preregistration": PREREGISTRATION,
        "source_binding": source_binding(),
        "environment": {"platform": platform.platform(), "mlx": mx.__version__,
                        "mlx_lm": mlx_lm.__version__, "fingerprint": fingerprint(),
                        "chip": static_facts().get("chip")},
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "exploratory_forced_load": bool(args.exploratory_forced_load),
        "readiness_condition_skipped": (
            "the start load check was skipped on explicit authorisation. Every runtime gate "
            "stayed in force. This run can never be CONFIRMED and its parameter can never "
            "enter silicon_profile as a product parameter"
            if args.exploratory_forced_load else None),
        "load_average_at_start": list(load_at_start),
        "load_average_at_end": list(os.getloadavg()),
        "children": children,
        "child_failures": failures,
        "correctness": {"differences": differences, "identical": not differences},
        "comparisons": comparisons,
        "resource_gate": {"applied": "pressure", "passed": gate_ok,
                          "swapout_counter_delta": (max(swapouts) - min(swapouts)) if len(swapouts) >= 2 else None,
                          "swap_growth_above_start_bytes": (max(swaps) - swaps[0]) if swaps else None,
                          "memory_pressure_levels": levels},
        "resource_samples": samples,
        "fallbacks": fallbacks,
        "verdict": verdict,
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"verdict": verdict, "failures": failures,
                      "correctness_identical": not differences,
                      "comparisons": {k: {"median": v["median"], "ci": [v["ci_low"], v["ci_high"]]}
                                      for k, v in comparisons.items()}}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
