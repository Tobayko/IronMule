#!/usr/bin/env python3
"""A fresh install learns this machine from nothing, and says how long that took.

Everything IronMule knows about this Mac was earned over many studies. `B75` asks the
question a new user actually faces: starting with no profile, no known winners and no
performance history, can the system measure its way to a defensible local decision, and what
does that cost in probes and in minutes?

**The isolation is structural, not a promise.** This file never opens a `B66`, `B69` or `B72`
record. It cannot consult a known winner because it has no path to one. Static facts a fresh
install would also have -- SoC, memory, GPU generation, library versions -- are allowed and
used. The historical evidence exists and stays sealed until `tools/b75_ground_truth.py` runs,
which happens only after the decision is written.

**The learner starts empty.** `qualified_actions` is the reference and nothing else. No
optimisation may be selected without evidence this run produced itself, and every gate that
guards a real qualification guards this one: byte identity per projection, an A/A arm,
drift, and the `B65` resource gate.

**The metric that matters is not the speedup.** It is `time_to_useful_hardware_knowledge`:
how much measurement had to happen before the machine could say something it was entitled to
act on. A system that finds the right answer after an hour of probing has not solved the
cold-start problem.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
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

TOOLS = PROJECT_ROOT / "tools"
MODEL = "mlx-community/gemma-3-12b-it-4bit"
GEOMETRY = (4, 8)
#: The decision. Everything else this file does exists to answer it.
DECISION_QUESTION = ("reference, or the (4, 8) SIMD geometry for the K=3840 quantised "
                     "matvec, on this machine, for single-token decode")
#: The deep probe's design, fixed here before it is needed: the blocked, rotated shape a
#: fast sequential pass cannot substitute for.
DEEP_BLOCKS = 10
DEEP_REPEATS = 5
#: The qualification run. Deliberately smaller than a full stack proof, and it qualifies only
#: what it measured.
QUALIFY_BLOCKS = 4
QUALIFY_REPEATS = 5
QUALIFY_WARMUPS = 2
QUALIFY_CLASSES = (("single_short", 1, "strict", 32, "reference"),)
ARMS = ("reference", "candidate", "reference_aa")
#: A candidate is adopted only if its interval clears the reference outright and the A/A arm
#: shows the machine was quiet enough for that to mean anything.
ADOPT_MAX_CI_HIGH = 1.0
AA_MAX_OFFSET = 0.05
AA_MAX_HALF_WIDTH = 0.05

PREREGISTRATION = {
    "experiment": "B75_cold_start_local_learning",
    "question": ("starting from nothing, can this machine measure its way to a defensible "
                 "local decision about a known kind of optimisation, and what does that "
                 "cost"),
    "not_a_cross_hardware_test": ("this says nothing about any other Mac. B73 is that test "
                                  "and it has not run for want of a second machine"),
    "isolation": ("this file opens no B66, B69 or B72 record. Static facts a fresh install "
                  "would also have are allowed. Historical evidence stays sealed until "
                  "tools/b75_ground_truth.py runs, after the decision is written"),
    "initial_state": {"knowledge": "empty", "confidence": "unknown",
                      "qualified_actions": ["reference"]},
    "decision_question": DECISION_QUESTION,
    "decisions_the_learner_may_take": ["MEASURE_MORE", "REFERENCE", "CANDIDATE"],
    "fast_probes": ("B71's probe set, unchanged. A feature whose own noise gate fails stays "
                    "missing"),
    "deep_probe": {"when": "only when a fast pass leaves a decision-relevant feature missing",
                   "what": "the geometry response, blocked and rotated",
                   "blocks": DEEP_BLOCKS, "repeats": DEEP_REPEATS,
                   "no_broad_search": "one geometry, the one under question, and the library"},
    "qualification": {"blocks": QUALIFY_BLOCKS, "repeats": QUALIFY_REPEATS,
                      "warmups": QUALIFY_WARMUPS,
                      "classes": [row[0] for row in QUALIFY_CLASSES],
                      "one_arm_per_process": True,
                      "gates": ("byte identity for every admitted projection before any "
                                "token is timed, token and stop-reason identity across "
                                "arms, an A/A arm, drift, and the B65 resource gate"),
                      "scope": ("it qualifies the classes it measured and nothing else. A "
                                "cold start does not get to generalise across workloads any "
                                "more than across machines")},
    "adoption_rule": (f"the candidate's 95 per cent interval must lie entirely below "
                      f"{ADOPT_MAX_CI_HIGH} against the reference, and the A/A arm must sit "
                      f"within {AA_MAX_OFFSET} of 1.0 with a half width at most "
                      f"{AA_MAX_HALF_WIDTH}. Otherwise the reference stands"),
    "primary_metric": ("time_to_useful_hardware_knowledge: the measurement time and probe "
                       "count spent before a decision this machine was entitled to act on. "
                       "The speedup is secondary"),
    "no_learning_from_blocked_or_invalid": True,
    "history_immutable": "every measurement is appended; none is rewritten",
}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), TOOLS / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bootstrap(ratios, resamples: int = 10000, seed: int = 20260910) -> dict:
    rng = random.Random(seed)
    if not ratios:
        return {"median": None, "ci_low": None, "ci_high": None, "n": 0}
    draws = sorted(median([rng.choice(ratios) for _ in ratios]) for _ in range(resamples))
    return {"median": median(ratios), "ci_low": draws[int(0.025 * resamples)],
            "ci_high": draws[int(0.975 * resamples) - 1], "n": len(ratios), "ratios": ratios}


# --------------------------------------------------------------------------- probes


def fast_probes(state: dict) -> dict:
    """B71's probe set, unchanged, and what it cost."""
    b71 = _load("b71_quick_characterization.py")
    started = time.perf_counter()
    evidence = "B75_fast"
    import mlx.core as mx

    bandwidth, curve = b71.probe_bandwidth(evidence)
    cache = b71.probe_cache(evidence, 48 * 1024 * 1024)
    alignment = b71.probe_alignment(evidence)
    widths = b71.probe_widths(evidence)
    fixed, marginal = b71.probe_eval(evidence)
    geometry = b71.probe_geometry(evidence)
    elapsed = time.perf_counter() - started

    usable = [m for m in geometry if m.context.get("usable")]
    measured = b71.MeasuredResponses(
        dram_bandwidth=bandwidth, bandwidth_by_working_set=curve,
        cache_residency_ratio=cache, k_alignment_ratio=alignment, width_response=widths,
        eval_fixed_cost=fixed, eval_marginal_cost=marginal,
        geometry_response=tuple(usable))
    relations = b71.relations_for(measured)
    features = {}
    for name, row in relations.items():
        features[name] = {"value": row["value"], "evidence_id": evidence,
                          "from": row["from"], "note": row["note"]}
    for entry in curve:
        pass
    return {
        "probe": "fast", "probe_set_id": b71.PROBE_SET_ID,
        "seconds": elapsed, "probe_count": 6,
        "features": features,
        "missing": [name for name in ("geometry_4_8_ratio", "cache_to_dram_ratio",
                                      "m16_cost_per_row_vs_m1")
                    if name not in features],
        "geometry_probe_detail": [m.as_dict() for m in geometry],
        "uncertainty": {name: (m.spread if m.spread is not None else None)
                        for name, m in (("cache_residency_ratio", cache),
                                        ("k_alignment_ratio", alignment),
                                        ("eval_fixed_cost", fixed))
                        if m is not None},
    }


def deep_geometry_probe() -> dict:
    """Blocked and rotated, which is the thing a fast sequential pass cannot substitute for."""
    import mlx.core as mx
    from ironmule import kernel_registry, qmv_k3840 as qmv

    b71 = _load("b71_quick_characterization.py")
    started = time.perf_counter()
    k, n = 3840, 15360
    weights, scales, biases = b71._quantised(k, n, 4)
    x = mx.random.normal((1, k)).astype(mx.bfloat16)
    mx.eval(x)
    reference = [mx.quantized_matmul(x, w, s, b, transpose=True,
                                     group_size=b71.GROUP_SIZE, bits=b71.BITS)
                 for w, s, b in zip(weights, scales, biases)]
    mx.eval(reference)
    wanted = [bytes(memoryview(mx.array(r))) for r in reference]
    shape = qmv.shape_array(k, n)
    simdgroups, results = GEOMETRY
    rows_per_group = simdgroups * results
    source = qmv.BODY.format(
        num_simdgroups=simdgroups, results_per_simdgroup=results,
        pack_factor=qmv.PACK_FACTOR, bytes_per_pack=qmv.BYTES_PER_PACK,
        values_per_thread=qmv.VALUES_PER_THREAD, block_size=qmv.BLOCK_SIZE,
        group_size=qmv.GROUP_SIZE, scale_step_per_thread=qmv.SCALE_STEP_PER_THREAD,
        in_vec_size_decl=f"constexpr int in_vec_size = {k};",
        main_loop=qmv.FIXED_LOOP.replace("15", str(k // qmv.BLOCK_SIZE)), tail="")
    kernel = kernel_registry.build(
        f"b75_deep_sg{simdgroups}_r{results}",
        input_names=["w", "scales", "biases", "x", "shape"], output_names=["out"],
        source=source, header=qmv.HEADER, ensure_row_contiguous=True,
        template={"fixed_k": k, "num_simdgroups": simdgroups,
                  "results_per_simdgroup": results})
    groups = n // rows_per_group

    def candidate():
        mx.eval([kernel(inputs=[w, s, b, x, shape], output_shapes=[(1, n)],
                        output_dtypes=[x.dtype],
                        grid=(qmv.SIMD_SIZE, simdgroups * groups, 1),
                        threadgroup=(qmv.SIMD_SIZE, simdgroups, 1))[0]
                 for w, s, b in zip(weights, scales, biases)])

    def library():
        mx.eval([mx.quantized_matmul(x, w, s, b, transpose=True,
                                     group_size=b71.GROUP_SIZE, bits=b71.BITS)
                 for w, s, b in zip(weights, scales, biases)])

    got = [kernel(inputs=[w, s, b, x, shape], output_shapes=[(1, n)],
                  output_dtypes=[x.dtype],
                  grid=(qmv.SIMD_SIZE, simdgroups * groups, 1),
                  threadgroup=(qmv.SIMD_SIZE, simdgroups, 1))[0]
           for w, s, b in zip(weights, scales, biases)]
    mx.eval(got)
    identical = all(bytes(memoryview(mx.array(o))) == want for o, want in zip(got, wanted))
    if not identical:
        return {"probe": "deep_geometry", "byte_identical": False,
                "seconds": time.perf_counter() - started,
                "usable": False,
                "reason": "not byte identical to the library on the same buffers; never timed"}

    arms = {"library": library, "candidate": candidate, "library_aa": library}
    names = list(arms)
    for call in arms.values():
        for _ in range(3):
            call()
    per_block = []
    for block in range(DEEP_BLOCKS):
        shift = block % len(names)
        order = names[shift:] + names[:shift]
        timings = {}
        for arm in order:
            samples = []
            for _ in range(DEEP_REPEATS):
                start = time.perf_counter_ns()
                arms[arm]()
                samples.append(time.perf_counter_ns() - start)
            timings[arm] = statistics.median(samples)
        per_block.append(timings)
    ratios = {arm: bootstrap([b[arm] / b["library"] for b in per_block])
              for arm in ("candidate", "library_aa")}
    aa = ratios["library_aa"]
    aa_half = (aa["ci_high"] - aa["ci_low"]) / 2 if aa["ci_high"] is not None else None
    aa_ok = (aa_half is not None and aa_half <= AA_MAX_HALF_WIDTH
             and abs(aa["median"] - 1.0) <= AA_MAX_OFFSET)
    return {"probe": "deep_geometry", "byte_identical": True, "usable": bool(aa_ok),
            "seconds": time.perf_counter() - started,
            "blocks": DEEP_BLOCKS, "repeats": DEEP_REPEATS,
            "ratios": ratios, "aa_half_width": aa_half, "aa_ok": aa_ok,
            "reason": ("" if aa_ok else "the A/A arm did not clear its gate, so nothing the "
                                        "candidate did here can be read"),
            "per_block": per_block}


# --------------------------------------------------------------------------- qualify


CHILD = r'''
import json, sys
sys.path.insert(0, {root!r})
sys.path.insert(0, {tools!r})
from b69_stack_proof import child_main
print("@@" + json.dumps(child_main(json.loads(sys.argv[1])), sort_keys=True, allow_nan=False), flush=True)
'''


def qualification_run() -> dict:
    """A small proof of its own, on this machine, using no historical number."""
    from ironmule.hw import (installed_memory_bytes, memory_pressure_level, swap_used_bytes,
                             vm_counters)
    import resource as resource_module

    started = time.perf_counter()
    total_pages = (installed_memory_bytes() or 0) // 16384

    def probe(label):
        counters = vm_counters() or {}
        free_pages = (counters.get("free_count", 0) + counters.get("speculative_count", 0)
                      + counters.get("inactive_count", 0))
        return {"label": label, "swap_used_bytes": swap_used_bytes(),
                "memory_pressure_level": memory_pressure_level(),
                "load_average": list(os.getloadavg()),
                "memory_free_percent": (100.0 * free_pages / total_pages) if total_pages else None}

    samples = [probe("start")]
    blocks, failures = [], []
    for index in range(QUALIFY_BLOCKS):
        shift = index % len(ARMS)
        order = ARMS[shift:] + ARMS[:shift]
        entry = {"block": index, "order": list(order), "children": {}}
        for arm in order:
            spec = {"model": MODEL, "arm": "reference" if arm == "reference_aa" else arm,
                    "geometry": list(GEOMETRY),
                    "classes": [list(row) for row in QUALIFY_CLASSES],
                    "repeats": QUALIFY_REPEATS, "warmups": QUALIFY_WARMUPS,
                    "root": str(PROJECT_ROOT)}
            process = subprocess.Popen(
                [sys.executable, "-u", "-c",
                 CHILD.format(root=str(PROJECT_ROOT), tools=str(TOOLS)), json.dumps(spec)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                cwd=str(PROJECT_ROOT),
                env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
            stdout, stderr = process.communicate(timeout=1800)
            marker = next((l[2:] for l in stdout.splitlines() if l.startswith("@@")), None)
            if process.returncode != 0 or marker is None:
                failures.append({"block": index, "arm": arm,
                                 "returncode": process.returncode,
                                 "stderr_tail": stderr[-2000:]})
                break
            entry["children"][arm] = json.loads(marker)
            print(f"  qualify block {index} {arm} done", flush=True)
        blocks.append(entry)
        samples.append(probe(f"after_block_{index}"))
        if failures:
            break
    samples.append(probe("end"))

    complete = [b for b in blocks if set(b["children"]) == set(ARMS)]
    differences, comparisons = [], {}
    for name, *_rest in QUALIFY_CLASSES:
        base_tokens = None
        for entry in complete:
            base = entry["children"]["reference"]["classes"][name]
            base_tokens = base_tokens or base["tokens"]
            for arm in ARMS:
                row = entry["children"][arm]["classes"][name]
                if row["tokens"] != base_tokens or row["stop_reasons"] != base["stop_reasons"]:
                    differences.append({"block": entry["block"], "arm": arm, "class": name})
        comparisons[name] = {
            arm: bootstrap([entry["children"][arm]["classes"][name]["wall_ns"]
                            / entry["children"]["reference"]["classes"][name]["wall_ns"]
                            for entry in complete]) for arm in ("candidate", "reference_aa")}

    walls = [sum(entry["children"][arm]["classes"][name]["wall_ns"] / 1e6
                 for arm in ARMS for name, *_r in QUALIFY_CLASSES) for entry in complete]
    wall_median = median(walls) if walls else None
    disturbed = ([i for i, w in enumerate(walls) if abs(w / wall_median - 1.0) > 0.25]
                 if wall_median else [])
    swaps = [s["swap_used_bytes"] for s in samples if s["swap_used_bytes"] is not None]
    frees = [s["memory_free_percent"] for s in samples if s["memory_free_percent"] is not None]
    levels = [s["memory_pressure_level"] for s in samples]
    fallbacks = sum(entry["children"][arm]["classes"][name]["fallbacks"]
                    for entry in complete for arm in ARMS for name, *_r in QUALIFY_CLASSES)
    gate = []
    if not swaps or max(swaps) > swaps[0]:
        gate.append("swap in use rose above its start")
    if not frees or min(frees) < 10.0:
        gate.append("free memory below 10 per cent")
    if not all(level == 1 and level is not None for level in levels):
        gate.append("memory pressure not normal at every probe")
    return {"probe": "qualification", "seconds": time.perf_counter() - started,
            "blocks": len(complete), "child_failures": failures,
            "correctness_identical": not differences, "differences": differences,
            "fallbacks": fallbacks, "disturbed_blocks": disturbed,
            "resource_gate_passed": not gate, "resource_gate_reasons": gate,
            "comparisons": comparisons, "resource_samples": samples,
            "children_run": sum(len(b["children"]) for b in blocks),
            "byte_check": ("performed inside every candidate child before any token was "
                           "timed, by the same installer the stack proof uses")}


# --------------------------------------------------------------------------- learner


def decide(state: dict) -> dict:
    """MEASURE_MORE, REFERENCE or CANDIDATE, with the reason and the uncertainty."""
    features = state["features"]
    geometry = features.get("geometry_4_8_ratio")
    if geometry is None:
        return {"decision": "MEASURE_MORE",
                "target": "geometry_4_8_ratio",
                "reason": ("the decision is about that geometry and the fast pass could not "
                           "measure it: its own noise gate refused to emit a value. Nothing "
                           "else in the vector answers the question"),
                "uncertainty": "unbounded for this question"}
    interval = geometry.get("ci")
    if interval is None:
        # A point estimate is not decision grade. The first run of this file treated a
        # fast-probe number without an interval as `no evidence for the candidate` and
        # answered REFERENCE, which is safe and is still the wrong reason: the honest
        # answer to `I have a number and no idea how good it is` is to go and measure.
        return {"decision": "MEASURE_MORE",
                "target": "geometry_4_8_ratio",
                "reason": (f"the fast pass produced a point estimate of {geometry['value']:.4f} "
                           "with no interval. A number without a spread cannot clear an "
                           "adoption rule that is stated as an interval, and treating it as "
                           "evidence against the candidate would be reading silence as an "
                           "answer"),
                "uncertainty": "point estimate only"}
    if interval[1] < ADOPT_MAX_CI_HIGH:
        return {"decision": "CANDIDATE",
                "reason": (f"the geometry's interval {interval} lies entirely below "
                           f"{ADOPT_MAX_CI_HIGH} against the library on this machine's own "
                           "buffers, and the A/A arm cleared its gate"),
                "uncertainty": (interval[1] - interval[0]) / 2}
    return {"decision": "REFERENCE",
            "reason": "no locally measured evidence puts the candidate below the reference",
            "uncertainty": None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preregister", type=Path)
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once

    if args.preregister:
        write_once(args.preregister, json.dumps(
            {"preregistration": PREREGISTRATION, "written_at":
             datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True))
        print(f"preregistration written to {args.preregister}")
        return 0

    from ironmule.hw import fingerprint, installed_memory_bytes, static_facts
    from ironmule.tune import gpu_busy
    import mlx.core as mx
    import mlx_lm

    busy = gpu_busy()
    if busy:
        raise SystemExit(f"another model process is running, refusing to measure ({busy})")

    wall_started = time.perf_counter()
    facts = static_facts()
    state = {
        "knowledge": "empty", "confidence": "unknown",
        "qualified_actions": ["reference"],
        "static_facts_a_fresh_install_also_has": {
            "hardware_fingerprint": fingerprint(), "chip": facts.get("chip"),
            "gpu_architecture": mx.device_info().get("architecture"),
            "gpu_cores": facts.get("gpu_cores"),
            "unified_memory_bytes": installed_memory_bytes(),
            "os_release": facts.get("os_release"),
            "mlx": mx.__version__, "mlx_lm": mlx_lm.__version__},
        "features": {}, "history": [],
    }

    print("fast probes", flush=True)
    fast = fast_probes(state)
    state["history"].append(fast)
    state["features"].update(fast["features"])
    first = decide(state)
    state["history"].append({"step": "decision", **first})
    print(f"  -> {first['decision']}: {first['reason'][:90]}", flush=True)

    deep = None
    if first["decision"] == "MEASURE_MORE":
        print("deep probe", flush=True)
        deep = deep_geometry_probe()
        state["history"].append(deep)
        if deep.get("usable"):
            row = deep["ratios"]["candidate"]
            state["features"]["geometry_4_8_ratio"] = {
                "value": row["median"], "ci": [row["ci_low"], row["ci_high"]],
                "evidence_id": "B75_deep", "from": ["B75_deep"],
                "note": "blocked and rotated, with an A/A arm that cleared its gate"}
        second = decide(state)
        state["history"].append({"step": "decision", **second})
        print(f"  -> {second['decision']}: {second['reason'][:90]}", flush=True)
        current = second
    else:
        current = first

    qualification = None
    if current["decision"] == "CANDIDATE":
        print("qualification run", flush=True)
        qualification = qualification_run()
        state["history"].append(qualification)
        blocked = (bool(qualification["child_failures"])
                   or not qualification["correctness_identical"]
                   or qualification["fallbacks"]
                   or qualification["disturbed_blocks"]
                   or not qualification["resource_gate_passed"])
        adopted = {}
        for name, row in qualification["comparisons"].items():
            candidate, aa = row["candidate"], row["reference_aa"]
            aa_half = ((aa["ci_high"] - aa["ci_low"]) / 2
                       if aa["ci_high"] is not None else None)
            aa_ok = (aa_half is not None and aa_half <= AA_MAX_HALF_WIDTH
                     and abs(aa["median"] - 1.0) <= AA_MAX_OFFSET)
            adopted[name] = {
                "candidate": candidate, "reference_aa": aa, "aa_half_width": aa_half,
                "aa_ok": aa_ok,
                "adopted": bool(not blocked and aa_ok and candidate["ci_high"] is not None
                                and candidate["ci_high"] < ADOPT_MAX_CI_HIGH),
            }
        qualification["adoption"] = adopted
        qualification["blocked"] = blocked
        if blocked:
            final = {"decision": "REFERENCE",
                     "reason": ("the qualification run did not pass its own gates, so "
                                "nothing it measured may be learned from"),
                     "uncertainty": None}
        elif all(row["adopted"] for row in adopted.values()):
            final = {"decision": "CANDIDATE",
                     "reason": ("qualified locally against the reference, with every gate "
                                "passed and an A/A arm that cleared"),
                     "uncertainty": {name: (row["candidate"]["ci_high"]
                                            - row["candidate"]["ci_low"]) / 2
                                     for name, row in adopted.items()}}
            state["qualified_actions"] = ["reference", "k3840_geometry_sg4_r8"]
        else:
            final = {"decision": "REFERENCE",
                     "reason": "the local qualification did not clear the adoption rule",
                     "uncertainty": None}
    else:
        final = current
    state["history"].append({"step": "final_decision", **final})

    elapsed = time.perf_counter() - wall_started
    probes = [h for h in state["history"] if "probe" in h]
    measurement_seconds = sum(h.get("seconds", 0.0) for h in probes)
    record = {
        "experiment": "B75_cold_start_local_learning",
        "preregistration": PREREGISTRATION,
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "isolation": ("this record was produced without opening any B66, B69 or B72 file. "
                      "The comparison against history is a separate tool and runs after this"),
        "environment": {"platform": platform.platform(), "mlx": mx.__version__,
                        "mlx_lm": mlx_lm.__version__},
        "initial_state": {"knowledge": "empty", "confidence": "unknown",
                          "qualified_actions": ["reference"]},
        "static_facts": state["static_facts_a_fresh_install_also_has"],
        "history": state["history"],
        "features_learned": state["features"],
        "qualified_actions_after": state["qualified_actions"],
        "final_decision": final,
        "cost": {
            "time_to_useful_hardware_knowledge_seconds": measurement_seconds,
            "wall_seconds_including_analysis": elapsed,
            "probe_count": len(probes),
            "probe_breakdown": {h["probe"]: round(h.get("seconds", 0.0), 2) for h in probes},
            "model_loads": (qualification["children_run"] if qualification else 0),
        },
        "source_binding": {n: hashlib.sha256((PROJECT_ROOT / n).read_bytes()).hexdigest()
                           for n in ("tools/b75_cold_start.py", "tools/b71_quick_characterization.py",
                                     "tools/b69_stack_proof.py", "ironmule/qmv_k3840.py")},
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"final_decision": final["decision"],
                      "qualified_actions": state["qualified_actions"],
                      "time_to_useful_knowledge_s": round(measurement_seconds, 1),
                      "probe_count": len(probes),
                      "breakdown": record["cost"]["probe_breakdown"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
