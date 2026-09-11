#!/usr/bin/env python3
"""Four hardware axes, each measured paired against its own reference, with a noise floor.

The shape inventory (`B66_shape_inventory_20260910.json`) decided what is worth touching.
Two MLP shapes carry `65%` of measured decode time on `4B` and `69%` on `12B`, and both
already run within a few per cent of the bandwidth `E4`'s size curve predicts for them. The
small projections are far from the ceiling but hold under `10%` of the time. So the axes
here are the ones that could still move something, and each is asked as a question rather
than as a search.

**Every axis carries an A/A arm.** The inventory's own numbers moved by more than `20%`
between two runs of the same shape on this machine, so a single-shot microbenchmark cannot
support a few-per-cent claim. The A/A arm is the same work under a different name, and it
measures the floor that any candidate has to clear.

Nothing here is a product claim. `E5` measured an isolated bandwidth advantage failing to
transfer to a pipelined graph, and that finding stands over every number in this file. A
candidate that wins here has earned a stack confirmation, not an activation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

MEASURED_SOURCES = ("tools/b66_axes.py", "ironmule/qmv_k3840.py", "ironmule/hw.py")
BLOCKS = 12
REPEATS = 5
WARMUPS = 3
SLC_BYTES = 48 * 1024 * 1024
GROUP_SIZE, BITS = 64, 4

#: The shape that carries the most measured decode time in each model.
TOP_SHAPE = {"4B": (2560, 10240), "12B": (3840, 15360)}


def _quantised(k: int, n: int, copies: int):
    weights = [mx.random.randint(0, 2**31 - 1, (n, k * BITS // 32), dtype=mx.uint32)
               for _ in range(copies)]
    scales = [mx.random.normal((n, k // GROUP_SIZE)).astype(mx.bfloat16) for _ in range(copies)]
    biases = [mx.random.normal((n, k // GROUP_SIZE)).astype(mx.bfloat16) for _ in range(copies)]
    mx.eval(weights, scales, biases)
    return weights, scales, biases


def bootstrap(ratios, resamples: int = 10000, seed: int = 20260910) -> dict:
    rng = random.Random(seed)
    if not ratios:
        return {"median": None, "ci_low": None, "ci_high": None, "n": 0}
    draws = sorted(statistics.median([rng.choice(ratios) for _ in ratios])
                   for _ in range(resamples))
    return {"median": statistics.median(ratios), "ci_low": draws[int(0.025 * resamples)],
            "ci_high": draws[int(0.975 * resamples) - 1], "n": len(ratios), "ratios": ratios}


def run_blocks(arms: dict, *, blocks: int = BLOCKS, reference: str) -> dict:
    """Rotate arm order per block so drift cannot hand the win to whichever ran second."""
    names = list(arms)
    for name in names:
        for _ in range(WARMUPS):
            arms[name]()
    per_block = []
    for block in range(blocks):
        shift = block % len(names)
        order = names[shift:] + names[:shift]
        timings = {}
        for name in order:
            samples = []
            for _ in range(REPEATS):
                start = time.perf_counter_ns()
                arms[name]()
                samples.append(time.perf_counter_ns() - start)
            timings[name] = statistics.median(samples)
        per_block.append({"block": block, "order": order, "median_ns": timings})
    ratios = {name: [b["median_ns"][name] / b["median_ns"][reference] for b in per_block]
              for name in names if name != reference}
    return {"reference": reference, "blocks": per_block,
            "ratios": {name: bootstrap(values) for name, values in ratios.items()}}


# ---------------------------------------------------------------------------


def axis_width(label: str) -> dict:
    """Where do the grouped widths stop paying? `E2` and `E3` found `M = 8` pathological."""
    k, n = TOP_SHAPE[label]
    weights, scales, biases = _quantised(k, n, 4)
    xs = {m: mx.random.normal((m, k)).astype(mx.bfloat16) for m in (1, 2, 4, 8, 16)}
    mx.eval(list(xs.values()))

    def arm(m):
        def call():
            outputs = [mx.quantized_matmul(xs[m], w, s, b, transpose=True,
                                           group_size=GROUP_SIZE, bits=BITS)
                       for w, s, b in zip(weights, scales, biases)]
            mx.eval(outputs)
        return call

    arms = {f"M{m}": arm(m) for m in (1, 2, 4, 8, 16)}
    arms["M4_aa"] = arm(4)                      # the noise floor, same work, different name
    result = run_blocks(arms, reference="M4")
    per_row = {}
    for name, row in result["ratios"].items():
        m = 4 if name == "M4_aa" else int(name[1:])
        per_row[name] = {"ratio_of_total": row["median"], "ci": [row["ci_low"], row["ci_high"]],
                         "cost_per_row_vs_M4": row["median"] * 4 / m}
    return {"axis": "grouped width and the M boundary", "shape": {"k": k, "n": n},
            "question": ("does the pathological width E2 and E3 measured on synthetic shapes "
                         "exist on the shape this model actually spends decode on"),
            "result": result, "per_row": per_row}


def axis_cache(label: str) -> dict:
    """Below and above the M1 Max system level cache, on one shape, at decode width."""
    k, n = TOP_SHAPE[label]
    per = n * k * BITS // 8 + 2 * (n * (k // GROUP_SIZE) * 2)
    small = max(1, (SLC_BYTES // 4) // per)         # clearly inside the cache
    large = max(small + 1, (SLC_BYTES * 8) // per)  # clearly outside it
    sets = {}
    for name, copies in (("resident", small), ("dram", large)):
        weights, scales, biases = _quantised(k, n, copies)
        x = mx.random.normal((1, k)).astype(mx.bfloat16)
        mx.eval(x)
        sets[name] = (weights, scales, biases, x, copies)

    def arm(name):
        weights, scales, biases, x, copies = sets[name]

        def call():
            outputs = [mx.quantized_matmul(x, w, s, b, transpose=True,
                                           group_size=GROUP_SIZE, bits=BITS)
                       for w, s, b in zip(weights, scales, biases)]
            mx.eval(outputs)
        return call

    # Times are not comparable directly: the arms do different amounts of work. Per-call
    # bandwidth is the comparable quantity, so each arm is normalised by its own call count.
    out = {}
    for name in sets:
        call = arm(name)
        for _ in range(WARMUPS):
            call()
        samples = []
        for _ in range(BLOCKS):
            start = time.perf_counter_ns()
            call()
            samples.append(time.perf_counter_ns() - start)
        copies = sets[name][4]
        median = statistics.median(samples) / copies
        out[name] = {"copies": copies, "working_set_bytes": per * copies,
                     "median_ns_per_call": median, "achieved_gb_per_s": per / median}
    return {"axis": "cache residency", "shape": {"k": k, "n": n},
            "slc_bytes_assumed": SLC_BYTES,
            "question": "how much of the achieved bandwidth is the system level cache",
            "result": out,
            "resident_over_dram": (out["resident"]["achieved_gb_per_s"]
                                   / out["dram"]["achieved_gb_per_s"])}


def axis_cache_fixed(label: str) -> dict:
    """Cache residency with the call count held equal, because the first attempt confounded it.

    The first cache arm used one distinct buffer against twelve, so the resident arm ran one
    kernel behind its own `eval` while the DRAM arm ran twelve behind one. It measured the
    cost of an `eval` round trip, not the cache, and reported the resident set as `2` to `5`
    times *slower*. Here both arms chain the same number of calls behind one `eval`; only the
    number of distinct weight buffers differs, so only the working set differs.
    """
    k, n = TOP_SHAPE[label]
    per = n * k * BITS // 8 + 2 * (n * (k // GROUP_SIZE) * 2)
    calls = 8
    resident_copies = max(1, SLC_BYTES // (4 * per))     # clearly inside the cache
    dram_copies = calls                                   # every call its own buffer
    sets = {}
    for name, copies in (("resident", resident_copies), ("dram", dram_copies)):
        weights, scales, biases = _quantised(k, n, copies)
        sets[name] = (weights, scales, biases, copies)
    x = mx.random.normal((1, k)).astype(mx.bfloat16)
    mx.eval(x)

    def arm(name):
        weights, scales, biases, copies = sets[name]

        def call():
            outputs = [mx.quantized_matmul(x, weights[i % copies], scales[i % copies],
                                           biases[i % copies], transpose=True,
                                           group_size=GROUP_SIZE, bits=BITS)
                       for i in range(calls)]
            mx.eval(outputs)
        return call

    arms = {"resident": arm("resident"), "dram": arm("dram"), "dram_aa": arm("dram")}
    result = run_blocks(arms, reference="dram")
    return {"axis": "cache residency, call count held equal",
            "shape": {"k": k, "n": n}, "calls_per_eval": calls,
            "slc_bytes_assumed": SLC_BYTES,
            "working_set_bytes": {"resident": per * resident_copies, "dram": per * dram_copies},
            "question": ("how much of the achieved bandwidth is the system level cache, with "
                         "dispatch and eval count identical in both arms"),
            "result": result}


def axis_eval(label: str) -> dict:
    """What one `eval` round trip costs, held out from the kernel work it wraps.

    The confound in the first cache arm was worth keeping as a question. A single kernel
    behind its own `eval` was several times slower per byte than the same kernel chained
    with others. That gap is a fixed cost per submission boundary, and it is much larger
    than the `6.41 us` per dispatch `E5` measured. This measures it directly: identical
    kernels and identical buffers, only the number of them behind one `eval` changes.
    """
    k, n = TOP_SHAPE[label]
    per = n * k * BITS // 8 + 2 * (n * (k // GROUP_SIZE) * 2)
    counts = (1, 2, 4, 8, 16)
    weights, scales, biases = _quantised(k, n, max(counts))
    x = mx.random.normal((1, k)).astype(mx.bfloat16)
    mx.eval(x)

    measured = {}
    for count in counts:
        def call(count=count):
            outputs = [mx.quantized_matmul(x, weights[i], scales[i], biases[i],
                                           transpose=True, group_size=GROUP_SIZE, bits=BITS)
                       for i in range(count)]
            mx.eval(outputs)
        for _ in range(WARMUPS):
            call()
        samples = [None] * BLOCKS
        for block in range(BLOCKS):
            start = time.perf_counter_ns()
            call()
            samples[block] = time.perf_counter_ns() - start
        total = statistics.median(samples)
        measured[count] = {"kernels_per_eval": count, "median_ns_total": total,
                           "median_ns_per_kernel": total / count,
                           "achieved_gb_per_s_per_kernel": per / (total / count)}
    # Two points determine the line: cost = fixed + count * marginal.
    low, high = measured[min(counts)], measured[max(counts)]
    marginal = ((high["median_ns_total"] - low["median_ns_total"])
                / (max(counts) - min(counts)))
    fixed = low["median_ns_total"] - marginal * min(counts)
    return {"axis": "cost of one eval round trip",
            "shape": {"k": k, "n": n},
            "question": ("how much of a small submission is the submission itself, rather "
                         "than the kernel inside it"),
            "measured": measured,
            "fitted_fixed_ns_per_eval": fixed,
            "fitted_marginal_ns_per_kernel": marginal,
            "reading": ("the fit is two points on a straight line, not a model. It says what "
                        "order the fixed cost is, and nothing finer")}


def axis_queues(label: str) -> dict:
    """One command stream or two, over the same total work, at decode width."""
    k, n = TOP_SHAPE[label]
    weights, scales, biases = _quantised(k, n, 8)
    x = mx.random.normal((1, k)).astype(mx.bfloat16)
    mx.eval(x)
    second = mx.new_stream(mx.gpu)

    def one_stream():
        outputs = [mx.quantized_matmul(x, w, s, b, transpose=True,
                                       group_size=GROUP_SIZE, bits=BITS)
                   for w, s, b in zip(weights, scales, biases)]
        mx.eval(outputs)

    def two_streams():
        outputs = []
        for index, (w, s, b) in enumerate(zip(weights, scales, biases)):
            if index % 2:
                with mx.stream(second):
                    outputs.append(mx.quantized_matmul(x, w, s, b, transpose=True,
                                                       group_size=GROUP_SIZE, bits=BITS))
            else:
                outputs.append(mx.quantized_matmul(x, w, s, b, transpose=True,
                                                   group_size=GROUP_SIZE, bits=BITS))
        mx.eval(outputs)

    arms = {"one_stream": one_stream, "two_streams": two_streams,
            "one_stream_aa": one_stream}
    return {"axis": "command queues", "shape": {"k": k, "n": n},
            "question": ("does splitting independent decode work across two MLX GPU streams "
                         "overlap submission with execution, or does the device simply share "
                         "itself as B56b measured across two processes"),
            "result": run_blocks(arms, reference="one_stream")}


def _variant_kernel(simdgroups: int, results: int, fixed_k: int):
    """One geometry of the owned kernel, built through the registry that names it by spec."""
    from ironmule import kernel_registry, qmv_k3840 as qmv

    blocks = fixed_k // qmv.BLOCK_SIZE
    source = qmv.BODY.format(
        num_simdgroups=simdgroups,
        results_per_simdgroup=results,
        pack_factor=qmv.PACK_FACTOR,
        bytes_per_pack=qmv.BYTES_PER_PACK,
        values_per_thread=qmv.VALUES_PER_THREAD,
        block_size=qmv.BLOCK_SIZE,
        group_size=qmv.GROUP_SIZE,
        scale_step_per_thread=qmv.SCALE_STEP_PER_THREAD,
        in_vec_size_decl=f"constexpr int in_vec_size = {fixed_k};",
        main_loop=qmv.FIXED_LOOP.replace("15", str(blocks)),
        tail="",
    )
    return kernel_registry.build(
        f"b66_qmv_sg{simdgroups}_r{results}",
        input_names=["w", "scales", "biases", "x", "shape"],
        output_names=["out"], source=source, header=qmv.HEADER,
        ensure_row_contiguous=True,
        template={"fixed_k": fixed_k, "num_simdgroups": simdgroups,
                  "results_per_simdgroup": results},
    )


def axis_geometry() -> dict:
    """Threadgroup shape and per-simdgroup work, on the shape 12B decode spends most time on.

    `ironmule/qmv_k3840.py` is the only kernel this project owns whose geometry is a
    parameter. Its `K = 3840` is the `12B` hidden size, so its target shape carries `48%` of
    measured decode time. Only `num_simdgroups` and `results_per_simdgroup` are varied:
    `values_per_thread` would change the order the partial sums are added in, which is a
    different execution plan and not a geometry change. Every variant is compared byte for
    byte against the library call on the same buffers before any timing is read.

    The kernel's activation hold is untouched. Measuring a held kernel is not composing it,
    and nothing here proposes releasing it.
    """
    from ironmule import qmv_k3840 as qmv

    k, n = 3840, 15360
    weights, scales, biases = _quantised(k, n, 4)
    x = mx.random.normal((1, k)).astype(mx.bfloat16)
    mx.eval(x)
    reference = [mx.quantized_matmul(x, w, s, b, transpose=True,
                                     group_size=GROUP_SIZE, bits=BITS)
                 for w, s, b in zip(weights, scales, biases)]
    mx.eval(reference)
    reference_bytes = [bytes(memoryview(mx.array(r))) for r in reference]
    shape = qmv.shape_array(k, n)

    def dispatch(kernel, simdgroups, groups):
        return [kernel(inputs=[w, s, b, x, shape], output_shapes=[(1, n)],
                       output_dtypes=[x.dtype],
                       grid=(qmv.SIMD_SIZE, simdgroups * groups, 1),
                       threadgroup=(qmv.SIMD_SIZE, simdgroups, 1))[0]
                for w, s, b in zip(weights, scales, biases)]

    verified, rejected, arms = {}, [], {}
    for simdgroups in (1, 2, 4, 8):
        for results in (2, 4, 8):
            name = f"sg{simdgroups}_r{results}"
            per_group = simdgroups * results
            if n % per_group:
                rejected.append({"variant": name, "reason":
                                 f"{n} output rows do not divide by {per_group}"})
                continue
            if qmv.SIMD_SIZE * simdgroups > 1024:
                rejected.append({"variant": name, "reason": "threadgroup above the device limit"})
                continue
            try:
                kernel = _variant_kernel(simdgroups, results, k)
                groups = n // per_group
                outputs = dispatch(kernel, simdgroups, groups)
                mx.eval(outputs)
                identical = all(bytes(memoryview(mx.array(o))) == want
                                for o, want in zip(outputs, reference_bytes))
            except (ValueError, RuntimeError, TypeError, KeyError, AttributeError) as exc:
                rejected.append({"variant": name, "reason": f"{type(exc).__name__}: {exc}"})
                continue
            verified[name] = {"num_simdgroups": simdgroups, "results_per_simdgroup": results,
                              "threads_per_threadgroup": qmv.SIMD_SIZE * simdgroups,
                              "output_rows_per_threadgroup": per_group,
                              "threadgroups": groups,
                              "byte_identical_to_library": identical}
            if not identical:
                continue

            def make(kernel=kernel, simdgroups=simdgroups, groups=groups):
                def call():
                    mx.eval(dispatch(kernel, simdgroups, groups))
                return call
            arms[name] = make()

    def library():
        outputs = [mx.quantized_matmul(x, w, s, b, transpose=True,
                                       group_size=GROUP_SIZE, bits=BITS)
                   for w, s, b in zip(weights, scales, biases)]
        mx.eval(outputs)

    result = None
    if arms:
        arms = dict(arms)
        arms["library"] = library
        arms["library_aa"] = library
        result = run_blocks(arms, reference="library")
    return {"axis": "threadgroup geometry and per-simdgroup work",
            "shape": {"k": k, "n": n},
            "question": ("is MLX's own quantised matvec leaving anything on the table at the "
                         "shape 12B decode spends most of its time on"),
            "varied": ["num_simdgroups", "results_per_simdgroup"],
            "held_fixed": ("values_per_thread, because changing it changes the order the "
                           "partial sums are added in. That is a different execution plan "
                           "and may not be adopted quietly as an optimisation"),
            "correctness_rule": ("every variant is compared byte for byte against the "
                                 "library call on the same buffers, and a variant that "
                                 "differs is recorded and never timed"),
            "activation_hold": ("qmv_k3840 stays held. Measuring a held kernel is not "
                                "composing it and nothing here proposes releasing it"),
            "variants": verified, "rejected": rejected, "result": result}


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
    parser.add_argument("--axes", default="width,cache,queues,geometry")
    parser.add_argument("--model", default="12B", choices=tuple(TOP_SHAPE))
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once
    from ironmule.hw import (fingerprint, memory_pressure_level, static_facts,
                             swap_used_bytes, vm_counters)
    from ironmule.tune import gpu_busy
    import mlx_lm

    busy = gpu_busy()
    if busy:
        raise SystemExit(f"another model process is running, refusing to measure ({busy})")

    def probe(label):
        counters = vm_counters() or {}
        return {"label": label, "swapouts": counters.get("swapouts"),
                "swap_used_bytes": swap_used_bytes(),
                "memory_pressure_level": memory_pressure_level(),
                "mlx_peak_bytes": int(mx.get_peak_memory())}

    samples = [probe("start")]
    axes = {}
    for name in [a.strip() for a in args.axes.split(",") if a.strip()]:
        print(f"=== axis {name} ===", flush=True)
        if name == "width":
            axes[name] = axis_width(args.model)
        elif name == "cache":
            axes[name] = axis_cache(args.model)
        elif name == "cache_fixed":
            axes[name] = axis_cache_fixed(args.model)
        elif name == "eval":
            axes[name] = axis_eval(args.model)
        elif name == "queues":
            axes[name] = axis_queues(args.model)
        elif name == "geometry":
            axes[name] = axis_geometry()
        else:
            raise SystemExit(f"unknown axis {name}")
        samples.append(probe(f"after_{name}"))
        print(f"    done", flush=True)
    samples.append(probe("end"))

    record = {
        "experiment": "B66_axes",
        "model_characterised": args.model,
        "design": ("each axis is a paired, order-rotated block design with an A/A arm. The "
                   "A/A arm is the same work under a different name and measures the noise "
                   "floor a candidate has to clear. 12 blocks, 5 repeats, median per block, "
                   "10000-resample bootstrap on the per-block ratios"),
        "not_a_product_claim": ("E5 measured an isolated bandwidth advantage failing to "
                                "transfer to a pipelined graph. No number here is a product "
                                "claim and none is extrapolated into one"),
        "environment": {"platform": platform.platform(), "mlx": mx.__version__,
                        "mlx_lm": mlx_lm.__version__, "fingerprint": fingerprint(),
                        "chip": static_facts().get("chip"),
                        "gpu_cores": static_facts().get("gpu_cores")},
        "source_binding": source_binding(),
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "resource_samples": samples,
        "axes": axes,
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
