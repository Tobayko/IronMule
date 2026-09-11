#!/usr/bin/env python3
"""A machine describes itself in a few minutes, before anyone has tuned anything on it.

Every probe here is an already-established mechanism, not a search. `E4` measured achieved
bandwidth rising with matrix size and saturating; `B66` measured cache residency with the
call count held equal, the cost per row against grouped width with its step at `16`, the
fixed cost of a submission boundary, and the effect of SIMD geometry with rows per simdgroup
dominating threads per threadgroup; `B68` measured that MLX's command buffer limits carry
none of it and closed that axis. Nothing is swept open-endedly and no autotuning happens.

**It runs on a machine that has never been tuned.** No model is loaded, no profile is read,
no `silicon_profile` needs to exist. The probes are synthetic quantised matmuls at the
shapes a Gemma-class model really runs, so a new Mac can answer before it has earned
anything.

**The geometry probe is a probe.** It builds a candidate kernel, checks it byte for byte
against the library on the same buffers, and times it. It activates nothing, and a variant
that is not byte identical is recorded as such and never timed.

**One machine proves nothing about hardware in general.** This produces a vector for the
machine it runs on. Whether such a vector describes optimisation response better than a chip
name is a question that needs a second machine, and the preregistration says exactly which
numbers from one would settle it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ironmule.characterization import (SCHEMA, Conditions, HardwareCharacterizationVector,  # noqa: E402
                                       Measurement, MeasuredResponses, StaticFacts,
                                       relations_for)

PROBE_SET_ID = "ironmule.quick_characterization.v3"
MEASURED_SOURCES = ("tools/b71_quick_characterization.py", "ironmule/characterization.py",
                    "ironmule/qmv_k3840.py")
GROUP_SIZE, BITS = 64, 4
K, N = 3840, 15360                 # a Gemma 12B MLP projection: the shape decode lives on
ALIGNED_K = 4096                   # the nearest power-of-two K, at the same output width
WORKING_SETS = (1024, 4096, 16384, 65536)   # output widths, smallest matrix first
WIDTHS = (1, 2, 4, 8, 16)
CHAIN_LENGTHS = (1, 2, 4, 8, 16)
GEOMETRIES = ((4, 8),)             # the one `B69` confirmed on this machine, as a probe
DISTINCT_BYTES = 512 * 1024 * 1024
REPEATS = 25
WARMUPS = 5
#: `v1` ran in `1.1 s` of a `420 s` budget with a geometry spread of `0.58` against a ratio
#: of `0.95`. `v2` raised repeats and added an A/A arm, which then measured a noise floor of
#: `0.9885 +/- 0.606` and made the candidate at `1.0024` unreadable. `v3` keeps that design
#: and adds the consequence: a geometry measurement is emitted only when its own A/A arm
#: says the machine was quiet enough to have measured anything. Earlier runs are kept.

#: Declared before v3 ran. `B66` measured an A/A arm of `1.0109` with a `95 per cent`
#: interval inside `+/- 3 per cent` on this same quantity, so a floor an order looser than
#: that is generous rather than convenient.
GEOMETRY_AA_MAX_SPREAD = 0.10
GEOMETRY_AA_MAX_OFFSET = 0.05
#: Declared before anything runs. A probe set that needs longer than this is not a quick
#: characterisation and the run says so rather than quietly taking the time.
BUDGET_SECONDS = 420.0

PREREGISTRATION = {
    "experiment": "B71_quick_characterization",
    "the_decision_this_is_for": {
        "primary": ("can a Mac nobody has tuned estimate whether a known optimisation is "
                    "worth trying on it? Concretely: the K=3840 SIMD geometry (4, 8), which "
                    "B69 confirmed against the complete 12B stack here. Today that question "
                    "costs a full tune plus a six-block stack proof. The claim to be tested "
                    "later is that a few minutes of probing predicts the sign"),
        "secondary": ("which known machine does this one behave like? Not which chip it is "
                      "called -- a fingerprint already answers that exactly and predicts "
                      "nothing"),
        "why_the_features_could_carry_it": (
            "the geometry that won here raises output rows per simdgroup, which raises "
            "arithmetic intensity per loaded activation. Whether that pays depends on how "
            "far the machine is from its bandwidth ceiling at that matrix size, which is "
            "exactly what the bandwidth-by-working-set and cache-residency probes measure. "
            "If those two carried no relation to the geometry response across machines, the "
            "vector would be a summary table and this entry would close as "
            "B71_DATA_CONTRACT_ONLY"),
    },
    "probe_set_id": PROBE_SET_ID,
    "probes": {
        "bandwidth_by_working_set": f"K={K}, M=1, output widths {list(WORKING_SETS)}, at "
                                    f"least {DISTINCT_BYTES // (1 << 20)} MB of distinct "
                                    "weight buffers per point (E4's method)",
        "cache_residency": "the same shape with the call count and eval count held equal in "
                           "both arms, only the number of distinct buffers differing "
                           "(B66's corrected design, after its first attempt confounded "
                           "dispatch with residency)",
        "k_alignment": f"K={K} against K={ALIGNED_K} at the same output width, compared per "
                       "weight byte because the two move different amounts",
        "width_response": f"widths {list(WIDTHS)} on K={K}, N={N}, reported as cost per row",
        "eval_cost": f"{list(CHAIN_LENGTHS)} identical kernels behind one eval, fixed and "
                     "marginal cost from the two endpoints",
        "geometry": f"{list(GEOMETRIES)} against the library call on the same buffers, "
                    "byte checked before being timed, with an A/A arm as the noise floor. "
                    f"The number is emitted only if that arm's relative spread is at most "
                    f"{GEOMETRY_AA_MAX_SPREAD} and it sits within {GEOMETRY_AA_MAX_OFFSET} "
                    "of 1.0; otherwise the field stays missing. A probe, not an activation",
    },
    "budget_seconds": BUDGET_SECONDS,
    "why_v3": ("v1 ran in 1.1 seconds of a 420 second budget with a geometry spread of 0.58 "
               "against a ratio of 0.95. v2 raised repeats from 7 to 25, warmups from 3 to "
               "5, and added an A/A arm, which then measured a noise floor of 0.9885 with a "
               "relative spread of 0.606 and made the candidate at 1.0024 unreadable. v3 "
               "keeps that design and draws the consequence: a geometry number is emitted "
               "only when its own A/A arm says the machine was quiet enough to have "
               "measured anything, and otherwise the field is missing. v3 also fixes a unit "
               "error in v2, which recorded a nanosecond half range as the spread of a "
               "bytes-per-nanosecond value. Every earlier run is kept as it was measured"),
    "no_open_search": ("every probe is a mechanism this project already measured. No "
                       "autotuning, no geometry search, and the command buffer axis is not "
                       "probed at all because B68 closed it with DEFAULT_WINS"),
    "missing_stays_missing": ("a probe that does not run, or whose correctness check fails, "
                             "leaves its field empty. Nothing is imputed"),
    "resource_gate": ("B65 in full from native probes: macOS pressure normal at every "
                      "sample, free memory at or above 10 per cent, swap in use never above "
                      "its start. A run that fails it is recorded and its vector is marked "
                      "invalid rather than used"),
    "hypothesis_H1": ("a compact hardware vector describes optimisation response better than "
                      "a chip name or fingerprint alone"),
    "what_one_machine_can_reach": ("VECTOR_IMPLEMENTED_AND_SELF_CONSISTENT and no more. H1 "
                                   "needs variation across machines and there is one here"),
    "what_a_second_mac_would_settle": (
        "run this same probe set on a Mac with a different memory system -- an M-series "
        "with materially different bandwidth or cache, an M4 Pro or an M1 without the Max "
        "memory width -- and then run B69's stack proof there for the same (4, 8) geometry. "
        "H1 survives if the geometry response measured in minutes has the same sign as the "
        "stack result measured in hours, on both machines, and if the bandwidth and cache "
        "relations differ between them in the direction that predicts it. H1 is refuted if "
        "the two machines produce similar vectors and opposite stack results, or similar "
        "stack results from clearly different vectors. Two machines settle the sign; they "
        "do not settle a magnitude"),
    "nothing_is_activated": ("no router change, no kernel release, no product profile, no "
                             "learned policy. B72 may train offline on the exported rows; "
                             "that is a separate entry"),
}


def _quantised(k: int, n: int, copies: int):
    weights = [mx.random.randint(0, 2**31 - 1, (n, k * BITS // 32), dtype=mx.uint32)
               for _ in range(copies)]
    scales = [mx.random.normal((n, k // GROUP_SIZE)).astype(mx.bfloat16) for _ in range(copies)]
    biases = [mx.random.normal((n, k // GROUP_SIZE)).astype(mx.bfloat16) for _ in range(copies)]
    mx.eval(weights, scales, biases)
    return weights, scales, biases


def weight_bytes(k: int, n: int) -> int:
    return n * k * BITS // 8 + 2 * (n * (k // GROUP_SIZE) * 2)


def _time(call, repeats: int = REPEATS, warmups: int = WARMUPS) -> tuple[float, float, list]:
    """Median, half range, and the samples themselves. The spread is never optional."""
    for _ in range(warmups):
        call()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        call()
        samples.append(time.perf_counter_ns() - start)
    spread = (max(samples) - min(samples)) / 2 if len(samples) > 1 else None
    return statistics.median(samples), spread, samples


def _matmul_chain(weights, scales, biases, x, k: int):
    def call():
        mx.eval([mx.quantized_matmul(x, w, s, b, transpose=True,
                                     group_size=GROUP_SIZE, bits=BITS)
                 for w, s, b in zip(weights, scales, biases)])
    return call


def probe_bandwidth(evidence: str) -> tuple[Measurement | None, tuple[Measurement, ...]]:
    points = []
    for width in WORKING_SETS:
        per = weight_bytes(K, width)
        copies = max(2, min(32, DISTINCT_BYTES // max(per, 1)))
        weights, scales, biases = _quantised(K, width, copies)
        x = mx.random.normal((1, K)).astype(mx.bfloat16)
        mx.eval(x)
        median, spread, samples = _time(_matmul_chain(weights, scales, biases, x, K))
        per_call = median / copies
        # The spread has to be in the unit of the value. A half range in nanoseconds is not
        # a half range in bytes per nanosecond, and v2 wrote one as if it were.
        fastest, slowest = min(samples) / copies, max(samples) / copies
        rate_spread = (per / fastest - per / slowest) / 2 if fastest and slowest else None
        points.append(Measurement(
            value=per / per_call, unit="bytes_per_ns", evidence_id=evidence,
            spread=rate_spread, samples=REPEATS,
            context={"k": K, "n": width, "m": 1, "weight_bytes": per,
                     "distinct_copies": copies, "working_set_bytes": per * copies,
                     "ns_per_call": per_call, "ns_per_call_half_range": (spread / copies)
                     if spread else None}))
    largest = max(points, key=lambda p: p.context["weight_bytes"]) if points else None
    return largest, tuple(points)


def probe_cache(evidence: str, cache_bytes: int) -> Measurement | None:
    """Both arms run the same number of kernels behind one eval. Only the buffers differ."""
    calls = 8
    per = weight_bytes(K, N)
    resident_copies = max(1, cache_bytes // (4 * per))
    x = mx.random.normal((1, K)).astype(mx.bfloat16)
    mx.eval(x)
    timings = {}
    for name, copies in (("resident", resident_copies), ("dram", calls)):
        weights, scales, biases = _quantised(K, N, copies)

        def call(weights=weights, scales=scales, biases=biases, copies=copies):
            mx.eval([mx.quantized_matmul(x, weights[i % copies], scales[i % copies],
                                         biases[i % copies], transpose=True,
                                         group_size=GROUP_SIZE, bits=BITS)
                     for i in range(calls)])
        timings[name] = _time(call)[:2]
    if not timings["dram"][0]:
        return None
    return Measurement(
        value=timings["resident"][0] / timings["dram"][0], unit="ratio", evidence_id=evidence,
        spread=None, samples=REPEATS,
        context={"k": K, "n": N, "calls_per_eval": calls,
                 "resident_working_set_bytes": per * resident_copies,
                 "dram_working_set_bytes": per * calls,
                 "last_level_cache_bytes_assumed": cache_bytes,
                 "resident_ns": timings["resident"][0], "dram_ns": timings["dram"][0]})


def probe_alignment(evidence: str) -> Measurement | None:
    """The model's own K against the nearest power of two, compared per weight byte."""
    results = {}
    for k in (K, ALIGNED_K):
        per = weight_bytes(k, N)
        copies = max(2, min(16, DISTINCT_BYTES // max(per, 1)))
        weights, scales, biases = _quantised(k, N, copies)
        x = mx.random.normal((1, k)).astype(mx.bfloat16)
        mx.eval(x)
        median, spread, _ = _time(_matmul_chain(weights, scales, biases, x, k))
        results[k] = (median / copies / per, spread, per)
    if not results[ALIGNED_K][0]:
        return None
    return Measurement(
        value=results[K][0] / results[ALIGNED_K][0], unit="ratio", evidence_id=evidence,
        spread=None, samples=REPEATS,
        context={"unaligned_k": K, "aligned_k": ALIGNED_K, "n": N,
                 "ns_per_weight_byte_unaligned": results[K][0],
                 "ns_per_weight_byte_aligned": results[ALIGNED_K][0],
                 "note": "compared per weight byte because the two K move different amounts"})


def probe_widths(evidence: str) -> tuple[Measurement, ...]:
    weights, scales, biases = _quantised(K, N, 4)
    out = []
    for width in WIDTHS:
        x = mx.random.normal((width, K)).astype(mx.bfloat16)
        mx.eval(x)
        median, spread, _ = _time(_matmul_chain(weights, scales, biases, x, K))
        per_call = median / 4
        out.append(Measurement(
            value=per_call / width, unit="ns_per_row", evidence_id=evidence,
            spread=(spread / 4 / width) if spread else None, samples=REPEATS,
            context={"width": width, "k": K, "n": N, "ns_per_call": per_call}))
    return tuple(out)


def probe_eval(evidence: str) -> tuple[Measurement | None, Measurement | None]:
    weights, scales, biases = _quantised(K, N, max(CHAIN_LENGTHS))
    x = mx.random.normal((1, K)).astype(mx.bfloat16)
    mx.eval(x)
    totals, spreads = {}, {}
    for count in CHAIN_LENGTHS:
        def call(count=count):
            mx.eval([mx.quantized_matmul(x, weights[i], scales[i], biases[i], transpose=True,
                                         group_size=GROUP_SIZE, bits=BITS)
                     for i in range(count)])
        totals[count], spreads[count], _ = _time(call)
    low, high = min(CHAIN_LENGTHS), max(CHAIN_LENGTHS)
    marginal = (totals[high] - totals[low]) / (high - low)
    fixed = totals[low] - marginal * low
    # The marginal is not constant across the chain: the per-kernel increment measured here
    # is reported alongside so a reader can see how far from a line the points are, rather
    # than trusting a two-point fit that cannot show it.
    increments = {f"{a}_to_{b}": (totals[b] - totals[a]) / (b - a)
                  for a, b in zip(CHAIN_LENGTHS, CHAIN_LENGTHS[1:])}
    context = {"chain_lengths": list(CHAIN_LENGTHS),
               "median_ns_total": {str(c): totals[c] for c in CHAIN_LENGTHS},
               "half_range_ns": {str(c): spreads[c] for c in CHAIN_LENGTHS},
               "marginal_ns_between_points": increments,
               "fit": ("two endpoints on a straight line. The per-point increments above "
                       "show how far from a line the machine actually was; where they "
                       "disagree, the fit is an order and nothing finer")}
    return (Measurement(fixed, "ns", evidence, spread=spreads[low], samples=REPEATS,
                        context=context),
            Measurement(marginal, "ns", evidence, spread=spreads[high], samples=REPEATS,
                        context=context))


def probe_geometry(evidence: str) -> tuple[Measurement, ...]:
    """A candidate geometry against the library, byte checked first. Activates nothing."""
    from ironmule import kernel_registry, qmv_k3840 as qmv

    weights, scales, biases = _quantised(K, N, 4)
    x = mx.random.normal((1, K)).astype(mx.bfloat16)
    mx.eval(x)
    reference = [mx.quantized_matmul(x, w, s, b, transpose=True,
                                     group_size=GROUP_SIZE, bits=BITS)
                 for w, s, b in zip(weights, scales, biases)]
    mx.eval(reference)
    wanted = [bytes(memoryview(mx.array(r))) for r in reference]
    shape = qmv.shape_array(K, N)
    library = _matmul_chain(weights, scales, biases, x, K)
    library_ns, library_spread, _ = _time(library)
    # The noise floor: the same work under another name. A candidate that does not clear it
    # has not been measured, whatever its median says.
    aa_ns, aa_spread, _ = _time(library)
    aa_ratio = aa_ns / library_ns if library_ns else 0.0
    aa_relative_spread = (aa_spread / library_ns) if (aa_spread and library_ns) else None
    # Whether this machine was quiet enough for the probe to have measured anything. The
    # thresholds were fixed before this ran; the candidate is timed either way and reported
    # either way, and only the vector field is withheld when the floor swamps it.
    floor_ok = (aa_relative_spread is not None
                and aa_relative_spread <= GEOMETRY_AA_MAX_SPREAD
                and abs(aa_ratio - 1.0) <= GEOMETRY_AA_MAX_OFFSET)
    out = [Measurement(
        value=aa_ratio, unit="ratio", evidence_id=evidence, spread=aa_relative_spread,
        samples=REPEATS,
        context={"geometry": "library_aa", "usable": False, "is_noise_floor": True,
                 "floor_ok": floor_ok,
                 "max_spread_allowed": GEOMETRY_AA_MAX_SPREAD,
                 "max_offset_allowed": GEOMETRY_AA_MAX_OFFSET,
                 "reason": ("the A/A control: the library against itself. When its spread "
                            "swamps the effect, no geometry number from this run means "
                            "anything and the vector field stays missing"),
                 "library_ns": library_ns, "candidate_ns": aa_ns})]
    for simdgroups, results in GEOMETRIES:
        rows = simdgroups * results
        if N % rows:
            continue
        try:
            source = qmv.BODY.format(
                num_simdgroups=simdgroups, results_per_simdgroup=results,
                pack_factor=qmv.PACK_FACTOR, bytes_per_pack=qmv.BYTES_PER_PACK,
                values_per_thread=qmv.VALUES_PER_THREAD, block_size=qmv.BLOCK_SIZE,
                group_size=qmv.GROUP_SIZE, scale_step_per_thread=qmv.SCALE_STEP_PER_THREAD,
                in_vec_size_decl=f"constexpr int in_vec_size = {K};",
                main_loop=qmv.FIXED_LOOP.replace("15", str(K // qmv.BLOCK_SIZE)), tail="")
            kernel = kernel_registry.build(
                f"b71_probe_sg{simdgroups}_r{results}",
                input_names=["w", "scales", "biases", "x", "shape"], output_names=["out"],
                source=source, header=qmv.HEADER, ensure_row_contiguous=True,
                template={"fixed_k": K, "num_simdgroups": simdgroups,
                          "results_per_simdgroup": results})
            groups = N // rows

            def call(kernel=kernel, simdgroups=simdgroups, groups=groups):
                mx.eval([kernel(inputs=[w, s, b, x, shape], output_shapes=[(1, N)],
                                output_dtypes=[x.dtype],
                                grid=(qmv.SIMD_SIZE, simdgroups * groups, 1),
                                threadgroup=(qmv.SIMD_SIZE, simdgroups, 1))[0]
                         for w, s, b in zip(weights, scales, biases)])
            call()
            got = [kernel(inputs=[w, s, b, x, shape], output_shapes=[(1, N)],
                          output_dtypes=[x.dtype],
                          grid=(qmv.SIMD_SIZE, simdgroups * groups, 1),
                          threadgroup=(qmv.SIMD_SIZE, simdgroups, 1))[0]
                   for w, s, b in zip(weights, scales, biases)]
            mx.eval(got)
            identical = all(bytes(memoryview(mx.array(o))) == want
                            for o, want in zip(got, wanted))
        except (ValueError, RuntimeError, TypeError, KeyError, AttributeError) as exc:
            out.append(Measurement(
                float("inf") if False else 0.0, "ratio", evidence, samples=0,
                context={"geometry": f"{simdgroups}_{results}", "usable": False,
                         "reason": f"{type(exc).__name__}: {exc}"}))
            continue
        if not identical:
            out.append(Measurement(
                0.0, "ratio", evidence, samples=0,
                context={"geometry": f"{simdgroups}_{results}", "usable": False,
                         "reason": "not byte identical to the library on the same buffers"}))
            continue
        median, spread, _ = _time(call)
        out.append(Measurement(
            value=median / library_ns, unit="ratio", evidence_id=evidence,
            spread=(spread / library_ns) if spread and library_ns else None, samples=REPEATS,
            context={"geometry": f"{simdgroups}_{results}", "usable": bool(floor_ok),
                     "withheld_because_the_noise_floor_swamps_it": not floor_ok,
                     "num_simdgroups": simdgroups, "results_per_simdgroup": results,
                     "k": K, "n": N, "m": 1, "library_ns": library_ns,
                     "candidate_ns": median, "byte_identical": True,
                     "this_is_a_probe": "not an activation, and not a stack result"}))
    return tuple(out)


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
    parser.add_argument("--cache-bytes", type=int, default=48 * 1024 * 1024,
                        help="assumed last-level cache size; only labels the cache probe")
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once

    if args.preregister:
        write_once(args.preregister, json.dumps(
            {"preregistration": PREREGISTRATION, "source_binding": source_binding(),
             "written_at": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True))
        print(f"preregistration written to {args.preregister}")
        return 0

    from ironmule.hw import (fingerprint, installed_memory_bytes, memory_pressure_level,
                             static_facts, swap_used_bytes, vm_counters)
    from ironmule.tune import gpu_busy
    import mlx_lm

    busy = gpu_busy()
    if busy:
        raise SystemExit(f"another model process is running, refusing to measure ({busy})")

    total_pages = (installed_memory_bytes() or 0) // 16384

    def sample(label):
        counters = vm_counters() or {}
        free_pages = (counters.get("free_count", 0) + counters.get("speculative_count", 0)
                      + counters.get("inactive_count", 0))
        return {"label": label, "swap_used_bytes": swap_used_bytes(),
                "memory_pressure_level": memory_pressure_level(),
                "load_average": list(os.getloadavg()),
                "memory_free_percent": (100.0 * free_pages / total_pages) if total_pages else None}

    evidence = args.out.stem
    samples = [sample("start")]
    started = time.perf_counter()
    facts = static_facts()

    print("probe bandwidth", flush=True)
    bandwidth, curve = probe_bandwidth(evidence)
    samples.append(sample("after_bandwidth"))
    print("probe cache", flush=True)
    cache = probe_cache(evidence, args.cache_bytes)
    samples.append(sample("after_cache"))
    print("probe alignment", flush=True)
    alignment = probe_alignment(evidence)
    samples.append(sample("after_alignment"))
    print("probe widths", flush=True)
    widths = probe_widths(evidence)
    samples.append(sample("after_widths"))
    print("probe eval", flush=True)
    fixed, marginal = probe_eval(evidence)
    samples.append(sample("after_eval"))
    print("probe geometry", flush=True)
    geometry = probe_geometry(evidence)
    samples.append(sample("end"))
    elapsed = time.perf_counter() - started

    swaps = [s["swap_used_bytes"] for s in samples if s["swap_used_bytes"] is not None]
    frees = [s["memory_free_percent"] for s in samples if s["memory_free_percent"] is not None]
    levels = [s["memory_pressure_level"] for s in samples]
    reasons = []
    if not swaps or max(swaps) > swaps[0]:
        reasons.append("swap in use rose above its value at the start")
    if not frees or min(frees) < 10.0:
        reasons.append("free memory below 10 per cent")
    if not all(level == 1 and level is not None for level in levels):
        reasons.append("macOS memory pressure not normal at every sample, or unreadable")
    if elapsed > BUDGET_SECONDS:
        reasons.append(f"the probe set took {elapsed:.0f}s, over its {BUDGET_SECONDS:.0f}s budget")

    measured = MeasuredResponses(
        dram_bandwidth=bandwidth, bandwidth_by_working_set=curve,
        cache_residency_ratio=cache, k_alignment_ratio=alignment,
        width_response=widths, eval_fixed_cost=fixed, eval_marginal_cost=marginal,
        geometry_response=tuple(m for m in geometry if m.context.get("usable")))
    vector = HardwareCharacterizationVector(
        schema=SCHEMA,
        static=StaticFacts(
            hardware_fingerprint=fingerprint(), chip=facts.get("chip") or "",
            gpu_architecture=mx.device_info().get("architecture", ""),
            gpu_cores=facts.get("gpu_cores"), cpu_logical=facts.get("cpu_logical"),
            unified_memory_bytes=installed_memory_bytes(),
            os_release=facts.get("os_release") or "", mlx=mx.__version__,
            mlx_lm=mlx_lm.__version__),
        measured=measured,
        conditions=Conditions(
            measured_at=datetime.now(timezone.utc).isoformat(),
            probe_wall_seconds=elapsed,
            load_average_at_start=tuple(samples[0]["load_average"]),
            load_average_at_end=tuple(samples[-1]["load_average"]),
            memory_pressure_normal_throughout=all(level == 1 for level in levels),
            swap_grew=bool(swaps) and max(swaps) > swaps[0],
            min_memory_free_percent=min(frees) if frees else None,
            resource_gate_passed=not reasons, resource_gate_reasons=tuple(reasons),
            code_binding_sha256=source_binding()["combined_sha256"],
            probe_set_id=PROBE_SET_ID,
            correctness_checked=any(m.context.get("byte_identical") for m in geometry),
            notes=("no model was loaded and no profile was read. Every probe is a mechanism "
                   "E4, B66 or B68 already established")),
        relations={})
    vector = HardwareCharacterizationVector(
        vector.schema, vector.static, vector.measured, vector.conditions,
        relations_for(measured))

    record = {
        "experiment": "B71_quick_characterization",
        "preregistration": PREREGISTRATION,
        "source_binding": source_binding(),
        "environment": {"platform": platform.platform(), "mlx": mx.__version__,
                        "mlx_lm": mlx_lm.__version__},
        "vector": vector.as_dict(),
        "geometry_probes_including_unusable": [m.as_dict() for m in geometry],
        "resource_samples": samples,
        "probe_wall_seconds": elapsed,
        "budget_seconds": BUDGET_SECONDS,
    "why_v3": ("v1 ran in 1.1 seconds of a 420 second budget with a geometry spread of 0.58 "
               "against a ratio of 0.95. v2 raised repeats from 7 to 25, warmups from 3 to "
               "5, and added an A/A arm, which then measured a noise floor of 0.9885 with a "
               "relative spread of 0.606 and made the candidate at 1.0024 unreadable. v3 "
               "keeps that design and draws the consequence: a geometry number is emitted "
               "only when its own A/A arm says the machine was quiet enough to have "
               "measured anything, and otherwise the field is missing. v3 also fixes a unit "
               "error in v2, which recorded a nanosecond half range as the spread of a "
               "bytes-per-nanosecond value. Every earlier run is kept as it was measured"),
        "verdict": ("VECTOR_INVALID" if reasons
                    else "VECTOR_IMPLEMENTED_AND_SELF_CONSISTENT"),
        "why_not_more": ("H1 needs variation across machines and this is one machine. No "
                         "generalisation is claimed"),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"verdict": record["verdict"], "seconds": round(elapsed, 1),
                      "missing": list(vector.missing()),
                      "relations": {k: round(v["value"], 4) for k, v in vector.relations.items()},
                      "gate": vector.conditions.resource_gate_reasons or "passed"},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
