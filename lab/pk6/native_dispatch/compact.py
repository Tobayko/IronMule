"""Size a candidate unified KV pool from measured request demand, preserving the individual context limit."""

from pathlib import Path
import argparse
import json
import math

from lab.pk6.mixed_progress.grade import lines
from lab.pk6.replay_http import checksum
from lab.pk6.warm_repeat import save


def plan(primary):
    meta = json.loads((primary / "target.json").read_bytes())
    source = primary / "waves-R1.jsonl"
    waves = lines(source)
    if len(waves) != 10 or not (primary / "end-R1.json").is_file():
        raise ValueError("incomplete discovery process")
    context = 2048
    # No prefix reuse credit. Reserve the full answer and additional draft space for every request.
    demand = [sum(r["prompt_tokens"] + 32 + 16 for r in w["requests"]) for w in waves]
    cells = 1 << math.ceil(math.log2(max(2 * context, 1.25 * max(demand) + context)))
    return {
        "schema": "ironmule.unified-kv-study.v1",
        "hardware": meta["hardware"],
        "gguf": meta["gguf_sha256"],
        "source": checksum(source),
        "training_seed": waves[0]["seed"],
        "context": context,
        "measured_wave_demand": demand,
        "cells": cells,
        "reference_cells": 32 * context,
        "qualification": "Queue requests above physical capacity, preserving their original work and individual context. The independent seed tests the forecast.",
    }


def packing_budget(http):
    from lab.pk6.native_dispatch.score import run

    result = run(http)
    first, compact = (result["arms"][k] for k in ("R1", "H1"))
    peak, base = (r["resources"]["peak_gpu_mib"] for r in (compact, first))
    if (
        compact["excluded_from_promotion"]
        or set(compact["quality"].values()) != {32}
        or not compact["proof"]["verified_compact_pool"]
        or not 0 < peak < base
    ):
        raise ValueError("missing compact-pool headroom or quality witness")
    device = result["binding"][0]
    reserve = 6144
    gib = int((device["memory_mib"] - peak - reserve) // 1024)
    if gib < 1:
        raise ValueError("packed cache does not fit the measured reserve")
    waves = next(http.rglob("waves-H1.jsonl"))
    receipt = json.loads((waves.parent / "target.json").read_bytes())
    return {
        "hardware": device,
        "stage": "simulated next profile",
        "measured_serving_peak_mib": peak,
        "measured_reference_peak_mib": base,
        "reserve_mib": reserve,
        "cache_gib": gib,
        "pool_cells": receipt["pool_plan"]["cells"],
        "source": checksum(waves),
    }


def explain_pressure(root):
    """Compare cache reuse and time inside actual mixed compute calls at 32 clients."""
    from lab.pk6.mixed_progress import grade
    from lab.pk6.native_dispatch.score import waves, read

    pairs = {}
    for seed in (1, 2):
        candidate, reference = f"H{seed}", f"R{seed}"
        arms = {}
        for arm in (reference, candidate):
            ends = list(root.rglob(f"end-{arm}.json"))
            if len(ends) > 1:
                raise ValueError(f"ambiguous process for {arm}")
            if not ends:
                break
            path = ends[0].parent
            target = read(path / "target.json")
            rows = [w for w in waves(path, arm) if w["width"] == 32 and w["repeat"] < 3]
            bound = grade.bindings(path, arm, rows)
            cells = []
            for w, cell in zip(rows, bound["cells"], strict=True):
                phases = cell["phases"]
                cells.append(
                    {
                        "repeat": w["repeat"],
                        "http_wall_s": w["elapsed_s"],
                        "input_tokens": cell["prompt_tokens"],
                        "reused_tokens": cell["reused_tokens"],
                        "mixed_calls": phases.get("mixed", {}).get("calls", 0),
                        "mixed_llama_decode_s": phases.get("mixed", {}).get(
                            "llama_decode_s", 0.0
                        ),
                    }
                )
            arms[arm] = (
                target,
                rows,
                {"waves": cells, "trace_sha256": bound["trace_sha256"]},
            )
        if len(arms) < 2:
            continue
        rt, rw, reference_data = arms[reference]
        ht, hw, candidate_data = arms[candidate]
        if any(rt[k] != ht[k] for k in ("hardware", "source", "gguf_sha256")):
            raise ValueError("pressure profiles used different targets")
        pair = grade.paired(rw, hw)
        if not pair["same_fixed_work"]:
            raise ValueError("different work in long-document waves")
        pairs[str(rw[0]["seed"])] = {
            reference: reference_data,
            candidate: candidate_data,
        }
    return {
        "pairs": pairs,
        "scope": "Three successive waves at 32 clients per process. A mixed call evaluated prompt and decode rows together. The effect of allocation, admission and attention range cannot be isolated from these two profiles.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("primary", type=Path)
    p.add_argument("output", type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--explain-pressure",
        action="store_true",
        help="replay the long-document archive",
    )
    mode.add_argument(
        "--packed-memory",
        action="store_true",
        help="read the HTTP archive and estimate cache headroom from the compact process",
    )
    a = p.parse_args()
    evaluate = (
        explain_pressure
        if a.explain_pressure
        else packing_budget
        if a.packed_memory
        else plan
    )
    save(a.output, evaluate(a.primary))
