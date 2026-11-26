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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("primary", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--packed-memory",
        action="store_true",
        help="read the HTTP archive and estimate cache headroom from the compact process",
    )
    a = p.parse_args()
    save(a.output, packing_budget(a.primary) if a.packed_memory else plan(a.primary))
