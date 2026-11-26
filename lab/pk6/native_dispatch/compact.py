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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("primary", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    save(a.output, plan(a.primary))
